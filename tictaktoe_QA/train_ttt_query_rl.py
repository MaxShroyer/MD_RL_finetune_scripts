#!/usr/bin/env python3
"""Query-skill RL finetuning for TicTacToe QA."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import random
import string
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from dotenv import load_dotenv
from PIL import Image

try:
    from tqdm.auto import tqdm  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    class _SimpleTqdm:  # pragma: no cover
        def __init__(self, iterable, *args: Any, **kwargs: Any) -> None:
            self._iterable = iterable

        def __iter__(self):
            return iter(self._iterable)

        def set_postfix(self, *args: Any, **kwargs: Any) -> None:
            return

    def tqdm(iterable=None, *args, **kwargs):  # type: ignore
        if iterable is None:
            iterable = []
        return _SimpleTqdm(iterable, *args, **kwargs)

try:
    import wandb  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    class _WandbRun:
        def __init__(self) -> None:
            self.summary: dict[str, Any] = {}

        def finish(self) -> None:
            return

    class _WandbShim:
        @staticmethod
        def init(*args: Any, **kwargs: Any) -> _WandbRun:
            print("wandb not installed; continuing without remote logging.")
            return _WandbRun()

        @staticmethod
        def log(*args: Any, **kwargs: Any) -> None:
            return

    wandb = _WandbShim()

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tuna_sdk import QueryOutput, QueryRequest, QuerySettings, TunaClient  # noqa: E402
from tuna_sdk.errors import TunaAPIError, TunaNetworkError  # noqa: E402

DEFAULT_BASE_URL = "https://api.moondream.ai/v1"
DEFAULT_FINAL_EVAL_SPLITS = [
    "val",
    "test",
    "benchmark_top50_canonical",
    "benchmark_top50_paraphrase",
]
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "query_rl_default.json"
DEFAULT_REASONING = False

SUPPORTED_TASKS = {
    "best_move",
    "has_winning_move",
    "is_terminal",
    "legal_moves_count",
    "legal_moves_list",
    "turn_player",
    "winner",
}
DEFAULT_TASK_SAMPLING_WEIGHTS: dict[str, float] = {
    "best_move": 1.0,
    "legal_moves_count": 1.0,
    "legal_moves_list": 1.0,
}
DEFAULT_MAX_TOKENS_BY_TASK: dict[str, int] = {
    "legal_moves_list": 800,
}

REQUIRED_ROW_FIELDS = (
    "row_id",
    "split",
    "task_type",
    "question",
    "final_answer_json",
    "best_move_canonical_json",
    "best_move_optimal_set_json",
    "image_path",
)


@dataclass(frozen=True)
class QAExample:
    row_id: str
    split: str
    task_type: str
    question: str
    image_path: Path
    expected_answer: dict[str, Any]
    best_move_canonical: Optional[int]
    best_move_optimal_set: frozenset[int]


@dataclass(frozen=True)
class ScoreOutcome:
    reward: float
    parse_success: bool
    task_correct: bool
    json_object_parsed: bool = False
    best_move_set_correct: bool = False
    best_move_canonical_correct: bool = False
    exact_non_best_correct: bool = False


def _repo_relative(*parts: str) -> Path:
    return Path(__file__).resolve().parent.joinpath(*parts)


def _random_suffix(length: int = 6) -> str:
    chars = string.ascii_lowercase + string.digits
    return "".join(random.choices(chars, k=length))


def _resolve_config_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path

    from_cwd = (Path.cwd() / path).resolve()
    if from_cwd.exists():
        return from_cwd

    from_script = (_repo_relative(path.as_posix())).resolve()
    if from_script.exists():
        return from_script

    return from_cwd


def _load_json_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        if config_path == DEFAULT_CONFIG_PATH:
            return {}
        raise FileNotFoundError(f"Config file not found: {config_path}")

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a JSON object: {config_path}")
    return payload


def _cfg_str(config: dict[str, Any], key: str, fallback: str) -> str:
    value = config.get(key, fallback)
    return str(value) if value is not None else fallback


def _cfg_int(config: dict[str, Any], key: str, fallback: int) -> int:
    value = config.get(key, fallback)
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _cfg_float(config: dict[str, Any], key: str, fallback: float) -> float:
    value = config.get(key, fallback)
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _cfg_list_str(config: dict[str, Any], key: str, fallback: list[str]) -> list[str]:
    value = config.get(key)
    if not isinstance(value, list):
        return list(fallback)
    out: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            out.append(text)
    return out or list(fallback)


def _cfg_dict(config: dict[str, Any], key: str, fallback: dict[str, Any]) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        return dict(fallback)
    return dict(value)


def _cfg_bool(config: dict[str, Any], key: str, fallback: bool) -> bool:
    value = config.get(key, fallback)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y", "on"}:
            return True
        if lowered in {"false", "0", "no", "n", "off"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return fallback


def _parse_json_object_arg(raw_value: str, arg_name: str) -> dict[str, Any]:
    text = str(raw_value or "").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{arg_name} must be valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{arg_name} must decode to a JSON object")
    return payload


def _normalize_task_sampling_weights(raw_map: dict[str, Any], *, source: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for raw_task, raw_weight in raw_map.items():
        task = str(raw_task).strip()
        if task not in SUPPORTED_TASKS:
            raise ValueError(f"{source}: unknown task_type '{task}'")
        try:
            weight = float(raw_weight)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{source}: weight for task_type '{task}' must be numeric") from exc
        if weight <= 0.0:
            raise ValueError(f"{source}: weight for task_type '{task}' must be > 0")
        out[task] = weight
    return out


def _normalize_max_tokens_by_task(raw_map: dict[str, Any], *, source: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for raw_task, raw_max_tokens in raw_map.items():
        task = str(raw_task).strip()
        if task not in SUPPORTED_TASKS:
            raise ValueError(f"{source}: unknown task_type '{task}'")
        max_tokens = _parse_int(raw_max_tokens)
        if max_tokens is None or max_tokens <= 0:
            raise ValueError(f"{source}: max_tokens for task_type '{task}' must be > 0")
        out[task] = int(max_tokens)
    return out


def _resolve_task_sampling_weights(
    *,
    config_map: dict[str, Any],
    cli_override_json: str,
) -> dict[str, float]:
    resolved = {task: 1.0 for task in SUPPORTED_TASKS}
    resolved.update(_normalize_task_sampling_weights(config_map, source="config task_sampling_weights"))

    if str(cli_override_json or "").strip():
        cli_map = _parse_json_object_arg(cli_override_json, "--task-sampling-weights-json")
        resolved.update(
            _normalize_task_sampling_weights(
                cli_map,
                source="--task-sampling-weights-json",
            )
        )
    return resolved


def _resolve_max_tokens_by_task(
    *,
    config_map: dict[str, Any],
    cli_override_json: str,
) -> dict[str, int]:
    resolved = _normalize_max_tokens_by_task(config_map, source="config max_tokens_by_task")

    if str(cli_override_json or "").strip():
        cli_map = _parse_json_object_arg(cli_override_json, "--max-tokens-by-task-json")
        resolved.update(
            _normalize_max_tokens_by_task(
                cli_map,
                source="--max-tokens-by-task-json",
            )
        )
    return resolved


def _build_parser(config: dict[str, Any], config_path: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RL query finetuning for TicTacToe QA")
    parser.add_argument("--config", default=str(config_path))
    parser.add_argument("--env-file", default=_cfg_str(config, "env_file", str(_repo_relative(".env"))))
    parser.add_argument("--api-key", default=_cfg_str(config, "api_key", ""))
    parser.add_argument("--base-url", default=_cfg_str(config, "base_url", ""))

    parser.add_argument(
        "--dataset-dir",
        default=_cfg_str(config, "dataset_dir", str(_repo_relative("synth_dataset/outputs/v1"))),
    )
    parser.add_argument("--train-split", default=_cfg_str(config, "train_split", "train"))
    parser.add_argument("--val-split", default=_cfg_str(config, "val_split", "val"))
    parser.add_argument(
        "--final-eval-splits",
        nargs="+",
        default=_cfg_list_str(config, "final_eval_splits", DEFAULT_FINAL_EVAL_SPLITS),
        help="Split names to evaluate after training.",
    )

    parser.add_argument("--finetune-id", default=_cfg_str(config, "finetune_id", ""))
    parser.add_argument("--finetune-name", default=_cfg_str(config, "finetune_name", ""))
    parser.add_argument("--rank", type=int, default=_cfg_int(config, "rank", 16))

    parser.add_argument("--seed", type=int, default=_cfg_int(config, "seed", 42))
    parser.add_argument("--num-steps", type=int, default=_cfg_int(config, "num_steps", 100))
    parser.add_argument("--resume-step", type=int, default=_cfg_int(config, "resume_step", 0))
    parser.add_argument("--batch-size", type=int, default=_cfg_int(config, "batch_size", 16))
    parser.add_argument("--group-size", type=int, default=_cfg_int(config, "group_size", 4))
    parser.add_argument("--lr", type=float, default=_cfg_float(config, "lr", 2e-3))
    parser.add_argument("--max-workers", type=int, default=_cfg_int(config, "max_workers", 4))
    parser.add_argument("--rollout-retries", type=int, default=_cfg_int(config, "rollout_retries", 2))
    parser.add_argument(
        "--rollout-retry-backoff-s",
        type=float,
        default=_cfg_float(config, "rollout_retry_backoff_s", 1.0),
    )

    parser.add_argument("--temperature", type=float, default=_cfg_float(config, "temperature", 1.0))
    parser.add_argument("--top-p", type=float, default=_cfg_float(config, "top_p", 0.9))
    parser.add_argument("--max-tokens", type=int, default=_cfg_int(config, "max_tokens", 256))
    parser.add_argument(
        "--task-sampling-weights-json",
        default="",
        help="JSON object override for task sampling weights: {\"task_type\": weight}.",
    )
    parser.add_argument(
        "--max-tokens-by-task-json",
        default="",
        help="JSON object override for per-task max token caps: {\"task_type\": int}.",
    )
    reasoning_group = parser.add_mutually_exclusive_group()
    reasoning_group.add_argument(
        "--reasoning",
        dest="reasoning",
        action="store_true",
        help="Enable reasoning mode in query requests.",
    )
    reasoning_group.add_argument(
        "--no-reasoning",
        dest="reasoning",
        action="store_false",
        help="Disable reasoning mode in query requests.",
    )
    parser.set_defaults(reasoning=_cfg_bool(config, "reasoning", DEFAULT_REASONING))

    parser.add_argument("--eval-every", type=int, default=_cfg_int(config, "eval_every", 20))
    parser.add_argument("--save-every", type=int, default=_cfg_int(config, "save_every", 20))
    parser.add_argument("--eval-batch-size", type=int, default=_cfg_int(config, "eval_batch_size", 32))
    parser.add_argument(
        "--eval-max-samples",
        type=int,
        default=_cfg_int(config, "eval_max_samples", 1000),
        help="Max samples per eval split. <=0 means full split.",
    )

    parser.add_argument(
        "--best-metric",
        choices=[
            "eval_reward_mean",
            "eval_best_move_set_accuracy",
            "eval_best_move_canonical_accuracy",
            "eval_exact_accuracy_non_best_move",
            "eval_json_parse_rate",
        ],
        default=_cfg_str(config, "best_metric", "eval_reward_mean"),
    )
    parser.add_argument(
        "--best-move-optimal-reward",
        type=float,
        default=_cfg_float(config, "best_move_optimal_reward", 0.7),
    )

    parser.add_argument("--wandb-project", default=_cfg_str(config, "wandb_project", "moondream-ttt-query-rl"))
    parser.add_argument("--wandb-run-name", default=_cfg_str(config, "wandb_run_name", ""))
    parser.add_argument(
        "--no-progress",
        action="store_true",
        default=_cfg_bool(config, "no_progress", False),
        help="Disable tqdm progress bars.",
    )

    return parser


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    pre_args, _ = pre_parser.parse_known_args(argv)

    config_path = _resolve_config_path(pre_args.config)
    config = _load_json_config(config_path)
    parser = _build_parser(config, config_path)
    args = parser.parse_args(argv)

    args.config = str(_resolve_config_path(args.config))
    args.task_sampling_weights = _resolve_task_sampling_weights(
        config_map=_cfg_dict(
            config,
            "task_sampling_weights",
            DEFAULT_TASK_SAMPLING_WEIGHTS,
        ),
        cli_override_json=args.task_sampling_weights_json,
    )
    args.max_tokens_by_task = _resolve_max_tokens_by_task(
        config_map=_cfg_dict(
            config,
            "max_tokens_by_task",
            DEFAULT_MAX_TOKENS_BY_TASK,
        ),
        cli_override_json=args.max_tokens_by_task_json,
    )
    return args


def _resolve_env_file(env_file: str) -> str:
    path = Path(env_file).expanduser()
    if path.is_absolute():
        return str(path)

    from_cwd = (Path.cwd() / path).resolve()
    if from_cwd.exists():
        return str(from_cwd)

    from_repo = (REPO_ROOT / path).resolve()
    if from_repo.exists():
        return str(from_repo)

    from_script = (_repo_relative(path.as_posix())).resolve()
    if from_script.exists():
        return str(from_script)

    return str(from_cwd)


def _validate_args(args: argparse.Namespace) -> None:
    if args.num_steps < 0:
        raise ValueError("--num-steps must be >= 0")
    if args.resume_step < 0:
        raise ValueError("--resume-step must be >= 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.group_size <= 0:
        raise ValueError("--group-size must be > 0")
    if args.max_workers <= 0:
        raise ValueError("--max-workers must be > 0")
    if args.lr <= 0.0:
        raise ValueError("--lr must be > 0")
    if not (0.0 <= args.temperature <= 2.0):
        raise ValueError("--temperature must be in [0,2]")
    if not (0.0 < args.top_p <= 1.0):
        raise ValueError("--top-p must be in (0,1]")
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be > 0")
    if args.eval_every < 0:
        raise ValueError("--eval-every must be >= 0")
    if args.save_every < 0:
        raise ValueError("--save-every must be >= 0")
    if args.eval_batch_size <= 0:
        raise ValueError("--eval-batch-size must be > 0")
    if args.rollout_retries < 0:
        raise ValueError("--rollout-retries must be >= 0")
    if args.rollout_retry_backoff_s <= 0.0:
        raise ValueError("--rollout-retry-backoff-s must be > 0")
    if not (0.0 <= args.best_move_optimal_reward <= 1.0):
        raise ValueError("--best-move-optimal-reward must be in [0,1]")

    if args.finetune_id and args.finetune_name:
        raise ValueError("Provide either --finetune-id or --finetune-name, not both")



def _progress_enabled(no_progress: bool) -> bool:
    if no_progress:
        return False
    return sys.stderr.isatty()


def _dedupe_splits(splits: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for split in splits:
        item = str(split).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _to_data_url(image: Image.Image, *, quality: int = 92) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=max(1, min(100, int(quality))))
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _row_col_to_move(row: int, col: int) -> Optional[int]:
    if row < 1 or row > 3 or col < 1 or col > 3:
        return None
    return ((row - 1) * 3) + col


def _parse_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value == 0:
            return False
        if value == 1:
            return True
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "t", "yes", "y", "1"}:
            return True
        if lowered in {"false", "f", "no", "n", "0"}:
            return False
    return None


def _coerce_player(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    lowered = value.strip().lower()
    if lowered == "x":
        return "X"
    if lowered == "o":
        return "O"
    return None


def _coerce_winner(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    lowered = value.strip().lower().replace(" ", "_")
    mapping = {
        "x": "X",
        "o": "O",
        "draw": "draw",
        "in_progress": "in_progress",
        "inprogress": "in_progress",
    }
    return mapping.get(lowered)


def _normalize_legal_moves(payload: Any) -> Optional[tuple[tuple[int, int], ...]]:
    if not isinstance(payload, list):
        return None
    moves: list[tuple[int, int]] = []
    for item in payload:
        if not isinstance(item, dict):
            return None
        row = _parse_int(item.get("row"))
        col = _parse_int(item.get("col"))
        if row is None or col is None:
            return None
        if row < 1 or row > 3 or col < 1 or col > 3:
            return None
        moves.append((row, col))
    return tuple(moves)


def _move_from_payload_obj(payload: Any) -> Optional[int]:
    if not isinstance(payload, dict):
        return None

    if "move" in payload:
        move = _parse_int(payload.get("move"))
        if move is None or move < 1 or move > 9:
            return None
        return move

    row = _parse_int(payload.get("row"))
    col = _parse_int(payload.get("col"))
    if row is None or col is None:
        return None
    return _row_col_to_move(row, col)


def _best_move_from_json(payload_json: str) -> Optional[int]:
    try:
        payload = json.loads(payload_json)
    except json.JSONDecodeError:
        return None
    return _move_from_payload_obj(payload)


def _as_move_set(payload_json: str) -> set[int]:
    try:
        payload = json.loads(payload_json)
    except json.JSONDecodeError:
        return set()
    if not isinstance(payload, list):
        return set()

    out: set[int] = set()
    for item in payload:
        move = _move_from_payload_obj(item)
        if move is not None:
            out.add(move)
    return out


def _normalize_non_best_answer(task_type: str, payload: Any) -> Optional[Any]:
    if not isinstance(payload, dict):
        return None

    if task_type == "winner":
        winner = _coerce_winner(payload.get("winner"))
        if winner is None:
            return None
        return {"winner": winner}

    if task_type == "is_terminal":
        is_terminal = _coerce_bool(payload.get("is_terminal"))
        if is_terminal is None:
            return None
        return {"is_terminal": is_terminal}

    if task_type == "has_winning_move":
        has_winning_move = _coerce_bool(payload.get("has_winning_move"))
        if has_winning_move is None:
            return None
        return {"has_winning_move": has_winning_move}

    if task_type == "turn_player":
        player = _coerce_player(payload.get("player"))
        if player is None:
            return None
        return {"player": player}

    if task_type == "legal_moves_count":
        count = _parse_int(payload.get("legal_move_count"))
        if count is None or count < 0 or count > 9:
            return None
        return {"legal_move_count": count}

    if task_type == "legal_moves_list":
        legal_moves = _normalize_legal_moves(payload.get("legal_moves"))
        if legal_moves is None:
            return None
        return {"legal_moves": legal_moves}

    return None


def _json_object_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    n = len(text)
    for start in range(n):
        if text[start] != "{":
            continue

        depth = 0
        in_string = False
        escaped = False
        for idx in range(start, n):
            char = text[idx]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
                continue
            if char == "{":
                depth += 1
                continue
            if char == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : idx + 1])
                    break
                if depth < 0:
                    break
    return candidates


def _parse_prediction_json(answer_text: str) -> Optional[dict[str, Any]]:
    if not isinstance(answer_text, str):
        return None

    stripped = answer_text.strip()
    if not stripped:
        return None

    try:
        payload = json.loads(stripped)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    for candidate in _json_object_candidates(stripped):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload

    return None


def _score_payload_for_example(
    example: QAExample,
    pred_payload: Optional[dict[str, Any]],
    *,
    best_move_optimal_reward: float,
) -> ScoreOutcome:
    if pred_payload is None:
        return ScoreOutcome(
            reward=0.0,
            parse_success=False,
            task_correct=False,
            json_object_parsed=False,
        )

    if example.task_type == "best_move":
        move = _move_from_payload_obj(pred_payload)
        if move is None:
            return ScoreOutcome(
                reward=0.0,
                parse_success=False,
                task_correct=False,
                json_object_parsed=True,
            )
        set_correct = move in example.best_move_optimal_set if move is not None else False
        canonical_correct = move == example.best_move_canonical if move is not None else False
        if canonical_correct:
            reward = 1.0
        elif set_correct:
            reward = float(best_move_optimal_reward)
        else:
            reward = 0.0

        return ScoreOutcome(
            reward=reward,
            parse_success=True,
            task_correct=set_correct,
            json_object_parsed=True,
            best_move_set_correct=set_correct,
            best_move_canonical_correct=canonical_correct,
        )

    gt_norm = _normalize_non_best_answer(example.task_type, example.expected_answer)
    pred_norm = _normalize_non_best_answer(example.task_type, pred_payload)
    if pred_norm is None:
        return ScoreOutcome(
            reward=0.0,
            parse_success=False,
            task_correct=False,
            json_object_parsed=True,
        )
    exact_correct = gt_norm is not None and pred_norm is not None and gt_norm == pred_norm

    return ScoreOutcome(
        reward=1.0 if exact_correct else 0.0,
        parse_success=True,
        task_correct=exact_correct,
        json_object_parsed=True,
        exact_non_best_correct=exact_correct,
    )


def _extract_rollout_answer(rollout: Any) -> str:
    output = getattr(rollout, "output", None)
    if isinstance(output, QueryOutput):
        return output.answer or ""

    answer = getattr(output, "answer", None)
    if isinstance(answer, str):
        return answer
    return ""


def _score_rollout_for_example(
    rollout: Any,
    example: QAExample,
    *,
    best_move_optimal_reward: float,
) -> ScoreOutcome:
    answer_text = _extract_rollout_answer(rollout)
    pred_payload = _parse_prediction_json(answer_text)
    return _score_payload_for_example(
        example,
        pred_payload,
        best_move_optimal_reward=best_move_optimal_reward,
    )


def _resolve_image_path(row: dict[str, Any], dataset_dir: Path) -> Optional[Path]:
    candidates: list[str] = []
    for key in ("image_path", "image"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip())

    for raw in candidates:
        path = Path(raw).expanduser()
        if path.is_file():
            return path.resolve()

        if not path.is_absolute():
            joined = (dataset_dir / path).resolve()
            if joined.is_file():
                return joined

        basename = path.name
        if basename:
            fallback = (dataset_dir / "images" / basename).resolve()
            if fallback.is_file():
                return fallback

    return None


def _build_example(
    row: dict[str, Any],
    *,
    split_name: str,
    dataset_dir: Path,
    line_number: int,
) -> Optional[QAExample]:
    missing = [field for field in REQUIRED_ROW_FIELDS if field not in row]
    if missing:
        raise ValueError(
            f"split={split_name} line={line_number} missing required fields: {missing}"
        )

    task_type = str(row["task_type"]).strip()
    if task_type not in SUPPORTED_TASKS:
        print(
            f"split={split_name} line={line_number} unsupported task_type='{task_type}'; skipping row"
        )
        return None

    image_path = _resolve_image_path(row, dataset_dir)
    if image_path is None:
        print(f"split={split_name} line={line_number} missing image file; skipping row")
        return None

    try:
        expected_answer = json.loads(str(row["final_answer_json"]))
    except json.JSONDecodeError:
        print(f"split={split_name} line={line_number} invalid final_answer_json; skipping row")
        return None

    if not isinstance(expected_answer, dict):
        print(f"split={split_name} line={line_number} final_answer_json is not an object; skipping row")
        return None

    best_move_canonical = _best_move_from_json(str(row["best_move_canonical_json"]))
    best_move_optimal_set = frozenset(_as_move_set(str(row["best_move_optimal_set_json"])))

    return QAExample(
        row_id=str(row["row_id"]),
        split=str(row["split"]),
        task_type=task_type,
        question=str(row["question"]),
        image_path=image_path,
        expected_answer=expected_answer,
        best_move_canonical=best_move_canonical,
        best_move_optimal_set=best_move_optimal_set,
    )


def _load_split_examples(dataset_dir: Path, split_name: str) -> list[QAExample]:
    path = dataset_dir / "jsonl" / f"{split_name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"split JSONL not found: {path}")

    examples: list[QAExample] = []
    skipped = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                skipped += 1
                continue
            example = _build_example(
                row,
                split_name=split_name,
                dataset_dir=dataset_dir,
                line_number=line_number,
            )
            if example is None:
                skipped += 1
                continue
            examples.append(example)

    print(f"loaded split={split_name} rows={len(examples)} skipped={skipped} from {path}")
    if not examples:
        raise ValueError(f"split={split_name} contains no usable rows: {path}")
    return examples


def _prepare_requests(
    examples: list[QAExample],
    *,
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_tokens_by_task: dict[str, int],
    reasoning: bool,
) -> tuple[list[QueryRequest], list[QAExample]]:
    requests: list[QueryRequest] = []
    active: list[QAExample] = []

    for item in examples:
        try:
            image = Image.open(item.image_path).convert("RGB")
        except (FileNotFoundError, OSError) as exc:
            print(f"image load failed for {item.image_path}: {exc}; skipping")
            continue

        requests.append(
            QueryRequest(
                question=item.question,
                image_url=_to_data_url(image, quality=92),
                reasoning=bool(reasoning),
                settings=QuerySettings(
                    temperature=float(temperature),
                    top_p=float(top_p),
                    max_tokens=int(max_tokens_by_task.get(item.task_type, max_tokens)),
                ),
            )
        )
        active.append(item)

    return requests, active


def _is_rate_limit_error(exc: Exception) -> bool:
    if isinstance(exc, TunaAPIError) and exc.status_code == 429:
        return True
    return "too many requests" in str(exc).lower()


def _truncate(value: str, limit: int = 600) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "...<truncated>"


def _error_details(exc: Exception) -> str:
    if isinstance(exc, TunaAPIError):
        parts = [
            f"type=TunaAPIError",
            f"message={exc}",
            f"status_code={exc.status_code}",
        ]
        if exc.request_id:
            parts.append(f"request_id={exc.request_id}")
        body = exc.response_body
        if body is not None:
            if isinstance(body, (dict, list)):
                body_text = json.dumps(body, ensure_ascii=False, separators=(",", ":"))
            else:
                body_text = str(body)
            parts.append(f"response_body={_truncate(body_text)}")
        return " | ".join(parts)

    if isinstance(exc, TunaNetworkError):
        cause = getattr(exc, "cause", None)
        if cause is None:
            return f"type=TunaNetworkError | message={exc}"
        return f"type=TunaNetworkError | message={exc} | cause={type(cause).__name__}: {cause}"

    return f"type={type(exc).__name__} | message={exc}"


def _rollouts_batch_with_retry(
    *,
    finetune: Any,
    requests: list[QueryRequest],
    num_rollouts: int,
    max_workers: int,
    retries: int,
    backoff_s: float,
    context: str,
):
    worker_count = max(1, min(max_workers, len(requests)))
    attempt = 0
    total_attempts = retries + 1
    while True:
        try:
            return finetune.rollouts_batch(
                requests=requests,
                num_rollouts=num_rollouts,
                max_workers=worker_count,
            )
        except (TunaAPIError, TunaNetworkError) as exc:
            should_retry = isinstance(exc, TunaNetworkError) or _is_rate_limit_error(exc)
            if (not should_retry) or attempt >= retries:
                print(
                    f"{context}: rollouts_batch failed with no further retries. "
                    f"attempt={attempt + 1}/{total_attempts} "
                    f"workers={worker_count} "
                    f"details: {_error_details(exc)}"
                )
                raise
            delay = max(0.1, float(backoff_s)) * (2**attempt)
            next_worker_count = max(1, worker_count // 2)
            print(
                f"{context}: retrying rollouts_batch. "
                f"attempt={attempt + 1}/{total_attempts} "
                f"workers={worker_count}->{next_worker_count} "
                f"sleep={delay:.2f}s "
                f"details: {_error_details(exc)}"
            )
            time.sleep(delay)
            worker_count = next_worker_count
            attempt += 1


def _evaluate_split(
    *,
    finetune: Any,
    examples: list[QAExample],
    split_name: str,
    seed: int,
    batch_size: int,
    max_workers: int,
    max_samples: Optional[int],
    rollout_retries: int,
    rollout_retry_backoff_s: float,
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_tokens_by_task: dict[str, int],
    reasoning: bool,
    best_move_optimal_reward: float,
    show_progress: bool,
) -> dict[str, float]:
    rng = random.Random(seed)
    indices = list(range(len(examples)))
    rng.shuffle(indices)
    if max_samples is not None:
        indices = indices[: max(0, min(max_samples, len(indices)))]

    total_scored = 0
    reward_sum = 0.0
    object_parse_count = 0
    parse_success_count = 0

    best_move_total = 0
    best_move_set_correct = 0
    best_move_canonical_correct = 0

    non_best_total = 0
    non_best_exact_correct = 0

    per_task_total: Counter[str] = Counter()
    per_task_correct: Counter[str] = Counter()

    def _consume_batch(batch_examples: list[QAExample]) -> None:
        nonlocal total_scored, reward_sum, object_parse_count, parse_success_count
        nonlocal best_move_total, best_move_set_correct, best_move_canonical_correct
        nonlocal non_best_total, non_best_exact_correct

        if not batch_examples:
            return

        requests, active_examples = _prepare_requests(
            batch_examples,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            max_tokens_by_task=max_tokens_by_task,
            reasoning=reasoning,
        )
        if not requests:
            return

        try:
            results = _rollouts_batch_with_retry(
                finetune=finetune,
                requests=requests,
                num_rollouts=1,
                max_workers=min(max_workers, len(requests)),
                retries=rollout_retries,
                backoff_s=rollout_retry_backoff_s,
                context=f"eval split={split_name}",
            )
        except (TunaAPIError, TunaNetworkError) as exc:
            print(
                f"eval split={split_name}: rollouts_batch failed; skipping batch. "
                f"details: {_error_details(exc)}"
            )
            return

        if len(results) != len(active_examples):
            print(
                f"warning: eval split={split_name} got {len(results)} results for "
                f"{len(active_examples)} requests; scoring aligned subset"
            )

        for item, result in zip(active_examples, results):
            per_task_total[item.task_type] += 1
            total_scored += 1

            if not result.rollouts:
                outcome = ScoreOutcome(reward=0.0, parse_success=False, task_correct=False)
            else:
                outcome = _score_rollout_for_example(
                    result.rollouts[0],
                    item,
                    best_move_optimal_reward=best_move_optimal_reward,
                )

            reward_sum += float(outcome.reward)
            if outcome.json_object_parsed:
                object_parse_count += 1
            if outcome.parse_success:
                parse_success_count += 1
            if outcome.task_correct:
                per_task_correct[item.task_type] += 1

            if item.task_type == "best_move":
                best_move_total += 1
                if outcome.best_move_set_correct:
                    best_move_set_correct += 1
                if outcome.best_move_canonical_correct:
                    best_move_canonical_correct += 1
            else:
                non_best_total += 1
                if outcome.exact_non_best_correct:
                    non_best_exact_correct += 1

    batch: list[QAExample] = []
    eval_iter = tqdm(
        indices,
        desc=f"eval:{split_name}",
        total=len(indices),
        dynamic_ncols=True,
        leave=False,
        disable=not show_progress,
    )
    for idx in eval_iter:
        batch.append(examples[idx])
        if len(batch) >= batch_size:
            _consume_batch(batch)
            batch = []

    if batch:
        _consume_batch(batch)

    if total_scored == 0:
        metrics: dict[str, float] = {
            "eval_samples": 0,
            "eval_reward_mean": 0.0,
            "eval_json_object_rate": 0.0,
            "eval_json_parse_rate": 0.0,
            "eval_best_move_set_accuracy": 0.0,
            "eval_best_move_canonical_accuracy": 0.0,
            "eval_exact_accuracy_non_best_move": 0.0,
        }
        for task in sorted(SUPPORTED_TASKS):
            metrics[f"eval_task_accuracy_{task}"] = 0.0
            metrics[f"eval_task_count_{task}"] = 0.0
        return metrics

    metrics = {
        "eval_samples": float(total_scored),
        "eval_reward_mean": reward_sum / total_scored,
        "eval_json_object_rate": object_parse_count / total_scored,
        "eval_json_parse_rate": parse_success_count / total_scored,
        "eval_best_move_set_accuracy": best_move_set_correct / max(1, best_move_total),
        "eval_best_move_canonical_accuracy": best_move_canonical_correct / max(1, best_move_total),
        "eval_exact_accuracy_non_best_move": non_best_exact_correct / max(1, non_best_total),
    }

    for task in sorted(per_task_total.keys()):
        metrics[f"eval_task_accuracy_{task}"] = per_task_correct[task] / max(1, per_task_total[task])
        metrics[f"eval_task_count_{task}"] = float(per_task_total[task])

    return metrics


def _metric_prefix_for_split(split_name: str) -> str:
    sanitized = "".join(ch if ch.isalnum() else "_" for ch in split_name.strip())
    return f"final_{sanitized}_"


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    args.env_file = _resolve_env_file(args.env_file)
    show_progress = _progress_enabled(args.no_progress)

    load_dotenv(args.env_file, override=False)
    if not args.api_key:
        args.api_key = os.environ.get("MOONDREAM_API_KEY", "")
    if not args.base_url:
        args.base_url = os.environ.get("TUNA_BASE_URL", DEFAULT_BASE_URL)

    _validate_args(args)
    if not args.api_key:
        raise ValueError("MOONDREAM_API_KEY is required")

    if args.eval_max_samples is not None and args.eval_max_samples <= 0:
        args.eval_max_samples = None

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"dataset_dir not found: {dataset_dir}")

    final_eval_splits = _dedupe_splits(list(args.final_eval_splits))
    if not final_eval_splits:
        raise ValueError("--final-eval-splits must contain at least one split")

    rng = random.Random(args.seed)

    train_examples = _load_split_examples(dataset_dir, args.train_split)
    val_examples = _load_split_examples(dataset_dir, args.val_split)
    train_examples_by_task: dict[str, list[QAExample]] = {}
    for item in train_examples:
        train_examples_by_task.setdefault(item.task_type, []).append(item)
    if not train_examples_by_task:
        raise ValueError("train split has no task-indexable examples")

    sampling_tasks = sorted(train_examples_by_task.keys())
    sampling_weights = [float(args.task_sampling_weights.get(task, 1.0)) for task in sampling_tasks]
    if not sampling_tasks:
        raise ValueError("no tasks available for weighted sampling")

    final_eval_examples: dict[str, list[QAExample]] = {}
    for split in final_eval_splits:
        if split == args.val_split:
            final_eval_examples[split] = val_examples
        else:
            final_eval_examples[split] = _load_split_examples(dataset_dir, split)

    if not args.finetune_id and not args.finetune_name:
        args.finetune_name = f"ttt-query-rl-{_random_suffix()}"

    client = TunaClient(api_key=args.api_key, base_url=args.base_url)
    if args.finetune_id:
        finetune = client.get_finetune(args.finetune_id)
    else:
        finetune = client.create_finetune(name=args.finetune_name, rank=args.rank)

    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or None,
        config={
            "config": args.config,
            "env_file": args.env_file,
            "base_url": args.base_url,
            "dataset_dir": str(dataset_dir),
            "train_split": args.train_split,
            "val_split": args.val_split,
            "final_eval_splits": final_eval_splits,
            "train_rows": len(train_examples),
            "val_rows": len(val_examples),
            "finetune_id": finetune.finetune_id,
            "finetune_name": finetune.name,
            "rank": args.rank,
            "seed": args.seed,
            "num_steps": args.num_steps,
            "resume_step": args.resume_step,
            "batch_size": args.batch_size,
            "group_size": args.group_size,
            "lr": args.lr,
            "max_workers": args.max_workers,
            "rollout_retries": args.rollout_retries,
            "rollout_retry_backoff_s": args.rollout_retry_backoff_s,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "max_tokens_by_task": args.max_tokens_by_task,
            "reasoning": args.reasoning,
            "task_sampling_weights": {
                task: float(args.task_sampling_weights.get(task, 1.0))
                for task in sorted(args.task_sampling_weights.keys())
            },
            "eval_every": args.eval_every,
            "save_every": args.save_every,
            "eval_batch_size": args.eval_batch_size,
            "eval_max_samples": args.eval_max_samples,
            "best_metric": args.best_metric,
            "best_move_optimal_reward": args.best_move_optimal_reward,
        },
    )
    run.summary["finetune_id"] = finetune.finetune_id

    best_metric_value: Optional[float] = None
    best_step: Optional[int] = None

    if args.eval_every > 0:
        print("running baseline validation eval...")
        baseline_metrics = _evaluate_split(
            finetune=finetune,
            examples=val_examples,
            split_name=args.val_split,
            seed=args.seed + 101,
            batch_size=args.eval_batch_size,
            max_workers=args.max_workers,
            max_samples=args.eval_max_samples,
            rollout_retries=args.rollout_retries,
            rollout_retry_backoff_s=args.rollout_retry_backoff_s,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            max_tokens_by_task=args.max_tokens_by_task,
            reasoning=args.reasoning,
            best_move_optimal_reward=args.best_move_optimal_reward,
            show_progress=show_progress,
        )
        wandb.log({f"baseline_{k}": v for k, v in baseline_metrics.items()}, step=args.resume_step)
        wandb.log(baseline_metrics, step=args.resume_step)

        baseline_metric = float(baseline_metrics.get(args.best_metric, 0.0))
        best_metric_value = baseline_metric
        best_step = args.resume_step
        print(
            f"baseline eval step={args.resume_step} reward={baseline_metrics['eval_reward_mean']:.4f} "
            f"obj_parse={baseline_metrics['eval_json_object_rate']:.4f} "
            f"parse={baseline_metrics['eval_json_parse_rate']:.4f} "
            f"best_move_set={baseline_metrics['eval_best_move_set_accuracy']:.4f} "
            f"best_move_canon={baseline_metrics['eval_best_move_canonical_accuracy']:.4f} "
            f"exact_non_best={baseline_metrics['eval_exact_accuracy_non_best_move']:.4f}"
        )

    step_iter = tqdm(
        range(args.num_steps),
        desc="train",
        total=args.num_steps,
        dynamic_ncols=True,
        disable=not show_progress,
    )
    sampled_task_running_counts: Counter[str] = Counter()
    sampled_task_running_total = 0
    for step in step_iter:
        global_step = args.resume_step + step

        sampled_tasks = rng.choices(sampling_tasks, weights=sampling_weights, k=args.batch_size)
        sampled_task_counts = Counter(sampled_tasks)
        sampled_task_running_counts.update(sampled_task_counts)
        sampled_task_running_total += len(sampled_tasks)
        batch = [rng.choice(train_examples_by_task[task_name]) for task_name in sampled_tasks]
        requests, active_examples = _prepare_requests(
            batch,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            max_tokens_by_task=args.max_tokens_by_task,
            reasoning=args.reasoning,
        )
        if not requests:
            print(f"step {global_step}: no valid requests in batch; skipping")
            continue

        try:
            results = _rollouts_batch_with_retry(
                finetune=finetune,
                requests=requests,
                num_rollouts=args.group_size,
                max_workers=min(args.max_workers, len(requests)),
                retries=args.rollout_retries,
                backoff_s=args.rollout_retry_backoff_s,
                context=f"train step {global_step}",
            )
        except (TunaAPIError, TunaNetworkError) as exc:
            print(
                f"step {global_step}: rollouts_batch failed; skipping step. "
                f"details: {_error_details(exc)}"
            )
            continue

        if len(results) != len(active_examples):
            print(
                f"warning: step {global_step} got {len(results)} rollout results for "
                f"{len(active_examples)} requests; training on aligned subset"
            )

        groups = []
        rewards_all: list[float] = []
        object_parses = 0
        parse_successes = 0

        for item, result in zip(active_examples, results):
            if not result.rollouts:
                continue

            rewards: list[float] = []
            for rollout in result.rollouts:
                outcome = _score_rollout_for_example(
                    rollout,
                    item,
                    best_move_optimal_reward=args.best_move_optimal_reward,
                )
                rewards.append(outcome.reward)
                rewards_all.append(outcome.reward)
                if outcome.json_object_parsed:
                    object_parses += 1
                if outcome.parse_success:
                    parse_successes += 1

            if rewards:
                groups.append(result.to_group(rewards=rewards))

        if not groups:
            print(f"step {global_step}: no train groups produced; skipping")
            continue

        try:
            train_out = finetune.train_step(groups=groups, lr=args.lr)
        except (TunaAPIError, TunaNetworkError) as exc:
            print(
                f"step {global_step}: train_step failed; skipping step. "
                f"details: {_error_details(exc)}"
            )
            continue

        reward_mean = float(np.mean(rewards_all)) if rewards_all else 0.0
        reward_var = float(np.var(rewards_all)) if rewards_all else 0.0
        object_parse_rate = object_parses / max(1, len(rewards_all))
        parse_rate = parse_successes / max(1, len(rewards_all))

        train_metrics: dict[str, float] = {
            "reward_mean": reward_mean,
            "reward_var": reward_var,
            "train_json_object_rate": object_parse_rate,
            "train_json_parse_rate": parse_rate,
            "accepted_groups": float(len(groups)),
            "kl": float(train_out.kl or 0.0),
            "router_kl": float(train_out.router_kl or 0.0),
            "grad_norm": float(train_out.grad_norm or 0.0),
        }
        for task_name in sampling_tasks:
            step_count = sampled_task_counts.get(task_name, 0)
            train_metrics[f"batch_task_count_{task_name}"] = float(step_count)
            train_metrics[f"batch_task_share_{task_name}"] = step_count / max(1, args.batch_size)
            train_metrics[f"running_task_share_{task_name}"] = (
                sampled_task_running_counts.get(task_name, 0) / max(1, sampled_task_running_total)
            )

        wandb.log(train_metrics, step=global_step)

        print(
            f"step {global_step} reward={reward_mean:.4f} "
            f"obj_parse_rate={object_parse_rate:.4f} parse_rate={parse_rate:.4f} "
            f"kl={float(train_out.kl or 0.0):.4f} "
            f"router_kl={float(train_out.router_kl or 0.0):.4f} "
            f"grad_norm={float(train_out.grad_norm or 0.0):.4f}"
        )
        if show_progress:
            step_iter.set_postfix(
                reward=f"{reward_mean:.3f}",
                obj_parse=f"{object_parse_rate:.3f}",
                parse=f"{parse_rate:.3f}",
                kl=f"{float(train_out.kl or 0.0):.3f}",
            )

        if args.eval_every > 0 and (global_step + 1) % args.eval_every == 0:
            eval_metrics = _evaluate_split(
                finetune=finetune,
                examples=val_examples,
                split_name=args.val_split,
                seed=args.seed + 1000 + global_step,
                batch_size=args.eval_batch_size,
                max_workers=args.max_workers,
                max_samples=args.eval_max_samples,
                rollout_retries=args.rollout_retries,
                rollout_retry_backoff_s=args.rollout_retry_backoff_s,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                max_tokens_by_task=args.max_tokens_by_task,
                reasoning=args.reasoning,
                best_move_optimal_reward=args.best_move_optimal_reward,
                show_progress=show_progress,
            )
            wandb.log(eval_metrics, step=global_step)
            metric_value = float(eval_metrics.get(args.best_metric, 0.0))
            print(
                f"eval step {global_step} reward={eval_metrics['eval_reward_mean']:.4f} "
                f"obj_parse={eval_metrics['eval_json_object_rate']:.4f} "
                f"parse={eval_metrics['eval_json_parse_rate']:.4f} "
                f"best_move_set={eval_metrics['eval_best_move_set_accuracy']:.4f} "
                f"best_move_canon={eval_metrics['eval_best_move_canonical_accuracy']:.4f} "
                f"exact_non_best={eval_metrics['eval_exact_accuracy_non_best_move']:.4f}"
            )

            if best_metric_value is None or metric_value > best_metric_value:
                best_metric_value = metric_value
                best_step = global_step
                finetune.save_checkpoint()
                print(
                    f"new best {args.best_metric}={metric_value:.4f} at step {global_step}; checkpoint saved"
                )

        if args.save_every > 0 and (global_step + 1) % args.save_every == 0:
            finetune.save_checkpoint()

    finetune.save_checkpoint()

    final_eval_step = args.resume_step + args.num_steps
    for idx, (split_name, split_examples) in enumerate(final_eval_examples.items()):
        eval_metrics = _evaluate_split(
            finetune=finetune,
            examples=split_examples,
            split_name=split_name,
            seed=args.seed + 5000 + idx,
            batch_size=args.eval_batch_size,
            max_workers=args.max_workers,
            max_samples=args.eval_max_samples,
            rollout_retries=args.rollout_retries,
            rollout_retry_backoff_s=args.rollout_retry_backoff_s,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            max_tokens_by_task=args.max_tokens_by_task,
            reasoning=args.reasoning,
            best_move_optimal_reward=args.best_move_optimal_reward,
            show_progress=show_progress,
        )
        prefix = _metric_prefix_for_split(split_name)
        wandb.log({f"{prefix}{k}": v for k, v in eval_metrics.items()}, step=final_eval_step)

        print(
            f"final eval split={split_name} reward={eval_metrics['eval_reward_mean']:.4f} "
            f"obj_parse={eval_metrics['eval_json_object_rate']:.4f} "
            f"parse={eval_metrics['eval_json_parse_rate']:.4f} "
            f"best_move_set={eval_metrics['eval_best_move_set_accuracy']:.4f} "
            f"best_move_canon={eval_metrics['eval_best_move_canonical_accuracy']:.4f} "
            f"exact_non_best={eval_metrics['eval_exact_accuracy_non_best_move']:.4f}"
        )

    run.summary["best_metric_name"] = args.best_metric
    run.summary["best_metric_value"] = float(best_metric_value or 0.0)
    run.summary["best_metric_step"] = int(best_step if best_step is not None else -1)
    run.summary["finetune_id"] = finetune.finetune_id
    run.summary["train_rows"] = len(train_examples)
    run.summary["val_rows"] = len(val_examples)

    run.finish()
    client.close()

    print(
        f"done. finetune_id={finetune.finetune_id} best_{args.best_metric}={best_metric_value} "
        f"best_step={best_step}"
    )


if __name__ == "__main__":
    main()
