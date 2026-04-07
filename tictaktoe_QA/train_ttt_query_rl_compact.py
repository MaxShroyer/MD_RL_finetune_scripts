#!/usr/bin/env python3
"""Query RL finetuning example for TicTacToe QA.

Requires:
  pip install datasets pillow python-dotenv wandb
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import random
import re
import string
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, pvariance
from typing import Any, Optional

from dotenv import load_dotenv
from PIL import Image

try:
    from tqdm.auto import tqdm  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    class _SimpleTqdm:  # pragma: no cover
        def __init__(self, iterable: Any, *_args: Any, **_kwargs: Any) -> None:
            self._iterable = iterable

        def __iter__(self):
            return iter(self._iterable)

        def set_postfix(self, *_args: Any, **_kwargs: Any) -> None:
            return

    def tqdm(iterable=None, *args, **kwargs):  # type: ignore
        return _SimpleTqdm(iterable or [], *args, **kwargs)

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
        def init(*_args: Any, **_kwargs: Any) -> _WandbRun:
            print("wandb not installed; continuing without remote logging.")
            return _WandbRun()

        @staticmethod
        def log(*_args: Any, **_kwargs: Any) -> None:
            return

    wandb = _WandbShim()

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from async_checkpoint_eval import (
    CheckpointEvalResult,
    DispatchHandle,
    dispatch_checkpoint_eval,
    drain_checkpoint_eval_jobs,
    poll_checkpoint_eval_jobs,
)
from tuna_sdk import QueryRequest, QuerySettings, TunaClient  # noqa: E402
from tuna_sdk.errors import TunaAPIError, TunaNetworkError  # noqa: E402

DEFAULT_BASE_URL = "https://api.moondream.ai/v1"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "query_rl_compact.json"
DEFAULT_DATASET_DIR = Path(__file__).resolve().parent / "synth_dataset" / "outputs" / "v2"
DEFAULT_ENV_FILE = Path(__file__).resolve().parent / ".env"
DEFAULT_DATASET_SOURCE = "hf_hub"
SUPPORTED_DATASET_SOURCES = ("hf_hub", "local_jsonl")
DEFAULT_HF_DATASET_REPO_ID = "maxs-m87/tictactoe-qa-v1"
DEFAULT_HF_DATASET_REVISION = "main"
CANONICAL_TASK_TYPES: tuple[str, ...] = (
    "best_move",
    "has_winning_move",
    "turn_player",
    "winner",
    "is_game_over",
    "available_moves_count",
    "available_moves_list",
)
TASK_TYPE_ALIASES = {
    "is_terminal": "is_game_over",
    "legal_moves_count": "available_moves_count",
    "legal_moves_list": "available_moves_list",
}
ANSWER_KEY_ALIASES = {
    "is_game_over": {"is_game_over": "is_game_over", "is_terminal": "is_game_over"},
    "available_moves_count": {"available_move_count": "available_move_count", "legal_move_count": "available_move_count"},
    "available_moves_list": {"available_moves": "available_moves", "legal_moves": "available_moves"},
}
DEFAULT_FINAL_EVAL_SPLITS = ["val", "test"]
DEFAULT_TASK_SAMPLING_WEIGHTS = {
    "best_move": 1.0,
    "available_moves_count": 1.0,
    "available_moves_list": 1.0,
}
DEFAULT_MAX_TOKENS_BY_TASK = {"available_moves_list": 800}
BEST_MOVE_REWARD_MODES = ("ranked", "hybrid_strict", "binary")
BEST_METRICS = (
    "eval_reward_mean",
    "eval_best_move_set_accuracy",
    "eval_best_move_canonical_accuracy",
    "eval_exact_accuracy_non_best_move",
    "eval_json_parse_rate",
)
INTRA_TASK_SAMPLING_STRATEGIES: dict[str, set[str]] = {
    "turn_player": {"balanced_player"},
    "available_moves_count": {"uniform_count"},
    "best_move": {"uniform_canonical_move", "center_hard_negative"},
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
CONFIG_DERIVED_KEYS = {
    "task_sampling_weights",
    "max_tokens_by_task",
    "intra_task_sampling_json",
}
WANDB_TRAIN_KEYS = (
    "reward_mean",
    "train_json_object_rate",
    "train_json_parse_rate",
    "train_best_move_valid_prediction_count",
    "train_best_move_valid_prediction_rate",
    "off_policy_group_fraction",
    "replay_buffer_size",
    "kl",
)
WANDB_EVAL_KEYS = (
    "eval_samples",
    "eval_reward_mean",
    "eval_json_object_rate",
    "eval_json_parse_rate",
    "eval_best_move_set_accuracy",
    "eval_best_move_canonical_accuracy",
    "eval_best_move_valid_prediction_count",
    "eval_best_move_valid_prediction_rate",
    "eval_exact_accuracy_non_best_move",
    "eval_best_move_center_prediction_rate",
    "eval_best_move_invalid_prediction_rate",
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
    best_move_scores: tuple[tuple[int, int, int], ...] = tuple()
    best_move_legal_moves: frozenset[int] = frozenset()


@dataclass(frozen=True)
class ScoreOutcome:
    reward: float
    parse_success: bool
    task_correct: bool
    json_object_parsed: bool = False
    best_move_set_correct: bool = False
    best_move_canonical_correct: bool = False
    exact_non_best_correct: bool = False
    best_move_valid_prediction: bool = False


def normalize_task_type(task_type: str, *, allow_unknown: bool = False) -> str:
    normalized = TASK_TYPE_ALIASES.get(str(task_type).strip(), str(task_type).strip())
    if normalized in CANONICAL_TASK_TYPES:
        return normalized
    if allow_unknown:
        return normalized
    raise ValueError(f"unknown task_type: {task_type}")


def normalize_answer_payload_for_task(task_type: str, payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    aliases = ANSWER_KEY_ALIASES.get(normalize_task_type(task_type, allow_unknown=True))
    if not aliases:
        return payload
    out: dict[str, Any] = {}
    for old_key, canonical_key in aliases.items():
        if old_key in payload and canonical_key not in out:
            out[canonical_key] = payload[old_key]
    return out or payload


def _normalize_dataset_source(raw_source: str) -> str:
    source = str(raw_source or "").strip().lower()
    if source in SUPPORTED_DATASET_SOURCES:
        return source
    raise ValueError(f"dataset_source must be one of {sorted(SUPPORTED_DATASET_SOURCES)}, got: {raw_source!r}")


def _resolve_hf_token(raw_token: str) -> str:
    return (
        str(raw_token or "").strip()
        or os.environ.get("HF_TOKEN", "").strip()
        or os.environ.get("HUGGINGFACE_HUB_TOKEN", "").strip()
    )


def _safe_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    return cleaned or "default"


def _cache_dir(raw_cache_dir: str) -> Path:
    return Path(raw_cache_dir).expanduser().resolve() if str(raw_cache_dir or "").strip() else (Path.cwd() / ".cache" / "tictactoe_qa").resolve()


def _persist_image_bytes(image_bytes: bytes, *, image_cache_root: Path) -> Optional[Path]:
    if not image_bytes:
        return None
    digest = hashlib.sha1(image_bytes).hexdigest()  # noqa: S324
    out_path = image_cache_root / f"{digest}.png"
    if out_path.exists():
        return out_path.resolve()
    image_cache_root.mkdir(parents=True, exist_ok=True)
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.convert("RGB").save(out_path, format="PNG")
        return out_path.resolve()
    except OSError:
        fallback = image_cache_root / f"{digest}.img"
        if not fallback.exists():
            fallback.write_bytes(image_bytes)
        return fallback.resolve()


def _existing_path(raw_path: str, *, dataset_dir: Optional[Path]) -> Optional[Path]:
    text = str(raw_path or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if path.is_file():
        return path.resolve()
    if dataset_dir is not None and not path.is_absolute():
        joined = (dataset_dir / path).resolve()
        if joined.is_file():
            return joined
    return None


def _resolve_hf_row_image_path(
    row: dict[str, Any],
    *,
    dataset_dir: Optional[Path],
    image_cache_root: Path,
) -> Optional[Path]:
    for key in ("image_path", "image"):
        value = row.get(key)
        if isinstance(value, str):
            resolved = _existing_path(value, dataset_dir=dataset_dir)
            if resolved is not None:
                return resolved
    image_payload = row.get("image")
    if isinstance(image_payload, dict):
        payload_path = image_payload.get("path")
        if isinstance(payload_path, str):
            resolved = _existing_path(payload_path, dataset_dir=dataset_dir)
            if resolved is not None:
                return resolved
        payload_bytes = image_payload.get("bytes")
        if isinstance(payload_bytes, (bytes, bytearray)):
            return _persist_image_bytes(bytes(payload_bytes), image_cache_root=image_cache_root)
    if isinstance(image_payload, (bytes, bytearray)):
        return _persist_image_bytes(bytes(image_payload), image_cache_root=image_cache_root)
    return None


def _load_local_jsonl_rows(*, dataset_dir: Path, split_name: str) -> list[dict[str, Any]]:
    path = dataset_dir / "jsonl" / f"{split_name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"split JSONL not found: {path}")
    rows: list[dict[str, Any]] = []
    skipped = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(payload, dict):
                skipped += 1
                continue
            rows.append(payload)
    print(f"loaded split={split_name} rows={len(rows)} skipped={skipped} from {path}")
    return rows


def _load_hf_rows(
    *,
    split_name: str,
    dataset_dir: Optional[Path],
    hf_dataset_repo_id: str,
    hf_dataset_revision: str,
    hf_token: str,
    hf_cache_dir: str,
) -> list[dict[str, Any]]:
    try:
        from datasets import Image as HFImage  # type: ignore
        from datasets import load_dataset  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "datasets is required for dataset_source='hf_hub'. Install with: pip install datasets"
        ) from exc

    repo_id = str(hf_dataset_repo_id or "").strip()
    if not repo_id:
        raise ValueError("hf_dataset_repo_id is required when dataset_source='hf_hub'")
    revision = str(hf_dataset_revision or DEFAULT_HF_DATASET_REVISION).strip() or DEFAULT_HF_DATASET_REVISION
    cache_dir = _cache_dir(hf_cache_dir)
    token = _resolve_hf_token(hf_token)
    load_kwargs: dict[str, Any] = {"split": split_name, "revision": revision, "cache_dir": str(cache_dir)}
    if token:
        load_kwargs["token"] = token
    try:
        dataset = load_dataset(repo_id, **load_kwargs)
    except TypeError:
        token_value = load_kwargs.pop("token", None)
        if token_value:
            load_kwargs["use_auth_token"] = token_value
        dataset = load_dataset(repo_id, **load_kwargs)
    if "image" in getattr(dataset, "column_names", []):
        dataset = dataset.cast_column("image", HFImage(decode=False))
    image_cache_root = cache_dir / "hf_images" / _safe_component(repo_id) / _safe_component(revision) / _safe_component(split_name)
    rows: list[dict[str, Any]] = []
    skipped = 0
    for raw_row in dataset:
        if not isinstance(raw_row, dict):
            skipped += 1
            continue
        row = dict(raw_row)
        image_path = _resolve_hf_row_image_path(row, dataset_dir=dataset_dir, image_cache_root=image_cache_root)
        if image_path is not None:
            row["image_path"] = str(image_path)
            row["image"] = str(image_path)
        rows.append(row)
    print(f"loaded split={split_name} rows={len(rows)} skipped={skipped} from hf_hub repo={repo_id} revision={revision}")
    return rows


def _load_split_rows(
    *,
    dataset_source: str,
    split_name: str,
    dataset_dir: Optional[Path],
    hf_dataset_repo_id: str,
    hf_dataset_revision: str,
    hf_token: str,
    hf_cache_dir: str,
) -> list[dict[str, Any]]:
    source = _normalize_dataset_source(dataset_source)
    if source == "local_jsonl":
        if dataset_dir is None:
            raise ValueError("dataset_dir is required when dataset_source='local_jsonl'")
        return _load_local_jsonl_rows(dataset_dir=dataset_dir, split_name=split_name)
    return _load_hf_rows(
        split_name=split_name,
        dataset_dir=dataset_dir,
        hf_dataset_repo_id=hf_dataset_repo_id,
        hf_dataset_revision=hf_dataset_revision,
        hf_token=hf_token,
        hf_cache_dir=hf_cache_dir,
    )


def _random_suffix(length: int = 6) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return "".join(random.choices(alphabet, k=length))


def _resolve_path(raw_path: str, *, default: Optional[Path] = None) -> Path:
    text = str(raw_path or "").strip()
    path = Path(text).expanduser() if text else (default or Path())
    if path.is_absolute():
        return path
    for base in (Path.cwd(), REPO_ROOT, Path(__file__).resolve().parent):
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    return (Path.cwd() / path).resolve()


def _load_json_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a JSON object: {config_path}")
    return payload


def _config_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null", "inherit"}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_optional_float_arg(raw_value: str) -> Optional[float]:
    text = str(raw_value or "").strip().lower()
    if text in {"", "none", "null", "inherit"}:
        return None
    try:
        return float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected float or inherit/none/null, got: {raw_value!r}"
        ) from exc


def _parse_json_arg(raw_value: str, arg_name: str) -> dict[str, Any]:
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


def _normalize_config_defaults(config: dict[str, Any]) -> dict[str, Any]:
    defaults = dict(config)
    if "final_eval_splits" in defaults and not isinstance(defaults["final_eval_splits"], list):
        defaults["final_eval_splits"] = [str(defaults["final_eval_splits"])]
    if "eval_temperature" in defaults:
        defaults["eval_temperature"] = _config_optional_float(defaults.get("eval_temperature"))
    if "eval_top_p" in defaults:
        defaults["eval_top_p"] = _config_optional_float(defaults.get("eval_top_p"))
    if "eval_reasoning" in defaults:
        value = defaults.get("eval_reasoning")
        defaults["eval_reasoning"] = None if value in {"inherit", "none", "null", ""} else _coerce_bool(value)
    return defaults


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
        if lowered in {"true", "t", "yes", "y", "1", "on"}:
            return True
        if lowered in {"false", "f", "no", "n", "0", "off"}:
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
    return {
        "x": "X",
        "o": "O",
        "draw": "draw",
        "in_progress": "in_progress",
        "inprogress": "in_progress",
    }.get(value.strip().lower().replace(" ", "_"))


def _normalize_legal_moves(payload: Any) -> Optional[tuple[tuple[int, int], ...]]:
    if not isinstance(payload, list):
        return None
    moves: list[tuple[int, int]] = []
    for item in payload:
        if not isinstance(item, dict):
            return None
        row = _parse_int(item.get("row"))
        col = _parse_int(item.get("col"))
        if row is None or col is None or row < 1 or row > 3 or col < 1 or col > 3:
            return None
        moves.append((row, col))
    return tuple(moves)


def _normalize_non_best_answer(task_type: str, payload: Any) -> Optional[dict[str, Any]]:
    if not isinstance(payload, dict):
        return None
    task_type = normalize_task_type(task_type, allow_unknown=True)
    payload = normalize_answer_payload_for_task(task_type, payload)
    if task_type == "winner":
        winner = _coerce_winner(payload.get("winner"))
        return None if winner is None else {"winner": winner}
    if task_type == "is_game_over":
        is_game_over = _coerce_bool(payload.get("is_game_over"))
        return None if is_game_over is None else {"is_game_over": is_game_over}
    if task_type == "has_winning_move":
        has_winning_move = _coerce_bool(payload.get("has_winning_move"))
        return None if has_winning_move is None else {"has_winning_move": has_winning_move}
    if task_type == "turn_player":
        player = _coerce_player(payload.get("player"))
        return None if player is None else {"player": player}
    if task_type == "available_moves_count":
        count = _parse_int(payload.get("available_move_count"))
        return None if count is None or count < 0 or count > 9 else {"available_move_count": count}
    if task_type == "available_moves_list":
        moves = _normalize_legal_moves(payload.get("available_moves"))
        return None if moves is None else {"available_moves": moves}
    return None


def _task_map(
    raw_map: dict[str, Any],
    *,
    source: str,
    value_kind: str,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for raw_task, raw_value in raw_map.items():
        try:
            task = normalize_task_type(str(raw_task).strip())
        except ValueError as exc:
            raise ValueError(f"{source}: unknown task_type '{raw_task}'") from exc
        if value_kind == "float":
            try:
                value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{source}: weight for task_type '{task}' must be numeric") from exc
            if value < 0.0:
                raise ValueError(f"{source}: weight for task_type '{task}' must be >= 0")
            out[task] = value
        elif value_kind == "int":
            value = _parse_int(raw_value)
            if value is None or value <= 0:
                raise ValueError(f"{source}: max_tokens for task_type '{task}' must be > 0")
            out[task] = value
        else:
            strategy = str(raw_value).strip().lower()
            allowed = INTRA_TASK_SAMPLING_STRATEGIES.get(task)
            if not allowed:
                raise ValueError(f"{source}: task_type '{task}' does not support intra-task sampling")
            if strategy not in allowed:
                raise ValueError(
                    f"{source}: strategy '{strategy}' is not valid for task_type '{task}'. allowed={sorted(allowed)}"
                )
            out[task] = strategy
    return out


def _resolve_task_sampling_weights(config_map: dict[str, Any], cli_override_json: str) -> dict[str, float]:
    resolved = {task: 1.0 for task in CANONICAL_TASK_TYPES}
    resolved.update(_task_map(config_map, source="config task_sampling_weights", value_kind="float"))
    if str(cli_override_json or "").strip():
        resolved.update(
            _task_map(
                _parse_json_arg(cli_override_json, "--task-sampling-weights-json"),
                source="--task-sampling-weights-json",
                value_kind="float",
            )
        )
    return resolved


def _resolve_max_tokens_by_task(config_map: dict[str, Any], cli_override_json: str) -> dict[str, int]:
    resolved = _task_map(config_map, source="config max_tokens_by_task", value_kind="int")
    if str(cli_override_json or "").strip():
        resolved.update(
            _task_map(
                _parse_json_arg(cli_override_json, "--max-tokens-by-task-json"),
                source="--max-tokens-by-task-json",
                value_kind="int",
            )
        )
    return resolved


def _resolve_intra_task_sampling(config_map: dict[str, Any], cli_override_json: str) -> dict[str, str]:
    resolved = _task_map(config_map, source="config intra_task_sampling_json", value_kind="str")
    if str(cli_override_json or "").strip():
        resolved.update(
            _task_map(
                _parse_json_arg(cli_override_json, "--intra-task-sampling-json"),
                source="--intra-task-sampling-json",
                value_kind="str",
            )
        )
    return resolved


def _task_sampling_weights_for(tasks: list[str], weights_by_task: dict[str, float]) -> list[float]:
    weights = [float(weights_by_task.get(task, 1.0)) for task in tasks]
    if tasks and not any(weight > 0.0 for weight in weights):
        raise ValueError("effective task_sampling_weights must include at least one positive task weight")
    return weights


def _active_eval_tasks(task_sampling_weights: dict[str, float]) -> set[str]:
    return {task for task, weight in task_sampling_weights.items() if float(weight) > 0.0}


def _filter_examples_by_task_weight(
    examples: list[QAExample],
    *,
    active_tasks: set[str],
) -> list[QAExample]:
    return [example for example in examples if example.task_type in active_tasks]


def _intra_task_bucket(example: QAExample, strategy: str) -> Optional[str]:
    if strategy == "balanced_player":
        payload = _normalize_non_best_answer("turn_player", example.expected_answer)
        return payload.get("player") if payload else None
    if strategy == "uniform_count":
        payload = _normalize_non_best_answer("available_moves_count", example.expected_answer)
        count = _parse_int(payload.get("available_move_count")) if payload else None
        return None if count is None else str(count)
    if strategy == "uniform_canonical_move":
        move = _parse_int(example.best_move_canonical)
        return None if move is None or move < 1 or move > 9 else str(move)
    if strategy == "center_hard_negative":
        return "other" if 5 in example.best_move_optimal_set else "center_not_optimal"
    return None


def _prepare_intra_task_sampling_groups(
    *,
    train_examples_by_task: dict[str, list[QAExample]],
    intra_task_sampling: dict[str, str],
    best_move_center_not_optimal_ratio: float,
) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for task_name, strategy in sorted(intra_task_sampling.items()):
        buckets: dict[str, list[QAExample]] = {}
        for example in train_examples_by_task.get(task_name, []):
            key = _intra_task_bucket(example, strategy)
            if key is not None:
                buckets.setdefault(key, []).append(example)
        buckets = {key: rows for key, rows in buckets.items() if rows}
        if len(buckets) < 2:
            continue
        payload: dict[str, Any] = {"bucket_keys": sorted(buckets), "buckets": buckets}
        if strategy == "center_hard_negative":
            ratio = float(best_move_center_not_optimal_ratio)
            weights = {
                key: (ratio if key == "center_not_optimal" else 1.0 - ratio)
                for key in sorted(buckets)
            }
            total = sum(max(0.0, value) for value in weights.values())
            if total > 0.0:
                payload["bucket_pick_weights"] = {
                    key: max(0.0, value) / total for key, value in weights.items()
                }
        groups[task_name] = payload
    return groups


def _sample_training_example(
    *,
    task_name: str,
    train_examples_by_task: dict[str, list[QAExample]],
    intra_task_sampling_groups: dict[str, dict[str, Any]],
    rng: random.Random,
) -> QAExample:
    rows = train_examples_by_task[task_name]
    group = intra_task_sampling_groups.get(task_name)
    if not group:
        return rng.choice(rows)
    bucket_keys = list(group.get("bucket_keys", []))
    buckets = group.get("buckets", {})
    if not bucket_keys or not isinstance(buckets, dict):
        return rng.choice(rows)
    weights_map = group.get("bucket_pick_weights", {})
    weights = None
    if isinstance(weights_map, dict):
        candidate = [float(weights_map.get(key, 0.0)) for key in bucket_keys]
        if any(weight > 0.0 for weight in candidate):
            weights = candidate
    bucket_key = rng.choices(bucket_keys, weights=weights, k=1)[0] if weights else rng.choice(bucket_keys)
    bucket_rows = buckets.get(bucket_key)
    return rng.choice(bucket_rows) if isinstance(bucket_rows, list) and bucket_rows else rng.choice(rows)


def _to_data_url(image: Image.Image, *, quality: int = 92) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=max(1, min(100, int(quality))))
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _row_col_to_move(row: int, col: int) -> Optional[int]:
    if row < 1 or row > 3 or col < 1 or col > 3:
        return None
    return ((row - 1) * 3) + col


def _move_from_payload(payload: Any) -> Optional[int]:
    if not isinstance(payload, dict):
        return None
    if "move" in payload:
        move = _parse_int(payload.get("move"))
        return move if move is not None and 1 <= move <= 9 else None
    row = _parse_int(payload.get("row"))
    col = _parse_int(payload.get("col"))
    return None if row is None or col is None else _row_col_to_move(row, col)


def _parse_prediction_json(answer_text: str) -> Optional[dict[str, Any]]:
    if not isinstance(answer_text, str):
        return None
    text = answer_text.strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        pass
    depth = 0
    start: Optional[int] = None
    in_string = False
    escaped = False
    for index, char in enumerate(text):
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
            if depth == 0:
                start = index
            depth += 1
            continue
        if char == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    payload = json.loads(text[start : index + 1])
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    return payload
    return None


def _best_move_from_json(payload_json: str) -> Optional[int]:
    try:
        return _move_from_payload(json.loads(payload_json))
    except json.JSONDecodeError:
        return None


def _move_set_from_json(payload_json: str) -> frozenset[int]:
    try:
        payload = json.loads(payload_json)
    except json.JSONDecodeError:
        return frozenset()
    if not isinstance(payload, list):
        return frozenset()
    moves = {_move_from_payload(item) for item in payload}
    return frozenset(move for move in moves if move is not None)


def _scores_by_move_from_json(payload_json: str) -> tuple[tuple[int, int, int], ...]:
    try:
        payload = json.loads(payload_json)
    except json.JSONDecodeError:
        return tuple()
    if not isinstance(payload, dict):
        return tuple()
    rows: list[tuple[int, int, int]] = []
    for raw_move, raw_score in payload.items():
        move = _parse_int(raw_move)
        if move is None or move < 1 or move > 9 or not isinstance(raw_score, dict):
            continue
        value = _parse_int(raw_score.get("value"))
        depth = _parse_int(raw_score.get("depth"))
        if value is not None and depth is not None:
            rows.append((move, value, depth))
    rows.sort(key=lambda item: item[0])
    return tuple(rows)


def _best_move_rank_key(value: int, depth: int) -> tuple[int, int]:
    return (value, -depth) if value == 1 else (value, depth)


def _ranked_best_move_reward(move: int, *, scores_by_move: dict[int, tuple[int, int]]) -> float:
    predicted = scores_by_move.get(move)
    if predicted is None:
        return 0.0
    if len(scores_by_move) <= 1:
        return 1.0
    predicted_key = _best_move_rank_key(int(predicted[0]), int(predicted[1]))
    better_count = sum(
        1
        for value, depth in scores_by_move.values()
        if _best_move_rank_key(int(value), int(depth)) > predicted_key
    )
    return max(0.0, min(1.0, 1.0 - (better_count / float(len(scores_by_move) - 1))))


def _best_move_prediction_is_valid(
    move: Optional[int],
    *,
    example: QAExample,
) -> bool:
    if move is None:
        return False
    if example.best_move_legal_moves:
        return move in example.best_move_legal_moves
    if example.best_move_scores:
        return any(scored_move == move for scored_move, _value, _depth in example.best_move_scores)
    return False


def _score_payload_for_example(
    example: QAExample,
    pred_payload: Optional[dict[str, Any]],
    *,
    best_move_optimal_reward: float,
    best_move_reward_mode: str = "ranked",
    best_move_wrong_rank_scale: float = 1.0,
) -> ScoreOutcome:
    if pred_payload is None:
        return ScoreOutcome(reward=0.0, parse_success=False, task_correct=False)
    if example.task_type == "best_move":
        move = _move_from_payload(pred_payload)
        valid_prediction = _best_move_prediction_is_valid(move, example=example)
        if move is None:
            return ScoreOutcome(
                reward=0.0,
                parse_success=False,
                task_correct=False,
                json_object_parsed=True,
                best_move_valid_prediction=valid_prediction,
            )
        set_correct = move in example.best_move_optimal_set
        canonical_correct = move == example.best_move_canonical
        scores = {m: (value, depth) for m, value, depth in example.best_move_scores}
        if best_move_reward_mode == "binary":
            reward = 1.0 if set_correct else 0.0
        elif best_move_reward_mode == "hybrid_strict":
            reward = (
                1.0
                if set_correct
                else float(best_move_wrong_rank_scale) * _ranked_best_move_reward(move, scores_by_move=scores)
            ) if scores else (1.0 if set_correct else 0.0)
        else:
            reward = (
                _ranked_best_move_reward(move, scores_by_move=scores)
                if scores
                else (1.0 if canonical_correct else float(best_move_optimal_reward) if set_correct else 0.0)
            )
        return ScoreOutcome(
            reward=max(0.0, min(1.0, float(reward))),
            parse_success=True,
            task_correct=set_correct,
            json_object_parsed=True,
            best_move_set_correct=set_correct,
            best_move_canonical_correct=canonical_correct,
            best_move_valid_prediction=valid_prediction,
        )
    gt_payload = _normalize_non_best_answer(example.task_type, example.expected_answer)
    pred_norm = _normalize_non_best_answer(example.task_type, pred_payload)
    if gt_payload is None or pred_norm is None:
        return ScoreOutcome(
            reward=0.0,
            parse_success=False,
            task_correct=False,
            json_object_parsed=True,
        )
    exact_correct = gt_payload == pred_norm
    return ScoreOutcome(
        reward=1.0 if exact_correct else 0.0,
        parse_success=True,
        task_correct=exact_correct,
        json_object_parsed=True,
        exact_non_best_correct=exact_correct,
    )


def _extract_rollout_answer(rollout: Any) -> str:
    output = getattr(rollout, "output", None)
    answer = getattr(output, "answer", None)
    return answer if isinstance(answer, str) else ""


def _score_rollout_for_example(
    rollout: Any,
    example: QAExample,
    *,
    best_move_optimal_reward: float,
    best_move_reward_mode: str = "ranked",
    best_move_wrong_rank_scale: float = 1.0,
) -> ScoreOutcome:
    return _score_payload_for_example(
        example,
        _parse_prediction_json(_extract_rollout_answer(rollout)),
        best_move_optimal_reward=best_move_optimal_reward,
        best_move_reward_mode=best_move_reward_mode,
        best_move_wrong_rank_scale=best_move_wrong_rank_scale,
    )


def _resolve_image_path(row: dict[str, Any], dataset_dir: Optional[Path]) -> Optional[Path]:
    candidates: list[str] = []
    for key in ("image_path", "image"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip())
    for raw_path in candidates:
        path = Path(raw_path).expanduser()
        if path.is_file():
            return path.resolve()
        if dataset_dir is not None and not path.is_absolute():
            joined = (dataset_dir / path).resolve()
            if joined.is_file():
                return joined
            fallback = (dataset_dir / "images" / path.name).resolve()
            if fallback.is_file():
                return fallback
    return None


def _build_example(
    row: dict[str, Any],
    *,
    split_name: str,
    dataset_dir: Optional[Path],
    line_number: int,
) -> Optional[QAExample]:
    missing = [field for field in REQUIRED_ROW_FIELDS if field not in row]
    if missing:
        raise ValueError(f"split={split_name} line={line_number} missing required fields: {missing}")
    try:
        task_type = normalize_task_type(str(row["task_type"]).strip())
    except ValueError:
        print(f"split={split_name} line={line_number} unsupported task_type='{row['task_type']}'; skipping row")
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
    best_move_scores = _scores_by_move_from_json(str(row.get("scores_by_move_json", "")))
    best_move_legal_moves = _move_set_from_json(str(row.get("legal_moves_json", "")))
    if not best_move_legal_moves and best_move_scores:
        best_move_legal_moves = frozenset(move for move, _value, _depth in best_move_scores)
    return QAExample(
        row_id=str(row["row_id"]),
        split=str(row["split"]),
        task_type=task_type,
        question=str(row["question"]),
        image_path=image_path,
        expected_answer=expected_answer,
        best_move_canonical=_best_move_from_json(str(row["best_move_canonical_json"])),
        best_move_optimal_set=_move_set_from_json(str(row["best_move_optimal_set_json"])),
        best_move_scores=best_move_scores,
        best_move_legal_moves=best_move_legal_moves,
    )


def _load_split_examples(
    *,
    split_name: str,
    dataset_source: str,
    dataset_dir: Optional[Path],
    hf_dataset_repo_id: str,
    hf_dataset_revision: str,
    hf_token: str,
    hf_cache_dir: str,
) -> list[QAExample]:
    rows = _load_split_rows(
        dataset_source=dataset_source,
        split_name=split_name,
        dataset_dir=dataset_dir,
        hf_dataset_repo_id=hf_dataset_repo_id,
        hf_dataset_revision=hf_dataset_revision,
        hf_token=hf_token,
        hf_cache_dir=hf_cache_dir,
    )
    examples: list[QAExample] = []
    skipped = 0
    for line_number, row in enumerate(rows, start=1):
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
    print(f"usable split={split_name} examples={len(examples)} skipped={skipped}")
    if not examples:
        raise ValueError(f"split={split_name} contains no usable rows")
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
    active_examples: list[QAExample] = []
    for example in examples:
        try:
            with Image.open(example.image_path) as image:
                image_url = _to_data_url(image.convert("RGB"))
        except (FileNotFoundError, OSError) as exc:
            print(f"image load failed for {example.image_path}: {exc}; skipping")
            continue
        requests.append(
            QueryRequest(
                question=example.question,
                image_url=image_url,
                reasoning=bool(reasoning),
                settings=QuerySettings(
                    temperature=float(temperature),
                    top_p=float(top_p),
                    max_tokens=int(max_tokens_by_task.get(example.task_type, max_tokens)),
                ),
            )
        )
        active_examples.append(example)
    return requests, active_examples


def _error_message(exc: Exception) -> str:
    if isinstance(exc, TunaAPIError):
        request_id = f" request_id={exc.request_id}" if exc.request_id else ""
        return f"TunaAPIError status={exc.status_code}{request_id} message={exc}"
    if isinstance(exc, TunaNetworkError):
        cause = getattr(exc, "cause", None)
        if cause is not None:
            return f"TunaNetworkError message={exc} cause={type(cause).__name__}: {cause}"
    return f"{type(exc).__name__}: {exc}"


def _rollouts_batch_with_retry(
    *,
    finetune: Any,
    requests: list[QueryRequest],
    num_rollouts: int,
    max_workers: int,
    retries: int,
    backoff_s: float,
    context: str,
) -> Any:
    worker_count = max(1, min(max_workers, len(requests)))
    for attempt in range(retries + 1):
        try:
            return finetune.rollouts_batch(
                requests=requests,
                num_rollouts=num_rollouts,
                max_workers=worker_count,
            )
        except (TunaAPIError, TunaNetworkError) as exc:
            should_retry = isinstance(exc, TunaNetworkError) or (
                isinstance(exc, TunaAPIError) and exc.status_code == 429
            ) or "too many requests" in str(exc).lower()
            if not should_retry or attempt >= retries:
                print(
                    f"{context}: rollouts_batch failed with no further retries. "
                    f"attempt={attempt + 1}/{retries + 1} workers={worker_count} details={_error_message(exc)}"
                )
                raise
            delay = max(0.1, float(backoff_s)) * (2**attempt)
            next_workers = max(1, worker_count // 2)
            print(
                f"{context}: retrying rollouts_batch attempt={attempt + 1}/{retries + 1} "
                f"workers={worker_count}->{next_workers} sleep={delay:.2f}s details={_error_message(exc)}"
            )
            time.sleep(delay)
            worker_count = next_workers


def _save_checkpoint(*, finetune: Any, context: str) -> Optional[int]:
    try:
        checkpoint = getattr(finetune.save_checkpoint(), "checkpoint", None)
        raw_step = getattr(checkpoint, "step", None)
        return None if raw_step is None else int(raw_step)
    except (TunaAPIError, TunaNetworkError) as exc:
        print(f"{context}: checkpoint save failed; continuing. details={_error_message(exc)}")
        return None


def _build_async_checkpoint_eval_command(
    *,
    args: argparse.Namespace,
    finetune_id: str,
    split_name: str,
    checkpoint_step: int,
    metrics_json_path: Path,
    predictions_jsonl_path: Path,
    task_types: list[str],
    eval_temperature: float,
    eval_top_p: float,
    eval_reasoning: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        str((Path(__file__).resolve().parent / "benchmark_ttt_query.py").resolve()),
        "--env-file",
        str(args.env_file),
        "--base-url",
        str(args.base_url),
        "--dataset-source",
        str(args.dataset_source),
        "--split",
        str(split_name),
        "--finetune-id",
        str(finetune_id),
        "--checkpoint-step",
        str(int(checkpoint_step)),
        "--temperature",
        str(float(eval_temperature)),
        "--top-p",
        str(float(eval_top_p)),
        "--max-tokens",
        str(int(args.max_tokens)),
        "--best-move-optimal-reward",
        str(float(args.best_move_optimal_reward)),
        "--best-move-reward-mode",
        str(args.best_move_reward_mode),
        "--best-move-wrong-rank-scale",
        str(float(args.best_move_wrong_rank_scale)),
        "--output-json",
        str(metrics_json_path),
        "--predictions-jsonl",
        str(predictions_jsonl_path),
        "--checkpoint-fallback-policy",
        "exact",
        "--checkpoint-ready-max-wait-s",
        "300",
        "--checkpoint-ready-poll-interval-s",
        "5",
        "--no-progress",
    ]
    if args.dataset_source == "local_jsonl":
        cmd.extend(["--dataset-dir", str(args.dataset_dir)])
    else:
        cmd.extend(["--hf-dataset-repo-id", str(args.hf_dataset_repo_id)])
        cmd.extend(["--hf-dataset-revision", str(args.hf_dataset_revision)])
        if str(args.hf_cache_dir).strip():
            cmd.extend(["--hf-cache-dir", str(args.hf_cache_dir)])
    if task_types:
        cmd.extend(["--task-types", *list(task_types)])
    cmd.append("--reasoning" if bool(eval_reasoning) else "--no-reasoning")
    return cmd


def _ingest_async_checkpoint_eval_results(
    *,
    args: argparse.Namespace,
    run: Any,
    results: list[CheckpointEvalResult],
    log_step: int,
    best_metric_value: Optional[float],
    best_step: Optional[int],
    best_checkpoint_step: Optional[int],
    latest_checkpoint_step: Optional[int],
) -> tuple[Optional[float], Optional[int], Optional[int], Optional[int], int]:
    success_count = 0
    for result in results:
        source_step = int(result.metadata.get("step_for_log", result.checkpoint_step))
        arrival_step = int(log_step)
        if result.status != "succeeded" or result.metrics_payload is None:
            print(
                f"async checkpoint eval failed step={source_step} "
                f"checkpoint_step={result.checkpoint_step} log={result.stdout_log_path}"
            )
            continue
        success_count += 1
        metrics = dict(result.metrics_payload)
        log_payload = _wandb_eval_metrics(metrics)
        log_payload["async_eval_source_step"] = int(source_step)
        log_payload["async_eval_checkpoint_step"] = int(result.checkpoint_step)
        wandb.log(log_payload, step=arrival_step)
        metric_value = float(metrics.get(args.best_metric, 0.0))
        latest_checkpoint_step = int(result.checkpoint_step)
        run.summary["latest_checkpoint_step"] = int(result.checkpoint_step)
        run.summary["latest_async_eval_metric"] = metric_value
        run.summary["latest_async_eval_step"] = int(source_step)
        if best_metric_value is None or metric_value > best_metric_value:
            best_metric_value = metric_value
            best_step = int(source_step)
            best_checkpoint_step = int(result.checkpoint_step)
            run.summary["best_checkpoint_step"] = int(result.checkpoint_step)
        print(
            f"async checkpoint eval completed step={source_step} checkpoint_step={result.checkpoint_step} "
            f"{args.best_metric}={metric_value:.4f} logged_at_step={arrival_step}"
        )
    return best_metric_value, best_step, best_checkpoint_step, latest_checkpoint_step, success_count


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
    best_move_reward_mode: str,
    best_move_wrong_rank_scale: float,
    show_progress: bool,
    fixed_indices: Optional[list[int]] = None,
) -> dict[str, float]:
    indices = [idx for idx in (fixed_indices or range(len(examples))) if 0 <= idx < len(examples)]
    if fixed_indices is None:
        rng = random.Random(seed)
        rng.shuffle(indices)
    if max_samples is not None:
        indices = indices[: max(0, min(int(max_samples), len(indices)))]

    reward_values: list[float] = []
    object_parse_count = 0
    parse_success_count = 0
    best_move_total = 0
    best_move_set_correct = 0
    best_move_canonical_correct = 0
    best_move_valid_prediction_count = 0
    best_move_center_prediction_count = 0
    best_move_invalid_prediction_count = 0
    non_best_total = 0
    non_best_exact_correct = 0
    per_task_total: Counter[str] = Counter()
    per_task_correct: Counter[str] = Counter()

    def score_batch(batch_examples: list[QAExample]) -> None:
        nonlocal object_parse_count, parse_success_count
        nonlocal best_move_total, best_move_set_correct, best_move_canonical_correct
        nonlocal best_move_valid_prediction_count
        nonlocal best_move_center_prediction_count, best_move_invalid_prediction_count
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
            print(f"eval split={split_name}: skipping batch after error. details={_error_message(exc)}")
            return
        if len(results) != len(active_examples):
            print(
                f"warning: eval split={split_name} got {len(results)} results for {len(active_examples)} requests"
            )
        for example, result in zip(active_examples, results):
            per_task_total[example.task_type] += 1
            rollout = result.rollouts[0] if getattr(result, "rollouts", None) else None
            pred_payload = _parse_prediction_json(_extract_rollout_answer(rollout)) if rollout else None
            pred_move = _move_from_payload(pred_payload) if pred_payload else None
            outcome = (
                _score_payload_for_example(
                    example,
                    pred_payload,
                    best_move_optimal_reward=best_move_optimal_reward,
                    best_move_reward_mode=best_move_reward_mode,
                    best_move_wrong_rank_scale=best_move_wrong_rank_scale,
                )
                if rollout
                else ScoreOutcome(reward=0.0, parse_success=False, task_correct=False)
            )
            reward_values.append(float(outcome.reward))
            if outcome.json_object_parsed:
                object_parse_count += 1
            if outcome.parse_success:
                parse_success_count += 1
            if outcome.task_correct:
                per_task_correct[example.task_type] += 1
            if example.task_type == "best_move":
                best_move_total += 1
                if outcome.best_move_valid_prediction:
                    best_move_valid_prediction_count += 1
                if pred_move == 5:
                    best_move_center_prediction_count += 1
                if pred_move is None:
                    best_move_invalid_prediction_count += 1
                if outcome.best_move_set_correct:
                    best_move_set_correct += 1
                if outcome.best_move_canonical_correct:
                    best_move_canonical_correct += 1
            else:
                non_best_total += 1
                if outcome.exact_non_best_correct:
                    non_best_exact_correct += 1

    batch: list[QAExample] = []
    for index in tqdm(
        indices,
        desc=f"eval:{split_name}",
        total=len(indices),
        dynamic_ncols=True,
        leave=False,
        disable=not show_progress,
    ):
        batch.append(examples[index])
        if len(batch) >= batch_size:
            score_batch(batch)
            batch = []
    if batch:
        score_batch(batch)

    metrics = {
        "eval_samples": float(len(reward_values)),
        "eval_reward_mean": fmean(reward_values) if reward_values else 0.0,
        "eval_json_object_rate": object_parse_count / max(1, len(reward_values)),
        "eval_json_parse_rate": parse_success_count / max(1, len(reward_values)),
        "eval_best_move_set_accuracy": best_move_set_correct / max(1, best_move_total),
        "eval_best_move_canonical_accuracy": best_move_canonical_correct / max(1, best_move_total),
        "eval_best_move_valid_prediction_count": float(best_move_valid_prediction_count),
        "eval_best_move_valid_prediction_rate": best_move_valid_prediction_count / max(1, best_move_total),
        "eval_best_move_center_prediction_rate": best_move_center_prediction_count / max(1, best_move_total),
        "eval_best_move_invalid_prediction_rate": best_move_invalid_prediction_count / max(1, best_move_total),
        "eval_exact_accuracy_non_best_move": non_best_exact_correct / max(1, non_best_total),
    }
    for task in sorted(CANONICAL_TASK_TYPES):
        metrics[f"eval_task_accuracy_{task}"] = per_task_correct[task] / max(1, per_task_total[task])
        metrics[f"eval_task_count_{task}"] = float(per_task_total[task])
    return metrics


def _fixed_eval_indices(
    *,
    split_examples: dict[str, list[QAExample]],
    fixed_subset_size: int,
    fixed_subset_seed: int,
    max_samples: Optional[int],
) -> dict[str, list[int]]:
    if fixed_subset_size <= 0:
        return {}
    out: dict[str, list[int]] = {}
    for split_name, examples in split_examples.items():
        indices = list(range(len(examples)))
        rng = random.Random(int(fixed_subset_seed) + sum((idx + 1) * ord(ch) for idx, ch in enumerate(split_name)))
        rng.shuffle(indices)
        limit = min(len(indices), int(fixed_subset_size))
        if max_samples is not None:
            limit = min(limit, int(max_samples))
        out[split_name] = indices[:limit]
    return out


def _compose_train_groups(
    *,
    on_policy_groups: list[Any],
    replay_groups: list[Any],
    off_policy: bool,
    off_policy_mix_ratio: float,
    off_policy_warmup_steps: int,
    off_policy_min_buffer_groups: int,
    global_step: int,
    rng: random.Random,
) -> tuple[list[Any], int]:
    if (
        not on_policy_groups
        or not off_policy
        or off_policy_mix_ratio <= 0.0
        or global_step < off_policy_warmup_steps
        or len(replay_groups) < off_policy_min_buffer_groups
    ):
        return list(on_policy_groups), 0
    off_policy_count = min(
        max(1, int(round(len(on_policy_groups) * off_policy_mix_ratio))),
        len(on_policy_groups),
        len(replay_groups),
    )
    keep_count = max(0, len(on_policy_groups) - off_policy_count)
    selected_on_policy = (
        list(on_policy_groups)
        if keep_count >= len(on_policy_groups)
        else rng.sample(list(on_policy_groups), k=keep_count)
    )
    mixed = selected_on_policy + rng.sample(list(replay_groups), k=off_policy_count)
    rng.shuffle(mixed)
    return mixed, off_policy_count


def _metric_prefix(split_name: str) -> str:
    return "final_" + "".join(ch if ch.isalnum() else "_" for ch in split_name.strip()) + "_"


def _wandb_eval_metrics(metrics: dict[str, float]) -> dict[str, float]:
    selected = {key: metrics[key] for key in WANDB_EVAL_KEYS if key in metrics}
    for task in sorted(CANONICAL_TASK_TYPES):
        count_key = f"eval_task_count_{task}"
        metric_key = f"eval_task_accuracy_{task}"
        if float(metrics.get(count_key, 0.0)) > 0.0 and metric_key in metrics:
            selected[metric_key] = metrics[metric_key]
    return selected


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TicTacToe QA query RL finetuning example.")
    add = parser.add_argument
    add("--config", default=None, help="Optional JSON config with keys matching CLI argument names.")
    add("--env-file", default=str(DEFAULT_ENV_FILE))
    add("--api-key", default=os.environ.get("MOONDREAM_API_KEY"))
    add("--base-url", default=os.environ.get("TUNA_BASE_URL", DEFAULT_BASE_URL))
    add(
        "--dataset-source",
        choices=sorted(SUPPORTED_DATASET_SOURCES),
        default=DEFAULT_DATASET_SOURCE,
    )
    add("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    add("--hf-dataset-repo-id", default=DEFAULT_HF_DATASET_REPO_ID)
    add("--hf-dataset-revision", default=DEFAULT_HF_DATASET_REVISION)
    add("--hf-token", default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"))
    add("--hf-cache-dir", default="")
    add("--train-split", default="train")
    add("--val-split", default="val")
    add("--final-eval-splits", nargs="+", default=DEFAULT_FINAL_EVAL_SPLITS)
    add("--finetune-id", default="")
    add("--finetune-name", default="")
    add("--rank", type=int, default=16)
    add("--seed", type=int, default=42)
    add("--num-steps", type=int, default=100)
    add("--resume-step", type=int, default=0)
    add("--batch-size", type=int, default=16)
    add("--group-size", type=int, default=4)
    add("--lr", type=float, default=2e-3)
    add("--max-workers", type=int, default=4)
    add("--rollout-retries", type=int, default=2)
    add("--rollout-retry-backoff-s", type=float, default=1.0)
    add("--temperature", type=float, default=1.0)
    add("--top-p", type=float, default=0.9)
    add("--max-tokens", type=int, default=256)
    add(
        "--eval-temperature",
        type=_parse_optional_float_arg,
        default=None,
        help="Eval temperature override. Use inherit/none/null to reuse rollout temperature.",
    )
    add(
        "--eval-top-p",
        type=_parse_optional_float_arg,
        default=None,
        help="Eval top-p override. Use inherit/none/null to reuse rollout top-p.",
    )
    add("--task-sampling-weights-json", default="")
    add("--max-tokens-by-task-json", default="")

    off_policy_group = parser.add_mutually_exclusive_group()
    off_policy_group.add_argument("--off-policy", dest="off_policy", action="store_true")
    off_policy_group.add_argument("--no-off-policy", dest="off_policy", action="store_false")
    parser.set_defaults(off_policy=False)
    add("--off-policy-mix-ratio", type=float, default=0.5)
    add("--off-policy-buffer-size", type=int, default=4096)
    add("--off-policy-warmup-steps", type=int, default=10)
    add("--off-policy-min-buffer-groups", type=int, default=64)

    reasoning_group = parser.add_mutually_exclusive_group()
    reasoning_group.add_argument("--reasoning", dest="reasoning", action="store_true")
    reasoning_group.add_argument("--no-reasoning", dest="reasoning", action="store_false")
    parser.set_defaults(reasoning=False)

    eval_reasoning_group = parser.add_mutually_exclusive_group()
    eval_reasoning_group.add_argument("--eval-reasoning", dest="eval_reasoning", action="store_true")
    eval_reasoning_group.add_argument("--no-eval-reasoning", dest="eval_reasoning", action="store_false")
    eval_reasoning_group.add_argument("--eval-reasoning-inherit", dest="eval_reasoning", action="store_const", const=None)
    parser.set_defaults(eval_reasoning=None)

    add("--eval-every", type=int, default=20)
    add("--save-every", type=int, default=20)
    save_eval_group = parser.add_mutually_exclusive_group()
    save_eval_group.add_argument("--save-on-eval", dest="save_on_eval", action="store_true")
    save_eval_group.add_argument("--no-save-on-eval", dest="save_on_eval", action="store_false")
    parser.set_defaults(save_on_eval=True)
    async_eval_group = parser.add_mutually_exclusive_group()
    async_eval_group.add_argument("--async-checkpoint-eval", dest="async_checkpoint_eval", action="store_true")
    async_eval_group.add_argument("--no-async-checkpoint-eval", dest="async_checkpoint_eval", action="store_false")
    parser.set_defaults(async_checkpoint_eval=False)
    add(
        "--async-checkpoint-eval-dir",
        default=str(Path(__file__).resolve().parent / "outputs" / "async_checkpoint_eval"),
    )
    add("--async-checkpoint-eval-max-inflight", type=int, default=1)
    async_drain_group = parser.add_mutually_exclusive_group()
    async_drain_group.add_argument(
        "--async-checkpoint-eval-drain-on-exit",
        dest="async_checkpoint_eval_drain_on_exit",
        action="store_true",
    )
    async_drain_group.add_argument(
        "--no-async-checkpoint-eval-drain-on-exit",
        dest="async_checkpoint_eval_drain_on_exit",
        action="store_false",
    )
    parser.set_defaults(async_checkpoint_eval_drain_on_exit=True)
    add("--eval-batch-size", type=int, default=32)
    add("--eval-max-samples", type=int, default=1000)
    add("--eval-fixed-subset-size", type=int, default=0)
    add("--eval-fixed-subset-seed", type=int, default=1337)
    add("--best-metric", choices=list(BEST_METRICS), default="eval_reward_mean")
    add("--best-move-optimal-reward", type=float, default=0.7)
    add("--best-move-reward-mode", choices=list(BEST_MOVE_REWARD_MODES), default="ranked")
    add("--best-move-wrong-rank-scale", type=float, default=1.0)
    add("--best-move-center-not-optimal-ratio", type=float, default=0.7)
    add("--intra-task-sampling-json", default="")

    final_eval_group = parser.add_mutually_exclusive_group()
    final_eval_group.add_argument("--skip-final-eval", dest="skip_final_eval", action="store_true")
    final_eval_group.add_argument("--no-skip-final-eval", dest="skip_final_eval", action="store_false")
    parser.set_defaults(skip_final_eval=False)

    add("--wandb-project", default="moondream-ttt-query-rl")
    add("--wandb-run-name", default="")
    add("--no-progress", action="store_true", default=False)
    return parser


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=None)
    pre_args, _ = pre_parser.parse_known_args(argv)
    config_path: Optional[Path] = None
    config: dict[str, Any] = {}
    if pre_args.config:
        config_path = _resolve_path(pre_args.config, default=DEFAULT_CONFIG_PATH)
        config = _normalize_config_defaults(_load_json_config(config_path))

    parser = _build_parser()
    parser_keys = {action.dest for action in parser._actions if action.dest != "help"}
    if config:
        parser.set_defaults(
            **{
                key: value
                for key, value in config.items()
                if key in parser_keys and key not in CONFIG_DERIVED_KEYS
            }
        )
        ignored = sorted(key for key in config if key not in parser_keys and key not in CONFIG_DERIVED_KEYS)
        if ignored:
            print(f"ignoring unknown config keys in {config_path}: {ignored}")

    args = parser.parse_args(argv)
    args.config = str(config_path) if config_path is not None else ""
    args.task_sampling_weights = _resolve_task_sampling_weights(
        dict(config.get("task_sampling_weights", DEFAULT_TASK_SAMPLING_WEIGHTS))
        if isinstance(config.get("task_sampling_weights", DEFAULT_TASK_SAMPLING_WEIGHTS), dict)
        else dict(DEFAULT_TASK_SAMPLING_WEIGHTS),
        args.task_sampling_weights_json,
    )
    args.max_tokens_by_task = _resolve_max_tokens_by_task(
        dict(config.get("max_tokens_by_task", DEFAULT_MAX_TOKENS_BY_TASK))
        if isinstance(config.get("max_tokens_by_task", DEFAULT_MAX_TOKENS_BY_TASK), dict)
        else dict(DEFAULT_MAX_TOKENS_BY_TASK),
        args.max_tokens_by_task_json,
    )
    args.intra_task_sampling = _resolve_intra_task_sampling(
        dict(config.get("intra_task_sampling_json", {}))
        if isinstance(config.get("intra_task_sampling_json", {}), dict)
        else {},
        args.intra_task_sampling_json,
    )
    args.async_checkpoint_eval_dir = str(_resolve_path(args.async_checkpoint_eval_dir))
    return args


def _validate_args(args: argparse.Namespace) -> None:
    args.dataset_source = _normalize_dataset_source(args.dataset_source)
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
    if not 0.0 <= args.temperature <= 2.0:
        raise ValueError("--temperature must be in [0,2]")
    if not 0.0 < args.top_p <= 1.0:
        raise ValueError("--top-p must be in (0,1]")
    if args.eval_temperature is not None and not 0.0 <= float(args.eval_temperature) <= 2.0:
        raise ValueError("--eval-temperature must be in [0,2] when set")
    if args.eval_top_p is not None and not 0.0 < float(args.eval_top_p) <= 1.0:
        raise ValueError("--eval-top-p must be in (0,1] when set")
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be > 0")
    if args.eval_every < 0:
        raise ValueError("--eval-every must be >= 0")
    if args.save_every < 0:
        raise ValueError("--save-every must be >= 0")
    if args.eval_batch_size <= 0:
        raise ValueError("--eval-batch-size must be > 0")
    if args.eval_fixed_subset_size < 0:
        raise ValueError("--eval-fixed-subset-size must be >= 0")
    if args.rollout_retries < 0:
        raise ValueError("--rollout-retries must be >= 0")
    if args.rollout_retry_backoff_s <= 0.0:
        raise ValueError("--rollout-retry-backoff-s must be > 0")
    if not 0.0 <= args.off_policy_mix_ratio <= 1.0:
        raise ValueError("--off-policy-mix-ratio must be in [0,1]")
    if args.off_policy_buffer_size <= 0:
        raise ValueError("--off-policy-buffer-size must be > 0")
    if args.off_policy_warmup_steps < 0:
        raise ValueError("--off-policy-warmup-steps must be >= 0")
    if args.off_policy_min_buffer_groups <= 0:
        raise ValueError("--off-policy-min-buffer-groups must be > 0")
    if args.off_policy_min_buffer_groups > args.off_policy_buffer_size:
        raise ValueError("--off-policy-min-buffer-groups must be <= --off-policy-buffer-size")
    if not 0.0 <= args.best_move_optimal_reward <= 1.0:
        raise ValueError("--best-move-optimal-reward must be in [0,1]")
    if not 0.0 <= args.best_move_wrong_rank_scale <= 1.0:
        raise ValueError("--best-move-wrong-rank-scale must be in [0,1]")
    if not 0.0 <= args.best_move_center_not_optimal_ratio <= 1.0:
        raise ValueError("--best-move-center-not-optimal-ratio must be in [0,1]")
    if args.finetune_id and args.finetune_name:
        raise ValueError("Provide either --finetune-id or --finetune-name, not both")
    if args.dataset_source == "local_jsonl" and not str(args.dataset_dir).strip():
        raise ValueError("--dataset-dir is required when --dataset-source=local_jsonl")
    if args.dataset_source == "hf_hub" and not str(args.hf_dataset_repo_id).strip():
        raise ValueError("--hf-dataset-repo-id is required when --dataset-source=hf_hub")
    if args.async_checkpoint_eval_max_inflight <= 0:
        raise ValueError("--async-checkpoint-eval-max-inflight must be > 0")
    if args.async_checkpoint_eval and not args.save_on_eval:
        raise ValueError("--async-checkpoint-eval requires --save-on-eval")


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    args.env_file = str(_resolve_path(args.env_file, default=DEFAULT_ENV_FILE))
    load_dotenv(args.env_file, override=False)
    if not args.api_key:
        args.api_key = os.environ.get("MOONDREAM_API_KEY", "")
    if not args.base_url:
        args.base_url = os.environ.get("TUNA_BASE_URL", DEFAULT_BASE_URL)
    _validate_args(args)
    if args.off_policy and args.reasoning:
        print("WARNING: off-policy + reasoning is unsafe for this trainer; do not run with both enabled.")
    if not args.api_key:
        raise ValueError("MOONDREAM_API_KEY is required")
    args.hf_token = _resolve_hf_token(args.hf_token)
    if args.eval_max_samples is not None and args.eval_max_samples <= 0:
        args.eval_max_samples = None

    dataset_dir: Optional[Path] = None
    if args.dataset_source == "local_jsonl":
        dataset_dir = Path(args.dataset_dir).expanduser().resolve()
        if not dataset_dir.exists():
            raise FileNotFoundError(f"dataset_dir not found: {dataset_dir}")

    final_eval_splits = []
    seen_splits: set[str] = set()
    for split in args.final_eval_splits:
        name = str(split).strip()
        if name and name not in seen_splits:
            seen_splits.add(name)
            final_eval_splits.append(name)
    if not final_eval_splits:
        raise ValueError("--final-eval-splits must contain at least one split")

    eval_temperature = float(args.temperature) if args.eval_temperature is None else float(args.eval_temperature)
    eval_top_p = float(args.top_p) if args.eval_top_p is None else float(args.eval_top_p)
    eval_reasoning = bool(args.reasoning) if args.eval_reasoning is None else bool(args.eval_reasoning)
    show_progress = (not args.no_progress) and sys.stderr.isatty()

    train_examples = _load_split_examples(
        split_name=args.train_split,
        dataset_source=args.dataset_source,
        dataset_dir=dataset_dir,
        hf_dataset_repo_id=args.hf_dataset_repo_id,
        hf_dataset_revision=args.hf_dataset_revision,
        hf_token=args.hf_token,
        hf_cache_dir=args.hf_cache_dir,
    )
    val_examples = _load_split_examples(
        split_name=args.val_split,
        dataset_source=args.dataset_source,
        dataset_dir=dataset_dir,
        hf_dataset_repo_id=args.hf_dataset_repo_id,
        hf_dataset_revision=args.hf_dataset_revision,
        hf_token=args.hf_token,
        hf_cache_dir=args.hf_cache_dir,
    )

    train_examples_by_task: dict[str, list[QAExample]] = {}
    for example in train_examples:
        train_examples_by_task.setdefault(example.task_type, []).append(example)
    if not train_examples_by_task:
        raise ValueError("train split has no usable examples")

    sampling_tasks = sorted(train_examples_by_task)
    sampling_weights = _task_sampling_weights_for(sampling_tasks, args.task_sampling_weights)
    active_eval_tasks = _active_eval_tasks(args.task_sampling_weights)
    val_before_filter = len(val_examples)
    val_examples = _filter_examples_by_task_weight(val_examples, active_tasks=active_eval_tasks)
    if len(val_examples) != val_before_filter:
        print(
            f"filtered eval split={args.val_split}: kept={len(val_examples)} dropped={val_before_filter - len(val_examples)}"
        )
    print("eval task filter (weight>0): " + ", ".join(sorted(active_eval_tasks)))

    intra_task_sampling_groups = _prepare_intra_task_sampling_groups(
        train_examples_by_task=train_examples_by_task,
        intra_task_sampling=args.intra_task_sampling,
        best_move_center_not_optimal_ratio=float(args.best_move_center_not_optimal_ratio),
    )
    for task_name, strategy in sorted(args.intra_task_sampling.items()):
        rows = train_examples_by_task.get(task_name, [])
        if not rows:
            print(f"intra-task sampling ignored for task={task_name}: no train rows available")
            continue
        group = intra_task_sampling_groups.get(task_name)
        if group is None:
            print(f"intra-task sampling ignored for task={task_name}: strategy={strategy} needs >=2 buckets")
            continue
        bucket_sizes = {key: len(group["buckets"][key]) for key in group["bucket_keys"]}
        print(
            f"intra-task sampling active for task={task_name}: strategy={strategy} "
            f"buckets={bucket_sizes} weights={group.get('bucket_pick_weights', {})}"
        )

    split_examples_cache: dict[str, list[QAExample]] = {args.val_split: val_examples}

    def load_eval_split(split_name: str) -> list[QAExample]:
        if split_name not in split_examples_cache:
            examples = _load_split_examples(
                split_name=split_name,
                dataset_source=args.dataset_source,
                dataset_dir=dataset_dir,
                hf_dataset_repo_id=args.hf_dataset_repo_id,
                hf_dataset_revision=args.hf_dataset_revision,
                hf_token=args.hf_token,
                hf_cache_dir=args.hf_cache_dir,
            )
            filtered = _filter_examples_by_task_weight(examples, active_tasks=active_eval_tasks)
            if len(filtered) != len(examples):
                print(
                    f"filtered eval split={split_name}: kept={len(filtered)} dropped={len(examples) - len(filtered)}"
                )
            split_examples_cache[split_name] = filtered
        return split_examples_cache[split_name]

    final_eval_examples = {split: load_eval_split(split) for split in final_eval_splits}
    fixed_eval_indices_by_split = _fixed_eval_indices(
        split_examples={args.val_split: val_examples, **final_eval_examples},
        fixed_subset_size=int(args.eval_fixed_subset_size),
        fixed_subset_seed=int(args.eval_fixed_subset_seed),
        max_samples=args.eval_max_samples,
    )
    if fixed_eval_indices_by_split:
        print(
            "using fixed eval subsets: "
            + ", ".join(f"{name}={len(indices)}" for name, indices in sorted(fixed_eval_indices_by_split.items()))
        )

    if not args.finetune_id and not args.finetune_name:
        args.finetune_name = f"ttt-query-rl-compact-{_random_suffix()}"

    client = TunaClient(api_key=args.api_key, base_url=args.base_url)
    run = None
    try:
        finetune = client.get_finetune(args.finetune_id) if args.finetune_id else client.create_finetune(name=args.finetune_name, rank=args.rank)
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or None,
            config={
                "config": args.config,
                "env_file": args.env_file,
                "base_url": args.base_url,
                "dataset_source": args.dataset_source,
                "dataset_dir": str(dataset_dir) if dataset_dir is not None else "",
                "hf_dataset_repo_id": args.hf_dataset_repo_id,
                "hf_dataset_revision": args.hf_dataset_revision,
                "hf_cache_dir": args.hf_cache_dir,
                "train_split": args.train_split,
                "val_split": args.val_split,
                "final_eval_splits": final_eval_splits,
                "train_rows": len(train_examples),
                "val_rows": len(val_examples),
                "finetune_id": finetune.finetune_id,
                "finetune_name": finetune.name,
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
                "eval_temperature": eval_temperature,
                "eval_top_p": eval_top_p,
                "eval_reasoning": eval_reasoning,
                "max_tokens": args.max_tokens,
                "max_tokens_by_task": dict(args.max_tokens_by_task),
                "reasoning": args.reasoning,
                "task_sampling_weights": dict(args.task_sampling_weights),
                "intra_task_sampling_json": dict(args.intra_task_sampling),
                "off_policy": args.off_policy,
                "off_policy_mix_ratio": args.off_policy_mix_ratio,
                "off_policy_buffer_size": args.off_policy_buffer_size,
                "off_policy_warmup_steps": args.off_policy_warmup_steps,
                "off_policy_min_buffer_groups": args.off_policy_min_buffer_groups,
                "eval_every": args.eval_every,
                "save_every": args.save_every,
                "save_on_eval": args.save_on_eval,
                "eval_batch_size": args.eval_batch_size,
                "eval_max_samples": args.eval_max_samples,
                "eval_fixed_subset_size": args.eval_fixed_subset_size,
                "eval_fixed_subset_seed": args.eval_fixed_subset_seed,
                "skip_final_eval": args.skip_final_eval,
                "best_metric": args.best_metric,
                "best_move_optimal_reward": args.best_move_optimal_reward,
                "best_move_reward_mode": args.best_move_reward_mode,
                "best_move_wrong_rank_scale": args.best_move_wrong_rank_scale,
                "best_move_center_not_optimal_ratio": args.best_move_center_not_optimal_ratio,
            },
        )
        run.summary["finetune_id"] = finetune.finetune_id

        best_metric_value: Optional[float] = None
        best_step: Optional[int] = None
        best_checkpoint_step: Optional[int] = None
        latest_checkpoint_step: Optional[int] = None
        completed_steps = 0
        replay_buffer: deque[Any] = deque(maxlen=int(args.off_policy_buffer_size))
        rng = random.Random(args.seed)
        async_eval_jobs: list[DispatchHandle] = []
        async_eval_success_count = 0

        if args.eval_every > 0:
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
                temperature=eval_temperature,
                top_p=eval_top_p,
                max_tokens=args.max_tokens,
                max_tokens_by_task=args.max_tokens_by_task,
                reasoning=eval_reasoning,
                best_move_optimal_reward=args.best_move_optimal_reward,
                best_move_reward_mode=args.best_move_reward_mode,
                best_move_wrong_rank_scale=args.best_move_wrong_rank_scale,
                show_progress=show_progress,
                fixed_indices=fixed_eval_indices_by_split.get(args.val_split),
            )
            wandb.log(
                {f"baseline_{key}": value for key, value in _wandb_eval_metrics(baseline_metrics).items()},
                step=args.resume_step,
            )
            best_metric_value = float(baseline_metrics.get(args.best_metric, 0.0))
            best_step = args.resume_step
            print(
                f"baseline eval step={args.resume_step} reward={baseline_metrics['eval_reward_mean']:.4f} "
                f"obj_parse={baseline_metrics['eval_json_object_rate']:.4f} "
                f"parse={baseline_metrics['eval_json_parse_rate']:.4f} "
                f"best_move_valid={baseline_metrics.get('eval_best_move_valid_prediction_count', 0.0):.0f}/"
                f"{baseline_metrics.get('eval_task_count_best_move', 0.0):.0f} "
                f"({baseline_metrics.get('eval_best_move_valid_prediction_rate', 0.0):.4f})"
            )
            if args.save_on_eval:
                baseline_saved_step = _save_checkpoint(
                    finetune=finetune,
                    context=f"baseline eval step={args.resume_step}",
                )
                if baseline_saved_step is not None:
                    latest_checkpoint_step = int(baseline_saved_step)
                    print("baseline checkpoint saved (save_on_eval=true)")

        step_iter = tqdm(
            range(args.num_steps),
            desc="train",
            total=args.num_steps,
            dynamic_ncols=True,
            disable=not show_progress,
        )
        for step in step_iter:
            global_step = args.resume_step + step
            completed_steps = step + 1
            if async_eval_jobs:
                async_eval_jobs, completed_async_results = poll_checkpoint_eval_jobs(async_eval_jobs)
                (
                    best_metric_value,
                    best_step,
                    best_checkpoint_step,
                    latest_checkpoint_step,
                    completed_successes,
                ) = _ingest_async_checkpoint_eval_results(
                    args=args,
                    run=run,
                    results=completed_async_results,
                    log_step=int(global_step),
                    best_metric_value=best_metric_value,
                    best_step=best_step,
                    best_checkpoint_step=best_checkpoint_step,
                    latest_checkpoint_step=latest_checkpoint_step,
                )
                async_eval_success_count += int(completed_successes)
            sampled_tasks = rng.choices(sampling_tasks, weights=sampling_weights, k=args.batch_size)
            batch = [
                _sample_training_example(
                    task_name=task,
                    train_examples_by_task=train_examples_by_task,
                    intra_task_sampling_groups=intra_task_sampling_groups,
                    rng=rng,
                )
                for task in sampled_tasks
            ]
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
                print(f"step {global_step}: rollouts_batch failed; skipping. details={_error_message(exc)}")
                continue
            if len(results) != len(active_examples):
                print(
                    f"warning: step {global_step} got {len(results)} rollout results for {len(active_examples)} requests"
                )
            on_policy_groups: list[Any] = []
            rewards_all: list[float] = []
            object_parses = 0
            parse_successes = 0
            best_move_rollout_count = 0
            best_move_valid_prediction_count = 0
            for example, result in zip(active_examples, results):
                rollouts = list(getattr(result, "rollouts", []) or [])
                if not rollouts:
                    continue
                rewards: list[float] = []
                for rollout in rollouts:
                    outcome = _score_rollout_for_example(
                        rollout,
                        example,
                        best_move_optimal_reward=args.best_move_optimal_reward,
                        best_move_reward_mode=args.best_move_reward_mode,
                        best_move_wrong_rank_scale=args.best_move_wrong_rank_scale,
                    )
                    rewards.append(float(outcome.reward))
                    rewards_all.append(float(outcome.reward))
                    if outcome.json_object_parsed:
                        object_parses += 1
                    if outcome.parse_success:
                        parse_successes += 1
                    if example.task_type == "best_move":
                        best_move_rollout_count += 1
                        if outcome.best_move_valid_prediction:
                            best_move_valid_prediction_count += 1
                if rewards:
                    on_policy_groups.append(result.to_group(rewards=rewards))
            if not on_policy_groups:
                print(f"step {global_step}: no train groups produced; skipping")
                continue

            train_groups, off_policy_count = _compose_train_groups(
                on_policy_groups=on_policy_groups,
                replay_groups=list(replay_buffer),
                off_policy=bool(args.off_policy),
                off_policy_mix_ratio=float(args.off_policy_mix_ratio),
                off_policy_warmup_steps=int(args.off_policy_warmup_steps),
                off_policy_min_buffer_groups=int(args.off_policy_min_buffer_groups),
                global_step=int(global_step),
                rng=rng,
            )
            replay_buffer.extend(on_policy_groups)
            if not train_groups:
                print(f"step {global_step}: no train groups selected; skipping")
                continue
            try:
                train_out = finetune.train_step(groups=train_groups, lr=args.lr)
            except (TunaAPIError, TunaNetworkError) as exc:
                print(f"step {global_step}: train_step failed; skipping. details={_error_message(exc)}")
                continue

            reward_mean = fmean(rewards_all) if rewards_all else 0.0
            reward_var = pvariance(rewards_all) if len(rewards_all) > 1 else 0.0
            train_metrics = {
                "reward_mean": reward_mean,
                "reward_var": reward_var,
                "train_json_object_rate": object_parses / max(1, len(rewards_all)),
                "train_json_parse_rate": parse_successes / max(1, len(rewards_all)),
                "train_best_move_valid_prediction_count": float(best_move_valid_prediction_count),
                "train_best_move_valid_prediction_rate": (
                    best_move_valid_prediction_count / max(1, best_move_rollout_count)
                ),
                "accepted_groups": float(len(train_groups)),
                "on_policy_groups": float(len(train_groups) - off_policy_count),
                "off_policy_groups": float(off_policy_count),
                "off_policy_group_fraction": off_policy_count / max(1, len(train_groups)),
                "replay_buffer_size": float(len(replay_buffer)),
                "kl": float(getattr(train_out, "kl", 0.0) or 0.0),
                "router_kl": float(getattr(train_out, "router_kl", 0.0) or 0.0),
                "grad_norm": float(getattr(train_out, "grad_norm", 0.0) or 0.0),
            }
            wandb.log({key: train_metrics[key] for key in WANDB_TRAIN_KEYS if key in train_metrics}, step=global_step)
            print(
                f"step {global_step} reward={reward_mean:.4f} "
                f"obj_parse_rate={train_metrics['train_json_object_rate']:.4f} "
                f"parse_rate={train_metrics['train_json_parse_rate']:.4f} "
                f"best_move_valid={best_move_valid_prediction_count}/{best_move_rollout_count} "
                f"({train_metrics['train_best_move_valid_prediction_rate']:.4f}) "
                f"offp={off_policy_count}/{len(train_groups)} replay={len(replay_buffer)} "
                f"kl={train_metrics['kl']:.4f}"
            )
            if show_progress:
                step_iter.set_postfix(
                    reward=f"{reward_mean:.3f}",
                    parse=f"{train_metrics['train_json_parse_rate']:.3f}",
                    kl=f"{train_metrics['kl']:.3f}",
                )

            if args.eval_every > 0 and (global_step + 1) % args.eval_every == 0:
                if args.async_checkpoint_eval:
                    saved_step = _save_checkpoint(
                        finetune=finetune,
                        context=f"periodic eval step={global_step}",
                    )
                    if saved_step is not None:
                        latest_checkpoint_step = int(saved_step)
                        job = dispatch_checkpoint_eval(
                            trainer="ttt_query_rl_compact",
                            finetune_id=str(finetune.finetune_id),
                            checkpoint_step=int(saved_step),
                            selection_metric=str(args.best_metric),
                            base_dir=str(args.async_checkpoint_eval_dir),
                            command_builder=lambda metrics_json_path, predictions_jsonl_path, _stdout_log_path: _build_async_checkpoint_eval_command(
                                args=args,
                                finetune_id=str(finetune.finetune_id),
                                split_name=str(args.val_split),
                                checkpoint_step=int(saved_step),
                                metrics_json_path=metrics_json_path,
                                predictions_jsonl_path=predictions_jsonl_path,
                                task_types=sorted(active_eval_tasks),
                                eval_temperature=float(eval_temperature),
                                eval_top_p=float(eval_top_p),
                                eval_reasoning=bool(eval_reasoning),
                            ),
                            metadata={
                                "step_for_log": int(global_step),
                                "split_name": str(args.val_split),
                            },
                            env_overrides={
                                "MOONDREAM_API_KEY": str(args.api_key),
                                "HF_TOKEN": str(args.hf_token),
                            },
                            max_inflight=int(args.async_checkpoint_eval_max_inflight),
                            inflight_jobs=async_eval_jobs,
                        )
                        if job is None:
                            print(
                                f"async checkpoint eval skipped step={global_step} checkpoint_step={saved_step} "
                                f"reason=max_inflight"
                            )
                        else:
                            async_eval_jobs.append(job)
                            print(
                                f"async checkpoint eval dispatched step={global_step} checkpoint_step={saved_step} "
                                f"job_dir={job.job_dir}"
                            )
                else:
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
                        temperature=eval_temperature,
                        top_p=eval_top_p,
                        max_tokens=args.max_tokens,
                        max_tokens_by_task=args.max_tokens_by_task,
                        reasoning=eval_reasoning,
                        best_move_optimal_reward=args.best_move_optimal_reward,
                        best_move_reward_mode=args.best_move_reward_mode,
                        best_move_wrong_rank_scale=args.best_move_wrong_rank_scale,
                        show_progress=show_progress,
                        fixed_indices=fixed_eval_indices_by_split.get(args.val_split),
                    )
                    wandb.log(_wandb_eval_metrics(eval_metrics), step=global_step)
                    metric_value = float(eval_metrics.get(args.best_metric, 0.0))
                    print(
                        f"eval step {global_step} reward={eval_metrics['eval_reward_mean']:.4f} "
                        f"obj_parse={eval_metrics['eval_json_object_rate']:.4f} "
                        f"parse={eval_metrics['eval_json_parse_rate']:.4f} "
                        f"best_move_valid={eval_metrics.get('eval_best_move_valid_prediction_count', 0.0):.0f}/"
                        f"{eval_metrics.get('eval_task_count_best_move', 0.0):.0f} "
                        f"({eval_metrics.get('eval_best_move_valid_prediction_rate', 0.0):.4f})"
                    )
                    if best_metric_value is None or metric_value > best_metric_value:
                        best_metric_value = metric_value
                        best_step = global_step
                        if args.save_on_eval:
                            print(f"new best {args.best_metric}={metric_value:.4f} at step {global_step}")
                        else:
                            best_saved_step = _save_checkpoint(
                                finetune=finetune,
                                context=f"best metric checkpoint step={global_step}",
                            )
                            if best_saved_step is not None:
                                best_checkpoint_step = int(best_saved_step)
                                latest_checkpoint_step = int(best_saved_step)
                                print(
                                    f"new best {args.best_metric}={metric_value:.4f} at step {global_step}; checkpoint saved"
                                )
                    if args.save_on_eval:
                        saved_step = _save_checkpoint(
                            finetune=finetune,
                            context=f"periodic eval step={global_step}",
                        )
                        if saved_step is not None:
                            latest_checkpoint_step = int(saved_step)
                            print(f"checkpoint saved at eval step={global_step} (save_on_eval=true)")

            if args.save_every > 0 and (global_step + 1) % args.save_every == 0:
                saved_step = _save_checkpoint(finetune=finetune, context=f"save_every checkpoint step={global_step}")
                if saved_step is not None:
                    latest_checkpoint_step = int(saved_step)

        final_saved_step = _save_checkpoint(finetune=finetune, context="final checkpoint save")
        if final_saved_step is not None:
            latest_checkpoint_step = int(final_saved_step)
        if args.async_checkpoint_eval and bool(args.async_checkpoint_eval_drain_on_exit):
            completed_async_results = drain_checkpoint_eval_jobs(async_eval_jobs)
            (
                best_metric_value,
                best_step,
                best_checkpoint_step,
                latest_checkpoint_step,
                completed_successes,
            ) = _ingest_async_checkpoint_eval_results(
                args=args,
                run=run,
                results=completed_async_results,
                log_step=int(args.resume_step + completed_steps),
                best_metric_value=best_metric_value,
                best_step=best_step,
                best_checkpoint_step=best_checkpoint_step,
                latest_checkpoint_step=latest_checkpoint_step,
            )
            async_eval_success_count += int(completed_successes)
        final_eval_step = args.resume_step + int(completed_steps)
        if args.skip_final_eval:
            print("skip_final_eval=true; skipping final split eval pass.")
        else:
            for offset, (split_name, examples) in enumerate(final_eval_examples.items()):
                eval_metrics = _evaluate_split(
                    finetune=finetune,
                    examples=examples,
                    split_name=split_name,
                    seed=args.seed + 5000 + offset,
                    batch_size=args.eval_batch_size,
                    max_workers=args.max_workers,
                    max_samples=args.eval_max_samples,
                    rollout_retries=args.rollout_retries,
                    rollout_retry_backoff_s=args.rollout_retry_backoff_s,
                    temperature=eval_temperature,
                    top_p=eval_top_p,
                    max_tokens=args.max_tokens,
                    max_tokens_by_task=args.max_tokens_by_task,
                    reasoning=eval_reasoning,
                    best_move_optimal_reward=args.best_move_optimal_reward,
                    best_move_reward_mode=args.best_move_reward_mode,
                    best_move_wrong_rank_scale=args.best_move_wrong_rank_scale,
                    show_progress=show_progress,
                    fixed_indices=fixed_eval_indices_by_split.get(split_name),
                )
                wandb.log(
                    {f"{_metric_prefix(split_name)}{key}": value for key, value in _wandb_eval_metrics(eval_metrics).items()},
                    step=final_eval_step,
                )
                print(
                    f"final eval split={split_name} reward={eval_metrics['eval_reward_mean']:.4f} "
                    f"obj_parse={eval_metrics['eval_json_object_rate']:.4f} "
                    f"parse={eval_metrics['eval_json_parse_rate']:.4f} "
                    f"best_move_valid={eval_metrics.get('eval_best_move_valid_prediction_count', 0.0):.0f}/"
                    f"{eval_metrics.get('eval_task_count_best_move', 0.0):.0f} "
                    f"({eval_metrics.get('eval_best_move_valid_prediction_rate', 0.0):.4f})"
                )

        run.summary["best_metric_name"] = args.best_metric
        run.summary["best_metric_value"] = float(best_metric_value or 0.0)
        run.summary["best_metric_step"] = int(best_step if best_step is not None else -1)
        run.summary["best_checkpoint_step"] = int(best_checkpoint_step if best_checkpoint_step is not None else -1)
        run.summary["latest_checkpoint_step"] = int(latest_checkpoint_step if latest_checkpoint_step is not None else -1)
        run.summary["finetune_id"] = finetune.finetune_id
        run.summary["train_rows"] = len(train_examples)
        run.summary["val_rows"] = len(val_examples)
        run.summary["completed_steps"] = int(completed_steps)
        run.summary["target_steps"] = int(args.num_steps)
        run.summary["save_on_eval"] = bool(args.save_on_eval)
        run.summary["async_checkpoint_eval_enabled"] = bool(args.async_checkpoint_eval)
        run.summary["async_checkpoint_eval_success_count"] = int(async_eval_success_count)
        run.summary["skip_final_eval"] = bool(args.skip_final_eval)
        print(
            f"done. finetune_id={finetune.finetune_id} "
            f"best_{args.best_metric}={best_metric_value} best_step={best_step} completed_steps={completed_steps}"
        )
    finally:
        if run is not None:
            run.finish()
        client.close()


if __name__ == "__main__":
    main()
