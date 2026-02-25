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
from collections import Counter, deque
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

from tictaktoe_QA import data_loader as dataset_loader  # noqa: E402
from tuna_sdk import QueryOutput, QueryRequest, QuerySettings, TunaClient  # noqa: E402
from tuna_sdk.errors import TunaAPIError, TunaNetworkError  # noqa: E402

DEFAULT_BASE_URL = "https://api.moondream.ai/v1"
DEFAULT_DATASET_SOURCE = dataset_loader.DEFAULT_DATASET_SOURCE
DEFAULT_HF_DATASET_REPO_ID = dataset_loader.DEFAULT_HF_DATASET_REPO_ID
DEFAULT_HF_DATASET_REVISION = dataset_loader.DEFAULT_HF_DATASET_REVISION
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
MAIN_TRAIN_WANDB_METRIC_KEYS = (
    "reward_mean",
    "train_json_object_rate",
    "train_json_parse_rate",
    "off_policy_group_fraction",
    "replay_buffer_size",
    "kl",
)
MAIN_EVAL_WANDB_METRIC_KEYS = (
    "eval_samples",
    "eval_reward_mean",
    "eval_json_object_rate",
    "eval_json_parse_rate",
    "eval_best_move_set_accuracy",
    "eval_best_move_canonical_accuracy",
    "eval_exact_accuracy_non_best_move",
)

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

TRAIN_CONFIG_ALLOWED_KEYS = {
    "api_key",
    "base_url",
    "batch_size",
    "best_metric",
    "best_move_optimal_reward",
    "checkpoint_avg_metric",
    "checkpoint_avg_splits",
    "checkpoint_ranking_output",
    "dataset_dir",
    "dataset_source",
    "early_stop",
    "early_stop_mode",
    "env_file",
    "eval_batch_size",
    "eval_every",
    "eval_fixed_subset_seed",
    "eval_fixed_subset_size",
    "eval_max_samples",
    "eval_reasoning",
    "eval_temperature",
    "eval_top_p",
    "final_eval_splits",
    "finetune_id",
    "finetune_name",
    "group_size",
    "hf_cache_dir",
    "hf_dataset_repo_id",
    "hf_dataset_revision",
    "hf_token",
    "lr",
    "max_tokens",
    "max_tokens_by_task",
    "max_workers",
    "no_progress",
    "num_steps",
    "off_policy",
    "off_policy_buffer_size",
    "off_policy_min_buffer_groups",
    "off_policy_mix_ratio",
    "off_policy_warmup_steps",
    "rank",
    "reasoning",
    "resume_step",
    "rollout_retries",
    "rollout_retry_backoff_s",
    "save_every",
    "skip_final_eval",
    "save_on_eval",
    "seed",
    "task_sampling_weights",
    "temperature",
    "top_p",
    "train_split",
    "val_split",
    "wandb_project",
    "wandb_run_name",
}

EARLY_STOP_MODES = ("conservative", "balanced", "aggressive")


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


def _validate_config_keys(config: dict[str, Any], *, config_path: Path) -> None:
    unknown = sorted(key for key in config.keys() if key not in TRAIN_CONFIG_ALLOWED_KEYS)
    if unknown:
        raise ValueError(
            f"Unknown config key(s) in {config_path}: {unknown}. "
            "Remove typos or update script support."
        )


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


def _cfg_optional_float(config: dict[str, Any], key: str) -> Optional[float]:
    value = config.get(key)
    if value is None:
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"", "none", "null", "inherit"}:
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _cfg_optional_bool(config: dict[str, Any], key: str) -> Optional[bool]:
    value = config.get(key)
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"", "none", "null", "inherit"}:
            return None
        if lowered in {"true", "1", "yes", "y", "on"}:
            return True
        if lowered in {"false", "0", "no", "n", "off"}:
            return False
    if isinstance(value, (int, float)):
        if value == 0:
            return False
        if value == 1:
            return True
    return None


def _parse_optional_float_arg(raw_value: str) -> Optional[float]:
    text = str(raw_value or "").strip().lower()
    if text in {"", "none", "null", "inherit"}:
        return None
    try:
        return float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected float or inherit/none/null, got: {raw_value!r}") from exc


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
        "--dataset-source",
        choices=sorted(dataset_loader.SUPPORTED_DATASET_SOURCES),
        default=_cfg_str(config, "dataset_source", DEFAULT_DATASET_SOURCE),
        help="Dataset source: HF Hub or local JSONL directory.",
    )
    parser.add_argument(
        "--dataset-dir",
        default=_cfg_str(config, "dataset_dir", str(_repo_relative("synth_dataset/outputs/v1"))),
        help="Local dataset dir (required when --dataset-source=local_jsonl).",
    )
    parser.add_argument(
        "--hf-dataset-repo-id",
        default=_cfg_str(config, "hf_dataset_repo_id", DEFAULT_HF_DATASET_REPO_ID),
    )
    parser.add_argument(
        "--hf-dataset-revision",
        default=_cfg_str(config, "hf_dataset_revision", DEFAULT_HF_DATASET_REVISION),
    )
    parser.add_argument("--hf-token", default=_cfg_str(config, "hf_token", ""))
    parser.add_argument("--hf-cache-dir", default=_cfg_str(config, "hf_cache_dir", ""))
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
        "--eval-temperature",
        type=_parse_optional_float_arg,
        default=_cfg_optional_float(config, "eval_temperature"),
        help="Eval temperature override. Use inherit/none/null to reuse rollout temperature.",
    )
    parser.add_argument(
        "--eval-top-p",
        type=_parse_optional_float_arg,
        default=_cfg_optional_float(config, "eval_top_p"),
        help="Eval top-p override. Use inherit/none/null to reuse rollout top-p.",
    )
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

    off_policy_group = parser.add_mutually_exclusive_group()
    off_policy_group.add_argument(
        "--off-policy",
        dest="off_policy",
        action="store_true",
        help="Enable off-policy replay mixing in train_step updates.",
    )
    off_policy_group.add_argument(
        "--no-off-policy",
        dest="off_policy",
        action="store_false",
        help="Disable off-policy replay mixing in train_step updates.",
    )
    parser.set_defaults(off_policy=_cfg_bool(config, "off_policy", False))
    parser.add_argument(
        "--off-policy-mix-ratio",
        type=float,
        default=_cfg_float(config, "off_policy_mix_ratio", 0.5),
        help="Fraction of train groups sourced from replay when off-policy is active.",
    )
    parser.add_argument(
        "--off-policy-buffer-size",
        type=int,
        default=_cfg_int(config, "off_policy_buffer_size", 4096),
        help="Max number of train groups stored in off-policy replay buffer.",
    )
    parser.add_argument(
        "--off-policy-warmup-steps",
        type=int,
        default=_cfg_int(config, "off_policy_warmup_steps", 10),
        help="Minimum number of train steps before off-policy replay can be used.",
    )
    parser.add_argument(
        "--off-policy-min-buffer-groups",
        type=int,
        default=_cfg_int(config, "off_policy_min_buffer_groups", 64),
        help="Minimum replay groups required before off-policy replay is used.",
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
    eval_reasoning_group = parser.add_mutually_exclusive_group()
    eval_reasoning_group.add_argument(
        "--eval-reasoning",
        dest="eval_reasoning",
        action="store_true",
        help="Force reasoning=true for eval requests.",
    )
    eval_reasoning_group.add_argument(
        "--no-eval-reasoning",
        dest="eval_reasoning",
        action="store_false",
        help="Force reasoning=false for eval requests.",
    )
    eval_reasoning_group.add_argument(
        "--eval-reasoning-inherit",
        dest="eval_reasoning",
        action="store_const",
        const=None,
        help="Inherit eval reasoning from rollout --reasoning.",
    )
    parser.set_defaults(eval_reasoning=_cfg_optional_bool(config, "eval_reasoning"))

    parser.add_argument("--eval-every", type=int, default=_cfg_int(config, "eval_every", 20))
    parser.add_argument("--save-every", type=int, default=_cfg_int(config, "save_every", 20))
    save_on_eval_group = parser.add_mutually_exclusive_group()
    save_on_eval_group.add_argument(
        "--save-on-eval",
        dest="save_on_eval",
        action="store_true",
        help="Save a checkpoint at each periodic evaluation.",
    )
    save_on_eval_group.add_argument(
        "--no-save-on-eval",
        dest="save_on_eval",
        action="store_false",
        help="Disable automatic checkpoint save at periodic evaluations.",
    )
    parser.set_defaults(save_on_eval=_cfg_bool(config, "save_on_eval", True))
    parser.add_argument("--eval-batch-size", type=int, default=_cfg_int(config, "eval_batch_size", 32))
    parser.add_argument(
        "--eval-max-samples",
        type=int,
        default=_cfg_int(config, "eval_max_samples", 1000),
        help="Max samples per eval split. <=0 means full split.",
    )
    parser.add_argument(
        "--eval-fixed-subset-size",
        type=int,
        default=_cfg_int(config, "eval_fixed_subset_size", 0),
        help="Fixed deterministic eval subset size per split. <=0 disables fixed subsets.",
    )
    parser.add_argument(
        "--eval-fixed-subset-seed",
        type=int,
        default=_cfg_int(config, "eval_fixed_subset_seed", 1337),
        help="Seed used for deterministic eval fixed subset index generation.",
    )
    parser.add_argument(
        "--checkpoint-avg-splits",
        nargs="+",
        default=_cfg_list_str(config, "checkpoint_avg_splits", ["val", "test"]),
        help="Splits used for periodic checkpoint average ranking.",
    )
    parser.add_argument(
        "--checkpoint-avg-metric",
        choices=["eval_reward_mean"],
        default=_cfg_str(config, "checkpoint_avg_metric", "eval_reward_mean"),
    )
    parser.add_argument(
        "--checkpoint-ranking-output",
        default=_cfg_str(config, "checkpoint_ranking_output", ""),
        help="Path to write periodic checkpoint ranking JSON.",
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
    early_stop_group = parser.add_mutually_exclusive_group()
    early_stop_group.add_argument(
        "--early-stop",
        dest="early_stop",
        action="store_true",
        help="Enable periodic eval-based early termination rules.",
    )
    early_stop_group.add_argument(
        "--no-early-stop",
        dest="early_stop",
        action="store_false",
        help="Disable eval-based early termination rules.",
    )
    parser.set_defaults(early_stop=_cfg_bool(config, "early_stop", False))
    parser.add_argument(
        "--early-stop-mode",
        choices=list(EARLY_STOP_MODES),
        default=_cfg_str(config, "early_stop_mode", "balanced"),
        help="Threshold profile for early-stop collapse/plateau checks.",
    )
    skip_final_eval_group = parser.add_mutually_exclusive_group()
    skip_final_eval_group.add_argument(
        "--skip-final-eval",
        dest="skip_final_eval",
        action="store_true",
        help="Skip final post-training split evals (useful for sweep screening).",
    )
    skip_final_eval_group.add_argument(
        "--no-skip-final-eval",
        dest="skip_final_eval",
        action="store_false",
        help="Run final post-training split evals.",
    )
    parser.set_defaults(skip_final_eval=_cfg_bool(config, "skip_final_eval", False))

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
    _validate_config_keys(config, config_path=config_path)
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


def _resolve_output_path(raw_path: str, *, fallback_name: str) -> Path:
    text = str(raw_path or "").strip()
    if text:
        path = Path(text).expanduser()
        if path.is_absolute():
            return path
        return (Path.cwd() / path).resolve()
    return _repo_relative("outputs", fallback_name).resolve()


def _validate_args(args: argparse.Namespace) -> None:
    args.dataset_source = dataset_loader.normalize_dataset_source(args.dataset_source)
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
    if args.eval_temperature is not None and not (0.0 <= float(args.eval_temperature) <= 2.0):
        raise ValueError("--eval-temperature must be in [0,2] when set")
    if args.eval_top_p is not None and not (0.0 < float(args.eval_top_p) <= 1.0):
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
    if not (0.0 <= args.off_policy_mix_ratio <= 1.0):
        raise ValueError("--off-policy-mix-ratio must be in [0,1]")
    if args.off_policy_buffer_size <= 0:
        raise ValueError("--off-policy-buffer-size must be > 0")
    if args.off_policy_warmup_steps < 0:
        raise ValueError("--off-policy-warmup-steps must be >= 0")
    if args.off_policy_min_buffer_groups <= 0:
        raise ValueError("--off-policy-min-buffer-groups must be > 0")
    if args.off_policy_min_buffer_groups > args.off_policy_buffer_size:
        raise ValueError(
            "--off-policy-min-buffer-groups must be <= --off-policy-buffer-size"
        )
    if not (0.0 <= args.best_move_optimal_reward <= 1.0):
        raise ValueError("--best-move-optimal-reward must be in [0,1]")
    if args.early_stop_mode not in EARLY_STOP_MODES:
        raise ValueError(f"--early-stop-mode must be one of {list(EARLY_STOP_MODES)}")
    if not args.checkpoint_avg_splits:
        raise ValueError("--checkpoint-avg-splits must contain at least one split")

    if args.finetune_id and args.finetune_name:
        raise ValueError("Provide either --finetune-id or --finetune-name, not both")
    if args.dataset_source == "local_jsonl" and not str(args.dataset_dir).strip():
        raise ValueError("--dataset-dir is required when --dataset-source=local_jsonl")
    if args.dataset_source == "hf_hub" and not str(args.hf_dataset_repo_id).strip():
        raise ValueError("--hf-dataset-repo-id is required when --dataset-source=hf_hub")



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


def _resolve_image_path(row: dict[str, Any], dataset_dir: Optional[Path]) -> Optional[Path]:
    candidates: list[str] = []
    for key in ("image_path", "image"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip())

    for raw in candidates:
        path = Path(raw).expanduser()
        if path.is_file():
            return path.resolve()

        if not path.is_absolute() and dataset_dir is not None:
            joined = (dataset_dir / path).resolve()
            if joined.is_file():
                return joined

        basename = path.name
        if basename and dataset_dir is not None:
            fallback = (dataset_dir / "images" / basename).resolve()
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
    rows = dataset_loader.load_split_rows(
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
    fixed_indices: Optional[list[int]] = None,
) -> dict[str, float]:
    if fixed_indices is not None:
        indices = [idx for idx in fixed_indices if 0 <= idx < len(examples)]
    else:
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


def _sanitize_split_name(split_name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in split_name.strip())


def _numeric_metrics_only(metrics: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            out[key] = float(value)
    return out


def _split_seed_offset(split_name: str) -> int:
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(str(split_name)))


def _build_fixed_eval_indices(
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
        if not examples:
            out[split_name] = []
            continue
        indices = list(range(len(examples)))
        rng = random.Random(int(fixed_subset_seed) + _split_seed_offset(split_name))
        rng.shuffle(indices)
        limit = min(len(indices), int(fixed_subset_size))
        if max_samples is not None:
            limit = min(limit, int(max_samples))
        out[split_name] = indices[:limit]
    return out


def _early_stop_thresholds(mode: str) -> dict[str, float]:
    normalized = str(mode).strip().lower()
    if normalized == "conservative":
        return {
            "parse_floor": 0.50,
            "parse_streak": 3.0,
            "reward_drop": 0.30,
            "reward_drop_streak": 3.0,
            "plateau_window": 5.0,
            "plateau_min_improvement": 0.005,
        }
    if normalized == "aggressive":
        return {
            "parse_floor": 0.70,
            "parse_streak": 2.0,
            "reward_drop": 0.15,
            "reward_drop_streak": 2.0,
            "plateau_window": 3.0,
            "plateau_min_improvement": 0.02,
        }
    return {
        "parse_floor": 0.60,
        "parse_streak": 2.0,
        "reward_drop": 0.20,
        "reward_drop_streak": 2.0,
        "plateau_window": 4.0,
        "plateau_min_improvement": 0.01,
    }


def _should_early_stop(
    *,
    mode: str,
    parse_streak: int,
    reward_drop_streak: int,
    recent_eval_rewards: list[float],
) -> tuple[bool, str, bool]:
    thresholds = _early_stop_thresholds(mode)
    if parse_streak >= int(thresholds["parse_streak"]):
        return True, "collapse_parse_floor", True
    if reward_drop_streak >= int(thresholds["reward_drop_streak"]):
        return True, "collapse_reward_drop", True

    plateau_window = int(thresholds["plateau_window"])
    if plateau_window > 0 and len(recent_eval_rewards) >= plateau_window:
        window = recent_eval_rewards[-plateau_window:]
        if window and (max(window) - min(window)) < float(thresholds["plateau_min_improvement"]):
            return True, "plateau_no_improvement", False

    return False, "", False


def _default_training_status(*, target_steps: int, early_stop_mode: str) -> dict[str, Any]:
    return {
        "stopped_early": False,
        "stop_reason": "",
        "completed_steps": 0,
        "target_steps": int(target_steps),
        "early_stop_mode": str(early_stop_mode),
        "collapse_detected": False,
    }


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
    if not on_policy_groups:
        return [], 0
    if not off_policy:
        return list(on_policy_groups), 0
    if off_policy_mix_ratio <= 0.0:
        return list(on_policy_groups), 0
    if global_step < off_policy_warmup_steps:
        return list(on_policy_groups), 0
    if len(replay_groups) < off_policy_min_buffer_groups:
        return list(on_policy_groups), 0

    desired_off_policy = int(round(len(on_policy_groups) * off_policy_mix_ratio))
    if desired_off_policy <= 0:
        desired_off_policy = 1
    desired_off_policy = min(
        desired_off_policy,
        len(on_policy_groups),
        len(replay_groups),
    )
    if desired_off_policy <= 0:
        return list(on_policy_groups), 0

    keep_on_policy = len(on_policy_groups) - desired_off_policy
    if keep_on_policy <= 0:
        selected_on_policy: list[Any] = []
    elif keep_on_policy >= len(on_policy_groups):
        selected_on_policy = list(on_policy_groups)
    else:
        selected_on_policy = rng.sample(list(on_policy_groups), k=keep_on_policy)

    selected_off_policy = rng.sample(list(replay_groups), k=desired_off_policy)
    mixed = selected_on_policy + selected_off_policy
    rng.shuffle(mixed)
    return mixed, desired_off_policy


def _rank_checkpoint_eval_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    saved_entries = [item for item in history if bool(item.get("checkpoint_saved"))]
    candidates = saved_entries if saved_entries else history
    return sorted(
        candidates,
        key=lambda item: float(item.get("avg_eval_reward_mean", 0.0)),
        reverse=True,
    )


def _build_checkpoint_ranking_payload(
    *,
    finetune_id: str,
    checkpoint_avg_metric: str,
    checkpoint_avg_splits: list[str],
    checkpoint_eval_history: list[dict[str, Any]],
    training_status: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    ranking = _rank_checkpoint_eval_history(checkpoint_eval_history)
    best_avg_eval_reward = float(ranking[0]["avg_eval_reward_mean"]) if ranking else 0.0
    best_avg_eval_reward_step = int(ranking[0]["step"]) if ranking else -1
    status_payload = dict(training_status or {})
    status_payload.setdefault("stopped_early", False)
    status_payload.setdefault("stop_reason", "")
    status_payload.setdefault("completed_steps", 0)
    status_payload.setdefault("target_steps", 0)
    status_payload.setdefault("early_stop_mode", "balanced")
    status_payload.setdefault("collapse_detected", False)
    return {
        "finetune_id": finetune_id,
        "checkpoint_avg_metric": checkpoint_avg_metric,
        "checkpoint_avg_splits": checkpoint_avg_splits,
        "best_avg_eval_reward": best_avg_eval_reward,
        "best_avg_eval_reward_step": best_avg_eval_reward_step,
        "training_status": status_payload,
        "rankings": ranking,
    }


def _filter_wandb_metrics(metrics: dict[str, float], keys: tuple[str, ...]) -> dict[str, float]:
    return {key: metrics[key] for key in keys if key in metrics}


def _select_eval_wandb_metrics(metrics: dict[str, float]) -> dict[str, float]:
    selected = _filter_wandb_metrics(metrics, MAIN_EVAL_WANDB_METRIC_KEYS)
    for task in sorted(SUPPORTED_TASKS):
        key = f"eval_task_accuracy_{task}"
        if key in metrics:
            selected[key] = metrics[key]
    return selected


def _is_checkpoint_not_found_error(exc: Exception) -> bool:
    if not isinstance(exc, TunaAPIError):
        return False
    if exc.status_code == 404 and "checkpoint" in str(exc).lower():
        return True
    return "checkpoint not found" in str(exc).lower()


def _try_save_checkpoint(
    *,
    finetune: Any,
    context: str,
) -> bool:
    try:
        finetune.save_checkpoint()
        return True
    except (TunaAPIError, TunaNetworkError) as exc:
        if _is_checkpoint_not_found_error(exc):
            print(
                f"{context}: checkpoint unavailable yet; continuing. "
                f"details: {_error_details(exc)}"
            )
            return False
        print(
            f"{context}: checkpoint save failed; continuing. "
            f"details: {_error_details(exc)}"
        )
        return False


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
    args.hf_token = dataset_loader.resolve_hf_token(args.hf_token)

    if args.eval_max_samples is not None and args.eval_max_samples <= 0:
        args.eval_max_samples = None

    eval_temperature = float(args.temperature) if args.eval_temperature is None else float(args.eval_temperature)
    eval_top_p = float(args.top_p) if args.eval_top_p is None else float(args.eval_top_p)
    eval_reasoning = bool(args.reasoning) if args.eval_reasoning is None else bool(args.eval_reasoning)
    eval_fixed_subset_size = int(args.eval_fixed_subset_size)
    eval_fixed_subset_seed = int(args.eval_fixed_subset_seed)

    dataset_dir: Optional[Path] = None
    if args.dataset_source == "local_jsonl":
        dataset_dir = Path(args.dataset_dir).expanduser().resolve()
        if not dataset_dir.exists():
            raise FileNotFoundError(f"dataset_dir not found: {dataset_dir}")

    final_eval_splits = _dedupe_splits(list(args.final_eval_splits))
    if not final_eval_splits:
        raise ValueError("--final-eval-splits must contain at least one split")
    checkpoint_avg_splits = _dedupe_splits(list(args.checkpoint_avg_splits))
    if not checkpoint_avg_splits:
        raise ValueError("--checkpoint-avg-splits must contain at least one split")

    rng = random.Random(args.seed)

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
    for item in train_examples:
        train_examples_by_task.setdefault(item.task_type, []).append(item)
    if not train_examples_by_task:
        raise ValueError("train split has no task-indexable examples")

    sampling_tasks = sorted(train_examples_by_task.keys())
    sampling_weights = [float(args.task_sampling_weights.get(task, 1.0)) for task in sampling_tasks]
    if not sampling_tasks:
        raise ValueError("no tasks available for weighted sampling")

    split_examples_cache: dict[str, list[QAExample]] = {
        args.train_split: train_examples,
        args.val_split: val_examples,
    }

    def _examples_for_split(split_name: str) -> list[QAExample]:
        if split_name not in split_examples_cache:
            split_examples_cache[split_name] = _load_split_examples(
                split_name=split_name,
                dataset_source=args.dataset_source,
                dataset_dir=dataset_dir,
                hf_dataset_repo_id=args.hf_dataset_repo_id,
                hf_dataset_revision=args.hf_dataset_revision,
                hf_token=args.hf_token,
                hf_cache_dir=args.hf_cache_dir,
            )
        return split_examples_cache[split_name]

    final_eval_examples: dict[str, list[QAExample]] = {}
    for split in final_eval_splits:
        final_eval_examples[split] = _examples_for_split(split)
    checkpoint_avg_examples = {split: _examples_for_split(split) for split in checkpoint_avg_splits}
    eval_splits_for_fixed_indices: dict[str, list[QAExample]] = {args.val_split: val_examples}
    eval_splits_for_fixed_indices.update(checkpoint_avg_examples)
    eval_splits_for_fixed_indices.update(final_eval_examples)
    fixed_eval_indices_by_split = _build_fixed_eval_indices(
        split_examples=eval_splits_for_fixed_indices,
        fixed_subset_size=eval_fixed_subset_size,
        fixed_subset_seed=eval_fixed_subset_seed,
        max_samples=args.eval_max_samples,
    )
    if fixed_eval_indices_by_split:
        print(
            "using fixed eval subsets: "
            + ", ".join(
                f"{name}={len(indices)}" for name, indices in sorted(fixed_eval_indices_by_split.items())
            )
        )

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
            "dataset_source": args.dataset_source,
            "dataset_dir": str(dataset_dir) if dataset_dir is not None else "",
            "hf_dataset_repo_id": args.hf_dataset_repo_id,
            "hf_dataset_revision": args.hf_dataset_revision,
            "hf_cache_dir": args.hf_cache_dir,
            "train_split": args.train_split,
            "val_split": args.val_split,
            "final_eval_splits": final_eval_splits,
            "checkpoint_avg_splits": checkpoint_avg_splits,
            "checkpoint_avg_metric": args.checkpoint_avg_metric,
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
            "eval_temperature": eval_temperature,
            "eval_top_p": eval_top_p,
            "eval_reasoning": eval_reasoning,
            "eval_temperature_override": args.eval_temperature,
            "eval_top_p_override": args.eval_top_p,
            "eval_reasoning_override": args.eval_reasoning,
            "max_tokens": args.max_tokens,
            "off_policy": args.off_policy,
            "off_policy_mix_ratio": args.off_policy_mix_ratio,
            "off_policy_buffer_size": args.off_policy_buffer_size,
            "off_policy_warmup_steps": args.off_policy_warmup_steps,
            "off_policy_min_buffer_groups": args.off_policy_min_buffer_groups,
            "max_tokens_by_task": args.max_tokens_by_task,
            "reasoning": args.reasoning,
            "task_sampling_weights": {
                task: float(args.task_sampling_weights.get(task, 1.0))
                for task in sorted(args.task_sampling_weights.keys())
            },
            "eval_every": args.eval_every,
            "save_every": args.save_every,
            "save_on_eval": args.save_on_eval,
            "eval_batch_size": args.eval_batch_size,
            "eval_max_samples": args.eval_max_samples,
            "eval_fixed_subset_size": eval_fixed_subset_size,
            "eval_fixed_subset_seed": eval_fixed_subset_seed,
            "early_stop": args.early_stop,
            "early_stop_mode": args.early_stop_mode,
            "skip_final_eval": args.skip_final_eval,
            "best_metric": args.best_metric,
            "best_move_optimal_reward": args.best_move_optimal_reward,
            "checkpoint_ranking_output": args.checkpoint_ranking_output,
        },
    )
    run.summary["finetune_id"] = finetune.finetune_id

    best_metric_value: Optional[float] = None
    best_step: Optional[int] = None
    best_eval_reward_seen: Optional[float] = None
    low_parse_streak = 0
    reward_drop_streak = 0
    recent_eval_rewards: list[float] = []
    completed_steps = 0
    training_status = _default_training_status(
        target_steps=int(args.num_steps),
        early_stop_mode=str(args.early_stop_mode),
    )
    checkpoint_eval_history: list[dict[str, Any]] = []
    replay_buffer: deque[Any] = deque(maxlen=int(args.off_policy_buffer_size))

    def _run_checkpoint_eval(
        *,
        step_for_log: int,
        seed_base: int,
        stage_label: str,
    ) -> dict[str, dict[str, float]]:
        by_split: dict[str, dict[str, float]] = {}
        for split_idx, split_name in enumerate(checkpoint_avg_splits):
            split_metrics = _evaluate_split(
                finetune=finetune,
                examples=checkpoint_avg_examples[split_name],
                split_name=split_name,
                seed=seed_base + split_idx,
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
                show_progress=show_progress,
                fixed_indices=fixed_eval_indices_by_split.get(split_name),
            )
            by_split[split_name] = split_metrics
            prefix = f"checkpoint_{_sanitize_split_name(split_name)}_"
            split_wandb_metrics = _select_eval_wandb_metrics(split_metrics)
            wandb.log({f"{prefix}{k}": v for k, v in split_wandb_metrics.items()}, step=step_for_log)
            print(
                f"{stage_label} eval step={step_for_log} split={split_name} "
                f"reward={split_metrics['eval_reward_mean']:.4f} "
                f"obj_parse={split_metrics['eval_json_object_rate']:.4f} "
                f"parse={split_metrics['eval_json_parse_rate']:.4f}"
            )

        avg_eval_reward = float(
            np.mean([float(m.get(args.checkpoint_avg_metric, 0.0)) for m in by_split.values()])
        )
        wandb.log({"checkpoint_avg_eval_reward_mean": avg_eval_reward}, step=step_for_log)
        print(
            f"{stage_label} checkpoint average step={step_for_log} "
            f"{args.checkpoint_avg_metric}={avg_eval_reward:.4f}"
        )

        checkpoint_eval_history.append(
            {
                "step": int(step_for_log),
                "avg_eval_reward_mean": avg_eval_reward,
                "checkpoint_avg_metric": args.checkpoint_avg_metric,
                "split_metrics": {k: _numeric_metrics_only(v) for k, v in by_split.items()},
            }
        )
        return by_split

    if args.eval_every > 0:
        print("running baseline checkpoint-average eval...")
        baseline_by_split = _run_checkpoint_eval(
            step_for_log=args.resume_step,
            seed_base=args.seed + 101,
            stage_label="baseline",
        )
        baseline_metrics = baseline_by_split.get(args.val_split)
        if baseline_metrics is None:
            baseline_metrics = _evaluate_split(
                finetune=finetune,
                examples=val_examples,
                split_name=args.val_split,
                seed=args.seed + 151,
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
                show_progress=show_progress,
                fixed_indices=fixed_eval_indices_by_split.get(args.val_split),
            )
        baseline_wandb_metrics = _select_eval_wandb_metrics(baseline_metrics)
        wandb.log({f"baseline_{k}": v for k, v in baseline_wandb_metrics.items()}, step=args.resume_step)

        baseline_metric = float(baseline_metrics.get(args.best_metric, 0.0))
        best_metric_value = baseline_metric
        best_step = args.resume_step
        best_eval_reward_seen = float(baseline_metrics.get("eval_reward_mean", 0.0))
        print(
            f"baseline eval step={args.resume_step} reward={baseline_metrics['eval_reward_mean']:.4f} "
            f"obj_parse={baseline_metrics['eval_json_object_rate']:.4f} "
            f"parse={baseline_metrics['eval_json_parse_rate']:.4f} "
            f"best_move_set={baseline_metrics['eval_best_move_set_accuracy']:.4f} "
            f"best_move_canon={baseline_metrics['eval_best_move_canonical_accuracy']:.4f} "
            f"exact_non_best={baseline_metrics['eval_exact_accuracy_non_best_move']:.4f}"
        )
        if args.save_on_eval:
            baseline_saved = _try_save_checkpoint(
                finetune=finetune,
                context=f"baseline eval step={args.resume_step}",
            )
            if checkpoint_eval_history:
                checkpoint_eval_history[-1]["checkpoint_saved"] = bool(baseline_saved)
            if baseline_saved:
                print("baseline checkpoint saved (save_on_eval=true)")
        elif checkpoint_eval_history:
            checkpoint_eval_history[-1]["checkpoint_saved"] = False

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

        sampled_tasks = rng.choices(sampling_tasks, weights=sampling_weights, k=args.batch_size)
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

        on_policy_groups: list[Any] = []
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
                on_policy_groups.append(result.to_group(rewards=rewards))

        if not on_policy_groups:
            print(f"step {global_step}: no train groups produced; skipping")
            continue

        train_groups, off_policy_groups_used = _compose_train_groups(
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
            "accepted_groups": float(len(train_groups)),
            "on_policy_groups": float(len(train_groups) - off_policy_groups_used),
            "off_policy_groups": float(off_policy_groups_used),
            "off_policy_group_fraction": float(
                off_policy_groups_used / max(1, len(train_groups))
            ),
            "replay_buffer_size": float(len(replay_buffer)),
            "kl": float(train_out.kl or 0.0),
            "router_kl": float(train_out.router_kl or 0.0),
            "grad_norm": float(train_out.grad_norm or 0.0),
        }
        wandb.log(_filter_wandb_metrics(train_metrics, MAIN_TRAIN_WANDB_METRIC_KEYS), step=global_step)

        print(
            f"step {global_step} reward={reward_mean:.4f} "
            f"obj_parse_rate={object_parse_rate:.4f} parse_rate={parse_rate:.4f} "
            f"offp={off_policy_groups_used}/{len(train_groups)} "
            f"replay={len(replay_buffer)} "
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
            eval_by_split = _run_checkpoint_eval(
                step_for_log=global_step,
                seed_base=args.seed + 1000 + (global_step * 13),
                stage_label="checkpoint",
            )
            eval_metrics = eval_by_split.get(args.val_split)
            if eval_metrics is None:
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
                    show_progress=show_progress,
                    fixed_indices=fixed_eval_indices_by_split.get(args.val_split),
                )
            wandb.log(_select_eval_wandb_metrics(eval_metrics), step=global_step)
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
                if not args.save_on_eval:
                    best_saved = _try_save_checkpoint(
                        finetune=finetune,
                        context=f"best metric checkpoint step={global_step}",
                    )
                    if checkpoint_eval_history:
                        checkpoint_eval_history[-1]["checkpoint_saved"] = bool(best_saved)
                    if best_saved:
                        print(
                            f"new best {args.best_metric}={metric_value:.4f} at step {global_step}; checkpoint saved"
                        )
                    else:
                        print(
                            f"new best {args.best_metric}={metric_value:.4f} at step {global_step}; checkpoint save skipped"
                        )
                else:
                    print(
                        f"new best {args.best_metric}={metric_value:.4f} at step {global_step}"
                    )

            if args.save_on_eval:
                eval_saved = _try_save_checkpoint(
                    finetune=finetune,
                    context=f"periodic eval step={global_step}",
                )
                if checkpoint_eval_history:
                    checkpoint_eval_history[-1]["checkpoint_saved"] = bool(eval_saved)
                if eval_saved:
                    print(f"checkpoint saved at eval step={global_step} (save_on_eval=true)")
            elif checkpoint_eval_history and "checkpoint_saved" not in checkpoint_eval_history[-1]:
                checkpoint_eval_history[-1]["checkpoint_saved"] = False

            eval_reward_value = float(eval_metrics.get("eval_reward_mean", 0.0))
            eval_parse_rate_value = float(eval_metrics.get("eval_json_parse_rate", 0.0))
            thresholds = _early_stop_thresholds(args.early_stop_mode)
            prior_best_eval_reward = (
                eval_reward_value if best_eval_reward_seen is None else float(best_eval_reward_seen)
            )

            if eval_parse_rate_value < float(thresholds["parse_floor"]):
                low_parse_streak += 1
            else:
                low_parse_streak = 0

            if eval_reward_value <= (prior_best_eval_reward - float(thresholds["reward_drop"])):
                reward_drop_streak += 1
            else:
                reward_drop_streak = 0

            if best_eval_reward_seen is None or eval_reward_value > best_eval_reward_seen:
                best_eval_reward_seen = eval_reward_value

            recent_eval_rewards.append(eval_reward_value)
            if args.early_stop:
                should_stop, stop_reason, collapse_detected = _should_early_stop(
                    mode=args.early_stop_mode,
                    parse_streak=low_parse_streak,
                    reward_drop_streak=reward_drop_streak,
                    recent_eval_rewards=recent_eval_rewards,
                )
                if should_stop:
                    training_status.update(
                        {
                            "stopped_early": True,
                            "stop_reason": stop_reason,
                            "completed_steps": int(completed_steps),
                            "collapse_detected": bool(collapse_detected),
                        }
                    )
                    print(
                        f"early stop triggered at step={global_step}: reason={stop_reason} "
                        f"parse_streak={low_parse_streak} reward_drop_streak={reward_drop_streak}"
                    )
                    break

        if args.save_every > 0 and (global_step + 1) % args.save_every == 0:
            _try_save_checkpoint(
                finetune=finetune,
                context=f"save_every checkpoint step={global_step}",
            )

    if not bool(training_status.get("stopped_early")):
        training_status["completed_steps"] = int(completed_steps)

    _try_save_checkpoint(
        finetune=finetune,
        context="final checkpoint save",
    )

    final_eval_step = args.resume_step + int(completed_steps)
    if args.skip_final_eval:
        print("skip_final_eval=true; skipping final split eval pass.")
    else:
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
                temperature=eval_temperature,
                top_p=eval_top_p,
                max_tokens=args.max_tokens,
                max_tokens_by_task=args.max_tokens_by_task,
                reasoning=eval_reasoning,
                best_move_optimal_reward=args.best_move_optimal_reward,
                show_progress=show_progress,
                fixed_indices=fixed_eval_indices_by_split.get(split_name),
            )
            prefix = _metric_prefix_for_split(split_name)
            final_eval_wandb_metrics = _select_eval_wandb_metrics(eval_metrics)
            wandb.log({f"{prefix}{k}": v for k, v in final_eval_wandb_metrics.items()}, step=final_eval_step)

            print(
                f"final eval split={split_name} reward={eval_metrics['eval_reward_mean']:.4f} "
                f"obj_parse={eval_metrics['eval_json_object_rate']:.4f} "
                f"parse={eval_metrics['eval_json_parse_rate']:.4f} "
                f"best_move_set={eval_metrics['eval_best_move_set_accuracy']:.4f} "
                f"best_move_canon={eval_metrics['eval_best_move_canonical_accuracy']:.4f} "
                f"exact_non_best={eval_metrics['eval_exact_accuracy_non_best_move']:.4f}"
            )

    ranking_payload = _build_checkpoint_ranking_payload(
        finetune_id=finetune.finetune_id,
        checkpoint_avg_metric=args.checkpoint_avg_metric,
        checkpoint_avg_splits=checkpoint_avg_splits,
        checkpoint_eval_history=checkpoint_eval_history,
        training_status=training_status,
    )
    checkpoint_ranking = ranking_payload["rankings"]
    best_avg_eval_reward = float(ranking_payload["best_avg_eval_reward"])
    best_avg_eval_reward_step = int(ranking_payload["best_avg_eval_reward_step"])
    if checkpoint_ranking:
        print(
            "best checkpoint by average eval reward: "
            f"step={best_avg_eval_reward_step} avg_eval_reward_mean={best_avg_eval_reward:.4f} "
            f"splits={checkpoint_avg_splits}"
        )
    else:
        print("no checkpoint eval history to rank (eval_every may be 0).")

    ranking_output_path = _resolve_output_path(
        args.checkpoint_ranking_output,
        fallback_name=f"checkpoint_ranking_{finetune.finetune_id}.json",
    )
    ranking_output_path.parent.mkdir(parents=True, exist_ok=True)
    ranking_output_path.write_text(
        json.dumps(ranking_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"wrote checkpoint ranking JSON: {ranking_output_path}")

    run.summary["best_metric_name"] = args.best_metric
    run.summary["best_metric_value"] = float(best_metric_value or 0.0)
    run.summary["best_metric_step"] = int(best_step if best_step is not None else -1)
    run.summary["finetune_id"] = finetune.finetune_id
    run.summary["train_rows"] = len(train_examples)
    run.summary["val_rows"] = len(val_examples)
    run.summary["best_avg_eval_reward"] = float(best_avg_eval_reward)
    run.summary["best_avg_eval_reward_step"] = int(best_avg_eval_reward_step)
    run.summary["best_avg_eval_reward_splits"] = ",".join(checkpoint_avg_splits)
    run.summary["checkpoint_ranking_output"] = str(ranking_output_path)
    run.summary["stopped_early"] = bool(training_status.get("stopped_early"))
    run.summary["early_stop_reason"] = str(training_status.get("stop_reason", ""))
    run.summary["completed_steps"] = int(training_status.get("completed_steps", completed_steps))
    run.summary["target_steps"] = int(training_status.get("target_steps", args.num_steps))
    run.summary["collapse_detected"] = bool(training_status.get("collapse_detected"))

    run.finish()
    client.close()

    print(
        f"done. finetune_id={finetune.finetune_id} best_{args.best_metric}={best_metric_value} "
        f"best_step={best_step} best_avg_eval_reward={best_avg_eval_reward} "
        f"best_avg_eval_reward_step={best_avg_eval_reward_step} "
        f"stopped_early={training_status.get('stopped_early')} "
        f"stop_reason={training_status.get('stop_reason', '')}"
    )


if __name__ == "__main__":
    main()
