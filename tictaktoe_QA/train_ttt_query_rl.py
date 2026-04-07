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
import shlex
import subprocess
from statistics import fmean, pvariance
import string
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

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
if __package__ in {None, ""} and str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from async_checkpoint_eval import (
    CheckpointEvalResult,
    DispatchHandle,
    dispatch_checkpoint_eval,
    drain_checkpoint_eval_jobs,
    poll_checkpoint_eval_jobs,
)
from tuna_sdk import QueryOutput, QueryRequest, QuerySettings, TunaClient  # noqa: E402
from tuna_sdk.errors import TunaAPIError, TunaNetworkError  # noqa: E402

DEFAULT_BASE_URL = "https://api.moondream.ai/v1"
DEFAULT_DATASET_SOURCE = "hf_hub"
SUPPORTED_DATASET_SOURCES = ("hf_hub", "local_jsonl")
DEFAULT_HF_DATASET_REPO_ID = "maxs-m87/tictactoe-qa-v1"
DEFAULT_HF_DATASET_REVISION = "main"
DEFAULT_FINAL_EVAL_SPLITS = [
    "val",
    "test"
]
DEFAULT_REASONING = False

CANONICAL_TASK_TYPES: tuple[str, ...] = (
    "best_move",
    "has_winning_move",
    "turn_player",
    "winner",
    "is_game_over",
    "available_moves_count",
    "available_moves_list",
)
TASK_TYPE_ALIASES: dict[str, str] = {
    "is_terminal": "is_game_over",
    "legal_moves_count": "available_moves_count",
    "legal_moves_list": "available_moves_list",
}
TASK_TYPE_SET = set(CANONICAL_TASK_TYPES)
ANSWER_KEY_ALIASES: dict[str, dict[str, str]] = {
    "is_game_over": {
        "is_game_over": "is_game_over",
        "is_terminal": "is_game_over",
    },
    "available_moves_count": {
        "available_move_count": "available_move_count",
        "legal_move_count": "available_move_count",
    },
    "available_moves_list": {
        "available_moves": "available_moves",
        "legal_moves": "available_moves",
    },
}

SUPPORTED_TASKS = set(CANONICAL_TASK_TYPES)
DEFAULT_TASK_SAMPLING_WEIGHTS: dict[str, float] = {
    "best_move": 1.0,
    "available_moves_count": 1.0,
    "available_moves_list": 1.0,
}
DEFAULT_MAX_TOKENS_BY_TASK: dict[str, int] = {
    "available_moves_list": 800,
}
MAIN_TRAIN_WANDB_METRIC_KEYS = (
    "reward_mean",
    "train_json_object_rate",
    "train_json_parse_rate",
    "train_best_move_valid_prediction_count",
    "train_best_move_valid_prediction_rate",
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
    "eval_best_move_valid_prediction_count",
    "eval_best_move_valid_prediction_rate",
    "eval_best_move_center_prediction_rate",
    "eval_best_move_invalid_prediction_rate",
)
CHECKPOINT_AVG_METRIC_CHOICES = (
    "eval_reward_mean",
    "eval_best_move_set_accuracy",
    "eval_best_move_canonical_accuracy",
    "eval_json_parse_rate",
)
BEST_MOVE_REWARD_MODES = ("ranked", "hybrid_strict", "binary")
INTRA_TASK_SAMPLING_STRATEGIES: dict[str, set[str]] = {
    "turn_player": {"balanced_player"},
    "available_moves_count": {"uniform_count"},
    "best_move": {"uniform_canonical_move", "center_hard_negative"},
}
WANDB_LOG_PROFILES = ("legacy", "lean")

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
    "api_key_env_var",
    "async_checkpoint_eval",
    "async_checkpoint_eval_dir",
    "async_checkpoint_eval_drain_on_exit",
    "async_checkpoint_eval_max_inflight",
    "auto_benchmark_best_checkpoint",
    "auto_benchmark_config",
    "auto_benchmark_output_json",
    "auto_benchmark_predictions_jsonl",
    "base_url",
    "batch_size",
    "best_metric",
    "best_move_center_not_optimal_ratio",
    "best_move_optimal_reward",
    "best_move_reward_mode",
    "best_move_wrong_rank_scale",
    "checkpoint_avg_metric",
    "checkpoint_avg_splits",
    "checkpoint_ranking_output",
    "center_bias_gate_after_evals",
    "center_bias_gate_enabled",
    "center_bias_gate_min_best_move_samples",
    "center_bias_gate_threshold",
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
    "intra_task_sampling_json",
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
    "train_max_samples",
    "train_split",
    "train_subset_seed",
    "val_split",
    "wandb_project",
    "wandb_run_name",
    "wandb_log_profile",
}


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
    # Tuple entries are (move, value, depth), parsed from scores_by_move_json.
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


def _repo_relative(*parts: str) -> Path:
    return Path(__file__).resolve().parent.joinpath(*parts)


def _default_config_path() -> Path:
    return _repo_relative("configs", "query_rl_default.json")


def _default_benchmark_config_path() -> Path:
    return _repo_relative("configs", "benchmark_default.json")


def normalize_task_type(task_type: str, *, allow_unknown: bool = False) -> str:
    normalized = TASK_TYPE_ALIASES.get(str(task_type).strip(), str(task_type).strip())
    if normalized in TASK_TYPE_SET:
        return normalized
    if allow_unknown:
        return normalized
    raise ValueError(f"unknown task_type: {task_type}")


def normalize_answer_payload_for_task(task_type: str, payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload

    canonical_task = normalize_task_type(task_type, allow_unknown=True)
    aliases = ANSWER_KEY_ALIASES.get(canonical_task)
    if not aliases:
        return payload

    out: dict[str, Any] = {}
    for old_key, canonical_key in aliases.items():
        if old_key in payload and canonical_key not in out:
            out[canonical_key] = payload[old_key]

    if out:
        return out
    return payload


def resolve_hf_token(raw_token: str) -> str:
    token = str(raw_token or "").strip()
    if token:
        return token
    return (
        os.environ.get("HF_TOKEN", "").strip()
        or os.environ.get("HUGGINGFACE_HUB_TOKEN", "").strip()
    )


def normalize_dataset_source(raw_source: str) -> str:
    value = str(raw_source or "").strip().lower()
    if value in SUPPORTED_DATASET_SOURCES:
        return value
    raise ValueError(
        f"dataset_source must be one of {sorted(SUPPORTED_DATASET_SOURCES)}, got: {raw_source!r}"
    )


def _safe_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    return cleaned or "default"


def _resolve_hf_cache_dir(raw_cache_dir: str) -> Path:
    text = str(raw_cache_dir or "").strip()
    if text:
        return Path(text).expanduser().resolve()
    return (Path.cwd() / ".cache" / "tictaktoe_QA").resolve()


def _persist_image_bytes(
    image_bytes: bytes,
    *,
    image_cache_root: Path,
) -> Optional[Path]:
    if not image_bytes:
        return None

    digest = hashlib.sha1(image_bytes).hexdigest()  # noqa: S324
    out_path = image_cache_root / f"{digest}.png"
    if out_path.exists():
        return out_path.resolve()

    image_cache_root.mkdir(parents=True, exist_ok=True)
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img.convert("RGB").save(out_path, format="PNG")
        return out_path.resolve()
    except OSError:
        fallback = image_cache_root / f"{digest}.img"
        if not fallback.exists():
            fallback.write_bytes(image_bytes)
        return fallback.resolve()


def _resolve_path_candidate(raw_path: str, *, dataset_dir: Optional[Path]) -> Optional[Path]:
    text = str(raw_path or "").strip()
    if not text:
        return None

    path = Path(text).expanduser()
    if path.is_file():
        return path.resolve()

    if not path.is_absolute() and dataset_dir is not None:
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
            resolved = _resolve_path_candidate(value, dataset_dir=dataset_dir)
            if resolved is not None:
                return resolved

    image_payload = row.get("image")
    if isinstance(image_payload, dict):
        payload_path = image_payload.get("path")
        if isinstance(payload_path, str):
            resolved = _resolve_path_candidate(payload_path, dataset_dir=dataset_dir)
            if resolved is not None:
                return resolved

        payload_bytes = image_payload.get("bytes")
        if isinstance(payload_bytes, (bytes, bytearray)):
            return _persist_image_bytes(
                bytes(payload_bytes),
                image_cache_root=image_cache_root,
            )

    if isinstance(image_payload, (bytes, bytearray)):
        return _persist_image_bytes(
            bytes(image_payload),
            image_cache_root=image_cache_root,
        )

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

    cache_dir = _resolve_hf_cache_dir(hf_cache_dir)
    repo_id = str(hf_dataset_repo_id or "").strip()
    if not repo_id:
        raise ValueError("hf_dataset_repo_id is required when dataset_source='hf_hub'")

    revision = str(hf_dataset_revision or DEFAULT_HF_DATASET_REVISION).strip() or DEFAULT_HF_DATASET_REVISION
    resolved_token = resolve_hf_token(hf_token)

    load_kwargs: dict[str, Any] = {
        "split": split_name,
        "revision": revision,
        "cache_dir": str(cache_dir),
    }
    if resolved_token:
        load_kwargs["token"] = resolved_token

    try:
        ds = load_dataset(repo_id, **load_kwargs)
    except TypeError:
        token_value = load_kwargs.pop("token", None)
        if token_value:
            load_kwargs["use_auth_token"] = token_value
        ds = load_dataset(repo_id, **load_kwargs)

    if "image" in getattr(ds, "column_names", []):
        ds = ds.cast_column("image", HFImage(decode=False))

    image_cache_root = (
        cache_dir
        / "hf_images"
        / _safe_component(repo_id)
        / _safe_component(revision)
        / _safe_component(split_name)
    )

    rows: list[dict[str, Any]] = []
    skipped = 0
    for raw_row in ds:
        if not isinstance(raw_row, dict):
            skipped += 1
            continue

        row = dict(raw_row)
        image_path = _resolve_hf_row_image_path(
            row,
            dataset_dir=dataset_dir,
            image_cache_root=image_cache_root,
        )
        if image_path is not None:
            row["image_path"] = str(image_path)
            row["image"] = str(image_path)
        rows.append(row)

    print(
        "loaded split="
        f"{split_name} rows={len(rows)} skipped={skipped} "
        f"from hf_hub repo={repo_id} revision={revision}"
    )
    return rows


def load_split_rows(
    *,
    dataset_source: str,
    split_name: str,
    dataset_dir: Optional[Path],
    hf_dataset_repo_id: str,
    hf_dataset_revision: str,
    hf_token: str,
    hf_cache_dir: str,
) -> list[dict[str, Any]]:
    source = normalize_dataset_source(dataset_source)
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
    chars = string.ascii_lowercase + string.digits
    return "".join(random.choices(chars, k=length))


def _resolve_config_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path

    from_cwd = (Path.cwd() / path).resolve()
    if from_cwd.exists():
        return from_cwd

    from_repo = (REPO_ROOT / path).resolve()
    if from_repo.exists():
        return from_repo

    from_script = (_repo_relative(path.as_posix())).resolve()
    if from_script.exists():
        return from_script

    return from_cwd


def _load_json_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        if config_path == _default_config_path():
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
        try:
            task = normalize_task_type(str(raw_task).strip())
        except ValueError as exc:
            raise ValueError(f"{source}: unknown task_type '{raw_task}'") from exc
        try:
            weight = float(raw_weight)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{source}: weight for task_type '{task}' must be numeric") from exc
        if weight < 0.0:
            raise ValueError(f"{source}: weight for task_type '{task}' must be >= 0")
        out[task] = weight
    return out


def _normalize_max_tokens_by_task(raw_map: dict[str, Any], *, source: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for raw_task, raw_max_tokens in raw_map.items():
        try:
            task = normalize_task_type(str(raw_task).strip())
        except ValueError as exc:
            raise ValueError(f"{source}: unknown task_type '{raw_task}'") from exc
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


def _weights_for_sampling_tasks(
    *,
    tasks: list[str],
    task_sampling_weights: dict[str, float],
) -> list[float]:
    weights = [float(task_sampling_weights.get(task, 1.0)) for task in tasks]
    if tasks and not any(weight > 0.0 for weight in weights):
        raise ValueError(
            "effective task_sampling_weights for available train tasks must include at least one positive weight"
        )
    return weights


def _active_tasks_from_sampling_weights(task_sampling_weights: dict[str, float]) -> set[str]:
    return {
        task
        for task in SUPPORTED_TASKS
        if float(task_sampling_weights.get(task, 1.0)) > 0.0
    }


def _filter_examples_by_active_tasks(
    examples: list[QAExample],
    *,
    active_tasks: set[str],
) -> list[QAExample]:
    if not active_tasks:
        return []
    return [item for item in examples if item.task_type in active_tasks]


def _intra_task_bucket_value(
    example: QAExample,
    *,
    strategy: str,
) -> Optional[str]:
    if strategy == "balanced_player":
        normalized = _normalize_non_best_answer("turn_player", example.expected_answer)
        if not isinstance(normalized, dict):
            return None
        player = normalized.get("player")
        if isinstance(player, str) and player in {"X", "O"}:
            return player
        return None
    if strategy == "uniform_count":
        normalized = _normalize_non_best_answer("available_moves_count", example.expected_answer)
        if not isinstance(normalized, dict):
            return None
        count = _parse_int(normalized.get("available_move_count"))
        if count is None:
            return None
        return str(int(count))
    if strategy == "uniform_canonical_move":
        if example.task_type != "best_move":
            return None
        canonical = _parse_int(example.best_move_canonical)
        if canonical is None or canonical < 1 or canonical > 9:
            return None
        return str(int(canonical))
    if strategy == "center_hard_negative":
        if example.task_type != "best_move":
            return None
        center_move = 5
        if center_move in example.best_move_optimal_set:
            return "other"
        return "center_not_optimal"
    return None


def _build_intra_task_buckets(
    examples: list[QAExample],
    *,
    strategy: str,
) -> dict[str, list[QAExample]]:
    buckets: dict[str, list[QAExample]] = {}
    for item in examples:
        bucket_key = _intra_task_bucket_value(item, strategy=strategy)
        if bucket_key is None:
            continue
        buckets.setdefault(bucket_key, []).append(item)
    return {key: rows for key, rows in buckets.items() if rows}


def _prepare_intra_task_sampling_groups(
    *,
    train_examples_by_task: dict[str, list[QAExample]],
    intra_task_sampling: dict[str, str],
    best_move_center_not_optimal_ratio: float,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for task_name, strategy in sorted(intra_task_sampling.items()):
        rows = list(train_examples_by_task.get(task_name, []))
        if not rows:
            continue
        buckets = _build_intra_task_buckets(rows, strategy=strategy)
        if len(buckets) < 2:
            continue
        group_payload: dict[str, Any] = {
            "strategy": strategy,
            "bucket_keys": sorted(buckets.keys()),
            "buckets": buckets,
        }
        if strategy == "center_hard_negative":
            center_not_optimal_key = "center_not_optimal"
            other_key = "other"
            weights: dict[str, float] = {}
            for key in sorted(buckets.keys()):
                if key == center_not_optimal_key:
                    weights[key] = float(best_move_center_not_optimal_ratio)
                elif key == other_key:
                    weights[key] = float(1.0 - best_move_center_not_optimal_ratio)
                else:
                    weights[key] = 1.0
            total = sum(max(0.0, float(value)) for value in weights.values())
            if total > 0.0:
                group_payload["bucket_pick_weights"] = {
                    key: max(0.0, float(weights[key])) / total for key in sorted(weights.keys())
                }
        out[task_name] = group_payload
    return out


def _sample_training_example(
    *,
    task_name: str,
    train_examples_by_task: dict[str, list[QAExample]],
    intra_task_sampling_groups: dict[str, dict[str, Any]],
    rng: random.Random,
) -> QAExample:
    default_rows = train_examples_by_task[task_name]
    group = intra_task_sampling_groups.get(task_name)
    if not group:
        return rng.choice(default_rows)

    bucket_keys = list(group.get("bucket_keys", []))
    buckets = group.get("buckets", {})
    if not bucket_keys or not isinstance(buckets, dict):
        return rng.choice(default_rows)

    bucket_pick_weights_map = group.get("bucket_pick_weights", {})
    bucket_pick_weights: Optional[list[float]] = None
    if isinstance(bucket_pick_weights_map, dict):
        candidate = [float(bucket_pick_weights_map.get(key, 0.0)) for key in bucket_keys]
        if any(weight > 0.0 for weight in candidate):
            bucket_pick_weights = candidate

    if bucket_pick_weights is not None:
        bucket_key = rng.choices(bucket_keys, weights=bucket_pick_weights, k=1)[0]
    else:
        bucket_key = rng.choice(bucket_keys)
    bucket_rows = buckets.get(bucket_key)
    if isinstance(bucket_rows, list) and bucket_rows:
        return rng.choice(bucket_rows)
    return rng.choice(default_rows)


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


def _normalize_intra_task_sampling(raw_map: dict[str, Any], *, source: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw_task, raw_strategy in raw_map.items():
        try:
            task = normalize_task_type(str(raw_task).strip())
        except ValueError as exc:
            raise ValueError(f"{source}: unknown task_type '{raw_task}'") from exc
        strategy = str(raw_strategy).strip().lower()
        if not strategy:
            continue
        allowed = INTRA_TASK_SAMPLING_STRATEGIES.get(task)
        if not allowed:
            raise ValueError(
                f"{source}: task_type '{task}' does not support intra-task sampling strategies"
            )
        if strategy not in allowed:
            raise ValueError(
                f"{source}: strategy '{strategy}' is not valid for task_type '{task}'. "
                f"allowed={sorted(allowed)}"
            )
        out[task] = strategy
    return out


def _resolve_intra_task_sampling(
    *,
    config_map: dict[str, Any],
    cli_override_json: str,
) -> dict[str, str]:
    resolved = _normalize_intra_task_sampling(
        config_map,
        source="config intra_task_sampling_json",
    )
    if str(cli_override_json or "").strip():
        cli_map = _parse_json_object_arg(cli_override_json, "--intra-task-sampling-json")
        resolved.update(
            _normalize_intra_task_sampling(
                cli_map,
                source="--intra-task-sampling-json",
            )
        )
    return resolved


def _build_parser(config: dict[str, Any], config_path: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Query RL finetuning for TicTacToe QA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=str(config_path),
        help="JSON config file. CLI flags override config values.",
    )
    parser.add_argument("--env-file", default=_cfg_str(config, "env_file", str(_repo_relative(".env"))))
    parser.add_argument("--api-key", default=_cfg_str(config, "api_key", ""))
    parser.add_argument("--api-key-env-var", default=_cfg_str(config, "api_key_env_var", ""))
    parser.add_argument("--base-url", default=_cfg_str(config, "base_url", ""))

    parser.add_argument(
        "--dataset-source",
        choices=sorted(SUPPORTED_DATASET_SOURCES),
        default=_cfg_str(config, "dataset_source", DEFAULT_DATASET_SOURCE),
        help="Dataset source: HF Hub or local JSONL directory.",
    )
    parser.add_argument(
        "--dataset-dir",
        default=_cfg_str(config, "dataset_dir", str(_repo_relative("synth_dataset/outputs/v2"))),
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
    parser.add_argument(
        "--train-max-samples",
        type=int,
        default=_cfg_int(config, "train_max_samples", 0),
        help="Max train samples to keep after loading. <=0 means full split.",
    )
    parser.add_argument(
        "--train-subset-seed",
        type=int,
        default=_cfg_int(config, "train_subset_seed", -1),
        help="Seed for deterministic train subset selection. <0 reuses --seed.",
    )
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
    async_eval_group = parser.add_mutually_exclusive_group()
    async_eval_group.add_argument("--async-checkpoint-eval", dest="async_checkpoint_eval", action="store_true")
    async_eval_group.add_argument("--no-async-checkpoint-eval", dest="async_checkpoint_eval", action="store_false")
    parser.set_defaults(async_checkpoint_eval=_cfg_bool(config, "async_checkpoint_eval", False))
    parser.add_argument(
        "--async-checkpoint-eval-dir",
        default=_cfg_str(config, "async_checkpoint_eval_dir", str(_repo_relative("outputs", "async_checkpoint_eval"))),
    )
    parser.add_argument(
        "--async-checkpoint-eval-max-inflight",
        type=int,
        default=_cfg_int(config, "async_checkpoint_eval_max_inflight", 1),
    )
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
    parser.set_defaults(
        async_checkpoint_eval_drain_on_exit=_cfg_bool(config, "async_checkpoint_eval_drain_on_exit", True)
    )
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
        choices=list(CHECKPOINT_AVG_METRIC_CHOICES),
        default=_cfg_str(config, "checkpoint_avg_metric", "eval_reward_mean"),
    )
    parser.add_argument(
        "--checkpoint-ranking-output",
        default=_cfg_str(config, "checkpoint_ranking_output", ""),
        help="Path to write periodic checkpoint ranking JSON.",
    )

    parser.add_argument(
        "--best-metric",
        choices=list(CHECKPOINT_AVG_METRIC_CHOICES),
        default=_cfg_str(config, "best_metric", "eval_reward_mean"),
    )
    parser.add_argument(
        "--best-move-optimal-reward",
        type=float,
        default=_cfg_float(config, "best_move_optimal_reward", 0.7),
    )
    parser.add_argument(
        "--best-move-reward-mode",
        choices=list(BEST_MOVE_REWARD_MODES),
        default=_cfg_str(config, "best_move_reward_mode", "ranked"),
    )
    parser.add_argument(
        "--best-move-wrong-rank-scale",
        type=float,
        default=_cfg_float(config, "best_move_wrong_rank_scale", 1.0),
    )
    parser.add_argument(
        "--best-move-center-not-optimal-ratio",
        type=float,
        default=_cfg_float(config, "best_move_center_not_optimal_ratio", 0.7),
        help=(
            "When intra-task strategy center_hard_negative is active for best_move, "
            "target sampling probability for rows where center is not optimal."
        ),
    )
    parser.add_argument(
        "--intra-task-sampling-json",
        default="",
        help=(
            "JSON object for intra-task bucket sampling strategies: "
            "{\"turn_player\":\"balanced_player\",\"available_moves_count\":\"uniform_count\","
            "\"best_move\":\"center_hard_negative\"}."
        ),
    )
    center_bias_gate_group = parser.add_mutually_exclusive_group()
    center_bias_gate_group.add_argument(
        "--center-bias-gate",
        dest="center_bias_gate_enabled",
        action="store_true",
        help="Enable early stop gate for excessive best_move center prediction bias.",
    )
    center_bias_gate_group.add_argument(
        "--no-center-bias-gate",
        dest="center_bias_gate_enabled",
        action="store_false",
        help="Disable early stop gate for best_move center prediction bias.",
    )
    parser.set_defaults(center_bias_gate_enabled=_cfg_bool(config, "center_bias_gate_enabled", False))
    parser.add_argument(
        "--center-bias-gate-threshold",
        type=float,
        default=_cfg_float(config, "center_bias_gate_threshold", 0.6),
        help="Trigger gate when eval_best_move_center_prediction_rate exceeds this threshold.",
    )
    parser.add_argument(
        "--center-bias-gate-after-evals",
        type=int,
        default=_cfg_int(config, "center_bias_gate_after_evals", 1),
        help="Number of periodic eval windows before center-bias gate activates.",
    )
    parser.add_argument(
        "--center-bias-gate-min-best-move-samples",
        type=int,
        default=_cfg_int(config, "center_bias_gate_min_best_move_samples", 100),
        help="Minimum best_move eval rows required before center-bias gate can trigger.",
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
        choices=["conservative", "balanced", "aggressive"],
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
    auto_bench_group = parser.add_mutually_exclusive_group()
    auto_bench_group.add_argument(
        "--auto-benchmark-best-checkpoint",
        dest="auto_benchmark_best_checkpoint",
        action="store_true",
        help="After training, benchmark the best checkpoint automatically.",
    )
    auto_bench_group.add_argument(
        "--no-auto-benchmark-best-checkpoint",
        dest="auto_benchmark_best_checkpoint",
        action="store_false",
        help="Disable automatic post-training benchmark execution.",
    )
    parser.set_defaults(
        auto_benchmark_best_checkpoint=_cfg_bool(config, "auto_benchmark_best_checkpoint", True)
    )
    parser.add_argument(
        "--auto-benchmark-config",
        default=_cfg_str(config, "auto_benchmark_config", str(_default_benchmark_config_path())),
        help="Benchmark config file used for automatic post-training benchmark.",
    )
    parser.add_argument(
        "--auto-benchmark-output-json",
        default=_cfg_str(config, "auto_benchmark_output_json", ""),
        help="Optional metrics path override for automatic post-training benchmark.",
    )
    parser.add_argument(
        "--auto-benchmark-predictions-jsonl",
        default=_cfg_str(config, "auto_benchmark_predictions_jsonl", ""),
        help="Optional predictions path override for automatic post-training benchmark.",
    )

    parser.add_argument("--wandb-project", default=_cfg_str(config, "wandb_project", "moondream-ttt-query-rl"))
    parser.add_argument("--wandb-run-name", default=_cfg_str(config, "wandb_run_name", ""))
    parser.add_argument(
        "--wandb-log-profile",
        choices=list(WANDB_LOG_PROFILES),
        default=_cfg_str(config, "wandb_log_profile", "legacy"),
        help="legacy keeps all prefixed eval streams; lean keeps only core streams.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        default=_cfg_bool(config, "no_progress", False),
        help="Disable tqdm progress bars.",
    )

    return parser


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=str(_default_config_path()))
    pre_args, _ = pre_parser.parse_known_args(argv)

    config_path = _resolve_config_path(pre_args.config)
    config = _load_json_config(config_path)
    _validate_config_keys(config, config_path=config_path)
    parser = _build_parser(config, config_path)
    args = parser.parse_args(argv)

    args.config = str(_resolve_config_path(args.config))
    args.auto_benchmark_config = str(_resolve_config_path(args.auto_benchmark_config))
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
    args.intra_task_sampling = _resolve_intra_task_sampling(
        config_map=_cfg_dict(config, "intra_task_sampling_json", {}),
        cli_override_json=args.intra_task_sampling_json,
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


def _resolve_api_key(*, explicit_api_key: str, api_key_env_var: str) -> str:
    direct = str(explicit_api_key or "").strip()
    if direct:
        return direct

    env_var_name = str(api_key_env_var or "").strip()
    if env_var_name:
        resolved = os.environ.get(env_var_name, "").strip()
        if resolved:
            return resolved

    return os.environ.get("MOONDREAM_API_KEY", "").strip()


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


def _resolve_output_path(raw_path: str, *, fallback_name: str) -> Path:
    text = str(raw_path or "").strip()
    if text:
        path = Path(text).expanduser()
        if path.is_absolute():
            return path
        return (Path.cwd() / path).resolve()
    return _repo_relative("outputs", fallback_name).resolve()


def _validate_args(args: argparse.Namespace) -> None:
    args.dataset_source = normalize_dataset_source(args.dataset_source)
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
    if args.train_max_samples < 0:
        raise ValueError("--train-max-samples must be >= 0")
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
    if not (0.0 <= args.best_move_wrong_rank_scale <= 1.0):
        raise ValueError("--best-move-wrong-rank-scale must be in [0,1]")
    if not (0.0 <= args.best_move_center_not_optimal_ratio <= 1.0):
        raise ValueError("--best-move-center-not-optimal-ratio must be in [0,1]")
    if not (0.0 <= args.center_bias_gate_threshold <= 1.0):
        raise ValueError("--center-bias-gate-threshold must be in [0,1]")
    if args.center_bias_gate_after_evals < 0:
        raise ValueError("--center-bias-gate-after-evals must be >= 0")
    if args.center_bias_gate_min_best_move_samples <= 0:
        raise ValueError("--center-bias-gate-min-best-move-samples must be > 0")
    if args.wandb_log_profile not in WANDB_LOG_PROFILES:
        raise ValueError(f"--wandb-log-profile must be one of {list(WANDB_LOG_PROFILES)}")
    if not args.checkpoint_avg_splits:
        raise ValueError("--checkpoint-avg-splits must contain at least one split")
    if args.async_checkpoint_eval_max_inflight <= 0:
        raise ValueError("--async-checkpoint-eval-max-inflight must be > 0")
    if args.async_checkpoint_eval and not args.save_on_eval:
        raise ValueError("--async-checkpoint-eval requires --save-on-eval")
    if args.async_checkpoint_eval and bool(args.early_stop):
        raise ValueError("--async-checkpoint-eval is incompatible with --early-stop")
    if args.async_checkpoint_eval and bool(args.center_bias_gate_enabled):
        raise ValueError("--async-checkpoint-eval is incompatible with --center-bias-gate-enabled")

    if args.finetune_id and args.finetune_name:
        raise ValueError("Provide either --finetune-id or --finetune-name, not both")
    if args.dataset_source == "local_jsonl" and not str(args.dataset_dir).strip():
        raise ValueError("--dataset-dir is required when --dataset-source=local_jsonl")
    if args.dataset_source == "hf_hub" and not str(args.hf_dataset_repo_id).strip():
        raise ValueError("--hf-dataset-repo-id is required when --dataset-source=hf_hub")
    if bool(args.auto_benchmark_best_checkpoint):
        auto_benchmark_config_path = Path(str(args.auto_benchmark_config)).expanduser().resolve()
        if not auto_benchmark_config_path.exists():
            raise FileNotFoundError(
                f"--auto-benchmark-config not found: {auto_benchmark_config_path}"
            )


def _warn_on_unsafe_mode_combo(*, off_policy: bool, reasoning: bool) -> None:
    if bool(off_policy) and bool(reasoning):
        print(
            "WARNING: off-policy + reasoning is an unsafe combination for this trainer; "
            "do not run with both enabled."
        )



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


def _scores_by_move_from_json(payload_json: str) -> tuple[tuple[int, int, int], ...]:
    try:
        payload = json.loads(payload_json)
    except json.JSONDecodeError:
        return tuple()
    if not isinstance(payload, dict):
        return tuple()

    out: list[tuple[int, int, int]] = []
    for move_key, score_payload in payload.items():
        move = _parse_int(move_key)
        if move is None or move < 1 or move > 9:
            continue
        if not isinstance(score_payload, dict):
            continue
        value = _parse_int(score_payload.get("value"))
        depth = _parse_int(score_payload.get("depth"))
        if value is None or depth is None:
            continue
        out.append((move, int(value), int(depth)))

    out.sort(key=lambda item: item[0])
    return tuple(out)


def _best_move_prediction_is_valid(
    move: Optional[int],
    *,
    example: QAExample,
) -> bool:
    if move is None:
        return False

    legal_moves = example.best_move_legal_moves
    if legal_moves:
        return move in legal_moves

    if example.best_move_scores:
        return any(scored_move == move for scored_move, _value, _depth in example.best_move_scores)

    return False


def _best_move_rank_key(*, value: int, depth: int) -> tuple[int, int]:
    # For winning lines (value=1), faster wins are better (lower depth).
    if value == 1:
        return (value, -depth)
    return (value, depth)


def _ranked_best_move_reward(
    move: int,
    *,
    scores_by_move: dict[int, tuple[int, int]],
) -> float:
    pred = scores_by_move.get(move)
    if pred is None:
        return 0.0

    total = len(scores_by_move)
    if total <= 1:
        return 1.0

    pred_key = _best_move_rank_key(value=int(pred[0]), depth=int(pred[1]))
    better_count = sum(
        1
        for value, depth in scores_by_move.values()
        if _best_move_rank_key(value=int(value), depth=int(depth)) > pred_key
    )
    reward = 1.0 - (float(better_count) / float(total - 1))
    return max(0.0, min(1.0, reward))


def _normalize_non_best_answer(task_type: str, payload: Any) -> Optional[Any]:
    if not isinstance(payload, dict):
        return None
    task_type = normalize_task_type(task_type, allow_unknown=True)
    payload = normalize_answer_payload_for_task(task_type, payload)

    if task_type == "winner":
        winner = _coerce_winner(payload.get("winner"))
        if winner is None:
            return None
        return {"winner": winner}

    if task_type == "is_game_over":
        is_game_over = _coerce_bool(payload.get("is_game_over"))
        if is_game_over is None:
            return None
        return {"is_game_over": is_game_over}

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

    if task_type == "available_moves_count":
        count = _parse_int(payload.get("available_move_count"))
        if count is None or count < 0 or count > 9:
            return None
        return {"available_move_count": count}

    if task_type == "available_moves_list":
        available_moves = _normalize_legal_moves(payload.get("available_moves"))
        if available_moves is None:
            return None
        return {"available_moves": available_moves}

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
    best_move_reward_mode: str = "ranked",
    best_move_wrong_rank_scale: float = 1.0,
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
        valid_prediction = _best_move_prediction_is_valid(move, example=example)
        legality_known = bool(example.best_move_legal_moves or example.best_move_scores)
        if move is None:
            return ScoreOutcome(
                reward=0.0,
                parse_success=False,
                task_correct=False,
                json_object_parsed=True,
                best_move_valid_prediction=valid_prediction,
            )
        if legality_known and not valid_prediction:
            return ScoreOutcome(
                reward=0.0,
                parse_success=True,
                task_correct=False,
                json_object_parsed=True,
                best_move_set_correct=False,
                best_move_canonical_correct=False,
                best_move_valid_prediction=False,
            )

        set_correct = move in example.best_move_optimal_set if move is not None else False
        canonical_correct = move == example.best_move_canonical if move is not None else False
        scores_by_move = {m: (value, depth) for m, value, depth in example.best_move_scores}
        if best_move_reward_mode == "binary":
            reward = 1.0 if set_correct else 0.0
        elif best_move_reward_mode == "hybrid_strict":
            if set_correct:
                reward = 1.0
            elif scores_by_move:
                ranked_reward = _ranked_best_move_reward(move, scores_by_move=scores_by_move)
                reward = float(best_move_wrong_rank_scale) * float(ranked_reward)
            else:
                reward = 0.0
        else:
            if scores_by_move:
                reward = _ranked_best_move_reward(move, scores_by_move=scores_by_move)
            elif canonical_correct:
                reward = 1.0
            elif set_correct:
                reward = float(best_move_optimal_reward)
            else:
                reward = 0.0

        return ScoreOutcome(
            reward=max(0.0, min(1.0, float(reward))),
            parse_success=True,
            task_correct=set_correct,
            json_object_parsed=True,
            best_move_set_correct=set_correct,
            best_move_canonical_correct=canonical_correct,
            best_move_valid_prediction=valid_prediction,
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
    best_move_reward_mode: str = "ranked",
    best_move_wrong_rank_scale: float = 1.0,
) -> ScoreOutcome:
    answer_text = _extract_rollout_answer(rollout)
    pred_payload = _parse_prediction_json(answer_text)
    return _score_payload_for_example(
        example,
        pred_payload,
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

    raw_task_type = str(row["task_type"]).strip()
    try:
        task_type = normalize_task_type(raw_task_type)
    except ValueError:
        print(
            f"split={split_name} line={line_number} unsupported task_type='{raw_task_type}'; skipping row"
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
    best_move_scores = _scores_by_move_from_json(str(row.get("scores_by_move_json", "")))
    best_move_legal_moves = frozenset(_as_move_set(str(row.get("legal_moves_json", ""))))
    if not best_move_legal_moves and best_move_scores:
        best_move_legal_moves = frozenset(move for move, _value, _depth in best_move_scores)

    return QAExample(
        row_id=str(row["row_id"]),
        split=str(row["split"]),
        task_type=task_type,
        question=str(row["question"]),
        image_path=image_path,
        expected_answer=expected_answer,
        best_move_canonical=best_move_canonical,
        best_move_optimal_set=best_move_optimal_set,
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
    rows = load_split_rows(
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


def _is_transient_rollout_error(exc: Exception) -> bool:
    if isinstance(exc, TunaNetworkError):
        return True
    if not isinstance(exc, TunaAPIError):
        return False
    if _is_rate_limit_error(exc):
        return True
    if exc.status_code in {408, 502, 503, 504, 524}:
        return True
    body = exc.response_body
    body_text = str(body if body is not None else exc).lower()
    return "error code: 524" in body_text or "timeout" in body_text


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
            should_retry = _is_transient_rollout_error(exc)
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
    best_move_reward_mode: str,
    best_move_wrong_rank_scale: float,
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
    best_move_valid_prediction_count = 0
    best_move_center_prediction_count = 0
    best_move_invalid_prediction_count = 0

    per_task_total: Counter[str] = Counter()
    per_task_correct: Counter[str] = Counter()

    def _consume_batch(batch_examples: list[QAExample]) -> None:
        nonlocal total_scored, reward_sum, object_parse_count, parse_success_count
        nonlocal best_move_total, best_move_set_correct, best_move_canonical_correct
        nonlocal best_move_valid_prediction_count
        nonlocal best_move_center_prediction_count, best_move_invalid_prediction_count

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
                pred_move: Optional[int] = None
            else:
                rollout = result.rollouts[0]
                answer_text = _extract_rollout_answer(rollout)
                pred_payload = _parse_prediction_json(answer_text)
                pred_move = _move_from_payload_obj(pred_payload)
                outcome = _score_payload_for_example(
                    item,
                    pred_payload,
                    best_move_optimal_reward=best_move_optimal_reward,
                    best_move_reward_mode=best_move_reward_mode,
                    best_move_wrong_rank_scale=best_move_wrong_rank_scale,
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
            "eval_best_move_valid_prediction_count": 0.0,
            "eval_best_move_valid_prediction_rate": 0.0,
            "eval_best_move_center_prediction_rate": 0.0,
            "eval_best_move_invalid_prediction_rate": 0.0,
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
        "eval_best_move_valid_prediction_count": float(best_move_valid_prediction_count),
        "eval_best_move_valid_prediction_rate": (
            best_move_valid_prediction_count / max(1, best_move_total)
        ),
        "eval_best_move_center_prediction_rate": (
            best_move_center_prediction_count / max(1, best_move_total)
        ),
        "eval_best_move_invalid_prediction_rate": (
            best_move_invalid_prediction_count / max(1, best_move_total)
        ),
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


def _subset_examples_deterministically(
    examples: list[QAExample],
    *,
    split_name: str,
    max_samples: Optional[int],
    subset_seed: int,
) -> list[QAExample]:
    if max_samples is None or max_samples <= 0 or len(examples) <= int(max_samples):
        return list(examples)
    indices = list(range(len(examples)))
    rng = random.Random(int(subset_seed) + _split_seed_offset(split_name))
    rng.shuffle(indices)
    keep = set(indices[: int(max_samples)])
    return [example for idx, example in enumerate(examples) if idx in keep]


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


def _rank_checkpoint_eval_history(
    history: list[dict[str, Any]],
    *,
    metric_key: str = "avg_checkpoint_metric",
) -> list[dict[str, Any]]:
    saved_entries = [item for item in history if bool(item.get("checkpoint_saved"))]
    candidates = saved_entries if saved_entries else history
    return sorted(
        candidates,
        key=lambda item: float(item.get(metric_key, item.get("avg_eval_reward_mean", 0.0))),
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
    ranking = _rank_checkpoint_eval_history(
        checkpoint_eval_history,
        metric_key="avg_checkpoint_metric",
    )
    best_avg_checkpoint_metric = (
        float(ranking[0].get("avg_checkpoint_metric", ranking[0].get("avg_eval_reward_mean", 0.0)))
        if ranking
        else 0.0
    )
    best_avg_checkpoint_metric_step = int(ranking[0]["step"]) if ranking else -1
    best_avg_eval_reward = (
        float(ranking[0].get("avg_eval_reward_mean", best_avg_checkpoint_metric))
        if ranking
        else 0.0
    )
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
        "best_avg_checkpoint_metric": best_avg_checkpoint_metric,
        "best_avg_checkpoint_metric_step": best_avg_checkpoint_metric_step,
        "best_avg_eval_reward": best_avg_eval_reward,
        "best_avg_eval_reward_step": best_avg_checkpoint_metric_step,
        "training_status": status_payload,
        "rankings": ranking,
    }


def _filter_wandb_metrics(metrics: dict[str, float], keys: tuple[str, ...]) -> dict[str, float]:
    return {key: metrics[key] for key in keys if key in metrics}


def _select_eval_wandb_metrics(metrics: dict[str, float]) -> dict[str, float]:
    selected = _filter_wandb_metrics(metrics, MAIN_EVAL_WANDB_METRIC_KEYS)
    for task in sorted(SUPPORTED_TASKS):
        key = f"eval_task_accuracy_{task}"
        count_key = f"eval_task_count_{task}"
        task_count = float(metrics.get(count_key, 0.0))
        if key in metrics and task_count > 0.0:
            selected[key] = metrics[key]
    return selected


def _should_log_prefixed_eval_streams(*, wandb_log_profile: str) -> bool:
    return str(wandb_log_profile).strip().lower() == "legacy"


def _checkpoint_wandb_payload(
    *,
    avg_checkpoint_metric: float,
    avg_eval_reward_mean: float,
    wandb_log_profile: str,
) -> dict[str, float]:
    if _should_log_prefixed_eval_streams(wandb_log_profile=wandb_log_profile):
        return {
            "checkpoint_avg_metric": float(avg_checkpoint_metric),
            "checkpoint_avg_eval_reward_mean": float(avg_eval_reward_mean),
        }
    return {
        "checkpoint_avg_metric_value": float(avg_checkpoint_metric),
    }


def _should_trigger_center_bias_gate(
    *,
    enabled: bool,
    eval_index: int,
    gate_after_evals: int,
    center_prediction_rate: float,
    gate_threshold: float,
    best_move_samples: float,
    min_best_move_samples: int,
) -> bool:
    if not bool(enabled):
        return False
    if int(eval_index) < int(gate_after_evals):
        return False
    if float(best_move_samples) < float(min_best_move_samples):
        return False
    return float(center_prediction_rate) > float(gate_threshold)


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


def _save_checkpoint_step(
    *,
    finetune: Any,
    context: str,
) -> Optional[int]:
    try:
        save_result = finetune.save_checkpoint()
        checkpoint = getattr(save_result, "checkpoint", None)
        checkpoint_step = getattr(checkpoint, "step", None)
        if checkpoint_step is None:
            print(f"{context}: checkpoint save returned no step")
            return None
        print(f"{context}: checkpoint save succeeded (checkpoint_step={int(checkpoint_step)})")
        return int(checkpoint_step)
    except (TunaAPIError, TunaNetworkError) as exc:
        if _is_checkpoint_not_found_error(exc):
            print(
                f"{context}: checkpoint unavailable yet; continuing. "
                f"details: {_error_details(exc)}"
            )
            return None
        print(
            f"{context}: checkpoint save failed; continuing. "
            f"details: {_error_details(exc)}"
        )
        return None


def _build_async_checkpoint_eval_command(
    *,
    args: argparse.Namespace,
    finetune_id: str,
    checkpoint_step: int,
    checkpoint_avg_splits: list[str],
    active_eval_tasks: set[str],
    eval_temperature: float,
    eval_top_p: float,
    eval_reasoning: bool,
    metrics_json_path: Path,
    predictions_jsonl_path: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        str((_repo_relative("benchmark_ttt_checkpoint_average.py")).resolve()),
        "--env-file",
        str(args.env_file),
        "--base-url",
        str(args.base_url),
        "--dataset-source",
        str(args.dataset_source),
        "--finetune-id",
        str(finetune_id),
        "--checkpoint-step",
        str(int(checkpoint_step)),
        "--checkpoint-avg-metric",
        str(args.checkpoint_avg_metric),
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
        "--avg-splits",
        *list(checkpoint_avg_splits),
        "--task-types",
        *sorted(active_eval_tasks),
        "--no-progress",
    ]
    if args.dataset_source == "local_jsonl":
        cmd.extend(["--dataset-dir", str(args.dataset_dir)])
    else:
        cmd.extend(["--hf-dataset-repo-id", str(args.hf_dataset_repo_id)])
        cmd.extend(["--hf-dataset-revision", str(args.hf_dataset_revision)])
        if str(args.hf_cache_dir).strip():
            cmd.extend(["--hf-cache-dir", str(args.hf_cache_dir)])
    if int(args.eval_max_samples or 0) > 0:
        cmd.extend(["--max-samples", str(int(args.eval_max_samples))])
    cmd.append("--reasoning" if bool(eval_reasoning) else "--no-reasoning")
    return cmd


def _ingest_async_checkpoint_eval_results(
    *,
    args: argparse.Namespace,
    checkpoint_eval_history: list[dict[str, Any]],
    results: list[CheckpointEvalResult],
    log_step: int,
    best_metric_value: Optional[float],
    best_step: Optional[int],
    best_eval_reward_seen: Optional[float],
) -> tuple[Optional[float], Optional[int], Optional[float], int]:
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
        payload = result.metrics_payload
        split_metrics_raw = payload.get("split_metrics", {})
        if not isinstance(split_metrics_raw, dict):
            print(f"async checkpoint eval malformed split_metrics step={source_step}")
            continue
        split_metrics: dict[str, dict[str, float]] = {
            str(split_name): {
                str(metric_name): float(metric_value)
                for metric_name, metric_value in metrics.items()
                if isinstance(metrics, dict)
                and isinstance(metric_value, (int, float))
            }
            for split_name, metrics in split_metrics_raw.items()
            if isinstance(metrics, dict)
        }
        avg_checkpoint_metric = float(payload.get("avg_checkpoint_metric", 0.0))
        avg_eval_reward_mean = float(payload.get("avg_eval_reward_mean", 0.0))
        checkpoint_eval_history.append(
            {
                "step": int(source_step),
                "avg_checkpoint_metric": avg_checkpoint_metric,
                "avg_eval_reward_mean": avg_eval_reward_mean,
                "checkpoint_avg_metric": args.checkpoint_avg_metric,
                "split_metrics": {k: _numeric_metrics_only(v) for k, v in split_metrics.items()},
                "checkpoint_saved": True,
                "saved_checkpoint_step": int(result.checkpoint_step),
            }
        )
        log_metadata = {
            "async_eval_source_step": int(source_step),
            "async_eval_checkpoint_step": int(result.checkpoint_step),
        }
        for split_name, metrics in split_metrics.items():
            if _should_log_prefixed_eval_streams(wandb_log_profile=args.wandb_log_profile):
                prefix = f"checkpoint_{_sanitize_split_name(split_name)}_"
                prefixed_metrics = {
                    f"{prefix}{k}": v for k, v in _select_eval_wandb_metrics(metrics).items()
                }
                wandb.log(
                    {**prefixed_metrics, **log_metadata},
                    step=arrival_step,
                )
            _print_eval_summary(
                label="async checkpoint eval",
                metrics=metrics,
                step=source_step,
                split_name=split_name,
            )
        wandb.log(
            {
                **_checkpoint_wandb_payload(
                    avg_checkpoint_metric=avg_checkpoint_metric,
                    avg_eval_reward_mean=avg_eval_reward_mean,
                    wandb_log_profile=args.wandb_log_profile,
                ),
                **log_metadata,
            },
            step=arrival_step,
        )
        print(
            f"async checkpoint average step={source_step} checkpoint_step={result.checkpoint_step} "
            f"{args.checkpoint_avg_metric}={avg_checkpoint_metric:.4f} logged_at_step={arrival_step}"
        )
        val_metrics = split_metrics.get(args.val_split)
        if val_metrics is not None:
            wandb.log(
                {**_select_eval_wandb_metrics(val_metrics), **log_metadata},
                step=arrival_step,
            )
            metric_value = float(val_metrics.get(args.best_metric, 0.0))
            if best_metric_value is None or metric_value > best_metric_value:
                best_metric_value = metric_value
                best_step = int(source_step)
            eval_reward_value = float(val_metrics.get("eval_reward_mean", 0.0))
            if best_eval_reward_seen is None or eval_reward_value > best_eval_reward_seen:
                best_eval_reward_seen = eval_reward_value
        success_count += 1
    return best_metric_value, best_step, best_eval_reward_seen, success_count


def _select_checkpoint_step_for_auto_benchmark(
    *,
    ranking_payload: dict[str, Any],
    fallback_step: Optional[int],
) -> Optional[int]:
    ranked_checkpoint_metric_step = _parse_int(ranking_payload.get("best_avg_checkpoint_metric_step"))
    if ranked_checkpoint_metric_step is not None and ranked_checkpoint_metric_step >= 0:
        return int(ranked_checkpoint_metric_step)
    ranked_step = _parse_int(ranking_payload.get("best_avg_eval_reward_step"))
    if ranked_step is not None and ranked_step >= 0:
        return int(ranked_step)
    if fallback_step is not None and int(fallback_step) >= 0:
        return int(fallback_step)
    return None


def _default_auto_benchmark_artifact_paths(
    *,
    finetune_id: str,
    checkpoint_step: Optional[int],
) -> tuple[Path, Path]:
    step_tag = f"step{checkpoint_step}" if checkpoint_step is not None else "latest"
    run_tag = time.strftime("%Y%m%d_%H%M%S")
    metrics_path = _resolve_output_path(
        "",
        fallback_name=f"benchmark_auto_{finetune_id}_{step_tag}_{run_tag}.json",
    )
    predictions_path = _resolve_output_path(
        "",
        fallback_name=f"benchmark_auto_{finetune_id}_{step_tag}_{run_tag}_predictions.jsonl",
    )
    return metrics_path, predictions_path


def _benchmark_config_task_types(config_path: str) -> Optional[list[str]]:
    path = Path(str(config_path)).expanduser().resolve()
    config = _load_json_config(path)
    raw_values = config.get("task_types")
    if not isinstance(raw_values, list):
        return None

    out: list[str] = []
    seen: set[str] = set()
    for value in raw_values:
        for piece in str(value).split(","):
            task_type = piece.strip()
            if not task_type:
                continue
            try:
                task_type = normalize_task_type(task_type)
            except ValueError as exc:
                raise ValueError(
                    f"auto benchmark config has unknown task_type '{task_type}' in {path}"
                ) from exc
            if task_type in seen:
                continue
            seen.add(task_type)
            out.append(task_type)

    return out or None


def _infer_auto_benchmark_task_types(args: argparse.Namespace) -> Optional[list[str]]:
    configured_task_types = _benchmark_config_task_types(str(args.auto_benchmark_config))
    if configured_task_types:
        return None

    active_tasks = sorted(_active_tasks_from_sampling_weights(args.task_sampling_weights))
    if len(active_tasks) == 1:
        return active_tasks
    return None


def _build_auto_benchmark_command(
    *,
    args: argparse.Namespace,
    finetune_id: str,
    checkpoint_step: Optional[int],
    dataset_dir: Optional[Path],
    output_json: Path,
    predictions_jsonl: Path,
    task_types: Optional[list[str]],
) -> list[str]:
    cmd = [
        sys.executable,
        str(_repo_relative("benchmark_ttt_query.py").resolve()),
        "--config",
        str(args.auto_benchmark_config),
        "--env-file",
        str(args.env_file),
        "--base-url",
        str(args.base_url),
        "--finetune-id",
        str(finetune_id),
        "--dataset-source",
        str(args.dataset_source),
        "--best-move-optimal-reward",
        str(float(args.best_move_optimal_reward)),
        "--best-move-reward-mode",
        str(args.best_move_reward_mode),
        "--best-move-wrong-rank-scale",
        str(float(args.best_move_wrong_rank_scale)),
        "--output-json",
        str(output_json),
        "--predictions-jsonl",
        str(predictions_jsonl),
    ]
    if checkpoint_step is not None:
        cmd.extend(["--checkpoint-step", str(int(checkpoint_step))])
    if args.dataset_source == "local_jsonl":
        if dataset_dir is not None:
            cmd.extend(["--dataset-dir", str(dataset_dir)])
    else:
        cmd.extend(["--hf-dataset-repo-id", str(args.hf_dataset_repo_id)])
        cmd.extend(["--hf-dataset-revision", str(args.hf_dataset_revision)])
        if str(args.hf_cache_dir).strip():
            cmd.extend(["--hf-cache-dir", str(args.hf_cache_dir)])
    if task_types:
        cmd.extend(["--task-types", ",".join(task_types)])
    if bool(args.no_progress):
        cmd.append("--no-progress")
    return cmd


def _run_auto_benchmark(
    *,
    args: argparse.Namespace,
    finetune_id: str,
    checkpoint_step: Optional[int],
    dataset_dir: Optional[Path],
    output_json_override: str,
    predictions_jsonl_override: str,
) -> tuple[bool, Path, Path, Optional[dict[str, Any]]]:
    default_output_json_path, default_predictions_jsonl_path = _default_auto_benchmark_artifact_paths(
        finetune_id=finetune_id,
        checkpoint_step=checkpoint_step,
    )

    if str(output_json_override or "").strip():
        output_json_path = _resolve_output_path(
            output_json_override,
            fallback_name=f"benchmark_auto_{finetune_id}.json",
        )
    else:
        output_json_path = default_output_json_path

    if str(predictions_jsonl_override or "").strip():
        predictions_jsonl_path = _resolve_output_path(
            predictions_jsonl_override,
            fallback_name=f"benchmark_auto_{finetune_id}_predictions.jsonl",
        )
    else:
        predictions_jsonl_path = default_predictions_jsonl_path

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    inferred_task_types = _infer_auto_benchmark_task_types(args)
    if inferred_task_types:
        print(
            "auto benchmark: inferred task filter from training weights: "
            + ",".join(inferred_task_types)
        )

    cmd = _build_auto_benchmark_command(
        args=args,
        finetune_id=finetune_id,
        checkpoint_step=checkpoint_step,
        dataset_dir=dataset_dir,
        output_json=output_json_path,
        predictions_jsonl=predictions_jsonl_path,
        task_types=inferred_task_types,
    )

    env = dict(os.environ)
    if str(args.api_key).strip():
        env["MOONDREAM_API_KEY"] = str(args.api_key).strip()
    if str(args.hf_token).strip():
        env.setdefault("HF_TOKEN", str(args.hf_token).strip())

    print("auto benchmark: running command")
    print("  " + " ".join(shlex.quote(part) for part in cmd))

    completed = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if completed.returncode != 0:
        print(f"auto benchmark failed with exit_code={completed.returncode}")
        if completed.stdout.strip():
            print("auto benchmark stdout:")
            print(_truncate(completed.stdout.strip(), limit=1200))
        if completed.stderr.strip():
            print("auto benchmark stderr:")
            print(_truncate(completed.stderr.strip(), limit=1200))
        return False, output_json_path, predictions_jsonl_path, None

    metrics_payload: Optional[dict[str, Any]] = None
    try:
        if output_json_path.exists():
            parsed = json.loads(output_json_path.read_text(encoding="utf-8"))
            if isinstance(parsed, dict):
                metrics_payload = parsed
    except (OSError, json.JSONDecodeError) as exc:
        print(f"auto benchmark: unable to parse metrics JSON ({output_json_path}): {exc}")

    if completed.stdout.strip():
        print("auto benchmark stdout:")
        print(_truncate(completed.stdout.strip(), limit=1200))
    if completed.stderr.strip():
        print("auto benchmark stderr:")
        print(_truncate(completed.stderr.strip(), limit=1200))

    print(
        "auto benchmark completed: "
        f"metrics={output_json_path} predictions={predictions_jsonl_path}"
    )
    return True, output_json_path, predictions_jsonl_path, metrics_payload


def _build_dataset_load_kwargs(
    args: argparse.Namespace,
    *,
    dataset_dir: Optional[Path],
) -> dict[str, Any]:
    return {
        "dataset_source": args.dataset_source,
        "dataset_dir": dataset_dir,
        "hf_dataset_repo_id": args.hf_dataset_repo_id,
        "hf_dataset_revision": args.hf_dataset_revision,
        "hf_token": args.hf_token,
        "hf_cache_dir": args.hf_cache_dir,
    }


def _build_eval_kwargs(
    args: argparse.Namespace,
    *,
    eval_temperature: float,
    eval_top_p: float,
    eval_reasoning: bool,
    show_progress: bool,
) -> dict[str, Any]:
    return {
        "batch_size": args.eval_batch_size,
        "max_workers": args.max_workers,
        "max_samples": args.eval_max_samples,
        "rollout_retries": args.rollout_retries,
        "rollout_retry_backoff_s": args.rollout_retry_backoff_s,
        "temperature": eval_temperature,
        "top_p": eval_top_p,
        "max_tokens": args.max_tokens,
        "max_tokens_by_task": args.max_tokens_by_task,
        "reasoning": eval_reasoning,
        "best_move_optimal_reward": args.best_move_optimal_reward,
        "best_move_reward_mode": args.best_move_reward_mode,
        "best_move_wrong_rank_scale": args.best_move_wrong_rank_scale,
        "show_progress": show_progress,
    }


def _run_split_eval(
    *,
    finetune: Any,
    split_name: str,
    examples: list[QAExample],
    seed: int,
    eval_kwargs: dict[str, Any],
    fixed_indices: Optional[list[int]],
) -> dict[str, float]:
    return _evaluate_split(
        finetune=finetune,
        examples=examples,
        split_name=split_name,
        seed=seed,
        fixed_indices=fixed_indices,
        **eval_kwargs,
    )


def _print_eval_summary(
    *,
    label: str,
    metrics: dict[str, float],
    step: Optional[int] = None,
    split_name: Optional[str] = None,
) -> None:
    prefix_parts = [label]
    if step is not None:
        prefix_parts.append(f"step={step}")
    if split_name is not None:
        prefix_parts.append(f"split={split_name}")
    prefix = " ".join(prefix_parts)
    print(
        f"{prefix} reward={metrics['eval_reward_mean']:.4f} "
        f"obj_parse={metrics['eval_json_object_rate']:.4f} "
        f"parse={metrics['eval_json_parse_rate']:.4f} "
        f"best_move_set={metrics['eval_best_move_set_accuracy']:.4f} "
        f"best_move_canon={metrics['eval_best_move_canonical_accuracy']:.4f} "
        f"best_move_valid={metrics.get('eval_best_move_valid_prediction_count', 0.0):.0f}/"
        f"{metrics.get('eval_task_count_best_move', 0.0):.0f} "
        f"({metrics.get('eval_best_move_valid_prediction_rate', 0.0):.4f})"
    )


def _build_wandb_run_config(
    *,
    args: argparse.Namespace,
    finetune: Any,
    dataset_dir: Optional[Path],
    final_eval_splits: list[str],
    checkpoint_avg_splits: list[str],
    train_rows: int,
    val_rows: int,
    eval_temperature: float,
    eval_top_p: float,
    eval_reasoning: bool,
    eval_fixed_subset_size: int,
    eval_fixed_subset_seed: int,
) -> dict[str, Any]:
    return {
        "config": args.config,
        "env_file": args.env_file,
        "api_key_env_var": str(args.api_key_env_var or ""),
        "base_url": args.base_url,
        "dataset_source": args.dataset_source,
        "dataset_dir": str(dataset_dir) if dataset_dir is not None else "",
        "hf_dataset_repo_id": args.hf_dataset_repo_id,
        "hf_dataset_revision": args.hf_dataset_revision,
        "hf_cache_dir": args.hf_cache_dir,
        "train_split": args.train_split,
        "train_max_samples": args.train_max_samples,
        "train_subset_seed": args.train_subset_seed,
        "val_split": args.val_split,
        "final_eval_splits": final_eval_splits,
        "checkpoint_avg_splits": checkpoint_avg_splits,
        "checkpoint_avg_metric": args.checkpoint_avg_metric,
        "train_rows": train_rows,
        "val_rows": val_rows,
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
        "max_tokens_by_task": dict(args.max_tokens_by_task),
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
        "best_move_reward_mode": args.best_move_reward_mode,
        "best_move_wrong_rank_scale": args.best_move_wrong_rank_scale,
        "best_move_center_not_optimal_ratio": args.best_move_center_not_optimal_ratio,
        "center_bias_gate_enabled": bool(args.center_bias_gate_enabled),
        "center_bias_gate_threshold": args.center_bias_gate_threshold,
        "center_bias_gate_after_evals": args.center_bias_gate_after_evals,
        "center_bias_gate_min_best_move_samples": args.center_bias_gate_min_best_move_samples,
        "intra_task_sampling_json": dict(args.intra_task_sampling),
        "checkpoint_ranking_output": args.checkpoint_ranking_output,
        "auto_benchmark_best_checkpoint": bool(args.auto_benchmark_best_checkpoint),
        "auto_benchmark_config": str(args.auto_benchmark_config),
        "auto_benchmark_output_json": str(args.auto_benchmark_output_json or ""),
        "auto_benchmark_predictions_jsonl": str(args.auto_benchmark_predictions_jsonl or ""),
        "wandb_log_profile": str(args.wandb_log_profile),
    }


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    args.env_file = _resolve_env_file(args.env_file)
    args.async_checkpoint_eval_dir = str(
        _resolve_path(args.async_checkpoint_eval_dir, default=_repo_relative("outputs", "async_checkpoint_eval"))
    )
    args.api_key_env_var = str(args.api_key_env_var or "").strip()
    show_progress = _progress_enabled(args.no_progress)

    load_dotenv(args.env_file, override=False)
    args.api_key = _resolve_api_key(
        explicit_api_key=args.api_key,
        api_key_env_var=args.api_key_env_var,
    )
    if not args.base_url:
        args.base_url = os.environ.get("TUNA_BASE_URL", DEFAULT_BASE_URL)

    _validate_args(args)
    _warn_on_unsafe_mode_combo(off_policy=bool(args.off_policy), reasoning=bool(args.reasoning))
    if not args.api_key:
        if args.api_key_env_var:
            raise ValueError(
                "Missing Moondream API key. Pass --api-key, set "
                f"{args.api_key_env_var} in {args.env_file}, or set MOONDREAM_API_KEY."
            )
        raise ValueError("MOONDREAM_API_KEY is required")
    args.hf_token = resolve_hf_token(args.hf_token)

    if args.eval_max_samples is not None and args.eval_max_samples <= 0:
        args.eval_max_samples = None
    if args.train_max_samples is not None and args.train_max_samples <= 0:
        args.train_max_samples = None

    eval_temperature = float(args.temperature) if args.eval_temperature is None else float(args.eval_temperature)
    eval_top_p = float(args.top_p) if args.eval_top_p is None else float(args.eval_top_p)
    eval_reasoning = bool(args.reasoning) if args.eval_reasoning is None else bool(args.eval_reasoning)
    eval_fixed_subset_size = int(args.eval_fixed_subset_size)
    eval_fixed_subset_seed = int(args.eval_fixed_subset_seed)
    train_subset_seed = int(args.seed) if int(args.train_subset_seed) < 0 else int(args.train_subset_seed)

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
    dataset_load_kwargs = _build_dataset_load_kwargs(args, dataset_dir=dataset_dir)

    train_examples = _load_split_examples(
        split_name=args.train_split,
        **dataset_load_kwargs,
    )
    train_examples = _subset_examples_deterministically(
        train_examples,
        split_name=args.train_split,
        max_samples=args.train_max_samples,
        subset_seed=train_subset_seed,
    )
    if args.train_max_samples is not None:
        print(
            f"train subset active: split={args.train_split} "
            f"kept={len(train_examples)} max_samples={args.train_max_samples} "
            f"seed={train_subset_seed}"
        )
    val_examples = _load_split_examples(
        split_name=args.val_split,
        **dataset_load_kwargs,
    )
    train_examples_by_task: dict[str, list[QAExample]] = {}
    for item in train_examples:
        train_examples_by_task.setdefault(item.task_type, []).append(item)
    if not train_examples_by_task:
        raise ValueError("train split has no task-indexable examples")
    intra_task_sampling_groups = _prepare_intra_task_sampling_groups(
        train_examples_by_task=train_examples_by_task,
        intra_task_sampling=args.intra_task_sampling,
        best_move_center_not_optimal_ratio=float(args.best_move_center_not_optimal_ratio),
    )
    if args.intra_task_sampling:
        for task_name, strategy in sorted(args.intra_task_sampling.items()):
            rows = train_examples_by_task.get(task_name, [])
            if not rows:
                print(
                    f"intra-task sampling ignored for task={task_name}: no train rows available"
                )
                continue
            group = intra_task_sampling_groups.get(task_name)
            if group is None:
                print(
                    f"intra-task sampling ignored for task={task_name}: strategy={strategy} "
                    "requires >=2 non-empty buckets"
                )
                continue
            bucket_sizes = {
                key: len(group["buckets"][key])
                for key in group["bucket_keys"]
                if isinstance(group["buckets"].get(key), list)
            }
            print(
                f"intra-task sampling active for task={task_name}: strategy={strategy} "
                f"buckets={bucket_sizes} "
                f"weights={group.get('bucket_pick_weights', {})}"
            )

    sampling_tasks = sorted(train_examples_by_task.keys())
    sampling_weights = _weights_for_sampling_tasks(
        tasks=sampling_tasks,
        task_sampling_weights=args.task_sampling_weights,
    )
    if not sampling_tasks:
        raise ValueError("no tasks available for weighted sampling")

    active_eval_tasks = _active_tasks_from_sampling_weights(args.task_sampling_weights)
    print(
        "eval task filter (weight>0): "
        + ", ".join(sorted(active_eval_tasks))
    )
    val_before_filter = len(val_examples)
    val_examples = _filter_examples_by_active_tasks(val_examples, active_tasks=active_eval_tasks)
    if len(val_examples) != val_before_filter:
        print(
            f"filtered eval split={args.val_split}: kept={len(val_examples)} "
            f"dropped={val_before_filter - len(val_examples)}"
        )

    split_examples_cache: dict[str, list[QAExample]] = {
        args.val_split: val_examples,
    }

    def _examples_for_split(split_name: str) -> list[QAExample]:
        if split_name not in split_examples_cache:
            raw_examples = _load_split_examples(
                split_name=split_name,
                **dataset_load_kwargs,
            )
            filtered_examples = _filter_examples_by_active_tasks(
                raw_examples,
                active_tasks=active_eval_tasks,
            )
            if len(filtered_examples) != len(raw_examples):
                print(
                    f"filtered eval split={split_name}: kept={len(filtered_examples)} "
                    f"dropped={len(raw_examples) - len(filtered_examples)}"
                )
            split_examples_cache[split_name] = filtered_examples
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

    eval_kwargs = _build_eval_kwargs(
        args,
        eval_temperature=eval_temperature,
        eval_top_p=eval_top_p,
        eval_reasoning=eval_reasoning,
        show_progress=show_progress,
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
        config=_build_wandb_run_config(
            args=args,
            finetune=finetune,
            dataset_dir=dataset_dir,
            final_eval_splits=final_eval_splits,
            checkpoint_avg_splits=checkpoint_avg_splits,
            train_rows=len(train_examples),
            val_rows=len(val_examples),
            eval_temperature=eval_temperature,
            eval_top_p=eval_top_p,
            eval_reasoning=eval_reasoning,
            eval_fixed_subset_size=eval_fixed_subset_size,
            eval_fixed_subset_seed=eval_fixed_subset_seed,
        ),
    )
    run.summary["finetune_id"] = finetune.finetune_id

    best_metric_value: Optional[float] = None
    best_step: Optional[int] = None
    best_eval_reward_seen: Optional[float] = None
    low_parse_streak = 0
    reward_drop_streak = 0
    recent_eval_rewards: list[float] = []
    completed_steps = 0
    periodic_eval_count = 0
    training_status = _default_training_status(
        target_steps=int(args.num_steps),
        early_stop_mode=str(args.early_stop_mode),
    )
    checkpoint_eval_history: list[dict[str, Any]] = []
    replay_buffer: deque[Any] = deque(maxlen=int(args.off_policy_buffer_size))
    async_eval_jobs: list[DispatchHandle] = []
    async_eval_success_count = 0

    def _run_checkpoint_eval(
        *,
        step_for_log: int,
        seed_base: int,
        stage_label: str,
    ) -> dict[str, dict[str, float]]:
        by_split: dict[str, dict[str, float]] = {}
        for split_idx, split_name in enumerate(checkpoint_avg_splits):
            split_metrics = _run_split_eval(
                finetune=finetune,
                examples=checkpoint_avg_examples[split_name],
                split_name=split_name,
                seed=seed_base + split_idx,
                eval_kwargs=eval_kwargs,
                fixed_indices=fixed_eval_indices_by_split.get(split_name),
            )
            by_split[split_name] = split_metrics
            if _should_log_prefixed_eval_streams(wandb_log_profile=args.wandb_log_profile):
                prefix = f"checkpoint_{_sanitize_split_name(split_name)}_"
                split_wandb_metrics = _select_eval_wandb_metrics(split_metrics)
                wandb.log({f"{prefix}{k}": v for k, v in split_wandb_metrics.items()}, step=step_for_log)
            _print_eval_summary(
                label=f"{stage_label} eval",
                metrics=split_metrics,
                step=step_for_log,
                split_name=split_name,
            )

        avg_checkpoint_metric = (
            float(fmean(float(m.get(args.checkpoint_avg_metric, 0.0)) for m in by_split.values()))
            if by_split
            else 0.0
        )
        avg_eval_reward_mean = (
            float(fmean(float(m.get("eval_reward_mean", 0.0)) for m in by_split.values()))
            if by_split
            else 0.0
        )
        wandb.log(
            _checkpoint_wandb_payload(
                avg_checkpoint_metric=avg_checkpoint_metric,
                avg_eval_reward_mean=avg_eval_reward_mean,
                wandb_log_profile=args.wandb_log_profile,
            ),
            step=step_for_log,
        )
        print(
            f"{stage_label} checkpoint average step={step_for_log} "
            f"{args.checkpoint_avg_metric}={avg_checkpoint_metric:.4f}"
        )

        checkpoint_eval_history.append(
            {
                "step": int(step_for_log),
                "avg_checkpoint_metric": avg_checkpoint_metric,
                "avg_eval_reward_mean": avg_eval_reward_mean,
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
            baseline_metrics = _run_split_eval(
                finetune=finetune,
                examples=val_examples,
                split_name=args.val_split,
                seed=args.seed + 151,
                eval_kwargs=eval_kwargs,
                fixed_indices=fixed_eval_indices_by_split.get(args.val_split),
            )
        baseline_wandb_metrics = _select_eval_wandb_metrics(baseline_metrics)
        if _should_log_prefixed_eval_streams(wandb_log_profile=args.wandb_log_profile):
            wandb.log({f"baseline_{k}": v for k, v in baseline_wandb_metrics.items()}, step=args.resume_step)

        baseline_metric = float(baseline_metrics.get(args.best_metric, 0.0))
        best_metric_value = baseline_metric
        best_step = args.resume_step
        best_eval_reward_seen = float(baseline_metrics.get("eval_reward_mean", 0.0))
        _print_eval_summary(
            label="baseline eval",
            metrics=baseline_metrics,
            step=args.resume_step,
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
        if async_eval_jobs:
            async_eval_jobs, completed_async_results = poll_checkpoint_eval_jobs(async_eval_jobs)
            (
                best_metric_value,
                best_step,
                best_eval_reward_seen,
                completed_successes,
            ) = _ingest_async_checkpoint_eval_results(
                args=args,
                checkpoint_eval_history=checkpoint_eval_history,
                results=completed_async_results,
                log_step=int(global_step),
                best_metric_value=best_metric_value,
                best_step=best_step,
                best_eval_reward_seen=best_eval_reward_seen,
            )
            async_eval_success_count += int(completed_successes)

        sampled_tasks = rng.choices(sampling_tasks, weights=sampling_weights, k=args.batch_size)
        batch = [
            _sample_training_example(
                task_name=task_name,
                train_examples_by_task=train_examples_by_task,
                intra_task_sampling_groups=intra_task_sampling_groups,
                rng=rng,
            )
            for task_name in sampled_tasks
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
        best_move_rollout_count = 0
        best_move_valid_prediction_count = 0

        for item, result in zip(active_examples, results):
            if not result.rollouts:
                continue

            rewards: list[float] = []
            for rollout in result.rollouts:
                outcome = _score_rollout_for_example(
                    rollout,
                    item,
                    best_move_optimal_reward=args.best_move_optimal_reward,
                    best_move_reward_mode=args.best_move_reward_mode,
                    best_move_wrong_rank_scale=args.best_move_wrong_rank_scale,
                )
                rewards.append(outcome.reward)
                rewards_all.append(outcome.reward)
                if outcome.json_object_parsed:
                    object_parses += 1
                if outcome.parse_success:
                    parse_successes += 1
                if item.task_type == "best_move":
                    best_move_rollout_count += 1
                    if outcome.best_move_valid_prediction:
                        best_move_valid_prediction_count += 1

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

        reward_mean = float(fmean(rewards_all)) if rewards_all else 0.0
        reward_var = float(pvariance(rewards_all)) if len(rewards_all) > 1 else 0.0
        object_parse_rate = object_parses / max(1, len(rewards_all))
        parse_rate = parse_successes / max(1, len(rewards_all))

        train_metrics: dict[str, float] = {
            "reward_mean": reward_mean,
            "reward_var": reward_var,
            "train_json_object_rate": object_parse_rate,
            "train_json_parse_rate": parse_rate,
            "train_best_move_valid_prediction_count": float(best_move_valid_prediction_count),
            "train_best_move_valid_prediction_rate": (
                best_move_valid_prediction_count / max(1, best_move_rollout_count)
            ),
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
            f"best_move_valid={best_move_valid_prediction_count}/{best_move_rollout_count} "
            f"({train_metrics['train_best_move_valid_prediction_rate']:.4f}) "
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
            periodic_eval_count += 1
            if args.async_checkpoint_eval:
                saved_step = _save_checkpoint_step(
                    finetune=finetune,
                    context=f"periodic eval step={global_step}",
                )
                if saved_step is not None:
                    job = dispatch_checkpoint_eval(
                        trainer="ttt_query_rl",
                        finetune_id=str(finetune.finetune_id),
                        checkpoint_step=int(saved_step),
                        selection_metric=str(args.checkpoint_avg_metric),
                        base_dir=str(args.async_checkpoint_eval_dir),
                        command_builder=lambda metrics_json_path, predictions_jsonl_path, _stdout_log_path: _build_async_checkpoint_eval_command(
                            args=args,
                            finetune_id=str(finetune.finetune_id),
                            checkpoint_step=int(saved_step),
                            checkpoint_avg_splits=list(checkpoint_avg_splits),
                            active_eval_tasks=set(active_eval_tasks),
                            eval_temperature=float(eval_temperature),
                            eval_top_p=float(eval_top_p),
                            eval_reasoning=bool(eval_reasoning),
                            metrics_json_path=metrics_json_path,
                            predictions_jsonl_path=predictions_jsonl_path,
                        ),
                        metadata={"step_for_log": int(global_step)},
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
                eval_by_split = _run_checkpoint_eval(
                    step_for_log=global_step,
                    seed_base=args.seed + 1000 + (global_step * 13),
                    stage_label="checkpoint",
                )
                eval_metrics = eval_by_split.get(args.val_split)
                if eval_metrics is None:
                    eval_metrics = _run_split_eval(
                        finetune=finetune,
                        examples=val_examples,
                        split_name=args.val_split,
                        seed=args.seed + 1000 + global_step,
                        eval_kwargs=eval_kwargs,
                        fixed_indices=fixed_eval_indices_by_split.get(args.val_split),
                    )
                wandb.log(_select_eval_wandb_metrics(eval_metrics), step=global_step)
                metric_value = float(eval_metrics.get(args.best_metric, 0.0))
                _print_eval_summary(
                    label="eval",
                    metrics=eval_metrics,
                    step=global_step,
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

                center_prediction_rate = float(
                    eval_metrics.get("eval_best_move_center_prediction_rate", 0.0)
                )
                best_move_eval_samples = float(
                    eval_metrics.get("eval_task_count_best_move", 0.0)
                )
                if _should_trigger_center_bias_gate(
                    enabled=bool(args.center_bias_gate_enabled),
                    eval_index=int(periodic_eval_count),
                    gate_after_evals=int(args.center_bias_gate_after_evals),
                    center_prediction_rate=center_prediction_rate,
                    gate_threshold=float(args.center_bias_gate_threshold),
                    best_move_samples=best_move_eval_samples,
                    min_best_move_samples=int(args.center_bias_gate_min_best_move_samples),
                ):
                    training_status.update(
                        {
                            "stopped_early": True,
                            "stop_reason": "collapse_center_bias",
                            "completed_steps": int(completed_steps),
                            "collapse_detected": True,
                        }
                    )
                    print(
                        f"center-bias gate triggered at step={global_step}: "
                        f"center_rate={center_prediction_rate:.4f} "
                        f"threshold={float(args.center_bias_gate_threshold):.4f} "
                        f"best_move_samples={best_move_eval_samples:.0f}"
                    )
                    break

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
    if args.async_checkpoint_eval and bool(args.async_checkpoint_eval_drain_on_exit):
        completed_async_results = drain_checkpoint_eval_jobs(async_eval_jobs)
        (
            best_metric_value,
            best_step,
            best_eval_reward_seen,
            completed_successes,
        ) = _ingest_async_checkpoint_eval_results(
            args=args,
            checkpoint_eval_history=checkpoint_eval_history,
            results=completed_async_results,
            log_step=int(args.resume_step + completed_steps),
            best_metric_value=best_metric_value,
            best_step=best_step,
            best_eval_reward_seen=best_eval_reward_seen,
        )
        async_eval_success_count += int(completed_successes)

    final_eval_step = args.resume_step + int(completed_steps)
    if args.skip_final_eval:
        print("skip_final_eval=true; skipping final split eval pass.")
    else:
        for idx, (split_name, split_examples) in enumerate(final_eval_examples.items()):
            eval_metrics = _run_split_eval(
                finetune=finetune,
                examples=split_examples,
                split_name=split_name,
                seed=args.seed + 5000 + idx,
                eval_kwargs=eval_kwargs,
                fixed_indices=fixed_eval_indices_by_split.get(split_name),
            )
            prefix = _metric_prefix_for_split(split_name)
            final_eval_wandb_metrics = _select_eval_wandb_metrics(eval_metrics)
            if _should_log_prefixed_eval_streams(wandb_log_profile=args.wandb_log_profile):
                wandb.log({f"{prefix}{k}": v for k, v in final_eval_wandb_metrics.items()}, step=final_eval_step)

            _print_eval_summary(
                label="final eval",
                metrics=eval_metrics,
                split_name=split_name,
            )

    ranking_payload = _build_checkpoint_ranking_payload(
        finetune_id=finetune.finetune_id,
        checkpoint_avg_metric=args.checkpoint_avg_metric,
        checkpoint_avg_splits=checkpoint_avg_splits,
        checkpoint_eval_history=checkpoint_eval_history,
        training_status=training_status,
    )
    checkpoint_ranking = ranking_payload["rankings"]
    best_avg_checkpoint_metric = float(ranking_payload.get("best_avg_checkpoint_metric", 0.0))
    best_avg_checkpoint_metric_step = int(ranking_payload.get("best_avg_checkpoint_metric_step", -1))
    best_avg_eval_reward = float(ranking_payload["best_avg_eval_reward"])
    best_avg_eval_reward_step = int(ranking_payload["best_avg_eval_reward_step"])
    if checkpoint_ranking:
        print(
            "best checkpoint by configured average metric: "
            f"step={best_avg_checkpoint_metric_step} "
            f"{args.checkpoint_avg_metric}={best_avg_checkpoint_metric:.4f} "
            f"avg_eval_reward_mean={best_avg_eval_reward:.4f} "
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

    auto_benchmark_ran = False
    auto_benchmark_success = False
    auto_benchmark_checkpoint_step: Optional[int] = None
    auto_benchmark_metrics_path: Optional[Path] = None
    auto_benchmark_predictions_path: Optional[Path] = None
    auto_benchmark_metrics: Optional[dict[str, Any]] = None
    if bool(args.auto_benchmark_best_checkpoint):
        auto_benchmark_checkpoint_step = _select_checkpoint_step_for_auto_benchmark(
            ranking_payload=ranking_payload,
            fallback_step=best_step,
        )
        auto_benchmark_ran = True
        try:
            (
                auto_benchmark_success,
                auto_benchmark_metrics_path,
                auto_benchmark_predictions_path,
                auto_benchmark_metrics,
            ) = _run_auto_benchmark(
                args=args,
                finetune_id=finetune.finetune_id,
                checkpoint_step=auto_benchmark_checkpoint_step,
                dataset_dir=dataset_dir,
                output_json_override=str(args.auto_benchmark_output_json),
                predictions_jsonl_override=str(args.auto_benchmark_predictions_jsonl),
            )
        except Exception as exc:
            auto_benchmark_success = False
            print(f"auto benchmark failed with unexpected exception: {exc}")
    else:
        print("auto benchmark disabled (auto_benchmark_best_checkpoint=false).")

    run.summary["best_metric_name"] = args.best_metric
    run.summary["best_metric_value"] = float(best_metric_value or 0.0)
    run.summary["best_metric_step"] = int(best_step if best_step is not None else -1)
    run.summary["wandb_log_profile"] = str(args.wandb_log_profile)
    run.summary["best_move_center_not_optimal_ratio"] = float(args.best_move_center_not_optimal_ratio)
    run.summary["center_bias_gate_enabled"] = bool(args.center_bias_gate_enabled)
    run.summary["center_bias_gate_threshold"] = float(args.center_bias_gate_threshold)
    run.summary["finetune_id"] = finetune.finetune_id
    run.summary["train_rows"] = len(train_examples)
    run.summary["train_max_samples"] = int(args.train_max_samples or 0)
    run.summary["train_subset_seed"] = int(train_subset_seed)
    run.summary["val_rows"] = len(val_examples)
    run.summary["checkpoint_avg_metric_name"] = args.checkpoint_avg_metric
    run.summary["best_avg_checkpoint_metric"] = float(best_avg_checkpoint_metric)
    run.summary["best_avg_checkpoint_metric_step"] = int(best_avg_checkpoint_metric_step)
    run.summary["best_avg_eval_reward"] = float(best_avg_eval_reward)
    run.summary["best_avg_eval_reward_step"] = int(best_avg_eval_reward_step)
    run.summary["best_avg_eval_reward_splits"] = ",".join(checkpoint_avg_splits)
    run.summary["checkpoint_ranking_output"] = str(ranking_output_path)
    run.summary["stopped_early"] = bool(training_status.get("stopped_early"))
    run.summary["early_stop_reason"] = str(training_status.get("stop_reason", ""))
    run.summary["completed_steps"] = int(training_status.get("completed_steps", completed_steps))
    run.summary["target_steps"] = int(training_status.get("target_steps", args.num_steps))
    run.summary["collapse_detected"] = bool(training_status.get("collapse_detected"))
    run.summary["auto_benchmark_enabled"] = bool(args.auto_benchmark_best_checkpoint)
    run.summary["auto_benchmark_ran"] = bool(auto_benchmark_ran)
    run.summary["auto_benchmark_success"] = bool(auto_benchmark_success)
    run.summary["auto_benchmark_checkpoint_step"] = int(
        auto_benchmark_checkpoint_step if auto_benchmark_checkpoint_step is not None else -1
    )
    run.summary["auto_benchmark_output_json"] = (
        str(auto_benchmark_metrics_path) if auto_benchmark_metrics_path is not None else ""
    )
    run.summary["auto_benchmark_predictions_jsonl"] = (
        str(auto_benchmark_predictions_path) if auto_benchmark_predictions_path is not None else ""
    )
    run.summary["async_checkpoint_eval_enabled"] = bool(args.async_checkpoint_eval)
    run.summary["async_checkpoint_eval_success_count"] = int(async_eval_success_count)
    if auto_benchmark_metrics is not None:
        for key in (
            "eval_reward_mean",
            "eval_json_parse_rate",
            "eval_best_move_set_accuracy",
            "eval_best_move_canonical_accuracy",
        ):
            if isinstance(auto_benchmark_metrics.get(key), (int, float)):
                run.summary[f"auto_benchmark_{key}"] = float(auto_benchmark_metrics[key])

    run.finish()
    client.close()

    print(
        f"done. finetune_id={finetune.finetune_id} best_{args.best_metric}={best_metric_value} "
        f"best_step={best_step} best_avg_checkpoint_metric={best_avg_checkpoint_metric} "
        f"best_avg_checkpoint_metric_step={best_avg_checkpoint_metric_step} "
        f"best_avg_eval_reward={best_avg_eval_reward} "
        f"auto_benchmark_success={auto_benchmark_success} "
        f"auto_benchmark_step={auto_benchmark_checkpoint_step} "
        f"stopped_early={training_status.get('stopped_early')} "
        f"stop_reason={training_status.get('stop_reason', '')}"
    )


if __name__ == "__main__":
    main()
