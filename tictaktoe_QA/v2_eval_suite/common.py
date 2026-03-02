"""Shared helpers for the V2-only evaluation suite."""

from __future__ import annotations

import csv
import fnmatch
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from tictaktoe_QA import train_ttt_query_rl as train_utils
from tictaktoe_QA.task_schema import normalize_task_type

V2_HF_DATASET_REPO_ID = "maxs-m87/tictactoe-qa-v2"
HARD_TASK_TYPES: tuple[str, ...] = (
    "best_move",
    "available_moves_count",
    "available_moves_list",
)
BEST_MOVE_BUCKET_ORDER: tuple[str, ...] = (
    "best_move",
    "second_best",
    "third_best",
    "fourth_plus",
    "invalid_move",
    "improper_response_format",
    "request_error",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def pkg_relative(*parts: str) -> Path:
    return package_root().joinpath(*parts)


def resolve_path(raw_path: str, *, search_roots: Sequence[Path] = ()) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()

    roots: list[Path] = []
    for root in (Path.cwd(), *search_roots):
        root_resolved = Path(root).resolve()
        if root_resolved not in roots:
            roots.append(root_resolved)

    for root in roots:
        candidate = (root / path).resolve()
        if candidate.exists():
            return candidate

    return (Path.cwd() / path).resolve()


def load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in: {path}")
    return payload


def validate_config_keys(
    config: dict[str, Any],
    *,
    allowed_keys: set[str],
    config_path: Path,
) -> None:
    unknown = sorted(k for k in config.keys() if k not in allowed_keys)
    if unknown:
        raise ValueError(
            f"Unknown config key(s) in {config_path}: {unknown}. "
            "Remove typos or update script support."
        )


def cfg_str(config: dict[str, Any], key: str, fallback: str) -> str:
    value = config.get(key, fallback)
    if value is None:
        return fallback
    return str(value)


def cfg_int(config: dict[str, Any], key: str, fallback: int) -> int:
    value = config.get(key, fallback)
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def cfg_float(config: dict[str, Any], key: str, fallback: float) -> float:
    value = config.get(key, fallback)
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def cfg_bool(config: dict[str, Any], key: str, fallback: bool) -> bool:
    value = config.get(key, fallback)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return fallback


def cfg_list_str(config: dict[str, Any], key: str, fallback: list[str]) -> list[str]:
    value = config.get(key)
    if value is None:
        return list(fallback)
    if isinstance(value, str):
        return parse_pattern_list(value)
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                out.append(text)
        return out
    return list(fallback)


def parse_pattern_list(raw_value: str | None) -> list[str]:
    if not raw_value:
        return []
    pieces = re.split(r"[\s,]+", str(raw_value))
    return [piece for piece in (p.strip() for p in pieces) if piece]


def matches_any_pattern(value: str, patterns: Sequence[str]) -> bool:
    if not patterns:
        return True
    basename = Path(value).name
    for pattern in patterns:
        if fnmatch.fnmatch(value, pattern) or fnmatch.fnmatch(basename, pattern):
            return True
    return False


def normalize_task_types(raw_values: Optional[Sequence[str]]) -> list[str]:
    if not raw_values:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for value in raw_values:
        for piece in str(value).split(","):
            candidate = piece.strip()
            if not candidate:
                continue
            normalized = normalize_task_type(candidate)
            if normalized in seen:
                continue
            seen.add(normalized)
            out.append(normalized)
    return out


def slugify(text: str) -> str:
    lowered = str(text).strip().lower()
    slug = re.sub(r"[^a-z0-9._-]+", "_", lowered)
    slug = slug.strip("_")
    return slug or "run"


def utc_timestamp_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def write_csv(path: Path, *, fieldnames: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object in {path}:{line_number}")
            yield payload


def load_rows_by_id(*, dataset_dir: Path, split: str) -> dict[str, dict[str, Any]]:
    split_path = dataset_dir / "jsonl" / f"{split}.jsonl"
    if not split_path.exists():
        raise FileNotFoundError(f"split JSONL not found: {split_path}")
    by_id: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(split_path):
        row_id = str(row.get("row_id", "")).strip()
        if not row_id:
            continue
        by_id[row_id] = row
    return by_id


def is_request_error_record(record: dict[str, Any]) -> bool:
    return str(record.get("status", "")).strip().lower() == "request_error"


def scores_by_move_from_row(row: dict[str, Any]) -> dict[int, tuple[int, int]]:
    parsed = train_utils._scores_by_move_from_json(str(row.get("scores_by_move_json", "")))
    return {int(move): (int(value), int(depth)) for move, value, depth in parsed}


def dense_rank_for_move(move: int, *, scores_by_move: dict[int, tuple[int, int]]) -> Optional[int]:
    pred = scores_by_move.get(int(move))
    if pred is None:
        return None

    unique_keys = {
        train_utils._best_move_rank_key(value=int(value), depth=int(depth))
        for value, depth in scores_by_move.values()
    }
    sorted_keys = sorted(unique_keys, reverse=True)
    pred_key = train_utils._best_move_rank_key(value=int(pred[0]), depth=int(pred[1]))
    for idx, key in enumerate(sorted_keys, start=1):
        if key == pred_key:
            return idx
    return None


def bucket_from_dense_rank(rank: int) -> str:
    if rank <= 1:
        return "best_move"
    if rank == 2:
        return "second_best"
    if rank == 3:
        return "third_best"
    return "fourth_plus"


def classify_best_move_prediction(
    prediction: dict[str, Any],
    *,
    ground_truth_row: dict[str, Any],
) -> str:
    if is_request_error_record(prediction):
        return "request_error"

    if not bool(prediction.get("json_object_parsed", False)):
        return "improper_response_format"

    if not bool(prediction.get("parse_success", False)):
        return "improper_response_format"

    move = train_utils._move_from_payload_obj(prediction.get("prediction_json"))
    if move is None:
        return "improper_response_format"

    scores_by_move = scores_by_move_from_row(ground_truth_row)
    if move not in scores_by_move:
        return "invalid_move"

    rank = dense_rank_for_move(int(move), scores_by_move=scores_by_move)
    if rank is None:
        return "invalid_move"

    return bucket_from_dense_rank(rank)


def _path_matches_v2(path: Path, *, dataset_dir: Path) -> bool:
    normalized = path.resolve().as_posix()
    if "/synth_dataset/outputs/v2" in normalized:
        return True
    dataset_norm = dataset_dir.resolve().as_posix()
    if normalized == dataset_norm:
        return True
    return normalized.startswith(dataset_norm + "/")


def is_v2_metrics_payload(
    metrics_payload: dict[str, Any] | None,
    *,
    dataset_dir: Path,
    expected_hf_repo_id: str = V2_HF_DATASET_REPO_ID,
) -> bool:
    if not isinstance(metrics_payload, dict):
        return False

    hf_repo = str(metrics_payload.get("hf_dataset_repo_id", "")).strip()
    if hf_repo == expected_hf_repo_id:
        return True

    raw_dataset_dir = str(metrics_payload.get("dataset_dir", "")).strip()
    if raw_dataset_dir:
        try:
            resolved = Path(raw_dataset_dir).expanduser().resolve()
        except OSError:
            return False
        return _path_matches_v2(resolved, dataset_dir=dataset_dir)

    return False


def extract_openrouter_answer_text(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""

    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""

    first = choices[0]
    if not isinstance(first, dict):
        return ""

    message = first.get("message")
    if not isinstance(message, dict):
        return ""

    content = message.get("content")
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
            elif isinstance(item, str) and item:
                parts.append(item)
        return "\n".join(parts)

    return ""


def truncate_text(text: str, *, limit: int = 800) -> str:
    raw = str(text)
    if len(raw) <= limit:
        return raw
    return raw[:limit] + "...<truncated>"


def truncate_json_payload(payload: Any, *, limit: int = 1500) -> str:
    try:
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        text = str(payload)
    return truncate_text(text, limit=limit)
