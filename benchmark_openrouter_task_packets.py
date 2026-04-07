#!/usr/bin/env python3
"""Benchmark GPT-5.4 via OpenRouter against task packet datasets."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import socket
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
from dotenv import load_dotenv
from PIL import Image
from scipy.optimize import linear_sum_assignment

import task_packet_benchmark_common as task_packet_common


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "outputs" / "task_sample_packets" / "openrouter_gpt_5_4_config.json"
DEFAULT_API_BASE = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-5.4"

_JSON_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_README_SAMPLE_COUNT_PATTERN = re.compile(r"sample count:\s*`?(\d+)`?", re.IGNORECASE)
_AERIAL_INDEX_PREFIX_PATTERN = re.compile(r"^\d+_")
_STATE_FARM_PACKET_IMAGE_PATTERN = re.compile(r"^(?P<index>\d+)_+(?P<sample_id>.+)\.(?:jpg|jpeg|png)$", re.IGNORECASE)


@dataclass(frozen=True)
class Box:
    x_min: float
    y_min: float
    x_max: float
    y_max: float


@dataclass(frozen=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True)
class Sample:
    sample_index: int
    sample_id: str
    source_image_path: str
    prompt: str
    task_type: str
    notes: str
    timestamp: str
    image: Image.Image
    ground_truth_boxes: list[Box]
    base_record: dict[str, Any]


@dataclass(frozen=True)
class RequestVariant:
    name: str
    detail: str
    use_response_format: bool
    require_parameters: bool
    use_response_healing: bool


@dataclass(frozen=True)
class TaskSpec:
    name: str
    skill: str
    prompt: str
    task_dir: Path
    source_kind: str
    dataset_id: Optional[str]
    split: Optional[str]
    iou_threshold: float
    reference_metrics_path: Optional[Path]
    reference_samples_path: Optional[Path]
    reference_compare_path: Optional[Path]
    reference_readme_path: Optional[Path]
    aerial_subset_dataset_path: Optional[Path]
    aerial_packet_path: Optional[Path]
    aerial_benchmark_path: Optional[Path]

    @property
    def artifact_stem(self) -> str:
        return f"openrouter_gpt_5_4_{self.skill}"

    @property
    def metrics_path(self) -> Path:
        return self.task_dir / f"{self.artifact_stem}.metrics.json"

    @property
    def predictions_path(self) -> Path:
        return self.task_dir / f"{self.artifact_stem}.predictions.jsonl"

    @property
    def packet_samples_path(self) -> Path:
        return self.task_dir / f"{self.artifact_stem}.packet_samples.json"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cfg_str(config: dict[str, Any], key: str, fallback: str) -> str:
    value = config.get(key, fallback)
    return fallback if value is None else str(value)


def _cfg_float(config: dict[str, Any], key: str, fallback: float) -> float:
    value = config.get(key, fallback)
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _cfg_int(config: dict[str, Any], key: str, fallback: int) -> int:
    value = config.get(key, fallback)
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _cfg_list_str(config: dict[str, Any], key: str, fallback: list[str]) -> list[str]:
    value = config.get(key)
    if value is None:
        return list(fallback)
    if isinstance(value, str):
        items = [part.strip() for part in value.split(",")]
        return [item for item in items if item]
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            text = str(item or "").strip()
            if text:
                out.append(text)
        return out or list(fallback)
    return list(fallback)


def _resolve_path(raw_path: str, *, roots: tuple[Path, ...]) -> Path:
    path = Path(str(raw_path or "")).expanduser()
    if path.is_absolute():
        return path.resolve()
    for root in (Path.cwd(), *roots):
        candidate = (root / path).resolve()
        if candidate.exists():
            return candidate
    return (Path.cwd() / path).resolve()


def _resolve_env_file_path(raw_env_file: str, *, repo_root: Path = REPO_ROOT) -> str:
    raw = str(raw_env_file or "").strip()
    candidate_paths: list[Path] = []
    repo_root = repo_root.resolve()

    def add_candidate(raw_candidate: str) -> None:
        candidate = Path(str(raw_candidate or "")).expanduser()
        if not candidate.is_absolute():
            candidate = repo_root / candidate
        candidate_paths.append(candidate.resolve())

    if raw:
        add_candidate(raw)
        if raw == ".env":
            add_candidate(".env.staging")
            add_candidate(".env")
    else:
        add_candidate(".env.staging")
        add_candidate(".env")
    seen: set[str] = set()
    for resolved in candidate_paths:
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        if resolved.exists():
            return str(resolved)
    fallback = Path(raw if raw else ".env").expanduser()
    if not fallback.is_absolute():
        fallback = repo_root / fallback
    return str(fallback.resolve())


def _resolve_openrouter_api_key(explicit_api_key: str, api_key_env_var: str) -> str:
    cli_key = str(explicit_api_key or "").strip()
    if cli_key:
        return cli_key

    preferred_env_var = str(api_key_env_var or "").strip()
    if preferred_env_var:
        preferred_value = os.environ.get(preferred_env_var, "").strip()
        if preferred_value:
            return preferred_value

    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if openrouter_key:
        return openrouter_key

    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if openai_key:
        raise ValueError(
            "OPENROUTER_API_KEY is required for OpenRouter benchmarks. "
            "OPENAI_API_KEY was found but is not valid for OpenRouter auth."
        )

    checked_env = preferred_env_var if preferred_env_var else "OPENROUTER_API_KEY"
    raise ValueError(f"OPENROUTER_API_KEY is required (checked --api-key, {checked_env}, and OPENROUTER_API_KEY)")


def _load_json_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        if config_path == DEFAULT_CONFIG_PATH:
            return {}
        raise FileNotFoundError(f"Config file not found: {config_path}")
    payload = _load_json(config_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a JSON object: {config_path}")
    return payload


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    raw_argv = list(argv) if argv is not None else list(sys.argv[1:])
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    pre_args, _ = pre_parser.parse_known_args(raw_argv)
    config_path = _resolve_path(pre_args.config, roots=(REPO_ROOT,))
    config = _load_json_config(config_path)

    parser = argparse.ArgumentParser(description="Benchmark GPT-5.4 via OpenRouter on task packet datasets.")
    parser.add_argument("--config", default=str(config_path))
    parser.add_argument("--env-file", default=_cfg_str(config, "env_file", ".env"))
    parser.add_argument("--api-key", default=_cfg_str(config, "api_key", ""))
    parser.add_argument("--api-key-env-var", default=_cfg_str(config, "api_key_env_var", "OPENROUTER_API_KEY"))
    parser.add_argument("--api-base", default=_cfg_str(config, "api_base", DEFAULT_API_BASE))
    parser.add_argument("--model", default=_cfg_str(config, "model", DEFAULT_MODEL))
    parser.add_argument("--temperature", type=float, default=_cfg_float(config, "temperature", 0.0))
    parser.add_argument("--top-p", type=float, default=_cfg_float(config, "top_p", 1.0))
    parser.add_argument("--detail", default=_cfg_str(config, "detail", "original"))
    parser.add_argument("--max-tokens-detect", type=int, default=_cfg_int(config, "max_tokens_detect", 400))
    parser.add_argument("--max-tokens-point", type=int, default=_cfg_int(config, "max_tokens_point", 300))
    parser.add_argument("--timeout", type=float, default=_cfg_float(config, "timeout", 90.0))
    parser.add_argument(
        "--retry-429-max-retries",
        type=int,
        default=_cfg_int(config, "retry_429_max_retries", 3),
    )
    parser.add_argument(
        "--retry-429-backoff-s",
        type=float,
        default=_cfg_float(config, "retry_429_backoff_s", 1.0),
    )
    parser.add_argument(
        "--retry-429-max-backoff-s",
        type=float,
        default=_cfg_float(config, "retry_429_max_backoff_s", 12.0),
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Optional subset of fixed task names to run.",
    )
    args = parser.parse_args(raw_argv)
    args.config = str(_resolve_path(args.config, roots=(REPO_ROOT,)))
    args.env_file = _resolve_env_file_path(args.env_file, repo_root=REPO_ROOT) if str(args.env_file).strip() else ""
    if args.tasks is None:
        args.tasks = _cfg_list_str(config, "tasks", ["state_farm", "player_with_ball", "aerial"])
    else:
        args.tasks = [str(item).strip() for item in args.tasks if str(item).strip()]
    return args


def _build_auth_headers(api_key: str) -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key.strip()}",
    }
    referer = os.environ.get("OPENROUTER_HTTP_REFERER", "").strip()
    if referer:
        headers["HTTP-Referer"] = referer
    title = os.environ.get("OPENROUTER_APP_NAME", "").strip()
    if title:
        headers["X-Title"] = title
    return headers


def _to_data_url(image: Image.Image, *, format: str = "JPEG", quality: int = 90) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=format, quality=max(1, min(100, int(quality))))
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    mime = "image/jpeg" if format.upper() == "JPEG" else f"image/{format.lower()}"
    return f"data:{mime};base64,{encoded}"


def _extract_openrouter_answer_text(response_payload: dict[str, Any]) -> str:
    choices = response_payload.get("choices")
    if isinstance(choices, list) and choices:
        first_choice = choices[0]
        if isinstance(first_choice, dict):
            message = first_choice.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    return content.strip()
                if isinstance(content, list):
                    chunks: list[str] = []
                    for item in content:
                        if not isinstance(item, dict):
                            continue
                        text = item.get("text")
                        if isinstance(text, str) and text.strip():
                            chunks.append(text.strip())
                    if chunks:
                        return "\n".join(chunks).strip()
    output_text = response_payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()
    return ""


def _try_parse_json_object(text: str) -> Optional[dict[str, Any]]:
    if not text:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _find_braced_json_object(text: str) -> Optional[dict[str, Any]]:
    for start_idx, ch in enumerate(text):
        if ch != "{":
            continue
        depth = 0
        in_string = False
        escaped = False
        for end_idx in range(start_idx, len(text)):
            cur = text[end_idx]
            if in_string:
                if escaped:
                    escaped = False
                elif cur == "\\":
                    escaped = True
                elif cur == '"':
                    in_string = False
                continue
            if cur == '"':
                in_string = True
                continue
            if cur == "{":
                depth += 1
                continue
            if cur == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start_idx : end_idx + 1]
                    payload = _try_parse_json_object(candidate)
                    if payload is not None:
                        return payload
                if depth < 0:
                    break
    return None


def _extract_first_json_object(text: str) -> Optional[dict[str, Any]]:
    stripped = str(text or "").strip()
    if not stripped:
        return None
    direct = _try_parse_json_object(stripped)
    if direct is not None:
        return direct
    for match in _JSON_FENCE_PATTERN.finditer(stripped):
        fenced = str(match.group(1) or "").strip()
        if not fenced:
            continue
        payload = _try_parse_json_object(fenced)
        if payload is not None:
            return payload
        payload = _find_braced_json_object(fenced)
        if payload is not None:
            return payload
    return _find_braced_json_object(stripped)


def _normalize_coord(value: float, *, size: int) -> float:
    coord = float(value)
    if abs(coord) > 1.5 and size > 0:
        coord = coord / float(size)
    return max(0.0, min(1.0, coord))


def _coerce_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_box_item(item: Any, *, width: int, height: int) -> Optional[Box]:
    if not isinstance(item, dict):
        return None
    x_min = _coerce_float(item.get("x_min", item.get("xmin")))
    y_min = _coerce_float(item.get("y_min", item.get("ymin")))
    x_max = _coerce_float(item.get("x_max", item.get("xmax")))
    y_max = _coerce_float(item.get("y_max", item.get("ymax")))
    if None in (x_min, y_min, x_max, y_max):
        return None
    x0 = _normalize_coord(float(x_min), size=width)
    y0 = _normalize_coord(float(y_min), size=height)
    x1 = _normalize_coord(float(x_max), size=width)
    y1 = _normalize_coord(float(y_max), size=height)
    if x1 <= x0 or y1 <= y0:
        return None
    return Box(x_min=x0, y_min=y0, x_max=x1, y_max=y1)


def _parse_point_item(item: Any, *, width: int, height: int) -> Optional[Point]:
    if not isinstance(item, dict):
        return None
    raw_x = _coerce_float(item.get("x", item.get("cx")))
    raw_y = _coerce_float(item.get("y", item.get("cy")))
    if raw_x is None or raw_y is None:
        return None
    return Point(x=_normalize_coord(raw_x, size=width), y=_normalize_coord(raw_y, size=height))


def _parse_boxes_collection(raw: Any, *, width: int, height: int) -> list[Box]:
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    out: list[Box] = []
    for item in raw:
        parsed = _parse_box_item(item, width=width, height=height)
        if parsed is not None:
            out.append(parsed)
    return out


def _parse_points_collection(raw: Any, *, width: int, height: int) -> list[Point]:
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    out: list[Point] = []
    for item in raw:
        parsed = _parse_point_item(item, width=width, height=height)
        if parsed is not None:
            out.append(parsed)
    return out


def _candidate_explicit_empty(raw: Any) -> bool:
    return isinstance(raw, list) and len(raw) == 0


def _extract_detect_candidates(payload: dict[str, Any]) -> list[Any]:
    out: list[Any] = []
    for key in ("objects", "boxes", "detections"):
        if key in payload:
            out.append(payload.get(key))
    output = payload.get("output")
    if isinstance(output, dict):
        for key in ("objects", "boxes", "detections"):
            if key in output:
                out.append(output.get(key))
    if all(key in payload for key in ("x_min", "y_min", "x_max", "y_max")):
        out.append([payload])
    if isinstance(output, dict) and all(key in output for key in ("x_min", "y_min", "x_max", "y_max")):
        out.append([output])
    return out


def _extract_point_candidates(payload: dict[str, Any]) -> list[Any]:
    out: list[Any] = []
    for key in ("points", "point", "coordinates"):
        if key in payload:
            out.append(payload.get(key))
    output = payload.get("output")
    if isinstance(output, dict):
        for key in ("points", "point", "coordinates"):
            if key in output:
                out.append(output.get(key))
    if all(key in payload for key in ("x", "y")):
        out.append([payload])
    if isinstance(output, dict) and all(key in output for key in ("x", "y")):
        out.append([output])
    return out


def parse_openrouter_prediction(
    *,
    skill: str,
    answer_text: str,
    raw_response: dict[str, Any],
    image_width: int,
    image_height: int,
) -> tuple[list[Box], list[Point], bool]:
    payload = _extract_first_json_object(answer_text)
    if payload is None:
        payload = _extract_first_json_object(_extract_openrouter_answer_text(raw_response))
    if payload is None:
        return [], [], False
    if skill == "point":
        for candidate in _extract_point_candidates(payload):
            points = _parse_points_collection(candidate, width=image_width, height=image_height)
            if points or _candidate_explicit_empty(candidate):
                return [], points, True
        return [], [], False
    for candidate in _extract_detect_candidates(payload):
        boxes = _parse_boxes_collection(candidate, width=image_width, height=image_height)
        if boxes or _candidate_explicit_empty(candidate):
            return boxes, [], True
    return [], [], False


def _serialize_boxes(boxes: list[Box]) -> list[dict[str, float]]:
    return [
        {
            "x_min": float(box.x_min),
            "y_min": float(box.y_min),
            "x_max": float(box.x_max),
            "y_max": float(box.y_max),
        }
        for box in boxes
    ]


def _serialize_points(points: list[Point]) -> list[dict[str, float]]:
    return [{"x": float(point.x), "y": float(point.y)} for point in points]


def _box_iou(left: Box, right: Box) -> float:
    inter_x_min = max(left.x_min, right.x_min)
    inter_y_min = max(left.y_min, right.y_min)
    inter_x_max = min(left.x_max, right.x_max)
    inter_y_max = min(left.y_max, right.y_max)
    inter_w = max(0.0, inter_x_max - inter_x_min)
    inter_h = max(0.0, inter_y_max - inter_y_min)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0
    left_area = max(0.0, left.x_max - left.x_min) * max(0.0, left.y_max - left.y_min)
    right_area = max(0.0, right.x_max - right.x_min) * max(0.0, right.y_max - right.y_min)
    union = left_area + right_area - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def _match_ious(predicted: list[Box], ground_truth: list[Box]) -> np.ndarray:
    n_pred = len(predicted)
    n_gt = len(ground_truth)
    if n_pred == 0 or n_gt == 0:
        return np.array([], dtype=np.float32)
    size = max(n_pred, n_gt)
    iou_matrix = np.zeros((size, size), dtype=np.float32)
    for gt_idx, gt_box in enumerate(ground_truth):
        for pred_idx, pred_box in enumerate(predicted):
            iou_matrix[gt_idx, pred_idx] = _box_iou(pred_box, gt_box)
    cost = 1.0 - iou_matrix
    row_idx, col_idx = linear_sum_assignment(cost)
    return iou_matrix[row_idx, col_idx]


def _matched_detect_indices(predicted: list[Box], ground_truth: list[Box], *, iou_threshold: float) -> set[int]:
    n_pred = len(predicted)
    n_gt = len(ground_truth)
    if n_pred == 0 or n_gt == 0:
        return set()
    size = max(n_pred, n_gt)
    iou_matrix = np.zeros((size, size), dtype=np.float32)
    for gt_idx, gt_box in enumerate(ground_truth):
        for pred_idx, pred_box in enumerate(predicted):
            iou_matrix[gt_idx, pred_idx] = _box_iou(pred_box, gt_box)
    cost = 1.0 - iou_matrix
    row_idx, col_idx = linear_sum_assignment(cost)
    matched: set[int] = set()
    for gt_idx, pred_idx in zip(row_idx.tolist(), col_idx.tolist()):
        if gt_idx >= n_gt or pred_idx >= n_pred:
            continue
        if iou_matrix[gt_idx, pred_idx] >= float(iou_threshold):
            matched.add(pred_idx)
    return matched


def reward_miou(predicted: list[Box], ground_truth: list[Box]) -> float:
    if not predicted and not ground_truth:
        return 1.0
    if not predicted or not ground_truth:
        return 0.0
    matches = _match_ious(predicted, ground_truth)
    denom = max(len(predicted), len(ground_truth))
    return float(matches.sum()) / float(denom) if denom else 0.0


def reward_f1(predicted: list[Box], ground_truth: list[Box], *, iou_threshold: float) -> float:
    if not predicted and not ground_truth:
        return 1.0
    if not predicted or not ground_truth:
        return 0.0
    matches = _match_ious(predicted, ground_truth)
    true_pos = float((matches >= float(iou_threshold)).sum())
    precision = true_pos / float(len(predicted)) if predicted else 0.0
    recall = true_pos / float(len(ground_truth)) if ground_truth else 0.0
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def count_tp_fp_fn(predicted: list[Box], ground_truth: list[Box], *, iou_threshold: float) -> tuple[int, int, int]:
    n_pred = len(predicted)
    n_gt = len(ground_truth)
    if n_pred == 0 and n_gt == 0:
        return 0, 0, 0
    if n_pred == 0:
        return 0, 0, n_gt
    if n_gt == 0:
        return 0, n_pred, 0
    matched = _matched_detect_indices(predicted, ground_truth, iou_threshold=float(iou_threshold))
    tp = len(matched)
    fp = n_pred - tp
    fn = n_gt - tp
    return tp, fp, fn


def _point_in_box(point: Point, box: Box) -> bool:
    return box.x_min <= point.x <= box.x_max and box.y_min <= point.y <= box.y_max


def _matched_point_indices(points: list[Point], ground_truth: list[Box]) -> set[int]:
    n_points = len(points)
    n_gt = len(ground_truth)
    if n_points == 0 or n_gt == 0:
        return set()
    size = max(n_points, n_gt)
    score = np.zeros((size, size), dtype=np.float32)
    for gt_idx, gt_box in enumerate(ground_truth):
        for point_idx, point in enumerate(points):
            score[gt_idx, point_idx] = 1.0 if _point_in_box(point, gt_box) else 0.0
    row_idx, col_idx = linear_sum_assignment(-score)
    matched: set[int] = set()
    for gt_idx, point_idx in zip(row_idx.tolist(), col_idx.tolist()):
        if gt_idx >= n_gt or point_idx >= n_points:
            continue
        if score[gt_idx, point_idx] >= 0.5:
            matched.add(point_idx)
    return matched


def count_tp_fp_fn_points(points: list[Point], ground_truth: list[Box]) -> tuple[int, int, int]:
    n_points = len(points)
    n_gt = len(ground_truth)
    if n_points == 0 and n_gt == 0:
        return 0, 0, 0
    if n_points == 0:
        return 0, 0, n_gt
    if n_gt == 0:
        return 0, n_points, 0
    tp = len(_matched_point_indices(points, ground_truth))
    fp = n_points - tp
    fn = n_gt - tp
    return tp, fp, fn


def reward_f1_points(points: list[Point], ground_truth: list[Box]) -> float:
    tp, fp, fn = count_tp_fp_fn_points(points, ground_truth)
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0
    denom = (2.0 * float(tp)) + float(fp) + float(fn)
    return 0.0 if denom <= 0.0 else (2.0 * float(tp)) / denom


def normalize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "eval_f1": float(metrics.get("eval_f1", 0.0)),
        "eval_f1_macro": float(metrics.get("eval_f1_macro", 0.0)),
        "eval_miou": float(metrics.get("eval_miou", 0.0)),
        "tp": int(metrics.get("tp", metrics.get("eval_true_pos", 0))),
        "fp": int(metrics.get("fp", metrics.get("eval_false_pos", 0))),
        "fn": int(metrics.get("fn", metrics.get("eval_false_neg", 0))),
        "samples": int(metrics.get("samples", metrics.get("eval_samples", metrics.get("tasks", metrics.get("base_samples", 0))))),
    }


def _parse_markdown_number(value: str) -> Optional[float]:
    cleaned = str(value or "").strip().strip("`").replace(",", "")
    if cleaned in {"", "-", "n/a", "N/A"}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _parse_readme_benchmark_table(readme_path: Path, *, sample_count_fallback: int = 0) -> tuple[dict[str, Any], dict[str, Any]]:
    text = readme_path.read_text(encoding="utf-8")
    sample_count_match = _README_SAMPLE_COUNT_PATTERN.search(text)
    sample_count = int(sample_count_match.group(1)) if sample_count_match else int(sample_count_fallback)
    metric_map = {
        "f1": "eval_f1",
        "macro f1": "eval_f1_macro",
        "miou": "eval_miou",
        "true positives": "tp",
        "false positives": "fp",
        "false negatives": "fn",
    }
    before: dict[str, Any] = {"samples": sample_count}
    after: dict[str, Any] = {"samples": sample_count}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("|"):
            continue
        parts = [part.strip() for part in line.strip("|").split("|")]
        if len(parts) < 3:
            continue
        label = parts[0].lower()
        if label not in metric_map:
            continue
        before_value = _parse_markdown_number(parts[1])
        after_value = _parse_markdown_number(parts[2])
        if before_value is None or after_value is None:
            continue
        key = metric_map[label]
        before[key] = before_value
        after[key] = after_value
    if not any(key in before for key in ("eval_f1", "eval_f1_macro", "eval_miou", "tp", "fp", "fn")):
        raise ValueError(f"Unable to parse benchmark metrics from README: {readme_path}")
    return normalize_metrics(before), normalize_metrics(after)


def _list_state_farm_packet_images(task_dir: Path) -> list[tuple[int, str, Path]]:
    imgs_dir = task_dir / "imgs"
    if not imgs_dir.exists():
        return []
    items: list[tuple[int, str, Path]] = []
    for path in sorted(imgs_dir.iterdir()):
        if not path.is_file():
            continue
        match = _STATE_FARM_PACKET_IMAGE_PATTERN.match(path.name)
        if not match:
            continue
        items.append((int(match.group("index")), str(match.group("sample_id")), path))
    return sorted(items, key=lambda item: item[0])


def _list_numbered_packet_images(task_dir: Path) -> list[tuple[int, Path]]:
    imgs_dir = task_dir / "imgs"
    if not imgs_dir.exists():
        return []
    items: list[tuple[int, Path]] = []
    for path in sorted(imgs_dir.iterdir()):
        if not path.is_file():
            continue
        stem = path.stem
        if "_" not in stem:
            continue
        prefix = stem.split("_", 1)[0]
        if not prefix.isdigit():
            continue
        items.append((int(prefix), path))
    return sorted(items, key=lambda item: item[0])


def _count_packet_images(task_dir: Path) -> int:
    imgs_dir = task_dir / "imgs"
    if not imgs_dir.exists():
        return 0
    return sum(1 for path in imgs_dir.iterdir() if path.is_file())


def _canonical_aerial_sample_key(value: str) -> str:
    text = Path(str(value or "")).name.lower()
    for suffix in (".jpeg", ".jpg", ".png"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
            break
    text = _AERIAL_INDEX_PREFIX_PATTERN.sub("", text)
    if text.endswith("_airplane"):
        text = text[: -len("_airplane")]
    if text.endswith("_bgneg"):
        text = text[: -len("_bgneg")]
    return text.replace(".", "_")


def _iter_aerial_row_keys(row: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    image_payload = row.get("image")
    if isinstance(image_payload, dict):
        image_path = str(image_payload.get("path") or "").strip()
        if image_path:
            keys.append(_canonical_aerial_sample_key(image_path))
    for field in ("source_image_id", "source_base_id"):
        raw = str(row.get(field) or "").strip()
        if raw:
            keys.append(_canonical_aerial_sample_key(raw))
    out: list[str] = []
    seen: set[str] = set()
    for key in keys:
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _load_aerial_benchmark_order(benchmark_path: Path) -> list[str]:
    payload = _load_json(benchmark_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object in {benchmark_path}")
    baseline = payload.get("baseline")
    if not isinstance(baseline, dict):
        return []
    visualization_paths = baseline.get("visualization_paths")
    if not isinstance(visualization_paths, list):
        return []
    order: list[str] = []
    for raw_path in visualization_paths:
        key = _canonical_aerial_sample_key(str(raw_path or ""))
        if key:
            order.append(key)
    return order


def aggregate_prediction_metrics(
    records: list[dict[str, Any]],
    *,
    skill: str,
    iou_threshold: float,
) -> dict[str, Any]:
    total_f1 = 0.0
    total_miou = 0.0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_latency_ms = 0.0
    successful = 0
    failed = 0
    positive = 0
    negative = 0
    parse_success = 0
    for record in records:
        if bool(record.get("failed")):
            failed += 1
            continue
        gt_boxes = [
            Box(
                x_min=float(item["x_min"]),
                y_min=float(item["y_min"]),
                x_max=float(item["x_max"]),
                y_max=float(item["y_max"]),
            )
            for item in record.get("ground_truth_boxes", [])
        ]
        if gt_boxes:
            positive += 1
        else:
            negative += 1
        if skill == "point":
            points = [
                Point(x=float(item["x"]), y=float(item["y"]))
                for item in record.get("pred_points", [])
            ]
            f1 = reward_f1_points(points, gt_boxes)
            miou = 0.0
            tp, fp, fn = count_tp_fp_fn_points(points, gt_boxes)
        else:
            boxes = [
                Box(
                    x_min=float(item["x_min"]),
                    y_min=float(item["y_min"]),
                    x_max=float(item["x_max"]),
                    y_max=float(item["y_max"]),
                )
                for item in record.get("pred_boxes", [])
            ]
            f1 = reward_f1(boxes, gt_boxes, iou_threshold=float(iou_threshold))
            miou = reward_miou(boxes, gt_boxes)
            tp, fp, fn = count_tp_fp_fn(boxes, gt_boxes, iou_threshold=float(iou_threshold))
        total_f1 += f1
        total_miou += miou
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_latency_ms += float(record.get("latency_ms", 0.0) or 0.0)
        successful += 1
        if bool(record.get("parse_success")):
            parse_success += 1
    micro_denom = (2 * total_tp) + total_fp + total_fn
    micro_f1 = 1.0 if micro_denom == 0 else (2.0 * total_tp) / float(micro_denom)
    return {
        "skill": skill,
        "samples": successful,
        "total_samples": len(records),
        "failed_samples": failed,
        "positive_samples": positive,
        "negative_samples": negative,
        "eval_f1": micro_f1 if successful > 0 else 0.0,
        "eval_f1_macro": (total_f1 / float(successful)) if successful > 0 else 0.0,
        "eval_miou": (total_miou / float(successful)) if successful > 0 else 0.0,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "avg_latency_ms": (total_latency_ms / float(successful)) if successful > 0 else 0.0,
        "parse_success_rate": (parse_success / float(successful)) if successful > 0 else 0.0,
        "iou_threshold": float(iou_threshold),
    }


def compute_metric_deltas(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    keys = ("eval_f1", "eval_f1_macro", "eval_miou", "tp", "fp", "fn", "samples")
    deltas: dict[str, Any] = {}
    for key in keys:
        if key not in reference or key not in candidate:
            continue
        left = reference[key]
        right = candidate[key]
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            deltas[key] = float(right) - float(left)
    return deltas


def _normalize_id(value: str) -> str:
    return str(value or "").strip().lower()


def _latest_snapshot_path(dataset_id: str) -> Path:
    hub_dir = Path.home() / ".cache" / "huggingface" / "hub" / f"datasets--{dataset_id.replace('/', '--')}"
    ref_path = hub_dir / "refs" / "main"
    if ref_path.exists():
        snapshot_id = ref_path.read_text(encoding="utf-8").strip()
        snapshot_path = hub_dir / "snapshots" / snapshot_id
        if snapshot_path.exists():
            return snapshot_path
    snapshots_dir = hub_dir / "snapshots"
    snapshots = sorted(snapshots_dir.iterdir()) if snapshots_dir.exists() else []
    if not snapshots:
        raise FileNotFoundError(f"No cached snapshots found for {dataset_id}")
    return snapshots[-1]


def _find_cached_parquet_path(dataset_id: str, split: str) -> Path:
    snapshot_candidates: list[Path] = []
    try:
        snapshot_candidates.append(_latest_snapshot_path(dataset_id))
    except FileNotFoundError:
        pass
    hub_dir = Path.home() / ".cache" / "huggingface" / "hub" / f"datasets--{dataset_id.replace('/', '--')}"
    snapshots_dir = hub_dir / "snapshots"
    if snapshots_dir.exists():
        for snapshot in sorted(snapshots_dir.iterdir(), reverse=True):
            if snapshot not in snapshot_candidates:
                snapshot_candidates.append(snapshot)
    for snapshot in snapshot_candidates:
        matches = sorted((snapshot / "data").glob(f"{split}-*.parquet"))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Unable to find cached parquet for {dataset_id} split={split}")


def _read_parquet_rows(parquet_path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    return pq.read_table(parquet_path).to_pylist()


def _image_from_bytes_payload(image_payload: dict[str, Any]) -> Image.Image:
    raw_bytes = image_payload.get("bytes")
    if raw_bytes is None:
        raise ValueError("Image bytes missing from dataset row")
    return Image.open(io.BytesIO(raw_bytes)).convert("RGB")


def _image_from_local_path(image_path: Path) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def _numbered_packet_sample_id(image_path: Path) -> str:
    stem = image_path.stem
    if "_" not in stem:
        return stem
    prefix, rest = stem.split("_", 1)
    return rest if prefix.isdigit() and rest else stem


def _parse_boxes_from_annotation(value: Any, *, width: int, height: int) -> list[Box]:
    if value in (None, "", []):
        return []
    raw = value
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        try:
            raw = json.loads(text)
        except json.JSONDecodeError:
            return []
    return _parse_boxes_collection(raw, width=width, height=height)


def _find_reference_dir(task_dir: Path) -> Optional[Path]:
    candidate_dirs = [path for path in sorted(task_dir.iterdir()) if path.is_dir()]
    for path in candidate_dirs:
        if path.name.startswith("openrouter_gpt_5_4"):
            continue
        if (path / "sample_before_after.json").exists() or (path / "metrics.json").exists():
            return path
    return None


def _find_single_json_glob(task_dir: Path, pattern: str) -> Optional[Path]:
    matches = sorted(task_dir.glob(pattern))
    return matches[0] if matches else None


def build_task_registry(repo_root: Path = REPO_ROOT) -> dict[str, TaskSpec]:
    packet_root = repo_root / "outputs" / "task_sample_packets"

    state_farm_dir = packet_root / "state_farm"
    state_farm_ref_dir = _find_reference_dir(state_farm_dir)

    player_dir = packet_root / "player_with_ball"
    player_compare_path = _find_single_json_glob(player_dir, "*_baseline_vs_ft/*.json")
    player_readme_path = player_dir / "README.md"

    aerial_dir = packet_root / "aerial"
    aerial_packet_path = aerial_dir / "samples.before_after.json"
    aerial_benchmark_path = _find_single_json_glob(aerial_dir, "benchmark_*.json")

    specs = {
        "state_farm": TaskSpec(
            name="state_farm",
            skill="detect",
            prompt="State Farm logo",
            task_dir=state_farm_dir,
            source_kind="cached_hf_parquet",
            dataset_id="maxs-m87/NBA_StateFarm_Splits_01",
            split="validation",
            iou_threshold=0.4,
            reference_metrics_path=(state_farm_ref_dir / "metrics.json") if state_farm_ref_dir else None,
            reference_samples_path=(state_farm_ref_dir / "sample_before_after.json") if state_farm_ref_dir else None,
            reference_compare_path=None,
            reference_readme_path=state_farm_dir / "README.md",
            aerial_subset_dataset_path=None,
            aerial_packet_path=None,
            aerial_benchmark_path=None,
        ),
        "player_with_ball": TaskSpec(
            name="player_with_ball",
            skill="detect",
            prompt="Player with the ball",
            task_dir=player_dir,
            source_kind="cached_hf_parquet",
            dataset_id="maxs-m87/Ball-Holder-splits-v1",
            split="test",
            iou_threshold=0.4,
            reference_metrics_path=None,
            reference_samples_path=None,
            reference_compare_path=player_compare_path,
            reference_readme_path=player_readme_path if player_readme_path.exists() else None,
            aerial_subset_dataset_path=None,
            aerial_packet_path=None,
            aerial_benchmark_path=None,
        ),
        "aerial": TaskSpec(
            name="aerial",
            skill="point",
            prompt="airplane",
            task_dir=aerial_dir,
            source_kind="cached_hf_parquet_packet_benchmark",
            dataset_id="maxs-m87/aerial_airport_point_v2",
            split="test",
            iou_threshold=0.0,
            reference_metrics_path=None,
            reference_samples_path=None,
            reference_compare_path=None,
            reference_readme_path=aerial_dir / "README.md",
            aerial_subset_dataset_path=repo_root / "outputs" / "advertising_subsets" / "aerial_manual_top100" / "subset" / "dataset",
            aerial_packet_path=aerial_packet_path if aerial_packet_path.exists() else None,
            aerial_benchmark_path=aerial_benchmark_path,
        ),
    }
    return specs


def load_state_farm_samples(spec: TaskSpec) -> list[Sample]:
    parquet_path = _find_cached_parquet_path(str(spec.dataset_id), str(spec.split))
    rows = _read_parquet_rows(parquet_path)
    row_by_path = {
        _normalize_id(str(row.get("image", {}).get("path") or "")): row
        for row in rows
        if isinstance(row.get("image"), dict)
    }
    state_farm_packet_images = _list_state_farm_packet_images(spec.task_dir)
    state_farm_packet_image_by_sample_id = {
        _normalize_id(sample_id): packet_image_path
        for _, sample_id, packet_image_path in state_farm_packet_images
    }
    if spec.reference_samples_path is not None and spec.reference_samples_path.exists():
        packet_samples = _load_json(spec.reference_samples_path)
        if not isinstance(packet_samples, list):
            raise ValueError(f"Expected list in {spec.reference_samples_path}")
        samples: list[Sample] = []
        for index, record in enumerate(packet_samples, start=1):
            source_image_path = str(record.get("source_image_path") or "")
            row = row_by_path.get(_normalize_id(source_image_path))
            if row is None:
                raise KeyError(f"State Farm source image missing from cached parquet: {source_image_path}")
            sample_id = str(record.get("sample_id") or Path(source_image_path).stem)
            packet_image_path = state_farm_packet_image_by_sample_id.get(_normalize_id(sample_id))
            if packet_image_path is None:
                raise FileNotFoundError(f"State Farm packet image not found for sample_id={sample_id}")
            image = _image_from_local_path(packet_image_path)
            gt_boxes = [
                Box(
                    x_min=float(item["x_min"]),
                    y_min=float(item["y_min"]),
                    x_max=float(item["x_max"]),
                    y_max=float(item["y_max"]),
                )
                for item in record.get("ground_truth_boxes", [])
            ]
            samples.append(
                Sample(
                    sample_index=int(record.get("sample_index", index)),
                    sample_id=sample_id,
                    source_image_path=source_image_path,
                    prompt=str(record.get("prompt") or spec.prompt),
                    task_type=str(record.get("task_type") or row.get("type") or "region"),
                    notes=str(record.get("notes") or ""),
                    timestamp=str(record.get("timestamp") or ""),
                    image=image,
                    ground_truth_boxes=gt_boxes,
                    base_record=dict(record),
                )
            )
        return samples

    if not state_farm_packet_images:
        raise FileNotFoundError(
            f"State Farm packet samples not found. reference_samples_path={spec.reference_samples_path} packet_imgs_dir={spec.task_dir / 'imgs'}"
        )

    samples: list[Sample] = []
    for packet_index, sample_id, packet_image_path in state_farm_packet_images:
        source_image_path = f"{sample_id}.jpg"
        row = row_by_path.get(_normalize_id(source_image_path))
        if row is None:
            raise KeyError(f"State Farm source image missing from cached parquet: {source_image_path}")
        image = _image_from_local_path(packet_image_path)
        gt_boxes = _parse_boxes_from_annotation(row.get("answer_boxes"), width=image.width, height=image.height)
        base_record = {
            "sample_index": int(packet_index),
            "sample_id": sample_id,
            "source_image_path": source_image_path,
            "prompt": str(row.get("prompt") or spec.prompt),
            "task_type": str(row.get("type") or "region"),
            "ground_truth_boxes": _serialize_boxes(gt_boxes),
            "notes": str(row.get("notes") or ""),
            "timestamp": str(row.get("timestamp") or ""),
            "packet_image_path": str(packet_image_path.relative_to(REPO_ROOT)),
        }
        samples.append(
            Sample(
                sample_index=int(packet_index),
                sample_id=sample_id,
                source_image_path=source_image_path,
                prompt=str(row.get("prompt") or spec.prompt),
                task_type=str(row.get("type") or "region"),
                notes=str(row.get("notes") or ""),
                timestamp=str(row.get("timestamp") or ""),
                image=image,
                ground_truth_boxes=gt_boxes,
                base_record=base_record,
            )
        )
    return samples


def load_player_with_ball_samples(spec: TaskSpec) -> list[Sample]:
    parquet_path = _find_cached_parquet_path(str(spec.dataset_id), str(spec.split))
    rows = _read_parquet_rows(parquet_path)
    compare_payload = _load_json(spec.reference_compare_path) if spec.reference_compare_path and spec.reference_compare_path.exists() else {}
    compare_samples = compare_payload.get("samples") if isinstance(compare_payload, dict) else None
    row_by_sample_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        image_payload = row.get("image")
        if not isinstance(image_payload, dict):
            continue
        sample_id = Path(str(image_payload.get("path") or "")).stem
        if sample_id:
            row_by_sample_id[_normalize_id(sample_id)] = row
    packet_images = _list_numbered_packet_images(spec.task_dir)
    if not packet_images:
        raise FileNotFoundError(f"Ball Holder packet images not found: {spec.task_dir / 'imgs'}")
    if compare_samples is not None and (not isinstance(compare_samples, list) or len(compare_samples) != len(packet_images)):
        raise ValueError("Ball Holder reference sample count does not match packet image count")
    samples: list[Sample] = []
    for index, packet_image_path in packet_images:
        sample_id = _numbered_packet_sample_id(packet_image_path)
        row = row_by_sample_id.get(_normalize_id(sample_id))
        if row is None:
            raise KeyError(f"Ball Holder packet image missing from cached parquet: {packet_image_path.name}")
        image = _image_from_local_path(packet_image_path)
        width, height = image.size
        source_image_path = str(row.get("image", {}).get("path") or f"{index:04d}.jpg")
        gt_boxes = _parse_boxes_from_annotation(row.get("answer_boxes"), width=width, height=height)
        base_record: dict[str, Any] = {
            "sample_index": index,
            "sample_id": sample_id,
            "source_image_path": source_image_path,
            "prompt": str(row.get("prompt") or spec.prompt),
            "task_type": str(row.get("type") or "region"),
            "ground_truth_boxes": _serialize_boxes(gt_boxes),
            "notes": str(row.get("notes") or ""),
            "timestamp": str(row.get("timestamp") or ""),
            "packet_image_path": str(packet_image_path.relative_to(REPO_ROOT)),
        }
        if isinstance(compare_samples, list):
            compare_record = compare_samples[index - 1]
            if isinstance(compare_record, dict):
                if "before" in compare_record:
                    base_record["before"] = compare_record["before"]
                if "after" in compare_record:
                    base_record["after"] = compare_record["after"]
                reference_sample_id = str(compare_record.get("sample_id") or "")
                if reference_sample_id and reference_sample_id != sample_id:
                    base_record["reference_sample_id"] = reference_sample_id
        samples.append(
            Sample(
                sample_index=index,
                sample_id=sample_id,
                source_image_path=source_image_path,
                prompt=str(row.get("prompt") or spec.prompt),
                task_type=str(row.get("type") or "region"),
                notes=str(row.get("notes") or ""),
                timestamp=str(row.get("timestamp") or ""),
                image=image,
                ground_truth_boxes=gt_boxes,
                base_record=base_record,
            )
        )
    return samples


def load_aerial_samples(spec: TaskSpec) -> list[Sample]:
    packet_images = _list_numbered_packet_images(spec.task_dir)
    packet_images_by_key = {
        _canonical_aerial_sample_key(path.name): path
        for _, path in packet_images
    }
    if spec.aerial_packet_path is not None and spec.aerial_packet_path.exists():
        if spec.aerial_subset_dataset_path is None or not spec.aerial_subset_dataset_path.exists():
            raise FileNotFoundError(f"Aerial subset dataset not found: {spec.aerial_subset_dataset_path}")
        packet_samples = _load_json(spec.aerial_packet_path)
        if not isinstance(packet_samples, list):
            raise ValueError(f"Expected list in {spec.aerial_packet_path}")
        from datasets import load_from_disk

        dataset = load_from_disk(str(spec.aerial_subset_dataset_path))["test"]
        row_by_id: dict[str, dict[str, Any]] = {}
        for row in dataset:
            row_by_id[_normalize_id(str(row.get("source_image_id") or ""))] = row
        samples: list[Sample] = []
        for index, record in enumerate(packet_samples, start=1):
            source_image_id = str(record.get("source_image_id") or record.get("sample_id") or "")
            row = row_by_id.get(_normalize_id(source_image_id))
            if row is None:
                raise KeyError(f"Aerial subset row not found for source_image_id={source_image_id}")
            packet_image_path = packet_images_by_key.get(_canonical_aerial_sample_key(source_image_id))
            if packet_image_path is None:
                raise FileNotFoundError(f"Aerial packet image not found for source_image_id={source_image_id}")
            image = _image_from_local_path(packet_image_path)
            gt_boxes = [
                Box(
                    x_min=float(item["x_min"]),
                    y_min=float(item["y_min"]),
                    x_max=float(item["x_max"]),
                    y_max=float(item["y_max"]),
                )
                for item in record.get("ground_truth_boxes", [])
            ]
            samples.append(
                Sample(
                    sample_index=index,
                    sample_id=str(record.get("sample_id") or source_image_id),
                    source_image_path=str(record.get("image_file") or source_image_id),
                    prompt=spec.prompt,
                    task_type="point",
                    notes="",
                    timestamp="",
                    image=image,
                    ground_truth_boxes=gt_boxes,
                    base_record={**dict(record), "packet_image_path": str(packet_image_path.relative_to(REPO_ROOT))},
                )
            )
        return samples

    if not spec.dataset_id:
        raise ValueError("Aerial cached HF dataset settings are required")

    rows: list[dict[str, Any]] = []
    seen_row_paths: set[str] = set()
    for split_name in ("train", "validation", "test"):
        try:
            parquet_path = _find_cached_parquet_path(str(spec.dataset_id), split_name)
        except FileNotFoundError:
            continue
        for row in _read_parquet_rows(parquet_path):
            image_payload = row.get("image")
            if not isinstance(image_payload, dict):
                continue
            image_path = str(image_payload.get("path") or "").strip()
            if not image_path or image_path in seen_row_paths:
                continue
            seen_row_paths.add(image_path)
            rows.append(row)

    row_by_key: dict[str, dict[str, Any]] = {}
    for row in rows:
        for key in _iter_aerial_row_keys(row):
            row_by_key.setdefault(key, row)

    ordered_rows: list[dict[str, Any]]
    if spec.aerial_benchmark_path is not None and spec.aerial_benchmark_path.exists():
        ordered_keys = _load_aerial_benchmark_order(spec.aerial_benchmark_path)
        if ordered_keys:
            missing = [key for key in ordered_keys if key not in row_by_key]
            if missing:
                raise KeyError(f"Aerial benchmark rows missing from cached parquet: {missing}")
            ordered_rows = [row_by_key[key] for key in ordered_keys]
        else:
            ordered_rows = rows
    else:
        if not packet_images:
            raise FileNotFoundError(
                f"Aerial packet sources not found. packet_json={spec.aerial_packet_path} benchmark_json={spec.aerial_benchmark_path} packet_imgs_dir={spec.task_dir / 'imgs'}"
            )
        ordered_rows = []
        for _packet_index, packet_image_path in packet_images:
            key = _canonical_aerial_sample_key(packet_image_path.name)
            row = row_by_key.get(key)
            if row is None:
                raise KeyError(f"Aerial packet image missing from cached parquets: {packet_image_path.name}")
            ordered_rows.append(row)

    samples: list[Sample] = []
    for index, row in enumerate(ordered_rows, start=1):
        source_image_id = str(row.get("source_image_id") or Path(str(row.get("image", {}).get("path") or "")).stem)
        source_image_path = str(row.get("image", {}).get("path") or source_image_id)
        packet_image_path = packet_images_by_key.get(_canonical_aerial_sample_key(source_image_path))
        if packet_image_path is None:
            raise FileNotFoundError(f"Aerial packet image not found for source_image_path={source_image_path}")
        image = _image_from_local_path(packet_image_path)
        gt_boxes = _parse_boxes_from_annotation(row.get("answer_boxes"), width=image.width, height=image.height)
        base_record = {
            "sample_index": index,
            "sample_id": source_image_id,
            "source_image_id": source_image_id,
            "source_image_path": source_image_path,
            "prompt": spec.prompt,
            "task_type": "point",
            "ground_truth_boxes": _serialize_boxes(gt_boxes),
            "notes": "",
            "timestamp": "",
            "source_split": str(row.get("source_split") or spec.split),
            "source_dataset": str(row.get("source_dataset") or spec.dataset_id),
            "source_collection": str(row.get("source_collection") or ""),
            "source_variant": str(row.get("source_variant") or ""),
            "source_is_synthetic": bool(row.get("source_is_synthetic", False)),
            "source_base_id": str(row.get("source_base_id") or ""),
            "split_group_id": str(row.get("split_group_id") or ""),
            "class_count": int(row.get("class_count", 0) or 0),
            "packet_image_path": str(packet_image_path.relative_to(REPO_ROOT)) if packet_image_path is not None else None,
        }
        samples.append(
            Sample(
                sample_index=index,
                sample_id=source_image_id,
                source_image_path=source_image_path,
                prompt=spec.prompt,
                task_type="point",
                notes="",
                timestamp="",
                image=image,
                ground_truth_boxes=gt_boxes,
                base_record=base_record,
            )
        )
    return samples


def load_task_samples(spec: TaskSpec) -> list[Sample]:
    if spec.name == "state_farm":
        return load_state_farm_samples(spec)
    if spec.name == "player_with_ball":
        return load_player_with_ball_samples(spec)
    if spec.name == "aerial":
        return load_aerial_samples(spec)
    raise ValueError(f"Unsupported task: {spec.name}")


def load_reference_metrics(spec: TaskSpec) -> tuple[dict[str, Any], dict[str, Any], str]:
    if spec.name == "state_farm":
        if spec.reference_metrics_path is not None and spec.reference_metrics_path.exists():
            payload = _load_json(spec.reference_metrics_path)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected object in {spec.reference_metrics_path}")
            return (
                normalize_metrics(dict(payload.get("baseline") or {})),
                normalize_metrics(dict(payload.get("checkpoint") or {})),
                str(spec.reference_metrics_path.relative_to(REPO_ROOT)),
            )
        if spec.reference_readme_path is None or not spec.reference_readme_path.exists():
            raise FileNotFoundError(
                f"State Farm reference metrics not found. metrics_json={spec.reference_metrics_path} readme={spec.reference_readme_path}"
            )
        before, after = _parse_readme_benchmark_table(
            spec.reference_readme_path,
            sample_count_fallback=len(_list_state_farm_packet_images(spec.task_dir)),
        )
        return before, after, str(spec.reference_readme_path.relative_to(REPO_ROOT))
    if spec.name == "player_with_ball":
        if spec.reference_compare_path is not None and spec.reference_compare_path.exists():
            payload = _load_json(spec.reference_compare_path)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected object in {spec.reference_compare_path}")
            return (
                normalize_metrics(dict(payload.get("baseline") or {})),
                normalize_metrics(dict(payload.get("candidate") or {})),
                str(spec.reference_compare_path.relative_to(REPO_ROOT)),
            )
        if spec.reference_readme_path is None or not spec.reference_readme_path.exists():
            raise FileNotFoundError(
                f"Ball Holder reference metrics not found. compare_json={spec.reference_compare_path} readme={spec.reference_readme_path}"
            )
        before, after = _parse_readme_benchmark_table(
            spec.reference_readme_path,
            sample_count_fallback=_count_packet_images(spec.task_dir),
        )
        return before, after, str(spec.reference_readme_path.relative_to(REPO_ROOT))
    if spec.name == "aerial":
        if spec.aerial_packet_path is not None and spec.aerial_packet_path.exists():
            payload = _load_json(spec.aerial_packet_path)
            if not isinstance(payload, list):
                raise ValueError(f"Expected list in {spec.aerial_packet_path}")
            before_records: list[dict[str, Any]] = []
            after_records: list[dict[str, Any]] = []
            for item in payload:
                if not isinstance(item, dict):
                    continue
                before_point = item.get("before", {}).get("point")
                after_point = item.get("after", {}).get("point")
                if isinstance(before_point, dict):
                    before_records.append(before_point)
                if isinstance(after_point, dict):
                    after_records.append(after_point)
            return (
                normalize_metrics(_aggregate_reference_point_records(before_records)),
                normalize_metrics(_aggregate_reference_point_records(after_records)),
                str(spec.aerial_packet_path.relative_to(REPO_ROOT)),
            )
        if spec.aerial_benchmark_path is None or not spec.aerial_benchmark_path.exists():
            if spec.reference_readme_path is None or not spec.reference_readme_path.exists():
                raise FileNotFoundError(
                    f"Aerial reference metrics not found. packet_json={spec.aerial_packet_path} benchmark_json={spec.aerial_benchmark_path} readme={spec.reference_readme_path}"
                )
            before, after = _parse_readme_benchmark_table(
                spec.reference_readme_path,
                sample_count_fallback=_count_packet_images(spec.task_dir),
            )
            return before, after, str(spec.reference_readme_path.relative_to(REPO_ROOT))
        payload = _load_json(spec.aerial_benchmark_path)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected object in {spec.aerial_benchmark_path}")
        return (
            normalize_metrics(dict(payload.get("baseline") or {})),
            normalize_metrics(dict(payload.get("candidate") or {})),
            str(spec.aerial_benchmark_path.relative_to(REPO_ROOT)),
        )
    raise ValueError(f"Unsupported task: {spec.name}")


def build_task_registry(repo_root: Optional[Path] = None) -> dict[str, task_packet_common.TaskSpec]:
    resolved_root = REPO_ROOT if repo_root is None else Path(repo_root)
    return task_packet_common.build_task_registry(resolved_root)


def load_state_farm_samples(spec: TaskSpec) -> list[task_packet_common.Sample]:
    return task_packet_common.load_state_farm_samples(spec, repo_root=REPO_ROOT)


def load_player_with_ball_samples(spec: TaskSpec) -> list[task_packet_common.Sample]:
    return task_packet_common.load_player_with_ball_samples(spec, repo_root=REPO_ROOT)


def load_aerial_samples(spec: TaskSpec) -> list[task_packet_common.Sample]:
    return task_packet_common.load_aerial_samples(spec, repo_root=REPO_ROOT)


def load_task_samples(spec: TaskSpec) -> list[task_packet_common.Sample]:
    return task_packet_common.load_task_samples(spec, repo_root=REPO_ROOT)


def load_reference_metrics(spec: TaskSpec) -> tuple[dict[str, Any], dict[str, Any], str]:
    return task_packet_common.load_reference_metrics(spec, repo_root=REPO_ROOT)


def _aggregate_reference_point_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_f1 = 0.0
    total_miou = 0.0
    count = 0
    for record in records:
        total_tp += int(record.get("tp", 0))
        total_fp += int(record.get("fp", 0))
        total_fn += int(record.get("fn", 0))
        total_f1 += float(record.get("task_f1", 0.0))
        total_miou += float(record.get("task_miou", 0.0))
        count += 1
    denom = (2 * total_tp) + total_fp + total_fn
    micro_f1 = 1.0 if denom == 0 else (2.0 * total_tp) / float(denom)
    return {
        "eval_f1": micro_f1 if count > 0 else 0.0,
        "eval_f1_macro": (total_f1 / float(count)) if count > 0 else 0.0,
        "eval_miou": (total_miou / float(count)) if count > 0 else 0.0,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "samples": count,
    }


def build_response_schema(skill: str) -> dict[str, Any]:
    if skill == "point":
        return {
            "name": "point_response",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "points": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "x": {"type": "number", "description": "Normalized x coordinate in [0,1]."},
                                "y": {"type": "number", "description": "Normalized y coordinate in [0,1]."},
                            },
                            "required": ["x", "y"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["points"],
                "additionalProperties": False,
            },
        }
    return {
        "name": "detect_response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "objects": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "x_min": {"type": "number", "description": "Normalized x_min coordinate in [0,1]."},
                            "y_min": {"type": "number", "description": "Normalized y_min coordinate in [0,1]."},
                            "x_max": {"type": "number", "description": "Normalized x_max coordinate in [0,1]."},
                            "y_max": {"type": "number", "description": "Normalized y_max coordinate in [0,1]."},
                        },
                        "required": ["x_min", "y_min", "x_max", "y_max"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["objects"],
            "additionalProperties": False,
        },
    }


def build_instruction(*, skill: str, prompt: str, use_response_format: bool) -> str:
    if skill == "point":
        base = (
            "Analyze the image and return one point inside each visible instance of the target object. "
            f"Target object: {prompt}. "
            "Return an empty list when the object is absent. "
            "Coordinates must be normalized to [0,1]."
        )
        if use_response_format:
            return base
        return f'{base} Return JSON only in this shape: {{"points":[{{"x":0.5,"y":0.5}}]}}.'
    base = (
        "Analyze the image and return tight bounding boxes for each visible instance of the target object. "
        f"Target object: {prompt}. "
        "Return an empty list when the object is absent. "
        "Coordinates must be normalized to [0,1]."
    )
    if use_response_format:
        return base
    return f'{base} Return JSON only in this shape: {{"objects":[{{"x_min":0.1,"y_min":0.1,"x_max":0.2,"y_max":0.2}}]}}.'


def build_request_variants(detail: str) -> list[RequestVariant]:
    normalized_detail = str(detail or "").strip() or "original"
    fallback_detail = "high" if normalized_detail == "original" else normalized_detail
    variants = [
        RequestVariant(
            name="strict_schema",
            detail=normalized_detail,
            use_response_format=True,
            require_parameters=True,
            use_response_healing=True,
        ),
        RequestVariant(
            name="schema_without_require_parameters",
            detail=normalized_detail,
            use_response_format=True,
            require_parameters=False,
            use_response_healing=True,
        ),
        RequestVariant(
            name="schema_without_plugins",
            detail=normalized_detail,
            use_response_format=True,
            require_parameters=False,
            use_response_healing=False,
        ),
    ]
    if fallback_detail != normalized_detail:
        variants.append(
            RequestVariant(
                name="schema_high_detail",
                detail=fallback_detail,
                use_response_format=True,
                require_parameters=False,
                use_response_healing=False,
            )
        )
    variants.append(
        RequestVariant(
            name="prompt_json_fallback",
            detail=fallback_detail,
            use_response_format=False,
            require_parameters=False,
            use_response_healing=False,
        )
    )
    return variants


def build_openrouter_payload(
    *,
    model: str,
    skill: str,
    prompt: str,
    image_url: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    request_variant: RequestVariant,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": build_instruction(
                            skill=skill,
                            prompt=prompt,
                            use_response_format=bool(request_variant.use_response_format),
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": image_url, "detail": request_variant.detail}},
                ],
            }
        ],
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": int(max_tokens),
    }
    if request_variant.use_response_format:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": build_response_schema(skill),
        }
    if request_variant.require_parameters:
        payload["provider"] = {
            "require_parameters": True,
        }
    if request_variant.use_response_healing:
        payload["plugins"] = [{"id": "response-healing"}]
    return payload


def call_openrouter_chat_api(
    *,
    api_base: str,
    api_key: str,
    payload_variants: list[tuple[RequestVariant, dict[str, Any]]],
    timeout: float,
    retry_429_max_retries: int,
    retry_429_backoff_s: float,
    retry_429_max_backoff_s: float,
) -> tuple[str, dict[str, Any], float, str]:
    endpoint = api_base.rstrip("/") + "/chat/completions"
    attempted_variants: list[str] = []
    last_error: Optional[RuntimeError] = None
    for request_variant, payload in payload_variants:
        attempted_variants.append(request_variant.name)
        attempt = 0
        while True:
            request = urllib.request.Request(
                endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers=_build_auth_headers(api_key),
                method="POST",
            )
            started = time.monotonic()
            try:
                with urllib.request.urlopen(request, timeout=float(timeout)) as response:
                    body = response.read().decode("utf-8", errors="replace")
                latency_ms = (time.monotonic() - started) * 1000.0
                data = json.loads(body) if body else {}
                if not isinstance(data, dict):
                    data = {}
                return _extract_openrouter_answer_text(data), data, latency_ms, request_variant.name
            except urllib.error.HTTPError as exc:
                latency_ms = (time.monotonic() - started) * 1000.0
                retry_after_s = 0.0
                detail = ""
                try:
                    detail = exc.read().decode("utf-8", errors="replace")
                except Exception:
                    detail = ""
                if exc.code == 429:
                    retry_after = str(exc.headers.get("Retry-After") or "").strip()
                    if retry_after:
                        try:
                            retry_after_s = max(0.0, float(retry_after))
                        except (TypeError, ValueError):
                            retry_after_s = 0.0
                if exc.code == 429 and attempt < max(0, int(retry_429_max_retries)):
                    backoff = max(0.0, float(retry_429_backoff_s)) * (2.0**attempt)
                    capped = min(max(0.0, float(retry_429_max_backoff_s)), backoff)
                    sleep_s = max(retry_after_s, capped)
                    if sleep_s > 0.0:
                        time.sleep(sleep_s)
                    attempt += 1
                    continue
                error = RuntimeError(
                    f"OpenRouter HTTP error status={exc.code} reason={exc.reason} latency_ms={latency_ms:.1f} "
                    f"variant={request_variant.name} detail={detail}"
                )
                if _is_parameter_routing_404(exc.code, detail):
                    last_error = error
                    break
                raise error from exc
            except (TimeoutError, socket.timeout, urllib.error.URLError) as exc:
                raise RuntimeError(f"OpenRouter request failed variant={request_variant.name}: {exc}") from exc
    if last_error is not None:
        raise RuntimeError(f"{last_error} attempted_variants={attempted_variants}")
    raise RuntimeError(f"OpenRouter request failed without a successful variant attempted_variants={attempted_variants}")


def _is_parameter_routing_404(status_code: int, detail: str) -> bool:
    if int(status_code) != 404:
        return False
    text = str(detail or "").lower()
    return "no endpoints found that can handle the requested parameters" in text


def evaluate_sample(
    *,
    sample: Sample,
    spec: TaskSpec,
    args: argparse.Namespace,
) -> dict[str, Any]:
    request_variants = build_request_variants(str(args.detail))
    payload_variants = [
        (
            request_variant,
            build_openrouter_payload(
                model=str(args.model),
                skill=spec.skill,
                prompt=sample.prompt,
                image_url=_to_data_url(sample.image, format="JPEG", quality=90),
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                max_tokens=int(args.max_tokens_point if spec.skill == "point" else args.max_tokens_detect),
                request_variant=request_variant,
            ),
        )
        for request_variant in request_variants
    ]
    try:
        answer_text, raw_response, latency_ms, request_variant_name = call_openrouter_chat_api(
            api_base=str(args.api_base),
            api_key=str(args.api_key),
            payload_variants=payload_variants,
            timeout=float(args.timeout),
            retry_429_max_retries=int(args.retry_429_max_retries),
            retry_429_backoff_s=float(args.retry_429_backoff_s),
            retry_429_max_backoff_s=float(args.retry_429_max_backoff_s),
        )
    except Exception as exc:
        return {
            "sample_index": sample.sample_index,
            "sample_id": sample.sample_id,
            "source_image_path": sample.source_image_path,
            "skill": spec.skill,
            "model": str(args.model),
            "prompt": sample.prompt,
            "ground_truth_boxes": _serialize_boxes(sample.ground_truth_boxes),
            "pred_boxes": [],
            "pred_points": [],
            "pred_count": 0,
            "gt_count": len(sample.ground_truth_boxes),
            "task_f1": None,
            "task_miou": None,
            "tp": None,
            "fp": None,
            "fn": None,
            "latency_ms": None,
            "parse_success": False,
            "failed": True,
            "error": str(exc),
            "request_variant": None,
        }

    pred_boxes, pred_points, parse_success = parse_openrouter_prediction(
        skill=spec.skill,
        answer_text=answer_text,
        raw_response=raw_response,
        image_width=sample.image.width,
        image_height=sample.image.height,
    )
    if spec.skill == "point":
        f1 = reward_f1_points(pred_points, sample.ground_truth_boxes)
        miou = 0.0
        tp, fp, fn = count_tp_fp_fn_points(pred_points, sample.ground_truth_boxes)
    else:
        f1 = reward_f1(pred_boxes, sample.ground_truth_boxes, iou_threshold=float(spec.iou_threshold))
        miou = reward_miou(pred_boxes, sample.ground_truth_boxes)
        tp, fp, fn = count_tp_fp_fn(pred_boxes, sample.ground_truth_boxes, iou_threshold=float(spec.iou_threshold))
    return {
        "sample_index": sample.sample_index,
        "sample_id": sample.sample_id,
        "source_image_path": sample.source_image_path,
        "skill": spec.skill,
        "model": str(args.model),
        "prompt": sample.prompt,
        "ground_truth_boxes": _serialize_boxes(sample.ground_truth_boxes),
        "pred_boxes": _serialize_boxes(pred_boxes),
        "pred_points": _serialize_points(pred_points),
        "pred_count": len(pred_points) if spec.skill == "point" else len(pred_boxes),
        "gt_count": len(sample.ground_truth_boxes),
        "task_f1": float(f1),
        "task_miou": float(miou),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "latency_ms": float(latency_ms),
        "parse_success": bool(parse_success),
        "failed": False,
        "error": None,
        "request_variant": request_variant_name,
    }


def merge_packet_samples_with_predictions(
    base_records: list[dict[str, Any]],
    prediction_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    predictions_by_id = {
        _normalize_id(str(record.get("sample_id") or "")): record
        for record in prediction_records
    }
    merged: list[dict[str, Any]] = []
    missing: list[str] = []
    for base_record in base_records:
        sample_id = str(base_record.get("sample_id") or "")
        prediction = predictions_by_id.get(_normalize_id(sample_id))
        if prediction is None:
            missing.append(sample_id)
            continue
        payload = dict(base_record)
        gpt_payload: dict[str, Any] = {
            "model": prediction["model"],
            "skill": prediction["skill"],
            "prompt": prediction["prompt"],
            "request_variant": prediction.get("request_variant"),
            "task_f1": prediction["task_f1"],
            "task_miou": prediction["task_miou"],
            "tp": prediction["tp"],
            "fp": prediction["fp"],
            "fn": prediction["fn"],
            "pred_count": prediction["pred_count"],
            "gt_count": prediction["gt_count"],
            "latency_ms": prediction["latency_ms"],
            "parse_success": prediction["parse_success"],
            "failed": prediction["failed"],
            "error": prediction["error"],
        }
        if prediction["skill"] == "point":
            gpt_payload["pred_points"] = list(prediction.get("pred_points", []))
        else:
            gpt_payload["pred_boxes"] = list(prediction.get("pred_boxes", []))
        payload["gpt_5_4"] = gpt_payload
        merged.append(payload)
    if missing:
        raise KeyError(f"Missing GPT-5.4 predictions for sample IDs: {missing}")
    return merged


def build_task_summary(
    *,
    spec: TaskSpec,
    reference_before: dict[str, Any],
    reference_after: dict[str, Any],
    gpt_metrics: dict[str, Any],
) -> dict[str, Any]:
    scope: dict[str, Any] = {
        "task": spec.name,
        "skill": spec.skill,
    }
    if spec.name == "aerial":
        scope.update(
            {
                "source": "cached_hf_parquet_packet_benchmark",
                "dataset_id": spec.dataset_id,
                "split": spec.split,
                "packet_path": str(spec.aerial_packet_path.relative_to(REPO_ROOT)) if spec.aerial_packet_path else None,
                "benchmark_artifact_path": str(spec.aerial_benchmark_path.relative_to(REPO_ROOT))
                if spec.aerial_benchmark_path
                else None,
                "subset_dataset_path": str(spec.aerial_subset_dataset_path.relative_to(REPO_ROOT))
                if spec.aerial_subset_dataset_path and spec.aerial_subset_dataset_path.exists()
                else None,
            }
        )
    elif spec.source_kind == "cached_hf_parquet":
        scope.update(
            {
                "source": "cached_hf_parquet",
                "dataset_id": spec.dataset_id,
                "split": spec.split,
            }
        )
    else:
        scope.update(
            {
                "source": "task_packet_subset",
                "packet_path": str(spec.aerial_packet_path.relative_to(REPO_ROOT)) if spec.aerial_packet_path else None,
                "subset_dataset_path": str(spec.aerial_subset_dataset_path.relative_to(REPO_ROOT))
                if spec.aerial_subset_dataset_path
                else None,
            }
        )
    return {
        "benchmark_scope": scope,
        "reference_source": load_reference_metrics(spec)[2],
        "reference_metrics": {
            "before": reference_before,
            "after": reference_after,
        },
        "gpt_5_4_metrics": {
            **normalize_metrics(gpt_metrics),
            "failed_samples": int(gpt_metrics.get("failed_samples", 0)),
            "parse_success_rate": float(gpt_metrics.get("parse_success_rate", 0.0)),
            "avg_latency_ms": float(gpt_metrics.get("avg_latency_ms", 0.0)),
        },
        "delta_vs_reference": {
            "before": compute_metric_deltas(reference_before, normalize_metrics(gpt_metrics)),
            "after": compute_metric_deltas(reference_after, normalize_metrics(gpt_metrics)),
        },
        "artifacts": {
            "metrics_json": str(spec.metrics_path.relative_to(REPO_ROOT)),
            "predictions_jsonl": str(spec.predictions_path.relative_to(REPO_ROOT)),
            "packet_samples_json": str(spec.packet_samples_path.relative_to(REPO_ROOT)),
        },
    }


def run_task(spec: TaskSpec, args: argparse.Namespace) -> dict[str, Any]:
    samples = load_task_samples(spec)
    prediction_records: list[dict[str, Any]] = []
    for index, sample in enumerate(samples, start=1):
        record = evaluate_sample(sample=sample, spec=spec, args=args)
        prediction_records.append(record)
        print(f"{spec.name}: processed {index}/{len(samples)} sample_id={sample.sample_id}", flush=True)

    metrics = aggregate_prediction_metrics(prediction_records, skill=spec.skill, iou_threshold=float(spec.iou_threshold))
    metrics_payload = {
        "task": spec.name,
        "skill": spec.skill,
        "model": str(args.model),
        "benchmark_scope": {
            "source_kind": spec.source_kind,
            "dataset_id": spec.dataset_id,
            "split": spec.split,
        },
        **metrics,
        "config": {
            "config": str(args.config),
            "api_base": str(args.api_base),
            "model": str(args.model),
            "detail": str(args.detail),
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "max_tokens_detect": int(args.max_tokens_detect),
            "max_tokens_point": int(args.max_tokens_point),
            "timeout": float(args.timeout),
            "retry_429_max_retries": int(args.retry_429_max_retries),
            "retry_429_backoff_s": float(args.retry_429_backoff_s),
            "retry_429_max_backoff_s": float(args.retry_429_max_backoff_s),
        },
    }
    base_records = [sample.base_record for sample in samples]
    packet_samples_payload = merge_packet_samples_with_predictions(base_records, prediction_records)
    _write_json(spec.metrics_path, metrics_payload)
    spec.predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with spec.predictions_path.open("w", encoding="utf-8") as handle:
        for record in prediction_records:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")
    _write_json(spec.packet_samples_path, packet_samples_payload)
    task_packet_common.refresh_simple_task_samples(
        task_dir=spec.task_dir,
        skill=spec.skill,
        repo_root=REPO_ROOT,
        gpt_packet_samples=packet_samples_payload,
    )
    reference_before, reference_after, _ = load_reference_metrics(spec)
    return build_task_summary(
        spec=spec,
        reference_before=reference_before,
        reference_after=reference_after,
        gpt_metrics=metrics_payload,
    )


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    if str(args.env_file).strip():
        load_dotenv(args.env_file, override=False)
    args.api_key = _resolve_openrouter_api_key(str(args.api_key), str(args.api_key_env_var))

    registry = build_task_registry(REPO_ROOT)
    unknown_tasks = [task for task in args.tasks if task not in registry]
    if unknown_tasks:
        raise ValueError(f"Unknown task(s): {unknown_tasks}. Available: {sorted(registry)}")

    summary: dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "model": str(args.model),
        "api_base": str(args.api_base),
        "tasks": {},
    }
    for task_name in args.tasks:
        spec = registry[task_name]
        summary["tasks"][task_name] = run_task(spec, args)
    summary_path = REPO_ROOT / "outputs" / "task_sample_packets" / "openrouter_gpt_5_4_summary.json"
    _write_json(summary_path, summary)
    print(f"wrote summary -> {summary_path}")


if __name__ == "__main__":
    main()
