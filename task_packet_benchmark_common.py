from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment


REPO_ROOT = Path(__file__).resolve().parent

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
    packet_image_path: Optional[str]
    prompt: str
    task_type: str
    notes: str
    timestamp: str
    image: Image.Image
    ground_truth_boxes: list[Box]
    base_record: dict[str, Any]


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
    finetune_id: str = ""
    checkpoint_step: Optional[int] = None

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


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _resolve_repo_path(raw_path: str | Path, *, repo_root: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return repo_root / path


def serialize_boxes(boxes: list[Box]) -> list[dict[str, float]]:
    return [
        {
            "x_min": float(box.x_min),
            "y_min": float(box.y_min),
            "x_max": float(box.x_max),
            "y_max": float(box.y_max),
        }
        for box in boxes
    ]


def serialize_points(points: list[Point]) -> list[dict[str, float]]:
    return [{"x": float(point.x), "y": float(point.y)} for point in points]


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


def _normalize_id(value: str) -> str:
    return str(value or "").strip().lower()


def _repo_relative(path: Path, *, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


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


def _find_latest_json_glob(task_dir: Path, pattern: str) -> Optional[Path]:
    matches = sorted(task_dir.glob(pattern))
    return matches[-1] if matches else None


def _count_packet_images(task_dir: Path) -> int:
    imgs_dir = task_dir / "imgs"
    if not imgs_dir.exists():
        return 0
    return sum(1 for path in imgs_dir.iterdir() if path.is_file())


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


def _player_packet_sample_id(packet_image_path: Path) -> str:
    stem = packet_image_path.stem
    if "_" not in stem:
        return stem
    return stem.split("_", 1)[1]


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


def _load_dataset_from_disk(dataset_path: Path) -> Any:
    from datasets import load_from_disk

    return load_from_disk(str(dataset_path))


def _image_from_bytes_payload(image_payload: dict[str, Any]) -> Image.Image:
    raw_bytes = image_payload.get("bytes")
    if raw_bytes is None:
        raise ValueError("Image bytes missing from dataset row")
    return Image.open(io.BytesIO(raw_bytes)).convert("RGB")


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
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    boxes: list[Box] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        x_min = item.get("x_min", item.get("xmin"))
        y_min = item.get("y_min", item.get("ymin"))
        x_max = item.get("x_max", item.get("xmax"))
        y_max = item.get("y_max", item.get("ymax"))
        if None not in (x_min, y_min, x_max, y_max):
            try:
                left = float(x_min)
                top = float(y_min)
                right = float(x_max)
                bottom = float(y_max)
            except (TypeError, ValueError):
                continue
            if max(abs(left), abs(top), abs(right), abs(bottom)) > 1.5 and width > 0 and height > 0:
                left /= float(width)
                top /= float(height)
                right /= float(width)
                bottom /= float(height)
            left = max(0.0, min(1.0, left))
            top = max(0.0, min(1.0, top))
            right = max(0.0, min(1.0, right))
            bottom = max(0.0, min(1.0, bottom))
            if right > left and bottom > top:
                boxes.append(Box(x_min=left, y_min=top, x_max=right, y_max=bottom))
            continue
        box = item.get("box")
        if isinstance(box, dict):
            try:
                x_center = float(box["x_center"])
                y_center = float(box["y_center"])
                box_w = float(box["width"])
                box_h = float(box["height"])
            except (KeyError, TypeError, ValueError):
                continue
            left = x_center - box_w / 2.0
            top = y_center - box_h / 2.0
            right = x_center + box_w / 2.0
            bottom = y_center + box_h / 2.0
            left = max(0.0, min(1.0, left))
            top = max(0.0, min(1.0, top))
            right = max(0.0, min(1.0, right))
            bottom = max(0.0, min(1.0, bottom))
            if right > left and bottom > top:
                boxes.append(Box(x_min=left, y_min=top, x_max=right, y_max=bottom))
    return boxes


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


def reward_f1_points(points: list[Point], ground_truth: list[Box]) -> float:
    tp, fp, fn = count_tp_fp_fn_points(points, ground_truth)
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0
    denom = (2.0 * float(tp)) + float(fp) + float(fn)
    return 0.0 if denom <= 0.0 else (2.0 * float(tp)) / denom


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
        "iou_threshold": float(iou_threshold),
    }


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


def build_task_registry(repo_root: Path = REPO_ROOT) -> dict[str, TaskSpec]:
    repo_root = Path(repo_root)
    packet_root = repo_root / "outputs" / "task_sample_packets"

    state_finetune_id = "01KFYJ3T93RST3147ANRCJ8VA2"
    state_checkpoint_step = 139
    player_finetune_id = "01KGNDXEC2XF529TCHYVZFT9JA"
    player_checkpoint_step = 60
    aerial_finetune_id = "01KMRFGRKPRQYCEXCTQ05Q4D7H"
    aerial_checkpoint_step = 148

    state_farm_dir = packet_root / "state_farm"
    state_farm_ref_dir = _find_reference_dir(state_farm_dir)
    if state_farm_ref_dir is None:
        legacy_state_dir = (
            repo_root
            / "_DEPICATED_MDstatefarmRL"
            / "outputs"
            / f"compare_{state_finetune_id}_step{state_checkpoint_step}_validation"
        )
        if legacy_state_dir.exists():
            state_farm_ref_dir = legacy_state_dir

    player_dir = packet_root / "player_with_ball"
    player_compare_path = _find_single_json_glob(player_dir, "*_baseline_vs_ft/*.json")
    if player_compare_path is None:
        player_compare_path = _find_latest_json_glob(
            repo_root / "_DEPICATED_MDBallHolder" / "outputs",
            f"staging_{player_finetune_id}_baseline_vs_step{player_checkpoint_step}_*_metrics.json",
        )
    player_readme_path = player_dir / "README.md"

    aerial_dir = packet_root / "aerial"
    aerial_packet_path = aerial_dir / "samples.before_after.json"
    aerial_benchmark_path = _find_single_json_glob(aerial_dir, "benchmark_*.json")
    if aerial_benchmark_path is None:
        legacy_aerial_path = (
            repo_root
            / "aerial_airport"
            / "outputs"
            / "benchmarks"
            / f"benchmark_aerial_airport_point_hf_point_v2_{aerial_finetune_id}_step{aerial_checkpoint_step}.json"
        )
        if legacy_aerial_path.exists():
            aerial_benchmark_path = legacy_aerial_path

    return {
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
            finetune_id=state_finetune_id,
            checkpoint_step=state_checkpoint_step,
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
            finetune_id=player_finetune_id,
            checkpoint_step=player_checkpoint_step,
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
            finetune_id=aerial_finetune_id,
            checkpoint_step=aerial_checkpoint_step,
        ),
    }


def load_state_farm_samples(spec: TaskSpec, *, repo_root: Path = REPO_ROOT) -> list[Sample]:
    parquet_path = _find_cached_parquet_path(str(spec.dataset_id), str(spec.split))
    rows = _read_parquet_rows(parquet_path)
    row_by_path = {
        _normalize_id(str(row.get("image", {}).get("path") or "")): row
        for row in rows
        if isinstance(row.get("image"), dict)
    }

    packet_images = _list_state_farm_packet_images(spec.task_dir)
    if packet_images:
        samples: list[Sample] = []
        for packet_index, sample_id, packet_image_path in packet_images:
            source_image_path = f"{sample_id}.jpg"
            row = row_by_path.get(_normalize_id(source_image_path))
            if row is None:
                raise KeyError(f"State Farm source image missing from cached parquet: {source_image_path}")
            image = _image_from_bytes_payload(row["image"])
            gt_boxes = _parse_boxes_from_annotation(row.get("answer_boxes"), width=image.width, height=image.height)
            prompt = str(row.get("prompt") or spec.prompt)
            packet_rel = _repo_relative(packet_image_path, repo_root=repo_root)
            base_record = {
                "sample_index": int(packet_index),
                "sample_id": sample_id,
                "source_image_path": source_image_path,
                "packet_image_path": packet_rel,
                "prompt": prompt,
                "task_type": str(row.get("type") or "region"),
                "ground_truth_boxes": serialize_boxes(gt_boxes),
                "notes": str(row.get("notes") or ""),
                "timestamp": str(row.get("timestamp") or ""),
            }
            samples.append(
                Sample(
                    sample_index=int(packet_index),
                    sample_id=sample_id,
                    source_image_path=source_image_path,
                    packet_image_path=packet_rel,
                    prompt=prompt,
                    task_type=str(row.get("type") or "region"),
                    notes=str(row.get("notes") or ""),
                    timestamp=str(row.get("timestamp") or ""),
                    image=image,
                    ground_truth_boxes=gt_boxes,
                    base_record=base_record,
                )
            )
        return samples

    if spec.reference_samples_path is None or not spec.reference_samples_path.exists():
        raise FileNotFoundError(
            f"State Farm packet samples not found. reference_samples_path={spec.reference_samples_path} packet_imgs_dir={spec.task_dir / 'imgs'}"
        )

    packet_samples = _load_json(spec.reference_samples_path)
    if not isinstance(packet_samples, list):
        raise ValueError(f"Expected list in {spec.reference_samples_path}")
    samples = []
    for index, record in enumerate(packet_samples, start=1):
        source_image_path = str(record.get("source_image_path") or "")
        row = row_by_path.get(_normalize_id(source_image_path))
        if row is None:
            raise KeyError(f"State Farm source image missing from cached parquet: {source_image_path}")
        image = _image_from_bytes_payload(row["image"])
        gt_boxes = [
            Box(
                x_min=float(item["x_min"]),
                y_min=float(item["y_min"]),
                x_max=float(item["x_max"]),
                y_max=float(item["y_max"]),
            )
            for item in record.get("ground_truth_boxes", [])
        ]
        packet_rel = str(record.get("packet_image_path") or "") or None
        base_record = dict(record)
        if packet_rel is not None:
            base_record["packet_image_path"] = packet_rel
        samples.append(
            Sample(
                sample_index=int(record.get("sample_index", index)),
                sample_id=str(record.get("sample_id") or Path(source_image_path).stem),
                source_image_path=source_image_path,
                packet_image_path=packet_rel,
                prompt=str(record.get("prompt") or row.get("prompt") or spec.prompt),
                task_type=str(record.get("task_type") or row.get("type") or "region"),
                notes=str(record.get("notes") or ""),
                timestamp=str(record.get("timestamp") or ""),
                image=image,
                ground_truth_boxes=gt_boxes,
                base_record=base_record,
            )
        )
    return samples


def load_player_with_ball_samples(spec: TaskSpec, *, repo_root: Path = REPO_ROOT) -> list[Sample]:
    parquet_path = _find_cached_parquet_path(str(spec.dataset_id), str(spec.split))
    rows = _read_parquet_rows(parquet_path)
    row_by_sample_id = {
        _normalize_id(Path(str(row.get("image", {}).get("path") or f"{index:04d}.jpg")).stem): row
        for index, row in enumerate(rows, start=1)
    }

    compare_payload = _load_json(spec.reference_compare_path) if spec.reference_compare_path and spec.reference_compare_path.exists() else {}
    compare_samples_raw = compare_payload.get("samples") if isinstance(compare_payload, dict) else None
    compare_samples: dict[str, dict[str, Any]] = {}
    if isinstance(compare_samples_raw, list):
        for item in compare_samples_raw:
            if not isinstance(item, dict):
                continue
            sample_id = str(item.get("sample_id") or "")
            if sample_id:
                compare_samples[_normalize_id(sample_id)] = item

    packet_images = _list_numbered_packet_images(spec.task_dir)
    if packet_images:
        samples: list[Sample] = []
        for packet_index, packet_image_path in packet_images:
            sample_id = _player_packet_sample_id(packet_image_path)
            row = row_by_sample_id.get(_normalize_id(sample_id))
            if row is None:
                raise KeyError(f"Player With Ball source image missing from cached parquet: {sample_id}")
            image = _image_from_bytes_payload(row["image"])
            width, height = image.size
            source_image_path = str(row.get("image", {}).get("path") or f"{sample_id}.jpg")
            gt_boxes = _parse_boxes_from_annotation(row.get("answer_boxes"), width=width, height=height)
            prompt = str(row.get("prompt") or spec.prompt)
            packet_rel = _repo_relative(packet_image_path, repo_root=repo_root)
            base_record: dict[str, Any] = {
                "sample_index": int(packet_index),
                "sample_id": sample_id,
                "source_image_path": source_image_path,
                "packet_image_path": packet_rel,
                "prompt": prompt,
                "task_type": str(row.get("type") or "region"),
                "ground_truth_boxes": serialize_boxes(gt_boxes),
                "notes": str(row.get("notes") or ""),
                "timestamp": str(row.get("timestamp") or ""),
            }
            compare_record = compare_samples.get(_normalize_id(sample_id))
            if compare_record is not None:
                if "before" in compare_record:
                    base_record["before"] = compare_record["before"]
                if "after" in compare_record:
                    base_record["after"] = compare_record["after"]
            samples.append(
                Sample(
                    sample_index=int(packet_index),
                    sample_id=sample_id,
                    source_image_path=source_image_path,
                    packet_image_path=packet_rel,
                    prompt=prompt,
                    task_type=str(row.get("type") or "region"),
                    notes=str(row.get("notes") or ""),
                    timestamp=str(row.get("timestamp") or ""),
                    image=image,
                    ground_truth_boxes=gt_boxes,
                    base_record=base_record,
                )
            )
        return samples

    samples = []
    for index, row in enumerate(rows, start=1):
        image = _image_from_bytes_payload(row["image"])
        width, height = image.size
        source_image_path = str(row.get("image", {}).get("path") or f"{index:04d}.jpg")
        sample_id = Path(source_image_path).stem
        gt_boxes = _parse_boxes_from_annotation(row.get("answer_boxes"), width=width, height=height)
        prompt = str(row.get("prompt") or spec.prompt)
        base_record: dict[str, Any] = {
            "sample_index": index,
            "sample_id": sample_id,
            "source_image_path": source_image_path,
            "packet_image_path": None,
            "prompt": prompt,
            "task_type": str(row.get("type") or "region"),
            "ground_truth_boxes": serialize_boxes(gt_boxes),
            "notes": str(row.get("notes") or ""),
            "timestamp": str(row.get("timestamp") or ""),
        }
        compare_record = compare_samples.get(_normalize_id(sample_id))
        if compare_record is not None:
            if "before" in compare_record:
                base_record["before"] = compare_record["before"]
            if "after" in compare_record:
                base_record["after"] = compare_record["after"]
        samples.append(
            Sample(
                sample_index=index,
                sample_id=sample_id,
                source_image_path=source_image_path,
                packet_image_path=None,
                prompt=prompt,
                task_type=str(row.get("type") or "region"),
                notes=str(row.get("notes") or ""),
                timestamp=str(row.get("timestamp") or ""),
                image=image,
                ground_truth_boxes=gt_boxes,
                base_record=base_record,
            )
        )
    return samples


def load_aerial_samples(spec: TaskSpec, *, repo_root: Path = REPO_ROOT) -> list[Sample]:
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

    packet_images = _list_numbered_packet_images(spec.task_dir)
    if packet_images:
        ordered_rows: list[tuple[int, Path, dict[str, Any]]] = []
        for packet_index, packet_image_path in packet_images:
            key = _canonical_aerial_sample_key(packet_image_path.name)
            row = row_by_key.get(key)
            if row is None:
                raise KeyError(f"Aerial packet image missing from cached parquets: {packet_image_path.name}")
            ordered_rows.append((packet_index, packet_image_path, row))
    elif spec.aerial_benchmark_path is not None and spec.aerial_benchmark_path.exists():
        ordered_keys = _load_aerial_benchmark_order(spec.aerial_benchmark_path)
        ordered_rows = []
        for index, key in enumerate(ordered_keys, start=1):
            row = row_by_key.get(key)
            if row is None:
                raise KeyError(f"Aerial benchmark rows missing from cached parquet: {key}")
            ordered_rows.append((index, Path(""), row))
    else:
        raise FileNotFoundError(
            f"Aerial packet sources not found. packet_imgs_dir={spec.task_dir / 'imgs'} benchmark_json={spec.aerial_benchmark_path}"
        )

    samples: list[Sample] = []
    for packet_index, packet_image_path, row in ordered_rows:
        image = _image_from_bytes_payload(row["image"])
        width, height = image.size
        source_image_id = str(row.get("source_image_id") or Path(str(row.get("image", {}).get("path") or "")).stem)
        source_image_path = str(row.get("image", {}).get("path") or source_image_id)
        gt_boxes = _parse_boxes_from_annotation(row.get("answer_boxes"), width=width, height=height)
        prompt = str(row.get("prompt") or spec.prompt)
        packet_rel = _repo_relative(packet_image_path, repo_root=repo_root) if packet_image_path else None
        base_record = {
            "sample_index": int(packet_index),
            "sample_id": source_image_id,
            "source_image_id": source_image_id,
            "source_image_path": source_image_path,
            "packet_image_path": packet_rel,
            "prompt": prompt,
            "task_type": "point",
            "ground_truth_boxes": serialize_boxes(gt_boxes),
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
        }
        samples.append(
            Sample(
                sample_index=int(packet_index),
                sample_id=source_image_id,
                source_image_path=source_image_path,
                packet_image_path=packet_rel,
                prompt=prompt,
                task_type="point",
                notes="",
                timestamp="",
                image=image,
                ground_truth_boxes=gt_boxes,
                base_record=base_record,
            )
        )
    return samples


def load_task_samples(spec: TaskSpec, *, repo_root: Path = REPO_ROOT) -> list[Sample]:
    if spec.name == "state_farm":
        return load_state_farm_samples(spec, repo_root=repo_root)
    if spec.name == "player_with_ball":
        return load_player_with_ball_samples(spec, repo_root=repo_root)
    if spec.name == "aerial":
        return load_aerial_samples(spec, repo_root=repo_root)
    raise ValueError(f"Unsupported task: {spec.name}")


def _iter_artifact_dirs(task_dir: Path) -> list[Path]:
    candidates = [task_dir, task_dir / "raw_json"]
    seen: set[str] = set()
    out: list[Path] = []
    for candidate in candidates:
        key = str(candidate.resolve())
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            out.append(candidate)
    return out


def _find_latest_rerun_manifest_path(task_dir: Path) -> Optional[Path]:
    manifests: list[Path] = []
    reruns_dir = task_dir / "reruns"
    if reruns_dir.exists():
        manifests.extend(sorted(reruns_dir.glob("*/manifest.json")))
    for artifact_dir in _iter_artifact_dirs(task_dir):
        direct_manifest = artifact_dir / "manifest.json"
        if direct_manifest.exists():
            manifests.append(direct_manifest)
    if not manifests:
        return None
    ranked: list[tuple[str, str, Path]] = []
    for manifest_path in manifests:
        generated_utc = ""
        try:
            payload = _load_json(manifest_path)
            if isinstance(payload, dict):
                generated_utc = str(payload.get("generated_utc") or "")
        except (OSError, json.JSONDecodeError, ValueError):
            generated_utc = ""
        ranked.append((generated_utc, manifest_path.parent.name, manifest_path))
    return max(ranked, key=lambda item: (item[0], item[1]))[2]


def _find_latest_gpt_packet_samples_path(task_dir: Path) -> Optional[Path]:
    matches: list[Path] = []
    for artifact_dir in _iter_artifact_dirs(task_dir):
        matches.extend(sorted(artifact_dir.glob("openrouter_gpt_5_4_*.packet_samples.json")))
    return matches[-1] if matches else None


def _simple_prediction_value(record: dict[str, Any], *, skill: str) -> list[dict[str, Any]]:
    pred_key = "pred_points" if skill == "point" else "pred_boxes"
    raw_value = record.get(pred_key)
    if not isinstance(raw_value, list):
        return []
    return [item for item in raw_value if isinstance(item, dict)]


def _fallback_sibling_records_path(manifest_path: Path, raw_path: str, *, filename: str) -> Path:
    raw_name = Path(str(raw_path or "")).name
    if raw_name:
        candidate = manifest_path.parent / raw_name
        if candidate.exists():
            return candidate
    candidate = manifest_path.parent / filename
    if candidate.exists():
        return candidate
    return manifest_path.parent / raw_name if raw_name else candidate


def _load_manifest_records(manifest_path: Path, *, repo_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest_payload = _load_json(manifest_path)
    if not isinstance(manifest_payload, dict):
        raise ValueError(f"Expected object in {manifest_path}")
    baseline_raw_path = str(manifest_payload.get("baseline", {}).get("records_jsonl") or "")
    checkpoint_raw_path = str(manifest_payload.get("checkpoint", {}).get("records_jsonl") or "")

    baseline_path = _resolve_repo_path(baseline_raw_path, repo_root=repo_root)
    checkpoint_path = _resolve_repo_path(checkpoint_raw_path, repo_root=repo_root)
    if not baseline_path.exists():
        baseline_path = _fallback_sibling_records_path(manifest_path, baseline_raw_path, filename="baseline.records.jsonl")
    if not checkpoint_path.exists():
        checkpoint_path = _fallback_sibling_records_path(manifest_path, checkpoint_raw_path, filename="checkpoint.records.jsonl")
    return _load_jsonl(baseline_path), _load_jsonl(checkpoint_path)


def _find_manual_checkpoint_bbox_path(task_dir: Path) -> Optional[Path]:
    preferred = [
        task_dir / "use this for checkpoint bboxs.json",
        task_dir / "raw_json" / "use this for checkpoint bboxs.json",
    ]
    for path in preferred:
        if path.exists():
            return path
    matches = sorted(task_dir.rglob("use this for checkpoint bboxs.json"))
    return matches[0] if matches else None


def _apply_manual_bbox_overrides(
    *,
    skill: str,
    baseline_records: list[dict[str, Any]],
    checkpoint_records: list[dict[str, Any]],
    manual_path: Optional[Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if manual_path is None or not manual_path.exists():
        return baseline_records, checkpoint_records
    payload = _load_json(manual_path)
    if not isinstance(payload, list):
        raise ValueError(f"Expected list in {manual_path}")

    pred_key = "pred_points" if skill == "point" else "pred_boxes"
    baseline_manual: dict[str, list[dict[str, Any]]] = {}
    checkpoint_manual: dict[str, list[dict[str, Any]]] = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        sample_id = _normalize_id(str(item.get("sample_id") or ""))
        if not sample_id:
            continue
        baseline_pred = item.get("baseline", {}).get(pred_key)
        after_pred = item.get("after", {}).get(pred_key)
        if isinstance(baseline_pred, list):
            baseline_manual[sample_id] = [pred for pred in baseline_pred if isinstance(pred, dict)]
        if isinstance(after_pred, list):
            checkpoint_manual[sample_id] = [pred for pred in after_pred if isinstance(pred, dict)]

    def apply(records: list[dict[str, Any]], overrides: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
        updated: list[dict[str, Any]] = []
        for record in records:
            sample_id = _normalize_id(str(record.get("sample_id") or ""))
            override = overrides.get(sample_id)
            if override is None:
                updated.append(record)
                continue
            needs_override = bool(record.get("failed")) or not _simple_prediction_value(record, skill=skill)
            if not needs_override:
                updated.append(record)
                continue
            patched = dict(record)
            patched[pred_key] = override
            patched["failed"] = False
            patched["error"] = None
            updated.append(patched)
        return updated

    return apply(baseline_records, baseline_manual), apply(checkpoint_records, checkpoint_manual)


def _task_relative_image_path(
    raw_path: Any,
    *,
    task_dir: Path,
    repo_root: Path,
) -> str:
    text = str(raw_path or "").strip()
    if not text:
        return ""
    path = Path(text)
    if not path.is_absolute() and not text.startswith(str(task_dir.name) + "/"):
        direct_candidate = task_dir / path
        if direct_candidate.exists():
            return str(path)
    try:
        resolved = _resolve_repo_path(path, repo_root=repo_root).resolve()
        return str(resolved.relative_to(task_dir.resolve()))
    except (ValueError, FileNotFoundError):
        return text


def build_simple_task_samples(
    *,
    task_dir: Path,
    repo_root: Path,
    skill: str,
    gpt_packet_samples: list[dict[str, Any]],
    baseline_records: list[dict[str, Any]],
    checkpoint_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    baseline_by_id = {
        _normalize_id(str(record.get("sample_id") or "")): record
        for record in baseline_records
        if isinstance(record, dict)
    }
    checkpoint_by_id = {
        _normalize_id(str(record.get("sample_id") or "")): record
        for record in checkpoint_records
        if isinstance(record, dict)
    }
    merged: list[dict[str, Any]] = []
    missing: list[str] = []
    for packet_sample in gpt_packet_samples:
        if not isinstance(packet_sample, dict):
            continue
        sample_id = str(packet_sample.get("sample_id") or "")
        normalized_id = _normalize_id(sample_id)
        baseline_record = baseline_by_id.get(normalized_id)
        checkpoint_record = checkpoint_by_id.get(normalized_id)
        gpt_record = packet_sample.get("gpt_5_4")
        if baseline_record is None or checkpoint_record is None or not isinstance(gpt_record, dict):
            missing.append(sample_id)
            continue
        ground_truth = packet_sample.get("ground_truth_boxes")
        if not isinstance(ground_truth, list):
            ground_truth = checkpoint_record.get("ground_truth_boxes")
        if not isinstance(ground_truth, list):
            ground_truth = baseline_record.get("ground_truth_boxes")
        image_path = (
            packet_sample.get("packet_image_path")
            or checkpoint_record.get("packet_image_path")
            or baseline_record.get("packet_image_path")
            or packet_sample.get("source_image_path")
            or checkpoint_record.get("source_image_path")
            or baseline_record.get("source_image_path")
            or ""
        )
        merged.append(
            {
                "image": _task_relative_image_path(image_path, task_dir=task_dir, repo_root=repo_root),
                "prompt": str(
                    packet_sample.get("prompt")
                    or checkpoint_record.get("prompt")
                    or baseline_record.get("prompt")
                    or ""
                ),
                "gt": ground_truth if isinstance(ground_truth, list) else [],
                "baseline": _simple_prediction_value(baseline_record, skill=skill),
                "finetune": _simple_prediction_value(checkpoint_record, skill=skill),
                "gpt_5.4": _simple_prediction_value(gpt_record, skill=skill),
            }
        )
    if missing:
        raise KeyError(f"Missing baseline/checkpoint/GPT data for sample IDs: {missing}")
    return merged


def refresh_simple_task_samples(
    *,
    task_dir: Path,
    skill: str,
    repo_root: Path = REPO_ROOT,
    gpt_packet_samples: Optional[list[dict[str, Any]]] = None,
    baseline_records: Optional[list[dict[str, Any]]] = None,
    checkpoint_records: Optional[list[dict[str, Any]]] = None,
) -> Optional[Path]:
    task_dir = Path(task_dir)
    repo_root = Path(repo_root)

    if gpt_packet_samples is None:
        gpt_packet_path = _find_latest_gpt_packet_samples_path(task_dir)
        if gpt_packet_path is None:
            return None
        payload = _load_json(gpt_packet_path)
        if not isinstance(payload, list):
            raise ValueError(f"Expected list in {gpt_packet_path}")
        gpt_packet_samples = payload

    if baseline_records is None or checkpoint_records is None:
        manifest_path = _find_latest_rerun_manifest_path(task_dir)
        if manifest_path is None:
            return None
        baseline_records, checkpoint_records = _load_manifest_records(manifest_path, repo_root=repo_root)

    manual_path = _find_manual_checkpoint_bbox_path(task_dir)
    baseline_records, checkpoint_records = _apply_manual_bbox_overrides(
        skill=skill,
        baseline_records=baseline_records,
        checkpoint_records=checkpoint_records,
        manual_path=manual_path,
    )

    merged = build_simple_task_samples(
        task_dir=task_dir,
        repo_root=repo_root,
        skill=skill,
        gpt_packet_samples=gpt_packet_samples,
        baseline_records=baseline_records,
        checkpoint_records=checkpoint_records,
    )
    output_path = task_dir / "samples.json"
    _write_json(output_path, merged)
    return output_path


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


def load_reference_metrics(spec: TaskSpec, *, repo_root: Path = REPO_ROOT) -> tuple[dict[str, Any], dict[str, Any], str]:
    repo_root = Path(repo_root).resolve()
    if spec.name == "state_farm":
        if spec.reference_metrics_path is not None and spec.reference_metrics_path.exists():
            payload = _load_json(spec.reference_metrics_path)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected object in {spec.reference_metrics_path}")
            return (
                normalize_metrics(dict(payload.get("baseline") or {})),
                normalize_metrics(dict(payload.get("checkpoint") or {})),
                _repo_relative(spec.reference_metrics_path, repo_root=repo_root),
            )
        if spec.reference_readme_path is None or not spec.reference_readme_path.exists():
            raise FileNotFoundError(
                f"State Farm reference metrics not found. metrics_json={spec.reference_metrics_path} readme={spec.reference_readme_path}"
            )
        before, after = _parse_readme_benchmark_table(
            spec.reference_readme_path,
            sample_count_fallback=len(_list_state_farm_packet_images(spec.task_dir)),
        )
        return before, after, _repo_relative(spec.reference_readme_path, repo_root=repo_root)

    if spec.name == "player_with_ball":
        if spec.reference_compare_path is not None and spec.reference_compare_path.exists():
            payload = _load_json(spec.reference_compare_path)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected object in {spec.reference_compare_path}")
            return (
                normalize_metrics(dict(payload.get("baseline") or {})),
                normalize_metrics(dict(payload.get("candidate") or {})),
                _repo_relative(spec.reference_compare_path, repo_root=repo_root),
            )
        if spec.reference_readme_path is None or not spec.reference_readme_path.exists():
            raise FileNotFoundError(
                f"Ball Holder reference metrics not found. compare_json={spec.reference_compare_path} readme={spec.reference_readme_path}"
            )
        before, after = _parse_readme_benchmark_table(
            spec.reference_readme_path,
            sample_count_fallback=_count_packet_images(spec.task_dir),
        )
        return before, after, _repo_relative(spec.reference_readme_path, repo_root=repo_root)

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
                _repo_relative(spec.aerial_packet_path, repo_root=repo_root),
            )
        if spec.aerial_benchmark_path is not None and spec.aerial_benchmark_path.exists():
            payload = _load_json(spec.aerial_benchmark_path)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected object in {spec.aerial_benchmark_path}")
            return (
                normalize_metrics(dict(payload.get("baseline") or {})),
                normalize_metrics(dict(payload.get("candidate") or {})),
                _repo_relative(spec.aerial_benchmark_path, repo_root=repo_root),
            )
        if spec.reference_readme_path is None or not spec.reference_readme_path.exists():
            raise FileNotFoundError(
                f"Aerial reference metrics not found. packet_json={spec.aerial_packet_path} benchmark_json={spec.aerial_benchmark_path} readme={spec.reference_readme_path}"
            )
        before, after = _parse_readme_benchmark_table(
            spec.reference_readme_path,
            sample_count_fallback=_count_packet_images(spec.task_dir),
        )
        return before, after, _repo_relative(spec.reference_readme_path, repo_root=repo_root)

    raise ValueError(f"Unsupported task: {spec.name}")
