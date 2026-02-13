"""Benchmark State Farm detection on local val set with a hosted Roboflow model.

Requires:
  pip install pillow numpy python-dotenv requests
"""

from __future__ import annotations

import argparse
import io
import json
import os
import time
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


DEFAULT_ROBOFLOW_BASE_URL = "https://detect.roboflow.com"


@dataclass(frozen=True)
class Box:
    x_min: float
    y_min: float
    x_max: float
    y_max: float


def _normalize_label(value: Optional[str]) -> str:
    if not value:
        return ""
    return " ".join(str(value).strip().lower().split())


def _labels_match(predicted_label: str, expected_label: str) -> bool:
    pred = _normalize_label(predicted_label)
    exp = _normalize_label(expected_label)
    if not exp:
        return True
    if not pred:
        return False
    if pred == exp:
        return True
    return exp in pred or pred in exp


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def _box_from_normalized(x_min: float, y_min: float, x_max: float, y_max: float) -> Box:
    x_min = _clamp(float(x_min), 0.0, 1.0)
    y_min = _clamp(float(y_min), 0.0, 1.0)
    x_max = _clamp(float(x_max), 0.0, 1.0)
    y_max = _clamp(float(y_max), 0.0, 1.0)
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("Invalid bbox after clipping")
    return Box(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)


def _parse_val_boxes(answer: Optional[list]) -> List[Box]:
    if not answer:
        return []
    boxes: List[Box] = []
    for item in answer:
        box = item.get("box") if isinstance(item, dict) else None
        if not isinstance(box, dict):
            continue
        try:
            x_center = float(box["x_center"])
            y_center = float(box["y_center"])
            width = float(box["width"])
            height = float(box["height"])
        except (KeyError, TypeError, ValueError):
            continue
        x_min = x_center - width / 2.0
        y_min = y_center - height / 2.0
        x_max = x_center + width / 2.0
        y_max = y_center + height / 2.0
        try:
            boxes.append(_box_from_normalized(x_min, y_min, x_max, y_max))
        except ValueError:
            continue
    return boxes


def _iter_val_samples(
    *,
    val_json_path: str,
    image_dir: str,
    max_boxes: Optional[int],
) -> Iterable[tuple[Image.Image, List[Box], str]]:
    with open(val_json_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    val_base_dir = os.path.dirname(os.path.abspath(val_json_path))
    for item in data:
        filename = item.get("filename")
        if not filename:
            continue
        if os.path.isabs(filename):
            image_path = filename
        else:
            image_path = os.path.join(image_dir, filename)
            if not os.path.exists(image_path):
                fallback_candidates = [
                    os.path.join(val_base_dir, filename),
                    os.path.join(image_dir, os.path.basename(filename)),
                ]
                for candidate in fallback_candidates:
                    if os.path.exists(candidate):
                        image_path = candidate
                        break
        try:
            image = Image.open(image_path).convert("RGB")
        except (FileNotFoundError, OSError) as exc:
            print(f"eval image load failed: {image_path} ({exc}). skipping sample.")
            continue
        boxes = _parse_val_boxes(item.get("answer"))
        if max_boxes is not None:
            boxes = boxes[:max_boxes]
        yield image, boxes, filename


def _with_progress(
    samples: Iterable[tuple[Image.Image, List[Box], str]],
    *,
    disable: bool,
) -> Iterator[tuple[Image.Image, List[Box], str]]:
    if disable or tqdm is None:
        yield from samples
        return
    yield from tqdm(samples, desc="roboflow eval", unit="img")


def _draw_box(
    canvas: Image.Image,
    box: Box,
    width: int,
    height: int,
    color: tuple[int, int, int, int],
    label: Optional[str] = None,
    outline_color: Optional[tuple[int, int, int]] = None,
    line_width: int = 8,
    fill_alpha: int = 18,
    font: Optional[ImageFont.ImageFont] = None,
) -> Image.Image:
    x_min = int(round(box.x_min * width))
    y_min = int(round(box.y_min * height))
    x_max = int(round(box.x_max * width))
    y_max = int(round(box.y_max * height))

    if fill_alpha > 0:
        overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        fill_color = (color[0], color[1], color[2], fill_alpha)
        overlay_draw.rectangle([x_min, y_min, x_max, y_max], fill=fill_color)
        canvas = Image.alpha_composite(canvas, overlay)

    draw = ImageDraw.Draw(canvas)
    if outline_color:
        draw.rectangle([x_min, y_min, x_max, y_max], outline=(*outline_color, 255), width=line_width + 2)
    draw.rectangle([x_min, y_min, x_max, y_max], outline=(*color[:3], 255), width=line_width)

    if label:
        font = font or ImageFont.load_default()
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        padding = 4
        label_x = x_min
        label_y = y_min - text_height - padding * 2 - line_width
        if label_y < 0:
            label_y = y_min + line_width + 2
        draw.rectangle(
            [label_x, label_y, label_x + text_width + padding * 2, label_y + text_height + padding * 2],
            fill=(*color[:3], 255),
        )
        draw.text((label_x + padding, label_y + padding), label, fill=(255, 255, 255), font=font)
    return canvas


def _load_label_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _save_viz(
    *,
    image: Image.Image,
    gt_boxes: List[Box],
    pred_boxes: List[Box],
    output_path: str,
    iou_threshold: float = 0.5,
) -> None:
    width, height = image.size
    canvas = image.copy().convert("RGBA")
    label_font = _load_label_font(size=40)
    gt_color = (0, 100, 255, 255)
    correct_color = (13, 245, 96, 255)
    wrong_color = (255, 50, 50, 255)

    pred_is_correct = []
    pred_best_ious: List[float] = []
    matched_gt = set()
    for pred in pred_boxes:
        best_iou = 0.0
        best_gt_idx = -1
        for gt_idx, gt in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            iou = _box_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            pred_is_correct.append(True)
            matched_gt.add(best_gt_idx)
        else:
            pred_is_correct.append(False)
        pred_best_ious.append(best_iou)

    for box in gt_boxes:
        canvas = _draw_box(
            canvas,
            box,
            width,
            height,
            gt_color,
            label="GT",
            line_width=6,
            fill_alpha=12,
            font=label_font,
        )

    for i, box in enumerate(pred_boxes):
        is_correct = pred_is_correct[i] if i < len(pred_is_correct) else False
        color = correct_color if is_correct else wrong_color
        label = f"Pred (TP, IoU={pred_best_ious[i]:.2f})" if is_correct else "Pred (FP)"
        canvas = _draw_box(
            canvas,
            box,
            width,
            height,
            color,
            label=label,
            outline_color=(0, 0, 0),
            line_width=6,
            fill_alpha=18,
            font=label_font,
        )

    if not pred_boxes:
        tag_text = "no predictions"
        draw = ImageDraw.Draw(canvas)
        text_bbox = draw.textbbox((0, 0), tag_text, font=label_font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        padding = 8
        tag_width = text_width + padding * 2
        tag_height = text_height + padding * 2
        tag_x = width - tag_width - 10
        tag_y = 10

        overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle(
            [tag_x, tag_y, tag_x + tag_width, tag_y + tag_height],
            fill=(50, 50, 50, 128),
        )
        canvas = Image.alpha_composite(canvas, overlay)
        draw = ImageDraw.Draw(canvas)
        draw.text((tag_x + padding, tag_y + padding), tag_text, fill=(255, 255, 255), font=label_font)

    canvas.convert("RGB").save(output_path)


def _box_iou(a: Box, b: Box) -> float:
    inter_x_min = max(a.x_min, b.x_min)
    inter_y_min = max(a.y_min, b.y_min)
    inter_x_max = min(a.x_max, b.x_max)
    inter_y_max = min(a.y_max, b.y_max)
    inter_w = max(0.0, inter_x_max - inter_x_min)
    inter_h = max(0.0, inter_y_max - inter_y_min)
    inter = inter_w * inter_h
    if inter == 0.0:
        return 0.0
    area_a = max(0.0, a.x_max - a.x_min) * max(0.0, a.y_max - a.y_min)
    area_b = max(0.0, b.x_max - b.x_min) * max(0.0, b.y_max - b.y_min)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _match_ious(predicted: List[Box], ground_truth: List[Box]) -> np.ndarray:
    n_gt = len(ground_truth)
    n_pred = len(predicted)
    if n_gt == 0 or n_pred == 0:
        return np.array([], dtype=np.float32)
    size = max(n_gt, n_pred)
    iou_matrix = np.zeros((size, size), dtype=np.float32)
    for i, gt in enumerate(ground_truth):
        for j, pred in enumerate(predicted):
            iou_matrix[i, j] = _box_iou(pred, gt)
    cost = 1.0 - iou_matrix
    row_idx, col_idx = np.array(np.unravel_index(np.argsort(cost, axis=None), cost.shape))
    used_rows = set()
    used_cols = set()
    matches = []
    for r, c in zip(row_idx, col_idx):
        if r in used_rows or c in used_cols:
            continue
        used_rows.add(r)
        used_cols.add(c)
        matches.append(iou_matrix[r, c])
        if len(used_rows) == size or len(used_cols) == size:
            break
    return np.array(matches, dtype=np.float32)


def _reward_f1(
    predicted: List[Box],
    ground_truth: List[Box],
    *,
    iou_threshold: float = 0.5,
) -> float:
    if not predicted and not ground_truth:
        return 1.0
    if not predicted or not ground_truth:
        return 0.0
    matches = _match_ious(predicted, ground_truth)
    true_pos = float((matches >= iou_threshold).sum())
    precision = true_pos / len(predicted) if predicted else 0.0
    recall = true_pos / len(ground_truth) if ground_truth else 0.0
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _reward_miou(predicted: List[Box], ground_truth: List[Box]) -> float:
    if not predicted and not ground_truth:
        return 1.0
    if not predicted or not ground_truth:
        return 0.0
    matches = _match_ious(predicted, ground_truth)
    denom = max(len(predicted), len(ground_truth))
    return float(matches.sum()) / float(denom) if denom else 0.0


def _count_tp_fp_fn(
    predicted: List[Box],
    ground_truth: List[Box],
    *,
    iou_threshold: float = 0.5,
) -> tuple[int, int, int]:
    n_pred = len(predicted)
    n_gt = len(ground_truth)
    if n_pred == 0 and n_gt == 0:
        return 0, 0, 0
    if n_pred == 0:
        return 0, 0, n_gt
    if n_gt == 0:
        return 0, n_pred, 0
    matches = _match_ious(predicted, ground_truth)
    true_pos = int((matches >= iou_threshold).sum())
    false_pos = n_pred - true_pos
    false_neg = n_gt - true_pos
    return true_pos, false_pos, false_neg


def _to_percent(value: float, name: str) -> int:
    scaled = float(value)
    if scaled <= 1.0:
        scaled *= 100.0
    scaled = _clamp(scaled, 0.0, 100.0)
    out = int(round(scaled))
    if out < 0 or out > 100:
        raise ValueError(f"{name} must be between 0 and 1, or 0 and 100")
    return out


def _build_model_id(model_id: Optional[str], project: Optional[str], version: Optional[int]) -> str:
    if model_id:
        return model_id.strip().strip("/")
    if not project or version is None:
        raise ValueError("Provide --model-id (project/version) or both --project and --version.")
    return f"{project.strip().strip('/')}/{int(version)}"


def _prediction_to_box(
    prediction: dict,
    *,
    image_width: int,
    image_height: int,
) -> Optional[Box]:
    if {"x", "y", "width", "height"}.issubset(prediction.keys()):
        try:
            x = float(prediction["x"])
            y = float(prediction["y"])
            width = float(prediction["width"])
            height = float(prediction["height"])
        except (TypeError, ValueError):
            return None
        x_min = (x - width / 2.0) / float(image_width)
        y_min = (y - height / 2.0) / float(image_height)
        x_max = (x + width / 2.0) / float(image_width)
        y_max = (y + height / 2.0) / float(image_height)
    elif {"x_min", "y_min", "x_max", "y_max"}.issubset(prediction.keys()):
        try:
            x_min = float(prediction["x_min"])
            y_min = float(prediction["y_min"])
            x_max = float(prediction["x_max"])
            y_max = float(prediction["y_max"])
        except (TypeError, ValueError):
            return None
        if max(abs(x_min), abs(y_min), abs(x_max), abs(y_max)) > 1.5:
            x_min /= float(image_width)
            y_min /= float(image_height)
            x_max /= float(image_width)
            y_max /= float(image_height)
    else:
        return None
    try:
        return _box_from_normalized(x_min, y_min, x_max, y_max)
    except ValueError:
        return None


def _extract_boxes(
    raw_payload: dict,
    *,
    image_width: int,
    image_height: int,
    class_name: Optional[str],
) -> tuple[List[Box], List[dict]]:
    raw_predictions = raw_payload.get("predictions")
    if not isinstance(raw_predictions, list):
        raw_predictions = []

    boxes: List[Box] = []
    normalized_objects: List[dict] = []
    for item in raw_predictions:
        if not isinstance(item, dict):
            continue
        label = (
            item.get("class")
            or item.get("class_name")
            or item.get("label")
            or item.get("name")
            or ""
        )
        if class_name and not _labels_match(str(label), class_name):
            continue
        box = _prediction_to_box(item, image_width=image_width, image_height=image_height)
        if box is None:
            continue
        boxes.append(box)
        normalized_objects.append(
            {
                "x_min": box.x_min,
                "y_min": box.y_min,
                "x_max": box.x_max,
                "y_max": box.y_max,
                "class": str(label),
                "confidence": item.get("confidence"),
            }
        )
    return boxes, normalized_objects


def _call_detect(
    *,
    api_base: str,
    api_key: str,
    model_id: str,
    image: Image.Image,
    confidence: float,
    overlap: float,
    class_name: Optional[str],
    timeout: float,
) -> tuple[List[Box], dict, dict]:
    url = f"{api_base.rstrip('/')}/{model_id.lstrip('/')}"
    confidence_percent = _to_percent(confidence, "confidence")
    overlap_percent = _to_percent(overlap, "overlap")
    params = {
        "api_key": api_key,
        "confidence": confidence_percent,
        "overlap": overlap_percent,
    }

    image_bytes = io.BytesIO()
    image.save(image_bytes, format="JPEG", quality=90)
    raw_bytes = image_bytes.getvalue()

    attempts = [
        {
            "request_kwargs": {
                "data": raw_bytes,
                "headers": {"Content-Type": "application/octet-stream"},
            },
            "name": "raw-octet-stream",
        },
        {
            "request_kwargs": {
                "data": raw_bytes,
                "headers": {"Content-Type": "application/x-www-form-urlencoded"},
            },
            "name": "raw-form-urlencoded",
        },
        {
            "request_kwargs": {
                "files": {"file": ("image.jpg", raw_bytes, "image/jpeg")},
            },
            "name": "multipart-file",
        },
    ]

    errors: List[str] = []
    for attempt in attempts:
        try:
            response = requests.post(
                url,
                params=params,
                timeout=timeout,
                **attempt["request_kwargs"],
            )
        except requests.RequestException as exc:
            errors.append(f"{attempt['name']}: request error ({exc})")
            continue

        if response.status_code >= 400:
            body = response.text.strip()
            if len(body) > 220:
                body = body[:220] + "..."
            errors.append(f"{attempt['name']}: HTTP {response.status_code} ({body})")
            continue

        try:
            raw_payload = response.json()
        except ValueError:
            body = response.text.strip()
            if len(body) > 220:
                body = body[:220] + "..."
            errors.append(f"{attempt['name']}: non-JSON response ({body})")
            continue

        width, height = image.size
        pred_boxes, normalized_objects = _extract_boxes(
            raw_payload,
            image_width=width,
            image_height=height,
            class_name=class_name,
        )
        request_meta = {
            "model_id": model_id,
            "confidence_percent": confidence_percent,
            "overlap_percent": overlap_percent,
            "class_name_filter": class_name,
            "transport": attempt["name"],
        }
        response_meta = {
            "objects": normalized_objects,
            "roboflow": raw_payload,
        }
        return pred_boxes, request_meta, response_meta

    raise RuntimeError("Roboflow inference failed. " + " | ".join(errors))


def _evaluate_model(
    *,
    model_id: str,
    val_json_path: str,
    image_dir: str,
    api_base: str,
    api_key: str,
    confidence: float,
    overlap: float,
    class_name: Optional[str],
    timeout: float,
    sleep: float,
    max_samples: Optional[int],
    max_boxes: Optional[int],
    viz_dir: Optional[str],
    viz_samples: int,
    iou_threshold: float,
    disable_progress: bool,
) -> dict[str, float]:
    total_f1 = 0.0
    total_miou = 0.0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    count = 0
    failures = 0

    if viz_dir:
        os.makedirs(viz_dir, exist_ok=True)
    raw_requests = []
    saved_viz = 0

    samples = _iter_val_samples(
        val_json_path=val_json_path,
        image_dir=image_dir,
        max_boxes=max_boxes,
    )
    for image, gt_boxes, filename in _with_progress(samples, disable=disable_progress):
        if max_samples is not None and count >= max_samples:
            break
        try:
            pred_boxes, raw_request, raw_response = _call_detect(
                api_base=api_base,
                api_key=api_key,
                model_id=model_id,
                image=image,
                confidence=confidence,
                overlap=overlap,
                class_name=class_name,
                timeout=timeout,
            )
        except Exception as exc:
            failures += 1
            print(f"roboflow: detect failed ({exc}). skipping sample.")
            continue
        total_f1 += _reward_f1(pred_boxes, gt_boxes, iou_threshold=iou_threshold)
        total_miou += _reward_miou(pred_boxes, gt_boxes)
        tp, fp, fn = _count_tp_fp_fn(pred_boxes, gt_boxes, iou_threshold=iou_threshold)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        count += 1
        raw_requests.append(
            {
                "filename": filename,
                "label": "roboflow",
                "model": model_id,
                "iou_threshold": iou_threshold,
                "request": raw_request,
                "response": raw_response,
            }
        )
        if viz_dir and saved_viz < viz_samples:
            base_name = os.path.splitext(os.path.basename(filename))[0]
            out_name = f"{base_name}_roboflow.png"
            out_path = os.path.join(viz_dir, out_name)
            _save_viz(
                image=image,
                gt_boxes=gt_boxes,
                pred_boxes=pred_boxes,
                output_path=out_path,
                iou_threshold=iou_threshold,
            )
            saved_viz += 1
        if sleep:
            time.sleep(sleep)

    if viz_dir and raw_requests:
        raw_out_path = os.path.join(viz_dir, "roboflow_requests.json")
        with open(raw_out_path, "w", encoding="utf-8") as handle:
            json.dump(raw_requests, handle, indent=2)

    if count == 0:
        return {
            "eval_f1": 0.0,
            "eval_miou": 0.0,
            "eval_true_pos": 0,
            "eval_false_pos": 0,
            "eval_false_neg": 0,
            "eval_f1_macro": 0.0,
            "eval_failures": failures,
            "eval_samples": 0,
        }
    micro_denom = 2 * total_tp + total_fp + total_fn
    micro_f1 = 1.0 if micro_denom == 0 else (2 * total_tp) / micro_denom
    return {
        "eval_f1": micro_f1,
        "eval_f1_macro": total_f1 / count,
        "eval_miou": total_miou / count,
        "eval_true_pos": total_tp,
        "eval_false_pos": total_fp,
        "eval_false_neg": total_fn,
        "eval_failures": failures,
        "eval_samples": count,
    }


def _print_metrics(label: str, metrics: dict[str, float]) -> None:
    print(
        f"{label} f1={metrics['eval_f1']:.3f} "
        f"macro_f1={metrics['eval_f1_macro']:.3f} "
        f"miou={metrics['eval_miou']:.3f} "
        f"tp={metrics['eval_true_pos']} "
        f"fp={metrics['eval_false_pos']} "
        f"fn={metrics['eval_false_neg']} "
        f"samples={metrics['eval_samples']} "
        f"failures={metrics['eval_failures']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark State Farm val set with Roboflow hosted inference.")
    parser.add_argument("--env-file", "--env", default=".env")
    parser.add_argument("--api-key", default=os.environ.get("ROBOFLOW_API_KEY"))
    parser.add_argument("--api-base", default=os.environ.get("ROBOFLOW_BASE_URL", DEFAULT_ROBOFLOW_BASE_URL))
    parser.add_argument("--model-id", default=os.environ.get("ROBOFLOW_MODEL_ID"))
    parser.add_argument("--project", default=os.environ.get("ROBOFLOW_PROJECT"))
    parser.add_argument("--version", type=int, default=None)
    parser.add_argument("--val-json", default="val/post_train_benchmark/post_train_benchmark.json")
    parser.add_argument("--val-image-dir", default="val/post_train_benchmark")
    parser.add_argument("--class-name", default=os.environ.get("ROBOFLOW_CLASS_NAME"))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-boxes", type=int, default=None)
    parser.add_argument("--confidence", type=float, default=0.3, help="0..1 or 0..100")
    parser.add_argument("--overlap", type=float, default=0.5, help="0..1 or 0..100")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--iou-threshold", type=float, default=0.4)
    parser.add_argument("--viz-dir", default="outputs/val_viz/roboflow")
    parser.add_argument("--viz-samples", type=int, default=25)
    parser.add_argument("--metrics-json", default="outputs/val_metrics_roboflow.json")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar.")
    args = parser.parse_args()

    if args.env_file:
        load_dotenv(args.env_file, override=False)
    if not args.api_key:
        args.api_key = os.environ.get("ROBOFLOW_API_KEY")
    if args.version is None and os.environ.get("ROBOFLOW_VERSION"):
        try:
            args.version = int(os.environ["ROBOFLOW_VERSION"])
        except ValueError:
            pass
    if not args.api_key:
        raise ValueError("ROBOFLOW_API_KEY is required")

    model_id = _build_model_id(args.model_id, args.project, args.version)
    metrics = _evaluate_model(
        model_id=model_id,
        val_json_path=args.val_json,
        image_dir=args.val_image_dir,
        api_base=args.api_base,
        api_key=args.api_key,
        confidence=args.confidence,
        overlap=args.overlap,
        class_name=args.class_name,
        timeout=args.timeout,
        sleep=args.sleep,
        max_samples=args.max_samples,
        max_boxes=args.max_boxes,
        viz_dir=args.viz_dir,
        viz_samples=args.viz_samples,
        iou_threshold=args.iou_threshold,
        disable_progress=args.no_progress,
    )
    _print_metrics("roboflow", metrics)

    if args.metrics_json:
        output_payload = {
            "roboflow": metrics,
            "config": {
                "model_id": model_id,
                "api_base": args.api_base,
                "val_json": args.val_json,
                "val_image_dir": args.val_image_dir,
                "class_name": args.class_name,
                "iou_threshold": args.iou_threshold,
                "max_samples": args.max_samples,
                "max_boxes": args.max_boxes,
                "confidence": args.confidence,
                "overlap": args.overlap,
                "viz_dir": args.viz_dir,
            },
        }
        os.makedirs(os.path.dirname(args.metrics_json) or ".", exist_ok=True)
        with open(args.metrics_json, "w", encoding="utf-8") as handle:
            json.dump(output_payload, handle, indent=2)


if __name__ == "__main__":
    main()
