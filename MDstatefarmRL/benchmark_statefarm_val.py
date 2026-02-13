"""Benchmark State Farm detection on local val set.

Requires:
  pip install pillow numpy python-dotenv
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv


OBJECT_NAME = "statefarm logo"
DEFAULT_MODEL = "moondream3-preview"


@dataclass(frozen=True)
class Box:
    x_min: float
    y_min: float
    x_max: float
    y_max: float


def _to_data_url(
    image: Image.Image,
    *,
    format: str = "JPEG",
    quality: int = 90,
) -> str:
    buf = io.BytesIO()
    image.save(buf, format=format, quality=quality)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _build_auth_headers(api_key: str) -> dict[str, str]:
    header_name = os.environ.get("MOONDREAM_AUTH_HEADER", "X-Moondream-Auth")
    user_agent = os.environ.get("MOONDREAM_USER_AGENT")
    key = api_key.strip()
    if header_name.lower() == "authorization" and not key.lower().startswith("bearer "):
        key = f"Bearer {key}"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        header_name: key,
    }
    if user_agent:
        headers["User-Agent"] = user_agent
    else:
        # Default away from urllib's Python UA; Cloudflare may block it.
        headers["User-Agent"] = (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    return headers


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


def _draw_box(
    canvas: Image.Image,
    box: Box,
    width: int,
    height: int,
    color: tuple[int, int, int, int],
    label: Optional[str] = None,
    outline_color: Optional[tuple[int, int, int]] = None,
    line_width: int = 30,
    fill_alpha: int = 18,
    font: Optional[ImageFont.ImageFont] = None,
) -> Image.Image:
    x_min = int(round(box.x_min * width))
    y_min = int(round(box.y_min * height))
    x_max = int(round(box.x_max * width))
    y_max = int(round(box.y_max * height))

    # Draw light fill on an overlay to preserve transparency
    if fill_alpha > 0:
        overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        fill_color = (color[0], color[1], color[2], fill_alpha)
        overlay_draw.rectangle([x_min, y_min, x_max, y_max], fill=fill_color)
        canvas = Image.alpha_composite(canvas, overlay)

    draw = ImageDraw.Draw(canvas)
    # Draw black outline first (if specified), then colored outline on top
    if outline_color:
        draw.rectangle([x_min, y_min, x_max, y_max], outline=(*outline_color, 255), width=line_width + 2)
    draw.rectangle([x_min, y_min, x_max, y_max], outline=(*color[:3], 255), width=line_width)
    
    if label:
        # Draw label background and text above the box
        font = font or ImageFont.load_default()
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        padding = 4
        label_x = x_min
        label_y = y_min - text_height - padding * 2 - line_width
        if label_y < 0:
            label_y = y_min + line_width + 2  # Put label inside box if no room above
        # Draw background rectangle
        label_bg_color = (*color[:3], 255)
        draw.rectangle(
            [label_x, label_y, label_x + text_width + padding * 2, label_y + text_height + padding * 2],
            fill=label_bg_color,
        )
        # Draw text
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
    label_font = _load_label_font(size=96)
    # Colors
    gt_color = (0, 100, 255, 255)  # Blue for GT
    correct_color = (13, 245, 96, 255)  # Green for correct predictions
    wrong_color = (255, 50, 50, 255)  # Red for wrong predictions
    
    # Determine which predictions are correct (match a GT box with IoU >= threshold)
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
    
    # Draw GT boxes first (blue)
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
    
    # Draw prediction boxes (green if correct, red if wrong, with black outline)
    for i, box in enumerate(pred_boxes):
        is_correct = pred_is_correct[i] if i < len(pred_is_correct) else False
        color = correct_color if is_correct else wrong_color
        if is_correct:
            best_iou = pred_best_ious[i] if i < len(pred_best_ious) else 0.0
            label = f"Pred (TP, IoU={best_iou:.2f})"
        else:
            label = "Pred (FP)"
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
    
    # Add "no predictions" tag if there are no predictions
    if not pred_boxes:
        tag_text = "no predictions"
        font = label_font
        draw = ImageDraw.Draw(canvas)
        text_bbox = draw.textbbox((0, 0), tag_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        padding = 8
        tag_width = text_width + padding * 2
        tag_height = text_height + padding * 2
        tag_x = width - tag_width - 10  # 10px margin from right edge
        tag_y = 10  # 10px margin from top
        
        # Create overlay for semi-transparent background
        overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        # Draw 50% transparent gray/black background
        overlay_draw.rectangle(
            [tag_x, tag_y, tag_x + tag_width, tag_y + tag_height],
            fill=(50, 50, 50, 128),  # Gray with 50% transparency
        )
        # Composite the overlay onto the canvas
        canvas = Image.alpha_composite(canvas, overlay)
        # Draw text on top
        draw = ImageDraw.Draw(canvas)
        draw.text((tag_x + padding, tag_y + padding), tag_text, fill=(255, 255, 255), font=font)
    
    # Convert back to RGB for saving
    canvas = canvas.convert("RGB")
    canvas.save(output_path)


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


def _extract_boxes(payload: dict) -> List[Box]:
    raw_boxes = payload.get("objects")
    if raw_boxes is None and isinstance(payload.get("output"), dict):
        raw_boxes = payload["output"].get("objects")
    if raw_boxes is None:
        return []
    boxes: List[Box] = []
    for item in raw_boxes or []:
        try:
            boxes.append(_box_from_normalized(item["x_min"], item["y_min"], item["x_max"], item["y_max"]))
        except (KeyError, TypeError, ValueError):
            continue
    return boxes


def _call_detect(
    *,
    api_base: str,
    api_key: str,
    model: str,
    image: Image.Image,
    object_name: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_objects: int,
    timeout: float,
) -> tuple[List[Box], dict, dict]:
    url = api_base.rstrip("/") + "/detect"
    payload = {
        "model": model,
        "object": object_name,
        "image_url": _to_data_url(image),
        "settings": {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "max_objects": max_objects,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=_build_auth_headers(api_key), method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8") if exc.fp else ""
        detail = error_body.strip() or exc.reason
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc
    parsed = json.loads(body) if body else {}
    return _extract_boxes(parsed), payload, parsed


def _evaluate_model(
    *,
    label: str,
    model: str,
    val_json_path: str,
    image_dir: str,
    api_base: str,
    api_key: str,
    object_name: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_objects: int,
    timeout: float,
    sleep: float,
    max_samples: Optional[int],
    max_boxes: Optional[int],
    viz_dir: Optional[str],
    viz_samples: int,
    iou_threshold: float,
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

    for image, gt_boxes, filename in _iter_val_samples(
        val_json_path=val_json_path,
        image_dir=image_dir,
        max_boxes=max_boxes,
    ):
        if max_samples is not None and count >= max_samples:
            break
        try:
            pred_boxes, raw_request, raw_response = _call_detect(
                api_base=api_base,
                api_key=api_key,
                model=model,
                image=image,
                object_name=object_name,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                max_objects=max_objects,
                timeout=timeout,
            )
        except Exception as exc:
            failures += 1
            print(f"{label}: detect failed ({exc}). skipping sample.")
            continue
        total_f1 += _reward_f1(pred_boxes, gt_boxes, iou_threshold=iou_threshold)
        total_miou += _reward_miou(pred_boxes, gt_boxes)
        tp, fp, fn = _count_tp_fp_fn(pred_boxes, gt_boxes, iou_threshold=iou_threshold)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        count += 1
        sanitized_request = dict(raw_request)
        sanitized_request.pop("image_url", None)
        raw_requests.append(
            {
                "filename": filename,
                "label": label,
                "model": model,
                "iou_threshold": iou_threshold,
                "request": sanitized_request,
                "response": raw_response,
            }
        )
        if viz_dir and saved_viz < viz_samples:
            base_name = os.path.splitext(os.path.basename(filename))[0]
            out_name = f"{base_name}_{label}.png"
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
        raw_out_path = os.path.join(viz_dir, f"{label}_requests.json")
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
    parser = argparse.ArgumentParser(description="Benchmark State Farm val set.")
    parser.add_argument("--env-file", "--env", default=".env")
    parser.add_argument("--api-key", default=os.environ.get("MOONDREAM_API_KEY"))
    parser.add_argument("--api-base", default=os.environ.get("MOONDREAM_BASE_URL", "https://api.moondream.ai/v1"))
    parser.add_argument("--val-json", default="val/post_train_benchmark/post_train_benchmark.json")
    parser.add_argument("--val-image-dir", default="val/post_train_benchmark/post_train_benchmark")
    parser.add_argument("--object-name", default=OBJECT_NAME)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-boxes", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-objects", type=int, default=50)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--iou-threshold", type=float, default=0.4)
    parser.add_argument("--baseline-model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline evaluation and only run checkpoint (if provided).",
    )
    parser.add_argument("--finetune-id", default=None)
    parser.add_argument("--checkpoint-step", type=int, default=None)
    parser.add_argument("--viz-dir", default="outputs/val_viz/new")
    parser.add_argument("--viz-samples", type=int, default=25)
    parser.add_argument("--metrics-json", default="outputs/val_metrics.json")

    args = parser.parse_args()

    if args.env_file:
        load_dotenv(args.env_file, override=False)
    if not args.api_key:
        args.api_key = os.environ.get("MOONDREAM_API_KEY")
    if not args.api_key:
        raise ValueError("MOONDREAM_API_KEY is required")

    baseline_metrics = None
    if not args.skip_baseline:
        baseline_metrics = _evaluate_model(
            label="baseline",
            model=args.baseline_model,
            val_json_path=args.val_json,
            image_dir=args.val_image_dir,
            api_base=args.api_base,
            api_key=args.api_key,
            object_name=args.object_name,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            max_objects=args.max_objects,
            timeout=args.timeout,
            sleep=args.sleep,
            max_samples=args.max_samples,
            max_boxes=args.max_boxes,
            viz_dir=os.path.join(args.viz_dir, "baseline") if args.viz_dir else None,
            viz_samples=args.viz_samples,
            iou_threshold=args.iou_threshold,
        )
        _print_metrics("baseline", baseline_metrics)

    checkpoint_viz_dir = None
    if args.finetune_id and args.checkpoint_step is not None:
        model = f"{DEFAULT_MODEL}/{args.finetune_id}@{args.checkpoint_step}"
        if args.viz_dir:
            checkpoint_viz_dir = os.path.join(args.viz_dir, "checkpoint", args.finetune_id)
        tuned_metrics = _evaluate_model(
            label="checkpoint",
            model=model,
            val_json_path=args.val_json,
            image_dir=args.val_image_dir,
            api_base=args.api_base,
            api_key=args.api_key,
            object_name=args.object_name,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            max_objects=args.max_objects,
            timeout=args.timeout,
            sleep=args.sleep,
            max_samples=args.max_samples,
            max_boxes=args.max_boxes,
            viz_dir=checkpoint_viz_dir,
            viz_samples=args.viz_samples,
            iou_threshold=args.iou_threshold,
        )
        _print_metrics("checkpoint", tuned_metrics)
    elif args.finetune_id or args.checkpoint_step is not None:
        print("To run checkpoint eval, provide both --finetune-id and --checkpoint-step.")

    if args.metrics_json:
        output_payload = {
            "baseline": baseline_metrics,
            "checkpoint": tuned_metrics if args.finetune_id and args.checkpoint_step is not None else None,
            "config": {
                "object_name": args.object_name,
                "iou_threshold": args.iou_threshold,
                "max_samples": args.max_samples,
                "max_boxes": args.max_boxes,
                "baseline_model": args.baseline_model,
                "skip_baseline": args.skip_baseline,
                "finetune_id": args.finetune_id,
                "checkpoint_step": args.checkpoint_step,
                "viz_dir": args.viz_dir,
                "checkpoint_viz_dir": checkpoint_viz_dir,
            },
        }
        os.makedirs(os.path.dirname(args.metrics_json) or ".", exist_ok=True)
        with open(args.metrics_json, "w", encoding="utf-8") as handle:
            json.dump(output_payload, handle, indent=2)


if __name__ == "__main__":
    main()
