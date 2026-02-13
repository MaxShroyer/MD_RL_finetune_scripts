"""Benchmark ball-holder detection on an unseen split.

Supports Hub or local datasets and optional visualization overlays.

Requires:
  pip install datasets pillow numpy scipy python-dotenv
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
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment

from tuna_sdk import DetectRequest, DetectSettings, TunaClient
from tuna_sdk.errors import TunaAPIError, TunaNetworkError


DEFAULT_DATASET = "maxs-m87/Ball-Holder-splits-v1"
DEFAULT_OBJECT_NAME = "Player with ball in hand"
DEFAULT_MODEL = "moondream3-preview"


@dataclass(frozen=True)
class Box:
    x_min: float
    y_min: float
    x_max: float
    y_max: float


def _repo_relative(*parts: str) -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, *parts)


def _to_data_url(image: Image.Image, *, format: str = "JPEG", quality: int = 90) -> str:
    buf = io.BytesIO()
    fmt = format.upper()
    image.save(buf, format=fmt, quality=quality)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = "image/jpeg" if fmt == "JPEG" else "image/png"
    return f"data:{mime};base64,{encoded}"


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
        "User-Agent": user_agent
        or (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        ),
    }
    return headers


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def _box_from_normalized(x_min: float, y_min: float, x_max: float, y_max: float) -> Box:
    x_min = _clamp(float(x_min), 0.0, 1.0)
    y_min = _clamp(float(y_min), 0.0, 1.0)
    x_max = _clamp(float(x_max), 0.0, 1.0)
    y_max = _clamp(float(y_max), 0.0, 1.0)
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("Invalid normalized bbox")
    return Box(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)


def _bbox_xyxy_to_box(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    width: int,
    height: int,
) -> Box:
    x_min = _clamp(float(x_min), 0.0, float(width))
    y_min = _clamp(float(y_min), 0.0, float(height))
    x_max = _clamp(float(x_max), 0.0, float(width))
    y_max = _clamp(float(y_max), 0.0, float(height))
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("Invalid pixel bbox")
    return Box(
        x_min=x_min / width,
        y_min=y_min / height,
        x_max=x_max / width,
        y_max=y_max / height,
    )


def _parse_boxes_value(value: object, width: int, height: int) -> tuple[List[Box], bool]:
    if value is None:
        return [], False
    raw = value
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return [], False
        try:
            raw = json.loads(text)
        except json.JSONDecodeError:
            return [], True
    if not raw:
        return [], False
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return [], False

    boxes: List[Box] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        if all(key in item for key in ("x_min", "y_min", "x_max", "y_max")):
            try:
                x_min = float(item["x_min"])
                y_min = float(item["y_min"])
                x_max = float(item["x_max"])
                y_max = float(item["y_max"])
            except (TypeError, ValueError):
                continue
            try:
                if max(abs(x_min), abs(y_min), abs(x_max), abs(y_max)) > 1.5:
                    boxes.append(_bbox_xyxy_to_box(x_min, y_min, x_max, y_max, width, height))
                else:
                    boxes.append(_box_from_normalized(x_min, y_min, x_max, y_max))
            except ValueError:
                continue
            continue

        box = item.get("box")
        if isinstance(box, dict) and all(k in box for k in ("x_center", "y_center", "width", "height")):
            try:
                x_center = float(box["x_center"])
                y_center = float(box["y_center"])
                box_w = float(box["width"])
                box_h = float(box["height"])
            except (TypeError, ValueError):
                continue
            try:
                boxes.append(
                    _box_from_normalized(
                        x_center - box_w / 2.0,
                        y_center - box_h / 2.0,
                        x_center + box_w / 2.0,
                        y_center + box_h / 2.0,
                    )
                )
            except ValueError:
                continue
    return boxes, False


def _parse_gt_boxes(
    row: dict,
    *,
    width: int,
    height: int,
    annotation_field: str,
    fallback_field: Optional[str],
) -> List[Box]:
    boxes, malformed = _parse_boxes_value(row.get(annotation_field), width, height)
    if boxes or malformed:
        return boxes
    if fallback_field:
        fallback_boxes, _ = _parse_boxes_value(row.get(fallback_field), width, height)
        return fallback_boxes
    return []


def _iter_samples(
    *,
    dataset_name: str,
    dataset_path: Optional[str],
    split: str,
    token: Optional[str],
    annotation_field: str,
    fallback_field: Optional[str],
    max_samples: Optional[int],
) -> Iterable[tuple[Image.Image, List[Box], str]]:
    count = 0
    if dataset_path:
        dataset_obj = load_from_disk(dataset_path)
        if isinstance(dataset_obj, DatasetDict):
            if split not in dataset_obj:
                available = ", ".join(dataset_obj.keys())
                raise ValueError(f"Split '{split}' not found. Available: {available}")
            ds: Dataset = dataset_obj[split]
        else:
            ds = dataset_obj
        for idx, row in enumerate(ds):
            if max_samples is not None and count >= max_samples:
                break
            image = row["image"].convert("RGB")
            width, height = image.size
            boxes = _parse_gt_boxes(
                row,
                width=width,
                height=height,
                annotation_field=annotation_field,
                fallback_field=fallback_field,
            )
            count += 1
            sample_id = str(row.get("id") or row.get("image_id") or idx)
            yield image, boxes, sample_id
        return

    ds = load_dataset(dataset_name, split=split, streaming=True, token=token)
    for idx, row in enumerate(ds):
        if max_samples is not None and count >= max_samples:
            break
        image = row["image"].convert("RGB")
        width, height = image.size
        boxes = _parse_gt_boxes(
            row,
            width=width,
            height=height,
            annotation_field=annotation_field,
            fallback_field=fallback_field,
        )
        count += 1
        sample_id = str(row.get("id") or row.get("image_id") or idx)
        yield image, boxes, sample_id


def _extract_pred_boxes(payload: dict) -> List[Box]:
    raw_boxes = payload.get("objects")
    if raw_boxes is None and isinstance(payload.get("output"), dict):
        raw_boxes = payload["output"].get("objects")
    boxes: List[Box] = []
    for item in raw_boxes or []:
        if not isinstance(item, dict):
            continue
        try:
            boxes.append(_box_from_normalized(item["x_min"], item["y_min"], item["x_max"], item["y_max"]))
        except (KeyError, TypeError, ValueError):
            continue
    return boxes


def _call_detect_api(
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
) -> List[Box]:
    url = api_base.rstrip("/") + "/detect"
    payload = {
        "model": model,
        "object": object_name,
        "image_url": _to_data_url(image, format="JPEG", quality=90),
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
        detail = error_body.strip() or str(exc.reason)
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc
    parsed = json.loads(body) if body else {}
    return _extract_pred_boxes(parsed)


def _call_tuning_rollouts(
    *,
    finetune,
    image: Image.Image,
    object_name: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_objects: int,
) -> List[Box]:
    req = DetectRequest(
        object_name=object_name,
        image_url=_to_data_url(image, format="JPEG", quality=90),
        settings=DetectSettings(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            max_objects=max_objects,
        ),
    )
    result = finetune.rollouts(num_rollouts=1, request=req)
    if not result.rollouts:
        return []
    objects = getattr(result.rollouts[0].output, "objects", None) or []
    return [Box(x_min=o.x_min, y_min=o.y_min, x_max=o.x_max, y_max=o.y_max) for o in objects]


def _call_tuning_rollouts_batch(
    *,
    finetune,
    images: List[Image.Image],
    object_name: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_objects: int,
    max_workers: int,
) -> List[List[Box]]:
    requests = [
        DetectRequest(
            object_name=object_name,
            image_url=_to_data_url(image, format="JPEG", quality=90),
            settings=DetectSettings(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                max_objects=max_objects,
            ),
        )
        for image in images
    ]
    results = finetune.rollouts_batch(requests=requests, num_rollouts=1, max_workers=max_workers)
    out: List[List[Box]] = []
    for result in results:
        if not result.rollouts:
            out.append([])
            continue
        objects = getattr(result.rollouts[0].output, "objects", None) or []
        out.append([Box(x_min=o.x_min, y_min=o.y_min, x_max=o.x_max, y_max=o.y_max) for o in objects])
    return out


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
    row_idx, col_idx = linear_sum_assignment(cost)
    return iou_matrix[row_idx, col_idx]


def _reward_miou(predicted: List[Box], ground_truth: List[Box]) -> float:
    if not predicted and not ground_truth:
        return 1.0
    if not predicted or not ground_truth:
        return 0.0
    matches = _match_ious(predicted, ground_truth)
    denom = max(len(predicted), len(ground_truth))
    return float(matches.sum()) / float(denom) if denom else 0.0


def _reward_f1(predicted: List[Box], ground_truth: List[Box], *, iou_threshold: float = 0.5) -> float:
    n_pred = len(predicted)
    n_gt = len(ground_truth)
    if n_pred == 0 and n_gt == 0:
        return 1.0
    if n_pred == 0 or n_gt == 0:
        return 0.0
    matches = _match_ious(predicted, ground_truth)
    true_pos = float((matches >= iou_threshold).sum())
    precision = true_pos / n_pred if n_pred else 0.0
    recall = true_pos / n_gt if n_gt else 0.0
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


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


def _load_label_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _draw_box(
    canvas: Image.Image,
    box: Box,
    *,
    color: tuple[int, int, int],
    label: str,
    line_width: int,
    font: ImageFont.ImageFont,
) -> None:
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    x_min = int(round(box.x_min * width))
    y_min = int(round(box.y_min * height))
    x_max = int(round(box.x_max * width))
    y_max = int(round(box.y_max * height))
    draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=line_width)
    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    pad = 4
    bg_x0 = x_min
    bg_y0 = max(0, y_min - text_h - 2 * pad)
    bg_x1 = min(width, bg_x0 + text_w + 2 * pad)
    bg_y1 = min(height, bg_y0 + text_h + 2 * pad)
    draw.rectangle([bg_x0, bg_y0, bg_x1, bg_y1], fill=color)
    draw.text((bg_x0 + pad, bg_y0 + pad), label, fill=(255, 255, 255), font=font)


def _save_viz(
    *,
    image: Image.Image,
    gt_boxes: List[Box],
    pred_boxes: List[Box],
    iou_threshold: float,
    output_path: str,
) -> None:
    canvas = image.copy()
    font = _load_label_font(size=max(12, int(round(max(image.size) * 0.03))))
    line_width = max(2, int(round(max(image.size) * 0.005)))
    for box in gt_boxes:
        _draw_box(canvas, box, color=(52, 152, 219), label="GT", line_width=line_width, font=font)
    for box in pred_boxes:
        max_iou = 0.0
        if gt_boxes:
            max_iou = max(_box_iou(box, gt) for gt in gt_boxes)
        is_match = max_iou >= iou_threshold
        color = (46, 204, 113) if is_match else (231, 76, 60)
        label = f"PRED {max_iou:.2f}"
        _draw_box(canvas, box, color=color, label=label, line_width=line_width, font=font)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    canvas.save(output_path)


def _should_save_viz(*, save_viz: bool, viz_saved: int, viz_limit: Optional[int]) -> bool:
    if not save_viz:
        return False
    if viz_limit is None:
        return True
    return viz_saved < viz_limit


def _print_metrics(label: str, metrics: dict[str, object]) -> None:
    if not metrics or "eval_f1" not in metrics:
        error = metrics.get("error") if isinstance(metrics, dict) else "unknown error"
        print(f"{label} metrics unavailable: {error}")
        return
    print(
        f"{label} f1={metrics['eval_f1']:.3f} "
        f"macro_f1={metrics['eval_f1_macro']:.3f} "
        f"miou={metrics['eval_miou']:.3f} "
        f"tp={metrics['tp']} "
        f"fp={metrics['fp']} "
        f"fn={metrics['fn']} "
        f"samples={metrics['samples']} "
        f"neg={metrics['negative_samples']} "
        f"pos={metrics['positive_samples']}"
    )


def _evaluate_model(
    *,
    label: str,
    args: argparse.Namespace,
    model: str,
    finetune_id: str,
    checkpoint_step: Optional[int],
    inference_mode: str,
    base_model: str,
    viz_dir: Optional[str],
) -> dict[str, object]:
    dataset_path = args.dataset_path.strip() or None
    model = model.strip()
    finetune_id = finetune_id.strip()

    if checkpoint_step is not None and checkpoint_step < 0:
        raise ValueError("--checkpoint-step must be >= 0")
    if checkpoint_step is not None and not finetune_id:
        raise ValueError("--checkpoint-step requires --finetune-id")

    resolved_mode = inference_mode
    if resolved_mode == "auto":
        if checkpoint_step is not None:
            resolved_mode = "detect_api"
        elif finetune_id and not model:
            resolved_mode = "tuning_interface"
        else:
            resolved_mode = "detect_api"

    if resolved_mode == "detect_api":
        if not model and finetune_id and checkpoint_step is not None:
            model = f"{base_model.rstrip('/')}/{finetune_id}@{checkpoint_step}"
        if not model:
            raise ValueError(
                "detect_api mode requires --model, or both --finetune-id and --checkpoint-step."
            )
    elif resolved_mode == "tuning_interface":
        if not finetune_id:
            raise ValueError("tuning_interface mode requires --finetune-id")
        if checkpoint_step is not None:
            raise ValueError(
                "--checkpoint-step is not supported in tuning_interface mode. "
                "Use --inference-mode detect_api to evaluate a saved checkpoint step."
            )

    client = None
    finetune = None
    if resolved_mode == "tuning_interface":
        client = TunaClient(api_key=args.api_key, base_url=args.api_base)
        finetune = client.get_finetune(finetune_id)

    total = 0
    total_miou = 0.0
    total_f1 = 0.0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    positive_samples = 0
    negative_samples = 0
    latency_sum = 0.0
    viz_saved = 0

    samples_iter = _iter_samples(
        dataset_name=args.dataset_name,
        dataset_path=dataset_path,
        split=args.split,
        token=args.hf_token,
        annotation_field=args.annotation_field,
        fallback_field=args.fallback_field,
        max_samples=args.max_samples,
    )

    save_viz = args.save_viz and bool(viz_dir)

    try:
        if finetune is None:
            for image, gt_boxes, sample_id in samples_iter:
                start = time.monotonic()
                try:
                    pred_boxes = _call_detect_api(
                        api_base=args.api_base,
                        api_key=args.api_key,
                        model=model,
                        image=image,
                        object_name=args.object_name,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_tokens=args.max_tokens,
                        max_objects=args.max_objects,
                        timeout=args.timeout,
                    )
                except Exception as exc:
                    print(f"{label}: detect failed for sample {sample_id}: {exc}. skipping sample.")
                    continue
                latency = time.monotonic() - start

                total += 1
                latency_sum += latency
                if gt_boxes:
                    positive_samples += 1
                else:
                    negative_samples += 1

                miou = _reward_miou(pred_boxes, gt_boxes)
                f1 = _reward_f1(pred_boxes, gt_boxes, iou_threshold=args.iou_threshold)
                tp, fp, fn = _count_tp_fp_fn(pred_boxes, gt_boxes, iou_threshold=args.iou_threshold)
                total_miou += miou
                total_f1 += f1
                total_tp += tp
                total_fp += fp
                total_fn += fn

                if _should_save_viz(save_viz=save_viz, viz_saved=viz_saved, viz_limit=args.viz_limit):
                    out_path = os.path.join(viz_dir, f"{sample_id}.png")
                    _save_viz(
                        image=image,
                        gt_boxes=gt_boxes,
                        pred_boxes=pred_boxes,
                        iou_threshold=args.iou_threshold,
                        output_path=out_path,
                    )
                    viz_saved += 1

                if args.progress_every > 0 and total % args.progress_every == 0:
                    max_part = f"/{args.max_samples}" if args.max_samples is not None else ""
                    print(f"{label}: progress {total}{max_part} samples")
        else:
            # Tuning-interface inference (RL finetune state): batch requests for speed.
            batch_images: List[Image.Image] = []
            batch_gt: List[List[Box]] = []
            batch_ids: List[str] = []
            for image, gt_boxes, sample_id in samples_iter:
                batch_images.append(image)
                batch_gt.append(gt_boxes)
                batch_ids.append(sample_id)
                if len(batch_images) < max(1, int(args.batch_size)):
                    continue

                start = time.monotonic()
                try:
                    batch_pred = _call_tuning_rollouts_batch(
                        finetune=finetune,
                        images=batch_images,
                        object_name=args.object_name,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_tokens=args.max_tokens,
                        max_objects=args.max_objects,
                        max_workers=max(1, min(int(args.max_workers), len(batch_images))),
                    )
                except (TunaAPIError, TunaNetworkError, RuntimeError) as exc:
                    print(f"{label}: tuning rollouts_batch failed: {exc}. skipping batch.")
                    batch_images, batch_gt, batch_ids = [], [], []
                    continue
                latency = time.monotonic() - start

                per_sample_latency = latency / max(1, len(batch_images))
                for pred_boxes, gt_boxes_item, sample_id_item, image_item in zip(
                    batch_pred, batch_gt, batch_ids, batch_images
                ):
                    total += 1
                    latency_sum += per_sample_latency
                    if gt_boxes_item:
                        positive_samples += 1
                    else:
                        negative_samples += 1

                    miou = _reward_miou(pred_boxes, gt_boxes_item)
                    f1 = _reward_f1(pred_boxes, gt_boxes_item, iou_threshold=args.iou_threshold)
                    tp, fp, fn = _count_tp_fp_fn(pred_boxes, gt_boxes_item, iou_threshold=args.iou_threshold)
                    total_miou += miou
                    total_f1 += f1
                    total_tp += tp
                    total_fp += fp
                    total_fn += fn

                    if _should_save_viz(
                        save_viz=save_viz, viz_saved=viz_saved, viz_limit=args.viz_limit
                    ):
                        out_path = os.path.join(viz_dir, f"{sample_id_item}.png")
                        _save_viz(
                            image=image_item,
                            gt_boxes=gt_boxes_item,
                            pred_boxes=pred_boxes,
                            iou_threshold=args.iou_threshold,
                            output_path=out_path,
                        )
                        viz_saved += 1

                    if args.progress_every > 0 and total % args.progress_every == 0:
                        max_part = f"/{args.max_samples}" if args.max_samples is not None else ""
                        print(f"{label}: progress {total}{max_part} samples")

                batch_images, batch_gt, batch_ids = [], [], []

            if batch_images:
                start = time.monotonic()
                try:
                    batch_pred = _call_tuning_rollouts_batch(
                        finetune=finetune,
                        images=batch_images,
                        object_name=args.object_name,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_tokens=args.max_tokens,
                        max_objects=args.max_objects,
                        max_workers=max(1, min(int(args.max_workers), len(batch_images))),
                    )
                except (TunaAPIError, TunaNetworkError, RuntimeError) as exc:
                    print(f"{label}: tuning rollouts_batch failed: {exc}. skipping final batch.")
                    batch_pred = []
                latency = time.monotonic() - start
                per_sample_latency = latency / max(1, len(batch_images))
                for pred_boxes, gt_boxes_item, sample_id_item, image_item in zip(
                    batch_pred, batch_gt, batch_ids, batch_images
                ):
                    total += 1
                    latency_sum += per_sample_latency
                    if gt_boxes_item:
                        positive_samples += 1
                    else:
                        negative_samples += 1

                    miou = _reward_miou(pred_boxes, gt_boxes_item)
                    f1 = _reward_f1(pred_boxes, gt_boxes_item, iou_threshold=args.iou_threshold)
                    tp, fp, fn = _count_tp_fp_fn(pred_boxes, gt_boxes_item, iou_threshold=args.iou_threshold)
                    total_miou += miou
                    total_f1 += f1
                    total_tp += tp
                    total_fp += fp
                    total_fn += fn

                    if _should_save_viz(
                        save_viz=save_viz, viz_saved=viz_saved, viz_limit=args.viz_limit
                    ):
                        out_path = os.path.join(viz_dir, f"{sample_id_item}.png")
                        _save_viz(
                            image=image_item,
                            gt_boxes=gt_boxes_item,
                            pred_boxes=pred_boxes,
                            iou_threshold=args.iou_threshold,
                            output_path=out_path,
                        )
                        viz_saved += 1
    finally:
        if client is not None:
            client.close()

    if total == 0:
        return {
            "label": label,
            "samples": 0,
            "error": "No samples were evaluated.",
        }

    micro_denom = 2 * total_tp + total_fp + total_fn
    micro_f1 = 1.0 if micro_denom == 0 else (2 * total_tp) / micro_denom

    return {
        "label": label,
        "model": model or None,
        "finetune_id": finetune_id or None,
        "checkpoint_step": checkpoint_step,
        "inference_mode": resolved_mode,
        "dataset_name": args.dataset_name,
        "dataset_path": dataset_path,
        "split": args.split,
        "samples": total,
        "positive_samples": positive_samples,
        "negative_samples": negative_samples,
        "eval_miou": total_miou / total,
        "eval_f1": micro_f1,
        "eval_f1_macro": total_f1 / total,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "avg_latency_sec": latency_sum / total,
        "save_viz": save_viz,
        "viz_limit": args.viz_limit,
        "viz_saved": viz_saved,
        "iou_threshold": args.iou_threshold,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark ball-holder detection on unseen split.")
    parser.add_argument("--api-key", default=os.environ.get("MOONDREAM_API_KEY"))
    parser.add_argument("--env-file", "--env", default=_repo_relative(".env"))
    parser.add_argument("--api-base", default="https://api.moondream.ai/v1")
    parser.add_argument("--baseline-model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline evaluation and only run model/finetune if provided.",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Inference API model name (used with /detect). Required unless --finetune-id is set.",
    )
    parser.add_argument(
        "--finetune-id",
        default="",
        help=(
            "If set, uses the RL tuning interface (/v1/tuning/rollouts) via tuna_sdk to run inference "
            "against the finetune state."
        ),
    )
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        default=None,
        help=(
            "Checkpoint step for finetuned model inference via /detect. "
            "When set with --finetune-id, model is built as moondream3-preview/<finetune_id>@<step>."
        ),
    )
    parser.add_argument(
        "--base-model",
        default="moondream3-preview",
        help="Base model prefix used when constructing model from --finetune-id and --checkpoint-step.",
    )
    parser.add_argument(
        "--inference-mode",
        choices=["auto", "detect_api", "tuning_interface"],
        default="auto",
        help=(
            "auto: infer mode from flags (prefers tuning interface for --finetune-id, "
            "but uses detect_api if --checkpoint-step is provided)."
        ),
    )

    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-path", default="", help="Optional local dataset path from save_to_disk().")
    parser.add_argument("--split", default="post_val")
    parser.add_argument("--annotation-field", default="answer_boxes")
    parser.add_argument("--fallback-field", default="answer")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"))
    parser.add_argument("--max-samples", type=int, default=None)

    parser.add_argument("--object-name", default=DEFAULT_OBJECT_NAME)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-objects", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--iou-threshold", type=float, default=0.4)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for tuning-interface inference.")
    parser.add_argument("--max-workers", type=int, default=16, help="Max workers for tuning-interface inference.")

    parser.add_argument(
        "--save-viz",
        dest="save_viz",
        action="store_true",
        default=True,
        help="Save visualization overlays. Enabled by default.",
    )
    parser.add_argument(
        "--no-save-viz",
        dest="save_viz",
        action="store_false",
        help="Disable visualization overlay saving.",
    )
    parser.add_argument(
        "--viz-limit",
        type=int,
        default=None,
        help="Max number of visualization images to save. Default: unlimited (all evaluated samples).",
    )
    parser.add_argument("--viz-dir", default=_repo_relative("outputs", "benchmark_viz"))
    parser.add_argument("--out-json", default=_repo_relative("outputs", "benchmark_metrics.json"))
    args = parser.parse_args()

    load_dotenv(args.env_file, override=False)
    if not args.api_key:
        args.api_key = os.environ.get("MOONDREAM_API_KEY")
    if not args.hf_token:
        args.hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not args.api_key:
        raise ValueError("MOONDREAM_API_KEY is required")

    has_candidate = bool(args.model.strip() or args.finetune_id.strip())
    if args.skip_baseline and not has_candidate:
        raise ValueError(
            "Provide --model or --finetune-id, or omit --skip-baseline to run baseline."
        )

    baseline_metrics = None
    if not args.skip_baseline:
        baseline_viz_dir = os.path.join(args.viz_dir, "baseline") if args.viz_dir else None
        baseline_metrics = _evaluate_model(
            label="baseline",
            args=args,
            model=args.baseline_model,
            finetune_id="",
            checkpoint_step=None,
            inference_mode="detect_api",
            base_model=args.base_model,
            viz_dir=baseline_viz_dir,
        )
        _print_metrics("baseline", baseline_metrics)

    candidate_metrics = None
    if has_candidate:
        if args.skip_baseline:
            candidate_viz_dir = args.viz_dir
        else:
            candidate_viz_dir = os.path.join(args.viz_dir, "candidate") if args.viz_dir else None
        candidate_metrics = _evaluate_model(
            label="candidate",
            args=args,
            model=args.model,
            finetune_id=args.finetune_id,
            checkpoint_step=args.checkpoint_step,
            inference_mode=args.inference_mode,
            base_model=args.base_model,
            viz_dir=candidate_viz_dir,
        )
        _print_metrics("candidate", candidate_metrics)

    if not args.out_json:
        return

    if baseline_metrics is not None and candidate_metrics is not None:
        payload = {
            "baseline": baseline_metrics,
            "candidate": candidate_metrics,
            "config": {
                "baseline_model": args.baseline_model,
                "skip_baseline": args.skip_baseline,
                "model": args.model.strip() or None,
                "finetune_id": args.finetune_id.strip() or None,
                "checkpoint_step": args.checkpoint_step,
                "inference_mode": args.inference_mode,
                "base_model": args.base_model,
                "dataset_name": args.dataset_name,
                "dataset_path": args.dataset_path.strip() or None,
                "split": args.split,
                "object_name": args.object_name,
                "iou_threshold": args.iou_threshold,
                "max_samples": args.max_samples,
                "viz_dir": args.viz_dir,
            },
        }
    elif baseline_metrics is not None:
        payload = baseline_metrics
    else:
        payload = candidate_metrics or {"error": "No evaluation was run."}

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
