"""Evaluate and visualize a Moondream finetune checkpoint.

Requires:
  pip install datasets pillow numpy scipy
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
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
from datasets import load_dataset, load_from_disk
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment


AMAZON_DATASET = "maxs-m87/Amazon_logo_nba"
OBJECT_NAME = "amazon logo"


@dataclass(frozen=True)
class Box:
    x_min: float
    y_min: float
    x_max: float
    y_max: float


def _load_env_file(path: str) -> None:
    if not path or not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            if key and key not in os.environ:
                os.environ[key] = value


def _build_auth_headers(api_key: str) -> dict[str, str]:
    # Match behavior in train script: allow overriding header name + UA.
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
    # Avoid WAF/edge blocks on urllib's default UA.
    headers["User-Agent"] = user_agent or (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
    return headers


def _fetch_json(url: str, *, api_key: str, timeout: float = 30.0) -> dict:
    req = urllib.request.Request(url, headers=_build_auth_headers(api_key), method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body) if body else {}


def _list_saved_checkpoint_steps(
    *,
    api_key: str,
    finetune_id: str,
    api_base: str = "https://api.moondream.ai/v1",
    timeout: float = 30.0,
    limit: int = 100,
) -> List[int]:
    # Uses the finetuning API (not the inference API):
    # GET /v1/tuning/finetunes/:finetuneId/checkpoints
    url = api_base.rstrip("/") + f"/tuning/finetunes/{finetune_id}/checkpoints?limit={int(limit)}"
    data = _fetch_json(url, api_key=api_key, timeout=timeout)
    checkpoints = data.get("checkpoints") or []
    steps: List[int] = []
    for item in checkpoints:
        if not isinstance(item, dict):
            continue
        step = item.get("step")
        try:
            steps.append(int(step))
        except (TypeError, ValueError):
            continue
    steps.sort()
    return steps


def _to_data_url(image: Image.Image, *, format: str = "JPEG", quality: int = 90) -> str:
    buf = io.BytesIO()
    image.save(buf, format=format, quality=quality)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = "image/jpeg" if format.upper() == "JPEG" else "image/png"
    return f"data:{mime};base64,{encoded}"


def _maybe_downscale_image(image: Image.Image, *, max_edge: Optional[int]) -> Image.Image:
    if not max_edge:
        return image
    max_edge_i = int(max_edge)
    if max_edge_i <= 0:
        return image
    w, h = image.size
    longest = max(w, h)
    if longest <= max_edge_i:
        return image
    scale = max_edge_i / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return image.resize((new_w, new_h), resample=Image.BICUBIC)


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


def _bbox_xyxy_to_box(bbox_xyxy: List[float], width: int, height: int) -> Box:
    if len(bbox_xyxy) != 4:
        raise ValueError(f"Expected bbox length 4, got {len(bbox_xyxy)}")
    x_min, y_min, x_max, y_max = bbox_xyxy
    x_min = _clamp(float(x_min), 0.0, float(width))
    y_min = _clamp(float(y_min), 0.0, float(height))
    x_max = _clamp(float(x_max), 0.0, float(width))
    y_max = _clamp(float(y_max), 0.0, float(height))
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("Invalid bbox after clipping")
    return Box(
        x_min=x_min / width,
        y_min=y_min / height,
        x_max=x_max / width,
        y_max=y_max / height,
    )


def _parse_amazon_boxes(answer_boxes: Optional[str], width: int, height: int) -> List[Box]:
    if not answer_boxes:
        return []
    raw = json.loads(answer_boxes) if isinstance(answer_boxes, str) else answer_boxes
    boxes: List[Box] = []
    for item in raw or []:
        try:
            x_min = float(item["x_min"])
            y_min = float(item["y_min"])
            x_max = float(item["x_max"])
            y_max = float(item["y_max"])
        except (KeyError, TypeError, ValueError):
            continue
        try:
            if max(x_max, y_max) > 1.5:
                boxes.append(_bbox_xyxy_to_box([x_min, y_min, x_max, y_max], width, height))
            else:
                boxes.append(_box_from_normalized(x_min, y_min, x_max, y_max))
        except ValueError:
            continue
    return boxes


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


def _reward_f1(predicted: List[Box], ground_truth: List[Box]) -> float:
    if not predicted and not ground_truth:
        return 1.0
    if not predicted or not ground_truth:
        return 0.0
    matches = _match_ious(predicted, ground_truth)
    true_pos = float((matches >= 0.5).sum())
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
    max_image_edge: Optional[int] = 1536,
    retries: int = 5,
    retry_backoff_s: float = 1.0,
) -> List[Box]:
    url = api_base.rstrip("/") + "/detect"
    # Public docs for /v1/detect require only {image_url, object}.
    # Some deployments accept "model", but "settings" is not documented and can cause server errors.
    image = _maybe_downscale_image(image, max_edge=max_image_edge)
    payload = {
        "object": object_name,
        "image_url": _to_data_url(image, format="JPEG", quality=90),
    }
    if model and model != "moondream3-preview":
        payload["model"] = model
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers=_build_auth_headers(api_key),
        method="POST",
    )
    attempts = max(1, int(retries) + 1)
    last_exc: Optional[Exception] = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8")
            parsed = json.loads(body) if body else {}
            return _extract_boxes(parsed)
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8") if exc.fp else ""
            detail = error_body.strip() or exc.reason
            last_exc = RuntimeError(f"HTTP {exc.code} from {url}: {detail}")
            if exc.code in (500, 502, 503, 504) and attempt < attempts - 1:
                time.sleep(min(30.0, max(0.1, float(retry_backoff_s) * (2**attempt))))
                continue
            raise last_exc from exc
        except urllib.error.URLError as exc:
            last_exc = RuntimeError(f"network error calling {url}: {exc}")
            if attempt < attempts - 1:
                time.sleep(min(30.0, max(0.1, float(retry_backoff_s) * (2**attempt))))
                continue
            raise last_exc from exc
        except Exception as exc:
            last_exc = exc
            break
    if last_exc:
        raise last_exc
    return []


def _iter_samples(
    *,
    dataset_name: str,
    dataset_path: Optional[str],
    split: str,
    token: Optional[str],
    max_boxes: Optional[int],
) -> Iterable[tuple[Image.Image, List[Box], int]]:
    if dataset_path:
        dataset_obj = load_from_disk(dataset_path)
        if hasattr(dataset_obj, "keys"):
            if split not in dataset_obj:
                raise ValueError(f"Split '{split}' not found in local dataset.")
            ds = dataset_obj[split]
        else:
            ds = dataset_obj
    else:
        try:
            ds = load_dataset(dataset_name, split=split, streaming=True, token=token)
        except ValueError as exc:
            alias_map = {"val": "validation", "valid": "validation", "validation": "val"}
            alias = alias_map.get(split)
            if alias:
                print(f"split '{split}' not found; trying alias '{alias}'")
                ds = load_dataset(dataset_name, split=alias, streaming=True, token=token)
            else:
                raise exc
    for idx, row in enumerate(ds):
        image = row["image"].convert("RGB")
        width, height = image.size
        boxes = _parse_amazon_boxes(row.get("answer_boxes"), width, height)
        if max_boxes is not None:
            boxes = boxes[:max_boxes]
        yield image, boxes, idx


def _iter_val_images(
    *,
    val_dir: str,
) -> Iterable[tuple[Image.Image, List[Box], int, str]]:
    paths = []
    root = Path(val_dir)
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        paths.extend(sorted(root.glob(ext)))
    for idx, path in enumerate(paths):
        image = Image.open(path).convert("RGB")
        yield image, [], idx, str(path)


def _draw_boxes(image: Image.Image, boxes: List[Box], color: str) -> Image.Image:
    if not boxes:
        return image
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for box in boxes:
        draw.rectangle(
            [
                box.x_min * width,
                box.y_min * height,
                box.x_max * width,
                box.y_max * height,
            ],
            outline=color,
            width=max(1, int(round(min(width, height) * 0.003))),
        )
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Amazon logo checkpoint.")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--api-key", default=os.environ.get("MOONDREAM_API_KEY"))
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"))
    parser.add_argument("--api-base", default=os.environ.get("MOONDREAM_BASE_URL", "https://api.moondream.ai/v1"))
    parser.add_argument("--dataset-name", default=AMAZON_DATASET)
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--finetune-id", default=None)
    parser.add_argument("--checkpoint-step", type=int, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--max-boxes", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-objects", type=int, default=50)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--viz-samples", type=int, default=25)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--val-dir", default=None)
    parser.add_argument(
        "--max-image-edge",
        type=int,
        default=1536,
        help="Downscale images before sending so max(width,height) <= this (0 disables).",
    )
    parser.add_argument("--retries", type=int, default=5, help="Retries on 5xx / transient network errors.")
    parser.add_argument("--retry-backoff", type=float, default=1.0, help="Exponential backoff base seconds.")
    args = parser.parse_args()

    _load_env_file(args.env_file)
    if not args.api_key:
        args.api_key = os.environ.get("MOONDREAM_API_KEY")
    if not args.hf_token:
        args.hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not args.api_key:
        raise ValueError("MOONDREAM_API_KEY is required")

    if args.model:
        model = args.model
    elif args.finetune_id and args.checkpoint_step is not None:
        model = f"moondream3-preview/{args.finetune_id}@{args.checkpoint_step}"
    else:
        model = "moondream3-preview"

    # Preflight: finetuned inference only works for SAVED checkpoints.
    if args.finetune_id and args.checkpoint_step is not None:
        try:
            saved_steps = _list_saved_checkpoint_steps(
                api_key=args.api_key,
                finetune_id=args.finetune_id,
                api_base=args.api_base,
                timeout=min(30.0, float(args.timeout)),
            )
        except Exception as exc:
            print(f"warning: could not list saved checkpoints for {args.finetune_id}: {exc}")
            saved_steps = []
        if saved_steps and int(args.checkpoint_step) not in set(saved_steps):
            preview = saved_steps[-10:] if len(saved_steps) > 10 else saved_steps
            raise RuntimeError(
                "Checkpoint step is not saved for inference.\n"
                f"Requested step: {args.checkpoint_step}\n"
                f"Saved steps (last {len(preview)}): {preview}\n"
                "Tip: training scripts often save every N steps, which can produce saved steps like 19,39,...,799.\n"
                "Pick a saved step (e.g. 799) or save the checkpoint in Moondream tuning first."
            )
    output_dir = args.output_dir or os.path.join("outputs", f"checkpoint_eval_{args.checkpoint_step}")
    viz_dir = os.path.join(output_dir, "viz")
    os.makedirs(viz_dir, exist_ok=True)

    total_f1 = 0.0
    total_miou = 0.0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    count = 0
    failures = 0

    if args.val_dir:
        iterator = _iter_val_images(val_dir=args.val_dir)
        use_ground_truth = False
    else:
        iterator = _iter_samples(
            dataset_name=args.dataset_name,
            dataset_path=args.dataset_path,
            split=args.split,
            token=args.hf_token,
            max_boxes=args.max_boxes,
        )
        use_ground_truth = True

    def _max_samples_reached(n_done: int) -> bool:
        # Treat max_samples <= 0 as "no limit" for convenience.
        if args.max_samples is None:
            return False
        if int(args.max_samples) <= 0:
            return False
        return n_done >= int(args.max_samples)

    for item in iterator:
        if use_ground_truth:
            image, gt_boxes, idx = item
            source_name = None
        else:
            image, gt_boxes, idx, source_name = item
        if _max_samples_reached(count):
            break
        try:
            pred_boxes = _call_detect(
                api_base=args.api_base,
                api_key=args.api_key,
                model=model,
                image=image,
                object_name=OBJECT_NAME,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                max_objects=args.max_objects,
                timeout=args.timeout,
                max_image_edge=int(args.max_image_edge) if args.max_image_edge else None,
                retries=int(args.retries),
                retry_backoff_s=float(args.retry_backoff),
            )
        except Exception as exc:
            failures += 1
            label = f"idx={idx}"
            if source_name:
                label = f"{source_name}"
            print(f"detect failed for {label}: {exc}")
            continue

        if use_ground_truth:
            total_f1 += _reward_f1(pred_boxes, gt_boxes)
            total_miou += _reward_miou(pred_boxes, gt_boxes)
            tp, fp, fn = _count_tp_fp_fn(pred_boxes, gt_boxes)
            total_tp += tp
            total_fp += fp
            total_fn += fn

        if count < args.viz_samples:
            vis = image.copy()
            if use_ground_truth:
                vis = _draw_boxes(vis, gt_boxes, color="lime")
            vis = _draw_boxes(vis, pred_boxes, color="red")
            vis.save(os.path.join(viz_dir, f"{idx:05d}_boxes.png"))

        count += 1
        if args.sleep > 0:
            time.sleep(args.sleep)

    if count == 0:
        raise RuntimeError(f"No samples evaluated successfully (failures={failures}).")
    if use_ground_truth:
        micro_denom = 2 * total_tp + total_fp + total_fn
        micro_f1 = 1.0 if micro_denom == 0 else (2 * total_tp) / micro_denom
        macro_f1 = total_f1 / count
        miou = total_miou / count
    else:
        micro_f1 = None
        macro_f1 = None
        miou = None
    summary = {
        "samples": count,
        "failures": failures,
        "eval_f1": micro_f1,
        "eval_f1_macro": macro_f1,
        "eval_miou": miou,
        "eval_true_pos": total_tp if use_ground_truth else None,
        "eval_false_pos": total_fp if use_ground_truth else None,
        "eval_false_neg": total_fn if use_ground_truth else None,
        "model": model,
        "split": args.split,
        "val_dir": args.val_dir,
        "has_ground_truth": use_ground_truth,
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
