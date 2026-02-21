"""Detect finetuning example for Amazon logo variants.

Requires:
  pip install datasets pillow numpy scipy wandb
"""

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
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import wandb
from datasets import load_dataset, load_from_disk
from dotenv import load_dotenv
from PIL import Image, ImageEnhance
from scipy.optimize import linear_sum_assignment

# Ensure repo-root imports (tuna_sdk) work when this file is run directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TUNA_SDK_PATH = os.environ.get("TUNA_SDK_PATH")
if TUNA_SDK_PATH:
    sdk_src = os.path.join(TUNA_SDK_PATH, "src")
    if os.path.isdir(sdk_src) and sdk_src not in sys.path:
        sys.path.insert(0, sdk_src)
    elif TUNA_SDK_PATH not in sys.path:
        sys.path.insert(0, TUNA_SDK_PATH)

from tuna_sdk import (  # noqa: E402
    DetectAnnotation,
    DetectOutput,
    DetectRequest,
    DetectSettings,
    Rollout,
    TrainStepGroup,
    TunaClient,
)
from tuna_sdk.errors import TunaAPIError, TunaNetworkError  # noqa: E402

AMAZON_DATASET = "maxs-m87/Amazon_NBA_re"
OBJECT_NAME = "amazon logo"


@dataclass(frozen=True)
class Sample:
    image: Image.Image
    boxes: List[DetectAnnotation]
    object_name: str
    source: str
    prompt_override: Optional[str] = None


@dataclass(frozen=True)
class AugmentConfig:
    flip_p: float
    crop_p: float
    crop_scale_min: float
    crop_scale_max: float
    resize_min: float
    resize_max: float
    stretch_p: float
    stretch_min: float
    stretch_max: float
    color_p: float
    brightness_min: float
    brightness_max: float
    contrast_min: float
    contrast_max: float
    saturation_min: float
    saturation_max: float
    hue_p: float
    hue_shift_deg_min: float
    hue_shift_deg_max: float
    noise_p: float
    noise_pixel_fraction_max: float


def _to_data_url(
    image: Image.Image,
    *,
    format: str = "PNG",
    quality: Optional[int] = None,
) -> str:
    buf = io.BytesIO()
    fmt = format.upper()
    save_kwargs: dict[str, int] = {}
    if fmt == "JPEG":
        quality_value = 95 if quality is None else int(round(quality))
        save_kwargs["quality"] = max(1, min(100, quality_value))
    image.save(buf, format=fmt, **save_kwargs)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    if fmt == "JPEG":
        mime = "image/jpeg"
    elif fmt == "PNG":
        mime = "image/png"
    else:
        mime = f"image/{fmt.lower()}"
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
    }
    if user_agent:
        headers["User-Agent"] = user_agent
    else:
        headers["User-Agent"] = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    return headers


def _random_suffix(length: int = 6) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def _random_jpeg_quality(rng: random.Random) -> int:
    return int(round(rng.uniform(0.5, 1.0) * 100))


def _box_from_normalized(x_min: float, y_min: float, x_max: float, y_max: float) -> DetectAnnotation:
    x_min = _clamp(float(x_min), 0.0, 1.0)
    y_min = _clamp(float(y_min), 0.0, 1.0)
    x_max = _clamp(float(x_max), 0.0, 1.0)
    y_max = _clamp(float(y_max), 0.0, 1.0)
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("Invalid bbox after clipping")
    return DetectAnnotation(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)


def _bbox_xyxy_to_box(bbox_xyxy: List[float], width: int, height: int) -> DetectAnnotation:
    if len(bbox_xyxy) != 4:
        raise ValueError(f"Expected bbox length 4, got {len(bbox_xyxy)}")
    x_min, y_min, x_max, y_max = bbox_xyxy
    x_min = _clamp(float(x_min), 0.0, float(width))
    y_min = _clamp(float(y_min), 0.0, float(height))
    x_max = _clamp(float(x_max), 0.0, float(width))
    y_max = _clamp(float(y_max), 0.0, float(height))
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("Invalid bbox after clipping")
    return DetectAnnotation(
        x_min=x_min / width,
        y_min=y_min / height,
        x_max=x_max / width,
        y_max=y_max / height,
    )


def _parse_amazon_boxes(
    answer_boxes: Optional[str],
    width: int,
    height: int,
) -> tuple[List[DetectAnnotation], set[str]]:
    if not answer_boxes:
        return [], set()
    raw = answer_boxes
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return [], set()
        try:
            raw = json.loads(text)
        except json.JSONDecodeError:
            return [], set()
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return [], set()
    boxes: List[DetectAnnotation] = []
    variants: set[str] = set()
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        try:
            x_min = float(item["x_min"])
            y_min = float(item["y_min"])
            x_max = float(item["x_max"])
            y_max = float(item["y_max"])
        except (KeyError, TypeError, ValueError):
            continue
        attributes = item.get("attributes") if isinstance(item, dict) else None
        for attr in attributes or []:
            key = attr.get("key") if isinstance(attr, dict) else None
            value = attr.get("value") if isinstance(attr, dict) else None
            if not key or not value:
                continue
            key_norm = str(key).strip().lower()
            if key_norm in ("variant", "varient"):
                variants.add(str(value).strip())
        try:
            if max(x_max, y_max) > 1.5:
                boxes.append(_bbox_xyxy_to_box([x_min, y_min, x_max, y_max], width, height))
            else:
                boxes.append(_box_from_normalized(x_min, y_min, x_max, y_max))
        except ValueError:
            continue
    return boxes, variants


def _parse_amazon_val_boxes(answer: Optional[list]) -> List[DetectAnnotation]:
    if not answer:
        return []
    boxes: List[DetectAnnotation] = []
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


def _iter_local_eval_samples(
    *,
    val_json_path: str,
    image_dir: str,
    max_boxes: Optional[int],
) -> Iterable[Sample]:
    with open(val_json_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    for item in data:
        filename = item.get("filename")
        if not filename:
            continue
        image_path = filename if os.path.isabs(filename) else os.path.join(image_dir, filename)
        try:
            image = Image.open(image_path).convert("RGB")
        except (FileNotFoundError, OSError) as exc:
            print(f"eval image load failed: {image_path} ({exc}). skipping sample.")
            continue
        boxes = _parse_amazon_val_boxes(item.get("answer"))
        if max_boxes is not None:
            boxes = boxes[:max_boxes]
        yield Sample(image=image, boxes=boxes, object_name=OBJECT_NAME, source="eval")


def _stream_amazon_samples(
    *,
    dataset_name: str,
    dataset_path: Optional[str],
    split: str,
    token: Optional[str],
    seed: int,
    buffer_size: int,
    max_boxes: Optional[int],
    empty_keep_prob: float,
) -> Iterable[Sample]:
    rng = random.Random(seed)
    if dataset_path:
        dataset_obj = load_from_disk(dataset_path)
        if hasattr(dataset_obj, "keys"):
            if split not in dataset_obj:
                raise ValueError(f"Split '{split}' not found in local dataset.")
            local_ds = dataset_obj[split]
        else:
            local_ds = dataset_obj
        epoch = 0
        while True:
            shuffled = local_ds.shuffle(seed=seed + epoch)
            for row in shuffled:
                image = row["image"].convert("RGB")
                width, height = image.size
                boxes, variants = _parse_amazon_boxes(row.get("answer_boxes"), width, height)
                if max_boxes is not None:
                    boxes = boxes[:max_boxes]
                if not boxes and empty_keep_prob < 1.0 and rng.random() > empty_keep_prob:
                    continue
                prompt_override = None
                if variants:
                    variant_text = ", ".join(sorted(variants))
                    prompt_override = f"{OBJECT_NAME} variant {variant_text}"
                yield Sample(
                    image=image,
                    boxes=boxes,
                    object_name=OBJECT_NAME,
                    source="amazon",
                    prompt_override=prompt_override,
                )
            epoch += 1
    else:
        resolved_split = split
        while True:
            try:
                ds = load_dataset(dataset_name, split=resolved_split, streaming=True, token=token)
            except ValueError as exc:
                if "[" in resolved_split and "]" in resolved_split:
                    base_split = resolved_split.split("[", 1)[0]
                    print(
                        f"split '{resolved_split}' not supported for streaming; "
                        f"falling back to '{base_split}'. Consider --val-json or --eval-max-samples."
                    )
                    resolved_split = base_split
                    ds = load_dataset(dataset_name, split=resolved_split, streaming=True, token=token)
                else:
                    raise exc
            ds = ds.shuffle(seed=seed, buffer_size=buffer_size)
            for row in ds:
                image = row["image"].convert("RGB")
                width, height = image.size
                boxes, variants = _parse_amazon_boxes(row.get("answer_boxes"), width, height)
                if max_boxes is not None:
                    boxes = boxes[:max_boxes]
                if not boxes and empty_keep_prob < 1.0 and rng.random() > empty_keep_prob:
                    continue
                prompt_override = None
                if variants:
                    variant_text = ", ".join(sorted(variants))
                    prompt_override = f"{OBJECT_NAME} variant {variant_text}"
                yield Sample(
                    image=image,
                    boxes=boxes,
                    object_name=OBJECT_NAME,
                    source="amazon",
                    prompt_override=prompt_override,
                )


def _format_object_name(name: str, rng: random.Random) -> str:
    base = name.strip()
    if rng.random() < 0.5:
        base = base.lower()
    else:
        if base and base[0].isalpha():
            base = base[0].upper() + base[1:].lower()
    if rng.random() < 0.5 and "logo" not in {part.strip(".,:;!?") for part in base.lower().split()}:
        base = f"{base} logo"
    return base


def _sample_prompt(sample: Sample, rng: random.Random) -> str:
    if sample.prompt_override:
        return sample.prompt_override
    return _format_object_name(sample.object_name, rng)


def _horizontal_flip(image: Image.Image, boxes: List[DetectAnnotation]) -> tuple[Image.Image, List[DetectAnnotation]]:
    flipped = [
        DetectAnnotation(
            x_min=1.0 - box.x_max,
            y_min=box.y_min,
            x_max=1.0 - box.x_min,
            y_max=box.y_max,
        )
        for box in boxes
    ]
    return image.transpose(Image.FLIP_LEFT_RIGHT), flipped


def _random_crop(
    image: Image.Image,
    boxes: List[DetectAnnotation],
    rng: random.Random,
    *,
    scale_min: float,
    scale_max: float,
) -> tuple[Image.Image, List[DetectAnnotation]]:
    width, height = image.size
    if width < 2 or height < 2:
        return image, boxes
    scale_w = rng.uniform(scale_min, scale_max)
    scale_h = rng.uniform(scale_min, scale_max)
    crop_w = max(1, int(width * scale_w))
    crop_h = max(1, int(height * scale_h))
    if crop_w >= width and crop_h >= height:
        return image, boxes
    left = rng.randint(0, max(0, width - crop_w)) if width > crop_w else 0
    top = rng.randint(0, max(0, height - crop_h)) if height > crop_h else 0
    right = left + crop_w
    bottom = top + crop_h

    kept: List[DetectAnnotation] = []
    for box in boxes:
        x_min = box.x_min * width
        y_min = box.y_min * height
        x_max = box.x_max * width
        y_max = box.y_max * height
        if x_min >= left and y_min >= top and x_max <= right and y_max <= bottom:
            kept.append(
                DetectAnnotation(
                    x_min=(x_min - left) / crop_w,
                    y_min=(y_min - top) / crop_h,
                    x_max=(x_max - left) / crop_w,
                    y_max=(y_max - top) / crop_h,
                )
            )
    return image.crop((left, top, right, bottom)), kept


def _random_stretch(
    image: Image.Image,
    rng: random.Random,
    *,
    scale_min: float,
    scale_max: float,
) -> Image.Image:
    width, height = image.size
    scale_x = rng.uniform(scale_min, scale_max)
    scale_y = rng.uniform(scale_min, scale_max)
    new_width = max(1, int(width * scale_x))
    new_height = max(1, int(height * scale_y))
    if new_width == width and new_height == height:
        return image
    return image.resize((new_width, new_height), resample=Image.BICUBIC)


def _random_resize(
    image: Image.Image,
    boxes: List[DetectAnnotation],
    rng: random.Random,
    *,
    scale_min: float,
    scale_max: float,
) -> tuple[Image.Image, List[DetectAnnotation]]:
    width, height = image.size
    scale = rng.uniform(scale_min, scale_max)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    if new_width == width and new_height == height:
        return image, boxes
    resized = image.resize((new_width, new_height), resample=Image.BICUBIC)
    adjusted: List[DetectAnnotation] = []
    for box in boxes:
        try:
            adjusted.append(_box_from_normalized(box.x_min, box.y_min, box.x_max, box.y_max))
        except ValueError:
            continue
    return resized, adjusted


def _color_jitter(image: Image.Image, rng: random.Random, config: AugmentConfig) -> Image.Image:
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(config.brightness_min, config.brightness_max))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(config.contrast_min, config.contrast_max))
    image = ImageEnhance.Color(image).enhance(rng.uniform(config.saturation_min, config.saturation_max))
    return image


def _hue_shift(image: Image.Image, rng: random.Random, config: AugmentConfig) -> Image.Image:
    shift_degrees = rng.uniform(config.hue_shift_deg_min, config.hue_shift_deg_max)
    shift = shift_degrees / 360.0
    if abs(shift) < 1e-6:
        return image
    hsv = np.asarray(image.convert("HSV"), dtype=np.uint8).copy()
    hue_delta = int(round(shift * 255.0))
    hsv[..., 0] = ((hsv[..., 0].astype(np.int16) + hue_delta) % 256).astype(np.uint8)
    return Image.fromarray(hsv, mode="HSV").convert("RGB")


def _add_noise(image: Image.Image, rng_np: np.random.Generator, config: AugmentConfig) -> Image.Image:
    arr = np.asarray(image).copy()
    height, width = arr.shape[:2]
    pixel_count = height * width
    if pixel_count == 0:
        return image
    noisy_fraction = float(rng_np.uniform(0.0, config.noise_pixel_fraction_max))
    noisy_pixels = int(round(pixel_count * noisy_fraction))
    if noisy_pixels <= 0:
        return image
    noisy_pixels = min(noisy_pixels, pixel_count)
    flat_indices = rng_np.choice(pixel_count, size=noisy_pixels, replace=False)
    ys = flat_indices // width
    xs = flat_indices % width
    arr[ys, xs] = rng_np.integers(0, 256, size=(noisy_pixels, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _augment_sample(
    sample: Sample,
    rng: random.Random,
    rng_np: np.random.Generator,
    config: AugmentConfig,
    *,
    augment_prob: float,
) -> Sample:
    image = sample.image
    boxes = list(sample.boxes)

    image, boxes = _random_resize(
        image,
        boxes,
        rng,
        scale_min=config.resize_min,
        scale_max=config.resize_max,
    )
    if rng.random() >= augment_prob:
        return Sample(image=image, boxes=boxes, object_name=sample.object_name, source=sample.source)

    if rng.random() < config.crop_p:
        image, boxes = _random_crop(
            image,
            boxes,
            rng,
            scale_min=config.crop_scale_min,
            scale_max=config.crop_scale_max,
        )
    if rng.random() < config.flip_p:
        image, boxes = _horizontal_flip(image, boxes)
    if rng.random() < config.stretch_p:
        image = _random_stretch(
            image,
            rng,
            scale_min=config.stretch_min,
            scale_max=config.stretch_max,
        )
    if rng.random() < config.color_p:
        image = _color_jitter(image, rng, config)
    if rng.random() < config.hue_p:
        image = _hue_shift(image, rng, config)
    if rng.random() < config.noise_p:
        image = _add_noise(image, rng_np, config)

    return Sample(image=image, boxes=boxes, object_name=sample.object_name, source=sample.source)


def _box_iou(a: DetectAnnotation, b: DetectAnnotation) -> float:
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


def _match_ious(predicted: List[DetectAnnotation], ground_truth: List[DetectAnnotation]) -> np.ndarray:
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


def _reward_f1(predicted: List[DetectAnnotation], ground_truth: List[DetectAnnotation]) -> float:
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


def _reward_miou(predicted: List[DetectAnnotation], ground_truth: List[DetectAnnotation]) -> float:
    if not predicted and not ground_truth:
        return 1.0
    if not predicted or not ground_truth:
        return 0.0
    matches = _match_ious(predicted, ground_truth)
    denom = max(len(predicted), len(ground_truth))
    return float(matches.sum()) / float(denom) if denom else 0.0


def _count_tp_fp_fn(
    predicted: List[DetectAnnotation],
    ground_truth: List[DetectAnnotation],
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


def _reward_from_rollouts(
    rollouts: List[Rollout],
    gt_boxes: List[DetectAnnotation],
    *,
    fn_penalty: float,
    iou_threshold: float = 0.5,
) -> List[float]:
    rewards = []
    for rollout in rollouts:
        pred_boxes = rollout.output.objects or []
        base_reward = _reward_miou(pred_boxes, gt_boxes)
        _, _, false_neg = _count_tp_fp_fn(pred_boxes, gt_boxes, iou_threshold=iou_threshold)
        if gt_boxes:
            fn_rate = false_neg / max(len(gt_boxes), 1)
        else:
            fn_rate = 0.0
        rewards.append(base_reward - fn_penalty * (fn_rate ** 2))
    return rewards


def _extract_boxes(payload: dict) -> List[DetectAnnotation]:
    raw_boxes = payload.get("objects")
    if raw_boxes is None and isinstance(payload.get("output"), dict):
        raw_boxes = payload["output"].get("objects")
    if raw_boxes is None:
        return []
    boxes: List[DetectAnnotation] = []
    for item in raw_boxes or []:
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
) -> List[DetectAnnotation]:
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
        detail = error_body.strip() or exc.reason
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc
    parsed = json.loads(body) if body else {}
    return _extract_boxes(parsed)


def _iter_eval_samples(
    *,
    dataset_name: str,
    dataset_path: Optional[str],
    split: str,
    token: Optional[str],
    max_boxes: Optional[int],
) -> Iterable[Sample]:
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
            if "[" in split and "]" in split:
                base_split = split.split("[", 1)[0]
                print(
                    f"split '{split}' not supported for streaming; "
                    f"falling back to '{base_split}'. Consider --val-json or --eval-max-samples."
                )
                ds = load_dataset(dataset_name, split=base_split, streaming=True, token=token)
            else:
                raise exc
    for row in ds:
        image = row["image"].convert("RGB")
        width, height = image.size
        boxes, variants = _parse_amazon_boxes(row.get("answer_boxes"), width, height)
        if max_boxes is not None:
            boxes = boxes[:max_boxes]
        prompt_override = None
        if variants:
            variant_text = ", ".join(sorted(variants))
            prompt_override = f"{OBJECT_NAME} variant {variant_text}"
        yield Sample(
            image=image,
            boxes=boxes,
            object_name=OBJECT_NAME,
            source="eval",
            prompt_override=prompt_override,
        )


def _evaluate(
    *,
    finetune,
    dataset_name: str,
    dataset_path: Optional[str],
    split: str,
    token: Optional[str],
    val_json_path: Optional[str],
    val_image_dir: Optional[str],
    batch_size: int,
    max_boxes: Optional[int],
    max_samples: Optional[int],
    rng: random.Random,
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_objects: int,
    max_workers: int,
    eval_progress_every: int,
) -> dict[str, float]:
    total_f1 = 0.0
    total_miou = 0.0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    count = 0
    batch: List[Sample] = []

    if max_samples is None:
        print("eval: max_samples not set; full split evaluation may take a while.")
    if val_json_path:
        eval_iter = _iter_local_eval_samples(
            val_json_path=val_json_path,
            image_dir=val_image_dir or ".",
            max_boxes=max_boxes,
        )
    else:
        eval_iter = _iter_eval_samples(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            split=split,
            token=token,
            max_boxes=max_boxes,
        )
    for sample in eval_iter:
        batch.append(sample)
        if max_samples is not None and count + len(batch) >= max_samples:
            batch = batch[: max_samples - count]
        if len(batch) < batch_size and (max_samples is None or count + len(batch) < max_samples):
            continue

        requests = [
            DetectRequest(
                object_name="amazon logo",
                image_url=_to_data_url(item.image),
                settings=DetectSettings(
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    max_objects=max_objects,
                ),
            )
            for item in batch
        ]
        try:
            results = finetune.rollouts_batch(
                requests=requests,
                num_rollouts=1,
                max_workers=min(max_workers, len(batch)),
            )
        except (TunaAPIError, TunaNetworkError) as exc:
            print(f"eval rollouts_batch failed: {exc}. skipping batch.")
            batch = []
            continue
        for sample_item, result in zip(batch, results):
            pred_boxes = result.rollouts[0].output.objects if result.rollouts else []
            pred_boxes = pred_boxes or []
            total_f1 += _reward_f1(pred_boxes, sample_item.boxes)
            total_miou += _reward_miou(pred_boxes, sample_item.boxes)
            tp, fp, fn = _count_tp_fp_fn(pred_boxes, sample_item.boxes)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            count += 1
        batch = []
        if eval_progress_every > 0 and count > 0 and count % eval_progress_every == 0:
            if max_samples is not None:
                print(f"eval progress: {count}/{max_samples} samples")
            else:
                print(f"eval progress: {count} samples")
        if max_samples is not None and count >= max_samples:
            break

    if batch:
        requests = [
            DetectRequest(
                object_name="amazon logo",
                image_url=_to_data_url(item.image),
                settings=DetectSettings(
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    max_objects=max_objects,
                ),
            )
            for item in batch
        ]
        try:
            results = finetune.rollouts_batch(
                requests=requests,
                num_rollouts=1,
                max_workers=min(max_workers, len(batch)),
            )
        except (TunaAPIError, TunaNetworkError) as exc:
            print(f"eval rollouts_batch failed: {exc}. skipping final batch.")
            results = []
        for sample_item, result in zip(batch, results):
            pred_boxes = result.rollouts[0].output.objects if result.rollouts else []
            pred_boxes = pred_boxes or []
            total_f1 += _reward_f1(pred_boxes, sample_item.boxes)
            total_miou += _reward_miou(pred_boxes, sample_item.boxes)
            tp, fp, fn = _count_tp_fp_fn(pred_boxes, sample_item.boxes)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            count += 1
        if eval_progress_every > 0 and count > 0 and count % eval_progress_every == 0:
            if max_samples is not None:
                print(f"eval progress: {count}/{max_samples} samples")
            else:
                print(f"eval progress: {count} samples")

    if count == 0:
        return {
            "eval_f1": 0.0,
            "eval_miou": 0.0,
            "eval_true_pos": 0,
            "eval_false_pos": 0,
            "eval_false_neg": 0,
            "eval_f1_macro": 0.0,
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
    }


def _evaluate_api(
    *,
    model: str,
    dataset_name: str,
    dataset_path: Optional[str],
    split: str,
    token: Optional[str],
    val_json_path: Optional[str],
    val_image_dir: Optional[str],
    max_boxes: Optional[int],
    max_samples: Optional[int],
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_objects: int,
    timeout: float,
    api_base: str,
    api_key: str,
    eval_progress_every: int,
) -> dict[str, float]:
    total_f1 = 0.0
    total_miou = 0.0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    count = 0

    if max_samples is None:
        print("eval: max_samples not set; full split evaluation may take a while.")
    if val_json_path:
        eval_iter = _iter_local_eval_samples(
            val_json_path=val_json_path,
            image_dir=val_image_dir or ".",
            max_boxes=max_boxes,
        )
    else:
        eval_iter = _iter_eval_samples(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            split=split,
            token=token,
            max_boxes=max_boxes,
        )
    for sample in eval_iter:
        if max_samples is not None and count >= max_samples:
            break
        try:
            pred_boxes = _call_detect_api(
                api_base=api_base,
                api_key=api_key,
                model=model,
                image=sample.image,
                object_name=OBJECT_NAME,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                max_objects=max_objects,
                timeout=timeout,
            )
        except Exception as exc:
            print(f"eval detect failed: {exc}. skipping sample.")
            continue
        total_f1 += _reward_f1(pred_boxes, sample.boxes)
        total_miou += _reward_miou(pred_boxes, sample.boxes)
        tp, fp, fn = _count_tp_fp_fn(pred_boxes, sample.boxes)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        count += 1
        if eval_progress_every > 0 and count > 0 and count % eval_progress_every == 0:
            if max_samples is not None:
                print(f"eval progress: {count}/{max_samples} samples")
            else:
                print(f"eval progress: {count} samples")

    if count == 0:
        return {
            "eval_f1": 0.0,
            "eval_miou": 0.0,
            "eval_true_pos": 0,
            "eval_false_pos": 0,
            "eval_false_neg": 0,
            "eval_f1_macro": 0.0,
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
    }




def _make_val_splits(split: str, val_fraction: float) -> tuple[str, str]:
    clamped = max(0.0, min(0.99, val_fraction))
    train_pct = int(round((1.0 - clamped) * 100))
    train_pct = max(1, min(99, train_pct))
    train_split = f"{split}[:{train_pct}%]"
    val_split = f"{split}[{train_pct}%:]"
    return train_split, val_split


def _list_available_splits(
    *,
    dataset_name: str,
    dataset_path: Optional[str],
    token: Optional[str],
) -> list[str]:
    if dataset_path:
        dataset_obj = load_from_disk(dataset_path)
        if hasattr(dataset_obj, "keys"):
            return list(dataset_obj.keys())
        return []
    try:
        dataset_obj = load_dataset(dataset_name, streaming=True, token=token)
    except Exception:
        return []
    if hasattr(dataset_obj, "keys"):
        return list(dataset_obj.keys())
    return []


def _resolve_split_name(
    split: str,
    *,
    available_splits: list[str],
    kind: str,
) -> str:
    if not available_splits:
        return split
    if split in available_splits:
        return split
    raise ValueError(f"{kind} split '{split}' not found. Available splits: {available_splits}")


def _resolve_val_split(
    val_split: str,
    *,
    available_splits: list[str],
    train_split: str,
) -> Optional[str]:
    if not available_splits:
        return None if val_split == "auto" else val_split

    if val_split == "auto":
        for candidate in ("validation", "val", "valid", "dev", "test"):
            if candidate in available_splits and candidate != train_split:
                return candidate
        for candidate in available_splits:
            if candidate != train_split:
                return candidate
        return None

    if val_split in available_splits:
        return val_split
    raise ValueError(f"val split '{val_split}' not found. Available splits: {available_splits}")


def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--env-file", "--env", default=".env")
    pre_args, _ = pre_parser.parse_known_args()
    if pre_args.env_file:
        load_dotenv(pre_args.env_file, override=False)

    parser = argparse.ArgumentParser(description="Amazon logo detect finetuning.")
    parser.add_argument("--env-file", "--env", default=pre_args.env_file)
    parser.add_argument("--api-key", default=os.environ.get("MOONDREAM_API_KEY"))
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"))
    parser.add_argument("--base-url", default=os.environ.get("TUNA_BASE_URL", "https://api.moondream.ai/v1"))
    parser.add_argument("--detect-base-url", default=os.environ.get("MOONDREAM_BASE_URL"))
    parser.add_argument("--dataset-name", default=AMAZON_DATASET)
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--finetune-name", default=None)
    parser.add_argument("--finetune-id", default=None)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--resume-step", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2.5e-3)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-objects", type=int, default=50)
    parser.add_argument("--max-workers", type=int, default=32)
    parser.add_argument("--max-boxes", type=int, default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--val-split",
        default="auto",
        help="Validation split name (e.g. 'validation'/'val'), or 'auto' to pick from the dataset's splits.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--buffer-size", type=int, default=1000)
    parser.add_argument("--wandb-project", default="tuna-amazon-detect")
    parser.add_argument("--augment-prob", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-max-samples", type=int, default=None)
    parser.add_argument("--eval-timeout", type=float, default=60.0)
    parser.add_argument("--eval-progress-every", type=int, default=20)
    parser.add_argument("--val-json", default=None)
    parser.add_argument("--val-image-dir", default=None)
    parser.add_argument("--empty-keep-prob", type=float, default=0.5)
    parser.add_argument("--fn-penalty", type=float, default=0.5)
    parser.add_argument(
        "--off-policy",
        action="store_true",
        help="Inject a ground-truth detect rollout when reward variance is too low.",
    )
    parser.add_argument(
        "--off-policy-std-thresh",
        type=float,
        default=0.05,
        help="If per-group reward std is below this, consider GT injection.",
    )
    parser.add_argument(
        "--off-policy-max-reward",
        type=float,
        default=0.1,
        help="Only inject if the best rollout reward is below this threshold.",
    )
    parser.add_argument(
        "--off-policy-min-reward",
        type=float,
        default=0.5,
        help="Reward assigned to injected GT rollout (lower-bounded).",
    )
    parser.add_argument(
        "--off-policy-reward-scale",
        type=float,
        default=2.0,
        help="Injected reward = max(min_reward, scale * max_reward), capped at 1.0.",
    )
    parser.add_argument(
        "--best-metric",
        choices=["eval_miou", "eval_f1", "eval_f1_macro"],
        default="eval_miou",
    )
    args = parser.parse_args()

    if args.env_file:
        load_dotenv(args.env_file, override=False)
    if not args.api_key:
        args.api_key = os.environ.get("MOONDREAM_API_KEY")
    if not args.hf_token:
        args.hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    if not args.api_key:
        raise ValueError("MOONDREAM_API_KEY is required")
    if not args.hf_token:
        raise ValueError("HF_TOKEN or HUGGINGFACE_HUB_TOKEN is required for gated datasets")
    if args.resume_step < 0:
        raise ValueError("Resume step must be >= 0")
    if args.empty_keep_prob < 0.0 or args.empty_keep_prob > 1.0:
        raise ValueError("--empty-keep-prob must be in [0.0, 1.0]")
    if args.fn_penalty < 0.0:
        raise ValueError("--fn-penalty must be >= 0.0")
    if args.off_policy_std_thresh < 0.0:
        raise ValueError("--off-policy-std-thresh must be >= 0.0")
    if args.off_policy_max_reward < 0.0:
        raise ValueError("--off-policy-max-reward must be >= 0.0")
    if args.off_policy_min_reward <= 0.0:
        raise ValueError("--off-policy-min-reward must be > 0.0")
    if args.off_policy_reward_scale <= 0.0:
        raise ValueError("--off-policy-reward-scale must be > 0.0")
    if args.eval_max_samples is not None and args.eval_max_samples <= 0:
        raise ValueError("--eval-max-samples must be > 0 when provided")
    if args.eval_batch_size <= 0:
        raise ValueError("--eval-batch-size must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.group_size <= 0:
        raise ValueError("--group-size must be > 0")

    detect_base_url = args.detect_base_url or args.base_url

    if args.dataset_path:
        available_splits = _list_available_splits(
            dataset_name=args.dataset_name,
            dataset_path=args.dataset_path,
            token=args.hf_token,
        )
        train_split = _resolve_split_name(args.split, available_splits=available_splits, kind="train")
        resolved_val = _resolve_val_split(args.val_split, available_splits=available_splits, train_split=train_split)
        val_split = resolved_val or train_split
    elif args.val_json:
        train_split = args.split
        val_split = args.split
    else:
        available_splits = _list_available_splits(
            dataset_name=args.dataset_name,
            dataset_path=None,
            token=args.hf_token,
        )
        train_split = _resolve_split_name(args.split, available_splits=available_splits, kind="train")
        resolved_val = _resolve_val_split(args.val_split, available_splits=available_splits, train_split=train_split)
        if resolved_val is None:
            train_split, val_split = _make_val_splits(args.split, args.val_fraction)
        else:
            val_split = resolved_val

    rng = random.Random(args.seed)
    rng_np = np.random.default_rng(args.seed)
    augment_config = AugmentConfig(
        flip_p=0.5,
        crop_p=0.5,
        crop_scale_min=0.8,
        crop_scale_max=1.0,
        resize_min=0.5,
        resize_max=1.0,
        stretch_p=0.0,
        stretch_min=0.8,
        stretch_max=1.2,
        color_p=0.0,
        brightness_min=0.7,
        brightness_max=1.3,
        contrast_min=0.7,
        contrast_max=1.3,
        saturation_min=0.7,
        saturation_max=1.3,
        hue_p=0.5,
        hue_shift_deg_min=-15.0,
        hue_shift_deg_max=15.0,
        noise_p=0.3,
        noise_pixel_fraction_max=0.001,
    )

    amazon_stream = _stream_amazon_samples(
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        split=train_split,
        token=args.hf_token,
        seed=args.seed,
        buffer_size=args.buffer_size,
        max_boxes=args.max_boxes,
        empty_keep_prob=args.empty_keep_prob,
    )

    if args.finetune_id and args.finetune_name:
        raise ValueError("Provide either --finetune-id or --finetune-name, not both")
    if not args.finetune_id and not args.finetune_name:
        args.finetune_name = f"amazon-detect-{_random_suffix()}"

    client = TunaClient(api_key=args.api_key, base_url=args.base_url)
    if args.finetune_id:
        finetune = client.get_finetune(args.finetune_id)
    else:
        finetune = client.create_finetune(name=args.finetune_name, rank=args.rank)

    run = wandb.init(
        project=args.wandb_project,
        config={
            "api_base_url": args.base_url,
            "detect_base_url": detect_base_url,
            "finetune_id": finetune.finetune_id,
            "finetune_name": finetune.name,
            "lora_rank": args.rank,
            "amazon_dataset": args.dataset_name,
            "dataset_path": args.dataset_path,
            "split": args.split,
            "train_split": train_split,
            "val_split": val_split,
            "seed": args.seed,
            "buffer_size": args.buffer_size,
            "max_boxes": args.max_boxes,
            "num_steps": args.num_steps,
            "resume_step": args.resume_step,
            "batch_size": args.batch_size,
            "group_size": args.group_size,
            "lr": args.lr,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "max_objects": args.max_objects,
            "max_workers": args.max_workers,
            "augment_prob": args.augment_prob,
            "reward": "miou_minus_fn",
            "eval_every": args.eval_every,
            "save_every": args.save_every,
            "best_metric": args.best_metric,
            "val_json": args.val_json,
            "val_image_dir": args.val_image_dir,
            "empty_keep_prob": args.empty_keep_prob,
            "fn_penalty": args.fn_penalty,
            "off_policy": args.off_policy,
            "off_policy_std_thresh": args.off_policy_std_thresh,
            "off_policy_max_reward": args.off_policy_max_reward,
            "off_policy_min_reward": args.off_policy_min_reward,
            "off_policy_reward_scale": args.off_policy_reward_scale,
        },
    )
    run.summary["finetune_id"] = finetune.finetune_id

    baseline_metrics = _evaluate_api(
        model="moondream3-preview",
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        split=val_split,
        token=args.hf_token,
        val_json_path=args.val_json,
        val_image_dir=args.val_image_dir,
        max_boxes=args.max_boxes,
        max_samples=args.eval_max_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        max_objects=args.max_objects,
        timeout=args.eval_timeout,
        api_base=detect_base_url,
        api_key=args.api_key,
        eval_progress_every=args.eval_progress_every,
    )
    wandb.log({f"baseline_{key}": value for key, value in baseline_metrics.items()}, step=args.resume_step)
    print(
        f"baseline eval (raw moondream) f1={baseline_metrics['eval_f1']:.3f} "
        f"macro_f1={baseline_metrics['eval_f1_macro']:.3f} "
        f"miou={baseline_metrics['eval_miou']:.3f} "
        f"tp={baseline_metrics['eval_true_pos']} "
        f"fp={baseline_metrics['eval_false_pos']} "
        f"fn={baseline_metrics['eval_false_neg']}"
    )

    did_initial_eval = False
    best_metric_value: Optional[float] = None
    best_checkpoint_step: Optional[int] = None
    if args.eval_every > 0:
        eval_metrics = _evaluate(
            finetune=finetune,
            dataset_name=args.dataset_name,
            dataset_path=args.dataset_path,
            split=val_split,
            token=args.hf_token,
            val_json_path=args.val_json,
            val_image_dir=args.val_image_dir,
            batch_size=args.eval_batch_size,
            max_boxes=args.max_boxes,
            max_samples=args.eval_max_samples,
            rng=rng,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            max_objects=args.max_objects,
            max_workers=args.max_workers,
            eval_progress_every=args.eval_progress_every,
        )
        wandb.log(eval_metrics, step=args.resume_step)
        print(
            f"eval step {args.resume_step} (pre-train) f1={eval_metrics['eval_f1']:.3f} "
            f"macro_f1={eval_metrics['eval_f1_macro']:.3f} "
            f"miou={eval_metrics['eval_miou']:.3f} "
            f"tp={eval_metrics['eval_true_pos']} "
            f"fp={eval_metrics['eval_false_pos']} "
            f"fn={eval_metrics['eval_false_neg']}"
        )
        did_initial_eval = True
        best_metric_value = eval_metrics.get(args.best_metric)
        best_checkpoint_step = args.resume_step

    for step in range(args.num_steps):
        global_step = args.resume_step + step
        step_start = time.monotonic()
        batch: List[Sample] = []
        for _ in range(args.batch_size):
            sample = next(amazon_stream)
            batch.append(_augment_sample(sample, rng, rng_np, augment_config, augment_prob=args.augment_prob))

        requests = [
            DetectRequest(
                object_name=_sample_prompt(sample, rng),
                image_url=_to_data_url(
                    sample.image,
                    format="JPEG",
                    quality=_random_jpeg_quality(rng),
                ),
                settings=DetectSettings(
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens,
                    max_objects=args.max_objects,
                ),
            )
            for sample in batch
        ]

        try:
            print(f"step {global_step} starting rollouts_batch (batch={len(batch)} group={args.group_size})")
            rollout_start = time.monotonic()
            results = finetune.rollouts_batch(
                requests=requests,
                num_rollouts=args.group_size,
                max_workers=min(args.max_workers, args.batch_size),
            )
            rollout_end = time.monotonic()
        except (TunaAPIError, TunaNetworkError) as exc:
            print(f"rollouts_batch failed at step {global_step}: {exc}. skipping step.")
            continue

        groups = []
        rewards_all: List[float] = []
        off_policy_injected = 0
        for sample, result in zip(batch, results):
            rollouts = list(result.rollouts)
            rewards = _reward_from_rollouts(rollouts, sample.boxes, fn_penalty=args.fn_penalty)

            # Off-policy "GT injection": if reward variance is ~0 (often all zeros),
            # swap one rollout with the ground-truth boxes and assign it a non-zero reward.
            if args.off_policy and rewards and rollouts and sample.boxes:
                mean_reward = sum(rewards) / len(rewards)
                reward_var = sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)
                reward_std = reward_var**0.5
                max_reward = max(rewards)
                if reward_std < args.off_policy_std_thresh and max_reward < args.off_policy_max_reward:
                    replace_idx = rng.randrange(len(rollouts))
                    rollout = rollouts[replace_idx]
                    rollouts[replace_idx] = Rollout(
                        skill=rollout.skill,
                        finish_reason=rollout.finish_reason,
                        output=DetectOutput(objects=list(sample.boxes)),
                        answer_tokens=list(rollout.answer_tokens),
                        thinking_tokens=list(rollout.thinking_tokens),
                        coords=list(rollout.coords),
                        sizes=list(rollout.sizes),
                    )
                    injected_reward = max(
                        float(args.off_policy_min_reward),
                        min(1.0, float(args.off_policy_reward_scale) * float(max_reward)),
                    )
                    rewards[replace_idx] = injected_reward
                    off_policy_injected += 1

            groups.append(TrainStepGroup(request=result.request, rollouts=rollouts, rewards=rewards))
            rewards_all.extend(rewards)

        try:
            print(f"step {global_step} starting train_step (groups={len(groups)})")
            train_start = time.monotonic()
            train_out = finetune.train_step(groups=groups, lr=args.lr)
            train_end = time.monotonic()
        except (TunaAPIError, TunaNetworkError) as exc:
            print(f"train_step failed at step {global_step}: {exc}. skipping step.")
            continue
        reward_mean = float(np.mean(rewards_all)) if rewards_all else 0.0
        reward_var = float(np.var(rewards_all)) if rewards_all else 0.0

        metrics = {
            "reward_mean": reward_mean,
            "reward_var": reward_var,
            "accepted_groups": len(groups),
            "off_policy_injected": off_policy_injected,
            "kl": train_out.kl,
            "router_kl": train_out.router_kl,
            "grad_norm": train_out.grad_norm,
        }
        wandb.log(metrics, step=global_step)
        step_total = time.monotonic() - step_start
        rollout_time = (rollout_end - rollout_start) if "rollout_end" in locals() else None
        train_time = (train_end - train_start) if "train_end" in locals() else None
        print(
            f"step {global_step} reward={reward_mean:.3f} "
            f"kl={train_out.kl} router_kl={train_out.router_kl} grad_norm={train_out.grad_norm} "
            f"rollout_s={rollout_time:.2f} train_s={train_time:.2f} total_s={step_total:.2f}"
        )

        if args.eval_every > 0 and (global_step + 1) % args.eval_every == 0 and not (did_initial_eval and step == 0):
            eval_metrics = _evaluate(
                finetune=finetune,
                dataset_name=args.dataset_name,
                dataset_path=args.dataset_path,
                split=val_split,
                token=args.hf_token,
                val_json_path=args.val_json,
                val_image_dir=args.val_image_dir,
                batch_size=args.eval_batch_size,
                max_boxes=args.max_boxes,
                max_samples=args.eval_max_samples,
                rng=rng,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                max_objects=args.max_objects,
                max_workers=args.max_workers,
                eval_progress_every=args.eval_progress_every,
            )
            wandb.log(eval_metrics, step=global_step)
            print(
                f"eval step {global_step} f1={eval_metrics['eval_f1']:.3f} "
                f"macro_f1={eval_metrics['eval_f1_macro']:.3f} "
                f"miou={eval_metrics['eval_miou']:.3f} "
                f"tp={eval_metrics['eval_true_pos']} "
                f"fp={eval_metrics['eval_false_pos']} "
                f"fn={eval_metrics['eval_false_neg']}"
            )
            current_metric = eval_metrics.get(args.best_metric)
            if current_metric is not None and (
                best_metric_value is None or current_metric > best_metric_value
            ):
                best_metric_value = current_metric
                best_checkpoint_step = global_step
                finetune.save_checkpoint()

        if args.save_every > 0 and (global_step + 1) % args.save_every == 0:
            finetune.save_checkpoint()

    finetune.save_checkpoint()

    if best_checkpoint_step is not None:
        model = f"moondream3-preview/{finetune.finetune_id}@{best_checkpoint_step}"
        best_metrics = _evaluate_api(
            model=model,
            dataset_name=args.dataset_name,
            dataset_path=args.dataset_path,
            split=val_split,
            token=args.hf_token,
            val_json_path=args.val_json,
            val_image_dir=args.val_image_dir,
            max_boxes=args.max_boxes,
            max_samples=args.eval_max_samples,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            max_objects=args.max_objects,
            timeout=args.eval_timeout,
            api_base=detect_base_url,
            api_key=args.api_key,
            eval_progress_every=args.eval_progress_every,
        )
        wandb.log({f"best_checkpoint_{key}": value for key, value in best_metrics.items()}, step=best_checkpoint_step)
        print(
            f"best checkpoint eval (step {best_checkpoint_step}) "
            f"f1={best_metrics['eval_f1']:.3f} "
            f"macro_f1={best_metrics['eval_f1_macro']:.3f} "
            f"miou={best_metrics['eval_miou']:.3f} "
            f"tp={best_metrics['eval_true_pos']} "
            f"fp={best_metrics['eval_false_pos']} "
            f"fn={best_metrics['eval_false_neg']}"
        )

    wandb.finish()
    client.close()


if __name__ == "__main__":
    main()
