"""Detect finetuning example for omega vs other sports logos.

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
from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np
import wandb
from datasets import load_dataset
from PIL import Image, ImageEnhance
from scipy.optimize import linear_sum_assignment

from tuna_sdk import (
    DetectAnnotation,
    DetectRequest,
    DetectSettings,
    Rollout,
    TunaClient,
)
from tuna_sdk.errors import TunaAPIError, TunaNetworkError

OMEGA_DATASET = "maxs-m87/Omega02"
OTHER_DATASET = "moondream/flickr_logos"
TEST_DATASET = "moondream/omega_swimming"
NO_OMEGA_DATASET = "maxs-m87/No_Omega"
OBJECT_NAME = "omega"
OTHER_LOGO_CLASSES = [
    "HP",
    "adidas_symbol",
    "adidas_text",
    "aldi",
    "apple",
    "becks_symbol",
    "becks_text",
    "bmw",
    "carlsberg_symbol",
    "carlsberg_text",
    "chimay_symbol",
    "chimay_text",
    "cocacola",
    "corona_symbol",
    "corona_text",
    "dhl",
    "erdinger_symbol",
    "erdinger_text",
    "esso_symbol",
    "esso_text",
    "fedex",
    "ferrari",
    "ford",
    "fosters_symbol",
    "fosters_text",
    "google",
    "guinness_symbol",
    "guinness_text",
    "heineken",
    "milka",
    "nvidia_symbol",
    "nvidia_text",
    "paulaner_symbol",
    "paulaner_text",
    "pepsi_symbol",
    "pepsi_text",
    "rittersport",
    "shell",
    "singha_symbol",
    "singha_text",
    "starbucks",
    "stellaartois_symbol",
    "stellaartois_text",
    "texaco",
    "tsingtao_symbol",
    "tsingtao_text",
    "ups",
]


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
    noise_p: float
    noise_std_min: float
    noise_std_max: float


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


def _random_suffix(length: int = 6) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def _random_jpeg_quality(rng: random.Random) -> int:
    return int(round(rng.uniform(0.5, 1.0) * 100))


def _normalize_label(name: str) -> str:
    cleaned = name.replace("_", " ").replace("-", " ").strip()
    return " ".join(cleaned.split())


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


def _bbox_xywh_to_box(bbox_xywh: List[float], width: int, height: int) -> DetectAnnotation:
    if len(bbox_xywh) != 4:
        raise ValueError(f"Expected bbox length 4, got {len(bbox_xywh)}")
    x_min, y_min, w, h = bbox_xywh
    return _bbox_xyxy_to_box([x_min, y_min, x_min + w, y_min + h], width, height)


def _parse_omega_boxes(answer_boxes: Optional[str], width: int, height: int) -> List[DetectAnnotation]:
    if not answer_boxes:
        return []
    raw = json.loads(answer_boxes) if isinstance(answer_boxes, str) else answer_boxes
    boxes: List[DetectAnnotation] = []
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


def _select_other_logo(
    objects: dict,
    width: int,
    height: int,
    rng: random.Random,
    fallback_categories: List[str],
) -> tuple[str, List[DetectAnnotation]]:
    bboxes = objects.get("bbox") or []
    categories = objects.get("category") or []
    by_category: dict[str, List[List[float]]] = {}
    for bbox, category in zip(bboxes, categories):
        if not bbox or category is None:
            continue
        normalized = _normalize_label(str(category))
        if not normalized:
            continue
        by_category.setdefault(normalized, []).append(bbox)

    if not by_category:
        normalized_fallback = [_normalize_label(name) for name in fallback_categories]
        normalized_fallback = [name for name in normalized_fallback if name]
        if not normalized_fallback:
            return "", []
        return rng.choice(normalized_fallback), []

    category = rng.choice(list(by_category.keys()))
    boxes: List[DetectAnnotation] = []
    for bbox in by_category[category]:
        try:
            boxes.append(_bbox_xyxy_to_box(bbox, width, height))
        except ValueError:
            continue
    return category, boxes


def _stream_omega_samples(
    *,
    dataset_name: str,
    split: str,
    token: str,
    seed: int,
    buffer_size: int,
    max_boxes: Optional[int],
) -> Iterable[Sample]:
    while True:
        ds = load_dataset(dataset_name, split=split, streaming=True, token=token)
        if seed:
            ds = ds.shuffle(seed=seed, buffer_size=buffer_size)
        for row in ds:
            image = row["image"].convert("RGB")
            width, height = image.size
            boxes = _parse_omega_boxes(row.get("answer_boxes"), width, height)
            if max_boxes is not None:
                boxes = boxes[:max_boxes]
            yield Sample(image=image, boxes=boxes, object_name=OBJECT_NAME, source="omega")


def _stream_other_samples(
    *,
    dataset_name: str,
    split: str,
    token: str,
    seed: int,
    buffer_size: int,
    max_boxes: Optional[int],
    rng: random.Random,
    omega_prompt_prob: float,
    wrong_class_prob: float,
) -> Iterable[Sample]:
    class_names: set[str] = {_normalize_label(name) for name in OTHER_LOGO_CLASSES if _normalize_label(name)}
    while True:
        ds = load_dataset(dataset_name, split=split, streaming=True, token=token)
        if seed:
            ds = ds.shuffle(seed=seed, buffer_size=buffer_size)
        for row in ds:
            image = row["image"].convert("RGB")
            width, height = image.size
            objects = row.get("objects") or {}
            categories = objects.get("category") or []
            for category in categories:
                if category:
                    normalized = _normalize_label(str(category))
                    if normalized:
                        class_names.add(normalized)
            object_name, boxes = _select_other_logo(
                objects,
                width,
                height,
                rng,
                OTHER_LOGO_CLASSES,
            )
            if not object_name:
                continue
            prompt_override = None
            roll = rng.random()
            if roll < omega_prompt_prob:
                prompt_override = "omega" if rng.random() < 0.5 else "omega logo"
                boxes = []
                object_name = OBJECT_NAME
            elif roll < omega_prompt_prob + wrong_class_prob:
                alternatives = [name for name in class_names if name != object_name]
                if alternatives:
                    object_name = rng.choice(alternatives)
                    boxes = []
                else:
                    prompt_override = "omega" if rng.random() < 0.5 else "omega logo"
                    boxes = []
                    object_name = OBJECT_NAME
            if max_boxes is not None:
                boxes = boxes[:max_boxes]
            yield Sample(
                image=image,
                boxes=boxes,
                object_name=object_name,
                source="other",
                prompt_override=prompt_override,
            )


def _stream_no_omega_samples(
    *,
    dataset_name: str,
    split: str,
    token: str,
    seed: int,
    buffer_size: int,
    rng: random.Random,
) -> Iterable[Sample]:
    while True:
        ds = load_dataset(dataset_name, split=split, streaming=True, token=token)
        if seed:
            ds = ds.shuffle(seed=seed, buffer_size=buffer_size)
        for row in ds:
            image = row["image"].convert("RGB")
            prompt_override = "omega" if rng.random() < 0.5 else "omega logo"
            yield Sample(
                image=image,
                boxes=[],
                object_name=OBJECT_NAME,
                source="no_omega",
                prompt_override=prompt_override,
            )


def _stream_test_samples(
    *,
    dataset_name: str,
    split: str,
    token: str,
    seed: int,
    buffer_size: int,
    max_boxes: Optional[int],
    rng: random.Random,
) -> Iterable[Sample]:
    while True:
        ds = load_dataset(dataset_name, split=split, streaming=True, token=token)
        if seed:
            ds = ds.shuffle(seed=seed, buffer_size=buffer_size)
        for row in ds:
            image = row["image"].convert("RGB")
            width, height = image.size
            objects = row.get("objects") or {}
            boxes: List[DetectAnnotation] = []
            for bbox in objects.get("bbox") or []:
                try:
                    boxes.append(_bbox_xywh_to_box(bbox, width, height))
                except ValueError:
                    continue
            if max_boxes is not None:
                boxes = boxes[:max_boxes]
            prompt_override = "omega" if rng.random() < 0.5 else "omega logo"
            yield Sample(
                image=image,
                boxes=boxes,
                object_name=OBJECT_NAME,
                source="test",
                prompt_override=prompt_override,
            )


def _format_object_name(name: str, rng: random.Random) -> str:
    base = name.strip()
    if rng.random() < 0.5:
        base = base.lower()
    else:
        if base and base[0].isalpha():
            base = base[0].upper() + base[1:].lower()
    if rng.random() < 0.5:
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
    # Normalized box coordinates remain valid after a resize.
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


def _add_noise(image: Image.Image, rng_np: np.random.Generator, config: AugmentConfig) -> Image.Image:
    arr = np.asarray(image).astype(np.float32)
    std = rng_np.uniform(config.noise_std_min, config.noise_std_max)
    noise = rng_np.normal(0.0, std, size=arr.shape)
    arr = np.clip(arr + noise, 0.0, 255.0).astype(np.uint8)
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


def _reward_from_rollouts(rollouts: List[Rollout], gt_boxes: List[DetectAnnotation]) -> List[float]:
    rewards = []
    for rollout in rollouts:
        pred_boxes = rollout.output.objects or []
        rewards.append(_reward_miou(pred_boxes, gt_boxes))
    return rewards


def _iter_test_samples(
    *,
    dataset_name: str,
    split: str,
    token: str,
    max_boxes: Optional[int],
) -> Iterable[Sample]:
    ds = load_dataset(dataset_name, split=split, streaming=True, token=token)
    for row in ds:
        image = row["image"].convert("RGB")
        width, height = image.size
        objects = row.get("objects") or {}
        boxes: List[DetectAnnotation] = []
        for bbox in objects.get("bbox") or []:
            try:
                boxes.append(_bbox_xywh_to_box(bbox, width, height))
            except ValueError:
                continue
        if max_boxes is not None:
            boxes = boxes[:max_boxes]
        yield Sample(image=image, boxes=boxes, object_name=OBJECT_NAME, source="test")


def _evaluate(
    *,
    finetune,
    dataset_name: str,
    split: str,
    token: str,
    batch_size: int,
    max_boxes: Optional[int],
    max_samples: Optional[int],
    rng: random.Random,
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_objects: int,
    max_workers: int,
) -> dict[str, float]:
    total_f1 = 0.0
    total_miou = 0.0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    count = 0
    batch: List[Sample] = []

    for sample in _iter_test_samples(
        dataset_name=dataset_name,
        split=split,
        token=token,
        max_boxes=max_boxes,
    ):
        batch.append(sample)
        if max_samples is not None and count + len(batch) >= max_samples:
            batch = batch[: max_samples - count]
        if len(batch) < batch_size and (max_samples is None or count + len(batch) < max_samples):
            continue

        requests = [
            DetectRequest(
                object_name="omega logo",
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
        if max_samples is not None and count >= max_samples:
            break

    if batch:
        requests = [
            DetectRequest(
                object_name="omega logo",
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Omega logo detect finetuning.")
    parser.add_argument("--api-key", default=os.environ.get("MOONDREAM_API_KEY"))
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"))
    parser.add_argument("--base-url", default=os.environ.get("TUNA_BASE_URL", "https://api.moondream.ai/v1"))
    parser.add_argument("--finetune-name", default=None)
    parser.add_argument("--finetune-id", default=None)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--resume-step", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2.5e-3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-objects", type=int, default=50)
    parser.add_argument("--max-workers", type=int, default=32)
    parser.add_argument("--max-boxes", type=int, default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--buffer-size", type=int, default=1000)
    parser.add_argument("--wandb-project", default="tuna-omega-detect")
    parser.add_argument("--omega-prob", type=float, default=0.5)
    parser.add_argument("--no-omega-prob", type=float, default=0.1)
    parser.add_argument("--test-prob", type=float, default=0.05)
    parser.add_argument("--augment-prob", type=float, default=0.2)
    parser.add_argument("--other-omega-prompt-prob", type=float, default=0.1)
    parser.add_argument("--other-wrong-class-prob", type=float, default=0.1)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-max-samples", type=int, default=None)
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("MOONDREAM_API_KEY is required")
    if not args.hf_token:
        raise ValueError("HF_TOKEN or HUGGINGFACE_HUB_TOKEN is required for gated datasets")
    if args.resume_step < 0:
        raise ValueError("Resume step must be >= 0")
    if args.test_prob < 0.0 or args.test_prob > 1.0:
        raise ValueError("Test sampling probability must be between 0.0 and 1.0")
    if args.no_omega_prob < 0.0 or args.no_omega_prob > 1.0:
        raise ValueError("No-omega sampling probability must be between 0.0 and 1.0")
    if args.test_prob + args.no_omega_prob > 1.0:
        raise ValueError("Sum of test and no-omega probabilities must be <= 1.0")
    if args.other_omega_prompt_prob < 0.0 or args.other_wrong_class_prob < 0.0:
        raise ValueError("Other prompt probabilities must be >= 0.0")
    if args.other_omega_prompt_prob + args.other_wrong_class_prob > 1.0:
        raise ValueError("Sum of other prompt probabilities must be <= 1.0")

    rng = random.Random(args.seed)
    rng_np = np.random.default_rng(args.seed)
    omega_aug = AugmentConfig(
        flip_p=0.5,
        crop_p=0.5,
        crop_scale_min=0.6,
        crop_scale_max=1.0,
        resize_min=0.5,
        resize_max=1.0,
        stretch_p=0.5,
        stretch_min=0.8,
        stretch_max=1.2,
        color_p=0.5,
        brightness_min=0.7,
        brightness_max=1.3,
        contrast_min=0.7,
        contrast_max=1.3,
        saturation_min=0.7,
        saturation_max=1.3,
        noise_p=0.3,
        noise_std_min=3.0,
        noise_std_max=12.0,
    )
    other_aug = AugmentConfig(
        flip_p=0.3,
        crop_p=0.3,
        crop_scale_min=0.7,
        crop_scale_max=1.0,
        resize_min=0.5,
        resize_max=1.0,
        stretch_p=0.3,
        stretch_min=0.9,
        stretch_max=1.1,
        color_p=0.3,
        brightness_min=0.8,
        brightness_max=1.2,
        contrast_min=0.8,
        contrast_max=1.2,
        saturation_min=0.8,
        saturation_max=1.2,
        noise_p=0.2,
        noise_std_min=2.0,
        noise_std_max=8.0,
    )

    omega_stream = _stream_omega_samples(
        dataset_name=OMEGA_DATASET,
        split=args.split,
        token=args.hf_token,
        seed=args.seed,
        buffer_size=args.buffer_size,
        max_boxes=args.max_boxes,
    )
    other_stream = _stream_other_samples(
        dataset_name=OTHER_DATASET,
        split=args.split,
        token=args.hf_token,
        seed=args.seed,
        buffer_size=args.buffer_size,
        max_boxes=args.max_boxes,
        rng=rng,
        omega_prompt_prob=args.other_omega_prompt_prob,
        wrong_class_prob=args.other_wrong_class_prob,
    )
    no_omega_stream = _stream_no_omega_samples(
        dataset_name=NO_OMEGA_DATASET,
        split="train",
        token=args.hf_token,
        seed=args.seed,
        buffer_size=args.buffer_size,
        rng=rng,
    )
    test_stream = None
    if args.test_prob > 0:
        test_stream = _stream_test_samples(
            dataset_name=TEST_DATASET,
            split="train",
            token=args.hf_token,
            seed=args.seed,
            buffer_size=args.buffer_size,
            max_boxes=args.max_boxes,
            rng=rng,
        )

    if args.finetune_id and args.finetune_name:
        raise ValueError("Provide either --finetune-id or --finetune-name, not both")
    if not args.finetune_id and not args.finetune_name:
        args.finetune_name = f"omega-detect-{_random_suffix()}"

    client = TunaClient(api_key=args.api_key, base_url=args.base_url)
    if args.finetune_id:
        finetune = client.get_finetune(args.finetune_id)
    else:
        finetune = client.create_finetune(name=args.finetune_name, rank=args.rank)

    run = wandb.init(
        project=args.wandb_project,
        config={
            "api_base_url": args.base_url,
            "finetune_id": finetune.finetune_id,
            "finetune_name": finetune.name,
            "lora_rank": args.rank,
            "omega_dataset": OMEGA_DATASET,
            "other_dataset": OTHER_DATASET,
            "test_dataset": TEST_DATASET,
            "no_omega_dataset": NO_OMEGA_DATASET,
            "split": args.split,
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
            "omega_prob": args.omega_prob,
            "no_omega_prob": args.no_omega_prob,
            "test_prob": args.test_prob,
            "augment_prob": args.augment_prob,
            "other_omega_prompt_prob": args.other_omega_prompt_prob,
            "other_wrong_class_prob": args.other_wrong_class_prob,
            "reward": "miou",
            "eval_every": args.eval_every,
            "save_every": args.save_every,
        },
    )
    run.summary["finetune_id"] = finetune.finetune_id

    did_initial_eval = False
    if args.eval_every > 0:
        eval_metrics = _evaluate(
            finetune=finetune,
            dataset_name=TEST_DATASET,
            split="train",
            token=args.hf_token,
            batch_size=args.eval_batch_size,
            max_boxes=args.max_boxes,
            max_samples=args.eval_max_samples,
            rng=rng,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            max_objects=args.max_objects,
            max_workers=args.max_workers,
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

    for step in range(args.num_steps):
        global_step = args.resume_step + step
        batch: List[Sample] = []
        for _ in range(args.batch_size):
            roll = rng.random()
            if roll < args.test_prob and test_stream is not None:
                sample = next(test_stream)
                config = omega_aug
            elif roll < args.test_prob + args.no_omega_prob:
                sample = next(no_omega_stream)
                config = omega_aug
            else:
                if rng.random() < args.omega_prob:
                    sample = next(omega_stream)
                else:
                    sample = next(other_stream)
                config = omega_aug if sample.source == "omega" else other_aug
            batch.append(_augment_sample(sample, rng, rng_np, config, augment_prob=args.augment_prob))

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
            results = finetune.rollouts_batch(
                requests=requests,
                num_rollouts=args.group_size,
                max_workers=min(args.max_workers, args.batch_size),
            )
        except (TunaAPIError, TunaNetworkError) as exc:
            print(f"rollouts_batch failed at step {global_step}: {exc}. skipping step.")
            continue

        groups = []
        rewards_all: List[float] = []
        for sample, result in zip(batch, results):
            rewards = _reward_from_rollouts(result.rollouts, sample.boxes)
            groups.append(result.to_group(rewards=rewards))
            rewards_all.extend(rewards)

        try:
            train_out = finetune.train_step(groups=groups, lr=args.lr)
        except (TunaAPIError, TunaNetworkError) as exc:
            print(f"train_step failed at step {global_step}: {exc}. skipping step.")
            continue
        reward_mean = float(np.mean(rewards_all)) if rewards_all else 0.0
        reward_var = float(np.var(rewards_all)) if rewards_all else 0.0

        metrics = {
            "reward_mean": reward_mean,
            "reward_var": reward_var,
            "accepted_groups": len(groups),
            "kl": train_out.kl,
            "router_kl": train_out.router_kl,
            "grad_norm": train_out.grad_norm,
        }
        wandb.log(metrics, step=global_step)
        print(
            f"step {global_step} reward={reward_mean:.3f} "
            f"kl={train_out.kl} router_kl={train_out.router_kl} grad_norm={train_out.grad_norm}"
        )

        if args.eval_every > 0 and (global_step + 1) % args.eval_every == 0 and not (did_initial_eval and step == 0):
            eval_metrics = _evaluate(
                finetune=finetune,
                dataset_name=TEST_DATASET,
                split="train",
                token=args.hf_token,
                batch_size=args.eval_batch_size,
                max_boxes=args.max_boxes,
                max_samples=args.eval_max_samples,
                rng=rng,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                max_objects=args.max_objects,
                max_workers=args.max_workers,
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

        if args.save_every > 0 and (global_step + 1) % args.save_every == 0:
            finetune.save_checkpoint()

    finetune.save_checkpoint()
    wandb.finish()
    client.close()


if __name__ == "__main__":
    main()
