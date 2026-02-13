"""RL finetuning loop for Moondream ball-holder detection.

Requires:
  pip install datasets pillow numpy scipy wandb python-dotenv
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import random
import string
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np
import wandb
from datasets import Dataset, DatasetDict, get_dataset_split_names, load_dataset, load_from_disk
from dotenv import load_dotenv
from PIL import Image, ImageEnhance
from scipy.optimize import linear_sum_assignment

from tuna_sdk import (
    DetectAnnotation,
    DetectOutput,
    DetectRequest,
    DetectSettings,
    Rollout,
    TrainStepGroup,
    TunaClient,
)
from tuna_sdk.errors import TunaAPIError, TunaNetworkError


DEFAULT_DATASET = "maxs-m87/Ball-Holder-splits-v1"
DEFAULT_OBJECT_NAME = "ball holder"
PROMPT_VARIANTS = [
    "ball holder",
    "player with the ball",
    "person with the ball",
    "ballhandler",
    "ball handler",
    "player in possession",
    "player holding the basketball",
    "offensive player with the ball",
]

VAL_SPLIT_CANDIDATES = ("validation", "val", "dev")


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
    noise_std_min: float
    noise_std_max: float


def _repo_relative(*parts: str) -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, *parts)


def _random_suffix(length: int = 6) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


def _to_data_url(
    image: Image.Image,
    *,
    format: str = "JPEG",
    quality: int = 90,
) -> str:
    buf = io.BytesIO()
    fmt = format.upper()
    save_kwargs: dict[str, int] = {}
    if fmt == "JPEG":
        save_kwargs["quality"] = max(1, min(100, int(round(quality))))
    image.save(buf, format=fmt, **save_kwargs)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = "image/jpeg" if fmt == "JPEG" else "image/png"
    return f"data:{mime};base64,{encoded}"


def _random_jpeg_quality(rng: random.Random) -> int:
    return int(round(rng.uniform(55, 98)))


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def _box_from_normalized(x_min: float, y_min: float, x_max: float, y_max: float) -> DetectAnnotation:
    x_min = _clamp(float(x_min), 0.0, 1.0)
    y_min = _clamp(float(y_min), 0.0, 1.0)
    x_max = _clamp(float(x_max), 0.0, 1.0)
    y_max = _clamp(float(y_max), 0.0, 1.0)
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("Invalid normalized bbox")
    return DetectAnnotation(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)


def _bbox_xyxy_to_box(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    width: int,
    height: int,
) -> DetectAnnotation:
    x_min = _clamp(float(x_min), 0.0, float(width))
    y_min = _clamp(float(y_min), 0.0, float(height))
    x_max = _clamp(float(x_max), 0.0, float(width))
    y_max = _clamp(float(y_max), 0.0, float(height))
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("Invalid pixel bbox")
    return DetectAnnotation(
        x_min=x_min / width,
        y_min=y_min / height,
        x_max=x_max / width,
        y_max=y_max / height,
    )


def _parse_answer_boxes(value: object, width: int, height: int) -> tuple[List[DetectAnnotation], bool]:
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

    boxes: List[DetectAnnotation] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        # format 1: direct xyxy
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

        # format 2: nested center-style box (statefarm val style)
        box = item.get("box")
        if isinstance(box, dict) and all(key in box for key in ("x_center", "y_center", "width", "height")):
            try:
                x_center = float(box["x_center"])
                y_center = float(box["y_center"])
                box_w = float(box["width"])
                box_h = float(box["height"])
            except (TypeError, ValueError):
                continue
            x_min = x_center - box_w / 2.0
            y_min = y_center - box_h / 2.0
            x_max = x_center + box_w / 2.0
            y_max = y_center + box_h / 2.0
            try:
                boxes.append(_box_from_normalized(x_min, y_min, x_max, y_max))
            except ValueError:
                continue
    return boxes, False


def _parse_boxes_from_row(
    row: dict,
    *,
    width: int,
    height: int,
    annotation_field: str,
    fallback_field: Optional[str],
) -> tuple[List[DetectAnnotation], bool]:
    boxes, malformed = _parse_answer_boxes(row.get(annotation_field), width, height)
    if boxes or malformed:
        return boxes, malformed
    if fallback_field:
        fallback_boxes, fallback_malformed = _parse_answer_boxes(row.get(fallback_field), width, height)
        return fallback_boxes, fallback_malformed
    return [], False


def _to_sample(
    row: dict,
    *,
    annotation_field: str,
    fallback_field: Optional[str],
    max_boxes: Optional[int],
) -> Optional[Sample]:
    image = row.get("image")
    if image is None:
        return None
    image = image.convert("RGB")
    width, height = image.size
    boxes, _ = _parse_boxes_from_row(
        row,
        width=width,
        height=height,
        annotation_field=annotation_field,
        fallback_field=fallback_field,
    )
    if max_boxes is not None:
        boxes = boxes[:max_boxes]
    return Sample(image=image, boxes=boxes, object_name=DEFAULT_OBJECT_NAME, source="ballholder")


def _make_val_split_expr(train_split: str, val_fraction: float) -> tuple[str, str]:
    val_pct = int(round(val_fraction * 100))
    val_pct = max(1, min(99, val_pct))
    train_expr = f"{train_split}[:-{val_pct}%]"
    val_expr = f"{train_split}[-{val_pct}%:]"
    return train_expr, val_expr


def _load_local_split(dataset_path: str, split: str) -> Dataset:
    dataset_obj = load_from_disk(dataset_path)
    if isinstance(dataset_obj, DatasetDict):
        if split not in dataset_obj:
            available = ", ".join(dataset_obj.keys())
            raise ValueError(f"Split '{split}' not found in local dataset. Available: {available}")
        return dataset_obj[split]
    return dataset_obj


def _resolve_hf_splits(
    *,
    dataset_name: str,
    token: Optional[str],
    requested_train_split: str,
    requested_val_split: str,
) -> tuple[str, Optional[str]]:
    """Resolve train/val split names for an HF dataset.

    Returns (train_split, val_split_or_none). If val split is None, caller should auto-split.
    """

    try:
        split_names = list(get_dataset_split_names(dataset_name, token=token))
    except Exception:
        split_names = []

    train_split = requested_train_split or "train"
    if split_names and train_split not in split_names:
        train_split = "train" if "train" in split_names else split_names[0]

    if requested_val_split:
        if split_names and requested_val_split not in split_names:
            raise ValueError(
                f"--val-split '{requested_val_split}' not found in dataset splits: {split_names}"
            )
        return train_split, requested_val_split if requested_val_split else None

    if split_names:
        for candidate in VAL_SPLIT_CANDIDATES:
            if candidate in split_names:
                return train_split, candidate
        # As a last resort, allow "test" as validation if nothing else exists.
        if "test" in split_names:
            return train_split, "test"

    return train_split, None


def _stream_samples_from_dataset(
    ds: Dataset,
    *,
    seed: int,
    max_boxes: Optional[int],
    empty_keep_prob: float,
    rng: random.Random,
    annotation_field: str,
    fallback_field: Optional[str],
) -> Iterable[Sample]:
    epoch = 0
    while True:
        shuffled = ds.shuffle(seed=seed + epoch) if seed else ds
        for row in shuffled:
            sample = _to_sample(
                row,
                annotation_field=annotation_field,
                fallback_field=fallback_field,
                max_boxes=max_boxes,
            )
            if sample is None:
                continue
            if not sample.boxes and empty_keep_prob < 1.0 and rng.random() > empty_keep_prob:
                continue
            yield sample
        epoch += 1


def _iter_samples_from_dataset(
    ds: Dataset,
    *,
    max_boxes: Optional[int],
    annotation_field: str,
    fallback_field: Optional[str],
) -> Iterable[Sample]:
    for row in ds:
        sample = _to_sample(
            row,
            annotation_field=annotation_field,
            fallback_field=fallback_field,
            max_boxes=max_boxes,
        )
        if sample is not None:
            yield sample


def _stream_training_samples(
    *,
    dataset_name: str,
    dataset_path: Optional[str],
    split: str,
    token: Optional[str],
    seed: int,
    buffer_size: int,
    annotation_field: str,
    fallback_field: Optional[str],
    max_boxes: Optional[int],
    empty_keep_prob: float,
    rng: random.Random,
) -> Iterable[Sample]:
    epoch = 0
    if dataset_path:
        local_ds = _load_local_split(dataset_path, split)
        while True:
            shuffled = local_ds.shuffle(seed=seed + epoch) if seed else local_ds
            for row in shuffled:
                sample = _to_sample(
                    row,
                    annotation_field=annotation_field,
                    fallback_field=fallback_field,
                    max_boxes=max_boxes,
                )
                if sample is None:
                    continue
                if not sample.boxes and empty_keep_prob < 1.0 and rng.random() > empty_keep_prob:
                    continue
                yield sample
            epoch += 1
        return

    while True:
        resolved_split = split
        try:
            ds = load_dataset(dataset_name, split=resolved_split, streaming=True, token=token)
        except ValueError as exc:
            if "[" in resolved_split and "]" in resolved_split:
                resolved_split = resolved_split.split("[", 1)[0]
                print(
                    f"split '{split}' is not supported in streaming mode, "
                    f"falling back to '{resolved_split}'."
                )
                ds = load_dataset(dataset_name, split=resolved_split, streaming=True, token=token)
            else:
                raise exc
        if seed:
            ds = ds.shuffle(seed=seed + epoch, buffer_size=buffer_size)
        for row in ds:
            sample = _to_sample(
                row,
                annotation_field=annotation_field,
                fallback_field=fallback_field,
                max_boxes=max_boxes,
            )
            if sample is None:
                continue
            if not sample.boxes and empty_keep_prob < 1.0 and rng.random() > empty_keep_prob:
                continue
            yield sample
        epoch += 1


def _iter_eval_samples(
    *,
    dataset_name: str,
    dataset_path: Optional[str],
    split: str,
    token: Optional[str],
    annotation_field: str,
    fallback_field: Optional[str],
    max_boxes: Optional[int],
) -> Iterable[Sample]:
    if dataset_path:
        ds = _load_local_split(dataset_path, split)
        for row in ds:
            sample = _to_sample(
                row,
                annotation_field=annotation_field,
                fallback_field=fallback_field,
                max_boxes=max_boxes,
            )
            if sample is not None:
                yield sample
        return

    ds = load_dataset(dataset_name, split=split, streaming=True, token=token)
    for row in ds:
        sample = _to_sample(
            row,
            annotation_field=annotation_field,
            fallback_field=fallback_field,
            max_boxes=max_boxes,
        )
        if sample is not None:
            yield sample


def _sample_prompt(sample: Sample, rng: random.Random) -> str:
    if sample.prompt_override:
        return sample.prompt_override
    prompt = rng.choice(PROMPT_VARIANTS)
    if rng.random() < 0.15:
        prompt = f"the {prompt}"
    if rng.random() < 0.5:
        prompt = prompt.lower()
    return prompt


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
    adjusted = []
    for box in boxes:
        try:
            adjusted.append(_box_from_normalized(box.x_min, box.y_min, box.x_max, box.y_max))
        except ValueError:
            continue
    return resized, adjusted


def _random_crop_include_boxes(
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

    if boxes:
        # Keep all current GT boxes inside crop if possible.
        x_min_px = min(box.x_min for box in boxes) * width
        y_min_px = min(box.y_min for box in boxes) * height
        x_max_px = max(box.x_max for box in boxes) * width
        y_max_px = max(box.y_max for box in boxes) * height

        if (x_max_px - x_min_px) > crop_w or (y_max_px - y_min_px) > crop_h:
            return image, boxes

        left_low = int(np.floor(max(0.0, x_max_px - crop_w)))
        left_high = int(np.ceil(min(x_min_px, width - crop_w)))
        top_low = int(np.floor(max(0.0, y_max_px - crop_h)))
        top_high = int(np.ceil(min(y_min_px, height - crop_h)))
        if left_low > left_high or top_low > top_high:
            return image, boxes
        left = rng.randint(left_low, left_high)
        top = rng.randint(top_low, top_high)
    else:
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
        box_area = max(0.0, x_max - x_min) * max(0.0, y_max - y_min)
        if box_area <= 0.0:
            return image, boxes

        inter_x_min = max(x_min, float(left))
        inter_y_min = max(y_min, float(top))
        inter_x_max = min(x_max, float(right))
        inter_y_max = min(y_max, float(bottom))
        inter_w = max(0.0, inter_x_max - inter_x_min)
        inter_h = max(0.0, inter_y_max - inter_y_min)
        inter_area = inter_w * inter_h

        # Revert crop if more than 30% of any GT box is cut off / outside.
        # (i.e., keep at least 70% of GT area inside the crop).
        if (inter_area / box_area) < 0.70:
            return image, boxes

        kept.append(
            DetectAnnotation(
                x_min=(inter_x_min - left) / crop_w,
                y_min=(inter_y_min - top) / crop_h,
                x_max=(inter_x_max - left) / crop_w,
                y_max=(inter_y_max - top) / crop_h,
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


def _color_jitter(image: Image.Image, rng: random.Random, cfg: AugmentConfig) -> Image.Image:
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(cfg.brightness_min, cfg.brightness_max))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(cfg.contrast_min, cfg.contrast_max))
    image = ImageEnhance.Color(image).enhance(rng.uniform(cfg.saturation_min, cfg.saturation_max))
    return image


def _hue_shift(image: Image.Image, rng: random.Random, cfg: AugmentConfig) -> Image.Image:
    shift_degrees = rng.uniform(cfg.hue_shift_deg_min, cfg.hue_shift_deg_max)
    shift = shift_degrees / 360.0
    if abs(shift) < 1e-6:
        return image
    hsv = np.asarray(image.convert("HSV"), dtype=np.uint8).copy()
    hue_delta = int(round(shift * 255.0))
    hsv[..., 0] = ((hsv[..., 0].astype(np.int16) + hue_delta) % 256).astype(np.uint8)
    return Image.fromarray(hsv, mode="HSV").convert("RGB")


def _add_noise(image: Image.Image, rng_np: np.random.Generator, cfg: AugmentConfig) -> Image.Image:
    arr = np.asarray(image).astype(np.float32)
    std = float(rng_np.uniform(cfg.noise_std_min, cfg.noise_std_max))
    if std <= 0.0:
        return image
    noise = rng_np.normal(0.0, std, size=arr.shape)
    arr = np.clip(arr + noise, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _augment_sample(
    sample: Sample,
    rng: random.Random,
    rng_np: np.random.Generator,
    cfg: AugmentConfig,
    *,
    augment_prob: float,
) -> Sample:
    image = sample.image
    boxes = list(sample.boxes)

    image, boxes = _random_resize(image, boxes, rng, scale_min=cfg.resize_min, scale_max=cfg.resize_max)
    if rng.random() >= augment_prob:
        return Sample(
            image=image,
            boxes=boxes,
            object_name=sample.object_name,
            source=sample.source,
            prompt_override=sample.prompt_override,
        )

    if rng.random() < cfg.crop_p:
        image, boxes = _random_crop_include_boxes(
            image,
            boxes,
            rng,
            scale_min=cfg.crop_scale_min,
            scale_max=cfg.crop_scale_max,
        )
    if rng.random() < cfg.flip_p:
        image, boxes = _horizontal_flip(image, boxes)
    if rng.random() < cfg.stretch_p:
        image = _random_stretch(image, rng, scale_min=cfg.stretch_min, scale_max=cfg.stretch_max)
    if rng.random() < cfg.color_p:
        image = _color_jitter(image, rng, cfg)
    if rng.random() < cfg.hue_p:
        image = _hue_shift(image, rng, cfg)
    if rng.random() < cfg.noise_p:
        image = _add_noise(image, rng_np, cfg)

    return Sample(
        image=image,
        boxes=boxes,
        object_name=sample.object_name,
        source=sample.source,
        prompt_override=sample.prompt_override,
    )


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


def _reward_miou(predicted: List[DetectAnnotation], ground_truth: List[DetectAnnotation]) -> float:
    if not predicted and not ground_truth:
        return 1.0
    if not predicted or not ground_truth:
        return 0.0
    matches = _match_ious(predicted, ground_truth)
    denom = max(len(predicted), len(ground_truth))
    return float(matches.sum()) / float(denom) if denom else 0.0


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
    fp_penalty: float,
    iou_threshold: float = 0.5,
) -> List[float]:
    rewards: List[float] = []
    for rollout in rollouts:
        pred_boxes = rollout.output.objects or []
        if not gt_boxes:
            fp = len(pred_boxes)
            reward = 1.0 - (fp_penalty * min(fp, 3))
            rewards.append(float(_clamp(reward, 0.0, 1.0)))
            continue

        base = _reward_miou(pred_boxes, gt_boxes)
        tp, fp, fn = _count_tp_fp_fn(pred_boxes, gt_boxes, iou_threshold=iou_threshold)
        del tp
        fn_rate = fn / max(len(gt_boxes), 1)
        fp_rate = fp / max(len(pred_boxes), 1) if pred_boxes else 0.0
        reward = base - fn_penalty * fn_rate - fp_penalty * fp_rate
        rewards.append(float(_clamp(reward, 0.0, 1.0)))
    return rewards


def _evaluate(
    *,
    finetune,
    dataset_name: str,
    dataset_path: Optional[str],
    split: str,
    token: Optional[str],
    annotation_field: str,
    fallback_field: Optional[str],
    eval_dataset: Optional[Dataset],
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
    positive_samples = 0
    negative_samples = 0
    count = 0
    batch: List[Sample] = []

    if eval_dataset is not None:
        sample_iter = _iter_samples_from_dataset(
            eval_dataset,
            max_boxes=max_boxes,
            annotation_field=annotation_field,
            fallback_field=fallback_field,
        )
    else:
        sample_iter = _iter_eval_samples(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            split=split,
            token=token,
            annotation_field=annotation_field,
            fallback_field=fallback_field,
            max_boxes=max_boxes,
        )

    for sample in sample_iter:
        batch.append(sample)
        if max_samples is not None and count + len(batch) >= max_samples:
            batch = batch[: max_samples - count]
        if len(batch) < batch_size and (max_samples is None or count + len(batch) < max_samples):
            continue

        requests = [
            DetectRequest(
                object_name=_sample_prompt(item, rng),
                image_url=_to_data_url(item.image, format="JPEG", quality=92),
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

        for item, result in zip(batch, results):
            pred_boxes = result.rollouts[0].output.objects if result.rollouts else []
            pred_boxes = pred_boxes or []
            total_f1 += _reward_f1(pred_boxes, item.boxes)
            total_miou += _reward_miou(pred_boxes, item.boxes)
            tp, fp, fn = _count_tp_fp_fn(pred_boxes, item.boxes)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            if item.boxes:
                positive_samples += 1
            else:
                negative_samples += 1
            count += 1

        batch = []
        if eval_progress_every > 0 and count > 0 and count % eval_progress_every == 0:
            max_part = f"/{max_samples}" if max_samples is not None else ""
            print(f"eval progress: {count}{max_part} samples")
        if max_samples is not None and count >= max_samples:
            break

    if batch:
        requests = [
            DetectRequest(
                object_name=_sample_prompt(item, rng),
                image_url=_to_data_url(item.image, format="JPEG", quality=92),
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
        for item, result in zip(batch, results):
            pred_boxes = result.rollouts[0].output.objects if result.rollouts else []
            pred_boxes = pred_boxes or []
            total_f1 += _reward_f1(pred_boxes, item.boxes)
            total_miou += _reward_miou(pred_boxes, item.boxes)
            tp, fp, fn = _count_tp_fp_fn(pred_boxes, item.boxes)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            if item.boxes:
                positive_samples += 1
            else:
                negative_samples += 1
            count += 1

    if count == 0:
        return {
            "eval_f1": 0.0,
            "eval_f1_macro": 0.0,
            "eval_miou": 0.0,
            "eval_true_pos": 0,
            "eval_false_pos": 0,
            "eval_false_neg": 0,
            "eval_positive_samples": 0,
            "eval_negative_samples": 0,
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
        "eval_positive_samples": positive_samples,
        "eval_negative_samples": negative_samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="RL finetune Moondream for ball-holder detection.")
    parser.add_argument("--api-key", default=os.environ.get("MOONDREAM_API_KEY"))
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"))
    parser.add_argument("--env-file", "--env", default=_repo_relative(".env"))
    parser.add_argument("--base-url", default=os.environ.get("TUNA_BASE_URL", "https://api.moondream.ai/v1"))

    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-path", default="", help="Optional local dataset path from save_to_disk().")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--val-split",
        default="",
        help="Validation split name. If omitted, uses dataset's validation/val/dev when present; otherwise auto-splits.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--annotation-field", default="answer_boxes")
    parser.add_argument("--fallback-field", default="answer")

    parser.add_argument("--finetune-id", default="")
    parser.add_argument("--finetune-name", default="")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--resume-step", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--buffer-size", type=int, default=1000)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--max-boxes", type=int, default=1)
    parser.add_argument("--empty-keep-prob", type=float, default=0.5)

    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--max-objects", type=int, default=1)

    parser.add_argument("--augment-prob", type=float, default=0.9)
    parser.add_argument("--fn-penalty", type=float, default=0.35)
    parser.add_argument("--fp-penalty", type=float, default=0.15)
    parser.add_argument("--iou-threshold", type=float, default=0.45)

    parser.add_argument("--off-policy", action="store_true")
    parser.add_argument("--off-policy-std-thresh", type=float, default=0.02)
    parser.add_argument("--off-policy-max-reward", type=float, default=0.15)
    parser.add_argument("--off-policy-min-reward", type=float, default=0.15)
    parser.add_argument("--off-policy-reward-scale", type=float, default=2.0)

    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-max-samples", type=int, default=2000)
    parser.add_argument("--eval-progress-every", type=int, default=200)
    parser.add_argument("--best-metric", choices=["eval_miou", "eval_f1", "eval_f1_macro"], default="eval_miou")

    parser.add_argument("--wandb-project", default="moondream-ballholder-rl")
    parser.add_argument("--wandb-run-name", default="")
    args = parser.parse_args()

    load_dotenv(args.env_file, override=False)
    if not args.api_key:
        args.api_key = os.environ.get("MOONDREAM_API_KEY")
    if not args.hf_token:
        args.hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    if not args.api_key:
        raise ValueError("MOONDREAM_API_KEY is required")
    if not args.hf_token:
        raise ValueError("HF_TOKEN or HUGGINGFACE_HUB_TOKEN is required")
    if args.resume_step < 0:
        raise ValueError("--resume-step must be >= 0")
    if not (0.0 <= args.empty_keep_prob <= 1.0):
        raise ValueError("--empty-keep-prob must be in [0, 1]")
    if args.finetune_id and args.finetune_name:
        raise ValueError("Provide either --finetune-id or --finetune-name, not both")
    if args.off_policy_min_reward <= 0.0:
        raise ValueError("--off-policy-min-reward must be > 0")
    if args.off_policy_reward_scale <= 0.0:
        raise ValueError("--off-policy-reward-scale must be > 0")

    dataset_path = args.dataset_path.strip() or None
    rng = random.Random(args.seed)
    rng_np = np.random.default_rng(args.seed)
    requested_val_split = args.val_split.strip()
    eval_dataset: Optional[Dataset] = None

    if dataset_path:
        dataset_obj = load_from_disk(dataset_path)
        if isinstance(dataset_obj, DatasetDict):
            train_split = args.split if args.split in dataset_obj else ("train" if "train" in dataset_obj else args.split)
            if requested_val_split:
                val_split = requested_val_split
                if val_split not in dataset_obj:
                    raise ValueError(f"--val-split '{val_split}' not found in local dataset splits: {list(dataset_obj)}")
            else:
                val_split = "val" if "val" in dataset_obj else ("validation" if "validation" in dataset_obj else "")
        else:
            train_split = args.split
            val_split = requested_val_split

        if not val_split:
            # Local dataset without an explicit val split: auto-split deterministically.
            full_ds = _load_local_split(dataset_path, train_split)
            split_ds = full_ds.train_test_split(test_size=args.val_fraction, seed=args.seed, shuffle=True)
            train_ds = split_ds["train"]
            eval_dataset = split_ds["test"]
            val_split = f"auto({train_split})"
            train_stream = _stream_samples_from_dataset(
                train_ds,
                seed=args.seed,
                max_boxes=args.max_boxes,
                empty_keep_prob=args.empty_keep_prob,
                rng=rng,
                annotation_field=args.annotation_field,
                fallback_field=args.fallback_field,
            )
        else:
            train_stream = _stream_training_samples(
                dataset_name=args.dataset_name,
                dataset_path=dataset_path,
                split=train_split,
                token=args.hf_token,
                seed=args.seed,
                buffer_size=args.buffer_size,
                annotation_field=args.annotation_field,
                fallback_field=args.fallback_field,
                max_boxes=args.max_boxes,
                empty_keep_prob=args.empty_keep_prob,
                rng=rng,
            )
    else:
        train_split, resolved_val = _resolve_hf_splits(
            dataset_name=args.dataset_name,
            token=args.hf_token,
            requested_train_split=args.split,
            requested_val_split=requested_val_split,
        )
        if resolved_val is None:
            # HF dataset without an explicit val split: auto-split deterministically in-code.
            full_ds = load_dataset(args.dataset_name, split=train_split, token=args.hf_token, streaming=False)
            split_ds = full_ds.train_test_split(test_size=args.val_fraction, seed=args.seed, shuffle=True)
            train_ds = split_ds["train"]
            eval_dataset = split_ds["test"]
            val_split = f"auto({train_split})"
            train_stream = _stream_samples_from_dataset(
                train_ds,
                seed=args.seed,
                max_boxes=args.max_boxes,
                empty_keep_prob=args.empty_keep_prob,
                rng=rng,
                annotation_field=args.annotation_field,
                fallback_field=args.fallback_field,
            )
        else:
            val_split = resolved_val
            train_stream = _stream_training_samples(
                dataset_name=args.dataset_name,
                dataset_path=None,
                split=train_split,
                token=args.hf_token,
                seed=args.seed,
                buffer_size=args.buffer_size,
                annotation_field=args.annotation_field,
                fallback_field=args.fallback_field,
                max_boxes=args.max_boxes,
                empty_keep_prob=args.empty_keep_prob,
                rng=rng,
            )

    if not args.finetune_id and not args.finetune_name:
        args.finetune_name = f"ballholder-detect-{_random_suffix()}"
    augment_config = AugmentConfig(
        flip_p=0.5,
        crop_p=0.5,
        crop_scale_min=0.75,
        crop_scale_max=1.0,
        resize_min=0.6,
        resize_max=1.0,
        stretch_p=0.3,
        stretch_min=0.9,
        stretch_max=1.1,
        color_p=0.4,
        brightness_min=0.75,
        brightness_max=1.25,
        contrast_min=0.75,
        contrast_max=1.25,
        saturation_min=0.75,
        saturation_max=1.25,
        hue_p=0.2,
        hue_shift_deg_min=-10.0,
        hue_shift_deg_max=10.0,
        noise_p=0.25,
        noise_std_min=2.0,
        noise_std_max=10.0,
    )

    client = TunaClient(api_key=args.api_key, base_url=args.base_url)
    if args.finetune_id:
        finetune = client.get_finetune(args.finetune_id)
    else:
        finetune = client.create_finetune(name=args.finetune_name, rank=args.rank)

    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or None,
        config={
            "api_base_url": args.base_url,
            "finetune_id": finetune.finetune_id,
            "finetune_name": finetune.name,
            "lora_rank": args.rank,
            "dataset_name": args.dataset_name,
            "dataset_path": dataset_path,
            "train_split": train_split,
            "val_split": val_split,
            "auto_val_split": eval_dataset is not None,
            "auto_val_fraction": args.val_fraction if eval_dataset is not None else None,
            "annotation_field": args.annotation_field,
            "fallback_field": args.fallback_field,
            "seed": args.seed,
            "buffer_size": args.buffer_size,
            "max_boxes": args.max_boxes,
            "max_objects": args.max_objects,
            "num_steps": args.num_steps,
            "resume_step": args.resume_step,
            "batch_size": args.batch_size,
            "group_size": args.group_size,
            "lr": args.lr,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "augment_prob": args.augment_prob,
            "empty_keep_prob": args.empty_keep_prob,
            "fn_penalty": args.fn_penalty,
            "fp_penalty": args.fp_penalty,
            "iou_threshold": args.iou_threshold,
            "off_policy": args.off_policy,
            "off_policy_std_thresh": args.off_policy_std_thresh,
            "off_policy_max_reward": args.off_policy_max_reward,
            "off_policy_min_reward": args.off_policy_min_reward,
            "off_policy_reward_scale": args.off_policy_reward_scale,
            "eval_every": args.eval_every,
            "save_every": args.save_every,
            "eval_batch_size": args.eval_batch_size,
            "eval_max_samples": args.eval_max_samples,
            "best_metric": args.best_metric,
        },
    )
    run.summary["finetune_id"] = finetune.finetune_id

    did_initial_eval = False
    best_metric_value: Optional[float] = None
    best_checkpoint_step: Optional[int] = None
    if args.eval_every > 0 and args.resume_step == 0 and val_split:
        eval_metrics = _evaluate(
            finetune=finetune,
            dataset_name=args.dataset_name,
            dataset_path=dataset_path,
            split=val_split,
            token=args.hf_token,
            annotation_field=args.annotation_field,
            fallback_field=args.fallback_field,
            eval_dataset=eval_dataset,
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
        did_initial_eval = True
        best_metric_value = eval_metrics.get(args.best_metric)
        best_checkpoint_step = args.resume_step
        print(
            f"eval step {args.resume_step} (pre-train) "
            f"miou={eval_metrics['eval_miou']:.3f} "
            f"f1={eval_metrics['eval_f1']:.3f} "
            f"macro_f1={eval_metrics['eval_f1_macro']:.3f}"
        )

    for step in range(args.num_steps):
        global_step = args.resume_step + step
        step_start = time.monotonic()

        batch: List[Sample] = []
        for _ in range(args.batch_size):
            raw_sample = next(train_stream)
            batch.append(
                _augment_sample(
                    raw_sample,
                    rng,
                    rng_np,
                    augment_config,
                    augment_prob=args.augment_prob,
                )
            )

        requests = [
            DetectRequest(
                object_name=_sample_prompt(sample, rng),
                image_url=_to_data_url(sample.image, format="JPEG", quality=_random_jpeg_quality(rng)),
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

        groups: List[TrainStepGroup] = []
        rewards_all: List[float] = []
        reward_miou_all: List[float] = []
        reward_f1_all: List[float] = []
        off_policy_injected_total = 0
        off_policy_injected_positive = 0
        off_policy_injected_negative = 0
        pos_samples_in_batch = 0
        neg_samples_in_batch = 0

        for sample, result in zip(batch, results):
            rollouts = list(result.rollouts)
            if sample.boxes:
                pos_samples_in_batch += 1
            else:
                neg_samples_in_batch += 1

            rewards = _reward_from_rollouts(
                rollouts,
                sample.boxes,
                fn_penalty=args.fn_penalty,
                fp_penalty=args.fp_penalty,
                iou_threshold=args.iou_threshold,
            )
            rollout_mious = [_reward_miou((r.output.objects or []), sample.boxes) for r in rollouts]
            rollout_f1s = [_reward_f1((r.output.objects or []), sample.boxes) for r in rollouts]

            if args.off_policy and rewards and rollouts:
                mean_reward = sum(rewards) / len(rewards)
                reward_var = sum((value - mean_reward) ** 2 for value in rewards) / len(rewards)
                reward_std = reward_var**0.5
                max_reward = max(rewards)
                if reward_std < args.off_policy_std_thresh and max_reward < args.off_policy_max_reward:
                    replace_idx = rng.randrange(len(rollouts))
                    old_rollout = rollouts[replace_idx]
                    replacement_objects = list(sample.boxes)
                    rollouts[replace_idx] = Rollout(
                        skill=old_rollout.skill,
                        finish_reason=old_rollout.finish_reason,
                        output=DetectOutput(objects=replacement_objects),
                        answer_tokens=list(old_rollout.answer_tokens),
                        thinking_tokens=list(old_rollout.thinking_tokens),
                        coords=list(old_rollout.coords),
                        sizes=list(old_rollout.sizes),
                    )
                    injected_reward = max(
                        float(args.off_policy_min_reward),
                        min(1.0, float(args.off_policy_reward_scale) * float(max_reward)),
                    )
                    rewards[replace_idx] = injected_reward
                    off_policy_injected_total += 1
                    if sample.boxes:
                        off_policy_injected_positive += 1
                    else:
                        off_policy_injected_negative += 1

            groups.append(TrainStepGroup(request=result.request, rollouts=rollouts, rewards=rewards))
            rewards_all.extend(rewards)
            reward_miou_all.extend(rollout_mious)
            reward_f1_all.extend(rollout_f1s)

        try:
            train_start = time.monotonic()
            train_out = finetune.train_step(groups=groups, lr=args.lr)
            train_end = time.monotonic()
        except (TunaAPIError, TunaNetworkError) as exc:
            print(f"train_step failed at step {global_step}: {exc}. skipping step.")
            continue

        reward_mean = float(np.mean(rewards_all)) if rewards_all else 0.0
        reward_var = float(np.var(rewards_all)) if rewards_all else 0.0
        reward_miou_mean = float(np.mean(reward_miou_all)) if reward_miou_all else 0.0
        reward_f1_mean = float(np.mean(reward_f1_all)) if reward_f1_all else 0.0

        metrics = {
            "reward_mean": reward_mean,
            "reward_var": reward_var,
            "reward_miou_mean": reward_miou_mean,
            "reward_f1_mean": reward_f1_mean,
            "accepted_groups": len(groups),
            "batch_positive_samples": pos_samples_in_batch,
            "batch_negative_samples": neg_samples_in_batch,
            "off_policy_injected": off_policy_injected_total,
            "off_policy_injected_positive": off_policy_injected_positive,
            "off_policy_injected_negative": off_policy_injected_negative,
            "kl": train_out.kl if train_out.kl is not None else 0.0,
            "router_kl": train_out.router_kl if train_out.router_kl is not None else 0.0,
            "grad_norm": train_out.grad_norm if train_out.grad_norm is not None else 0.0,
        }
        wandb.log(metrics, step=global_step)

        step_total = time.monotonic() - step_start
        rollout_time = rollout_end - rollout_start
        train_time = train_end - train_start
        print(
            f"step {global_step} reward={reward_mean:.3f} miou={reward_miou_mean:.3f} "
            f"kl={metrics['kl']:.4f} grad_norm={metrics['grad_norm']:.4f} "
            f"rollout_s={rollout_time:.2f} train_s={train_time:.2f} total_s={step_total:.2f}"
        )

        if args.eval_every > 0 and val_split and (global_step + 1) % args.eval_every == 0 and not (
            did_initial_eval and step == 0
        ):
            eval_metrics = _evaluate(
                finetune=finetune,
                dataset_name=args.dataset_name,
                dataset_path=dataset_path,
                split=val_split,
                token=args.hf_token,
                annotation_field=args.annotation_field,
                fallback_field=args.fallback_field,
                eval_dataset=eval_dataset,
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
                f"eval step {global_step} "
                f"miou={eval_metrics['eval_miou']:.3f} "
                f"f1={eval_metrics['eval_f1']:.3f} "
                f"macro_f1={eval_metrics['eval_f1_macro']:.3f} "
                f"tp={eval_metrics['eval_true_pos']} "
                f"fp={eval_metrics['eval_false_pos']} "
                f"fn={eval_metrics['eval_false_neg']}"
            )
            current_metric = eval_metrics.get(args.best_metric)
            if current_metric is not None and (best_metric_value is None or current_metric > best_metric_value):
                best_metric_value = current_metric
                best_checkpoint_step = global_step
                finetune.save_checkpoint()

        if args.save_every > 0 and (global_step + 1) % args.save_every == 0:
            finetune.save_checkpoint()

    finetune.save_checkpoint()
    if best_checkpoint_step is not None:
        run.summary["best_checkpoint_step"] = best_checkpoint_step
        if best_metric_value is not None:
            run.summary[f"best_{args.best_metric}"] = best_metric_value
    run.finish()
    client.close()

    print(
        f"done. finetune_id={finetune.finetune_id} "
        f"best_step={best_checkpoint_step} best_{args.best_metric}={best_metric_value}"
    )


if __name__ == "__main__":
    main()
