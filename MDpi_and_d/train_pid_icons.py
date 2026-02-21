#!/usr/bin/env python3
"""Class-conditional RL finetuning for PI&D symbol detection.

Training behavior:
- For each sample, build class-conditional prompt candidates.
- Train on one sampled task per base sample (BallHolder-style batching).
- For empty samples, create random negative prompts.
- Optional random negative prompts for non-empty samples.

This matches single-prompt detect APIs while covering many classes.
"""

from __future__ import annotations

import argparse
import base64
import itertools
import io
import json
import os
import random
import string
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np
from datasets import Dataset, DatasetDict, get_dataset_split_names, load_dataset, load_from_disk
from dotenv import load_dotenv
from PIL import Image, ImageEnhance
from scipy.optimize import linear_sum_assignment

try:
    import wandb  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    class _WandbRun:
        def __init__(self) -> None:
            self.summary: dict[str, Any] = {}

        def finish(self) -> None:
            return

    class _WandbShim:
        @staticmethod
        def init(*args: Any, **kwargs: Any) -> _WandbRun:
            print("wandb not installed; continuing without remote logging.")
            return _WandbRun()

        @staticmethod
        def log(*args: Any, **kwargs: Any) -> None:
            return

    wandb = _WandbShim()

# Ensure repo-root imports (tuna_sdk) work when this file is run directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tuna_sdk import DetectAnnotation, DetectOutput, DetectRequest, DetectSettings, Rollout, TrainStepGroup, TunaClient
from tuna_sdk import PointAnnotation, PointOutput, PointRequest, PointSettings
from tuna_sdk.errors import TunaAPIError, TunaNetworkError


def _repo_relative(*parts: str) -> Path:
    return Path(__file__).resolve().parent.joinpath(*parts)


def _random_suffix(length: int = 6) -> str:
    chars = string.ascii_lowercase + string.digits
    return "".join(random.choices(chars, k=length))


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _to_data_url(image: Image.Image, *, quality: int = 90) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=max(1, min(100, int(quality))))
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


@dataclass(frozen=True)
class ClassBox:
    class_uid: str
    class_name: str
    box: DetectAnnotation


@dataclass(frozen=True)
class BaseSample:
    image: Image.Image
    boxes: list[ClassBox]
    source: str


@dataclass(frozen=True)
class TaskSample:
    image: Image.Image
    prompt: str
    gt_boxes: list[DetectAnnotation]
    class_name: str
    is_positive: bool
    source: str


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
    hue_delta_min: float
    hue_delta_max: float
    noise_p: float
    noise_std_min: float
    noise_std_max: float


@dataclass
class UsageStats:
    rows_seen: int = 0
    rows_with_boxes: int = 0
    rows_without_boxes: int = 0
    tasks_generated: int = 0
    tasks_generated_positive: int = 0
    tasks_generated_negative: int = 0
    tasks_consumed: int = 0
    tasks_consumed_positive: int = 0
    tasks_consumed_negative: int = 0
    source_rows_seen: Counter[str] = field(default_factory=Counter)
    source_tasks_generated: Counter[str] = field(default_factory=Counter)
    source_tasks_consumed: Counter[str] = field(default_factory=Counter)
    class_tasks_generated: Counter[str] = field(default_factory=Counter)
    class_tasks_consumed: Counter[str] = field(default_factory=Counter)


VAL_SPLIT_CANDIDATES = ("validation", "val", "dev", "test", "post_val")


def _extract_class_catalog(payload: Any) -> list[tuple[str, str]]:
    raw_catalog: Any = None
    if isinstance(payload, dict) and isinstance(payload.get("class_catalog"), list):
        raw_catalog = payload.get("class_catalog")
    elif isinstance(payload, list) and payload and all(isinstance(item, dict) for item in payload):
        raw_catalog = payload

    if not isinstance(raw_catalog, list):
        return []

    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in raw_catalog:
        if not isinstance(item, dict):
            continue
        class_name = str(item.get("class_name", "")).strip()
        if not class_name:
            continue
        class_uid = str(item.get("class_uid") or class_name).strip()
        pair = (class_uid, class_name)
        if pair in seen:
            continue
        seen.add(pair)
        out.append(pair)
        print(f"found class in catalog: uid='{class_uid}', name='{class_name}'")
    return out


def _load_class_catalog(class_names_file: str, dataset_path: Optional[str]) -> list[tuple[str, str]]:
    if class_names_file:
        path = Path(class_names_file).expanduser().resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        catalog = _extract_class_catalog(payload)
        if catalog:
            return catalog
        if isinstance(payload, list):
            names = [str(item).strip() for item in payload if str(item).strip()]
            if names:
                return [(name, name) for name in sorted(set(names))]

    if dataset_path:
        meta_path = Path(dataset_path).expanduser().resolve() / "metadata.json"
        if meta_path.exists():
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
            catalog = _extract_class_catalog(payload)
            if catalog:
                return catalog

    return []


def _load_class_names(class_names_file: str, dataset_path: Optional[str]) -> list[str]:
    catalog = _load_class_catalog(class_names_file, dataset_path)
    if catalog:
        names = [name for _, name in catalog if name]
        if names:
            return sorted(set(names))

    if class_names_file:
        path = Path(class_names_file).expanduser().resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "class_catalog" in payload:
            names = [str(item.get("class_name", "")).strip() for item in payload["class_catalog"]]
            names = [name for name in names if name]
            if names:
                return sorted(set(names))
        if isinstance(payload, list):
            names = [str(item).strip() for item in payload if str(item).strip()]
            if names:
                return sorted(set(names))

    if dataset_path:
        meta_path = Path(dataset_path).expanduser().resolve() / "metadata.json"
        if meta_path.exists():
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
            catalog = payload.get("class_catalog") or []
            names = [str(item.get("class_name", "")).strip() for item in catalog if isinstance(item, dict)]
            names = [name for name in names if name]
            if names:
                return sorted(set(names))

    return []


def _class_name_has_digit(name: str) -> bool:
    return any(char.isdigit() for char in name)


def _analyze_class_catalog(catalog: list[tuple[str, str]]) -> tuple[dict[str, list[str]], list[str]]:
    by_name: dict[str, set[str]] = {}
    for class_uid, class_name in catalog:
        if not class_name:
            continue
        by_name.setdefault(class_name, set()).add(class_uid or class_name)

    duplicate_names = {
        class_name: sorted(uids)
        for class_name, uids in by_name.items()
        if len(uids) > 1
    }
    numeric_names = sorted(name for name in by_name if _class_name_has_digit(name))
    return duplicate_names, numeric_names


def _parse_answer_boxes(value: Any, width: int, height: int) -> list[ClassBox]:
    if value is None:
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

    parsed: list[ClassBox] = []
    for item in raw:
        if not isinstance(item, dict):
            continue

        class_name = str(item.get("class_name") or item.get("source_class_name") or "").strip()
        class_uid = str(item.get("class_uid") or class_name or "").strip()
        if not class_name:
            continue

        x_min = item.get("x_min")
        y_min = item.get("y_min")
        x_max = item.get("x_max")
        y_max = item.get("y_max")
        try:
            x_min_f = float(x_min)
            y_min_f = float(y_min)
            x_max_f = float(x_max)
            y_max_f = float(y_max)
        except (TypeError, ValueError):
            continue

        # Support both normalized and pixel coords.
        if max(abs(x_min_f), abs(y_min_f), abs(x_max_f), abs(y_max_f)) > 1.5:
            if width <= 0 or height <= 0:
                continue
            x_min_f = x_min_f / width
            y_min_f = y_min_f / height
            x_max_f = x_max_f / width
            y_max_f = y_max_f / height

        x_min_f = _clamp(x_min_f)
        y_min_f = _clamp(y_min_f)
        x_max_f = _clamp(x_max_f)
        y_max_f = _clamp(y_max_f)
        if x_max_f <= x_min_f or y_max_f <= y_min_f:
            continue

        parsed.append(
            ClassBox(
                class_uid=class_uid,
                class_name=class_name,
                box=DetectAnnotation(x_min=x_min_f, y_min=y_min_f, x_max=x_max_f, y_max=y_max_f),
            )
        )

    return parsed


def _to_base_sample(row: dict) -> Optional[BaseSample]:
    image = row.get("image")
    if image is None:
        return None
    image = image.convert("RGB")
    width, height = image.size
    boxes = _parse_answer_boxes(row.get("answer_boxes"), width=width, height=height)
    source = str(row.get("source_collection") or row.get("source_dataset") or "unknown")
    return BaseSample(image=image, boxes=boxes, source=source)


def _group_boxes_by_class(boxes: list[ClassBox]) -> dict[str, tuple[str, list[DetectAnnotation]]]:
    grouped: dict[str, tuple[str, list[DetectAnnotation]]] = {}
    for item in boxes:
        if item.class_uid not in grouped:
            grouped[item.class_uid] = (item.class_name, [item.box])
        else:
            grouped[item.class_uid][1].append(item.box)
    return grouped


def _prompt_for_class(class_name: str, *, style: str = "detect_phrase") -> str:
    normalized_style = (style or "detect_phrase").strip().lower()
    if normalized_style == "class_name":
        return class_name
    # This string is sent as the `detect`/`point` "object" to Moondream.
    return f"{class_name} icon or icons"


def _tasks_from_base_sample(
    sample: BaseSample,
    *,
    all_class_names: list[str],
    rng: random.Random,
    neg_prompts_per_empty: int,
    neg_prompts_per_nonempty: int,
    prompt_style: str = "detect_phrase",
) -> list[TaskSample]:
    tasks: list[TaskSample] = []
    grouped = _group_boxes_by_class(sample.boxes)
    present_names = {item[0] for item in grouped.values()}

    if grouped:
        for _, (class_name, boxes) in grouped.items():
            tasks.append(
                TaskSample(
                    image=sample.image,
                    prompt=_prompt_for_class(class_name, style=prompt_style),
                    gt_boxes=list(boxes),
                    class_name=class_name,
                    is_positive=True,
                    source=sample.source,
                )
            )

        absent = [name for name in all_class_names if name not in present_names]
        if absent and neg_prompts_per_nonempty > 0:
            picks = rng.sample(absent, k=min(neg_prompts_per_nonempty, len(absent)))
            for class_name in picks:
                tasks.append(
                    TaskSample(
                        image=sample.image,
                        prompt=_prompt_for_class(class_name, style=prompt_style),
                        gt_boxes=[],
                        class_name=class_name,
                        is_positive=False,
                        source=sample.source,
                    )
                )
        return tasks

    if not all_class_names:
        return []

    if neg_prompts_per_empty <= 0:
        return []

    k = min(neg_prompts_per_empty, len(all_class_names))
    if k <= 0:
        return []
    picks = rng.sample(all_class_names, k=min(k, len(all_class_names)))
    for class_name in picks:
        tasks.append(
            TaskSample(
                image=sample.image,
                prompt=_prompt_for_class(class_name, style=prompt_style),
                gt_boxes=[],
                class_name=class_name,
                is_positive=False,
                source=sample.source,
            )
        )
    return tasks


def _box_from_normalized(x_min: float, y_min: float, x_max: float, y_max: float) -> DetectAnnotation:
    x_min = _clamp(float(x_min))
    y_min = _clamp(float(y_min))
    x_max = _clamp(float(x_max))
    y_max = _clamp(float(y_max))
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("invalid normalized bbox")
    return DetectAnnotation(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)


def _horizontal_flip(image: Image.Image, boxes: list[ClassBox]) -> tuple[Image.Image, list[ClassBox]]:
    flipped: list[ClassBox] = []
    for item in boxes:
        box = item.box
        flipped_box = DetectAnnotation(
            x_min=1.0 - box.x_max,
            y_min=box.y_min,
            x_max=1.0 - box.x_min,
            y_max=box.y_max,
        )
        flipped.append(
            ClassBox(
                class_uid=item.class_uid,
                class_name=item.class_name,
                box=flipped_box,
            )
        )
    return image.transpose(Image.FLIP_LEFT_RIGHT), flipped


def _random_crop(
    image: Image.Image,
    boxes: list[ClassBox],
    rng: random.Random,
    *,
    scale_min: float,
    scale_max: float,
) -> tuple[Image.Image, list[ClassBox]]:
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

    kept: list[ClassBox] = []
    for item in boxes:
        box = item.box
        x_min = box.x_min * width
        y_min = box.y_min * height
        x_max = box.x_max * width
        y_max = box.y_max * height
        if x_min >= left and y_min >= top and x_max <= right and y_max <= bottom:
            kept.append(
                ClassBox(
                    class_uid=item.class_uid,
                    class_name=item.class_name,
                    box=DetectAnnotation(
                        x_min=(x_min - left) / crop_w,
                        y_min=(y_min - top) / crop_h,
                        x_max=(x_max - left) / crop_w,
                        y_max=(y_max - top) / crop_h,
                    ),
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
    boxes: list[ClassBox],
    rng: random.Random,
    *,
    scale_min: float,
    scale_max: float,
) -> tuple[Image.Image, list[ClassBox]]:
    width, height = image.size
    scale = rng.uniform(scale_min, scale_max)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    if new_width == width and new_height == height:
        return image, boxes
    resized = image.resize((new_width, new_height), resample=Image.BICUBIC)
    adjusted: list[ClassBox] = []
    for item in boxes:
        box = item.box
        try:
            adjusted_box = _box_from_normalized(box.x_min, box.y_min, box.x_max, box.y_max)
        except ValueError:
            continue
        adjusted.append(
            ClassBox(
                class_uid=item.class_uid,
                class_name=item.class_name,
                box=adjusted_box,
            )
        )
    return resized, adjusted


def _color_jitter(image: Image.Image, rng: random.Random, config: AugmentConfig) -> Image.Image:
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(config.brightness_min, config.brightness_max))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(config.contrast_min, config.contrast_max))
    image = ImageEnhance.Color(image).enhance(rng.uniform(config.saturation_min, config.saturation_max))
    return image


def _hue_shift(image: Image.Image, rng: random.Random, config: AugmentConfig) -> Image.Image:
    delta = rng.uniform(config.hue_delta_min, config.hue_delta_max)
    shift = int(round(delta * 255.0))
    if shift == 0:
        return image
    hsv = np.asarray(image.convert("HSV"), dtype=np.uint8).copy()
    hsv[..., 0] = ((hsv[..., 0].astype(np.int16) + shift) % 256).astype(np.uint8)
    return Image.fromarray(hsv, mode="HSV").convert("RGB")


def _add_noise(image: Image.Image, rng_np: np.random.Generator, config: AugmentConfig) -> Image.Image:
    arr = np.asarray(image).astype(np.float32)
    std = rng_np.uniform(config.noise_std_min, config.noise_std_max)
    noise = rng_np.normal(0.0, std, size=arr.shape)
    arr = np.clip(arr + noise, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _augment_base_sample(
    sample: BaseSample,
    rng: random.Random,
    rng_np: np.random.Generator,
    config: AugmentConfig,
    *,
    augment_prob: float,
) -> BaseSample:
    image = sample.image
    boxes = list(sample.boxes)

    # Keep augment_prob as a top-level gate so disabling/reducing augmentation
    # also disables random resizing.
    if rng.random() >= augment_prob:
        return BaseSample(image=image, boxes=boxes, source=sample.source)

    image, boxes = _random_resize(
        image,
        boxes,
        rng,
        scale_min=config.resize_min,
        scale_max=config.resize_max,
    )

    if rng.random() < config.crop_p:
        pre_crop_image = image
        pre_crop_boxes = list(boxes)
        cropped_image, cropped_boxes = _random_crop(
            image,
            boxes,
            rng,
            scale_min=config.crop_scale_min,
            scale_max=config.crop_scale_max,
        )
        # Match StateFarm behavior: avoid converting positives into empty labels via crop.
        if pre_crop_boxes and not cropped_boxes:
            image, boxes = pre_crop_image, pre_crop_boxes
        else:
            image, boxes = cropped_image, cropped_boxes
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

    return BaseSample(image=image, boxes=boxes, source=sample.source)


def _default_augment_config() -> AugmentConfig:
    return AugmentConfig(
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
        hue_p=0.5,
        hue_delta_min=-0.08,
        hue_delta_max=0.08,
        noise_p=0.3,
        noise_std_min=3.0,
        noise_std_max=12.0,
    )


def _iter_dataset_rows(ds: Dataset, seed: int) -> Iterable[dict]:
    epoch = 0
    while True:
        shuffled = ds.shuffle(seed=seed + epoch) if seed else ds
        for row in shuffled:
            yield row
        epoch += 1


def _load_local_split(dataset_path: str, split: str) -> Dataset:
    dataset_obj = load_from_disk(dataset_path)
    if isinstance(dataset_obj, DatasetDict):
        if split not in dataset_obj:
            available = ", ".join(dataset_obj.keys())
            raise ValueError(f"Split '{split}' not found in local dataset. Available: {available}")
        return dataset_obj[split]
    return dataset_obj


def _iter_local_rows(dataset_path: str, split: str, seed: int) -> Iterable[dict]:
    ds = _load_local_split(dataset_path, split)
    return _iter_dataset_rows(ds, seed)


def _iter_hf_rows(dataset_name: str, split: str, token: Optional[str], seed: int, buffer_size: int) -> Iterable[dict]:
    epoch = 0
    while True:
        ds = load_dataset(dataset_name, split=split, token=token, streaming=True)
        if seed:
            ds = ds.shuffle(seed=seed + epoch, buffer_size=buffer_size)
        for row in ds:
            yield row
        epoch += 1


def _iter_local_rows_once(dataset_obj: Dataset | DatasetDict, split: Optional[str]) -> Iterable[dict]:
    if isinstance(dataset_obj, DatasetDict):
        if not split:
            return iter(())
        if split not in dataset_obj:
            available = ", ".join(dataset_obj.keys())
            raise ValueError(f"Split '{split}' not found. Available: {available}")
        ds: Dataset = dataset_obj[split]
    else:
        ds = dataset_obj
    return iter(ds)


def _iter_hf_rows_once(dataset_name: str, split: Optional[str], token: Optional[str]) -> Iterable[dict]:
    if not split:
        return iter(())
    return load_dataset(dataset_name, split=split, token=token, streaming=True)


def _resolve_hf_splits(
    *,
    dataset_name: str,
    token: Optional[str],
    requested_train_split: str,
    requested_val_split: str,
) -> tuple[str, Optional[str]]:
    try:
        names = list(get_dataset_split_names(dataset_name, token=token))
    except Exception:
        names = []

    train_split = requested_train_split or "train"
    if names and train_split not in names:
        train_split = "train" if "train" in names else names[0]

    if requested_val_split:
        if names and requested_val_split not in names:
            raise ValueError(f"--val-split '{requested_val_split}' not found in dataset splits: {names}")
        return train_split, requested_val_split

    for candidate in VAL_SPLIT_CANDIDATES:
        if candidate in names:
            return train_split, candidate
    return train_split, None


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


def _match_ious(predicted: list[DetectAnnotation], ground_truth: list[DetectAnnotation]) -> np.ndarray:
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


def _reward_miou(
    predicted: list[DetectAnnotation],
    ground_truth: list[DetectAnnotation],
    *,
    fn_penalty_exponent: float = 1.0,
    fp_penalty_exponent: float = 1.0,
    fn_iou_threshold: float = 0.5,
) -> float:
    if not predicted and not ground_truth:
        return 1.0
    if not predicted or not ground_truth:
        return 0.0
    matches = _match_ious(predicted, ground_truth)
    true_pos = int((matches >= fn_iou_threshold).sum())
    false_pos = len(predicted) - true_pos
    false_neg = len(ground_truth) - true_pos
    weighted_pred = float(true_pos) + (float(false_pos) ** float(fp_penalty_exponent))
    weighted_gt = float(true_pos) + (float(false_neg) ** float(fn_penalty_exponent))
    denom = max(weighted_pred, weighted_gt)
    return float(matches.sum()) / float(denom) if denom else 0.0


def _reward_f1(predicted: list[DetectAnnotation], ground_truth: list[DetectAnnotation]) -> float:
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


def _reward_f1_weighted(
    predicted: list[DetectAnnotation],
    ground_truth: list[DetectAnnotation],
    *,
    iou_threshold: float = 0.5,
    fn_penalty_exponent: float = 1.0,
    fp_penalty_exponent: float = 1.0,
) -> float:
    tp, fp, fn = _count_tp_fp_fn(predicted, ground_truth, iou_threshold=iou_threshold)
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0
    weighted_fp = float(fp) ** float(fp_penalty_exponent)
    weighted_fn = float(fn) ** float(fn_penalty_exponent)
    denom = (2.0 * float(tp)) + weighted_fp + weighted_fn
    if denom <= 0.0:
        return 0.0
    return (2.0 * float(tp)) / denom


def _count_tp_fp_fn(
    predicted: list[DetectAnnotation],
    ground_truth: list[DetectAnnotation],
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


def _point_in_box(point: PointAnnotation, box: DetectAnnotation) -> bool:
    return (box.x_min <= point.x <= box.x_max) and (box.y_min <= point.y <= box.y_max)


def _match_points_in_boxes(points: list[PointAnnotation], ground_truth: list[DetectAnnotation]) -> int:
    n_points = len(points)
    n_gt = len(ground_truth)
    if n_points == 0 or n_gt == 0:
        return 0
    size = max(n_points, n_gt)
    score = np.zeros((size, size), dtype=np.float32)
    for i, gt in enumerate(ground_truth):
        for j, pt in enumerate(points):
            score[i, j] = 1.0 if _point_in_box(pt, gt) else 0.0
    cost = -score
    row_idx, col_idx = linear_sum_assignment(cost)
    return int(score[row_idx, col_idx].sum())


def _count_tp_fp_fn_points(
    points: list[PointAnnotation],
    ground_truth: list[DetectAnnotation],
) -> tuple[int, int, int]:
    n_points = len(points)
    n_gt = len(ground_truth)
    if n_points == 0 and n_gt == 0:
        return 0, 0, 0
    if n_points == 0:
        return 0, 0, n_gt
    if n_gt == 0:
        return 0, n_points, 0
    tp = _match_points_in_boxes(points, ground_truth)
    fp = n_points - tp
    fn = n_gt - tp
    return tp, fp, fn


def _reward_f1_points(
    points: list[PointAnnotation],
    ground_truth: list[DetectAnnotation],
    *,
    fn_penalty_exponent: float = 1.0,
    fp_penalty_exponent: float = 1.0,
) -> float:
    tp, fp, fn = _count_tp_fp_fn_points(points, ground_truth)
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0
    weighted_fp = float(fp) ** float(fp_penalty_exponent)
    weighted_fn = float(fn) ** float(fn_penalty_exponent)
    denom = (2.0 * float(tp)) + weighted_fp + weighted_fn
    if denom <= 0.0:
        return 0.0
    return (2.0 * float(tp)) / denom


def _rewards_for_rollouts(
    rollouts: list[Rollout],
    gt_boxes: list[DetectAnnotation],
    *,
    skill: str = "detect",
    reward_metric: str = "f1",
    fn_penalty_exponent: float = 1.0,
    fp_penalty_exponent: float = 1.0,
    neg_reward_weight: float = 1.0,
) -> list[float]:
    rewards: list[float] = []
    is_negative_task = len(gt_boxes) == 0
    effective_skill = (skill or "detect").strip().lower()
    if effective_skill == "point" and reward_metric == "miou":
        print("warning: reward_metric='miou' is not defined for point outputs; using point F1 instead.")
        reward_metric = "f1"
    for rollout in rollouts:
        if effective_skill == "point":
            output = rollout.output
            pred_points = output.points if isinstance(output, PointOutput) else []
            reward = _reward_f1_points(
                pred_points,
                gt_boxes,
                fn_penalty_exponent=fn_penalty_exponent,
                fp_penalty_exponent=fp_penalty_exponent,
            )
        else:
            output = rollout.output
            pred_boxes = output.objects if isinstance(output, DetectOutput) else []
            if reward_metric == "miou":
                reward = _reward_miou(
                    pred_boxes,
                    gt_boxes,
                    fn_penalty_exponent=fn_penalty_exponent,
                    fp_penalty_exponent=fp_penalty_exponent,
                )
            else:
                reward = _reward_f1_weighted(
                    pred_boxes,
                    gt_boxes,
                    fn_penalty_exponent=fn_penalty_exponent,
                    fp_penalty_exponent=fp_penalty_exponent,
                )
        if is_negative_task:
            reward *= float(neg_reward_weight)
        rewards.append(float(reward))
    return rewards


def _is_rate_limit_error(exc: Exception) -> bool:
    if isinstance(exc, TunaAPIError) and exc.status_code == 429:
        return True
    return "too many requests" in str(exc).lower()


def _rollouts_batch_with_retry(
    *,
    finetune,
    requests: list[DetectRequest | PointRequest],
    num_rollouts: int,
    max_workers: int,
    retries: int,
    backoff_s: float,
    context: str,
):
    worker_count = max(1, min(max_workers, len(requests)))
    attempt = 0
    while True:
        try:
            return finetune.rollouts_batch(
                requests=requests,
                num_rollouts=num_rollouts,
                max_workers=worker_count,
            )
        except (TunaAPIError, TunaNetworkError) as exc:
            should_retry = isinstance(exc, TunaNetworkError) or _is_rate_limit_error(exc)
            if (not should_retry) or attempt >= retries:
                raise
            delay = max(0.1, backoff_s) * (2**attempt)
            print(
                f"{context}: retrying rollouts_batch attempt {attempt + 1}/{retries} "
                f"after {delay:.2f}s due to {type(exc).__name__}: {exc}"
            )
            time.sleep(delay)
            worker_count = max(1, worker_count // 2)
            attempt += 1


def _evaluate(
    *,
    finetune,
    eval_rows: Iterable[dict],
    all_class_names: list[str],
    rng: random.Random,
    neg_prompts_per_empty: int,
    neg_prompts_per_nonempty: int,
    max_samples: int,
    batch_size: int,
    max_workers: int,
    rollout_retries: int,
    rollout_retry_backoff_s: float,
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_objects: int,
    skill: str,
    point_prompt_style: str,
) -> dict[str, float]:
    tasks: list[TaskSample] = []
    total = 0
    total_f1 = 0.0
    total_miou = 0.0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    effective_skill = (skill or "detect").strip().lower()

    def _drain_batch(batch: list[TaskSample]) -> None:
        nonlocal total, total_f1, total_miou, total_tp, total_fp, total_fn
        if not batch:
            return
        chunk_size = max(1, min(max_workers, len(batch)))
        for offset in range(0, len(batch), chunk_size):
            chunk = batch[offset : offset + chunk_size]
            if effective_skill == "point":
                requests = [
                    PointRequest(
                        object_name=item.prompt,
                        image_url=_to_data_url(item.image, quality=92),
                        settings=PointSettings(
                            temperature=temperature,
                            top_p=top_p,
                            max_tokens=max_tokens,
                        ),
                    )
                    for item in chunk
                ]
            else:
                requests = [
                    DetectRequest(
                        object_name=item.prompt,
                        image_url=_to_data_url(item.image, quality=92),
                        settings=DetectSettings(
                            temperature=temperature,
                            top_p=top_p,
                            max_tokens=max_tokens,
                            max_objects=max_objects,
                        ),
                    )
                    for item in chunk
                ]
            try:
                results = _rollouts_batch_with_retry(
                    finetune=finetune,
                    requests=requests,
                    num_rollouts=1,
                    max_workers=min(max_workers, len(chunk)),
                    retries=rollout_retries,
                    backoff_s=rollout_retry_backoff_s,
                    context="eval",
                )
            except (TunaAPIError, TunaNetworkError) as exc:
                print(f"eval rollouts_batch failed: {exc}. skipping chunk")
                continue

            if len(results) != len(chunk):
                print(
                    f"warning: eval returned {len(results)} results for {len(chunk)} requests; "
                    "only aligned results are scored."
                )

            for item, result in zip(chunk, results):
                if not result.rollouts:
                    if effective_skill == "point":
                        pred_points: list[PointAnnotation] = []
                        total_f1 += _reward_f1_points(pred_points, item.gt_boxes)
                        tp, fp, fn = _count_tp_fp_fn_points(pred_points, item.gt_boxes)
                    else:
                        pred_boxes: list[DetectAnnotation] = []
                        total_f1 += _reward_f1(pred_boxes, item.gt_boxes)
                        total_miou += _reward_miou(pred_boxes, item.gt_boxes)
                        tp, fp, fn = _count_tp_fp_fn(pred_boxes, item.gt_boxes)
                    total_tp += tp
                    total_fp += fp
                    total_fn += fn
                    total += 1
                    continue

                rollout0 = result.rollouts[0]
                if effective_skill == "point":
                    output = rollout0.output
                    pred_points = output.points if isinstance(output, PointOutput) else []
                    total_f1 += _reward_f1_points(pred_points, item.gt_boxes)
                    tp, fp, fn = _count_tp_fp_fn_points(pred_points, item.gt_boxes)
                else:
                    output = rollout0.output
                    pred_boxes = output.objects if isinstance(output, DetectOutput) else []
                    total_f1 += _reward_f1(pred_boxes, item.gt_boxes)
                    total_miou += _reward_miou(pred_boxes, item.gt_boxes)
                    tp, fp, fn = _count_tp_fp_fn(pred_boxes, item.gt_boxes)
                total_tp += tp
                total_fp += fp
                total_fn += fn
                total += 1

    for row in eval_rows:
        base = _to_base_sample(row)
        if base is None:
            continue
        task_list = _tasks_from_base_sample(
            base,
            all_class_names=all_class_names,
            rng=rng,
            neg_prompts_per_empty=neg_prompts_per_empty,
            neg_prompts_per_nonempty=neg_prompts_per_nonempty,
            prompt_style=point_prompt_style,
        )
        for task in task_list:
            tasks.append(task)
            if len(tasks) >= batch_size:
                _drain_batch(tasks)
                tasks = []
            if max_samples and total >= max_samples:
                break
        if max_samples and total >= max_samples:
            break

    if tasks and (not max_samples or total < max_samples):
        _drain_batch(tasks)

    if total == 0:
        return {
            "eval_tasks": 0,
            "eval_f1": 0.0,
            "eval_f1_macro": 0.0,
            "eval_miou": 0.0,
            "eval_tp": 0,
            "eval_fp": 0,
            "eval_fn": 0,
        }

    micro_denom = 2 * total_tp + total_fp + total_fn
    micro_f1 = 1.0 if micro_denom == 0 else (2 * total_tp) / micro_denom
    return {
        "eval_tasks": total,
        "eval_f1": micro_f1,
        "eval_f1_macro": total_f1 / total,
        "eval_miou": total_miou / total,
        "eval_tp": total_tp,
        "eval_fp": total_fp,
        "eval_fn": total_fn,
    }


def _local_split_size(dataset_obj: Dataset | DatasetDict, split: Optional[str]) -> Optional[int]:
    if isinstance(dataset_obj, DatasetDict):
        if not split or split not in dataset_obj:
            return None
        return len(dataset_obj[split])
    return len(dataset_obj)


def _counter_top(counter: Counter[str], top_k: int) -> list[tuple[str, int]]:
    if top_k <= 0:
        return []
    return counter.most_common(top_k)


def _usage_snapshot(
    usage: UsageStats,
    *,
    total_train_rows: Optional[int],
    top_k: int,
) -> dict[str, Any]:
    row_coverage = 0.0
    if total_train_rows and total_train_rows > 0:
        row_coverage = usage.rows_seen / float(total_train_rows)

    return {
        "rows_seen": usage.rows_seen,
        "rows_with_boxes": usage.rows_with_boxes,
        "rows_without_boxes": usage.rows_without_boxes,
        "tasks_generated": usage.tasks_generated,
        "tasks_generated_positive": usage.tasks_generated_positive,
        "tasks_generated_negative": usage.tasks_generated_negative,
        "tasks_consumed": usage.tasks_consumed,
        "tasks_consumed_positive": usage.tasks_consumed_positive,
        "tasks_consumed_negative": usage.tasks_consumed_negative,
        "rows_seen_fraction": row_coverage,
        "source_rows_top": _counter_top(usage.source_rows_seen, top_k),
        "source_tasks_generated_top": _counter_top(usage.source_tasks_generated, top_k),
        "source_tasks_consumed_top": _counter_top(usage.source_tasks_consumed, top_k),
        "class_tasks_generated_top": _counter_top(usage.class_tasks_generated, top_k),
        "class_tasks_consumed_top": _counter_top(usage.class_tasks_consumed, top_k),
    }


def _print_usage_snapshot(snapshot: dict[str, Any], *, prefix: str) -> None:
    rows_seen = int(snapshot.get("rows_seen", 0))
    rows_seen_fraction = float(snapshot.get("rows_seen_fraction", 0.0)) * 100.0
    tasks_generated = int(snapshot.get("tasks_generated", 0))
    tasks_consumed = int(snapshot.get("tasks_consumed", 0))
    print(
        f"{prefix} rows_seen={rows_seen} ({rows_seen_fraction:.2f}%) "
        f"tasks_generated={tasks_generated} tasks_consumed={tasks_consumed}"
    )
    print(f"{prefix} source_rows_top={snapshot.get('source_rows_top', [])}")
    print(f"{prefix} source_tasks_consumed_top={snapshot.get('source_tasks_consumed_top', [])}")
    print(f"{prefix} class_tasks_consumed_top={snapshot.get('class_tasks_consumed_top', [])}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RL finetune Moondream for PI&D icon detection (class-conditional).")
    parser.add_argument("--env-file", default=str(_repo_relative(".env")))
    parser.add_argument("--api-key", default=os.environ.get("MOONDREAM_API_KEY"))
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"))
    parser.add_argument("--base-url", default=os.environ.get("TUNA_BASE_URL", "https://api.moondream.ai/v1"))

    parser.add_argument("--dataset-path", default="")
    parser.add_argument("--dataset-name", default="maxs-m87/pid-icons-merged" )
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--val-split",
        default="",
        help="Validation split name. If omitted, uses dataset validation/val/dev/test/post_val when present; otherwise auto-splits.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--class-names-file", default="")
    parser.add_argument(
        "--allow-duplicate-class-names",
        action="store_true",
        help="Deprecated no-op: duplicate class_name values now warn and do not fail.",
    )
    parser.add_argument(
        "--fail-on-numeric-class-names",
        action="store_true",
        help="Fail if any class name contains digits (0-9).",
    )

    parser.add_argument("--finetune-id", default="")
    parser.add_argument("--finetune-name", default="")
    parser.add_argument("--rank", type=int, default=16)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--buffer-size", type=int, default=100)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--resume-step", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--group-size", type=int, default=6)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--rollout-retries", type=int, default=2)
    parser.add_argument("--rollout-retry-backoff-s", type=float, default=1.0)

    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-objects", type=int, default=50)
    parser.add_argument(
        "--skill",
        choices=["detect", "point"],
        default="detect",
        help="Which Moondream vision skill to train/eval against. In 'point' mode, a prediction counts as correct if the returned point lies inside any GT bbox.",
    )
    parser.add_argument(
        "--reward-metric",
        choices=["f1", "miou"],
        default="f1",
        help="Training reward metric for rollouts. Use 'f1' to focus optimization on detection F1.",
    )
    parser.add_argument(
        "--point-prompt-style",
        choices=["detect_phrase", "class_name"],
        default="detect_phrase",
        help="Prompt style for class-conditional point training/eval tasks.",
    )

    parser.add_argument("--off-policy", action="store_true")
    parser.add_argument("--off-policy-std-thresh", type=float, default=0.02)
    parser.add_argument("--off-policy-max-reward", type=float, default=0.15)
    parser.add_argument("--off-policy-min-reward", type=float, default=0.15)
    parser.add_argument("--off-policy-reward-scale", type=float, default=2.0)
    parser.add_argument(
        "--fn-penalty-exponent",
        type=float,
        default=1.0,
        help="Exponent for false negatives in reward denominator via FN^exp; >1.0 penalizes missed GT boxes more.",
    )
    parser.add_argument(
        "--fp-penalty-exponent",
        type=float,
        default=1.0,
        help="Exponent for false positives in reward denominator via FP^exp; >1.0 penalizes extra detections more.",
    )

    parser.add_argument("--neg-prompts-per-empty", type=int, default=1)
    parser.add_argument("--neg-prompts-per-nonempty", type=int, default=0)
    parser.add_argument(
        "--pos-task-prob",
        type=float,
        default=0.95,
        help=(
            "When a sampled base image has positive tasks, choose a positive task with this probability "
            "(otherwise sample a negative task from that same image when available)."
        ),
    )
    parser.add_argument(
        "--neg-reward-weight",
        type=float,
        default=0.5,
        help="Scale factor applied to rewards for negative tasks (no GT boxes). Range: (0, 1].",
    )
    parser.add_argument(
        "--use-recall-first-preset",
        action="store_true",
        help=(
            "Apply recall-first point defaults: lr=5e-4, pos_task_prob=0.995, "
            "neg_prompts_per_nonempty=0, neg_prompts_per_empty=1, neg_reward_weight=0.15, "
            "fn_penalty_exponent=2.0, fp_penalty_exponent=1.0." 
        ),
    )
    parser.add_argument(
        "--recall-gate-step",
        type=int,
        default=40,
        help="First eval step at/after which recall TP gate is checked.",
    )
    parser.add_argument(
        "--recall-drop-threshold",
        type=float,
        default=0.25,
        help="Maximum allowed TP drop vs baseline for recall gate. 0.25 means <=25%% drop.",
    )
    parser.add_argument(
        "--f1-improvement-target",
        type=float,
        default=0.01,
        help="Required best eval_f1 improvement over baseline eval_f1.",
    )
    parser.add_argument("--augment-prob", type=float, default=0.5)
    parser.add_argument("--eval-temperature", type=float, default=0.0)
    parser.add_argument("--eval-top-p", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-max-samples", type=int, default=200)
    parser.add_argument("--eval-batch-size", type=int, default=20)

    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument(
        "--usage-report-every",
        type=int,
        default=30,
        help="Print running dataset/task usage stats every N steps (<=0 disables periodic usage logs).",
    )
    parser.add_argument(
        "--usage-top-k",
        type=int,
        default=30,
        help="How many sources/classes to include in usage distribution summaries.",
    )
    parser.add_argument("--wandb-project", default="moondream-pid-icons-rl")
    parser.add_argument("--wandb-run-name", default="")
    args = parser.parse_args()

    if args.use_recall_first_preset:
        args.lr = 5e-4
        args.pos_task_prob = 0.995
        args.neg_prompts_per_nonempty = 0
        args.neg_prompts_per_empty = 1
        args.neg_reward_weight = 0.15
        args.fn_penalty_exponent = 2.0
        args.fp_penalty_exponent = 1.0
        print(
            "applied recall-first preset: "
            "lr=5e-4 pos_task_prob=0.995 neg_prompts_per_nonempty=0 "
            "neg_prompts_per_empty=1 neg_reward_weight=0.15 fn_exp=2.0 fp_exp=1.0"
        )

    load_dotenv(args.env_file, override=False)
    if not args.api_key:
        args.api_key = os.environ.get("MOONDREAM_API_KEY")
    if not args.hf_token:
        args.hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not args.base_url:
        args.base_url = os.environ.get("TUNA_BASE_URL", "https://api.moondream.ai/v1")

    if not args.api_key:
        raise ValueError("MOONDREAM_API_KEY is required")
    if args.neg_prompts_per_empty < 0 or args.neg_prompts_per_nonempty < 0:
        raise ValueError("negative prompt counts must be >= 0")
    if not (0.0 <= args.pos_task_prob <= 1.0):
        raise ValueError("--pos-task-prob must be in [0, 1]")
    if args.neg_reward_weight <= 0.0:
        raise ValueError("--neg-reward-weight must be > 0")
    if args.neg_reward_weight > 1.0:
        print("warning: --neg-reward-weight > 1; clamping to 1.0")
        args.neg_reward_weight = 1.0
    if not (0.0 <= args.augment_prob <= 1.0):
        raise ValueError("--augment-prob must be in [0, 1]")
    if not (0.0 < args.val_fraction < 1.0):
        raise ValueError("--val-fraction must be in (0, 1)")
    if args.usage_top_k < 0:
        raise ValueError("--usage-top-k must be >= 0")
    if args.off_policy_min_reward <= 0.0:
        raise ValueError("--off-policy-min-reward must be > 0")
    if args.off_policy_reward_scale <= 0.0:
        raise ValueError("--off-policy-reward-scale must be > 0")
    if args.fn_penalty_exponent < 1.0:
        raise ValueError("--fn-penalty-exponent must be >= 1.0")
    if args.fp_penalty_exponent < 1.0:
        raise ValueError("--fp-penalty-exponent must be >= 1.0")
    if args.rollout_retries < 0:
        raise ValueError("--rollout-retries must be >= 0")
    if args.rollout_retry_backoff_s <= 0.0:
        raise ValueError("--rollout-retry-backoff-s must be > 0")
    if args.recall_gate_step < 0:
        raise ValueError("--recall-gate-step must be >= 0")
    if not (0.0 <= args.recall_drop_threshold < 1.0):
        raise ValueError("--recall-drop-threshold must be in [0, 1)")
    if args.f1_improvement_target < 0.0:
        raise ValueError("--f1-improvement-target must be >= 0")

    dataset_path = args.dataset_path.strip()
    dataset_name = args.dataset_name.strip()
    use_local = bool(dataset_path)

    all_class_names = _load_class_names(args.class_names_file, dataset_path if use_local else None)
    class_catalog = _load_class_catalog(args.class_names_file, dataset_path if use_local else None)

    rng = random.Random(args.seed)
    rng_np = np.random.default_rng(args.seed)
    augment_config = _default_augment_config()
    usage = UsageStats()
    total_train_rows: Optional[int] = None
    total_val_rows: Optional[int] = None
    discovery_train_rows_consumed = 0

    eval_rows_factory: Callable[[], Iterable[dict]] = lambda: iter(())
    has_eval_rows = False
    train_split = args.split
    requested_val_split = args.val_split.strip()
    val_split: Optional[str] = requested_val_split or None
    auto_val_split = False
    eval_dataset: Optional[Dataset] = None

    if use_local:
        dataset_obj = load_from_disk(dataset_path)
        if isinstance(dataset_obj, DatasetDict):
            train_split = args.split if args.split in dataset_obj else ("train" if "train" in dataset_obj else args.split)
            if requested_val_split:
                if requested_val_split not in dataset_obj:
                    raise ValueError(
                        f"--val-split '{requested_val_split}' not found in local dataset splits: {list(dataset_obj)}"
                    )
                val_split = requested_val_split
            else:
                val_split = "val" if "val" in dataset_obj else ("validation" if "validation" in dataset_obj else None)
        else:
            train_split = args.split
            val_split = requested_val_split or None

        if not val_split:
            full_ds = _load_local_split(dataset_path, train_split)
            split_ds = full_ds.train_test_split(test_size=args.val_fraction, seed=args.seed, shuffle=True)
            train_ds = split_ds["train"]
            eval_dataset = split_ds["test"]
            train_row_iter = _iter_dataset_rows(train_ds, args.seed)
            total_train_rows = len(train_ds)
            total_val_rows = len(eval_dataset)
            val_split = f"auto({train_split})"
            auto_val_split = True
            has_eval_rows = True

            def _local_auto_eval_rows() -> Iterable[dict]:
                return iter(eval_dataset) if eval_dataset is not None else iter(())

            eval_rows_factory = _local_auto_eval_rows
        else:
            train_ds = _load_local_split(dataset_path, train_split)
            train_row_iter = _iter_dataset_rows(train_ds, args.seed)
            total_train_rows = len(train_ds)
            total_val_rows = _local_split_size(dataset_obj, val_split)
            has_eval_rows = True

            def _local_eval_rows() -> Iterable[dict]:
                return _iter_local_rows_once(dataset_obj, val_split)

            eval_rows_factory = _local_eval_rows
    else:
        if not dataset_name:
            raise ValueError("Provide --dataset-path or --dataset-name")

        train_split, resolved_val = _resolve_hf_splits(
            dataset_name=dataset_name,
            token=args.hf_token,
            requested_train_split=args.split,
            requested_val_split=requested_val_split,
        )
        if resolved_val is None:
            full_ds = load_dataset(dataset_name, split=train_split, token=args.hf_token, streaming=False)
            split_ds = full_ds.train_test_split(test_size=args.val_fraction, seed=args.seed, shuffle=True)
            train_ds = split_ds["train"]
            eval_dataset = split_ds["test"]
            train_row_iter = _iter_dataset_rows(train_ds, args.seed)
            total_train_rows = len(train_ds)
            total_val_rows = len(eval_dataset)
            val_split = f"auto({train_split})"
            auto_val_split = True
            has_eval_rows = True

            def _hf_auto_eval_rows() -> Iterable[dict]:
                return iter(eval_dataset) if eval_dataset is not None else iter(())

            eval_rows_factory = _hf_auto_eval_rows
        else:
            val_split = resolved_val
            train_row_iter = _iter_hf_rows(dataset_name, train_split, args.hf_token, args.seed, args.buffer_size)
            has_eval_rows = True

            def _hf_eval_rows() -> Iterable[dict]:
                return _iter_hf_rows_once(dataset_name, val_split, args.hf_token)

            eval_rows_factory = _hf_eval_rows

    if not all_class_names:
        # Fallback: scan eval rows first, then train stream as needed.
        discovered: set[str] = set()
        if has_eval_rows:
            for row in itertools.islice(eval_rows_factory(), 5000):
                base = _to_base_sample(row)
                if not base:
                    continue
                for item in base.boxes:
                    if item.class_name:
                        discovered.add(item.class_name)
        if not discovered:
            for _ in range(2000):
                row = next(train_row_iter)
                discovery_train_rows_consumed += 1
                base = _to_base_sample(row)
                if not base:
                    continue
                for item in base.boxes:
                    if item.class_name:
                        discovered.add(item.class_name)
        all_class_names = sorted(discovered)

    if not all_class_names:
        raise ValueError("Could not resolve class names for prompting.")

    if not class_catalog:
        class_catalog = [(name, name) for name in all_class_names]
    duplicate_class_names, numeric_class_names = _analyze_class_catalog(class_catalog)

    if duplicate_class_names:
        duplicate_preview = "; ".join(
            f'"{class_name}" -> {uids}'
            for class_name, uids in list(sorted(duplicate_class_names.items()))[:10]
        )
        duplicate_msg = (
            "duplicate class names mapped to multiple class_uids found. "
            "This can create conflicting rewards for class-conditional prompts. "
            f"Examples: {duplicate_preview}"
        )
        print(f"warning: {duplicate_msg}")

    if numeric_class_names:
        numeric_preview = ", ".join(f'"{name}"' for name in numeric_class_names[:15])
        numeric_msg = (
            "class names with digits detected. This may indicate label parsing issues. "
            f"Examples: {numeric_preview}"
        )
        if args.fail_on_numeric_class_names:
            raise ValueError(numeric_msg)
        print(f"warning: {numeric_msg}")

    if not args.finetune_id and not args.finetune_name:
        args.finetune_name = f"pid-icons-{args.skill}-{_random_suffix()}"

    expected_tasks = args.num_steps * args.batch_size
    if total_train_rows is not None:
        print(
            f"dataset usage plan: train_rows_total={total_train_rows} "
            f"requested_steps={args.num_steps} batch_size={args.batch_size} "
            f"expected_tasks_consumed={expected_tasks}"
        )
    else:
        print(
            "dataset usage plan: train_rows_total=unknown (streaming) "
            f"requested_steps={args.num_steps} batch_size={args.batch_size} "
            f"expected_tasks_consumed={expected_tasks}"
        )
    if total_val_rows is not None:
        print(f"dataset usage plan: val_rows_total={total_val_rows}")
    if discovery_train_rows_consumed > 0:
        print(f"dataset usage note: consumed {discovery_train_rows_consumed} training rows for class discovery")
    print(
        "run control: "
        f"num_steps={args.num_steps} resume_step={args.resume_step} "
        f"eval_every={args.eval_every} save_every={args.save_every} "
        f"off_policy={args.off_policy}"
    )
    effective_point_prompt_style = args.point_prompt_style if args.skill == "point" else "detect_phrase"
    if args.skill != "point" and args.point_prompt_style != "detect_phrase":
        print("warning: --point-prompt-style is only applied when --skill=point; using detect_phrase.")

    client = TunaClient(api_key=args.api_key, base_url=args.base_url)
    if args.finetune_id:
        finetune = client.get_finetune(args.finetune_id)
    else:
        finetune = client.create_finetune(name=args.finetune_name, rank=args.rank)

    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or None,
        config={
            "finetune_id": finetune.finetune_id,
            "dataset_path": dataset_path or None,
            "dataset_name": dataset_name or None,
            "train_split": train_split,
            "val_split": val_split,
            "auto_val_split": auto_val_split,
            "auto_val_fraction": args.val_fraction if auto_val_split else None,
            "train_rows_total": total_train_rows,
            "val_rows_total": total_val_rows,
            "expected_tasks_consumed": expected_tasks,
            "class_discovery_train_rows_consumed": discovery_train_rows_consumed,
            "class_count": len(all_class_names),
            "neg_prompts_per_empty": args.neg_prompts_per_empty,
            "neg_prompts_per_nonempty": args.neg_prompts_per_nonempty,
            "augment_prob": args.augment_prob,
            "num_steps": args.num_steps,
            "resume_step": args.resume_step,
            "batch_size": args.batch_size,
            "group_size": args.group_size,
            "lr": args.lr,
            "max_workers": args.max_workers,
            "rollout_retries": args.rollout_retries,
            "rollout_retry_backoff_s": args.rollout_retry_backoff_s,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "eval_temperature": args.eval_temperature,
            "eval_top_p": args.eval_top_p,
            "max_tokens": args.max_tokens,
            "max_objects": args.max_objects,
            "skill": args.skill,
            "point_prompt_style": args.point_prompt_style,
            "effective_point_prompt_style": effective_point_prompt_style,
            "reward_metric": args.reward_metric,
            "seed": args.seed,
            "off_policy": args.off_policy,
            "off_policy_std_thresh": args.off_policy_std_thresh,
            "off_policy_max_reward": args.off_policy_max_reward,
            "off_policy_min_reward": args.off_policy_min_reward,
            "off_policy_reward_scale": args.off_policy_reward_scale,
            "fn_penalty_exponent": args.fn_penalty_exponent,
            "fp_penalty_exponent": args.fp_penalty_exponent,
            "pos_task_prob": args.pos_task_prob,
            "neg_reward_weight": args.neg_reward_weight,
            "use_recall_first_preset": args.use_recall_first_preset,
            "recall_gate_step": args.recall_gate_step,
            "recall_drop_threshold": args.recall_drop_threshold,
            "f1_improvement_target": args.f1_improvement_target,
            "allow_duplicate_class_names": args.allow_duplicate_class_names,
            "fail_on_numeric_class_names": args.fail_on_numeric_class_names,
            "duplicate_class_name_count": len(duplicate_class_names),
            "numeric_class_name_count": len(numeric_class_names),
            "usage_report_every": args.usage_report_every,
            "usage_top_k": args.usage_top_k,
        },
    )

    def _next_task() -> TaskSample:
        while True:
            row = next(train_row_iter)
            base = _to_base_sample(row)
            if base is None:
                continue
            usage.rows_seen += 1
            usage.source_rows_seen[base.source] += 1
            if base.boxes:
                usage.rows_with_boxes += 1
            else:
                usage.rows_without_boxes += 1
            base = _augment_base_sample(
                base,
                rng,
                rng_np,
                augment_config,
                augment_prob=args.augment_prob,
            )
            new_tasks = _tasks_from_base_sample(
                base,
                all_class_names=all_class_names,
                rng=rng,
                neg_prompts_per_empty=args.neg_prompts_per_empty,
                neg_prompts_per_nonempty=args.neg_prompts_per_nonempty,
                prompt_style=effective_point_prompt_style,
            )
            if not new_tasks:
                continue
            usage.tasks_generated += len(new_tasks)
            for task in new_tasks:
                usage.source_tasks_generated[task.source] += 1
                usage.class_tasks_generated[task.class_name] += 1
                if task.is_positive:
                    usage.tasks_generated_positive += 1
                else:
                    usage.tasks_generated_negative += 1
            positives = [task for task in new_tasks if task.is_positive]
            negatives = [task for task in new_tasks if not task.is_positive]
            if positives:
                if rng.random() < float(args.pos_task_prob):
                    selected_task = rng.choice(positives)
                elif negatives:
                    selected_task = rng.choice(negatives)
                else:
                    selected_task = rng.choice(positives)
            else:
                selected_task = rng.choice(negatives)
            usage.tasks_consumed += 1
            usage.source_tasks_consumed[selected_task.source] += 1
            usage.class_tasks_consumed[selected_task.class_name] += 1
            if selected_task.is_positive:
                usage.tasks_consumed_positive += 1
            else:
                usage.tasks_consumed_negative += 1
            return selected_task

    best_metric: Optional[float] = None
    best_step: Optional[int] = None
    successful_updates = args.resume_step
    baseline_eval_metric: Optional[float] = None
    baseline_eval_tp: Optional[float] = None
    recall_gate_pass: Optional[bool] = None
    recall_gate_eval_step: Optional[int] = None
    recall_gate_eval_tp: Optional[float] = None
    recall_gate_min_tp: Optional[float] = None
    eval_events_logged = 0

    def _run_and_log_eval(*, trigger: str, step_for_log: int) -> Optional[dict[str, float]]:
        nonlocal eval_events_logged
        eval_events_logged += 1
        event_payload: dict[str, Any] = {
            "eval_event_index": eval_events_logged,
            "eval_event_trigger": trigger,
            "eval_event_step": step_for_log,
        }
        try:
            eval_rng = random.Random(args.seed + 12345)
            eval_metrics = _evaluate(
                finetune=finetune,
                eval_rows=eval_rows_factory(),
                all_class_names=all_class_names,
                rng=eval_rng,
                neg_prompts_per_empty=args.neg_prompts_per_empty,
                neg_prompts_per_nonempty=args.neg_prompts_per_nonempty,
                max_samples=args.eval_max_samples,
                batch_size=args.eval_batch_size,
                max_workers=args.max_workers,
                rollout_retries=args.rollout_retries,
                rollout_retry_backoff_s=args.rollout_retry_backoff_s,
                temperature=args.eval_temperature,
                top_p=args.eval_top_p,
                max_tokens=args.max_tokens,
                max_objects=args.max_objects,
                skill=args.skill,
                point_prompt_style=effective_point_prompt_style,
            )
            event_payload.update(eval_metrics)
            event_payload["eval_event_success"] = 1
            if trigger == "baseline":
                event_payload.update({f"baseline_{key}": value for key, value in eval_metrics.items()})
            wandb.log(event_payload, step=step_for_log)
            return eval_metrics
        except Exception as exc:
            event_payload["eval_event_success"] = 0
            event_payload["eval_event_error"] = f"{type(exc).__name__}: {exc}"
            wandb.log(event_payload, step=step_for_log)
            print(f"eval {trigger} failed at step {step_for_log}: {type(exc).__name__}: {exc}")
            return None

    if args.eval_every > 0 and has_eval_rows:
        baseline_step = max(0, args.resume_step - 1)
        print("running baseline eval before training...")
        baseline_metrics = _run_and_log_eval(trigger="baseline", step_for_log=baseline_step)
        if baseline_metrics is not None:
            metric_key = "eval_miou" if args.skill == "detect" else "eval_f1"
            baseline_eval_metric = float(baseline_metrics.get(metric_key, 0.0))
            baseline_eval_tp = float(baseline_metrics.get("eval_tp", 0.0))
            run.summary[f"baseline_{metric_key}"] = baseline_eval_metric
            run.summary["baseline_eval_tp"] = baseline_eval_tp
            run.summary["baseline_metric_key"] = metric_key
            print(
                f"baseline eval step {baseline_step} tasks={baseline_metrics['eval_tasks']} "
                f"miou={baseline_metrics['eval_miou']:.4f} f1={baseline_metrics['eval_f1']:.4f} "
                f"macro_f1={baseline_metrics['eval_f1_macro']:.4f}"
            )

    for step in range(args.num_steps):
        global_step = args.resume_step + step
        step_start = time.monotonic()

        batch = [_next_task() for _ in range(args.batch_size)]

        if args.skill == "point":
            requests = [
                PointRequest(
                    object_name=item.prompt,
                    image_url=_to_data_url(item.image, quality=92),
                    settings=PointSettings(
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_tokens=args.max_tokens,
                    ),
                )
                for item in batch
            ]
        else:
            requests = [
                DetectRequest(
                    object_name=item.prompt,
                    image_url=_to_data_url(item.image, quality=92),
                    settings=DetectSettings(
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_tokens=args.max_tokens,
                        max_objects=args.max_objects,
                    ),
                )
                for item in batch
            ]

        try:
            rollout_start = time.monotonic()
            results = _rollouts_batch_with_retry(
                finetune=finetune,
                requests=requests,
                num_rollouts=args.group_size,
                max_workers=min(args.max_workers, args.batch_size),
                retries=args.rollout_retries,
                backoff_s=args.rollout_retry_backoff_s,
                context=f"train step {global_step}",
            )
            rollout_end = time.monotonic()
        except (TunaAPIError, TunaNetworkError) as exc:
            print(f"rollouts_batch failed at step {global_step}: {exc}. skipping step")
            continue

        groups: list[TrainStepGroup] = []
        all_rewards: list[float] = []
        off_policy_injected_total = 0
        off_policy_injected_positive = 0
        off_policy_injected_negative = 0
        off_policy_considered = 0
        off_policy_trigger_low_max = 0
        off_policy_trigger_low_mean = 0
        off_policy_skipped_high_reward = 0
        off_policy_skipped_high_variance = 0
        train_tp = 0
        train_fp = 0
        train_fn = 0
        for item, result in zip(batch, results):
            rollouts = list(result.rollouts)
            rewards = _rewards_for_rollouts(
                rollouts,
                item.gt_boxes,
                skill=args.skill,
                reward_metric=args.reward_metric,
                fn_penalty_exponent=args.fn_penalty_exponent,
                fp_penalty_exponent=args.fp_penalty_exponent,
                neg_reward_weight=args.neg_reward_weight,
            )

            if args.off_policy and rewards and rollouts:
                off_policy_considered += 1
                mean_reward = sum(rewards) / len(rewards)
                reward_var = sum((value - mean_reward) ** 2 for value in rewards) / len(rewards)
                reward_std = reward_var**0.5
                max_reward = max(rewards)
                low_max_reward = max_reward < args.off_policy_max_reward
                low_mean_reward = mean_reward < args.off_policy_max_reward
                should_inject = low_max_reward or (
                    low_mean_reward and reward_std < args.off_policy_std_thresh
                )

                if should_inject:
                    if low_max_reward:
                        off_policy_trigger_low_max += 1
                    else:
                        off_policy_trigger_low_mean += 1
                    replace_idx = int(np.argmin(np.asarray(rewards, dtype=np.float32)))
                    old_rollout = rollouts[replace_idx]
                    replacement_objects = list(item.gt_boxes)
                    replacement_points = [
                        PointAnnotation(x=(box.x_min + box.x_max) / 2.0, y=(box.y_min + box.y_max) / 2.0)
                        for box in replacement_objects
                    ]
                    rollouts[replace_idx] = Rollout(
                        skill=old_rollout.skill,
                        finish_reason=old_rollout.finish_reason,
                        output=(
                            PointOutput(points=replacement_points)
                            if args.skill == "point"
                            else DetectOutput(objects=replacement_objects)
                        ),
                        answer_tokens=list(old_rollout.answer_tokens),
                        thinking_tokens=list(old_rollout.thinking_tokens),
                        coords=list(old_rollout.coords),
                        sizes=list(old_rollout.sizes),
                    )
                    reward_anchor = max(float(max_reward), float(mean_reward))
                    injected_reward = max(
                        float(args.off_policy_min_reward),
                        min(1.0, float(args.off_policy_reward_scale) * reward_anchor),
                    )
                    if not item.gt_boxes:
                        injected_reward *= float(args.neg_reward_weight)
                    rewards[replace_idx] = injected_reward
                    off_policy_injected_total += 1
                    if item.is_positive:
                        off_policy_injected_positive += 1
                    else:
                        off_policy_injected_negative += 1
                else:
                    if max_reward >= args.off_policy_max_reward:
                        off_policy_skipped_high_reward += 1
                    elif reward_std >= args.off_policy_std_thresh:
                        off_policy_skipped_high_variance += 1

            if args.skill == "point":
                for rollout in rollouts:
                    output = rollout.output
                    pred_points = output.points if isinstance(output, PointOutput) else []
                    tp, fp, fn = _count_tp_fp_fn_points(pred_points, item.gt_boxes)
                    train_tp += tp
                    train_fp += fp
                    train_fn += fn
            else:
                for rollout in rollouts:
                    output = rollout.output
                    pred_boxes = output.objects if isinstance(output, DetectOutput) else []
                    tp, fp, fn = _count_tp_fp_fn(pred_boxes, item.gt_boxes)
                    train_tp += tp
                    train_fp += fp
                    train_fn += fn

            groups.append(TrainStepGroup(request=result.request, rollouts=rollouts, rewards=rewards))
            all_rewards.extend(rewards)

        try:
            train_start = time.monotonic()
            train_out = finetune.train_step(groups=groups, lr=args.lr)
            train_end = time.monotonic()
        except (TunaAPIError, TunaNetworkError) as exc:
            print(f"train_step failed at step {global_step}: {exc}. skipping step")
            continue
        successful_updates += 1

        reward_mean = float(np.mean(all_rewards)) if all_rewards else 0.0
        reward_var = float(np.var(all_rewards)) if all_rewards else 0.0
        pos_tasks = sum(1 for item in batch if item.is_positive)
        neg_tasks = len(batch) - pos_tasks
        precision_denom = train_tp + train_fp
        recall_denom = train_tp + train_fn
        micro_denom = (2 * train_tp) + train_fp + train_fn
        train_precision = 1.0 if precision_denom == 0 else train_tp / precision_denom
        train_recall = 1.0 if recall_denom == 0 else train_tp / recall_denom
        train_f1 = 1.0 if micro_denom == 0 else (2 * train_tp) / micro_denom
        rows_seen_fraction = (
            usage.rows_seen / float(total_train_rows)
            if total_train_rows and total_train_rows > 0
            else 0.0
        )

        wandb.log(
            {
                "reward_mean": reward_mean,
                "reward_var": reward_var,
                "batch_positive_tasks": pos_tasks,
                "batch_negative_tasks": neg_tasks,
                "train_tp": train_tp,
                "train_fp": train_fp,
                "train_fn": train_fn,
                "train_precision": train_precision,
                "train_recall": train_recall,
                "train_f1": train_f1,
                "accepted_groups": len(groups),
                "off_policy_injected": off_policy_injected_total,
                "off_policy_injected_positive": off_policy_injected_positive,
                "off_policy_injected_negative": off_policy_injected_negative,
                "off_policy_considered": off_policy_considered,
                "off_policy_trigger_low_max": off_policy_trigger_low_max,
                "off_policy_trigger_low_mean": off_policy_trigger_low_mean,
                "off_policy_skipped_high_reward": off_policy_skipped_high_reward,
                "off_policy_skipped_high_variance": off_policy_skipped_high_variance,
                "kl": train_out.kl if train_out.kl is not None else 0.0,
                "router_kl": train_out.router_kl if train_out.router_kl is not None else 0.0,
                "grad_norm": train_out.grad_norm if train_out.grad_norm is not None else 0.0,
                "rows_seen": usage.rows_seen,
                "rows_seen_fraction": rows_seen_fraction,
                "tasks_generated_total": usage.tasks_generated,
                "tasks_generated_positive_total": usage.tasks_generated_positive,
                "tasks_generated_negative_total": usage.tasks_generated_negative,
                "tasks_consumed_total": usage.tasks_consumed,
                "tasks_consumed_positive_total": usage.tasks_consumed_positive,
                "tasks_consumed_negative_total": usage.tasks_consumed_negative,
            },
            step=global_step,
        )

        total_s = time.monotonic() - step_start
        print(
            f"step {global_step} reward={reward_mean:.4f} kl={float(train_out.kl or 0.0):.4f} "
            f"train_p={train_precision:.3f} train_r={train_recall:.3f} "
            f"pos={pos_tasks} neg={neg_tasks} rollout_s={(rollout_end-rollout_start):.2f} "
            f"train_s={(train_end-train_start):.2f} total_s={total_s:.2f} "
            f"offp={off_policy_injected_total}/{off_policy_considered} updates={successful_updates}"
        )
        if args.usage_report_every > 0 and (global_step + 1) % args.usage_report_every == 0:
            usage_summary = _usage_snapshot(usage, total_train_rows=total_train_rows, top_k=args.usage_top_k)
            _print_usage_snapshot(usage_summary, prefix=f"usage step {global_step}")

        if args.eval_every > 0 and has_eval_rows and successful_updates % args.eval_every == 0:
            eval_metrics = _run_and_log_eval(trigger="periodic", step_for_log=global_step)
            if eval_metrics is None:
                continue
            metric_key = "eval_miou" if args.skill == "detect" else "eval_f1"
            if baseline_eval_metric is not None:
                delta_key = f"{metric_key}_delta_vs_baseline"
                eval_metrics[delta_key] = float(eval_metrics.get(metric_key, 0.0)) - baseline_eval_metric
                wandb.log({delta_key: eval_metrics[delta_key]}, step=global_step)
            if (
                args.skill == "point"
                and baseline_eval_tp is not None
                and recall_gate_pass is None
                and global_step >= args.recall_gate_step
            ):
                recall_gate_eval_step = global_step
                recall_gate_eval_tp = float(eval_metrics.get("eval_tp", 0.0))
                recall_gate_min_tp = float(baseline_eval_tp) * (1.0 - float(args.recall_drop_threshold))
                recall_gate_pass = bool(recall_gate_eval_tp >= recall_gate_min_tp)
                wandb.log(
                    {
                        "recall_gate_step": recall_gate_eval_step,
                        "recall_gate_eval_tp": recall_gate_eval_tp,
                        "recall_gate_min_tp": recall_gate_min_tp,
                        "recall_gate_pass": int(recall_gate_pass),
                    },
                    step=global_step,
                )
                print(
                    f"recall gate step {recall_gate_eval_step}: "
                    f"tp={recall_gate_eval_tp:.1f} min_tp={recall_gate_min_tp:.1f} "
                    f"pass={recall_gate_pass}"
                )
            print(
                f"eval step {global_step} tasks={eval_metrics['eval_tasks']} "
                f"miou={eval_metrics['eval_miou']:.4f} f1={eval_metrics['eval_f1']:.4f} "
                f"macro_f1={eval_metrics['eval_f1_macro']:.4f} "
                f"updates={successful_updates}"
            )

            metric = float(eval_metrics.get(metric_key, 0.0))
            if best_metric is None or metric > best_metric:
                best_metric = metric
                best_step = global_step
                finetune.save_checkpoint()

        if args.save_every > 0 and (global_step + 1) % args.save_every == 0:
            finetune.save_checkpoint()

    finetune.save_checkpoint()
    metric_key = "eval_miou" if args.skill == "detect" else "eval_f1"
    f1_target_value: Optional[float] = None
    f1_target_pass: Optional[bool] = None
    if best_step is not None:
        run.summary["best_step"] = best_step
        run.summary[f"best_{metric_key}"] = best_metric
        run.summary["best_metric_key"] = metric_key
    if metric_key == "eval_f1" and baseline_eval_metric is not None and best_metric is not None:
        f1_target_value = float(baseline_eval_metric) + float(args.f1_improvement_target)
        f1_target_pass = bool(float(best_metric) >= f1_target_value)
        run.summary["f1_target_value"] = f1_target_value
        run.summary["f1_target_pass"] = int(f1_target_pass)
    run.summary["f1_target_evaluated"] = int(f1_target_pass is not None)
    run.summary["recall_gate_evaluated"] = int(recall_gate_pass is not None)
    if recall_gate_pass is not None:
        run.summary["recall_gate_pass"] = int(recall_gate_pass)
    if recall_gate_eval_step is not None:
        run.summary["recall_gate_eval_step"] = int(recall_gate_eval_step)
    if recall_gate_eval_tp is not None:
        run.summary["recall_gate_eval_tp"] = float(recall_gate_eval_tp)
    if recall_gate_min_tp is not None:
        run.summary["recall_gate_min_tp"] = float(recall_gate_min_tp)
    final_usage = _usage_snapshot(usage, total_train_rows=total_train_rows, top_k=args.usage_top_k)
    _print_usage_snapshot(final_usage, prefix="usage final")
    run.summary["rows_seen"] = int(final_usage["rows_seen"])
    run.summary["rows_seen_fraction"] = float(final_usage["rows_seen_fraction"])
    run.summary["tasks_generated_total"] = int(final_usage["tasks_generated"])
    run.summary["tasks_generated_positive_total"] = int(final_usage["tasks_generated_positive"])
    run.summary["tasks_generated_negative_total"] = int(final_usage["tasks_generated_negative"])
    run.summary["tasks_consumed_total"] = int(final_usage["tasks_consumed"])
    run.summary["tasks_consumed_positive_total"] = int(final_usage["tasks_consumed_positive"])
    run.summary["tasks_consumed_negative_total"] = int(final_usage["tasks_consumed_negative"])
    run.summary["usage_source_rows_top_json"] = json.dumps(final_usage["source_rows_top"])
    run.summary["usage_source_tasks_consumed_top_json"] = json.dumps(final_usage["source_tasks_consumed_top"])
    run.summary["usage_class_tasks_consumed_top_json"] = json.dumps(final_usage["class_tasks_consumed_top"])
    run.summary["eval_events_logged"] = int(eval_events_logged)
    run.summary["finetune_id"] = finetune.finetune_id
    run.finish()
    client.close()

    print(
        f"done. finetune_id={finetune.finetune_id} "
        f"best_step={best_step} best_metric={best_metric} "
        f"recall_gate_pass={recall_gate_pass} f1_target_pass={f1_target_pass}"
    )


if __name__ == "__main__":
    main()
