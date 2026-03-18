#!/usr/bin/env python3
"""Class-conditional RL finetuning for football detection."""

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
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

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

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from football_detect.common import (  # noqa: E402
    NormalizedBox,
    clamp,
    config_to_cli_args,
    discover_class_names,
    default_prompt_for_class,
    load_json_config,
    normalize_class_name,
    parse_box_element_annotations,
    repo_relative,
    resolve_config_path,
)
from tuna_sdk import (  # noqa: E402
    DetectAnnotation,
    DetectOutput,
    DetectRequest,
    DetectSettings,
    Rollout,
    RolloutsRequest,
    TrainStepGroup,
    TunaClient,
)
from tuna_sdk.errors import TunaAPIError, TunaNetworkError  # noqa: E402

DEFAULT_CONFIG_PATH = repo_relative("configs", "train_football_detect_default.json")
DEFAULT_BASE_URL = "https://api.moondream.ai/v1"
VAL_SPLIT_CANDIDATES = ("validation", "val", "dev", "test", "post_val")


def _random_suffix(length: int = 6) -> str:
    chars = string.ascii_lowercase + string.digits
    return "".join(random.choices(chars, k=length))


def _to_data_url(image: Image.Image, *, quality: int = 90) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=max(1, min(100, int(quality))))
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


@dataclass(frozen=True)
class _ReasoningDetectRequest(DetectRequest):
    reasoning: bool = False

    def to_payload(self) -> dict[str, Any]:
        payload = super().to_payload()
        if bool(self.reasoning):
            payload["reasoning"] = True
        return payload


@dataclass(frozen=True)
class ClassBox:
    class_name: str
    box: DetectAnnotation
    prompt: Optional[str] = None


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


def _to_detect_annotation(box: NormalizedBox) -> DetectAnnotation:
    return DetectAnnotation(
        x_min=float(box.x_min),
        y_min=float(box.y_min),
        x_max=float(box.x_max),
        y_max=float(box.y_max),
    )


def _box_from_normalized(x_min: float, y_min: float, x_max: float, y_max: float) -> DetectAnnotation:
    x0 = clamp(float(x_min))
    y0 = clamp(float(y_min))
    x1 = clamp(float(x_max))
    y1 = clamp(float(y_max))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("invalid normalized bbox")
    return DetectAnnotation(x_min=x0, y_min=y0, x_max=x1, y_max=y1)


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


def _extract_class_names_from_file(path_str: str) -> list[str]:
    if not path_str:
        return []
    path = Path(path_str).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("class_catalog"), list):
        names = [
            str(item.get("class_name") or "").strip()
            for item in payload["class_catalog"]
            if isinstance(item, dict)
        ]
        return sorted({name for name in names if name})
    if isinstance(payload, list):
        if payload and all(isinstance(item, dict) for item in payload):
            names = [str(item.get("class_name") or "").strip() for item in payload]
            return sorted({name for name in names if name})
        return sorted({str(item).strip() for item in payload if str(item).strip()})
    raise ValueError(f"Unsupported class names payload in {path}")


def _parse_prompt_overrides_json(raw_value: str) -> dict[str, str]:
    text = str(raw_value or "").strip()
    if not text:
        return {}
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("--prompt-overrides-json must decode to an object")
    out: dict[str, str] = {}
    for key, value in payload.items():
        name = normalize_class_name(key)
        prompt = str(value or "").strip()
        if not name or not prompt:
            continue
        out[name] = prompt
    return out


def _prompt_for_class(class_name: str, *, prompt_overrides: Mapping[str, str]) -> str:
    normalized = normalize_class_name(class_name)
    override = str(prompt_overrides.get(normalized, "") or "").strip()
    if override:
        return override
    return default_prompt_for_class(normalized)


def _to_base_sample(row: Mapping[str, Any]) -> Optional[BaseSample]:
    image = row.get("image")
    if image is None:
        return None
    image = image.convert("RGB")
    width, height = image.size
    parsed_boxes = parse_box_element_annotations(row.get("answer_boxes"), width=width, height=height)
    boxes: list[ClassBox] = []
    row_class_name = normalize_class_name(row.get("class_name"))
    row_prompt = str(row.get("prompt") or "").strip()
    task_schema = str(row.get("task_schema") or "").strip()
    use_row_prompt = bool(row_prompt) and bool(row_class_name) and (
        task_schema == "per_box_element" or len(parsed_boxes) == 1
    )
    for item in parsed_boxes:
        annotation = _to_detect_annotation(item.box)
        prompt = row_prompt if use_row_prompt and row_class_name == item.class_name else None
        boxes.append(ClassBox(class_name=item.class_name, box=annotation, prompt=prompt))
    source = str(row.get("source_collection") or row.get("source_dataset") or row.get("source") or "unknown")
    return BaseSample(image=image, boxes=boxes, source=source)


def tasks_from_base_sample(
    sample: BaseSample,
    *,
    all_class_names: list[str],
    rng: random.Random,
    neg_prompts_per_empty: int,
    neg_prompts_per_nonempty: int,
    prompt_overrides: Mapping[str, str],
) -> list[TaskSample]:
    tasks: list[TaskSample] = []
    present_names = {item.class_name for item in sample.boxes}

    if sample.boxes:
        for item in sample.boxes:
            tasks.append(
                TaskSample(
                    image=sample.image,
                    prompt=item.prompt or _prompt_for_class(item.class_name, prompt_overrides=prompt_overrides),
                    gt_boxes=[item.box],
                    class_name=item.class_name,
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
                        prompt=_prompt_for_class(class_name, prompt_overrides=prompt_overrides),
                        gt_boxes=[],
                        class_name=class_name,
                        is_positive=False,
                        source=sample.source,
                    )
                )
        return tasks

    if not all_class_names or neg_prompts_per_empty <= 0:
        return []
    picks = rng.sample(all_class_names, k=min(neg_prompts_per_empty, len(all_class_names)))
    for class_name in picks:
        tasks.append(
            TaskSample(
                image=sample.image,
                prompt=_prompt_for_class(class_name, prompt_overrides=prompt_overrides),
                gt_boxes=[],
                class_name=class_name,
                is_positive=False,
                source=sample.source,
            )
        )
    return tasks


def _horizontal_flip(image: Image.Image, boxes: list[DetectAnnotation]) -> tuple[Image.Image, list[DetectAnnotation]]:
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
    boxes: list[DetectAnnotation],
    rng: random.Random,
    *,
    scale_min: float,
    scale_max: float,
) -> tuple[Image.Image, list[DetectAnnotation]]:
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

    kept: list[DetectAnnotation] = []
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
    boxes: list[DetectAnnotation],
    rng: random.Random,
    *,
    scale_min: float,
    scale_max: float,
) -> tuple[Image.Image, list[DetectAnnotation]]:
    width, height = image.size
    scale = rng.uniform(scale_min, scale_max)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    if new_width == width and new_height == height:
        return image, boxes
    resized = image.resize((new_width, new_height), resample=Image.BICUBIC)
    adjusted: list[DetectAnnotation] = []
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


def augment_task_sample(
    sample: TaskSample,
    rng: random.Random,
    rng_np: np.random.Generator,
    config: AugmentConfig,
    *,
    augment_prob: float,
) -> TaskSample:
    image = sample.image
    boxes = list(sample.gt_boxes)

    image, boxes = _random_resize(
        image,
        boxes,
        rng,
        scale_min=config.resize_min,
        scale_max=config.resize_max,
    )
    if rng.random() >= augment_prob:
        return TaskSample(
            image=image,
            prompt=sample.prompt,
            gt_boxes=boxes,
            class_name=sample.class_name,
            is_positive=sample.is_positive,
            source=sample.source,
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
        if sample.is_positive and pre_crop_boxes and not cropped_boxes:
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

    return TaskSample(
        image=image,
        prompt=sample.prompt,
        gt_boxes=boxes,
        class_name=sample.class_name,
        is_positive=sample.is_positive,
        source=sample.source,
    )


def _iter_dataset_rows(ds: Dataset, seed: int) -> Iterable[dict]:
    epoch = 0
    while True:
        shuffled = ds.shuffle(seed=seed + epoch) if seed else ds
        for row in shuffled:
            yield row
        epoch += 1


def _load_local_dataset_dict(dataset_path: str) -> DatasetDict:
    dataset_obj = load_from_disk(dataset_path)
    if not isinstance(dataset_obj, DatasetDict):
        raise ValueError("Local dataset path must contain a DatasetDict with train/val splits.")
    return dataset_obj


def _iter_local_rows_once(dataset_obj: DatasetDict, split: str) -> Iterable[dict]:
    if split not in dataset_obj:
        available = ", ".join(dataset_obj.keys())
        raise ValueError(f"Split '{split}' not found. Available: {available}")
    return iter(dataset_obj[split])


def _iter_hf_rows(dataset_name: str, split: str, token: Optional[str], seed: int, buffer_size: int) -> Iterable[dict]:
    epoch = 0
    while True:
        ds = load_dataset(dataset_name, split=split, token=token, streaming=True)
        if seed:
            ds = ds.shuffle(seed=seed + epoch, buffer_size=buffer_size)
        for row in ds:
            yield row
        epoch += 1


def _iter_hf_rows_once(dataset_name: str, split: str, token: Optional[str]) -> Iterable[dict]:
    return load_dataset(dataset_name, split=split, token=token, streaming=True)


def _resolve_val_split(available_splits: list[str], train_split: str, val_split: str) -> str:
    if not available_splits:
        raise ValueError("Dataset exposes no splits.")

    train_base_split = train_split.split("[", 1)[0]
    if not val_split:
        for candidate in VAL_SPLIT_CANDIDATES:
            if candidate in available_splits and candidate != train_base_split:
                return candidate
        raise ValueError(
            f"Dataset must already contain a validation split. Available splits: {available_splits}"
        )
    if val_split in available_splits:
        return val_split
    if "[" in val_split and "]" in val_split:
        base_split = val_split.split("[", 1)[0]
        if base_split in available_splits:
            return val_split
    raise ValueError(f"val split '{val_split}' not found. Available splits: {available_splits}")


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


def _rewards_for_rollouts(
    rollouts: list[Rollout],
    gt_boxes: list[DetectAnnotation],
    *,
    reward_metric: str,
    fn_penalty_exponent: float,
    fp_penalty_exponent: float,
    neg_reward_weight: float,
) -> list[float]:
    rewards: list[float] = []
    is_negative_task = len(gt_boxes) == 0
    for rollout in rollouts:
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


def _format_tuna_error(exc: Exception) -> str:
    parts = [f"{type(exc).__name__}: {exc}"]
    if isinstance(exc, TunaAPIError):
        if exc.status_code is not None:
            parts.append(f"status={exc.status_code}")
        if exc.request_id:
            parts.append(f"request_id={exc.request_id}")
        if exc.response_body is not None:
            body = exc.response_body
            if not isinstance(body, str):
                try:
                    body = json.dumps(body, ensure_ascii=True)
                except Exception:
                    body = str(body)
            body_text = str(body).strip().replace("\n", " ")
            if len(body_text) > 400:
                body_text = body_text[:400] + "..."
            if body_text:
                parts.append(f"body={body_text}")
    elif isinstance(exc, TunaNetworkError) and getattr(exc, "cause", None) is not None:
        parts.append(f"cause={exc.cause}")
    return " | ".join(parts)


def _is_reasoning_unsupported_error(exc: Exception) -> bool:
    if not isinstance(exc, TunaAPIError):
        return False
    if int(getattr(exc, "status_code", 0) or 0) != 422:
        return False
    body = getattr(exc, "response_body", None)
    if body is None:
        text = str(exc)
        return "reasoning" in text and "extra_forbidden" in text
    body_text = json.dumps(body, ensure_ascii=True) if isinstance(body, dict) else str(body)
    lowered = body_text.lower()
    return "reasoning" in lowered and "extra_forbidden" in lowered


def _rollouts_batch_with_retry(
    *,
    finetune,
    requests: list[DetectRequest],
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
                f"after {delay:.2f}s due to {_format_tuna_error(exc)}"
            )
            time.sleep(delay)
            worker_count = max(1, worker_count // 2)
            attempt += 1


def _evaluate(
    *,
    finetune,
    eval_rows: Iterable[dict],
    all_class_names: list[str],
    prompt_overrides: Mapping[str, str],
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
    reasoning: bool,
) -> dict[str, float]:
    tasks: list[TaskSample] = []
    total = 0
    total_f1 = 0.0
    total_miou = 0.0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    reasoning_failure_hint_emitted = False

    def _drain_batch(batch: list[TaskSample]) -> None:
        nonlocal total, total_f1, total_miou, total_tp, total_fp, total_fn, reasoning_failure_hint_emitted
        if not batch:
            return
        chunk_size = max(1, min(max_workers, len(batch)))
        for offset in range(0, len(batch), chunk_size):
            chunk = batch[offset : offset + chunk_size]
            requests = [
                _ReasoningDetectRequest(
                    object_name=item.prompt,
                    image_url=_to_data_url(item.image, quality=92),
                    settings=DetectSettings(
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                        max_objects=max_objects,
                    ),
                    reasoning=bool(reasoning),
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
                if bool(reasoning) and _is_reasoning_unsupported_error(exc):
                    raise ValueError(
                        "API rejects request reasoning for tuning rollouts (422 extra_forbidden). "
                        "Use --no-reasoning --no-eval-reasoning for train/eval rollouts."
                    ) from exc
                print(f"eval rollouts_batch failed: {_format_tuna_error(exc)}. skipping chunk")
                if bool(reasoning) and not reasoning_failure_hint_emitted:
                    print(
                        "hint: this may indicate reasoning is not supported for detect tuning rollouts in "
                        "your current API path. Try --no-reasoning --no-eval-reasoning."
                    )
                    reasoning_failure_hint_emitted = True
                continue

            for item, result in zip(chunk, results):
                pred_boxes: list[DetectAnnotation] = []
                if result.rollouts:
                    rollout0 = result.rollouts[0]
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
        task_list = tasks_from_base_sample(
            base,
            all_class_names=all_class_names,
            rng=rng,
            neg_prompts_per_empty=neg_prompts_per_empty,
            neg_prompts_per_nonempty=neg_prompts_per_nonempty,
            prompt_overrides=prompt_overrides,
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


def _print_usage_snapshot(snapshot: Mapping[str, Any], *, prefix: str) -> None:
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


def _resolve_env_file(env_file: str) -> str:
    path = Path(env_file).expanduser()
    if path.is_absolute():
        return str(path)

    from_cwd = (Path.cwd() / path).resolve()
    if from_cwd.exists():
        return str(from_cwd)

    from_repo = (REPO_ROOT / path).resolve()
    if from_repo.exists():
        return str(from_repo)

    from_script = (repo_relative(path.as_posix())).resolve()
    if from_script.exists():
        return str(from_script)

    return str(from_cwd)


def _resolve_runtime_env(args: argparse.Namespace) -> argparse.Namespace:
    args.env_file = _resolve_env_file(str(args.env_file))
    args.api_key_env_var = str(getattr(args, "api_key_env_var", "") or "").strip() or "MOONDREAM_API_KEY"
    load_dotenv(args.env_file, override=True)
    if not args.api_key:
        args.api_key = os.environ.get(args.api_key_env_var, "")
    if not args.api_key and args.api_key_env_var != "MOONDREAM_API_KEY":
        args.api_key = os.environ.get("MOONDREAM_API_KEY", "")
    if not args.hf_token:
        args.hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not args.base_url:
        args.base_url = os.environ.get("TUNA_BASE_URL", DEFAULT_BASE_URL)
    return args


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    raw_argv = list(argv) if argv is not None else list(sys.argv[1:])
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    pre_args, _ = pre_parser.parse_known_args(raw_argv)
    config_path = resolve_config_path(pre_args.config, script_dir=SCRIPT_DIR)
    config = load_json_config(config_path, default_path=DEFAULT_CONFIG_PATH)

    parser = argparse.ArgumentParser(description="RL finetune Moondream for football detection.")
    parser.add_argument("--config", default=str(config_path))
    parser.add_argument("--env-file", default=str(repo_relative(".env")))
    parser.add_argument("--api-key", default="")
    parser.add_argument("--api-key-env-var", default="MOONDREAM_API_KEY")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"))
    parser.add_argument("--base-url", default="")

    parser.add_argument(
        "--dataset-path",
        default=str(repo_relative("outputs", "maxs-m87_football_detect_no_split_splits")),
    )
    parser.add_argument("--dataset-name", default="")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--val-split",
        default="",
        help="Validation split name. If omitted, uses validation/val/dev/test/post_val when present.",
    )
    parser.add_argument("--class-names-file", default="")
    parser.add_argument("--include-classes", nargs="*", default=None)
    parser.add_argument("--exclude-classes", nargs="*", default=None)
    parser.add_argument("--prompt-overrides-json", default="{}")

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
        "--reward-metric",
        choices=["f1", "miou"],
        default="f1",
        help="Training reward metric for detect rollouts.",
    )

    reasoning_group = parser.add_mutually_exclusive_group()
    reasoning_group.add_argument(
        "--reasoning",
        dest="reasoning",
        action="store_true",
        help="Enable Moondream reasoning for training rollout requests.",
    )
    reasoning_group.add_argument(
        "--no-reasoning",
        dest="reasoning",
        action="store_false",
        help="Disable Moondream reasoning for training rollout requests.",
    )
    parser.set_defaults(reasoning=False)

    eval_reasoning_group = parser.add_mutually_exclusive_group()
    eval_reasoning_group.add_argument(
        "--eval-reasoning",
        dest="eval_reasoning",
        action="store_true",
        help="Force reasoning=true for eval requests.",
    )
    eval_reasoning_group.add_argument(
        "--no-eval-reasoning",
        dest="eval_reasoning",
        action="store_false",
        help="Force reasoning=false for eval requests.",
    )
    eval_reasoning_group.add_argument(
        "--eval-reasoning-inherit",
        dest="eval_reasoning",
        action="store_const",
        const=None,
        help="Inherit eval reasoning from --reasoning.",
    )
    parser.set_defaults(eval_reasoning=None)

    parser.add_argument("--off-policy", action="store_true")
    parser.add_argument(
        "--allow-off-policy-with-reasoning",
        action="store_true",
        help="Allow off-policy GT injection even when --reasoning is enabled.",
    )
    parser.add_argument("--off-policy-std-thresh", type=float, default=0.02)
    parser.add_argument("--off-policy-max-reward", type=float, default=0.15)
    parser.add_argument("--off-policy-min-reward", type=float, default=0.15)
    parser.add_argument("--off-policy-reward-scale", type=float, default=2.0)
    parser.add_argument(
        "--fn-penalty-exponent",
        type=float,
        default=1.0,
        help="Exponent for false negatives in reward denominator via FN^exp.",
    )
    parser.add_argument(
        "--fp-penalty-exponent",
        type=float,
        default=1.0,
        help="Exponent for false positives in reward denominator via FP^exp.",
    )

    parser.add_argument("--neg-prompts-per-empty", type=int, default=0)
    parser.add_argument("--neg-prompts-per-nonempty", type=int, default=1)
    parser.add_argument(
        "--pos-task-prob",
        type=float,
        default=0.95,
        help="When an image has positive tasks, choose a positive task with this probability.",
    )
    parser.add_argument(
        "--neg-reward-weight",
        type=float,
        default=0.5,
        help="Scale factor applied to rewards for negative tasks (no GT boxes).",
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
        help="How many sources/classes to include in usage summaries.",
    )
    parser.add_argument("--wandb-project", default="moondream-football-detect-rl")
    parser.add_argument("--wandb-run-name", default="")

    option_to_dest: dict[str, str] = {}
    for action in parser._actions:
        if not action.option_strings:
            continue
        for opt in action.option_strings:
            option_to_dest[opt] = action.dest
    overridden_dests = {option_to_dest[arg] for arg in raw_argv if arg in option_to_dest}
    config_cli_args = config_to_cli_args(
        parser,
        config,
        config_path=config_path,
        overridden_dests=overridden_dests,
    )
    args = parser.parse_args(config_cli_args + raw_argv)
    args.config = str(resolve_config_path(args.config, script_dir=SCRIPT_DIR))
    args.include_classes = list(args.include_classes or [])
    args.exclude_classes = list(args.exclude_classes or [])
    args.prompt_overrides = _parse_prompt_overrides_json(args.prompt_overrides_json)
    return args


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    args = _resolve_runtime_env(args)

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

    eval_reasoning = bool(args.reasoning) if args.eval_reasoning is None else bool(args.eval_reasoning)
    off_policy_injection_allowed = bool(
        args.off_policy and (not args.reasoning or args.allow_off_policy_with_reasoning)
    )
    if args.off_policy and args.reasoning and not args.allow_off_policy_with_reasoning:
        print(
            "warning: off-policy injection is disabled while --reasoning is enabled. "
            "Set --allow-off-policy-with-reasoning to override."
        )

    dataset_path = str(args.dataset_path or "").strip()
    dataset_name = str(args.dataset_name or "").strip()
    use_local = bool(dataset_path)
    if use_local and dataset_name:
        print("warning: both --dataset-path and --dataset-name provided; using --dataset-path.")
    if not use_local and not dataset_name:
        raise ValueError("Provide --dataset-path or --dataset-name.")

    rng = random.Random(args.seed)
    rng_np = np.random.default_rng(args.seed)
    augment_config = _default_augment_config()
    usage = UsageStats()
    total_train_rows: Optional[int] = None
    total_val_rows: Optional[int] = None

    eval_rows_factory: Callable[[], Iterable[dict]]
    train_row_iter: Iterable[dict]
    train_split = args.split.strip() or "train"
    requested_val_split = args.val_split.strip()
    val_split: str

    if use_local:
        dataset_obj = _load_local_dataset_dict(dataset_path)
        if train_split not in dataset_obj:
            raise ValueError(f"train split '{train_split}' not found in local dataset splits: {list(dataset_obj)}")
        val_split = _resolve_val_split(list(dataset_obj.keys()), train_split, requested_val_split)
        train_ds = dataset_obj[train_split]
        train_row_iter = _iter_dataset_rows(train_ds, args.seed)
        total_train_rows = len(train_ds)
        total_val_rows = len(dataset_obj[val_split])

        def _local_eval_rows() -> Iterable[dict]:
            return _iter_local_rows_once(dataset_obj, val_split)

        eval_rows_factory = _local_eval_rows
        class_source_rows: Iterable[Mapping[str, Any]] = iter(train_ds)
    else:
        split_names = list(get_dataset_split_names(dataset_name, token=args.hf_token))
        if train_split not in split_names:
            raise ValueError(f"train split '{train_split}' not found in dataset splits: {split_names}")
        val_split = _resolve_val_split(split_names, train_split, requested_val_split)
        train_row_iter = _iter_hf_rows(dataset_name, train_split, args.hf_token, args.seed, args.buffer_size)

        def _hf_eval_rows() -> Iterable[dict]:
            return _iter_hf_rows_once(dataset_name, val_split, args.hf_token)

        eval_rows_factory = _hf_eval_rows
        class_source_rows = _iter_hf_rows_once(dataset_name, train_split, args.hf_token)

    all_class_names = _extract_class_names_from_file(args.class_names_file)
    if not all_class_names:
        all_class_names = discover_class_names(class_source_rows)
    if not all_class_names:
        raise ValueError("Could not resolve class names for prompting from the train split.")

    if args.include_classes:
        include_set = set(args.include_classes)
        missing = [name for name in args.include_classes if name not in set(all_class_names)]
        if missing:
            raise ValueError(f"--include-classes contains unknown class names: {missing}")
        all_class_names = [name for name in all_class_names if name in include_set]
    if args.exclude_classes:
        exclude_set = set(args.exclude_classes)
        all_class_names = [name for name in all_class_names if name not in exclude_set]
    if not all_class_names:
        raise ValueError("Class filters removed every class.")

    prompt_override_unknown = sorted(set(args.prompt_overrides) - set(all_class_names))
    if prompt_override_unknown:
        print(f"warning: prompt overrides for unknown classes ignored: {prompt_override_unknown}")
        args.prompt_overrides = {
            key: value for key, value in args.prompt_overrides.items() if key in set(all_class_names)
        }

    if not args.finetune_id and not args.finetune_name:
        args.finetune_name = f"football-detect-{_random_suffix()}"

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
    print(
        "run control: "
        f"num_steps={args.num_steps} resume_step={args.resume_step} "
        f"eval_every={args.eval_every} save_every={args.save_every} "
        f"off_policy={args.off_policy} off_policy_injection_allowed={off_policy_injection_allowed} "
        f"reasoning_train={bool(args.reasoning)} reasoning_eval={eval_reasoning}"
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
            "config": args.config,
            "finetune_id": finetune.finetune_id,
            "dataset_path": dataset_path or None,
            "dataset_name": dataset_name or None,
            "train_split": train_split,
            "val_split": val_split,
            "train_rows_total": total_train_rows,
            "val_rows_total": total_val_rows,
            "expected_tasks_consumed": expected_tasks,
            "class_count": len(all_class_names),
            "class_names": list(all_class_names),
            "prompt_overrides": dict(args.prompt_overrides),
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
            "reward_metric": args.reward_metric,
            "seed": args.seed,
            "off_policy": args.off_policy,
            "allow_off_policy_with_reasoning": args.allow_off_policy_with_reasoning,
            "off_policy_injection_allowed": off_policy_injection_allowed,
            "off_policy_std_thresh": args.off_policy_std_thresh,
            "off_policy_max_reward": args.off_policy_max_reward,
            "off_policy_min_reward": args.off_policy_min_reward,
            "off_policy_reward_scale": args.off_policy_reward_scale,
            "fn_penalty_exponent": args.fn_penalty_exponent,
            "fp_penalty_exponent": args.fp_penalty_exponent,
            "pos_task_prob": args.pos_task_prob,
            "neg_reward_weight": args.neg_reward_weight,
            "reasoning": bool(args.reasoning),
            "eval_reasoning": bool(eval_reasoning),
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

            task_candidates = tasks_from_base_sample(
                base,
                all_class_names=all_class_names,
                rng=rng,
                neg_prompts_per_empty=args.neg_prompts_per_empty,
                neg_prompts_per_nonempty=args.neg_prompts_per_nonempty,
                prompt_overrides=args.prompt_overrides,
            )
            if not task_candidates:
                continue
            usage.tasks_generated += len(task_candidates)
            for task in task_candidates:
                usage.source_tasks_generated[task.source] += 1
                usage.class_tasks_generated[task.class_name] += 1
                if task.is_positive:
                    usage.tasks_generated_positive += 1
                else:
                    usage.tasks_generated_negative += 1

            positives = [task for task in task_candidates if task.is_positive]
            negatives = [task for task in task_candidates if not task.is_positive]
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
            return augment_task_sample(
                selected_task,
                rng,
                rng_np,
                augment_config,
                augment_prob=args.augment_prob,
            )

    best_metric: Optional[float] = None
    best_step: Optional[int] = None
    successful_updates = args.resume_step
    baseline_eval_metric: Optional[float] = None
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
                prompt_overrides=args.prompt_overrides,
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
                reasoning=eval_reasoning,
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

    if args.eval_every > 0:
        baseline_step = max(0, args.resume_step - 1)
        print("running baseline eval before training...")
        baseline_metrics = _run_and_log_eval(trigger="baseline", step_for_log=baseline_step)
        if baseline_metrics is not None:
            baseline_eval_metric = float(baseline_metrics.get("eval_miou", 0.0))
            run.summary["baseline_eval_miou"] = baseline_eval_metric
            print(
                f"baseline eval step {baseline_step} tasks={baseline_metrics['eval_tasks']} "
                f"miou={baseline_metrics['eval_miou']:.4f} f1={baseline_metrics['eval_f1']:.4f} "
                f"macro_f1={baseline_metrics['eval_f1_macro']:.4f}"
            )

    for step in range(args.num_steps):
        global_step = args.resume_step + step
        step_start = time.monotonic()

        batch = [_next_task() for _ in range(args.batch_size)]
        requests = [
            _ReasoningDetectRequest(
                object_name=item.prompt,
                image_url=_to_data_url(item.image, quality=92),
                settings=DetectSettings(
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens,
                    max_objects=args.max_objects,
                ),
                reasoning=bool(args.reasoning),
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
            if bool(args.reasoning) and _is_reasoning_unsupported_error(exc):
                raise ValueError(
                    "API rejects request reasoning for tuning rollouts (422 extra_forbidden). "
                    "Use --no-reasoning --no-eval-reasoning for train/eval rollouts."
                ) from exc
            print(f"rollouts_batch failed at step {global_step}: {_format_tuna_error(exc)}. skipping step")
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
        off_policy_skipped_reasoning_guard = 0
        train_tp = 0
        train_fp = 0
        train_fn = 0

        for idx, (item, result) in enumerate(zip(batch, results)):
            rollouts = list(result.rollouts)
            rewards = _rewards_for_rollouts(
                rollouts,
                item.gt_boxes,
                reward_metric=args.reward_metric,
                fn_penalty_exponent=args.fn_penalty_exponent,
                fp_penalty_exponent=args.fp_penalty_exponent,
                neg_reward_weight=args.neg_reward_weight,
            )

            if off_policy_injection_allowed and rewards and rollouts:
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
                    rollouts[replace_idx] = Rollout(
                        skill=old_rollout.skill,
                        finish_reason=old_rollout.finish_reason,
                        output=DetectOutput(objects=replacement_objects),
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
            elif args.off_policy and rewards and rollouts and not off_policy_injection_allowed:
                off_policy_skipped_reasoning_guard += 1

            for rollout in rollouts:
                output = rollout.output
                pred_boxes = output.objects if isinstance(output, DetectOutput) else []
                tp, fp, fn = _count_tp_fp_fn(pred_boxes, item.gt_boxes)
                train_tp += tp
                train_fp += fp
                train_fn += fn

            group_request = result.request
            if idx < len(requests):
                group_request = RolloutsRequest(
                    finetune_id=result.request.finetune_id,
                    num_rollouts=result.request.num_rollouts,
                    request=requests[idx],
                    ground_truth=result.request.ground_truth,
                    org_id=result.request.org_id,
                )
            groups.append(TrainStepGroup(request=group_request, rollouts=rollouts, rewards=rewards))
            all_rewards.extend(rewards)

        try:
            train_start = time.monotonic()
            train_out = finetune.train_step(groups=groups, lr=args.lr)
            train_end = time.monotonic()
        except (TunaAPIError, TunaNetworkError) as exc:
            print(f"train_step failed at step {global_step}: {_format_tuna_error(exc)}. skipping step")
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
                "off_policy_skipped_reasoning_guard": off_policy_skipped_reasoning_guard,
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
            f"offp={off_policy_injected_total}/{off_policy_considered} "
            f"offp_guarded={off_policy_skipped_reasoning_guard} updates={successful_updates}"
        )
        if args.usage_report_every > 0 and (global_step + 1) % args.usage_report_every == 0:
            usage_summary = _usage_snapshot(usage, total_train_rows=total_train_rows, top_k=args.usage_top_k)
            _print_usage_snapshot(usage_summary, prefix=f"usage step {global_step}")

        if args.eval_every > 0 and successful_updates % args.eval_every == 0:
            eval_metrics = _run_and_log_eval(trigger="periodic", step_for_log=global_step)
            if eval_metrics is None:
                continue
            if baseline_eval_metric is not None:
                delta = float(eval_metrics.get("eval_miou", 0.0)) - baseline_eval_metric
                wandb.log({"eval_miou_delta_vs_baseline": delta}, step=global_step)
            print(
                f"eval step {global_step} tasks={eval_metrics['eval_tasks']} "
                f"miou={eval_metrics['eval_miou']:.4f} f1={eval_metrics['eval_f1']:.4f} "
                f"macro_f1={eval_metrics['eval_f1_macro']:.4f} updates={successful_updates}"
            )

            metric = float(eval_metrics.get("eval_miou", 0.0))
            if best_metric is None or metric > best_metric:
                best_metric = metric
                best_step = global_step
                finetune.save_checkpoint()

        if args.save_every > 0 and (global_step + 1) % args.save_every == 0:
            finetune.save_checkpoint()

    finetune.save_checkpoint()
    if best_step is not None:
        run.summary["best_step"] = best_step
        run.summary["best_eval_miou"] = best_metric
    final_usage = _usage_snapshot(usage, total_train_rows=total_train_rows, top_k=args.usage_top_k)
    _print_usage_snapshot(final_usage, prefix="usage final")
    run.summary["rows_seen"] = int(final_usage["rows_seen"])
    run.summary["tasks_generated"] = int(final_usage["tasks_generated"])
    run.summary["tasks_consumed"] = int(final_usage["tasks_consumed"])
    run.finish()


if __name__ == "__main__":
    main()
