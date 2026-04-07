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
import shlex
import subprocess
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

from async_checkpoint_eval import (
    CheckpointEvalResult,
    dispatch_checkpoint_eval,
    drain_checkpoint_eval_jobs,
    poll_checkpoint_eval_jobs,
)
from finetune_checkpoints import save_checkpoint_step

os.environ.setdefault("WANDB_START_METHOD", "thread")
os.environ.setdefault("WANDB__SERVICE_WAIT", "300")

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

from tuna_sdk import (
    DetectAnnotation,
    DetectOutput,
    DetectRequest,
    DetectSettings,
    Rollout,
    RolloutsRequest,
    TrainStepGroup,
    TunaClient,
)
from tuna_sdk import PointAnnotation, PointOutput, PointRequest, PointSettings
from tuna_sdk.errors import TunaAPIError, TunaNetworkError

from aerial_airport.runtime_tiling import (
    Box2D as RuntimeBox,
    Point2D as RuntimePoint,
    TileWindow,
    build_tile_windows,
    clip_box_to_window,
    crop_image_to_tiles,
    merge_boxes as merge_runtime_boxes,
    merge_points as merge_runtime_points,
    map_box_from_tile,
    map_point_from_tile,
)


def _repo_relative(*parts: str) -> Path:
    return Path(__file__).resolve().parent.joinpath(*parts)


DEFAULT_CONFIG_PATH = _repo_relative("configs", "train_pid_icons_default.json")


def _resolve_config_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    from_cwd = (Path.cwd() / path).resolve()
    if from_cwd.exists():
        return from_cwd
    from_repo = (REPO_ROOT / path).resolve()
    if from_repo.exists():
        return from_repo
    from_script = (_repo_relative(path.as_posix())).resolve()
    if from_script.exists():
        return from_script
    return from_cwd


def _load_json_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        if config_path == DEFAULT_CONFIG_PATH:
            return {}
        raise FileNotFoundError(f"Config file not found: {config_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a JSON object: {config_path}")
    return payload


def _option_for_action(action: argparse.Action) -> str:
    for opt in action.option_strings:
        if opt.startswith("--"):
            return opt
    return action.option_strings[0]


def _config_to_cli_args(
    parser: argparse.ArgumentParser,
    config: dict[str, Any],
    *,
    config_path: Path,
    overridden_dests: Optional[set[str]] = None,
) -> list[str]:
    overridden = set(overridden_dests or set())
    by_dest: dict[str, list[argparse.Action]] = {}
    for action in parser._actions:
        if not action.option_strings or action.dest == "help":
            continue
        by_dest.setdefault(action.dest, []).append(action)

    unknown = sorted(key for key in config if key not in by_dest)
    if unknown:
        raise ValueError(
            f"Unknown config key(s) in {config_path}: {unknown}. "
            "Remove typos or update script support."
        )

    cli_args: list[str] = []
    for key, raw_value in config.items():
        if key in overridden:
            continue
        actions = by_dest[key]
        const_actions = [a for a in actions if isinstance(a, argparse._StoreConstAction)]
        store_actions = [a for a in actions if not isinstance(a, argparse._StoreConstAction)]

        if raw_value is None:
            matched = next((a for a in const_actions if getattr(a, "const", object()) is None), None)
            if matched is not None:
                cli_args.append(_option_for_action(matched))
            continue

        if isinstance(raw_value, bool):
            matched = next((a for a in const_actions if getattr(a, "const", object()) is raw_value), None)
            if matched is not None:
                cli_args.append(_option_for_action(matched))
            continue

        if not store_actions:
            continue

        action = store_actions[0]
        cli_args.append(_option_for_action(action))
        if isinstance(raw_value, list):
            cli_args.extend(str(item) for item in raw_value)
        else:
            cli_args.append(str(raw_value))
    return cli_args


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
class _ReasoningPointRequest(PointRequest):
    reasoning: bool = False

    def to_payload(self) -> dict[str, Any]:
        payload = super().to_payload()
        if bool(self.reasoning):
            payload["reasoning"] = True
        return payload


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
TEST_SPLIT_CANDIDATES = ("test", "post_val")
SELECTION_METRIC_CHOICES = ("f1", "f1_macro", "miou")


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


def _single_class_point_mode_warnings(
    *,
    skill: str,
    all_class_names: list[str],
) -> list[str]:
    if skill != "point" or len(all_class_names) != 1:
        return []
    return [
        "warning: single-class point mode: --neg-prompts-per-nonempty is inert because non-empty rows have no absent classes.",
        "warning: single-class point mode: --pos-task-prob is inert because non-empty rows only generate positive tasks.",
        "warning: single-class point mode: --max-objects is inert because point requests do not use object caps.",
        "warning: single-class point mode: eval_miou is not meaningful; rank runs by eval_f1 instead.",
    ]


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


def _micro_f1_from_counts(tp: int, fp: int, fn: int) -> float:
    denom = (2 * int(tp)) + int(fp) + int(fn)
    if denom == 0:
        return 1.0
    return (2.0 * float(tp)) / float(denom)


def _selection_metric_key(selection_metric: str, *, prefix: str = "eval") -> str:
    metric = str(selection_metric or "").strip()
    if metric not in SELECTION_METRIC_CHOICES:
        raise ValueError(f"unknown selection metric: {selection_metric}")
    return f"{prefix}_{metric}"


def _selection_metric_value(metrics: Mapping[str, Any], selection_metric: str, *, prefix: str = "eval") -> float:
    return float(metrics.get(_selection_metric_key(selection_metric, prefix=prefix), 0.0))


def _prefix_eval_metrics(metrics: Mapping[str, Any], *, prefix: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in metrics.items():
        if key.startswith("eval_"):
            out[f"{prefix}{key[len('eval_'):]}"] = value
        else:
            out[f"{prefix}{key}"] = value
    return out


def _numeric_eval_payload(metrics: Mapping[str, Any]) -> dict[str, float]:
    return {
        str(key): float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float, np.integer, np.floating))
    }


def _build_async_checkpoint_eval_command(
    *,
    args: argparse.Namespace,
    split_name: str,
    finetune_id: str,
    checkpoint_step: int,
    effective_point_prompt_style: str,
    metrics_json_path: Path,
    records_jsonl_path: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(str(args.async_checkpoint_eval_benchmark_script)).expanduser().resolve()),
        "--env-file",
        str(args.env_file),
        "--api-base",
        str(args.base_url),
        "--split",
        str(split_name),
        "--finetune-id",
        str(finetune_id),
        "--checkpoint-step",
        str(int(checkpoint_step)),
        "--skip-baseline",
        "--skill",
        str(args.skill),
        "--point-prompt-style",
        str(effective_point_prompt_style),
        "--temperature",
        str(float(args.eval_temperature)),
        "--top-p",
        str(float(args.eval_top_p)),
        "--max-tokens",
        str(int(args.max_tokens)),
        "--max-objects",
        str(int(args.max_objects)),
        "--out-json",
        str(metrics_json_path),
        "--records-jsonl",
        str(records_jsonl_path),
        "--checkpoint-fallback-policy",
        "exact",
        "--checkpoint-ready-max-wait-s",
        "300",
        "--checkpoint-ready-poll-interval-s",
        "5",
        "--neg-prompts-per-empty",
        str(int(args.neg_prompts_per_empty)),
        "--neg-prompts-per-nonempty",
        str(int(args.neg_prompts_per_nonempty)),
        "--seed",
        str(int(args.seed)),
    ]
    if bool(args.reasoning if args.eval_reasoning is None else args.eval_reasoning):
        cmd.append("--reasoning")
    else:
        cmd.append("--no-reasoning")
    if bool(args.runtime_tiling):
        cmd.extend(
            [
                "--runtime-tiling",
                "--tile-grid-size",
                str(int(args.tile_grid_size)),
                "--tile-overlap",
                str(float(args.tile_overlap)),
                "--tile-point-merge-radius",
                str(float(args.tile_point_merge_radius)),
                "--tile-box-merge-iou",
                str(float(args.tile_box_merge_iou)),
            ]
        )
    if int(args.eval_max_samples or 0) > 0:
        cmd.extend(["--max-samples", str(int(args.eval_max_samples))])
    if str(args.dataset_path or "").strip():
        cmd.extend(["--dataset-path", str(args.dataset_path)])
    elif str(args.dataset_name or "").strip():
        cmd.extend(["--dataset-name", str(args.dataset_name)])
    if str(args.class_names_file or "").strip():
        cmd.extend(["--class-names-file", str(args.class_names_file)])
    include_classes = list(getattr(args, "include_classes", []) or [])
    exclude_classes = list(getattr(args, "exclude_classes", []) or [])
    if include_classes:
        cmd.extend(["--include-classes", *include_classes])
    if exclude_classes:
        cmd.extend(["--exclude-classes", *exclude_classes])
    return cmd


def _ingest_async_checkpoint_eval_results(
    *,
    args: argparse.Namespace,
    run: Any,
    results: list[CheckpointEvalResult],
    log_step: int,
    baseline_eval_metric: Optional[float],
    baseline_eval_tp: Optional[float],
    best_metric: Optional[float],
    best_step: Optional[int],
    best_checkpoint_step: Optional[int],
    latest_checkpoint_step: Optional[int],
    recall_gate_pass: Optional[bool],
    recall_gate_eval_step: Optional[int],
    recall_gate_eval_tp: Optional[float],
    recall_gate_min_tp: Optional[float],
) -> tuple[
    Optional[float],
    Optional[int],
    Optional[int],
    Optional[int],
    Optional[bool],
    Optional[int],
    Optional[float],
    Optional[float],
    int,
]:
    success_count = 0
    metric_key = _selection_metric_key(args.selection_metric)
    for result in results:
        source_step = int(result.metadata.get("step_for_log", result.checkpoint_step))
        arrival_step = int(log_step)
        if result.status != "succeeded" or result.metrics_payload is None:
            print(
                f"async checkpoint eval failed step={source_step} "
                f"checkpoint_step={result.checkpoint_step} log={result.stdout_log_path}"
            )
            continue
        metrics = dict(result.metrics_payload)
        numeric_payload = _numeric_eval_payload(metrics)
        numeric_payload["async_eval_source_step"] = int(source_step)
        numeric_payload["async_eval_checkpoint_step"] = int(result.checkpoint_step)
        if baseline_eval_metric is not None:
            delta_key = f"{metric_key}_delta_vs_baseline"
            numeric_payload[delta_key] = (
                _selection_metric_value(metrics, args.selection_metric) - baseline_eval_metric
            )
        wandb.log(numeric_payload, step=arrival_step)
        latest_checkpoint_step = int(result.checkpoint_step)
        run.summary["latest_checkpoint_step"] = int(result.checkpoint_step)
        if (
            args.skill == "point"
            and baseline_eval_tp is not None
            and recall_gate_pass is None
            and source_step >= args.recall_gate_step
        ):
            recall_gate_eval_step = int(source_step)
            recall_gate_eval_tp = float(metrics.get("eval_tp", 0.0))
            recall_gate_min_tp = float(baseline_eval_tp) * (1.0 - float(args.recall_drop_threshold))
            recall_gate_pass = bool(recall_gate_eval_tp >= recall_gate_min_tp)
            wandb.log(
                {
                    "recall_gate_step": recall_gate_eval_step,
                    "recall_gate_eval_tp": recall_gate_eval_tp,
                    "recall_gate_min_tp": recall_gate_min_tp,
                    "recall_gate_pass": int(recall_gate_pass),
                    "async_eval_source_step": int(source_step),
                    "async_eval_checkpoint_step": int(result.checkpoint_step),
                },
                step=arrival_step,
            )
            print(
                f"recall gate step {recall_gate_eval_step}: "
                f"tp={recall_gate_eval_tp:.1f} min_tp={recall_gate_min_tp:.1f} "
                f"pass={recall_gate_pass}"
            )
        print(
            f"async eval step {source_step} checkpoint_step={result.checkpoint_step} "
            f"tasks={int(metrics.get('eval_tasks', 0))} "
            f"miou={float(metrics.get('eval_miou', 0.0)):.4f} "
            f"f1={float(metrics.get('eval_f1', 0.0)):.4f} "
            f"macro_f1={float(metrics.get('eval_f1_macro', 0.0)):.4f} "
            f"{args.selection_metric}={_selection_metric_value(metrics, args.selection_metric):.4f} "
            f"logged_at_step={arrival_step}"
        )

        metric = _selection_metric_value(metrics, args.selection_metric)
        if best_metric is None or metric > best_metric:
            best_metric = metric
            best_step = int(source_step)
            best_checkpoint_step = int(result.checkpoint_step)
            run.summary["best_step"] = int(best_step)
            run.summary[f"best_{metric_key}"] = float(best_metric)
            run.summary["best_metric_key"] = metric_key
            run.summary["best_checkpoint_step"] = int(best_checkpoint_step)
            run.summary["best_eval_f1"] = float(metrics.get("eval_f1", 0.0))
            run.summary["best_eval_f1_macro"] = float(metrics.get("eval_f1_macro", 0.0))
            run.summary["best_eval_miou"] = float(metrics.get("eval_miou", 0.0))
        success_count += 1
    return (
        best_metric,
        best_step,
        best_checkpoint_step,
        latest_checkpoint_step,
        recall_gate_pass,
        recall_gate_eval_step,
        recall_gate_eval_tp,
        recall_gate_min_tp,
        success_count,
    )


def _resolve_test_split(
    available_splits: list[str],
    *,
    train_split: str,
    val_split: str,
    test_split: str,
) -> str:
    if not available_splits:
        raise ValueError("Dataset exposes no splits.")

    train_base_split = train_split.split("[", 1)[0]
    val_base_split = val_split.split("[", 1)[0]
    blocked = {train_base_split, val_base_split}

    if test_split:
        if test_split in available_splits:
            resolved = test_split
        elif "[" in test_split and "]" in test_split:
            base_split = test_split.split("[", 1)[0]
            if base_split not in available_splits:
                raise ValueError(f"test split '{test_split}' not found. Available splits: {available_splits}")
            resolved = test_split
        else:
            raise ValueError(f"test split '{test_split}' not found. Available splits: {available_splits}")
        if resolved.split("[", 1)[0] in blocked:
            raise ValueError("test split must differ from both train and validation splits")
        return resolved

    for candidate in TEST_SPLIT_CANDIDATES:
        if candidate in available_splits and candidate not in blocked:
            return candidate
    raise ValueError(
        f"Dataset must already contain a held-out test split. Available splits: {available_splits}"
    )


def _empty_rollout(*, skill: str) -> Rollout:
    effective_skill = (skill or "detect").strip().lower()
    if effective_skill == "point":
        output: PointOutput | DetectOutput = PointOutput(points=[])
    else:
        output = DetectOutput(objects=[])
    return Rollout(
        skill=effective_skill,
        finish_reason="stop",
        output=output,
        answer_tokens=[],
        thinking_tokens=[],
        coords=[],
        sizes=[],
    )


def _request_from_task(
    task: TaskSample,
    *,
    skill: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_objects: int,
    reasoning: bool,
) -> DetectRequest | PointRequest:
    effective_skill = (skill or "detect").strip().lower()
    if effective_skill == "point":
        return _ReasoningPointRequest(
            object_name=task.prompt,
            image_url=_to_data_url(task.image, quality=92),
            settings=PointSettings(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            ),
            reasoning=bool(reasoning),
        )
    return _ReasoningDetectRequest(
        object_name=task.prompt,
        image_url=_to_data_url(task.image, quality=92),
        settings=DetectSettings(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            max_objects=max_objects,
        ),
        reasoning=bool(reasoning),
    )


def _tile_requests_for_task(
    task: TaskSample,
    *,
    skill: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_objects: int,
    reasoning: bool,
    runtime_tiling: bool,
    tile_grid_size: int,
    tile_overlap: float,
) -> tuple[list[TileWindow], list[DetectRequest | PointRequest]]:
    if not runtime_tiling:
        return [], [
            _request_from_task(
                task,
                skill=skill,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                max_objects=max_objects,
                reasoning=reasoning,
            )
        ]

    tiled_images = crop_image_to_tiles(task.image, grid_size=tile_grid_size, overlap=tile_overlap)
    windows: list[TileWindow] = []
    requests: list[DetectRequest | PointRequest] = []
    for window, image in tiled_images:
        windows.append(window)
        tiled_task = TaskSample(
            image=image,
            prompt=task.prompt,
            gt_boxes=task.gt_boxes,
            class_name=task.class_name,
            is_positive=task.is_positive,
            source=task.source,
        )
        requests.append(
            _request_from_task(
                tiled_task,
                skill=skill,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                max_objects=max_objects,
                reasoning=reasoning,
            )
        )
    return windows, requests


def _runtime_point(point: PointAnnotation) -> RuntimePoint:
    return RuntimePoint(x=float(point.x), y=float(point.y))


def _runtime_box(box: DetectAnnotation) -> RuntimeBox:
    return RuntimeBox(
        x_min=float(box.x_min),
        y_min=float(box.y_min),
        x_max=float(box.x_max),
        y_max=float(box.y_max),
    )


def _detect_from_runtime_box(box: RuntimeBox) -> DetectAnnotation:
    return DetectAnnotation(
        x_min=float(box.x_min),
        y_min=float(box.y_min),
        x_max=float(box.x_max),
        y_max=float(box.y_max),
    )


def _point_from_runtime_point(point: RuntimePoint) -> PointAnnotation:
    return PointAnnotation(x=float(point.x), y=float(point.y))


def _merge_rollouts_across_tiles(
    *,
    tile_results: list[Any],
    tile_windows: list[TileWindow],
    skill: str,
    expected_rollouts: int,
    point_merge_radius: float,
    box_merge_iou: float,
) -> list[Rollout]:
    effective_skill = (skill or "detect").strip().lower()
    merged_rollouts: list[Rollout] = []

    for rollout_index in range(expected_rollouts):
        template_rollout: Optional[Rollout] = None
        merged_points: list[RuntimePoint] = []
        merged_boxes: list[RuntimeBox] = []
        for tile_idx, result in enumerate(tile_results):
            if result is None or rollout_index >= len(getattr(result, "rollouts", [])):
                continue
            rollout = result.rollouts[rollout_index]
            if template_rollout is None:
                template_rollout = rollout
            output = rollout.output
            window = tile_windows[tile_idx]
            if effective_skill == "point":
                points = output.points if isinstance(output, PointOutput) else []
                for point in points:
                    merged_points.append(map_point_from_tile(_runtime_point(point), window))
            else:
                boxes = output.objects if isinstance(output, DetectOutput) else []
                for box in boxes:
                    mapped = map_box_from_tile(_runtime_box(box), window)
                    if mapped is not None:
                        merged_boxes.append(mapped)

        if effective_skill == "point":
            output = PointOutput(points=[_point_from_runtime_point(point) for point in merge_runtime_points(merged_points, radius=point_merge_radius)])
        else:
            output = DetectOutput(objects=[_detect_from_runtime_box(box) for box in merge_runtime_boxes(merged_boxes, iou_threshold=box_merge_iou)])

        if template_rollout is None:
            merged_rollouts.append(_empty_rollout(skill=effective_skill))
        else:
            merged_rollouts.append(
                Rollout(
                    skill=str(getattr(template_rollout, "skill", effective_skill)),
                    finish_reason=str(getattr(template_rollout, "finish_reason", "stop")),
                    output=output,
                    answer_tokens=list(getattr(template_rollout, "answer_tokens", [])),
                    thinking_tokens=list(getattr(template_rollout, "thinking_tokens", [])),
                    coords=list(getattr(template_rollout, "coords", [])),
                    sizes=list(getattr(template_rollout, "sizes", [])),
                )
            )
    return merged_rollouts


def _update_kl_guard(
    *,
    kl_value: float,
    warning_threshold: float,
    stop_threshold: float,
    stop_consecutive: int,
    consecutive_hits: int,
) -> tuple[int, bool, bool]:
    warn = bool(warning_threshold > 0.0 and kl_value >= warning_threshold)
    if stop_threshold > 0.0 and kl_value >= stop_threshold:
        consecutive_hits += 1
    else:
        consecutive_hits = 0
    stop = bool(stop_threshold > 0.0 and consecutive_hits >= max(1, int(stop_consecutive)))
    return consecutive_hits, warn, stop


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
    if isinstance(body, dict):
        body_text = json.dumps(body, ensure_ascii=True)
    else:
        body_text = str(body)
    lowered = body_text.lower()
    return "reasoning" in lowered and "extra_forbidden" in lowered


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
    reasoning: bool,
    runtime_tiling: bool,
    tile_grid_size: int,
    tile_overlap: float,
    tile_point_merge_radius: float,
    tile_box_merge_iou: float,
) -> dict[str, float]:
    tasks: list[TaskSample] = []
    total = 0
    total_f1 = 0.0
    total_miou = 0.0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    positive_tasks = 0
    positive_f1_sum = 0.0
    positive_tp = 0
    positive_fp = 0
    positive_fn = 0
    negative_tasks = 0
    negative_f1_sum = 0.0
    negative_tp = 0
    negative_fp = 0
    negative_fn = 0
    effective_skill = (skill or "detect").strip().lower()
    reasoning_failure_hint_emitted = False

    def _drain_batch(batch: list[TaskSample]) -> None:
        nonlocal total, total_f1, total_miou, total_tp, total_fp, total_fn
        nonlocal positive_tasks, positive_f1_sum, positive_tp, positive_fp, positive_fn
        nonlocal negative_tasks, negative_f1_sum, negative_tp, negative_fp, negative_fn
        nonlocal reasoning_failure_hint_emitted
        if not batch:
            return
        chunk_size = max(1, min(max_workers, len(batch)))
        for offset in range(0, len(batch), chunk_size):
            chunk = batch[offset : offset + chunk_size]
            flat_requests: list[DetectRequest | PointRequest] = []
            tile_windows_per_task: list[list[TileWindow]] = []
            request_counts: list[int] = []
            for item in chunk:
                tile_windows, requests = _tile_requests_for_task(
                    item,
                    skill=effective_skill,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    max_objects=max_objects,
                    reasoning=reasoning,
                    runtime_tiling=runtime_tiling,
                    tile_grid_size=tile_grid_size,
                    tile_overlap=tile_overlap,
                )
                tile_windows_per_task.append(tile_windows)
                request_counts.append(len(requests))
                flat_requests.extend(requests)
            try:
                results = _rollouts_batch_with_retry(
                    finetune=finetune,
                    requests=flat_requests,
                    num_rollouts=1,
                    max_workers=min(max_workers, len(flat_requests)),
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
                        "hint: this may indicate reasoning is not supported for detect/point tuning rollouts in "
                        "your current API path. Try --no-reasoning --no-eval-reasoning."
                    )
                    reasoning_failure_hint_emitted = True
                continue

            if len(results) != len(flat_requests):
                print(
                    f"warning: eval returned {len(results)} results for {len(flat_requests)} requests; "
                    "only aligned results are scored."
                )

            result_index = 0
            for item, tile_windows, request_count in zip(chunk, tile_windows_per_task, request_counts):
                task_results = list(results[result_index : result_index + request_count])
                result_index += request_count

                if runtime_tiling:
                    merged_rollouts = _merge_rollouts_across_tiles(
                        tile_results=task_results,
                        tile_windows=tile_windows,
                        skill=effective_skill,
                        expected_rollouts=1,
                        point_merge_radius=tile_point_merge_radius,
                        box_merge_iou=tile_box_merge_iou,
                    )
                    rollout0 = merged_rollouts[0] if merged_rollouts else _empty_rollout(skill=effective_skill)
                else:
                    if not task_results or not task_results[0].rollouts:
                        rollout0 = _empty_rollout(skill=effective_skill)
                    else:
                        rollout0 = task_results[0].rollouts[0]

                if not task_results or not rollout0:
                    if effective_skill == "point":
                        pred_points: list[PointAnnotation] = []
                        f1_value = _reward_f1_points(pred_points, item.gt_boxes)
                        tp, fp, fn = _count_tp_fp_fn_points(pred_points, item.gt_boxes)
                    else:
                        pred_boxes: list[DetectAnnotation] = []
                        f1_value = _reward_f1(pred_boxes, item.gt_boxes)
                        miou_value = _reward_miou(pred_boxes, item.gt_boxes)
                        tp, fp, fn = _count_tp_fp_fn(pred_boxes, item.gt_boxes)
                        total_miou += miou_value
                    total_f1 += f1_value
                    total_tp += tp
                    total_fp += fp
                    total_fn += fn
                    if item.is_positive:
                        positive_tasks += 1
                        positive_f1_sum += f1_value
                        positive_tp += tp
                        positive_fp += fp
                        positive_fn += fn
                    else:
                        negative_tasks += 1
                        negative_f1_sum += f1_value
                        negative_tp += tp
                        negative_fp += fp
                        negative_fn += fn
                    total += 1
                    continue

                if effective_skill == "point":
                    output = rollout0.output
                    pred_points = output.points if isinstance(output, PointOutput) else []
                    f1_value = _reward_f1_points(pred_points, item.gt_boxes)
                    tp, fp, fn = _count_tp_fp_fn_points(pred_points, item.gt_boxes)
                else:
                    output = rollout0.output
                    pred_boxes = output.objects if isinstance(output, DetectOutput) else []
                    f1_value = _reward_f1(pred_boxes, item.gt_boxes)
                    miou_value = _reward_miou(pred_boxes, item.gt_boxes)
                    tp, fp, fn = _count_tp_fp_fn(pred_boxes, item.gt_boxes)
                    total_miou += miou_value
                total_f1 += f1_value
                total_tp += tp
                total_fp += fp
                total_fn += fn
                if item.is_positive:
                    positive_tasks += 1
                    positive_f1_sum += f1_value
                    positive_tp += tp
                    positive_fp += fp
                    positive_fn += fn
                else:
                    negative_tasks += 1
                    negative_f1_sum += f1_value
                    negative_tp += tp
                    negative_fp += fp
                    negative_fn += fn
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
            "eval_positive_tasks": 0,
            "eval_positive_f1": 0.0,
            "eval_positive_tp": 0,
            "eval_positive_fp": 0,
            "eval_positive_fn": 0,
            "eval_negative_tasks": 0,
            "eval_negative_f1": 0.0,
            "eval_negative_tp": 0,
            "eval_negative_fp": 0,
            "eval_negative_fn": 0,
        }

    micro_f1 = _micro_f1_from_counts(total_tp, total_fp, total_fn)
    return {
        "eval_tasks": total,
        "eval_f1": micro_f1,
        "eval_f1_macro": total_f1 / total,
        "eval_miou": total_miou / total,
        "eval_tp": total_tp,
        "eval_fp": total_fp,
        "eval_fn": total_fn,
        "eval_positive_tasks": positive_tasks,
        "eval_positive_f1": (_micro_f1_from_counts(positive_tp, positive_fp, positive_fn) if positive_tasks else 0.0),
        "eval_positive_f1_macro": (positive_f1_sum / positive_tasks) if positive_tasks else 0.0,
        "eval_positive_tp": positive_tp,
        "eval_positive_fp": positive_fp,
        "eval_positive_fn": positive_fn,
        "eval_negative_tasks": negative_tasks,
        "eval_negative_f1": (_micro_f1_from_counts(negative_tp, negative_fp, negative_fn) if negative_tasks else 0.0),
        "eval_negative_f1_macro": (negative_f1_sum / negative_tasks) if negative_tasks else 0.0,
        "eval_negative_tp": negative_tp,
        "eval_negative_fp": negative_fp,
        "eval_negative_fn": negative_fn,
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

    from_script = (_repo_relative(path.as_posix())).resolve()
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
        args.base_url = os.environ.get("TUNA_BASE_URL", "https://api.moondream.ai/v1")
    return args


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    raw_argv = list(argv) if argv is not None else list(sys.argv[1:])
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    pre_args, _ = pre_parser.parse_known_args(raw_argv)
    config_path = _resolve_config_path(pre_args.config)
    config = _load_json_config(config_path)

    parser = argparse.ArgumentParser(description="RL finetune Moondream for PI&D icon detection (class-conditional).")
    parser.add_argument("--config", default=str(config_path))
    parser.add_argument("--env-file", default=str(_repo_relative(".env")))
    parser.add_argument("--api-key", default="")
    parser.add_argument("--api-key-env-var", default="MOONDREAM_API_KEY")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"))
    parser.add_argument("--base-url", default="")

    parser.add_argument("--dataset-path", default="")
    parser.add_argument("--dataset-name", default="maxs-m87/pid-icons-merged" )
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--val-split",
        default="",
        help="Validation split name. If omitted, uses dataset validation/val/dev/test/post_val when present; otherwise auto-splits.",
    )
    parser.add_argument(
        "--test-split",
        default="",
        help="Held-out test split name. If omitted, uses test/post_val when available.",
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
        "--selection-metric",
        choices=SELECTION_METRIC_CHOICES,
        default="",
        help="Validation metric used to select the best checkpoint. Defaults to F1 for point and reward-aligned for detect.",
    )
    parser.add_argument(
        "--point-prompt-style",
        choices=["detect_phrase", "class_name"],
        default="detect_phrase",
        help="Prompt style for class-conditional point training/eval tasks.",
    )
    parser.add_argument(
        "--runtime-tiling",
        action="store_true",
        help="Run training and eval on runtime-generated tiles instead of the full image in one request.",
    )
    parser.add_argument("--tile-grid-size", type=int, default=3)
    parser.add_argument("--tile-overlap", type=float, default=0.10)
    parser.add_argument("--tile-point-merge-radius", type=float, default=0.015)
    parser.add_argument("--tile-box-merge-iou", type=float, default=0.50)

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
    parser.add_argument(
        "--kl-warning-threshold",
        type=float,
        default=0.0,
        help="Log a warning when a train step KL reaches this threshold. <=0 disables warnings.",
    )
    parser.add_argument(
        "--kl-stop-threshold",
        type=float,
        default=0.0,
        help="Stop training early when train-step KL reaches this threshold for N consecutive updates. <=0 disables stopping.",
    )
    parser.add_argument(
        "--kl-stop-consecutive",
        type=int,
        default=1,
        help="How many consecutive KL threshold hits are required before early stop.",
    )
    parser.add_argument("--augment-prob", type=float, default=0.5)
    parser.add_argument("--eval-temperature", type=float, default=0.0)
    parser.add_argument("--eval-top-p", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-max-samples", type=int, default=200)
    parser.add_argument("--eval-batch-size", type=int, default=20)
    parser.add_argument(
        "--run-final-test",
        action="store_true",
        help="Evaluate the best validation checkpoint on a held-out test split after training.",
    )
    async_eval_group = parser.add_mutually_exclusive_group()
    async_eval_group.add_argument(
        "--async-checkpoint-eval",
        dest="async_checkpoint_eval",
        action="store_true",
        help="Benchmark saved checkpoints in detached subprocesses during periodic eval.",
    )
    async_eval_group.add_argument(
        "--no-async-checkpoint-eval",
        dest="async_checkpoint_eval",
        action="store_false",
        help="Run periodic validation inline against the live finetune.",
    )
    parser.set_defaults(async_checkpoint_eval=False)
    parser.add_argument(
        "--async-checkpoint-eval-dir",
        default=str(_repo_relative("outputs", "async_checkpoint_eval")),
    )
    parser.add_argument("--async-checkpoint-eval-max-inflight", type=int, default=1)
    async_drain_group = parser.add_mutually_exclusive_group()
    async_drain_group.add_argument(
        "--async-checkpoint-eval-drain-on-exit",
        dest="async_checkpoint_eval_drain_on_exit",
        action="store_true",
    )
    async_drain_group.add_argument(
        "--no-async-checkpoint-eval-drain-on-exit",
        dest="async_checkpoint_eval_drain_on_exit",
        action="store_false",
    )
    parser.set_defaults(async_checkpoint_eval_drain_on_exit=True)

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
    parser.add_argument(
        "--async-checkpoint-eval-benchmark-script",
        default=str(_repo_relative("benchmark_pid_icons.py").resolve()),
        help=argparse.SUPPRESS,
    )
    option_to_dest: dict[str, str] = {}
    for action in parser._actions:
        if not action.option_strings:
            continue
        for opt in action.option_strings:
            option_to_dest[opt] = action.dest
    overridden_dests = {option_to_dest[arg] for arg in raw_argv if arg in option_to_dest}
    config_cli_args = _config_to_cli_args(
        parser,
        config,
        config_path=config_path,
        overridden_dests=overridden_dests,
    )
    args = parser.parse_args(config_cli_args + raw_argv)
    args.config = str(_resolve_config_path(args.config))

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

    return args


def run(args: argparse.Namespace) -> None:
    args = _resolve_runtime_env(args)
    args.async_checkpoint_eval_dir = str(_resolve_config_path(str(args.async_checkpoint_eval_dir)))
    if not args.api_key:
        raise ValueError("MOONDREAM_API_KEY is required")
    if args.finetune_id and args.finetune_name:
        raise ValueError("Provide either --finetune-id or --finetune-name, not both")
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
    if args.kl_warning_threshold < 0.0:
        raise ValueError("--kl-warning-threshold must be >= 0")
    if args.kl_stop_threshold < 0.0:
        raise ValueError("--kl-stop-threshold must be >= 0")
    if args.kl_stop_consecutive <= 0:
        raise ValueError("--kl-stop-consecutive must be > 0")
    if args.tile_grid_size <= 0:
        raise ValueError("--tile-grid-size must be > 0")
    if not (0.0 <= args.tile_overlap < 1.0):
        raise ValueError("--tile-overlap must be in [0, 1)")
    if args.tile_point_merge_radius < 0.0:
        raise ValueError("--tile-point-merge-radius must be >= 0")
    if not (0.0 <= args.tile_box_merge_iou <= 1.0):
        raise ValueError("--tile-box-merge-iou must be in [0, 1]")
    if args.async_checkpoint_eval_max_inflight <= 0:
        raise ValueError("--async-checkpoint-eval-max-inflight must be > 0")
    if args.run_final_test and args.eval_every <= 0:
        raise ValueError("--run-final-test requires --eval-every > 0 so a best validation checkpoint can be selected")

    eval_reasoning = bool(args.reasoning) if args.eval_reasoning is None else bool(args.eval_reasoning)
    selection_metric = str(args.selection_metric or "").strip()
    if not selection_metric:
        if args.skill == "point":
            selection_metric = "f1"
        else:
            selection_metric = "miou" if args.reward_metric == "miou" else "f1"
    if selection_metric not in SELECTION_METRIC_CHOICES:
        raise ValueError(f"--selection-metric must be one of {SELECTION_METRIC_CHOICES}")
    if args.skill == "point" and selection_metric == "miou":
        print("warning: --selection-metric=miou is not meaningful for point runs; using f1.")
        selection_metric = "f1"
    args.selection_metric = selection_metric
    off_policy_injection_allowed = bool(
        args.off_policy and (not args.reasoning or args.allow_off_policy_with_reasoning) and not args.runtime_tiling
    )
    if args.off_policy and args.reasoning and not args.allow_off_policy_with_reasoning:
        print(
            "warning: off-policy injection is disabled while --reasoning is enabled. "
            "Set --allow-off-policy-with-reasoning to override."
        )
    if args.off_policy and args.runtime_tiling:
        print(
            "warning: off-policy injection is disabled while --runtime-tiling is enabled. "
            "Merged rewards are still used for runtime-tiled training."
        )

    dataset_path = args.dataset_path.strip()
    dataset_name = args.dataset_name.strip()
    use_local = bool(dataset_path)

    all_class_names = _load_class_names(args.class_names_file, dataset_path if use_local else None)
    class_catalog = _load_class_catalog(args.class_names_file, dataset_path if use_local else None)
    for warning_text in _single_class_point_mode_warnings(
        skill=args.skill,
        all_class_names=all_class_names,
    ):
        print(warning_text)

    rng = random.Random(args.seed)
    rng_np = np.random.default_rng(args.seed)
    augment_config = _default_augment_config()
    usage = UsageStats()
    total_train_rows: Optional[int] = None
    total_val_rows: Optional[int] = None
    discovery_train_rows_consumed = 0

    eval_rows_factory: Callable[[], Iterable[dict]] = lambda: iter(())
    has_eval_rows = False
    test_rows_factory: Callable[[], Iterable[dict]] = lambda: iter(())
    has_test_rows = False
    train_split = args.split
    requested_val_split = args.val_split.strip()
    requested_test_split = args.test_split.strip()
    val_split: Optional[str] = requested_val_split or None
    test_split: Optional[str] = requested_test_split or None
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
            if args.run_final_test or requested_test_split:
                if not val_split:
                    raise ValueError("--run-final-test requires an explicit validation split when using a local Dataset")
                test_split = _resolve_test_split(
                    list(dataset_obj.keys()),
                    train_split=train_split,
                    val_split=val_split,
                    test_split=requested_test_split,
                )
                has_test_rows = True

                def _local_test_rows() -> Iterable[dict]:
                    return _iter_local_rows_once(dataset_obj, test_split)

                test_rows_factory = _local_test_rows
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
        try:
            available_hf_splits = list(get_dataset_split_names(dataset_name, token=args.hf_token))
        except Exception:
            available_hf_splits = []
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
        if args.run_final_test or requested_test_split:
            if not val_split:
                raise ValueError("--run-final-test requires a validation split.")
            test_split = _resolve_test_split(
                available_hf_splits,
                train_split=train_split,
                val_split=val_split,
                test_split=requested_test_split,
            )
            has_test_rows = True

            def _hf_test_rows() -> Iterable[dict]:
                return _iter_hf_rows_once(dataset_name, test_split, args.hf_token)

            test_rows_factory = _hf_test_rows

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
        f"off_policy={args.off_policy} off_policy_injection_allowed={off_policy_injection_allowed} "
        f"reasoning_train={bool(args.reasoning)} reasoning_eval={eval_reasoning} "
        f"runtime_tiling={bool(args.runtime_tiling)} selection_metric={args.selection_metric} "
        f"run_final_test={bool(args.run_final_test)}"
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
            "config": args.config,
            "finetune_id": finetune.finetune_id,
            "dataset_path": dataset_path or None,
            "dataset_name": dataset_name or None,
            "train_split": train_split,
            "val_split": val_split,
            "test_split": test_split,
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
            "reasoning": bool(args.reasoning),
            "eval_reasoning": bool(eval_reasoning),
            "eval_reasoning_override": args.eval_reasoning,
            "point_prompt_style": args.point_prompt_style,
            "effective_point_prompt_style": effective_point_prompt_style,
            "reward_metric": args.reward_metric,
            "selection_metric": args.selection_metric,
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
            "runtime_tiling": bool(args.runtime_tiling),
            "tile_grid_size": int(args.tile_grid_size),
            "tile_overlap": float(args.tile_overlap),
            "tile_point_merge_radius": float(args.tile_point_merge_radius),
            "tile_box_merge_iou": float(args.tile_box_merge_iou),
            "run_final_test": bool(args.run_final_test),
            "async_checkpoint_eval": bool(args.async_checkpoint_eval),
            "async_checkpoint_eval_dir": str(args.async_checkpoint_eval_dir),
            "async_checkpoint_eval_max_inflight": int(args.async_checkpoint_eval_max_inflight),
            "async_checkpoint_eval_drain_on_exit": bool(args.async_checkpoint_eval_drain_on_exit),
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
    best_checkpoint_step: Optional[int] = None
    latest_checkpoint_step: Optional[int] = None
    successful_updates = args.resume_step
    baseline_eval_metric: Optional[float] = None
    baseline_eval_tp: Optional[float] = None
    recall_gate_pass: Optional[bool] = None
    recall_gate_eval_step: Optional[int] = None
    recall_gate_eval_tp: Optional[float] = None
    recall_gate_min_tp: Optional[float] = None
    eval_events_logged = 0
    kl_consecutive_hits = 0
    stopped_early = False
    early_stop_reason = ""
    async_eval_jobs: list[Any] = []
    async_eval_success_count = 0

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
                reasoning=eval_reasoning,
                runtime_tiling=bool(args.runtime_tiling),
                tile_grid_size=int(args.tile_grid_size),
                tile_overlap=float(args.tile_overlap),
                tile_point_merge_radius=float(args.tile_point_merge_radius),
                tile_box_merge_iou=float(args.tile_box_merge_iou),
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
            metric_key = _selection_metric_key(args.selection_metric)
            baseline_eval_metric = _selection_metric_value(baseline_metrics, args.selection_metric)
            baseline_eval_tp = float(baseline_metrics.get("eval_tp", 0.0))
            run.summary[f"baseline_{metric_key}"] = baseline_eval_metric
            run.summary["baseline_eval_tp"] = baseline_eval_tp
            run.summary["baseline_metric_key"] = metric_key
            print(
                f"baseline eval step {baseline_step} tasks={baseline_metrics['eval_tasks']} "
                f"miou={baseline_metrics['eval_miou']:.4f} f1={baseline_metrics['eval_f1']:.4f} "
                f"macro_f1={baseline_metrics['eval_f1_macro']:.4f} "
                f"pos_f1={baseline_metrics['eval_positive_f1']:.4f} "
                f"neg_f1={baseline_metrics['eval_negative_f1']:.4f}"
            )

    for step in range(args.num_steps):
        global_step = args.resume_step + step
        step_start = time.monotonic()

        if args.async_checkpoint_eval:
            async_eval_jobs, completed_async_results = poll_checkpoint_eval_jobs(async_eval_jobs)
            (
                best_metric,
                best_step,
                best_checkpoint_step,
                latest_checkpoint_step,
                recall_gate_pass,
                recall_gate_eval_step,
                recall_gate_eval_tp,
                recall_gate_min_tp,
                completed_successes,
            ) = _ingest_async_checkpoint_eval_results(
                args=args,
                run=run,
                results=completed_async_results,
                log_step=int(global_step),
                baseline_eval_metric=baseline_eval_metric,
                baseline_eval_tp=baseline_eval_tp,
                best_metric=best_metric,
                best_step=best_step,
                best_checkpoint_step=best_checkpoint_step,
                latest_checkpoint_step=latest_checkpoint_step,
                recall_gate_pass=recall_gate_pass,
                recall_gate_eval_step=recall_gate_eval_step,
                recall_gate_eval_tp=recall_gate_eval_tp,
                recall_gate_min_tp=recall_gate_min_tp,
            )
            async_eval_success_count += int(completed_successes)

        batch = [_next_task() for _ in range(args.batch_size)]
        logical_requests: list[DetectRequest | PointRequest] = []
        task_requests_per_item: list[list[DetectRequest | PointRequest]] = []
        tile_windows_per_item: list[list[TileWindow]] = []
        flat_requests: list[DetectRequest | PointRequest] = []
        for item in batch:
            logical_requests.append(
                _request_from_task(
                    item,
                    skill=args.skill,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens,
                    max_objects=args.max_objects,
                    reasoning=bool(args.reasoning),
                )
            )
            tile_windows, task_requests = _tile_requests_for_task(
                item,
                skill=args.skill,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                max_objects=args.max_objects,
                reasoning=bool(args.reasoning),
                runtime_tiling=bool(args.runtime_tiling),
                tile_grid_size=int(args.tile_grid_size),
                tile_overlap=float(args.tile_overlap),
            )
            task_requests_per_item.append(task_requests)
            tile_windows_per_item.append(tile_windows)
            flat_requests.extend(task_requests)

        try:
            rollout_start = time.monotonic()
            results = _rollouts_batch_with_retry(
                finetune=finetune,
                requests=flat_requests,
                num_rollouts=args.group_size,
                max_workers=min(args.max_workers, len(flat_requests)),
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
        if len(results) != len(flat_requests):
            print(
                f"warning: train step {global_step} returned {len(results)} results for "
                f"{len(flat_requests)} requests; only aligned results are used."
            )

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
        result_index = 0
        for idx, item in enumerate(batch):
            task_requests = task_requests_per_item[idx]
            tile_windows = tile_windows_per_item[idx]
            task_results = list(results[result_index : result_index + len(task_requests)])
            result_index += len(task_requests)

            if args.runtime_tiling:
                merged_rollouts = _merge_rollouts_across_tiles(
                    tile_results=task_results,
                    tile_windows=tile_windows,
                    skill=args.skill,
                    expected_rollouts=args.group_size,
                    point_merge_radius=float(args.tile_point_merge_radius),
                    box_merge_iou=float(args.tile_box_merge_iou),
                )
                rewards = _rewards_for_rollouts(
                    merged_rollouts,
                    item.gt_boxes,
                    skill=args.skill,
                    reward_metric=args.reward_metric,
                    fn_penalty_exponent=args.fn_penalty_exponent,
                    fp_penalty_exponent=args.fp_penalty_exponent,
                    neg_reward_weight=args.neg_reward_weight,
                )
                if args.skill == "point":
                    for rollout in merged_rollouts:
                        output = rollout.output
                        pred_points = output.points if isinstance(output, PointOutput) else []
                        tp, fp, fn = _count_tp_fp_fn_points(pred_points, item.gt_boxes)
                        train_tp += tp
                        train_fp += fp
                        train_fn += fn
                else:
                    for rollout in merged_rollouts:
                        output = rollout.output
                        pred_boxes = output.objects if isinstance(output, DetectOutput) else []
                        tp, fp, fn = _count_tp_fp_fn(pred_boxes, item.gt_boxes)
                        train_tp += tp
                        train_fp += fp
                        train_fn += fn

                for tile_request, tile_result in zip(task_requests, task_results):
                    rollouts = list(tile_result.rollouts)
                    if not rollouts:
                        continue
                    group_request = RolloutsRequest(
                        finetune_id=finetune.finetune_id,
                        num_rollouts=len(rollouts),
                        request=tile_request,
                        ground_truth=getattr(tile_result.request, "ground_truth", None),
                        org_id=getattr(tile_result.request, "org_id", None),
                    )
                    groups.append(TrainStepGroup(request=group_request, rollouts=rollouts, rewards=list(rewards[: len(rollouts)])))
                all_rewards.extend(rewards)
                continue

            result = task_results[0] if task_results else None
            rollouts = list(result.rollouts) if result is not None else []
            rewards = _rewards_for_rollouts(
                rollouts,
                item.gt_boxes,
                skill=args.skill,
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
            elif args.off_policy and rewards and rollouts and not off_policy_injection_allowed:
                off_policy_skipped_reasoning_guard += 1

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

            request_obj = logical_requests[idx]
            group_request = RolloutsRequest(
                finetune_id=finetune.finetune_id,
                num_rollouts=len(rollouts),
                request=request_obj,
                ground_truth=getattr(result.request, "ground_truth", None) if result is not None else None,
                org_id=getattr(result.request, "org_id", None) if result is not None else None,
            )
            groups.append(TrainStepGroup(request=group_request, rollouts=rollouts, rewards=rewards))
            all_rewards.extend(rewards)

        if not groups:
            print(f"train step {global_step}: no valid rollout groups after alignment; skipping step")
            continue

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
        kl_value = float(train_out.kl or 0.0)
        kl_consecutive_hits, kl_warning_triggered, kl_stop_triggered = _update_kl_guard(
            kl_value=kl_value,
            warning_threshold=float(args.kl_warning_threshold),
            stop_threshold=float(args.kl_stop_threshold),
            stop_consecutive=int(args.kl_stop_consecutive),
            consecutive_hits=kl_consecutive_hits,
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
                "kl": kl_value,
                "router_kl": train_out.router_kl if train_out.router_kl is not None else 0.0,
                "grad_norm": train_out.grad_norm if train_out.grad_norm is not None else 0.0,
                "kl_warning_triggered": int(kl_warning_triggered),
                "kl_stop_triggered": int(kl_stop_triggered),
                "kl_stop_consecutive_hits": kl_consecutive_hits,
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
            f"step {global_step} reward={reward_mean:.4f} kl={kl_value:.4f} "
            f"train_p={train_precision:.3f} train_r={train_recall:.3f} "
            f"pos={pos_tasks} neg={neg_tasks} rollout_s={(rollout_end-rollout_start):.2f} "
            f"train_s={(train_end-train_start):.2f} total_s={total_s:.2f} "
            f"offp={off_policy_injected_total}/{off_policy_considered} "
            f"offp_guarded={off_policy_skipped_reasoning_guard} updates={successful_updates}"
        )
        if kl_warning_triggered:
            print(
                f"warning: step {global_step} kl={kl_value:.4f} exceeded "
                f"warning threshold {float(args.kl_warning_threshold):.4f}"
            )
        if kl_stop_triggered:
            stopped_early = True
            early_stop_reason = (
                f"kl={kl_value:.4f} reached stop threshold {float(args.kl_stop_threshold):.4f} "
                f"for {kl_consecutive_hits} consecutive update(s)"
            )
            print(f"stopping early at step {global_step}: {early_stop_reason}")
            break
        if args.usage_report_every > 0 and (global_step + 1) % args.usage_report_every == 0:
            usage_summary = _usage_snapshot(usage, total_train_rows=total_train_rows, top_k=args.usage_top_k)
            _print_usage_snapshot(usage_summary, prefix=f"usage step {global_step}")

        if args.eval_every > 0 and has_eval_rows and successful_updates % args.eval_every == 0:
            if args.async_checkpoint_eval:
                checkpoint_step = save_checkpoint_step(
                    finetune=finetune,
                    context=f"periodic eval step={global_step}",
                )
                if checkpoint_step is None:
                    print(
                        f"async checkpoint eval skipped step={global_step}: "
                        "checkpoint save did not return an exact saved step"
                    )
                else:
                    latest_checkpoint_step = int(checkpoint_step)
                    run.summary["latest_checkpoint_step"] = int(latest_checkpoint_step)
                    job = dispatch_checkpoint_eval(
                        trainer=f"pid_icons_{args.skill}",
                        finetune_id=str(finetune.finetune_id),
                        checkpoint_step=int(checkpoint_step),
                        selection_metric=str(args.selection_metric),
                        base_dir=str(args.async_checkpoint_eval_dir),
                        command_builder=lambda metrics_json_path, predictions_jsonl_path, _stdout_log_path: _build_async_checkpoint_eval_command(
                            args=args,
                            split_name=str(val_split),
                            finetune_id=str(finetune.finetune_id),
                            checkpoint_step=int(checkpoint_step),
                            effective_point_prompt_style=effective_point_prompt_style,
                            metrics_json_path=metrics_json_path,
                            records_jsonl_path=predictions_jsonl_path,
                        ),
                        metadata={
                            "step_for_log": int(global_step),
                            "split_name": str(val_split),
                        },
                        env_overrides={
                            "MOONDREAM_API_KEY": str(args.api_key),
                            "HF_TOKEN": str(args.hf_token),
                        },
                        max_inflight=int(args.async_checkpoint_eval_max_inflight),
                        inflight_jobs=async_eval_jobs,
                    )
                    if job is None:
                        print(
                            f"async checkpoint eval skipped step={global_step} checkpoint_step={checkpoint_step} "
                            "reason=max_inflight"
                        )
                    else:
                        async_eval_jobs.append(job)
                        print(
                            f"async checkpoint eval dispatched step={global_step} checkpoint_step={checkpoint_step} "
                            f"job_dir={job.job_dir}"
                        )
            else:
                eval_metrics = _run_and_log_eval(trigger="periodic", step_for_log=global_step)
                if eval_metrics is None:
                    continue
                metric_key = _selection_metric_key(args.selection_metric)
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
                    f"pos_f1={eval_metrics['eval_positive_f1']:.4f} "
                    f"neg_f1={eval_metrics['eval_negative_f1']:.4f} "
                    f"updates={successful_updates}"
                )

                metric = _selection_metric_value(eval_metrics, args.selection_metric)
                if best_metric is None or metric > best_metric:
                    best_metric = metric
                    best_step = global_step
                    checkpoint_step = save_checkpoint_step(
                        finetune=finetune,
                        context=f"best metric checkpoint step={global_step}",
                    )
                    if checkpoint_step is not None:
                        best_checkpoint_step = int(checkpoint_step)
                        latest_checkpoint_step = int(checkpoint_step)
                        run.summary["latest_checkpoint_step"] = int(latest_checkpoint_step)

        if args.save_every > 0 and (global_step + 1) % args.save_every == 0:
            checkpoint_step = save_checkpoint_step(
                finetune=finetune,
                context=f"save_every checkpoint step={global_step}",
            )
            if checkpoint_step is not None:
                latest_checkpoint_step = int(checkpoint_step)
                run.summary["latest_checkpoint_step"] = int(latest_checkpoint_step)

    checkpoint_step = save_checkpoint_step(
        finetune=finetune,
        context="final checkpoint save",
    )
    if checkpoint_step is not None:
        latest_checkpoint_step = int(checkpoint_step)
        run.summary["latest_checkpoint_step"] = int(latest_checkpoint_step)
    if args.async_checkpoint_eval and (bool(args.async_checkpoint_eval_drain_on_exit) or bool(args.run_final_test)):
        completed_async_results = drain_checkpoint_eval_jobs(async_eval_jobs)
        (
            best_metric,
            best_step,
            best_checkpoint_step,
            latest_checkpoint_step,
            recall_gate_pass,
            recall_gate_eval_step,
            recall_gate_eval_tp,
            recall_gate_min_tp,
            completed_successes,
        ) = _ingest_async_checkpoint_eval_results(
            args=args,
            run=run,
            results=completed_async_results,
            log_step=int(args.resume_step + args.num_steps),
            baseline_eval_metric=baseline_eval_metric,
            baseline_eval_tp=baseline_eval_tp,
            best_metric=best_metric,
            best_step=best_step,
            best_checkpoint_step=best_checkpoint_step,
            latest_checkpoint_step=latest_checkpoint_step,
            recall_gate_pass=recall_gate_pass,
            recall_gate_eval_step=recall_gate_eval_step,
            recall_gate_eval_tp=recall_gate_eval_tp,
            recall_gate_min_tp=recall_gate_min_tp,
        )
        async_eval_success_count += int(completed_successes)
    metric_key = _selection_metric_key(args.selection_metric)
    f1_target_value: Optional[float] = None
    f1_target_pass: Optional[bool] = None
    if best_step is not None:
        run.summary["best_step"] = best_step
        run.summary[f"best_{metric_key}"] = best_metric
        run.summary["best_metric_key"] = metric_key
    if best_checkpoint_step is not None:
        run.summary["best_checkpoint_step"] = int(best_checkpoint_step)
    if latest_checkpoint_step is not None:
        run.summary["latest_checkpoint_step"] = int(latest_checkpoint_step)
    if args.skill == "point" and baseline_eval_metric is not None and best_metric is not None:
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
    run.summary["stopped_early"] = int(stopped_early)
    run.summary["early_stop_reason"] = early_stop_reason
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
    run.summary["async_checkpoint_eval_enabled"] = int(bool(args.async_checkpoint_eval))
    run.summary["async_checkpoint_eval_success_count"] = int(async_eval_success_count)
    if args.run_final_test and has_test_rows:
        print("running final test eval after training...")
        test_log_step = max(best_step or 0, args.resume_step + int(args.num_steps))
        if args.async_checkpoint_eval:
            if best_checkpoint_step is None:
                print("warning: final test skipped because no best validation checkpoint was saved.")
                run.summary["test_evaluated"] = 0
            else:
                test_job_dir = (
                    Path(str(args.async_checkpoint_eval_dir)).expanduser().resolve()
                    / "final_test"
                    / str(finetune.finetune_id)
                )
                test_metrics_path = test_job_dir / f"metrics_step{int(best_checkpoint_step):06d}.json"
                test_records_path = test_job_dir / f"records_step{int(best_checkpoint_step):06d}.jsonl"
                cmd = _build_async_checkpoint_eval_command(
                    args=args,
                    split_name=str(test_split),
                    finetune_id=str(finetune.finetune_id),
                    checkpoint_step=int(best_checkpoint_step),
                    effective_point_prompt_style=effective_point_prompt_style,
                    metrics_json_path=test_metrics_path,
                    records_jsonl_path=test_records_path,
                )
                env = dict(os.environ)
                env["MOONDREAM_API_KEY"] = str(args.api_key)
                if str(args.hf_token or "").strip():
                    env["HF_TOKEN"] = str(args.hf_token)
                print("final test benchmark command:")
                print("  " + " ".join(shlex.quote(part) for part in cmd))
                completed = subprocess.run(
                    cmd,
                    check=False,
                    capture_output=True,
                    text=True,
                    env=env,
                )
                if completed.returncode != 0:
                    run.summary["test_evaluated"] = 0
                    run.summary["test_eval_error"] = (
                        f"benchmark exit_code={completed.returncode} checkpoint_step={best_checkpoint_step}"
                    )
                    print(
                        f"final test benchmark failed checkpoint_step={best_checkpoint_step} "
                        f"split={test_split} exit_code={completed.returncode}"
                    )
                    if completed.stdout.strip():
                        print(completed.stdout.strip()[:1200])
                    if completed.stderr.strip():
                        print(completed.stderr.strip()[:1200])
                else:
                    try:
                        test_metrics = json.loads(test_metrics_path.read_text(encoding="utf-8"))
                    except Exception as exc:
                        run.summary["test_evaluated"] = 0
                        run.summary["test_eval_error"] = f"{type(exc).__name__}: {exc}"
                        print(f"final test metrics parse failed: {type(exc).__name__}: {exc}")
                    else:
                        prefixed_test_metrics = _prefix_eval_metrics(test_metrics, prefix="test_")
                        wandb.log(prefixed_test_metrics, step=test_log_step)
                        for key, value in prefixed_test_metrics.items():
                            run.summary[key] = value
                        run.summary["test_evaluated"] = 1
                        try:
                            del run.summary["test_eval_error"]
                        except KeyError:
                            pass
                        print(
                            f"final test checkpoint_step={best_checkpoint_step} split={test_split} "
                            f"tasks={prefixed_test_metrics.get('test_tasks', 0)} "
                            f"miou={float(prefixed_test_metrics.get('test_miou', 0.0)):.4f} "
                            f"f1={float(prefixed_test_metrics.get('test_f1', 0.0)):.4f} "
                            f"macro_f1={float(prefixed_test_metrics.get('test_f1_macro', 0.0)):.4f}"
                        )
        else:
            try:
                test_metrics = _evaluate(
                    finetune=finetune,
                    eval_rows=test_rows_factory(),
                    all_class_names=all_class_names,
                    rng=random.Random(args.seed + 54321),
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
                    reasoning=eval_reasoning,
                    runtime_tiling=bool(args.runtime_tiling),
                    tile_grid_size=int(args.tile_grid_size),
                    tile_overlap=float(args.tile_overlap),
                    tile_point_merge_radius=float(args.tile_point_merge_radius),
                    tile_box_merge_iou=float(args.tile_box_merge_iou),
                )
                prefixed_test_metrics = _prefix_eval_metrics(test_metrics, prefix="test_")
                wandb.log(prefixed_test_metrics, step=test_log_step)
                for key, value in prefixed_test_metrics.items():
                    run.summary[key] = value
                run.summary["test_evaluated"] = 1
            except Exception as exc:
                run.summary["test_evaluated"] = 0
                run.summary["test_eval_error"] = f"{type(exc).__name__}: {exc}"
                print(f"final test eval failed: {type(exc).__name__}: {exc}")
    else:
        run.summary["test_evaluated"] = 0
    run.finish()
    client.close()

    print(
        f"done. finetune_id={finetune.finetune_id} "
        f"best_step={best_step} best_metric={best_metric} "
        f"recall_gate_pass={recall_gate_pass} f1_target_pass={f1_target_pass} "
        f"stopped_early={stopped_early}"
    )


def main(argv: Optional[list[str]] = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
