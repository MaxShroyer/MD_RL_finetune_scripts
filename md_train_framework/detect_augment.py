from __future__ import annotations

import io
import random
from dataclasses import dataclass
from typing import Any

from md_train_framework.samples import DetectSample, materialize_image_pil
from tuna_sdk import DetectAnnotation

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:
    from PIL import Image, ImageEnhance
except ModuleNotFoundError:  # pragma: no cover
    Image = None  # type: ignore[assignment]
    ImageEnhance = None  # type: ignore[assignment]


@dataclass(frozen=True)
class DetectAugmentConfig:
    augment_prob: float = 0.0
    flip_p: float = 0.0
    crop_p: float = 0.0
    crop_scale_min: float = 1.0
    crop_scale_max: float = 1.0
    stretch_p: float = 0.0
    stretch_min: float = 1.0
    stretch_max: float = 1.0
    color_p: float = 0.0
    brightness_min: float = 1.0
    brightness_max: float = 1.0
    contrast_min: float = 1.0
    contrast_max: float = 1.0
    saturation_min: float = 1.0
    saturation_max: float = 1.0
    hue_p: float = 0.0
    hue_shift_min: float = 0.0
    hue_shift_max: float = 0.0
    noise_p: float = 0.0
    noise_std_min: float = 0.0
    noise_std_max: float = 0.0
    prevent_positive_to_empty: bool = True
    min_box_area_keep_ratio: float = 0.0
    output_format: str = "JPEG"
    jpeg_quality_min: int = 90
    jpeg_quality_max: int = 90


def build_detect_augment_config(overrides: dict[str, Any]) -> DetectAugmentConfig:
    return DetectAugmentConfig(
        augment_prob=_clamp01(overrides.get("augment_prob", 0.0)),
        flip_p=_clamp01(overrides.get("flip_p", 0.0)),
        crop_p=_clamp01(overrides.get("crop_p", 0.0)),
        crop_scale_min=max(0.05, float(overrides.get("crop_scale_min", 1.0) or 1.0)),
        crop_scale_max=max(0.05, float(overrides.get("crop_scale_max", 1.0) or 1.0)),
        stretch_p=_clamp01(overrides.get("stretch_p", 0.0)),
        stretch_min=max(0.1, float(overrides.get("stretch_min", 1.0) or 1.0)),
        stretch_max=max(0.1, float(overrides.get("stretch_max", 1.0) or 1.0)),
        color_p=_clamp01(overrides.get("color_p", 0.0)),
        brightness_min=max(0.0, float(overrides.get("brightness_min", 1.0) or 1.0)),
        brightness_max=max(0.0, float(overrides.get("brightness_max", 1.0) or 1.0)),
        contrast_min=max(0.0, float(overrides.get("contrast_min", 1.0) or 1.0)),
        contrast_max=max(0.0, float(overrides.get("contrast_max", 1.0) or 1.0)),
        saturation_min=max(0.0, float(overrides.get("saturation_min", 1.0) or 1.0)),
        saturation_max=max(0.0, float(overrides.get("saturation_max", 1.0) or 1.0)),
        hue_p=_clamp01(overrides.get("hue_p", 0.0)),
        hue_shift_min=_resolve_hue_shift(overrides, "hue_shift_min", "hue_shift_deg_min"),
        hue_shift_max=_resolve_hue_shift(overrides, "hue_shift_max", "hue_shift_deg_max"),
        noise_p=_clamp01(overrides.get("noise_p", 0.0)),
        noise_std_min=max(0.0, float(overrides.get("noise_std_min", 0.0) or 0.0)),
        noise_std_max=max(0.0, float(overrides.get("noise_std_max", 0.0) or 0.0)),
        prevent_positive_to_empty=bool(overrides.get("prevent_positive_to_empty", True)),
        min_box_area_keep_ratio=_clamp01(overrides.get("min_box_area_keep_ratio", 0.0)),
        output_format=str(overrides.get("train_image_format", "JPEG") or "JPEG").strip().upper(),
        jpeg_quality_min=max(1, min(100, int(overrides.get("jpeg_quality_min", 90) or 90))),
        jpeg_quality_max=max(1, min(100, int(overrides.get("jpeg_quality_max", 90) or 90))),
    )


def maybe_augment_detect_sample(sample: DetectSample, *, rng: random.Random, config: DetectAugmentConfig) -> DetectSample:
    if Image is None:
        return sample
    image = materialize_image_pil(sample.image_url, sample.image_ref)
    if image is None:
        return sample
    boxes = list(sample.boxes)
    if config.augment_prob > 0.0 and rng.random() < config.augment_prob:
        image, boxes = _apply_augmentations(image=image, boxes=boxes, rng=rng, config=config)
    image_ref = _encode_image_ref(image=image, rng=rng, config=config)
    return DetectSample(
        sample_id=sample.sample_id,
        split=sample.split,
        object_name=sample.object_name,
        image_url="",
        image_ref=image_ref,
        boxes=tuple(boxes),
        meta=dict(sample.meta),
    )


def _apply_augmentations(
    *,
    image: Any,
    boxes: list[DetectAnnotation],
    rng: random.Random,
    config: DetectAugmentConfig,
) -> tuple[Any, list[DetectAnnotation]]:
    if config.crop_p > 0.0 and rng.random() < config.crop_p:
        image, boxes = _random_crop(image=image, boxes=boxes, rng=rng, config=config)
    if config.flip_p > 0.0 and rng.random() < config.flip_p:
        image, boxes = _horizontal_flip(image=image, boxes=boxes)
    if config.stretch_p > 0.0 and rng.random() < config.stretch_p:
        image = _random_stretch(image=image, rng=rng, config=config)
    if ImageEnhance is not None and config.color_p > 0.0 and rng.random() < config.color_p:
        image = _apply_color_jitter(image=image, rng=rng, config=config)
    if np is not None and config.hue_p > 0.0 and rng.random() < config.hue_p:
        image = _hue_shift(image=image, rng=rng, config=config)
    if np is not None and config.noise_p > 0.0 and rng.random() < config.noise_p:
        image = _add_noise(image=image, rng=rng, config=config)
    return image, boxes


def _encode_image_ref(*, image: Any, rng: random.Random, config: DetectAugmentConfig) -> dict[str, Any]:
    buffer = io.BytesIO()
    fmt = "PNG" if config.output_format == "PNG" else "JPEG"
    save_kwargs: dict[str, Any] = {}
    if fmt == "JPEG":
        low = min(config.jpeg_quality_min, config.jpeg_quality_max)
        high = max(config.jpeg_quality_min, config.jpeg_quality_max)
        save_kwargs["quality"] = int(round(rng.uniform(low, high)))
    image.save(buffer, format=fmt, **save_kwargs)
    mime_type = "image/png" if fmt == "PNG" else "image/jpeg"
    return {"bytes": buffer.getvalue(), "mime_type": mime_type}


def _horizontal_flip(*, image: Any, boxes: list[DetectAnnotation]) -> tuple[Any, list[DetectAnnotation]]:
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


def _random_crop(*, image: Any, boxes: list[DetectAnnotation], rng: random.Random, config: DetectAugmentConfig) -> tuple[Any, list[DetectAnnotation]]:
    width, height = image.size
    scale_low = min(config.crop_scale_min, config.crop_scale_max)
    scale_high = max(config.crop_scale_min, config.crop_scale_max)
    crop_w = max(1, int(width * rng.uniform(scale_low, scale_high)))
    crop_h = max(1, int(height * rng.uniform(scale_low, scale_high)))
    if crop_w >= width and crop_h >= height:
        return image, boxes
    left = rng.randint(0, max(0, width - crop_w)) if crop_w < width else 0
    top = rng.randint(0, max(0, height - crop_h)) if crop_h < height else 0
    right = left + crop_w
    bottom = top + crop_h
    kept: list[DetectAnnotation] = []
    for box in boxes:
        x_min = box.x_min * width
        y_min = box.y_min * height
        x_max = box.x_max * width
        y_max = box.y_max * height
        inter_x_min = max(x_min, left)
        inter_y_min = max(y_min, top)
        inter_x_max = min(x_max, right)
        inter_y_max = min(y_max, bottom)
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            continue
        original_area = max(1e-9, (x_max - x_min) * (y_max - y_min))
        kept_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        if kept_area / original_area < max(0.0, config.min_box_area_keep_ratio):
            continue
        kept.append(
            DetectAnnotation(
                x_min=(inter_x_min - left) / crop_w,
                y_min=(inter_y_min - top) / crop_h,
                x_max=(inter_x_max - left) / crop_w,
                y_max=(inter_y_max - top) / crop_h,
            )
        )
    if boxes and config.prevent_positive_to_empty and not kept:
        return image, boxes
    return image.crop((left, top, right, bottom)), kept


def _random_stretch(*, image: Any, rng: random.Random, config: DetectAugmentConfig) -> Any:
    width, height = image.size
    scale_low = min(config.stretch_min, config.stretch_max)
    scale_high = max(config.stretch_min, config.stretch_max)
    scale_x = rng.uniform(scale_low, scale_high)
    scale_y = rng.uniform(scale_low, scale_high)
    stretched = image.resize(
        (max(1, int(round(width * scale_x))), max(1, int(round(height * scale_y)))),
        resample=Image.BICUBIC,
    )
    return stretched.resize((width, height), resample=Image.BICUBIC)


def _apply_color_jitter(*, image: Any, rng: random.Random, config: DetectAugmentConfig) -> Any:
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(config.brightness_min, config.brightness_max))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(config.contrast_min, config.contrast_max))
    return ImageEnhance.Color(image).enhance(rng.uniform(config.saturation_min, config.saturation_max))


def _hue_shift(*, image: Any, rng: random.Random, config: DetectAugmentConfig) -> Any:
    shift = rng.uniform(config.hue_shift_min, config.hue_shift_max)
    if abs(shift) <= 1e-9:
        return image
    hsv = np.array(image.convert("HSV"), copy=True)
    hue_delta = int(round(shift * 255.0))
    hsv[..., 0] = ((hsv[..., 0].astype(np.int16) + hue_delta) % 256).astype(np.uint8)
    return Image.fromarray(hsv, mode="HSV").convert("RGB")


def _add_noise(*, image: Any, rng: random.Random, config: DetectAugmentConfig) -> Any:
    std_low = min(config.noise_std_min, config.noise_std_max)
    std_high = max(config.noise_std_min, config.noise_std_max)
    std = rng.uniform(std_low, std_high)
    if std <= 0.0:
        return image
    arr = np.asarray(image).astype(np.float32)
    noise = np.random.default_rng(rng.randint(0, 2**31 - 1)).normal(0.0, std, size=arr.shape)
    arr = np.clip(arr + noise, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _resolve_hue_shift(overrides: dict[str, Any], value_key: str, deg_key: str) -> float:
    if str(overrides.get(value_key, "")).strip():
        return float(overrides.get(value_key) or 0.0)
    if str(overrides.get(deg_key, "")).strip():
        return float(overrides.get(deg_key) or 0.0) / 360.0
    return 0.0


def _clamp01(value: Any) -> float:
    return max(0.0, min(1.0, float(value or 0.0)))
