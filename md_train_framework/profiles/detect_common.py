from __future__ import annotations

from typing import Optional

from md_train_framework.datasets import create_dataset_adapter
from md_train_framework.samples import DetectSample, _detect_boxes, _resolve_image_source, _sample_id, _sample_meta


def build_constant_object_detect_samples(
    config: object,
    *,
    split_name: str,
    object_name: str,
    max_boxes: Optional[int] = None,
) -> list[DetectSample]:
    adapter = create_dataset_adapter(config)  # type: ignore[arg-type]
    samples: list[DetectSample] = []
    for index, row in enumerate(adapter.iter_split(split_name), start=1):
        if not isinstance(row, dict):
            continue
        image_url, image_ref = _resolve_image_source(  # type: ignore[arg-type]
            config,
            row,
            split=split_name,
            dataset_index=index - 1,
        )
        if not image_url and image_ref is None:
            continue
        boxes = tuple(_detect_boxes(row))
        if max_boxes is not None and max_boxes >= 0:
            boxes = boxes[: int(max_boxes)]
        samples.append(
            DetectSample(
                sample_id=_sample_id(row, split=split_name, index=index),
                split=split_name,
                object_name=str(object_name),
                image_url=image_url or "",
                image_ref=image_ref,
                boxes=boxes,
                meta=_sample_meta(row),
            )
        )
    return samples


def profile_object_name(*, default_name: str, override_value: Optional[object]) -> str:
    text = str(override_value or "").strip()
    return text or default_name
