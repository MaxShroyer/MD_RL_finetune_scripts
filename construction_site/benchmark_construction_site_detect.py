#!/usr/bin/env python3
"""Thin wrapper around the shared detect benchmark for ConstructionSite."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MDpi_and_d import benchmark_pid_icons as _base

from construction_site.common import (
    DEFAULT_DATASET_NAME,
    DEFAULT_STAGING_API_BASE,
    DETECT_PROMPT_OVERRIDES,
    build_detect_boxes,
    repo_relative,
)

DEFAULT_CONFIG_PATH = repo_relative("configs", "benchmark_construction_site_detect_default.json")
_ORIGINAL_TO_BASE_SAMPLE = _base._to_base_sample


def _prompt_for_class(
    class_name: str,
    *,
    style: str = "detect_phrase",
    prompt_overrides: Optional[dict[str, str]] = None,
) -> str:
    override = str((prompt_overrides or {}).get(class_name, "") or "").strip()
    if override:
        return override
    prompt = str(DETECT_PROMPT_OVERRIDES.get(class_name, "") or "").strip()
    if prompt:
        return prompt
    return class_name


_base._prompt_for_class = _prompt_for_class


def _to_base_sample(row: dict, fallback_id: int):
    if row.get("answer_boxes") is not None:
        return _ORIGINAL_TO_BASE_SAMPLE(row, fallback_id)

    image = row.get("image")
    if image is None:
        return None
    image = image.convert("RGB")
    boxes = [
        _base.ClassBox(
            class_uid=str(item["class_uid"]),
            class_name=str(item["class_name"]),
            box=_base.Box(
                x_min=float(item["x_min"]),
                y_min=float(item["y_min"]),
                x_max=float(item["x_max"]),
                y_max=float(item["y_max"]),
            ),
        )
        for item in build_detect_boxes(row)
    ]
    sample_id = str(row.get("source_image_id") or row.get("image_id") or row.get("id") or fallback_id)
    return _base.BaseSample(image=image, boxes=boxes, sample_id=sample_id)


_base._to_base_sample = _to_base_sample


def parse_args(argv: Optional[list[str]] = None):
    raw_argv = list(argv) if argv is not None else list(os.sys.argv[1:])
    if "--config" not in raw_argv:
        raw_argv = ["--config", str(DEFAULT_CONFIG_PATH), *raw_argv]
    args = _base.parse_args(raw_argv)
    args.skill = "detect"
    if not str(args.api_base).strip():
        args.api_base = DEFAULT_STAGING_API_BASE
    if not str(args.dataset_path).strip() and not str(args.dataset_name).strip():
        args.dataset_name = DEFAULT_DATASET_NAME
    return args


def main(argv: Optional[list[str]] = None) -> None:
    _base.run(parse_args(argv))


if __name__ == "__main__":
    main()
