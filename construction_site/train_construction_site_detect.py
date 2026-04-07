#!/usr/bin/env python3
"""Thin wrapper around the shared detect RL trainer for ConstructionSite."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MDpi_and_d import train_pid_icons as _base

from construction_site.common import DEFAULT_STAGING_API_BASE, DETECT_PROMPT_OVERRIDES, build_detect_boxes, repo_relative

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = repo_relative("configs", "train_construction_site_detect_default.json")
_ORIGINAL_TO_BASE_SAMPLE = _base._to_base_sample
_ALLOWED_DETECT_CLASS_NAMES: set[str] = set()


def _prompt_for_class(class_name: str, *, style: str = "detect_phrase") -> str:
    prompt = str(DETECT_PROMPT_OVERRIDES.get(class_name, "") or "").strip()
    if prompt:
        return prompt
    return class_name


_base._prompt_for_class = _prompt_for_class


def _to_base_sample(row: dict[str, Any]):
    if row.get("answer_boxes") is not None:
        sample = _ORIGINAL_TO_BASE_SAMPLE(row)
    else:
        image = row.get("image")
        if image is None:
            return None
        image = image.convert("RGB")
        boxes = [
            _base.ClassBox(
                class_uid=str(item["class_uid"]),
                class_name=str(item["class_name"]),
                box=_base.DetectAnnotation(
                    x_min=float(item["x_min"]),
                    y_min=float(item["y_min"]),
                    x_max=float(item["x_max"]),
                    y_max=float(item["y_max"]),
                ),
            )
            for item in build_detect_boxes(row)
        ]
        source = str(row.get("source_collection") or row.get("source_dataset") or "construction_site")
        sample = _base.BaseSample(image=image, boxes=boxes, source=source)
    if sample is None or not _ALLOWED_DETECT_CLASS_NAMES:
        return sample
    filtered_boxes = [item for item in sample.boxes if item.class_name in _ALLOWED_DETECT_CLASS_NAMES]
    return _base.BaseSample(image=sample.image, boxes=filtered_boxes, source=sample.source)


_base._to_base_sample = _to_base_sample


def parse_args(argv: Optional[list[str]] = None):
    raw_argv = list(argv) if argv is not None else list(os.sys.argv[1:])
    if "--config" not in raw_argv:
        raw_argv = ["--config", str(DEFAULT_CONFIG_PATH), *raw_argv]
    args = _base.parse_args(raw_argv)
    global _ALLOWED_DETECT_CLASS_NAMES
    dataset_path = str(args.dataset_path or "").strip() or None
    class_catalog = _base._load_class_catalog(str(args.class_names_file or ""), dataset_path)
    _ALLOWED_DETECT_CLASS_NAMES = {class_name for _, class_name in class_catalog if class_name}
    args.skill = "detect"
    if not str(args.base_url).strip():
        args.base_url = DEFAULT_STAGING_API_BASE
    if not args.finetune_id and str(args.finetune_name).startswith("pid-icons-"):
        args.finetune_name = f"construction-site-detect-{_base._random_suffix()}"
    args.async_checkpoint_eval_benchmark_script = str(
        (SCRIPT_DIR / "benchmark_construction_site_detect.py").resolve()
    )
    return args


def main(argv: Optional[list[str]] = None) -> None:
    _base.run(parse_args(argv))


if __name__ == "__main__":
    main()
