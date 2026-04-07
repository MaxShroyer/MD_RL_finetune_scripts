#!/usr/bin/env python3
"""Thin wrapper around the football detect benchmark for bone fracture detection."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from football_detect import benchmark_football_detect as _base
from football_detect import train_football_detect as _football_base

from bone_fracture.common import (
    DEFAULT_DETECT_HF_DATASET_NAME,
    DEFAULT_STAGING_API_BASE,
    default_prompt_for_class,
    discover_class_names,
    normalize_class_name,
    parse_box_element_annotations,
    repo_relative,
)

_football_base.default_prompt_for_class = default_prompt_for_class
_football_base.discover_class_names = discover_class_names
_football_base.normalize_class_name = normalize_class_name
_football_base.parse_box_element_annotations = parse_box_element_annotations

DEFAULT_CONFIG_PATH = repo_relative("configs", "benchmark_bone_fracture_detect_default.json")


def parse_args(argv: Optional[list[str]] = None):
    raw_argv = list(argv) if argv is not None else list(os.sys.argv[1:])
    if "--config" not in raw_argv and DEFAULT_CONFIG_PATH.exists():
        raw_argv = ["--config", str(DEFAULT_CONFIG_PATH), *raw_argv]
    if "--base-url" not in raw_argv and "--api-base" not in raw_argv:
        raw_argv = ["--base-url", str(DEFAULT_STAGING_API_BASE), *raw_argv]
    if "--dataset-name" not in raw_argv and "--dataset-path" not in raw_argv:
        raw_argv = ["--dataset-name", str(DEFAULT_DETECT_HF_DATASET_NAME), *raw_argv]
    return raw_argv


def main(argv: Optional[list[str]] = None) -> None:
    _base.main(parse_args(argv))


if __name__ == "__main__":
    main()
