#!/usr/bin/env python3
"""Thin wrapper around the shared point RL trainer for local aerial datasets."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]


def _prime_site_wandb() -> None:
    if "wandb" in sys.modules:
        return
    original_sys_path = list(sys.path)
    repo_root_str = str(REPO_ROOT.resolve())
    cwd_str = str(Path.cwd().resolve())
    try:
        sys.path[:] = [
            entry
            for entry in original_sys_path
            if entry
            and str(Path(entry).resolve()) not in {repo_root_str, cwd_str}
        ]
        try:
            importlib.import_module("wandb")
        except ModuleNotFoundError:
            return
    finally:
        sys.path[:] = original_sys_path


_prime_site_wandb()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from MDpi_and_d import train_pid_icons as _base
except ModuleNotFoundError:
    from _DEPICATED_MDpi_and_d import train_pid_icons as _base

from aerial_airport.common import DEFAULT_STAGING_API_BASE, repo_relative

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = repo_relative("configs", "train_aerial_airport_point_default.json")
_ORIGINAL_TO_BASE_SAMPLE = _base._to_base_sample
_ALLOWED_POINT_CLASS_NAMES: set[str] = set()


if not hasattr(_base.wandb, "init"):
    class _WandbRun:
        def __init__(self) -> None:
            self.summary: dict[str, object] = {}

        def finish(self) -> None:
            return

    class _WandbShim:
        @staticmethod
        def init(*args, **kwargs):
            print("wandb package unavailable; continuing without remote logging.")
            return _WandbRun()

        @staticmethod
        def log(*args, **kwargs) -> None:
            return

    _base.wandb = _WandbShim()


def _to_base_sample(row: dict):
    sample = _ORIGINAL_TO_BASE_SAMPLE(row)
    if sample is None or not _ALLOWED_POINT_CLASS_NAMES:
        return sample
    filtered_boxes = [item for item in sample.boxes if item.class_name in _ALLOWED_POINT_CLASS_NAMES]
    return _base.BaseSample(image=sample.image, boxes=filtered_boxes, source=sample.source)


def parse_args(argv: Optional[list[str]] = None):
    raw_argv = list(argv) if argv is not None else list(os.sys.argv[1:])
    if "--config" not in raw_argv:
        raw_argv = ["--config", str(DEFAULT_CONFIG_PATH), *raw_argv]
    _base._to_base_sample = _to_base_sample
    args = _base.parse_args(raw_argv)
    global _ALLOWED_POINT_CLASS_NAMES
    dataset_path = str(args.dataset_path or "").strip() or None
    class_catalog = _base._load_class_catalog(str(args.class_names_file or ""), dataset_path)
    _ALLOWED_POINT_CLASS_NAMES = {class_name for _, class_name in class_catalog if class_name}
    if not str(args.base_url).strip():
        args.base_url = DEFAULT_STAGING_API_BASE
    if not args.finetune_id and str(args.finetune_name).startswith("pid-icons-"):
        args.finetune_name = f"visdrone-vid-point-{_base._random_suffix()}"
    args.async_checkpoint_eval_benchmark_script = str(
        (SCRIPT_DIR / "benchmark_aerial_airport_point.py").resolve()
    )
    return args


def main(argv: Optional[list[str]] = None) -> None:
    _base.run(parse_args(argv))


if __name__ == "__main__":
    main()
