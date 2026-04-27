#!/usr/bin/env python3
"""Thin wrapper around the shared detect RL trainer for Inspector MD."""

from __future__ import annotations

import importlib
import io
import os
import sys
from pathlib import Path
from typing import Optional

from PIL import Image

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
            if entry and str(Path(entry).resolve()) not in {repo_root_str, cwd_str}
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

from inspector_md.common import DEFAULT_BASE_URL, repo_relative, resolve_inspector_dataset_path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = repo_relative("configs", "train_inspector_detect_default.json")
_ORIGINAL_TO_BASE_SAMPLE = _base._to_base_sample
_ALLOWED_CLASS_NAMES: set[str] = set()


def _prompt_for_class(class_name: str, *, style: str = "class_name") -> str:
    return str(class_name).strip()


def _load_row_image(row: dict):
    image = row.get("image")
    if image is None:
        return None
    if hasattr(image, "convert"):
        return image.convert("RGB")
    if isinstance(image, dict):
        if image.get("bytes"):
            with Image.open(io.BytesIO(image["bytes"])) as opened:
                return opened.convert("RGB")
        if image.get("path"):
            with Image.open(str(image["path"])) as opened:
                return opened.convert("RGB")
        return None
    if isinstance(image, str):
        with Image.open(image) as opened:
            return opened.convert("RGB")
    return None


def _to_base_sample(row: dict):
    sample = None
    original_error: Exception | None = None
    try:
        sample = _ORIGINAL_TO_BASE_SAMPLE(row)
    except Exception as exc:  # Fallback for string/dict-backed image rows.
        original_error = exc
    if sample is None:
        image = _load_row_image(row)
        if image is None:
            if original_error is not None:
                raise original_error
            return None
        width, height = image.size
        boxes = _base._parse_answer_boxes(row.get("answer_boxes"), width=width, height=height)
        source = str(row.get("source_collection") or row.get("source_dataset") or "unknown")
        sample = _base.BaseSample(image=image, boxes=boxes, source=source)
    if not _ALLOWED_CLASS_NAMES:
        return sample
    filtered = [item for item in sample.boxes if item.class_name in _ALLOWED_CLASS_NAMES]
    return _base.BaseSample(image=sample.image, boxes=filtered, source=sample.source)


def parse_args(argv: Optional[list[str]] = None):
    raw_argv = list(argv) if argv is not None else list(os.sys.argv[1:])
    if "--config" not in raw_argv:
        raw_argv = ["--config", str(DEFAULT_CONFIG_PATH), *raw_argv]
    _base._prompt_for_class = _prompt_for_class
    _base._to_base_sample = _to_base_sample
    args = _base.parse_args(raw_argv)
    resolved_dataset_path, dataset_warning = resolve_inspector_dataset_path(
        str(args.dataset_path or ""),
        task="detect",
        module_root=SCRIPT_DIR,
    )
    if dataset_warning:
        print(f"warning: {dataset_warning}")
    args.dataset_path = str(resolved_dataset_path) if resolved_dataset_path is not None else ""
    global _ALLOWED_CLASS_NAMES
    dataset_path = str(args.dataset_path or "").strip() or None
    class_catalog = _base._load_class_catalog(str(args.class_names_file or ""), dataset_path)
    _ALLOWED_CLASS_NAMES = {class_name for _, class_name in class_catalog if class_name}
    args.skill = "detect"
    if not str(args.base_url).strip():
        args.base_url = DEFAULT_BASE_URL
    if not args.finetune_id and str(args.finetune_name).startswith("pid-icons-"):
        args.finetune_name = f"inspector-detect-{_base._random_suffix()}"
    args.async_checkpoint_eval_benchmark_script = str((SCRIPT_DIR / "benchmark_inspector_detect.py").resolve())
    return args


def main(argv: Optional[list[str]] = None) -> None:
    _base.run(parse_args(argv))


if __name__ == "__main__":
    main()
