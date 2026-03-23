#!/usr/bin/env python3
"""Thin wrapper around the shared point RL trainer for bone fracture abnormality pointing."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MDpi_and_d import train_pid_icons as _base

from bone_fracture.common import (
    DEFAULT_POINT_WANDB_PROJECT,
    DEFAULT_STAGING_API_BASE,
    repo_relative,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = repo_relative("configs", "train_bone_fracture_point_default.json")


def _resolve_module_path(raw_path: str) -> str:
    path = Path(str(raw_path or "")).expanduser()
    if not str(raw_path or "").strip():
        return str(path)
    if path.is_absolute():
        return str(path.resolve())
    parts = path.parts
    if parts and parts[0] == SCRIPT_DIR.name:
        return str((REPO_ROOT / path).resolve())
    return str((SCRIPT_DIR / path).resolve())


def parse_args(argv: Optional[list[str]] = None):
    raw_argv = list(argv) if argv is not None else list(os.sys.argv[1:])
    if "--config" not in raw_argv:
        raw_argv = ["--config", str(DEFAULT_CONFIG_PATH), *raw_argv]
    args = _base.parse_args(raw_argv)
    if not str(args.base_url).strip():
        args.base_url = DEFAULT_STAGING_API_BASE
    if str(getattr(args, "env_file", "") or "").strip():
        args.env_file = _resolve_module_path(args.env_file)
    if str(getattr(args, "dataset_path", "") or "").strip():
        args.dataset_path = _resolve_module_path(args.dataset_path)
    if str(getattr(args, "class_names_file", "") or "").strip():
        args.class_names_file = _resolve_module_path(args.class_names_file)
    if not str(getattr(args, "wandb_project", "") or "").strip():
        args.wandb_project = DEFAULT_POINT_WANDB_PROJECT
    if not args.finetune_id and str(args.finetune_name).startswith("pid-icons-"):
        args.finetune_name = f"bone-fracture-point-{_base._random_suffix()}"
    return args


def main(argv: Optional[list[str]] = None) -> None:
    _base.run(parse_args(argv))


if __name__ == "__main__":
    main()
