#!/usr/bin/env python3
"""Thin wrapper around the shared detect benchmark for local aerial datasets."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from MDpi_and_d import benchmark_pid_icons as _base
except ModuleNotFoundError:
    from _DEPICATED_MDpi_and_d import benchmark_pid_icons as _base

from aerial_airport.common import DEFAULT_STAGING_API_BASE, detect_prompt_for_class, repo_relative

DEFAULT_CONFIG_PATH = repo_relative("configs", "benchmark_aerial_airport_detect_default.json")
_call_detect_api = _base._call_detect_api


def _prompt_for_class(class_name: str, *, style: str = "detect_phrase", prompt_overrides=None) -> str:
    return detect_prompt_for_class(class_name, prompt_overrides=prompt_overrides)


def parse_args(argv: Optional[list[str]] = None):
    raw_argv = list(argv) if argv is not None else list(os.sys.argv[1:])
    if "--config" not in raw_argv:
        raw_argv = ["--config", str(DEFAULT_CONFIG_PATH), *raw_argv]
    _base._prompt_for_class = _prompt_for_class
    args = _base.parse_args(raw_argv)
    args.skill = "detect"
    if not str(args.api_base).strip():
        args.api_base = DEFAULT_STAGING_API_BASE
    return args


def main(argv: Optional[list[str]] = None) -> None:
    _base._call_detect_api = _call_detect_api
    _base.run(parse_args(argv))


if __name__ == "__main__":
    main()
