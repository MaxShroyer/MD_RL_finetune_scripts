#!/usr/bin/env python3
"""Thin wrapper around the shared point benchmark for the Aerial Airport dataset."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MDpi_and_d import benchmark_pid_icons as _base

from aerial_airport.common import DEFAULT_STAGING_API_BASE, repo_relative

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = repo_relative("configs", "benchmark_aerial_airport_point_default.json")


def parse_args(argv: Optional[list[str]] = None):
    raw_argv = list(argv) if argv is not None else list(os.sys.argv[1:])
    if "--config" not in raw_argv:
        raw_argv = ["--config", str(DEFAULT_CONFIG_PATH), *raw_argv]
    args = _base.parse_args(raw_argv)
    if not str(args.api_base).strip():
        args.api_base = DEFAULT_STAGING_API_BASE
    return args


def main(argv: Optional[list[str]] = None) -> None:
    _base.run(parse_args(argv))


if __name__ == "__main__":
    main()
