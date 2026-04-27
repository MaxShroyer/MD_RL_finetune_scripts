from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from md_train_framework.cli import main as framework_main


def run_with_default_config(command: str, default_config_path: Path, argv: Optional[list[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--config" not in args:
        args = ["--config", str(default_config_path), *args]
    return int(framework_main([str(command), *args]))


def run_eval_with_default_config(default_config_path: Path, argv: Optional[list[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    command = "replay-eval" if "--finetune-id" in args else "baseline"
    if "--config" not in args:
        args = ["--config", str(default_config_path), *args]
    return int(framework_main([command, *args]))
