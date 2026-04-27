from __future__ import annotations

from pathlib import Path

from md_train_framework.cli import main


if __name__ == "__main__":
    config_path = Path(__file__).resolve().parents[1] / "configs" / "detect_quickstart.json"
    raise SystemExit(main(["train", "--config", str(config_path)]))
