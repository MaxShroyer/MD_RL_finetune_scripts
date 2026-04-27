from __future__ import annotations

from pathlib import Path

from md_train_framework.examples.common import run_with_default_config

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "default.json"


def main(argv: list[str] | None = None) -> int:
    return run_with_default_config("compare", DEFAULT_CONFIG_PATH, argv=argv)


if __name__ == "__main__":
    raise SystemExit(main())
