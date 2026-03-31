#!/usr/bin/env python3
"""Example wrapper for ConstructionSite caption query training."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from construction_site import train_construction_site_query_caption as _impl


def main(argv: Optional[list[str]] = None) -> None:
    _impl.main(argv)


if __name__ == "__main__":
    main()

