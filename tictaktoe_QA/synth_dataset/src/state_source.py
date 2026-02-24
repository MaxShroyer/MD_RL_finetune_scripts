"""Cloudwalk tic-tac-toe dataset fetch/cache utilities."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MAIN_URL = "https://raw.githubusercontent.com/cloudwalk/tictactoe-dataset/main/dataset.json"
TOP50_URL = "https://raw.githubusercontent.com/cloudwalk/tictactoe-dataset/main/dataset_top50.json"


@dataclass(frozen=True)
class CloudwalkData:
    """Container for loaded Cloudwalk boards."""

    main_boards: dict[str, dict[str, Any]]
    top50_boards: dict[str, dict[str, Any]]
    cache_dir: Path


def _download_json(url: str, timeout_sec: float = 30.0) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": "ttt-qa-builder/1.0"})
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:  # noqa: S310
        payload = resp.read().decode("utf-8")
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object from {url}, got {type(data).__name__}")
    return data


def _read_cached(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_boards_payload(name: str, payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    boards = payload.get("boards")
    if not isinstance(boards, dict):
        raise ValueError(f"{name} missing 'boards' object")
    if not boards:
        raise ValueError(f"{name} has empty boards payload")
    return boards


def load_cloudwalk_data(
    *,
    cache_dir: Path,
    allow_network: bool = True,
    main_url: str = MAIN_URL,
    top50_url: str = TOP50_URL,
) -> CloudwalkData:
    """Load dataset JSONs from network with cache fallback.

    Policy:
    - Try network first when allowed.
    - If network fails, fallback to cache.
    - If both unavailable, raise.
    """

    cache_dir.mkdir(parents=True, exist_ok=True)
    main_cache_path = cache_dir / "dataset.json"
    top50_cache_path = cache_dir / "dataset_top50.json"

    main_payload: dict[str, Any] | None = None
    top50_payload: dict[str, Any] | None = None

    if allow_network:
        try:
            main_payload = _download_json(main_url)
            main_cache_path.write_text(json.dumps(main_payload), encoding="utf-8")
        except (OSError, urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
            main_payload = None

        try:
            top50_payload = _download_json(top50_url)
            top50_cache_path.write_text(json.dumps(top50_payload), encoding="utf-8")
        except (OSError, urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
            top50_payload = None

    if main_payload is None:
        main_payload = _read_cached(main_cache_path)
    if top50_payload is None:
        top50_payload = _read_cached(top50_cache_path)

    main_boards = _validate_boards_payload("dataset.json", main_payload)
    top50_boards = _validate_boards_payload("dataset_top50.json", top50_payload)
    return CloudwalkData(main_boards=main_boards, top50_boards=top50_boards, cache_dir=cache_dir)
