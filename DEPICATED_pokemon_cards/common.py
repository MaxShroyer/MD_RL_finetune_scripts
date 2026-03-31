"""Shared helpers for the PokemonCards suite."""

from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
from typing import Any, Iterable, Optional

from PIL import Image

try:
    from dotenv import load_dotenv as _third_party_load_dotenv
except ModuleNotFoundError:  # pragma: no cover
    _third_party_load_dotenv = None

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = Path(__file__).resolve().parent


def repo_root() -> Path:
    return REPO_ROOT


def package_root() -> Path:
    return PACKAGE_ROOT


def resolve_path(raw_path: str | Path, *, search_roots: Iterable[Path] = ()) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()

    from_cwd = (Path.cwd() / path).resolve()
    if from_cwd.exists():
        return from_cwd

    for root in search_roots:
        candidate = (root / path).resolve()
        if candidate.exists():
            return candidate

    return from_cwd


def load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def validate_config_keys(config: dict[str, Any], *, allowed_keys: set[str], config_path: Path) -> None:
    unknown = sorted(key for key in config.keys() if key not in allowed_keys)
    if unknown:
        raise ValueError(f"Unknown config key(s) in {config_path}: {unknown}")


def cfg_str(config: dict[str, Any], key: str, fallback: str) -> str:
    value = config.get(key, fallback)
    return str(value) if value is not None else fallback


def cfg_int(config: dict[str, Any], key: str, fallback: int) -> int:
    value = config.get(key, fallback)
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def cfg_float(config: dict[str, Any], key: str, fallback: float) -> float:
    value = config.get(key, fallback)
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def cfg_bool(config: dict[str, Any], key: str, fallback: bool) -> bool:
    value = config.get(key, fallback)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y", "on"}:
            return True
        if lowered in {"false", "0", "no", "n", "off"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return fallback


def cfg_optional_bool(config: dict[str, Any], key: str) -> Optional[bool]:
    if key not in config:
        return None
    value = config.get(key)
    if value is None:
        return None
    return cfg_bool({key: value}, key, False)


def cfg_list_str(config: dict[str, Any], key: str, fallback: list[str]) -> list[str]:
    value = config.get(key)
    if not isinstance(value, list):
        return list(fallback)
    out: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            out.append(text)
    return out or list(fallback)


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_dotenv_if_available(path: str | Path) -> None:
    if _third_party_load_dotenv is None:
        return
    resolved = resolve_path(path, search_roots=(repo_root(), package_root()))
    if resolved.exists():
        _third_party_load_dotenv(resolved)


def to_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def image_path_to_data_url(image_path: Path) -> str:
    with Image.open(image_path) as image:
        return to_data_url(image)


def progress_enabled(no_progress: bool) -> bool:
    if no_progress:
        return False
    return os.isatty(2)


def truncate_text(text: str, *, limit: int = 800) -> str:
    raw = str(text)
    if len(raw) <= limit:
        return raw
    return raw[:limit] + "...<truncated>"


def write_json(path: Path, payload: Any) -> None:
    ensure_parent_dir(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

