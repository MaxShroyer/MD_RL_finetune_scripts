"""Shared helpers for the P&ID OpenRouter evaluation suite."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence


_JSON_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def package_root() -> Path:
    return Path(__file__).resolve().parent


def resolve_path(raw_path: str, *, search_roots: Sequence[Path] = ()) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()

    roots: list[Path] = []
    for root in (Path.cwd(), *search_roots):
        root_resolved = Path(root).resolve()
        if root_resolved not in roots:
            roots.append(root_resolved)

    for root in roots:
        candidate = (root / path).resolve()
        if candidate.exists():
            return candidate

    return (Path.cwd() / path).resolve()


def load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in: {path}")
    return payload


def validate_config_keys(
    config: dict[str, Any],
    *,
    allowed_keys: set[str],
    config_path: Path,
) -> None:
    unknown = sorted(k for k in config.keys() if k not in allowed_keys)
    if unknown:
        raise ValueError(
            f"Unknown config key(s) in {config_path}: {unknown}. "
            "Remove typos or update script support."
        )


def cfg_str(config: dict[str, Any], key: str, fallback: str) -> str:
    value = config.get(key, fallback)
    if value is None:
        return fallback
    return str(value)


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


def cfg_list_str(config: dict[str, Any], key: str, fallback: list[str]) -> list[str]:
    value = config.get(key)
    if value is None:
        return list(fallback)
    if isinstance(value, str):
        items = [part.strip() for part in value.split(",")]
        return [item for item in items if item]
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                out.append(text)
        return out
    return list(fallback)


def slugify(text: str) -> str:
    lowered = str(text).strip().lower()
    slug = re.sub(r"[^a-z0-9._-]+", "_", lowered)
    slug = slug.strip("_")
    return slug or "run"


def utc_timestamp_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def truncate_text(text: str, *, limit: int = 200) -> str:
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def truncate_json_payload(payload: Any, *, limit: int = 2000) -> str:
    try:
        text = json.dumps(payload, ensure_ascii=False)
    except Exception:
        text = str(payload)
    return truncate_text(text, limit=limit)


def extract_openrouter_answer_text(response_payload: dict[str, Any]) -> str:
    choices = response_payload.get("choices")
    if isinstance(choices, list) and choices:
        first_choice = choices[0]
        if isinstance(first_choice, dict):
            message = first_choice.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    return content.strip()
                if isinstance(content, list):
                    chunks: list[str] = []
                    for item in content:
                        if not isinstance(item, dict):
                            continue
                        text = item.get("text")
                        if isinstance(text, str):
                            chunks.append(text)
                    if chunks:
                        return "\n".join(chunks).strip()

    output_text = response_payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    data = response_payload.get("data")
    if isinstance(data, dict):
        text = data.get("text")
        if isinstance(text, str):
            return text.strip()

    return ""


def _try_parse_json_object(text: str) -> Optional[dict[str, Any]]:
    if not text:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict):
        return payload
    return None


def _find_braced_json_object(text: str) -> Optional[dict[str, Any]]:
    for start_idx, ch in enumerate(text):
        if ch != "{":
            continue

        depth = 0
        in_string = False
        escaped = False
        for end_idx in range(start_idx, len(text)):
            cur = text[end_idx]
            if in_string:
                if escaped:
                    escaped = False
                elif cur == "\\":
                    escaped = True
                elif cur == '"':
                    in_string = False
                continue

            if cur == '"':
                in_string = True
                continue
            if cur == "{":
                depth += 1
                continue
            if cur == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start_idx : end_idx + 1]
                    payload = _try_parse_json_object(candidate)
                    if payload is not None:
                        return payload
                if depth < 0:
                    break
    return None


def extract_first_json_object(text: str) -> Optional[dict[str, Any]]:
    stripped = (text or "").strip()
    if not stripped:
        return None

    direct = _try_parse_json_object(stripped)
    if direct is not None:
        return direct

    for match in _JSON_FENCE_PATTERN.finditer(stripped):
        fenced = (match.group(1) or "").strip()
        if not fenced:
            continue
        payload = _try_parse_json_object(fenced)
        if payload is not None:
            return payload
        payload = _find_braced_json_object(fenced)
        if payload is not None:
            return payload

    return _find_braced_json_object(stripped)
