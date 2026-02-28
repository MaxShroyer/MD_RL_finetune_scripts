"""Canonical task and answer schema helpers for TicTacToe QA."""

from __future__ import annotations

from typing import Any

CANONICAL_TASK_TYPES: tuple[str, ...] = (
    "best_move",
    "has_winning_move",
    "turn_player",
    "winner",
    "is_game_over",
    "available_moves_count",
    "available_moves_list",
)

TASK_TYPE_ALIASES: dict[str, str] = {
    "is_terminal": "is_game_over",
    "legal_moves_count": "available_moves_count",
    "legal_moves_list": "available_moves_list",
}

TASK_TYPE_SET = set(CANONICAL_TASK_TYPES)

ANSWER_KEY_ALIASES: dict[str, dict[str, str]] = {
    "is_game_over": {
        "is_game_over": "is_game_over",
        "is_terminal": "is_game_over",
    },
    "available_moves_count": {
        "available_move_count": "available_move_count",
        "legal_move_count": "available_move_count",
    },
    "available_moves_list": {
        "available_moves": "available_moves",
        "legal_moves": "available_moves",
    },
}


def normalize_task_type(task_type: str, *, allow_unknown: bool = False) -> str:
    """Map legacy task names to canonical names."""
    normalized = TASK_TYPE_ALIASES.get(str(task_type).strip(), str(task_type).strip())
    if normalized in TASK_TYPE_SET:
        return normalized
    if allow_unknown:
        return normalized
    raise ValueError(f"unknown task_type: {task_type}")


def normalize_task_mapping(
    raw_map: dict[str, Any],
    *,
    allow_unknown: bool = False,
) -> dict[str, Any]:
    """Normalize map keys that represent task names."""
    out: dict[str, Any] = {}
    for key, value in raw_map.items():
        out[normalize_task_type(str(key), allow_unknown=allow_unknown)] = value
    return out


def normalize_answer_payload_for_task(task_type: str, payload: Any) -> Any:
    """Normalize legacy answer keys to canonical keys for a task."""
    if not isinstance(payload, dict):
        return payload

    canonical_task = normalize_task_type(task_type, allow_unknown=True)
    aliases = ANSWER_KEY_ALIASES.get(canonical_task)
    if not aliases:
        return payload

    out: dict[str, Any] = {}
    for old_key, canonical_key in aliases.items():
        if old_key in payload and canonical_key not in out:
            out[canonical_key] = payload[old_key]

    if out:
        return out
    return payload
