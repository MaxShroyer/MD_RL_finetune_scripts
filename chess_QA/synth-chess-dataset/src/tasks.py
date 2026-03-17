"""Task construction helpers and answer payload builders."""

from __future__ import annotations

import copy
import random
from collections import Counter
from typing import Any

TASK_TYPES = (
    "list_all_pieces",
    "count_by_color",
    "list_color_pieces",
    "color_presence_check",
)

CANONICAL_COLORS = ("white", "black")


def _square_sort_key(square: str) -> tuple[int, int]:
    """Sort squares in board-reading order: a8..h8, a7..h7, ..., a1..h1."""

    text = str(square).strip().lower()
    if len(text) != 2:
        return (99, 99)
    file_char, rank_char = text[0], text[1]
    if file_char < "a" or file_char > "h":
        return (99, 99)
    if rank_char < "1" or rank_char > "8":
        return (99, 99)
    file_idx = ord(file_char) - ord("a")
    rank = int(rank_char)
    return (8 - rank, file_idx)


def _extract_square(piece_entry: dict[str, Any]) -> str:
    pos = piece_entry.get("position", {})
    if not isinstance(pos, dict):
        return ""
    square = str(pos.get("square", "")).strip()
    return square


def _piece_sort_key(piece_entry: dict[str, Any]) -> tuple[Any, ...]:
    piece = str(piece_entry.get("piece", ""))
    square = _extract_square(piece_entry)
    has_square = 0 if square else 1
    board_key = _square_sort_key(square)
    return (piece, has_square, board_key, square)


def _position_sort_key(square: str) -> tuple[Any, ...]:
    square = str(square).strip()
    has_square = 0 if square else 1
    board_key = _square_sort_key(square)
    return (has_square, board_key, square)


def sorted_piece_entries(pieces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = copy.deepcopy(pieces)
    out.sort(key=_piece_sort_key)
    return out


def build_list_all_pieces_answer(pieces: list[dict[str, Any]]) -> dict[str, Any]:
    return {"pieces": [_group_piece_positions(pieces)]}


def _piece_color(piece_name: str) -> str:
    if piece_name.startswith("white_"):
        return "white"
    if piece_name.startswith("black_"):
        return "black"
    return ""


def build_count_by_color_answer(pieces: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter()
    for item in pieces:
        piece_name = str(item.get("piece", "")).strip()
        color = _piece_color(piece_name)
        if color:
            counts[color] += 1
    return {
        "white_piece_count": int(counts.get("white", 0)),
        "black_piece_count": int(counts.get("black", 0)),
    }


def build_list_color_pieces_answer(
    pieces: list[dict[str, Any]],
    color: str,
) -> dict[str, Any]:
    return {"color": color, "pieces": [_group_piece_positions(pieces, color_filter=color)]}


def build_color_presence_check_answer(
    pieces: list[dict[str, Any]],
    color: str,
    *,
    include_count: bool = True,
) -> dict[str, Any]:
    count = 0
    for item in pieces:
        piece_name = str(item.get("piece", "")).strip()
        if _piece_color(piece_name) == color:
            count += 1
    payload: dict[str, Any] = {"color": color, "present": bool(count > 0)}
    if include_count:
        payload["count"] = int(count)
    return payload


def _present_colors(pieces: list[dict[str, Any]]) -> set[str]:
    present: set[str] = set()
    for item in pieces:
        piece_name = str(item.get("piece", "")).strip()
        color = _piece_color(piece_name)
        if color:
            present.add(color)
    return present


def _group_piece_positions(
    pieces: list[dict[str, Any]],
    *,
    color_filter: str | None = None,
) -> dict[str, str | list[str]]:
    grouped: dict[str, list[str]] = {}
    for item in sorted_piece_entries(pieces):
        piece_name = str(item.get("piece", "")).strip()
        if not piece_name:
            continue
        if color_filter and _piece_color(piece_name) != color_filter:
            continue
        square = _extract_square(item)
        if not square:
            continue
        grouped.setdefault(piece_name, []).append(square)

    out: dict[str, str | list[str]] = {}
    for piece_name in sorted(grouped):
        squares = sorted(grouped[piece_name], key=_position_sort_key)
        out[piece_name] = squares[0] if len(squares) == 1 else squares
    return out


def _choose_list_color(pieces: list[dict[str, Any]], rng: random.Random) -> str:
    present = sorted(_present_colors(pieces))
    if present:
        return rng.choice(present)
    return rng.choice(list(CANONICAL_COLORS))


def _choose_presence_color(pieces: list[dict[str, Any]], rng: random.Random) -> str:
    present = _present_colors(pieces)
    absent = [color for color in CANONICAL_COLORS if color not in present]
    choose_positive = bool(rng.random() < 0.5)
    if choose_positive and present:
        return rng.choice(sorted(present))
    if absent:
        return rng.choice(absent)
    return rng.choice(list(CANONICAL_COLORS))


def choose_deterministic_task_query(
    task_type: str,
    pieces: list[dict[str, Any]],
    *,
    record_key: str,
) -> str | None:
    rng = random.Random(f"{task_type}:{record_key}:query")
    if task_type == "list_color_pieces":
        return _choose_list_color(pieces, rng)
    if task_type == "color_presence_check":
        return _choose_presence_color(pieces, rng)
    return None


def build_task_answer(
    task_type: str,
    pieces: list[dict[str, Any]],
    *,
    rng: random.Random,
    queried_piece: str | None = None,
    answer_version: str = "v1",
) -> tuple[dict[str, Any], str | None]:
    """Build final answer payload for a task.

    Returns:
        (answer_payload, queried_piece_if_any)
    """

    normalized_answer_version = str(answer_version).strip().lower()
    if normalized_answer_version not in {"v1", "v2"}:
        raise ValueError(f"Unknown answer_version: {answer_version}")

    if task_type == "list_all_pieces":
        return build_list_all_pieces_answer(pieces), None
    if task_type == "count_by_color":
        return build_count_by_color_answer(pieces), None
    if task_type == "list_color_pieces":
        target_color = queried_piece or _choose_list_color(pieces, rng)
        return build_list_color_pieces_answer(pieces, target_color), target_color
    if task_type == "color_presence_check":
        target_color = queried_piece or _choose_presence_color(pieces, rng)
        return (
            build_color_presence_check_answer(
                pieces,
                target_color,
                include_count=(normalized_answer_version != "v2"),
            ),
            target_color,
        )
    raise ValueError(f"Unknown task_type: {task_type}")


def build_balanced_mixed_task_counts(total_rows: int) -> dict[str, int]:
    """Distribute rows across mixed tasks as evenly as possible."""

    if total_rows <= 0:
        raise ValueError(f"total_rows must be > 0, got {total_rows}")

    base, remainder = divmod(total_rows, len(TASK_TYPES))
    counts: dict[str, int] = {}
    for idx, task in enumerate(TASK_TYPES):
        counts[task] = base + (1 if idx < remainder else 0)
    return counts


def build_mixed_task_plan(
    *,
    seed: int,
    total_rows: int,
    task_counts: dict[str, int] | None = None,
) -> list[str]:
    """Build the shuffled task assignment list for mixed dataset rows."""

    if task_counts is None:
        task_counts = build_balanced_mixed_task_counts(total_rows)

    expected_total = sum(int(task_counts.get(task, 0)) for task in TASK_TYPES)
    if total_rows != expected_total:
        raise ValueError(f"total_rows={total_rows} does not match task_counts total={expected_total}")

    task_list: list[str] = []
    for task in TASK_TYPES:
        count = int(task_counts.get(task, 0))
        task_list.extend([task] * count)

    rng = random.Random(f"mixed_task_plan:{seed}")
    rng.shuffle(task_list)
    return task_list
