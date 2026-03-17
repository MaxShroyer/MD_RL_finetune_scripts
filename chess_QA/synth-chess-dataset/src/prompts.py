"""Prompt templates for synthetic chess QA tasks."""

from __future__ import annotations

import random

TASK_TYPES = (
    "list_all_pieces",
    "count_by_color",
    "list_color_pieces",
    "color_presence_check",
)

_PROMPT_POOLS_V1: dict[str, tuple[str, ...]] = {
    "list_all_pieces": (
        'List every chess piece and square from this image. Return JSON only as {"pieces":[{"black_bishop":"c8","black_king":"g8"}]}.',
        'Identify all pieces visible on the board with square notation. Return JSON only as {"pieces":[{"black_bishop":"c8","black_king":"g8"}]}.',
        'Extract the full board piece layout using algebraic squares. Return JSON only as {"pieces":[{"black_bishop":"c8","black_king":"g8"}]}.',
        'Provide all detected pieces with square positions. Return JSON only as {"pieces":[{"black_bishop":"c8","black_king":"g8"}]}.',
    ),
    "count_by_color": (
        'Count total white and total black pieces in this image. Return JSON only as {"white_piece_count":8,"black_piece_count":7}.',
        'Give the total number of white pieces and black pieces. Return JSON only as {"white_piece_count":8,"black_piece_count":7}.',
        'Compute board counts by color only (not by piece type). Return JSON only as {"white_piece_count":8,"black_piece_count":7}.',
        'Report how many white pieces and black pieces are present. Return JSON only as {"white_piece_count":8,"black_piece_count":7}.',
    ),
    "list_color_pieces": (
        'List all "<color>" pieces with their squares. Return JSON only as {"color":"<color>","pieces":[{"black_knight":"c7","black_king":"g8"}]}.',
        'Show every "<color>" piece on the board. Return JSON only as {"color":"<color>","pieces":[{"black_knight":"c7","black_king":"g8"}]}.',
        'Provide the full list of "<color>" pieces and positions. Return JSON only as {"color":"<color>","pieces":[{"black_knight":"c7","black_king":"g8"}]}.',
        'Extract only "<color>" pieces from this board. Return JSON only as {"color":"<color>","pieces":[{"black_knight":"c7","black_king":"g8"}]}.',
    ),
    "color_presence_check": (
        'Are there any "<color>" pieces present? Return JSON only as {"color":"<color>","present":true,"count":7}.',
        'Check if color "<color>" appears on this board and report count. Return JSON only as {"color":"<color>","present":true,"count":7}.',
        'Determine whether "<color>" pieces exist in this position. Return JSON only as {"color":"<color>","present":true,"count":7}.',
        'For "<color>", return presence and total quantity. Return JSON only as {"color":"<color>","present":true,"count":7}.',
    ),
}

_PROMPT_POOLS_V2: dict[str, tuple[str, ...]] = {
    "list_all_pieces": (
        'List all pieces and their squares. Return JSON only as {"pieces":[{"black_bishop":"c8","black_king":"g8"}]}.',
    ),
    "count_by_color": (
        'How many white pieces and black pieces are on the board? Return JSON only as {"white_piece_count":8,"black_piece_count":7}.',
    ),
    "list_color_pieces": (
        'List all "<color>" pieces and their squares. Return JSON only as {"color":"<color>","pieces":[{"black_knight":"c7","black_king":"g8"}]}.',
    ),
    "color_presence_check": (
        'Is there any "<color>" piece on the board? Return JSON only as {"color":"<color>","present":true}.',
    ),
}

_PROMPT_POOLS_BY_SET: dict[str, dict[str, tuple[str, ...]]] = {
    "v1": _PROMPT_POOLS_V1,
    "v2": _PROMPT_POOLS_V2,
}


def choose_prompt(
    task_type: str,
    *,
    rng: random.Random,
    queried_piece: str | None = None,
    prompt_set: str = "v1",
) -> tuple[str, str]:
    """Select one prompt variant for a task."""

    normalized_prompt_set = str(prompt_set).strip().lower()
    if normalized_prompt_set not in _PROMPT_POOLS_BY_SET:
        raise ValueError(f"Unknown prompt_set: {prompt_set}")

    prompt_pools = _PROMPT_POOLS_BY_SET[normalized_prompt_set]
    if task_type not in prompt_pools:
        raise ValueError(f"Unknown task_type: {task_type}")

    variants = prompt_pools[task_type]
    variant_idx = rng.randrange(len(variants))
    template = variants[variant_idx]
    if "<piece>" in template:
        if not queried_piece:
            raise ValueError(f"queried_piece is required for task_type={task_type}")
        question = template.replace("<piece>", queried_piece)
    elif "<color>" in template:
        if not queried_piece:
            raise ValueError(f"queried_piece is required for task_type={task_type}")
        question = template.replace("<color>", queried_piece)
    else:
        question = template
    if normalized_prompt_set == "v2":
        return question, f"{task_type}_v2_canonical"
    return question, task_type
