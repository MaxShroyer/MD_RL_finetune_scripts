"""Prompt templates for TicTacToe QA tasks."""

from __future__ import annotations

import random
from dataclasses import dataclass

from .label_engine import BoardRecord
from tictaktoe_QA.task_schema import CANONICAL_TASK_TYPES, normalize_task_type

TASK_TYPES = CANONICAL_TASK_TYPES


@dataclass(frozen=True)
class PromptSpec:
    question: str
    prompt_variant_id: str
    explicit_player: bool


_CANONICAL_TEMPLATES: dict[str, str] = {
    "best_move": "You are {player_ref}. Choose the best next move. Return only JSON {{\"row\":<1-3>,\"col\":<1-3>}}.",
    "has_winning_move": "You are {player_ref}. Is there an immediate winning move this turn? Return only JSON {{\"has_winning_move\":true}} or {{\"has_winning_move\":false}}.",
    "turn_player": "Identify which mark moves next. Return only JSON {{\"player\":\"X\"}} or {{\"player\":\"O\"}}.",
    "winner": "Classify this board result as X, O, draw, or in_progress. Return only JSON {{\"winner\":\"X\"}}, {{\"winner\":\"O\"}}, {{\"winner\":\"draw\"}}, or {{\"winner\":\"in_progress\"}}.",
    "is_game_over": "Is the game over in this board state? Return only JSON {{\"is_game_over\":true}} or {{\"is_game_over\":false}}.",
    "available_moves_count": "How many available moves are there now? Return only JSON {{\"available_move_count\":<int>}}.",
    "available_moves_list": "List all available moves in board order (top-left to bottom-right). Return only JSON {{\"available_moves\":[{{\"row\":1,\"col\":1}}]}}.",
}

_PARAPHRASE_TEMPLATES: dict[str, str] = {
    "best_move": "Playing as {player_ref}, choose the strongest move from this position. Return only JSON {{\"row\":<1-3>,\"col\":<1-3>}}.",
    "has_winning_move": "From {player_ref}'s perspective, can the game be won immediately this turn? Return only JSON {{\"has_winning_move\":true}} or {{\"has_winning_move\":false}}.",
    "turn_player": "Identify which mark should play next on the shown board. Return only JSON {{\"player\":\"X\"}} or {{\"player\":\"O\"}}.",
    "winner": "Determine the board result: X, O, draw, or in_progress. Return only JSON {{\"winner\":\"X\"}}, {{\"winner\":\"O\"}}, {{\"winner\":\"draw\"}}, or {{\"winner\":\"in_progress\"}}.",
    "is_game_over": "Does this board already represent game over? Return only JSON {{\"is_game_over\":true}} or {{\"is_game_over\":false}}.",
    "available_moves_count": "Count open cells where the next player can move. Return only JSON {{\"available_move_count\":<int>}}.",
    "available_moves_list": "Provide every available move in top-left to bottom-right order. Return only JSON {{\"available_moves\":[{{\"row\":1,\"col\":1}}]}}.",
}


_MOVE_TEMPLATES_EXPLICIT: dict[str, tuple[str, ...]] = {
    "best_move": (
        "You are {player_ref}. Choose the best next move. Return only JSON {{\"row\":<1-3>,\"col\":<1-3>}}.",
        "As {player_ref}, choose the optimal next move. Return only JSON {{\"row\":<1-3>,\"col\":<1-3>}}.",
        "{player_ref} to play: pick the strongest move now. Return only JSON {{\"row\":<1-3>,\"col\":<1-3>}}.",
    ),
    "has_winning_move": (
        "You are {player_ref}. Is there a one-move win available right now? Return only JSON {{\"has_winning_move\":true}} or {{\"has_winning_move\":false}}.",
        "As {player_ref}, can you win immediately this turn? Return only JSON {{\"has_winning_move\":true}} or {{\"has_winning_move\":false}}.",
        "{player_ref} to move: does an instant winning move exist? Return only JSON {{\"has_winning_move\":true}} or {{\"has_winning_move\":false}}.",
    ),
}

_MOVE_TEMPLATES_IMPLICIT: dict[str, tuple[str, ...]] = {
    "has_winning_move": (
        "Infer the player to move from the board. Does that player have an immediate win? Return only JSON {{\"has_winning_move\":true}} or {{\"has_winning_move\":false}}.",
        "From the shown board alone, determine if the next player has a one-turn winning move. Return only JSON {{\"has_winning_move\":true}} or {{\"has_winning_move\":false}}.",
    ),
}

_OTHER_TEMPLATES: dict[str, tuple[str, ...]] = {
    "turn_player": (
        "Whose move is it? Return only JSON {{\"player\":\"X\"}} or {{\"player\":\"O\"}}.",
        "Identify the next player to act in this board. Return only JSON {{\"player\":\"X\"}} or {{\"player\":\"O\"}}.",
        "Determine the side to move now. Return only JSON {{\"player\":\"X\"}} or {{\"player\":\"O\"}}.",
    ),
    "winner": (
        "Who won this board? Use X, O, draw, or in_progress. Return only JSON {{\"winner\":\"X\"}}, {{\"winner\":\"O\"}}, {{\"winner\":\"draw\"}}, or {{\"winner\":\"in_progress\"}}.",
        "What is the game outcome in this position: X, O, draw, or in_progress? Return only JSON {{\"winner\":\"X\"}}, {{\"winner\":\"O\"}}, {{\"winner\":\"draw\"}}, or {{\"winner\":\"in_progress\"}}.",
        "Classify the board result as X, O, draw, or in_progress. Return only JSON {{\"winner\":\"X\"}}, {{\"winner\":\"O\"}}, {{\"winner\":\"draw\"}}, or {{\"winner\":\"in_progress\"}}.",
    ),
    "is_game_over": (
        "Is this game already over? Return only JSON {{\"is_game_over\":true}} or {{\"is_game_over\":false}}.",
        "Determine whether this board is game over. Return only JSON {{\"is_game_over\":true}} or {{\"is_game_over\":false}}.",
        "Has the game ended in this position? Return only JSON {{\"is_game_over\":true}} or {{\"is_game_over\":false}}.",
    ),
    "available_moves_count": (
        "How many available moves are there now? Return only JSON {{\"available_move_count\":<int>}}.",
        "Count open cells where the next player can move. Return only JSON {{\"available_move_count\":<int>}}.",
        "Compute the number of available moves in this board state. Return only JSON {{\"available_move_count\":<int>}}.",
    ),
    "available_moves_list": (
        "List all available moves in board order as row/col JSON objects. Return only JSON {{\"available_moves\":[{{\"row\":1,\"col\":1}}]}}.",
        "Provide every currently available move, top-left to bottom-right. Return only JSON {{\"available_moves\":[{{\"row\":1,\"col\":1}}]}}.",
        "Enumerate available positions as row/col entries sorted in reading order. Return only JSON {{\"available_moves\":[{{\"row\":1,\"col\":1}}]}}.",
    ),
}


def _player_ref(player: str, rng: random.Random) -> str:
    if player == "X":
        options = ("X", "cross (X)")
    else:
        options = ("O", "circle (O)")
    return rng.choice(options)


def choose_prompt(
    *,
    task_type: str,
    record: BoardRecord,
    rng: random.Random,
    benchmark_track: str | None = None,
    explicit_player_override: bool | None = None,
) -> PromptSpec:
    """Select prompt template with deterministic benchmark behavior."""

    task_type = normalize_task_type(task_type)

    if task_type not in TASK_TYPES:
        raise ValueError(f"unknown task_type: {task_type}")

    player_ref = _player_ref(record.player_to_move, rng)

    if benchmark_track == "canonical":
        tmpl = _CANONICAL_TEMPLATES[task_type]
        q = tmpl.format(player_ref=player_ref)
        return PromptSpec(question=q, prompt_variant_id=f"canonical:{task_type}", explicit_player=True)

    if benchmark_track == "paraphrase":
        tmpl = _PARAPHRASE_TEMPLATES[task_type]
        q = tmpl.format(player_ref=player_ref)
        return PromptSpec(question=q, prompt_variant_id=f"paraphrase:{task_type}", explicit_player=True)

    if task_type in {"best_move", "has_winning_move"}:
        if task_type == "best_move":
            explicit = True
        elif explicit_player_override is None:
            explicit = rng.random() < 0.7
        else:
            explicit = explicit_player_override

        if explicit:
            tmpl = rng.choice(_MOVE_TEMPLATES_EXPLICIT[task_type])
            variant = f"main:{task_type}:explicit"
        else:
            tmpl = rng.choice(_MOVE_TEMPLATES_IMPLICIT[task_type])
            variant = f"main:{task_type}:implicit"
        return PromptSpec(question=tmpl.format(player_ref=player_ref), prompt_variant_id=variant, explicit_player=explicit)

    tmpl = rng.choice(_OTHER_TEMPLATES[task_type])
    return PromptSpec(question=tmpl.format(player_ref=player_ref), prompt_variant_id=f"main:{task_type}", explicit_player=False)
