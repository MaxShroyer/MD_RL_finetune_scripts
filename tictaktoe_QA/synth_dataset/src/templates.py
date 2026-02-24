"""Prompt templates for TicTacToe QA tasks."""

from __future__ import annotations

import random
from dataclasses import dataclass

from .label_engine import BoardRecord

TASK_TYPES = (
    "best_move",
    "has_winning_move",
    "turn_player",
    "winner",
    "is_terminal",
    "legal_moves_count",
    "legal_moves_list",
)


@dataclass(frozen=True)
class PromptSpec:
    question: str
    prompt_variant_id: str
    explicit_player: bool


_CANONICAL_TEMPLATES: dict[str, str] = {
    "best_move": "You are {player_ref}. What is the next best move? Respond with JSON: {{\"row\":<1-3>,\"col\":<1-3>}}.",
    "has_winning_move": "You are {player_ref}. Is there an immediate winning move this turn? Respond JSON: {{\"has_winning_move\":true|false}}.",
    "turn_player": "Whose move is it in this board state? Respond JSON: {{\"player\":\"X\"|\"O\"}}.",
    "winner": "Who has won in this board state? Use X, O, draw, or in_progress. Respond JSON: {{\"winner\":\"...\"}}.",
    "is_terminal": "Is this game state terminal? Respond JSON: {{\"is_terminal\":true|false}}.",
    "legal_moves_count": "How many legal moves are available now? Respond JSON: {{\"legal_move_count\":<int>}}.",
    "legal_moves_list": "List all legal moves as row/col pairs sorted by board order. Respond JSON: {{\"legal_moves\":[{{\"row\":1,\"col\":1}}]}}.",
}

_PARAPHRASE_TEMPLATES: dict[str, str] = {
    "best_move": "Playing as {player_ref}, choose the strongest move from this position. Output only JSON {{\"row\":r,\"col\":c}}.",
    "has_winning_move": "From {player_ref}'s perspective, can the game be won immediately on this turn? Output JSON {{\"has_winning_move\":true|false}}.",
    "turn_player": "Identify which mark should play next on the shown board. Return JSON {{\"player\":\"X\"|\"O\"}}.",
    "winner": "Determine the current result of the board: X, O, draw, or in_progress. Return JSON {{\"winner\":\"...\"}}.",
    "is_terminal": "Does this board already represent game over? Return JSON {{\"is_terminal\":true|false}}.",
    "legal_moves_count": "Count the currently open cells where a legal move can be made. Return JSON {{\"legal_move_count\":n}}.",
    "legal_moves_list": "Provide every legal move in top-left to bottom-right order as row/col JSON objects.",
}


_MOVE_TEMPLATES_EXPLICIT: dict[str, tuple[str, ...]] = {
    "best_move": (
        "You are {player_ref}. What is the next best move? Return JSON {{\"row\":<1-3>,\"col\":<1-3>}}.",
        "As {player_ref}, choose the optimal next move. Answer JSON {{\"row\":r,\"col\":c}}.",
        "{player_ref} to play: pick the strongest move now. Respond with JSON {{\"row\":r,\"col\":c}}.",
    ),
    "has_winning_move": (
        "You are {player_ref}. Is there a one-move win available right now? Return JSON {{\"has_winning_move\":true|false}}.",
        "As {player_ref}, can you win immediately this turn? Return JSON {{\"has_winning_move\":true|false}}.",
        "{player_ref} to move: does an instant winning move exist? Answer JSON {{\"has_winning_move\":true|false}}.",
    ),
}

_MOVE_TEMPLATES_IMPLICIT: dict[str, tuple[str, ...]] = {
    "best_move": (
        "Infer whose turn it is from the board, then choose the best next move. Return JSON {{\"row\":<1-3>,\"col\":<1-3>}}.",
        "Without being told the player, infer the side to move and give the optimal move as JSON {{\"row\":r,\"col\":c}}.",
    ),
    "has_winning_move": (
        "Infer the player to move from the board. Does that player have an immediate win? Return JSON {{\"has_winning_move\":true|false}}.",
        "From the shown board alone, determine if the next player has a one-turn winning move. Output JSON {{\"has_winning_move\":true|false}}.",
    ),
}

_OTHER_TEMPLATES: dict[str, tuple[str, ...]] = {
    "turn_player": (
        "Whose move is it? Return JSON {{\"player\":\"X\"|\"O\"}}.",
        "Identify the next player to act in this board. Answer JSON {{\"player\":\"X\"|\"O\"}}.",
        "Determine the side to move now. Return JSON {{\"player\":\"X\"|\"O\"}}.",
    ),
    "winner": (
        "Who won this board? Use X, O, draw, or in_progress. Return JSON {{\"winner\":\"...\"}}.",
        "What is the game outcome in this position: X, O, draw, or in_progress? Respond JSON {{\"winner\":\"...\"}}.",
        "Classify the board result as X, O, draw, or in_progress. Output JSON {{\"winner\":\"...\"}}.",
    ),
    "is_terminal": (
        "Is this game already over? Return JSON {{\"is_terminal\":true|false}}.",
        "Determine whether this board is terminal. Respond JSON {{\"is_terminal\":true|false}}.",
        "Has the game ended in this position? Output JSON {{\"is_terminal\":true|false}}.",
    ),
    "legal_moves_count": (
        "How many legal moves are available now? Return JSON {{\"legal_move_count\":<int>}}.",
        "Count open cells where the next player can move. Answer JSON {{\"legal_move_count\":n}}.",
        "Compute the number of legal moves in this board state. Output JSON {{\"legal_move_count\":n}}.",
    ),
    "legal_moves_list": (
        "List all legal moves in board order as row/col JSON objects. Return JSON {{\"legal_moves\":[{{\"row\":1,\"col\":1}}]}}.",
        "Provide every currently legal move, top-left to bottom-right. Respond JSON {{\"legal_moves\":[...]}}.",
        "Enumerate legal positions as row/col entries sorted in reading order. Output JSON {{\"legal_moves\":[...]}}.",
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
        if explicit_player_override is None:
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
