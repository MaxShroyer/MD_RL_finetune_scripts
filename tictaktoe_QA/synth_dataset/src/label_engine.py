"""Labeling and board-logic helpers for TicTacToe QA dataset generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

WIN_LINES: tuple[tuple[int, int, int], ...] = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
)


@dataclass(frozen=True)
class BoardRecord:
    """Normalized board with precomputed labels needed by QA tasks."""

    state_key: str
    state: tuple[str | int, ...]
    player_to_move: str
    symmetry_group: str
    depth_complexity: int
    choice_complexity_num: int
    choice_complexity_den: int
    legal_moves: tuple[int, ...]
    winner_label: str
    is_terminal: bool
    immediate_winning_moves: tuple[int, ...]
    best_move_optimal_set: tuple[int, ...]
    best_move_canonical: int | None
    scores_by_move_json: str


@dataclass(frozen=True)
class BestMoveLabels:
    best_move_optimal_set: tuple[int, ...]
    best_move_canonical: int | None


def move_to_row_col(move: int) -> tuple[int, int]:
    if move < 1 or move > 9:
        raise ValueError(f"move out of range: {move}")
    idx = move - 1
    return (idx // 3) + 1, (idx % 3) + 1


def row_col_to_move(row: int, col: int) -> int:
    if row < 1 or row > 3 or col < 1 or col > 3:
        raise ValueError(f"row/col out of range: row={row}, col={col}")
    return (row - 1) * 3 + col


def _cell_value(cell: str | int) -> int:
    if cell == "X":
        return 1
    if cell == "O":
        return -1
    return 0


def legal_moves_from_state(state: tuple[str | int, ...] | list[str | int]) -> tuple[int, ...]:
    moves: list[int] = []
    for idx, cell in enumerate(state):
        if isinstance(cell, int):
            moves.append(idx + 1)
    return tuple(sorted(moves))


def winner_from_state(state: tuple[str | int, ...] | list[str | int]) -> str:
    values = [_cell_value(cell) for cell in state]
    for a, b, c in WIN_LINES:
        line_sum = values[a] + values[b] + values[c]
        if line_sum == 3:
            return "X"
        if line_sum == -3:
            return "O"
    if all(v != 0 for v in values):
        return "draw"
    return "in_progress"


def is_terminal_state(state: tuple[str | int, ...] | list[str | int]) -> bool:
    return winner_from_state(state) != "in_progress"


def apply_move(state: tuple[str | int, ...] | list[str | int], move: int, player: str) -> tuple[str | int, ...]:
    if player not in {"X", "O"}:
        raise ValueError(f"invalid player: {player}")
    idx = move - 1
    state_list = list(state)
    if not isinstance(state_list[idx], int):
        raise ValueError(f"illegal move {move} for state")
    state_list[idx] = player
    return tuple(state_list)


def immediate_winning_moves(
    state: tuple[str | int, ...] | list[str | int],
    player_to_move: str,
) -> tuple[int, ...]:
    wins: list[int] = []
    for move in legal_moves_from_state(state):
        nxt = apply_move(state, move, player_to_move)
        if winner_from_state(nxt) == player_to_move:
            wins.append(move)
    return tuple(sorted(wins))


def _choice_complexity(category: dict[str, Any]) -> tuple[int, int]:
    cc = category.get("choice_complexity")
    if not isinstance(cc, list) or len(cc) != 2:
        return (0, 1)
    try:
        num = int(cc[0])
        den = int(cc[1])
    except (TypeError, ValueError):
        return (0, 1)
    if den <= 0:
        return (0, 1)
    return (num, den)


def best_move_labels(scores: list[dict[str, Any]]) -> BestMoveLabels:
    movable = [s for s in scores if isinstance(s, dict) and s.get("move") is not None]
    if not movable:
        return BestMoveLabels(best_move_optimal_set=tuple(), best_move_canonical=None)

    max_value = max(int(s["value"]) for s in movable)
    best_by_value = [s for s in movable if int(s["value"]) == max_value]

    if max_value == 1:
        depth_target = min(int(s["depth"]) for s in best_by_value)
    else:
        depth_target = max(int(s["depth"]) for s in best_by_value)

    best_full = [s for s in best_by_value if int(s["depth"]) == depth_target]
    optimal_moves = sorted(int(s["move"]) for s in best_full)

    canonical: int | None
    if not optimal_moves:
        canonical = None
    else:
        canonical = sorted(optimal_moves, key=lambda m: move_to_row_col(m))[0]

    return BestMoveLabels(best_move_optimal_set=tuple(optimal_moves), best_move_canonical=canonical)


def build_board_record(state_key: str, payload: dict[str, Any]) -> BoardRecord:
    state_raw = payload.get("state")
    if not isinstance(state_raw, list) or len(state_raw) != 9:
        raise ValueError(f"invalid state for board {state_key}")

    state: tuple[str | int, ...] = tuple(state_raw)
    player = str(payload.get("player", "")).strip()
    if player not in {"X", "O"}:
        raise ValueError(f"invalid player for board {state_key}: {player!r}")

    category = payload.get("category") if isinstance(payload.get("category"), dict) else {}
    symmetry_group = str(category.get("symmetry_group", state_key))
    depth_complexity = int(category.get("depth_complexity", 0) or 0)
    choice_num, choice_den = _choice_complexity(category)

    legal_moves = legal_moves_from_state(state)
    winner = winner_from_state(state)
    terminal = winner != "in_progress"
    immediate = immediate_winning_moves(state, player) if not terminal else tuple()

    scores = payload.get("scores") if isinstance(payload.get("scores"), list) else []
    best_labels = best_move_labels(scores)

    move_scores: dict[str, dict[str, int]] = {}
    for score in scores:
        if not isinstance(score, dict):
            continue
        move = score.get("move")
        if move is None:
            continue
        try:
            move_int = int(move)
            move_scores[str(move_int)] = {
                "value": int(score.get("value", 0)),
                "depth": int(score.get("depth", 0)),
            }
        except (TypeError, ValueError):
            continue

    return BoardRecord(
        state_key=state_key,
        state=state,
        player_to_move=player,
        symmetry_group=symmetry_group,
        depth_complexity=depth_complexity,
        choice_complexity_num=choice_num,
        choice_complexity_den=choice_den,
        legal_moves=legal_moves,
        winner_label=winner,
        is_terminal=terminal,
        immediate_winning_moves=immediate,
        best_move_optimal_set=best_labels.best_move_optimal_set,
        best_move_canonical=best_labels.best_move_canonical,
        scores_by_move_json=json.dumps(move_scores, separators=(",", ":"), sort_keys=True),
    )


def parse_state_key(state_key: str) -> tuple[str | int, ...]:
    """Parse compact state key like '12OX5X789' into 9 cells."""
    cells: list[str | int] = []
    for ch in state_key:
        if ch in {"X", "O"}:
            cells.append(ch)
        elif ch.isdigit():
            cells.append(int(ch))
    if len(cells) != 9:
        raise ValueError(f"could not parse state_key into 9 cells: {state_key}")
    return tuple(cells)


def board_to_text_grid(state: tuple[str | int, ...]) -> str:
    parts = ["-" if isinstance(c, int) else str(c) for c in state]
    return (
        f"{parts[0]} {parts[1]} {parts[2]}\n"
        f"{parts[3]} {parts[4]} {parts[5]}\n"
        f"{parts[6]} {parts[7]} {parts[8]}"
    )
