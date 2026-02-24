"""Unit tests for tic-tac-toe label logic."""

from __future__ import annotations

import unittest

from tictaktoe_QA.synth_dataset.src.label_engine import (
    best_move_labels,
    immediate_winning_moves,
    is_terminal_state,
    legal_moves_from_state,
    move_to_row_col,
    row_col_to_move,
    winner_from_state,
)


class LabelEngineTests(unittest.TestCase):
    def test_row_col_roundtrip(self) -> None:
        for move in range(1, 10):
            row, col = move_to_row_col(move)
            self.assertEqual(row_col_to_move(row, col), move)

    def test_winner_and_terminal(self) -> None:
        state_x_win = ("X", "X", "X", 4, "O", 6, "O", 8, 9)
        self.assertEqual(winner_from_state(state_x_win), "X")
        self.assertTrue(is_terminal_state(state_x_win))

        state_draw = ("X", "O", "X", "X", "O", "O", "O", "X", "X")
        self.assertEqual(winner_from_state(state_draw), "draw")
        self.assertTrue(is_terminal_state(state_draw))

        state_in_progress = ("X", "O", "X", 4, "O", 6, "O", 8, 9)
        self.assertEqual(winner_from_state(state_in_progress), "in_progress")
        self.assertFalse(is_terminal_state(state_in_progress))

    def test_immediate_winning_moves(self) -> None:
        # X can win immediately at move 3.
        state = ("X", "X", 3, "O", 5, 6, "O", 8, 9)
        wins = immediate_winning_moves(state, "X")
        self.assertEqual(wins, (3,))

    def test_best_move_canonical_tie_break(self) -> None:
        scores = [
            {"value": 1, "depth": 7, "move": 5},
            {"value": 1, "depth": 7, "move": 1},
            {"value": 1, "depth": 8, "move": 9},
            {"value": 0, "depth": 9, "move": 2},
        ]
        labels = best_move_labels(scores)
        self.assertEqual(set(labels.best_move_optimal_set), {1, 5})
        self.assertEqual(labels.best_move_canonical, 1)

    def test_best_move_depth_rule_for_non_win(self) -> None:
        scores = [
            {"value": 0, "depth": 9, "move": 9},
            {"value": 0, "depth": 8, "move": 1},
            {"value": -1, "depth": 7, "move": 2},
        ]
        labels = best_move_labels(scores)
        self.assertEqual(labels.best_move_optimal_set, (9,))
        self.assertEqual(labels.best_move_canonical, 9)

    def test_legal_move_normalization(self) -> None:
        state = ("X", 2, "O", 4, 5, "X", "O", 8, 9)
        self.assertEqual(legal_moves_from_state(state), (2, 4, 5, 8, 9))


if __name__ == "__main__":
    unittest.main()
