from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tictaktoe_QA import train_ttt_query_rl as mod


class ParsePredictionTests(unittest.TestCase):
    def test_parse_plain_json(self) -> None:
        payload = mod._parse_prediction_json('{"winner":"draw"}')
        self.assertEqual(payload, {"winner": "draw"})

    def test_parse_reason_final_wrapped_json(self) -> None:
        text = "Reason: center controls diagonals.\nFinal: {\"row\":2,\"col\":2}"
        payload = mod._parse_prediction_json(text)
        self.assertEqual(payload, {"row": 2, "col": 2})

    def test_parse_invalid_json(self) -> None:
        payload = mod._parse_prediction_json("not json at all")
        self.assertIsNone(payload)


class RewardPolicyTests(unittest.TestCase):
    def _best_move_example(self) -> mod.QAExample:
        return mod.QAExample(
            row_id="r0",
            split="train",
            task_type="best_move",
            question="q",
            image_path=Path("/tmp/unused.png"),
            expected_answer={"row": 2, "col": 2},
            best_move_canonical=5,
            best_move_optimal_set=frozenset({1, 5}),
        )

    def test_best_move_canonical_reward(self) -> None:
        example = self._best_move_example()
        out = mod._score_payload_for_example(
            example,
            {"row": 2, "col": 2},
            best_move_optimal_reward=0.7,
        )
        self.assertEqual(out.reward, 1.0)
        self.assertTrue(out.best_move_set_correct)
        self.assertTrue(out.best_move_canonical_correct)

    def test_best_move_optimal_noncanonical_reward(self) -> None:
        example = self._best_move_example()
        out = mod._score_payload_for_example(
            example,
            {"row": 1, "col": 1},
            best_move_optimal_reward=0.7,
        )
        self.assertEqual(out.reward, 0.7)
        self.assertTrue(out.best_move_set_correct)
        self.assertFalse(out.best_move_canonical_correct)

    def test_best_move_incorrect_reward(self) -> None:
        example = self._best_move_example()
        out = mod._score_payload_for_example(
            example,
            {"row": 3, "col": 3},
            best_move_optimal_reward=0.7,
        )
        self.assertEqual(out.reward, 0.0)
        self.assertFalse(out.best_move_set_correct)
        self.assertFalse(out.best_move_canonical_correct)

    def test_turn_player_normalization(self) -> None:
        example = mod.QAExample(
            row_id="r1",
            split="train",
            task_type="turn_player",
            question="q",
            image_path=Path("/tmp/unused.png"),
            expected_answer={"player": "X"},
            best_move_canonical=None,
            best_move_optimal_set=frozenset(),
        )
        out = mod._score_payload_for_example(
            example,
            {"player": "x"},
            best_move_optimal_reward=0.7,
        )
        self.assertEqual(out.reward, 1.0)
        self.assertTrue(out.exact_non_best_correct)

    def test_legal_moves_list_ordered_match(self) -> None:
        example = mod.QAExample(
            row_id="r2",
            split="train",
            task_type="legal_moves_list",
            question="q",
            image_path=Path("/tmp/unused.png"),
            expected_answer={"legal_moves": [{"row": 1, "col": 1}, {"row": 2, "col": 3}]},
            best_move_canonical=None,
            best_move_optimal_set=frozenset(),
        )
        out = mod._score_payload_for_example(
            example,
            {"legal_moves": [{"row": 1, "col": 1}, {"row": 2, "col": 3}]},
            best_move_optimal_reward=0.7,
        )
        self.assertEqual(out.reward, 1.0)

    def test_legal_moves_list_truncated_dict_fails_parse_success(self) -> None:
        example = mod.QAExample(
            row_id="r3",
            split="train",
            task_type="legal_moves_list",
            question="q",
            image_path=Path("/tmp/unused.png"),
            expected_answer={"legal_moves": [{"row": 1, "col": 1}]},
            best_move_canonical=None,
            best_move_optimal_set=frozenset(),
        )
        out = mod._score_payload_for_example(
            example,
            {"row": 1, "col": 1},
            best_move_optimal_reward=0.7,
        )
        self.assertEqual(out.reward, 0.0)
        self.assertFalse(out.parse_success)
        self.assertTrue(out.json_object_parsed)


class SchemaValidationTests(unittest.TestCase):
    def _example(self, task_type: str, expected_answer: dict[str, object]) -> mod.QAExample:
        return mod.QAExample(
            row_id=f"row_{task_type}",
            split="train",
            task_type=task_type,
            question="q",
            image_path=Path("/tmp/unused.png"),
            expected_answer=expected_answer,
            best_move_canonical=5 if task_type == "best_move" else None,
            best_move_optimal_set=frozenset({5}) if task_type == "best_move" else frozenset(),
        )

    def test_malformed_dict_payloads_fail_parse_success_by_task(self) -> None:
        cases = [
            ("best_move", {"row": 2, "col": 2}, {"row": 0, "col": 4}),
            ("winner", {"winner": "draw"}, {"winner": "maybe"}),
            ("is_terminal", {"is_terminal": False}, {"is_terminal": "sometimes"}),
            ("has_winning_move", {"has_winning_move": True}, {"has_winning_move": "unknown"}),
            ("turn_player", {"player": "X"}, {"player": "Q"}),
            ("legal_moves_count", {"legal_move_count": 3}, {"legal_move_count": -2}),
            (
                "legal_moves_list",
                {"legal_moves": [{"row": 1, "col": 1}]},
                {"legal_moves": [{"row": 1, "col": 4}]},
            ),
        ]

        for task_type, expected_answer, bad_payload in cases:
            with self.subTest(task_type=task_type):
                out = mod._score_payload_for_example(
                    self._example(task_type, expected_answer),
                    bad_payload,
                    best_move_optimal_reward=0.7,
                )
                self.assertEqual(out.reward, 0.0)
                self.assertFalse(out.parse_success)
                self.assertTrue(out.json_object_parsed)

    def test_non_object_prediction_reports_object_parse_false(self) -> None:
        out = mod._score_payload_for_example(
            self._example("winner", {"winner": "draw"}),
            None,
            best_move_optimal_reward=0.7,
        )
        self.assertFalse(out.parse_success)
        self.assertFalse(out.json_object_parsed)


class PathResolutionTests(unittest.TestCase):
    def test_image_path_fallback_to_dataset_images_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = Path(tmp)
            images_dir = dataset_dir / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            fallback = images_dir / "foo.png"
            Image.new("RGB", (8, 8), color=(255, 255, 255)).save(fallback)

            row = {
                "image_path": "/abs/path/that/does/not/exist/foo.png",
                "image": "/abs/path/that/does/not/exist/foo.png",
            }
            resolved = mod._resolve_image_path(row, dataset_dir)
            self.assertEqual(resolved, fallback.resolve())


class ConfigPrecedenceTests(unittest.TestCase):
    def test_cli_overrides_config_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "cfg.json"
            cfg_path.write_text(
                json.dumps(
                    {
                        "num_steps": 10,
                        "batch_size": 9,
                        "group_size": 3,
                        "dataset_dir": "tictaktoe_QA/synth_dataset/outputs/smoke_full_jsonl",
                        "final_eval_splits": ["val", "test"],
                    }
                ),
                encoding="utf-8",
            )

            args = mod.parse_args(
                [
                    "--config",
                    str(cfg_path),
                    "--num-steps",
                    "20",
                    "--batch-size",
                    "4",
                ]
            )

            self.assertEqual(args.num_steps, 20)
            self.assertEqual(args.batch_size, 4)
            self.assertEqual(args.group_size, 3)
            self.assertEqual(args.final_eval_splits, ["val", "test"])

    def test_reasoning_and_json_overrides_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "cfg.json"
            cfg_path.write_text(
                json.dumps(
                    {
                        "dataset_dir": "tictaktoe_QA/synth_dataset/outputs/smoke_full_jsonl",
                        "reasoning": False,
                        "task_sampling_weights": {
                            "best_move": 2.0,
                            "legal_moves_count": 2.5,
                            "legal_moves_list": 3.0,
                        },
                        "max_tokens_by_task": {"legal_moves_list": 512},
                    }
                ),
                encoding="utf-8",
            )

            args = mod.parse_args(
                [
                    "--config",
                    str(cfg_path),
                    "--reasoning",
                    "--task-sampling-weights-json",
                    "{\"best_move\":7.0}",
                    "--max-tokens-by-task-json",
                    "{\"best_move\":196}",
                ]
            )

            self.assertTrue(args.reasoning)
            self.assertEqual(args.task_sampling_weights["best_move"], 7.0)
            self.assertEqual(args.task_sampling_weights["winner"], 1.0)
            self.assertEqual(args.max_tokens_by_task["legal_moves_list"], 512)
            self.assertEqual(args.max_tokens_by_task["best_move"], 196)


class SamplingConfigTests(unittest.TestCase):
    def test_task_sampling_weights_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown task_type"):
            mod._resolve_task_sampling_weights(
                config_map={"not_a_task": 2.0},
                cli_override_json="",
            )

        with self.assertRaisesRegex(ValueError, "must be > 0"):
            mod._resolve_task_sampling_weights(
                config_map={"best_move": 0.0},
                cli_override_json="",
            )

    def test_max_tokens_by_task_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown task_type"):
            mod._resolve_max_tokens_by_task(
                config_map={"bad_task": 200},
                cli_override_json="",
            )

        with self.assertRaisesRegex(ValueError, "must be > 0"):
            mod._resolve_max_tokens_by_task(
                config_map={"best_move": 0},
                cli_override_json="",
            )

    def test_weighted_sampling_oversamples_weak_tasks(self) -> None:
        weights = mod._resolve_task_sampling_weights(
            config_map=mod.DEFAULT_TASK_SAMPLING_WEIGHTS,
            cli_override_json="",
        )
        tasks = sorted(mod.SUPPORTED_TASKS)
        weighted = [weights[task] for task in tasks]
        uniform = [1.0 for _ in tasks]

        rng_weighted = random.Random(7)
        rng_uniform = random.Random(7)
        weighted_draws = rng_weighted.choices(tasks, weights=weighted, k=4000)
        uniform_draws = rng_uniform.choices(tasks, weights=uniform, k=4000)

        weighted_counts = {task: weighted_draws.count(task) for task in tasks}
        uniform_counts = {task: uniform_draws.count(task) for task in tasks}

        self.assertGreater(weighted_counts["legal_moves_list"], uniform_counts["legal_moves_list"])
        self.assertGreater(weighted_counts["best_move"], uniform_counts["best_move"])


if __name__ == "__main__":
    unittest.main()
