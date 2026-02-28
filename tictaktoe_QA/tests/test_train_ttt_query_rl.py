from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from tictaktoe_QA import train_ttt_query_rl as mod
from tuna_sdk.errors import TunaAPIError


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
            expected_answer={"row": 1, "col": 1},
            best_move_canonical=1,
            best_move_optimal_set=frozenset({1}),
            best_move_scores=((1, 1, 6), (5, 1, 8), (9, 0, 9)),
        )

    def test_best_move_canonical_reward(self) -> None:
        example = self._best_move_example()
        out = mod._score_payload_for_example(
            example,
            {"row": 1, "col": 1},
            best_move_optimal_reward=0.7,
        )
        self.assertEqual(out.reward, 1.0)
        self.assertTrue(out.best_move_set_correct)
        self.assertTrue(out.best_move_canonical_correct)

    def test_best_move_ranked_noncanonical_reward(self) -> None:
        example = self._best_move_example()
        out = mod._score_payload_for_example(
            example,
            {"row": 2, "col": 2},
            best_move_optimal_reward=0.7,
        )
        self.assertAlmostEqual(out.reward, 0.5, places=6)
        self.assertFalse(out.best_move_set_correct)
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

    def test_best_move_ranked_tied_best_moves_get_full_reward(self) -> None:
        example = mod.QAExample(
            row_id="r0_tied",
            split="train",
            task_type="best_move",
            question="q",
            image_path=Path("/tmp/unused.png"),
            expected_answer={"row": 1, "col": 1},
            best_move_canonical=1,
            best_move_optimal_set=frozenset({1, 5}),
            best_move_scores=((1, 1, 6), (5, 1, 6), (9, 0, 9)),
        )
        out = mod._score_payload_for_example(
            example,
            {"row": 2, "col": 2},
            best_move_optimal_reward=0.7,
        )
        self.assertEqual(out.reward, 1.0)
        self.assertTrue(out.best_move_set_correct)
        self.assertFalse(out.best_move_canonical_correct)

    def test_best_move_fallback_reward_without_scores(self) -> None:
        example = mod.QAExample(
            row_id="r0_fallback",
            split="train",
            task_type="best_move",
            question="q",
            image_path=Path("/tmp/unused.png"),
            expected_answer={"row": 2, "col": 2},
            best_move_canonical=5,
            best_move_optimal_set=frozenset({1, 5}),
            best_move_scores=tuple(),
        )
        out = mod._score_payload_for_example(
            example,
            {"row": 1, "col": 1},
            best_move_optimal_reward=0.7,
        )
        self.assertEqual(out.reward, 0.7)
        self.assertTrue(out.best_move_set_correct)
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

    def test_available_moves_list_ordered_match(self) -> None:
        example = mod.QAExample(
            row_id="r2",
            split="train",
            task_type="available_moves_list",
            question="q",
            image_path=Path("/tmp/unused.png"),
            expected_answer={"available_moves": [{"row": 1, "col": 1}, {"row": 2, "col": 3}]},
            best_move_canonical=None,
            best_move_optimal_set=frozenset(),
        )
        out = mod._score_payload_for_example(
            example,
            {"available_moves": [{"row": 1, "col": 1}, {"row": 2, "col": 3}]},
            best_move_optimal_reward=0.7,
        )
        self.assertEqual(out.reward, 1.0)

    def test_available_moves_list_truncated_dict_fails_parse_success(self) -> None:
        example = mod.QAExample(
            row_id="r3",
            split="train",
            task_type="available_moves_list",
            question="q",
            image_path=Path("/tmp/unused.png"),
            expected_answer={"available_moves": [{"row": 1, "col": 1}]},
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

    def test_available_move_legacy_answer_keys_are_accepted(self) -> None:
        count_example = mod.QAExample(
            row_id="r_legacy_count",
            split="train",
            task_type="available_moves_count",
            question="q",
            image_path=Path("/tmp/unused.png"),
            expected_answer={"available_move_count": 2},
            best_move_canonical=None,
            best_move_optimal_set=frozenset(),
        )
        count_out = mod._score_payload_for_example(
            count_example,
            {"legal_move_count": 2},
            best_move_optimal_reward=0.7,
        )
        self.assertEqual(count_out.reward, 1.0)

        game_over_example = mod.QAExample(
            row_id="r_legacy_game_over",
            split="train",
            task_type="is_game_over",
            question="q",
            image_path=Path("/tmp/unused.png"),
            expected_answer={"is_game_over": False},
            best_move_canonical=None,
            best_move_optimal_set=frozenset(),
        )
        game_over_out = mod._score_payload_for_example(
            game_over_example,
            {"is_terminal": False},
            best_move_optimal_reward=0.7,
        )
        self.assertEqual(game_over_out.reward, 1.0)

        list_example = mod.QAExample(
            row_id="r_legacy_list",
            split="train",
            task_type="available_moves_list",
            question="q",
            image_path=Path("/tmp/unused.png"),
            expected_answer={"available_moves": [{"row": 1, "col": 1}]},
            best_move_canonical=None,
            best_move_optimal_set=frozenset(),
        )
        list_out = mod._score_payload_for_example(
            list_example,
            {"legal_moves": [{"row": 1, "col": 1}]},
            best_move_optimal_reward=0.7,
        )
        self.assertEqual(list_out.reward, 1.0)


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
            ("is_game_over", {"is_game_over": False}, {"is_game_over": "sometimes"}),
            ("has_winning_move", {"has_winning_move": True}, {"has_winning_move": "unknown"}),
            ("turn_player", {"player": "X"}, {"player": "Q"}),
            ("available_moves_count", {"available_move_count": 3}, {"available_move_count": -2}),
            (
                "available_moves_list",
                {"available_moves": [{"row": 1, "col": 1}]},
                {"available_moves": [{"row": 1, "col": 4}]},
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

    def test_build_example_normalizes_legacy_task_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = Path(tmp)
            image_path = dataset_dir / "img.png"
            Image.new("RGB", (8, 8), color=(255, 255, 255)).save(image_path)

            row = {
                "row_id": "r1",
                "split": "train",
                "task_type": "legal_moves_count",
                "question": "q",
                "final_answer_json": "{\"legal_move_count\":2}",
                "best_move_canonical_json": "null",
                "best_move_optimal_set_json": "[]",
                "image_path": str(image_path),
            }
            example = mod._build_example(
                row,
                split_name="train",
                dataset_dir=dataset_dir,
                line_number=1,
            )
            self.assertIsNotNone(example)
            assert example is not None
            self.assertEqual(example.task_type, "available_moves_count")


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
                        "dataset_dir": "synth_dataset/outputs/smoke_full_jsonl",
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
                        "dataset_dir": "synth_dataset/outputs/smoke_full_jsonl",
                        "reasoning": False,
                        "task_sampling_weights": {
                            "best_move": 2.0,
                            "available_moves_count": 2.5,
                            "available_moves_list": 3.0,
                        },
                        "max_tokens_by_task": {"available_moves_list": 512},
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
            self.assertEqual(args.max_tokens_by_task["available_moves_list"], 512)
            self.assertEqual(args.max_tokens_by_task["best_move"], 196)

    def test_off_policy_config_and_cli_override_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "cfg.json"
            cfg_path.write_text(
                json.dumps(
                    {
                        "off_policy": True,
                        "off_policy_mix_ratio": 0.75,
                        "off_policy_buffer_size": 2048,
                        "off_policy_warmup_steps": 12,
                        "off_policy_min_buffer_groups": 96,
                    }
                ),
                encoding="utf-8",
            )

            args = mod.parse_args(
                [
                    "--config",
                    str(cfg_path),
                    "--no-off-policy",
                    "--off-policy-mix-ratio",
                    "0.25",
                ]
            )
            self.assertFalse(args.off_policy)
            self.assertAlmostEqual(args.off_policy_mix_ratio, 0.25, places=6)
            self.assertEqual(args.off_policy_buffer_size, 2048)
            self.assertEqual(args.off_policy_warmup_steps, 12)
            self.assertEqual(args.off_policy_min_buffer_groups, 96)

    def test_warn_on_off_policy_reasoning_combo(self) -> None:
        with patch("builtins.print") as mock_print:
            mod._warn_on_unsafe_mode_combo(off_policy=True, reasoning=True)
        self.assertTrue(
            any(
                "off-policy + reasoning" in str(call.args[0])
                for call in mock_print.call_args_list
                if call.args
            )
        )

    def test_no_warn_without_off_policy_reasoning_combo(self) -> None:
        with patch("builtins.print") as mock_print:
            mod._warn_on_unsafe_mode_combo(off_policy=False, reasoning=True)
            mod._warn_on_unsafe_mode_combo(off_policy=True, reasoning=False)
            mod._warn_on_unsafe_mode_combo(off_policy=False, reasoning=False)
        self.assertFalse(mock_print.called)

    def test_unknown_config_key_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "cfg.json"
            cfg_path.write_text(
                json.dumps(
                    {
                        "dataset_dir": "synth_dataset/outputs/smoke_full_jsonl",
                        "resoning": True,
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Unknown config key"):
                mod.parse_args(["--config", str(cfg_path)])

    def test_hf_dataset_source_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "cfg.json"
            cfg_path.write_text(
                json.dumps(
                    {
                        "dataset_source": "hf_hub",
                        "hf_dataset_repo_id": "maxs-m87/tictactoe-qa-v1",
                        "hf_dataset_revision": "main",
                        "checkpoint_avg_splits": ["val", "test"],
                    }
                ),
                encoding="utf-8",
            )

            args = mod.parse_args(["--config", str(cfg_path)])
            self.assertEqual(args.dataset_source, "hf_hub")
            self.assertEqual(args.hf_dataset_repo_id, "maxs-m87/tictactoe-qa-v1")
            self.assertEqual(args.hf_dataset_revision, "main")
            self.assertEqual(args.checkpoint_avg_splits, ["val", "test"])

    def test_eval_and_skip_final_eval_overrides_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "cfg.json"
            cfg_path.write_text(
                json.dumps(
                    {
                        "reasoning": True,
                        "eval_temperature": None,
                        "eval_top_p": None,
                        "eval_reasoning": None,
                        "skip_final_eval": False,
                    }
                ),
                encoding="utf-8",
            )

            args = mod.parse_args(
                [
                    "--config",
                    str(cfg_path),
                    "--eval-temperature",
                    "0.0",
                    "--eval-top-p",
                    "1.0",
                    "--no-eval-reasoning",
                    "--skip-final-eval",
                ]
            )
            self.assertEqual(args.eval_temperature, 0.0)
            self.assertEqual(args.eval_top_p, 1.0)
            self.assertFalse(args.eval_reasoning)
            self.assertTrue(args.skip_final_eval)

    def test_eval_fallback_values_when_unset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "cfg.json"
            cfg_path.write_text(
                json.dumps(
                    {
                        "temperature": 0.7,
                        "top_p": 0.95,
                        "reasoning": True,
                    }
                ),
                encoding="utf-8",
            )
            args = mod.parse_args(["--config", str(cfg_path)])
            self.assertIsNone(args.eval_temperature)
            self.assertIsNone(args.eval_top_p)
            self.assertIsNone(args.eval_reasoning)
            self.assertEqual(args.temperature if args.eval_temperature is None else args.eval_temperature, 0.7)
            self.assertEqual(args.top_p if args.eval_top_p is None else args.eval_top_p, 0.95)
            self.assertTrue(args.reasoning if args.eval_reasoning is None else args.eval_reasoning)

    def test_eval_override_validation(self) -> None:
        args = mod.parse_args(["--eval-temperature", "2.5"])
        with self.assertRaisesRegex(ValueError, "--eval-temperature must be in \\[0,2\\]"):
            mod._validate_args(args)


class CheckpointRankingTests(unittest.TestCase):
    def test_rank_checkpoint_eval_history(self) -> None:
        history = [
            {"step": 10, "avg_eval_reward_mean": 0.40},
            {"step": 20, "avg_eval_reward_mean": 0.55},
            {"step": 30, "avg_eval_reward_mean": 0.50},
        ]
        ranked = mod._rank_checkpoint_eval_history(history)
        self.assertEqual([item["step"] for item in ranked], [20, 30, 10])

    def test_build_checkpoint_ranking_payload(self) -> None:
        payload = mod._build_checkpoint_ranking_payload(
            finetune_id="ft_123",
            checkpoint_avg_metric="eval_reward_mean",
            checkpoint_avg_splits=["val", "test"],
            checkpoint_eval_history=[
                {"step": 5, "avg_eval_reward_mean": 0.2},
                {"step": 9, "avg_eval_reward_mean": 0.7},
            ],
        )
        self.assertEqual(payload["best_avg_eval_reward_step"], 9)
        self.assertAlmostEqual(payload["best_avg_eval_reward"], 0.7, places=6)
        self.assertEqual(payload["checkpoint_avg_metric"], "eval_reward_mean")
        self.assertEqual(len(payload["rankings"]), 2)
        self.assertIn("training_status", payload)
        self.assertFalse(payload["training_status"]["stopped_early"])

    def test_build_checkpoint_ranking_payload_preserves_training_status(self) -> None:
        payload = mod._build_checkpoint_ranking_payload(
            finetune_id="ft_123",
            checkpoint_avg_metric="eval_reward_mean",
            checkpoint_avg_splits=["val", "test"],
            checkpoint_eval_history=[],
            training_status={
                "stopped_early": True,
                "stop_reason": "collapse_parse_floor",
                "completed_steps": 40,
                "target_steps": 100,
                "early_stop_mode": "balanced",
                "collapse_detected": True,
            },
        )
        self.assertTrue(payload["training_status"]["stopped_early"])
        self.assertEqual(payload["training_status"]["stop_reason"], "collapse_parse_floor")
        self.assertTrue(payload["training_status"]["collapse_detected"])

    def test_rank_checkpoint_eval_history_prefers_saved_candidates(self) -> None:
        history = [
            {"step": 10, "avg_eval_reward_mean": 0.90, "checkpoint_saved": False},
            {"step": 20, "avg_eval_reward_mean": 0.50, "checkpoint_saved": True},
            {"step": 30, "avg_eval_reward_mean": 0.60, "checkpoint_saved": True},
        ]
        ranked = mod._rank_checkpoint_eval_history(history)
        self.assertEqual([item["step"] for item in ranked], [30, 20])


class AutoBenchmarkTests(unittest.TestCase):
    def test_parse_args_auto_benchmark_defaults_and_overrides(self) -> None:
        args_default = mod.parse_args([])
        self.assertTrue(args_default.auto_benchmark_best_checkpoint)

        args_disabled = mod.parse_args(["--no-auto-benchmark-best-checkpoint"])
        self.assertFalse(args_disabled.auto_benchmark_best_checkpoint)

    def test_select_checkpoint_step_for_auto_benchmark(self) -> None:
        selected = mod._select_checkpoint_step_for_auto_benchmark(
            ranking_payload={"best_avg_eval_reward_step": 148},
            fallback_step=129,
        )
        self.assertEqual(selected, 148)

        fallback = mod._select_checkpoint_step_for_auto_benchmark(
            ranking_payload={"best_avg_eval_reward_step": -1},
            fallback_step=129,
        )
        self.assertEqual(fallback, 129)

        missing = mod._select_checkpoint_step_for_auto_benchmark(
            ranking_payload={"best_avg_eval_reward_step": -1},
            fallback_step=None,
        )
        self.assertIsNone(missing)

    def test_build_auto_benchmark_command_uses_checkpoint_and_dataset_overrides(self) -> None:
        args = SimpleNamespace(
            auto_benchmark_config="configs/benchmark_default.json",
            env_file=".env",
            base_url="https://api.moondream.ai/v1",
            dataset_source="local_jsonl",
            best_move_optimal_reward=0.7,
            hf_dataset_repo_id="repo",
            hf_dataset_revision="main",
            hf_cache_dir="",
            no_progress=True,
        )
        cmd = mod._build_auto_benchmark_command(
            args=args,
            finetune_id="ft_123",
            checkpoint_step=149,
            dataset_dir=Path("/tmp/ds"),
            output_json=Path("/tmp/metrics.json"),
            predictions_jsonl=Path("/tmp/preds.jsonl"),
        )
        self.assertIn("--checkpoint-step", cmd)
        self.assertIn("149", cmd)
        self.assertIn("--dataset-dir", cmd)
        self.assertIn("/tmp/ds", cmd)
        self.assertIn("--no-progress", cmd)


class CheckpointSaveTests(unittest.TestCase):
    def test_try_save_checkpoint_handles_checkpoint_not_found(self) -> None:
        finetune = SimpleNamespace(
            save_checkpoint=lambda: (_ for _ in ()).throw(
                TunaAPIError("Checkpoint not found", status_code=404)
            )
        )
        ok = mod._try_save_checkpoint(finetune=finetune, context="test")
        self.assertFalse(ok)


class SamplingConfigTests(unittest.TestCase):
    def test_legacy_task_names_normalize_in_weight_and_token_maps(self) -> None:
        weights = mod._resolve_task_sampling_weights(
            config_map={"legal_moves_count": 2.0, "legal_moves_list": 3.0},
            cli_override_json="",
        )
        self.assertEqual(weights["available_moves_count"], 2.0)
        self.assertEqual(weights["available_moves_list"], 3.0)

        max_tokens_by_task = mod._resolve_max_tokens_by_task(
            config_map={"legal_moves_list": 512},
            cli_override_json="",
        )
        self.assertEqual(max_tokens_by_task["available_moves_list"], 512)

    def test_task_sampling_weights_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown task_type"):
            mod._resolve_task_sampling_weights(
                config_map={"not_a_task": 2.0},
                cli_override_json="",
            )

        with self.assertRaisesRegex(ValueError, "must be >= 0"):
            mod._resolve_task_sampling_weights(
                config_map={"best_move": -0.5},
                cli_override_json="",
            )

    def test_task_sampling_weights_allows_zero(self) -> None:
        weights = mod._resolve_task_sampling_weights(
            config_map={"best_move": 0.0},
            cli_override_json="",
        )
        self.assertEqual(weights["best_move"], 0.0)

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
            config_map={
                "best_move": 2.0,
                "available_moves_count": 2.0,
                "available_moves_list": 5.0,
            },
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

        self.assertGreater(weighted_counts["available_moves_list"], uniform_counts["available_moves_list"])
        self.assertGreater(weighted_counts["best_move"], uniform_counts["best_move"])

    def test_weights_for_sampling_tasks_requires_positive_total(self) -> None:
        with self.assertRaisesRegex(ValueError, "must include at least one positive weight"):
            mod._weights_for_sampling_tasks(
                tasks=["best_move", "turn_player"],
                task_sampling_weights={"best_move": 0.0, "turn_player": 0.0},
            )

        weights = mod._weights_for_sampling_tasks(
            tasks=["best_move", "turn_player"],
            task_sampling_weights={"best_move": 1.0, "turn_player": 0.0},
        )
        self.assertEqual(weights, [1.0, 0.0])

    def test_active_tasks_from_sampling_weights(self) -> None:
        weights = {task: 0.0 for task in mod.SUPPORTED_TASKS}
        weights["best_move"] = 1.0
        active_tasks = mod._active_tasks_from_sampling_weights(weights)
        self.assertEqual(active_tasks, {"best_move"})

    def test_filter_examples_by_active_tasks(self) -> None:
        examples = [
            mod.QAExample(
                row_id="r_best",
                split="val",
                task_type="best_move",
                question="q",
                image_path=Path("/tmp/unused.png"),
                expected_answer={"row": 1, "col": 1},
                best_move_canonical=1,
                best_move_optimal_set=frozenset({1}),
            ),
            mod.QAExample(
                row_id="r_turn",
                split="val",
                task_type="turn_player",
                question="q",
                image_path=Path("/tmp/unused.png"),
                expected_answer={"player": "X"},
                best_move_canonical=None,
                best_move_optimal_set=frozenset(),
            ),
        ]
        filtered = mod._filter_examples_by_active_tasks(examples, active_tasks={"best_move"})
        self.assertEqual([item.row_id for item in filtered], ["r_best"])


class EvalSubsetAndEarlyStopTests(unittest.TestCase):
    def _example(self, idx: int) -> mod.QAExample:
        return mod.QAExample(
            row_id=f"row_{idx}",
            split="val",
            task_type="best_move",
            question="q",
            image_path=Path("/tmp/unused.png"),
            expected_answer={"row": 2, "col": 2},
            best_move_canonical=5,
            best_move_optimal_set=frozenset({5}),
        )

    def test_fixed_eval_indices_reproducible(self) -> None:
        split_examples = {
            "val": [self._example(i) for i in range(20)],
            "test": [self._example(i + 100) for i in range(10)],
        }
        first = mod._build_fixed_eval_indices(
            split_examples=split_examples,
            fixed_subset_size=8,
            fixed_subset_seed=1337,
            max_samples=12,
        )
        second = mod._build_fixed_eval_indices(
            split_examples=split_examples,
            fixed_subset_size=8,
            fixed_subset_seed=1337,
            max_samples=12,
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first["val"]), 8)
        self.assertEqual(len(first["test"]), 8)

        third = mod._build_fixed_eval_indices(
            split_examples=split_examples,
            fixed_subset_size=8,
            fixed_subset_seed=777,
            max_samples=12,
        )
        self.assertNotEqual(first["val"], third["val"])

    def test_should_early_stop_collapse_and_plateau(self) -> None:
        stop, reason, collapse = mod._should_early_stop(
            mode="balanced",
            parse_streak=2,
            reward_drop_streak=0,
            recent_eval_rewards=[0.4, 0.5],
        )
        self.assertTrue(stop)
        self.assertEqual(reason, "collapse_parse_floor")
        self.assertTrue(collapse)

        stop, reason, collapse = mod._should_early_stop(
            mode="balanced",
            parse_streak=0,
            reward_drop_streak=0,
            recent_eval_rewards=[0.50, 0.505, 0.507, 0.506],
        )
        self.assertTrue(stop)
        self.assertEqual(reason, "plateau_no_improvement")
        self.assertFalse(collapse)

    def test_should_early_stop_not_triggered_with_clean_metrics(self) -> None:
        stop, reason, collapse = mod._should_early_stop(
            mode="balanced",
            parse_streak=0,
            reward_drop_streak=0,
            recent_eval_rewards=[0.4, 0.43, 0.45, 0.48],
        )
        self.assertFalse(stop)
        self.assertEqual(reason, "")
        self.assertFalse(collapse)


class OffPolicyMixTests(unittest.TestCase):
    def test_compose_train_groups_skips_off_policy_before_warmup(self) -> None:
        on_policy = ["on0", "on1", "on2", "on3"]
        replay = ["rp0", "rp1", "rp2", "rp3", "rp4", "rp5"]

        mixed, off_policy_count = mod._compose_train_groups(
            on_policy_groups=on_policy,
            replay_groups=replay,
            off_policy=True,
            off_policy_mix_ratio=0.5,
            off_policy_warmup_steps=10,
            off_policy_min_buffer_groups=2,
            global_step=3,
            rng=random.Random(7),
        )
        self.assertEqual(mixed, on_policy)
        self.assertEqual(off_policy_count, 0)

    def test_compose_train_groups_mixes_replay_after_warmup(self) -> None:
        on_policy = ["on0", "on1", "on2", "on3"]
        replay = ["rp0", "rp1", "rp2", "rp3", "rp4", "rp5"]

        mixed, off_policy_count = mod._compose_train_groups(
            on_policy_groups=on_policy,
            replay_groups=replay,
            off_policy=True,
            off_policy_mix_ratio=0.5,
            off_policy_warmup_steps=1,
            off_policy_min_buffer_groups=2,
            global_step=4,
            rng=random.Random(11),
        )
        replay_in_mixed = sum(1 for item in mixed if item.startswith("rp"))

        self.assertEqual(len(mixed), len(on_policy))
        self.assertEqual(off_policy_count, 2)
        self.assertEqual(replay_in_mixed, off_policy_count)


if __name__ == "__main__":
    unittest.main()
