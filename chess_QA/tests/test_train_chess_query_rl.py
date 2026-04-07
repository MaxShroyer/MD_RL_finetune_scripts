from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chess_QA import data_loader
from chess_QA import train_chess_query_rl as mod


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _dataset_root() -> Path:
    return _repo_root() / "chess_QA" / "synth-chess-dataset" / "outputs"


def _default_loader_kwargs() -> dict[str, str | None]:
    return {
        "hf_dataset_repo_id": "",
        "hf_dataset_revision": "main",
        "hf_token": "",
        "hf_cache_dir": "",
    }


class DatasetLoaderTests(unittest.TestCase):
    def test_dataset_variant_tag_supports_v2_names(self) -> None:
        self.assertEqual(
            data_loader.normalize_dataset_variant_tag("piece_position_v2_dataset2"),
            "piece_position_v2_dataset2",
        )
        self.assertEqual(
            data_loader.normalize_dataset_variant_tag("mixed_tasks_v2_dataset2"),
            "mixed_tasks_v2_dataset2",
        )
        self.assertEqual(
            data_loader.normalize_dataset_variant_tag("piece_position_v2_osfstorage"),
            "piece_position_v2_osfstorage",
        )
        self.assertEqual(
            data_loader.normalize_dataset_variant_tag("mixed_tasks_v2_osfstorage"),
            "mixed_tasks_v2_osfstorage",
        )

    def test_dataset_variant_tag_resolution_loads_both_variants(self) -> None:
        root = _dataset_root()

        piece_rows = data_loader.load_split_rows(
            dataset_source="local_jsonl",
            dataset_variant_tag="piece_position_v1",
            split_name="train",
            dataset_dir=root,
            **_default_loader_kwargs(),
        )
        mixed_rows = data_loader.load_split_rows(
            dataset_source="local_jsonl",
            dataset_variant_tag="mixed_tasks_v1",
            split_name="train",
            dataset_dir=root,
            **_default_loader_kwargs(),
        )

        self.assertGreater(len(piece_rows), 0)
        self.assertGreater(len(mixed_rows), 0)

        piece_tasks = {str(row.get("task_type", "")) for row in piece_rows if isinstance(row, dict)}
        mixed_tasks = {str(row.get("task_type", "")) for row in mixed_rows if isinstance(row, dict)}

        self.assertEqual(piece_tasks, {"list_all_pieces"})
        self.assertEqual(
            mixed_tasks,
            {"list_all_pieces", "count_by_color", "list_color_pieces", "color_presence_check"},
        )

    def test_local_loader_skips_malformed_jsonl_line_with_warning(self) -> None:
        root = _dataset_root()
        with patch("builtins.print") as mock_print:
            rows = data_loader.load_split_rows(
                dataset_source="local_jsonl",
                dataset_variant_tag="piece_position_v1",
                split_name="test",
                dataset_dir=root,
                **_default_loader_kwargs(),
            )

        self.assertGreater(len(rows), 0)
        self.assertTrue(
            any("invalid_json" in str(call.args[0]) for call in mock_print.call_args_list if call.args)
        )

    def test_local_loader_supports_osf_variant_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            variant = root / "piece_position_v2_osfstorage" / "jsonl"
            variant.mkdir(parents=True, exist_ok=True)
            (variant / "train.jsonl").write_text(
                json.dumps(
                    {
                        "row_id": "r1",
                        "split": "train",
                        "task_type": "list_all_pieces",
                        "question": "q",
                        "final_answer_json": "{\"pieces\":[{}]}",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            rows = data_loader.load_split_rows(
                dataset_source="local_jsonl",
                dataset_variant_tag="piece_position_v2_osfstorage",
                split_name="train",
                dataset_dir=root,
                **_default_loader_kwargs(),
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["row_id"], "r1")


class PathAndSchemaTests(unittest.TestCase):
    def test_imges_prefix_resolves_from_dataset_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "imges").mkdir(parents=True, exist_ok=True)
            image_path = root / "imges" / "sample.jpg"
            Image.new("RGB", (10, 10), color=(255, 255, 255)).save(image_path)

            resolved = mod._resolve_image_path(
                {"image_path": "/imges/sample.jpg"},
                dataset_dir=root,
                dataset_variant_tag="piece_position_v1",
            )
            self.assertEqual(resolved, image_path.resolve())

    def test_simple_dotenv_loader_parses_and_respects_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"KEEP_ME": "original"}, clear=True):
            env_path = Path(tmp) / ".env"
            env_path.write_text(
                'MOONDREAM_API_KEY=abc123\n'
                'TUNA_BASE_URL="https://example.test"\n'
                "KEEP_ME=from_file\n"
                "COMMENTED=value # trailing comment\n"
                "export EXPORTED_KEY=ok\n",
                encoding="utf-8",
            )

            loaded = mod._load_simple_dotenv(env_path, override=False)

            self.assertTrue(loaded)
            self.assertEqual(os.environ["MOONDREAM_API_KEY"], "abc123")
            self.assertEqual(os.environ["TUNA_BASE_URL"], "https://example.test")
            self.assertEqual(os.environ["KEEP_ME"], "original")
            self.assertEqual(os.environ["COMMENTED"], "value")
            self.assertEqual(os.environ["EXPORTED_KEY"], "ok")

            mod._load_simple_dotenv(env_path, override=True)
            self.assertEqual(os.environ["KEEP_ME"], "from_file")

    def test_prepare_requests_respects_image_jpeg_quality(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "board.jpg"
            Image.effect_noise((1200, 800), 100.0).convert("RGB").save(image_path)
            example = mod.QAExample(
                row_id="row1",
                split="train",
                task_type="list_all_pieces",
                question="List every chess piece and square from this image.",
                image_path=image_path,
                expected_answer={"pieces": [{"white_king": "e1"}]},
            )

            requests_q92, _ = mod._prepare_requests(
                [example],
                temperature=1.0,
                top_p=0.9,
                max_tokens=900,
                max_tokens_by_task={"list_all_pieces": 900},
                reasoning=False,
                image_jpeg_quality=92,
            )
            requests_q65, _ = mod._prepare_requests(
                [example],
                temperature=1.0,
                top_p=0.9,
                max_tokens=900,
                max_tokens_by_task={"list_all_pieces": 900},
                reasoning=False,
                image_jpeg_quality=65,
            )

            self.assertEqual(len(requests_q92), 1)
            self.assertEqual(len(requests_q65), 1)
            self.assertLess(len(str(requests_q65[0].image_url)), len(str(requests_q92[0].image_url)))


class ScoringTests(unittest.TestCase):
    def _example(self, task_type: str, expected_answer: dict[str, object]) -> mod.QAExample:
        return mod.QAExample(
            row_id=f"r_{task_type}",
            split="train",
            task_type=task_type,
            question="q",
            image_path=Path("/tmp/unused.png"),
            expected_answer=expected_answer,
            queried_piece=None,
        )

    def test_piece_list_scoring_is_order_insensitive(self) -> None:
        example = self._example(
            "list_all_pieces",
            {
                "pieces": [
                    {
                        "white_king": "e1",
                        "black_king": "e8",
                        "white_pawn": ["a2", "b2"],
                    }
                ]
            },
        )
        pred = {
            "pieces": [
                {
                    "white_pawn": ["b2", "a2"],
                    "black_king": "e8",
                    "white_king": "e1",
                }
            ]
        }
        out = mod._score_payload_for_example(example, pred)
        self.assertTrue(out.parse_success)
        self.assertTrue(out.task_correct)
        self.assertEqual(out.reward, 1.0)

    def test_dense_reward_correct_square_wrong_piece_is_positive(self) -> None:
        example = self._example(
            "list_all_pieces",
            {"pieces": [{"white_king": "e1", "black_king": "e8"}]},
        )
        pred = {"pieces": [{"white_queen": "e1", "black_king": "e8"}]}
        out = mod._score_payload_for_example(example, pred)
        self.assertGreater(out.reward, 0.0)
        self.assertLess(out.reward, 1.0)
        self.assertFalse(out.task_correct)
        self.assertFalse(out.exact_match)

    def test_dense_reward_correct_piece_wrong_square_is_positive(self) -> None:
        example = self._example(
            "list_all_pieces",
            {"pieces": [{"white_king": "e1", "black_king": "e8"}]},
        )
        pred = {"pieces": [{"white_king": "a1", "black_king": "e8"}]}
        out = mod._score_payload_for_example(example, pred)
        self.assertGreater(out.reward, 0.0)
        self.assertLess(out.reward, 1.0)
        self.assertFalse(out.task_correct)

    def test_dense_reward_partial_subset_is_intermediate(self) -> None:
        example = self._example(
            "list_all_pieces",
            {"pieces": [{"white_king": "e1", "black_king": "e8"}]},
        )
        pred = {"pieces": [{"black_king": "e8"}]}
        out = mod._score_payload_for_example(example, pred)
        self.assertGreater(out.reward, 0.0)
        self.assertLess(out.reward, 1.0)
        self.assertFalse(out.task_correct)

    def test_duplicate_ground_truth_square_does_not_require_duplicate_prediction(self) -> None:
        example = self._example(
            "list_all_pieces",
            {
                "pieces": [
                    {
                        "black_pawn": ["f7", "f7", "e6"],
                    }
                ]
            },
        )
        pred = {
            "pieces": [
                {
                    "black_pawn": ["e6", "f7"],
                }
            ]
        }
        out = mod._score_payload_for_example(example, pred)
        self.assertTrue(out.task_correct)
        self.assertEqual(out.reward, 1.0)

    def test_board_metrics_are_perfect_for_exact_piece_map(self) -> None:
        example = self._example(
            "list_all_pieces",
            {"pieces": [{"white_king": "e1", "black_king": "e8", "white_pawn": "a2"}]},
        )
        out = mod._score_payload_for_example(
            example,
            {"pieces": [{"white_pawn": "a2", "black_king": "e8", "white_king": "e1"}]},
        )
        self.assertIsNotNone(out.board_metrics)
        assert out.board_metrics is not None
        self.assertEqual(out.board_metrics.board_square_errors, 0)
        self.assertTrue(out.board_metrics.board_at_1)
        self.assertTrue(out.board_metrics.board_at_2)
        self.assertAlmostEqual(out.board_metrics.square_accuracy, 1.0, places=6)
        self.assertAlmostEqual(out.board_metrics.typed_square_f1, 1.0, places=6)
        self.assertEqual(out.board_metrics.piece_count_abs_error, 0)
        self.assertEqual(out.board_metrics.piece_type_count_l1, 0)
        self.assertEqual(out.board_metrics.pred_square_collision_count, 0)

    def test_board_at_1_accepts_one_square_label_error(self) -> None:
        example = self._example(
            "list_all_pieces",
            {"pieces": [{"white_king": "e1", "black_king": "e8"}]},
        )
        out = mod._score_payload_for_example(
            example,
            {"pieces": [{"white_queen": "e1", "black_king": "e8"}]},
        )
        self.assertFalse(out.exact_match)
        self.assertIsNotNone(out.board_metrics)
        assert out.board_metrics is not None
        self.assertEqual(out.board_metrics.board_square_errors, 1)
        self.assertTrue(out.board_metrics.board_at_1)
        self.assertTrue(out.board_metrics.board_at_2)
        self.assertAlmostEqual(out.board_metrics.square_accuracy, 63.0 / 64.0, places=6)

    def test_board_at_2_accepts_two_square_errors_but_not_board_at_1(self) -> None:
        example = self._example(
            "list_all_pieces",
            {"pieces": [{"white_king": "e1", "black_king": "e8"}]},
        )
        out = mod._score_payload_for_example(
            example,
            {"pieces": [{"white_queen": "e1", "black_king": "e8", "black_rook": "a1"}]},
        )
        self.assertIsNotNone(out.board_metrics)
        assert out.board_metrics is not None
        self.assertEqual(out.board_metrics.board_square_errors, 2)
        self.assertFalse(out.board_metrics.board_at_1)
        self.assertTrue(out.board_metrics.board_at_2)

    def test_board_metrics_track_square_collisions(self) -> None:
        example = self._example(
            "list_all_pieces",
            {"pieces": [{"white_king": "e1", "black_king": "e8"}]},
        )
        out = mod._score_payload_for_example(
            example,
            {"pieces": [{"white_king": "e1", "black_king": "e1"}]},
        )
        self.assertIsNotNone(out.board_metrics)
        assert out.board_metrics is not None
        self.assertEqual(out.board_metrics.pred_square_collision_count, 1)
        self.assertEqual(out.board_metrics.board_square_errors, 2)
        self.assertLess(out.board_metrics.square_accuracy, 1.0)

    def test_hallucinated_piece_reduces_precision_and_square_accuracy(self) -> None:
        example = self._example(
            "list_all_pieces",
            {"pieces": [{"white_king": "e1", "black_king": "e8"}]},
        )
        out = mod._score_payload_for_example(
            example,
            {"pieces": [{"white_king": "e1", "black_king": "e8", "white_queen": "a1"}]},
        )
        self.assertIsNotNone(out.board_metrics)
        assert out.board_metrics is not None
        self.assertLess(out.board_metrics.typed_square_precision, 1.0)
        self.assertLess(out.board_metrics.square_precision, 1.0)
        self.assertLess(out.board_metrics.square_accuracy, 1.0)
        self.assertEqual(out.board_metrics.piece_count_abs_error, 1)

    def test_missing_pieces_reduce_recall_and_board_at_k(self) -> None:
        example = self._example(
            "list_all_pieces",
            {"pieces": [{"white_king": "e1", "black_king": "e8", "white_pawn": ["a2", "b2"]}]},
        )
        out = mod._score_payload_for_example(
            example,
            {"pieces": [{"white_king": "e1", "black_king": "e8"}]},
        )
        self.assertIsNotNone(out.board_metrics)
        assert out.board_metrics is not None
        self.assertLess(out.board_metrics.typed_square_recall, 1.0)
        self.assertLess(out.board_metrics.square_recall, 1.0)
        self.assertEqual(out.board_metrics.board_square_errors, 2)
        self.assertFalse(out.board_metrics.board_at_1)
        self.assertTrue(out.board_metrics.board_at_2)

    def test_list_color_dense_reward_gates_on_color(self) -> None:
        example = self._example(
            "list_color_pieces",
            {"color": "white", "pieces": [{"white_king": "e1"}]},
        )
        wrong_color = mod._score_payload_for_example(
            example,
            {"color": "black", "pieces": [{"white_king": "e1"}]},
        )
        self.assertEqual(wrong_color.reward, 0.0)
        self.assertFalse(wrong_color.task_correct)

    def test_parse_failure_stays_zero_reward(self) -> None:
        example = self._example(
            "list_all_pieces",
            {"pieces": [{"white_king": "e1"}]},
        )
        out = mod._score_payload_for_example(example, {"pieces": [{"white_king": 1}]})
        self.assertEqual(out.reward, 0.0)
        self.assertFalse(out.parse_success)
        self.assertTrue(out.json_object_parsed)

    def test_invalid_square_prediction_is_rejected(self) -> None:
        example = self._example(
            "list_all_pieces",
            {"pieces": [{"white_king": "e1"}]},
        )
        out = mod._score_payload_for_example(example, {"pieces": [{"white_king": "i9"}]})
        self.assertEqual(out.reward, 0.0)
        self.assertFalse(out.parse_success)
        self.assertTrue(out.json_object_parsed)

    def test_unknown_piece_key_remains_scoreable(self) -> None:
        example = self._example(
            "list_all_pieces",
            {
                "pieces": [
                    {
                        "unknown_bishop": "e4",
                        "white_king": "b5",
                    }
                ]
            },
        )
        pred = {
            "pieces": [
                {
                    "Unknown_Bishop": "e4",
                    "white_king": "b5",
                }
            ]
        }
        out = mod._score_payload_for_example(example, pred)
        self.assertTrue(out.task_correct)
        self.assertEqual(out.reward, 1.0)

    def test_task_specific_scoring_positive_and_negative(self) -> None:
        count_example = self._example(
            "count_by_color",
            {"white_piece_count": 8, "black_piece_count": 7},
        )
        self.assertTrue(
            mod._score_payload_for_example(
                count_example,
                {"white_piece_count": 8, "black_piece_count": 7},
            ).task_correct
        )
        self.assertFalse(
            mod._score_payload_for_example(
                count_example,
                {"white_piece_count": 8, "black_piece_count": 6},
            ).task_correct
        )

        presence_example = self._example(
            "color_presence_check",
            {"color": "white", "present": True, "count": 2},
        )
        self.assertTrue(
            mod._score_payload_for_example(
                presence_example,
                {"color": "white", "present": True, "count": 2},
            ).task_correct
        )
        self.assertFalse(
            mod._score_payload_for_example(
                presence_example,
                {"color": "black", "present": True, "count": 2},
            ).task_correct
        )

        list_color_example = self._example(
            "list_color_pieces",
            {"color": "black", "pieces": [{"black_king": "e8", "black_rook": ["a8", "h8"]}]},
        )
        self.assertTrue(
            mod._score_payload_for_example(
                list_color_example,
                {"color": "black", "pieces": [{"black_rook": ["h8", "a8"], "black_king": "e8"}]},
            ).task_correct
        )
        self.assertFalse(
            mod._score_payload_for_example(
                list_color_example,
                {"color": "white", "pieces": [{"black_king": "e8", "black_rook": ["a8", "h8"]}]},
            ).task_correct
        )

        list_all_example = self._example(
            "list_all_pieces",
            {"pieces": [{"white_king": "e1", "black_king": "e8"}]},
        )
        self.assertTrue(
            mod._score_payload_for_example(
                list_all_example,
                {"pieces": [{"black_king": "e8", "white_king": "e1"}]},
            ).task_correct
        )
        self.assertFalse(
            mod._score_payload_for_example(
                list_all_example,
                {"pieces": [{"black_king": "e8"}]},
            ).task_correct
        )

    def test_count_by_color_dense_reward_is_intermediate_for_near_miss(self) -> None:
        example = self._example(
            "count_by_color",
            {"white_piece_count": 8, "black_piece_count": 7},
        )
        close = mod._score_payload_for_example(
            example,
            {"white_piece_count": 8, "black_piece_count": 6},
        )
        far = mod._score_payload_for_example(
            example,
            {"white_piece_count": 8, "black_piece_count": 2},
        )
        self.assertGreater(close.reward, 0.0)
        self.assertLess(close.reward, 1.0)
        self.assertGreater(close.reward, far.reward)

    def test_color_presence_without_count_parses_and_scores_binary(self) -> None:
        example = self._example(
            "color_presence_check",
            {"color": "white", "present": True},
        )
        out = mod._score_payload_for_example(
            example,
            {"color": "white", "present": True},
        )
        self.assertTrue(out.parse_success)
        self.assertEqual(out.reward, 1.0)
        self.assertTrue(out.task_correct)

    def test_color_presence_reward_ignores_count_even_when_exact_match_does_not(self) -> None:
        example = self._example(
            "color_presence_check",
            {"color": "white", "present": True, "count": 2},
        )
        out = mod._score_payload_for_example(
            example,
            {"color": "white", "present": True, "count": 99},
        )
        self.assertEqual(out.reward, 1.0)
        self.assertFalse(out.task_correct)
        self.assertFalse(out.exact_match)

    def test_dense_partial_v2_prefers_closer_piece_count_and_keeps_exact_one(self) -> None:
        example = self._example(
            "list_all_pieces",
            {"pieces": [{"white_king": "e1", "black_king": "e8", "white_pawn": "a2"}]},
        )
        close = mod._score_payload_for_example(
            example,
            {"pieces": [{"white_king": "e1", "black_king": "e8"}]},
            list_piece_reward_mode="dense_partial_v2",
        )
        far = mod._score_payload_for_example(
            example,
            {"pieces": [{"black_king": "e8"}]},
            list_piece_reward_mode="dense_partial_v2",
        )
        exact = mod._score_payload_for_example(
            example,
            {"pieces": [{"white_king": "e1", "black_king": "e8", "white_pawn": "a2"}]},
            list_piece_reward_mode="dense_partial_v2",
        )
        self.assertGreater(close.reward, far.reward)
        self.assertEqual(exact.reward, 1.0)


class ConfigPrecedenceTests(unittest.TestCase):
    def test_config_loader_accepts_line_comments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "cfg.json"
            cfg_path.write_text(
                "\n".join(
                    [
                        "// staging ci config",
                        "{",
                        '  "base_url": "https://api-staging.moondream.ai/v1",',
                        '  "env_file": ".env-staging",',
                        '  "group_size": 4',
                        "}",
                    ]
                ),
                encoding="utf-8",
            )

            args = mod.parse_args(["--config", str(cfg_path)])

            self.assertEqual(args.base_url, "https://api-staging.moondream.ai/v1")
            self.assertEqual(args.env_file, ".env-staging")
            self.assertEqual(args.group_size, 4)

    def test_cli_overrides_dataset_variant_and_json_maps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "cfg.json"
            cfg_path.write_text(
                json.dumps(
                    {
                        "dataset_source": "local_jsonl",
                        "dataset_dir": "chess_QA/synth-chess-dataset/outputs",
                        "dataset_variant_tag": "piece_position_v1",
                        "task_sampling_weights": {
                            "list_all_pieces": 2.0,
                            "count_by_color": 0.5,
                        },
                        "max_tokens_by_task": {
                            "count_by_color": 111,
                        },
                    }
                ),
                encoding="utf-8",
            )

            args = mod.parse_args(
                [
                    "--config",
                    str(cfg_path),
                    "--dataset-variant-tag",
                    "mixed_tasks_v1",
                    "--task-sampling-weights-json",
                    '{"count_by_color":3.0}',
                    "--max-tokens-by-task-json",
                    '{"list_all_pieces":777}',
                ]
            )

            self.assertEqual(args.dataset_variant_tag, "mixed_tasks_v1")
            self.assertEqual(args.task_sampling_weights["count_by_color"], 3.0)
            self.assertEqual(args.task_sampling_weights["list_all_pieces"], 2.0)
            self.assertEqual(args.max_tokens_by_task["list_all_pieces"], 777)
            self.assertEqual(args.max_tokens_by_task["count_by_color"], 111)

    def test_off_policy_and_reward_weights_cli_override_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "cfg.json"
            cfg_path.write_text(
                json.dumps(
                    {
                        "dataset_source": "local_jsonl",
                        "dataset_dir": "chess_QA/synth-chess-dataset/outputs",
                        "dataset_variant_tag": "piece_position_v1",
                        "off_policy": True,
                        "off_policy_mix_ratio": 0.75,
                        "off_policy_buffer_size": 2048,
                        "off_policy_warmup_steps": 12,
                        "off_policy_min_buffer_groups": 96,
                        "list_piece_reward_mode": "exact_binary",
                        "list_piece_reward_weights": {
                            "typed_f1": 0.8,
                            "square_f1": 0.1,
                            "piece_recall": 0.1,
                        },
                    }
                ),
                encoding="utf-8",
            )

            args = mod.parse_args(
                [
                    "--config",
                    str(cfg_path),
                    "--no-off-policy",
                    "--list-piece-reward-mode",
                    "dense_partial_v1",
                    "--list-piece-reward-weights-json",
                    '{"typed_f1":0.5,"square_f1":0.25,"piece_recall":0.25}',
                ]
            )

            self.assertFalse(args.off_policy)
            self.assertAlmostEqual(args.off_policy_mix_ratio, 0.75, places=6)
            self.assertEqual(args.off_policy_buffer_size, 2048)
            self.assertEqual(args.off_policy_warmup_steps, 12)
            self.assertEqual(args.off_policy_min_buffer_groups, 96)
            self.assertEqual(args.list_piece_reward_mode, "dense_partial_v1")
            self.assertAlmostEqual(args.list_piece_reward_weights["typed_f1"], 0.5, places=6)
            self.assertAlmostEqual(args.list_piece_reward_weights["square_f1"], 0.25, places=6)
            self.assertAlmostEqual(args.list_piece_reward_weights["piece_recall"], 0.25, places=6)

    def test_eval_prediction_config_keys_are_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "cfg.json"
            cfg_path.write_text(
                json.dumps(
                    {
                        "dataset_source": "local_jsonl",
                        "dataset_dir": "chess_QA/synth-chess-dataset/outputs",
                        "dataset_variant_tag": "piece_position_v1",
                        "save_eval_predictions": True,
                        "eval_predictions_output_dir": "chess_QA/outputs/eval_predictions",
                        "list_piece_reward_mode": "dense_partial_v2",
                    }
                ),
                encoding="utf-8",
            )

            args = mod.parse_args(["--config", str(cfg_path)])

            self.assertTrue(args.save_eval_predictions)
            self.assertEqual(args.eval_predictions_output_dir, "chess_QA/outputs/eval_predictions")
            self.assertEqual(args.list_piece_reward_mode, "dense_partial_v2")

    def test_board_metric_config_keys_are_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "cfg.json"
            cfg_path.write_text(
                json.dumps(
                    {
                        "dataset_source": "local_jsonl",
                        "dataset_dir": "chess_QA/synth-chess-dataset/outputs",
                        "dataset_variant_tag": "piece_position_v1",
                        "best_metric": "eval_board_at_1_accuracy",
                        "checkpoint_avg_metric": "eval_typed_square_f1",
                    }
                ),
                encoding="utf-8",
            )

            args = mod.parse_args(["--config", str(cfg_path)])

            self.assertEqual(args.best_metric, "eval_board_at_1_accuracy")
            self.assertEqual(args.checkpoint_avg_metric, "eval_typed_square_f1")

    def test_osf_baseline_config_parses(self) -> None:
        args = mod.parse_args(
            ["--config", str(_repo_root() / "chess_QA" / "configs" / "query_rl_chess_piece_position_v2_osf_baseline_reasoning.json")]
        )

        self.assertEqual(args.dataset_variant_tag, "piece_position_v2_osfstorage")
        self.assertEqual(args.list_piece_reward_mode, "dense_partial_v2")
        self.assertTrue(args.save_eval_predictions)
        self.assertEqual(args.best_metric, "eval_board_at_1_accuracy")
        self.assertEqual(args.checkpoint_avg_metric, "eval_board_at_1_accuracy")

    def test_individual_task_configs_parse_with_single_active_task(self) -> None:
        config_root = _repo_root() / "chess_QA" / "configs" / "individual_tasks"
        expectations = {
            "query_rl_chess_individual_list_all_pieces_v2.json": ("list_all_pieces", 900),
            "query_rl_chess_individual_count_by_color_v2.json": ("count_by_color", 200),
            "query_rl_chess_individual_list_color_pieces_v2.json": ("list_color_pieces", 900),
            "query_rl_chess_individual_color_presence_check_v2.json": ("color_presence_check", 200),
        }

        for filename, (active_task, max_tokens) in expectations.items():
            with self.subTest(config=filename):
                args = mod.parse_args(["--config", str(config_root / filename)])

                self.assertEqual(args.dataset_variant_tag, "mixed_tasks_v2_dataset2")
                self.assertTrue(args.save_eval_predictions)
                self.assertEqual(args.eval_fixed_subset_size, 32)
                self.assertEqual(args.max_tokens, max_tokens)
                self.assertEqual(
                    {task for task, weight in args.task_sampling_weights.items() if float(weight) > 0.0},
                    {active_task},
                )

    def test_cicd_piece_position_configs_parse_with_staging_settings(self) -> None:
        config_root = _repo_root() / "chess_QA" / "configs" / "cicd"
        expectations = {
            "cicd_query_rl_chess_all_baseline_no_reasoning_rewardfix.json": (False, False, "CICID_GPUB_MOONDREAM_API_KEY_1"),
            "cicd_query_rl_chess_all_baseline_reasoning_rewardfix_group4.json": (False, True, "CICID_GPUB_MOONDREAM_API_KEY_2"),
            "cicd_query_rl_chess_all_offpolicy_no_reasoning_safe_mix.json": (True, False, "CICID_GPUB_MOONDREAM_API_KEY_3"),
            "cicd_query_rl_chess_all_offpolicy_reasoning_rank24_safe_mix.json": (True, True, "CICID_GPUB_MOONDREAM_API_KEY_4"),
        }

        for filename, (off_policy, reasoning, api_key_env_var) in expectations.items():
            with self.subTest(config=filename):
                args = mod.parse_args(["--config", str(config_root / filename)])

                self.assertEqual(args.env_file, ".env-staging")
                self.assertEqual(args.base_url, "https://api-staging.moondream.ai/v1")
                self.assertEqual(args.api_key_env_var, api_key_env_var)
                self.assertEqual(args.image_jpeg_quality, 65)
                self.assertEqual(args.off_policy, off_policy)
                self.assertEqual(args.reasoning, reasoning)
                self.assertEqual(args.best_metric, "eval_board_at_1_accuracy")
                self.assertEqual(args.checkpoint_avg_metric, "eval_board_at_1_accuracy")


class TrainingSmokeTests(unittest.TestCase):
    class _FakeRolloutResult:
        def __init__(self, answer: str, num_rollouts: int) -> None:
            self.rollouts = [
                SimpleNamespace(output=SimpleNamespace(answer=answer))
                for _ in range(max(1, int(num_rollouts)))
            ]

        def to_group(self, rewards: list[float]) -> dict[str, object]:
            return {"rewards": list(rewards)}

    class _FakeFinetune:
        def __init__(self, answer: str) -> None:
            self.finetune_id = "ft_chess_smoke"
            self.name = "ft_chess_smoke"
            self._answer = answer
            self.saved_checkpoints = 0
            self.train_steps = 0

        def rollouts_batch(
            self,
            *,
            requests: list[object],
            num_rollouts: int,
            max_workers: int,
        ) -> list["TrainingSmokeTests._FakeRolloutResult"]:
            _ = max_workers
            return [
                TrainingSmokeTests._FakeRolloutResult(self._answer, num_rollouts)
                for _ in requests
            ]

        def train_step(self, *, groups: list[object], lr: float) -> SimpleNamespace:
            _ = groups
            _ = lr
            self.train_steps += 1
            return SimpleNamespace(kl=0.01, router_kl=0.0, grad_norm=1.0)

        def save_checkpoint(self) -> SimpleNamespace:
            self.saved_checkpoints += 1
            return SimpleNamespace(
                ok=True,
                checkpoint=SimpleNamespace(
                    checkpoint_id=f"ckpt_{self.saved_checkpoints:03d}",
                    step=self.train_steps,
                ),
            )

    class _FakeClient:
        def __init__(self, *, answer: str) -> None:
            self._finetune = TrainingSmokeTests._FakeFinetune(answer)

        def create_finetune(self, *, name: str, rank: int) -> "TrainingSmokeTests._FakeFinetune":
            _ = rank
            self._finetune.name = name
            return self._finetune

        def get_finetune(self, finetune_id: str) -> "TrainingSmokeTests._FakeFinetune":
            _ = finetune_id
            return self._finetune

        def close(self) -> None:
            return

    class _UnauthorizedClient:
        def __init__(self, *, request_id: str = "req_auth_test") -> None:
            self._request_id = request_id

        def create_finetune(self, *, name: str, rank: int) -> "TrainingSmokeTests._FakeFinetune":
            _ = name
            _ = rank
            raise mod.TunaAPIError(
                "Unauthorized",
                status_code=401,
                request_id=self._request_id,
            )

        def get_finetune(self, finetune_id: str) -> "TrainingSmokeTests._FakeFinetune":
            _ = finetune_id
            raise mod.TunaAPIError(
                "Unauthorized",
                status_code=401,
                request_id=self._request_id,
            )

        def close(self) -> None:
            return

    class _FakeRun:
        def __init__(self) -> None:
            self.summary: dict[str, object] = {}
            self.finished = False

        def finish(self) -> None:
            self.finished = True

    class _FakeWandb:
        def __init__(self) -> None:
            self.logs: list[tuple[dict[str, float], int | None]] = []
            self.run: TrainingSmokeTests._FakeRun | None = None

        def init(self, *args: object, **kwargs: object) -> "TrainingSmokeTests._FakeRun":
            _ = args
            _ = kwargs
            self.run = TrainingSmokeTests._FakeRun()
            return self.run

        def log(self, payload: dict[str, float], step: int | None = None) -> None:
            self.logs.append((dict(payload), step))

    def _write_split(self, path: Path, *, split_name: str, row_id: str) -> None:
        payload = {
            "row_id": row_id,
            "split": split_name,
            "task_type": "count_by_color",
            "question": "Report how many white pieces and black pieces are present.",
            "final_answer_json": json.dumps({"white_piece_count": 1, "black_piece_count": 0}),
            "image": "/imges/board.jpg",
            "image_path": "/imges/board.jpg",
            "source_image_id": "board.jpg",
        }
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    def _write_list_piece_split(self, path: Path, *, split_name: str, row_id: str) -> None:
        payload = {
            "row_id": row_id,
            "split": split_name,
            "task_type": "list_all_pieces",
            "question": "List every chess piece and square from this image.",
            "final_answer_json": json.dumps(
                {"pieces": [{"white_king": "e1", "black_king": "e8"}]}
            ),
            "image": "/imges/board.jpg",
            "image_path": "/imges/board.jpg",
            "source_image_id": "board.jpg",
        }
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    def test_one_step_training_with_mocked_client_writes_checkpoint_ranking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "outputs"
            variant = root / "mixed_tasks_v1"
            jsonl_dir = variant / "jsonl"
            imges_dir = root / "imges"
            jsonl_dir.mkdir(parents=True, exist_ok=True)
            imges_dir.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (10, 10), color=(255, 255, 255)).save(imges_dir / "board.jpg")

            self._write_split(jsonl_dir / "train.jsonl", split_name="train", row_id="r_train")
            self._write_split(jsonl_dir / "val.jsonl", split_name="val", row_id="r_val")
            self._write_split(jsonl_dir / "test.jsonl", split_name="test", row_id="r_test")

            ranking_path = Path(tmp) / "checkpoint_ranking_smoke.json"
            cfg_path = Path(tmp) / "cfg.json"
            cfg_path.write_text(
                json.dumps(
                    {
                        "dataset_source": "local_jsonl",
                        "dataset_dir": str(root),
                        "dataset_variant_tag": "mixed_tasks_v1",
                        "train_split": "train",
                        "val_split": "val",
                        "final_eval_splits": ["val"],
                        "checkpoint_avg_splits": ["val"],
                        "checkpoint_avg_metric": "eval_reward_mean",
                        "checkpoint_ranking_output": str(ranking_path),
                        "task_sampling_weights": {
                            "count_by_color": 1.0,
                            "list_all_pieces": 0.0,
                            "list_color_pieces": 0.0,
                            "color_presence_check": 0.0,
                        },
                        "max_tokens_by_task": {
                            "count_by_color": 64,
                        },
                        "num_steps": 1,
                        "batch_size": 1,
                        "group_size": 1,
                        "eval_every": 1,
                        "save_every": 1,
                        "save_on_eval": True,
                        "eval_batch_size": 1,
                        "eval_max_samples": 1,
                        "auto_benchmark_best_checkpoint": False,
                        "no_progress": True,
                    }
                ),
                encoding="utf-8",
            )

            fake_wandb = self._FakeWandb()
            clients: list[TrainingSmokeTests._FakeClient] = []

            def _client_factory(*args: object, **kwargs: object) -> "TrainingSmokeTests._FakeClient":
                _ = args
                _ = kwargs
                client = TrainingSmokeTests._FakeClient(
                    answer=json.dumps({"white_piece_count": 1, "black_piece_count": 0})
                )
                clients.append(client)
                return client

            with patch.object(mod, "wandb", fake_wandb), patch.object(mod, "TunaClient", side_effect=_client_factory):
                mod.main(
                    [
                        "--config",
                        str(cfg_path),
                        "--api-key",
                        "test_key",
                        "--base-url",
                        "https://example.invalid/v1",
                        "--no-progress",
                    ]
                )

            self.assertTrue(ranking_path.exists())
            payload = json.loads(ranking_path.read_text(encoding="utf-8"))
            self.assertIn("rankings", payload)
            self.assertGreaterEqual(len(payload["rankings"]), 1)
            self.assertIn("saved_checkpoint_step", payload["rankings"][0])
            self.assertIn("saved_checkpoint_id", payload["rankings"][0])
            self.assertEqual(
                payload["rankings"][0]["split_metrics"]["val"]["eval_board_metric_samples"],
                0.0,
            )

            self.assertGreater(len(fake_wandb.logs), 0)
            self.assertEqual(len(clients), 1)
            self.assertGreaterEqual(clients[0]._finetune.saved_checkpoints, 1)

    def test_explicit_env_file_overrides_existing_shell_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {"MOONDREAM_API_KEY": "shell_key_should_not_win"},
            clear=False,
        ):
            root = Path(tmp) / "outputs"
            variant = root / "mixed_tasks_v1"
            jsonl_dir = variant / "jsonl"
            imges_dir = root / "imges"
            jsonl_dir.mkdir(parents=True, exist_ok=True)
            imges_dir.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (10, 10), color=(255, 255, 255)).save(imges_dir / "board.jpg")

            self._write_split(jsonl_dir / "train.jsonl", split_name="train", row_id="r_train")
            self._write_split(jsonl_dir / "val.jsonl", split_name="val", row_id="r_val")
            self._write_split(jsonl_dir / "test.jsonl", split_name="test", row_id="r_test")

            env_path = Path(tmp) / ".env-staging"
            env_path.write_text("MOONDREAM_API_KEY=file_key_should_win\n", encoding="utf-8")

            cfg_path = Path(tmp) / "cfg.json"
            cfg_path.write_text(
                json.dumps(
                    {
                        "env_file": str(env_path),
                        "dataset_source": "local_jsonl",
                        "dataset_dir": str(root),
                        "dataset_variant_tag": "mixed_tasks_v1",
                        "train_split": "train",
                        "val_split": "val",
                        "final_eval_splits": ["val"],
                        "checkpoint_avg_splits": ["val"],
                        "task_sampling_weights": {
                            "count_by_color": 1.0,
                            "list_all_pieces": 0.0,
                            "list_color_pieces": 0.0,
                            "color_presence_check": 0.0,
                        },
                        "max_tokens_by_task": {"count_by_color": 64},
                        "num_steps": 1,
                        "batch_size": 1,
                        "group_size": 1,
                        "eval_every": 1,
                        "save_every": 1,
                        "save_on_eval": True,
                        "eval_batch_size": 1,
                        "eval_max_samples": 1,
                        "auto_benchmark_best_checkpoint": False,
                        "no_progress": True,
                    }
                ),
                encoding="utf-8",
            )

            fake_wandb = self._FakeWandb()
            seen_api_keys: list[str] = []

            def _client_factory(*args: object, **kwargs: object) -> "TrainingSmokeTests._FakeClient":
                _ = args
                seen_api_keys.append(str(kwargs.get("api_key", "")))
                return TrainingSmokeTests._FakeClient(
                    answer=json.dumps({"white_piece_count": 1, "black_piece_count": 0})
                )

            with patch.object(mod, "wandb", fake_wandb), patch.object(mod, "TunaClient", side_effect=_client_factory):
                mod.main(
                    [
                        "--config",
                        str(cfg_path),
                        "--base-url",
                        "https://example.invalid/v1",
                        "--no-progress",
                    ]
                )

            self.assertEqual(seen_api_keys, ["file_key_should_win"])

    def test_api_key_env_var_selects_named_key_from_env_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {"MOONDREAM_API_KEY": "generic_key_should_not_win"},
            clear=False,
        ):
            root = Path(tmp) / "outputs"
            variant = root / "mixed_tasks_v1"
            jsonl_dir = variant / "jsonl"
            imges_dir = root / "imges"
            jsonl_dir.mkdir(parents=True, exist_ok=True)
            imges_dir.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (10, 10), color=(255, 255, 255)).save(imges_dir / "board.jpg")

            self._write_split(jsonl_dir / "train.jsonl", split_name="train", row_id="r_train")
            self._write_split(jsonl_dir / "val.jsonl", split_name="val", row_id="r_val")
            self._write_split(jsonl_dir / "test.jsonl", split_name="test", row_id="r_test")

            env_path = Path(tmp) / ".env-staging"
            env_path.write_text(
                "CICID_GPUB_MOONDREAM_API_KEY_3=file_key_from_named_env_var\n",
                encoding="utf-8",
            )

            cfg_path = Path(tmp) / "cfg.json"
            cfg_path.write_text(
                json.dumps(
                    {
                        "env_file": str(env_path),
                        "api_key_env_var": "CICID_GPUB_MOONDREAM_API_KEY_3",
                        "dataset_source": "local_jsonl",
                        "dataset_dir": str(root),
                        "dataset_variant_tag": "mixed_tasks_v1",
                        "train_split": "train",
                        "val_split": "val",
                        "final_eval_splits": ["val"],
                        "checkpoint_avg_splits": ["val"],
                        "task_sampling_weights": {
                            "count_by_color": 1.0,
                            "list_all_pieces": 0.0,
                            "list_color_pieces": 0.0,
                            "color_presence_check": 0.0,
                        },
                        "max_tokens_by_task": {"count_by_color": 64},
                        "num_steps": 1,
                        "batch_size": 1,
                        "group_size": 1,
                        "eval_every": 1,
                        "save_every": 1,
                        "save_on_eval": True,
                        "eval_batch_size": 1,
                        "eval_max_samples": 1,
                        "auto_benchmark_best_checkpoint": False,
                        "no_progress": True,
                    }
                ),
                encoding="utf-8",
            )

            fake_wandb = self._FakeWandb()
            seen_api_keys: list[str] = []

            def _client_factory(*args: object, **kwargs: object) -> "TrainingSmokeTests._FakeClient":
                _ = args
                seen_api_keys.append(str(kwargs.get("api_key", "")))
                return TrainingSmokeTests._FakeClient(
                    answer=json.dumps({"white_piece_count": 1, "black_piece_count": 0})
                )

            with patch.object(mod, "wandb", fake_wandb), patch.object(mod, "TunaClient", side_effect=_client_factory):
                mod.main(
                    [
                        "--config",
                        str(cfg_path),
                        "--base-url",
                        "https://example.invalid/v1",
                        "--no-progress",
                    ]
                )

            self.assertEqual(seen_api_keys, ["file_key_from_named_env_var"])

    def test_unauthorized_finetune_bootstrap_raises_clear_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "outputs"
            variant = root / "mixed_tasks_v1"
            jsonl_dir = variant / "jsonl"
            imges_dir = root / "imges"
            jsonl_dir.mkdir(parents=True, exist_ok=True)
            imges_dir.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (10, 10), color=(255, 255, 255)).save(imges_dir / "board.jpg")

            self._write_split(jsonl_dir / "train.jsonl", split_name="train", row_id="r_train")
            self._write_split(jsonl_dir / "val.jsonl", split_name="val", row_id="r_val")
            self._write_split(jsonl_dir / "test.jsonl", split_name="test", row_id="r_test")

            env_path = Path(tmp) / ".env-staging"
            env_path.write_text("MOONDREAM_API_KEY=file_key_for_auth_message\n", encoding="utf-8")

            cfg_path = Path(tmp) / "cfg.json"
            cfg_path.write_text(
                json.dumps(
                    {
                        "env_file": str(env_path),
                        "dataset_source": "local_jsonl",
                        "dataset_dir": str(root),
                        "dataset_variant_tag": "mixed_tasks_v1",
                        "train_split": "train",
                        "val_split": "val",
                        "final_eval_splits": ["val"],
                        "checkpoint_avg_splits": ["val"],
                        "task_sampling_weights": {
                            "count_by_color": 1.0,
                            "list_all_pieces": 0.0,
                            "list_color_pieces": 0.0,
                            "color_presence_check": 0.0,
                        },
                        "max_tokens_by_task": {"count_by_color": 64},
                        "num_steps": 1,
                        "batch_size": 1,
                        "group_size": 1,
                        "eval_every": 1,
                        "save_every": 1,
                        "save_on_eval": True,
                        "eval_batch_size": 1,
                        "eval_max_samples": 1,
                        "auto_benchmark_best_checkpoint": False,
                        "no_progress": True,
                    }
                ),
                encoding="utf-8",
            )

            fake_wandb = self._FakeWandb()

            def _client_factory(*args: object, **kwargs: object) -> "TrainingSmokeTests._UnauthorizedClient":
                _ = args
                _ = kwargs
                return TrainingSmokeTests._UnauthorizedClient(request_id="req_auth_123")

            with patch.object(mod, "wandb", fake_wandb), patch.object(mod, "TunaClient", side_effect=_client_factory):
                with self.assertRaises(SystemExit) as ctx:
                    mod.main(
                        [
                            "--config",
                            str(cfg_path),
                            "--base-url",
                            "https://api-staging.moondream.ai/v1",
                            "--no-progress",
                        ]
                    )

            message = str(ctx.exception)
            self.assertIn("Unauthorized while creating finetune", message)
            self.assertIn(str(env_path), message)
            self.assertIn("https://api-staging.moondream.ai/v1", message)
            self.assertIn("request_id=req_auth_123", message)
            self.assertIn("check_moondream_key.py", message)

    def test_checkpoint_ranking_prefers_saved_checkpoint_step(self) -> None:
        payload = mod._build_checkpoint_ranking_payload(
            finetune_id="ft_test",
            checkpoint_avg_metric="eval_reward_mean",
            checkpoint_avg_splits=["val"],
            checkpoint_eval_history=[
                {
                    "step": 19,
                    "avg_checkpoint_metric": 0.25,
                    "avg_eval_reward_mean": 0.25,
                    "checkpoint_saved": True,
                    "saved_checkpoint_id": "ckpt_saved",
                    "saved_checkpoint_step": 7,
                    "split_metrics": {"val": {"eval_reward_mean": 0.25}},
                }
            ],
        )

        self.assertEqual(payload["best_avg_checkpoint_metric_step"], 7)
        self.assertEqual(payload["best_avg_checkpoint_metric_eval_step"], 19)
        self.assertEqual(payload["best_avg_eval_reward_step"], 7)
        self.assertEqual(payload["best_avg_eval_reward_eval_step"], 19)

    def test_checkpoint_ranking_supports_board_at_1_metric(self) -> None:
        payload = mod._build_checkpoint_ranking_payload(
            finetune_id="ft_test",
            checkpoint_avg_metric="eval_board_at_1_accuracy",
            checkpoint_avg_splits=["val"],
            checkpoint_eval_history=[
                {
                    "step": 29,
                    "avg_checkpoint_metric": 0.75,
                    "avg_eval_reward_mean": 0.10,
                    "checkpoint_saved": True,
                    "saved_checkpoint_id": "ckpt_saved",
                    "saved_checkpoint_step": 11,
                    "split_metrics": {
                        "val": {
                            "eval_board_at_1_accuracy": 0.75,
                            "eval_reward_mean": 0.10,
                        }
                    },
                }
            ],
        )

        self.assertEqual(payload["checkpoint_avg_metric"], "eval_board_at_1_accuracy")
        self.assertEqual(payload["best_avg_checkpoint_metric"], 0.75)
        self.assertEqual(payload["best_avg_checkpoint_metric_step"], 11)
        self.assertEqual(payload["best_avg_checkpoint_metric_eval_step"], 29)

    def test_eval_prediction_artifacts_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "outputs"
            variant = root / "mixed_tasks_v1"
            jsonl_dir = variant / "jsonl"
            imges_dir = root / "imges"
            eval_predictions_dir = Path(tmp) / "eval_predictions"
            jsonl_dir.mkdir(parents=True, exist_ok=True)
            imges_dir.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (10, 10), color=(255, 255, 255)).save(imges_dir / "board.jpg")

            self._write_split(jsonl_dir / "train.jsonl", split_name="train", row_id="r_train")
            self._write_split(jsonl_dir / "val.jsonl", split_name="val", row_id="r_val")
            self._write_split(jsonl_dir / "test.jsonl", split_name="test", row_id="r_test")

            ranking_path = Path(tmp) / "checkpoint_ranking_eval_preds.json"
            cfg_path = Path(tmp) / "cfg.json"
            cfg_path.write_text(
                json.dumps(
                    {
                        "dataset_source": "local_jsonl",
                        "dataset_dir": str(root),
                        "dataset_variant_tag": "mixed_tasks_v1",
                        "train_split": "train",
                        "val_split": "val",
                        "final_eval_splits": ["val"],
                        "checkpoint_avg_splits": ["val"],
                        "checkpoint_avg_metric": "eval_reward_mean",
                        "checkpoint_ranking_output": str(ranking_path),
                        "task_sampling_weights": {
                            "count_by_color": 1.0,
                            "list_all_pieces": 0.0,
                            "list_color_pieces": 0.0,
                            "color_presence_check": 0.0,
                        },
                        "max_tokens_by_task": {
                            "count_by_color": 64,
                        },
                        "num_steps": 1,
                        "batch_size": 1,
                        "group_size": 1,
                        "eval_every": 1,
                        "save_every": 1,
                        "save_on_eval": True,
                        "save_eval_predictions": True,
                        "eval_predictions_output_dir": str(eval_predictions_dir),
                        "eval_batch_size": 1,
                        "eval_max_samples": 1,
                        "eval_fixed_subset_size": 1,
                        "auto_benchmark_best_checkpoint": False,
                        "no_progress": True,
                    }
                ),
                encoding="utf-8",
            )

            fake_wandb = self._FakeWandb()

            def _client_factory(*args: object, **kwargs: object) -> "TrainingSmokeTests._FakeClient":
                _ = args
                _ = kwargs
                return TrainingSmokeTests._FakeClient(
                    answer=json.dumps({"white_piece_count": 1, "black_piece_count": 0})
                )

            with patch.object(mod, "wandb", fake_wandb), patch.object(mod, "TunaClient", side_effect=_client_factory):
                mod.main(
                    [
                        "--config",
                        str(cfg_path),
                        "--api-key",
                        "test_key",
                        "--base-url",
                        "https://example.invalid/v1",
                        "--no-progress",
                    ]
                )

            artifacts = sorted(eval_predictions_dir.glob("*.jsonl"))
            self.assertGreaterEqual(len(artifacts), 1)
            rows = [json.loads(line) for line in artifacts[0].read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertGreaterEqual(len(rows), 1)
            self.assertIn("raw_answer_text", rows[0])
            self.assertIn("reward", rows[0])
            self.assertIn("row_id", rows[0])
            self.assertIn("board_square_errors", rows[0])
            self.assertIsNone(rows[0]["board_square_errors"])

    def test_list_piece_eval_prediction_artifacts_include_board_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "outputs"
            variant = root / "mixed_tasks_v1"
            jsonl_dir = variant / "jsonl"
            imges_dir = root / "imges"
            eval_predictions_dir = Path(tmp) / "eval_predictions"
            jsonl_dir.mkdir(parents=True, exist_ok=True)
            imges_dir.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (10, 10), color=(255, 255, 255)).save(imges_dir / "board.jpg")

            self._write_list_piece_split(jsonl_dir / "train.jsonl", split_name="train", row_id="r_train")
            self._write_list_piece_split(jsonl_dir / "val.jsonl", split_name="val", row_id="r_val")
            self._write_list_piece_split(jsonl_dir / "test.jsonl", split_name="test", row_id="r_test")

            ranking_path = Path(tmp) / "checkpoint_ranking_list_piece_eval_preds.json"
            cfg_path = Path(tmp) / "cfg.json"
            cfg_path.write_text(
                json.dumps(
                    {
                        "dataset_source": "local_jsonl",
                        "dataset_dir": str(root),
                        "dataset_variant_tag": "mixed_tasks_v1",
                        "train_split": "train",
                        "val_split": "val",
                        "final_eval_splits": ["val"],
                        "checkpoint_avg_splits": ["val"],
                        "checkpoint_avg_metric": "eval_board_at_1_accuracy",
                        "checkpoint_ranking_output": str(ranking_path),
                        "task_sampling_weights": {
                            "count_by_color": 0.0,
                            "list_all_pieces": 1.0,
                            "list_color_pieces": 0.0,
                            "color_presence_check": 0.0,
                        },
                        "max_tokens_by_task": {
                            "list_all_pieces": 128,
                        },
                        "save_eval_predictions": True,
                        "eval_predictions_output_dir": str(eval_predictions_dir),
                        "num_steps": 1,
                        "batch_size": 1,
                        "group_size": 1,
                        "eval_every": 1,
                        "save_every": 1,
                        "save_on_eval": True,
                        "eval_batch_size": 1,
                        "eval_max_samples": 1,
                        "eval_fixed_subset_size": 1,
                        "auto_benchmark_best_checkpoint": False,
                        "no_progress": True,
                    }
                ),
                encoding="utf-8",
            )

            fake_wandb = self._FakeWandb()

            def _client_factory(*args: object, **kwargs: object) -> "TrainingSmokeTests._FakeClient":
                _ = args
                _ = kwargs
                return TrainingSmokeTests._FakeClient(
                    answer=json.dumps({"pieces": [{"white_queen": "e1", "black_king": "e8"}]})
                )

            with patch.object(mod, "wandb", fake_wandb), patch.object(mod, "TunaClient", side_effect=_client_factory):
                mod.main(
                    [
                        "--config",
                        str(cfg_path),
                        "--api-key",
                        "test_key",
                        "--base-url",
                        "https://example.invalid/v1",
                        "--no-progress",
                    ]
                )

            artifacts = sorted(eval_predictions_dir.glob("*.jsonl"))
            self.assertGreaterEqual(len(artifacts), 1)
            rows = [json.loads(line) for line in artifacts[0].read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertGreaterEqual(len(rows), 1)
            self.assertIn("board_square_errors", rows[0])
            self.assertIn("board_at_1", rows[0])
            self.assertIn("typed_square_f1", rows[0])
            self.assertIn("pred_square_collision_count", rows[0])
            self.assertEqual(rows[0]["board_square_errors"], 1)
            self.assertTrue(rows[0]["board_at_1"])
            self.assertAlmostEqual(rows[0]["square_accuracy"], 63.0 / 64.0, places=6)

    def test_dense_reward_training_logs_non_binary_reward_mean(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "outputs"
            variant = root / "mixed_tasks_v1"
            jsonl_dir = variant / "jsonl"
            imges_dir = root / "imges"
            jsonl_dir.mkdir(parents=True, exist_ok=True)
            imges_dir.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (10, 10), color=(255, 255, 255)).save(imges_dir / "board.jpg")

            self._write_list_piece_split(jsonl_dir / "train.jsonl", split_name="train", row_id="r_train")
            self._write_list_piece_split(jsonl_dir / "val.jsonl", split_name="val", row_id="r_val")
            self._write_list_piece_split(jsonl_dir / "test.jsonl", split_name="test", row_id="r_test")

            ranking_path = Path(tmp) / "checkpoint_ranking_dense_smoke.json"
            cfg_path = Path(tmp) / "cfg.json"
            cfg_path.write_text(
                json.dumps(
                    {
                        "dataset_source": "local_jsonl",
                        "dataset_dir": str(root),
                        "dataset_variant_tag": "mixed_tasks_v1",
                        "train_split": "train",
                        "val_split": "val",
                        "final_eval_splits": ["val"],
                        "checkpoint_avg_splits": ["val"],
                        "checkpoint_avg_metric": "eval_board_at_1_accuracy",
                        "best_metric": "eval_board_at_1_accuracy",
                        "checkpoint_ranking_output": str(ranking_path),
                        "task_sampling_weights": {
                            "count_by_color": 0.0,
                            "list_all_pieces": 1.0,
                            "list_color_pieces": 0.0,
                            "color_presence_check": 0.0,
                        },
                        "max_tokens_by_task": {
                            "list_all_pieces": 128,
                        },
                        "list_piece_reward_mode": "dense_partial_v1",
                        "list_piece_reward_weights": {
                            "typed_f1": 0.6,
                            "square_f1": 0.2,
                            "piece_recall": 0.2,
                        },
                        "num_steps": 1,
                        "batch_size": 1,
                        "group_size": 1,
                        "eval_every": 1,
                        "save_every": 1,
                        "save_on_eval": True,
                        "eval_batch_size": 1,
                        "eval_max_samples": 1,
                        "auto_benchmark_best_checkpoint": False,
                        "no_progress": True,
                    }
                ),
                encoding="utf-8",
            )

            fake_wandb = self._FakeWandb()

            def _client_factory(*args: object, **kwargs: object) -> "TrainingSmokeTests._FakeClient":
                _ = args
                _ = kwargs
                return TrainingSmokeTests._FakeClient(
                    answer=json.dumps({"pieces": [{"white_queen": "e1", "black_king": "e8"}]})
                )

            with patch.object(mod, "wandb", fake_wandb), patch.object(mod, "TunaClient", side_effect=_client_factory):
                mod.main(
                    [
                        "--config",
                        str(cfg_path),
                        "--api-key",
                        "test_key",
                        "--base-url",
                        "https://example.invalid/v1",
                        "--no-progress",
                    ]
                )

            reward_values: list[float] = []
            for payload, _step in fake_wandb.logs:
                if "reward_mean" in payload:
                    reward_values.append(float(payload["reward_mean"]))

            self.assertTrue(any(0.0 < value < 1.0 for value in reward_values))
            self.assertTrue(
                any("eval_board_at_1_accuracy" in payload for payload, _step in fake_wandb.logs)
            )

            ranking_payload = json.loads(ranking_path.read_text(encoding="utf-8"))
            self.assertEqual(ranking_payload["checkpoint_avg_metric"], "eval_board_at_1_accuracy")
            self.assertEqual(
                ranking_payload["rankings"][0]["split_metrics"]["val"]["eval_board_at_1_accuracy"],
                1.0,
            )


class AsyncCheckpointEvalTests(unittest.TestCase):
    def test_async_checkpoint_eval_requires_save_on_eval(self) -> None:
        args = mod.parse_args(["--async-checkpoint-eval", "--no-save-on-eval"])
        with self.assertRaisesRegex(ValueError, "requires --save-on-eval"):
            mod._validate_args(args)

    def test_async_checkpoint_eval_rejects_early_stop(self) -> None:
        args = mod.parse_args(["--async-checkpoint-eval", "--early-stop"])
        with self.assertRaisesRegex(ValueError, "not compatible with --early-stop"):
            mod._validate_args(args)

    def test_async_checkpoint_eval_command_uses_aggregate_benchmark(self) -> None:
        args = mod.parse_args(
            [
                "--dataset-source",
                "local_jsonl",
                "--dataset-dir",
                str(_dataset_root()),
                "--dataset-variant-tag",
                "mixed_tasks_v1",
                "--env-file",
                str(REPO_ROOT / "chess_QA" / ".env"),
                "--base-url",
                "https://example.invalid/v1",
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cmd = mod._build_async_checkpoint_eval_command(
                args=args,
                finetune_id="ft_123",
                checkpoint_step=23,
                checkpoint_avg_splits=["val", "test"],
                dataset_dir=_dataset_root(),
                eval_temperature=0.0,
                eval_top_p=1.0,
                eval_reasoning=False,
                active_eval_tasks={"list_all_pieces", "count_by_color"},
                metrics_json_path=tmp_path / "metrics.json",
                predictions_jsonl_path=tmp_path / "predictions.jsonl",
            )
        self.assertTrue(str(Path(cmd[1]).resolve()).endswith("chess_QA/benchmark_chess_checkpoint_average.py"))
        self.assertIn("--checkpoint-fallback-policy", cmd)
        self.assertIn("exact", cmd)
        self.assertIn("--avg-splits", cmd)
        self.assertIn("--task-types", cmd)


if __name__ == "__main__":
    unittest.main()
