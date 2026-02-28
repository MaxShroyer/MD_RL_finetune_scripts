from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tictaktoe_QA import benchmark_ttt_query as mod


class BenchmarkArgTests(unittest.TestCase):
    def test_config_and_cli_override_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "benchmark.json"
            cfg_path.write_text(
                json.dumps(
                    {
                        "dataset_dir": "synth_dataset/outputs/smoke_full_jsonl",
                        "split": "test",
                        "max_tokens": 111,
                        "reasoning": False,
                        "task_types": ["best_move", "available_moves_count"],
                        "checkpoint_step": 148,
                    }
                ),
                encoding="utf-8",
            )

            args = mod._parse_args(
                [
                    "--config",
                    str(cfg_path),
                    "--reasoning",
                    "--max-tokens",
                    "222",
                    "--task-types",
                    "best_move,available_moves_list",
                ]
            )

            self.assertEqual(args.max_tokens, 222)
            self.assertTrue(args.reasoning)
            self.assertEqual(args.task_types, ["best_move", "available_moves_list"])
            self.assertEqual(args.checkpoint_step, 148)

    def test_task_types_from_config_when_cli_omitted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "benchmark.json"
            cfg_path.write_text(
                json.dumps(
                    {
                        "dataset_dir": "synth_dataset/outputs/smoke_full_jsonl",
                        "task_types": ["best_move", "available_moves_count"],
                    }
                ),
                encoding="utf-8",
            )

            args = mod._parse_args(["--config", str(cfg_path)])
            self.assertEqual(args.task_types, ["best_move", "available_moves_count"])

    def test_task_types_legacy_aliases_normalized(self) -> None:
        args = mod._parse_args(["--task-types", "best_move,legal_moves_count,legal_moves_list"])
        self.assertEqual(args.task_types, ["best_move", "available_moves_count", "available_moves_list"])

    def test_task_types_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown task_type"):
            mod._normalize_task_types(["best_move,bad_task"])

    def test_unknown_config_key_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "benchmark.json"
            cfg_path.write_text(
                json.dumps(
                    {
                        "dataset_source": "hf_hub",
                        "resoning": True,
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Unknown config key"):
                mod._parse_args(["--config", str(cfg_path)])

    def test_hf_dataset_source_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "benchmark.json"
            cfg_path.write_text(
                json.dumps(
                    {
                        "dataset_source": "hf_hub",
                        "hf_dataset_repo_id": "maxs-m87/tictactoe-qa-v1",
                        "hf_dataset_revision": "main",
                    }
                ),
                encoding="utf-8",
            )

            args = mod._parse_args(["--config", str(cfg_path)])
            self.assertEqual(args.dataset_source, "hf_hub")
            self.assertEqual(args.hf_dataset_repo_id, "maxs-m87/tictactoe-qa-v1")
            self.assertEqual(args.hf_dataset_revision, "main")


class QueryPayloadTests(unittest.TestCase):
    def test_reasoning_field_present_when_set(self) -> None:
        payload = mod._build_query_payload(
            model="model",
            question="q",
            image_url="data:image/jpeg;base64,abc",
            temperature=0.0,
            top_p=1.0,
            max_tokens=32,
            reasoning=True,
        )
        self.assertIn("reasoning", payload)
        self.assertTrue(payload["reasoning"])

    def test_reasoning_field_absent_when_none(self) -> None:
        payload = mod._build_query_payload(
            model="model",
            question="q",
            image_url="data:image/jpeg;base64,abc",
            temperature=0.0,
            top_p=1.0,
            max_tokens=32,
            reasoning=None,
        )
        self.assertNotIn("reasoning", payload)


if __name__ == "__main__":
    unittest.main()
