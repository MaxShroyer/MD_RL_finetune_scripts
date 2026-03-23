from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tictaktoe_QA import run_best_checkpoint_benchmarks as mod


class BestCheckpointBenchmarkUtilityTests(unittest.TestCase):
    def test_load_manifest_rejects_missing_finetune_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    [
                        {
                            "task": "best_move",
                            "checkpoint_step": 79,
                        }
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "missing finetune_id"):
                mod._load_manifest(manifest_path)

    def test_load_manifest_rejects_invalid_checkpoint_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    [
                        {
                            "task": "winner",
                            "finetune_id": "ft-123",
                            "checkpoint_step": "abc",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "invalid checkpoint_step"):
                mod._load_manifest(manifest_path)

    def test_load_manifest_rejects_duplicate_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    [
                        {
                            "task": "is_terminal",
                            "finetune_id": "ft-123",
                            "checkpoint_step": 89,
                        },
                        {
                            "task": "is_game_over",
                            "finetune_id": "ft-456",
                            "checkpoint_step": 99,
                        },
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Duplicate task"):
                mod._load_manifest(manifest_path)

    def test_build_candidate_command_includes_task_finetune_and_step(self) -> None:
        entry = mod.ManifestEntry(task="best_move", finetune_id="ft-123", checkpoint_step=79)
        cmd = mod._build_candidate_command(
            benchmark_config=Path("/tmp/benchmark.json"),
            entry=entry,
            output_json=Path("/tmp/candidate_metrics.json"),
            predictions_jsonl=Path("/tmp/candidate_predictions.jsonl"),
            max_samples=200,
            no_progress=True,
        )
        self.assertEqual(cmd[0], mod.sys.executable)
        self.assertIn("--task-types", cmd)
        self.assertEqual(cmd[cmd.index("--task-types") + 1], "best_move")
        self.assertIn("--finetune-id", cmd)
        self.assertEqual(cmd[cmd.index("--finetune-id") + 1], "ft-123")
        self.assertIn("--checkpoint-step", cmd)
        self.assertEqual(cmd[cmd.index("--checkpoint-step") + 1], "79")
        self.assertIn("--max-samples", cmd)
        self.assertEqual(cmd[cmd.index("--max-samples") + 1], "200")
        self.assertIn("--no-progress", cmd)

    def test_build_baseline_command_uses_baseline_model_and_clears_finetune(self) -> None:
        cmd = mod._build_baseline_command(
            benchmark_config=Path("/tmp/benchmark.json"),
            task="is_game_over",
            output_json=Path("/tmp/baseline_metrics.json"),
            predictions_jsonl=None,
            max_samples=None,
            no_progress=False,
        )
        self.assertIn("--task-types", cmd)
        self.assertEqual(cmd[cmd.index("--task-types") + 1], "is_game_over")
        self.assertIn("--model", cmd)
        self.assertEqual(cmd[cmd.index("--model") + 1], mod.DEFAULT_BASELINE_MODEL)
        self.assertIn("--finetune-id", cmd)
        self.assertEqual(cmd[cmd.index("--finetune-id") + 1], "")
        self.assertIn("--checkpoint-step", cmd)
        self.assertEqual(cmd[cmd.index("--checkpoint-step") + 1], "-1")
        self.assertIn("--predictions-jsonl", cmd)
        self.assertEqual(cmd[cmd.index("--predictions-jsonl") + 1], "")

    def test_build_summary_row_computes_delta_and_failed_status(self) -> None:
        entry = mod.ManifestEntry(task="winner", finetune_id="ft-123", checkpoint_step=99)
        row = mod._build_summary_row(
            entry=entry,
            candidate_output_json=Path("/tmp/candidate_metrics.json"),
            candidate_payload={
                "by_task": {
                    "winner": {
                        "accuracy": 0.8,
                        "count": 29,
                    }
                }
            },
            candidate_return_code=0,
            baseline_output_json=Path("/tmp/baseline_metrics.json"),
            baseline_payload=None,
            baseline_return_code=1,
            include_baseline=True,
        )
        self.assertAlmostEqual(float(row["candidate_accuracy"]), 0.8, places=6)
        self.assertIsNone(row["baseline_accuracy"])
        self.assertIsNone(row["delta_vs_baseline"])
        self.assertEqual(int(row["count"]), 29)
        self.assertEqual(row["status"], "failed")


class BestCheckpointBenchmarkMainTests(unittest.TestCase):
    def test_main_runs_candidate_and_baseline_and_writes_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "manifest.json"
            benchmark_config = root / "benchmark_default.json"
            output_dir = root / "outputs"

            manifest_path.write_text(
                json.dumps(
                    [
                        {
                            "task": "best_move",
                            "finetune_id": "ft-best-move",
                            "checkpoint_step": 79,
                        },
                        {
                            "task": "is_game_over",
                            "finetune_id": "ft-game-over",
                            "checkpoint_step": 89,
                        },
                    ]
                ),
                encoding="utf-8",
            )
            benchmark_config.write_text("{}", encoding="utf-8")

            def _fake_subprocess_run(cmd, text, capture_output, cwd):  # type: ignore[no-untyped-def]
                output_json = Path(cmd[cmd.index("--output-json") + 1])
                task = cmd[cmd.index("--task-types") + 1]
                model = cmd[cmd.index("--model") + 1]
                output_json.parent.mkdir(parents=True, exist_ok=True)
                if model == mod.DEFAULT_BASELINE_MODEL:
                    accuracy = 0.3 if task == "best_move" else 0.55
                else:
                    accuracy = 0.42 if task == "best_move" else 0.9
                output_json.write_text(
                    json.dumps(
                        {
                            "by_task": {
                                task: {
                                    "accuracy": accuracy,
                                    "count": 40 if task == "best_move" else 32,
                                }
                            }
                        }
                    ),
                    encoding="utf-8",
                )
                return SimpleNamespace(returncode=0, stdout=f"ok {task}", stderr="")

            with patch("subprocess.run", side_effect=_fake_subprocess_run):
                mod.main(
                    [
                        "--manifest",
                        str(manifest_path),
                        "--benchmark-config",
                        str(benchmark_config),
                        "--output-dir",
                        str(output_dir),
                        "--no-progress",
                    ]
                )

            run_dirs = sorted(output_dir.glob("*"))
            self.assertEqual(len(run_dirs), 1)
            summary_path = run_dirs[0] / "summary.json"
            summary_csv_path = run_dirs[0] / "summary.csv"
            self.assertTrue(summary_path.exists())
            self.assertTrue(summary_csv_path.exists())

            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["results"]), 2)

            best_move_row = payload["results"][0]
            self.assertEqual(best_move_row["task"], "best_move")
            self.assertAlmostEqual(float(best_move_row["candidate_accuracy"]), 0.42, places=6)
            self.assertAlmostEqual(float(best_move_row["baseline_accuracy"]), 0.3, places=6)
            self.assertAlmostEqual(float(best_move_row["delta_vs_baseline"]), 0.12, places=6)
            self.assertEqual(int(best_move_row["count"]), 40)
            self.assertEqual(best_move_row["status"], "completed")

            is_game_over_row = payload["results"][1]
            self.assertEqual(is_game_over_row["task"], "is_game_over")
            self.assertAlmostEqual(float(is_game_over_row["candidate_accuracy"]), 0.9, places=6)
            self.assertAlmostEqual(float(is_game_over_row["baseline_accuracy"]), 0.55, places=6)
            self.assertEqual(is_game_over_row["status"], "completed")


if __name__ == "__main__":
    unittest.main()
