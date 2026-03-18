from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tictaktoe_QA import run_best_move_lr_sweep as mod


class LRSweepUtilityTests(unittest.TestCase):
    def test_parse_lrs(self) -> None:
        self.assertEqual(mod._parse_lrs("1e-4, 2e-4,5e-4"), [1e-4, 2e-4, 5e-4])
        with self.assertRaises(ValueError):
            mod._parse_lrs(" , ")

    def test_build_run_command_includes_expected_overrides(self) -> None:
        cmd = mod._build_run_command(
            base_config=Path("/tmp/base.json"),
            lr_value=2e-4,
            seed=42,
            group_size=6,
            ranking_output=Path("/tmp/ranking.json"),
            benchmark_output_json=Path("/tmp/benchmark.json"),
            benchmark_predictions_jsonl=Path("/tmp/preds.jsonl"),
            num_steps=120,
            no_progress=True,
        )
        self.assertIn("--lr", cmd)
        self.assertEqual(cmd[cmd.index("--lr") + 1], "0.0002")
        self.assertIn("--group-size", cmd)
        self.assertEqual(cmd[cmd.index("--group-size") + 1], "6")
        self.assertIn("--num-steps", cmd)
        self.assertEqual(cmd[cmd.index("--num-steps") + 1], "120")
        self.assertIn("--no-progress", cmd)

    def test_rank_results_prefers_checkpoint_metric_then_eval_then_parse_rate(self) -> None:
        results = [
            {
                "lr": 1e-4,
                "status": "completed",
                "best_avg_checkpoint_metric": 0.6,
                "auto_eval_best_move_set_accuracy": 0.4,
                "auto_eval_json_parse_rate": 0.95,
            },
            {
                "lr": 2e-4,
                "status": "completed",
                "best_avg_checkpoint_metric": 0.6,
                "auto_eval_best_move_set_accuracy": 0.5,
                "auto_eval_json_parse_rate": 0.8,
            },
            {
                "lr": 5e-4,
                "status": "failed",
                "best_avg_checkpoint_metric": 0.99,
                "auto_eval_best_move_set_accuracy": 0.99,
                "auto_eval_json_parse_rate": 0.99,
            },
        ]
        ranked = mod._rank_results(results)
        self.assertEqual(ranked[0]["lr"], 2e-4)
        self.assertEqual(ranked[-1]["status"], "failed")


class LRSweepMainTests(unittest.TestCase):
    def test_main_runs_trials_and_writes_ranked_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_config = root / "base.json"
            base_config.write_text("{}", encoding="utf-8")
            output_dir = root / "out"

            def _fake_subprocess_run(cmd, text, capture_output):  # type: ignore[no-untyped-def]
                lr_value = float(cmd[cmd.index("--lr") + 1])
                ranking_path = Path(cmd[cmd.index("--checkpoint-ranking-output") + 1])
                benchmark_path = Path(cmd[cmd.index("--auto-benchmark-output-json") + 1])
                ranking_path.parent.mkdir(parents=True, exist_ok=True)
                benchmark_path.parent.mkdir(parents=True, exist_ok=True)

                if abs(lr_value - 1e-4) < 1e-12:
                    ranking = {"best_avg_checkpoint_metric": 0.4, "best_avg_checkpoint_metric_step": 20}
                    benchmark = {"eval_best_move_set_accuracy": 0.45, "eval_json_parse_rate": 0.9}
                elif abs(lr_value - 2e-4) < 1e-12:
                    ranking = {"best_avg_checkpoint_metric": 0.6, "best_avg_checkpoint_metric_step": 40}
                    benchmark = {"eval_best_move_set_accuracy": 0.4, "eval_json_parse_rate": 0.95}
                else:
                    ranking = {"best_avg_checkpoint_metric": 0.6, "best_avg_checkpoint_metric_step": 60}
                    benchmark = {"eval_best_move_set_accuracy": 0.5, "eval_json_parse_rate": 0.8}

                ranking_path.write_text(json.dumps(ranking), encoding="utf-8")
                benchmark_path.write_text(json.dumps(benchmark), encoding="utf-8")
                return SimpleNamespace(returncode=0, stdout="ok", stderr="")

            with patch("subprocess.run", side_effect=_fake_subprocess_run):
                mod.main(
                    [
                        "--base-config",
                        str(base_config),
                        "--output-dir",
                        str(output_dir),
                        "--seed",
                        "42",
                        "--group-size",
                        "6",
                        "--lrs",
                        "1e-4,2e-4,5e-4",
                        "--num-steps",
                        "100",
                        "--no-progress",
                    ]
                )

            sweep_dirs = sorted(output_dir.glob("sweep_*"))
            self.assertEqual(len(sweep_dirs), 1)
            summary_path = sweep_dirs[0] / "summary.json"
            summary_csv_path = sweep_dirs[0] / "summary.csv"
            self.assertTrue(summary_path.exists())
            self.assertTrue(summary_csv_path.exists())

            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(len(payload.get("results", [])), 3)
            top = payload["results"][0]
            self.assertAlmostEqual(float(top["lr"]), 5e-4, places=12)
            self.assertAlmostEqual(float(top["best_avg_checkpoint_metric"]), 0.6, places=6)
            self.assertAlmostEqual(float(top["auto_eval_best_move_set_accuracy"]), 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
