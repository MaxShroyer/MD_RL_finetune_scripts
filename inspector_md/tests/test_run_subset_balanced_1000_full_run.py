from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inspector_md import run_subset_balanced_1000_full_run as run_mod


class SubsetFullRunTests(unittest.TestCase):
    def test_parse_detect_point_training_log_extracts_metric_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "detect.log"
            log_path.write_text(
                "\n".join(
                    [
                        "resolved_finetune_id=ft_detect_123",
                        "eval step 40 tasks=68 pos_tasks=40 neg_tasks=28 miou=0.3210 f1=0.5500 macro_f1=0.5000 pos_f1=0.6100 neg_f1=0.4700 positive_f1=0.6100 updates=40",
                        "done. finetune_id=ft_detect_123 best_step=40 best_metric=0.6100 recall_gate_pass=None f1_target_pass=None stopped_early=False",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            parsed = run_mod._parse_detect_point_training_log(log_path)
            self.assertEqual(parsed["resolved_finetune_id"], "ft_detect_123")
            self.assertEqual(parsed["best_step"], 40)
            self.assertAlmostEqual(parsed["best_metric"], 0.61)
            self.assertIsNone(parsed["recall_gate_pass"])
            self.assertIsNone(parsed["f1_target_pass"])
            self.assertFalse(parsed["stopped_early"])

    def test_pick_detect_or_point_candidate_prefers_sft_on_regression(self) -> None:
        candidate = run_mod._pick_detect_or_point_candidate(
            sft_run={"resolved_finetune_id": "ft_1", "best_metric": 0.62, "best_step": 40},
            rl_run={"resolved_finetune_id": "ft_1", "best_metric": 0.58, "best_step": 80},
        )
        self.assertEqual(candidate["preferred_stage"], "sft")
        self.assertEqual(candidate["finetune_id"], "ft_1")
        self.assertEqual(candidate["preferred_checkpoint_step"], 40)

    def test_build_train_command_uses_reasoning_rl_shape(self) -> None:
        args = run_mod.parse_args(
            [
                "--stages",
                "train",
            ]
        )
        reasoning_spec = next(spec for spec in run_mod._train_specs(args) if spec["key"] == "query_reasoning")
        command = run_mod._build_train_command(
            reasoning_spec,
            dependency_finetune_id="",
            python_executable=sys.executable,
        )
        self.assertIn("--mode", command)
        self.assertIn("rl", command)
        self.assertIn("--reasoning", command)
        self.assertIn("--sft-steps", command)
        self.assertIn("0", command)


if __name__ == "__main__":
    unittest.main()
