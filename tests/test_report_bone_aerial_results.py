from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import report_bone_aerial_results as mod


class ExtractWandbScalarTests(unittest.TestCase):
    def test_extracts_simple_scalar_value(self) -> None:
        text = (
            "config:\n"
            "    value: /tmp/demo.json\n"
            "finetune_id:\n"
            "    value: 01ABC\n"
        )
        self.assertEqual(mod._extract_wandb_scalar(text, "config"), "/tmp/demo.json")
        self.assertEqual(mod._extract_wandb_scalar(text, "finetune_id"), "01ABC")

    def test_returns_none_for_missing_or_null_value(self) -> None:
        text = "config:\n    value: null\n"
        self.assertIsNone(mod._extract_wandb_scalar(text, "config"))
        self.assertIsNone(mod._extract_wandb_scalar(text, "missing"))


class SelectionTests(unittest.TestCase):
    def _run(
        self,
        *,
        run_id: str,
        config_path: str,
        best_eval_f1: float | None,
        eval_f1: float | None,
        eval_f1_macro: float | None,
        created_at: str,
        state: str = "finished",
        best_eval_miou: float | None = None,
    ) -> mod.RemoteRun:
        return mod.RemoteRun(
            project="demo",
            run_id=run_id,
            name=run_id,
            state=state,
            created_at=created_at,
            url=None,
            config_path=config_path,
            finetune_id=None,
            best_step=10,
            best_checkpoint_step=None,
            best_eval_f1=best_eval_f1,
            best_eval_f1_macro=None,
            best_eval_miou=best_eval_miou,
            eval_f1=eval_f1,
            eval_f1_macro=eval_f1_macro,
            eval_miou=0.0,
            test_f1=None,
            test_f1_macro=None,
            test_miou=None,
        )

    def test_select_angle_only_tie_leaders_orders_by_current_eval_then_macro_then_recency(self) -> None:
        runs = [
            self._run(
                run_id="primary",
                config_path="bone_fracture/configs/cicd/cicd_train_bone_fracture_point_angle_only_recall_primary_anchor.json",
                best_eval_f1=0.7,
                eval_f1=0.5,
                eval_f1_macro=0.5,
                created_at="2026-03-23T04:38:21Z",
            ),
            self._run(
                run_id="anchor",
                config_path="bone_fracture/configs/cicd/cicd_train_bone_fracture_point_angle_only_recall_offpolicy_anchor.json",
                best_eval_f1=0.7,
                eval_f1=0.6,
                eval_f1_macro=0.6,
                created_at="2026-03-23T04:38:41Z",
            ),
            self._run(
                run_id="lite",
                config_path="bone_fracture/configs/cicd/cicd_train_bone_fracture_point_angle_only_recall_offpolicy_lite.json",
                best_eval_f1=0.7,
                eval_f1=0.4444444444444444,
                eval_f1_macro=0.4733333333333334,
                created_at="2026-03-23T04:38:45Z",
            ),
            self._run(
                run_id="lower",
                config_path="bone_fracture/configs/cicd/cicd_train_bone_fracture_point_angle_only_recall_offpolicy_lite.json",
                best_eval_f1=0.6666666666666666,
                eval_f1=0.5,
                eval_f1_macro=0.5,
                created_at="2026-03-23T06:17:58Z",
            ),
        ]
        ordered = mod._select_angle_only_tie_leaders(runs)
        self.assertEqual([run.run_id for run in ordered], ["anchor", "primary", "lite"])

    def test_best_finished_run_ignores_non_finished_state(self) -> None:
        finished = self._run(
            run_id="finished",
            config_path="bone_fracture/configs/cicd/demo.json",
            best_eval_f1=0.1,
            eval_f1=0.1,
            eval_f1_macro=0.1,
            created_at="2026-03-23T01:00:00Z",
            state="finished",
            best_eval_miou=0.4,
        )
        running = self._run(
            run_id="running",
            config_path="bone_fracture/configs/cicd/demo.json",
            best_eval_f1=0.9,
            eval_f1=0.9,
            eval_f1_macro=0.9,
            created_at="2026-03-23T02:00:00Z",
            state="running",
            best_eval_miou=0.9,
        )
        winner = mod._best_finished_run([finished, running], "best_eval_miou")
        assert winner is not None
        self.assertEqual(winner.run_id, "finished")


class BenchmarkParsingTests(unittest.TestCase):
    def test_parse_model_finetune_checkpoint(self) -> None:
        finetune_id, step = mod._parse_model_finetune_checkpoint("moondream3-preview/01ABC@129")
        self.assertEqual(finetune_id, "01ABC")
        self.assertEqual(step, 129)


if __name__ == "__main__":
    unittest.main()
