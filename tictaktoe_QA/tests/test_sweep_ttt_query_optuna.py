from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tictaktoe_QA import sweep_ttt_query_optuna as mod


class SweepStageSpecTests(unittest.TestCase):
    def test_default_stage_trial_counts(self) -> None:
        base_config = {"seed": 42}
        sweep_config = copy.deepcopy(mod.DEFAULT_SWEEP_CONFIG)

        stage_a_specs = mod._build_stage_a_specs(base_config, sweep_config)
        self.assertEqual(len(stage_a_specs), 4)

        stage_a_results = [
            {"trial_key": spec.trial_key, "resolved_config": spec.resolved_config, "status": "completed", "objective": 1.0 - idx * 0.1}
            for idx, spec in enumerate(stage_a_specs[:2])
        ]
        stage_b_specs = mod._build_stage_b_specs(
            stage_a_results,
            sweep_config,
            search_reasoning_stage_b=False,
        )
        self.assertEqual(len(stage_b_specs), 6)

        stage_b_results = [
            {
                "trial_key": spec.trial_key,
                "resolved_config": spec.resolved_config,
                "status": "completed",
                "objective": 1.0,
                "stopped_early": False,
                "collapse_detected": False,
            }
            for spec in stage_b_specs[:2]
        ]
        stage_c_specs = mod._build_stage_c_specs(stage_b_results, sweep_config)
        self.assertEqual(len(stage_c_specs), 6)

        final_specs = mod._build_final_specs(stage_b_results, final_num_steps=200, final_seed_count=2)
        self.assertEqual(len(final_specs), 4)

    def test_stage_b_reasoning_default_and_optional_search(self) -> None:
        sweep_config = copy.deepcopy(mod.DEFAULT_SWEEP_CONFIG)
        parent_results = [
            {"trial_key": "p0", "resolved_config": {"reasoning": True}},
            {"trial_key": "p1", "resolved_config": {"reasoning": True}},
        ]

        default_specs = mod._build_stage_b_specs(
            parent_results,
            sweep_config,
            search_reasoning_stage_b=False,
        )
        self.assertEqual(len(default_specs), 6)
        self.assertTrue(all(spec.params["reasoning"] is True for spec in default_specs))

        expanded_specs = mod._build_stage_b_specs(
            parent_results,
            sweep_config,
            search_reasoning_stage_b=True,
        )
        self.assertEqual(len(expanded_specs), 12)
        reasoning_values = {bool(spec.params["reasoning"]) for spec in expanded_specs}
        self.assertEqual(reasoning_values, {True, False})

    def test_stage_c_conditional_ratio_075(self) -> None:
        sweep_config = copy.deepcopy(mod.DEFAULT_SWEEP_CONFIG)
        parent_results = [
            {
                "trial_key": "stable_parent",
                "resolved_config": {"seed": 42},
                "status": "completed",
                "stopped_early": False,
                "collapse_detected": False,
            },
            {
                "trial_key": "unstable_parent",
                "resolved_config": {"seed": 43},
                "status": "completed",
                "stopped_early": True,
                "collapse_detected": False,
            },
        ]

        specs = mod._build_stage_c_specs(parent_results, sweep_config)
        stable_ratios = {
            float(spec.params["off_policy_mix_ratio"])
            for spec in specs
            if spec.parent_key == "stable_parent"
        }
        unstable_ratios = {
            float(spec.params["off_policy_mix_ratio"])
            for spec in specs
            if spec.parent_key == "unstable_parent"
        }

        self.assertIn(0.75, stable_ratios)
        self.assertNotIn(0.0, stable_ratios)
        self.assertIn(0.0, unstable_ratios)
        self.assertNotIn(0.75, unstable_ratios)


class SweepUtilityTests(unittest.TestCase):
    def test_resolve_parallelism_auto_with_cap(self) -> None:
        with patch("os.cpu_count", return_value=16):
            value = mod._resolve_parallelism("auto", 4)
        self.assertEqual(value, 4)

    def test_objective_from_payload_prefers_metric_and_falls_back(self) -> None:
        payload = {
            "best_avg_eval_reward": 0.7,
            "nested": {"value": 0.9},
        }
        self.assertAlmostEqual(mod._objective_from_ranking_payload(payload, "nested.value"), 0.9, places=6)
        self.assertAlmostEqual(mod._objective_from_ranking_payload(payload, "does.not.exist"), 0.7, places=6)

    def test_aggregate_final_results_orders_tiebreakers(self) -> None:
        final_results = [
            {"status": "completed", "parent_key": "a", "objective": 0.8, "parse_rate": 0.70, "trial_key": "a0"},
            {"status": "completed", "parent_key": "a", "objective": 0.8, "parse_rate": 0.70, "trial_key": "a1"},
            {"status": "completed", "parent_key": "b", "objective": 0.8, "parse_rate": 0.80, "trial_key": "b0"},
            {"status": "completed", "parent_key": "b", "objective": 0.8, "parse_rate": 0.80, "trial_key": "b1"},
        ]
        rows = mod._aggregate_final_results(final_results)
        self.assertEqual(rows[0]["parent_key"], "b")


class SweepExecutionTests(unittest.TestCase):
    def test_run_stage_handles_subprocess_failure_without_crashing(self) -> None:
        spec = mod.TrialSpec(
            stage="stage_a",
            trial_key="trial_fail",
            params={"lr": 1e-4},
            resolved_config={"seed": 42, "num_steps": 1},
        )

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)

            with patch(
                "subprocess.run",
                return_value=SimpleNamespace(returncode=1, stdout="", stderr="boom"),
            ):
                results = mod._run_stage(
                    stage_name="stage_a",
                    specs=[spec],
                    study=None,
                    trial_output_dir=out_dir,
                    objective_metric="best_avg_eval_reward",
                    dry_run=False,
                    parallelism=1,
                    resume=False,
                )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["status"], "failed")
        self.assertEqual(results[0]["trial_key"], "trial_fail")

    def test_stage_runs_across_shared_study_without_dynamic_space_errors(self) -> None:
        if mod.optuna is None:
            self.skipTest("optuna not installed")

        def fake_run_trial(
            spec: mod.TrialSpec,
            *,
            trial_output_dir: Path,
            objective_metric: str,
            dry_run: bool,
        ) -> dict[str, object]:
            return {
                "stage": spec.stage,
                "trial_key": spec.trial_key,
                "parent_key": spec.parent_key,
                "params": dict(spec.params),
                "resolved_config": dict(spec.resolved_config),
                "config_path": str(trial_output_dir / spec.stage / spec.trial_key / "config.json"),
                "ranking_path": str(trial_output_dir / spec.stage / spec.trial_key / "checkpoint_ranking.json"),
                "log_path": str(trial_output_dir / spec.stage / spec.trial_key / "train.log"),
                "status": "completed",
                "objective": 0.9,
                "parse_rate": 0.8,
                "finetune_id": "ft_dummy",
                "stopped_early": False,
                "stop_reason": "",
                "collapse_detected": False,
                "return_code": 0,
            }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            study = mod._open_study(
                storage_path=root / "optuna.db",
                study_name="sweep_dynamic_space_guard",
                seed=42,
            )
            stage_a_specs = [
                mod.TrialSpec(
                    stage="stage_a",
                    trial_key="stage_a_lr_1e4",
                    params={"lr": 1e-4},
                    resolved_config={"lr": 1e-4},
                )
            ]
            stage_c_specs = [
                mod.TrialSpec(
                    stage="stage_c",
                    trial_key="stage_c_mix_025",
                    params={"off_policy_mix_ratio": 0.25, "off_policy_warmup_steps": 10},
                    resolved_config={"off_policy_mix_ratio": 0.25, "off_policy_warmup_steps": 10},
                )
            ]

            with patch("tictaktoe_QA.sweep_ttt_query_optuna._run_trial", side_effect=fake_run_trial):
                stage_a_results = mod._run_stage(
                    stage_name="stage_a",
                    specs=stage_a_specs,
                    study=study,
                    trial_output_dir=root / "runs",
                    objective_metric="best_avg_eval_reward",
                    dry_run=False,
                    parallelism=1,
                    resume=False,
                )
                stage_c_results = mod._run_stage(
                    stage_name="stage_c",
                    specs=stage_c_specs,
                    study=study,
                    trial_output_dir=root / "runs",
                    objective_metric="best_avg_eval_reward",
                    dry_run=False,
                    parallelism=1,
                    resume=False,
                )

        self.assertEqual(stage_a_results[0]["status"], "completed")
        self.assertEqual(stage_c_results[0]["status"], "completed")


if __name__ == "__main__":
    unittest.main()
