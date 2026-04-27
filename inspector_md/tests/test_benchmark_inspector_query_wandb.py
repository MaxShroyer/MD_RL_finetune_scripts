from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from inspector_md import benchmark_inspector_query as benchmark_mod
from inspector_md import train_inspector_query as train_query_mod


class BenchmarkInspectorQueryWandbTests(unittest.TestCase):
    def test_run_benchmark_logs_eval_metrics_to_wandb(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            output_json = tmp / "benchmark.metrics.json"
            predictions_jsonl = tmp / "benchmark.predictions.jsonl"
            api_key_pool = SimpleNamespace(
                slots=[SimpleNamespace(api_key="test-key")],
                env_var_names=["TEST_MOONDREAM_API_KEY"],
            )
            fake_run = SimpleNamespace(summary={}, url="https://wandb.example/benchmark-run")
            init_calls: list[dict[str, object]] = []
            log_calls: list[tuple[dict[str, float], int]] = []

            class _FakeGrader:
                def __init__(self, **kwargs) -> None:
                    self.model_id = str(kwargs["model_id"])
                    self.profile = str(kwargs["profile"])
                    self.rubric_version = str(kwargs["rubric_version"])

            args = benchmark_mod.parse_args(
                [
                    "--api-key",
                    "test-key",
                    "--grader-api-key",
                    "grader-key",
                    "--dataset-dir",
                    str(tmp / "dataset"),
                    "--output-json",
                    str(output_json),
                    "--predictions-jsonl",
                    str(predictions_jsonl),
                    "--max-samples",
                    "2",
                    "--wandb-project",
                    "benchmark-proj",
                    "--wandb-run-name",
                    "benchmark-run",
                ]
            )

            with patch.object(benchmark_mod.common, "maybe_load_env_file", return_value=None), patch.object(
                benchmark_mod.common,
                "resolve_api_key_pool",
                return_value=api_key_pool,
            ), patch.object(
                benchmark_mod,
                "_load_split_examples",
                return_value=[SimpleNamespace(task_type="issues", row_id="row-1")],
            ), patch.object(
                benchmark_mod.openrouter_grader,
                "resolve_openrouter_api_key",
                return_value="grader-key",
            ), patch.object(
                benchmark_mod.openrouter_grader,
                "OpenRouterGrader",
                _FakeGrader,
            ), patch.object(
                benchmark_mod,
                "MoondreamInspectorClient",
                return_value=SimpleNamespace(),
            ), patch.object(
                benchmark_mod,
                "_evaluate_split",
                return_value={
                    "count": 2.0,
                    "reward_mean": 0.5,
                    "parse_rate": 1.0,
                },
            ), patch.object(
                benchmark_mod,
                "_build_benchmark_diagnostics",
                return_value={
                    "strict_reward_mean": 0.4,
                    "metric_warnings": ["example warning"],
                    "headline_metrics": {"reward_mean": 0.5},
                },
            ), patch.object(
                benchmark_mod.wandb,
                "init",
                side_effect=lambda **kwargs: init_calls.append(dict(kwargs)) or fake_run,
                create=True,
            ), patch.object(
                benchmark_mod.wandb,
                "log",
                side_effect=lambda payload, step: log_calls.append((dict(payload), int(step))),
                create=True,
            ), patch.object(
                benchmark_mod.wandb,
                "Api",
                object,
                create=True,
            ):
                summary = benchmark_mod.run_benchmark(args)

            self.assertEqual(init_calls[0]["project"], "benchmark-proj")
            self.assertEqual(init_calls[0]["name"], "benchmark-run")
            payload, step = log_calls[0]
            self.assertEqual(step, 0)
            self.assertAlmostEqual(payload["eval/count"], 2.0)
            self.assertAlmostEqual(payload["eval/reward_mean"], 0.5)
            self.assertAlmostEqual(payload["eval/parse_rate"], 1.0)
            self.assertAlmostEqual(payload["eval/strict_reward_mean"], 0.4)
            self.assertEqual(summary["wandb_url"], "https://wandb.example/benchmark-run")
            self.assertEqual(fake_run.summary["benchmark_split"], "validation")
            self.assertEqual(fake_run.summary["benchmark_model"], "moondream3-preview")
            self.assertEqual(fake_run.summary["benchmark_metric_warnings"], "example warning")
            self.assertAlmostEqual(fake_run.summary["eval/reward_mean"], 0.5)

    def test_train_eval_command_propagates_benchmark_client_retry_settings(self) -> None:
        args = train_query_mod.parse_args(
            [
                "--api-key",
                "test-key",
                "--grader-api-key",
                "grader-key",
                "--dataset-dir",
                "inspector_md/outputs/v2_launch_20260415/subset_balanced_1000/inspector_query_issues_v2",
                "--run-output-dir",
                "inspector_md/outputs/tmp_query_retry_check",
                "--mode",
                "sft",
                "--sft-steps",
                "0",
                "--rl-steps",
                "0",
                "--client-max-retries",
                "0",
                "--client-backoff-base-s",
                "1",
                "--client-backoff-max-s",
                "1",
                "--post-create-warmup-s",
                "0",
            ]
        )
        command = train_query_mod._build_async_query_eval_command(
            args=args,
            finetune_id="ft-test",
            checkpoint_step=7,
            metrics_json_path=Path("/tmp/metrics.json"),
            predictions_jsonl_path=Path("/tmp/predictions.jsonl"),
        )
        self.assertIn("--client-max-retries", command)
        self.assertIn("--client-backoff-base-s", command)
        self.assertIn("--client-backoff-max-s", command)
        self.assertEqual(command[command.index("--client-max-retries") + 1], "0")
        self.assertEqual(command[command.index("--client-backoff-base-s") + 1], "1.0")
        self.assertEqual(command[command.index("--client-backoff-max-s") + 1], "1.0")


if __name__ == "__main__":
    unittest.main()
