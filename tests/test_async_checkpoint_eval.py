from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import async_checkpoint_eval as async_mod
import finetune_checkpoints as checkpoints_mod


def _write_script(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")


class SaveCheckpointStepTests(unittest.TestCase):
    def test_checkpoint_step_from_save_result_coerces_int(self) -> None:
        save_result = SimpleNamespace(checkpoint=SimpleNamespace(step="17"))
        self.assertEqual(checkpoints_mod.checkpoint_step_from_save_result(save_result), 17)

    def test_save_checkpoint_step_returns_none_when_missing(self) -> None:
        finetune = SimpleNamespace(save_checkpoint=lambda: SimpleNamespace(checkpoint=SimpleNamespace(step=None)))
        self.assertIsNone(
            checkpoints_mod.save_checkpoint_step(
                finetune=finetune,
                context="unit-test",
            )
        )


class ResolveCheckpointStepTests(unittest.TestCase):
    def test_resolve_checkpoint_step_uses_nearest_saved_fallback(self) -> None:
        with patch.object(checkpoints_mod, "list_saved_checkpoint_steps", return_value=[5, 10, 15]):
            resolved_step, used_fallback = checkpoints_mod.resolve_checkpoint_step(
                api_base="https://example.test",
                api_key="key",
                finetune_id="ft_123",
                requested_step=12,
                fallback_policy="nearest_saved",
            )
        self.assertEqual(resolved_step, 10)
        self.assertTrue(used_fallback)

    def test_resolve_checkpoint_step_exact_waits_for_exact_match(self) -> None:
        with patch.object(checkpoints_mod, "wait_for_exact_checkpoint", return_value=[5, 12, 15]):
            resolved_step, used_fallback = checkpoints_mod.resolve_checkpoint_step(
                api_base="https://example.test",
                api_key="key",
                finetune_id="ft_123",
                requested_step=12,
                fallback_policy="exact",
            )
        self.assertEqual(resolved_step, 12)
        self.assertFalse(used_fallback)

    def test_resolve_checkpoint_step_exact_raises_when_checkpoint_missing(self) -> None:
        with patch.object(checkpoints_mod, "wait_for_exact_checkpoint", return_value=[5, 10, 15]):
            with self.assertRaises(ValueError):
                checkpoints_mod.resolve_checkpoint_step(
                    api_base="https://example.test",
                    api_key="key",
                    finetune_id="ft_123",
                    requested_step=12,
                    fallback_policy="exact",
                )


class AsyncCheckpointEvalTests(unittest.TestCase):
    def test_dispatch_and_drain_checkpoint_eval_job(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            script_path = tmp_path / "worker.py"
            _write_script(
                script_path,
                (
                    "import json, os, sys\n"
                    "metrics_path = sys.argv[1]\n"
                    "records_path = sys.argv[2]\n"
                    "with open(metrics_path, 'w', encoding='utf-8') as handle:\n"
                    "    json.dump({'eval_f1': 0.75, 'env_value': os.environ.get('ASYNC_TEST_ENV', '')}, handle)\n"
                    "with open(records_path, 'w', encoding='utf-8') as handle:\n"
                    "    handle.write(os.environ.get('ASYNC_TEST_ENV', ''))\n"
                    "print('worker complete')\n"
                ),
            )

            job = async_mod.dispatch_checkpoint_eval(
                trainer="unit_test",
                finetune_id="ft_123",
                checkpoint_step=42,
                selection_metric="eval_f1",
                base_dir=str(tmp_path / "jobs"),
                command_builder=lambda metrics_json_path, predictions_jsonl_path, _stdout_log_path: [
                    sys.executable,
                    str(script_path),
                    str(metrics_json_path),
                    str(predictions_jsonl_path),
                ],
                metadata={"step_for_log": 9},
                env_overrides={"ASYNC_TEST_ENV": "propagated"},
                max_inflight=1,
                inflight_jobs=[],
            )

            self.assertIsNotNone(job)
            assert job is not None
            results = async_mod.drain_checkpoint_eval_jobs([job], poll_interval_s=0.05)
            self.assertEqual(len(results), 1)
            result = results[0]
            self.assertEqual(result.status, "succeeded")
            self.assertEqual(result.metrics_payload, {"eval_f1": 0.75, "env_value": "propagated"})
            self.assertEqual(result.metadata["step_for_log"], 9)
            job_payload = json.loads(result.job_json_path.read_text(encoding="utf-8"))
            self.assertEqual(job_payload["status"], "succeeded")
            self.assertEqual(job_payload["checkpoint_step"], 42)
            self.assertEqual(job_payload["selection_metric"], "eval_f1")
            self.assertEqual(job_payload["step_for_log"], 9)

    def test_dispatch_respects_max_inflight(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            script_path = tmp_path / "slow_worker.py"
            _write_script(
                script_path,
                (
                    "import json, sys, time\n"
                    "time.sleep(0.3)\n"
                    "with open(sys.argv[1], 'w', encoding='utf-8') as handle:\n"
                    "    json.dump({'eval_f1': 0.5}, handle)\n"
                ),
            )

            first_job = async_mod.dispatch_checkpoint_eval(
                trainer="unit_test",
                finetune_id="ft_123",
                checkpoint_step=1,
                selection_metric="eval_f1",
                base_dir=str(tmp_path / "jobs"),
                command_builder=lambda metrics_json_path, predictions_jsonl_path, _stdout_log_path: [
                    sys.executable,
                    str(script_path),
                    str(metrics_json_path),
                    str(predictions_jsonl_path),
                ],
                max_inflight=1,
                inflight_jobs=[],
            )
            self.assertIsNotNone(first_job)
            assert first_job is not None

            blocked_job = async_mod.dispatch_checkpoint_eval(
                trainer="unit_test",
                finetune_id="ft_123",
                checkpoint_step=2,
                selection_metric="eval_f1",
                base_dir=str(tmp_path / "jobs"),
                command_builder=lambda metrics_json_path, predictions_jsonl_path, _stdout_log_path: [
                    sys.executable,
                    str(script_path),
                    str(metrics_json_path),
                    str(predictions_jsonl_path),
                ],
                max_inflight=1,
                inflight_jobs=[first_job],
            )
            self.assertIsNone(blocked_job)
            async_mod.drain_checkpoint_eval_jobs([first_job], poll_interval_s=0.05)

    def test_missing_metrics_marks_job_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            script_path = tmp_path / "no_metrics_worker.py"
            _write_script(script_path, "print('done without metrics')\n")

            job = async_mod.dispatch_checkpoint_eval(
                trainer="unit_test",
                finetune_id="ft_123",
                checkpoint_step=99,
                selection_metric="eval_f1",
                base_dir=str(tmp_path / "jobs"),
                command_builder=lambda metrics_json_path, predictions_jsonl_path, _stdout_log_path: [
                    sys.executable,
                    str(script_path),
                    str(metrics_json_path),
                    str(predictions_jsonl_path),
                ],
                max_inflight=1,
                inflight_jobs=[],
            )
            self.assertIsNotNone(job)
            assert job is not None
            results = async_mod.drain_checkpoint_eval_jobs([job], poll_interval_s=0.05)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].status, "failed")
            self.assertIsNone(results[0].metrics_payload)


if __name__ == "__main__":
    unittest.main()
