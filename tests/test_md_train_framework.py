from __future__ import annotations

import io
import json
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from md_train_framework import cli
from md_train_framework.config import load_framework_config, save_framework_config
from md_train_framework.datasets import inspect_dataset
from md_train_framework.registry import RunRegistry


class MdTrainFrameworkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmpdir.name)
        self.base_config_path = REPO_ROOT / "md_train_framework" / "configs" / "detect_quickstart.json"
        self.config = load_framework_config(self.base_config_path)
        self.config_path = self.tmp_path / "detect_quickstart.json"
        payload = self.config.to_dict(include_meta=False)
        payload["dataset"]["path"] = str(self.config.resolved_path(self.config.dataset.path))
        payload["dataset"]["image_root"] = str(self.config.resolved_path(self.config.dataset.image_root))
        payload["dataset"]["split_files"] = {
            key: str(self.config.resolved_path(value))
            for key, value in payload["dataset"]["split_files"].items()
        }
        payload["logging"]["run_root"] = str(self.tmp_path / "runs")
        payload["logging"]["registry_path"] = str(self.tmp_path / "runs.db")
        payload["eval"]["async_checkpoint_eval_dir"] = str(self.tmp_path / "async")
        payload["recovery"]["quarantine_path"] = str(self.tmp_path / "recovery" / "quarantine.jsonl")
        payload["recovery"]["resume_queue_path"] = str(self.tmp_path / "recovery" / "resume_queue.jsonl")
        save_framework_config(self.config_path, payload)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_dataset_inspection_reads_toy_dataset(self) -> None:
        inspection = inspect_dataset(load_framework_config(self.config_path))
        split_counts = {split.name: split.count for split in inspection.splits}
        self.assertEqual(split_counts["train"], 2)
        self.assertEqual(split_counts["validation"], 1)
        self.assertEqual(split_counts["test"], 1)

    def test_quickstart_commands_run_end_to_end(self) -> None:
        outputs: list[str] = []
        for argv in (
            ["inspect-dataset", "--config", str(self.config_path)],
            ["dry-run", "--config", str(self.config_path)],
            ["baseline", "--config", str(self.config_path)],
            ["train", "--config", str(self.config_path)],
            ["leaderboard", "--config", str(self.config_path)],
        ):
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = cli.main(argv)
            self.assertEqual(code, 0)
            outputs.append(stdout.getvalue())
        self.assertIn("selection_metric_value", outputs[2])
        self.assertIn("run_id", outputs[3])
        self.assertIn('"rows"', outputs[4])

    def test_compare_and_sweep_plan(self) -> None:
        with redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(["baseline", "--config", str(self.config_path)]), 0)
        train_stdout = io.StringIO()
        with redirect_stdout(train_stdout):
            self.assertEqual(cli.main(["train", "--config", str(self.config_path)]), 0)
        registry = RunRegistry(self.tmp_path / "runs.db")
        records = registry.list_records()
        baseline_id = next(record.record_id for record in records if record.record_type == "baseline")
        run_id = next(record.run_id for record in records if record.record_type == "run")
        compare_stdout = io.StringIO()
        with redirect_stdout(compare_stdout):
            self.assertEqual(
                cli.main(
                    [
                        "compare",
                        "--registry-path",
                        str(self.tmp_path / "runs.db"),
                        "--left",
                        run_id,
                        "--right",
                        baseline_id,
                    ]
                ),
                0,
            )
        self.assertIn("selection_metric_delta", compare_stdout.getvalue())
        sweep_stdout = io.StringIO()
        with redirect_stdout(sweep_stdout):
            self.assertEqual(
                cli.main(
                    [
                        "sweep",
                        "--config",
                        str(self.config_path),
                        "--max-candidates",
                        "4",
                        "--plan-only",
                    ]
                ),
                0,
            )
        self.assertIn("generated_candidates", sweep_stdout.getvalue())

    def test_import_legacy_runs_scans_temp_run(self) -> None:
        legacy_root = self.tmp_path / "legacy"
        run_dir = legacy_root / "demo_task" / "run_001"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run_config.json").write_text(
            json.dumps(
                {
                    "dataset_path": "demo",
                    "selection_metric": "eval_f1",
                    "skill": "detect",
                    "task_name": "legacy_detect",
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "train_summary.json").write_text(
            json.dumps(
                {
                    "finetune_id": "ft_legacy",
                    "best_selection_metric": 0.77,
                    "best_selection_metric_name": "eval_f1",
                    "final_eval": {"validation": {"eval_f1": 0.77, "eval_miou": 0.70}},
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "eval_history.jsonl").write_text(
            '{"step": 10, "stage": "rl", "split": "validation", "metrics": {"eval_f1": 0.70, "eval_miou": 0.65}}\n',
            encoding="utf-8",
        )
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(
                cli.main(
                    [
                        "import-legacy-runs",
                        "--registry-path",
                        str(self.tmp_path / "runs.db"),
                        "--root",
                        str(legacy_root),
                    ]
                ),
                0,
            )
        registry = RunRegistry(self.tmp_path / "runs.db")
        imported = registry.list_records(task="legacy_detect")
        self.assertTrue(imported)

    def test_readme_commands_exist(self) -> None:
        readme = (REPO_ROOT / "md_train_framework" / "README.md").read_text(encoding="utf-8")
        self.assertIn("python -m md_train_framework inspect-dataset", readme)
        self.assertIn("python -m md_train_framework baseline", readme)
        self.assertIn("python -m md_train_framework train", readme)

    def test_extract_step_ignores_timestamp_digits(self) -> None:
        self.assertEqual(cli._extract_step("step000030_2026-04-16T06-24-40-00-00"), 30)
        self.assertEqual(cli._extract_step("checkpoint-step000139_2026-04-01"), 139)
        self.assertIsNone(cli._extract_step("checkpoint_2026-04-16T06-24-40-00-00"))


if __name__ == "__main__":
    unittest.main()
