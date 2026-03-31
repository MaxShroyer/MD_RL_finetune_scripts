from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from datasets import Dataset, DatasetDict, load_from_disk
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import curate_bone_aerial_advertising_subsets as mod


def _write_dataset(path: Path, sample_ids: list[str]) -> Path:
    rows = [
        {
            "image": Image.new("RGB", (16, 16), color=(255, 255, 255)),
            "answer_boxes": "[]",
            "source_image_id": sample_id,
        }
        for sample_id in sample_ids
    ]
    dataset = Dataset.from_list(rows)
    DatasetDict({"test": dataset}).save_to_disk(str(path))
    return path


class AggregationTests(unittest.TestCase):
    def test_aggregate_sample_deltas_averages_task_scores_per_sample(self) -> None:
        baseline = [
            {
                "sample_id": "sample-1",
                "class_name": "plane",
                "prompt": "plane",
                "task_f1": 0.2,
                "task_miou": 0.0,
                "failed": False,
            },
            {
                "sample_id": "sample-1",
                "class_name": "bird",
                "prompt": "bird",
                "task_f1": 0.4,
                "task_miou": 0.0,
                "failed": False,
            },
        ]
        candidate = [
            {
                "sample_id": "sample-1",
                "class_name": "plane",
                "prompt": "plane",
                "task_f1": 0.8,
                "task_miou": 0.0,
                "failed": False,
            },
            {
                "sample_id": "sample-1",
                "class_name": "bird",
                "prompt": "bird",
                "task_f1": 0.6,
                "task_miou": 0.0,
                "failed": False,
            },
        ]

        summaries = mod.aggregate_sample_deltas(baseline_records=baseline, candidate_records=candidate)

        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0]["sample_id"], "sample-1")
        self.assertEqual(summaries[0]["matched_tasks"], 2)
        self.assertAlmostEqual(summaries[0]["delta_f1"], 0.4)
        self.assertAlmostEqual(summaries[0]["delta_miou"], 0.0)
        self.assertAlmostEqual(summaries[0]["delta_score"], 0.4)

    def test_save_subset_dataset_preserves_test_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = _write_dataset(Path(tmp) / "source", ["keep-me", "drop-me"])
            subset_path = Path(tmp) / "subset"
            spec = mod.BranchSpec(
                branch_id="demo",
                label="Demo",
                benchmark_config=Path(tmp) / "unused.json",
                dataset_name="",
                local_dataset_path=dataset_path,
                split="test",
                run_matcher=lambda run: True,
            )

            result = mod.save_subset_dataset(
                spec=spec,
                selected_samples=[{"sample_id": "keep-me", "delta_score": 1.0}],
                output_dir=subset_path,
            )

            loaded = load_from_disk(str(subset_path))

        self.assertEqual(result["selected_count"], 1)
        self.assertEqual(list(loaded.keys()), ["test"])
        self.assertEqual(loaded["test"][0]["source_image_id"], "keep-me")

    def test_select_marketing_samples_backfills_to_target_count(self) -> None:
        runs = [
            {
                "run": mod.LocalRun("run-a", "cfg/a.json", "FT-A", "demo", 10, 0.9, 0.2),
                "sample_summaries": [
                    {"sample_id": "s1", "matched_tasks": 1, "delta_f1": 0.8, "delta_miou": 0.0, "delta_score": 0.8},
                    {"sample_id": "s2", "matched_tasks": 1, "delta_f1": 0.0, "delta_miou": 0.0, "delta_score": 0.0},
                    {"sample_id": "s3", "matched_tasks": 1, "delta_f1": -0.2, "delta_miou": 0.0, "delta_score": -0.2},
                ],
            }
        ]

        selected = mod.select_marketing_samples(runs, target_count=3)

        self.assertEqual([row["sample_id"] for row in selected], ["s1", "s2", "s3"])
        self.assertEqual(selected[0]["selection_reason"], "positive_delta")
        self.assertEqual(selected[1]["selection_reason"], "ranked_backfill")
        self.assertEqual(selected[2]["selection_reason"], "ranked_backfill")


class ProcessBranchSmokeTests(unittest.TestCase):
    def _records_for(self, sample_ids: list[str], values: dict[str, tuple[float, float]], skill: str) -> list[dict[str, object]]:
        class_name = "plane" if skill == "point" else "fracture"
        prompt = class_name
        return [
            {
                "label": "candidate",
                "skill": skill,
                "model": "dummy",
                "run_id": None,
                "finetune_id": None,
                "checkpoint_step": None,
                "dataset_name": None,
                "dataset_path": None,
                "split": "test",
                "sample_id": sample_id,
                "task_key": f"{sample_id}::{class_name}::{prompt}",
                "class_name": class_name,
                "prompt": prompt,
                "is_positive": True,
                "gt_count": 1,
                "pred_count": 1,
                "tp": 1,
                "fp": 0,
                "fn": 0,
                "task_f1": values[sample_id][0],
                "task_miou": values[sample_id][1],
                "latency_sec": 0.01,
                "failed": False,
                "error": None,
            }
            for sample_id in sample_ids
        ]

    def test_process_branch_smoke_for_point_and_detect(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            point_dataset = _write_dataset(tmp_path / "point_source", ["p1", "p2"])
            detect_dataset = _write_dataset(tmp_path / "detect_source", ["d1", "d2"])
            point_spec = mod.BranchSpec(
                branch_id="demo_point",
                label="Demo Point",
                benchmark_config=tmp_path / "unused_point.json",
                dataset_name="",
                local_dataset_path=point_dataset,
                split="test",
                run_matcher=lambda run: run.config_path.startswith("point"),
            )
            detect_spec = mod.BranchSpec(
                branch_id="demo_detect",
                label="Demo Detect",
                benchmark_config=tmp_path / "unused_detect.json",
                dataset_name="",
                local_dataset_path=detect_dataset,
                split="test",
                run_matcher=lambda run: run.config_path.startswith("detect"),
            )
            runs = [
                mod.LocalRun("point-a", "point/a.json", "FT-POINT-A", "point", 11, 0.8, None),
                mod.LocalRun("point-b", "point/b.json", "FT-POINT-B", "point", 12, 0.9, None),
                mod.LocalRun("detect-a", "detect/a.json", "FT-DETECT-A", "detect", 21, 0.3, 0.6),
                mod.LocalRun("detect-b", "detect/b.json", "FT-DETECT-B", "detect", 22, 0.4, 0.7),
            ]

            point_values = {
                "": {"p1": (0.1, 0.0), "p2": (0.1, 0.0)},
                "point-a": {"p1": (0.9, 0.0), "p2": (0.1, 0.0)},
                "point-b": {"p1": (0.2, 0.0), "p2": (0.8, 0.0)},
            }
            detect_values = {
                "": {"d1": (0.1, 0.1), "d2": (0.1, 0.1)},
                "detect-a": {"d1": (0.7, 0.8), "d2": (0.1, 0.1)},
                "detect-b": {"d1": (0.2, 0.2), "d2": (0.9, 0.9)},
            }

            def fake_run_benchmark(
                *,
                spec: mod.BranchSpec,
                out_json: Path,
                records_jsonl: Path,
                dataset_name: str,
                dataset_path: str,
                run_id: str,
                finetune_id: str,
                checkpoint_step: int | None,
                skip_baseline: bool,
                max_samples: int | None,
                viz_samples: int,
                split: str | None,
            ) -> dict[str, object]:
                source_path = Path(dataset_path) if dataset_path else spec.local_dataset_path
                loaded = load_from_disk(str(source_path))
                sample_ids = [str(row["source_image_id"]) for row in loaded["test"]]
                branch_values = point_values if spec.branch_id == "demo_point" else detect_values
                skill = "point" if spec.branch_id == "demo_point" else "detect"
                metrics_by_sample = branch_values[run_id]
                task_records = self._records_for(sample_ids, metrics_by_sample, skill=skill)
                metrics = {
                    "label": "candidate" if run_id else "baseline",
                    "skill": skill,
                    "split": "test",
                    "eval_f1": sum(metrics_by_sample[sample_id][0] for sample_id in sample_ids) / len(sample_ids),
                    "eval_miou": sum(metrics_by_sample[sample_id][1] for sample_id in sample_ids) / len(sample_ids),
                }
                out_json.parent.mkdir(parents=True, exist_ok=True)
                out_json.write_text(json.dumps(metrics), encoding="utf-8")
                records_jsonl.parent.mkdir(parents=True, exist_ok=True)
                records_jsonl.write_text(
                    "".join(json.dumps(row) + "\n" for row in task_records),
                    encoding="utf-8",
                )
                return metrics

            with patch.object(mod, "_run_benchmark", side_effect=fake_run_benchmark):
                point_summary = mod.process_branch(
                    spec=point_spec,
                    runs=runs,
                    output_root=tmp_path / "out",
                    max_runs=None,
                    max_samples=None,
                )
                detect_summary = mod.process_branch(
                    spec=detect_spec,
                    runs=runs,
                    output_root=tmp_path / "out",
                    max_runs=None,
                    max_samples=None,
                )

            self.assertEqual(point_summary["selected_sample_count"], 2)
            self.assertEqual(point_summary["display_winner"]["run_id"], "point-b")
            self.assertAlmostEqual(point_summary["subset_winner"]["eval_f1"], 0.5)
            self.assertEqual(detect_summary["selected_sample_count"], 2)
            self.assertEqual(detect_summary["display_winner"]["run_id"], "detect-b")
            self.assertAlmostEqual(detect_summary["subset_winner"]["eval_miou"], 0.55)
            self.assertTrue(Path(point_summary["leaderboard_path"]).exists())
            self.assertTrue(Path(detect_summary["leaderboard_path"]).exists())

    def test_benchmark_args_include_visualization_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            spec = mod.BranchSpec(
                branch_id="demo",
                label="Demo",
                benchmark_config=tmp_path / "benchmark.json",
                dataset_name="demo-dataset",
                local_dataset_path=tmp_path / "dataset",
                split="test",
                run_matcher=lambda run: True,
            )

            argv = mod._benchmark_args(
                spec=spec,
                out_json=tmp_path / "branch" / "candidate.metrics.json",
                records_jsonl=tmp_path / "branch" / "candidate.records.jsonl",
                dataset_name="demo-dataset",
                dataset_path="",
                run_id="run-1",
                finetune_id="FT-1",
                checkpoint_step=12,
                skip_baseline=True,
                max_samples=25,
                viz_samples=40,
            )

        self.assertIn("--viz-samples", argv)
        self.assertIn("40", argv)
        self.assertIn("--viz-dir", argv)
        self.assertIn(str((tmp_path / "branch" / "viz").resolve()), argv)


if __name__ == "__main__":
    unittest.main()
