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

import curate_aerial_manual_top100 as mod
import curate_bone_aerial_advertising_subsets as shared


def _write_dataset_dict(path: Path, split_to_ids: dict[str, list[str]]) -> Path:
    dataset_dict = DatasetDict(
        {
            split_name: Dataset.from_list(
                [
                    {
                        "image": Image.new("RGB", (16, 16), color=(255, 255, 255)),
                        "answer_boxes": "[]",
                        "source_image_id": sample_id,
                    }
                    for sample_id in sample_ids
                ]
            )
            for split_name, sample_ids in split_to_ids.items()
        }
    )
    dataset_dict.save_to_disk(str(path))
    return path


def _records_for(sample_ids: list[str], metrics_by_sample: dict[str, tuple[float, float]], *, skill: str) -> list[dict[str, object]]:
    class_name = "airplane"
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
            "task_f1": metrics_by_sample[sample_id][0],
            "task_miou": metrics_by_sample[sample_id][1],
            "latency_sec": 0.01,
            "failed": False,
            "error": None,
        }
        for sample_id in sample_ids
    ]


class CombineToolSummariesTests(unittest.TestCase):
    def test_combine_tool_sample_summaries_averages_across_tools(self) -> None:
        combined = mod.combine_tool_sample_summaries(
            {
                "point": [
                    {"sample_id": "s1", "matched_tasks": 1, "delta_f1": 0.8, "delta_miou": 0.0, "delta_score": 0.8},
                    {"sample_id": "s2", "matched_tasks": 1, "delta_f1": 0.2, "delta_miou": 0.0, "delta_score": 0.2},
                ],
                "detect": [
                    {"sample_id": "s1", "matched_tasks": 1, "delta_f1": 0.4, "delta_miou": 0.6, "delta_score": 1.0},
                    {"sample_id": "s2", "matched_tasks": 1, "delta_f1": -0.1, "delta_miou": 0.0, "delta_score": -0.1},
                ],
            }
        )

        self.assertEqual([row["sample_id"] for row in combined], ["s1", "s2"])
        self.assertAlmostEqual(combined[0]["combined_delta_score"], 0.9)
        self.assertEqual(combined[0]["winning_tool_id"], "detect")
        self.assertAlmostEqual(combined[1]["combined_delta_score"], 0.05)

    def test_tool_benchmark_args_force_staging_api(self) -> None:
        branch_spec = shared._branch_specs()["aerial_point"]
        tool = mod.ManualToolSpec(
            tool_id="point",
            label="Aerial Point",
            branch_spec=branch_spec,
            benchmark_main=lambda argv=None: None,
            finetune_id="FT-POINT",
            checkpoint_step=12,
        )
        with tempfile.TemporaryDirectory() as tmp:
            argv = mod._tool_benchmark_args(
                tool=tool,
                out_json=Path(tmp) / "metrics.json",
                records_jsonl=Path(tmp) / "records.jsonl",
                dataset_name="maxs-m87/aerial_airport_point_v2",
                dataset_path="",
                split="test",
                finetune_id=tool.finetune_id,
                checkpoint_step=tool.checkpoint_step,
                skip_baseline=True,
                max_samples=25,
                viz_samples=40,
            )

        self.assertIn("--api-base", argv)
        self.assertIn(mod.DEFAULT_STAGING_API_BASE, argv)
        self.assertIn("--viz-dir", argv)
        self.assertIn("--skip-baseline", argv)

    def test_select_top_samples_excludes_tool_regressions_by_default(self) -> None:
        rows = [
            {
                "sample_id": "keep-a",
                "combined_delta_score": 0.7,
                "point_delta_score": 0.3,
                "detect_delta_score": 1.1,
            },
            {
                "sample_id": "drop-regression",
                "combined_delta_score": 0.9,
                "point_delta_score": 1.0,
                "detect_delta_score": -0.1,
            },
            {
                "sample_id": "keep-b",
                "combined_delta_score": 0.5,
                "point_delta_score": 0.2,
                "detect_delta_score": 0.8,
            },
        ]

        selected = mod.select_top_samples(rows, target_count=2)
        selected_with_regressions = mod.select_top_samples(rows, target_count=2, allow_tool_regressions=True)

        self.assertEqual([row["sample_id"] for row in selected], ["keep-a", "keep-b"])
        self.assertEqual([row["sample_id"] for row in selected_with_regressions], ["drop-regression", "keep-a"])


class ProcessManualAerialTop100Tests(unittest.TestCase):
    def test_process_manual_aerial_top100_builds_shared_subset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_path = _write_dataset_dict(
                tmp_path / "aerial_dataset",
                {
                    "train": ["t1", "t2"],
                    "validation": ["v1", "v2"],
                    "test": ["x1", "x2"],
                },
            )
            point_spec = shared.BranchSpec(
                branch_id="aerial_point",
                label="Aerial Point",
                benchmark_config=tmp_path / "point.json",
                dataset_name="",
                local_dataset_path=dataset_path,
                split="test",
                run_matcher=lambda run: True,
            )
            detect_spec = shared.BranchSpec(
                branch_id="aerial_detect",
                label="Aerial Detect",
                benchmark_config=tmp_path / "detect.json",
                dataset_name="",
                local_dataset_path=dataset_path,
                split="test",
                run_matcher=lambda run: True,
            )
            tools = [
                mod.ManualToolSpec(
                    tool_id="point",
                    label="Aerial Point",
                    branch_spec=point_spec,
                    benchmark_main=lambda argv=None: None,
                    finetune_id="POINT-FT",
                    checkpoint_step=139,
                ),
                mod.ManualToolSpec(
                    tool_id="detect",
                    label="Aerial Detect",
                    branch_spec=detect_spec,
                    benchmark_main=lambda argv=None: None,
                    finetune_id="DETECT-FT",
                    checkpoint_step=250,
                ),
            ]

            point_values = {
                "": {"t1": (0.1, 0.0), "t2": (0.1, 0.0), "v1": (0.1, 0.0), "v2": (0.1, 0.0), "x1": (0.1, 0.0), "x2": (0.1, 0.0)},
                "POINT-FT": {"t1": (0.9, 0.0), "t2": (0.5, 0.0), "v1": (0.4, 0.0), "v2": (0.1, 0.0), "x1": (0.2, 0.0), "x2": (0.0, 0.0)},
            }
            detect_values = {
                "": {"t1": (0.1, 0.1), "t2": (0.1, 0.1), "v1": (0.1, 0.1), "v2": (0.1, 0.1), "x1": (0.1, 0.1), "x2": (0.1, 0.1)},
                "DETECT-FT": {"t1": (0.8, 0.8), "t2": (0.6, 0.6), "v1": (0.3, 0.4), "v2": (0.1, 0.1), "x1": (0.0, 0.1), "x2": (0.2, 0.2)},
            }

            def fake_run_tool_benchmark(
                *,
                tool: mod.ManualToolSpec,
                out_json: Path,
                records_jsonl: Path,
                dataset_name: str,
                dataset_path: str,
                split: str,
                finetune_id: str,
                checkpoint_step: int | None,
                skip_baseline: bool,
                max_samples: int | None,
                viz_samples: int,
            ) -> dict[str, object]:
                source_path = Path(dataset_path) if dataset_path else tool.branch_spec.local_dataset_path
                loaded = load_from_disk(str(source_path))
                sample_ids = [str(row["source_image_id"]) for row in loaded["test"]]
                metrics_by_sample = point_values[finetune_id] if tool.tool_id == "point" else detect_values[finetune_id]
                skill = "point" if tool.tool_id == "point" else "detect"
                task_records = _records_for(sample_ids, metrics_by_sample, skill=skill)
                metrics = {
                    "label": "candidate" if finetune_id else "baseline",
                    "skill": skill,
                    "split": split,
                    "base_samples": len(sample_ids),
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

            with patch.object(mod, "_run_tool_benchmark", side_effect=fake_run_tool_benchmark):
                summary = mod.process_manual_aerial_top100(
                    tools=tools,
                    output_root=tmp_path / "out",
                    selection_split="train+validation+test",
                    target_selected_samples=4,
                    max_samples=None,
                    viz_samples=0,
                )

            loaded_subset = load_from_disk(str(tmp_path / "out" / "subset" / "dataset"))
            selected_ids = [str(row["sample_id"]) for row in shared._load_jsonl(tmp_path / "out" / "selected_samples.jsonl")]
            report_exists = (tmp_path / "out" / "report.json").exists()

        self.assertEqual(summary["selected_sample_count"], 4)
        self.assertEqual(selected_ids, ["t1", "t2", "v1", "v2"])
        self.assertEqual(len(loaded_subset["test"]), 4)
        self.assertEqual(summary["tools"]["point"]["subset_candidate"]["skill"], "point")
        self.assertEqual(summary["tools"]["detect"]["subset_candidate"]["skill"], "detect")
        self.assertEqual(summary["staging_api_base"], mod.DEFAULT_STAGING_API_BASE)
        self.assertTrue(report_exists)


if __name__ == "__main__":
    unittest.main()
