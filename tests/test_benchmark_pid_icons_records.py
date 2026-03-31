from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from datasets import Dataset, DatasetDict
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MDpi_and_d import benchmark_pid_icons as mod


def _write_dataset(root: Path) -> Path:
    rows = [
        {
            "image": Image.new("RGB", (20, 20), color=(255, 255, 255)),
            "answer_boxes": json.dumps(
                [
                    {
                        "x_min": 0.1,
                        "y_min": 0.1,
                        "x_max": 0.4,
                        "y_max": 0.4,
                        "class_uid": "plane",
                        "class_name": "plane",
                    }
                ]
            ),
            "source_image_id": "positive_sample",
        },
        {
            "image": Image.new("RGB", (20, 20), color=(240, 240, 240)),
            "answer_boxes": "[]",
            "source_image_id": "negative_sample",
        },
    ]
    dataset = Dataset.from_list(rows)
    dataset_path = root / "dataset"
    DatasetDict({"test": dataset}).save_to_disk(str(dataset_path))
    return dataset_path


def _write_nonfracture_detect_dataset(root: Path) -> Path:
    rows = [
        {
            "image": Image.new("RGB", (20, 20), color=(220, 220, 220)),
            "answer_boxes": json.dumps(
                [
                    {
                        "x_min": 0.2,
                        "y_min": 0.2,
                        "x_max": 0.6,
                        "y_max": 0.6,
                        "class_uid": "implant",
                        "class_name": "implant",
                    }
                ]
            ),
            "source_image_id": "subset-negative",
        },
    ]
    dataset = Dataset.from_list(rows)
    dataset_path = root / "detect_subset"
    DatasetDict({"test": dataset}).save_to_disk(str(dataset_path))
    return dataset_path


class BenchmarkTaskRecordTests(unittest.TestCase):
    def test_payload_is_unchanged_when_records_jsonl_is_omitted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = _write_dataset(Path(tmp))
            out_json = Path(tmp) / "metrics.json"
            with patch("MDpi_and_d.benchmark_pid_icons._call_point_api", return_value=[]):
                mod.main(
                    [
                        "--dataset-path",
                        str(dataset_path),
                        "--dataset-name",
                        "",
                        "--split",
                        "test",
                        "--skill",
                        "point",
                        "--skip-baseline",
                        "--model",
                        "dummy-model",
                        "--api-key",
                        "test-key",
                        "--neg-prompts-per-empty",
                        "1",
                        "--neg-prompts-per-nonempty",
                        "0",
                        "--progress-every",
                        "0",
                        "--out-json",
                        str(out_json),
                    ]
                )

            payload = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(payload["skill"], "point")
            self.assertEqual(payload["split"], "test")
            self.assertIn("eval_f1", payload)
            self.assertIn("per_class", payload)
            self.assertNotIn("records_jsonl", payload)

    def test_point_mode_writes_task_records_with_negative_rows_and_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = _write_dataset(Path(tmp))
            out_json = Path(tmp) / "metrics.json"
            records_jsonl = Path(tmp) / "records.jsonl"
            with patch(
                "MDpi_and_d.benchmark_pid_icons._call_point_api",
                side_effect=[
                    [mod.Point(0.2, 0.2)],
                    TimeoutError("point boom"),
                ],
            ):
                mod.main(
                    [
                        "--dataset-path",
                        str(dataset_path),
                        "--dataset-name",
                        "",
                        "--split",
                        "test",
                        "--skill",
                        "point",
                        "--point-prompt-style",
                        "class_name",
                        "--skip-baseline",
                        "--model",
                        "moondream3-preview/01POINT@7",
                        "--run-id",
                        "point-run",
                        "--api-key",
                        "test-key",
                        "--neg-prompts-per-empty",
                        "1",
                        "--neg-prompts-per-nonempty",
                        "0",
                        "--progress-every",
                        "0",
                        "--out-json",
                        str(out_json),
                        "--records-jsonl",
                        str(records_jsonl),
                    ]
                )

            rows = [json.loads(line) for line in records_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(rows), 2)
            success = next(row for row in rows if row["sample_id"] == "positive_sample")
            failure = next(row for row in rows if row["sample_id"] == "negative_sample")

            self.assertEqual(success["label"], "candidate")
            self.assertEqual(success["run_id"], "point-run")
            self.assertEqual(success["finetune_id"], "01POINT")
            self.assertEqual(success["checkpoint_step"], 7)
            self.assertTrue(success["is_positive"])
            self.assertEqual(success["gt_count"], 1)
            self.assertEqual(success["pred_count"], 1)
            self.assertEqual(success["tp"], 1)
            self.assertEqual(success["fp"], 0)
            self.assertEqual(success["fn"], 0)
            self.assertAlmostEqual(success["task_f1"], 1.0)
            self.assertAlmostEqual(success["task_miou"], 0.0)
            self.assertFalse(success["failed"])

            self.assertFalse(failure["is_positive"])
            self.assertTrue(failure["failed"])
            self.assertIsNone(failure["pred_count"])
            self.assertIn("point boom", failure["error"])

    def test_detect_mode_writes_task_records_with_negative_rows_and_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = _write_dataset(Path(tmp))
            out_json = Path(tmp) / "metrics.json"
            records_jsonl = Path(tmp) / "records.jsonl"
            with patch(
                "MDpi_and_d.benchmark_pid_icons._call_detect_api",
                side_effect=[
                    [mod.Box(0.1, 0.1, 0.4, 0.4)],
                    TimeoutError("detect boom"),
                ],
            ):
                mod.main(
                    [
                        "--dataset-path",
                        str(dataset_path),
                        "--dataset-name",
                        "",
                        "--split",
                        "test",
                        "--skill",
                        "detect",
                        "--skip-baseline",
                        "--model",
                        "moondream3-preview/01DETECT@9",
                        "--run-id",
                        "detect-run",
                        "--api-key",
                        "test-key",
                        "--neg-prompts-per-empty",
                        "1",
                        "--neg-prompts-per-nonempty",
                        "0",
                        "--progress-every",
                        "0",
                        "--out-json",
                        str(out_json),
                        "--records-jsonl",
                        str(records_jsonl),
                    ]
                )

            rows = [json.loads(line) for line in records_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(rows), 2)
            success = next(row for row in rows if row["sample_id"] == "positive_sample")
            failure = next(row for row in rows if row["sample_id"] == "negative_sample")

            self.assertEqual(success["label"], "candidate")
            self.assertEqual(success["run_id"], "detect-run")
            self.assertEqual(success["finetune_id"], "01DETECT")
            self.assertEqual(success["checkpoint_step"], 9)
            self.assertTrue(success["is_positive"])
            self.assertEqual(success["pred_count"], 1)
            self.assertEqual(success["tp"], 1)
            self.assertEqual(success["fp"], 0)
            self.assertEqual(success["fn"], 0)
            self.assertAlmostEqual(success["task_f1"], 1.0)
            self.assertAlmostEqual(success["task_miou"], 1.0)
            self.assertFalse(success["failed"])

            self.assertFalse(failure["is_positive"])
            self.assertTrue(failure["failed"])
            self.assertIsNone(failure["pred_count"])
            self.assertIn("detect boom", failure["error"])

    def test_detect_mode_uses_include_classes_fallback_for_negative_only_subset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = _write_nonfracture_detect_dataset(Path(tmp))
            out_json = Path(tmp) / "metrics.json"
            records_jsonl = Path(tmp) / "records.jsonl"
            with patch("MDpi_and_d.benchmark_pid_icons._call_detect_api", return_value=[]):
                mod.main(
                    [
                        "--dataset-path",
                        str(dataset_path),
                        "--dataset-name",
                        "",
                        "--split",
                        "test",
                        "--skill",
                        "detect",
                        "--include-classes",
                        "fracture",
                        "--skip-baseline",
                        "--model",
                        "dummy-model",
                        "--api-key",
                        "test-key",
                        "--neg-prompts-per-empty",
                        "1",
                        "--neg-prompts-per-nonempty",
                        "1",
                        "--progress-every",
                        "0",
                        "--out-json",
                        str(out_json),
                        "--records-jsonl",
                        str(records_jsonl),
                    ]
                )

            payload = json.loads(out_json.read_text(encoding="utf-8"))
            rows = [json.loads(line) for line in records_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]

            self.assertEqual(payload["skill"], "detect")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["sample_id"], "subset-negative")
            self.assertEqual(rows[0]["class_name"], "fracture")
            self.assertFalse(rows[0]["is_positive"])
            self.assertFalse(rows[0]["failed"])
            self.assertEqual(rows[0]["gt_count"], 0)
            self.assertEqual(rows[0]["pred_count"], 0)


if __name__ == "__main__":
    unittest.main()
