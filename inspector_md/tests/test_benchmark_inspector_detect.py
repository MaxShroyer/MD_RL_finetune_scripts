from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inspector_md import benchmark_inspector_detect as detect_mod
from inspector_md.moondream_client import DetectAnnotation


def _write_image(path: Path, color: str = "white") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (120, 90), color).save(path)
    return path


class DetectBenchmarkTests(unittest.TestCase):
    def test_detect_benchmark_reports_loose_and_center_hit_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            image_path = _write_image(tmp / "sample.png")
            manifest_path = tmp / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    [
                        {
                            "row_id": "row-1",
                            "split": "test",
                            "source_dataset": "MBDD2025",
                            "source_metadata": {},
                            "image_path": str(image_path),
                            "inspection_request": "Inspect the wall for visible crack defects.",
                            "asset_context": "Exterior wall",
                            "hard_example": False,
                            "expected_proposals": [{"issue_code": "crack_defect", "evidence": "visible crack"}],
                            "expected_findings": [
                                {
                                    "finding_id": "gt-1",
                                    "issue_code": "crack_defect",
                                    "title": "Crack Defect",
                                    "box": {"x_min": 0.2, "y_min": 0.2, "x_max": 0.8, "y_max": 0.8},
                                    "evidence": ["visible crack"],
                                    "severity": "moderate",
                                    "recommended_action": "repair crack",
                                    "cost_band": "medium",
                                    "possible_compliance_issue": True,
                                    "insufficient_evidence": False,
                                    "compliance_note": "possible issue",
                                    "source_detect_labels": ["surface crack"],
                                    "spatial_ref_index": 0
                                }
                            ]
                        }
                    ]
                ),
                encoding="utf-8",
            )

            class _FakeClient:
                def __init__(self, **kwargs):
                    pass

                def detect_boxes(self, **kwargs):
                    return [
                        DetectAnnotation(x_min=0.3, y_min=0.3, x_max=0.5, y_max=0.5),
                    ]

            args = detect_mod.parse_args(
                [
                    "--api-key",
                    "test-key",
                    "--dataset-manifest",
                    str(manifest_path),
                    "--output-json",
                    str(tmp / "metrics.json"),
                    "--predictions-jsonl",
                    str(tmp / "records.jsonl"),
                ]
            )
            with patch.object(detect_mod, "MoondreamInspectorClient", _FakeClient):
                summary = detect_mod.run_benchmark(args)
            self.assertEqual(summary["finding_count"], 1)
            self.assertEqual(summary["recall_iou_0_5"], 0.0)
            self.assertEqual(summary["recall_iou_0_1"], 1.0)
            self.assertEqual(summary["center_hit_rate"], 1.0)
            self.assertGreater(summary["mean_best_iou"], 0.1)
            records = [json.loads(line) for line in (tmp / "records.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(records[0]["matched_iou_0_5"], 0.0)
            self.assertEqual(records[0]["matched_iou_0_1"], 1.0)
            self.assertEqual(records[0]["center_hit"], 1.0)


if __name__ == "__main__":
    unittest.main()
