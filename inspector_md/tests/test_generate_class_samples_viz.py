from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inspector_md import generate_class_samples_viz as viz_mod


def _write_image(path: Path, color: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (96, 72), color).save(path)
    return path


class ClassSamplesVizTests(unittest.TestCase):
    def test_generate_visualizations_writes_contact_sheets_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            image_a = _write_image(tmp / "a.png", "white")
            image_b = _write_image(tmp / "b.png", "lightgray")
            manifest_path = tmp / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    [
                        {
                            "row_id": "row-a",
                            "split": "train",
                            "image_path": str(image_a),
                            "inspection_request": "Inspect the facade.",
                            "asset_context": "Concrete exterior.",
                            "hard_example": True,
                            "source_dataset": "CODEBRIM",
                            "expected_proposals": [
                                {"issue_code": "surface_spalling", "evidence": "Spalled material is visible."}
                            ],
                            "expected_findings": [
                                {
                                    "finding_id": "row-a_finding_001",
                                    "issue_code": "surface_spalling",
                                    "title": "Surface Spalling",
                                    "box": {"x_min": 0.1, "y_min": 0.15, "x_max": 0.7, "y_max": 0.8},
                                    "evidence": ["Spalled material is visible."],
                                    "severity": "major",
                                    "recommended_action": "Repair the concrete surface.",
                                    "cost_band": "high",
                                    "possible_compliance_issue": True,
                                    "insufficient_evidence": False,
                                    "compliance_note": "Possible compliance issue related to surface_stability.",
                                    "source_detect_labels": ["surface spalling"],
                                    "spatial_ref_index": 0,
                                }
                            ],
                        },
                        {
                            "row_id": "row-b",
                            "split": "test",
                            "image_path": str(image_b),
                            "inspection_request": "Inspect the wall.",
                            "asset_context": "Close-up wall image.",
                            "hard_example": False,
                            "source_dataset": "MBDD2025",
                            "expected_proposals": [
                                {"issue_code": "crack_defect", "evidence": "A surface crack is visible."}
                            ],
                            "expected_findings": [
                                {
                                    "finding_id": "row-b_finding_001",
                                    "issue_code": "crack_defect",
                                    "title": "Crack Defect",
                                    "box": {"x_min": 0.2, "y_min": 0.2, "x_max": 0.8, "y_max": 0.5},
                                    "evidence": ["A surface crack is visible."],
                                    "severity": "moderate",
                                    "recommended_action": "Inspect and repair the crack.",
                                    "cost_band": "medium",
                                    "possible_compliance_issue": True,
                                    "insufficient_evidence": False,
                                    "compliance_note": "Possible compliance issue related to surface_stability.",
                                    "source_detect_labels": ["surface crack"],
                                    "spatial_ref_index": 0,
                                }
                            ],
                        },
                    ]
                ),
                encoding="utf-8",
            )
            args = viz_mod.parse_args(
                [
                    "--source-manifest",
                    str(manifest_path),
                    "--output-dir",
                    str(tmp / "viz"),
                    "--samples-per-class",
                    "2",
                    "--columns",
                    "2",
                ]
            )
            summary = viz_mod.generate_visualizations(args)
            summary_path = Path(summary["output_dir"]) / "summary.json"
            self.assertTrue(summary_path.exists())
            self.assertTrue((Path(summary["output_dir"]) / "overview.png").exists())
            self.assertTrue((Path(summary["output_dir"]) / "surface_spalling" / "contact_sheet.png").exists())
            self.assertTrue((Path(summary["output_dir"]) / "crack_defect" / "contact_sheet.png").exists())
            loaded_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(sorted(loaded_summary["rendered_issue_codes"]), ["crack_defect", "surface_spalling"])


if __name__ == "__main__":
    unittest.main()
