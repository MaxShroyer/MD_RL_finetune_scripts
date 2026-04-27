from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inspector_md import build_balanced_subset_manifest as subset_mod


class BalancedSubsetManifestTests(unittest.TestCase):
    def test_build_subset_balances_primary_buckets(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            manifest_path = tmp / "synthetic_manifest.json"
            rows = [
                {"row_id": "neg-1", "split": "train", "source_dataset": "CODEBRIM", "hard_example": False, "expected_proposals": []},
                {"row_id": "neg-2", "split": "validation", "source_dataset": "CODEBRIM", "hard_example": False, "expected_proposals": []},
                {"row_id": "neg-3", "split": "test", "source_dataset": "CODEBRIM", "hard_example": False, "expected_proposals": []},
                {"row_id": "crack-1", "split": "train", "source_dataset": "MBDD2025", "hard_example": False, "expected_proposals": [{"issue_code": "crack_defect"}]},
                {"row_id": "crack-2", "split": "validation", "source_dataset": "MBDD2025", "hard_example": False, "expected_proposals": [{"issue_code": "crack_defect"}]},
                {"row_id": "crack-3", "split": "test", "source_dataset": "MBDD2025", "hard_example": False, "expected_proposals": [{"issue_code": "crack_defect"}]},
                {"row_id": "corrosion-1", "split": "train", "source_dataset": "MBDD2025", "hard_example": True, "expected_proposals": [{"issue_code": "corrosion_rust"}]},
                {"row_id": "corrosion-2", "split": "validation", "source_dataset": "MBDD2025", "hard_example": True, "expected_proposals": [{"issue_code": "corrosion_rust"}]},
                {"row_id": "corrosion-3", "split": "test", "source_dataset": "MBDD2025", "hard_example": True, "expected_proposals": [{"issue_code": "corrosion_rust"}]},
                {
                    "row_id": "multi-1",
                    "split": "train",
                    "source_dataset": "MBDD2025",
                    "hard_example": True,
                    "expected_proposals": [{"issue_code": "crack_defect"}, {"issue_code": "corrosion_rust"}],
                },
                {
                    "row_id": "multi-2",
                    "split": "validation",
                    "source_dataset": "MBDD2025",
                    "hard_example": True,
                    "expected_proposals": [{"issue_code": "crack_defect"}, {"issue_code": "corrosion_rust"}],
                },
                {
                    "row_id": "multi-3",
                    "split": "test",
                    "source_dataset": "MBDD2025",
                    "hard_example": True,
                    "expected_proposals": [{"issue_code": "crack_defect"}, {"issue_code": "corrosion_rust"}],
                },
            ]
            manifest_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
            output_dir = tmp / "subset_out"

            args = subset_mod.parse_args(
                [
                    "--source-manifest",
                    str(manifest_path),
                    "--output-dir",
                    str(output_dir),
                    "--max-samples",
                    "6",
                    "--seed",
                    "7",
                ]
            )
            summary = subset_mod.build_subset(args)

            subset_rows = json.loads((output_dir / "synthetic_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(len(subset_rows), 6)
            self.assertEqual(summary["selected_bucket_counts"]["__negative__"], 2)
            self.assertEqual(summary["selected_bucket_counts"]["corrosion_rust"], 2)
            self.assertEqual(summary["selected_bucket_counts"]["crack_defect"], 2)
            self.assertTrue(all("subset_bucket" in row for row in subset_rows))
            self.assertEqual(
                {row["subset_bucket"] for row in subset_rows},
                {"__negative__", "corrosion_rust", "crack_defect"},
            )

    def test_build_subset_redistributes_when_bucket_is_small(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            manifest_path = tmp / "synthetic_manifest.json"
            rows = [
                {"row_id": "neg-1", "split": "train", "source_dataset": "CODEBRIM", "hard_example": False, "expected_proposals": []},
                {"row_id": "crack-1", "split": "train", "source_dataset": "MBDD2025", "hard_example": False, "expected_proposals": [{"issue_code": "crack_defect"}]},
                {"row_id": "crack-2", "split": "validation", "source_dataset": "MBDD2025", "hard_example": False, "expected_proposals": [{"issue_code": "crack_defect"}]},
                {"row_id": "crack-3", "split": "test", "source_dataset": "MBDD2025", "hard_example": False, "expected_proposals": [{"issue_code": "crack_defect"}]},
                {"row_id": "corrosion-1", "split": "train", "source_dataset": "MBDD2025", "hard_example": True, "expected_proposals": [{"issue_code": "corrosion_rust"}]},
                {"row_id": "corrosion-2", "split": "validation", "source_dataset": "MBDD2025", "hard_example": True, "expected_proposals": [{"issue_code": "corrosion_rust"}]},
                {"row_id": "corrosion-3", "split": "test", "source_dataset": "MBDD2025", "hard_example": True, "expected_proposals": [{"issue_code": "corrosion_rust"}]},
            ]
            manifest_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
            output_dir = tmp / "subset_out"

            args = subset_mod.parse_args(
                [
                    "--source-manifest",
                    str(manifest_path),
                    "--output-dir",
                    str(output_dir),
                    "--max-samples",
                    "6",
                    "--seed",
                    "11",
                ]
            )
            summary = subset_mod.build_subset(args)

            self.assertEqual(summary["selected_row_count"], 6)
            self.assertEqual(summary["selected_bucket_counts"]["__negative__"], 1)
            self.assertEqual(sum(summary["selected_bucket_counts"].values()), 6)


if __name__ == "__main__":
    unittest.main()
