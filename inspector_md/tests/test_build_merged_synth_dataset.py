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

from inspector_md import build_inspector_dataset as build_dataset_mod
from inspector_md import build_merged_synth_dataset as merge_mod
from inspector_md import common, ontology


def _write_image(path: Path, *, size: tuple[int, int] = (80, 60), color: str = "white") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _build_mbdd_xml(image_name: str) -> str:
    return f"""<?xml version='1.0' encoding='utf-8'?>
<annotation>
  <filename>{image_name}</filename>
  <size>
    <width>100</width>
    <height>80</height>
    <depth>3</depth>
  </size>
  <object>
    <name>crack</name>
    <bndbox>
      <xmin>10</xmin>
      <ymin>8</ymin>
      <xmax>70</xmax>
      <ymax>44</ymax>
    </bndbox>
  </object>
</annotation>
"""


def _build_codebrim_xml(entries: dict[str, dict[str, int]]) -> str:
    lines = ["<?xml version='1.0' encoding='utf-8'?>", "<Annotation>"]
    ordered_labels = ("Background", "Crack", "Spallation", "Efflorescence", "ExposedBars", "CorrosionStain")
    for name, labels in entries.items():
        lines.append(f'  <Defect name="{name}">')
        for label in ordered_labels:
            value = int(labels.get(label, 0))
            lines.append(f"    <{label}>{value}</{label}>")
        lines.append("  </Defect>")
    lines.append("</Annotation>")
    return "\n".join(lines) + "\n"


def _create_raw_fixture(root: Path) -> None:
    mbdd_root = root / "MBDD2025"
    _write_image(mbdd_root / "JPEGImages" / "mbdd_sample.jpg", size=(100, 80), color="lightgray")
    _write_text(mbdd_root / "Annotations" / "mbdd_sample.xml", _build_mbdd_xml("mbdd_sample.jpg"))
    (mbdd_root / "Labels").mkdir(parents=True, exist_ok=True)

    cubit_root = root / "CUBIT-Det"
    _write_image(cubit_root / "images" / "train2017" / "train_labeled.jpg", size=(120, 90), color="silver")
    _write_text(cubit_root / "labels" / "train2017" / "train_labeled.txt", "0 0.50 0.50 0.40 0.30\n")
    _write_image(cubit_root / "images" / "train2017" / "train_unlabeled.jpg", size=(120, 90), color="gray")
    _write_image(cubit_root / "images" / "test2017" / "test_labeled.jpg", size=(120, 90), color="white")
    _write_text(cubit_root / "labels" / "test2017" / "test_labeled.txt", "2 0.40 0.60 0.20 0.40\n")
    _write_image(cubit_root / "images" / "val2017" / "val_only.jpg", size=(120, 90), color="black")
    (cubit_root / "labels" / "val2017").mkdir(parents=True, exist_ok=True)

    codebrim_root = root / "classification_dataset_balanced"
    _write_image(codebrim_root / "train" / "defects" / "defect_1.png", size=(64, 64), color="tan")
    _write_image(codebrim_root / "train" / "background" / "background_1.png", size=(64, 64), color="navy")
    for split in ("val", "test"):
        (codebrim_root / split / "defects").mkdir(parents=True, exist_ok=True)
        (codebrim_root / split / "background").mkdir(parents=True, exist_ok=True)
    _write_text(
        codebrim_root / "metadata" / "defects.xml",
        _build_codebrim_xml(
            {
                "defect_1.png": {
                    "Background": 0,
                    "Crack": 0,
                    "Spallation": 1,
                    "Efflorescence": 0,
                    "ExposedBars": 0,
                    "CorrosionStain": 0,
                }
            }
        ),
    )
    _write_text(
        codebrim_root / "metadata" / "background.xml",
        _build_codebrim_xml(
            {
                "background_1.png": {
                    "Background": 1,
                    "Crack": 0,
                    "Spallation": 0,
                    "Efflorescence": 0,
                    "ExposedBars": 0,
                    "CorrosionStain": 0,
                }
            }
        ),
    )


def _fake_openrouter_call(**kwargs):
    prompt_payload = json.loads(kwargs["messages"][1]["content"][0]["text"])
    record_payload = prompt_payload["record"]
    annotations = list(record_payload.get("annotations") or [])
    if bool(record_payload.get("is_negative")):
        payload = {
            "inspection_request": "Inspect this crop and confirm whether any supported defects are visible.",
            "asset_context": "Background crop for a structural inspection dataset.",
            "reasoning_text": "No supported defect annotation is present, so this should stay a negative example.",
            "proposals": [],
            "findings": [],
        }
        return json.dumps(payload), {"id": "fake-negative"}, 5.0

    seen_issue_codes: set[str] = set()
    proposals: list[dict[str, str]] = []
    findings: list[dict[str, object]] = []
    for annotation in annotations:
        issue_code = annotation["issue_code"]
        if issue_code not in seen_issue_codes:
            seen_issue_codes.add(issue_code)
            proposals.append(
                {
                    "issue_code": issue_code,
                    "evidence": f"Visible evidence of {ontology.get_issue(issue_code).title.lower()} is present.",
                }
            )
        findings.append(
            {
                "annotation_index": annotation["annotation_index"],
                "issue_code": issue_code,
                "title": ontology.get_issue(issue_code).title,
                "evidence": [f"Observed {ontology.get_issue(issue_code).title.lower()} in the labeled region."],
                "severity": "major" if issue_code in {"crack_defect", "surface_spalling"} else "moderate",
                "recommended_action": ontology.get_issue(issue_code).default_recommended_action,
                "insufficient_evidence": False,
            }
        )
    payload = {
        "inspection_request": "Inspect this building image for visible defects.",
        "asset_context": "Merged structural-inspection training example.",
        "reasoning_text": "The response is derived from the provided labeled regions only.",
        "proposals": proposals,
        "findings": findings,
    }
    return json.dumps(payload), {"id": "fake-positive"}, 7.5


class MergedSynthDatasetTests(unittest.TestCase):
    def test_normalize_stage_parses_and_materializes_expected_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            raw_root = tmp / "raw_datasets"
            output_dir = tmp / "dataset_out"
            _create_raw_fixture(raw_root)

            args = merge_mod.parse_args(
                [
                    "--stage",
                    "normalize",
                    "--raw-dataset-root",
                    str(raw_root),
                    "--output-dir",
                    str(output_dir),
                ]
            )
            summary = merge_mod.build_merged_synth_dataset(args)

            self.assertEqual(summary["stage"], "normalize")
            base_records = common.load_jsonl(output_dir / "base_records.jsonl")
            stats = json.loads((output_dir / "stats.json").read_text(encoding="utf-8"))
            mapping_summary = json.loads((output_dir / "mapping_summary.json").read_text(encoding="utf-8"))

            self.assertEqual(len(base_records), 5)
            self.assertEqual(stats["record_count"], 5)
            self.assertEqual(stats["positive_record_count"], 4)
            self.assertEqual(stats["negative_record_count"], 1)
            self.assertEqual(stats["source_counts"]["MBDD2025"], 1)
            self.assertEqual(stats["source_counts"]["CUBIT-Det"], 2)
            self.assertEqual(stats["source_counts"]["CODEBRIM"], 2)
            self.assertEqual(mapping_summary["sources"]["MBDD2025"]["raw_to_issue_code"]["crack"], "crack_defect")
            self.assertEqual(mapping_summary["sources"]["CUBIT-Det"]["raw_to_issue_code"]["moisture"], "moisture_intrusion")
            self.assertEqual(mapping_summary["sources"]["CODEBRIM"]["raw_to_issue_code"]["Spallation"], "surface_spalling")

            mbdd_record = next(row for row in base_records if row["source_dataset"] == "MBDD2025")
            self.assertIn(mbdd_record["split"], {"train", "validation", "test"})
            self.assertTrue(Path(mbdd_record["image_path"]).exists())

            codebrim_defect = next(row for row in base_records if row["row_id"].startswith("codebrim:train:defects"))
            self.assertEqual(codebrim_defect["annotations"][0]["box"], {"x_min": 0.0, "y_min": 0.0, "x_max": 1.0, "y_max": 1.0})

            codebrim_background = next(row for row in base_records if row["row_id"].startswith("codebrim:train:background"))
            self.assertTrue(codebrim_background["is_negative"])
            self.assertEqual(codebrim_background["annotations"], [])

            raw_paths = {Path(row["raw_image_path"]).name for row in base_records}
            self.assertNotIn("train_unlabeled.jpg", raw_paths)
            self.assertNotIn("val_only.jpg", raw_paths)

            summary_repeat = merge_mod.build_merged_synth_dataset(args)
            repeated_records = common.load_jsonl(output_dir / "base_records.jsonl")
            repeated_mbdd_record = next(row for row in repeated_records if row["source_dataset"] == "MBDD2025")
            self.assertEqual(summary_repeat["normalize"]["record_count"], 5)
            self.assertEqual(repeated_mbdd_record["split"], mbdd_record["split"])

    def test_synthesize_stage_uses_cache_and_manifest_feeds_builder(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            raw_root = tmp / "raw_datasets"
            output_dir = tmp / "dataset_out"
            _create_raw_fixture(raw_root)

            args = merge_mod.parse_args(
                [
                    "--stage",
                    "all",
                    "--raw-dataset-root",
                    str(raw_root),
                    "--output-dir",
                    str(output_dir),
                    "--teacher-model-id",
                    "openrouter/fake-model",
                    "--api-key",
                    "fake-openrouter-key",
                    "--max-positive-records",
                    "2",
                    "--max-hard-negatives",
                    "1",
                ]
            )
            summary = merge_mod.build_merged_synth_dataset(args, call_openrouter_fn=_fake_openrouter_call)
            synthetic_manifest = json.loads((output_dir / "synthetic_manifest.json").read_text(encoding="utf-8"))
            cache_rows = common.load_jsonl(output_dir / "openrouter_cache.jsonl")

            self.assertEqual(summary["synthesize"]["synthesis_mode"], "openrouter")
            self.assertEqual(summary["synthesize"]["selected_record_count"], 3)
            self.assertEqual(summary["synthesize"]["manifest_row_count"], 3)
            self.assertEqual(len(synthetic_manifest), 3)
            self.assertEqual(len(cache_rows), 3)
            self.assertTrue(all("expected_proposals" in row and "expected_findings" in row for row in synthetic_manifest))
            self.assertTrue(
                all(
                    row["inspection_request"] == merge_mod.prompt_library.CANONICAL_QUERY_INSPECTION_REQUEST
                    for row in synthetic_manifest
                )
            )

            def _unexpected_network(**kwargs):
                raise AssertionError("OpenRouter caller should not be used when cache is populated.")

            summary_cached = merge_mod.build_merged_synth_dataset(args, call_openrouter_fn=_unexpected_network)
            self.assertEqual(summary_cached["synthesize"]["used_cache_count"], 3)

            build_args = build_dataset_mod.parse_args(
                [
                    "--source-manifest",
                    str(output_dir / "synthetic_manifest.json"),
                    "--output-root",
                    str(tmp / "builder_outputs"),
                    "--detect-output-dir",
                    str(tmp / "builder_outputs" / "detect"),
                    "--point-output-dir",
                    str(tmp / "builder_outputs" / "point"),
                    "--query-proposal-output-dir",
                    str(tmp / "builder_outputs" / "proposal"),
                    "--query-finding-output-dir",
                    str(tmp / "builder_outputs" / "finding"),
                    "--query-reasoning-output-dir",
                    str(tmp / "builder_outputs" / "reasoning"),
                    "--query-text-refresh-mode",
                    "template_only",
                ]
            )
            build_summary = build_dataset_mod.build_dataset(build_args)
            self.assertTrue((Path(build_summary["query_finding_output_dir"]) / "jsonl" / "train.jsonl").exists())
            self.assertTrue((tmp / "builder_outputs" / "source_manifest.normalized.json").exists())

    def test_template_only_synthesis_writes_builder_ready_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            raw_root = tmp / "raw_datasets"
            output_dir = tmp / "dataset_out"
            _create_raw_fixture(raw_root)

            args = merge_mod.parse_args(
                [
                    "--stage",
                    "all",
                    "--raw-dataset-root",
                    str(raw_root),
                    "--output-dir",
                    str(output_dir),
                    "--max-positive-records",
                    "2",
                    "--max-hard-negatives",
                    "1",
                ]
            )
            summary = merge_mod.build_merged_synth_dataset(args)
            synthetic_manifest = json.loads((output_dir / "synthetic_manifest.json").read_text(encoding="utf-8"))
            stats = json.loads((output_dir / "stats.json").read_text(encoding="utf-8"))
            build_summary = json.loads((output_dir / "build_summary.json").read_text(encoding="utf-8"))

            self.assertEqual(summary["synthesize"]["synthesis_mode"], "template_only")
            self.assertEqual(len(synthetic_manifest), 3)
            self.assertEqual(stats["synthesized_record_count"], 3)
            self.assertEqual(build_summary["synthesize"]["synthesis_mode"], "template_only")
            self.assertTrue(
                all(
                    row["inspection_request"] == merge_mod.prompt_library.CANONICAL_QUERY_INSPECTION_REQUEST
                    for row in synthetic_manifest
                )
            )
            first_positive = next(row for row in synthetic_manifest if row["expected_findings"])
            self.assertTrue(first_positive["expected_proposals"])
            self.assertTrue(first_positive["expected_findings"][0]["source_detect_labels"])


if __name__ == "__main__":
    unittest.main()
