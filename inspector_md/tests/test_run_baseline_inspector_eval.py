from __future__ import annotations

import argparse
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inspector_md import run_baseline_inspector_eval as baseline_mod
from inspector_md import task_schema


def _write_image(path: Path, color: str = "white") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (120, 90), color).save(path)
    return path


class BaselineEvalArtifactTests(unittest.TestCase):
    def test_filter_samples_excludes_codebrim_for_detect_evals(self) -> None:
        sample_a = baseline_mod.EvalSample(
            row_id="a",
            split="test",
            source_dataset="CODEBRIM",
            source_metadata={},
            request=task_schema.InspectionRequest(image_path="/tmp/a.png", inspection_request="inspect", asset_context=""),
            expected_proposals=[],
            expected_findings=[],
        )
        sample_b = baseline_mod.EvalSample(
            row_id="b",
            split="test",
            source_dataset="MBDD2025",
            source_metadata={},
            request=task_schema.InspectionRequest(image_path="/tmp/b.png", inspection_request="inspect", asset_context=""),
            expected_proposals=[],
            expected_findings=[],
        )
        filtered = baseline_mod._filter_samples(
            [sample_a, sample_b],
            split="test",
            seed=42,
            max_samples=0,
            excluded_source_datasets={"CODEBRIM"},
        )
        self.assertEqual([sample.row_id for sample in filtered], ["b"])

    def test_non_exact_issue_code_match_gets_partial_credit(self) -> None:
        proposal_metrics = baseline_mod._proposal_metrics(
            [task_schema.IssueProposal(issue_code="efflorescence_deposit", evidence="white mineral deposit", confidence=1.0)],
            [task_schema.IssueProposal(issue_code="water_staining", evidence="staining visible", confidence=1.0)],
        )
        self.assertEqual(proposal_metrics["precision"], 0.5)
        self.assertEqual(proposal_metrics["recall"], 0.5)
        self.assertEqual(proposal_metrics["f1"], 0.5)

    def test_write_sample_artifacts_creates_expected_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            image_path = _write_image(tmp / "sample.png")
            sample = baseline_mod.EvalSample(
                row_id="row-1",
                split="test",
                source_dataset="CODEBRIM",
                source_metadata={"note": "synthetic"},
                request=task_schema.InspectionRequest(
                    image_path=str(image_path),
                    inspection_request="Inspect this image.",
                    asset_context="Concrete wall.",
                ),
                expected_proposals=[
                    task_schema.IssueProposal(issue_code="surface_spalling", evidence="spalling visible", confidence=0.9)
                ],
                expected_findings=[
                    task_schema.Finding.from_payload(
                        {
                            "finding_id": "gt-1",
                            "issue_code": "surface_spalling",
                            "title": "Surface Spalling",
                            "box": {"x_min": 0.1, "y_min": 0.2, "x_max": 0.8, "y_max": 0.7},
                            "evidence": ["spalling visible"],
                            "severity": "major",
                            "recommended_action": "Repair the damaged concrete.",
                            "cost_band": "high",
                            "possible_compliance_issue": True,
                            "insufficient_evidence": False,
                            "compliance_note": "Possible compliance issue related to surface_stability.",
                            "source_detect_labels": ["surface spalling"],
                            "spatial_ref_index": 0,
                        }
                    )
                ],
            )
            report = task_schema.InspectionReport.from_payload(
                {
                    "request": sample.request.to_payload(),
                    "detect_model": "moondream3-preview",
                    "query_model": "moondream3-preview",
                    "findings": [
                        {
                            "finding_id": "pred-1",
                            "issue_code": "surface_spalling",
                            "title": "Surface Spalling",
                            "box": {"x_min": 0.12, "y_min": 0.22, "x_max": 0.82, "y_max": 0.72},
                            "evidence": ["spalling visible in the highlighted region"],
                            "severity": "major",
                            "recommended_action": "Repair the damaged concrete.",
                            "cost_band": "very_high",
                            "possible_compliance_issue": True,
                            "insufficient_evidence": False,
                            "compliance_note": "Possible compliance issue related to surface_stability.",
                            "source_detect_labels": ["surface spalling"],
                            "spatial_ref_index": 0,
                        }
                    ],
                    "summary": {"finding_count": 1},
                    "trace": {"proposals": [item.to_payload() for item in sample.expected_proposals]},
                }
            )
            events = [
                {"task": "proposal_query", "prompt": "proposal prompt", "parsed_response": {"proposals": ["x"]}},
                {"task": "detect", "prompt": "surface spalling", "parsed_response": {"boxes": [1]}},
                {"task": "finding_query", "prompt": "finding prompt", "parsed_response": {"finding": {"issue_code": "surface_spalling"}}},
            ]
            metrics = {
                "proposal_precision": 1.0,
                "proposal_recall": 1.0,
                "proposal_f1": 1.0,
                "localization_precision": 1.0,
                "localization_recall": 1.0,
                "localization_f1": 1.0,
                "end_to_end_score": 0.9,
                "finding_schema_valid": 1.0,
            }
            output = baseline_mod._write_sample_artifacts(
                output_dir=tmp / "eval",
                sample=sample,
                report=report,
                events=events,
                metrics=metrics,
                save_viz=True,
                max_viz_dim=800,
            )
            sample_dir = Path(output["sample_dir"])
            self.assertTrue((sample_dir / "record.json").exists())
            self.assertTrue((sample_dir / "tasks.json").exists())
            self.assertTrue((sample_dir / "tasks.md").exists())
            self.assertTrue((sample_dir / "report.json").exists())
            self.assertTrue((sample_dir / "punch_list.txt").exists())
            self.assertTrue((sample_dir / "comparison.png").exists())

    def test_compute_query_judge_details_scores_missing_and_unmatched_findings_as_zero(self) -> None:
        sample = baseline_mod.EvalSample(
            row_id="row-1",
            split="test",
            source_dataset="MBDD2025",
            source_metadata={},
            request=task_schema.InspectionRequest(image_path="/tmp/sample.png", inspection_request="Inspect this image.", asset_context=""),
            expected_proposals=[],
            expected_findings=[
                task_schema.Finding.from_payload(
                    {
                        "finding_id": "gt-1",
                        "issue_code": "surface_spalling",
                        "title": "Surface Spalling",
                        "box": {"x_min": 0.1, "y_min": 0.1, "x_max": 0.4, "y_max": 0.4},
                        "evidence": ["spalling visible"],
                        "severity": "major",
                        "recommended_action": "repair the damaged concrete",
                        "cost_band": "high",
                        "possible_compliance_issue": False,
                        "insufficient_evidence": False,
                        "compliance_note": "",
                        "source_detect_labels": ["surface spalling"],
                        "spatial_ref_index": 0,
                    }
                )
            ],
        )
        report = task_schema.InspectionReport.from_payload(
            {
                "request": sample.request.to_payload(),
                "detect_model": "moondream3-preview",
                "query_model": "moondream3-preview",
                "findings": [
                    {
                        "finding_id": "pred-1",
                        "issue_code": "corrosion_rust",
                        "title": "Corrosion or Rust",
                        "box": {"x_min": 0.6, "y_min": 0.6, "x_max": 0.9, "y_max": 0.9},
                        "evidence": ["rust visible"],
                        "severity": "moderate",
                        "recommended_action": "repair the corroded metal",
                        "cost_band": "medium",
                        "possible_compliance_issue": False,
                        "insufficient_evidence": False,
                        "compliance_note": "",
                        "source_detect_labels": ["rusted metal"],
                        "spatial_ref_index": 0,
                    }
                ],
                "summary": {"finding_count": 1},
                "trace": {"proposals": []},
            }
        )
        details = baseline_mod._compute_query_judge_details(
            sample=sample,
            report=report,
            events=[{"task": "finding_query", "answer_text": "Rust is visible on the metal."}],
            grader=object(),
            coarse_iou_threshold=0.1,
        )
        self.assertEqual(details["metrics"]["finding_judge_score"], 0.0)
        self.assertEqual(len(details["finding_judgements"]), 2)
        self.assertEqual(details["finding_judgements"][0]["judge_output"]["score"], 0.0)
        self.assertEqual(details["finding_judgements"][1]["judge_output"]["score"], 0.0)

    def test_run_baseline_eval_retries_timeout_once_and_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            image_path = _write_image(tmp / "sample.png")
            sample = baseline_mod.EvalSample(
                row_id="row-1",
                split="test",
                source_dataset="MBDD2025",
                source_metadata={},
                request=task_schema.InspectionRequest(
                    image_path=str(image_path),
                    inspection_request="Inspect this image.",
                    asset_context="Concrete wall.",
                ),
                expected_proposals=[],
                expected_findings=[],
            )

            class FakePool:
                env_var_names = ["KEY_1"]
                slots = ["slot"]

            class FakeClient:
                def __init__(self, **kwargs):
                    self.events = []

                def start_sample(self, row_id):
                    self.events = []

            class FakePipeline:
                attempts = 0

                def __init__(self, *, client, models, settings):
                    self.client = client

                def run(self, request):
                    FakePipeline.attempts += 1
                    if FakePipeline.attempts == 1:
                        raise TimeoutError("transient timeout")
                    return task_schema.InspectionReport.from_payload(
                        {
                            "request": request.to_payload(),
                            "detect_model": "moondream3-preview",
                            "query_model": "moondream3-preview",
                            "findings": [],
                            "summary": {"finding_count": 0},
                            "trace": {"proposals": []},
                        }
                    )

            args = argparse.Namespace(
                env_file=str(tmp / ".env"),
                api_key="",
                api_key_env_vars=["KEY_1"],
                base_url="https://api-staging.moondream.ai/v1",
                dataset_manifest=str(tmp / "manifest.json"),
                split="all",
                max_samples=0,
                seed=42,
                detect_finetune_id="",
                query_finetune_id="",
                base_model="moondream3-preview",
                reasoning=False,
                detect_max_objects=24,
                iou_threshold=0.5,
                coarse_iou_threshold=0.1,
                detect_exclude_source_datasets=[],
                proposal_prompt_style="request_only",
                finding_prompt_style="minimal",
                query_response_mode="json",
                grading_mode="rule",
                grader_api_key="",
                grader_api_key_env_var="OPENROUTER_API_KEY",
                grader_api_base="https://openrouter.ai/api/v1",
                grader_model_id="openai/gpt-4.1-mini",
                grader_profile="balanced",
                grader_rubric_version="query_rubric_v1",
                grader_timeout=60.0,
                output_dir=str(tmp / "eval"),
                save_viz=False,
                max_viz_dim=800,
                timeout=10.0,
            )
            with (
                patch.object(baseline_mod.common, "maybe_load_env_file", return_value=None),
                patch.object(baseline_mod.common, "resolve_api_key_pool", return_value=FakePool()),
                patch.object(baseline_mod, "_load_samples", return_value=[sample]),
                patch.object(baseline_mod, "TracingMoondreamInspectorClient", FakeClient),
                patch.object(baseline_mod, "InspectorPipeline", FakePipeline),
            ):
                summary = baseline_mod.run_baseline_eval(args)
            self.assertEqual(summary["successful_sample_count"], 1)
            self.assertEqual(summary["failed_sample_count"], 0)
            self.assertEqual(summary["samples"][0]["retry_count"], 1)

    def test_run_baseline_eval_persistent_timeout_records_stage_and_retry_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            image_path = _write_image(tmp / "sample.png")
            sample = baseline_mod.EvalSample(
                row_id="row-1",
                split="test",
                source_dataset="MBDD2025",
                source_metadata={},
                request=task_schema.InspectionRequest(
                    image_path=str(image_path),
                    inspection_request="Inspect this image.",
                    asset_context="Concrete wall.",
                ),
                expected_proposals=[],
                expected_findings=[],
            )

            class FakePool:
                env_var_names = ["KEY_1"]
                slots = ["slot"]

            class FakeClient:
                def __init__(self, **kwargs):
                    self.events = []

                def start_sample(self, row_id):
                    self.events = []

            class FakePipeline:
                def __init__(self, *, client, models, settings):
                    self.client = client

                def run(self, request):
                    raise TimeoutError("persistent timeout")

            args = argparse.Namespace(
                env_file=str(tmp / ".env"),
                api_key="",
                api_key_env_vars=["KEY_1"],
                base_url="https://api-staging.moondream.ai/v1",
                dataset_manifest=str(tmp / "manifest.json"),
                split="all",
                max_samples=0,
                seed=42,
                detect_finetune_id="",
                query_finetune_id="",
                base_model="moondream3-preview",
                reasoning=False,
                detect_max_objects=24,
                iou_threshold=0.5,
                coarse_iou_threshold=0.1,
                detect_exclude_source_datasets=[],
                proposal_prompt_style="request_only",
                finding_prompt_style="minimal",
                query_response_mode="json",
                grading_mode="rule",
                grader_api_key="",
                grader_api_key_env_var="OPENROUTER_API_KEY",
                grader_api_base="https://openrouter.ai/api/v1",
                grader_model_id="openai/gpt-4.1-mini",
                grader_profile="balanced",
                grader_rubric_version="query_rubric_v1",
                grader_timeout=60.0,
                output_dir=str(tmp / "eval"),
                save_viz=False,
                max_viz_dim=800,
                timeout=10.0,
            )
            with (
                patch.object(baseline_mod.common, "maybe_load_env_file", return_value=None),
                patch.object(baseline_mod.common, "resolve_api_key_pool", return_value=FakePool()),
                patch.object(baseline_mod, "_load_samples", return_value=[sample]),
                patch.object(baseline_mod, "TracingMoondreamInspectorClient", FakeClient),
                patch.object(baseline_mod, "InspectorPipeline", FakePipeline),
            ):
                summary = baseline_mod.run_baseline_eval(args)
            self.assertEqual(summary["successful_sample_count"], 0)
            self.assertEqual(summary["failed_sample_count"], 1)
            self.assertEqual(summary["failures"][0]["retry_count"], 1)
            self.assertEqual(summary["failures"][0]["failure_stage"], "proposal_query")


if __name__ == "__main__":
    unittest.main()
