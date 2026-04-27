from __future__ import annotations

import json
import contextlib
import io
import math
import os
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inspector_md import analyze_query_benchmarks as analyze_query_benchmarks_mod
from inspector_md import common, ontology, openrouter_grader, prompt_library, query_compact, task_schema
from inspector_md import benchmark_inspector_detect as bench_detect_mod
from inspector_md import benchmark_inspector_pipeline as benchmark_mod
from inspector_md import benchmark_inspector_query as bench_query_mod
from inspector_md import build_inspector_dataset as build_mod
from inspector_md import build_inspector_report as report_mod
from inspector_md import build_merged_synth_dataset as merge_mod
from inspector_md import check_inspector_finetune_readiness as readiness_mod
from inspector_md import generate_class_samples_viz as viz_mod
from inspector_md import moondream_client
from inspector_md import pipeline as pipeline_mod
from inspector_md import run_baseline_inspector_eval as baseline_eval_mod
from inspector_md import run_inspector_dataset_refresh_batches as refresh_batches_mod
from inspector_md import run_inspector_pipeline as run_pipeline_mod
from inspector_md import run_inspector_sweep as sweep_mod
from inspector_md import train_inspector_detect as train_detect_mod
from inspector_md import train_inspector_point as train_point_mod
from inspector_md import train_inspector_query as train_query_mod
from tuna_sdk import QueryRequest, QuerySFTTarget, TrainStepGroup
from tuna_sdk.errors import TunaAPIError


def _write_image(path: Path, color: str = "white") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 48), color).save(path)
    return path


def _fake_query_refresh_call(**kwargs):
    return (
        json.dumps(
            {
                "inspection_request": "Inspect this image for visible signs of roof cover damage.",
                "asset_context": "Close-up exterior roof image.",
                "reasoning_text": "Use visible damage only.",
                "proposals": [
                    {
                        "issue_code": "roof_cover_damage",
                        "evidence": "Missing shingles are visible on the roof surface.",
                    }
                ],
                "findings": [
                    {
                        "finding_index": 0,
                        "issue_code": "roof_cover_damage",
                        "title": "Roof Cover Damage",
                        "evidence": ["Missing shingles are visible on the roof surface."],
                        "severity": "major",
                        "recommended_action": "Repair the damaged roof covering.",
                        "insufficient_evidence": False,
                    }
                ],
            }
        ),
        {"id": "fake-query-refresh"},
        5.0,
    )


class OntologyAndSchemaTests(unittest.TestCase):
    def test_aliases_normalize_to_canonical_issue_code(self) -> None:
        self.assertEqual(ontology.normalize_issue_code("roof damage"), "roof_cover_damage")
        self.assertEqual(ontology.normalize_issue_code("blocked access"), "blocked_egress_or_access")

    def test_loose_issue_normalization_repairs_common_query_label_corruption(self) -> None:
        self.assertEqual(ontology.normalize_issue_code("crack_visible", loose=True), "crack_defect")
        self.assertEqual(ontology.normalize_issue_code("crack_definition", loose=True), "crack_defect")
        self.assertEqual(ontology.normalize_issue_code("corrosion_rust_rust", loose=True), "corrosion_rust")
        self.assertEqual(ontology.normalize_issue_code("corrosion_corrosion", loose=True), "corrosion_rust")
        self.assertEqual(ontology.normalize_issue_code("material_abef", loose=True), "material_abscission")
        self.assertEqual(ontology.normalize_issue_code("subrosion", loose=True), "corrosion_rust")
        with self.assertRaises(ValueError):
            ontology.normalize_issue_code("crack_visible")

    def test_issue_match_score_supports_family_level_partial_credit(self) -> None:
        self.assertEqual(ontology.issue_match_score("efflorescence_deposit", "efflorescence_deposit"), 1.0)
        self.assertEqual(ontology.issue_match_score("efflorescence_deposit", "water_staining"), 0.5)
        self.assertEqual(ontology.issue_match_score("efflorescence_deposit", "corrosion_rust"), 0.0)

    def test_detect_labels_are_specific_and_not_generic(self) -> None:
        for record in ontology.ISSUE_CATALOG:
            self.assertTrue(record.detect_labels)
            for label in record.detect_labels:
                self.assertFalse(ontology.is_generic_detect_label(label))

    def test_finding_normalization_stabilizes_payload(self) -> None:
        finding = task_schema.Finding.from_payload(
            {
                "finding_id": "f1",
                "issue_code": "roof damage",
                "title": "Roof Cover Damage",
                "box": [0.1, 0.2, 0.4, 0.5],
                "evidence": ["missing shingles visible"],
                "severity": "major",
                "recommended_action": "repair roof",
                "cost_band": "high",
                "possible_compliance_issue": True,
                "insufficient_evidence": False,
                "compliance_note": "possible code concern",
                "source_detect_labels": ["missing shingle"],
                "spatial_ref_index": 0,
            }
        )
        self.assertEqual(finding.issue_code, "roof_cover_damage")
        self.assertEqual(finding.box.to_payload()["x_min"], 0.1)


class QueryTrainerRetryTests(unittest.TestCase):
    def test_query_parse_args_accepts_both_alias(self) -> None:
        args = train_query_mod.parse_args(["--mode", "both"])
        self.assertEqual(args.mode, "sft_then_rl")

    def test_query_sft_target_omits_reasoning_by_default(self) -> None:
        example = train_query_mod.QueryExample(
            row_id="row-1",
            split="train",
            task_type="proposal",
            image_path=Path("/tmp/example.jpg"),
            inspection_request="Inspect this image.",
            asset_context="",
            spatial_refs=[],
            question="Inspect this image.",
            target_text="corrosion_rust | Rust visible in annotated area.",
            target_format="compact_text",
            final_answer_json='{"proposals":[]}',
            reasoning_text="Visible rust on beam edge supports the finding.",
            hard_example=False,
            query_text_refresh_mode="openrouter",
        )
        target = train_query_mod._sft_target(example)
        self.assertEqual(target.answer, example.target_text)
        self.assertIsNone(target.reasoning)

    def test_query_sft_target_includes_reasoning_only_when_requested(self) -> None:
        example = train_query_mod.QueryExample(
            row_id="row-2",
            split="train",
            task_type="finding",
            image_path=Path("/tmp/example.jpg"),
            inspection_request="Inspect this image.",
            asset_context="",
            spatial_refs=[[0.1, 0.1, 0.2, 0.2]],
            question="Inspect this image.",
            target_text="crack_defect | Crack Defect | moderate | false | Visible crack. | Repair cracks.",
            target_format="compact_text",
            final_answer_json='{"issue_code":"crack_defect"}',
            reasoning_text="A visible crack is present in the highlighted region.",
            hard_example=True,
            query_text_refresh_mode="openrouter",
        )
        target = train_query_mod._sft_target(example, include_reasoning=True)
        self.assertEqual(target.answer, example.target_text)
        self.assertEqual(target.reasoning, example.reasoning_text)

    def test_query_reasoning_sft_fails_fast_before_backend_calls(self) -> None:
        with self.assertRaisesRegex(ValueError, "Query SFT with --reasoning is currently unsupported"):
            train_query_mod.main(
                [
                    "--config",
                    str(common.repo_relative("configs", "train_inspector_query_reasoning_hard.json")),
                    "--mode",
                    "sft",
                    "--sft-steps",
                    "1",
                    "--rl-steps",
                    "0",
                ]
            )

    def test_query_sft_group_retry_recovers_from_transient_524(self) -> None:
        group = TrainStepGroup.from_sft(
            request=QueryRequest(question="Inspect this image."),
            targets=[QuerySFTTarget(answer="none")],
        )

        call_count = {"value": 0}

        class _FakeFinetune:
            def train_step(self, *, groups, lr):
                call_count["value"] += 1
                if call_count["value"] < 3:
                    raise TunaAPIError(
                        "error code: 524",
                        status_code=524,
                        response_body="error code: 524",
                        request_id="req-524",
                    )
                return SimpleNamespace(
                    step=1,
                    applied=True,
                    sft_loss=0.25,
                    kl=0.0,
                    router_kl=0.0,
                    grad_norm=0.0,
                    reward_mean=0.0,
                    reward_std=0.0,
                )

        with patch.object(train_query_mod.time, "sleep", return_value=None):
            result = train_query_mod._train_query_sft_groups(
                finetune=_FakeFinetune(),
                groups=[group],
                lr=2e-4,
                train_step_max_retries=3,
                train_step_retry_backoff_base_s=0.1,
                train_step_retry_backoff_max_s=0.2,
                step_label="query sft step 1",
            )
        self.assertTrue(result.applied)
        self.assertEqual(call_count["value"], 3)

    def test_query_train_step_retry_does_not_retry_non_transient_400(self) -> None:
        call_count = {"value": 0}

        def _invoke():
            call_count["value"] += 1
            raise TunaAPIError(
                "bad request",
                status_code=400,
                response_body="bad request",
                request_id="req-400",
            )

        with patch.object(train_query_mod.time, "sleep", return_value=None):
            with self.assertRaises(TunaAPIError):
                train_query_mod._query_train_step_with_retry(
                    invoke=_invoke,
                    context="query sft step 1",
                    max_retries=3,
                    backoff_base_s=0.1,
                    backoff_max_s=0.2,
                )
        self.assertEqual(call_count["value"], 1)

    def test_query_sft_groups_keep_partial_microbatch_progress(self) -> None:
        groups = [
            TrainStepGroup.from_sft(
                request=QueryRequest(question="Inspect this image."),
                targets=[QuerySFTTarget(answer="none")],
            ),
            TrainStepGroup.from_sft(
                request=QueryRequest(question="Inspect this image."),
                targets=[QuerySFTTarget(answer="none")],
            ),
        ]

        call_count = {"value": 0}

        class _FakeFinetune:
            def train_step(self, *, groups, lr):
                call_count["value"] += 1
                if call_count["value"] == 1:
                    return SimpleNamespace(
                        step=1,
                        applied=True,
                        sft_loss=0.25,
                        kl=0.0,
                        router_kl=0.0,
                        grad_norm=0.0,
                        reward_mean=0.0,
                        reward_std=0.0,
                    )
                raise TunaAPIError(
                    "error code: 500",
                    status_code=500,
                    response_body={"detail": "Internal Server Error"},
                    request_id="req-500",
                )

        with patch.object(train_query_mod.time, "sleep", return_value=None):
            result = train_query_mod._train_query_sft_groups(
                finetune=_FakeFinetune(),
                groups=groups,
                lr=2e-4,
                train_step_max_retries=1,
                train_step_retry_backoff_base_s=0.1,
                train_step_retry_backoff_max_s=0.2,
                step_label="query sft step 1",
            )
        self.assertTrue(result.applied)
        self.assertEqual(result.successful_microbatch_count, 1)
        self.assertEqual(result.failed_microbatch_count, 1)

    def test_query_main_logs_sft_and_rl_phase_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            image_path = _write_image(tmp / "example.jpg")
            example = train_query_mod.QueryExample(
                row_id="row-1",
                split="train",
                task_type="finding",
                image_path=image_path,
                inspection_request="Inspect this image.",
                asset_context="Test asset",
                spatial_refs=[[0.1, 0.1, 0.2, 0.2]],
                question="Inspect this image.",
                target_text="crack_defect | Crack Defect | moderate | false | Visible crack. | Repair cracks.",
                target_format="compact_text",
                final_answer_json='{"issue_code":"crack_defect"}',
                reasoning_text="Visible crack in the highlighted region.",
                hard_example=False,
                query_text_refresh_mode="openrouter",
            )
            examples = [example, example, example, example]

            class _DummyProgress:
                def __init__(self, total: int, desc: str) -> None:
                    self.total = total
                    self.desc = desc

                def set_postfix(self, *args, **kwargs) -> None:
                    return None

                def update(self, *args, **kwargs) -> None:
                    return None

                def close(self) -> None:
                    return None

            class _FakeRun:
                def __init__(self) -> None:
                    self.summary: dict[str, object] = {}

                def finish(self) -> None:
                    return None

            class _FakeRolloutResult:
                def __init__(self, *, num_rollouts: int) -> None:
                    self.rollouts = [
                        SimpleNamespace(output=SimpleNamespace(answer="crack_defect | Crack Defect"))
                        for _ in range(num_rollouts)
                    ]

                def to_group(self, *, rewards):
                    return SimpleNamespace(rewards=list(rewards))

            class _FakeFinetune:
                def __init__(self) -> None:
                    self.finetune_id = "ft-test"

                def train_step(self, *, groups, lr):
                    return SimpleNamespace(
                        applied=True,
                        sft_loss=0.25 if not getattr(groups[0], "rewards", None) else 0.05,
                        kl=0.01 if not getattr(groups[0], "rewards", None) else 0.02,
                        router_kl=0.0,
                        grad_norm=0.0,
                        reward_mean=0.0,
                        reward_std=0.0,
                    )

                def rollouts_batch(self, *, requests, num_rollouts, max_workers):
                    return [_FakeRolloutResult(num_rollouts=num_rollouts) for _ in requests]

                def save_checkpoint(self):
                    return SimpleNamespace(checkpoint=SimpleNamespace(step=7))

            class _FakeTunaClient:
                def __init__(self, **kwargs) -> None:
                    pass

                def create_finetune(self, **kwargs):
                    return _FakeFinetune()

                def get_finetune(self, finetune_id):
                    return _FakeFinetune()

            class _FakeGrader:
                def __init__(self, **kwargs) -> None:
                    self.model_id = "fake-grader"

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                with patch.object(train_query_mod.common, "maybe_load_env_file", return_value=None), patch.object(
                    train_query_mod, "_require_query_dataset_ready", return_value={"query_text_refresh_mode": "openrouter"}
                ), patch.object(
                    train_query_mod, "_load_split_examples", return_value=examples
                ), patch.object(
                    train_query_mod.openrouter_grader, "resolve_openrouter_api_key", return_value="or-key"
                ), patch.object(
                    train_query_mod.openrouter_grader, "OpenRouterGrader", _FakeGrader
                ), patch.object(
                    train_query_mod, "MoondreamInspectorClient", return_value=SimpleNamespace()
                ), patch.object(
                    train_query_mod, "TunaClient", _FakeTunaClient
                ), patch.object(
                    train_query_mod.wandb, "init", return_value=_FakeRun(), create=True
                ), patch.object(
                    train_query_mod.wandb, "log", return_value=None, create=True
                ), patch.object(
                    train_query_mod, "_make_progress_bar", side_effect=lambda total, desc: _DummyProgress(total, desc)
                ), patch.object(
                    train_query_mod, "_score_answer_text",
                    return_value=(
                        SimpleNamespace(reward=0.4, parse_success=True, task_correct=True),
                        SimpleNamespace(payload={"issue_code": "crack_defect"}),
                    ),
                ), patch.object(
                    train_query_mod, "_judge_answer_with_cache",
                    return_value={"score": 0.8, "degraded": False},
                ), patch.object(
                    train_query_mod, "_evaluate_split", return_value={"reward_mean": 0.5}
                ), patch.object(train_query_mod.time, "sleep", return_value=None):
                    train_query_mod.main(
                        [
                            "--api-key",
                            "test-key",
                            "--grader-api-key",
                            "or-key",
                            "--dataset-dir",
                            str(tmp),
                            "--run-output-dir",
                            str(tmp / "runs"),
                            "--mode",
                            "sft_then_rl",
                            "--sft-steps",
                            "1",
                            "--rl-steps",
                            "1",
                            "--eval-every",
                            "0",
                            "--save-every",
                            "0",
                            "--no-async-checkpoint-eval",
                            "--post-create-warmup-s",
                            "0",
                            "--final-eval-splits",
                            "validation",
                        ]
                    )

            output = stdout.getvalue()
            self.assertIn("query training plan: mode=sft_then_rl sft_steps=1 rl_steps=1 | RL starts automatically after SFT.", output)
            self.assertIn("starting query sft phase at step 1", output)
            self.assertIn("step 1 sft loss=", output)
            self.assertIn("query sft phase complete at step 1; starting rl phase at step 2", output)
            self.assertIn("step 2 rl reward=", output)


class PromptLibraryTests(unittest.TestCase):
    def test_prompts_are_shorter_and_remove_roof_example_leakage(self) -> None:
        localized = task_schema.LocalizedIssue.from_payload(
            {
                "issue_code": "surface_spalling",
                "box": [0.1, 0.1, 0.4, 0.5],
                "source_detect_labels": ["surface spalling"],
            }
        )
        proposal_prompt = prompt_library.build_visible_issue_question(
            inspection_request="Inspect this image for visible defects.",
            asset_context="Concrete wall crop",
        )
        finding_prompt = prompt_library.build_finding_question(
            localized_issue=localized,
            inspection_request="Inspect this image for visible defects.",
            asset_context="Concrete wall crop",
        )
        self.assertLess(len(proposal_prompt), 900)
        self.assertLess(len(finding_prompt), 650)
        self.assertIn('"issue_code": "<allowed_issue_code>"', proposal_prompt)
        self.assertIn('"issue_code": "surface_spalling"', finding_prompt)
        self.assertNotIn("Missing shingles are visible", finding_prompt)
        self.assertNotIn('"issue_code": "roof_cover_damage"', finding_prompt)

    def test_request_only_prompt_style_returns_plain_request(self) -> None:
        localized = task_schema.LocalizedIssue.from_payload(
            {
                "issue_code": "surface_spalling",
                "box": [0.1, 0.1, 0.4, 0.5],
                "source_detect_labels": ["surface spalling"],
            }
        )
        proposal_prompt = prompt_library.build_visible_issue_question_with_style(
            inspection_request=prompt_library.CANONICAL_QUERY_INSPECTION_REQUEST,
            asset_context="Concrete wall crop",
            prompt_style="request_only",
        )
        finding_prompt = prompt_library.build_finding_question_with_style(
            localized_issue=localized,
            inspection_request=prompt_library.CANONICAL_QUERY_INSPECTION_REQUEST,
            asset_context="Concrete wall crop",
            prompt_style="minimal",
        )
        self.assertEqual(proposal_prompt, prompt_library.CANONICAL_QUERY_INSPECTION_REQUEST)
        self.assertEqual(finding_prompt, prompt_library.CANONICAL_REGION_ONLY_QUERY_QUESTION)
        self.assertNotIn("surface_spalling", finding_prompt)
        self.assertNotIn("surface spalling", finding_prompt.lower())


class QueryCompactTests(unittest.TestCase):
    def test_compact_proposal_round_trip(self) -> None:
        proposals = [
            task_schema.IssueProposal(issue_code="roof_cover_damage", evidence="Missing shingles visible", confidence=1.0),
            task_schema.IssueProposal(issue_code="water_staining", evidence="Dark staining visible", confidence=1.0),
        ]
        text = query_compact.format_proposal_target(proposals)
        payload = query_compact.parse_proposal_answer(text)
        self.assertIsNotNone(payload)
        self.assertEqual(
            [item.issue_code for item in task_schema.normalize_issue_proposals(payload or {})],
            ["roof_cover_damage", "water_staining"],
        )

    def test_proposal_parse_recovers_free_text_issue_mentions(self) -> None:
        payload = query_compact.parse_proposal_answer(
            "The image shows cracks in the asphalt surface, broken asphalt pieces, and small debris scattered across the road."
        )
        self.assertIsNotNone(payload)
        self.assertEqual(
            [item.issue_code for item in task_schema.normalize_issue_proposals(payload or {})],
            ["crack_defect"],
        )

    def test_proposal_parse_supports_no_issue_free_text(self) -> None:
        payload = query_compact.parse_proposal_answer("No visible building or site issues.")
        self.assertEqual(task_schema.normalize_issue_proposals(payload or {}), [])

    def test_proposal_parse_prefers_specific_phrase_over_generic_family_match(self) -> None:
        payload = query_compact.parse_proposal_answer("A foundation crack is visible near the base of the wall.")
        self.assertIsNotNone(payload)
        self.assertEqual(
            [item.issue_code for item in task_schema.normalize_issue_proposals(payload or {})],
            ["foundation_settlement_sign"],
        )

    def test_compact_finding_parse_supports_single_line_format(self) -> None:
        payload = query_compact.parse_finding_answer(
            "surface_spalling | Surface Spalling | major | false | Loose concrete is visible | Repair the damaged concrete",
        )
        self.assertIsNotNone(payload)
        finding = task_schema.Finding.from_payload(
            {
                "finding_id": "f1",
                "box": {"x_min": 0.0, "y_min": 0.0, "x_max": 1.0, "y_max": 1.0},
                "cost_band": "unknown",
                "possible_compliance_issue": False,
                "compliance_note": "",
                "source_detect_labels": ["surface spalling"],
                "spatial_ref_index": 0,
                **dict((payload or {}).get("finding") or {}),
            }
        )
        self.assertEqual(finding.issue_code, "surface_spalling")
        self.assertEqual(finding.severity, "major")

    def test_issue_list_parse_repairs_loose_issue_labels_from_json_answers(self) -> None:
        payload, method = query_compact.parse_issue_list_answer_detailed(
            json.dumps(
                {
                    "issues": [
                        {"type": "crack_visible", "reasoning": "A visible crack runs through the wall surface."},
                        {"type": "material_abef", "reasoning": "Missing surface material is visible."},
                    ]
                }
            )
        )
        self.assertEqual(method, "json")
        self.assertIsNotNone(payload)
        self.assertEqual(
            [item.issue_code for item in task_schema.normalize_issue_proposals(payload or {})],
            ["crack_defect", "material_abscission"],
        )


class TrainerWrapperTests(unittest.TestCase):
    def test_detect_dataset_path_falls_back_to_local_smoke_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            module_root = Path(tmpdir) / "inspector_md"
            (module_root / "outputs" / "openrouter_refresh_smoke" / "detect").mkdir(parents=True)
            (module_root / "outputs" / "openrouter_refresh_smoke" / "detect" / "dataset_dict.json").write_text("{}", encoding="utf-8")
            (module_root / "outputs" / "openrouter_refresh_smoke" / "detect" / "metadata.json").write_text("{}", encoding="utf-8")

            resolved, warning = common.resolve_inspector_dataset_path(
                "inspector_md/outputs/inspector_detect_v999",
                task="detect",
                repo_root=Path(tmpdir),
                module_root=module_root,
            )

        self.assertTrue(str(resolved).endswith("openrouter_refresh_smoke/detect"))
        self.assertIn("using", warning)

    def test_custom_missing_dataset_path_does_not_auto_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            module_root = Path(tmpdir) / "inspector_md"
            (module_root / "outputs" / "smoke_merged_synth_v1" / "detect").mkdir(parents=True)
            (module_root / "outputs" / "smoke_merged_synth_v1" / "detect" / "dataset_dict.json").write_text("{}", encoding="utf-8")
            (module_root / "outputs" / "smoke_merged_synth_v1" / "detect" / "metadata.json").write_text("{}", encoding="utf-8")

            with self.assertRaises(FileNotFoundError):
                common.resolve_inspector_dataset_path(
                    "tmp/custom-detect-dataset",
                    task="detect",
                    repo_root=Path(tmpdir),
                    module_root=module_root,
                )

    def test_detect_wrapper_loads_path_backed_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = _write_image(Path(tmpdir) / "detect.png")
            row = {
                "image": str(image_path),
                "answer_boxes": [
                    {
                        "class_uid": "inspector_md:missing_shingle",
                        "class_name": "missing shingle",
                        "x_min": 0.1,
                        "y_min": 0.2,
                        "x_max": 0.4,
                        "y_max": 0.5,
                    }
                ],
                "source_dataset": "MBDD2025",
            }
            original_allowed = set(train_detect_mod._ALLOWED_CLASS_NAMES)
            train_detect_mod._ALLOWED_CLASS_NAMES = {"missing shingle"}
            try:
                sample = train_detect_mod._to_base_sample(row)
            finally:
                train_detect_mod._ALLOWED_CLASS_NAMES = original_allowed
        self.assertIsNotNone(sample)
        self.assertEqual(sample.image.size, (64, 48))
        self.assertEqual(len(sample.boxes), 1)
        self.assertEqual(sample.boxes[0].class_name, "missing shingle")

    def test_point_wrapper_loads_path_backed_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = _write_image(Path(tmpdir) / "point.png")
            row = {
                "image": str(image_path),
                "answer_boxes": [
                    {
                        "class_uid": "inspector_md:surface_spalling",
                        "class_name": "concrete spalling",
                        "x_min": 0.2,
                        "y_min": 0.2,
                        "x_max": 0.6,
                        "y_max": 0.8,
                    }
                ],
                "source_dataset": "CUBIT-Det",
            }
            original_allowed = set(train_point_mod._ALLOWED_CLASS_NAMES)
            train_point_mod._ALLOWED_CLASS_NAMES = {"concrete spalling"}
            try:
                sample = train_point_mod._to_base_sample(row)
            finally:
                train_point_mod._ALLOWED_CLASS_NAMES = original_allowed
        self.assertIsNotNone(sample)
        self.assertEqual(sample.image.size, (64, 48))
        self.assertEqual(len(sample.boxes), 1)
        self.assertEqual(sample.boxes[0].class_name, "concrete spalling")

    def test_detect_config_defaults_to_positive_selection_metric(self) -> None:
        args = train_detect_mod.parse_args([])
        self.assertEqual(args.selection_metric, "positive_f1")
        self.assertEqual(args.eval_min_positive_tasks, 16)
        self.assertEqual(args.sft_bootstrap_steps, 80)


class ApiKeyPoolTests(unittest.TestCase):
    def test_build_auth_headers_include_accept_and_browser_user_agent(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            headers = common.build_auth_headers("test-key")
        self.assertEqual(headers["X-Moondream-Auth"], "test-key")
        self.assertEqual(headers["Accept"], "application/json")
        self.assertIn("Mozilla/5.0", headers["User-Agent"])

    def test_build_auth_headers_support_authorization_override(self) -> None:
        with patch.dict(os.environ, {"MOONDREAM_AUTH_HEADER": "Authorization"}, clear=True):
            headers = common.build_auth_headers("test-key")
        self.assertEqual(headers["Authorization"], "Bearer test-key")
        self.assertNotIn("X-Moondream-Auth", headers)

    def test_default_staging_pool_requires_all_four_keys(self) -> None:
        env = {
            "CICID_GPUB_MOONDREAM_API_KEY_1": "key-1",
            "CICID_GPUB_MOONDREAM_API_KEY_2": "key-2",
            "CICID_GPUB_MOONDREAM_API_KEY_3": "key-3",
        }
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(ValueError):
                common.resolve_api_key_pool(api_key_env_vars=common.DEFAULT_API_KEY_ENV_VARS)

    def test_round_robin_pool_cycles_evenly(self) -> None:
        env = {
            "CICID_GPUB_MOONDREAM_API_KEY_1": "key-1",
            "CICID_GPUB_MOONDREAM_API_KEY_2": "key-2",
            "CICID_GPUB_MOONDREAM_API_KEY_3": "key-3",
            "CICID_GPUB_MOONDREAM_API_KEY_4": "key-4",
        }
        with patch.dict(os.environ, env, clear=True):
            pool = common.resolve_api_key_pool(api_key_env_vars=common.DEFAULT_API_KEY_ENV_VARS)
        seen = [pool.next_slot().env_var for _ in range(8)]
        self.assertEqual(seen, list(common.DEFAULT_API_KEY_ENV_VARS) * 2)


class ClientTests(unittest.TestCase):
    def test_request_payloads_include_spatial_refs_and_skill_shapes(self) -> None:
        calls: list[tuple[str, str, list[dict[str, object]]]] = []

        def fake_post_json(**kwargs):
            calls.append((str(kwargs["endpoint"]), str(kwargs["api_key"]), list(kwargs["payloads"])))
            endpoint = str(kwargs["endpoint"])
            if endpoint == "/query":
                return ({"answer": '{"finding":{"issue_code":"roof_cover_damage","title":"Roof Cover Damage","evidence":["missing shingles"],"severity":"major","recommended_action":"repair roof","insufficient_evidence":false}}'}, 12.0)
            if endpoint == "/detect":
                return ({"objects": [{"x_min": 0.1, "y_min": 0.1, "x_max": 0.4, "y_max": 0.4}]}, 8.0)
            if endpoint == "/point":
                return ({"points": [{"x": 0.3, "y": 0.4}]}, 7.0)
            raise AssertionError(endpoint)

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = _write_image(Path(tmpdir) / "sample.png")
            env = {
                "CICID_GPUB_MOONDREAM_API_KEY_1": "key-1",
                "CICID_GPUB_MOONDREAM_API_KEY_2": "key-2",
                "CICID_GPUB_MOONDREAM_API_KEY_3": "key-3",
                "CICID_GPUB_MOONDREAM_API_KEY_4": "key-4",
            }
            with patch.dict(os.environ, env, clear=True):
                pool = common.resolve_api_key_pool(api_key_env_vars=common.DEFAULT_API_KEY_ENV_VARS)
            client = moondream_client.MoondreamInspectorClient(api_key_pool=pool, post_json=fake_post_json)
            client.detect_boxes(model="moondream3-preview", image_path=image_path, detect_label="missing shingle")
            client.point_locations(model="moondream3-preview", image_path=image_path, point_label="missing shingle")
            localized = task_schema.LocalizedIssue(
                issue_code="roof_cover_damage",
                box=task_schema.Box.from_payload([0.1, 0.1, 0.4, 0.4]),
                evidence=["missing shingles visible"],
                source_detect_labels=["missing shingle"],
            )
            result = client.query_finding_fields(
                model="moondream3-preview",
                image_path=image_path,
                localized_issue=localized,
                inspection_request="inspect roof",
            )
            self.assertIn("finding", result.payload)
        endpoints = [item[0] for item in calls]
        api_keys = [item[1] for item in calls]
        self.assertEqual(endpoints, ["/detect", "/point", "/query"])
        self.assertEqual(api_keys, ["key-1", "key-2", "key-3"])
        query_payload = calls[-1][2][0]
        self.assertIn("spatial_refs", query_payload)
        self.assertEqual(query_payload["spatial_refs"], [[0.1, 0.1, 0.4, 0.4]])

    def test_openrouter_grader_normalizes_proposals_and_findings(self) -> None:
        calls: list[list[dict[str, object]]] = []

        def fake_call_api(**kwargs):
            calls.append(list(kwargs["messages"]))
            prompt_text = str(kwargs["messages"][-1]["content"])
            if "Expected proposals:" in prompt_text:
                return (
                    '{"satisfies_gt":"partial","grounded_visible_evidence":"yes","unsupported_claims":"minor","extra_issue_claims":"minor","verbosity":"acceptable","reason":"partial family match"}',
                    {"ok": True},
                    12.0,
                )
            if "Expected finding:" in prompt_text:
                return (
                    '{"satisfies_gt":"yes","grounded_visible_evidence":"yes","recommended_action_quality":"good","unsupported_claims":"none","verbosity":"concise","reason":"grounded and complete"}',
                    {"ok": True},
                    6.0,
                )
            if "Allowed issue_code:" in prompt_text:
                return (
                    '{"finding": {"issue_code": "surface_spalling", "title": "Surface Spalling", "evidence": ["loose concrete visible"], "severity": "major", "recommended_action": "repair the damaged concrete", "insufficient_evidence": false}, "notes": "normalized"}',
                    {"ok": True},
                    8.0,
                )
            return (
                '{"proposals": [{"issue_code": "water_staining", "evidence": "white staining visible"}], "notes": "normalized"}',
                {"ok": True},
                7.0,
            )

        grader = openrouter_grader.OpenRouterGrader(api_key="or-key", call_api_fn=fake_call_api)
        proposals = grader.normalize_proposals(
            inspection_request="Inspect this image for visible signs of staining.",
            asset_context="Concrete wall crop",
            answer_text="There appears to be white staining on the wall.",
        )
        self.assertEqual(proposals["proposals"][0]["issue_code"], "water_staining")
        grade = grader.grade_proposals(
            inspection_request="Inspect this image for visible signs of staining.",
            asset_context="Concrete wall crop",
            expected_proposals=[task_schema.IssueProposal(issue_code="efflorescence_deposit", evidence="white mineral deposit", confidence=1.0)],
            answer_text="There appears to be white staining on the wall.",
        )
        self.assertEqual(grade["satisfies_gt"], "partial")
        self.assertAlmostEqual(grade["score"], 0.35)
        localized = task_schema.LocalizedIssue.from_payload(
            {
                "issue_code": "surface_spalling",
                "box": [0.1, 0.1, 0.4, 0.5],
                "source_detect_labels": ["surface spalling"],
            }
        )
        finding = grader.normalize_finding(
            localized_issue=localized,
            inspection_request="Inspect the highlighted region.",
            asset_context="Concrete wall crop",
            answer_text="The highlighted area shows loose concrete and should be repaired.",
        )
        self.assertEqual(finding["finding"]["issue_code"], "surface_spalling")
        predicted_finding = task_schema.Finding.from_payload(
            {
                "finding_id": "pred-1",
                "issue_code": "surface_spalling",
                "title": "Surface Spalling",
                "box": {"x_min": 0.1, "y_min": 0.1, "x_max": 0.4, "y_max": 0.5},
                "evidence": ["loose concrete visible"],
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
        expected_finding = task_schema.Finding.from_payload(
            {
                "finding_id": "gt-1",
                "issue_code": "surface_spalling",
                "title": "Surface Spalling",
                "box": {"x_min": 0.12, "y_min": 0.12, "x_max": 0.42, "y_max": 0.52},
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
        finding_grade = grader.grade_finding(
            inspection_request="Inspect the highlighted region.",
            asset_context="Concrete wall crop",
            predicted_finding=predicted_finding,
            expected_finding=expected_finding,
            answer_text="The highlighted area shows loose concrete and should be repaired.",
        )
        self.assertEqual(finding_grade["satisfies_gt"], "yes")
        self.assertAlmostEqual(finding_grade["score"], 1.0)
        self.assertEqual(len(calls), 4)

    def test_openrouter_grader_fallback_scoring_is_safe(self) -> None:
        proposal = openrouter_grader.score_proposal_judgement({"satisfies_gt": "bogus"})
        self.assertEqual(proposal["satisfies_gt"], "no")
        self.assertEqual(proposal["score"], 0.0)
        finding = openrouter_grader.score_finding_judgement({"recommended_action_quality": "bogus"})
        self.assertEqual(finding["recommended_action_quality"], "poor")
        self.assertEqual(finding["score"], 0.0)


class PipelineTests(unittest.TestCase):
    class _FakeClient:
        def propose_issues(self, **kwargs):
            proposals = [
                task_schema.IssueProposal(issue_code="roof_cover_damage", evidence="missing shingles visible", confidence=0.9),
                task_schema.IssueProposal(issue_code="roof_cover_damage", evidence="roof defect near ridge", confidence=0.8),
            ]
            return proposals, moondream_client.QueryResult(payload={"proposals": [item.to_payload() for item in proposals]}, raw_response={"ok": True}, latency_ms=1.0)

        def detect_boxes(self, **kwargs):
            return [
                moondream_client.DetectAnnotation(x_min=0.1, y_min=0.1, x_max=0.4, y_max=0.4),
                moondream_client.DetectAnnotation(x_min=0.12, y_min=0.12, x_max=0.41, y_max=0.41),
            ]

        def query_finding_fields(self, **kwargs):
            return moondream_client.QueryResult(
                payload={
                    "finding": {
                        "issue_code": "roof_cover_damage",
                        "title": "Roof Cover Damage",
                        "evidence": ["Missing shingles are visible."],
                        "severity": "major",
                        "recommended_action": "Repair the damaged roof covering.",
                        "insufficient_evidence": False,
                    }
                },
                raw_response={"ok": True},
                latency_ms=2.0,
            )

    def test_pipeline_merges_overlapping_boxes_and_builds_report(self) -> None:
        pipeline = pipeline_mod.InspectorPipeline(
            client=self._FakeClient(),
            models=pipeline_mod.PipelineModels(detect_model="detect-ft", query_model="query-ft"),
            settings=pipeline_mod.PipelineSettings(iou_merge_threshold=0.5),
        )
        report = pipeline.run(
            {
                "image_path": "/tmp/example.png",
                "inspection_request": "Inspect the roof",
                "asset_context": "Exterior elevation",
            }
        )
        self.assertEqual(len(report.findings), 1)
        self.assertEqual(report.findings[0].issue_code, "roof_cover_damage")
        self.assertEqual(report.findings[0].cost_band, "very_high")
        self.assertIn("Roof Cover Damage", pipeline_mod.report_to_punch_list_text(report))

    def test_pipeline_handles_no_findings(self) -> None:
        class NoFindingClient(self._FakeClient):
            def propose_issues(self, **kwargs):
                return [], moondream_client.QueryResult(payload={"proposals": []}, raw_response={"ok": True}, latency_ms=1.0)

        pipeline = pipeline_mod.InspectorPipeline(
            client=NoFindingClient(),
            models=pipeline_mod.PipelineModels(detect_model="detect-ft", query_model="query-ft"),
        )
        report = pipeline.run({"image_path": "/tmp/example.png", "inspection_request": "Inspect", "asset_context": ""})
        self.assertEqual(report.findings, [])
        self.assertIn("No visible punch-list findings", pipeline_mod.report_to_punch_list_text(report))


class BuilderTests(unittest.TestCase):
    def test_issue_builder_emits_spatial_refs_and_coverage_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            image_path = _write_image(tmp / "sample.png")
            manifest_path = tmp / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    [
                        {
                            "row_id": "row1",
                            "split": "train",
                            "image_path": str(image_path),
                            "inspection_request": "Inspect the facade",
                            "asset_context": "Exterior wall with visible defects",
                            "hard_example": True,
                            "expected_proposals": [
                                {"issue_code": "crack_defect", "evidence": "A crack is visible on the wall."},
                                {"issue_code": "corrosion_rust", "evidence": "Rust staining is visible on the metal edge."},
                            ],
                            "expected_findings": [
                                {
                                    "issue_code": "crack_defect",
                                    "title": "Crack Defect",
                                    "box": {"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.6},
                                    "evidence": ["A crack is visible on the wall."],
                                    "severity": "moderate",
                                    "recommended_action": "Inspect the crack pattern and repair the cracked surface where appropriate.",
                                    "cost_band": "medium",
                                    "possible_compliance_issue": True,
                                    "insufficient_evidence": False,
                                    "compliance_note": "possible issue",
                                    "source_detect_labels": ["surface crack"],
                                    "spatial_ref_index": 1
                                },
                                {
                                    "issue_code": "corrosion_rust",
                                    "title": "Corrosion or Rust",
                                    "box": {"x_min": 0.45, "y_min": 0.25, "x_max": 0.7, "y_max": 0.7},
                                    "evidence": ["Rust staining is visible on the metal edge."],
                                    "severity": "moderate",
                                    "recommended_action": "Remove corrosion where feasible and repair or replace compromised metal components.",
                                    "cost_band": "medium",
                                    "possible_compliance_issue": True,
                                    "insufficient_evidence": False,
                                    "compliance_note": "possible issue",
                                    "source_detect_labels": ["rusted metal"],
                                    "spatial_ref_index": 0
                                }
                            ]
                        }
                    ]
                ),
                encoding="utf-8",
            )
            args = build_mod.parse_args(
                [
                    "--source-manifest",
                    str(manifest_path),
                    "--output-root",
                    str(tmp / "outputs"),
                    "--detect-output-dir",
                    str(tmp / "outputs" / "detect"),
                    "--point-output-dir",
                    str(tmp / "outputs" / "point"),
                    "--query-output-dir",
                    str(tmp / "outputs" / "issues"),
                    "--query-text-refresh-mode",
                    "template_only",
                ]
            )
            summary = build_mod.build_dataset(args)
            metadata = json.loads((Path(summary["query_output_dir"]) / "metadata.json").read_text(encoding="utf-8"))
            issue_rows = common.load_jsonl(Path(summary["query_output_dir"]) / "jsonl" / "train.jsonl")
            self.assertEqual(len(issue_rows), 1)
            spatial_refs = json.loads(issue_rows[0]["spatial_refs_json"])
            self.assertEqual(len(spatial_refs), 2)
            self.assertEqual(spatial_refs[0]["x_min"], 0.45)
            self.assertEqual(spatial_refs[1]["x_min"], 0.1)
            self.assertEqual(metadata["issue_box_counts"]["crack_defect"], 1)
            self.assertEqual(metadata["issue_box_counts"]["corrosion_rust"], 1)
            self.assertEqual(metadata["query_spatial_refs_nonempty_count"], 1)
            self.assertIn("roof_cover_damage", metadata["missing_issue_codes"])

    def test_builder_writes_expected_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            image_path = _write_image(tmp / "sample.png")
            manifest_path = tmp / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    [
                        {
                            "row_id": "row1",
                            "split": "train",
                            "image_path": str(image_path),
                            "inspection_request": "Inspect the roof",
                            "asset_context": "Exterior",
                            "hard_example": True,
                            "expected_proposals": [{"issue_code": "roof damage", "evidence": "missing shingles visible"}],
                            "expected_findings": [
                                {
                                    "issue_code": "roof damage",
                                    "title": "Roof Cover Damage",
                                    "box": {"x_min": 0.1, "y_min": 0.1, "x_max": 0.4, "y_max": 0.4},
                                    "evidence": ["missing shingles visible"],
                                    "severity": "major",
                                    "recommended_action": "repair roof",
                                    "cost_band": "high",
                                    "possible_compliance_issue": True,
                                    "insufficient_evidence": False,
                                    "compliance_note": "possible issue",
                                    "source_detect_labels": ["missing shingle"],
                                    "spatial_ref_index": 0
                                }
                            ]
                        }
                    ]
                ),
                encoding="utf-8",
            )
            args = build_mod.parse_args(
                [
                    "--source-manifest",
                    str(manifest_path),
                    "--output-root",
                    str(tmp / "outputs"),
                    "--detect-output-dir",
                    str(tmp / "outputs" / "detect"),
                    "--point-output-dir",
                    str(tmp / "outputs" / "point"),
                    "--query-proposal-output-dir",
                    str(tmp / "outputs" / "proposal"),
                    "--query-finding-output-dir",
                    str(tmp / "outputs" / "finding"),
                    "--query-reasoning-output-dir",
                    str(tmp / "outputs" / "reasoning"),
                    "--query-text-refresh-mode",
                    "template_only",
                ]
            )
            summary = build_mod.build_dataset(args)
            self.assertTrue(Path(summary["detect_output_dir"]).exists())
            self.assertTrue((Path(summary["query_finding_output_dir"]) / "jsonl" / "train.jsonl").exists())
            self.assertTrue((Path(summary["query_reasoning_output_dir"]) / "jsonl" / "train.jsonl").exists())
            metadata = json.loads((Path(summary["query_finding_output_dir"]) / "metadata.json").read_text(encoding="utf-8"))
            proposal_rows = common.load_jsonl(Path(summary["query_proposal_output_dir"]) / "jsonl" / "train.jsonl")
            finding_rows = common.load_jsonl(Path(summary["query_finding_output_dir"]) / "jsonl" / "train.jsonl")
            normalized_manifest = json.loads((tmp / "outputs" / "source_manifest.normalized.json").read_text(encoding="utf-8"))
            self.assertIn("detect_class_catalog", metadata)
            self.assertEqual(metadata["query_target_format"], "compact_text")
            self.assertIn("query_proposal_split_counts", metadata)
            self.assertEqual(normalized_manifest[0]["expected_proposals"][0]["issue_code"], "roof_cover_damage")
            self.assertEqual(normalized_manifest[0]["expected_findings"][0]["finding_id"], "row1_finding_001")
            self.assertEqual(proposal_rows[0]["inspection_request"], prompt_library.CANONICAL_QUERY_INSPECTION_REQUEST)
            self.assertEqual(proposal_rows[0]["question"], prompt_library.CANONICAL_QUERY_INSPECTION_REQUEST)
            self.assertIn("roof_cover_damage |", proposal_rows[0]["target_text"])
            self.assertEqual(finding_rows[0]["inspection_request"], prompt_library.CANONICAL_QUERY_INSPECTION_REQUEST)
            self.assertEqual(finding_rows[0]["question"], prompt_library.CANONICAL_REGION_ONLY_QUERY_QUESTION)
            self.assertIn("|", finding_rows[0]["target_text"])

    def test_builder_can_refresh_query_text_with_openrouter(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            image_path = _write_image(tmp / "sample.png")
            manifest_path = tmp / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    [
                        {
                            "row_id": "row1",
                            "split": "train",
                            "image_path": str(image_path),
                            "inspection_request": "Inspect the roof",
                            "asset_context": "Exterior",
                            "hard_example": True,
                            "expected_proposals": [{"issue_code": "roof_cover_damage", "evidence": "missing shingles visible"}],
                            "expected_findings": [
                                {
                                    "issue_code": "roof_cover_damage",
                                    "title": "Roof Cover Damage",
                                    "box": {"x_min": 0.1, "y_min": 0.1, "x_max": 0.4, "y_max": 0.4},
                                    "evidence": ["missing shingles visible"],
                                    "severity": "major",
                                    "recommended_action": "repair roof",
                                    "cost_band": "high",
                                    "possible_compliance_issue": True,
                                    "insufficient_evidence": False,
                                    "compliance_note": "possible issue",
                                    "source_detect_labels": ["missing shingle"],
                                    "spatial_ref_index": 0
                                }
                            ]
                        }
                    ]
                ),
                encoding="utf-8",
            )
            args = build_mod.parse_args(
                [
                    "--source-manifest",
                    str(manifest_path),
                    "--output-root",
                    str(tmp / "outputs"),
                    "--detect-output-dir",
                    str(tmp / "outputs" / "detect"),
                    "--point-output-dir",
                    str(tmp / "outputs" / "point"),
                    "--query-proposal-output-dir",
                    str(tmp / "outputs" / "proposal"),
                    "--query-finding-output-dir",
                    str(tmp / "outputs" / "finding"),
                    "--query-reasoning-output-dir",
                    str(tmp / "outputs" / "reasoning"),
                    "--query-text-refresh-mode",
                    "openrouter",
                    "--query-teacher-model-id",
                    "openai/gpt-4.1-mini",
                    "--query-teacher-api-key",
                    "test-openrouter-key",
                    "--query-text-cache-jsonl",
                    str(tmp / "outputs" / "query_text_cache.jsonl"),
                ]
            )
            summary = build_mod.build_dataset(args, call_openrouter_fn=_fake_query_refresh_call)
            metadata = json.loads((Path(summary["query_finding_output_dir"]) / "metadata.json").read_text(encoding="utf-8"))
            proposal_rows = common.load_jsonl(Path(summary["query_proposal_output_dir"]) / "jsonl" / "train.jsonl")
            finding_rows = common.load_jsonl(Path(summary["query_finding_output_dir"]) / "jsonl" / "train.jsonl")
            self.assertEqual(metadata["query_text_refresh_mode"], "openrouter")
            self.assertEqual(metadata["query_refreshed_record_count"], 1)
            self.assertEqual(metadata["query_cache_hits"], 0)
            self.assertTrue((tmp / "outputs" / "query_text_cache.jsonl").exists())
            self.assertEqual(proposal_rows[0]["query_text_refresh_mode"], "openrouter")
            self.assertEqual(finding_rows[0]["query_text_refresh_mode"], "openrouter")
            self.assertEqual(proposal_rows[0]["inspection_request"], prompt_library.CANONICAL_QUERY_INSPECTION_REQUEST)
            self.assertEqual(finding_rows[0]["inspection_request"], prompt_library.CANONICAL_QUERY_INSPECTION_REQUEST)
            self.assertEqual(proposal_rows[0]["question"], prompt_library.CANONICAL_QUERY_INSPECTION_REQUEST)
            self.assertEqual(finding_rows[0]["question"], prompt_library.CANONICAL_REGION_ONLY_QUERY_QUESTION)
            self.assertNotIn("roof cover damage", proposal_rows[0]["question"].lower())
            self.assertNotIn("roof cover damage", finding_rows[0]["question"].lower())
            self.assertIn("roof_cover_damage | Missing shingles are visible", proposal_rows[0]["target_text"])

    def test_query_refresh_normalization_ignores_prompt_override_and_cache_key_uses_prompt_version(self) -> None:
        record = {
            "row_id": "row-1",
            "inspection_request": "Inspect the roof",
            "asset_context": "Exterior",
            "expected_proposals": [{"issue_code": "roof_cover_damage", "evidence": "missing shingles visible"}],
            "expected_findings": [
                {
                    "issue_code": "roof_cover_damage",
                    "title": "Roof Cover Damage",
                    "box": {"x_min": 0.1, "y_min": 0.1, "x_max": 0.4, "y_max": 0.4},
                    "evidence": ["missing shingles visible"],
                    "severity": "major",
                    "recommended_action": "repair roof",
                    "cost_band": "high",
                    "possible_compliance_issue": True,
                    "insufficient_evidence": False,
                    "compliance_note": "possible issue",
                    "source_detect_labels": ["missing shingle"],
                    "spatial_ref_index": 0,
                }
            ],
        }
        normalized = build_mod._normalize_query_refresh_bundle(
            record,
            {
                "inspection_request": "Inspect this image for visible signs of roof cover damage.",
                "asset_context": "Refreshed exterior context.",
                "reasoning_text": "Use visible damage only.",
                "proposals": [{"issue_code": "roof_cover_damage", "evidence": "Missing shingles are visible."}],
                "findings": [],
            },
        )
        self.assertEqual(normalized["inspection_request"], prompt_library.CANONICAL_QUERY_INSPECTION_REQUEST)
        self.assertEqual(normalized["asset_context"], "Refreshed exterior context.")
        key_a = build_mod._query_cache_key_for_record(record, model_id="openai/gpt-4.1-mini")
        key_b = build_mod._query_cache_key_for_record({**record, "inspection_request": "Inspect the wall"}, model_id="openai/gpt-4.1-mini")
        self.assertEqual(key_a, key_b)
        with patch.object(build_mod, "QUERY_REFRESH_PROMPT_VERSION", "inspector_md_query_refresh_v999"):
            self.assertNotEqual(
                key_a,
                build_mod._query_cache_key_for_record(record, model_id="openai/gpt-4.1-mini"),
            )

    def test_builder_supports_query_refresh_only_shards(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            image_path_a = _write_image(tmp / "sample-a.png")
            image_path_b = _write_image(tmp / "sample-b.png", color="gray")
            manifest_path = tmp / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    [
                        {
                            "row_id": "row-a",
                            "split": "train",
                            "image_path": str(image_path_a),
                            "inspection_request": "Inspect the roof",
                            "asset_context": "Exterior",
                            "hard_example": True,
                            "expected_proposals": [{"issue_code": "roof_cover_damage", "evidence": "missing shingles visible"}],
                            "expected_findings": [
                                {
                                    "issue_code": "roof_cover_damage",
                                    "title": "Roof Cover Damage",
                                    "box": {"x_min": 0.1, "y_min": 0.1, "x_max": 0.4, "y_max": 0.4},
                                    "evidence": ["missing shingles visible"],
                                    "severity": "major",
                                    "recommended_action": "repair roof",
                                    "cost_band": "high",
                                    "possible_compliance_issue": True,
                                    "insufficient_evidence": False,
                                    "compliance_note": "possible issue",
                                    "source_detect_labels": ["missing shingle"],
                                    "spatial_ref_index": 0
                                }
                            ]
                        },
                        {
                            "row_id": "row-b",
                            "split": "train",
                            "image_path": str(image_path_b),
                            "inspection_request": "Inspect the path",
                            "asset_context": "Exterior",
                            "hard_example": False,
                            "expected_proposals": [{"issue_code": "blocked_egress_or_access", "evidence": "route is blocked"}],
                            "expected_findings": [
                                {
                                    "issue_code": "blocked_egress_or_access",
                                    "title": "Blocked Egress or Access",
                                    "box": {"x_min": 0.2, "y_min": 0.2, "x_max": 0.5, "y_max": 0.6},
                                    "evidence": ["route is blocked"],
                                    "severity": "moderate",
                                    "recommended_action": "clear path",
                                    "cost_band": "medium",
                                    "possible_compliance_issue": True,
                                    "insufficient_evidence": False,
                                    "compliance_note": "possible issue",
                                    "source_detect_labels": ["blocked access route"],
                                    "spatial_ref_index": 0
                                }
                            ]
                        }
                    ]
                ),
                encoding="utf-8",
            )
            args = build_mod.parse_args(
                [
                    "--source-manifest",
                    str(manifest_path),
                    "--output-root",
                    str(tmp / "outputs"),
                    "--query-text-refresh-mode",
                    "openrouter",
                    "--query-teacher-model-id",
                    "openai/gpt-4.1-mini",
                    "--query-teacher-api-key",
                    "test-openrouter-key",
                    "--query-text-cache-jsonl",
                    str(tmp / "outputs" / "query_text_cache.shard.jsonl"),
                    "--query-refresh-only",
                    "--query-refresh-num-shards",
                    "2",
                    "--query-refresh-shard-index",
                    "0",
                    "--query-refresh-max-concurrency",
                    "2",
                ]
            )
            summary = build_mod.build_dataset(args, call_openrouter_fn=_fake_query_refresh_call)
            self.assertTrue(bool(summary["query_refresh_only"]))
            self.assertEqual(summary["query_refresh_num_shards"], 2)
            self.assertEqual(summary["query_refresh_shard_index"], 0)
            self.assertEqual(summary["query_refresh_max_concurrency"], 2)
            self.assertGreaterEqual(int(summary["selected_record_count"]), 1)
            self.assertLessEqual(int(summary["selected_record_count"]), 2)
            self.assertTrue((tmp / "outputs" / "query_text_cache.shard.jsonl").exists())
            self.assertTrue((tmp / "outputs" / "query_refresh_summary.shard_000_of_002.json").exists())

    def test_builder_filters_reasoning_subset_to_hard_examples(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            image_path_a = _write_image(tmp / "sample-a.png")
            image_path_b = _write_image(tmp / "sample-b.png", color="gray")
            manifest_path = tmp / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    [
                        {
                            "row_id": "row-hard",
                            "split": "train",
                            "image_path": str(image_path_a),
                            "inspection_request": "Inspect the roof",
                            "asset_context": "Exterior",
                            "hard_example": True,
                            "expected_proposals": [{"issue_code": "roof_cover_damage", "evidence": "missing shingles visible"}],
                            "expected_findings": [
                                {
                                    "issue_code": "roof_cover_damage",
                                    "title": "Roof Cover Damage",
                                    "box": {"x_min": 0.1, "y_min": 0.1, "x_max": 0.4, "y_max": 0.4},
                                    "evidence": ["missing shingles visible"],
                                    "severity": "major",
                                    "recommended_action": "repair roof",
                                    "cost_band": "high",
                                    "possible_compliance_issue": True,
                                    "insufficient_evidence": False,
                                    "compliance_note": "possible issue",
                                    "source_detect_labels": ["missing shingle"],
                                    "spatial_ref_index": 0
                                }
                            ]
                        },
                        {
                            "row_id": "row-easy",
                            "split": "train",
                            "image_path": str(image_path_b),
                            "inspection_request": "Inspect the path",
                            "asset_context": "Exterior",
                            "hard_example": False,
                            "expected_proposals": [{"issue_code": "blocked_egress_or_access", "evidence": "route is blocked"}],
                            "expected_findings": [
                                {
                                    "issue_code": "blocked_egress_or_access",
                                    "title": "Blocked Egress or Access",
                                    "box": {"x_min": 0.2, "y_min": 0.2, "x_max": 0.5, "y_max": 0.6},
                                    "evidence": ["route is blocked"],
                                    "severity": "moderate",
                                    "recommended_action": "clear path",
                                    "cost_band": "medium",
                                    "possible_compliance_issue": True,
                                    "insufficient_evidence": False,
                                    "compliance_note": "possible issue",
                                    "source_detect_labels": ["blocked access route"],
                                    "spatial_ref_index": 0
                                }
                            ]
                        }
                    ]
                ),
                encoding="utf-8",
            )
            args = build_mod.parse_args(
                [
                    "--source-manifest",
                    str(manifest_path),
                    "--output-root",
                    str(tmp / "outputs"),
                    "--detect-output-dir",
                    str(tmp / "outputs" / "detect"),
                    "--point-output-dir",
                    str(tmp / "outputs" / "point"),
                    "--query-proposal-output-dir",
                    str(tmp / "outputs" / "proposal"),
                    "--query-finding-output-dir",
                    str(tmp / "outputs" / "finding"),
                    "--query-reasoning-output-dir",
                    str(tmp / "outputs" / "reasoning"),
                    "--query-text-refresh-mode",
                    "template_only",
                ]
            )
            build_mod.build_dataset(args)
            reasoning_rows = common.load_jsonl(tmp / "outputs" / "reasoning" / "jsonl" / "train.jsonl")
            self.assertEqual(len(reasoning_rows), 1)
            self.assertEqual(reasoning_rows[0]["row_id"], "row-hard_finding_001")

    def test_builder_excludes_codebrim_from_detect_and_point_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            image_path = _write_image(tmp / "codebrim.png")
            manifest_path = tmp / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    [
                        {
                            "row_id": "codebrim-row",
                            "split": "train",
                            "image_path": str(image_path),
                            "inspection_request": "Inspect this defect crop",
                            "asset_context": "Concrete crop",
                            "hard_example": True,
                            "source_dataset": "CODEBRIM",
                            "source_metadata": {"category": "defects"},
                            "expected_proposals": [{"issue_code": "efflorescence_deposit", "evidence": "white deposit visible"}],
                            "expected_findings": [
                                {
                                    "issue_code": "efflorescence_deposit",
                                    "title": "Efflorescence Deposit",
                                    "box": {"x_min": 0.0, "y_min": 0.0, "x_max": 1.0, "y_max": 1.0},
                                    "evidence": ["white deposit visible"],
                                    "severity": "moderate",
                                    "recommended_action": "inspect moisture source",
                                    "cost_band": "low",
                                    "possible_compliance_issue": True,
                                    "insufficient_evidence": False,
                                    "compliance_note": "possible issue",
                                    "source_detect_labels": ["efflorescence deposit"],
                                    "spatial_ref_index": 0
                                }
                            ]
                        }
                    ]
                ),
                encoding="utf-8",
            )
            args = build_mod.parse_args(
                [
                    "--source-manifest",
                    str(manifest_path),
                    "--output-root",
                    str(tmp / "outputs"),
                    "--detect-output-dir",
                    str(tmp / "outputs" / "detect"),
                    "--point-output-dir",
                    str(tmp / "outputs" / "point"),
                    "--query-proposal-output-dir",
                    str(tmp / "outputs" / "proposal"),
                    "--query-finding-output-dir",
                    str(tmp / "outputs" / "finding"),
                    "--query-reasoning-output-dir",
                    str(tmp / "outputs" / "reasoning"),
                    "--query-text-refresh-mode",
                    "template_only",
                ]
            )
            summary = build_mod.build_dataset(args)
            metadata = json.loads((Path(summary["detect_output_dir"]) / "metadata.json").read_text(encoding="utf-8"))
            proposal_rows = common.load_jsonl(tmp / "outputs" / "proposal" / "jsonl" / "train.jsonl")
            self.assertEqual(metadata["detect_exclude_source_datasets"], ["CODEBRIM"])
            self.assertEqual(metadata["detect_split_counts"]["train"], 0)
            self.assertEqual(len(proposal_rows), 1)

    def test_builder_rejects_invalid_manifest_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            image_path = _write_image(tmp / "sample.png")
            manifest_path = tmp / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    [
                        {
                            "row_id": "row1",
                            "split": "train",
                            "image_path": str(image_path),
                            "inspection_request": "Inspect the roof",
                            "asset_context": "Exterior",
                            "hard_example": True,
                            "expected_proposals": [{"issue_code": "roof_cover_damage", "evidence": "missing shingles visible"}],
                            "expected_findings": [
                                {
                                    "issue_code": "roof_cover_damage",
                                    "title": "Roof Cover Damage",
                                    "box": {"x_min": 0.1, "y_min": 0.1, "x_max": 0.4, "y_max": 0.4},
                                    "evidence": ["missing shingles visible"],
                                    "severity": "major",
                                    "recommended_action": "repair roof",
                                    "cost_band": "high",
                                    "possible_compliance_issue": True,
                                    "insufficient_evidence": False,
                                    "compliance_note": "possible issue",
                                    "source_detect_labels": ["damage"],
                                    "spatial_ref_index": 0
                                }
                            ]
                        }
                    ]
                ),
                encoding="utf-8",
            )
            args = build_mod.parse_args(
                [
                    "--source-manifest",
                    str(manifest_path),
                    "--output-root",
                    str(tmp / "outputs"),
                    "--query-text-refresh-mode",
                    "template_only",
                ]
            )
            with self.assertRaises(ValueError):
                build_mod.build_dataset(args)


class ConfigAndSweepTests(unittest.TestCase):
    def test_positive_fraction_scheduler_tracks_three_to_one_mix(self) -> None:
        pos = 0
        neg = 0
        decisions: list[str] = []
        for _ in range(8):
            choose_positive = train_detect_mod._base._should_sample_positive_next(
                consumed_positive=pos,
                consumed_negative=neg,
                target_positive_fraction=0.75,
            )
            decisions.append("P" if choose_positive else "N")
            if choose_positive:
                pos += 1
            else:
                neg += 1
        self.assertEqual(decisions, ["P", "P", "P", "N", "P", "P", "P", "N"])

    def test_effective_num_steps_resolves_full_train_pass(self) -> None:
        steps, applied = train_detect_mod._base._resolve_effective_num_steps(
            configured_num_steps=120,
            train_row_passes=1.0,
            total_train_rows=11394,
            batch_size=4,
        )
        self.assertTrue(applied)
        self.assertEqual(steps, 2849)

    def test_train_metric_helper_marks_empty_batches_undefined(self) -> None:
        precision, recall, f1 = train_detect_mod._base._binary_metrics_from_counts(0, 0, 0)
        self.assertTrue(math.isnan(precision))
        self.assertTrue(math.isnan(recall))
        self.assertTrue(math.isnan(f1))

        precision, recall, f1 = train_detect_mod._base._binary_metrics_from_counts(3, 1, 2)
        self.assertAlmostEqual(precision, 0.75)
        self.assertAlmostEqual(recall, 0.6)
        self.assertAlmostEqual(f1, 2.0 * 3.0 / 9.0)

    def test_inspector_trainers_load_dataset_class_catalog_by_default(self) -> None:
        detect_args = train_detect_mod.parse_args(
            [
                "--dataset-path",
                str(REPO_ROOT / "inspector_md" / "outputs" / "inspector_detect_v1"),
            ]
        )
        point_args = train_point_mod.parse_args(
            [
                "--dataset-path",
                str(REPO_ROOT / "inspector_md" / "outputs" / "inspector_point_v1"),
            ]
        )
        self.assertEqual(detect_args.class_names_file, "")
        self.assertEqual(point_args.class_names_file, "")
        self.assertEqual(detect_args.train_row_passes, 1.0)
        self.assertEqual(point_args.train_row_passes, 1.0)
        self.assertEqual(detect_args.target_positive_fraction, 0.75)
        self.assertEqual(point_args.target_positive_fraction, 0.75)
        self.assertEqual(detect_args.neg_prompts_per_nonempty, 1)
        self.assertEqual(point_args.neg_prompts_per_nonempty, 1)
        self.assertIn("surface crack", train_detect_mod._ALLOWED_CLASS_NAMES)
        self.assertIn("water leakage", train_detect_mod._ALLOWED_CLASS_NAMES)
        self.assertIn("material abscission", train_point_mod._ALLOWED_CLASS_NAMES)
        self.assertIn("surface bulge", train_point_mod._ALLOWED_CLASS_NAMES)

    def test_all_configs_parse(self) -> None:
        config_root = REPO_ROOT / "inspector_md" / "configs"
        for config_path in sorted(config_root.rglob("*.json")):
            with self.subTest(config=str(config_path.relative_to(REPO_ROOT))):
                name = config_path.name
                if name == "inspector_detect_class_catalog.json":
                    payload = json.loads(config_path.read_text(encoding="utf-8"))
                    self.assertIn("class_catalog", payload)
                elif name == "build_merged_synth_dataset_default.json":
                    args = merge_mod.parse_args(["--config", str(config_path)])
                    self.assertEqual(args.stage, "normalize")
                    self.assertTrue(str(args.output_dir).endswith("merged_synth_v1"))
                elif name == "generate_class_samples_viz_default.json":
                    args = viz_mod.parse_args(["--config", str(config_path)])
                    self.assertTrue(str(args.source_manifest).endswith("synthetic_manifest.json"))
                    self.assertTrue(str(args.output_dir).endswith("class_samples_viz_merged_synth_v1"))
                elif name == "run_baseline_inspector_eval_default.json":
                    args = baseline_eval_mod.parse_args(["--config", str(config_path)])
                    self.assertEqual(args.base_url, common.DEFAULT_BASE_URL)
                    self.assertEqual(args.api_key_env_vars, list(common.DEFAULT_API_KEY_ENV_VARS))
                    self.assertEqual(args.detect_exclude_source_datasets, ["CODEBRIM"])
                    self.assertTrue(str(args.dataset_manifest).endswith("smoke_manifest.json"))
                elif name == "train_inspector_detect_default.json":
                    args = train_detect_mod.parse_args(["--config", str(config_path)])
                    self.assertEqual(args.sft_bootstrap_steps, 0)
                elif name == "train_inspector_detect_sft_default.json":
                    args = train_detect_mod.parse_args(["--config", str(config_path)])
                    self.assertEqual(args.sft_bootstrap_steps, 0)
                elif name == "train_inspector_point_default.json":
                    args = train_point_mod.parse_args(["--config", str(config_path)])
                    self.assertEqual(args.sft_bootstrap_steps, 0)
                elif name == "train_inspector_point_sft_default.json":
                    args = train_point_mod.parse_args(["--config", str(config_path)])
                    self.assertEqual(args.sft_bootstrap_steps, 0)
                elif name == "check_inspector_finetune_readiness_default.json":
                    args = readiness_mod.parse_args(["--config", str(config_path)])
                    self.assertTrue(bool(args.require_query_text_refresh))
                    self.assertTrue(bool(args.verify_openrouter_judge))
                    self.assertTrue(str(args.query_finding_dataset_dir).endswith("inspector_query_finding_v1"))
                elif name == "benchmark_inspector_detect_default.json":
                    args = bench_detect_mod.parse_args(["--config", str(config_path)])
                    self.assertEqual(args.base_url, common.DEFAULT_BASE_URL)
                    self.assertEqual(args.api_key_env_vars, list(common.DEFAULT_API_KEY_ENV_VARS))
                    self.assertEqual(args.detect_exclude_source_datasets, ["CODEBRIM"])
                    self.assertEqual(args.coarse_iou_threshold, 0.1)
                    self.assertTrue(str(args.output_json).endswith("inspector_detect.metrics.json"))
                elif name == "benchmark_inspector_query_default.json":
                    args = bench_query_mod.parse_args(["--config", str(config_path)])
                    self.assertEqual(args.base_url, common.DEFAULT_BASE_URL)
                    self.assertEqual(args.api_key_env_vars, list(common.DEFAULT_API_KEY_ENV_VARS))
                    self.assertEqual(args.answer_parse_mode, "strict_raw")
                    self.assertTrue(str(args.dataset_dir).endswith("inspector_query_issues_v2"))
                    self.assertTrue(str(args.output_json).endswith("inspector_query.metrics.json"))
                elif name == "run_inspector_dataset_refresh_batches_default.json":
                    args = refresh_batches_mod.parse_args(["--config", str(config_path)])
                    self.assertEqual(args.num_shards, 8)
                    self.assertEqual(args.max_parallel_jobs, 4)
                    self.assertEqual(args.per_worker_concurrency, 1)
                    self.assertEqual(args.max_shard_attempts, 2)
                    self.assertTrue(bool(args.resume))
                    self.assertTrue(bool(args.finalize))
                    self.assertTrue(str(args.builder_config).endswith("build_inspector_dataset_default.json"))
                elif name.startswith("run_inspector_sweep") or name.startswith("run_inspector_detect_only_sweep"):
                    args = sweep_mod.parse_args(["--config", str(config_path)])
                    self.assertEqual(args.base_url, common.DEFAULT_BASE_URL)
                    self.assertEqual(args.api_key_env_vars, list(common.DEFAULT_API_KEY_ENV_VARS))
                    self.assertEqual(args.max_parallel_jobs, 4)
                    self.assertEqual(args.query_max_parallel_jobs, 1)
                    if name == "run_inspector_sweep_default.json":
                        self.assertIn("detect_sft", args.families)
                        self.assertIn("query_reasoning_rl", args.families)
                    else:
                        self.assertEqual(args.families, ["detect_sft", "detect_rl"])
                        self.assertEqual(args.ranks, [32, 48, 64])
                        self.assertEqual(args.lrs, [5e-05, 2e-05])
                elif name == "publish_inspector_md_hf_dataset_default.json":
                    from inspector_md import publish_inspector_md_hf_dataset as publish_hf_mod

                    args = publish_hf_mod.parse_args(["--config", str(config_path)])
                    self.assertFalse(bool(args.push))
                    self.assertTrue(bool(args.private))
                    self.assertTrue(str(args.detect_input_dir).endswith("inspector_detect_v1"))
                    self.assertTrue(str(args.query_finding_input_dir).endswith("inspector_query_finding_v1"))
                elif name.startswith("build_"):
                    args = build_mod.parse_args(["--config", str(config_path)])
                    self.assertEqual(args.detect_exclude_source_datasets, ["CODEBRIM"])
                    self.assertTrue(str(args.detect_output_dir).endswith("inspector_detect_v1"))
                    self.assertEqual(args.query_text_refresh_mode, "openrouter")
                    self.assertTrue(str(args.source_manifest).endswith("synthetic_manifest.json"))
                elif name.startswith("run_"):
                    args = run_pipeline_mod.parse_args(["--config", str(config_path), "--image-path", "/tmp/example.png"])
                    self.assertEqual(args.base_url, common.DEFAULT_BASE_URL)
                    self.assertEqual(args.api_key_env_vars, list(common.DEFAULT_API_KEY_ENV_VARS))
                    self.assertTrue(str(args.output_json).endswith(".json"))
                elif name.startswith("benchmark_"):
                    args = benchmark_mod.parse_args(["--config", str(config_path)])
                    self.assertEqual(args.base_url, common.DEFAULT_BASE_URL)
                    self.assertEqual(args.api_key_env_vars, list(common.DEFAULT_API_KEY_ENV_VARS))
                    self.assertEqual(args.detect_exclude_source_datasets, ["CODEBRIM"])
                    self.assertEqual(args.coarse_iou_threshold, 0.1)
                    self.assertTrue(str(args.output_json).endswith(".json"))
                elif "query" in name:
                    args = train_query_mod.parse_args(["--config", str(config_path)])
                    self.assertEqual(args.base_url, common.DEFAULT_BASE_URL)
                    self.assertEqual(args.api_key_env_vars, list(common.DEFAULT_API_KEY_ENV_VARS))
                    self.assertIn(args.batch_size, {4, 8})
                    self.assertEqual(args.num_rollouts, 8)
                    self.assertEqual(args.sft_max_tokens, 128)
                    self.assertEqual(args.eval_max_tokens, 128)
                    self.assertTrue(bool(args.async_checkpoint_eval))
                    self.assertTrue(str(args.wandb_project).startswith("moondream-inspector-query"))
                elif "detect" in name:
                    args = train_detect_mod.parse_args(["--config", str(config_path)])
                    self.assertEqual(args.skill, "detect")
                    self.assertEqual(args.selection_metric, "positive_f1")
                    self.assertEqual(args.eval_min_positive_tasks, 16)
                elif "point" in name:
                    args = train_point_mod.parse_args(["--config", str(config_path)])
                    self.assertEqual(args.skill, "point")
                else:
                    self.fail(f"Unhandled config path: {config_path}")

    def test_sweep_manifest_enforces_policy(self) -> None:
        args = sweep_mod.parse_args(["--dry-run"])
        runs = sweep_mod.build_sweep_runs(args)
        self.assertTrue(runs)
        self.assertEqual(args.base_url, common.DEFAULT_BASE_URL)
        self.assertEqual(args.api_key_env_vars, list(common.DEFAULT_API_KEY_ENV_VARS))
        self.assertTrue(all(run["num_rollouts"] == 8 for run in runs))
        self.assertTrue(all(run["rollout_stream_max_concurrency"] == 4 for run in runs))
        self.assertTrue(all(run["rollout_stream_buffer_size"] == 8 for run in runs))
        self.assertTrue(all(run["groups_per_step"] in {4, 8} for run in runs))
        self.assertFalse(any(run["off_policy"] and run["reasoning"] for run in runs))
        self.assertTrue(all(run["base_url"] == common.DEFAULT_BASE_URL for run in runs))
        self.assertTrue(all(run["assigned_api_key_env_var"] in common.DEFAULT_API_KEY_ENV_VARS for run in runs))
        counts = Counter(run["assigned_api_key_env_var"] for run in runs)
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
        self.assertFalse(any(run["family"] == "query_offpolicy_rl" for run in runs))
        self.assertTrue(any(run["stage"] == "sft" for run in runs))
        self.assertTrue(any(run["stage"] == "rl" for run in runs))
        self.assertTrue(any(run["depends_on_run_name"] for run in runs if run["stage"] == "rl"))
        first_four_families = [run["family"] for run in runs[:4]]
        self.assertEqual(
            first_four_families,
            ["detect_sft", "point_sft", "query_proposal_sft", "query_finding_sft"],
        )

    def test_sweep_reserved_trainer_slots_preserve_detect_query_split(self) -> None:
        args = sweep_mod.parse_args(
            [
                "--dry-run",
                "--families",
                "detect_sft",
                "detect_rl",
                "query_proposal_sft",
                "query_proposal_rl",
                "query_finding_sft",
                "query_finding_rl",
                "--max-parallel-jobs",
                "4",
            ]
        )
        runs = sweep_mod.build_sweep_runs(args)
        reserved = sweep_mod._reserved_trainer_slots(runs, max_parallel=4)
        self.assertEqual(reserved, {"detect": 2, "query": 2})

    def test_query_failure_details_marks_gateway_errors_transient(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "query.log"
            log_path.write_text(
                "query sft train_step failed at step 1: TunaAPIError: error code: 524 | status=524 | request_id=req\n",
                encoding="utf-8",
            )
            details = sweep_mod._query_failure_details(log_path)
        self.assertEqual(details["status_code"], 524)
        self.assertTrue(details["transient"])

    def test_sweep_rl_command_uses_dependency_finetune_id_without_finetune_name(self) -> None:
        args = sweep_mod.parse_args(["--dry-run", "--families", "detect_sft", "detect_rl"])
        runs = sweep_mod.build_sweep_runs(args)
        rl_run = next(run for run in runs if run["family"] == "detect_rl")
        command = sweep_mod._materialize_launch_command(rl_run, dependency_finetune_id="ft_123")
        self.assertIn("--finetune-id", command)
        self.assertIn("ft_123", command)
        self.assertNotIn("--finetune-name", command)
        self.assertIn("--api-key-env-var", command)

    def test_readiness_summary_reports_missing_query_refresh(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            image_path = _write_image(tmp / "sample.png")
            manifest_path = tmp / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    [
                        {
                            "row_id": "row1",
                            "split": "train",
                            "image_path": str(image_path),
                            "inspection_request": "Inspect the roof",
                            "asset_context": "Exterior",
                            "hard_example": True,
                            "expected_proposals": [{"issue_code": "roof_cover_damage", "evidence": "missing shingles visible"}],
                            "expected_findings": [
                                {
                                    "issue_code": "roof_cover_damage",
                                    "title": "Roof Cover Damage",
                                    "box": {"x_min": 0.1, "y_min": 0.1, "x_max": 0.4, "y_max": 0.4},
                                    "evidence": ["missing shingles visible"],
                                    "severity": "major",
                                    "recommended_action": "repair roof",
                                    "cost_band": "high",
                                    "possible_compliance_issue": True,
                                    "insufficient_evidence": False,
                                    "compliance_note": "possible issue",
                                    "source_detect_labels": ["missing shingle"],
                                    "spatial_ref_index": 0
                                }
                            ]
                        }
                    ]
                ),
                encoding="utf-8",
            )
            outputs = tmp / "outputs"
            args = build_mod.parse_args(
                [
                    "--source-manifest",
                    str(manifest_path),
                    "--output-root",
                    str(outputs),
                    "--detect-output-dir",
                    str(outputs / "detect"),
                    "--point-output-dir",
                    str(outputs / "point"),
                    "--query-proposal-output-dir",
                    str(outputs / "proposal"),
                    "--query-finding-output-dir",
                    str(outputs / "finding"),
                    "--query-reasoning-output-dir",
                    str(outputs / "reasoning"),
                    "--query-text-refresh-mode",
                    "template_only",
                ]
            )
            build_mod.build_dataset(args)
            readiness_args = readiness_mod.parse_args(
                [
                    "--detect-dataset-path",
                    str(outputs / "detect"),
                    "--point-dataset-path",
                    str(outputs / "point"),
                    "--query-proposal-dataset-dir",
                    str(outputs / "proposal"),
                    "--query-finding-dataset-dir",
                    str(outputs / "finding"),
                    "--query-reasoning-dataset-dir",
                    str(outputs / "reasoning"),
                    "--no-verify-openrouter-judge",
                    "--output-json",
                    str(outputs / "readiness.json"),
                ]
            )
            summary = readiness_mod.evaluate_readiness(readiness_args)
            self.assertFalse(summary["ready"])
            teacher_check = next(item for item in summary["checks"] if item["name"] == "query_teacher_cache")
            self.assertFalse(teacher_check["ok"])

    def test_readiness_summary_passes_for_openrouter_refreshed_query_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            image_path = _write_image(tmp / "sample.png")
            manifest_path = tmp / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    [
                        {
                            "row_id": "row1",
                            "split": "train",
                            "image_path": str(image_path),
                            "inspection_request": "Inspect the roof",
                            "asset_context": "Exterior",
                            "hard_example": True,
                            "expected_proposals": [{"issue_code": "roof_cover_damage", "evidence": "missing shingles visible"}],
                            "expected_findings": [
                                {
                                    "issue_code": "roof_cover_damage",
                                    "title": "Roof Cover Damage",
                                    "box": {"x_min": 0.1, "y_min": 0.1, "x_max": 0.4, "y_max": 0.4},
                                    "evidence": ["missing shingles visible"],
                                    "severity": "major",
                                    "recommended_action": "repair roof",
                                    "cost_band": "high",
                                    "possible_compliance_issue": True,
                                    "insufficient_evidence": False,
                                    "compliance_note": "possible issue",
                                    "source_detect_labels": ["missing shingle"],
                                    "spatial_ref_index": 0
                                }
                            ]
                        }
                    ]
                ),
                encoding="utf-8",
            )
            outputs = tmp / "outputs"
            args = build_mod.parse_args(
                [
                    "--source-manifest",
                    str(manifest_path),
                    "--output-root",
                    str(outputs),
                    "--detect-output-dir",
                    str(outputs / "detect"),
                    "--point-output-dir",
                    str(outputs / "point"),
                    "--query-proposal-output-dir",
                    str(outputs / "proposal"),
                    "--query-finding-output-dir",
                    str(outputs / "finding"),
                    "--query-reasoning-output-dir",
                    str(outputs / "reasoning"),
                    "--query-text-refresh-mode",
                    "openrouter",
                    "--query-teacher-model-id",
                    "openai/gpt-4.1-mini",
                    "--query-teacher-api-key",
                    "test-openrouter-key",
                    "--query-text-cache-jsonl",
                    str(outputs / "query_text_cache.jsonl"),
                ]
            )
            build_mod.build_dataset(args, call_openrouter_fn=_fake_query_refresh_call)
            readiness_args = readiness_mod.parse_args(
                [
                    "--detect-dataset-path",
                    str(outputs / "detect"),
                    "--point-dataset-path",
                    str(outputs / "point"),
                    "--query-proposal-dataset-dir",
                    str(outputs / "proposal"),
                    "--query-finding-dataset-dir",
                    str(outputs / "finding"),
                    "--query-reasoning-dataset-dir",
                    str(outputs / "reasoning"),
                    "--no-verify-openrouter-judge",
                    "--output-json",
                    str(outputs / "readiness.json"),
                ]
            )
            summary = readiness_mod.evaluate_readiness(readiness_args)
            self.assertTrue(summary["ready"])
            teacher_check = next(item for item in summary["checks"] if item["name"] == "query_teacher_cache")
            self.assertTrue(teacher_check["ok"])
            self.assertTrue(summary["smoke_commands"])

    def test_query_benchmark_run_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            output_json = tmp / "metrics.json"
            predictions_jsonl = tmp / "predictions.jsonl"
            fake_pool = common.ApiKeyPool([common.ApiKeySlot(index=0, env_var="CICID_GPUB_MOONDREAM_API_KEY_1", api_key="md-key")])

            class _FakeGrader:
                def __init__(self, *args, **kwargs) -> None:
                    self.model_id = "openai/gpt-4.1-mini"
                    self.profile = "balanced"
                    self.rubric_version = "query_rubric_v1"

            with patch.object(bench_query_mod.common, "maybe_load_env_file", return_value=True), patch.object(
                bench_query_mod.common,
                "resolve_api_key_pool",
                return_value=fake_pool,
            ), patch.object(
                bench_query_mod.openrouter_grader,
                "resolve_openrouter_api_key",
                return_value="or-key",
            ), patch.object(
                bench_query_mod.openrouter_grader,
                "OpenRouterGrader",
                _FakeGrader,
            ), patch.object(
                bench_query_mod,
                "_load_split_examples",
                return_value=[],
            ), patch.object(
                bench_query_mod,
                "_evaluate_split",
                return_value={
                    "count": 3.0,
                    "reward_mean": 0.6,
                    "local_reward_mean": 0.4,
                    "judge_score_mean": 0.7,
                    "judge_degraded_rate": 0.0,
                    "json_parse_rate": 1.0,
                    "task_correct_rate": 0.5,
                    "issue_f1": 0.5,
                    "evidence_f1": 0.4,
                    "severity_accuracy": 0.3,
                    "insufficient_accuracy": 0.2,
                },
            ) as eval_mock, patch.object(
                bench_query_mod,
                "_build_benchmark_diagnostics",
                return_value={
                    "answer_parse_mode": "strict_raw",
                    "comparison_safe_answer_parse_mode": True,
                    "strict_reward_mean": 0.6,
                    "strict_local_reward_mean": 0.4,
                    "strict_judge_score_mean": 0.7,
                    "strict_parse_rate": 1.0,
                    "strict_issue_f1": 0.5,
                    "strict_reasoning_f1": 0.4,
                    "strict_task_correct_rate": 0.5,
                    "response_json_rate": 1.0,
                    "response_compact_text_rate": 0.0,
                    "response_normalized_rate": 0.0,
                    "response_heuristic_rate": 0.0,
                    "response_unparsed_rate": 0.0,
                    "response_legacy_fallback_rate": 0.0,
                    "response_generic_no_issue_rate": 0.0,
                    "response_parse_method_distribution": {"json": 3},
                    "judge_cache_jsonl": str(tmp / "predictions.judge_cache.jsonl"),
                    "metric_warnings": [],
                    "headline_metrics": {"reward_mean": 0.6},
                },
            ):
                args = bench_query_mod.parse_args(
                    [
                        "--output-json",
                        str(output_json),
                        "--predictions-jsonl",
                        str(predictions_jsonl),
                    ]
                )
                summary = bench_query_mod.run_benchmark(args)
            self.assertEqual(summary["grader_model_id"], "openai/gpt-4.1-mini")
            self.assertEqual(summary["answer_parse_mode"], "strict_raw")
            self.assertTrue(bool(summary["comparison_safe_answer_parse_mode"]))
            self.assertEqual(summary["reward_mean"], 0.6)
            self.assertEqual(eval_mock.call_args.kwargs["answer_parse_mode"], "strict_raw")
            self.assertTrue(output_json.exists())

    def test_query_benchmark_analysis_uses_strict_metrics_for_mixed_parse_modes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            strict_json = tmp / "strict.json"
            normalized_json = tmp / "normalized.json"
            strict_json.write_text(
                json.dumps(
                    {
                        "dataset_dir": "inspector_md/outputs/inspector_query_issues_v2",
                        "answer_parse_mode": "strict_raw",
                        "comparison_safe_answer_parse_mode": True,
                        "finetune_id": "strict_run",
                        "resolved_checkpoint_step": 30,
                        "reward_mean": 0.4,
                        "local_reward_mean": 0.3,
                        "judge_score_mean": 0.45,
                        "json_parse_rate": 0.25,
                        "task_correct_rate": 0.2,
                        "issue_f1": 0.15,
                        "evidence_f1": 0.1,
                        "strict_reward_mean": 0.4,
                        "strict_local_reward_mean": 0.3,
                        "strict_judge_score_mean": 0.45,
                        "strict_parse_rate": 0.25,
                        "strict_task_correct_rate": 0.2,
                        "strict_issue_f1": 0.15,
                        "strict_reasoning_f1": 0.1,
                    }
                ),
                encoding="utf-8",
            )
            normalized_json.write_text(
                json.dumps(
                    {
                        "dataset_dir": "inspector_md/outputs/inspector_query_issues_v2",
                        "answer_parse_mode": "grader_normalize",
                        "comparison_safe_answer_parse_mode": False,
                        "finetune_id": "normalized_run",
                        "resolved_checkpoint_step": 45,
                        "reward_mean": 0.9,
                        "local_reward_mean": 0.8,
                        "judge_score_mean": 0.95,
                        "json_parse_rate": 1.0,
                        "task_correct_rate": 0.85,
                        "issue_f1": 0.8,
                        "evidence_f1": 0.75,
                        "strict_reward_mean": 0.1,
                        "strict_local_reward_mean": 0.05,
                        "strict_judge_score_mean": 0.12,
                        "strict_parse_rate": 0.0,
                        "strict_task_correct_rate": 0.0,
                        "strict_issue_f1": 0.0,
                        "strict_reasoning_f1": 0.0,
                    }
                ),
                encoding="utf-8",
            )

            rows = analyze_query_benchmarks_mod._load_rows([strict_json, normalized_json])
            self.assertEqual(rows[0]["finetune_id"], "strict_run")
            self.assertEqual(rows[0]["comparison_metric_mode"], "strict_rescored")
            normalized_row = next(row for row in rows if row["finetune_id"] == "normalized_run")
            self.assertEqual(normalized_row["comparison_reward_mean"], 0.1)
            self.assertEqual(normalized_row["comparison_issue_f1"], 0.0)
            self.assertEqual(normalized_row["comparison_evidence_f1"], 0.0)

            summary = analyze_query_benchmarks_mod.build_summary(rows, title="test")
            self.assertEqual(summary["comparison_metric_mode"], "strict_rescored")

    def test_refresh_batch_launcher_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            args = refresh_batches_mod.parse_args(
                [
                    "--output-root",
                    str(tmp / "outputs"),
                    "--query-text-cache-jsonl",
                    str(tmp / "outputs" / "merged_cache.jsonl"),
                    "--shard-cache-dir",
                    str(tmp / "outputs" / "shards"),
                    "--log-dir",
                    str(tmp / "outputs" / "logs"),
                    "--num-shards",
                    "3",
                    "--max-parallel-jobs",
                    "2",
                    "--per-worker-concurrency",
                    "2",
                    "--dry-run",
                ]
            )
            summary = refresh_batches_mod.run_batches(args)
            self.assertTrue(bool(summary["dry_run"]))
            self.assertEqual(summary["num_shards"], 3)
            self.assertEqual(summary["max_parallel_jobs"], 2)
            self.assertEqual(summary["per_worker_concurrency"], 2)
            self.assertEqual(len(summary["shard_commands"]), 3)


class ReportBuilderTests(unittest.TestCase):
    def test_report_builder_aggregates_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            metrics_path = tmp / "metrics.json"
            metrics_path.write_text(
                json.dumps(
                    {
                        "proposal_precision": 0.5,
                        "proposal_recall": 0.4,
                        "proposal_f1": 0.44,
                        "localization_precision": 0.8,
                        "localization_recall": 0.7,
                        "localization_f1": 0.74,
                        "finding_schema_valid_rate": 1.0,
                        "end_to_end_score": 0.6,
                    }
                ),
                encoding="utf-8",
            )
            args = report_mod.parse_args(
                [
                    "--input-jsons",
                    str(metrics_path),
                    "--output-json",
                    str(tmp / "report.json"),
                    "--output-md",
                    str(tmp / "report.md"),
                ]
            )
            rows = report_mod._load_metrics(args.input_jsons)
            summary = report_mod._aggregate(rows)
            common.write_json(args.output_json, summary)
            common.write_text(args.output_md, report_mod._render_markdown(summary))
            self.assertTrue(Path(args.output_json).exists())
            self.assertIn("Proposal Quality", Path(args.output_md).read_text(encoding="utf-8"))


class RefreshAndDetectRobustnessTests(unittest.TestCase):
    def test_query_refresh_flushes_partial_cache_before_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            image_path_a = _write_image(tmp / "sample-a.png")
            image_path_b = _write_image(tmp / "sample-b.png", color="gray")
            manifest_path = tmp / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    [
                        {
                            "row_id": "row-a",
                            "split": "train",
                            "image_path": str(image_path_a),
                            "inspection_request": "Inspect the roof",
                            "asset_context": "Exterior",
                            "hard_example": True,
                            "expected_proposals": [{"issue_code": "roof_cover_damage", "evidence": "missing shingles visible"}],
                            "expected_findings": [
                                {
                                    "issue_code": "roof_cover_damage",
                                    "title": "Roof Cover Damage",
                                    "box": {"x_min": 0.1, "y_min": 0.1, "x_max": 0.4, "y_max": 0.4},
                                    "evidence": ["missing shingles visible"],
                                    "severity": "major",
                                    "recommended_action": "repair roof",
                                    "cost_band": "high",
                                    "possible_compliance_issue": True,
                                    "insufficient_evidence": False,
                                    "compliance_note": "possible issue",
                                    "source_detect_labels": ["missing shingle"],
                                    "spatial_ref_index": 0,
                                }
                            ],
                        },
                        {
                            "row_id": "row-b",
                            "split": "train",
                            "image_path": str(image_path_b),
                            "inspection_request": "Inspect the roof",
                            "asset_context": "Exterior",
                            "hard_example": True,
                            "expected_proposals": [{"issue_code": "roof_cover_damage", "evidence": "missing shingles visible"}],
                            "expected_findings": [
                                {
                                    "issue_code": "roof_cover_damage",
                                    "title": "Roof Cover Damage",
                                    "box": {"x_min": 0.2, "y_min": 0.2, "x_max": 0.5, "y_max": 0.5},
                                    "evidence": ["missing shingles visible"],
                                    "severity": "major",
                                    "recommended_action": "repair roof",
                                    "cost_band": "high",
                                    "possible_compliance_issue": True,
                                    "insufficient_evidence": False,
                                    "compliance_note": "possible issue",
                                    "source_detect_labels": ["missing shingle"],
                                    "spatial_ref_index": 0,
                                }
                            ],
                        },
                    ]
                ),
                encoding="utf-8",
            )

            call_count = {"value": 0}

            def _flaky_refresh(**kwargs):
                call_count["value"] += 1
                if call_count["value"] == 1:
                    return _fake_query_refresh_call(**kwargs)
                raise TimeoutError("simulated refresh timeout")

            cache_path = tmp / "outputs" / "query_text_cache.jsonl"
            args = build_mod.parse_args(
                [
                    "--source-manifest",
                    str(manifest_path),
                    "--output-root",
                    str(tmp / "outputs"),
                    "--query-text-refresh-mode",
                    "openrouter",
                    "--query-teacher-model-id",
                    "openai/gpt-4.1-mini",
                    "--query-teacher-api-key",
                    "test-openrouter-key",
                    "--query-text-cache-jsonl",
                    str(cache_path),
                    "--query-refresh-only",
                    "--query-refresh-cache-flush-every",
                    "1",
                    "--query-refresh-retries",
                    "0",
                    "--no-progress",
                ]
            )
            with self.assertRaises(TimeoutError):
                build_mod.build_dataset(args, call_openrouter_fn=_flaky_refresh)
            cached_rows = common.load_jsonl(cache_path)
            self.assertEqual(len(cached_rows), 1)

    def test_refresh_batch_launcher_resumes_completed_shards(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            outputs = tmp / "outputs"
            shard_cache_dir = outputs / "shards"
            log_dir = outputs / "logs"
            shard_cache_dir.mkdir(parents=True, exist_ok=True)
            log_dir.mkdir(parents=True, exist_ok=True)
            common.write_jsonl(
                shard_cache_dir / "query_text_cache.shard_000_of_002.jsonl",
                [{"cache_key": "k0", "row_id": "row-0", "bundle": {}}],
            )
            common.write_json(
                outputs / "query_refresh_summary.shard_000_of_002.json",
                {"query_refresh_only": True, "selected_record_count": 1},
            )
            launched: list[list[str]] = []

            class _FakeProc:
                _next_pid = 1000

                def __init__(self, cmd, **kwargs) -> None:
                    type(self)._next_pid += 1
                    self.pid = type(self)._next_pid
                    self._done = False
                    self.returncode = 0
                    launched.append(list(cmd))
                    shard_index = int(cmd[cmd.index("--query-refresh-shard-index") + 1])
                    cache_path = Path(cmd[cmd.index("--query-text-cache-jsonl") + 1])
                    summary_path = outputs / f"query_refresh_summary.shard_{shard_index:03d}_of_002.json"
                    common.write_jsonl(cache_path, [{"cache_key": f"k{shard_index}", "row_id": f"row-{shard_index}", "bundle": {}}])
                    common.write_json(summary_path, {"query_refresh_only": True, "selected_record_count": 1})

                def poll(self):
                    if not self._done:
                        self._done = True
                        return 0
                    return 0

            with patch.object(refresh_batches_mod.subprocess, "Popen", _FakeProc):
                args = refresh_batches_mod.parse_args(
                    [
                        "--output-root",
                        str(outputs),
                        "--query-text-cache-jsonl",
                        str(outputs / "merged_cache.jsonl"),
                        "--shard-cache-dir",
                        str(shard_cache_dir),
                        "--log-dir",
                        str(log_dir),
                        "--num-shards",
                        "2",
                        "--max-parallel-jobs",
                        "1",
                        "--per-worker-concurrency",
                        "1",
                        "--no-finalize",
                    ]
                )
                summary = refresh_batches_mod.run_batches(args)
            self.assertEqual(len(launched), 1)
            self.assertIn("--query-refresh-shard-index", launched[0])
            self.assertEqual(launched[0][launched[0].index("--query-refresh-shard-index") + 1], "1")
            skipped = [item for item in summary["shards"] if item["status"] == "skipped_complete"]
            self.assertEqual(len(skipped), 1)

    def test_detect_benchmark_emits_eval_compatibility_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            image_path = _write_image(tmp / "detect.png")
            output_json = tmp / "metrics.json"
            predictions_jsonl = tmp / "predictions.jsonl"
            fake_pool = common.ApiKeyPool(
                [common.ApiKeySlot(index=0, env_var="CICID_GPUB_MOONDREAM_API_KEY_1", api_key="md-key")]
            )
            sample = benchmark_mod.ExpectedSample(
                row_id="row-1",
                split="validation",
                source_dataset="MBDD2025",
                request=task_schema.InspectionRequest.from_payload(
                    {
                        "image_path": str(image_path),
                        "inspection_request": "Inspect the image",
                        "asset_context": "Exterior",
                    }
                ),
                expected_proposals=[],
                expected_findings=[
                    task_schema.Finding.from_payload(
                        {
                            "finding_id": "f1",
                            "issue_code": "surface_spalling",
                            "title": "Surface Spalling",
                            "box": {"x_min": 0.1, "y_min": 0.1, "x_max": 0.4, "y_max": 0.4},
                            "evidence": ["spalling visible"],
                            "severity": "major",
                            "recommended_action": "repair",
                            "cost_band": "high",
                            "possible_compliance_issue": False,
                            "insufficient_evidence": False,
                            "compliance_note": "",
                            "source_detect_labels": ["concrete spalling"],
                            "spatial_ref_index": 0,
                        }
                    )
                ],
            )

            class _FakeClient:
                def __init__(self, *args, **kwargs) -> None:
                    return None

                def detect_boxes(self, **kwargs):
                    return [moondream_client.DetectAnnotation(x_min=0.1, y_min=0.1, x_max=0.4, y_max=0.4)]

            with patch.object(bench_detect_mod.common, "maybe_load_env_file", return_value=True), patch.object(
                bench_detect_mod.common,
                "resolve_api_key_pool",
                return_value=fake_pool,
            ), patch.object(
                bench_detect_mod,
                "_load_expected_samples",
                return_value=[sample],
            ), patch.object(
                bench_detect_mod,
                "MoondreamInspectorClient",
                _FakeClient,
            ):
                args = bench_detect_mod.parse_args(
                    [
                        "--dataset-manifest",
                        str(tmp / "unused.json"),
                        "--output-json",
                        str(output_json),
                        "--predictions-jsonl",
                        str(predictions_jsonl),
                    ]
                )
                summary = bench_detect_mod.run_benchmark(args)
            self.assertEqual(summary["eval_tasks"], 1)
            self.assertEqual(summary["eval_positive_tasks"], 1)
            self.assertEqual(summary["eval_positive_task_shortfall"], 0)
            self.assertGreater(summary["eval_positive_f1"], 0.0)
            self.assertEqual(summary["eval_negative_tasks"], 0)


if __name__ == "__main__":
    unittest.main()
