#!/usr/bin/env python3
"""Run a detailed baseline evaluation for Inspector MD with per-sample artifacts."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import socket
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Optional

from PIL import Image, ImageDraw, ImageFont
from tuna_sdk import QueryRequest, QuerySettings

from inspector_md import common, ontology, openrouter_grader, prompt_library, task_schema
from inspector_md.benchmark_inspector_pipeline import (
    _center_hit_box,
    _finding_pair_score,
    _localization_metrics_from_matches,
    _match_findings,
    _match_findings_by_center,
    _mean_best_iou,
    _proposal_metrics,
)
from inspector_md.moondream_client import MoondreamInspectorClient, QueryResult
from inspector_md.pipeline import InspectorPipeline, PipelineModels, PipelineSettings, box_iou as pipeline_box_iou, report_to_punch_list_text

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = common.repo_relative("configs", "run_baseline_inspector_eval_default.json")

GT_COLOR = (22, 163, 74)
PRED_COLOR = (220, 38, 38)


@dataclass(frozen=True)
class EvalSample:
    row_id: str
    split: str
    source_dataset: str
    source_metadata: dict[str, Any]
    request: task_schema.InspectionRequest
    expected_proposals: list[task_schema.IssueProposal]
    expected_findings: list[task_schema.Finding]


def _resolve_model(base_model: str, finetune_id: str) -> str:
    finetune = str(finetune_id or "").strip()
    return f"{base_model}/{finetune}" if finetune else str(base_model).strip()


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    raw_argv = list(argv) if argv is not None else list(os.sys.argv[1:])
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    pre_args, _ = pre_parser.parse_known_args(raw_argv)
    config_path = common.resolve_config_path(pre_args.config, script_dir=SCRIPT_DIR)
    config = common.load_json_config(config_path, default_path=DEFAULT_CONFIG_PATH)
    single_key_requested = "--api-key-env-var" in raw_argv or (
        "api_key_env_var" in config and "--api-key-env-vars" not in raw_argv
    )
    multi_key_requested = "--api-key-env-vars" in raw_argv or (
        "api_key_env_vars" in config and "--api-key-env-var" not in raw_argv
    )

    parser = argparse.ArgumentParser(description="Run a detailed baseline evaluation with per-sample artifacts.")
    parser.add_argument("--config", default=str(config_path))
    parser.add_argument("--env-file", default=str(common.repo_relative(".env.staging")))
    parser.add_argument("--api-key", default="")
    parser.add_argument("--api-key-env-var", default=common.DEFAULT_API_KEY_ENV_VAR)
    parser.add_argument("--api-key-env-vars", nargs="+", default=list(common.DEFAULT_API_KEY_ENV_VARS))
    parser.add_argument("--base-url", default=common.DEFAULT_BASE_URL)
    parser.add_argument("--dataset-manifest", default=str(common.repo_relative("dataset", "merged_synth_v1", "smoke_manifest.json")))
    parser.add_argument("--split", default="all")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--detect-finetune-id", default="")
    parser.add_argument("--query-finetune-id", default="")
    parser.add_argument("--base-model", default=common.DEFAULT_BASE_MODEL)
    parser.add_argument("--reasoning", action="store_true")
    parser.add_argument("--detect-max-objects", type=int, default=24)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--coarse-iou-threshold", type=float, default=0.1)
    parser.add_argument("--detect-exclude-source-datasets", nargs="*", default=["CODEBRIM"])
    parser.add_argument("--proposal-prompt-style", choices=("structured", "request_only"), default="request_only")
    parser.add_argument("--finding-prompt-style", choices=("structured", "minimal"), default="minimal")
    parser.add_argument("--query-response-mode", choices=("json", "text"), default="text")
    parser.add_argument("--grading-mode", choices=("rule", "openrouter"), default="openrouter")
    parser.add_argument("--grader-api-key", default="")
    parser.add_argument("--grader-api-key-env-var", default=openrouter_grader.DEFAULT_OPENROUTER_ENV_VAR)
    parser.add_argument("--grader-api-base", default=openrouter_grader.DEFAULT_OPENROUTER_API_BASE)
    parser.add_argument("--grader-model-id", default=openrouter_grader.DEFAULT_GRADER_MODEL)
    parser.add_argument("--grader-profile", default=openrouter_grader.DEFAULT_GRADER_PROFILE)
    parser.add_argument("--grader-rubric-version", default=openrouter_grader.DEFAULT_GRADER_RUBRIC_VERSION)
    parser.add_argument("--grader-timeout", type=float, default=60.0)
    parser.add_argument("--output-dir", default=str(common.repo_relative("outputs", "baseline_eval_smoke_v1")))
    parser.add_argument("--save-viz", action="store_true", default=True)
    parser.add_argument("--max-viz-dim", type=int, default=1280)
    parser.add_argument("--timeout", type=float, default=120.0)

    option_to_dest: dict[str, str] = {}
    for action in parser._actions:
        if not action.option_strings:
            continue
        for opt in action.option_strings:
            option_to_dest[opt] = action.dest
    overridden = {option_to_dest[arg] for arg in raw_argv if arg in option_to_dest}
    config_cli_args = common.config_to_cli_args(parser, config, config_path=config_path, overridden_dests=overridden)
    args = parser.parse_args(config_cli_args + raw_argv)
    args.config = str(common.resolve_config_path(args.config, script_dir=SCRIPT_DIR))
    args.env_file = str(common.resolve_path(args.env_file, module_root=SCRIPT_DIR))
    args.dataset_manifest = common.resolve_path(args.dataset_manifest, module_root=SCRIPT_DIR)
    args.output_dir = common.resolve_path(args.output_dir, module_root=SCRIPT_DIR)
    if single_key_requested and not multi_key_requested:
        args.api_key_env_vars = [args.api_key_env_var]
    else:
        args.api_key_env_vars = common.normalize_api_key_env_vars(args.api_key_env_vars)
    split = str(args.split or "").strip().lower()
    if split not in {"all", "train", "validation", "test", "val"}:
        raise ValueError(f"Unsupported split: {args.split!r}")
    args.split = "validation" if split == "val" else split
    args.detect_exclude_source_datasets = [str(item).strip() for item in list(args.detect_exclude_source_datasets or []) if str(item).strip()]
    return args


def _sample_slug(sample: EvalSample) -> str:
    text = f"{sample.source_dataset}_{sample.split}_{sample.row_id}"
    text = "".join(ch if ch.isalnum() else "_" for ch in text.lower())
    return "_".join(part for part in text.split("_") if part)


def _load_samples(manifest_path: Path) -> list[EvalSample]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("evaluation manifest must be a JSON array")
    samples: list[EvalSample] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        request = task_schema.InspectionRequest.from_payload(
            {
                "image_path": str(item.get("image_path") or ""),
                "inspection_request": str(item.get("inspection_request") or ""),
                "asset_context": str(item.get("asset_context") or ""),
            }
        )
        samples.append(
            EvalSample(
                row_id=str(item.get("row_id") or request.image_path),
                split=str(item.get("split") or "test"),
                source_dataset=str(item.get("source_dataset") or item.get("source_metadata", {}).get("source_dataset") or "unknown"),
                source_metadata=dict(item.get("source_metadata") or {}),
                request=request,
                expected_proposals=task_schema.normalize_issue_proposals(item.get("expected_proposals") or []),
                expected_findings=task_schema.normalize_findings(item.get("expected_findings") or []),
            )
        )
    return samples


def _load_font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "Arial.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _fit_size(width: int, height: int, *, max_dim: int) -> tuple[int, int, float]:
    scale = min(1.0, float(max_dim) / float(max(width, height)))
    return max(1, int(round(width * scale))), max(1, int(round(height * scale))), scale


def _draw_labeled_box(
    draw: ImageDraw.ImageDraw,
    *,
    box: task_schema.Box,
    scale: float,
    color: tuple[int, int, int],
    label: str,
    font: ImageFont.ImageFont,
    line_width: int,
) -> None:
    x_min = box.x_min * scale
    y_min = box.y_min * scale
    x_max = box.x_max * scale
    y_max = box.y_max * scale
    draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=line_width)
    if not label:
        return
    padding = 4
    bbox = draw.textbbox((0, 0), label, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    draw.rectangle(
        [x_min, max(0, y_min - text_height - (padding * 2) - 2), x_min + text_width + (padding * 2), max(0, y_min - 2)],
        fill=color,
    )
    draw.text((x_min + padding, max(0, y_min - text_height - padding - 2)), label, fill=(255, 255, 255), font=font)


def _render_comparison_viz(
    *,
    sample: EvalSample,
    predicted_findings: list[task_schema.Finding],
    sample_metrics: dict[str, float],
    output_path: Path,
    max_dim: int,
) -> None:
    title_font = _load_font(20)
    label_font = _load_font(14)
    small_font = _load_font(12)

    with Image.open(Path(sample.request.image_path)) as image:
        image = image.convert("RGB")
        new_width, new_height, image_scale = _fit_size(image.width, image.height, max_dim=max_dim)
        canvas = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    header_height = 82
    output = Image.new("RGB", (canvas.width, canvas.height + header_height), (250, 250, 250))
    output.paste(canvas, (0, header_height))
    draw = ImageDraw.Draw(output)
    draw.rectangle([0, 0, output.width, header_height], fill=(255, 255, 255))
    draw.text((16, 10), f"{sample.row_id} | {sample.source_dataset} | {sample.split}", fill=(25, 25, 25), font=title_font)
    metrics_text = (
        f"proposal_f1={sample_metrics['proposal_f1']:.3f} "
        f"localization_f1={sample_metrics['localization_f1']:.3f} "
        f"end_to_end={sample_metrics['end_to_end_score']:.3f}"
    )
    draw.text((16, 40), metrics_text, fill=(70, 70, 70), font=small_font)
    draw.text((16, 58), "Green = expected, Red = predicted", fill=(70, 70, 70), font=small_font)

    overlay = ImageDraw.Draw(output)
    pixel_scale = float(canvas.width)
    for finding in sample.expected_findings:
        shifted = task_schema.Box(
            x_min=finding.box.x_min * pixel_scale,
            y_min=(finding.box.y_min * canvas.height) + header_height,
            x_max=finding.box.x_max * pixel_scale,
            y_max=(finding.box.y_max * canvas.height) + header_height,
        )
        _draw_labeled_box(
            overlay,
            box=shifted,
            scale=1.0,
            color=GT_COLOR,
            label=f"GT {finding.issue_code}",
            font=label_font,
            line_width=3,
        )
    for finding in predicted_findings:
        shifted = task_schema.Box(
            x_min=finding.box.x_min * canvas.width,
            y_min=(finding.box.y_min * canvas.height) + header_height,
            x_max=finding.box.x_max * canvas.width,
            y_max=(finding.box.y_max * canvas.height) + header_height,
        )
        _draw_labeled_box(
            overlay,
            box=shifted,
            scale=1.0,
            color=PRED_COLOR,
            label=f"PR {finding.issue_code}",
            font=label_font,
            line_width=3,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.save(output_path)


class TracingMoondreamInspectorClient(MoondreamInspectorClient):
    def __init__(
        self,
        *,
        proposal_prompt_style: str = "structured",
        finding_prompt_style: str = "structured",
        query_response_mode: str = "json",
        grader: Optional[openrouter_grader.OpenRouterGrader] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.proposal_prompt_style = str(proposal_prompt_style or "structured").strip().lower()
        self.finding_prompt_style = str(finding_prompt_style or "structured").strip().lower()
        self.query_response_mode = str(query_response_mode or "json").strip().lower()
        self.grader = grader
        self.events: list[dict[str, Any]] = []
        self._active_row_id = ""

    def start_sample(self, row_id: str) -> None:
        self._active_row_id = str(row_id)
        self.events = []

    def _append_event(self, payload: dict[str, Any]) -> None:
        self.events.append({"row_id": self._active_row_id, **payload})

    def propose_issues(
        self,
        *,
        model: str,
        image_path: str | Path,
        inspection_request: str,
        asset_context: str = "",
        reasoning: bool = False,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 512,
    ) -> tuple[list[task_schema.IssueProposal], QueryResult]:
        question = prompt_library.build_visible_issue_question_with_style(
            inspection_request=inspection_request,
            asset_context=asset_context,
            prompt_style=self.proposal_prompt_style,
            variation_key=str(image_path),
        )
        request = QueryRequest(
            question=question,
            image_url=self._load_image_url(image_path),
            reasoning=bool(reasoning),
            settings=QuerySettings(
                temperature=float(temperature),
                top_p=float(top_p),
                max_tokens=int(max_tokens),
            ),
        )
        result = self.query_raw(model=model, request=request)
        grader_response: Optional[dict[str, Any]] = None
        parsed_payload = result.payload
        if self.query_response_mode == "text":
            if self.grader is None:
                raise ValueError("query_response_mode='text' requires an OpenRouter grader.")
            grader_response = self.grader.normalize_issues(
                inspection_request=inspection_request,
                asset_context=asset_context,
                answer_text=result.answer_text,
            )
            parsed_payload = {"issues": list(grader_response.get("issues") or [])}
        proposals = task_schema.normalize_issue_proposals(parsed_payload)
        traced_result = QueryResult(
            payload=parsed_payload,
            raw_response=result.raw_response,
            latency_ms=result.latency_ms,
            answer_text=result.answer_text,
        )
        self._append_event(
            {
                "task": "proposal_query",
                "model": str(model),
                "prompt": question,
                "request_payload": request.to_payload(),
                "answer_text": result.answer_text,
                "parsed_response": parsed_payload,
                "raw_response": result.raw_response,
                "grader_response": grader_response,
                "latency_ms": result.latency_ms,
                "normalized_response": [item.to_payload() for item in proposals],
            }
        )
        return proposals, traced_result

    def detect_boxes(
        self,
        *,
        model: str,
        image_path: str | Path,
        detect_label: str,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 256,
        max_objects: int = 24,
    ):
        boxes = super().detect_boxes(
            model=model,
            image_path=image_path,
            detect_label=detect_label,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            max_objects=max_objects,
        )
        self._append_event(
            {
                "task": "detect",
                "model": str(model),
                "prompt": str(detect_label),
                "request_payload": {
                    "object_name": str(detect_label),
                    "temperature": float(temperature),
                    "top_p": float(top_p),
                    "max_tokens": int(max_tokens),
                    "max_objects": int(max_objects),
                },
                "parsed_response": {
                    "boxes": [
                        {"x_min": item.x_min, "y_min": item.y_min, "x_max": item.x_max, "y_max": item.y_max} for item in boxes
                    ]
                },
            }
        )
        return boxes

    def query_finding_fields(
        self,
        *,
        model: str,
        image_path: str | Path,
        localized_issue: task_schema.LocalizedIssue,
        inspection_request: str,
        asset_context: str = "",
        reasoning: bool = False,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 384,
    ) -> QueryResult:
        question = prompt_library.build_finding_question_with_style(
            localized_issue=localized_issue,
            inspection_request=inspection_request,
            asset_context=asset_context,
            prompt_style=self.finding_prompt_style,
        )
        request = QueryRequest(
            question=question,
            image_url=self._load_image_url(image_path),
            spatial_refs=[prompt_library.box_to_spatial_ref(localized_issue.box)],
            reasoning=bool(reasoning),
            settings=QuerySettings(
                temperature=float(temperature),
                top_p=float(top_p),
                max_tokens=int(max_tokens),
            ),
        )
        result = self.query_raw(model=model, request=request)
        grader_response: Optional[dict[str, Any]] = None
        parsed_payload = result.payload
        if self.query_response_mode == "text":
            if self.grader is None:
                raise ValueError("query_response_mode='text' requires an OpenRouter grader.")
            grader_response = self.grader.normalize_finding(
                localized_issue=localized_issue,
                inspection_request=inspection_request,
                asset_context=asset_context,
                answer_text=result.answer_text,
            )
            parsed_payload = {"finding": dict(grader_response.get("finding") or {})}
        traced_result = QueryResult(
            payload=parsed_payload,
            raw_response=result.raw_response,
            latency_ms=result.latency_ms,
            answer_text=result.answer_text,
        )
        self._append_event(
            {
                "task": "finding_query",
                "model": str(model),
                "issue_code": localized_issue.issue_code,
                "prompt": question,
                "request_payload": request.to_payload(),
                "answer_text": result.answer_text,
                "parsed_response": parsed_payload,
                "raw_response": result.raw_response,
                "grader_response": grader_response,
                "latency_ms": result.latency_ms,
                "spatial_refs": [prompt_library.box_to_spatial_ref(localized_issue.box)],
            }
        )
        return traced_result


def _filter_samples(
    samples: list[EvalSample],
    *,
    split: str,
    seed: int,
    max_samples: int,
    excluded_source_datasets: set[str],
) -> list[EvalSample]:
    filtered = [
        sample
        for sample in samples
        if (split == "all" or sample.split == split) and sample.source_dataset not in excluded_source_datasets
    ]
    random.Random(seed).shuffle(filtered)
    if int(max_samples) > 0:
        filtered = filtered[: int(max_samples)]
    return filtered


def _safe_mean(values: list[float], *, default: float = 0.0) -> float:
    return fmean(values) if values else float(default)


def _finding_query_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [event for event in events if str(event.get("task") or "") == "finding_query"]


def _best_finding_judge_matches(
    expected: list[task_schema.Finding],
    predicted: list[task_schema.Finding],
    *,
    coarse_iou_threshold: float,
) -> dict[int, int]:
    matches: dict[int, int] = {}
    remaining_gt = set(range(len(expected)))
    for pred_index, pred in enumerate(predicted):
        best_gt_index = -1
        best_issue_score = 0.0
        best_iou = -1.0
        for gt_index in list(remaining_gt):
            gt = expected[gt_index]
            issue_score = ontology.issue_match_score(gt.issue_code, pred.issue_code)
            if issue_score <= 0.0:
                continue
            score = pipeline_box_iou(pred.box, gt.box)
            if score < float(coarse_iou_threshold) and not _center_hit_box(gt.box, pred.box):
                continue
            if issue_score > best_issue_score or (issue_score == best_issue_score and score > best_iou):
                best_issue_score = issue_score
                best_iou = score
                best_gt_index = gt_index
        if best_gt_index >= 0:
            matches[pred_index] = best_gt_index
            remaining_gt.discard(best_gt_index)
    return matches


def _proposal_judge_defaults() -> dict[str, Any]:
    return openrouter_grader.score_issue_judgement(
        {
            "satisfies_gt": "no",
            "grounded_visible_evidence": "no",
            "unsupported_claims": "none",
            "extra_issue_claims": "none",
            "verbosity": "acceptable",
            "reason": "no_proposal_judgement",
        }
    )
def _compute_query_judge_details(
    *,
    sample: EvalSample,
    report: task_schema.InspectionReport,
    events: list[dict[str, Any]],
    grader: Optional[openrouter_grader.OpenRouterGrader],
    coarse_iou_threshold: float,
) -> dict[str, Any]:
    proposal_event = _proposal_event(events)
    proposal_judgement = proposal_event.get("judge_output") if isinstance(proposal_event, dict) else None
    if not isinstance(proposal_judgement, dict):
        proposal_judgement = _proposal_judge_defaults()
    proposal_judgement = openrouter_grader.score_issue_judgement(proposal_judgement)
    if isinstance(proposal_event, dict):
        proposal_event["judge_output"] = proposal_judgement
        proposal_event["judge_score"] = proposal_judgement["score"]

    return {
        "proposal_judge": proposal_judgement,
        "finding_judgements": [],
        "metrics": {
            "proposal_judge_score": float(proposal_judgement["score"]),
            "proposal_judge_yes_rate": 1.0 if proposal_judgement["satisfies_gt"] == "yes" else 0.0,
            "proposal_judge_partial_rate": 1.0 if proposal_judgement["satisfies_gt"] == "partial" else 0.0,
            "proposal_unsupported_claim_rate": 1.0 if proposal_judgement["unsupported_claims"] != "none" else 0.0,
            "proposal_verbose_rate": 1.0 if proposal_judgement["verbosity"] == "verbose" else 0.0,
            "finding_judge_score": 0.0,
            "finding_judge_yes_rate": 0.0,
            "finding_judge_partial_rate": 0.0,
            "finding_unsupported_claim_rate": 0.0,
            "finding_verbose_rate": 0.0,
            "query_judge_score": float(proposal_judgement["score"]),
        },
    }


def _infer_failure_stage(*, stage: str, events: list[dict[str, Any]]) -> str:
    normalized = str(stage or "").strip().lower()
    if normalized and normalized != "pipeline":
        return normalized
    if not events:
        return "proposal_query"
    last_task = str(events[-1].get("task") or "").strip().lower()
    if last_task == "proposal_query":
        return "detect"
    if last_task == "detect":
        return "finding_query"
    if last_task == "finding_query":
        return "finding_query"
    return last_task or "proposal_query"


def _sample_metrics(
    *,
    sample: EvalSample,
    report: task_schema.InspectionReport,
    iou_threshold: float,
    coarse_iou_threshold: float,
    query_judge_details: Optional[dict[str, Any]] = None,
) -> dict[str, float]:
    predicted_proposals = task_schema.normalize_issue_proposals(report.trace.get("proposals") or [])
    proposal_metrics = _proposal_metrics(sample.expected_proposals, predicted_proposals)
    strict_matches = _match_findings(sample.expected_findings, report.findings, iou_threshold=float(iou_threshold))
    strict_metrics = _localization_metrics_from_matches(
        expected=sample.expected_findings,
        predicted=report.findings,
        matches=strict_matches,
    )
    loose_matches = _match_findings(
        sample.expected_findings,
        report.findings,
        iou_threshold=float(coarse_iou_threshold),
    )
    loose_metrics = _localization_metrics_from_matches(
        expected=sample.expected_findings,
        predicted=report.findings,
        matches=loose_matches,
    )
    center_matches = _match_findings_by_center(sample.expected_findings, report.findings)
    center_metrics = _localization_metrics_from_matches(
        expected=sample.expected_findings,
        predicted=report.findings,
        matches=center_matches,
    )
    pair_scores = [_finding_pair_score(gt, pred) for gt, pred, _ in strict_matches]
    if sample.expected_findings:
        pair_scores.extend([0.0] * max(0, len(sample.expected_findings) - len(strict_matches)))
    end_to_end = fmean(pair_scores) if pair_scores else 0.0
    query_judge_metrics = dict((query_judge_details or {}).get("metrics") or {})
    return {
        "proposal_precision": proposal_metrics["precision"],
        "proposal_recall": proposal_metrics["recall"],
        "proposal_f1": proposal_metrics["f1"],
        "localization_precision": strict_metrics["precision"],
        "localization_recall": strict_metrics["recall"],
        "localization_f1": strict_metrics["f1"],
        "localization_precision_iou_0_1": loose_metrics["precision"],
        "localization_recall_iou_0_1": loose_metrics["recall"],
        "localization_f1_iou_0_1": loose_metrics["f1"],
        "localization_precision_center_hit": center_metrics["precision"],
        "localization_recall_center_hit": center_metrics["recall"],
        "localization_f1_center_hit": center_metrics["f1"],
        "mean_best_iou": _mean_best_iou(sample.expected_findings, report.findings),
        "end_to_end_score": end_to_end,
        "finding_schema_valid": 1.0 if all(isinstance(item, task_schema.Finding) for item in report.findings) else 0.0,
        "proposal_judge_score": float(query_judge_metrics.get("proposal_judge_score", 0.0) or 0.0),
        "proposal_judge_yes_rate": float(query_judge_metrics.get("proposal_judge_yes_rate", 0.0) or 0.0),
        "proposal_judge_partial_rate": float(query_judge_metrics.get("proposal_judge_partial_rate", 0.0) or 0.0),
        "proposal_unsupported_claim_rate": float(query_judge_metrics.get("proposal_unsupported_claim_rate", 0.0) or 0.0),
        "proposal_verbose_rate": float(query_judge_metrics.get("proposal_verbose_rate", 0.0) or 0.0),
        "finding_judge_score": float(query_judge_metrics.get("finding_judge_score", 0.0) or 0.0),
        "finding_judge_yes_rate": float(query_judge_metrics.get("finding_judge_yes_rate", 0.0) or 0.0),
        "finding_judge_partial_rate": float(query_judge_metrics.get("finding_judge_partial_rate", 0.0) or 0.0),
        "finding_unsupported_claim_rate": float(query_judge_metrics.get("finding_unsupported_claim_rate", 0.0) or 0.0),
        "finding_verbose_rate": float(query_judge_metrics.get("finding_verbose_rate", 0.0) or 0.0),
        "query_judge_score": float(query_judge_metrics.get("query_judge_score", 0.0) or 0.0),
    }


def _write_sample_artifacts(
    *,
    output_dir: Path,
    sample: EvalSample,
    report: task_schema.InspectionReport,
    events: list[dict[str, Any]],
    metrics: dict[str, float],
    save_viz: bool,
    max_viz_dim: int,
    retry_count: int = 0,
    judge_details: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    sample_dir = output_dir / "samples" / sample.source_dataset / sample.split / _sample_slug(sample)
    sample_dir.mkdir(parents=True, exist_ok=True)
    record_payload = {
        "row_id": sample.row_id,
        "source_dataset": sample.source_dataset,
        "split": sample.split,
        "source_image_path": sample.request.image_path,
        "request": sample.request.to_payload(),
        "expected": {
            "proposals": [item.to_payload() for item in sample.expected_proposals],
            "findings": [item.to_payload() for item in sample.expected_findings],
        },
        "predicted": report.to_payload(),
        "metrics": metrics,
        "tasks": events,
        "source_metadata": sample.source_metadata,
        "retry_count": int(retry_count),
    }
    if isinstance(judge_details, dict):
        record_payload["query_judging"] = dict(judge_details)
    common.write_json(sample_dir / "record.json", record_payload)
    common.write_json(sample_dir / "tasks.json", {"tasks": events})
    common.write_json(sample_dir / "report.json", report.to_payload())
    common.write_text(sample_dir / "punch_list.txt", report_to_punch_list_text(report))

    task_lines = [
        f"Row: {sample.row_id}",
        f"Source image: {sample.request.image_path}",
        f"Source dataset: {sample.source_dataset}",
        f"Split: {sample.split}",
        "",
    ]
    for index, event in enumerate(events, start=1):
        response_text = str(event.get("answer_text") or "").strip()
        task_lines.extend(
            [
                f"Task {index}: {event.get('task', 'unknown')}",
                f"Prompt: {event.get('prompt', '')}",
            ]
        )
        if response_text:
            task_lines.append(f"Response: {response_text}")
        else:
            task_lines.append(f"Response: {json.dumps(event.get('parsed_response') or event.get('raw_response') or {}, ensure_ascii=False)}")
        if event.get("parsed_response") is not None:
            task_lines.append(f"Parsed: {json.dumps(event.get('parsed_response') or {}, ensure_ascii=False)}")
        if event.get("grader_response") is not None:
            task_lines.append(f"Grader: {json.dumps(event.get('grader_response') or {}, ensure_ascii=False)}")
        if event.get("judge_output") is not None:
            task_lines.append(f"Judge: {json.dumps(event.get('judge_output') or {}, ensure_ascii=False)}")
        if event.get("judge_score") is not None:
            task_lines.append(f"Judge score: {event.get('judge_score')}")
        task_lines.append("")
    if isinstance(judge_details, dict):
        task_lines.extend(
            [
                "Query judge summary:",
                json.dumps(dict(judge_details.get("metrics") or {}), ensure_ascii=False),
                "",
            ]
        )
    common.write_text(sample_dir / "tasks.md", "\n".join(task_lines).rstrip() + "\n")

    viz_path = sample_dir / "comparison.png"
    if save_viz:
        _render_comparison_viz(
            sample=sample,
            predicted_findings=report.findings,
            sample_metrics=metrics,
            output_path=viz_path,
            max_dim=int(max_viz_dim),
        )
    return {
        "status": "ok",
        "row_id": sample.row_id,
        "source_dataset": sample.source_dataset,
        "split": sample.split,
        "sample_dir": str(sample_dir),
        "record_json": str(sample_dir / "record.json"),
        "tasks_md": str(sample_dir / "tasks.md"),
        "report_json": str(sample_dir / "report.json"),
        "punch_list_txt": str(sample_dir / "punch_list.txt"),
        "comparison_png": str(viz_path) if save_viz else "",
        "metrics": metrics,
        "retry_count": int(retry_count),
    }


def _write_failed_sample_artifacts(
    *,
    output_dir: Path,
    sample: EvalSample,
    events: list[dict[str, Any]],
    error: Exception,
    retry_count: int = 0,
    failure_stage: str = "",
) -> dict[str, Any]:
    sample_dir = output_dir / "samples" / sample.source_dataset / sample.split / _sample_slug(sample)
    sample_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "row_id": sample.row_id,
        "source_dataset": sample.source_dataset,
        "split": sample.split,
        "source_image_path": sample.request.image_path,
        "request": sample.request.to_payload(),
        "tasks": events,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "source_metadata": sample.source_metadata,
        "retry_count": int(retry_count),
        "failure_stage": str(failure_stage or "").strip(),
    }
    common.write_json(sample_dir / "error.json", payload)
    common.write_json(sample_dir / "tasks.json", {"tasks": events})
    task_lines = [
        f"Row: {sample.row_id}",
        f"Source image: {sample.request.image_path}",
        f"Source dataset: {sample.source_dataset}",
        f"Split: {sample.split}",
        f"Error: {type(error).__name__}: {error}",
        "",
    ]
    for index, event in enumerate(events, start=1):
        response_text = str(event.get("answer_text") or "").strip()
        task_lines.extend(
            [
                f"Task {index}: {event.get('task', 'unknown')}",
                f"Prompt: {event.get('prompt', '')}",
            ]
        )
        if response_text:
            task_lines.append(f"Response: {response_text}")
        else:
            task_lines.append(f"Response: {json.dumps(event.get('parsed_response') or event.get('raw_response') or {}, ensure_ascii=False)}")
        if event.get("parsed_response") is not None:
            task_lines.append(f"Parsed: {json.dumps(event.get('parsed_response') or {}, ensure_ascii=False)}")
        if event.get("grader_response") is not None:
            task_lines.append(f"Grader: {json.dumps(event.get('grader_response') or {}, ensure_ascii=False)}")
        if event.get("judge_output") is not None:
            task_lines.append(f"Judge: {json.dumps(event.get('judge_output') or {}, ensure_ascii=False)}")
        if event.get("judge_score") is not None:
            task_lines.append(f"Judge score: {event.get('judge_score')}")
        task_lines.append("")
    common.write_text(sample_dir / "tasks.md", "\n".join(task_lines).rstrip() + "\n")
    return {
        "status": "error",
        "row_id": sample.row_id,
        "source_dataset": sample.source_dataset,
        "split": sample.split,
        "sample_dir": str(sample_dir),
        "error_json": str(sample_dir / "error.json"),
        "tasks_md": str(sample_dir / "tasks.md"),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "retry_count": int(retry_count),
        "failure_stage": str(failure_stage or "").strip(),
    }


def _proposal_event(events: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    for event in events:
        if str(event.get("task") or "") == "proposal_query":
            return event
    return None


def run_baseline_eval(args: argparse.Namespace) -> dict[str, Any]:
    common.maybe_load_env_file(args.env_file, override=False)
    api_key_pool = common.resolve_api_key_pool(explicit_api_key=args.api_key, api_key_env_vars=args.api_key_env_vars)
    excluded_source_datasets = {str(item) for item in list(args.detect_exclude_source_datasets or []) if str(item)}
    samples = _filter_samples(
        _load_samples(Path(args.dataset_manifest)),
        split=str(args.split),
        seed=int(args.seed),
        max_samples=int(args.max_samples),
        excluded_source_datasets=excluded_source_datasets,
    )
    print(f"preflight base_url={args.base_url} key_slots={api_key_pool.env_var_names or ['<explicit>']}")
    grader: Optional[openrouter_grader.OpenRouterGrader] = None
    if str(args.query_response_mode).strip().lower() == "text" or str(args.grading_mode).strip().lower() == "openrouter":
        grader_api_key = openrouter_grader.resolve_openrouter_api_key(
            explicit_api_key=args.grader_api_key,
            api_key_env_var=args.grader_api_key_env_var,
        )
        grader = openrouter_grader.OpenRouterGrader(
            api_key=grader_api_key,
            model_id=args.grader_model_id,
            api_base=args.grader_api_base,
            timeout=float(args.grader_timeout),
            profile=args.grader_profile,
            rubric_version=args.grader_rubric_version,
        )
    client = TracingMoondreamInspectorClient(
        api_key_pool=api_key_pool,
        base_url=args.base_url,
        timeout=float(args.timeout),
        proposal_prompt_style=args.proposal_prompt_style,
        finding_prompt_style=args.finding_prompt_style,
        query_response_mode=args.query_response_mode,
        grader=grader,
    )
    pipeline = InspectorPipeline(
        client=client,
        models=PipelineModels(
            detect_model=_resolve_model(args.base_model, args.detect_finetune_id),
            query_model=_resolve_model(args.base_model, args.query_finetune_id),
            point_model=_resolve_model(args.base_model, args.detect_finetune_id),
        ),
        settings=PipelineSettings(
            detect_max_objects=int(args.detect_max_objects),
            reasoning=bool(args.reasoning),
            iou_merge_threshold=float(args.iou_threshold),
        ),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_rows: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, float]] = []
    failure_rows: list[dict[str, Any]] = []
    for sample in samples:
        retry_count = 0
        for attempt_index in range(2):
            client.start_sample(sample.row_id)
            stage = "pipeline"
            try:
                report = pipeline.run(sample.request)
                query_judge_details: Optional[dict[str, Any]] = None
                if grader is not None and str(args.grading_mode).strip().lower() == "openrouter":
                    stage = "proposal_judge"
                    proposal_event = _proposal_event(client.events)
                    if proposal_event is not None:
                        proposal_judgement = grader.grade_issues(
                            inspection_request=sample.request.inspection_request,
                            asset_context=sample.request.asset_context,
                            expected_issues=sample.expected_proposals,
                            answer_text=str(proposal_event.get("answer_text") or ""),
                        )
                        proposal_event["judge_output"] = proposal_judgement
                        proposal_event["judge_score"] = proposal_judgement["score"]
                    stage = "query_judge"
                    query_judge_details = _compute_query_judge_details(
                        sample=sample,
                        report=report,
                        events=client.events,
                        grader=grader,
                        coarse_iou_threshold=float(args.coarse_iou_threshold),
                    )
                metrics = _sample_metrics(
                    sample=sample,
                    report=report,
                    iou_threshold=float(args.iou_threshold),
                    coarse_iou_threshold=float(args.coarse_iou_threshold),
                    query_judge_details=query_judge_details,
                )
                stage = "artifact_write"
                sample_rows.append(
                    _write_sample_artifacts(
                        output_dir=output_dir,
                        sample=sample,
                        report=report,
                        events=list(client.events),
                        metrics=metrics,
                        save_viz=bool(args.save_viz),
                        max_viz_dim=int(args.max_viz_dim),
                        retry_count=retry_count,
                        judge_details=query_judge_details,
                    )
                )
                metrics_rows.append(metrics)
                break
            except (TimeoutError, socket.timeout) as exc:
                if attempt_index == 0:
                    retry_count = 1
                    continue
                failure = _write_failed_sample_artifacts(
                    output_dir=output_dir,
                    sample=sample,
                    events=list(client.events),
                    error=exc,
                    retry_count=retry_count,
                    failure_stage=_infer_failure_stage(stage=stage, events=client.events),
                )
                sample_rows.append(failure)
                failure_rows.append(failure)
                break
            except Exception as exc:
                failure = _write_failed_sample_artifacts(
                    output_dir=output_dir,
                    sample=sample,
                    events=list(client.events),
                    error=exc,
                    retry_count=retry_count,
                    failure_stage=_infer_failure_stage(stage=stage, events=client.events),
                )
                sample_rows.append(failure)
                failure_rows.append(failure)
                break

    summary = {
        "sample_count": len(samples),
        "successful_sample_count": len(metrics_rows),
        "failed_sample_count": len(failure_rows),
        "base_url": args.base_url,
        "api_key_env_vars": api_key_pool.env_var_names,
        "api_key_slot_count": len(api_key_pool.slots),
        "dataset_manifest": str(Path(args.dataset_manifest).resolve()),
        "split": args.split,
        "detect_exclude_source_datasets": sorted(excluded_source_datasets),
        "detect_model": _resolve_model(args.base_model, args.detect_finetune_id),
        "query_model": _resolve_model(args.base_model, args.query_finetune_id),
        "proposal_prompt_style": args.proposal_prompt_style,
        "finding_prompt_style": args.finding_prompt_style,
        "query_response_mode": args.query_response_mode,
        "grading_mode": args.grading_mode,
        "grader_model_id": args.grader_model_id if grader is not None else "",
        "grader_profile": args.grader_profile if grader is not None else "",
        "grader_rubric_version": args.grader_rubric_version if grader is not None else "",
        "proposal_precision": fmean(row["proposal_precision"] for row in metrics_rows) if metrics_rows else 0.0,
        "proposal_recall": fmean(row["proposal_recall"] for row in metrics_rows) if metrics_rows else 0.0,
        "proposal_f1": fmean(row["proposal_f1"] for row in metrics_rows) if metrics_rows else 0.0,
        "localization_precision": fmean(row["localization_precision"] for row in metrics_rows) if metrics_rows else 0.0,
        "localization_recall": fmean(row["localization_recall"] for row in metrics_rows) if metrics_rows else 0.0,
        "localization_f1": fmean(row["localization_f1"] for row in metrics_rows) if metrics_rows else 0.0,
        "localization_precision_iou_0_1": fmean(row["localization_precision_iou_0_1"] for row in metrics_rows) if metrics_rows else 0.0,
        "localization_recall_iou_0_1": fmean(row["localization_recall_iou_0_1"] for row in metrics_rows) if metrics_rows else 0.0,
        "localization_f1_iou_0_1": fmean(row["localization_f1_iou_0_1"] for row in metrics_rows) if metrics_rows else 0.0,
        "localization_precision_center_hit": fmean(row["localization_precision_center_hit"] for row in metrics_rows) if metrics_rows else 0.0,
        "localization_recall_center_hit": fmean(row["localization_recall_center_hit"] for row in metrics_rows) if metrics_rows else 0.0,
        "localization_f1_center_hit": fmean(row["localization_f1_center_hit"] for row in metrics_rows) if metrics_rows else 0.0,
        "mean_best_iou": fmean(row["mean_best_iou"] for row in metrics_rows) if metrics_rows else 0.0,
        "finding_schema_valid_rate": fmean(row["finding_schema_valid"] for row in metrics_rows) if metrics_rows else 0.0,
        "end_to_end_score": fmean(row["end_to_end_score"] for row in metrics_rows) if metrics_rows else 0.0,
        "proposal_judge_score": fmean(row["proposal_judge_score"] for row in metrics_rows) if metrics_rows else 0.0,
        "proposal_judge_yes_rate": fmean(row["proposal_judge_yes_rate"] for row in metrics_rows) if metrics_rows else 0.0,
        "proposal_judge_partial_rate": fmean(row["proposal_judge_partial_rate"] for row in metrics_rows) if metrics_rows else 0.0,
        "proposal_unsupported_claim_rate": fmean(row["proposal_unsupported_claim_rate"] for row in metrics_rows) if metrics_rows else 0.0,
        "proposal_verbose_rate": fmean(row["proposal_verbose_rate"] for row in metrics_rows) if metrics_rows else 0.0,
        "finding_judge_score": fmean(row["finding_judge_score"] for row in metrics_rows) if metrics_rows else 0.0,
        "finding_judge_yes_rate": fmean(row["finding_judge_yes_rate"] for row in metrics_rows) if metrics_rows else 0.0,
        "finding_judge_partial_rate": fmean(row["finding_judge_partial_rate"] for row in metrics_rows) if metrics_rows else 0.0,
        "finding_unsupported_claim_rate": fmean(row["finding_unsupported_claim_rate"] for row in metrics_rows) if metrics_rows else 0.0,
        "finding_verbose_rate": fmean(row["finding_verbose_rate"] for row in metrics_rows) if metrics_rows else 0.0,
        "query_judge_score": fmean(row["query_judge_score"] for row in metrics_rows) if metrics_rows else 0.0,
        "samples": sample_rows,
        "failures": failure_rows,
    }
    common.write_json(output_dir / "summary.json", summary)
    common.write_jsonl(output_dir / "records.jsonl", sample_rows)
    return summary


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    summary = run_baseline_eval(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
