#!/usr/bin/env python3
"""Benchmark the Inspector MD end-to-end runtime pipeline."""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Optional

from finetune_checkpoints import resolve_checkpoint_step

from inspector_md import common, ontology, openrouter_grader, task_schema
from inspector_md.moondream_client import MoondreamInspectorClient
from inspector_md.pipeline import InspectorPipeline, PipelineModels, PipelineSettings, box_iou

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = common.repo_relative("configs", "benchmark_inspector_pipeline_default.json")


@dataclass(frozen=True)
class ExpectedSample:
    row_id: str
    split: str
    source_dataset: str
    request: task_schema.InspectionRequest
    expected_proposals: list[task_schema.IssueProposal]
    expected_findings: list[task_schema.Finding]


def _resolve_model(
    base_model: str,
    finetune_id: str,
    *,
    checkpoint_step: Optional[int] = None,
    api_base: str = common.DEFAULT_BASE_URL,
    api_key: str = "",
    checkpoint_fallback_policy: str = "nearest_saved",
    checkpoint_ready_max_wait_s: float = 300.0,
    checkpoint_ready_poll_interval_s: float = 5.0,
) -> tuple[str, Optional[int], bool]:
    finetune = str(finetune_id or "").strip()
    if not finetune:
        return str(base_model).strip(), None, False
    if checkpoint_step is None:
        return f"{base_model}/{finetune}", None, False
    resolved_checkpoint_step, used_fallback = resolve_checkpoint_step(
        api_base=str(api_base),
        api_key=str(api_key),
        finetune_id=finetune,
        requested_step=int(checkpoint_step),
        fallback_policy=str(checkpoint_fallback_policy),
        ready_max_wait_s=float(checkpoint_ready_max_wait_s),
        ready_poll_interval_s=float(checkpoint_ready_poll_interval_s),
    )
    return f"{base_model}/{finetune}@{int(resolved_checkpoint_step)}", int(resolved_checkpoint_step), bool(used_fallback)


def _load_expected_samples(manifest_path: Path) -> list[ExpectedSample]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("benchmark manifest must be a JSON array")
    samples: list[ExpectedSample] = []
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
        expected_proposals = task_schema.normalize_issue_proposals(item.get("expected_proposals") or [])
        expected_findings = task_schema.normalize_findings(item.get("expected_findings") or [])
        samples.append(
                ExpectedSample(
                    row_id=str(item.get("row_id") or request.image_path),
                    split=str(item.get("split") or "test"),
                    source_dataset=str(item.get("source_dataset") or item.get("source_metadata", {}).get("source_dataset") or "unknown"),
                    request=request,
                    expected_proposals=expected_proposals,
                    expected_findings=expected_findings,
                )
        )
    return samples


def _proposal_metrics(expected: list[task_schema.IssueProposal], predicted: list[task_schema.IssueProposal]) -> dict[str, float]:
    remaining_pred = list(predicted)
    match_score = 0.0
    for expected_item in expected:
        best_index = -1
        best_score = 0.0
        for index, predicted_item in enumerate(remaining_pred):
            score = ontology.issue_match_score(expected_item.issue_code, predicted_item.issue_code)
            if score > best_score:
                best_score = score
                best_index = index
        if best_index >= 0 and best_score > 0.0:
            match_score += best_score
            remaining_pred.pop(best_index)
    precision = match_score / float(max(1, len(predicted)))
    recall = match_score / float(max(1, len(expected)))
    denom = precision + recall
    f1 = 0.0 if denom <= 0.0 else (2.0 * precision * recall) / denom
    return {"precision": precision, "recall": recall, "f1": f1}


def _match_findings(
    expected: list[task_schema.Finding],
    predicted: list[task_schema.Finding],
    *,
    iou_threshold: float,
) -> list[tuple[task_schema.Finding, task_schema.Finding, float]]:
    matches: list[tuple[task_schema.Finding, task_schema.Finding, float]] = []
    remaining_pred = list(predicted)
    for gt in expected:
        best_index = -1
        best_issue_score = 0.0
        best_iou = -1.0
        for index, pred in enumerate(remaining_pred):
            issue_score = ontology.issue_match_score(gt.issue_code, pred.issue_code)
            if issue_score <= 0.0:
                continue
            score = box_iou(pred.box, gt.box)
            if score < float(iou_threshold):
                continue
            if issue_score > best_issue_score or (issue_score == best_issue_score and score > best_iou):
                best_issue_score = issue_score
                best_iou = score
                best_index = index
        if best_index >= 0:
            matches.append((gt, remaining_pred.pop(best_index), best_issue_score))
    return matches


def _center_hit_box(box_a: task_schema.Box, box_b: task_schema.Box) -> bool:
    ax = (box_a.x_min + box_a.x_max) / 2.0
    ay = (box_a.y_min + box_a.y_max) / 2.0
    bx = (box_b.x_min + box_b.x_max) / 2.0
    by = (box_b.y_min + box_b.y_max) / 2.0
    a_in_b = box_b.x_min <= ax <= box_b.x_max and box_b.y_min <= ay <= box_b.y_max
    b_in_a = box_a.x_min <= bx <= box_a.x_max and box_a.y_min <= by <= box_a.y_max
    return bool(a_in_b or b_in_a)


def _match_findings_by_center(
    expected: list[task_schema.Finding],
    predicted: list[task_schema.Finding],
) -> list[tuple[task_schema.Finding, task_schema.Finding, float]]:
    matches: list[tuple[task_schema.Finding, task_schema.Finding, float]] = []
    remaining_pred = list(predicted)
    for gt in expected:
        best_index = -1
        best_issue_score = 0.0
        best_iou = -1.0
        for index, pred in enumerate(remaining_pred):
            issue_score = ontology.issue_match_score(gt.issue_code, pred.issue_code)
            if issue_score <= 0.0:
                continue
            if not _center_hit_box(gt.box, pred.box):
                continue
            score = box_iou(pred.box, gt.box)
            if issue_score > best_issue_score or (issue_score == best_issue_score and score > best_iou):
                best_issue_score = issue_score
                best_iou = score
                best_index = index
        if best_index >= 0:
            matches.append((gt, remaining_pred.pop(best_index), best_issue_score))
    return matches


def _localization_metrics_from_matches(
    *,
    expected: list[task_schema.Finding],
    predicted: list[task_schema.Finding],
    matches: list[tuple[task_schema.Finding, task_schema.Finding, float]],
) -> dict[str, float]:
    match_score = sum(issue_score for _, _, issue_score in matches)
    precision = match_score / float(max(1, len(predicted)))
    recall = match_score / float(max(1, len(expected)))
    denom = precision + recall
    f1 = 0.0 if denom <= 0.0 else (2.0 * precision * recall) / denom
    return {"precision": precision, "recall": recall, "f1": f1}


def _mean_best_iou(expected: list[task_schema.Finding], predicted: list[task_schema.Finding]) -> float:
    if not expected:
        return 0.0
    best_scores: list[float] = []
    for gt in expected:
        best_score = 0.0
        for pred in predicted:
            if ontology.issue_match_score(gt.issue_code, pred.issue_code) <= 0.0:
                continue
            best_score = max(best_score, box_iou(pred.box, gt.box))
        best_scores.append(best_score)
    return fmean(best_scores) if best_scores else 0.0


def _finding_pair_score(expected: task_schema.Finding, predicted: task_schema.Finding) -> float:
    issue_match = ontology.issue_match_score(expected.issue_code, predicted.issue_code)
    evidence = common.token_f1(" ".join(expected.evidence), " ".join(predicted.evidence))
    return fmean([issue_match, evidence])


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

    parser = argparse.ArgumentParser(description="Benchmark the Inspector MD runtime pipeline.")
    parser.add_argument("--config", default=str(config_path))
    parser.add_argument("--env-file", default=str(common.repo_relative(".env.staging")))
    parser.add_argument("--api-key", default="")
    parser.add_argument("--api-key-env-var", default=common.DEFAULT_API_KEY_ENV_VAR)
    parser.add_argument("--api-key-env-vars", nargs="+", default=list(common.DEFAULT_API_KEY_ENV_VARS))
    parser.add_argument("--base-url", default=common.DEFAULT_BASE_URL)
    parser.add_argument("--dataset-manifest", default=str(common.repo_relative("outputs", "source_manifest.normalized.json")))
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--detect-finetune-id", default="")
    parser.add_argument("--query-finetune-id", default="")
    parser.add_argument("--checkpoint-step", type=int, default=None)
    parser.add_argument("--checkpoint-fallback-policy", choices=("nearest_saved", "exact"), default="nearest_saved")
    parser.add_argument("--checkpoint-ready-max-wait-s", type=float, default=300.0)
    parser.add_argument("--checkpoint-ready-poll-interval-s", type=float, default=5.0)
    parser.add_argument("--base-model", default=common.DEFAULT_BASE_MODEL)
    parser.add_argument("--reasoning", action="store_true")
    parser.add_argument("--proposal-prompt-style", choices=("structured", "request_only"), default="request_only")
    parser.add_argument("--finding-prompt-style", choices=("structured", "minimal"), default="minimal")
    parser.add_argument("--query-normalization-mode", choices=("local_only", "openrouter_fallback"), default="local_only")
    parser.add_argument("--grader-api-key", default="")
    parser.add_argument("--grader-api-key-env-var", default=openrouter_grader.DEFAULT_OPENROUTER_ENV_VAR)
    parser.add_argument("--grader-api-base", default=openrouter_grader.DEFAULT_OPENROUTER_API_BASE)
    parser.add_argument("--grader-model-id", default=openrouter_grader.DEFAULT_GRADER_MODEL)
    parser.add_argument("--grader-profile", default=openrouter_grader.DEFAULT_GRADER_PROFILE)
    parser.add_argument("--grader-rubric-version", default=openrouter_grader.DEFAULT_GRADER_RUBRIC_VERSION)
    parser.add_argument("--grader-timeout", type=float, default=60.0)
    parser.add_argument("--detect-max-objects", type=int, default=24)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--coarse-iou-threshold", type=float, default=0.1)
    parser.add_argument("--detect-exclude-source-datasets", nargs="*", default=["CODEBRIM"])
    parser.add_argument("--output-json", default=str(common.repo_relative("outputs", "benchmarks", "inspector_pipeline.metrics.json")))
    parser.add_argument("--predictions-jsonl", default=str(common.repo_relative("outputs", "benchmarks", "inspector_pipeline.predictions.jsonl")))

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
    args.output_json = common.resolve_path(args.output_json, module_root=SCRIPT_DIR)
    args.predictions_jsonl = common.resolve_path(args.predictions_jsonl, module_root=SCRIPT_DIR)
    if single_key_requested and not multi_key_requested:
        args.api_key_env_vars = [args.api_key_env_var]
    else:
        args.api_key_env_vars = common.normalize_api_key_env_vars(args.api_key_env_vars)
    args.detect_exclude_source_datasets = [str(item).strip() for item in list(args.detect_exclude_source_datasets or []) if str(item).strip()]
    return args


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    common.maybe_load_env_file(args.env_file, override=False)
    api_key_pool = common.resolve_api_key_pool(explicit_api_key=args.api_key, api_key_env_vars=args.api_key_env_vars)
    detect_model, resolved_detect_checkpoint_step, used_detect_checkpoint_fallback = _resolve_model(
        args.base_model,
        args.detect_finetune_id,
        checkpoint_step=args.checkpoint_step if str(args.detect_finetune_id or "").strip() else None,
        api_base=args.base_url,
        api_key=api_key_pool.slots[0].api_key,
        checkpoint_fallback_policy=args.checkpoint_fallback_policy,
        checkpoint_ready_max_wait_s=args.checkpoint_ready_max_wait_s,
        checkpoint_ready_poll_interval_s=args.checkpoint_ready_poll_interval_s,
    )
    query_model, resolved_query_checkpoint_step, used_query_checkpoint_fallback = _resolve_model(
        args.base_model,
        args.query_finetune_id,
        checkpoint_step=args.checkpoint_step if str(args.query_finetune_id or "").strip() else None,
        api_base=args.base_url,
        api_key=api_key_pool.slots[0].api_key,
        checkpoint_fallback_policy=args.checkpoint_fallback_policy,
        checkpoint_ready_max_wait_s=args.checkpoint_ready_max_wait_s,
        checkpoint_ready_poll_interval_s=args.checkpoint_ready_poll_interval_s,
    )
    excluded_source_datasets = {str(item) for item in list(args.detect_exclude_source_datasets or []) if str(item)}
    samples = [
        sample
        for sample in _load_expected_samples(Path(args.dataset_manifest))
        if sample.split == args.split and sample.source_dataset not in excluded_source_datasets
    ]
    random.Random(args.seed).shuffle(samples)
    if int(args.max_samples) > 0:
        samples = samples[: int(args.max_samples)]
    print(f"preflight base_url={args.base_url} key_slots={api_key_pool.env_var_names or ['<explicit>']}")
    client = MoondreamInspectorClient(api_key_pool=api_key_pool, base_url=args.base_url)
    normalizer = None
    if str(args.query_normalization_mode).strip().lower() == "openrouter_fallback":
        grader_api_key = openrouter_grader.resolve_openrouter_api_key(
            explicit_api_key=args.grader_api_key,
            api_key_env_var=args.grader_api_key_env_var,
        )
        normalizer = openrouter_grader.OpenRouterGrader(
            api_key=grader_api_key,
            model_id=args.grader_model_id,
            api_base=args.grader_api_base,
            timeout=float(args.grader_timeout),
            profile=args.grader_profile,
            rubric_version=args.grader_rubric_version,
        )
    pipeline = InspectorPipeline(
        client=client,
        models=PipelineModels(
            detect_model=detect_model,
            query_model=query_model,
            point_model=detect_model,
        ),
        settings=PipelineSettings(
            detect_max_objects=int(args.detect_max_objects),
            reasoning=bool(args.reasoning),
            iou_merge_threshold=float(args.iou_threshold),
            proposal_prompt_style=str(args.proposal_prompt_style),
            finding_prompt_style=str(args.finding_prompt_style),
        ),
        normalizer=normalizer,
    )
    issue_precision_values: list[float] = []
    issue_recall_values: list[float] = []
    issue_f1_values: list[float] = []
    localization_precision_values: list[float] = []
    localization_recall_values: list[float] = []
    localization_f1_values: list[float] = []
    localization_precision_iou_0_1_values: list[float] = []
    localization_recall_iou_0_1_values: list[float] = []
    localization_f1_iou_0_1_values: list[float] = []
    localization_precision_center_hit_values: list[float] = []
    localization_recall_center_hit_values: list[float] = []
    localization_f1_center_hit_values: list[float] = []
    best_iou_values: list[float] = []
    finding_schema_valid: list[float] = []
    end_to_end_scores: list[float] = []
    args.predictions_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with Path(args.predictions_jsonl).open("w", encoding="utf-8") as handle:
        for sample in samples:
            report = pipeline.run(sample.request)
            predicted_proposals = task_schema.normalize_issue_proposals(report.trace.get("proposals") or [])
            issue_metrics = _proposal_metrics(sample.expected_proposals, predicted_proposals)
            issue_precision_values.append(issue_metrics["precision"])
            issue_recall_values.append(issue_metrics["recall"])
            issue_f1_values.append(issue_metrics["f1"])
            strict_matches = _match_findings(sample.expected_findings, report.findings, iou_threshold=float(args.iou_threshold))
            strict_metrics = _localization_metrics_from_matches(
                expected=sample.expected_findings,
                predicted=report.findings,
                matches=strict_matches,
            )
            loose_matches = _match_findings(
                sample.expected_findings,
                report.findings,
                iou_threshold=float(args.coarse_iou_threshold),
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
            localization_precision_values.append(strict_metrics["precision"])
            localization_recall_values.append(strict_metrics["recall"])
            localization_f1_values.append(strict_metrics["f1"])
            localization_precision_iou_0_1_values.append(loose_metrics["precision"])
            localization_recall_iou_0_1_values.append(loose_metrics["recall"])
            localization_f1_iou_0_1_values.append(loose_metrics["f1"])
            localization_precision_center_hit_values.append(center_metrics["precision"])
            localization_recall_center_hit_values.append(center_metrics["recall"])
            localization_f1_center_hit_values.append(center_metrics["f1"])
            best_iou_values.append(_mean_best_iou(sample.expected_findings, report.findings))
            finding_schema_valid.append(1.0 if all(isinstance(item, task_schema.Finding) for item in report.findings) else 0.0)
            pair_scores = [_finding_pair_score(gt, pred) for gt, pred, _ in strict_matches]
            if sample.expected_findings:
                pair_scores.extend([0.0] * max(0, len(sample.expected_findings) - len(strict_matches)))
            end_to_end_scores.append(fmean(pair_scores) if pair_scores else 0.0)
            handle.write(
                json.dumps(
                    {
                        "row_id": sample.row_id,
                        "split": sample.split,
                        "issue_metrics": issue_metrics,
                        "localization_f1": strict_metrics["f1"],
                        "localization_f1_iou_0_1": loose_metrics["f1"],
                        "localization_f1_center_hit": center_metrics["f1"],
                        "mean_best_iou": best_iou_values[-1],
                        "end_to_end_score": end_to_end_scores[-1],
                        "report": report.to_payload(),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    metrics = {
        "sample_count": len(samples),
        "base_url": args.base_url,
        "api_key_env_vars": api_key_pool.env_var_names,
        "api_key_slot_count": len(api_key_pool.slots),
        "detect_exclude_source_datasets": sorted(excluded_source_datasets),
        "checkpoint_step": None if args.checkpoint_step is None else int(args.checkpoint_step),
        "resolved_detect_checkpoint_step": resolved_detect_checkpoint_step,
        "used_detect_checkpoint_fallback": bool(used_detect_checkpoint_fallback),
        "resolved_query_checkpoint_step": resolved_query_checkpoint_step,
        "used_query_checkpoint_fallback": bool(used_query_checkpoint_fallback),
        "query_normalization_mode": str(args.query_normalization_mode),
        "grader_model_id": normalizer.model_id if normalizer is not None else "",
        "issue_precision": fmean(issue_precision_values) if issue_precision_values else 0.0,
        "issue_recall": fmean(issue_recall_values) if issue_recall_values else 0.0,
        "issue_f1": fmean(issue_f1_values) if issue_f1_values else 0.0,
        "localization_precision": fmean(localization_precision_values) if localization_precision_values else 0.0,
        "localization_recall": fmean(localization_recall_values) if localization_recall_values else 0.0,
        "localization_f1": fmean(localization_f1_values) if localization_f1_values else 0.0,
        "localization_precision_iou_0_1": fmean(localization_precision_iou_0_1_values) if localization_precision_iou_0_1_values else 0.0,
        "localization_recall_iou_0_1": fmean(localization_recall_iou_0_1_values) if localization_recall_iou_0_1_values else 0.0,
        "localization_f1_iou_0_1": fmean(localization_f1_iou_0_1_values) if localization_f1_iou_0_1_values else 0.0,
        "localization_precision_center_hit": fmean(localization_precision_center_hit_values) if localization_precision_center_hit_values else 0.0,
        "localization_recall_center_hit": fmean(localization_recall_center_hit_values) if localization_recall_center_hit_values else 0.0,
        "localization_f1_center_hit": fmean(localization_f1_center_hit_values) if localization_f1_center_hit_values else 0.0,
        "mean_best_iou": fmean(best_iou_values) if best_iou_values else 0.0,
        "finding_schema_valid_rate": fmean(finding_schema_valid) if finding_schema_valid else 0.0,
        "end_to_end_score": fmean(end_to_end_scores) if end_to_end_scores else 0.0,
        "predictions_jsonl": str(args.predictions_jsonl),
    }
    common.write_json(Path(args.output_json), metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
