#!/usr/bin/env python3
"""Benchmark Inspector MD detect localization directly from the source manifest."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from statistics import fmean
from typing import Any, Optional

try:
    from finetune_checkpoints import resolve_checkpoint_step
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from finetune_checkpoints import resolve_checkpoint_step

from inspector_md import common, ontology, task_schema
from inspector_md.benchmark_inspector_pipeline import (
    _center_hit_box,
    _load_expected_samples,
    _match_findings,
    _match_findings_by_center,
    _mean_best_iou,
)
from inspector_md.moondream_client import DetectAnnotation, MoondreamInspectorClient
from inspector_md.pipeline import box_iou

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = common.repo_relative("configs", "benchmark_inspector_detect_default.json")


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

    parser = argparse.ArgumentParser(description="Benchmark Inspector MD detect localization.")
    parser.add_argument("--config", default=str(config_path))
    parser.add_argument("--env-file", default=str(common.repo_relative(".env.staging")))
    parser.add_argument("--api-key", default="")
    parser.add_argument("--api-key-env-var", default=common.DEFAULT_API_KEY_ENV_VAR)
    parser.add_argument("--api-key-env-vars", nargs="+", default=list(common.DEFAULT_API_KEY_ENV_VARS))
    parser.add_argument("--base-url", "--api-base", dest="base_url", default=common.DEFAULT_BASE_URL)
    parser.add_argument("--dataset-manifest", default=str(common.repo_relative("dataset", "merged_synth_v1", "smoke_manifest.json")))
    parser.add_argument("--dataset-path", default="")
    parser.add_argument("--split", default="all")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--detect-finetune-id", "--finetune-id", dest="detect_finetune_id", default="")
    parser.add_argument("--checkpoint-step", type=int, default=None)
    parser.add_argument("--checkpoint-fallback-policy", choices=("nearest_saved", "exact"), default="nearest_saved")
    parser.add_argument("--checkpoint-ready-max-wait-s", type=float, default=300.0)
    parser.add_argument("--checkpoint-ready-poll-interval-s", type=float, default=5.0)
    parser.add_argument("--base-model", default=common.DEFAULT_BASE_MODEL)
    parser.add_argument("--detect-max-objects", "--max-objects", dest="detect_max_objects", type=int, default=24)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--coarse-iou-threshold", type=float, default=0.1)
    parser.add_argument("--detect-exclude-source-datasets", nargs="*", default=["CODEBRIM"])
    parser.add_argument("--output-json", "--out-json", dest="output_json", default=str(common.repo_relative("outputs", "benchmarks", "inspector_detect.metrics.json")))
    parser.add_argument("--predictions-jsonl", "--records-jsonl", dest="predictions_jsonl", default=str(common.repo_relative("outputs", "benchmarks", "inspector_detect.records.jsonl")))
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skill", default="detect")
    parser.add_argument("--point-prompt-style", default="detect_phrase")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--neg-prompts-per-empty", type=int, default=1)
    parser.add_argument("--neg-prompts-per-nonempty", type=int, default=1)
    parser.add_argument("--reasoning", action=argparse.BooleanOptionalAction, default=False)

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
    args.dataset_path = common.resolve_path(args.dataset_path, module_root=SCRIPT_DIR) if str(args.dataset_path or "").strip() else None
    args.output_json = common.resolve_path(args.output_json, module_root=SCRIPT_DIR)
    args.predictions_jsonl = common.resolve_path(args.predictions_jsonl, module_root=SCRIPT_DIR)
    if args.dataset_path is not None:
        candidate_manifest = Path(args.dataset_path).resolve().parent / "source_manifest.normalized.json"
        if candidate_manifest.exists():
            args.dataset_manifest = candidate_manifest
        else:
            raise ValueError(
                f"Unable to resolve source manifest from dataset path: {args.dataset_path}"
            )
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


def _finding_labels(finding: task_schema.Finding) -> list[str]:
    labels = list(finding.source_detect_labels) or list(ontology.detect_labels_for_issue(finding.issue_code))
    deduped: list[str] = []
    seen: set[str] = set()
    for label in labels:
        value = str(label).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _prediction_from_box(
    *,
    finding: task_schema.Finding,
    label: str,
    box: DetectAnnotation,
    index: int,
) -> task_schema.Finding:
    issue = ontology.get_issue(finding.issue_code)
    return task_schema.Finding.from_payload(
        {
            "finding_id": f"{finding.finding_id or 'finding'}_{index:03d}",
            "issue_code": ontology.issue_code_for_detect_label(label) or finding.issue_code,
            "title": issue.title,
            "box": {
                "x_min": box.x_min,
                "y_min": box.y_min,
                "x_max": box.x_max,
                "y_max": box.y_max,
            },
            "evidence": [],
            "severity": "unknown",
            "recommended_action": issue.default_recommended_action,
            "cost_band": issue.default_cost_band,
            "possible_compliance_issue": False,
            "insufficient_evidence": False,
            "compliance_note": "",
            "source_detect_labels": [label],
            "spatial_ref_index": 0,
        }
    )


def _best_box_details(finding: task_schema.Finding, predicted: list[task_schema.Finding]) -> dict[str, Any]:
    best_iou = 0.0
    best_center_hit = False
    best_label = ""
    best_box_payload: dict[str, Any] = {}
    for item in predicted:
        if ontology.issue_match_score(finding.issue_code, item.issue_code) <= 0.0:
            continue
        score = round(box_iou(item.box, finding.box), 6)
        center_hit = _center_hit_box(finding.box, item.box)
        if score > best_iou or (score == best_iou and center_hit and not best_center_hit):
            best_iou = score
            best_center_hit = center_hit
            best_label = next(iter(item.source_detect_labels), "")
            best_box_payload = item.box.to_payload()
    return {
        "best_iou": best_iou,
        "center_hit": best_center_hit,
        "best_label": best_label,
        "best_box": best_box_payload,
    }


def _empty_summary_bucket() -> dict[str, float]:
    return {
        "count": 0.0,
        "recall_iou_0_5": 0.0,
        "recall_iou_0_1": 0.0,
        "center_hit_rate": 0.0,
        "mean_best_iou": 0.0,
        "mean_box_count": 0.0,
    }


def _update_bucket(bucket: dict[str, float], *, record: dict[str, Any]) -> None:
    bucket["count"] += 1.0
    bucket["recall_iou_0_5"] += float(record["matched_iou_0_5"])
    bucket["recall_iou_0_1"] += float(record["matched_iou_0_1"])
    bucket["center_hit_rate"] += float(record["center_hit"])
    bucket["mean_best_iou"] += float(record["best_iou"])
    bucket["mean_box_count"] += float(record["predicted_box_count"])


def _finalize_bucket(bucket: dict[str, float]) -> dict[str, float]:
    count = float(bucket.get("count", 0.0))
    if count <= 0.0:
        return dict(bucket)
    return {
        "count": count,
        "recall_iou_0_5": bucket["recall_iou_0_5"] / count,
        "recall_iou_0_1": bucket["recall_iou_0_1"] / count,
        "center_hit_rate": bucket["center_hit_rate"] / count,
        "mean_best_iou": bucket["mean_best_iou"] / count,
        "mean_box_count": bucket["mean_box_count"] / count,
    }


def _micro_f1_from_counts(tp: int, fp: int, fn: int) -> float:
    denom = (2 * int(tp)) + int(fp) + int(fn)
    if denom == 0:
        return 1.0
    return (2.0 * float(tp)) / float(denom)


def _filtered_samples(args: argparse.Namespace):
    excluded = {str(item) for item in list(args.detect_exclude_source_datasets or []) if str(item)}
    samples = [
        sample
        for sample in _load_expected_samples(Path(args.dataset_manifest))
        if (args.split == "all" or sample.split == args.split)
        and sample.source_dataset not in excluded
        and sample.expected_findings
    ]
    random.Random(args.seed).shuffle(samples)
    if int(args.max_samples) > 0:
        samples = samples[: int(args.max_samples)]
    return samples, excluded


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    common.maybe_load_env_file(args.env_file, override=False)
    api_key_pool = common.resolve_api_key_pool(explicit_api_key=args.api_key, api_key_env_vars=args.api_key_env_vars)
    samples, excluded_source_datasets = _filtered_samples(args)
    detect_model, resolved_checkpoint_step, used_checkpoint_fallback = _resolve_model(
        args.base_model,
        args.detect_finetune_id,
        checkpoint_step=args.checkpoint_step,
        api_base=args.base_url,
        api_key=api_key_pool.slots[0].api_key,
        checkpoint_fallback_policy=args.checkpoint_fallback_policy,
        checkpoint_ready_max_wait_s=args.checkpoint_ready_max_wait_s,
        checkpoint_ready_poll_interval_s=args.checkpoint_ready_poll_interval_s,
    )
    print(f"preflight base_url={args.base_url} key_slots={api_key_pool.env_var_names or ['<explicit>']}")
    client = MoondreamInspectorClient(api_key_pool=api_key_pool, base_url=args.base_url)

    records: list[dict[str, Any]] = []
    issue_buckets: dict[str, dict[str, float]] = {}
    label_buckets: dict[str, dict[str, float]] = {}

    for sample in samples:
        for finding in sample.expected_findings:
            label_results: list[dict[str, Any]] = []
            predicted_findings: list[task_schema.Finding] = []
            total_box_count = 0
            for label in _finding_labels(finding):
                boxes = client.detect_boxes(
                    model=detect_model,
                    image_path=sample.request.image_path,
                    detect_label=label,
                    max_objects=int(args.detect_max_objects),
                )
                total_box_count += len(boxes)
                label_predicted = [
                    _prediction_from_box(finding=finding, label=label, box=box, index=index)
                    for index, box in enumerate(boxes, start=1)
                ]
                predicted_findings.extend(label_predicted)
                best = _best_box_details(finding, label_predicted)
                label_record = {
                    "label": label,
                    "predicted_box_count": len(label_predicted),
                    "matched_iou_0_5": 1.0 if best["best_iou"] >= float(args.iou_threshold) else 0.0,
                    "matched_iou_0_1": 1.0 if best["best_iou"] >= float(args.coarse_iou_threshold) else 0.0,
                    "center_hit": 1.0 if best["center_hit"] else 0.0,
                    "best_iou": best["best_iou"],
                    "best_box": best["best_box"],
                }
                label_results.append(label_record)
                label_bucket = label_buckets.get(label)
                if label_bucket is None:
                    label_buckets[label] = _empty_summary_bucket()
                _update_bucket(label_buckets[label], record=label_record)

            strict_matches = _match_findings([finding], predicted_findings, iou_threshold=float(args.iou_threshold))
            loose_matches = _match_findings([finding], predicted_findings, iou_threshold=float(args.coarse_iou_threshold))
            center_matches = _match_findings_by_center([finding], predicted_findings)
            best_iou = _mean_best_iou([finding], predicted_findings)
            center_hit = bool(center_matches)
            record = {
                "row_id": sample.row_id,
                "split": sample.split,
                "source_dataset": sample.source_dataset,
                "image_path": sample.request.image_path,
                "finding_id": finding.finding_id,
                "issue_code": finding.issue_code,
                "title": finding.title,
                "source_detect_labels": _finding_labels(finding),
                "predicted_box_count": total_box_count,
                "matched_iou_0_5": 1.0 if strict_matches else 0.0,
                "matched_iou_0_1": 1.0 if loose_matches else 0.0,
                "center_hit": 1.0 if center_hit else 0.0,
                "best_iou": best_iou,
                "label_results": label_results,
            }
            records.append(record)
            _update_bucket(issue_buckets.setdefault(finding.issue_code, _empty_summary_bucket()), record=record)

    args.predictions_jsonl.parent.mkdir(parents=True, exist_ok=True)
    common.write_jsonl(args.predictions_jsonl, records)
    eval_tp = int(sum(int(record["matched_iou_0_5"]) for record in records))
    eval_fn = int(sum(1 - int(record["matched_iou_0_5"]) for record in records))
    eval_fp = int(sum(max(0, int(record["predicted_box_count"]) - int(record["matched_iou_0_5"])) for record in records))
    eval_f1 = _micro_f1_from_counts(eval_tp, eval_fp, eval_fn)
    eval_f1_macro = (
        fmean(
            _micro_f1_from_counts(
                int(record["matched_iou_0_5"]),
                max(0, int(record["predicted_box_count"]) - int(record["matched_iou_0_5"])),
                1 - int(record["matched_iou_0_5"]),
            )
            for record in records
        )
        if records
        else 0.0
    )
    summary = {
        "sample_count": len(samples),
        "finding_count": len(records),
        "base_url": args.base_url,
        "api_key_env_vars": api_key_pool.env_var_names,
        "api_key_slot_count": len(api_key_pool.slots),
        "dataset_manifest": str(Path(args.dataset_manifest).resolve()),
        "split": args.split,
        "detect_exclude_source_datasets": sorted(excluded_source_datasets),
        "detect_model": detect_model,
        "checkpoint_step": None if args.checkpoint_step is None else int(args.checkpoint_step),
        "resolved_checkpoint_step": resolved_checkpoint_step,
        "used_checkpoint_fallback": bool(used_checkpoint_fallback),
        "iou_threshold": float(args.iou_threshold),
        "coarse_iou_threshold": float(args.coarse_iou_threshold),
        "eval_tasks": len(records),
        "eval_f1": eval_f1,
        "eval_f1_macro": eval_f1_macro,
        "eval_miou": fmean(float(record["best_iou"]) for record in records) if records else 0.0,
        "eval_tp": eval_tp,
        "eval_fp": eval_fp,
        "eval_fn": eval_fn,
        "eval_positive_tasks": len(records),
        "eval_positive_f1": eval_f1,
        "eval_positive_f1_macro": eval_f1_macro,
        "eval_positive_tp": eval_tp,
        "eval_positive_fp": eval_fp,
        "eval_positive_fn": eval_fn,
        "eval_negative_tasks": 0,
        "eval_negative_f1": 0.0,
        "eval_negative_f1_macro": 0.0,
        "eval_negative_tp": 0,
        "eval_negative_fp": 0,
        "eval_negative_fn": 0,
        "eval_target_min_positive_tasks": len(records),
        "eval_positive_task_shortfall": 0,
        "recall_iou_0_5": fmean(float(record["matched_iou_0_5"]) for record in records) if records else 0.0,
        "recall_iou_0_1": fmean(float(record["matched_iou_0_1"]) for record in records) if records else 0.0,
        "center_hit_rate": fmean(float(record["center_hit"]) for record in records) if records else 0.0,
        "mean_best_iou": fmean(float(record["best_iou"]) for record in records) if records else 0.0,
        "per_issue_code": {key: _finalize_bucket(value) for key, value in sorted(issue_buckets.items())},
        "per_detect_label": {key: _finalize_bucket(value) for key, value in sorted(label_buckets.items())},
        "predictions_jsonl": str(args.predictions_jsonl),
    }
    common.write_json(args.output_json, summary)
    return summary


def main(argv: Optional[list[str]] = None) -> None:
    summary = run_benchmark(parse_args(argv))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
