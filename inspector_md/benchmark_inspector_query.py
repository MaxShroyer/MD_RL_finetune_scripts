#!/usr/bin/env python3
"""Benchmark Inspector MD query finetunes on full-image JSON issue-list tasks."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any, Optional

try:
    from finetune_checkpoints import resolve_checkpoint_step
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from finetune_checkpoints import resolve_checkpoint_step

from inspector_md import common, openrouter_grader
from inspector_md.moondream_client import MoondreamInspectorClient
from inspector_md.train_inspector_query import (
    _evaluate_split,
    _json_safe,
    _load_split_examples,
    _score_answer_text,
    wandb,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = common.repo_relative("configs", "benchmark_inspector_query_default.json")
_GENERIC_NO_ISSUE_MARKERS = (
    "no visible building or site issues",
    "no visible building issues",
    "no visible site issues",
    "no visible issues",
    "no issues visible",
    "no obvious issues",
    "no visible defects",
    "no defects visible",
    "no building or site issues",
    "does not show any building or site issues",
)
_PRIMARY_PARSE_METHODS = {"json", "compact_text"}
_HEURISTIC_PARSE_METHODS = {"heuristic_text", "heuristic_none"}


def _resolve_model(
    *,
    base_model: str,
    finetune_id: str,
    checkpoint_step: Optional[int],
    api_base: str,
    tuna_api_key: str,
    fallback_policy: str,
    ready_max_wait_s: float,
    ready_poll_interval_s: float,
) -> tuple[str, Optional[int], bool]:
    finetune = str(finetune_id or "").strip()
    if not finetune:
        return str(base_model).strip(), None, False
    if checkpoint_step is None:
        return f"{str(base_model).rstrip('/')}/{finetune}", None, False
    resolved_checkpoint_step, used_fallback = resolve_checkpoint_step(
        api_base=str(api_base),
        api_key=str(tuna_api_key),
        finetune_id=finetune,
        requested_step=int(checkpoint_step),
        fallback_policy=str(fallback_policy),
        ready_max_wait_s=float(ready_max_wait_s),
        ready_poll_interval_s=float(ready_poll_interval_s),
    )
    return f"{str(base_model).rstrip('/')}/{finetune}@{int(resolved_checkpoint_step)}", int(resolved_checkpoint_step), bool(used_fallback)


def _looks_like_generic_no_issue(answer_text: Any) -> bool:
    normalized = common.normalize_text(answer_text)
    return any(marker in normalized for marker in _GENERIC_NO_ISSUE_MARKERS)


def _judge_cache_path_for_predictions(predictions_jsonl: Path) -> Path:
    return predictions_jsonl.with_suffix(".judge_cache.jsonl")


def _checkpoint_eval_stagger_delay_s(*, finetune_id: str, checkpoint_step: Optional[int]) -> float:
    finetune = str(finetune_id or "").strip()
    if not finetune or checkpoint_step is None:
        return 0.0
    seed = f"{finetune}:{int(checkpoint_step)}"
    return random.Random(seed).uniform(0.0, 5.0)


def _build_benchmark_diagnostics(
    *,
    examples: list[Any],
    predictions_path: Path,
    metrics: dict[str, Any],
    task_type: str,
    answer_parse_mode: str,
) -> dict[str, Any]:
    predictions = [json.loads(line) for line in predictions_path.open("r", encoding="utf-8")]
    by_row_id = {str(example.row_id): example for example in examples}
    paired_rows = [
        (by_row_id[str(row["row_id"])], row)
        for row in predictions
        if str(row["row_id"]) in by_row_id
    ]
    strict_evaluations = [
        {
            "score_result": _score_answer_text(example, row.get("answer_text"), grader=None),
            "judge_score": float(row.get("judge_score", 0.0) or 0.0),
        }
        for example, row in paired_rows
    ]
    parse_method_values = [str(row.get("parse_method") or "").strip() or "unparsed" for row in predictions]
    strict_local_reward_values = [float(item["score_result"][0].reward) for item in strict_evaluations]
    strict_judge_score_values = [float(item["judge_score"]) for item in strict_evaluations]
    strict_reward_values = [
        (0.7 * judge_score) + (0.3 * local_reward)
        for local_reward, judge_score in zip(strict_local_reward_values, strict_judge_score_values, strict=False)
    ]
    diagnostics: dict[str, Any] = {
        "answer_parse_mode": str(answer_parse_mode),
        "comparison_safe_answer_parse_mode": bool(str(answer_parse_mode) == "strict_raw"),
        "response_json_rate": fmean(1.0 if method == "json" else 0.0 for method in parse_method_values) if predictions else 0.0,
        "response_compact_text_rate": fmean(1.0 if method == "compact_text" else 0.0 for method in parse_method_values) if predictions else 0.0,
        "response_normalized_rate": fmean(1.0 if method == "openrouter_normalize" else 0.0 for method in parse_method_values) if predictions else 0.0,
        "response_heuristic_rate": fmean(1.0 if method in _HEURISTIC_PARSE_METHODS else 0.0 for method in parse_method_values) if predictions else 0.0,
        "response_unparsed_rate": fmean(1.0 if method == "unparsed" else 0.0 for method in parse_method_values) if predictions else 0.0,
        "response_legacy_fallback_rate": fmean(
            1.0 if method not in (_PRIMARY_PARSE_METHODS | {"openrouter_normalize"}) else 0.0
            for method in parse_method_values
        ) if predictions else 0.0,
        "response_generic_no_issue_rate": fmean(
            1.0 if _looks_like_generic_no_issue(row.get("answer_text")) else 0.0 for row in predictions
        ) if predictions else 0.0,
        "response_parse_method_distribution": dict(Counter(parse_method_values)),
        "judge_cache_jsonl": str(_judge_cache_path_for_predictions(predictions_path)),
        "metric_warnings": [],
    }
    diagnostics["strict_reward_mean"] = fmean(strict_reward_values) if strict_reward_values else 0.0
    diagnostics["strict_local_reward_mean"] = fmean(strict_local_reward_values) if strict_local_reward_values else 0.0
    diagnostics["strict_judge_score_mean"] = fmean(strict_judge_score_values) if strict_judge_score_values else 0.0
    diagnostics["strict_parse_rate"] = fmean(
        1.0 if item["score_result"][0].parse_success else 0.0 for item in strict_evaluations
    ) if strict_evaluations else 0.0
    diagnostics["strict_issue_precision"] = fmean(
        item["score_result"][0].issue_precision for item in strict_evaluations
    ) if strict_evaluations else 0.0
    diagnostics["strict_issue_recall"] = fmean(
        item["score_result"][0].issue_recall for item in strict_evaluations
    ) if strict_evaluations else 0.0
    diagnostics["strict_issue_f1"] = fmean(
        item["score_result"][0].issue_f1 for item in strict_evaluations
    ) if strict_evaluations else 0.0
    diagnostics["strict_reasoning_f1"] = fmean(
        item["score_result"][0].reasoning_f1 for item in strict_evaluations
    ) if strict_evaluations else 0.0
    diagnostics["strict_extra_issue_rate"] = fmean(
        item["score_result"][0].extra_issue_rate for item in strict_evaluations
    ) if strict_evaluations else 0.0
    diagnostics["strict_empty_list_accuracy"] = fmean(
        item["score_result"][0].empty_list_accuracy for item in strict_evaluations
    ) if strict_evaluations else 0.0
    diagnostics["strict_task_correct_rate"] = fmean(
        1.0 if item["score_result"][0].task_correct else 0.0 for item in strict_evaluations
    ) if strict_evaluations else 0.0
    diagnostics["strict_parse_method_distribution"] = dict(
        Counter(str(item["score_result"][1].method or "").strip() or "unparsed" for item in strict_evaluations)
    )
    diagnostics["predicted_issue_count_distribution"] = dict(
        Counter(int(item["score_result"][0].predicted_issue_count) for item in strict_evaluations)
    )
    generic_no_issue_rows = [row for row in predictions if _looks_like_generic_no_issue(row.get("answer_text"))]
    diagnostics["generic_no_issue_count"] = len(generic_no_issue_rows)
    diagnostics["generic_no_issue_positive_judge_count"] = sum(
        1 for row in generic_no_issue_rows if float(row.get("judge_score", 0.0) or 0.0) > 0.0
    )
    diagnostics["generic_no_issue_perfect_judge_count"] = sum(
        1 for row in generic_no_issue_rows if float(row.get("judge_score", 0.0) or 0.0) >= 0.999
    )
    diagnostics["headline_metrics"] = {
        "reward_mean": float(metrics.get("reward_mean", 0.0) or 0.0),
        "local_reward_mean": float(metrics.get("local_reward_mean", 0.0) or 0.0),
        "issue_precision": float(metrics.get("issue_precision", 0.0) or 0.0),
        "issue_recall": float(metrics.get("issue_recall", 0.0) or 0.0),
        "issue_f1": float(metrics.get("issue_f1", 0.0) or 0.0),
        "reasoning_f1": float(metrics.get("reasoning_f1", 0.0) or 0.0),
        "judge_score_mean": float(metrics.get("judge_score_mean", 0.0) or 0.0),
        "task_correct_rate": float(metrics.get("task_correct_rate", 0.0) or 0.0),
        "extra_issue_rate": float(metrics.get("extra_issue_rate", 0.0) or 0.0),
        "predicted_issue_count_mean": float(metrics.get("predicted_issue_count_mean", 0.0) or 0.0),
        "response_json_rate": float(diagnostics["response_json_rate"]),
        "response_compact_text_rate": float(diagnostics["response_compact_text_rate"]),
        "strict_reward_mean": float(diagnostics["strict_reward_mean"]),
        "strict_issue_f1": float(diagnostics["strict_issue_f1"]),
        "response_generic_no_issue_rate": float(diagnostics["response_generic_no_issue_rate"]),
    }
    if float(metrics.get("inference_error_count", 0.0) or 0.0) > 0.0:
        diagnostics["metric_warnings"].append(
            "Some eval rows were skipped after repeated Moondream inference failures; metrics are computed on completed rows only."
        )
    if str(answer_parse_mode) != "strict_raw":
        diagnostics["metric_warnings"].append(
            "Primary query metrics allow grader-based answer normalization. Use `--answer-parse-mode strict_raw` for comparison-safe baseline vs finetune benchmarking."
        )
    if float(diagnostics["response_normalized_rate"]) > 0.0:
        diagnostics["metric_warnings"].append(
            "Some answers required `openrouter_normalize`; raw-answer metrics may be materially worse than the primary benchmark metrics."
        )
    if float(diagnostics["response_legacy_fallback_rate"]) > 0.0:
        diagnostics["metric_warnings"].append(
            "Some answers only parsed through heuristic or fallback paths. Prefer direct compact-text or JSON outputs."
        )
    if float(metrics.get("reward_mean", 0.0) or 0.0) - float(diagnostics["strict_reward_mean"]) > 0.05:
        diagnostics["metric_warnings"].append(
            "Primary reward diverges from strict raw-answer reward by more than 0.05. Inspect normalized or fallback parsing before comparing models."
        )
    if float(diagnostics["response_unparsed_rate"]) > 0.0:
        diagnostics["metric_warnings"].append(
            "Some raw answers were completely unparsed under strict scoring."
        )
    return diagnostics


def _write_benchmark_notes(path: Path, summary: dict[str, Any]) -> None:
    notes: list[str] = []
    notes.append(f"# Query Benchmark Notes: `{summary.get('model', '')}`")
    notes.append("")
    notes.append(f"- dataset_dir: `{summary.get('dataset_dir', '')}`")
    notes.append(f"- split: `{summary.get('split', '')}`")
    notes.append(f"- count: `{summary.get('count', 0)}`")
    notes.append(f"- answer_parse_mode: `{summary.get('answer_parse_mode', '')}`")
    notes.append("")
    warnings = list(summary.get("metric_warnings") or [])
    if warnings:
        notes.append("## Warnings")
        for item in warnings:
            notes.append(f"- {item}")
        notes.append("")
    omitted_metrics = list(summary.get("omitted_metrics") or [])
    if omitted_metrics:
        notes.append("## Omitted Metrics")
        for item in omitted_metrics:
            notes.append(f"- {item}")
        notes.append("")
    headline = dict(summary.get("headline_metrics") or {})
    if headline:
        notes.append("## Headline Metrics")
        for key, value in headline.items():
            notes.append(f"- {key}: `{value}`")
        notes.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(notes), encoding="utf-8")


def _wandb_metric_payload(summary: dict[str, Any]) -> dict[str, float]:
    payload: dict[str, float] = {}
    for key, value in summary.items():
        if isinstance(value, bool):
            payload[f"eval/{key}"] = float(int(value))
            continue
        if isinstance(value, (int, float)):
            payload[f"eval/{key}"] = float(value)
    return payload


def _wandb_config_payload(args: argparse.Namespace) -> dict[str, Any]:
    return _json_safe(
        {
            **vars(args),
            "dataset_dir": str(Path(args.dataset_dir)),
            "output_json": str(Path(args.output_json)),
            "predictions_jsonl": str(Path(args.predictions_jsonl)),
            "judge_cache_jsonl": str(Path(args.judge_cache_jsonl)) if args.judge_cache_jsonl is not None else "",
        }
    )


def _wandb_log_benchmark(*, args: argparse.Namespace, summary: dict[str, Any]) -> Optional[str]:
    project = str(args.wandb_project or "").strip()
    if not project:
        return None
    if not callable(getattr(wandb, "init", None)) or not callable(getattr(wandb, "log", None)) or not hasattr(wandb, "Api"):
        print("wandb benchmark logging requested but wandb SDK is unavailable; skipping remote logging.")
        return None
    run = wandb.init(
        project=project,
        name=str(args.wandb_run_name or "").strip() or None,
        config=_wandb_config_payload(args),
    )
    log_step = (
        int(args.wandb_step)
        if args.wandb_step is not None
        else int(summary.get("resolved_checkpoint_step") or summary.get("checkpoint_step") or 0)
    )
    metric_payload = _wandb_metric_payload(summary)
    if metric_payload:
        wandb.log(metric_payload, step=log_step)
    for key in (
        "dataset_dir",
        "split",
        "model",
        "finetune_id",
        "checkpoint_step",
        "resolved_checkpoint_step",
        "used_checkpoint_fallback",
        "grader_model_id",
        "grader_profile",
        "grader_rubric_version",
        "answer_parse_mode",
        "predictions_jsonl",
        "judge_cache_jsonl",
        "notes_markdown",
    ):
        if key in summary:
            run.summary[f"benchmark_{key}"] = summary[key]
    warnings = list(summary.get("metric_warnings") or [])
    if warnings:
        run.summary["benchmark_metric_warnings"] = "\n".join(str(item) for item in warnings)
    for key, value in metric_payload.items():
        run.summary[key] = value
    finish = getattr(run, "finish", None)
    if callable(finish):
        finish()
    return getattr(run, "url", None)


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

    parser = argparse.ArgumentParser(description="Benchmark Inspector MD query training tasks.")
    parser.add_argument("--config", default=str(config_path))
    parser.add_argument("--env-file", default=str(common.repo_relative(".env.staging")))
    parser.add_argument("--api-key", default="")
    parser.add_argument("--api-key-env-var", default=common.DEFAULT_API_KEY_ENV_VAR)
    parser.add_argument("--api-key-env-vars", nargs="+", default=list(common.DEFAULT_API_KEY_ENV_VARS))
    parser.add_argument("--base-url", default=common.DEFAULT_BASE_URL)
    parser.add_argument("--dataset-dir", default=str(common.repo_relative("outputs", "inspector_query_issues_v2")))
    parser.add_argument("--split", default="validation")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="")
    parser.add_argument("--finetune-id", default="")
    parser.add_argument("--checkpoint-step", type=int, default=None)
    parser.add_argument("--checkpoint-fallback-policy", choices=("nearest_saved", "exact"), default="nearest_saved")
    parser.add_argument("--checkpoint-ready-max-wait-s", type=float, default=300.0)
    parser.add_argument("--checkpoint-ready-poll-interval-s", type=float, default=5.0)
    parser.add_argument("--base-model", default=common.DEFAULT_BASE_MODEL)
    parser.add_argument("--reasoning", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--client-max-retries", type=int, default=4)
    parser.add_argument("--client-backoff-base-s", type=float, default=2.0)
    parser.add_argument("--client-backoff-max-s", type=float, default=20.0)
    parser.add_argument("--grader-api-key", default="")
    parser.add_argument("--grader-api-key-env-var", default=openrouter_grader.DEFAULT_OPENROUTER_ENV_VAR)
    parser.add_argument("--grader-api-base", default=openrouter_grader.DEFAULT_OPENROUTER_API_BASE)
    parser.add_argument("--grader-model-id", default=openrouter_grader.DEFAULT_GRADER_MODEL)
    parser.add_argument("--grader-profile", default=openrouter_grader.DEFAULT_GRADER_PROFILE)
    parser.add_argument("--grader-rubric-version", default=openrouter_grader.DEFAULT_GRADER_RUBRIC_VERSION)
    parser.add_argument("--grader-timeout", type=float, default=60.0)
    parser.add_argument("--answer-parse-mode", choices=("strict_raw", "grader_normalize"), default="strict_raw")
    parser.add_argument("--output-json", default=str(common.repo_relative("outputs", "benchmarks", "inspector_query.metrics.json")))
    parser.add_argument("--predictions-jsonl", default=str(common.repo_relative("outputs", "benchmarks", "inspector_query.predictions.jsonl")))
    parser.add_argument("--judge-cache-jsonl", default="")
    parser.add_argument("--reuse-judge-cache", action="store_true")
    parser.add_argument("--wandb-project", default="")
    parser.add_argument("--wandb-run-name", default="")
    parser.add_argument("--wandb-step", type=int, default=None)

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
    args.dataset_dir = common.resolve_path(args.dataset_dir, module_root=SCRIPT_DIR)
    args.output_json = common.resolve_path(args.output_json, module_root=SCRIPT_DIR)
    args.predictions_jsonl = common.resolve_path(args.predictions_jsonl, module_root=SCRIPT_DIR)
    args.judge_cache_jsonl = common.resolve_path(args.judge_cache_jsonl, module_root=SCRIPT_DIR) if str(args.judge_cache_jsonl or "").strip() else None
    if single_key_requested and not multi_key_requested:
        args.api_key_env_vars = [args.api_key_env_var]
    else:
        args.api_key_env_vars = common.normalize_api_key_env_vars(args.api_key_env_vars)
    split = str(args.split or "").strip().lower()
    if split not in {"train", "validation", "test", "val"}:
        raise ValueError(f"Unsupported split: {args.split!r}")
    args.split = "validation" if split == "val" else split
    if int(args.client_max_retries) < 0:
        raise ValueError("--client-max-retries must be >= 0")
    if float(args.client_backoff_base_s) <= 0:
        raise ValueError("--client-backoff-base-s must be > 0")
    if float(args.client_backoff_max_s) < float(args.client_backoff_base_s):
        raise ValueError("--client-backoff-max-s must be >= --client-backoff-base-s")
    if args.wandb_step is not None and int(args.wandb_step) < 0:
        raise ValueError("--wandb-step must be >= 0")
    return args


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    common.maybe_load_env_file(args.env_file, override=False)
    api_key_pool = common.resolve_api_key_pool(explicit_api_key=args.api_key, api_key_env_vars=args.api_key_env_vars)
    tuna_api_key = api_key_pool.slots[0].api_key
    model = str(args.model or "").strip()
    resolved_checkpoint_step: Optional[int] = None
    used_checkpoint_fallback = False
    if not model:
        model, resolved_checkpoint_step, used_checkpoint_fallback = _resolve_model(
            base_model=args.base_model,
            finetune_id=args.finetune_id,
            checkpoint_step=args.checkpoint_step,
            api_base=args.base_url,
            tuna_api_key=tuna_api_key,
            fallback_policy=args.checkpoint_fallback_policy,
            ready_max_wait_s=args.checkpoint_ready_max_wait_s,
            ready_poll_interval_s=args.checkpoint_ready_poll_interval_s,
        )
    grader_api_key = openrouter_grader.resolve_openrouter_api_key(
        explicit_api_key=str(args.grader_api_key or ""),
        api_key_env_var=str(args.grader_api_key_env_var or openrouter_grader.DEFAULT_OPENROUTER_ENV_VAR),
    )
    grader = openrouter_grader.OpenRouterGrader(
        api_key=grader_api_key,
        model_id=args.grader_model_id,
        api_base=args.grader_api_base,
        timeout=float(args.grader_timeout),
        profile=args.grader_profile,
        rubric_version=args.grader_rubric_version,
    )
    client = MoondreamInspectorClient(
        api_key_pool=api_key_pool,
        base_url=args.base_url,
        timeout=float(args.timeout),
        max_retries=int(args.client_max_retries),
        backoff_base_s=float(args.client_backoff_base_s),
        backoff_max_s=float(args.client_backoff_max_s),
    )
    examples = _load_split_examples(split_name=args.split, dataset_dir=Path(args.dataset_dir))
    task_type = str(examples[0].task_type) if examples else ""
    predictions_path = Path(args.predictions_jsonl)
    judge_cache_path = Path(args.judge_cache_jsonl) if args.judge_cache_jsonl is not None else _judge_cache_path_for_predictions(predictions_path)
    if judge_cache_path.exists() and not bool(args.reuse_judge_cache):
        judge_cache_path.unlink()
    print(f"preflight base_url={args.base_url} key_slots={api_key_pool.env_var_names or ['<explicit>']} grader_model={grader.model_id}")
    stagger_delay_s = _checkpoint_eval_stagger_delay_s(
        finetune_id=str(args.finetune_id or ""),
        checkpoint_step=resolved_checkpoint_step if resolved_checkpoint_step is not None else args.checkpoint_step,
    )
    if stagger_delay_s > 0.0:
        print(f"staggering checkpoint eval start by {stagger_delay_s:.1f}s to reduce backend spikes")
        time.sleep(stagger_delay_s)
    metrics = _evaluate_split(
        inference_client=client,
        model=model,
        examples=examples,
        split_name=args.split,
        seed=int(args.seed),
        max_samples=None if int(args.max_samples) <= 0 else int(args.max_samples),
        reasoning=bool(args.reasoning),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        max_tokens=int(args.max_tokens),
        grader=grader,
        judge_cache={},
        judge_cache_path=judge_cache_path,
        predictions_path=predictions_path,
        answer_parse_mode=str(args.answer_parse_mode),
    )
    diagnostics = _build_benchmark_diagnostics(
        examples=examples,
        predictions_path=predictions_path,
        metrics=metrics,
        task_type=task_type,
        answer_parse_mode=str(args.answer_parse_mode),
    )
    summary = {
        "dataset_dir": str(Path(args.dataset_dir)),
        "split": str(args.split),
        "base_url": str(args.base_url),
        "api_key_env_vars": list(api_key_pool.env_var_names),
        "api_key_slot_count": len(api_key_pool.slots),
        "model": str(model),
        "finetune_id": str(args.finetune_id or ""),
        "checkpoint_step": None if args.checkpoint_step is None else int(args.checkpoint_step),
        "resolved_checkpoint_step": resolved_checkpoint_step,
        "used_checkpoint_fallback": bool(used_checkpoint_fallback),
        "grader_model_id": grader.model_id,
        "grader_profile": grader.profile,
        "grader_rubric_version": grader.rubric_version,
        "answer_parse_mode": str(args.answer_parse_mode),
        **metrics,
        **diagnostics,
        "predictions_jsonl": str(predictions_path),
        "judge_cache_jsonl": str(judge_cache_path),
    }
    for metric_name in list(summary.get("omitted_metrics") or []):
        summary.pop(str(metric_name), None)
    notes_path = Path(args.output_json).with_suffix(".notes.md")
    summary["notes_markdown"] = str(notes_path)
    common.write_json(Path(args.output_json), summary)
    _write_benchmark_notes(notes_path, summary)
    wandb_url = _wandb_log_benchmark(args=args, summary=summary)
    if wandb_url:
        summary["wandb_url"] = str(wandb_url)
        common.write_json(Path(args.output_json), summary)
    print(json.dumps(summary, indent=2))
    return summary


def main(argv: Optional[list[str]] = None) -> None:
    run_benchmark(parse_args(argv))


if __name__ == "__main__":
    main()
