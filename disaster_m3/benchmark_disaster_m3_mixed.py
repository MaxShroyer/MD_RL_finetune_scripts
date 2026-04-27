#!/usr/bin/env python3
"""Benchmark mixed-skill DisasterM3 manifests against Moondream inference APIs."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from dataclasses import asdict
from pathlib import Path
from statistics import fmean
from typing import Any, Optional

from PIL import Image

try:
    from tqdm.auto import tqdm  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    def tqdm(iterable=None, *args, **kwargs):  # type: ignore
        return iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from disaster_m3 import common  # noqa: E402
from disaster_m3 import train_disaster_m3_mixed as train_utils  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = common.repo_relative("configs", "benchmark_disaster_m3_mixed_default.json")


def _slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", str(text or "").strip().lower()).strip("_") or "model"


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    raw_argv = list(argv) if argv is not None else list(os.sys.argv[1:])
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    pre_args, _ = pre_parser.parse_known_args(raw_argv)
    config_path = common.resolve_config_path(pre_args.config, script_dir=SCRIPT_DIR)
    config = common.load_json_config(config_path, default_path=DEFAULT_CONFIG_PATH)

    parser = argparse.ArgumentParser(description="Benchmark the DisasterM3 mixed-skill local dataset.")
    parser.add_argument("--config", default=str(config_path))
    parser.add_argument("--env-file", default=str(common.repo_relative(".env.staging")))
    parser.add_argument("--api-key", default="")
    parser.add_argument("--api-key-env-var", default=common.DEFAULT_API_KEY_ENV_VAR)
    parser.add_argument("--base-url", default=common.DEFAULT_BASE_URL)
    parser.add_argument("--dataset-dir", default=str(common.DEFAULT_OUTPUT_DIR))
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--model", default="")
    parser.add_argument("--finetune-id", default="")
    parser.add_argument("--checkpoint-step", type=int, default=-1)
    parser.add_argument("--checkpoint-fallback-policy", choices=("nearest_saved", "exact"), default="nearest_saved")
    parser.add_argument("--checkpoint-ready-max-wait-s", type=float, default=0.0)
    parser.add_argument("--checkpoint-ready-poll-interval-s", type=float, default=5.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=384)
    reasoning_group = parser.add_mutually_exclusive_group()
    reasoning_group.add_argument("--reasoning", dest="reasoning", action="store_true")
    reasoning_group.add_argument("--no-reasoning", dest="reasoning", action="store_false")
    parser.set_defaults(reasoning=False)
    parser.add_argument("--image-quality", type=int, default=common.DEFAULT_JPEG_QUALITY)
    parser.add_argument("--detect-max-objects", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retry-429-max-retries", type=int, default=2)
    parser.add_argument("--retry-429-backoff-s", type=float, default=1.0)
    parser.add_argument("--retry-429-max-backoff-s", type=float, default=8.0)
    parser.add_argument("--retry-5xx-max-retries", type=int, default=2)
    parser.add_argument("--retry-5xx-backoff-s", type=float, default=2.0)
    parser.add_argument("--retry-5xx-max-backoff-s", type=float, default=16.0)
    parser.add_argument("--benchmark-output-dir", default=str(common.DEFAULT_BENCHMARK_OUTPUT_DIR))
    parser.add_argument("--output-json", default="")
    parser.add_argument("--predictions-jsonl", default="")
    parser.add_argument("--no-progress", action="store_true")

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
    args.env_file = str(common.resolve_path(args.env_file, repo_root=REPO_ROOT, module_root=SCRIPT_DIR))
    args.dataset_dir = common.resolve_path(args.dataset_dir, repo_root=REPO_ROOT, module_root=SCRIPT_DIR)
    args.benchmark_output_dir = common.resolve_path(args.benchmark_output_dir, repo_root=REPO_ROOT, module_root=SCRIPT_DIR)
    args.checkpoint_step = None if int(args.checkpoint_step) < 0 else int(args.checkpoint_step)
    if int(args.max_samples) <= 0:
        args.max_samples = None
    if str(args.output_json or "").strip():
        args.output_json = str(common.resolve_path(args.output_json, repo_root=REPO_ROOT, module_root=SCRIPT_DIR))
    if str(args.predictions_jsonl or "").strip():
        args.predictions_jsonl = str(common.resolve_path(args.predictions_jsonl, repo_root=REPO_ROOT, module_root=SCRIPT_DIR))
    return args


def _prediction_paths(args: argparse.Namespace, *, model_label: str) -> tuple[Path, Path]:
    output_dir = Path(args.benchmark_output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = (
        Path(args.output_json).expanduser().resolve()
        if str(args.output_json or "").strip()
        else output_dir / f"{args.split}_{model_label}_summary.json"
    )
    predictions_path = (
        Path(args.predictions_jsonl).expanduser().resolve()
        if str(args.predictions_jsonl or "").strip()
        else output_dir / f"{args.split}_{model_label}_predictions.jsonl"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    return summary_path, predictions_path


def _latency_p95(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = int(round(0.95 * (len(sorted_values) - 1)))
    return float(sorted_values[index])


def _ground_truth_payload(record: common.MixedTaskRecord) -> Any:
    if record.skill == "query":
        return json.loads(record.final_answer_json or "{}")
    if record.skill == "detect":
        return json.loads(record.answer_boxes_json or "[]")
    return {
        "points": json.loads(record.answer_points_json or "[]"),
        "boxes": json.loads(record.answer_boxes_json or "[]"),
    }


def _parsed_prediction_for_record(record: common.MixedTaskRecord, prediction: Any) -> Any:
    if record.skill == "query":
        return common.parse_prediction_json(str(prediction or ""))
    return prediction


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    args.api_key = common.resolve_api_key(api_key=args.api_key, api_key_env_var=args.api_key_env_var, env_file=args.env_file)
    model_resolution = common.resolve_inference_model(
        api_base=str(args.base_url),
        api_key=str(args.api_key),
        model=str(args.model),
        finetune_id=str(args.finetune_id),
        checkpoint_step=args.checkpoint_step,
        timeout=float(args.timeout),
        fallback_policy=str(args.checkpoint_fallback_policy),
        checkpoint_ready_max_wait_s=float(args.checkpoint_ready_max_wait_s),
        checkpoint_ready_poll_interval_s=float(args.checkpoint_ready_poll_interval_s),
    )
    summary_path, predictions_path = _prediction_paths(args, model_label=_slugify(model_resolution.model))
    records = common.load_mixed_records(dataset_dir=args.dataset_dir, split_name=args.split)
    indices = list(range(len(records)))
    random.Random(int(args.seed)).shuffle(indices)
    if args.max_samples is not None:
        indices = indices[: int(args.max_samples)]

    used_records: list[common.MixedTaskRecord] = []
    outcomes: list[train_utils.MixedScoreOutcome] = []
    latency_values: list[float] = []
    request_failures = 0

    with predictions_path.open("w", encoding="utf-8") as predictions_handle:
        iterator = tqdm(
            indices,
            desc=f"benchmark:{args.split}",
            total=len(indices),
            dynamic_ncols=True,
            disable=not common.progress_enabled(bool(args.no_progress)),
        )
        for index in iterator:
            record = records[index]
            try:
                with Image.open(record.image_path) as image:
                    image_url = common.to_data_url(image.convert("RGB"), quality=int(args.image_quality))
            except (FileNotFoundError, OSError) as exc:
                request_failures += 1
                print(f"row_id={record.row_id}: image load failed. details={exc}")
                continue
            try:
                prediction, raw_response, latency_ms = common.call_inference_api(
                    skill=record.skill,
                    api_base=str(args.base_url),
                    api_key=str(args.api_key),
                    model=model_resolution.model,
                    question=record.question,
                    object_name=record.object_name,
                    image_url=image_url,
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                    max_tokens=int(args.max_tokens),
                    reasoning=bool(args.reasoning) if record.skill == "query" else None,
                    timeout=float(args.timeout),
                    retry_429_max_retries=int(args.retry_429_max_retries),
                    retry_429_backoff_s=float(args.retry_429_backoff_s),
                    retry_429_max_backoff_s=float(args.retry_429_max_backoff_s),
                    retry_5xx_max_retries=int(args.retry_5xx_max_retries),
                    retry_5xx_backoff_s=float(args.retry_5xx_backoff_s),
                    retry_5xx_max_backoff_s=float(args.retry_5xx_max_backoff_s),
                    max_objects=int(args.detect_max_objects),
                )
            except Exception as exc:
                request_failures += 1
                print(f"row_id={record.row_id}: inference failed. details={common.error_message(exc)}")
                continue

            latency_values.append(float(latency_ms))
            if record.skill == "query":
                outcome = train_utils.score_prediction_for_record(
                    record,
                    query_answer=str(prediction or ""),
                    strict_query_rewards=False,
                )
            elif record.skill == "detect":
                outcome = train_utils.score_prediction_for_record(
                    record,
                    detect_boxes=list(prediction or []),
                    strict_query_rewards=False,
                )
            else:
                outcome = train_utils.score_prediction_for_record(
                    record,
                    point_coords=list(prediction or []),
                    strict_query_rewards=False,
                )
            used_records.append(record)
            outcomes.append(outcome)
            parsed_prediction = _parsed_prediction_for_record(record, prediction)
            predictions_handle.write(
                json.dumps(
                    {
                        "row_id": record.row_id,
                        "split": record.split,
                        "task_name": record.task_name,
                        "task_family": record.task_family,
                        "skill": record.skill,
                        "question": record.question,
                        "object_name": record.object_name,
                        "ground_truth": _ground_truth_payload(record),
                        "prediction": prediction,
                        "parsed_prediction": parsed_prediction,
                        "reward": outcome.reward,
                        "parse_success": outcome.parse_success,
                        "task_correct": outcome.task_correct,
                        "grading": asdict(outcome),
                        "raw_response": raw_response,
                        "latency_ms": latency_ms,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    metrics = train_utils.summarize_outcomes(used_records, outcomes)
    summary = {
        "split": str(args.split),
        "model": model_resolution.model,
        "finetune_id": model_resolution.finetune_id,
        "requested_checkpoint_step": model_resolution.requested_checkpoint_step,
        "resolved_checkpoint_step": model_resolution.resolved_checkpoint_step,
        "used_checkpoint_fallback": bool(model_resolution.used_checkpoint_fallback),
        "requested_samples": len(indices),
        "evaluated_samples": len(used_records),
        "request_failures": int(request_failures),
        "latency_mean_ms": fmean(latency_values) if latency_values else 0.0,
        "latency_p95_ms": _latency_p95(latency_values),
        "metrics": metrics,
        "overall_reward_mean": float(metrics["overall"]["reward_mean"]),
        "overall_task_accuracy": float(metrics["overall"]["task_accuracy"]),
        "overall_parse_success_rate": float(metrics["overall"]["parse_success_rate"]),
    }
    common.write_json(summary_path, summary)
    print(
        "saved benchmark:",
        summary_path,
        f"samples={summary['evaluated_samples']}",
        f"failures={summary['request_failures']}",
        f"reward={summary['metrics']['overall']['reward_mean']:.4f}",
    )


if __name__ == "__main__":
    main()
