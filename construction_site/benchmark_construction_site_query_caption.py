#!/usr/bin/env python3
"""Benchmark ConstructionSite caption query models."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from statistics import fmean
from typing import Any, Optional

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from construction_site.common import config_to_cli_args, load_json_config, parse_prediction_json, repo_relative, resolve_config_path  # noqa: E402
from construction_site import query_common  # noqa: E402
from construction_site import train_construction_site_query_caption as train_utils  # noqa: E402

DEFAULT_BASE_URL = "https://api-staging.moondream.ai/v1"
DEFAULT_CONFIG_PATH = repo_relative("configs", "benchmark_construction_site_query_caption_default.json")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    raw_argv = list(argv) if argv is not None else list(os.sys.argv[1:])
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    pre_args, _ = pre_parser.parse_known_args(raw_argv)
    config_path = resolve_config_path(pre_args.config, script_dir=Path(__file__).resolve().parent)
    config = load_json_config(config_path, default_path=DEFAULT_CONFIG_PATH)

    parser = argparse.ArgumentParser(description="Benchmark ConstructionSite caption query models.")
    parser.add_argument("--config", default=str(config_path))
    parser.add_argument("--env-file", default=str(repo_relative(".env.staging")))
    parser.add_argument("--api-key", default="")
    parser.add_argument("--api-key-env-var", default="CICID_GPUB_MOONDREAM_API_KEY_1")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--dataset-source", choices=["local_jsonl"], default="local_jsonl")
    parser.add_argument("--dataset-dir", default=str(repo_relative("outputs", "construction_site_query_caption_v1")))
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--model", default="")
    parser.add_argument("--finetune-id", default="")
    parser.add_argument("--checkpoint-step", type=int, default=-1)
    parser.add_argument(
        "--checkpoint-fallback-policy",
        choices=["nearest_saved", "exact"],
        default="nearest_saved",
    )
    parser.add_argument("--checkpoint-ready-max-wait-s", type=float, default=0.0)
    parser.add_argument("--checkpoint-ready-poll-interval-s", type=float, default=5.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=320)
    reasoning_group = parser.add_mutually_exclusive_group()
    reasoning_group.add_argument("--reasoning", dest="reasoning", action="store_true")
    reasoning_group.add_argument("--no-reasoning", dest="reasoning", action="store_false")
    parser.set_defaults(reasoning=False)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retry-429-max-retries", type=int, default=2)
    parser.add_argument("--retry-429-backoff-s", type=float, default=1.0)
    parser.add_argument("--retry-429-max-backoff-s", type=float, default=8.0)
    parser.add_argument("--retry-5xx-max-retries", type=int, default=2)
    parser.add_argument("--retry-5xx-backoff-s", type=float, default=2.0)
    parser.add_argument("--retry-5xx-max-backoff-s", type=float, default=16.0)
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--predictions-jsonl", default="")
    parser.add_argument("--no-progress", action="store_true")

    option_to_dest: dict[str, str] = {}
    for action in parser._actions:
        if not action.option_strings:
            continue
        for opt in action.option_strings:
            option_to_dest[opt] = action.dest
    overridden_dests = {option_to_dest[arg] for arg in raw_argv if arg in option_to_dest}
    config_cli_args = config_to_cli_args(
        parser,
        config,
        config_path=config_path,
        overridden_dests=overridden_dests,
    )
    args = parser.parse_args(config_cli_args + raw_argv)
    args.config = str(resolve_config_path(args.config, script_dir=Path(__file__).resolve().parent))
    args.env_file = query_common.resolve_env_file(
        args.env_file,
        repo_root=REPO_ROOT,
        module_root=Path(__file__).resolve().parent,
    )
    args.dataset_dir = query_common.resolve_path(
        args.dataset_dir,
        repo_root=REPO_ROOT,
        module_root=Path(__file__).resolve().parent,
    )
    if args.max_samples <= 0:
        args.max_samples = None
    args.checkpoint_step = None if int(args.checkpoint_step) < 0 else int(args.checkpoint_step)
    return args


def _resolve_api_key(args: argparse.Namespace) -> str:
    if str(args.api_key or "").strip():
        return str(args.api_key).strip()
    value = os.environ.get(str(args.api_key_env_var or "MOONDREAM_API_KEY"), "")
    if str(value or "").strip():
        return str(value).strip()
    value = os.environ.get("MOONDREAM_API_KEY", "")
    if str(value or "").strip():
        return str(value).strip()
    raise ValueError("MOONDREAM_API_KEY is required")


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    query_common.load_dotenv(args.env_file, override=False)
    args.api_key = _resolve_api_key(args)
    try:
        model_resolution = query_common.resolve_query_inference_model(
            api_base=args.base_url,
            api_key=args.api_key,
            model=args.model,
            finetune_id=args.finetune_id,
            checkpoint_step=args.checkpoint_step,
            timeout=args.timeout,
            fallback_policy=str(args.checkpoint_fallback_policy),
            checkpoint_ready_max_wait_s=float(args.checkpoint_ready_max_wait_s),
            checkpoint_ready_poll_interval_s=float(args.checkpoint_ready_poll_interval_s),
        )
    except ValueError as exc:
        raise SystemExit(str(exc))
    model = model_resolution.model
    if not args.skip_preflight:
        try:
            query_common.preflight_query_api(
                api_base=args.base_url,
                api_key=args.api_key,
                model=model,
                timeout=args.timeout,
                reasoning=args.reasoning,
                retry_429_max_retries=args.retry_429_max_retries,
                retry_429_backoff_s=args.retry_429_backoff_s,
                retry_429_max_backoff_s=args.retry_429_max_backoff_s,
                retry_5xx_max_retries=args.retry_5xx_max_retries,
                retry_5xx_backoff_s=args.retry_5xx_backoff_s,
                retry_5xx_max_backoff_s=args.retry_5xx_max_backoff_s,
            )
        except Exception as exc:
            raise SystemExit(
                f"query preflight failed for model={model}. details={query_common.error_message(exc)}"
            )
    examples = train_utils._load_split_examples(split_name=args.split, dataset_dir=args.dataset_dir)
    indices = list(range(len(examples)))
    random.Random(args.seed).shuffle(indices)
    if args.max_samples is not None:
        indices = indices[: args.max_samples]

    reward_values: list[float] = []
    json_object_count = 0
    parse_success_count = 0
    task_correct_count = 0
    caption_f1_values: list[float] = []
    attribute_values: list[float] = []
    object_values: list[float] = []
    length_values: list[float] = []
    request_failures = 0
    latency_values: list[float] = []

    predictions_handle = None
    predictions_path = Path(args.predictions_jsonl).expanduser().resolve() if args.predictions_jsonl else None
    if predictions_path is not None:
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        predictions_handle = predictions_path.open("w", encoding="utf-8")

    try:
        for index in query_common.tqdm(
            indices,
            desc=f"benchmark:{args.split}",
            total=len(indices),
            dynamic_ncols=True,
            disable=not query_common.progress_enabled(bool(args.no_progress)),
        ):
            example = examples[index]
            try:
                with Image.open(example.image_path) as image:
                    image_url = query_common.to_data_url(image.convert("RGB"))
            except (FileNotFoundError, OSError) as exc:
                request_failures += 1
                print(f"row={example.row_id}: image load failed ({exc}); skipping")
                continue
            try:
                answer_text, raw_response, latency_ms = query_common.call_query_api(
                    api_base=args.base_url,
                    api_key=args.api_key,
                    model=model,
                    question=example.question,
                    image_url=image_url,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens,
                    reasoning=args.reasoning,
                    timeout=args.timeout,
                    retry_429_max_retries=args.retry_429_max_retries,
                    retry_429_backoff_s=args.retry_429_backoff_s,
                    retry_429_max_backoff_s=args.retry_429_max_backoff_s,
                    retry_5xx_max_retries=args.retry_5xx_max_retries,
                    retry_5xx_backoff_s=args.retry_5xx_backoff_s,
                    retry_5xx_max_backoff_s=args.retry_5xx_max_backoff_s,
                )
            except Exception as exc:
                request_failures += 1
                print(f"row={example.row_id}: query failed. details={query_common.error_message(exc)}")
                continue
            latency_values.append(latency_ms)
            payload = parse_prediction_json(answer_text)
            outcome = train_utils._score_payload_for_example(example, payload)
            reward_values.append(float(outcome.reward))
            caption_f1_values.append(float(outcome.caption_token_f1))
            attribute_values.append(float(outcome.attribute_hit_rate))
            object_values.append(float(outcome.object_hit_rate))
            length_values.append(float(outcome.length_score))
            if outcome.json_object_parsed:
                json_object_count += 1
            if outcome.parse_success:
                parse_success_count += 1
            if outcome.task_correct:
                task_correct_count += 1
            if predictions_handle is not None:
                predictions_handle.write(
                    json.dumps(
                        {
                            "row_id": example.row_id,
                            "answer": answer_text,
                            "prediction_json": payload,
                            "reward": outcome.reward,
                            "caption_token_f1": outcome.caption_token_f1,
                            "attribute_hit_rate": outcome.attribute_hit_rate,
                            "object_hit_rate": outcome.object_hit_rate,
                            "length_score": outcome.length_score,
                            "raw_response": raw_response,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    finally:
        if predictions_handle is not None:
            predictions_handle.close()

    total = len(reward_values)
    sorted_latency = sorted(latency_values)
    p95_latency_ms = 0.0
    if sorted_latency:
        p95_latency_ms = sorted_latency[int(round(0.95 * (len(sorted_latency) - 1)))]
    metrics = {
        "model": model,
        "finetune_id": str(model_resolution.finetune_id or args.finetune_id or ""),
        "checkpoint_step": int(model_resolution.resolved_checkpoint_step)
        if model_resolution.resolved_checkpoint_step is not None
        else -1,
        "requested_checkpoint_step": int(model_resolution.requested_checkpoint_step)
        if model_resolution.requested_checkpoint_step is not None
        else -1,
        "resolved_checkpoint_step": int(model_resolution.resolved_checkpoint_step)
        if model_resolution.resolved_checkpoint_step is not None
        else -1,
        "config": args.config,
        "split": args.split,
        "requested_rows": len(indices),
        "evaluated_rows": total,
        "request_failures": request_failures,
        "eval_reward_mean": fmean(reward_values) if reward_values else 0.0,
        "eval_json_object_rate": json_object_count / max(1, total),
        "eval_json_parse_rate": parse_success_count / max(1, total),
        "eval_caption_token_f1": fmean(caption_f1_values) if caption_f1_values else 0.0,
        "eval_attribute_hit_rate": fmean(attribute_values) if attribute_values else 0.0,
        "eval_object_hit_rate": fmean(object_values) if object_values else 0.0,
        "eval_length_score": fmean(length_values) if length_values else 0.0,
        "eval_task_correct_rate": task_correct_count / max(1, total),
        "latency_avg_ms": fmean(latency_values) if latency_values else 0.0,
        "latency_p95_ms": p95_latency_ms,
    }
    print(json.dumps(metrics, indent=2, sort_keys=True))
    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if predictions_path is not None:
        print(f"wrote predictions JSONL: {predictions_path}")


if __name__ == "__main__":
    main()
