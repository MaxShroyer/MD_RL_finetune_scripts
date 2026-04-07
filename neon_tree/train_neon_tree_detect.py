#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import deque
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from async_checkpoint_eval import (
    CheckpointEvalResult,
    DispatchHandle,
    dispatch_checkpoint_eval,
    drain_checkpoint_eval_jobs,
    poll_checkpoint_eval_jobs,
)
from neon_tree import benchmark_neon_tree_detect as benchmark_mod
from neon_tree import common
from tuna_sdk import DetectOutput, DetectRequest, DetectSettings, TrainStepGroup, TunaClient

try:
    import wandb  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    class _WandbRun:
        def __init__(self) -> None:
            self.summary: dict[str, Any] = {}

        def finish(self) -> None:
            return

    class _WandbShim:
        @staticmethod
        def init(*args: Any, **kwargs: Any) -> _WandbRun:
            print("wandb not installed; continuing without remote logging.")
            return _WandbRun()

        @staticmethod
        def log(*args: Any, **kwargs: Any) -> None:
            return

    wandb = _WandbShim()

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = common.repo_relative("configs", "current", "train_neon_tree_detect_default.json")
SELECTION_METRIC_CHOICES = ("miou", "f1", "f1_macro")
MAX_NO_TILE_TRAIN_PIXELS = 25_000_000


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    raw_argv = list(argv) if argv is not None else list(os.sys.argv[1:])
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    pre_args, _ = pre_parser.parse_known_args(raw_argv)
    config_path = common.resolve_config_path(pre_args.config, script_dir=SCRIPT_DIR)
    config = common.load_json_config(config_path, default_path=DEFAULT_CONFIG_PATH)

    parser = argparse.ArgumentParser(description="Train a NEON tree detect finetune on staging.")
    parser.add_argument("--config", default=str(config_path))
    parser.add_argument("--env-file", default=str(common.repo_relative(".env.staging")))
    parser.add_argument("--api-key", default="")
    parser.add_argument("--api-key-env-var", default=common.DEFAULT_API_KEY_ENV_VAR)
    parser.add_argument("--base-url", default=common.DEFAULT_STAGING_API_BASE)
    parser.add_argument("--hf-token", default="")
    parser.add_argument("--hf-dataset-repo-id", default=common.DEFAULT_HF_DATASET_REPO_ID)
    parser.add_argument("--hf-dataset-revision", default=common.DEFAULT_HF_DATASET_REVISION)
    parser.add_argument("--hf-cache-dir", default="")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="validation")
    parser.add_argument("--prompt", default=common.DEFAULT_PROMPT)
    parser.add_argument("--base-model", default=common.DEFAULT_BASE_MODEL)
    parser.add_argument("--finetune-id", default="")
    parser.add_argument("--finetune-name", default="")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-objects", type=int, default=256)
    parser.add_argument("--selection-metric", choices=SELECTION_METRIC_CHOICES, default="f1")
    parser.add_argument("--reward-metric", choices=("miou", "f1", "hybrid"), default="hybrid")
    parser.add_argument("--reward-fn-beta", type=float, default=2.0)
    parser.add_argument("--reward-oversize-ratio-cap", type=float, default=4.0)
    parser.add_argument("--reward-huge-area-cap", type=float, default=0.15)
    parser.add_argument("--reward-collapse-centers-cap", type=float, default=3.0)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--eval-max-samples", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--request-retries", type=int, default=2)
    parser.add_argument("--request-retry-backoff-s", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    off_policy_group = parser.add_mutually_exclusive_group()
    off_policy_group.add_argument("--off-policy", dest="off_policy", action="store_true")
    off_policy_group.add_argument("--no-off-policy", dest="off_policy", action="store_false")
    parser.set_defaults(off_policy=False)
    parser.add_argument("--off-policy-mix-ratio", type=float, default=0.25)
    parser.add_argument("--off-policy-buffer-size", type=int, default=1024)
    parser.add_argument("--off-policy-warmup-steps", type=int, default=10)
    parser.add_argument("--off-policy-min-buffer-groups", type=int, default=64)
    parser.add_argument("--positive-tile-probability", type=float, default=0.8)
    parser.add_argument("--include-empty-tiles", dest="include_empty_tiles", action="store_true")
    parser.add_argument("--no-include-empty-tiles", dest="include_empty_tiles", action="store_false")
    parser.set_defaults(include_empty_tiles=True)
    parser.add_argument("--tiling-enabled", dest="tiling_enabled", action="store_true")
    parser.add_argument("--no-tiling-enabled", dest="tiling_enabled", action="store_false")
    parser.set_defaults(tiling_enabled=True)
    parser.add_argument("--tile-width", type=int, default=1024)
    parser.add_argument("--tile-height", type=int, default=1024)
    parser.add_argument("--tile-overlap-x", type=int, default=128)
    parser.add_argument("--tile-overlap-y", type=int, default=128)
    parser.add_argument("--merge-iou-threshold", type=float, default=0.5)
    parser.add_argument("--wandb-project", default="moondream-neon-tree-detect")
    parser.add_argument("--wandb-run-name", default="")
    parser.add_argument("--wandb-log-profile", default="lean")
    async_group = parser.add_mutually_exclusive_group()
    async_group.add_argument("--async-checkpoint-eval", dest="async_checkpoint_eval", action="store_true")
    async_group.add_argument("--no-async-checkpoint-eval", dest="async_checkpoint_eval", action="store_false")
    parser.set_defaults(async_checkpoint_eval=True)
    parser.add_argument(
        "--async-checkpoint-eval-dir",
        default=str(common.repo_relative("outputs", "async_checkpoint_eval")),
    )
    parser.add_argument("--async-checkpoint-eval-max-inflight", type=int, default=1)
    drain_group = parser.add_mutually_exclusive_group()
    drain_group.add_argument("--async-checkpoint-eval-drain-on-exit", dest="async_checkpoint_eval_drain_on_exit", action="store_true")
    drain_group.add_argument("--no-async-checkpoint-eval-drain-on-exit", dest="async_checkpoint_eval_drain_on_exit", action="store_false")
    parser.set_defaults(async_checkpoint_eval_drain_on_exit=True)
    auto_group = parser.add_mutually_exclusive_group()
    auto_group.add_argument("--auto-benchmark-best-checkpoint", dest="auto_benchmark_best_checkpoint", action="store_true")
    auto_group.add_argument("--no-auto-benchmark-best-checkpoint", dest="auto_benchmark_best_checkpoint", action="store_false")
    parser.set_defaults(auto_benchmark_best_checkpoint=True)
    parser.add_argument("--auto-benchmark-output-json", default="")

    option_to_dest: dict[str, str] = {}
    for action in parser._actions:
        if not action.option_strings:
            continue
        for opt in action.option_strings:
            option_to_dest[opt] = action.dest
    overridden = {option_to_dest[arg] for arg in raw_argv if arg in option_to_dest}
    config_cli_args = common.config_to_cli_args(
        parser,
        config,
        config_path=config_path,
        overridden_dests=overridden,
    )
    args = parser.parse_args(config_cli_args + raw_argv)
    args.config = str(common.resolve_config_path(args.config, script_dir=SCRIPT_DIR))
    return args


def selection_metric_value(metrics: dict[str, Any], selection_metric: str) -> float:
    if selection_metric == "f1":
        return float(metrics.get("eval_f1", 0.0))
    if selection_metric == "f1_macro":
        return float(metrics.get("eval_f1_macro", 0.0))
    return float(metrics.get("eval_miou", 0.0))


def build_async_checkpoint_eval_command(
    *,
    args: argparse.Namespace,
    finetune_id: str,
    checkpoint_step: int,
    metrics_json_path: Path,
    predictions_jsonl_path: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        str((SCRIPT_DIR / "benchmark_neon_tree_detect.py").resolve()),
        "--env-file",
        str(args.env_file),
        "--api-key-env-var",
        str(args.api_key_env_var),
        "--base-url",
        str(args.base_url),
        "--hf-dataset-repo-id",
        str(args.hf_dataset_repo_id),
        "--hf-dataset-revision",
        str(args.hf_dataset_revision),
        "--split",
        str(args.val_split),
        "--base-model",
        str(args.base_model),
        "--finetune-id",
        str(finetune_id),
        "--checkpoint-step",
        str(int(checkpoint_step)),
        "--checkpoint-fallback-policy",
        "exact",
        "--checkpoint-ready-max-wait-s",
        "300",
        "--checkpoint-ready-poll-interval-s",
        "5",
        "--prompt",
        str(args.prompt),
        "--temperature",
        "0",
        "--top-p",
        "1",
        "--max-tokens",
        str(int(args.max_tokens)),
        "--max-objects",
        str(int(args.max_objects)),
        "--timeout",
        str(float(args.timeout)),
        "--request-retries",
        str(int(args.request_retries)),
        "--request-retry-backoff-s",
        str(float(args.request_retry_backoff_s)),
        "--output-json",
        str(metrics_json_path),
        "--predictions-jsonl",
        str(predictions_jsonl_path),
    ]
    if bool(args.tiling_enabled):
        cmd.append("--tiling-enabled")
    else:
        cmd.append("--no-tiling-enabled")
    cmd.extend(["--tile-width", str(int(args.tile_width))])
    cmd.extend(["--tile-height", str(int(args.tile_height))])
    cmd.extend(["--tile-overlap-x", str(int(args.tile_overlap_x))])
    cmd.extend(["--tile-overlap-y", str(int(args.tile_overlap_y))])
    cmd.extend(["--merge-iou-threshold", str(float(args.merge_iou_threshold))])
    if int(args.eval_max_samples or 0) > 0:
        cmd.extend(["--max-samples", str(int(args.eval_max_samples))])
    if str(args.hf_cache_dir or "").strip():
        cmd.extend(["--hf-cache-dir", str(args.hf_cache_dir)])
    return cmd


def ingest_async_results(
    *,
    results: list[CheckpointEvalResult],
    run: Any,
    selection_metric: str,
    baseline_metrics: Optional[dict[str, Any]],
    log_step: int,
    current_best_metric: Optional[float],
    current_best_step: Optional[int],
    current_best_checkpoint_step: Optional[int],
) -> tuple[Optional[float], Optional[int], Optional[int], int]:
    success_count = 0
    baseline_selection = selection_metric_value(baseline_metrics or {}, selection_metric) if baseline_metrics else None
    for result in results:
        source_step = int(result.metadata.get("step_for_log", result.checkpoint_step))
        if result.status != "succeeded" or result.metrics_payload is None:
            print(
                f"async checkpoint eval failed step={source_step} "
                f"checkpoint_step={result.checkpoint_step} log={result.stdout_log_path}"
            )
            continue
        success_count += 1
        metrics = dict(result.metrics_payload)
        payload = {
            "eval_f1": float(metrics.get("eval_f1", 0.0)),
            "eval_f1_macro": float(metrics.get("eval_f1_macro", 0.0)),
            "eval_miou": float(metrics.get("eval_miou", 0.0)),
            "async_eval_checkpoint_step": int(result.checkpoint_step),
            "async_eval_source_step": int(source_step),
        }
        if baseline_selection is not None:
            payload["eval_selection_metric_delta_vs_baseline"] = selection_metric_value(metrics, selection_metric) - baseline_selection
        wandb.log(payload, step=int(log_step))
        metric = selection_metric_value(metrics, selection_metric)
        if current_best_metric is None or metric > current_best_metric:
            current_best_metric = metric
            current_best_step = int(source_step)
            current_best_checkpoint_step = int(result.checkpoint_step)
            run.summary["best_eval_f1"] = float(metrics.get("eval_f1", 0.0))
            run.summary["best_eval_f1_macro"] = float(metrics.get("eval_f1_macro", 0.0))
            run.summary["best_eval_miou"] = float(metrics.get("eval_miou", 0.0))
            run.summary["best_checkpoint_step"] = int(result.checkpoint_step)
        run.summary["latest_checkpoint_step"] = int(result.checkpoint_step)
        print(
            f"async checkpoint eval completed step={source_step} checkpoint_step={result.checkpoint_step} "
            f"{selection_metric}={metric:.4f} logged_at_step={int(log_step)}"
        )
    return current_best_metric, current_best_step, current_best_checkpoint_step, success_count


def run_benchmark_now(
    *,
    args: argparse.Namespace,
    model: str,
    finetune_id: str,
    checkpoint_step: Optional[int],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    tiling = common.tiling_config_from_args(args)
    metrics, _ = benchmark_mod.evaluate_rows(
        rows=rows,
        model=model,
        api_base=str(args.base_url),
        api_key=str(args.api_key),
        prompt=str(args.prompt),
        tiling=tiling,
        temperature=0.0,
        top_p=1.0,
        max_tokens=int(args.max_tokens),
        max_objects=int(args.max_objects),
        timeout=float(args.timeout),
        request_retries=int(args.request_retries),
        request_retry_backoff_s=float(args.request_retry_backoff_s),
    )
    return {
        "split": str(args.val_split),
        "model": model,
        "finetune_id": str(finetune_id or ""),
        "checkpoint_step": checkpoint_step,
        **metrics,
    }


def validate_non_tiled_train_images(
    *,
    args: argparse.Namespace,
    train_ds: Any,
) -> None:
    if bool(args.tiling_enabled):
        return
    for row in common.iter_rows(train_ds):
        image = common.load_image(row.get("image"))
        pixels = int(image.width * image.height)
        if pixels <= MAX_NO_TILE_TRAIN_PIXELS:
            continue
        data_url_chars = len(common.to_data_url(image))
        data_url_mb = float(data_url_chars) / 1_000_000.0
        estimated_batch_mb = data_url_mb * float(max(1, int(args.batch_size)))
        source_image_id = common.source_image_id_from_row(dict(row), fallback="row")
        raise ValueError(
            "tiling must stay enabled for this dataset. "
            f"Found non-tiled train image {source_image_id} at {image.width}x{image.height} "
            f"({pixels:,} pixels); its JPEG data URL is about {data_url_mb:.1f} MB. "
            f"With batch_size={int(args.batch_size)}, the train_step request is roughly "
            f"{estimated_batch_mb:.1f} MB before rollout metadata, which exceeds the staging payload limit "
            "and leads to 413 Payload Too Large. "
            "Use tiling, or downscale/rebuild the training dataset before disabling tiling."
        )


def validate_args(args: argparse.Namespace) -> None:
    if str(args.wandb_log_profile).strip().lower() != "lean":
        raise ValueError("wandb_log_profile must be 'lean'")
    if not str(args.hf_dataset_repo_id or "").strip():
        raise ValueError("hf_dataset_repo_id is required")
    if not (0.0 <= float(args.off_policy_mix_ratio) <= 1.0):
        raise ValueError("--off-policy-mix-ratio must be in [0,1]")
    if int(args.off_policy_buffer_size) <= 0:
        raise ValueError("--off-policy-buffer-size must be > 0")
    if int(args.off_policy_warmup_steps) < 0:
        raise ValueError("--off-policy-warmup-steps must be >= 0")
    if int(args.off_policy_min_buffer_groups) <= 0:
        raise ValueError("--off-policy-min-buffer-groups must be > 0")
    if int(args.off_policy_min_buffer_groups) > int(args.off_policy_buffer_size):
        raise ValueError("--off-policy-min-buffer-groups must be <= --off-policy-buffer-size")
    if float(args.reward_fn_beta) <= 0.0:
        raise ValueError("--reward-fn-beta must be > 0")
    if float(args.reward_oversize_ratio_cap) <= 0.0:
        raise ValueError("--reward-oversize-ratio-cap must be > 0")
    if float(args.reward_huge_area_cap) <= 0.0:
        raise ValueError("--reward-huge-area-cap must be > 0")
    if float(args.reward_collapse_centers_cap) <= 0.0:
        raise ValueError("--reward-collapse-centers-cap must be > 0")


def compose_train_groups(
    *,
    on_policy_groups: list[TrainStepGroup],
    replay_groups: deque[TrainStepGroup],
    off_policy: bool,
    off_policy_mix_ratio: float,
    off_policy_warmup_steps: int,
    off_policy_min_buffer_groups: int,
    global_step: int,
    rng: random.Random,
) -> tuple[list[TrainStepGroup], int]:
    if (
        not on_policy_groups
        or not off_policy
        or off_policy_mix_ratio <= 0.0
        or global_step < off_policy_warmup_steps
        or len(replay_groups) < off_policy_min_buffer_groups
    ):
        return list(on_policy_groups), 0
    off_policy_count = min(
        max(1, int(round(len(on_policy_groups) * off_policy_mix_ratio))),
        len(on_policy_groups),
        len(replay_groups),
    )
    keep_count = max(0, len(on_policy_groups) - off_policy_count)
    selected_on_policy = (
        list(on_policy_groups)
        if keep_count >= len(on_policy_groups)
        else rng.sample(list(on_policy_groups), k=keep_count)
    )
    mixed = selected_on_policy + rng.sample(list(replay_groups), k=off_policy_count)
    rng.shuffle(mixed)
    return mixed, off_policy_count


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    validate_args(args)

    args.api_key = common.resolve_api_key(
        api_key=args.api_key,
        api_key_env_var=args.api_key_env_var,
        env_file=args.env_file,
    )
    args.hf_token = common.resolve_hf_token(args.hf_token, env_file=args.env_file)
    args.async_checkpoint_eval_dir = str(common.resolve_config_path(args.async_checkpoint_eval_dir, script_dir=SCRIPT_DIR))

    train_ds = common.hf_split_rows(
        repo_id=str(args.hf_dataset_repo_id),
        split=str(args.train_split),
        revision=str(args.hf_dataset_revision),
        hf_token=str(args.hf_token),
        cache_dir=str(args.hf_cache_dir),
    )
    val_ds = common.hf_split_rows(
        repo_id=str(args.hf_dataset_repo_id),
        split=str(args.val_split),
        revision=str(args.hf_dataset_revision),
        hf_token=str(args.hf_token),
        cache_dir=str(args.hf_cache_dir),
    )
    val_rows = [dict(row) for row in common.iter_rows(val_ds)]
    if int(args.eval_max_samples or 0) > 0:
        val_rows = val_rows[: int(args.eval_max_samples)]
    train_size = len(train_ds)
    if train_size <= 0:
        raise ValueError("train split is empty")
    validate_non_tiled_train_images(args=args, train_ds=train_ds)

    rng = random.Random(int(args.seed))
    tiling = common.tiling_config_from_args(args)
    client = TunaClient(api_key=args.api_key, base_url=args.base_url)
    finetune = client.get_finetune(str(args.finetune_id)) if str(args.finetune_id or "").strip() else client.create_finetune(
        name=str(args.finetune_name or f"neon-tree-detect-{common.random_suffix()}"),
        rank=int(args.rank),
    )

    run = wandb.init(
        project=str(args.wandb_project),
        name=str(args.wandb_run_name or "") or None,
        config={
            "config": str(args.config),
            "finetune_id": finetune.finetune_id,
            "finetune_name": finetune.name,
            "hf_dataset_repo_id": args.hf_dataset_repo_id,
            "hf_dataset_revision": args.hf_dataset_revision,
            "train_split": args.train_split,
            "val_split": args.val_split,
            "batch_size": int(args.batch_size),
            "group_size": int(args.group_size),
            "num_steps": int(args.num_steps),
            "lr": float(args.lr),
            "selection_metric": str(args.selection_metric),
            "reward_metric": str(args.reward_metric),
            "reward_fn_beta": float(args.reward_fn_beta),
            "reward_oversize_ratio_cap": float(args.reward_oversize_ratio_cap),
            "reward_huge_area_cap": float(args.reward_huge_area_cap),
            "reward_collapse_centers_cap": float(args.reward_collapse_centers_cap),
            "off_policy": bool(args.off_policy),
            "off_policy_mix_ratio": float(args.off_policy_mix_ratio),
            "off_policy_buffer_size": int(args.off_policy_buffer_size),
            "off_policy_warmup_steps": int(args.off_policy_warmup_steps),
            "off_policy_min_buffer_groups": int(args.off_policy_min_buffer_groups),
            "tiling_enabled": bool(args.tiling_enabled),
            "tile_width": int(args.tile_width),
            "tile_height": int(args.tile_height),
            "tile_overlap_x": int(args.tile_overlap_x),
            "tile_overlap_y": int(args.tile_overlap_y),
            "merge_iou_threshold": float(args.merge_iou_threshold),
            "wandb_log_profile": "lean",
            "api_key_env_var": str(args.api_key_env_var),
        },
    )
    run.summary["finetune_id"] = finetune.finetune_id
    run.summary["finetune_name"] = finetune.name

    print("running baseline benchmark before training...")
    baseline_metrics = run_benchmark_now(
        args=args,
        model=str(args.base_model),
        finetune_id="",
        checkpoint_step=None,
        rows=val_rows,
    )
    baseline_step = 0
    wandb.log(
        {
            "eval_f1": float(baseline_metrics.get("eval_f1", 0.0)),
            "eval_f1_macro": float(baseline_metrics.get("eval_f1_macro", 0.0)),
            "eval_miou": float(baseline_metrics.get("eval_miou", 0.0)),
            "async_eval_checkpoint_step": 0,
        },
        step=baseline_step,
    )
    run.summary["baseline_eval_f1"] = float(baseline_metrics.get("eval_f1", 0.0))
    run.summary["baseline_eval_f1_macro"] = float(baseline_metrics.get("eval_f1_macro", 0.0))
    run.summary["baseline_eval_miou"] = float(baseline_metrics.get("eval_miou", 0.0))
    print(
        f"baseline validation tasks={baseline_metrics['eval_tasks']} "
        f"miou={baseline_metrics['eval_miou']:.4f} f1={baseline_metrics['eval_f1']:.4f} "
        f"macro_f1={baseline_metrics['eval_f1_macro']:.4f}"
    )

    async_jobs: list[DispatchHandle] = []
    replay_buffer: deque[TrainStepGroup] = deque(maxlen=int(args.off_policy_buffer_size))
    best_metric: Optional[float] = None
    best_step: Optional[int] = None
    best_checkpoint_step: Optional[int] = None
    latest_checkpoint_step: Optional[int] = None
    async_eval_success_count = 0
    successful_updates = 0

    for step in range(int(args.num_steps)):
        global_step = int(step) + 1
        if async_jobs:
            async_jobs, completed = poll_checkpoint_eval_jobs(async_jobs)
            best_metric, best_step, best_checkpoint_step, completed_count = ingest_async_results(
                results=completed,
                run=run,
                selection_metric=str(args.selection_metric),
                baseline_metrics=baseline_metrics,
                log_step=global_step,
                current_best_metric=best_metric,
                current_best_step=best_step,
                current_best_checkpoint_step=best_checkpoint_step,
            )
            async_eval_success_count += int(completed_count)

        batch_tasks = [
            common.task_from_row(
                dict(train_ds[rng.randrange(train_size)]),
                tiling=tiling,
                rng=rng,
                include_empty_tiles=bool(args.include_empty_tiles),
                positive_tile_probability=float(args.positive_tile_probability),
            )
            for _ in range(int(args.batch_size))
        ]
        requests = [
            DetectRequest(
                object_name=str(args.prompt),
                image_url=common.to_data_url(task.image),
                settings=DetectSettings(
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                    max_tokens=int(args.max_tokens),
                    max_objects=int(args.max_objects),
                ),
            )
            for task in batch_tasks
        ]

        rollouts_result = finetune.rollouts_batch(
            requests=requests,
            num_rollouts=int(args.group_size),
            max_workers=min(int(args.max_workers), int(args.batch_size)),
        )

        groups: list[TrainStepGroup] = []
        selected_rewards: list[float] = []
        reward_f1_all: list[float] = []
        reward_miou_all: list[float] = []
        reward_hybrid_all: list[float] = []
        reward_soft_fbeta_all: list[float] = []
        reward_loc_all: list[float] = []
        reward_oversize_penalty_all: list[float] = []
        reward_huge_penalty_all: list[float] = []
        reward_collapse_penalty_all: list[float] = []
        train_tp = 0
        train_fp = 0
        train_fn = 0
        total_pred_boxes = 0.0
        giant_box_count = 0.0
        collapse_box_count = 0.0
        for task, result in zip(batch_tasks, rollouts_result):
            rewards: list[float] = []
            for rollout in result.rollouts:
                pred_boxes = rollout.output.objects if isinstance(rollout.output, DetectOutput) else []
                f1_value = common.reward_f1(pred_boxes, task.gt_boxes)
                miou_value = common.reward_miou(pred_boxes, task.gt_boxes)
                hybrid_breakdown = common.hybrid_reward_breakdown(
                    pred_boxes,
                    task.gt_boxes,
                    fn_beta=float(args.reward_fn_beta),
                    oversize_ratio_cap=float(args.reward_oversize_ratio_cap),
                    huge_area_cap=float(args.reward_huge_area_cap),
                    collapse_centers_cap=float(args.reward_collapse_centers_cap),
                )
                reward_f1_all.append(f1_value)
                reward_miou_all.append(miou_value)
                reward_hybrid_all.append(hybrid_breakdown.reward)
                reward_soft_fbeta_all.append(hybrid_breakdown.soft_fbeta)
                reward_loc_all.append(hybrid_breakdown.loc_term)
                reward_oversize_penalty_all.append(hybrid_breakdown.matched_oversize_penalty)
                reward_huge_penalty_all.append(hybrid_breakdown.absolute_huge_penalty)
                reward_collapse_penalty_all.append(hybrid_breakdown.collapse_penalty)
                reward_metric = str(args.reward_metric)
                if reward_metric == "miou":
                    rewards.append(miou_value)
                elif reward_metric == "f1":
                    rewards.append(f1_value)
                else:
                    rewards.append(hybrid_breakdown.reward)
                tp, fp, fn = common.count_tp_fp_fn(pred_boxes, task.gt_boxes)
                train_tp += tp
                train_fp += fp
                train_fn += fn
                total_pred_boxes += float(len(pred_boxes))
                giant_box_count += hybrid_breakdown.giant_box_rate * float(len(pred_boxes))
                collapse_box_count += hybrid_breakdown.collapse_box_rate * float(len(pred_boxes))
            selected_rewards.extend(rewards)
            groups.append(TrainStepGroup(request=result.request, rollouts=list(result.rollouts), rewards=rewards))

        train_groups, off_policy_count = compose_train_groups(
            on_policy_groups=groups,
            replay_groups=replay_buffer,
            off_policy=bool(args.off_policy),
            off_policy_mix_ratio=float(args.off_policy_mix_ratio),
            off_policy_warmup_steps=int(args.off_policy_warmup_steps),
            off_policy_min_buffer_groups=int(args.off_policy_min_buffer_groups),
            global_step=global_step,
            rng=rng,
        )
        train_out = finetune.train_step(groups=train_groups, lr=float(args.lr))
        if bool(args.off_policy):
            replay_buffer.extend(groups)
        successful_updates += 1
        micro_denom = (2 * train_tp) + train_fp + train_fn
        train_f1 = 1.0 if micro_denom == 0 else (2 * train_tp) / float(micro_denom)
        precision_denom = train_tp + train_fp
        recall_denom = train_tp + train_fn
        train_precision = 1.0 if precision_denom == 0 else train_tp / float(precision_denom)
        train_recall = 1.0 if recall_denom == 0 else train_tp / float(recall_denom)
        wandb.log(
            {
                "reward_mean": float(np.mean(selected_rewards)) if selected_rewards else 0.0,
                "reward_f1_mean": float(np.mean(reward_f1_all)) if reward_f1_all else 0.0,
                "reward_miou_mean": float(np.mean(reward_miou_all)) if reward_miou_all else 0.0,
                "reward_hybrid_mean": float(np.mean(reward_hybrid_all)) if reward_hybrid_all else 0.0,
                "reward_soft_fbeta_mean": float(np.mean(reward_soft_fbeta_all)) if reward_soft_fbeta_all else 0.0,
                "reward_loc_mean": float(np.mean(reward_loc_all)) if reward_loc_all else 0.0,
                "reward_oversize_penalty_mean": float(np.mean(reward_oversize_penalty_all))
                if reward_oversize_penalty_all
                else 0.0,
                "reward_huge_penalty_mean": float(np.mean(reward_huge_penalty_all)) if reward_huge_penalty_all else 0.0,
                "reward_collapse_penalty_mean": float(np.mean(reward_collapse_penalty_all))
                if reward_collapse_penalty_all
                else 0.0,
                "train_f1": float(train_f1),
                "train_precision": float(train_precision),
                "train_recall": float(train_recall),
                "giant_box_rate": 0.0 if total_pred_boxes <= 0.0 else float(giant_box_count / total_pred_boxes),
                "collapse_box_rate": 0.0 if total_pred_boxes <= 0.0 else float(collapse_box_count / total_pred_boxes),
                "kl": float(train_out.kl or 0.0),
            },
            step=global_step,
        )
        print(
            f"step {global_step} reward={float(np.mean(selected_rewards)) if selected_rewards else 0.0:.4f} "
            f"f1={train_f1:.4f} precision={train_precision:.4f} recall={train_recall:.4f} "
            f"kl={float(train_out.kl or 0.0):.4f} "
            f"offp={off_policy_count}/{len(train_groups)} replay={len(replay_buffer)}"
        )

        checkpoint_saved_for_eval = False
        if int(args.eval_every) > 0 and successful_updates % int(args.eval_every) == 0:
            saved_checkpoint = finetune.save_checkpoint()
            latest_checkpoint_step = int(getattr(getattr(saved_checkpoint, "checkpoint", None), "step", global_step))
            run.summary["latest_checkpoint_step"] = int(latest_checkpoint_step)
            checkpoint_saved_for_eval = True
            if bool(args.async_checkpoint_eval):
                job = dispatch_checkpoint_eval(
                    trainer="neon_tree_detect",
                    finetune_id=str(finetune.finetune_id),
                    checkpoint_step=int(latest_checkpoint_step),
                    selection_metric=str(args.selection_metric),
                    base_dir=str(args.async_checkpoint_eval_dir),
                    command_builder=lambda metrics_json_path, predictions_jsonl_path, _stdout_log_path: build_async_checkpoint_eval_command(
                        args=args,
                        finetune_id=str(finetune.finetune_id),
                        checkpoint_step=int(latest_checkpoint_step),
                        metrics_json_path=metrics_json_path,
                        predictions_jsonl_path=predictions_jsonl_path,
                    ),
                    metadata={"step_for_log": int(global_step), "split_name": str(args.val_split)},
                    env_overrides={
                        str(args.api_key_env_var): str(args.api_key),
                        "MOONDREAM_API_KEY": str(args.api_key),
                        "HF_TOKEN": str(args.hf_token),
                    },
                    max_inflight=int(args.async_checkpoint_eval_max_inflight),
                    inflight_jobs=async_jobs,
                )
                if job is not None:
                    async_jobs.append(job)
                    print(
                        f"async checkpoint eval dispatched step={global_step} checkpoint_step={latest_checkpoint_step} "
                        f"job_dir={job.job_dir}"
                    )
            else:
                model = f"{str(args.base_model).rstrip('/')}/{finetune.finetune_id}@{int(latest_checkpoint_step)}"
                metrics = run_benchmark_now(
                    args=args,
                    model=model,
                    finetune_id=finetune.finetune_id,
                    checkpoint_step=int(latest_checkpoint_step),
                    rows=val_rows,
                )
                payload = {
                    "eval_f1": float(metrics.get("eval_f1", 0.0)),
                    "eval_f1_macro": float(metrics.get("eval_f1_macro", 0.0)),
                    "eval_miou": float(metrics.get("eval_miou", 0.0)),
                    "async_eval_checkpoint_step": int(latest_checkpoint_step),
                    "eval_selection_metric_delta_vs_baseline": selection_metric_value(metrics, str(args.selection_metric))
                    - selection_metric_value(baseline_metrics, str(args.selection_metric)),
                }
                wandb.log(payload, step=global_step)
                metric = selection_metric_value(metrics, str(args.selection_metric))
                if best_metric is None or metric > best_metric:
                    best_metric = metric
                    best_step = int(global_step)
                    best_checkpoint_step = int(latest_checkpoint_step)
                    run.summary["best_eval_f1"] = float(metrics.get("eval_f1", 0.0))
                    run.summary["best_eval_f1_macro"] = float(metrics.get("eval_f1_macro", 0.0))
                    run.summary["best_eval_miou"] = float(metrics.get("eval_miou", 0.0))
                    run.summary["best_checkpoint_step"] = int(latest_checkpoint_step)

        if int(args.save_every) > 0 and successful_updates % int(args.save_every) == 0 and not checkpoint_saved_for_eval:
            saved_checkpoint = finetune.save_checkpoint()
            latest_checkpoint_step = int(getattr(getattr(saved_checkpoint, "checkpoint", None), "step", global_step))
            run.summary["latest_checkpoint_step"] = int(latest_checkpoint_step)

    final_saved = finetune.save_checkpoint()
    latest_checkpoint_step = int(getattr(getattr(final_saved, "checkpoint", None), "step", successful_updates))
    run.summary["latest_checkpoint_step"] = int(latest_checkpoint_step)

    if bool(args.async_checkpoint_eval) and bool(args.async_checkpoint_eval_drain_on_exit) and async_jobs:
        completed = drain_checkpoint_eval_jobs(async_jobs)
        best_metric, best_step, best_checkpoint_step, completed_count = ingest_async_results(
            results=completed,
            run=run,
            selection_metric=str(args.selection_metric),
            baseline_metrics=baseline_metrics,
            log_step=successful_updates,
            current_best_metric=best_metric,
            current_best_step=best_step,
            current_best_checkpoint_step=best_checkpoint_step,
        )
        async_eval_success_count += int(completed_count)

    run.summary["async_checkpoint_eval_success_count"] = int(async_eval_success_count)
    if best_checkpoint_step is None:
        best_checkpoint_step = latest_checkpoint_step

    if bool(args.auto_benchmark_best_checkpoint) and best_checkpoint_step is not None:
        output_json = str(args.auto_benchmark_output_json or "").strip()
        if not output_json:
            output_json = str(
                common.repo_relative(
                    "outputs",
                    "benchmarks",
                    f"benchmark_best_{finetune.finetune_id}_step{int(best_checkpoint_step)}.json",
                )
            )
        output_path = common.resolve_config_path(output_json, script_dir=SCRIPT_DIR)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = build_async_checkpoint_eval_command(
            args=args,
            finetune_id=str(finetune.finetune_id),
            checkpoint_step=int(best_checkpoint_step),
            metrics_json_path=output_path,
            predictions_jsonl_path=output_path.with_suffix(".predictions.jsonl"),
        )
        env = dict(os.environ)
        env[str(args.api_key_env_var)] = str(args.api_key)
        env["MOONDREAM_API_KEY"] = str(args.api_key)
        if str(args.hf_token).strip():
            env["HF_TOKEN"] = str(args.hf_token)
        subprocess.run(cmd, check=True, env=env)
        if output_path.exists():
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            run.summary["best_eval_f1"] = float(payload.get("eval_f1", 0.0))
            run.summary["best_eval_f1_macro"] = float(payload.get("eval_f1_macro", 0.0))
            run.summary["best_eval_miou"] = float(payload.get("eval_miou", 0.0))
            run.summary["best_checkpoint_step"] = int(best_checkpoint_step)
            print(f"auto benchmark wrote {output_path}")

    run.finish()
    print(
        f"done finetune_id={finetune.finetune_id} "
        f"best_checkpoint_step={best_checkpoint_step} latest_checkpoint_step={latest_checkpoint_step}"
    )


if __name__ == "__main__":
    main()
