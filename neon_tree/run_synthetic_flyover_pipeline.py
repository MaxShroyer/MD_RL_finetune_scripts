#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neon_tree import benchmark_neon_tree_detect as benchmark_mod
from neon_tree import common
from neon_tree import generate_synthetic_flyover as flyover_mod
from neon_tree import track_neon_tree_video as track_mod

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = common.repo_relative("configs", "current", "run_synthetic_flyover_pipeline_default.json")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    raw_argv = list(argv) if argv is not None else list(os.sys.argv[1:])
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    pre_args, _ = pre_parser.parse_known_args(raw_argv)
    config_path = common.resolve_config_path(pre_args.config, script_dir=SCRIPT_DIR)
    config = common.load_json_config(config_path, default_path=DEFAULT_CONFIG_PATH)

    parser = argparse.ArgumentParser(description="Generate, benchmark, and track a smooth synthetic NEON flyover clip.")
    parser.add_argument("--config", default=str(config_path))
    parser.add_argument("--env-file", default=str(common.repo_relative(".env.staging")))
    parser.add_argument("--api-key", default="")
    parser.add_argument("--api-key-env-var", default=common.DEFAULT_API_KEY_ENV_VAR)
    parser.add_argument("--base-url", default=common.DEFAULT_STAGING_API_BASE)
    parser.add_argument("--hf-token", default="")
    parser.add_argument("--dataset-source", choices=["hf_hub"], default="hf_hub")
    parser.add_argument("--hf-dataset-repo-id", default=common.DEFAULT_HF_DATASET_REPO_ID)
    parser.add_argument("--hf-dataset-revision", default=common.DEFAULT_HF_DATASET_REVISION)
    parser.add_argument("--hf-cache-dir", default="")
    parser.add_argument("--source-split", default="train")
    parser.add_argument("--window-width", type=int, default=400)
    parser.add_argument("--window-height", type=int, default=400)
    parser.add_argument("--step-x", type=int, default=32)
    parser.add_argument("--step-y", type=int, default=32)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--path-style", choices=["serpentine"], default="serpentine")
    parser.add_argument("--max-source-rasters", type=int, default=1)
    parser.add_argument("--source-selection", choices=["most_boxes", "largest_area", "lexical"], default="most_boxes")
    parser.add_argument("--output-root", default=str(common.repo_relative("outputs", "flyovers")))
    parser.add_argument("--model", default="")
    parser.add_argument("--base-model", default=common.DEFAULT_BASE_MODEL)
    parser.add_argument("--finetune-id", default="")
    parser.add_argument("--checkpoint-step", type=int, default=-1)
    parser.add_argument("--checkpoint-fallback-policy", choices=["nearest_saved", "exact"], default="nearest_saved")
    parser.add_argument("--checkpoint-ready-max-wait-s", type=float, default=0.0)
    parser.add_argument("--checkpoint-ready-poll-interval-s", type=float, default=5.0)
    parser.add_argument("--prompt", default=common.DEFAULT_PROMPT)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-objects", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--request-retries", type=int, default=3)
    parser.add_argument("--request-retry-backoff-s", type=float, default=8.0)
    parser.add_argument("--tiling-enabled", dest="tiling_enabled", action="store_true")
    parser.add_argument("--no-tiling-enabled", dest="tiling_enabled", action="store_false")
    parser.set_defaults(tiling_enabled=True)
    parser.add_argument("--tile-width", type=int, default=1024)
    parser.add_argument("--tile-height", type=int, default=1024)
    parser.add_argument("--tile-overlap-x", type=int, default=128)
    parser.add_argument("--tile-overlap-y", type=int, default=128)
    parser.add_argument("--merge-iou-threshold", type=float, default=0.5)
    parser.add_argument("--benchmark-max-samples", type=int, default=256)
    parser.add_argument("--tracking-max-frames", type=int, default=256)
    parser.add_argument("--track-activation-threshold", type=float, default=0.25)
    parser.add_argument("--lost-track-buffer", type=int, default=30)
    parser.add_argument("--minimum-matching-threshold", type=float, default=0.8)
    parser.add_argument("--minimum-consecutive-frames", type=int, default=1)
    parser.add_argument("--track-frame-rate", type=int, default=12)

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
    if int(args.checkpoint_step) < 0:
        args.checkpoint_step = None
    return args


def _benchmark_clip(args: argparse.Namespace, *, clip: dict[str, Any], model: str, resolved_step: Optional[int]) -> Path:
    rows = common.load_detection_rows(
        dataset_source="synthetic_flyover",
        split="synthetic",
        hf_dataset_repo_id="",
        hf_dataset_revision="",
        hf_token="",
        hf_cache_dir="",
        clip_manifest=str(clip["manifest_path"]),
    )
    if int(args.benchmark_max_samples or 0) > 0:
        rows = rows[: int(args.benchmark_max_samples)]
    metrics, _ = benchmark_mod.evaluate_rows(
        rows=rows,
        model=model,
        api_base=str(args.base_url),
        api_key=str(args.api_key),
        prompt=str(args.prompt),
        tiling=common.tiling_config_from_args(args),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        max_tokens=int(args.max_tokens),
        max_objects=int(args.max_objects),
        timeout=float(args.timeout),
        request_retries=int(args.request_retries),
        request_retry_backoff_s=float(args.request_retry_backoff_s),
    )
    payload = {
        "dataset_source": "synthetic_flyover",
        "split": "synthetic",
        "clip_id": str(clip["clip_id"]),
        "source_image_id": str(clip["source_image_id"]),
        "model": model,
        "finetune_id": str(args.finetune_id or ""),
        "checkpoint_step": resolved_step,
        **metrics,
    }
    output_path = Path(str(clip["manifest_path"])).resolve().parent / "benchmark.json"
    common.write_json(output_path, payload)
    return output_path


def _track_clip(args: argparse.Namespace, *, clip: dict[str, Any], model: str) -> dict[str, str]:
    clip_dir = Path(str(clip["manifest_path"])).resolve().parent
    output_jsonl = clip_dir / "predicted_tracks.jsonl"
    render_output = clip_dir / "predicted_tracks.mp4"
    track_mod.main(
        [
            "--env-file",
            str(args.env_file),
            "--api-key",
            str(args.api_key),
            "--base-url",
            str(args.base_url),
            "--video",
            str(clip["clean_video_path"]),
            "--model",
            str(model),
            "--prompt",
            str(args.prompt),
            "--temperature",
            str(args.temperature),
            "--top-p",
            str(args.top_p),
            "--max-tokens",
            str(args.max_tokens),
            "--max-objects",
            str(args.max_objects),
            "--timeout",
            str(args.timeout),
            "--request-retries",
            str(args.request_retries),
            "--request-retry-backoff-s",
            str(args.request_retry_backoff_s),
            "--output-jsonl",
            str(output_jsonl),
            "--render-output",
            str(render_output),
            "--track-activation-threshold",
            str(args.track_activation_threshold),
            "--lost-track-buffer",
            str(args.lost_track_buffer),
            "--minimum-matching-threshold",
            str(args.minimum_matching_threshold),
            "--minimum-consecutive-frames",
            str(args.minimum_consecutive_frames),
            "--frame-rate",
            str(args.track_frame_rate),
            "--max-frames",
            str(args.tracking_max_frames),
            "--tile-width",
            str(args.tile_width),
            "--tile-height",
            str(args.tile_height),
            "--tile-overlap-x",
            str(args.tile_overlap_x),
            "--tile-overlap-y",
            str(args.tile_overlap_y),
            "--merge-iou-threshold",
            str(args.merge_iou_threshold),
            "--tiling-enabled" if bool(args.tiling_enabled) else "--no-tiling-enabled",
        ]
    )
    return {
        "predicted_tracks_jsonl": str(output_jsonl.resolve()),
        "predicted_tracks_video": str(render_output.resolve()),
    }


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    args.api_key = common.resolve_api_key(
        api_key=args.api_key,
        api_key_env_var=args.api_key_env_var,
        env_file=args.env_file,
    )
    args.hf_token = common.resolve_hf_token(args.hf_token, env_file=args.env_file)
    generated = flyover_mod.generate_flyover_clips(args)
    model, resolved_step = benchmark_mod.resolve_model(args)

    outputs: list[dict[str, Any]] = []
    for clip in generated["clips"]:
        benchmark_json = _benchmark_clip(args, clip=clip, model=model, resolved_step=resolved_step)
        tracking_outputs = _track_clip(args, clip=clip, model=model)
        outputs.append(
            {
                "clip_manifest": str(clip["manifest_path"]),
                "benchmark_json": str(benchmark_json.resolve()),
                **tracking_outputs,
            }
        )

    summary_path = Path(generated["run_dir"]) / "pipeline_summary.json"
    common.write_json(
        summary_path,
        {
            "run_dir": str(generated["run_dir"]),
            "model": model,
            "finetune_id": str(args.finetune_id or ""),
            "checkpoint_step": resolved_step,
            "outputs": outputs,
        },
    )
    print(f"synthetic_flyover_pipeline_complete clips={len(outputs)} run_dir={generated['run_dir']}")


if __name__ == "__main__":
    main()
