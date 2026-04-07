#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finetune_checkpoints import resolve_checkpoint_step
from neon_tree import common
from neon_tree import tracking_utils

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = common.repo_relative("configs", "current", "benchmark_neon_tree_detect_default.json")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    raw_argv = list(argv) if argv is not None else list(os.sys.argv[1:])
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    pre_args, _ = pre_parser.parse_known_args(raw_argv)
    config_path = common.resolve_config_path(pre_args.config, script_dir=SCRIPT_DIR)
    config = common.load_json_config(config_path, default_path=DEFAULT_CONFIG_PATH)

    parser = argparse.ArgumentParser(description="Benchmark NEON tree detect models.")
    parser.add_argument("--config", default=str(config_path))
    parser.add_argument("--env-file", default=str(common.repo_relative(".env.staging")))
    parser.add_argument("--api-key", default="")
    parser.add_argument("--api-key-env-var", default=common.DEFAULT_API_KEY_ENV_VAR)
    parser.add_argument("--base-url", default=common.DEFAULT_STAGING_API_BASE)
    parser.add_argument("--hf-token", default="")
    parser.add_argument("--dataset-source", choices=["hf_hub", "synthetic_flyover"], default="hf_hub")
    parser.add_argument("--hf-dataset-repo-id", default=common.DEFAULT_HF_DATASET_REPO_ID)
    parser.add_argument("--hf-dataset-revision", default=common.DEFAULT_HF_DATASET_REVISION)
    parser.add_argument("--hf-cache-dir", default="")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--clip-manifest", default="")
    parser.add_argument("--synthetic-gt-jsonl", default="")
    parser.add_argument("--synthetic-video", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--base-model", default=common.DEFAULT_BASE_MODEL)
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
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-objects", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--request-retries", type=int, default=2)
    parser.add_argument("--request-retry-backoff-s", type=float, default=5.0)
    parser.add_argument("--prompt", default=common.DEFAULT_PROMPT)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--predictions-jsonl", default="")
    parser.add_argument("--tracking-output-jsonl", default="")
    parser.add_argument("--tracking-render-output", default="")
    parser.add_argument("--track-activation-threshold", type=float, default=0.25)
    parser.add_argument("--lost-track-buffer", type=int, default=30)
    parser.add_argument("--minimum-matching-threshold", type=float, default=0.8)
    parser.add_argument("--minimum-consecutive-frames", type=int, default=1)
    parser.add_argument("--tracking-frame-rate", type=int, default=30)
    parser.add_argument("--tiling-enabled", dest="tiling_enabled", action="store_true")
    parser.add_argument("--no-tiling-enabled", dest="tiling_enabled", action="store_false")
    parser.set_defaults(tiling_enabled=False)
    parser.add_argument("--tile-width", type=int, default=1024)
    parser.add_argument("--tile-height", type=int, default=1024)
    parser.add_argument("--tile-overlap-x", type=int, default=128)
    parser.add_argument("--tile-overlap-y", type=int, default=128)
    parser.add_argument("--merge-iou-threshold", type=float, default=0.5)
    parser.add_argument("--max-samples", type=int, default=0)

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


def resolve_model(args: argparse.Namespace) -> tuple[str, Optional[int]]:
    if str(args.model or "").strip():
        return str(args.model).strip(), args.checkpoint_step
    if not str(args.finetune_id or "").strip():
        return str(args.base_model).strip(), None
    if args.checkpoint_step is None:
        return f"{str(args.base_model).rstrip('/')}/{str(args.finetune_id).strip()}", None
    resolved_step, used_fallback = resolve_checkpoint_step(
        api_base=str(args.base_url),
        api_key=str(args.api_key),
        finetune_id=str(args.finetune_id).strip(),
        requested_step=int(args.checkpoint_step),
        fallback_policy=str(args.checkpoint_fallback_policy),
        ready_max_wait_s=float(args.checkpoint_ready_max_wait_s),
        ready_poll_interval_s=float(args.checkpoint_ready_poll_interval_s),
    )
    if used_fallback:
        print(
            f"warning: requested checkpoint step={int(args.checkpoint_step)} not available; "
            f"using nearest saved step={resolved_step}"
        )
    return f"{str(args.base_model).rstrip('/')}/{str(args.finetune_id).strip()}@{int(resolved_step)}", int(resolved_step)


def evaluate_rows(
    *,
    rows: list[dict[str, Any]],
    model: str,
    api_base: str,
    api_key: str,
    prompt: str,
    tiling: common.TilingConfig,
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_objects: int,
    timeout: float,
    request_retries: int = 2,
    request_retry_backoff_s: float = 5.0,
    detector: Optional[Callable[[Any], list[common.DetectAnnotation]]] = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    total_f1 = 0.0
    total_miou = 0.0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    predictions: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        image = common.load_image(row.get("image"))
        gt_boxes = common.parse_answer_boxes(row.get("answer_boxes"))
        if detector is None:
            pred_boxes = common.runtime_detect(
                image=image,
                model=model,
                api_base=api_base,
                api_key=api_key,
                prompt=prompt,
                tiling=tiling,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                max_objects=max_objects,
                timeout=timeout,
                retries=request_retries,
                retry_backoff_s=request_retry_backoff_s,
            )
        else:
            pred_boxes = detector(row)
        total_f1 += common.reward_f1(pred_boxes, gt_boxes)
        total_miou += common.reward_miou(pred_boxes, gt_boxes)
        tp, fp, fn = common.count_tp_fp_fn(pred_boxes, gt_boxes)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        predictions.append(
            {
                "row_index": int(index),
                "source_image_id": common.source_image_id_from_row(row, fallback=f"row_{index:06d}"),
                "pred_boxes": [box.to_payload() for box in pred_boxes],
                "gt_boxes": [box.to_payload() for box in gt_boxes],
            }
        )

    tasks = len(rows)
    micro_denom = (2 * total_tp) + total_fp + total_fn
    metrics = {
        "eval_tasks": int(tasks),
        "eval_f1": 1.0 if micro_denom == 0 else (2 * total_tp) / float(micro_denom),
        "eval_f1_macro": 0.0 if tasks <= 0 else total_f1 / float(tasks),
        "eval_miou": 0.0 if tasks <= 0 else total_miou / float(tasks),
    }
    return metrics, predictions


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    args.api_key = common.resolve_api_key(
        api_key=args.api_key,
        api_key_env_var=args.api_key_env_var,
        env_file=args.env_file,
    )
    args.hf_token = common.resolve_hf_token(args.hf_token, env_file=args.env_file)
    if str(args.dataset_source) == "hf_hub" and not str(args.hf_dataset_repo_id or "").strip():
        raise ValueError("hf_dataset_repo_id is required")

    rows = common.load_detection_rows(
        dataset_source=str(args.dataset_source),
        split=str(args.split),
        hf_dataset_repo_id=str(args.hf_dataset_repo_id),
        hf_dataset_revision=str(args.hf_dataset_revision),
        hf_token=str(args.hf_token),
        hf_cache_dir=str(args.hf_cache_dir),
        clip_manifest=str(args.clip_manifest),
        synthetic_gt_jsonl=str(args.synthetic_gt_jsonl),
        synthetic_video=str(args.synthetic_video),
    )
    if int(args.max_samples or 0) > 0:
        rows = rows[: int(args.max_samples)]

    model, resolved_step = resolve_model(args)
    tiling = common.tiling_config_from_args(args)
    metrics, predictions = evaluate_rows(
        rows=rows,
        model=model,
        api_base=str(args.base_url),
        api_key=str(args.api_key),
        prompt=str(args.prompt),
        tiling=tiling,
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        max_tokens=int(args.max_tokens),
        max_objects=int(args.max_objects),
        timeout=float(args.timeout),
        request_retries=int(args.request_retries),
        request_retry_backoff_s=float(args.request_retry_backoff_s),
    )
    payload = {
        "dataset_source": str(args.dataset_source),
        "split": str(args.split),
        "model": model,
        "finetune_id": str(args.finetune_id or ""),
        "checkpoint_step": resolved_step,
        **metrics,
    }
    if str(args.tracking_output_jsonl or "").strip() or str(args.tracking_render_output or "").strip():
        if str(args.dataset_source) != "synthetic_flyover":
            raise ValueError("tracking outputs are only supported for dataset_source=synthetic_flyover")
        if not str(args.tracking_output_jsonl or "").strip():
            raise ValueError("tracking_output_jsonl is required when requesting benchmark tracking output")
        tracking_summary = tracking_utils.track_predictions(
            rows=rows,
            predictions=predictions,
            prompt=str(args.prompt),
            output_jsonl=str(args.tracking_output_jsonl),
            render_output=str(args.tracking_render_output),
            frame_rate=int(args.tracking_frame_rate),
            track_activation_threshold=float(args.track_activation_threshold),
            lost_track_buffer=int(args.lost_track_buffer),
            minimum_matching_threshold=float(args.minimum_matching_threshold),
            minimum_consecutive_frames=int(args.minimum_consecutive_frames),
            max_frames=int(args.max_samples or 0),
        )
        payload.update(tracking_summary)

    if str(args.output_json or "").strip():
        output_path = Path(str(args.output_json)).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if str(args.predictions_jsonl or "").strip():
        predictions_path = Path(str(args.predictions_jsonl)).expanduser().resolve()
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        with predictions_path.open("w", encoding="utf-8") as handle:
            for row in predictions:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    print(
        f"benchmark split={args.split} tasks={payload['eval_tasks']} "
        f"miou={payload['eval_miou']:.4f} f1={payload['eval_f1']:.4f} "
        f"macro_f1={payload['eval_f1_macro']:.4f}"
    )


if __name__ == "__main__":
    main()
