#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neon_tree import benchmark_neon_tree_detect as benchmark_mod
from neon_tree import common
from neon_tree import tracking_utils

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = common.repo_relative("configs", "current", "track_neon_tree_video_default.json")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    raw_argv = list(argv) if argv is not None else list(os.sys.argv[1:])
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    pre_args, _ = pre_parser.parse_known_args(raw_argv)
    config_path = common.resolve_config_path(pre_args.config, script_dir=SCRIPT_DIR)
    config = common.load_json_config(config_path, default_path=DEFAULT_CONFIG_PATH)

    parser = argparse.ArgumentParser(description="Track trees across a video or frame directory.")
    parser.add_argument("--config", default=str(config_path))
    parser.add_argument("--env-file", default=str(common.repo_relative(".env.staging")))
    parser.add_argument("--api-key", default="")
    parser.add_argument("--api-key-env-var", default=common.DEFAULT_API_KEY_ENV_VAR)
    parser.add_argument("--base-url", default=common.DEFAULT_STAGING_API_BASE)
    parser.add_argument("--video", default="")
    parser.add_argument("--frames-dir", default="")
    parser.add_argument("--output-jsonl", default=str(common.repo_relative("outputs", "tracking", "tracks.jsonl")))
    parser.add_argument("--render-output", default="")
    parser.add_argument("--base-model", default=common.DEFAULT_BASE_MODEL)
    parser.add_argument("--model", default="")
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
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--request-retries", type=int, default=2)
    parser.add_argument("--request-retry-backoff-s", type=float, default=5.0)
    parser.add_argument("--tiling-enabled", dest="tiling_enabled", action="store_true")
    parser.add_argument("--no-tiling-enabled", dest="tiling_enabled", action="store_false")
    parser.set_defaults(tiling_enabled=True)
    parser.add_argument("--tile-width", type=int, default=1024)
    parser.add_argument("--tile-height", type=int, default=1024)
    parser.add_argument("--tile-overlap-x", type=int, default=128)
    parser.add_argument("--tile-overlap-y", type=int, default=128)
    parser.add_argument("--merge-iou-threshold", type=float, default=0.5)
    parser.add_argument("--track-activation-threshold", type=float, default=0.25)
    parser.add_argument("--lost-track-buffer", type=int, default=30)
    parser.add_argument("--minimum-matching-threshold", type=float, default=0.8)
    parser.add_argument("--minimum-consecutive-frames", type=int, default=1)
    parser.add_argument("--frame-rate", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=0)

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
    if not str(args.video or "").strip() and not str(args.frames_dir or "").strip():
        raise ValueError("Provide --video or --frames-dir")
    return args

def _iter_video_frames(video_path: Path) -> Iterable[tuple[int, str, np.ndarray]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield index, f"{video_path.name}:{index:06d}", frame
            index += 1
    finally:
        cap.release()


def _iter_directory_frames(frames_dir: Path) -> Iterable[tuple[int, str, np.ndarray]]:
    paths = [path for path in sorted(frames_dir.iterdir()) if path.suffix.lower() in common.IMAGE_SUFFIXES]
    for index, path in enumerate(paths):
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        yield index, path.name, frame

def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    args.api_key = common.resolve_api_key(
        api_key=args.api_key,
        api_key_env_var=args.api_key_env_var,
        env_file=args.env_file,
    )
    sv = tracking_utils.require_supervision()
    tracker = tracking_utils.make_tracker(
        sv,
        track_activation_threshold=float(args.track_activation_threshold),
        lost_track_buffer=int(args.lost_track_buffer),
        minimum_matching_threshold=float(args.minimum_matching_threshold),
        minimum_consecutive_frames=int(args.minimum_consecutive_frames),
        frame_rate=int(args.frame_rate),
    )
    model, resolved_step = benchmark_mod.resolve_model(args)
    tiling = common.tiling_config_from_args(args)

    output_jsonl = Path(str(args.output_jsonl)).expanduser().resolve()
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    render_output = str(args.render_output or "").strip()
    render_frames_dir: Optional[Path] = None
    if render_output and str(args.video or "").strip():
        video_path = Path(str(args.video)).expanduser().resolve()
        cap = cv2.VideoCapture(str(video_path))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or float(args.frame_rate))
        cap.release()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(render_output, fourcc, fps or float(args.frame_rate), (width, height))
    elif render_output:
        render_frames_dir = Path(render_output).expanduser().resolve()
        render_frames_dir.mkdir(parents=True, exist_ok=True)

    frame_iter = (
        _iter_video_frames(Path(str(args.video)).expanduser().resolve())
        if str(args.video or "").strip()
        else _iter_directory_frames(Path(str(args.frames_dir)).expanduser().resolve())
    )

    with output_jsonl.open("w", encoding="utf-8") as handle:
        for frame_index, source_frame, frame_bgr in frame_iter:
            if int(args.max_frames or 0) > 0 and frame_index >= int(args.max_frames):
                break
            image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image_rgb)
            boxes = common.runtime_detect(
                image=image,
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
                retries=int(args.request_retries),
                retry_backoff_s=float(args.request_retry_backoff_s),
            )
            xyxy = np.array(
                [
                    [
                        box.x_min * float(image.width),
                        box.y_min * float(image.height),
                        box.x_max * float(image.width),
                        box.y_max * float(image.height),
                    ]
                    for box in boxes
                ],
                dtype=np.float32,
            )
            if len(xyxy) == 0:
                detections = sv.Detections(
                    xyxy=np.empty((0, 4), dtype=np.float32),
                    confidence=np.empty((0,), dtype=np.float32),
                    class_id=np.empty((0,), dtype=int),
                )
            else:
                detections = sv.Detections(
                    xyxy=xyxy,
                    confidence=np.ones((len(xyxy),), dtype=np.float32),
                    class_id=np.zeros((len(xyxy),), dtype=int),
                )
            tracked = tracker.update_with_detections(detections)
            tracked_xyxy = np.asarray(tracked.xyxy) if getattr(tracked, "xyxy", None) is not None else np.empty((0, 4))
            tracked_ids = np.asarray(tracked.tracker_id) if getattr(tracked, "tracker_id", None) is not None else np.empty((0,))
            confidences = np.asarray(tracked.confidence) if getattr(tracked, "confidence", None) is not None else np.ones((len(tracked_xyxy),))

            for box, tracker_id, confidence in zip(tracked_xyxy, tracked_ids, confidences):
                x1, y1, x2, y2 = [float(value) for value in box.tolist()]
                record = {
                    "frame_index": int(frame_index),
                    "source_frame": str(source_frame),
                    "track_id": int(tracker_id),
                    "confidence": float(confidence),
                    "prompt": str(args.prompt),
                    "bbox": {
                        "x_min": common.clamp(x1 / float(image.width)),
                        "y_min": common.clamp(y1 / float(image.height)),
                        "x_max": common.clamp(x2 / float(image.width)),
                        "y_max": common.clamp(y2 / float(image.height)),
                    },
                }
                handle.write(json.dumps(record, sort_keys=True) + "\n")

            if writer is not None:
                writer.write(tracking_utils.render_frame(frame_bgr, tracked_xyxy, tracked_ids))
            elif render_frames_dir is not None:
                rendered = tracking_utils.render_frame(frame_bgr, tracked_xyxy, tracked_ids)
                cv2.imwrite(str(render_frames_dir / f"{frame_index:06d}.jpg"), rendered)

    if writer is not None:
        writer.release()
    print(
        f"tracking complete model={model} finetune_id={args.finetune_id or ''} "
        f"checkpoint_step={resolved_step} output_jsonl={output_jsonl}"
    )


if __name__ == "__main__":
    main()
