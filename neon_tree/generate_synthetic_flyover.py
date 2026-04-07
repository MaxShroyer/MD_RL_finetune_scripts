#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neon_tree import common

Image.MAX_IMAGE_PIXELS = None

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = common.repo_relative("configs", "current", "generate_synthetic_flyover_default.json")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    raw_argv = list(argv) if argv is not None else list(os.sys.argv[1:])
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    pre_args, _ = pre_parser.parse_known_args(raw_argv)
    config_path = common.resolve_config_path(pre_args.config, script_dir=SCRIPT_DIR)
    config = common.load_json_config(config_path, default_path=DEFAULT_CONFIG_PATH)

    parser = argparse.ArgumentParser(description="Generate smooth synthetic flyover clips from NEON training rasters.")
    parser.add_argument("--config", default=str(config_path))
    parser.add_argument("--env-file", default=str(common.repo_relative(".env.staging")))
    parser.add_argument("--hf-token", default="")
    parser.add_argument("--dataset-source", choices=["hf_hub"], default="hf_hub")
    parser.add_argument("--hf-dataset-repo-id", default=common.DEFAULT_HF_DATASET_REPO_ID)
    parser.add_argument("--hf-dataset-revision", default=common.DEFAULT_HF_DATASET_REVISION)
    parser.add_argument("--hf-cache-dir", default="")
    parser.add_argument("--source-split", default="train")
    parser.add_argument("--window-width", type=int, default=400)
    parser.add_argument("--window-height", type=int, default=400)
    parser.add_argument("--step-x", type=int, default=200)
    parser.add_argument("--step-y", type=int, default=200)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--path-style", choices=["serpentine"], default="serpentine")
    parser.add_argument("--max-source-rasters", type=int, default=0)
    parser.add_argument("--source-selection", choices=["most_boxes", "largest_area", "lexical"], default="most_boxes")
    parser.add_argument("--output-root", default=str(common.repo_relative("outputs", "flyovers")))

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


def _render_gt_frame(frame_bgr: np.ndarray, boxes: list[common.DetectAnnotation], track_ids: list[str]) -> np.ndarray:
    rendered = frame_bgr.copy()
    height, width = rendered.shape[:2]
    for box, track_id in zip(boxes, track_ids):
        x1 = int(round(box.x_min * float(width)))
        y1 = int(round(box.y_min * float(height)))
        x2 = int(round(box.x_max * float(width)))
        y2 = int(round(box.y_max * float(height)))
        cv2.rectangle(rendered, (x1, y1), (x2, y2), (0, 180, 255), 2)
        cv2.putText(rendered, str(track_id), (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 180, 255), 1)
    return rendered


def _row_meta(
    row: dict[str, Any],
    *,
    window_width: int,
    window_height: int,
) -> Optional[dict[str, Any]]:
    image = common.load_image(row.get("image"))
    width, height = image.size
    try:
        image.close()
    except Exception:
        pass
    if width < int(window_width) or height < int(window_height):
        return None
    boxes = common.parse_answer_boxes(row.get("answer_boxes"))
    source_image_id = common.source_image_id_from_row(row, fallback="row")
    return {
        "row": row,
        "source_image_id": source_image_id,
        "tree_count": len(boxes),
        "width": width,
        "height": height,
        "area": width * height,
    }


def select_source_rows(
    rows: list[dict[str, Any]],
    *,
    window_width: int,
    window_height: int,
    max_source_rasters: int,
    source_selection: str,
) -> list[dict[str, Any]]:
    candidates = [
        meta
        for meta in (_row_meta(row, window_width=window_width, window_height=window_height) for row in rows)
        if meta is not None
    ]
    if str(source_selection) == "largest_area":
        candidates.sort(key=lambda item: (-int(item["area"]), str(item["source_image_id"])))
    elif str(source_selection) == "lexical":
        candidates.sort(key=lambda item: str(item["source_image_id"]))
    else:
        candidates.sort(key=lambda item: (-int(item["tree_count"]), str(item["source_image_id"])))
    limit = int(max_source_rasters)
    if limit <= 0:
        return candidates
    return candidates[:limit]


def generate_clip_from_row(
    *,
    row: dict[str, Any],
    source_image_id: str,
    output_dir: Path,
    window_width: int,
    window_height: int,
    step_x: int,
    step_y: int,
    fps: int,
    path_style: str,
) -> dict[str, Any]:
    image = common.load_image(row.get("image"))
    gt_boxes = common.parse_answer_boxes(row.get("answer_boxes"))
    gt_track_ids = common.synthetic_track_ids(source_image_id, len(gt_boxes))
    windows = common.build_flyover_windows(
        width=image.width,
        height=image.height,
        window_width=window_width,
        window_height=window_height,
        step_x=step_x,
        step_y=step_y,
        path_style=path_style,
    )
    if not windows:
        raise ValueError(f"No eligible flyover windows for {source_image_id}")

    clip_dir = output_dir / source_image_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    clean_video_path = clip_dir / "clean.mp4"
    overlay_video_path = clip_dir / "overlay_gt.mp4"
    gt_jsonl_path = clip_dir / "gt.jsonl"
    manifest_path = clip_dir / "clip_manifest.json"
    clean_writer = cv2.VideoWriter(
        str(clean_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (int(window_width), int(window_height)),
    )
    overlay_writer = cv2.VideoWriter(
        str(overlay_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (int(window_width), int(window_height)),
    )
    if not clean_writer.isOpened() or not overlay_writer.isOpened():
        raise RuntimeError(f"Unable to open flyover writers for {source_image_id}")

    with gt_jsonl_path.open("w", encoding="utf-8") as handle:
        for frame_index, window in enumerate(windows):
            crop = image.crop((window.left, window.top, window.right, window.bottom)).convert("RGB")
            frame_bgr = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2BGR)
            frame_boxes, frame_track_ids = common.project_boxes_to_window(gt_boxes, window=window, track_ids=gt_track_ids)
            record = {
                "clip_id": source_image_id,
                "source_image_id": source_image_id,
                "frame_index": int(frame_index),
                "window_bounds": {
                    "left": int(window.left),
                    "top": int(window.top),
                    "right": int(window.right),
                    "bottom": int(window.bottom),
                },
                "gt_boxes": common.boxes_to_payload(frame_boxes),
                "gt_track_ids": [str(track_id) for track_id in frame_track_ids],
                "frame_width": int(window_width),
                "frame_height": int(window_height),
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            clean_writer.write(frame_bgr)
            overlay_writer.write(_render_gt_frame(frame_bgr, frame_boxes, frame_track_ids))

    clean_writer.release()
    overlay_writer.release()
    try:
        image.close()
    except Exception:
        pass

    manifest = {
        "clip_id": source_image_id,
        "source_image_id": source_image_id,
        "tree_count": len(gt_boxes),
        "frame_count": len(windows),
        "frame_width": int(window_width),
        "frame_height": int(window_height),
        "window_width": int(window_width),
        "window_height": int(window_height),
        "step_x": int(step_x),
        "step_y": int(step_y),
        "fps": int(fps),
        "path_style": str(path_style),
        "clean_video_path": str(clean_video_path.resolve()),
        "overlay_video_path": str(overlay_video_path.resolve()),
        "gt_jsonl_path": str(gt_jsonl_path.resolve()),
        "manifest_path": str(manifest_path.resolve()),
    }
    common.write_json(manifest_path, manifest)
    return manifest


def generate_flyover_clips(args: argparse.Namespace) -> dict[str, Any]:
    hf_token = common.resolve_hf_token(str(args.hf_token), env_file=str(args.env_file))
    rows = common.load_detection_rows(
        dataset_source=str(args.dataset_source),
        split=str(args.source_split),
        hf_dataset_repo_id=str(args.hf_dataset_repo_id),
        hf_dataset_revision=str(args.hf_dataset_revision),
        hf_token=str(hf_token),
        hf_cache_dir=str(args.hf_cache_dir),
    )
    selected = select_source_rows(
        rows,
        window_width=int(args.window_width),
        window_height=int(args.window_height),
        max_source_rasters=int(args.max_source_rasters),
        source_selection=str(args.source_selection),
    )
    if not selected:
        raise RuntimeError("No eligible training rasters found for synthetic flyover generation.")

    run_dir = Path(str(args.output_root)).expanduser().resolve() / f"synthetic_flyover_{common.timestamp_slug()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifests = [
        generate_clip_from_row(
            row=dict(item["row"]),
            source_image_id=str(item["source_image_id"]),
            output_dir=run_dir,
            window_width=int(args.window_width),
            window_height=int(args.window_height),
            step_x=int(args.step_x),
            step_y=int(args.step_y),
            fps=int(args.fps),
            path_style=str(args.path_style),
        )
        for item in selected
    ]
    run_manifest = {
        "run_dir": str(run_dir),
        "dataset_source": str(args.dataset_source),
        "source_split": str(args.source_split),
        "window_width": int(args.window_width),
        "window_height": int(args.window_height),
        "step_x": int(args.step_x),
        "step_y": int(args.step_y),
        "fps": int(args.fps),
        "path_style": str(args.path_style),
        "max_source_rasters": int(args.max_source_rasters),
        "source_selection": str(args.source_selection),
        "clip_manifests": [str(item["manifest_path"]) for item in manifests],
    }
    common.write_json(run_dir / "run_manifest.json", run_manifest)
    return {"run_dir": run_dir, "clips": manifests, "run_manifest": run_manifest}


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    result = generate_flyover_clips(args)
    print(
        f"generated_synthetic_flyover clips={len(result['clips'])} "
        f"run_dir={result['run_dir']}"
    )


if __name__ == "__main__":
    main()
