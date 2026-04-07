from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import cv2
import numpy as np

from neon_tree import common


VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def require_supervision():
    try:
        import supervision as sv  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("supervision is required for ByteTrack tracking. Install it with `pip install supervision`.") from exc
    return sv


def make_tracker(
    sv: Any,
    *,
    track_activation_threshold: float,
    lost_track_buffer: int,
    minimum_matching_threshold: float,
    minimum_consecutive_frames: int,
    frame_rate: int,
) -> Any:
    kwargs_new = {
        "track_activation_threshold": float(track_activation_threshold),
        "lost_track_buffer": int(lost_track_buffer),
        "minimum_matching_threshold": float(minimum_matching_threshold),
        "minimum_consecutive_frames": int(minimum_consecutive_frames),
        "frame_rate": int(frame_rate),
    }
    try:
        return sv.ByteTrack(**kwargs_new)
    except TypeError:
        kwargs_old = {
            "track_thresh": float(track_activation_threshold),
            "track_buffer": int(lost_track_buffer),
            "match_thresh": float(minimum_matching_threshold),
            "frame_rate": int(frame_rate),
        }
        return sv.ByteTrack(**kwargs_old)


def render_frame(frame: np.ndarray, xyxy: np.ndarray, tracker_ids: np.ndarray) -> np.ndarray:
    out = frame.copy()
    for box, tracker_id in zip(xyxy, tracker_ids):
        x1, y1, x2, y2 = [int(value) for value in box.tolist()]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(out, f"#{int(tracker_id)}", (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
    return out


def _boxes_to_xyxy(boxes: Sequence[common.DetectAnnotation], *, width: int, height: int) -> np.ndarray:
    return np.array(
        [
            [
                box.x_min * float(width),
                box.y_min * float(height),
                box.x_max * float(width),
                box.y_max * float(height),
            ]
            for box in boxes
        ],
        dtype=np.float32,
    )


def _make_render_target(
    *,
    render_output: str,
    width: int,
    height: int,
    frame_rate: int,
) -> tuple[Optional[Any], Optional[Path]]:
    output = str(render_output or "").strip()
    if not output:
        return None, None
    path = Path(output).expanduser().resolve()
    if path.suffix.lower() in VIDEO_SUFFIXES:
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(frame_rate), (int(width), int(height)))
        return writer, None
    path.mkdir(parents=True, exist_ok=True)
    return None, path


def track_predictions(
    *,
    rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    prompt: str,
    output_jsonl: str,
    render_output: str,
    frame_rate: int,
    track_activation_threshold: float,
    lost_track_buffer: int,
    minimum_matching_threshold: float,
    minimum_consecutive_frames: int,
    max_frames: int = 0,
) -> dict[str, Any]:
    if len(rows) != len(predictions):
        raise ValueError(f"rows/predictions length mismatch: {len(rows)} != {len(predictions)}")
    if not rows:
        raise ValueError("No rows to track")

    sv = require_supervision()
    tracker = make_tracker(
        sv,
        track_activation_threshold=track_activation_threshold,
        lost_track_buffer=lost_track_buffer,
        minimum_matching_threshold=minimum_matching_threshold,
        minimum_consecutive_frames=minimum_consecutive_frames,
        frame_rate=frame_rate,
    )
    first_image = common.load_image(rows[0].get("image"))
    width, height = first_image.size
    try:
        first_image.close()
    except Exception:
        pass

    output_path = Path(str(output_jsonl)).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer, render_dir = _make_render_target(
        render_output=render_output,
        width=width,
        height=height,
        frame_rate=frame_rate,
    )
    records = 0
    track_ids_seen: set[int] = set()
    frames_with_tracks: set[int] = set()

    with output_path.open("w", encoding="utf-8") as handle:
        for frame_index, (row, prediction) in enumerate(zip(rows, predictions)):
            if int(max_frames or 0) > 0 and frame_index >= int(max_frames):
                break
            image = common.load_image(row.get("image"))
            image_rgb = np.array(image.convert("RGB"))
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            pred_boxes = common.parse_answer_boxes(prediction.get("pred_boxes"))
            xyxy = _boxes_to_xyxy(pred_boxes, width=image.width, height=image.height)
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
                handle.write(
                    json.dumps(
                        {
                            "frame_index": int(frame_index),
                            "source_frame": str(prediction.get("source_image_id") or common.source_image_id_from_row(row, fallback=f"row_{frame_index:06d}")),
                            "track_id": int(tracker_id),
                            "confidence": float(confidence),
                            "prompt": str(prompt),
                            "bbox": {
                                "x_min": common.clamp(x1 / float(image.width)),
                                "y_min": common.clamp(y1 / float(image.height)),
                                "x_max": common.clamp(x2 / float(image.width)),
                                "y_max": common.clamp(y2 / float(image.height)),
                            },
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                records += 1
                track_ids_seen.add(int(tracker_id))
                frames_with_tracks.add(int(frame_index))

            if writer is not None:
                writer.write(render_frame(image_bgr, tracked_xyxy, tracked_ids))
            elif render_dir is not None:
                cv2.imwrite(str(render_dir / f"{frame_index:06d}.jpg"), render_frame(image_bgr, tracked_xyxy, tracked_ids))
            try:
                image.close()
            except Exception:
                pass

    if writer is not None:
        writer.release()
    return {
        "tracking_output_jsonl": str(output_path),
        "tracking_render_output": str(Path(render_output).expanduser().resolve()) if str(render_output or "").strip() else "",
        "tracking_records": int(records),
        "tracking_tracks": int(len(track_ids_seen)),
        "tracking_frames_with_tracks": int(len(frames_with_tracks)),
    }
