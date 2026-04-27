#!/usr/bin/env python3
"""Run ball-holder detection over a full video and render overlay videos.

This tool is intentionally split into two phases:
1. Inference writes per-frame prediction sidecars as JSONL.
2. Rendering rebuilds overlay videos from those saved predictions.

That makes the visualization reproducible without paying for inference again.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import os
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
from dotenv import load_dotenv


DEFAULT_API_BASE = "https://api.moondream.ai/v1"
DEFAULT_STAGING_API_BASE = "https://api-staging.moondream.ai/v1"
DEFAULT_BASELINE_MODEL = "moondream3-preview"
DEFAULT_OBJECT_NAME = "Player with ball in hand"
DEFAULT_VIDEO_PATH = Path(__file__).resolve().parent / "outputs" / "local_ballholder_dataset" / "Trim_NBA_gameplay.mp4"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "video_ballholder_trim_nba_gameplay"
DEFAULT_ENV_FILE = Path(__file__).resolve().parent / ".env"


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int
    duration_sec: float


@dataclass(frozen=True)
class ModelSpec:
    label: str
    model: str
    finetune_id: Optional[str]
    checkpoint_step: Optional[int]
    predictions_path: Path
    render_path: Path


def _normalize_api_base(api_base: str) -> str:
    base = (api_base or "").strip().rstrip("/")
    if base.endswith("/tuning"):
        base = base[: -len("/tuning")].rstrip("/")
    return base


def _build_auth_headers(api_key: str) -> dict[str, str]:
    header_name = os.environ.get("MOONDREAM_AUTH_HEADER", "X-Moondream-Auth")
    user_agent = os.environ.get("MOONDREAM_USER_AGENT")
    key = api_key.strip()
    if header_name.lower() == "authorization" and not key.lower().startswith("bearer "):
        key = f"Bearer {key}"
    return {
        "Content-Type": "application/json",
        "Accept": "application/json",
        header_name: key,
        "User-Agent": user_agent
        or (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        ),
    }


def _box_from_payload(item: Any) -> Optional[dict[str, float]]:
    if not isinstance(item, dict):
        return None
    try:
        x_min = max(0.0, min(1.0, float(item["x_min"])))
        y_min = max(0.0, min(1.0, float(item["y_min"])))
        x_max = max(0.0, min(1.0, float(item["x_max"])))
        y_max = max(0.0, min(1.0, float(item["y_max"])))
    except (KeyError, TypeError, ValueError):
        return None
    if x_max <= x_min or y_max <= y_min:
        return None
    return {
        "x_min": x_min,
        "y_min": y_min,
        "x_max": x_max,
        "y_max": y_max,
    }


def _extract_boxes(payload: Any) -> list[dict[str, float]]:
    if not isinstance(payload, dict):
        return []
    raw_boxes = payload.get("objects")
    if raw_boxes is None and isinstance(payload.get("output"), dict):
        raw_boxes = payload["output"].get("objects")
    boxes: list[dict[str, float]] = []
    for item in raw_boxes or []:
        box = _box_from_payload(item)
        if box is not None:
            boxes.append(box)
    return boxes


def _jpeg_bytes_to_data_url(jpeg_bytes: bytes) -> str:
    encoded = base64.b64encode(jpeg_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _call_detect_api(
    *,
    api_base: str,
    api_key: str,
    model: str,
    jpeg_bytes: bytes,
    object_name: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_objects: int,
    timeout: float,
    retries: int,
    retry_backoff_s: float,
) -> list[dict[str, float]]:
    url = api_base.rstrip("/") + "/detect"
    payload = {
        "model": model,
        "object": object_name,
        "image_url": _jpeg_bytes_to_data_url(jpeg_bytes),
        "settings": {
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": int(max_tokens),
            "max_objects": int(max_objects),
        },
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=_build_auth_headers(api_key),
        method="POST",
    )
    attempts = max(0, int(retries)) + 1
    last_error: Optional[Exception] = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8")
            parsed = json.loads(body) if body else {}
            return _extract_boxes(parsed)
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8") if exc.fp else ""
            request_id = exc.headers.get("x-request-id") if exc.headers else None
            suffix = f" (x-request-id={request_id})" if request_id else ""
            raise RuntimeError(f"HTTP {exc.code} from {url}: {(error_body or exc.reason).strip()}{suffix}") from exc
        except (TimeoutError, socket.timeout, urllib.error.URLError) as exc:
            last_error = exc
            if attempt >= attempts - 1:
                break
            delay = max(0.0, float(retry_backoff_s)) * float(attempt + 1)
            print(
                f"retry model={model} attempt={attempt + 1}/{attempts - 1} "
                f"delay_s={delay:.1f}"
            )
            time.sleep(delay)
    raise RuntimeError(f"detect request failed after retries: {last_error}") from last_error


def _probe_video(video_path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()
    duration_sec = (frame_count / fps) if fps > 0 and frame_count > 0 else 0.0
    return VideoInfo(
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration_sec=duration_sec,
    )


def _resize_frame(frame_bgr, *, max_side: int) -> Any:
    if int(max_side or 0) <= 0:
        return frame_bgr
    height, width = frame_bgr.shape[:2]
    longest = max(width, height)
    if longest <= int(max_side):
        return frame_bgr
    scale = float(max_side) / float(longest)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    return cv2.resize(frame_bgr, (resized_width, resized_height), interpolation=cv2.INTER_AREA)


def _encode_frame_jpeg(frame_bgr, *, jpeg_quality: int) -> bytes:
    ok, encoded = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok:
        raise RuntimeError("Failed to encode frame as JPEG")
    return encoded.tobytes()


def _infer_frame(
    *,
    frame_index: int,
    timestamp_sec: float,
    source_width: int,
    source_height: int,
    detect_width: int,
    detect_height: int,
    jpeg_bytes: bytes,
    api_base: str,
    api_key: str,
    model: str,
    object_name: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_objects: int,
    timeout: float,
    retries: int,
    retry_backoff_s: float,
) -> dict[str, Any]:
    started = time.monotonic()
    try:
        boxes = _call_detect_api(
            api_base=api_base,
            api_key=api_key,
            model=model,
            jpeg_bytes=jpeg_bytes,
            object_name=object_name,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            max_objects=max_objects,
            timeout=timeout,
            retries=retries,
            retry_backoff_s=retry_backoff_s,
        )
        return {
            "frame_index": int(frame_index),
            "timestamp_sec": float(timestamp_sec),
            "frame_width": int(source_width),
            "frame_height": int(source_height),
            "detect_width": int(detect_width),
            "detect_height": int(detect_height),
            "predicted_boxes": boxes,
            "latency_sec": float(time.monotonic() - started),
            "failed": False,
            "error": None,
        }
    except Exception as exc:
        return {
            "frame_index": int(frame_index),
            "timestamp_sec": float(timestamp_sec),
            "frame_width": int(source_width),
            "frame_height": int(source_height),
            "detect_width": int(detect_width),
            "detect_height": int(detect_height),
            "predicted_boxes": [],
            "latency_sec": float(time.monotonic() - started),
            "failed": True,
            "error": str(exc),
        }


def _load_prediction_index(predictions_path: Path) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    if not predictions_path.exists():
        return records
    with predictions_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
                frame_index = int(payload["frame_index"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
            records[frame_index] = payload
    return records


def _iter_video_frames(video_path: Path, *, start_frame: int, end_frame: Optional[int]):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))
    frame_index = int(start_frame)
    try:
        while True:
            if end_frame is not None and frame_index > end_frame:
                break
            ok, frame = cap.read()
            if not ok:
                break
            yield frame_index, frame
            frame_index += 1
    finally:
        cap.release()


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _resolve_best_local_candidate() -> tuple[str, Optional[str], Optional[int], Optional[Path]]:
    outputs_dir = Path(__file__).resolve().parent / "outputs"
    candidates: list[tuple[float, float, str, Optional[str], Optional[int], Path]] = []
    for path in sorted(outputs_dir.glob("staging_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        model = str(payload.get("model") or "").strip()
        if not model:
            continue
        miou = float(payload.get("eval_miou") or 0.0)
        f1 = float(payload.get("eval_f1") or 0.0)
        finetune_id = payload.get("finetune_id")
        checkpoint_step = payload.get("checkpoint_step")
        try:
            checkpoint_value = int(checkpoint_step) if checkpoint_step is not None else None
        except (TypeError, ValueError):
            checkpoint_value = None
        candidates.append((miou, f1, model, finetune_id, checkpoint_value, path))
    if not candidates:
        raise FileNotFoundError("No local candidate metrics found in _DEPICATED_MDBallHolder/outputs")
    best = max(candidates, key=lambda item: (item[0], item[1]))
    return best[2], best[3], best[4], best[5]


def _resolve_candidate_model(args: argparse.Namespace) -> tuple[str, Optional[str], Optional[int], Optional[Path]]:
    model = str(args.candidate_model or "").strip()
    finetune_id = str(args.candidate_finetune_id or "").strip() or None
    checkpoint_step_raw = args.candidate_checkpoint_step
    checkpoint_step: Optional[int]
    if checkpoint_step_raw is None or str(checkpoint_step_raw).strip() == "":
        checkpoint_step = None
    else:
        checkpoint_step = int(checkpoint_step_raw)

    if model:
        return model, finetune_id, checkpoint_step, None
    if finetune_id and checkpoint_step is not None:
        return f"{args.base_model.rstrip('/')}/{finetune_id}@{checkpoint_step}", finetune_id, checkpoint_step, None
    return _resolve_best_local_candidate()


def _build_model_specs(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    candidate_model: str,
    candidate_finetune_id: Optional[str],
    candidate_checkpoint_step: Optional[int],
) -> list[ModelSpec]:
    return [
        ModelSpec(
            label="before_baseline",
            model=str(args.baseline_model).strip(),
            finetune_id=None,
            checkpoint_step=None,
            predictions_path=output_dir / "before_baseline_predictions.jsonl",
            render_path=output_dir / "before_baseline_overlay.mp4",
        ),
        ModelSpec(
            label="after_candidate",
            model=candidate_model,
            finetune_id=candidate_finetune_id,
            checkpoint_step=candidate_checkpoint_step,
            predictions_path=output_dir / "after_candidate_predictions.jsonl",
            render_path=output_dir / "after_candidate_overlay.mp4",
        ),
    ]


def _infer_video_for_model(
    *,
    video_path: Path,
    video_info: VideoInfo,
    model_spec: ModelSpec,
    args: argparse.Namespace,
) -> dict[str, Any]:
    predictions_path = model_spec.predictions_path
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_prediction_index(predictions_path) if args.reuse_existing else {}

    start_frame = max(0, int(args.start_frame))
    end_frame = None if int(args.end_frame) < 0 else int(args.end_frame)
    if end_frame is not None and end_frame < start_frame:
        raise ValueError("--end-frame must be >= --start-frame")

    target_frames = 0
    pending: set[concurrent.futures.Future[dict[str, Any]]] = set()
    completed = 0
    submitted = 0
    failures = 0
    started_at = time.monotonic()
    in_flight_limit = max(1, int(args.workers)) * 2
    file_mode = "a" if args.reuse_existing and predictions_path.exists() else "w"

    print(
        f"infer start label={model_spec.label} model={model_spec.model} "
        f"workers={int(args.workers)} reuse_existing={bool(args.reuse_existing)}"
    )

    with predictions_path.open(file_mode, encoding="utf-8") as handle:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
            for frame_index, frame_bgr in _iter_video_frames(
                video_path,
                start_frame=start_frame,
                end_frame=end_frame,
            ):
                if frame_index in existing:
                    continue
                target_frames += 1
                frame_height, frame_width = frame_bgr.shape[:2]
                detect_frame = _resize_frame(frame_bgr, max_side=int(args.detect_max_side))
                detect_height, detect_width = detect_frame.shape[:2]
                jpeg_bytes = _encode_frame_jpeg(detect_frame, jpeg_quality=int(args.jpeg_quality))
                timestamp_sec = float(frame_index) / float(video_info.fps) if video_info.fps > 0 else 0.0
                future = executor.submit(
                    _infer_frame,
                    frame_index=frame_index,
                    timestamp_sec=timestamp_sec,
                    source_width=frame_width,
                    source_height=frame_height,
                    detect_width=detect_width,
                    detect_height=detect_height,
                    jpeg_bytes=jpeg_bytes,
                    api_base=str(args.api_base),
                    api_key=str(args.api_key),
                    model=model_spec.model,
                    object_name=str(args.object_name),
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                    max_tokens=int(args.max_tokens),
                    max_objects=int(args.max_objects),
                    timeout=float(args.timeout),
                    retries=int(args.request_retries),
                    retry_backoff_s=float(args.request_retry_backoff_s),
                )
                pending.add(future)
                submitted += 1

                if len(pending) < in_flight_limit:
                    continue
                done, pending = concurrent.futures.wait(
                    pending,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for item in done:
                    record = item.result()
                    record.update(
                        {
                            "model_label": model_spec.label,
                            "model": model_spec.model,
                            "finetune_id": model_spec.finetune_id,
                            "checkpoint_step": model_spec.checkpoint_step,
                            "object_name": str(args.object_name),
                            "source_video": str(video_path),
                        }
                    )
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                    completed += 1
                    if record.get("failed"):
                        failures += 1
                    if completed % max(1, int(args.progress_every)) == 0:
                        elapsed = max(0.001, time.monotonic() - started_at)
                        print(
                            f"infer progress label={model_spec.label} completed={completed} "
                            f"submitted={submitted} fps={completed / elapsed:.2f} failures={failures}"
                        )

            while pending:
                done, pending = concurrent.futures.wait(
                    pending,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for item in done:
                    record = item.result()
                    record.update(
                        {
                            "model_label": model_spec.label,
                            "model": model_spec.model,
                            "finetune_id": model_spec.finetune_id,
                            "checkpoint_step": model_spec.checkpoint_step,
                            "object_name": str(args.object_name),
                            "source_video": str(video_path),
                        }
                    )
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                    completed += 1
                    if record.get("failed"):
                        failures += 1
                    if completed % max(1, int(args.progress_every)) == 0:
                        elapsed = max(0.001, time.monotonic() - started_at)
                        print(
                            f"infer progress label={model_spec.label} completed={completed} "
                            f"submitted={submitted} fps={completed / elapsed:.2f} failures={failures}"
                        )

    all_records = _load_prediction_index(predictions_path)
    first_frame = start_frame
    last_frame = end_frame if end_frame is not None else max(0, video_info.frame_count - 1)
    expected_frame_count = max(0, last_frame - first_frame + 1)
    available = sum(1 for idx in range(first_frame, last_frame + 1) if idx in all_records)
    print(
        f"infer done label={model_spec.label} model={model_spec.model} "
        f"available={available}/{expected_frame_count} new_completed={completed} failures={failures}"
    )
    return {
        "label": model_spec.label,
        "model": model_spec.model,
        "finetune_id": model_spec.finetune_id,
        "checkpoint_step": model_spec.checkpoint_step,
        "predictions_path": str(predictions_path),
        "frames_expected": int(expected_frame_count),
        "frames_available": int(available),
        "frames_reused": int(max(0, available - completed)),
        "frames_inferred": int(completed),
        "failed_frames": int(sum(1 for idx in range(first_frame, last_frame + 1) if all_records.get(idx, {}).get("failed"))),
        "elapsed_sec": float(time.monotonic() - started_at),
    }


def _draw_boxes(frame_bgr, boxes: list[dict[str, Any]], *, label: str) -> Any:
    rendered = frame_bgr.copy()
    height, width = rendered.shape[:2]
    line_width = max(2, int(round(max(width, height) * 0.0025)))
    font_scale = max(0.5, min(1.1, max(width, height) / 1400.0))
    for box in boxes:
        try:
            x1 = int(round(float(box["x_min"]) * width))
            y1 = int(round(float(box["y_min"]) * height))
            x2 = int(round(float(box["x_max"]) * width))
            y2 = int(round(float(box["y_max"]) * height))
        except (KeyError, TypeError, ValueError):
            continue
        cv2.rectangle(rendered, (x1, y1), (x2, y2), (0, 220, 0), line_width)
        cv2.putText(
            rendered,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 220, 0),
            max(1, line_width - 1),
            cv2.LINE_AA,
        )
    return rendered


def _render_video_from_predictions(
    *,
    video_path: Path,
    video_info: VideoInfo,
    predictions_path: Path,
    output_path: Path,
    start_frame: int,
    end_frame: Optional[int],
) -> dict[str, Any]:
    predictions = _load_prediction_index(predictions_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(video_info.fps or 30.0),
        (int(video_info.width), int(video_info.height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open output video for writing: {output_path}")

    rendered_frames = 0
    frames_with_boxes = 0
    missing_predictions = 0
    started_at = time.monotonic()
    try:
        for frame_index, frame_bgr in _iter_video_frames(video_path, start_frame=start_frame, end_frame=end_frame):
            record = predictions.get(frame_index)
            if record is None:
                missing_predictions += 1
                writer.write(frame_bgr)
                rendered_frames += 1
                continue
            boxes = list(record.get("predicted_boxes") or [])
            if boxes:
                frames_with_boxes += 1
                frame_bgr = _draw_boxes(frame_bgr, boxes, label="BALL HOLDER")
            writer.write(frame_bgr)
            rendered_frames += 1
    finally:
        writer.release()

    print(
        f"render done path={output_path} frames={rendered_frames} "
        f"frames_with_boxes={frames_with_boxes} missing_predictions={missing_predictions}"
    )
    return {
        "output_path": str(output_path),
        "rendered_frames": int(rendered_frames),
        "frames_with_boxes": int(frames_with_boxes),
        "missing_predictions": int(missing_predictions),
        "elapsed_sec": float(time.monotonic() - started_at),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ball-holder inference over a full video and render overlays.")
    parser.add_argument("--video", default=str(DEFAULT_VIDEO_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE))
    parser.add_argument("--api-key", default="")
    parser.add_argument("--api-base", default="")
    parser.add_argument("--object-name", default=DEFAULT_OBJECT_NAME)
    parser.add_argument("--baseline-model", default=DEFAULT_BASELINE_MODEL)
    parser.add_argument("--base-model", default=DEFAULT_BASELINE_MODEL)
    parser.add_argument("--candidate-model", default="")
    parser.add_argument("--candidate-finetune-id", default="")
    parser.add_argument("--candidate-checkpoint-step", default="")
    parser.add_argument("--infer-only", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--reuse-existing", dest="reuse_existing", action="store_true", default=True)
    parser.add_argument("--no-reuse-existing", dest="reuse_existing", action="store_false")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--detect-max-side", type=int, default=0)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-objects", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--request-retries", type=int, default=2)
    parser.add_argument("--request-retry-backoff-s", type=float, default=5.0)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.infer_only and args.render_only:
        raise ValueError("Use at most one of --infer-only and --render-only")

    candidate_model, candidate_finetune_id, candidate_checkpoint_step, candidate_source_path = _resolve_candidate_model(args)

    load_dotenv(args.env_file, override=False)
    if not str(args.api_key).strip():
        args.api_key = os.environ.get("MOONDREAM_API_KEY", "")
    if not str(args.api_key).strip():
        raise ValueError("MOONDREAM_API_KEY is required")

    if not str(args.api_base).strip():
        args.api_base = (
            os.environ.get("TUNA_BASE_URL")
            or os.environ.get("MOONDREAM_API_BASE")
            or (
                DEFAULT_STAGING_API_BASE
                if candidate_source_path is not None and candidate_source_path.name.startswith("staging_")
                else ""
            )
            or DEFAULT_API_BASE
        )
    args.api_base = _normalize_api_base(str(args.api_base))
    if not args.api_base:
        raise ValueError("--api-base must not be empty")

    video_path = Path(str(args.video)).expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    output_dir = Path(str(args.output_dir)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    video_info = _probe_video(video_path)
    model_specs = _build_model_specs(
        args=args,
        output_dir=output_dir,
        candidate_model=candidate_model,
        candidate_finetune_id=candidate_finetune_id,
        candidate_checkpoint_step=candidate_checkpoint_step,
    )
    manifest_path = output_dir / "manifest.json"

    manifest: dict[str, Any] = {
        "video": {
            "path": str(video_path),
            "width": int(video_info.width),
            "height": int(video_info.height),
            "fps": float(video_info.fps),
            "frame_count": int(video_info.frame_count),
            "duration_sec": float(video_info.duration_sec),
        },
        "config": {
            "api_base": str(args.api_base),
            "object_name": str(args.object_name),
            "workers": int(args.workers),
            "detect_max_side": int(args.detect_max_side),
            "jpeg_quality": int(args.jpeg_quality),
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "max_tokens": int(args.max_tokens),
            "max_objects": int(args.max_objects),
            "timeout": float(args.timeout),
            "request_retries": int(args.request_retries),
            "request_retry_backoff_s": float(args.request_retry_backoff_s),
            "reuse_existing": bool(args.reuse_existing),
            "start_frame": int(args.start_frame),
            "end_frame": None if int(args.end_frame) < 0 else int(args.end_frame),
        },
        "models": [
            {
                "label": spec.label,
                "model": spec.model,
                "finetune_id": spec.finetune_id,
                "checkpoint_step": spec.checkpoint_step,
                "source_metrics_path": str(candidate_source_path) if spec.label == "after_candidate" and candidate_source_path is not None else None,
                "predictions_path": str(spec.predictions_path),
                "render_path": str(spec.render_path),
            }
            for spec in model_specs
        ],
        "runs": {},
    }

    if not args.render_only:
        for spec in model_specs:
            manifest["runs"][spec.label] = {
                "inference": _infer_video_for_model(
                    video_path=video_path,
                    video_info=video_info,
                    model_spec=spec,
                    args=args,
                )
            }
            _write_manifest(manifest_path, manifest)

    if not args.infer_only:
        start_frame = max(0, int(args.start_frame))
        end_frame = None if int(args.end_frame) < 0 else int(args.end_frame)
        for spec in model_specs:
            if not spec.predictions_path.exists():
                raise FileNotFoundError(
                    f"Predictions file missing for render: {spec.predictions_path}. "
                    "Run inference first or omit --render-only."
                )
            model_run = manifest["runs"].setdefault(spec.label, {})
            model_run["render"] = _render_video_from_predictions(
                video_path=video_path,
                video_info=video_info,
                predictions_path=spec.predictions_path,
                output_path=spec.render_path,
                start_frame=start_frame,
                end_frame=end_frame,
            )
            _write_manifest(manifest_path, manifest)

    _write_manifest(manifest_path, manifest)
    print(f"manifest written -> {manifest_path}")


if __name__ == "__main__":
    main()
