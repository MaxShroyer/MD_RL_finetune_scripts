#!/usr/bin/env python3
"""Rerun task-packet fineune benchmarks and export graphics-friendly records."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import random
import socket
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from PIL import Image

import task_packet_benchmark_common as task_packet_common


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_ENV_FILE = ".env.staging"
DEFAULT_API_BASE = "https://api.moondream.ai/v1"
DEFAULT_BASE_MODEL = "moondream3-preview"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _resolve_env_file_path(raw_env_file: str, *, repo_root: Path = REPO_ROOT) -> str:
    raw = str(raw_env_file or "").strip()
    candidate_paths: list[Path] = []
    repo_root = repo_root.resolve()

    def add_candidate(raw_candidate: str) -> None:
        candidate = Path(str(raw_candidate or "")).expanduser()
        if not candidate.is_absolute():
            candidate = repo_root / candidate
        candidate_paths.append(candidate.resolve())

    if raw:
        add_candidate(raw)
    else:
        add_candidate(DEFAULT_ENV_FILE)
    seen: set[str] = set()
    for resolved in candidate_paths:
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        if resolved.exists():
            return str(resolved)
    fallback = Path(raw if raw else DEFAULT_ENV_FILE).expanduser()
    if not fallback.is_absolute():
        fallback = repo_root / fallback
    return str(fallback.resolve())


def _resolve_api_key(explicit_api_key: str, api_key_env_var: str) -> str:
    cli_key = str(explicit_api_key or "").strip()
    if cli_key:
        return cli_key

    preferred_env_var = str(api_key_env_var or "").strip()
    if preferred_env_var:
        preferred_value = os.environ.get(preferred_env_var, "").strip()
        if preferred_value:
            return preferred_value

    fallback = os.environ.get("MOONDREAM_API_KEY", "").strip()
    if fallback:
        return fallback
    raise ValueError(
        f"Moondream API key is required (checked --api-key, {preferred_env_var or 'MOONDREAM_API_KEY'}, and MOONDREAM_API_KEY)"
    )


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerun fineune benchmarks for task packet exports.")
    parser.add_argument("--env-file", default=DEFAULT_ENV_FILE)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--api-key-env-var", default="MOONDREAM_API_KEY")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--tasks", nargs="*", default=["state_farm", "player_with_ball", "aerial"])
    parser.add_argument("--baseline-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens-detect", type=int, default=256)
    parser.add_argument("--max-tokens-point", type=int, default=256)
    parser.add_argument("--max-objects", type=int, default=50)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retry-429-max-retries", type=int, default=6)
    parser.add_argument("--retry-429-backoff-s", type=float, default=0.5)
    parser.add_argument("--retry-429-max-backoff-s", type=float, default=12.0)
    parser.add_argument("--retry-timeout-max-retries", type=int, default=3)
    parser.add_argument("--retry-timeout-backoff-s", type=float, default=2.0)
    parser.add_argument("--retry-timeout-max-backoff-s", type=float, default=20.0)
    parser.add_argument("--run-stamp", default="")
    args = parser.parse_args(argv)
    args.env_file = _resolve_env_file_path(str(args.env_file), repo_root=REPO_ROOT)
    args.tasks = [str(item).strip() for item in args.tasks if str(item).strip()]
    return args


def _build_auth_headers(api_key: str) -> dict[str, str]:
    header_name = os.environ.get("MOONDREAM_AUTH_HEADER", "X-Moondream-Auth")
    user_agent = os.environ.get("MOONDREAM_USER_AGENT") or (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
    key = api_key.strip()
    if header_name.lower() == "authorization" and not key.lower().startswith("bearer "):
        key = f"Bearer {key}"
    return {
        "Content-Type": "application/json",
        "Accept": "application/json",
        header_name: key,
        "User-Agent": user_agent,
    }


def _to_data_url(image: Image.Image, *, format: str = "JPEG", quality: int = 90) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=format, quality=max(1, min(100, int(quality))))
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    mime = "image/jpeg" if format.upper() == "JPEG" else f"image/{format.lower()}"
    return f"data:{mime};base64,{encoded}"


def _normalize_coord(value: Any, *, size: int) -> float:
    coord = float(value)
    if abs(coord) > 1.5 and size > 0:
        coord = coord / float(size)
    return max(0.0, min(1.0, coord))


def _post_with_fallback_payloads(
    *,
    api_base: str,
    api_key: str,
    endpoint: str,
    payloads: list[dict[str, Any]],
    timeout: float,
    retry_429_max_retries: int,
    retry_429_backoff_s: float,
    retry_429_max_backoff_s: float,
    retry_timeout_max_retries: int,
    retry_timeout_backoff_s: float,
    retry_timeout_max_backoff_s: float,
) -> tuple[dict[str, Any], float]:
    first_request_error: Optional[Exception] = None
    for payload in payloads:
        attempts = 0
        request_started = time.monotonic()
        while True:
            req = urllib.request.Request(
                api_base.rstrip("/") + endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers=_build_auth_headers(api_key),
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    body = resp.read().decode("utf-8")
                data = json.loads(body) if body else {}
                if not isinstance(data, dict):
                    data = {}
                return data, (time.monotonic() - request_started) * 1000.0
            except urllib.error.HTTPError as exc:
                detail = ""
                retry_after_s = 0.0
                try:
                    raw = exc.read().decode("utf-8", errors="replace")
                    if raw:
                        detail = raw.strip().replace("\n", " ")
                except Exception:
                    detail = ""
                if exc.code == 429:
                    retry_after_header = (exc.headers.get("Retry-After") or "").strip()
                    if retry_after_header:
                        try:
                            retry_after_s = max(0.0, float(retry_after_header))
                        except (TypeError, ValueError):
                            retry_after_s = 0.0
                if detail:
                    max_len = 300
                    if len(detail) > max_len:
                        detail = detail[:max_len] + "..."
                    exc.msg = f"{exc.msg}: {detail}"
                if exc.code == 429 and attempts < max(0, int(retry_429_max_retries)):
                    backoff_s = max(0.0, float(retry_429_backoff_s)) * (2.0**attempts)
                    capped_backoff_s = min(max(0.0, float(retry_429_max_backoff_s)), backoff_s)
                    sleep_s = max(retry_after_s, capped_backoff_s) * random.uniform(0.9, 1.1)
                    if sleep_s > 0.0:
                        time.sleep(sleep_s)
                    attempts += 1
                    continue
                if first_request_error is None:
                    first_request_error = exc
                break
            except (TimeoutError, socket.timeout, urllib.error.URLError) as exc:
                reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
                detail = str(reason or exc).strip()
                lower_detail = detail.lower()
                is_timeout = isinstance(reason, (TimeoutError, socket.timeout)) or "timed out" in lower_detail
                if not is_timeout:
                    raise
                if attempts < max(0, int(retry_timeout_max_retries)):
                    backoff_s = max(0.0, float(retry_timeout_backoff_s)) * (2.0**attempts)
                    capped_backoff_s = min(max(0.0, float(retry_timeout_max_backoff_s)), backoff_s)
                    sleep_s = capped_backoff_s * random.uniform(0.9, 1.1)
                    if sleep_s > 0.0:
                        time.sleep(sleep_s)
                    attempts += 1
                    continue
                if first_request_error is None:
                    first_request_error = TimeoutError(detail or "The read operation timed out")
                break
    if first_request_error is not None:
        raise first_request_error
    raise RuntimeError(f"Moondream request failed without a successful payload for endpoint {endpoint}")


def _extract_detect_boxes(payload: dict[str, Any], *, width: int, height: int) -> list[task_packet_common.Box]:
    objects = payload.get("objects")
    if objects is None and isinstance(payload.get("output"), dict):
        objects = payload["output"].get("objects")
    boxes: list[task_packet_common.Box] = []
    for item in objects or []:
        if not isinstance(item, dict):
            continue
        try:
            box = task_packet_common.Box(
                x_min=_normalize_coord(item["x_min"], size=width),
                y_min=_normalize_coord(item["y_min"], size=height),
                x_max=_normalize_coord(item["x_max"], size=width),
                y_max=_normalize_coord(item["y_max"], size=height),
            )
        except (KeyError, TypeError, ValueError):
            continue
        if box.x_max > box.x_min and box.y_max > box.y_min:
            boxes.append(box)
    return boxes


def _extract_points(payload: dict[str, Any], *, width: int, height: int) -> list[task_packet_common.Point]:
    points_raw: Any = payload.get("points")
    if points_raw is None and isinstance(payload.get("output"), dict):
        output_payload = payload["output"]
        points_raw = output_payload.get("points")
        if points_raw is None and ("x" in output_payload and "y" in output_payload):
            points_raw = [output_payload]
    if points_raw is None and ("x" in payload and "y" in payload):
        points_raw = [payload]
    if isinstance(points_raw, dict):
        points_raw = [points_raw]
    if not isinstance(points_raw, list):
        return []

    points: list[task_packet_common.Point] = []
    for item in points_raw:
        if not isinstance(item, dict):
            continue
        raw_x = item.get("x")
        raw_y = item.get("y")
        if raw_x is None or raw_y is None:
            continue
        try:
            points.append(
                task_packet_common.Point(
                    x=_normalize_coord(raw_x, size=width),
                    y=_normalize_coord(raw_y, size=height),
                )
            )
        except (TypeError, ValueError):
            continue
    return points


def call_detect_api_raw(
    *,
    api_base: str,
    api_key: str,
    model: str,
    image: Image.Image,
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_objects: int,
    timeout: float,
    retry_429_max_retries: int,
    retry_429_backoff_s: float,
    retry_429_max_backoff_s: float,
    retry_timeout_max_retries: int,
    retry_timeout_backoff_s: float,
    retry_timeout_max_backoff_s: float,
) -> tuple[list[task_packet_common.Box], dict[str, Any], float]:
    image_url = _to_data_url(image, quality=90)
    settings = {
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": int(max_tokens),
        "max_objects": int(max_objects),
    }
    payloads = [
        {
            "model": model,
            "skill": "detect",
            "image_url": image_url,
            "object": prompt,
            "settings": settings,
        },
        {
            "model": model,
            "image_url": image_url,
            "object": prompt,
            "settings": settings,
        },
    ]
    raw_response, latency_ms = _post_with_fallback_payloads(
        api_base=api_base,
        api_key=api_key,
        endpoint="/detect",
        payloads=payloads,
        timeout=timeout,
        retry_429_max_retries=retry_429_max_retries,
        retry_429_backoff_s=retry_429_backoff_s,
        retry_429_max_backoff_s=retry_429_max_backoff_s,
        retry_timeout_max_retries=retry_timeout_max_retries,
        retry_timeout_backoff_s=retry_timeout_backoff_s,
        retry_timeout_max_backoff_s=retry_timeout_max_backoff_s,
    )
    return _extract_detect_boxes(raw_response, width=image.width, height=image.height), raw_response, latency_ms


def call_point_api_raw(
    *,
    api_base: str,
    api_key: str,
    model: str,
    image: Image.Image,
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout: float,
    retry_429_max_retries: int,
    retry_429_backoff_s: float,
    retry_429_max_backoff_s: float,
    retry_timeout_max_retries: int,
    retry_timeout_backoff_s: float,
    retry_timeout_max_backoff_s: float,
) -> tuple[list[task_packet_common.Point], dict[str, Any], float]:
    image_url = _to_data_url(image, quality=90)
    settings = {
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": int(max_tokens),
    }
    payloads = [
        {
            "model": model,
            "skill": "point",
            "image_url": image_url,
            "object": prompt,
            "settings": settings,
        },
        {
            "model": model,
            "image_url": image_url,
            "object": prompt,
            "settings": settings,
        },
    ]
    raw_response, latency_ms = _post_with_fallback_payloads(
        api_base=api_base,
        api_key=api_key,
        endpoint="/point",
        payloads=payloads,
        timeout=timeout,
        retry_429_max_retries=retry_429_max_retries,
        retry_429_backoff_s=retry_429_backoff_s,
        retry_429_max_backoff_s=retry_429_max_backoff_s,
        retry_timeout_max_retries=retry_timeout_max_retries,
        retry_timeout_backoff_s=retry_timeout_backoff_s,
        retry_timeout_max_backoff_s=retry_timeout_max_backoff_s,
    )
    return _extract_points(raw_response, width=image.width, height=image.height), raw_response, latency_ms


def _packet_image_abspath(packet_image_path: Optional[str]) -> Optional[str]:
    if not packet_image_path:
        return None
    path = Path(packet_image_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return str(path.resolve())


def evaluate_packet_sample(
    *,
    sample: task_packet_common.Sample,
    spec: task_packet_common.TaskSpec,
    run_label: str,
    model: str,
    finetune_id: Optional[str],
    checkpoint_step: Optional[int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    try:
        if spec.skill == "point":
            pred_points, raw_response, latency_ms = call_point_api_raw(
                api_base=str(args.api_base),
                api_key=str(args.api_key),
                model=model,
                image=sample.image,
                prompt=sample.prompt,
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                max_tokens=int(args.max_tokens_point),
                timeout=float(args.timeout),
                retry_429_max_retries=int(args.retry_429_max_retries),
                retry_429_backoff_s=float(args.retry_429_backoff_s),
                retry_429_max_backoff_s=float(args.retry_429_max_backoff_s),
                retry_timeout_max_retries=int(args.retry_timeout_max_retries),
                retry_timeout_backoff_s=float(args.retry_timeout_backoff_s),
                retry_timeout_max_backoff_s=float(args.retry_timeout_max_backoff_s),
            )
            pred_boxes: list[task_packet_common.Box] = []
            task_f1 = task_packet_common.reward_f1_points(pred_points, sample.ground_truth_boxes)
            task_miou = 0.0
            tp, fp, fn = task_packet_common.count_tp_fp_fn_points(pred_points, sample.ground_truth_boxes)
        else:
            pred_boxes, raw_response, latency_ms = call_detect_api_raw(
                api_base=str(args.api_base),
                api_key=str(args.api_key),
                model=model,
                image=sample.image,
                prompt=sample.prompt,
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                max_tokens=int(args.max_tokens_detect),
                max_objects=int(args.max_objects),
                timeout=float(args.timeout),
                retry_429_max_retries=int(args.retry_429_max_retries),
                retry_429_backoff_s=float(args.retry_429_backoff_s),
                retry_429_max_backoff_s=float(args.retry_429_max_backoff_s),
                retry_timeout_max_retries=int(args.retry_timeout_max_retries),
                retry_timeout_backoff_s=float(args.retry_timeout_backoff_s),
                retry_timeout_max_backoff_s=float(args.retry_timeout_max_backoff_s),
            )
            pred_points = []
            task_f1 = task_packet_common.reward_f1(
                pred_boxes,
                sample.ground_truth_boxes,
                iou_threshold=float(spec.iou_threshold),
            )
            task_miou = task_packet_common.reward_miou(pred_boxes, sample.ground_truth_boxes)
            tp, fp, fn = task_packet_common.count_tp_fp_fn(
                pred_boxes,
                sample.ground_truth_boxes,
                iou_threshold=float(spec.iou_threshold),
            )
    except Exception as exc:
        return {
            "task": spec.name,
            "skill": spec.skill,
            "run_label": run_label,
            "model": model,
            "finetune_id": finetune_id,
            "checkpoint_step": checkpoint_step,
            "sample_index": int(sample.sample_index),
            "sample_id": sample.sample_id,
            "prompt": sample.prompt,
            "source_image_path": sample.source_image_path,
            "packet_image_path": sample.packet_image_path,
            "packet_image_abspath": _packet_image_abspath(sample.packet_image_path),
            "ground_truth_boxes": task_packet_common.serialize_boxes(sample.ground_truth_boxes),
            "pred_boxes": [],
            "pred_points": [],
            "task_f1": None,
            "task_miou": None,
            "tp": None,
            "fp": None,
            "fn": None,
            "latency_ms": None,
            "failed": True,
            "error": str(exc),
            "raw_response": None,
            "task_type": sample.task_type,
            "notes": sample.notes,
            "timestamp": sample.timestamp,
        }

    return {
        "task": spec.name,
        "skill": spec.skill,
        "run_label": run_label,
        "model": model,
        "finetune_id": finetune_id,
        "checkpoint_step": checkpoint_step,
        "sample_index": int(sample.sample_index),
        "sample_id": sample.sample_id,
        "prompt": sample.prompt,
        "source_image_path": sample.source_image_path,
        "packet_image_path": sample.packet_image_path,
        "packet_image_abspath": _packet_image_abspath(sample.packet_image_path),
        "ground_truth_boxes": task_packet_common.serialize_boxes(sample.ground_truth_boxes),
        "pred_boxes": task_packet_common.serialize_boxes(pred_boxes),
        "pred_points": task_packet_common.serialize_points(pred_points),
        "task_f1": float(task_f1),
        "task_miou": float(task_miou),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "latency_ms": float(latency_ms),
        "failed": False,
        "error": None,
        "raw_response": raw_response,
        "task_type": sample.task_type,
        "notes": sample.notes,
        "timestamp": sample.timestamp,
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")


def _build_task_output_dir(spec: task_packet_common.TaskSpec, run_stamp: str) -> Path:
    return spec.task_dir / "reruns" / run_stamp


def run_task(
    spec: task_packet_common.TaskSpec,
    args: argparse.Namespace,
    *,
    run_stamp: str,
) -> dict[str, Any]:
    samples = task_packet_common.load_task_samples(spec, repo_root=REPO_ROOT)
    output_dir = _build_task_output_dir(spec, run_stamp)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_model = str(args.baseline_model).strip() or DEFAULT_BASE_MODEL
    checkpoint_base_model = str(args.base_model).strip() or DEFAULT_BASE_MODEL
    if not str(spec.finetune_id).strip() or spec.checkpoint_step is None:
        raise ValueError(f"Task {spec.name} is missing finetune checkpoint defaults")
    checkpoint_model = f"{checkpoint_base_model.rstrip('/')}/{spec.finetune_id}@{int(spec.checkpoint_step)}"

    baseline_records: list[dict[str, Any]] = []
    checkpoint_records: list[dict[str, Any]] = []
    for index, sample in enumerate(samples, start=1):
        baseline_records.append(
            evaluate_packet_sample(
                sample=sample,
                spec=spec,
                run_label="baseline",
                model=baseline_model,
                finetune_id=None,
                checkpoint_step=None,
                args=args,
            )
        )
        checkpoint_records.append(
            evaluate_packet_sample(
                sample=sample,
                spec=spec,
                run_label="checkpoint",
                model=checkpoint_model,
                finetune_id=spec.finetune_id,
                checkpoint_step=int(spec.checkpoint_step),
                args=args,
            )
        )
        print(f"{spec.name}: processed {index}/{len(samples)} sample_id={sample.sample_id}", flush=True)

    baseline_metrics = {
        "task": spec.name,
        "skill": spec.skill,
        "run_label": "baseline",
        "model": baseline_model,
        "finetune_id": None,
        "checkpoint_step": None,
        **task_packet_common.aggregate_prediction_metrics(
            baseline_records,
            skill=spec.skill,
            iou_threshold=float(spec.iou_threshold),
        ),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    checkpoint_metrics = {
        "task": spec.name,
        "skill": spec.skill,
        "run_label": "checkpoint",
        "model": checkpoint_model,
        "finetune_id": spec.finetune_id,
        "checkpoint_step": int(spec.checkpoint_step),
        **task_packet_common.aggregate_prediction_metrics(
            checkpoint_records,
            skill=spec.skill,
            iou_threshold=float(spec.iou_threshold),
        ),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }

    baseline_records_path = output_dir / "baseline.records.jsonl"
    checkpoint_records_path = output_dir / "checkpoint.records.jsonl"
    baseline_metrics_path = output_dir / "baseline.metrics.json"
    checkpoint_metrics_path = output_dir / "checkpoint.metrics.json"
    manifest_path = output_dir / "manifest.json"

    _write_jsonl(baseline_records_path, baseline_records)
    _write_jsonl(checkpoint_records_path, checkpoint_records)
    _write_json(baseline_metrics_path, baseline_metrics)
    _write_json(checkpoint_metrics_path, checkpoint_metrics)

    manifest = {
        "task": spec.name,
        "skill": spec.skill,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "task_dir": task_packet_common._repo_relative(spec.task_dir, repo_root=REPO_ROOT),
        "output_dir": task_packet_common._repo_relative(output_dir, repo_root=REPO_ROOT),
        "sample_count": len(samples),
        "prompt_fallback": spec.prompt,
        "api_base": str(args.api_base),
        "baseline": {
            "model": baseline_model,
            "finetune_id": None,
            "checkpoint_step": None,
            "records_jsonl": task_packet_common._repo_relative(baseline_records_path, repo_root=REPO_ROOT),
            "metrics_json": task_packet_common._repo_relative(baseline_metrics_path, repo_root=REPO_ROOT),
        },
        "checkpoint": {
            "model": checkpoint_model,
            "finetune_id": spec.finetune_id,
            "checkpoint_step": int(spec.checkpoint_step),
            "records_jsonl": task_packet_common._repo_relative(checkpoint_records_path, repo_root=REPO_ROOT),
            "metrics_json": task_packet_common._repo_relative(checkpoint_metrics_path, repo_root=REPO_ROOT),
        },
    }
    _write_json(manifest_path, manifest)
    task_packet_common.refresh_simple_task_samples(
        task_dir=spec.task_dir,
        skill=spec.skill,
        repo_root=REPO_ROOT,
        baseline_records=baseline_records,
        checkpoint_records=checkpoint_records,
    )
    return manifest


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    if str(args.env_file).strip():
        load_dotenv(args.env_file, override=False)
    args.api_key = _resolve_api_key(str(args.api_key), str(args.api_key_env_var))

    registry = task_packet_common.build_task_registry(REPO_ROOT)
    unknown_tasks = [task for task in args.tasks if task not in registry]
    if unknown_tasks:
        raise ValueError(f"Unknown task(s): {unknown_tasks}. Available: {sorted(registry)}")

    run_stamp = str(args.run_stamp).strip() or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    manifests: dict[str, Any] = {}
    for task_name in args.tasks:
        manifests[task_name] = run_task(registry[task_name], args, run_stamp=run_stamp)
    print(json.dumps({"run_stamp": run_stamp, "tasks": manifests}, indent=2))


if __name__ == "__main__":
    main()
