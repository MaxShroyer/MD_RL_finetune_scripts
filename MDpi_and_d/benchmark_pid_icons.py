#!/usr/bin/env python3
"""Benchmark class-conditional PI&D symbol detection.

Evaluation mode:
- For each sample, run one detect prompt per class present in GT.
- Add random negative prompts for absent classes and for empty images.
- Report micro/macro F1, mIoU, latency, and per-class tp/fp/fn.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from dotenv import load_dotenv
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment


def _repo_relative(*parts: str) -> Path:
    return Path(__file__).resolve().parent.joinpath(*parts)


def _to_data_url(image: Image.Image, *, quality: int = 90) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=max(1, min(100, int(quality))))
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


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


def _call_detect_api(
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
) -> list["Box"]:
    raw_prompt = str(prompt).strip()
    detect_object = raw_prompt
    lower = raw_prompt.lower()
    if lower.startswith("detect the "):
        detect_object = raw_prompt[len("detect the ") :].strip()
    if len(detect_object) >= 2 and detect_object[0] == detect_object[-1] and detect_object[0] in {"'", '"'}:
        candidate = detect_object[1:-1].strip()
        if candidate:
            detect_object = candidate

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
            "object": detect_object,
            "settings": settings,
        },
        {
            "model": model,
            "image_url": image_url,
            "object": detect_object,
            "settings": settings,
        },
    ]

    data: dict[str, Any] = {}
    first_http_error: Optional[urllib.error.HTTPError] = None
    for payload in payloads:
        attempts = 0
        while True:
            req = urllib.request.Request(
                api_base.rstrip("/") + "/detect",
                data=json.dumps(payload).encode("utf-8"),
                headers=_build_auth_headers(api_key),
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    body = resp.read().decode("utf-8")
                data = json.loads(body) if body else {}
                first_http_error = None
                break
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
                if first_http_error is None:
                    first_http_error = exc
                break
        if first_http_error is None:
            break
    if first_http_error is not None:
        raise first_http_error

    objects = data.get("objects")
    if objects is None and isinstance(data.get("output"), dict):
        objects = data["output"].get("objects")

    boxes: list[Box] = []
    for item in objects or []:
        if not isinstance(item, dict):
            continue
        try:
            boxes.append(
                Box(
                    x_min=max(0.0, min(1.0, float(item["x_min"]))),
                    y_min=max(0.0, min(1.0, float(item["y_min"]))),
                    x_max=max(0.0, min(1.0, float(item["x_max"]))),
                    y_max=max(0.0, min(1.0, float(item["y_max"]))),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return [b for b in boxes if b.x_max > b.x_min and b.y_max > b.y_min]


@dataclass(frozen=True)
class Box:
    x_min: float
    y_min: float
    x_max: float
    y_max: float


@dataclass(frozen=True)
class ClassBox:
    class_uid: str
    class_name: str
    box: Box


@dataclass(frozen=True)
class BaseSample:
    image: Image.Image
    boxes: list[ClassBox]
    sample_id: str


@dataclass(frozen=True)
class TaskSample:
    image: Image.Image
    prompt: str
    gt_boxes: list[Box]
    class_name: str
    sample_id: str


def _load_class_names(class_names_file: str, dataset_path: Optional[str]) -> list[str]:
    if class_names_file:
        payload = json.loads(Path(class_names_file).expanduser().read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "class_catalog" in payload:
            names = [str(item.get("class_name", "")).strip() for item in payload["class_catalog"]]
            return sorted({name for name in names if name})
        if isinstance(payload, list):
            names = [str(item).strip() for item in payload]
            return sorted({name for name in names if name})

    if dataset_path:
        meta = Path(dataset_path).expanduser().resolve() / "metadata.json"
        if meta.exists():
            payload = json.loads(meta.read_text(encoding="utf-8"))
            catalog = payload.get("class_catalog") or []
            names = [str(item.get("class_name", "")).strip() for item in catalog if isinstance(item, dict)]
            names = [name for name in names if name]
            if names:
                return sorted(set(names))

    return []


def _parse_answer_boxes(value: Any, width: int, height: int) -> list[ClassBox]:
    if value is None:
        return []
    raw = value
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        try:
            raw = json.loads(text)
        except json.JSONDecodeError:
            return []

    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return []

    parsed: list[ClassBox] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        class_name = str(item.get("class_name") or item.get("source_class_name") or "").strip()
        class_uid = str(item.get("class_uid") or class_name).strip()
        if not class_name:
            continue
        try:
            x_min = float(item.get("x_min"))
            y_min = float(item.get("y_min"))
            x_max = float(item.get("x_max"))
            y_max = float(item.get("y_max"))
        except (TypeError, ValueError):
            continue

        if max(abs(x_min), abs(y_min), abs(x_max), abs(y_max)) > 1.5:
            if width <= 0 or height <= 0:
                continue
            x_min /= width
            y_min /= height
            x_max /= width
            y_max /= height

        x_min = max(0.0, min(1.0, x_min))
        y_min = max(0.0, min(1.0, y_min))
        x_max = max(0.0, min(1.0, x_max))
        y_max = max(0.0, min(1.0, y_max))
        if x_max <= x_min or y_max <= y_min:
            continue

        parsed.append(ClassBox(class_uid=class_uid, class_name=class_name, box=Box(x_min, y_min, x_max, y_max)))

    return parsed


def _to_base_sample(row: dict, fallback_id: int) -> Optional[BaseSample]:
    image = row.get("image")
    if image is None:
        return None
    image = image.convert("RGB")
    width, height = image.size
    boxes = _parse_answer_boxes(row.get("answer_boxes"), width=width, height=height)
    sample_id = str(row.get("source_image_id") or row.get("id") or fallback_id)
    return BaseSample(image=image, boxes=boxes, sample_id=sample_id)


def _prompt_for_class(class_name: str) -> str:
    # This string is sent as the /detect API "object". Keep it aligned with training/inference usage.
    return f"{class_name} icon or icons"


def _tasks_from_sample(
    sample: BaseSample,
    *,
    all_class_names: list[str],
    rng: random.Random,
    neg_prompts_per_empty: int,
    neg_prompts_per_nonempty: int,
) -> list[TaskSample]:
    grouped: dict[str, tuple[str, list[Box]]] = {}
    for item in sample.boxes:
        if item.class_uid not in grouped:
            grouped[item.class_uid] = (item.class_name, [item.box])
        else:
            grouped[item.class_uid][1].append(item.box)

    tasks: list[TaskSample] = []
    present_names = {entry[0] for entry in grouped.values()}

    for _, (class_name, boxes) in grouped.items():
        tasks.append(
            TaskSample(
                image=sample.image,
                prompt=_prompt_for_class(class_name),
                gt_boxes=list(boxes),
                class_name=class_name,
                sample_id=sample.sample_id,
            )
        )

    if grouped:
        absent = [name for name in all_class_names if name not in present_names]
        if absent and neg_prompts_per_nonempty > 0:
            picks = rng.sample(absent, k=min(neg_prompts_per_nonempty, len(absent)))
            for class_name in picks:
                tasks.append(
                    TaskSample(
                        image=sample.image,
                        prompt=_prompt_for_class(class_name),
                        gt_boxes=[],
                        class_name=class_name,
                        sample_id=sample.sample_id,
                    )
                )
    else:
        if all_class_names:
            k = max(1, neg_prompts_per_empty)
            picks = rng.sample(all_class_names, k=min(k, len(all_class_names)))
            for class_name in picks:
                tasks.append(
                    TaskSample(
                        image=sample.image,
                        prompt=_prompt_for_class(class_name),
                        gt_boxes=[],
                        class_name=class_name,
                        sample_id=sample.sample_id,
                    )
                )

    return tasks


def _iter_rows(dataset_path: str, dataset_name: str, split: str, token: Optional[str], max_samples: Optional[int]) -> Iterable[dict]:
    count = 0
    if dataset_path:
        dataset_obj = load_from_disk(dataset_path)
        if isinstance(dataset_obj, DatasetDict):
            if split not in dataset_obj:
                available = ", ".join(dataset_obj.keys())
                raise ValueError(f"Split '{split}' not found. Available: {available}")
            ds: Dataset = dataset_obj[split]
        else:
            ds = dataset_obj
        for row in ds:
            yield row
            count += 1
            if max_samples is not None and count >= max_samples:
                break
        return

    ds = load_dataset(dataset_name, split=split, token=token, streaming=True)
    for row in ds:
        yield row
        count += 1
        if max_samples is not None and count >= max_samples:
            break


def _safe_slug(value: str, max_len: int = 64) -> str:
    text = "".join(ch if ch.isalnum() else "_" for ch in value.lower()).strip("_")
    while "__" in text:
        text = text.replace("__", "_")
    if not text:
        text = "item"
    return text[:max_len]


def _draw_box(drawer: ImageDraw.ImageDraw, box: Box, width: int, height: int, color: str, line_width: int) -> None:
    x0 = int(round(box.x_min * width))
    y0 = int(round(box.y_min * height))
    x1 = int(round(box.x_max * width))
    y1 = int(round(box.y_max * height))
    drawer.rectangle((x0, y0, x1, y1), outline=color, width=line_width)


def _save_task_visualization(
    *,
    out_dir: Path,
    label: str,
    sample_idx: int,
    task: TaskSample,
    pred_boxes: list[Box],
    f1: float,
    miou: float,
    tp: int,
    fp: int,
    fn: int,
) -> Optional[str]:
    image = task.image.copy().convert("RGB")
    width, height = image.size
    if width <= 0 or height <= 0:
        return None
    drawer = ImageDraw.Draw(image)
    line_width = max(2, int(round(min(width, height) * 0.004)))

    for gt_box in task.gt_boxes:
        _draw_box(drawer, gt_box, width, height, color="#2ECC71", line_width=line_width)
    for pred_box in pred_boxes:
        _draw_box(drawer, pred_box, width, height, color="#E74C3C", line_width=line_width)

    caption = (
        f"{label} | sample={task.sample_id} | class={task.class_name} "
        f"| gt={len(task.gt_boxes)} pred={len(pred_boxes)} tp={tp} fp={fp} fn={fn} "
        f"| f1={f1:.3f} miou={miou:.3f}"
    )
    text_height = max(20, int(round(height * 0.06)))
    overlay = Image.new("RGB", (width, text_height), color=(0, 0, 0))
    image.paste(overlay, (0, 0))
    drawer = ImageDraw.Draw(image)
    drawer.text((8, max(2, text_height // 4)), caption, fill=(255, 255, 255))

    label_dir = out_dir / _safe_slug(label)
    label_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{sample_idx:04d}_{_safe_slug(task.sample_id, 48)}_{_safe_slug(task.class_name, 48)}.jpg"
    path = label_dir / filename
    image.save(path, format="JPEG", quality=92)
    return str(path)


def _box_iou(a: Box, b: Box) -> float:
    inter_x_min = max(a.x_min, b.x_min)
    inter_y_min = max(a.y_min, b.y_min)
    inter_x_max = min(a.x_max, b.x_max)
    inter_y_max = min(a.y_max, b.y_max)
    inter_w = max(0.0, inter_x_max - inter_x_min)
    inter_h = max(0.0, inter_y_max - inter_y_min)
    inter = inter_w * inter_h
    if inter == 0.0:
        return 0.0
    area_a = max(0.0, a.x_max - a.x_min) * max(0.0, a.y_max - a.y_min)
    area_b = max(0.0, b.x_max - b.x_min) * max(0.0, b.y_max - b.y_min)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _match_ious(predicted: list[Box], ground_truth: list[Box]) -> np.ndarray:
    n_gt = len(ground_truth)
    n_pred = len(predicted)
    if n_gt == 0 or n_pred == 0:
        return np.array([], dtype=np.float32)
    size = max(n_gt, n_pred)
    iou_matrix = np.zeros((size, size), dtype=np.float32)
    for i, gt in enumerate(ground_truth):
        for j, pred in enumerate(predicted):
            iou_matrix[i, j] = _box_iou(pred, gt)
    cost = 1.0 - iou_matrix
    row_idx, col_idx = linear_sum_assignment(cost)
    return iou_matrix[row_idx, col_idx]


def _reward_miou(predicted: list[Box], ground_truth: list[Box]) -> float:
    if not predicted and not ground_truth:
        return 1.0
    if not predicted or not ground_truth:
        return 0.0
    matches = _match_ious(predicted, ground_truth)
    denom = max(len(predicted), len(ground_truth))
    return float(matches.sum()) / float(denom) if denom else 0.0


def _reward_f1(predicted: list[Box], ground_truth: list[Box]) -> float:
    if not predicted and not ground_truth:
        return 1.0
    if not predicted or not ground_truth:
        return 0.0
    matches = _match_ious(predicted, ground_truth)
    true_pos = float((matches >= 0.5).sum())
    precision = true_pos / len(predicted) if predicted else 0.0
    recall = true_pos / len(ground_truth) if ground_truth else 0.0
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _count_tp_fp_fn(predicted: list[Box], ground_truth: list[Box], iou_threshold: float) -> tuple[int, int, int]:
    n_pred = len(predicted)
    n_gt = len(ground_truth)
    if n_pred == 0 and n_gt == 0:
        return 0, 0, 0
    if n_pred == 0:
        return 0, 0, n_gt
    if n_gt == 0:
        return 0, n_pred, 0
    matches = _match_ious(predicted, ground_truth)
    true_pos = int((matches >= iou_threshold).sum())
    false_pos = n_pred - true_pos
    false_neg = n_gt - true_pos
    return true_pos, false_pos, false_neg


def _evaluate_model(label: str, model: str, args: argparse.Namespace, all_class_names: list[str]) -> dict[str, Any]:
    rng = random.Random(args.seed)
    last_request_end: Optional[float] = None
    viz_saved = 0
    viz_paths: list[str] = []
    viz_out_dir = Path(args.viz_dir).expanduser().resolve() if args.viz_dir else None

    total_base_samples = 0
    total_tasks = 0
    total_f1 = 0.0
    total_miou = 0.0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_latency = 0.0
    failed_tasks = 0

    per_class: dict[str, dict[str, Any]] = {}

    def _class_stats(name: str) -> dict[str, Any]:
        if name not in per_class:
            per_class[name] = {"tasks": 0, "tp": 0, "fp": 0, "fn": 0, "f1_sum": 0.0, "miou_sum": 0.0}
        return per_class[name]

    row_iter = _iter_rows(
        dataset_path=args.dataset_path.strip(),
        dataset_name=args.dataset_name,
        split=args.split,
        token=args.hf_token,
        max_samples=args.max_samples,
    )

    for idx, row in enumerate(row_iter):
        sample = _to_base_sample(row, idx)
        if sample is None:
            continue
        total_base_samples += 1

        tasks = _tasks_from_sample(
            sample,
            all_class_names=all_class_names,
            rng=rng,
            neg_prompts_per_empty=args.neg_prompts_per_empty,
            neg_prompts_per_nonempty=args.neg_prompts_per_nonempty,
        )

        for task in tasks:
            if args.min_request_interval_s > 0.0 and last_request_end is not None:
                wait_s = float(args.min_request_interval_s) - (time.monotonic() - last_request_end)
                if wait_s > 0.0:
                    time.sleep(wait_s)
            start = time.monotonic()
            try:
                pred_boxes = _call_detect_api(
                    api_base=args.api_base,
                    api_key=args.api_key,
                    model=model,
                    image=task.image,
                    prompt=task.prompt,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens,
                    max_objects=args.max_objects,
                    timeout=args.timeout,
                    retry_429_max_retries=args.retry_429_max_retries,
                    retry_429_backoff_s=args.retry_429_backoff_s,
                    retry_429_max_backoff_s=args.retry_429_max_backoff_s,
                )
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
                print(f"{label}: detect failed for sample={task.sample_id} class={task.class_name}: {exc}")
                failed_tasks += 1
                last_request_end = time.monotonic()
                continue
            latency = time.monotonic() - start
            last_request_end = time.monotonic()

            f1 = _reward_f1(pred_boxes, task.gt_boxes)
            miou = _reward_miou(pred_boxes, task.gt_boxes)
            tp, fp, fn = _count_tp_fp_fn(pred_boxes, task.gt_boxes, iou_threshold=args.iou_threshold)

            total_tasks += 1
            total_f1 += f1
            total_miou += miou
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_latency += latency

            stats = _class_stats(task.class_name)
            stats["tasks"] += 1
            stats["tp"] += tp
            stats["fp"] += fp
            stats["fn"] += fn
            stats["f1_sum"] += f1
            stats["miou_sum"] += miou

            if args.viz_samples > 0 and viz_saved < args.viz_samples and viz_out_dir is not None:
                out_path = _save_task_visualization(
                    out_dir=viz_out_dir,
                    label=label,
                    sample_idx=viz_saved,
                    task=task,
                    pred_boxes=pred_boxes,
                    f1=f1,
                    miou=miou,
                    tp=tp,
                    fp=fp,
                    fn=fn,
                )
                if out_path:
                    viz_paths.append(out_path)
                    viz_saved += 1

            if args.progress_every > 0 and total_tasks > 0 and total_tasks % args.progress_every == 0:
                print(f"{label}: progress tasks={total_tasks}")

    if total_tasks == 0:
        return {
            "label": label,
            "model": model,
            "error": "No tasks evaluated",
            "base_samples": total_base_samples,
            "tasks": 0,
            "failed_tasks": failed_tasks,
            "attempted_tasks": failed_tasks,
            "visualizations_saved": viz_saved,
            "visualization_paths": viz_paths,
        }

    micro_denom = 2 * total_tp + total_fp + total_fn
    micro_f1 = 1.0 if micro_denom == 0 else (2 * total_tp) / micro_denom

    per_class_payload: dict[str, Any] = {}
    for class_name, stats in sorted(per_class.items()):
        tasks = int(stats["tasks"])
        denom = 2 * int(stats["tp"]) + int(stats["fp"]) + int(stats["fn"])
        micro = 1.0 if denom == 0 else (2 * int(stats["tp"])) / denom
        per_class_payload[class_name] = {
            "tasks": tasks,
            "tp": int(stats["tp"]),
            "fp": int(stats["fp"]),
            "fn": int(stats["fn"]),
            "f1_micro": micro,
            "f1_macro": (float(stats["f1_sum"]) / tasks) if tasks else 0.0,
            "miou": (float(stats["miou_sum"]) / tasks) if tasks else 0.0,
        }

    return {
        "label": label,
        "model": model,
        "dataset_name": args.dataset_name or None,
        "dataset_path": args.dataset_path.strip() or None,
        "split": args.split,
        "base_samples": total_base_samples,
        "tasks": total_tasks,
        "failed_tasks": failed_tasks,
        "attempted_tasks": total_tasks + failed_tasks,
        "eval_f1": micro_f1,
        "eval_f1_macro": total_f1 / total_tasks,
        "eval_miou": total_miou / total_tasks,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "avg_latency_sec": total_latency / total_tasks,
        "visualizations_saved": viz_saved,
        "visualization_paths": viz_paths,
        "per_class": per_class_payload,
    }


def _print_summary(title: str, metrics: dict[str, Any]) -> None:
    if "error" in metrics:
        print(f"{title}: error={metrics['error']}")
        return
    print(
        f"{title}: base_samples={metrics.get('base_samples', 0)} tasks={metrics['tasks']} "
        f"miou={metrics['eval_miou']:.4f} "
        f"f1={metrics['eval_f1']:.4f} macro_f1={metrics['eval_f1_macro']:.4f} "
        f"tp={metrics['tp']} fp={metrics['fp']} fn={metrics['fn']} "
        f"failed={metrics.get('failed_tasks', 0)} latency={metrics['avg_latency_sec']:.3f}s"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark PI&D class-conditional detect performance.")
    parser.add_argument("--env-file", default=str(_repo_relative(".env")))
    parser.add_argument("--api-key", default=os.environ.get("MOONDREAM_API_KEY"))
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"))
    parser.add_argument("--api-base", default="https://api.moondream.ai/v1")

    parser.add_argument("--dataset-path", default=str(_repo_relative("outputs", "pid_icons_merged")))
    parser.add_argument("--dataset-name", default="")
    parser.add_argument("--split", default="post_val")
    parser.add_argument("--class-names-file", default="")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max base samples to read from the selected eval split (applies before class-conditional task expansion).",
    )
    parser.add_argument(
        "--max-post-train-eval-samples",
        type=int,
        default=None,
        help="Alias for --max-samples, intended for post-train eval benchmarking.",
    )
    parser.add_argument(
        "--viz-samples",
        type=int,
        default=0,
        help="Number of evaluated tasks to visualize per model label (0 disables).",
    )
    parser.add_argument(
        "--viz-dir",
        default=str(_repo_relative("outputs", "benchmark_viz")),
        help="Directory where benchmark visualizations are written.",
    )

    parser.add_argument("--model", default="")
    parser.add_argument("--finetune-id", default="")
    parser.add_argument("--checkpoint-step", type=int, default=None)
    parser.add_argument("--base-model", default="moondream3-preview")

    parser.add_argument("--baseline-model", default="moondream3-preview")
    parser.add_argument("--skip-baseline", action="store_true")

    parser.add_argument("--neg-prompts-per-empty", type=int, default=2)
    parser.add_argument("--neg-prompts-per-nonempty", type=int, default=1)

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-objects", type=int, default=50)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retry-429-max-retries", type=int, default=6)
    parser.add_argument("--retry-429-backoff-s", type=float, default=0.5)
    parser.add_argument("--retry-429-max-backoff-s", type=float, default=12.0)
    parser.add_argument("--min-request-interval-s", type=float, default=0.0)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--out-json", default=str(_repo_relative("outputs", "benchmark_pid_icons.json")))
    args = parser.parse_args()

    load_dotenv(args.env_file, override=False)
    if not args.api_key:
        args.api_key = os.environ.get("MOONDREAM_API_KEY")
    if not args.hf_token:
        args.hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    if not args.api_key:
        raise ValueError("MOONDREAM_API_KEY is required")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be > 0 when provided")
    if args.max_post_train_eval_samples is not None and args.max_post_train_eval_samples <= 0:
        raise ValueError("--max-post-train-eval-samples must be > 0 when provided")
    if (
        args.max_samples is not None
        and args.max_post_train_eval_samples is not None
        and args.max_samples != args.max_post_train_eval_samples
    ):
        raise ValueError("--max-samples and --max-post-train-eval-samples disagree; set only one or match values.")
    if args.max_post_train_eval_samples is not None:
        args.max_samples = args.max_post_train_eval_samples
    if args.viz_samples < 0:
        raise ValueError("--viz-samples must be >= 0")

    all_class_names = _load_class_names(args.class_names_file, args.dataset_path.strip() or None)
    if not all_class_names:
        raise ValueError("Could not resolve class names. Provide --class-names-file or use local dataset metadata.")

    candidate_model = args.model.strip()
    if not candidate_model and args.finetune_id.strip() and args.checkpoint_step is not None:
        candidate_model = f"{args.base_model.rstrip('/')}/{args.finetune_id.strip()}@{int(args.checkpoint_step)}"

    if args.skip_baseline and not candidate_model:
        raise ValueError("Provide --model or (--finetune-id with --checkpoint-step) when using --skip-baseline")

    baseline_metrics = None
    if not args.skip_baseline:
        baseline_metrics = _evaluate_model("baseline", args.baseline_model, args, all_class_names)
        _print_summary("baseline", baseline_metrics)

    candidate_metrics = None
    if candidate_model:
        candidate_metrics = _evaluate_model("candidate", candidate_model, args, all_class_names)
        _print_summary("candidate", candidate_metrics)

    if baseline_metrics is not None and candidate_metrics is not None:
        payload: dict[str, Any] = {
            "baseline": baseline_metrics,
            "candidate": candidate_metrics,
            "config": {
                "baseline_model": args.baseline_model,
                "candidate_model": candidate_model,
                "dataset_path": args.dataset_path.strip() or None,
                "dataset_name": args.dataset_name or None,
                "split": args.split,
                "max_samples": args.max_samples,
                "viz_samples": args.viz_samples,
                "viz_dir": str(Path(args.viz_dir).expanduser().resolve()) if args.viz_dir else None,
                "seed": args.seed,
            },
        }
    elif baseline_metrics is not None:
        payload = baseline_metrics
    else:
        payload = candidate_metrics or {"error": "No evaluation was run."}

    out_path = Path(args.out_json).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved metrics -> {out_path}")


if __name__ == "__main__":
    main()
