#!/usr/bin/env python3
"""Rank Inspector point/detect runs and build a before/after report package.

This script uses repository-local run metadata to shortlist candidate finetunes,
then benchmarks a small, stratified validation slice on the same tasks for:
- baseline `moondream3-preview`
- shortlisted checkpoint candidates

It writes:
- `report.json`
- `report.md`
- per-task JSONL records
- baseline/candidate overlay visualizations for the most improved sample in each class
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from datasets import load_from_disk
from dotenv import load_dotenv
from PIL import Image

from finetune_checkpoints import list_saved_checkpoint_steps
from inspector_md import common

try:
    from MDpi_and_d import benchmark_pid_icons as bm
except ModuleNotFoundError:
    from _DEPICATED_MDpi_and_d import benchmark_pid_icons as bm


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT_ROOT = common.repo_relative("outputs")
DEFAULT_ENV_FILE = common.repo_relative(".env.staging")
DEFAULT_CLASS_NAMES_FILE = ""
DEFAULT_BASE_URL = common.DEFAULT_BASE_URL
DEFAULT_BASE_MODEL = common.DEFAULT_BASE_MODEL
DEFAULT_API_KEY_ENV_VARS = list(common.DEFAULT_API_KEY_ENV_VARS)

BASELINE_MODEL = DEFAULT_BASE_MODEL

POINT_CONFIG = {
    "dataset_path": common.repo_relative("outputs", "inspector_point_v1"),
    "split": "validation",
    "skill": "point",
    "max_tokens": 256,
    "max_objects": 32,
    "selection_metric": "eval_f1",
}

DETECT_CONFIG = {
    "dataset_path": common.repo_relative("outputs", "inspector_detect_v1"),
    "split": "validation",
    "skill": "detect",
    "max_tokens": 512,
    "max_objects": 48,
    "selection_metric": "eval_miou",
}


@dataclass(frozen=True)
class RunEvidence:
    task: str
    run_name: str
    finetune_id: str
    source_log: str
    logged_baseline_f1: float
    logged_baseline_miou: float
    logged_positive_f1: float


@dataclass(frozen=True)
class CheckpointSpec:
    task: str
    finetune_id: str
    checkpoint_step: int
    source_run_name: str
    source_log: str


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE))
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key-env-vars", nargs="+", default=list(DEFAULT_API_KEY_ENV_VARS))
    parser.add_argument("--class-names-file", default=str(DEFAULT_CLASS_NAMES_FILE))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-class-target", type=int, default=2)
    parser.add_argument("--max-base-rows", type=int, default=10)
    parser.add_argument("--detect-top-runs", type=int, default=2)
    parser.add_argument("--detect-checkpoints-per-run", type=int, default=3)
    parser.add_argument("--point-top-runs", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--output-dir", default="")
    return parser.parse_args(argv)


def _load_runtime_api_keys(env_file: Path, env_vars: list[str]) -> list[str]:
    load_dotenv(env_file, override=True)
    keys: list[str] = []
    for env_var in env_vars:
        value = os.environ.get(str(env_var).strip(), "").strip()
        if value:
            keys.append(value)
    if not keys:
        raise ValueError(f"no API keys found in {env_file} for env vars: {env_vars}")
    return keys


def _load_row_image(row: dict[str, Any]) -> Optional[Image.Image]:
    image = row.get("image")
    if image is None:
        return None
    if hasattr(image, "convert"):
        return image.convert("RGB")
    if isinstance(image, dict):
        if image.get("bytes"):
            with Image.open(io.BytesIO(image["bytes"])) as opened:
                return opened.convert("RGB")
        if image.get("path"):
            with Image.open(str(image["path"])) as opened:
                return opened.convert("RGB")
        return None
    if isinstance(image, str):
        with Image.open(image) as opened:
            return opened.convert("RGB")
    return None


def _sample_from_row(row: dict[str, Any], fallback_id: int) -> Optional[bm.BaseSample]:
    image = _load_row_image(row)
    if image is None:
        return None
    width, height = image.size
    boxes = bm._parse_answer_boxes(row.get("answer_boxes"), width=width, height=height)
    sample_id = str(row.get("source_image_id") or row.get("id") or fallback_id)
    return bm.BaseSample(image=image, boxes=boxes, sample_id=sample_id)


def _row_class_names(row: dict[str, Any]) -> set[str]:
    raw = row.get("answer_boxes")
    try:
        payload = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError:
        payload = []
    if not isinstance(payload, list):
        payload = [payload] if isinstance(payload, dict) else []

    classes: set[str] = set()
    for item in payload:
        if not isinstance(item, dict):
            continue
        for key in ("class_name", "source_class_name"):
            value = str(item.get(key) or "").strip()
            if value:
                classes.add(value)
        attrs = item.get("attributes")
        if isinstance(attrs, list):
            for attr in attrs:
                if not isinstance(attr, dict):
                    continue
                value = str(attr.get("class_name") or "").strip()
                if value:
                    classes.add(value)
    return classes


def _select_eval_rows(
    *,
    dataset_path: Path,
    split: str,
    class_names: list[str],
    per_class_target: int,
    max_base_rows: int,
    seed: int,
) -> list[dict[str, Any]]:
    ds = load_from_disk(str(dataset_path))[split]
    indexed_rows = []
    for idx, row in enumerate(ds):
        classes = _row_class_names(row)
        indexed_rows.append({"index": idx, "row": row, "classes": classes})

    rng = random.Random(seed)
    rng.shuffle(indexed_rows)
    remaining_need = {name: int(per_class_target) for name in class_names}
    selected: list[dict[str, Any]] = []
    available = list(indexed_rows)

    while available and len(selected) < max(1, int(max_base_rows)):
        best_index = -1
        best_gain = -1
        best_cardinality = -1
        for idx, info in enumerate(available):
            gain = sum(1 for name in info["classes"] if remaining_need.get(name, 0) > 0)
            cardinality = len(info["classes"])
            if gain > best_gain or (gain == best_gain and cardinality > best_cardinality):
                best_index = idx
                best_gain = gain
                best_cardinality = cardinality
        if best_index < 0:
            break
        picked = available.pop(best_index)
        if best_gain <= 0 and all(value <= 0 for value in remaining_need.values()):
            break
        selected.append(picked)
        for name in picked["classes"]:
            if remaining_need.get(name, 0) > 0:
                remaining_need[name] -= 1
        if all(value <= 0 for value in remaining_need.values()) and len(selected) >= min(max_base_rows, len(class_names)):
            break

    if len(selected) < max(1, int(max_base_rows)):
        selected_indices = {item["index"] for item in selected}
        for info in indexed_rows:
            if info["index"] in selected_indices:
                continue
            selected.append(info)
            if len(selected) >= max(1, int(max_base_rows)):
                break

    return [item["row"] for item in sorted(selected, key=lambda item: int(item["index"]))]


def _discover_run_evidence(task: str) -> list[RunEvidence]:
    if task == "detect":
        patterns = [
            common.repo_relative("outputs", "inspector_sweep_logs").glob("detect_*.log"),
            common.repo_relative("outputs", "inspector_detect_only_sweep_logs").glob("detect_*.log"),
        ]
    else:
        patterns = [
            common.repo_relative("outputs", "inspector_sweep_logs").glob("point_*.log"),
            common.repo_relative("outputs", "inspector_point_only_sweep_logs").glob("point_*.log"),
        ]

    finetune_re = re.compile(r"resolved_finetune_id=([A-Z0-9]+)")
    baseline_re = re.compile(
        r"baseline eval step \d+ tasks=\d+ pos_tasks=\d+ neg_tasks=\d+ "
        r"miou=([0-9.]+) f1=([0-9.]+) macro_f1=([0-9.]+) pos_f1=([0-9.]+)"
    )

    best_by_finetune: dict[str, RunEvidence] = {}
    for group in patterns:
        for path in group:
            text = path.read_text(encoding="utf-8", errors="replace")
            finetune_match = finetune_re.search(text)
            baseline_match = baseline_re.search(text)
            if finetune_match is None or baseline_match is None:
                continue
            finetune_id = finetune_match.group(1)
            logged_miou = float(baseline_match.group(1))
            logged_f1 = float(baseline_match.group(2))
            logged_pos_f1 = float(baseline_match.group(4))
            evidence = RunEvidence(
                task=task,
                run_name=path.stem,
                finetune_id=finetune_id,
                source_log=str(path),
                logged_baseline_f1=logged_f1,
                logged_baseline_miou=logged_miou,
                logged_positive_f1=logged_pos_f1,
            )
            current = best_by_finetune.get(finetune_id)
            if current is None:
                best_by_finetune[finetune_id] = evidence
                continue
            current_score = (current.logged_positive_f1, current.logged_baseline_f1, current.logged_baseline_miou)
            candidate_score = (evidence.logged_positive_f1, evidence.logged_baseline_f1, evidence.logged_baseline_miou)
            if candidate_score > current_score:
                best_by_finetune[finetune_id] = evidence

    if task == "detect":
        return sorted(
            best_by_finetune.values(),
            key=lambda item: (item.logged_positive_f1, item.logged_baseline_miou, item.logged_baseline_f1, item.run_name),
            reverse=True,
        )
    return sorted(
        best_by_finetune.values(),
        key=lambda item: (item.logged_positive_f1, item.logged_baseline_f1, item.logged_baseline_miou, item.run_name),
        reverse=True,
    )


def _pick_checkpoint_subset(steps: list[int], limit: int) -> list[int]:
    unique = sorted({int(step) for step in steps})
    if len(unique) <= limit:
        return unique
    if limit <= 1:
        return [unique[-1]]
    indices = {0, len(unique) - 1}
    while len(indices) < limit:
        target_fraction = len(indices) / float(limit - 1)
        target_index = int(round(target_fraction * (len(unique) - 1)))
        indices.add(max(0, min(len(unique) - 1, target_index)))
        if len(indices) >= limit:
            break
        midpoint = len(unique) // 2
        indices.add(midpoint)
        if len(indices) >= limit:
            break
        for idx in range(len(unique)):
            if idx not in indices:
                indices.add(idx)
                if len(indices) >= limit:
                    break
    return [unique[idx] for idx in sorted(indices)]


def _build_checkpoint_specs(
    *,
    task: str,
    evidence: list[RunEvidence],
    api_base: str,
    api_key: str,
    top_runs: int,
    detect_checkpoints_per_run: int,
) -> list[CheckpointSpec]:
    out: list[CheckpointSpec] = []
    for item in evidence[: max(1, int(top_runs))]:
        saved_steps = list_saved_checkpoint_steps(api_base=api_base, api_key=api_key, finetune_id=item.finetune_id)
        if task == "detect":
            selected_steps = _pick_checkpoint_subset(saved_steps, max(1, int(detect_checkpoints_per_run)))
        else:
            selected_steps = saved_steps
        for step in selected_steps:
            out.append(
                CheckpointSpec(
                    task=task,
                    finetune_id=item.finetune_id,
                    checkpoint_step=int(step),
                    source_run_name=item.run_name,
                    source_log=item.source_log,
                )
            )
    return out


def _serialize_boxes(boxes: list[Any]) -> list[dict[str, float]]:
    return [
        {
            "x_min": float(box.x_min),
            "y_min": float(box.y_min),
            "x_max": float(box.x_max),
            "y_max": float(box.y_max),
        }
        for box in boxes
    ]


def _serialize_points(points: list[Any]) -> list[dict[str, float]]:
    return [{"x": float(point.x), "y": float(point.y)} for point in points]


def _model_id_for_checkpoint(base_model: str, finetune_id: str, checkpoint_step: int) -> str:
    return f"{base_model}/{finetune_id}@{int(checkpoint_step)}"


def _evaluate_tasks(
    *,
    task_name: str,
    model_label: str,
    model_id: str,
    tasks: list[bm.TaskSample],
    api_base: str,
    api_keys: list[str],
    max_tokens: int,
    max_objects: int,
    timeout: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    effective_skill = task_name
    total_f1 = 0.0
    total_miou = 0.0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    positive_tasks = 0
    negative_tasks = 0
    failed_tasks = 0
    per_class: dict[str, dict[str, float]] = defaultdict(lambda: {"tasks": 0.0, "f1_sum": 0.0, "miou_sum": 0.0, "tp": 0.0, "fp": 0.0, "fn": 0.0})
    records: list[dict[str, Any]] = []

    for index, task in enumerate(tasks):
        api_key = api_keys[index % len(api_keys)]
        pred_boxes: list[Any] = []
        pred_points: list[Any] = []
        error: Optional[str] = None
        latency = None
        started = time.monotonic()
        try:
            if effective_skill == "point":
                pred_points = bm._call_point_api(
                    api_base=api_base,
                    api_key=api_key,
                    model=model_id,
                    image=task.image,
                    prompt=task.prompt,
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=int(max_tokens),
                    timeout=float(timeout),
                    retry_429_max_retries=6,
                    retry_429_backoff_s=0.5,
                    retry_429_max_backoff_s=12.0,
                    retry_timeout_max_retries=3,
                    retry_timeout_backoff_s=2.0,
                    retry_timeout_max_backoff_s=20.0,
                    reasoning=False,
                )
            else:
                pred_boxes = bm._call_detect_api(
                    api_base=api_base,
                    api_key=api_key,
                    model=model_id,
                    image=task.image,
                    prompt=task.prompt,
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=int(max_tokens),
                    max_objects=int(max_objects),
                    timeout=float(timeout),
                    retry_429_max_retries=6,
                    retry_429_backoff_s=0.5,
                    retry_429_max_backoff_s=12.0,
                    retry_timeout_max_retries=3,
                    retry_timeout_backoff_s=2.0,
                    retry_timeout_max_backoff_s=20.0,
                    reasoning=False,
                )
            latency = time.monotonic() - started
        except Exception as exc:  # pragma: no cover - network/runtime
            error = f"{type(exc).__name__}: {exc}"
            failed_tasks += 1

        if effective_skill == "point":
            f1 = bm._reward_f1_points(pred_points, task.gt_boxes) if error is None else 0.0
            miou = 0.0
            tp, fp, fn = bm._count_tp_fp_fn_points(pred_points, task.gt_boxes) if error is None else (0, 0, len(task.gt_boxes))
            pred_count = len(pred_points)
        else:
            f1 = bm._reward_f1(pred_boxes, task.gt_boxes) if error is None else 0.0
            miou = bm._reward_miou(pred_boxes, task.gt_boxes) if error is None else 0.0
            tp, fp, fn = bm._count_tp_fp_fn(pred_boxes, task.gt_boxes, iou_threshold=0.5) if error is None else (0, 0, len(task.gt_boxes))
            pred_count = len(pred_boxes)

        total_f1 += float(f1)
        total_miou += float(miou)
        total_tp += int(tp)
        total_fp += int(fp)
        total_fn += int(fn)
        if task.gt_boxes:
            positive_tasks += 1
        else:
            negative_tasks += 1

        class_bucket = per_class[task.class_name]
        class_bucket["tasks"] += 1.0
        class_bucket["f1_sum"] += float(f1)
        class_bucket["miou_sum"] += float(miou)
        class_bucket["tp"] += float(tp)
        class_bucket["fp"] += float(fp)
        class_bucket["fn"] += float(fn)

        records.append(
            {
                "model_label": model_label,
                "skill": effective_skill,
                "model_id": model_id,
                "task_key": f"{task.sample_id}::{task.class_name}::{task.prompt}",
                "sample_id": task.sample_id,
                "class_name": task.class_name,
                "prompt": task.prompt,
                "is_positive": bool(task.gt_boxes),
                "gt_count": len(task.gt_boxes),
                "pred_count": int(pred_count),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "task_f1": float(f1),
                "task_miou": float(miou),
                "latency_sec": latency,
                "failed": bool(error),
                "error": error,
                "gt_boxes": _serialize_boxes(task.gt_boxes),
                "pred_boxes": _serialize_boxes(pred_boxes),
                "pred_points": _serialize_points(pred_points),
                "_task": task,
            }
        )

    micro_denom = (2 * total_tp) + total_fp + total_fn
    eval_f1 = 1.0 if micro_denom == 0 else (2.0 * float(total_tp)) / float(micro_denom)
    task_count = len(tasks)
    metrics = {
        "skill": effective_skill,
        "model_label": model_label,
        "model_id": model_id,
        "tasks": task_count,
        "failed_tasks": failed_tasks,
        "positive_tasks": positive_tasks,
        "negative_tasks": negative_tasks,
        "eval_f1": float(eval_f1),
        "eval_f1_macro": float(total_f1 / task_count) if task_count else 0.0,
        "eval_miou": float(total_miou / task_count) if task_count else 0.0,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "per_class": {},
    }
    for class_name, bucket in sorted(per_class.items()):
        tasks_for_class = max(1.0, float(bucket["tasks"]))
        denom = (2.0 * float(bucket["tp"])) + float(bucket["fp"]) + float(bucket["fn"])
        micro = 1.0 if denom == 0.0 else (2.0 * float(bucket["tp"])) / denom
        metrics["per_class"][class_name] = {
            "tasks": int(bucket["tasks"]),
            "tp": int(bucket["tp"]),
            "fp": int(bucket["fp"]),
            "fn": int(bucket["fn"]),
            "f1_micro": float(micro),
            "f1_macro": float(bucket["f1_sum"] / tasks_for_class),
            "miou": float(bucket["miou_sum"] / tasks_for_class),
        }
    return metrics, records


def _checkpoint_score(metrics: dict[str, Any], task: str) -> tuple[float, float]:
    if task == "detect":
        return (float(metrics.get("eval_miou", 0.0)), float(metrics.get("eval_f1", 0.0)))
    return (float(metrics.get("eval_f1", 0.0)), float(metrics.get("eval_f1_macro", 0.0)))


def _best_candidate(
    *,
    task: str,
    rows: list[dict[str, Any]],
    class_names: list[str],
    checkpoint_specs: list[CheckpointSpec],
    api_base: str,
    api_keys: list[str],
    timeout: float,
) -> dict[str, Any]:
    config = POINT_CONFIG if task == "point" else DETECT_CONFIG
    tasks: list[bm.TaskSample] = []
    for index, row in enumerate(rows):
        sample = _sample_from_row(row, fallback_id=index)
        if sample is None:
            continue
        tasks.extend(
            bm._tasks_from_sample(
                sample,
                all_class_names=class_names,
                rng=random.Random(42),
                neg_prompts_per_empty=1,
                neg_prompts_per_nonempty=0,
                prompt_style="class_name",
                prompt_overrides={},
            )
        )

    print(f"[{task}] selected base_rows={len(rows)} expanded_tasks={len(tasks)}")
    baseline_metrics, baseline_records = _evaluate_tasks(
        task_name=task,
        model_label="baseline",
        model_id=BASELINE_MODEL,
        tasks=tasks,
        api_base=api_base,
        api_keys=api_keys,
        max_tokens=int(config["max_tokens"]),
        max_objects=int(config["max_objects"]),
        timeout=float(timeout),
    )
    print(
        f"[{task}] baseline eval_f1={baseline_metrics['eval_f1']:.4f} "
        f"eval_miou={baseline_metrics['eval_miou']:.4f} failed={baseline_metrics['failed_tasks']}"
    )

    candidate_results: list[dict[str, Any]] = []
    for spec in checkpoint_specs:
        model_id = _model_id_for_checkpoint(BASELINE_MODEL, spec.finetune_id, spec.checkpoint_step)
        print(f"[{task}] benchmarking finetune={spec.finetune_id} checkpoint={spec.checkpoint_step}")
        metrics, records = _evaluate_tasks(
            task_name=task,
            model_label=f"{spec.finetune_id}@{spec.checkpoint_step}",
            model_id=model_id,
            tasks=tasks,
            api_base=api_base,
            api_keys=api_keys,
            max_tokens=int(config["max_tokens"]),
            max_objects=int(config["max_objects"]),
            timeout=float(timeout),
        )
        candidate_results.append(
            {
                "spec": spec,
                "metrics": metrics,
                "records": records,
            }
        )
        print(
            f"[{task}] finetune={spec.finetune_id} checkpoint={spec.checkpoint_step} "
            f"eval_f1={metrics['eval_f1']:.4f} eval_miou={metrics['eval_miou']:.4f} "
            f"failed={metrics['failed_tasks']}"
        )

    if not candidate_results:
        raise ValueError(f"no candidate checkpoints available for task={task}")

    best = max(candidate_results, key=lambda item: _checkpoint_score(item["metrics"], task))
    return {
        "tasks": tasks,
        "baseline_metrics": baseline_metrics,
        "baseline_records": baseline_records,
        "candidates": candidate_results,
        "best": best,
    }


def _metric_table_rows(task: str, baseline: dict[str, Any], candidate: dict[str, Any]) -> list[dict[str, Any]]:
    keys = ["eval_f1", "eval_f1_macro", "tp", "fp", "fn", "failed_tasks"]
    if task == "detect":
        keys.insert(2, "eval_miou")
    rows: list[dict[str, Any]] = []
    for key in keys:
        base_value = baseline.get(key, 0.0)
        candidate_value = candidate.get(key, 0.0)
        if isinstance(base_value, (int, float)) and isinstance(candidate_value, (int, float)):
            delta = float(candidate_value) - float(base_value)
        else:
            delta = None
        rows.append(
            {
                "metric": key,
                "baseline": base_value,
                "candidate": candidate_value,
                "delta": delta,
            }
        )
    return rows


def _record_index(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(record["task_key"]): record for record in records}


def _improvement_score(task: str, baseline_record: dict[str, Any], candidate_record: dict[str, Any]) -> tuple[float, float]:
    if task == "detect":
        return (
            float(candidate_record.get("task_miou", 0.0)) - float(baseline_record.get("task_miou", 0.0)),
            float(candidate_record.get("task_f1", 0.0)) - float(baseline_record.get("task_f1", 0.0)),
        )
    return (
        float(candidate_record.get("task_f1", 0.0)) - float(baseline_record.get("task_f1", 0.0)),
        float(candidate_record.get("task_miou", 0.0)) - float(baseline_record.get("task_miou", 0.0)),
    )


def _select_most_improved_by_class(
    *,
    task: str,
    baseline_records: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    base_by_key = _record_index(baseline_records)
    cand_by_key = _record_index(candidate_records)
    by_class: dict[str, tuple[tuple[float, float], dict[str, Any], dict[str, Any]]] = {}
    for task_key, candidate_record in cand_by_key.items():
        baseline_record = base_by_key.get(task_key)
        if baseline_record is None:
            continue
        class_name = str(candidate_record.get("class_name") or "")
        score = _improvement_score(task, baseline_record, candidate_record)
        current = by_class.get(class_name)
        if current is None or score > current[0]:
            by_class[class_name] = (score, baseline_record, candidate_record)
    return [(class_name, item[1], item[2]) for class_name, item in sorted(by_class.items())]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            payload = {key: value for key, value in row.items() if not key.startswith("_")}
            handle.write(json.dumps(payload, sort_keys=True))
            handle.write("\n")


def _render_metric_markdown(rows: list[dict[str, Any]]) -> str:
    lines = ["| Metric | Baseline | Result | Delta |", "| --- | ---: | ---: | ---: |"]
    for row in rows:
        baseline = row["baseline"]
        candidate = row["candidate"]
        delta = row["delta"]
        if isinstance(baseline, float):
            baseline_text = f"{baseline:.4f}"
        else:
            baseline_text = str(baseline)
        if isinstance(candidate, float):
            candidate_text = f"{candidate:.4f}"
        else:
            candidate_text = str(candidate)
        if isinstance(delta, float):
            delta_text = f"{delta:+.4f}"
        else:
            delta_text = str(delta)
        lines.append(f"| {row['metric']} | {baseline_text} | {candidate_text} | {delta_text} |")
    return "\n".join(lines)


def _render_run_evidence_markdown(evidence: list[RunEvidence]) -> str:
    lines = [
        "| Run | Finetune | Logged Baseline F1 | Logged Baseline mIoU | Logged Positive F1 | Source |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for item in evidence:
        lines.append(
            f"| {item.run_name} | {item.finetune_id} | {item.logged_baseline_f1:.4f} | "
            f"{item.logged_baseline_miou:.4f} | {item.logged_positive_f1:.4f} | {item.source_log} |"
        )
    return "\n".join(lines)


def _render_candidate_markdown(task: str, candidates: list[dict[str, Any]], selection_metric: str) -> str:
    lines = [
        f"| Finetune | Checkpoint | {selection_metric} | Eval F1 | Eval mIoU | Failed Tasks | Source Run |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in sorted(candidates, key=lambda row: _checkpoint_score(row["metrics"], task), reverse=True):
        spec: CheckpointSpec = item["spec"]
        metrics = item["metrics"]
        lines.append(
            f"| {spec.finetune_id} | {spec.checkpoint_step} | "
            f"{float(metrics.get(selection_metric, 0.0)):.4f} | {float(metrics.get('eval_f1', 0.0)):.4f} | "
            f"{float(metrics.get('eval_miou', 0.0)):.4f} | {int(metrics.get('failed_tasks', 0))} | {spec.source_run_name} |"
        )
    return "\n".join(lines)


def _save_showcase_visuals(
    *,
    task: str,
    improved_rows: list[tuple[str, dict[str, Any], dict[str, Any]]],
    out_dir: Path,
) -> list[dict[str, Any]]:
    showcase: list[dict[str, Any]] = []
    visuals_dir = out_dir / "visuals"
    for sample_index, (class_name, baseline_record, candidate_record) in enumerate(improved_rows):
        task_obj = candidate_record["_task"]
        baseline_path = bm._save_task_visualization(
            out_dir=visuals_dir,
            label=f"{task}_baseline",
            sample_idx=sample_index,
            task=task_obj,
            skill=task,
            pred_boxes=[] if task == "point" else [bm.Box(**box) for box in baseline_record["pred_boxes"]],
            pred_points=[bm.Point(**point) for point in baseline_record["pred_points"]] if task == "point" else [],
            iou_threshold=0.5,
            f1=float(baseline_record.get("task_f1", 0.0)),
            miou=float(baseline_record.get("task_miou", 0.0)),
            tp=int(baseline_record.get("tp", 0)),
            fp=int(baseline_record.get("fp", 0)),
            fn=int(baseline_record.get("fn", 0)),
        )
        candidate_path = bm._save_task_visualization(
            out_dir=visuals_dir,
            label=f"{task}_candidate",
            sample_idx=sample_index,
            task=task_obj,
            skill=task,
            pred_boxes=[] if task == "point" else [bm.Box(**box) for box in candidate_record["pred_boxes"]],
            pred_points=[bm.Point(**point) for point in candidate_record["pred_points"]] if task == "point" else [],
            iou_threshold=0.5,
            f1=float(candidate_record.get("task_f1", 0.0)),
            miou=float(candidate_record.get("task_miou", 0.0)),
            tp=int(candidate_record.get("tp", 0)),
            fp=int(candidate_record.get("fp", 0)),
            fn=int(candidate_record.get("fn", 0)),
        )
        showcase.append(
            {
                "class_name": class_name,
                "sample_id": candidate_record["sample_id"],
                "prompt": candidate_record["prompt"],
                "baseline_task_f1": float(baseline_record.get("task_f1", 0.0)),
                "candidate_task_f1": float(candidate_record.get("task_f1", 0.0)),
                "baseline_task_miou": float(baseline_record.get("task_miou", 0.0)),
                "candidate_task_miou": float(candidate_record.get("task_miou", 0.0)),
                "baseline_visualization": baseline_path,
                "candidate_visualization": candidate_path,
            }
        )
    return showcase


def _render_showcase_markdown(title: str, rows: list[dict[str, Any]]) -> str:
    sections = [f"## {title}"]
    for row in rows:
        sections.append(f"### {row['class_name']}")
        sections.append(f"- Sample: `{row['sample_id']}`")
        sections.append(f"- Prompt: `{row['prompt']}`")
        sections.append(
            f"- Baseline: f1={row['baseline_task_f1']:.4f}, miou={row['baseline_task_miou']:.4f}, "
            f"viz={row['baseline_visualization']}"
        )
        sections.append(
            f"- Result: f1={row['candidate_task_f1']:.4f}, miou={row['candidate_task_miou']:.4f}, "
            f"viz={row['candidate_visualization']}"
        )
    return "\n".join(sections)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    seed = int(args.seed)
    rng = random.Random(seed)
    env_file = common.resolve_path(args.env_file, module_root=SCRIPT_DIR)
    class_names_file = (
        common.resolve_path(args.class_names_file, module_root=SCRIPT_DIR)
        if str(args.class_names_file or "").strip()
        else None
    )
    api_keys = _load_runtime_api_keys(env_file, list(args.api_key_env_vars))
    output_dir = (
        common.resolve_path(args.output_dir, module_root=SCRIPT_DIR)
        if str(args.output_dir or "").strip()
        else DEFAULT_OUTPUT_ROOT / f"point_detect_report_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    report_payload: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "seed": seed,
        "base_url": args.base_url,
        "env_file": str(env_file),
        "output_dir": str(output_dir),
        "limitations": [
            "Inspector async checkpoint eval artifacts are incomplete in this repo because post-train eval jobs failed before writing metrics.",
            "This report uses repository run metadata to shortlist candidates, then re-benchmarks a stratified validation slice to pick the best checkpoint for each task.",
        ],
        "tasks": {},
    }

    markdown_sections = [
        "# Inspector Point / Detect Report",
        "",
        f"- Generated: `{report_payload['generated_at']}`",
        f"- Output dir: `{output_dir}`",
        f"- Seed: `{seed}`",
        f"- Base URL: `{args.base_url}`",
        "",
        "## Limitations",
        "- Async checkpoint eval outputs were not available for most runs, so shortlist ranking starts from sweep logs rather than completed eval JSONs.",
        "- Final checkpoint selection below comes from a clean re-benchmark on a stratified validation slice, not the broken async eval pipeline.",
    ]

    for task, config in (("point", POINT_CONFIG), ("detect", DETECT_CONFIG)):
        task_output_dir = output_dir / task
        task_output_dir.mkdir(parents=True, exist_ok=True)
        class_names = bm._load_class_names(str(class_names_file or ""), str(config["dataset_path"]))
        selected_rows = _select_eval_rows(
            dataset_path=Path(config["dataset_path"]),
            split=str(config["split"]),
            class_names=class_names,
            per_class_target=int(args.per_class_target),
            max_base_rows=int(args.max_base_rows),
            seed=rng.randint(0, 10**9),
        )
        evidence = _discover_run_evidence(task)
        checkpoint_specs = _build_checkpoint_specs(
            task=task,
            evidence=evidence,
            api_base=str(args.base_url),
            api_key=api_keys[0],
            top_runs=int(args.point_top_runs if task == "point" else args.detect_top_runs),
            detect_checkpoints_per_run=int(args.detect_checkpoints_per_run),
        )
        benchmark_payload = _best_candidate(
            task=task,
            rows=selected_rows,
            class_names=class_names,
            checkpoint_specs=checkpoint_specs,
            api_base=str(args.base_url),
            api_keys=api_keys,
            timeout=float(args.timeout),
        )
        best_spec: CheckpointSpec = benchmark_payload["best"]["spec"]
        best_metrics = benchmark_payload["best"]["metrics"]
        best_records = benchmark_payload["best"]["records"]
        baseline_metrics = benchmark_payload["baseline_metrics"]
        baseline_records = benchmark_payload["baseline_records"]
        metric_rows = _metric_table_rows(task, baseline_metrics, best_metrics)
        improved_rows = _select_most_improved_by_class(
            task=task,
            baseline_records=baseline_records,
            candidate_records=best_records,
        )
        showcase = _save_showcase_visuals(task=task, improved_rows=improved_rows, out_dir=task_output_dir)

        _write_jsonl(task_output_dir / "baseline.records.jsonl", baseline_records)
        for candidate in benchmark_payload["candidates"]:
            spec: CheckpointSpec = candidate["spec"]
            _write_jsonl(
                task_output_dir / f"{spec.finetune_id}@{spec.checkpoint_step}.records.jsonl",
                candidate["records"],
            )

        task_payload = {
            "config": {
                "dataset_path": str(config["dataset_path"]),
                "split": str(config["split"]),
                "selection_metric": str(config["selection_metric"]),
                "per_class_target": int(args.per_class_target),
                "max_base_rows": int(args.max_base_rows),
                "selected_row_count": len(selected_rows),
            },
            "run_evidence": [item.__dict__ for item in evidence],
            "checkpoint_specs": [item.__dict__ for item in checkpoint_specs],
            "baseline_metrics": baseline_metrics,
            "candidate_metrics": [
                {
                    "spec": candidate["spec"].__dict__,
                    "metrics": candidate["metrics"],
                }
                for candidate in benchmark_payload["candidates"]
            ],
            "best_checkpoint": best_spec.__dict__,
            "best_metrics": best_metrics,
            "metric_table": metric_rows,
            "showcase": showcase,
        }
        report_payload["tasks"][task] = task_payload

        markdown_sections.extend(
            [
                "",
                f"## {task.title()}",
                "",
                "### Run Evidence",
                _render_run_evidence_markdown(evidence),
                "",
                "### Candidate Checkpoints",
                _render_candidate_markdown(task, benchmark_payload["candidates"], str(config["selection_metric"])),
                "",
                f"### Best Checkpoint",
                f"- Finetune: `{best_spec.finetune_id}`",
                f"- Checkpoint: `{best_spec.checkpoint_step}`",
                f"- Source run: `{best_spec.source_run_name}`",
                "",
                "### Baseline vs Result",
                _render_metric_markdown(metric_rows),
                "",
                _render_showcase_markdown(f"{task.title()} Most Improved Samples", showcase),
            ]
        )

    _write_json(output_dir / "report.json", report_payload)
    (output_dir / "report.md").write_text("\n".join(markdown_sections) + "\n", encoding="utf-8")
    print(f"saved report -> {output_dir / 'report.md'}")
    print(f"saved payload -> {output_dir / 'report.json'}")


if __name__ == "__main__":
    main()
