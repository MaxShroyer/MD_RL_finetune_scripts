#!/usr/bin/env python3
"""Curate bone/aerial advertising subsets from local W&B runs and benchmark deltas."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MDpi_and_d import benchmark_pid_icons as benchmark_mod

DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "advertising_subsets"


@dataclass(frozen=True)
class BranchSpec:
    branch_id: str
    label: str
    benchmark_config: Path
    dataset_name: str
    local_dataset_path: Path
    split: str
    run_matcher: Callable[["LocalRun"], bool]


@dataclass(frozen=True)
class LocalRun:
    run_id: str
    config_path: str
    finetune_id: str
    dataset_ref: str
    best_step: int
    best_eval_f1: Optional[float]
    best_eval_miou: Optional[float]

    @property
    def score_hint(self) -> float:
        return float(self.best_eval_f1 or 0.0) + float(self.best_eval_miou or 0.0)


def _branch_specs() -> dict[str, BranchSpec]:
    def is_aerial_point(run: LocalRun) -> bool:
        return run.config_path.startswith("aerial_airport/configs/cicd/") and "train_aerial_airport_point" in run.config_path

    def is_aerial_detect(run: LocalRun) -> bool:
        return "train_aerial_airport_detect" in run.config_path

    def is_bone_point_angle_only(run: LocalRun) -> bool:
        return "train_bone_fracture_point" in run.config_path and "bone_fracture_point_angle_only_break_point_v1" in run.dataset_ref

    def is_bone_point_full(run: LocalRun) -> bool:
        return (
            "train_bone_fracture_point" in run.config_path
            and "bone_fracture_point_v1" in run.dataset_ref
            and "angle_only" not in run.dataset_ref
        )

    def is_bone_detect(run: LocalRun) -> bool:
        return "train_bone_fracture_detect" in run.config_path

    return {
        "aerial_point": BranchSpec(
            branch_id="aerial_point",
            label="Aerial Point",
            benchmark_config=REPO_ROOT / "aerial_airport" / "configs" / "benchmark_aerial_airport_point_best.json",
            dataset_name="maxs-m87/aerial_airport_point_v2",
            local_dataset_path=REPO_ROOT / "aerial_airport" / "outputs" / "maxs-m87_aerial_airport_point_v2",
            split="test",
            run_matcher=is_aerial_point,
        ),
        "aerial_detect": BranchSpec(
            branch_id="aerial_detect",
            label="Aerial Detect",
            benchmark_config=REPO_ROOT / "aerial_airport" / "configs" / "benchmark_aerial_airport_detect_default.json",
            dataset_name="maxs-m87/aerial_airport_point_v2",
            local_dataset_path=REPO_ROOT / "aerial_airport" / "outputs" / "maxs-m87_aerial_airport_point_v2",
            split="test",
            run_matcher=is_aerial_detect,
        ),
        "bone_point_angle_only": BranchSpec(
            branch_id="bone_point_angle_only",
            label="Bone Point Angle-Only",
            benchmark_config=REPO_ROOT / "bone_fracture" / "configs" / "benchmark_bone_fracture_point_best.json",
            dataset_name="maxs-m87/bone_fracture_point_angle_only_break_point_v1",
            local_dataset_path=REPO_ROOT / "bone_fracture" / "outputs" / "maxs-m87_bone_fracture_point_angle_only_break_point_v1",
            split="test",
            run_matcher=is_bone_point_angle_only,
        ),
        "bone_point_full": BranchSpec(
            branch_id="bone_point_full",
            label="Bone Point Full-Data",
            benchmark_config=REPO_ROOT / "bone_fracture" / "configs" / "benchmark_bone_fracture_point_full_best.json",
            dataset_name="maxs-m87/bone_fracture_point_v1",
            local_dataset_path=REPO_ROOT / "bone_fracture" / "outputs" / "maxs-m87_bone_fracture_point_v1",
            split="test",
            run_matcher=is_bone_point_full,
        ),
        "bone_detect": BranchSpec(
            branch_id="bone_detect",
            label="Bone Detect",
            benchmark_config=REPO_ROOT / "bone_fracture" / "configs" / "benchmark_bone_fracture_detect_best.json",
            dataset_name="maxs-m87/bone_fracture_detect_v1",
            local_dataset_path=REPO_ROOT / "bone_fracture" / "outputs" / "maxs-m87_bone_fracture_detect_v1",
            split="test",
            run_matcher=is_bone_detect,
        ),
    }


def _extract_wandb_scalar(text: str, key: str) -> Optional[str]:
    lines = text.splitlines()
    inside_key = False
    for line in lines:
        if not inside_key:
            if line == f"{key}:":
                inside_key = True
            continue
        if line and not line.startswith(" "):
            break
        stripped = line.strip()
        if stripped.startswith("value:"):
            raw = stripped.split(":", 1)[1].strip()
            if raw in {"", "null", "None"}:
                return None
            return raw.strip('"').strip("'")
    return None


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _repo_rel(value: Optional[str]) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    prefix = str(REPO_ROOT) + "/"
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def discover_local_runs(*, wandb_roots: list[Path]) -> list[LocalRun]:
    runs: list[LocalRun] = []
    for wandb_root in wandb_roots:
        if not wandb_root.exists():
            continue
        for run_dir in sorted(wandb_root.glob("run-*")):
            config_path = run_dir / "files" / "config.yaml"
            summary_path = run_dir / "files" / "wandb-summary.json"
            if not config_path.exists() or not summary_path.exists():
                continue
            config_text = config_path.read_text(encoding="utf-8", errors="replace")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            run_id = run_dir.name.rsplit("-", 1)[-1]
            finetune_id = (_extract_wandb_scalar(config_text, "finetune_id") or "").strip()
            config_rel = _repo_rel(_extract_wandb_scalar(config_text, "config"))
            dataset_path = _repo_rel(_extract_wandb_scalar(config_text, "dataset_path"))
            dataset_name = _repo_rel(_extract_wandb_scalar(config_text, "dataset_name"))
            dataset_ref = dataset_path or dataset_name
            best_step = _coerce_int(summary.get("best_checkpoint_step") or summary.get("best_step"))
            if not finetune_id or best_step is None or not config_rel:
                continue
            runs.append(
                LocalRun(
                    run_id=run_id,
                    config_path=config_rel,
                    finetune_id=finetune_id,
                    dataset_ref=dataset_ref,
                    best_step=best_step,
                    best_eval_f1=_coerce_float(summary.get("best_eval_f1")),
                    best_eval_miou=_coerce_float(summary.get("best_eval_miou")),
                )
            )
    return runs


def _branch_runs(spec: BranchSpec, runs: list[LocalRun], *, max_runs: Optional[int]) -> list[LocalRun]:
    matched = [run for run in runs if spec.run_matcher(run)]
    matched.sort(key=lambda run: (run.score_hint, run.run_id), reverse=True)
    if max_runs is not None:
        matched = matched[: max(0, int(max_runs))]
    return matched


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def _safe_name(value: str) -> str:
    text = "".join(ch if ch.isalnum() else "_" for ch in str(value or "").strip())
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_") or "dataset"


def _materialize_selection_source(
    *,
    spec: BranchSpec,
    split: str,
    output_dir: Path,
) -> dict[str, str]:
    split_name = str(split or spec.split)
    if "+" not in split_name:
        return {
            "dataset_name": spec.dataset_name,
            "dataset_path": "",
            "benchmark_split": split_name,
            "source_split": split_name,
        }
    merged_dataset = _load_dataset_split(spec=spec, split=split_name)
    materialized_path = output_dir / f"source_{_safe_name(split_name)}"
    if not materialized_path.exists():
        DatasetDict({spec.split: merged_dataset}).save_to_disk(str(materialized_path))
    return {
        "dataset_name": "",
        "dataset_path": str(materialized_path),
        "benchmark_split": spec.split,
        "source_split": split_name,
    }


def _benchmark_args(
    *,
    spec: BranchSpec,
    out_json: Path,
    records_jsonl: Path,
    dataset_name: str,
    dataset_path: str,
    run_id: str,
    finetune_id: str,
    checkpoint_step: Optional[int],
    skip_baseline: bool,
    max_samples: Optional[int],
    viz_samples: int,
    split: Optional[str] = None,
) -> list[str]:
    argv = [
        "--config",
        str(spec.benchmark_config),
        "--dataset-name",
        dataset_name,
        "--dataset-path",
        dataset_path,
        "--split",
        str(split or spec.split),
        "--out-json",
        str(out_json),
        "--records-jsonl",
        str(records_jsonl),
        "--run-id",
        run_id,
    ]
    # Explicitly blank these so branch-level baseline runs do not inherit
    # candidate finetune settings from the benchmark config file.
    argv.extend(["--model", "", "--finetune-id", ""])
    if max_samples is not None:
        argv.extend(["--max-samples", str(int(max_samples))])
    if viz_samples > 0:
        argv.extend(
            [
                "--viz-samples",
                str(int(viz_samples)),
                "--viz-dir",
                str((out_json.parent / "viz").resolve()),
            ]
        )
    if skip_baseline:
        argv.append("--skip-baseline")
    if finetune_id:
        argv.extend(["--finetune-id", finetune_id])
    if checkpoint_step is not None:
        argv.extend(["--checkpoint-step", str(int(checkpoint_step))])
    return argv


def _run_benchmark(
    *,
    spec: BranchSpec,
    out_json: Path,
    records_jsonl: Path,
    dataset_name: str,
    dataset_path: str,
    run_id: str,
    finetune_id: str,
    checkpoint_step: Optional[int],
    skip_baseline: bool,
    max_samples: Optional[int],
    viz_samples: int,
    split: Optional[str] = None,
) -> dict[str, Any]:
    argv = _benchmark_args(
        spec=spec,
        out_json=out_json,
        records_jsonl=records_jsonl,
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        run_id=run_id,
        finetune_id=finetune_id,
        checkpoint_step=checkpoint_step,
        skip_baseline=skip_baseline,
        max_samples=max_samples,
        viz_samples=viz_samples,
        split=split,
    )
    benchmark_mod.main(argv)
    return _load_json(out_json)


def _task_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(record.get("sample_id") or ""),
        str(record.get("class_name") or ""),
        str(record.get("prompt") or ""),
    )


def aggregate_sample_deltas(
    *,
    baseline_records: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    baseline_by_key = {
        _task_key(record): record
        for record in baseline_records
        if not bool(record.get("failed"))
    }
    grouped: dict[str, dict[str, Any]] = {}
    for record in candidate_records:
        if bool(record.get("failed")):
            continue
        key = _task_key(record)
        baseline = baseline_by_key.get(key)
        if baseline is None:
            continue
        sample_id = str(record.get("sample_id") or "")
        delta_f1 = float(record.get("task_f1") or 0.0) - float(baseline.get("task_f1") or 0.0)
        delta_miou = float(record.get("task_miou") or 0.0) - float(baseline.get("task_miou") or 0.0)
        delta_score = delta_f1 + delta_miou
        state = grouped.setdefault(
            sample_id,
            {
                "sample_id": sample_id,
                "matched_tasks": 0,
                "delta_f1_sum": 0.0,
                "delta_miou_sum": 0.0,
                "delta_score_sum": 0.0,
            },
        )
        state["matched_tasks"] += 1
        state["delta_f1_sum"] += delta_f1
        state["delta_miou_sum"] += delta_miou
        state["delta_score_sum"] += delta_score
    summaries: list[dict[str, Any]] = []
    for sample_id, state in sorted(grouped.items()):
        matched_tasks = int(state["matched_tasks"])
        summaries.append(
            {
                "sample_id": sample_id,
                "matched_tasks": matched_tasks,
                "delta_f1": float(state["delta_f1_sum"]) / matched_tasks,
                "delta_miou": float(state["delta_miou_sum"]) / matched_tasks,
                "delta_score": float(state["delta_score_sum"]) / matched_tasks,
            }
        )
    summaries.sort(key=lambda item: (float(item["delta_score"]), item["sample_id"]), reverse=True)
    return summaries


def select_marketing_samples(
    candidate_runs: list[dict[str, Any]],
    *,
    target_count: Optional[int] = None,
) -> list[dict[str, Any]]:
    best_by_sample: dict[str, dict[str, Any]] = {}
    for candidate in candidate_runs:
        run = candidate["run"]
        for sample in candidate["sample_summaries"]:
            sample_id = str(sample["sample_id"])
            current = best_by_sample.get(sample_id)
            if current is not None and float(current["delta_score"]) >= float(sample["delta_score"]):
                continue
            best_by_sample[sample_id] = {
                "sample_id": sample_id,
                "matched_tasks": int(sample["matched_tasks"]),
                "delta_f1": float(sample["delta_f1"]),
                "delta_miou": float(sample["delta_miou"]),
                "delta_score": float(sample["delta_score"]),
                "source_run_id": run.run_id,
                "source_finetune_id": run.finetune_id,
                "source_checkpoint_step": run.best_step,
                "source_config_path": run.config_path,
            }
    ranked = sorted(
        best_by_sample.values(),
        key=lambda item: (float(item["delta_score"]), item["sample_id"]),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    for row in ranked:
        if float(row["delta_score"]) <= 0.0:
            continue
        selected.append({**row, "selection_reason": "positive_delta"})
    if target_count is not None and len(selected) < max(0, int(target_count)):
        selected_ids = {str(row["sample_id"]) for row in selected}
        for row in ranked:
            sample_id = str(row["sample_id"])
            if sample_id in selected_ids:
                continue
            selected.append({**row, "selection_reason": "ranked_backfill"})
            selected_ids.add(sample_id)
            if len(selected) >= max(0, int(target_count)):
                break
    selected.sort(key=lambda item: (float(item["delta_score"]), item["sample_id"]), reverse=True)
    return selected


def _row_sample_id(row: dict[str, Any], fallback_id: int) -> str:
    return str(row.get("source_image_id") or row.get("id") or fallback_id)


def _load_dataset_split(
    *,
    spec: BranchSpec,
    dataset_path: Optional[Path] = None,
    split: Optional[str] = None,
) -> Dataset:
    split_name = str(split or spec.split)
    split_parts = [part.strip() for part in split_name.split("+") if part.strip()]

    def _select_from_dataset_dict(dataset_obj: DatasetDict) -> Dataset:
        if len(split_parts) == 1:
            part = split_parts[0]
            if part not in dataset_obj:
                available = ", ".join(dataset_obj.keys())
                raise ValueError(f"Split '{part}' not found. Available: {available}")
            return dataset_obj[part]
        missing = [part for part in split_parts if part not in dataset_obj]
        if missing:
            available = ", ".join(dataset_obj.keys())
            raise ValueError(f"Split(s) {missing} not found. Available: {available}")
        return concatenate_datasets([dataset_obj[part] for part in split_parts])

    if dataset_path is not None:
        dataset_obj = load_from_disk(str(dataset_path))
        if isinstance(dataset_obj, DatasetDict):
            return _select_from_dataset_dict(dataset_obj)
        return dataset_obj
    if spec.local_dataset_path.exists():
        dataset_obj = load_from_disk(str(spec.local_dataset_path))
        if isinstance(dataset_obj, DatasetDict):
            return _select_from_dataset_dict(dataset_obj)
        return dataset_obj
    return load_dataset(spec.dataset_name, split=split_name)


def save_subset_dataset(
    *,
    spec: BranchSpec,
    selected_samples: list[dict[str, Any]],
    output_dir: Path,
    source_split: Optional[str] = None,
    source_dataset_path: Optional[Path] = None,
) -> dict[str, Any]:
    selected_ids = {str(item["sample_id"]) for item in selected_samples}
    source_dataset = _load_dataset_split(spec=spec, split=source_split, dataset_path=source_dataset_path)
    if not selected_ids:
        return {
            "dataset_path": None,
            "split": spec.split,
            "selected_count": 0,
            "source_split": str(source_split or spec.split),
        }
    filtered = source_dataset.filter(
        lambda row, idx: _row_sample_id(row, idx) in selected_ids,
        with_indices=True,
    )
    if len(filtered) == 0:
        return {
            "dataset_path": None,
            "split": spec.split,
            "selected_count": 0,
            "source_split": str(source_split or spec.split),
        }
    subset = DatasetDict({spec.split: filtered})
    subset.save_to_disk(str(output_dir))
    return {
        "dataset_path": str(output_dir),
        "split": spec.split,
        "selected_count": len(filtered),
        "source_split": str(source_split or spec.split),
    }


def _metrics_score(metrics: Optional[dict[str, Any]]) -> float:
    if not metrics or "error" in metrics:
        return float("-inf")
    return float(metrics.get("eval_f1") or 0.0) + float(metrics.get("eval_miou") or 0.0)


def _metric_value(metrics: Optional[dict[str, Any]], key: str) -> float:
    if not metrics or "error" in metrics:
        return float("nan")
    return float(metrics.get(key) or 0.0)


def _branch_markdown(summary: dict[str, Any]) -> str:
    lines = [f"## {summary['label']}"]
    lines.append(f"- eligible runs: {summary['eligible_run_count']}")
    lines.append(f"- benchmarked runs: {summary['benchmarked_run_count']}")
    lines.append(f"- selected samples: {summary['selected_sample_count']}")
    winner = summary.get("display_winner")
    if winner:
        lines.append(
            f"- display winner: {winner['run_id']} @ step {winner['checkpoint_step']} "
            f"(subset score={winner['subset_score']:.6f})"
        )
        lines.append(
            f"- full before/after: baseline f1={_metric_value(summary['full_baseline'], 'eval_f1'):.6f}, "
            f"miou={_metric_value(summary['full_baseline'], 'eval_miou'):.6f} -> "
            f"winner f1={_metric_value(summary['full_winner'], 'eval_f1'):.6f}, "
            f"miou={_metric_value(summary['full_winner'], 'eval_miou'):.6f}"
        )
        lines.append(
            f"- subset before/after: baseline f1={_metric_value(summary['subset_baseline'], 'eval_f1'):.6f}, "
            f"miou={_metric_value(summary['subset_baseline'], 'eval_miou'):.6f} -> "
            f"winner f1={_metric_value(summary['subset_winner'], 'eval_f1'):.6f}, "
            f"miou={_metric_value(summary['subset_winner'], 'eval_miou'):.6f}"
        )
    else:
        lines.append("- display winner: none")
    return "\n".join(lines) + "\n"


def process_branch(
    *,
    spec: BranchSpec,
    runs: list[LocalRun],
    output_root: Path,
    max_runs: Optional[int],
    max_samples: Optional[int],
    viz_samples: int = 0,
    selection_split: Optional[str] = None,
    target_selected_samples: Optional[int] = None,
) -> dict[str, Any]:
    subset_source_split = str(selection_split or spec.split)
    branch_dir = output_root / spec.branch_id
    full_dir = branch_dir / "full"
    subset_dir = branch_dir / "subset"
    selection_dir = branch_dir / "selection_source"
    reports_dir = branch_dir / "reports"
    baseline_metrics_path = full_dir / "baseline.metrics.json"
    baseline_records_path = full_dir / "baseline.records.jsonl"
    selection_source = _materialize_selection_source(
        spec=spec,
        split=subset_source_split,
        output_dir=selection_dir,
    )
    print(f"[{spec.branch_id}] running baseline benchmark...")
    baseline_metrics = _run_benchmark(
        spec=spec,
        out_json=baseline_metrics_path,
        records_jsonl=baseline_records_path,
        dataset_name=spec.dataset_name,
        dataset_path="",
        run_id="",
        finetune_id="",
        checkpoint_step=None,
        skip_baseline=False,
        max_samples=max_samples,
        viz_samples=viz_samples,
        split=spec.split,
    )
    branch_runs = _branch_runs(spec, runs, max_runs=max_runs)

    if subset_source_split == spec.split:
        selection_baseline_metrics = baseline_metrics
        selection_baseline_records = _load_jsonl(baseline_records_path)
    else:
        selection_baseline_metrics_path = selection_dir / "baseline.metrics.json"
        selection_baseline_records_path = selection_dir / "baseline.records.jsonl"
        print(f"[{spec.branch_id}] running selection-source baseline benchmark on split {subset_source_split}...")
        selection_baseline_metrics = _run_benchmark(
            spec=spec,
            out_json=selection_baseline_metrics_path,
            records_jsonl=selection_baseline_records_path,
            dataset_name=selection_source["dataset_name"],
            dataset_path=selection_source["dataset_path"],
            run_id="",
            finetune_id="",
            checkpoint_step=None,
            skip_baseline=False,
            max_samples=max_samples,
            viz_samples=0,
            split=selection_source["benchmark_split"],
        )
        selection_baseline_records = _load_jsonl(selection_baseline_records_path)

    candidate_rows: list[dict[str, Any]] = []
    for run in branch_runs:
        run_dir = full_dir / "runs" / run.run_id
        metrics_path = run_dir / "candidate.metrics.json"
        records_path = run_dir / "candidate.records.jsonl"
        samples_path = run_dir / "sample_deltas.jsonl"
        print(f"[{spec.branch_id}] benchmarking run {run.run_id} @ step {run.best_step}...")
        try:
            metrics = _run_benchmark(
                spec=spec,
                out_json=metrics_path,
                records_jsonl=records_path,
                dataset_name=spec.dataset_name,
                dataset_path="",
                run_id=run.run_id,
                finetune_id=run.finetune_id,
                checkpoint_step=run.best_step,
                skip_baseline=True,
                max_samples=max_samples,
                viz_samples=viz_samples,
                split=spec.split,
            )
        except Exception as exc:
            print(f"[{spec.branch_id}] skipping run {run.run_id}: {exc}")
            continue
        if subset_source_split == spec.split:
            selection_candidate_records = _load_jsonl(records_path)
        else:
            selection_run_dir = selection_dir / "runs" / run.run_id
            selection_metrics_path = selection_run_dir / "candidate.metrics.json"
            selection_records_path = selection_run_dir / "candidate.records.jsonl"
            print(f"[{spec.branch_id}] benchmarking selection-source run {run.run_id} on split {subset_source_split}...")
            try:
                _run_benchmark(
                    spec=spec,
                    out_json=selection_metrics_path,
                    records_jsonl=selection_records_path,
                    dataset_name=selection_source["dataset_name"],
                    dataset_path=selection_source["dataset_path"],
                    run_id=run.run_id,
                    finetune_id=run.finetune_id,
                    checkpoint_step=run.best_step,
                    skip_baseline=True,
                    max_samples=max_samples,
                    viz_samples=0,
                    split=selection_source["benchmark_split"],
                )
            except Exception as exc:
                print(f"[{spec.branch_id}] skipping selection-source run {run.run_id}: {exc}")
                continue
            selection_candidate_records = _load_jsonl(selection_records_path)
        sample_summaries = aggregate_sample_deltas(
            baseline_records=selection_baseline_records,
            candidate_records=selection_candidate_records,
        )
        _write_jsonl(samples_path, sample_summaries)
        candidate_rows.append(
            {
                "run": run,
                "full_metrics": metrics,
                "full_metrics_path": str(metrics_path),
                "full_records_path": str(records_path),
                "sample_summaries": sample_summaries,
            }
        )

    selected_samples = select_marketing_samples(candidate_rows, target_count=target_selected_samples)
    selected_manifest_path = subset_dir / "selected_samples.jsonl"
    _write_jsonl(selected_manifest_path, selected_samples)
    subset_dataset_path = subset_dir / "dataset"
    subset_info = save_subset_dataset(
        spec=spec,
        selected_samples=selected_samples,
        output_dir=subset_dataset_path,
        source_split=selection_source["benchmark_split"],
        source_dataset_path=(Path(selection_source["dataset_path"]) if selection_source["dataset_path"] else None),
    )

    summary = {
        "branch_id": spec.branch_id,
        "label": spec.label,
        "dataset_name": spec.dataset_name,
        "selection_split": subset_source_split,
        "target_selected_samples": target_selected_samples,
        "eligible_run_count": len(branch_runs),
        "benchmarked_run_count": len(candidate_rows),
        "selected_sample_count": len(selected_samples),
        "selected_samples_manifest": str(selected_manifest_path),
        "subset_dataset": subset_info,
        "full_baseline": baseline_metrics,
        "selection_source_baseline": selection_baseline_metrics,
        "subset_baseline": None,
        "display_winner": None,
        "full_winner": None,
        "subset_winner": None,
        "leaderboard_path": None,
    }

    if subset_info["selected_count"] == 0:
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_json_path = reports_dir / "summary.json"
        report_md_path = reports_dir / "summary.md"
        _write_json(report_json_path, summary)
        report_md_path.write_text(_branch_markdown(summary), encoding="utf-8")
        return summary

    subset_baseline_path = subset_dir / "baseline.metrics.json"
    subset_baseline_records_path = subset_dir / "baseline.records.jsonl"
    print(f"[{spec.branch_id}] benchmarking subset baseline...")
    subset_baseline = _run_benchmark(
        spec=spec,
        out_json=subset_baseline_path,
        records_jsonl=subset_baseline_records_path,
        dataset_name="",
        dataset_path=str(subset_dataset_path),
        run_id="",
        finetune_id="",
        checkpoint_step=None,
        skip_baseline=False,
        max_samples=max_samples,
        viz_samples=viz_samples,
        split=spec.split,
    )

    leaderboard: list[dict[str, Any]] = []
    for candidate in candidate_rows:
        run = candidate["run"]
        subset_run_dir = subset_dir / "runs" / run.run_id
        subset_metrics_path = subset_run_dir / "candidate.metrics.json"
        subset_records_path = subset_run_dir / "candidate.records.jsonl"
        print(f"[{spec.branch_id}] benchmarking subset run {run.run_id}...")
        try:
            subset_metrics = _run_benchmark(
                spec=spec,
                out_json=subset_metrics_path,
                records_jsonl=subset_records_path,
                dataset_name="",
                dataset_path=str(subset_dataset_path),
                run_id=run.run_id,
                finetune_id=run.finetune_id,
                checkpoint_step=run.best_step,
                skip_baseline=True,
                max_samples=max_samples,
                viz_samples=viz_samples,
                split=spec.split,
            )
        except Exception as exc:
            print(f"[{spec.branch_id}] subset benchmark failed for {run.run_id}: {exc}")
            subset_metrics = {"error": str(exc)}
        leaderboard.append(
            {
                "run_id": run.run_id,
                "finetune_id": run.finetune_id,
                "checkpoint_step": run.best_step,
                "config_path": run.config_path,
                "full_metrics": candidate["full_metrics"],
                "subset_metrics": subset_metrics,
                "full_score": _metrics_score(candidate["full_metrics"]),
                "subset_score": _metrics_score(subset_metrics),
                "subset_miou": _metric_value(subset_metrics, "eval_miou"),
            }
        )
    leaderboard.sort(
        key=lambda item: (
            float(item["subset_score"]),
            float(item["subset_miou"]),
            float(item["full_score"]),
            item["run_id"],
        ),
        reverse=True,
    )
    leaderboard_path = reports_dir / "leaderboard.json"
    _write_json(leaderboard_path, leaderboard)

    winner = None
    for item in leaderboard:
        if "error" in item["subset_metrics"]:
            continue
        winner = item
        break

    summary["subset_baseline"] = subset_baseline
    summary["leaderboard_path"] = str(leaderboard_path)
    if winner is not None:
        summary["display_winner"] = {
            "run_id": winner["run_id"],
            "finetune_id": winner["finetune_id"],
            "checkpoint_step": winner["checkpoint_step"],
            "subset_score": float(winner["subset_score"]),
            "full_score": float(winner["full_score"]),
        }
        summary["full_winner"] = winner["full_metrics"]
        summary["subset_winner"] = winner["subset_metrics"]

    report_json_path = reports_dir / "summary.json"
    report_md_path = reports_dir / "summary.md"
    _write_json(report_json_path, summary)
    report_md_path.parent.mkdir(parents=True, exist_ok=True)
    report_md_path.write_text(_branch_markdown(summary), encoding="utf-8")
    return summary


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    specs = _branch_specs()
    parser = argparse.ArgumentParser(description="Build bone/aerial advertising subsets from benchmark deltas.")
    parser.add_argument(
        "--branch",
        dest="branches",
        action="append",
        choices=sorted(specs),
        help="Branch to process. Repeat to process multiple branches. Defaults to all.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-runs-per-branch", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--viz-samples", type=int, default=0)
    parser.add_argument(
        "--selection-split",
        default="",
        help="Dataset split used to source/select advertising samples. Supports expressions like train+validation+test. Defaults to the canonical benchmark split.",
    )
    parser.add_argument(
        "--target-selected-samples",
        type=int,
        default=None,
        help="Target number of selected samples per branch. If positive-delta samples are fewer, the curator backfills with the next best ranked samples.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    specs = _branch_specs()
    selected = list(args.branches or sorted(specs))
    runs = discover_local_runs(wandb_roots=[REPO_ROOT / "wandb", REPO_ROOT / "bone_fracture" / "wandb"])
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    for branch_id in selected:
        summaries.append(
            process_branch(
                spec=specs[branch_id],
                runs=runs,
                output_root=output_root,
                max_runs=args.max_runs_per_branch,
                max_samples=args.max_samples,
                viz_samples=args.viz_samples,
                selection_split=(args.selection_split or None),
                target_selected_samples=args.target_selected_samples,
            )
        )

    report_json_path = output_root / "report.json"
    report_md_path = output_root / "report.md"
    _write_json(report_json_path, {"branches": summaries})
    report_md_path.write_text("".join(_branch_markdown(summary) for summary in summaries), encoding="utf-8")
    print(f"saved report -> {report_json_path}")


if __name__ == "__main__":
    main()
