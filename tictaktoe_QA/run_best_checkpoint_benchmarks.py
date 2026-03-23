#!/usr/bin/env python3
"""Benchmark the best saved TicTacToe QA checkpoints per task."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tictaktoe_QA.task_schema import normalize_task_type

DEFAULT_MANIFEST_PATH = (
    Path(__file__).resolve().parent / "configs" / "benchmark_best_checkpoints_manifest.json"
)
DEFAULT_BENCHMARK_CONFIG = Path(__file__).resolve().parent / "configs" / "benchmark_default.json"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "best_checkpoint_validation"
DEFAULT_BASELINE_MODEL = "moondream3-preview"


@dataclass(frozen=True)
class ManifestEntry:
    task: str
    finetune_id: str
    checkpoint_step: int
    source: str = ""
    note: str = ""


def _parse_tasks(raw_values: Optional[list[str]]) -> Optional[list[str]]:
    if not raw_values:
        return None

    out: list[str] = []
    seen: set[str] = set()
    for value in raw_values:
        for piece in str(value).split(","):
            task = piece.strip()
            if not task:
                continue
            normalized = normalize_task_type(task)
            if normalized in seen:
                continue
            seen.add(normalized)
            out.append(normalized)

    return out or None


def _load_manifest(manifest_path: Path) -> list[ManifestEntry]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"--manifest not found: {manifest_path}")

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Manifest must be a JSON array: {manifest_path}")

    entries: list[ManifestEntry] = []
    seen_tasks: set[str] = set()
    for idx, raw_entry in enumerate(payload):
        if not isinstance(raw_entry, dict):
            raise ValueError(f"Manifest entry #{idx} must be a JSON object")

        task_raw = str(raw_entry.get("task", "")).strip()
        if not task_raw:
            raise ValueError(f"Manifest entry #{idx} is missing task")
        task = normalize_task_type(task_raw)
        if task in seen_tasks:
            raise ValueError(f"Duplicate task in manifest: {task}")
        seen_tasks.add(task)

        finetune_id = str(raw_entry.get("finetune_id", "")).strip()
        if not finetune_id:
            raise ValueError(f"Manifest entry #{idx} ({task}) is missing finetune_id")

        checkpoint_raw = raw_entry.get("checkpoint_step")
        try:
            checkpoint_step = int(checkpoint_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Manifest entry #{idx} ({task}) has invalid checkpoint_step: {checkpoint_raw}"
            ) from exc
        if checkpoint_step < 0:
            raise ValueError(f"Manifest entry #{idx} ({task}) has invalid checkpoint_step: {checkpoint_step}")

        entries.append(
            ManifestEntry(
                task=task,
                finetune_id=finetune_id,
                checkpoint_step=checkpoint_step,
                source=str(raw_entry.get("source", "")).strip(),
                note=str(raw_entry.get("note", "")).strip(),
            )
        )

    return entries


def _select_entries(entries: list[ManifestEntry], tasks: Optional[list[str]]) -> list[ManifestEntry]:
    if not tasks:
        return list(entries)

    selected_by_task = {entry.task: entry for entry in entries}
    missing = [task for task in tasks if task not in selected_by_task]
    if missing:
        raise ValueError(f"Requested task(s) not found in manifest: {missing}")
    return [selected_by_task[task] for task in tasks]


def _build_benchmark_command(
    *,
    benchmark_config: Path,
    task: str,
    output_json: Path,
    predictions_jsonl: Optional[Path],
    max_samples: Optional[int],
    no_progress: bool,
    model: str,
    finetune_id: str,
    checkpoint_step: int,
) -> list[str]:
    command = [
        sys.executable,
        str((Path(__file__).resolve().parent / "benchmark_ttt_query.py").resolve()),
        "--config",
        str(benchmark_config),
        "--task-types",
        str(task),
        "--model",
        str(model),
        "--finetune-id",
        str(finetune_id),
        "--checkpoint-step",
        str(int(checkpoint_step)),
        "--output-json",
        str(output_json),
        "--predictions-jsonl",
        str(predictions_jsonl) if predictions_jsonl is not None else "",
    ]
    if max_samples is not None and int(max_samples) > 0:
        command.extend(["--max-samples", str(int(max_samples))])
    if no_progress:
        command.append("--no-progress")
    return command


def _build_candidate_command(
    *,
    benchmark_config: Path,
    entry: ManifestEntry,
    output_json: Path,
    predictions_jsonl: Optional[Path],
    max_samples: Optional[int],
    no_progress: bool,
) -> list[str]:
    return _build_benchmark_command(
        benchmark_config=benchmark_config,
        task=entry.task,
        output_json=output_json,
        predictions_jsonl=predictions_jsonl,
        max_samples=max_samples,
        no_progress=no_progress,
        model="",
        finetune_id=entry.finetune_id,
        checkpoint_step=entry.checkpoint_step,
    )


def _build_baseline_command(
    *,
    benchmark_config: Path,
    task: str,
    output_json: Path,
    predictions_jsonl: Optional[Path],
    max_samples: Optional[int],
    no_progress: bool,
) -> list[str]:
    return _build_benchmark_command(
        benchmark_config=benchmark_config,
        task=task,
        output_json=output_json,
        predictions_jsonl=predictions_jsonl,
        max_samples=max_samples,
        no_progress=no_progress,
        model=DEFAULT_BASELINE_MODEL,
        finetune_id="",
        checkpoint_step=-1,
    )


def _load_json_if_exists(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _extract_task_metrics(payload: Optional[dict[str, Any]], task: str) -> tuple[Optional[float], Optional[int]]:
    if not isinstance(payload, dict):
        return (None, None)

    by_task = payload.get("by_task")
    if not isinstance(by_task, dict):
        return (None, None)

    task_metrics: Optional[dict[str, Any]] = None
    for raw_key, raw_value in by_task.items():
        if not isinstance(raw_value, dict):
            continue
        normalized_key = normalize_task_type(str(raw_key), allow_unknown=True)
        if normalized_key == task:
            task_metrics = raw_value
            break
    if not isinstance(task_metrics, dict):
        return (None, None)

    accuracy_raw = task_metrics.get("accuracy")
    count_raw = task_metrics.get("count")
    try:
        accuracy = float(accuracy_raw)
    except (TypeError, ValueError):
        accuracy = None
    try:
        count = int(count_raw)
    except (TypeError, ValueError):
        count = None
    return (accuracy, count)


def _status_for_summary(
    *,
    candidate_return_code: int,
    candidate_accuracy: Optional[float],
    baseline_return_code: Optional[int],
    baseline_accuracy: Optional[float],
    include_baseline: bool,
) -> str:
    if int(candidate_return_code) != 0 or candidate_accuracy is None:
        return "failed"
    if include_baseline and (
        baseline_return_code is None or int(baseline_return_code) != 0 or baseline_accuracy is None
    ):
        return "failed"
    return "completed"


def _build_summary_row(
    *,
    entry: ManifestEntry,
    candidate_output_json: Path,
    candidate_payload: Optional[dict[str, Any]],
    candidate_return_code: int,
    baseline_output_json: Optional[Path],
    baseline_payload: Optional[dict[str, Any]],
    baseline_return_code: Optional[int],
    include_baseline: bool,
) -> dict[str, Any]:
    candidate_accuracy, candidate_count = _extract_task_metrics(candidate_payload, entry.task)
    baseline_accuracy, baseline_count = _extract_task_metrics(baseline_payload, entry.task)

    count: Optional[int] = candidate_count if candidate_count is not None else baseline_count
    delta_vs_baseline: Optional[float] = None
    if candidate_accuracy is not None and baseline_accuracy is not None:
        delta_vs_baseline = candidate_accuracy - baseline_accuracy

    return {
        "task": entry.task,
        "finetune_id": entry.finetune_id,
        "checkpoint_step": int(entry.checkpoint_step),
        "candidate_accuracy": candidate_accuracy,
        "baseline_accuracy": baseline_accuracy,
        "delta_vs_baseline": delta_vs_baseline,
        "count": count,
        "candidate_output_json": str(candidate_output_json),
        "baseline_output_json": str(baseline_output_json) if baseline_output_json is not None else "",
        "status": _status_for_summary(
            candidate_return_code=candidate_return_code,
            candidate_accuracy=candidate_accuracy,
            baseline_return_code=baseline_return_code,
            baseline_accuracy=baseline_accuracy,
            include_baseline=include_baseline,
        ),
        "source": entry.source,
        "note": entry.note,
    }


def _run_benchmark(command: list[str], *, task_dir: Path, prefix: str) -> int:
    proc = subprocess.run(command, text=True, capture_output=True, cwd=str(REPO_ROOT))
    (task_dir / f"{prefix}_stdout.log").write_text(proc.stdout, encoding="utf-8")
    (task_dir / f"{prefix}_stderr.log").write_text(proc.stderr, encoding="utf-8")
    return int(proc.returncode)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark the best saved TicTacToe checkpoints per task.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--benchmark-config", default=str(DEFAULT_BENCHMARK_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Optional subset of manifest tasks to run; accepts comma-separated values.",
    )
    parser.add_argument("--max-samples", type=int, default=0, help="<=0 keeps the benchmark config value.")
    baseline_group = parser.add_mutually_exclusive_group()
    baseline_group.add_argument("--include-baseline", dest="include_baseline", action="store_true")
    baseline_group.add_argument("--no-include-baseline", dest="include_baseline", action="store_false")
    parser.set_defaults(include_baseline=True)

    predictions_group = parser.add_mutually_exclusive_group()
    predictions_group.add_argument("--write-predictions", dest="write_predictions", action="store_true")
    predictions_group.add_argument("--no-write-predictions", dest="write_predictions", action="store_false")
    parser.set_defaults(write_predictions=True)

    parser.add_argument("--no-progress", action="store_true", default=False)
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    args = _build_parser().parse_args(argv)

    manifest_path = Path(args.manifest).expanduser().resolve()
    benchmark_config = Path(args.benchmark_config).expanduser().resolve()
    if not benchmark_config.exists():
        raise FileNotFoundError(f"--benchmark-config not found: {benchmark_config}")

    manifest_entries = _load_manifest(manifest_path)
    selected_tasks = _parse_tasks(args.tasks)
    entries = _select_entries(manifest_entries, selected_tasks)
    if not entries:
        raise ValueError("No manifest entries selected")

    output_dir = Path(args.output_dir).expanduser().resolve()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_root = output_dir / timestamp
    run_root.mkdir(parents=True, exist_ok=True)

    max_samples = None if int(args.max_samples) <= 0 else int(args.max_samples)
    rows: list[dict[str, Any]] = []
    for entry in entries:
        task_dir = run_root / entry.task
        task_dir.mkdir(parents=True, exist_ok=True)

        candidate_output_json = task_dir / "candidate_metrics.json"
        candidate_predictions_jsonl = (
            task_dir / "candidate_predictions.jsonl" if bool(args.write_predictions) else None
        )
        candidate_command = _build_candidate_command(
            benchmark_config=benchmark_config,
            entry=entry,
            output_json=candidate_output_json,
            predictions_jsonl=candidate_predictions_jsonl,
            max_samples=max_samples,
            no_progress=bool(args.no_progress),
        )
        print(f"running candidate {entry.task}: {' '.join(candidate_command)}")
        candidate_return_code = _run_benchmark(candidate_command, task_dir=task_dir, prefix="candidate")
        if candidate_return_code != 0:
            print(f"candidate failed task={entry.task} return_code={candidate_return_code}")
        candidate_payload = _load_json_if_exists(candidate_output_json)

        baseline_output_json: Optional[Path] = None
        baseline_payload: Optional[dict[str, Any]] = None
        baseline_return_code: Optional[int] = None
        if bool(args.include_baseline):
            baseline_output_json = task_dir / "baseline_metrics.json"
            baseline_predictions_jsonl = (
                task_dir / "baseline_predictions.jsonl" if bool(args.write_predictions) else None
            )
            baseline_command = _build_baseline_command(
                benchmark_config=benchmark_config,
                task=entry.task,
                output_json=baseline_output_json,
                predictions_jsonl=baseline_predictions_jsonl,
                max_samples=max_samples,
                no_progress=bool(args.no_progress),
            )
            print(f"running baseline {entry.task}: {' '.join(baseline_command)}")
            baseline_return_code = _run_benchmark(
                baseline_command,
                task_dir=task_dir,
                prefix="baseline",
            )
            if baseline_return_code != 0:
                print(f"baseline failed task={entry.task} return_code={baseline_return_code}")
            baseline_payload = _load_json_if_exists(baseline_output_json)

        rows.append(
            _build_summary_row(
                entry=entry,
                candidate_output_json=candidate_output_json,
                candidate_payload=candidate_payload,
                candidate_return_code=candidate_return_code,
                baseline_output_json=baseline_output_json,
                baseline_payload=baseline_payload,
                baseline_return_code=baseline_return_code,
                include_baseline=bool(args.include_baseline),
            )
        )

    summary_json_path = run_root / "summary.json"
    summary_csv_path = run_root / "summary.csv"
    summary_payload = {
        "manifest": str(manifest_path),
        "benchmark_config": str(benchmark_config),
        "run_root": str(run_root),
        "include_baseline": bool(args.include_baseline),
        "write_predictions": bool(args.write_predictions),
        "results": rows,
    }
    summary_json_path.write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    with summary_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "task",
                "finetune_id",
                "checkpoint_step",
                "candidate_accuracy",
                "baseline_accuracy",
                "delta_vs_baseline",
                "count",
                "candidate_output_json",
                "baseline_output_json",
                "status",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in writer.fieldnames})

    print(f"wrote summary: {summary_json_path}")
    print(f"wrote summary CSV: {summary_csv_path}")


if __name__ == "__main__":
    main()
