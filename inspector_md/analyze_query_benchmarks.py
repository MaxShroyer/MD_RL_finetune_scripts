#!/usr/bin/env python3
"""Build compact comparison tables and charts for Inspector MD query benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inspector_md import common

SCRIPT_DIR = Path(__file__).resolve().parent
_STRICT_COMPARISON_METRIC_MAP = {
    "reward_mean": "strict_reward_mean",
    "local_reward_mean": "strict_local_reward_mean",
    "judge_score_mean": "strict_judge_score_mean",
    "json_parse_rate": "strict_parse_rate",
    "task_correct_rate": "strict_task_correct_rate",
    "issue_f1": "strict_issue_f1",
    "evidence_f1": "strict_reasoning_f1",
}


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Inspector MD query benchmark outputs.")
    parser.add_argument("--input-jsons", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--title", default="Inspector MD Query Benchmark Comparison")
    args = parser.parse_args(argv)
    args.input_jsons = [common.resolve_path(path, module_root=SCRIPT_DIR) for path in list(args.input_jsons)]
    args.output_dir = common.resolve_path(args.output_dir, module_root=SCRIPT_DIR)
    return args


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_") or "run"


def _task_label(dataset_dir: str) -> str:
    text = str(dataset_dir or "").lower()
    if "query_proposal" in text:
        return "proposal"
    if "subset_balanced_1000" in text and "query_finding" in text:
        return "finding_subset1k"
    if "query_finding" in text:
        return "finding_full"
    return "query"


def _run_label(row: dict[str, Any]) -> str:
    finetune_id = str(row.get("finetune_id") or "").strip()
    resolved_step = row.get("resolved_checkpoint_step")
    task = _task_label(str(row.get("dataset_dir") or ""))
    if finetune_id and resolved_step is not None:
        return f"{task}:{finetune_id[:8]}@{int(resolved_step)}"
    if finetune_id:
        return f"{task}:{finetune_id[:8]}"
    model = str(row.get("model") or "").strip()
    return f"{task}:{model or 'unknown'}"


def _use_strict_comparison(rows: list[dict[str, Any]]) -> bool:
    return any(not bool(row.get("comparison_safe_answer_parse_mode", False)) for row in rows)


def _comparison_value(row: dict[str, Any], key: str, *, use_strict: bool) -> Any:
    if use_strict:
        strict_key = _STRICT_COMPARISON_METRIC_MAP.get(key, "")
        if strict_key and strict_key in row:
            return row.get(strict_key)
    return row.get(key)


def _annotate_comparison_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    use_strict = _use_strict_comparison(rows)
    comparison_mode = "strict_rescored" if use_strict else "primary"
    notes: list[str] = []
    if use_strict:
        notes.append(
            "At least one benchmark used a non-comparison-safe answer parse mode; tables and charts are using strict rescored metrics for every row."
        )
    parse_modes = sorted({str(row.get("answer_parse_mode") or "") for row in rows if str(row.get("answer_parse_mode") or "").strip()})
    for row in rows:
        row["comparison_metric_mode"] = comparison_mode
        row["comparison_metric_notes"] = list(notes)
        row["comparison_reward_mean"] = _comparison_value(row, "reward_mean", use_strict=use_strict)
        row["comparison_local_reward_mean"] = _comparison_value(row, "local_reward_mean", use_strict=use_strict)
        row["comparison_judge_score_mean"] = _comparison_value(row, "judge_score_mean", use_strict=use_strict)
        row["comparison_parse_rate"] = _comparison_value(row, "json_parse_rate", use_strict=use_strict)
        row["comparison_task_correct_rate"] = _comparison_value(row, "task_correct_rate", use_strict=use_strict)
        row["comparison_issue_f1"] = _comparison_value(row, "issue_f1", use_strict=use_strict)
        row["comparison_evidence_f1"] = _comparison_value(row, "evidence_f1", use_strict=use_strict)
        row["comparison_answer_parse_modes"] = list(parse_modes)
    rows.sort(key=lambda item: float(item.get("comparison_reward_mean", 0.0) or 0.0), reverse=True)
    return rows


def _load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        row = dict(payload)
        row["source_json"] = str(path)
        row["run_label"] = _run_label(row)
        row["task_label"] = _task_label(str(row.get("dataset_dir") or ""))
        rows.append(row)
    return _annotate_comparison_metrics(rows)


def _format_float(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.4f}"


def _write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    fields = [
        "run_label",
        "task_label",
        "answer_parse_mode",
        "comparison_metric_mode",
        "finetune_id",
        "checkpoint_step",
        "resolved_checkpoint_step",
        "used_checkpoint_fallback",
        "count",
        "comparison_reward_mean",
        "comparison_local_reward_mean",
        "comparison_judge_score_mean",
        "comparison_parse_rate",
        "comparison_task_correct_rate",
        "comparison_issue_f1",
        "comparison_evidence_f1",
        "reward_mean",
        "local_reward_mean",
        "judge_score_mean",
        "json_parse_rate",
        "task_correct_rate",
        "issue_f1",
        "evidence_f1",
        "severity_accuracy",
        "insufficient_accuracy",
        "dataset_dir",
        "predictions_jsonl",
        "source_json",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _write_markdown(rows: list[dict[str, Any]], output_path: Path, *, title: str) -> None:
    comparison_mode = str(rows[0].get("comparison_metric_mode") or "primary") if rows else "primary"
    lines = [
        f"# {title}",
        "",
        f"- comparison_metric_mode: `{comparison_mode}`",
        "",
        "| Run | Task | Parse Mode | Requested | Resolved | Reward | Local | Judge | Parse | Correct | Issue F1 | Evidence F1 |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    if rows:
        for note in list(rows[0].get("comparison_metric_notes") or []):
            lines.append(f"- note: {note}")
        if list(rows[0].get("comparison_metric_notes") or []):
            lines.append("")
    for row in rows:
        requested = str(row.get("checkpoint_step") if row.get("checkpoint_step") is not None else "")
        resolved = str(row.get("resolved_checkpoint_step") if row.get("resolved_checkpoint_step") is not None else "")
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("run_label") or ""),
                    str(row.get("task_label") or ""),
                    str(row.get("answer_parse_mode") or ""),
                    requested,
                    resolved,
                    _format_float(row.get("comparison_reward_mean")),
                    _format_float(row.get("comparison_local_reward_mean")),
                    _format_float(row.get("comparison_judge_score_mean")),
                    _format_float(row.get("comparison_parse_rate")),
                    _format_float(row.get("comparison_task_correct_rate")),
                    _format_float(row.get("comparison_issue_f1")),
                    _format_float(row.get("comparison_evidence_f1")),
                ]
            )
            + " |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_core_metrics(rows: list[dict[str, Any]], output_path: Path, *, title: str) -> None:
    labels = [str(row.get("run_label") or "") for row in rows]
    metrics = [
        ("comparison_reward_mean", "Reward"),
        ("comparison_task_correct_rate", "Task Correct"),
        ("comparison_issue_f1", "Issue F1"),
        ("comparison_evidence_f1", "Evidence F1"),
        ("comparison_parse_rate", "Parse"),
    ]
    x = list(range(len(labels)))
    width = 0.14 if rows else 0.14
    fig, ax = plt.subplots(figsize=(max(10.0, len(labels) * 2.3), 6.5), constrained_layout=True)
    for index, (key, label) in enumerate(metrics):
        offset = (index - ((len(metrics) - 1) / 2.0)) * width
        values = [float(row.get(key, 0.0) or 0.0) for row in rows]
        ax.bar([item + offset for item in x], values, width=width, label=label)
    ax.set_ylim(0.0, 1.05)
    ax.set_title(title)
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend(ncols=len(metrics), fontsize=9, loc="upper center")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_reward_components(rows: list[dict[str, Any]], output_path: Path, *, title: str) -> None:
    labels = [str(row.get("run_label") or "") for row in rows]
    keys = [
        ("comparison_reward_mean", "Reward"),
        ("comparison_local_reward_mean", "Local"),
        ("comparison_judge_score_mean", "Judge"),
    ]
    height = max(4.5, len(labels) * 1.0)
    fig, axes = plt.subplots(1, len(keys), figsize=(15.0, height), sharey=True, constrained_layout=True)
    if hasattr(axes, "ravel"):
        axes = list(axes.ravel())
    elif not isinstance(axes, list):
        axes = [axes]
    for ax, (key, label) in zip(axes, keys):
        values = [float(row.get(key, 0.0) or 0.0) for row in rows]
        ax.barh(labels, values, color="#3b82f6")
        ax.set_xlim(0.0, max(1.0, math.ceil(max(values or [1.0]) * 10.0) / 10.0))
        ax.set_title(label)
        ax.grid(axis="x", alpha=0.25)
        for index, value in enumerate(values):
            ax.text(min(value + 0.01, ax.get_xlim()[1] - 0.02), index, f"{value:.3f}", va="center", fontsize=9)
    fig.suptitle(title)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_summary(rows: list[dict[str, Any]], *, title: str) -> dict[str, Any]:
    return {
        "title": title,
        "benchmark_count": len(rows),
        "comparison_metric_mode": str(rows[0].get("comparison_metric_mode") or "primary") if rows else "primary",
        "comparison_metric_notes": list(rows[0].get("comparison_metric_notes") or []) if rows else [],
        "rows": rows,
        "best_by_reward_mean": rows[0] if rows else None,
    }


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    rows = _load_rows(args.input_jsons)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary(rows, title=str(args.title))
    common.write_json(args.output_dir / "summary.json", summary)
    _write_csv(rows, args.output_dir / "comparison_table.csv")
    _write_markdown(rows, args.output_dir / "comparison_table.md", title=str(args.title))
    if rows:
        _plot_core_metrics(rows, args.output_dir / "comparison_core_metrics.png", title=str(args.title))
        _plot_reward_components(rows, args.output_dir / "comparison_reward_components.png", title=str(args.title))
    print(f"saved summary -> {args.output_dir / 'summary.json'}")
    print(f"saved table -> {args.output_dir / 'comparison_table.md'}")
    if rows:
        print(f"saved chart -> {args.output_dir / 'comparison_core_metrics.png'}")
        print(f"saved chart -> {args.output_dir / 'comparison_reward_components.png'}")


if __name__ == "__main__":
    main()
