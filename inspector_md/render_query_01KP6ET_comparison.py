#!/usr/bin/env python3
"""Render slice-matched comparison visuals for query run 01KP6ET."""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import fmean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inspector_md.train_inspector_query import _load_split_examples, _score_answer_text


RUN_ID = "01KP6ET92P3Q9F2TB3XCTE8VE7"
RUN_DIR = Path("inspector_md/outputs/runs/inspector_query") / RUN_ID
DATASET_DIR = Path("inspector_md/outputs/inspector_query_finding_v1")
OUT_DIR = Path("inspector_md/outputs/benchmarks/query_01KP6ET_visual_comparison")

BASELINE_SFT_DIR = Path("inspector_md/outputs/benchmarks/query_01KP6ET_prod_baseline_seed42_n16")
BASELINE_RL_DIR = Path("inspector_md/outputs/benchmarks/query_01KP6ET_prod_baseline_seed82_n16")

BASELINE_COLOR = "#9ca3af"
SFT_COLOR = "#93c5fd"
RL_COLOR = "#2563eb"

_GENERIC_NO_ISSUE_MARKERS = (
    "no visible building or site issues",
    "no visible building issues",
    "no visible site issues",
    "no visible issues",
    "no issues visible",
    "no obvious issues",
    "no visible defects",
    "no defects visible",
    "no building or site issues",
)


def _example_index() -> dict[str, object]:
    examples = _load_split_examples(split_name="validation", dataset_dir=DATASET_DIR)
    return {str(example.row_id): example for example in examples}


def _looks_like_generic_no_issue(answer_text: str) -> bool:
    normalized = str(answer_text or "").strip().lower()
    return any(marker in normalized for marker in _GENERIC_NO_ISSUE_MARKERS)


def _prediction_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _rescore_prediction_path(path: Path) -> dict:
    by_row_id = _example_index()
    rows = _prediction_rows(path)
    issue_correct_values: list[float] = []
    issue_f1_values: list[float] = []
    task_correct_values: list[float] = []
    evidence_f1_values: list[float] = []
    parse_success_values: list[float] = []
    judge_values: list[float] = []
    word_counts: list[float] = []
    compact_values: list[float] = []
    generic_no_issue_values: list[float] = []
    parse_method_distribution: Counter[str] = Counter()

    for row in rows:
        example = by_row_id.get(str(row.get("row_id") or ""))
        if example is None:
            continue
        answer_text = str(row.get("answer_text") or "")
        score_outcome, parse_outcome = _score_answer_text(example, answer_text, grader=None)
        parse_method = str(parse_outcome.method or row.get("parse_method") or "").strip()
        parse_method_distribution[parse_method or "unparsed"] += 1
        issue_correct_values.append(1.0 if float(score_outcome.issue_f1) >= 0.999 else 0.0)
        issue_f1_values.append(float(score_outcome.issue_f1))
        task_correct_values.append(1.0 if score_outcome.task_correct else 0.0)
        evidence_f1_values.append(float(score_outcome.reasoning_f1))
        parse_success_values.append(1.0 if score_outcome.parse_success else 0.0)
        judge_values.append(float(row.get("judge_score", 0.0) or 0.0))
        word_counts.append(float(len(answer_text.split())))
        compact_values.append(1.0 if parse_method == "compact_text" else 0.0)
        generic_no_issue_values.append(1.0 if _looks_like_generic_no_issue(answer_text) else 0.0)

    return {
        "count": len(issue_correct_values),
        "issue_correct_rate": fmean(issue_correct_values) if issue_correct_values else 0.0,
        "issue_f1": fmean(issue_f1_values) if issue_f1_values else 0.0,
        "task_correct_rate": fmean(task_correct_values) if task_correct_values else 0.0,
        "evidence_f1": fmean(evidence_f1_values) if evidence_f1_values else 0.0,
        "parse_rate": fmean(parse_success_values) if parse_success_values else 0.0,
        "judge_score_mean": fmean(judge_values) if judge_values else 0.0,
        "mean_words": fmean(word_counts) if word_counts else 0.0,
        "response_compact_rate": fmean(compact_values) if compact_values else 0.0,
        "response_generic_no_issue_rate": fmean(generic_no_issue_values) if generic_no_issue_values else 0.0,
        "parse_method_distribution": dict(parse_method_distribution),
    }


def _row_ids(path: Path) -> set[str]:
    return {str(row.get("row_id") or "") for row in _prediction_rows(path)}


def _build_rows() -> list[dict]:
    baseline_sft_path = BASELINE_SFT_DIR / "predictions.jsonl"
    baseline_rl_path = BASELINE_RL_DIR / "predictions.jsonl"
    sft_path = RUN_DIR / "eval_predictions" / "final_validation.jsonl"
    rl_path = RUN_DIR / "eval_predictions" / "rl_step_0040_validation.jsonl"

    baseline_sft_metrics = _rescore_prediction_path(baseline_sft_path)
    baseline_rl_metrics = _rescore_prediction_path(baseline_rl_path)
    sft_metrics = _rescore_prediction_path(sft_path)
    rl_metrics = _rescore_prediction_path(rl_path)

    return [
        {
            "slice_id": "seed42_n16_matches_sft",
            "slice_label": "Seed 42 / n=16 (matches SFT final)",
            "slice_seed": 42,
            "label": "Baseline",
            "model_label": "moondream3-preview",
            "phase": "baseline",
            "checkpoint_step": None,
            "source_note": (
            "Live rerun on 2026-04-16 against https://api.moondream.ai/v1 using `MOONDREAM_API_KEY` "
            "and base `moondream3-preview`. This slice exactly matches the saved SFT final validation row IDs. "
            "Metrics are rescored from raw `answer_text`, not the target-conditioned normalized `prediction` field."
        ),
            "predictions_path": str(baseline_sft_path),
            **baseline_sft_metrics,
        },
        {
            "slice_id": "seed42_n16_matches_sft",
            "slice_label": "Seed 42 / n=16 (matches SFT final)",
            "slice_seed": 42,
            "label": "SFT",
            "model_label": f"{RUN_ID[:8]}@350",
            "phase": "sft_pre_rl",
            "checkpoint_step": 350,
            "source_note": (
            "Rescored from saved `final_validation.jsonl` predictions using the saved raw "
            "`answer_text` on 2026-04-16."
        ),
            "predictions_path": str(sft_path),
            **sft_metrics,
        },
        {
            "slice_id": "seed82_n16_matches_rl40",
            "slice_label": "Seed 82 / n=16 (matches RL step 0040)",
            "slice_seed": 82,
            "label": "Baseline",
            "model_label": "moondream3-preview",
            "phase": "baseline",
            "checkpoint_step": None,
            "source_note": (
            "Live rerun on 2026-04-16 against https://api.moondream.ai/v1 using `MOONDREAM_API_KEY` "
            "and base `moondream3-preview`. This slice exactly matches the saved RL step 0040 validation row IDs. "
            "Metrics are rescored from raw `answer_text`, not the target-conditioned normalized `prediction` field."
        ),
            "predictions_path": str(baseline_rl_path),
            **baseline_rl_metrics,
        },
        {
            "slice_id": "seed82_n16_matches_rl40",
            "slice_label": "Seed 82 / n=16 (matches RL step 0040)",
            "slice_seed": 82,
            "label": "RL After SFT",
            "model_label": f"{RUN_ID[:8]}@391",
            "phase": "rl_after_sft",
            "checkpoint_step": 391,
            "source_note": (
            "Rescored from saved `rl_step_0040_validation.jsonl` predictions using the saved raw "
            "`answer_text` on 2026-04-16."
        ),
            "predictions_path": str(rl_path),
            **rl_metrics,
        },
    ]


def _write_json(rows: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "comparison_rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def _write_csv(rows: list[dict]) -> None:
    fields = [
        "slice_id",
        "slice_label",
        "slice_seed",
        "label",
        "model_label",
        "phase",
        "checkpoint_step",
        "count",
        "issue_correct_rate",
        "issue_f1",
        "task_correct_rate",
        "evidence_f1",
        "judge_score_mean",
        "parse_rate",
        "response_compact_rate",
        "mean_words",
        "response_generic_no_issue_rate",
        "predictions_path",
        "source_note",
    ]
    with (OUT_DIR / "comparison_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_notes(rows: list[dict]) -> None:
    baseline_sft_path = BASELINE_SFT_DIR / "predictions.jsonl"
    baseline_rl_path = BASELINE_RL_DIR / "predictions.jsonl"
    sft_path = RUN_DIR / "eval_predictions" / "final_validation.jsonl"
    rl_path = RUN_DIR / "eval_predictions" / "rl_step_0040_validation.jsonl"
    notes = {
        "generated_at": "2026-04-16",
        "run_id": RUN_ID,
        "selected_best_checkpoint": 391,
        "selected_sft_pre_rl_checkpoint": 350,
        "comparison_mode": "slice_matched",
        "single_graph_mode": True,
        "baseline_kind": "live_production_moondream3_preview",
        "baseline_issue_correct_available": True,
        "full_set_requested": True,
        "full_set_validation_count": 9448,
        "full_set_three_way_available": False,
        "full_set_blockers": [
            {
                "model": "moondream3-preview/01KP6ET92P3Q9F2TB3XCTE8VE7@350",
                "base_url": "https://api-staging.moondream.ai/v1",
                "result": "timed_out",
            },
            {
                "model": "moondream3-preview/01KP6ET92P3Q9F2TB3XCTE8VE7@391",
                "base_url": "https://api-staging.moondream.ai/v1",
                "result": "timed_out",
            },
            {
                "model": "moondream3-preview/01KP6ET92P3Q9F2TB3XCTE8VE7@350",
                "base_url": "https://api.moondream.ai/v1",
                "result": "http_500_internal_server_error",
            },
            {
                "model": "moondream3-preview/01KP6ET92P3Q9F2TB3XCTE8VE7@391",
                "base_url": "https://api.moondream.ai/v1",
                "result": "http_500_internal_server_error",
            },
        ],
        "baseline_probes": [
            {
                "slice_id": "seed42_n16_matches_sft",
                "date": "2026-04-16",
                "base_url": "https://api.moondream.ai/v1",
                "api_key_env_var": "MOONDREAM_API_KEY",
                "model": "moondream3-preview",
                "seed": 42,
                "max_samples": 16,
                "result": "success",
            },
            {
                "slice_id": "seed82_n16_matches_rl40",
                "date": "2026-04-16",
                "base_url": "https://api.moondream.ai/v1",
                "api_key_env_var": "MOONDREAM_API_KEY",
                "model": "moondream3-preview",
                "seed": 82,
                "max_samples": 16,
                "result": "success",
            },
        ],
        "slice_matches": [
            {
                "slice_id": "seed42_n16_matches_sft",
                "baseline_predictions": str(baseline_sft_path),
                "model_predictions": str(sft_path),
                "row_overlap": len(_row_ids(baseline_sft_path) & _row_ids(sft_path)),
            },
            {
                "slice_id": "seed82_n16_matches_rl40",
                "baseline_predictions": str(baseline_rl_path),
                "model_predictions": str(rl_path),
                "row_overlap": len(_row_ids(baseline_rl_path) & _row_ids(rl_path)),
            },
        ],
        "artifacts": [
            "comparison_rows.json",
            "comparison_table.csv",
            "comparison_table.md",
            "grouped_performance_comparison.png",
            "task_correct_rate_comparison.png",
            "sft_to_rl_timeline.png",
        ],
        "rows": rows,
    }
    (OUT_DIR / "comparison_notes.json").write_text(json.dumps(notes, indent=2), encoding="utf-8")


def _write_markdown(rows: list[dict]) -> None:
    lines = [
        "# Query Comparison: Slice-Matched Baseline vs SFT/RL",
        "",
        "## Slice-Matched Setup",
        "- The saved `SFT` final and `RL step 0040` validation files are two different 16-row slices.",
        "- `Baseline` was rerun twice on production `moondream3-preview`: `seed=42` to match the SFT slice and `seed=82` to match the RL slice.",
        "- All rows are rescored from raw `answer_text` with no grader normalization. This avoids the target-conditioned `openrouter_normalize` leak on baseline finding rows.",
        "- A true full-set three-way rerun is currently blocked: the finetune checkpoints time out on staging and return `HTTP 500` on production, even for `n=1` probes.",
        "",
        "## Main Metrics",
        "- `Issue Label` = exact issue-code match rate on this single-finding task.",
        "- `Reasoning / Evidence` = token-F1 match on the supporting evidence/action text.",
        "- `Overall (Judge)` = saved holistic answer-quality score from the benchmark judge.",
        "- `Task Correct` = exact issue match plus near-perfect evidence match.",
        "",
        "## Table",
        "",
        "| Slice | Run | Count | Issue Label | Reasoning / Evidence | Task Correct | Overall (Judge) | Compact | Mean Words |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['slice_label']} | {row['label']} | {int(row['count'])} | "
            f"{float(row.get('issue_correct_rate', 0.0)):.4f} | "
            f"{float(row.get('evidence_f1', 0.0)):.4f} | "
            f"{float(row.get('task_correct_rate', 0.0)):.4f} | "
            f"{float(row.get('judge_score_mean', 0.0)):.4f} | "
            f"{float(row.get('response_compact_rate', 0.0)):.4f} | "
            f"{float(row.get('mean_words', 0.0)):.2f} |"
        )
    lines += [
        "",
        "## Source Notes",
    ]
    for row in rows:
        lines.append(f"- `{row['slice_label']} / {row['label']}`: {row['source_note']}")
    (OUT_DIR / "comparison_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_primary_metric(rows: list[dict]) -> None:
    colors = [BASELINE_COLOR, SFT_COLOR, BASELINE_COLOR, RL_COLOR]
    labels = [
        "Baseline\nseed 42",
        "SFT\nseed 42",
        "Baseline\nseed 82",
        "RL After SFT\nseed 82",
    ]
    values = [float(row.get("task_correct_rate", 0.0) or 0.0) for row in rows]

    fig, ax = plt.subplots(figsize=(9.4, 5.2), constrained_layout=True)
    bars = ax.bar(labels, values, color=colors, edgecolor="#334155")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Task Correct Rate")
    ax.set_title("Slice-Matched Task Correct Rate")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + (bar.get_width() / 2.0),
            value + 0.02,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    fig.savefig(OUT_DIR / "task_correct_rate_comparison.png", dpi=180)
    plt.close(fig)


def _plot_grouped_metrics(rows: list[dict]) -> None:
    metric_labels = ["Issue Label", "Reasoning / Evidence", "Overall (Judge)"]
    metric_keys = ["issue_correct_rate", "evidence_f1", "judge_score_mean"]
    metric_colors = ["#ef4444", "#f59e0b", "#2563eb"]
    width = 0.22
    x_positions = list(range(len(rows)))
    tick_labels = [
        "Baseline\nseed 42",
        "SFT\nseed 42",
        "Baseline\nseed 82",
        "RL After SFT\nseed 82",
    ]

    fig, ax = plt.subplots(figsize=(12.8, 6.2))
    for metric_index, (metric_label, metric_key) in enumerate(zip(metric_labels, metric_keys)):
        offset = (metric_index - 1) * width
        values = [float(row.get(metric_key, 0.0) or 0.0) for row in rows]
        bar_positions = [position + offset for position in x_positions]
        bars = ax.bar(
            bar_positions,
            values,
            width=width,
            color=metric_colors[metric_index],
            edgecolor="#334155",
            label=metric_label,
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + (bar.get_width() / 2.0),
                value + 0.02,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Query Metrics In One Graph")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper center", ncols=3, fontsize=9)
    fig.text(
        0.5,
        0.02,
        "Baseline reruns use production moondream3-preview. Seed 42 matches SFT; seed 82 matches RL step 0040. "
        "Full-set SFT/RL reruns are currently blocked by backend failures.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#475569",
    )
    fig.tight_layout(rect=(0.0, 0.07, 1.0, 0.92))
    fig.savefig(OUT_DIR / "grouped_performance_comparison.png", dpi=180)
    plt.close(fig)


def _timeline_points() -> list[tuple[str, float, float]]:
    rows: list[tuple[str, float, float]] = []
    for path in sorted((RUN_DIR / "eval_predictions").glob("sft_step_*_validation.jsonl")):
        match = re.search(r"sft_step_(\d+)_validation\.jsonl$", path.name)
        if match is None:
            continue
        metrics = _rescore_prediction_path(path)
        rows.append((f"SFT {int(match.group(1))}", float(metrics["task_correct_rate"]), float(metrics["evidence_f1"])))
    final_metrics = _rescore_prediction_path(RUN_DIR / "eval_predictions" / "final_validation.jsonl")
    rows.append(("SFT final", float(final_metrics["task_correct_rate"]), float(final_metrics["evidence_f1"])))
    for path in sorted((RUN_DIR / "eval_predictions").glob("rl_step_*_validation.jsonl")):
        match = re.search(r"rl_step_(\d+)_validation\.jsonl$", path.name)
        if match is None:
            continue
        metrics = _rescore_prediction_path(path)
        rows.append((f"RL {int(match.group(1))}", float(metrics["task_correct_rate"]), float(metrics["evidence_f1"])))
    return rows


def _plot_timeline() -> None:
    rows = _timeline_points()
    if not rows:
        return
    xs = list(range(len(rows)))
    labels = [label for label, _task_correct, _evidence in rows]
    task_values = [task_correct for _label, task_correct, _evidence in rows]
    evidence_values = [evidence for _label, _task_correct, evidence in rows]

    fig, ax = plt.subplots(figsize=(12, 5.6), constrained_layout=True)
    ax.plot(xs, task_values, marker="o", color=RL_COLOR, label="Overall Task")
    ax.plot(xs, evidence_values, marker="o", color=SFT_COLOR, label="Reasoning / Evidence")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Saved Eval Progression (rescored on 2026-04-16)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper center", ncols=2, fontsize=9)
    fig.savefig(OUT_DIR / "sft_to_rl_timeline.png", dpi=180)
    plt.close(fig)


def main() -> None:
    rows = _build_rows()
    _write_json(rows)
    _write_csv(rows)
    _write_notes(rows)
    _write_markdown(rows)
    _plot_primary_metric(rows)
    _plot_grouped_metrics(rows)
    _plot_timeline()
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
