#!/usr/bin/env python3
"""Analyze V2 benchmark outputs and generate best-move bucket reports."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tictaktoe_QA.task_schema import normalize_task_type  # noqa: E402
from tictaktoe_QA.v2_eval_suite import common  # noqa: E402

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "analysis_v2_default.json"
ANALYSIS_CONFIG_ALLOWED_KEYS = {
    "dataset_dir",
    "expected_hf_dataset_repo_id",
    "force_include_runs",
    "include_runs",
    "out_dir",
    "outputs_dir",
    "split",
    "v2_only",
}


@dataclass
class TaskStats:
    rows: int = 0
    request_errors: int = 0
    parse_success: int = 0
    correct: int = 0

    def non_error_rows(self) -> int:
        return max(0, self.rows - self.request_errors)

    def accuracy(self) -> float:
        denom = max(1, self.non_error_rows())
        return self.correct / denom

    def parse_rate(self) -> float:
        denom = max(1, self.non_error_rows())
        return self.parse_success / denom


@dataclass
class RunAnalysis:
    run_id: str
    prediction_path: Path
    model: str
    metrics_path: Optional[Path] = None
    metrics_payload: dict[str, Any] = field(default_factory=dict)
    bucket_counts: Counter[str] = field(default_factory=Counter)
    task_stats: dict[str, TaskStats] = field(
        default_factory=lambda: {task: TaskStats() for task in common.HARD_TASK_TYPES}
    )
    total_rows: int = 0
    best_move_rows: int = 0
    missing_gt_best_move_rows: int = 0

    def bucket_total(self) -> int:
        return sum(self.bucket_counts.get(bucket, 0) for bucket in common.BEST_MOVE_BUCKET_ORDER)

    def bucket_pct(self, bucket: str) -> float:
        total = max(1, self.bucket_total())
        return self.bucket_counts.get(bucket, 0) / total


def _resolve_config_path(raw_path: str) -> Path:
    return common.resolve_path(raw_path, search_roots=(common.repo_root(), common.package_root()))


def _load_json_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        if config_path == DEFAULT_CONFIG_PATH:
            return {}
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return common.load_json_object(config_path)


def _build_parser(config: dict[str, Any], config_path: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze V2 benchmark prediction outputs")
    parser.add_argument("--config", default=str(config_path))
    parser.add_argument("--outputs-dir", default=common.cfg_str(config, "outputs_dir", "tictaktoe_QA/outputs"))
    parser.add_argument(
        "--dataset-dir",
        default=common.cfg_str(config, "dataset_dir", "tictaktoe_QA/synth_dataset/outputs/v2"),
    )
    parser.add_argument(
        "--out-dir",
        default=common.cfg_str(config, "out_dir", "tictaktoe_QA/v2_eval_suite/outputs/analysis"),
    )
    parser.add_argument("--split", default=common.cfg_str(config, "split", "test"))
    parser.add_argument("--include-runs", default=common.cfg_str(config, "include_runs", ""))

    v2_group = parser.add_mutually_exclusive_group()
    v2_group.add_argument("--v2-only", dest="v2_only", action="store_true")
    v2_group.add_argument("--no-v2-only", dest="v2_only", action="store_false")
    parser.set_defaults(v2_only=common.cfg_bool(config, "v2_only", True))
    return parser


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    pre_args, _ = pre_parser.parse_known_args(argv)

    config_path = _resolve_config_path(pre_args.config)
    config = _load_json_config(config_path)
    common.validate_config_keys(config, allowed_keys=ANALYSIS_CONFIG_ALLOWED_KEYS, config_path=config_path)

    parser = _build_parser(config, config_path)
    args = parser.parse_args(argv)

    args.config = str(_resolve_config_path(args.config))
    args.outputs_dir = common.resolve_path(args.outputs_dir, search_roots=(common.repo_root(),))
    args.dataset_dir = common.resolve_path(args.dataset_dir, search_roots=(common.repo_root(),))
    args.out_dir = common.resolve_path(args.out_dir, search_roots=(common.repo_root(),))
    args.include_patterns = common.parse_pattern_list(args.include_runs)
    args.force_include_patterns = common.cfg_list_str(config, "force_include_runs", [])
    args.expected_hf_dataset_repo_id = common.cfg_str(
        config,
        "expected_hf_dataset_repo_id",
        common.V2_HF_DATASET_REPO_ID,
    )
    return args


def _discover_prediction_files(outputs_dir: Path) -> list[Path]:
    return sorted(path for path in outputs_dir.glob("*_predictions.jsonl") if path.is_file())


def _load_metrics_payload(metrics_path: Optional[Path]) -> dict[str, Any]:
    if metrics_path is None or not metrics_path.exists():
        return {}
    text = metrics_path.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    payload = json.loads(text)
    if not isinstance(payload, dict):
        return {}
    return payload


def _find_metrics_path(prediction_path: Path, run_id: str) -> Optional[Path]:
    for ext in (".json", ".jsonl"):
        candidate = prediction_path.parent / f"{run_id}{ext}"
        if candidate.exists() and candidate != prediction_path:
            return candidate
    return None


def _analyze_prediction_file(
    prediction_path: Path,
    *,
    run_id: str,
    model: str,
    metrics_path: Optional[Path],
    metrics_payload: dict[str, Any],
    rows_by_id: dict[str, dict[str, Any]],
) -> RunAnalysis:
    out = RunAnalysis(
        run_id=run_id,
        prediction_path=prediction_path,
        model=model,
        metrics_path=metrics_path,
        metrics_payload=metrics_payload,
    )

    for record in common.iter_jsonl(prediction_path):
        out.total_rows += 1
        task_type = normalize_task_type(str(record.get("task_type", "")), allow_unknown=True)

        if task_type in out.task_stats:
            stat = out.task_stats[task_type]
            stat.rows += 1
            if common.is_request_error_record(record):
                stat.request_errors += 1
            else:
                if bool(record.get("parse_success", False)):
                    stat.parse_success += 1
                if bool(record.get("task_correct", False)):
                    stat.correct += 1

        if task_type != "best_move":
            continue

        out.best_move_rows += 1
        if common.is_request_error_record(record):
            out.bucket_counts["request_error"] += 1
            continue

        row_id = str(record.get("row_id", "")).strip()
        gt_row = rows_by_id.get(row_id)
        if gt_row is None:
            out.missing_gt_best_move_rows += 1
            continue

        bucket = common.classify_best_move_prediction(record, ground_truth_row=gt_row)
        out.bucket_counts[bucket] += 1

    return out


def _plot_bucket_distribution(runs: list[RunAnalysis], out_path: Path, *, as_percentage: bool) -> None:
    fig_width = max(10.0, float(len(runs)) * 1.8)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    labels = [run.run_id for run in runs]
    positions = list(range(len(runs)))
    bottom = [0.0 for _ in runs]
    colors = {
        "best_move": "#2E7D32",
        "second_best": "#66BB6A",
        "third_best": "#FFA726",
        "fourth_plus": "#EF5350",
        "invalid_move": "#8D6E63",
        "improper_response_format": "#546E7A",
        "request_error": "#5E35B1",
    }

    for bucket in common.BEST_MOVE_BUCKET_ORDER:
        values: list[float] = []
        for run in runs:
            count = float(run.bucket_counts.get(bucket, 0))
            if as_percentage:
                total = max(1, run.bucket_total())
                values.append((count / total) * 100.0)
            else:
                values.append(count)

        ax.bar(
            positions,
            values,
            bottom=bottom,
            label=bucket,
            color=colors.get(bucket),
            edgecolor="white",
            linewidth=0.5,
        )
        bottom = [b + v for b, v in zip(bottom, values)]

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_title("Best-Move Bucket Distribution")
    ax.set_ylabel("Percent of best_move rows" if as_percentage else "Rows")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.6)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _plot_hard_task_accuracy(runs: list[RunAnalysis], out_path: Path) -> None:
    fig_width = max(10.0, float(len(runs)) * 1.8)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    positions = list(range(len(runs)))
    labels = [run.run_id for run in runs]
    tasks = ["best_move", "available_moves_count", "available_moves_list"]
    colors = {
        "best_move": "#2E7D32",
        "available_moves_count": "#1E88E5",
        "available_moves_list": "#F57C00",
    }
    width = 0.24

    for idx, task in enumerate(tasks):
        offset = (idx - 1) * width
        task_positions = [pos + offset for pos in positions]
        values = [run.task_stats[task].accuracy() for run in runs]
        ax.bar(task_positions, values, width=width, label=task, color=colors.get(task))

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("Hard-Task Accuracy by Run/Model")
    ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _write_report(
    out_path: Path,
    *,
    analyzed_runs: list[RunAnalysis],
    excluded_runs: list[tuple[str, str]],
    args: argparse.Namespace,
) -> None:
    lines: list[str] = []
    lines.append("# V2 Benchmark Analysis Report")
    lines.append("")
    lines.append(f"- generated_utc: {common.utc_timestamp_tag()}")
    lines.append(f"- config: `{args.config}`")
    lines.append(f"- split: `{args.split}`")
    lines.append(f"- dataset_dir: `{args.dataset_dir}`")
    lines.append(f"- outputs_dir: `{args.outputs_dir}`")
    lines.append(f"- v2_only: `{args.v2_only}`")
    lines.append(f"- analyzed_runs: `{len(analyzed_runs)}`")
    lines.append(f"- excluded_runs: `{len(excluded_runs)}`")
    lines.append("")

    if analyzed_runs:
        top_run = max(analyzed_runs, key=lambda run: run.bucket_pct("best_move"))
        lines.append("## Key Finding")
        lines.append("")
        lines.append(
            f"Top best-move bucket rate: `{top_run.run_id}` "
            f"({top_run.bucket_pct('best_move') * 100.0:.2f}% best_move over bucketed rows)."
        )
        lines.append("")

    if excluded_runs:
        lines.append("## Excluded Runs")
        lines.append("")
        for run_id, reason in excluded_runs:
            lines.append(f"- `{run_id}`: {reason}")
        lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- best_move buckets use dense ranking from `scores_by_move_json` with training rank semantics.")
    lines.append("- rows with missing V2 ground truth are tracked and excluded from bucket percentages.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)

    rows_by_id = common.load_rows_by_id(dataset_dir=args.dataset_dir, split=args.split)
    prediction_files = _discover_prediction_files(args.outputs_dir)

    analyzed_runs: list[RunAnalysis] = []
    excluded_runs: list[tuple[str, str]] = []

    for prediction_path in prediction_files:
        run_id = prediction_path.name.removesuffix("_predictions.jsonl")

        if not common.matches_any_pattern(run_id, args.include_patterns) and not common.matches_any_pattern(
            prediction_path.as_posix(), args.include_patterns
        ):
            continue

        metrics_path = _find_metrics_path(prediction_path, run_id)
        metrics_payload = _load_metrics_payload(metrics_path)
        model = str(metrics_payload.get("model", "")).strip() or run_id

        force_included = common.matches_any_pattern(run_id, args.force_include_patterns) or common.matches_any_pattern(
            prediction_path.as_posix(), args.force_include_patterns
        )
        is_v2 = common.is_v2_metrics_payload(
            metrics_payload,
            dataset_dir=args.dataset_dir,
            expected_hf_repo_id=args.expected_hf_dataset_repo_id,
        )

        if args.v2_only and not (is_v2 or force_included):
            excluded_runs.append((run_id, "non_v2_run"))
            continue

        analyzed_runs.append(
            _analyze_prediction_file(
                prediction_path,
                run_id=run_id,
                model=model,
                metrics_path=metrics_path,
                metrics_payload=metrics_payload,
                rows_by_id=rows_by_id,
            )
        )

    if not analyzed_runs:
        raise ValueError(
            "No benchmark prediction files selected for analysis. "
            "Check --outputs-dir / --include-runs / --v2-only filters."
        )

    run_out_dir = args.out_dir / common.utc_timestamp_tag()
    run_out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    bucket_rows: list[dict[str, Any]] = []

    for run in analyzed_runs:
        metrics = run.metrics_payload
        summary_rows.append(
            {
                "run_id": run.run_id,
                "model": run.model,
                "prediction_file": run.prediction_path.name,
                "metrics_file": run.metrics_path.name if run.metrics_path else "",
                "eval_reward_mean": float(metrics.get("eval_reward_mean", 0.0)),
                "eval_json_parse_rate": float(metrics.get("eval_json_parse_rate", 0.0)),
                "total_prediction_rows": run.total_rows,
                "best_move_rows": run.best_move_rows,
                "best_move_missing_gt_rows": run.missing_gt_best_move_rows,
                "best_move_set_accuracy": run.task_stats["best_move"].accuracy(),
                "available_moves_count_accuracy": run.task_stats["available_moves_count"].accuracy(),
                "available_moves_list_accuracy": run.task_stats["available_moves_list"].accuracy(),
                "best_move_parse_rate": run.task_stats["best_move"].parse_rate(),
                "available_moves_count_parse_rate": run.task_stats["available_moves_count"].parse_rate(),
                "available_moves_list_parse_rate": run.task_stats["available_moves_list"].parse_rate(),
            }
        )

        bucket_row: dict[str, Any] = {
            "run_id": run.run_id,
            "model": run.model,
            "bucketed_best_move_rows": run.bucket_total(),
            "best_move_rows": run.best_move_rows,
            "missing_gt_best_move_rows": run.missing_gt_best_move_rows,
        }
        for bucket in common.BEST_MOVE_BUCKET_ORDER:
            count = int(run.bucket_counts.get(bucket, 0))
            bucket_row[f"{bucket}_count"] = count
            bucket_row[f"{bucket}_pct"] = run.bucket_pct(bucket)
        bucket_rows.append(bucket_row)

    common.write_csv(
        run_out_dir / "summary_table.csv",
        fieldnames=list(summary_rows[0].keys()),
        rows=summary_rows,
    )
    common.write_csv(
        run_out_dir / "best_move_bucket_table.csv",
        fieldnames=list(bucket_rows[0].keys()),
        rows=bucket_rows,
    )

    _plot_bucket_distribution(
        analyzed_runs,
        run_out_dir / "best_move_bucket_distribution_pct.png",
        as_percentage=True,
    )
    _plot_bucket_distribution(
        analyzed_runs,
        run_out_dir / "best_move_bucket_distribution_counts.png",
        as_percentage=False,
    )
    _plot_hard_task_accuracy(analyzed_runs, run_out_dir / "hard_task_accuracy_by_model.png")
    _write_report(
        run_out_dir / "analysis_report.md",
        analyzed_runs=analyzed_runs,
        excluded_runs=excluded_runs,
        args=args,
    )

    print(f"analyzed runs: {len(analyzed_runs)}")
    print(f"excluded runs: {len(excluded_runs)}")
    print(f"wrote analysis artifacts: {run_out_dir}")


if __name__ == "__main__":
    main()
