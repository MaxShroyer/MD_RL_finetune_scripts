#!/usr/bin/env python3
"""Aggregate Inspector MD benchmark outputs into JSON and Markdown summaries."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import fmean
from typing import Any, Optional

from inspector_md import common

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_JSON_PATH = common.repo_relative("outputs", "benchmarks", "inspector_pipeline.report.json")
DEFAULT_MD_PATH = common.repo_relative("outputs", "benchmarks", "inspector_pipeline.report.md")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Inspector MD benchmark summaries.")
    parser.add_argument("--input-jsons", nargs="+", default=[str(common.repo_relative("outputs", "benchmarks", "inspector_pipeline.metrics.json"))])
    parser.add_argument("--output-json", default=str(DEFAULT_JSON_PATH))
    parser.add_argument("--output-md", default=str(DEFAULT_MD_PATH))
    args = parser.parse_args(argv)
    args.input_jsons = [common.resolve_path(path, module_root=SCRIPT_DIR) for path in list(args.input_jsons)]
    args.output_json = common.resolve_path(args.output_json, module_root=SCRIPT_DIR)
    args.output_md = common.resolve_path(args.output_md, module_root=SCRIPT_DIR)
    return args


def _load_metrics(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"benchmark_count": 0}
    keys = [
        "proposal_precision",
        "proposal_recall",
        "proposal_f1",
        "localization_precision",
        "localization_recall",
        "localization_f1",
        "localization_precision_iou_0_1",
        "localization_recall_iou_0_1",
        "localization_f1_iou_0_1",
        "localization_precision_center_hit",
        "localization_recall_center_hit",
        "localization_f1_center_hit",
        "mean_best_iou",
        "finding_schema_valid_rate",
        "end_to_end_score",
    ]
    return {
        "benchmark_count": len(rows),
        "summary": {
            key: fmean(float(row.get(key, 0.0)) for row in rows)
            for key in keys
        },
        "sources": [str(path) for path in []],
        "raw_rows": rows,
    }


def _render_markdown(summary: dict[str, Any]) -> str:
    if int(summary.get("benchmark_count", 0)) <= 0:
        return "# Inspector MD Benchmark Report\n\nNo benchmark metrics were found.\n"
    metrics = summary.get("summary") or {}
    lines = [
        "# Inspector MD Benchmark Report",
        "",
        "## Proposal Quality",
        f"- Proposal precision: `{float(metrics.get('proposal_precision', 0.0)):.4f}`",
        f"- Proposal recall: `{float(metrics.get('proposal_recall', 0.0)):.4f}`",
        f"- Proposal F1: `{float(metrics.get('proposal_f1', 0.0)):.4f}`",
        "",
        "## Localization Quality",
        f"- Localization precision: `{float(metrics.get('localization_precision', 0.0)):.4f}`",
        f"- Localization recall: `{float(metrics.get('localization_recall', 0.0)):.4f}`",
        f"- Localization F1: `{float(metrics.get('localization_f1', 0.0)):.4f}`",
        f"- Localization precision @ IoU 0.1: `{float(metrics.get('localization_precision_iou_0_1', 0.0)):.4f}`",
        f"- Localization recall @ IoU 0.1: `{float(metrics.get('localization_recall_iou_0_1', 0.0)):.4f}`",
        f"- Localization F1 @ IoU 0.1: `{float(metrics.get('localization_f1_iou_0_1', 0.0)):.4f}`",
        f"- Localization precision center-hit: `{float(metrics.get('localization_precision_center_hit', 0.0)):.4f}`",
        f"- Localization recall center-hit: `{float(metrics.get('localization_recall_center_hit', 0.0)):.4f}`",
        f"- Localization F1 center-hit: `{float(metrics.get('localization_f1_center_hit', 0.0)):.4f}`",
        f"- Mean best IoU: `{float(metrics.get('mean_best_iou', 0.0)):.4f}`",
        "",
        "## Finding Schema Quality",
        f"- Finding schema valid rate: `{float(metrics.get('finding_schema_valid_rate', 0.0)):.4f}`",
        "",
        "## End-to-End Punch List Quality",
        f"- End-to-end score: `{float(metrics.get('end_to_end_score', 0.0)):.4f}`",
        "",
    ]
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    rows = _load_metrics(args.input_jsons)
    summary = _aggregate(rows)
    common.write_json(Path(args.output_json), summary)
    common.write_text(Path(args.output_md), _render_markdown(summary))
    print(f"saved report json: {args.output_json}")
    print(f"saved report md: {args.output_md}")


if __name__ == "__main__":
    main()
