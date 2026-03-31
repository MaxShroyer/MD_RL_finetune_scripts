#!/usr/bin/env python3
"""Build a shared aerial advertising subset from explicit point/detect finetunes."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aerial_airport import benchmark_aerial_airport_detect as detect_benchmark_mod
from aerial_airport import benchmark_aerial_airport_point as point_benchmark_mod
from aerial_airport.common import DEFAULT_STAGING_API_BASE

import curate_bone_aerial_advertising_subsets as shared

DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "advertising_subsets" / "aerial_manual_top100"
DEFAULT_POINT_MODEL = "01KMPQNDSKG7PSMMPWDYKDRMF0@139"
DEFAULT_DETECT_MODEL = "01KMPQSVPRYBZRQV8GEQZBVVEJ@250"


@dataclass(frozen=True)
class ManualToolSpec:
    tool_id: str
    label: str
    branch_spec: shared.BranchSpec
    benchmark_main: Callable[[Optional[list[str]]], None]
    finetune_id: str
    checkpoint_step: int

    @property
    def model_ref(self) -> str:
        return f"{self.finetune_id}@{self.checkpoint_step}"

    @property
    def run_id(self) -> str:
        return f"{self.tool_id}_{self.finetune_id[:8]}_{self.checkpoint_step}"


def _parse_model_ref(raw_value: str, *, option_name: str) -> tuple[str, int]:
    text = str(raw_value or "").strip().strip('"').strip("'")
    finetune_id, sep, step_text = text.partition("@")
    if not sep or not finetune_id.strip() or not step_text.strip():
        raise ValueError(f"{option_name} must look like <finetune_id>@<checkpoint_step>")
    try:
        checkpoint_step = int(step_text.strip())
    except ValueError as exc:
        raise ValueError(f"{option_name} checkpoint step must be an integer: {raw_value}") from exc
    if checkpoint_step < 0:
        raise ValueError(f"{option_name} checkpoint step must be >= 0: {raw_value}")
    return finetune_id.strip(), checkpoint_step


def _manual_tool_specs(args: argparse.Namespace) -> list[ManualToolSpec]:
    branch_specs = shared._branch_specs()
    point_id, point_step = _parse_model_ref(args.point_model, option_name="--point-model")
    detect_id, detect_step = _parse_model_ref(args.detect_model, option_name="--detect-model")
    return [
        ManualToolSpec(
            tool_id="point",
            label="Aerial Point",
            branch_spec=branch_specs["aerial_point"],
            benchmark_main=point_benchmark_mod.main,
            finetune_id=point_id,
            checkpoint_step=point_step,
        ),
        ManualToolSpec(
            tool_id="detect",
            label="Aerial Detect",
            branch_spec=branch_specs["aerial_detect"],
            benchmark_main=detect_benchmark_mod.main,
            finetune_id=detect_id,
            checkpoint_step=detect_step,
        ),
    ]


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


def _tool_benchmark_args(
    *,
    tool: ManualToolSpec,
    out_json: Path,
    records_jsonl: Path,
    dataset_name: str,
    dataset_path: str,
    split: str,
    finetune_id: str,
    checkpoint_step: Optional[int],
    skip_baseline: bool,
    max_samples: Optional[int],
    viz_samples: int,
) -> list[str]:
    argv = [
        "--config",
        str(tool.branch_spec.benchmark_config),
        "--api-base",
        DEFAULT_STAGING_API_BASE,
        "--dataset-name",
        dataset_name,
        "--dataset-path",
        dataset_path,
        "--split",
        split,
        "--out-json",
        str(out_json),
        "--records-jsonl",
        str(records_jsonl),
        "--run-id",
        tool.run_id if finetune_id else "",
        "--model",
        "",
        "--finetune-id",
        "",
    ]
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


def _run_tool_benchmark(
    *,
    tool: ManualToolSpec,
    out_json: Path,
    records_jsonl: Path,
    dataset_name: str,
    dataset_path: str,
    split: str,
    finetune_id: str,
    checkpoint_step: Optional[int],
    skip_baseline: bool,
    max_samples: Optional[int],
    viz_samples: int,
) -> dict[str, Any]:
    argv = _tool_benchmark_args(
        tool=tool,
        out_json=out_json,
        records_jsonl=records_jsonl,
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        split=split,
        finetune_id=finetune_id,
        checkpoint_step=checkpoint_step,
        skip_baseline=skip_baseline,
        max_samples=max_samples,
        viz_samples=viz_samples,
    )
    tool.benchmark_main(argv)
    return _load_json(out_json)


def combine_tool_sample_summaries(
    tool_summaries: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    by_tool: dict[str, dict[str, dict[str, Any]]] = {
        tool_id: {str(row["sample_id"]): row for row in rows}
        for tool_id, rows in tool_summaries.items()
    }
    all_sample_ids = sorted({sample_id for rows in by_tool.values() for sample_id in rows})
    combined_rows: list[dict[str, Any]] = []
    for sample_id in all_sample_ids:
        row: dict[str, Any] = {"sample_id": sample_id}
        score_values: list[float] = []
        f1_values: list[float] = []
        miou_values: list[float] = []
        winning_tool_id = ""
        winning_tool_delta = float("-inf")
        for tool_id in sorted(by_tool):
            sample = by_tool[tool_id].get(sample_id)
            prefix = f"{tool_id}_"
            if sample is None:
                row[f"{prefix}matched_tasks"] = 0
                row[f"{prefix}delta_f1"] = None
                row[f"{prefix}delta_miou"] = None
                row[f"{prefix}delta_score"] = None
                continue
            matched_tasks = int(sample["matched_tasks"])
            delta_f1 = float(sample["delta_f1"])
            delta_miou = float(sample["delta_miou"])
            delta_score = float(sample["delta_score"])
            row[f"{prefix}matched_tasks"] = matched_tasks
            row[f"{prefix}delta_f1"] = delta_f1
            row[f"{prefix}delta_miou"] = delta_miou
            row[f"{prefix}delta_score"] = delta_score
            score_values.append(delta_score)
            f1_values.append(delta_f1)
            miou_values.append(delta_miou)
            if delta_score > winning_tool_delta:
                winning_tool_delta = delta_score
                winning_tool_id = tool_id
        if not score_values:
            continue
        tool_count = len(score_values)
        row["tool_count"] = tool_count
        row["combined_delta_f1"] = sum(f1_values) / tool_count
        row["combined_delta_miou"] = sum(miou_values) / tool_count
        row["combined_delta_score"] = sum(score_values) / tool_count
        row["winning_tool_id"] = winning_tool_id or None
        row["winning_tool_delta_score"] = winning_tool_delta if winning_tool_id else None
        combined_rows.append(row)
    combined_rows.sort(
        key=lambda item: (float(item["combined_delta_score"]), str(item["sample_id"])),
        reverse=True,
    )
    return combined_rows


def _has_tool_regression(row: dict[str, Any]) -> bool:
    for key, value in row.items():
        if not key.endswith("_delta_score") or key == "combined_delta_score" or key == "winning_tool_delta_score":
            continue
        if value is not None and float(value) < 0.0:
            return True
    return False


def select_top_samples(
    combined_rows: list[dict[str, Any]],
    *,
    target_count: Optional[int],
    allow_tool_regressions: bool = False,
) -> list[dict[str, Any]]:
    ranked = [row for row in combined_rows if allow_tool_regressions or not _has_tool_regression(row)]
    ranked.sort(
        key=lambda item: (float(item.get("combined_delta_score") or 0.0), str(item.get("sample_id") or "")),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    limit = None if target_count is None else max(0, int(target_count))
    for row in ranked:
        if float(row["combined_delta_score"]) <= 0.0:
            continue
        selected.append({**row, "selection_reason": "positive_delta"})
        if limit is not None and len(selected) >= limit:
            break
    if limit is not None and len(selected) < limit:
        selected_ids = {str(row["sample_id"]) for row in selected}
        for row in ranked:
            sample_id = str(row["sample_id"])
            if sample_id in selected_ids:
                continue
            selected.append({**row, "selection_reason": "ranked_backfill"})
            selected_ids.add(sample_id)
            if len(selected) >= limit:
                break
    for index, row in enumerate(selected, start=1):
        row["selection_rank"] = index
    return selected


def _tool_markdown(
    *,
    tool: ManualToolSpec,
    full_baseline: dict[str, Any],
    full_candidate: dict[str, Any],
    subset_baseline: dict[str, Any],
    subset_candidate: dict[str, Any],
) -> list[str]:
    return [
        f"## {tool.label}",
        f"- finetune: `{tool.model_ref}`",
        (
            f"- selection-source before/after: "
            f"baseline f1={float(full_baseline.get('eval_f1') or 0.0):.6f}, "
            f"miou={float(full_baseline.get('eval_miou') or 0.0):.6f} -> "
            f"candidate f1={float(full_candidate.get('eval_f1') or 0.0):.6f}, "
            f"miou={float(full_candidate.get('eval_miou') or 0.0):.6f}"
        ),
        (
            f"- subset before/after: "
            f"baseline f1={float(subset_baseline.get('eval_f1') or 0.0):.6f}, "
            f"miou={float(subset_baseline.get('eval_miou') or 0.0):.6f} -> "
            f"candidate f1={float(subset_candidate.get('eval_f1') or 0.0):.6f}, "
            f"miou={float(subset_candidate.get('eval_miou') or 0.0):.6f}"
        ),
    ]


def _report_markdown(summary: dict[str, Any], tools: list[ManualToolSpec]) -> str:
    lines = [
        "# Aerial Manual Top-100",
        f"- staging api base: `{summary['staging_api_base']}`",
        f"- source split: `{summary['selection_split']}`",
        f"- target selected samples: {summary['target_selected_samples']}",
        f"- selected sample count: {summary['selected_sample_count']}",
        f"- subset dataset path: `{summary['subset_dataset']['dataset_path']}`",
    ]
    for tool in tools:
        tool_summary = summary["tools"][tool.tool_id]
        lines.extend(
            _tool_markdown(
                tool=tool,
                full_baseline=tool_summary["selection_source_baseline"],
                full_candidate=tool_summary["selection_source_candidate"],
                subset_baseline=tool_summary["subset_baseline"],
                subset_candidate=tool_summary["subset_candidate"],
            )
        )
    return "\n".join(lines) + "\n"


def process_manual_aerial_top100(
    *,
    tools: list[ManualToolSpec],
    output_root: Path,
    selection_split: str,
    target_selected_samples: int,
    max_samples: Optional[int],
    viz_samples: int,
    allow_tool_regressions: bool = False,
) -> dict[str, Any]:
    if not tools:
        raise ValueError("At least one tool is required")
    reference_spec = tools[0].branch_spec
    selection_source = shared._materialize_selection_source(
        spec=reference_spec,
        split=selection_split,
        output_dir=output_root / "selection_source",
    )

    tool_reports: dict[str, dict[str, Any]] = {}
    tool_sample_summaries: dict[str, list[dict[str, Any]]] = {}

    for tool in tools:
        tool_dir = output_root / tool.tool_id
        full_dir = tool_dir / "selection_source"
        baseline_metrics_path = full_dir / "baseline.metrics.json"
        baseline_records_path = full_dir / "baseline.records.jsonl"
        candidate_metrics_path = full_dir / "candidate.metrics.json"
        candidate_records_path = full_dir / "candidate.records.jsonl"
        sample_deltas_path = full_dir / "sample_deltas.jsonl"

        print(f"[{tool.tool_id}] running selection-source baseline benchmark on staging...")
        baseline_metrics = _run_tool_benchmark(
            tool=tool,
            out_json=baseline_metrics_path,
            records_jsonl=baseline_records_path,
            dataset_name=selection_source["dataset_name"],
            dataset_path=selection_source["dataset_path"],
            split=selection_source["benchmark_split"],
            finetune_id="",
            checkpoint_step=None,
            skip_baseline=False,
            max_samples=max_samples,
            viz_samples=viz_samples,
        )
        print(f"[{tool.tool_id}] running selection-source candidate benchmark {tool.model_ref} on staging...")
        candidate_metrics = _run_tool_benchmark(
            tool=tool,
            out_json=candidate_metrics_path,
            records_jsonl=candidate_records_path,
            dataset_name=selection_source["dataset_name"],
            dataset_path=selection_source["dataset_path"],
            split=selection_source["benchmark_split"],
            finetune_id=tool.finetune_id,
            checkpoint_step=tool.checkpoint_step,
            skip_baseline=True,
            max_samples=max_samples,
            viz_samples=viz_samples,
        )
        baseline_records = shared._load_jsonl(baseline_records_path)
        candidate_records = shared._load_jsonl(candidate_records_path)
        sample_summaries = shared.aggregate_sample_deltas(
            baseline_records=baseline_records,
            candidate_records=candidate_records,
        )
        _write_jsonl(sample_deltas_path, sample_summaries)
        tool_sample_summaries[tool.tool_id] = sample_summaries
        tool_reports[tool.tool_id] = {
            "tool_id": tool.tool_id,
            "label": tool.label,
            "finetune_id": tool.finetune_id,
            "checkpoint_step": tool.checkpoint_step,
            "selection_source_baseline": baseline_metrics,
            "selection_source_candidate": candidate_metrics,
            "selection_source_baseline_records": str(baseline_records_path),
            "selection_source_candidate_records": str(candidate_records_path),
            "sample_deltas_path": str(sample_deltas_path),
        }

    combined_rows = combine_tool_sample_summaries(tool_sample_summaries)
    combined_deltas_path = output_root / "combined_sample_deltas.jsonl"
    _write_jsonl(combined_deltas_path, combined_rows)

    selected_samples = select_top_samples(
        combined_rows,
        target_count=target_selected_samples,
        allow_tool_regressions=allow_tool_regressions,
    )
    selected_manifest_path = output_root / "selected_samples.jsonl"
    _write_jsonl(selected_manifest_path, selected_samples)

    subset_dataset_path = output_root / "subset" / "dataset"
    subset_info = shared.save_subset_dataset(
        spec=reference_spec,
        selected_samples=selected_samples,
        output_dir=subset_dataset_path,
        source_split=selection_source["benchmark_split"],
        source_dataset_path=(Path(selection_source["dataset_path"]) if selection_source["dataset_path"] else None),
    )

    if subset_info["selected_count"] == 0:
        summary = {
            "dataset_name": reference_spec.dataset_name,
            "staging_api_base": DEFAULT_STAGING_API_BASE,
            "selection_split": selection_split,
            "target_selected_samples": target_selected_samples,
            "selected_sample_count": len(selected_samples),
            "selected_samples_manifest": str(selected_manifest_path),
            "combined_sample_deltas_path": str(combined_deltas_path),
            "subset_dataset": subset_info,
            "tools": tool_reports,
        }
        report_json_path = output_root / "report.json"
        report_md_path = output_root / "report.md"
        _write_json(report_json_path, summary)
        report_md_path.write_text(_report_markdown(summary, tools), encoding="utf-8")
        print(f"saved report -> {report_json_path}")
        return summary

    for tool in tools:
        subset_dir = output_root / tool.tool_id / "subset"
        subset_baseline_path = subset_dir / "baseline.metrics.json"
        subset_baseline_records_path = subset_dir / "baseline.records.jsonl"
        subset_candidate_path = subset_dir / "candidate.metrics.json"
        subset_candidate_records_path = subset_dir / "candidate.records.jsonl"
        print(f"[{tool.tool_id}] running subset baseline benchmark on staging...")
        subset_baseline = _run_tool_benchmark(
            tool=tool,
            out_json=subset_baseline_path,
            records_jsonl=subset_baseline_records_path,
            dataset_name="",
            dataset_path=str(subset_dataset_path),
            split=reference_spec.split,
            finetune_id="",
            checkpoint_step=None,
            skip_baseline=False,
            max_samples=max_samples,
            viz_samples=viz_samples,
        )
        print(f"[{tool.tool_id}] running subset candidate benchmark {tool.model_ref} on staging...")
        subset_candidate = _run_tool_benchmark(
            tool=tool,
            out_json=subset_candidate_path,
            records_jsonl=subset_candidate_records_path,
            dataset_name="",
            dataset_path=str(subset_dataset_path),
            split=reference_spec.split,
            finetune_id=tool.finetune_id,
            checkpoint_step=tool.checkpoint_step,
            skip_baseline=True,
            max_samples=max_samples,
            viz_samples=viz_samples,
        )
        tool_reports[tool.tool_id]["subset_baseline"] = subset_baseline
        tool_reports[tool.tool_id]["subset_candidate"] = subset_candidate
        tool_reports[tool.tool_id]["subset_baseline_records"] = str(subset_baseline_records_path)
        tool_reports[tool.tool_id]["subset_candidate_records"] = str(subset_candidate_records_path)

    summary = {
        "dataset_name": reference_spec.dataset_name,
        "staging_api_base": DEFAULT_STAGING_API_BASE,
        "selection_split": selection_split,
        "target_selected_samples": target_selected_samples,
        "selected_sample_count": len(selected_samples),
        "selected_samples_manifest": str(selected_manifest_path),
        "combined_sample_deltas_path": str(combined_deltas_path),
        "subset_dataset": subset_info,
        "tools": tool_reports,
    }
    report_json_path = output_root / "report.json"
    report_md_path = output_root / "report.md"
    _write_json(report_json_path, summary)
    report_md_path.write_text(_report_markdown(summary, tools), encoding="utf-8")
    print(f"saved report -> {report_json_path}")
    return summary


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a shared aerial top-N advertising subset from explicit point/detect finetunes."
    )
    parser.add_argument("--point-model", default=DEFAULT_POINT_MODEL, help="Point finetune in <id>@<step> form.")
    parser.add_argument("--detect-model", default=DEFAULT_DETECT_MODEL, help="Detect finetune in <id>@<step> form.")
    parser.add_argument("--selection-split", default="train+validation+test")
    parser.add_argument("--target-selected-samples", type=int, default=100)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--viz-samples", type=int, default=0)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--allow-tool-regressions",
        action="store_true",
        help="Allow samples where one tool regresses as long as the combined delta stays high.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    tools = _manual_tool_specs(args)
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    process_manual_aerial_top100(
        tools=tools,
        output_root=output_root,
        selection_split=str(args.selection_split or "train+validation+test"),
        target_selected_samples=max(0, int(args.target_selected_samples)),
        max_samples=args.max_samples,
        viz_samples=int(args.viz_samples or 0),
        allow_tool_regressions=bool(args.allow_tool_regressions),
    )


if __name__ == "__main__":
    main()
