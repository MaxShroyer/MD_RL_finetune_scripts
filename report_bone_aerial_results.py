#!/usr/bin/env python3
"""Generate a bone/aerial results report from remote W&B and local eval artifacts."""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_ENTITY = "maxshroyer49-na"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "reports"
DEFAULT_JSON_PATH = DEFAULT_OUTPUT_DIR / "bone_aerial_results_report.json"
DEFAULT_MD_PATH = DEFAULT_OUTPUT_DIR / "bone_aerial_results_report.md"

AERIAL_PROJECT = "moondream-aerial-airport-point-rl"
BONE_POINT_PROJECT = "moondream-bone-fracture-point-rl"
BONE_DETECT_PROJECT = "moondream-bone-fracture-detect-rl"

FINISHED_STATES = {"finished"}


@dataclass(frozen=True)
class LocalRunMeta:
    run_id: str
    config_path: Optional[str]
    finetune_id: Optional[str]


@dataclass(frozen=True)
class RemoteRun:
    project: str
    run_id: str
    name: str
    state: str
    created_at: Optional[str]
    url: Optional[str]
    config_path: Optional[str]
    finetune_id: Optional[str]
    best_step: Optional[int]
    best_checkpoint_step: Optional[int]
    best_eval_f1: Optional[float]
    best_eval_f1_macro: Optional[float]
    best_eval_miou: Optional[float]
    eval_f1: Optional[float]
    eval_f1_macro: Optional[float]
    eval_miou: Optional[float]
    test_f1: Optional[float]
    test_f1_macro: Optional[float]
    test_miou: Optional[float]

    @property
    def is_finished(self) -> bool:
        return self.state in FINISHED_STATES

    @property
    def created_sort_key(self) -> tuple[int, str]:
        if not self.created_at:
            return (0, "")
        return (1, self.created_at)


@dataclass(frozen=True)
class BenchmarkMetrics:
    artifact_path: str
    label: str
    finetune_id: Optional[str]
    checkpoint_step: Optional[int]
    eval_f1: Optional[float]
    eval_f1_macro: Optional[float]
    eval_miou: Optional[float]


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


def _fmt_num(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def _markdown_text(value: Optional[str]) -> str:
    text = (value or "").strip()
    return text.replace("|", "\\|")


def _remove_repo_root_from_sys_path() -> list[str]:
    removed: list[str] = []
    repo_root = REPO_ROOT.resolve()
    for entry in list(sys.path):
        try:
            resolved = Path(entry or ".").resolve()
        except Exception:
            continue
        if resolved == repo_root:
            removed.append(entry)
            sys.path.remove(entry)
    return removed


def _restore_sys_path(entries: list[str]) -> None:
    for entry in reversed(entries):
        sys.path.insert(0, entry)


def _load_wandb_module() -> Any:
    removed = _remove_repo_root_from_sys_path()
    try:
        wandb = importlib.import_module("wandb")
    finally:
        _restore_sys_path(removed)
    if not hasattr(wandb, "Api"):
        raise ImportError(
            "Unable to import the real W&B package. Run this script with the project virtualenv, "
            "for example `./.venv/bin/python report_bone_aerial_results.py`."
        )
    return wandb


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


def _collect_local_run_meta() -> dict[str, LocalRunMeta]:
    out: dict[str, LocalRunMeta] = {}
    for wandb_root in (REPO_ROOT / "wandb", REPO_ROOT / "bone_fracture" / "wandb"):
        if not wandb_root.exists():
            continue
        for run_dir in sorted(wandb_root.glob("run-*")):
            config_path = run_dir / "files" / "config.yaml"
            if not config_path.exists():
                continue
            text = config_path.read_text(encoding="utf-8", errors="replace")
            run_id = run_dir.name.rsplit("-", 1)[-1]
            out[run_id] = LocalRunMeta(
                run_id=run_id,
                config_path=_extract_wandb_scalar(text, "config"),
                finetune_id=_extract_wandb_scalar(text, "finetune_id"),
            )
    return out


def _repo_rel(value: Optional[str]) -> Optional[str]:
    text = (value or "").strip()
    if not text:
        return None
    try:
        path = Path(text)
        if path.is_absolute():
            return str(path.resolve().relative_to(REPO_ROOT))
    except Exception:
        return text
    prefix = f"{REPO_ROOT}/"
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def _collect_remote_runs(
    *,
    entity: str,
    project: str,
    api: Any,
    local_meta: dict[str, LocalRunMeta],
) -> list[RemoteRun]:
    runs: list[RemoteRun] = []
    for run in api.runs(f"{entity}/{project}", order="-created_at"):
        local = local_meta.get(run.id)
        summary = run.summary or {}
        runs.append(
            RemoteRun(
                project=project,
                run_id=run.id,
                name=str(run.name or ""),
                state=str(run.state or ""),
                created_at=getattr(run, "created_at", None),
                url=getattr(run, "url", None),
                config_path=_repo_rel((local.config_path if local else None) or run.config.get("config")),
                finetune_id=(local.finetune_id if local else None) or run.config.get("finetune_id"),
                best_step=_coerce_int(summary.get("best_step") or summary.get("best_checkpoint_step")),
                best_checkpoint_step=_coerce_int(summary.get("best_checkpoint_step")),
                best_eval_f1=_coerce_float(summary.get("best_eval_f1")),
                best_eval_f1_macro=_coerce_float(summary.get("best_eval_f1_macro")),
                best_eval_miou=_coerce_float(summary.get("best_eval_miou")),
                eval_f1=_coerce_float(summary.get("eval_f1")),
                eval_f1_macro=_coerce_float(summary.get("eval_f1_macro")),
                eval_miou=_coerce_float(summary.get("eval_miou")),
                test_f1=_coerce_float(summary.get("test_f1")),
                test_f1_macro=_coerce_float(summary.get("test_f1_macro")),
                test_miou=_coerce_float(summary.get("test_miou")),
            )
        )
    return runs


def _parse_model_finetune_checkpoint(model_name: Any) -> tuple[Optional[str], Optional[int]]:
    text = str(model_name or "").strip()
    if "/" not in text or "@" not in text:
        return (None, None)
    _, tail = text.split("/", 1)
    finetune_id, _, step_text = tail.partition("@")
    return (finetune_id or None, _coerce_int(step_text))


def _load_benchmark_metrics(path: Path, label: str) -> BenchmarkMetrics:
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidate = payload.get("candidate", {})
    finetune_id, checkpoint_step = _parse_model_finetune_checkpoint(candidate.get("model"))
    return BenchmarkMetrics(
        artifact_path=str(path.relative_to(REPO_ROOT)),
        label=label,
        finetune_id=finetune_id,
        checkpoint_step=checkpoint_step,
        eval_f1=_coerce_float(candidate.get("eval_f1")),
        eval_f1_macro=_coerce_float(candidate.get("eval_f1_macro")),
        eval_miou=_coerce_float(candidate.get("eval_miou")),
    )


def _match_config_suffix(run: RemoteRun, suffix: str) -> bool:
    return bool(run.config_path and run.config_path.endswith(suffix))


def _best_finished_run(runs: Iterable[RemoteRun], metric_name: str) -> Optional[RemoteRun]:
    finished = [run for run in runs if run.is_finished]
    if not finished:
        return None
    return _best_run(finished, metric_name)


def _best_run(runs: Iterable[RemoteRun], metric_name: str) -> Optional[RemoteRun]:
    candidates = list(runs)
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda run: (
            getattr(run, metric_name) is not None,
            getattr(run, metric_name) if getattr(run, metric_name) is not None else float("-inf"),
            run.created_sort_key,
        ),
    )


def _current_eval_sort_key(run: RemoteRun) -> tuple[float, float, tuple[int, str]]:
    return (
        run.eval_f1 if run.eval_f1 is not None else float("-inf"),
        run.eval_f1_macro if run.eval_f1_macro is not None else float("-inf"),
        run.created_sort_key,
    )


def _select_angle_only_tie_leaders(runs: Iterable[RemoteRun]) -> list[RemoteRun]:
    angle_only = [
        run
        for run in runs
        if run.is_finished
        and run.config_path
        and "bone_fracture_point_angle_only_recall_" in run.config_path
        and run.best_eval_f1 is not None
    ]
    if not angle_only:
        return []
    top_f1 = max(run.best_eval_f1 for run in angle_only if run.best_eval_f1 is not None)
    tied = [run for run in angle_only if run.best_eval_f1 == top_f1]
    return sorted(tied, key=_current_eval_sort_key, reverse=True)


def _recent_non_finished_runs(
    runs: Iterable[RemoteRun],
    *,
    created_after: Optional[str] = None,
    limit: int = 5,
) -> list[RemoteRun]:
    filtered = [run for run in runs if not run.is_finished]
    if created_after:
        filtered = [run for run in filtered if (run.created_at or "") > created_after]
    return sorted(filtered, key=lambda run: run.created_sort_key, reverse=True)[:limit]


def _row_for_run(
    *,
    branch: str,
    run: RemoteRun,
    note: str,
    held_out: Optional[BenchmarkMetrics] = None,
    use_detect_test_metrics: bool = False,
) -> dict[str, Any]:
    held_out_f1 = held_out.eval_f1 if held_out else None
    held_out_f1_macro = held_out.eval_f1_macro if held_out else None
    held_out_miou = held_out.eval_miou if held_out else None
    held_out_source = held_out.artifact_path if held_out else None
    if use_detect_test_metrics:
        held_out_f1 = run.test_f1
        held_out_f1_macro = run.test_f1_macro
        held_out_miou = run.test_miou
        held_out_source = "wandb:test_summary"
    best_macro = run.best_eval_f1_macro
    best_macro_source = "wandb:best_eval_f1_macro"
    if best_macro is None and run.eval_f1_macro is not None:
        best_macro = run.eval_f1_macro
        best_macro_source = "wandb:current_eval_f1_macro"
    return {
        "branch": branch,
        "run_id": run.run_id,
        "run_name": run.name,
        "state": run.state,
        "created_at": run.created_at,
        "config": run.config_path,
        "finetune_id": run.finetune_id,
        "winning_step": run.best_checkpoint_step or run.best_step,
        "wandb_best_eval_f1": run.best_eval_f1,
        "wandb_best_eval_f1_macro": best_macro,
        "wandb_best_eval_f1_macro_source": best_macro_source,
        "wandb_best_eval_miou": run.best_eval_miou,
        "current_eval_f1": run.eval_f1,
        "current_eval_f1_macro": run.eval_f1_macro,
        "current_eval_miou": run.eval_miou,
        "held_out_eval_f1": held_out_f1,
        "held_out_eval_f1_macro": held_out_f1_macro,
        "held_out_eval_miou": held_out_miou,
        "held_out_source": held_out_source,
        "note": note,
    }


def _row_for_benchmark(*, branch: str, benchmark: BenchmarkMetrics, note: str) -> dict[str, Any]:
    return {
        "branch": branch,
        "run_id": None,
        "run_name": benchmark.label,
        "state": "benchmark_artifact",
        "created_at": None,
        "config": None,
        "finetune_id": benchmark.finetune_id,
        "winning_step": benchmark.checkpoint_step,
        "wandb_best_eval_f1": None,
        "wandb_best_eval_f1_macro": None,
        "wandb_best_eval_f1_macro_source": None,
        "wandb_best_eval_miou": None,
        "current_eval_f1": None,
        "current_eval_f1_macro": None,
        "current_eval_miou": None,
        "held_out_eval_f1": benchmark.eval_f1,
        "held_out_eval_f1_macro": benchmark.eval_f1_macro,
        "held_out_eval_miou": benchmark.eval_miou,
        "held_out_source": benchmark.artifact_path,
        "note": note,
    }


def _find_best_for_config(runs: Iterable[RemoteRun], config_suffix: str, metric_name: str) -> Optional[RemoteRun]:
    matched = [run for run in runs if _match_config_suffix(run, config_suffix)]
    return _best_run(matched, metric_name)


def _non_finished_row(branch: str, run: RemoteRun) -> dict[str, Any]:
    return {
        "branch": branch,
        "run_id": run.run_id,
        "run_name": run.name,
        "state": run.state,
        "created_at": run.created_at,
        "config": run.config_path,
        "finetune_id": run.finetune_id,
        "winning_step": run.best_checkpoint_step or run.best_step,
        "wandb_best_eval_f1": run.best_eval_f1,
        "wandb_best_eval_f1_macro": run.best_eval_f1_macro,
        "wandb_best_eval_f1_macro_source": "wandb:best_eval_f1_macro" if run.best_eval_f1_macro is not None else None,
        "wandb_best_eval_miou": run.best_eval_miou,
        "current_eval_f1": run.eval_f1,
        "current_eval_f1_macro": run.eval_f1_macro,
        "current_eval_miou": run.eval_miou,
        "held_out_eval_f1": None,
        "held_out_eval_f1_macro": None,
        "held_out_eval_miou": None,
        "held_out_source": None,
        "note": "Newer non-finished run. Visible for context but not eligible for current-best ranking.",
    }


def build_report(*, entity: str = DEFAULT_ENTITY) -> dict[str, Any]:
    wandb = _load_wandb_module()
    api = wandb.Api(timeout=20)
    local_meta = _collect_local_run_meta()

    aerial_runs = _collect_remote_runs(entity=entity, project=AERIAL_PROJECT, api=api, local_meta=local_meta)
    bone_point_runs = _collect_remote_runs(entity=entity, project=BONE_POINT_PROJECT, api=api, local_meta=local_meta)
    bone_detect_runs = _collect_remote_runs(entity=entity, project=BONE_DETECT_PROJECT, api=api, local_meta=local_meta)

    aerial_tiling_winner = _best_finished_run(aerial_runs, "best_eval_f1")
    aerial_standard_benchmark = _load_benchmark_metrics(
        REPO_ROOT / "aerial_airport" / "outputs" / "benchmarks" / "benchmark_aerial_airport_point_best.json",
        label="aerial benchmark-backed standard/v2",
    )
    bone_angle_benchmark = _load_benchmark_metrics(
        REPO_ROOT / "bone_fracture" / "outputs" / "benchmarks" / "benchmark_bone_fracture_point_step60.json",
        label="bone angle-only benchmark-backed",
    )
    bone_full_benchmark = _load_benchmark_metrics(
        REPO_ROOT / "bone_fracture" / "outputs" / "benchmarks" / "benchmark_bone_fracture_point_full_best.json",
        label="bone full-data benchmark-backed",
    )

    angle_tie_leaders = _select_angle_only_tie_leaders(bone_point_runs)
    bone_full_winner = _best_finished_run(
        [
            run
            for run in bone_point_runs
            if _match_config_suffix(
                run,
                "bone_fracture/configs/cicd/cicd_train_bone_fracture_point_full_recall_offpolicy_lite.json",
            )
        ],
        "best_eval_f1",
    )
    bone_detect_f1_leader = _find_best_for_config(
        bone_detect_runs,
        "bone_fracture/configs/cicd/cicd_train_bone_fracture_detect_fracture_only_f1_klguard.json",
        "best_eval_f1",
    ) or _best_finished_run(bone_detect_runs, "best_eval_f1")
    bone_detect_miou_leader = _find_best_for_config(
        bone_detect_runs,
        "bone_fracture/configs/cicd/cicd_train_bone_fracture_detect_fracture_promptmix_primary_anchor.json",
        "best_eval_miou",
    ) or _best_finished_run(bone_detect_runs, "best_eval_miou")

    if aerial_tiling_winner is None:
        raise RuntimeError("Unable to find a finished aerial winner on W&B.")
    if bone_full_winner is None:
        raise RuntimeError("Unable to find a finished bone full-data leader on W&B.")
    if bone_detect_f1_leader is None or bone_detect_miou_leader is None:
        raise RuntimeError("Unable to find finished bone detect leaders on W&B.")

    current_best_rows: list[dict[str, Any]] = [
        _row_for_run(
            branch="aerial overall (tiling winner)",
            run=aerial_tiling_winner,
            held_out=None,
            note=(
                "Current finished aerial winner by W&B best_eval_f1. "
                "No direct held-out benchmark exists for this checkpoint."
            ),
        ),
    ]
    current_best_rows.extend(
        _row_for_run(
            branch="bone point angle-only",
            run=run,
            held_out=None,
            note=(
                "Tied on best_eval_f1=0.700 across the current finished angle-only leaders. "
                "Direct held-out eval is unavailable for this checkpoint."
            ),
        )
        for run in angle_tie_leaders
    )
    current_best_rows.append(
        _row_for_run(
            branch="bone point full-data",
            run=bone_full_winner,
            held_out=None,
            note="Current finished full-data point leader by W&B best_eval_f1. Direct held-out eval is unavailable.",
        )
    )
    current_best_rows.append(
        _row_for_run(
            branch="bone detect F1 leader",
            run=bone_detect_f1_leader,
            use_detect_test_metrics=True,
            note=(
                "Plan-locked detect F1 leader from fracture_only_f1_klguard. "
                "Held-out metrics come from W&B test_* summary fields."
            ),
        )
    )
    current_best_rows.append(
        _row_for_run(
            branch="bone detect mIoU leader",
            run=bone_detect_miou_leader,
            use_detect_test_metrics=True,
            note="Finished detect leader by W&B best_eval_miou. Held-out metrics come from W&B test_* summary fields.",
        )
    )

    benchmark_backed_rows = [
        _row_for_benchmark(
            branch="aerial benchmark-backed standard/v2",
            benchmark=aerial_standard_benchmark,
            note="Latest local held-out aerial benchmark artifact for the standard/v2 branch.",
        ),
        _row_for_benchmark(
            branch="bone point angle-only benchmark-backed",
            benchmark=bone_angle_benchmark,
            note="Best available local held-out eval for the bone angle-only branch.",
        ),
        _row_for_benchmark(
            branch="bone point full-data benchmark-backed",
            benchmark=bone_full_benchmark,
            note="Best available local held-out eval for the bone full-data branch.",
        ),
    ]

    featured_run_ids = {str(row["run_id"]) for row in current_best_rows if row.get("run_id")}
    non_finished_rows = []
    non_finished_rows.extend(
        _non_finished_row("aerial", run)
        for run in _recent_non_finished_runs(aerial_runs, created_after=aerial_tiling_winner.created_at)
        if run.run_id not in featured_run_ids
    )
    non_finished_rows.extend(
        _non_finished_row("bone point", run)
        for run in _recent_non_finished_runs(
            bone_point_runs,
            created_after=max((run.created_at or "") for run in angle_tie_leaders),
            limit=3,
        )
        if run.run_id not in featured_run_ids
    )
    non_finished_rows.extend(
        _non_finished_row("bone detect", run)
        for run in _recent_non_finished_runs(bone_detect_runs, created_after=bone_detect_f1_leader.created_at, limit=3)
        if run.run_id not in featured_run_ids
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "wandb_entity": entity,
            "projects": [AERIAL_PROJECT, BONE_POINT_PROJECT, BONE_DETECT_PROJECT],
            "finished_states_used_for_ranking": sorted(FINISHED_STATES),
        },
        "assumptions": [
            "Current best means best completed run, not a partial in-progress snapshot.",
            "Angle-only bone point and full-data bone point are reported separately because they use different datasets.",
            "If W&B does not expose best_eval_f1_macro, the report falls back to current eval_f1_macro and labels the source explicitly.",
            "If a checkpoint has no direct held-out eval artifact, held-out fields remain blank instead of being inferred from validation.",
        ],
        "current_best_rows": current_best_rows,
        "benchmark_backed_rows": benchmark_backed_rows,
        "non_finished_rows": non_finished_rows,
        "validation_notes": [
            "Remote W&B under maxshroyer49-na is treated as the source of truth for current run state.",
            "Local cached config.yaml files are used to recover exact config paths and finetune IDs when W&B run.config is incomplete.",
            "Point-task held-out metrics come from local benchmark JSON artifacts; detect held-out metrics come from W&B test_* summary fields.",
        ],
    }


def _section_markdown(title: str, rows: list[dict[str, Any]]) -> str:
    lines = [f"## {title}", ""]
    if not rows:
        lines.append("_None_")
        lines.append("")
        return "\n".join(lines)
    headers = [
        "Branch",
        "Run ID",
        "Config",
        "Finetune ID",
        "Step",
        "Best F1",
        "Best Macro F1",
        "Best mIoU",
        "Current F1",
        "Current Macro F1",
        "Current mIoU",
        "Held-out F1",
        "Held-out Macro F1",
        "Held-out mIoU",
        "Note",
    ]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _markdown_text(row.get("branch")),
                    _markdown_text(row.get("run_id")),
                    _markdown_text(row.get("config")),
                    _markdown_text(row.get("finetune_id")),
                    _markdown_text(str(row.get("winning_step") or "")),
                    _fmt_num(row.get("wandb_best_eval_f1")),
                    _fmt_num(row.get("wandb_best_eval_f1_macro")),
                    _fmt_num(row.get("wandb_best_eval_miou")),
                    _fmt_num(row.get("current_eval_f1")),
                    _fmt_num(row.get("current_eval_f1_macro")),
                    _fmt_num(row.get("current_eval_miou")),
                    _fmt_num(row.get("held_out_eval_f1")),
                    _fmt_num(row.get("held_out_eval_f1_macro")),
                    _fmt_num(row.get("held_out_eval_miou")),
                    _markdown_text(row.get("note")),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Bone And Aerial Results Report",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "## Source",
        "",
        f"- W&B entity: `{report['source']['wandb_entity']}`",
        f"- Ranking only considers finished states: `{', '.join(report['source']['finished_states_used_for_ranking'])}`",
        "",
        "## Assumptions",
        "",
    ]
    lines.extend(f"- {item}" for item in report["assumptions"])
    lines.append("")
    lines.append(_section_markdown("Current Best", report["current_best_rows"]))
    lines.append(_section_markdown("Benchmark-Backed Held-Out Results", report["benchmark_backed_rows"]))
    lines.append(_section_markdown("Recent Non-Finished Runs", report["non_finished_rows"]))
    lines.append("## Validation Notes")
    lines.append("")
    lines.extend(f"- {item}" for item in report["validation_notes"])
    lines.append("")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a bone/aerial results report from W&B and local benchmarks.")
    parser.add_argument("--wandb-entity", default=DEFAULT_ENTITY)
    parser.add_argument("--out-json", default=str(DEFAULT_JSON_PATH))
    parser.add_argument("--out-md", default=str(DEFAULT_MD_PATH))
    parser.add_argument("--print-markdown", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    report = build_report(entity=str(args.wandb_entity))
    markdown = render_markdown(report)

    out_json = Path(args.out_json).expanduser().resolve()
    out_md = Path(args.out_md).expanduser().resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_md.write_text(markdown, encoding="utf-8")

    print(f"wrote report JSON: {out_json}")
    print(f"wrote report Markdown: {out_md}")
    if args.print_markdown:
        print()
        print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
