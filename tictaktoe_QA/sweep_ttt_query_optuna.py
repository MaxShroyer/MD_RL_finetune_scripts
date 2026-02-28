#!/usr/bin/env python3
"""Staged Optuna sweep runner for TicTacToe query RL training."""

from __future__ import annotations

import argparse
import copy
import json
import os
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

try:
    import optuna  # type: ignore
    from optuna.samplers import TPESampler  # type: ignore
    from optuna.trial import TrialState  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    optuna = None
    TPESampler = None
    TrialState = None

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT_PATH = Path(__file__).resolve().parent / "train_ttt_query_rl.py"
DEFAULT_BASE_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "query_rl_off_policy.json"
DEFAULT_SWEEP_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "query_rl_sweep_optuna.json"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "sweeps"

DEFAULT_SWEEP_CONFIG: dict[str, Any] = {
    "objective_metric": "best_avg_eval_reward",
    "screening_overrides": {
        "num_steps": 100,
        "eval_every": 10,
        "eval_fixed_subset_size": 256,
        "eval_fixed_subset_seed": 1337,
        "eval_temperature": 0.0,
        "eval_top_p": 1.0,
        "max_tokens": 64,
        "max_tokens_by_task": {
            "best_move": 32,
            "available_moves_count": 32,
            "available_moves_list": 128,
        },
        "top_p": 1.0,
        "reasoning": True,
        "early_stop": True,
        "early_stop_mode": "balanced",
        "skip_final_eval": True,
    },
    "stage_a": {
        "lr_values": [1e-4, 2e-4, 5e-4, 1e-3],
        "group_size": 4,
        "temperature": 0.7,
        "off_policy": False,
        "keep_top_k": 2,
    },
    "stage_b": {
        "group_size_values": [2, 4, 8],
        "keep_top_k": 2,
    },
    "stage_c": {
        "mix_ratios_default": [0.0, 0.25, 0.5],
        "mix_ratios_if_stable": [0.25, 0.5, 0.75],
        "warmup_buffer_pairs": [[10, 64], [30, 128]],
        "keep_top_k": 2,
    },
}


@dataclass(frozen=True)
class TrialSpec:
    stage: str
    trial_key: str
    params: dict[str, Any]
    resolved_config: dict[str, Any]
    parent_key: str = ""


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _format_value_for_key(value: Any) -> str:
    if isinstance(value, float):
        if value == 0:
            return "0"
        text = f"{value:.6g}".replace("+", "")
        return text.replace(".", "p")
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value)
    out = []
    for ch in text.lower():
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    collapsed = "".join(out).strip("_")
    return collapsed or "x"


def _resolve_parallelism(parallelism: str, parallel_cap: int) -> int:
    cap = max(1, int(parallel_cap))
    if str(parallelism).strip().lower() == "auto":
        return min(max(1, os.cpu_count() or 1), cap)
    return min(max(1, int(parallelism)), cap)


def _objective_from_ranking_payload(payload: dict[str, Any], objective_metric: str) -> float:
    metric = str(objective_metric).strip()
    if not metric:
        metric = "best_avg_eval_reward"

    current: Any = payload
    for part in metric.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            current = None
            break

    if isinstance(current, (int, float)):
        return float(current)

    fallback = payload.get("best_avg_eval_reward", 0.0)
    if isinstance(fallback, (int, float)):
        return float(fallback)
    return 0.0


def _parse_rate_from_ranking_payload(payload: dict[str, Any]) -> float:
    rankings = payload.get("rankings")
    if not isinstance(rankings, list) or not rankings:
        return 0.0

    first = rankings[0]
    if not isinstance(first, dict):
        return 0.0

    split_metrics = first.get("split_metrics")
    if not isinstance(split_metrics, dict):
        return 0.0

    rates: list[float] = []
    for metrics in split_metrics.values():
        if not isinstance(metrics, dict):
            continue
        value = metrics.get("eval_json_parse_rate")
        if isinstance(value, (int, float)):
            rates.append(float(value))

    if not rates:
        return 0.0
    return float(statistics.mean(rates))


def _is_trial_stable(result: dict[str, Any]) -> bool:
    if str(result.get("status")) != "completed":
        return False
    if bool(result.get("collapse_detected")):
        return False
    if bool(result.get("stopped_early")):
        return False
    return True


def _build_stage_a_specs(base_config: dict[str, Any], sweep_config: dict[str, Any]) -> list[TrialSpec]:
    screening = copy.deepcopy(sweep_config.get("screening_overrides") or {})
    stage_a = copy.deepcopy(sweep_config.get("stage_a") or {})
    lr_values = list(stage_a.get("lr_values") or [1e-4, 2e-4, 5e-4, 1e-3])

    base = _deep_merge(base_config, screening)
    base = _deep_merge(
        base,
        {
            "off_policy": bool(stage_a.get("off_policy", False)),
            "group_size": int(stage_a.get("group_size", 4)),
            "temperature": float(stage_a.get("temperature", 0.7)),
        },
    )

    specs: list[TrialSpec] = []
    for lr in lr_values:
        lr_value = float(lr)
        resolved = _deep_merge(base, {"lr": lr_value})
        key = f"stage_a_lr_{_format_value_for_key(lr_value)}"
        specs.append(
            TrialSpec(
                stage="stage_a",
                trial_key=key,
                params={"lr": lr_value},
                resolved_config=resolved,
            )
        )
    return specs


def _build_stage_b_specs(
    parent_results: list[dict[str, Any]],
    sweep_config: dict[str, Any],
    *,
    search_reasoning_stage_b: bool,
) -> list[TrialSpec]:
    stage_b = copy.deepcopy(sweep_config.get("stage_b") or {})
    group_sizes = [int(v) for v in (stage_b.get("group_size_values") or [2, 4, 8])]

    reasoning_values = [True]
    if search_reasoning_stage_b:
        reasoning_values = [True, False]

    specs: list[TrialSpec] = []
    for parent in parent_results:
        parent_key = str(parent.get("trial_key", ""))
        parent_config = parent.get("resolved_config")
        if not isinstance(parent_config, dict):
            continue

        for group_size in group_sizes:
            for reasoning in reasoning_values:
                resolved = _deep_merge(
                    parent_config,
                    {
                        "off_policy": False,
                        "group_size": int(group_size),
                        "reasoning": bool(reasoning),
                    },
                )
                key = (
                    f"stage_b_parent_{_format_value_for_key(parent_key)}"
                    f"_gs_{_format_value_for_key(group_size)}"
                    f"_reason_{_format_value_for_key(reasoning)}"
                )
                specs.append(
                    TrialSpec(
                        stage="stage_b",
                        trial_key=key,
                        params={
                            "parent_key": parent_key,
                            "group_size": int(group_size),
                            "reasoning": bool(reasoning),
                        },
                        resolved_config=resolved,
                        parent_key=parent_key,
                    )
                )
    return specs


def _build_stage_c_specs(parent_results: list[dict[str, Any]], sweep_config: dict[str, Any]) -> list[TrialSpec]:
    stage_c = copy.deepcopy(sweep_config.get("stage_c") or {})
    default_ratios = [float(v) for v in (stage_c.get("mix_ratios_default") or [0.0, 0.25, 0.5])]
    stable_ratios = [float(v) for v in (stage_c.get("mix_ratios_if_stable") or [0.25, 0.5, 0.75])]
    raw_pairs = stage_c.get("warmup_buffer_pairs") or [[10, 64], [30, 128]]
    warmup_pairs = [(int(pair[0]), int(pair[1])) for pair in raw_pairs]

    specs: list[TrialSpec] = []
    for parent in parent_results:
        parent_key = str(parent.get("trial_key", ""))
        parent_config = parent.get("resolved_config")
        if not isinstance(parent_config, dict):
            continue

        ratios = stable_ratios if _is_trial_stable(parent) else default_ratios
        if len(ratios) < 3:
            ratios = (ratios + default_ratios)[:3]
        ratios = ratios[:3]

        for idx, ratio in enumerate(ratios):
            warmup_steps, min_groups = warmup_pairs[idx % len(warmup_pairs)]
            resolved = _deep_merge(
                parent_config,
                {
                    "off_policy": True,
                    "off_policy_mix_ratio": float(ratio),
                    "off_policy_warmup_steps": int(warmup_steps),
                    "off_policy_min_buffer_groups": int(min_groups),
                },
            )
            key = (
                f"stage_c_parent_{_format_value_for_key(parent_key)}"
                f"_mix_{_format_value_for_key(ratio)}"
                f"_warm_{_format_value_for_key(warmup_steps)}"
                f"_minbuf_{_format_value_for_key(min_groups)}"
            )
            specs.append(
                TrialSpec(
                    stage="stage_c",
                    trial_key=key,
                    params={
                        "parent_key": parent_key,
                        "off_policy_mix_ratio": float(ratio),
                        "off_policy_warmup_steps": int(warmup_steps),
                        "off_policy_min_buffer_groups": int(min_groups),
                    },
                    resolved_config=resolved,
                    parent_key=parent_key,
                )
            )
    return specs


def _build_final_specs(
    parent_results: list[dict[str, Any]],
    *,
    final_num_steps: int,
    final_seed_count: int,
) -> list[TrialSpec]:
    specs: list[TrialSpec] = []
    for parent_idx, parent in enumerate(parent_results):
        parent_key = str(parent.get("trial_key", f"parent_{parent_idx}"))
        parent_config = parent.get("resolved_config")
        if not isinstance(parent_config, dict):
            continue

        base_seed = int(parent_config.get("seed", 42))
        for seed_idx in range(max(1, int(final_seed_count))):
            seed_value = base_seed + seed_idx
            resolved = _deep_merge(
                parent_config,
                {
                    "seed": int(seed_value),
                    "num_steps": int(final_num_steps),
                    "skip_final_eval": False,
                    "early_stop": False,
                },
            )
            key = (
                f"final_parent_{_format_value_for_key(parent_key)}"
                f"_seed_{_format_value_for_key(seed_value)}"
            )
            specs.append(
                TrialSpec(
                    stage="final_confirmation",
                    trial_key=key,
                    params={
                        "parent_key": parent_key,
                        "seed": int(seed_value),
                    },
                    resolved_config=resolved,
                    parent_key=parent_key,
                )
            )
    return specs


def _run_trial(
    spec: TrialSpec,
    *,
    trial_output_dir: Path,
    objective_metric: str,
    dry_run: bool,
) -> dict[str, Any]:
    stage_dir = trial_output_dir / spec.stage / spec.trial_key
    stage_dir.mkdir(parents=True, exist_ok=True)

    config_path = stage_dir / "config.json"
    ranking_path = stage_dir / "checkpoint_ranking.json"
    log_path = stage_dir / "train.log"

    resolved = copy.deepcopy(spec.resolved_config)
    resolved["checkpoint_ranking_output"] = str(ranking_path)
    if not str(resolved.get("wandb_run_name", "")).strip():
        resolved["wandb_run_name"] = f"{spec.stage}_{spec.trial_key}"

    _write_json(config_path, resolved)

    cmd = [sys.executable, str(TRAIN_SCRIPT_PATH), "--config", str(config_path)]
    record: dict[str, Any] = {
        "stage": spec.stage,
        "trial_key": spec.trial_key,
        "parent_key": spec.parent_key,
        "params": copy.deepcopy(spec.params),
        "resolved_config": resolved,
        "config_path": str(config_path),
        "ranking_path": str(ranking_path),
        "log_path": str(log_path),
        "command": cmd,
        "status": "pending",
        "objective": float("-inf"),
        "parse_rate": 0.0,
        "finetune_id": "",
        "stopped_early": False,
        "stop_reason": "",
        "collapse_detected": False,
        "return_code": None,
    }

    if dry_run:
        record["status"] = "dry_run"
        log_path.write_text(" ".join(cmd) + "\n", encoding="utf-8")
        return record

    started_at = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    elapsed_s = time.time() - started_at

    stdout_text = proc.stdout or ""
    stderr_text = proc.stderr or ""
    log_path.write_text(
        stdout_text + "\n\n[stderr]\n" + stderr_text,
        encoding="utf-8",
    )

    record["return_code"] = int(proc.returncode)
    record["elapsed_s"] = float(elapsed_s)

    if proc.returncode != 0:
        record["status"] = "failed"
        record["error"] = "trainer_exit_nonzero"
        return record

    if not ranking_path.exists():
        record["status"] = "failed"
        record["error"] = "missing_checkpoint_ranking_output"
        return record

    try:
        ranking_payload = _load_json(ranking_path)
    except Exception as exc:  # pragma: no cover
        record["status"] = "failed"
        record["error"] = f"ranking_parse_error: {exc}"
        return record

    training_status = ranking_payload.get("training_status")
    if not isinstance(training_status, dict):
        training_status = {}

    record["status"] = "completed"
    record["objective"] = _objective_from_ranking_payload(ranking_payload, objective_metric)
    record["parse_rate"] = _parse_rate_from_ranking_payload(ranking_payload)
    record["finetune_id"] = str(ranking_payload.get("finetune_id", ""))
    record["stopped_early"] = bool(training_status.get("stopped_early", False))
    record["stop_reason"] = str(training_status.get("stop_reason", ""))
    record["collapse_detected"] = bool(training_status.get("collapse_detected", False))
    return record


def _sort_stage_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def _score(item: dict[str, Any]) -> float:
        status = str(item.get("status", ""))
        if status != "completed":
            return float("-inf")
        value = item.get("objective")
        if isinstance(value, (int, float)):
            return float(value)
        return float("-inf")

    return sorted(results, key=_score, reverse=True)


def _top_completed(results: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    sorted_results = _sort_stage_results(results)
    out: list[dict[str, Any]] = []
    for item in sorted_results:
        if str(item.get("status")) != "completed":
            continue
        out.append(item)
        if len(out) >= max(1, int(k)):
            break
    return out


def _to_study_record(trial: Any) -> dict[str, Any]:
    attrs = dict(getattr(trial, "user_attrs", {}) or {})
    stage = str(attrs.get("stage", ""))
    trial_key = str(attrs.get("trial_key", ""))
    status = str(attrs.get("status", ""))
    if not status:
        state = getattr(trial, "state", None)
        if TrialState is not None and state == TrialState.COMPLETE:
            status = "completed"
        elif TrialState is not None and state == TrialState.FAIL:
            status = "failed"
        else:
            status = "unknown"

    objective = float("-inf")
    value = getattr(trial, "value", None)
    if isinstance(value, (int, float)):
        objective = float(value)

    record: dict[str, Any] = {
        "stage": stage,
        "trial_key": trial_key,
        "parent_key": str(attrs.get("parent_key", "")),
        "params": attrs.get("params", {}),
        "config_path": str(attrs.get("config_path", "")),
        "ranking_path": str(attrs.get("ranking_path", "")),
        "log_path": str(attrs.get("log_path", "")),
        "status": status,
        "objective": objective,
        "parse_rate": float(attrs.get("parse_rate", 0.0) or 0.0),
        "finetune_id": str(attrs.get("finetune_id", "")),
        "stopped_early": bool(attrs.get("stopped_early", False)),
        "stop_reason": str(attrs.get("stop_reason", "")),
        "collapse_detected": bool(attrs.get("collapse_detected", False)),
        "return_code": attrs.get("return_code"),
        "trial_number": int(getattr(trial, "number", -1)),
    }

    config_path = record.get("config_path")
    if isinstance(config_path, str) and config_path:
        path = Path(config_path)
        if path.exists():
            try:
                payload = _load_json(path)
                record["resolved_config"] = payload
            except Exception:
                pass

    return record


def _run_stage(
    *,
    stage_name: str,
    specs: list[TrialSpec],
    study: Optional[Any],
    trial_output_dir: Path,
    objective_metric: str,
    dry_run: bool,
    parallelism: int,
    resume: bool,
) -> list[dict[str, Any]]:
    if not specs:
        return []

    existing_by_key: dict[str, dict[str, Any]] = {}
    if study is not None and resume:
        for trial in study.get_trials(deepcopy=False):
            attrs = dict(getattr(trial, "user_attrs", {}) or {})
            if str(attrs.get("stage", "")) != stage_name:
                continue
            key = str(attrs.get("trial_key", ""))
            if not key:
                continue
            state = getattr(trial, "state", None)
            if TrialState is not None and state not in (TrialState.COMPLETE, TrialState.FAIL):
                continue
            existing_by_key[key] = _to_study_record(trial)

    pending_specs: list[TrialSpec] = []
    results: list[dict[str, Any]] = []
    for spec in specs:
        if resume and spec.trial_key in existing_by_key:
            results.append(existing_by_key[spec.trial_key])
        else:
            pending_specs.append(spec)

    if not pending_specs:
        return _sort_stage_results(results)

    bundles: list[tuple[Any, TrialSpec]] = []
    if study is not None and not dry_run:
        for spec in pending_specs:
            trial = study.ask()
            trial.set_user_attr("stage", stage_name)
            trial.set_user_attr("trial_key", spec.trial_key)
            trial.set_user_attr("parent_key", spec.parent_key)
            trial.set_user_attr("params", copy.deepcopy(spec.params))
            bundles.append((trial, spec))
    else:
        bundles = [(None, spec) for spec in pending_specs]

    def _worker(entry: tuple[Any, TrialSpec]) -> tuple[Any, dict[str, Any]]:
        trial, spec = entry
        record = _run_trial(
            spec,
            trial_output_dir=trial_output_dir,
            objective_metric=objective_metric,
            dry_run=dry_run,
        )
        return trial, record

    futures = []
    if parallelism > 1 and len(bundles) > 1:
        with ThreadPoolExecutor(max_workers=parallelism) as executor:
            for entry in bundles:
                futures.append(executor.submit(_worker, entry))
            for future in as_completed(futures):
                trial, record = future.result()
                if trial is not None and study is not None and not dry_run:
                    trial.set_user_attr("status", record.get("status"))
                    trial.set_user_attr("config_path", record.get("config_path"))
                    trial.set_user_attr("ranking_path", record.get("ranking_path"))
                    trial.set_user_attr("log_path", record.get("log_path"))
                    trial.set_user_attr("parse_rate", record.get("parse_rate", 0.0))
                    trial.set_user_attr("finetune_id", record.get("finetune_id", ""))
                    trial.set_user_attr("stopped_early", record.get("stopped_early", False))
                    trial.set_user_attr("stop_reason", record.get("stop_reason", ""))
                    trial.set_user_attr("collapse_detected", record.get("collapse_detected", False))
                    trial.set_user_attr("return_code", record.get("return_code"))
                    if str(record.get("status")) == "completed":
                        study.tell(trial, float(record.get("objective", 0.0)))
                    else:
                        study.tell(trial, state=TrialState.FAIL)
                    record["trial_number"] = int(trial.number)
                results.append(record)
    else:
        for entry in bundles:
            trial, record = _worker(entry)
            if trial is not None and study is not None and not dry_run:
                trial.set_user_attr("status", record.get("status"))
                trial.set_user_attr("config_path", record.get("config_path"))
                trial.set_user_attr("ranking_path", record.get("ranking_path"))
                trial.set_user_attr("log_path", record.get("log_path"))
                trial.set_user_attr("parse_rate", record.get("parse_rate", 0.0))
                trial.set_user_attr("finetune_id", record.get("finetune_id", ""))
                trial.set_user_attr("stopped_early", record.get("stopped_early", False))
                trial.set_user_attr("stop_reason", record.get("stop_reason", ""))
                trial.set_user_attr("collapse_detected", record.get("collapse_detected", False))
                trial.set_user_attr("return_code", record.get("return_code"))
                if str(record.get("status")) == "completed":
                    study.tell(trial, float(record.get("objective", 0.0)))
                else:
                    study.tell(trial, state=TrialState.FAIL)
                record["trial_number"] = int(trial.number)
            results.append(record)

    return _sort_stage_results(results)


def _open_study(*, storage_path: Path, study_name: str, seed: int) -> Any:
    if optuna is None or TPESampler is None:
        raise ModuleNotFoundError(
            "optuna is required for non-dry-run sweeps. Install with: pip install optuna"
        )

    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{storage_path}"
    sampler = TPESampler(seed=int(seed))
    return optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        load_if_exists=True,
    )


def _aggregate_final_results(final_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in final_results:
        if str(item.get("status")) != "completed":
            continue
        parent_key = str(item.get("parent_key", ""))
        grouped.setdefault(parent_key, []).append(item)

    rows: list[dict[str, Any]] = []
    for parent_key, runs in grouped.items():
        objectives = [float(run.get("objective", float("-inf"))) for run in runs]
        parse_rates = [float(run.get("parse_rate", 0.0)) for run in runs]
        if not objectives:
            continue
        mean_objective = float(statistics.mean(objectives))
        std_objective = float(statistics.pstdev(objectives)) if len(objectives) > 1 else 0.0
        mean_parse_rate = float(statistics.mean(parse_rates)) if parse_rates else 0.0
        rows.append(
            {
                "parent_key": parent_key,
                "num_runs": len(runs),
                "mean_objective": mean_objective,
                "std_objective": std_objective,
                "mean_parse_rate": mean_parse_rate,
                "trial_keys": [str(run.get("trial_key", "")) for run in runs],
            }
        )

    rows.sort(
        key=lambda item: (
            -float(item.get("mean_objective", float("-inf"))),
            -float(item.get("mean_parse_rate", 0.0)),
            float(item.get("std_objective", float("inf"))),
        )
    )
    return rows


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optuna staged sweep for TicTacToe query RL.")
    parser.add_argument("--base-config", default=str(DEFAULT_BASE_CONFIG_PATH))
    parser.add_argument("--sweep-config", default=str(DEFAULT_SWEEP_CONFIG_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--study-name", default="ttt_query_optuna")
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", dest="resume", action="store_true")
    resume_group.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    parser.add_argument("--parallelism", default="auto", help="auto or explicit integer")
    parser.add_argument("--parallel-cap", type=int, default=4)
    parser.add_argument("--search-reasoning-stage-b", action="store_true", default=False)
    parser.add_argument("--final-top-k", type=int, default=2)
    parser.add_argument("--final-seeds", type=int, default=2)
    parser.add_argument("--final-num-steps", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true", default=False)
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    args = _build_parser().parse_args(argv)

    base_config_path = Path(args.base_config).expanduser().resolve()
    sweep_config_path = Path(args.sweep_config).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    study_dir = output_root / str(args.study_name)
    study_dir.mkdir(parents=True, exist_ok=True)

    base_config = _load_json(base_config_path)
    sweep_config = _deep_merge(DEFAULT_SWEEP_CONFIG, _load_json(sweep_config_path))
    objective_metric = str(sweep_config.get("objective_metric", "best_avg_eval_reward"))

    parallelism = _resolve_parallelism(args.parallelism, int(args.parallel_cap))
    storage_path = study_dir / "optuna.db"

    resolved_sweep_payload = {
        "base_config_path": str(base_config_path),
        "sweep_config_path": str(sweep_config_path),
        "study_name": str(args.study_name),
        "resume": bool(args.resume),
        "dry_run": bool(args.dry_run),
        "parallelism": int(parallelism),
        "parallel_cap": int(args.parallel_cap),
        "search_reasoning_stage_b": bool(args.search_reasoning_stage_b),
        "final_top_k": int(args.final_top_k),
        "final_seeds": int(args.final_seeds),
        "final_num_steps": int(args.final_num_steps),
        "objective_metric": objective_metric,
        "sweep_config": sweep_config,
    }
    _write_json(study_dir / "resolved_sweep_config.json", resolved_sweep_payload)

    if args.dry_run:
        study = None
    else:
        seed = int(base_config.get("seed", 42))
        study = _open_study(storage_path=storage_path, study_name=str(args.study_name), seed=seed)

    stage_a_specs = _build_stage_a_specs(base_config, sweep_config)
    stage_a_results = _run_stage(
        stage_name="stage_a",
        specs=stage_a_specs,
        study=study,
        trial_output_dir=study_dir / "runs",
        objective_metric=objective_metric,
        dry_run=bool(args.dry_run),
        parallelism=parallelism,
        resume=bool(args.resume),
    )
    _write_json(study_dir / "stage_a_results.json", {"results": stage_a_results})

    keep_top_a = int((sweep_config.get("stage_a") or {}).get("keep_top_k", 2))
    top_a = _top_completed(stage_a_results, keep_top_a)

    stage_b_specs = _build_stage_b_specs(
        top_a,
        sweep_config,
        search_reasoning_stage_b=bool(args.search_reasoning_stage_b),
    )
    stage_b_results = _run_stage(
        stage_name="stage_b",
        specs=stage_b_specs,
        study=study,
        trial_output_dir=study_dir / "runs",
        objective_metric=objective_metric,
        dry_run=bool(args.dry_run),
        parallelism=parallelism,
        resume=bool(args.resume),
    )
    _write_json(study_dir / "stage_b_results.json", {"results": stage_b_results})

    keep_top_b = int((sweep_config.get("stage_b") or {}).get("keep_top_k", 2))
    top_b = _top_completed(stage_b_results, keep_top_b)

    stage_c_specs = _build_stage_c_specs(top_b, sweep_config)
    stage_c_results = _run_stage(
        stage_name="stage_c",
        specs=stage_c_specs,
        study=study,
        trial_output_dir=study_dir / "runs",
        objective_metric=objective_metric,
        dry_run=bool(args.dry_run),
        parallelism=parallelism,
        resume=bool(args.resume),
    )
    _write_json(study_dir / "stage_c_results.json", {"results": stage_c_results})

    keep_top_c = int(min(max(1, int(args.final_top_k)), 2))
    top_c = _top_completed(stage_c_results, keep_top_c)

    final_specs = _build_final_specs(
        top_c,
        final_num_steps=int(args.final_num_steps),
        final_seed_count=int(args.final_seeds),
    )
    final_results = _run_stage(
        stage_name="final_confirmation",
        specs=final_specs,
        study=study,
        trial_output_dir=study_dir / "runs",
        objective_metric=objective_metric,
        dry_run=bool(args.dry_run),
        parallelism=parallelism,
        resume=bool(args.resume),
    )

    final_rows = _aggregate_final_results(final_results)
    _write_json(
        study_dir / "final_confirmation_results.json",
        {
            "results": final_results,
            "aggregated": final_rows,
        },
    )

    best_config_payload: dict[str, Any] = {
        "best_parent_key": "",
        "best_config": {},
        "selection": {
            "method": "mean_objective_then_mean_parse_rate_then_lower_std",
            "objective_metric": objective_metric,
        },
    }

    if final_rows:
        winner_key = str(final_rows[0].get("parent_key", ""))
        winner_candidates = [item for item in top_c if str(item.get("trial_key", "")) == winner_key]
        if winner_candidates:
            winner = winner_candidates[0]
            best_config_payload["best_parent_key"] = winner_key
            best_config_payload["best_config"] = copy.deepcopy(winner.get("resolved_config") or {})
            best_config_payload["selection"]["winner_stats"] = final_rows[0]

    _write_json(study_dir / "best_config.json", best_config_payload)

    summary = {
        "study_name": str(args.study_name),
        "output_dir": str(study_dir),
        "dry_run": bool(args.dry_run),
        "resume": bool(args.resume),
        "parallelism": int(parallelism),
        "objective_metric": objective_metric,
        "counts": {
            "stage_a": len(stage_a_results),
            "stage_b": len(stage_b_results),
            "stage_c": len(stage_c_results),
            "final_confirmation": len(final_results),
        },
        "best_parent_key": str(best_config_payload.get("best_parent_key", "")),
        "best_objective": float(final_rows[0]["mean_objective"]) if final_rows else float("-inf"),
        "artifacts": {
            "resolved_sweep_config": str(study_dir / "resolved_sweep_config.json"),
            "stage_a_results": str(study_dir / "stage_a_results.json"),
            "stage_b_results": str(study_dir / "stage_b_results.json"),
            "stage_c_results": str(study_dir / "stage_c_results.json"),
            "final_confirmation_results": str(study_dir / "final_confirmation_results.json"),
            "best_config": str(study_dir / "best_config.json"),
            "summary": str(study_dir / "summary.json"),
            "optuna_db": str(storage_path),
        },
    }
    _write_json(study_dir / "summary.json", summary)

    print(
        f"sweep complete study={args.study_name} dry_run={args.dry_run} "
        f"best_parent={summary['best_parent_key']} best_objective={summary['best_objective']}"
    )


if __name__ == "__main__":
    main()
