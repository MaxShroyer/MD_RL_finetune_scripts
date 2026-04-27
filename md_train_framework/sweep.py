from __future__ import annotations

import itertools
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from md_train_framework.config import FrameworkConfig, apply_overrides, save_framework_config
from md_train_framework.registry import RunRegistry
from md_train_framework.utils import append_jsonl, ensure_dir, slugify, stable_json_hash


@dataclass(frozen=True)
class SweepCandidate:
    name: str
    config: FrameworkConfig
    config_path: Path
    stage: str


def default_sweep_axes(config: FrameworkConfig) -> dict[str, list[Any]]:
    from md_train_framework.compat import resolve_backend
    from md_train_framework.profiles import default_profile_sweep_axes

    profile_id = str(resolve_backend(config).profile or "").strip()
    profile_axes = default_profile_sweep_axes(config, profile_id=profile_id)
    if profile_axes:
        return profile_axes
    if config.skill.id == "query":
        return {
            "lr": [5e-5, 1e-4, 2e-4, 5e-4],
            "rank": [16, 24, 32],
            "num_rollouts": [4, 8],
            "batch_size": [4, 8, 16],
            "off_policy": [False, True],
            "off_policy_mix_ratio": [0.25, 0.5],
            "reasoning": [False, True],
        }
    return {
        "lr": [5e-5, 1e-4, 2e-4, 5e-4],
        "rank": [8, 16, 32],
        "batch_size": [8, 16, 32],
        "group_size": [2, 4, 8],
        "reward_metric": ["f1", "miou"],
        "off_policy": [False, True],
    }


def generate_sweep_candidates(
    config: FrameworkConfig,
    *,
    registry: Optional[RunRegistry] = None,
    output_dir: Optional[Path] = None,
    max_candidates: int = 16,
) -> list[SweepCandidate]:
    axes = default_sweep_axes(config)
    if config.sweep.query_axes and config.skill.id == "query":
        axes.update(config.sweep.query_axes)
    if config.sweep.detect_point_axes and config.skill.id in {"detect", "point"}:
        axes.update(config.sweep.detect_point_axes)
    chosen_keys = _choose_axes(axes, config.skill.id)
    axis_values = [axes[key] for key in chosen_keys]
    sweep_root = ensure_dir(output_dir or config.resolved_path("md_train_framework/outputs/sweeps") / slugify(config.task.name))
    candidates: list[SweepCandidate] = []
    for index, combo in enumerate(itertools.product(*axis_values), start=1):
        if len(candidates) >= max(1, int(max_candidates)):
            break
        overrides = _combo_to_overrides(config, dict(zip(chosen_keys, combo, strict=False)))
        candidate = apply_overrides(config, overrides)
        if registry is not None and registry.seen_config(
            config_hash=candidate.config_hash,
            dataset_fingerprint=_dataset_fingerprint(candidate),
        ):
            continue
        candidate_path = sweep_root / f"{index:03d}_{slugify(_candidate_name(combo))}.json"
        save_framework_config(candidate_path, candidate)
        candidates.append(
            SweepCandidate(
                name=_candidate_name(combo),
                config=candidate,
                config_path=candidate_path,
                stage="full",
            )
        )
    return candidates


def stage1_candidates(candidates: Iterable[SweepCandidate], *, scale: float) -> list[SweepCandidate]:
    staged: list[SweepCandidate] = []
    for candidate in candidates:
        overrides = {
            "phases": [
                {
                    **phase.to_dict(),
                    "steps": max(1, int(round(phase.steps * max(0.05, float(scale))))),
                }
                for phase in candidate.config.phases
            ]
        }
        stage1_config = apply_overrides(candidate.config, overrides)
        stage1_path = candidate.config_path.with_name(f"{candidate.config_path.stem}_stage1.json")
        save_framework_config(stage1_path, stage1_config)
        staged.append(
            SweepCandidate(
                name=f"{candidate.name}-stage1",
                config=stage1_config,
                config_path=stage1_path,
                stage="stage1",
            )
        )
    return staged


class SweepOrchestrator:
    def __init__(self, registry: RunRegistry, *, python_executable: Optional[str] = None) -> None:
        self.registry = registry
        self.python_executable = python_executable or sys.executable

    def run(
        self,
        candidates: list[SweepCandidate],
        *,
        stagger_seconds: float,
        max_parallel: int,
        quarantine_path: Path,
    ) -> dict[str, Any]:
        pending = list(candidates)
        active: list[tuple[subprocess.Popen[str], SweepCandidate, float]] = []
        completed: list[dict[str, Any]] = []
        ensure_dir(quarantine_path.parent)
        while pending or active:
            while pending and len(active) < max(1, int(max_parallel)):
                candidate = pending.pop(0)
                process = subprocess.Popen(
                    [
                        self.python_executable,
                        "-m",
                        "md_train_framework",
                        "train",
                        "--config",
                        str(candidate.config_path),
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    cwd=str(Path.cwd()),
                    text=True,
                )
                active.append((process, candidate, time.monotonic()))
                time.sleep(max(0.0, float(stagger_seconds)))
            next_active: list[tuple[subprocess.Popen[str], SweepCandidate, float]] = []
            for process, candidate, started_at in active:
                returncode = process.poll()
                if returncode is None:
                    next_active.append((process, candidate, started_at))
                    continue
                if returncode != 0:
                    append_jsonl(
                        quarantine_path,
                        {
                            "stage": candidate.stage,
                            "candidate": candidate.name,
                            "config_path": str(candidate.config_path),
                            "returncode": returncode,
                        },
                    )
                completed.append(
                    {
                        "candidate": candidate.name,
                        "config_path": str(candidate.config_path),
                        "returncode": returncode,
                        "duration_s": round(time.monotonic() - started_at, 3),
                        "config_hash": candidate.config.config_hash,
                    }
                )
            active = next_active
            if active:
                time.sleep(0.5)
        return {"completed": completed, "quarantine_path": str(quarantine_path)}

    def select_top_candidates(
        self,
        candidates: Iterable[SweepCandidate],
        *,
        top_k: int,
    ) -> list[SweepCandidate]:
        scored: list[tuple[float, SweepCandidate]] = []
        for candidate in candidates:
            matching = [
                record
                for record in self.registry.list_records(skill=candidate.config.skill.id, task=candidate.config.task.name)
                if record.config_hash == candidate.config.config_hash
                and record.dataset_fingerprint == _dataset_fingerprint(candidate.config)
            ]
            if not matching:
                continue
            best = max(matching, key=lambda item: float(item.selection_metric_value or float("-inf")))
            scored.append((float(best.selection_metric_value or float("-inf")), candidate))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [candidate for _, candidate in scored[: max(1, int(top_k))]]


def _choose_axes(axes: dict[str, list[Any]], skill: str) -> list[str]:
    if any(":" in key for key in axes):
        return list(axes.keys())
    if skill == "query":
        return ["lr", "rank", "num_rollouts", "batch_size", "off_policy"]
    return ["lr", "rank", "batch_size", "group_size", "reward_metric"]


def _combo_to_overrides(config: FrameworkConfig, combo: dict[str, Any]) -> dict[str, Any]:
    phases = [phase.to_dict() for phase in config.phases]
    rl_index = next((idx for idx, phase in enumerate(phases) if phase.get("mode") == "rl"), len(phases) - 1)
    sft_index = next((idx for idx, phase in enumerate(phases) if phase.get("mode") == "sft"), None)
    if "lr" in combo:
        phases[rl_index]["lr"] = combo["lr"]
    if "batch_size" in combo:
        phases[rl_index]["batch_size"] = combo["batch_size"]
        if sft_index is not None:
            phases[sft_index]["batch_size"] = min(combo["batch_size"], 8) if config.skill.id == "query" else combo["batch_size"]
    if "group_size" in combo:
        phases[rl_index]["group_size"] = combo["group_size"]
    if "num_rollouts" in combo:
        phases[rl_index]["num_rollouts"] = combo["num_rollouts"]
    if "reasoning" in combo:
        phases[rl_index]["reasoning"] = combo["reasoning"]
        if sft_index is not None:
            phases[sft_index]["reasoning"] = combo["reasoning"]
    overrides: dict[str, Any] = {
        "phases": phases,
        "backend": {
            "id": config.backend.id,
            "train_overrides": {},
            "benchmark_overrides": {},
        },
    }
    if "rank" in combo:
        overrides["backend"]["train_overrides"]["rank"] = combo["rank"]
    if "off_policy" in combo:
        overrides["backend"]["train_overrides"]["off_policy"] = combo["off_policy"]
    if "off_policy_mix_ratio" in combo:
        overrides["backend"]["train_overrides"]["off_policy_mix_ratio"] = combo["off_policy_mix_ratio"]
    if "reward_metric" in combo:
        overrides["backend"]["train_overrides"]["reward_metric"] = combo["reward_metric"]
        overrides["reward"] = {
            **config.reward.to_dict(),
            "selection_metric": f"eval_{combo['reward_metric']}" if combo["reward_metric"] in {"f1", "miou"} else config.reward.selection_metric,
        }
    for key, value in combo.items():
        if ":" not in key:
            continue
        _apply_named_override(
            overrides=overrides,
            key=key,
            value=value,
            base_phases=phases,
            rl_index=rl_index,
            sft_index=sft_index,
        )
    return overrides


def _apply_named_override(
    *,
    overrides: dict[str, Any],
    key: str,
    value: Any,
    base_phases: list[dict[str, Any]],
    rl_index: int,
    sft_index: Optional[int],
) -> None:
    sections = key.split(":")
    if len(sections) < 2:
        return
    scope = sections[0]
    if scope == "phase" and len(sections) == 3:
        phase_name = sections[1]
        field_name = sections[2]
        phase_index = rl_index if phase_name == "rl" else sft_index if phase_name == "sft" else None
        if phase_index is None:
            return
        overrides["phases"][phase_index][field_name] = value
        return
    if scope == "backend" and len(sections) == 2:
        overrides["backend"].setdefault("train_overrides", {})[sections[1]] = value
        return
    if scope == "reward" and len(sections) == 2:
        reward_payload = dict(overrides.get("reward") or {})
        reward_payload[sections[1]] = value
        overrides["reward"] = reward_payload
        return
    if scope == "eval" and len(sections) == 2:
        eval_payload = dict(overrides.get("eval") or {})
        eval_payload[sections[1]] = value
        overrides["eval"] = eval_payload
        return
    if scope == "logging" and len(sections) == 2:
        logging_payload = dict(overrides.get("logging") or {})
        logging_payload[sections[1]] = value
        overrides["logging"] = logging_payload
        return
    if scope == "dataset" and len(sections) == 2:
        dataset_payload = dict(overrides.get("dataset") or {})
        dataset_payload[sections[1]] = value
        overrides["dataset"] = dataset_payload
        return
    if scope == "extra" and len(sections) == 2:
        overrides[sections[1]] = value


def _candidate_name(combo: Iterable[Any]) -> str:
    parts = [slugify(str(value), default="value") for value in combo]
    return "-".join(parts)


def _dataset_fingerprint(config: FrameworkConfig) -> str:
    return stable_json_hash(config.dataset_identity())
