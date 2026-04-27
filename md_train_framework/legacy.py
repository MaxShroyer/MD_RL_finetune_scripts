from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Optional

from md_train_framework.registry import RegistryRecord, RunRegistry
from md_train_framework.utils import read_json, stable_json_hash


RUN_CONFIG = "run_config.json"
TRAIN_SUMMARY = "train_summary.json"
EVAL_HISTORY = "eval_history.jsonl"
ASYNC_METRICS = "metrics.json"


def default_legacy_roots(repo_root: Path) -> list[Path]:
    candidates = [
        repo_root / "inspector_md" / "outputs" / "runs",
        repo_root / "football_detect" / "outputs",
        repo_root / "bone_fracture" / "outputs",
        repo_root / "neon_tree" / "outputs",
        repo_root / "vqa_rad" / "outputs",
        repo_root / "construction_site" / "outputs",
        repo_root / "disaster_m3" / "outputs" / "runs",
    ]
    return [path for path in candidates if path.exists()]


def import_legacy_runs(registry: RunRegistry, *, roots: Iterable[str | Path], repo_root: Path) -> dict[str, Any]:
    imported_runs = 0
    imported_checkpoints = 0
    imported_async = 0
    for raw_root in roots:
        root = Path(raw_root).expanduser()
        if not root.is_absolute():
            root = (repo_root / root).resolve()
        if not root.exists():
            continue
        for run_config_path in root.rglob(RUN_CONFIG):
            if "/wandb/" in str(run_config_path):
                continue
            result = _import_run_dir(registry, run_config_path.parent)
            imported_runs += result["runs"]
            imported_checkpoints += result["checkpoints"]
        for metrics_path in root.rglob(ASYNC_METRICS):
            if "async_checkpoint_eval" not in str(metrics_path):
                continue
            if _import_async_metrics(registry, metrics_path):
                imported_async += 1
    return {
        "imported_runs": imported_runs,
        "imported_checkpoints": imported_checkpoints,
        "imported_async_metrics": imported_async,
    }


def _import_run_dir(registry: RunRegistry, run_dir: Path) -> dict[str, int]:
    run_config = read_json(run_dir / RUN_CONFIG, default={}) or {}
    summary = read_json(run_dir / TRAIN_SUMMARY, default={}) or {}
    if not isinstance(run_config, dict):
        run_config = {}
    if not isinstance(summary, dict):
        summary = {}
    run_id = str(summary.get("finetune_id") or run_dir.name)
    skill = _guess_skill(run_config, run_dir)
    task = _guess_task(run_config, run_dir)
    mode = str(summary.get("mode") or run_config.get("mode") or _guess_mode(run_config)).strip() or "rl"
    backend_id = str(run_config.get("backend_id") or task).strip() or task
    selection_metric_name = _selection_metric_name(run_config, summary)
    best_value = _selection_metric_value(summary, selection_metric_name)
    baseline_metrics = _extract_baseline_metrics(summary)
    config_hash = stable_json_hash(run_config)
    dataset_fingerprint = stable_json_hash(
        {
            "dataset_dir": run_config.get("dataset_dir", ""),
            "dataset_path": run_config.get("dataset_path", ""),
            "dataset_name": run_config.get("dataset_name", ""),
            "train_split": run_config.get("train_split", run_config.get("split", "")),
            "val_split": run_config.get("val_split", ""),
        }
    )
    registry.upsert(
        RegistryRecord(
            record_type="run",
            run_id=run_id,
            skill=skill,
            task=task,
            mode=mode,
            backend_id=backend_id,
            status="imported",
            config_hash=config_hash,
            dataset_fingerprint=dataset_fingerprint,
            finetune_id=str(summary.get("finetune_id") or run_config.get("finetune_id") or ""),
            selection_metric_name=selection_metric_name,
            selection_metric_value=best_value,
            metrics=_extract_final_eval_metrics(summary),
            baseline_metrics=baseline_metrics,
            artifact_paths={
                "run_dir": str(run_dir),
                "run_config_json": str(run_dir / RUN_CONFIG),
                "train_summary_json": str(run_dir / TRAIN_SUMMARY),
                "eval_history_jsonl": str(run_dir / EVAL_HISTORY),
            },
            source_provenance={"source": "legacy_import", "path": str(run_dir)},
            metadata={"summary_keys": sorted(summary.keys())[:50]},
        )
    )
    checkpoints = 0
    eval_history_path = run_dir / EVAL_HISTORY
    if eval_history_path.exists():
        with eval_history_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                payload = json.loads(text)
                if not isinstance(payload, dict):
                    continue
                step = int(payload.get("step", 0))
                metrics = dict(payload.get("metrics", {}) or {})
                value = _lookup_metric(metrics, selection_metric_name)
                registry.upsert(
                    RegistryRecord(
                        record_type="checkpoint",
                        run_id=run_id,
                        parent_run_id=run_id,
                        skill=skill,
                        task=task,
                        mode=mode,
                        backend_id=backend_id,
                        status="imported",
                        config_hash=config_hash,
                        dataset_fingerprint=dataset_fingerprint,
                        finetune_id=str(summary.get("finetune_id") or run_config.get("finetune_id") or ""),
                        checkpoint_step=step,
                        selection_metric_name=selection_metric_name,
                        selection_metric_value=value,
                        metrics=metrics,
                        baseline_metrics=baseline_metrics,
                        artifact_paths={
                            "run_dir": str(run_dir),
                            "eval_history_jsonl": str(eval_history_path),
                        },
                        source_provenance={"source": "legacy_import", "path": str(eval_history_path)},
                        metadata={"stage": payload.get("stage", ""), "split": payload.get("split", "")},
                    )
                )
                checkpoints += 1
    return {"runs": 1, "checkpoints": checkpoints}


def _import_async_metrics(registry: RunRegistry, metrics_path: Path) -> bool:
    payload = read_json(metrics_path, default={}) or {}
    if not isinstance(payload, dict):
        return False
    parts = metrics_path.parts
    try:
        finetune_id = parts[-3]
        trainer = parts[-4]
    except IndexError:
        return False
    step_match = re.search(r"step0*([0-9]+)", metrics_path.parent.name)
    checkpoint_step = int(step_match.group(1)) if step_match else None
    selection_metric_name = _metric_name_from_payload(payload)
    value = _lookup_metric(payload, selection_metric_name)
    record = RegistryRecord(
        record_type="checkpoint",
        run_id=finetune_id,
        parent_run_id=finetune_id,
        skill=_guess_skill({"skill": trainer}, metrics_path),
        task=trainer,
        mode="rl",
        backend_id=trainer,
        status="imported_async_eval",
        config_hash=stable_json_hash({"trainer": trainer, "finetune_id": finetune_id}),
        dataset_fingerprint=stable_json_hash({"trainer": trainer}),
        finetune_id=finetune_id,
        checkpoint_step=checkpoint_step,
        selection_metric_name=selection_metric_name,
        selection_metric_value=value,
        metrics=payload,
        artifact_paths={"metrics_json": str(metrics_path)},
        source_provenance={"source": "legacy_async_eval", "path": str(metrics_path)},
    )
    registry.upsert(record)
    return True


def _guess_skill(run_config: dict[str, Any], path: Path) -> str:
    skill = str(run_config.get("skill", "")).strip()
    if skill:
        return skill
    path_text = str(path).lower()
    if "query" in path_text:
        return "query"
    if "point" in path_text:
        return "point"
    return "detect"


def _guess_task(run_config: dict[str, Any], path: Path) -> str:
    for key in ("task", "task_name", "wandb_run_name", "finetune_name"):
        value = str(run_config.get(key, "")).strip()
        if value:
            return value
    parts = [part for part in path.parts if part not in {"outputs", "runs"}]
    return parts[-2] if len(parts) >= 2 else path.name


def _guess_mode(run_config: dict[str, Any]) -> str:
    if str(run_config.get("mode", "")).strip():
        return str(run_config["mode"])
    if int(run_config.get("sft_steps", 0) or 0) > 0 and int(run_config.get("rl_steps", 0) or 0) > 0:
        return "sft_then_rl"
    if int(run_config.get("sft_steps", 0) or 0) > 0:
        return "sft"
    return "rl"


def _selection_metric_name(run_config: dict[str, Any], summary: dict[str, Any]) -> str:
    for key in ("best_metric", "selection_metric", "best_selection_metric_name"):
        for payload in (summary, run_config):
            value = str(payload.get(key, "")).strip()
            if value:
                return value
    return _metric_name_from_payload(_extract_final_eval_metrics(summary))


def _selection_metric_value(summary: dict[str, Any], metric_name: str) -> Optional[float]:
    for key in ("best_metric_value", "best_selection_metric"):
        value = summary.get(key)
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return _lookup_metric(_extract_final_eval_metrics(summary), metric_name)


def _extract_final_eval_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    final_eval = summary.get("final_eval", {})
    if isinstance(final_eval, dict):
        for split in ("validation", "val", "test"):
            split_payload = final_eval.get(split)
            if isinstance(split_payload, dict):
                return dict(split_payload)
    return {}


def _extract_baseline_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in summary.items():
        if str(key).startswith("baseline_"):
            payload[str(key)] = value
    return payload


def _metric_name_from_payload(payload: dict[str, Any]) -> str:
    for candidate in ("eval_reward_mean", "reward_mean", "eval_f1", "f1", "eval_miou", "miou"):
        if candidate in payload:
            return candidate
    return "score"


def _lookup_metric(payload: dict[str, Any], name: str) -> Optional[float]:
    candidates = [name]
    if name.startswith("eval_"):
        candidates.append(name[5:])
    else:
        candidates.append(f"eval_{name}")
    for candidate in candidates:
        try:
            return float(payload[candidate])
        except (KeyError, TypeError, ValueError):
            continue
    return None
