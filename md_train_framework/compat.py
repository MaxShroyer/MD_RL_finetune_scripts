from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from md_train_framework.artifacts import RunPaths, write_benchmark_metrics, write_run_config
from md_train_framework.config import FrameworkConfig, PhaseSpec
from md_train_framework.metrics import MetricPolicy
from md_train_framework.rewards import RewardPreset
from md_train_framework.runtime import (
    TrainOutcome,
    create_client_and_finetune,
    evaluate_current_finetune,
    make_trainer,
)
from md_train_framework.utils import now_utc_iso, read_json, slugify, write_json
from md_train_framework.wandb_logger import WandbLogger


@dataclass(frozen=True)
class BackendDefinition:
    id: str
    skill: str
    mode: str = "live"
    profile: str = "generic"
    supports_checkpoint_replay: bool = False


@dataclass(frozen=True)
class BackendRunResult:
    status: str
    finetune_id: str
    selection_metric_name: str
    selection_metric_value: Optional[float]
    metrics: dict[str, Any]
    summary: dict[str, Any]
    artifact_paths: dict[str, str]
    rendered_backend_config: dict[str, Any]
    baseline_metrics: dict[str, Any]


_BACKENDS: dict[str, BackendDefinition] = {
    "mock_detect": BackendDefinition(id="mock_detect", skill="detect", mode="mock"),
    "mock_point": BackendDefinition(id="mock_point", skill="point", mode="mock"),
    "mock_query": BackendDefinition(id="mock_query", skill="query", mode="mock"),
    "generic_detect": BackendDefinition(id="generic_detect", skill="detect"),
    "generic_point": BackendDefinition(id="generic_point", skill="point"),
    "generic_query": BackendDefinition(id="generic_query", skill="query"),
    "ballholder_detect": BackendDefinition(id="ballholder_detect", skill="detect", profile="ballholder_detect"),
    "statefarm_detect": BackendDefinition(id="statefarm_detect", skill="detect", profile="statefarm_detect"),
    "pandid_point": BackendDefinition(id="pandid_point", skill="point", profile="pandid_point"),
    "football_detect": BackendDefinition(id="football_detect", skill="detect", profile="football"),
    "neon_tree_detect": BackendDefinition(id="neon_tree_detect", skill="detect", profile="neon_tree"),
    "bone_fracture_detect": BackendDefinition(id="bone_fracture_detect", skill="detect", profile="bone_detect"),
    "construction_site_detect": BackendDefinition(id="construction_site_detect", skill="detect", profile="construction_detect"),
    "inspector_detect": BackendDefinition(id="inspector_detect", skill="detect", profile="inspector_detect"),
    "aerial_airport_detect": BackendDefinition(id="aerial_airport_detect", skill="detect", profile="airport_detect"),
    "bone_fracture_point": BackendDefinition(id="bone_fracture_point", skill="point", profile="bone_point"),
    "inspector_point": BackendDefinition(id="inspector_point", skill="point", profile="inspector_point"),
    "aerial_airport_point": BackendDefinition(id="aerial_airport_point", skill="point", profile="airport_point"),
    "inspector_query": BackendDefinition(id="inspector_query", skill="query", profile="inspector_query"),
    "vqa_rad_query": BackendDefinition(id="vqa_rad_query", skill="query", profile="vqa_rad"),
    "construction_site_query_caption": BackendDefinition(id="construction_site_query_caption", skill="query", profile="construction_caption"),
    "construction_site_query_rule_vqa": BackendDefinition(id="construction_site_query_rule_vqa", skill="query", profile="construction_rule_vqa"),
    "chess_query": BackendDefinition(id="chess_query", skill="query", profile="chess"),
    "ttt_query": BackendDefinition(id="ttt_query", skill="query", profile="ttt"),
}


def resolve_backend(config: FrameworkConfig) -> BackendDefinition:
    backend_id = str(config.backend.id or "").strip()
    if not backend_id:
        if config.skill.id == "query":
            backend_id = "generic_query"
        elif config.skill.id == "point":
            backend_id = "generic_point"
        else:
            backend_id = "generic_detect"
    backend = _BACKENDS.get(backend_id)
    if backend is None:
        raise KeyError(f"unknown backend id: {backend_id}")
    if backend.skill != config.skill.id:
        raise ValueError(f"backend {backend.id} does not match skill {config.skill.id}")
    return backend


def backend_ids(*, skill: Optional[str] = None) -> list[str]:
    backends = sorted(_BACKENDS.values(), key=lambda item: item.id)
    if skill is None:
        return [backend.id for backend in backends]
    return [backend.id for backend in backends if backend.skill == skill]


def api_key_slots(config: FrameworkConfig, rendered_backend_config: dict[str, Any]) -> list[str]:
    slots: list[str] = []
    api_key_env_vars = rendered_backend_config.get("api_key_env_vars")
    if isinstance(api_key_env_vars, list):
        slots.extend(str(item) for item in api_key_env_vars if str(item).strip())
    api_key_env_var = str(rendered_backend_config.get("api_key_env_var", "")).strip()
    if api_key_env_var:
        slots.append(api_key_env_var)
    return list(dict.fromkeys(slots))


def render_train_config(config: FrameworkConfig, paths: RunPaths, *, reward_preset: RewardPreset) -> dict[str, Any]:
    backend = resolve_backend(config)
    return {
        "backend_id": backend.id,
        "backend_mode": backend.mode,
        "backend_profile": backend.profile,
        "api_key_env_var": str(config.backend.train_overrides.get("api_key_env_var", config.extra.get("api_key_env_var", "MOONDREAM_API_KEY"))),
        "api_key_env_vars": list(config.backend.train_overrides.get("api_key_env_vars", [])),
        "base_url": str(config.backend.train_overrides.get("base_url", config.extra.get("base_url", "https://api.moondream.ai/v1"))),
        "env_file": str(config.backend.train_overrides.get("env_file", config.extra.get("env_file", ""))),
        "rank": int(config.backend.train_overrides.get("rank", 16)),
        "seed": int(config.backend.train_overrides.get("seed", config.extra.get("seed", 42))),
        "dataset": config.dataset.to_dict(),
        "phases": [phase.to_dict() for phase in config.phases],
        "reward": reward_preset.to_dict() if hasattr(reward_preset, "to_dict") else reward_preset.__dict__,
        "eval": config.eval.to_dict(),
        "logging": {
            **config.logging.to_dict(),
            "run_dir": str(paths.run_dir),
            "async_eval_dir": str(paths.async_eval_dir),
        },
        "recovery": config.recovery.to_dict(),
        "train_overrides": dict(config.backend.train_overrides),
    }


def render_benchmark_config(
    config: FrameworkConfig,
    paths: RunPaths,
    *,
    reward_preset: RewardPreset,
    finetune_id: str = "",
    checkpoint_step: Optional[int] = None,
) -> dict[str, Any]:
    payload = render_train_config(config, paths, reward_preset=reward_preset)
    payload["finetune_id"] = str(finetune_id)
    payload["checkpoint_step"] = checkpoint_step
    return payload


def train_backend(config: FrameworkConfig, paths: RunPaths, reward_preset: RewardPreset, metric_policy: MetricPolicy) -> BackendRunResult:
    backend = resolve_backend(config)
    rendered = render_train_config(config, paths, reward_preset=reward_preset)
    write_run_config(paths, config, rendered_backend_config=rendered)
    if backend.mode == "mock":
        return _run_mock_train(config, paths, metric_policy, rendered)
    client, finetune = create_client_and_finetune(config)
    logger = WandbLogger(
        enabled=bool(config.logging.enable_wandb),
        project=config.logging.wandb_project,
        run_name=config.logging.wandb_run_name or paths.run_id,
        config_payload=rendered,
    )
    try:
        trainer = make_trainer(
            config=config,
            reward_preset=reward_preset,
            metric_policy=metric_policy,
            paths=paths,
            finetune=finetune,
            logger=logger,
        )
        outcome = trainer.run()
        return BackendRunResult(
            status="succeeded",
            finetune_id=outcome.finetune_id,
            selection_metric_name=outcome.selection_metric_name,
            selection_metric_value=outcome.selection_metric_value,
            metrics=outcome.metrics,
            summary=outcome.summary,
            artifact_paths={**outcome.artifact_paths, "backend_id": backend.id},
            rendered_backend_config=rendered,
            baseline_metrics=dict(outcome.baseline_metrics),
        )
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()


def baseline_backend(
    config: FrameworkConfig,
    paths: RunPaths,
    reward_preset: RewardPreset,
    metric_policy: MetricPolicy,
    *,
    finetune_id: str = "",
    checkpoint_step: Optional[int] = None,
) -> BackendRunResult:
    backend = resolve_backend(config)
    rendered = render_benchmark_config(
        config,
        paths,
        reward_preset=reward_preset,
        finetune_id=finetune_id,
        checkpoint_step=checkpoint_step,
    )
    write_run_config(paths, config, rendered_backend_config=rendered)
    if backend.mode == "mock":
        return _run_mock_baseline(config, paths, metric_policy, rendered)
    if checkpoint_step is not None:
        replay = _load_saved_checkpoint_eval(paths, checkpoint_step)
        if replay is None:
            raise ValueError(
                "checkpoint replay requires metrics captured at save time in tuna-sdk-only mode. "
                "No saved checkpoint eval artifact was found for that step."
            )
        write_benchmark_metrics(paths, replay["metrics"])
        if replay["predictions_path"].exists():
            shutil.copy2(replay["predictions_path"], paths.predictions_jsonl)
        return BackendRunResult(
            status="replayed",
            finetune_id=str(finetune_id),
            selection_metric_name=metric_policy.selection_metric,
            selection_metric_value=_metric_value(replay["metrics"], metric_policy.selection_metric),
            metrics=replay["metrics"],
            summary={
                "mode": "replay",
                "checkpoint_step": int(checkpoint_step),
                "supports_checkpoint_replay": True,
                "replayed_from": str(replay["metrics_path"]),
            },
            artifact_paths=paths.artifact_paths(),
            rendered_backend_config=rendered,
            baseline_metrics={},
        )
    client, finetune = create_client_and_finetune(_config_with_optional_finetune(config, finetune_id=finetune_id))
    created_temp = not str(finetune_id).strip()
    try:
        metrics, predictions = evaluate_current_finetune(
            config=config,
            reward_preset=reward_preset,
            metric_policy=metric_policy,
            paths=paths,
            finetune=finetune,
            split_name=config.dataset.val_split,
        )
        write_benchmark_metrics(paths, metrics)
        with paths.predictions_jsonl.open("w", encoding="utf-8") as handle:
            for item in predictions:
                handle.write(json.dumps(item, ensure_ascii=True, sort_keys=True))
                handle.write("\n")
        return BackendRunResult(
            status="succeeded",
            finetune_id=finetune.finetune_id,
            selection_metric_name=metric_policy.selection_metric,
            selection_metric_value=_metric_value(metrics, metric_policy.selection_metric),
            metrics=metrics,
            summary={
                "mode": "baseline",
                "created_temp_finetune": created_temp,
                "finetune_id": finetune.finetune_id,
                "logged_at": now_utc_iso(),
            },
            artifact_paths=paths.artifact_paths(),
            rendered_backend_config=rendered,
            baseline_metrics={},
        )
    finally:
        if created_temp and bool(config.backend.benchmark_overrides.get("delete_temp_baseline_finetune", True)):
            delete = getattr(finetune, "delete", None)
            if callable(delete):
                try:
                    delete()
                except Exception:
                    pass
        close = getattr(client, "close", None)
        if callable(close):
            close()


def replay_eval_backend(
    config: FrameworkConfig,
    paths: RunPaths,
    reward_preset: RewardPreset,
    metric_policy: MetricPolicy,
    *,
    finetune_id: str,
    checkpoint_step: Optional[int],
) -> BackendRunResult:
    return baseline_backend(
        config,
        paths,
        reward_preset,
        metric_policy,
        finetune_id=finetune_id,
        checkpoint_step=checkpoint_step,
    )


def _metric_value(metrics: dict[str, Any], selection_metric: str) -> Optional[float]:
    for candidate in (selection_metric, selection_metric[5:] if selection_metric.startswith("eval_") else f"eval_{selection_metric}"):
        try:
            return float(metrics[candidate])
        except (KeyError, TypeError, ValueError):
            continue
    return None


def _config_with_optional_finetune(config: FrameworkConfig, *, finetune_id: str) -> FrameworkConfig:
    if not str(finetune_id).strip():
        return config
    payload = config.to_dict(include_meta=False)
    payload.setdefault("backend", {}).setdefault("train_overrides", {})["finetune_id"] = str(finetune_id)
    return FrameworkConfig.from_dict(payload, config_path=config.config_path)


def _run_mock_train(
    config: FrameworkConfig,
    paths: RunPaths,
    metric_policy: MetricPolicy,
    rendered_backend_config: dict[str, Any],
) -> BackendRunResult:
    total_steps = sum(max(0, phase.steps) for phase in config.phases)
    save_every = max(1, int(config.eval.save_every))
    selection_metric = metric_policy.selection_metric
    baseline_metrics = _mock_metrics(config.skill.id, selection_metric, 0.32 if config.skill.id != "query" else 0.40)
    best_metrics = dict(baseline_metrics)
    best_value = _metric_value(best_metrics, selection_metric) or 0.0
    for step in range(save_every, max(save_every, total_steps) + 1, save_every):
        progress = step / max(1, total_steps)
        metrics = _mock_metrics(config.skill.id, selection_metric, best_value + (0.25 * progress))
        best_metrics = metrics
        best_value = _metric_value(metrics, selection_metric) or best_value
        event = {
            "step": step,
            "checkpoint_step": step,
            "stage": "rl" if config.mode != "sft" else "sft",
            "split": config.dataset.val_split,
            "metrics": metrics,
        }
        with paths.eval_history_jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=True, sort_keys=True))
            handle.write("\n")
        metrics_dir = paths.async_eval_dir / config.skill.id / paths.run_id / f"step{step:06d}_mock"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        write_json(metrics_dir / "metrics.json", metrics)
        (metrics_dir / "predictions.jsonl").write_text('{"sample_id":"mock"}\n', encoding="utf-8")
        write_json(metrics_dir / "job.json", {"status": "completed", "checkpoint_step": step})
        pointer_payload = {
            "run_id": paths.run_id,
            "task": config.task.name,
            "skill": config.skill.id,
            "step": step,
            "checkpoint_step": step,
            "split_name": config.dataset.val_split,
            "selection_metric_name": selection_metric,
            "selection_metric_value": _metric_value(metrics, selection_metric),
            "metrics": metrics,
            "metrics_json": str(metrics_dir / "metrics.json"),
            "predictions_jsonl": str(metrics_dir / "predictions.jsonl"),
            "job_json": str(metrics_dir / "job.json"),
        }
        write_json(paths.latest_checkpoint_json, {"status": "latest", **pointer_payload})
        write_json(paths.best_checkpoint_json, {"status": "best", **pointer_payload})
    summary = {
        "status": "succeeded",
        "finetune_id": f"mock-{paths.run_id}",
        "selection_metric_name": selection_metric,
        "selection_metric_value": best_value,
        "best_checkpoint_step": max(save_every, total_steps),
        "latest_checkpoint_step": max(save_every, total_steps),
        "baseline_metrics": baseline_metrics,
        "best_metrics": best_metrics,
        "latest_metrics": best_metrics,
        "final_eval": {config.dataset.val_split: best_metrics},
        "supports_checkpoint_replay": True,
        "checkpoint_eval_note": "Mock backend stores checkpoint metrics locally.",
    }
    write_json(paths.train_summary_json, summary)
    return BackendRunResult(
        status="succeeded",
        finetune_id=f"mock-{paths.run_id}",
        selection_metric_name=selection_metric,
        selection_metric_value=best_value,
        metrics=best_metrics,
        summary=summary,
        artifact_paths=paths.artifact_paths(),
        rendered_backend_config=rendered_backend_config,
        baseline_metrics=baseline_metrics,
    )


def _run_mock_baseline(
    config: FrameworkConfig,
    paths: RunPaths,
    metric_policy: MetricPolicy,
    rendered_backend_config: dict[str, Any],
) -> BackendRunResult:
    metrics = _mock_metrics(config.skill.id, metric_policy.selection_metric, 0.32 if config.skill.id != "query" else 0.40)
    write_benchmark_metrics(paths, metrics)
    paths.predictions_jsonl.write_text('{"sample_id":"mock-1","status":"ok"}\n', encoding="utf-8")
    return BackendRunResult(
        status="succeeded",
        finetune_id="",
        selection_metric_name=metric_policy.selection_metric,
        selection_metric_value=_metric_value(metrics, metric_policy.selection_metric),
        metrics=metrics,
        summary={"mode": "baseline"},
        artifact_paths=paths.artifact_paths(),
        rendered_backend_config=rendered_backend_config,
        baseline_metrics={},
    )


def _mock_metrics(skill: str, selection_metric: str, value: float) -> dict[str, Any]:
    clipped = max(0.0, min(0.99, float(value)))
    if skill == "query":
        metrics = {
            "reward_mean": clipped,
            "accuracy": max(0.0, clipped - 0.08),
            "balanced_accuracy": max(0.0, clipped - 0.05),
            "macro_f1": max(0.0, clipped - 0.04),
            "micro_f1": max(0.0, clipped - 0.03),
            "json_parse_rate": min(1.0, clipped + 0.1),
        }
    elif skill == "point":
        metrics = {
            "reward_mean": clipped,
            "eval_f1": clipped,
            "eval_precision": max(0.0, clipped - 0.05),
            "eval_recall": min(0.99, clipped + 0.03),
            "eval_tp": int(10 + clipped * 10),
            "eval_fp": max(0, int(4 - clipped * 3)),
            "eval_fn": max(0, int(5 - clipped * 4)),
        }
    else:
        metrics = {
            "reward_mean": clipped,
            "eval_f1": clipped,
            "eval_f1_macro": max(0.0, clipped - 0.02),
            "eval_miou": max(0.0, clipped - 0.04),
            "eval_tp": int(12 + clipped * 10),
            "eval_fp": max(0, int(4 - clipped * 3)),
            "eval_fn": max(0, int(5 - clipped * 4)),
        }
    if selection_metric not in metrics:
        metrics[selection_metric] = clipped
    return metrics


def _load_saved_checkpoint_eval(paths: RunPaths, checkpoint_step: int) -> Optional[dict[str, Any]]:
    target = f"step{int(checkpoint_step):06d}_"
    for metrics_path in paths.async_eval_dir.rglob("metrics.json"):
        if target in metrics_path.parent.name:
            metrics = read_json(metrics_path, default={}) or {}
            return {
                "metrics": metrics,
                "metrics_path": metrics_path,
                "predictions_path": metrics_path.parent / "predictions.jsonl",
            }
    return None
