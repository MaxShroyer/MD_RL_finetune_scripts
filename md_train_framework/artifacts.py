from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from md_train_framework.config import FrameworkConfig
from md_train_framework.utils import append_jsonl, ensure_dir, now_utc_iso, random_suffix, slugify, stable_json_hash, write_json


@dataclass(frozen=True)
class RunPaths:
    run_id: str
    run_dir: Path
    config_json: Path
    rendered_backend_config_json: Path
    eval_history_jsonl: Path
    train_summary_json: Path
    best_checkpoint_json: Path
    latest_checkpoint_json: Path
    benchmark_metrics_json: Path
    predictions_jsonl: Path
    failure_log_jsonl: Path
    stdout_log: Path
    async_eval_dir: Path

    def artifact_paths(self) -> dict[str, str]:
        return {
            "run_dir": str(self.run_dir),
            "run_config_json": str(self.config_json),
            "rendered_backend_config_json": str(self.rendered_backend_config_json),
            "eval_history_jsonl": str(self.eval_history_jsonl),
            "train_summary_json": str(self.train_summary_json),
            "best_checkpoint_json": str(self.best_checkpoint_json),
            "latest_checkpoint_json": str(self.latest_checkpoint_json),
            "benchmark_metrics_json": str(self.benchmark_metrics_json),
            "predictions_jsonl": str(self.predictions_jsonl),
            "failure_log_jsonl": str(self.failure_log_jsonl),
            "stdout_log": str(self.stdout_log),
            "async_eval_dir": str(self.async_eval_dir),
        }


def create_run_paths(config: FrameworkConfig, *, suffix: str = "") -> RunPaths:
    run_root = config.resolved_path(config.logging.run_root)
    task_slug = slugify(config.task.name, default="task")
    run_id = f"{slugify(config.skill.id)}-{task_slug}-{random_suffix()}"
    if suffix:
        run_id = f"{run_id}-{slugify(suffix)}"
    run_dir = ensure_dir(run_root / task_slug / run_id)
    return RunPaths(
        run_id=run_id,
        run_dir=run_dir,
        config_json=run_dir / "run_config.json",
        rendered_backend_config_json=run_dir / "backend_rendered_config.json",
        eval_history_jsonl=run_dir / "eval_history.jsonl",
        train_summary_json=run_dir / "train_summary.json",
        best_checkpoint_json=run_dir / "best_checkpoint.json",
        latest_checkpoint_json=run_dir / "latest_checkpoint.json",
        benchmark_metrics_json=run_dir / "benchmark_metrics.json",
        predictions_jsonl=run_dir / "predictions.jsonl",
        failure_log_jsonl=run_dir / "failure_log.jsonl",
        stdout_log=run_dir / "stdout.log",
        async_eval_dir=run_dir / "async_checkpoint_eval",
    )


def write_run_config(paths: RunPaths, config: FrameworkConfig, *, rendered_backend_config: Mapping[str, Any] | None = None) -> None:
    write_json(paths.config_json, config.to_dict(include_meta=False))
    if rendered_backend_config is not None:
        write_json(paths.rendered_backend_config_json, dict(rendered_backend_config))


def record_failure(paths: RunPaths, *, stage: str, message: str, payload: Mapping[str, Any] | None = None) -> None:
    record = {
        "logged_at": now_utc_iso(),
        "stage": str(stage),
        "message": str(message),
    }
    if payload:
        record.update(dict(payload))
    append_jsonl(paths.failure_log_jsonl, record)


def write_eval_event(paths: RunPaths, payload: Mapping[str, Any]) -> None:
    append_jsonl(paths.eval_history_jsonl, dict(payload))


def write_train_summary(paths: RunPaths, payload: Mapping[str, Any]) -> None:
    write_json(paths.train_summary_json, dict(payload))


def write_benchmark_metrics(paths: RunPaths, payload: Mapping[str, Any]) -> None:
    write_json(paths.benchmark_metrics_json, dict(payload))


def dataset_fingerprint_from_summary(summary: Mapping[str, Any]) -> str:
    return stable_json_hash(summary)
