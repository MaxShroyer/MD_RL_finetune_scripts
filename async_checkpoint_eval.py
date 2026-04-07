from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional


@dataclass
class DispatchHandle:
    trainer: str
    finetune_id: str
    checkpoint_step: int
    selection_metric: str
    job_id: str
    job_dir: Path
    job_json_path: Path
    metrics_json_path: Path
    predictions_jsonl_path: Path
    stdout_log_path: Path
    command: list[str]
    metadata: dict[str, Any]
    process: subprocess.Popen[str]
    started_at: float


@dataclass(frozen=True)
class CheckpointEvalResult:
    trainer: str
    finetune_id: str
    checkpoint_step: int
    selection_metric: str
    status: str
    returncode: int
    job_dir: Path
    job_json_path: Path
    metrics_json_path: Path
    predictions_jsonl_path: Path
    stdout_log_path: Path
    command: list[str]
    started_at: float
    completed_at: float
    metrics_payload: Optional[dict[str, Any]]
    metadata: dict[str, Any]


def _safe_component(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return "unknown"
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text)


def _job_id(*, checkpoint_step: int) -> str:
    return f"step{int(checkpoint_step):06d}_{time.strftime('%Y%m%d_%H%M%S')}"


def default_async_checkpoint_eval_dir(
    *,
    base_dir: str,
    trainer: str,
    finetune_id: str,
) -> Path:
    return Path(str(base_dir)).expanduser().resolve() / _safe_component(trainer) / _safe_component(finetune_id)


def _write_job_metadata(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_metrics_payload(metrics_path: Path) -> Optional[dict[str, Any]]:
    if not metrics_path.exists():
        return None
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def dispatch_checkpoint_eval(
    *,
    trainer: str,
    finetune_id: str,
    checkpoint_step: int,
    selection_metric: str,
    base_dir: str,
    command: Optional[list[str]] = None,
    command_builder: Optional[Callable[[Path, Path, Path], list[str]]] = None,
    metadata: Optional[dict[str, Any]] = None,
    env_overrides: Optional[dict[str, str]] = None,
    max_inflight: int,
    inflight_jobs: Iterable[DispatchHandle],
) -> Optional[DispatchHandle]:
    active_jobs = [job for job in inflight_jobs if job.process.poll() is None]
    if len(active_jobs) >= max(1, int(max_inflight)):
        return None

    started_at = time.time()
    job_id = _job_id(checkpoint_step=int(checkpoint_step))
    job_dir = default_async_checkpoint_eval_dir(
        base_dir=base_dir,
        trainer=trainer,
        finetune_id=finetune_id,
    ) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    job_json_path = job_dir / "job.json"
    metrics_json_path = job_dir / "metrics.json"
    predictions_jsonl_path = job_dir / "predictions.jsonl"
    stdout_log_path = job_dir / "stdout.log"
    resolved_command = (
        list(command_builder(metrics_json_path, predictions_jsonl_path, stdout_log_path))
        if command_builder is not None
        else list(command or [])
    )
    if not resolved_command:
        raise ValueError("dispatch_checkpoint_eval requires command or command_builder")
    payload = {
        "trainer": str(trainer),
        "finetune_id": str(finetune_id),
        "checkpoint_step": int(checkpoint_step),
        "selection_metric": str(selection_metric),
        "status": "running",
        "command": list(resolved_command),
        "command_shell": " ".join(shlex.quote(part) for part in resolved_command),
        "started_at": float(started_at),
        "completed_at": None,
        "returncode": None,
        "job_id": job_id,
        "job_dir": str(job_dir),
        "metrics_json": str(metrics_json_path),
        "predictions_jsonl": str(predictions_jsonl_path),
        "stdout_log": str(stdout_log_path),
    }
    if metadata:
        payload.update(metadata)
    _write_job_metadata(job_json_path, payload)

    env = dict(os.environ)
    if env_overrides:
        env.update({str(key): str(value) for key, value in env_overrides.items()})

    log_handle = stdout_log_path.open("a", encoding="utf-8")
    try:
        process = subprocess.Popen(
            resolved_command,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(Path.cwd()),
            env=env,
            start_new_session=True,
        )
    finally:
        log_handle.close()

    return DispatchHandle(
        trainer=str(trainer),
        finetune_id=str(finetune_id),
        checkpoint_step=int(checkpoint_step),
        selection_metric=str(selection_metric),
        job_id=job_id,
        job_dir=job_dir,
        job_json_path=job_json_path,
        metrics_json_path=metrics_json_path,
        predictions_jsonl_path=predictions_jsonl_path,
        stdout_log_path=stdout_log_path,
        command=list(resolved_command),
        metadata=dict(metadata or {}),
        process=process,
        started_at=float(started_at),
    )


def _finalize_job(handle: DispatchHandle) -> Optional[CheckpointEvalResult]:
    returncode = handle.process.poll()
    if returncode is None:
        return None
    completed_at = time.time()
    metrics_payload = _load_metrics_payload(handle.metrics_json_path)
    status = "succeeded" if returncode == 0 and metrics_payload is not None else "failed"
    payload = {
        "trainer": handle.trainer,
        "finetune_id": handle.finetune_id,
        "checkpoint_step": int(handle.checkpoint_step),
        "selection_metric": handle.selection_metric,
        "status": status,
        "command": list(handle.command),
        "command_shell": " ".join(shlex.quote(part) for part in handle.command),
        "started_at": float(handle.started_at),
        "completed_at": float(completed_at),
        "returncode": int(returncode),
        "job_id": handle.job_id,
        "job_dir": str(handle.job_dir),
        "metrics_json": str(handle.metrics_json_path),
        "predictions_jsonl": str(handle.predictions_jsonl_path),
        "stdout_log": str(handle.stdout_log_path),
    }
    payload.update(handle.metadata)
    if metrics_payload is not None:
        payload["metrics_payload"] = metrics_payload
    _write_job_metadata(handle.job_json_path, payload)
    return CheckpointEvalResult(
        trainer=handle.trainer,
        finetune_id=handle.finetune_id,
        checkpoint_step=int(handle.checkpoint_step),
        selection_metric=handle.selection_metric,
        status=status,
        returncode=int(returncode),
        job_dir=handle.job_dir,
        job_json_path=handle.job_json_path,
        metrics_json_path=handle.metrics_json_path,
        predictions_jsonl_path=handle.predictions_jsonl_path,
        stdout_log_path=handle.stdout_log_path,
        command=list(handle.command),
        started_at=float(handle.started_at),
        completed_at=float(completed_at),
        metrics_payload=metrics_payload,
        metadata=dict(handle.metadata),
    )


def poll_checkpoint_eval_jobs(
    jobs: list[DispatchHandle],
) -> tuple[list[DispatchHandle], list[CheckpointEvalResult]]:
    remaining: list[DispatchHandle] = []
    completed: list[CheckpointEvalResult] = []
    for handle in jobs:
        result = _finalize_job(handle)
        if result is None:
            remaining.append(handle)
            continue
        completed.append(result)
    return remaining, completed


def drain_checkpoint_eval_jobs(
    jobs: list[DispatchHandle],
    *,
    poll_interval_s: float = 1.0,
) -> list[CheckpointEvalResult]:
    remaining = list(jobs)
    completed: list[CheckpointEvalResult] = []
    while remaining:
        next_remaining, just_finished = poll_checkpoint_eval_jobs(remaining)
        completed.extend(just_finished)
        if not next_remaining:
            break
        time.sleep(max(0.1, float(poll_interval_s)))
        remaining = next_remaining
    return completed
