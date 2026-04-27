from __future__ import annotations

import json
import random
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from md_train_framework.artifacts import RunPaths, write_eval_event, write_train_summary
from md_train_framework.config import FrameworkConfig, PhaseSpec
from md_train_framework.datasets import create_dataset_adapter
from md_train_framework.metrics import MetricPolicy
from md_train_framework.rewards import RewardPreset
from md_train_framework.samples import DetectSample, NormalizedSample, PointSample, QuerySample, normalize_row
from md_train_framework.scoring import (
    DetectScore,
    PointScore,
    QueryScore,
    aggregate_detect,
    aggregate_point,
    aggregate_query,
    score_detect,
    score_point,
    score_query,
)
from md_train_framework.utils import append_jsonl, now_utc_iso, slugify, write_json
from md_train_framework.wandb_logger import WandbLogger
from tuna_sdk import PointSFTTarget, RetryConfig, TunaAPIError, TunaClient, TunaNetworkError, TrainStepGroup
from tuna_sdk.retry import compute_backoff_delay

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover
    load_dotenv = None  # type: ignore[assignment]


TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504, 520, 524}


def _split_seed_offset(split_name: str) -> int:
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(str(split_name)))


@dataclass
class StepCheckpointEval:
    step: int
    checkpoint_step: Optional[int]
    split_name: str
    metrics: dict[str, Any]
    predictions_path: Path
    metrics_path: Path
    job_json_path: Path


@dataclass
class TrainOutcome:
    finetune_id: str
    selection_metric_name: str
    selection_metric_value: float
    metrics: dict[str, Any]
    summary: dict[str, Any]
    artifact_paths: dict[str, str]
    baseline_metrics: dict[str, Any] = field(default_factory=dict)


class BaseTrainer:
    def __init__(
        self,
        *,
        config: FrameworkConfig,
        reward_preset: RewardPreset,
        metric_policy: MetricPolicy,
        paths: RunPaths,
        finetune: Any,
        rng: random.Random,
        logger: WandbLogger,
    ) -> None:
        self.config = config
        self.reward_preset = reward_preset
        self.metric_policy = metric_policy
        self.paths = paths
        self.finetune = finetune
        self.rng = rng
        self.logger = logger
        self.rl_update_count = 0
        self.initial_finetune_id = str(getattr(finetune, "finetune_id", "") or "")
        self.phase_finetune_ids: list[dict[str, str]] = []
        self.guard_events: list[dict[str, Any]] = []
        self._fixed_eval_indices_by_split: dict[str, tuple[int, ...]] = {}
        self._current_rl_reward_stats: dict[str, float] = {}
        self.train_samples, self.val_samples, self.test_samples = self._load_samples()
        self.off_policy_buffer: deque[TrainStepGroup] = deque(
            maxlen=max(1, int(self._override("off_policy_buffer_size", 128)))
        )

    def _load_samples(self) -> tuple[list[NormalizedSample], list[NormalizedSample], list[NormalizedSample]]:
        adapter = create_dataset_adapter(self.config)
        return (
            self._materialize_split(adapter, self.config.dataset.train_split),
            self._materialize_split(adapter, self.config.dataset.val_split),
            self._materialize_split(adapter, self.config.dataset.test_split),
        )

    def _materialize_split(self, adapter: Any, split_name: str) -> list[NormalizedSample]:
        samples: list[NormalizedSample] = []
        for index, row in enumerate(adapter.iter_split(split_name), start=1):
            if not isinstance(row, dict):
                continue
            try:
                samples.append(normalize_row(self.config, split_name, row, index=index))
            except Exception as exc:
                self._record_failure(
                    stage="dataset",
                    message=f"sample normalization failed: {type(exc).__name__}: {exc}",
                    payload={"split": split_name, "row_index": index},
                )
        return samples

    def run(self) -> TrainOutcome:
        baseline_metrics: dict[str, Any] = {}
        best_metric_value = 0.0
        best_metrics: dict[str, Any] = {}
        best_checkpoint_step: Optional[int] = None
        latest_checkpoint_step: Optional[int] = None
        latest_metrics: dict[str, Any] = {}
        global_step = 0
        checkpoint_evals: list[StepCheckpointEval] = []
        final_eval: dict[str, Any] = {}
        status = "running"
        stop_reason = ""
        failure_message = ""
        exc_to_raise: Optional[Exception] = None

        try:
            baseline_event = self.evaluate_current(
                split_name=self.config.dataset.val_split,
                log_step=0,
                stage="baseline",
                checkpoint_step=None,
            )
            baseline_metrics = dict(baseline_event.metrics)
            latest_metrics = dict(baseline_metrics)
            best_metric_value = self._metric_value(baseline_metrics)
            best_metrics = dict(baseline_metrics)
            self._write_checkpoint_pointer(kind="best", event=baseline_event)
            self._write_checkpoint_pointer(kind="latest", event=baseline_event)
            self._write_progress_summary(
                status=status,
                baseline_metrics=baseline_metrics,
                best_metrics=best_metrics,
                best_metric_value=best_metric_value,
                best_checkpoint_step=best_checkpoint_step,
                latest_metrics=latest_metrics,
                latest_checkpoint_step=latest_checkpoint_step,
                final_eval=final_eval,
                global_step=global_step,
                checkpoint_evals=checkpoint_evals,
                stop_reason=stop_reason,
                failure_message=failure_message,
            )

            for phase in self.config.phases:
                self._activate_phase_finetune(phase)
                for _ in range(max(0, int(phase.steps))):
                    global_step += 1
                    train_payload = self._run_phase_step(phase=phase, global_step=global_step)
                    self.logger.log(train_payload, step=global_step)
                    if self.config.eval.save_every <= 0 or global_step % int(self.config.eval.save_every) != 0:
                        continue
                    checkpoint_step = self._save_checkpoint(global_step)
                    if checkpoint_step is None:
                        continue
                    latest_checkpoint_step = checkpoint_step
                    checkpoint_eval = self.evaluate_current(
                        split_name=self.config.dataset.val_split,
                        log_step=global_step,
                        stage=phase.mode,
                        checkpoint_step=checkpoint_step,
                    )
                    checkpoint_evals.append(checkpoint_eval)
                    latest_metrics = dict(checkpoint_eval.metrics)
                    self._write_checkpoint_pointer(kind="latest", event=checkpoint_eval)
                    metric_value = self._metric_value(checkpoint_eval.metrics)
                    guard_decision = self._checkpoint_guard_decision(
                        phase=phase,
                        global_step=global_step,
                        checkpoint_event=checkpoint_eval,
                        baseline_metrics=baseline_metrics,
                        best_metrics=best_metrics,
                    )
                    if guard_decision is not None:
                        status = "stopped_early"
                        stop_reason = str(guard_decision.get("message", "checkpoint guard triggered"))
                        self.guard_events.append(dict(guard_decision))
                        self._record_failure(
                            stage="guard",
                            message=stop_reason,
                            payload={
                                "step": global_step,
                                "checkpoint_step": checkpoint_step,
                                **{k: v for k, v in guard_decision.items() if k != "message"},
                            },
                        )
                        self._append_quarantine(
                            reason=stop_reason,
                            payload={
                                "step": global_step,
                                "checkpoint_step": checkpoint_step,
                                "selection_metric_name": self.metric_policy.selection_metric,
                                "selection_metric_value": metric_value,
                                **{k: v for k, v in guard_decision.items() if k != "message"},
                            },
                        )
                        break
                    if metric_value >= best_metric_value:
                        best_metric_value = metric_value
                        best_metrics = dict(checkpoint_eval.metrics)
                        best_checkpoint_step = checkpoint_step
                        self._write_checkpoint_pointer(kind="best", event=checkpoint_eval)
                    self._write_progress_summary(
                        status=status,
                        baseline_metrics=baseline_metrics,
                        best_metrics=best_metrics,
                        best_metric_value=best_metric_value,
                        best_checkpoint_step=best_checkpoint_step,
                        latest_metrics=latest_metrics,
                        latest_checkpoint_step=latest_checkpoint_step,
                        final_eval=final_eval,
                        global_step=global_step,
                        checkpoint_evals=checkpoint_evals,
                        stop_reason=stop_reason,
                        failure_message=failure_message,
                    )
                if status == "stopped_early":
                    break

            if status == "running":
                for split_name in self._final_eval_splits():
                    split_samples = self._split_samples(split_name)
                    if not split_samples:
                        continue
                    event = self.evaluate_current(
                        split_name=split_name,
                        log_step=global_step,
                        stage="final",
                        checkpoint_step=best_checkpoint_step,
                    )
                    final_eval[split_name] = dict(event.metrics)
                status = "succeeded"
        except Exception as exc:
            status = "failed"
            failure_message = f"{type(exc).__name__}: {exc}"
            exc_to_raise = exc
        finally:
            summary = self._build_summary(
                status=status,
                baseline_metrics=baseline_metrics,
                best_metrics=best_metrics,
                best_metric_value=best_metric_value,
                best_checkpoint_step=best_checkpoint_step,
                latest_metrics=latest_metrics,
                latest_checkpoint_step=latest_checkpoint_step,
                final_eval=final_eval,
                global_step=global_step,
                checkpoint_evals=checkpoint_evals,
                stop_reason=stop_reason,
                failure_message=failure_message,
            )
            write_train_summary(self.paths, summary)
            self.logger.summary["status"] = status
            self.logger.summary["selection_metric_name"] = self.metric_policy.selection_metric
            self.logger.summary["selection_metric_value"] = float(best_metric_value)
            self.logger.summary["finetune_id"] = self.finetune.finetune_id
            self.logger.summary["initial_finetune_id"] = self.initial_finetune_id
            self.logger.summary["final_finetune_id"] = self.finetune.finetune_id
            if stop_reason:
                self.logger.summary["early_stop_reason"] = stop_reason
            if failure_message:
                self.logger.summary["failure_message"] = failure_message
            self.logger.finish()

        if exc_to_raise is not None:
            raise exc_to_raise

        return TrainOutcome(
            finetune_id=self.finetune.finetune_id,
            selection_metric_name=self.metric_policy.selection_metric,
            selection_metric_value=float(best_metric_value),
            metrics=best_metrics or baseline_metrics,
            summary=summary,
            artifact_paths=self.paths.artifact_paths(),
            baseline_metrics=baseline_metrics,
        )

    def evaluate_current(
        self,
        *,
        split_name: str,
        log_step: int,
        stage: str,
        checkpoint_step: Optional[int],
    ) -> StepCheckpointEval:
        samples = self._split_samples(split_name)
        sampled = self._sample_eval_subset(split_name, samples)
        metrics, predictions = self._evaluate_samples(sampled)
        event_payload = {
            "step": int(log_step),
            "checkpoint_step": checkpoint_step,
            "stage": str(stage),
            "split": str(split_name),
            "metrics": metrics,
        }
        write_eval_event(self.paths, event_payload)
        metrics_dir = self.paths.async_eval_dir / self.config.skill.id / self.paths.run_id / self._job_name(checkpoint_step, log_step)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = metrics_dir / "metrics.json"
        predictions_path = metrics_dir / "predictions.jsonl"
        job_json_path = metrics_dir / "job.json"
        write_json(metrics_path, metrics)
        with predictions_path.open("w", encoding="utf-8") as handle:
            for item in predictions:
                handle.write(json.dumps(item, ensure_ascii=True, sort_keys=True))
                handle.write("\n")
        write_json(
            job_json_path,
            {
                "status": "completed",
                "logged_at": now_utc_iso(),
                "step": int(log_step),
                "checkpoint_step": checkpoint_step,
                "split_name": str(split_name),
                "metrics_json": str(metrics_path),
                "predictions_jsonl": str(predictions_path),
            },
        )
        log_payload = self.metric_policy.filter_metrics(metrics)
        log_payload["split"] = str(split_name)
        log_payload["stage"] = str(stage)
        self.logger.log(log_payload, step=int(log_step))
        return StepCheckpointEval(
            step=int(log_step),
            checkpoint_step=int(checkpoint_step) if checkpoint_step is not None else None,
            split_name=str(split_name),
            metrics=metrics,
            predictions_path=predictions_path,
            metrics_path=metrics_path,
            job_json_path=job_json_path,
        )

    def _final_eval_splits(self) -> list[str]:
        configured = [str(item).strip() for item in self.config.eval.final_splits if str(item).strip()]
        if configured:
            ordered = configured
        else:
            ordered = [self.config.dataset.val_split, self.config.dataset.test_split]
        seen: set[str] = set()
        output: list[str] = []
        for split_name in ordered:
            if split_name not in seen:
                seen.add(split_name)
                output.append(split_name)
        return output

    def _build_summary(
        self,
        *,
        status: str,
        baseline_metrics: dict[str, Any],
        best_metrics: dict[str, Any],
        best_metric_value: float,
        best_checkpoint_step: Optional[int],
        latest_metrics: dict[str, Any],
        latest_checkpoint_step: Optional[int],
        final_eval: dict[str, Any],
        global_step: int,
        checkpoint_evals: list[StepCheckpointEval],
        stop_reason: str,
        failure_message: str,
    ) -> dict[str, Any]:
        summary = {
            "status": str(status),
            "finetune_id": self.finetune.finetune_id,
            "initial_finetune_id": self.initial_finetune_id,
            "final_finetune_id": self.finetune.finetune_id,
            "phase_finetune_ids": list(self.phase_finetune_ids),
            "finetune_id_transitioned": len({item["finetune_id"] for item in self.phase_finetune_ids if item.get("finetune_id")}) > 1,
            "mode": self.config.mode,
            "skill": self.config.skill.id,
            "task": self.config.task.name,
            "reward_preset": self.reward_preset.id,
            "selection_metric_name": self.metric_policy.selection_metric,
            "selection_metric_value": float(best_metric_value),
            "best_checkpoint_step": best_checkpoint_step,
            "latest_checkpoint_step": latest_checkpoint_step,
            "baseline_metrics": dict(baseline_metrics),
            "best_metrics": dict(best_metrics),
            "latest_metrics": dict(latest_metrics),
            "final_eval": dict(final_eval),
            "global_step": int(global_step),
            "async_checkpoint_eval_count": len(checkpoint_evals),
            "guard_events": list(self.guard_events),
            "stopped_early": bool(status == "stopped_early"),
            "early_stop_reason": str(stop_reason),
            "failure_message": str(failure_message),
            "logged_at": now_utc_iso(),
            "supports_checkpoint_replay": False,
            "checkpoint_eval_note": "Checkpoint metrics are captured at save time in tuna-sdk-only mode.",
        }
        summary.update(
            self._summary_extras(
                baseline_metrics=baseline_metrics,
                best_metrics=best_metrics,
                latest_metrics=latest_metrics,
                global_step=global_step,
            )
        )
        return summary

    def _write_progress_summary(
        self,
        *,
        status: str,
        baseline_metrics: dict[str, Any],
        best_metrics: dict[str, Any],
        best_metric_value: float,
        best_checkpoint_step: Optional[int],
        latest_metrics: dict[str, Any],
        latest_checkpoint_step: Optional[int],
        final_eval: dict[str, Any],
        global_step: int,
        checkpoint_evals: list[StepCheckpointEval],
        stop_reason: str,
        failure_message: str,
    ) -> None:
        write_train_summary(
            self.paths,
            self._build_summary(
                status=status,
                baseline_metrics=baseline_metrics,
                best_metrics=best_metrics,
                best_metric_value=best_metric_value,
                best_checkpoint_step=best_checkpoint_step,
                latest_metrics=latest_metrics,
                latest_checkpoint_step=latest_checkpoint_step,
                final_eval=final_eval,
                global_step=global_step,
                checkpoint_evals=checkpoint_evals,
                stop_reason=stop_reason,
                failure_message=failure_message,
            ),
        )

    def _write_checkpoint_pointer(self, *, kind: str, event: StepCheckpointEval) -> None:
        path = self.paths.best_checkpoint_json if kind == "best" else self.paths.latest_checkpoint_json
        write_json(
            path,
            {
                "run_id": self.paths.run_id,
                "task": self.config.task.name,
                "skill": self.config.skill.id,
                "status": kind,
                "finetune_id": str(getattr(self.finetune, "finetune_id", "") or ""),
                "step": int(event.step),
                "checkpoint_step": event.checkpoint_step,
                "split_name": str(event.split_name),
                "selection_metric_name": self.metric_policy.selection_metric,
                "selection_metric_value": self._metric_value(event.metrics),
                "metrics": dict(event.metrics),
                "metrics_json": str(event.metrics_path),
                "predictions_jsonl": str(event.predictions_path),
                "job_json": str(event.job_json_path),
                "logged_at": now_utc_iso(),
            },
        )

    def _append_quarantine(self, *, reason: str, payload: Optional[dict[str, Any]] = None) -> None:
        record = {
            "logged_at": now_utc_iso(),
            "run_id": self.paths.run_id,
            "task": self.config.task.name,
            "skill": self.config.skill.id,
            "reason": str(reason),
        }
        if payload:
            record.update(payload)
        append_jsonl(self.config.resolved_path(self.config.recovery.quarantine_path), record)

    def _checkpoint_guard_decision(
        self,
        *,
        phase: PhaseSpec,
        global_step: int,
        checkpoint_event: StepCheckpointEval,
        baseline_metrics: dict[str, Any],
        best_metrics: dict[str, Any],
    ) -> Optional[dict[str, Any]]:
        profile_decision = self._profile_checkpoint_guard_decision(
            phase=phase,
            global_step=global_step,
            checkpoint_event=checkpoint_event,
            baseline_metrics=baseline_metrics,
            best_metrics=best_metrics,
        )
        if profile_decision is not None:
            return profile_decision
        metrics = checkpoint_event.metrics
        if self.config.skill.id == "detect" and bool(self._override("detect_f1_guard_enabled", True)):
            baseline_f1 = float(baseline_metrics.get("eval_f1", 0.0) or 0.0)
            baseline_miou = float(baseline_metrics.get("eval_miou", 0.0) or 0.0)
            current_f1 = float(metrics.get("eval_f1", 0.0) or 0.0)
            current_miou = float(metrics.get("eval_miou", 0.0) or 0.0)
            min_ratio = max(0.0, float(self._override("detect_f1_guard_min_ratio", 0.25)))
            min_gain = max(0.0, float(self._override("detect_miou_guard_min_gain", 0.05)))
            if baseline_f1 > 0.0 and current_miou >= (baseline_miou + min_gain) and current_f1 <= (baseline_f1 * min_ratio):
                return {
                    "guard": "detect_f1_collapse",
                    "message": (
                        f"detect guard triggered at step {global_step}: "
                        f"eval_f1 dropped from {baseline_f1:.4f} to {current_f1:.4f} "
                        f"while eval_miou rose from {baseline_miou:.4f} to {current_miou:.4f}"
                    ),
                    "baseline_eval_f1": baseline_f1,
                    "baseline_eval_miou": baseline_miou,
                    "eval_f1": current_f1,
                    "eval_miou": current_miou,
                }
        raw_parse_threshold = self._override("min_json_parse_rate", "")
        if str(raw_parse_threshold).strip():
            threshold = max(0.0, min(1.0, float(raw_parse_threshold)))
            min_step = max(0, int(self._override("json_parse_guard_min_step", 0)))
            current_parse = float(metrics.get("eval_json_parse_rate", metrics.get("json_parse_rate", 1.0)) or 0.0)
            if global_step >= min_step and current_parse < threshold:
                return {
                    "guard": "json_parse_collapse",
                    "message": (
                        f"json parse guard triggered at step {global_step}: "
                        f"eval_json_parse_rate={current_parse:.4f} fell below {threshold:.4f}"
                    ),
                    "eval_json_parse_rate": current_parse,
                    "min_json_parse_rate": threshold,
                }
        return None

    def _profile_checkpoint_guard_decision(
        self,
        *,
        phase: PhaseSpec,
        global_step: int,
        checkpoint_event: StepCheckpointEval,
        baseline_metrics: dict[str, Any],
        best_metrics: dict[str, Any],
    ) -> Optional[dict[str, Any]]:
        del phase, global_step, checkpoint_event, baseline_metrics, best_metrics
        return None

    def _summary_extras(
        self,
        *,
        baseline_metrics: dict[str, Any],
        best_metrics: dict[str, Any],
        latest_metrics: dict[str, Any],
        global_step: int,
    ) -> dict[str, Any]:
        del baseline_metrics, best_metrics, latest_metrics, global_step
        return {}

    def _run_phase_step(self, *, phase: PhaseSpec, global_step: int) -> dict[str, Any]:
        if phase.mode == "sft":
            return self._run_sft_step(phase=phase, global_step=global_step)
        return self._run_rl_step(phase=phase, global_step=global_step)

    def _activate_phase_finetune(self, phase: PhaseSpec) -> None:
        target_finetune_id = self._phase_finetune_id(phase)
        current_finetune_id = str(getattr(self.finetune, "finetune_id", "") or "")
        if target_finetune_id and target_finetune_id != current_finetune_id:
            client = getattr(self.finetune, "_client", None)
            if client is None:
                raise ValueError(
                    f"phase finetune override requested for {phase.name or phase.mode}, "
                    "but the current finetune handle cannot resolve a new finetune object"
                )
            self.finetune = client.get_finetune(target_finetune_id)
            current_finetune_id = str(getattr(self.finetune, "finetune_id", "") or target_finetune_id)
        phase_name = str(phase.name or phase.mode or f"phase_{len(self.phase_finetune_ids) + 1}")
        record = {"phase": phase_name, "mode": str(phase.mode or ""), "finetune_id": current_finetune_id}
        if not self.phase_finetune_ids or self.phase_finetune_ids[-1] != record:
            self.phase_finetune_ids.append(record)

    def _phase_override_value(self, phase: PhaseSpec, *suffixes: str) -> str:
        for suffix in suffixes:
            value = str(phase.extra.get(suffix, "")).strip()
            if value:
                return value
        for suffix in suffixes:
            for key in (f"{phase.name}_{suffix}", f"{phase.mode}_{suffix}"):
                value = str(self.config.backend.train_overrides.get(key, "")).strip()
                if value:
                    return value
        return ""

    def _resolve_named_phase_finetune_id(
        self,
        *,
        phase: PhaseSpec,
        finetune_name: str = "",
        finetune_name_prefix: str = "",
    ) -> str:
        client = getattr(self.finetune, "_client", None)
        if client is None or not hasattr(client, "iter_finetunes"):
            raise ValueError(
                f"phase finetune name override requested for {phase.name or phase.mode}, "
                "but the current finetune handle cannot enumerate finetunes"
            )
        target_name = str(finetune_name).strip()
        target_prefix = str(finetune_name_prefix).strip()
        matches: list[Any] = []
        for item in client.iter_finetunes(page_size=100):
            name = str(getattr(item, "name", "") or "")
            if target_name and name == target_name:
                matches.append(item)
            elif target_prefix and name.startswith(target_prefix):
                matches.append(item)
        if not matches:
            needle = target_name or target_prefix
            raise ValueError(
                f"no finetune matched phase override {needle!r} for {phase.name or phase.mode}"
            )
        matches.sort(
            key=lambda item: (
                int(getattr(item, "updated_at_ms", 0) or 0),
                int(getattr(item, "created_at_ms", 0) or 0),
            ),
            reverse=True,
        )
        return str(getattr(matches[0], "finetune_id", "") or "")

    def _phase_finetune_id(self, phase: PhaseSpec) -> str:
        explicit = self._phase_override_value(phase, "finetune_id")
        if explicit:
            return explicit
        finetune_name = self._phase_override_value(phase, "finetune_name")
        if finetune_name:
            return self._resolve_named_phase_finetune_id(phase=phase, finetune_name=finetune_name)
        finetune_name_prefix = self._phase_override_value(phase, "finetune_name_prefix")
        if finetune_name_prefix:
            return self._resolve_named_phase_finetune_id(
                phase=phase,
                finetune_name_prefix=finetune_name_prefix,
            )
        return ""

    def _run_sft_step(self, *, phase: PhaseSpec, global_step: int) -> dict[str, Any]:
        batch = self._sample_train_batch(phase.batch_size)
        groups = [self._sample_to_sft_group(sample, phase=phase) for sample in batch]
        groups = [group for group in groups if group is not None]
        response = self._retry_train_step(groups=groups, lr=phase.lr, context=f"sft step {global_step}")
        return {
            "mode": "sft",
            "group_count": len(groups),
            "batch_size": len(batch),
            "sft_loss": float(response.sft_loss or 0.0),
            "kl": float(response.kl or 0.0),
            "router_kl": float(response.router_kl or 0.0),
            "grad_norm": float(response.grad_norm or 0.0),
            "lr": float(phase.lr),
        }

    def _run_rl_step(self, *, phase: PhaseSpec, global_step: int) -> dict[str, Any]:
        batch = self._sample_train_batch(phase.batch_size)
        requests = [self._sample_to_request(sample, phase=phase, for_eval=False) for sample in batch]
        ground_truths = [self._sample_ground_truth(sample) for sample in batch]
        results = self._retry_rollouts(
            requests=requests,
            num_rollouts=int(phase.num_rollouts or phase.group_size or 4),
            ground_truths=ground_truths,
            context=f"rl step {global_step}",
        )
        new_groups: list[TrainStepGroup] = []
        rewards: list[float] = []
        for sample, result in zip(batch, results, strict=False):
            sample_rewards: list[float] = []
            for rollout in result.rollouts:
                reward = float(self._score_rollout(sample, rollout.output).reward)
                sample_rewards.append(reward)
            new_groups.append(result.to_group(rewards=sample_rewards))
            rewards.extend(sample_rewards)
        reward_mean = sum(rewards) / max(1, len(rewards))
        self._current_rl_reward_stats = {
            "reward_mean": reward_mean,
            "reward_std": _std(rewards),
            "reward_max": max(rewards) if rewards else 0.0,
        }
        groups = list(new_groups)
        groups.extend(self._off_policy_groups(phase=phase, rl_group_count=len(groups)))
        response = self._retry_train_step(groups=groups, lr=phase.lr, context=f"rl train_step {global_step}")
        for group in new_groups:
            self.off_policy_buffer.append(group)
        self.rl_update_count += 1
        return {
            "mode": "rl",
            "group_count": len(groups),
            "batch_size": len(batch),
            "reward_mean": reward_mean,
            "reward_std": self._current_rl_reward_stats["reward_std"],
            "kl": float(response.kl or 0.0),
            "router_kl": float(response.router_kl or 0.0),
            "grad_norm": float(response.grad_norm or 0.0),
            "lr": float(phase.lr),
            "off_policy_enabled": bool(self._override("off_policy", False)),
        }

    def _retry_rollouts(
        self,
        *,
        requests: list[Any],
        num_rollouts: int,
        ground_truths: list[Any],
        context: str,
        max_workers: Optional[int] = None,
    ) -> list[Any]:
        payload_ground_truths = ground_truths if bool(self._override("send_ground_truths", False)) else None
        retries = max(0, int(self.config.recovery.train_step_max_retries))
        configured_max_workers = max(1, min(len(requests), int(max_workers or self._override("max_workers", 4))))
        retry_max_workers = max(1, min(configured_max_workers, int(self._override("retry_rollout_max_workers", 1))))
        for attempt in range(retries + 1):
            attempt_max_workers = configured_max_workers if attempt == 0 else retry_max_workers
            try:
                return self.finetune.rollouts_batch(
                    requests=requests,
                    num_rollouts=int(num_rollouts),
                    ground_truths=payload_ground_truths,
                    max_workers=attempt_max_workers,
                )
            except (TunaAPIError, TunaNetworkError) as exc:
                details = _exception_details(exc)
                if not _is_transient_error(exc) or attempt >= retries:
                    self._record_failure(
                        stage="rollouts",
                        message=f"{context} failed: {type(exc).__name__}: {exc}",
                        payload={"attempt": attempt + 1, "max_workers": attempt_max_workers, **details},
                    )
                    raise
                self._record_failure(
                    stage="rollouts",
                    message=f"{context} transient failure: {type(exc).__name__}: {exc}",
                    payload={"attempt": attempt + 1, "max_workers": attempt_max_workers, **details},
                )
                time.sleep(
                    compute_backoff_delay(
                        attempt,
                        base=max(0.1, float(self.config.recovery.cooldown_s) / 10.0),
                        max_delay=max(1.0, float(self.config.recovery.cooldown_s)),
                        jitter=0.1,
                    )
                )
        raise RuntimeError(f"{context} exhausted retry loop")

    def _retry_train_step(self, *, groups: list[TrainStepGroup], lr: float, context: str) -> Any:
        retries = max(0, int(self.config.recovery.train_step_max_retries))
        for attempt in range(retries + 1):
            try:
                return self.finetune.train_step(groups=groups, lr=float(lr))
            except (TunaAPIError, TunaNetworkError) as exc:
                details = _exception_details(exc)
                if not _is_transient_error(exc) or attempt >= retries:
                    self._record_failure(
                        stage="train_step",
                        message=f"{context} failed: {type(exc).__name__}: {exc}",
                        payload={"attempt": attempt + 1, **details},
                    )
                    raise
                self._record_failure(
                    stage="train_step",
                    message=f"{context} transient failure: {type(exc).__name__}: {exc}",
                    payload={"attempt": attempt + 1, **details},
                )
                time.sleep(
                    compute_backoff_delay(
                        attempt,
                        base=max(0.1, float(self.config.recovery.cooldown_s) / 10.0),
                        max_delay=max(1.0, float(self.config.recovery.cooldown_s)),
                        jitter=0.1,
                    )
                )
        raise RuntimeError(f"{context} exhausted retry loop")

    def _save_checkpoint(self, global_step: int) -> Optional[int]:
        try:
            saved = self.finetune.save_checkpoint()
        except (TunaAPIError, TunaNetworkError) as exc:
            self._record_failure(
                stage="checkpoint",
                message=f"checkpoint save failed: {type(exc).__name__}: {exc}",
                payload={"step": global_step},
            )
            return None
        checkpoint = getattr(saved, "checkpoint", None)
        try:
            return int(getattr(checkpoint, "step", None))
        except (TypeError, ValueError):
            return None

    def _sample_train_batch(self, batch_size: int) -> list[NormalizedSample]:
        if not self.train_samples:
            raise ValueError("training split is empty")
        return [self.rng.choice(self.train_samples) for _ in range(max(1, int(batch_size)))]

    def _sample_eval_subset(self, split_name: str, samples: list[NormalizedSample]) -> list[NormalizedSample]:
        if not samples:
            return []
        fixed_subset_size = max(0, int(self.config.eval.fixed_subset_size))
        if fixed_subset_size > 0:
            limit = min(len(samples), fixed_subset_size)
            if self.config.eval.max_samples is not None:
                limit = min(limit, max(1, int(self.config.eval.max_samples)))
            if len(samples) <= limit:
                return list(samples)
            indices = self._fixed_eval_indices_by_split.get(split_name)
            if indices is None or len(indices) != limit or indices[-1] >= len(samples):
                shuffled = list(range(len(samples)))
                fixed_rng = random.Random(int(self.config.eval.fixed_subset_seed) + _split_seed_offset(split_name))
                fixed_rng.shuffle(shuffled)
                indices = tuple(sorted(shuffled[:limit]))
                self._fixed_eval_indices_by_split[split_name] = indices
            return [samples[index] for index in indices]
        if self.config.eval.max_samples is None:
            return list(samples)
        max_samples = max(1, int(self.config.eval.max_samples))
        if len(samples) <= max_samples:
            return list(samples)
        return list(self.rng.sample(samples, k=max_samples))

    def _split_samples(self, split_name: str) -> list[NormalizedSample]:
        if split_name == self.config.dataset.train_split:
            return self.train_samples
        if split_name == self.config.dataset.val_split:
            return self.val_samples
        if split_name == self.config.dataset.test_split:
            return self.test_samples
        return []

    def _job_name(self, checkpoint_step: Optional[int], log_step: int) -> str:
        target = checkpoint_step if checkpoint_step is not None else log_step
        return f"step{int(target):06d}_{slugify(now_utc_iso().replace(':', '-'))}"

    def _metric_value(self, metrics: dict[str, Any]) -> float:
        key = self.metric_policy.selection_metric
        for candidate in (key, key[5:] if key.startswith("eval_") else f"eval_{key}"):
            try:
                return float(metrics[candidate])
            except (KeyError, TypeError, ValueError):
                continue
        return 0.0

    def _override(self, key: str, default: Any) -> Any:
        return self.config.backend.train_overrides.get(key, self.config.backend.extra.get(key, default))

    def _record_failure(self, *, stage: str, message: str, payload: Optional[dict[str, Any]] = None) -> None:
        record = {
            "logged_at": now_utc_iso(),
            "stage": stage,
            "message": message,
        }
        if payload:
            record.update(payload)
        with self.paths.failure_log_jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True))
            handle.write("\n")

    def _off_policy_groups(self, *, phase: PhaseSpec, rl_group_count: int) -> list[TrainStepGroup]:
        if not bool(self._override("off_policy", False)) or not self.off_policy_buffer:
            return []
        replayable_groups = [group for group in self.off_policy_buffer if str(group.mode or "").strip().lower() == "rl"]
        if not replayable_groups:
            return []
        min_buffer_groups = max(0, int(self._override("off_policy_min_buffer_groups", 0)))
        warmup_steps = max(0, int(self._override("off_policy_warmup_steps", 0)))
        if len(replayable_groups) < min_buffer_groups:
            return []
        if self.rl_update_count < warmup_steps:
            return []
        ratio = float(self._override("off_policy_mix_ratio", 0.25))
        inject_count = max(0, min(len(replayable_groups), int(round(max(1, rl_group_count) * ratio))))
        if inject_count <= 0:
            return []
        return list(self.rng.sample(replayable_groups, k=inject_count))

    def _evaluate_samples(self, samples: list[NormalizedSample]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        if not samples:
            return self._empty_metrics(), []
        outputs = self._predict_samples(samples)
        predictions: list[dict[str, Any]] = []
        scores: list[Any] = []
        for sample, output in zip(samples, outputs, strict=False):
            score = self._score_rollout(sample, output)
            scores.append(score)
            predictions.append(self._prediction_record(sample, output, score))
        return self._aggregate_scores(scores), predictions

    def _predict_samples(self, samples: list[NormalizedSample]) -> list[Any]:
        outputs: list[Any] = []
        max_workers = max(1, min(len(samples), int(self._override("eval_max_workers", self._override("max_workers", 4)))))
        batch_size = max(1, int(self._override("eval_batch_size", max_workers)))
        total = len(samples)
        for start in range(0, total, batch_size):
            chunk = samples[start : start + batch_size]
            requests = [self._sample_to_request(sample, phase=None, for_eval=True) for sample in chunk]
            ground_truths = [self._sample_ground_truth(sample) for sample in chunk]
            results = self._retry_rollouts(
                requests=requests,
                num_rollouts=1,
                ground_truths=ground_truths,
                context=f"eval chunk {start // batch_size + 1}",
                max_workers=max_workers,
            )
            for result in results:
                if not result.rollouts:
                    outputs.append(self._empty_output())
                else:
                    outputs.append(result.rollouts[0].output)
            if total > batch_size:
                print(
                    json.dumps(
                        {
                            "event": "eval_progress",
                            "task": self.config.task.name,
                            "skill": self.config.skill.id,
                            "completed": min(total, start + len(chunk)),
                            "total": total,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        return outputs

    def _empty_metrics(self) -> dict[str, Any]:
        raise NotImplementedError

    def _empty_output(self) -> Any:
        raise NotImplementedError

    def _sample_to_request(self, sample: NormalizedSample, *, phase: Optional[PhaseSpec], for_eval: bool) -> Any:
        raise NotImplementedError

    def _sample_to_sft_group(self, sample: NormalizedSample, *, phase: PhaseSpec) -> Optional[TrainStepGroup]:
        raise NotImplementedError

    def _sample_ground_truth(self, sample: NormalizedSample) -> Any:
        raise NotImplementedError

    def _score_rollout(self, sample: NormalizedSample, output: Any) -> Any:
        raise NotImplementedError

    def _aggregate_scores(self, scores: list[Any]) -> dict[str, Any]:
        raise NotImplementedError

    def _prediction_record(self, sample: NormalizedSample, output: Any, score: Any) -> dict[str, Any]:
        raise NotImplementedError


class DetectTrainer(BaseTrainer):
    def _sample_train_batch(self, batch_size: int) -> list[NormalizedSample]:
        from md_train_framework.detect_augment import build_detect_augment_config, maybe_augment_detect_sample

        if not self.train_samples:
            raise ValueError("training split is empty")
        target = max(1, int(batch_size))
        empty_keep_prob = max(0.0, min(1.0, float(self._override("empty_keep_prob", 1.0))))
        augment_config = build_detect_augment_config(self.config.backend.train_overrides)
        batch: list[DetectSample] = []
        attempts = 0
        max_attempts = max(32, target * 32)
        while len(batch) < target and attempts < max_attempts:
            attempts += 1
            sample = self.rng.choice(self.train_samples)
            if not isinstance(sample, DetectSample):
                continue
            if not sample.boxes and empty_keep_prob < 1.0 and self.rng.random() > empty_keep_prob:
                continue
            batch.append(maybe_augment_detect_sample(sample, rng=self.rng, config=augment_config))
        while len(batch) < target:
            sample = self.rng.choice(self.train_samples)
            if isinstance(sample, DetectSample):
                batch.append(maybe_augment_detect_sample(sample, rng=self.rng, config=augment_config))
        return batch

    def _empty_metrics(self) -> dict[str, Any]:
        return aggregate_detect([])

    def _empty_output(self) -> Any:
        from tuna_sdk import DetectOutput

        return DetectOutput(objects=[])

    def _sample_to_request(self, sample: DetectSample, *, phase: Optional[PhaseSpec], for_eval: bool) -> Any:
        from tuna_sdk import DetectRequest, DetectSettings

        max_tokens = self.config.eval.max_tokens if for_eval else (phase.max_tokens if phase and phase.max_tokens is not None else 256)
        return DetectRequest(
            object_name=self._request_object_name(sample, for_eval=for_eval),
            image_url=sample.request(
                temperature=self.config.eval.temperature if for_eval else float(phase.temperature or 1.0),
                top_p=self.config.eval.top_p if for_eval else float(phase.top_p or 1.0),
                max_tokens=max_tokens or 256,
                max_objects=int(self._override("max_objects", 32)),
            ).image_url,
            settings=DetectSettings(
                temperature=self.config.eval.temperature if for_eval else float(phase.temperature or 1.0),
                top_p=self.config.eval.top_p if for_eval else float(phase.top_p or 1.0),
                max_tokens=max_tokens or 256,
                max_objects=int(self._override("max_objects", 32)),
            ),
        )

    def _sample_to_sft_group(self, sample: DetectSample, *, phase: PhaseSpec) -> Optional[TrainStepGroup]:
        from tuna_sdk import DetectRequest, DetectSettings, DetectSFTTarget

        request = DetectRequest(
            object_name=self._request_object_name(sample, for_eval=False),
            image_url=sample.request(
                temperature=float(phase.temperature or 0.0),
                top_p=float(phase.top_p or 1.0),
                max_tokens=int(phase.max_tokens or 256),
                max_objects=int(self._override("max_objects", 32)),
            ).image_url,
            settings=DetectSettings(
                temperature=float(phase.temperature or 0.0),
                top_p=float(phase.top_p or 1.0),
                max_tokens=int(phase.max_tokens or 256),
                max_objects=int(self._override("max_objects", 32)),
            ),
        )
        return TrainStepGroup.from_sft(request=request, targets=[DetectSFTTarget(boxes=list(sample.boxes))])

    def _sample_ground_truth(self, sample: DetectSample) -> Any:
        return sample.ground_truth()

    def _score_rollout(self, sample: DetectSample, output: Any) -> DetectScore:
        return score_detect(
            ground_truth=sample.boxes,
            output=output,
            reward_preset_id=self.reward_preset.id,
            iou_threshold=float(self._override("iou_threshold", 0.5)),
            fn_penalty_weight=float(self._override("detect_fn_penalty_weight", 0.0)),
            fn_penalty_exponent=float(self._override("detect_fn_penalty_exponent", 1.0)),
            fp_penalty_weight=float(self._override("detect_fp_penalty_weight", 0.0)),
            fp_penalty_exponent=float(self._override("detect_fp_penalty_exponent", 1.0)),
            empty_refusal_penalty=float(self._override("detect_empty_refusal_penalty", 0.0)),
            allow_negative_reward=bool(self._override("detect_allow_negative_reward", False)),
        )

    def _aggregate_scores(self, scores: list[DetectScore]) -> dict[str, Any]:
        return aggregate_detect(scores)

    def _prediction_record(self, sample: DetectSample, output: Any, score: DetectScore) -> dict[str, Any]:
        return {
            "sample_id": sample.sample_id,
            "prompt": sample.object_name,
            "prediction": [item.to_payload() for item in output.objects],
            "ground_truth": [item.to_payload() for item in sample.boxes],
            "metrics": score.__dict__,
        }

    def _request_object_name(self, sample: DetectSample, *, for_eval: bool) -> str:
        if for_eval:
            eval_override = str(self._override("eval_object_name", "") or "").strip()
            if eval_override:
                return eval_override
            return str(sample.object_name)
        variants = self._training_prompt_variants(sample)
        if variants:
            return str(self.rng.choice(variants))
        return str(sample.object_name)

    def _training_prompt_variants(self, sample: DetectSample) -> list[str]:
        del sample
        raw = self._override("prompt_variants", [])
        if isinstance(raw, list):
            return [str(item).strip() for item in raw if str(item).strip()]
        return []


class PointTrainer(BaseTrainer):
    def _empty_metrics(self) -> dict[str, Any]:
        return aggregate_point([])

    def _empty_output(self) -> Any:
        from tuna_sdk import PointOutput

        return PointOutput(points=[])

    def _sample_to_request(self, sample: PointSample, *, phase: Optional[PhaseSpec], for_eval: bool) -> Any:
        max_tokens = self.config.eval.max_tokens if for_eval else (phase.max_tokens if phase and phase.max_tokens is not None else 128)
        return sample.request(
            temperature=self.config.eval.temperature if for_eval else float(phase.temperature or 1.0),
            top_p=self.config.eval.top_p if for_eval else float(phase.top_p or 1.0),
            max_tokens=max_tokens or 128,
        )

    def _sample_to_sft_group(self, sample: PointSample, *, phase: PhaseSpec) -> Optional[TrainStepGroup]:
        target_mode = str(self._override("point_sft_target_mode", "boxes_if_available")).strip().lower()
        boxes = list(sample.boxes) or None
        points = list(sample.points) or None
        if target_mode == "boxes":
            points = None
        elif target_mode == "points":
            boxes = None
        elif target_mode == "boxes_if_available":
            if boxes:
                points = None
        elif target_mode not in {"points_and_boxes", ""}:
            raise ValueError(f"unsupported point_sft_target_mode: {target_mode}")
        request = sample.request(
            temperature=float(phase.temperature or 0.0),
            top_p=float(phase.top_p or 1.0),
            max_tokens=int(phase.max_tokens or 128),
        )
        target = PointSFTTarget(points=points, boxes=boxes)
        return TrainStepGroup.from_sft(request=request, targets=[target])

    def _sample_ground_truth(self, sample: PointSample) -> Any:
        return sample.ground_truth()

    def _score_rollout(self, sample: PointSample, output: Any) -> PointScore:
        return score_point(
            ground_truth_points=sample.points,
            ground_truth_boxes=sample.boxes,
            output=output,
            reward_preset_id=self.reward_preset.id,
            distance_threshold=float(self._override("point_distance_threshold", 0.08)),
        )

    def _aggregate_scores(self, scores: list[PointScore]) -> dict[str, Any]:
        return aggregate_point(scores)

    def _prediction_record(self, sample: PointSample, output: Any, score: PointScore) -> dict[str, Any]:
        return {
            "sample_id": sample.sample_id,
            "prompt": sample.object_name,
            "prediction": [item.to_payload() for item in output.points],
            "ground_truth_points": [item.to_payload() for item in sample.points],
            "ground_truth_boxes": [item.to_payload() for item in sample.boxes],
            "metrics": score.__dict__,
        }


class QueryTrainer(BaseTrainer):
    def _empty_metrics(self) -> dict[str, Any]:
        return aggregate_query([])

    def _empty_output(self) -> Any:
        from tuna_sdk import QueryOutput

        return QueryOutput(answer="")

    def _sample_to_request(self, sample: QuerySample, *, phase: Optional[PhaseSpec], for_eval: bool) -> Any:
        max_tokens = self.config.eval.max_tokens if for_eval else (phase.max_tokens if phase and phase.max_tokens is not None else 128)
        reasoning_flag = False if for_eval else bool(phase.reasoning) if phase is not None else False
        return sample.request(
            temperature=self.config.eval.temperature if for_eval else float(phase.temperature or 1.0),
            top_p=self.config.eval.top_p if for_eval else float(phase.top_p or 1.0),
            max_tokens=max_tokens or 128,
            reasoning=reasoning_flag,
        )

    def _sample_to_sft_group(self, sample: QuerySample, *, phase: PhaseSpec) -> Optional[TrainStepGroup]:
        return sample.sft_group(
            temperature=float(phase.temperature or 0.0),
            top_p=float(phase.top_p or 1.0),
            max_tokens=int(phase.max_tokens or 128),
            reasoning=bool(phase.reasoning),
        )

    def _sample_ground_truth(self, sample: QuerySample) -> Any:
        return None

    def _score_rollout(self, sample: QuerySample, output: Any) -> QueryScore:
        return score_query(
            target_text=sample.answer,
            output=output,
            reward_preset_id=self.reward_preset.id,
        )

    def _aggregate_scores(self, scores: list[QueryScore]) -> dict[str, Any]:
        return aggregate_query(scores)

    def _prediction_record(self, sample: QuerySample, output: Any, score: QueryScore) -> dict[str, Any]:
        return {
            "sample_id": sample.sample_id,
            "question": sample.question,
            "prediction": output.answer,
            "target": sample.answer,
            "metrics": score.__dict__,
        }


def make_trainer(
    *,
    config: FrameworkConfig,
    reward_preset: RewardPreset,
    metric_policy: MetricPolicy,
    paths: RunPaths,
    finetune: Any,
    logger: WandbLogger,
) -> BaseTrainer:
    profile_id = _resolve_backend_profile(config)
    if profile_id:
        from md_train_framework.profiles import build_profile_trainer

        trainer = build_profile_trainer(
            profile_id=profile_id,
            config=config,
            reward_preset=reward_preset,
            metric_policy=metric_policy,
            paths=paths,
            finetune=finetune,
            logger=logger,
        )
        if trainer is not None:
            return trainer
    rng = random.Random(int(config.backend.train_overrides.get("seed", config.extra.get("seed", 42))))
    if config.skill.id == "query":
        return QueryTrainer(
            config=config,
            reward_preset=reward_preset,
            metric_policy=metric_policy,
            paths=paths,
            finetune=finetune,
            rng=rng,
            logger=logger,
        )
    if config.skill.id == "point":
        return PointTrainer(
            config=config,
            reward_preset=reward_preset,
            metric_policy=metric_policy,
            paths=paths,
            finetune=finetune,
            rng=rng,
            logger=logger,
        )
    return DetectTrainer(
        config=config,
        reward_preset=reward_preset,
        metric_policy=metric_policy,
        paths=paths,
        finetune=finetune,
        rng=rng,
        logger=logger,
    )


def create_client_and_finetune(config: FrameworkConfig) -> tuple[TunaClient, Any]:
    api_key = _resolve_api_key(config)
    if not api_key:
        raise ValueError("No Moondream API key resolved for live training. Set api_key or api_key_env_var in the framework config.")
    base_url = _resolve_base_url(config)
    retry_timeout = float(config.backend.train_overrides.get("timeout", config.extra.get("timeout", 180.0)))
    client_retry = _resolve_client_retry(config)
    client = TunaClient(api_key=api_key, base_url=base_url, timeout=retry_timeout, retry=client_retry)
    finetune_id = str(config.backend.train_overrides.get("finetune_id", config.extra.get("finetune_id", ""))).strip()
    if config.recovery.api_health_preflight and api_key:
        client.list_finetunes(limit=1)
    if finetune_id:
        return client, client.get_finetune(finetune_id)
    finetune_name = str(config.backend.train_overrides.get("finetune_name", "")).strip() or f"md-framework-{slugify(config.task.name)}-{int(time.time())}"
    rank = int(config.backend.train_overrides.get("rank", 16))
    return client, client.create_finetune(name=finetune_name, rank=rank)


def evaluate_current_finetune(
    *,
    config: FrameworkConfig,
    reward_preset: RewardPreset,
    metric_policy: MetricPolicy,
    paths: RunPaths,
    finetune: Any,
    split_name: str,
    checkpoint_step: Optional[int] = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    logger = WandbLogger(
        enabled=False,
        project=config.logging.wandb_project,
        run_name="",
        config_payload={},
    )
    trainer = make_trainer(
        config=config,
        reward_preset=reward_preset,
        metric_policy=metric_policy,
        paths=paths,
        finetune=finetune,
        logger=logger,
    )
    event = trainer.evaluate_current(
        split_name=split_name,
        log_step=checkpoint_step or 0,
        stage="replay",
        checkpoint_step=checkpoint_step,
    )
    predictions: list[dict[str, Any]] = []
    if event.predictions_path.exists():
        with event.predictions_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if text:
                    predictions.append(json.loads(text))
    return event.metrics, predictions


def _resolve_api_key(config: FrameworkConfig) -> str:
    _load_env_file(config)
    direct = str(config.backend.train_overrides.get("api_key", config.extra.get("api_key", ""))).strip()
    if direct:
        return direct
    env_var = str(config.backend.train_overrides.get("api_key_env_var", config.extra.get("api_key_env_var", "MOONDREAM_API_KEY"))).strip()
    value = os_environ(env_var)
    if value:
        return value
    env_vars = config.backend.train_overrides.get("api_key_env_vars")
    if isinstance(env_vars, list):
        for env_var in env_vars:
            value = os_environ(str(env_var))
            if value:
                return value
    return ""


def _resolve_base_url(config: FrameworkConfig) -> str:
    _load_env_file(config)
    return str(config.backend.train_overrides.get("base_url", config.extra.get("base_url", "https://api.moondream.ai/v1"))).strip() or "https://api.moondream.ai/v1"


def _resolve_client_retry(config: FrameworkConfig) -> Optional[RetryConfig]:
    raw_max_retries = config.backend.train_overrides.get("client_max_retries", config.extra.get("client_max_retries"))
    if raw_max_retries is None:
        return None
    return RetryConfig(
        max_retries=max(0, int(raw_max_retries)),
        backoff_base=max(0.0, float(config.backend.train_overrides.get("client_backoff_base", config.extra.get("client_backoff_base", 1.0)))),
        backoff_max=max(0.0, float(config.backend.train_overrides.get("client_backoff_max", config.extra.get("client_backoff_max", 20.0)))),
        jitter=max(0.0, float(config.backend.train_overrides.get("client_retry_jitter", config.extra.get("client_retry_jitter", 0.1)))),
    )


def _resolve_backend_profile(config: FrameworkConfig) -> str:
    explicit = str(config.backend.extra.get("profile", "")).strip()
    if explicit:
        return explicit
    from md_train_framework.compat import resolve_backend

    return str(resolve_backend(config).profile or "").strip()


def os_environ(key: str) -> str:
    import os

    return str(os.environ.get(key, "")).strip()


def _load_env_file(config: FrameworkConfig) -> None:
    if load_dotenv is None:
        return
    raw_path = str(config.backend.train_overrides.get("env_file", config.extra.get("env_file", ""))).strip()
    if not raw_path:
        return
    path = config.resolved_path(raw_path)
    if path.exists():
        load_dotenv(path, override=False)


def _is_transient_error(exc: Exception) -> bool:
    if isinstance(exc, TunaNetworkError):
        return True
    if isinstance(exc, TunaAPIError):
        return int(exc.status_code or 0) in TRANSIENT_STATUS_CODES
    return False


def _exception_details(exc: Exception) -> dict[str, Any]:
    if isinstance(exc, TunaAPIError):
        return {
            "request_id": str(exc.request_id or ""),
            "status_code": int(exc.status_code or 0),
            "response_body": exc.response_body,
        }
    if isinstance(exc, TunaNetworkError):
        return {
            "network_cause": repr(exc.cause),
        }
    return {}


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return variance ** 0.5
