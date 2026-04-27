from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from md_train_framework.utils import deep_merge, read_json, stable_json_hash


def _coerce_section(value: Any, *, field_name: str, default_name: str = "") -> dict[str, Any]:
    if isinstance(value, str):
        return {"name": value} if field_name == "task" else {"id": value}
    if value is None:
        return {"name": default_name} if field_name == "task" else {"id": default_name}
    if isinstance(value, dict):
        return dict(value)
    raise TypeError(f"{field_name} must be a string or object")


def _coerce_list_of_dicts(value: Any, *, field_name: str) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list")
    output: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            raise TypeError(f"{field_name} items must be objects")
        output.append(dict(item))
    return output


@dataclass(frozen=True)
class SkillSpec:
    id: str
    label: str = ""
    family: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Any) -> "SkillSpec":
        data = _coerce_section(raw, field_name="skill", default_name="detect")
        skill_id = str(data.pop("id", data.pop("name", "detect"))).strip() or "detect"
        label = str(data.pop("label", skill_id)).strip()
        family = str(data.pop("family", skill_id)).strip() or skill_id
        return cls(id=skill_id, label=label, family=family, extra=data)

    def to_dict(self) -> dict[str, Any]:
        payload = {"id": self.id}
        if self.label and self.label != self.id:
            payload["label"] = self.label
        if self.family and self.family != self.id:
            payload["family"] = self.family
        payload.update(self.extra)
        return payload


@dataclass(frozen=True)
class TaskSpec:
    name: str
    label: str = ""
    family: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Any) -> "TaskSpec":
        data = _coerce_section(raw, field_name="task", default_name="quickstart")
        name = str(data.pop("name", data.pop("id", "quickstart"))).strip() or "quickstart"
        label = str(data.pop("label", name)).strip()
        family = str(data.pop("family", name)).strip() or name
        return cls(name=name, label=label, family=family, extra=data)

    def to_dict(self) -> dict[str, Any]:
        payload = {"name": self.name}
        if self.label and self.label != self.name:
            payload["label"] = self.label
        if self.family and self.family != self.name:
            payload["family"] = self.family
        payload.update(self.extra)
        return payload


@dataclass(frozen=True)
class DatasetSpec:
    source: str = "local_jsonl"
    path: str = ""
    name: str = ""
    train_split: str = "train"
    val_split: str = "validation"
    test_split: str = "test"
    image_root: str = ""
    split_files: dict[str, str] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Any) -> "DatasetSpec":
        data = dict(raw or {})
        source = str(data.pop("source", "local_jsonl")).strip() or "local_jsonl"
        split_files = dict(data.pop("split_files", {}) or {})
        return cls(
            source=source,
            path=str(data.pop("path", data.pop("dataset_path", ""))).strip(),
            name=str(data.pop("name", data.pop("dataset_name", ""))).strip(),
            train_split=str(data.pop("train_split", "train")).strip() or "train",
            val_split=str(data.pop("val_split", "validation")).strip() or "validation",
            test_split=str(data.pop("test_split", "test")).strip() or "test",
            image_root=str(data.pop("image_root", "")).strip(),
            split_files=split_files,
            extra=data,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "source": self.source,
            "path": self.path,
            "name": self.name,
            "train_split": self.train_split,
            "val_split": self.val_split,
            "test_split": self.test_split,
        }
        if self.image_root:
            payload["image_root"] = self.image_root
        if self.split_files:
            payload["split_files"] = dict(self.split_files)
        payload.update(self.extra)
        return payload


@dataclass(frozen=True)
class PhaseSpec:
    name: str
    mode: str
    steps: int
    lr: float
    batch_size: int
    group_size: Optional[int] = None
    num_rollouts: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    reasoning: Optional[bool] = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Any, *, default_name: str) -> "PhaseSpec":
        data = dict(raw or {})
        name = str(data.pop("name", default_name)).strip() or default_name
        mode = str(data.pop("mode", name)).strip() or name
        return cls(
            name=name,
            mode=mode,
            steps=int(data.pop("steps", 0)),
            lr=float(data.pop("lr", 2e-4)),
            batch_size=int(data.pop("batch_size", 8)),
            group_size=_optional_int(data.pop("group_size", None)),
            num_rollouts=_optional_int(data.pop("num_rollouts", None)),
            temperature=_optional_float(data.pop("temperature", None)),
            top_p=_optional_float(data.pop("top_p", None)),
            max_tokens=_optional_int(data.pop("max_tokens", None)),
            reasoning=_optional_bool(data.pop("reasoning", None)),
            extra=data,
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "mode": self.mode,
            "steps": self.steps,
            "lr": self.lr,
            "batch_size": self.batch_size,
        }
        if self.group_size is not None:
            payload["group_size"] = self.group_size
        if self.num_rollouts is not None:
            payload["num_rollouts"] = self.num_rollouts
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if self.reasoning is not None:
            payload["reasoning"] = self.reasoning
        payload.update(self.extra)
        return payload


@dataclass(frozen=True)
class RewardSpec:
    preset: str
    selection_metric: str = ""
    weights: dict[str, float] = field(default_factory=dict)
    penalties: dict[str, float] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Any) -> "RewardSpec":
        data = dict(raw or {})
        return cls(
            preset=str(data.pop("preset", data.pop("id", ""))).strip(),
            selection_metric=str(data.pop("selection_metric", "")).strip(),
            weights={str(k): float(v) for k, v in dict(data.pop("weights", {}) or {}).items()},
            penalties={str(k): float(v) for k, v in dict(data.pop("penalties", {}) or {}).items()},
            extra=data,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {"preset": self.preset}
        if self.selection_metric:
            payload["selection_metric"] = self.selection_metric
        if self.weights:
            payload["weights"] = dict(self.weights)
        if self.penalties:
            payload["penalties"] = dict(self.penalties)
        payload.update(self.extra)
        return payload


@dataclass(frozen=True)
class EvalSpec:
    eval_every: int = 10
    save_every: int = 10
    max_samples: Optional[int] = 64
    fixed_subset_size: int = 0
    fixed_subset_seed: int = 1337
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: Optional[int] = None
    async_checkpoint_eval: bool = True
    async_checkpoint_eval_dir: str = "outputs/async_checkpoint_eval"
    async_checkpoint_eval_max_inflight: int = 1
    async_checkpoint_eval_drain_on_exit: bool = True
    final_splits: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Any) -> "EvalSpec":
        data = dict(raw or {})
        final_splits = [str(item) for item in list(data.pop("final_splits", []) or [])]
        return cls(
            eval_every=int(data.pop("eval_every", 10)),
            save_every=int(data.pop("save_every", 10)),
            max_samples=_optional_int(data.pop("max_samples", data.pop("eval_max_samples", 64))),
            fixed_subset_size=max(0, int(data.pop("fixed_subset_size", data.pop("eval_fixed_subset_size", 0)))),
            fixed_subset_seed=int(data.pop("fixed_subset_seed", data.pop("eval_fixed_subset_seed", 1337))),
            temperature=float(data.pop("temperature", data.pop("eval_temperature", 0.0))),
            top_p=float(data.pop("top_p", data.pop("eval_top_p", 1.0))),
            max_tokens=_optional_int(data.pop("max_tokens", data.pop("eval_max_tokens", None))),
            async_checkpoint_eval=bool(data.pop("async_checkpoint_eval", True)),
            async_checkpoint_eval_dir=str(data.pop("async_checkpoint_eval_dir", "outputs/async_checkpoint_eval")).strip(),
            async_checkpoint_eval_max_inflight=int(data.pop("async_checkpoint_eval_max_inflight", 1)),
            async_checkpoint_eval_drain_on_exit=bool(data.pop("async_checkpoint_eval_drain_on_exit", True)),
            final_splits=final_splits,
            extra=data,
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "eval_every": self.eval_every,
            "save_every": self.save_every,
            "max_samples": self.max_samples,
            "fixed_subset_size": self.fixed_subset_size,
            "fixed_subset_seed": self.fixed_subset_seed,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "async_checkpoint_eval": self.async_checkpoint_eval,
            "async_checkpoint_eval_dir": self.async_checkpoint_eval_dir,
            "async_checkpoint_eval_max_inflight": self.async_checkpoint_eval_max_inflight,
            "async_checkpoint_eval_drain_on_exit": self.async_checkpoint_eval_drain_on_exit,
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if self.final_splits:
            payload["final_splits"] = list(self.final_splits)
        payload.update(self.extra)
        return payload


@dataclass(frozen=True)
class LoggingSpec:
    run_root: str = "md_train_framework/outputs/runs"
    registry_path: str = "md_train_framework/outputs/runs.db"
    wandb_project: str = "moondream-md-train-framework"
    wandb_run_name: str = ""
    enable_wandb: bool = True
    compact_wandb: bool = True
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Any) -> "LoggingSpec":
        data = dict(raw or {})
        return cls(
            run_root=str(data.pop("run_root", "md_train_framework/outputs/runs")).strip(),
            registry_path=str(data.pop("registry_path", "md_train_framework/outputs/runs.db")).strip(),
            wandb_project=str(data.pop("wandb_project", "moondream-md-train-framework")).strip(),
            wandb_run_name=str(data.pop("wandb_run_name", "")).strip(),
            enable_wandb=bool(data.pop("enable_wandb", True)),
            compact_wandb=bool(data.pop("compact_wandb", True)),
            extra=data,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "run_root": self.run_root,
            "registry_path": self.registry_path,
            "wandb_project": self.wandb_project,
            "wandb_run_name": self.wandb_run_name,
            "enable_wandb": self.enable_wandb,
            "compact_wandb": self.compact_wandb,
        }
        payload.update(self.extra)
        return payload


@dataclass(frozen=True)
class SweepSpec:
    enabled: bool = False
    max_parallel: int = 2
    stagger_seconds: float = 3.0
    stage1_scale: float = 0.25
    continue_top_k: int = 2
    query_axes: dict[str, list[Any]] = field(default_factory=dict)
    detect_point_axes: dict[str, list[Any]] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Any) -> "SweepSpec":
        data = dict(raw or {})
        return cls(
            enabled=bool(data.pop("enabled", False)),
            max_parallel=int(data.pop("max_parallel", 2)),
            stagger_seconds=float(data.pop("stagger_seconds", 3.0)),
            stage1_scale=float(data.pop("stage1_scale", 0.25)),
            continue_top_k=int(data.pop("continue_top_k", 2)),
            query_axes={str(k): list(v) for k, v in dict(data.pop("query_axes", {}) or {}).items()},
            detect_point_axes={str(k): list(v) for k, v in dict(data.pop("detect_point_axes", {}) or {}).items()},
            extra=data,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "enabled": self.enabled,
            "max_parallel": self.max_parallel,
            "stagger_seconds": self.stagger_seconds,
            "stage1_scale": self.stage1_scale,
            "continue_top_k": self.continue_top_k,
            "query_axes": dict(self.query_axes),
            "detect_point_axes": dict(self.detect_point_axes),
        }
        payload.update(self.extra)
        return payload


@dataclass(frozen=True)
class RecoverySpec:
    train_step_max_retries: int = 4
    cooldown_s: float = 15.0
    quarantine_path: str = "md_train_framework/outputs/recovery/quarantine.jsonl"
    resume_queue_path: str = "md_train_framework/outputs/recovery/resume_queue.jsonl"
    api_health_preflight: bool = True
    transient_failure_codes: list[int] = field(default_factory=lambda: [429, 500, 502, 503, 504])
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Any) -> "RecoverySpec":
        data = dict(raw or {})
        codes = [int(code) for code in list(data.pop("transient_failure_codes", [429, 500, 502, 503, 504]))]
        return cls(
            train_step_max_retries=int(data.pop("train_step_max_retries", 4)),
            cooldown_s=float(data.pop("cooldown_s", 15.0)),
            quarantine_path=str(data.pop("quarantine_path", "md_train_framework/outputs/recovery/quarantine.jsonl")).strip(),
            resume_queue_path=str(data.pop("resume_queue_path", "md_train_framework/outputs/recovery/resume_queue.jsonl")).strip(),
            api_health_preflight=bool(data.pop("api_health_preflight", True)),
            transient_failure_codes=codes,
            extra=data,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "train_step_max_retries": self.train_step_max_retries,
            "cooldown_s": self.cooldown_s,
            "quarantine_path": self.quarantine_path,
            "resume_queue_path": self.resume_queue_path,
            "api_health_preflight": self.api_health_preflight,
            "transient_failure_codes": list(self.transient_failure_codes),
        }
        payload.update(self.extra)
        return payload


@dataclass(frozen=True)
class BackendSpec:
    id: str
    train_overrides: dict[str, Any] = field(default_factory=dict)
    benchmark_overrides: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Any) -> "BackendSpec":
        data = _coerce_section(raw, field_name="backend", default_name="")
        backend_id = str(data.pop("id", data.pop("name", ""))).strip()
        return cls(
            id=backend_id,
            train_overrides=dict(data.pop("train_overrides", {}) or {}),
            benchmark_overrides=dict(data.pop("benchmark_overrides", {}) or {}),
            extra=data,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "id": self.id,
            "train_overrides": dict(self.train_overrides),
            "benchmark_overrides": dict(self.benchmark_overrides),
        }
        payload.update(self.extra)
        return payload


@dataclass(frozen=True)
class FrameworkConfig:
    skill: SkillSpec
    task: TaskSpec
    dataset: DatasetSpec
    phases: tuple[PhaseSpec, ...]
    reward: RewardSpec
    eval: EvalSpec
    logging: LoggingSpec
    sweep: SweepSpec
    recovery: RecoverySpec
    backend: BackendSpec
    config_path: Optional[Path] = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def mode(self) -> str:
        phase_modes = [phase.mode for phase in self.phases]
        if phase_modes == ["sft"]:
            return "sft"
        if phase_modes == ["rl"]:
            return "rl"
        if phase_modes == ["sft", "rl"]:
            return "sft_then_rl"
        return "custom"

    @property
    def config_hash(self) -> str:
        return stable_json_hash(self.to_dict(include_meta=False))

    def dataset_identity(self) -> dict[str, Any]:
        return self.dataset.to_dict()

    def to_dict(self, *, include_meta: bool = True) -> dict[str, Any]:
        payload = {
            "skill": self.skill.to_dict(),
            "task": self.task.to_dict(),
            "dataset": self.dataset.to_dict(),
            "phases": [phase.to_dict() for phase in self.phases],
            "reward": self.reward.to_dict(),
            "eval": self.eval.to_dict(),
            "logging": self.logging.to_dict(),
            "sweep": self.sweep.to_dict(),
            "recovery": self.recovery.to_dict(),
            "backend": self.backend.to_dict(),
        }
        payload.update(self.extra)
        if include_meta and self.config_path is not None:
            payload["config_path"] = str(self.config_path)
        return payload

    def resolved_path(self, value: str) -> Path:
        raw = Path(str(value or "")).expanduser()
        if raw.is_absolute() or self.config_path is None:
            return raw
        return (self.config_path.parent / raw).resolve()

    @classmethod
    def from_dict(cls, raw: dict[str, Any], *, config_path: Optional[Path] = None) -> "FrameworkConfig":
        data = dict(raw or {})
        phases = _coerce_list_of_dicts(data.pop("phases", []), field_name="phases")
        if not phases:
            phase_mode = str(data.get("mode", "rl")).strip() or "rl"
            if phase_mode == "sft_then_rl":
                phases = [
                    _default_phase_for_mode("sft", {}),
                    _default_phase_for_mode("rl", {}),
                ]
            else:
                phases = [_default_phase_for_mode(phase_mode, {})]
        return cls(
            skill=SkillSpec.from_dict(data.pop("skill", "detect")),
            task=TaskSpec.from_dict(data.pop("task", "quickstart")),
            dataset=DatasetSpec.from_dict(data.pop("dataset", {})),
            phases=tuple(PhaseSpec.from_dict(phase, default_name=f"phase_{idx}") for idx, phase in enumerate(phases, start=1)),
            reward=RewardSpec.from_dict(data.pop("reward", {})),
            eval=EvalSpec.from_dict(data.pop("eval", {})),
            logging=LoggingSpec.from_dict(data.pop("logging", {})),
            sweep=SweepSpec.from_dict(data.pop("sweep", {})),
            recovery=RecoverySpec.from_dict(data.pop("recovery", {})),
            backend=BackendSpec.from_dict(data.pop("backend", {})),
            config_path=config_path.resolve() if config_path is not None else None,
            extra=data,
        )


def load_framework_config(path: str | Path) -> FrameworkConfig:
    config_path = Path(path).expanduser().resolve()
    payload = read_json(config_path)
    if not isinstance(payload, dict):
        raise ValueError(f"framework config must be a JSON object: {config_path}")
    return FrameworkConfig.from_dict(payload, config_path=config_path)


def save_framework_config(path: str | Path, config: FrameworkConfig | dict[str, Any]) -> None:
    output_path = Path(path).expanduser().resolve()
    payload = config.to_dict(include_meta=False) if isinstance(config, FrameworkConfig) else dict(config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def apply_overrides(config: FrameworkConfig, overrides: dict[str, Any]) -> FrameworkConfig:
    merged = deep_merge(config.to_dict(include_meta=False), overrides)
    return FrameworkConfig.from_dict(merged, config_path=config.config_path)


def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    return int(value)


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def _optional_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    return bool(value)


def _default_phase_for_mode(mode: str, task_data: dict[str, Any]) -> dict[str, Any]:
    normalized = str(mode or "rl").strip()
    if normalized == "sft":
        return {"name": "sft", "mode": "sft", "steps": 80, "lr": 2e-4, "batch_size": 8}
    if normalized == "sft_then_rl":
        return {"name": "sft", "mode": "sft", "steps": 80, "lr": 2e-4, "batch_size": 8}
    return {"name": "rl", "mode": "rl", "steps": 120, "lr": 2e-4, "batch_size": 8}
