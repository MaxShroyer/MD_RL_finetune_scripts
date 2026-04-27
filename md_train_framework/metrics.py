from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from md_train_framework.rewards import RewardPreset


_COMMON_NUMERIC = {
    "reward",
    "reward_mean",
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "micro_f1",
    "miou",
    "precision",
    "recall",
    "tp",
    "fp",
    "fn",
    "tn",
    "json_parse_rate",
    "judge_score_mean",
    "judge_degraded_rate",
    "task_correct_rate",
    "issue_f1",
    "evidence_f1",
    "severity_accuracy",
    "insufficient_accuracy",
    "token_f1",
}


@dataclass(frozen=True)
class MetricPolicy:
    skill: str
    reward_preset: str
    selection_metric: str
    allowed_metrics: tuple[str, ...]

    def filter_metrics(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        allowed = set(self.allowed_metrics)
        for key, value in payload.items():
            normalized = str(key)
            key_suffix = normalized[5:] if normalized.startswith("eval_") else normalized
            if key_suffix in allowed or normalized in allowed:
                output[normalized] = value
        return output


def build_metric_policy(skill: str, preset: RewardPreset, *, requested_selection_metric: str = "") -> MetricPolicy:
    allowed = set(_COMMON_NUMERIC)
    allowed.update(preset.metrics)
    if skill in {"detect", "point"}:
        allowed.update({"eval_f1", "eval_f1_macro", "eval_miou", "eval_tp", "eval_fp", "eval_fn"})
    if skill == "query":
        allowed.update({"reward_mean", "local_reward_mean", "judge_score_mean", "json_parse_rate"})
    selection_metric = requested_selection_metric.strip() or preset.selection_metric
    allowed.add(selection_metric)
    return MetricPolicy(
        skill=skill,
        reward_preset=preset.id,
        selection_metric=selection_metric,
        allowed_metrics=tuple(sorted(allowed)),
    )


def metric_deltas(left: Mapping[str, Any], right: Mapping[str, Any], *, keys: Iterable[str] | None = None) -> dict[str, float]:
    metric_keys = list(keys or sorted(set(left.keys()) & set(right.keys())))
    deltas: dict[str, float] = {}
    for key in metric_keys:
        try:
            deltas[str(key)] = float(left[key]) - float(right[key])
        except (KeyError, TypeError, ValueError):
            continue
    return deltas
