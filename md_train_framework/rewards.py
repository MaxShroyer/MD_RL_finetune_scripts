from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class RewardPreset:
    id: str
    skill: str
    family: str
    selection_metric: str
    metrics: tuple[str, ...]
    description: str
    legacy_overrides: dict[str, Any] = field(default_factory=dict)


_CANONICAL_PRESETS = {
    "detect_f1": RewardPreset(
        id="detect_f1",
        skill="detect",
        family="detect_f1",
        selection_metric="eval_f1",
        metrics=("reward", "eval_f1", "eval_f1_macro", "eval_miou", "eval_tp", "eval_fp", "eval_fn"),
        description="Generic detect preset that ranks checkpoints by F1.",
        legacy_overrides={"reward_metric": "f1", "selection_metric": "f1"},
    ),
    "detect_miou": RewardPreset(
        id="detect_miou",
        skill="detect",
        family="detect_miou",
        selection_metric="eval_miou",
        metrics=("reward", "eval_f1", "eval_f1_macro", "eval_miou", "eval_tp", "eval_fp", "eval_fn"),
        description="Generic detect preset that ranks checkpoints by mIoU.",
        legacy_overrides={"reward_metric": "miou", "selection_metric": "miou"},
    ),
    "point_f1": RewardPreset(
        id="point_f1",
        skill="point",
        family="point_f1",
        selection_metric="eval_f1",
        metrics=("reward", "eval_f1", "eval_precision", "eval_recall", "eval_tp", "eval_fp", "eval_fn"),
        description="Generic point-localization preset that rewards F1.",
        legacy_overrides={"reward_metric": "f1"},
    ),
    "point_recall_first": RewardPreset(
        id="point_recall_first",
        skill="point",
        family="point_recall_first",
        selection_metric="eval_f1",
        metrics=("reward", "eval_f1", "eval_precision", "eval_recall", "eval_tp", "eval_fp", "eval_fn"),
        description="Generic point-localization preset that emphasizes recall before F1.",
        legacy_overrides={"reward_metric": "f1", "use_recall_first_preset": True},
    ),
    "query_exact": RewardPreset(
        id="query_exact",
        skill="query",
        family="query_exact",
        selection_metric="eval_reward_mean",
        metrics=("reward_mean", "accuracy", "balanced_accuracy", "macro_f1", "micro_f1", "token_f1", "json_parse_rate"),
        description="Generic query preset that rewards exact or near-exact answers.",
        legacy_overrides={"best_metric": "eval_reward_mean"},
    ),
    "query_token_f1": RewardPreset(
        id="query_token_f1",
        skill="query",
        family="query_token_f1",
        selection_metric="eval_reward_mean",
        metrics=("reward_mean", "token_f1", "macro_f1", "micro_f1", "json_parse_rate"),
        description="Generic query preset that rewards token overlap.",
        legacy_overrides={"best_metric": "eval_reward_mean"},
    ),
    "query_soft_hybrid": RewardPreset(
        id="query_soft_hybrid",
        skill="query",
        family="query_soft_hybrid",
        selection_metric="eval_reward_mean",
        metrics=("reward_mean", "accuracy", "macro_f1", "micro_f1", "token_f1", "json_parse_rate", "json_f1"),
        description="Generic query preset that blends exact match, token overlap, and JSON structure quality.",
        legacy_overrides={"best_metric": "eval_reward_mean"},
    ),
    "query_judge_hybrid": RewardPreset(
        id="query_judge_hybrid",
        skill="query",
        family="query_judge_hybrid",
        selection_metric="eval_reward_mean",
        metrics=(
            "reward_mean",
            "local_reward_mean",
            "judge_score_mean",
            "parse_rate",
            "task_correct_rate",
            "issue_precision",
            "issue_recall",
            "issue_f1",
            "reasoning_f1",
            "extra_issue_rate",
            "empty_list_accuracy",
            "predicted_issue_count_mean",
        ),
        description="Generic query preset for hybrid local-scoring plus judge-assisted evaluation.",
        legacy_overrides={"best_metric": "eval_reward_mean", "grader_profile": "balanced"},
    ),
    "query_ranked_reward": RewardPreset(
        id="query_ranked_reward",
        skill="query",
        family="query_ranked_reward",
        selection_metric="eval_reward_mean",
        metrics=("reward_mean", "eval_reward_mean", "accuracy", "balanced_accuracy", "macro_f1", "micro_f1", "json_parse_rate"),
        description="Generic query preset for task-specific ranked rewards.",
        legacy_overrides={"best_metric": "eval_reward_mean"},
    ),
}

_ALIASES = {
    "football_detect_f1": "detect_f1",
    "neon_tree_detect_hybrid": "detect_f1",
    "football_detect_miou": "detect_miou",
    "ballholder_detect_miou": "detect_miou",
    "statefarm_detect_miou": "detect_miou",
    "bone_fracture_point_f1": "point_f1",
    "bone_fracture_point_recall_first": "point_recall_first",
    "pandid_point_recall_first": "point_recall_first",
    "inspector_local_judge_hybrid": "query_judge_hybrid",
    "vqa_rad_query_reward": "query_exact",
    "chess_piece_position_balanced": "query_exact",
    "ttt_board_macro_f1": "query_exact",
    "construction_site_caption_token_f1": "query_token_f1",
    "construction_site_rule_vqa_soft": "query_soft_hybrid",
    "ttt_ranked_reward": "query_ranked_reward",
}


def resolve_reward_preset_id(preset_id: str) -> str:
    key = str(preset_id or "").strip()
    return _ALIASES.get(key, key)


def get_reward_preset(preset_id: str) -> RewardPreset:
    resolved_id = resolve_reward_preset_id(preset_id)
    preset = _CANONICAL_PRESETS.get(resolved_id)
    if preset is None:
        raise KeyError(f"unknown reward preset: {preset_id}")
    return preset


def list_reward_presets(*, skill: Optional[str] = None) -> list[RewardPreset]:
    presets = list(_CANONICAL_PRESETS.values())
    if skill is None:
        return sorted(presets, key=lambda item: item.id)
    return sorted((preset for preset in presets if preset.skill == skill), key=lambda item: item.id)


def list_reward_aliases() -> dict[str, str]:
    return dict(sorted(_ALIASES.items()))
