from __future__ import annotations

import random
from typing import Any, Optional

from md_train_framework.config import FrameworkConfig
from md_train_framework.metrics import MetricPolicy
from md_train_framework.rewards import RewardPreset
from md_train_framework.runtime import BaseTrainer
from md_train_framework.wandb_logger import WandbLogger

from .ballholder_detect import BallHolderDetectTrainer
from .pandid_point import PandidPointTrainer
from .statefarm_detect import StatefarmDetectTrainer
from .ttt_query import TTTQueryTrainer


_PROFILE_TRAINERS = {
    "ballholder_detect": BallHolderDetectTrainer,
    "statefarm_detect": StatefarmDetectTrainer,
    "pandid_point": PandidPointTrainer,
    "ttt": TTTQueryTrainer,
    "ttt_query": TTTQueryTrainer,
}


def build_profile_trainer(
    *,
    profile_id: str,
    config: FrameworkConfig,
    reward_preset: RewardPreset,
    metric_policy: MetricPolicy,
    paths: Any,
    finetune: Any,
    logger: WandbLogger,
) -> Optional[BaseTrainer]:
    trainer_cls = _PROFILE_TRAINERS.get(str(profile_id or "").strip())
    if trainer_cls is None:
        return None
    rng = random.Random(int(config.backend.train_overrides.get("seed", config.extra.get("seed", 42))))
    return trainer_cls(
        config=config,
        reward_preset=reward_preset,
        metric_policy=metric_policy,
        paths=paths,
        finetune=finetune,
        rng=rng,
        logger=logger,
    )


def default_profile_sweep_axes(config: FrameworkConfig, *, profile_id: str) -> dict[str, list[Any]]:
    normalized = str(profile_id or "").strip()
    if normalized == "pandid_point":
        return {
            "phase:rl:lr": [5e-5, 1e-4],
            "phase:rl:batch_size": [32, 16],
            "phase:rl:group_size": [8],
            "backend:off_policy_mix_ratio": [0.25, 0.5],
        }
    if normalized in {"ttt", "ttt_query"}:
        return {
            "phase:rl:lr": [1e-3, 5e-4],
            "phase:rl:group_size": [8, 4],
            "phase:rl:reasoning": [True, False],
            "backend:off_policy_mix_ratio": [0.25, 0.5],
            "backend:off_policy_warmup_steps": [10, 30],
            "backend:off_policy_min_buffer_groups": [64, 128],
        }
    if normalized == "ballholder_detect":
        return {
            "phase:rl:lr": [2e-3, 1e-3, 2.5e-3],
            "phase:rl:batch_size": [8, 16],
            "phase:rl:group_size": [4, 8],
            "backend:rank": [8, 16],
            "backend:off_policy": [False],
        }
    if normalized == "statefarm_detect":
        return {
            "phase:rl:lr": [2.5e-3, 2e-3, 1e-3],
            "phase:rl:batch_size": [8, 16],
            "phase:rl:group_size": [4, 8],
            "backend:rank": [8, 16, 32],
            "backend:off_policy": [False],
        }
    return {}
