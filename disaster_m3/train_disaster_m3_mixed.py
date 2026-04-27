#!/usr/bin/env python3
"""Train a mixed query/point/detect DisasterM3 Moondream finetune."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import string
import sys
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import fmean
from types import SimpleNamespace
from typing import Any, Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from disaster_m3 import common  # noqa: E402
from async_checkpoint_eval import (  # noqa: E402
    CheckpointEvalResult,
    DispatchHandle,
    dispatch_checkpoint_eval,
    drain_checkpoint_eval_jobs,
    poll_checkpoint_eval_jobs,
)
from tuna_sdk import DetectOutput, PointOutput, QueryOutput, TunaClient  # noqa: E402
from tuna_sdk.errors import TunaAPIError, TunaNetworkError  # noqa: E402

class _WandbRun:
    def __init__(self) -> None:
        self.summary: dict[str, Any] = {}

    def finish(self) -> None:
        return


class _WandbShim:
    @staticmethod
    def init(*_args: Any, **_kwargs: Any) -> _WandbRun:
        print("wandb unavailable or shadowed locally; continuing without remote logging.")
        return _WandbRun()

    @staticmethod
    def log(*_args: Any, **_kwargs: Any) -> None:
        return


try:
    import wandb as _wandb  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    wandb = _WandbShim()
else:
    if hasattr(_wandb, "init") and hasattr(_wandb, "log"):
        wandb = _wandb
    else:  # pragma: no cover
        module_path = getattr(_wandb, "__file__", None) or getattr(_wandb, "__path__", None)
        print(f"wandb import resolved to a non-package module ({module_path}); using local shim instead.")
        wandb = _WandbShim()

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = common.repo_relative("configs", "train_disaster_m3_mixed_default.json")


@dataclass(frozen=True)
class StageConfig:
    name: str
    num_steps: int
    batch_size: int
    group_size: int
    temperature: float
    top_p: float
    max_tokens: int
    lr: float
    reasoning: bool
    use_server_ground_truth: bool
    strict_query_rewards: bool


@dataclass(frozen=True)
class MixedScoreOutcome:
    reward: float
    parse_success: bool
    task_correct: bool
    json_object_parsed: bool
    exact_match: float = 0.0
    answer_set_f1: float = 0.0
    count_score: float = 0.0
    detect_f1: float = 0.0
    detect_miou: float = 0.0
    point_f1: float = 0.0
    point_distance_score: float = 0.0
    description_token_f1: float = 0.0
    description_field_coverage: float = 0.0
    recovery_action_score: float = 0.0
    needs_recovery_accuracy: float = 0.0
    change_awareness_score: float = 0.0
    concise_score: float = 0.0


@dataclass
class MetricBucket:
    reward_values: list[float] = field(default_factory=list)
    parse_success_count: int = 0
    json_object_count: int = 0
    task_correct_count: int = 0
    exact_match_values: list[float] = field(default_factory=list)
    answer_set_f1_values: list[float] = field(default_factory=list)
    count_values: list[float] = field(default_factory=list)
    detect_f1_values: list[float] = field(default_factory=list)
    detect_miou_values: list[float] = field(default_factory=list)
    point_f1_values: list[float] = field(default_factory=list)
    point_distance_values: list[float] = field(default_factory=list)
    description_token_f1_values: list[float] = field(default_factory=list)
    description_field_coverage_values: list[float] = field(default_factory=list)
    recovery_action_values: list[float] = field(default_factory=list)
    needs_recovery_values: list[float] = field(default_factory=list)
    change_awareness_values: list[float] = field(default_factory=list)
    concise_values: list[float] = field(default_factory=list)


def _random_suffix(length: int = 6) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return "".join(random.choices(alphabet, k=length))


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    raw_argv = list(argv) if argv is not None else list(os.sys.argv[1:])
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    pre_args, _ = pre_parser.parse_known_args(raw_argv)
    config_path = common.resolve_config_path(pre_args.config, script_dir=SCRIPT_DIR)
    config = common.load_json_config(config_path, default_path=DEFAULT_CONFIG_PATH)

    parser = argparse.ArgumentParser(description="Train a mixed-skill DisasterM3 finetune.")
    parser.add_argument("--config", default=str(config_path))
    parser.add_argument("--env-file", default=str(common.repo_relative(".env.staging")))
    parser.add_argument("--api-key", default="")
    parser.add_argument("--api-key-env-var", default=common.DEFAULT_API_KEY_ENV_VAR)
    parser.add_argument("--base-url", default=common.DEFAULT_BASE_URL)
    parser.add_argument("--dataset-dir", default=str(common.DEFAULT_OUTPUT_DIR))
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--mode", choices=("bootstrap", "rl", "bootstrap_then_rl", "full_pipeline"), default="full_pipeline")
    parser.add_argument("--finetune-id", default="")
    parser.add_argument("--finetune-name", default="")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--image-quality", type=int, default=common.DEFAULT_JPEG_QUALITY)
    parser.add_argument("--detect-max-objects", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--eval-max-samples", type=int, default=128)
    parser.add_argument("--eval-sample-count", type=int, default=25)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--rollout-retries", type=int, default=2)
    parser.add_argument("--rollout-retry-backoff-s", type=float, default=5.0)
    parser.add_argument("--wandb-project", default="moondream-disaster-m3-mixed")
    parser.add_argument("--wandb-run-name", default="")
    parser.add_argument("--run-output-dir", default=str(common.repo_relative("outputs", "runs")))
    parser.add_argument("--benchmark-output-dir", default=str(common.DEFAULT_BENCHMARK_OUTPUT_DIR))
    off_policy_group = parser.add_mutually_exclusive_group()
    off_policy_group.add_argument("--off-policy", dest="off_policy", action="store_true")
    off_policy_group.add_argument("--no-off-policy", dest="off_policy", action="store_false")
    parser.set_defaults(off_policy=False)
    parser.add_argument("--off-policy-mix-ratio", type=float, default=0.25)
    parser.add_argument("--off-policy-buffer-size", type=int, default=2048)
    parser.add_argument("--off-policy-warmup-steps", type=int, default=10)
    parser.add_argument("--off-policy-min-buffer-groups", type=int, default=64)
    async_eval_group = parser.add_mutually_exclusive_group()
    async_eval_group.add_argument("--async-checkpoint-eval", dest="async_checkpoint_eval", action="store_true")
    async_eval_group.add_argument("--no-async-checkpoint-eval", dest="async_checkpoint_eval", action="store_false")
    parser.set_defaults(async_checkpoint_eval=False)
    parser.add_argument("--async-checkpoint-eval-dir", default=str(common.DEFAULT_ASYNC_CHECKPOINT_EVAL_DIR))
    parser.add_argument("--async-checkpoint-eval-max-inflight", type=int, default=1)
    async_drain_group = parser.add_mutually_exclusive_group()
    async_drain_group.add_argument(
        "--async-checkpoint-eval-drain-on-exit",
        dest="async_checkpoint_eval_drain_on_exit",
        action="store_true",
    )
    async_drain_group.add_argument(
        "--no-async-checkpoint-eval-drain-on-exit",
        dest="async_checkpoint_eval_drain_on_exit",
        action="store_false",
    )
    parser.set_defaults(async_checkpoint_eval_drain_on_exit=True)

    parser.add_argument("--bootstrap-steps", type=int, default=60)
    parser.add_argument("--bootstrap-batch-size", type=int, default=32)
    parser.add_argument("--bootstrap-group-size", type=int, default=8)
    parser.add_argument("--bootstrap-temperature", type=float, default=0.15)
    parser.add_argument("--bootstrap-top-p", type=float, default=1.0)
    parser.add_argument("--bootstrap-max-tokens", type=int, default=320)
    parser.add_argument("--bootstrap-lr", type=float, default=1e-4)
    parser.add_argument("--bootstrap-reasoning", action="store_true")

    parser.add_argument("--rl-steps", type=int, default=120)
    parser.add_argument("--rl-batch-size", type=int, default=32)
    parser.add_argument("--rl-group-size", type=int, default=8)
    parser.add_argument("--rl-temperature", type=float, default=0.8)
    parser.add_argument("--rl-top-p", type=float, default=1.0)
    parser.add_argument("--rl-max-tokens", type=int, default=384)
    parser.add_argument("--rl-lr", type=float, default=8e-5)
    parser.add_argument("--rl-reasoning", action="store_true")
    parser.add_argument("--rl-only-steps", type=int, default=0)

    option_to_dest: dict[str, str] = {}
    for action in parser._actions:
        if not action.option_strings:
            continue
        for opt in action.option_strings:
            option_to_dest[opt] = action.dest
    overridden = {option_to_dest[arg] for arg in raw_argv if arg in option_to_dest}
    config_cli_args = common.config_to_cli_args(parser, config, config_path=config_path, overridden_dests=overridden)
    args = parser.parse_args(config_cli_args + raw_argv)
    args.config = str(common.resolve_config_path(args.config, script_dir=SCRIPT_DIR))
    args.dataset_dir = common.resolve_path(args.dataset_dir, repo_root=REPO_ROOT, module_root=SCRIPT_DIR)
    args.env_file = str(common.resolve_path(args.env_file, repo_root=REPO_ROOT, module_root=SCRIPT_DIR))
    args.run_output_dir = common.resolve_path(args.run_output_dir, repo_root=REPO_ROOT, module_root=SCRIPT_DIR)
    args.benchmark_output_dir = common.resolve_path(args.benchmark_output_dir, repo_root=REPO_ROOT, module_root=SCRIPT_DIR)
    args.async_checkpoint_eval_dir = common.resolve_path(
        args.async_checkpoint_eval_dir,
        repo_root=REPO_ROOT,
        module_root=SCRIPT_DIR,
    )
    if not str(args.finetune_name or "").strip():
        args.finetune_name = f"disaster-m3-mixed-{_random_suffix()}"
    if int(args.rl_only_steps) <= 0:
        args.rl_only_steps = int(args.bootstrap_steps) + int(args.rl_steps)
    return args


def _stage_configs(args: argparse.Namespace) -> tuple[StageConfig, StageConfig, StageConfig]:
    bootstrap = StageConfig(
        name="bootstrap",
        num_steps=int(args.bootstrap_steps),
        batch_size=int(args.bootstrap_batch_size),
        group_size=int(args.bootstrap_group_size),
        temperature=float(args.bootstrap_temperature),
        top_p=float(args.bootstrap_top_p),
        max_tokens=int(args.bootstrap_max_tokens),
        lr=float(args.bootstrap_lr),
        reasoning=bool(args.bootstrap_reasoning),
        use_server_ground_truth=True,
        strict_query_rewards=True,
    )
    rl = StageConfig(
        name="rl",
        num_steps=int(args.rl_steps),
        batch_size=int(args.rl_batch_size),
        group_size=int(args.rl_group_size),
        temperature=float(args.rl_temperature),
        top_p=float(args.rl_top_p),
        max_tokens=int(args.rl_max_tokens),
        lr=float(args.rl_lr),
        reasoning=bool(args.rl_reasoning),
        use_server_ground_truth=False,
        strict_query_rewards=False,
    )
    rl_only = StageConfig(
        name="rl_only",
        num_steps=int(args.rl_only_steps),
        batch_size=int(args.rl_batch_size),
        group_size=int(args.rl_group_size),
        temperature=float(args.rl_temperature),
        top_p=float(args.rl_top_p),
        max_tokens=int(args.rl_max_tokens),
        lr=float(args.rl_lr),
        reasoning=bool(args.rl_reasoning),
        use_server_ground_truth=False,
        strict_query_rewards=False,
    )
    return bootstrap, rl, rl_only


def _validate_args(args: argparse.Namespace) -> None:
    if int(args.eval_every) < 0:
        raise ValueError("--eval-every must be >= 0")
    if int(args.save_every) < 0:
        raise ValueError("--save-every must be >= 0")
    if int(args.max_workers) <= 0:
        raise ValueError("--max-workers must be > 0")
    if int(args.bootstrap_batch_size) <= 0 or int(args.rl_batch_size) <= 0:
        raise ValueError("batch sizes must be > 0")
    if int(args.bootstrap_group_size) <= 0 or int(args.rl_group_size) <= 0:
        raise ValueError("group sizes must be > 0")
    if not (0.0 <= float(args.off_policy_mix_ratio) <= 1.0):
        raise ValueError("--off-policy-mix-ratio must be in [0,1]")
    if int(args.off_policy_buffer_size) <= 0:
        raise ValueError("--off-policy-buffer-size must be > 0")
    if int(args.off_policy_warmup_steps) < 0:
        raise ValueError("--off-policy-warmup-steps must be >= 0")
    if int(args.off_policy_min_buffer_groups) <= 0:
        raise ValueError("--off-policy-min-buffer-groups must be > 0")
    if int(args.off_policy_min_buffer_groups) > int(args.off_policy_buffer_size):
        raise ValueError("--off-policy-min-buffer-groups must be <= --off-policy-buffer-size")
    if int(args.async_checkpoint_eval_max_inflight) <= 0:
        raise ValueError("--async-checkpoint-eval-max-inflight must be > 0")
    if bool(args.async_checkpoint_eval) and int(args.save_every) <= 0:
        raise ValueError("--async-checkpoint-eval requires --save-every > 0")
    if bool(args.off_policy) and (bool(args.rl_reasoning) or bool(args.bootstrap_reasoning)):
        raise ValueError("off-policy and reasoning cannot be enabled together in the same run")


def _classification_prediction(payload: Mapping[str, Any]) -> tuple[str, list[str]]:
    answer = str(payload.get("answer") or "").strip().upper()
    answers = common.parse_mcq_letters(payload.get("answers") or payload.get("answer"))
    return answer, answers


def _change_awareness_score(text: str) -> float:
    normalized = common.normalize_text(text)
    if not normalized:
        return 0.0
    strong_tokens = (
        "damage",
        "damaged",
        "destroyed",
        "collapsed",
        "flood",
        "burned",
        "wildfire",
        "washed",
        "change",
        "impact",
        "debris",
        "rebuild",
        "restoration",
        "recovery",
    )
    if any(token in normalized for token in strong_tokens):
        return 1.0
    if any(token in normalized for token in ("no visible change", "unchanged", "no significant damage")):
        return 0.85
    return 0.25


def _score_query_payload(record: common.MixedTaskRecord, payload: Optional[dict[str, Any]], *, strict: bool) -> MixedScoreOutcome:
    target = json.loads(record.final_answer_json or "{}") if str(record.final_answer_json or "").strip() else {}
    if not isinstance(target, dict):
        target = {}
    if payload is None:
        return MixedScoreOutcome(reward=0.0, parse_success=False, task_correct=False, json_object_parsed=False)
    query_kind = common.detect_query_kind(record.task_name, record.metadata)
    if query_kind == "multi_choice_single_answer":
        pred_answer, _ = _classification_prediction(payload)
        gt_answer = str(target.get("answer") or "").strip().upper()
        exact = 1.0 if pred_answer and pred_answer == gt_answer else 0.0
        reward = exact if strict else (1.0 if exact else 0.05)
        return MixedScoreOutcome(
            reward=float(reward),
            parse_success=bool(pred_answer),
            task_correct=bool(exact),
            json_object_parsed=True,
            exact_match=exact,
            answer_set_f1=exact,
        )
    if query_kind == "multi_choice_multi_answer":
        _, pred_answers = _classification_prediction(payload)
        gt_answers = common.parse_mcq_letters(target.get("answers") or [])
        score = common.set_f1(gt_answers, pred_answers)
        reward = score if not strict else (1.0 if score >= 0.999 else 0.0)
        return MixedScoreOutcome(
            reward=float(reward),
            parse_success=bool(pred_answers),
            task_correct=score >= 0.999,
            json_object_parsed=True,
            exact_match=1.0 if score >= 0.999 else 0.0,
            answer_set_f1=score,
        )
    if query_kind == "count":
        pred_count = common.parse_number_from_text(payload.get("count"))
        gt_count = common.parse_number_from_text(target.get("count"))
        if pred_count is None or gt_count is None:
            return MixedScoreOutcome(reward=0.0, parse_success=False, task_correct=False, json_object_parsed=True)
        score = common.relative_count_score(gt_count, pred_count)
        exact_match = math.isclose(float(pred_count), float(gt_count), rel_tol=1e-4, abs_tol=1e-4)
        reward = score if not strict else (1.0 if exact_match else 0.0)
        return MixedScoreOutcome(
            reward=float(reward),
            parse_success=True,
            task_correct=bool(exact_match),
            json_object_parsed=True,
            exact_match=1.0 if exact_match else 0.0,
            count_score=score,
        )
    if query_kind == "description":
        keys = ("disaster", "building", "road", "vegetation", "water_body", "agriculture", "conclusion")
        field_f1s: list[float] = []
        filled = 0
        predicted_chunks: list[str] = []
        target_chunks: list[str] = []
        for key in keys:
            pred_value = str(payload.get(key) or "").strip()
            gt_value = str(target.get(key) or "").strip()
            if pred_value:
                filled += 1
            field_f1s.append(common.token_f1(gt_value, pred_value))
            predicted_chunks.append(pred_value)
            target_chunks.append(gt_value)
        coverage = filled / float(len(keys))
        token_score = fmean(field_f1s) if field_f1s else 0.0
        concise = common.brevity_score(" ".join(target_chunks), " ".join(predicted_chunks), lower=0.5, upper=1.5)
        change_awareness = _change_awareness_score(" ".join(predicted_chunks))
        reward = (0.55 * token_score) + (0.20 * coverage) + (0.15 * concise) + (0.10 * change_awareness)
        if strict:
            reward = 1.0 if token_score >= 0.95 and coverage >= 0.99 else reward * 0.25
        return MixedScoreOutcome(
            reward=float(common.clamp(reward)),
            parse_success=filled > 0,
            task_correct=token_score >= 0.80 and coverage >= 0.85,
            json_object_parsed=True,
            description_token_f1=token_score,
            description_field_coverage=coverage,
            change_awareness_score=change_awareness,
            concise_score=concise,
        )
    if query_kind == "recovery":
        needs_pred = payload.get("needs_recovery")
        needs_gt = target.get("needs_recovery")
        if isinstance(needs_pred, bool):
            needs_pred_bool = needs_pred
        else:
            needs_pred_bool = str(needs_pred).strip().lower() in {"true", "1", "yes"}
        needs_gt_bool = bool(needs_gt)
        needs_accuracy = 1.0 if needs_pred_bool == needs_gt_bool else 0.0
        immediate_pred = str(payload.get("immediate_recovery") or "").strip()
        long_pred = str(payload.get("long_term_recovery") or "").strip()
        immediate_gt = str(target.get("immediate_recovery") or "").strip()
        long_gt = str(target.get("long_term_recovery") or "").strip()
        action_score = fmean(
            [
                common.token_f1(immediate_gt, immediate_pred),
                common.token_f1(long_gt, long_pred),
            ]
        )
        concise = common.clamp(
            min(
                1.0,
                50.0 / float(max(1, len(common.normalize_text(immediate_pred).split()))),
                50.0 / float(max(1, len(common.normalize_text(long_pred).split()))),
            )
        )
        change_awareness = _change_awareness_score(f"{immediate_pred} {long_pred}")
        reward = (0.40 * needs_accuracy) + (0.35 * action_score) + (0.15 * concise) + (0.10 * change_awareness)
        if strict:
            reward = 1.0 if needs_accuracy >= 0.999 and action_score >= 0.95 else reward * 0.25
        return MixedScoreOutcome(
            reward=float(common.clamp(reward)),
            parse_success=isinstance(payload, dict),
            task_correct=needs_accuracy >= 0.999 and action_score >= 0.75,
            json_object_parsed=True,
            exact_match=needs_accuracy,
            recovery_action_score=action_score,
            needs_recovery_accuracy=needs_accuracy,
            change_awareness_score=change_awareness,
            concise_score=concise,
        )
    pred_answer = str(payload.get("answer") or "").strip()
    gt_answer = str(target.get("answer") or "").strip()
    token_score = common.token_f1(gt_answer, pred_answer)
    concise = common.brevity_score(gt_answer, pred_answer)
    reward = token_score if strict else ((0.85 * token_score) + (0.15 * concise))
    return MixedScoreOutcome(
        reward=float(common.clamp(reward)),
        parse_success=bool(pred_answer),
        task_correct=token_score >= 0.95,
        json_object_parsed=True,
        exact_match=1.0 if token_score >= 0.999 else 0.0,
        concise_score=concise,
    )


def _point_distance_score(predicted: list[Any], ground_truth: list[Any]) -> float:
    if not predicted and not ground_truth:
        return 1.0
    if not predicted or not ground_truth:
        return 0.0
    distances: list[float] = []
    used_gt: set[int] = set()
    for pred in predicted:
        best_distance = None
        best_index = -1
        for index, gt in enumerate(ground_truth):
            if index in used_gt:
                continue
            distance = math.dist((float(pred.x), float(pred.y)), (float(gt.x), float(gt.y)))
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_index = index
        if best_distance is not None and best_index >= 0:
            used_gt.add(best_index)
            distances.append(best_distance)
    if not distances:
        return 0.0
    return common.clamp(1.0 - (fmean(distances) / 0.25))


def _score_spatial_rollout(record: common.MixedTaskRecord, rollout: Any) -> MixedScoreOutcome:
    gt_boxes = common.deserialize_boxes(record.answer_boxes_json)
    gt_points = common.deserialize_points(record.answer_points_json)
    if record.skill == "detect":
        output = getattr(rollout, "output", None)
        pred_boxes = output.objects if isinstance(output, DetectOutput) else []
        detect_f1 = common.reward_f1(pred_boxes, gt_boxes)
        detect_miou = common.reward_miou(pred_boxes, gt_boxes)
        reward = (0.55 * detect_f1) + (0.45 * detect_miou)
        return MixedScoreOutcome(
            reward=float(common.clamp(reward)),
            parse_success=True,
            task_correct=detect_f1 >= 0.80,
            json_object_parsed=False,
            detect_f1=detect_f1,
            detect_miou=detect_miou,
        )
    output = getattr(rollout, "output", None)
    pred_points = output.points if isinstance(output, PointOutput) else []
    if gt_boxes:
        point_f1 = common.reward_f1_points(pred_points, gt_boxes)
        distance_score = point_f1
    else:
        point_f1 = 0.0
        distance_score = _point_distance_score(pred_points, gt_points)
    reward = max(point_f1, distance_score)
    return MixedScoreOutcome(
        reward=float(common.clamp(reward)),
        parse_success=True,
        task_correct=reward >= 0.80,
        json_object_parsed=False,
        point_f1=point_f1,
        point_distance_score=distance_score,
    )


def score_rollout_for_record(record: common.MixedTaskRecord, rollout: Any, *, strict_query_rewards: bool) -> MixedScoreOutcome:
    if record.skill == "query":
        output = getattr(rollout, "output", None)
        answer = output.answer if isinstance(output, QueryOutput) else ""
        return _score_query_payload(record, common.parse_prediction_json(str(answer or "")), strict=strict_query_rewards)
    return _score_spatial_rollout(record, rollout)


def score_prediction_for_record(
    record: common.MixedTaskRecord,
    *,
    query_answer: str = "",
    detect_boxes: Optional[list[dict[str, Any]]] = None,
    point_coords: Optional[list[dict[str, Any]]] = None,
    strict_query_rewards: bool,
) -> MixedScoreOutcome:
    if record.skill == "query":
        rollout = SimpleNamespace(output=QueryOutput(answer=str(query_answer or "")))
        return score_rollout_for_record(record, rollout, strict_query_rewards=strict_query_rewards)
    if record.skill == "detect":
        objects = [
            common.DetectAnnotation(
                x_min=common.clamp(float(item.get("x_min"))),
                y_min=common.clamp(float(item.get("y_min"))),
                x_max=common.clamp(float(item.get("x_max"))),
                y_max=common.clamp(float(item.get("y_max"))),
            )
            for item in (detect_boxes or [])
        ]
        rollout = SimpleNamespace(output=DetectOutput(objects=objects))
        return score_rollout_for_record(record, rollout, strict_query_rewards=strict_query_rewards)
    points = [
        common.PointAnnotation(
            x=common.clamp(float(item.get("x"))),
            y=common.clamp(float(item.get("y"))),
        )
        for item in (point_coords or [])
    ]
    rollout = SimpleNamespace(output=PointOutput(points=points))
    return score_rollout_for_record(record, rollout, strict_query_rewards=strict_query_rewards)


def _record_metrics(bucket: MetricBucket, outcome: MixedScoreOutcome) -> None:
    bucket.reward_values.append(float(outcome.reward))
    if outcome.parse_success:
        bucket.parse_success_count += 1
    if outcome.json_object_parsed:
        bucket.json_object_count += 1
    if outcome.task_correct:
        bucket.task_correct_count += 1
    bucket.exact_match_values.append(float(outcome.exact_match))
    bucket.answer_set_f1_values.append(float(outcome.answer_set_f1))
    bucket.count_values.append(float(outcome.count_score))
    bucket.detect_f1_values.append(float(outcome.detect_f1))
    bucket.detect_miou_values.append(float(outcome.detect_miou))
    bucket.point_f1_values.append(float(outcome.point_f1))
    bucket.point_distance_values.append(float(outcome.point_distance_score))
    bucket.description_token_f1_values.append(float(outcome.description_token_f1))
    bucket.description_field_coverage_values.append(float(outcome.description_field_coverage))
    bucket.recovery_action_values.append(float(outcome.recovery_action_score))
    bucket.needs_recovery_values.append(float(outcome.needs_recovery_accuracy))
    bucket.change_awareness_values.append(float(outcome.change_awareness_score))
    bucket.concise_values.append(float(outcome.concise_score))


def _finalize_bucket(bucket: MetricBucket) -> dict[str, float]:
    total = len(bucket.reward_values)
    return {
        "samples": float(total),
        "reward_mean": fmean(bucket.reward_values) if bucket.reward_values else 0.0,
        "parse_success_rate": bucket.parse_success_count / float(max(1, total)),
        "json_object_rate": bucket.json_object_count / float(max(1, total)),
        "task_accuracy": bucket.task_correct_count / float(max(1, total)),
        "exact_match_mean": fmean(bucket.exact_match_values) if bucket.exact_match_values else 0.0,
        "answer_set_f1_mean": fmean(bucket.answer_set_f1_values) if bucket.answer_set_f1_values else 0.0,
        "count_score_mean": fmean(bucket.count_values) if bucket.count_values else 0.0,
        "detect_f1_mean": fmean(bucket.detect_f1_values) if bucket.detect_f1_values else 0.0,
        "detect_miou_mean": fmean(bucket.detect_miou_values) if bucket.detect_miou_values else 0.0,
        "point_f1_mean": fmean(bucket.point_f1_values) if bucket.point_f1_values else 0.0,
        "point_distance_mean": fmean(bucket.point_distance_values) if bucket.point_distance_values else 0.0,
        "description_token_f1_mean": fmean(bucket.description_token_f1_values) if bucket.description_token_f1_values else 0.0,
        "description_field_coverage_mean": (
            fmean(bucket.description_field_coverage_values) if bucket.description_field_coverage_values else 0.0
        ),
        "recovery_action_mean": fmean(bucket.recovery_action_values) if bucket.recovery_action_values else 0.0,
        "needs_recovery_accuracy_mean": fmean(bucket.needs_recovery_values) if bucket.needs_recovery_values else 0.0,
        "change_awareness_mean": fmean(bucket.change_awareness_values) if bucket.change_awareness_values else 0.0,
        "concise_score_mean": fmean(bucket.concise_values) if bucket.concise_values else 0.0,
    }


def summarize_outcomes(records: list[common.MixedTaskRecord], outcomes: list[MixedScoreOutcome]) -> dict[str, Any]:
    overall = MetricBucket()
    by_skill: dict[str, MetricBucket] = {}
    by_task: dict[str, MetricBucket] = {}
    for record, outcome in zip(records, outcomes):
        _record_metrics(overall, outcome)
        by_skill.setdefault(record.skill, MetricBucket())
        _record_metrics(by_skill[record.skill], outcome)
        by_task.setdefault(record.task_name, MetricBucket())
        _record_metrics(by_task[record.task_name], outcome)
    return {
        "overall": _finalize_bucket(overall),
        "by_skill": {key: _finalize_bucket(bucket) for key, bucket in sorted(by_skill.items())},
        "by_task": {key: _finalize_bucket(bucket) for key, bucket in sorted(by_task.items())},
    }


def compose_train_groups(
    *,
    on_policy_groups: list[Any],
    replay_groups: deque[Any],
    off_policy: bool,
    off_policy_mix_ratio: float,
    off_policy_warmup_steps: int,
    off_policy_min_buffer_groups: int,
    global_step: int,
    rng: random.Random,
) -> tuple[list[Any], int]:
    if (
        not on_policy_groups
        or not off_policy
        or off_policy_mix_ratio <= 0.0
        or global_step < off_policy_warmup_steps
        or len(replay_groups) < off_policy_min_buffer_groups
    ):
        return list(on_policy_groups), 0
    off_policy_count = min(
        max(1, int(round(len(on_policy_groups) * off_policy_mix_ratio))),
        len(on_policy_groups),
        len(replay_groups),
    )
    keep_count = max(0, len(on_policy_groups) - off_policy_count)
    selected_on_policy = (
        list(on_policy_groups)
        if keep_count >= len(on_policy_groups)
        else rng.sample(list(on_policy_groups), k=keep_count)
    )
    mixed = selected_on_policy + rng.sample(list(replay_groups), k=off_policy_count)
    rng.shuffle(mixed)
    return mixed, off_policy_count


def _query_prediction_payload(answer_text: str) -> tuple[Any, Any]:
    parsed = common.parse_prediction_json(answer_text)
    return parsed, answer_text


def _detect_prediction_payload(output: Any) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    objects = output.objects if isinstance(output, DetectOutput) else []
    serialized = [
        {
            "x_min": float(item.x_min),
            "y_min": float(item.y_min),
            "x_max": float(item.x_max),
            "y_max": float(item.y_max),
        }
        for item in objects
    ]
    return serialized, serialized


def _point_prediction_payload(output: Any) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    points = output.points if isinstance(output, PointOutput) else []
    serialized = [
        {
            "x": float(item.x),
            "y": float(item.y),
        }
        for item in points
    ]
    return serialized, serialized


def _ground_truth_payload(record: common.MixedTaskRecord) -> Any:
    if record.skill == "query":
        return json.loads(record.final_answer_json or "{}")
    if record.skill == "detect":
        return json.loads(record.answer_boxes_json or "[]")
    return {
        "points": json.loads(record.answer_points_json or "[]"),
        "boxes": json.loads(record.answer_boxes_json or "[]"),
    }


def _eval_row(
    *,
    record: common.MixedTaskRecord,
    prediction: Any,
    raw_response: Any,
    outcome: MixedScoreOutcome,
) -> dict[str, Any]:
    return {
        "row_id": record.row_id,
        "split": record.split,
        "task_name": record.task_name,
        "task_family": record.task_family,
        "skill": record.skill,
        "question": record.question,
        "object_name": record.object_name,
        "ground_truth": _ground_truth_payload(record),
        "prediction": prediction,
        "raw_response": raw_response,
        "grading": asdict(outcome),
        "reward": float(outcome.reward),
        "parse_success": bool(outcome.parse_success),
        "task_correct": bool(outcome.task_correct),
    }


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _write_sample_jsonl(path: Path, rows: list[Mapping[str, Any]], *, sample_count: int) -> None:
    if sample_count <= 0:
        return
    _write_jsonl(path, rows[: int(sample_count)])


def _extract_metric_sections(payload: Mapping[str, Any]) -> dict[str, Any]:
    metrics = payload.get("metrics")
    if isinstance(metrics, dict):
        return metrics
    return dict(payload)


def _overall_reward_mean(payload: Mapping[str, Any]) -> float:
    if "overall_reward_mean" in payload:
        return float(payload.get("overall_reward_mean") or 0.0)
    metrics = _extract_metric_sections(payload)
    overall = metrics.get("overall")
    if isinstance(overall, dict):
        return float(overall.get("reward_mean") or 0.0)
    return 0.0


def _append_eval_history(
    *,
    path: Path,
    stage_name: str,
    split_name: str,
    step: int,
    checkpoint_step: Optional[int],
    payload: Mapping[str, Any],
    source: str,
    predictions_jsonl: str = "",
) -> None:
    metrics = _extract_metric_sections(payload)
    record = {
        "stage": stage_name,
        "split": split_name,
        "step": int(step),
        "checkpoint_step": None if checkpoint_step is None else int(checkpoint_step),
        "source": source,
        "overall": metrics.get("overall", {}),
        "by_task": metrics.get("by_task", {}),
        "by_skill": metrics.get("by_skill", {}),
        "predictions_jsonl": predictions_jsonl,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _copy_sample_predictions(*, source_path: Path, dest_path: Path, sample_count: int) -> None:
    if sample_count <= 0 or not source_path.exists():
        return
    lines: list[str] = []
    with source_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                lines.append(line)
            if len(lines) >= int(sample_count):
                break
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_text("".join(lines), encoding="utf-8")


def _build_async_checkpoint_eval_command(
    *,
    args: argparse.Namespace,
    stage: StageConfig,
    finetune_id: str,
    split_name: str,
    checkpoint_step: int,
    metrics_json_path: Path,
    predictions_jsonl_path: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        str((Path(__file__).resolve().parent / "benchmark_disaster_m3_mixed.py").resolve()),
        "--env-file",
        str(args.env_file),
        "--base-url",
        str(args.base_url),
        "--dataset-dir",
        str(args.dataset_dir),
        "--split",
        str(split_name),
        "--finetune-id",
        str(finetune_id),
        "--checkpoint-step",
        str(int(checkpoint_step)),
        "--temperature",
        "0.0",
        "--top-p",
        "1.0",
        "--max-tokens",
        str(int(stage.max_tokens)),
        "--output-json",
        str(metrics_json_path),
        "--predictions-jsonl",
        str(predictions_jsonl_path),
        "--checkpoint-fallback-policy",
        "exact",
        "--checkpoint-ready-max-wait-s",
        "300",
        "--checkpoint-ready-poll-interval-s",
        "5",
        "--max-samples",
        str(int(args.eval_max_samples) if int(args.eval_max_samples) > 0 else 0),
        "--no-progress",
    ]
    cmd.append("--reasoning" if bool(stage.reasoning) else "--no-reasoning")
    return cmd


def _ingest_async_checkpoint_eval_results(
    *,
    args: argparse.Namespace,
    run: Any,
    stage: StageConfig,
    stage_dir: Path,
    eval_history_path: Path,
    results: list[CheckpointEvalResult],
    best_metric_value: Optional[float],
    best_checkpoint_step: Optional[int],
    latest_checkpoint_step: Optional[int],
) -> tuple[Optional[float], Optional[int], Optional[int], int]:
    success_count = 0
    for result in results:
        step_for_log = int(result.metadata.get("step_for_log", result.checkpoint_step))
        split_name = str(result.metadata.get("split_name") or args.val_split)
        if result.status != "succeeded" or result.metrics_payload is None:
            print(
                f"async checkpoint eval failed stage={stage.name} step={step_for_log} "
                f"checkpoint_step={result.checkpoint_step} log={result.stdout_log_path}"
            )
            continue
        success_count += 1
        metric_value = _overall_reward_mean(result.metrics_payload)
        latest_checkpoint_step = max(int(latest_checkpoint_step or 0), int(result.checkpoint_step))
        run.summary[f"{stage.name}_latest_checkpoint_step"] = int(latest_checkpoint_step)
        run.summary[f"{stage.name}_latest_async_eval_metric"] = float(metric_value)
        run.summary[f"{stage.name}_latest_async_eval_step"] = int(step_for_log)
        _append_eval_history(
            path=eval_history_path,
            stage_name=stage.name,
            split_name=split_name,
            step=step_for_log,
            checkpoint_step=int(result.checkpoint_step),
            payload=result.metrics_payload,
            source="async",
            predictions_jsonl=str(result.predictions_jsonl_path),
        )
        _copy_sample_predictions(
            source_path=result.predictions_jsonl_path,
            dest_path=stage_dir / "eval_samples" / f"step_{step_for_log:06d}_{split_name}.jsonl",
            sample_count=int(args.eval_sample_count),
        )
        if split_name == str(args.val_split) and (best_metric_value is None or metric_value > best_metric_value):
            best_metric_value = metric_value
            best_checkpoint_step = int(result.checkpoint_step)
            run.summary[f"{stage.name}_best_metric"] = float(metric_value)
            run.summary[f"{stage.name}_best_checkpoint_step"] = int(result.checkpoint_step)
            run.summary[f"{stage.name}_best_metric_step"] = int(step_for_log)
        print(
            f"async checkpoint eval completed stage={stage.name} step={step_for_log} "
            f"checkpoint_step={result.checkpoint_step} reward={metric_value:.4f}"
        )
    return best_metric_value, best_checkpoint_step, latest_checkpoint_step, success_count


def _skill_weights(records: list[common.MixedTaskRecord]) -> list[float]:
    counts = Counter(record.skill for record in records)
    return [1.0 / float(max(1, counts[record.skill])) for record in records]


def _rollouts_batch_with_retry(
    *,
    finetune: Any,
    requests: list[Any],
    ground_truths: list[Any],
    num_rollouts: int,
    max_workers: int,
    retries: int,
    backoff_s: float,
    context: str,
) -> Any:
    worker_count = max(1, min(max_workers, len(requests)))
    for attempt in range(max(0, int(retries)) + 1):
        try:
            return finetune.rollouts_batch(
                requests=requests,
                ground_truths=ground_truths,
                num_rollouts=num_rollouts,
                max_workers=worker_count,
            )
        except (TunaAPIError, TunaNetworkError) as exc:
            should_retry = isinstance(exc, TunaNetworkError) or (
                isinstance(exc, TunaAPIError) and exc.status_code == 429
            ) or "too many requests" in str(exc).lower()
            if not should_retry or attempt >= int(retries):
                print(
                    f"{context}: rollouts_batch failed with no further retries. "
                    f"attempt={attempt + 1}/{int(retries) + 1} workers={worker_count} "
                    f"details={common.error_message(exc)}"
                )
                raise
            delay = max(0.1, float(backoff_s)) * (2**attempt)
            next_workers = max(1, worker_count // 2)
            print(
                f"{context}: retrying rollouts_batch attempt={attempt + 1}/{int(retries) + 1} "
                f"workers={worker_count}->{next_workers} sleep={delay:.2f}s details={common.error_message(exc)}"
            )
            time.sleep(delay)
            worker_count = next_workers


def evaluate_split(
    *,
    finetune: Any,
    records: list[common.MixedTaskRecord],
    stage: StageConfig,
    max_workers: int,
    image_quality: int,
    max_objects: int,
    max_samples: Optional[int],
    seed: int,
    retries: int,
    backoff_s: float,
    predictions_path: Optional[Path] = None,
    sample_output_path: Optional[Path] = None,
    sample_count: int = 25,
) -> dict[str, Any]:
    selected = list(records)
    random.Random(seed).shuffle(selected)
    if max_samples is not None and max_samples > 0:
        selected = selected[: int(max_samples)]
    if not selected:
        return {"overall": {"reward_mean": 0.0, "samples": 0.0}, "by_skill": {}, "by_task": {}}
    requests: list[Any] = []
    ground_truths: list[Any] = []
    active_records: list[common.MixedTaskRecord] = []
    for record in selected:
        try:
            request, ground_truth = common.prepare_request_for_record(
                record,
                temperature=0.0,
                top_p=1.0,
                max_tokens=int(stage.max_tokens),
                reasoning=bool(stage.reasoning),
                image_quality=image_quality,
                max_objects=max_objects,
            )
        except (FileNotFoundError, OSError) as exc:
            print(f"eval skip row_id={record.row_id} reason=image_load details={exc}")
            continue
        requests.append(request)
        ground_truths.append(None)
        active_records.append(record)
    if not requests:
        return {"overall": {"reward_mean": 0.0, "samples": 0.0}, "by_skill": {}, "by_task": {}}
    results = _rollouts_batch_with_retry(
        finetune=finetune,
        requests=requests,
        ground_truths=ground_truths,
        num_rollouts=1,
        max_workers=max_workers,
        retries=retries,
        backoff_s=backoff_s,
        context="eval",
    )
    outcomes: list[MixedScoreOutcome] = []
    used_records: list[common.MixedTaskRecord] = []
    eval_rows: list[dict[str, Any]] = []
    for record, result in zip(active_records, results):
        if not result.rollouts:
            continue
        rollout = result.rollouts[0]
        outcome = score_rollout_for_record(record, rollout, strict_query_rewards=False)
        if record.skill == "query":
            output = getattr(rollout, "output", None)
            answer_text = output.answer if isinstance(output, QueryOutput) else ""
            prediction, raw_response = _query_prediction_payload(str(answer_text or ""))
        elif record.skill == "detect":
            prediction, raw_response = _detect_prediction_payload(getattr(rollout, "output", None))
        else:
            prediction, raw_response = _point_prediction_payload(getattr(rollout, "output", None))
        outcomes.append(outcome)
        used_records.append(record)
        eval_rows.append(
            _eval_row(
                record=record,
                prediction=prediction,
                raw_response=raw_response,
                outcome=outcome,
            )
        )
    if predictions_path is not None:
        _write_jsonl(predictions_path, eval_rows)
    if sample_output_path is not None:
        _write_sample_jsonl(sample_output_path, eval_rows, sample_count=sample_count)
    return summarize_outcomes(used_records, outcomes)


def _stage_run_dir(run_root: Path, *, label: str) -> Path:
    out = run_root / label
    out.mkdir(parents=True, exist_ok=True)
    return out


def _train_stage(
    *,
    finetune: Any,
    train_records: list[common.MixedTaskRecord],
    val_records: list[common.MixedTaskRecord],
    test_records: list[common.MixedTaskRecord],
    stage: StageConfig,
    args: argparse.Namespace,
    run: Any,
    run_dir: Path,
    seed_offset: int,
) -> dict[str, Any]:
    if stage.num_steps <= 0:
        return {"stage": stage.name, "skipped": True}
    rng = random.Random(int(args.seed) + int(seed_offset))
    train_weights = _skill_weights(train_records)
    best_val_reward: Optional[float] = None
    best_saved_step: Optional[int] = None
    latest_saved_step: Optional[int] = None
    replay_buffer: deque[Any] = deque(maxlen=int(args.off_policy_buffer_size))
    async_eval_jobs: list[DispatchHandle] = []
    async_eval_success_count = 0
    stage_dir = run_dir
    eval_history_path = stage_dir / "eval_history.jsonl"
    off_policy_enabled = bool(args.off_policy and stage.name in {"rl", "rl_only"})
    for global_step in range(stage.num_steps):
        if async_eval_jobs:
            async_eval_jobs, completed_async_results = poll_checkpoint_eval_jobs(async_eval_jobs)
            (
                best_val_reward,
                best_saved_step,
                latest_saved_step,
                completed_success_count,
            ) = _ingest_async_checkpoint_eval_results(
                args=args,
                run=run,
                stage=stage,
                stage_dir=stage_dir,
                eval_history_path=eval_history_path,
                results=completed_async_results,
                best_metric_value=best_val_reward,
                best_checkpoint_step=best_saved_step,
                latest_checkpoint_step=latest_saved_step,
            )
            async_eval_success_count += int(completed_success_count)
        batch = rng.choices(train_records, weights=train_weights, k=int(stage.batch_size))
        requests: list[Any] = []
        ground_truths: list[Any] = []
        active_records: list[common.MixedTaskRecord] = []
        for record in batch:
            try:
                request, ground_truth = common.prepare_request_for_record(
                    record,
                    temperature=stage.temperature,
                    top_p=stage.top_p,
                    max_tokens=stage.max_tokens,
                    reasoning=stage.reasoning,
                    image_quality=int(args.image_quality),
                    max_objects=int(args.detect_max_objects),
                )
            except (FileNotFoundError, OSError) as exc:
                print(f"train skip row_id={record.row_id} reason=image_load details={exc}")
                continue
            requests.append(request)
            ground_truths.append(
                ground_truth if stage.use_server_ground_truth and record.skill in {"point", "detect"} else None
            )
            active_records.append(record)
        if not requests:
            print(f"stage={stage.name} step={global_step}: no usable requests; skipping")
            continue
        try:
            results = _rollouts_batch_with_retry(
                finetune=finetune,
                requests=requests,
                ground_truths=ground_truths,
                num_rollouts=stage.group_size,
                max_workers=min(int(args.max_workers), len(requests)),
                retries=int(args.rollout_retries),
                backoff_s=float(args.rollout_retry_backoff_s),
                context=f"{stage.name} step={global_step}",
            )
        except (TunaAPIError, TunaNetworkError) as exc:
            print(f"stage={stage.name} step={global_step}: rollouts failed. details={common.error_message(exc)}")
            continue
        train_outcomes: list[MixedScoreOutcome] = []
        train_records_used: list[common.MixedTaskRecord] = []
        groups: list[Any] = []
        for record, result in zip(active_records, results):
            if not result.rollouts:
                continue
            rewards: list[float] = []
            if stage.use_server_ground_truth and record.skill in {"point", "detect"} and result.rewards is not None:
                rewards = [float(value) for value in result.rewards]
                for rollout in result.rollouts:
                    outcome = score_rollout_for_record(record, rollout, strict_query_rewards=stage.strict_query_rewards)
                    train_outcomes.append(outcome)
                    train_records_used.append(record)
            else:
                for rollout in result.rollouts:
                    outcome = score_rollout_for_record(record, rollout, strict_query_rewards=stage.strict_query_rewards)
                    rewards.append(float(outcome.reward))
                    train_outcomes.append(outcome)
                    train_records_used.append(record)
            groups.append(result.to_group(rewards=rewards))
        if not groups:
            print(f"stage={stage.name} step={global_step}: no train groups produced; skipping")
            continue
        train_groups, off_policy_count = compose_train_groups(
            on_policy_groups=groups,
            replay_groups=replay_buffer,
            off_policy=off_policy_enabled,
            off_policy_mix_ratio=float(args.off_policy_mix_ratio),
            off_policy_warmup_steps=int(args.off_policy_warmup_steps),
            off_policy_min_buffer_groups=int(args.off_policy_min_buffer_groups),
            global_step=global_step,
            rng=rng,
        )
        try:
            train_out = finetune.train_step(groups=train_groups, lr=stage.lr)
        except (TunaAPIError, TunaNetworkError) as exc:
            print(f"stage={stage.name} step={global_step}: train_step failed. details={common.error_message(exc)}")
            continue
        if off_policy_enabled:
            replay_buffer.extend(groups)
        train_summary = summarize_outcomes(train_records_used, train_outcomes)
        train_metrics = {
            f"{stage.name}_train_reward_mean": float(train_summary["overall"]["reward_mean"]),
            f"{stage.name}_train_parse_success_rate": float(train_summary["overall"]["parse_success_rate"]),
            f"{stage.name}_train_task_accuracy": float(train_summary["overall"]["task_accuracy"]),
            f"{stage.name}_on_policy_groups": float(len(train_groups) - off_policy_count),
            f"{stage.name}_off_policy_groups": float(off_policy_count),
            f"{stage.name}_off_policy_group_fraction": off_policy_count / float(max(1, len(train_groups))),
            f"{stage.name}_replay_buffer_size": float(len(replay_buffer)),
            f"{stage.name}_kl": float(train_out.kl or 0.0),
            f"{stage.name}_router_kl": float(train_out.router_kl or 0.0),
            f"{stage.name}_grad_norm": float(train_out.grad_norm or 0.0),
        }
        wandb.log(train_metrics, step=global_step)
        print(
            f"stage={stage.name} step={global_step + 1}/{stage.num_steps} "
            f"reward={train_metrics[f'{stage.name}_train_reward_mean']:.4f} "
            f"parse={train_metrics[f'{stage.name}_train_parse_success_rate']:.4f} "
            f"offp={off_policy_count}/{len(train_groups)} "
            f"kl={train_metrics[f'{stage.name}_kl']:.4f}"
        )
        saved_this_step: Optional[int] = None
        if int(args.eval_every) > 0 and (global_step + 1) % int(args.eval_every) == 0:
            if bool(args.async_checkpoint_eval):
                saved_this_step = common.checkpoint_save_step(
                    finetune=finetune,
                    context=f"{stage.name} async-eval checkpoint save step={global_step + 1}",
                )
                if saved_this_step is not None:
                    latest_saved_step = int(saved_this_step)
                    run.summary[f"{stage.name}_latest_saved_step"] = int(saved_this_step)
                    job = dispatch_checkpoint_eval(
                        trainer=f"disaster_m3_{stage.name}",
                        finetune_id=str(finetune.finetune_id),
                        checkpoint_step=int(saved_this_step),
                        selection_metric="overall_reward_mean",
                        base_dir=str(args.async_checkpoint_eval_dir),
                        command_builder=lambda metrics_json_path, predictions_jsonl_path, _stdout_log_path: _build_async_checkpoint_eval_command(
                            args=args,
                            stage=stage,
                            finetune_id=str(finetune.finetune_id),
                            split_name=str(args.val_split),
                            checkpoint_step=int(saved_this_step),
                            metrics_json_path=metrics_json_path,
                            predictions_jsonl_path=predictions_jsonl_path,
                        ),
                        metadata={
                            "step_for_log": int(global_step + 1),
                            "split_name": str(args.val_split),
                        },
                        env_overrides={"MOONDREAM_API_KEY": str(args.api_key)},
                        max_inflight=int(args.async_checkpoint_eval_max_inflight),
                        inflight_jobs=async_eval_jobs,
                    )
                    if job is None:
                        print(
                            f"async checkpoint eval skipped stage={stage.name} step={global_step + 1} "
                            f"checkpoint_step={saved_this_step} reason=max_inflight"
                        )
                    else:
                        async_eval_jobs.append(job)
                        print(
                            f"async checkpoint eval dispatched stage={stage.name} step={global_step + 1} "
                            f"checkpoint_step={saved_this_step} job_dir={job.job_dir}"
                        )
            else:
                predictions_path = stage_dir / "eval_predictions" / f"step_{global_step + 1:06d}_{args.val_split}.jsonl"
                sample_output_path = stage_dir / "eval_samples" / f"step_{global_step + 1:06d}_{args.val_split}.jsonl"
                val_summary = evaluate_split(
                    finetune=finetune,
                    records=val_records,
                    stage=stage,
                    max_workers=min(int(args.max_workers), max(1, len(val_records))),
                    image_quality=int(args.image_quality),
                    max_objects=int(args.detect_max_objects),
                    max_samples=int(args.eval_max_samples) if int(args.eval_max_samples) > 0 else None,
                    seed=int(args.seed) + global_step,
                    retries=int(args.rollout_retries),
                    backoff_s=float(args.rollout_retry_backoff_s),
                    predictions_path=predictions_path,
                    sample_output_path=sample_output_path,
                    sample_count=int(args.eval_sample_count),
                )
                reward_mean = float(val_summary["overall"]["reward_mean"])
                wandb.log({f"{stage.name}_val_reward_mean": reward_mean}, step=global_step)
                _append_eval_history(
                    path=eval_history_path,
                    stage_name=stage.name,
                    split_name=str(args.val_split),
                    step=int(global_step + 1),
                    checkpoint_step=None,
                    payload=val_summary,
                    source="sync",
                    predictions_jsonl=str(predictions_path),
                )
                print(f"stage={stage.name} eval step={global_step + 1} reward={reward_mean:.4f}")
                if best_val_reward is None or reward_mean > best_val_reward:
                    best_val_reward = reward_mean
                    best_saved_step = common.checkpoint_save_step(
                        finetune=finetune,
                        context=f"{stage.name} best-checkpoint save step={global_step + 1}",
                    )
                    if best_saved_step is not None:
                        saved_this_step = int(best_saved_step)
                        latest_saved_step = int(best_saved_step)
                        run.summary[f"{stage.name}_best_saved_step"] = int(best_saved_step)
                        run.summary[f"{stage.name}_best_metric"] = float(reward_mean)
        if int(args.save_every) > 0 and (global_step + 1) % int(args.save_every) == 0 and saved_this_step is None:
            latest_saved_step = common.checkpoint_save_step(
                finetune=finetune,
                context=f"{stage.name} periodic-checkpoint save step={global_step + 1}",
            )
    final_saved_step = common.checkpoint_save_step(finetune=finetune, context=f"{stage.name} final-checkpoint save")
    if final_saved_step is not None:
        latest_saved_step = final_saved_step
    if bool(args.async_checkpoint_eval) and bool(args.async_checkpoint_eval_drain_on_exit):
        completed_async_results = drain_checkpoint_eval_jobs(async_eval_jobs)
        (
            best_val_reward,
            best_saved_step,
            latest_saved_step,
            completed_success_count,
        ) = _ingest_async_checkpoint_eval_results(
            args=args,
            run=run,
            stage=stage,
            stage_dir=stage_dir,
            eval_history_path=eval_history_path,
            results=completed_async_results,
            best_metric_value=best_val_reward,
            best_checkpoint_step=best_saved_step,
            latest_checkpoint_step=latest_saved_step,
        )
        async_eval_success_count += int(completed_success_count)
    final_val = evaluate_split(
        finetune=finetune,
        records=val_records,
        stage=stage,
        max_workers=min(int(args.max_workers), max(1, len(val_records))),
        image_quality=int(args.image_quality),
        max_objects=int(args.detect_max_objects),
        max_samples=int(args.eval_max_samples) if int(args.eval_max_samples) > 0 else None,
        seed=int(args.seed) + 900 + seed_offset,
        retries=int(args.rollout_retries),
        backoff_s=float(args.rollout_retry_backoff_s),
        predictions_path=stage_dir / "eval_predictions" / f"final_{args.val_split}.jsonl",
        sample_output_path=stage_dir / "eval_samples" / f"final_{args.val_split}.jsonl",
        sample_count=int(args.eval_sample_count),
    )
    final_test = evaluate_split(
        finetune=finetune,
        records=test_records,
        stage=stage,
        max_workers=min(int(args.max_workers), max(1, len(test_records))),
        image_quality=int(args.image_quality),
        max_objects=int(args.detect_max_objects),
        max_samples=int(args.eval_max_samples) if int(args.eval_max_samples) > 0 else None,
        seed=int(args.seed) + 1200 + seed_offset,
        retries=int(args.rollout_retries),
        backoff_s=float(args.rollout_retry_backoff_s),
        predictions_path=stage_dir / "eval_predictions" / f"final_{args.test_split}.jsonl",
        sample_output_path=stage_dir / "eval_samples" / f"final_{args.test_split}.jsonl",
        sample_count=int(args.eval_sample_count),
    )
    _append_eval_history(
        path=eval_history_path,
        stage_name=stage.name,
        split_name=str(args.val_split),
        step=int(stage.num_steps),
        checkpoint_step=latest_saved_step,
        payload=final_val,
        source="final_sync",
        predictions_jsonl=str(stage_dir / "eval_predictions" / f"final_{args.val_split}.jsonl"),
    )
    _append_eval_history(
        path=eval_history_path,
        stage_name=stage.name,
        split_name=str(args.test_split),
        step=int(stage.num_steps),
        checkpoint_step=latest_saved_step,
        payload=final_test,
        source="final_sync",
        predictions_jsonl=str(stage_dir / "eval_predictions" / f"final_{args.test_split}.jsonl"),
    )
    summary = {
        "stage": stage.name,
        "finetune_id": str(finetune.finetune_id),
        "best_saved_step": None if best_saved_step is None else int(best_saved_step),
        "latest_saved_step": None if latest_saved_step is None else int(latest_saved_step),
        "best_val_reward": None if best_val_reward is None else float(best_val_reward),
        "async_checkpoint_eval_enabled": bool(args.async_checkpoint_eval),
        "async_checkpoint_eval_success_count": int(async_eval_success_count),
        "eval_history_path": str(eval_history_path),
        "final_val": final_val,
        "final_test": final_test,
    }
    common.write_json(run_dir / f"{stage.name}_summary.json", summary)
    run.summary[f"{stage.name}_final_val_reward_mean"] = float(final_val["overall"]["reward_mean"])
    run.summary[f"{stage.name}_final_test_reward_mean"] = float(final_test["overall"]["reward_mean"])
    run.summary[f"{stage.name}_async_checkpoint_eval_enabled"] = bool(args.async_checkpoint_eval)
    run.summary[f"{stage.name}_async_checkpoint_eval_success_count"] = int(async_eval_success_count)
    return summary


def _resolve_finetune(client: TunaClient, *, finetune_id: str, finetune_name: str, rank: int) -> Any:
    if str(finetune_id or "").strip():
        return client.get_finetune(str(finetune_id).strip())
    return client.create_finetune(name=str(finetune_name).strip(), rank=int(rank))


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    _validate_args(args)
    bootstrap_stage, rl_stage, rl_only_stage = _stage_configs(args)
    args.api_key = common.resolve_api_key(api_key=args.api_key, api_key_env_var=args.api_key_env_var, env_file=args.env_file)
    train_records = common.load_mixed_records(dataset_dir=args.dataset_dir, split_name=args.train_split)
    val_records = common.load_mixed_records(dataset_dir=args.dataset_dir, split_name=args.val_split)
    test_records = common.load_mixed_records(dataset_dir=args.dataset_dir, split_name=args.test_split)

    run_root = Path(args.run_output_dir).expanduser().resolve() / args.finetune_name
    run_root.mkdir(parents=True, exist_ok=True)
    common.write_json(
        run_root / "run_config.json",
        {
            "mode": args.mode,
            "dataset_dir": str(args.dataset_dir),
            "train_count": len(train_records),
            "val_count": len(val_records),
            "test_count": len(test_records),
            "rank": int(args.rank),
            "max_workers": int(args.max_workers),
            "off_policy": bool(args.off_policy),
            "off_policy_mix_ratio": float(args.off_policy_mix_ratio),
            "off_policy_buffer_size": int(args.off_policy_buffer_size),
            "off_policy_warmup_steps": int(args.off_policy_warmup_steps),
            "off_policy_min_buffer_groups": int(args.off_policy_min_buffer_groups),
            "async_checkpoint_eval": bool(args.async_checkpoint_eval),
            "async_checkpoint_eval_dir": str(args.async_checkpoint_eval_dir),
            "eval_every": int(args.eval_every),
            "save_every": int(args.save_every),
            "eval_sample_count": int(args.eval_sample_count),
            "bootstrap_stage": bootstrap_stage.__dict__,
            "rl_stage": rl_stage.__dict__,
            "rl_only_stage": rl_only_stage.__dict__,
        },
    )

    client = TunaClient(api_key=args.api_key, base_url=args.base_url, timeout=float(args.timeout))
    run = wandb.init(
        project=str(args.wandb_project),
        name=str(args.wandb_run_name or args.finetune_name),
        config={
            "mode": args.mode,
            "dataset_dir": str(args.dataset_dir),
            "rank": int(args.rank),
            "base_url": str(args.base_url),
            "max_workers": int(args.max_workers),
            "off_policy": bool(args.off_policy),
            "off_policy_mix_ratio": float(args.off_policy_mix_ratio),
            "off_policy_buffer_size": int(args.off_policy_buffer_size),
            "off_policy_warmup_steps": int(args.off_policy_warmup_steps),
            "off_policy_min_buffer_groups": int(args.off_policy_min_buffer_groups),
            "async_checkpoint_eval": bool(args.async_checkpoint_eval),
            "async_checkpoint_eval_dir": str(args.async_checkpoint_eval_dir),
            "eval_every": int(args.eval_every),
            "save_every": int(args.save_every),
            "eval_sample_count": int(args.eval_sample_count),
        },
    )
    try:
        summaries: dict[str, Any] = {}
        if args.mode == "bootstrap":
            finetune = _resolve_finetune(
                client,
                finetune_id=str(args.finetune_id),
                finetune_name=str(args.finetune_name),
                rank=int(args.rank),
            )
            run.summary["bootstrap_finetune_id"] = str(finetune.finetune_id)
            summaries["bootstrap"] = _train_stage(
                finetune=finetune,
                train_records=train_records,
                val_records=val_records,
                test_records=test_records,
                stage=bootstrap_stage,
                args=args,
                run=run,
                run_dir=_stage_run_dir(run_root, label="bootstrap"),
                seed_offset=0,
            )
        elif args.mode == "rl":
            finetune = _resolve_finetune(
                client,
                finetune_id=str(args.finetune_id),
                finetune_name=str(args.finetune_name),
                rank=int(args.rank),
            )
            run.summary["rl_finetune_id"] = str(finetune.finetune_id)
            summaries["rl"] = _train_stage(
                finetune=finetune,
                train_records=train_records,
                val_records=val_records,
                test_records=test_records,
                stage=rl_stage,
                args=args,
                run=run,
                run_dir=_stage_run_dir(run_root, label="rl"),
                seed_offset=1000,
            )
        elif args.mode == "bootstrap_then_rl":
            finetune = _resolve_finetune(
                client,
                finetune_id=str(args.finetune_id),
                finetune_name=str(args.finetune_name),
                rank=int(args.rank),
            )
            run.summary["bootstrap_then_rl_finetune_id"] = str(finetune.finetune_id)
            summaries["bootstrap"] = _train_stage(
                finetune=finetune,
                train_records=train_records,
                val_records=val_records,
                test_records=test_records,
                stage=bootstrap_stage,
                args=args,
                run=run,
                run_dir=_stage_run_dir(run_root, label="bootstrap"),
                seed_offset=0,
            )
            summaries["rl"] = _train_stage(
                finetune=finetune,
                train_records=train_records,
                val_records=val_records,
                test_records=test_records,
                stage=rl_stage,
                args=args,
                run=run,
                run_dir=_stage_run_dir(run_root, label="rl"),
                seed_offset=1000,
            )
        else:
            bootstrap_finetune = client.create_finetune(name=f"{args.finetune_name}-bootstrap", rank=int(args.rank))
            run.summary["bootstrap_finetune_id"] = str(bootstrap_finetune.finetune_id)
            summaries["bootstrap"] = _train_stage(
                finetune=bootstrap_finetune,
                train_records=train_records,
                val_records=val_records,
                test_records=test_records,
                stage=bootstrap_stage,
                args=args,
                run=run,
                run_dir=_stage_run_dir(run_root, label="bootstrap"),
                seed_offset=0,
            )
            summaries["rl"] = _train_stage(
                finetune=bootstrap_finetune,
                train_records=train_records,
                val_records=val_records,
                test_records=test_records,
                stage=rl_stage,
                args=args,
                run=run,
                run_dir=_stage_run_dir(run_root, label="rl"),
                seed_offset=1000,
            )
            run.summary["bootstrap_then_rl_finetune_id"] = str(bootstrap_finetune.finetune_id)
            rl_only_finetune = client.create_finetune(name=f"{args.finetune_name}-rl-only", rank=int(args.rank))
            run.summary["rl_only_finetune_id"] = str(rl_only_finetune.finetune_id)
            summaries["rl_only"] = _train_stage(
                finetune=rl_only_finetune,
                train_records=train_records,
                val_records=val_records,
                test_records=test_records,
                stage=rl_only_stage,
                args=args,
                run=run,
                run_dir=_stage_run_dir(run_root, label="rl_only"),
                seed_offset=2000,
            )
        common.write_json(run_root / "summary.json", summaries)
        print(f"done. summary={run_root / 'summary.json'}")
    finally:
        run.finish()
        close = getattr(client, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    main()
