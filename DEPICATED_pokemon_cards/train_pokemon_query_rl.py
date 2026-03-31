#!/usr/bin/env python3
"""Train a Moondream query finetune for PokemonCards reasoning distillation."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    from tqdm.auto import tqdm  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    class _SimpleTqdm:
        def __init__(self, iterable, *args: Any, **kwargs: Any) -> None:
            self._iterable = iterable

        def __iter__(self):
            return iter(self._iterable)

        def set_postfix(self, *args: Any, **kwargs: Any) -> None:
            return

    def tqdm(iterable=None, *args, **kwargs):  # type: ignore
        if iterable is None:
            iterable = []
        return _SimpleTqdm(iterable, *args, **kwargs)

try:
    import wandb  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    class _WandbRun:
        def __init__(self) -> None:
            self.summary: dict[str, Any] = {}

        def finish(self) -> None:
            return

    class _WandbShim:
        @staticmethod
        def init(*args: Any, **kwargs: Any) -> _WandbRun:
            print("wandb not installed; continuing without remote logging.")
            return _WandbRun()

        @staticmethod
        def log(*args: Any, **kwargs: Any) -> None:
            return

    wandb = _WandbShim()

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from DEPICATED_pokemon_cards import data_loader  # noqa: E402
from DEPICATED_pokemon_cards.common import (  # noqa: E402
    cfg_bool,
    cfg_float,
    cfg_int,
    cfg_list_str,
    cfg_optional_bool,
    cfg_str,
    ensure_parent_dir,
    image_path_to_data_url,
    load_dotenv_if_available,
    load_json_object,
    progress_enabled,
    resolve_path,
    validate_config_keys,
    write_json,
)
from DEPICATED_pokemon_cards.scoring import (  # noqa: E402
    answer_reward_for_task,
    combined_reward,
    parse_prediction_json,
    rationale_reward_from_texts,
)
from DEPICATED_pokemon_cards.task_schema import CANONICAL_TASK_TYPES, normalize_answer_for_task, normalize_task_type  # noqa: E402
from tuna_sdk import QueryRequest, QuerySettings, TunaClient  # noqa: E402
from tuna_sdk.errors import TunaAPIError, TunaNetworkError  # noqa: E402

DEFAULT_BASE_URL = "https://api.moondream.ai/v1"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "stage1_no_reasoning_cicd.json"
DEFAULT_DATASET_SOURCE = data_loader.DEFAULT_DATASET_SOURCE
DEFAULT_DATASET_DIR = "pokemon_cards/outputs/thefusion21_pokemoncards_v1"
DEFAULT_HF_DATASET_REPO_ID = data_loader.DEFAULT_HF_DATASET_REPO_ID
DEFAULT_HF_DATASET_REVISION = data_loader.DEFAULT_HF_DATASET_REVISION
DEFAULT_TASK_SAMPLING_WEIGHTS: dict[str, float] = {
    "card_identity": 0.40,
    "card_core": 0.30,
    "attack_overview": 0.20,
    "card_summary": 0.10,
}
DEFAULT_MAX_TOKENS_BY_TASK: dict[str, int] = {
    "card_identity": 96,
    "card_core": 192,
    "attack_overview": 160,
    "card_summary": 96,
}
DEFAULT_FINAL_EVAL_SPLITS = ["val", "test"]
CHECKPOINT_METRIC_CHOICES = (
    "eval_combined_reward_mean",
    "eval_answer_reward_mean",
    "eval_rationale_reward_mean",
    "eval_json_parse_rate",
)
TRAIN_CONFIG_ALLOWED_KEYS = {
    "api_key",
    "api_key_env_var",
    "base_url",
    "batch_size",
    "best_metric",
    "checkpoint_avg_metric",
    "checkpoint_avg_splits",
    "checkpoint_ranking_output",
    "dataset_dir",
    "dataset_source",
    "env_file",
    "eval_batch_size",
    "eval_every",
    "eval_max_samples",
    "eval_predictions_output_dir",
    "eval_reasoning",
    "eval_temperature",
    "eval_top_p",
    "final_eval_splits",
    "finetune_id",
    "finetune_name",
    "group_size",
    "hf_cache_dir",
    "hf_dataset_repo_id",
    "hf_dataset_revision",
    "hf_token",
    "lr",
    "max_tokens",
    "max_tokens_by_task",
    "max_workers",
    "no_progress",
    "num_steps",
    "off_policy",
    "off_policy_buffer_size",
    "off_policy_min_buffer_groups",
    "off_policy_mix_ratio",
    "off_policy_warmup_steps",
    "rank",
    "reasoning",
    "resume_step",
    "save_eval_predictions",
    "save_every",
    "save_on_eval",
    "seed",
    "task_sampling_weights",
    "temperature",
    "top_p",
    "train_split",
    "val_split",
    "wandb_project",
    "wandb_run_name",
}


@dataclass(frozen=True)
class QAExample:
    row_id: str
    split: str
    task_type: str
    question: str
    image_path: Path
    expected_answer: dict[str, Any]
    teacher_rationale_text: str
    source_metadata: dict[str, Any]


_IMAGE_DATA_URL_CACHE: dict[Path, str] = {}


def _random_suffix(length: int = 6) -> str:
    import string

    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


def _progress_tqdm(iterable, *, desc: str, no_progress: bool):
    return tqdm(iterable, desc=desc, disable=not progress_enabled(no_progress))


def _image_data_url(path: Path) -> str:
    resolved = path.resolve()
    cached = _IMAGE_DATA_URL_CACHE.get(resolved)
    if cached is not None:
        return cached
    data_url = image_path_to_data_url(resolved)
    _IMAGE_DATA_URL_CACHE[resolved] = data_url
    return data_url


def _load_json_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        if config_path == DEFAULT_CONFIG_PATH:
            return {}
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return load_json_object(config_path)


def _normalize_task_sampling_weights(raw_map: dict[str, Any]) -> dict[str, float]:
    out = dict(DEFAULT_TASK_SAMPLING_WEIGHTS)
    for key, value in raw_map.items():
        task_type = normalize_task_type(str(key))
        out[task_type] = max(0.0, float(value))
    return out


def _normalize_max_tokens_by_task(raw_map: dict[str, Any]) -> dict[str, int]:
    out = dict(DEFAULT_MAX_TOKENS_BY_TASK)
    for key, value in raw_map.items():
        task_type = normalize_task_type(str(key))
        out[task_type] = max(1, int(value))
    return out


def _build_parser(config: dict[str, Any], config_path: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PokemonCards query RL finetune.")
    parser.add_argument("--config", default=str(config_path))
    parser.add_argument("--env-file", default=cfg_str(config, "env_file", ".env"))
    parser.add_argument("--api-key", default=cfg_str(config, "api_key", ""))
    parser.add_argument("--api-key-env-var", default=cfg_str(config, "api_key_env_var", "MOONDREAM_API_KEY"))
    parser.add_argument("--base-url", default=cfg_str(config, "base_url", DEFAULT_BASE_URL))

    parser.add_argument(
        "--dataset-source",
        choices=sorted(data_loader.SUPPORTED_DATASET_SOURCES),
        default=cfg_str(config, "dataset_source", DEFAULT_DATASET_SOURCE),
    )
    parser.add_argument("--dataset-dir", default=cfg_str(config, "dataset_dir", DEFAULT_DATASET_DIR))
    parser.add_argument("--hf-dataset-repo-id", default=cfg_str(config, "hf_dataset_repo_id", DEFAULT_HF_DATASET_REPO_ID))
    parser.add_argument("--hf-dataset-revision", default=cfg_str(config, "hf_dataset_revision", DEFAULT_HF_DATASET_REVISION))
    parser.add_argument("--hf-token", default=cfg_str(config, "hf_token", ""))
    parser.add_argument("--hf-cache-dir", default=cfg_str(config, "hf_cache_dir", ""))
    parser.add_argument("--train-split", default=cfg_str(config, "train_split", "train"))
    parser.add_argument("--val-split", default=cfg_str(config, "val_split", "val"))

    parser.add_argument("--finetune-id", default=cfg_str(config, "finetune_id", ""))
    parser.add_argument("--finetune-name", default=cfg_str(config, "finetune_name", f"pokemon-cards-{_random_suffix()}"))
    parser.add_argument("--rank", type=int, default=cfg_int(config, "rank", 32))
    parser.add_argument("--num-steps", type=int, default=cfg_int(config, "num_steps", 500))
    parser.add_argument("--resume-step", type=int, default=cfg_int(config, "resume_step", 0))
    parser.add_argument("--batch-size", type=int, default=cfg_int(config, "batch_size", 4))
    parser.add_argument("--group-size", type=int, default=cfg_int(config, "group_size", 2))
    parser.add_argument("--lr", type=float, default=cfg_float(config, "lr", 1e-5))
    parser.add_argument("--temperature", type=float, default=cfg_float(config, "temperature", 1.0))
    parser.add_argument("--top-p", type=float, default=cfg_float(config, "top_p", 0.9))
    parser.add_argument("--max-tokens", type=int, default=cfg_int(config, "max_tokens", 256))
    parser.add_argument("--reasoning", dest="reasoning", action="store_true")
    parser.add_argument("--no-reasoning", dest="reasoning", action="store_false")
    parser.set_defaults(reasoning=cfg_bool(config, "reasoning", False))
    parser.add_argument("--eval-reasoning", dest="eval_reasoning", action="store_true")
    parser.add_argument("--no-eval-reasoning", dest="eval_reasoning", action="store_false")
    parser.add_argument("--eval-reasoning-inherit", dest="eval_reasoning", action="store_const", const=None)
    parser.set_defaults(eval_reasoning=cfg_optional_bool(config, "eval_reasoning"))

    parser.add_argument("--off-policy", dest="off_policy", action="store_true")
    parser.add_argument("--no-off-policy", dest="off_policy", action="store_false")
    parser.set_defaults(off_policy=cfg_bool(config, "off_policy", False))
    parser.add_argument("--off-policy-mix-ratio", type=float, default=cfg_float(config, "off_policy_mix_ratio", 0.25))
    parser.add_argument("--off-policy-buffer-size", type=int, default=cfg_int(config, "off_policy_buffer_size", 4096))
    parser.add_argument("--off-policy-warmup-steps", type=int, default=cfg_int(config, "off_policy_warmup_steps", 30))
    parser.add_argument(
        "--off-policy-min-buffer-groups",
        type=int,
        default=cfg_int(config, "off_policy_min_buffer_groups", 128),
    )
    parser.add_argument("--max-workers", type=int, default=cfg_int(config, "max_workers", 1))
    parser.add_argument("--seed", type=int, default=cfg_int(config, "seed", 42))

    parser.add_argument("--eval-every", type=int, default=cfg_int(config, "eval_every", 10))
    parser.add_argument("--eval-max-samples", type=int, default=cfg_int(config, "eval_max_samples", 1000))
    parser.add_argument("--eval-batch-size", type=int, default=cfg_int(config, "eval_batch_size", 8))
    parser.add_argument("--eval-temperature", type=float, default=cfg_float(config, "eval_temperature", 0.0))
    parser.add_argument("--eval-top-p", type=float, default=cfg_float(config, "eval_top_p", 1.0))
    parser.add_argument("--save-every", type=int, default=cfg_int(config, "save_every", 10))
    parser.add_argument("--save-on-eval", action="store_true", default=cfg_bool(config, "save_on_eval", True))
    parser.add_argument("--save-eval-predictions", action="store_true", default=cfg_bool(config, "save_eval_predictions", True))
    parser.add_argument("--eval-predictions-output-dir", default=cfg_str(config, "eval_predictions_output_dir", "pokemon_cards/outputs/eval_predictions"))

    parser.add_argument("--checkpoint-ranking-output", default=cfg_str(config, "checkpoint_ranking_output", "pokemon_cards/outputs/checkpoint_ranking.json"))
    parser.add_argument(
        "--checkpoint-avg-metric",
        choices=list(CHECKPOINT_METRIC_CHOICES),
        default=cfg_str(config, "checkpoint_avg_metric", "eval_combined_reward_mean"),
    )
    parser.add_argument(
        "--best-metric",
        choices=list(CHECKPOINT_METRIC_CHOICES),
        default=cfg_str(config, "best_metric", "eval_combined_reward_mean"),
    )
    parser.add_argument("--wandb-project", default=cfg_str(config, "wandb_project", "moondream-pokemon-cards-query-rl"))
    parser.add_argument("--wandb-run-name", default=cfg_str(config, "wandb_run_name", ""))
    parser.add_argument("--no-progress", action="store_true", default=cfg_bool(config, "no_progress", False))
    return parser


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    pre_args, _ = pre_parser.parse_known_args(argv)
    config_path = resolve_path(pre_args.config, search_roots=(REPO_ROOT, Path(__file__).resolve().parent))
    config = _load_json_config(config_path)
    validate_config_keys(config, allowed_keys=TRAIN_CONFIG_ALLOWED_KEYS, config_path=config_path)

    parser = _build_parser(config, config_path)
    args = parser.parse_args(argv)
    args.config = str(config_path)
    args.env_file = resolve_path(args.env_file, search_roots=(REPO_ROOT, Path(__file__).resolve().parent))
    args.dataset_dir = resolve_path(args.dataset_dir, search_roots=(REPO_ROOT,))
    args.eval_predictions_output_dir = resolve_path(args.eval_predictions_output_dir, search_roots=(REPO_ROOT,))
    args.checkpoint_ranking_output = resolve_path(args.checkpoint_ranking_output, search_roots=(REPO_ROOT,))
    args.dataset_source = data_loader.normalize_dataset_source(args.dataset_source)

    args.final_eval_splits = cfg_list_str(config, "final_eval_splits", DEFAULT_FINAL_EVAL_SPLITS)
    args.checkpoint_avg_splits = cfg_list_str(config, "checkpoint_avg_splits", ["val", "test"])
    args.task_sampling_weights = _normalize_task_sampling_weights(
        config.get("task_sampling_weights", DEFAULT_TASK_SAMPLING_WEIGHTS)
        if isinstance(config.get("task_sampling_weights"), dict)
        else DEFAULT_TASK_SAMPLING_WEIGHTS
    )
    args.max_tokens_by_task = _normalize_max_tokens_by_task(
        config.get("max_tokens_by_task", DEFAULT_MAX_TOKENS_BY_TASK)
        if isinstance(config.get("max_tokens_by_task"), dict)
        else DEFAULT_MAX_TOKENS_BY_TASK
    )
    return args


def _resolve_api_key(args: argparse.Namespace) -> str:
    if str(args.api_key or "").strip():
        return str(args.api_key).strip()
    env_name = str(args.api_key_env_var or "MOONDREAM_API_KEY").strip() or "MOONDREAM_API_KEY"
    value = os.environ.get(env_name, "").strip()
    if value:
        return value
    value = os.environ.get("MOONDREAM_API_KEY", "").strip()
    if value:
        return value
    raise ValueError("Moondream API key is required")


def _validate_args(args: argparse.Namespace) -> None:
    if args.rank == 32 and float(args.lr) > 1e-5:
        raise ValueError("rank 32 configs require lr <= 1e-5")
    if bool(args.reasoning) and bool(args.off_policy):
        raise ValueError("reasoning configs reject off_policy=true")
    if not args.checkpoint_avg_splits:
        raise ValueError("checkpoint_avg_splits must be non-empty")
    if not any(float(args.task_sampling_weights.get(task, 0.0)) > 0.0 for task in CANONICAL_TASK_TYPES):
        raise ValueError("task_sampling_weights must enable at least one task")
    if args.batch_size <= 0 or args.group_size <= 0:
        raise ValueError("batch_size and group_size must be > 0")
    if args.num_steps <= 0:
        raise ValueError("num_steps must be > 0")


def _parse_row(row: dict[str, Any], *, dataset_dir: Path, require_teacher_rationale: bool) -> Optional[QAExample]:
    task_type = normalize_task_type(str(row.get("task_type", "")))
    expected_payload = json.loads(str(row.get("final_answer_json", "{}")))
    expected_answer = normalize_answer_for_task(task_type, expected_payload)
    if expected_answer is None:
        return None
    teacher_rationale_text = str(row.get("teacher_rationale_text", "") or "").strip()
    if require_teacher_rationale and not teacher_rationale_text:
        return None
    image_path = (dataset_dir / str(row.get("image_path", ""))).resolve()
    if not image_path.is_file():
        return None
    source_meta = {}
    source_meta_raw = str(row.get("source_metadata_json", "") or "")
    if source_meta_raw:
        try:
            source_meta = json.loads(source_meta_raw)
        except json.JSONDecodeError:
            source_meta = {}
    return QAExample(
        row_id=str(row.get("row_id", "")),
        split=str(row.get("split", "")),
        task_type=task_type,
        question=str(row.get("question", "")),
        image_path=image_path,
        expected_answer=expected_answer,
        teacher_rationale_text=teacher_rationale_text,
        source_metadata=source_meta if isinstance(source_meta, dict) else {},
    )


def _load_examples_for_split(
    *,
    args: argparse.Namespace,
    split_name: str,
    require_teacher_rationale: bool,
) -> list[QAExample]:
    rows = data_loader.load_split_rows(
        dataset_source=args.dataset_source,
        split_name=split_name,
        dataset_dir=Path(args.dataset_dir),
        hf_dataset_repo_id=str(args.hf_dataset_repo_id),
        hf_dataset_revision=str(args.hf_dataset_revision),
        hf_token=str(args.hf_token),
        hf_cache_dir=str(args.hf_cache_dir),
    )
    out: list[QAExample] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        parsed = _parse_row(row, dataset_dir=Path(args.dataset_dir), require_teacher_rationale=require_teacher_rationale)
        if parsed is not None:
            out.append(parsed)
    return out


def _group_examples_by_task(examples: list[QAExample]) -> dict[str, list[QAExample]]:
    grouped: dict[str, list[QAExample]] = defaultdict(list)
    for example in examples:
        grouped[example.task_type].append(example)
    return dict(grouped)


def _sample_train_batch(
    *,
    examples_by_task: dict[str, list[QAExample]],
    task_sampling_weights: dict[str, float],
    batch_size: int,
    rng: random.Random,
) -> list[QAExample]:
    available_tasks = [task for task, examples in examples_by_task.items() if examples and float(task_sampling_weights.get(task, 0.0)) > 0.0]
    if not available_tasks:
        raise ValueError("no train tasks available after applying task_sampling_weights")
    weights = [float(task_sampling_weights.get(task, 0.0)) for task in available_tasks]
    batch: list[QAExample] = []
    for _ in range(batch_size):
        task_type = rng.choices(available_tasks, weights=weights, k=1)[0]
        batch.append(rng.choice(examples_by_task[task_type]))
    return batch


def _maybe_limit_examples(examples: list[QAExample], *, max_samples: int, seed: int) -> list[QAExample]:
    if max_samples <= 0 or len(examples) <= max_samples:
        return list(examples)
    rng = random.Random(seed)
    return rng.sample(list(examples), k=max_samples)


def _request_for_example(
    example: QAExample,
    *,
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_tokens_by_task: dict[str, int],
    reasoning: bool,
) -> QueryRequest:
    effective_max_tokens = int(max_tokens_by_task.get(example.task_type, max_tokens))
    return QueryRequest(
        question=example.question,
        image_url=_image_data_url(example.image_path),
        reasoning=bool(reasoning),
        settings=QuerySettings(
            temperature=float(temperature),
            top_p=float(top_p),
            max_tokens=effective_max_tokens,
        ),
    )


def _score_rollout_for_example(rollout: Any, example: QAExample, *, use_reasoning_reward: bool) -> dict[str, Any]:
    answer_text = str(getattr(rollout.output, "answer", "") or "")
    pred_payload = parse_prediction_json(answer_text)
    answer_reward = (
        answer_reward_for_task(
            example.task_type,
            pred_payload,
            example.expected_answer,
            expected_metadata=example.source_metadata,
        )
        if pred_payload is not None
        else 0.0
    )
    reasoning_obj = getattr(rollout.output, "reasoning", None)
    rationale_text = str(getattr(reasoning_obj, "text", "") or "")
    rationale_reward = (
        rationale_reward_from_texts(example.task_type, rationale_text, example.teacher_rationale_text)
        if use_reasoning_reward and example.teacher_rationale_text
        else 0.0
    )
    total_reward = combined_reward(answer_reward, rationale_reward, use_reasoning_reward=use_reasoning_reward)
    return {
        "reward": float(total_reward),
        "answer_reward": float(answer_reward),
        "rationale_reward": float(rationale_reward),
        "json_parse_success": pred_payload is not None,
        "prediction_payload": pred_payload,
        "reasoning_text": rationale_text,
    }


def _compose_train_groups(
    *,
    on_policy_groups: list[Any],
    replay_groups: list[Any],
    off_policy: bool,
    off_policy_mix_ratio: float,
    off_policy_warmup_steps: int,
    off_policy_min_buffer_groups: int,
    global_step: int,
    rng: random.Random,
) -> tuple[list[Any], int]:
    if not on_policy_groups:
        return [], 0
    if not off_policy or off_policy_mix_ratio <= 0.0:
        return list(on_policy_groups), 0
    if global_step < off_policy_warmup_steps:
        return list(on_policy_groups), 0
    if len(replay_groups) < off_policy_min_buffer_groups:
        return list(on_policy_groups), 0

    desired_off_policy = int(round(len(on_policy_groups) * off_policy_mix_ratio))
    desired_off_policy = max(1, min(desired_off_policy, len(on_policy_groups), len(replay_groups)))
    keep_on_policy = len(on_policy_groups) - desired_off_policy
    selected_on_policy = [] if keep_on_policy <= 0 else rng.sample(list(on_policy_groups), k=keep_on_policy)
    selected_off_policy = rng.sample(list(replay_groups), k=desired_off_policy)
    mixed = selected_on_policy + selected_off_policy
    rng.shuffle(mixed)
    return mixed, desired_off_policy


def _empty_eval_metrics(split_name: str) -> dict[str, float]:
    return {
        f"{split_name}_eval_samples": 0.0,
        f"{split_name}_eval_answer_reward_mean": 0.0,
        f"{split_name}_eval_rationale_reward_mean": 0.0,
        f"{split_name}_eval_combined_reward_mean": 0.0,
        f"{split_name}_eval_json_parse_rate": 0.0,
    }


def _evaluate_split(
    *,
    finetune: Any,
    split_name: str,
    examples: list[QAExample],
    batch_size: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_tokens_by_task: dict[str, int],
    reasoning: bool,
    use_reasoning_reward: bool,
    no_progress: bool,
    save_predictions: bool,
    predictions_output_dir: Path,
    checkpoint_label: str,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    if not examples:
        return _empty_eval_metrics(split_name), []

    answer_rewards: list[float] = []
    rationale_rewards: list[float] = []
    combined_rewards: list[float] = []
    parse_success = 0
    predictions: list[dict[str, Any]] = []
    task_reward_values: dict[str, list[float]] = defaultdict(list)

    iterator = range(0, len(examples), batch_size)
    for start in _progress_tqdm(iterator, desc=f"eval:{split_name}", no_progress=no_progress):
        batch = examples[start : start + batch_size]
        requests = [
            _request_for_example(
                example,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                max_tokens_by_task=max_tokens_by_task,
                reasoning=reasoning,
            )
            for example in batch
        ]
        results = finetune.rollouts_batch(
            requests=requests,
            num_rollouts=1,
            max_workers=max(1, min(int(batch_size), int(max(1, batch_size)))),
        )
        for example, result in zip(batch, results):
            outcome = _score_rollout_for_example(result.rollouts[0], example, use_reasoning_reward=use_reasoning_reward)
            answer_rewards.append(float(outcome["answer_reward"]))
            rationale_rewards.append(float(outcome["rationale_reward"]))
            combined_rewards.append(float(outcome["reward"]))
            parse_success += int(bool(outcome["json_parse_success"]))
            task_reward_values[example.task_type].append(float(outcome["answer_reward"]))
            predictions.append(
                {
                    "row_id": example.row_id,
                    "split": split_name,
                    "task_type": example.task_type,
                    "question": example.question,
                    "expected_answer": example.expected_answer,
                    "teacher_rationale_text": example.teacher_rationale_text,
                    "predicted_answer_payload": outcome["prediction_payload"],
                    "predicted_answer_text": str(getattr(result.rollouts[0].output, "answer", "") or ""),
                    "predicted_reasoning_text": str(outcome["reasoning_text"]),
                    "answer_reward": float(outcome["answer_reward"]),
                    "rationale_reward": float(outcome["rationale_reward"]),
                    "combined_reward": float(outcome["reward"]),
                    "json_parse_success": bool(outcome["json_parse_success"]),
                }
            )

    metrics: dict[str, float] = {
        f"{split_name}_eval_samples": float(len(examples)),
        f"{split_name}_eval_answer_reward_mean": float(np.mean(answer_rewards)) if answer_rewards else 0.0,
        f"{split_name}_eval_rationale_reward_mean": float(np.mean(rationale_rewards)) if rationale_rewards else 0.0,
        f"{split_name}_eval_combined_reward_mean": float(np.mean(combined_rewards)) if combined_rewards else 0.0,
        f"{split_name}_eval_json_parse_rate": float(parse_success) / float(max(1, len(examples))),
    }
    for task_type in CANONICAL_TASK_TYPES:
        values = task_reward_values.get(task_type, [])
        metrics[f"{split_name}_eval_answer_reward_{task_type}"] = float(np.mean(values)) if values else 0.0

    if save_predictions:
        out_path = predictions_output_dir / f"{checkpoint_label}_{split_name}.jsonl"
        ensure_parent_dir(out_path)
        with out_path.open("w", encoding="utf-8") as handle:
            for item in predictions:
                handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    return metrics, predictions


def _metric_value(metrics: dict[str, float], *, split_name: str, metric_name: str) -> float:
    key = f"{split_name}_{metric_name}"
    return float(metrics.get(key, 0.0))


def _ranking_entry(
    *,
    step: int,
    by_split: dict[str, dict[str, float]],
    checkpoint_avg_metric: str,
    checkpoint_avg_splits: list[str],
) -> dict[str, Any]:
    avg_value = float(
        np.mean(
            [
                _metric_value(by_split.get(split_name, {}), split_name=split_name, metric_name=checkpoint_avg_metric)
                for split_name in checkpoint_avg_splits
            ]
        )
    )
    return {
        "step": int(step),
        "checkpoint_avg_metric_name": checkpoint_avg_metric,
        "checkpoint_avg_metric_value": avg_value,
        "by_split": by_split,
    }


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)
    load_dotenv_if_available(args.env_file)
    args.api_key = _resolve_api_key(args)
    _validate_args(args)

    rng = random.Random(int(args.seed))
    np.random.seed(int(args.seed))

    require_teacher_rationale = bool(args.reasoning)
    train_examples = _load_examples_for_split(
        args=args,
        split_name=str(args.train_split),
        require_teacher_rationale=require_teacher_rationale,
    )
    if not train_examples:
        raise ValueError("no training examples available")
    train_examples_by_task = _group_examples_by_task(train_examples)

    eval_reasoning = bool(args.reasoning) if args.eval_reasoning is None else bool(args.eval_reasoning)
    eval_splits = list(dict.fromkeys([str(args.val_split)] + list(args.final_eval_splits) + list(args.checkpoint_avg_splits)))
    eval_examples_by_split: dict[str, list[QAExample]] = {}
    for split_name in eval_splits:
        examples = _load_examples_for_split(
            args=args,
            split_name=split_name,
            require_teacher_rationale=require_teacher_rationale and bool(eval_reasoning),
        )
        eval_examples_by_split[split_name] = _maybe_limit_examples(examples, max_samples=int(args.eval_max_samples), seed=int(args.seed) + len(split_name))

    client = TunaClient(api_key=args.api_key, base_url=str(args.base_url))
    finetune = client.get_finetune(str(args.finetune_id)) if str(args.finetune_id).strip() else client.create_finetune(
        name=str(args.finetune_name),
        rank=int(args.rank),
    )

    run = wandb.init(
        project=str(args.wandb_project),
        name=(str(args.wandb_run_name).strip() or None),
        config={
            "dataset_source": args.dataset_source,
            "dataset_dir": str(args.dataset_dir),
            "train_split": str(args.train_split),
            "val_split": str(args.val_split),
            "final_eval_splits": list(args.final_eval_splits),
            "finetune_id": finetune.finetune_id,
            "finetune_name": finetune.name,
            "rank": int(args.rank),
            "num_steps": int(args.num_steps),
            "batch_size": int(args.batch_size),
            "group_size": int(args.group_size),
            "lr": float(args.lr),
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "max_tokens": int(args.max_tokens),
            "max_tokens_by_task": dict(args.max_tokens_by_task),
            "reasoning": bool(args.reasoning),
            "eval_reasoning": bool(eval_reasoning),
            "off_policy": bool(args.off_policy),
            "off_policy_mix_ratio": float(args.off_policy_mix_ratio),
            "off_policy_buffer_size": int(args.off_policy_buffer_size),
            "off_policy_warmup_steps": int(args.off_policy_warmup_steps),
            "off_policy_min_buffer_groups": int(args.off_policy_min_buffer_groups),
            "checkpoint_avg_splits": list(args.checkpoint_avg_splits),
            "checkpoint_avg_metric": str(args.checkpoint_avg_metric),
            "best_metric": str(args.best_metric),
            "task_sampling_weights": dict(args.task_sampling_weights),
        },
    )
    run.summary["finetune_id"] = finetune.finetune_id

    replay_buffer: deque[Any] = deque(maxlen=int(args.off_policy_buffer_size) if int(args.off_policy_buffer_size) > 0 else None)
    ranking_entries: list[dict[str, Any]] = []
    best_metric_value: Optional[float] = None
    best_step: Optional[int] = None
    global_step = int(args.resume_step)

    for step_idx in range(int(args.num_steps)):
        global_step += 1
        batch = _sample_train_batch(
            examples_by_task=train_examples_by_task,
            task_sampling_weights=dict(args.task_sampling_weights),
            batch_size=int(args.batch_size),
            rng=rng,
        )
        requests = [
            _request_for_example(
                example,
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                max_tokens=int(args.max_tokens),
                max_tokens_by_task=dict(args.max_tokens_by_task),
                reasoning=bool(args.reasoning),
            )
            for example in batch
        ]
        results = finetune.rollouts_batch(
            requests=requests,
            num_rollouts=int(args.group_size),
            max_workers=int(args.max_workers),
        )

        on_policy_groups = []
        train_answer_rewards: list[float] = []
        train_rationale_rewards: list[float] = []
        train_total_rewards: list[float] = []
        train_parse_success = 0
        task_counter = Counter(example.task_type for example in batch)
        for example, result in zip(batch, results):
            rollout_rewards: list[float] = []
            for rollout in result.rollouts:
                outcome = _score_rollout_for_example(rollout, example, use_reasoning_reward=bool(args.reasoning))
                rollout_rewards.append(float(outcome["reward"]))
                train_answer_rewards.append(float(outcome["answer_reward"]))
                train_rationale_rewards.append(float(outcome["rationale_reward"]))
                train_total_rewards.append(float(outcome["reward"]))
                train_parse_success += int(bool(outcome["json_parse_success"]))
            on_policy_groups.append(result.to_group(rewards=rollout_rewards))

        train_groups, off_policy_groups = _compose_train_groups(
            on_policy_groups=on_policy_groups,
            replay_groups=list(replay_buffer),
            off_policy=bool(args.off_policy),
            off_policy_mix_ratio=float(args.off_policy_mix_ratio),
            off_policy_warmup_steps=int(args.off_policy_warmup_steps),
            off_policy_min_buffer_groups=int(args.off_policy_min_buffer_groups),
            global_step=global_step,
            rng=rng,
        )
        train_out = finetune.train_step(groups=train_groups, lr=float(args.lr))
        if bool(args.off_policy):
            replay_buffer.extend(on_policy_groups)

        metrics: dict[str, float] = {
            "global_step": float(global_step),
            "reward_mean": float(np.mean(train_total_rewards)) if train_total_rewards else 0.0,
            "answer_reward_mean": float(np.mean(train_answer_rewards)) if train_answer_rewards else 0.0,
            "rationale_reward_mean": float(np.mean(train_rationale_rewards)) if train_rationale_rewards else 0.0,
            "train_json_parse_rate": float(train_parse_success) / float(max(1, len(train_total_rewards))),
            "accepted_groups": float(len(train_groups)),
            "on_policy_groups": float(len(on_policy_groups)),
            "off_policy_groups": float(off_policy_groups),
            "off_policy_group_fraction": float(off_policy_groups) / float(max(1, len(train_groups))),
            "replay_buffer_size": float(len(replay_buffer)),
            "kl": float(train_out.kl or 0.0),
            "router_kl": float(train_out.router_kl or 0.0),
            "grad_norm": float(train_out.grad_norm or 0.0),
        }
        for task_type, count in sorted(task_counter.items()):
            metrics[f"train_task_count_{task_type}"] = float(count)

        should_eval = int(args.eval_every) > 0 and (global_step % int(args.eval_every) == 0)
        if should_eval:
            split_metrics: dict[str, dict[str, float]] = {}
            checkpoint_label = f"step{global_step:06d}"
            for split_name in args.checkpoint_avg_splits:
                split_eval_metrics, _predictions = _evaluate_split(
                    finetune=finetune,
                    split_name=split_name,
                    examples=list(eval_examples_by_split.get(split_name, [])),
                    batch_size=int(args.eval_batch_size),
                    temperature=float(args.eval_temperature),
                    top_p=float(args.eval_top_p),
                    max_tokens=int(args.max_tokens),
                    max_tokens_by_task=dict(args.max_tokens_by_task),
                    reasoning=bool(eval_reasoning),
                    use_reasoning_reward=bool(eval_reasoning),
                    no_progress=bool(args.no_progress),
                    save_predictions=bool(args.save_eval_predictions),
                    predictions_output_dir=Path(args.eval_predictions_output_dir),
                    checkpoint_label=checkpoint_label,
                )
                split_metrics[split_name] = split_eval_metrics
                metrics.update(split_eval_metrics)

            ranking_entry = _ranking_entry(
                step=global_step,
                by_split=split_metrics,
                checkpoint_avg_metric=str(args.checkpoint_avg_metric),
                checkpoint_avg_splits=list(args.checkpoint_avg_splits),
            )
            ranking_entries.append(ranking_entry)
            current_metric_value = float(ranking_entry["checkpoint_avg_metric_value"])
            if bool(args.save_on_eval):
                finetune.save_checkpoint()
            if best_metric_value is None or current_metric_value > best_metric_value:
                best_metric_value = current_metric_value
                best_step = global_step

        wandb.log(metrics, step=global_step)
        print(
            f"step {global_step}/{int(args.resume_step) + int(args.num_steps)} "
            f"reward={metrics['reward_mean']:.4f} kl={metrics['kl']:.4f}"
        )

        if int(args.save_every) > 0 and (global_step % int(args.save_every) == 0):
            finetune.save_checkpoint()

    finetune.save_checkpoint()
    ranking_payload = {
        "finetune_id": finetune.finetune_id,
        "best_step": int(best_step) if best_step is not None else None,
        "best_metric_name": str(args.checkpoint_avg_metric),
        "best_metric_value": float(best_metric_value or 0.0),
        "entries": ranking_entries,
    }
    write_json(Path(args.checkpoint_ranking_output), ranking_payload)
    run.summary["best_step"] = int(best_step) if best_step is not None else -1
    run.summary["best_metric_value"] = float(best_metric_value or 0.0)
    run.finish()
    client.close()


if __name__ == "__main__":
    main()
