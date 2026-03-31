#!/usr/bin/env python3
"""Train a query-model finetune for ConstructionSite dense captioning."""

from __future__ import annotations

import argparse
import json
import os
import random
import string
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from construction_site.common import (  # noqa: E402
    CAPTION_TASK_TYPE,
    config_to_cli_args,
    load_json_config,
    parse_prediction_json,
    repo_relative,
    resolve_config_path,
    token_f1,
)
from construction_site import query_common  # noqa: E402
from tuna_sdk import TunaClient  # noqa: E402
from tuna_sdk.errors import TunaAPIError, TunaNetworkError  # noqa: E402

DEFAULT_BASE_URL = "https://api.moondream.ai/v1"
DEFAULT_CONFIG_PATH = repo_relative("configs", "train_construction_site_query_caption_default.json")
DEFAULT_FINAL_EVAL_SPLITS = ["validation", "test"]
BEST_METRIC_CHOICES = (
    "eval_reward_mean",
    "eval_json_parse_rate",
    "eval_caption_token_f1",
    "eval_attribute_hit_rate",
)


@dataclass(frozen=True)
class CaptionExample:
    row_id: str
    split: str
    task_type: str
    question: str
    image_path: Path
    reference_caption: str
    attribute_tags: tuple[str, ...]
    object_tags: tuple[str, ...]


@dataclass(frozen=True)
class CaptionScoreOutcome(query_common.ScoreOutcome):
    caption_token_f1: float
    attribute_hit_rate: float
    object_hit_rate: float
    length_score: float


def _random_suffix(length: int = 6) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return "".join(random.choices(alphabet, k=length))


def _parse_string_list(raw_value: Any) -> tuple[str, ...]:
    if isinstance(raw_value, str):
        text = raw_value.strip()
        if not text:
            return tuple()
        raw_value = json.loads(text)
    if not isinstance(raw_value, list):
        return tuple()
    out = []
    for item in raw_value:
        text = " ".join(str(item or "").strip().lower().split())
        if text:
            out.append(text)
    return tuple(sorted(set(out)))


def _parse_caption_payload(payload: Any) -> Optional[str]:
    if not isinstance(payload, dict):
        return None
    caption = str(payload.get("caption") or "").strip()
    return caption or None


def _mentions_tag(text: str, tag: str) -> bool:
    lower_text = str(text or "").lower()
    parts = [piece for piece in str(tag or "").lower().split() if piece]
    return bool(parts) and all(part in lower_text for part in parts)


def _tag_hit_rate(prediction: str, tags: tuple[str, ...]) -> float:
    if not tags:
        return 1.0
    hits = sum(1 for tag in tags if _mentions_tag(prediction, tag))
    return float(hits) / float(max(1, len(tags)))


def _length_score(reference_caption: str, predicted_caption: str) -> float:
    ref_len = max(1, len(reference_caption.split()))
    pred_len = len(predicted_caption.split())
    ratio = float(pred_len) / float(max(1, ref_len))
    if 0.6 <= ratio <= 1.6:
        return 1.0
    if ratio <= 0.0:
        return 0.0
    if ratio < 0.6:
        return max(0.0, ratio / 0.6)
    return max(0.0, 1.0 - min(1.0, (ratio - 1.6) / 1.6))


def _build_example(row: dict[str, Any], *, dataset_dir: Path, split_name: str, line_number: int) -> Optional[CaptionExample]:
    if str(row.get("task_type") or "").strip() != CAPTION_TASK_TYPE:
        return None
    image_path = query_common.existing_path(str(row.get("image_path") or ""), dataset_dir=dataset_dir)
    if image_path is None:
        raise FileNotFoundError(f"split={split_name} line={line_number} image_path not found: {row.get('image_path')!r}")
    return CaptionExample(
        row_id=str(row.get("row_id") or f"{split_name}_{line_number:06d}"),
        split=str(row.get("split") or split_name),
        task_type=CAPTION_TASK_TYPE,
        question=str(row.get("question") or ""),
        image_path=image_path,
        reference_caption=str(row.get("reference_caption") or ""),
        attribute_tags=_parse_string_list(row.get("attribute_tags_json")),
        object_tags=_parse_string_list(row.get("object_tags_json")),
    )


def _load_split_examples(*, split_name: str, dataset_dir: Path) -> list[CaptionExample]:
    rows = query_common.load_local_jsonl_rows(dataset_dir=dataset_dir, split_name=split_name)
    examples: list[CaptionExample] = []
    skipped = 0
    for line_number, row in enumerate(rows, start=1):
        example = _build_example(row, dataset_dir=dataset_dir, split_name=split_name, line_number=line_number)
        if example is None:
            skipped += 1
            continue
        examples.append(example)
    print(f"usable split={split_name} examples={len(examples)} skipped={skipped}")
    if not examples:
        raise ValueError(f"split={split_name} contains no usable caption rows")
    return examples


def _score_payload_for_example(example: CaptionExample, payload: Any) -> CaptionScoreOutcome:
    predicted_caption = _parse_caption_payload(payload)
    json_object_parsed = isinstance(payload, dict)
    parse_success = predicted_caption is not None
    if predicted_caption is None:
        return CaptionScoreOutcome(
            reward=0.0,
            parse_success=False,
            task_correct=False,
            json_object_parsed=json_object_parsed,
            caption_token_f1=0.0,
            attribute_hit_rate=0.0,
            object_hit_rate=0.0,
            length_score=0.0,
        )
    caption_f1 = token_f1(example.reference_caption, predicted_caption)
    attribute_hit_rate = _tag_hit_rate(predicted_caption, example.attribute_tags)
    object_hit_rate = _tag_hit_rate(predicted_caption, example.object_tags)
    length_score = _length_score(example.reference_caption, predicted_caption)
    reward = (
        0.20
        + (0.55 * caption_f1)
        + (0.15 * attribute_hit_rate)
        + (0.05 * object_hit_rate)
        + (0.05 * length_score)
    )
    reward = max(0.0, min(1.0, reward))
    task_correct = caption_f1 >= 0.80
    return CaptionScoreOutcome(
        reward=reward,
        parse_success=True,
        task_correct=task_correct,
        json_object_parsed=json_object_parsed,
        caption_token_f1=caption_f1,
        attribute_hit_rate=attribute_hit_rate,
        object_hit_rate=object_hit_rate,
        length_score=length_score,
    )


def _score_rollout(rollout: Any, example: CaptionExample) -> CaptionScoreOutcome:
    answer = ""
    if rollout is not None and getattr(rollout, "output", None) is not None:
        answer = str(getattr(rollout.output, "answer", "") or "")
    payload = parse_prediction_json(answer)
    return _score_payload_for_example(example, payload)


def _evaluate_split(
    *,
    finetune: Any,
    examples: list[CaptionExample],
    split_name: str,
    seed: int,
    batch_size: int,
    max_workers: int,
    max_samples: Optional[int],
    rollout_retries: int,
    rollout_retry_backoff_s: float,
    temperature: float,
    top_p: float,
    max_tokens: int,
    reasoning: bool,
    show_progress: bool,
    predictions_path: Optional[Path] = None,
) -> dict[str, float]:
    indices = list(range(len(examples)))
    random.Random(seed).shuffle(indices)
    if max_samples is not None:
        indices = indices[: max(0, min(int(max_samples), len(indices)))]

    reward_values: list[float] = []
    object_parse_count = 0
    parse_success_count = 0
    task_correct_count = 0
    caption_f1_values: list[float] = []
    attribute_values: list[float] = []
    object_values: list[float] = []
    length_values: list[float] = []

    predictions_handle = None
    if predictions_path is not None:
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        predictions_handle = predictions_path.open("w", encoding="utf-8")

    try:
        batch: list[CaptionExample] = []
        for index in query_common.tqdm(
            indices,
            desc=f"eval:{split_name}",
            total=len(indices),
            dynamic_ncols=True,
            leave=False,
            disable=not show_progress,
        ):
            batch.append(examples[index])
            if len(batch) < batch_size:
                continue
            requests, active_examples = query_common.prepare_requests(
                batch,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                reasoning=reasoning,
            )
            if requests:
                try:
                    results = query_common.rollouts_batch_with_retry(
                        finetune=finetune,
                        requests=requests,
                        num_rollouts=1,
                        max_workers=min(max_workers, len(requests)),
                        retries=rollout_retries,
                        backoff_s=rollout_retry_backoff_s,
                        context=f"eval split={split_name}",
                    )
                except (TunaAPIError, TunaNetworkError) as exc:
                    print(f"eval split={split_name}: skipping batch after error. details={query_common.error_message(exc)}")
                    batch = []
                    continue
                for example, result in zip(active_examples, results):
                    rollout = result.rollouts[0] if getattr(result, "rollouts", None) else None
                    outcome = _score_rollout(rollout, example)
                    reward_values.append(float(outcome.reward))
                    caption_f1_values.append(float(outcome.caption_token_f1))
                    attribute_values.append(float(outcome.attribute_hit_rate))
                    object_values.append(float(outcome.object_hit_rate))
                    length_values.append(float(outcome.length_score))
                    if outcome.json_object_parsed:
                        object_parse_count += 1
                    if outcome.parse_success:
                        parse_success_count += 1
                    if outcome.task_correct:
                        task_correct_count += 1
                    if predictions_handle is not None:
                        answer = ""
                        if rollout is not None and getattr(rollout, "output", None) is not None:
                            answer = str(getattr(rollout.output, "answer", "") or "")
                        predictions_handle.write(
                            json.dumps(
                                {
                                    "row_id": example.row_id,
                                    "split": example.split,
                                    "task_type": example.task_type,
                                    "answer": answer,
                                    "reward": outcome.reward,
                                    "parse_success": outcome.parse_success,
                                    "caption_token_f1": outcome.caption_token_f1,
                                    "attribute_hit_rate": outcome.attribute_hit_rate,
                                    "object_hit_rate": outcome.object_hit_rate,
                                    "length_score": outcome.length_score,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
            batch = []
        if batch:
            requests, active_examples = query_common.prepare_requests(
                batch,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                reasoning=reasoning,
            )
            if requests:
                try:
                    results = query_common.rollouts_batch_with_retry(
                        finetune=finetune,
                        requests=requests,
                        num_rollouts=1,
                        max_workers=min(max_workers, len(requests)),
                        retries=rollout_retries,
                        backoff_s=rollout_retry_backoff_s,
                        context=f"eval split={split_name}",
                    )
                except (TunaAPIError, TunaNetworkError) as exc:
                    print(f"eval split={split_name}: skipping tail batch after error. details={query_common.error_message(exc)}")
                    results = []
                for example, result in zip(active_examples, results):
                    rollout = result.rollouts[0] if getattr(result, "rollouts", None) else None
                    outcome = _score_rollout(rollout, example)
                    reward_values.append(float(outcome.reward))
                    caption_f1_values.append(float(outcome.caption_token_f1))
                    attribute_values.append(float(outcome.attribute_hit_rate))
                    object_values.append(float(outcome.object_hit_rate))
                    length_values.append(float(outcome.length_score))
                    if outcome.json_object_parsed:
                        object_parse_count += 1
                    if outcome.parse_success:
                        parse_success_count += 1
                    if outcome.task_correct:
                        task_correct_count += 1
                    if predictions_handle is not None:
                        answer = ""
                        if rollout is not None and getattr(rollout, "output", None) is not None:
                            answer = str(getattr(rollout.output, "answer", "") or "")
                        predictions_handle.write(
                            json.dumps(
                                {
                                    "row_id": example.row_id,
                                    "split": example.split,
                                    "task_type": example.task_type,
                                    "answer": answer,
                                    "reward": outcome.reward,
                                    "parse_success": outcome.parse_success,
                                    "caption_token_f1": outcome.caption_token_f1,
                                    "attribute_hit_rate": outcome.attribute_hit_rate,
                                    "object_hit_rate": outcome.object_hit_rate,
                                    "length_score": outcome.length_score,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
    finally:
        if predictions_handle is not None:
            predictions_handle.close()

    total = len(reward_values)
    return {
        "eval_samples": float(total),
        "eval_reward_mean": fmean(reward_values) if reward_values else 0.0,
        "eval_json_object_rate": object_parse_count / max(1, total),
        "eval_json_parse_rate": parse_success_count / max(1, total),
        "eval_caption_token_f1": fmean(caption_f1_values) if caption_f1_values else 0.0,
        "eval_attribute_hit_rate": fmean(attribute_values) if attribute_values else 0.0,
        "eval_object_hit_rate": fmean(object_values) if object_values else 0.0,
        "eval_length_score": fmean(length_values) if length_values else 0.0,
        "eval_task_correct_rate": task_correct_count / max(1, total),
    }
def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    raw_argv = list(argv) if argv is not None else list(os.sys.argv[1:])
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    pre_args, _ = pre_parser.parse_known_args(raw_argv)
    config_path = resolve_config_path(pre_args.config, script_dir=Path(__file__).resolve().parent)
    config = load_json_config(config_path, default_path=DEFAULT_CONFIG_PATH)

    parser = argparse.ArgumentParser(description="ConstructionSite dense caption query RL trainer.")
    parser.add_argument("--config", default=str(config_path))
    parser.add_argument("--env-file", default=str(repo_relative(".env")))
    parser.add_argument("--api-key", default="")
    parser.add_argument("--api-key-env-var", default="MOONDREAM_API_KEY")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--dataset-source", choices=["local_jsonl"], default="local_jsonl")
    parser.add_argument("--dataset-dir", default=str(repo_relative("outputs", "construction_site_query_caption_v1")))
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="validation")
    parser.add_argument("--final-eval-splits", nargs="+", default=DEFAULT_FINAL_EVAL_SPLITS)
    parser.add_argument("--finetune-id", default="")
    parser.add_argument("--finetune-name", default="")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-steps", type=int, default=500)
    parser.add_argument("--resume-step", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--rollout-retries", type=int, default=2)
    parser.add_argument("--rollout-retry-backoff-s", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--eval-temperature", type=float, default=0.0)
    parser.add_argument("--eval-top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=320)
    reasoning_group = parser.add_mutually_exclusive_group()
    reasoning_group.add_argument("--reasoning", dest="reasoning", action="store_true")
    reasoning_group.add_argument("--no-reasoning", dest="reasoning", action="store_false")
    parser.set_defaults(reasoning=False)
    off_policy_group = parser.add_mutually_exclusive_group()
    off_policy_group.add_argument("--off-policy", dest="off_policy", action="store_true")
    off_policy_group.add_argument("--no-off-policy", dest="off_policy", action="store_false")
    parser.set_defaults(off_policy=False)
    parser.add_argument("--off-policy-mix-ratio", type=float, default=0.5)
    parser.add_argument("--off-policy-buffer-size", type=int, default=4096)
    parser.add_argument("--off-policy-warmup-steps", type=int, default=10)
    parser.add_argument("--off-policy-min-buffer-groups", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--save-on-eval", dest="save_on_eval", action="store_true")
    parser.add_argument("--no-save-on-eval", dest="save_on_eval", action="store_false")
    parser.set_defaults(save_on_eval=True)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--eval-max-samples", type=int, default=500)
    parser.add_argument("--save-eval-predictions", dest="save_eval_predictions", action="store_true")
    parser.add_argument("--no-save-eval-predictions", dest="save_eval_predictions", action="store_false")
    parser.set_defaults(save_eval_predictions=True)
    parser.add_argument(
        "--eval-predictions-output-dir",
        default=str(repo_relative("outputs", "eval_predictions", "construction_site_query_caption")),
    )
    parser.add_argument("--best-metric", choices=BEST_METRIC_CHOICES, default="eval_caption_token_f1")
    parser.add_argument("--skip-final-eval", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--wandb-project", default="moondream-construction-site-query-caption-rl")
    parser.add_argument("--wandb-run-name", default="")

    option_to_dest: dict[str, str] = {}
    for action in parser._actions:
        if not action.option_strings:
            continue
        for opt in action.option_strings:
            option_to_dest[opt] = action.dest
    overridden_dests = {option_to_dest[arg] for arg in raw_argv if arg in option_to_dest}
    config_cli_args = config_to_cli_args(
        parser,
        config,
        config_path=config_path,
        overridden_dests=overridden_dests,
    )
    args = parser.parse_args(config_cli_args + raw_argv)
    args.config = str(resolve_config_path(args.config, script_dir=Path(__file__).resolve().parent))
    args.env_file = query_common.resolve_env_file(
        args.env_file,
        repo_root=REPO_ROOT,
        module_root=Path(__file__).resolve().parent,
    )
    args.dataset_dir = query_common.resolve_path(
        args.dataset_dir,
        repo_root=REPO_ROOT,
        module_root=Path(__file__).resolve().parent,
    )
    args.eval_predictions_output_dir = str(
        query_common.resolve_path(
            args.eval_predictions_output_dir,
            repo_root=REPO_ROOT,
            module_root=Path(__file__).resolve().parent,
        )
    )
    return args


def _validate_args(args: argparse.Namespace) -> None:
    if args.eval_every < 0:
        raise ValueError("--eval-every must be >= 0")
    if args.save_every < 0:
        raise ValueError("--save-every must be >= 0")
    if args.batch_size <= 0 or args.group_size <= 0:
        raise ValueError("--batch-size and --group-size must be > 0")
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be > 0")
    if not (0.0 <= args.off_policy_mix_ratio <= 1.0):
        raise ValueError("--off-policy-mix-ratio must be in [0,1]")
    if args.off_policy_buffer_size <= 0:
        raise ValueError("--off-policy-buffer-size must be > 0")
    if args.off_policy_warmup_steps < 0:
        raise ValueError("--off-policy-warmup-steps must be >= 0")
    if args.off_policy_min_buffer_groups <= 0:
        raise ValueError("--off-policy-min-buffer-groups must be > 0")
    if args.off_policy_min_buffer_groups > args.off_policy_buffer_size:
        raise ValueError("--off-policy-min-buffer-groups must be <= --off-policy-buffer-size")


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    query_common.load_dotenv(args.env_file, override=False)
    if not args.api_key:
        args.api_key = os.environ.get(str(args.api_key_env_var or "MOONDREAM_API_KEY"), "")
    if not args.api_key:
        args.api_key = os.environ.get("MOONDREAM_API_KEY", "")
    if not args.base_url:
        args.base_url = os.environ.get("TUNA_BASE_URL") or DEFAULT_BASE_URL
    _validate_args(args)
    if not args.api_key:
        raise ValueError("MOONDREAM_API_KEY is required")

    train_examples = _load_split_examples(split_name=args.train_split, dataset_dir=args.dataset_dir)
    val_examples = _load_split_examples(split_name=args.val_split, dataset_dir=args.dataset_dir)

    client = TunaClient(api_key=args.api_key, base_url=args.base_url)
    finetune = client.get_finetune(args.finetune_id) if args.finetune_id else client.create_finetune(
        name=args.finetune_name or f"construction-site-query-caption-{_random_suffix()}",
        rank=args.rank,
    )
    run = query_common.wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or None,
        config={
            "config": args.config,
            "dataset_dir": str(args.dataset_dir),
            "train_split": args.train_split,
            "val_split": args.val_split,
            "final_eval_splits": list(args.final_eval_splits),
            "finetune_id": finetune.finetune_id,
            "rank": args.rank,
            "num_steps": args.num_steps,
            "batch_size": args.batch_size,
            "group_size": args.group_size,
            "lr": args.lr,
            "max_tokens": args.max_tokens,
            "reasoning": bool(args.reasoning),
            "off_policy": bool(args.off_policy),
            "off_policy_mix_ratio": float(args.off_policy_mix_ratio),
            "off_policy_buffer_size": int(args.off_policy_buffer_size),
            "off_policy_warmup_steps": int(args.off_policy_warmup_steps),
            "off_policy_min_buffer_groups": int(args.off_policy_min_buffer_groups),
            "eval_every": int(args.eval_every),
            "save_every": int(args.save_every),
            "best_metric": str(args.best_metric),
        },
    )
    run.summary["finetune_id"] = finetune.finetune_id
    rng = random.Random(args.seed)
    replay_buffer: deque[Any] = deque(maxlen=int(args.off_policy_buffer_size))
    best_metric_value: Optional[float] = None
    best_checkpoint_step: Optional[int] = None
    latest_checkpoint_step: Optional[int] = None

    for global_step in range(int(args.resume_step), int(args.num_steps)):
        batch = [rng.choice(train_examples) for _ in range(int(args.batch_size))]
        requests, active_examples = query_common.prepare_requests(
            batch,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            reasoning=bool(args.reasoning),
        )
        if not requests:
            print(f"step {global_step}: no usable requests; skipping")
            continue
        try:
            results = query_common.rollouts_batch_with_retry(
                finetune=finetune,
                requests=requests,
                num_rollouts=int(args.group_size),
                max_workers=min(int(args.max_workers), len(requests)),
                retries=int(args.rollout_retries),
                backoff_s=float(args.rollout_retry_backoff_s),
                context=f"train step={global_step}",
            )
        except (TunaAPIError, TunaNetworkError) as exc:
            print(f"step {global_step}: rollouts failed; skipping. details={query_common.error_message(exc)}")
            continue

        on_policy_groups: list[Any] = []
        reward_values: list[float] = []
        parse_success_count = 0
        for example, result in zip(active_examples, results):
            rewards: list[float] = []
            for rollout in result.rollouts:
                outcome = _score_rollout(rollout, example)
                rewards.append(float(outcome.reward))
                reward_values.append(float(outcome.reward))
                parse_success_count += int(outcome.parse_success)
            on_policy_groups.append(result.to_group(rewards=rewards))

        train_groups, off_policy_count = query_common.compose_train_groups(
            on_policy_groups=on_policy_groups,
            replay_groups=replay_buffer,
            off_policy=bool(args.off_policy),
            off_policy_mix_ratio=float(args.off_policy_mix_ratio),
            off_policy_warmup_steps=int(args.off_policy_warmup_steps),
            off_policy_min_buffer_groups=int(args.off_policy_min_buffer_groups),
            global_step=global_step,
            rng=rng,
        )
        try:
            train_out = finetune.train_step(groups=train_groups, lr=args.lr)
        except (TunaAPIError, TunaNetworkError) as exc:
            print(f"step {global_step}: train_step failed; skipping. details={query_common.error_message(exc)}")
            continue
        if args.off_policy:
            replay_buffer.extend(on_policy_groups)

        train_metrics = {
            "reward_mean": fmean(reward_values) if reward_values else 0.0,
            "train_json_parse_rate": parse_success_count / max(1, len(reward_values)),
            "on_policy_groups": float(len(train_groups) - off_policy_count),
            "off_policy_groups": float(off_policy_count),
            "off_policy_group_fraction": off_policy_count / max(1, len(train_groups)),
            "replay_buffer_size": float(len(replay_buffer)),
            "kl": float(train_out.kl or 0.0),
            "router_kl": float(train_out.router_kl or 0.0),
            "grad_norm": float(train_out.grad_norm or 0.0),
        }

        if args.eval_every > 0 and (global_step + 1) % int(args.eval_every) == 0:
            predictions_path = None
            if args.save_eval_predictions:
                predictions_path = Path(args.eval_predictions_output_dir) / f"step{global_step + 1:06d}_validation.jsonl"
            eval_metrics = _evaluate_split(
                finetune=finetune,
                examples=val_examples,
                split_name=args.val_split,
                seed=args.seed + global_step + 1,
                batch_size=int(args.eval_batch_size),
                max_workers=int(args.max_workers),
                max_samples=int(args.eval_max_samples) if args.eval_max_samples else None,
                rollout_retries=int(args.rollout_retries),
                rollout_retry_backoff_s=float(args.rollout_retry_backoff_s),
                temperature=float(args.eval_temperature),
                top_p=float(args.eval_top_p),
                max_tokens=int(args.max_tokens),
                reasoning=bool(args.reasoning),
                show_progress=query_common.progress_enabled(bool(args.no_progress)),
                predictions_path=predictions_path,
            )
            train_metrics.update(eval_metrics)
            metric_value = float(eval_metrics.get(str(args.best_metric), 0.0))
            if best_metric_value is None or metric_value > best_metric_value:
                best_metric_value = metric_value
                run.summary["best_metric"] = metric_value
                run.summary["best_metric_name"] = str(args.best_metric)
                run.summary["best_metric_step"] = global_step + 1
                if args.save_on_eval:
                    saved_step = query_common.save_checkpoint(
                        finetune=finetune,
                        context=f"best metric checkpoint step={global_step + 1}",
                    )
                    if saved_step is not None:
                        best_checkpoint_step = int(saved_step)
                        latest_checkpoint_step = int(saved_step)
                        run.summary["best_checkpoint_step"] = int(saved_step)
                        run.summary["latest_checkpoint_step"] = int(saved_step)
            elif args.save_on_eval:
                saved_step = query_common.save_checkpoint(
                    finetune=finetune,
                    context=f"periodic eval checkpoint step={global_step + 1}",
                )
                if saved_step is not None:
                    latest_checkpoint_step = int(saved_step)
                    run.summary["latest_checkpoint_step"] = int(saved_step)

        query_common.wandb.log(train_metrics, step=global_step)
        print(
            f"step {global_step + 1}/{args.num_steps} reward={train_metrics['reward_mean']:.4f} "
            f"parse={train_metrics['train_json_parse_rate']:.4f} "
            f"offp={off_policy_count}/{len(train_groups)} kl={train_metrics['kl']:.4f}"
        )

        if args.save_every > 0 and (global_step + 1) % int(args.save_every) == 0:
            saved_step = query_common.save_checkpoint(
                finetune=finetune,
                context=f"save_every checkpoint step={global_step + 1}",
            )
            if saved_step is not None:
                latest_checkpoint_step = int(saved_step)
                run.summary["latest_checkpoint_step"] = int(saved_step)

    saved_step = query_common.save_checkpoint(finetune=finetune, context="final checkpoint save")
    if saved_step is not None:
        latest_checkpoint_step = int(saved_step)
        run.summary["latest_checkpoint_step"] = int(saved_step)
    if best_checkpoint_step is not None:
        run.summary["best_checkpoint_step"] = int(best_checkpoint_step)
    if latest_checkpoint_step is not None:
        run.summary["latest_checkpoint_step"] = int(latest_checkpoint_step)

    if not args.skip_final_eval:
        for split_name in list(args.final_eval_splits):
            examples = _load_split_examples(split_name=split_name, dataset_dir=args.dataset_dir)
            predictions_path = None
            if args.save_eval_predictions:
                predictions_path = Path(args.eval_predictions_output_dir) / f"final_{split_name}.jsonl"
            metrics = _evaluate_split(
                finetune=finetune,
                examples=examples,
                split_name=split_name,
                seed=args.seed + 999,
                batch_size=int(args.eval_batch_size),
                max_workers=int(args.max_workers),
                max_samples=None,
                rollout_retries=int(args.rollout_retries),
                rollout_retry_backoff_s=float(args.rollout_retry_backoff_s),
                temperature=float(args.eval_temperature),
                top_p=float(args.eval_top_p),
                max_tokens=int(args.max_tokens),
                reasoning=bool(args.reasoning),
                show_progress=query_common.progress_enabled(bool(args.no_progress)),
                predictions_path=predictions_path,
            )
            prefix = f"final_{split_name}_"
            for key, value in metrics.items():
                run.summary[prefix + key] = value

    run.finish()
    client.close()


if __name__ == "__main__":
    main()
