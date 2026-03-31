#!/usr/bin/env python3
"""Train a query-model finetune for VQA-RAD."""

from __future__ import annotations

import argparse
import json
import os
import random
import string
import sys
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from construction_site import query_common  # noqa: E402
from tuna_sdk import TunaClient  # noqa: E402
from tuna_sdk.errors import TunaAPIError, TunaNetworkError  # noqa: E402
from vqa_rad.common import (  # noqa: E402
    ANSWER_TYPE_CLOSE,
    ANSWER_TYPE_OPEN,
    DEFAULT_QUERY_OUTPUT_DIR,
    DEFAULT_STAGING_API_BASE,
    PREDICTION_FORMAT_NONE,
    PREDICTION_FORMAT_PLAIN_TEXT,
    QUERY_TASK_TYPE,
    brevity_score,
    clamp,
    config_to_cli_args,
    extract_prediction_answer,
    infer_answer_type,
    load_json_config,
    normalize_close_answer,
    normalize_open_answer,
    numeric_match,
    parse_answer_payload,
    parse_prediction_json,
    repo_relative,
    resolve_config_path,
    token_f1,
)

DEFAULT_CONFIG_PATH = repo_relative("configs", "train_vqa_rad_query_default.json")
DEFAULT_FINAL_EVAL_SPLITS = ["validation", "test"]
DEFAULT_BASE_URL = "https://api.moondream.ai/v1"
DEFAULT_ANSWER_TYPE_SAMPLING_WEIGHTS = {
    ANSWER_TYPE_CLOSE: 1.0,
    ANSWER_TYPE_OPEN: 1.5,
}
BEST_METRIC_CHOICES = (
    "eval_balanced_accuracy",
    "eval_overall_accuracy",
    "eval_close_accuracy",
    "eval_open_accuracy",
    "eval_reward_mean",
    "eval_json_parse_rate",
)


@dataclass(frozen=True)
class VQARadExample:
    row_id: str
    split: str
    source_split: str
    task_type: str
    question: str
    image_path: Path
    question_family: str
    answer_type: str
    answer_text: str
    normalized_answer: str
    image_group_id: str


@dataclass(frozen=True)
class VQARadScoreOutcome(query_common.ScoreOutcome):
    answer_type: str
    question_family: str
    normalized_prediction: str
    prediction_format: str
    exact_match: float
    close_accuracy: float
    open_accuracy: float
    open_token_f1: float
    numeric_match: float
    brevity_score: float
    strict_parse_success: bool


@dataclass
class MetricState:
    reward_values: list[float] = field(default_factory=list)
    open_token_f1_values: list[float] = field(default_factory=list)
    json_object_count: int = 0
    parse_success_count: int = 0
    strict_parse_success_count: int = 0
    plain_text_count: int = 0
    task_correct_count: int = 0
    close_total: int = 0
    close_correct: int = 0
    open_total: int = 0
    open_correct: int = 0
    family_total: Counter[str] = field(default_factory=Counter)
    family_correct: Counter[str] = field(default_factory=Counter)


def _random_suffix(length: int = 6) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return "".join(random.choices(alphabet, k=length))


def _parse_answer_type_sampling_weights(raw_value: Any) -> dict[str, float]:
    payload = raw_value
    if isinstance(payload, str):
        text = payload.strip()
        if not text:
            payload = {}
        else:
            payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("--answer-type-sampling-weights must be a JSON object")
    weights = dict(DEFAULT_ANSWER_TYPE_SAMPLING_WEIGHTS)
    for key, value in payload.items():
        name = str(key).strip()
        if name not in weights:
            raise ValueError(f"Unsupported answer_type weight key: {name!r}")
        weights[name] = float(value)
    for key, value in weights.items():
        if value <= 0.0:
            raise ValueError(f"answer_type weight must be > 0 for {key}")
    return weights


def _build_example(row: dict[str, Any], *, dataset_dir: Path, split_name: str, line_number: int) -> Optional[VQARadExample]:
    if str(row.get("task_type") or "").strip() != QUERY_TASK_TYPE:
        return None
    image_path = query_common.existing_path(str(row.get("image_path") or ""), dataset_dir=dataset_dir)
    if image_path is None:
        raise FileNotFoundError(f"split={split_name} line={line_number} image_path not found: {row.get('image_path')!r}")
    final_answer = parse_prediction_json(str(row.get("final_answer_json") or ""))
    answer_text = str(row.get("answer_text") or parse_answer_payload(final_answer) or "").strip()
    if not answer_text:
        raise ValueError(f"split={split_name} line={line_number} missing answer_text")
    answer_type = str(row.get("answer_type") or infer_answer_type(answer_text)).strip()
    normalized_answer = normalize_close_answer(answer_text) if answer_type == ANSWER_TYPE_CLOSE else normalize_open_answer(answer_text)
    return VQARadExample(
        row_id=str(row.get("row_id") or f"{split_name}_{line_number:06d}"),
        split=str(row.get("split") or split_name),
        source_split=str(row.get("source_split") or split_name),
        task_type=QUERY_TASK_TYPE,
        question=str(row.get("question") or ""),
        image_path=image_path,
        question_family=str(row.get("question_family") or "other"),
        answer_type=answer_type,
        answer_text=answer_text,
        normalized_answer=normalized_answer,
        image_group_id=str(row.get("image_group_id") or ""),
    )


def _load_split_examples(*, split_name: str, dataset_dir: Path) -> list[VQARadExample]:
    rows = query_common.load_local_jsonl_rows(dataset_dir=dataset_dir, split_name=split_name)
    examples: list[VQARadExample] = []
    skipped = 0
    for line_number, row in enumerate(rows, start=1):
        example = _build_example(row, dataset_dir=dataset_dir, split_name=split_name, line_number=line_number)
        if example is None:
            skipped += 1
            continue
        examples.append(example)
    print(f"usable split={split_name} examples={len(examples)} skipped={skipped}")
    if not examples:
        raise ValueError(f"split={split_name} contains no usable vqa_rad rows")
    return examples


def _zero_outcome(
    *,
    answer_type: str,
    question_family: str,
    json_object_parsed: bool,
    prediction_format: str = PREDICTION_FORMAT_NONE,
) -> VQARadScoreOutcome:
    return VQARadScoreOutcome(
        reward=0.0,
        parse_success=False,
        task_correct=False,
        json_object_parsed=json_object_parsed,
        answer_type=answer_type,
        question_family=question_family,
        normalized_prediction="",
        prediction_format=prediction_format,
        exact_match=0.0,
        close_accuracy=0.0,
        open_accuracy=0.0,
        open_token_f1=0.0,
        numeric_match=0.0,
        brevity_score=0.0,
        strict_parse_success=False,
    )


def _score_answer_text(
    example: VQARadExample,
    answer_text: str,
    *,
    open_exact_weight: float,
    open_token_f1_weight: float,
    open_numeric_weight: float,
    open_brevity_weight: float,
) -> VQARadScoreOutcome:
    extraction = extract_prediction_answer(answer_text, answer_type=example.answer_type)
    predicted_answer = extraction.answer
    json_object_parsed = extraction.json_object_parsed
    if predicted_answer is None:
        return _zero_outcome(
            answer_type=example.answer_type,
            question_family=example.question_family,
            json_object_parsed=json_object_parsed,
            prediction_format=extraction.prediction_format,
        )

    if example.answer_type == ANSWER_TYPE_CLOSE:
        normalized_prediction = normalize_close_answer(predicted_answer)
        exact_match = 1.0 if normalized_prediction == example.normalized_answer else 0.0
        reward = 1.0 if exact_match else 0.05
        return VQARadScoreOutcome(
            reward=float(reward),
            parse_success=True,
            task_correct=bool(exact_match),
            json_object_parsed=json_object_parsed,
            answer_type=example.answer_type,
            question_family=example.question_family,
            normalized_prediction=normalized_prediction,
            prediction_format=extraction.prediction_format,
            exact_match=exact_match,
            close_accuracy=exact_match,
            open_accuracy=0.0,
            open_token_f1=0.0,
            numeric_match=0.0,
            brevity_score=1.0,
            strict_parse_success=bool(extraction.strict_parse_success),
        )

    normalized_prediction = normalize_open_answer(predicted_answer)
    exact_match = 1.0 if normalized_prediction and normalized_prediction == example.normalized_answer else 0.0
    open_f1 = token_f1(example.normalized_answer, normalized_prediction)
    numeric = numeric_match(example.answer_text, predicted_answer)
    brevity = brevity_score(example.answer_text, predicted_answer)
    reward = clamp(
        0.05
        + (float(open_exact_weight) * exact_match)
        + (float(open_token_f1_weight) * open_f1)
        + (float(open_numeric_weight) * numeric)
        + (float(open_brevity_weight) * brevity)
    )
    return VQARadScoreOutcome(
        reward=float(reward),
        parse_success=True,
        task_correct=bool(exact_match),
        json_object_parsed=json_object_parsed,
        answer_type=example.answer_type,
        question_family=example.question_family,
        normalized_prediction=normalized_prediction,
        prediction_format=extraction.prediction_format,
        exact_match=exact_match,
        close_accuracy=0.0,
        open_accuracy=exact_match,
        open_token_f1=open_f1,
        numeric_match=numeric,
        brevity_score=brevity,
        strict_parse_success=bool(extraction.strict_parse_success),
    )


def _score_rollout(
    rollout: Any,
    example: VQARadExample,
    *,
    open_exact_weight: float,
    open_token_f1_weight: float,
    open_numeric_weight: float,
    open_brevity_weight: float,
) -> VQARadScoreOutcome:
    answer = ""
    if rollout is not None and getattr(rollout, "output", None) is not None:
        answer = str(getattr(rollout.output, "answer", "") or "")
    return _score_answer_text(
        example,
        answer,
        open_exact_weight=open_exact_weight,
        open_token_f1_weight=open_token_f1_weight,
        open_numeric_weight=open_numeric_weight,
        open_brevity_weight=open_brevity_weight,
    )


def _new_metric_state() -> MetricState:
    return MetricState()


def _record_outcome(state: MetricState, example: VQARadExample, outcome: VQARadScoreOutcome) -> None:
    state.reward_values.append(float(outcome.reward))
    if example.answer_type == ANSWER_TYPE_CLOSE:
        state.close_total += 1
        state.close_correct += int(outcome.close_accuracy >= 1.0)
    else:
        state.open_total += 1
        state.open_correct += int(outcome.open_accuracy >= 1.0)
        state.open_token_f1_values.append(float(outcome.open_token_f1))
    state.family_total[example.question_family] += 1
    state.family_correct[example.question_family] += int(outcome.task_correct)
    if outcome.json_object_parsed:
        state.json_object_count += 1
    if outcome.parse_success:
        state.parse_success_count += 1
    if outcome.strict_parse_success:
        state.strict_parse_success_count += 1
    if outcome.prediction_format == PREDICTION_FORMAT_PLAIN_TEXT:
        state.plain_text_count += 1
    if outcome.task_correct:
        state.task_correct_count += 1


def _family_breakdown(state: MetricState) -> dict[str, dict[str, float]]:
    return {
        family: {
            "rows": float(total),
            "accuracy": float(state.family_correct.get(family, 0)) / float(max(1, total)),
        }
        for family, total in sorted(state.family_total.items())
    }


def _finalize_metrics(
    state: MetricState,
    *,
    prefix: str,
    include_family_breakdown: bool = False,
) -> dict[str, Any]:
    total = len(state.reward_values)
    close_accuracy = float(state.close_correct) / float(max(1, state.close_total)) if state.close_total else 0.0
    open_accuracy = float(state.open_correct) / float(max(1, state.open_total)) if state.open_total else 0.0
    if state.close_total and state.open_total:
        balanced_accuracy = (close_accuracy + open_accuracy) / 2.0
    elif state.close_total:
        balanced_accuracy = close_accuracy
    elif state.open_total:
        balanced_accuracy = open_accuracy
    else:
        balanced_accuracy = 0.0
    metrics: dict[str, Any] = {
        f"{prefix}samples": float(total),
        f"{prefix}reward_mean": fmean(state.reward_values) if state.reward_values else 0.0,
        f"{prefix}json_object_rate": state.json_object_count / max(1, total),
        f"{prefix}json_parse_rate": state.parse_success_count / max(1, total),
        f"{prefix}strict_json_parse_rate": state.strict_parse_success_count / max(1, total),
        f"{prefix}plain_text_rate": state.plain_text_count / max(1, total),
        f"{prefix}overall_accuracy": state.task_correct_count / max(1, total),
        f"{prefix}balanced_accuracy": balanced_accuracy,
        f"{prefix}close_accuracy": close_accuracy,
        f"{prefix}open_accuracy": open_accuracy,
        f"{prefix}open_token_f1": fmean(state.open_token_f1_values) if state.open_token_f1_values else 0.0,
        f"{prefix}close_rows": float(state.close_total),
        f"{prefix}open_rows": float(state.open_total),
    }
    if include_family_breakdown:
        metrics["question_family_breakdown"] = _family_breakdown(state)
    return metrics


def _train_log_metrics(state: MetricState) -> dict[str, float]:
    finalized = _finalize_metrics(state, prefix="train_")
    return {
        "reward_mean": float(finalized["train_reward_mean"]),
        "train_json_object_rate": float(finalized["train_json_object_rate"]),
        "train_json_parse_rate": float(finalized["train_json_parse_rate"]),
        "train_strict_json_parse_rate": float(finalized["train_strict_json_parse_rate"]),
        "train_plain_text_rate": float(finalized["train_plain_text_rate"]),
        "train_overall_accuracy": float(finalized["train_overall_accuracy"]),
        "train_balanced_accuracy": float(finalized["train_balanced_accuracy"]),
        "train_close_accuracy": float(finalized["train_close_accuracy"]),
        "train_open_accuracy": float(finalized["train_open_accuracy"]),
        "train_open_token_f1": float(finalized["train_open_token_f1"]),
    }


def _prediction_record(
    *,
    example: VQARadExample,
    answer_text: str,
    outcome: VQARadScoreOutcome,
    prediction_json: Optional[dict[str, Any]],
    raw_response: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "row_id": example.row_id,
        "split": example.split,
        "source_split": example.source_split,
        "answer_type": example.answer_type,
        "question_family": example.question_family,
        "answer": answer_text,
        "prediction_json": prediction_json,
        "prediction_format": outcome.prediction_format,
        "normalized_prediction": outcome.normalized_prediction,
        "reward": outcome.reward,
        "parse_success": outcome.parse_success,
        "strict_parse_success": outcome.strict_parse_success,
        "task_correct": outcome.task_correct,
        "exact_match": outcome.exact_match,
        "close_accuracy": outcome.close_accuracy,
        "open_accuracy": outcome.open_accuracy,
        "open_token_f1": outcome.open_token_f1,
        "numeric_match": outcome.numeric_match,
        "brevity_score": outcome.brevity_score,
    }
    if raw_response is not None:
        record["raw_response"] = raw_response
    return record


def _evaluate_split(
    *,
    finetune: Any,
    examples: list[VQARadExample],
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
    open_exact_weight: float,
    open_token_f1_weight: float,
    open_numeric_weight: float,
    open_brevity_weight: float,
    show_progress: bool,
    predictions_path: Optional[Path] = None,
) -> dict[str, Any]:
    indices = list(range(len(examples)))
    random.Random(seed).shuffle(indices)
    if max_samples is not None:
        indices = indices[: max(0, min(int(max_samples), len(indices)))]

    state = _new_metric_state()
    predictions_handle = None
    if predictions_path is not None:
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        predictions_handle = predictions_path.open("w", encoding="utf-8")

    try:
        batch: list[VQARadExample] = []
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
                    outcome = _score_rollout(
                        rollout,
                        example,
                        open_exact_weight=open_exact_weight,
                        open_token_f1_weight=open_token_f1_weight,
                        open_numeric_weight=open_numeric_weight,
                        open_brevity_weight=open_brevity_weight,
                    )
                    _record_outcome(state, example, outcome)
                    if predictions_handle is not None:
                        answer_text = ""
                        if rollout is not None and getattr(rollout, "output", None) is not None:
                            answer_text = str(getattr(rollout.output, "answer", "") or "")
                        predictions_handle.write(
                            json.dumps(
                                _prediction_record(
                                    example=example,
                                    answer_text=answer_text,
                                    outcome=outcome,
                                    prediction_json=parse_prediction_json(answer_text),
                                ),
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
                    outcome = _score_rollout(
                        rollout,
                        example,
                        open_exact_weight=open_exact_weight,
                        open_token_f1_weight=open_token_f1_weight,
                        open_numeric_weight=open_numeric_weight,
                        open_brevity_weight=open_brevity_weight,
                    )
                    _record_outcome(state, example, outcome)
                    if predictions_handle is not None:
                        answer_text = ""
                        if rollout is not None and getattr(rollout, "output", None) is not None:
                            answer_text = str(getattr(rollout.output, "answer", "") or "")
                        predictions_handle.write(
                            json.dumps(
                                _prediction_record(
                                    example=example,
                                    answer_text=answer_text,
                                    outcome=outcome,
                                    prediction_json=parse_prediction_json(answer_text),
                                ),
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
    finally:
        if predictions_handle is not None:
            predictions_handle.close()

    return _finalize_metrics(state, prefix="eval_")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    raw_argv = list(argv) if argv is not None else list(os.sys.argv[1:])
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    pre_args, _ = pre_parser.parse_known_args(raw_argv)
    config_path = resolve_config_path(pre_args.config, script_dir=Path(__file__).resolve().parent)
    config = load_json_config(config_path, default_path=DEFAULT_CONFIG_PATH)

    parser = argparse.ArgumentParser(description="VQA-RAD query RL trainer.")
    parser.add_argument("--config", default=str(config_path))
    parser.add_argument("--env-file", default=str(repo_relative(".env.staging")))
    parser.add_argument("--api-key", default="")
    parser.add_argument("--api-key-env-var", default="CICID_GPUB_MOONDREAM_API_KEY_1")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--dataset-source", choices=["local_jsonl"], default="local_jsonl")
    parser.add_argument("--dataset-dir", default=str(DEFAULT_QUERY_OUTPUT_DIR))
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="validation")
    parser.add_argument("--final-eval-splits", nargs="+", default=DEFAULT_FINAL_EVAL_SPLITS)
    parser.add_argument("--finetune-id", default="")
    parser.add_argument("--finetune-name", default="")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-steps", type=int, default=400)
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
    parser.add_argument("--max-tokens", type=int, default=64)
    reasoning_group = parser.add_mutually_exclusive_group()
    reasoning_group.add_argument("--reasoning", dest="reasoning", action="store_true")
    reasoning_group.add_argument("--no-reasoning", dest="reasoning", action="store_false")
    parser.set_defaults(reasoning=False)
    parser.add_argument("--answer-type-sampling-weights", default=json.dumps(DEFAULT_ANSWER_TYPE_SAMPLING_WEIGHTS))
    off_policy_group = parser.add_mutually_exclusive_group()
    off_policy_group.add_argument("--off-policy", dest="off_policy", action="store_true")
    off_policy_group.add_argument("--no-off-policy", dest="off_policy", action="store_false")
    parser.set_defaults(off_policy=False)
    parser.add_argument("--off-policy-mix-ratio", type=float, default=0.25)
    parser.add_argument("--off-policy-buffer-size", type=int, default=1024)
    parser.add_argument("--off-policy-warmup-steps", type=int, default=10)
    parser.add_argument("--off-policy-min-buffer-groups", type=int, default=64)
    parser.add_argument("--open-exact-weight", type=float, default=0.55)
    parser.add_argument("--open-token-f1-weight", type=float, default=0.25)
    parser.add_argument("--open-numeric-weight", type=float, default=0.10)
    parser.add_argument("--open-brevity-weight", type=float, default=0.05)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--save-on-eval", dest="save_on_eval", action="store_true")
    parser.add_argument("--no-save-on-eval", dest="save_on_eval", action="store_false")
    parser.set_defaults(save_on_eval=True)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--eval-max-samples", type=int, default=256)
    parser.add_argument("--save-eval-predictions", dest="save_eval_predictions", action="store_true")
    parser.add_argument("--no-save-eval-predictions", dest="save_eval_predictions", action="store_false")
    parser.set_defaults(save_eval_predictions=True)
    parser.add_argument(
        "--eval-predictions-output-dir",
        default=str(repo_relative("outputs", "eval_predictions", "vqa_rad_query")),
    )
    parser.add_argument("--best-metric", choices=BEST_METRIC_CHOICES, default="eval_balanced_accuracy")
    parser.add_argument("--skip-final-eval", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--wandb-project", default="moondream-vqa-rad-query-rl")
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
    args.answer_type_sampling_weights = _parse_answer_type_sampling_weights(args.answer_type_sampling_weights)
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
    train_weights = [
        float(args.answer_type_sampling_weights.get(example.answer_type, 1.0))
        for example in train_examples
    ]

    client = TunaClient(api_key=args.api_key, base_url=args.base_url)
    finetune = client.get_finetune(args.finetune_id) if args.finetune_id else client.create_finetune(
        name=args.finetune_name or f"vqa-rad-query-{_random_suffix()}",
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
            "answer_type_sampling_weights": dict(args.answer_type_sampling_weights),
            "off_policy": bool(args.off_policy),
            "off_policy_mix_ratio": float(args.off_policy_mix_ratio),
            "off_policy_buffer_size": int(args.off_policy_buffer_size),
            "off_policy_warmup_steps": int(args.off_policy_warmup_steps),
            "off_policy_min_buffer_groups": int(args.off_policy_min_buffer_groups),
            "open_exact_weight": float(args.open_exact_weight),
            "open_token_f1_weight": float(args.open_token_f1_weight),
            "open_numeric_weight": float(args.open_numeric_weight),
            "open_brevity_weight": float(args.open_brevity_weight),
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
        batch = rng.choices(train_examples, weights=train_weights, k=int(args.batch_size))
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
        train_state = _new_metric_state()
        for example, result in zip(active_examples, results):
            rewards: list[float] = []
            for rollout in result.rollouts:
                outcome = _score_rollout(
                    rollout,
                    example,
                    open_exact_weight=float(args.open_exact_weight),
                    open_token_f1_weight=float(args.open_token_f1_weight),
                    open_numeric_weight=float(args.open_numeric_weight),
                    open_brevity_weight=float(args.open_brevity_weight),
                )
                rewards.append(float(outcome.reward))
                _record_outcome(train_state, example, outcome)
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

        train_metrics = _train_log_metrics(train_state)
        train_metrics.update(
            {
                "on_policy_groups": float(len(train_groups) - off_policy_count),
                "off_policy_groups": float(off_policy_count),
                "off_policy_group_fraction": off_policy_count / max(1, len(train_groups)),
                "replay_buffer_size": float(len(replay_buffer)),
                "kl": float(train_out.kl or 0.0),
                "router_kl": float(train_out.router_kl or 0.0),
                "grad_norm": float(train_out.grad_norm or 0.0),
            }
        )

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
                open_exact_weight=float(args.open_exact_weight),
                open_token_f1_weight=float(args.open_token_f1_weight),
                open_numeric_weight=float(args.open_numeric_weight),
                open_brevity_weight=float(args.open_brevity_weight),
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
            f"bal={train_metrics['train_balanced_accuracy']:.4f} "
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
                open_exact_weight=float(args.open_exact_weight),
                open_token_f1_weight=float(args.open_token_f1_weight),
                open_numeric_weight=float(args.open_numeric_weight),
                open_brevity_weight=float(args.open_brevity_weight),
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
