#!/usr/bin/env python3
"""Train a query-model finetune for ConstructionSite rule VQA."""

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

from async_checkpoint_eval import (
    CheckpointEvalResult,
    DispatchHandle,
    dispatch_checkpoint_eval,
    drain_checkpoint_eval_jobs,
    poll_checkpoint_eval_jobs,
)
from construction_site.common import (  # noqa: E402
    RULE_VQA_TASK_TYPE,
    config_to_cli_args,
    load_json_config,
    parse_prediction_json,
    repo_relative,
    resolve_config_path,
    set_f1,
    token_f1,
)
from construction_site import query_common  # noqa: E402
from tuna_sdk import TunaClient  # noqa: E402
from tuna_sdk.errors import TunaAPIError, TunaNetworkError  # noqa: E402

DEFAULT_BASE_URL = "https://api.moondream.ai/v1"
DEFAULT_CONFIG_PATH = repo_relative("configs", "train_construction_site_query_rule_vqa_default.json")
DEFAULT_FINAL_EVAL_SPLITS = ["validation", "test"]
BEST_METRIC_CHOICES = (
    "eval_reward_mean",
    "eval_json_parse_rate",
    "eval_rule_set_accuracy",
    "eval_rule_set_f1",
    "eval_reason_token_f1",
    "eval_positive_rule_set_f1",
    "eval_positive_strict_rule_set_f1",
    "eval_strict_reward_mean",
    "eval_strict_json_parse_rate",
    "eval_strict_rule_set_accuracy",
    "eval_strict_rule_set_f1",
    "eval_strict_reason_token_f1",
)
VALID_RULE_IDS = {1, 2, 3, 4}
SOFT_RULE_ID_HINTS: dict[int, tuple[str, ...]] = {
    1: ("1", "ppe", "hard hat", "protective clothing"),
    2: ("2", "height", "harness"),
    3: ("3", "excavation", "guardrail", "warning barrier", "drop hazard"),
    4: ("4", "excavator", "blind spot", "operating radius"),
}


@dataclass(frozen=True)
class RuleVQAExample:
    row_id: str
    split: str
    task_type: str
    question: str
    image_path: Path
    violated_rules: tuple[int, ...]
    reasons: dict[int, str]


@dataclass(frozen=True)
class RuleVQAScoreOutcome(query_common.ScoreOutcome):
    rule_set_accuracy: float
    rule_set_f1: float
    reason_token_f1: float
    no_violation_correct: bool
    hallucinated_rule_count: int
    strict_reward: float
    strict_parse_success: bool
    strict_task_correct: bool
    strict_rule_set_accuracy: float
    strict_rule_set_f1: float
    strict_reason_token_f1: float
    strict_no_violation_correct: bool
    strict_hallucinated_rule_count: int


def _random_suffix(length: int = 6) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return "".join(random.choices(alphabet, k=length))


def _normalize_rule_id(value: Any) -> Optional[int]:
    try:
        rule_id = int(value)
    except (TypeError, ValueError):
        return None
    return rule_id if rule_id in VALID_RULE_IDS else None


def _text_length_score(reference_text: str, predicted_text: str) -> float:
    ref_len = max(1, len(str(reference_text or "").split()))
    pred_len = len(str(predicted_text or "").split())
    ratio = float(pred_len) / float(ref_len)
    if 0.5 <= ratio <= 1.75:
        return 1.0
    if ratio <= 0.0:
        return 0.0
    if ratio < 0.5:
        return max(0.0, ratio / 0.5)
    return max(0.0, 1.0 - min(1.0, (ratio - 1.75) / 1.75))


def _soft_normalize_rule_id(value: Any) -> Optional[int]:
    strict_rule_id = _normalize_rule_id(value)
    if strict_rule_id is not None:
        return strict_rule_id
    text = " ".join(str(value or "").strip().lower().split())
    if not text:
        return None
    for rule_id, hints in SOFT_RULE_ID_HINTS.items():
        if any(hint in text for hint in hints):
            return rule_id
    return None


def _normalize_rule_ids(value: Any) -> Optional[tuple[int, ...]]:
    if not isinstance(value, list):
        return None
    normalized: set[int] = set()
    for item in value:
        rule_id = _normalize_rule_id(item)
        if rule_id is None:
            return None
        normalized.add(rule_id)
    return tuple(sorted(normalized))


def _normalize_reasons(value: Any) -> Optional[dict[int, str]]:
    if not isinstance(value, dict):
        return None
    reasons: dict[int, str] = {}
    for raw_key, raw_reason in value.items():
        rule_id = _normalize_rule_id(raw_key)
        if rule_id is None:
            return None
        reason = str(raw_reason or "").strip()
        reasons[rule_id] = reason
    return reasons


def _parse_rule_payload(payload: Any) -> Optional[tuple[tuple[int, ...], dict[int, str]]]:
    if not isinstance(payload, dict):
        return None
    violated_rules = _normalize_rule_ids(payload.get("violated_rules"))
    reasons = _normalize_reasons(payload.get("reasons"))
    if violated_rules is None or reasons is None:
        return None
    return violated_rules, reasons


def _soften_rule_payload(payload: Any) -> Optional[dict[str, Any]]:
    if not isinstance(payload, dict):
        return None

    raw_rules = payload.get("violated_rules")
    if not isinstance(raw_rules, list):
        return None

    predicted_rules: set[int] = set()
    predicted_reasons: dict[int, str] = {}

    def add_reason(rule_id: int, value: Any) -> None:
        text = str(value or "").strip()
        if text:
            predicted_reasons[rule_id] = text

    for item in raw_rules:
        if isinstance(item, dict):
            nested_rules = item.get("violated_rules")
            nested_reasons = item.get("reasons")
            if nested_rules is not None:
                nested_payload = _soften_rule_payload(
                    {
                        "violated_rules": nested_rules,
                        "reasons": nested_reasons if nested_reasons is not None else {},
                    }
                )
                nested_parsed = _parse_rule_payload(nested_payload) if nested_payload is not None else None
                if nested_parsed is not None:
                    for rule_id in nested_parsed[0]:
                        predicted_rules.add(rule_id)
                    predicted_reasons.update(nested_parsed[1])
                continue
            if len(item) == 1:
                raw_key, raw_value = next(iter(item.items()))
                rule_id = _soft_normalize_rule_id(raw_key) or _soft_normalize_rule_id(raw_value)
                if rule_id is None:
                    continue
                predicted_rules.add(rule_id)
                if _soft_normalize_rule_id(raw_value) is None:
                    add_reason(rule_id, raw_value)
                continue
        rule_id = _soft_normalize_rule_id(item)
        if rule_id is not None:
            predicted_rules.add(rule_id)

    raw_reasons = payload.get("reasons")
    if isinstance(raw_reasons, dict):
        for raw_key, raw_value in raw_reasons.items():
            rule_id = _soft_normalize_rule_id(raw_key) or _soft_normalize_rule_id(raw_value)
            if rule_id is None:
                continue
            add_reason(rule_id, raw_value)
    elif isinstance(raw_reasons, list):
        reason_texts = [str(item or "").strip() for item in raw_reasons if str(item or "").strip()]
        sorted_rules = sorted(predicted_rules)
        if len(reason_texts) == len(sorted_rules):
            for rule_id, text in zip(sorted_rules, reason_texts):
                add_reason(rule_id, text)
        elif len(reason_texts) == 1 and len(sorted_rules) == 1:
            add_reason(sorted_rules[0], reason_texts[0])
    elif raw_reasons is None:
        pass

    return {
        "violated_rules": sorted(predicted_rules),
        "reasons": {str(rule_id): text for rule_id, text in sorted(predicted_reasons.items())},
    }


def _build_example(row: dict[str, Any], *, dataset_dir: Path, split_name: str, line_number: int) -> Optional[RuleVQAExample]:
    if str(row.get("task_type") or "").strip() != RULE_VQA_TASK_TYPE:
        return None
    image_path = query_common.existing_path(str(row.get("image_path") or ""), dataset_dir=dataset_dir)
    if image_path is None:
        raise FileNotFoundError(f"split={split_name} line={line_number} image_path not found: {row.get('image_path')!r}")
    final_answer = json.loads(str(row.get("final_answer_json") or "{}"))
    parsed = _parse_rule_payload(final_answer)
    if parsed is None:
        raise ValueError(f"split={split_name} line={line_number} invalid final_answer_json schema")
    violated_rules, reasons = parsed
    return RuleVQAExample(
        row_id=str(row.get("row_id") or f"{split_name}_{line_number:06d}"),
        split=str(row.get("split") or split_name),
        task_type=RULE_VQA_TASK_TYPE,
        question=str(row.get("question") or ""),
        image_path=image_path,
        violated_rules=violated_rules,
        reasons=reasons,
    )


def _load_split_examples(*, split_name: str, dataset_dir: Path) -> list[RuleVQAExample]:
    rows = query_common.load_local_jsonl_rows(dataset_dir=dataset_dir, split_name=split_name)
    examples: list[RuleVQAExample] = []
    skipped = 0
    for line_number, row in enumerate(rows, start=1):
        example = _build_example(row, dataset_dir=dataset_dir, split_name=split_name, line_number=line_number)
        if example is None:
            skipped += 1
            continue
        examples.append(example)
    print(f"usable split={split_name} examples={len(examples)} skipped={skipped}")
    if not examples:
        raise ValueError(f"split={split_name} contains no usable rule-vqa rows")
    return examples


def _empty_score_fields() -> dict[str, Any]:
    return {
        "reward": 0.0,
        "parse_success": False,
        "task_correct": False,
        "rule_set_accuracy": 0.0,
        "rule_set_f1": 0.0,
        "reason_token_f1": 0.0,
        "no_violation_correct": False,
        "hallucinated_rule_count": 0,
    }


def _score_parsed_prediction(
    example: RuleVQAExample,
    predicted_rules: tuple[int, ...],
    predicted_reasons: dict[int, str],
) -> dict[str, Any]:
    reference_rules = tuple(example.violated_rules)
    exact_accuracy = 1.0 if predicted_rules == reference_rules else 0.0
    rule_f1 = set_f1(reference_rules, predicted_rules)
    overlap = sorted(set(reference_rules) & set(predicted_rules))
    if not reference_rules and not predicted_rules:
        reason_f1 = 1.0
        reason_length_score = 1.0
    elif not overlap:
        reason_f1 = 0.0
        reason_length_score = 0.0
    else:
        reason_f1 = fmean(
            token_f1(example.reasons.get(rule_id, ""), predicted_reasons.get(rule_id, ""))
            for rule_id in overlap
        )
        reason_length_score = fmean(
            _text_length_score(example.reasons.get(rule_id, ""), predicted_reasons.get(rule_id, ""))
            for rule_id in overlap
        )
    reason_quality = max(0.0, min(1.0, (0.85 * reason_f1) + (0.15 * reason_length_score)))
    hallucinated_rules = sorted(set(predicted_rules) - set(reference_rules))
    hallucination_penalty = min(0.45, 0.15 * len(hallucinated_rules))
    if not reference_rules:
        reward = 0.85 if not predicted_rules else max(0.0, 0.10 - hallucination_penalty)
    else:
        reward = max(
            0.0,
            min(1.0, (0.70 * rule_f1) + (0.20 * reason_quality) + (0.10 * exact_accuracy) - hallucination_penalty),
        )
    no_violation_correct = not reference_rules and not predicted_rules
    task_correct = bool(exact_accuracy) and (reason_quality >= 0.80 if reference_rules else True)
    return {
        "reward": reward,
        "parse_success": True,
        "task_correct": task_correct,
        "rule_set_accuracy": exact_accuracy,
        "rule_set_f1": rule_f1,
        "reason_token_f1": reason_f1,
        "no_violation_correct": no_violation_correct,
        "hallucinated_rule_count": len(hallucinated_rules),
    }


def _score_payload_for_example(example: RuleVQAExample, payload: Any) -> RuleVQAScoreOutcome:
    json_object_parsed = isinstance(payload, dict)
    strict_parsed = _parse_rule_payload(payload)
    soft_payload = _soften_rule_payload(payload)
    soft_parsed = _parse_rule_payload(soft_payload) if soft_payload is not None else None

    strict_fields = _empty_score_fields()
    if strict_parsed is not None:
        strict_fields = _score_parsed_prediction(example, strict_parsed[0], strict_parsed[1])

    soft_fields = _empty_score_fields()
    if soft_parsed is not None:
        soft_fields = _score_parsed_prediction(example, soft_parsed[0], soft_parsed[1])

    return RuleVQAScoreOutcome(
        reward=float(soft_fields["reward"]),
        parse_success=bool(soft_fields["parse_success"]),
        task_correct=bool(soft_fields["task_correct"]),
        json_object_parsed=json_object_parsed,
        rule_set_accuracy=float(soft_fields["rule_set_accuracy"]),
        rule_set_f1=float(soft_fields["rule_set_f1"]),
        reason_token_f1=float(soft_fields["reason_token_f1"]),
        no_violation_correct=bool(soft_fields["no_violation_correct"]),
        hallucinated_rule_count=int(soft_fields["hallucinated_rule_count"]),
        strict_reward=float(strict_fields["reward"]),
        strict_parse_success=bool(strict_fields["parse_success"]),
        strict_task_correct=bool(strict_fields["task_correct"]),
        strict_rule_set_accuracy=float(strict_fields["rule_set_accuracy"]),
        strict_rule_set_f1=float(strict_fields["rule_set_f1"]),
        strict_reason_token_f1=float(strict_fields["reason_token_f1"]),
        strict_no_violation_correct=bool(strict_fields["no_violation_correct"]),
        strict_hallucinated_rule_count=int(strict_fields["hallucinated_rule_count"]),
    )


def _score_rollout(rollout: Any, example: RuleVQAExample) -> RuleVQAScoreOutcome:
    answer = ""
    if rollout is not None and getattr(rollout, "output", None) is not None:
        answer = str(getattr(rollout.output, "answer", "") or "")
    payload = parse_prediction_json(answer)
    return _score_payload_for_example(example, payload)


def _evaluate_split(
    *,
    finetune: Any,
    examples: list[RuleVQAExample],
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
    strict_reward_values: list[float] = []
    object_parse_count = 0
    parse_success_count = 0
    strict_parse_success_count = 0
    task_correct_count = 0
    strict_task_correct_count = 0
    rule_acc_values: list[float] = []
    strict_rule_acc_values: list[float] = []
    rule_f1_values: list[float] = []
    strict_rule_f1_values: list[float] = []
    reason_values: list[float] = []
    strict_reason_values: list[float] = []
    hallucinated_counts: list[int] = []
    strict_hallucinated_counts: list[int] = []
    positive_reward_values: list[float] = []
    positive_strict_reward_values: list[float] = []
    positive_rule_f1_values: list[float] = []
    positive_strict_rule_f1_values: list[float] = []
    no_violation_total = 0
    no_violation_correct = 0
    strict_no_violation_correct = 0

    predictions_handle = None
    if predictions_path is not None:
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        predictions_handle = predictions_path.open("w", encoding="utf-8")

    def record_outcome(example: RuleVQAExample, outcome: RuleVQAScoreOutcome, answer: str) -> None:
        nonlocal object_parse_count
        nonlocal parse_success_count
        nonlocal strict_parse_success_count
        nonlocal task_correct_count
        nonlocal strict_task_correct_count
        nonlocal no_violation_total
        nonlocal no_violation_correct
        nonlocal strict_no_violation_correct

        reward_values.append(float(outcome.reward))
        strict_reward_values.append(float(outcome.strict_reward))
        rule_acc_values.append(float(outcome.rule_set_accuracy))
        strict_rule_acc_values.append(float(outcome.strict_rule_set_accuracy))
        rule_f1_values.append(float(outcome.rule_set_f1))
        strict_rule_f1_values.append(float(outcome.strict_rule_set_f1))
        reason_values.append(float(outcome.reason_token_f1))
        strict_reason_values.append(float(outcome.strict_reason_token_f1))
        hallucinated_counts.append(int(outcome.hallucinated_rule_count))
        strict_hallucinated_counts.append(int(outcome.strict_hallucinated_rule_count))
        if example.violated_rules:
            positive_reward_values.append(float(outcome.reward))
            positive_strict_reward_values.append(float(outcome.strict_reward))
            positive_rule_f1_values.append(float(outcome.rule_set_f1))
            positive_strict_rule_f1_values.append(float(outcome.strict_rule_set_f1))
        if not example.violated_rules:
            no_violation_total += 1
            no_violation_correct += int(outcome.no_violation_correct)
            strict_no_violation_correct += int(outcome.strict_no_violation_correct)
        if outcome.json_object_parsed:
            object_parse_count += 1
        if outcome.parse_success:
            parse_success_count += 1
        if outcome.strict_parse_success:
            strict_parse_success_count += 1
        if outcome.task_correct:
            task_correct_count += 1
        if outcome.strict_task_correct:
            strict_task_correct_count += 1
        if predictions_handle is not None:
            predictions_handle.write(
                json.dumps(
                    {
                        "row_id": example.row_id,
                        "split": example.split,
                        "task_type": example.task_type,
                        "is_positive_example": bool(example.violated_rules),
                        "answer": answer,
                        "reward": outcome.reward,
                        "parse_success": outcome.parse_success,
                        "rule_set_accuracy": outcome.rule_set_accuracy,
                        "rule_set_f1": outcome.rule_set_f1,
                        "reason_token_f1": outcome.reason_token_f1,
                        "hallucinated_rule_count": outcome.hallucinated_rule_count,
                        "strict_reward": outcome.strict_reward,
                        "strict_parse_success": outcome.strict_parse_success,
                        "strict_rule_set_accuracy": outcome.strict_rule_set_accuracy,
                        "strict_rule_set_f1": outcome.strict_rule_set_f1,
                        "strict_reason_token_f1": outcome.strict_reason_token_f1,
                        "strict_hallucinated_rule_count": outcome.strict_hallucinated_rule_count,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    try:
        batch: list[RuleVQAExample] = []
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
                    answer = ""
                    if rollout is not None and getattr(rollout, "output", None) is not None:
                        answer = str(getattr(rollout.output, "answer", "") or "")
                    record_outcome(example, outcome, answer)
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
                    answer = ""
                    if rollout is not None and getattr(rollout, "output", None) is not None:
                        answer = str(getattr(rollout.output, "answer", "") or "")
                    record_outcome(example, outcome, answer)
    finally:
        if predictions_handle is not None:
            predictions_handle.close()

    total = len(reward_values)
    positive_total = len(positive_reward_values)
    return {
        "eval_samples": float(total),
        "eval_positive_samples": float(positive_total),
        "eval_reward_mean": fmean(reward_values) if reward_values else 0.0,
        "eval_positive_reward_mean": fmean(positive_reward_values) if positive_reward_values else 0.0,
        "eval_strict_reward_mean": fmean(strict_reward_values) if strict_reward_values else 0.0,
        "eval_positive_strict_reward_mean": fmean(positive_strict_reward_values) if positive_strict_reward_values else 0.0,
        "eval_json_object_rate": object_parse_count / max(1, total),
        "eval_json_parse_rate": parse_success_count / max(1, total),
        "eval_strict_json_parse_rate": strict_parse_success_count / max(1, total),
        "eval_rule_set_accuracy": fmean(rule_acc_values) if rule_acc_values else 0.0,
        "eval_strict_rule_set_accuracy": fmean(strict_rule_acc_values) if strict_rule_acc_values else 0.0,
        "eval_rule_set_f1": fmean(rule_f1_values) if rule_f1_values else 0.0,
        "eval_positive_rule_set_f1": fmean(positive_rule_f1_values) if positive_rule_f1_values else 0.0,
        "eval_strict_rule_set_f1": fmean(strict_rule_f1_values) if strict_rule_f1_values else 0.0,
        "eval_positive_strict_rule_set_f1": (
            fmean(positive_strict_rule_f1_values) if positive_strict_rule_f1_values else 0.0
        ),
        "eval_reason_token_f1": fmean(reason_values) if reason_values else 0.0,
        "eval_strict_reason_token_f1": fmean(strict_reason_values) if strict_reason_values else 0.0,
        "eval_no_violation_accuracy": no_violation_correct / max(1, no_violation_total),
        "eval_strict_no_violation_accuracy": strict_no_violation_correct / max(1, no_violation_total),
        "eval_task_correct_rate": task_correct_count / max(1, total),
        "eval_strict_task_correct_rate": strict_task_correct_count / max(1, total),
        "eval_hallucinated_rules_mean": fmean(float(value) for value in hallucinated_counts) if hallucinated_counts else 0.0,
        "eval_strict_hallucinated_rules_mean": (
            fmean(float(value) for value in strict_hallucinated_counts) if strict_hallucinated_counts else 0.0
        ),
    }


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    raw_argv = list(argv) if argv is not None else list(os.sys.argv[1:])
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    pre_args, _ = pre_parser.parse_known_args(raw_argv)
    config_path = resolve_config_path(pre_args.config, script_dir=Path(__file__).resolve().parent)
    config = load_json_config(config_path, default_path=DEFAULT_CONFIG_PATH)

    parser = argparse.ArgumentParser(description="ConstructionSite rule-VQA query RL trainer.")
    parser.add_argument("--config", default=str(config_path))
    parser.add_argument("--env-file", default=str(repo_relative(".env")))
    parser.add_argument("--api-key", default="")
    parser.add_argument("--api-key-env-var", default="MOONDREAM_API_KEY")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--dataset-source", choices=["local_jsonl"], default="local_jsonl")
    parser.add_argument("--dataset-dir", default=str(repo_relative("outputs", "construction_site_query_rule_vqa_v1")))
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
    parser.add_argument("--max-tokens", type=int, default=256)
    reasoning_group = parser.add_mutually_exclusive_group()
    reasoning_group.add_argument("--reasoning", dest="reasoning", action="store_true")
    reasoning_group.add_argument("--no-reasoning", dest="reasoning", action="store_false")
    parser.set_defaults(reasoning=False)
    off_policy_group = parser.add_mutually_exclusive_group()
    off_policy_group.add_argument("--off-policy", dest="off_policy", action="store_true")
    off_policy_group.add_argument("--no-off-policy", dest="off_policy", action="store_false")
    parser.set_defaults(off_policy=True)
    parser.add_argument("--off-policy-mix-ratio", type=float, default=0.5)
    parser.add_argument("--off-policy-buffer-size", type=int, default=4096)
    parser.add_argument("--off-policy-warmup-steps", type=int, default=10)
    parser.add_argument("--off-policy-min-buffer-groups", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--save-on-eval", dest="save_on_eval", action="store_true")
    parser.add_argument("--no-save-on-eval", dest="save_on_eval", action="store_false")
    parser.set_defaults(save_on_eval=True)
    async_eval_group = parser.add_mutually_exclusive_group()
    async_eval_group.add_argument("--async-checkpoint-eval", dest="async_checkpoint_eval", action="store_true")
    async_eval_group.add_argument("--no-async-checkpoint-eval", dest="async_checkpoint_eval", action="store_false")
    parser.set_defaults(async_checkpoint_eval=False)
    parser.add_argument(
        "--async-checkpoint-eval-dir",
        default=str(repo_relative("outputs", "async_checkpoint_eval")),
    )
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
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--eval-max-samples", type=int, default=500)
    parser.add_argument("--save-eval-predictions", dest="save_eval_predictions", action="store_true")
    parser.add_argument("--no-save-eval-predictions", dest="save_eval_predictions", action="store_false")
    parser.set_defaults(save_eval_predictions=True)
    parser.add_argument(
        "--eval-predictions-output-dir",
        default=str(repo_relative("outputs", "eval_predictions", "construction_site_query_rule_vqa")),
    )
    parser.add_argument("--best-metric", choices=BEST_METRIC_CHOICES, default="eval_rule_set_accuracy")
    parser.add_argument("--skip-final-eval", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--wandb-project", default="moondream-construction-site-query-rule-vqa-rl")
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
    args.async_checkpoint_eval_dir = str(
        query_common.resolve_path(
            args.async_checkpoint_eval_dir,
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
    if args.async_checkpoint_eval_max_inflight <= 0:
        raise ValueError("--async-checkpoint-eval-max-inflight must be > 0")
    if args.async_checkpoint_eval and not args.save_on_eval:
        raise ValueError("--async-checkpoint-eval requires --save-on-eval")


def _build_async_checkpoint_eval_command(
    *,
    args: argparse.Namespace,
    finetune_id: str,
    split_name: str,
    checkpoint_step: int,
    metrics_json_path: Path,
    predictions_jsonl_path: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        str((Path(__file__).resolve().parent / "benchmark_construction_site_query_rule_vqa.py").resolve()),
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
        str(float(args.eval_temperature)),
        "--top-p",
        str(float(args.eval_top_p)),
        "--max-tokens",
        str(int(args.max_tokens)),
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
        "--no-progress",
    ]
    cmd.append("--reasoning" if bool(args.reasoning) else "--no-reasoning")
    return cmd


def _ingest_async_checkpoint_eval_results(
    *,
    args: argparse.Namespace,
    run: Any,
    results: list[CheckpointEvalResult],
    best_metric_value: Optional[float],
    best_checkpoint_step: Optional[int],
    latest_checkpoint_step: Optional[int],
) -> tuple[Optional[float], Optional[int], Optional[int], int]:
    success_count = 0
    for result in results:
        step_for_log = int(result.metadata.get("step_for_log", result.checkpoint_step))
        if result.status != "succeeded" or result.metrics_payload is None:
            print(
                f"async checkpoint eval failed step={step_for_log} "
                f"checkpoint_step={result.checkpoint_step} log={result.stdout_log_path}"
            )
            continue
        success_count += 1
        metrics = result.metrics_payload
        metric_value = float(metrics.get(str(args.best_metric), 0.0))
        latest_checkpoint_step = int(result.checkpoint_step)
        run.summary["latest_checkpoint_step"] = int(result.checkpoint_step)
        run.summary["latest_async_eval_metric"] = metric_value
        run.summary["latest_async_eval_step"] = int(step_for_log)
        if best_metric_value is None or metric_value > best_metric_value:
            best_metric_value = metric_value
            best_checkpoint_step = int(result.checkpoint_step)
            run.summary["best_metric"] = metric_value
            run.summary["best_metric_name"] = str(args.best_metric)
            run.summary["best_metric_step"] = int(step_for_log)
            run.summary["best_checkpoint_step"] = int(result.checkpoint_step)
        print(
            f"async checkpoint eval completed step={step_for_log} checkpoint_step={result.checkpoint_step} "
            f"{args.best_metric}={metric_value:.4f}"
        )
    return best_metric_value, best_checkpoint_step, latest_checkpoint_step, success_count


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
        name=args.finetune_name or f"construction-site-query-rule-vqa-{_random_suffix()}",
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
    async_eval_jobs: list[DispatchHandle] = []
    async_eval_success_count = 0

    for global_step in range(int(args.resume_step), int(args.num_steps)):
        async_eval_jobs, completed_async_results = poll_checkpoint_eval_jobs(async_eval_jobs)
        (
            best_metric_value,
            best_checkpoint_step,
            latest_checkpoint_step,
            completed_success_count,
        ) = _ingest_async_checkpoint_eval_results(
            args=args,
            run=run,
            results=completed_async_results,
            best_metric_value=best_metric_value,
            best_checkpoint_step=best_checkpoint_step,
            latest_checkpoint_step=latest_checkpoint_step,
        )
        async_eval_success_count += int(completed_success_count)
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
        strict_reward_values: list[float] = []
        parse_success_count = 0
        strict_parse_success_count = 0
        for example, result in zip(active_examples, results):
            rewards: list[float] = []
            for rollout in result.rollouts:
                outcome = _score_rollout(rollout, example)
                rewards.append(float(outcome.reward))
                reward_values.append(float(outcome.reward))
                strict_reward_values.append(float(outcome.strict_reward))
                parse_success_count += int(outcome.parse_success)
                strict_parse_success_count += int(outcome.strict_parse_success)
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
            "strict_reward_mean": fmean(strict_reward_values) if strict_reward_values else 0.0,
            "train_json_parse_rate": parse_success_count / max(1, len(reward_values)),
            "train_strict_json_parse_rate": strict_parse_success_count / max(1, len(reward_values)),
            "on_policy_groups": float(len(train_groups) - off_policy_count),
            "off_policy_groups": float(off_policy_count),
            "off_policy_group_fraction": off_policy_count / max(1, len(train_groups)),
            "replay_buffer_size": float(len(replay_buffer)),
            "kl": float(train_out.kl or 0.0),
            "router_kl": float(train_out.router_kl or 0.0),
            "grad_norm": float(train_out.grad_norm or 0.0),
        }

        if args.eval_every > 0 and (global_step + 1) % int(args.eval_every) == 0:
            if args.async_checkpoint_eval:
                saved_step = query_common.save_checkpoint(
                    finetune=finetune,
                    context=f"periodic eval checkpoint step={global_step + 1}",
                )
                if saved_step is not None:
                    latest_checkpoint_step = int(saved_step)
                    run.summary["latest_checkpoint_step"] = int(saved_step)
                    job = dispatch_checkpoint_eval(
                        trainer="construction_site_query_rule_vqa",
                        finetune_id=str(finetune.finetune_id),
                        checkpoint_step=int(saved_step),
                        selection_metric=str(args.best_metric),
                        base_dir=str(args.async_checkpoint_eval_dir),
                        command_builder=lambda metrics_json_path, predictions_jsonl_path, _stdout_log_path: _build_async_checkpoint_eval_command(
                            args=args,
                            finetune_id=str(finetune.finetune_id),
                            split_name=str(args.val_split),
                            checkpoint_step=int(saved_step),
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
                            f"async checkpoint eval skipped step={global_step + 1} checkpoint_step={saved_step} "
                            f"reason=max_inflight"
                        )
                    else:
                        async_eval_jobs.append(job)
                        print(
                            f"async checkpoint eval dispatched step={global_step + 1} checkpoint_step={saved_step} "
                            f"job_dir={job.job_dir}"
                        )
            else:
                predictions_path = None
                if args.save_eval_predictions:
                    predictions_path = (
                        Path(args.eval_predictions_output_dir) / f"step{global_step + 1:06d}_validation.jsonl"
                    )
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
            f"strict_reward={train_metrics['strict_reward_mean']:.4f} "
            f"parse={train_metrics['train_json_parse_rate']:.4f}/{train_metrics['train_strict_json_parse_rate']:.4f} "
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
    if args.async_checkpoint_eval and bool(args.async_checkpoint_eval_drain_on_exit):
        completed_async_results = drain_checkpoint_eval_jobs(async_eval_jobs)
        (
            best_metric_value,
            best_checkpoint_step,
            latest_checkpoint_step,
            completed_success_count,
        ) = _ingest_async_checkpoint_eval_results(
            args=args,
            run=run,
            results=completed_async_results,
            best_metric_value=best_metric_value,
            best_checkpoint_step=best_checkpoint_step,
            latest_checkpoint_step=latest_checkpoint_step,
        )
        async_eval_success_count += int(completed_success_count)
    run.summary["async_checkpoint_eval_enabled"] = bool(args.async_checkpoint_eval)
    run.summary["async_checkpoint_eval_success_count"] = int(async_eval_success_count)
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
