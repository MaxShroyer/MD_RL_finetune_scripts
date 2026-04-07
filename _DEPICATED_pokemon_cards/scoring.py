"""Shared scoring helpers for PokemonCards train and benchmark flows."""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from _DEPICATED_pokemon_cards.task_schema import (
    canonicalize_rationale_text,
    normalize_answer_for_task,
    normalize_task_type,
    parse_rationale_text,
    summary_support_from_payload,
)

CARD_CORE_REWARD_WEIGHTS: dict[str, float] = {
    "name": 0.25,
    "hp": 0.15,
    "set_name": 0.20,
    "stage": 0.15,
    "pokemon_types": 0.15,
    "rarity": 0.05,
    "evolves_from": 0.05,
}

ANSWER_REWARD_WEIGHT = 0.8
RATIONALE_REWARD_WEIGHT = 0.2
SUMMARY_FACT_REWARD_WEIGHT = 0.8
SUMMARY_LEXICAL_REWARD_WEIGHT = 0.2
SUMMARY_STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "card",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "its",
    "of",
    "or",
    "pokemon",
    "the",
    "this",
    "to",
    "with",
}
SUMMARY_TYPE_VOCAB = (
    "grass",
    "fire",
    "water",
    "lightning",
    "psychic",
    "fighting",
    "darkness",
    "metal",
    "dragon",
    "fairy",
    "colorless",
)


def json_object_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    n = len(text)
    for start in range(n):
        if text[start] != "{":
            continue
        depth = 0
        in_string = False
        escaped = False
        for idx in range(start, n):
            char = text[idx]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
                continue
            if char == "{":
                depth += 1
                continue
            if char == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : idx + 1])
                    break
                if depth < 0:
                    break
    return candidates


def parse_prediction_json(answer_text: str) -> Optional[dict[str, Any]]:
    text = str(answer_text or "").strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    for candidate in json_object_candidates(text):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def set_f1(pred: set[Any], gt: set[Any]) -> float:
    if not pred and not gt:
        return 1.0
    if not pred or not gt:
        return 0.0
    intersection = len(pred.intersection(gt))
    precision = float(intersection) / float(len(pred))
    recall = float(intersection) / float(len(gt))
    if precision + recall == 0.0:
        return 0.0
    return (2.0 * precision * recall) / (precision + recall)


def count_similarity(pred_count: int, gt_count: int) -> float:
    return max(0.0, 1.0 - (abs(int(pred_count) - int(gt_count)) / float(max(1, gt_count))))


def _summary_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text or "").lower())


def _summary_content_tokens(text: str) -> set[str]:
    return {token for token in _summary_tokens(text) if token not in SUMMARY_STOPWORDS}


def _phrase_token_coverage(summary_text: str, phrase: str) -> float:
    phrase_tokens = _summary_content_tokens(phrase)
    if not phrase_tokens:
        return 0.0
    summary_tokens = _summary_content_tokens(summary_text)
    return float(len(summary_tokens.intersection(phrase_tokens))) / float(len(phrase_tokens))


def _summary_number_mentioned(summary_text: str, value: int) -> float:
    return 1.0 if str(int(value)) in set(_summary_tokens(summary_text)) else 0.0


def _extract_type_mentions(summary_text: str) -> set[str]:
    summary_tokens = set(_summary_tokens(summary_text))
    return {token for token in SUMMARY_TYPE_VOCAB if token in summary_tokens}


def _attack_coverage(summary_text: str, attack_names: list[str]) -> float:
    expected = [str(name or "").strip() for name in attack_names if str(name or "").strip()]
    if not expected:
        return 1.0
    matches = 0
    for attack_name in expected:
        if _phrase_token_coverage(summary_text, attack_name) >= 1.0:
            matches += 1
    return float(matches) / float(len(expected))


def _summary_lexical_score(pred_summary: str, expected_summary: str) -> float:
    return set_f1(_summary_content_tokens(pred_summary), _summary_content_tokens(expected_summary))


def _summary_fact_score(pred_summary: str, expected_support: dict[str, Any]) -> float:
    weighted = 0.0
    total_weight = 0.0

    name = str(expected_support.get("name", "") or "").strip()
    if name:
        total_weight += 0.30
        weighted += 0.30 * _phrase_token_coverage(pred_summary, name)

    hp = expected_support.get("hp")
    if hp is not None:
        total_weight += 0.15
        weighted += 0.15 * _summary_number_mentioned(pred_summary, int(hp))

    set_name = str(expected_support.get("set_name", "") or "").strip()
    if set_name:
        total_weight += 0.20
        weighted += 0.20 * _phrase_token_coverage(pred_summary, set_name)

    stage = str(expected_support.get("stage", "") or "").strip()
    if stage:
        total_weight += 0.10
        weighted += 0.10 * _phrase_token_coverage(pred_summary, stage)

    expected_types = {str(item).lower() for item in expected_support.get("pokemon_types", []) if str(item).strip()}
    if expected_types:
        total_weight += 0.10
        weighted += 0.10 * set_f1(_extract_type_mentions(pred_summary), expected_types)

    expected_attacks = [str(item) for item in expected_support.get("attack_names", []) if str(item).strip()]
    if expected_attacks:
        total_weight += 0.15
        weighted += 0.15 * _attack_coverage(pred_summary, expected_attacks)

    if total_weight <= 0.0:
        return 0.0
    return weighted / total_weight


def summary_reward_from_texts(
    pred_summary: str,
    expected_summary: str,
    *,
    expected_support: Optional[dict[str, Any]] = None,
) -> float:
    lexical_score = _summary_lexical_score(pred_summary, expected_summary)
    if not expected_support:
        return lexical_score
    fact_score = _summary_fact_score(pred_summary, expected_support)
    return (
        (SUMMARY_FACT_REWARD_WEIGHT * fact_score)
        + (SUMMARY_LEXICAL_REWARD_WEIGHT * lexical_score)
    )


def answer_reward_for_task(
    task_type: str,
    pred_payload: Any,
    expected_payload: dict[str, Any],
    *,
    expected_metadata: Optional[dict[str, Any]] = None,
) -> float:
    canonical_task = normalize_task_type(task_type)
    pred = normalize_answer_for_task(canonical_task, pred_payload)
    expected = normalize_answer_for_task(canonical_task, expected_payload)
    if pred is None or expected is None:
        return 0.0

    if canonical_task == "card_identity":
        score = 0.0
        score += 1.0 if pred["name"].lower() == expected["name"].lower() else 0.0
        score += 1.0 if int(pred["hp"]) == int(expected["hp"]) else 0.0
        score += 1.0 if pred["set_name"].lower() == expected["set_name"].lower() else 0.0
        return score / 3.0

    if canonical_task == "card_core":
        reward = 0.0
        reward += CARD_CORE_REWARD_WEIGHTS["name"] * (
            1.0 if pred["name"].lower() == expected["name"].lower() else 0.0
        )
        reward += CARD_CORE_REWARD_WEIGHTS["hp"] * (
            1.0 if int(pred["hp"]) == int(expected["hp"]) else 0.0
        )
        reward += CARD_CORE_REWARD_WEIGHTS["set_name"] * (
            1.0 if pred["set_name"].lower() == expected["set_name"].lower() else 0.0
        )
        reward += CARD_CORE_REWARD_WEIGHTS["stage"] * (
            1.0 if (pred.get("stage") or "").lower() == (expected.get("stage") or "").lower() else 0.0
        )
        reward += CARD_CORE_REWARD_WEIGHTS["pokemon_types"] * set_f1(
            {item.lower() for item in pred.get("pokemon_types", [])},
            {item.lower() for item in expected.get("pokemon_types", [])},
        )
        reward += CARD_CORE_REWARD_WEIGHTS["rarity"] * (
            1.0 if (pred.get("rarity") or "").lower() == (expected.get("rarity") or "").lower() else 0.0
        )
        reward += CARD_CORE_REWARD_WEIGHTS["evolves_from"] * (
            1.0
            if (pred.get("evolves_from") or "").lower() == (expected.get("evolves_from") or "").lower()
            else 0.0
        )
        return max(0.0, min(1.0, reward))

    if canonical_task == "attack_overview":
        attack_f1 = set_f1(
            {item.lower() for item in pred.get("attack_names", [])},
            {item.lower() for item in expected.get("attack_names", [])},
        )
        count_score = count_similarity(int(pred.get("attack_count", 0)), int(expected.get("attack_count", 0)))
        return max(0.0, min(1.0, (0.7 * attack_f1) + (0.3 * count_score)))

    if canonical_task == "card_summary":
        expected_support = summary_support_from_payload(expected_metadata or {})
        if expected_support is None:
            expected_support = summary_support_from_payload(expected_payload)
        return max(
            0.0,
            min(
                1.0,
                summary_reward_from_texts(
                    str(pred.get("summary", "") or ""),
                    str(expected.get("summary", "") or ""),
                    expected_support=expected_support,
                ),
            ),
        )

    return 0.0


def rationale_reward_from_texts(task_type: str, rationale_text: str, expected_rationale_text: str) -> float:
    canonical_task = normalize_task_type(task_type)
    expected_slots = parse_rationale_text(canonical_task, canonicalize_rationale_text(canonical_task, expected_rationale_text))
    if not expected_slots:
        return 0.0

    pred_slots = parse_rationale_text(canonical_task, rationale_text)
    if not pred_slots:
        return 0.0

    scores: list[float] = []
    for key, expected_value in expected_slots.items():
        predicted_value = pred_slots.get(key, "none")
        if key in {"types", "attacks"}:
            pred_set = {item for item in predicted_value.split(",") if item and item != "none"}
            exp_set = {item for item in expected_value.split(",") if item and item != "none"}
            scores.append(set_f1(pred_set, exp_set))
        elif key in {"hp", "attack_count"}:
            try:
                pred_count = int(predicted_value)
                exp_count = int(expected_value)
            except ValueError:
                scores.append(0.0)
            else:
                scores.append(count_similarity(pred_count, exp_count))
        else:
            scores.append(1.0 if predicted_value == expected_value else 0.0)
    return float(sum(scores) / float(max(1, len(scores))))


def combined_reward(answer_reward: float, rationale_reward: float, *, use_reasoning_reward: bool) -> float:
    if not use_reasoning_reward:
        return float(answer_reward)
    return (ANSWER_REWARD_WEIGHT * float(answer_reward)) + (RATIONALE_REWARD_WEIGHT * float(rationale_reward))
