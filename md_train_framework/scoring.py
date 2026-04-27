from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from md_train_framework.rewards import get_reward_preset
from tuna_sdk import DetectAnnotation, DetectOutput, PointAnnotation, PointOutput, QueryOutput


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


@dataclass(frozen=True)
class DetectScore:
    reward: float
    precision: float
    recall: float
    f1: float
    miou: float
    tp: int
    fp: int
    fn: int
    pred_count: int
    gt_count: int
    empty_prediction: bool
    positive_empty_prediction: bool


@dataclass(frozen=True)
class PointScore:
    reward: float
    precision: float
    recall: float
    f1: float
    mean_distance: float
    tp: int
    fp: int
    fn: int


@dataclass(frozen=True)
class QueryScore:
    reward: float
    exact_match: float
    token_f1: float
    parse_success: float
    json_f1: float
    accuracy: float
    target_label: str
    predicted_label: str


def score_detect(
    *,
    ground_truth: Iterable[DetectAnnotation],
    output: DetectOutput,
    reward_preset_id: str,
    iou_threshold: float = 0.5,
    fn_penalty_weight: float = 0.0,
    fn_penalty_exponent: float = 1.0,
    fp_penalty_weight: float = 0.0,
    fp_penalty_exponent: float = 1.0,
    empty_refusal_penalty: float = 0.0,
    allow_negative_reward: bool = False,
) -> DetectScore:
    preset_family = _reward_family(reward_preset_id)
    gt_boxes = list(ground_truth)
    pred_boxes = list(output.objects)
    matches = _match_boxes(gt_boxes, pred_boxes, iou_threshold=iou_threshold)
    tp = len(matches)
    fp = max(0, len(pred_boxes) - tp)
    fn = max(0, len(gt_boxes) - tp)
    precision = _safe_div(tp, tp + fp, default=1.0)
    recall = _safe_div(tp, tp + fn, default=1.0)
    f1 = _safe_div(2 * precision * recall, precision + recall, default=1.0 if tp == fp == fn == 0 else 0.0)
    if not pred_boxes and not gt_boxes:
        miou = 1.0
    elif not pred_boxes or not gt_boxes:
        miou = 0.0
    else:
        miou = sum(match[2] for match in matches) / max(1, max(len(pred_boxes), len(gt_boxes)))
    reward = miou if preset_family == "detect_miou" else f1
    fn_rate = _safe_div(fn, len(gt_boxes), default=0.0)
    fp_rate = _safe_div(fp, len(pred_boxes), default=0.0)
    reward -= _exp_rate_penalty(fn_rate, weight=fn_penalty_weight, exponent=fn_penalty_exponent)
    reward -= _exp_rate_penalty(fp_rate, weight=fp_penalty_weight, exponent=fp_penalty_exponent)
    if gt_boxes and not pred_boxes:
        reward -= max(0.0, float(empty_refusal_penalty))
    if not allow_negative_reward:
        reward = max(0.0, reward)
    reward = min(1.0, reward)
    return DetectScore(
        reward=reward,
        precision=precision,
        recall=recall,
        f1=f1,
        miou=miou,
        tp=tp,
        fp=fp,
        fn=fn,
        pred_count=len(pred_boxes),
        gt_count=len(gt_boxes),
        empty_prediction=not pred_boxes,
        positive_empty_prediction=bool(gt_boxes) and not pred_boxes,
    )


def aggregate_detect(scores: Iterable[DetectScore]) -> dict[str, Any]:
    score_list = list(scores)
    if not score_list:
        return {
            "reward_mean": 0.0,
            "eval_precision": 0.0,
            "eval_recall": 0.0,
            "eval_f1": 0.0,
            "eval_f1_macro": 0.0,
            "eval_miou": 0.0,
            "eval_tp": 0,
            "eval_fp": 0,
            "eval_fn": 0,
            "eval_empty_prediction_rate": 0.0,
            "eval_positive_empty_prediction_rate": 0.0,
            "count": 0,
        }
    tp = sum(item.tp for item in score_list)
    fp = sum(item.fp for item in score_list)
    fn = sum(item.fn for item in score_list)
    precision = _safe_div(tp, tp + fp, default=1.0)
    recall = _safe_div(tp, tp + fn, default=1.0)
    f1 = _safe_div(2 * precision * recall, precision + recall, default=1.0 if tp == fp == fn == 0 else 0.0)
    return {
        "reward_mean": sum(item.reward for item in score_list) / len(score_list),
        "eval_precision": precision,
        "eval_recall": recall,
        "eval_f1": f1,
        "eval_f1_macro": sum(item.f1 for item in score_list) / len(score_list),
        "eval_miou": sum(item.miou for item in score_list) / len(score_list),
        "eval_tp": tp,
        "eval_fp": fp,
        "eval_fn": fn,
        "eval_empty_prediction_rate": sum(1.0 for item in score_list if item.empty_prediction) / len(score_list),
        "eval_positive_empty_prediction_rate": (
            sum(1.0 for item in score_list if item.positive_empty_prediction)
            / max(1, sum(1 for item in score_list if item.gt_count > 0))
        ),
        "count": len(score_list),
    }


def score_point(
    *,
    ground_truth_points: Iterable[PointAnnotation],
    ground_truth_boxes: Iterable[DetectAnnotation],
    output: PointOutput,
    reward_preset_id: str,
    distance_threshold: float = 0.08,
) -> PointScore:
    preset_family = _reward_family(reward_preset_id)
    gt_points = list(ground_truth_points)
    if not gt_points:
        gt_points = [
            PointAnnotation(
                x=(box.x_min + box.x_max) / 2.0,
                y=(box.y_min + box.y_max) / 2.0,
                width=max(0.0, box.x_max - box.x_min),
                height=max(0.0, box.y_max - box.y_min),
            )
            for box in ground_truth_boxes
        ]
    pred_points = list(output.points)
    matches = _match_points(gt_points, pred_points, ground_truth_boxes=list(ground_truth_boxes), distance_threshold=distance_threshold)
    tp = len(matches)
    fp = max(0, len(pred_points) - tp)
    fn = max(0, len(gt_points) - tp)
    precision = _safe_div(tp, tp + fp, default=1.0)
    recall = _safe_div(tp, tp + fn, default=1.0)
    f1 = _safe_div(2 * precision * recall, precision + recall, default=1.0 if tp == fp == fn == 0 else 0.0)
    mean_distance = sum(match[2] for match in matches) / max(1, tp)
    if preset_family == "point_recall_first":
        reward = 0.7 * recall + 0.3 * f1
    else:
        reward = f1
    return PointScore(
        reward=reward,
        precision=precision,
        recall=recall,
        f1=f1,
        mean_distance=mean_distance,
        tp=tp,
        fp=fp,
        fn=fn,
    )


def aggregate_point(scores: Iterable[PointScore]) -> dict[str, Any]:
    score_list = list(scores)
    if not score_list:
        return {
            "reward_mean": 0.0,
            "eval_precision": 0.0,
            "eval_recall": 0.0,
            "eval_f1": 0.0,
            "eval_tp": 0,
            "eval_fp": 0,
            "eval_fn": 0,
            "eval_mean_distance": 0.0,
            "count": 0,
        }
    tp = sum(item.tp for item in score_list)
    fp = sum(item.fp for item in score_list)
    fn = sum(item.fn for item in score_list)
    precision = _safe_div(tp, tp + fp, default=1.0)
    recall = _safe_div(tp, tp + fn, default=1.0)
    f1 = _safe_div(2 * precision * recall, precision + recall, default=1.0 if tp == fp == fn == 0 else 0.0)
    return {
        "reward_mean": sum(item.reward for item in score_list) / len(score_list),
        "eval_precision": precision,
        "eval_recall": recall,
        "eval_f1": f1,
        "eval_tp": tp,
        "eval_fp": fp,
        "eval_fn": fn,
        "eval_mean_distance": sum(item.mean_distance for item in score_list) / len(score_list),
        "count": len(score_list),
    }


def score_query(*, target_text: str, output: QueryOutput, reward_preset_id: str) -> QueryScore:
    preset_family = _reward_family(reward_preset_id)
    predicted = str(output.answer or "")
    target = str(target_text or "")
    exact = 1.0 if _normalize_text(predicted) == _normalize_text(target) else 0.0
    token_f1 = _token_f1(target, predicted)
    parsed_target = _maybe_json(target)
    parsed_pred = _maybe_json(predicted)
    parse_success = 1.0 if parsed_pred is not None else 0.0
    json_f1 = _json_f1(parsed_target, parsed_pred) if parsed_target is not None and parsed_pred is not None else 0.0
    if preset_family == "query_token_f1":
        reward = token_f1
    elif preset_family in {"query_soft_hybrid", "query_judge_hybrid"}:
        reward = (0.5 * exact) + (0.25 * token_f1) + (0.25 * json_f1)
    else:
        reward = max(exact, token_f1)
    normalized_target = _normalize_text(target)
    normalized_pred = _normalize_text(predicted)
    return QueryScore(
        reward=reward,
        exact_match=exact,
        token_f1=token_f1,
        parse_success=parse_success,
        json_f1=json_f1,
        accuracy=exact,
        target_label=normalized_target,
        predicted_label=normalized_pred,
    )


def aggregate_query(scores: Iterable[QueryScore]) -> dict[str, Any]:
    score_list = list(scores)
    if not score_list:
        return {
            "reward_mean": 0.0,
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "macro_f1": 0.0,
            "micro_f1": 0.0,
            "token_f1": 0.0,
            "json_parse_rate": 0.0,
            "json_f1": 0.0,
            "count": 0,
        }
    labels = sorted({item.target_label for item in score_list if item.target_label})
    per_label_f1: list[float] = []
    total_tp = total_fp = total_fn = 0
    for label in labels:
        tp = sum(1 for item in score_list if item.target_label == label and item.predicted_label == label)
        fp = sum(1 for item in score_list if item.target_label != label and item.predicted_label == label)
        fn = sum(1 for item in score_list if item.target_label == label and item.predicted_label != label)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        precision = _safe_div(tp, tp + fp, default=1.0)
        recall = _safe_div(tp, tp + fn, default=1.0)
        per_label_f1.append(_safe_div(2 * precision * recall, precision + recall, default=1.0 if tp == fp == fn == 0 else 0.0))
    balanced_accuracy = 0.0
    if labels:
        balanced_accuracy = sum(
            _safe_div(
                sum(1 for item in score_list if item.target_label == label and item.predicted_label == label),
                sum(1 for item in score_list if item.target_label == label),
                default=0.0,
            )
            for label in labels
        ) / len(labels)
    micro_precision = _safe_div(total_tp, total_tp + total_fp, default=1.0)
    micro_recall = _safe_div(total_tp, total_tp + total_fn, default=1.0)
    return {
        "reward_mean": sum(item.reward for item in score_list) / len(score_list),
        "accuracy": sum(item.accuracy for item in score_list) / len(score_list),
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": sum(per_label_f1) / len(per_label_f1) if per_label_f1 else 0.0,
        "micro_f1": _safe_div(2 * micro_precision * micro_recall, micro_precision + micro_recall, default=1.0),
        "token_f1": sum(item.token_f1 for item in score_list) / len(score_list),
        "json_parse_rate": sum(item.parse_success for item in score_list) / len(score_list),
        "json_f1": sum(item.json_f1 for item in score_list) / len(score_list),
        "count": len(score_list),
    }


def _match_boxes(
    gt_boxes: list[DetectAnnotation],
    pred_boxes: list[DetectAnnotation],
    *,
    iou_threshold: float,
) -> list[tuple[int, int, float]]:
    matches: list[tuple[int, int, float]] = []
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    candidates: list[tuple[float, int, int]] = []
    for gt_idx, gt in enumerate(gt_boxes):
        for pred_idx, pred in enumerate(pred_boxes):
            iou = _box_iou(gt, pred)
            if iou >= iou_threshold:
                candidates.append((iou, gt_idx, pred_idx))
    for iou, gt_idx, pred_idx in sorted(candidates, reverse=True):
        if gt_idx in used_gt or pred_idx in used_pred:
            continue
        used_gt.add(gt_idx)
        used_pred.add(pred_idx)
        matches.append((gt_idx, pred_idx, iou))
    return matches


def _match_points(
    gt_points: list[PointAnnotation],
    pred_points: list[PointAnnotation],
    *,
    ground_truth_boxes: list[DetectAnnotation],
    distance_threshold: float,
) -> list[tuple[int, int, float]]:
    matches: list[tuple[int, int, float]] = []
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    candidates: list[tuple[float, int, int]] = []
    for gt_idx, gt in enumerate(gt_points):
        box = ground_truth_boxes[gt_idx] if gt_idx < len(ground_truth_boxes) else None
        for pred_idx, pred in enumerate(pred_points):
            inside_box = False
            if box is not None:
                inside_box = box.x_min <= pred.x <= box.x_max and box.y_min <= pred.y <= box.y_max
            distance = math.dist((gt.x, gt.y), (pred.x, pred.y))
            if inside_box or distance <= distance_threshold:
                candidates.append((distance, gt_idx, pred_idx))
    for distance, gt_idx, pred_idx in sorted(candidates):
        if gt_idx in used_gt or pred_idx in used_pred:
            continue
        used_gt.add(gt_idx)
        used_pred.add(pred_idx)
        matches.append((gt_idx, pred_idx, distance))
    return matches


def _box_iou(left: DetectAnnotation, right: DetectAnnotation) -> float:
    x_min = max(left.x_min, right.x_min)
    y_min = max(left.y_min, right.y_min)
    x_max = min(left.x_max, right.x_max)
    y_max = min(left.y_max, right.y_max)
    inter_w = max(0.0, x_max - x_min)
    inter_h = max(0.0, y_max - y_min)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    left_area = max(0.0, left.x_max - left.x_min) * max(0.0, left.y_max - left.y_min)
    right_area = max(0.0, right.x_max - right.x_min) * max(0.0, right.y_max - right.y_min)
    union = max(1e-8, left_area + right_area - inter)
    return inter / union


def _reward_family(reward_preset_id: str) -> str:
    try:
        return str(get_reward_preset(reward_preset_id).family)
    except KeyError:
        return str(reward_preset_id or "").strip()


def _safe_div(num: float, denom: float, *, default: float) -> float:
    if denom == 0:
        return default
    return num / denom


def _exp_rate_penalty(rate: float, *, weight: float, exponent: float) -> float:
    clipped_rate = max(0.0, min(1.0, float(rate)))
    clipped_weight = max(0.0, float(weight))
    clipped_exponent = max(0.0, float(exponent))
    if clipped_rate == 0.0 or clipped_weight == 0.0:
        return 0.0
    if clipped_exponent <= 1e-6:
        return clipped_weight * clipped_rate
    numerator = math.exp(clipped_exponent * clipped_rate) - 1.0
    denominator = math.exp(clipped_exponent) - 1.0
    if denominator <= 0.0:
        return clipped_weight * clipped_rate
    return clipped_weight * (numerator / denominator)


def _normalize_text(text: str) -> str:
    normalized = " ".join(TOKEN_RE.findall(str(text).lower()))
    return normalized.strip()


def _token_f1(target: str, predicted: str) -> float:
    target_tokens = TOKEN_RE.findall(_normalize_text(target))
    predicted_tokens = TOKEN_RE.findall(_normalize_text(predicted))
    if not target_tokens and not predicted_tokens:
        return 1.0
    if not target_tokens or not predicted_tokens:
        return 0.0
    target_counts: dict[str, int] = {}
    predicted_counts: dict[str, int] = {}
    for token in target_tokens:
        target_counts[token] = target_counts.get(token, 0) + 1
    for token in predicted_tokens:
        predicted_counts[token] = predicted_counts.get(token, 0) + 1
    overlap = sum(min(target_counts.get(token, 0), predicted_counts.get(token, 0)) for token in target_counts)
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(target_tokens)
    return _safe_div(2 * precision * recall, precision + recall, default=0.0)


def _maybe_json(text: str) -> Optional[Any]:
    text = str(text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _json_f1(target: Any, predicted: Any) -> float:
    target_items = set(_flatten_json(target))
    predicted_items = set(_flatten_json(predicted))
    if not target_items and not predicted_items:
        return 1.0
    if not target_items or not predicted_items:
        return 0.0
    overlap = len(target_items & predicted_items)
    precision = overlap / len(predicted_items)
    recall = overlap / len(target_items)
    return _safe_div(2 * precision * recall, precision + recall, default=0.0)


def _flatten_json(value: Any, *, prefix: str = "") -> list[str]:
    if isinstance(value, dict):
        items: list[str] = []
        for key in sorted(value):
            items.extend(_flatten_json(value[key], prefix=f"{prefix}.{key}" if prefix else str(key)))
        return items
    if isinstance(value, list):
        items: list[str] = []
        for index, item in enumerate(value):
            items.extend(_flatten_json(item, prefix=f"{prefix}[{index}]"))
        return items
    return [f"{prefix}={value}"]
