from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Optional

from md_train_framework.datasets import create_dataset_adapter
from md_train_framework.runtime import QueryTrainer
from md_train_framework.samples import QuerySample, _resolve_image_source, _sample_id, _sample_meta


CANONICAL_TASK_TYPES: tuple[str, ...] = (
    "best_move",
    "has_winning_move",
    "turn_player",
    "winner",
    "is_game_over",
    "available_moves_count",
    "available_moves_list",
)
TASK_TYPE_ALIASES = {
    "is_terminal": "is_game_over",
    "legal_moves_count": "available_moves_count",
    "legal_moves_list": "available_moves_list",
}
ANSWER_KEY_ALIASES = {
    "is_game_over": {"is_game_over": "is_game_over", "is_terminal": "is_game_over"},
    "available_moves_count": {"available_move_count": "available_move_count", "legal_move_count": "available_move_count"},
    "available_moves_list": {"available_moves": "available_moves", "legal_moves": "available_moves"},
}


@dataclass(frozen=True)
class TTTScore:
    reward: float
    parse_success: bool
    task_correct: bool
    task_type: str
    best_move_canonical_correct: bool = False
    best_move_rank_reward: float = 0.0


def _parse_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "t", "yes", "y", "1", "on"}:
            return True
        if lowered in {"false", "f", "no", "n", "0", "off"}:
            return False
    return None


def _coerce_player(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    lowered = value.strip().lower()
    if lowered == "x":
        return "X"
    if lowered == "o":
        return "O"
    return None


def _coerce_winner(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    return {
        "x": "X",
        "o": "O",
        "draw": "draw",
        "in_progress": "in_progress",
        "inprogress": "in_progress",
    }.get(value.strip().lower().replace(" ", "_"))


def _normalize_task_type(task_type: str, *, allow_unknown: bool = False) -> str:
    normalized = TASK_TYPE_ALIASES.get(str(task_type).strip(), str(task_type).strip())
    if normalized in CANONICAL_TASK_TYPES:
        return normalized
    if allow_unknown:
        return normalized
    raise ValueError(f"unknown task_type: {task_type}")


def _normalize_answer_payload_for_task(task_type: str, payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    aliases = ANSWER_KEY_ALIASES.get(_normalize_task_type(task_type, allow_unknown=True))
    if not aliases:
        return payload
    out: dict[str, Any] = {}
    for old_key, canonical_key in aliases.items():
        if old_key in payload and canonical_key not in out:
            out[canonical_key] = payload[old_key]
    return out or payload


def _normalize_legal_moves(payload: Any) -> Optional[tuple[tuple[int, int], ...]]:
    if not isinstance(payload, list):
        return None
    moves: list[tuple[int, int]] = []
    for item in payload:
        if not isinstance(item, dict):
            return None
        row = _parse_int(item.get("row"))
        col = _parse_int(item.get("col"))
        if row is None or col is None or row < 1 or row > 3 or col < 1 or col > 3:
            return None
        moves.append((row, col))
    return tuple(moves)


def _normalize_non_best_answer(task_type: str, payload: Any) -> Optional[dict[str, Any]]:
    if not isinstance(payload, dict):
        return None
    task_type = _normalize_task_type(task_type, allow_unknown=True)
    payload = _normalize_answer_payload_for_task(task_type, payload)
    if task_type == "winner":
        winner = _coerce_winner(payload.get("winner"))
        return None if winner is None else {"winner": winner}
    if task_type == "is_game_over":
        value = _coerce_bool(payload.get("is_game_over"))
        return None if value is None else {"is_game_over": value}
    if task_type == "has_winning_move":
        value = _coerce_bool(payload.get("has_winning_move"))
        return None if value is None else {"has_winning_move": value}
    if task_type == "turn_player":
        player = _coerce_player(payload.get("player"))
        return None if player is None else {"player": player}
    if task_type == "available_moves_count":
        count = _parse_int(payload.get("available_move_count"))
        return None if count is None or count < 0 or count > 9 else {"available_move_count": count}
    if task_type == "available_moves_list":
        moves = _normalize_legal_moves(payload.get("available_moves"))
        return None if moves is None else {"available_moves": moves}
    return None


def _row_col_to_move(row: int, col: int) -> Optional[int]:
    if row < 1 or row > 3 or col < 1 or col > 3:
        return None
    return ((row - 1) * 3) + col


def _move_from_payload(payload: Any) -> Optional[int]:
    if not isinstance(payload, dict):
        return None
    if "move" in payload:
        move = _parse_int(payload.get("move"))
        return move if move is not None and 1 <= move <= 9 else None
    row = _parse_int(payload.get("row"))
    col = _parse_int(payload.get("col"))
    return None if row is None or col is None else _row_col_to_move(row, col)


def _parse_prediction_json(answer_text: str) -> Optional[dict[str, Any]]:
    if not isinstance(answer_text, str):
        return None
    text = answer_text.strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        pass
    depth = 0
    start: Optional[int] = None
    in_string = False
    escaped = False
    for index, char in enumerate(text):
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
            if depth == 0:
                start = index
            depth += 1
            continue
        if char == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    payload = json.loads(text[start : index + 1])
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    return payload
    return None


def _best_move_from_json(payload_json: str) -> Optional[int]:
    try:
        return _move_from_payload(json.loads(payload_json))
    except json.JSONDecodeError:
        return None


def _move_set_from_json(payload_json: str) -> frozenset[int]:
    try:
        payload = json.loads(payload_json)
    except json.JSONDecodeError:
        return frozenset()
    if not isinstance(payload, list):
        return frozenset()
    return frozenset(move for move in (_move_from_payload(item) for item in payload) if move is not None)


def _scores_by_move_from_json(payload_json: str) -> tuple[tuple[int, int, int], ...]:
    try:
        payload = json.loads(payload_json)
    except json.JSONDecodeError:
        return tuple()
    if not isinstance(payload, dict):
        return tuple()
    rows: list[tuple[int, int, int]] = []
    for raw_move, raw_score in payload.items():
        move = _parse_int(raw_move)
        if move is None or move < 1 or move > 9 or not isinstance(raw_score, dict):
            continue
        value = _parse_int(raw_score.get("value"))
        depth = _parse_int(raw_score.get("depth"))
        if value is not None and depth is not None:
            rows.append((move, value, depth))
    rows.sort(key=lambda item: item[0])
    return tuple(rows)


def _best_move_rank_key(value: int, depth: int) -> tuple[int, int]:
    return (value, -depth) if value == 1 else (value, depth)


def _ranked_best_move_reward(move: int, *, scores_by_move: dict[int, tuple[int, int]]) -> float:
    predicted = scores_by_move.get(move)
    if predicted is None:
        return 0.0
    if len(scores_by_move) <= 1:
        return 1.0
    predicted_key = _best_move_rank_key(int(predicted[0]), int(predicted[1]))
    better_count = sum(
        1
        for value, depth in scores_by_move.values()
        if _best_move_rank_key(int(value), int(depth)) > predicted_key
    )
    return max(0.0, min(1.0, 1.0 - (better_count / float(len(scores_by_move) - 1))))


def _extract_reasoning_text(answer_text: str) -> str:
    text = str(answer_text or "").strip()
    if not text:
        return ""
    match = re.search(r"Reason:\s*(.+?)(?:\nFinal:|\Z)", text, flags=re.DOTALL)
    return str(match.group(1)).strip() if match else ""


class TTTQueryTrainer(QueryTrainer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.task_sampling_weights = {
            task: 1.0 for task in CANONICAL_TASK_TYPES
        }
        self.task_sampling_weights.update(
            {
                _normalize_task_type(str(task), allow_unknown=True): float(value)
                for task, value in dict(self._override("task_sampling_weights", {}) or {}).items()
            }
        )
        self.max_tokens_by_task = {
            _normalize_task_type(str(task), allow_unknown=True): int(value)
            for task, value in dict(self._override("max_tokens_by_task", {}) or {}).items()
        }
        self.best_move_optimal_reward = float(self._override("best_move_optimal_reward", 0.7))
        self.best_move_reward_mode = str(self._override("best_move_reward_mode", "ranked")).strip() or "ranked"
        self.best_move_wrong_rank_scale = float(self._override("best_move_wrong_rank_scale", 0.5))
        self.train_samples_by_task: dict[str, list[QuerySample]] = {}
        for sample in self.train_samples:
            task_type = _normalize_task_type(str(sample.meta.get("task_type") or ""), allow_unknown=True)
            self.train_samples_by_task.setdefault(task_type, []).append(sample)

    def _load_samples(self):
        adapter = create_dataset_adapter(self.config)
        return (
            self._load_split(adapter, self.config.dataset.train_split),
            self._load_split(adapter, self.config.dataset.val_split),
            self._load_split(adapter, self.config.dataset.test_split),
        )

    def _load_split(self, adapter: Any, split_name: str) -> list[QuerySample]:
        samples: list[QuerySample] = []
        for index, row in enumerate(adapter.iter_split(split_name), start=1):
            if not isinstance(row, dict):
                continue
            raw_task_type = str(row.get("task_type") or "").strip()
            try:
                task_type = _normalize_task_type(raw_task_type)
            except ValueError:
                continue
            question = str(row.get("question") or row.get("query") or row.get("prompt") or "").strip()
            if not question:
                continue
            answer_payload = row.get("final_answer_json", row.get("answer"))
            if isinstance(answer_payload, str):
                answer_text = answer_payload
            elif isinstance(answer_payload, (dict, list)):
                answer_text = json.dumps(answer_payload, sort_keys=True)
            else:
                continue
            image_url, image_ref = _resolve_image_source(
                self.config,
                row,
                split=split_name,
                dataset_index=index - 1,
            )
            samples.append(
                QuerySample(
                    sample_id=_sample_id(row, split=split_name, index=index),
                    split=split_name,
                    question=question,
                    answer=answer_text,
                    image_url=image_url,
                    image_ref=image_ref,
                    reasoning=_extract_reasoning_text(str(row.get("answer_text") or row.get("reasoning") or "")),
                    meta={
                        **_sample_meta(row),
                        "task_type": task_type,
                        "best_move_canonical": _best_move_from_json(str(row.get("best_move_canonical_json", "null"))),
                        "best_move_optimal_set": _move_set_from_json(str(row.get("best_move_optimal_set_json", "[]"))),
                        "best_move_scores": _scores_by_move_from_json(str(row.get("scores_by_move_json", ""))),
                    },
                )
            )
        return samples

    def _split_samples(self, split_name: str):
        samples = super()._split_samples(split_name)
        active = {
            task
            for task, weight in self.task_sampling_weights.items()
            if float(weight) > 0.0
        }
        return [sample for sample in samples if str(sample.meta.get("task_type") or "") in active]

    def _sample_train_batch(self, batch_size: int):
        active_tasks = [
            task
            for task, samples in sorted(self.train_samples_by_task.items())
            if samples and float(self.task_sampling_weights.get(task, 1.0)) > 0.0
        ]
        if not active_tasks:
            return super()._sample_train_batch(batch_size)
        weights = [float(self.task_sampling_weights.get(task, 1.0)) for task in active_tasks]
        batch: list[QuerySample] = []
        for _ in range(max(1, int(batch_size))):
            task_name = self.rng.choices(active_tasks, weights=weights, k=1)[0]
            batch.append(self.rng.choice(self.train_samples_by_task[task_name]))
        return batch

    def _sample_to_request(self, sample: QuerySample, *, phase, for_eval: bool):
        default_tokens = self.config.eval.max_tokens if for_eval else (phase.max_tokens if phase and phase.max_tokens is not None else 128)
        task_type = str(sample.meta.get("task_type") or "")
        max_tokens = int(self.max_tokens_by_task.get(task_type, default_tokens or 128))
        reasoning_flag = False if for_eval else bool(phase.reasoning) if phase is not None else False
        from tuna_sdk import QueryRequest, QuerySettings

        return QueryRequest(
            question=sample.question,
            image_url=sample.image_url,
            spatial_refs=None,
            reasoning=reasoning_flag,
            settings=QuerySettings(
                temperature=self.config.eval.temperature if for_eval else float(phase.temperature or 1.0),
                top_p=self.config.eval.top_p if for_eval else float(phase.top_p or 1.0),
                max_tokens=max_tokens,
            ),
        )

    def _sample_to_sft_group(self, sample: QuerySample, *, phase):
        task_type = str(sample.meta.get("task_type") or "")
        optimal_set = sample.meta.get("best_move_optimal_set")
        if task_type == "best_move" and isinstance(optimal_set, frozenset) and len(optimal_set) != 1:
            return None
        if bool(phase.reasoning) and not str(sample.reasoning or "").strip():
            return None
        return super()._sample_to_sft_group(sample, phase=phase)

    def _score_rollout(self, sample: QuerySample, output: Any) -> TTTScore:
        task_type = str(sample.meta.get("task_type") or "")
        pred_payload = _parse_prediction_json(str(getattr(output, "answer", "") or ""))
        if pred_payload is None:
            return TTTScore(reward=0.0, parse_success=False, task_correct=False, task_type=task_type)
        if task_type == "best_move":
            move = _move_from_payload(pred_payload)
            canonical_move = _parse_int(sample.meta.get("best_move_canonical"))
            optimal_set = sample.meta.get("best_move_optimal_set")
            optimal_moves = optimal_set if isinstance(optimal_set, frozenset) else frozenset()
            if move is None:
                return TTTScore(reward=0.0, parse_success=False, task_correct=False, task_type=task_type)
            set_correct = move in optimal_moves
            canonical_correct = move == canonical_move
            scores_by_move = {
                scored_move: (value, depth)
                for scored_move, value, depth in sample.meta.get("best_move_scores", tuple())
            }
            ranked_reward = _ranked_best_move_reward(move, scores_by_move=scores_by_move) if scores_by_move else 0.0
            if self.best_move_reward_mode == "binary":
                reward = 1.0 if set_correct else 0.0
            elif self.best_move_reward_mode == "hybrid_strict":
                reward = 1.0 if set_correct else float(self.best_move_wrong_rank_scale) * ranked_reward
            else:
                reward = (
                    ranked_reward
                    if scores_by_move
                    else (1.0 if canonical_correct else self.best_move_optimal_reward if set_correct else 0.0)
                )
            return TTTScore(
                reward=max(0.0, min(1.0, float(reward))),
                parse_success=True,
                task_correct=set_correct,
                task_type=task_type,
                best_move_canonical_correct=canonical_correct,
                best_move_rank_reward=float(ranked_reward),
            )
        target_payload = _parse_prediction_json(sample.answer)
        gt_norm = _normalize_non_best_answer(task_type, target_payload)
        pred_norm = _normalize_non_best_answer(task_type, pred_payload)
        exact = bool(gt_norm is not None and pred_norm is not None and gt_norm == pred_norm)
        return TTTScore(
            reward=1.0 if exact else 0.0,
            parse_success=pred_norm is not None,
            task_correct=exact,
            task_type=task_type,
        )

    def _aggregate_scores(self, scores: list[TTTScore]) -> dict[str, Any]:
        if not scores:
            return {
                "reward_mean": 0.0,
                "eval_reward_mean": 0.0,
                "accuracy": 0.0,
                "eval_accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "eval_balanced_accuracy": 0.0,
                "macro_f1": 0.0,
                "micro_f1": 0.0,
                "json_parse_rate": 0.0,
                "eval_json_parse_rate": 0.0,
                "eval_best_move_canonical_accuracy": 0.0,
                "count": 0,
            }
        per_task_total: Counter[str] = Counter()
        per_task_correct: Counter[str] = Counter()
        per_task_reward: Counter[str] = Counter()
        reward_total = 0.0
        parse_successes = 0
        correct_total = 0
        best_move_total = 0
        best_move_canonical_correct = 0
        best_move_rank_reward_total = 0.0
        for score in scores:
            reward_total += float(score.reward)
            parse_successes += 1 if score.parse_success else 0
            correct_total += 1 if score.task_correct else 0
            per_task_total[score.task_type] += 1
            per_task_correct[score.task_type] += 1 if score.task_correct else 0
            per_task_reward[score.task_type] += float(score.reward)
            if score.task_type == "best_move":
                best_move_total += 1
                best_move_canonical_correct += 1 if score.best_move_canonical_correct else 0
                best_move_rank_reward_total += float(score.best_move_rank_reward)
        balanced_accuracy = (
            sum(per_task_correct[task] / max(1, per_task_total[task]) for task in sorted(per_task_total)) / max(1, len(per_task_total))
        )
        payload: dict[str, Any] = {
            "reward_mean": reward_total / len(scores),
            "eval_reward_mean": reward_total / len(scores),
            "accuracy": correct_total / len(scores),
            "eval_accuracy": correct_total / len(scores),
            "balanced_accuracy": balanced_accuracy,
            "eval_balanced_accuracy": balanced_accuracy,
            "macro_f1": balanced_accuracy,
            "micro_f1": correct_total / len(scores),
            "json_parse_rate": parse_successes / len(scores),
            "eval_json_parse_rate": parse_successes / len(scores),
            "eval_best_move_canonical_accuracy": best_move_canonical_correct / max(1, best_move_total),
            "eval_best_move_reward_mean": per_task_reward["best_move"] / max(1, best_move_total),
            "eval_best_move_rank_reward_mean": best_move_rank_reward_total / max(1, best_move_total),
            "eval_best_move_accuracy": per_task_correct["best_move"] / max(1, best_move_total),
            "count": len(scores),
        }
        for task_name in sorted(per_task_total):
            payload[f"eval_task_accuracy_{task_name}"] = per_task_correct[task_name] / max(1, per_task_total[task_name])
            payload[f"eval_task_count_{task_name}"] = float(per_task_total[task_name])
            payload[f"eval_task_reward_mean_{task_name}"] = per_task_reward[task_name] / max(1, per_task_total[task_name])
        return payload

    def _prediction_record(self, sample: QuerySample, output: Any, score: TTTScore) -> dict[str, Any]:
        return {
            "sample_id": sample.sample_id,
            "task_type": sample.meta.get("task_type"),
            "question": sample.question,
            "prediction": getattr(output, "answer", ""),
            "target": sample.answer,
            "metrics": {
                "reward": score.reward,
                "parse_success": score.parse_success,
                "task_correct": score.task_correct,
                "best_move_canonical_correct": score.best_move_canonical_correct,
                "best_move_rank_reward": score.best_move_rank_reward,
            },
        }

    def _profile_checkpoint_guard_decision(
        self,
        *,
        phase,
        global_step: int,
        checkpoint_event,
        baseline_metrics: dict[str, Any],
        best_metrics: dict[str, Any],
    ) -> Optional[dict[str, Any]]:
        del best_metrics
        if str(getattr(phase, "mode", "")).strip().lower() != "rl":
            return None
        reward_key = "eval_reward_mean"
        baseline_reward = float(baseline_metrics.get(reward_key, baseline_metrics.get("reward_mean", 0.0)) or 0.0)
        current_reward = float(checkpoint_event.metrics.get(reward_key, checkpoint_event.metrics.get("reward_mean", 0.0)) or 0.0)
        min_reward_gain = max(0.0, float(self._override("exactness_guard_min_reward_gain", 0.0)))
        reward_gain = current_reward - baseline_reward
        if reward_gain < min_reward_gain:
            return None

        guarded_metrics = (
            (
                "eval_accuracy",
                "overall accuracy",
                max(0.0, float(self._override("exactness_guard_max_accuracy_drop", 0.0))),
            ),
            (
                "eval_best_move_accuracy",
                "best-move accuracy",
                max(0.0, float(self._override("exactness_guard_max_best_move_drop", 0.0))),
            ),
            (
                "eval_best_move_canonical_accuracy",
                "canonical best-move accuracy",
                max(0.0, float(self._override("exactness_guard_max_best_move_canonical_drop", 0.0))),
            ),
        )
        regressions: list[tuple[str, str, float, float, float]] = []
        for metric_key, label, max_drop in guarded_metrics:
            if max_drop <= 0.0:
                continue
            baseline_value = float(baseline_metrics.get(metric_key, 0.0) or 0.0)
            current_value = float(checkpoint_event.metrics.get(metric_key, baseline_value) or 0.0)
            if baseline_value <= 0.0:
                continue
            drop = baseline_value - current_value
            if drop > max_drop:
                regressions.append((metric_key, label, baseline_value, current_value, drop))
        if not regressions:
            return None

        details = ", ".join(
            f"{label} {baseline_value:.4f}->{current_value:.4f}"
            for _, label, baseline_value, current_value, _ in regressions
        )
        payload: dict[str, Any] = {
            "guard": "ttt_exactness_tradeoff",
            "message": (
                f"ttt exactness guard triggered at step {global_step}: "
                f"eval_reward_mean improved by {reward_gain:.4f} while {details}"
            ),
            "baseline_eval_reward_mean": baseline_reward,
            "eval_reward_mean": current_reward,
            "reward_gain": reward_gain,
        }
        for metric_key, _, baseline_value, current_value, drop in regressions:
            payload[f"baseline_{metric_key}"] = baseline_value
            payload[metric_key] = current_value
            payload[f"{metric_key}_drop"] = drop
        return payload
