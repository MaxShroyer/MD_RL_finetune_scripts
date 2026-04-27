from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Any, Optional

from tuna_sdk import DetectAnnotation, PointAnnotation

from md_train_framework.datasets import create_dataset_adapter
from md_train_framework.runtime import PointTrainer
from md_train_framework.scoring import PointScore
from md_train_framework.samples import PointSample, _parse_boxes, _resolve_image_source, _sample_id, _sample_meta
from md_train_framework.utils import slugify


def _box_center(box: DetectAnnotation) -> PointAnnotation:
    return PointAnnotation(
        x=(box.x_min + box.x_max) / 2.0,
        y=(box.y_min + box.y_max) / 2.0,
        width=max(0.0, box.x_max - box.x_min),
        height=max(0.0, box.y_max - box.y_min),
    )


def _prompt_for_class(class_name: str, *, style: str) -> str:
    normalized = str(style or "").strip().lower()
    if normalized == "class_name":
        return str(class_name)
    return f"{class_name} icon or icons"


def _raw_answer_boxes(row: dict[str, Any]) -> list[dict[str, Any]]:
    raw = row.get("answer_boxes")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def _class_name_from_box(item: dict[str, Any]) -> str:
    for key in ("class_name", "source_class_name", "label", "object_name"):
        value = str(item.get(key) or "").strip()
        if value:
            return value
    return ""


def _group_boxes_by_class(row: dict[str, Any]) -> dict[str, list[DetectAnnotation]]:
    grouped: dict[str, list[DetectAnnotation]] = defaultdict(list)
    for item in _raw_answer_boxes(row):
        class_name = _class_name_from_box(item)
        if not class_name:
            continue
        boxes = _parse_boxes([item])
        if boxes:
            grouped[class_name].extend(boxes)
    return dict(grouped)


def _point_in_box(point: PointAnnotation, box: DetectAnnotation) -> bool:
    return (box.x_min <= point.x <= box.x_max) and (box.y_min <= point.y <= box.y_max)


def _match_points_to_boxes(points: tuple[PointAnnotation, ...], boxes: tuple[DetectAnnotation, ...]) -> list[tuple[int, int, float]]:
    matches: list[tuple[int, int, float]] = []
    used_points: set[int] = set()
    for box_index, box in enumerate(boxes):
        best_point_index = -1
        best_distance: Optional[float] = None
        center_x = (box.x_min + box.x_max) / 2.0
        center_y = (box.y_min + box.y_max) / 2.0
        for point_index, point in enumerate(points):
            if point_index in used_points or not _point_in_box(point, box):
                continue
            distance = math.hypot(point.x - center_x, point.y - center_y)
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_point_index = point_index
        if best_point_index >= 0:
            used_points.add(best_point_index)
            matches.append((box_index, best_point_index, float(best_distance or 0.0)))
    return matches


def _weighted_point_f1(tp: int, fp: int, fn: int, *, fn_penalty_exponent: float, fp_penalty_exponent: float) -> float:
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0
    weighted_fp = float(fp) ** float(fp_penalty_exponent)
    weighted_fn = float(fn) ** float(fn_penalty_exponent)
    denom = (2.0 * float(tp)) + weighted_fp + weighted_fn
    if denom <= 0.0:
        return 0.0
    return (2.0 * float(tp)) / denom


class PandidPointTrainer(PointTrainer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._positive_train_samples = [sample for sample in self.train_samples if isinstance(sample, PointSample) and sample.points]
        self._negative_train_samples = [sample for sample in self.train_samples if isinstance(sample, PointSample) and not sample.points]
        self.recall_gate_pass: Optional[bool] = None
        self.recall_gate_eval_step: Optional[int] = None
        self.recall_gate_eval_tp: Optional[float] = None
        self.recall_gate_min_tp: Optional[float] = None

    def _load_samples(self):
        adapter = create_dataset_adapter(self.config)
        raw_by_split = {
            self.config.dataset.train_split: [row for row in adapter.iter_split(self.config.dataset.train_split) if isinstance(row, dict)],
            self.config.dataset.val_split: [row for row in adapter.iter_split(self.config.dataset.val_split) if isinstance(row, dict)],
            self.config.dataset.test_split: [row for row in adapter.iter_split(self.config.dataset.test_split) if isinstance(row, dict)],
        }
        class_catalog = sorted(
            {
                class_name
                for rows in raw_by_split.values()
                for row in rows
                for class_name in _group_boxes_by_class(row)
            }
        )
        self.class_catalog = class_catalog
        return (
            self._expand_rows(raw_by_split[self.config.dataset.train_split], split_name=self.config.dataset.train_split),
            self._expand_rows(raw_by_split[self.config.dataset.val_split], split_name=self.config.dataset.val_split),
            self._expand_rows(raw_by_split[self.config.dataset.test_split], split_name=self.config.dataset.test_split),
        )

    def _expand_rows(self, rows: list[dict[str, Any]], *, split_name: str) -> list[PointSample]:
        prompt_style = str(self._override("point_prompt_style", "class_name"))
        neg_prompts_per_empty = max(0, int(self._override("neg_prompts_per_empty", 1)))
        neg_prompts_per_nonempty = max(0, int(self._override("neg_prompts_per_nonempty", 0)))
        expanded: list[PointSample] = []
        for index, row in enumerate(rows, start=1):
            image_url, image_ref = _resolve_image_source(
                self.config,
                row,
                split=split_name,
                dataset_index=index - 1,
            )
            if not image_url and image_ref is None:
                continue
            row_id = _sample_id(row, split=split_name, index=index)
            grouped = _group_boxes_by_class(row)
            present_classes = set(grouped)
            meta_base = _sample_meta(row)
            for class_name, boxes in sorted(grouped.items()):
                expanded.append(
                    PointSample(
                        sample_id=f"{row_id}-pos-{slugify(class_name)}",
                        split=split_name,
                        object_name=_prompt_for_class(class_name, style=prompt_style),
                        image_url=image_url or "",
                        image_ref=image_ref,
                        points=tuple(_box_center(box) for box in boxes),
                        boxes=tuple(boxes),
                        meta={**meta_base, "task_kind": "positive", "class_name": class_name},
                    )
                )
            neg_limit = neg_prompts_per_empty if not present_classes else neg_prompts_per_nonempty
            absent_classes = [class_name for class_name in self.class_catalog if class_name not in present_classes]
            if neg_limit > 0 and absent_classes:
                if len(absent_classes) > neg_limit:
                    absent_classes = sorted(self.rng.sample(absent_classes, k=neg_limit))
                for class_name in absent_classes:
                    expanded.append(
                        PointSample(
                            sample_id=f"{row_id}-neg-{slugify(class_name)}",
                            split=split_name,
                            object_name=_prompt_for_class(class_name, style=prompt_style),
                            image_url=image_url or "",
                            image_ref=image_ref,
                            points=tuple(),
                            boxes=tuple(),
                            meta={**meta_base, "task_kind": "negative", "class_name": class_name},
                        )
                    )
        return expanded

    def _sample_train_batch(self, batch_size: int):
        if not self.train_samples:
            raise ValueError("training split is empty")
        if not self._positive_train_samples or not self._negative_train_samples:
            return super()._sample_train_batch(batch_size)
        pos_task_prob = float(self._override("pos_task_prob", 0.95))
        batch: list[PointSample] = []
        for _ in range(max(1, int(batch_size))):
            if self.rng.random() < pos_task_prob:
                batch.append(self.rng.choice(self._positive_train_samples))
            else:
                batch.append(self.rng.choice(self._negative_train_samples))
        return batch

    def _sample_to_sft_group(self, sample: PointSample, *, phase):
        if not sample.points:
            return None
        return super()._sample_to_sft_group(sample, phase=phase)

    def _score_rollout(self, sample: PointSample, output: Any) -> PointScore:
        fn_penalty_exponent = max(1.0, float(self._override("fn_penalty_exponent", 1.0)))
        fp_penalty_exponent = max(1.0, float(self._override("fp_penalty_exponent", 1.0)))
        neg_reward_weight = max(0.0, min(1.0, float(self._override("neg_reward_weight", 1.0))))
        pred_points = tuple(getattr(output, "points", []) or [])
        gt_boxes = tuple(sample.boxes)
        if gt_boxes:
            matches = _match_points_to_boxes(pred_points, gt_boxes)
            tp = len(matches)
            fp = max(0, len(pred_points) - tp)
            fn = max(0, len(gt_boxes) - tp)
            precision = 1.0 if tp == fp == 0 else (float(tp) / float(tp + fp) if (tp + fp) > 0 else 1.0)
            recall = 1.0 if tp == fn == 0 else (float(tp) / float(tp + fn) if (tp + fn) > 0 else 1.0)
            f1 = _weighted_point_f1(
                tp,
                fp,
                fn,
                fn_penalty_exponent=fn_penalty_exponent,
                fp_penalty_exponent=fp_penalty_exponent,
            )
            mean_distance = sum(match[2] for match in matches) / max(1, tp)
            reward = (0.7 * recall) + (0.3 * f1) if bool(self._override("use_recall_first_preset", False)) else f1
            return PointScore(
                reward=float(max(0.0, min(1.0, reward))),
                precision=float(precision),
                recall=float(recall),
                f1=float(f1),
                mean_distance=float(mean_distance),
                tp=int(tp),
                fp=int(fp),
                fn=int(fn),
            )
        base = super()._score_rollout(sample, output)
        return PointScore(
            reward=float(base.reward) * neg_reward_weight,
            precision=float(base.precision),
            recall=float(base.recall),
            f1=float(base.f1),
            mean_distance=float(base.mean_distance),
            tp=int(base.tp),
            fp=int(base.fp),
            fn=int(base.fn),
        )

    def _off_policy_groups(self, *, phase, rl_group_count: int):
        current_mean = float(self._current_rl_reward_stats.get("reward_mean", 0.0))
        current_std = float(self._current_rl_reward_stats.get("reward_std", 0.0))
        current_max = float(self._current_rl_reward_stats.get("reward_max", 0.0))
        max_reward = float(self._override("off_policy_max_reward", 0.15))
        std_thresh = float(self._override("off_policy_std_thresh", 0.02))
        low_max_reward = current_max < max_reward
        low_mean_reward = current_mean < max_reward
        should_inject = low_max_reward or (low_mean_reward and current_std < std_thresh)
        if not should_inject:
            return []
        min_reward = float(self._override("off_policy_min_reward", 0.15))
        reward_scale = float(self._override("off_policy_reward_scale", 2.0))
        replay_groups = super()._off_policy_groups(phase=phase, rl_group_count=rl_group_count)
        adjusted = []
        for group in replay_groups:
            reward_anchor = max(group.rewards) if group.rewards else 0.0
            reward_floor = max(min_reward, min(1.0, reward_scale * float(reward_anchor)))
            adjusted.append(
                type(group).from_rl(
                    request=group.request,
                    rollouts=group.rollouts,
                    rewards=[max(float(reward_floor), float(item)) for item in group.rewards],
                )
            )
        return adjusted

    def _profile_checkpoint_guard_decision(
        self,
        *,
        phase,
        global_step: int,
        checkpoint_event,
        baseline_metrics: dict[str, Any],
        best_metrics: dict[str, Any],
    ):
        del phase, best_metrics
        recall_gate_step = max(0, int(self._override("recall_gate_step", 40)))
        if self.recall_gate_pass is not None or global_step < recall_gate_step:
            return None
        baseline_tp = float(baseline_metrics.get("eval_tp", 0.0) or 0.0)
        if baseline_tp <= 0.0:
            return None
        drop_threshold = max(0.0, min(0.99, float(self._override("recall_drop_threshold", 0.25))))
        current_tp = float(checkpoint_event.metrics.get("eval_tp", 0.0) or 0.0)
        min_tp = baseline_tp * (1.0 - drop_threshold)
        self.recall_gate_eval_step = int(global_step)
        self.recall_gate_eval_tp = current_tp
        self.recall_gate_min_tp = min_tp
        self.recall_gate_pass = bool(current_tp >= min_tp)
        if self.recall_gate_pass:
            return None
        return {
            "guard": "pandid_recall_gate",
            "message": (
                f"pandid recall gate triggered at step {global_step}: "
                f"eval_tp={current_tp:.1f} fell below minimum {min_tp:.1f}"
            ),
            "recall_gate_step": recall_gate_step,
            "recall_gate_eval_tp": current_tp,
            "recall_gate_min_tp": min_tp,
        }

    def _summary_extras(
        self,
        *,
        baseline_metrics: dict[str, Any],
        best_metrics: dict[str, Any],
        latest_metrics: dict[str, Any],
        global_step: int,
    ) -> dict[str, Any]:
        del latest_metrics, global_step
        baseline_f1 = float(baseline_metrics.get("eval_f1", 0.0) or 0.0)
        best_f1 = float(best_metrics.get("eval_f1", 0.0) or 0.0)
        f1_target_value = baseline_f1 + max(0.0, float(self._override("f1_improvement_target", 0.0)))
        f1_target_pass = bool(best_f1 >= f1_target_value) if baseline_metrics else None
        return {
            "f1_target_value": f1_target_value,
            "f1_target_pass": f1_target_pass,
            "f1_target_evaluated": f1_target_pass is not None,
            "recall_gate_evaluated": self.recall_gate_pass is not None,
            "recall_gate_pass": self.recall_gate_pass,
            "recall_gate_eval_step": self.recall_gate_eval_step,
            "recall_gate_eval_tp": self.recall_gate_eval_tp,
            "recall_gate_min_tp": self.recall_gate_min_tp,
        }
