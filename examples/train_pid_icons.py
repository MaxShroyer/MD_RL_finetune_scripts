"""Standalone PI&D icon finetuning example for maxs-m87/pandid_dataset_v2.

The same script supports both `detect` and `point` skills and now includes the
main training controls used in the stronger local runs:
- lightweight image augmentation
- weighted rewards
- GT-anchor off-policy injection
- positive / negative task balancing

Requires:
  pip install datasets pillow numpy scipy wandb
"""

from __future__ import annotations

import argparse
import base64
import io
import importlib
import json
import os
import random
import string
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, List, Optional

import numpy as np
from datasets import load_dataset
from PIL import Image, ImageEnhance
from scipy.optimize import linear_sum_assignment

if TYPE_CHECKING:
    from tuna_sdk import DetectAnnotation, PointAnnotation, Rollout

_DetectAnnotation: Any = None
_DetectOutput: Any = None
_DetectRequest: Any = None
_DetectSettings: Any = None
_PointAnnotation: Any = None
_PointOutput: Any = None
_PointRequest: Any = None
_PointSettings: Any = None
_Rollout: Any = None
_TrainStepGroup: Any = None
_TunaClient: Any = None

DATASET_NAME = "maxs-m87/pandid_dataset_v2"


def _import_wandb():
    try:
        return importlib.import_module("wandb")
    except ImportError as exc:
        raise RuntimeError("wandb is required to run training. Install it with `pip install wandb`.") from exc


def _require_tuna_sdk() -> None:
    global _DetectAnnotation, _DetectOutput, _DetectRequest, _DetectSettings
    global _PointAnnotation, _PointOutput, _PointRequest, _PointSettings
    global _Rollout, _TrainStepGroup, _TunaClient
    if _TunaClient is not None:
        return
    try:
        module = importlib.import_module("tuna_sdk")
    except ImportError as exc:
        raise RuntimeError(
            "tuna_sdk is required to run this example. Install the package with `pip install tuna-sdk` "
            "or `pip install -e .` from the SDK repo."
        ) from exc
    _DetectAnnotation = module.DetectAnnotation
    _DetectOutput = module.DetectOutput
    _DetectRequest = module.DetectRequest
    _DetectSettings = module.DetectSettings
    _PointAnnotation = module.PointAnnotation
    _PointOutput = module.PointOutput
    _PointRequest = module.PointRequest
    _PointSettings = module.PointSettings
    _Rollout = module.Rollout
    _TrainStepGroup = module.TrainStepGroup
    _TunaClient = module.TunaClient


@dataclass(frozen=True)
class ClassBox:
    class_name: str
    box: DetectAnnotation


@dataclass(frozen=True)
class BaseSample:
    image: Image.Image
    image_url: str
    boxes: list[ClassBox]


@dataclass(frozen=True)
class TaskSample:
    image: Image.Image
    image_url: str
    prompt: str
    boxes: list[DetectAnnotation]


def _to_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _random_suffix(length: int = 6) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


def _normalize_class_name(value: object) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _parse_answer_boxes(answer_boxes: object, *, width: int, height: int) -> list[ClassBox]:
    raw = answer_boxes
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        try:
            raw = json.loads(text)
        except json.JSONDecodeError:
            return []
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return []

    boxes: list[ClassBox] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        class_name = _normalize_class_name(item.get("class_name") or item.get("source_class_name"))
        if not class_name:
            continue
        try:
            x_min = float(item.get("x_min"))
            y_min = float(item.get("y_min"))
            x_max = float(item.get("x_max"))
            y_max = float(item.get("y_max"))
        except (TypeError, ValueError):
            continue

        if max(abs(x_min), abs(y_min), abs(x_max), abs(y_max)) > 1.5:
            if width <= 0 or height <= 0:
                continue
            x_min /= width
            y_min /= height
            x_max /= width
            y_max /= height

        x_min = max(0.0, min(1.0, x_min))
        y_min = max(0.0, min(1.0, y_min))
        x_max = max(0.0, min(1.0, x_max))
        y_max = max(0.0, min(1.0, y_max))
        if x_max <= x_min or y_max <= y_min:
            continue
        boxes.append(
            ClassBox(
                class_name=class_name,
                box=_DetectAnnotation(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max),
            )
        )
    return boxes


def _to_base_sample(row: dict[str, Any]) -> Optional[BaseSample]:
    image = row.get("image")
    if image is None:
        return None
    image = image.convert("RGB")
    width, height = image.size
    boxes = _parse_answer_boxes(row.get("answer_boxes"), width=width, height=height)
    return BaseSample(image=image, image_url=_to_data_url(image), boxes=boxes)


def _prompt_for_class(class_name: str, *, style: str) -> str:
    if style == "class_name":
        return class_name
    return f"{class_name} icon or icons"


def _group_boxes_by_class(boxes: list[ClassBox]) -> dict[str, list[DetectAnnotation]]:
    grouped: dict[str, list[DetectAnnotation]] = {}
    for item in boxes:
        grouped.setdefault(item.class_name, []).append(item.box)
    return grouped


def _tasks_from_base_sample(
    sample: BaseSample,
    *,
    all_class_names: list[str],
    rng: random.Random,
    neg_prompts_per_empty: int,
    neg_prompts_per_nonempty: int,
    point_prompt_style: str,
) -> list[TaskSample]:
    tasks: list[TaskSample] = []
    grouped = _group_boxes_by_class(sample.boxes)
    present_names = set(grouped)

    for class_name, boxes in grouped.items():
        tasks.append(
            TaskSample(
                image=sample.image,
                image_url=sample.image_url,
                prompt=_prompt_for_class(class_name, style=point_prompt_style),
                boxes=boxes,
            )
        )

    if grouped:
        absent = [name for name in all_class_names if name not in present_names]
        if absent and neg_prompts_per_nonempty > 0:
            picks = rng.sample(absent, k=min(neg_prompts_per_nonempty, len(absent)))
            for class_name in picks:
                tasks.append(
                    TaskSample(
                        image=sample.image,
                        image_url=sample.image_url,
                        prompt=_prompt_for_class(class_name, style=point_prompt_style),
                        boxes=[],
                    )
                )
        return tasks

    if neg_prompts_per_empty > 0 and all_class_names:
        picks = rng.sample(all_class_names, k=min(neg_prompts_per_empty, len(all_class_names)))
        for class_name in picks:
            tasks.append(
                TaskSample(
                    image=sample.image,
                    image_url=sample.image_url,
                    prompt=_prompt_for_class(class_name, style=point_prompt_style),
                    boxes=[],
                )
            )
    return tasks


def _iter_base_rows(
    *,
    dataset_name: str,
    split: str,
    token: Optional[str],
    seed: int,
    buffer_size: int,
) -> Iterable[BaseSample]:
    epoch = 0
    while True:
        ds = load_dataset(dataset_name, split=split, streaming=True, token=token)
        if seed:
            ds = ds.shuffle(seed=seed + epoch, buffer_size=buffer_size)
        for row in ds:
            sample = _to_base_sample(row)
            if sample is not None:
                yield sample
        epoch += 1


def _discover_class_names(
    *,
    dataset_name: str,
    split: str,
    token: Optional[str],
) -> list[str]:
    names: set[str] = set()
    ds = load_dataset(dataset_name, split=split, streaming=True, token=token)
    for row in ds:
        sample = _to_base_sample(row)
        if sample is None:
            continue
        for item in sample.boxes:
            names.add(item.class_name)
    return sorted(names)


def _horizontal_flip_class_boxes(boxes: list[ClassBox]) -> list[ClassBox]:
    return [
        ClassBox(
            class_name=item.class_name,
            box=_DetectAnnotation(
                x_min=1.0 - item.box.x_max,
                y_min=item.box.y_min,
                x_max=1.0 - item.box.x_min,
                y_max=item.box.y_max,
            ),
        )
        for item in boxes
    ]


def _random_crop_base_sample(sample: BaseSample, rng: random.Random) -> BaseSample:
    crop_w = rng.uniform(0.6, 1.0)
    crop_h = rng.uniform(0.6, 1.0)
    offset_x = rng.uniform(0.0, 1.0 - crop_w)
    offset_y = rng.uniform(0.0, 1.0 - crop_h)

    width, height = sample.image.size
    left = int(round(offset_x * width))
    top = int(round(offset_y * height))
    right = max(left + 1, int(round((offset_x + crop_w) * width)))
    bottom = max(top + 1, int(round((offset_y + crop_h) * height)))
    cropped_image = sample.image.crop((left, top, right, bottom))

    cropped_boxes: list[ClassBox] = []
    for item in sample.boxes:
        box = item.box
        x_min = max(box.x_min, offset_x)
        y_min = max(box.y_min, offset_y)
        x_max = min(box.x_max, offset_x + crop_w)
        y_max = min(box.y_max, offset_y + crop_h)
        if x_max <= x_min or y_max <= y_min:
            continue
        cropped_boxes.append(
            ClassBox(
                class_name=item.class_name,
                box=_DetectAnnotation(
                    x_min=(x_min - offset_x) / crop_w,
                    y_min=(y_min - offset_y) / crop_h,
                    x_max=(x_max - offset_x) / crop_w,
                    y_max=(y_max - offset_y) / crop_h,
                ),
            )
        )

    if sample.boxes and not cropped_boxes:
        return sample
    return BaseSample(image=cropped_image, image_url=_to_data_url(cropped_image), boxes=cropped_boxes)


def _color_jitter(image: Image.Image, rng: random.Random) -> Image.Image:
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.75, 1.25))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.75, 1.25))
    image = ImageEnhance.Color(image).enhance(rng.uniform(0.75, 1.25))
    return image


def _add_noise(image: Image.Image, rng: random.Random) -> Image.Image:
    arr = np.asarray(image, dtype=np.float32)
    std = rng.uniform(3.0, 12.0)
    noise = np.random.normal(0.0, std, size=arr.shape)
    arr = np.clip(arr + noise, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(arr)


def _augment_base_sample(sample: BaseSample, *, rng: random.Random, augment_prob: float) -> BaseSample:
    if augment_prob <= 0.0 or rng.random() >= augment_prob:
        return sample

    out = sample
    if rng.random() < 0.5:
        out = _random_crop_base_sample(out, rng)
    if rng.random() < 0.5:
        flipped = out.image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        out = BaseSample(image=flipped, image_url=_to_data_url(flipped), boxes=_horizontal_flip_class_boxes(out.boxes))
    if rng.random() < 0.5:
        jittered = _color_jitter(out.image, rng)
        out = BaseSample(image=jittered, image_url=_to_data_url(jittered), boxes=list(out.boxes))
    if rng.random() < 0.3:
        noised = _add_noise(out.image, rng)
        out = BaseSample(image=noised, image_url=_to_data_url(noised), boxes=list(out.boxes))
    return out


def _choose_train_task(tasks: list[TaskSample], *, rng: random.Random, pos_task_prob: float) -> Optional[TaskSample]:
    if not tasks:
        return None
    positive = [task for task in tasks if task.boxes]
    negative = [task for task in tasks if not task.boxes]
    if positive and negative:
        source = positive if rng.random() < pos_task_prob else negative
        return rng.choice(source)
    if positive:
        return rng.choice(positive)
    return rng.choice(negative)


def _iter_task_samples(
    *,
    dataset_name: str,
    split: str,
    token: Optional[str],
    seed: int,
    buffer_size: int,
    all_class_names: list[str],
    neg_prompts_per_empty: int,
    neg_prompts_per_nonempty: int,
    point_prompt_style: str,
    pos_task_prob: float,
    augment_prob: float,
) -> Iterable[TaskSample]:
    rng = random.Random(seed)
    for sample in _iter_base_rows(
        dataset_name=dataset_name,
        split=split,
        token=token,
        seed=seed,
        buffer_size=buffer_size,
    ):
        working_sample = _augment_base_sample(sample, rng=rng, augment_prob=augment_prob)
        tasks = _tasks_from_base_sample(
            working_sample,
            all_class_names=all_class_names,
            rng=rng,
            neg_prompts_per_empty=neg_prompts_per_empty,
            neg_prompts_per_nonempty=neg_prompts_per_nonempty,
            point_prompt_style=point_prompt_style,
        )
        task = _choose_train_task(tasks, rng=rng, pos_task_prob=pos_task_prob)
        if task is not None:
            yield task


def _load_eval_samples(
    *,
    dataset_name: str,
    split: str,
    token: Optional[str],
    seed: int,
    buffer_size: int,
    all_class_names: list[str],
    neg_prompts_per_empty: int,
    neg_prompts_per_nonempty: int,
    point_prompt_style: str,
    max_samples: int,
) -> list[TaskSample]:
    samples: list[TaskSample] = []
    rng = random.Random(seed)
    for sample in _iter_base_rows(
        dataset_name=dataset_name,
        split=split,
        token=token,
        seed=seed,
        buffer_size=buffer_size,
    ):
        tasks = _tasks_from_base_sample(
            sample,
            all_class_names=all_class_names,
            rng=rng,
            neg_prompts_per_empty=neg_prompts_per_empty,
            neg_prompts_per_nonempty=neg_prompts_per_nonempty,
            point_prompt_style=point_prompt_style,
        )
        for task in tasks:
            samples.append(task)
            if len(samples) >= max_samples:
                return samples
    return samples


def _box_iou(a: DetectAnnotation, b: DetectAnnotation) -> float:
    inter_x_min = max(a.x_min, b.x_min)
    inter_y_min = max(a.y_min, b.y_min)
    inter_x_max = min(a.x_max, b.x_max)
    inter_y_max = min(a.y_max, b.y_max)
    inter_w = max(0.0, inter_x_max - inter_x_min)
    inter_h = max(0.0, inter_y_max - inter_y_min)
    inter = inter_w * inter_h
    if inter == 0.0:
        return 0.0
    area_a = max(0.0, a.x_max - a.x_min) * max(0.0, a.y_max - a.y_min)
    area_b = max(0.0, b.x_max - b.x_min) * max(0.0, b.y_max - b.y_min)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _match_ious(predicted: list[DetectAnnotation], ground_truth: list[DetectAnnotation]) -> np.ndarray:
    if not predicted or not ground_truth:
        return np.array([], dtype=np.float32)
    size = max(len(predicted), len(ground_truth))
    iou_matrix = np.zeros((size, size), dtype=np.float32)
    for i, gt in enumerate(ground_truth):
        for j, pred in enumerate(predicted):
            iou_matrix[i, j] = _box_iou(pred, gt)
    row_idx, col_idx = linear_sum_assignment(1.0 - iou_matrix)
    return iou_matrix[row_idx, col_idx]


def _count_tp_fp_fn_detect(predicted: list[DetectAnnotation], ground_truth: list[DetectAnnotation]) -> tuple[int, int, int]:
    if not predicted and not ground_truth:
        return 0, 0, 0
    if not predicted:
        return 0, 0, len(ground_truth)
    if not ground_truth:
        return 0, len(predicted), 0
    matches = _match_ious(predicted, ground_truth)
    tp = int((matches >= 0.5).sum())
    return tp, len(predicted) - tp, len(ground_truth) - tp


def _reward_detect_f1_weighted(
    predicted: list[DetectAnnotation],
    ground_truth: list[DetectAnnotation],
    *,
    fn_penalty_exponent: float,
    fp_penalty_exponent: float,
    neg_reward_weight: float,
) -> float:
    tp, fp, fn = _count_tp_fp_fn_detect(predicted, ground_truth)
    if tp == 0 and fp == 0 and fn == 0:
        reward = 1.0
    else:
        weighted_fp = float(fp) ** float(fp_penalty_exponent)
        weighted_fn = float(fn) ** float(fn_penalty_exponent)
        denom = (2.0 * float(tp)) + weighted_fp + weighted_fn
        reward = 0.0 if denom <= 0.0 else (2.0 * float(tp)) / float(denom)
    if not ground_truth:
        reward *= float(neg_reward_weight)
    return reward


def _reward_detect_miou(
    predicted: list[DetectAnnotation],
    ground_truth: list[DetectAnnotation],
    *,
    fn_penalty_exponent: float,
    fp_penalty_exponent: float,
    neg_reward_weight: float,
) -> float:
    if not predicted and not ground_truth:
        reward = 1.0
    elif not predicted or not ground_truth:
        reward = 0.0
    else:
        matches = _match_ious(predicted, ground_truth)
        true_pos = int((matches >= 0.5).sum())
        false_pos = len(predicted) - true_pos
        false_neg = len(ground_truth) - true_pos
        weighted_pred = float(true_pos) + (float(false_pos) ** float(fp_penalty_exponent))
        weighted_gt = float(true_pos) + (float(false_neg) ** float(fn_penalty_exponent))
        denom = max(weighted_pred, weighted_gt)
        reward = float(matches.sum()) / float(denom) if denom else 0.0
    if not ground_truth:
        reward *= float(neg_reward_weight)
    return reward


def _match_points_in_boxes(points: list[PointAnnotation], boxes: list[DetectAnnotation]) -> int:
    if not points or not boxes:
        return 0
    size = max(len(points), len(boxes))
    score = np.zeros((size, size), dtype=np.float32)
    for i, box in enumerate(boxes):
        for j, point in enumerate(points):
            score[i, j] = 1.0 if box.x_min <= point.x <= box.x_max and box.y_min <= point.y <= box.y_max else 0.0
    row_idx, col_idx = linear_sum_assignment(-score)
    return int(score[row_idx, col_idx].sum())


def _count_tp_fp_fn_point(points: list[PointAnnotation], boxes: list[DetectAnnotation]) -> tuple[int, int, int]:
    if not points and not boxes:
        return 0, 0, 0
    if not points:
        return 0, 0, len(boxes)
    if not boxes:
        return 0, len(points), 0
    tp = _match_points_in_boxes(points, boxes)
    return tp, len(points) - tp, len(boxes) - tp


def _reward_point_f1_weighted(
    points: list[PointAnnotation],
    boxes: list[DetectAnnotation],
    *,
    fn_penalty_exponent: float,
    fp_penalty_exponent: float,
    neg_reward_weight: float,
) -> float:
    tp, fp, fn = _count_tp_fp_fn_point(points, boxes)
    if tp == 0 and fp == 0 and fn == 0:
        reward = 1.0
    else:
        weighted_fp = float(fp) ** float(fp_penalty_exponent)
        weighted_fn = float(fn) ** float(fn_penalty_exponent)
        denom = (2.0 * float(tp)) + weighted_fp + weighted_fn
        reward = 0.0 if denom <= 0.0 else (2.0 * float(tp)) / float(denom)
    if not boxes:
        reward *= float(neg_reward_weight)
    return reward


def _rewards_for_rollouts(
    rollouts: List[Rollout],
    *,
    skill: str,
    reward_metric: str,
    boxes: list[DetectAnnotation],
    fn_penalty_exponent: float,
    fp_penalty_exponent: float,
    neg_reward_weight: float,
) -> list[float]:
    rewards: list[float] = []
    for rollout in rollouts:
        if skill == "point":
            rewards.append(
                _reward_point_f1_weighted(
                    list(getattr(rollout.output, "points", []) or []),
                    boxes,
                    fn_penalty_exponent=fn_penalty_exponent,
                    fp_penalty_exponent=fp_penalty_exponent,
                    neg_reward_weight=neg_reward_weight,
                )
            )
        elif reward_metric == "miou":
            rewards.append(
                _reward_detect_miou(
                    list(getattr(rollout.output, "objects", []) or []),
                    boxes,
                    fn_penalty_exponent=fn_penalty_exponent,
                    fp_penalty_exponent=fp_penalty_exponent,
                    neg_reward_weight=neg_reward_weight,
                )
            )
        else:
            rewards.append(
                _reward_detect_f1_weighted(
                    list(getattr(rollout.output, "objects", []) or []),
                    boxes,
                    fn_penalty_exponent=fn_penalty_exponent,
                    fp_penalty_exponent=fp_penalty_exponent,
                    neg_reward_weight=neg_reward_weight,
                )
            )
    return rewards


def _ground_truth_points(boxes: list[DetectAnnotation]) -> list[PointAnnotation]:
    return [
        _PointAnnotation(
            x=(box.x_min + box.x_max) / 2.0,
            y=(box.y_min + box.y_max) / 2.0,
        )
        for box in boxes
    ]


def _build_train_group(
    *,
    skill: str,
    result,
    rewards: list[float],
    boxes: list[DetectAnnotation],
    off_policy: bool,
    off_policy_std_thresh: float,
    off_policy_max_reward: float,
    off_policy_min_reward: float,
    off_policy_reward_scale: float,
    neg_reward_weight: float,
) -> tuple[Any, dict[str, int]]:
    stats = {
        "off_policy_considered": 0,
        "off_policy_injected_total": 0,
        "off_policy_injected_positive": 0,
        "off_policy_injected_negative": 0,
    }
    rollouts = list(result.rollouts)
    final_rewards = list(rewards)
    if not off_policy or not final_rewards or not rollouts:
        return _TrainStepGroup(request=result.request, rollouts=rollouts, rewards=final_rewards), stats

    stats["off_policy_considered"] = 1
    mean_reward = float(np.mean(final_rewards))
    reward_std = float(np.std(final_rewards))
    max_reward = float(max(final_rewards))
    low_max_reward = max_reward < float(off_policy_max_reward)
    low_mean_reward = mean_reward < float(off_policy_max_reward)
    should_inject = low_max_reward or (low_mean_reward and reward_std < float(off_policy_std_thresh))
    if not should_inject:
        return _TrainStepGroup(request=result.request, rollouts=rollouts, rewards=final_rewards), stats

    replace_idx = int(np.argmin(np.asarray(final_rewards, dtype=np.float32)))
    old_rollout = rollouts[replace_idx]
    replacement_output = (
        _PointOutput(points=_ground_truth_points(boxes))
        if skill == "point"
        else _DetectOutput(objects=list(boxes))
    )
    rollouts[replace_idx] = _Rollout(
        skill=old_rollout.skill,
        finish_reason=old_rollout.finish_reason,
        output=replacement_output,
        answer_tokens=list(old_rollout.answer_tokens),
        thinking_tokens=list(old_rollout.thinking_tokens),
        coords=list(old_rollout.coords),
        sizes=list(old_rollout.sizes),
    )
    reward_anchor = max(max_reward, mean_reward)
    injected_reward = max(
        float(off_policy_min_reward),
        min(1.0, float(off_policy_reward_scale) * reward_anchor),
    )
    if not boxes:
        injected_reward *= float(neg_reward_weight)
    final_rewards[replace_idx] = float(injected_reward)
    stats["off_policy_injected_total"] = 1
    if boxes:
        stats["off_policy_injected_positive"] = 1
    else:
        stats["off_policy_injected_negative"] = 1
    return _TrainStepGroup(request=result.request, rollouts=rollouts, rewards=final_rewards), stats


def _task_request(
    sample: TaskSample,
    *,
    skill: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_objects: int,
):
    if skill == "point":
        return _PointRequest(
            object_name=sample.prompt,
            image_url=sample.image_url,
            settings=_PointSettings(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            ),
        )
    return _DetectRequest(
        object_name=sample.prompt,
        image_url=sample.image_url,
        settings=_DetectSettings(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            max_objects=max_objects,
        ),
    )


def _evaluate(
    *,
    finetune,
    samples: list[TaskSample],
    batch_size: int,
    skill: str,
    reward_metric: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_objects: int,
    fn_penalty_exponent: float,
    fp_penalty_exponent: float,
    neg_reward_weight: float,
) -> dict[str, float]:
    reward_values: list[float] = []
    miou_values: list[float] = []
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        requests = [
            _task_request(
                sample,
                skill=skill,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                max_objects=max_objects,
            )
            for sample in batch
        ]
        results = finetune.rollouts_batch(requests=requests, num_rollouts=1, max_workers=len(batch))
        for sample, result in zip(batch, results):
            rollout = result.rollouts[0]
            if skill == "point":
                pred_points = list(getattr(rollout.output, "points", []) or [])
                reward = _reward_point_f1_weighted(
                    pred_points,
                    sample.boxes,
                    fn_penalty_exponent=fn_penalty_exponent,
                    fp_penalty_exponent=fp_penalty_exponent,
                    neg_reward_weight=neg_reward_weight,
                )
                reward_values.append(reward)
                tp, fp, fn = _count_tp_fp_fn_point(pred_points, sample.boxes)
            else:
                pred_boxes = list(getattr(rollout.output, "objects", []) or [])
                reward = (
                    _reward_detect_miou(
                        pred_boxes,
                        sample.boxes,
                        fn_penalty_exponent=fn_penalty_exponent,
                        fp_penalty_exponent=fp_penalty_exponent,
                        neg_reward_weight=neg_reward_weight,
                    )
                    if reward_metric == "miou"
                    else _reward_detect_f1_weighted(
                        pred_boxes,
                        sample.boxes,
                        fn_penalty_exponent=fn_penalty_exponent,
                        fp_penalty_exponent=fp_penalty_exponent,
                        neg_reward_weight=neg_reward_weight,
                    )
                )
                reward_values.append(reward)
                miou_values.append(
                    _reward_detect_miou(
                        pred_boxes,
                        sample.boxes,
                        fn_penalty_exponent=fn_penalty_exponent,
                        fp_penalty_exponent=fp_penalty_exponent,
                        neg_reward_weight=1.0,
                    )
                )
                tp, fp, fn = _count_tp_fp_fn_detect(pred_boxes, sample.boxes)
            total_tp += tp
            total_fp += fp
            total_fn += fn

    denom = (2 * total_tp) + total_fp + total_fn
    eval_f1 = 1.0 if denom == 0 else (2.0 * float(total_tp)) / float(denom)
    metrics = {
        "eval_samples": float(len(samples)),
        "eval_reward_mean": float(np.mean(reward_values)) if reward_values else 0.0,
        "eval_f1": float(eval_f1),
    }
    if skill == "detect":
        metrics["eval_miou"] = float(np.mean(miou_values)) if miou_values else 0.0
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", default=os.environ.get("MOONDREAM_API_KEY"))
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"))
    parser.add_argument("--base-url", default=os.environ.get("TUNA_BASE_URL", "https://api.moondream.ai/v1"))
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="val")
    parser.add_argument("--skill", choices=["detect", "point"], default="detect")
    parser.add_argument("--class-names", default="")
    parser.add_argument("--point-prompt-style", choices=["detect_phrase", "class_name"], default="detect_phrase")
    parser.add_argument("--neg-prompts-per-empty", type=int, default=1)
    parser.add_argument("--neg-prompts-per-nonempty", type=int, default=0)
    parser.add_argument("--pos-task-prob", type=float, default=0.95)
    parser.add_argument("--finetune-name", default=f"pid-icons-{_random_suffix()}")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--group-size", type=int, default=6)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-objects", type=int, default=50)
    parser.add_argument("--reward-metric", choices=["f1", "miou"], default="f1")
    parser.add_argument("--augment-prob", type=float, default=0.5)
    parser.add_argument("--fn-penalty-exponent", type=float, default=1.0)
    parser.add_argument("--fp-penalty-exponent", type=float, default=1.0)
    parser.add_argument("--neg-reward-weight", type=float, default=0.5)
    parser.add_argument("--off-policy", dest="off_policy", action="store_true")
    parser.add_argument("--no-off-policy", dest="off_policy", action="store_false")
    parser.add_argument("--off-policy-std-thresh", type=float, default=0.02)
    parser.add_argument("--off-policy-max-reward", type=float, default=0.15)
    parser.add_argument("--off-policy-min-reward", type=float, default=0.15)
    parser.add_argument("--off-policy-reward-scale", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--buffer-size", type=int, default=512)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-max-samples", type=int, default=200)
    parser.add_argument("--eval-batch-size", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--wandb-project", default="tuna-pid-icons")
    parser.set_defaults(off_policy=False)
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("MOONDREAM_API_KEY is required")

    _require_tuna_sdk()
    if args.class_names.strip():
        class_names = sorted({_normalize_class_name(item) for item in args.class_names.split(",") if item.strip()})
    else:
        class_names = _discover_class_names(
            dataset_name=args.dataset_name,
            split=args.train_split,
            token=args.hf_token,
        )
    if not class_names:
        raise ValueError("No class names discovered for the dataset")

    client = _TunaClient(api_key=args.api_key, base_url=args.base_url)
    finetune = client.create_finetune(name=args.finetune_name, rank=args.rank)
    wandb = _import_wandb()

    run = wandb.init(
        project=args.wandb_project,
        config={
            "api_base_url": args.base_url,
            "dataset": args.dataset_name,
            "train_split": args.train_split,
            "eval_split": args.eval_split,
            "finetune_id": finetune.finetune_id,
            "finetune_name": finetune.name,
            "skill": args.skill,
            "class_count": len(class_names),
            "rank": args.rank,
            "num_steps": args.num_steps,
            "batch_size": args.batch_size,
            "group_size": args.group_size,
            "lr": args.lr,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "max_objects": args.max_objects,
            "reward_metric": args.reward_metric,
            "augment_prob": args.augment_prob,
            "fn_penalty_exponent": args.fn_penalty_exponent,
            "fp_penalty_exponent": args.fp_penalty_exponent,
            "neg_reward_weight": args.neg_reward_weight,
            "off_policy": args.off_policy,
            "off_policy_std_thresh": args.off_policy_std_thresh,
            "off_policy_max_reward": args.off_policy_max_reward,
            "off_policy_min_reward": args.off_policy_min_reward,
            "off_policy_reward_scale": args.off_policy_reward_scale,
            "pos_task_prob": args.pos_task_prob,
            "neg_prompts_per_empty": args.neg_prompts_per_empty,
            "neg_prompts_per_nonempty": args.neg_prompts_per_nonempty,
            "point_prompt_style": args.point_prompt_style,
        },
    )
    run.summary["finetune_id"] = finetune.finetune_id

    train_stream = _iter_task_samples(
        dataset_name=args.dataset_name,
        split=args.train_split,
        token=args.hf_token,
        seed=args.seed,
        buffer_size=args.buffer_size,
        all_class_names=class_names,
        neg_prompts_per_empty=args.neg_prompts_per_empty,
        neg_prompts_per_nonempty=args.neg_prompts_per_nonempty,
        point_prompt_style=args.point_prompt_style,
        pos_task_prob=args.pos_task_prob,
        augment_prob=args.augment_prob,
    )
    eval_samples = _load_eval_samples(
        dataset_name=args.dataset_name,
        split=args.eval_split,
        token=args.hf_token,
        seed=args.seed + 1,
        buffer_size=args.buffer_size,
        all_class_names=class_names,
        neg_prompts_per_empty=args.neg_prompts_per_empty,
        neg_prompts_per_nonempty=args.neg_prompts_per_nonempty,
        point_prompt_style=args.point_prompt_style,
        max_samples=args.eval_max_samples,
    )

    for step in range(args.num_steps):
        batch = [next(train_stream) for _ in range(args.batch_size)]
        requests = [
            _task_request(
                sample,
                skill=args.skill,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                max_objects=args.max_objects,
            )
            for sample in batch
        ]
        results = finetune.rollouts_batch(
            requests=requests,
            num_rollouts=args.group_size,
            max_workers=args.batch_size,
        )

        groups = []
        rewards_all: list[float] = []
        off_policy_considered = 0
        off_policy_injected_total = 0
        off_policy_injected_positive = 0
        off_policy_injected_negative = 0
        for sample, result in zip(batch, results):
            rewards = _rewards_for_rollouts(
                result.rollouts,
                skill=args.skill,
                reward_metric=args.reward_metric,
                boxes=sample.boxes,
                fn_penalty_exponent=args.fn_penalty_exponent,
                fp_penalty_exponent=args.fp_penalty_exponent,
                neg_reward_weight=args.neg_reward_weight,
            )
            group, stats = _build_train_group(
                skill=args.skill,
                result=result,
                rewards=rewards,
                boxes=sample.boxes,
                off_policy=args.off_policy,
                off_policy_std_thresh=args.off_policy_std_thresh,
                off_policy_max_reward=args.off_policy_max_reward,
                off_policy_min_reward=args.off_policy_min_reward,
                off_policy_reward_scale=args.off_policy_reward_scale,
                neg_reward_weight=args.neg_reward_weight,
            )
            groups.append(group)
            rewards_all.extend(group.rewards)
            off_policy_considered += stats["off_policy_considered"]
            off_policy_injected_total += stats["off_policy_injected_total"]
            off_policy_injected_positive += stats["off_policy_injected_positive"]
            off_policy_injected_negative += stats["off_policy_injected_negative"]

        train_out = finetune.train_step(groups=groups, lr=args.lr)
        metrics = {
            "reward_mean": float(np.mean(rewards_all)) if rewards_all else 0.0,
            "reward_var": float(np.var(rewards_all)) if rewards_all else 0.0,
            "accepted_groups": len(groups),
            "off_policy_considered": off_policy_considered,
            "off_policy_injected_total": off_policy_injected_total,
            "off_policy_injected_positive": off_policy_injected_positive,
            "off_policy_injected_negative": off_policy_injected_negative,
            "kl": train_out.kl,
            "router_kl": train_out.router_kl,
            "grad_norm": train_out.grad_norm,
        }

        if (step + 1) % args.eval_every == 0:
            metrics.update(
                _evaluate(
                    finetune=finetune,
                    samples=eval_samples,
                    batch_size=args.eval_batch_size,
                    skill=args.skill,
                    reward_metric=args.reward_metric,
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=args.max_tokens,
                    max_objects=args.max_objects,
                    fn_penalty_exponent=args.fn_penalty_exponent,
                    fp_penalty_exponent=args.fp_penalty_exponent,
                    neg_reward_weight=args.neg_reward_weight,
                )
            )

        wandb.log(metrics, step=step)
        print(
            f"step {step + 1}/{args.num_steps} reward={metrics['reward_mean']:.3f} "
            f"off_policy_injected={off_policy_injected_total} kl={train_out.kl} grad_norm={train_out.grad_norm}"
        )

        if (step + 1) % args.save_every == 0:
            finetune.save_checkpoint()

    finetune.save_checkpoint()
    wandb.finish()
    client.close()


if __name__ == "__main__":
    main()
