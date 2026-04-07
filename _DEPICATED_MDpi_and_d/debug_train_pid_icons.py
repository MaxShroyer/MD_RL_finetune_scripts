#!/usr/bin/env python3
"""Debug visualizer for class-conditional PI&D RL training.

This script mirrors the key path in train_pid_icons.py and writes per-step artifacts:
- request payload summaries
- augmentation traces
- model rollout responses and rewards
- visual overlays for source, augmented request, and predicted boxes
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageEnhance

import train_pid_icons as train
from tuna_sdk import DetectAnnotation, DetectOutput, DetectRequest, DetectSettings, Rollout, TrainStepGroup, TunaClient
from tuna_sdk.errors import TunaAPIError, TunaNetworkError


@dataclass(frozen=True)
class PreparedTask:
    source_sample: train.BaseSample
    augmented_sample: train.BaseSample
    task: train.TaskSample
    augment_trace: list[str]


def _slugify(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    clean = clean.strip("_")
    return clean or "item"


def _box_to_pixels(box: DetectAnnotation, width: int, height: int) -> tuple[int, int, int, int]:
    if width <= 0 or height <= 0:
        return 0, 0, 0, 0
    x0 = int(round(train._clamp(box.x_min) * (width - 1)))
    y0 = int(round(train._clamp(box.y_min) * (height - 1)))
    x1 = int(round(train._clamp(box.x_max) * (width - 1)))
    y1 = int(round(train._clamp(box.y_max) * (height - 1)))
    if x1 <= x0:
        x1 = min(width - 1, x0 + 1)
    if y1 <= y0:
        y1 = min(height - 1, y0 + 1)
    return x0, y0, x1, y1


def _overlay_boxes(
    image: Image.Image,
    layers: list[tuple[list[DetectAnnotation], tuple[int, int, int]]],
    caption_lines: list[str],
) -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    for boxes, color in layers:
        for box in boxes:
            draw.rectangle(_box_to_pixels(box, canvas.width, canvas.height), outline=color, width=3)

    caption_height = 18 * max(1, len(caption_lines)) + 8
    out = Image.new("RGB", (canvas.width, canvas.height + caption_height), (22, 22, 22))
    out.paste(canvas, (0, caption_height))
    txt = ImageDraw.Draw(out)
    y = 4
    for line in caption_lines:
        txt.text((6, y), line, fill=(245, 245, 245))
        y += 16
    return out


def _save_source_and_request_views(
    item_dir: Path,
    prepared: PreparedTask,
    *,
    prompt: str,
) -> dict[str, str]:
    source_boxes = [item.box for item in prepared.source_sample.boxes]
    augmented_boxes = [item.box for item in prepared.augmented_sample.boxes]
    task_gt_boxes = list(prepared.task.gt_boxes)

    source_view = _overlay_boxes(
        prepared.source_sample.image,
        [(source_boxes, (255, 210, 60))],
        [
            f"source | boxes={len(source_boxes)} | source={prepared.source_sample.source}",
        ],
    )
    source_path = item_dir / "source_boxes.jpg"
    source_view.save(source_path, format="JPEG", quality=92)

    request_view = _overlay_boxes(
        prepared.task.image,
        [
            (augmented_boxes, (80, 180, 255)),
            (task_gt_boxes, (80, 255, 120)),
        ],
        [
            f"request | prompt={prompt}",
            f"task_gt={len(task_gt_boxes)} | aug_boxes={len(augmented_boxes)}",
            f"augment: {'; '.join(prepared.augment_trace[:3])}",
        ],
    )
    request_path = item_dir / "request_augmented.jpg"
    request_view.save(request_path, format="JPEG", quality=92)

    return {
        "source_image": str(source_path),
        "request_image": str(request_path),
    }


def _save_rollout_views(
    item_dir: Path,
    prepared: PreparedTask,
    rollouts: list[Rollout],
    rewards: list[float],
    *,
    max_rollouts_visualized: int,
) -> list[str]:
    paths: list[str] = []
    for rollout_idx, rollout in enumerate(rollouts[:max_rollouts_visualized]):
        pred_boxes = rollout.output.objects if isinstance(rollout.output, DetectOutput) else []
        pred_boxes = pred_boxes or []
        reward = rewards[rollout_idx] if rollout_idx < len(rewards) else 0.0
        viz = _overlay_boxes(
            prepared.task.image,
            [
                (list(prepared.task.gt_boxes), (80, 255, 120)),
                (list(pred_boxes), (255, 90, 90)),
            ],
            [
                f"rollout={rollout_idx} finish={rollout.finish_reason}",
                f"reward={reward:.4f} | gt={len(prepared.task.gt_boxes)} pred={len(pred_boxes)}",
                f"prompt={prepared.task.prompt}",
            ],
        )
        rollout_path = item_dir / f"rollout_{rollout_idx:02d}.jpg"
        viz.save(rollout_path, format="JPEG", quality=92)
        paths.append(str(rollout_path))
    return paths


def _annotation_to_dict(box: DetectAnnotation) -> dict[str, float]:
    return {
        "x_min": float(box.x_min),
        "y_min": float(box.y_min),
        "x_max": float(box.x_max),
        "y_max": float(box.y_max),
    }


def _request_to_summary(request: DetectRequest, save_full_image_url: bool) -> dict[str, Any]:
    payload = request.to_payload()
    image_url = str(payload.get("image_url", ""))
    if save_full_image_url:
        payload["image_url"] = image_url
    else:
        payload["image_url"] = f"<redacted_data_url chars={len(image_url)}>"
    payload["image_url_chars"] = len(image_url)
    return payload


def _rollout_to_summary(rollout: Rollout, reward: float) -> dict[str, Any]:
    pred_boxes = rollout.output.objects if isinstance(rollout.output, DetectOutput) else []
    pred_boxes = pred_boxes or []
    return {
        "skill": rollout.skill,
        "finish_reason": rollout.finish_reason,
        "reward": float(reward),
        "predicted_boxes": [_annotation_to_dict(box) for box in pred_boxes],
        "answer_tokens": len(rollout.answer_tokens),
        "thinking_tokens": len(rollout.thinking_tokens),
    }


def _augment_base_sample_with_trace(
    sample: train.BaseSample,
    rng: random.Random,
    rng_np: np.random.Generator,
    config: train.AugmentConfig,
    *,
    augment_prob: float,
) -> tuple[train.BaseSample, list[str]]:
    image = sample.image
    boxes = list(sample.boxes)
    trace: list[str] = []

    width, height = image.size
    scale = rng.uniform(config.resize_min, config.resize_max)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    if new_width != width or new_height != height:
        image = image.resize((new_width, new_height), resample=Image.BICUBIC)
    trace.append(f"resize(scale={scale:.3f}) {width}x{height}->{image.width}x{image.height}")

    gate = rng.random()
    trace.append(f"augment_gate={gate:.3f} threshold={augment_prob:.3f}")
    if gate >= augment_prob:
        trace.append("augment_skip")
        return train.BaseSample(image=image, boxes=boxes, source=sample.source), trace

    crop_roll = rng.random()
    if crop_roll < config.crop_p:
        w, h = image.size
        if w < 2 or h < 2:
            trace.append(f"crop(skip_too_small roll={crop_roll:.3f})")
        else:
            scale_w = rng.uniform(config.crop_scale_min, config.crop_scale_max)
            scale_h = rng.uniform(config.crop_scale_min, config.crop_scale_max)
            crop_w = max(1, int(w * scale_w))
            crop_h = max(1, int(h * scale_h))
            if crop_w >= w and crop_h >= h:
                trace.append(
                    f"crop(noop roll={crop_roll:.3f} scale_w={scale_w:.3f} scale_h={scale_h:.3f})"
                )
            else:
                left = rng.randint(0, max(0, w - crop_w)) if w > crop_w else 0
                top = rng.randint(0, max(0, h - crop_h)) if h > crop_h else 0
                right = left + crop_w
                bottom = top + crop_h

                kept: list[train.ClassBox] = []
                for item in boxes:
                    box = item.box
                    x_min = box.x_min * w
                    y_min = box.y_min * h
                    x_max = box.x_max * w
                    y_max = box.y_max * h
                    if x_min >= left and y_min >= top and x_max <= right and y_max <= bottom:
                        kept.append(
                            train.ClassBox(
                                class_uid=item.class_uid,
                                class_name=item.class_name,
                                box=DetectAnnotation(
                                    x_min=(x_min - left) / crop_w,
                                    y_min=(y_min - top) / crop_h,
                                    x_max=(x_max - left) / crop_w,
                                    y_max=(y_max - top) / crop_h,
                                ),
                            )
                        )

                pre_crop_boxes = list(boxes)
                cropped_image = image.crop((left, top, right, bottom))
                if pre_crop_boxes and not kept:
                    trace.append(
                        "crop(reverted_empty "
                        f"roll={crop_roll:.3f} scale_w={scale_w:.3f} scale_h={scale_h:.3f} "
                        f"box_count={len(pre_crop_boxes)})"
                    )
                else:
                    image = cropped_image
                    boxes = kept
                    trace.append(
                        "crop(applied "
                        f"roll={crop_roll:.3f} scale_w={scale_w:.3f} scale_h={scale_h:.3f} "
                        f"left={left} top={top} kept={len(kept)})"
                    )
    else:
        trace.append(f"crop(skip roll={crop_roll:.3f} p={config.crop_p:.3f})")

    flip_roll = rng.random()
    if flip_roll < config.flip_p:
        flipped: list[train.ClassBox] = []
        for item in boxes:
            box = item.box
            flipped.append(
                train.ClassBox(
                    class_uid=item.class_uid,
                    class_name=item.class_name,
                    box=DetectAnnotation(
                        x_min=1.0 - box.x_max,
                        y_min=box.y_min,
                        x_max=1.0 - box.x_min,
                        y_max=box.y_max,
                    ),
                )
            )
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        boxes = flipped
        trace.append(f"flip(applied roll={flip_roll:.3f})")
    else:
        trace.append(f"flip(skip roll={flip_roll:.3f} p={config.flip_p:.3f})")

    stretch_roll = rng.random()
    if stretch_roll < config.stretch_p:
        w, h = image.size
        scale_x = rng.uniform(config.stretch_min, config.stretch_max)
        scale_y = rng.uniform(config.stretch_min, config.stretch_max)
        new_w = max(1, int(w * scale_x))
        new_h = max(1, int(h * scale_y))
        if new_w != w or new_h != h:
            image = image.resize((new_w, new_h), resample=Image.BICUBIC)
        trace.append(
            f"stretch(applied roll={stretch_roll:.3f} scale_x={scale_x:.3f} scale_y={scale_y:.3f} {w}x{h}->{new_w}x{new_h})"
        )
    else:
        trace.append(f"stretch(skip roll={stretch_roll:.3f} p={config.stretch_p:.3f})")

    color_roll = rng.random()
    if color_roll < config.color_p:
        brightness = rng.uniform(config.brightness_min, config.brightness_max)
        contrast = rng.uniform(config.contrast_min, config.contrast_max)
        saturation = rng.uniform(config.saturation_min, config.saturation_max)
        image = ImageEnhance.Brightness(image).enhance(brightness)
        image = ImageEnhance.Contrast(image).enhance(contrast)
        image = ImageEnhance.Color(image).enhance(saturation)
        trace.append(
            f"color(applied roll={color_roll:.3f} b={brightness:.3f} c={contrast:.3f} s={saturation:.3f})"
        )
    else:
        trace.append(f"color(skip roll={color_roll:.3f} p={config.color_p:.3f})")

    hue_roll = rng.random()
    if hue_roll < config.hue_p:
        delta = rng.uniform(config.hue_delta_min, config.hue_delta_max)
        shift = int(round(delta * 255.0))
        if shift != 0:
            hsv = np.asarray(image.convert("HSV"), dtype=np.uint8).copy()
            hsv[..., 0] = ((hsv[..., 0].astype(np.int16) + shift) % 256).astype(np.uint8)
            image = Image.fromarray(hsv, mode="HSV").convert("RGB")
        trace.append(f"hue(applied roll={hue_roll:.3f} delta={delta:.4f} shift={shift})")
    else:
        trace.append(f"hue(skip roll={hue_roll:.3f} p={config.hue_p:.3f})")

    noise_roll = rng.random()
    if noise_roll < config.noise_p:
        std = float(rng_np.uniform(config.noise_std_min, config.noise_std_max))
        arr = np.asarray(image).astype(np.float32)
        noise = rng_np.normal(0.0, std, size=arr.shape)
        arr = np.clip(arr + noise, 0.0, 255.0).astype(np.uint8)
        image = Image.fromarray(arr, mode="RGB")
        trace.append(f"noise(applied roll={noise_roll:.3f} std={std:.3f})")
    else:
        trace.append(f"noise(skip roll={noise_roll:.3f} p={config.noise_p:.3f})")

    return train.BaseSample(image=image, boxes=boxes, source=sample.source), trace


def _build_train_iterator(args: argparse.Namespace) -> tuple[Iterable[dict], list[str]]:
    dataset_path = args.dataset_path.strip()
    dataset_name = args.dataset_name.strip()
    use_local = bool(dataset_path)

    if use_local:
        row_iter = train._iter_local_rows(dataset_path, args.split, args.seed)
    else:
        if not dataset_name:
            raise ValueError("Provide --dataset-path or --dataset-name")
        row_iter = train._iter_hf_rows(dataset_name, args.split, args.hf_token, args.seed, args.buffer_size)

    class_names = train._load_class_names(args.class_names_file, dataset_path if use_local else None)
    if not class_names:
        discovered: set[str] = set()
        for _ in range(2000):
            row = next(row_iter)
            base = train._to_base_sample(row)
            if not base:
                continue
            for item in base.boxes:
                if item.class_name:
                    discovered.add(item.class_name)
        class_names = sorted(discovered)

    if not class_names:
        raise ValueError("Could not resolve class names for prompting")

    return row_iter, class_names


def _make_next_task_fn(
    *,
    row_iter: Iterable[dict],
    all_class_names: list[str],
    rng: random.Random,
    rng_np: np.random.Generator,
    augment_config: train.AugmentConfig,
    augment_prob: float,
    neg_prompts_per_empty: int,
    neg_prompts_per_nonempty: int,
):
    task_queue: list[PreparedTask] = []

    def _next_task() -> PreparedTask:
        while not task_queue:
            row = next(row_iter)
            source = train._to_base_sample(row)
            if source is None:
                continue
            augmented, trace = _augment_base_sample_with_trace(
                source,
                rng,
                rng_np,
                augment_config,
                augment_prob=augment_prob,
            )
            tasks = train._tasks_from_base_sample(
                augmented,
                all_class_names=all_class_names,
                rng=rng,
                neg_prompts_per_empty=neg_prompts_per_empty,
                neg_prompts_per_nonempty=neg_prompts_per_nonempty,
            )
            for task in tasks:
                task_queue.append(
                    PreparedTask(
                        source_sample=source,
                        augmented_sample=augmented,
                        task=task,
                        augment_trace=list(trace),
                    )
                )
            rng.shuffle(task_queue)
        return task_queue.pop()

    return _next_task


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug train_pid_icons.py with visualized requests/augmentations/responses")
    parser.add_argument("--env-file", default=str(train._repo_relative(".env")))
    parser.add_argument("--api-key", default=os.environ.get("MOONDREAM_API_KEY"))
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"))
    parser.add_argument("--base-url", default=os.environ.get("TUNA_BASE_URL", "https://api.moondream.ai/v1"))

    parser.add_argument("--dataset-path", default=str(train._repo_relative("outputs", "pid_icons_merged")))
    parser.add_argument("--dataset-name", default="")
    parser.add_argument("--split", default="train")
    parser.add_argument("--class-names-file", default="")

    parser.add_argument("--finetune-id", default="")
    parser.add_argument("--finetune-name", default="")
    parser.add_argument("--rank", type=int, default=8)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--buffer-size", type=int, default=1000)
    parser.add_argument("--debug-steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-workers", type=int, default=4)

    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-objects", type=int, default=10)
    parser.add_argument("--reward-metric", choices=["f1", "miou"], default="f1")

    parser.add_argument("--neg-prompts-per-empty", type=int, default=2)
    parser.add_argument("--neg-prompts-per-nonempty", type=int, default=1)
    parser.add_argument(
        "--neg-reward-weight",
        type=float,
        default=0.5,
        help="Scale factor applied to rewards for negative tasks (no GT boxes). Range: (0, 1].",
    )
    parser.add_argument("--augment-prob", type=float, default=0.5)
    parser.add_argument("--fn-penalty-exponent", type=float, default=1.0)
    parser.add_argument("--fp-penalty-exponent", type=float, default=1.0)

    parser.add_argument("--output-dir", default=str(train._repo_relative("outputs", "pid_debug")))
    parser.add_argument("--max-rollouts-visualized", type=int, default=4)
    parser.add_argument("--save-full-image-url", action="store_true")

    parser.add_argument("--apply-train-step", action="store_true")
    parser.add_argument("--lr", type=float, default=2e-3)
    args = parser.parse_args()

    load_dotenv(args.env_file, override=False)
    if not args.api_key:
        args.api_key = os.environ.get("MOONDREAM_API_KEY")
    if not args.hf_token:
        args.hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    if not args.api_key:
        raise ValueError("MOONDREAM_API_KEY is required")
    if args.debug_steps <= 0:
        raise ValueError("--debug-steps must be > 0")
    if args.batch_size <= 0 or args.group_size <= 0:
        raise ValueError("--batch-size and --group-size must be > 0")
    if args.neg_reward_weight <= 0.0:
        raise ValueError("--neg-reward-weight must be > 0")
    if args.neg_reward_weight > 1.0:
        print("warning: --neg-reward-weight > 1; clamping to 1.0")
        args.neg_reward_weight = 1.0
    if args.fn_penalty_exponent < 1.0:
        raise ValueError("--fn-penalty-exponent must be >= 1.0")
    if args.fp_penalty_exponent < 1.0:
        raise ValueError("--fp-penalty-exponent must be >= 1.0")

    out_root = Path(args.output_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    rng_np = np.random.default_rng(args.seed)
    augment_config = train._default_augment_config()

    row_iter, all_class_names = _build_train_iterator(args)
    next_task = _make_next_task_fn(
        row_iter=row_iter,
        all_class_names=all_class_names,
        rng=rng,
        rng_np=rng_np,
        augment_config=augment_config,
        augment_prob=args.augment_prob,
        neg_prompts_per_empty=args.neg_prompts_per_empty,
        neg_prompts_per_nonempty=args.neg_prompts_per_nonempty,
    )

    if not args.finetune_id and not args.finetune_name:
        args.finetune_name = f"pid-icons-debug-{train._random_suffix()}"

    client = TunaClient(api_key=args.api_key, base_url=args.base_url)
    try:
        if args.finetune_id:
            finetune = client.get_finetune(args.finetune_id)
        else:
            finetune = client.create_finetune(name=args.finetune_name, rank=args.rank)

        run_meta = {
            "finetune_id": finetune.finetune_id,
            "dataset_path": args.dataset_path,
            "dataset_name": args.dataset_name,
            "split": args.split,
            "class_count": len(all_class_names),
            "seed": args.seed,
            "debug_steps": args.debug_steps,
            "batch_size": args.batch_size,
            "group_size": args.group_size,
            "apply_train_step": args.apply_train_step,
            "reward_metric": args.reward_metric,
            "fn_penalty_exponent": args.fn_penalty_exponent,
            "fp_penalty_exponent": args.fp_penalty_exponent,
            "neg_reward_weight": args.neg_reward_weight,
            "created_at_unix_s": time.time(),
        }
        (out_root / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

        for step_idx in range(args.debug_steps):
            step_dir = out_root / f"step_{step_idx:04d}"
            step_dir.mkdir(parents=True, exist_ok=True)

            prepared_batch = [next_task() for _ in range(args.batch_size)]
            requests = [
                DetectRequest(
                    object_name=item.task.prompt,
                    image_url=train._to_data_url(item.task.image, quality=92),
                    settings=DetectSettings(
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_tokens=args.max_tokens,
                        max_objects=args.max_objects,
                    ),
                )
                for item in prepared_batch
            ]

            request_entries: list[dict[str, Any]] = []
            for idx, (prepared, request) in enumerate(zip(prepared_batch, requests)):
                class_slug = _slugify(prepared.task.class_name)
                item_dir = step_dir / f"item_{idx:02d}_{class_slug}"
                item_dir.mkdir(parents=True, exist_ok=True)
                image_paths = _save_source_and_request_views(item_dir, prepared, prompt=prepared.task.prompt)
                request_entries.append(
                    {
                        "item_idx": idx,
                        "prompt": prepared.task.prompt,
                        "class_name": prepared.task.class_name,
                        "is_positive": prepared.task.is_positive,
                        "source": prepared.task.source,
                        "augment_trace": list(prepared.augment_trace),
                        "gt_boxes": [_annotation_to_dict(box) for box in prepared.task.gt_boxes],
                        "request_payload": _request_to_summary(request, args.save_full_image_url),
                        "visualizations": image_paths,
                    }
                )

            request_path = step_dir / "requests.json"
            request_path.write_text(json.dumps(request_entries, indent=2), encoding="utf-8")

            try:
                results = finetune.rollouts_batch(
                    requests=requests,
                    num_rollouts=args.group_size,
                    max_workers=min(args.max_workers, args.batch_size),
                )
            except (TunaAPIError, TunaNetworkError) as exc:
                error_path = step_dir / "error.txt"
                error_path.write_text(str(exc), encoding="utf-8")
                print(f"step {step_idx}: rollouts_batch failed: {exc}")
                continue

            if len(results) != len(prepared_batch):
                print(
                    f"step {step_idx}: warning result/request mismatch ({len(results)} vs {len(prepared_batch)}). "
                    "Only aligned items will be written."
                )

            response_entries: list[dict[str, Any]] = []
            groups_for_train: list[TrainStepGroup] = []
            all_rewards: list[float] = []
            positive_count = 0

            for idx, (prepared, result) in enumerate(zip(prepared_batch, results)):
                rewards = train._rewards_for_rollouts(
                    list(result.rollouts),
                    prepared.task.gt_boxes,
                    reward_metric=args.reward_metric,
                    fn_penalty_exponent=args.fn_penalty_exponent,
                    fp_penalty_exponent=args.fp_penalty_exponent,
                    neg_reward_weight=args.neg_reward_weight,
                )
                all_rewards.extend(rewards)
                if prepared.task.is_positive:
                    positive_count += 1

                class_slug = _slugify(prepared.task.class_name)
                item_dir = step_dir / f"item_{idx:02d}_{class_slug}"
                rollout_paths = _save_rollout_views(
                    item_dir,
                    prepared,
                    list(result.rollouts),
                    rewards,
                    max_rollouts_visualized=max(1, args.max_rollouts_visualized),
                )

                response_entries.append(
                    {
                        "item_idx": idx,
                        "prompt": prepared.task.prompt,
                        "class_name": prepared.task.class_name,
                        "is_positive": prepared.task.is_positive,
                        "gt_count": len(prepared.task.gt_boxes),
                        "rollout_count": len(result.rollouts),
                        "rollouts": [
                            _rollout_to_summary(rollout, rewards[r_idx] if r_idx < len(rewards) else 0.0)
                            for r_idx, rollout in enumerate(result.rollouts)
                        ],
                        "rollout_visualizations": rollout_paths,
                    }
                )

                groups_for_train.append(
                    TrainStepGroup(
                        request=result.request,
                        rollouts=list(result.rollouts),
                        rewards=list(rewards),
                    )
                )

            response_path = step_dir / "responses.json"
            response_path.write_text(json.dumps(response_entries, indent=2), encoding="utf-8")

            train_step_summary: dict[str, Any] = {"applied": False}
            if args.apply_train_step and groups_for_train:
                try:
                    train_out = finetune.train_step(groups=groups_for_train, lr=args.lr)
                    train_step_summary = {
                        "applied": True,
                        "lr": args.lr,
                        "kl": float(train_out.kl or 0.0),
                        "router_kl": float(train_out.router_kl or 0.0),
                        "grad_norm": float(train_out.grad_norm or 0.0),
                    }
                except (TunaAPIError, TunaNetworkError) as exc:
                    train_step_summary = {"applied": False, "error": str(exc)}

            step_summary = {
                "step": step_idx,
                "batch_size": len(prepared_batch),
                "positive_tasks": positive_count,
                "negative_tasks": len(prepared_batch) - positive_count,
                "mean_reward": float(np.mean(all_rewards)) if all_rewards else 0.0,
                "reward_var": float(np.var(all_rewards)) if all_rewards else 0.0,
                "train_step": train_step_summary,
            }
            (step_dir / "summary.json").write_text(json.dumps(step_summary, indent=2), encoding="utf-8")

            print(
                f"step {step_idx}: requests={len(prepared_batch)} responses={len(response_entries)} "
                f"mean_reward={step_summary['mean_reward']:.4f} dir={step_dir}"
            )

        print(f"debug run complete. artifacts: {out_root}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
