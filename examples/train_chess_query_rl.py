"""Barebones query finetuning example for maxs-m87/chess-qa-synth-v1.

The default public configuration targets the piece-position variant:
- dataset variant: piece_position_v2_dataset2
- task focus: list_all_pieces

Requires:
  pip install datasets pillow wandb
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
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, List, Optional

import numpy as np
from datasets import load_dataset
from PIL import Image

if TYPE_CHECKING:
    from tuna_sdk import QueryRequest, Rollout

_QueryRequest: Any = None
_QuerySettings: Any = None
_TunaClient: Any = None

DATASET_NAME = "maxs-m87/chess-qa-synth-v1"
DATASET_VARIANT = "piece_position_v2_dataset2"
ACTIVE_TASKS = ("list_all_pieces",)
LIST_PIECE_REWARD_WEIGHTS = {"typed_f1": 0.6, "square_f1": 0.2, "piece_recall": 0.2}


def _import_wandb():
    try:
        return importlib.import_module("wandb")
    except ImportError as exc:
        raise RuntimeError("wandb is required to run training. Install it with `pip install wandb`.") from exc


def _require_tuna_sdk() -> None:
    global _QueryRequest, _QuerySettings, _TunaClient
    if _TunaClient is not None:
        return
    try:
        module = importlib.import_module("tuna_sdk")
    except ImportError as exc:
        raise RuntimeError(
            "tuna_sdk is required to run this example. Install the package with `pip install tuna-sdk` "
            "or `pip install -e .` from the SDK repo."
        ) from exc
    _QueryRequest = module.QueryRequest
    _QuerySettings = module.QuerySettings
    _TunaClient = module.TunaClient


@dataclass(frozen=True)
class QAExample:
    task_type: str
    question: str
    image_url: str
    expected_answer: dict[str, Any]


def _to_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _random_suffix(length: int = 6) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


def _parse_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value == 0:
            return False
        if value == 1:
            return True
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    return None


def _coerce_color(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    return normalized if normalized in {"white", "black"} else None


def _normalize_piece_key(value: Any) -> Optional[str]:
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text or None


def _normalize_square(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    square = value.strip().lower()
    if len(square) != 2 or square[0] not in "abcdefgh" or square[1] not in "12345678":
        return None
    return square


def _normalize_square_set(value: Any) -> Optional[frozenset[str]]:
    raw_items = list(value) if isinstance(value, (list, tuple, set)) else [value]
    squares: set[str] = set()
    for item in raw_items:
        square = _normalize_square(item)
        if square is None:
            return None
        squares.add(square)
    return frozenset(squares)


def _normalize_piece_map(pieces_payload: Any) -> Optional[tuple[tuple[str, tuple[str, ...]], ...]]:
    if isinstance(pieces_payload, dict):
        maps = [pieces_payload]
    elif isinstance(pieces_payload, list):
        maps = pieces_payload
    else:
        return None
    merged: dict[str, set[str]] = {}
    for piece_map in maps:
        if not isinstance(piece_map, dict):
            return None
        for raw_piece_key, raw_squares in piece_map.items():
            piece_key = _normalize_piece_key(raw_piece_key)
            square_set = _normalize_square_set(raw_squares)
            if piece_key is None or square_set is None:
                return None
            merged.setdefault(piece_key, set()).update(square_set)
    canonical: list[tuple[str, tuple[str, ...]]] = []
    for piece_key, squares in merged.items():
        canonical.append((piece_key, tuple(sorted(squares))))
    canonical.sort(key=lambda item: item[0])
    return tuple(canonical)


def _normalize_answer_for_task(task_type: str, payload: Any) -> Optional[dict[str, Any]]:
    if not isinstance(payload, dict):
        return None
    if task_type == "count_by_color":
        white_piece_count = _parse_int(payload.get("white_piece_count", payload.get("white_count")))
        black_piece_count = _parse_int(payload.get("black_piece_count", payload.get("black_count")))
        if white_piece_count is None or black_piece_count is None or white_piece_count < 0 or black_piece_count < 0:
            return None
        return {"white_piece_count": white_piece_count, "black_piece_count": black_piece_count}
    if task_type == "color_presence_check":
        color = _coerce_color(payload.get("color", payload.get("piece_color")))
        present = _coerce_bool(payload.get("present", payload.get("exists")))
        if color is None or present is None:
            return None
        normalized: dict[str, Any] = {"color": color, "present": present}
        if "count" in payload:
            count = _parse_int(payload.get("count"))
            if count is None or count < 0:
                return None
            normalized["count"] = count
        return normalized
    if task_type == "list_color_pieces":
        color = _coerce_color(payload.get("color", payload.get("piece_color")))
        pieces = _normalize_piece_map(payload.get("pieces"))
        if color is None or pieces is None:
            return None
        return {"color": color, "pieces": pieces}
    if task_type == "list_all_pieces":
        pieces = _normalize_piece_map(payload.get("pieces", payload))
        if pieces is None:
            return None
        return {"pieces": pieces}
    return None


def _json_object_candidates(text: str) -> list[str]:
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


def _parse_prediction_json(answer_text: str) -> Optional[dict[str, Any]]:
    text = str(answer_text or "").strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    for candidate in _json_object_candidates(text):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _typed_square_set_from_piece_map(piece_map: tuple[tuple[str, tuple[str, ...]], ...]) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for piece_key, squares in piece_map:
        for square in squares:
            out.add((piece_key, square))
    return out


def _square_set_from_piece_map(piece_map: tuple[tuple[str, tuple[str, ...]], ...]) -> set[str]:
    out: set[str] = set()
    for _, squares in piece_map:
        out.update(squares)
    return out


def _piece_counts_from_piece_map(piece_map: tuple[tuple[str, tuple[str, ...]], ...]) -> dict[str, int]:
    return {piece_key: len(squares) for piece_key, squares in piece_map}


def _f1_from_sets(pred: set[Any], gt: set[Any]) -> float:
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


def _piece_recall_score(pred_counts: dict[str, int], gt_counts: dict[str, int]) -> float:
    gt_total = sum(gt_counts.values())
    matched = sum(min(pred_counts.get(key, 0), gt_counts.get(key, 0)) for key in gt_counts.keys())
    return float(matched) / float(max(1, gt_total))


def _count_similarity(*, pred_count: int, gt_count: int) -> float:
    return max(0.0, 1.0 - (abs(pred_count - gt_count) / float(max(1, gt_count))))


def _dense_list_piece_reward(
    *,
    gt_piece_map: tuple[tuple[str, tuple[str, ...]], ...],
    pred_piece_map: tuple[tuple[str, tuple[str, ...]], ...],
) -> float:
    typed_f1 = _f1_from_sets(
        _typed_square_set_from_piece_map(pred_piece_map),
        _typed_square_set_from_piece_map(gt_piece_map),
    )
    square_f1 = _f1_from_sets(
        _square_set_from_piece_map(pred_piece_map),
        _square_set_from_piece_map(gt_piece_map),
    )
    piece_recall = _piece_recall_score(
        _piece_counts_from_piece_map(pred_piece_map),
        _piece_counts_from_piece_map(gt_piece_map),
    )
    reward = (
        LIST_PIECE_REWARD_WEIGHTS["typed_f1"] * typed_f1
        + LIST_PIECE_REWARD_WEIGHTS["square_f1"] * square_f1
        + LIST_PIECE_REWARD_WEIGHTS["piece_recall"] * piece_recall
    )
    return max(0.0, min(1.0, float(reward)))


def _normalize_task_type(task_type: str) -> str:
    aliases = {
        "piece_position": "list_all_pieces",
        "list_pieces": "list_all_pieces",
        "presence_check": "color_presence_check",
    }
    return aliases.get(str(task_type).strip(), str(task_type).strip())


def _parse_example(row: dict) -> Optional[QAExample]:
    task_type = _normalize_task_type(str(row.get("task_type") or "").strip())
    if task_type not in ACTIVE_TASKS:
        return None
    image = row.get("image")
    if image is None:
        return None
    try:
        payload = json.loads(str(row["final_answer_json"]))
    except (KeyError, json.JSONDecodeError):
        return None
    expected_answer = _normalize_answer_for_task(task_type, payload)
    if expected_answer is None:
        return None
    return QAExample(
        task_type=task_type,
        question=str(row.get("question") or "").strip(),
        image_url=_to_data_url(image.convert("RGB")),
        expected_answer=expected_answer,
    )


def _iter_examples(
    *,
    dataset_name: str,
    dataset_variant_tag: str,
    split: str,
    token: Optional[str],
    seed: int,
    buffer_size: int,
) -> Iterable[QAExample]:
    epoch = 0
    while True:
        ds = load_dataset(dataset_name, name=dataset_variant_tag, split=split, streaming=True, token=token)
        if seed:
            ds = ds.shuffle(seed=seed + epoch, buffer_size=buffer_size)
        for row in ds:
            example = _parse_example(row)
            if example is not None:
                yield example
        epoch += 1


def _load_eval_examples(
    *,
    dataset_name: str,
    dataset_variant_tag: str,
    split: str,
    token: Optional[str],
    seed: int,
    buffer_size: int,
    max_samples: int,
) -> list[QAExample]:
    examples: list[QAExample] = []
    stream = _iter_examples(
        dataset_name=dataset_name,
        dataset_variant_tag=dataset_variant_tag,
        split=split,
        token=token,
        seed=seed,
        buffer_size=buffer_size,
    )
    while len(examples) < max_samples:
        examples.append(next(stream))
    return examples


def _request_for_example(
    example: QAExample,
    *,
    temperature: float,
    top_p: float,
    max_tokens: int,
    reasoning: bool,
) -> QueryRequest:
    return _QueryRequest(
        question=example.question,
        image_url=example.image_url,
        reasoning=reasoning,
        settings=_QuerySettings(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        ),
    )


def _score_rollout(rollout: Rollout, example: QAExample) -> dict[str, float | bool]:
    payload = _parse_prediction_json(str(getattr(rollout.output, "answer", "") or ""))
    if payload is None:
        return {"reward": 0.0, "json_parse_success": False, "exact_match": False}
    pred = _normalize_answer_for_task(example.task_type, payload)
    if pred is None:
        return {"reward": 0.0, "json_parse_success": False, "exact_match": False}

    exact_match = pred == example.expected_answer
    reward = 1.0 if exact_match else 0.0
    if example.task_type in {"list_all_pieces", "list_color_pieces"}:
        gt_piece_map = example.expected_answer["pieces"]
        pred_piece_map = pred["pieces"]
        reward = _dense_list_piece_reward(
            gt_piece_map=gt_piece_map,
            pred_piece_map=pred_piece_map,
        )
    elif example.task_type == "count_by_color":
        reward = (
            _count_similarity(
                pred_count=int(pred["white_piece_count"]),
                gt_count=int(example.expected_answer["white_piece_count"]),
            )
            + _count_similarity(
                pred_count=int(pred["black_piece_count"]),
                gt_count=int(example.expected_answer["black_piece_count"]),
            )
        ) / 2.0
    elif example.task_type == "color_presence_check":
        reward = 1.0 if exact_match else 0.0

    return {"reward": float(reward), "json_parse_success": True, "exact_match": exact_match}


def _reward_from_rollouts(rollouts: List[Rollout], example: QAExample) -> list[float]:
    return [float(_score_rollout(rollout, example)["reward"]) for rollout in rollouts]


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
    desired_off_policy = max(1, desired_off_policy)
    desired_off_policy = min(desired_off_policy, len(on_policy_groups), len(replay_groups))
    if desired_off_policy <= 0:
        return list(on_policy_groups), 0

    keep_on_policy = len(on_policy_groups) - desired_off_policy
    selected_on_policy = (
        list(on_policy_groups)
        if keep_on_policy >= len(on_policy_groups)
        else ([] if keep_on_policy <= 0 else rng.sample(list(on_policy_groups), k=keep_on_policy))
    )
    selected_off_policy = rng.sample(list(replay_groups), k=desired_off_policy)
    mixed = selected_on_policy + selected_off_policy
    rng.shuffle(mixed)
    return mixed, desired_off_policy


def _evaluate(
    *,
    finetune,
    examples: list[QAExample],
    batch_size: int,
    max_tokens: int,
    reasoning: bool,
) -> dict[str, float]:
    reward_values: list[float] = []
    parse_success = 0
    exact_match_count = 0

    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        requests = [
            _request_for_example(
                example,
                temperature=0.0,
                top_p=1.0,
                max_tokens=max_tokens,
                reasoning=reasoning,
            )
            for example in batch
        ]
        results = finetune.rollouts_batch(requests=requests, num_rollouts=1, max_workers=len(batch))
        for example, result in zip(batch, results):
            outcome = _score_rollout(result.rollouts[0], example)
            reward_values.append(float(outcome["reward"]))
            parse_success += int(bool(outcome["json_parse_success"]))
            exact_match_count += int(bool(outcome["exact_match"]))

    return {
        "eval_samples": float(len(examples)),
        "eval_reward_mean": float(np.mean(reward_values)) if reward_values else 0.0,
        "eval_json_parse_rate": float(parse_success) / float(max(1, len(examples))),
        "eval_exact_accuracy": float(exact_match_count) / float(max(1, len(examples))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", default=os.environ.get("MOONDREAM_API_KEY"))
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"))
    parser.add_argument("--base-url", default=os.environ.get("TUNA_BASE_URL", "https://api.moondream.ai/v1"))
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--dataset-variant-tag", default=DATASET_VARIANT)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="val")
    parser.add_argument("--finetune-name", default=f"chess-query-{_random_suffix()}")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--num-steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--reasoning", dest="reasoning", action="store_true")
    parser.add_argument("--no-reasoning", dest="reasoning", action="store_false")
    parser.add_argument("--off-policy", dest="off_policy", action="store_true")
    parser.add_argument("--no-off-policy", dest="off_policy", action="store_false")
    parser.add_argument("--off-policy-mix-ratio", type=float, default=0.5)
    parser.add_argument("--off-policy-buffer-size", type=int, default=4096)
    parser.add_argument("--off-policy-warmup-steps", type=int, default=10)
    parser.add_argument("--off-policy-min-buffer-groups", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--buffer-size", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-max-samples", type=int, default=1000)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--wandb-project", default="tuna-chess-query")
    parser.set_defaults(reasoning=False, off_policy=True)
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("MOONDREAM_API_KEY is required")

    _require_tuna_sdk()
    client = _TunaClient(api_key=args.api_key, base_url=args.base_url)
    finetune = client.create_finetune(name=args.finetune_name, rank=args.rank)
    wandb = _import_wandb()

    run = wandb.init(
        project=args.wandb_project,
        config={
            "api_base_url": args.base_url,
            "dataset": args.dataset_name,
            "dataset_variant_tag": args.dataset_variant_tag,
            "train_split": args.train_split,
            "eval_split": args.eval_split,
            "finetune_id": finetune.finetune_id,
            "finetune_name": finetune.name,
            "rank": args.rank,
            "num_steps": args.num_steps,
            "batch_size": args.batch_size,
            "group_size": args.group_size,
            "lr": args.lr,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "reasoning": args.reasoning,
            "off_policy": args.off_policy,
            "off_policy_mix_ratio": args.off_policy_mix_ratio,
            "off_policy_buffer_size": args.off_policy_buffer_size,
            "off_policy_warmup_steps": args.off_policy_warmup_steps,
            "off_policy_min_buffer_groups": args.off_policy_min_buffer_groups,
            "active_tasks": list(ACTIVE_TASKS),
        },
    )
    run.summary["finetune_id"] = finetune.finetune_id
    rng = random.Random(args.seed)
    replay_buffer: deque[Any] = deque(
        maxlen=args.off_policy_buffer_size if args.off_policy_buffer_size > 0 else None
    )

    train_stream = _iter_examples(
        dataset_name=args.dataset_name,
        dataset_variant_tag=args.dataset_variant_tag,
        split=args.train_split,
        token=args.hf_token,
        seed=args.seed,
        buffer_size=args.buffer_size,
    )
    eval_examples = _load_eval_examples(
        dataset_name=args.dataset_name,
        dataset_variant_tag=args.dataset_variant_tag,
        split=args.eval_split,
        token=args.hf_token,
        seed=args.seed + 1,
        buffer_size=args.buffer_size,
        max_samples=args.eval_max_samples,
    )

    for step in range(args.num_steps):
        batch = [next(train_stream) for _ in range(args.batch_size)]
        requests = [
            _request_for_example(
                example,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                reasoning=args.reasoning,
            )
            for example in batch
        ]
        results = finetune.rollouts_batch(
            requests=requests,
            num_rollouts=args.group_size,
            max_workers=args.batch_size,
        )

        on_policy_groups = []
        rewards_all: list[float] = []
        for example, result in zip(batch, results):
            rewards = _reward_from_rollouts(result.rollouts, example)
            on_policy_groups.append(result.to_group(rewards=rewards))
            rewards_all.extend(rewards)

        train_groups, off_policy_groups = _compose_train_groups(
            on_policy_groups=on_policy_groups,
            replay_groups=list(replay_buffer),
            off_policy=args.off_policy,
            off_policy_mix_ratio=args.off_policy_mix_ratio,
            off_policy_warmup_steps=args.off_policy_warmup_steps,
            off_policy_min_buffer_groups=args.off_policy_min_buffer_groups,
            global_step=step,
            rng=rng,
        )
        train_out = finetune.train_step(groups=train_groups, lr=args.lr)
        if args.off_policy:
            replay_buffer.extend(on_policy_groups)
        metrics = {
            "reward_mean": float(np.mean(rewards_all)) if rewards_all else 0.0,
            "reward_var": float(np.var(rewards_all)) if rewards_all else 0.0,
            "accepted_groups": len(train_groups),
            "on_policy_groups": len(on_policy_groups),
            "off_policy_groups": off_policy_groups,
            "off_policy_group_fraction": float(off_policy_groups) / float(max(1, len(train_groups))),
            "replay_buffer_size": len(replay_buffer),
            "kl": train_out.kl,
            "router_kl": train_out.router_kl,
            "grad_norm": train_out.grad_norm,
        }

        if (step + 1) % args.eval_every == 0:
            eval_metrics = _evaluate(
                finetune=finetune,
                examples=eval_examples,
                batch_size=args.eval_batch_size,
                max_tokens=args.max_tokens,
                reasoning=args.reasoning,
            )
            metrics.update(eval_metrics)

        wandb.log(metrics, step=step)
        print(
            f"step {step + 1}/{args.num_steps} reward={metrics['reward_mean']:.3f} "
            f"kl={train_out.kl} grad_norm={train_out.grad_norm}"
        )

        if (step + 1) % args.save_every == 0:
            finetune.save_checkpoint()

    finetune.save_checkpoint()
    wandb.finish()
    client.close()


if __name__ == "__main__":
    main()
