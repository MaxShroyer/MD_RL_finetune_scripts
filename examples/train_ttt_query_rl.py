"""Barebones query finetuning example for maxs-m87/tictactoe-qa-v1.

This example keeps the original hard-task emphasis:
- best_move
- available_moves_count
- available_moves_list

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

DATASET_NAME = "maxs-m87/tictactoe-qa-v1"
ACTIVE_TASKS = ("best_move", "available_moves_count", "available_moves_list")
BEST_MOVE_OPTIMAL_REWARD = 0.7
MAX_TOKENS_BY_TASK = {"available_moves_list": 384}


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
    best_move_canonical: Optional[int] = None
    best_move_optimal_set: frozenset[int] = frozenset()
    best_move_scores: tuple[tuple[int, int, int], ...] = tuple()


def _to_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _random_suffix(length: int = 6) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


def _normalize_task_type(task_type: str) -> str:
    aliases = {
        "is_terminal": "is_game_over",
        "legal_moves_count": "available_moves_count",
        "legal_moves_list": "available_moves_list",
    }
    return aliases.get(str(task_type).strip(), str(task_type).strip())


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


def _normalize_moves(value: Any) -> Optional[tuple[tuple[int, int], ...]]:
    if not isinstance(value, list):
        return None
    moves: list[tuple[int, int]] = []
    for item in value:
        if not isinstance(item, dict):
            return None
        row = _parse_int(item.get("row"))
        col = _parse_int(item.get("col"))
        if row is None or col is None or row < 1 or row > 3 or col < 1 or col > 3:
            return None
        moves.append((row, col))
    return tuple(moves)


def _move_from_payload_obj(payload: Any) -> Optional[int]:
    if not isinstance(payload, dict):
        return None
    if "move" in payload:
        move = _parse_int(payload.get("move"))
        if move is None or move < 1 or move > 9:
            return None
        return move
    row = _parse_int(payload.get("row"))
    col = _parse_int(payload.get("col"))
    if row is None or col is None or row < 1 or row > 3 or col < 1 or col > 3:
        return None
    return ((row - 1) * 3) + col


def _best_move_from_json(payload_json: str) -> Optional[int]:
    try:
        payload = json.loads(payload_json)
    except json.JSONDecodeError:
        return None
    return _move_from_payload_obj(payload)


def _move_set_from_json(payload_json: str) -> frozenset[int]:
    try:
        payload = json.loads(payload_json)
    except json.JSONDecodeError:
        return frozenset()
    if not isinstance(payload, list):
        return frozenset()
    out: set[int] = set()
    for item in payload:
        move = _move_from_payload_obj(item)
        if move is not None:
            out.add(move)
    return frozenset(out)


def _scores_by_move_from_json(payload_json: str) -> tuple[tuple[int, int, int], ...]:
    try:
        payload = json.loads(payload_json)
    except json.JSONDecodeError:
        return tuple()
    if not isinstance(payload, dict):
        return tuple()
    out: list[tuple[int, int, int]] = []
    for move_key, score_payload in payload.items():
        move = _parse_int(move_key)
        if move is None or move < 1 or move > 9 or not isinstance(score_payload, dict):
            continue
        value = _parse_int(score_payload.get("value"))
        depth = _parse_int(score_payload.get("depth"))
        if value is None or depth is None:
            continue
        out.append((move, value, depth))
    out.sort(key=lambda item: item[0])
    return tuple(out)


def _normalize_expected_answer(task_type: str, payload: Any) -> Optional[dict[str, Any]]:
    if not isinstance(payload, dict):
        return None
    if task_type == "available_moves_count":
        count = _parse_int(payload.get("available_move_count", payload.get("legal_move_count")))
        if count is None or count < 0 or count > 9:
            return None
        return {"available_move_count": count}
    if task_type == "available_moves_list":
        moves = _normalize_moves(payload.get("available_moves", payload.get("legal_moves")))
        if moves is None:
            return None
        return {"available_moves": moves}
    if task_type == "best_move":
        return {}
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


def _best_move_rank_key(*, value: int, depth: int) -> tuple[int, int]:
    if value == 1:
        return value, -depth
    return value, depth


def _ranked_best_move_reward(move: int, *, scores_by_move: dict[int, tuple[int, int]]) -> float:
    pred = scores_by_move.get(move)
    if pred is None:
        return 0.0
    total = len(scores_by_move)
    if total <= 1:
        return 1.0
    pred_key = _best_move_rank_key(value=pred[0], depth=pred[1])
    better_count = sum(
        1
        for value, depth in scores_by_move.values()
        if _best_move_rank_key(value=value, depth=depth) > pred_key
    )
    return max(0.0, min(1.0, 1.0 - (float(better_count) / float(total - 1))))


def _parse_example(row: dict) -> Optional[QAExample]:
    task_type = _normalize_task_type(str(row.get("task_type") or "").strip())
    if task_type not in ACTIVE_TASKS:
        return None
    image = row.get("image")
    if image is None:
        return None
    try:
        expected_payload = json.loads(str(row["final_answer_json"]))
    except (KeyError, json.JSONDecodeError):
        return None
    expected_answer = _normalize_expected_answer(task_type, expected_payload)
    if expected_answer is None:
        return None
    return QAExample(
        task_type=task_type,
        question=str(row.get("question") or "").strip(),
        image_url=_to_data_url(image.convert("RGB")),
        expected_answer=expected_answer,
        best_move_canonical=_best_move_from_json(str(row.get("best_move_canonical_json", ""))),
        best_move_optimal_set=_move_set_from_json(str(row.get("best_move_optimal_set_json", ""))),
        best_move_scores=_scores_by_move_from_json(str(row.get("scores_by_move_json", ""))),
    )


def _iter_examples(
    *,
    dataset_name: str,
    split: str,
    token: Optional[str],
    seed: int,
    buffer_size: int,
) -> Iterable[QAExample]:
    epoch = 0
    while True:
        ds = load_dataset(dataset_name, split=split, streaming=True, token=token)
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
    split: str,
    token: Optional[str],
    seed: int,
    buffer_size: int,
    max_samples: int,
) -> list[QAExample]:
    examples: list[QAExample] = []
    stream = _iter_examples(
        dataset_name=dataset_name,
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
    effective_max_tokens = MAX_TOKENS_BY_TASK.get(example.task_type, max_tokens)
    return _QueryRequest(
        question=example.question,
        image_url=example.image_url,
        reasoning=reasoning,
        settings=_QuerySettings(
            temperature=temperature,
            top_p=top_p,
            max_tokens=effective_max_tokens,
        ),
    )


def _score_rollout(rollout: Rollout, example: QAExample) -> dict[str, float | bool]:
    payload = _parse_prediction_json(str(getattr(rollout.output, "answer", "") or ""))
    if payload is None:
        return {
            "reward": 0.0,
            "json_parse_success": False,
            "best_move_set_correct": False,
            "best_move_canonical_correct": False,
            "non_best_exact_correct": False,
        }

    if example.task_type == "best_move":
        move = _move_from_payload_obj(payload)
        if move is None:
            return {
                "reward": 0.0,
                "json_parse_success": False,
                "best_move_set_correct": False,
                "best_move_canonical_correct": False,
                "non_best_exact_correct": False,
            }
        if example.best_move_scores:
            score_map = {move_key: (value, depth) for move_key, value, depth in example.best_move_scores}
            reward = _ranked_best_move_reward(move, scores_by_move=score_map)
        elif move == example.best_move_canonical:
            reward = 1.0
        elif move in example.best_move_optimal_set:
            reward = BEST_MOVE_OPTIMAL_REWARD
        else:
            reward = 0.0
        return {
            "reward": reward,
            "json_parse_success": True,
            "best_move_set_correct": move in example.best_move_optimal_set,
            "best_move_canonical_correct": move == example.best_move_canonical,
            "non_best_exact_correct": False,
        }

    normalized = _normalize_expected_answer(example.task_type, payload)
    exact_match = normalized == example.expected_answer
    return {
        "reward": 1.0 if exact_match else 0.0,
        "json_parse_success": normalized is not None,
        "best_move_set_correct": False,
        "best_move_canonical_correct": False,
        "non_best_exact_correct": exact_match,
    }


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
    best_move_total = 0
    best_move_set_correct = 0
    best_move_canonical_correct = 0
    non_best_total = 0
    non_best_exact_correct = 0

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
            if example.task_type == "best_move":
                best_move_total += 1
                best_move_set_correct += int(bool(outcome["best_move_set_correct"]))
                best_move_canonical_correct += int(bool(outcome["best_move_canonical_correct"]))
            else:
                non_best_total += 1
                non_best_exact_correct += int(bool(outcome["non_best_exact_correct"]))

    return {
        "eval_samples": float(len(examples)),
        "eval_reward_mean": float(np.mean(reward_values)) if reward_values else 0.0,
        "eval_json_parse_rate": float(parse_success) / float(max(1, len(examples))),
        "eval_best_move_set_accuracy": float(best_move_set_correct) / float(max(1, best_move_total)),
        "eval_best_move_canonical_accuracy": float(best_move_canonical_correct) / float(max(1, best_move_total)),
        "eval_exact_accuracy_non_best_move": float(non_best_exact_correct) / float(max(1, non_best_total)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", default=os.environ.get("MOONDREAM_API_KEY"))
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"))
    parser.add_argument("--base-url", default=os.environ.get("TUNA_BASE_URL", "https://api.moondream.ai/v1"))
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="val")
    parser.add_argument("--finetune-name", default=f"ttt-query-{_random_suffix()}")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--reasoning", dest="reasoning", action="store_true")
    parser.add_argument("--no-reasoning", dest="reasoning", action="store_false")
    parser.add_argument("--off-policy", dest="off_policy", action="store_true")
    parser.add_argument("--no-off-policy", dest="off_policy", action="store_false")
    parser.add_argument("--off-policy-mix-ratio", type=float, default=0.5)
    parser.add_argument("--off-policy-buffer-size", type=int, default=4096)
    parser.add_argument("--off-policy-warmup-steps", type=int, default=10)
    parser.add_argument("--off-policy-min-buffer-groups", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--buffer-size", type=int, default=512)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-max-samples", type=int, default=200)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--wandb-project", default="tuna-ttt-query")
    parser.set_defaults(reasoning=True, off_policy=True)
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
        split=args.train_split,
        token=args.hf_token,
        seed=args.seed,
        buffer_size=args.buffer_size,
    )
    eval_examples = _load_eval_examples(
        dataset_name=args.dataset_name,
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
