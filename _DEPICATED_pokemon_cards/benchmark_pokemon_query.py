#!/usr/bin/env python3
"""Benchmark PokemonCards query models against local or HF splits."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    from tqdm.auto import tqdm  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    class _SimpleTqdm:
        def __init__(self, iterable, *args: Any, **kwargs: Any) -> None:
            self._iterable = iterable

        def __iter__(self):
            return iter(self._iterable)

        def set_postfix(self, *args: Any, **kwargs: Any) -> None:
            return

    def tqdm(iterable=None, *args, **kwargs):  # type: ignore
        if iterable is None:
            iterable = []
        return _SimpleTqdm(iterable, *args, **kwargs)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from _DEPICATED_pokemon_cards import data_loader  # noqa: E402
from _DEPICATED_pokemon_cards.common import (  # noqa: E402
    cfg_bool,
    cfg_float,
    cfg_int,
    cfg_str,
    ensure_parent_dir,
    image_path_to_data_url,
    load_dotenv_if_available,
    load_json_object,
    progress_enabled,
    resolve_path,
    truncate_text,
    validate_config_keys,
    write_json,
)
from _DEPICATED_pokemon_cards.scoring import (  # noqa: E402
    answer_reward_for_task,
    combined_reward,
    parse_prediction_json,
    rationale_reward_from_texts,
    set_f1,
)
from _DEPICATED_pokemon_cards.task_schema import normalize_answer_for_task, normalize_task_type  # noqa: E402

DEFAULT_BASE_URL = "https://api.moondream.ai/v1"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "benchmark_default.json"
BENCHMARK_CONFIG_ALLOWED_KEYS = {
    "api_key",
    "api_key_env_var",
    "base_url",
    "checkpoint_step",
    "dataset_dir",
    "dataset_source",
    "env_file",
    "finetune_id",
    "hf_cache_dir",
    "hf_dataset_repo_id",
    "hf_dataset_revision",
    "hf_token",
    "max_samples",
    "max_tokens",
    "model",
    "no_progress",
    "output_json",
    "predictions_jsonl",
    "reasoning",
    "seed",
    "split",
    "temperature",
    "timeout",
    "top_p",
}


class QueryAPIError(Exception):
    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        request_id: str = "",
        response_body: str = "",
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.request_id = request_id
        self.response_body = response_body


def _build_auth_headers(api_key: str) -> dict[str, str]:
    header_name = os.environ.get("MOONDREAM_AUTH_HEADER", "X-Moondream-Auth")
    key = api_key.strip()
    if header_name.lower() == "authorization" and not key.lower().startswith("bearer "):
        key = f"Bearer {key}"
    return {
        "Content-Type": "application/json",
        "Accept": "application/json",
        header_name: key,
        "User-Agent": "pokemon-cards-benchmark/0.1",
    }


def _resolve_model_identifier(*, model: str, finetune_id: str, checkpoint_step: Optional[int]) -> str:
    if str(model or "").strip():
        return str(model).strip()
    ftid = str(finetune_id or "").strip()
    if ftid:
        if checkpoint_step is not None:
            return f"moondream3-preview/{ftid}@{checkpoint_step}"
        return f"moondream3-preview/{ftid}"
    return "moondream3-preview"


def _build_request_payload(
    *,
    model: str,
    question: str,
    image_url: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    reasoning: Optional[bool],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "question": question,
        "image_url": image_url,
        "settings": {
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": int(max_tokens),
        },
    }
    if reasoning is not None:
        payload["reasoning"] = bool(reasoning)
    return payload


def _call_query_api(
    *,
    api_base: str,
    api_key: str,
    model: str,
    question: str,
    image_url: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    reasoning: Optional[bool],
    timeout: float,
) -> tuple[dict[str, Any], float]:
    payload = _build_request_payload(
        model=model,
        question=question,
        image_url=image_url,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        reasoning=reasoning,
    )
    endpoint = api_base.rstrip("/") + "/query"
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=_build_auth_headers(api_key),
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=float(timeout)) as response:
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        request_id = exc.headers.get("x-request-id", "")
        body_text = exc.read().decode("utf-8", errors="replace")
        raise QueryAPIError(
            f"HTTP {exc.code} {exc.reason}",
            status_code=exc.code,
            request_id=request_id,
            response_body=body_text,
        ) from exc
    except urllib.error.URLError as exc:
        raise QueryAPIError(f"Network error: {exc}") from exc

    latency_ms = (time.monotonic() - started) * 1000.0
    data = json.loads(body) if body else {}
    if not isinstance(data, dict):
        data = {}
    return data, latency_ms


@dataclass(frozen=True)
class QAExample:
    row_id: str
    task_type: str
    question: str
    image_path: Path
    expected_answer: dict[str, Any]
    teacher_rationale_text: str
    source_metadata: dict[str, Any]


def _load_json_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        if config_path == DEFAULT_CONFIG_PATH:
            return {}
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return load_json_object(config_path)


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    pre_args, _ = pre_parser.parse_known_args(argv)
    config_path = resolve_path(pre_args.config, search_roots=(REPO_ROOT, Path(__file__).resolve().parent))
    config = _load_json_config(config_path)
    validate_config_keys(config, allowed_keys=BENCHMARK_CONFIG_ALLOWED_KEYS, config_path=config_path)

    parser = argparse.ArgumentParser(description="Benchmark PokemonCards query model.")
    parser.add_argument("--config", default=str(config_path))
    parser.add_argument("--env-file", default=cfg_str(config, "env_file", ".env"))
    parser.add_argument("--api-key", default=cfg_str(config, "api_key", ""))
    parser.add_argument("--api-key-env-var", default=cfg_str(config, "api_key_env_var", "MOONDREAM_API_KEY"))
    parser.add_argument("--base-url", default=cfg_str(config, "base_url", DEFAULT_BASE_URL))
    parser.add_argument("--dataset-source", choices=sorted(data_loader.SUPPORTED_DATASET_SOURCES), default=cfg_str(config, "dataset_source", "local_jsonl"))
    parser.add_argument("--dataset-dir", default=cfg_str(config, "dataset_dir", "pokemon_cards/outputs/thefusion21_pokemoncards_v1"))
    parser.add_argument("--hf-dataset-repo-id", default=cfg_str(config, "hf_dataset_repo_id", ""))
    parser.add_argument("--hf-dataset-revision", default=cfg_str(config, "hf_dataset_revision", "main"))
    parser.add_argument("--hf-token", default=cfg_str(config, "hf_token", ""))
    parser.add_argument("--hf-cache-dir", default=cfg_str(config, "hf_cache_dir", ""))
    parser.add_argument("--split", default=cfg_str(config, "split", "test"))
    parser.add_argument("--model", default=cfg_str(config, "model", ""))
    parser.add_argument("--finetune-id", default=cfg_str(config, "finetune_id", ""))
    parser.add_argument("--checkpoint-step", type=int, default=cfg_int(config, "checkpoint_step", -1))
    parser.add_argument("--temperature", type=float, default=cfg_float(config, "temperature", 0.0))
    parser.add_argument("--top-p", type=float, default=cfg_float(config, "top_p", 1.0))
    parser.add_argument("--max-tokens", type=int, default=cfg_int(config, "max_tokens", 256))
    parser.add_argument("--timeout", type=float, default=cfg_float(config, "timeout", 90.0))
    parser.add_argument("--reasoning", dest="reasoning", action="store_true")
    parser.add_argument("--no-reasoning", dest="reasoning", action="store_false")
    parser.set_defaults(reasoning=cfg_bool(config, "reasoning", False))
    parser.add_argument("--max-samples", type=int, default=cfg_int(config, "max_samples", 0))
    parser.add_argument("--seed", type=int, default=cfg_int(config, "seed", 42))
    parser.add_argument("--output-json", default=cfg_str(config, "output_json", ""))
    parser.add_argument("--predictions-jsonl", default=cfg_str(config, "predictions_jsonl", ""))
    parser.add_argument("--no-progress", action="store_true", default=cfg_bool(config, "no_progress", False))
    args = parser.parse_args(argv)
    args.dataset_dir = resolve_path(args.dataset_dir, search_roots=(REPO_ROOT,))
    args.env_file = resolve_path(args.env_file, search_roots=(REPO_ROOT, Path(__file__).resolve().parent))
    return args


def _resolve_api_key(args: argparse.Namespace) -> str:
    key = str(getattr(args, "api_key", "") or "").strip()
    if key:
        return key
    env_name = str(getattr(args, "api_key_env_var", "") or "MOONDREAM_API_KEY").strip() or "MOONDREAM_API_KEY"
    env_key = os.environ.get(env_name, "").strip()
    if env_key:
        return env_key
    env_key = os.environ.get("MOONDREAM_API_KEY", "").strip()
    if env_key:
        return env_key
    raise ValueError("MOONDREAM_API_KEY is required")


def _load_examples(args: argparse.Namespace) -> list[QAExample]:
    rows = data_loader.load_split_rows(
        dataset_source=args.dataset_source,
        split_name=str(args.split),
        dataset_dir=Path(args.dataset_dir),
        hf_dataset_repo_id=str(args.hf_dataset_repo_id),
        hf_dataset_revision=str(args.hf_dataset_revision),
        hf_token=str(args.hf_token),
        hf_cache_dir=str(args.hf_cache_dir),
    )
    examples: list[QAExample] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        task_type = normalize_task_type(str(row.get("task_type", "")))
        expected = normalize_answer_for_task(task_type, json.loads(str(row.get("final_answer_json", "{}"))))
        if expected is None:
            continue
        image_path = (Path(args.dataset_dir) / str(row.get("image_path", ""))).resolve()
        if not image_path.is_file():
            continue
        source_meta = {}
        source_meta_raw = str(row.get("source_metadata_json", "") or "")
        if source_meta_raw:
            try:
                source_meta = json.loads(source_meta_raw)
            except json.JSONDecodeError:
                source_meta = {}
        examples.append(
            QAExample(
                row_id=str(row.get("row_id", "")),
                task_type=task_type,
                question=str(row.get("question", "")),
                image_path=image_path,
                expected_answer=expected,
                teacher_rationale_text=str(row.get("teacher_rationale_text", "") or ""),
                source_metadata=source_meta if isinstance(source_meta, dict) else {},
            )
        )
    if int(args.max_samples) > 0 and len(examples) > int(args.max_samples):
        rng = random.Random(int(args.seed))
        examples = rng.sample(examples, k=int(args.max_samples))
    return examples


def _score_example(
    *,
    example: QAExample,
    response_payload: dict[str, Any],
    use_reasoning_reward: bool,
) -> dict[str, Any]:
    answer_text = str(response_payload.get("answer", "") or "")
    reasoning_payload = response_payload.get("reasoning")
    reasoning_text = ""
    if isinstance(reasoning_payload, dict):
        reasoning_text = str(reasoning_payload.get("text", "") or "")
    elif isinstance(reasoning_payload, str):
        reasoning_text = reasoning_payload

    pred_payload = parse_prediction_json(answer_text)
    normalized_pred = normalize_answer_for_task(example.task_type, pred_payload) if pred_payload is not None else None
    answer_reward = (
        answer_reward_for_task(
            example.task_type,
            pred_payload,
            example.expected_answer,
            expected_metadata=example.source_metadata,
        )
        if pred_payload is not None
        else 0.0
    )
    rationale_reward = (
        rationale_reward_from_texts(example.task_type, reasoning_text, example.teacher_rationale_text)
        if use_reasoning_reward and example.teacher_rationale_text
        else 0.0
    )
    total_reward = combined_reward(answer_reward, rationale_reward, use_reasoning_reward=use_reasoning_reward)

    attack_name_f1 = None
    if example.task_type == "attack_overview" and normalized_pred is not None:
        attack_name_f1 = set_f1(
            {item.lower() for item in normalized_pred.get("attack_names", [])},
            {item.lower() for item in example.expected_answer.get("attack_names", [])},
        )

    return {
        "pred_payload": normalized_pred,
        "predicted_answer_text": answer_text,
        "predicted_reasoning_text": reasoning_text,
        "json_parse_success": pred_payload is not None,
        "answer_reward": float(answer_reward),
        "rationale_reward": float(rationale_reward),
        "combined_reward": float(total_reward),
        "attack_name_f1": attack_name_f1,
        "identity_exact": bool(example.task_type == "card_identity" and float(answer_reward) == 1.0),
        "summary_exact": bool(example.task_type == "card_summary" and float(answer_reward) == 1.0),
    }


def run_benchmark(
    *,
    args: argparse.Namespace,
    call_api_fn=_call_query_api,
) -> dict[str, Any]:
    load_dotenv_if_available(args.env_file)
    api_key = _resolve_api_key(args)
    model = _resolve_model_identifier(
        model=str(args.model),
        finetune_id=str(args.finetune_id),
        checkpoint_step=(None if int(args.checkpoint_step) < 0 else int(args.checkpoint_step)),
    )
    examples = _load_examples(args)

    answer_rewards: list[float] = []
    rationale_rewards: list[float] = []
    combined_rewards: list[float] = []
    latencies_ms: list[float] = []
    parse_success = 0
    identity_total = 0
    identity_exact = 0
    summary_total = 0
    summary_exact = 0
    summary_rewards: list[float] = []
    core_rewards: list[float] = []
    attack_name_f1_values: list[float] = []
    task_counts: Counter[str] = Counter()
    task_reward_values: dict[str, list[float]] = defaultdict(list)
    predictions: list[dict[str, Any]] = []

    progress = tqdm(examples, desc="benchmark", disable=not progress_enabled(bool(args.no_progress)))
    for example in progress:
        task_counts[example.task_type] += 1
        response_payload: dict[str, Any]
        latency_ms = 0.0
        error_text = ""
        try:
            response_payload, latency_ms = call_api_fn(
                api_base=str(args.base_url),
                api_key=api_key,
                model=model,
                question=example.question,
                image_url=image_path_to_data_url(example.image_path),
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                max_tokens=int(args.max_tokens),
                reasoning=bool(args.reasoning),
                timeout=float(args.timeout),
            )
        except QueryAPIError as exc:
            response_payload = {}
            error_text = f"{exc} | request_id={exc.request_id} | body={truncate_text(exc.response_body)}"

        outcome = _score_example(
            example=example,
            response_payload=response_payload,
            use_reasoning_reward=bool(args.reasoning),
        )
        answer_rewards.append(float(outcome["answer_reward"]))
        rationale_rewards.append(float(outcome["rationale_reward"]))
        combined_rewards.append(float(outcome["combined_reward"]))
        latencies_ms.append(float(latency_ms))
        parse_success += int(bool(outcome["json_parse_success"]))
        task_reward_values[example.task_type].append(float(outcome["answer_reward"]))

        if example.task_type == "card_identity":
            identity_total += 1
            identity_exact += int(bool(outcome["identity_exact"]))
        elif example.task_type == "card_core":
            core_rewards.append(float(outcome["answer_reward"]))
        elif example.task_type == "attack_overview" and outcome["attack_name_f1"] is not None:
            attack_name_f1_values.append(float(outcome["attack_name_f1"]))
        elif example.task_type == "card_summary":
            summary_total += 1
            summary_exact += int(bool(outcome["summary_exact"]))
            summary_rewards.append(float(outcome["answer_reward"]))

        predictions.append(
            {
                "row_id": example.row_id,
                "task_type": example.task_type,
                "question": example.question,
                "expected_answer": example.expected_answer,
                "teacher_rationale_text": example.teacher_rationale_text,
                "predicted_answer_payload": outcome["pred_payload"],
                "predicted_answer_text": outcome["predicted_answer_text"],
                "predicted_reasoning_text": outcome["predicted_reasoning_text"],
                "answer_reward": float(outcome["answer_reward"]),
                "rationale_reward": float(outcome["rationale_reward"]),
                "combined_reward": float(outcome["combined_reward"]),
                "json_parse_success": bool(outcome["json_parse_success"]),
                "latency_ms": float(latency_ms),
                "error": error_text,
            }
        )

    metrics: dict[str, Any] = {
        "model": model,
        "split": str(args.split),
        "samples": len(examples),
        "answer_reward_mean": float(np.mean(answer_rewards)) if answer_rewards else 0.0,
        "rationale_reward_mean": float(np.mean(rationale_rewards)) if rationale_rewards else 0.0,
        "combined_reward_mean": float(np.mean(combined_rewards)) if combined_rewards else 0.0,
        "json_parse_rate": float(parse_success) / float(max(1, len(examples))),
        "latency_ms_mean": float(np.mean(latencies_ms)) if latencies_ms else 0.0,
        "identity_exact_accuracy": float(identity_exact) / float(max(1, identity_total)),
        "core_reward_mean": float(np.mean(core_rewards)) if core_rewards else 0.0,
        "attack_name_f1": float(np.mean(attack_name_f1_values)) if attack_name_f1_values else 0.0,
        "summary_reward_mean": float(np.mean(summary_rewards)) if summary_rewards else 0.0,
        "summary_full_credit_rate": float(summary_exact) / float(max(1, summary_total)),
        "summary_exact_accuracy": float(summary_exact) / float(max(1, summary_total)),
        "task_counts": dict(sorted(task_counts.items())),
        "task_answer_reward_mean": {
            task_type: (float(np.mean(values)) if values else 0.0)
            for task_type, values in sorted(task_reward_values.items())
        },
    }

    if args.predictions_jsonl:
        predictions_path = Path(args.predictions_jsonl).expanduser().resolve()
        ensure_parent_dir(predictions_path)
        with predictions_path.open("w", encoding="utf-8") as handle:
            for item in predictions:
                handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    if args.output_json:
        write_json(Path(args.output_json).expanduser().resolve(), metrics)
    return metrics


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)
    metrics = run_benchmark(args=args)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
