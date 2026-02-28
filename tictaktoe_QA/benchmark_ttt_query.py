#!/usr/bin/env python3
"""Benchmark query model performance on TicTacToe QA splits."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from PIL import Image

try:
    from tqdm.auto import tqdm  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    class _SimpleTqdm:  # pragma: no cover
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

from tictaktoe_QA import data_loader as dataset_loader  # noqa: E402
from tictaktoe_QA.task_schema import normalize_task_type  # noqa: E402
from tictaktoe_QA import train_ttt_query_rl as train_utils  # noqa: E402

DEFAULT_BASE_URL = "https://api.moondream.ai/v1"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "benchmark_default.json"

BENCHMARK_CONFIG_ALLOWED_KEYS = {
    "api_key",
    "base_url",
    "best_move_optimal_reward",
    "checkpoint_step",
    "config",
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
    "retry_429_backoff_s",
    "retry_429_max_backoff_s",
    "retry_429_max_retries",
    "seed",
    "split",
    "task_types",
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


def _repo_relative(*parts: str) -> Path:
    return Path(__file__).resolve().parent.joinpath(*parts)


def _resolve_config_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path

    from_cwd = (Path.cwd() / path).resolve()
    if from_cwd.exists():
        return from_cwd

    from_script = (_repo_relative(path.as_posix())).resolve()
    if from_script.exists():
        return from_script

    return from_cwd


def _load_json_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        if config_path == DEFAULT_CONFIG_PATH:
            return {}
        raise FileNotFoundError(f"Config file not found: {config_path}")

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a JSON object: {config_path}")
    return payload


def _validate_config_keys(config: dict[str, Any], *, config_path: Path) -> None:
    unknown = sorted(key for key in config.keys() if key not in BENCHMARK_CONFIG_ALLOWED_KEYS)
    if unknown:
        raise ValueError(
            f"Unknown config key(s) in {config_path}: {unknown}. "
            "Remove typos or update script support."
        )


def _cfg_str(config: dict[str, Any], key: str, fallback: str) -> str:
    value = config.get(key, fallback)
    return str(value) if value is not None else fallback


def _cfg_int(config: dict[str, Any], key: str, fallback: int) -> int:
    value = config.get(key, fallback)
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _cfg_float(config: dict[str, Any], key: str, fallback: float) -> float:
    value = config.get(key, fallback)
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _cfg_bool(config: dict[str, Any], key: str, fallback: bool) -> bool:
    value = config.get(key, fallback)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y", "on"}:
            return True
        if lowered in {"false", "0", "no", "n", "off"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return fallback


def _cfg_list_str(config: dict[str, Any], key: str, fallback: list[str]) -> list[str]:
    value = config.get(key)
    if not isinstance(value, list):
        return list(fallback)
    out: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            out.append(text)
    return out or list(fallback)


def _resolve_env_file(env_file: str) -> str:
    path = Path(env_file).expanduser()
    if path.is_absolute():
        return str(path)

    from_cwd = (Path.cwd() / path).resolve()
    if from_cwd.exists():
        return str(from_cwd)

    from_repo = (REPO_ROOT / path).resolve()
    if from_repo.exists():
        return str(from_repo)

    from_script = (_repo_relative(path.as_posix())).resolve()
    if from_script.exists():
        return str(from_script)

    return str(from_cwd)


def _progress_enabled(no_progress: bool) -> bool:
    if no_progress:
        return False
    return sys.stderr.isatty()


def _truncate(text: str, limit: int = 600) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "...<truncated>"


def _build_auth_headers(api_key: str) -> dict[str, str]:
    header_name = os.environ.get("MOONDREAM_AUTH_HEADER", "X-Moondream-Auth")
    user_agent = os.environ.get("MOONDREAM_USER_AGENT") or (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
    key = api_key.strip()
    if header_name.lower() == "authorization" and not key.lower().startswith("bearer "):
        key = f"Bearer {key}"
    return {
        "Content-Type": "application/json",
        "Accept": "application/json",
        header_name: key,
        "User-Agent": user_agent,
    }


def _resolve_model_identifier(
    *,
    model: str,
    finetune_id: str,
    checkpoint_step: Optional[int],
) -> str:
    if model.strip():
        return model.strip()

    ftid = finetune_id.strip()
    if ftid:
        if checkpoint_step is not None:
            return f"moondream3-preview/{ftid}@{checkpoint_step}"
        return f"moondream3-preview/{ftid}"

    return "moondream3-preview"


def _normalize_task_types(raw_values: Optional[list[str]]) -> Optional[list[str]]:
    if not raw_values:
        return None

    out: list[str] = []
    seen: set[str] = set()
    for value in raw_values:
        for piece in str(value).split(","):
            task_type = piece.strip()
            if not task_type:
                continue
            try:
                task_type = normalize_task_type(task_type)
            except ValueError as exc:
                raise ValueError(f"Unknown task_type in --task-types/config: {task_type}") from exc
            if task_type in seen:
                continue
            seen.add(task_type)
            out.append(task_type)

    return out or None


def _extract_answer_text(payload: Any) -> str:
    if isinstance(payload, dict):
        answer = payload.get("answer")
        if isinstance(answer, str):
            return answer

        output = payload.get("output")
        if isinstance(output, dict):
            out_answer = output.get("answer")
            if isinstance(out_answer, str):
                return out_answer

    return ""


def _http_error_details(exc: urllib.error.HTTPError) -> tuple[str, str]:
    request_id = str(exc.headers.get("x-request-id") or exc.headers.get("X-Request-Id") or "")
    body_text = ""
    try:
        body_text = exc.read().decode("utf-8", errors="replace")
    except Exception:
        body_text = ""
    return request_id, body_text


def _build_query_payload(
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
    retry_429_max_retries: int,
    retry_429_backoff_s: float,
    retry_429_max_backoff_s: float,
) -> tuple[str, dict[str, Any], float]:
    payload = _build_query_payload(
        model=model,
        question=question,
        image_url=image_url,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        reasoning=reasoning,
    )

    endpoint = api_base.rstrip("/") + "/query"
    retries = max(0, int(retry_429_max_retries))
    attempt = 0

    while True:
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=_build_auth_headers(api_key),
            method="POST",
        )

        started = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=float(timeout)) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            latency_ms = (time.monotonic() - started) * 1000.0
            data = json.loads(body) if body else {}
            if not isinstance(data, dict):
                data = {}
            return _extract_answer_text(data), data, latency_ms
        except urllib.error.HTTPError as exc:
            latency_ms = (time.monotonic() - started) * 1000.0
            request_id, body_text = _http_error_details(exc)

            retry_after_s = 0.0
            if exc.code == 429:
                retry_after_header = (exc.headers.get("Retry-After") or "").strip()
                if retry_after_header:
                    try:
                        retry_after_s = max(0.0, float(retry_after_header))
                    except (TypeError, ValueError):
                        retry_after_s = 0.0

            if exc.code == 429 and attempt < retries:
                exp_backoff = max(0.0, float(retry_429_backoff_s)) * (2.0**attempt)
                capped_backoff = min(max(0.0, float(retry_429_max_backoff_s)), exp_backoff)
                sleep_s = max(retry_after_s, capped_backoff)
                print(
                    "query retry: "
                    f"status=429 attempt={attempt + 1}/{retries + 1} "
                    f"sleep={sleep_s:.2f}s latency_ms={latency_ms:.1f} "
                    f"request_id={request_id or '-'} body={_truncate(body_text)}"
                )
                if sleep_s > 0.0:
                    time.sleep(sleep_s)
                attempt += 1
                continue

            raise QueryAPIError(
                f"HTTP {exc.code} {exc.reason}",
                status_code=exc.code,
                request_id=request_id,
                response_body=body_text,
            )
        except urllib.error.URLError as exc:
            raise QueryAPIError(f"Network error: {exc}") from exc


def _error_details(exc: Exception) -> str:
    if isinstance(exc, QueryAPIError):
        parts = [f"message={exc}"]
        if exc.status_code is not None:
            parts.append(f"status_code={exc.status_code}")
        if exc.request_id:
            parts.append(f"request_id={exc.request_id}")
        if exc.response_body:
            parts.append(f"response_body={_truncate(exc.response_body)}")
        return " | ".join(parts)
    return f"{type(exc).__name__}: {exc}"


def _build_parser(config: dict[str, Any], config_path: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark TicTacToe query model on QA splits")
    parser.add_argument("--config", default=str(config_path))

    parser.add_argument("--env-file", default=_cfg_str(config, "env_file", str(_repo_relative(".env"))))
    parser.add_argument("--api-key", default=_cfg_str(config, "api_key", ""))
    parser.add_argument("--base-url", default=_cfg_str(config, "base_url", ""))

    parser.add_argument(
        "--dataset-source",
        choices=sorted(dataset_loader.SUPPORTED_DATASET_SOURCES),
        default=_cfg_str(config, "dataset_source", dataset_loader.DEFAULT_DATASET_SOURCE),
        help="Dataset source: HF Hub or local JSONL directory.",
    )
    parser.add_argument(
        "--dataset-dir",
        default=_cfg_str(config, "dataset_dir", str(_repo_relative("synth_dataset/outputs/v2"))),
        help="Local dataset dir (required when --dataset-source=local_jsonl).",
    )
    parser.add_argument(
        "--hf-dataset-repo-id",
        default=_cfg_str(config, "hf_dataset_repo_id", dataset_loader.DEFAULT_HF_DATASET_REPO_ID),
    )
    parser.add_argument(
        "--hf-dataset-revision",
        default=_cfg_str(config, "hf_dataset_revision", dataset_loader.DEFAULT_HF_DATASET_REVISION),
    )
    parser.add_argument("--hf-token", default=_cfg_str(config, "hf_token", ""))
    parser.add_argument("--hf-cache-dir", default=_cfg_str(config, "hf_cache_dir", ""))
    parser.add_argument("--split", default=_cfg_str(config, "split", "test"))
    parser.add_argument("--seed", type=int, default=_cfg_int(config, "seed", 42))
    parser.add_argument(
        "--max-samples",
        type=int,
        default=_cfg_int(config, "max_samples", 0),
        help="Max rows to evaluate; <=0 uses full split.",
    )
    parser.add_argument(
        "--task-types",
        nargs="*",
        default=None,
        help="Optional task filter list; accepts comma-separated values.",
    )

    parser.add_argument("--model", default=_cfg_str(config, "model", ""))
    parser.add_argument("--finetune-id", default=_cfg_str(config, "finetune_id", ""))
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        default=_cfg_int(config, "checkpoint_step", -1),
        help="Optional checkpoint step for model id generation with --finetune-id.",
    )

    parser.add_argument("--temperature", type=float, default=_cfg_float(config, "temperature", 0.0))
    parser.add_argument("--top-p", type=float, default=_cfg_float(config, "top_p", 1.0))
    parser.add_argument("--max-tokens", type=int, default=_cfg_int(config, "max_tokens", 256))
    parser.add_argument("--timeout", type=float, default=_cfg_float(config, "timeout", 60.0))
    reasoning_group = parser.add_mutually_exclusive_group()
    reasoning_group.add_argument(
        "--reasoning",
        dest="reasoning",
        action="store_true",
        help="Enable reasoning mode in /query payloads.",
    )
    reasoning_group.add_argument(
        "--no-reasoning",
        dest="reasoning",
        action="store_false",
        help="Disable reasoning mode in /query payloads.",
    )
    parser.set_defaults(reasoning=_cfg_bool(config, "reasoning", False))

    parser.add_argument("--retry-429-max-retries", type=int, default=_cfg_int(config, "retry_429_max_retries", 2))
    parser.add_argument("--retry-429-backoff-s", type=float, default=_cfg_float(config, "retry_429_backoff_s", 1.0))
    parser.add_argument(
        "--retry-429-max-backoff-s",
        type=float,
        default=_cfg_float(config, "retry_429_max_backoff_s", 8.0),
    )

    parser.add_argument(
        "--best-move-optimal-reward",
        type=float,
        default=_cfg_float(config, "best_move_optimal_reward", 0.7),
    )

    parser.add_argument(
        "--output-json",
        default=_cfg_str(config, "output_json", ""),
        help="Optional metrics output path.",
    )
    parser.add_argument(
        "--predictions-jsonl",
        default=_cfg_str(config, "predictions_jsonl", ""),
        help="Optional predictions output JSONL path.",
    )
    parser.add_argument("--no-progress", action="store_true", default=_cfg_bool(config, "no_progress", False))

    return parser


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    pre_args, _ = pre_parser.parse_known_args(argv)

    config_path = _resolve_config_path(pre_args.config)
    config = _load_json_config(config_path)
    _validate_config_keys(config, config_path=config_path)
    parser = _build_parser(config, config_path)
    args = parser.parse_args(argv)

    args.config = str(_resolve_config_path(args.config))
    args.env_file = _resolve_env_file(args.env_file)
    if args.max_samples <= 0:
        args.max_samples = None

    if int(args.checkpoint_step) < 0:
        args.checkpoint_step = None
    else:
        args.checkpoint_step = int(args.checkpoint_step)

    if args.task_types is None:
        args.task_types = _normalize_task_types(_cfg_list_str(config, "task_types", []))
    else:
        args.task_types = _normalize_task_types(args.task_types)

    return args


def _validate_args(args: argparse.Namespace) -> None:
    args.dataset_source = dataset_loader.normalize_dataset_source(args.dataset_source)
    if not (0.0 <= args.temperature <= 2.0):
        raise ValueError("--temperature must be in [0,2]")
    if not (0.0 < args.top_p <= 1.0):
        raise ValueError("--top-p must be in (0,1]")
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be > 0")
    if args.timeout <= 0.0:
        raise ValueError("--timeout must be > 0")
    if args.retry_429_max_retries < 0:
        raise ValueError("--retry-429-max-retries must be >= 0")
    if args.retry_429_backoff_s < 0.0:
        raise ValueError("--retry-429-backoff-s must be >= 0")
    if args.retry_429_max_backoff_s < 0.0:
        raise ValueError("--retry-429-max-backoff-s must be >= 0")
    if not (0.0 <= args.best_move_optimal_reward <= 1.0):
        raise ValueError("--best-move-optimal-reward must be in [0,1]")
    if args.checkpoint_step is not None and args.checkpoint_step < 0:
        raise ValueError("--checkpoint-step must be >= 0")
    if args.dataset_source == "local_jsonl" and not str(args.dataset_dir).strip():
        raise ValueError("--dataset-dir is required when --dataset-source=local_jsonl")
    if args.dataset_source == "hf_hub" and not str(args.hf_dataset_repo_id).strip():
        raise ValueError("--hf-dataset-repo-id is required when --dataset-source=hf_hub")



def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)
    load_dotenv(args.env_file, override=False)

    if not args.api_key:
        args.api_key = os.environ.get("MOONDREAM_API_KEY", "")
    if not args.base_url:
        args.base_url = (
            os.environ.get("MOONDREAM_BASE_URL")
            or os.environ.get("TUNA_BASE_URL")
            or DEFAULT_BASE_URL
        )

    _validate_args(args)
    if not args.api_key:
        raise ValueError("MOONDREAM_API_KEY is required")
    args.hf_token = dataset_loader.resolve_hf_token(args.hf_token)

    model = _resolve_model_identifier(
        model=args.model,
        finetune_id=args.finetune_id,
        checkpoint_step=args.checkpoint_step,
    )

    dataset_dir: Optional[Path] = None
    if args.dataset_source == "local_jsonl":
        dataset_dir = Path(args.dataset_dir).expanduser().resolve()

    examples = train_utils._load_split_examples(
        split_name=args.split,
        dataset_source=args.dataset_source,
        dataset_dir=dataset_dir,
        hf_dataset_repo_id=args.hf_dataset_repo_id,
        hf_dataset_revision=args.hf_dataset_revision,
        hf_token=args.hf_token,
        hf_cache_dir=args.hf_cache_dir,
    )

    rng = random.Random(args.seed)
    if args.task_types:
        allowed_tasks = set(args.task_types)
        indices = [idx for idx, item in enumerate(examples) if item.task_type in allowed_tasks]
    else:
        indices = list(range(len(examples)))
    rng.shuffle(indices)
    if args.max_samples is not None:
        indices = indices[: args.max_samples]

    show_progress = _progress_enabled(args.no_progress)

    total_scored = 0
    reward_sum = 0.0
    object_parse_count = 0
    parse_success_count = 0
    request_failure_count = 0

    best_move_total = 0
    best_move_set_correct = 0
    best_move_canonical_correct = 0

    non_best_total = 0
    non_best_exact_correct = 0

    per_task_total: Counter[str] = Counter()
    per_task_correct: Counter[str] = Counter()

    latency_values_ms: list[float] = []

    predictions_handle = None
    predictions_path = Path(args.predictions_jsonl).expanduser().resolve() if args.predictions_jsonl else None
    if predictions_path:
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        predictions_handle = predictions_path.open("w", encoding="utf-8")

    try:
        progress = tqdm(
            indices,
            desc=f"benchmark:{args.split}",
            total=len(indices),
            dynamic_ncols=True,
            disable=not show_progress,
        )

        for idx in progress:
            item = examples[idx]
            per_task_total[item.task_type] += 1

            try:
                image = Image.open(item.image_path).convert("RGB")
            except (FileNotFoundError, OSError) as exc:
                request_failure_count += 1
                print(f"row={item.row_id}: image load failed ({exc}); skipping")
                continue

            try:
                answer_text, raw_response, latency_ms = _call_query_api(
                    api_base=args.base_url,
                    api_key=args.api_key,
                    model=model,
                    question=item.question,
                    image_url=train_utils._to_data_url(image, quality=92),
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens,
                    reasoning=args.reasoning,
                    timeout=args.timeout,
                    retry_429_max_retries=args.retry_429_max_retries,
                    retry_429_backoff_s=args.retry_429_backoff_s,
                    retry_429_max_backoff_s=args.retry_429_max_backoff_s,
                )
            except Exception as exc:
                request_failure_count += 1
                print(f"row={item.row_id}: query failed. details: {_error_details(exc)}")
                if predictions_handle is not None:
                    predictions_handle.write(
                        json.dumps(
                            {
                                "row_id": item.row_id,
                                "split": item.split,
                                "task_type": item.task_type,
                                "status": "request_error",
                                "error": _error_details(exc),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                continue

            latency_values_ms.append(latency_ms)
            pred_payload = train_utils._parse_prediction_json(answer_text)
            outcome = train_utils._score_payload_for_example(
                item,
                pred_payload,
                best_move_optimal_reward=args.best_move_optimal_reward,
            )

            total_scored += 1
            reward_sum += float(outcome.reward)
            if outcome.json_object_parsed:
                object_parse_count += 1
            if outcome.parse_success:
                parse_success_count += 1
            if outcome.task_correct:
                per_task_correct[item.task_type] += 1

            if item.task_type == "best_move":
                best_move_total += 1
                if outcome.best_move_set_correct:
                    best_move_set_correct += 1
                if outcome.best_move_canonical_correct:
                    best_move_canonical_correct += 1
            else:
                non_best_total += 1
                if outcome.exact_non_best_correct:
                    non_best_exact_correct += 1

            if predictions_handle is not None:
                predictions_handle.write(
                    json.dumps(
                        {
                            "row_id": item.row_id,
                            "split": item.split,
                            "task_type": item.task_type,
                            "question": item.question,
                            "answer": answer_text,
                            "prediction_json": pred_payload,
                            "json_object_parsed": outcome.json_object_parsed,
                            "parse_success": outcome.parse_success,
                            "reward": outcome.reward,
                            "task_correct": outcome.task_correct,
                            "best_move_set_correct": outcome.best_move_set_correct,
                            "best_move_canonical_correct": outcome.best_move_canonical_correct,
                            "exact_non_best_correct": outcome.exact_non_best_correct,
                            "latency_ms": latency_ms,
                            "raw_response": raw_response,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            if show_progress and total_scored % 10 == 0:
                parse_rate = parse_success_count / max(1, total_scored)
                reward_mean = reward_sum / max(1, total_scored)
                progress.set_postfix(
                    scored=total_scored,
                    parse=f"{parse_rate:.3f}",
                    reward=f"{reward_mean:.3f}",
                    fails=request_failure_count,
                )
    finally:
        if predictions_handle is not None:
            predictions_handle.close()

    avg_latency_ms = sum(latency_values_ms) / max(1, len(latency_values_ms))
    p95_latency_ms = 0.0
    if latency_values_ms:
        sorted_lat = sorted(latency_values_ms)
        p95_idx = int(round(0.95 * (len(sorted_lat) - 1)))
        p95_latency_ms = sorted_lat[p95_idx]

    metrics: dict[str, Any] = {
        "model": model,
        "finetune_id": str(args.finetune_id or ""),
        "checkpoint_step": int(args.checkpoint_step) if args.checkpoint_step is not None else -1,
        "config": args.config,
        "split": args.split,
        "dataset_source": args.dataset_source,
        "dataset_dir": str(dataset_dir) if dataset_dir is not None else "",
        "hf_dataset_repo_id": args.hf_dataset_repo_id,
        "hf_dataset_revision": args.hf_dataset_revision,
        "reasoning": bool(args.reasoning),
        "task_types": list(args.task_types or []),
        "requested_rows": len(indices),
        "evaluated_rows": total_scored,
        "request_failures": request_failure_count,
        "json_object_fail": (total_scored - object_parse_count),
        "json_parse_fail": (total_scored - parse_success_count),
        "eval_reward_mean": (reward_sum / max(1, total_scored)),
        "eval_json_object_rate": (object_parse_count / max(1, total_scored)),
        "eval_json_parse_rate": (parse_success_count / max(1, total_scored)),
        "eval_best_move_set_accuracy": (best_move_set_correct / max(1, best_move_total)),
        "eval_best_move_canonical_accuracy": (
            best_move_canonical_correct / max(1, best_move_total)
        ),
        "eval_exact_accuracy_non_best_move": (
            non_best_exact_correct / max(1, non_best_total)
        ),
        "latency_avg_ms": avg_latency_ms,
        "latency_p95_ms": p95_latency_ms,
        "by_task": {
            task: {
                "count": int(per_task_total[task]),
                "correct": int(per_task_correct[task]),
                "accuracy": (per_task_correct[task] / max(1, per_task_total[task])),
            }
            for task in sorted(per_task_total.keys())
        },
    }

    print("benchmark metrics:")
    print(json.dumps(metrics, indent=2, sort_keys=True))

    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
        print(f"wrote metrics JSON: {output_path}")

    if predictions_path:
        print(f"wrote predictions JSONL: {predictions_path}")


if __name__ == "__main__":
    main()
