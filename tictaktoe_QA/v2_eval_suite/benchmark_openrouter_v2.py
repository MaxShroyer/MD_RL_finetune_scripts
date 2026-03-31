#!/usr/bin/env python3
"""Benchmark V2 TicTacToe tasks against OpenRouter chat-completions models."""

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
from typing import Any, Callable, Optional

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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tictaktoe_QA import data_loader as dataset_loader  # noqa: E402
from tictaktoe_QA import train_ttt_query_rl as train_utils  # noqa: E402
from tictaktoe_QA.v2_eval_suite import common  # noqa: E402

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "openrouter_v2_quick.json"
DEFAULT_MODELS: list[dict[str, Any]] = [
    {"label": "claude_frontier", "model_id": "anthropic/claude-3.7-sonnet"},
    {"label": "chatgpt_frontier", "model_id": "openai/gpt-5.1"},
    {"label": "qwen_vl_frontier", "model_id": "qwen/qwen-2.5-vl-7b-instruct"},
]
SAMPLE_STRATEGIES = ("random", "stratified")
STRATIFY_SUPPORTED_TASKS = {"turn_player", "available_moves_count"}

OPENROUTER_CONFIG_ALLOWED_KEYS = {
    "api_base",
    "api_key",
    "best_move_optimal_reward",
    "dataset_dir",
    "dataset_source",
    "env_file",
    "hf_cache_dir",
    "hf_dataset_repo_id",
    "hf_dataset_revision",
    "hf_token",
    "max_samples_per_task",
    "max_tokens",
    "models",
    "retry_429_backoff_s",
    "retry_429_max_backoff_s",
    "retry_429_max_retries",
    "sample_strategy",
    "seed",
    "split",
    "stratify_tasks",
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


def _progress_enabled(no_progress: bool) -> bool:
    if no_progress:
        return False
    return sys.stderr.isatty()


def _resolve_config_path(raw_path: str) -> Path:
    return common.resolve_path(raw_path, search_roots=(common.repo_root(), common.package_root()))


def _resolve_env_file(raw_path: str) -> Path:
    return common.resolve_path(raw_path, search_roots=(common.repo_root(), common.package_root()))


def _load_json_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return common.load_json_object(config_path)


def _parse_models(raw_models: Any) -> list[dict[str, Any]]:
    if raw_models is None:
        return [dict(item) for item in DEFAULT_MODELS]

    payload = raw_models
    if isinstance(raw_models, str):
        text = raw_models.strip()
        if not text:
            return [dict(item) for item in DEFAULT_MODELS]
        payload = json.loads(text)

    out: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        for label, model_id in payload.items():
            label_text = str(label).strip()
            model_text = str(model_id).strip()
            if label_text and model_text:
                out.append({"label": label_text, "model_id": model_text})
    elif isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                continue
            label_text = str(item.get("label", "")).strip()
            model_text = str(item.get("model_id", "")).strip()
            if label_text and model_text:
                entry: dict[str, Any] = {"label": label_text, "model_id": model_text}
                request_overrides = item.get("request_overrides")
                if isinstance(request_overrides, dict):
                    entry["request_overrides"] = dict(request_overrides)
                out.append(entry)

    if not out:
        raise ValueError("models must be a non-empty list of {label, model_id}")
    return out


def _build_parser(config: dict[str, Any], config_path: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark TicTacToe V2 tasks via OpenRouter")
    parser.add_argument("--config", required=True, default=str(config_path))
    parser.add_argument("--out-dir", default="tictaktoe_QA/v2_eval_suite/outputs/openrouter")

    parser.add_argument("--env-file", default=common.cfg_str(config, "env_file", "tictaktoe_QA/.env"))
    parser.add_argument("--api-key", default=common.cfg_str(config, "api_key", ""))
    parser.add_argument("--api-base", default=common.cfg_str(config, "api_base", "https://openrouter.ai/api/v1"))

    parser.add_argument(
        "--dataset-source",
        choices=sorted(dataset_loader.SUPPORTED_DATASET_SOURCES),
        default=common.cfg_str(config, "dataset_source", "local_jsonl"),
    )
    parser.add_argument(
        "--dataset-dir",
        default=common.cfg_str(config, "dataset_dir", "tictaktoe_QA/synth_dataset/outputs/v2"),
    )
    parser.add_argument(
        "--hf-dataset-repo-id",
        default=common.cfg_str(config, "hf_dataset_repo_id", common.V2_HF_DATASET_REPO_ID),
    )
    parser.add_argument(
        "--hf-dataset-revision",
        default=common.cfg_str(config, "hf_dataset_revision", dataset_loader.DEFAULT_HF_DATASET_REVISION),
    )
    parser.add_argument("--hf-token", default=common.cfg_str(config, "hf_token", ""))
    parser.add_argument("--hf-cache-dir", default=common.cfg_str(config, "hf_cache_dir", ""))

    parser.add_argument("--split", default=common.cfg_str(config, "split", "test"))
    parser.add_argument("--task-types", nargs="*", default=None)
    parser.add_argument(
        "--max-samples-per-task",
        type=int,
        default=common.cfg_int(config, "max_samples_per_task", 80),
        help="0 means full rows per selected task.",
    )
    parser.add_argument("--seed", type=int, default=common.cfg_int(config, "seed", 42))
    parser.add_argument(
        "--sample-strategy",
        choices=list(SAMPLE_STRATEGIES),
        default=common.cfg_str(config, "sample_strategy", "random"),
    )
    parser.add_argument(
        "--stratify-tasks",
        nargs="*",
        default=None,
        help=(
            "Task types to stratify when sample_strategy=stratified. "
            "Supported tasks: turn_player, available_moves_count."
        ),
    )

    parser.add_argument("--temperature", type=float, default=common.cfg_float(config, "temperature", 0.0))
    parser.add_argument("--top-p", type=float, default=common.cfg_float(config, "top_p", 1.0))
    parser.add_argument("--max-tokens", type=int, default=common.cfg_int(config, "max_tokens", 256))
    parser.add_argument("--timeout", type=float, default=common.cfg_float(config, "timeout", 90.0))

    parser.add_argument("--retry-429-max-retries", type=int, default=common.cfg_int(config, "retry_429_max_retries", 2))
    parser.add_argument("--retry-429-backoff-s", type=float, default=common.cfg_float(config, "retry_429_backoff_s", 1.0))
    parser.add_argument(
        "--retry-429-max-backoff-s",
        type=float,
        default=common.cfg_float(config, "retry_429_max_backoff_s", 8.0),
    )

    parser.add_argument(
        "--best-move-optimal-reward",
        type=float,
        default=common.cfg_float(config, "best_move_optimal_reward", 0.7),
    )

    parser.add_argument(
        "--models-json",
        default="",
        help="JSON override for models list. Accepts list[{label,model_id,request_overrides?}] or {label:model_id}.",
    )
    parser.add_argument("--no-progress", action="store_true", default=False)
    return parser


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    pre_args, _ = pre_parser.parse_known_args(argv)

    config_path = _resolve_config_path(pre_args.config)
    config = _load_json_config(config_path)
    common.validate_config_keys(config, allowed_keys=OPENROUTER_CONFIG_ALLOWED_KEYS, config_path=config_path)

    parser = _build_parser(config, config_path)
    args = parser.parse_args(argv)

    args.config = str(_resolve_config_path(args.config))
    args.env_file = str(_resolve_env_file(args.env_file))
    args.out_dir = common.resolve_path(args.out_dir, search_roots=(common.repo_root(),))
    args.dataset_source = dataset_loader.normalize_dataset_source(args.dataset_source)
    if args.dataset_source == "local_jsonl":
        args.dataset_dir = common.resolve_path(args.dataset_dir, search_roots=(common.repo_root(),))
    else:
        args.dataset_dir = common.resolve_path(args.dataset_dir, search_roots=(common.repo_root(),))

    if args.task_types is None:
        default_task_types = common.cfg_list_str(config, "task_types", list(common.HARD_TASK_TYPES))
        args.task_types = common.normalize_task_types(default_task_types)
    else:
        args.task_types = common.normalize_task_types(args.task_types)
    if args.stratify_tasks is None:
        default_stratify_tasks = common.cfg_list_str(config, "stratify_tasks", [])
        args.stratify_tasks = common.normalize_task_types(default_stratify_tasks)
    else:
        args.stratify_tasks = common.normalize_task_types(args.stratify_tasks)

    models_from_config = config.get("models")
    args.models = _parse_models(models_from_config)
    if str(args.models_json).strip():
        args.models = _parse_models(args.models_json)

    args.hf_token = dataset_loader.resolve_hf_token(args.hf_token)
    return args


def _validate_args(args: argparse.Namespace) -> None:
    if args.max_samples_per_task < 0:
        raise ValueError("--max-samples-per-task must be >= 0")
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
    if not args.task_types:
        raise ValueError("At least one task_type is required")


def _build_auth_headers(api_key: str) -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key.strip()}",
    }
    referer = os.environ.get("OPENROUTER_HTTP_REFERER", "").strip()
    if referer:
        headers["HTTP-Referer"] = referer
    title = os.environ.get("OPENROUTER_APP_NAME", "").strip()
    if title:
        headers["X-Title"] = title
    return headers


def _resolve_openrouter_api_key(cli_api_key: str) -> str:
    cli_key = str(cli_api_key or "").strip()
    if cli_key:
        return cli_key

    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if openrouter_key:
        return openrouter_key

    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if openai_key:
        raise ValueError(
            "OPENROUTER_API_KEY is required for OpenRouter benchmarks. "
            "OPENAI_API_KEY was found but is not valid for OpenRouter auth."
        )
    raise ValueError("OPENROUTER_API_KEY is required")


def _http_error_details(exc: urllib.error.HTTPError) -> tuple[str, str]:
    request_id = str(exc.headers.get("x-request-id") or exc.headers.get("X-Request-Id") or "")
    body_text = ""
    try:
        body_text = exc.read().decode("utf-8", errors="replace")
    except Exception:
        body_text = ""
    return request_id, body_text


def _preflight_openrouter_auth(
    *,
    api_base: str,
    api_key: str,
    timeout: float,
) -> None:
    endpoint = api_base.rstrip("/") + "/models"
    req = urllib.request.Request(
        endpoint,
        headers=_build_auth_headers(api_key),
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=float(timeout)) as resp:
            _ = resp.read()
    except urllib.error.HTTPError as exc:
        request_id, body_text = _http_error_details(exc)
        raise QueryAPIError(
            f"HTTP {exc.code} {exc.reason}",
            status_code=exc.code,
            request_id=request_id,
            response_body=body_text,
        ) from exc
    except urllib.error.URLError as exc:
        raise QueryAPIError(f"Network error: {exc}") from exc


def _fetch_openrouter_model_ids(
    *,
    api_base: str,
    api_key: str,
    timeout: float,
) -> set[str]:
    endpoint = api_base.rstrip("/") + "/models"
    req = urllib.request.Request(
        endpoint,
        headers=_build_auth_headers(api_key),
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=float(timeout)) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        request_id, body_text = _http_error_details(exc)
        raise QueryAPIError(
            f"HTTP {exc.code} {exc.reason}",
            status_code=exc.code,
            request_id=request_id,
            response_body=body_text,
        ) from exc
    except urllib.error.URLError as exc:
        raise QueryAPIError(f"Network error: {exc}") from exc

    payload = json.loads(body) if body else {}
    if not isinstance(payload, dict):
        return set()
    data = payload.get("data")
    if not isinstance(data, list):
        return set()

    out: set[str] = set()
    for item in data:
        if not isinstance(item, dict):
            continue
        model_id = str(item.get("id", "")).strip()
        if model_id:
            out.add(model_id)
    return out


def _fetch_openrouter_model_catalog(
    *,
    api_base: str,
    api_key: str,
    timeout: float,
) -> dict[str, dict[str, Any]]:
    endpoint = api_base.rstrip("/") + "/models"
    req = urllib.request.Request(
        endpoint,
        headers=_build_auth_headers(api_key),
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=float(timeout)) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        request_id, body_text = _http_error_details(exc)
        raise QueryAPIError(
            f"HTTP {exc.code} {exc.reason}",
            status_code=exc.code,
            request_id=request_id,
            response_body=body_text,
        ) from exc
    except urllib.error.URLError as exc:
        raise QueryAPIError(f"Network error: {exc}") from exc

    payload = json.loads(body) if body else {}
    if not isinstance(payload, dict):
        return {}
    data = payload.get("data")
    if not isinstance(data, list):
        return {}

    out: dict[str, dict[str, Any]] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        model_id = str(item.get("id", "")).strip()
        if model_id:
            out[model_id] = item
    return out


def _model_supports_image_input(model_entry: dict[str, Any]) -> Optional[bool]:
    def _list_has_image(value: Any) -> bool:
        if not isinstance(value, list):
            return False
        return any("image" in str(item).strip().lower() for item in value)

    def _str_has_image(value: Any) -> bool:
        return "image" in str(value).strip().lower()

    architecture = model_entry.get("architecture")
    if isinstance(architecture, dict):
        if _str_has_image(architecture.get("modality")):
            return True
        if _list_has_image(architecture.get("input_modalities")):
            return True

    if _list_has_image(model_entry.get("input_modalities")):
        return True
    if _list_has_image(model_entry.get("supported_input_modalities")):
        return True
    if _list_has_image(model_entry.get("modalities")):
        return True

    if "image" in json.dumps(model_entry, ensure_ascii=False).lower():
        return None
    return False


def _validate_requested_model_ids(
    *,
    models: list[dict[str, Any]],
    available_model_ids: set[str],
) -> list[dict[str, Any]]:
    if not available_model_ids:
        return [dict(item) for item in models]

    valid_models: list[dict[str, Any]] = []
    invalid: list[str] = []
    for entry in models:
        model_id = str(entry.get("model_id", "")).strip()
        if model_id and model_id not in available_model_ids:
            invalid.append(model_id)
            continue
        valid_models.append(dict(entry))
    if not invalid:
        return valid_models

    invalid_unique = sorted(set(invalid))
    parts = [f"invalid OpenRouter model_id(s): {', '.join(invalid_unique)}"]
    suggestions: list[str] = []
    for model_id in invalid_unique:
        provider = model_id.split("/", 1)[0] if "/" in model_id else ""
        if not provider:
            continue
        provider_matches = sorted(
            mid for mid in available_model_ids if mid.startswith(provider + "/")
        )
        if provider_matches:
            suggestions.append(f"{model_id} -> try one of: {', '.join(provider_matches[:5])}")
    if suggestions:
        parts.extend(suggestions)
    parts.append("Use --models-json or config models with IDs from OpenRouter /models.")
    message = " | ".join(parts)
    if not valid_models:
        raise ValueError(message)

    kept = ", ".join(
        f"{entry['label']}={entry['model_id']}"
        for entry in valid_models
    )
    print(f"WARNING: {message} | continuing with valid models: {kept}")
    return valid_models


def _filter_models_for_image_input(
    *,
    models: list[dict[str, Any]],
    model_catalog: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    if not model_catalog:
        return [dict(item) for item in models]

    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for entry in models:
        model_id = str(entry.get("model_id", "")).strip()
        catalog_entry = model_catalog.get(model_id)
        if catalog_entry is None:
            kept.append(dict(entry))
            continue
        support = _model_supports_image_input(catalog_entry)
        if support is False:
            rejected.append(dict(entry))
            continue
        kept.append(dict(entry))

    if not rejected:
        return kept

    rejected_ids = sorted({item["model_id"] for item in rejected})
    parts = [f"model(s) without image input support for this benchmark: {', '.join(rejected_ids)}"]
    suggestions: list[str] = []
    for model_id in rejected_ids:
        provider = model_id.split("/", 1)[0] if "/" in model_id else ""
        if not provider:
            continue
        provider_image_models = sorted(
            mid
            for mid, payload in model_catalog.items()
            if mid.startswith(provider + "/") and _model_supports_image_input(payload) is True
        )
        if provider_image_models:
            suggestions.append(f"{model_id} -> try one of: {', '.join(provider_image_models[:5])}")
    if suggestions:
        parts.extend(suggestions)
    message = " | ".join(parts)

    if not kept:
        raise ValueError(message)
    kept_labels = ", ".join(f"{item['label']}={item['model_id']}" for item in kept)
    print(f"WARNING: {message} | continuing with image-capable models: {kept_labels}")
    return kept


def _build_chat_payload(
    *,
    model_id: str,
    question: str,
    image_url: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    request_overrides: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": int(max_tokens),
    }
    if request_overrides:
        payload.update(request_overrides)
    return payload


def _call_openrouter_chat_api(
    *,
    api_base: str,
    api_key: str,
    model_id: str,
    question: str,
    image_url: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout: float,
    retry_429_max_retries: int,
    retry_429_backoff_s: float,
    retry_429_max_backoff_s: float,
    request_overrides: Optional[dict[str, Any]] = None,
) -> tuple[str, dict[str, Any], float]:
    payload = _build_chat_payload(
        model_id=model_id,
        question=question,
        image_url=image_url,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        request_overrides=request_overrides,
    )

    endpoint = api_base.rstrip("/") + "/chat/completions"
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
            answer_text = common.extract_openrouter_answer_text(data)
            return answer_text, data, latency_ms
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
                    "openrouter retry: "
                    f"status=429 attempt={attempt + 1}/{retries + 1} "
                    f"sleep={sleep_s:.2f}s latency_ms={latency_ms:.1f} "
                    f"request_id={request_id or '-'}"
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
            parts.append(f"response_body={common.truncate_text(exc.response_body, limit=500)}")
        return " | ".join(parts)
    return f"{type(exc).__name__}: {exc}"


def _stratify_bucket_key_for_example(
    item: train_utils.QAExample,
    *,
    task_name: str,
) -> Optional[str]:
    if task_name == "turn_player":
        normalized = train_utils._normalize_non_best_answer("turn_player", item.expected_answer)
        if not isinstance(normalized, dict):
            return None
        player = normalized.get("player")
        if isinstance(player, str) and player in {"X", "O"}:
            return player
        return None
    if task_name == "available_moves_count":
        normalized = train_utils._normalize_non_best_answer("available_moves_count", item.expected_answer)
        if not isinstance(normalized, dict):
            return None
        count = train_utils._parse_int(normalized.get("available_move_count"))
        if count is None:
            return None
        return str(int(count))
    return None


def _allocate_stratified_counts(
    *,
    bucket_sizes: dict[str, int],
    limit: int,
) -> dict[str, int]:
    keys = sorted(key for key, size in bucket_sizes.items() if size > 0)
    if limit <= 0 or not keys:
        return {key: 0 for key in keys}

    total_rows = sum(bucket_sizes[key] for key in keys)
    if limit >= total_rows:
        return {key: int(bucket_sizes[key]) for key in keys}

    allocations: dict[str, int] = {key: 0 for key in keys}
    if limit >= len(keys):
        for key in keys:
            allocations[key] = 1
        remaining = limit - len(keys)
    else:
        ranked_keys = sorted(keys, key=lambda key: (-bucket_sizes[key], key))
        for key in ranked_keys[:limit]:
            allocations[key] = 1
        remaining = 0

    if remaining <= 0:
        return allocations

    capacity = {key: max(0, bucket_sizes[key] - allocations[key]) for key in keys}
    capacity_total = sum(capacity.values())
    if capacity_total <= 0:
        return allocations

    base_add: dict[str, int] = {}
    fractional: list[tuple[float, str]] = []
    assigned = 0
    for key in keys:
        cap = capacity[key]
        if cap <= 0:
            base_add[key] = 0
            continue
        raw = (remaining * cap) / capacity_total
        add = min(cap, int(raw))
        base_add[key] = add
        assigned += add
        fractional.append((raw - add, key))

    for key in keys:
        allocations[key] += base_add.get(key, 0)

    leftovers = remaining - assigned
    if leftovers <= 0:
        return allocations

    fractional.sort(key=lambda item: (-item[0], item[1]))
    idx = 0
    while leftovers > 0 and fractional:
        _, key = fractional[idx % len(fractional)]
        if allocations[key] < bucket_sizes[key]:
            allocations[key] += 1
            leftovers -= 1
        idx += 1
        if idx > (len(fractional) * (remaining + 1)):
            break
    return allocations


def _sample_stratified_rows_for_task(
    rows: list[train_utils.QAExample],
    *,
    task_name: str,
    limit: int,
    rng: random.Random,
) -> list[train_utils.QAExample]:
    if limit <= 0 or not rows:
        return []

    buckets: dict[str, list[train_utils.QAExample]] = {}
    for item in rows:
        bucket_key = _stratify_bucket_key_for_example(item, task_name=task_name)
        if bucket_key is None:
            bucket_key = "__unknown__"
        buckets.setdefault(bucket_key, []).append(item)

    for bucket_rows in buckets.values():
        rng.shuffle(bucket_rows)

    allocations = _allocate_stratified_counts(
        bucket_sizes={key: len(bucket_rows) for key, bucket_rows in buckets.items()},
        limit=min(limit, len(rows)),
    )
    selected: list[train_utils.QAExample] = []
    for key in sorted(buckets.keys()):
        take = min(len(buckets[key]), int(allocations.get(key, 0)))
        if take > 0:
            selected.extend(buckets[key][:take])

    if len(selected) < min(limit, len(rows)):
        selected_ids = {id(item) for item in selected}
        spillover = [item for item in rows if id(item) not in selected_ids]
        rng.shuffle(spillover)
        remaining = min(limit, len(rows)) - len(selected)
        selected.extend(spillover[:remaining])

    rng.shuffle(selected)
    return selected[: min(limit, len(rows))]


def _sample_examples_by_task(
    examples: list[train_utils.QAExample],
    *,
    task_types: list[str],
    max_samples_per_task: int,
    seed: int,
    sample_strategy: str = "random",
    stratify_tasks: Optional[list[str]] = None,
) -> list[train_utils.QAExample]:
    rng = random.Random(seed)
    grouped: dict[str, list[train_utils.QAExample]] = {task: [] for task in task_types}
    for item in examples:
        if item.task_type in grouped:
            grouped[item.task_type].append(item)

    stratify_set = {
        task
        for task in (stratify_tasks or [])
        if task in STRATIFY_SUPPORTED_TASKS
    }
    selected: list[train_utils.QAExample] = []
    for task in task_types:
        rows = list(grouped.get(task, []))
        rng.shuffle(rows)
        if max_samples_per_task <= 0:
            selected.extend(rows)
            continue
        if sample_strategy == "stratified" and task in stratify_set:
            selected.extend(
                _sample_stratified_rows_for_task(
                    rows,
                    task_name=task,
                    limit=max_samples_per_task,
                    rng=rng,
                )
            )
            continue
        selected.extend(rows[:max_samples_per_task])

    rng.shuffle(selected)
    return selected


def _count_examples_by_task(examples: list[train_utils.QAExample]) -> dict[str, int]:
    out: Counter[str] = Counter()
    for item in examples:
        out[item.task_type] += 1
    return {task: int(out[task]) for task in sorted(out.keys())}


def _count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for _line in handle:
            count += 1
    return count


def _model_status_from_counts(*, requested_rows: int, prediction_lines: int) -> str:
    return "completed" if int(prediction_lines) == int(requested_rows) else "partial"


def _benchmark_model(
    *,
    model_label: str,
    model_id: str,
    examples: list[train_utils.QAExample],
    api_base: str,
    api_key: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout: float,
    retry_429_max_retries: int,
    retry_429_backoff_s: float,
    retry_429_max_backoff_s: float,
    request_overrides: Optional[dict[str, Any]],
    best_move_optimal_reward: float,
    predictions_path: Path,
    show_progress: bool,
    call_api_fn: Callable[..., tuple[str, dict[str, Any], float]] = _call_openrouter_chat_api,
) -> dict[str, Any]:
    total_scored = 0
    reward_sum = 0.0
    object_parse_count = 0
    parse_success_count = 0
    request_failure_count = 0
    prediction_lines_written = 0

    best_move_total = 0
    best_move_set_correct = 0
    best_move_canonical_correct = 0
    best_move_valid_prediction_count = 0
    best_move_wrong_reward_sum = 0.0
    best_move_wrong_count = 0

    non_best_total = 0
    non_best_exact_correct = 0
    turn_player_confusion: Counter[str] = Counter(
        {
            "X->X": 0,
            "X->O": 0,
            "O->X": 0,
            "O->O": 0,
        }
    )
    available_moves_count_delta_hist: Counter[int] = Counter()

    per_task_total: Counter[str] = Counter()
    per_task_correct: Counter[str] = Counter()
    per_task_parse_success: Counter[str] = Counter()
    per_task_request_failures: Counter[str] = Counter()

    latency_values_ms: list[float] = []

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with predictions_path.open("w", encoding="utf-8") as predictions_handle:
        progress = tqdm(
            examples,
            desc=f"openrouter:{model_label}",
            total=len(examples),
            dynamic_ncols=True,
            disable=not show_progress,
        )

        for item in progress:
            per_task_total[item.task_type] += 1

            try:
                image = Image.open(item.image_path).convert("RGB")
            except (FileNotFoundError, OSError) as exc:
                request_failure_count += 1
                per_task_request_failures[item.task_type] += 1
                predictions_handle.write(
                    json.dumps(
                        {
                            "row_id": item.row_id,
                            "split": item.split,
                            "task_type": item.task_type,
                            "model_label": model_label,
                            "model_id": model_id,
                            "status": "request_error",
                            "error": f"image load failed: {exc}",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                prediction_lines_written += 1
                continue

            try:
                answer_text, raw_response, latency_ms = call_api_fn(
                    api_base=api_base,
                    api_key=api_key,
                    model_id=model_id,
                    question=item.question,
                    image_url=train_utils._to_data_url(image, quality=92),
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    retry_429_max_retries=retry_429_max_retries,
                    retry_429_backoff_s=retry_429_backoff_s,
                    retry_429_max_backoff_s=retry_429_max_backoff_s,
                    request_overrides=request_overrides,
                )
            except Exception as exc:
                request_failure_count += 1
                per_task_request_failures[item.task_type] += 1
                predictions_handle.write(
                    json.dumps(
                        {
                            "row_id": item.row_id,
                            "split": item.split,
                            "task_type": item.task_type,
                            "model_label": model_label,
                            "model_id": model_id,
                            "status": "request_error",
                            "error": _error_details(exc),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                prediction_lines_written += 1
                continue

            latency_values_ms.append(latency_ms)
            pred_payload = train_utils._parse_prediction_json(answer_text)
            outcome = train_utils._score_payload_for_example(
                item,
                pred_payload,
                best_move_optimal_reward=best_move_optimal_reward,
            )

            total_scored += 1
            reward_sum += float(outcome.reward)
            if outcome.json_object_parsed:
                object_parse_count += 1
            if outcome.parse_success:
                parse_success_count += 1
                per_task_parse_success[item.task_type] += 1
            if outcome.task_correct:
                per_task_correct[item.task_type] += 1

            if item.task_type == "best_move":
                best_move_total += 1
                if outcome.best_move_valid_prediction:
                    best_move_valid_prediction_count += 1
                if outcome.best_move_set_correct:
                    best_move_set_correct += 1
                if outcome.best_move_canonical_correct:
                    best_move_canonical_correct += 1
                if not outcome.best_move_set_correct:
                    best_move_wrong_reward_sum += float(outcome.reward)
                    best_move_wrong_count += 1
            else:
                non_best_total += 1
                if outcome.exact_non_best_correct:
                    non_best_exact_correct += 1

            if item.task_type == "turn_player":
                gt_norm = train_utils._normalize_non_best_answer("turn_player", item.expected_answer)
                pred_norm = train_utils._normalize_non_best_answer("turn_player", pred_payload)
                gt_player = gt_norm.get("player") if isinstance(gt_norm, dict) else None
                pred_player = pred_norm.get("player") if isinstance(pred_norm, dict) else None
                if gt_player in {"X", "O"} and pred_player in {"X", "O"}:
                    turn_player_confusion[f"{gt_player}->{pred_player}"] += 1

            if item.task_type == "available_moves_count":
                gt_norm = train_utils._normalize_non_best_answer(
                    "available_moves_count",
                    item.expected_answer,
                )
                pred_norm = train_utils._normalize_non_best_answer(
                    "available_moves_count",
                    pred_payload,
                )
                gt_count = (
                    train_utils._parse_int(gt_norm.get("available_move_count"))
                    if isinstance(gt_norm, dict)
                    else None
                )
                pred_count = (
                    train_utils._parse_int(pred_norm.get("available_move_count"))
                    if isinstance(pred_norm, dict)
                    else None
                )
                if gt_count is not None and pred_count is not None:
                    available_moves_count_delta_hist[int(pred_count) - int(gt_count)] += 1

            predictions_handle.write(
                json.dumps(
                    {
                        "row_id": item.row_id,
                        "split": item.split,
                        "task_type": item.task_type,
                        "question": item.question,
                        "model_label": model_label,
                        "model_id": model_id,
                        "answer": answer_text,
                        "prediction_json": pred_payload,
                        "json_object_parsed": outcome.json_object_parsed,
                        "parse_success": outcome.parse_success,
                        "reward": outcome.reward,
                        "task_correct": outcome.task_correct,
                        "best_move_set_correct": outcome.best_move_set_correct,
                        "best_move_canonical_correct": outcome.best_move_canonical_correct,
                        "best_move_valid_prediction": outcome.best_move_valid_prediction,
                        "exact_non_best_correct": outcome.exact_non_best_correct,
                        "latency_ms": latency_ms,
                        "raw_response_excerpt": common.truncate_json_payload(raw_response, limit=2000),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            prediction_lines_written += 1

            if show_progress and total_scored % 10 == 0:
                parse_rate = parse_success_count / max(1, total_scored)
                reward_mean = reward_sum / max(1, total_scored)
                progress.set_postfix(
                    scored=total_scored,
                    parse=f"{parse_rate:.3f}",
                    reward=f"{reward_mean:.3f}",
                    fails=request_failure_count,
                )

    avg_latency_ms = sum(latency_values_ms) / max(1, len(latency_values_ms))
    p95_latency_ms = 0.0
    if latency_values_ms:
        sorted_lat = sorted(latency_values_ms)
        p95_idx = int(round(0.95 * (len(sorted_lat) - 1)))
        p95_latency_ms = sorted_lat[p95_idx]

    return {
        "model_label": model_label,
        "model": model_id,
        "split": examples[0].split if examples else "",
        "requested_rows": len(examples),
        "evaluated_rows": total_scored,
        "request_failures": request_failure_count,
        "json_object_fail": (total_scored - object_parse_count),
        "json_parse_fail": (total_scored - parse_success_count),
        "eval_reward_mean": (reward_sum / max(1, total_scored)),
        "eval_json_object_rate": (object_parse_count / max(1, total_scored)),
        "eval_json_parse_rate": (parse_success_count / max(1, total_scored)),
        "eval_best_move_set_accuracy": (best_move_set_correct / max(1, best_move_total)),
        "eval_best_move_canonical_accuracy": (best_move_canonical_correct / max(1, best_move_total)),
        "eval_best_move_valid_prediction_count": float(best_move_valid_prediction_count),
        "eval_best_move_valid_prediction_rate": (
            best_move_valid_prediction_count / max(1, best_move_total)
        ),
        "eval_exact_accuracy_non_best_move": (non_best_exact_correct / max(1, non_best_total)),
        "prediction_lines_written": int(prediction_lines_written),
        "latency_avg_ms": avg_latency_ms,
        "latency_p95_ms": p95_latency_ms,
        "diagnostics": {
            "best_move_wrong_reward_mean": (best_move_wrong_reward_sum / max(1, best_move_wrong_count)),
            "turn_player_confusion": {
                key: int(turn_player_confusion[key])
                for key in ("X->X", "X->O", "O->X", "O->O")
            },
            "available_moves_count_delta_hist": {
                str(delta): int(available_moves_count_delta_hist[delta])
                for delta in sorted(available_moves_count_delta_hist.keys())
            },
        },
        "by_task": {
            task: {
                "count": int(per_task_total[task]),
                "correct": int(per_task_correct[task]),
                "accuracy": (per_task_correct[task] / max(1, (per_task_total[task] - per_task_request_failures[task]))),
                "parse_success": int(per_task_parse_success[task]),
                "parse_rate": (per_task_parse_success[task] / max(1, (per_task_total[task] - per_task_request_failures[task]))),
                "request_failures": int(per_task_request_failures[task]),
            }
            for task in sorted(per_task_total.keys())
        },
    }


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)
    _validate_args(args)

    load_dotenv(args.env_file, override=False)
    args.api_key = _resolve_openrouter_api_key(args.api_key)
    try:
        _preflight_openrouter_auth(
            api_base=args.api_base,
            api_key=args.api_key,
            timeout=args.timeout,
        )
    except QueryAPIError as exc:
        details = _error_details(exc)
        if exc.status_code in {401, 403}:
            raise ValueError(
                "OpenRouter auth preflight failed (401/403). "
                "Check OPENROUTER_API_KEY, account access, and --api-base. "
                f"details: {details}"
            ) from exc
        raise ValueError(
            "OpenRouter preflight request failed before benchmark start. "
            f"details: {details}"
        ) from exc
    try:
        model_catalog = _fetch_openrouter_model_catalog(
            api_base=args.api_base,
            api_key=args.api_key,
            timeout=args.timeout,
        )
        available_model_ids = set(model_catalog.keys())
        args.models = _validate_requested_model_ids(
            models=args.models,
            available_model_ids=available_model_ids,
        )
        args.models = _filter_models_for_image_input(
            models=args.models,
            model_catalog=model_catalog,
        )
    except QueryAPIError as exc:
        raise ValueError(
            "Unable to fetch OpenRouter model catalog before benchmark start. "
            f"details: {_error_details(exc)}"
        ) from exc

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
    if args.sample_strategy == "stratified":
        unsupported_stratify = sorted(
            task for task in args.stratify_tasks if task not in STRATIFY_SUPPORTED_TASKS
        )
        if unsupported_stratify:
            print(
                "WARNING: stratify_tasks contains tasks without stratification support; "
                f"falling back to random for: {unsupported_stratify}"
            )
    selected_examples = _sample_examples_by_task(
        examples,
        task_types=args.task_types,
        max_samples_per_task=args.max_samples_per_task,
        seed=args.seed,
        sample_strategy=args.sample_strategy,
        stratify_tasks=args.stratify_tasks,
    )
    if not selected_examples:
        raise ValueError("No examples selected for benchmark. Check split/task filters.")

    run_dir = args.out_dir / f"openrouter_v2_{common.utc_timestamp_tag()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    selected_by_task = _count_examples_by_task(selected_examples)
    show_progress = _progress_enabled(args.no_progress)

    model_artifacts: list[dict[str, Any]] = []
    for model_entry in args.models:
        model_label = model_entry["label"]
        model_id = model_entry["model_id"]
        request_overrides = model_entry.get("request_overrides")
        if not isinstance(request_overrides, dict):
            request_overrides = None
        slug = common.slugify(model_label)

        metrics_path = run_dir / f"metrics_{slug}.json"
        predictions_path = run_dir / f"predictions_{slug}.jsonl"
        metrics_payload = _benchmark_model(
            model_label=model_label,
            model_id=model_id,
            examples=selected_examples,
            api_base=args.api_base,
            api_key=args.api_key,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            retry_429_max_retries=args.retry_429_max_retries,
            retry_429_backoff_s=args.retry_429_backoff_s,
            retry_429_max_backoff_s=args.retry_429_max_backoff_s,
            request_overrides=request_overrides,
            best_move_optimal_reward=args.best_move_optimal_reward,
            predictions_path=predictions_path,
            show_progress=show_progress,
        )

        metrics_payload.update(
            {
                "config": args.config,
                "dataset_source": args.dataset_source,
                "dataset_dir": str(dataset_dir) if dataset_dir is not None else "",
                "hf_dataset_repo_id": args.hf_dataset_repo_id,
                "hf_dataset_revision": args.hf_dataset_revision,
                "task_types": list(args.task_types),
                "sample_strategy": str(args.sample_strategy),
                "stratify_tasks": list(args.stratify_tasks),
                "max_samples_per_task": int(args.max_samples_per_task),
                "seed": int(args.seed),
                "api_base": args.api_base,
                "request_overrides": request_overrides or {},
            }
        )
        prediction_lines = _count_jsonl_lines(predictions_path)
        requested_rows = int(metrics_payload.get("requested_rows", len(selected_examples)))
        status = _model_status_from_counts(
            requested_rows=requested_rows,
            prediction_lines=prediction_lines,
        )
        metrics_payload["prediction_lines"] = int(prediction_lines)
        metrics_payload["status"] = status
        if status == "partial":
            print(
                f"WARNING: model={model_label} wrote predictions lines={prediction_lines} "
                f"but requested_rows={requested_rows}; marking status=partial"
            )
        metrics_path.write_text(json.dumps(metrics_payload, indent=2, sort_keys=True), encoding="utf-8")

        model_artifacts.append(
            {
                "label": model_label,
                "model_id": model_id,
                "metrics_file": metrics_path.name,
                "predictions_file": predictions_path.name,
                "status": status,
                "prediction_lines": int(prediction_lines),
                "eval_reward_mean": float(metrics_payload.get("eval_reward_mean", 0.0)),
                "eval_json_parse_rate": float(metrics_payload.get("eval_json_parse_rate", 0.0)),
            }
        )

    manifest = {
        "generated_utc": common.utc_timestamp_tag(),
        "config": args.config,
        "run_dir": str(run_dir),
        "dataset_source": args.dataset_source,
        "dataset_dir": str(dataset_dir) if dataset_dir is not None else "",
        "hf_dataset_repo_id": args.hf_dataset_repo_id,
        "hf_dataset_revision": args.hf_dataset_revision,
        "split": args.split,
        "task_types": list(args.task_types),
        "sample_strategy": str(args.sample_strategy),
        "stratify_tasks": list(args.stratify_tasks),
        "max_samples_per_task": int(args.max_samples_per_task),
        "seed": int(args.seed),
        "selected_rows": len(selected_examples),
        "selected_rows_by_task": selected_by_task,
        "models": model_artifacts,
    }
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(f"selected rows: {len(selected_examples)}")
    print(f"selected rows by task: {selected_by_task}")
    print(f"wrote outputs: {run_dir}")


if __name__ == "__main__":
    main()
