#!/usr/bin/env python3
"""Benchmark P&ID icon tasks against OpenRouter vision models."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

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

import benchmark_pid_icons as pid_bench  # noqa: E402
from pid_eval_suite import common  # noqa: E402

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "openrouter_pid_full.json"
SUPPORTED_SKILLS: tuple[str, ...] = ("detect", "point")
DEFAULT_SKILLS: list[str] = ["detect", "point"]
DEFAULT_MODELS: list[dict[str, Any]] = [
    {"label": "claude_frontier", "model_id": "anthropic/claude-3.7-sonnet"},
    {"label": "chatgpt_frontier", "model_id": "openai/gpt-5.1"},
    {"label": "qwen_vl_frontier", "model_id": "qwen/qwen-2.5-vl-7b-instruct"},
]

OPENROUTER_CONFIG_ALLOWED_KEYS = {
    "api_base",
    "api_key",
    "class_names_file",
    "dataset_name",
    "dataset_path",
    "env_file",
    "hf_token",
    "iou_threshold",
    "max_objects",
    "max_samples",
    "max_tokens",
    "min_request_interval_s",
    "models",
    "neg_prompts_per_empty",
    "neg_prompts_per_nonempty",
    "point_prompt_style",
    "progress_every",
    "retry_429_backoff_s",
    "retry_429_max_backoff_s",
    "retry_429_max_retries",
    "seed",
    "skills",
    "split",
    "temperature",
    "timeout",
    "top_p",
    "viz_dir",
    "viz_samples",
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


@dataclass
class ParsedPrediction:
    boxes: list[pid_bench.Box]
    points: list[pid_bench.Point]
    json_object_parsed: bool
    parse_success: bool
    parsed_payload: Optional[dict[str, Any]]


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
            if not label_text or not model_text:
                continue
            entry: dict[str, Any] = {"label": label_text, "model_id": model_text}
            request_overrides = item.get("request_overrides")
            if isinstance(request_overrides, dict):
                entry["request_overrides"] = dict(request_overrides)
            out.append(entry)

    if not out:
        raise ValueError("models must be a non-empty list of {label, model_id}")
    return out


def _parse_skills(raw_skills: Any) -> list[str]:
    if raw_skills is None:
        return list(DEFAULT_SKILLS)

    pieces: list[str] = []
    if isinstance(raw_skills, str):
        for part in raw_skills.split(","):
            text = part.strip().lower()
            if text:
                pieces.append(text)
    elif isinstance(raw_skills, (list, tuple)):
        for item in raw_skills:
            for part in str(item).split(","):
                text = part.strip().lower()
                if text:
                    pieces.append(text)

    if not pieces:
        return list(DEFAULT_SKILLS)

    out: list[str] = []
    seen: set[str] = set()
    for skill in pieces:
        if skill not in SUPPORTED_SKILLS:
            raise ValueError(f"Unsupported skill '{skill}'. Expected one of: {list(SUPPORTED_SKILLS)}")
        if skill in seen:
            continue
        seen.add(skill)
        out.append(skill)

    if not out:
        raise ValueError("At least one skill is required")
    return out


def _build_parser(config: dict[str, Any], config_path: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark P&ID icon tasks via OpenRouter models")
    parser.add_argument("--config", default=str(config_path))
    parser.add_argument("--out-dir", default="outputs/openrouter_pid")

    parser.add_argument("--env-file", default=common.cfg_str(config, "env_file", ".env"))
    parser.add_argument("--api-key", default=common.cfg_str(config, "api_key", ""))
    parser.add_argument("--api-base", default=common.cfg_str(config, "api_base", "https://openrouter.ai/api/v1"))
    parser.add_argument("--hf-token", default=common.cfg_str(config, "hf_token", ""))

    parser.add_argument("--dataset-path", default=common.cfg_str(config, "dataset_path", ""))
    parser.add_argument("--dataset-name", default=common.cfg_str(config, "dataset_name", "maxs-m87/pandid_dataset_v2"))
    parser.add_argument("--split", default=common.cfg_str(config, "split", "post_val"))
    parser.add_argument("--class-names-file", default=common.cfg_str(config, "class_names_file", ""))
    parser.add_argument(
        "--max-samples",
        type=int,
        default=common.cfg_int(config, "max_samples", 0),
        help="0 means full split.",
    )

    parser.add_argument(
        "--skills",
        nargs="*",
        default=None,
        help="Subset of skills to run. Supported: detect, point.",
    )
    parser.add_argument(
        "--point-prompt-style",
        choices=["detect_phrase", "class_name"],
        default=common.cfg_str(config, "point_prompt_style", "detect_phrase"),
    )

    parser.add_argument("--neg-prompts-per-empty", type=int, default=common.cfg_int(config, "neg_prompts_per_empty", 2))
    parser.add_argument(
        "--neg-prompts-per-nonempty",
        type=int,
        default=common.cfg_int(config, "neg_prompts_per_nonempty", 1),
    )

    parser.add_argument("--temperature", type=float, default=common.cfg_float(config, "temperature", 0.0))
    parser.add_argument("--top-p", type=float, default=common.cfg_float(config, "top_p", 1.0))
    parser.add_argument("--max-tokens", type=int, default=common.cfg_int(config, "max_tokens", 512))
    parser.add_argument("--max-objects", type=int, default=common.cfg_int(config, "max_objects", 50))
    parser.add_argument("--timeout", type=float, default=common.cfg_float(config, "timeout", 90.0))

    parser.add_argument("--retry-429-max-retries", type=int, default=common.cfg_int(config, "retry_429_max_retries", 2))
    parser.add_argument("--retry-429-backoff-s", type=float, default=common.cfg_float(config, "retry_429_backoff_s", 1.0))
    parser.add_argument(
        "--retry-429-max-backoff-s",
        type=float,
        default=common.cfg_float(config, "retry_429_max_backoff_s", 8.0),
    )
    parser.add_argument(
        "--min-request-interval-s",
        type=float,
        default=common.cfg_float(config, "min_request_interval_s", 0.0),
    )

    parser.add_argument("--iou-threshold", type=float, default=common.cfg_float(config, "iou_threshold", 0.5))
    parser.add_argument("--seed", type=int, default=common.cfg_int(config, "seed", 42))
    parser.add_argument("--progress-every", type=int, default=common.cfg_int(config, "progress_every", 100))

    parser.add_argument("--viz-samples", type=int, default=common.cfg_int(config, "viz_samples", 0))
    parser.add_argument(
        "--viz-dir",
        default=common.cfg_str(config, "viz_dir", "outputs/benchmark_viz_openrouter_pid"),
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
    args.out_dir = common.resolve_path(args.out_dir, search_roots=(common.repo_root(),))
    args.env_file = str(_resolve_env_file(args.env_file))

    dataset_path = str(args.dataset_path or "").strip()
    if dataset_path:
        args.dataset_path = str(common.resolve_path(dataset_path, search_roots=(common.repo_root(),)))
    else:
        args.dataset_path = ""

    class_names_file = str(args.class_names_file or "").strip()
    if class_names_file:
        args.class_names_file = str(common.resolve_path(class_names_file, search_roots=(common.repo_root(),)))
    else:
        args.class_names_file = ""

    args.viz_dir = str(common.resolve_path(args.viz_dir, search_roots=(common.repo_root(),)))

    if args.skills is None:
        args.skills = _parse_skills(config.get("skills"))
    else:
        args.skills = _parse_skills(args.skills)

    args.models = _parse_models(config.get("models"))
    if str(args.models_json).strip():
        args.models = _parse_models(args.models_json)

    return args


def _validate_args(args: argparse.Namespace) -> None:
    if args.max_samples < 0:
        raise ValueError("--max-samples must be >= 0")
    if args.neg_prompts_per_empty < 0:
        raise ValueError("--neg-prompts-per-empty must be >= 0")
    if args.neg_prompts_per_nonempty < 0:
        raise ValueError("--neg-prompts-per-nonempty must be >= 0")
    if not (0.0 <= args.temperature <= 2.0):
        raise ValueError("--temperature must be in [0,2]")
    if not (0.0 < args.top_p <= 1.0):
        raise ValueError("--top-p must be in (0,1]")
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be > 0")
    if args.max_objects <= 0:
        raise ValueError("--max-objects must be > 0")
    if args.timeout <= 0.0:
        raise ValueError("--timeout must be > 0")
    if args.retry_429_max_retries < 0:
        raise ValueError("--retry-429-max-retries must be >= 0")
    if args.retry_429_backoff_s < 0.0:
        raise ValueError("--retry-429-backoff-s must be >= 0")
    if args.retry_429_max_backoff_s < 0.0:
        raise ValueError("--retry-429-max-backoff-s must be >= 0")
    if args.min_request_interval_s < 0.0:
        raise ValueError("--min-request-interval-s must be >= 0")
    if not (0.0 <= args.iou_threshold <= 1.0):
        raise ValueError("--iou-threshold must be in [0,1]")
    if args.viz_samples < 0:
        raise ValueError("--viz-samples must be >= 0")
    if not args.skills:
        raise ValueError("At least one skill is required")


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


def _preflight_openrouter_auth(*, api_base: str, api_key: str, timeout: float) -> None:
    endpoint = api_base.rstrip("/") + "/models"
    req = urllib.request.Request(endpoint, headers=_build_auth_headers(api_key), method="GET")
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


def _fetch_openrouter_model_catalog(*, api_base: str, api_key: str, timeout: float) -> dict[str, dict[str, Any]]:
    endpoint = api_base.rstrip("/") + "/models"
    req = urllib.request.Request(endpoint, headers=_build_auth_headers(api_key), method="GET")
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


def _validate_requested_model_ids(*, models: list[dict[str, Any]], available_model_ids: set[str]) -> list[dict[str, Any]]:
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
        provider_matches = sorted(mid for mid in available_model_ids if mid.startswith(provider + "/"))
        if provider_matches:
            suggestions.append(f"{model_id} -> try one of: {', '.join(provider_matches[:5])}")
    if suggestions:
        parts.extend(suggestions)
    parts.append("Use --models-json or config models with IDs from OpenRouter /models.")
    message = " | ".join(parts)

    if not valid_models:
        raise ValueError(message)

    kept = ", ".join(f"{entry['label']}={entry['model_id']}" for entry in valid_models)
    print(f"WARNING: {message} | continuing with valid models: {kept}")
    return valid_models


def _filter_models_for_image_input(*, models: list[dict[str, Any]], model_catalog: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
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


def _detect_instruction(object_prompt: str) -> str:
    return (
        "You are analyzing a P&ID diagram image. "
        f"Find all instances of '{object_prompt}'. "
        "Return ONLY strict JSON with this schema: "
        '{"objects":[{"x_min":0.0,"y_min":0.0,"x_max":0.0,"y_max":0.0}]}. '
        "Coordinates must be normalized to [0,1]. "
        'If no instance exists, return exactly {"objects":[]} and nothing else.'
    )


def _point_instruction(object_prompt: str) -> str:
    return (
        "You are analyzing a P&ID diagram image. "
        f"Return point locations for all instances of '{object_prompt}'. "
        "Return ONLY strict JSON with this schema: "
        '{"points":[{"x":0.0,"y":0.0}]}. '
        "Coordinates must be normalized to [0,1]. "
        'If no instance exists, return exactly {"points":[]} and nothing else.'
    )


def _build_instruction(*, skill: str, object_prompt: str) -> str:
    if skill == "point":
        return _point_instruction(object_prompt)
    return _detect_instruction(object_prompt)


def _coerce_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_prediction_payload(answer_text: str, raw_response: dict[str, Any]) -> tuple[Optional[dict[str, Any]], bool]:
    texts: list[str] = []
    primary = str(answer_text or "").strip()
    if primary:
        texts.append(primary)

    secondary = common.extract_openrouter_answer_text(raw_response)
    secondary = str(secondary or "").strip()
    if secondary and secondary not in texts:
        texts.append(secondary)

    for text in texts:
        payload = common.extract_first_json_object(text)
        if payload is not None:
            return payload, True

    return None, False


def _normalize_xy(value: float, *, size: int) -> float:
    coord = float(value)
    if abs(coord) > 1.5 and size > 0:
        coord = coord / float(size)
    return max(0.0, min(1.0, coord))


def _parse_box_item(item: Any, *, width: int, height: int) -> Optional[pid_bench.Box]:
    if not isinstance(item, dict):
        return None

    x_min = _coerce_float(item.get("x_min", item.get("xmin")))
    y_min = _coerce_float(item.get("y_min", item.get("ymin")))
    x_max = _coerce_float(item.get("x_max", item.get("xmax")))
    y_max = _coerce_float(item.get("y_max", item.get("ymax")))

    if x_min is None or y_min is None or x_max is None or y_max is None:
        x_center = _coerce_float(item.get("x", item.get("cx")))
        y_center = _coerce_float(item.get("y", item.get("cy")))
        box_w = _coerce_float(item.get("width", item.get("w")))
        box_h = _coerce_float(item.get("height", item.get("h")))
        if x_center is None or y_center is None or box_w is None or box_h is None:
            return None

        x_center = _normalize_xy(x_center, size=width)
        y_center = _normalize_xy(y_center, size=height)
        if abs(box_w) > 1.5 and width > 0:
            box_w /= float(width)
        if abs(box_h) > 1.5 and height > 0:
            box_h /= float(height)
        box_w = max(0.0, min(1.0, float(box_w)))
        box_h = max(0.0, min(1.0, float(box_h)))

        x_min = x_center - (box_w / 2.0)
        y_min = y_center - (box_h / 2.0)
        x_max = x_center + (box_w / 2.0)
        y_max = y_center + (box_h / 2.0)

    x_min = _normalize_xy(x_min, size=width)
    y_min = _normalize_xy(y_min, size=height)
    x_max = _normalize_xy(x_max, size=width)
    y_max = _normalize_xy(y_max, size=height)
    if x_max <= x_min or y_max <= y_min:
        return None

    return pid_bench.Box(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)


def _parse_point_item(item: Any, *, width: int, height: int) -> Optional[pid_bench.Point]:
    if not isinstance(item, dict):
        return None

    x = _coerce_float(item.get("x", item.get("cx")))
    y = _coerce_float(item.get("y", item.get("cy")))
    if x is None or y is None:
        return None

    x = _normalize_xy(x, size=width)
    y = _normalize_xy(y, size=height)
    return pid_bench.Point(x=x, y=y)


def _parse_boxes_collection(raw: Any, *, width: int, height: int) -> list[pid_bench.Box]:
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return []

    out: list[pid_bench.Box] = []
    for item in raw:
        parsed = _parse_box_item(item, width=width, height=height)
        if parsed is not None:
            out.append(parsed)
    return out


def _parse_points_collection(raw: Any, *, width: int, height: int) -> list[pid_bench.Point]:
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return []

    out: list[pid_bench.Point] = []
    for item in raw:
        parsed = _parse_point_item(item, width=width, height=height)
        if parsed is not None:
            out.append(parsed)
    return out


def _extract_detect_candidates(payload: dict[str, Any]) -> list[Any]:
    out: list[Any] = []
    for key in ("objects", "boxes", "detections"):
        if key in payload:
            out.append(payload.get(key))

    output = payload.get("output")
    if isinstance(output, dict):
        for key in ("objects", "boxes", "detections"):
            if key in output:
                out.append(output.get(key))

    if all(k in payload for k in ("x_min", "y_min", "x_max", "y_max")):
        out.append([payload])
    if isinstance(output, dict) and all(k in output for k in ("x_min", "y_min", "x_max", "y_max")):
        out.append([output])

    return out


def _extract_point_candidates(payload: dict[str, Any]) -> list[Any]:
    out: list[Any] = []
    for key in ("points", "point", "coordinates"):
        if key in payload:
            out.append(payload.get(key))

    output = payload.get("output")
    if isinstance(output, dict):
        for key in ("points", "point", "coordinates"):
            if key in output:
                out.append(output.get(key))

    if all(k in payload for k in ("x", "y")):
        out.append([payload])
    if isinstance(output, dict) and all(k in output for k in ("x", "y")):
        out.append([output])

    return out


def _candidate_explicit_empty(raw: Any) -> bool:
    return isinstance(raw, list) and len(raw) == 0


def _parse_openrouter_prediction(
    *,
    skill: str,
    answer_text: str,
    raw_response: dict[str, Any],
    image_width: int,
    image_height: int,
) -> ParsedPrediction:
    payload, json_object_parsed = _extract_prediction_payload(answer_text, raw_response)
    if payload is None:
        return ParsedPrediction(
            boxes=[],
            points=[],
            json_object_parsed=False,
            parse_success=False,
            parsed_payload=None,
        )

    if skill == "point":
        candidates = _extract_point_candidates(payload)
        if not candidates:
            return ParsedPrediction(
                boxes=[],
                points=[],
                json_object_parsed=json_object_parsed,
                parse_success=False,
                parsed_payload=payload,
            )

        for candidate in candidates:
            points = _parse_points_collection(candidate, width=image_width, height=image_height)
            if points or _candidate_explicit_empty(candidate):
                return ParsedPrediction(
                    boxes=[],
                    points=points,
                    json_object_parsed=json_object_parsed,
                    parse_success=True,
                    parsed_payload=payload,
                )

        return ParsedPrediction(
            boxes=[],
            points=[],
            json_object_parsed=json_object_parsed,
            parse_success=True,
            parsed_payload=payload,
        )

    candidates = _extract_detect_candidates(payload)
    if not candidates:
        return ParsedPrediction(
            boxes=[],
            points=[],
            json_object_parsed=json_object_parsed,
            parse_success=False,
            parsed_payload=payload,
        )

    for candidate in candidates:
        boxes = _parse_boxes_collection(candidate, width=image_width, height=image_height)
        if boxes or _candidate_explicit_empty(candidate):
            return ParsedPrediction(
                boxes=boxes,
                points=[],
                json_object_parsed=json_object_parsed,
                parse_success=True,
                parsed_payload=payload,
            )

    return ParsedPrediction(
        boxes=[],
        points=[],
        json_object_parsed=json_object_parsed,
        parse_success=True,
        parsed_payload=payload,
    )


def _serialize_boxes(boxes: list[pid_bench.Box]) -> list[dict[str, float]]:
    return [
        {
            "x_min": float(box.x_min),
            "y_min": float(box.y_min),
            "x_max": float(box.x_max),
            "y_max": float(box.y_max),
        }
        for box in boxes
    ]


def _serialize_points(points: list[pid_bench.Point]) -> list[dict[str, float]]:
    return [{"x": float(point.x), "y": float(point.y)} for point in points]


def _count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for _ in handle:
            count += 1
    return count


def _artifact_status_from_counts(*, attempted_tasks: int, prediction_lines: int) -> str:
    return "completed" if int(attempted_tasks) == int(prediction_lines) else "partial"


def _evaluate_model_skill(
    *,
    skill: str,
    model_label: str,
    model_id: str,
    request_overrides: Optional[dict[str, Any]],
    args: argparse.Namespace,
    all_class_names: list[str],
    predictions_path: Path,
    show_progress: bool,
) -> dict[str, Any]:
    rng = random.Random(args.seed)
    prompt_style = args.point_prompt_style if skill == "point" else "detect_phrase"
    last_request_end: Optional[float] = None

    viz_saved = 0
    viz_paths: list[str] = []
    viz_out_dir = Path(args.viz_dir).expanduser().resolve() if args.viz_dir else None

    total_base_samples = 0
    total_tasks = 0
    failed_tasks = 0
    total_f1 = 0.0
    total_miou = 0.0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_latency = 0.0
    json_object_parsed_count = 0
    parse_success_count = 0

    per_class: dict[str, dict[str, Any]] = {}

    def _class_stats(name: str) -> dict[str, Any]:
        if name not in per_class:
            per_class[name] = {
                "tasks": 0,
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "f1_sum": 0.0,
                "miou_sum": 0.0,
            }
        return per_class[name]

    max_samples = None if int(args.max_samples) <= 0 else int(args.max_samples)

    row_iter = pid_bench._iter_rows(
        dataset_path=args.dataset_path.strip(),
        dataset_name=args.dataset_name,
        split=args.split,
        token=args.hf_token,
        max_samples=max_samples,
    )

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    row_progress = tqdm(row_iter, disable=not show_progress, desc=f"{skill}:{model_label}", dynamic_ncols=True)

    with predictions_path.open("w", encoding="utf-8") as predictions_handle:
        for row_idx, row in enumerate(row_progress):
            sample = pid_bench._to_base_sample(row, row_idx)
            if sample is None:
                continue
            total_base_samples += 1

            tasks = pid_bench._tasks_from_sample(
                sample,
                all_class_names=all_class_names,
                rng=rng,
                neg_prompts_per_empty=args.neg_prompts_per_empty,
                neg_prompts_per_nonempty=args.neg_prompts_per_nonempty,
                prompt_style=prompt_style,
            )

            for task in tasks:
                if args.min_request_interval_s > 0.0 and last_request_end is not None:
                    wait_s = float(args.min_request_interval_s) - (time.monotonic() - last_request_end)
                    if wait_s > 0.0:
                        time.sleep(wait_s)

                question = _build_instruction(skill=skill, object_prompt=task.prompt)
                image_url = pid_bench._to_data_url(task.image, quality=90)
                started = time.monotonic()

                try:
                    answer_text, raw_response, latency_ms = _call_openrouter_chat_api(
                        api_base=args.api_base,
                        api_key=args.api_key,
                        model_id=model_id,
                        question=question,
                        image_url=image_url,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_tokens=args.max_tokens,
                        timeout=args.timeout,
                        retry_429_max_retries=args.retry_429_max_retries,
                        retry_429_backoff_s=args.retry_429_backoff_s,
                        retry_429_max_backoff_s=args.retry_429_max_backoff_s,
                        request_overrides=request_overrides,
                    )
                except Exception as exc:
                    _ = time.monotonic() - started
                    failed_tasks += 1
                    last_request_end = time.monotonic()
                    predictions_handle.write(
                        json.dumps(
                            {
                                "status": "request_error",
                                "skill": skill,
                                "model_label": model_label,
                                "model_id": model_id,
                                "sample_id": task.sample_id,
                                "class_name": task.class_name,
                                "prompt": task.prompt,
                                "error": _error_details(exc),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    continue

                last_request_end = time.monotonic()
                latency_sec = float(latency_ms) / 1000.0

                parsed = _parse_openrouter_prediction(
                    skill=skill,
                    answer_text=answer_text,
                    raw_response=raw_response,
                    image_width=task.image.width,
                    image_height=task.image.height,
                )

                if parsed.json_object_parsed:
                    json_object_parsed_count += 1
                if parsed.parse_success:
                    parse_success_count += 1

                if skill == "point":
                    pred_points = parsed.points
                    pred_boxes: list[pid_bench.Box] = []
                    f1 = pid_bench._reward_f1_points(pred_points, task.gt_boxes)
                    miou = 0.0
                    tp, fp, fn = pid_bench._count_tp_fp_fn_points(pred_points, task.gt_boxes)
                else:
                    pred_boxes = parsed.boxes
                    pred_points = []
                    f1 = pid_bench._reward_f1(pred_boxes, task.gt_boxes)
                    miou = pid_bench._reward_miou(pred_boxes, task.gt_boxes)
                    tp, fp, fn = pid_bench._count_tp_fp_fn(pred_boxes, task.gt_boxes, iou_threshold=args.iou_threshold)

                total_tasks += 1
                total_f1 += f1
                total_miou += miou
                total_tp += tp
                total_fp += fp
                total_fn += fn
                total_latency += latency_sec

                stats = _class_stats(task.class_name)
                stats["tasks"] += 1
                stats["tp"] += tp
                stats["fp"] += fp
                stats["fn"] += fn
                stats["f1_sum"] += f1
                stats["miou_sum"] += miou

                if args.viz_samples > 0 and viz_saved < args.viz_samples and viz_out_dir is not None:
                    viz_path = pid_bench._save_task_visualization(
                        out_dir=viz_out_dir,
                        label=f"{skill}_{model_label}",
                        sample_idx=viz_saved,
                        task=task,
                        skill=skill,
                        pred_boxes=pred_boxes,
                        pred_points=pred_points,
                        iou_threshold=args.iou_threshold,
                        f1=f1,
                        miou=miou,
                        tp=tp,
                        fp=fp,
                        fn=fn,
                    )
                    if viz_path:
                        viz_paths.append(viz_path)
                        viz_saved += 1

                record: dict[str, Any] = {
                    "status": "ok",
                    "skill": skill,
                    "model_label": model_label,
                    "model_id": model_id,
                    "sample_id": task.sample_id,
                    "class_name": task.class_name,
                    "prompt": task.prompt,
                    "gt_count": len(task.gt_boxes),
                    "gt_boxes": _serialize_boxes(task.gt_boxes),
                    "pred_count": len(pred_points) if skill == "point" else len(pred_boxes),
                    "json_object_parsed": parsed.json_object_parsed,
                    "parse_success": parsed.parse_success,
                    "f1": f1,
                    "miou": miou,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "latency_ms": latency_ms,
                    "answer_excerpt": common.truncate_text(answer_text, limit=1200),
                    "parsed_payload_excerpt": common.truncate_json_payload(parsed.parsed_payload, limit=2000)
                    if parsed.parsed_payload is not None
                    else "",
                    "raw_response_excerpt": common.truncate_json_payload(raw_response, limit=2500),
                }
                if skill == "point":
                    record["pred_points"] = _serialize_points(pred_points)
                else:
                    record["pred_boxes"] = _serialize_boxes(pred_boxes)

                predictions_handle.write(json.dumps(record, ensure_ascii=False) + "\n")

                if args.progress_every > 0 and total_tasks > 0 and total_tasks % args.progress_every == 0:
                    print(
                        f"{skill}:{model_label}: progress tasks={total_tasks} "
                        f"failed={failed_tasks} parse={parse_success_count / max(1, total_tasks):.3f}"
                    )

    attempted_tasks = total_tasks + failed_tasks
    if total_tasks == 0:
        return {
            "label": model_label,
            "skill": skill,
            "model": model_id,
            "error": "No tasks evaluated",
            "base_samples": total_base_samples,
            "tasks": 0,
            "failed_tasks": failed_tasks,
            "attempted_tasks": attempted_tasks,
            "visualizations_saved": viz_saved,
            "visualization_paths": viz_paths,
            "json_object_parse_rate": 0.0,
            "parse_success_rate": 0.0,
        }

    micro_denom = 2 * total_tp + total_fp + total_fn
    micro_f1 = 1.0 if micro_denom == 0 else (2 * total_tp) / micro_denom

    per_class_payload: dict[str, Any] = {}
    for class_name, stats in sorted(per_class.items()):
        class_tasks = int(stats["tasks"])
        class_denom = 2 * int(stats["tp"]) + int(stats["fp"]) + int(stats["fn"])
        class_micro = 1.0 if class_denom == 0 else (2 * int(stats["tp"])) / class_denom
        per_class_payload[class_name] = {
            "tasks": class_tasks,
            "tp": int(stats["tp"]),
            "fp": int(stats["fp"]),
            "fn": int(stats["fn"]),
            "f1_micro": class_micro,
            "f1_macro": (float(stats["f1_sum"]) / class_tasks) if class_tasks else 0.0,
            "miou": (float(stats["miou_sum"]) / class_tasks) if class_tasks else 0.0,
        }

    return {
        "label": model_label,
        "skill": skill,
        "model": model_id,
        "dataset_name": args.dataset_name or None,
        "dataset_path": args.dataset_path.strip() or None,
        "split": args.split,
        "base_samples": total_base_samples,
        "tasks": total_tasks,
        "failed_tasks": failed_tasks,
        "attempted_tasks": attempted_tasks,
        "eval_f1": micro_f1,
        "eval_f1_macro": total_f1 / total_tasks,
        "eval_miou": total_miou / total_tasks,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "avg_latency_sec": total_latency / total_tasks,
        "json_object_parse_rate": json_object_parsed_count / total_tasks,
        "parse_success_rate": parse_success_count / total_tasks,
        "visualizations_saved": viz_saved,
        "visualization_paths": viz_paths,
        "per_class": per_class_payload,
    }


def _print_summary(*, skill: str, model_label: str, metrics: dict[str, Any]) -> None:
    title = f"{skill}:{model_label}"
    if "error" in metrics:
        print(f"{title}: error={metrics['error']}")
        return

    if skill == "point":
        print(
            f"{title}: base_samples={metrics.get('base_samples', 0)} tasks={metrics['tasks']} "
            f"f1={metrics['eval_f1']:.4f} macro_f1={metrics['eval_f1_macro']:.4f} "
            f"tp={metrics['tp']} fp={metrics['fp']} fn={metrics['fn']} "
            f"failed={metrics.get('failed_tasks', 0)} latency={metrics['avg_latency_sec']:.3f}s "
            f"parse={metrics.get('parse_success_rate', 0.0):.3f}"
        )
        return

    print(
        f"{title}: base_samples={metrics.get('base_samples', 0)} tasks={metrics['tasks']} "
        f"miou={metrics['eval_miou']:.4f} f1={metrics['eval_f1']:.4f} macro_f1={metrics['eval_f1_macro']:.4f} "
        f"tp={metrics['tp']} fp={metrics['fp']} fn={metrics['fn']} "
        f"failed={metrics.get('failed_tasks', 0)} latency={metrics['avg_latency_sec']:.3f}s "
        f"parse={metrics.get('parse_success_rate', 0.0):.3f}"
    )


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)
    _validate_args(args)

    load_dotenv(args.env_file, override=False)
    args.api_key = _resolve_openrouter_api_key(args.api_key)
    if not args.hf_token:
        args.hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

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
        args.models = _validate_requested_model_ids(models=args.models, available_model_ids=available_model_ids)
        args.models = _filter_models_for_image_input(models=args.models, model_catalog=model_catalog)
    except QueryAPIError as exc:
        raise ValueError(
            "Unable to fetch OpenRouter model catalog before benchmark start. "
            f"details: {_error_details(exc)}"
        ) from exc

    resolved_dataset_path, resolved_dataset_name = pid_bench._resolve_dataset_source(args.dataset_path, args.dataset_name)
    args.dataset_path = resolved_dataset_path
    args.dataset_name = resolved_dataset_name

    all_class_names = pid_bench._load_class_names(args.class_names_file, args.dataset_path or None)
    if not all_class_names:
        infer_max_samples = None if int(args.max_samples) <= 0 else int(args.max_samples)
        all_class_names = pid_bench._infer_class_names_from_dataset(
            dataset_path=args.dataset_path,
            dataset_name=args.dataset_name,
            split=args.split,
            token=args.hf_token,
            max_samples=infer_max_samples,
        )
        if all_class_names:
            print(f"discovered {len(all_class_names)} class names from dataset rows")
    if not all_class_names:
        raise ValueError("Could not resolve class names from class file, local metadata, or dataset rows.")

    run_dir = Path(args.out_dir).expanduser().resolve() / f"openrouter_pid_{common.utc_timestamp_tag()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    show_progress = _progress_enabled(args.no_progress)

    artifacts: list[dict[str, Any]] = []
    for skill in args.skills:
        for model_entry in args.models:
            model_label = model_entry["label"]
            model_id = model_entry["model_id"]
            request_overrides = model_entry.get("request_overrides")
            if not isinstance(request_overrides, dict):
                request_overrides = None

            slug = common.slugify(model_label)
            metrics_path = run_dir / f"metrics_{skill}_{slug}.json"
            predictions_path = run_dir / f"predictions_{skill}_{slug}.jsonl"

            metrics_payload = _evaluate_model_skill(
                skill=skill,
                model_label=model_label,
                model_id=model_id,
                request_overrides=request_overrides,
                args=args,
                all_class_names=all_class_names,
                predictions_path=predictions_path,
                show_progress=show_progress,
            )
            _print_summary(skill=skill, model_label=model_label, metrics=metrics_payload)

            metrics_payload.update(
                {
                    "config": args.config,
                    "api_base": args.api_base,
                    "dataset_name": args.dataset_name,
                    "dataset_path": args.dataset_path,
                    "split": args.split,
                    "skills": list(args.skills),
                    "max_samples": int(args.max_samples),
                    "point_prompt_style": args.point_prompt_style,
                    "neg_prompts_per_empty": int(args.neg_prompts_per_empty),
                    "neg_prompts_per_nonempty": int(args.neg_prompts_per_nonempty),
                    "temperature": float(args.temperature),
                    "top_p": float(args.top_p),
                    "max_tokens": int(args.max_tokens),
                    "max_objects": int(args.max_objects),
                    "timeout": float(args.timeout),
                    "retry_429_max_retries": int(args.retry_429_max_retries),
                    "retry_429_backoff_s": float(args.retry_429_backoff_s),
                    "retry_429_max_backoff_s": float(args.retry_429_max_backoff_s),
                    "min_request_interval_s": float(args.min_request_interval_s),
                    "iou_threshold": float(args.iou_threshold),
                    "seed": int(args.seed),
                    "request_overrides": request_overrides or {},
                }
            )

            prediction_lines = _count_jsonl_lines(predictions_path)
            attempted_tasks = int(metrics_payload.get("attempted_tasks", 0))
            status = _artifact_status_from_counts(attempted_tasks=attempted_tasks, prediction_lines=prediction_lines)
            metrics_payload["prediction_lines"] = int(prediction_lines)
            metrics_payload["status"] = status
            if status == "partial":
                print(
                    f"WARNING: skill={skill} model={model_label} wrote predictions lines={prediction_lines} "
                    f"but attempted_tasks={attempted_tasks}; marking status=partial"
                )

            metrics_path.write_text(json.dumps(metrics_payload, indent=2, sort_keys=True), encoding="utf-8")

            artifacts.append(
                {
                    "skill": skill,
                    "label": model_label,
                    "model_id": model_id,
                    "metrics_file": metrics_path.name,
                    "predictions_file": predictions_path.name,
                    "status": status,
                    "prediction_lines": int(prediction_lines),
                    "attempted_tasks": attempted_tasks,
                    "eval_f1": float(metrics_payload.get("eval_f1", 0.0)),
                    "eval_f1_macro": float(metrics_payload.get("eval_f1_macro", 0.0)),
                    "eval_miou": float(metrics_payload.get("eval_miou", 0.0)),
                    "failed_tasks": int(metrics_payload.get("failed_tasks", 0)),
                }
            )

    manifest = {
        "generated_utc": common.utc_timestamp_tag(),
        "config": args.config,
        "run_dir": str(run_dir),
        "api_base": args.api_base,
        "dataset_name": args.dataset_name,
        "dataset_path": args.dataset_path,
        "split": args.split,
        "max_samples": int(args.max_samples),
        "skills": list(args.skills),
        "models": [
            {
                "label": item["label"],
                "model_id": item["model_id"],
                "request_overrides": item.get("request_overrides", {}),
            }
            for item in args.models
        ],
        "artifacts": artifacts,
    }
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(f"wrote outputs: {run_dir}")


if __name__ == "__main__":
    main()
