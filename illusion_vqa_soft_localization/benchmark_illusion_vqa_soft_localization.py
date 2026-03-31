#!/usr/bin/env python3
"""Benchmark Moondream on IllusionVQA soft localization."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import random
import re
import socket
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

from PIL import Image

try:
    from datasets import load_dataset  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError("datasets is required. Install with `pip install datasets`.") from exc

try:
    from dotenv import load_dotenv as _load_dotenv
except ModuleNotFoundError:  # pragma: no cover
    _load_dotenv = None

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
MODULE_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_DATASET_REPO_ID = "csebuetnlp/illusionVQA-Soft-Localization"
DEFAULT_DATASET_REVISION = "main"
DEFAULT_STAGING_API_BASE = "https://api-staging.moondream.ai/v1"
DEFAULT_CONFIG_PATH = MODULE_ROOT / "configs" / "benchmark_default.json"
DEFAULT_API_KEY_ENV_VAR = "CICID_GPUB_MOONDREAM_API_KEY_1"
OPTION_LETTERS = tuple("abcdefghijklmnopqrstuvwxyz")

from construction_site import query_common as shared_query_common  # noqa: E402

BENCHMARK_CONFIG_ALLOWED_KEYS = {
    "api_key",
    "api_key_env_var",
    "base_url",
    "checkpoint_step",
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
    "temperature",
    "timeout",
    "top_p",
}

LETTER_PATTERNS = (
    re.compile(r"^\s*(?:answer\s*(?:is|:)\s*|option\s*)?[\(\[]?([a-z])[\)\].:,\s]*(?:$|[\r\n ])", re.IGNORECASE),
    re.compile(r"\b(?:answer|option)\s*(?:is|:)?\s*[\(\[]?([a-z])[\)\].:,\s]*(?:$|[\r\n ])", re.IGNORECASE),
)


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


def resolve_path(raw_path: str | Path, *, search_roots: tuple[Path, ...] = ()) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    from_cwd = (Path.cwd() / path).resolve()
    if from_cwd.exists():
        return from_cwd
    for root in search_roots:
        candidate = (root / path).resolve()
        if candidate.exists():
            return candidate
    return from_cwd


def load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def validate_config_keys(config: dict[str, Any], *, allowed_keys: set[str], config_path: Path) -> None:
    unknown = sorted(key for key in config if key not in allowed_keys)
    if unknown:
        raise ValueError(f"Unknown config key(s) in {config_path}: {unknown}")


def cfg_str(config: dict[str, Any], key: str, fallback: str) -> str:
    value = config.get(key, fallback)
    return str(value) if value is not None else fallback


def cfg_int(config: dict[str, Any], key: str, fallback: int) -> int:
    value = config.get(key, fallback)
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def cfg_float(config: dict[str, Any], key: str, fallback: float) -> float:
    value = config.get(key, fallback)
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def cfg_bool(config: dict[str, Any], key: str, fallback: bool) -> bool:
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


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_dotenv_if_available(path: str | Path) -> None:
    if _load_dotenv is None:
        return
    resolved = resolve_path(path, search_roots=(REPO_ROOT, MODULE_ROOT))
    if resolved.exists():
        _load_dotenv(resolved, override=False)


def progress_enabled(no_progress: bool) -> bool:
    return not no_progress and os.isatty(2)


def truncate_text(text: str, *, limit: int = 800) -> str:
    raw = str(text)
    if len(raw) <= limit:
        return raw
    return raw[:limit] + "...<truncated>"


def write_json(path: Path, payload: Any) -> None:
    ensure_parent_dir(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve_hf_token(raw_token: str) -> str:
    token = str(raw_token or "").strip()
    if token:
        return token
    return os.environ.get("HF_TOKEN", "").strip() or os.environ.get("HUGGINGFACE_HUB_TOKEN", "").strip()


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
    config_path = resolve_path(pre_args.config, search_roots=(REPO_ROOT, MODULE_ROOT))
    config = _load_json_config(config_path)
    validate_config_keys(config, allowed_keys=BENCHMARK_CONFIG_ALLOWED_KEYS, config_path=config_path)

    parser = argparse.ArgumentParser(description="Benchmark Moondream on IllusionVQA soft localization.")
    parser.add_argument("--config", default=str(config_path))
    parser.add_argument("--env-file", default=cfg_str(config, "env_file", ".env.staging"))
    parser.add_argument("--api-key", default=cfg_str(config, "api_key", ""))
    parser.add_argument("--api-key-env-var", default=cfg_str(config, "api_key_env_var", DEFAULT_API_KEY_ENV_VAR))
    parser.add_argument("--base-url", default=cfg_str(config, "base_url", DEFAULT_STAGING_API_BASE))
    parser.add_argument("--hf-dataset-repo-id", default=cfg_str(config, "hf_dataset_repo_id", DEFAULT_DATASET_REPO_ID))
    parser.add_argument("--hf-dataset-revision", default=cfg_str(config, "hf_dataset_revision", DEFAULT_DATASET_REVISION))
    parser.add_argument("--hf-token", default=cfg_str(config, "hf_token", ""))
    parser.add_argument("--hf-cache-dir", default=cfg_str(config, "hf_cache_dir", ""))
    parser.add_argument("--split", default=cfg_str(config, "split", "test"))
    parser.add_argument("--model", default=cfg_str(config, "model", ""))
    parser.add_argument("--finetune-id", default=cfg_str(config, "finetune_id", ""))
    parser.add_argument("--checkpoint-step", type=int, default=cfg_int(config, "checkpoint_step", -1))
    parser.add_argument("--temperature", type=float, default=cfg_float(config, "temperature", 0.0))
    parser.add_argument("--top-p", type=float, default=cfg_float(config, "top_p", 1.0))
    parser.add_argument("--max-tokens", type=int, default=cfg_int(config, "max_tokens", 16))
    parser.add_argument("--timeout", type=float, default=cfg_float(config, "timeout", 90.0))
    parser.add_argument("--reasoning", dest="reasoning", action="store_true")
    parser.add_argument("--no-reasoning", dest="reasoning", action="store_false")
    parser.set_defaults(reasoning=cfg_bool(config, "reasoning", False))
    parser.add_argument("--max-samples", type=int, default=cfg_int(config, "max_samples", 0))
    parser.add_argument("--seed", type=int, default=cfg_int(config, "seed", 42))
    parser.add_argument("--retry-429-max-retries", type=int, default=cfg_int(config, "retry_429_max_retries", 2))
    parser.add_argument("--retry-429-backoff-s", type=float, default=cfg_float(config, "retry_429_backoff_s", 1.0))
    parser.add_argument(
        "--retry-429-max-backoff-s",
        type=float,
        default=cfg_float(config, "retry_429_max_backoff_s", 8.0),
    )
    parser.add_argument("--output-json", default=cfg_str(config, "output_json", ""))
    parser.add_argument("--predictions-jsonl", default=cfg_str(config, "predictions_jsonl", ""))
    parser.add_argument("--no-progress", action="store_true", default=cfg_bool(config, "no_progress", False))
    args = parser.parse_args(argv)
    args.env_file = resolve_path(args.env_file, search_roots=(REPO_ROOT, MODULE_ROOT))
    return args


def _validate_args(args: argparse.Namespace) -> None:
    if not str(args.hf_dataset_repo_id or "").strip():
        raise ValueError("--hf-dataset-repo-id is required")
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


def _resolve_api_key(args: argparse.Namespace) -> str:
    key = str(getattr(args, "api_key", "") or "").strip()
    if key:
        return key
    env_name = str(getattr(args, "api_key_env_var", "") or DEFAULT_API_KEY_ENV_VAR).strip() or DEFAULT_API_KEY_ENV_VAR
    env_key = os.environ.get(env_name, "").strip()
    if env_key:
        return env_key
    fallback_names = ("CICID_GPUB_MOONDREAM_API_KEY_1", "MOONDREAM_API_KEY", "CICID_GPUB_MOONDREAM_API_KEY_2")
    for name in fallback_names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    raise ValueError("A Moondream API key is required")


def _resolve_base_url(raw_base_url: str) -> str:
    text = str(raw_base_url or "").strip()
    if text:
        return text
    return (
        os.environ.get("TUNA_BASE_URL", "").strip()
        or os.environ.get("MOONDREAM_BASE_URL", "").strip()
        or DEFAULT_STAGING_API_BASE
    )


def _resolve_model_identifier(*, model: str, finetune_id: str, checkpoint_step: Optional[int]) -> str:
    if str(model or "").strip():
        return str(model).strip()
    ftid = str(finetune_id or "").strip()
    if ftid:
        if checkpoint_step is not None:
            return f"moondream3-preview/{ftid}@{checkpoint_step}"
        return f"moondream3-preview/{ftid}"
    return "moondream3-preview"


def _build_auth_headers(api_key: str) -> dict[str, str]:
    header_name = os.environ.get("MOONDREAM_AUTH_HEADER", "X-Moondream-Auth")
    key = api_key.strip()
    if header_name.lower() == "authorization" and not key.lower().startswith("bearer "):
        key = f"Bearer {key}"
    return {
        "Content-Type": "application/json",
        "Accept": "application/json",
        header_name: key,
        "User-Agent": "illusion-vqa-soft-localization-benchmark/0.1",
    }


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
    retry_429_max_retries: int,
    retry_429_backoff_s: float,
    retry_429_max_backoff_s: float,
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
            with urllib.request.urlopen(req, timeout=float(timeout)) as response:
                body = response.read().decode("utf-8", errors="replace")
            latency_ms = (time.monotonic() - started) * 1000.0
            data = json.loads(body) if body else {}
            if not isinstance(data, dict):
                data = {}
            return data, latency_ms
        except urllib.error.HTTPError as exc:
            latency_ms = (time.monotonic() - started) * 1000.0
            request_id = str(exc.headers.get("x-request-id") or exc.headers.get("X-Request-Id") or "")
            body_text = exc.read().decode("utf-8", errors="replace")
            retry_after_s = 0.0
            if exc.code == 429:
                retry_after_header = str(exc.headers.get("Retry-After") or "").strip()
                if retry_after_header:
                    try:
                        retry_after_s = max(0.0, float(retry_after_header))
                    except (TypeError, ValueError):
                        retry_after_s = 0.0
            if exc.code == 429 and attempt < retries:
                exp_backoff = max(0.0, float(retry_429_backoff_s)) * (2.0**attempt)
                sleep_s = max(retry_after_s, min(float(retry_429_max_backoff_s), exp_backoff))
                print(
                    "query retry: "
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
            ) from exc
        except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
            raise QueryAPIError(f"Network error: {exc}") from exc


def _extract_answer_text(payload: Any) -> str:
    if isinstance(payload, dict):
        answer = payload.get("answer")
        if isinstance(answer, str):
            return answer
        output = payload.get("output")
        if isinstance(output, dict):
            nested_answer = output.get("answer")
            if isinstance(nested_answer, str):
                return nested_answer
    return ""


def _extract_reasoning_text(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    reasoning = payload.get("reasoning")
    if isinstance(reasoning, str):
        return reasoning
    if isinstance(reasoning, dict):
        text = reasoning.get("text")
        if isinstance(text, str):
            return text
    output = payload.get("output")
    if isinstance(output, dict):
        nested_reasoning = output.get("reasoning")
        if isinstance(nested_reasoning, str):
            return nested_reasoning
        if isinstance(nested_reasoning, dict):
            text = nested_reasoning.get("text")
            if isinstance(text, str):
                return text
    return ""


def _to_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _coerce_row_image(row: dict[str, Any]) -> Image.Image:
    image_value = row.get("image")
    if isinstance(image_value, Image.Image):
        return image_value.convert("RGB")
    decoded_image = row.get("Decoded_image")
    if isinstance(decoded_image, Image.Image):
        return decoded_image.convert("RGB")
    if isinstance(image_value, dict):
        image_bytes = image_value.get("bytes")
        if isinstance(image_bytes, (bytes, bytearray)):
            with Image.open(io.BytesIO(bytes(image_bytes))) as image:
                return image.convert("RGB")
        image_path = image_value.get("path")
        if isinstance(image_path, str) and Path(image_path).is_file():
            with Image.open(image_path) as image:
                return image.convert("RGB")
    if isinstance(image_value, str) and Path(image_value).is_file():
        with Image.open(image_value) as image:
            return image.convert("RGB")
    raise ValueError("row does not contain a decodable image")


def _normalize_whitespace(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def _normalize_compare_text(text: str) -> str:
    lowered = str(text or "").strip().lower()
    cleaned = re.sub(r"[^a-z0-9]+", " ", lowered)
    return " ".join(cleaned.split())


def format_mcq(options: list[str]) -> str:
    lines: list[str] = []
    for idx, option in enumerate(options):
        if idx >= len(OPTION_LETTERS):
            raise ValueError("too many options for letter assignment")
        letter = OPTION_LETTERS[idx]
        lines.append(f"{letter}. {option}")
    return "\n".join(lines)


def construct_mcq(options: list[str], answer: str) -> tuple[str, str]:
    lines: list[str] = []
    correct_letter = ""
    normalized_answer = _normalize_compare_text(answer)
    if normalized_answer in OPTION_LETTERS[: len(options)]:
        correct_letter = normalized_answer
    for idx, option in enumerate(options):
        if idx >= len(OPTION_LETTERS):
            raise ValueError("too many options for letter assignment")
        letter = OPTION_LETTERS[idx]
        lines.append(f"{letter}. {option}")
        if _normalize_compare_text(option) == normalized_answer:
            correct_letter = letter
    if not correct_letter:
        raise ValueError(f"answer {answer!r} not found in options")
    return "\n".join(lines), correct_letter


def build_question_prompt(question: str, options: list[str]) -> str:
    return "\n".join(
        [
            "You'll be given an image, an instruction and some choices.",
            "You have to select the correct one.",
            "Do not explain your reasoning.",
            "Answer with the option's letter from the given choices directly.",
            "",
            f"Question: {_normalize_whitespace(question)}",
            "Choices:",
            format_mcq(options),
            "",
            "Answer:",
        ]
    )


def _extract_json_candidate(text: str) -> Optional[dict[str, Any]]:
    stripped = str(text or "").strip()
    candidates: list[str] = []
    if stripped.startswith("{") and stripped.endswith("}"):
        candidates.append(stripped)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(stripped[start : end + 1])
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def normalize_prediction_letter(raw_text: str, options: list[str]) -> Optional[str]:
    stripped = str(raw_text or "").strip()
    if not stripped:
        return None

    option_map: dict[str, str] = {}
    normalized_option_map: dict[str, str] = {}
    for idx, option in enumerate(options):
        if idx >= len(OPTION_LETTERS):
            break
        letter = OPTION_LETTERS[idx]
        option_map[letter] = option
        normalized_option_map[_normalize_compare_text(option)] = letter

    payload = _extract_json_candidate(stripped)
    if payload is not None:
        for key in ("answer", "option", "choice", "letter", "selected_option"):
            value = payload.get(key)
            if isinstance(value, str):
                normalized = normalize_prediction_letter(value, options)
                if normalized is not None:
                    return normalized

    normalized_text = _normalize_compare_text(stripped)
    if normalized_text in option_map:
        return normalized_text
    if normalized_text in normalized_option_map:
        return normalized_option_map[normalized_text]

    for pattern in LETTER_PATTERNS:
        match = pattern.search(stripped)
        if match is None:
            continue
        letter = match.group(1).lower()
        if letter in option_map:
            return letter

    matched_letters: list[str] = []
    for normalized_option, letter in normalized_option_map.items():
        if normalized_option and normalized_option in normalized_text:
            matched_letters.append(letter)
    unique_letters = sorted(set(matched_letters))
    if len(unique_letters) == 1:
        return unique_letters[0]
    return None


def _load_dataset_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[int]]:
    load_kwargs: dict[str, Any] = {
        "split": str(args.split),
        "revision": str(args.hf_dataset_revision),
    }
    if str(args.hf_cache_dir or "").strip():
        load_kwargs["cache_dir"] = str(args.hf_cache_dir)
    if str(args.hf_token or "").strip():
        load_kwargs["token"] = str(args.hf_token)

    try:
        dataset = load_dataset(str(args.hf_dataset_repo_id), **load_kwargs)
    except TypeError:
        token_value = load_kwargs.pop("token", None)
        if token_value:
            load_kwargs["use_auth_token"] = token_value
        dataset = load_dataset(str(args.hf_dataset_repo_id), **load_kwargs)

    rows = [dict(item) for item in dataset if isinstance(item, dict)]
    indices = list(range(len(rows)))
    if int(args.max_samples) > 0 and len(indices) > int(args.max_samples):
        rng = random.Random(int(args.seed))
        indices = rng.sample(indices, k=int(args.max_samples))
    return rows, indices


def _row_str(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
        if value is not None and key in row:
            return str(value)
    return ""


def _row_options(row: dict[str, Any]) -> list[str]:
    for key in ("options", "Options"):
        value = row.get(key)
        if isinstance(value, list):
            return [_normalize_whitespace(str(item)) for item in value if str(item).strip()]
    letter_keys = ("A", "B", "C", "D", "E", "F")
    letter_options = [_normalize_whitespace(str(row.get(key) or "")) for key in letter_keys]
    letter_options = [item for item in letter_options if item]
    if letter_options:
        return letter_options
    raise ValueError("row is missing options")


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round(0.95 * (len(ordered) - 1)))
    return float(ordered[index])


def run_benchmark(
    *,
    args: argparse.Namespace,
    call_api_fn=_call_query_api,
) -> dict[str, Any]:
    load_dotenv_if_available(args.env_file)
    args.hf_token = resolve_hf_token(str(args.hf_token))
    api_key = _resolve_api_key(args)
    base_url = _resolve_base_url(str(args.base_url))
    try:
        model_resolution = shared_query_common.resolve_query_inference_model(
            api_base=base_url,
            api_key=api_key,
            model=str(args.model),
            finetune_id=str(args.finetune_id),
            checkpoint_step=(None if int(args.checkpoint_step) < 0 else int(args.checkpoint_step)),
            timeout=float(args.timeout),
        )
    except ValueError as exc:
        raise SystemExit(str(exc))
    model = model_resolution.model
    rows, selected_indices = _load_dataset_rows(args)

    total_scored = 0
    parse_success = 0
    correct = 0
    request_failures = 0
    latencies_ms: list[float] = []
    gold_letter_counts: Counter[str] = Counter()
    predicted_letter_counts: Counter[str] = Counter()
    accuracy_by_gold_letter: dict[str, list[int]] = defaultdict(list)
    accuracy_by_category: dict[str, list[int]] = defaultdict(list)
    predictions: list[dict[str, Any]] = []

    progress = tqdm(
        selected_indices,
        desc=f"benchmark:{args.split}",
        disable=not progress_enabled(bool(args.no_progress)),
    )
    for row_index in progress:
        row = rows[row_index]
        question = _row_str(row, "question", "Question")
        options = _row_options(row)
        answer = _row_str(row, "answer", "Answer")
        category = _row_str(row, "category", "Category") or "unknown"
        reasoning_reference = _row_str(row, "reasoning", "Reasoning")
        example_id = _row_str(row, "id", "ID") or str(row_index)
        if not question or not options or not answer:
            request_failures += 1
            predictions.append(
                {
                    "id": example_id,
                    "row_index": int(row_index),
                    "error": "row missing required fields",
                }
            )
            continue
        mcq_text, gold_letter = construct_mcq(options, answer)
        gold_letter_counts[gold_letter] += 1
        prompt = build_question_prompt(question, options)
        error_text = ""
        latency_ms = 0.0
        response_payload: dict[str, Any] = {}
        predicted_answer_text = ""
        predicted_reasoning_text = ""

        try:
            image = _coerce_row_image(row)
            try:
                image_url = _to_data_url(image)
            finally:
                image.close()
        except Exception as exc:
            request_failures += 1
            error_text = f"image_error={type(exc).__name__}: {exc}"
            predictions.append(
                {
                    "id": example_id,
                    "row_index": int(row_index),
                    "question": question,
                    "options": options,
                    "gold_answer_text": answer,
                    "gold_answer_letter": gold_letter,
                    "error": error_text,
                }
            )
            continue

        try:
            response_payload, latency_ms = call_api_fn(
                api_base=base_url,
                api_key=api_key,
                model=model,
                question=prompt,
                image_url=image_url,
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                max_tokens=int(args.max_tokens),
                reasoning=bool(args.reasoning),
                timeout=float(args.timeout),
                retry_429_max_retries=int(args.retry_429_max_retries),
                retry_429_backoff_s=float(args.retry_429_backoff_s),
                retry_429_max_backoff_s=float(args.retry_429_max_backoff_s),
            )
            predicted_answer_text = _extract_answer_text(response_payload)
            predicted_reasoning_text = _extract_reasoning_text(response_payload)
            latencies_ms.append(float(latency_ms))
        except QueryAPIError as exc:
            request_failures += 1
            error_text = f"{exc} | request_id={exc.request_id} | body={truncate_text(exc.response_body)}"
        except Exception as exc:  # pragma: no cover
            request_failures += 1
            error_text = f"{type(exc).__name__}: {exc}"

        predicted_letter = normalize_prediction_letter(predicted_answer_text, options)
        if predicted_letter is not None:
            parse_success += 1
            predicted_letter_counts[predicted_letter] += 1
        is_correct = bool(predicted_letter == gold_letter)
        total_scored += 1
        correct += int(is_correct)
        accuracy_by_gold_letter[gold_letter].append(int(is_correct))
        accuracy_by_category[category].append(int(is_correct))

        predictions.append(
            {
                "id": example_id,
                "row_index": int(row_index),
                "category": category,
                "question": question,
                "mcq": mcq_text,
                "options": options,
                "gold_answer_text": answer,
                "gold_answer_letter": gold_letter,
                "gold_reasoning": reasoning_reference,
                "predicted_answer_text": predicted_answer_text,
                "predicted_reasoning_text": predicted_reasoning_text,
                "predicted_answer_letter": predicted_letter,
                "is_correct": is_correct,
                "parse_success": bool(predicted_letter is not None),
                "latency_ms": float(latency_ms),
                "error": error_text,
                "raw_response": response_payload,
            }
        )

    metrics: dict[str, Any] = {
        "benchmark_mode": "zero_shot_single_image_mcq",
        "dataset_repo_id": str(args.hf_dataset_repo_id),
        "dataset_revision": str(args.hf_dataset_revision),
        "base_url": base_url,
        "model": model,
        "finetune_id": str(model_resolution.finetune_id or args.finetune_id or ""),
        "checkpoint_step": int(model_resolution.resolved_checkpoint_step)
        if model_resolution.resolved_checkpoint_step is not None
        else -1,
        "requested_checkpoint_step": int(model_resolution.requested_checkpoint_step)
        if model_resolution.requested_checkpoint_step is not None
        else -1,
        "resolved_checkpoint_step": int(model_resolution.resolved_checkpoint_step)
        if model_resolution.resolved_checkpoint_step is not None
        else -1,
        "split": str(args.split),
        "seed": int(args.seed),
        "reasoning_enabled": bool(args.reasoning),
        "requested_rows": len(selected_indices),
        "evaluated_rows": total_scored,
        "request_failures": int(request_failures),
        "accuracy": float(correct) / float(max(1, total_scored)),
        "correct": int(correct),
        "parse_rate": float(parse_success) / float(max(1, total_scored)),
        "latency_ms_mean": (sum(latencies_ms) / len(latencies_ms)) if latencies_ms else 0.0,
        "latency_ms_p95": _p95(latencies_ms),
        "gold_letter_counts": dict(sorted(gold_letter_counts.items())),
        "predicted_letter_counts": dict(sorted(predicted_letter_counts.items())),
        "accuracy_by_gold_letter": {
            letter: (sum(values) / len(values) if values else 0.0)
            for letter, values in sorted(accuracy_by_gold_letter.items())
        },
        "accuracy_by_category": {
            category: (sum(values) / len(values) if values else 0.0)
            for category, values in sorted(accuracy_by_category.items())
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
    _validate_args(args)
    metrics = run_benchmark(args=args)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
