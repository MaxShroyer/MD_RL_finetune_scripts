from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import random
import re
import string
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Iterator, Mapping, Optional, Sequence

from PIL import Image

from finetune_checkpoints import format_checkpoint_steps, resolve_checkpoint_step, save_checkpoint_step
from tuna_sdk import (
    DetectAnnotation,
    DetectGroundTruth,
    DetectRequest,
    DetectSettings,
    PointAnnotation,
    PointGroundTruth,
    PointRequest,
    PointSettings,
    QueryRequest,
    QuerySettings,
)
from tuna_sdk.errors import TunaAPIError, TunaNetworkError

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover
    load_dotenv = None  # type: ignore[assignment]

try:
    from huggingface_hub import snapshot_download
except ModuleNotFoundError:  # pragma: no cover
    snapshot_download = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_ROOT = Path(__file__).resolve().parent

DEFAULT_HF_REPO_ID = "Kingdrone-Junjue/DisasterM3"
DEFAULT_DATASET_ROOT = MODULE_ROOT
DEFAULT_DATASET_DIR = MODULE_ROOT / "dataset"
DEFAULT_OUTPUT_DIR = DEFAULT_DATASET_DIR / "full"
DEFAULT_SUBSET_OUTPUT_DIR = DEFAULT_DATASET_DIR / "subset_1000"
DEFAULT_RAW_DIR = MODULE_ROOT / "raw_dataset"
DEFAULT_PANEL_MAX_SIDE = 1024
DEFAULT_DIVIDER_PX = 4
DEFAULT_JPEG_QUALITY = 92
DEFAULT_BASE_MODEL = "moondream3-preview"
DEFAULT_BASE_URL = "https://api-staging.moondream.ai/v1"
DEFAULT_API_KEY_ENV_VAR = "CICID_GPUB_MOONDREAM_API_KEY_1"
DEFAULT_BENCHMARK_OUTPUT_DIR = REPO_ROOT / "outputs" / "benchmarks" / "disaster_m3"
DEFAULT_ASYNC_CHECKPOINT_EVAL_DIR = REPO_ROOT / "outputs" / "async_checkpoint_eval" / "disaster_m3"

RGB_SKIP_TOKENS = (
    "sar",
    "sentinel-1",
    "s1_",
    "_sar",
    "lidar",
    "hyperspectral",
    "hsi",
    "pointcloud",
    "mask",
    "masks",
)
SEGMENTATION_TOKENS = ("segmentation", "segment", "mask", "polygon", "polygons", "rle")
DESCRIPTION_TASK_TOKENS = ("caption", "description", "report")
RECOVERY_TASK_TOKENS = ("recovery", "restoration", "advice")
COUNTING_TASK_TOKENS = ("count", "counting")
POINT_TASK_TOKENS = ("point", "centroid", "center")
DETECT_TASK_TOKENS = ("bbox", "box", "localization", "locate", "detect")
MULTI_ANSWER_TASKS = {"bearing_body"}
NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}
FINETUNED_MODEL_RE = re.compile(
    r"^(?P<base_model>moondream3-preview)/(?P<finetune_id>[0-9A-Za-z_-]+)(?:@(?P<checkpoint_step>\d+))?$"
)
LETTER_RE = re.compile(r"\b([A-Z])\b")
INTEGER_RE = re.compile(r"-?\d+")
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def repo_relative(*parts: str) -> Path:
    return MODULE_ROOT.joinpath(*parts)


def random_suffix(length: int = 6) -> str:
    chars = string.ascii_lowercase + string.digits
    return "".join(random.choices(chars, k=length))


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def token_f1(reference: str, prediction: str) -> float:
    ref_tokens = normalize_text(reference).split()
    pred_tokens = normalize_text(prediction).split()
    if not ref_tokens and not pred_tokens:
        return 1.0
    if not ref_tokens or not pred_tokens:
        return 0.0
    ref_counts: dict[str, int] = {}
    pred_counts: dict[str, int] = {}
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    overlap = 0
    for token, count in ref_counts.items():
        overlap += min(count, pred_counts.get(token, 0))
    if overlap <= 0:
        return 0.0
    precision = overlap / float(len(pred_tokens))
    recall = overlap / float(len(ref_tokens))
    denom = precision + recall
    return 0.0 if denom <= 0.0 else (2.0 * precision * recall) / denom


def set_f1(reference: Iterable[Any], prediction: Iterable[Any]) -> float:
    ref_set = {str(item) for item in reference}
    pred_set = {str(item) for item in prediction}
    if not ref_set and not pred_set:
        return 1.0
    if not ref_set or not pred_set:
        return 0.0
    overlap = len(ref_set & pred_set)
    precision = overlap / float(len(pred_set))
    recall = overlap / float(len(ref_set))
    denom = precision + recall
    return 0.0 if denom <= 0.0 else (2.0 * precision * recall) / denom


def brevity_score(reference: str, prediction: str, *, lower: float = 0.6, upper: float = 1.6) -> float:
    ref_len = max(1, len(normalize_text(reference).split()))
    pred_len = len(normalize_text(prediction).split())
    if pred_len <= 0:
        return 0.0
    ratio = float(pred_len) / float(ref_len)
    if lower <= ratio <= upper:
        return 1.0
    if ratio < lower:
        return max(0.0, ratio / lower)
    return max(0.0, 1.0 - min(1.0, (ratio - upper) / max(upper, 1e-6)))


def relative_count_score(reference_count: float, predicted_count: float) -> float:
    ref_value = max(0.0, float(reference_count))
    pred_value = max(0.0, float(predicted_count))
    if ref_value == 0.0 and pred_value == 0.0:
        return 1.0
    denom = max(1.0, ref_value)
    error = abs(pred_value - ref_value) / float(denom)
    return clamp(1.0 - error)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def resolve_config_path(raw_path: str, *, script_dir: Path) -> Path:
    path = Path(str(raw_path or "")).expanduser()
    if path.is_absolute():
        return path
    for base in (Path.cwd(), REPO_ROOT, script_dir):
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    return (Path.cwd() / path).resolve()


def load_json_config(config_path: Path, *, default_path: Optional[Path] = None) -> dict[str, Any]:
    if not config_path.exists():
        if default_path is not None and config_path == default_path:
            return {}
        raise FileNotFoundError(f"Config file not found: {config_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a JSON object: {config_path}")
    return payload


def option_for_action(action: argparse.Action) -> str:
    for opt in action.option_strings:
        if opt.startswith("--"):
            return opt
    return action.option_strings[0]


def config_to_cli_args(
    parser: argparse.ArgumentParser,
    config: dict[str, Any],
    *,
    config_path: Path,
    overridden_dests: Optional[set[str]] = None,
) -> list[str]:
    overridden = set(overridden_dests or set())
    by_dest: dict[str, list[argparse.Action]] = {}
    for action in parser._actions:
        if not action.option_strings or action.dest == "help":
            continue
        by_dest.setdefault(action.dest, []).append(action)
    unknown = sorted(key for key in config if key not in by_dest)
    if unknown:
        raise ValueError(f"Unknown config key(s) in {config_path}: {unknown}")
    cli_args: list[str] = []
    for key, raw_value in config.items():
        if key in overridden:
            continue
        actions = by_dest[key]
        const_actions = [a for a in actions if isinstance(a, argparse._StoreConstAction)]
        store_actions = [a for a in actions if not isinstance(a, argparse._StoreConstAction)]
        if raw_value is None:
            matched = next((a for a in const_actions if getattr(a, "const", object()) is None), None)
            if matched is not None:
                cli_args.append(option_for_action(matched))
            continue
        if isinstance(raw_value, bool):
            matched = next((a for a in const_actions if getattr(a, "const", object()) is raw_value), None)
            if matched is not None:
                cli_args.append(option_for_action(matched))
                continue
        if not store_actions:
            continue
        action = store_actions[0]
        cli_args.append(option_for_action(action))
        if isinstance(raw_value, list):
            cli_args.extend(str(item) for item in raw_value)
        elif isinstance(raw_value, dict):
            cli_args.append(json.dumps(raw_value))
        else:
            cli_args.append(str(raw_value))
    return cli_args


def maybe_load_env_file(env_file: str) -> None:
    text = str(env_file or "").strip()
    if not text or load_dotenv is None:
        return
    load_dotenv(text, override=False)


def resolve_api_key(*, api_key: str, api_key_env_var: str, env_file: str) -> str:
    explicit = str(api_key or "").strip()
    if explicit:
        return explicit
    maybe_load_env_file(env_file)
    env_name = str(api_key_env_var or DEFAULT_API_KEY_ENV_VAR).strip() or DEFAULT_API_KEY_ENV_VAR
    value = str(os.environ.get(env_name) or "").strip()
    if value:
        return value
    fallback = str(os.environ.get("MOONDREAM_API_KEY") or "").strip()
    if fallback:
        return fallback
    raise ValueError(f"Missing API key. Checked explicit value, {env_name}, and MOONDREAM_API_KEY.")


def resolve_hf_token(hf_token: str, *, env_file: str) -> str:
    explicit = str(hf_token or "").strip()
    if explicit:
        return explicit
    maybe_load_env_file(env_file)
    return str(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or "").strip()


def resolve_path(raw_path: str, *, repo_root: Path, module_root: Path) -> Path:
    path = Path(str(raw_path or "")).expanduser()
    if path.is_absolute():
        return path.resolve()
    for base in (Path.cwd(), repo_root, module_root):
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    return (Path.cwd() / path).resolve()


def to_data_url(image: Image.Image, *, quality: int = DEFAULT_JPEG_QUALITY) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=max(1, min(100, int(quality))))
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


@dataclass(frozen=True)
class PanelPlacement:
    side: str
    x0: int
    y0: int
    width: int
    height: int
    source_width: int
    source_height: int
    blank: bool = False


@dataclass(frozen=True)
class CompositeLayout:
    panel_size: int
    divider_px: int
    output_width: int
    output_height: int
    left: PanelPlacement
    right: PanelPlacement


@dataclass(frozen=True)
class ImageReference:
    image_path: Path
    source_side: str


@dataclass(frozen=True)
class LabeledBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    class_name: str = ""
    source_side: str = "post"


@dataclass(frozen=True)
class LabeledPoint:
    x: float
    y: float
    class_name: str = ""
    source_side: str = "post"


@dataclass(frozen=True)
class InferenceModelResolution:
    model: str
    finetune_id: str
    requested_checkpoint_step: Optional[int]
    resolved_checkpoint_step: Optional[int]
    used_checkpoint_fallback: bool = False


@dataclass(frozen=True)
class MixedTaskRecord:
    row_id: str
    split: str
    task_name: str
    task_family: str
    skill: str
    image_path: Path
    question: str
    object_name: str
    final_answer_json: str
    answer_boxes_json: str
    answer_points_json: str
    source_pre_image_path: str
    source_post_image_path: str
    source_image_path: str
    metadata_json: str
    metadata: dict[str, Any] = field(repr=False)

    @classmethod
    def from_row(cls, row: Mapping[str, Any], *, dataset_dir: Path) -> "MixedTaskRecord":
        image_path = existing_path(str(row.get("image_path") or ""), dataset_dir=dataset_dir)
        if image_path is None:
            raise FileNotFoundError(f"image_path not found for row_id={row.get('row_id')!r}: {row.get('image_path')!r}")
        metadata_raw = row.get("metadata_json") or "{}"
        metadata = json.loads(str(metadata_raw or "{}"))
        if not isinstance(metadata, dict):
            metadata = {}
        return cls(
            row_id=str(row.get("row_id") or ""),
            split=str(row.get("split") or ""),
            task_name=str(row.get("task_name") or ""),
            task_family=str(row.get("task_family") or ""),
            skill=str(row.get("skill") or ""),
            image_path=image_path,
            question=str(row.get("question") or ""),
            object_name=str(row.get("object_name") or ""),
            final_answer_json=str(row.get("final_answer_json") or ""),
            answer_boxes_json=str(row.get("answer_boxes_json") or ""),
            answer_points_json=str(row.get("answer_points_json") or ""),
            source_pre_image_path=str(row.get("source_pre_image_path") or ""),
            source_post_image_path=str(row.get("source_post_image_path") or ""),
            source_image_path=str(row.get("source_image_path") or ""),
            metadata_json=str(metadata_raw),
            metadata=metadata,
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


class ImageIndex:
    def __init__(self, paths: Iterable[Path]) -> None:
        self.by_relative: dict[str, Path] = {}
        self.by_name: dict[str, list[Path]] = {}
        self.by_stem: dict[str, list[Path]] = {}
        for path in paths:
            resolved = path.resolve()
            rel = normalize_path_key(str(resolved))
            self.by_relative[rel] = resolved
            name_key = normalize_path_key(path.name)
            self.by_name.setdefault(name_key, []).append(resolved)
            stem_key = normalize_path_key(path.stem)
            self.by_stem.setdefault(stem_key, []).append(resolved)

    def resolve(self, raw_reference: str) -> Optional[Path]:
        text = normalize_path_key(raw_reference)
        if not text:
            return None
        direct = self.by_relative.get(text)
        if direct is not None:
            return direct
        basename = normalize_path_key(Path(text).name)
        if basename in self.by_name and len(self.by_name[basename]) == 1:
            return self.by_name[basename][0]
        stem = normalize_path_key(Path(text).stem)
        candidates = list(self.by_name.get(basename, [])) + list(self.by_stem.get(stem, []))
        if not candidates:
            return None
        unique = sorted({str(path): path for path in candidates}.values(), key=lambda item: len(str(item)))
        return unique[0]


def normalize_path_key(value: Any) -> str:
    text = str(value or "").strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text.lower()


def existing_path(raw_path: str, *, dataset_dir: Path) -> Optional[Path]:
    text = str(raw_path or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if path.is_file():
        return path.resolve()
    if not path.is_absolute():
        joined = (dataset_dir / path).resolve()
        if joined.is_file():
            return joined
    return None


def load_local_jsonl_rows(*, dataset_dir: Path, split_name: str) -> list[dict[str, Any]]:
    path = dataset_dir / "jsonl" / f"{split_name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"split JSONL not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            rows.append(payload)
    if not rows:
        raise ValueError(f"split={split_name} contains no rows")
    return rows


def load_mixed_records(*, dataset_dir: Path, split_name: str) -> list[MixedTaskRecord]:
    return [MixedTaskRecord.from_row(row, dataset_dir=dataset_dir) for row in load_local_jsonl_rows(dataset_dir=dataset_dir, split_name=split_name)]


def truncate(text: str, limit: int = 600) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "...<truncated>"


def error_message(exc: Exception) -> str:
    if isinstance(exc, TunaAPIError):
        request_id = f" request_id={exc.request_id}" if getattr(exc, "request_id", "") else ""
        return f"TunaAPIError status={exc.status_code}{request_id} message={exc}"
    if isinstance(exc, TunaNetworkError):
        cause = getattr(exc, "cause", None)
        if cause is not None:
            return f"TunaNetworkError message={exc} cause={type(cause).__name__}: {cause}"
    if isinstance(exc, QueryAPIError):
        request_id = f" request_id={exc.request_id}" if exc.request_id else ""
        response_body = f" body={truncate(exc.response_body)}" if exc.response_body else ""
        return f"QueryAPIError status={exc.status_code}{request_id} message={exc}{response_body}"
    return f"{type(exc).__name__}: {exc}"


def build_auth_headers(api_key: str) -> dict[str, str]:
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


def _extract_answer_text(payload: Any) -> str:
    if isinstance(payload, dict):
        answer = payload.get("answer")
        if isinstance(answer, str):
            return answer
        output = payload.get("output")
        if isinstance(output, dict):
            nested = output.get("answer")
            if isinstance(nested, str):
                return nested
    return ""


def _extract_boxes(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        objects = payload.get("objects")
        if isinstance(objects, list):
            return [dict(item) for item in objects if isinstance(item, dict)]
        output = payload.get("output")
        if isinstance(output, dict):
            nested = output.get("objects")
            if isinstance(nested, list):
                return [dict(item) for item in nested if isinstance(item, dict)]
    return []


def _extract_points(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        points = payload.get("points")
        if isinstance(points, list):
            return [dict(item) for item in points if isinstance(item, dict)]
        output = payload.get("output")
        if isinstance(output, dict):
            nested = output.get("points")
            if isinstance(nested, list):
                return [dict(item) for item in nested if isinstance(item, dict)]
    return []


def _http_error_details(exc: urllib.error.HTTPError) -> tuple[str, str]:
    request_id = str(exc.headers.get("x-request-id") or exc.headers.get("X-Request-Id") or "")
    body_text = ""
    try:
        body_text = exc.read().decode("utf-8", errors="replace")
    except Exception:
        body_text = ""
    return request_id, body_text


def resolve_model_identifier(*, model: str, finetune_id: str, checkpoint_step: Optional[int]) -> str:
    if str(model or "").strip():
        return str(model).strip()
    ftid = str(finetune_id or "").strip()
    if ftid:
        if checkpoint_step is not None:
            return f"moondream3-preview/{ftid}@{int(checkpoint_step)}"
        return f"moondream3-preview/{ftid}"
    return DEFAULT_BASE_MODEL


def _parse_finetuned_model(model: str) -> Optional[tuple[str, Optional[int]]]:
    match = FINETUNED_MODEL_RE.fullmatch(str(model or "").strip())
    if not match:
        return None
    finetune_id = str(match.group("finetune_id") or "").strip()
    checkpoint_raw = match.group("checkpoint_step")
    checkpoint_step = None if checkpoint_raw is None else int(checkpoint_raw)
    return finetune_id, checkpoint_step


def resolve_inference_model(
    *,
    api_base: str,
    api_key: str,
    model: str,
    finetune_id: str,
    checkpoint_step: Optional[int],
    timeout: float = 180.0,
    fallback_policy: str = "nearest_saved",
    checkpoint_ready_max_wait_s: float = 0.0,
    checkpoint_ready_poll_interval_s: float = 5.0,
) -> InferenceModelResolution:
    raw_model = str(model or "").strip()
    raw_finetune_id = str(finetune_id or "").strip()
    requested_step = None if checkpoint_step is None else int(checkpoint_step)
    parsed_model = _parse_finetuned_model(raw_model) if raw_model else None
    if raw_model:
        if parsed_model is None:
            return InferenceModelResolution(
                model=raw_model,
                finetune_id="",
                requested_checkpoint_step=None,
                resolved_checkpoint_step=None,
            )
        parsed_finetune_id, parsed_step = parsed_model
        if parsed_step is None:
            raise ValueError(
                "finetuned model strings must include a checkpoint step. "
                "Use moondream3-preview/{finetune_id}@{step}."
            )
        raw_finetune_id = parsed_finetune_id
        requested_step = int(parsed_step)
    elif raw_finetune_id:
        if requested_step is None:
            raise ValueError("--finetune-id requires --checkpoint-step for finetuned inference.")
    else:
        return InferenceModelResolution(
            model=DEFAULT_BASE_MODEL,
            finetune_id="",
            requested_checkpoint_step=None,
            resolved_checkpoint_step=None,
        )
    if requested_step is None:
        raise ValueError("checkpoint step is required for finetuned inference.")
    resolved_step, used_fallback = resolve_checkpoint_step(
        api_base=api_base,
        api_key=api_key,
        finetune_id=raw_finetune_id,
        requested_step=int(requested_step),
        timeout=timeout,
        fallback_policy=fallback_policy,
        ready_max_wait_s=checkpoint_ready_max_wait_s,
        ready_poll_interval_s=checkpoint_ready_poll_interval_s,
    )
    if used_fallback:
        print(
            "checkpoint resolution: "
            f"finetune_id={raw_finetune_id} requested_step={requested_step} "
            f"resolved_step={resolved_step} policy={fallback_policy}"
        )
    return InferenceModelResolution(
        model=f"{DEFAULT_BASE_MODEL}/{raw_finetune_id}@{resolved_step}",
        finetune_id=raw_finetune_id,
        requested_checkpoint_step=int(requested_step),
        resolved_checkpoint_step=int(resolved_step),
        used_checkpoint_fallback=used_fallback,
    )


def build_inference_payload(
    *,
    skill: str,
    model: str,
    question: str,
    object_name: str,
    image_url: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    reasoning: Optional[bool],
    max_objects: Optional[int] = None,
) -> dict[str, Any]:
    settings: dict[str, Any] = {
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": int(max_tokens),
    }
    payload: dict[str, Any] = {
        "model": model,
        "image_url": image_url,
        "settings": settings,
    }
    if skill == "query":
        payload["question"] = question
        if reasoning is not None:
            payload["reasoning"] = bool(reasoning)
    else:
        payload["object"] = object_name
        if skill == "detect" and max_objects is not None:
            payload["settings"]["max_objects"] = int(max_objects)
    return payload


def call_inference_api(
    *,
    skill: str,
    api_base: str,
    api_key: str,
    model: str,
    question: str,
    object_name: str,
    image_url: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    reasoning: Optional[bool],
    timeout: float,
    retry_429_max_retries: int,
    retry_429_backoff_s: float,
    retry_429_max_backoff_s: float,
    retry_5xx_max_retries: int,
    retry_5xx_backoff_s: float,
    retry_5xx_max_backoff_s: float,
    max_objects: Optional[int] = None,
) -> tuple[Any, dict[str, Any], float]:
    payload = build_inference_payload(
        skill=skill,
        model=model,
        question=question,
        object_name=object_name,
        image_url=image_url,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        reasoning=reasoning,
        max_objects=max_objects,
    )
    endpoint = api_base.rstrip("/") + f"/{skill}"
    retry_429_attempt = 0
    retry_5xx_attempt = 0
    retry_429_limit = max(0, int(retry_429_max_retries))
    retry_5xx_limit = max(0, int(retry_5xx_max_retries))
    while True:
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=build_auth_headers(api_key),
            method="POST",
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=float(timeout)) as response:
                body = response.read().decode("utf-8", errors="replace")
            latency_ms = (time.monotonic() - started) * 1000.0
            data = json.loads(body) if body else {}
            if not isinstance(data, dict):
                data = {}
            if skill == "query":
                return _extract_answer_text(data), data, latency_ms
            if skill == "detect":
                return _extract_boxes(data), data, latency_ms
            return _extract_points(data), data, latency_ms
        except urllib.error.HTTPError as exc:
            latency_ms = (time.monotonic() - started) * 1000.0
            request_id, body_text = _http_error_details(exc)
            retry_after_s = 0.0
            if exc.code == 429:
                header = (exc.headers.get("Retry-After") or "").strip()
                if header:
                    try:
                        retry_after_s = max(0.0, float(header))
                    except (TypeError, ValueError):
                        retry_after_s = 0.0
            if exc.code == 429 and retry_429_attempt < retry_429_limit:
                exp_backoff = max(0.0, float(retry_429_backoff_s)) * (2.0**retry_429_attempt)
                sleep_s = max(retry_after_s, min(max(0.0, float(retry_429_max_backoff_s)), exp_backoff))
                print(
                    f"{skill} retry: status=429 attempt={retry_429_attempt + 1}/{retry_429_limit + 1} "
                    f"sleep={sleep_s:.2f}s latency_ms={latency_ms:.1f} "
                    f"request_id={request_id or '-'} body={truncate(body_text)}"
                )
                if sleep_s > 0.0:
                    time.sleep(sleep_s)
                retry_429_attempt += 1
                continue
            if 500 <= exc.code <= 599 and retry_5xx_attempt < retry_5xx_limit:
                exp_backoff = max(0.0, float(retry_5xx_backoff_s)) * (2.0**retry_5xx_attempt)
                sleep_s = max(retry_after_s, min(max(0.0, float(retry_5xx_max_backoff_s)), exp_backoff))
                print(
                    f"{skill} retry: status={exc.code} attempt={retry_5xx_attempt + 1}/{retry_5xx_limit + 1} "
                    f"sleep={sleep_s:.2f}s latency_ms={latency_ms:.1f} "
                    f"request_id={request_id or '-'} body={truncate(body_text)}"
                )
                if sleep_s > 0.0:
                    time.sleep(sleep_s)
                retry_5xx_attempt += 1
                continue
            raise QueryAPIError(
                f"HTTP {exc.code} {exc.reason}",
                status_code=int(exc.code),
                request_id=request_id,
                response_body=body_text,
            ) from exc
        except urllib.error.URLError as exc:
            raise QueryAPIError(f"URL error: {exc}") from exc


def load_image_index(search_root: Path) -> ImageIndex:
    paths = [path for path in search_root.rglob("*") if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}]
    return ImageIndex(paths)


def snapshot_download_dataset(*, hf_repo_id: str, raw_dir: Path, hf_token: str) -> Path:
    if snapshot_download is None:
        raise RuntimeError("huggingface_hub is required for dataset download. Install with `pip install huggingface_hub`.")
    raw_dir.mkdir(parents=True, exist_ok=True)
    return Path(
        snapshot_download(
            repo_id=str(hf_repo_id),
            repo_type="dataset",
            token=str(hf_token or "") or None,
            local_dir=str(raw_dir),
            local_dir_use_symlinks=False,
        )
    ).resolve()


def extract_zip_if_needed(zip_path: Path, output_dir: Path) -> Path:
    if output_dir.exists() and any(output_dir.iterdir()):
        return output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as handle:
        handle.extractall(output_dir)
    return output_dir.resolve()


def discover_extracted_root(raw_dir: Path, *, archive_stem: str) -> Optional[Path]:
    direct = raw_dir / archive_stem
    if direct.exists():
        return direct.resolve()
    zip_path = raw_dir / f"{archive_stem}.zip"
    if zip_path.exists():
        return extract_zip_if_needed(zip_path, raw_dir / archive_stem)
    candidates = [path for path in raw_dir.glob(f"{archive_stem}*") if path.is_dir()]
    return candidates[0].resolve() if candidates else None


def is_rgb_reference(raw_reference: str) -> bool:
    text = normalize_path_key(raw_reference)
    if not text:
        return False
    return not any(token in text for token in RGB_SKIP_TOKENS)


def is_segmentation_task(task_name: str, row: Mapping[str, Any]) -> bool:
    task_text = normalize_text(task_name)
    if any(token in task_text for token in SEGMENTATION_TOKENS):
        return True
    for key in row.keys():
        if any(token in normalize_text(key) for token in SEGMENTATION_TOKENS):
            return True
    return False


def humanize_label(value: str) -> str:
    text = str(value or "").strip().replace("_", " ").replace("-", " ")
    return " ".join(text.split())


def infer_task_family(task_name: str, *, skill: str) -> str:
    task_text = normalize_text(task_name)
    if any(token in task_text for token in RECOVERY_TASK_TOKENS):
        return "restoration_advice"
    if any(token in task_text for token in DESCRIPTION_TASK_TOKENS):
        return "description"
    if any(token in task_text for token in COUNTING_TASK_TOKENS):
        return "counting"
    if "reason" in task_text:
        return "reasoning"
    if skill == "detect" or skill == "point":
        return "localization"
    return "recognition"


def question_from_messages(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    user_chunks: list[str] = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        if str(item.get("role") or "").strip().lower() != "user":
            continue
        content = item.get("content")
        if isinstance(content, str):
            user_chunks.append(content)
        elif isinstance(content, list):
            for chunk in content:
                if isinstance(chunk, dict) and str(chunk.get("type") or "").strip().lower() == "text":
                    user_chunks.append(str(chunk.get("text") or ""))
    return " ".join(text.strip() for text in user_chunks if str(text or "").strip())


def answer_from_messages(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    for item in messages:
        if not isinstance(item, dict):
            continue
        if str(item.get("role") or "").strip().lower() != "assistant":
            continue
        content = item.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = [str(chunk.get("text") or "").strip() for chunk in content if isinstance(chunk, dict)]
            joined = " ".join(part for part in parts if part)
            if joined:
                return joined
    return ""


def parse_prediction_json(answer_text: str) -> Optional[dict[str, Any]]:
    text = str(answer_text or "").strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    while start >= 0:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
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
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : index + 1]
                    try:
                        payload = json.loads(candidate)
                    except json.JSONDecodeError:
                        break
                    if isinstance(payload, dict):
                        return payload
                    break
        start = text.find("{", start + 1)
    return None


def parse_mcq_letters(value: Any) -> list[str]:
    if isinstance(value, list):
        out = []
        for item in value:
            text = str(item or "").strip().upper()
            if len(text) == 1 and "A" <= text <= "Z":
                out.append(text)
        return sorted(set(out))
    text = str(value or "").strip().upper()
    if not text:
        return []
    return sorted(set(match.group(1) for match in LETTER_RE.finditer(text)))


def parse_int_from_text(value: Any) -> Optional[int]:
    text = normalize_text(value)
    if not text:
        return None
    match = INTEGER_RE.search(text)
    if match:
        try:
            return int(match.group(0))
        except ValueError:
            return None
    for word, number in NUMBER_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", text):
            return number
    return None


def parse_number_from_text(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = normalize_text(value)
    if not text:
        return None
    match = NUMBER_RE.search(text.replace(",", ""))
    if match:
        try:
            return float(match.group(0))
        except ValueError:
            return None
    int_value = parse_int_from_text(text)
    return None if int_value is None else float(int_value)


def infer_split_from_path(path: Path, *, default_split: str = "") -> str:
    parts = [normalize_text(part) for part in path.parts]
    for part in parts:
        if part in {"train", "training"} or "train" in part:
            return "train"
        if part in {"val", "valid", "validation"} or "validation" in part or re.search(r"\bval\b", part):
            return "val"
        if part == "test" or "test" in part or "bench" in part or "benchmark" in part:
            return "test"
    return str(default_split or "").strip().lower()


def serialize_boxes(boxes: Sequence[DetectAnnotation]) -> str:
    return json.dumps(
        [
            {
                "x_min": float(box.x_min),
                "y_min": float(box.y_min),
                "x_max": float(box.x_max),
                "y_max": float(box.y_max),
            }
            for box in boxes
        ],
        separators=(",", ":"),
    )


def deserialize_boxes(value: Any) -> list[DetectAnnotation]:
    raw = value
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        raw = json.loads(text)
    if not isinstance(raw, list):
        return []
    out: list[DetectAnnotation] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            x_min = clamp(float(item.get("x_min")))
            y_min = clamp(float(item.get("y_min")))
            x_max = clamp(float(item.get("x_max")))
            y_max = clamp(float(item.get("y_max")))
        except (TypeError, ValueError):
            continue
        if x_max <= x_min or y_max <= y_min:
            continue
        out.append(DetectAnnotation(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max))
    return out


def serialize_points(points: Sequence[PointAnnotation]) -> str:
    return json.dumps(
        [{"x": float(point.x), "y": float(point.y)} for point in points],
        separators=(",", ":"),
    )


def deserialize_points(value: Any) -> list[PointAnnotation]:
    raw = value
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        raw = json.loads(text)
    if not isinstance(raw, list):
        return []
    out: list[PointAnnotation] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            out.append(PointAnnotation(x=clamp(float(item.get("x"))), y=clamp(float(item.get("y")))))
        except (TypeError, ValueError):
            continue
    return out


def load_json_payload(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def fit_image_into_panel(image: Image.Image, *, panel_size: int, fill_color: tuple[int, int, int]) -> tuple[Image.Image, PanelPlacement]:
    source = image.convert("RGB")
    src_w, src_h = source.size
    if src_w <= 0 or src_h <= 0:
        raise ValueError("image has invalid size")
    scale = min(float(panel_size) / float(src_w), float(panel_size) / float(src_h))
    scaled_w = max(1, int(round(src_w * scale)))
    scaled_h = max(1, int(round(src_h * scale)))
    resized = source.resize((scaled_w, scaled_h), resample=Image.Resampling.BICUBIC)
    panel = Image.new("RGB", (panel_size, panel_size), fill_color)
    x0 = max(0, (panel_size - scaled_w) // 2)
    y0 = max(0, (panel_size - scaled_h) // 2)
    panel.paste(resized, (x0, y0))
    placement = PanelPlacement(
        side="",
        x0=x0,
        y0=y0,
        width=scaled_w,
        height=scaled_h,
        source_width=src_w,
        source_height=src_h,
        blank=False,
    )
    return panel, placement


def blank_panel(*, panel_size: int, side: str, fill_color: tuple[int, int, int]) -> tuple[Image.Image, PanelPlacement]:
    return (
        Image.new("RGB", (panel_size, panel_size), fill_color),
        PanelPlacement(
            side=side,
            x0=0,
            y0=0,
            width=panel_size,
            height=panel_size,
            source_width=panel_size,
            source_height=panel_size,
            blank=True,
        ),
    )


def build_composite_image(
    *,
    pre_image: Optional[Image.Image],
    post_image: Optional[Image.Image],
    panel_size: int = DEFAULT_PANEL_MAX_SIDE,
    divider_px: int = DEFAULT_DIVIDER_PX,
    fill_color: tuple[int, int, int] = (242, 242, 242),
    divider_color: tuple[int, int, int] = (180, 180, 180),
) -> tuple[Image.Image, CompositeLayout]:
    if pre_image is None and post_image is None:
        raise ValueError("at least one image is required to build a composite")
    if pre_image is not None:
        left_panel, left_placement = fit_image_into_panel(pre_image, panel_size=panel_size, fill_color=fill_color)
        left_placement = PanelPlacement(side="pre", **{k: getattr(left_placement, k) for k in left_placement.__dataclass_fields__ if k != "side"})
    else:
        left_panel, left_placement = blank_panel(panel_size=panel_size, side="pre", fill_color=fill_color)
    if post_image is not None:
        right_panel, right_placement = fit_image_into_panel(post_image, panel_size=panel_size, fill_color=fill_color)
        right_placement = PanelPlacement(side="post", **{k: getattr(right_placement, k) for k in right_placement.__dataclass_fields__ if k != "side"})
    else:
        right_panel, right_placement = blank_panel(panel_size=panel_size, side="post", fill_color=fill_color)
    output_width = (panel_size * 2) + max(0, int(divider_px))
    composite = Image.new("RGB", (output_width, panel_size), divider_color)
    composite.paste(left_panel, (0, 0))
    composite.paste(right_panel, (panel_size + divider_px, 0))
    layout = CompositeLayout(
        panel_size=panel_size,
        divider_px=int(divider_px),
        output_width=output_width,
        output_height=panel_size,
        left=left_placement,
        right=PanelPlacement(
            side=right_placement.side,
            x0=right_placement.x0 + panel_size + divider_px,
            y0=right_placement.y0,
            width=right_placement.width,
            height=right_placement.height,
            source_width=right_placement.source_width,
            source_height=right_placement.source_height,
            blank=right_placement.blank,
        ),
    )
    return composite, layout


def _placement_for_side(layout: CompositeLayout, side: str) -> PanelPlacement:
    side_norm = normalize_text(side) or "post"
    if side_norm in {"pre", "left"}:
        return layout.left
    if side_norm in {"post", "right"}:
        return layout.right
    return layout.right if not layout.right.blank else layout.left


def remap_box(box: LabeledBox, *, layout: CompositeLayout) -> DetectAnnotation:
    placement = _placement_for_side(layout, box.source_side)
    if placement.blank:
        raise ValueError("cannot remap boxes onto a blank placement")
    x_min = placement.x0 + (clamp(box.x_min) * placement.width)
    x_max = placement.x0 + (clamp(box.x_max) * placement.width)
    y_min = placement.y0 + (clamp(box.y_min) * placement.height)
    y_max = placement.y0 + (clamp(box.y_max) * placement.height)
    return DetectAnnotation(
        x_min=clamp(x_min / float(layout.output_width)),
        y_min=clamp(y_min / float(layout.output_height)),
        x_max=clamp(x_max / float(layout.output_width)),
        y_max=clamp(y_max / float(layout.output_height)),
    )


def remap_point(point: LabeledPoint, *, layout: CompositeLayout) -> PointAnnotation:
    placement = _placement_for_side(layout, point.source_side)
    if placement.blank:
        raise ValueError("cannot remap points onto a blank placement")
    x = placement.x0 + (clamp(point.x) * placement.width)
    y = placement.y0 + (clamp(point.y) * placement.height)
    return PointAnnotation(
        x=clamp(x / float(layout.output_width)),
        y=clamp(y / float(layout.output_height)),
    )


def count_tp_fp_fn(
    predicted: Sequence[DetectAnnotation],
    ground_truth: Sequence[DetectAnnotation],
    *,
    iou_threshold: float = 0.5,
) -> tuple[int, int, int]:
    if not predicted and not ground_truth:
        return 0, 0, 0
    matched_gt: set[int] = set()
    tp = 0
    for pred in predicted:
        best_iou = 0.0
        best_index = -1
        for index, gt in enumerate(ground_truth):
            if index in matched_gt:
                continue
            iou = box_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_index = index
        if best_index >= 0 and best_iou >= float(iou_threshold):
            tp += 1
            matched_gt.add(best_index)
    fp = max(0, len(predicted) - tp)
    fn = max(0, len(ground_truth) - tp)
    return tp, fp, fn


def reward_f1(predicted: Sequence[DetectAnnotation], ground_truth: Sequence[DetectAnnotation], *, iou_threshold: float = 0.5) -> float:
    if not predicted and not ground_truth:
        return 1.0
    if not predicted or not ground_truth:
        return 0.0
    tp, fp, fn = count_tp_fp_fn(predicted, ground_truth, iou_threshold=iou_threshold)
    denom = (2 * tp) + fp + fn
    return 0.0 if denom <= 0 else (2.0 * float(tp)) / float(denom)


def reward_miou(predicted: Sequence[DetectAnnotation], ground_truth: Sequence[DetectAnnotation]) -> float:
    if not predicted and not ground_truth:
        return 1.0
    if not predicted or not ground_truth:
        return 0.0
    used_gt: set[int] = set()
    ious: list[float] = []
    for pred in predicted:
        best_iou = 0.0
        best_index = -1
        for index, gt in enumerate(ground_truth):
            if index in used_gt:
                continue
            iou = box_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_index = index
        if best_index >= 0:
            used_gt.add(best_index)
            ious.append(best_iou)
    denom = max(len(predicted), len(ground_truth))
    return sum(ious) / float(max(1, denom))


def reward_f1_points(points: Sequence[PointAnnotation], ground_truth: Sequence[DetectAnnotation]) -> float:
    if not points and not ground_truth:
        return 1.0
    if not points or not ground_truth:
        return 0.0
    matched_gt: set[int] = set()
    tp = 0
    for point in points:
        for index, gt in enumerate(ground_truth):
            if index in matched_gt:
                continue
            if gt.x_min <= point.x <= gt.x_max and gt.y_min <= point.y <= gt.y_max:
                matched_gt.add(index)
                tp += 1
                break
    fp = max(0, len(points) - tp)
    fn = max(0, len(ground_truth) - tp)
    denom = (2 * tp) + fp + fn
    return 0.0 if denom <= 0 else (2.0 * float(tp)) / float(denom)


def box_iou(a: DetectAnnotation, b: DetectAnnotation) -> float:
    inter_x0 = max(float(a.x_min), float(b.x_min))
    inter_y0 = max(float(a.y_min), float(b.y_min))
    inter_x1 = min(float(a.x_max), float(b.x_max))
    inter_y1 = min(float(a.y_max), float(b.y_max))
    if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
        return 0.0
    inter_area = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)
    area_a = max(0.0, float(a.x_max) - float(a.x_min)) * max(0.0, float(a.y_max) - float(a.y_min))
    area_b = max(0.0, float(b.x_max) - float(b.x_min)) * max(0.0, float(b.y_max) - float(b.y_min))
    union_area = area_a + area_b - inter_area
    return 0.0 if union_area <= 0.0 else inter_area / union_area


def parse_options_map(row: Mapping[str, Any]) -> dict[str, str]:
    raw_value = (
        row.get("options")
        or row.get("option_map")
        or row.get("choices")
        or row.get("options_list")
        or row.get("options_str")
        or row.get("option_str")
        or row.get("candidate_options")
    )
    if isinstance(raw_value, dict):
        return {str(key).strip().upper(): str(value).strip() for key, value in raw_value.items() if str(key).strip()}
    if isinstance(raw_value, list):
        out: dict[str, str] = {}
        for index, item in enumerate(raw_value):
            label = chr(ord("A") + index)
            out[label] = str(item).strip()
        return out
    text = str(raw_value or "").strip()
    if not text:
        return {}
    lines = [piece.strip() for piece in re.split(r"(?:\n|;)", text) if piece.strip()]
    out: dict[str, str] = {}
    for line in lines:
        match = re.match(r"^\(?([A-Z])[\).:\-]\s*(.+)$", line)
        if match:
            out[match.group(1).upper()] = match.group(2).strip()
    return out


def detect_query_kind(task_name: str, metadata: Mapping[str, Any]) -> str:
    query_kind = normalize_text(metadata.get("query_kind"))
    if query_kind:
        return query_kind
    task_text = normalize_text(task_name)
    if any(token in task_text for token in RECOVERY_TASK_TOKENS):
        return "recovery"
    if any(token in task_text for token in DESCRIPTION_TASK_TOKENS):
        return "description"
    if any(token in task_text for token in COUNTING_TASK_TOKENS):
        return "count"
    if task_name in MULTI_ANSWER_TASKS:
        return "multi_choice_multi_answer"
    options = metadata.get("options_by_letter")
    if isinstance(options, dict) and options:
        return "multi_choice_single_answer"
    return "free_text"


def prepare_request_for_record(
    record: MixedTaskRecord,
    *,
    temperature: float,
    top_p: float,
    max_tokens: int,
    reasoning: bool,
    image_quality: int = DEFAULT_JPEG_QUALITY,
    max_objects: Optional[int] = None,
) -> tuple[Any, Optional[Any]]:
    with Image.open(record.image_path) as image:
        image_url = to_data_url(image.convert("RGB"), quality=image_quality)
    if record.skill == "query":
        return (
            QueryRequest(
                question=record.question,
                image_url=image_url,
                reasoning=bool(reasoning),
                settings=QuerySettings(
                    temperature=float(temperature),
                    top_p=float(top_p),
                    max_tokens=int(max_tokens),
                ),
            ),
            None,
        )
    if record.skill == "point":
        gt_boxes = deserialize_boxes(record.answer_boxes_json)
        gt_points = deserialize_points(record.answer_points_json)
        ground_truth = None
        if gt_points:
            ground_truth = PointGroundTruth(points=gt_points)
        elif gt_boxes:
            ground_truth = PointGroundTruth(boxes=gt_boxes)
        return (
            PointRequest(
                object_name=record.object_name,
                image_url=image_url,
                settings=PointSettings(
                    temperature=float(temperature),
                    top_p=float(top_p),
                    max_tokens=int(max_tokens),
                ),
            ),
            ground_truth,
        )
    gt_boxes = deserialize_boxes(record.answer_boxes_json)
    return (
        DetectRequest(
            object_name=record.object_name,
            image_url=image_url,
            settings=DetectSettings(
                temperature=float(temperature),
                top_p=float(top_p),
                max_tokens=int(max_tokens),
                max_objects=None if max_objects is None else int(max_objects),
            ),
        ),
        DetectGroundTruth(boxes=gt_boxes),
    )


def progress_enabled(no_progress: bool) -> bool:
    if no_progress:
        return False
    return os.isatty(2)


def checkpoint_save_step(*, finetune: Any, context: str) -> Optional[int]:
    return save_checkpoint_step(finetune=finetune, context=context, error_formatter=error_message)
