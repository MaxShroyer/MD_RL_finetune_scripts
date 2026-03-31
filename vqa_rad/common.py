from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_ROOT = Path(__file__).resolve().parent

DEFAULT_DATASET_NAME = "flaviagiammarino/vqa-rad"
DEFAULT_STAGING_API_BASE = "https://api-staging.moondream.ai/v1"
DEFAULT_QUERY_OUTPUT_DIR = MODULE_ROOT / "outputs" / "vqa_rad_query_v1"
DEFAULT_QUERY_OUTPUT_DIR_V2 = MODULE_ROOT / "outputs" / "vqa_rad_query_v2"
DEFAULT_SHARED_IMAGE_DIR = MODULE_ROOT / "outputs" / "vqa_rad_query_images"

QUERY_TASK_TYPE = "vqa_rad_query"
ANSWER_TYPE_CLOSE = "close_ended"
ANSWER_TYPE_OPEN = "open_ended"
PROMPT_STYLE_LEGACY_JSON = "legacy_json_instruction"
PROMPT_STYLE_RAW_QUESTION = "raw_question"
PROMPT_STYLE_CHOICES = (
    PROMPT_STYLE_LEGACY_JSON,
    PROMPT_STYLE_RAW_QUESTION,
)
PREDICTION_FORMAT_STRICT_JSON = "strict_json"
PREDICTION_FORMAT_EMBEDDED_JSON = "embedded_json"
PREDICTION_FORMAT_PLAIN_TEXT = "plain_text"
PREDICTION_FORMAT_NONE = "none"
QUESTION_FAMILY_CHOICES = (
    "yes_no",
    "modality",
    "plane",
    "location",
    "abnormality",
    "organ_system",
    "count",
    "other",
)

QUESTION_TEMPLATE = "\n".join(
    [
        "Answer this radiology question using only the image.",
        'Return JSON only with the schema {{"answer":"..."}}.',
        'If the question is yes/no, answer exactly "yes" or "no".',
        "Otherwise answer with a short radiology phrase only.",
        "Do not explain.",
        "Question: {question}",
    ]
)

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
_NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")
_LEADING_YES_NO = re.compile(r"^(is|are|does|do|did|can|could|has|have|had|was|were|will)\b")
_ANSWER_LABEL_RE = re.compile(r"^(?:final\s+answer|answer)\s*[:\-]\s*", re.IGNORECASE)

_OPEN_ANSWER_ALIASES = {
    "x ray": "x ray",
    "xray": "x ray",
    "radiograph": "x ray",
    "radiography": "x ray",
    "ct": "ct",
    "ct scan": "ct",
    "computed tomography": "ct",
    "mri": "mri",
    "mr": "mri",
    "magnetic resonance imaging": "mri",
    "magnetic resonance image": "mri",
    "mri scan": "mri",
    "ultrasound": "ultrasound",
    "ultrasound scan": "ultrasound",
    "sonography": "ultrasound",
    "axial": "axial",
    "axial plane": "axial",
    "transverse": "axial",
    "transverse plane": "axial",
    "coronal": "coronal",
    "coronal plane": "coronal",
    "sagittal": "sagittal",
    "sagittal plane": "sagittal",
}


@dataclass(frozen=True)
class PredictionExtraction:
    answer: Optional[str]
    payload: Optional[dict[str, Any]]
    prediction_format: str
    strict_parse_success: bool
    json_object_parsed: bool


def repo_relative(*parts: str) -> Path:
    return MODULE_ROOT.joinpath(*parts)


def resolve_config_path(raw_path: str, *, script_dir: Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    from_cwd = (Path.cwd() / path).resolve()
    if from_cwd.exists():
        return from_cwd
    from_repo = (REPO_ROOT / path).resolve()
    if from_repo.exists():
        return from_repo
    from_script = (script_dir / path).resolve()
    if from_script.exists():
        return from_script
    return from_cwd


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
        raise ValueError(
            f"Unknown config key(s) in {config_path}: {unknown}. "
            "Remove typos or update script support."
        )

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
        cli_args.append(option_for_action(store_actions[0]))
        if isinstance(raw_value, list):
            cli_args.extend(str(item) for item in raw_value)
        elif isinstance(raw_value, dict):
            cli_args.append(json.dumps(raw_value))
        else:
            cli_args.append(str(raw_value))
    return cli_args


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def relative_path(from_dir: Path, to_path: Path) -> str:
    return os.path.relpath(str(to_path), start=str(from_dir))


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def json_object_candidates(text: str) -> list[str]:
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


def parse_prediction_json_strict(answer_text: str) -> Optional[dict[str, Any]]:
    text = str(answer_text or "").strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def parse_prediction_json(answer_text: str) -> Optional[dict[str, Any]]:
    strict_payload = parse_prediction_json_strict(answer_text)
    if strict_payload is not None:
        return strict_payload
    text = str(answer_text or "").strip()
    if not text:
        return None
    for candidate in json_object_candidates(text):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def tokenize_text(value: Any) -> list[str]:
    return [match.group(0).lower() for match in _TOKEN_PATTERN.finditer(str(value or ""))]


def token_f1(reference: Any, prediction: Any) -> float:
    ref_tokens = tokenize_text(reference)
    pred_tokens = tokenize_text(prediction)
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
    precision = overlap / max(1, len(pred_tokens))
    recall = overlap / max(1, len(ref_tokens))
    if precision + recall <= 0.0:
        return 0.0
    return (2.0 * precision * recall) / (precision + recall)


def normalize_free_text(value: Any) -> str:
    return " ".join(tokenize_text(value))


def normalize_question(value: Any) -> str:
    return normalize_free_text(value)


def normalize_close_answer(value: Any) -> str:
    text = normalize_free_text(value)
    if text in {"yes", "no"}:
        return text
    return text


def normalize_open_answer(value: Any) -> str:
    text = normalize_free_text(value)
    if not text:
        return ""
    return _OPEN_ANSWER_ALIASES.get(text, text)


def infer_answer_type(answer_text: Any) -> str:
    return ANSWER_TYPE_CLOSE if normalize_close_answer(answer_text) in {"yes", "no"} else ANSWER_TYPE_OPEN


def infer_question_family(question_text: Any) -> str:
    text = " ".join(str(question_text or "").strip().lower().split())
    if not text:
        return "other"
    if _LEADING_YES_NO.match(text):
        return "yes_no"
    if "organ system" in text:
        return "organ_system"
    if "plane" in text:
        return "plane"
    if "where is" in text or "location of" in text or "where are" in text:
        return "location"
    if "what abnormality" in text or "what condition" in text or "what finding" in text:
        return "abnormality"
    if "how many" in text or "number of" in text or "count" in text:
        return "count"
    if "how was this image taken" in text or "what modality" in text or "what type of image" in text:
        return "modality"
    return "other"


def numeric_match(reference: Any, prediction: Any) -> float:
    ref_numbers = _NUMBER_PATTERN.findall(str(reference or ""))
    pred_numbers = _NUMBER_PATTERN.findall(str(prediction or ""))
    if not ref_numbers and not pred_numbers:
        return 1.0
    if not ref_numbers or not pred_numbers:
        return 0.0
    return 1.0 if ref_numbers == pred_numbers else 0.0


def brevity_score(reference: Any, prediction: Any) -> float:
    ref_len = max(1, len(tokenize_text(reference)))
    pred_len = len(tokenize_text(prediction))
    if pred_len <= 0:
        return 0.0
    limit = max(ref_len + 4, int(round(ref_len * 2.0)))
    if pred_len <= limit:
        return 1.0
    overflow = pred_len - limit
    return clamp(1.0 - (float(overflow) / float(max(1, limit))))


def parse_answer_payload(payload: Any) -> Optional[str]:
    if not isinstance(payload, dict):
        return None
    answer = str(payload.get("answer") or "").strip()
    return answer or None


def _strip_wrapping_quotes(text: str) -> str:
    value = str(text or "").strip()
    while len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'", "`"}:
        value = value[1:-1].strip()
    return value


def clean_plain_text_answer(answer_text: Any, *, answer_type: Optional[str] = None) -> Optional[str]:
    text = str(answer_text or "").strip()
    if not text:
        return None
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    candidate = _strip_wrapping_quotes(lines[0])
    candidate = _ANSWER_LABEL_RE.sub("", candidate, count=1).strip()
    candidate = _strip_wrapping_quotes(candidate)
    if not candidate or candidate[0] in "{[":
        return None
    if answer_type == ANSWER_TYPE_CLOSE:
        tokens = tokenize_text(candidate)
        if tokens and tokens[0] in {"yes", "no"}:
            return tokens[0]
    return candidate


def extract_prediction_answer(answer_text: Any, *, answer_type: Optional[str] = None) -> PredictionExtraction:
    text = str(answer_text or "")
    strict_payload = parse_prediction_json_strict(text)
    strict_answer = parse_answer_payload(strict_payload)
    if strict_answer is not None:
        return PredictionExtraction(
            answer=strict_answer,
            payload=strict_payload,
            prediction_format=PREDICTION_FORMAT_STRICT_JSON,
            strict_parse_success=True,
            json_object_parsed=True,
        )

    soft_payload = parse_prediction_json(text)
    soft_answer = parse_answer_payload(soft_payload)
    if soft_answer is not None:
        return PredictionExtraction(
            answer=str(soft_answer),
            payload=soft_payload,
            prediction_format=PREDICTION_FORMAT_EMBEDDED_JSON,
            strict_parse_success=False,
            json_object_parsed=True,
        )

    plain_answer = clean_plain_text_answer(text, answer_type=answer_type)
    if plain_answer is not None:
        return PredictionExtraction(
            answer=plain_answer,
            payload=None,
            prediction_format=PREDICTION_FORMAT_PLAIN_TEXT,
            strict_parse_success=False,
            json_object_parsed=False,
        )
    return PredictionExtraction(
        answer=None,
        payload=soft_payload,
        prediction_format=PREDICTION_FORMAT_NONE,
        strict_parse_success=False,
        json_object_parsed=isinstance(soft_payload, dict),
    )


def make_prompt(question: str, *, prompt_style: str = PROMPT_STYLE_LEGACY_JSON) -> str:
    question_text = str(question or "").strip()
    if prompt_style == PROMPT_STYLE_RAW_QUESTION:
        return question_text
    if prompt_style != PROMPT_STYLE_LEGACY_JSON:
        raise ValueError(f"Unsupported prompt style: {prompt_style}")
    return QUESTION_TEMPLATE.format(question=question_text)
