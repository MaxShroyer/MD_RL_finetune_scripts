from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_ROOT = Path(__file__).resolve().parent

DEFAULT_DATASET_NAME = "LouisChen15/ConstructionSite"
DEFAULT_STAGING_API_BASE = "https://api-staging.moondream.ai/v1"

DEFAULT_DETECT_OUTPUT_DIR = MODULE_ROOT / "outputs" / "construction_site_detect_v1"
DEFAULT_CAPTION_OUTPUT_DIR = MODULE_ROOT / "outputs" / "construction_site_query_caption_v1"
DEFAULT_RULE_VQA_OUTPUT_DIR = MODULE_ROOT / "outputs" / "construction_site_query_rule_vqa_v1"
DEFAULT_SHARED_IMAGE_DIR = MODULE_ROOT / "outputs" / "construction_site_query_images"

SOURCE_TRAIN_SPLIT = "train"
SOURCE_TEST_SPLIT = "test"
LOCAL_VAL_SPLIT = "validation"

CAPTION_TASK_TYPE = "caption_dense"
RULE_VQA_TASK_TYPE = "rule_vqa"

CAPTION_QUESTION = (
    "Describe this construction site image in detail. Return JSON only with the schema "
    '{"caption":"..."}'
)

RULE_VQA_QUESTION = "\n".join(
    [
        "Inspect this construction site image for safety violations.",
        "Consider these rules:",
        "1. Workers on foot must use required PPE such as hard hats and proper protective clothing.",
        "2. Workers at height without edge protection must wear a safety harness.",
        "3. Open excavation edges and similar drop hazards need guardrails or warning barriers.",
        "4. Workers must stay out of excavator blind spots and operating radius.",
        'Return JSON only with the schema {"violated_rules":[1,4],"reasons":{"1":"...","4":"..."}}.',
        "Use an empty list and empty object when no rules are violated.",
    ]
)

RULE_ID_TO_TEXT: dict[int, str] = {
    1: "missing required PPE",
    2: "working at height without safety harness",
    3: "missing excavation edge protection or warning barrier",
    4: "worker inside excavator blind spot or operating radius",
}

DETECT_CLASS_CATALOG: list[dict[str, str]] = [
    {"class_uid": "construction_site:excavator", "class_name": "excavator", "prompt": "excavator"},
    {
        "class_uid": "construction_site:rebar",
        "class_name": "rebar",
        "prompt": "rebar or reinforcing steel bars",
    },
    {
        "class_uid": "construction_site:worker_with_white_hard_hat",
        "class_name": "worker_with_white_hard_hat",
        "prompt": "worker wearing a white hard hat",
    },
    {
        "class_uid": "construction_site:rule_1_violation",
        "class_name": "rule_1_violation",
        "prompt": "construction worker missing required PPE",
    },
    {
        "class_uid": "construction_site:rule_2_violation",
        "class_name": "rule_2_violation",
        "prompt": "worker at height without safety harness",
    },
    {
        "class_uid": "construction_site:rule_3_violation",
        "class_name": "rule_3_violation",
        "prompt": "open excavation edge without guardrail or warning barrier",
    },
    {
        "class_uid": "construction_site:rule_4_violation",
        "class_name": "rule_4_violation",
        "prompt": "worker inside excavator blind spot or operating radius",
    },
]

DETECT_PROMPT_OVERRIDES: dict[str, str] = {
    entry["class_name"]: entry["prompt"] for entry in DETECT_CLASS_CATALOG
}
CLASS_UID_BY_NAME: dict[str, str] = {
    entry["class_name"]: entry["class_uid"] for entry in DETECT_CLASS_CATALOG
}

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


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


def load_json_config(
    config_path: Path,
    *,
    default_path: Optional[Path] = None,
) -> dict[str, Any]:
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

        action = store_actions[0]
        cli_args.append(option_for_action(action))
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


def set_f1(reference: Iterable[Any], prediction: Iterable[Any]) -> float:
    ref = {str(item) for item in reference}
    pred = {str(item) for item in prediction}
    if not ref and not pred:
        return 1.0
    if not ref or not pred:
        return 0.0
    tp = len(ref & pred)
    precision = tp / max(1, len(pred))
    recall = tp / max(1, len(ref))
    if precision + recall <= 0.0:
        return 0.0
    return (2.0 * precision * recall) / (precision + recall)


def _coerce_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_box(raw_box: Any) -> Optional[dict[str, float]]:
    if not isinstance(raw_box, (list, tuple)) or len(raw_box) != 4:
        return None
    coords = [_coerce_float(value) for value in raw_box]
    if any(value is None for value in coords):
        return None
    x_min, y_min, x_max, y_max = (float(value) for value in coords if value is not None)
    x_min = clamp(x_min)
    y_min = clamp(y_min)
    x_max = clamp(x_max)
    y_max = clamp(y_max)
    if x_max <= x_min or y_max <= y_min:
        return None
    return {
        "x_min": x_min,
        "y_min": y_min,
        "x_max": x_max,
        "y_max": y_max,
    }


def normalize_box_list(raw_boxes: Any) -> list[dict[str, float]]:
    if raw_boxes is None:
        return []
    if isinstance(raw_boxes, str):
        text = raw_boxes.strip()
        if not text:
            return []
        try:
            raw_boxes = json.loads(text)
        except json.JSONDecodeError:
            return []
    boxes: list[dict[str, float]] = []
    for raw_box in raw_boxes if isinstance(raw_boxes, list) else []:
        normalized = normalize_box(raw_box)
        if normalized is not None:
            boxes.append(normalized)
    return boxes


def serialize_answer_boxes(boxes: Iterable[Mapping[str, Any]]) -> str:
    payload = [
        {
            "x_min": float(box["x_min"]),
            "y_min": float(box["y_min"]),
            "x_max": float(box["x_max"]),
            "y_max": float(box["y_max"]),
            "class_uid": str(box["class_uid"]),
            "class_name": str(box["class_name"]),
            "source_class_name": str(box.get("source_class_name", box["class_name"])),
        }
        for box in boxes
    ]
    return json.dumps(payload, separators=(",", ":"))


def build_detect_boxes(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    boxes: list[dict[str, Any]] = []
    class_box_sources = (
        ("excavator", row.get("excavator")),
        ("rebar", row.get("rebar")),
        ("worker_with_white_hard_hat", row.get("worker_with_white_hard_hat")),
    )
    for class_name, raw_boxes in class_box_sources:
        for box in normalize_box_list(raw_boxes):
            boxes.append(
                {
                    **box,
                    "class_uid": CLASS_UID_BY_NAME[class_name],
                    "class_name": class_name,
                    "source_class_name": class_name,
                }
            )

    for rule_id in sorted(RULE_ID_TO_TEXT):
        payload = row.get(f"rule_{rule_id}_violation")
        if not isinstance(payload, dict):
            continue
        for box in normalize_box_list(payload.get("bounding_box")):
            class_name = f"rule_{rule_id}_violation"
            boxes.append(
                {
                    **box,
                    "class_uid": CLASS_UID_BY_NAME[class_name],
                    "class_name": class_name,
                    "source_class_name": class_name,
                }
            )
    return boxes


def extract_caption_attribute_tags(row: Mapping[str, Any]) -> list[str]:
    out: list[str] = []
    for key in ("illumination", "camera_distance", "view", "quality_of_info"):
        value = " ".join(str(row.get(key) or "").strip().lower().split())
        if value:
            out.append(value)
    return out


def extract_caption_object_tags(row: Mapping[str, Any]) -> list[str]:
    tags: list[str] = []
    if normalize_box_list(row.get("excavator")):
        tags.append("excavator")
    if normalize_box_list(row.get("rebar")):
        tags.append("rebar")
    if normalize_box_list(row.get("worker_with_white_hard_hat")):
        tags.extend(["worker", "white hard hat"])
    violated_rules = extract_rule_reasons(row)
    if violated_rules:
        tags.append("safety violation")
    return sorted(set(tags))


def extract_rule_reasons(row: Mapping[str, Any]) -> dict[int, str]:
    reasons: dict[int, str] = {}
    for rule_id in sorted(RULE_ID_TO_TEXT):
        payload = row.get(f"rule_{rule_id}_violation")
        if not isinstance(payload, dict):
            continue
        reason = str(payload.get("reason") or "").strip()
        if reason:
            reasons[rule_id] = reason
    return reasons


def relative_path(from_dir: Path, to_path: Path) -> str:
    return os.path.relpath(str(to_path), start=str(from_dir))
