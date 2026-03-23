from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_ROOT = Path(__file__).resolve().parent

DEFAULT_ROBOFLOW_WORKSPACE = "roboflow-100"
DEFAULT_ROBOFLOW_PROJECT = "bone-fracture-7fylg"
DEFAULT_ROBOFLOW_VERSION = 2
DEFAULT_DETECT_HF_DATASET_NAME = "maxs-m87/bone_fracture_detect_v1"
DEFAULT_POINT_HF_DATASET_NAME = "maxs-m87/bone_fracture_point_v1"
DEFAULT_HF_DATASET_NAME = DEFAULT_DETECT_HF_DATASET_NAME
DEFAULT_DETECT_WANDB_PROJECT = "moondream-bone-fracture-detect-rl"
DEFAULT_POINT_WANDB_PROJECT = "moondream-bone-fracture-point-rl"
DEFAULT_STAGING_API_BASE = "https://api-staging.moondream.ai/v1"
DEFAULT_POINT_CLASS_NAME = "bone fracture or abnormality"
DEFAULT_POINT_CLASS_UID = "bone_fracture:bone_fracture_or_abnormality"
DEFAULT_SKILL = "point"
DEFAULT_POINT_PROMPT_STYLE = "class_name"
DEFAULT_REWARD_METRIC = "f1"

DEFAULT_CLASS_PROMPTS: dict[str, str] = {
    DEFAULT_POINT_CLASS_NAME: DEFAULT_POINT_CLASS_NAME,
    "fracture": "fracture line",
    "line fracture": "fracture line",
    "line": "reference line along the bone",
    "angle": "bone angle marker",
    "messed_up_angle": "abnormal bone angulation",
    "messed up angle": "abnormal bone angulation",
    "messed-up-angle": "abnormal bone angulation",
}


@dataclass(frozen=True)
class NormalizedBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float


@dataclass(frozen=True)
class BoxElementAnnotation:
    class_name: str
    box: NormalizedBox
    source_box_index: int
    source_element_index: int


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


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def normalize_class_name(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return " ".join(text.split())


def default_prompt_for_class(class_name: str) -> str:
    normalized = normalize_class_name(class_name)
    if not normalized:
        return ""
    return DEFAULT_CLASS_PROMPTS.get(normalized, normalized.replace("_", " "))


def build_class_catalog(class_names: Iterable[str]) -> list[dict[str, str]]:
    normalized_names = sorted({normalize_class_name(name) for name in class_names if normalize_class_name(name)})
    return [
        {
            "class_name": class_name,
            "prompt": default_prompt_for_class(class_name),
        }
        for class_name in normalized_names
    ]


def class_uid_for_name(class_name: str) -> str:
    normalized = normalize_class_name(class_name)
    if normalized == DEFAULT_POINT_CLASS_NAME:
        return DEFAULT_POINT_CLASS_UID
    safe = normalized.replace("/", "_").replace(" ", "_")
    return f"bone_fracture:{safe}"


def build_point_class_catalog(class_names: Iterable[str]) -> list[dict[str, str]]:
    normalized_names = sorted({normalize_class_name(name) for name in class_names if normalize_class_name(name)})
    return [
        {
            "class_uid": class_uid_for_name(class_name),
            "class_name": class_name,
            "prompt": default_prompt_for_class(class_name),
        }
        for class_name in normalized_names
    ]


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_xy(value: float, *, size: int) -> float:
    coord = float(value)
    if abs(coord) > 1.5 and size > 0:
        coord /= float(size)
    return clamp(coord)


def _load_jsonish(value: Any) -> Any:
    raw = value
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        raw = json.loads(text)
    return raw


def extract_element_class_names(attributes: Any) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for attr in _as_list(attributes):
        if not isinstance(attr, dict):
            continue
        key = str(attr.get("key") or "").strip().lower()
        if key != "element":
            continue
        raw_value = attr.get("value")
        for item in _as_list(raw_value):
            name = normalize_class_name(item)
            if not name or name in seen:
                continue
            seen.add(name)
            out.append(name)
    return out


def extract_all_class_names(answer_boxes: Any) -> list[str]:
    raw = _load_jsonish(answer_boxes)
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        for class_name in extract_element_class_names(item.get("attributes")):
            if class_name not in seen:
                seen.add(class_name)
                out.append(class_name)
    return out


def _box_from_xyxy(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    *,
    width: int,
    height: int,
) -> Optional[NormalizedBox]:
    x0 = _normalize_xy(x_min, size=width)
    y0 = _normalize_xy(y_min, size=height)
    x1 = _normalize_xy(x_max, size=width)
    y1 = _normalize_xy(y_max, size=height)
    if x1 <= x0 or y1 <= y0:
        return None
    return NormalizedBox(x_min=x0, y_min=y0, x_max=x1, y_max=y1)


def parse_box_item(item: Any, *, width: int, height: int) -> Optional[NormalizedBox]:
    if not isinstance(item, dict):
        return None

    x_min = _coerce_float(item.get("x_min", item.get("xmin")))
    y_min = _coerce_float(item.get("y_min", item.get("ymin")))
    x_max = _coerce_float(item.get("x_max", item.get("xmax")))
    y_max = _coerce_float(item.get("y_max", item.get("ymax")))
    if None not in (x_min, y_min, x_max, y_max):
        return _box_from_xyxy(
            float(x_min),
            float(y_min),
            float(x_max),
            float(y_max),
            width=width,
            height=height,
        )

    nested = item.get("box")
    if isinstance(nested, dict):
        parsed_nested = parse_box_item(nested, width=width, height=height)
        if parsed_nested is not None:
            return parsed_nested

    x_center = _coerce_float(item.get("x_center", item.get("cx", item.get("x"))))
    y_center = _coerce_float(item.get("y_center", item.get("cy", item.get("y"))))
    box_w = _coerce_float(item.get("width", item.get("w")))
    box_h = _coerce_float(item.get("height", item.get("h")))
    if None in (x_center, y_center, box_w, box_h):
        return None

    x_center_n = _normalize_xy(float(x_center), size=width)
    y_center_n = _normalize_xy(float(y_center), size=height)
    width_n = float(box_w)
    height_n = float(box_h)
    if abs(width_n) > 1.5 and width > 0:
        width_n /= float(width)
    if abs(height_n) > 1.5 and height > 0:
        height_n /= float(height)
    width_n = clamp(width_n)
    height_n = clamp(height_n)
    return _box_from_xyxy(
        x_center_n - (width_n / 2.0),
        y_center_n - (height_n / 2.0),
        x_center_n + (width_n / 2.0),
        y_center_n + (height_n / 2.0),
        width=1,
        height=1,
    )


def parse_box_element_annotations(answer_boxes: Any, *, width: int, height: int) -> list[BoxElementAnnotation]:
    raw = _load_jsonish(answer_boxes)
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return []

    annotations: list[BoxElementAnnotation] = []
    for source_box_index, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        class_names = extract_element_class_names(item.get("attributes"))
        if not class_names:
            continue
        box = parse_box_item(item, width=width, height=height)
        if box is None:
            continue
        for source_element_index, class_name in enumerate(class_names):
            annotations.append(
                BoxElementAnnotation(
                    class_name=class_name,
                    box=box,
                    source_box_index=source_box_index,
                    source_element_index=source_element_index,
                )
            )
    return annotations


def discover_class_names(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    discovered: set[str] = set()
    for row in rows:
        discovered.update(extract_all_class_names(row.get("answer_boxes")))
    return sorted(discovered)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
