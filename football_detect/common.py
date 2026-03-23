from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_ROOT = Path(__file__).resolve().parent

NOTE_BUCKETS: tuple[str, str, str] = ("close", "mid", "far")
DEFAULT_DATASET_NAME = "maxs-m87/football_detect_no_split"
DEFAULT_DATASET_REVISION = "6544f3946353a780683f2243b2a648c72ea5de17"
BALL_HOLDER_CLASS = "ball holder"
AREA_OF_FOCUS_CLASS = "area of focus"
OFFENSIVE_LINE_CLASS = "offensive line"
DEFENSIVE_LINE_CLASS = "defensive line"
PLAYERS_ON_FIELD_CLASS = "players on the field"
TACKLE_CLASS = "tackle"
LEGACY_LINE_OF_SCRIMMAGE_CLASS = "line of scrimmage"
LINE_INTERACTION_CLASS = "offensive line / defensive line"
DEFAULT_CLASS_PROMPTS: dict[str, str] = {
    BALL_HOLDER_CLASS: "ball carrier",
    AREA_OF_FOCUS_CLASS: "main action area",
    LINE_INTERACTION_CLASS: "offensive and defensive lines engaged after the snap",
    PLAYERS_ON_FIELD_CLASS: "outline of all players on the field",
}


@dataclass(frozen=True)
class NormalizedBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float


@dataclass(frozen=True)
class LabeledBox:
    class_names: tuple[str, ...]
    box: NormalizedBox


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


def round_half_up(value: float) -> int:
    return int(math.floor(float(value) + 0.5))


def parse_note_bucket(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized not in NOTE_BUCKETS:
        allowed = ", ".join(NOTE_BUCKETS)
        raise ValueError(f"Unsupported notes bucket {value!r}. Expected one of: {allowed}")
    return normalized


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


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def normalize_class_name(value: Any) -> str:
    name = str(value or "").strip()
    if not name:
        return ""
    if name == LEGACY_LINE_OF_SCRIMMAGE_CLASS:
        return LINE_INTERACTION_CLASS
    return name


def default_prompt_for_class(class_name: str) -> str:
    normalized = normalize_class_name(class_name)
    if not normalized:
        return ""
    return DEFAULT_CLASS_PROMPTS.get(normalized, normalized)


def build_class_catalog(class_names: Iterable[str]) -> list[dict[str, str]]:
    normalized_names = sorted({normalize_class_name(name) for name in class_names if normalize_class_name(name)})
    return [
        {
            "class_name": class_name,
            "prompt": default_prompt_for_class(class_name),
        }
        for class_name in normalized_names
    ]


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
    has_offensive_line = False
    has_defensive_line = False
    for item in raw:
        if not isinstance(item, dict):
            continue
        class_names = extract_element_class_names(item.get("attributes"))
        out.extend(class_names)
        if OFFENSIVE_LINE_CLASS in class_names:
            has_offensive_line = True
        if DEFENSIVE_LINE_CLASS in class_names:
            has_defensive_line = True
    if has_offensive_line and has_defensive_line:
        out.append(LINE_INTERACTION_CLASS)
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


def _union_boxes(boxes: Sequence[NormalizedBox]) -> Optional[NormalizedBox]:
    if not boxes:
        return None
    x_min = min(box.x_min for box in boxes)
    y_min = min(box.y_min for box in boxes)
    x_max = max(box.x_max for box in boxes)
    y_max = max(box.y_max for box in boxes)
    if x_max <= x_min or y_max <= y_min:
        return None
    return NormalizedBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)


def parse_answer_boxes(answer_boxes: Any, *, width: int, height: int) -> list[LabeledBox]:
    raw = _load_jsonish(answer_boxes)
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return []

    parsed: list[LabeledBox] = []
    line_boxes: list[NormalizedBox] = []
    has_offensive_line = False
    has_defensive_line = False
    for item in raw:
        if not isinstance(item, dict):
            continue
        class_names = extract_element_class_names(item.get("attributes"))
        if not class_names:
            continue
        box = parse_box_item(item, width=width, height=height)
        if box is None:
            continue
        parsed.append(LabeledBox(class_names=tuple(class_names), box=box))
        if OFFENSIVE_LINE_CLASS in class_names:
            has_offensive_line = True
            line_boxes.append(box)
        if DEFENSIVE_LINE_CLASS in class_names:
            has_defensive_line = True
            line_boxes.append(box)
    if has_offensive_line and has_defensive_line:
        merged_box = _union_boxes(line_boxes)
        if merged_box is not None:
            parsed.append(LabeledBox(class_names=(LINE_INTERACTION_CLASS,), box=merged_box))
    return parsed


def parse_box_element_annotations(answer_boxes: Any, *, width: int, height: int) -> list[BoxElementAnnotation]:
    raw = _load_jsonish(answer_boxes)
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return []

    annotations: list[BoxElementAnnotation] = []
    line_boxes: list[NormalizedBox] = []
    has_offensive_line = False
    has_defensive_line = False
    for source_box_index, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        class_names = extract_element_class_names(item.get("attributes"))
        if not class_names:
            continue
        box = parse_box_item(item, width=width, height=height)
        if box is None:
            continue
        if OFFENSIVE_LINE_CLASS in class_names:
            has_offensive_line = True
            line_boxes.append(box)
        if DEFENSIVE_LINE_CLASS in class_names:
            has_defensive_line = True
            line_boxes.append(box)
        for source_element_index, class_name in enumerate(class_names):
            annotations.append(
                BoxElementAnnotation(
                    class_name=class_name,
                    box=box,
                    source_box_index=source_box_index,
                    source_element_index=source_element_index,
                )
            )
    if has_offensive_line and has_defensive_line:
        merged_box = _union_boxes(line_boxes)
        if merged_box is not None:
            annotations.append(
                BoxElementAnnotation(
                    class_name=LINE_INTERACTION_CLASS,
                    box=merged_box,
                    source_box_index=-1,
                    source_element_index=0,
                )
            )
    return annotations


def discover_class_names(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    discovered: set[str] = set()
    for row in rows:
        discovered.update(extract_all_class_names(row.get("answer_boxes")))
    return sorted(discovered)


def largest_remainder_allocation(
    target_total: int,
    quotas: Mapping[str, float],
    *,
    caps: Optional[Mapping[str, int]] = None,
    order: Optional[Sequence[str]] = None,
) -> dict[str, int]:
    if target_total < 0:
        raise ValueError("target_total must be >= 0")

    ordered_keys = list(order or quotas.keys())
    if caps is None:
        caps_map = {key: max(0, round_half_up(max(0.0, float(quotas.get(key, 0.0))))) for key in ordered_keys}
    else:
        caps_map = {key: max(0, int(caps.get(key, 0))) for key in ordered_keys}
    if sum(caps_map.values()) < target_total:
        raise ValueError(
            f"Cannot allocate target_total={target_total}; only {sum(caps_map.values())} slots available."
        )

    floors: dict[str, int] = {}
    remainders: dict[str, float] = {}
    for key in ordered_keys:
        quota = max(0.0, float(quotas.get(key, 0.0)))
        floor_value = min(int(math.floor(quota)), caps_map[key])
        floors[key] = floor_value
        remainders[key] = quota - math.floor(quota)

    remaining = target_total - sum(floors.values())
    if remaining < 0:
        raise ValueError(
            f"Initial floor allocation exceeded target_total: target_total={target_total}, floors={floors}"
        )
    if remaining == 0:
        return floors

    candidates = [
        key
        for key in sorted(
            ordered_keys,
            key=lambda key: (-remainders[key], ordered_keys.index(key), key),
        )
        if floors[key] < caps_map[key]
    ]
    if remaining > len(candidates):
        raise ValueError(
            f"Need {remaining} extra allocations but only {len(candidates)} buckets can receive them."
        )
    for key in candidates[:remaining]:
        floors[key] += 1
    return floors


def allocate_val_counts(
    note_counts: Mapping[str, int],
    *,
    val_fraction: float,
) -> dict[str, int]:
    quotas = {note: int(note_counts.get(note, 0)) * float(val_fraction) for note in NOTE_BUCKETS}
    target_total = round_half_up(sum(quotas.values()))
    return largest_remainder_allocation(target_total, quotas, caps=note_counts, order=NOTE_BUCKETS)


def allocate_post_val_counts(
    val_counts: Mapping[str, int],
    *,
    holdout_count: int,
) -> dict[str, int]:
    total_val = sum(int(val_counts.get(note, 0)) for note in NOTE_BUCKETS)
    if holdout_count < 0:
        raise ValueError("holdout_count must be >= 0")
    if holdout_count > total_val:
        raise ValueError(
            f"holdout_count={holdout_count} exceeds available val pool size {total_val}."
        )
    if holdout_count == 0:
        return {note: 0 for note in NOTE_BUCKETS}
    if total_val == 0:
        raise ValueError("Cannot allocate post_val rows when val pool is empty.")
    quotas = {note: int(val_counts.get(note, 0)) * (float(holdout_count) / float(total_val)) for note in NOTE_BUCKETS}
    return largest_remainder_allocation(holdout_count, quotas, caps=val_counts, order=NOTE_BUCKETS)


def build_split_stats(split_rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    note_counts_by_split: dict[str, dict[str, int]] = {}
    raw_class_counts_by_split: dict[str, Counter[str]] = {}
    split_sizes: dict[str, int] = {}
    all_class_names: set[str] = set()

    for split_name, rows in split_rows.items():
        split_sizes[split_name] = len(rows)
        note_counter: Counter[str] = Counter()
        class_counter: Counter[str] = Counter()
        for row in rows:
            note = parse_note_bucket(row.get("notes"))
            note_counter[note] += 1
            class_counter.update(extract_all_class_names(row.get("answer_boxes")))
        note_counts_by_split[split_name] = {note: int(note_counter.get(note, 0)) for note in NOTE_BUCKETS}
        raw_class_counts_by_split[split_name] = class_counter
        all_class_names.update(class_counter.keys())

    class_catalog = sorted(all_class_names)
    class_counts_by_split = {
        split_name: {class_name: int(counter.get(class_name, 0)) for class_name in class_catalog}
        for split_name, counter in raw_class_counts_by_split.items()
    }

    return {
        "split_sizes": split_sizes,
        "note_buckets": list(NOTE_BUCKETS),
        "class_catalog": class_catalog,
        "note_bucket_counts": note_counts_by_split,
        "class_counts": class_counts_by_split,
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
