#!/usr/bin/env python3
"""Build a mixed-skill local DisasterM3 dataset with composite temporal images."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from disaster_m3 import common  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = common.repo_relative("configs", "build_disaster_m3_dataset_default.json")

QUESTION_KEYS = ("question", "prompt", "prompts", "instruction", "query", "caption_prompt")
ANSWER_KEYS = (
    "ground_truth_option",
    "training_answer",
    "answer",
    "answers",
    "response",
    "responses",
    "target",
    "label",
    "labels",
    "gt_answer",
    "ground_truth",
    "final_answer",
    "assistant",
    "assistant_response",
)
PRE_IMAGE_KEYS = ("pre_image_path", "pre_path", "pre_image", "before_image_path", "image_path_pre")
POST_IMAGE_KEYS = ("post_image_path", "post_path", "post_image", "after_image_path", "image_path_post")
SINGLE_IMAGE_KEYS = ("image_path", "image", "rgb_image_path", "img_path")
BOX_KEYS = (
    "answer_boxes",
    "answer_boxes_json",
    "boxes",
    "bbox",
    "bboxes",
    "bounding_box",
    "bounding_boxes",
    "grounding_boxes",
    "ground_truth_boxes",
    "annotations",
)
POINT_KEYS = ("answer_points", "answer_points_json", "points", "point", "centers", "centroids")
METADATA_KEYS = {
    *QUESTION_KEYS,
    *ANSWER_KEYS,
    *PRE_IMAGE_KEYS,
    *POST_IMAGE_KEYS,
    *SINGLE_IMAGE_KEYS,
    *BOX_KEYS,
    *POINT_KEYS,
    "subset",
    "subset_name",
    "task",
    "task_name",
    "task_type",
    "objects",
    "messages",
    "conversations",
    "options",
    "options_list",
    "option_str",
    "options_str",
    "candidate_options",
    "post_image_type",
    "image_type",
    "cls_description",
    "ground_truth_option",
    "training_answer",
    "split",
    "image_id",
    "id",
}


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    raw_argv = list(argv) if argv is not None else list(os.sys.argv[1:])
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    pre_args, _ = pre_parser.parse_known_args(raw_argv)
    config_path = common.resolve_config_path(pre_args.config, script_dir=SCRIPT_DIR)
    config = common.load_json_config(config_path, default_path=DEFAULT_CONFIG_PATH)

    parser = argparse.ArgumentParser(description="Build a local mixed-skill DisasterM3 dataset.")
    parser.add_argument("--config", default=str(config_path))
    parser.add_argument("--env-file", default=str(common.repo_relative(".env.staging")))
    parser.add_argument("--hf-repo-id", default=common.DEFAULT_HF_REPO_ID)
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or "")
    parser.add_argument("--dataset-root", default=str(common.DEFAULT_DATASET_ROOT))
    parser.add_argument("--raw-dir", default="")
    parser.add_argument("--output-dir", default=str(common.DEFAULT_OUTPUT_DIR))
    parser.add_argument("--panel-max-side", type=int, default=common.DEFAULT_PANEL_MAX_SIDE)
    parser.add_argument("--divider-px", type=int, default=common.DEFAULT_DIVIDER_PX)
    parser.add_argument("--jpeg-quality", type=int, default=common.DEFAULT_JPEG_QUALITY)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--subset-seed", type=int, default=42)
    parser.add_argument("--download", dest="download", action="store_true")
    parser.add_argument("--no-download", dest="download", action="store_false")
    parser.set_defaults(download=True)
    parser.add_argument("--force-download", action="store_true")

    option_to_dest: dict[str, str] = {}
    for action in parser._actions:
        if not action.option_strings:
            continue
        for opt in action.option_strings:
            option_to_dest[opt] = action.dest
    overridden = {option_to_dest[arg] for arg in raw_argv if arg in option_to_dest}
    config_cli_args = common.config_to_cli_args(parser, config, config_path=config_path, overridden_dests=overridden)
    args = parser.parse_args(config_cli_args + raw_argv)
    args.config = str(common.resolve_config_path(args.config, script_dir=SCRIPT_DIR))
    args.dataset_root = common.resolve_path(args.dataset_root, repo_root=REPO_ROOT, module_root=SCRIPT_DIR)
    args.raw_dir = (
        common.resolve_path(args.raw_dir, repo_root=REPO_ROOT, module_root=SCRIPT_DIR)
        if str(args.raw_dir or "").strip()
        else Path(args.dataset_root).expanduser().resolve() / "raw"
    )
    args.output_dir = common.resolve_path(args.output_dir, repo_root=REPO_ROOT, module_root=SCRIPT_DIR)
    args.env_file = str(common.resolve_path(args.env_file, repo_root=REPO_ROOT, module_root=SCRIPT_DIR))
    return args


def _slugify(text: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", str(text or "").strip().lower()).strip("_")
    return normalized or "item"


def _candidate_json_files(root: Path) -> list[Path]:
    candidates: list[Path] = []
    for path in sorted(root.rglob("*.json")):
        if not path.is_file():
            continue
        if path.name.startswith("."):
            continue
        candidates.append(path)
    return candidates


def _load_rows_from_json_file(path: Path) -> list[dict[str, Any]]:
    payload = common.load_json_payload(path)
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("data", "samples", "rows", "items", "records"):
            value = payload.get(key)
            if isinstance(value, list):
                return [dict(item) for item in value if isinstance(item, dict)]
        split_rows: list[dict[str, Any]] = []
        for split_key in ("train", "val", "valid", "validation", "test"):
            value = payload.get(split_key)
            if not isinstance(value, list):
                continue
            normalized_split = "val" if split_key in {"valid", "validation"} else split_key
            for item in value:
                if not isinstance(item, dict):
                    continue
                row = dict(item)
                row.setdefault("split", normalized_split)
                split_rows.append(row)
        if split_rows:
            return split_rows
    return []


def _looks_like_task_file(path: Path, rows: list[dict[str, Any]]) -> bool:
    if not rows:
        return False
    example = rows[0]
    if not isinstance(example, dict):
        return False
    keys = {common.normalize_text(key) for key in example.keys()}
    if any(key in keys for key in ("messages", "conversations")):
        return True
    if any(key in keys for key in QUESTION_KEYS + ANSWER_KEYS):
        return True
    if any(key in keys for key in PRE_IMAGE_KEYS + POST_IMAGE_KEYS + SINGLE_IMAGE_KEYS):
        return True
    return "benchmark_release" in path.name.lower()


def _infer_task_name(path: Path, row: Mapping[str, Any]) -> str:
    for key in ("subset", "subset_name", "task_name", "task_type", "task"):
        value = str(row.get(key) or "").strip()
        if value:
            return _slugify(value)
    stem = _slugify(path.stem)
    if stem in {"train", "val", "valid", "validation", "test"}:
        parent = path.parent.name
        if parent:
            return _slugify(parent)
    return stem


def _discover_split(path: Path, row: Mapping[str, Any], *, source_default: str) -> str:
    raw_split = str(row.get("split") or "").strip().lower()
    if raw_split in {"train", "val", "valid", "validation", "test"}:
        return "val" if raw_split in {"valid", "validation"} else raw_split
    inferred = common.infer_split_from_path(path, default_split=source_default)
    if inferred:
        return "val" if inferred in {"valid", "validation"} else inferred
    if source_default:
        return source_default
    raise ValueError(f"Unable to infer split for source file: {path}")


def _extract_text_by_keys(row: Mapping[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list):
            parts = [str(item).strip() for item in value if str(item or "").strip()]
            if parts:
                return " ".join(parts)
        if isinstance(value, dict):
            for nested_key in ("text", "prompt", "question", "content"):
                nested_value = value.get(nested_key)
                if isinstance(nested_value, str) and nested_value.strip():
                    return nested_value.strip()
    return ""


def _extract_question(row: Mapping[str, Any]) -> str:
    question = _extract_text_by_keys(row, QUESTION_KEYS)
    if question:
        return question
    for key in ("messages", "conversations"):
        value = row.get(key)
        question = common.question_from_messages(value)
        if question:
            return question
    return ""


def _extract_answer_text(row: Mapping[str, Any]) -> str:
    preferred_option = row.get("ground_truth_option")
    if isinstance(preferred_option, str) and preferred_option.strip():
        return preferred_option.strip()
    preferred_training = row.get("training_answer")
    if isinstance(preferred_training, str) and preferred_training.strip():
        return preferred_training.strip()
    for key in ANSWER_KEYS:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
        if isinstance(value, list) and value:
            return json.dumps(value, ensure_ascii=False)
        if isinstance(value, dict) and value:
            return json.dumps(value, ensure_ascii=False)
    for key in ("messages", "conversations"):
        answer = common.answer_from_messages(row.get(key))
        if answer:
            return answer
    return ""


def _resolve_source_image_path(raw_value: str, *, image_index: common.ImageIndex) -> Optional[Path]:
    if not raw_value:
        return None
    if not common.is_rgb_reference(raw_value):
        return None
    return image_index.resolve(raw_value)


def _extract_image_refs(row: Mapping[str, Any], *, image_index: common.ImageIndex) -> tuple[Optional[common.ImageReference], Optional[common.ImageReference], Optional[common.ImageReference]]:
    pre_ref = None
    post_ref = None
    single_ref = None
    for key in PRE_IMAGE_KEYS:
        resolved = _resolve_source_image_path(str(row.get(key) or ""), image_index=image_index)
        if resolved is not None:
            pre_ref = common.ImageReference(image_path=resolved, source_side="pre")
            break
    for key in POST_IMAGE_KEYS:
        resolved = _resolve_source_image_path(str(row.get(key) or ""), image_index=image_index)
        if resolved is not None:
            post_ref = common.ImageReference(image_path=resolved, source_side="post")
            break
    if pre_ref is None and post_ref is None:
        for key in SINGLE_IMAGE_KEYS:
            resolved = _resolve_source_image_path(str(row.get(key) or ""), image_index=image_index)
            if resolved is not None:
                side = "pre"
                key_text = common.normalize_text(key)
                if "post" in key_text or "after" in key_text:
                    side = "post"
                single_ref = common.ImageReference(image_path=resolved, source_side=side)
                break
    return pre_ref, post_ref, single_ref


def _image_reference_values(row: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    for key in (*PRE_IMAGE_KEYS, *POST_IMAGE_KEYS, *SINGLE_IMAGE_KEYS):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            values.append(value.strip())
    return values


def _default_annotation_side(*, pre_ref: Optional[common.ImageReference], post_ref: Optional[common.ImageReference], single_ref: Optional[common.ImageReference]) -> str:
    if post_ref is not None:
        return "post"
    if pre_ref is not None:
        return "pre"
    if single_ref is not None:
        return single_ref.source_side
    return "post"


def _image_sizes(
    *,
    pre_image: Optional[Image.Image],
    post_image: Optional[Image.Image],
    single_image: Optional[Image.Image],
    single_side: str,
) -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    if pre_image is not None:
        out["pre"] = pre_image.size
    if post_image is not None:
        out["post"] = post_image.size
    if single_image is not None:
        out[single_side] = single_image.size
    return out


def _normalize_box_values(values: Sequence[float], *, width: int, height: int) -> Optional[tuple[float, float, float, float]]:
    if len(values) != 4 or width <= 0 or height <= 0:
        return None
    x0, y0, x1, y1 = [float(value) for value in values]
    if max(abs(x0), abs(y0), abs(x1), abs(y1)) > 1.5:
        x0 /= float(width)
        y0 /= float(height)
        x1 /= float(width)
        y1 /= float(height)
    x_min = common.clamp(min(x0, x1))
    x_max = common.clamp(max(x0, x1))
    y_min = common.clamp(min(y0, y1))
    y_max = common.clamp(max(y0, y1))
    if x_max <= x_min or y_max <= y_min:
        return None
    return x_min, y_min, x_max, y_max


def _boxes_from_value(
    value: Any,
    *,
    class_name: str,
    source_side: str,
    side_sizes: Mapping[str, tuple[int, int]],
) -> list[common.LabeledBox]:
    width, height = side_sizes.get(source_side, side_sizes.get("post", side_sizes.get("pre", (0, 0))))
    out: list[common.LabeledBox] = []
    if isinstance(value, dict):
        if {"x_min", "y_min", "x_max", "y_max"} <= set(value.keys()):
            coords = _normalize_box_values(
                [
                    float(value["x_min"]),
                    float(value["y_min"]),
                    float(value["x_max"]),
                    float(value["y_max"]),
                ],
                width=width,
                height=height,
            )
            if coords is not None:
                out.append(
                    common.LabeledBox(
                        x_min=coords[0],
                        y_min=coords[1],
                        x_max=coords[2],
                        y_max=coords[3],
                        class_name=class_name,
                        source_side=source_side,
                    )
                )
            return out
        if {"x", "y", "width", "height"} <= set(value.keys()):
            coords = _normalize_box_values(
                [
                    float(value["x"]),
                    float(value["y"]),
                    float(value["x"]) + float(value["width"]),
                    float(value["y"]) + float(value["height"]),
                ],
                width=width,
                height=height,
            )
            if coords is not None:
                out.append(
                    common.LabeledBox(
                        x_min=coords[0],
                        y_min=coords[1],
                        x_max=coords[2],
                        y_max=coords[3],
                        class_name=class_name,
                        source_side=source_side,
                    )
                )
            return out
        for nested_key in ("bbox", "bboxes", "box", "boxes", "bounding_box", "bounding_boxes"):
            if nested_key in value:
                nested_class = class_name or common.humanize_label(str(value.get("class_name") or value.get("label") or ""))
                out.extend(
                    _boxes_from_value(
                        value[nested_key],
                        class_name=nested_class,
                        source_side=source_side,
                        side_sizes=side_sizes,
                    )
                )
        if out:
            return out
        for nested_key, nested_value in value.items():
            key_text = common.normalize_text(nested_key)
            nested_side = source_side
            if "pre" in key_text or "before" in key_text:
                nested_side = "pre"
            elif "post" in key_text or "after" in key_text:
                nested_side = "post"
            out.extend(
                _boxes_from_value(
                    nested_value,
                    class_name=class_name or common.humanize_label(str(nested_key)),
                    source_side=nested_side,
                    side_sizes=side_sizes,
                )
            )
        return out
    if isinstance(value, list):
        if len(value) == 4 and all(isinstance(item, (int, float)) for item in value):
            coords = _normalize_box_values([float(item) for item in value], width=width, height=height)
            if coords is not None:
                return [
                    common.LabeledBox(
                        x_min=coords[0],
                        y_min=coords[1],
                        x_max=coords[2],
                        y_max=coords[3],
                        class_name=class_name,
                        source_side=source_side,
                    )
                ]
            return []
        out = []
        for item in value:
            out.extend(
                _boxes_from_value(
                    item,
                    class_name=class_name,
                    source_side=source_side,
                    side_sizes=side_sizes,
                )
            )
        return out
    return []


def _points_from_value(
    value: Any,
    *,
    class_name: str,
    source_side: str,
    side_sizes: Mapping[str, tuple[int, int]],
) -> list[common.LabeledPoint]:
    width, height = side_sizes.get(source_side, side_sizes.get("post", side_sizes.get("pre", (0, 0))))
    out: list[common.LabeledPoint] = []
    if isinstance(value, dict):
        if {"x", "y"} <= set(value.keys()):
            x = float(value["x"])
            y = float(value["y"])
            if max(abs(x), abs(y)) > 1.5 and width > 0 and height > 0:
                x /= float(width)
                y /= float(height)
            return [common.LabeledPoint(x=common.clamp(x), y=common.clamp(y), class_name=class_name, source_side=source_side)]
        for nested_key in ("points", "point", "centers", "centroids"):
            if nested_key in value:
                out.extend(
                    _points_from_value(
                        value[nested_key],
                        class_name=class_name or common.humanize_label(str(value.get("class_name") or value.get("label") or "")),
                        source_side=source_side,
                        side_sizes=side_sizes,
                    )
                )
        if out:
            return out
        for nested_key, nested_value in value.items():
            out.extend(
                _points_from_value(
                    nested_value,
                    class_name=class_name or common.humanize_label(str(nested_key)),
                    source_side=source_side,
                    side_sizes=side_sizes,
                )
            )
        return out
    if isinstance(value, list):
        if len(value) == 2 and all(isinstance(item, (int, float)) for item in value):
            x = float(value[0])
            y = float(value[1])
            if max(abs(x), abs(y)) > 1.5 and width > 0 and height > 0:
                x /= float(width)
                y /= float(height)
            return [common.LabeledPoint(x=common.clamp(x), y=common.clamp(y), class_name=class_name, source_side=source_side)]
        out = []
        for item in value:
            out.extend(
                _points_from_value(
                    item,
                    class_name=class_name,
                    source_side=source_side,
                    side_sizes=side_sizes,
                )
            )
        return out
    return []


def _extract_boxes(
    row: Mapping[str, Any],
    *,
    side_sizes: Mapping[str, tuple[int, int]],
    default_side: str,
    default_object_name: str,
) -> list[common.LabeledBox]:
    out: list[common.LabeledBox] = []
    for key in BOX_KEYS:
        if key not in row:
            continue
        out.extend(
            _boxes_from_value(
                row[key],
                class_name=default_object_name,
                source_side=default_side,
                side_sizes=side_sizes,
            )
        )
    for key, value in row.items():
        if key in METADATA_KEYS:
            continue
        candidate = _boxes_from_value(
            value,
            class_name=common.humanize_label(str(key)),
            source_side=default_side,
            side_sizes=side_sizes,
        )
        if candidate:
            out.extend(candidate)
    dedup: dict[tuple[Any, ...], common.LabeledBox] = {}
    for item in out:
        dedup[(round(item.x_min, 6), round(item.y_min, 6), round(item.x_max, 6), round(item.y_max, 6), item.class_name, item.source_side)] = item
    return list(dedup.values())


def _extract_points(
    row: Mapping[str, Any],
    *,
    side_sizes: Mapping[str, tuple[int, int]],
    default_side: str,
    default_object_name: str,
) -> list[common.LabeledPoint]:
    out: list[common.LabeledPoint] = []
    for key in POINT_KEYS:
        if key not in row:
            continue
        out.extend(
            _points_from_value(
                row[key],
                class_name=default_object_name,
                source_side=default_side,
                side_sizes=side_sizes,
            )
        )
    dedup: dict[tuple[Any, ...], common.LabeledPoint] = {}
    for item in out:
        dedup[(round(item.x, 6), round(item.y, 6), item.class_name, item.source_side)] = item
    return list(dedup.values())


def _parse_description_answer(answer_text: str) -> dict[str, Any]:
    text = str(answer_text or "").strip()
    if not text:
        return {
            "disaster": "",
            "building": "",
            "road": "",
            "vegetation": "",
            "water_body": "",
            "agriculture": "",
            "conclusion": "",
        }
    sections = {
        "DISASTER": "disaster",
        "BUILDING": "building",
        "ROAD": "road",
        "VEGETATION": "vegetation",
        "WATER_BODY": "water_body",
        "AGRICULTURE": "agriculture",
        "CONCLUSION": "conclusion",
    }
    result = {value: "" for value in sections.values()}
    current_key = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        matched_key = None
        for header, out_key in sections.items():
            prefix = f"{header}:"
            if line.upper().startswith(prefix):
                matched_key = out_key
                result[out_key] = line[len(prefix) :].strip()
                current_key = out_key
                break
        if matched_key is None and current_key is not None:
            result[current_key] = " ".join(piece for piece in [result[current_key], line] if piece)
    if not any(result.values()):
        result["conclusion"] = text
    return result


def _parse_recovery_answer(answer_text: str) -> dict[str, Any]:
    text = str(answer_text or "").strip()
    if not text:
        return {
            "needs_recovery": False,
            "immediate_recovery": "",
            "long_term_recovery": "",
        }
    lower = common.normalize_text(text)
    if "no recovery" in lower or "no discernible impact" in lower or "no significant damage" in lower:
        return {
            "needs_recovery": False,
            "immediate_recovery": "",
            "long_term_recovery": "",
        }
    immediate = ""
    long_term = ""
    for line in text.splitlines():
        stripped = line.strip()
        upper = stripped.upper()
        if upper.startswith("IMMEDIATE_RECOVERY:"):
            immediate = stripped.split(":", 1)[1].strip()
        elif upper.startswith("LONG_TERM_RECOVERY:"):
            long_term = stripped.split(":", 1)[1].strip()
    if not immediate and not long_term:
        immediate = text
    return {
        "needs_recovery": True,
        "immediate_recovery": immediate,
        "long_term_recovery": long_term,
    }


def _normalize_query_target(
    *,
    task_name: str,
    answer_text: str,
    row: Mapping[str, Any],
) -> tuple[dict[str, Any], str, dict[str, str]]:
    options_by_letter = common.parse_options_map(row)
    task_kind = common.detect_query_kind(task_name, {"options_by_letter": options_by_letter})
    if task_kind == "description":
        return _parse_description_answer(answer_text), "description", options_by_letter
    if task_kind == "recovery":
        return _parse_recovery_answer(answer_text), "recovery", options_by_letter
    if task_kind == "count":
        answer_letters = common.parse_mcq_letters(answer_text)
        if answer_letters and options_by_letter:
            option_text = options_by_letter.get(answer_letters[0], "")
            count_value = common.parse_number_from_text(option_text)
        else:
            count_value = common.parse_number_from_text(answer_text)
        if count_value is None:
            raise ValueError(f"Unable to parse counting answer for task={task_name}: {answer_text!r}")
        normalized_count: int | float
        if float(count_value).is_integer():
            normalized_count = int(count_value)
        else:
            normalized_count = round(float(count_value), 6)
        return {"count": normalized_count}, "count", options_by_letter
    if task_name in common.MULTI_ANSWER_TASKS:
        letters = common.parse_mcq_letters(answer_text)
        if not letters:
            raise ValueError(f"Unable to parse multi-answer MCQ labels for task={task_name}: {answer_text!r}")
        return {"answers": letters}, "multi_choice_multi_answer", options_by_letter
    if options_by_letter:
        letters = common.parse_mcq_letters(answer_text)
        if len(letters) > 1:
            return {"answers": letters}, "multi_choice_multi_answer", options_by_letter
        if letters:
            return {"answer": letters[0]}, "multi_choice_single_answer", options_by_letter
    return {"answer": answer_text.strip()}, "free_text", options_by_letter


def _query_prompt(
    *,
    task_kind: str,
    original_question: str,
    options_by_letter: Mapping[str, str],
) -> str:
    temporal_prefix = "The image is a disaster-change composite: left is pre-disaster and right is post-disaster."
    if task_kind == "description":
        return (
            f"{temporal_prefix}\n"
            "Summarize the disaster scene in JSON only with keys "
            '{"disaster","building","road","vegetation","water_body","agriculture","conclusion"}.\n'
            f"Task: {original_question or 'Describe the disaster impacts visible across the paired images.'}"
        )
    if task_kind == "recovery":
        return (
            f"{temporal_prefix}\n"
            'Return JSON only with keys {"needs_recovery","immediate_recovery","long_term_recovery"}. '
            "Keep the recommendations concise and practical.\n"
            f"Task: {original_question or 'Provide restoration advice based on the disaster damage visible in the paired images.'}"
        )
    if task_kind == "count":
        return (
            f"{temporal_prefix}\n"
            'Return JSON only with the format {"count": <number>}.\n'
            f"Question: {original_question}"
        )
    if task_kind == "multi_choice_multi_answer":
        option_lines = "\n".join(f"{key}. {value}" for key, value in sorted(options_by_letter.items()))
        return (
            f"{temporal_prefix}\n"
            'Return JSON only with the format {"answers": ["A","C"]}.\n'
            f"Question: {original_question}\nOptions:\n{option_lines}"
        )
    if task_kind == "multi_choice_single_answer":
        option_lines = "\n".join(f"{key}. {value}" for key, value in sorted(options_by_letter.items()))
        return (
            f"{temporal_prefix}\n"
            'Return JSON only with the format {"answer": "A"}.\n'
            f"Question: {original_question}\nOptions:\n{option_lines}"
        )
    return (
        f"{temporal_prefix}\n"
        'Return JSON only with the format {"answer": "..."}.\n'
        f"Question: {original_question}"
    )


def _object_prompt(*, object_name: str, source_side: str) -> str:
    side_text = "right post-disaster panel" if source_side == "post" else "left pre-disaster panel"
    return f"{object_name} in the {side_text}"


def _save_composite_image(
    *,
    output_dir: Path,
    split: str,
    row_id: str,
    composite: Image.Image,
    jpeg_quality: int,
) -> Path:
    image_path = output_dir / "images" / split / f"{row_id}.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    composite.save(image_path, format="JPEG", quality=max(1, min(100, int(jpeg_quality))))
    return image_path.resolve().relative_to(output_dir.resolve())


def _spatial_skill(task_name: str, *, boxes: Sequence[common.LabeledBox], points: Sequence[common.LabeledPoint]) -> str:
    task_text = common.normalize_text(task_name)
    if points and (any(token in task_text for token in common.POINT_TASK_TOKENS) or not boxes):
        return "point"
    return "detect"


def _group_boxes_by_class(boxes: Sequence[common.LabeledBox], *, fallback_name: str) -> dict[str, list[common.LabeledBox]]:
    grouped: dict[str, list[common.LabeledBox]] = defaultdict(list)
    for item in boxes:
        name = common.humanize_label(item.class_name) or common.humanize_label(fallback_name) or "target"
        grouped[name].append(item)
    return grouped


def _group_points_by_class(points: Sequence[common.LabeledPoint], *, fallback_name: str) -> dict[str, list[common.LabeledPoint]]:
    grouped: dict[str, list[common.LabeledPoint]] = defaultdict(list)
    for item in points:
        name = common.humanize_label(item.class_name) or common.humanize_label(fallback_name) or "target"
        grouped[name].append(item)
    return grouped


def _source_json_rel(path: Path, *, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _source_image_rel(path_str: str, *, root: Path) -> str:
    text = str(path_str or "").strip()
    if not text:
        return ""
    path = Path(text).expanduser()
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve()) if path.exists() else text


def _build_query_manifest_row(
    *,
    row_id: str,
    split: str,
    task_name: str,
    question: str,
    image_path: Path,
    final_answer: dict[str, Any],
    query_kind: str,
    options_by_letter: Mapping[str, str],
    source_pre_image_path: str,
    source_post_image_path: str,
    source_image_path: str,
    source_json_path: str,
    source_row_index: int,
) -> dict[str, Any]:
    metadata = {
        "query_kind": query_kind,
        "options_by_letter": dict(options_by_letter),
        "source_json_path": source_json_path,
        "source_row_index": int(source_row_index),
    }
    return {
        "row_id": row_id,
        "split": split,
        "task_name": task_name,
        "task_family": common.infer_task_family(task_name, skill="query"),
        "skill": "query",
        "image_path": str(image_path).replace("\\", "/"),
        "question": question,
        "object_name": "",
        "final_answer_json": json.dumps(final_answer, ensure_ascii=False, separators=(",", ":")),
        "answer_boxes_json": "",
        "answer_points_json": "",
        "source_pre_image_path": source_pre_image_path,
        "source_post_image_path": source_post_image_path,
        "source_image_path": source_image_path,
        "metadata_json": json.dumps(metadata, ensure_ascii=False, separators=(",", ":")),
    }


def _build_spatial_manifest_row(
    *,
    row_id: str,
    split: str,
    task_name: str,
    skill: str,
    object_name: str,
    image_path: Path,
    boxes: Sequence[common.DetectAnnotation],
    points: Sequence[common.PointAnnotation],
    source_pre_image_path: str,
    source_post_image_path: str,
    source_image_path: str,
    source_json_path: str,
    source_row_index: int,
    annotation_side: str,
) -> dict[str, Any]:
    metadata = {
        "annotation_side": annotation_side,
        "source_json_path": source_json_path,
        "source_row_index": int(source_row_index),
    }
    return {
        "row_id": row_id,
        "split": split,
        "task_name": task_name,
        "task_family": common.infer_task_family(task_name, skill=skill),
        "skill": skill,
        "image_path": str(image_path).replace("\\", "/"),
        "question": "",
        "object_name": object_name,
        "final_answer_json": "",
        "answer_boxes_json": common.serialize_boxes(boxes),
        "answer_points_json": common.serialize_points(points),
        "source_pre_image_path": source_pre_image_path,
        "source_post_image_path": source_post_image_path,
        "source_image_path": source_image_path,
        "metadata_json": json.dumps(metadata, ensure_ascii=False, separators=(",", ":")),
    }


def _adapt_row(
    *,
    row: Mapping[str, Any],
    task_name: str,
    split: str,
    source_path: Path,
    source_row_index: int,
    image_index: common.ImageIndex,
    output_dir: Path,
    raw_root: Path,
    panel_size: int,
    divider_px: int,
    jpeg_quality: int,
    stats: Counter[str],
) -> list[dict[str, Any]]:
    if common.is_segmentation_task(task_name, row):
        stats["skipped_segmentation"] += 1
        return []
    question = _extract_question(row)
    answer_text = _extract_answer_text(row)
    pre_ref, post_ref, single_ref = _extract_image_refs(row, image_index=image_index)
    if pre_ref is None and post_ref is None and single_ref is None:
        raw_image_values = _image_reference_values(row)
        if raw_image_values and any(not common.is_rgb_reference(value) for value in raw_image_values):
            stats["skipped_non_rgb_variant"] += 1
        else:
            stats["skipped_missing_image"] += 1
        return []
    pre_image = None
    post_image = None
    single_image = None
    source_pre_path = ""
    source_post_path = ""
    source_single_path = ""
    if pre_ref is not None:
        source_pre_path = _source_image_rel(str(pre_ref.image_path), root=raw_root)
        with Image.open(pre_ref.image_path) as image:
            pre_image = image.convert("RGB").copy()
    if post_ref is not None:
        source_post_path = _source_image_rel(str(post_ref.image_path), root=raw_root)
        with Image.open(post_ref.image_path) as image:
            post_image = image.convert("RGB").copy()
    if single_ref is not None:
        source_single_path = _source_image_rel(str(single_ref.image_path), root=raw_root)
        with Image.open(single_ref.image_path) as image:
            single_image = image.convert("RGB").copy()
        if single_ref.source_side == "pre":
            pre_image = single_image
            source_pre_path = source_single_path
        else:
            post_image = single_image
            source_post_path = source_single_path
    composite, layout = common.build_composite_image(
        pre_image=pre_image,
        post_image=post_image,
        panel_size=panel_size,
        divider_px=divider_px,
    )
    row_base_id = str(row.get("id") or row.get("image_id") or f"{task_name}_{source_row_index:06d}")
    row_id = _slugify(f"{split}_{task_name}_{row_base_id}_{source_row_index}")
    composite_path = _save_composite_image(
        output_dir=output_dir,
        split=split,
        row_id=row_id,
        composite=composite,
        jpeg_quality=jpeg_quality,
    )
    side_sizes = _image_sizes(
        pre_image=pre_image,
        post_image=post_image,
        single_image=single_image,
        single_side=single_ref.source_side if single_ref is not None else "pre",
    )
    default_side = _default_annotation_side(pre_ref=pre_ref, post_ref=post_ref, single_ref=single_ref)
    default_object_name = common.humanize_label(str(row.get("object_name") or row.get("object") or task_name))
    boxes = _extract_boxes(
        row,
        side_sizes=side_sizes,
        default_side=default_side,
        default_object_name=default_object_name,
    )
    points = _extract_points(
        row,
        side_sizes=side_sizes,
        default_side=default_side,
        default_object_name=default_object_name,
    )
    out: list[dict[str, Any]] = []
    source_json_path = _source_json_rel(source_path, root=source_path.parents[2] if len(source_path.parents) > 2 else source_path.parent)
    if boxes or points:
        skill = _spatial_skill(task_name, boxes=boxes, points=points)
        grouped_boxes = _group_boxes_by_class(boxes, fallback_name=default_object_name) if boxes else {}
        if skill == "point":
            grouped_points = _group_points_by_class(points, fallback_name=default_object_name)
            if not grouped_points and grouped_boxes:
                for object_name, class_boxes in grouped_boxes.items():
                    grouped_points[object_name] = [
                        common.LabeledPoint(
                            x=(item.x_min + item.x_max) / 2.0,
                            y=(item.y_min + item.y_max) / 2.0,
                            class_name=object_name,
                            source_side=item.source_side,
                        )
                        for item in class_boxes
                    ]
            for object_name, class_points in sorted(grouped_points.items()):
                composite_points = [common.remap_point(point, layout=layout) for point in class_points]
                class_boxes = grouped_boxes.get(object_name, [])
                composite_boxes = [common.remap_box(box, layout=layout) for box in class_boxes]
                out.append(
                    _build_spatial_manifest_row(
                        row_id=_slugify(f"{row_id}_{object_name}_point"),
                        split=split,
                        task_name=task_name,
                        skill="point",
                        object_name=_object_prompt(object_name=object_name, source_side=class_points[0].source_side if class_points else default_side),
                        image_path=composite_path,
                        boxes=composite_boxes,
                        points=composite_points,
                        source_pre_image_path=source_pre_path,
                        source_post_image_path=source_post_path,
                        source_image_path=source_single_path,
                        source_json_path=source_json_path,
                        source_row_index=source_row_index,
                        annotation_side=class_points[0].source_side if class_points else default_side,
                    )
                )
        else:
            for object_name, class_boxes in sorted(grouped_boxes.items()):
                composite_boxes = [common.remap_box(box, layout=layout) for box in class_boxes]
                out.append(
                    _build_spatial_manifest_row(
                        row_id=_slugify(f"{row_id}_{object_name}_detect"),
                        split=split,
                        task_name=task_name,
                        skill="detect",
                        object_name=_object_prompt(object_name=object_name, source_side=class_boxes[0].source_side if class_boxes else default_side),
                        image_path=composite_path,
                        boxes=composite_boxes,
                        points=[],
                        source_pre_image_path=source_pre_path,
                        source_post_image_path=source_post_path,
                        source_image_path=source_single_path,
                        source_json_path=source_json_path,
                        source_row_index=source_row_index,
                        annotation_side=class_boxes[0].source_side if class_boxes else default_side,
                    )
                )
        return out
    if not question or not answer_text:
        stats["skipped_missing_query_target"] += 1
        return []
    final_answer, query_kind, options_by_letter = _normalize_query_target(task_name=task_name, answer_text=answer_text, row=row)
    out.append(
        _build_query_manifest_row(
            row_id=row_id,
            split=split,
            task_name=task_name,
            question=_query_prompt(task_kind=query_kind, original_question=question, options_by_letter=options_by_letter),
            image_path=composite_path,
            final_answer=final_answer,
            query_kind=query_kind,
            options_by_letter=options_by_letter,
            source_pre_image_path=source_pre_path,
            source_post_image_path=source_post_path,
            source_image_path=source_single_path,
            source_json_path=source_json_path,
            source_row_index=source_row_index,
        )
    )
    return out


def _write_split_jsonl(output_dir: Path, split: str, rows: Sequence[Mapping[str, Any]]) -> None:
    jsonl_dir = output_dir / "jsonl"
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    path = jsonl_dir / f"{split}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _collect_process_candidates(
    *,
    root: Path,
    default_split: str,
    image_index: common.ImageIndex,
    stats: Counter[str],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for json_path in _candidate_json_files(root):
        rows = _load_rows_from_json_file(json_path)
        if not _looks_like_task_file(json_path, rows):
            continue
        for row_index, row in enumerate(rows):
            task_name = _infer_task_name(json_path, row)
            if common.is_segmentation_task(task_name, row):
                stats["skipped_segmentation"] += 1
                continue
            split = _discover_split(json_path, row, source_default=default_split)
            pre_ref, post_ref, single_ref = _extract_image_refs(row, image_index=image_index)
            if pre_ref is None and post_ref is None and single_ref is None:
                raw_image_values = _image_reference_values(row)
                if raw_image_values and any(not common.is_rgb_reference(value) for value in raw_image_values):
                    stats["skipped_non_rgb_variant"] += 1
                else:
                    stats["skipped_missing_image"] += 1
                continue
            candidates.append(
                {
                    "row": row,
                    "task_name": task_name,
                    "split": split,
                    "source_path": json_path,
                    "source_row_index": row_index,
                }
            )
    return candidates


def _sample_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    max_samples: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if max_samples <= 0:
        return [dict(item) for item in candidates], {"used": False, "reason": "disabled"}
    if len(candidates) <= max_samples:
        return [dict(item) for item in candidates], {
            "used": False,
            "reason": "candidate_count_below_limit",
            "candidate_rows": len(candidates),
            "selected_rows": len(candidates),
        }

    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        by_split[str(item.get("split") or "train")].append(dict(item))
    split_names = [name for name in ("train", "val", "test") if by_split.get(name)]
    total_candidates = len(candidates)
    allocations: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    reserved = 0
    if max_samples >= len(split_names):
        for split_name in split_names:
            allocations[split_name] = 1
            reserved += 1
    remaining_budget = max_samples - reserved
    remaining_total = max(1, total_candidates - reserved)
    for split_name in split_names:
        available = len(by_split[split_name]) - allocations.get(split_name, 0)
        if available <= 0 or remaining_budget <= 0:
            remainders.append((0.0, split_name))
            continue
        exact = remaining_budget * (available / float(remaining_total))
        whole = min(available, int(exact))
        allocations[split_name] = allocations.get(split_name, 0) + whole
        remainders.append((exact - whole, split_name))
    assigned = sum(allocations.values())
    leftover = max(0, max_samples - assigned)
    for _, split_name in sorted(remainders, key=lambda item: (-item[0], item[1])):
        if leftover <= 0:
            break
        available = len(by_split[split_name]) - allocations.get(split_name, 0)
        if available <= 0:
            continue
        allocations[split_name] = allocations.get(split_name, 0) + 1
        leftover -= 1

    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    selected_counts: dict[str, int] = {}
    for split_name in split_names:
        pool = list(by_split[split_name])
        rng.shuffle(pool)
        keep = min(len(pool), allocations.get(split_name, 0))
        selected.extend(pool[:keep])
        selected_counts[split_name] = keep
    rng.shuffle(selected)
    return selected[:max_samples], {
        "used": True,
        "reason": "stratified_row_sampling",
        "candidate_rows": total_candidates,
        "selected_rows": min(len(selected), max_samples),
        "requested_max_samples": int(max_samples),
        "seed": int(seed),
        "selected_split_counts": selected_counts,
    }


def _group_key_for_split_fallback(row: Mapping[str, Any]) -> str:
    return "||".join(
        [
            str(row.get("source_pre_image_path") or ""),
            str(row.get("source_post_image_path") or ""),
            str(row.get("source_image_path") or ""),
        ]
    )


def _derive_missing_splits(split_rows: dict[str, list[dict[str, Any]]], *, seed: int = 42) -> dict[str, Any]:
    if not split_rows["train"]:
        return {"used": False, "reason": "no_train_rows"}
    need_val = not split_rows["val"]
    need_test = not split_rows["test"]
    if not need_val and not need_test:
        return {"used": False, "reason": "official_splits_present"}

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in split_rows["train"]:
        grouped[_group_key_for_split_fallback(row)].append(row)
    group_keys = sorted(grouped.keys())
    random.Random(seed).shuffle(group_keys)
    group_count = len(group_keys)
    if group_count <= 1:
        if need_val:
            split_rows["val"] = [dict(row, split="val") for row in split_rows["train"]]
        if need_test:
            source_rows = split_rows["val"] if split_rows["val"] else split_rows["train"]
            split_rows["test"] = [dict(row, split="test") for row in source_rows]
        return {
            "used": True,
            "reason": "single_group_clone",
            "groups": group_count,
            "derived_val_groups": group_count if need_val else 0,
            "derived_test_groups": group_count if need_test else 0,
        }

    target_val_groups = max(1, round(group_count * 0.05)) if need_val else 0
    target_test_groups = max(1, round(group_count * 0.05)) if need_test else 0
    max_holdout = max(1, group_count - 1)
    if target_val_groups + target_test_groups > max_holdout:
        overflow = (target_val_groups + target_test_groups) - max_holdout
        while overflow > 0 and target_test_groups > 0:
            target_test_groups -= 1
            overflow -= 1
        while overflow > 0 and target_val_groups > 0:
            target_val_groups -= 1
            overflow -= 1
        target_val_groups = max(target_val_groups, 1 if need_val else 0)
        target_test_groups = max(target_test_groups, 1 if need_test and group_count - target_val_groups > 1 else 0)

    val_keys = set(group_keys[:target_val_groups]) if need_val else set()
    test_start = target_val_groups
    test_end = test_start + target_test_groups
    test_keys = set(group_keys[test_start:test_end]) if need_test else set()

    remaining_train: list[dict[str, Any]] = []
    derived_val: list[dict[str, Any]] = []
    derived_test: list[dict[str, Any]] = []
    for key in group_keys:
        rows = grouped[key]
        if key in val_keys:
            derived_val.extend(dict(row, split="val") for row in rows)
        elif key in test_keys:
            derived_test.extend(dict(row, split="test") for row in rows)
        else:
            remaining_train.extend(rows)

    split_rows["train"] = remaining_train
    if need_val:
        split_rows["val"] = derived_val
    if need_test:
        split_rows["test"] = derived_test if derived_test else [dict(row, split="test") for row in derived_val]
    return {
        "used": True,
        "reason": "derived_holdout_from_train",
        "groups": group_count,
        "derived_val_groups": len(val_keys),
        "derived_test_groups": len(test_keys),
    }


def _counter_breakdown(
    counter: Mapping[str, int],
    *,
    raw_total: int,
    usable_total: int,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for key, count in sorted(counter.items()):
        out[str(key)] = {
            "count": int(count),
            "pct_of_raw_total": (float(count) / float(max(1, raw_total))) * 100.0,
            "pct_of_usable_total": (float(count) / float(max(1, usable_total))) * 100.0,
        }
    return out


def _safe_json_dict(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(str(text or "{}"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_json_list(text: str) -> list[Any]:
    try:
        payload = json.loads(str(text or "[]"))
    except json.JSONDecodeError:
        return []
    return payload if isinstance(payload, list) else []


def _sample_entry_from_manifest_row(row: Mapping[str, Any]) -> dict[str, Any]:
    gt_payload: Any
    if str(row.get("skill") or "") == "query":
        gt_payload = _safe_json_dict(str(row.get("final_answer_json") or "{}"))
    elif str(row.get("skill") or "") == "detect":
        gt_payload = _safe_json_list(str(row.get("answer_boxes_json") or "[]"))
    else:
        gt_payload = {
            "points": _safe_json_list(str(row.get("answer_points_json") or "[]")),
            "boxes": _safe_json_list(str(row.get("answer_boxes_json") or "[]")),
        }
    metadata = _safe_json_dict(str(row.get("metadata_json") or "{}"))
    return {
        "row_id": str(row.get("row_id") or ""),
        "split": str(row.get("split") or ""),
        "task_name": str(row.get("task_name") or ""),
        "task_family": str(row.get("task_family") or ""),
        "skill": str(row.get("skill") or ""),
        "question": str(row.get("question") or ""),
        "object_name": str(row.get("object_name") or ""),
        "image_path": str(row.get("image_path") or ""),
        "ground_truth": gt_payload,
        "source_pre_image_path": str(row.get("source_pre_image_path") or ""),
        "source_post_image_path": str(row.get("source_post_image_path") or ""),
        "source_image_path": str(row.get("source_image_path") or ""),
        "source_json_path": str(metadata.get("source_json_path") or ""),
        "source_row_index": metadata.get("source_row_index"),
    }


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_none_"
    header_line = "| " + " | ".join(headers) + " |"
    divider = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, divider, *body])


def _render_dataset_report_markdown(report: Mapping[str, Any]) -> str:
    totals = report["totals"]
    raw_task_rows = [
        [
            task_name,
            str(values["count"]),
            f"{values['pct_of_raw_total']:.2f}%",
        ]
        for task_name, values in report["raw_task_counts"].items()
    ]
    family_rows = [
        [
            task_name,
            str(values["count"]),
            f"{values['pct_of_usable_total']:.2f}%",
        ]
        for task_name, values in report["task_family_counts"].items()
    ]
    skill_rows = [
        [
            skill,
            str(values["count"]),
            f"{values['pct_of_usable_total']:.2f}%",
        ]
        for skill, values in report["skill_counts"].items()
    ]
    sample_lines: list[str] = []
    task_samples = report.get("task_samples", {})
    for task_name, samples in sorted(task_samples.items()):
        sample_lines.append(f"### `{task_name}`")
        for sample in list(samples)[:1]:
            sample_lines.append(f"- `row_id`: `{sample['row_id']}`")
            if sample.get("question"):
                sample_lines.append(f"- `question`: {sample['question']}")
            if sample.get("object_name"):
                sample_lines.append(f"- `object_name`: {sample['object_name']}")
            sample_lines.append(f"- `ground_truth`: `{json.dumps(sample['ground_truth'], ensure_ascii=False)}`")
    return "\n".join(
        [
            "# DisasterM3 Dataset Report",
            "",
            "## Totals",
            "",
            f"- raw_total_rows: {totals['raw_total_rows']}",
            f"- usable_source_rows: {totals['usable_source_rows']}",
            f"- built_manifest_rows: {totals['built_manifest_rows']}",
            f"- skipped_segmentation_rows: {totals['skipped_segmentation_rows']}",
            f"- skipped_non_rgb_rows: {totals['skipped_non_rgb_rows']}",
            f"- skipped_missing_image_rows: {totals['skipped_missing_image_rows']}",
            f"- skipped_missing_query_target_rows: {totals['skipped_missing_query_target_rows']}",
            "",
            "## Raw Tasks",
            "",
            _markdown_table(["task_name", "count", "% raw"], raw_task_rows),
            "",
            "## Task Families",
            "",
            _markdown_table(["task_family", "count", "% built"], family_rows),
            "",
            "## Skills",
            "",
            _markdown_table(["skill", "count", "% built"], skill_rows),
            "",
            "## Representative Samples",
            "",
            *sample_lines,
            "",
        ]
    )


def _scan_source_tree(
    *,
    root: Path,
    default_split: str,
    image_index: common.ImageIndex,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    raw_stats = Counter()
    raw_task_counts = Counter()
    raw_split_counts = Counter()
    for json_path in _candidate_json_files(root):
        rows = _load_rows_from_json_file(json_path)
        if not _looks_like_task_file(json_path, rows):
            continue
        for row_index, row in enumerate(rows):
            task_name = _infer_task_name(json_path, row)
            split = _discover_split(json_path, row, source_default=default_split)
            raw_stats["raw_total_rows"] += 1
            raw_task_counts[task_name] += 1
            raw_split_counts[split] += 1
            if common.is_segmentation_task(task_name, row):
                raw_stats["skipped_segmentation"] += 1
                continue
            pre_ref, post_ref, single_ref = _extract_image_refs(row, image_index=image_index)
            if pre_ref is None and post_ref is None and single_ref is None:
                raw_image_values = _image_reference_values(row)
                if raw_image_values and any(not common.is_rgb_reference(value) for value in raw_image_values):
                    raw_stats["skipped_non_rgb_variant"] += 1
                else:
                    raw_stats["skipped_missing_image"] += 1
                continue
            raw_stats["usable_source_rows"] += 1
            candidates.append(
                {
                    "row": row,
                    "task_name": task_name,
                    "split": split,
                    "source_path": json_path,
                    "source_row_index": row_index,
                }
            )
    return {
        "candidates": candidates,
        "raw_stats": dict(raw_stats),
        "raw_task_counts": dict(raw_task_counts),
        "raw_split_counts": dict(raw_split_counts),
    }


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    common.maybe_load_env_file(args.env_file)
    hf_token = common.resolve_hf_token(args.hf_token, env_file=args.env_file)
    raw_dir = Path(args.raw_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if args.download:
        if args.force_download and raw_dir.exists():
            for path in sorted(raw_dir.iterdir()):
                if path.is_file():
                    path.unlink()
        common.snapshot_download_dataset(hf_repo_id=str(args.hf_repo_id), raw_dir=raw_dir, hf_token=hf_token)
    if not raw_dir.exists():
        raise FileNotFoundError(f"raw_dir does not exist: {raw_dir}")
    instruct_root = common.discover_extracted_root(raw_dir, archive_stem="DisasterM3_Instruct")
    if instruct_root is None:
        raise FileNotFoundError(f"Could not find or extract DisasterM3_Instruct under {raw_dir}")
    bench_root = raw_dir / "DisasterM3_Bench"
    if not bench_root.exists():
        print(f"warning: benchmark directory not found under {raw_dir}; continuing without benchmark test split")
    image_index = common.load_image_index(raw_dir)
    split_rows: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    build_skip_stats = Counter()
    by_task = Counter()
    by_skill = Counter()
    by_task_family = Counter()

    def process_candidates(candidates: Sequence[Mapping[str, Any]]) -> None:
        for candidate in candidates:
            row = candidate["row"]
            task_name = str(candidate["task_name"])
            split = str(candidate["split"])
            source_path = Path(str(candidate["source_path"]))
            source_row_index = int(candidate["source_row_index"])
            manifest_rows = _adapt_row(
                row=row,
                task_name=task_name,
                split=split,
                source_path=source_path,
                source_row_index=source_row_index,
                image_index=image_index,
                output_dir=output_dir,
                raw_root=raw_dir,
                panel_size=int(args.panel_max_side),
                divider_px=int(args.divider_px),
                jpeg_quality=int(args.jpeg_quality),
                stats=build_skip_stats,
            )
            for manifest_row in manifest_rows:
                split_rows[split].append(manifest_row)
                by_task[str(manifest_row["task_name"])] += 1
                by_skill[str(manifest_row["skill"])] += 1
                by_task_family[str(manifest_row["task_family"])] += 1

    instruct_scan = _scan_source_tree(root=instruct_root, default_split="", image_index=image_index)
    all_candidates = list(instruct_scan["candidates"])
    raw_stats = Counter(instruct_scan["raw_stats"])
    raw_task_counts = Counter(instruct_scan["raw_task_counts"])
    raw_split_counts = Counter(instruct_scan["raw_split_counts"])
    if bench_root.exists():
        bench_scan = _scan_source_tree(root=bench_root, default_split="test", image_index=image_index)
        all_candidates.extend(bench_scan["candidates"])
        raw_stats.update(bench_scan["raw_stats"])
        raw_task_counts.update(bench_scan["raw_task_counts"])
        raw_split_counts.update(bench_scan["raw_split_counts"])

    subset_info = {"used": False, "reason": "disabled"}
    selected_candidates: list[dict[str, Any]]
    if int(args.max_samples) > 0:
        selected_candidates, subset_info = _sample_candidates(
            all_candidates,
            max_samples=int(args.max_samples),
            seed=int(args.subset_seed),
        )
        process_candidates(selected_candidates)
    else:
        selected_candidates = [dict(item) for item in all_candidates]
        process_candidates(selected_candidates)
    split_fallback = _derive_missing_splits(split_rows, seed=int(args.subset_seed))

    for split_name, rows in split_rows.items():
        if not rows:
            continue
        rows.sort(key=lambda item: str(item.get("row_id") or ""))
        _write_split_jsonl(output_dir, split_name, rows)

    task_samples: dict[str, list[dict[str, Any]]] = {}
    for task_name in sorted(by_task):
        collected: list[dict[str, Any]] = []
        for split_name in ("train", "val", "test"):
            for row in split_rows[split_name]:
                if str(row.get("task_name") or "") != task_name:
                    continue
                collected.append(_sample_entry_from_manifest_row(row))
                if len(collected) >= 2:
                    break
            if len(collected) >= 2:
                break
        task_samples[task_name] = collected

    raw_total_rows = int(raw_stats.get("raw_total_rows", 0))
    usable_source_rows = int(raw_stats.get("usable_source_rows", 0))
    built_manifest_rows = sum(len(rows) for rows in split_rows.values())

    metadata = {
        "hf_repo_id": str(args.hf_repo_id),
        "dataset_root": str(args.dataset_root),
        "raw_dir": str(raw_dir),
        "output_dir": str(output_dir),
        "panel_max_side": int(args.panel_max_side),
        "divider_px": int(args.divider_px),
        "jpeg_quality": int(args.jpeg_quality),
        "subset": subset_info,
        "split_counts": {split: len(rows) for split, rows in split_rows.items()},
        "split_fallback": split_fallback,
        "raw_split_counts": dict(sorted(raw_split_counts.items())),
        "skill_counts": dict(sorted(by_skill.items())),
        "task_family_counts": dict(sorted(by_task_family.items())),
        "task_counts": dict(sorted(by_task.items())),
    }
    build_stats = {
        "raw_total_rows": raw_total_rows,
        "usable_source_rows": usable_source_rows,
        "built_manifest_rows": built_manifest_rows,
        "usable_split_counts": {split: len(rows) for split, rows in split_rows.items()},
        "subset": subset_info,
        "split_fallback": split_fallback,
        "skill_counts": dict(sorted(by_skill.items())),
        "task_family_counts": dict(sorted(by_task_family.items())),
        "task_counts": dict(sorted(by_task.items())),
        "skip_counts": dict(sorted(build_skip_stats.items())),
        "raw_skip_counts": {
            "segmentation": int(raw_stats.get("skipped_segmentation", 0)),
            "non_rgb_variant": int(raw_stats.get("skipped_non_rgb_variant", 0)),
            "missing_image": int(raw_stats.get("skipped_missing_image", 0)),
        },
    }
    dataset_report = {
        "totals": {
            "raw_total_rows": raw_total_rows,
            "usable_source_rows": usable_source_rows,
            "built_manifest_rows": built_manifest_rows,
            "skipped_segmentation_rows": int(raw_stats.get("skipped_segmentation", 0)),
            "skipped_non_rgb_rows": int(raw_stats.get("skipped_non_rgb_variant", 0)),
            "skipped_missing_image_rows": int(raw_stats.get("skipped_missing_image", 0)),
            "skipped_missing_query_target_rows": int(build_skip_stats.get("skipped_missing_query_target", 0)),
        },
        "split_counts": {
            "raw": dict(sorted(raw_split_counts.items())),
            "built": {split: len(rows) for split, rows in split_rows.items()},
        },
        "raw_task_counts": _counter_breakdown(
            raw_task_counts,
            raw_total=raw_total_rows,
            usable_total=usable_source_rows,
        ),
        "task_counts": _counter_breakdown(
            by_task,
            raw_total=raw_total_rows,
            usable_total=built_manifest_rows,
        ),
        "task_family_counts": _counter_breakdown(
            by_task_family,
            raw_total=raw_total_rows,
            usable_total=built_manifest_rows,
        ),
        "skill_counts": _counter_breakdown(
            by_skill,
            raw_total=raw_total_rows,
            usable_total=built_manifest_rows,
        ),
        "task_samples": task_samples,
        "subset": subset_info,
        "split_fallback": split_fallback,
    }
    dataset_report_md = _render_dataset_report_markdown(dataset_report)
    common.write_json(output_dir / "metadata.json", metadata)
    common.write_json(output_dir / "build_stats.json", build_stats)
    common.write_json(output_dir / "dataset_report.json", dataset_report)
    common.write_json(output_dir / "task_samples.json", task_samples)
    (output_dir / "dataset_report.md").write_text(dataset_report_md, encoding="utf-8")
    return {
        "metadata": metadata,
        "build_stats": build_stats,
        "dataset_report": dataset_report,
    }


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    summary = build_dataset(args)
    print(
        "saved disaster_m3 dataset:",
        summary["metadata"]["output_dir"],
        f"train={summary['metadata']['split_counts'].get('train', 0)}",
        f"val={summary['metadata']['split_counts'].get('val', 0)}",
        f"test={summary['metadata']['split_counts'].get('test', 0)}",
    )


if __name__ == "__main__":
    main()
