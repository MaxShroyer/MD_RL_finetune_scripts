#!/usr/bin/env python3
"""Normalize the local Aerial Airport COCO export into an HF DatasetDict."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from datasets import Dataset, DatasetDict, Features, Image as HFImage, Value
from dotenv import load_dotenv
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aerial_airport.common import (
    DEFAULT_CLASS_NAME,
    DEFAULT_CLASS_UID,
    DEFAULT_HF_DATASET_NAME,
    DEFAULT_RAW_DATASET_DIR,
    build_class_catalog,
    class_uid_for_name,
    clamp,
    config_to_cli_args,
    discover_class_names,
    load_json_config,
    normalize_class_name,
    repo_relative,
    resolve_config_path,
    write_json,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = repo_relative("configs", "build_aerial_airport_hf_dataset_default.json")
SOURCE_SPLITS = ("train", "valid", "test")
OUTPUT_SPLIT_MAP = {
    "train": "train",
    "valid": "validation",
    "test": "test",
}
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
SOURCE_VARIANT_EXPORT = "raw_coco_export"
SOURCE_VARIANT_BACKGROUND_NEGATIVE = "background_negative"
TILING_CHOICES = ("none", "2x2")
BACKGROUND_SCALES = (0.35, 0.50)
BACKGROUND_GRID_POSITIONS = (0.0, 0.5, 1.0)
BACKGROUND_PADDING = 0.02
SPLIT_STRATEGIES = ("random_group", "stratified_empty_group")
EMPTY_FRACTION_TOLERANCE = 0.03


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    raw_argv = list(argv) if argv is not None else list(os.sys.argv[1:])
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    pre_args, _ = pre_parser.parse_known_args(raw_argv)
    config_path = resolve_config_path(pre_args.config, script_dir=SCRIPT_DIR)
    config = load_json_config(config_path, default_path=DEFAULT_CONFIG_PATH)

    parser = argparse.ArgumentParser(description="Build the local Aerial Airport COCO export into an HF DatasetDict.")
    parser.add_argument("--config", default=str(config_path))
    parser.add_argument("--env-file", "--env", default=str(repo_relative(".env.staging")))
    parser.add_argument("--raw-dataset-dir", default=str(DEFAULT_RAW_DATASET_DIR))
    parser.add_argument("--output-dir", default=str(repo_relative("outputs", "maxs-m87_aerial_airport_point_v2")))
    parser.add_argument("--tiling", choices=TILING_CHOICES, default="none")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument(
        "--split-strategy",
        choices=SPLIT_STRATEGIES,
        default="stratified_empty_group",
    )
    parser.add_argument("--target-empty-fraction", type=float, default=0.10)
    parser.add_argument("--push-to-hub", default=DEFAULT_HF_DATASET_NAME)
    parser.add_argument("--hub-val-split", default="validation")
    parser.add_argument("--hub-post-val-split", default="test")
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"),
    )

    option_to_dest: dict[str, str] = {}
    for action in parser._actions:
        if not action.option_strings:
            continue
        for opt in action.option_strings:
            option_to_dest[opt] = action.dest
    overridden_dests = {option_to_dest[arg] for arg in raw_argv if arg in option_to_dest}
    config_cli_args = config_to_cli_args(
        parser,
        config,
        config_path=config_path,
        overridden_dests=overridden_dests,
    )
    args = parser.parse_args(config_cli_args + raw_argv)
    args.config = str(resolve_config_path(args.config, script_dir=SCRIPT_DIR))
    return args


def _features() -> Features:
    return Features(
        {
            "image": HFImage(),
            "answer_boxes": Value("string"),
            "source_dataset": Value("string"),
            "source_collection": Value("string"),
            "source_variant": Value("string"),
            "source_is_synthetic": Value("bool"),
            "source_split": Value("string"),
            "source_image_id": Value("string"),
            "source_base_id": Value("string"),
            "split_group_id": Value("string"),
            "class_count": Value("int32"),
        }
    )


def _resolve_dir(path_str: str, *, fallback_name: str) -> Path:
    if path_str:
        return Path(path_str).expanduser().resolve()
    return repo_relative("outputs", fallback_name).resolve()


def _find_coco_annotation_file(split_dir: Path) -> Optional[Path]:
    preferred = split_dir / "_annotations.coco.json"
    if preferred.exists():
        return preferred
    matches = sorted(path for path in split_dir.glob("*.json") if path.is_file())
    if not matches:
        return None
    coco_matches = [path for path in matches if path.name.endswith(".coco.json")]
    if coco_matches:
        return coco_matches[0]
    return matches[0]


def _has_coco_split(split_dir: Path) -> bool:
    annotation_path = _find_coco_annotation_file(split_dir)
    return split_dir.exists() and annotation_path is not None


def _resolve_image_path(split_dir: Path, file_name: str) -> Path:
    candidate = split_dir / file_name
    if candidate.exists():
        return candidate
    basename = Path(file_name).name
    candidate = split_dir / basename
    if candidate.exists():
        return candidate
    stem = Path(basename).stem
    for suffix in IMAGE_SUFFIXES:
        alt = split_dir / f"{stem}{suffix}"
        if alt.exists():
            return alt
    raise FileNotFoundError(f"Image file referenced by COCO annotations not found under {split_dir}: {file_name}")


def _source_base_id(image_info: Mapping[str, Any]) -> str:
    extra = image_info.get("extra")
    if isinstance(extra, dict):
        extra_name = str(extra.get("name") or "").strip()
        if extra_name:
            return Path(extra_name).stem
    stem = Path(str(image_info.get("file_name") or "")).stem
    return stem.split(".rf.", 1)[0] if ".rf." in stem else stem


def _normalized_box_from_coco_bbox(
    bbox: Any,
    *,
    width: int,
    height: int,
) -> Optional[dict[str, float]]:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    try:
        x_min, y_min, box_w, box_h = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return None
    x_max = x_min + box_w
    y_max = y_min + box_h
    if width <= 0 or height <= 0:
        return None
    x0 = clamp(x_min / float(width))
    y0 = clamp(y_min / float(height))
    x1 = clamp(x_max / float(width))
    y1 = clamp(y_max / float(height))
    if x1 <= x0 or y1 <= y0:
        return None
    return {
        "x_min": x0,
        "y_min": y0,
        "x_max": x1,
        "y_max": y1,
    }


def _answer_boxes_payload(boxes: Iterable[Mapping[str, Any]]) -> str:
    payload = [
        {
            "x_min": float(box["x_min"]),
            "y_min": float(box["y_min"]),
            "x_max": float(box["x_max"]),
            "y_max": float(box["y_max"]),
            "class_uid": str(box["class_uid"]),
            "class_name": str(box["class_name"]),
            "source_class_name": str(box["source_class_name"]),
        }
        for box in boxes
    ]
    return json.dumps(payload, separators=(",", ":"))


def _build_row(
    *,
    image_path: Path,
    boxes: list[dict[str, Any]],
    source_dataset: str,
    source_collection: str,
    source_split: str,
    source_variant: str,
    source_is_synthetic: bool,
    source_image_id: str,
    source_base_id: str,
    split_group_id: str,
) -> dict[str, Any]:
    return {
        "image": str(image_path.resolve()),
        "answer_boxes": _answer_boxes_payload(boxes),
        "source_dataset": source_dataset,
        "source_collection": source_collection,
        "source_variant": source_variant,
        "source_is_synthetic": bool(source_is_synthetic),
        "source_split": source_split,
        "source_image_id": source_image_id,
        "source_base_id": source_base_id,
        "split_group_id": split_group_id,
        "class_count": len(boxes),
    }


def _crop_output_suffix(image_path: Path) -> str:
    suffix = image_path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return suffix
    return ".jpg"


def _save_crop(image: Image.Image, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() in {".jpg", ".jpeg"} and image.mode not in {"RGB", "L"}:
        image = image.convert("RGB")
    image.save(out_path)


def _tile_window_2x2(*, width: int, height: int, tile_row: int, tile_col: int) -> tuple[int, int, int, int]:
    mid_x = width // 2
    mid_y = height // 2
    left = 0 if tile_col == 0 else mid_x
    right = mid_x if tile_col == 0 else width
    top = 0 if tile_row == 0 else mid_y
    bottom = mid_y if tile_row == 0 else height
    return left, top, right, bottom


def _clip_box_to_window(
    box: Mapping[str, Any],
    *,
    window: Mapping[str, float],
) -> Optional[dict[str, Any]]:
    window_x_min = float(window["x_min"])
    window_y_min = float(window["y_min"])
    window_x_max = float(window["x_max"])
    window_y_max = float(window["y_max"])
    window_width = window_x_max - window_x_min
    window_height = window_y_max - window_y_min
    if window_width <= 0.0 or window_height <= 0.0:
        return None

    x_min = max(float(box["x_min"]), window_x_min)
    y_min = max(float(box["y_min"]), window_y_min)
    x_max = min(float(box["x_max"]), window_x_max)
    y_max = min(float(box["y_max"]), window_y_max)
    if x_max <= x_min or y_max <= y_min:
        return None

    clipped = dict(box)
    clipped["x_min"] = clamp((x_min - window_x_min) / window_width)
    clipped["y_min"] = clamp((y_min - window_y_min) / window_height)
    clipped["x_max"] = clamp((x_max - window_x_min) / window_width)
    clipped["y_max"] = clamp((y_max - window_y_min) / window_height)
    return clipped


def _tile_row_2x2(
    row: Mapping[str, Any],
    *,
    output_dir: Path,
) -> list[dict[str, Any]]:
    image_path = Path(str(row["image"]))
    answer_boxes = json.loads(str(row["answer_boxes"]))
    source_image_id = str(row["source_image_id"])
    source_split = str(row["source_split"])
    suffix = _crop_output_suffix(image_path)
    tiled_rows: list[dict[str, Any]] = []

    with Image.open(image_path) as image:
        width, height = image.size
        for tile_row in range(2):
            for tile_col in range(2):
                left, top, right, bottom = _tile_window_2x2(
                    width=width,
                    height=height,
                    tile_row=tile_row,
                    tile_col=tile_col,
                )
                tile_window = _normalized_window(left, top, right, bottom, width, height)
                tile_boxes = [
                    clipped
                    for box in answer_boxes
                    if (clipped := _clip_box_to_window(box, window=tile_window)) is not None
                ]
                tile_source_image_id = f"{source_image_id}__tile_r{tile_row}_c{tile_col}"
                out_path = output_dir / "tiles" / source_split / f"{tile_source_image_id}{suffix}"
                _save_crop(image.crop((left, top, right, bottom)), out_path)
                tiled_rows.append(
                    _build_row(
                        image_path=out_path,
                        boxes=tile_boxes,
                        source_dataset=str(row["source_dataset"]),
                        source_collection=str(row["source_collection"]),
                        source_split=source_split,
                        source_variant=str(row["source_variant"]),
                        source_is_synthetic=bool(row["source_is_synthetic"]),
                        source_image_id=tile_source_image_id,
                        source_base_id=str(row["source_base_id"]),
                        split_group_id=str(row["split_group_id"]),
                    )
                )
    return tiled_rows


def _apply_tiling(
    rows: list[dict[str, Any]],
    *,
    output_dir: Path,
    tiling: str,
) -> list[dict[str, Any]]:
    if tiling == "none":
        return rows
    if tiling == "2x2":
        tiled_rows: list[dict[str, Any]] = []
        for row in rows:
            tiled_rows.extend(_tile_row_2x2(row, output_dir=output_dir))
        return tiled_rows
    raise ValueError(f"Unsupported tiling mode: {tiling}")


def _rows_from_coco_split(
    *,
    split_dir: Path,
    source_dataset: str,
    source_collection: str,
    source_split: str,
) -> list[dict[str, Any]]:
    annotation_path = _find_coco_annotation_file(split_dir)
    if annotation_path is None:
        raise FileNotFoundError(f"COCO annotation file not found under {split_dir}")

    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    images = payload.get("images") or []
    annotations = payload.get("annotations") or []
    categories = payload.get("categories") or []

    categories_by_id = {
        int(item["id"]): normalize_class_name(item.get("name"))
        for item in categories
        if isinstance(item, dict) and "id" in item
    }
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in annotations:
        if not isinstance(annotation, dict):
            continue
        try:
            image_id = int(annotation["image_id"])
        except (KeyError, TypeError, ValueError):
            continue
        annotations_by_image[image_id].append(annotation)

    rows: list[dict[str, Any]] = []
    for image_info in images:
        if not isinstance(image_info, dict):
            continue
        try:
            image_id = int(image_info["id"])
            width = int(image_info["width"])
            height = int(image_info["height"])
        except (KeyError, TypeError, ValueError):
            continue

        file_name = str(image_info.get("file_name") or "").strip()
        if not file_name:
            continue
        image_path = _resolve_image_path(split_dir, file_name)
        image_annotations = annotations_by_image.get(image_id, [])
        boxes: list[dict[str, Any]] = []
        for annotation in image_annotations:
            category_name = normalize_class_name(categories_by_id.get(int(annotation.get("category_id", -1)), ""))
            if not category_name:
                continue
            normalized_box = _normalized_box_from_coco_bbox(annotation.get("bbox"), width=width, height=height)
            if normalized_box is None:
                continue
            boxes.append(
                {
                    **normalized_box,
                    "class_uid": class_uid_for_name(category_name),
                    "class_name": category_name,
                    "source_class_name": str(category_name),
                }
            )

        source_image_id = Path(file_name).stem
        source_base_id = _source_base_id(image_info)
        rows.append(
            _build_row(
                image_path=image_path,
                boxes=boxes,
                source_dataset=source_dataset,
                source_collection=source_collection,
                source_split=source_split,
                source_variant=SOURCE_VARIANT_EXPORT,
                source_is_synthetic=False,
                source_image_id=source_image_id,
                source_base_id=source_base_id,
                split_group_id=f"group:{source_base_id}",
            )
        )
    return rows


def _validate_split_fractions(*, val_fraction: float, test_fraction: float) -> None:
    if not (0.0 < val_fraction < 1.0):
        raise ValueError("--val-fraction must be in (0, 1)")
    if not (0.0 < test_fraction < 1.0):
        raise ValueError("--test-fraction must be in (0, 1)")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("--val-fraction + --test-fraction must be < 1")


def _target_split_row_counts(
    total_rows: int,
    *,
    val_fraction: float,
    test_fraction: float,
) -> dict[str, int]:
    _validate_split_fractions(val_fraction=val_fraction, test_fraction=test_fraction)
    if total_rows <= 0:
        raise ValueError("Cannot split an empty dataset.")

    test_rows = int(round(total_rows * test_fraction))
    val_rows = int(round(total_rows * val_fraction))
    if total_rows >= 3:
        if test_rows <= 0:
            test_rows = 1
        if val_rows <= 0:
            val_rows = 1
    if test_rows + val_rows >= total_rows:
        overflow = (test_rows + val_rows) - (total_rows - 1)
        while overflow > 0 and test_rows > 1:
            test_rows -= 1
            overflow -= 1
        while overflow > 0 and val_rows > 1:
            val_rows -= 1
            overflow -= 1
        if test_rows + val_rows >= total_rows:
            raise ValueError("Not enough rows to form train/validation/test splits.")
    train_rows = total_rows - test_rows - val_rows
    if train_rows <= 0:
        raise ValueError("Not enough rows to form a non-empty train split.")
    return {
        "train": train_rows,
        "validation": val_rows,
        "test": test_rows,
    }


def _group_rows_by_id(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["split_group_id"])].append(row)

    group_infos: list[dict[str, Any]] = []
    for group_id, group_rows in grouped.items():
        group_infos.append(
            {
                "group_id": group_id,
                "rows": group_rows,
                "row_count": len(group_rows),
                "empty_rows": _count_empty_rows(group_rows),
            }
        )
    return group_infos


def _split_balance_objective(
    *,
    assigned_group_counts: Mapping[str, int],
    assigned_row_counts: Mapping[str, int],
    assigned_empty_counts: Mapping[str, int],
    target_row_counts: Mapping[str, int],
    target_empty_counts: Mapping[str, float],
) -> float:
    score = 0.0
    for split_name in ("train", "validation", "test"):
        if assigned_group_counts[split_name] <= 0 or assigned_row_counts[split_name] <= 0:
            score += 1_000_000.0
            continue
        row_target = max(1.0, float(target_row_counts[split_name]))
        empty_target = max(1.0, float(target_empty_counts[split_name]))
        row_error = (float(assigned_row_counts[split_name]) - float(target_row_counts[split_name])) / row_target
        empty_error = (
            float(assigned_empty_counts[split_name]) - float(target_empty_counts[split_name])
        ) / empty_target
        score += (row_error * row_error) + (4.0 * empty_error * empty_error)
    return score


def _split_rows_by_group_random(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    val_fraction: float,
    test_fraction: float,
) -> dict[str, list[dict[str, Any]]]:
    _validate_split_fractions(val_fraction=val_fraction, test_fraction=test_fraction)
    if not rows:
        raise ValueError("Cannot split an empty dataset.")

    groups = sorted({str(row["split_group_id"]) for row in rows})
    rng = random.Random(seed)
    rng.shuffle(groups)

    total_groups = len(groups)
    test_groups_count = int(round(total_groups * test_fraction))
    val_groups_count = int(round(total_groups * val_fraction))
    if total_groups >= 3:
        if test_groups_count <= 0:
            test_groups_count = 1
        if val_groups_count <= 0:
            val_groups_count = 1
    if test_groups_count + val_groups_count >= total_groups:
        overflow = (test_groups_count + val_groups_count) - (total_groups - 1)
        while overflow > 0 and test_groups_count > 1:
            test_groups_count -= 1
            overflow -= 1
        while overflow > 0 and val_groups_count > 1:
            val_groups_count -= 1
            overflow -= 1
        if test_groups_count + val_groups_count >= total_groups:
            raise ValueError("Not enough groups to form train/validation/test splits.")

    test_groups = set(groups[:test_groups_count])
    val_groups = set(groups[test_groups_count : test_groups_count + val_groups_count])
    train_groups = set(groups[test_groups_count + val_groups_count :])

    split_rows = {"train": [], "validation": [], "test": []}
    for row in rows:
        group_id = str(row["split_group_id"])
        if group_id in test_groups:
            split_rows["test"].append(row)
        elif group_id in val_groups:
            split_rows["validation"].append(row)
        else:
            split_rows["train"].append(row)

    if not split_rows["train"] or not split_rows["validation"] or not split_rows["test"]:
        raise ValueError("Auto-splitting produced an empty train/validation/test split.")
    return split_rows


def _split_rows_by_group_stratified(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    val_fraction: float,
    test_fraction: float,
) -> dict[str, list[dict[str, Any]]]:
    _validate_split_fractions(val_fraction=val_fraction, test_fraction=test_fraction)
    if not rows:
        raise ValueError("Cannot split an empty dataset.")

    group_infos = _group_rows_by_id(rows)
    if len(group_infos) < 3:
        raise ValueError("Need at least 3 groups to form train/validation/test splits.")

    rng = random.Random(seed)
    rng.shuffle(group_infos)
    group_infos.sort(
        key=lambda info: (
            int(info["empty_rows"] > 0),
            float(info["empty_rows"]) / float(info["row_count"]),
            info["row_count"],
        ),
        reverse=True,
    )

    total_rows = sum(int(info["row_count"]) for info in group_infos)
    total_empty_rows = sum(int(info["empty_rows"]) for info in group_infos)
    global_empty_fraction = total_empty_rows / float(total_rows)
    target_row_counts = _target_split_row_counts(
        total_rows,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )
    target_empty_counts = {
        split_name: float(target_row_counts[split_name]) * global_empty_fraction
        for split_name in ("train", "validation", "test")
    }
    assignments: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    assigned_group_counts = {"train": 0, "validation": 0, "test": 0}
    assigned_row_counts = {"train": 0, "validation": 0, "test": 0}
    assigned_empty_counts = {"train": 0, "validation": 0, "test": 0}

    for index, group_info in enumerate(group_infos):
        remaining_groups = len(group_infos) - index
        missing_splits = [split_name for split_name, count in assigned_group_counts.items() if count <= 0]
        candidate_splits = (
            list(missing_splits)
            if missing_splits and remaining_groups == len(missing_splits)
            else ["train", "validation", "test"]
        )
        best_split: Optional[str] = None
        best_score: Optional[float] = None
        for split_name in candidate_splits:
            trial_group_counts = dict(assigned_group_counts)
            trial_row_counts = dict(assigned_row_counts)
            trial_empty_counts = dict(assigned_empty_counts)
            trial_group_counts[split_name] += 1
            trial_row_counts[split_name] += int(group_info["row_count"])
            trial_empty_counts[split_name] += int(group_info["empty_rows"])
            score = _split_balance_objective(
                assigned_group_counts=trial_group_counts,
                assigned_row_counts=trial_row_counts,
                assigned_empty_counts=trial_empty_counts,
                target_row_counts=target_row_counts,
                target_empty_counts=target_empty_counts,
            )
            if best_score is None or score < best_score:
                best_split = split_name
                best_score = score

        assert best_split is not None
        assignments[best_split].append(group_info)
        assigned_group_counts[best_split] += 1
        assigned_row_counts[best_split] += int(group_info["row_count"])
        assigned_empty_counts[best_split] += int(group_info["empty_rows"])

    improved = True
    while improved:
        improved = False
        current_score = _split_balance_objective(
            assigned_group_counts=assigned_group_counts,
            assigned_row_counts=assigned_row_counts,
            assigned_empty_counts=assigned_empty_counts,
            target_row_counts=target_row_counts,
            target_empty_counts=target_empty_counts,
        )
        for source_split in ("train", "validation", "test"):
            if len(assignments[source_split]) <= 1:
                continue
            for group_info in list(assignments[source_split]):
                for dest_split in ("train", "validation", "test"):
                    if source_split == dest_split:
                        continue
                    trial_group_counts = dict(assigned_group_counts)
                    trial_row_counts = dict(assigned_row_counts)
                    trial_empty_counts = dict(assigned_empty_counts)
                    trial_group_counts[source_split] -= 1
                    trial_group_counts[dest_split] += 1
                    trial_row_counts[source_split] -= int(group_info["row_count"])
                    trial_row_counts[dest_split] += int(group_info["row_count"])
                    trial_empty_counts[source_split] -= int(group_info["empty_rows"])
                    trial_empty_counts[dest_split] += int(group_info["empty_rows"])
                    score = _split_balance_objective(
                        assigned_group_counts=trial_group_counts,
                        assigned_row_counts=trial_row_counts,
                        assigned_empty_counts=trial_empty_counts,
                        target_row_counts=target_row_counts,
                        target_empty_counts=target_empty_counts,
                    )
                    if score + 1e-9 >= current_score:
                        continue
                    assignments[source_split].remove(group_info)
                    assignments[dest_split].append(group_info)
                    assigned_group_counts = trial_group_counts
                    assigned_row_counts = trial_row_counts
                    assigned_empty_counts = trial_empty_counts
                    current_score = score
                    improved = True
                    break
                if improved:
                    break
            if improved:
                break

    split_rows = {"train": [], "validation": [], "test": []}
    for split_name, split_groups in assignments.items():
        for group_info in split_groups:
            split_rows[split_name].extend(group_info["rows"])

    if not split_rows["train"] or not split_rows["validation"] or not split_rows["test"]:
        raise ValueError("Stratified auto-splitting produced an empty train/validation/test split.")
    return split_rows


def _split_rows_by_group(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    val_fraction: float,
    test_fraction: float,
    split_strategy: str,
) -> dict[str, list[dict[str, Any]]]:
    if split_strategy == "random_group":
        return _split_rows_by_group_random(
            rows,
            seed=seed,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
        )
    if split_strategy == "stratified_empty_group":
        return _split_rows_by_group_stratified(
            rows,
            seed=seed,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
        )
    raise ValueError(f"Unsupported split strategy: {split_strategy}")


def _normalized_window(left: int, top: int, right: int, bottom: int, width: int, height: int) -> dict[str, float]:
    return {
        "x_min": clamp(left / float(width)),
        "y_min": clamp(top / float(height)),
        "x_max": clamp(right / float(width)),
        "y_max": clamp(bottom / float(height)),
    }


def _expand_box(box: Mapping[str, Any], *, padding: float) -> dict[str, float]:
    return {
        "x_min": clamp(float(box["x_min"]) - padding),
        "y_min": clamp(float(box["y_min"]) - padding),
        "x_max": clamp(float(box["x_max"]) + padding),
        "y_max": clamp(float(box["y_max"]) + padding),
    }


def _boxes_overlap(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    return not (
        float(a["x_max"]) <= float(b["x_min"])
        or float(a["x_min"]) >= float(b["x_max"])
        or float(a["y_max"]) <= float(b["y_min"])
        or float(a["y_min"]) >= float(b["y_max"])
    )


def _box_center(box: Mapping[str, Any]) -> tuple[float, float]:
    return (
        (float(box["x_min"]) + float(box["x_max"])) / 2.0,
        (float(box["y_min"]) + float(box["y_max"])) / 2.0,
    )


def _choose_background_negative_window(
    boxes: list[Mapping[str, Any]],
    *,
    width: int,
    height: int,
) -> Optional[tuple[int, int, int, int]]:
    if width <= 0 or height <= 0 or not boxes:
        return None

    padded_boxes = [_expand_box(box, padding=BACKGROUND_PADDING) for box in boxes]
    centers = [_box_center(box) for box in boxes]
    best_window: Optional[tuple[int, int, int, int]] = None
    best_score: Optional[float] = None

    for scale in BACKGROUND_SCALES:
        crop_w = max(1, min(width, int(round(width * scale))))
        crop_h = max(1, min(height, int(round(height * scale))))
        x_max_start = max(0, width - crop_w)
        y_max_start = max(0, height - crop_h)
        x_positions = sorted({int(round(x_max_start * rel)) for rel in BACKGROUND_GRID_POSITIONS})
        y_positions = sorted({int(round(y_max_start * rel)) for rel in BACKGROUND_GRID_POSITIONS})

        for top in y_positions:
            for left in x_positions:
                right = min(width, left + crop_w)
                bottom = min(height, top + crop_h)
                if right <= left or bottom <= top:
                    continue
                candidate = _normalized_window(left, top, right, bottom, width, height)
                if any(_boxes_overlap(candidate, padded_box) for padded_box in padded_boxes):
                    continue
                center = _box_center(candidate)
                score = min(
                    ((center[0] - gt_center[0]) ** 2) + ((center[1] - gt_center[1]) ** 2)
                    for gt_center in centers
                )
                if best_score is None or score > best_score:
                    best_score = score
                    best_window = (left, top, right, bottom)

    return best_window


def _synthetic_negative_path(
    *,
    output_dir: Path,
    split_name: str,
    source_image_id: str,
) -> Path:
    return output_dir / "synthetic_negatives" / split_name / f"{source_image_id}__bgneg.jpg"


def _count_empty_rows(rows: Iterable[Mapping[str, Any]]) -> int:
    return sum(1 for row in rows if int(row.get("class_count", 0)) == 0)


def _required_synthetic_negative_count(
    *,
    total_rows: int,
    empty_rows: int,
    target_empty_fraction: float,
) -> int:
    if not (0.0 <= target_empty_fraction < 1.0):
        raise ValueError("--target-empty-fraction must be in [0, 1).")
    if total_rows <= 0 or target_empty_fraction <= 0.0:
        return 0

    needed = 0
    adjusted_total = total_rows
    adjusted_empty = empty_rows
    while adjusted_empty / float(adjusted_total) < target_empty_fraction:
        needed += 1
        adjusted_empty += 1
        adjusted_total += 1
    return needed


def _augment_with_background_negatives(
    rows: list[dict[str, Any]],
    *,
    output_dir: Path,
    split_name: str,
    target_empty_fraction: float,
) -> tuple[list[dict[str, Any]], int]:
    needed = _required_synthetic_negative_count(
        total_rows=len(rows),
        empty_rows=_count_empty_rows(rows),
        target_empty_fraction=target_empty_fraction,
    )
    if needed <= 0:
        return rows, 0

    augmented = list(rows)
    synthetic_count = 0
    positive_rows = sorted(
        (row for row in rows if int(row.get("class_count", 0)) > 0),
        key=lambda row: (str(row["source_base_id"]), str(row["source_image_id"])),
    )
    for row in positive_rows:
        if synthetic_count >= needed:
            break
        if int(row.get("class_count", 0)) <= 0:
            continue
        image_path = Path(str(row["image"]))
        boxes = json.loads(str(row["answer_boxes"]))
        with Image.open(image_path) as image:
            width, height = image.size
            window = _choose_background_negative_window(boxes, width=width, height=height)
            if window is None:
                continue
            left, top, right, bottom = window
            crop = image.crop((left, top, right, bottom))

        out_path = _synthetic_negative_path(
            output_dir=output_dir,
            split_name=split_name,
            source_image_id=str(row["source_image_id"]),
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(out_path, format="JPEG", quality=95)
        augmented.append(
            _build_row(
                image_path=out_path,
                boxes=[],
                source_dataset=str(row["source_dataset"]),
                source_collection=str(row["source_collection"]),
                source_split=str(row["source_split"]),
                source_variant=SOURCE_VARIANT_BACKGROUND_NEGATIVE,
                source_is_synthetic=True,
                source_image_id=f"{row['source_image_id']}__bgneg",
                source_base_id=str(row["source_base_id"]),
                split_group_id=str(row["split_group_id"]),
            )
        )
        synthetic_count += 1

    return augmented, synthetic_count


def _load_source_rows(
    raw_dataset_dir: Path,
    *,
    output_dir: Path,
    tiling: str,
    seed: int,
    val_fraction: float,
    test_fraction: float,
    split_strategy: str,
) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    source_dataset = f"local_coco:{raw_dataset_dir.name}"
    source_collection = f"local_coco:{raw_dataset_dir.name}"

    explicit_source_splits = {
        split_name: _rows_from_coco_split(
            split_dir=raw_dataset_dir / split_name,
            source_dataset=source_dataset,
            source_collection=source_collection,
            source_split=split_name,
        )
        for split_name in SOURCE_SPLITS
        if _has_coco_split(raw_dataset_dir / split_name)
    }
    explicit_source_splits = {
        split_name: _apply_tiling(
            rows,
            output_dir=output_dir,
            tiling=tiling,
        )
        for split_name, rows in explicit_source_splits.items()
    }
    source_split_names = sorted(explicit_source_splits)

    if set(source_split_names) == {"train", "valid", "test"}:
        split_rows = {
            OUTPUT_SPLIT_MAP[source_split]: rows
            for source_split, rows in explicit_source_splits.items()
        }
        return split_rows, source_split_names

    if set(source_split_names) == {"train"}:
        split_rows = _split_rows_by_group(
            explicit_source_splits["train"],
            seed=seed,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            split_strategy=split_strategy,
        )
        return split_rows, source_split_names

    raise ValueError(
        "Expected either a single COCO source split at raw_dataset_dir/train "
        "or explicit train/valid/test COCO directories."
    )


def build_dataset_dict_from_raw_dir(
    raw_dataset_dir: Path,
    *,
    output_dir: Path,
    seed: int,
    val_fraction: float,
    test_fraction: float,
    target_empty_fraction: float,
    split_strategy: str = "stratified_empty_group",
    tiling: str = "none",
) -> tuple[DatasetDict, dict[str, list[dict[str, Any]]], dict[str, int], list[str], dict[str, int]]:
    split_rows, source_split_names = _load_source_rows(
        raw_dataset_dir,
        output_dir=output_dir,
        tiling=tiling,
        seed=seed,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        split_strategy=split_strategy,
    )
    features = _features()
    raw_empty_row_counts = {
        split_name: _count_empty_rows(rows)
        for split_name, rows in split_rows.items()
    }
    synthetic_negative_counts: dict[str, int] = {}
    dataset_splits: dict[str, Dataset] = {}
    for split_name in ("train", "validation", "test"):
        rows, synthetic_count = _augment_with_background_negatives(
            split_rows[split_name],
            output_dir=output_dir,
            split_name=split_name,
            target_empty_fraction=target_empty_fraction,
        )
        split_rows[split_name] = rows
        synthetic_negative_counts[split_name] = synthetic_count
        dataset_splits[split_name] = Dataset.from_list(rows, features=features)

    class_names = discover_class_names(row for rows in split_rows.values() for row in rows)
    if not class_names:
        raise ValueError("No classes discovered in raw dataset.")
    if DEFAULT_CLASS_NAME not in class_names:
        raise ValueError(f"Expected class '{DEFAULT_CLASS_NAME}' in dataset but found {class_names}.")
    if any(class_name != DEFAULT_CLASS_NAME for class_name in class_names):
        raise ValueError(f"Unexpected class names in aerial airport dataset: {class_names}")

    return DatasetDict(dataset_splits), split_rows, synthetic_negative_counts, source_split_names, raw_empty_row_counts


def _build_stats(
    split_rows: dict[str, list[dict[str, Any]]],
    synthetic_negative_counts: Mapping[str, int],
    raw_empty_row_counts: Mapping[str, int],
    target_empty_fraction: float,
    tiling: str,
) -> dict[str, Any]:
    class_names = discover_class_names(row for rows in split_rows.values() for row in rows)
    split_sizes: dict[str, int] = {}
    pre_synthetic_split_sizes: dict[str, int] = {}
    class_counts: dict[str, dict[str, int]] = {}
    empty_row_counts: dict[str, int] = {}
    positive_row_counts: dict[str, int] = {}

    for split_name, rows in split_rows.items():
        split_sizes[split_name] = len(rows)
        pre_synthetic_split_sizes[split_name] = len(rows) - int(synthetic_negative_counts.get(split_name, 0))
        empty_count = 0
        positive_count = 0
        counts = {class_name: 0 for class_name in class_names}
        for row in rows:
            raw_boxes = json.loads(str(row["answer_boxes"]))
            if not raw_boxes:
                empty_count += 1
            else:
                positive_count += 1
            for item in raw_boxes:
                class_name = normalize_class_name(item.get("class_name") or item.get("source_class_name"))
                if class_name in counts:
                    counts[class_name] += 1
        class_counts[split_name] = counts
        empty_row_counts[split_name] = empty_count
        positive_row_counts[split_name] = positive_count

    total_rows = sum(split_sizes.values())
    total_empty_rows = sum(empty_row_counts.values())
    global_empty_row_fraction = (total_empty_rows / float(total_rows)) if total_rows > 0 else 0.0
    empty_row_fractions = {
        split_name: (
            float(empty_row_counts[split_name]) / float(split_sizes[split_name])
            if split_sizes[split_name] > 0
            else 0.0
        )
        for split_name in split_rows
    }
    empty_fraction_abs_delta = {
        split_name: abs(empty_row_fractions[split_name] - global_empty_row_fraction)
        for split_name in split_rows
    }

    return {
        "tiling": tiling,
        "split_sizes": split_sizes,
        "pre_synthetic_split_sizes": pre_synthetic_split_sizes,
        "class_catalog": list(class_names),
        "class_counts": class_counts,
        "empty_row_counts": empty_row_counts,
        "empty_row_fractions": empty_row_fractions,
        "empty_fraction_abs_delta": empty_fraction_abs_delta,
        "empty_fraction_tolerance": EMPTY_FRACTION_TOLERANCE,
        "global_empty_row_fraction": global_empty_row_fraction,
        "raw_empty_row_counts": dict(raw_empty_row_counts),
        "positive_row_counts": positive_row_counts,
        "synthetic_negative_counts": dict(synthetic_negative_counts),
        "target_empty_fraction": float(target_empty_fraction),
    }


def _build_metadata(
    *,
    args: argparse.Namespace,
    raw_dataset_dir: Path,
    source_split_names: list[str],
    stats: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    return {
        "config": args.config,
        "env_file": args.env_file,
        "raw_dataset_dir": str(raw_dataset_dir),
        "source_format": "coco",
        "source_split_names": source_split_names,
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "test_fraction": args.test_fraction,
        "split_strategy": args.split_strategy,
        "tiling": args.tiling,
        "target_empty_fraction": args.target_empty_fraction,
        "output_dir": str(output_dir),
        "push_to_hub": bool(args.push_to_hub),
        "hub_repo_id": args.push_to_hub or "",
        "hub_val_split": args.hub_val_split,
        "hub_post_val_split": args.hub_post_val_split,
        "default_skill": "point",
        "default_point_prompt_style": "class_name",
        "default_reward_metric": "f1",
        "class_catalog": build_class_catalog(list(stats["class_catalog"])),
        "split_sizes": stats["split_sizes"],
        "pre_synthetic_split_sizes": stats["pre_synthetic_split_sizes"],
        "class_counts": stats["class_counts"],
        "empty_row_counts": stats["empty_row_counts"],
        "empty_row_fractions": stats["empty_row_fractions"],
        "empty_fraction_abs_delta": stats["empty_fraction_abs_delta"],
        "empty_fraction_tolerance": stats["empty_fraction_tolerance"],
        "global_empty_row_fraction": stats["global_empty_row_fraction"],
        "raw_empty_row_counts": stats["raw_empty_row_counts"],
        "positive_row_counts": stats["positive_row_counts"],
        "synthetic_negative_counts": stats["synthetic_negative_counts"],
    }


def _prepare_hub_dataset(dataset_dict: DatasetDict, args: argparse.Namespace) -> DatasetDict:
    if args.hub_val_split == "validation" and args.hub_post_val_split == "test":
        return dataset_dict
    return DatasetDict(
        {
            "train": dataset_dict["train"],
            args.hub_val_split: dataset_dict["validation"],
            args.hub_post_val_split: dataset_dict["test"],
        }
    )


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    load_dotenv(args.env_file, override=False)
    if not args.hf_token:
        args.hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    args.push_to_hub = str(args.push_to_hub or "").strip()

    raw_dataset_dir = Path(args.raw_dataset_dir).expanduser().resolve()
    if not raw_dataset_dir.exists():
        raise FileNotFoundError(f"Raw dataset dir not found: {raw_dataset_dir}")
    output_dir = _resolve_dir(args.output_dir, fallback_name="maxs-m87_aerial_airport_point_v2")

    dataset_dict, split_rows, synthetic_negative_counts, source_split_names, raw_empty_row_counts = build_dataset_dict_from_raw_dir(
        raw_dataset_dir,
        output_dir=output_dir,
        seed=args.seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        target_empty_fraction=args.target_empty_fraction,
        split_strategy=args.split_strategy,
        tiling=args.tiling,
    )
    stats = _build_stats(
        split_rows,
        synthetic_negative_counts,
        raw_empty_row_counts,
        args.target_empty_fraction,
        args.tiling,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(output_dir))
    metadata = _build_metadata(
        args=args,
        raw_dataset_dir=raw_dataset_dir,
        source_split_names=source_split_names,
        stats=stats,
        output_dir=output_dir,
    )
    write_json(output_dir / "metadata.json", metadata)
    write_json(output_dir / "stats.json", stats)

    print(
        f"saved normalized dataset to {output_dir} "
        f"(train={len(dataset_dict['train'])}, validation={len(dataset_dict['validation'])}, test={len(dataset_dict['test'])})"
    )

    if args.split_strategy == "stratified_empty_group" and source_split_names == ["train"]:
        max_empty_delta = max(float(value) for value in stats["empty_fraction_abs_delta"].values())
        if max_empty_delta > EMPTY_FRACTION_TOLERANCE:
            print(
                "warning: stratified split empty-row balance missed target tolerance "
                f"({max_empty_delta:.4f} > {EMPTY_FRACTION_TOLERANCE:.4f})"
            )

    if args.push_to_hub:
        if not args.hf_token:
            raise ValueError("HF token required to push to hub.")
        hub_dataset = _prepare_hub_dataset(dataset_dict, args)
        hub_dataset.push_to_hub(args.push_to_hub, token=args.hf_token)
        print(
            "pushed dataset to "
            f"{args.push_to_hub} (train, {args.hub_val_split}, {args.hub_post_val_split})"
        )


if __name__ == "__main__":
    main()
