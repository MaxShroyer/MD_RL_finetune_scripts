#!/usr/bin/env python3
"""Build a point-first bone fracture dataset from local or Roboflow exports."""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys
import tempfile
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from datasets import Dataset, DatasetDict, Features, Image as HFImage, Value
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bone_fracture.common import (
    DEFAULT_POINT_CLASS_NAME,
    DEFAULT_POINT_CLASS_UID,
    DEFAULT_POINT_HF_DATASET_NAME,
    DEFAULT_POINT_PROMPT_STYLE,
    DEFAULT_REWARD_METRIC,
    DEFAULT_ROBOFLOW_PROJECT,
    DEFAULT_ROBOFLOW_VERSION,
    DEFAULT_ROBOFLOW_WORKSPACE,
    DEFAULT_SKILL,
    build_point_class_catalog,
    class_uid_for_name,
    clamp,
    config_to_cli_args,
    load_json_config,
    normalize_class_name,
    repo_relative,
    resolve_config_path,
    write_json,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = repo_relative("configs", "build_bone_fracture_hf_dataset_default.json")
ROBOFLOW_EXPORT_FORMAT = "voc"
SOURCE_SPLITS = ("train", "valid", "test")
OUTPUT_SPLIT_MAP = {
    "train": "train",
    "valid": "validation",
    "test": "test",
}
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
COCO_SPLIT_CANDIDATES = ("train", "valid", "val", "test")
SOURCE_VARIANT_COCO_EXPORT = "roboflow_coco_export"
SOURCE_VARIANT_VOC_EXPORT = "roboflow_voc_export"


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    raw_argv = list(argv) if argv is not None else list(os.sys.argv[1:])
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    pre_args, _ = pre_parser.parse_known_args(raw_argv)
    config_path = resolve_config_path(pre_args.config, script_dir=SCRIPT_DIR)
    config = load_json_config(config_path, default_path=DEFAULT_CONFIG_PATH)

    parser = argparse.ArgumentParser(description="Build the point-first bone fracture HF dataset.")
    parser.add_argument("--config", default=str(config_path))
    parser.add_argument("--env-file", "--env", default=str(repo_relative(".env.staging")))
    parser.add_argument("--roboflow-workspace", default=DEFAULT_ROBOFLOW_WORKSPACE)
    parser.add_argument("--roboflow-project", default=DEFAULT_ROBOFLOW_PROJECT)
    parser.add_argument("--roboflow-version", type=int, default=DEFAULT_ROBOFLOW_VERSION)
    parser.add_argument("--roboflow-api-key-env-var", default="ROBOFLOW_API_KEY")
    parser.add_argument("--download-dir", default=str(repo_relative("raw_dataset", "bone fracture.coco")))
    parser.add_argument("--output-dir", default=str(repo_relative("outputs", "maxs-m87_bone_fracture_point_v1")))
    parser.add_argument("--push-to-hub", default=DEFAULT_POINT_HF_DATASET_NAME)
    parser.add_argument("--hub-val-split", default="validation")
    parser.add_argument("--hub-post-val-split", default="test")
    parser.add_argument(
        "--include-source-class-names",
        nargs="*",
        default=[],
        help="Only keep boxes whose source label matches one of these names.",
    )
    parser.add_argument(
        "--drop-empty-rows-after-filter",
        action="store_true",
        help="Drop images that have no boxes left after source-label filtering.",
    )
    parser.add_argument(
        "--target-class-name",
        default=DEFAULT_POINT_CLASS_NAME,
        help="Collapsed point target name used for class_name/prompt generation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--single-split-val-fraction",
        type=float,
        default=0.2,
        help="When a local export only has train/, hold out this fraction for validation+test.",
    )
    parser.add_argument(
        "--single-split-test-fraction",
        type=float,
        default=0.5,
        help="When a local export only has train/, allocate this fraction of the holdout pool to test.",
    )
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


def _safe_slug(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def _resolve_dir(path_str: str, *, fallback_name: str) -> Path:
    if path_str:
        path = Path(path_str).expanduser()
        if path.is_absolute():
            return path.resolve()
        parts = path.parts
        if parts and parts[0] == SCRIPT_DIR.name:
            return (REPO_ROOT / path).resolve()
        return path.resolve()
    return repo_relative("outputs", fallback_name).resolve()


def _resolve_api_key(env_var_name: str) -> str:
    value = str(os.environ.get(env_var_name) or "").strip()
    if value:
        return value
    if env_var_name != "ROBOFLOW_API_KEY":
        fallback = str(os.environ.get("ROBOFLOW_API_KEY") or "").strip()
        if fallback:
            return fallback
    raise ValueError(f"Roboflow API key not found in ${env_var_name}.")


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


def _has_coco_annotations(split_dir: Path) -> bool:
    return _find_coco_annotation_file(split_dir) is not None


def _find_coco_export_root(search_root: Path) -> Optional[Path]:
    if not search_root.exists():
        return None
    if any(_has_coco_annotations(search_root / split_name) for split_name in COCO_SPLIT_CANDIDATES):
        return search_root
    for candidate in sorted(search_root.rglob("*")):
        if candidate.is_dir() and any(_has_coco_annotations(candidate / split_name) for split_name in COCO_SPLIT_CANDIDATES):
            return candidate
    return None


def _has_pascal_voc_annotations(split_dir: Path) -> bool:
    return split_dir.exists() and any(path.is_file() for path in split_dir.rglob("*.xml"))


def _looks_like_export_root(root: Path) -> bool:
    return all(_has_pascal_voc_annotations(root / split_name) for split_name in SOURCE_SPLITS)


def _find_export_root(search_root: Path) -> Optional[Path]:
    if not search_root.exists():
        return None
    if _looks_like_export_root(search_root):
        return search_root
    for candidate in sorted(search_root.rglob("*")):
        if candidate.is_dir() and _looks_like_export_root(candidate):
            return candidate
    return None


def _roboflow_export_endpoint(*, workspace: str, project: str, version: int, api_key: str) -> str:
    encoded_key = urllib.parse.quote(api_key, safe="")
    return f"https://api.roboflow.com/{workspace}/{project}/{version}/{ROBOFLOW_EXPORT_FORMAT}?api_key={encoded_key}"


def _download_roboflow_export_via_rest(
    *,
    workspace: str,
    project: str,
    version: int,
    api_key: str,
    download_dir: Path,
) -> Path:
    endpoint = _roboflow_export_endpoint(
        workspace=workspace,
        project=project,
        version=version,
        api_key=api_key,
    )
    print(f"sdk export discovery failed; falling back to Roboflow REST export: {endpoint}")
    with urllib.request.urlopen(endpoint) as response:
        payload = json.loads(response.read().decode("utf-8"))
    export = payload.get("export") if isinstance(payload, dict) else None
    export_link = str((export or {}).get("link") or "").strip()
    if not export_link:
        raise ValueError("Roboflow export endpoint did not return export.link")

    download_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix="roboflow_export_", suffix=".zip", delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)
    try:
        with urllib.request.urlopen(export_link) as response, tmp_path.open("wb") as handle:
            handle.write(response.read())
        with zipfile.ZipFile(tmp_path) as archive:
            archive.extractall(download_dir)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    export_root = _find_export_root(download_dir)
    if export_root is not None:
        return export_root
    raise FileNotFoundError(
        f"REST export downloaded from {export_link} but no train/valid/test Pascal VOC layout was found under {download_dir}."
    )


def _download_roboflow_export(
    *,
    workspace: str,
    project: str,
    version: int,
    api_key_env_var: str,
    download_dir: Path,
) -> Path:
    existing = _find_export_root(download_dir)
    if existing is not None:
        return existing

    try:
        from roboflow import Roboflow
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "roboflow package not installed. Install it with `pip install roboflow` to download the dataset."
        ) from exc

    api_key = _resolve_api_key(api_key_env_var)
    download_dir.mkdir(parents=True, exist_ok=True)
    rf = Roboflow(api_key=api_key)
    version_obj = rf.workspace(workspace).project(project).version(version)
    download_result = version_obj.download(model_format=ROBOFLOW_EXPORT_FORMAT, location=str(download_dir))
    candidate_roots = [download_dir, download_dir.parent, Path.cwd()]
    location = getattr(download_result, "location", None)
    if location:
        location_path = Path(str(location)).expanduser().resolve()
        print(f"roboflow sdk reported download location: {location_path}")
        candidate_roots.insert(0, location_path)
        candidate_roots.insert(1, location_path.parent)
    for candidate in candidate_roots:
        export_root = _find_export_root(candidate)
        if export_root is not None:
            return export_root
    return _download_roboflow_export_via_rest(
        workspace=workspace,
        project=project,
        version=version,
        api_key=api_key,
        download_dir=download_dir,
    )


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
    raise FileNotFoundError(f"Image file referenced by annotations not found under {split_dir}: {file_name}")


def _source_base_id_from_name(file_name: str) -> str:
    stem = Path(file_name).stem
    if ".rf." in stem:
        stem = stem.split(".rf.", 1)[0]
    return stem


def _source_class_key(value: Any) -> str:
    normalized = normalize_class_name(value)
    if not normalized:
        return ""
    return normalized.replace("-", "_").replace(" ", "_").lower()


def _normalize_source_class_filter(values: Optional[Iterable[str]]) -> set[str]:
    allowed: set[str] = set()
    for value in values or []:
        key = _source_class_key(value)
        if key:
            allowed.add(key)
    return allowed


def _filter_source_boxes(
    boxes: Iterable[Mapping[str, Any]],
    *,
    allowed_source_class_keys: Optional[set[str]],
) -> list[dict[str, Any]]:
    if not allowed_source_class_keys:
        return [dict(box) for box in boxes]
    filtered: list[dict[str, Any]] = []
    for box in boxes:
        if _source_class_key(box.get("source_class_name")) in allowed_source_class_keys:
            filtered.append(dict(box))
    return filtered


def _resolve_target_class(target_class_name: str) -> tuple[str, str]:
    class_name = normalize_class_name(target_class_name)
    if not class_name:
        raise ValueError("--target-class-name must not be empty.")
    return class_name, class_uid_for_name(class_name)


def _collapsed_box_payload(
    boxes: Iterable[Mapping[str, Any]],
    *,
    target_class_name: str,
    target_class_uid: str,
) -> str:
    payload = []
    for box in boxes:
        source_class_name = normalize_class_name(box.get("source_class_name"))
        payload.append(
            {
                "x_min": float(box["x_min"]),
                "y_min": float(box["y_min"]),
                "x_max": float(box["x_max"]),
                "y_max": float(box["y_max"]),
                "class_uid": target_class_uid,
                "class_name": target_class_name,
                "source_class_name": source_class_name,
                "attributes": [{"key": "element", "value": target_class_name}],
            }
        )
    return json.dumps(payload, separators=(",", ":"))


def _build_row(
    *,
    image_path: Path,
    boxes: list[dict[str, Any]],
    target_class_name: str,
    target_class_uid: str,
    source_dataset: str,
    source_collection: str,
    source_split: str,
    source_variant: str,
    source_image_id: str,
    source_base_id: str,
) -> dict[str, Any]:
    return {
        "image": str(image_path.resolve()),
        "answer_boxes": _collapsed_box_payload(
            boxes,
            target_class_name=target_class_name,
            target_class_uid=target_class_uid,
        ),
        "source_dataset": source_dataset,
        "source_collection": source_collection,
        "source_variant": source_variant,
        "source_is_synthetic": False,
        "source_split": source_split,
        "source_image_id": source_image_id,
        "source_base_id": source_base_id,
        "split_group_id": f"group:{source_base_id}",
        "class_count": len(boxes),
    }


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


def _parse_xml_annotation(xml_path: Path) -> tuple[int, int, str, list[dict[str, Any]]]:
    root = ET.parse(xml_path).getroot()
    filename = str(root.findtext("filename") or "").strip()
    width_text = root.findtext("size/width")
    height_text = root.findtext("size/height")
    if not filename:
        raise ValueError(f"Missing filename in {xml_path}")
    try:
        width = int(float(width_text or "0"))
        height = int(float(height_text or "0"))
    except ValueError as exc:
        raise ValueError(f"Invalid image size in {xml_path}") from exc
    if width <= 0 or height <= 0:
        raise ValueError(f"Non-positive image size in {xml_path}")

    boxes: list[dict[str, Any]] = []
    for obj in root.findall("object"):
        source_class_name = normalize_class_name(obj.findtext("name"))
        if not source_class_name:
            continue
        try:
            x_min = float(obj.findtext("bndbox/xmin") or "0")
            y_min = float(obj.findtext("bndbox/ymin") or "0")
            x_max = float(obj.findtext("bndbox/xmax") or "0")
            y_max = float(obj.findtext("bndbox/ymax") or "0")
        except ValueError:
            continue
        x0 = clamp(x_min / float(width))
        y0 = clamp(y_min / float(height))
        x1 = clamp(x_max / float(width))
        y1 = clamp(y_max / float(height))
        if x1 <= x0 or y1 <= y0:
            continue
        boxes.append(
            {
                "source_class_name": source_class_name,
                "x_min": x0,
                "y_min": y0,
                "x_max": x1,
                "y_max": y1,
            }
        )
    return width, height, filename, boxes


def _rows_from_voc_split(
    *,
    split_dir: Path,
    allowed_source_class_keys: Optional[set[str]],
    drop_empty_rows_after_filter: bool,
    target_class_name: str,
    target_class_uid: str,
    source_dataset: str,
    source_collection: str,
    source_split: str,
) -> list[dict[str, Any]]:
    if not split_dir.exists():
        raise FileNotFoundError(f"Expected Roboflow split dir not found: {split_dir}")

    rows_by_image_id: dict[str, dict[str, Any]] = {}
    for xml_path in sorted(split_dir.rglob("*.xml")):
        _, _, filename, boxes = _parse_xml_annotation(xml_path)
        boxes = _filter_source_boxes(boxes, allowed_source_class_keys=allowed_source_class_keys)
        if drop_empty_rows_after_filter and not boxes:
            continue
        image_path = _resolve_image_path(split_dir, filename)
        source_image_id = image_path.stem
        source_base_id = _source_base_id_from_name(image_path.name)
        rows_by_image_id[source_image_id] = _build_row(
            image_path=image_path,
            boxes=boxes,
            target_class_name=target_class_name,
            target_class_uid=target_class_uid,
            source_dataset=source_dataset,
            source_collection=source_collection,
            source_split=source_split,
            source_variant=SOURCE_VARIANT_VOC_EXPORT,
            source_image_id=source_image_id,
            source_base_id=source_base_id,
        )
    return list(rows_by_image_id.values())


def _rows_from_coco_split(
    *,
    split_dir: Path,
    allowed_source_class_keys: Optional[set[str]],
    drop_empty_rows_after_filter: bool,
    target_class_name: str,
    target_class_uid: str,
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
    annotations_by_image: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
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
        boxes: list[dict[str, Any]] = []
        for annotation in annotations_by_image.get(image_id, []):
            source_class_name = normalize_class_name(categories_by_id.get(int(annotation.get("category_id", -1)), ""))
            if not source_class_name:
                continue
            normalized_box = _normalized_box_from_coco_bbox(annotation.get("bbox"), width=width, height=height)
            if normalized_box is None:
                continue
            boxes.append(
                {
                    **normalized_box,
                    "source_class_name": source_class_name,
                }
            )
        boxes = _filter_source_boxes(boxes, allowed_source_class_keys=allowed_source_class_keys)
        if drop_empty_rows_after_filter and not boxes:
            continue

        source_image_id = Path(file_name).stem
        source_base_id = _source_base_id_from_name(file_name)
        rows.append(
            _build_row(
                image_path=image_path,
                boxes=boxes,
                target_class_name=target_class_name,
                target_class_uid=target_class_uid,
                source_dataset=source_dataset,
                source_collection=source_collection,
                source_split=source_split,
                source_variant=SOURCE_VARIANT_COCO_EXPORT,
                source_image_id=source_image_id,
                source_base_id=source_base_id,
            )
        )
    return rows


def _make_dataset(rows: list[dict[str, Any]], features: Features) -> Dataset:
    if rows:
        return Dataset.from_list(rows, features=features)
    return Dataset.from_dict({key: [] for key in features.keys()}, features=features)


def _group_buckets(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    group_has_positive: dict[str, bool] = {}
    for row in rows:
        group_id = str(row["split_group_id"])
        has_positive = int(row.get("class_count", 0)) > 0
        group_has_positive[group_id] = group_has_positive.get(group_id, False) or has_positive
    buckets = {"positive": [], "empty": []}
    for group_id, has_positive in sorted(group_has_positive.items()):
        buckets["positive" if has_positive else "empty"].append(group_id)
    return buckets


def _split_one_bucket(
    group_ids: list[str],
    *,
    seed: int,
    val_fraction: float,
    test_fraction: float,
    seed_offset: int,
) -> dict[str, set[str]]:
    rng = random.Random(seed + seed_offset)
    shuffled = list(group_ids)
    rng.shuffle(shuffled)

    total_groups = len(shuffled)
    if total_groups == 0:
        return {"train": set(), "validation": set(), "test": set()}

    holdout_total = int(round(total_groups * float(val_fraction)))
    if total_groups > 2 and holdout_total < 2:
        holdout_total = 2
    holdout_total = min(max(0, holdout_total), max(0, total_groups - 1))

    holdout_groups = shuffled[:holdout_total]
    train_groups = set(shuffled[holdout_total:])
    if not train_groups and holdout_groups:
        train_groups.add(holdout_groups.pop())

    test_total = int(round(len(holdout_groups) * float(test_fraction)))
    if holdout_groups and test_fraction > 0.0 and test_total <= 0:
        test_total = 1
    if len(holdout_groups) > 1:
        test_total = min(test_total, len(holdout_groups) - 1)
    else:
        test_total = 0

    test_groups = set(holdout_groups[:test_total])
    val_groups = set(holdout_groups[test_total:])
    if not val_groups and test_groups:
        moved = sorted(test_groups)[0]
        test_groups.remove(moved)
        val_groups.add(moved)

    return {
        "train": train_groups,
        "validation": val_groups,
        "test": test_groups,
    }


def _build_group_split(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    val_fraction: float,
    test_fraction: float,
) -> dict[str, set[str]]:
    if not rows:
        raise ValueError("No groups available for fallback split generation.")
    if not (0.0 < float(val_fraction) < 1.0):
        raise ValueError("--single-split-val-fraction must be in (0, 1)")
    if not (0.0 <= float(test_fraction) < 1.0):
        raise ValueError("--single-split-test-fraction must be in [0, 1)")

    grouped = _group_buckets(rows)
    split_sets = {
        "train": set(),
        "validation": set(),
        "test": set(),
    }
    for seed_offset, bucket_name in enumerate(("positive", "empty"), start=1):
        bucket_splits = _split_one_bucket(
            grouped[bucket_name],
            seed=seed,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            seed_offset=seed_offset,
        )
        for split_name in split_sets:
            split_sets[split_name].update(bucket_splits[split_name])
    return split_sets


def _split_rows_by_group(
    rows: list[dict[str, Any]],
    groups_by_split: dict[str, set[str]],
) -> dict[str, list[dict[str, Any]]]:
    split_rows: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for row in rows:
        group = str(row["split_group_id"])
        if group in groups_by_split["validation"]:
            split_rows["validation"].append(dict(row))
        elif group in groups_by_split["test"]:
            split_rows["test"].append(dict(row))
        else:
            split_rows["train"].append(dict(row))
    for split_name in split_rows:
        if not split_rows[split_name]:
            raise ValueError(f"Auto-splitting produced an empty {split_name} split.")
    return split_rows


def _build_dataset_dict_from_rows(split_rows: dict[str, list[dict[str, Any]]]) -> tuple[DatasetDict, list[str]]:
    features = _features()
    dataset_splits: dict[str, Dataset] = {}
    class_names: set[str] = set()
    for split_name in ("train", "validation", "test"):
        rows = split_rows.get(split_name, [])
        dataset_splits[split_name] = _make_dataset(rows, features)
        for row in rows:
            raw_boxes = json.loads(str(row["answer_boxes"]))
            for item in raw_boxes:
                class_name = normalize_class_name(item.get("class_name"))
                if class_name:
                    class_names.add(class_name)
    return DatasetDict(dataset_splits), sorted(class_names)


def build_dataset_dict_from_export(
    export_root: Path,
    *,
    include_source_class_names: Optional[Iterable[str]] = None,
    drop_empty_rows_after_filter: bool = False,
    workspace: str,
    project: str,
    target_class_name: str = DEFAULT_POINT_CLASS_NAME,
    version: int,
) -> tuple[DatasetDict, dict[str, list[dict[str, Any]]], list[str]]:
    source_collection = f"roboflow:{workspace}/{project}"
    source_dataset = f"roboflow:{workspace}/{project}:{version}"
    allowed_source_class_keys = _normalize_source_class_filter(include_source_class_names)
    resolved_target_class_name, resolved_target_class_uid = _resolve_target_class(target_class_name)
    split_rows: dict[str, list[dict[str, Any]]] = {}

    for source_split in SOURCE_SPLITS:
        target_split = OUTPUT_SPLIT_MAP[source_split]
        split_rows[target_split] = _rows_from_voc_split(
            split_dir=export_root / source_split,
            allowed_source_class_keys=allowed_source_class_keys,
            drop_empty_rows_after_filter=drop_empty_rows_after_filter,
            target_class_name=resolved_target_class_name,
            target_class_uid=resolved_target_class_uid,
            source_dataset=source_dataset,
            source_collection=source_collection,
            source_split=source_split,
        )

    dataset_dict, class_names = _build_dataset_dict_from_rows(split_rows)
    return dataset_dict, split_rows, class_names


def build_dataset_dict_from_coco_export(
    export_root: Path,
    *,
    include_source_class_names: Optional[Iterable[str]] = None,
    drop_empty_rows_after_filter: bool = False,
    workspace: str,
    project: str,
    version: int,
    seed: int,
    single_split_val_fraction: float,
    target_class_name: str = DEFAULT_POINT_CLASS_NAME,
    single_split_test_fraction: float,
) -> tuple[DatasetDict, dict[str, list[dict[str, Any]]], list[str]]:
    source_collection = f"roboflow:{workspace}/{project}"
    source_dataset = f"roboflow:{workspace}/{project}:{version}"
    allowed_source_class_keys = _normalize_source_class_filter(include_source_class_names)
    resolved_target_class_name, resolved_target_class_uid = _resolve_target_class(target_class_name)
    available_source_splits = [name for name in COCO_SPLIT_CANDIDATES if _has_coco_annotations(export_root / name)]
    if not available_source_splits:
        raise FileNotFoundError(f"No COCO split annotations found under {export_root}")

    if {"train", "valid", "test"}.issubset(set(available_source_splits)) or {"train", "val", "test"}.issubset(
        set(available_source_splits)
    ):
        split_rows: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
        split_aliases = {"train": "train", "valid": "validation", "val": "validation", "test": "test"}
        for source_split in available_source_splits:
            target_split = split_aliases.get(source_split)
            if not target_split:
                continue
            split_rows[target_split].extend(
                _rows_from_coco_split(
                    split_dir=export_root / source_split,
                    allowed_source_class_keys=allowed_source_class_keys,
                    drop_empty_rows_after_filter=drop_empty_rows_after_filter,
                    target_class_name=resolved_target_class_name,
                    target_class_uid=resolved_target_class_uid,
                    source_dataset=source_dataset,
                    source_collection=source_collection,
                    source_split=source_split,
                )
            )
        dataset_dict, class_names = _build_dataset_dict_from_rows(split_rows)
        return dataset_dict, split_rows, class_names

    if available_source_splits != ["train"]:
        raise ValueError(
            f"Unsupported local COCO split layout under {export_root}: {available_source_splits}. "
            "Expected train-only or train+valid/test."
        )

    train_rows = _rows_from_coco_split(
        split_dir=export_root / "train",
        allowed_source_class_keys=allowed_source_class_keys,
        drop_empty_rows_after_filter=drop_empty_rows_after_filter,
        target_class_name=resolved_target_class_name,
        target_class_uid=resolved_target_class_uid,
        source_dataset=source_dataset,
        source_collection=source_collection,
        source_split="train",
    )
    groups_by_split = _build_group_split(
        train_rows,
        seed=seed,
        val_fraction=single_split_val_fraction,
        test_fraction=single_split_test_fraction,
    )
    split_rows = _split_rows_by_group(train_rows, groups_by_split)
    dataset_dict, class_names = _build_dataset_dict_from_rows(split_rows)
    return dataset_dict, split_rows, class_names


def _raw_source_label_catalog(split_rows: Mapping[str, list[dict[str, Any]]]) -> list[str]:
    labels: set[str] = set()
    for rows in split_rows.values():
        for row in rows:
            for item in json.loads(str(row["answer_boxes"])):
                source_class_name = normalize_class_name(item.get("source_class_name"))
                if source_class_name:
                    labels.add(source_class_name)
    return sorted(labels)


def _build_stats(split_rows: dict[str, list[dict[str, Any]]], class_names: list[str]) -> dict[str, Any]:
    split_sizes: dict[str, int] = {}
    class_counts: dict[str, dict[str, int]] = {}
    raw_source_label_counts: dict[str, dict[str, int]] = {}
    empty_row_counts: dict[str, int] = {}
    positive_row_counts: dict[str, int] = {}
    raw_catalog = _raw_source_label_catalog(split_rows)

    for split_name, rows in split_rows.items():
        split_sizes[split_name] = len(rows)
        normalized_counts = {class_name: 0 for class_name in class_names}
        source_counts = {class_name: 0 for class_name in raw_catalog}
        empty_count = 0
        positive_count = 0
        for row in rows:
            raw_boxes = json.loads(str(row["answer_boxes"]))
            if not raw_boxes:
                empty_count += 1
                continue
            positive_count += 1
            for item in raw_boxes:
                class_name = normalize_class_name(item.get("class_name"))
                if class_name in normalized_counts:
                    normalized_counts[class_name] += 1
                source_class_name = normalize_class_name(item.get("source_class_name"))
                if source_class_name in source_counts:
                    source_counts[source_class_name] += 1
        class_counts[split_name] = normalized_counts
        raw_source_label_counts[split_name] = source_counts
        empty_row_counts[split_name] = empty_count
        positive_row_counts[split_name] = positive_count
    return {
        "split_sizes": split_sizes,
        "class_catalog": list(class_names),
        "raw_source_label_catalog": raw_catalog,
        "class_counts": class_counts,
        "raw_source_label_counts": raw_source_label_counts,
        "empty_row_counts": empty_row_counts,
        "positive_row_counts": positive_row_counts,
    }


def _build_metadata(
    *,
    args: argparse.Namespace,
    export_root: Path,
    export_format: str,
    class_names: list[str],
    stats: Mapping[str, Any],
    output_dir: Path,
    target_class_name: str,
    target_class_uid: str,
) -> dict[str, Any]:
    source_split_names = sorted(split_name for split_name, size in stats["split_sizes"].items() if size > 0)
    return {
        "config": args.config,
        "env_file": args.env_file,
        "roboflow_workspace": args.roboflow_workspace,
        "roboflow_project": args.roboflow_project,
        "roboflow_version": args.roboflow_version,
        "download_dir": str(Path(args.download_dir).expanduser().resolve()),
        "export_root": str(export_root),
        "source_format": export_format,
        "source_split_names": source_split_names,
        "seed": args.seed,
        "single_split_val_fraction": args.single_split_val_fraction,
        "single_split_test_fraction": args.single_split_test_fraction,
        "include_source_class_names": list(args.include_source_class_names),
        "drop_empty_rows_after_filter": bool(args.drop_empty_rows_after_filter),
        "output_dir": str(output_dir),
        "push_to_hub": bool(args.push_to_hub),
        "hub_repo_id": args.push_to_hub or "",
        "hub_val_split": args.hub_val_split,
        "hub_post_val_split": args.hub_post_val_split,
        "default_skill": DEFAULT_SKILL,
        "default_point_prompt_style": DEFAULT_POINT_PROMPT_STYLE,
        "default_reward_metric": DEFAULT_REWARD_METRIC,
        "default_target_class_name": target_class_name,
        "default_target_class_uid": target_class_uid,
        "default_target_prompt": target_class_name,
        "class_catalog": build_point_class_catalog(class_names),
        "raw_source_label_catalog": stats["raw_source_label_catalog"],
        "split_sizes": stats["split_sizes"],
        "class_counts": stats["class_counts"],
        "raw_source_label_counts": stats["raw_source_label_counts"],
        "empty_row_counts": stats["empty_row_counts"],
        "positive_row_counts": stats["positive_row_counts"],
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
    args.include_source_class_names = [
        normalize_class_name(value)
        for value in (args.include_source_class_names or [])
        if normalize_class_name(value)
    ]
    target_class_name, target_class_uid = _resolve_target_class(args.target_class_name)
    args.target_class_name = target_class_name

    download_dir = _resolve_dir(
        args.download_dir,
        fallback_name=f"downloads/{_safe_slug(args.roboflow_workspace)}_{_safe_slug(args.roboflow_project)}_v{args.roboflow_version}_{ROBOFLOW_EXPORT_FORMAT}",
    )
    output_dir = _resolve_dir(
        args.output_dir,
        fallback_name=f"{_safe_slug(args.roboflow_workspace)}_{_safe_slug(args.roboflow_project)}_point_v{args.roboflow_version}",
    )

    coco_export_root = _find_coco_export_root(download_dir)
    voc_export_root = _find_export_root(download_dir)
    if coco_export_root is not None:
        print(f"using local COCO export at {coco_export_root}")
        export_root = coco_export_root
        export_format = "coco"
        dataset_dict, split_rows, class_names = build_dataset_dict_from_coco_export(
            export_root,
            include_source_class_names=args.include_source_class_names,
            drop_empty_rows_after_filter=bool(args.drop_empty_rows_after_filter),
            workspace=args.roboflow_workspace,
            project=args.roboflow_project,
            version=args.roboflow_version,
            seed=args.seed,
            single_split_val_fraction=args.single_split_val_fraction,
            target_class_name=target_class_name,
            single_split_test_fraction=args.single_split_test_fraction,
        )
    else:
        if voc_export_root is None:
            export_root = _download_roboflow_export(
                workspace=args.roboflow_workspace,
                project=args.roboflow_project,
                version=args.roboflow_version,
                api_key_env_var=args.roboflow_api_key_env_var,
                download_dir=download_dir,
            )
        else:
            export_root = voc_export_root
        export_format = "voc"
        dataset_dict, split_rows, class_names = build_dataset_dict_from_export(
            export_root,
            include_source_class_names=args.include_source_class_names,
            drop_empty_rows_after_filter=bool(args.drop_empty_rows_after_filter),
            workspace=args.roboflow_workspace,
            project=args.roboflow_project,
            target_class_name=target_class_name,
            version=args.roboflow_version,
        )

    if class_names != [target_class_name]:
        raise ValueError(
            f"Expected collapsed class catalog ['{target_class_name}'], but found {class_names!r}"
        )
    for split_name in ("train", "validation", "test"):
        if len(dataset_dict[split_name]) <= 0:
            raise ValueError(
                f"Filtered dataset produced an empty {split_name} split. "
                "Adjust the source-label filter or drop-empty setting."
            )

    stats = _build_stats(split_rows, class_names)

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(output_dir))
    metadata = _build_metadata(
        args=args,
        export_root=export_root,
        export_format=export_format,
        class_names=class_names,
        stats=stats,
        output_dir=output_dir,
        target_class_name=target_class_name,
        target_class_uid=target_class_uid,
    )
    write_json(output_dir / "metadata.json", metadata)
    write_json(output_dir / "stats.json", stats)

    print(
        f"saved normalized dataset to {output_dir} "
        f"(train={len(dataset_dict['train'])}, validation={len(dataset_dict['validation'])}, test={len(dataset_dict['test'])})"
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
