#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Optional

from datasets import Dataset, DatasetDict, Features, Image as HFImage, Value

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neon_tree import common

try:
    from huggingface_hub import HfApi
except ModuleNotFoundError:  # pragma: no cover
    HfApi = None  # type: ignore[assignment]

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = common.repo_relative("configs", "current", "build_neon_tree_hf_dataset_default.json")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    raw_argv = list(argv) if argv is not None else list(os.sys.argv[1:])
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    pre_args, _ = pre_parser.parse_known_args(raw_argv)
    config_path = common.resolve_config_path(pre_args.config, script_dir=SCRIPT_DIR)
    config = common.load_json_config(config_path, default_path=DEFAULT_CONFIG_PATH)

    parser = argparse.ArgumentParser(description="Build the cleaned NEON tree detect HF dataset.")
    parser.add_argument("--config", default=str(config_path))
    parser.add_argument("--env-file", default=str(common.repo_relative(".env.staging")))
    parser.add_argument(
        "--raw-root",
        default=str(common.repo_relative("raw", "neon_tree_evaluation")),
    )
    parser.add_argument(
        "--output-dir",
        default=str(common.repo_relative("outputs", "dataset", "neon_tree_detect_v1")),
    )
    parser.add_argument("--push-to-hub", default=common.DEFAULT_HF_DATASET_REPO_ID)
    parser.add_argument("--hf-token", default="")
    parser.add_argument("--train-split-name", default="train")
    parser.add_argument("--validation-split-name", default="validation")

    option_to_dest: dict[str, str] = {}
    for action in parser._actions:
        if not action.option_strings:
            continue
        for opt in action.option_strings:
            option_to_dest[opt] = action.dest
    overridden = {option_to_dest[arg] for arg in raw_argv if arg in option_to_dest}
    config_cli_args = common.config_to_cli_args(
        parser,
        config,
        config_path=config_path,
        overridden_dests=overridden,
    )
    args = parser.parse_args(config_cli_args + raw_argv)
    args.config = str(common.resolve_config_path(args.config, script_dir=SCRIPT_DIR))
    return args


def _features() -> Features:
    return Features(
        {
            "image": HFImage(),
            "answer_boxes": Value("string"),
            "source_dataset": Value("string"),
            "source_split": Value("string"),
            "source_image_id": Value("string"),
            "source_base_id": Value("string"),
            "split_group_id": Value("string"),
            "site_code": Value("string"),
            "source_variant": Value("string"),
            "class_count": Value("int32"),
            "tile_id": Value("string"),
            "tile_bounds": Value("string"),
            "is_tiled": Value("bool"),
        }
    )


def _parse_xml_annotation(xml_path: Path) -> tuple[str, list[common.DetectAnnotation]]:
    root = ET.parse(xml_path).getroot()
    filename = str(root.findtext("filename") or "").strip()
    width_text = root.findtext("size/width")
    height_text = root.findtext("size/height")
    if not filename:
        raise ValueError(f"Missing filename in {xml_path}")
    width = int(float(width_text or "0"))
    height = int(float(height_text or "0"))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size in {xml_path}")

    boxes: list[common.DetectAnnotation] = []
    for obj in root.findall("object"):
        try:
            x_min = float(obj.findtext("bndbox/xmin") or "0")
            y_min = float(obj.findtext("bndbox/ymin") or "0")
            x_max = float(obj.findtext("bndbox/xmax") or "0")
            y_max = float(obj.findtext("bndbox/ymax") or "0")
        except ValueError:
            continue
        x0 = common.clamp(x_min / float(width))
        y0 = common.clamp(y_min / float(height))
        x1 = common.clamp(x_max / float(width))
        y1 = common.clamp(y_max / float(height))
        if x1 <= x0 or y1 <= y0:
            continue
        boxes.append(common.DetectAnnotation(x_min=x0, y_min=y0, x_max=x1, y_max=y1))
    return filename, boxes


def _candidate_xml_roots(raw_root: Path) -> list[Path]:
    candidates = [raw_root / "annotations", raw_root / "training", raw_root / "evaluation"]
    return [path for path in candidates if path.exists()]


def _iter_annotation_files(raw_root: Path) -> list[Path]:
    xmls: list[Path] = []
    for root in _candidate_xml_roots(raw_root):
        xmls.extend(sorted(root.rglob("*.xml")))
    return sorted({path.resolve() for path in xmls})


def _build_row(
    *,
    image_path: Path,
    boxes: list[common.DetectAnnotation],
    source_split: str,
) -> dict[str, Any]:
    source_image_id = image_path.stem
    site_code = common.detect_site_code(source_image_id)
    return {
        "image": str(image_path.resolve()),
        "answer_boxes": common.serialize_answer_boxes(boxes),
        "source_dataset": "neon_tree_evaluation",
        "source_split": str(source_split),
        "source_image_id": source_image_id,
        "source_base_id": source_image_id,
        "split_group_id": source_image_id,
        "site_code": site_code,
        "source_variant": "rgb_xml",
        "class_count": len(boxes),
        "tile_id": "",
        "tile_bounds": "",
        "is_tiled": False,
    }


def build_dataset_dict_from_raw_root(
    raw_root: Path,
    *,
    repo_id: str = common.DEFAULT_HF_DATASET_REPO_ID,
) -> tuple[DatasetDict, dict[str, Any], dict[str, Any], str]:
    training_dir = raw_root / "training"
    evaluation_dir = raw_root / "evaluation"
    if not training_dir.exists():
        raise FileNotFoundError(f"Missing training directory: {training_dir}")
    if not evaluation_dir.exists():
        raise FileNotFoundError(f"Missing evaluation directory: {evaluation_dir}")

    train_images = common.find_rgb_images(training_dir)
    validation_images = common.find_rgb_images(evaluation_dir)
    if not train_images:
        raise FileNotFoundError(f"No RGB training images found under {training_dir}")
    if not validation_images:
        raise FileNotFoundError(f"No RGB evaluation images found under {evaluation_dir}")

    train_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    skipped_unmatched = 0
    seen_keys: set[tuple[str, str]] = set()

    for xml_path in _iter_annotation_files(raw_root):
        try:
            filename, boxes = _parse_xml_annotation(xml_path)
        except ValueError:
            continue
        stem = Path(filename).stem.lower()
        train_image = train_images.get(stem)
        validation_image = validation_images.get(stem)
        if train_image is None and validation_image is None:
            skipped_unmatched += 1
            continue
        if train_image is not None:
            key = ("train", train_image.stem)
            if key not in seen_keys:
                train_rows.append(_build_row(image_path=train_image, boxes=boxes, source_split="training"))
                seen_keys.add(key)
        elif validation_image is not None:
            key = ("validation", validation_image.stem)
            if key not in seen_keys:
                validation_rows.append(_build_row(image_path=validation_image, boxes=boxes, source_split="evaluation"))
                seen_keys.add(key)

    dataset_dict = DatasetDict(
        {
            "train": Dataset.from_list(train_rows, features=_features()),
            "validation": Dataset.from_list(validation_rows, features=_features()),
        }
    )
    stats = {
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "train_boxes": sum(int(row["class_count"]) for row in train_rows),
        "validation_boxes": sum(int(row["class_count"]) for row in validation_rows),
        "skipped_unmatched_annotations": int(skipped_unmatched),
    }
    metadata = {
        "config": "",
        "raw_root": str(raw_root),
        "source_dataset": "NeonTreeEvaluation",
        "source_doi": "10.5281/zenodo.5914554",
        "source_repo": "https://github.com/weecology/NeonTreeEvaluation",
        "license": "CC BY 4.0",
        "label_name": "tree",
        "label_note": "Source annotations represent tree crowns in airborne RGB imagery.",
        "splits": {"train": "training", "validation": "evaluation annotated only"},
    }
    readme = common.build_dataset_card(
        repo_id=repo_id,
        metadata=metadata,
        stats=stats,
    )
    return dataset_dict, metadata, stats, readme


def _upload_dataset_card(*, repo_id: str, readme_path: Path, hf_token: str) -> None:
    if HfApi is None:
        return
    api = HfApi(token=hf_token or None)
    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    common.maybe_load_env_file(args.env_file)
    args.hf_token = common.resolve_hf_token(args.hf_token, env_file=args.env_file)

    raw_root = Path(str(args.raw_root)).expanduser().resolve()
    output_dir = Path(str(args.output_dir)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_dict, metadata, stats, readme = build_dataset_dict_from_raw_root(
        raw_root,
        repo_id=str(args.push_to_hub or common.DEFAULT_HF_DATASET_REPO_ID),
    )
    metadata = dict(metadata)
    metadata["config"] = str(args.config)
    metadata["output_dir"] = str(output_dir)
    metadata["push_to_hub"] = bool(str(args.push_to_hub or "").strip())
    metadata["hub_repo_id"] = str(args.push_to_hub or "").strip()

    dataset_dict.save_to_disk(str(output_dir))
    common.write_json(output_dir / "metadata.json", metadata)
    common.write_json(output_dir / "stats.json", stats)
    readme_path = output_dir / "README.md"
    readme_path.write_text(readme, encoding="utf-8")

    print(
        f"saved dataset to {output_dir} "
        f"(train={len(dataset_dict['train'])}, validation={len(dataset_dict['validation'])})"
    )

    if str(args.push_to_hub or "").strip():
        if not args.hf_token:
            raise ValueError("HF token required to push dataset to hub")
        dataset_dict.push_to_hub(str(args.push_to_hub).strip(), token=args.hf_token)
        _upload_dataset_card(repo_id=str(args.push_to_hub).strip(), readme_path=readme_path, hf_token=args.hf_token)
        print(f"pushed dataset to {str(args.push_to_hub).strip()}")


if __name__ == "__main__":
    main()
