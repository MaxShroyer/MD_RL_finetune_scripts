#!/usr/bin/env python3
"""Build local detect and query datasets from LouisChen15/ConstructionSite."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from datasets import Dataset, DatasetDict, Features, Image as HFImage, Value, load_dataset

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover
    def load_dotenv(*_args: Any, **_kwargs: Any) -> bool:
        return False

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from construction_site.common import (
    CAPTION_QUESTION,
    CAPTION_TASK_TYPE,
    DEFAULT_CAPTION_OUTPUT_DIR,
    DEFAULT_DATASET_NAME,
    DEFAULT_DETECT_OUTPUT_DIR,
    DEFAULT_RULE_VQA_OUTPUT_DIR,
    DEFAULT_SHARED_IMAGE_DIR,
    DETECT_CLASS_CATALOG,
    LOCAL_VAL_SPLIT,
    RULE_VQA_QUESTION,
    RULE_VQA_TASK_TYPE,
    SOURCE_TEST_SPLIT,
    SOURCE_TRAIN_SPLIT,
    build_detect_boxes,
    config_to_cli_args,
    extract_caption_attribute_tags,
    extract_caption_object_tags,
    extract_rule_reasons,
    load_json_config,
    relative_path,
    repo_relative,
    resolve_config_path,
    serialize_answer_boxes,
    write_json,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = repo_relative("configs", "build_construction_site_hf_dataset_default.json")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    raw_argv = list(argv) if argv is not None else list(os.sys.argv[1:])
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    pre_args, _ = pre_parser.parse_known_args(raw_argv)
    config_path = resolve_config_path(pre_args.config, script_dir=SCRIPT_DIR)
    config = load_json_config(config_path, default_path=DEFAULT_CONFIG_PATH)

    parser = argparse.ArgumentParser(description="Build ConstructionSite local detect and query datasets.")
    parser.add_argument("--config", default=str(config_path))
    parser.add_argument("--env-file", default=str(repo_relative(".env")))
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"))
    parser.add_argument("--hf-cache-dir", default="")
    parser.add_argument("--detect-output-dir", default=str(DEFAULT_DETECT_OUTPUT_DIR))
    parser.add_argument("--caption-output-dir", default=str(DEFAULT_CAPTION_OUTPUT_DIR))
    parser.add_argument("--rule-vqa-output-dir", default=str(DEFAULT_RULE_VQA_OUTPUT_DIR))
    parser.add_argument("--shared-image-dir", default=str(DEFAULT_SHARED_IMAGE_DIR))
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-detect", action="store_true")
    parser.add_argument("--skip-caption", action="store_true")
    parser.add_argument("--skip-rule-vqa", action="store_true")

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
            "image_caption": Value("string"),
            "illumination": Value("string"),
            "camera_distance": Value("string"),
            "view": Value("string"),
            "quality_of_info": Value("string"),
            "rule_reasons_json": Value("string"),
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


def _resolved_output(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path.cwd() / path).resolve()


def _load_source_rows(
    *,
    dataset_name: str,
    split_name: str,
    hf_token: str,
    hf_cache_dir: str,
) -> list[dict[str, Any]]:
    kwargs: dict[str, Any] = {"split": split_name}
    if hf_token:
        kwargs["token"] = hf_token
    if str(hf_cache_dir).strip():
        kwargs["cache_dir"] = hf_cache_dir
    try:
        dataset = load_dataset(dataset_name, **kwargs)
    except TypeError:
        token = kwargs.pop("token", None)
        if token:
            kwargs["use_auth_token"] = token
        dataset = load_dataset(dataset_name, **kwargs)
    rows: list[dict[str, Any]] = []
    for row in dataset:
        if isinstance(row, dict):
            rows.append(dict(row))
    if not rows:
        raise ValueError(f"split={split_name} from {dataset_name} yielded no rows")
    return rows


def _iter_source_rows(
    *,
    dataset_name: str,
    split_name: str,
    hf_token: str,
    hf_cache_dir: str,
):
    kwargs: dict[str, Any] = {"split": split_name, "streaming": True}
    if hf_token:
        kwargs["token"] = hf_token
    if str(hf_cache_dir).strip():
        kwargs["cache_dir"] = hf_cache_dir
    try:
        dataset = load_dataset(dataset_name, **kwargs)
    except TypeError:
        token = kwargs.pop("token", None)
        if token:
            kwargs["use_auth_token"] = token
        dataset = load_dataset(dataset_name, **kwargs)
    for row in dataset:
        if isinstance(row, dict):
            yield dict(row)


def _split_train_rows(rows: list[dict[str, Any]], *, seed: int, val_fraction: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    val_count = int(round(len(indices) * val_fraction))
    if len(indices) > 1:
        val_count = max(1, min(len(indices) - 1, val_count))
    else:
        val_count = 0
    val_indices = set(indices[:val_count])
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        if idx in val_indices:
            val_rows.append(row)
        else:
            train_rows.append(row)
    return train_rows, val_rows


def _image_id(row: dict[str, Any], *, fallback_index: int) -> str:
    image_id = str(row.get("image_id") or "").strip()
    if image_id:
        return image_id
    return f"row_{fallback_index:06d}"


def _persist_split_images(
    *,
    rows: list[dict[str, Any]],
    split_name: str,
    shared_image_dir: Path,
) -> dict[str, Path]:
    image_paths: dict[str, Path] = {}
    split_dir = shared_image_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows):
        image = row.get("image")
        if image is None:
            raise ValueError(f"split={split_name} row={index} missing image")
        image_id = _image_id(row, fallback_index=index)
        output_path = split_dir / f"{image_id}.png"
        if not output_path.exists():
            image.convert("RGB").save(output_path, format="PNG")
        image_paths[image_id] = output_path.resolve()
    return image_paths


def _persist_stream_image(
    *,
    row: dict[str, Any],
    split_name: str,
    shared_image_dir: Path,
    fallback_index: int,
) -> tuple[str, Path]:
    image = row.get("image")
    if image is None:
        raise ValueError(f"split={split_name} row={fallback_index} missing image")
    image_id = _image_id(row, fallback_index=fallback_index)
    split_dir = shared_image_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    output_path = split_dir / f"{image_id}.png"
    if not output_path.exists():
        image.convert("RGB").save(output_path, format="PNG")
    return image_id, output_path.resolve()


def _detect_rows(
    *,
    rows: list[dict[str, Any]],
    split_name: str,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        image_id = _image_id(row, fallback_index=index)
        boxes = build_detect_boxes(row)
        normalized.append(
            {
                "image": row["image"].convert("RGB"),
                "answer_boxes": serialize_answer_boxes(boxes),
                "image_caption": str(row.get("image_caption") or ""),
                "illumination": str(row.get("illumination") or ""),
                "camera_distance": str(row.get("camera_distance") or ""),
                "view": str(row.get("view") or ""),
                "quality_of_info": str(row.get("quality_of_info") or ""),
                "rule_reasons_json": json.dumps(extract_rule_reasons(row), sort_keys=True),
                "source_dataset": DEFAULT_DATASET_NAME,
                "source_collection": "construction_site_10k",
                "source_variant": "hf_raw",
                "source_is_synthetic": False,
                "source_split": split_name,
                "source_image_id": image_id,
                "source_base_id": image_id,
                "split_group_id": image_id,
                "class_count": len({box["class_name"] for box in boxes}),
            }
        )
    return normalized


def _caption_jsonl_rows(
    *,
    rows: list[dict[str, Any]],
    split_name: str,
    dataset_dir: Path,
    image_paths: dict[str, Path],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        out.append(
            _caption_jsonl_row(
                row=row,
                split_name=split_name,
                dataset_dir=dataset_dir,
                image_path=image_paths[_image_id(row, fallback_index=index)],
                fallback_index=index,
            )
        )
    return out


def _caption_jsonl_row(
    *,
    row: dict[str, Any],
    split_name: str,
    dataset_dir: Path,
    image_path: Path,
    fallback_index: int,
) -> dict[str, Any]:
    image_id = _image_id(row, fallback_index=fallback_index)
    return {
        "row_id": f"{split_name}_{image_id}",
        "split": split_name,
        "task_type": CAPTION_TASK_TYPE,
        "question": CAPTION_QUESTION,
        "final_answer_json": json.dumps({"caption": str(row.get("image_caption") or "")}, ensure_ascii=False),
        "reference_caption": str(row.get("image_caption") or ""),
        "attribute_tags_json": json.dumps(extract_caption_attribute_tags(row), ensure_ascii=False),
        "object_tags_json": json.dumps(extract_caption_object_tags(row), ensure_ascii=False),
        "image_path": relative_path(dataset_dir, image_path),
    }


def _rule_vqa_jsonl_rows(
    *,
    rows: list[dict[str, Any]],
    split_name: str,
    dataset_dir: Path,
    image_paths: dict[str, Path],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        out.append(
            _rule_vqa_jsonl_row(
                row=row,
                split_name=split_name,
                dataset_dir=dataset_dir,
                image_path=image_paths[_image_id(row, fallback_index=index)],
                fallback_index=index,
            )
        )
    return out


def _rule_vqa_jsonl_row(
    *,
    row: dict[str, Any],
    split_name: str,
    dataset_dir: Path,
    image_path: Path,
    fallback_index: int,
) -> dict[str, Any]:
    image_id = _image_id(row, fallback_index=fallback_index)
    rule_reasons = extract_rule_reasons(row)
    violated_rules = sorted(rule_reasons)
    return {
        "row_id": f"{split_name}_{image_id}",
        "split": split_name,
        "task_type": RULE_VQA_TASK_TYPE,
        "question": RULE_VQA_QUESTION,
        "final_answer_json": json.dumps(
            {
                "violated_rules": violated_rules,
                "reasons": {str(rule_id): text for rule_id, text in rule_reasons.items()},
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        "reference_rule_count": len(violated_rules),
        "image_path": relative_path(dataset_dir, image_path),
    }


def _write_jsonl_dataset(
    *,
    dataset_dir: Path,
    rows_by_split: dict[str, list[dict[str, Any]]],
    metadata: dict[str, Any],
    stats: dict[str, Any],
) -> None:
    jsonl_dir = dataset_dir / "jsonl"
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    for split_name, rows in rows_by_split.items():
        output_path = jsonl_dir / f"{split_name}.jsonl"
        output_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
    write_json(dataset_dir / "metadata.json", metadata)
    write_json(dataset_dir / "stats.json", stats)


def _assign_local_split(*, seed: int, image_id: str, val_fraction: float) -> str:
    digest = hashlib.sha1(f"{seed}:{image_id}".encode("utf-8")).hexdigest()
    bucket = int(digest[:12], 16) / float(16 ** 12)
    return LOCAL_VAL_SPLIT if bucket < float(val_fraction) else SOURCE_TRAIN_SPLIT


def _build_query_jsonl_datasets_streaming(
    *,
    dataset_name: str,
    hf_token: str,
    hf_cache_dir: str,
    caption_output_dir: Path,
    rule_vqa_output_dir: Path,
    shared_image_dir: Path,
    val_fraction: float,
    seed: int,
    build_caption: bool,
    build_rule_vqa: bool,
) -> None:
    shared_image_dir.mkdir(parents=True, exist_ok=True)
    split_names = (SOURCE_TRAIN_SPLIT, LOCAL_VAL_SPLIT, SOURCE_TEST_SPLIT)

    caption_handles: dict[str, Any] = {}
    rule_handles: dict[str, Any] = {}
    caption_counts = {split_name: 0 for split_name in split_names}
    rule_counts = {split_name: 0 for split_name in split_names}
    rule_positive_counts = {split_name: 0 for split_name in split_names}

    if build_caption:
        caption_jsonl_dir = caption_output_dir / "jsonl"
        caption_jsonl_dir.mkdir(parents=True, exist_ok=True)
        caption_handles = {
            split_name: (caption_jsonl_dir / f"{split_name}.jsonl").open("w", encoding="utf-8")
            for split_name in split_names
        }
    if build_rule_vqa:
        rule_jsonl_dir = rule_vqa_output_dir / "jsonl"
        rule_jsonl_dir.mkdir(parents=True, exist_ok=True)
        rule_handles = {
            split_name: (rule_jsonl_dir / f"{split_name}.jsonl").open("w", encoding="utf-8")
            for split_name in split_names
        }

    try:
        for source_split_name in (SOURCE_TRAIN_SPLIT, SOURCE_TEST_SPLIT):
            for index, row in enumerate(
                _iter_source_rows(
                    dataset_name=dataset_name,
                    split_name=source_split_name,
                    hf_token=hf_token,
                    hf_cache_dir=hf_cache_dir,
                )
            ):
                image_id, image_path = _persist_stream_image(
                    row=row,
                    split_name=SOURCE_TEST_SPLIT
                    if source_split_name == SOURCE_TEST_SPLIT
                    else _assign_local_split(seed=seed, image_id=_image_id(row, fallback_index=index), val_fraction=val_fraction),
                    shared_image_dir=shared_image_dir,
                    fallback_index=index,
                )
                local_split = SOURCE_TEST_SPLIT if source_split_name == SOURCE_TEST_SPLIT else _assign_local_split(
                    seed=seed,
                    image_id=image_id,
                    val_fraction=val_fraction,
                )
                if build_caption:
                    caption_row = _caption_jsonl_row(
                        row=row,
                        split_name=local_split,
                        dataset_dir=caption_output_dir,
                        image_path=image_path,
                        fallback_index=index,
                    )
                    caption_handles[local_split].write(json.dumps(caption_row, ensure_ascii=False) + "\n")
                    caption_counts[local_split] += 1
                if build_rule_vqa:
                    rule_row = _rule_vqa_jsonl_row(
                        row=row,
                        split_name=local_split,
                        dataset_dir=rule_vqa_output_dir,
                        image_path=image_path,
                        fallback_index=index,
                    )
                    rule_handles[local_split].write(json.dumps(rule_row, ensure_ascii=False) + "\n")
                    rule_counts[local_split] += 1
                    rule_positive_counts[local_split] += int(bool(json.loads(rule_row["final_answer_json"]).get("violated_rules")))
    finally:
        for handle in caption_handles.values():
            handle.close()
        for handle in rule_handles.values():
            handle.close()

    if build_caption:
        write_json(
            caption_output_dir / "metadata.json",
            {
                "dataset_name": dataset_name,
                "task_type": CAPTION_TASK_TYPE,
                "question": CAPTION_QUESTION,
                "shared_image_dir": str(shared_image_dir),
                "json_schema": {"caption": "..."},
                "val_fraction": float(val_fraction),
                "seed": int(seed),
            },
        )
        write_json(
            caption_output_dir / "stats.json",
            {
                "dataset_name": dataset_name,
                "task_type": CAPTION_TASK_TYPE,
                "splits": {split_name: {"rows": count} for split_name, count in caption_counts.items()},
            },
        )
    if build_rule_vqa:
        write_json(
            rule_vqa_output_dir / "metadata.json",
            {
                "dataset_name": dataset_name,
                "task_type": RULE_VQA_TASK_TYPE,
                "question": RULE_VQA_QUESTION,
                "shared_image_dir": str(shared_image_dir),
                "json_schema": {"violated_rules": [1], "reasons": {"1": "..."}},
                "val_fraction": float(val_fraction),
                "seed": int(seed),
            },
        )
        write_json(
            rule_vqa_output_dir / "stats.json",
            {
                "dataset_name": dataset_name,
                "task_type": RULE_VQA_TASK_TYPE,
                "splits": {
                    split_name: {
                        "rows": rule_counts[split_name],
                        "positive_rows": rule_positive_counts[split_name],
                        "negative_rows": rule_counts[split_name] - rule_positive_counts[split_name],
                    }
                    for split_name in split_names
                },
            },
        )


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    load_dotenv(args.env_file, override=False)
    if not args.hf_token:
        args.hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or ""
    if not (0.0 < float(args.val_fraction) < 1.0):
        raise ValueError("--val-fraction must be in (0, 1)")
    build_detect = not bool(args.skip_detect)
    build_caption = not bool(args.skip_caption)
    build_rule_vqa = not bool(args.skip_rule_vqa)
    if not any((build_detect, build_caption, build_rule_vqa)):
        raise ValueError("Nothing to build: all outputs were skipped")

    detect_output_dir = _resolved_output(args.detect_output_dir)
    caption_output_dir = _resolved_output(args.caption_output_dir)
    rule_vqa_output_dir = _resolved_output(args.rule_vqa_output_dir)
    shared_image_dir = _resolved_output(args.shared_image_dir)

    if not build_detect:
        _build_query_jsonl_datasets_streaming(
            dataset_name=args.dataset_name,
            hf_token=str(args.hf_token or ""),
            hf_cache_dir=str(args.hf_cache_dir or ""),
            caption_output_dir=caption_output_dir,
            rule_vqa_output_dir=rule_vqa_output_dir,
            shared_image_dir=shared_image_dir,
            val_fraction=float(args.val_fraction),
            seed=int(args.seed),
            build_caption=build_caption,
            build_rule_vqa=build_rule_vqa,
        )
        if build_caption:
            print(f"saved caption dataset: {caption_output_dir}")
        if build_rule_vqa:
            print(f"saved rule-vqa dataset: {rule_vqa_output_dir}")
        return

    source_train_rows = _load_source_rows(
        dataset_name=args.dataset_name,
        split_name=SOURCE_TRAIN_SPLIT,
        hf_token=str(args.hf_token or ""),
        hf_cache_dir=str(args.hf_cache_dir or ""),
    )
    source_test_rows = _load_source_rows(
        dataset_name=args.dataset_name,
        split_name=SOURCE_TEST_SPLIT,
        hf_token=str(args.hf_token or ""),
        hf_cache_dir=str(args.hf_cache_dir or ""),
    )

    train_rows, val_rows = _split_train_rows(
        source_train_rows,
        seed=int(args.seed),
        val_fraction=float(args.val_fraction),
    )

    source_rows_by_split = {
        SOURCE_TRAIN_SPLIT: train_rows,
        LOCAL_VAL_SPLIT: val_rows,
        SOURCE_TEST_SPLIT: source_test_rows,
    }

    shared_image_dir.mkdir(parents=True, exist_ok=True)
    image_paths_by_split: dict[str, dict[str, Path]] = {
        split_name: _persist_split_images(rows=rows, split_name=split_name, shared_image_dir=shared_image_dir)
        for split_name, rows in source_rows_by_split.items()
    }

    if build_detect:
        detect_splits = {
            split_name: Dataset.from_list(_detect_rows(rows=rows, split_name=split_name), features=_features())
            for split_name, rows in source_rows_by_split.items()
        }
        detect_dataset = DatasetDict(detect_splits)
        detect_output_dir.parent.mkdir(parents=True, exist_ok=True)
        detect_dataset.save_to_disk(str(detect_output_dir))

        detect_stats = {
            "dataset_name": args.dataset_name,
            "class_catalog": [entry["class_name"] for entry in DETECT_CLASS_CATALOG],
            "splits": {},
        }
        for split_name, rows in source_rows_by_split.items():
            box_counter: Counter[str] = Counter()
            row_count_with_boxes = 0
            for row in rows:
                boxes = build_detect_boxes(row)
                if boxes:
                    row_count_with_boxes += 1
                for box in boxes:
                    box_counter[str(box["class_name"])] += 1
            detect_stats["splits"][split_name] = {
                "rows": len(rows),
                "rows_with_boxes": row_count_with_boxes,
                "rows_without_boxes": len(rows) - row_count_with_boxes,
                "box_counts": dict(sorted(box_counter.items())),
            }
        detect_metadata = {
            "dataset_name": args.dataset_name,
            "source_train_split": SOURCE_TRAIN_SPLIT,
            "source_test_split": SOURCE_TEST_SPLIT,
            "local_val_split": LOCAL_VAL_SPLIT,
            "val_fraction": float(args.val_fraction),
            "seed": int(args.seed),
            "class_catalog": DETECT_CLASS_CATALOG,
            "shared_image_dir": str(shared_image_dir),
        }
        write_json(detect_output_dir / "metadata.json", detect_metadata)
        write_json(detect_output_dir / "stats.json", detect_stats)

    if build_caption:
        caption_rows_by_split = {
            split_name: _caption_jsonl_rows(
                rows=rows,
                split_name=split_name,
                dataset_dir=caption_output_dir,
                image_paths=image_paths_by_split[split_name],
            )
            for split_name, rows in source_rows_by_split.items()
        }
        caption_stats = {
            "dataset_name": args.dataset_name,
            "task_type": CAPTION_TASK_TYPE,
            "splits": {split_name: {"rows": len(rows)} for split_name, rows in caption_rows_by_split.items()},
        }
        caption_metadata = {
            "dataset_name": args.dataset_name,
            "task_type": CAPTION_TASK_TYPE,
            "question": CAPTION_QUESTION,
            "shared_image_dir": str(shared_image_dir),
            "json_schema": {"caption": "..."},
        }
        _write_jsonl_dataset(
            dataset_dir=caption_output_dir,
            rows_by_split=caption_rows_by_split,
            metadata=caption_metadata,
            stats=caption_stats,
        )

    if build_rule_vqa:
        rule_rows_by_split = {
            split_name: _rule_vqa_jsonl_rows(
                rows=rows,
                split_name=split_name,
                dataset_dir=rule_vqa_output_dir,
                image_paths=image_paths_by_split[split_name],
            )
            for split_name, rows in source_rows_by_split.items()
        }
        rule_positive_counts = {
            split_name: sum(1 for row in rows if json.loads(row["final_answer_json"]).get("violated_rules"))
            for split_name, rows in rule_rows_by_split.items()
        }
        rule_stats = {
            "dataset_name": args.dataset_name,
            "task_type": RULE_VQA_TASK_TYPE,
            "splits": {
                split_name: {
                    "rows": len(rows),
                    "positive_rows": rule_positive_counts[split_name],
                    "negative_rows": len(rows) - rule_positive_counts[split_name],
                }
                for split_name, rows in rule_rows_by_split.items()
            },
        }
        rule_metadata = {
            "dataset_name": args.dataset_name,
            "task_type": RULE_VQA_TASK_TYPE,
            "question": RULE_VQA_QUESTION,
            "shared_image_dir": str(shared_image_dir),
            "json_schema": {"violated_rules": [1], "reasons": {"1": "..."}},
        }
        _write_jsonl_dataset(
            dataset_dir=rule_vqa_output_dir,
            rows_by_split=rule_rows_by_split,
            metadata=rule_metadata,
            stats=rule_stats,
        )

    if build_detect:
        print(f"saved detect dataset: {detect_output_dir}")
    if build_caption:
        print(f"saved caption dataset: {caption_output_dir}")
    if build_rule_vqa:
        print(f"saved rule-vqa dataset: {rule_vqa_output_dir}")


if __name__ == "__main__":
    main()
