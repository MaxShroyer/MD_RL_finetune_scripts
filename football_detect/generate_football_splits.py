#!/usr/bin/env python3
"""Generate deterministic train/val/post_val splits for football detection."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from datasets import Dataset, DatasetDict, load_dataset
from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from football_detect.common import (
    DEFAULT_DATASET_NAME,
    DEFAULT_DATASET_REVISION,
    NOTE_BUCKETS,
    allocate_post_val_counts,
    allocate_val_counts,
    build_class_catalog,
    build_split_stats,
    config_to_cli_args,
    default_prompt_for_class,
    load_json_config,
    parse_box_element_annotations,
    parse_note_bucket,
    repo_relative,
    resolve_config_path,
    write_json,
)

DEFAULT_CONFIG_PATH = repo_relative("configs", "generate_football_splits_v2.json")


def _resolve_output_dir(output_dir: str, dataset_name: str) -> Path:
    if output_dir:
        return Path(output_dir).expanduser().resolve()
    safe_name = dataset_name.replace("/", "_")
    return repo_relative("outputs", f"{safe_name}_splits").resolve()


def _build_hub_splits(
    splits: DatasetDict,
    val_split_name: str,
    post_val_split_name: str,
) -> DatasetDict:
    if not val_split_name or not post_val_split_name:
        raise ValueError("Hub split names must be non-empty strings.")
    if val_split_name == "train" or post_val_split_name == "train":
        raise ValueError("Hub split names must differ from 'train'.")
    if val_split_name == post_val_split_name:
        raise ValueError("Hub split names must be distinct.")
    return DatasetDict(
        {
            "train": splits["train"],
            val_split_name: splits["val"],
            post_val_split_name: splits["post_val"],
        }
    )


def _dataset_from_rows(rows: Sequence[Mapping[str, Any]], *, features: Any) -> Dataset:
    items = [dict(row) for row in rows]
    if features is not None:
        return Dataset.from_list(items, features=features)
    return Dataset.from_list(items)


def _resolve_source_row_id(row: Mapping[str, Any], *, split_name: str, row_index: int) -> str:
    candidates = (
        row.get("source_row_id"),
        row.get("row_id"),
        row.get("id"),
        row.get("image_id"),
        row.get("uuid"),
    )
    for candidate in candidates:
        text = str(candidate or "").strip()
        if text:
            return text
    return f"{split_name}-{row_index}"


def _annotation_answer_box_payload(annotation: Any) -> str:
    payload = [
        {
            "x_min": float(annotation.box.x_min),
            "y_min": float(annotation.box.y_min),
            "x_max": float(annotation.box.x_max),
            "y_max": float(annotation.box.y_max),
            "attributes": [{"key": "element", "value": annotation.class_name}],
        }
    ]
    return json.dumps(payload)


def _flatten_split_rows(rows: Sequence[Mapping[str, Any]], *, split_name: str) -> list[dict[str, Any]]:
    flattened_rows: list[dict[str, Any]] = []
    for row_index, raw_row in enumerate(rows):
        row = dict(raw_row)
        image = row.get("image")
        if image is None:
            continue
        width, height = image.size
        annotations = parse_box_element_annotations(row.get("answer_boxes"), width=width, height=height)
        if not annotations:
            continue
        source_row_id = _resolve_source_row_id(row, split_name=split_name, row_index=row_index)
        for annotation in annotations:
            flattened = dict(row)
            flattened["class_name"] = annotation.class_name
            flattened["prompt"] = default_prompt_for_class(annotation.class_name)
            flattened["source_row_id"] = source_row_id
            flattened["source_box_index"] = int(annotation.source_box_index)
            flattened["source_element_index"] = int(annotation.source_element_index)
            flattened["task_schema"] = "per_box_element"
            flattened["answer_boxes"] = _annotation_answer_box_payload(annotation)
            flattened_rows.append(flattened)
    return flattened_rows


def build_splits_from_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    val_fraction: float,
    holdout_count: int,
    flatten_by_box_element: bool = False,
    features: Any = None,
) -> tuple[DatasetDict, dict[str, Any], dict[str, list[dict[str, Any]]]]:
    if not (0.0 < float(val_fraction) < 1.0):
        raise ValueError("val_fraction must be in (0, 1).")
    if holdout_count < 0:
        raise ValueError("holdout_count must be >= 0.")
    if not rows:
        raise ValueError("Dataset is empty.")

    rows_by_note: dict[str, list[dict[str, Any]]] = {note: [] for note in NOTE_BUCKETS}
    for raw_row in rows:
        row = dict(raw_row)
        note = parse_note_bucket(row.get("notes"))
        rows_by_note[note].append(row)

    note_counts = {note: len(rows_by_note[note]) for note in NOTE_BUCKETS}
    val_counts = allocate_val_counts(note_counts, val_fraction=val_fraction)
    total_val = sum(val_counts.values())
    if total_val <= 0:
        raise ValueError("val_fraction is too small; val split would be empty.")
    if holdout_count >= total_val:
        raise ValueError(
            f"holdout_count={holdout_count} leaves no rows in val. "
            f"Reduce holdout_count below the val pool size {total_val}."
        )
    post_val_counts = allocate_post_val_counts(val_counts, holdout_count=holdout_count)
    train_counts = {
        note: note_counts[note] - val_counts[note]
        for note in NOTE_BUCKETS
    }

    rng = random.Random(seed)
    split_rows: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "post_val": []}
    for note in NOTE_BUCKETS:
        bucket_rows = list(rows_by_note[note])
        rng.shuffle(bucket_rows)
        val_pool = bucket_rows[: val_counts[note]]
        train_rows = bucket_rows[val_counts[note] :]
        post_rows = val_pool[: post_val_counts[note]]
        val_rows = val_pool[post_val_counts[note] :]
        split_rows["train"].extend(train_rows)
        split_rows["val"].extend(val_rows)
        split_rows["post_val"].extend(post_rows)

    for split_name in split_rows:
        rng.shuffle(split_rows[split_name])

    materialized_split_rows = {
        split_name: (
            _flatten_split_rows(split_rows[split_name], split_name=split_name)
            if flatten_by_box_element
            else list(split_rows[split_name])
        )
        for split_name in ("train", "val", "post_val")
    }
    if flatten_by_box_element:
        for split_name in materialized_split_rows:
            rng.shuffle(materialized_split_rows[split_name])

    splits = DatasetDict(
        {
            split_name: _dataset_from_rows(
                materialized_split_rows[split_name],
                features=None if flatten_by_box_element else features,
            )
            for split_name in ("train", "val", "post_val")
        }
    )
    allocations = {
        "note_counts": note_counts,
        "val_counts": val_counts,
        "post_val_counts": post_val_counts,
        "train_counts": train_counts,
    }
    return splits, allocations, materialized_split_rows


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    raw_argv = list(argv) if argv is not None else list(sys.argv[1:])
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    pre_args, _ = pre_parser.parse_known_args(raw_argv)
    config_path = resolve_config_path(pre_args.config, script_dir=SCRIPT_DIR)
    config = load_json_config(config_path, default_path=DEFAULT_CONFIG_PATH)

    parser = argparse.ArgumentParser(description="Generate deterministic football detect splits.")
    parser.add_argument("--config", default=str(config_path))
    parser.add_argument("--env-file", "--env", default=str(repo_relative(".env")))
    parser.add_argument("--dataset", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--split", default="train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--holdout-count", type=int, default=13)
    parser.add_argument(
        "--output-dir",
        default=str(repo_relative("outputs", "maxs-m87_football_detect_no_split_splits")),
    )
    parser.add_argument("--push-to-hub", default="", help="Optional HF repo id to push splits to.")
    parser.add_argument("--hub-val-split", default="validation")
    parser.add_argument("--hub-post-val-split", default="test")
    parser.add_argument("--flatten-by-box-element", action="store_true")
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


def _materialize_source_rows(
    *,
    dataset_name: str,
    split: str,
    hf_token: Optional[str],
) -> tuple[list[dict[str, Any]], Any, Optional[str]]:
    revision = DEFAULT_DATASET_REVISION if dataset_name == DEFAULT_DATASET_NAME else None
    ds = load_dataset(
        dataset_name,
        split=split,
        token=hf_token,
        revision=revision,
        streaming=True,
    )
    features = getattr(ds, "features", None)
    rows: list[dict[str, Any]] = []
    for row in ds:
        row_dict = dict(row)
        parse_note_bucket(row_dict.get("notes"))
        rows.append(row_dict)
    if not rows:
        raise ValueError("Dataset is empty.")
    return rows, features, revision


def _build_metadata(
    *,
    args: argparse.Namespace,
    source_revision: Optional[str],
    output_dir: Path,
    stats: dict[str, Any],
    allocations: dict[str, Any],
) -> dict[str, Any]:
    class_catalog = build_class_catalog(stats["class_catalog"])
    return {
        "config": args.config,
        "env_file": args.env_file,
        "dataset_name": args.dataset,
        "dataset_revision": source_revision or "",
        "source_split": args.split,
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "holdout_count": args.holdout_count,
        "output_dir": str(output_dir),
        "push_to_hub": bool(args.push_to_hub),
        "hub_repo_id": args.push_to_hub or "",
        "hub_val_split": args.hub_val_split,
        "hub_post_val_split": args.hub_post_val_split,
        "flatten_by_box_element": bool(args.flatten_by_box_element),
        "note_buckets": list(NOTE_BUCKETS),
        "class_catalog": class_catalog,
        "split_sizes": stats["split_sizes"],
        "note_bucket_counts": stats["note_bucket_counts"],
        "class_counts": stats["class_counts"],
        "allocation_note_counts": allocations["note_counts"],
        "allocation_val_counts": allocations["val_counts"],
        "allocation_post_val_counts": allocations["post_val_counts"],
        "allocation_train_counts": allocations["train_counts"],
    }


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    args.push_to_hub = str(args.push_to_hub or "").strip()
    load_dotenv(args.env_file, override=False)
    if not args.hf_token:
        args.hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    rows, features, source_revision = _materialize_source_rows(
        dataset_name=args.dataset,
        split=args.split,
        hf_token=args.hf_token,
    )
    splits, allocations, split_rows = build_splits_from_rows(
        rows,
        seed=args.seed,
        val_fraction=args.val_fraction,
        holdout_count=args.holdout_count,
        flatten_by_box_element=bool(args.flatten_by_box_element),
        features=features,
    )
    stats = build_split_stats(split_rows)

    output_dir = _resolve_output_dir(args.output_dir, args.dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits.save_to_disk(str(output_dir))

    metadata = _build_metadata(
        args=args,
        source_revision=source_revision,
        output_dir=output_dir,
        stats=stats,
        allocations=allocations,
    )
    write_json(output_dir / "metadata.json", metadata)
    write_json(output_dir / "stats.json", stats)

    print(
        f"saved splits to {output_dir} "
        f"(train={len(splits['train'])}, val={len(splits['val'])}, post_val={len(splits['post_val'])})"
    )

    if args.push_to_hub:
        if not args.hf_token:
            raise ValueError("HF token required to push to hub.")
        if len(splits["post_val"]) == 0:
            hub_splits = DatasetDict({"train": splits["train"], args.hub_val_split: splits["val"]})
            print(f"post_val split empty; pushing train + {args.hub_val_split} only.")
        else:
            hub_splits = _build_hub_splits(
                splits=splits,
                val_split_name=args.hub_val_split,
                post_val_split_name=args.hub_post_val_split,
            )
        hub_splits.push_to_hub(args.push_to_hub, token=args.hf_token)
        print(
            "pushed dataset to "
            f"{args.push_to_hub} (train, {args.hub_val_split}, {args.hub_post_val_split})"
        )


if __name__ == "__main__":
    main()
