#!/usr/bin/env python3
"""Build a Dataset1-only PI&D dataset into HF DatasetDict.

Key guarantees:
- Uses Dataset1 class names from lable_key.txt.
- Treats all Dataset1 images as patched source by policy.
- Uses strict group-based split assignment to avoid train/test leakage.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

from datasets import Dataset, DatasetDict, Features, Image as HFImage, Value
from dotenv import load_dotenv


def _repo_relative(*parts: str) -> Path:
    return Path(__file__).resolve().parent.joinpath(*parts)


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _base_id_from_stem(stem: str) -> str:
    return stem.split("_", 1)[0] if "_" in stem else stem


def _load_label_key(path: Path) -> dict[int, str]:
    if not path.exists():
        raise FileNotFoundError(f"Label key not found: {path}")

    mapping: dict[int, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            match = re.match(r"^(\d+)\s+(.+)$", line)
            if not match:
                continue
            class_id = int(match.group(1))
            class_name = " ".join(match.group(2).split())
            mapping[class_id] = class_name

    if not mapping:
        raise ValueError(f"No class mappings parsed from {path}")
    return mapping


def _load_dataset1_split_hints(dataset1_dir: Path) -> tuple[dict[str, str], dict[str, int]]:
    per_base_hints: dict[str, set[str]] = defaultdict(set)
    seen_stems: set[str] = set()
    hint_rows = 0
    hint_files = [
        (dataset1_dir / "train (2).txt", "train"),
        (dataset1_dir / "val (1).txt", "val"),
    ]
    for path, split in hint_files:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                stem = Path(line).stem
                if not stem:
                    continue
                if stem in seen_stems:
                    continue
                seen_stems.add(stem)
                hint_rows += 1
                base_id = _base_id_from_stem(stem)
                per_base_hints[base_id].add(split)

    resolved_hints: dict[str, str] = {}
    conflicting_bases = 0
    for base_id, splits in per_base_hints.items():
        if len(splits) == 1:
            resolved_hints[base_id] = next(iter(splits))
        else:
            conflicting_bases += 1

    stats = {
        "hint_rows": hint_rows,
        "hint_base_ids": len(per_base_hints),
        "hint_base_ids_resolved": len(resolved_hints),
        "hint_base_ids_conflicting": conflicting_bases,
    }
    return resolved_hints, stats


def _yolo_to_xyxy(cx: float, cy: float, w: float, h: float) -> tuple[float, float, float, float]:
    x_min = cx - (w / 2.0)
    y_min = cy - (h / 2.0)
    x_max = cx + (w / 2.0)
    y_max = cy + (h / 2.0)
    return _clamp01(x_min), _clamp01(y_min), _clamp01(x_max), _clamp01(y_max)


def _parse_dataset1_label_file(label_path: Path, class_map: dict[int, str]) -> list[dict[str, Any]]:
    boxes: list[dict[str, Any]] = []
    if not label_path.exists() or label_path.stat().st_size == 0:
        return boxes

    with label_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            class_id = _safe_float(parts[0])
            cx = _safe_float(parts[1])
            cy = _safe_float(parts[2])
            w = _safe_float(parts[3])
            h = _safe_float(parts[4])
            if None in (class_id, cx, cy, w, h):
                continue

            cid = int(class_id)
            class_name = class_map.get(cid, f"dataset1_class_{cid}")
            x_min, y_min, x_max, y_max = _yolo_to_xyxy(float(cx), float(cy), float(w), float(h))
            if x_max <= x_min or y_max <= y_min:
                continue

            boxes.append(
                {
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max,
                    "class_uid": f"dataset1:{cid}",
                    "class_name": class_name,
                    "source_class_id": cid,
                    "source_class_name": class_name,
                }
            )
    return boxes


def _iter_dataset1_rows(
    *,
    dataset1_dir: Path,
    class_map: dict[int, str],
    split_hints: dict[str, str],
) -> Iterable[dict[str, Any]]:
    image_dir = dataset1_dir / "images (3)"
    label_dir = dataset1_dir / "labels (2)"
    if not image_dir.exists():
        raise FileNotFoundError(f"Dataset1 image dir not found: {image_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"Dataset1 label dir not found: {label_dir}")

    for image_path in sorted(image_dir.glob("*.jpg")):
        stem = image_path.stem
        label_path = label_dir / f"{stem}.txt"
        boxes = _parse_dataset1_label_file(label_path, class_map)
        base_id = _base_id_from_stem(stem)
        split_hint = split_hints.get(base_id, "")

        yield {
            "image": str(image_path),
            "answer_boxes": json.dumps(boxes, separators=(",", ":")),
            "source_dataset": "dataset1",
            "source_collection": "dataset1",
            "source_variant": "patched",
            "source_is_patched": True,
            "source_image_id": stem,
            "source_base_id": base_id,
            "split_group_id": f"dataset1:{base_id}",
            "source_split_hint": split_hint,
            "class_count": len(boxes),
        }


def _build_group_split(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    val_fraction: float,
    post_val_fraction: float,
    respect_source_splits: bool,
) -> dict[str, set[str]]:
    group_to_hint: dict[str, str] = {}
    groups: list[str] = []

    for row in rows:
        group = row["split_group_id"]
        if group not in group_to_hint:
            group_to_hint[group] = ""
            groups.append(group)
        hint = row.get("source_split_hint", "")
        if hint in {"val", "test"}:
            group_to_hint[group] = hint

    rng = random.Random(seed)
    locked_holdout = {
        g for g, hint in group_to_hint.items() if respect_source_splits and hint in {"val", "test"}
    }
    candidate = [g for g in groups if g not in locked_holdout]
    rng.shuffle(candidate)

    target_val_total = int(round(len(groups) * val_fraction))
    target_val_total = max(target_val_total, 1 if len(groups) > 1 else 0)

    additional_val_needed = max(0, target_val_total - len(locked_holdout))
    additional_val = set(candidate[:additional_val_needed])

    train_groups = set(candidate[additional_val_needed:])
    val_pool = set(locked_holdout) | additional_val

    val_pool_list = sorted(val_pool)
    rng.shuffle(val_pool_list)
    n_post = int(round(len(val_pool_list) * post_val_fraction))
    if len(val_pool_list) > 1 and post_val_fraction > 0.0:
        n_post = max(1, n_post)
    n_post = min(n_post, len(val_pool_list))

    post_groups = set(val_pool_list[:n_post])
    val_groups = set(val_pool_list[n_post:])

    if not train_groups and val_groups:
        moved = sorted(val_groups)[0]
        val_groups.remove(moved)
        train_groups.add(moved)

    if not train_groups and post_groups:
        moved = sorted(post_groups)[0]
        post_groups.remove(moved)
        train_groups.add(moved)

    if not train_groups:
        raise ValueError("No train groups left after split assignment.")

    if train_groups & val_groups or train_groups & post_groups or val_groups & post_groups:
        raise ValueError("Split leakage detected: group overlap between train/val/post_val")

    return {
        "train": train_groups,
        "val": val_groups,
        "post_val": post_groups,
    }


def _split_rows(rows: list[dict[str, Any]], groups_by_split: dict[str, set[str]]) -> dict[str, list[dict[str, Any]]]:
    split_rows: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "post_val": []}
    for row in rows:
        group = row["split_group_id"]
        if group in groups_by_split["train"]:
            split_rows["train"].append(row)
        elif group in groups_by_split["val"]:
            split_rows["val"].append(row)
        elif group in groups_by_split["post_val"]:
            split_rows["post_val"].append(row)
        else:
            split_rows["train"].append(row)
    return split_rows


def _features() -> Features:
    return Features(
        {
            "image": HFImage(),
            "answer_boxes": Value("string"),
            "source_dataset": Value("string"),
            "source_collection": Value("string"),
            "source_variant": Value("string"),
            "source_is_patched": Value("bool"),
            "source_image_id": Value("string"),
            "source_base_id": Value("string"),
            "split_group_id": Value("string"),
            "source_split_hint": Value("string"),
            "class_count": Value("int32"),
        }
    )


def _make_dataset(rows: list[dict[str, Any]], features: Features) -> Dataset:
    if rows:
        return Dataset.from_list(rows, features=features)
    return Dataset.from_list([], features=features)


def _class_catalog(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_uid: dict[str, dict[str, Any]] = {}
    for row in rows:
        try:
            boxes = json.loads(row["answer_boxes"])
        except json.JSONDecodeError:
            continue
        for box in boxes:
            class_uid = str(box.get("class_uid", "")).strip()
            if not class_uid:
                continue
            if class_uid not in by_uid:
                by_uid[class_uid] = {
                    "class_uid": class_uid,
                    "class_name": box.get("class_name"),
                    "source_class_name": box.get("source_class_name"),
                    "source_class_id": box.get("source_class_id"),
                }
    return [by_uid[key] for key in sorted(by_uid)]


def _split_stats(split_rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for split_name, rows in split_rows.items():
        positives = 0
        negatives = 0
        by_source: dict[str, int] = defaultdict(int)
        groups: set[str] = set()
        for row in rows:
            groups.add(row["split_group_id"])
            by_source[f"{row['source_collection']}:{row['source_variant']}"] += 1
            if int(row.get("class_count", 0)) > 0:
                positives += 1
            else:
                negatives += 1
        payload[split_name] = {
            "samples": len(rows),
            "groups": len(groups),
            "positives": positives,
            "negatives": negatives,
            "by_source": dict(sorted(by_source.items())),
        }
    return payload


def _validate_no_group_leakage(split_rows: dict[str, list[dict[str, Any]]]) -> None:
    groups = {
        split_name: {row["split_group_id"] for row in rows}
        for split_name, rows in split_rows.items()
    }
    pairs = [("train", "val"), ("train", "post_val"), ("val", "post_val")]
    for a, b in pairs:
        overlap = groups[a] & groups[b]
        if overlap:
            sample = sorted(overlap)[:5]
            raise ValueError(f"Group leakage between {a} and {b}: {sample}")


def _resolve_output_dir(path_str: str) -> Path:
    if path_str:
        return Path(path_str).expanduser().resolve()
    return _repo_relative("outputs", "pid_icons_dataset1")


def _build_hub_splits(
    dataset_dict: DatasetDict,
    *,
    hub_val_split: str,
    hub_post_split: str,
) -> DatasetDict:
    if not hub_val_split or not hub_post_split:
        raise ValueError("Hub split names must be non-empty.")
    if hub_val_split == "train" or hub_post_split == "train":
        raise ValueError("Hub split names cannot be 'train'.")
    if hub_val_split == hub_post_split:
        raise ValueError("Hub val/post split names must be different.")
    return DatasetDict(
        {
            "train": dataset_dict["train"],
            hub_val_split: dataset_dict["val"],
            hub_post_split: dataset_dict["post_val"],
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Dataset1 HF dataset with leakage-safe splits.")
    parser.add_argument("--env-file", default=str(_repo_relative(".env")))
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"),
    )
    parser.add_argument("--raw-root", default=str(_repo_relative("datasets", "raw")))
    parser.add_argument("--dataset1-dir", default="")
    parser.add_argument(
        "--label-key",
        default=str(_repo_relative("datasets", "raw", "Dataset1", "lable_key.txt")),
    )
    parser.add_argument("--output-dir", default="")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--post-val-fraction", type=float, default=0.25)
    parser.add_argument(
        "--respect-source-splits",
        action="store_true",
        default=True,
        help="Keep source val/test hints out of train (enabled by default).",
    )
    parser.add_argument(
        "--no-respect-source-splits",
        dest="respect_source_splits",
        action="store_false",
    )

    parser.add_argument("--push-to-hub", default="")
    parser.add_argument("--hub-val-split", default="validation")
    parser.add_argument("--hub-post-val-split", default="test")
    args = parser.parse_args()

    load_dotenv(args.env_file, override=False)
    if not args.hf_token:
        args.hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    if not (0.0 < args.val_fraction < 1.0):
        raise ValueError("--val-fraction must be in (0, 1)")
    if not (0.0 <= args.post_val_fraction < 1.0):
        raise ValueError("--post-val-fraction must be in [0, 1)")

    raw_root = Path(args.raw_root).expanduser().resolve()
    dataset1_dir = Path(args.dataset1_dir).expanduser().resolve() if args.dataset1_dir else raw_root / "Dataset1"
    label_key_path = Path(args.label_key).expanduser().resolve()
    output_dir = _resolve_output_dir(args.output_dir)

    class_map = _load_label_key(label_key_path)
    split_hints, split_hint_stats = _load_dataset1_split_hints(dataset1_dir)
    rows = list(
        _iter_dataset1_rows(
            dataset1_dir=dataset1_dir,
            class_map=class_map,
            split_hints=split_hints,
        )
    )

    if not rows:
        raise ValueError("No rows were collected from Dataset1 inputs.")

    groups_by_split = _build_group_split(
        rows,
        seed=args.seed,
        val_fraction=args.val_fraction,
        post_val_fraction=args.post_val_fraction,
        respect_source_splits=args.respect_source_splits,
    )
    split_rows = _split_rows(rows, groups_by_split)
    _validate_no_group_leakage(split_rows)

    features = _features()
    dataset_dict = DatasetDict(
        {
            "train": _make_dataset(split_rows["train"], features),
            "val": _make_dataset(split_rows["val"], features),
            "post_val": _make_dataset(split_rows["post_val"], features),
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(output_dir))

    catalog = _class_catalog(rows)
    stats = _split_stats(split_rows)
    metadata = {
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "post_val_fraction": args.post_val_fraction,
        "respect_source_splits": args.respect_source_splits,
        "total_samples": len(rows),
        "class_count": len(catalog),
        "class_catalog": catalog,
        "split_stats": stats,
        "dataset1_label_key_path": str(label_key_path),
        "dataset1_treated_as_patched": True,
        "split_hint_stats": split_hint_stats,
        "source_paths": {
            "dataset1": str(dataset1_dir),
        },
    }

    metadata_path = output_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(
        f"saved dataset1 dataset to {output_dir} "
        f"(train={len(dataset_dict['train'])}, val={len(dataset_dict['val'])}, post_val={len(dataset_dict['post_val'])})"
    )
    print(
        "split hints: "
        f"rows={split_hint_stats['hint_rows']} "
        f"base_ids={split_hint_stats['hint_base_ids']} "
        f"resolved={split_hint_stats['hint_base_ids_resolved']} "
        f"conflicting_ignored={split_hint_stats['hint_base_ids_conflicting']}"
    )
    print(f"metadata: {metadata_path}")

    if args.push_to_hub:
        if not args.hf_token:
            raise ValueError("HF token required when using --push-to-hub")
        hub_splits = _build_hub_splits(
            dataset_dict,
            hub_val_split=args.hub_val_split,
            hub_post_split=args.hub_post_val_split,
        )
        hub_splits.push_to_hub(args.push_to_hub, token=args.hf_token)
        print(
            "pushed dataset to "
            f"{args.push_to_hub} (train, {args.hub_val_split}, {args.hub_post_val_split})"
        )


if __name__ == "__main__":
    main()
