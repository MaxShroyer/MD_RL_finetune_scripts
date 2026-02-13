"""Create deterministic train/val/post_val splits for the Ball-Holder dataset.

This script is modeled after the State Farm / Amazon split builders in this repo.

Key features:
- Deterministic splits by seed
- Explicit control over negative (empty) sample fraction
- Optional streaming reservoir sampling via --hard-sample-limit

Notes:
- "Empty" / negative samples are those where the annotation field is null/None/empty.
- Positive samples are those where the annotation field parses to a non-empty list/dict.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Optional

from datasets import Dataset, DatasetDict, load_dataset, load_dataset_builder
from dotenv import load_dotenv


DEFAULT_DATASET = "maxs-m87/Ball-Holder"


def _repo_relative(*parts: str) -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, *parts)


def _resolve_output_dir(output_dir: str, dataset_name: str) -> str:
    if output_dir:
        return output_dir
    safe_name = dataset_name.replace("/", "_")
    return _repo_relative("outputs", f"{safe_name}_splits")


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


def _split_indices(
    indices: list[Any],
    train_count: int,
    val_count: int,
    post_count: int,
) -> tuple[list[Any], list[Any], list[Any]]:
    train = indices[:train_count]
    val = indices[train_count : train_count + val_count]
    post_val = indices[train_count + val_count : train_count + val_count + post_count]
    return train, val, post_val


def _reservoir_sample(
    reservoir: list[dict],
    sample: dict,
    seen_count: int,
    limit: int,
    rng: random.Random,
) -> None:
    if limit <= 0:
        return
    if len(reservoir) < limit:
        reservoir.append(sample)
        return
    replacement_idx = rng.randint(0, seen_count - 1)
    if replacement_idx < limit:
        reservoir[replacement_idx] = sample


def _has_annotation(row: dict, annotation_field: str) -> bool:
    value = row.get(annotation_field)
    if value is None:
        # Some datasets may store the alternate field.
        value = row.get("answer_boxes") if annotation_field != "answer_boxes" else row.get("answer")
    if value is None:
        return False
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return False
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            # Non-empty but malformed: treat as annotated so we don't silently drop positives.
            return True
        value = parsed
    if isinstance(value, (list, tuple, dict)):
        return len(value) > 0
    return bool(value)


def _build_hard_limited_splits(
    *,
    dataset_name: str,
    split: str,
    token: Optional[str],
    seed: int,
    val_fraction: float,
    holdout_count: int,
    annotation_field: str,
    empty_fraction: float,
    hard_sample_limit: int,
) -> DatasetDict:
    total_target = hard_sample_limit
    unann_target = int(round(total_target * empty_fraction))
    ann_target = total_target - unann_target
    if ann_target <= 0 or unann_target <= 0:
        raise ValueError("Not enough data per class to satisfy empty fraction.")

    val_ann = int(round(ann_target * val_fraction))
    val_unann = int(round(unann_target * val_fraction))
    if val_ann == 0 or val_unann == 0:
        raise ValueError("val split would be empty for one class; increase --val-fraction.")
    val_count = val_ann + val_unann
    if val_count == 0:
        raise ValueError("val split would be empty; increase --val-fraction.")

    # For tiny hard limits, auto-adjust holdout down (matching State Farm helper behavior).
    if holdout_count > val_count:
        holdout_count = max(0, val_count - 1)
        print(
            f"holdout-count auto-adjusted to {holdout_count} for "
            f"--hard-sample-limit={hard_sample_limit} (val size={val_count})."
        )

    post_unann = int(round(holdout_count * (val_unann / val_count))) if val_count else 0
    post_unann = min(post_unann, val_unann)
    post_ann = holdout_count - post_unann
    if post_ann > val_ann:
        post_ann = val_ann
        post_unann = min(val_unann, holdout_count - post_ann)

    val_adj_ann = val_ann - post_ann
    val_adj_unann = val_unann - post_unann
    train_ann = ann_target - val_ann
    train_unann = unann_target - val_unann
    if train_ann <= 0 or train_unann <= 0 or val_adj_ann < 0 or val_adj_unann < 0:
        raise ValueError("Split sizes invalid; adjust --val-fraction or --holdout-count.")

    rng = random.Random(seed)
    annotated_rows: list[dict] = []
    unannotated_rows: list[dict] = []
    ann_seen = 0
    unann_seen = 0

    stream_ds = load_dataset(dataset_name, split=split, token=token, streaming=True)
    for row in stream_ds:
        row_dict = dict(row)
        if _has_annotation(row_dict, annotation_field):
            ann_seen += 1
            _reservoir_sample(annotated_rows, row_dict, ann_seen, ann_target, rng)
        else:
            unann_seen += 1
            _reservoir_sample(unannotated_rows, row_dict, unann_seen, unann_target, rng)

    if len(annotated_rows) < ann_target or len(unannotated_rows) < unann_target:
        raise ValueError(
            "Not enough data to satisfy --hard-sample-limit with current --empty-fraction. "
            f"Needed ann/unann={ann_target}/{unann_target}, found seen={ann_seen}/{unann_seen}."
        )

    rng.shuffle(annotated_rows)
    rng.shuffle(unannotated_rows)

    ann_train, ann_val, ann_post = _split_indices(
        annotated_rows,
        train_ann,
        val_adj_ann,
        post_ann,
    )
    unann_train, unann_val, unann_post = _split_indices(
        unannotated_rows,
        train_unann,
        val_adj_unann,
        post_unann,
    )

    train_rows = ann_train + unann_train
    val_rows = ann_val + unann_val
    post_val_rows = ann_post + unann_post
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    rng.shuffle(post_val_rows)

    features = load_dataset_builder(dataset_name, token=token).info.features
    train_ds = Dataset.from_list(train_rows, features=features)
    val_ds = Dataset.from_list(val_rows, features=features) if val_rows else train_ds.select([])
    post_val_ds = (
        Dataset.from_list(post_val_rows, features=features) if post_val_rows else train_ds.select([])
    )
    return DatasetDict(train=train_ds, val=val_ds, post_val=post_val_ds)


def _build_splits(
    *,
    dataset_name: str,
    split: str,
    token: Optional[str],
    seed: int,
    val_fraction: float,
    holdout_count: int,
    annotation_field: str,
    shrink_fraction: float,
    empty_fraction: float,
    hard_sample_limit: Optional[int],
) -> DatasetDict:
    if val_fraction <= 0.0 or val_fraction >= 1.0:
        raise ValueError("--val-fraction must be in (0, 1).")
    if shrink_fraction <= 0.0 or shrink_fraction > 1.0:
        raise ValueError("--shrink-fraction must be in (0, 1].")
    if empty_fraction <= 0.0 or empty_fraction >= 1.0:
        raise ValueError("--empty-fraction must be in (0, 1).")
    if holdout_count < 0:
        raise ValueError("--holdout-count must be >= 0.")
    if hard_sample_limit is not None and hard_sample_limit <= 1:
        raise ValueError("--hard-sample-limit must be >= 2.")

    if hard_sample_limit is not None:
        return _build_hard_limited_splits(
            dataset_name=dataset_name,
            split=split,
            token=token,
            seed=seed,
            val_fraction=val_fraction,
            holdout_count=holdout_count,
            annotation_field=annotation_field,
            empty_fraction=empty_fraction,
            hard_sample_limit=hard_sample_limit,
        )

    ds = load_dataset(dataset_name, split=split, token=token)
    total = len(ds)
    if total == 0:
        raise ValueError("Dataset is empty.")

    annotated_indices = []
    unannotated_indices = []
    for idx, row in enumerate(ds):
        if _has_annotation(row, annotation_field):
            annotated_indices.append(idx)
        else:
            unannotated_indices.append(idx)

    if not annotated_indices or not unannotated_indices:
        raise ValueError("Need both annotated and unannotated samples to build splits.")

    rng = random.Random(seed)
    rng.shuffle(annotated_indices)
    rng.shuffle(unannotated_indices)

    total_target = int(round(total * shrink_fraction))
    if total_target < 2:
        raise ValueError("Shrink fraction too small; dataset would be empty.")

    max_total = min(
        int(len(annotated_indices) / (1.0 - empty_fraction)),
        int(len(unannotated_indices) / empty_fraction),
    )
    if max_total < 2:
        raise ValueError("Not enough data to build splits with the requested empty fraction.")
    if total_target > max_total:
        total_target = max_total

    unann_target = int(round(total_target * empty_fraction))
    unann_target = min(unann_target, len(unannotated_indices))
    ann_target = total_target - unann_target
    ann_target = min(ann_target, len(annotated_indices))
    total_target = ann_target + unann_target
    if ann_target <= 0 or unann_target <= 0:
        raise ValueError("Not enough data per class to satisfy empty fraction.")

    val_ann = int(round(ann_target * val_fraction))
    val_unann = int(round(unann_target * val_fraction))
    if val_ann == 0 or val_unann == 0:
        raise ValueError("val split would be empty for one class; increase --val-fraction.")
    val_count = val_ann + val_unann
    if val_count == 0:
        raise ValueError("val split would be empty; increase --val-fraction.")
    if holdout_count > val_count:
        raise ValueError(f"holdout-count ({holdout_count}) cannot exceed val split size ({val_count}).")

    post_unann = int(round(holdout_count * (val_unann / val_count))) if val_count else 0
    post_unann = min(post_unann, val_unann)
    post_ann = holdout_count - post_unann
    if post_ann > val_ann:
        post_ann = val_ann
        post_unann = min(val_unann, holdout_count - post_ann)

    val_adj_ann = val_ann - post_ann
    val_adj_unann = val_unann - post_unann
    train_ann = ann_target - val_ann
    train_unann = unann_target - val_unann
    if train_ann <= 0 or train_unann <= 0 or val_adj_ann < 0 or val_adj_unann < 0:
        raise ValueError("Split sizes invalid; adjust --val-fraction or --holdout-count.")

    annotated_indices = annotated_indices[:ann_target]
    unannotated_indices = unannotated_indices[:unann_target]

    ann_train, ann_val, ann_post = _split_indices(annotated_indices, train_ann, val_adj_ann, post_ann)
    unann_train, unann_val, unann_post = _split_indices(unannotated_indices, train_unann, val_adj_unann, post_unann)

    train_indices = ann_train + unann_train
    val_indices = ann_val + unann_val
    post_val_indices = ann_post + unann_post
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(post_val_indices)

    train_ds = ds.select(train_indices)
    val_ds = ds.select(val_indices) if val_indices else ds.select([])
    post_val_ds = ds.select(post_val_indices) if post_val_indices else ds.select([])
    return DatasetDict(train=train_ds, val=val_ds, post_val=post_val_ds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic Ball-Holder train/val/post_val splits.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="HF dataset name.")
    parser.add_argument("--split", default="train", help="Base split to slice.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--holdout-count", type=int, default=1000)
    parser.add_argument("--annotation-field", default="answer_boxes")
    parser.add_argument(
        "--shrink-fraction",
        type=float,
        default=1.0,
        help="Fraction of the original dataset to keep.",
    )
    parser.add_argument(
        "--empty-fraction",
        type=float,
        default=0.25,
        help="Fraction of each split that should be empty/unannotated (negatives).",
    )
    parser.add_argument(
        "--hard-sample-limit",
        type=int,
        default=None,
        help=(
            "Exact target number of samples to keep (overrides --shrink-fraction). "
            "Uses streaming reservoir sampling to avoid full dataset cache writes."
        ),
    )
    parser.add_argument(
        "--env-file",
        "--env",
        default=_repo_relative(".env"),
        help="Path to a dotenv file (defaults to MDBallHolder/.env).",
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--push-to-hub", default="", help="Optional: HF repo to push splits to.")
    parser.add_argument("--hub-val-split", default="validation")
    parser.add_argument("--hub-post-val-split", default="test")
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"),
    )
    args = parser.parse_args()

    load_dotenv(args.env_file, override=False)
    if not args.hf_token:
        args.hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    splits = _build_splits(
        dataset_name=args.dataset,
        split=args.split,
        token=args.hf_token,
        seed=args.seed,
        val_fraction=args.val_fraction,
        holdout_count=args.holdout_count,
        annotation_field=args.annotation_field,
        shrink_fraction=args.shrink_fraction,
        empty_fraction=args.empty_fraction,
        hard_sample_limit=args.hard_sample_limit,
    )

    output_dir = _resolve_output_dir(args.output_dir, args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    filtered_splits = DatasetDict({name: ds for name, ds in splits.items() if len(ds) > 0})
    filtered_splits.save_to_disk(output_dir)

    train_count = len(splits["train"])
    val_count = len(splits["val"])
    post_val_count = len(splits["post_val"])
    total = train_count + val_count + post_val_count
    print(
        f"saved splits to {output_dir} "
        f"(train={train_count}, val={val_count}, post_val={post_val_count}, total={total})"
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
