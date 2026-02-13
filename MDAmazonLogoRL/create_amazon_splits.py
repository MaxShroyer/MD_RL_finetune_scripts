"""Create deterministic train/val/post_val splits for a HF dataset."""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Optional

from datasets import DatasetDict, load_dataset


def _resolve_output_dir(output_dir: str, dataset_name: str) -> str:
    if output_dir:
        return output_dir
    safe_name = dataset_name.replace("/", "_")
    return os.path.join("outputs", f"{safe_name}_splits")


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


def _build_splits(
    dataset_name: str,
    split: str,
    token: Optional[str],
    seed: int,
    val_fraction: float,
    holdout_count: int,
) -> DatasetDict:
    ds = load_dataset(dataset_name, split=split, token=token)
    total = len(ds)
    if total == 0:
        raise ValueError("Dataset is empty.")
    if val_fraction <= 0.0 or val_fraction >= 1.0:
        raise ValueError("--val-fraction must be in (0, 1).")
    if holdout_count < 0:
        raise ValueError("--holdout-count must be >= 0.")

    def _has_boxes(row: dict) -> bool:
        value = row.get("answer_boxes")
        if not value:
            return False
        try:
            raw = json.loads(value) if isinstance(value, str) else value
        except json.JSONDecodeError:
            return False
        if not raw:
            return False
        return len(raw) > 0

    annotated_indices = []
    empty_indices = []
    for idx, row in enumerate(ds):
        if _has_boxes(row):
            annotated_indices.append(idx)
        else:
            empty_indices.append(idx)

    if not annotated_indices:
        raise ValueError("Need at least one annotated sample to cap empty fraction.")

    rng = random.Random(seed)
    rng.shuffle(annotated_indices)
    rng.shuffle(empty_indices)

    def _split_sizes(total_target: int) -> tuple[int, int, int, int]:
        val_count = int(round(total_target * val_fraction))
        if val_count == 0:
            raise ValueError("val split would be empty; increase --val-fraction.")
        if holdout_count > val_count:
            raise ValueError(
                f"holdout-count ({holdout_count}) cannot exceed val split size ({val_count})."
            )
        train_count = total_target - val_count
        post_val_count = holdout_count
        val_count_adj = val_count - post_val_count
        if train_count <= 0:
            raise ValueError("train split would be empty; adjust --val-fraction.")
        return train_count, val_count_adj, post_val_count, val_count

    def _max_empty_for_split(split_size: int) -> int:
        return split_size // 3

    annotated_count = len(annotated_indices)
    empty_count = len(empty_indices)
    total_target = total
    while total_target > 0:
        train_count, val_count_adj, post_val_count, _ = _split_sizes(total_target)
        max_empty_total = (
            _max_empty_for_split(train_count)
            + _max_empty_for_split(val_count_adj)
            + _max_empty_for_split(post_val_count)
        )
        empty_target = min(empty_count, max_empty_total)
        required_non_empty = total_target - empty_target
        if required_non_empty <= annotated_count:
            break
        total_target -= 1

    if total_target <= 0:
        raise ValueError("Not enough annotated samples to cap empty fraction at 1/3.")

    train_count, val_count_adj, post_val_count, _ = _split_sizes(total_target)
    max_empty_total = (
        _max_empty_for_split(train_count)
        + _max_empty_for_split(val_count_adj)
        + _max_empty_for_split(post_val_count)
    )
    empty_target = min(empty_count, max_empty_total)
    annotated_target = total_target - empty_target
    if annotated_target > annotated_count:
        raise ValueError("Not enough annotated samples to satisfy empty cap.")

    split_sizes = {"train": train_count, "val": val_count_adj, "post_val": post_val_count}
    max_empty = {name: _max_empty_for_split(size) for name, size in split_sizes.items()}
    ideal = {
        name: (empty_target * size / total_target) if total_target else 0.0
        for name, size in split_sizes.items()
    }
    empty_alloc = {name: min(max_empty[name], int(ideal[name])) for name in split_sizes}
    remaining = empty_target - sum(empty_alloc.values())
    remainders = sorted(
        ((name, ideal[name] - int(ideal[name])) for name in split_sizes),
        key=lambda item: item[1],
        reverse=True,
    )
    for name, _ in remainders:
        if remaining <= 0:
            break
        if empty_alloc[name] < max_empty[name]:
            empty_alloc[name] += 1
            remaining -= 1
    if remaining > 0:
        for name in ("train", "val", "post_val"):
            while remaining > 0 and empty_alloc[name] < max_empty[name]:
                empty_alloc[name] += 1
                remaining -= 1
    if remaining > 0:
        raise ValueError("Unable to allocate empty samples within 1/3 cap.")

    train_empty = empty_alloc["train"]
    val_empty = empty_alloc["val"]
    post_empty = empty_alloc["post_val"]
    train_ann = train_count - train_empty
    val_ann = val_count_adj - val_empty
    post_ann = post_val_count - post_empty
    if train_ann < 0 or val_ann < 0 or post_ann < 0:
        raise ValueError("Split sizes invalid; adjust --val-fraction or --holdout-count.")

    annotated_indices = annotated_indices[:annotated_target]
    empty_indices = empty_indices[:empty_target]

    def _split_indices(
        indices: list[int],
        train_count: int,
        val_count: int,
        post_count: int,
    ) -> tuple[list[int], list[int], list[int]]:
        train = indices[:train_count]
        val = indices[train_count : train_count + val_count]
        post_val = indices[train_count + val_count : train_count + val_count + post_count]
        return train, val, post_val

    ann_train, ann_val, ann_post = _split_indices(annotated_indices, train_ann, val_ann, post_ann)
    empty_train, empty_val, empty_post = _split_indices(empty_indices, train_empty, val_empty, post_empty)

    train_indices = ann_train + empty_train
    val_indices = ann_val + empty_val
    post_val_indices = ann_post + empty_post
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(post_val_indices)

    train_ds = ds.select(train_indices)
    val_ds = ds.select(val_indices) if val_indices else ds.select([])
    post_val_ds = ds.select(post_val_indices) if post_val_indices else ds.select([])

    return DatasetDict(
        train=train_ds,
        val=val_ds,
        post_val=post_val_ds,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create 90/10 splits with post-training holdout.")
    parser.add_argument("--dataset", required=True, help="HF dataset name, e.g. maxs-m87/Amazon_NBA_re")
    parser.add_argument("--split", default="train", help="Base split to slice")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--holdout-count", type=int, default=50)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--push-to-hub", default="")
    parser.add_argument(
        "--hub-val-split",
        default="validation",
        help="Split name to use for val when pushing to hub.",
    )
    parser.add_argument(
        "--hub-post-val-split",
        default="test",
        help="Split name to use for post_val when pushing to hub.",
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"),
    )
    args = parser.parse_args()

    splits = _build_splits(
        dataset_name=args.dataset,
        split=args.split,
        token=args.hf_token,
        seed=args.seed,
        val_fraction=args.val_fraction,
        holdout_count=args.holdout_count,
    )

    output_dir = _resolve_output_dir(args.output_dir, args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    splits.save_to_disk(output_dir)
    total = len(splits["train"]) + len(splits["val"]) + len(splits["post_val"])
    print(
        f"saved splits to {output_dir} "
        f"(train={len(splits['train'])}, val={len(splits['val'])}, post_val={len(splits['post_val'])}, total={total})"
    )

    if args.push_to_hub:
        if not args.hf_token:
            raise ValueError("HF token required to push to hub.")
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
