"""Analyze Amazon logo dataset labeling coverage.

Counts how many samples have at least one box vs empty.

Requires:
  pip install datasets
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional

from datasets import load_dataset


DATASET_NAME = "maxs-m87/Amazon-extended-labels-splits"


def _load_env_file(path: str) -> None:
    if not path or not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            if key and key not in os.environ:
                os.environ[key] = value


def _count_boxes(answer_boxes: Optional[str]) -> int:
    if not answer_boxes:
        return 0
    try:
        raw = json.loads(answer_boxes) if isinstance(answer_boxes, str) else answer_boxes
    except json.JSONDecodeError:
        return 0
    if not raw:
        return 0
    return len(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Amazon logo dataset labels.")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--buffer-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _load_env_file(args.env_file)
    if not args.hf_token:
        args.hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    ds = load_dataset(DATASET_NAME, split=args.split, streaming=True, token=args.hf_token)
    ds = ds.shuffle(seed=args.seed, buffer_size=args.buffer_size)

    total = 0
    with_boxes = 0
    empty = 0
    max_boxes = 0

    for row in ds:
        if args.max_samples is not None and total >= args.max_samples:
            break
        num_boxes = _count_boxes(row.get("answer_boxes"))
        total += 1
        if num_boxes > 0:
            with_boxes += 1
            max_boxes = max(max_boxes, num_boxes)
        else:
            empty += 1

    summary = {
        "dataset": DATASET_NAME,
        "split": args.split,
        "samples": total,
        "with_boxes": with_boxes,
        "empty": empty,
        "with_boxes_pct": (with_boxes / total) if total else 0.0,
        "empty_pct": (empty / total) if total else 0.0,
        "max_boxes_in_sample": max_boxes,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
