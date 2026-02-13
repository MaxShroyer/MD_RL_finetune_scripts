"""Analyze State Farm datasets for positive samples.

Counts how many samples have at least one box vs empty, for multiple datasets.

Requires:
  pip install datasets
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Optional

from datasets import load_dataset


DATASETS = [
    # "maxs-m87/NBA_StateFarm_splits_p1",
    # "maxs-m87/NBA_StateFarm_splits_1-8th",
    # "maxs-m87/NBA_StateFarm_splits_1-4th",
    # "maxs-m87/NBA_StateFarm_Splits_01",
    "maxs-m87/NBA-statefarm-hard-100",
    "maxs-m87/NBA-statefarm-hard-50",
    "maxs-m87/NBA-statefarm-hard-25",
]


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


def _count_boxes(answer_boxes: Optional[object]) -> int:
    if not answer_boxes:
        return 0
    raw = answer_boxes
    if isinstance(answer_boxes, str):
        try:
            raw = json.loads(answer_boxes)
        except json.JSONDecodeError:
            return 0
    if not raw:
        return 0
    if isinstance(raw, dict) and "answer" in raw:
        raw = raw["answer"]
    return len(raw) if isinstance(raw, list) else 0


def _count_dataset(dataset_name: str, *, split: str, hf_token: Optional[str]) -> Dict[str, int]:
    ds = load_dataset(dataset_name, split=split, streaming=True, token=hf_token)
    total = 0
    with_boxes = 0
    empty = 0
    for row in ds:
        total += 1
        answer = row.get("answer") or row.get("answer_boxes")
        num_boxes = _count_boxes(answer)
        if num_boxes > 0:
            with_boxes += 1
        else:
            empty += 1
    return {
        "samples": total,
        "pos_samples": with_boxes,
        "empty": empty,
        "pos_samples_pct": (with_boxes / total) if total else 0.0,
        "empty_pct": (empty / total) if total else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze State Farm datasets for positive samples.")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"))
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--datasets",
        default="",
        help="Comma-separated dataset names. If omitted, uses DATASETS in this file.",
    )
    parser.add_argument("--out-json", default="outputs/statefarm_dataset_stats.json")
    args = parser.parse_args()

    _load_env_file(args.env_file)
    if not args.hf_token:
        args.hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    if not datasets:
        datasets = list(DATASETS)

    results: Dict[str, Dict[str, int]] = {}
    for dataset in datasets:
        stats = _count_dataset(dataset, split=args.split, hf_token=args.hf_token)
        results[dataset] = stats

    payload = {
        "split": args.split,
        "datasets": results,
    }
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
