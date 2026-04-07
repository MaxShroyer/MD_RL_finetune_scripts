"""Analyze Ball-Holder dataset label distribution and basic quality stats.

Counts positive vs negative samples and summarizes annotation coverage.
Supports both Hub datasets and local datasets saved with datasets.save_to_disk().
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Iterable, Optional

from datasets import Dataset, DatasetDict, get_dataset_split_names, load_dataset, load_from_disk
from dotenv import load_dotenv


DEFAULT_DATASET = "maxs-m87/Ball-Holder"


@dataclass
class SplitStats:
    samples: int = 0
    positives: int = 0
    negatives: int = 0
    malformed_annotations: int = 0
    max_boxes_in_sample: int = 0
    positive_box_count_sum: int = 0
    image_width_min: Optional[int] = None
    image_width_max: Optional[int] = None
    image_width_sum: int = 0
    image_height_min: Optional[int] = None
    image_height_max: Optional[int] = None
    image_height_sum: int = 0

    def update_image_size(self, width: int, height: int) -> None:
        self.image_width_min = width if self.image_width_min is None else min(self.image_width_min, width)
        self.image_width_max = width if self.image_width_max is None else max(self.image_width_max, width)
        self.image_width_sum += width
        self.image_height_min = height if self.image_height_min is None else min(self.image_height_min, height)
        self.image_height_max = height if self.image_height_max is None else max(self.image_height_max, height)
        self.image_height_sum += height

    def finalize(self) -> dict:
        total = self.samples
        positives = self.positives
        negatives = self.negatives
        positive_pct = (positives / total) if total else 0.0
        negative_pct = (negatives / total) if total else 0.0
        avg_boxes_positive = (self.positive_box_count_sum / positives) if positives else 0.0
        neg_to_pos_ratio = (negatives / positives) if positives else None
        avg_width = (self.image_width_sum / total) if total else 0.0
        avg_height = (self.image_height_sum / total) if total else 0.0
        payload = asdict(self)
        payload.update(
            {
                "positive_pct": positive_pct,
                "negative_pct": negative_pct,
                "neg_to_pos_ratio": neg_to_pos_ratio,
                "avg_boxes_per_positive": avg_boxes_positive,
                "avg_image_width": avg_width,
                "avg_image_height": avg_height,
            }
        )
        return payload


def _repo_relative(*parts: str) -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, *parts)


def _parse_boxes_value(raw_value: object) -> tuple[int, bool]:
    if raw_value is None:
        return 0, False
    value = raw_value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return 0, False
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return 0, True
    if isinstance(value, dict):
        if "answer" in value and isinstance(value["answer"], list):
            return len(value["answer"]), False
        return (1 if len(value) > 0 else 0), False
    if isinstance(value, (list, tuple)):
        return len(value), False
    return (1 if bool(value) else 0), False


def _count_boxes(
    row: dict,
    *,
    annotation_field: str,
    fallback_field: Optional[str],
) -> tuple[int, bool]:
    count, malformed = _parse_boxes_value(row.get(annotation_field))
    if count > 0 or malformed:
        return count, malformed
    if fallback_field:
        fallback_count, fallback_malformed = _parse_boxes_value(row.get(fallback_field))
        return fallback_count, fallback_malformed
    return 0, False


def _iter_rows(
    *,
    dataset_name: str,
    dataset_path: Optional[str],
    split: str,
    token: Optional[str],
    streaming: bool,
    seed: int,
    buffer_size: int,
    shuffle: bool,
) -> Iterable[dict]:
    if dataset_path:
        dataset_obj = load_from_disk(dataset_path)
        if isinstance(dataset_obj, DatasetDict):
            if split not in dataset_obj:
                available = ", ".join(dataset_obj.keys())
                raise ValueError(f"Split '{split}' not found in local dataset. Available: {available}")
            ds: Dataset = dataset_obj[split]
        else:
            ds = dataset_obj
        if shuffle:
            ds = ds.shuffle(seed=seed)
        for row in ds:
            yield row
        return

    ds = load_dataset(dataset_name, split=split, token=token, streaming=streaming)
    if shuffle:
        if streaming:
            ds = ds.shuffle(seed=seed, buffer_size=buffer_size)
        else:
            ds = ds.shuffle(seed=seed)
    for row in ds:
        yield row


def _resolve_splits(
    *,
    dataset_name: str,
    dataset_path: Optional[str],
    requested_split: str,
    all_splits: bool,
    token: Optional[str],
) -> list[str]:
    if not all_splits:
        return [requested_split]
    if dataset_path:
        dataset_obj = load_from_disk(dataset_path)
        if isinstance(dataset_obj, DatasetDict):
            return list(dataset_obj.keys())
        return [requested_split]
    split_names = get_dataset_split_names(dataset_name, token=token)
    if not split_names:
        return [requested_split]
    return split_names


def _analyze_split(
    *,
    dataset_name: str,
    dataset_path: Optional[str],
    split: str,
    token: Optional[str],
    streaming: bool,
    seed: int,
    buffer_size: int,
    shuffle: bool,
    max_samples: Optional[int],
    annotation_field: str,
    fallback_field: Optional[str],
) -> dict:
    stats = SplitStats()
    for row in _iter_rows(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        split=split,
        token=token,
        streaming=streaming,
        seed=seed,
        buffer_size=buffer_size,
        shuffle=shuffle,
    ):
        if max_samples is not None and stats.samples >= max_samples:
            break

        image = row.get("image")
        if hasattr(image, "size"):
            width, height = image.size
            stats.update_image_size(int(width), int(height))

        box_count, malformed = _count_boxes(
            row,
            annotation_field=annotation_field,
            fallback_field=fallback_field,
        )
        stats.samples += 1
        if malformed:
            stats.malformed_annotations += 1
        if box_count > 0:
            stats.positives += 1
            stats.positive_box_count_sum += box_count
            stats.max_boxes_in_sample = max(stats.max_boxes_in_sample, box_count)
        else:
            stats.negatives += 1

    return stats.finalize()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Ball-Holder dataset coverage and balance.")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET, help="HF dataset name.")
    parser.add_argument(
        "--dataset-path",
        default="",
        help="Optional local path produced by datasets.save_to_disk().",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--all-splits",
        action="store_true",
        help="Analyze all available splits instead of only --split.",
    )
    parser.add_argument("--annotation-field", default="answer_boxes")
    parser.add_argument(
        "--fallback-field",
        default="answer",
        help="Fallback annotation field when --annotation-field is empty/null.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--streaming", action="store_true", help="Use streaming mode for HF datasets.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle rows before analysis.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--buffer-size", type=int, default=1000)
    parser.add_argument(
        "--env-file",
        "--env",
        default=_repo_relative(".env"),
        help="Path to dotenv file (defaults to MDBallHolder/.env).",
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"),
    )
    parser.add_argument(
        "--out-json",
        default=_repo_relative("outputs", "ballholder_dataset_stats.json"),
    )
    args = parser.parse_args()

    load_dotenv(args.env_file, override=False)
    if not args.hf_token:
        args.hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    dataset_path = args.dataset_path.strip() or None
    split_names = _resolve_splits(
        dataset_name=args.dataset_name,
        dataset_path=dataset_path,
        requested_split=args.split,
        all_splits=args.all_splits,
        token=args.hf_token,
    )

    split_results: dict[str, dict] = {}
    for split_name in split_names:
        split_results[split_name] = _analyze_split(
            dataset_name=args.dataset_name,
            dataset_path=dataset_path,
            split=split_name,
            token=args.hf_token,
            streaming=args.streaming,
            seed=args.seed,
            buffer_size=args.buffer_size,
            shuffle=args.shuffle,
            max_samples=args.max_samples,
            annotation_field=args.annotation_field,
            fallback_field=args.fallback_field,
        )

    payload = {
        "dataset_name": args.dataset_name,
        "dataset_path": dataset_path,
        "annotation_field": args.annotation_field,
        "fallback_field": args.fallback_field,
        "max_samples": args.max_samples,
        "splits": split_results,
    }
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

