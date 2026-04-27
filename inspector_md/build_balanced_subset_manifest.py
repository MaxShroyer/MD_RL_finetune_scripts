#!/usr/bin/env python3
"""Build a balanced subset manifest for Inspector MD."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inspector_md import common

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = common.repo_relative("configs", "build_balanced_subset_manifest_default.json")
DEFAULT_SOURCE_MANIFEST = common.repo_relative("dataset", "merged_synth_v1", "synthetic_manifest.json")
DEFAULT_OUTPUT_DIR = common.repo_relative("dataset", "subset_balanced_1000")
NEGATIVE_BUCKET = "__negative__"


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    raw_argv = list(argv) if argv is not None else list(os.sys.argv[1:])
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    pre_args, _ = pre_parser.parse_known_args(raw_argv)
    config_path = common.resolve_config_path(pre_args.config, script_dir=SCRIPT_DIR)
    config = common.load_json_config(config_path, default_path=DEFAULT_CONFIG_PATH)

    parser = argparse.ArgumentParser(description="Build a balanced subset of the Inspector MD source manifest.")
    parser.add_argument("--config", default=str(config_path))
    parser.add_argument("--source-manifest", default=str(DEFAULT_SOURCE_MANIFEST))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--negative-bucket-label", default=NEGATIVE_BUCKET)

    option_to_dest: dict[str, str] = {}
    for action in parser._actions:
        if not action.option_strings:
            continue
        for opt in action.option_strings:
            option_to_dest[opt] = action.dest
    overridden = {option_to_dest[arg] for arg in raw_argv if arg in option_to_dest}
    config_cli_args = common.config_to_cli_args(parser, config, config_path=config_path, overridden_dests=overridden)
    args = parser.parse_args(config_cli_args + raw_argv)
    args.config = str(common.resolve_config_path(args.config, script_dir=SCRIPT_DIR))
    args.source_manifest = common.resolve_path(args.source_manifest, module_root=SCRIPT_DIR)
    args.output_dir = common.resolve_path(args.output_dir, module_root=SCRIPT_DIR)
    if int(args.max_samples) <= 0:
        raise ValueError("--max-samples must be > 0")
    return args


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Manifest must be a JSON array: {path}")
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Manifest row {index} is not an object: {path}")
        rows.append(dict(item))
    return rows


def _row_buckets(row: Mapping[str, Any], *, negative_bucket_label: str) -> list[str]:
    proposals = row.get("expected_proposals")
    if not isinstance(proposals, list) or not proposals:
        return [str(negative_bucket_label)]
    issue_codes = sorted(
        {
            str(item.get("issue_code") or "").strip()
            for item in proposals
            if isinstance(item, dict) and str(item.get("issue_code") or "").strip()
        }
    )
    return issue_codes or [str(negative_bucket_label)]


def _allocate_even_targets(capacities: Mapping[str, int], total: int) -> dict[str, int]:
    targets = {key: 0 for key in capacities}
    active = {key for key, value in capacities.items() if int(value) > 0}
    remaining = int(total)
    while remaining > 0 and active:
        fair_share = remaining / float(len(active))
        exhausted: list[str] = []
        for key in sorted(active):
            capacity = int(capacities[key])
            current = int(targets[key])
            remaining_capacity = capacity - current
            if remaining_capacity <= 0:
                exhausted.append(key)
                continue
            if remaining_capacity <= fair_share:
                targets[key] = capacity
                remaining -= remaining_capacity
                exhausted.append(key)
        if exhausted:
            for key in exhausted:
                active.discard(key)
            continue
        base_share = remaining // len(active)
        if base_share > 0:
            for key in sorted(active):
                targets[key] += base_share
                remaining -= base_share
        if remaining <= 0:
            break
        for key in sorted(active):
            if remaining <= 0:
                break
            capacity = int(capacities[key])
            if targets[key] >= capacity:
                continue
            targets[key] += 1
            remaining -= 1
        active = {key for key in active if targets[key] < int(capacities[key])}
    return targets


def _allocate_proportional_targets(capacities: Mapping[str, int], total: int) -> dict[str, int]:
    allocations = {key: 0 for key in capacities}
    requested_total = int(total)
    available_total = sum(max(0, int(value)) for value in capacities.values())
    if requested_total <= 0 or available_total <= 0:
        return allocations
    capped_total = min(requested_total, available_total)
    remainders: list[tuple[float, str]] = []
    assigned = 0
    for key in sorted(capacities):
        capacity = max(0, int(capacities[key]))
        exact = capped_total * (capacity / float(available_total))
        whole = min(capacity, int(exact))
        allocations[key] = whole
        assigned += whole
        remainders.append((exact - whole, key))
    leftover = capped_total - assigned
    for _, key in sorted(remainders, key=lambda item: (-item[0], item[1])):
        if leftover <= 0:
            break
        capacity = max(0, int(capacities[key]))
        if allocations[key] >= capacity:
            continue
        allocations[key] += 1
        leftover -= 1
    return allocations


def _select_rows_for_bucket(
    rows: list[dict[str, Any]],
    *,
    target_count: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        split = str(row.get("split") or "").strip() or "train"
        by_split[split].append(dict(row))
    split_targets = _allocate_proportional_targets({split: len(items) for split, items in by_split.items()}, target_count)
    selected: list[dict[str, Any]] = []
    for split_name in sorted(by_split):
        pool = list(by_split[split_name])
        rng.shuffle(pool)
        selected.extend(pool[: int(split_targets.get(split_name, 0))])
    return selected


def build_subset(args: argparse.Namespace) -> dict[str, Any]:
    source_manifest = Path(args.source_manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_manifest(source_manifest)
    if len(rows) < int(args.max_samples):
        raise ValueError(
            f"Requested {int(args.max_samples)} samples but manifest only contains {len(rows)} rows: {source_manifest}"
        )

    membership_counts: Counter[str] = Counter()
    row_bucket_map: list[list[str]] = []
    for row in rows:
        buckets = _row_buckets(row, negative_bucket_label=str(args.negative_bucket_label))
        row_bucket_map.append(buckets)
        for bucket in buckets:
            membership_counts[bucket] += 1

    assigned_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row, buckets in zip(rows, row_bucket_map):
        primary_bucket = min(buckets, key=lambda bucket: (membership_counts[bucket], bucket))
        row_copy = dict(row)
        row_copy["subset_bucket"] = primary_bucket
        assigned_rows[primary_bucket].append(row_copy)

    capacities = {bucket: len(items) for bucket, items in assigned_rows.items()}
    bucket_targets = _allocate_even_targets(capacities, int(args.max_samples))
    rng = random.Random(int(args.seed))
    selected_rows: list[dict[str, Any]] = []
    for bucket_name in sorted(assigned_rows):
        selected_rows.extend(
            _select_rows_for_bucket(
                assigned_rows[bucket_name],
                target_count=int(bucket_targets.get(bucket_name, 0)),
                rng=rng,
            )
        )

    if len(selected_rows) != int(args.max_samples):
        raise RuntimeError(
            f"Balanced subset selection produced {len(selected_rows)} rows, expected {int(args.max_samples)}"
        )

    selected_rows.sort(key=lambda row: str(row.get("row_id") or ""))
    common.write_json(output_dir / "synthetic_manifest.json", selected_rows)

    selected_bucket_counts = Counter(str(row.get("subset_bucket") or "") for row in selected_rows)
    selected_split_counts = Counter(str(row.get("split") or "") for row in selected_rows)
    selected_source_counts = Counter(str(row.get("source_dataset") or "") for row in selected_rows)
    stats = {
        "record_count": len(selected_rows),
        "split_counts": dict(sorted(selected_split_counts.items())),
        "source_counts": dict(sorted(selected_source_counts.items())),
        "assigned_bucket_counts": dict(sorted(selected_bucket_counts.items())),
        "hard_example_count": sum(1 for row in selected_rows if bool(row.get("hard_example", False))),
        "negative_record_count": int(selected_bucket_counts.get(str(args.negative_bucket_label), 0)),
        "positive_record_count": len(selected_rows) - int(selected_bucket_counts.get(str(args.negative_bucket_label), 0)),
    }
    common.write_json(output_dir / "stats.json", stats)

    summary = {
        "source_manifest": str(source_manifest),
        "output_dir": str(output_dir),
        "requested_max_samples": int(args.max_samples),
        "selected_row_count": len(selected_rows),
        "seed": int(args.seed),
        "bucket_strategy": "rarest_member_issue_or_negative",
        "negative_bucket_label": str(args.negative_bucket_label),
        "available_membership_bucket_counts": dict(sorted(membership_counts.items())),
        "available_primary_bucket_counts": dict(sorted((bucket, len(items)) for bucket, items in assigned_rows.items())),
        "bucket_targets": dict(sorted(bucket_targets.items())),
        "selected_bucket_counts": dict(sorted(selected_bucket_counts.items())),
        "selected_split_counts": dict(sorted(selected_split_counts.items())),
        "selected_source_counts": dict(sorted(selected_source_counts.items())),
    }
    common.write_json(output_dir / "build_summary.json", summary)
    return summary


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    summary = build_subset(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
