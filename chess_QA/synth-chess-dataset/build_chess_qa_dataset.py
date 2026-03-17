#!/usr/bin/env python3
"""Build synthetic chess QA datasets from FEN and COCO chess sources."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from math import floor
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
REPO_ROOT = SCRIPT_DIR.parent.parent

from src.coco_decoder import load_coco_records
from src.exporters import build_stats, write_hf_dataset, write_json, write_jsonl
from src.fen_decoder import load_fen_records
from src.prompts import choose_prompt
from src.tasks import (
    TASK_TYPES,
    build_balanced_mixed_task_counts,
    build_mixed_task_plan,
    build_task_answer,
    choose_deterministic_task_query,
)

SPLIT_ORDER = ("train", "val", "test")
DATASET1_TARGET_ROWS = 500
DATASET2_TARGET_ROWS = 289
V2_SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
V2_PIECE_POSITION_NAME = "piece_position_v2_dataset2"
V2_MIXED_NAME = "mixed_tasks_v2_dataset2"
OSF_PIECE_POSITION_NAME = "piece_position_v2_osfstorage"
OSF_MIXED_NAME = "mixed_tasks_v2_osfstorage"

TOTAL_ROWS_PER_DATASET = DATASET1_TARGET_ROWS + DATASET2_TARGET_ROWS

DATASET1_SPLIT_COUNTS = {"train": 400, "val": 50, "test": 50}
DATASET2_SPLIT_COUNTS = {"train": 231, "val": 29, "test": 29}
FINAL_SPLIT_COUNTS = {
    "train": DATASET1_SPLIT_COUNTS["train"] + DATASET2_SPLIT_COUNTS["train"],
    "val": DATASET1_SPLIT_COUNTS["val"] + DATASET2_SPLIT_COUNTS["val"],
    "test": DATASET1_SPLIT_COUNTS["test"] + DATASET2_SPLIT_COUNTS["test"],
}
FINAL_SOURCE_SPLIT_COUNTS = {
    "train": {"samryan18/chess-dataset": 400, "dataset2_coco": 231},
    "val": {"samryan18/chess-dataset": 50, "dataset2_coco": 29},
    "test": {"samryan18/chess-dataset": 50, "dataset2_coco": 29},
}


def _str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build synthetic chess QA datasets.")
    parser.add_argument(
        "--dataset1-dir",
        default="chess_QA/synth-chess-dataset/rawdatasets/chess-dataset/labeled_originals",
    )
    parser.add_argument(
        "--dataset2-dir",
        default="chess_QA/synth-chess-dataset/rawdatasets/Chess Pieces.v23-raw.coco",
    )
    parser.add_argument(
        "--osf-dir",
        default="chess_QA/synth-chess-dataset/rawdatasets/osfstorage-archive",
    )
    parser.add_argument("--output-dir", default="chess_QA/synth-chess-dataset/outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy-images", type=_str2bool, default=True)
    parser.add_argument("--build-legacy-v1", type=_str2bool, default=False)
    parser.add_argument("--build-source-specific-v2", type=_str2bool, default=False)
    parser.add_argument("--mixed-task-only", type=_str2bool, default=False)
    parser.add_argument("--piece-position-name", default="piece_position_v1")
    parser.add_argument("--mixed-name", default="mixed_tasks_v1")
    parser.add_argument("--piece-position-v2-name", default=V2_PIECE_POSITION_NAME)
    parser.add_argument("--mixed-v2-name", default=V2_MIXED_NAME)
    parser.add_argument("--piece-position-v2-osf-name", default=OSF_PIECE_POSITION_NAME)
    parser.add_argument("--mixed-v2-osf-name", default=OSF_MIXED_NAME)
    parser.add_argument("--export-hf-dataset", type=_str2bool, default=False)
    parser.add_argument("--push-to-hub", type=_str2bool, default=False)
    parser.add_argument("--hf-repo-id", default="")
    parser.add_argument("--hf-token", default="")
    parser.add_argument("--hf-private", type=_str2bool, default=False)
    return parser.parse_args()


def _record_key(record: dict[str, Any]) -> str:
    return f"{record['source_dataset']}::{record['record_id']}"


def _resolve_input_dir(raw_path: str) -> Path:
    path = Path(str(raw_path or "").strip()).expanduser()
    if path.is_absolute():
        return path.resolve()

    candidates = [
        (Path.cwd() / path).resolve(),
        (SCRIPT_DIR / path).resolve(),
        (REPO_ROOT / path).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _resolve_output_dir(raw_path: str) -> Path:
    path = Path(str(raw_path or "").strip()).expanduser()
    if path.is_absolute():
        return path.resolve()

    candidates = [
        (Path.cwd() / path).resolve(),
        (SCRIPT_DIR / path).resolve(),
        (REPO_ROOT / path).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _shuffle_records(records: list[dict[str, Any]], *, seed: int, tag: str) -> list[dict[str, Any]]:
    shuffled = list(records)
    random.Random(f"{seed}:{tag}:shuffle").shuffle(shuffled)
    return shuffled


def _split_counts_from_ratios(total_rows: int, *, ratios: dict[str, float]) -> dict[str, int]:
    if total_rows <= 0:
        raise ValueError(f"total_rows must be > 0, got {total_rows}")

    raw_counts = {split: float(ratios[split]) * float(total_rows) for split in SPLIT_ORDER}
    split_counts = {split: int(floor(raw_counts[split])) for split in SPLIT_ORDER}
    remainder = int(total_rows - sum(split_counts.values()))

    if remainder > 0:
        ranked_splits = sorted(
            SPLIT_ORDER,
            key=lambda split: (raw_counts[split] - split_counts[split], -SPLIT_ORDER.index(split)),
            reverse=True,
        )
        for split in ranked_splits[:remainder]:
            split_counts[split] += 1

    return split_counts


def _split_source_records(
    records: list[dict[str, Any]],
    split_counts: dict[str, int],
    *,
    seed: int,
    tag: str,
) -> dict[str, list[dict[str, Any]]]:
    expected = sum(split_counts.values())
    if len(records) != expected:
        raise ValueError(f"{tag}: expected {expected} records for splitting, got {len(records)}")
    shuffled = _shuffle_records(records, seed=seed, tag=f"{tag}:split")
    out: dict[str, list[dict[str, Any]]] = {}
    start = 0
    for split in SPLIT_ORDER:
        count = int(split_counts[split])
        out[split] = shuffled[start : start + count]
        start += count
    return out


def _piece_square_key(piece_entry: dict[str, Any]) -> tuple[str, str]:
    piece_name = str(piece_entry.get("piece", "")).strip()
    position = piece_entry.get("position", {})
    square = ""
    if isinstance(position, dict):
        square = str(position.get("square", "")).strip().lower()
    return piece_name, square


def _clean_piece_records(
    records: list[dict[str, Any]],
    *,
    source_filter: list[str],
    source_prefix: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cleaned_records: list[dict[str, Any]] = []
    duplicate_labels_removed = 0
    conflict_records_dropped = 0
    conflict_square_count = 0

    for record in records:
        seen_piece_square: set[tuple[str, str]] = set()
        deduped_pieces: list[dict[str, Any]] = []
        square_to_pieces: dict[str, set[str]] = defaultdict(set)

        for piece_entry in list(record.get("pieces", [])):
            piece_name, square = _piece_square_key(piece_entry)
            if not piece_name or not square:
                deduped_pieces.append(piece_entry)
                continue
            piece_square = (piece_name, square)
            if piece_square in seen_piece_square:
                duplicate_labels_removed += 1
                continue
            seen_piece_square.add(piece_square)
            deduped_pieces.append(piece_entry)
            square_to_pieces[square].add(piece_name)

        conflicting_squares = [square for square, pieces in square_to_pieces.items() if len(pieces) > 1]
        if conflicting_squares:
            conflict_records_dropped += 1
            conflict_square_count += len(conflicting_squares)
            continue

        cleaned_record = dict(record)
        cleaned_record["pieces"] = deduped_pieces
        cleaned_records.append(cleaned_record)

    cleaning_summary = {
        "source_filter": list(source_filter),
        f"{source_prefix}_pre_clean_records": len(records),
        f"{source_prefix}_post_clean_records": len(cleaned_records),
        "duplicate_piece_square_labels_removed": int(duplicate_labels_removed),
        "conflict_records_dropped": int(conflict_records_dropped),
        "conflict_square_count": int(conflict_square_count),
    }
    return cleaned_records, cleaning_summary


def _clean_dataset2_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return _clean_piece_records(
        records,
        source_filter=["dataset2_coco"],
        source_prefix="dataset2",
    )


def _clean_osf_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return _clean_piece_records(
        records,
        source_filter=["osfstorage_archive"],
        source_prefix="osf",
    )


def _split_records_by_source_split(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    split_records: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLIT_ORDER}
    for record in records:
        split = str(record.get("source_split", "")).strip().lower()
        if split not in split_records:
            raise ValueError(f"Unsupported source_split for record {record.get('record_id')}: {split!r}")
        split_records[split].append(record)

    for split in SPLIT_ORDER:
        split_records[split].sort(key=_record_key)
    return split_records


def _split_counts_from_split_records(split_records: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    return {split: len(split_records.get(split, [])) for split in SPLIT_ORDER}


def _copy_or_reference_assets(
    records: list[dict[str, Any]],
    *,
    output_dir: Path,
    copy_images: bool,
) -> dict[str, str]:
    out: dict[str, str] = {}
    images_root = output_dir / "imges"

    for record in sorted(records, key=lambda r: _record_key(r)):
        source_path = Path(record["source_image_path"])
        key = _record_key(record)
        if not source_path.exists():
            raise FileNotFoundError(f"Missing source image: {source_path}")

        if not copy_images:
            out[key] = str(source_path.resolve())
            continue

        source_dataset = str(record.get("source_dataset", ""))
        source_image_id = Path(str(record.get("source_image_id", ""))).name
        source_split = str(record.get("source_split", "")).strip().lower()
        if source_dataset == "samryan18/chess-dataset":
            dest_name = source_image_id
        elif source_dataset == "dataset2_coco":
            split_tag = source_split if source_split else "unknown"
            dest_name = f"rf__{split_tag}__{source_image_id}"
        elif source_dataset == "osfstorage_archive":
            split_tag = source_split if source_split else "unknown"
            dest_name = f"osf__{split_tag}__{source_image_id}"
        else:
            raise ValueError(f"Unknown source_dataset for asset copy: {source_dataset}")

        images_root.mkdir(parents=True, exist_ok=True)
        dest_path = images_root / dest_name
        if not dest_path.exists():
            shutil.copy2(source_path, dest_path)
        out[key] = f"/imges/{dest_name}"
    return out


def _cleanup_legacy_assets(output_dir: Path) -> None:
    """Remove stale image folders from older output layouts."""

    legacy_assets = output_dir / "assets"
    if legacy_assets.exists():
        shutil.rmtree(legacy_assets)


def _assert_image_paths_in_output(
    image_path_by_key: dict[str, str],
    *,
    output_dir: Path,
    copy_images: bool,
) -> None:
    if not copy_images:
        return
    images_root = (output_dir / "imges").resolve()
    for key, image_path in image_path_by_key.items():
        rel = str(image_path)
        if not rel.startswith("/imges/"):
            raise ValueError(f"Expected image_path to start with /imges/ for record {key}, got: {rel}")
        disk_path = images_root / rel.split("/imges/", 1)[1]
        if not disk_path.exists():
            raise ValueError(f"Expected copied image missing for record {key}: {disk_path}")


def _final_json(answer_obj: dict[str, Any]) -> str:
    return json.dumps(answer_obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _build_row(
    *,
    row_id: str,
    split: str,
    task_type: str,
    question: str,
    prompt_variant_id: str,
    answer_json: str,
    image_path: str,
    record: dict[str, Any],
    queried_piece: str | None = None,
) -> dict[str, Any]:
    image_name = Path(image_path).name
    row: dict[str, Any] = {
        "row_id": row_id,
        "split": split,
        "task_type": task_type,
        "source_dataset": record["source_dataset"],
        "image": image_name,
        "image_path": image_path,
        "question": question,
        "answer_text": answer_json,
        "final_answer_json": answer_json,
        "prompt_variant_id": prompt_variant_id,
        "source_image_id": record["source_image_id"],
        "source_label_format": record["source_label_format"],
    }
    if queried_piece is not None:
        row["queried_piece"] = queried_piece
    return row


def _build_piece_position_rows(
    split_records: dict[str, list[dict[str, Any]]],
    *,
    image_path_by_key: dict[str, str],
    seed: int,
    prompt_set: str = "v1",
    answer_version: str = "v1",
) -> dict[str, list[dict[str, Any]]]:
    rows_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLIT_ORDER}
    rng = random.Random(f"{seed}:piece_position_rows")

    for split in SPLIT_ORDER:
        for idx, record in enumerate(split_records[split]):
            answer_obj, _ = build_task_answer(
                "list_all_pieces",
                record["pieces"],
                rng=rng,
                answer_version=answer_version,
            )
            question, variant = choose_prompt("list_all_pieces", rng=rng, prompt_set=prompt_set)
            answer_json = _final_json(answer_obj)
            image_path = image_path_by_key[_record_key(record)]
            row = _build_row(
                row_id=f"{split}_{idx:06d}",
                split=split,
                task_type="list_all_pieces",
                question=question,
                prompt_variant_id=variant,
                answer_json=answer_json,
                image_path=image_path,
                record=record,
            )
            rows_by_split[split].append(row)
    return rows_by_split


def _flatten_split_records(split_records: dict[str, list[dict[str, Any]]]) -> list[tuple[str, dict[str, Any]]]:
    flat: list[tuple[str, dict[str, Any]]] = []
    for split in SPLIT_ORDER:
        for record in split_records[split]:
            flat.append((split, record))
    return flat


def _merge_split_records(
    split_records_by_source: list[tuple[str, dict[str, list[dict[str, Any]]]]],
    *,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    merged: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLIT_ORDER}
    for split in SPLIT_ORDER:
        combined: list[dict[str, Any]] = []
        for source_name, source_splits in split_records_by_source:
            records = list(source_splits.get(split, []))
            source_tag = f"{source_name}:{split}"
            combined.extend(_shuffle_records(records, seed=seed, tag=source_tag))
        merged[split] = _shuffle_records(combined, seed=seed, tag=f"merged:{split}")
    return merged


def _build_mixed_rows(
    split_records: dict[str, list[dict[str, Any]]],
    *,
    image_path_by_key: dict[str, str],
    seed: int,
    expected_task_counts: dict[str, int],
) -> dict[str, list[dict[str, Any]]]:
    rows_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLIT_ORDER}
    row_counters: Counter[str] = Counter()
    flat_records = _flatten_split_records(split_records)
    task_plan = build_mixed_task_plan(seed=seed, total_rows=len(flat_records), task_counts=expected_task_counts)
    rng = random.Random(f"{seed}:mixed_rows")

    for plan_idx, (split, record) in enumerate(flat_records):
        task_type = task_plan[plan_idx]
        answer_obj, queried_piece = build_task_answer(task_type, record["pieces"], rng=rng)
        question, variant = choose_prompt(task_type, rng=rng, queried_piece=queried_piece)
        answer_json = _final_json(answer_obj)
        image_path = image_path_by_key[_record_key(record)]
        row_id = f"{split}_{row_counters[split]:06d}"
        row_counters[split] += 1

        row = _build_row(
            row_id=row_id,
            split=split,
            task_type=task_type,
            question=question,
            prompt_variant_id=variant,
            answer_json=answer_json,
            image_path=image_path,
            record=record,
            queried_piece=queried_piece,
        )
        rows_by_split[split].append(row)
    return rows_by_split


def _build_one_to_many_mixed_rows(
    split_records: dict[str, list[dict[str, Any]]],
    *,
    image_path_by_key: dict[str, str],
    seed: int,
    prompt_set: str = "v2",
    answer_version: str = "v2",
) -> dict[str, list[dict[str, Any]]]:
    rows_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLIT_ORDER}
    row_counters: Counter[str] = Counter()

    for split in SPLIT_ORDER:
        for record in split_records[split]:
            record_key = _record_key(record)
            image_path = image_path_by_key[record_key]
            for task_type in TASK_TYPES:
                queried_piece = choose_deterministic_task_query(
                    task_type,
                    record["pieces"],
                    record_key=record_key,
                )
                rng = random.Random(f"{seed}:mixed_v2:{record_key}:{task_type}")
                answer_obj, _ = build_task_answer(
                    task_type,
                    record["pieces"],
                    rng=rng,
                    queried_piece=queried_piece,
                    answer_version=answer_version,
                )
                question, variant = choose_prompt(
                    task_type,
                    rng=rng,
                    queried_piece=queried_piece,
                    prompt_set=prompt_set,
                )
                answer_json = _final_json(answer_obj)
                row_id = f"{split}_{row_counters[split]:06d}"
                row_counters[split] += 1
                row = _build_row(
                    row_id=row_id,
                    split=split,
                    task_type=task_type,
                    question=question,
                    prompt_variant_id=variant,
                    answer_json=answer_json,
                    image_path=image_path,
                    record=record,
                    queried_piece=queried_piece,
                )
                rows_by_split[split].append(row)
    return rows_by_split


def _assert_split_counts(
    rows_by_split: dict[str, list[dict[str, Any]]],
    *,
    expected_split_counts: dict[str, int],
) -> None:
    for split in SPLIT_ORDER:
        expected = int(expected_split_counts[split])
        observed = len(rows_by_split.get(split, []))
        if observed != expected:
            raise ValueError(f"Split count mismatch for {split}: expected {expected}, got {observed}")


def _assert_source_balance(
    rows_by_split: dict[str, list[dict[str, Any]]],
    *,
    expected_source_split_counts: dict[str, dict[str, int]],
) -> None:
    for split in SPLIT_ORDER:
        expected_by_source = expected_source_split_counts[split]
        counts = Counter(str(row.get("source_dataset", "")) for row in rows_by_split.get(split, []))
        for source_name, expected_count in expected_by_source.items():
            observed = int(counts.get(source_name, 0))
            if observed != expected_count:
                raise ValueError(
                    f"Source balance mismatch split={split} source={source_name}: "
                    f"expected {expected_count}, got {observed}"
                )


def _assert_mixed_task_balance(
    rows_by_split: dict[str, list[dict[str, Any]]],
    *,
    expected_task_counts: dict[str, int],
) -> None:
    counts = Counter()
    for split in SPLIT_ORDER:
        counts.update(str(row.get("task_type", "")) for row in rows_by_split.get(split, []))
    for task_type, expected in expected_task_counts.items():
        observed = int(counts.get(task_type, 0))
        if observed != expected:
            raise ValueError(f"Mixed task count mismatch {task_type}: expected {expected}, got {observed}")


def _assert_rows_source_only(
    rows_by_split: dict[str, list[dict[str, Any]]],
    *,
    source_dataset: str,
) -> None:
    for split in SPLIT_ORDER:
        observed_sources = {
            str(row.get("source_dataset", "")) for row in rows_by_split.get(split, []) if isinstance(row, dict)
        }
        if observed_sources and observed_sources != {source_dataset}:
            raise ValueError(
                f"Expected split={split} to contain only source_dataset={source_dataset}, "
                f"got {sorted(observed_sources)}"
            )


def _assert_rows_sources_allowed(
    rows_by_split: dict[str, list[dict[str, Any]]],
    *,
    allowed_sources: set[str],
) -> None:
    for split in SPLIT_ORDER:
        observed_sources = {
            str(row.get("source_dataset", "")) for row in rows_by_split.get(split, []) if isinstance(row, dict)
        }
        unexpected = observed_sources - set(allowed_sources)
        if unexpected:
            raise ValueError(
                f"Expected split={split} to contain only sources in {sorted(allowed_sources)}, "
                f"got unexpected sources {sorted(unexpected)}"
            )


def _assert_one_to_many_mixed_task_balance(
    rows_by_split: dict[str, list[dict[str, Any]]],
    *,
    cleaned_split_counts: dict[str, int],
) -> None:
    for split in SPLIT_ORDER:
        rows = rows_by_split.get(split, [])
        expected_per_task = int(cleaned_split_counts[split])
        task_counts = Counter(str(row.get("task_type", "")) for row in rows)
        for task_type in TASK_TYPES:
            observed = int(task_counts.get(task_type, 0))
            if observed != expected_per_task:
                raise ValueError(
                    f"Expected split={split} task_type={task_type} to have {expected_per_task} rows, got {observed}"
                )


def _build_metadata(
    *,
    dataset_name: str,
    rows_by_split: dict[str, list[dict[str, Any]]],
    seed: int,
    jsonl_paths: dict[str, Path],
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_counts = Counter()
    task_counts = Counter()
    for split in SPLIT_ORDER:
        rows = rows_by_split.get(split, [])
        source_counts.update(str(row.get("source_dataset", "")) for row in rows)
        task_counts.update(str(row.get("task_type", "")) for row in rows)

    metadata = {
        "dataset_name": dataset_name,
        "seed": seed,
        "total_rows": sum(len(rows_by_split.get(split, [])) for split in SPLIT_ORDER),
        "split_counts": {split: len(rows_by_split.get(split, [])) for split in SPLIT_ORDER},
        "source_counts": dict(sorted(source_counts.items())),
        "task_counts": dict(sorted(task_counts.items())),
        "jsonl_paths": {split: str(path.resolve()) for split, path in jsonl_paths.items()},
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    return metadata


def _build_and_write_v2_family(
    *,
    output_dir: Path,
    split_records: dict[str, list[dict[str, Any]]],
    image_path_by_key: dict[str, str],
    seed: int,
    allowed_sources: set[str],
    piece_dataset_name: str,
    mixed_dataset_name: str,
    build_piece_position: bool,
    export_hf_dataset: bool,
    push_to_hub: bool,
    hf_repo_id: str,
    hf_token: str,
    hf_private: bool,
    extra_metadata: dict[str, Any] | None = None,
    extra_stats: dict[str, Any] | None = None,
) -> dict[str, int]:
    split_counts = _split_counts_from_split_records(split_records)
    mixed_rows = _build_one_to_many_mixed_rows(
        split_records,
        image_path_by_key=image_path_by_key,
        seed=seed,
        prompt_set="v2",
        answer_version="v2",
    )

    _assert_split_counts(
        mixed_rows,
        expected_split_counts={split: int(split_counts[split]) * len(TASK_TYPES) for split in SPLIT_ORDER},
    )
    _assert_rows_sources_allowed(mixed_rows, allowed_sources=allowed_sources)
    _assert_one_to_many_mixed_task_balance(mixed_rows, cleaned_split_counts=split_counts)

    if build_piece_position:
        piece_rows = _build_piece_position_rows(
            split_records,
            image_path_by_key=image_path_by_key,
            seed=seed,
            prompt_set="v2",
            answer_version="v2",
        )
        _assert_split_counts(piece_rows, expected_split_counts=split_counts)
        _assert_rows_sources_allowed(piece_rows, allowed_sources=allowed_sources)
        _write_dataset(
            output_dir=output_dir,
            dataset_name=piece_dataset_name,
            rows_by_split=piece_rows,
            seed=seed,
            export_hf_dataset=export_hf_dataset,
            push_to_hub=push_to_hub,
            hf_repo_id=hf_repo_id,
            hf_token=hf_token,
            hf_private=hf_private,
            extra_metadata=extra_metadata,
            extra_stats=extra_stats,
        )
    _write_dataset(
        output_dir=output_dir,
        dataset_name=mixed_dataset_name,
        rows_by_split=mixed_rows,
        seed=seed,
        export_hf_dataset=export_hf_dataset,
        push_to_hub=push_to_hub,
        hf_repo_id=hf_repo_id,
        hf_token=hf_token,
        hf_private=hf_private,
        extra_metadata=extra_metadata,
        extra_stats=extra_stats,
    )
    return split_counts


def _write_dataset(
    *,
    output_dir: Path,
    dataset_name: str,
    rows_by_split: dict[str, list[dict[str, Any]]],
    seed: int,
    export_hf_dataset: bool,
    push_to_hub: bool,
    hf_repo_id: str,
    hf_token: str,
    hf_private: bool,
    extra_metadata: dict[str, Any] | None = None,
    extra_stats: dict[str, Any] | None = None,
) -> None:
    dataset_root = output_dir / dataset_name
    jsonl_paths = write_jsonl(rows_by_split, dataset_root / "jsonl")
    _validate_jsonl_roundtrip(jsonl_paths)
    stats = build_stats(rows_by_split)
    if extra_stats:
        stats.update(extra_stats)
    metadata = _build_metadata(
        dataset_name=dataset_name,
        rows_by_split=rows_by_split,
        seed=seed,
        jsonl_paths=jsonl_paths,
        extra_metadata=extra_metadata,
    )
    if export_hf_dataset or push_to_hub:
        hf_out = write_hf_dataset(
            rows_by_split,
            dataset_root / "hf_dataset",
            push_to_hub=push_to_hub,
            repo_id=hf_repo_id,
            token=hf_token,
            private=hf_private,
            config_name=dataset_name,
        )
        metadata["hf_dataset_path"] = str(hf_out.resolve())
        metadata["hf_pushed"] = bool(push_to_hub)
        metadata["hf_repo_id"] = hf_repo_id if push_to_hub else ""
        metadata["hf_config_name"] = dataset_name if push_to_hub else ""
    write_json(dataset_root / "stats.json", stats)
    write_json(dataset_root / "metadata.json", metadata)


def _validate_jsonl_roundtrip(jsonl_paths: dict[str, Path]) -> None:
    for split, path in jsonl_paths.items():
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSONL row written for split={split} line={line_number}: {path}"
                    ) from exc
                if not isinstance(payload, dict):
                    raise ValueError(
                        f"Expected JSON object for split={split} line={line_number}: {path}"
                    )


def main() -> None:
    args = _parse_args()
    dataset1_dir = _resolve_input_dir(str(args.dataset1_dir))
    dataset2_dir = _resolve_input_dir(str(args.dataset2_dir))
    osf_dir = _resolve_input_dir(str(args.osf_dir))
    output_dir = _resolve_output_dir(str(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    _cleanup_legacy_assets(output_dir)
    hf_token = str(args.hf_token).strip() or os.environ.get("HF_TOKEN", "")
    if args.push_to_hub and not str(args.hf_repo_id).strip():
        raise ValueError("--hf-repo-id is required when --push-to-hub=true")

    build_legacy_v1_requested = bool(args.build_legacy_v1)
    build_source_specific_v2_requested = bool(args.build_source_specific_v2)
    mixed_task_only = bool(args.mixed_task_only)
    build_legacy_v1 = False
    build_legacy_v1_reason = ""
    dataset1_records_v1: list[dict[str, Any]] = []
    dataset2_records_v1: list[dict[str, Any]] = []
    dataset2_all_records: list[dict[str, Any]] = []
    cleaned_dataset2_records: list[dict[str, Any]] = []
    dataset2_cleaning: dict[str, Any] = {}
    dataset2_available = False
    dataset2_skip_reason = ""

    if dataset2_dir.exists():
        dataset2_all_records = load_coco_records(dataset2_dir)
        if not dataset2_all_records:
            dataset2_skip_reason = f"dataset2 contains no records: {dataset2_dir}"
        else:
            cleaned_dataset2_records, dataset2_cleaning = _clean_dataset2_records(dataset2_all_records)
            if not cleaned_dataset2_records:
                dataset2_skip_reason = f"dataset2 has no usable records after cleaning: {dataset2_dir}"
            else:
                dataset2_available = True
    else:
        dataset2_skip_reason = f"dataset2 directory not found: {dataset2_dir}"

    osf_all_records: list[dict[str, Any]] = []
    cleaned_osf_records: list[dict[str, Any]] = []
    osf_validation: dict[str, Any] = {}
    osf_available = False
    osf_skip_reason = ""

    if osf_dir.exists():
        try:
            from src.osf_decoder import load_osf_records
        except ModuleNotFoundError as exc:
            if exc.name == "PIL":
                raise ModuleNotFoundError(
                    "OSF dataset support requires Pillow. Install it with "
                    "`pip install Pillow` or "
                    "`pip install -r chess_QA/synth-chess-dataset/requirements.txt`."
                ) from exc
            raise

        osf_all_records, osf_summary = load_osf_records(osf_dir)
        if not osf_all_records:
            osf_skip_reason = f"osf archive contains no valid records: {osf_dir}"
            osf_validation = dict(osf_summary)
        else:
            cleaned_osf_records, osf_cleaning = _clean_osf_records(osf_all_records)
            osf_validation = dict(osf_summary)
            osf_validation.update(osf_cleaning)
            if not cleaned_osf_records:
                osf_skip_reason = f"osf archive has no usable records after cleaning: {osf_dir}"
            else:
                osf_available = True
    else:
        osf_skip_reason = f"osf archive directory not found: {osf_dir}"

    if build_legacy_v1_requested and dataset1_dir.exists() and dataset2_available:
        dataset1_records_v1 = load_fen_records(dataset1_dir)
        dataset1_records_v1 = _shuffle_records(dataset1_records_v1, seed=args.seed, tag="dataset1")
        dataset2_records_v1 = _shuffle_records(dataset2_all_records, seed=args.seed, tag="dataset2")
        if len(dataset1_records_v1) < DATASET1_TARGET_ROWS:
            build_legacy_v1_reason = (
                f"dataset1 has {len(dataset1_records_v1)} rows, needs at least {DATASET1_TARGET_ROWS}"
            )
        elif len(dataset2_records_v1) < DATASET2_TARGET_ROWS:
            build_legacy_v1_reason = (
                f"dataset2 has {len(dataset2_records_v1)} rows, needs at least {DATASET2_TARGET_ROWS} for v1"
            )
        else:
            build_legacy_v1 = True
            dataset1_records_v1 = dataset1_records_v1[:DATASET1_TARGET_ROWS]
            dataset2_records_v1 = dataset2_records_v1[:DATASET2_TARGET_ROWS]
    elif build_legacy_v1_requested and not dataset1_dir.exists():
        build_legacy_v1_reason = f"dataset1 directory not found: {dataset1_dir}"
    elif build_legacy_v1_requested:
        build_legacy_v1_reason = f"dataset2 unavailable for v1 build: {dataset2_skip_reason}"

    build_dataset2_v2 = bool(dataset2_available)
    build_osf_v2 = bool(osf_available)

    asset_records: list[dict[str, Any]] = []
    if build_legacy_v1:
        asset_records.extend(dataset1_records_v1)
        asset_records.extend(dataset2_records_v1)
    if build_dataset2_v2:
        asset_records.extend(cleaned_dataset2_records)
    if build_osf_v2:
        asset_records.extend(cleaned_osf_records)

    if not asset_records:
        reasons = [build_legacy_v1_reason, dataset2_skip_reason, osf_skip_reason]
        raise ValueError("No datasets built. " + " | ".join(reason for reason in reasons if reason))

    image_path_by_key = _copy_or_reference_assets(asset_records, output_dir=output_dir, copy_images=args.copy_images)
    _assert_image_paths_in_output(image_path_by_key, output_dir=output_dir, copy_images=args.copy_images)

    built_datasets: list[str] = []
    status_messages: list[str] = []

    if build_legacy_v1:
        dataset1_split_records = _split_source_records(
            dataset1_records_v1,
            DATASET1_SPLIT_COUNTS,
            seed=args.seed,
            tag="dataset1",
        )
        dataset2_split_records = _split_source_records(
            dataset2_records_v1,
            DATASET2_SPLIT_COUNTS,
            seed=args.seed,
            tag="dataset2",
        )
        split_records_v1 = _merge_split_records(
            [
                ("dataset1", dataset1_split_records),
                ("dataset2", dataset2_split_records),
            ],
            seed=args.seed,
        )

        expected_mixed_task_counts = build_balanced_mixed_task_counts(
            total_rows=len(_flatten_split_records(split_records_v1))
        )
        mixed_rows_v1 = _build_mixed_rows(
            split_records_v1,
            image_path_by_key=image_path_by_key,
            seed=args.seed,
            expected_task_counts=expected_mixed_task_counts,
        )

        _assert_split_counts(mixed_rows_v1, expected_split_counts=FINAL_SPLIT_COUNTS)
        _assert_source_balance(mixed_rows_v1, expected_source_split_counts=FINAL_SOURCE_SPLIT_COUNTS)
        _assert_mixed_task_balance(mixed_rows_v1, expected_task_counts=expected_mixed_task_counts)

        if not mixed_task_only:
            piece_rows_v1 = _build_piece_position_rows(
                split_records_v1,
                image_path_by_key=image_path_by_key,
                seed=args.seed,
            )
            _assert_split_counts(piece_rows_v1, expected_split_counts=FINAL_SPLIT_COUNTS)
            _assert_source_balance(piece_rows_v1, expected_source_split_counts=FINAL_SOURCE_SPLIT_COUNTS)
            _write_dataset(
                output_dir=output_dir,
                dataset_name=str(args.piece_position_name),
                rows_by_split=piece_rows_v1,
                seed=args.seed,
                export_hf_dataset=bool(args.export_hf_dataset),
                push_to_hub=bool(args.push_to_hub),
                hf_repo_id=str(args.hf_repo_id).strip(),
                hf_token=hf_token,
                hf_private=bool(args.hf_private),
            )
        _write_dataset(
            output_dir=output_dir,
            dataset_name=str(args.mixed_name),
            rows_by_split=mixed_rows_v1,
            seed=args.seed,
            export_hf_dataset=bool(args.export_hf_dataset),
            push_to_hub=bool(args.push_to_hub),
            hf_repo_id=str(args.hf_repo_id).strip(),
            hf_token=hf_token,
            hf_private=bool(args.hf_private),
        )
        if not mixed_task_only:
            built_datasets.append(str(args.piece_position_name))
        built_datasets.append(str(args.mixed_name))
    elif build_legacy_v1_requested:
        status_messages.append(f"Skipping legacy v1 build: {build_legacy_v1_reason}")

    dataset2_v2_split_counts: dict[str, int] | None = None
    merged_v2_source_records: list[tuple[str, dict[str, list[dict[str, Any]]]]] = []
    merged_v2_source_details: dict[str, Any] = {}
    if build_dataset2_v2:
        dataset2_v2_split_counts = _split_counts_from_ratios(
            len(cleaned_dataset2_records),
            ratios=V2_SPLIT_RATIOS,
        )
        split_records_v2 = _split_source_records(
            cleaned_dataset2_records,
            dataset2_v2_split_counts,
            seed=args.seed,
            tag="dataset2_v2",
        )
        merged_v2_source_records.append(("dataset2_coco", split_records_v2))
        merged_v2_source_details["dataset2_coco"] = {
            "split_counts": dict(dataset2_v2_split_counts),
            "split_strategy": {"type": "ratio", "ratios": dict(V2_SPLIT_RATIOS)},
            "cleaning": dict(dataset2_cleaning),
        }
    else:
        status_messages.append(f"Skipping dataset2 v2 build: {dataset2_skip_reason}")

    osf_split_counts: dict[str, int] | None = None
    if build_source_specific_v2_requested and build_osf_v2:
        split_records_osf = _split_records_by_source_split(cleaned_osf_records)
        osf_split_counts = _split_counts_from_split_records(split_records_osf)
        merged_v2_source_records.append(("osfstorage_archive", split_records_osf))
        merged_v2_source_details["osfstorage_archive"] = {
            "split_counts": dict(osf_split_counts),
            "split_strategy": {"type": "native_source_splits", "split_counts": dict(osf_split_counts)},
            "validation": dict(osf_validation),
        }
        osf_extra_metadata = {
            "validation": dict(osf_validation),
            "split_strategy": {"type": "native_source_splits", "split_counts": dict(osf_split_counts)},
        }
        osf_extra_stats = {"validation": dict(osf_validation)}
        _build_and_write_v2_family(
            output_dir=output_dir,
            split_records=split_records_osf,
            image_path_by_key=image_path_by_key,
            seed=args.seed,
            allowed_sources={"osfstorage_archive"},
            piece_dataset_name=str(args.piece_position_v2_osf_name),
            mixed_dataset_name=str(args.mixed_v2_osf_name),
            build_piece_position=not mixed_task_only,
            export_hf_dataset=bool(args.export_hf_dataset),
            push_to_hub=bool(args.push_to_hub),
            hf_repo_id=str(args.hf_repo_id).strip(),
            hf_token=hf_token,
            hf_private=bool(args.hf_private),
            extra_metadata=osf_extra_metadata,
            extra_stats=osf_extra_stats,
        )
        if not mixed_task_only:
            built_datasets.append(str(args.piece_position_v2_osf_name))
        built_datasets.append(str(args.mixed_v2_osf_name))
    elif build_osf_v2:
        split_records_osf = _split_records_by_source_split(cleaned_osf_records)
        osf_split_counts = _split_counts_from_split_records(split_records_osf)
        merged_v2_source_records.append(("osfstorage_archive", split_records_osf))
        merged_v2_source_details["osfstorage_archive"] = {
            "split_counts": dict(osf_split_counts),
            "split_strategy": {"type": "native_source_splits", "split_counts": dict(osf_split_counts)},
            "validation": dict(osf_validation),
        }
    elif build_source_specific_v2_requested:
        status_messages.append(f"Skipping OSF v2 build: {osf_skip_reason}")

    v2_split_counts: dict[str, int] | None = None
    if merged_v2_source_records:
        if len(merged_v2_source_records) == 1:
            merged_v2_split_records = merged_v2_source_records[0][1]
        else:
            merged_v2_split_records = _merge_split_records(merged_v2_source_records, seed=args.seed)

        merged_source_names = [name for name, _ in merged_v2_source_records]
        merged_v2_extra_metadata: dict[str, Any] = {
            "source_inputs": list(merged_source_names),
            "source_details": dict(merged_v2_source_details),
            "split_strategy": {
                "type": "merged_input_sources",
                "sources": {
                    name: dict(details.get("split_strategy", {}))
                    for name, details in merged_v2_source_details.items()
                },
            },
        }
        merged_v2_extra_stats: dict[str, Any] = {"source_details": dict(merged_v2_source_details)}
        if build_dataset2_v2:
            merged_v2_extra_metadata["cleaning"] = dict(dataset2_cleaning)
            merged_v2_extra_stats["cleaning"] = dict(dataset2_cleaning)
        if build_osf_v2:
            merged_v2_extra_metadata["validation"] = dict(osf_validation)
            merged_v2_extra_stats["validation"] = dict(osf_validation)

        v2_split_counts = _build_and_write_v2_family(
            output_dir=output_dir,
            split_records=merged_v2_split_records,
            image_path_by_key=image_path_by_key,
            seed=args.seed,
            allowed_sources=set(merged_source_names),
            piece_dataset_name=str(args.piece_position_v2_name),
            mixed_dataset_name=str(args.mixed_v2_name),
            build_piece_position=not mixed_task_only,
            export_hf_dataset=bool(args.export_hf_dataset),
            push_to_hub=bool(args.push_to_hub),
            hf_repo_id=str(args.hf_repo_id).strip(),
            hf_token=hf_token,
            hf_private=bool(args.hf_private),
            extra_metadata=merged_v2_extra_metadata,
            extra_stats=merged_v2_extra_stats,
        )
        if not mixed_task_only:
            built_datasets.append(str(args.piece_position_v2_name))
        built_datasets.append(str(args.mixed_v2_name))
    else:
        status_messages.append("Skipping merged v2 build: no v2-capable input datasets were available")

    print("Build complete.")
    print(f"- output_dir: {output_dir.resolve()}")
    print(f"- built_datasets: {built_datasets}")
    if build_legacy_v1:
        print(f"- legacy_v1_rows_per_dataset: {TOTAL_ROWS_PER_DATASET}")
        print(f"- legacy_v1_split_counts: {FINAL_SPLIT_COUNTS}")
        print(f"- legacy_v1_source_split_counts: {FINAL_SOURCE_SPLIT_COUNTS}")
    if dataset2_v2_split_counts is not None:
        print(f"- v2_cleaning: {dataset2_cleaning}")
        print(f"- dataset2_source_split_counts: {dataset2_v2_split_counts}")
        print(
            "- dataset2_source_mixed_task_counts: "
            + str({task: len(cleaned_dataset2_records) for task in TASK_TYPES})
        )
    if osf_split_counts is not None:
        print(f"- osf_validation: {osf_validation}")
        print(f"- osf_source_split_counts: {osf_split_counts}")
        print(
            "- osf_source_mixed_task_counts: "
            + str({task: len(cleaned_osf_records) for task in TASK_TYPES})
        )
    if v2_split_counts is not None:
        print(f"- merged_v2_split_counts: {v2_split_counts}")
        print(
            "- merged_v2_mixed_task_counts: "
            + str({task: sum(v2_split_counts.values()) for task in TASK_TYPES})
        )
        print(f"- merged_v2_sources: {sorted(merged_v2_source_details)}")
    for message in status_messages:
        print(f"- {message}")
    print(f"- copy_images: {bool(args.copy_images)}")
    print(f"- mixed_task_only: {mixed_task_only}")
    print(f"- export_hf_dataset: {bool(args.export_hf_dataset)}")
    print(f"- push_to_hub: {bool(args.push_to_hub)}")


if __name__ == "__main__":
    main()
