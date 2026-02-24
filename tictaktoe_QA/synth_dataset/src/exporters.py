"""Export helpers for JSONL and Hugging Face datasets."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, Features, Image as HFImage, Value

REQUIRED_COLUMNS = (
    "row_id",
    "image",
    "image_path",
    "split",
    "task_type",
    "question",
    "answer_text",
    "final_answer_json",
    "messages_json",
    "state_key",
    "symmetry_group",
    "player_to_move",
    "winner_label",
    "is_terminal",
    "legal_moves_json",
    "best_move_canonical_json",
    "best_move_optimal_set_json",
    "depth_complexity",
    "choice_complexity_num",
    "choice_complexity_den",
    "colorway",
    "augmentation_profile",
    "prompt_variant_id",
    "source_name",
    "rationale_source",
    "scores_by_move_json",
)


def _features() -> Features:
    return Features(
        {
            "row_id": Value("string"),
            "image": HFImage(),
            "image_path": Value("string"),
            "split": Value("string"),
            "task_type": Value("string"),
            "question": Value("string"),
            "answer_text": Value("string"),
            "final_answer_json": Value("string"),
            "messages_json": Value("string"),
            "state_key": Value("string"),
            "symmetry_group": Value("string"),
            "player_to_move": Value("string"),
            "winner_label": Value("string"),
            "is_terminal": Value("bool"),
            "legal_moves_json": Value("string"),
            "best_move_canonical_json": Value("string"),
            "best_move_optimal_set_json": Value("string"),
            "depth_complexity": Value("int32"),
            "choice_complexity_num": Value("int32"),
            "choice_complexity_den": Value("int32"),
            "colorway": Value("string"),
            "augmentation_profile": Value("string"),
            "prompt_variant_id": Value("string"),
            "source_name": Value("string"),
            "rationale_source": Value("string"),
            "scores_by_move_json": Value("string"),
        }
    )


def _ensure_columns(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    for col in REQUIRED_COLUMNS:
        if col not in out:
            raise ValueError(f"row missing required column: {col}")
    return out


def write_jsonl(rows_by_split: dict[str, list[dict[str, Any]]], out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_paths: dict[str, Path] = {}
    for split_name, rows in rows_by_split.items():
        path = out_dir / f"{split_name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                payload = _ensure_columns(row)
                payload["image"] = payload["image_path"]
                handle.write(json.dumps(payload, separators=(",", ":"), ensure_ascii=False))
                handle.write("\n")
        out_paths[split_name] = path
    return out_paths


def write_hf_dataset(rows_by_split: dict[str, list[dict[str, Any]]], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    features = _features()

    ds_map: dict[str, Dataset] = {}
    for split_name, rows in rows_by_split.items():
        normalized: list[dict[str, Any]] = []
        for row in rows:
            payload = _ensure_columns(row)
            item = dict(payload)
            item["image"] = payload["image_path"]
            normalized.append(item)
        ds_map[split_name] = Dataset.from_list(normalized, features=features)

    dsd = DatasetDict(ds_map)
    dsd.save_to_disk(str(out_dir))
    return out_dir


def build_stats(rows_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    split_stats: dict[str, Any] = {}
    for split_name, rows in rows_by_split.items():
        task_counts = Counter(row["task_type"] for row in rows)
        colorway_counts = Counter(row["colorway"] for row in rows)
        state_count = len({row["state_key"] for row in rows})
        group_count = len({row["symmetry_group"] for row in rows})
        split_stats[split_name] = {
            "rows": len(rows),
            "states": state_count,
            "symmetry_groups": group_count,
            "task_counts": dict(sorted(task_counts.items())),
            "colorway_counts": dict(sorted(colorway_counts.items())),
        }

    explicit_total = 0
    implicit_total = 0
    for rows in rows_by_split.values():
        for row in rows:
            pvid = row.get("prompt_variant_id", "")
            if ":explicit" in pvid:
                explicit_total += 1
            elif ":implicit" in pvid:
                implicit_total += 1

    return {
        "split_stats": split_stats,
        "explicit_prompt_rows": explicit_total,
        "implicit_prompt_rows": implicit_total,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
