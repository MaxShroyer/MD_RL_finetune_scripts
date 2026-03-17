"""Decode chess piece annotations from the local osfstorage archive."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from src.fen_decoder import PIECE_TYPE_BY_SYMBOL

OSF_SPLITS = ("train", "val", "test")
VALID_PIECE_SYMBOLS = set(PIECE_TYPE_BY_SYMBOL)


def _round6(value: float) -> float:
    return round(float(value), 6)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def normalize_osf_piece_symbol(raw_symbol: str) -> str:
    symbol = str(raw_symbol or "").strip()
    if len(symbol) != 1:
        raise ValueError(f"Invalid OSF piece symbol: {raw_symbol!r}")

    lower = symbol.lower()
    if lower not in VALID_PIECE_SYMBOLS:
        raise ValueError(f"Unknown OSF piece symbol: {raw_symbol!r}")

    color = "white" if symbol.isupper() else "black"
    piece_type = PIECE_TYPE_BY_SYMBOL[lower]
    return f"{color}_{piece_type}"


def normalize_osf_bbox(
    box_xywh: list[float] | tuple[float, float, float, float],
    *,
    width: int,
    height: int,
) -> dict[str, Any] | None:
    if width <= 0 or height <= 0:
        return None
    if len(box_xywh) != 4:
        return None

    x, y, w, h = box_xywh
    x_min = _clamp(float(x), 0.0, float(width))
    y_min = _clamp(float(y), 0.0, float(height))
    x_max = _clamp(float(x) + float(w), 0.0, float(width))
    y_max = _clamp(float(y) + float(h), 0.0, float(height))

    if x_max <= x_min or y_max <= y_min:
        return None

    x_min_n = x_min / width
    y_min_n = y_min / height
    x_max_n = x_max / width
    y_max_n = y_max / height
    x_center_n = (x_min_n + x_max_n) / 2.0
    y_center_n = (y_min_n + y_max_n) / 2.0

    return {
        "x_center_norm": _round6(x_center_n),
        "y_center_norm": _round6(y_center_n),
        "bbox_norm": {
            "x_min": _round6(x_min_n),
            "y_min": _round6(y_min_n),
            "x_max": _round6(x_max_n),
            "y_max": _round6(y_max_n),
        },
    }


def _sort_piece_key(piece_entry: dict[str, Any]) -> tuple[str, str]:
    position = piece_entry.get("position", {})
    square = ""
    if isinstance(position, dict):
        square = str(position.get("square", "")).strip()
    return (str(piece_entry.get("piece", "")).strip(), square)


def _load_split_records(dataset_root: Path, split: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    split_root = dataset_root / split
    if not split_root.exists():
        raise FileNotFoundError(f"Missing OSF split directory: {split_root}")

    image_paths = {path.stem: path for path in sorted(split_root.glob("*.png"))}
    label_paths = {path.stem: path for path in sorted(split_root.glob("*.json"))}

    paired_stems = sorted(set(image_paths) & set(label_paths))
    summary = {
        "paired_records": len(paired_stems),
        "missing_image_pairs": len(set(label_paths) - set(image_paths)),
        "missing_json_pairs": len(set(image_paths) - set(label_paths)),
        "invalid_rows_skipped": 0,
    }

    records: list[dict[str, Any]] = []
    for stem in paired_stems:
        image_path = image_paths[stem]
        label_path = label_paths[stem]
        try:
            payload = json.loads(label_path.read_text(encoding="utf-8"))
            pieces_raw = payload["pieces"]
            if not isinstance(pieces_raw, list):
                raise ValueError("pieces must be a list")

            with Image.open(image_path) as image:
                width, height = image.size

            pieces: list[dict[str, Any]] = []
            for item in pieces_raw:
                if not isinstance(item, dict):
                    raise ValueError("piece entry must be an object")
                piece_name = normalize_osf_piece_symbol(str(item["piece"]))
                square = str(item["square"]).strip()
                position = normalize_osf_bbox(item["box"], width=width, height=height)
                if not square or position is None:
                    raise ValueError("piece entry must include valid square and box")
                position["square"] = square
                pieces.append({"piece": piece_name, "position": position})
        except (json.JSONDecodeError, KeyError, OSError, TypeError, ValueError):
            summary["invalid_rows_skipped"] += 1
            continue

        pieces.sort(key=_sort_piece_key)
        records.append(
            {
                "record_id": f"{split}:{stem}",
                "source_dataset": "osfstorage_archive",
                "source_split": split,
                "source_label_format": "osf_json_square_box",
                "source_image_id": image_path.name,
                "source_image_path": str(image_path.resolve()),
                "pieces": pieces,
            }
        )

    return records, summary


def load_osf_records(dataset_dir: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load and merge train/val/test OSF records from the local archive."""

    root = Path(dataset_dir)
    if not root.exists():
        raise FileNotFoundError(f"OSF dataset directory not found: {root}")

    records: list[dict[str, Any]] = []
    paired_records_by_split: dict[str, int] = {}
    missing_image_pairs_by_split: dict[str, int] = {}
    missing_json_pairs_by_split: dict[str, int] = {}
    invalid_rows_skipped_by_split: dict[str, int] = {}

    for split in OSF_SPLITS:
        split_records, split_summary = _load_split_records(root, split)
        records.extend(split_records)
        paired_records_by_split[split] = int(split_summary["paired_records"])
        missing_image_pairs_by_split[split] = int(split_summary["missing_image_pairs"])
        missing_json_pairs_by_split[split] = int(split_summary["missing_json_pairs"])
        invalid_rows_skipped_by_split[split] = int(split_summary["invalid_rows_skipped"])

    records.sort(key=lambda rec: (str(rec["source_split"]), str(rec["source_image_id"])))
    summary = {
        "paired_records_total": int(sum(paired_records_by_split.values())),
        "paired_records_by_split": paired_records_by_split,
        "missing_image_pairs_total": int(sum(missing_image_pairs_by_split.values())),
        "missing_image_pairs_by_split": missing_image_pairs_by_split,
        "missing_json_pairs_total": int(sum(missing_json_pairs_by_split.values())),
        "missing_json_pairs_by_split": missing_json_pairs_by_split,
        "invalid_rows_skipped_total": int(sum(invalid_rows_skipped_by_split.values())),
        "invalid_rows_skipped_by_split": invalid_rows_skipped_by_split,
    }
    return records, summary
