"""Decode chess piece annotations from Roboflow COCO exports."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

COCO_SPLITS = ("train", "valid", "test")

CANONICAL_CATEGORY_MAP = {
    "white-king": "white_king",
    "white-queen": "white_queen",
    "white-rook": "white_rook",
    "white-bishop": "white_bishop",
    "white-knight": "white_knight",
    "white-pawn": "white_pawn",
    "black-king": "black_king",
    "black-queen": "black_queen",
    "black-rook": "black_rook",
    "black-bishop": "black_bishop",
    "black-knight": "black_knight",
    "black-pawn": "black_pawn",
}
FILES = "abcdefgh"


def _round6(value: float) -> float:
    return round(float(value), 6)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _square_from_center_norm(x_center_norm: float, y_center_norm: float) -> str:
    # Approximate board square from normalized center assuming a board-aligned image.
    x = _clamp(float(x_center_norm), 0.0, 1.0 - 1e-9)
    y = _clamp(float(y_center_norm), 0.0, 1.0 - 1e-9)
    col_idx = int(x * 8.0)
    row_idx = int(y * 8.0)
    return f"{FILES[col_idx]}{8 - row_idx}"


def normalize_category_name(raw_name: str) -> str | None:
    """Map COCO category names to canonical piece labels.

    Returns None for generic labels that should be ignored.
    """

    if raw_name in CANONICAL_CATEGORY_MAP:
        return CANONICAL_CATEGORY_MAP[raw_name]
    if raw_name == "bishop":
        return "unknown_bishop"
    if raw_name == "pieces":
        return None
    # Keep forward-compatible behavior for unexpected labels.
    sanitized = raw_name.strip().lower().replace("-", "_").replace(" ", "_")
    return f"unknown_{sanitized}" if sanitized else None


def normalize_coco_bbox(
    bbox_xywh: list[float] | tuple[float, float, float, float],
    width: int,
    height: int,
) -> dict[str, Any] | None:
    """Convert COCO `[x,y,w,h]` to normalized bbox + center, clipping to bounds."""

    if width <= 0 or height <= 0:
        return None
    if len(bbox_xywh) != 4:
        return None

    x, y, w, h = bbox_xywh
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
        "square": _square_from_center_norm(x_center_n, y_center_n),
        "bbox_norm": {
            "x_min": _round6(x_min_n),
            "y_min": _round6(y_min_n),
            "x_max": _round6(x_max_n),
            "y_max": _round6(y_max_n),
        },
    }


def _sort_piece_key(piece_entry: dict[str, Any]) -> tuple[Any, ...]:
    pos = piece_entry.get("position", {})
    return (
        piece_entry.get("piece", ""),
        float(pos.get("y_center_norm", 0.0)),
        float(pos.get("x_center_norm", 0.0)),
    )


def _load_split_records(dataset_root: Path, split: str) -> list[dict[str, Any]]:
    annotations_path = dataset_root / split / "_annotations.coco.json"
    if not annotations_path.exists():
        raise FileNotFoundError(f"Missing annotation file: {annotations_path}")

    data = json.loads(annotations_path.read_text(encoding="utf-8"))
    categories = {int(cat["id"]): str(cat["name"]) for cat in data.get("categories", [])}
    images = data.get("images", [])
    annotations = data.get("annotations", [])

    by_image_id: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in annotations:
        try:
            image_id = int(ann["image_id"])
        except (KeyError, TypeError, ValueError):
            continue
        by_image_id[image_id].append(ann)

    records: list[dict[str, Any]] = []
    for image in sorted(images, key=lambda item: str(item.get("file_name", ""))):
        try:
            image_id = int(image["id"])
            file_name = str(image["file_name"])
            width = int(image["width"])
            height = int(image["height"])
        except (KeyError, TypeError, ValueError):
            continue

        image_path = dataset_root / split / file_name
        pieces: list[dict[str, Any]] = []

        anns = by_image_id.get(image_id, [])
        anns_sorted = sorted(
            anns,
            key=lambda ann: (
                str(categories.get(int(ann.get("category_id", -1)), "")),
                float((ann.get("bbox") or [0.0, 0.0])[1]),
                float((ann.get("bbox") or [0.0, 0.0])[0]),
                int(ann.get("id", 0)) if isinstance(ann.get("id"), int) else 0,
            ),
        )

        for ann in anns_sorted:
            try:
                category_name = categories[int(ann["category_id"])]
                bbox = ann["bbox"]
            except (KeyError, TypeError, ValueError):
                continue

            piece_name = normalize_category_name(category_name)
            if piece_name is None:
                continue
            position = normalize_coco_bbox(bbox, width, height)
            if position is None:
                continue
            pieces.append({"piece": piece_name, "position": position})

        pieces.sort(key=_sort_piece_key)
        records.append(
            {
                "record_id": f"{split}:{file_name}",
                "source_dataset": "dataset2_coco",
                "source_split": split,
                "source_label_format": "coco_bbox",
                "source_image_id": file_name,
                "source_image_path": str(image_path.resolve()),
                "pieces": pieces,
            }
        )

    return records


def load_coco_records(dataset_dir: str | Path) -> list[dict[str, Any]]:
    """Load and merge train/valid/test COCO records from dataset2."""

    root = Path(dataset_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset2 directory not found: {root}")

    records: list[dict[str, Any]] = []
    for split in COCO_SPLITS:
        records.extend(_load_split_records(root, split))

    records.sort(key=lambda rec: (str(rec["source_split"]), str(rec["source_image_id"])))
    return records
