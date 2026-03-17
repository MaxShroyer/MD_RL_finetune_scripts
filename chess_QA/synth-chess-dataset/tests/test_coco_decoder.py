from __future__ import annotations

import json
import sys
from pathlib import Path

TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from src.coco_decoder import load_coco_records, normalize_category_name, normalize_coco_bbox


def _write_coco_split(path: Path, payload: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "_annotations.coco.json").write_text(json.dumps(payload), encoding="utf-8")


def test_normalize_category_name() -> None:
    assert normalize_category_name("white-king") == "white_king"
    assert normalize_category_name("bishop") == "unknown_bishop"
    assert normalize_category_name("pieces") is None


def test_normalize_coco_bbox_clips_bounds() -> None:
    pos = normalize_coco_bbox([-10.0, -5.0, 25.0, 30.0], width=100, height=100)
    assert pos is not None
    assert pos["bbox_norm"]["x_min"] == 0.0
    assert pos["bbox_norm"]["y_min"] == 0.0
    assert pos["bbox_norm"]["x_max"] == 0.15
    assert pos["bbox_norm"]["y_max"] == 0.25
    assert pos["square"] == "a7"


def test_load_coco_records_merges_splits_and_handles_bishop(tmp_path: Path) -> None:
    categories = [
        {"id": 1, "name": "white-king"},
        {"id": 2, "name": "bishop"},
        {"id": 3, "name": "pieces"},
    ]
    train_payload = {
        "images": [{"id": 10, "file_name": "img1.jpg", "width": 100, "height": 100}],
        "annotations": [
            {"id": 1, "image_id": 10, "category_id": 1, "bbox": [10, 20, 30, 40]},
            {"id": 2, "image_id": 10, "category_id": 2, "bbox": [-10, -5, 20, 30]},
            {"id": 3, "image_id": 10, "category_id": 3, "bbox": [0, 0, 10, 10]},
        ],
        "categories": categories,
    }
    empty_payload = {"images": [], "annotations": [], "categories": categories}

    _write_coco_split(tmp_path / "train", train_payload)
    _write_coco_split(tmp_path / "valid", empty_payload)
    _write_coco_split(tmp_path / "test", empty_payload)

    records = load_coco_records(tmp_path)
    assert len(records) == 1
    rec = records[0]
    assert rec["source_dataset"] == "dataset2_coco"
    assert rec["source_split"] == "train"
    names = [piece["piece"] for piece in rec["pieces"]]
    assert "white_king" in names
    assert "unknown_bishop" in names
    # "pieces" category should be dropped
    assert "unknown_pieces" not in names
