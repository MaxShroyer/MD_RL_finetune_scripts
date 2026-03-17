from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image

TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from src.osf_decoder import load_osf_records, normalize_osf_bbox, normalize_osf_piece_symbol


def _write_png(path: Path, *, size: tuple[int, int] = (1200, 800)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(255, 255, 255)).save(path)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_normalize_osf_piece_symbol() -> None:
    assert normalize_osf_piece_symbol("K") == "white_king"
    assert normalize_osf_piece_symbol("q") == "black_queen"


def test_normalize_osf_bbox_clips_bounds() -> None:
    pos = normalize_osf_bbox([-10.0, -5.0, 25.0, 30.0], width=100, height=100)
    assert pos is not None
    assert pos["bbox_norm"]["x_min"] == 0.0
    assert pos["bbox_norm"]["y_min"] == 0.0
    assert pos["bbox_norm"]["x_max"] == 0.15
    assert pos["bbox_norm"]["y_max"] == 0.25


def test_load_osf_records_merges_splits_and_skips_invalid_rows(tmp_path: Path) -> None:
    for split in ("train", "val", "test"):
        (tmp_path / split).mkdir(parents=True, exist_ok=True)

    _write_png(tmp_path / "train" / "0000.png")
    _write_json(
        tmp_path / "train" / "0000.json",
        {
            "fen": "8/8/8/8/8/8/8/8",
            "pieces": [
                {"piece": "K", "square": "e1", "box": [10, 20, 30, 40]},
                {"piece": "k", "square": "e8", "box": [100, 120, 40, 50]},
            ],
        },
    )

    _write_png(tmp_path / "train" / "0001.png")
    _write_json(
        tmp_path / "train" / "0001.json",
        {
            "fen": "8/8/8/8/8/8/8/8",
            "pieces": [{"piece": "?", "square": "a1", "box": [0, 0, 10, 10]}],
        },
    )

    _write_json(
        tmp_path / "train" / "0002.json",
        {
            "fen": "8/8/8/8/8/8/8/8",
            "pieces": [{"piece": "Q", "square": "d1", "box": [0, 0, 10, 10]}],
        },
    )
    _write_png(tmp_path / "train" / "0003.png")

    records, summary = load_osf_records(tmp_path)

    assert len(records) == 1
    rec = records[0]
    assert rec["source_dataset"] == "osfstorage_archive"
    assert rec["source_split"] == "train"
    assert rec["source_label_format"] == "osf_json_square_box"
    assert rec["source_image_id"] == "0000.png"
    assert {piece["piece"] for piece in rec["pieces"]} == {"white_king", "black_king"}
    assert {piece["position"]["square"] for piece in rec["pieces"]} == {"e1", "e8"}
    assert summary["paired_records_by_split"] == {"train": 2, "val": 0, "test": 0}
    assert summary["missing_image_pairs_by_split"] == {"train": 1, "val": 0, "test": 0}
    assert summary["missing_json_pairs_by_split"] == {"train": 1, "val": 0, "test": 0}
    assert summary["invalid_rows_skipped_by_split"] == {"train": 1, "val": 0, "test": 0}
