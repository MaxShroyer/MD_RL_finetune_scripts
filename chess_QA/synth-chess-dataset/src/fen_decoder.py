"""Decode chess board labels embedded in filenames using FEN-like rank strings."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

FEN_SUFFIX_RE = re.compile(r"_(\d+)$")
PIECE_TYPE_BY_SYMBOL = {
    "k": "king",
    "q": "queen",
    "r": "rook",
    "b": "bishop",
    "n": "knight",
    "p": "pawn",
}
VALID_SYMBOLS = set(PIECE_TYPE_BY_SYMBOL)
FILES = "abcdefgh"


def _round6(value: float) -> float:
    return round(float(value), 6)


def strip_optional_suffix(stem: str) -> str:
    """Strip optional duplicate suffix from filename stem (for example `_2`)."""

    return FEN_SUFFIX_RE.sub("", stem)


def _square_from_row_col(row_idx: int, col_idx: int) -> str:
    file_char = FILES[col_idx]
    rank_num = 8 - row_idx
    return f"{file_char}{rank_num}"


def _position_from_row_col(row_idx: int, col_idx: int, *, include_square: bool) -> dict[str, Any]:
    x_min = col_idx / 8.0
    y_min = row_idx / 8.0
    x_max = (col_idx + 1) / 8.0
    y_max = (row_idx + 1) / 8.0
    pos: dict[str, Any] = {
        "x_center_norm": _round6((x_min + x_max) / 2.0),
        "y_center_norm": _round6((y_min + y_max) / 2.0),
        "bbox_norm": {
            "x_min": _round6(x_min),
            "y_min": _round6(y_min),
            "x_max": _round6(x_max),
            "y_max": _round6(y_max),
        },
    }
    if include_square:
        pos["square"] = _square_from_row_col(row_idx, col_idx)
    return pos


def decode_fen_board(fen_board: str, *, include_square: bool = True) -> list[dict[str, Any]]:
    """Decode a board string into piece records.

    Expected format: 8 ranks separated by `-`, each rank contains FEN symbols or
    digit run lengths.
    """

    ranks = fen_board.split("-")
    if len(ranks) != 8:
        raise ValueError(f"Expected 8 ranks in board label, got {len(ranks)}: {fen_board}")

    pieces: list[dict[str, Any]] = []
    for row_idx, rank in enumerate(ranks):
        col_idx = 0
        for char in rank:
            if char.isdigit():
                step = int(char)
                if step < 1 or step > 8:
                    raise ValueError(f"Invalid empty-square run '{char}' in rank '{rank}'")
                col_idx += step
                continue

            lower = char.lower()
            if lower not in VALID_SYMBOLS:
                raise ValueError(f"Invalid piece symbol '{char}' in rank '{rank}'")
            if col_idx >= 8:
                raise ValueError(f"Rank overflows board width in rank '{rank}'")

            color = "white" if char.isupper() else "black"
            piece_type = PIECE_TYPE_BY_SYMBOL[lower]
            piece_name = f"{color}_{piece_type}"
            position = _position_from_row_col(row_idx, col_idx, include_square=include_square)
            pieces.append({"piece": piece_name, "position": position})
            col_idx += 1

        if col_idx != 8:
            raise ValueError(f"Rank does not cover exactly 8 files: '{rank}'")

    return pieces


def parse_fen_filename(path: str | Path) -> dict[str, Any]:
    """Parse one labeled image path into a normalized source record."""

    image_path = Path(path)
    normalized_stem = strip_optional_suffix(image_path.stem)
    pieces = decode_fen_board(normalized_stem, include_square=True)
    return {
        "record_id": image_path.stem,
        "source_dataset": "samryan18/chess-dataset",
        "source_split": "original",
        "source_label_format": "fen_filename",
        "source_image_id": image_path.name,
        "source_image_path": str(image_path.resolve()),
        "pieces": pieces,
    }


def load_fen_records(dataset_dir: str | Path) -> list[dict[str, Any]]:
    """Load all labeled board images from dataset1 directory."""

    root = Path(dataset_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset1 directory not found: {root}")

    allowed_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in allowed_suffixes]
    files.sort(key=lambda p: p.name)
    return [parse_fen_filename(path) for path in files]
