from __future__ import annotations

import sys
from pathlib import Path

import pytest

TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from src.fen_decoder import decode_fen_board, parse_fen_filename, strip_optional_suffix


def test_strip_optional_suffix() -> None:
    assert strip_optional_suffix("rnbqkbnr-pppppppp-8-8-8-8-PPPPPPPP-RNBQKBNR_2") == (
        "rnbqkbnr-pppppppp-8-8-8-8-PPPPPPPP-RNBQKBNR"
    )
    assert strip_optional_suffix("8-8-8-8-8-8-8-8") == "8-8-8-8-8-8-8-8"


def test_decode_start_position() -> None:
    board = "rnbqkbnr-pppppppp-8-8-8-8-PPPPPPPP-RNBQKBNR"
    pieces = decode_fen_board(board)
    assert len(pieces) == 32

    piece_set = {(item["piece"], item["position"].get("square")) for item in pieces}
    assert ("white_king", "e1") in piece_set
    assert ("black_king", "e8") in piece_set
    assert ("white_queen", "d1") in piece_set
    assert ("black_queen", "d8") in piece_set


def test_parse_fen_filename_handles_suffix() -> None:
    fake_path = Path("/tmp/rnbqkbnr-pppppppp-8-8-8-8-PPPPPPPP-RNBQKBNR_2.JPG")
    record = parse_fen_filename(fake_path)
    assert record["source_dataset"] == "samryan18/chess-dataset"
    assert record["source_label_format"] == "fen_filename"
    assert len(record["pieces"]) == 32


def test_decode_fen_board_rejects_invalid() -> None:
    with pytest.raises(ValueError):
        decode_fen_board("8-8-8-8-8-8-8")  # 7 ranks
    with pytest.raises(ValueError):
        decode_fen_board("9-8-8-8-8-8-8-8")  # invalid run length
    with pytest.raises(ValueError):
        decode_fen_board("x7-8-8-8-8-8-8-8")  # invalid symbol
