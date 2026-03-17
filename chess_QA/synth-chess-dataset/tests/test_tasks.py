from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from src.tasks import (
    build_balanced_mixed_task_counts,
    build_mixed_task_plan,
    build_task_answer,
    choose_deterministic_task_query,
)


def _sample_pieces() -> list[dict]:
    return [
        {
            "piece": "white_king",
            "position": {
                "square": "e1",
                "x_center_norm": 0.5625,
                "y_center_norm": 0.9375,
                "bbox_norm": {"x_min": 0.5, "y_min": 0.875, "x_max": 0.625, "y_max": 1.0},
            },
        },
        {
            "piece": "black_king",
            "position": {
                "square": "e8",
                "x_center_norm": 0.5625,
                "y_center_norm": 0.0625,
                "bbox_norm": {"x_min": 0.5, "y_min": 0.0, "x_max": 0.625, "y_max": 0.125},
            },
        },
        {
            "piece": "white_knight",
            "position": {
                "square": "g1",
                "x_center_norm": 0.8125,
                "y_center_norm": 0.9375,
                "bbox_norm": {"x_min": 0.75, "y_min": 0.875, "x_max": 0.875, "y_max": 1.0},
            },
        },
    ]


def test_build_task_answers_json_parseable() -> None:
    pieces = _sample_pieces()
    rng = random.Random(42)
    for task in ("list_all_pieces", "count_by_color", "list_color_pieces", "color_presence_check"):
        answer_obj, queried = build_task_answer(task, pieces, rng=rng)
        payload = json.loads(json.dumps(answer_obj, sort_keys=True, separators=(",", ":")))
        assert isinstance(payload, dict)
        if task == "list_all_pieces":
            assert payload["pieces"]
            assert isinstance(payload["pieces"], list)
            assert isinstance(payload["pieces"][0], dict)
            values = list(payload["pieces"][0].values())
            assert all(isinstance(v, (str, list)) for v in values)
        if task == "count_by_color":
            assert "white_piece_count" in payload
            assert "black_piece_count" in payload
        if task == "list_color_pieces":
            assert payload["color"] in {"white", "black"}
            assert isinstance(payload["pieces"], list)
            assert isinstance(payload["pieces"][0], dict)
            assert all(key.startswith(payload["color"] + "_") for key in payload["pieces"][0].keys())
        if task == "color_presence_check":
            assert payload["color"] in {"white", "black"}
            assert isinstance(payload["present"], bool)
            assert isinstance(payload["count"], int)
        if task in {"list_color_pieces", "color_presence_check"}:
            assert queried is not None


def test_build_task_answers_v2_presence_omits_count() -> None:
    pieces = _sample_pieces()
    answer_obj, queried = build_task_answer(
        "color_presence_check",
        pieces,
        rng=random.Random(7),
        answer_version="v2",
    )
    assert queried in {"white", "black"}
    assert answer_obj["color"] in {"white", "black"}
    assert isinstance(answer_obj["present"], bool)
    assert "count" not in answer_obj


def test_choose_deterministic_task_query_is_stable() -> None:
    pieces = _sample_pieces()
    query_a = choose_deterministic_task_query(
        "list_color_pieces",
        pieces,
        record_key="dataset2_coco::train:board.jpg",
    )
    query_b = choose_deterministic_task_query(
        "list_color_pieces",
        pieces,
        record_key="dataset2_coco::train:board.jpg",
    )
    assert query_a == query_b


def test_mixed_task_plan_counts() -> None:
    expected = build_balanced_mixed_task_counts(500)
    plan = build_mixed_task_plan(seed=42, total_rows=500, task_counts=expected)
    assert len(plan) == 500
    counts = Counter(plan)
    assert dict(counts) == expected
