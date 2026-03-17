from __future__ import annotations

import random
import sys
from pathlib import Path

TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from src.prompts import choose_prompt


def test_choose_prompt_v2_returns_canonical_presence_prompt() -> None:
    question, variant = choose_prompt(
        "color_presence_check",
        rng=random.Random(42),
        queried_piece="black",
        prompt_set="v2",
    )
    assert question == (
        'Is there any "black" piece on the board? Return JSON only as '
        '{"color":"black","present":true}.'
    )
    assert variant == "color_presence_check_v2_canonical"


def test_choose_prompt_v2_returns_canonical_count_prompt() -> None:
    question, variant = choose_prompt(
        "count_by_color",
        rng=random.Random(42),
        prompt_set="v2",
    )
    assert question == (
        "How many white pieces and black pieces are on the board? "
        'Return JSON only as {"white_piece_count":8,"black_piece_count":7}.'
    )
    assert variant == "count_by_color_v2_canonical"
