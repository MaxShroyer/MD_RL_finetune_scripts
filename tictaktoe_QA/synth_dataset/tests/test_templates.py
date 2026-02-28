from __future__ import annotations

import random
import unittest

from tictaktoe_QA.synth_dataset.src.label_engine import BoardRecord
from tictaktoe_QA.synth_dataset.src.templates import choose_prompt


def _record(*, player: str = "X") -> BoardRecord:
    return BoardRecord(
        state_key="123456789",
        state=(1, 2, 3, 4, 5, 6, 7, 8, 9),
        player_to_move=player,
        symmetry_group="g",
        depth_complexity=0,
        choice_complexity_num=0,
        choice_complexity_den=1,
        legal_moves=(1, 2, 3, 4, 5, 6, 7, 8, 9),
        winner_label="in_progress",
        is_terminal=False,
        immediate_winning_moves=tuple(),
        best_move_optimal_set=(1,),
        best_move_canonical=1,
        scores_by_move_json="{}",
    )


class TemplateTests(unittest.TestCase):
    def test_best_move_is_always_explicit(self) -> None:
        rec = _record(player="X")
        for explicit_override in (None, False, True):
            prompt = choose_prompt(
                task_type="best_move",
                record=rec,
                rng=random.Random(7),
                explicit_player_override=explicit_override,
            )
            self.assertTrue(prompt.explicit_player)
            self.assertIn(":explicit", prompt.prompt_variant_id)
            self.assertNotIn("infer whose turn", prompt.question.lower())
            self.assertNotIn("without being told", prompt.question.lower())

    def test_available_move_prompts_do_not_use_legal_wording(self) -> None:
        rec = _record(player="O")
        seen_questions: list[str] = []
        for task_type in ("available_moves_count", "available_moves_list", "legal_moves_count", "legal_moves_list"):
            for track in (None, "canonical", "paraphrase"):
                prompt = choose_prompt(
                    task_type=task_type,
                    record=rec,
                    rng=random.Random(11),
                    benchmark_track=track,
                )
                seen_questions.append(prompt.question.lower())
        self.assertTrue(seen_questions)
        self.assertTrue(all("legal move" not in q for q in seen_questions))

    def test_available_moves_list_prompts_require_json_schema(self) -> None:
        rec = _record(player="X")
        prompts = [
            choose_prompt(task_type="available_moves_list", record=rec, rng=random.Random(seed)).question
            for seed in range(1, 5)
        ]
        prompts.append(
            choose_prompt(
                task_type="available_moves_list",
                record=rec,
                rng=random.Random(99),
                benchmark_track="canonical",
            ).question
        )
        prompts.append(
            choose_prompt(
                task_type="available_moves_list",
                record=rec,
                rng=random.Random(100),
                benchmark_track="paraphrase",
            ).question
        )
        for question in prompts:
            lowered = question.lower()
            self.assertIn("available_moves", question)
            self.assertIn('"row"', lowered)
            self.assertIn('"col"', lowered)
            self.assertIn("json", lowered)


if __name__ == "__main__":
    unittest.main()
