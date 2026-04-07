from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

from _DEPICATED_pokemon_cards import train_pokemon_query_rl as mod
from tuna_sdk import QueryOutput, QueryReasoningOutput


class TrainConfigTests(unittest.TestCase):
    def test_stage1_config_sets_no_reasoning(self) -> None:
        args = mod._parse_args(["--config", str((Path.cwd() / "pokemon_cards/configs/stage1_no_reasoning_cicd.json").resolve())])
        self.assertFalse(args.reasoning)
        self.assertFalse(args.eval_reasoning)

    def test_stage2_config_sets_reasoning(self) -> None:
        args = mod._parse_args(["--config", str((Path.cwd() / "pokemon_cards/configs/stage2_reasoning_distill_cicd.json").resolve())])
        self.assertTrue(args.reasoning)
        self.assertTrue(args.eval_reasoning)

    def test_rank32_rejects_lr_above_cap(self) -> None:
        args = SimpleNamespace(
            rank=32,
            lr=2e-5,
            reasoning=False,
            off_policy=False,
            checkpoint_avg_splits=["val"],
            task_sampling_weights={"card_identity": 1.0},
            batch_size=4,
            group_size=2,
            num_steps=10,
        )
        with self.assertRaisesRegex(ValueError, "lr <= 1e-5"):
            mod._validate_args(args)

    def test_reasoning_rejects_off_policy(self) -> None:
        args = SimpleNamespace(
            rank=16,
            lr=1e-5,
            reasoning=True,
            off_policy=True,
            checkpoint_avg_splits=["val"],
            task_sampling_weights={"card_identity": 1.0},
            batch_size=4,
            group_size=2,
            num_steps=10,
        )
        with self.assertRaisesRegex(ValueError, "off_policy=true"):
            mod._validate_args(args)


class RolloutScoringTests(unittest.TestCase):
    def _example(self, *, rationale: str) -> mod.QAExample:
        return mod.QAExample(
            row_id="row-1",
            split="train",
            task_type="card_identity",
            question="Return the card identity as JSON.",
            image_path=Path("/tmp/fake.png"),
            expected_answer={"name": "Pikachu", "hp": 60, "set_name": "Base Set"},
            teacher_rationale_text=rationale,
            source_metadata={},
        )

    def test_reasoning_reward_uses_reasoning_text(self) -> None:
        rollout = SimpleNamespace(
            output=QueryOutput(
                answer='{"name":"Pikachu","hp":60,"set_name":"Base Set"}',
                reasoning=QueryReasoningOutput(text="name=pikachu; hp=60; set=base set"),
            )
        )
        outcome = mod._score_rollout_for_example(
            rollout,
            self._example(rationale="name=pikachu; hp=60; set=base set"),
            use_reasoning_reward=True,
        )
        self.assertAlmostEqual(outcome["answer_reward"], 1.0)
        self.assertAlmostEqual(outcome["rationale_reward"], 1.0)
        self.assertAlmostEqual(outcome["reward"], 1.0)

    def test_missing_reasoning_text_gets_zero_rationale_reward(self) -> None:
        rollout = SimpleNamespace(
            output=QueryOutput(
                answer='{"name":"Pikachu","hp":60,"set_name":"Base Set"}',
                reasoning=None,
            )
        )
        outcome = mod._score_rollout_for_example(
            rollout,
            self._example(rationale="name=pikachu; hp=60; set=base set"),
            use_reasoning_reward=True,
        )
        self.assertAlmostEqual(outcome["answer_reward"], 1.0)
        self.assertAlmostEqual(outcome["rationale_reward"], 0.0)
        self.assertAlmostEqual(outcome["reward"], 0.8)


if __name__ == "__main__":
    unittest.main()

