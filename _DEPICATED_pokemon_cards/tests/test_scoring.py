from __future__ import annotations

import unittest

from _DEPICATED_pokemon_cards import scoring
from _DEPICATED_pokemon_cards import task_schema


class SummaryScoringTests(unittest.TestCase):
    def test_card_summary_reward_allows_natural_paraphrase_against_legacy_label(self) -> None:
        expected_payload = {
            "summary": "name=Scyther; hp=60; set=FireRed & LeafGreen; stage=Basic; types=Grass; attacks=Fury Cutter"
        }
        pred_payload = {
            "summary": "Scyther is a Basic Pokemon card from FireRed & LeafGreen with 60 HP. Its type is Grass and it has the attack Fury Cutter."
        }
        reward = scoring.answer_reward_for_task("card_summary", pred_payload, expected_payload)
        self.assertGreaterEqual(reward, 0.8)

    def test_card_summary_reward_penalizes_missing_key_facts(self) -> None:
        expected_payload = {
            "summary": "name=Scyther; hp=60; set=FireRed & LeafGreen; stage=Basic; types=Grass; attacks=Fury Cutter"
        }
        pred_payload = {
            "summary": "This is a Pokemon card with an attack."
        }
        reward = scoring.answer_reward_for_task("card_summary", pred_payload, expected_payload)
        self.assertLess(reward, 0.4)

    def test_summary_rationale_slots_support_structured_payload(self) -> None:
        slots = task_schema.rationale_slots_from_answer(
            "card_summary",
            {
                "name": "Scyther",
                "hp": 60,
                "set_name": "FireRed & LeafGreen",
                "stage": "Basic",
                "pokemon_types": ["Grass"],
                "attack_names": ["Fury Cutter"],
            },
        )
        self.assertEqual(
            slots,
            {
                "name": "scyther",
                "hp": "60",
                "set": "firered & leafgreen",
                "stage": "basic",
                "types": "grass",
                "attacks": "fury cutter",
            },
        )


if __name__ == "__main__":
    unittest.main()
