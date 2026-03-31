from __future__ import annotations

import unittest

from DEPICATED_pokemon_cards import build_pokemon_cards_dataset as mod


class ParseCaptionTests(unittest.TestCase):
    def test_parse_caption_extracts_core_fields_and_attacks(self) -> None:
        parsed = mod.parse_caption(
            'A Stage 2 Pokemon Card of type Grass, Metal with the title "Beedrill δ" and 90 HP '
            'of rarity "Rare Holo" evolved from Kakuna from the set Delta Species. '
            'It has the attack "Super Slash" with the cost Grass, Metal, Colorless, the energy cost 3 and the damage of 50+. '
            'It has the attack "Final Sting" with the cost Grass, the energy cost 1.'
        )
        self.assertEqual(parsed.title, "Beedrill δ")
        self.assertEqual(parsed.hp, 90)
        self.assertEqual(parsed.set_name, "Delta Species")
        self.assertEqual(parsed.stage, "Stage 2")
        self.assertEqual(parsed.pokemon_types, ["Grass", "Metal"])
        self.assertEqual(parsed.rarity, "Rare Holo")
        self.assertEqual(parsed.evolves_from, "Kakuna")
        self.assertEqual(parsed.attack_names, ["Super Slash", "Final Sting"])

    def test_noise_gating_keeps_only_identity_task(self) -> None:
        record = mod.SourceRecord(
            source_id="xy2-1",
            image_url="https://example.test/image.png",
            image_relpath="images/xy2-1.png",
            image_sha1="abc123",
            name="Blastoise",
            hp=100,
            set_name="Base Set 2",
            parsed_caption=mod.ParsedCaption(
                title="Caterpie",
                hp=40,
                set_name="Flashfire",
                stage="Basic",
                pokemon_types=["Grass"],
                rarity="Common",
                evolves_from=None,
                attack_names=["Rain Dance"],
            ),
            noisy_identity=True,
            summary_text="name=Blastoise; hp=100; set=Base Set 2; stage=Basic; types=Grass; attacks=Rain Dance",
        )
        rows = mod.task_rows_for_record(record, split_name="train")
        self.assertEqual([row["task_type"] for row in rows], ["card_identity"])


if __name__ == "__main__":
    unittest.main()

