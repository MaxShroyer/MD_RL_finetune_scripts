from __future__ import annotations

import unittest

from _DEPICATED_pokemon_cards import generate_pokemon_teacher_reasoning_openrouter as mod


class TeacherParsingTests(unittest.TestCase):
    def test_extract_openrouter_answer_text_from_string_content(self) -> None:
        payload = {
            "choices": [
                {
                    "message": {
                        "content": '{"answer":{"name":"Pikachu","hp":60,"set_name":"Base Set"},"rationale":"name=pikachu; hp=60; set=base set"}',
                        "reasoning": "hidden reasoning",
                    }
                }
            ]
        }
        self.assertIn('"answer"', mod.extract_openrouter_answer_text(payload))

    def test_extract_openrouter_answer_text_from_parts_array(self) -> None:
        payload = {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "text", "text": '{"answer":{"summary":"x"}'},
                            {"type": "text", "text": ',"rationale":"name=x; hp=1; set=y"}'},
                        ],
                        "reasoning_details": [{"type": "reasoning.text", "text": "details"}],
                    }
                }
            ]
        }
        text = mod.extract_openrouter_answer_text(payload)
        self.assertIn('"summary"', text)
        self.assertIn('"rationale"', text)

    def test_parse_teacher_payload_decodes_json_object(self) -> None:
        payload = mod._parse_teacher_payload(
            '{"answer":{"name":"Pikachu","hp":60,"set_name":"Base Set"},"rationale":"name=pikachu; hp=60; set=base set"}'
        )
        self.assertIsInstance(payload, dict)
        self.assertIn("answer", payload)
        self.assertIn("rationale", payload)


if __name__ == "__main__":
    unittest.main()

