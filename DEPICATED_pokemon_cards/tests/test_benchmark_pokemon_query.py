from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from DEPICATED_pokemon_cards import benchmark_pokemon_query as mod


class BenchmarkSmokeTests(unittest.TestCase):
    def test_resolve_api_key_prefers_named_env_var(self) -> None:
        args = mod._parse_args(
            [
                "--api-key-env-var",
                "CICID_GPUB_MOONDREAM_API_KEY_3",
                "--no-progress",
            ]
        )
        with mock.patch.dict(os.environ, {"CICID_GPUB_MOONDREAM_API_KEY_3": "secret-3"}, clear=False):
            self.assertEqual(mod._resolve_api_key(args), "secret-3")

    def test_run_benchmark_writes_prediction_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_dir = root / "dataset"
            images_dir = dataset_dir / "images"
            jsonl_dir = dataset_dir / "jsonl"
            images_dir.mkdir(parents=True)
            jsonl_dir.mkdir(parents=True)

            image_path = images_dir / "img.png"
            Image.new("RGB", (4, 4), (255, 0, 0)).save(image_path)

            row = {
                "row_id": "row-1",
                "split": "test",
                "task_type": "card_identity",
                "question": 'Return a JSON object with keys "name", "hp", and "set_name" for this Pokemon card.',
                "image_path": "images/img.png",
                "final_answer_json": json.dumps({"name": "Pikachu", "hp": 60, "set_name": "Base Set"}),
                "teacher_rationale_text": "name=pikachu; hp=60; set=base set",
                "teacher_model_meta_json": "{}",
                "source_metadata_json": "{}",
            }
            (jsonl_dir / "test.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

            args = mod._parse_args(
                [
                    "--dataset-dir",
                    str(dataset_dir),
                    "--split",
                    "test",
                    "--output-json",
                    str(root / "metrics.json"),
                    "--predictions-jsonl",
                    str(root / "predictions.jsonl"),
                    "--api-key",
                    "test-key",
                    "--model",
                    "moondream3-preview",
                    "--reasoning",
                    "--no-progress",
                ]
            )

            def fake_call_api_fn(**kwargs):
                return (
                    {
                        "answer": '{"name":"Pikachu","hp":60,"set_name":"Base Set"}',
                        "reasoning": {"text": "name=pikachu; hp=60; set=base set"},
                    },
                    12.5,
                )

            metrics = mod.run_benchmark(args=args, call_api_fn=fake_call_api_fn)
            self.assertEqual(metrics["samples"], 1)
            self.assertAlmostEqual(metrics["answer_reward_mean"], 1.0)
            self.assertAlmostEqual(metrics["rationale_reward_mean"], 1.0)
            self.assertAlmostEqual(metrics["combined_reward_mean"], 1.0)
            self.assertTrue((root / "metrics.json").exists())
            self.assertTrue((root / "predictions.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
