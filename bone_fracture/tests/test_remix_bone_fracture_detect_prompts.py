from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Value, load_from_disk

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bone_fracture import remix_bone_fracture_detect_prompts as mod


def _make_split(rows: list[dict[str, str]]) -> Dataset:
    features = Features(
        {
            "image": Value("string"),
            "answer_boxes": Value("string"),
            "class_name": Value("string"),
            "prompt": Value("string"),
            "task_schema": Value("string"),
            "source_collection": Value("string"),
            "source_dataset": Value("string"),
            "source_split": Value("string"),
            "source_image_id": Value("string"),
            "source_row_id": Value("string"),
            "source_box_index": Value("int32"),
            "source_element_index": Value("int32"),
            "source_base_id": Value("string"),
            "split_group_id": Value("string"),
        }
    )
    return Dataset.from_list(rows, features=features)


class RemixBoneFractureDetectPromptsTests(unittest.TestCase):
    def test_main_rewrites_prompts_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_dir = tmp_path / "input"
            output_dir = tmp_path / "output"

            row = {
                "image": "fake.jpg",
                "answer_boxes": '[{"x_min":0.1,"y_min":0.2,"x_max":0.3,"y_max":0.4,"attributes":[{"key":"element","value":"fracture"}]}]',
                "class_name": "fracture",
                "prompt": "fracture line",
                "task_schema": "per_box_element",
                "source_collection": "roboflow:test",
                "source_dataset": "roboflow:test:1",
                "source_split": "train",
                "source_image_id": "img-1",
                "source_row_id": "row-1",
                "source_box_index": 0,
                "source_element_index": 0,
                "source_base_id": "img-1",
                "split_group_id": "group:img-1",
            }
            dataset = DatasetDict(
                {
                    "train": _make_split([row, {**row, "source_row_id": "row-2", "source_image_id": "img-2"}]),
                    "validation": _make_split([{**row, "source_split": "validation", "source_row_id": "row-3"}]),
                    "test": _make_split([{**row, "source_split": "test", "source_row_id": "row-4"}]),
                }
            )
            dataset.save_to_disk(str(input_dir))

            source_metadata = {
                "class_catalog": [
                    {"class_name": "angle", "prompt": "bone angle marker"},
                    {"class_name": "fracture", "prompt": "fracture line"},
                ],
                "hub_repo_id": "",
            }
            source_stats = {"split_sizes": {"train": 2, "validation": 1, "test": 1}}
            (input_dir / "metadata.json").write_text(json.dumps(source_metadata), encoding="utf-8")
            (input_dir / "stats.json").write_text(json.dumps(source_stats), encoding="utf-8")

            variants = ["broken bone", "fractureed bone"]
            mod.main(
                [
                    "--input-dir",
                    str(input_dir),
                    "--output-dir",
                    str(output_dir),
                    "--seed",
                    "7",
                    "--prompt-variants-json",
                    json.dumps(variants),
                ]
            )

            rewritten = load_from_disk(str(output_dir))
            self.assertEqual(len(rewritten["train"]), 2)
            self.assertEqual(set(rewritten["train"]["prompt"]), set(variants))
            self.assertEqual(set(rewritten["validation"]["prompt"]), {"broken bone"})
            self.assertEqual(set(rewritten["test"]["prompt"]), {"broken bone"})

            metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["source_output_dir"], str(input_dir.resolve()))
            self.assertEqual(metadata["output_dir"], str(output_dir.resolve()))
            self.assertEqual(metadata["prompt_variants"], variants)
            self.assertEqual(metadata["class_catalog"][0]["prompt"], "broken bone")
            self.assertEqual(metadata["class_catalog"][1]["prompt"], "fractureed bone")

            stats = json.loads((output_dir / "stats.json").read_text(encoding="utf-8"))
            self.assertEqual(stats["prompt_counts"]["train"], {"broken bone": 1, "fractureed bone": 1})


if __name__ == "__main__":
    unittest.main()
