from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from datasets import load_from_disk
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from football_detect import common
from football_detect import generate_football_splits as mod


def _row(row_id: str, note: str, class_names: list[str]) -> dict[str, object]:
    answer_boxes = []
    for idx, class_name in enumerate(class_names):
        answer_boxes.append(
            {
                "x_min": 10 + (idx * 5),
                "y_min": 8 + (idx * 5),
                "x_max": 30 + (idx * 5),
                "y_max": 24 + (idx * 5),
                "attributes": [{"key": "element", "value": class_name}],
            }
        )
    return {
        "row_id": row_id,
        "notes": note,
        "image": Image.new("RGB", (100, 80), color=(255, 255, 255)),
        "answer_boxes": json.dumps(answer_boxes),
    }


def _row_with_boxes(row_id: str, note: str, answer_boxes: list[dict[str, object]]) -> dict[str, object]:
    return {
        "row_id": row_id,
        "notes": note,
        "image": Image.new("RGB", (100, 80), color=(255, 255, 255)),
        "answer_boxes": json.dumps(answer_boxes),
    }


class NoteParsingTests(unittest.TestCase):
    def test_parse_note_bucket_accepts_only_close_mid_far(self) -> None:
        self.assertEqual(common.parse_note_bucket("close"), "close")
        self.assertEqual(common.parse_note_bucket(" Mid "), "mid")
        self.assertEqual(common.parse_note_bucket("FAR"), "far")
        with self.assertRaises(ValueError):
            common.parse_note_bucket("sideline")


class AllocationTests(unittest.TestCase):
    def test_largest_remainder_allocation_is_deterministic(self) -> None:
        quotas = {"close": 1.5, "mid": 1.5, "far": 1.0}
        first = common.largest_remainder_allocation(4, quotas, caps={"close": 2, "mid": 2, "far": 2}, order=common.NOTE_BUCKETS)
        second = common.largest_remainder_allocation(4, quotas, caps={"close": 2, "mid": 2, "far": 2}, order=common.NOTE_BUCKETS)
        self.assertEqual(first, second)
        self.assertEqual(first, {"close": 2, "mid": 1, "far": 1})


class SplitBuilderTests(unittest.TestCase):
    def _rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for idx in range(5):
            rows.append(_row(f"close_{idx}", "close", ["ball holder"]))
        for idx in range(4):
            rows.append(_row(f"mid_{idx}", "mid", ["tackle"]))
        for idx in range(3):
            rows.append(_row(f"far_{idx}", "far", ["players on the field"]))
        return rows

    def test_build_splits_preserves_rows_and_bucket_counts(self) -> None:
        rows = self._rows()
        splits, allocations, split_rows = mod.build_splits_from_rows(
            rows,
            seed=42,
            val_fraction=0.25,
            holdout_count=1,
        )

        original_ids = {str(row["row_id"]) for row in rows}
        split_ids = set()
        for split_name in ("train", "val", "post_val"):
            split_ids.update(str(row["row_id"]) for row in split_rows[split_name])
        self.assertEqual(split_ids, original_ids)
        self.assertEqual(
            len(split_rows["train"]) + len(split_rows["val"]) + len(split_rows["post_val"]),
            len(rows),
        )
        self.assertEqual(len(splits["train"]) + len(splits["val"]) + len(splits["post_val"]), len(rows))

        stats = common.build_split_stats(split_rows)
        expected_train_counts = allocations["train_counts"]
        expected_val_counts = {
            note: allocations["val_counts"][note] - allocations["post_val_counts"][note]
            for note in common.NOTE_BUCKETS
        }
        self.assertEqual(stats["note_bucket_counts"]["train"], expected_train_counts)
        self.assertEqual(stats["note_bucket_counts"]["val"], expected_val_counts)
        self.assertEqual(stats["note_bucket_counts"]["post_val"], allocations["post_val_counts"])

    def test_metadata_and_stats_files_report_correct_counts(self) -> None:
        rows = self._rows()
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "splits"
            with patch.object(mod, "_materialize_source_rows", return_value=(rows, None, "rev-123")):
                mod.main(
                    [
                        "--dataset",
                        "fake/football",
                        "--split",
                        "train",
                        "--val-fraction",
                        "0.25",
                        "--holdout-count",
                        "1",
                        "--output-dir",
                        str(output_dir),
                        "--push-to-hub",
                        "",
                    ]
                )

            metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
            stats = json.loads((output_dir / "stats.json").read_text(encoding="utf-8"))
            ds = load_from_disk(str(output_dir))

        self.assertEqual(metadata["dataset_name"], "fake/football")
        self.assertEqual(metadata["dataset_revision"], "rev-123")
        self.assertEqual(metadata["class_catalog"][0]["class_name"], "ball holder")
        self.assertEqual(metadata["class_catalog"][0]["prompt"], "ball carrier")
        self.assertEqual(metadata["split_sizes"], stats["split_sizes"])
        self.assertEqual(metadata["note_bucket_counts"], stats["note_bucket_counts"])
        self.assertEqual(metadata["class_counts"], stats["class_counts"])
        self.assertEqual(len(ds["train"]) + len(ds["val"]) + len(ds["post_val"]), len(rows))
        self.assertEqual(stats["class_counts"]["train"]["ball holder"], 4)
        self.assertEqual(stats["class_counts"]["val"]["players on the field"], 1)
        self.assertEqual(stats["split_sizes"]["post_val"], 1)

    def test_flatten_split_rows_duplicates_one_box_per_element(self) -> None:
        rows = [
            _row_with_boxes(
                "multi_0",
                "close",
                [
                    {
                        "x_min": 10,
                        "y_min": 8,
                        "x_max": 30,
                        "y_max": 24,
                        "attributes": [{"key": "element", "value": ["ball holder", "area of focus"]}],
                    }
                ],
            )
        ]

        flattened = mod._flatten_split_rows(rows, split_name="train")

        self.assertEqual(len(flattened), 2)
        self.assertEqual({row["class_name"] for row in flattened}, {"ball holder", "area of focus"})
        self.assertEqual({row["prompt"] for row in flattened}, {"ball carrier", "main action area"})
        self.assertEqual({row["source_row_id"] for row in flattened}, {"multi_0"})
        self.assertEqual({row["task_schema"] for row in flattened}, {"per_box_element"})
        self.assertEqual({int(row["source_box_index"]) for row in flattened}, {0})
        self.assertEqual({int(row["source_element_index"]) for row in flattened}, {0, 1})

        payloads = [json.loads(str(row["answer_boxes"])) for row in flattened]
        self.assertTrue(all(len(payload) == 1 for payload in payloads))
        self.assertEqual(
            {
                payload[0]["attributes"][0]["value"]
                for payload in payloads
            },
            {"ball holder", "area of focus"},
        )

    def test_flatten_split_rows_relabels_and_merges_line_boxes(self) -> None:
        rows = [
            _row_with_boxes(
                "line_0",
                "close",
                [
                    {
                        "x_min": 10,
                        "y_min": 10,
                        "x_max": 20,
                        "y_max": 20,
                        "attributes": [{"key": "element", "value": "offensive line"}],
                    },
                    {
                        "x_min": 30,
                        "y_min": 10,
                        "x_max": 40,
                        "y_max": 20,
                        "attributes": [{"key": "element", "value": "defensive line"}],
                    },
                    {
                        "x_min": 12,
                        "y_min": 30,
                        "x_max": 38,
                        "y_max": 40,
                        "attributes": [{"key": "element", "value": "line of scrimmage"}],
                    },
                ],
            )
        ]

        flattened = mod._flatten_split_rows(rows, split_name="train")
        class_names = sorted(str(row["class_name"]) for row in flattened)
        self.assertEqual(
            class_names,
            [
                "defensive line",
                "offensive line",
                "offensive line / defensive line",
                "offensive line / defensive line",
            ],
        )
        merged_rows = [row for row in flattened if int(row["source_box_index"]) == -1]
        self.assertEqual(len(merged_rows), 1)
        self.assertEqual(
            json.loads(str(merged_rows[0]["answer_boxes"])),
            [
                {
                    "x_min": 0.1,
                    "y_min": 0.125,
                    "x_max": 0.4,
                    "y_max": 0.25,
                    "attributes": [{"key": "element", "value": "offensive line / defensive line"}],
                }
            ],
        )

    def test_flattened_rows_from_one_source_stay_in_one_split(self) -> None:
        rows = self._rows()
        rows.append(
            _row_with_boxes(
                "multi_split_source",
                "close",
                [
                    {
                        "x_min": 10,
                        "y_min": 8,
                        "x_max": 30,
                        "y_max": 24,
                        "attributes": [{"key": "element", "value": ["ball holder", "area of focus"]}],
                    }
                ],
            )
        )

        _, _, split_rows = mod.build_splits_from_rows(
            rows,
            seed=42,
            val_fraction=0.25,
            holdout_count=1,
            flatten_by_box_element=True,
        )

        split_names = {
            split_name
            for split_name, rows_in_split in split_rows.items()
            for row in rows_in_split
            if str(row.get("source_row_id")) == "multi_split_source"
        }
        self.assertEqual(len(split_names), 1)
