from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from datasets import load_from_disk
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MDpi_and_d import train_pid_icons as shared_train
from aerial_airport import benchmark_aerial_airport_detect as bench_detect_mod
from aerial_airport import benchmark_aerial_airport_point as bench_mod
from aerial_airport import build_aerial_airport_hf_dataset as build_mod
from aerial_airport import runtime_tiling as tiling_mod
from aerial_airport import train_aerial_airport_detect as train_detect_mod
from aerial_airport import train_aerial_airport_point as train_mod
from aerial_airport.common import DEFAULT_CLASS_NAME, DEFAULT_CLASS_UID, DEFAULT_STAGING_API_BASE


def _write_image(path: Path, *, size: tuple[int, int] = (100, 100), color: tuple[int, int, int] = (255, 255, 255)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path)


def _write_coco_annotations(
    path: Path,
    *,
    images: list[dict[str, object]],
    annotations: list[dict[str, object]],
) -> None:
    payload = {
        "info": {},
        "licenses": [],
        "categories": [
            {"id": 0, "name": "planes", "supercategory": "none"},
            {"id": 1, "name": "airplane", "supercategory": "planes"},
        ],
        "images": images,
        "annotations": annotations,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _build_coco_fixture(
    root: Path,
    *,
    split_names: tuple[str, ...],
    include_empty_rows: bool,
    positive_count: int = 1,
    empty_count: int | None = None,
    positive_bboxes: list[list[float]] | None = None,
) -> Path:
    for split in split_names:
        split_dir = root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        images: list[dict[str, object]] = []
        annotations: list[dict[str, object]] = []
        split_empty_count = empty_count if empty_count is not None else (1 if include_empty_rows else 0)
        bboxes = positive_bboxes if positive_bboxes is not None else [[5 + (index * 2), 5, 30, 30] for index in range(positive_count)]

        next_image_id = 1
        next_annotation_id = 1
        for index, bbox in enumerate(bboxes):
            positive_name = f"{split}_positive_{index}.jpg"
            _write_image(split_dir / positive_name)
            images.append(
                {
                    "id": next_image_id,
                    "file_name": positive_name,
                    "width": 100,
                    "height": 100,
                    "extra": {"name": f"{split}_positive_{index}.jpg"},
                }
            )
            annotations.append(
                {
                    "id": next_annotation_id,
                    "image_id": next_image_id,
                    "category_id": 1,
                    "bbox": bbox,
                    "area": 900,
                    "segmentation": [],
                    "iscrowd": 0,
                }
            )
            next_image_id += 1
            next_annotation_id += 1

        for empty_index in range(split_empty_count):
            empty_name = f"{split}_empty_{empty_index}.jpg"
            _write_image(split_dir / empty_name, color=(240, 240, 240))
            images.append(
                {
                    "id": next_image_id,
                    "file_name": empty_name,
                    "width": 100,
                    "height": 100,
                    "extra": {"name": empty_name},
                }
            )
            next_image_id += 1

        _write_coco_annotations(split_dir / "_annotations.coco.json", images=images, annotations=annotations)

    return root


def _empty_fraction(rows: list[dict[str, object]]) -> float:
    if not rows:
        return 0.0
    empty_count = sum(1 for row in rows if int(row["class_count"]) == 0)
    return empty_count / float(len(rows))


class BuilderTests(unittest.TestCase):
    def test_build_dataset_preserves_explicit_split_names_and_empty_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw_root = _build_coco_fixture(
                Path(tmp) / "raw_dataset" / "Aerial Airport.coco",
                split_names=("train", "valid", "test"),
                include_empty_rows=True,
            )
            dataset_dict, split_rows, synthetic_counts, source_split_names, raw_empty_row_counts = build_mod.build_dataset_dict_from_raw_dir(
                raw_root,
                output_dir=Path(tmp) / "out",
                seed=42,
                val_fraction=0.1,
                test_fraction=0.1,
                target_empty_fraction=0.1,
            )

        self.assertEqual(sorted(dataset_dict.keys()), ["test", "train", "validation"])
        self.assertEqual(source_split_names, ["test", "train", "valid"])
        self.assertEqual(len(dataset_dict["train"]), 2)
        self.assertEqual(len(dataset_dict["validation"]), 2)
        self.assertEqual(len(dataset_dict["test"]), 2)
        self.assertEqual(synthetic_counts, {"train": 0, "validation": 0, "test": 0})
        self.assertEqual(raw_empty_row_counts, {"train": 1, "validation": 1, "test": 1})

        train_empty = next(row for row in split_rows["train"] if int(row["class_count"]) == 0)
        valid_empty = next(row for row in split_rows["validation"] if int(row["class_count"]) == 0)
        train_positive = next(row for row in split_rows["train"] if int(row["class_count"]) > 0)

        self.assertEqual(json.loads(str(train_empty["answer_boxes"])), [])
        self.assertEqual(json.loads(str(valid_empty["answer_boxes"])), [])
        positive_boxes = json.loads(str(train_positive["answer_boxes"]))
        self.assertEqual(positive_boxes[0]["class_name"], DEFAULT_CLASS_NAME)
        self.assertEqual(positive_boxes[0]["class_uid"], DEFAULT_CLASS_UID)

    def test_tiling_none_preserves_image_level_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "out"
            raw_root = _build_coco_fixture(
                Path(tmp) / "raw_dataset" / "Aerial Airport.coco",
                split_names=("train", "valid", "test"),
                include_empty_rows=False,
            )
            dataset_dict, split_rows, synthetic_counts, _, raw_empty_row_counts = build_mod.build_dataset_dict_from_raw_dir(
                raw_root,
                output_dir=output_dir,
                seed=42,
                val_fraction=0.1,
                test_fraction=0.1,
                target_empty_fraction=0.0,
                tiling="none",
            )

            self.assertFalse((output_dir / "tiles").exists())

        self.assertEqual(len(dataset_dict["train"]), 1)
        self.assertEqual(len(dataset_dict["validation"]), 1)
        self.assertEqual(len(dataset_dict["test"]), 1)
        self.assertEqual(raw_empty_row_counts, {"train": 0, "validation": 0, "test": 0})
        self.assertEqual(synthetic_counts, {"train": 0, "validation": 0, "test": 0})
        self.assertTrue(all("__tile_" not in str(row["source_image_id"]) for row in split_rows["train"]))

    def test_tiling_2x2_emits_all_tiles_keeps_empty_tiles_and_renormalizes_boxes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "out"
            raw_root = _build_coco_fixture(
                Path(tmp) / "raw_dataset" / "Aerial Airport.coco",
                split_names=("train", "valid", "test"),
                include_empty_rows=False,
                positive_bboxes=[[10, 10, 10, 10]],
            )
            dataset_dict, split_rows, synthetic_counts, _, raw_empty_row_counts = build_mod.build_dataset_dict_from_raw_dir(
                raw_root,
                output_dir=output_dir,
                seed=42,
                val_fraction=0.1,
                test_fraction=0.1,
                target_empty_fraction=0.0,
                tiling="2x2",
            )

            self.assertEqual(len(list((output_dir / "tiles" / "train").glob("*"))), 4)

        self.assertEqual(len(dataset_dict["train"]), 4)
        self.assertEqual(len(dataset_dict["validation"]), 4)
        self.assertEqual(len(dataset_dict["test"]), 4)
        self.assertEqual(raw_empty_row_counts, {"train": 3, "validation": 3, "test": 3})
        self.assertEqual(synthetic_counts, {"train": 0, "validation": 0, "test": 0})
        self.assertEqual(sum(int(row["class_count"]) == 0 for row in split_rows["train"]), 3)

        positive_tile = next(row for row in split_rows["train"] if int(row["class_count"]) > 0)
        self.assertEqual(str(positive_tile["source_image_id"]), "train_positive_0__tile_r0_c0")
        positive_box = json.loads(str(positive_tile["answer_boxes"]))[0]
        self.assertAlmostEqual(positive_box["x_min"], 0.2, places=8)
        self.assertAlmostEqual(positive_box["y_min"], 0.2, places=8)
        self.assertAlmostEqual(positive_box["x_max"], 0.4, places=8)
        self.assertAlmostEqual(positive_box["y_max"], 0.4, places=8)

    def test_tiling_2x2_clips_boundary_crossing_boxes_into_each_overlapping_tile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw_root = _build_coco_fixture(
                Path(tmp) / "raw_dataset" / "Aerial Airport.coco",
                split_names=("train", "valid", "test"),
                include_empty_rows=False,
                positive_bboxes=[[40, 20, 20, 20]],
            )
            _, split_rows, synthetic_counts, _, raw_empty_row_counts = build_mod.build_dataset_dict_from_raw_dir(
                raw_root,
                output_dir=Path(tmp) / "out",
                seed=42,
                val_fraction=0.1,
                test_fraction=0.1,
                target_empty_fraction=0.0,
                tiling="2x2",
            )

        self.assertEqual(raw_empty_row_counts, {"train": 2, "validation": 2, "test": 2})
        self.assertEqual(synthetic_counts, {"train": 0, "validation": 0, "test": 0})
        train_positive_tiles = {
            str(row["source_image_id"]): json.loads(str(row["answer_boxes"]))[0]
            for row in split_rows["train"]
            if int(row["class_count"]) > 0
        }
        self.assertEqual(
            sorted(train_positive_tiles),
            ["train_positive_0__tile_r0_c0", "train_positive_0__tile_r0_c1"],
        )

        left_box = train_positive_tiles["train_positive_0__tile_r0_c0"]
        self.assertAlmostEqual(left_box["x_min"], 0.8, places=8)
        self.assertAlmostEqual(left_box["y_min"], 0.4, places=8)
        self.assertAlmostEqual(left_box["x_max"], 1.0, places=8)
        self.assertAlmostEqual(left_box["y_max"], 0.8, places=8)

        right_box = train_positive_tiles["train_positive_0__tile_r0_c1"]
        self.assertAlmostEqual(right_box["x_min"], 0.0, places=8)
        self.assertAlmostEqual(right_box["y_min"], 0.4, places=8)
        self.assertAlmostEqual(right_box["x_max"], 0.2, places=8)
        self.assertAlmostEqual(right_box["y_max"], 0.8, places=8)

    def test_synthetic_background_negative_is_added_when_split_has_no_empty_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw_root = _build_coco_fixture(
                Path(tmp) / "raw_dataset" / "Aerial Airport.coco",
                split_names=("train", "valid", "test"),
                include_empty_rows=False,
            )
            output_dir = Path(tmp) / "out"
            dataset_dict, split_rows, synthetic_counts, _, raw_empty_row_counts = build_mod.build_dataset_dict_from_raw_dir(
                raw_root,
                output_dir=output_dir,
                seed=42,
                val_fraction=0.1,
                test_fraction=0.1,
                target_empty_fraction=0.1,
            )

            self.assertEqual(len(dataset_dict["train"]), 2)
            self.assertEqual(len(dataset_dict["validation"]), 2)
            self.assertEqual(len(dataset_dict["test"]), 2)
            self.assertEqual(synthetic_counts, {"train": 1, "validation": 1, "test": 1})
            self.assertEqual(raw_empty_row_counts, {"train": 0, "validation": 0, "test": 0})

            synthetic_row = next(row for row in split_rows["train"] if bool(row["source_is_synthetic"]))
            self.assertEqual(synthetic_row["source_variant"], "background_negative")
            self.assertEqual(int(synthetic_row["class_count"]), 0)
            self.assertEqual(json.loads(str(synthetic_row["answer_boxes"])), [])
            self.assertEqual(len(list((output_dir / "synthetic_negatives" / "train").glob("*.jpg"))), 1)

    def test_background_negative_tops_up_when_split_has_some_empty_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw_root = _build_coco_fixture(
                Path(tmp) / "raw_dataset" / "Aerial Airport.coco",
                split_names=("train", "valid", "test"),
                include_empty_rows=True,
                positive_count=4,
            )
            dataset_dict, split_rows, synthetic_counts, _, raw_empty_row_counts = build_mod.build_dataset_dict_from_raw_dir(
                raw_root,
                output_dir=Path(tmp) / "out",
                seed=42,
                val_fraction=0.1,
                test_fraction=0.1,
                target_empty_fraction=0.25,
            )

        self.assertEqual(raw_empty_row_counts, {"train": 1, "validation": 1, "test": 1})
        self.assertEqual(synthetic_counts, {"train": 1, "validation": 1, "test": 1})
        self.assertEqual(len(dataset_dict["train"]), 6)
        self.assertEqual(len(dataset_dict["validation"]), 6)
        self.assertEqual(len(dataset_dict["test"]), 6)
        self.assertEqual(sum(int(row["class_count"]) == 0 for row in split_rows["train"]), 2)
        self.assertEqual(sum(int(row["class_count"]) == 0 for row in split_rows["validation"]), 2)
        self.assertEqual(sum(int(row["class_count"]) == 0 for row in split_rows["test"]), 2)

    def test_background_window_avoids_padded_boxes(self) -> None:
        boxes = [
            {
                "x_min": 0.05,
                "y_min": 0.05,
                "x_max": 0.35,
                "y_max": 0.35,
                "class_uid": DEFAULT_CLASS_UID,
                "class_name": DEFAULT_CLASS_NAME,
                "source_class_name": DEFAULT_CLASS_NAME,
            }
        ]
        window = build_mod._choose_background_negative_window(boxes, width=100, height=100)
        self.assertIsNotNone(window)
        assert window is not None
        candidate = build_mod._normalized_window(*window, 100, 100)
        padded_box = build_mod._expand_box(boxes[0], padding=build_mod.BACKGROUND_PADDING)
        self.assertFalse(build_mod._boxes_overlap(candidate, padded_box))

    def test_single_train_split_is_auto_split_into_train_validation_test(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw_root = _build_coco_fixture(
                Path(tmp) / "raw_dataset" / "Aerial Airport.coco",
                split_names=("train",),
                include_empty_rows=False,
                positive_count=4,
            )
            dataset_dict, split_rows, synthetic_counts, source_split_names, raw_empty_row_counts = build_mod.build_dataset_dict_from_raw_dir(
                raw_root,
                output_dir=Path(tmp) / "out",
                seed=42,
                val_fraction=0.25,
                test_fraction=0.25,
                target_empty_fraction=0.1,
            )

        self.assertEqual(source_split_names, ["train"])
        self.assertEqual(sorted(dataset_dict.keys()), ["test", "train", "validation"])
        self.assertEqual(len(dataset_dict["train"]), 3)
        self.assertEqual(len(dataset_dict["validation"]), 2)
        self.assertEqual(len(dataset_dict["test"]), 2)
        self.assertEqual(raw_empty_row_counts, {"train": 0, "validation": 0, "test": 0})
        self.assertEqual(synthetic_counts, {"train": 1, "validation": 1, "test": 1})
        self.assertEqual(sum(len(rows) for rows in split_rows.values()), 7)

    def test_tiling_2x2_auto_split_keeps_all_tiles_for_each_source_image_together(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw_root = _build_coco_fixture(
                Path(tmp) / "raw_dataset" / "Aerial Airport.coco",
                split_names=("train",),
                include_empty_rows=False,
                positive_count=6,
            )
            _, split_rows, synthetic_counts, source_split_names, raw_empty_row_counts = build_mod.build_dataset_dict_from_raw_dir(
                raw_root,
                output_dir=Path(tmp) / "out",
                seed=42,
                val_fraction=0.2,
                test_fraction=0.2,
                target_empty_fraction=0.0,
                tiling="2x2",
            )

        self.assertEqual(source_split_names, ["train"])
        self.assertEqual(sum(raw_empty_row_counts.values()), 18)
        self.assertEqual(synthetic_counts, {"train": 0, "validation": 0, "test": 0})
        self.assertEqual(sum(len(rows) for rows in split_rows.values()), 24)

        splits_by_source: dict[str, set[str]] = {}
        tile_counts_by_source: dict[str, int] = {}
        for split_name, rows in split_rows.items():
            for row in rows:
                source_base_id = str(row["source_base_id"])
                splits_by_source.setdefault(source_base_id, set()).add(split_name)
                tile_counts_by_source[source_base_id] = tile_counts_by_source.get(source_base_id, 0) + 1

        self.assertTrue(all(len(split_names) == 1 for split_names in splits_by_source.values()))
        self.assertTrue(all(tile_count == 4 for tile_count in tile_counts_by_source.values()))

    def test_stratified_auto_split_balances_empty_rows_and_preserves_group_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw_root = _build_coco_fixture(
                Path(tmp) / "raw_dataset" / "Aerial Airport.coco",
                split_names=("train",),
                include_empty_rows=True,
                positive_count=30,
                empty_count=6,
            )
            dataset_dict, split_rows, synthetic_counts, source_split_names, raw_empty_row_counts = build_mod.build_dataset_dict_from_raw_dir(
                raw_root,
                output_dir=Path(tmp) / "out",
                seed=42,
                val_fraction=0.2,
                test_fraction=0.2,
                target_empty_fraction=0.1,
                split_strategy="stratified_empty_group",
            )

        self.assertEqual(source_split_names, ["train"])
        self.assertEqual(sum(raw_empty_row_counts.values()), 6)
        self.assertTrue(all(count > 0 for count in raw_empty_row_counts.values()))
        self.assertEqual(synthetic_counts, {"train": 0, "validation": 0, "test": 0})
        self.assertEqual(len(dataset_dict["train"]), 22)
        self.assertEqual(len(dataset_dict["validation"]), 7)
        self.assertEqual(len(dataset_dict["test"]), 7)
        global_empty_fraction = 6 / 36.0
        for split_name in ("train", "validation", "test"):
            fraction = _empty_fraction(split_rows[split_name])
            self.assertLessEqual(
                abs(fraction - global_empty_fraction),
                build_mod.EMPTY_FRACTION_TOLERANCE,
                split_name,
            )

        by_split = {
            split_name: {str(row["source_base_id"]) for row in rows}
            for split_name, rows in split_rows.items()
        }
        self.assertTrue(by_split["train"].isdisjoint(by_split["validation"]))
        self.assertTrue(by_split["train"].isdisjoint(by_split["test"]))
        self.assertTrue(by_split["validation"].isdisjoint(by_split["test"]))

    def test_main_writes_metadata_and_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw_root = _build_coco_fixture(
                Path(tmp) / "raw_dataset" / "Aerial Airport.coco",
                split_names=("train", "valid", "test"),
                include_empty_rows=True,
            )
            output_dir = Path(tmp) / "out"
            build_mod.main(
                [
                    "--raw-dataset-dir",
                    str(raw_root),
                    "--output-dir",
                    str(output_dir),
                    "--target-empty-fraction",
                    "0.1",
                    "--push-to-hub",
                    "",
                ]
            )

            metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
            stats = json.loads((output_dir / "stats.json").read_text(encoding="utf-8"))
            ds = load_from_disk(str(output_dir))

        self.assertEqual(Path(metadata["raw_dataset_dir"]).name, "Aerial Airport.coco")
        self.assertEqual(metadata["source_format"], "coco")
        self.assertEqual(metadata["source_split_names"], ["test", "train", "valid"])
        self.assertEqual(metadata["split_strategy"], "stratified_empty_group")
        self.assertEqual(metadata["tiling"], "none")
        self.assertEqual(metadata["target_empty_fraction"], 0.1)
        self.assertEqual(metadata["class_catalog"][0]["class_name"], DEFAULT_CLASS_NAME)
        self.assertEqual(metadata["class_catalog"][0]["class_uid"], DEFAULT_CLASS_UID)
        self.assertEqual(metadata["pre_synthetic_split_sizes"], {"train": 2, "validation": 2, "test": 2})
        self.assertEqual(stats["tiling"], "none")
        self.assertEqual(stats["pre_synthetic_split_sizes"], {"train": 2, "validation": 2, "test": 2})
        self.assertEqual(metadata["split_sizes"], stats["split_sizes"])
        self.assertEqual(metadata["empty_row_fractions"], stats["empty_row_fractions"])
        self.assertEqual(metadata["empty_fraction_abs_delta"], stats["empty_fraction_abs_delta"])
        self.assertEqual(metadata["global_empty_row_fraction"], stats["global_empty_row_fraction"])
        self.assertEqual(metadata["raw_empty_row_counts"], stats["raw_empty_row_counts"])
        self.assertEqual(metadata["synthetic_negative_counts"], stats["synthetic_negative_counts"])
        self.assertEqual(stats["raw_empty_row_counts"], {"train": 1, "validation": 1, "test": 1})
        self.assertEqual(stats["empty_fraction_tolerance"], build_mod.EMPTY_FRACTION_TOLERANCE)
        self.assertEqual(len(ds["train"]), 2)
        self.assertEqual(len(ds["validation"]), 2)
        self.assertEqual(len(ds["test"]), 2)

    def test_main_writes_tiling_metadata_and_stats_for_2x2(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw_root = _build_coco_fixture(
                Path(tmp) / "raw_dataset" / "Aerial Airport.coco",
                split_names=("train", "valid", "test"),
                include_empty_rows=False,
            )
            output_dir = Path(tmp) / "out"
            build_mod.main(
                [
                    "--raw-dataset-dir",
                    str(raw_root),
                    "--output-dir",
                    str(output_dir),
                    "--tiling",
                    "2x2",
                    "--target-empty-fraction",
                    "0.0",
                    "--push-to-hub",
                    "",
                ]
            )

            metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
            stats = json.loads((output_dir / "stats.json").read_text(encoding="utf-8"))

        self.assertEqual(metadata["tiling"], "2x2")
        self.assertEqual(stats["tiling"], "2x2")
        self.assertEqual(metadata["pre_synthetic_split_sizes"], {"train": 4, "validation": 4, "test": 4})
        self.assertEqual(stats["pre_synthetic_split_sizes"], {"train": 4, "validation": 4, "test": 4})
        self.assertEqual(stats["raw_empty_row_counts"], {"train": 3, "validation": 3, "test": 3})
        self.assertEqual(stats["synthetic_negative_counts"], {"train": 0, "validation": 0, "test": 0})


class WrapperConfigTests(unittest.TestCase):
    def test_build_wrapper_defaults_enable_stratified_v2_split(self) -> None:
        args = build_mod.parse_args([])
        self.assertEqual(args.output_dir, "aerial_airport/outputs/maxs-m87_aerial_airport_point_v2")
        self.assertEqual(args.push_to_hub, "")
        self.assertEqual(args.split_strategy, "stratified_empty_group")
        self.assertEqual(args.tiling, "none")

    def test_build_tiling_config_parses(self) -> None:
        config_path = REPO_ROOT / "aerial_airport" / "configs" / "build_aerial_airport_hf_dataset_tiling.json"
        args = build_mod.parse_args(["--config", str(config_path)])
        self.assertEqual(args.output_dir, "aerial_airport/outputs/maxs-m87_aerial_airport_point_tiling_v1")
        self.assertEqual(args.tiling, "2x2")
        self.assertEqual(args.push_to_hub, "")

    def test_train_wrapper_defaults_resolve_to_airport_point_settings(self) -> None:
        args = train_mod.parse_args([])
        self.assertEqual(args.dataset_name, "maxs-m87/aerial_airport_point_v2")
        self.assertEqual(args.env_file, "aerial_airport/.env.staging")
        self.assertEqual(args.base_url, DEFAULT_STAGING_API_BASE)
        self.assertEqual(args.dataset_path, "aerial_airport/outputs/maxs-m87_aerial_airport_point_v2")
        self.assertEqual(args.val_split, "validation")
        self.assertEqual(args.skill, "point")
        self.assertEqual(args.point_prompt_style, "class_name")
        self.assertEqual(args.reward_metric, "f1")
        self.assertEqual(args.wandb_project, "moondream-aerial-airport-point-rl")

    def test_train_tiling_config_parses(self) -> None:
        config_path = REPO_ROOT / "aerial_airport" / "configs" / "train_aerial_airport_point_tiling.json"
        args = train_mod.parse_args(["--config", str(config_path)])
        self.assertEqual(args.dataset_name, "maxs-m87/aerial_airport_point_v2")
        self.assertEqual(args.dataset_path, "aerial_airport/outputs/maxs-m87_aerial_airport_point_v2")
        self.assertTrue(args.runtime_tiling)
        self.assertEqual(args.tile_grid_size, 3)
        self.assertAlmostEqual(args.tile_overlap, 0.1, places=8)
        self.assertAlmostEqual(args.tile_point_merge_radius, 0.015, places=8)
        self.assertEqual(args.wandb_run_name, "aerial-airport-point-tiling-v1")

    def test_detect_wrapper_defaults_resolve_to_airport_detect_settings(self) -> None:
        args = train_detect_mod.parse_args([])
        self.assertEqual(args.dataset_name, "maxs-m87/aerial_airport_point_v2")
        self.assertEqual(args.env_file, "aerial_airport/.env.staging")
        self.assertEqual(args.base_url, DEFAULT_STAGING_API_BASE)
        self.assertEqual(args.dataset_path, "aerial_airport/outputs/maxs-m87_aerial_airport_point_v2")
        self.assertEqual(args.val_split, "validation")
        self.assertEqual(args.test_split, "test")
        self.assertEqual(args.skill, "detect")
        self.assertEqual(args.selection_metric, "f1")
        self.assertTrue(args.run_final_test)
        self.assertEqual(args.max_objects, 150)

    def test_benchmark_wrapper_defaults_resolve_to_airport_point_settings(self) -> None:
        args = bench_mod.parse_args([])
        self.assertEqual(args.dataset_name, "maxs-m87/aerial_airport_point_v2")
        self.assertEqual(args.env_file, "aerial_airport/.env.staging")
        self.assertEqual(args.api_base, DEFAULT_STAGING_API_BASE)
        self.assertEqual(args.dataset_path, "aerial_airport/outputs/maxs-m87_aerial_airport_point_v2")
        self.assertEqual(args.split, "test")
        self.assertEqual(args.skill, "point")
        self.assertEqual(args.point_prompt_style, "class_name")

    def test_benchmark_tiling_config_parses(self) -> None:
        config_path = REPO_ROOT / "aerial_airport" / "configs" / "benchmark_aerial_airport_point_tiling.json"
        args = bench_mod.parse_args(["--config", str(config_path)])
        self.assertEqual(args.dataset_name, "maxs-m87/aerial_airport_point_v2")
        self.assertEqual(args.dataset_path, "aerial_airport/outputs/maxs-m87_aerial_airport_point_v2")
        self.assertTrue(args.runtime_tiling)
        self.assertEqual(args.tile_grid_size, 3)
        self.assertAlmostEqual(args.tile_overlap, 0.1, places=8)
        self.assertAlmostEqual(args.tile_point_merge_radius, 0.015, places=8)
        self.assertEqual(args.viz_dir, "aerial_airport/outputs/benchmark_viz/tiling_point")
        self.assertEqual(args.out_json, "aerial_airport/outputs/benchmarks/benchmark_aerial_airport_point_tiling.json")

    def test_detect_benchmark_wrapper_defaults_resolve_to_airport_detect_settings(self) -> None:
        args = bench_detect_mod.parse_args([])
        self.assertEqual(args.dataset_name, "maxs-m87/aerial_airport_point_v2")
        self.assertEqual(args.env_file, "aerial_airport/.env.staging")
        self.assertEqual(args.api_base, DEFAULT_STAGING_API_BASE)
        self.assertEqual(args.dataset_path, "aerial_airport/outputs/maxs-m87_aerial_airport_point_v2")
        self.assertEqual(args.split, "test")
        self.assertEqual(args.skill, "detect")
        self.assertEqual(args.max_objects, 150)

    def test_api_key_env_var_precedence_matches_football_pattern(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {"MOONDREAM_API_KEY": "generic_shell_key_should_not_win"},
            clear=False,
        ):
            env_path = Path(tmp) / ".env.airport"
            env_path.write_text(
                "MOONDREAM_API_KEY_AIRPORT=file_key_from_named_env_var\n",
                encoding="utf-8",
            )

            args = train_mod.parse_args(
                [
                    "--env-file",
                    str(env_path),
                    "--api-key-env-var",
                    "MOONDREAM_API_KEY_AIRPORT",
                    "--base-url",
                    "https://api-staging.moondream.ai/v1",
                ]
            )
            resolved = shared_train._resolve_runtime_env(args)

        self.assertEqual(resolved.api_key, "file_key_from_named_env_var")
        self.assertEqual(resolved.api_key_env_var, "MOONDREAM_API_KEY_AIRPORT")
        self.assertEqual(resolved.base_url, "https://api-staging.moondream.ai/v1")

    def test_recall_preset_overrides_manual_lr(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "cfg.json"
            cfg_path.write_text(
                json.dumps(
                    {
                        "dataset_name": "custom/aerial-airport",
                        "use_recall_first_preset": True,
                        "lr": 0.0001,
                    }
                ),
                encoding="utf-8",
            )

            args = train_mod.parse_args(["--config", str(cfg_path)])

        self.assertTrue(args.use_recall_first_preset)
        self.assertAlmostEqual(args.lr, 5e-4, places=8)
        self.assertEqual(args.neg_prompts_per_nonempty, 0)
        self.assertAlmostEqual(args.neg_reward_weight, 0.15, places=8)

    def test_wrapper_config_precedence_and_existing_cicd_configs_parse(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "cfg.json"
            cfg_path.write_text(
                json.dumps(
                    {
                        "dataset_name": "custom/aerial-airport",
                        "api_key_env_var": "CICID_GPUB_MOONDREAM_API_KEY_1",
                        "off_policy": True,
                        "group_size": 4,
                    }
                ),
                encoding="utf-8",
            )

            args = train_mod.parse_args(
                [
                    "--config",
                    str(cfg_path),
                    "--api-key-env-var",
                    "CICID_GPUB_MOONDREAM_API_KEY_4",
                    "--group-size",
                    "6",
                ]
            )

        self.assertEqual(args.dataset_name, "custom/aerial-airport")
        self.assertEqual(args.api_key_env_var, "CICID_GPUB_MOONDREAM_API_KEY_4")
        self.assertEqual(args.group_size, 6)
        self.assertTrue(args.off_policy)

        config_root = REPO_ROOT / "aerial_airport" / "configs" / "cicd"
        expectations = {
            "cicd_train_aerial_airport_point_control.json": (False, False, 1.0, 1.0, 1, 0.95, 0.5, "CICID_GPUB_MOONDREAM_API_KEY_1"),
            "cicd_train_aerial_airport_point_recall_primary.json": (False, True, 2.0, 1.0, 0, 0.995, 0.15, "CICID_GPUB_MOONDREAM_API_KEY_2"),
            "cicd_train_aerial_airport_point_recall_offpolicy.json": (True, True, 2.0, 1.0, 0, 0.995, 0.15, "CICID_GPUB_MOONDREAM_API_KEY_3"),
        }
        for filename, (
            off_policy,
            recall_preset,
            fn_exp,
            fp_exp,
            neg_nonempty,
            pos_task_prob,
            neg_reward_weight,
            api_key_env_var,
        ) in expectations.items():
            with self.subTest(config=filename):
                args = train_mod.parse_args(["--config", str(config_root / filename)])
                self.assertEqual(args.dataset_name, "maxs-m87/aerial_airport_point_v2")
                self.assertEqual(args.env_file, "aerial_airport/.env.staging")
                self.assertEqual(args.base_url, DEFAULT_STAGING_API_BASE)
                self.assertEqual(args.dataset_path, "aerial_airport/outputs/maxs-m87_aerial_airport_point_v2")
                self.assertEqual(args.val_split, "validation")
                self.assertEqual(args.skill, "point")
                self.assertEqual(args.point_prompt_style, "class_name")
                self.assertEqual(args.reward_metric, "f1")
                self.assertEqual(args.api_key_env_var, api_key_env_var)
                self.assertEqual(args.off_policy, off_policy)
                self.assertEqual(args.use_recall_first_preset, recall_preset)
                self.assertEqual(args.fn_penalty_exponent, fn_exp)
                self.assertEqual(args.fp_penalty_exponent, fp_exp)
                self.assertEqual(args.neg_prompts_per_nonempty, neg_nonempty)
                self.assertEqual(args.pos_task_prob, pos_task_prob)
                self.assertEqual(args.neg_reward_weight, neg_reward_weight)

    def test_followup_experiment_configs_parse(self) -> None:
        config_root = REPO_ROOT / "aerial_airport" / "configs" / "cicd"
        expectations = {
            "cicd_train_aerial_airport_point_recall_lr5e4_r8.json": (8, 8, 4, True, False, 5e-4, "CICID_GPUB_MOONDREAM_API_KEY_1", "aerial-airport-point-recall-lr5e4-r8"),
            "cicd_train_aerial_airport_point_recall_lr2e4_r16.json": (16, 16, 8, False, False, 2e-4, "CICID_GPUB_MOONDREAM_API_KEY_2", "aerial-airport-point-recall-lr2e4-r16"),
            "cicd_train_aerial_airport_point_recall_lr2e4_r8.json": (8, 8, 4, False, False, 2e-4, "CICID_GPUB_MOONDREAM_API_KEY_3", "aerial-airport-point-recall-lr2e4-r8"),
            "cicd_train_aerial_airport_point_recall_offpolicy_gentle_lr2e4_r8.json": (8, 8, 4, False, True, 2e-4, "CICID_GPUB_MOONDREAM_API_KEY_4", "aerial-airport-point-recall-offpolicy-gentle-lr2e4-r8"),
        }
        for filename, (
            rank,
            batch_size,
            group_size,
            recall_preset,
            off_policy,
            lr,
            api_key_env_var,
            wandb_run_name,
        ) in expectations.items():
            with self.subTest(config=filename):
                args = train_mod.parse_args(["--config", str(config_root / filename)])
                self.assertEqual(args.dataset_name, "maxs-m87/aerial_airport_point_v2")
                self.assertEqual(args.dataset_path, "aerial_airport/outputs/maxs-m87_aerial_airport_point_v2")
                self.assertEqual(args.rank, rank)
                self.assertEqual(args.batch_size, batch_size)
                self.assertEqual(args.group_size, group_size)
                self.assertEqual(args.use_recall_first_preset, recall_preset)
                self.assertEqual(args.off_policy, off_policy)
                self.assertAlmostEqual(args.lr, lr, places=8)
                self.assertEqual(args.api_key_env_var, api_key_env_var)
                self.assertEqual(args.wandb_run_name, wandb_run_name)
                self.assertAlmostEqual(args.kl_warning_threshold, 0.0, places=8)
                self.assertAlmostEqual(args.kl_stop_threshold, 0.0, places=8)
                self.assertEqual(args.kl_stop_consecutive, 1)
                self.assertEqual(args.num_steps, 120)

    def test_v2_repair_experiment_configs_parse(self) -> None:
        config_root = REPO_ROOT / "aerial_airport" / "configs" / "cicd"
        expectations = {
            "cicd_train_aerial_airport_point_v2_recall_anchor_explicit.json": (
                16,
                8,
                5e-4,
                False,
                2.0,
                1.0,
                0.15,
                "CICID_GPUB_MOONDREAM_API_KEY_1",
                "aerial-airport-point-v2-recall-anchor-explicit",
            ),
            "cicd_train_aerial_airport_point_v2_recall_anchor_lowlr.json": (
                16,
                8,
                1e-4,
                False,
                2.0,
                1.0,
                0.15,
                "CICID_GPUB_MOONDREAM_API_KEY_2",
                "aerial-airport-point-v2-recall-anchor-lowlr",
            ),
            "cicd_train_aerial_airport_point_v2_recall_precision_pressure.json": (
                16,
                8,
                5e-4,
                False,
                2.0,
                2.0,
                0.5,
                "CICID_GPUB_MOONDREAM_API_KEY_3",
                "aerial-airport-point-v2-recall-precision-pressure",
            ),
            "cicd_train_aerial_airport_point_v2_recall_linear_fn.json": (
                16,
                8,
                5e-4,
                False,
                1.0,
                1.0,
                0.15,
                "CICID_GPUB_MOONDREAM_API_KEY_1",
                "aerial-airport-point-v2-recall-linear-fn",
            ),
            "cicd_train_aerial_airport_point_v2_recall_precision_offpolicy_light.json": (
                16,
                8,
                5e-4,
                True,
                2.0,
                2.0,
                0.5,
                "CICID_GPUB_MOONDREAM_API_KEY_4",
                "aerial-airport-point-v2-recall-precision-offpolicy-light",
            ),
        }
        for filename, (
            rank,
            group_size,
            lr,
            off_policy,
            fn_exp,
            fp_exp,
            neg_reward_weight,
            api_key_env_var,
            wandb_run_name,
        ) in expectations.items():
            with self.subTest(config=filename):
                args = train_mod.parse_args(["--config", str(config_root / filename)])
                self.assertEqual(args.dataset_name, "maxs-m87/aerial_airport_point_v2")
                self.assertEqual(args.dataset_path, "aerial_airport/outputs/maxs-m87_aerial_airport_point_v2")
                self.assertEqual(args.rank, rank)
                self.assertEqual(args.batch_size, 16)
                self.assertEqual(args.group_size, group_size)
                self.assertAlmostEqual(args.lr, lr, places=8)
                self.assertEqual(args.off_policy, off_policy)
                self.assertFalse(args.use_recall_first_preset)
                self.assertAlmostEqual(args.fn_penalty_exponent, fn_exp, places=8)
                self.assertAlmostEqual(args.fp_penalty_exponent, fp_exp, places=8)
                self.assertEqual(args.neg_prompts_per_empty, 1)
                self.assertEqual(args.neg_prompts_per_nonempty, 0)
                self.assertAlmostEqual(args.neg_reward_weight, neg_reward_weight, places=8)
                self.assertEqual(args.api_key_env_var, api_key_env_var)
                self.assertEqual(args.wandb_run_name, wandb_run_name)
                self.assertEqual(args.num_steps, 120)
                self.assertEqual(args.eval_every, 5)
                self.assertEqual(args.save_every, 5)

    def test_tiling_round1_configs_parse(self) -> None:
        config_root = REPO_ROOT / "aerial_airport" / "configs" / "tiling_round1"
        expectations = {
            "cicd_train_aerial_airport_point_tiling_v1_recall_anchor_explicit.json": (
                False,
                5e-4,
                2.0,
                1.0,
                0.15,
                "CICID_GPUB_MOONDREAM_API_KEY_1",
                "aerial-airport-point-tiling-v1-recall-anchor-explicit",
            ),
            "cicd_train_aerial_airport_point_tiling_v1_recall_anchor_lowlr.json": (
                False,
                1e-4,
                2.0,
                1.0,
                0.15,
                "CICID_GPUB_MOONDREAM_API_KEY_2",
                "aerial-airport-point-tiling-v1-recall-anchor-lowlr",
            ),
            "cicd_train_aerial_airport_point_tiling_v1_recall_anchor_r32_b32_lowlr.json": (
                False,
                5e-5,
                2.0,
                1.0,
                0.15,
                "CICID_GPUB_MOONDREAM_API_KEY_2",
                "aerial-airport-point-tiling-v1-recall-anchor-r32-b32-lowlr",
            ),
            "cicd_train_aerial_airport_point_tiling_v1_recall_precision_pressure.json": (
                False,
                5e-4,
                2.0,
                2.0,
                0.5,
                "CICID_GPUB_MOONDREAM_API_KEY_3",
                "aerial-airport-point-tiling-v1-recall-precision-pressure",
            ),
            "cicd_train_aerial_airport_point_tiling_v1_recall_precision_offpolicy_light.json": (
                True,
                5e-4,
                2.0,
                2.0,
                0.5,
                "CICID_GPUB_MOONDREAM_API_KEY_4",
                "aerial-airport-point-tiling-v1-recall-precision-offpolicy-light",
            ),
        }
        for filename, (
            off_policy,
            lr,
            fn_exp,
            fp_exp,
            neg_reward_weight,
            api_key_env_var,
            wandb_run_name,
        ) in expectations.items():
            with self.subTest(config=filename):
                args = train_mod.parse_args(["--config", str(config_root / filename)])
                self.assertEqual(args.dataset_name, "maxs-m87/aerial_airport_point_v2")
                self.assertEqual(args.dataset_path, "aerial_airport/outputs/maxs-m87_aerial_airport_point_v2")
                expected_rank = 32 if "r32_b32" in filename else 16
                expected_batch_size = 32 if "r32_b32" in filename else 16
                self.assertEqual(args.rank, expected_rank)
                self.assertEqual(args.batch_size, expected_batch_size)
                self.assertEqual(args.group_size, 8)
                self.assertAlmostEqual(args.lr, lr, places=8)
                self.assertEqual(args.off_policy, off_policy)
                self.assertFalse(args.use_recall_first_preset)
                self.assertEqual(args.selection_metric, "f1")
                self.assertTrue(args.runtime_tiling)
                self.assertEqual(args.tile_grid_size, 3)
                self.assertAlmostEqual(args.tile_overlap, 0.1, places=8)
                self.assertAlmostEqual(args.tile_point_merge_radius, 0.015, places=8)
                self.assertAlmostEqual(args.fn_penalty_exponent, fn_exp, places=8)
                self.assertAlmostEqual(args.fp_penalty_exponent, fp_exp, places=8)
                self.assertEqual(args.neg_prompts_per_empty, 1)
                self.assertEqual(args.neg_prompts_per_nonempty, 0)
                self.assertAlmostEqual(args.neg_reward_weight, neg_reward_weight, places=8)
                self.assertEqual(args.api_key_env_var, api_key_env_var)
                self.assertEqual(args.wandb_run_name, wandb_run_name)
                self.assertEqual(args.num_steps, 300 if "anchor_plus" not in filename else 500)
                self.assertEqual(args.eval_every, 5)
                self.assertEqual(args.save_every, 5)

    def test_detect_tiling_round1_configs_parse(self) -> None:
        config_root = REPO_ROOT / "aerial_airport" / "configs" / "tiling_round1"
        expectations = {
            "cicd_train_aerial_airport_detect_v1_control.json": (False, 1e-4, 300, 1.0, "CICID_GPUB_MOONDREAM_API_KEY_1"),
            "cicd_train_aerial_airport_detect_v1_runtime_tiling.json": (True, 1e-4, 300, 1.0, "CICID_GPUB_MOONDREAM_API_KEY_2"),
            "cicd_train_aerial_airport_detect_v1_runtime_tiling_fn_focus.json": (True, 1e-4, 300, 2.0, "CICID_GPUB_MOONDREAM_API_KEY_3"),
            "cicd_train_aerial_airport_detect_v1_runtime_tiling_lowlr_long.json": (True, 5e-5, 500, 2.0, "CICID_GPUB_MOONDREAM_API_KEY_4"),
        }
        for filename, (runtime_tiling, lr, num_steps, fn_exp, api_key_env_var) in expectations.items():
            with self.subTest(config=filename):
                args = train_detect_mod.parse_args(["--config", str(config_root / filename)])
                self.assertEqual(args.dataset_name, "maxs-m87/aerial_airport_point_v2")
                self.assertEqual(args.dataset_path, "aerial_airport/outputs/maxs-m87_aerial_airport_point_v2")
                self.assertEqual(args.skill, "detect")
                self.assertEqual(args.selection_metric, "f1")
                self.assertEqual(args.rank, 16)
                self.assertEqual(args.batch_size, 8)
                self.assertEqual(args.group_size, 4)
                self.assertAlmostEqual(args.lr, lr, places=8)
                self.assertEqual(args.num_steps, num_steps)
                self.assertEqual(args.runtime_tiling, runtime_tiling)
                self.assertAlmostEqual(args.tile_overlap, 0.1, places=8)
                self.assertAlmostEqual(args.tile_box_merge_iou, 0.5, places=8)
                self.assertAlmostEqual(args.fn_penalty_exponent, fn_exp, places=8)
                self.assertEqual(args.max_objects, 150)
                self.assertEqual(args.neg_prompts_per_empty, 1)
                self.assertEqual(args.neg_prompts_per_nonempty, 0)
                self.assertAlmostEqual(args.neg_reward_weight, 0.15, places=8)
                self.assertTrue(args.run_final_test)
                self.assertEqual(args.api_key_env_var, api_key_env_var)


class RuntimeTilingTests(unittest.TestCase):
    def test_runtime_tiling_windows_cover_full_image_with_expected_overlap(self) -> None:
        windows = tiling_mod.build_tile_windows(width=100, height=100, grid_size=3, overlap=0.1)
        self.assertEqual(len(windows), 9)
        first = windows[0]
        second = windows[1]
        last = windows[-1]
        self.assertAlmostEqual(first.x_min, 0.0, places=8)
        self.assertAlmostEqual(first.y_min, 0.0, places=8)
        self.assertAlmostEqual(last.x_max, 1.0, places=8)
        self.assertAlmostEqual(last.y_max, 1.0, places=8)
        self.assertLess(second.x_min, first.x_max)
        self.assertAlmostEqual(first.x_max - second.x_min, 0.0357142857, places=6)

    def test_runtime_tiling_point_merge_collapses_boundary_duplicates(self) -> None:
        merged = tiling_mod.merge_points(
            [
                tiling_mod.Point2D(x=0.34, y=0.34),
                tiling_mod.Point2D(x=0.345, y=0.345),
                tiling_mod.Point2D(x=0.80, y=0.80),
            ],
            radius=0.015,
        )
        self.assertEqual(len(merged), 2)
        self.assertTrue(any(abs(point.x - 0.3425) < 1e-6 for point in merged))
        self.assertTrue(any(abs(point.x - 0.80) < 1e-6 for point in merged))

    def test_runtime_tiling_box_merge_clusters_overlap_duplicates(self) -> None:
        merged = tiling_mod.merge_boxes(
            [
                tiling_mod.Box2D(x_min=0.30, y_min=0.30, x_max=0.40, y_max=0.40),
                tiling_mod.Box2D(x_min=0.31, y_min=0.30, x_max=0.41, y_max=0.40),
                tiling_mod.Box2D(x_min=0.70, y_min=0.70, x_max=0.80, y_max=0.80),
            ],
            iou_threshold=0.5,
        )
        self.assertEqual(len(merged), 2)
        self.assertTrue(any(abs(box.x_min - 0.305) < 1e-6 for box in merged))
        self.assertTrue(any(abs(box.x_min - 0.70) < 1e-6 for box in merged))

    def test_benchmark_runtime_tiling_visualization_marks_removed_duplicates(self) -> None:
        tile_windows = tiling_mod.build_tile_windows(width=100, height=100, grid_size=3, overlap=0.1)
        raw_points = [
            bench_mod._RuntimeMappedPoint(point=bench_mod.Point(x=0.34, y=0.34), tile_row=0, tile_col=0),
            bench_mod._RuntimeMappedPoint(point=bench_mod.Point(x=0.345, y=0.345), tile_row=0, tile_col=1),
            bench_mod._RuntimeMappedPoint(point=bench_mod.Point(x=0.80, y=0.80), tile_row=2, tile_col=2),
        ]
        merged_points, removed_duplicate_indices = bench_mod._cluster_runtime_mapped_points(
            raw_points,
            radius=0.015,
        )

        self.assertEqual(len(merged_points), 2)
        self.assertEqual(removed_duplicate_indices, {1})

        with tempfile.TemporaryDirectory() as tmp:
            out_path = bench_mod._save_task_visualization(
                out_dir=Path(tmp),
                label="candidate",
                sample_idx=0,
                task=bench_mod.TaskSample(
                    image=Image.new("RGB", (100, 100), color=(255, 255, 255)),
                    prompt="airplane",
                    gt_boxes=[bench_mod.Box(x_min=0.30, y_min=0.30, x_max=0.38, y_max=0.38)],
                    class_name=DEFAULT_CLASS_NAME,
                    sample_id="viz-sample",
                ),
                pred_points=merged_points,
                f1=1.0,
                tp=1,
                fp=0,
                fn=0,
                runtime_tiling_viz=bench_mod._RuntimeTilingViz(
                    tile_windows=tile_windows,
                    raw_points=raw_points,
                    removed_duplicate_indices=removed_duplicate_indices,
                ),
            )

            self.assertIsNotNone(out_path)
            assert out_path is not None
            viz_path = Path(out_path)
            self.assertTrue(viz_path.exists())
            self.assertGreater(viz_path.stat().st_size, 0)

    def test_shared_eval_runtime_tiling_merges_boundary_points_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "positive.jpg"
            _write_image(image_path, size=(100, 100))
            positive_row = {
                "image": Image.open(image_path).convert("RGB"),
                "answer_boxes": json.dumps(
                    [
                        {
                            "x_min": 0.32,
                            "y_min": 0.32,
                            "x_max": 0.36,
                            "y_max": 0.36,
                            "class_uid": DEFAULT_CLASS_UID,
                            "class_name": DEFAULT_CLASS_NAME,
                            "source_class_name": DEFAULT_CLASS_NAME,
                        }
                    ]
                ),
                "source_collection": "unit",
                "source_dataset": "unit",
                "class_count": 1,
            }

            tile_windows = tiling_mod.build_tile_windows(width=100, height=100, grid_size=3, overlap=0.1)

            def _fake_rollouts(*args, **kwargs):
                requests = kwargs["requests"]
                self.assertEqual(len(requests), 9)
                outputs = []
                for tile_index, window in enumerate(tile_windows):
                    if tile_index in {0, 1, 3, 4}:
                        local_x = (0.34 - window.x_min) / (window.x_max - window.x_min)
                        local_y = (0.34 - window.y_min) / (window.y_max - window.y_min)
                        points = [shared_train.PointAnnotation(x=local_x, y=local_y)]
                    else:
                        points = []
                    outputs.append(
                        SimpleNamespace(
                            rollouts=[SimpleNamespace(output=shared_train.PointOutput(points=points))]
                        )
                    )
                return outputs

            with patch.object(shared_train, "_rollouts_batch_with_retry", side_effect=_fake_rollouts):
                metrics = shared_train._evaluate(
                    finetune=object(),
                    eval_rows=[positive_row],
                    all_class_names=[DEFAULT_CLASS_NAME],
                    rng=shared_train.random.Random(42),
                    neg_prompts_per_empty=1,
                    neg_prompts_per_nonempty=0,
                    max_samples=10,
                    batch_size=1,
                    max_workers=9,
                    rollout_retries=0,
                    rollout_retry_backoff_s=0.1,
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=16,
                    max_objects=10,
                    skill="point",
                    point_prompt_style="class_name",
                    reasoning=False,
                    runtime_tiling=True,
                    tile_grid_size=3,
                    tile_overlap=0.1,
                    tile_point_merge_radius=0.015,
                    tile_box_merge_iou=0.5,
                )

        self.assertEqual(metrics["eval_tp"], 1)
        self.assertEqual(metrics["eval_fp"], 0)
        self.assertEqual(metrics["eval_fn"], 0)
        self.assertAlmostEqual(metrics["eval_f1"], 1.0, places=8)

class SmokeTests(unittest.TestCase):
    def test_single_class_point_runs_emit_inert_knob_warnings(self) -> None:
        warnings = shared_train._single_class_point_mode_warnings(
            skill="point",
            all_class_names=[DEFAULT_CLASS_NAME],
        )
        self.assertEqual(len(warnings), 4)
        self.assertTrue(any("neg-prompts-per-nonempty" in item for item in warnings))
        self.assertTrue(any("pos-task-prob" in item for item in warnings))
        self.assertTrue(any("max-objects" in item for item in warnings))
        self.assertTrue(any("eval_miou" in item for item in warnings))

    def test_built_rows_support_shared_point_task_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw_root = _build_coco_fixture(
                Path(tmp) / "raw_dataset" / "Aerial Airport.coco",
                split_names=("train", "valid", "test"),
                include_empty_rows=False,
            )
            dataset_dict, _, _, _, _ = build_mod.build_dataset_dict_from_raw_dir(
                raw_root,
                output_dir=Path(tmp) / "out",
                seed=42,
                val_fraction=0.1,
                test_fraction=0.1,
                target_empty_fraction=0.1,
            )

            train_rows = [dataset_dict["train"][idx] for idx in range(len(dataset_dict["train"]))]
            positive_row = next(row for row in train_rows if int(row["class_count"]) > 0)
            negative_row = next(row for row in train_rows if int(row["class_count"]) == 0)

            discovered = sorted(
                {
                    item.class_name
                    for row in train_rows
                    for item in (shared_train._to_base_sample(row).boxes if shared_train._to_base_sample(row) else [])
                }
            )
            self.assertEqual(discovered, [DEFAULT_CLASS_NAME])

            positive_base = shared_train._to_base_sample(positive_row)
            negative_base = shared_train._to_base_sample(negative_row)
            self.assertIsNotNone(positive_base)
            self.assertIsNotNone(negative_base)
            assert positive_base is not None
            assert negative_base is not None

            positive_tasks = shared_train._tasks_from_base_sample(
                positive_base,
                all_class_names=[DEFAULT_CLASS_NAME],
                rng=shared_train.random.Random(42),
                neg_prompts_per_empty=1,
                neg_prompts_per_nonempty=0,
                prompt_style="class_name",
            )
            negative_tasks = shared_train._tasks_from_base_sample(
                negative_base,
                all_class_names=[DEFAULT_CLASS_NAME],
                rng=shared_train.random.Random(42),
                neg_prompts_per_empty=1,
                neg_prompts_per_nonempty=0,
                prompt_style="class_name",
            )

            self.assertTrue(any(task.is_positive for task in positive_tasks))
            self.assertTrue(any(not task.is_positive for task in negative_tasks))

    def test_shared_eval_reports_positive_and_negative_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            positive_path = Path(tmp) / "positive.jpg"
            negative_path = Path(tmp) / "negative.jpg"
            _write_image(positive_path)
            _write_image(negative_path, color=(240, 240, 240))

            positive_row = {
                "image": Image.open(positive_path).convert("RGB"),
                "answer_boxes": json.dumps(
                    [
                        {
                            "x_min": 0.1,
                            "y_min": 0.1,
                            "x_max": 0.4,
                            "y_max": 0.4,
                            "class_uid": DEFAULT_CLASS_UID,
                            "class_name": DEFAULT_CLASS_NAME,
                            "source_class_name": DEFAULT_CLASS_NAME,
                        }
                    ]
                ),
                "source_collection": "unit",
                "source_dataset": "unit",
                "class_count": 1,
            }
            negative_row = {
                "image": Image.open(negative_path).convert("RGB"),
                "answer_boxes": "[]",
                "source_collection": "unit",
                "source_dataset": "unit",
                "class_count": 0,
            }

            call_index = {"value": 0}

            def _fake_rollouts(*args, **kwargs):
                idx = call_index["value"]
                call_index["value"] += 1
                if idx == 0:
                    output = shared_train.PointOutput(
                        points=[shared_train.PointAnnotation(x=0.2, y=0.2)]
                    )
                else:
                    output = shared_train.PointOutput(points=[])
                return [SimpleNamespace(rollouts=[SimpleNamespace(output=output)])]

            with patch.object(shared_train, "_rollouts_batch_with_retry", side_effect=_fake_rollouts):
                metrics = shared_train._evaluate(
                    finetune=object(),
                    eval_rows=[positive_row, negative_row],
                    all_class_names=[DEFAULT_CLASS_NAME],
                    rng=shared_train.random.Random(42),
                    neg_prompts_per_empty=1,
                    neg_prompts_per_nonempty=0,
                    max_samples=10,
                    batch_size=1,
                    max_workers=1,
                    rollout_retries=0,
                    rollout_retry_backoff_s=0.1,
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=16,
                    max_objects=10,
                    skill="point",
                    point_prompt_style="class_name",
                    reasoning=False,
                    runtime_tiling=False,
                    tile_grid_size=3,
                    tile_overlap=0.1,
                    tile_point_merge_radius=0.015,
                    tile_box_merge_iou=0.5,
                )

        self.assertEqual(metrics["eval_positive_tasks"], 1)
        self.assertEqual(metrics["eval_negative_tasks"], 1)
        self.assertAlmostEqual(metrics["eval_positive_f1"], 1.0, places=8)
        self.assertAlmostEqual(metrics["eval_negative_f1"], 1.0, places=8)
        self.assertEqual(metrics["eval_positive_tp"], 1)
        self.assertEqual(metrics["eval_positive_fp"], 0)
        self.assertEqual(metrics["eval_positive_fn"], 0)
        self.assertEqual(metrics["eval_negative_tp"], 0)
        self.assertEqual(metrics["eval_negative_fp"], 0)
        self.assertEqual(metrics["eval_negative_fn"], 0)

    def test_kl_guard_warns_and_stops(self) -> None:
        hits, warn, stop = shared_train._update_kl_guard(
            kl_value=12.0,
            warning_threshold=10.0,
            stop_threshold=1000.0,
            stop_consecutive=2,
            consecutive_hits=0,
        )
        self.assertEqual(hits, 0)
        self.assertTrue(warn)
        self.assertFalse(stop)

        hits, warn, stop = shared_train._update_kl_guard(
            kl_value=1500.0,
            warning_threshold=10.0,
            stop_threshold=1000.0,
            stop_consecutive=2,
            consecutive_hits=0,
        )
        self.assertEqual(hits, 1)
        self.assertTrue(warn)
        self.assertFalse(stop)

        hits, warn, stop = shared_train._update_kl_guard(
            kl_value=1600.0,
            warning_threshold=10.0,
            stop_threshold=1000.0,
            stop_consecutive=2,
            consecutive_hits=hits,
        )
        self.assertEqual(hits, 2)
        self.assertTrue(warn)
        self.assertTrue(stop)

    def test_benchmark_smoke_writes_json_for_v2_style_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw_root = _build_coco_fixture(
                Path(tmp) / "raw_dataset" / "Aerial Airport.coco",
                split_names=("train", "valid", "test"),
                include_empty_rows=True,
            )
            output_dir = Path(tmp) / "out"
            build_mod.main(
                [
                    "--raw-dataset-dir",
                    str(raw_root),
                    "--output-dir",
                    str(output_dir),
                    "--push-to-hub",
                    "",
                ]
            )
            out_json = Path(tmp) / "bench.json"

            with patch("aerial_airport.benchmark_aerial_airport_point._call_point_api", return_value=[]):
                bench_mod.main(
                    [
                        "--dataset-path",
                        str(output_dir),
                        "--dataset-name",
                        "",
                        "--skip-baseline",
                        "--model",
                        "dummy-model",
                        "--api-key",
                        "test-key",
                        "--max-samples",
                        "2",
                        "--out-json",
                        str(out_json),
                    ]
                )

            payload = json.loads(out_json.read_text(encoding="utf-8"))

        self.assertEqual(payload["skill"], "point")
        self.assertEqual(payload["split"], "test")
        self.assertIn("eval_f1", payload)
        self.assertIn("per_class", payload)


if __name__ == "__main__":
    unittest.main()
