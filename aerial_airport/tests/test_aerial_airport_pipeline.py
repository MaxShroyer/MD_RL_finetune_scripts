from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from datasets import load_from_disk
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aerial_airport import benchmark_aerial_airport_detect as bench_detect_mod
from aerial_airport import benchmark_aerial_airport_point as bench_point_mod
from aerial_airport import build_aerial_airport_hf_dataset as build_mod
from aerial_airport import runtime_tiling as tiling_mod
from aerial_airport import train_aerial_airport_detect as train_detect_mod
from aerial_airport import train_aerial_airport_point as train_point_mod
from aerial_airport.common import (
    DEFAULT_CLASS_NAME,
    DEFAULT_CLASS_UID,
    DEFAULT_STAGING_API_BASE,
    VISDRONE_CLASS_NAMES,
)


def _write_image(
    path: Path,
    *,
    size: tuple[int, int] = (100, 100),
    color: tuple[int, int, int] = (255, 255, 255),
) -> None:
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


def _build_coco_fixture(root: Path) -> Path:
    for split in ("train", "valid", "test"):
        split_dir = root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        _write_image(split_dir / f"{split}_positive.jpg")
        _write_image(split_dir / f"{split}_empty.jpg", color=(240, 240, 240))
        _write_coco_annotations(
            split_dir / "_annotations.coco.json",
            images=[
                {
                    "id": 1,
                    "file_name": f"{split}_positive.jpg",
                    "width": 100,
                    "height": 100,
                    "extra": {"name": f"{split}_positive.jpg"},
                },
                {
                    "id": 2,
                    "file_name": f"{split}_empty.jpg",
                    "width": 100,
                    "height": 100,
                    "extra": {"name": f"{split}_empty.jpg"},
                },
            ],
            annotations=[
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [10, 12, 30, 20],
                    "area": 600,
                    "segmentation": [],
                    "iscrowd": 0,
                }
            ],
        )
    return root


def _write_visdrone_sequence(
    split_dir: Path,
    *,
    sequence_name: str,
    frame_count: int,
    annotations: list[dict[str, object]],
    size: tuple[int, int] = (100, 100),
) -> None:
    sequence_dir = split_dir / "sequences" / sequence_name
    annotation_dir = split_dir / "annotations"
    sequence_dir.mkdir(parents=True, exist_ok=True)
    annotation_dir.mkdir(parents=True, exist_ok=True)

    for frame_index in range(1, frame_count + 1):
        _write_image(sequence_dir / f"{frame_index:07d}.jpg", size=size)

    lines = []
    for annotation in annotations:
        bbox = annotation["bbox"]
        lines.append(
            ",".join(
                [
                    str(annotation["frame_index"]),
                    str(annotation["track_id"]),
                    str(bbox[0]),
                    str(bbox[1]),
                    str(bbox[2]),
                    str(bbox[3]),
                    "1",
                    str(annotation["category_id"]),
                    str(annotation.get("truncation", 0)),
                    str(annotation.get("occlusion", 0)),
                ]
            )
        )
    (annotation_dir / f"{sequence_name}.txt").write_text(
        ("\n".join(lines) + "\n") if lines else "",
        encoding="utf-8",
    )


def _build_visdrone_fixture(root: Path) -> Path:
    train_dir = root / "VisDrone2019-VID-train"
    val_dir = root / "VisDrone2019-VID-val"
    test_dir = root / "VisDrone2019-VID-test-dev"

    _write_visdrone_sequence(
        train_dir,
        sequence_name="uav0000013_00000_v",
        frame_count=6,
        annotations=[
            {
                "frame_index": 1,
                "track_id": 1,
                "bbox": [10, 20, 30, 20],
                "category_id": 4,
                "truncation": 0,
                "occlusion": 1,
            },
            {
                "frame_index": 1,
                "track_id": 99,
                "bbox": [1, 1, 5, 5],
                "category_id": 11,
                "truncation": 0,
                "occlusion": 0,
            },
            {
                "frame_index": 6,
                "track_id": 2,
                "bbox": [55, 60, 10, 12],
                "category_id": 5,
                "truncation": 1,
                "occlusion": 2,
            },
        ],
    )
    _write_visdrone_sequence(
        train_dir,
        sequence_name="uav0000014_00000_v",
        frame_count=1,
        annotations=[],
    )
    _write_visdrone_sequence(
        val_dir,
        sequence_name="uav0000086_00000_v",
        frame_count=2,
        annotations=[
            {
                "frame_index": 1,
                "track_id": 3,
                "bbox": [20, 25, 18, 14],
                "category_id": 9,
                "truncation": 0,
                "occlusion": 0,
            },
            {
                "frame_index": 2,
                "track_id": 4,
                "bbox": [2, 2, 3, 3],
                "category_id": 0,
                "truncation": 0,
                "occlusion": 0,
            },
        ],
    )
    _write_visdrone_sequence(
        test_dir,
        sequence_name="uav0000101_03667_v",
        frame_count=1,
        annotations=[
            {
                "frame_index": 1,
                "track_id": 5,
                "bbox": [40, 40, 20, 18],
                "category_id": 10,
                "truncation": 0,
                "occlusion": 1,
            }
        ],
    )
    return root


def _write_yolo_labels(path: Path, rows: list[list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(" ".join(str(value) for value in row) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def _build_yolo_test_fixture(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "data.yaml").write_text("names:\n  0: aircraft\n", encoding="utf-8")
    test_images_dir = root / "test" / "images"
    test_labels_dir = root / "test" / "labels"
    _write_image(test_images_dir / "airport_positive.jpg")
    _write_image(test_images_dir / "airport_empty.jpg", color=(240, 240, 240))
    _write_yolo_labels(
        test_labels_dir / "airport_positive.txt",
        [[0, 0.50, 0.60, 0.20, 0.40]],
    )
    return root


def _boxes_for_row(row: dict[str, object]) -> list[dict[str, object]]:
    return json.loads(str(row["answer_boxes"]))


def _make_multiclass_row(image_path: Path) -> dict[str, object]:
    return {
        "image": Image.open(image_path).convert("RGB"),
        "answer_boxes": json.dumps(
            [
                {
                    "x_min": 0.10,
                    "y_min": 0.10,
                    "x_max": 0.30,
                    "y_max": 0.30,
                    "class_uid": "aerial_airport:car",
                    "class_name": "car",
                    "source_class_name": "car",
                },
                {
                    "x_min": 0.40,
                    "y_min": 0.40,
                    "x_max": 0.70,
                    "y_max": 0.70,
                    "class_uid": "aerial_airport:bus",
                    "class_name": "bus",
                    "source_class_name": "bus",
                },
            ]
        ),
        "source_collection": "unit",
        "source_dataset": "unit",
        "class_count": 2,
    }


class BuilderTests(unittest.TestCase):
    def test_yolo_builder_can_require_expected_output_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw_root = _build_yolo_test_fixture(Path(tmp) / "raw_dataset" / "hrplanesv2")
            with self.assertRaisesRegex(ValueError, "Missing required output split"):
                build_mod.build_dataset_dict_from_raw_dir(
                    raw_root,
                    output_dir=Path(tmp) / "out",
                    source_format="yolo_dir",
                    seed=42,
                    val_fraction=0.1,
                    test_fraction=0.1,
                    target_empty_fraction=0.0,
                    required_output_splits=("train", "validation", "test"),
                )

    def test_yolo_builder_supports_test_only_split_and_aircraft_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw_root = _build_yolo_test_fixture(Path(tmp) / "raw_dataset" / "hrplanesv2")
            dataset_dict, split_rows, synthetic_counts, source_split_names, raw_empty_row_counts = (
                build_mod.build_dataset_dict_from_raw_dir(
                    raw_root,
                    output_dir=Path(tmp) / "out",
                    source_format="yolo_dir",
                    seed=42,
                    val_fraction=0.1,
                    test_fraction=0.1,
                    target_empty_fraction=0.0,
                )
            )

        self.assertEqual(sorted(dataset_dict.keys()), ["test"])
        self.assertEqual(source_split_names, ["test"])
        self.assertEqual(len(dataset_dict["test"]), 2)
        self.assertEqual(synthetic_counts, {"test": 0})
        self.assertEqual(raw_empty_row_counts, {"test": 1})

        positive_row = next(row for row in split_rows["test"] if int(row["class_count"]) > 0)
        empty_row = next(row for row in split_rows["test"] if int(row["class_count"]) == 0)
        positive_boxes = _boxes_for_row(positive_row)

        self.assertEqual(str(positive_row["source_variant"]), "yolo_image")
        self.assertEqual(str(positive_row["source_split"]), "test")
        self.assertEqual(str(positive_row["source_image_id"]), "airport_positive")
        self.assertEqual(positive_boxes[0]["class_name"], DEFAULT_CLASS_NAME)
        self.assertEqual(positive_boxes[0]["class_uid"], DEFAULT_CLASS_UID)
        self.assertEqual(_boxes_for_row(empty_row), [])

    def test_visdrone_builder_maps_splits_keeps_empty_frames_and_preserves_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw_root = _build_visdrone_fixture(Path(tmp) / "raw_dataset")
            dataset_dict, split_rows, synthetic_counts, source_split_names, raw_empty_row_counts = (
                build_mod.build_dataset_dict_from_raw_dir(
                    raw_root,
                    output_dir=Path(tmp) / "out",
                    source_format="visdrone_vid",
                    seed=42,
                    val_fraction=0.1,
                    test_fraction=0.1,
                    target_empty_fraction=0.0,
                    frame_stride_train=5,
                    frame_stride_validation=1,
                    frame_stride_test=1,
                )
            )

        self.assertEqual(sorted(dataset_dict.keys()), ["test", "train", "validation"])
        self.assertEqual(
            source_split_names,
            ["VisDrone2019-VID-train", "VisDrone2019-VID-val", "VisDrone2019-VID-test-dev"],
        )
        self.assertEqual(len(dataset_dict["train"]), 3)
        self.assertEqual(len(dataset_dict["validation"]), 2)
        self.assertEqual(len(dataset_dict["test"]), 1)
        self.assertEqual(synthetic_counts, {"train": 0, "validation": 0, "test": 0})
        self.assertEqual(raw_empty_row_counts, {"train": 1, "validation": 1, "test": 0})

        frame1 = next(row for row in split_rows["train"] if int(row["source_frame_index"]) == 1)
        frame6 = next(row for row in split_rows["train"] if int(row["source_frame_index"]) == 6)
        empty_row = next(row for row in split_rows["train"] if int(row["class_count"]) == 0)

        self.assertEqual(str(frame1["source_variant"]), "visdrone_vid_frame")
        self.assertEqual(str(frame1["source_split"]), "VisDrone2019-VID-train")
        self.assertEqual(str(frame1["source_sequence_name"]), "uav0000013_00000_v")
        self.assertEqual(str(frame1["source_frame_name"]), "0000001.jpg")
        self.assertEqual(str(frame1["split_group_id"]), "sequence:uav0000013_00000_v")
        self.assertEqual(str(frame1["source_image_id"]), "uav0000013_00000_v__0000001")

        frame1_boxes = _boxes_for_row(frame1)
        self.assertEqual(len(frame1_boxes), 1)
        self.assertEqual(frame1_boxes[0]["class_name"], "car")
        self.assertEqual(frame1_boxes[0]["track_id"], 1)
        self.assertEqual(frame1_boxes[0]["truncation"], 0)
        self.assertEqual(frame1_boxes[0]["occlusion"], 1)
        self.assertEqual(frame1_boxes[0]["source_category_id"], 4)

        frame6_boxes = _boxes_for_row(frame6)
        self.assertEqual(frame6_boxes[0]["class_name"], "car")
        self.assertEqual(frame6_boxes[0]["source_class_name"], "van")
        self.assertEqual(frame6_boxes[0]["track_id"], 2)
        self.assertEqual(frame6_boxes[0]["frame_index"], 6)

        self.assertEqual(int(empty_row["source_frame_index"]), 1)
        self.assertEqual(str(empty_row["source_sequence_name"]), "uav0000014_00000_v")
        self.assertEqual(_boxes_for_row(empty_row), [])

    def test_legacy_airport_builder_mode_still_works(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw_root = _build_coco_fixture(Path(tmp) / "raw_dataset" / "Aerial Airport.coco")
            dataset_dict, split_rows, synthetic_counts, source_split_names, raw_empty_row_counts = (
                build_mod.build_dataset_dict_from_raw_dir(
                    raw_root,
                    output_dir=Path(tmp) / "out",
                    source_format="airport_coco",
                    seed=42,
                    val_fraction=0.1,
                    test_fraction=0.1,
                    target_empty_fraction=0.1,
                )
            )

        self.assertEqual(source_split_names, ["test", "train", "valid"])
        self.assertEqual(len(dataset_dict["train"]), 2)
        self.assertEqual(len(dataset_dict["validation"]), 2)
        self.assertEqual(len(dataset_dict["test"]), 2)
        self.assertEqual(synthetic_counts, {"train": 0, "validation": 0, "test": 0})
        self.assertEqual(raw_empty_row_counts, {"train": 1, "validation": 1, "test": 1})

        positive_row = next(row for row in split_rows["train"] if int(row["class_count"]) > 0)
        positive_boxes = _boxes_for_row(positive_row)
        self.assertEqual(positive_boxes[0]["class_name"], DEFAULT_CLASS_NAME)
        self.assertEqual(positive_boxes[0]["class_uid"], DEFAULT_CLASS_UID)

    def test_main_writes_visdrone_metadata_and_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw_root = _build_visdrone_fixture(Path(tmp) / "raw_dataset")
            output_dir = Path(tmp) / "out"
            build_mod.main(
                [
                    "--raw-dataset-dir",
                    str(raw_root),
                    "--source-format",
                    "visdrone_vid",
                    "--output-dir",
                    str(output_dir),
                    "--frame-stride-train",
                    "5",
                    "--frame-stride-validation",
                    "1",
                    "--frame-stride-test",
                    "1",
                    "--target-empty-fraction",
                    "0.0",
                    "--push-to-hub",
                    "",
                ]
            )

            metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
            stats = json.loads((output_dir / "stats.json").read_text(encoding="utf-8"))
            dataset_dict = load_from_disk(str(output_dir))

        self.assertEqual(metadata["source_format"], "visdrone_vid")
        self.assertEqual(metadata["source_split_names"], ["VisDrone2019-VID-train", "VisDrone2019-VID-val", "VisDrone2019-VID-test-dev"])
        self.assertEqual(metadata["frame_stride_train"], 5)
        self.assertEqual(metadata["frame_stride_validation"], 1)
        self.assertEqual(metadata["frame_stride_test"], 1)
        self.assertEqual(metadata["pre_synthetic_split_sizes"], {"train": 3, "validation": 2, "test": 1})
        self.assertEqual(metadata["synthetic_negative_counts"], {"train": 0, "validation": 0, "test": 0})
        self.assertEqual(stats["source_format"], "visdrone_vid")
        self.assertEqual(stats["class_catalog"], ["bus", "car", "motor"])
        self.assertEqual(stats["raw_empty_row_counts"], {"train": 1, "validation": 1, "test": 0})
        self.assertEqual(len(dataset_dict["train"]), 3)
        self.assertEqual(len(dataset_dict["validation"]), 2)
        self.assertEqual(len(dataset_dict["test"]), 1)


class WrapperConfigTests(unittest.TestCase):
    def test_build_wrapper_defaults_target_visdrone(self) -> None:
        args = build_mod.parse_args([])
        self.assertEqual(args.source_format, "visdrone_vid")
        self.assertEqual(args.raw_dataset_dir, "aerial_airport/raw_dataset")
        self.assertEqual(args.output_dir, "aerial_airport/outputs/maxs-m87_visdrone_vid_frames_car_van_merged_v2")
        self.assertEqual(args.frame_stride_train, 5)
        self.assertEqual(args.frame_stride_validation, 1)
        self.assertEqual(args.frame_stride_test, 1)
        self.assertEqual(args.target_empty_fraction, 0.0)

    def test_default_train_and_benchmark_configs_parse_to_visdrone_settings(self) -> None:
        point_args = train_point_mod.parse_args([])
        detect_args = train_detect_mod.parse_args([])
        point_bench_args = bench_point_mod.parse_args([])
        detect_bench_args = bench_detect_mod.parse_args([])

        self.assertEqual(point_args.dataset_name, "maxs-m87/visdrone_vid_frames_car_van_merged_v2")
        self.assertEqual(point_args.dataset_path, "aerial_airport/outputs/maxs-m87_visdrone_vid_frames_car_van_merged_v2")
        self.assertEqual(point_args.class_names_file, "aerial_airport/configs/visdrone/visdrone_class_catalog.json")
        self.assertEqual(point_args.rank, 16)
        self.assertEqual(point_args.batch_size, 16)
        self.assertEqual(point_args.group_size, 8)
        self.assertAlmostEqual(point_args.lr, 1e-4, places=8)
        self.assertEqual(point_args.num_steps, 200)
        self.assertEqual(point_args.reward_metric, "f1")
        self.assertEqual(point_args.point_prompt_style, "class_name")
        self.assertAlmostEqual(point_args.fn_penalty_exponent, 2.0, places=8)
        self.assertEqual(point_args.neg_prompts_per_empty, 1)
        self.assertEqual(point_args.neg_prompts_per_nonempty, 0)
        self.assertAlmostEqual(point_args.neg_reward_weight, 0.15, places=8)
        self.assertEqual(point_args.base_url, DEFAULT_STAGING_API_BASE)

        self.assertEqual(detect_args.dataset_name, "maxs-m87/visdrone_vid_frames_car_van_merged_v2")
        self.assertEqual(detect_args.dataset_path, "aerial_airport/outputs/maxs-m87_visdrone_vid_frames_car_van_merged_v2")
        self.assertEqual(detect_args.class_names_file, "aerial_airport/configs/visdrone/visdrone_class_catalog.json")
        self.assertEqual(detect_args.rank, 16)
        self.assertEqual(detect_args.batch_size, 8)
        self.assertEqual(detect_args.group_size, 4)
        self.assertAlmostEqual(detect_args.lr, 5e-5, places=8)
        self.assertEqual(detect_args.num_steps, 500)
        self.assertTrue(detect_args.runtime_tiling)
        self.assertEqual(detect_args.tile_grid_size, 3)
        self.assertAlmostEqual(detect_args.tile_overlap, 0.1, places=8)
        self.assertAlmostEqual(detect_args.tile_box_merge_iou, 0.5, places=8)
        self.assertEqual(detect_args.reward_metric, "f1")
        self.assertEqual(detect_args.selection_metric, "f1")
        self.assertAlmostEqual(detect_args.fn_penalty_exponent, 2.0, places=8)
        self.assertAlmostEqual(detect_args.fp_penalty_exponent, 1.0, places=8)
        self.assertTrue(detect_args.run_final_test)
        self.assertEqual(detect_args.base_url, DEFAULT_STAGING_API_BASE)

        self.assertEqual(point_bench_args.dataset_name, "")
        self.assertEqual(point_bench_args.dataset_path, "aerial_airport/outputs/maxs-m87_visdrone_vid_frames_car_van_merged_v2")
        self.assertEqual(point_bench_args.class_names_file, "aerial_airport/configs/visdrone/visdrone_class_catalog.json")
        self.assertEqual(point_bench_args.split, "test")
        self.assertEqual(point_bench_args.skill, "point")
        self.assertEqual(point_bench_args.point_prompt_style, "class_name")

        self.assertEqual(detect_bench_args.dataset_name, "")
        self.assertEqual(detect_bench_args.dataset_path, "aerial_airport/outputs/maxs-m87_visdrone_vid_frames_car_van_merged_v2")
        self.assertEqual(detect_bench_args.class_names_file, "aerial_airport/configs/visdrone/visdrone_class_catalog.json")
        self.assertEqual(detect_bench_args.include_classes, [])
        self.assertEqual(detect_bench_args.exclude_classes, [])
        self.assertEqual(detect_bench_args.split, "test")
        self.assertEqual(detect_bench_args.skill, "detect")
        self.assertTrue(detect_bench_args.runtime_tiling)

    def test_legacy_configs_still_parse(self) -> None:
        legacy_root = REPO_ROOT / "aerial_airport" / "configs" / "legacy"
        build_args = build_mod.parse_args(["--config", str(legacy_root / "build_aerial_airport_hf_dataset_default.json")])
        tiling_args = build_mod.parse_args(["--config", str(legacy_root / "build_aerial_airport_hf_dataset_tiling.json")])
        train_args = train_point_mod.parse_args(["--config", str(legacy_root / "train_aerial_airport_point_default.json")])
        detect_bench_args = bench_detect_mod.parse_args(["--config", str(legacy_root / "benchmark_aerial_airport_detect_default.json")])

        self.assertEqual(build_args.source_format, "airport_coco")
        self.assertEqual(build_args.raw_dataset_dir, "aerial_airport/raw_dataset/Aerial Airport.coco")
        self.assertEqual(build_args.output_dir, "aerial_airport/outputs/maxs-m87_aerial_airport_point_v2")
        self.assertEqual(tiling_args.source_format, "airport_coco")
        self.assertEqual(tiling_args.tiling, "2x2")
        self.assertEqual(train_args.dataset_name, "maxs-m87/aerial_airport_point_v2")
        self.assertEqual(train_args.dataset_path, "aerial_airport/outputs/maxs-m87_aerial_airport_point_v2")
        self.assertEqual(detect_bench_args.dataset_name, "maxs-m87/aerial_airport_point_v2")
        self.assertEqual(detect_bench_args.dataset_path, "aerial_airport/outputs/maxs-m87_aerial_airport_point_v2")

    def test_visdrone_isolated_configs_parse(self) -> None:
        visdrone_root = REPO_ROOT / "aerial_airport" / "configs" / "visdrone"
        detect_control_args = train_detect_mod.parse_args(["--config", str(visdrone_root / "train_aerial_airport_detect_control.json")])
        point_car_args = train_point_mod.parse_args(["--config", str(visdrone_root / "train_aerial_airport_point_car.json")])
        detect_car_args = train_detect_mod.parse_args(["--config", str(visdrone_root / "train_aerial_airport_detect_car.json")])
        bench_car_args = bench_detect_mod.parse_args(["--config", str(visdrone_root / "benchmark_aerial_airport_detect_car.json")])

        self.assertFalse(detect_control_args.runtime_tiling)
        self.assertAlmostEqual(detect_control_args.lr, 1e-4, places=8)
        self.assertEqual(detect_control_args.num_steps, 300)
        self.assertEqual(point_car_args.class_names_file, "aerial_airport/configs/visdrone/visdrone_class_catalog_car.json")
        self.assertEqual(detect_car_args.class_names_file, "aerial_airport/configs/visdrone/visdrone_class_catalog_car.json")
        self.assertEqual(bench_car_args.include_classes, ["car"])
        self.assertEqual(bench_car_args.class_names_file, "aerial_airport/configs/visdrone/visdrone_class_catalog_car.json")
        self.assertFalse((visdrone_root / "train_aerial_airport_detect_van.json").exists())
        self.assertFalse((visdrone_root / "benchmark_aerial_airport_detect_van.json").exists())

    def test_hrplanesv2_full_build_and_train_configs_parse(self) -> None:
        hrplanes_root = REPO_ROOT / "aerial_airport" / "configs" / "hrplanesv2"
        build_args = build_mod.parse_args(["--config", str(hrplanes_root / "build_aerial_airport_hf_dataset_full.json")])
        point_args = train_point_mod.parse_args(["--config", str(hrplanes_root / "cicd_train_aerial_airport_point_hrplanesv2_best.json")])
        detect_args = train_detect_mod.parse_args(["--config", str(hrplanes_root / "cicd_train_aerial_airport_detect_hrplanesv2_best.json")])

        self.assertEqual(build_args.source_format, "yolo_dir")
        self.assertEqual(build_args.raw_dataset_dir, "aerial_airport/raw_dataset/hrplanesv2")
        self.assertEqual(build_args.output_dir, "aerial_airport/outputs/hrplanesv2_full_v1")
        self.assertEqual(build_args.require_output_splits, ["train", "validation", "test"])

        self.assertEqual(point_args.dataset_path, "aerial_airport/outputs/hrplanesv2_full_v1")
        self.assertEqual(point_args.dataset_name, "")
        self.assertEqual(point_args.class_names_file, "aerial_airport/configs/hrplanesv2/hrplanesv2_class_catalog.json")
        self.assertTrue(point_args.runtime_tiling)
        self.assertTrue(point_args.off_policy)
        self.assertEqual(point_args.batch_size, 32)
        self.assertEqual(point_args.group_size, 8)
        self.assertAlmostEqual(point_args.lr, 5e-5, places=10)

        self.assertEqual(detect_args.dataset_path, "aerial_airport/outputs/hrplanesv2_full_v1")
        self.assertEqual(detect_args.dataset_name, "")
        self.assertEqual(detect_args.class_names_file, "aerial_airport/configs/hrplanesv2/hrplanesv2_class_catalog.json")
        self.assertFalse(detect_args.runtime_tiling)
        self.assertFalse(detect_args.off_policy)
        self.assertEqual(detect_args.batch_size, 8)
        self.assertEqual(detect_args.group_size, 4)
        self.assertAlmostEqual(detect_args.lr, 1e-4, places=10)

    def test_detect_prompt_generation_uses_plain_class_names(self) -> None:
        self.assertEqual(train_detect_mod._prompt_for_class("car"), "car")
        self.assertEqual(train_detect_mod._prompt_for_class("awning-tricycle"), "awning-tricycle")
        self.assertEqual(bench_detect_mod._prompt_for_class("motor"), "motor")
        self.assertNotIn("icon", train_detect_mod._prompt_for_class("car"))

    def test_point_and_detect_wrappers_filter_boxes_by_class_catalog(self) -> None:
        visdrone_root = REPO_ROOT / "aerial_airport" / "configs" / "visdrone"
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "sample.jpg"
            _write_image(image_path)
            row = _make_multiclass_row(image_path)

            train_point_mod.parse_args(["--config", str(visdrone_root / "train_aerial_airport_point_car.json")])
            point_sample = train_point_mod._to_base_sample(row)
            self.assertIsNotNone(point_sample)
            assert point_sample is not None
            self.assertEqual([box.class_name for box in point_sample.boxes], ["car"])

            train_detect_mod.parse_args(["--config", str(visdrone_root / "train_aerial_airport_detect_car.json")])
            detect_sample = train_detect_mod._to_base_sample(row)
            self.assertIsNotNone(detect_sample)
            assert detect_sample is not None
            self.assertEqual([box.class_name for box in detect_sample.boxes], ["car"])


class RuntimeTilingTests(unittest.TestCase):
    def test_runtime_tiling_windows_cover_full_image_with_expected_overlap(self) -> None:
        windows = tiling_mod.build_tile_windows(width=100, height=100, grid_size=3, overlap=0.1)
        self.assertEqual(len(windows), 9)
        self.assertAlmostEqual(windows[0].x_min, 0.0, places=8)
        self.assertAlmostEqual(windows[0].y_min, 0.0, places=8)
        self.assertAlmostEqual(windows[-1].x_max, 1.0, places=8)
        self.assertAlmostEqual(windows[-1].y_max, 1.0, places=8)
        self.assertLess(windows[1].x_min, windows[0].x_max)
        self.assertAlmostEqual(windows[0].x_max - windows[1].x_min, 0.0357142857, places=6)

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


class CommonTests(unittest.TestCase):
    def test_visdrone_class_catalog_matches_official_training_classes(self) -> None:
        self.assertEqual(
            VISDRONE_CLASS_NAMES,
            [
                "pedestrian",
                "people",
                "bicycle",
                "car",
                "truck",
                "tricycle",
                "awning-tricycle",
                "bus",
                "motor",
            ],
        )


if __name__ == "__main__":
    unittest.main()
