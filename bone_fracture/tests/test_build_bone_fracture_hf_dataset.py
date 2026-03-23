from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from datasets import load_from_disk
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bone_fracture import build_bone_fracture_hf_dataset as mod
from bone_fracture.common import DEFAULT_POINT_CLASS_NAME, DEFAULT_POINT_CLASS_UID


def _write_sample(
    split_dir: Path,
    stem: str,
    *,
    size: tuple[int, int],
    boxes: list[tuple[str, tuple[int, int, int, int]]],
) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    image_path = split_dir / f"{stem}.jpg"
    Image.new("RGB", size, color=(255, 255, 255)).save(image_path)

    width, height = size
    objects = []
    for class_name, (x_min, y_min, x_max, y_max) in boxes:
        objects.append(
            f"""
    <object>
        <name>{class_name}</name>
        <bndbox>
            <xmin>{x_min}</xmin>
            <ymin>{y_min}</ymin>
            <xmax>{x_max}</xmax>
            <ymax>{y_max}</ymax>
        </bndbox>
    </object>""".rstrip()
        )
    xml_payload = f"""<annotation>
    <folder>{split_dir.name}</folder>
    <filename>{image_path.name}</filename>
    <size>
        <width>{width}</width>
        <height>{height}</height>
        <depth>3</depth>
    </size>
{''.join(objects)}
</annotation>
"""
    (split_dir / f"{stem}.xml").write_text(xml_payload, encoding="utf-8")


def _make_voc_export_root(root: Path) -> Path:
    export_root = root / "rf_export"
    _write_sample(
        export_root / "train",
        "train_a",
        size=(100, 80),
        boxes=[
            ("fracture", (10, 20, 30, 60)),
            ("angle", (50, 10, 90, 40)),
        ],
    )
    _write_sample(
        export_root / "valid",
        "valid_a",
        size=(120, 100),
        boxes=[],
    )
    _write_sample(
        export_root / "test",
        "test_a",
        size=(64, 64),
        boxes=[("line", (8, 8, 32, 56))],
    )
    return export_root


def _save_image(path: Path, *, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(255, 255, 255)).save(path)


def _make_coco_train_only_export(root: Path) -> Path:
    export_root = root / "bone fracture.coco"
    train_dir = export_root / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    images = [
        {"id": 1, "file_name": "101_jpg.rf.aaa111.jpg", "width": 100, "height": 80},
        {"id": 2, "file_name": "101_jpg.rf.bbb222.jpg", "width": 100, "height": 80},
        {"id": 3, "file_name": "202_jpg.rf.ccc333.jpg", "width": 110, "height": 88},
        {"id": 4, "file_name": "303_jpg.rf.ddd444.jpg", "width": 120, "height": 90},
        {"id": 5, "file_name": "404_jpg.rf.eee555.jpg", "width": 90, "height": 90},
        {"id": 6, "file_name": "505_jpg.rf.fff666.jpg", "width": 90, "height": 90},
        {"id": 7, "file_name": "606_jpg.rf.ggg777.jpg", "width": 90, "height": 90},
        {"id": 8, "file_name": "707_jpg.rf.hhh888.jpg", "width": 90, "height": 90},
    ]
    for image in images:
        _save_image(train_dir / image["file_name"], size=(image["width"], image["height"]))
    payload = {
        "images": images,
        "annotations": [
            {"id": 11, "image_id": 1, "category_id": 2, "bbox": [10, 20, 20, 40]},
            {"id": 12, "image_id": 2, "category_id": 3, "bbox": [20, 10, 40, 10]},
            {"id": 13, "image_id": 3, "category_id": 1, "bbox": [12, 18, 48, 36]},
            {"id": 14, "image_id": 4, "category_id": 4, "bbox": [8, 8, 42, 52]},
            {"id": 15, "image_id": 4, "category_id": 3, "bbox": [50, 15, 30, 25]},
        ],
        "categories": [
            {"id": 0, "name": "bone-fracture", "supercategory": "none"},
            {"id": 1, "name": "angle", "supercategory": "bone-fracture"},
            {"id": 2, "name": "fracture", "supercategory": "bone-fracture"},
            {"id": 3, "name": "line", "supercategory": "bone-fracture"},
            {"id": 4, "name": "messed_up_angle", "supercategory": "bone-fracture"},
        ],
    }
    (train_dir / "_annotations.coco.json").write_text(json.dumps(payload), encoding="utf-8")
    return export_root


def _make_coco_explicit_split_export(root: Path) -> Path:
    export_root = root / "bone fracture.coco"
    split_defs = {
        "train": {
            "images": [{"id": 1, "file_name": "train_angle.jpg", "width": 100, "height": 80}],
            "annotations": [{"id": 11, "image_id": 1, "category_id": 1, "bbox": [12, 18, 48, 36]}],
        },
        "valid": {
            "images": [{"id": 2, "file_name": "valid_messed.jpg", "width": 110, "height": 88}],
            "annotations": [{"id": 12, "image_id": 2, "category_id": 4, "bbox": [8, 8, 42, 52]}],
        },
        "test": {
            "images": [{"id": 3, "file_name": "test_mixed.jpg", "width": 120, "height": 90}],
            "annotations": [
                {"id": 13, "image_id": 3, "category_id": 1, "bbox": [20, 10, 40, 20]},
                {"id": 14, "image_id": 3, "category_id": 3, "bbox": [60, 18, 30, 24]},
            ],
        },
    }
    categories = [
        {"id": 0, "name": "bone-fracture", "supercategory": "none"},
        {"id": 1, "name": "angle", "supercategory": "bone-fracture"},
        {"id": 2, "name": "fracture", "supercategory": "bone-fracture"},
        {"id": 3, "name": "line", "supercategory": "bone-fracture"},
        {"id": 4, "name": "messed_up_angle", "supercategory": "bone-fracture"},
    ]
    for split_name, split_payload in split_defs.items():
        split_dir = export_root / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for image in split_payload["images"]:
            _save_image(split_dir / image["file_name"], size=(image["width"], image["height"]))
        payload = {
            "images": split_payload["images"],
            "annotations": split_payload["annotations"],
            "categories": categories,
        }
        (split_dir / "_annotations.coco.json").write_text(json.dumps(payload), encoding="utf-8")
    return export_root


class BuildBoneFracturePointDatasetTests(unittest.TestCase):
    def test_resolve_dir_keeps_module_prefixed_paths_rooted_at_repo(self) -> None:
        resolved = mod._resolve_dir("bone_fracture/outputs/demo", fallback_name="unused")
        self.assertEqual(resolved, REPO_ROOT / "bone_fracture" / "outputs" / "demo")

    def test_rest_export_fallback_extracts_zip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            download_dir = tmp_path / "download"

            archive_buffer = io.BytesIO()
            with zipfile.ZipFile(archive_buffer, "w") as archive:
                archive.writestr("rf_export/train/sample.jpg", b"fake-image")
                archive.writestr(
                    "rf_export/train/sample.xml",
                    """<annotation>
<filename>sample.jpg</filename>
<size><width>100</width><height>80</height><depth>3</depth></size>
<object><name>fracture</name><bndbox><xmin>10</xmin><ymin>20</ymin><xmax>30</xmax><ymax>40</ymax></bndbox></object>
</annotation>""",
                )
                archive.writestr("rf_export/valid/sample.jpg", b"fake-image")
                archive.writestr(
                    "rf_export/valid/sample.xml",
                    """<annotation>
<filename>sample.jpg</filename>
<size><width>100</width><height>80</height><depth>3</depth></size>
</annotation>""",
                )
                archive.writestr("rf_export/test/sample.jpg", b"fake-image")
                archive.writestr(
                    "rf_export/test/sample.xml",
                    """<annotation>
<filename>sample.jpg</filename>
<size><width>100</width><height>80</height><depth>3</depth></size>
<object><name>line</name><bndbox><xmin>10</xmin><ymin>20</ymin><xmax>30</xmax><ymax>40</ymax></bndbox></object>
</annotation>""",
                )

            class _Resp:
                def __init__(self, payload: bytes) -> None:
                    self._payload = payload

                def read(self) -> bytes:
                    return self._payload

                def __enter__(self) -> "_Resp":
                    return self

                def __exit__(self, exc_type, exc, tb) -> None:
                    return None

            responses = [
                _Resp(json.dumps({"export": {"link": "https://example.com/export.zip"}}).encode("utf-8")),
                _Resp(archive_buffer.getvalue()),
            ]

            def _fake_urlopen(url: str):
                self.assertTrue(url)
                return responses.pop(0)

            with patch.object(mod.urllib.request, "urlopen", side_effect=_fake_urlopen):
                export_root = mod._download_roboflow_export_via_rest(
                    workspace="roboflow-100",
                    project="bone-fracture-7fylg",
                    version=2,
                    api_key="rf-test-key",
                    download_dir=download_dir,
                )

        self.assertEqual(export_root, download_dir / "rf_export")

    def test_build_dataset_dict_from_export_collapses_to_single_point_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            export_root = _make_voc_export_root(Path(tmp))

            dataset_dict, split_rows, class_names = mod.build_dataset_dict_from_export(
                export_root,
                workspace="roboflow-100",
                project="bone-fracture-7fylg",
                version=2,
            )

        self.assertEqual(list(dataset_dict.keys()), ["train", "validation", "test"])
        self.assertEqual(class_names, [DEFAULT_POINT_CLASS_NAME])
        self.assertEqual(len(dataset_dict["train"]), 1)
        self.assertEqual(len(dataset_dict["validation"]), 1)
        self.assertEqual(len(dataset_dict["test"]), 1)

        train_row = split_rows["train"][0]
        self.assertEqual(train_row["class_count"], 2)
        payload = json.loads(train_row["answer_boxes"])
        self.assertEqual(len(payload), 2)
        self.assertEqual({item["class_name"] for item in payload}, {DEFAULT_POINT_CLASS_NAME})
        self.assertEqual({item["class_uid"] for item in payload}, {DEFAULT_POINT_CLASS_UID})
        self.assertEqual({item["source_class_name"] for item in payload}, {"fracture", "angle"})
        self.assertEqual({item["attributes"][0]["value"] for item in payload}, {DEFAULT_POINT_CLASS_NAME})

        validation_row = split_rows["validation"][0]
        self.assertEqual(validation_row["class_count"], 0)
        self.assertEqual(json.loads(validation_row["answer_boxes"]), [])

    def test_build_dataset_dict_from_coco_train_only_export_creates_grouped_stratified_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            export_root = _make_coco_train_only_export(Path(tmp))

            dataset_dict, split_rows, class_names = mod.build_dataset_dict_from_coco_export(
                export_root,
                workspace="roboflow-100",
                project="bone-fracture-7fylg",
                version=2,
                seed=42,
                single_split_val_fraction=0.67,
                single_split_test_fraction=0.5,
            )

        self.assertEqual(class_names, [DEFAULT_POINT_CLASS_NAME])
        self.assertEqual(sum(len(rows) for rows in split_rows.values()), 8)
        self.assertEqual(list(dataset_dict.keys()), ["train", "validation", "test"])

        train_groups = {row["split_group_id"] for row in split_rows["train"]}
        val_groups = {row["split_group_id"] for row in split_rows["validation"]}
        test_groups = {row["split_group_id"] for row in split_rows["test"]}
        self.assertFalse(train_groups & val_groups)
        self.assertFalse(train_groups & test_groups)
        self.assertFalse(val_groups & test_groups)

        grouped_101_splits = {
            split_name
            for split_name, rows in split_rows.items()
            for row in rows
            if row["source_base_id"] == "101_jpg"
        }
        self.assertEqual(grouped_101_splits, {"train"})

        empty_by_split = {
            split_name: sum(1 for row in rows if int(row["class_count"]) == 0)
            for split_name, rows in split_rows.items()
        }
        self.assertGreaterEqual(empty_by_split["validation"], 1)
        self.assertGreaterEqual(empty_by_split["test"], 1)

        multi_box_row = next(row for rows in split_rows.values() for row in rows if row["source_base_id"] == "303_jpg")
        payload = json.loads(multi_box_row["answer_boxes"])
        self.assertEqual(len(payload), 2)
        self.assertEqual({item["source_class_name"] for item in payload}, {"line", "messed_up_angle"})

    def test_main_writes_metadata_stats_and_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            export_root = _make_coco_train_only_export(tmp_path)
            output_dir = tmp_path / "out"

            mod.main(
                [
                    "--download-dir",
                    str(export_root),
                    "--output-dir",
                    str(output_dir),
                    "--push-to-hub",
                    "",
                    "--single-split-val-fraction",
                    "0.67",
                    "--single-split-test-fraction",
                    "0.5",
                ]
            )

            ds = load_from_disk(str(output_dir))
            metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
            stats = json.loads((output_dir / "stats.json").read_text(encoding="utf-8"))

        self.assertEqual(list(ds.keys()), ["train", "validation", "test"])
        self.assertEqual(metadata["class_catalog"][0]["class_name"], DEFAULT_POINT_CLASS_NAME)
        self.assertEqual(metadata["class_catalog"][0]["class_uid"], DEFAULT_POINT_CLASS_UID)
        self.assertEqual(metadata["default_skill"], "point")
        self.assertEqual(metadata["default_point_prompt_style"], "class_name")
        self.assertEqual(metadata["default_reward_metric"], "f1")
        self.assertNotIn("bone-fracture", metadata["raw_source_label_catalog"])
        self.assertEqual(metadata["raw_source_label_counts"], stats["raw_source_label_counts"])
        self.assertEqual(metadata["positive_row_counts"], stats["positive_row_counts"])
        self.assertEqual(metadata["empty_row_counts"], stats["empty_row_counts"])
        for split_name, split_size in stats["split_sizes"].items():
            self.assertEqual(
                split_size,
                stats["positive_row_counts"][split_name] + stats["empty_row_counts"][split_name],
            )

    def test_main_can_filter_to_angle_variants_and_use_custom_target_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            export_root = _make_coco_explicit_split_export(tmp_path)
            output_dir = tmp_path / "angle_only"

            mod.main(
                [
                    "--download-dir",
                    str(export_root),
                    "--output-dir",
                    str(output_dir),
                    "--push-to-hub",
                    "",
                    "--include-source-class-names",
                    "angle",
                    "messed up angle",
                    "--drop-empty-rows-after-filter",
                    "--target-class-name",
                    "point where the bone is broken",
                ]
            )

            ds = load_from_disk(str(output_dir))
            metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))

        self.assertEqual(list(ds.keys()), ["train", "validation", "test"])
        self.assertEqual(metadata["default_target_class_name"], "point where the bone is broken")
        self.assertEqual(metadata["default_target_prompt"], "point where the bone is broken")
        self.assertEqual(metadata["include_source_class_names"], ["angle", "messed up angle"])
        self.assertEqual(metadata["raw_source_label_catalog"], ["angle", "messed_up_angle"])
        self.assertEqual(metadata["class_catalog"][0]["class_name"], "point where the bone is broken")
        self.assertNotEqual(metadata["class_catalog"][0]["class_uid"], DEFAULT_POINT_CLASS_UID)
        self.assertEqual(metadata["class_catalog"][0]["prompt"], "point where the bone is broken")

        for split_name in ("train", "validation", "test"):
            self.assertEqual(len(ds[split_name]), 1)
            row = ds[split_name][0]
            payload = json.loads(row["answer_boxes"])
            self.assertTrue(payload)
            self.assertEqual({item["class_name"] for item in payload}, {"point where the bone is broken"})
            self.assertLessEqual({item["source_class_name"] for item in payload}, {"angle", "messed_up_angle"})
