from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image

TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

import build_chess_qa_dataset as mod


def _piece(piece_name: str, square: str) -> dict:
    return {
        "piece": piece_name,
        "position": {
            "square": square,
            "x_center_norm": 0.5,
            "y_center_norm": 0.5,
            "bbox_norm": {"x_min": 0.4, "y_min": 0.4, "x_max": 0.6, "y_max": 0.6},
        },
    }


def _record(
    *,
    record_id: str,
    pieces: list[dict],
    split: str = "train",
    source_dataset: str = "dataset2_coco",
    source_label_format: str = "coco_bbox",
    image_name: str | None = None,
) -> dict:
    return {
        "record_id": record_id,
        "source_dataset": source_dataset,
        "source_split": split,
        "source_label_format": source_label_format,
        "source_image_id": image_name or f"{record_id}.jpg",
        "source_image_path": f"/tmp/{image_name or f'{record_id}.jpg'}",
        "pieces": pieces,
    }


def _write_image(path: Path, *, size: tuple[int, int] = (64, 64)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(255, 255, 255)).save(path)


def _write_osf_example(root: Path, *, split: str, stem: str, pieces: list[dict]) -> None:
    split_dir = root / split
    _write_image(split_dir / f"{stem}.png", size=(1200, 800))
    payload = {
        "camera": {"angle": 45},
        "corners": [[0, 0], [10, 0], [10, 10], [0, 10]],
        "fen": "8/8/8/8/8/8/8/8",
        "lighting": "bright",
        "pieces": pieces,
        "white_turn": True,
    }
    (split_dir / f"{stem}.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_coco_split(root: Path, *, split: str, payload: dict, image_names: list[str]) -> None:
    split_dir = root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "_annotations.coco.json").write_text(json.dumps(payload), encoding="utf-8")
    for name in image_names:
        _write_image(split_dir / name, size=(100, 100))


def test_clean_dataset2_records_dedupes_and_drops_conflicts() -> None:
    dedupable = _record(
        record_id="dedupe",
        pieces=[
            _piece("white_king", "e1"),
            _piece("white_king", "e1"),
            _piece("black_king", "e8"),
        ],
    )
    conflicting = _record(
        record_id="conflict",
        pieces=[
            _piece("white_king", "e4"),
            _piece("black_queen", "e4"),
        ],
    )

    cleaned, summary = mod._clean_dataset2_records([dedupable, conflicting])

    assert len(cleaned) == 1
    assert cleaned[0]["record_id"] == "dedupe"
    assert len(cleaned[0]["pieces"]) == 2
    assert summary["dataset2_pre_clean_records"] == 2
    assert summary["dataset2_post_clean_records"] == 1
    assert summary["duplicate_piece_square_labels_removed"] == 1
    assert summary["conflict_records_dropped"] == 1


def test_build_one_to_many_mixed_rows_creates_four_rows_per_image() -> None:
    record = _record(
        record_id="board",
        pieces=[
            _piece("white_king", "e1"),
            _piece("black_king", "e8"),
            _piece("white_pawn", "a2"),
        ],
    )
    split_records = {"train": [record], "val": [], "test": []}
    image_path_by_key = {mod._record_key(record): "/imges/board.jpg"}

    rows_by_split = mod._build_one_to_many_mixed_rows(
        split_records,
        image_path_by_key=image_path_by_key,
        seed=42,
    )

    train_rows = rows_by_split["train"]
    assert len(train_rows) == len(mod.TASK_TYPES)
    assert {row["task_type"] for row in train_rows} == set(mod.TASK_TYPES)
    presence_row = next(row for row in train_rows if row["task_type"] == "color_presence_check")
    assert json.loads(presence_row["final_answer_json"]) in (
        {"color": "white", "present": True},
        {"color": "black", "present": True},
    )


def test_build_and_write_v2_family_can_skip_piece_position_dataset(tmp_path: Path) -> None:
    record = _record(
        record_id="board",
        pieces=[
            _piece("white_king", "e1"),
            _piece("black_king", "e8"),
        ],
    )
    split_records = {"train": [record], "val": [], "test": []}
    output_dir = tmp_path / "outputs"

    mod._build_and_write_v2_family(
        output_dir=output_dir,
        split_records=split_records,
        image_path_by_key={mod._record_key(record): "/imges/board.jpg"},
        seed=42,
        allowed_sources={"dataset2_coco"},
        piece_dataset_name="piece_demo",
        mixed_dataset_name="mixed_demo",
        build_piece_position=False,
        export_hf_dataset=False,
        push_to_hub=False,
        hf_repo_id="",
        hf_token="",
        hf_private=False,
    )

    assert not (output_dir / "piece_demo").exists()
    mixed_metadata = json.loads((output_dir / "mixed_demo" / "metadata.json").read_text(encoding="utf-8"))
    assert mixed_metadata["split_counts"] == {"train": 4, "val": 0, "test": 0}
    assert mixed_metadata["task_counts"] == {task: 1 for task in sorted(mod.TASK_TYPES)}


def test_validate_jsonl_roundtrip_rejects_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"ok":true}\nnot-json\n', encoding="utf-8")

    try:
        mod._validate_jsonl_roundtrip({"train": path})
    except ValueError as exc:
        assert "Invalid JSONL row written" in str(exc)
    else:
        raise AssertionError("expected ValueError for malformed JSONL")


def test_build_metadata_merges_extra_metadata(tmp_path: Path) -> None:
    rows_by_split = {"train": [], "val": [], "test": []}
    jsonl_paths = {
        "train": tmp_path / "train.jsonl",
        "val": tmp_path / "val.jsonl",
        "test": tmp_path / "test.jsonl",
    }
    metadata = mod._build_metadata(
        dataset_name="demo",
        rows_by_split=rows_by_split,
        seed=7,
        jsonl_paths=jsonl_paths,
        extra_metadata={"cleaning": {"ok": True}},
    )

    assert metadata["cleaning"] == {"ok": True}


def test_main_builds_merged_v2_outputs_from_osf_when_only_osf_is_available(tmp_path: Path, monkeypatch) -> None:
    osf_root = tmp_path / "osfstorage-archive"
    for split in mod.SPLIT_ORDER:
        (osf_root / split).mkdir(parents=True, exist_ok=True)

    _write_osf_example(
        osf_root,
        split="train",
        stem="0000",
        pieces=[
            {"piece": "K", "square": "e1", "box": [10, 20, 30, 40]},
            {"piece": "k", "square": "e8", "box": [100, 120, 40, 50]},
        ],
    )
    _write_osf_example(
        osf_root,
        split="train",
        stem="0001",
        pieces=[
            {"piece": "Q", "square": "d1", "box": [40, 50, 30, 30]},
        ],
    )
    _write_osf_example(
        osf_root,
        split="val",
        stem="0002",
        pieces=[
            {"piece": "r", "square": "a8", "box": [20, 30, 20, 20]},
        ],
    )
    _write_osf_example(
        osf_root,
        split="test",
        stem="0003",
        pieces=[
            {"piece": "P", "square": "a2", "box": [60, 70, 20, 20]},
        ],
    )

    output_dir = tmp_path / "outputs"
    argv = [
        "build_chess_qa_dataset.py",
        "--dataset1-dir",
        str(tmp_path / "missing-dataset1"),
        "--dataset2-dir",
        str(tmp_path / "missing-dataset2"),
        "--osf-dir",
        str(osf_root),
        "--output-dir",
        str(output_dir),
        "--copy-images",
        "true",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    mod.main()

    merged_piece_root = output_dir / mod.V2_PIECE_POSITION_NAME
    merged_mixed_root = output_dir / mod.V2_MIXED_NAME
    piece_root = output_dir / mod.OSF_PIECE_POSITION_NAME
    mixed_root = output_dir / mod.OSF_MIXED_NAME
    assert merged_piece_root.exists()
    assert merged_mixed_root.exists()
    assert not piece_root.exists()
    assert not mixed_root.exists()

    merged_piece_metadata = json.loads((merged_piece_root / "metadata.json").read_text(encoding="utf-8"))
    merged_mixed_metadata = json.loads((merged_mixed_root / "metadata.json").read_text(encoding="utf-8"))
    assert merged_piece_metadata["split_counts"] == {"train": 2, "val": 1, "test": 1}
    assert merged_mixed_metadata["split_counts"] == {"train": 8, "val": 4, "test": 4}
    assert merged_piece_metadata["source_counts"] == {"osfstorage_archive": 4}
    assert merged_piece_metadata["source_inputs"] == ["osfstorage_archive"]
    assert merged_piece_metadata["split_strategy"]["type"] == "merged_input_sources"
    first_row = json.loads((merged_piece_root / "jsonl" / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert first_row["source_dataset"] == "osfstorage_archive"
    assert first_row["image_path"] == "/imges/osf__train__0000.png"
    assert (output_dir / "imges" / "osf__train__0000.png").exists()


def test_main_skips_missing_osf_and_keeps_dataset2_v2(tmp_path: Path, monkeypatch) -> None:
    dataset2_root = tmp_path / "dataset2"
    categories = [{"id": 1, "name": "white-king"}]
    train_payload = {
        "images": [{"id": 1, "file_name": "board.jpg", "width": 100, "height": 100}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 30, 40]}],
        "categories": categories,
    }
    empty_payload = {"images": [], "annotations": [], "categories": categories}
    _write_coco_split(dataset2_root, split="train", payload=train_payload, image_names=["board.jpg"])
    _write_coco_split(dataset2_root, split="valid", payload=empty_payload, image_names=[])
    _write_coco_split(dataset2_root, split="test", payload=empty_payload, image_names=[])

    output_dir = tmp_path / "outputs"
    argv = [
        "build_chess_qa_dataset.py",
        "--dataset1-dir",
        str(tmp_path / "missing-dataset1"),
        "--dataset2-dir",
        str(dataset2_root),
        "--osf-dir",
        str(tmp_path / "missing-osf"),
        "--output-dir",
        str(output_dir),
        "--copy-images",
        "true",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    mod.main()

    dataset2_metadata = json.loads(
        (output_dir / mod.V2_PIECE_POSITION_NAME / "metadata.json").read_text(encoding="utf-8")
    )
    assert dataset2_metadata["cleaning"]["dataset2_post_clean_records"] == 1
    assert not (output_dir / mod.OSF_PIECE_POSITION_NAME).exists()


def test_main_mixed_task_only_writes_only_merged_v2_mixed_output(tmp_path: Path, monkeypatch) -> None:
    dataset2_root = tmp_path / "dataset2"
    categories = [{"id": 1, "name": "white-king"}]
    train_payload = {
        "images": [{"id": 1, "file_name": "board.jpg", "width": 100, "height": 100}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 30, 40]}],
        "categories": categories,
    }
    empty_payload = {"images": [], "annotations": [], "categories": categories}
    _write_coco_split(dataset2_root, split="train", payload=train_payload, image_names=["board.jpg"])
    _write_coco_split(dataset2_root, split="valid", payload=empty_payload, image_names=[])
    _write_coco_split(dataset2_root, split="test", payload=empty_payload, image_names=[])

    output_dir = tmp_path / "outputs"
    argv = [
        "build_chess_qa_dataset.py",
        "--dataset1-dir",
        str(tmp_path / "missing-dataset1"),
        "--dataset2-dir",
        str(dataset2_root),
        "--osf-dir",
        str(tmp_path / "missing-osf"),
        "--output-dir",
        str(output_dir),
        "--copy-images",
        "true",
        "--mixed-task-only",
        "true",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    mod.main()

    assert not (output_dir / mod.V2_PIECE_POSITION_NAME).exists()
    mixed_root = output_dir / mod.V2_MIXED_NAME
    assert mixed_root.exists()
    mixed_metadata = json.loads((mixed_root / "metadata.json").read_text(encoding="utf-8"))
    assert mixed_metadata["split_counts"] == {"train": 4, "val": 0, "test": 0}
    assert mixed_metadata["source_counts"] == {"dataset2_coco": 4}
    assert not (output_dir / mod.OSF_MIXED_NAME).exists()


def test_main_merges_dataset2_and_osf_into_main_v2_outputs(tmp_path: Path, monkeypatch) -> None:
    dataset2_root = tmp_path / "dataset2"
    categories = [{"id": 1, "name": "white-king"}]
    train_payload = {
        "images": [{"id": 1, "file_name": "board.jpg", "width": 100, "height": 100}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 30, 40]}],
        "categories": categories,
    }
    empty_payload = {"images": [], "annotations": [], "categories": categories}
    _write_coco_split(dataset2_root, split="train", payload=train_payload, image_names=["board.jpg"])
    _write_coco_split(dataset2_root, split="valid", payload=empty_payload, image_names=[])
    _write_coco_split(dataset2_root, split="test", payload=empty_payload, image_names=[])

    osf_root = tmp_path / "osfstorage-archive"
    for split in mod.SPLIT_ORDER:
        (osf_root / split).mkdir(parents=True, exist_ok=True)
    _write_osf_example(
        osf_root,
        split="train",
        stem="0000",
        pieces=[{"piece": "K", "square": "e1", "box": [10, 20, 30, 40]}],
    )
    _write_osf_example(
        osf_root,
        split="val",
        stem="0001",
        pieces=[{"piece": "k", "square": "e8", "box": [20, 30, 20, 20]}],
    )

    output_dir = tmp_path / "outputs"
    argv = [
        "build_chess_qa_dataset.py",
        "--dataset1-dir",
        str(tmp_path / "missing-dataset1"),
        "--dataset2-dir",
        str(dataset2_root),
        "--osf-dir",
        str(osf_root),
        "--output-dir",
        str(output_dir),
        "--copy-images",
        "true",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    mod.main()

    merged_metadata = json.loads(
        (output_dir / mod.V2_PIECE_POSITION_NAME / "metadata.json").read_text(encoding="utf-8")
    )
    merged_rows = [
        json.loads(line)
        for line in (output_dir / mod.V2_PIECE_POSITION_NAME / "jsonl" / "train.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    assert merged_metadata["source_inputs"] == ["dataset2_coco", "osfstorage_archive"]
    assert merged_metadata["source_counts"] == {"dataset2_coco": 1, "osfstorage_archive": 2}
    assert merged_metadata["split_counts"] == {"train": 2, "val": 1, "test": 0}
    assert merged_metadata["split_strategy"]["type"] == "merged_input_sources"
    assert {row["source_dataset"] for row in merged_rows} == {"dataset2_coco", "osfstorage_archive"}
    assert not (output_dir / mod.OSF_PIECE_POSITION_NAME).exists()


def test_main_can_build_source_specific_osf_outputs_when_enabled(tmp_path: Path, monkeypatch) -> None:
    osf_root = tmp_path / "osfstorage-archive"
    for split in mod.SPLIT_ORDER:
        (osf_root / split).mkdir(parents=True, exist_ok=True)

    _write_osf_example(
        osf_root,
        split="train",
        stem="0000",
        pieces=[{"piece": "K", "square": "e1", "box": [10, 20, 30, 40]}],
    )

    output_dir = tmp_path / "outputs"
    argv = [
        "build_chess_qa_dataset.py",
        "--dataset2-dir",
        str(tmp_path / "missing-dataset2"),
        "--osf-dir",
        str(osf_root),
        "--output-dir",
        str(output_dir),
        "--copy-images",
        "true",
        "--build-source-specific-v2",
        "true",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    mod.main()

    assert (output_dir / mod.V2_PIECE_POSITION_NAME).exists()
    assert (output_dir / mod.V2_MIXED_NAME).exists()
    assert (output_dir / mod.OSF_PIECE_POSITION_NAME).exists()
    assert (output_dir / mod.OSF_MIXED_NAME).exists()
