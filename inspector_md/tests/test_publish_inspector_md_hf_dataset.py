from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
from datasets import Dataset, DatasetDict, Features, Value
from PIL import Image

from inspector_md import publish_inspector_md_hf_dataset as publish_mod
from inspector_md import prompt_library


def _write_image(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 8), "red").save(path)
    return str(path)


def test_load_query_split_casts_image_and_drops_local_path(tmp_path: Path) -> None:
    image_path = _write_image(tmp_path / "sample.jpg")
    jsonl_path = tmp_path / "train.jsonl"
    jsonl_path.write_text(
        json.dumps(
            {
                "row_id": "row-1",
                "split": "train",
                "task_type": "proposal",
                "question": prompt_library.CANONICAL_QUERY_INSPECTION_REQUEST,
                "image_path": image_path,
                "inspection_request": prompt_library.CANONICAL_QUERY_INSPECTION_REQUEST,
                "asset_context": "Metal structure.",
                "spatial_refs_json": "[]",
                "reasoning_text": "",
                "hard_example": False,
                "target_text": "corrosion_rust | Rust visible in annotated area.",
                "target_format": "compact_text",
                "final_answer_json": "{\"proposals\": []}",
                "query_text_refresh_mode": "openrouter",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = publish_mod._load_query_split(jsonl_path)

    assert "image" in dataset.features
    assert str(dataset.features["image"]).startswith("Image")
    row = dataset[0]
    assert row["row_id"] == "row-1"
    assert row["target_text"] == "corrosion_rust | Rust visible in annotated area."
    assert row["image"].size == (12, 8)


def test_load_image_dataset_dict_casts_image_column(tmp_path: Path) -> None:
    image_path = _write_image(tmp_path / "sample.jpg")
    dataset_dir = tmp_path / "detect"
    dataset = Dataset.from_list(
        [
            {
                "image": image_path,
                "answer_boxes": "[]",
                "source_dataset": "seed",
                "source_collection": "seed",
                "source_variant": "seed",
                "source_is_synthetic": True,
                "source_split": "train",
                "source_image_id": "img-1",
                "source_base_id": "img-1",
                "split_group_id": "img-1",
                "class_count": 0,
            }
        ],
        features=Features(
            {
                "image": Value("string"),
                "answer_boxes": Value("string"),
                "source_dataset": Value("string"),
                "source_collection": Value("string"),
                "source_variant": Value("string"),
                "source_is_synthetic": Value("bool"),
                "source_split": Value("string"),
                "source_image_id": Value("string"),
                "source_base_id": Value("string"),
                "split_group_id": Value("string"),
                "class_count": Value("int32"),
            }
        ),
    )
    DatasetDict({"train": dataset}).save_to_disk(str(dataset_dir))

    converted = publish_mod._load_image_dataset_dict(dataset_dir)

    assert str(converted["train"].features["image"]).startswith("Image")
    assert converted["train"][0]["image"].size == (12, 8)


def test_build_dataset_card_mentions_private_guardrail() -> None:
    card = publish_mod._build_dataset_card(
        repo_id="maxs-m87/inspector_md_full_v1",
        private=True,
        config_summaries={
            "detect": {
                "task_label": "Detect",
                "split_counts": {"train": 1, "validation": 1, "test": 1},
                "metadata": {"query_text_refresh_mode": "openrouter"},
            }
        },
        build_summary={"detect_output_dir": "/tmp/detect"},
    )

    assert "private by default" in card
    assert "`detect`" in card


def test_parse_args_accepts_config_subset() -> None:
    args = publish_mod.parse_args(
        [
            "--config",
            "inspector_md/configs/publish_inspector_md_hf_dataset_default.json",
            "--configs",
            "query_proposal",
            "query_finding",
        ]
    )

    assert args.configs == ["query_proposal", "query_finding"]


def test_image_embed_storage_compat_patch_handles_null_mask(tmp_path: Path) -> None:
    image_path = _write_image(tmp_path / "sample.jpg")
    publish_mod._apply_hf_image_embed_storage_compat_patch()
    feature = publish_mod.HFImage()
    base_storage = pa.StructArray.from_arrays(
        [
            pa.array([None], type=pa.binary()),
            pa.array([image_path], type=pa.string()),
        ],
        names=["bytes", "path"],
    )
    storage = pa.chunked_array([base_storage])

    embedded = feature.embed_storage(storage)

    assert embedded.type == feature.pa_type
    payload = embedded.to_pylist()[0]
    assert payload["bytes"] is not None
    assert payload["path"] == Path(image_path).name


def test_push_dataset_dict_with_shard_fallback_retries_on_arrow_overflow() -> None:
    calls: list[str] = []

    class FakeDatasetDict:
        def push_to_hub(self, **kwargs) -> None:
            calls.append(str(kwargs["max_shard_size"]))
            if len(calls) == 1:
                raise RuntimeError("offset overflow while concatenating arrays, consider casting input from `binary` to `large_binary` first.")

    publish_mod._push_dataset_dict_with_shard_fallback(
        FakeDatasetDict(),  # type: ignore[arg-type]
        repo_id="maxs-m87/inspector_md_full_v1",
        config_name="query_proposal",
        set_default=False,
        private=True,
        token="token",
        commit_message="Add query_proposal config",
        embed_external_files=True,
        initial_max_shard_size="100MB",
    )

    assert calls == ["100MB", "25MB"]
