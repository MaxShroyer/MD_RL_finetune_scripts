from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tictaktoe_QA import data_loader as mod


class LocalJsonlLoaderTests(unittest.TestCase):
    def test_load_split_rows_local_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = Path(tmp)
            jsonl_dir = dataset_dir / "jsonl"
            jsonl_dir.mkdir(parents=True, exist_ok=True)
            split_path = jsonl_dir / "test.jsonl"
            split_path.write_text(
                "\n".join(
                    [
                        json.dumps({"row_id": "r0", "task_type": "winner"}),
                        json.dumps({"row_id": "r1", "task_type": "best_move"}),
                    ]
                ),
                encoding="utf-8",
            )

            rows = mod.load_split_rows(
                dataset_source="local_jsonl",
                split_name="test",
                dataset_dir=dataset_dir,
                hf_dataset_repo_id="",
                hf_dataset_revision="",
                hf_token="",
                hf_cache_dir="",
            )
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["row_id"], "r0")


class HfImageResolutionTests(unittest.TestCase):
    def test_hf_row_uses_existing_image_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "img.png"
            Image.new("RGB", (8, 8), color=(255, 255, 255)).save(image_path)

            resolved = mod._resolve_hf_row_image_path(
                {"image": {"path": str(image_path)}},
                dataset_dir=None,
                image_cache_root=Path(tmp) / "cache",
            )
            self.assertEqual(resolved, image_path.resolve())

    def test_hf_row_persists_image_bytes_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            buf = io.BytesIO()
            Image.new("RGB", (8, 8), color=(0, 0, 0)).save(buf, format="PNG")
            image_bytes = buf.getvalue()
            cache_root = Path(tmp) / "cache"

            first = mod._resolve_hf_row_image_path(
                {"image": {"bytes": image_bytes}},
                dataset_dir=None,
                image_cache_root=cache_root,
            )
            second = mod._resolve_hf_row_image_path(
                {"image": {"bytes": image_bytes}},
                dataset_dir=None,
                image_cache_root=cache_root,
            )

            self.assertIsNotNone(first)
            self.assertEqual(first, second)
            self.assertTrue(first.is_file())  # type: ignore[union-attr]


if __name__ == "__main__":
    unittest.main()
