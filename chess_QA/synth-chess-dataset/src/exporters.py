"""Export and summary helpers for generated chess QA datasets."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

REQUIRED_COLUMNS = (
    "row_id",
    "split",
    "task_type",
    "source_dataset",
    "image",
    "image_path",
    "question",
    "answer_text",
    "final_answer_json",
    "prompt_variant_id",
    "source_image_id",
    "source_label_format",
)

SPLIT_ORDER = ("train", "val", "test")


def _ensure_columns(row: dict[str, Any]) -> dict[str, Any]:
    payload = dict(row)
    for col in REQUIRED_COLUMNS:
        if col not in payload:
            raise ValueError(f"Row is missing required column '{col}'")
    return payload


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out_path


def write_jsonl(rows_by_split: dict[str, list[dict[str, Any]]], out_dir: str | Path) -> dict[str, Path]:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split in SPLIT_ORDER:
        rows = rows_by_split.get(split, [])
        path = root / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                payload = _ensure_columns(row)
                handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True))
                handle.write("\n")
        paths[split] = path
    return paths


def write_hf_dataset(
    rows_by_split: dict[str, list[dict[str, Any]]],
    out_dir: str | Path,
    *,
    push_to_hub: bool = False,
    repo_id: str = "",
    token: str = "",
    private: bool = False,
    config_name: str | None = None,
) -> Path:
    """Write a local HF dataset and optionally push to hub.

    Image columns are exported as `datasets.Image` and loaded from local paths.
    """

    try:
        from datasets import Dataset, DatasetDict, Features, Image as HFImage, Value
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on env extras
        raise RuntimeError(
            "Missing optional dependency 'datasets'. Install requirements.txt to enable HF export."
        ) from exc

    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)

    dataset_root = root.parent

    features = Features(
        {
            "row_id": Value("string"),
            "split": Value("string"),
            "task_type": Value("string"),
            "source_dataset": Value("string"),
            "image": HFImage(),
            "image_path": Value("string"),
            "question": Value("string"),
            "answer_text": Value("string"),
            "final_answer_json": Value("string"),
            "prompt_variant_id": Value("string"),
            "source_image_id": Value("string"),
            "source_label_format": Value("string"),
            "queried_piece": Value("string"),
        }
    )

    ds_map: dict[str, Any] = {}
    for split in SPLIT_ORDER:
        rows = rows_by_split.get(split, [])
        payload_rows: list[dict[str, Any]] = []
        for row in rows:
            item = _ensure_columns(row)
            image_rel = str(item.get("image_path", ""))
            if image_rel.startswith("/"):
                resolved_image = dataset_root.parent / image_rel.lstrip("/")
            else:
                resolved_image = dataset_root / image_rel
            item["image"] = str(resolved_image.resolve())
            if "queried_piece" not in item:
                item["queried_piece"] = ""
            payload_rows.append(item)
        ds_map[split] = Dataset.from_list(payload_rows, features=features)

    dsd = DatasetDict(ds_map)
    dsd.save_to_disk(str(root))

    if push_to_hub:
        cleaned_repo = repo_id.strip()
        if not cleaned_repo:
            raise ValueError("repo_id is required when push_to_hub=True")
        kwargs: dict[str, Any] = {"repo_id": cleaned_repo, "private": bool(private)}
        cleaned_token = token.strip()
        if cleaned_token:
            kwargs["token"] = cleaned_token
        if config_name:
            kwargs["config_name"] = config_name
        dsd.push_to_hub(**kwargs)

    return root


def build_stats(rows_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    split_stats: dict[str, Any] = {}
    overall_task_counts: Counter[str] = Counter()
    overall_source_counts: Counter[str] = Counter()
    overall_prompt_counts: Counter[str] = Counter()

    for split in SPLIT_ORDER:
        rows = rows_by_split.get(split, [])
        task_counts = Counter(str(row.get("task_type", "")) for row in rows)
        source_counts = Counter(str(row.get("source_dataset", "")) for row in rows)
        prompt_counts = Counter(str(row.get("prompt_variant_id", "")) for row in rows)

        overall_task_counts.update(task_counts)
        overall_source_counts.update(source_counts)
        overall_prompt_counts.update(prompt_counts)

        split_stats[split] = {
            "rows": len(rows),
            "task_counts": dict(sorted(task_counts.items())),
            "source_counts": dict(sorted(source_counts.items())),
            "prompt_variant_counts": dict(sorted(prompt_counts.items())),
        }

    total_rows = sum(len(rows_by_split.get(split, [])) for split in SPLIT_ORDER)
    return {
        "total_rows": total_rows,
        "split_stats": split_stats,
        "task_counts": dict(sorted(overall_task_counts.items())),
        "source_counts": dict(sorted(overall_source_counts.items())),
        "prompt_variant_counts": dict(sorted(overall_prompt_counts.items())),
    }
