from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

from md_train_framework.config import DatasetSpec, FrameworkConfig
from md_train_framework.utils import stable_json_hash

try:
    from datasets import DatasetDict, Image as HFImage, load_dataset, load_from_disk
except ModuleNotFoundError:  # pragma: no cover
    DatasetDict = None  # type: ignore[assignment]
    HFImage = None  # type: ignore[assignment]
    load_dataset = None  # type: ignore[assignment]
    load_from_disk = None  # type: ignore[assignment]


IMAGE_KEYS = ("image_path", "image", "image_file", "image_filename")
LABEL_KEYS = ("label", "class_name", "class_names", "task_type", "category")


@dataclass(frozen=True)
class SplitInspection:
    name: str
    count: int
    fields: tuple[str, ...]
    missing_images: int
    sample_preview: tuple[dict[str, Any], ...]
    label_counts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetInspection:
    source: str
    dataset_path: str
    dataset_name: str
    skill: str
    fingerprint: str
    splits: tuple[SplitInspection, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "dataset_path": self.dataset_path,
            "dataset_name": self.dataset_name,
            "skill": self.skill,
            "fingerprint": self.fingerprint,
            "splits": [
                {
                    "name": split.name,
                    "count": split.count,
                    "fields": list(split.fields),
                    "missing_images": split.missing_images,
                    "sample_preview": list(split.sample_preview),
                    "label_counts": dict(split.label_counts),
                }
                for split in self.splits
            ],
        }


class DatasetAdapter:
    def __init__(self, config: FrameworkConfig) -> None:
        self.config = config
        self.spec = config.dataset

    def iter_split(self, split: str) -> Iterator[dict[str, Any]]:
        raise NotImplementedError

    def inspect(self, *, max_preview: int = 3) -> DatasetInspection:
        split_names = [self.spec.train_split, self.spec.val_split, self.spec.test_split]
        split_summaries: list[SplitInspection] = []
        for split_name in split_names:
            split_summaries.append(_inspect_rows(self.config, split_name, self.iter_split(split_name), max_preview=max_preview))
        return DatasetInspection(
            source=self.spec.source,
            dataset_path=self.spec.path,
            dataset_name=self.spec.name,
            skill=self.config.skill.id,
            fingerprint=stable_json_hash(
                {
                    "source": self.spec.source,
                    "path": self.spec.path,
                    "name": self.spec.name,
                    "splits": [split.__dict__ for split in split_summaries],
                }
            ),
            splits=tuple(split_summaries),
        )


class LocalJsonlAdapter(DatasetAdapter):
    def iter_split(self, split: str) -> Iterator[dict[str, Any]]:
        path = self._resolve_split_path(split)
        if not path.exists():
            return iter(())
        def _iterator() -> Iterator[dict[str, Any]]:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    text = line.strip()
                    if not text:
                        continue
                    payload = json.loads(text)
                    if isinstance(payload, dict):
                        yield payload
        return _iterator()

    def _resolve_split_path(self, split: str) -> Path:
        if split in self.spec.split_files:
            return self.config.resolved_path(self.spec.split_files[split])
        base = self.config.resolved_path(self.spec.path)
        if base.is_file():
            return base
        candidates = [f"{split}.jsonl"]
        if split == self.spec.val_split:
            candidates.extend(["validation.jsonl", "val.jsonl"])
        if split == self.spec.test_split:
            candidates.extend(["test.jsonl", "post_val.jsonl"])
        for name in candidates:
            candidate = base / name
            if candidate.exists():
                return candidate
        return base / f"{split}.jsonl"


class HuggingFaceDiskAdapter(DatasetAdapter):
    def iter_split(self, split: str) -> Iterator[dict[str, Any]]:
        if load_from_disk is None:
            raise RuntimeError("datasets is required for hf_disk datasets")
        dataset = load_from_disk(str(self.config.resolved_path(self.spec.path)))
        if isinstance(dataset, DatasetDict):
            if split not in dataset:
                return iter(())
            split_dataset = _disable_image_decode(dataset[split])
        else:
            split_dataset = _disable_image_decode(dataset)
        return (_lightweight_row(dict(row)) for row in split_dataset)


class HuggingFaceHubAdapter(DatasetAdapter):
    def iter_split(self, split: str) -> Iterator[dict[str, Any]]:
        if load_dataset is None:
            raise RuntimeError("datasets is required for hf_hub datasets")
        dataset = _disable_image_decode(load_dataset(self.spec.name, split=split))
        return (_lightweight_row(dict(row)) for row in dataset)


class RepoDatasetDirAdapter(LocalJsonlAdapter):
    pass


def create_dataset_adapter(config: FrameworkConfig) -> DatasetAdapter:
    source = config.dataset.source
    if source == "local_jsonl":
        return LocalJsonlAdapter(config)
    if source == "hf_disk":
        return HuggingFaceDiskAdapter(config)
    if source == "hf_hub":
        return HuggingFaceHubAdapter(config)
    if source == "repo_dir":
        return RepoDatasetDirAdapter(config)
    raise ValueError(f"unsupported dataset source: {source}")


def inspect_dataset(config: FrameworkConfig, *, max_preview: int = 3) -> DatasetInspection:
    return create_dataset_adapter(config).inspect(max_preview=max_preview)


def _disable_image_decode(dataset: Any) -> Any:
    if HFImage is None:
        return dataset
    features = getattr(dataset, "features", None)
    if not isinstance(features, dict):
        return dataset
    for column, feature in features.items():
        if isinstance(feature, HFImage) or type(feature).__name__ == "Image":
            try:
                dataset = dataset.cast_column(column, HFImage(decode=False))
            except Exception:
                continue
    return dataset


def _lightweight_row(row: dict[str, Any]) -> dict[str, Any]:
    payload = dict(row)
    image = payload.get("image")
    if isinstance(image, dict):
        payload["image"] = {
            "path": str(image.get("path") or ""),
            "bytes_present": bool(image.get("bytes")),
        }
    return payload


def _inspect_rows(config: FrameworkConfig, split_name: str, rows: Iterable[dict[str, Any]], *, max_preview: int) -> SplitInspection:
    fields: set[str] = set()
    preview: list[dict[str, Any]] = []
    missing_images = 0
    label_counts: dict[str, int] = {}
    count = 0
    for row in rows:
        count += 1
        fields.update(row.keys())
        if len(preview) < max_preview:
            preview.append(_preview_row(row))
        image_path = _resolve_image_path(config, row)
        if image_path is not None and not image_path.exists():
            missing_images += 1
        label = _extract_label(row)
        if label:
            label_counts[label] = label_counts.get(label, 0) + 1
    return SplitInspection(
        name=split_name,
        count=count,
        fields=tuple(sorted(fields)),
        missing_images=missing_images,
        sample_preview=tuple(preview),
        label_counts=dict(sorted(label_counts.items())),
    )


def _resolve_image_path(config: FrameworkConfig, row: dict[str, Any]) -> Optional[Path]:
    for key in IMAGE_KEYS:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            raw = Path(value).expanduser()
            if raw.is_absolute():
                return raw
            if config.dataset.image_root:
                return config.resolved_path(config.dataset.image_root) / raw
            if config.dataset.path:
                return config.resolved_path(config.dataset.path).parent / raw
            return config.config_path.parent / raw if config.config_path is not None else raw
    return None


def _extract_label(row: dict[str, Any]) -> str:
    for key in LABEL_KEYS:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list) and value:
            return str(value[0])
    return ""


def _preview_row(row: dict[str, Any]) -> dict[str, Any]:
    preview: dict[str, Any] = {}
    for key, value in list(row.items())[:8]:
        if isinstance(value, (str, int, float, bool)) or value is None:
            preview[key] = value
        elif isinstance(value, list):
            preview[key] = value[:3]
        else:
            preview[key] = type(value).__name__
    return preview
