from __future__ import annotations

import base64
import io
import json
import mimetypes
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from md_train_framework.config import FrameworkConfig
from tuna_sdk import (
    DetectAnnotation,
    DetectGroundTruth,
    DetectRequest,
    DetectSFTTarget,
    DetectSettings,
    PointAnnotation,
    PointGroundTruth,
    PointRequest,
    PointSFTTarget,
    PointSettings,
    QueryRequest,
    QuerySFTTarget,
    QuerySettings,
    TrainStepGroup,
)

try:
    from datasets import DatasetDict, Image as HFImage, load_dataset, load_from_disk
except ModuleNotFoundError:  # pragma: no cover
    DatasetDict = None  # type: ignore[assignment]
    HFImage = None  # type: ignore[assignment]
    load_dataset = None  # type: ignore[assignment]
    load_from_disk = None  # type: ignore[assignment]

try:
    from PIL import Image
except ModuleNotFoundError:  # pragma: no cover
    Image = None  # type: ignore[assignment]


@dataclass(frozen=True)
class DetectSample:
    sample_id: str
    split: str
    object_name: str
    image_url: str
    boxes: tuple[DetectAnnotation, ...]
    image_ref: Any = None
    meta: dict[str, Any] = field(default_factory=dict)

    def request(self, *, temperature: float, top_p: float, max_tokens: int, max_objects: Optional[int]) -> DetectRequest:
        return DetectRequest(
            object_name=self.object_name,
            image_url=_materialize_image_url(self.image_url, self.image_ref),
            settings=DetectSettings(
                temperature=float(temperature),
                top_p=float(top_p),
                max_tokens=int(max_tokens),
                max_objects=None if max_objects is None else int(max_objects),
            ),
        )

    def ground_truth(self) -> DetectGroundTruth:
        return DetectGroundTruth(boxes=list(self.boxes))

    def sft_group(self, *, temperature: float, top_p: float, max_tokens: int, max_objects: Optional[int]) -> TrainStepGroup:
        return TrainStepGroup.from_sft(
            request=self.request(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                max_objects=max_objects,
            ),
            targets=[DetectSFTTarget(boxes=list(self.boxes))],
        )


@dataclass(frozen=True)
class PointSample:
    sample_id: str
    split: str
    object_name: str
    image_url: str
    points: tuple[PointAnnotation, ...]
    boxes: tuple[DetectAnnotation, ...] = ()
    image_ref: Any = None
    meta: dict[str, Any] = field(default_factory=dict)

    def request(self, *, temperature: float, top_p: float, max_tokens: int) -> PointRequest:
        return PointRequest(
            object_name=self.object_name,
            image_url=_materialize_image_url(self.image_url, self.image_ref),
            settings=PointSettings(
                temperature=float(temperature),
                top_p=float(top_p),
                max_tokens=int(max_tokens),
            ),
        )

    def ground_truth(self) -> PointGroundTruth:
        return PointGroundTruth(points=list(self.points), boxes=list(self.boxes) or None)

    def sft_group(self, *, temperature: float, top_p: float, max_tokens: int) -> TrainStepGroup:
        return TrainStepGroup.from_sft(
            request=self.request(temperature=temperature, top_p=top_p, max_tokens=max_tokens),
            targets=[PointSFTTarget(points=list(self.points), boxes=list(self.boxes) or None)],
        )


@dataclass(frozen=True)
class QuerySample:
    sample_id: str
    split: str
    question: str
    answer: str
    image_url: Optional[str] = None
    image_ref: Any = None
    spatial_refs: tuple[tuple[float, float, float, float], ...] = ()
    reasoning: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def request(self, *, temperature: float, top_p: float, max_tokens: int, reasoning: bool) -> QueryRequest:
        return QueryRequest(
            question=self.question,
            image_url=_materialize_image_url(self.image_url, self.image_ref) or None,
            spatial_refs=[list(item) for item in self.spatial_refs] or None,
            reasoning=bool(reasoning),
            settings=QuerySettings(
                temperature=float(temperature),
                top_p=float(top_p),
                max_tokens=int(max_tokens),
            ),
        )

    def sft_group(self, *, temperature: float, top_p: float, max_tokens: int, reasoning: bool) -> TrainStepGroup:
        target_reasoning = self.reasoning if reasoning and self.reasoning else None
        return TrainStepGroup.from_sft(
            request=self.request(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                reasoning=reasoning,
            ),
            targets=[QuerySFTTarget(answer=self.answer, reasoning=target_reasoning)],
        )


NormalizedSample = DetectSample | PointSample | QuerySample


def normalize_row(config: FrameworkConfig, split: str, row: dict[str, Any], *, index: int) -> NormalizedSample:
    skill = config.skill.id
    sample_id = _sample_id(row, split=split, index=index)
    image_url, image_ref = _resolve_image_source(config, row, split=split, dataset_index=index - 1)
    if skill == "detect":
        return DetectSample(
            sample_id=sample_id,
            split=split,
            object_name=_object_name(row, fallback="object"),
            image_url=image_url or "",
            image_ref=image_ref,
            boxes=tuple(_detect_boxes(row)),
            meta=_sample_meta(row),
        )
    if skill == "point":
        return PointSample(
            sample_id=sample_id,
            split=split,
            object_name=_object_name(row, fallback="object"),
            image_url=image_url or "",
            image_ref=image_ref,
            points=tuple(_point_annotations(row)),
            boxes=tuple(_detect_boxes(row)),
            meta=_sample_meta(row),
        )
    return QuerySample(
        sample_id=sample_id,
        split=split,
        question=_query_text(row),
        answer=_query_answer(row),
        image_url=image_url,
        image_ref=image_ref,
        spatial_refs=tuple(tuple(float(value) for value in ref[:4]) for ref in _spatial_refs(row)),
        reasoning=str(row.get("reasoning") or row.get("target_reasoning") or ""),
        meta=_sample_meta(row),
    )


def _sample_id(row: dict[str, Any], *, split: str, index: int) -> str:
    for key in ("id", "row_id", "sample_id", "uid"):
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return f"{split}-{index:06d}"


def _object_name(row: dict[str, Any], *, fallback: str) -> str:
    for key in ("prompt", "object_name", "object", "class_name", "label"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return fallback


def _query_text(row: dict[str, Any]) -> str:
    for key in ("question", "query", "prompt", "instruction"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError("query sample missing question-like field")


def _query_answer(row: dict[str, Any]) -> str:
    for key in ("answer", "target_text", "target", "expected", "response", "final_answer_json"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, (dict, list)):
            return json.dumps(value, sort_keys=True)
    raise ValueError("query sample missing answer-like field")


def _spatial_refs(row: dict[str, Any]) -> list[list[float]]:
    value = row.get("spatial_refs")
    if isinstance(value, list):
        refs: list[list[float]] = []
        for item in value:
            if isinstance(item, list) and len(item) >= 4:
                refs.append([float(item[0]), float(item[1]), float(item[2]), float(item[3])])
        return refs
    return []


def _detect_boxes(row: dict[str, Any]) -> list[DetectAnnotation]:
    for key in ("boxes", "answer_boxes", "annotations", "objects"):
        value = row.get(key)
        parsed = _parse_boxes(value)
        if parsed:
            return parsed
    return []


def _point_annotations(row: dict[str, Any]) -> list[PointAnnotation]:
    for key in ("points", "answer_points", "point"):
        value = row.get(key)
        parsed = _parse_points(value)
        if parsed:
            return parsed
    boxes = _detect_boxes(row)
    if boxes:
        centers = []
        for box in boxes:
            centers.append(
                PointAnnotation(
                    x=(box.x_min + box.x_max) / 2.0,
                    y=(box.y_min + box.y_max) / 2.0,
                    width=max(0.0, box.x_max - box.x_min),
                    height=max(0.0, box.y_max - box.y_min),
                )
            )
        return centers
    return []


def _parse_boxes(value: Any) -> list[DetectAnnotation]:
    boxes: list[DetectAnnotation] = []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return boxes
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return boxes
    if not isinstance(value, list):
        return boxes
    for item in value:
        if isinstance(item, DetectAnnotation):
            boxes.append(item)
            continue
        if isinstance(item, dict):
            if {"x_min", "y_min", "x_max", "y_max"} <= set(item):
                boxes.append(
                    DetectAnnotation(
                        x_min=float(item["x_min"]),
                        y_min=float(item["y_min"]),
                        x_max=float(item["x_max"]),
                        y_max=float(item["y_max"]),
                    )
                )
            elif {"x", "y", "width", "height"} <= set(item):
                x = float(item["x"])
                y = float(item["y"])
                w = float(item["width"])
                h = float(item["height"])
                boxes.append(DetectAnnotation(x_min=x, y_min=y, x_max=x + w, y_max=y + h))
            continue
        if isinstance(item, (list, tuple)) and len(item) >= 4:
            boxes.append(
                DetectAnnotation(
                    x_min=float(item[0]),
                    y_min=float(item[1]),
                    x_max=float(item[2]),
                    y_max=float(item[3]),
                )
            )
    return boxes


def _parse_points(value: Any) -> list[PointAnnotation]:
    points: list[PointAnnotation] = []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return points
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return points
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        return points
    for item in value:
        if isinstance(item, PointAnnotation):
            points.append(item)
            continue
        if isinstance(item, dict):
            x = item.get("x")
            y = item.get("y")
            if x is None or y is None:
                continue
            points.append(
                PointAnnotation(
                    x=float(x),
                    y=float(y),
                    width=None if item.get("width") is None else float(item.get("width")),
                    height=None if item.get("height") is None else float(item.get("height")),
                )
            )
            continue
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            points.append(
                PointAnnotation(
                    x=float(item[0]),
                    y=float(item[1]),
                    width=float(item[2]) if len(item) > 2 else None,
                    height=float(item[3]) if len(item) > 3 else None,
                )
            )
    return points


def _resolve_image_source(
    config: FrameworkConfig,
    row: dict[str, Any],
    *,
    split: Optional[str] = None,
    dataset_index: Optional[int] = None,
) -> tuple[Optional[str], Any]:
    for key in ("image_url",):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip(), None
    image = row.get("image")
    if isinstance(image, dict):
        image_path = str(image.get("path") or "").strip()
        if image_path:
            resolved = _resolve_path(config, image_path)
            if resolved.exists():
                return None, resolved
        if bool(image.get("bytes_present")) and split is not None and dataset_index is not None and config.dataset.source in {"hf_hub", "hf_disk"}:
            return None, {
                "dataset_image_ref": {
                    "source": str(config.dataset.source),
                    "name": str(config.dataset.name),
                    "path": str(config.dataset.path),
                    "split": str(split),
                    "index": int(dataset_index),
                }
            }
        if image.get("bytes"):
            return None, {"bytes": bytes(image["bytes"]), "mime_type": "image/png"}
    image_path = row.get("image_path") or row.get("image_file") or row.get("image_filename")
    if isinstance(image_path, str) and image_path.strip():
        resolved = _resolve_path(config, image_path)
        if resolved.exists():
            return None, resolved
    if Image is not None and image is not None and hasattr(image, "save"):
        return None, image
    return None, None


def _resolve_image_url(config: FrameworkConfig, row: dict[str, Any]) -> Optional[str]:
    image_url, image_ref = _resolve_image_source(config, row)
    resolved = _materialize_image_url(image_url, image_ref)
    return resolved or None


def _resolve_path(config: FrameworkConfig, raw_path: str) -> Path:
    path = Path(str(raw_path)).expanduser()
    if path.is_absolute():
        return path
    if config.dataset.image_root:
        return (config.resolved_path(config.dataset.image_root) / path.name).resolve()
    if config.dataset.path:
        base = config.resolved_path(config.dataset.path)
        if base.is_dir():
            candidate = base / path
            if candidate.exists():
                return candidate
            return (base / path.name).resolve()
        return (base.parent / path).resolve()
    return config.resolved_path(raw_path)


def _file_to_data_url(path: Path) -> str:
    data = path.read_bytes()
    mime_type = mimetypes.guess_type(str(path))[0] or "image/png"
    return _bytes_to_data_url(data, mime_type=mime_type)


def _bytes_to_data_url(data: bytes, *, mime_type: str) -> str:
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _materialize_image_url(image_url: Optional[str], image_ref: Any) -> str:
    if isinstance(image_url, str) and image_url.strip():
        return image_url.strip()
    if image_ref is None:
        return ""
    if isinstance(image_ref, Path):
        return _file_to_data_url(image_ref)
    if isinstance(image_ref, dict):
        if image_ref.get("dataset_image_ref"):
            return _dataset_ref_to_data_url(dict(image_ref["dataset_image_ref"]))
        if image_ref.get("path"):
            return _file_to_data_url(Path(str(image_ref["path"])))
        if image_ref.get("bytes"):
            mime_type = str(image_ref.get("mime_type") or "image/png")
            return _bytes_to_data_url(bytes(image_ref["bytes"]), mime_type=mime_type)
        return ""
    if Image is not None and hasattr(image_ref, "save"):
        return _pil_to_data_url(image_ref)
    return ""


def materialize_image_pil(image_url: Optional[str], image_ref: Any) -> Any:
    if Image is None:
        return None
    if isinstance(image_ref, Path):
        with Image.open(image_ref) as handle:
            return handle.convert("RGB")
    if isinstance(image_ref, dict):
        if image_ref.get("dataset_image_ref"):
            return _dataset_ref_to_pil(dict(image_ref["dataset_image_ref"]))
        if image_ref.get("path"):
            with Image.open(Path(str(image_ref["path"]))) as handle:
                return handle.convert("RGB")
        if image_ref.get("bytes"):
            with Image.open(io.BytesIO(bytes(image_ref["bytes"]))) as handle:
                return handle.convert("RGB")
        return None
    if hasattr(image_ref, "copy") and hasattr(image_ref, "convert"):
        return image_ref.copy().convert("RGB")
    text = str(image_url or "").strip()
    if not text:
        return None
    if text.startswith("data:") and "," in text:
        _, encoded = text.split(",", 1)
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as handle:
            return handle.convert("RGB")
    return None


def _sample_meta(row: dict[str, Any]) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    for key, value in row.items():
        if key == "image":
            if isinstance(value, dict) and value.get("path"):
                meta[key] = {"path": str(value["path"])}
            elif isinstance(value, dict) and value.get("bytes"):
                meta[key] = {"bytes_present": True}
            elif value is not None:
                meta[key] = type(value).__name__
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            meta[key] = value
            continue
        if isinstance(value, list):
            meta[key] = value[:8]
            continue
        if isinstance(value, dict):
            meta[key] = value
            continue
        meta[key] = type(value).__name__
    return meta


@lru_cache(maxsize=32)
def _load_dataset_for_image_ref(source: str, name: str, path: str, split: str) -> Any:
    if source == "hf_hub":
        if load_dataset is None:
            raise RuntimeError("datasets is required for hf_hub image refs")
        dataset = load_dataset(name, split=split)
    elif source == "hf_disk":
        if load_from_disk is None:
            raise RuntimeError("datasets is required for hf_disk image refs")
        dataset = load_from_disk(path)
        if DatasetDict is not None and isinstance(dataset, DatasetDict):
            dataset = dataset[split]
    else:
        raise ValueError(f"unsupported dataset image ref source: {source}")
    if HFImage is not None:
        features = getattr(dataset, "features", None)
        image_feature = None if not isinstance(features, dict) else features.get("image")
        if image_feature is not None and (isinstance(image_feature, HFImage) or type(image_feature).__name__ == "Image"):
            dataset = dataset.cast_column("image", HFImage(decode=False))
    return dataset


def _dataset_ref_to_data_url(ref: dict[str, Any]) -> str:
    dataset = _load_dataset_for_image_ref(
        str(ref.get("source") or ""),
        str(ref.get("name") or ""),
        str(ref.get("path") or ""),
        str(ref.get("split") or ""),
    )
    row = dataset[int(ref.get("index") or 0)]
    image = row.get("image")
    if isinstance(image, dict):
        if image.get("bytes"):
            return _bytes_to_data_url(bytes(image["bytes"]), mime_type="image/png")
        if image.get("path"):
            candidate = Path(str(image["path"]))
            if candidate.exists():
                return _file_to_data_url(candidate)
    raise FileNotFoundError(f"dataset image ref could not resolve image bytes for {ref}")


def _dataset_ref_to_pil(ref: dict[str, Any]) -> Any:
    if Image is None:
        return None
    dataset = _load_dataset_for_image_ref(
        str(ref.get("source") or ""),
        str(ref.get("name") or ""),
        str(ref.get("path") or ""),
        str(ref.get("split") or ""),
    )
    row = dataset[int(ref.get("index") or 0)]
    image = row.get("image")
    if hasattr(image, "convert"):
        return image.convert("RGB")
    if isinstance(image, dict):
        if image.get("bytes"):
            with Image.open(io.BytesIO(bytes(image["bytes"]))) as handle:
                return handle.convert("RGB")
        if image.get("path"):
            candidate = Path(str(image["path"]))
            if candidate.exists():
                with Image.open(candidate) as handle:
                    return handle.convert("RGB")
    raise FileNotFoundError(f"dataset image ref could not resolve image bytes for {ref}")


def _pil_to_data_url(image: Any) -> str:
    if Image is None:
        raise RuntimeError("Pillow is required for in-memory image samples")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return _bytes_to_data_url(buffer.getvalue(), mime_type="image/png")
