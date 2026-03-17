"""Shared dataset loading helpers for Chess QA train/benchmark scripts."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
from pathlib import Path
from typing import Any, Optional

from PIL import Image

DEFAULT_DATASET_SOURCE = "local_jsonl"
SUPPORTED_DATASET_SOURCES = ("hf_hub", "local_jsonl")
DEFAULT_HF_DATASET_REPO_ID = "maxs-m87/chess-qa-synth-v1"
DEFAULT_HF_DATASET_REVISION = "main"
DEFAULT_DATASET_VARIANT_TAG = "piece_position_v1"
SUPPORTED_DATASET_VARIANT_TAGS = (
    "piece_position_v1",
    "mixed_tasks_v1",
    "piece_position_v2_dataset2",
    "mixed_tasks_v2_dataset2",
    "piece_position_v2_osfstorage",
    "mixed_tasks_v2_osfstorage",
)


def resolve_hf_token(raw_token: str) -> str:
    token = str(raw_token or "").strip()
    if token:
        return token
    return (
        os.environ.get("HF_TOKEN", "").strip()
        or os.environ.get("HUGGINGFACE_HUB_TOKEN", "").strip()
    )


def normalize_dataset_source(raw_source: str) -> str:
    value = str(raw_source or "").strip().lower()
    if value in SUPPORTED_DATASET_SOURCES:
        return value
    raise ValueError(
        f"dataset_source must be one of {sorted(SUPPORTED_DATASET_SOURCES)}, got: {raw_source!r}"
    )


def normalize_dataset_variant_tag(raw_variant: str, *, allow_unknown: bool = False) -> str:
    value = str(raw_variant or "").strip()
    if not value:
        value = DEFAULT_DATASET_VARIANT_TAG
    if value in SUPPORTED_DATASET_VARIANT_TAGS or allow_unknown:
        return value
    raise ValueError(
        "dataset_variant_tag must be one of "
        f"{sorted(SUPPORTED_DATASET_VARIANT_TAGS)}, got: {raw_variant!r}"
    )


def _safe_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    return cleaned or "default"


def _resolve_cache_dir(raw_cache_dir: str) -> Path:
    text = str(raw_cache_dir or "").strip()
    if text:
        return Path(text).expanduser().resolve()
    return (Path.cwd() / ".cache" / "chess_QA").resolve()


def resolve_dataset_variant_dir(*, dataset_dir: Path, dataset_variant_tag: str) -> Path:
    root = Path(dataset_dir).expanduser().resolve()
    tag = normalize_dataset_variant_tag(dataset_variant_tag)

    candidate = root / tag
    if candidate.exists():
        return candidate

    # Compatibility fallback if a user points directly at the variant folder.
    if root.name == tag and (root / "jsonl").exists():
        return root

    raise FileNotFoundError(
        f"dataset variant directory not found under dataset_dir. "
        f"dataset_dir={root} dataset_variant_tag={tag} expected={candidate}"
    )


def _persist_image_bytes(
    image_bytes: bytes,
    *,
    image_cache_root: Path,
) -> Optional[Path]:
    if not image_bytes:
        return None

    digest = hashlib.sha1(image_bytes).hexdigest()  # noqa: S324
    out_path = image_cache_root / f"{digest}.png"
    if out_path.exists():
        return out_path.resolve()

    image_cache_root.mkdir(parents=True, exist_ok=True)
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img.convert("RGB").save(out_path, format="PNG")
        return out_path.resolve()
    except OSError:
        fallback = image_cache_root / f"{digest}.img"
        if not fallback.exists():
            fallback.write_bytes(image_bytes)
        return fallback.resolve()


def _resolve_path_candidate(
    raw_path: str,
    *,
    dataset_dir_root: Optional[Path],
    dataset_variant_dir: Optional[Path],
) -> Optional[Path]:
    text = str(raw_path or "").strip()
    if not text:
        return None

    # Dataset rows often store absolute-style relpaths like "/imges/foo.jpg".
    if text.startswith("/imges/") and dataset_dir_root is not None:
        mapped = (dataset_dir_root / text.lstrip("/")).resolve()
        if mapped.is_file():
            return mapped

    path = Path(text).expanduser()
    if path.is_file():
        return path.resolve()

    if not path.is_absolute():
        if dataset_variant_dir is not None:
            joined = (dataset_variant_dir / path).resolve()
            if joined.is_file():
                return joined
        if dataset_dir_root is not None:
            joined = (dataset_dir_root / path).resolve()
            if joined.is_file():
                return joined

    basename = path.name
    if basename:
        if dataset_dir_root is not None:
            in_imges = (dataset_dir_root / "imges" / basename).resolve()
            if in_imges.is_file():
                return in_imges
            in_images = (dataset_dir_root / "images" / basename).resolve()
            if in_images.is_file():
                return in_images
        if dataset_variant_dir is not None:
            in_variant_images = (dataset_variant_dir / "images" / basename).resolve()
            if in_variant_images.is_file():
                return in_variant_images

    return None


def _resolve_hf_row_image_path(
    row: dict[str, Any],
    *,
    dataset_dir_root: Optional[Path],
    dataset_variant_dir: Optional[Path],
    image_cache_root: Path,
) -> Optional[Path]:
    for key in ("image_path", "image"):
        value = row.get(key)
        if isinstance(value, str):
            resolved = _resolve_path_candidate(
                value,
                dataset_dir_root=dataset_dir_root,
                dataset_variant_dir=dataset_variant_dir,
            )
            if resolved is not None:
                return resolved

    image_payload = row.get("image")
    if isinstance(image_payload, dict):
        payload_path = image_payload.get("path")
        if isinstance(payload_path, str):
            resolved = _resolve_path_candidate(
                payload_path,
                dataset_dir_root=dataset_dir_root,
                dataset_variant_dir=dataset_variant_dir,
            )
            if resolved is not None:
                return resolved

        payload_bytes = image_payload.get("bytes")
        if isinstance(payload_bytes, (bytes, bytearray)):
            return _persist_image_bytes(
                bytes(payload_bytes),
                image_cache_root=image_cache_root,
            )

    if isinstance(image_payload, (bytes, bytearray)):
        return _persist_image_bytes(
            bytes(image_payload),
            image_cache_root=image_cache_root,
        )

    return None


def _load_local_jsonl_rows(
    *,
    dataset_dir_root: Path,
    dataset_variant_tag: str,
    split_name: str,
) -> list[dict[str, Any]]:
    variant_dir = resolve_dataset_variant_dir(
        dataset_dir=dataset_dir_root,
        dataset_variant_tag=dataset_variant_tag,
    )
    path = variant_dir / "jsonl" / f"{split_name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"split JSONL not found: {path}")

    rows: list[dict[str, Any]] = []
    skipped_invalid_json = 0
    skipped_non_object = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                skipped_invalid_json += 1
                print(
                    f"split={split_name} path={path} line={line_number} "
                    f"invalid_json={exc}; skipping"
                )
                continue
            if not isinstance(payload, dict):
                skipped_non_object += 1
                continue
            rows.append(payload)

    print(
        f"loaded split={split_name} rows={len(rows)} "
        f"skipped_invalid_json={skipped_invalid_json} "
        f"skipped_non_object={skipped_non_object} "
        f"from {path}"
    )
    return rows


def _load_hf_rows(
    *,
    split_name: str,
    dataset_variant_tag: str,
    dataset_dir_root: Optional[Path],
    hf_dataset_repo_id: str,
    hf_dataset_revision: str,
    hf_token: str,
    hf_cache_dir: str,
) -> list[dict[str, Any]]:
    try:
        from datasets import Image as HFImage  # type: ignore
        from datasets import load_dataset  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "datasets is required for dataset_source='hf_hub'. Install with: pip install datasets"
        ) from exc

    cache_dir = _resolve_cache_dir(hf_cache_dir)
    repo_id = str(hf_dataset_repo_id or "").strip()
    if not repo_id:
        raise ValueError("hf_dataset_repo_id is required when dataset_source='hf_hub'")

    variant = normalize_dataset_variant_tag(dataset_variant_tag, allow_unknown=True)
    revision = str(hf_dataset_revision or DEFAULT_HF_DATASET_REVISION).strip() or DEFAULT_HF_DATASET_REVISION
    resolved_token = resolve_hf_token(hf_token)

    base_kwargs: dict[str, Any] = {
        "split": split_name,
        "revision": revision,
        "cache_dir": str(cache_dir),
    }
    if resolved_token:
        base_kwargs["token"] = resolved_token

    attempts: list[dict[str, Any]] = []
    kwargs_with_name = dict(base_kwargs)
    if variant:
        kwargs_with_name["name"] = variant
    attempts.append(kwargs_with_name)
    if "name" in kwargs_with_name:
        attempts.append(dict(base_kwargs))

    ds = None
    last_exc: Optional[Exception] = None
    for load_kwargs in attempts:
        try:
            ds = load_dataset(repo_id, **load_kwargs)
            break
        except TypeError:
            # Backward compatibility for datasets versions expecting use_auth_token.
            retry_kwargs = dict(load_kwargs)
            token_value = retry_kwargs.pop("token", None)
            if token_value:
                retry_kwargs["use_auth_token"] = token_value
            try:
                ds = load_dataset(repo_id, **retry_kwargs)
                break
            except Exception as exc:  # pragma: no cover - network/runtime dependent
                last_exc = exc
                continue
        except Exception as exc:  # pragma: no cover - network/runtime dependent
            last_exc = exc
            continue

    if ds is None:
        raise RuntimeError(
            f"Failed to load HF dataset repo={repo_id} split={split_name} variant={variant}: {last_exc}"
        )

    if "image" in getattr(ds, "column_names", []):
        ds = ds.cast_column("image", HFImage(decode=False))

    dataset_variant_dir: Optional[Path] = None
    if dataset_dir_root is not None:
        try:
            dataset_variant_dir = resolve_dataset_variant_dir(
                dataset_dir=dataset_dir_root,
                dataset_variant_tag=dataset_variant_tag,
            )
        except FileNotFoundError:
            dataset_variant_dir = None

    image_cache_root = (
        cache_dir
        / "hf_images"
        / _safe_component(repo_id)
        / _safe_component(variant)
        / _safe_component(revision)
        / _safe_component(split_name)
    )

    rows: list[dict[str, Any]] = []
    skipped = 0
    for raw_row in ds:
        if not isinstance(raw_row, dict):
            skipped += 1
            continue

        row = dict(raw_row)
        image_path = _resolve_hf_row_image_path(
            row,
            dataset_dir_root=dataset_dir_root,
            dataset_variant_dir=dataset_variant_dir,
            image_cache_root=image_cache_root,
        )
        if image_path is not None:
            row["image_path"] = str(image_path)
            row["image"] = str(image_path)
        rows.append(row)

    print(
        "loaded split="
        f"{split_name} rows={len(rows)} skipped={skipped} "
        f"from hf_hub repo={repo_id} variant={variant} revision={revision}"
    )
    return rows


def load_split_rows(
    *,
    dataset_source: str,
    dataset_variant_tag: str,
    split_name: str,
    dataset_dir: Optional[Path],
    hf_dataset_repo_id: str,
    hf_dataset_revision: str,
    hf_token: str,
    hf_cache_dir: str,
) -> list[dict[str, Any]]:
    source = normalize_dataset_source(dataset_source)
    variant = normalize_dataset_variant_tag(dataset_variant_tag, allow_unknown=(source == "hf_hub"))

    if source == "local_jsonl":
        if dataset_dir is None:
            raise ValueError("dataset_dir is required when dataset_source='local_jsonl'")
        return _load_local_jsonl_rows(
            dataset_dir_root=dataset_dir,
            dataset_variant_tag=variant,
            split_name=split_name,
        )

    return _load_hf_rows(
        split_name=split_name,
        dataset_variant_tag=variant,
        dataset_dir_root=dataset_dir,
        hf_dataset_repo_id=hf_dataset_repo_id,
        hf_dataset_revision=hf_dataset_revision,
        hf_token=hf_token,
        hf_cache_dir=hf_cache_dir,
    )
