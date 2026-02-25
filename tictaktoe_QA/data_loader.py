"""Shared dataset loading helpers for TicTacToe QA train/benchmark scripts."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
from pathlib import Path
from typing import Any, Optional

from PIL import Image

DEFAULT_DATASET_SOURCE = "hf_hub"
SUPPORTED_DATASET_SOURCES = ("hf_hub", "local_jsonl")
DEFAULT_HF_DATASET_REPO_ID = "maxs-m87/tictactoe-qa-v1"
DEFAULT_HF_DATASET_REVISION = "main"


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


def _safe_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    return cleaned or "default"


def _resolve_cache_dir(raw_cache_dir: str) -> Path:
    text = str(raw_cache_dir or "").strip()
    if text:
        return Path(text).expanduser().resolve()
    return (Path.cwd() / ".cache" / "tictaktoe_QA").resolve()


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


def _resolve_path_candidate(raw_path: str, *, dataset_dir: Optional[Path]) -> Optional[Path]:
    text = str(raw_path or "").strip()
    if not text:
        return None

    path = Path(text).expanduser()
    if path.is_file():
        return path.resolve()

    if not path.is_absolute() and dataset_dir is not None:
        joined = (dataset_dir / path).resolve()
        if joined.is_file():
            return joined

    return None


def _resolve_hf_row_image_path(
    row: dict[str, Any],
    *,
    dataset_dir: Optional[Path],
    image_cache_root: Path,
) -> Optional[Path]:
    for key in ("image_path", "image"):
        value = row.get(key)
        if isinstance(value, str):
            resolved = _resolve_path_candidate(value, dataset_dir=dataset_dir)
            if resolved is not None:
                return resolved

    image_payload = row.get("image")
    if isinstance(image_payload, dict):
        payload_path = image_payload.get("path")
        if isinstance(payload_path, str):
            resolved = _resolve_path_candidate(payload_path, dataset_dir=dataset_dir)
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


def _load_local_jsonl_rows(*, dataset_dir: Path, split_name: str) -> list[dict[str, Any]]:
    path = dataset_dir / "jsonl" / f"{split_name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"split JSONL not found: {path}")

    rows: list[dict[str, Any]] = []
    skipped = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(payload, dict):
                skipped += 1
                continue
            rows.append(payload)

    print(f"loaded split={split_name} rows={len(rows)} skipped={skipped} from {path}")
    return rows


def _load_hf_rows(
    *,
    split_name: str,
    dataset_dir: Optional[Path],
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

    revision = str(hf_dataset_revision or DEFAULT_HF_DATASET_REVISION).strip() or DEFAULT_HF_DATASET_REVISION
    resolved_token = resolve_hf_token(hf_token)

    load_kwargs: dict[str, Any] = {
        "split": split_name,
        "revision": revision,
        "cache_dir": str(cache_dir),
    }
    if resolved_token:
        load_kwargs["token"] = resolved_token

    try:
        ds = load_dataset(repo_id, **load_kwargs)
    except TypeError:
        token_value = load_kwargs.pop("token", None)
        if token_value:
            load_kwargs["use_auth_token"] = token_value
        ds = load_dataset(repo_id, **load_kwargs)

    if "image" in getattr(ds, "column_names", []):
        ds = ds.cast_column("image", HFImage(decode=False))

    image_cache_root = (
        cache_dir
        / "hf_images"
        / _safe_component(repo_id)
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
            dataset_dir=dataset_dir,
            image_cache_root=image_cache_root,
        )
        if image_path is not None:
            row["image_path"] = str(image_path)
            row["image"] = str(image_path)
        rows.append(row)

    print(
        "loaded split="
        f"{split_name} rows={len(rows)} skipped={skipped} "
        f"from hf_hub repo={repo_id} revision={revision}"
    )
    return rows


def load_split_rows(
    *,
    dataset_source: str,
    split_name: str,
    dataset_dir: Optional[Path],
    hf_dataset_repo_id: str,
    hf_dataset_revision: str,
    hf_token: str,
    hf_cache_dir: str,
) -> list[dict[str, Any]]:
    source = normalize_dataset_source(dataset_source)
    if source == "local_jsonl":
        if dataset_dir is None:
            raise ValueError("dataset_dir is required when dataset_source='local_jsonl'")
        return _load_local_jsonl_rows(dataset_dir=dataset_dir, split_name=split_name)

    return _load_hf_rows(
        split_name=split_name,
        dataset_dir=dataset_dir,
        hf_dataset_repo_id=hf_dataset_repo_id,
        hf_dataset_revision=hf_dataset_revision,
        hf_token=hf_token,
        hf_cache_dir=hf_cache_dir,
    )
