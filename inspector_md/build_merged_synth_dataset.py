#!/usr/bin/env python3
"""Merge and synthetically enrich raw defect datasets for Inspector MD."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Optional

from PIL import Image

from inspector_md import common, ontology, prompt_library, rule_stubs, task_schema

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = common.repo_relative("configs", "build_merged_synth_dataset_default.json")
DEFAULT_OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
PROMPT_VERSION = "inspector_md_merged_synth_v1"
DEFAULT_OPENROUTER_ENV_VAR = "OPENROUTER_API_KEY"

MBDD2025_LABEL_MAP = {
    "crack": "crack_defect",
    "corrosion": "corrosion_rust",
    "leakage": "water_leakage",
    "abscission": "material_abscission",
    "bulge": "surface_bulge",
}
CUBIT_DET_LABEL_MAP = {
    "0": ("crack", "crack_defect"),
    "1": ("spalling", "surface_spalling"),
    "2": ("moisture", "moisture_intrusion"),
}
CODEBRIM_LABEL_MAP = {
    "Crack": "crack_defect",
    "Spallation": "surface_spalling",
    "Efflorescence": "efflorescence_deposit",
    "ExposedBars": "exposed_rebar",
    "CorrosionStain": "corrosion_rust",
}
HARD_EXAMPLE_ISSUES = {
    "moisture_intrusion",
    "water_leakage",
    "efflorescence_deposit",
    "exposed_rebar",
    "material_abscission",
    "surface_bulge",
}
CROP_ANNOTATION_TYPES = {"full_image_crop", "negative_crop"}
QUERY_V2_CROP_ISSUE_CODE_EXCEPTIONS = {"exposed_rebar"}


class OpenRouterSynthesisError(Exception):
    pass


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    raw_argv = list(argv) if argv is not None else list(os.sys.argv[1:])
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    pre_args, _ = pre_parser.parse_known_args(raw_argv)
    config_path = common.resolve_config_path(pre_args.config, script_dir=SCRIPT_DIR)
    config = common.load_json_config(config_path, default_path=DEFAULT_CONFIG_PATH)

    parser = argparse.ArgumentParser(description="Merge and synthesize a builder-ready Inspector MD dataset.")
    parser.add_argument("--config", default=str(config_path))
    parser.add_argument("--stage", choices=("normalize", "synthesize", "all"), default="normalize")
    parser.add_argument("--raw-dataset-root", default=str(common.repo_relative("raw_datasets")))
    parser.add_argument("--output-dir", default=str(common.repo_relative("dataset", "merged_synth_v1")))
    parser.add_argument("--env-file", default=str(common.repo_relative(".env.staging")))
    parser.add_argument("--api-key", default="")
    parser.add_argument("--api-key-env-var", default=DEFAULT_OPENROUTER_ENV_VAR)
    parser.add_argument("--api-base", default=DEFAULT_OPENROUTER_API_BASE)
    parser.add_argument("--teacher-model-id", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mbdd-train-fraction", type=float, default=0.7)
    parser.add_argument("--mbdd-validation-fraction", type=float, default=0.15)
    parser.add_argument("--max-positive-records", type=int, default=0)
    parser.add_argument("--max-hard-negatives", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--retry-429-max-retries", type=int, default=3)
    parser.add_argument("--retry-429-backoff-s", type=float, default=1.0)
    parser.add_argument("--retry-429-max-backoff-s", type=float, default=12.0)

    option_to_dest: dict[str, str] = {}
    for action in parser._actions:
        if not action.option_strings:
            continue
        for opt in action.option_strings:
            option_to_dest[opt] = action.dest
    overridden = {option_to_dest[arg] for arg in raw_argv if arg in option_to_dest}
    config_cli_args = common.config_to_cli_args(parser, config, config_path=config_path, overridden_dests=overridden)
    args = parser.parse_args(config_cli_args + raw_argv)
    args.config = str(common.resolve_config_path(args.config, script_dir=SCRIPT_DIR))
    args.raw_dataset_root = common.resolve_path(args.raw_dataset_root, module_root=SCRIPT_DIR)
    args.output_dir = common.resolve_path(args.output_dir, module_root=SCRIPT_DIR)
    args.env_file = str(common.resolve_path(args.env_file, module_root=SCRIPT_DIR))
    if float(args.mbdd_train_fraction) <= 0.0 or float(args.mbdd_validation_fraction) < 0.0:
        raise ValueError("MBDD split fractions must be positive.")
    if float(args.mbdd_train_fraction) + float(args.mbdd_validation_fraction) >= 1.0:
        raise ValueError("MBDD train + validation fractions must be < 1.0")
    return args


def _resolve_openrouter_api_key(explicit_api_key: str, api_key_env_var: str) -> str:
    explicit = str(explicit_api_key or "").strip()
    if explicit:
        return explicit
    preferred = str(api_key_env_var or "").strip()
    if preferred:
        preferred_value = str(os.environ.get(preferred) or "").strip()
        if preferred_value:
            return preferred_value
    default_value = str(os.environ.get(DEFAULT_OPENROUTER_ENV_VAR) or "").strip()
    if default_value:
        return default_value
    raise ValueError(f"{DEFAULT_OPENROUTER_ENV_VAR} is required for OpenRouter synthesis.")


def _build_openrouter_headers(api_key: str) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {str(api_key).strip()}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    referer = str(os.environ.get("OPENROUTER_HTTP_REFERER") or "").strip()
    if referer:
        headers["HTTP-Referer"] = referer
    title = str(os.environ.get("OPENROUTER_APP_NAME") or "").strip()
    if title:
        headers["X-Title"] = title
    return headers


def _extract_openrouter_answer_text(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    choice = choices[0]
    if not isinstance(choice, dict):
        return ""
    message = choice.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(str(item["text"]))
        return "\n".join(parts)
    return ""


def _http_error_details(exc: urllib.error.HTTPError) -> tuple[str, str]:
    request_id = str(exc.headers.get("x-request-id") or "").strip()
    try:
        body = exc.read().decode("utf-8", errors="replace")
    except Exception:  # pragma: no cover
        body = ""
    return request_id, body


def _call_openrouter_chat_api(
    *,
    api_base: str,
    api_key: str,
    model_id: str,
    messages: list[dict[str, Any]],
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout: float,
    retry_429_max_retries: int,
    retry_429_backoff_s: float,
    retry_429_max_backoff_s: float,
) -> tuple[str, dict[str, Any], float]:
    payload = {
        "model": str(model_id).strip(),
        "messages": list(messages),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": int(max_tokens),
        "response_format": {"type": "json_object"},
    }
    endpoint = str(api_base).rstrip("/") + "/chat/completions"
    attempt = 0
    retries = max(0, int(retry_429_max_retries))
    while True:
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=_build_openrouter_headers(api_key),
            method="POST",
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=float(timeout)) as response:
                body = response.read().decode("utf-8", errors="replace")
            latency_ms = (time.monotonic() - started) * 1000.0
            data = json.loads(body) if body else {}
            if not isinstance(data, dict):
                data = {}
            return _extract_openrouter_answer_text(data), data, latency_ms
        except urllib.error.HTTPError as exc:
            latency_ms = (time.monotonic() - started) * 1000.0
            request_id, body = _http_error_details(exc)
            if exc.code == 429 and attempt < retries:
                backoff = min(float(retry_429_max_backoff_s), float(retry_429_backoff_s) * (2.0**attempt))
                if backoff > 0.0:
                    time.sleep(backoff)
                attempt += 1
                continue
            raise OpenRouterSynthesisError(
                f"HTTP {exc.code} request_id={request_id or '-'} latency_ms={latency_ms:.1f} body={common.truncate(body, limit=400)}"
            ) from exc
        except urllib.error.URLError as exc:
            raise OpenRouterSynthesisError(f"Network error: {exc}") from exc


def _stable_bucket(key: str) -> float:
    digest = hashlib.sha1(str(key).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / float(0xFFFFFFFF)


def _canonical_split(raw_split: str) -> str:
    split = str(raw_split or "").strip().lower()
    if split.startswith("train"):
        return "train"
    if split.startswith("val"):
        return "validation"
    if split.startswith("test"):
        return "test"
    raise ValueError(f"Unsupported split: {raw_split!r}")


def _mbdd_split(image_id: str, *, train_fraction: float, validation_fraction: float) -> str:
    bucket = _stable_bucket(image_id)
    if bucket < float(train_fraction):
        return "train"
    if bucket < float(train_fraction) + float(validation_fraction):
        return "validation"
    return "test"


def _normalize_box(x_min: float, y_min: float, x_max: float, y_max: float) -> dict[str, float]:
    return task_schema.Box.from_payload([x_min, y_min, x_max, y_max]).to_payload()


def _safe_normalize_box(
    *,
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    excluded_source: dict[str, Any],
    excluded_key: str,
) -> Optional[dict[str, float]]:
    try:
        return _normalize_box(x_min, y_min, x_max, y_max)
    except ValueError:
        excluded_source[excluded_key] = int(excluded_source.get(excluded_key, 0)) + 1
        return None


def _asset_name_for_record(row_id: str, source_path: Path, *, prefix: str = "") -> str:
    clean_row_id = str(row_id).replace(":", "_").replace("/", "_")
    name = source_path.name
    return f"{prefix}{clean_row_id}__{name}" if prefix else f"{clean_row_id}__{name}"


def _materialize_asset(source_path: Path, destination_path: Path) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path.exists():
        return
    try:
        os.link(source_path, destination_path)
    except OSError:
        shutil.copy2(source_path, destination_path)


def _first_detect_label(issue_code: str) -> str:
    return ontology.detect_labels_for_issue(issue_code)[0]


def _issue_title(issue_code: str) -> str:
    return ontology.get_issue(issue_code).title


def _source_annotation(
    *,
    annotation_index: int,
    issue_code: str,
    source_label: str,
    box: dict[str, float],
    source_dataset: str,
    original_split: str,
    annotation_type: str,
    raw_annotation_path: str,
    raw_class_id: str = "",
) -> dict[str, Any]:
    return {
        "annotation_index": int(annotation_index),
        "source_label": str(source_label),
        "issue_code": str(issue_code),
        "box": dict(box),
        "source_detect_labels": [_first_detect_label(issue_code)],
        "source_metadata": {
            "source_dataset": source_dataset,
            "original_split": original_split,
            "annotation_type": annotation_type,
            "raw_annotation_path": raw_annotation_path,
            "raw_class_id": str(raw_class_id or ""),
        },
    }


def _base_record(
    *,
    row_id: str,
    split: str,
    source_dataset: str,
    original_split: str,
    image_path: str,
    raw_image_path: str,
    annotation_type: str,
    annotations: list[dict[str, Any]],
    hard_negative_candidate: bool = False,
    source_metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "split": split,
        "source_dataset": source_dataset,
        "original_split": original_split,
        "image_path": image_path,
        "raw_image_path": raw_image_path,
        "annotation_type": annotation_type,
        "is_negative": not bool(annotations),
        "hard_negative_candidate": bool(hard_negative_candidate),
        "annotations": list(annotations),
        "source_metadata": dict(source_metadata or {}),
    }


def _update_mapping_summary(
    summary: dict[str, Any],
    *,
    source_dataset: str,
    source_label: str,
    issue_code: str,
) -> None:
    source_summary = summary.setdefault("sources", {}).setdefault(source_dataset, {})
    raw_label_counts = source_summary.setdefault("raw_label_counts", {})
    raw_label_counts[source_label] = int(raw_label_counts.get(source_label, 0)) + 1
    issue_counts = source_summary.setdefault("issue_code_counts", {})
    issue_counts[issue_code] = int(issue_counts.get(issue_code, 0)) + 1
    mapping = source_summary.setdefault("raw_to_issue_code", {})
    mapping[source_label] = issue_code


def _record_provenance(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "row_id": record["row_id"],
        "source_dataset": record["source_dataset"],
        "split": record["split"],
        "original_split": record["original_split"],
        "image_path": record["image_path"],
        "raw_image_path": record["raw_image_path"],
        "annotation_type": record["annotation_type"],
        "annotation_count": len(record["annotations"]),
        "is_negative": bool(record["is_negative"]),
        "source_metadata": dict(record.get("source_metadata") or {}),
    }


def _load_xml(path: Path) -> ET.Element:
    return ET.parse(path).getroot()


def _normalize_mbdd_dataset(args: argparse.Namespace, *, mapping_summary: dict[str, Any], excluded: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    root = Path(args.raw_dataset_root) / "MBDD2025"
    annotations_dir = root / "Annotations"
    images_dir = root / "JPEGImages"
    assets_root = Path(args.output_dir) / "assets" / "MBDD2025"
    excluded_source = excluded.setdefault("MBDD2025", {})
    for xml_path in sorted(annotations_dir.glob("*.xml")):
        image_id = xml_path.stem
        image_path = images_dir / f"{image_id}.jpg"
        if not image_path.is_file():
            excluded_source["missing_image"] = int(excluded_source.get("missing_image", 0)) + 1
            continue
        root_xml = _load_xml(xml_path)
        width = float(root_xml.findtext("size/width") or 0.0)
        height = float(root_xml.findtext("size/height") or 0.0)
        if width <= 0.0 or height <= 0.0:
            excluded_source["invalid_size"] = int(excluded_source.get("invalid_size", 0)) + 1
            continue
        split = _mbdd_split(
            image_id,
            train_fraction=float(args.mbdd_train_fraction),
            validation_fraction=float(args.mbdd_validation_fraction),
        )
        row_id = f"mbdd2025:{image_id}"
        local_image_path = assets_root / split / _asset_name_for_record(row_id, image_path)
        _materialize_asset(image_path, local_image_path)
        annotations: list[dict[str, Any]] = []
        for index, obj in enumerate(root_xml.findall("object")):
            raw_label = str(obj.findtext("name") or "").strip().lower()
            issue_code = MBDD2025_LABEL_MAP.get(raw_label)
            if not issue_code:
                excluded_source[f"unknown_label:{raw_label or '<empty>'}"] = int(
                    excluded_source.get(f"unknown_label:{raw_label or '<empty>'}", 0)
                ) + 1
                continue
            box_node = obj.find("bndbox")
            if box_node is None:
                excluded_source["missing_box"] = int(excluded_source.get("missing_box", 0)) + 1
                continue
            box = _safe_normalize_box(
                x_min=float(box_node.findtext("xmin") or 0.0) / width,
                y_min=float(box_node.findtext("ymin") or 0.0) / height,
                x_max=float(box_node.findtext("xmax") or 0.0) / width,
                y_max=float(box_node.findtext("ymax") or 0.0) / height,
                excluded_source=excluded_source,
                excluded_key="invalid_box",
            )
            if box is None:
                continue
            annotations.append(
                _source_annotation(
                    annotation_index=index,
                    issue_code=issue_code,
                    source_label=raw_label,
                    box=box,
                    source_dataset="MBDD2025",
                    original_split="generated",
                    annotation_type="bbox",
                    raw_annotation_path=str(xml_path),
                )
            )
            _update_mapping_summary(mapping_summary, source_dataset="MBDD2025", source_label=raw_label, issue_code=issue_code)
        if not annotations:
            excluded_source["empty_annotations"] = int(excluded_source.get("empty_annotations", 0)) + 1
            continue
        records.append(
            _base_record(
                row_id=row_id,
                split=split,
                source_dataset="MBDD2025",
                original_split="generated",
                image_path=str(local_image_path),
                raw_image_path=str(image_path),
                annotation_type="bbox",
                annotations=annotations,
                source_metadata={"raw_annotation_path": str(xml_path)},
            )
        )
    return records


def _resolve_image_for_stem(image_dir: Path, stem: str) -> Optional[Path]:
    matches = sorted(path for path in image_dir.glob(f"{stem}.*") if path.is_file())
    return matches[0] if matches else None


def _normalize_cubit_dataset(args: argparse.Namespace, *, mapping_summary: dict[str, Any], excluded: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    root = Path(args.raw_dataset_root) / "CUBIT-Det"
    labels_root = root / "labels"
    images_root = root / "images"
    assets_root = Path(args.output_dir) / "assets" / "CUBIT-Det"
    excluded_source = excluded.setdefault("CUBIT-Det", {})
    labeled_stems_by_split: dict[str, set[str]] = {"train2017": set(), "test2017": set()}
    for original_split in ("train2017", "test2017"):
        label_dir = labels_root / original_split
        image_dir = images_root / original_split
        split = _canonical_split(original_split)
        for label_path in sorted(label_dir.glob("*.txt")):
            labeled_stems_by_split[original_split].add(label_path.stem)
            image_path = _resolve_image_for_stem(image_dir, label_path.stem)
            if image_path is None:
                excluded_source["missing_image"] = int(excluded_source.get("missing_image", 0)) + 1
                continue
            annotations: list[dict[str, Any]] = []
            for index, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
                parts = line.split()
                if len(parts) != 5:
                    excluded_source["invalid_label_row"] = int(excluded_source.get("invalid_label_row", 0)) + 1
                    continue
                raw_class_id, x_center, y_center, width, height = parts
                mapping = CUBIT_DET_LABEL_MAP.get(raw_class_id)
                if mapping is None:
                    excluded_source[f"unknown_class_id:{raw_class_id}"] = int(
                        excluded_source.get(f"unknown_class_id:{raw_class_id}", 0)
                    ) + 1
                    continue
                source_label, issue_code = mapping
                x_center = float(x_center)
                y_center = float(y_center)
                width = float(width)
                height = float(height)
                box = _safe_normalize_box(
                    x_min=x_center - (width / 2.0),
                    y_min=y_center - (height / 2.0),
                    x_max=x_center + (width / 2.0),
                    y_max=y_center + (height / 2.0),
                    excluded_source=excluded_source,
                    excluded_key="invalid_box",
                )
                if box is None:
                    continue
                annotations.append(
                    _source_annotation(
                        annotation_index=index,
                        issue_code=issue_code,
                        source_label=source_label,
                        box=box,
                        source_dataset="CUBIT-Det",
                        original_split=original_split,
                        annotation_type="bbox",
                        raw_annotation_path=str(label_path),
                        raw_class_id=raw_class_id,
                    )
                )
                _update_mapping_summary(mapping_summary, source_dataset="CUBIT-Det", source_label=source_label, issue_code=issue_code)
            if not annotations:
                excluded_source["empty_annotations"] = int(excluded_source.get("empty_annotations", 0)) + 1
                continue
            row_id = f"cubit-det:{original_split}:{label_path.stem}"
            local_image_path = assets_root / original_split / _asset_name_for_record(row_id, image_path)
            _materialize_asset(image_path, local_image_path)
            records.append(
                _base_record(
                    row_id=row_id,
                    split=split,
                    source_dataset="CUBIT-Det",
                    original_split=original_split,
                    image_path=str(local_image_path),
                    raw_image_path=str(image_path),
                    annotation_type="bbox",
                    annotations=annotations,
                    source_metadata={"raw_annotation_path": str(label_path)},
                )
            )
    val_image_dir = images_root / "val2017"
    excluded_source["excluded_val2017_unlabeled_images"] = sum(1 for path in val_image_dir.iterdir() if path.is_file()) if val_image_dir.exists() else 0
    train_image_dir = images_root / "train2017"
    if train_image_dir.exists():
        excluded_source["excluded_train2017_unlabeled_images"] = sum(
            1 for path in train_image_dir.iterdir() if path.is_file() and path.stem not in labeled_stems_by_split["train2017"]
        )
    return records


def _load_codebrim_metadata(path: Path) -> dict[str, list[str]]:
    records: dict[str, list[str]] = {}
    root = _load_xml(path)
    for defect in root.findall("Defect"):
        name = str(defect.attrib.get("name") or "").strip()
        if not name:
            continue
        active = [child.tag for child in list(defect) if str(child.text or "").strip() == "1"]
        records[name] = active
    return records


def _normalize_codebrim_dataset(args: argparse.Namespace, *, mapping_summary: dict[str, Any], excluded: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    root = Path(args.raw_dataset_root) / "classification_dataset_balanced"
    assets_root = Path(args.output_dir) / "assets" / "CODEBRIM"
    defects_meta = _load_codebrim_metadata(root / "metadata" / "defects.xml")
    background_meta = _load_codebrim_metadata(root / "metadata" / "background.xml")
    excluded_source = excluded.setdefault("CODEBRIM", {})
    for original_split in ("train", "val", "test"):
        split = _canonical_split(original_split)
        defects_dir = root / original_split / "defects"
        background_dir = root / original_split / "background"
        for image_path in sorted(defects_dir.glob("*")):
            if not image_path.is_file():
                continue
            active_labels = defects_meta.get(image_path.name)
            if active_labels is None:
                excluded_source["missing_defect_metadata"] = int(excluded_source.get("missing_defect_metadata", 0)) + 1
                continue
            annotations: list[dict[str, Any]] = []
            for index, raw_label in enumerate(active_labels):
                if raw_label == "Background":
                    continue
                issue_code = CODEBRIM_LABEL_MAP.get(raw_label)
                if not issue_code:
                    excluded_source[f"unknown_label:{raw_label}"] = int(excluded_source.get(f"unknown_label:{raw_label}", 0)) + 1
                    continue
                annotations.append(
                    _source_annotation(
                        annotation_index=index,
                        issue_code=issue_code,
                        source_label=raw_label,
                        box=_normalize_box(0.0, 0.0, 1.0, 1.0),
                        source_dataset="CODEBRIM",
                        original_split=original_split,
                        annotation_type="full_image_crop",
                        raw_annotation_path=str(root / "metadata" / "defects.xml"),
                    )
                )
                _update_mapping_summary(mapping_summary, source_dataset="CODEBRIM", source_label=raw_label, issue_code=issue_code)
            if not annotations:
                excluded_source["empty_defect_labels"] = int(excluded_source.get("empty_defect_labels", 0)) + 1
                continue
            row_id = f"codebrim:{original_split}:defects:{image_path.stem}"
            local_image_path = assets_root / original_split / _asset_name_for_record(row_id, image_path, prefix="defects__")
            _materialize_asset(image_path, local_image_path)
            records.append(
                _base_record(
                    row_id=row_id,
                    split=split,
                    source_dataset="CODEBRIM",
                    original_split=original_split,
                    image_path=str(local_image_path),
                    raw_image_path=str(image_path),
                    annotation_type="full_image_crop",
                    annotations=annotations,
                    source_metadata={"raw_annotation_path": str(root / "metadata" / "defects.xml"), "category": "defects"},
                )
            )
        for image_path in sorted(background_dir.glob("*")):
            if not image_path.is_file():
                continue
            active_labels = background_meta.get(image_path.name)
            if active_labels is None:
                excluded_source["missing_background_metadata"] = int(excluded_source.get("missing_background_metadata", 0)) + 1
                continue
            if "Background" not in active_labels:
                excluded_source["invalid_background_label"] = int(excluded_source.get("invalid_background_label", 0)) + 1
                continue
            row_id = f"codebrim:{original_split}:background:{image_path.stem}"
            local_image_path = assets_root / original_split / _asset_name_for_record(row_id, image_path, prefix="background__")
            _materialize_asset(image_path, local_image_path)
            records.append(
                _base_record(
                    row_id=row_id,
                    split=split,
                    source_dataset="CODEBRIM",
                    original_split=original_split,
                    image_path=str(local_image_path),
                    raw_image_path=str(image_path),
                    annotation_type="negative_crop",
                    annotations=[],
                    hard_negative_candidate=True,
                    source_metadata={"raw_annotation_path": str(root / "metadata" / "background.xml"), "category": "background"},
                )
            )
    return records


def normalize_raw_records(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    mapping_summary: dict[str, Any] = {"sources": {}}
    excluded: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    records.extend(_normalize_mbdd_dataset(args, mapping_summary=mapping_summary, excluded=excluded))
    records.extend(_normalize_cubit_dataset(args, mapping_summary=mapping_summary, excluded=excluded))
    records.extend(_normalize_codebrim_dataset(args, mapping_summary=mapping_summary, excluded=excluded))
    filtered_records: list[dict[str, Any]] = []
    removed_crop_records = 0
    kept_crop_exception_records = 0
    for record in records:
        annotation_type = str(record.get("annotation_type") or "").strip().lower()
        if annotation_type in CROP_ANNOTATION_TYPES:
            issue_codes = {
                str(annotation.get("issue_code") or "").strip()
                for annotation in list(record.get("annotations") or [])
                if str(annotation.get("issue_code") or "").strip()
            }
            if issue_codes & QUERY_V2_CROP_ISSUE_CODE_EXCEPTIONS:
                kept_crop_exception_records += 1
            else:
                removed_crop_records += 1
                continue
        filtered_records.append(record)
    if removed_crop_records > 0:
        excluded.setdefault("global_filters", {})
        excluded["global_filters"]["removed_crop_records"] = int(removed_crop_records)
    if kept_crop_exception_records > 0:
        excluded.setdefault("global_filters", {})
        excluded["global_filters"]["kept_crop_exception_records"] = int(kept_crop_exception_records)
    records = filtered_records
    provenance = [_record_provenance(record) for record in records]
    stats = summarize_records(records, excluded=excluded)
    return records, provenance, stats, mapping_summary


def summarize_records(records: list[dict[str, Any]], *, excluded: Optional[dict[str, Any]] = None, synthesized_count: int = 0) -> dict[str, Any]:
    source_counts = Counter(record["source_dataset"] for record in records)
    split_counts = Counter(record["split"] for record in records)
    issue_counts = Counter()
    negative_count = 0
    positive_count = 0
    for record in records:
        if record["annotations"]:
            positive_count += 1
        else:
            negative_count += 1
        for annotation in record["annotations"]:
            issue_counts[annotation["issue_code"]] += 1
    return {
        "record_count": len(records),
        "positive_record_count": positive_count,
        "negative_record_count": negative_count,
        "source_counts": dict(sorted(source_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "issue_code_counts": dict(sorted(issue_counts.items())),
        "synthesized_record_count": int(synthesized_count),
        "excluded": dict(excluded or {}),
    }


def _write_normalize_outputs(
    *,
    output_dir: Path,
    base_records: list[dict[str, Any]],
    provenance: list[dict[str, Any]],
    stats: dict[str, Any],
    mapping_summary: dict[str, Any],
) -> None:
    common.write_jsonl(output_dir / "base_records.jsonl", base_records)
    common.write_jsonl(output_dir / "provenance.jsonl", provenance)
    common.write_json(output_dir / "stats.json", stats)
    common.write_json(output_dir / "mapping_summary.json", mapping_summary)


def _load_openrouter_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = common.load_jsonl(path)
    cache: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get("cache_key") or "").strip()
        if key:
            cache[key] = row
    return cache


def _write_openrouter_cache(path: Path, cache: dict[str, dict[str, Any]]) -> None:
    rows = [cache[key] for key in sorted(cache)]
    common.write_jsonl(path, rows)


def _cache_key_for_record(record: dict[str, Any], *, model_id: str) -> str:
    payload = {
        "prompt_version": PROMPT_VERSION,
        "model_id": model_id,
        "row_id": record["row_id"],
        "split": record["split"],
        "annotations": [
            {
                "annotation_index": item["annotation_index"],
                "issue_code": item["issue_code"],
                "source_label": item["source_label"],
                "box": item["box"],
            }
            for item in record["annotations"]
        ],
        "is_negative": bool(record["is_negative"]),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _record_issue_codes(record: dict[str, Any]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for annotation in record["annotations"]:
        issue_code = annotation["issue_code"]
        if issue_code not in seen:
            seen.add(issue_code)
            out.append(issue_code)
    return out


def _default_inspection_request(record: dict[str, Any]) -> str:
    return prompt_library.CANONICAL_QUERY_INSPECTION_REQUEST


def _default_asset_context(record: dict[str, Any]) -> str:
    source = record["source_dataset"]
    if source == "MBDD2025":
        return "UAV-captured building inspection image."
    if source == "CUBIT-Det":
        return "High-resolution infrastructure inspection image."
    if source == "CODEBRIM":
        return "Concrete defect crop for close-up visual inspection."
    return "Structure inspection image."


def _default_reasoning_text(record: dict[str, Any]) -> str:
    if record["is_negative"]:
        return "The image is treated as a hard negative because no supported defect annotations are present."
    return "The synthetic labels are grounded in the provided source annotations and converted to the expanded Inspector MD ontology."


def _default_proposals(record: dict[str, Any]) -> list[dict[str, Any]]:
    proposals: list[dict[str, Any]] = []
    seen: set[str] = set()
    for annotation in record["annotations"]:
        issue_code = annotation["issue_code"]
        if issue_code in seen:
            continue
        seen.add(issue_code)
        proposals.append(
            {
                "issue_code": issue_code,
                "evidence": f"Visible evidence consistent with {_issue_title(issue_code).lower()} is present in the annotated region.",
            }
        )
    return proposals


def _default_findings(record: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for annotation in record["annotations"]:
        issue = ontology.get_issue(annotation["issue_code"])
        findings.append(
            {
                "annotation_index": annotation["annotation_index"],
                "issue_code": annotation["issue_code"],
                "title": issue.title,
                "evidence": [f"Visible evidence consistent with {issue.title.lower()} is present in the annotated region."],
                "recommended_action": issue.default_recommended_action,
                "insufficient_evidence": False,
            }
        )
    return findings


def _default_synthetic_payload(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "inspection_request": _default_inspection_request(record),
        "asset_context": _default_asset_context(record),
        "reasoning_text": _default_reasoning_text(record),
        "proposals": [] if record["is_negative"] else _default_proposals(record),
        "findings": [] if record["is_negative"] else _default_findings(record),
    }


def _build_prompt_messages(record: dict[str, Any]) -> list[dict[str, Any]]:
    with Image.open(Path(record["image_path"])) as image:
        image_url = common.to_data_url(image.convert("RGB"))
    prompt_payload = {
        "task": "Generate training text for Inspector MD from the provided labeled inspection image.",
        "constraints": {
            "allowed_issue_codes": _record_issue_codes(record),
            "return_empty_findings_for_negative": True,
            "cost_policy": "Do not emit prices or exact dollar values.",
            "compliance_policy": "Do not emit legal judgments. Use only text fields listed in the schema.",
        },
        "record": {
            "row_id": record["row_id"],
            "source_dataset": record["source_dataset"],
            "split": record["split"],
            "is_negative": bool(record["is_negative"]),
            "annotation_type": record["annotation_type"],
            "annotations": [
                {
                    "annotation_index": item["annotation_index"],
                    "issue_code": item["issue_code"],
                    "source_label": item["source_label"],
                    "box": item["box"],
                }
                for item in record["annotations"]
            ],
        },
        "response_schema": {
            "asset_context": "string",
            "reasoning_text": "string",
            "proposals": [{"issue_code": "string", "evidence": "string"}],
            "findings": [
                {
                    "annotation_index": "integer",
                    "issue_code": "string",
                    "title": "string",
                    "evidence": ["string"],
                    "recommended_action": "string",
                    "insufficient_evidence": "boolean",
                }
            ],
        },
    }
    system_text = (
        "You are generating synthetic training text for a building-inspection copilot. "
        "Use only the provided annotations. Return valid JSON only. "
        "For negative examples, return empty proposals and empty findings."
    )
    user_text = json.dumps(prompt_payload, ensure_ascii=False, indent=2)
    return [
        {"role": "system", "content": system_text},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        },
    ]


def _as_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        text = str(value).strip()
        return [text] if text else []
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text:
            out.append(text)
    return out


def _normalize_synthetic_payload(record: dict[str, Any], payload: Any) -> dict[str, Any]:
    template = _default_synthetic_payload(record)
    if not isinstance(payload, dict):
        payload = {}
    if record["is_negative"]:
        return {
            "inspection_request": str(template["inspection_request"]).strip(),
            "asset_context": str(payload.get("asset_context") or template["asset_context"]).strip(),
            "reasoning_text": str(payload.get("reasoning_text") or template["reasoning_text"]).strip(),
            "proposals": [],
            "findings": [],
        }
    allowed_issue_codes = set(_record_issue_codes(record))
    proposals_by_issue = {item["issue_code"]: dict(item) for item in template["proposals"]}
    for raw_item in list(payload.get("proposals") or []):
        if not isinstance(raw_item, dict):
            continue
        issue_code_raw = raw_item.get("issue_code")
        try:
            issue_code = ontology.normalize_issue_code(issue_code_raw)
        except ValueError:
            continue
        if issue_code not in allowed_issue_codes:
            continue
        evidence = str(raw_item.get("evidence") or "").strip()
        if evidence:
            proposals_by_issue[issue_code] = {"issue_code": issue_code, "evidence": evidence}

    findings_by_index = {item["annotation_index"]: dict(item) for item in template["findings"]}
    for raw_item in list(payload.get("findings") or []):
        if not isinstance(raw_item, dict):
            continue
        try:
            annotation_index = int(raw_item.get("annotation_index"))
        except (TypeError, ValueError):
            continue
        match = next((item for item in record["annotations"] if int(item["annotation_index"]) == annotation_index), None)
        if match is None:
            continue
        try:
            issue_code = ontology.normalize_issue_code(raw_item.get("issue_code"))
        except ValueError:
            issue_code = match["issue_code"]
        if issue_code != match["issue_code"]:
            continue
        evidence = _as_string_list(raw_item.get("evidence")) or findings_by_index[annotation_index]["evidence"]
        findings_by_index[annotation_index] = {
            "annotation_index": annotation_index,
            "issue_code": issue_code,
            "title": str(raw_item.get("title") or findings_by_index[annotation_index]["title"]).strip(),
            "evidence": evidence,
            "recommended_action": str(
                raw_item.get("recommended_action") or findings_by_index[annotation_index]["recommended_action"]
            ).strip(),
            "insufficient_evidence": bool(raw_item.get("insufficient_evidence", findings_by_index[annotation_index]["insufficient_evidence"])),
        }
    return {
        "inspection_request": str(template["inspection_request"]).strip(),
        "asset_context": str(payload.get("asset_context") or template["asset_context"]).strip(),
        "reasoning_text": str(payload.get("reasoning_text") or template["reasoning_text"]).strip(),
        "proposals": list(proposals_by_issue.values()),
        "findings": [findings_by_index[item["annotation_index"]] for item in record["annotations"]],
    }


def _is_hard_example(record: dict[str, Any]) -> bool:
    if record["is_negative"]:
        return bool(record["hard_negative_candidate"])
    issue_codes = {annotation["issue_code"] for annotation in record["annotations"]}
    if len(record["annotations"]) > 1 or len(issue_codes) > 1:
        return True
    return bool(issue_codes & HARD_EXAMPLE_ISSUES)


def _select_records_for_synthesis(records: list[dict[str, Any]], *, max_positive_records: int, max_hard_negatives: int) -> list[dict[str, Any]]:
    positives = [record for record in records if not record["is_negative"]]
    if int(max_positive_records) > 0:
        positives = positives[: int(max_positive_records)]
    negatives = [
        record
        for record in records
        if record["is_negative"] and record["source_dataset"] == "CODEBRIM" and record["hard_negative_candidate"]
    ]
    negatives.sort(key=lambda item: _stable_bucket(item["row_id"]))
    if int(max_hard_negatives) > 0:
        negatives = negatives[: int(max_hard_negatives)]
    return positives + negatives


def _synthetic_manifest_row(record: dict[str, Any], synthesized: dict[str, Any]) -> dict[str, Any]:
    findings_payload: list[dict[str, Any]] = []
    by_index = {item["annotation_index"]: item for item in list(synthesized.get("findings") or []) if isinstance(item, dict)}
    for annotation in record["annotations"]:
        index = int(annotation["annotation_index"])
        generated = by_index.get(index, {})
        issue_code = annotation["issue_code"]
        base_payload = {
            "finding_id": f"{record['row_id'].replace(':', '_')}_finding_{index + 1:03d}",
            "issue_code": issue_code,
            "title": str(generated.get("title") or _issue_title(issue_code)).strip(),
            "box": dict(annotation["box"]),
            "evidence": _as_string_list(generated.get("evidence")),
            "recommended_action": str(generated.get("recommended_action") or ontology.get_issue(issue_code).default_recommended_action).strip(),
            "cost_band": ontology.get_issue(issue_code).default_cost_band,
            "possible_compliance_issue": False,
            "insufficient_evidence": bool(generated.get("insufficient_evidence", False)),
            "compliance_note": "",
            "source_detect_labels": list(annotation["source_detect_labels"]),
            "spatial_ref_index": index,
        }
        findings_payload.append(rule_stubs.apply_rule_stubs(task_schema.Finding.from_payload(base_payload)).to_payload())
    proposals = [task_schema.IssueProposal.from_payload(item).to_payload() for item in list(synthesized.get("proposals") or [])]
    return {
        "row_id": record["row_id"],
        "split": record["split"],
        "image_path": record["image_path"],
        "inspection_request": str(synthesized.get("inspection_request") or _default_inspection_request(record)).strip(),
        "asset_context": str(synthesized.get("asset_context") or _default_asset_context(record)).strip(),
        "hard_example": _is_hard_example(record),
        "expected_proposals": proposals,
        "expected_findings": findings_payload,
        "reasoning_text": str(synthesized.get("reasoning_text") or _default_reasoning_text(record)).strip(),
        "source_dataset": record["source_dataset"],
        "source_metadata": {
            "original_split": record["original_split"],
            "annotation_type": record["annotation_type"],
            **dict(record.get("source_metadata") or {}),
        },
    }


def synthesize_manifest(
    args: argparse.Namespace,
    *,
    base_records: list[dict[str, Any]],
    call_openrouter_fn: Optional[Callable[..., tuple[str, dict[str, Any], float]]] = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    teacher_model_id = str(args.teacher_model_id or "").strip()
    synthesis_mode = "openrouter" if teacher_model_id else "template_only"
    cache_path = Path(args.output_dir) / "openrouter_cache.jsonl"
    cache = _load_openrouter_cache(cache_path)
    synthesize_rows = _select_records_for_synthesis(
        base_records,
        max_positive_records=int(args.max_positive_records),
        max_hard_negatives=int(args.max_hard_negatives),
    )
    caller = call_openrouter_fn or _call_openrouter_chat_api
    api_key = ""
    if synthesis_mode == "openrouter":
        common.maybe_load_env_file(args.env_file, override=False)
        api_key = _resolve_openrouter_api_key(args.api_key, args.api_key_env_var)
    manifest_rows: list[dict[str, Any]] = []
    used_cache = 0
    for record in synthesize_rows:
        cache_key = _cache_key_for_record(record, model_id=teacher_model_id or "template_only")
        cached = cache.get(cache_key)
        raw_payload: Any = {}
        if synthesis_mode == "openrouter" and cached is not None:
            used_cache += 1
            raw_payload = common.parse_prediction_json(str(cached.get("answer_text") or "")) or cached.get("parsed_payload") or {}
        elif synthesis_mode == "openrouter":
            answer_text, raw_response, latency_ms = caller(
                api_base=args.api_base,
                api_key=api_key,
                model_id=teacher_model_id,
                messages=_build_prompt_messages(record),
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                max_tokens=int(args.max_tokens),
                timeout=float(args.timeout),
                retry_429_max_retries=int(args.retry_429_max_retries),
                retry_429_backoff_s=float(args.retry_429_backoff_s),
                retry_429_max_backoff_s=float(args.retry_429_max_backoff_s),
            )
            raw_payload = common.parse_prediction_json(answer_text) or {}
            cache[cache_key] = {
                "cache_key": cache_key,
                "row_id": record["row_id"],
                "model_id": teacher_model_id,
                "prompt_version": PROMPT_VERSION,
                "answer_text": answer_text,
                "parsed_payload": raw_payload,
                "raw_response": raw_response,
                "latency_ms": latency_ms,
            }
        manifest_rows.append(_synthetic_manifest_row(record, _normalize_synthetic_payload(record, raw_payload)))
    _write_openrouter_cache(cache_path, cache)
    summary = {
        "synthesis_mode": synthesis_mode,
        "teacher_model_id": teacher_model_id,
        "selected_record_count": len(synthesize_rows),
        "used_cache_count": used_cache,
        "manifest_row_count": len(manifest_rows),
    }
    return manifest_rows, cache, summary


def _load_base_records(output_dir: Path) -> list[dict[str, Any]]:
    path = Path(output_dir) / "base_records.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing normalized base records: {path}")
    return common.load_jsonl(path)


def build_merged_synth_dataset(
    args: argparse.Namespace,
    *,
    call_openrouter_fn: Optional[Callable[..., tuple[str, dict[str, Any], float]]] = None,
) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"stage": args.stage, "output_dir": str(output_dir)}
    base_records: list[dict[str, Any]]
    stats: dict[str, Any]
    if args.stage in {"normalize", "all"}:
        base_records, provenance, stats, mapping_summary = normalize_raw_records(args)
        _write_normalize_outputs(
            output_dir=output_dir,
            base_records=base_records,
            provenance=provenance,
            stats=stats,
            mapping_summary=mapping_summary,
        )
        summary["normalize"] = {
            "base_records_jsonl": str(output_dir / "base_records.jsonl"),
            "provenance_jsonl": str(output_dir / "provenance.jsonl"),
            "stats_json": str(output_dir / "stats.json"),
            "mapping_summary_json": str(output_dir / "mapping_summary.json"),
            "record_count": len(base_records),
        }
    else:
        base_records = _load_base_records(output_dir)
        stats = summarize_records(base_records)
    if args.stage in {"synthesize", "all"}:
        manifest_rows, _cache, synth_summary = synthesize_manifest(args, base_records=base_records, call_openrouter_fn=call_openrouter_fn)
        common.write_json(output_dir / "synthetic_manifest.json", manifest_rows)
        updated_stats = dict(stats)
        updated_stats["synthesized_record_count"] = len(manifest_rows)
        common.write_json(output_dir / "stats.json", updated_stats)
        summary["synthesize"] = {
            "synthetic_manifest_json": str(output_dir / "synthetic_manifest.json"),
            "openrouter_cache_jsonl": str(output_dir / "openrouter_cache.jsonl"),
            **synth_summary,
        }
    common.write_json(output_dir / "build_summary.json", summary)
    return summary


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    summary = build_merged_synth_dataset(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
