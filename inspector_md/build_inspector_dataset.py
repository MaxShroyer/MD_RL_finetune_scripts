#!/usr/bin/env python3
"""Build detect/point/query datasets for Inspector MD."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Optional

from datasets import Dataset, DatasetDict, Features, Value
from PIL import Image
try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover - optional dependency fallback
    _tqdm = None

from inspector_md import common, ontology, openrouter_grader, prompt_library, query_compact, task_schema

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = common.repo_relative("configs", "build_inspector_dataset_default.json")
QUERY_REFRESH_PROMPT_VERSION = "inspector_md_query_refresh_v2"
CROP_ANNOTATION_TYPES = {"full_image_crop", "negative_crop"}
QUERY_V2_CROP_ISSUE_CODE_EXCEPTIONS = {"exposed_rebar"}
REQUIRED_MANIFEST_FIELDS = (
    "row_id",
    "image_path",
    "split",
    "inspection_request",
    "asset_context",
    "hard_example",
    "expected_proposals",
    "expected_findings",
)
REQUIRED_PROPOSAL_FIELDS = ("issue_code", "evidence")
REQUIRED_FINDING_FIELDS = (
    "issue_code",
    "title",
    "box",
    "evidence",
    "recommended_action",
    "cost_band",
    "possible_compliance_issue",
    "insufficient_evidence",
    "compliance_note",
    "source_detect_labels",
    "spatial_ref_index",
)


class _NullProgressBar:
    def update(self, _n: int = 1) -> None:
        return None

    def set_postfix(self, *args: Any, **kwargs: Any) -> None:
        return None

    def close(self) -> None:
        return None


def _make_progress_bar(*, total: Optional[int], desc: str, enabled: bool):
    if (not enabled) or _tqdm is None:
        return _NullProgressBar()
    return _tqdm(total=total, desc=desc, unit="item", dynamic_ncols=True, leave=False)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    raw_argv = list(argv) if argv is not None else list(os.sys.argv[1:])
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    pre_args, _ = pre_parser.parse_known_args(raw_argv)
    config_path = common.resolve_config_path(pre_args.config, script_dir=SCRIPT_DIR)
    config = common.load_json_config(config_path, default_path=DEFAULT_CONFIG_PATH)

    parser = argparse.ArgumentParser(description="Build local detect/point/query datasets for Inspector MD.")
    parser.add_argument("--config", default=str(config_path))
    parser.add_argument("--env-file", default=str(common.repo_relative(".env.staging")))
    parser.add_argument("--source-manifest", default="")
    parser.add_argument("--output-root", default=str(common.repo_relative("outputs")))
    parser.add_argument("--detect-output-dir", default=str(common.repo_relative("outputs", "inspector_detect_v1")))
    parser.add_argument("--point-output-dir", default=str(common.repo_relative("outputs", "inspector_point_v1")))
    parser.add_argument("--query-output-dir", default=str(common.repo_relative("outputs", "inspector_query_issues_v2")))
    parser.add_argument("--query-proposal-output-dir", default="")
    parser.add_argument("--query-finding-output-dir", default="")
    parser.add_argument("--query-reasoning-output-dir", default="")
    parser.add_argument("--detect-exclude-source-datasets", nargs="*", default=["CODEBRIM"])
    parser.add_argument("--query-text-refresh-mode", choices=("template_only", "openrouter"), default="template_only")
    parser.add_argument("--query-teacher-model-id", default="")
    parser.add_argument("--query-teacher-api-base", default=openrouter_grader.DEFAULT_OPENROUTER_API_BASE)
    parser.add_argument("--query-teacher-api-key", default="")
    parser.add_argument("--query-teacher-api-key-env-var", default=openrouter_grader.DEFAULT_OPENROUTER_ENV_VAR)
    parser.add_argument(
        "--query-sft-format",
        choices=(query_compact.TARGET_FORMAT_JSON_ISSUE_LIST,),
        default=query_compact.TARGET_FORMAT_JSON_ISSUE_LIST,
    )
    parser.add_argument("--query-text-cache-jsonl", default=str(common.repo_relative("outputs", "inspector_query_text_cache.jsonl")))
    parser.add_argument("--query-teacher-timeout", type=float, default=90.0)
    parser.add_argument("--query-refresh-max-concurrency", type=int, default=1)
    parser.add_argument("--query-refresh-retries", type=int, default=2)
    parser.add_argument("--query-refresh-retry-backoff-s", type=float, default=2.0)
    parser.add_argument("--query-refresh-cache-flush-every", type=int, default=25)
    parser.add_argument("--query-refresh-num-shards", type=int, default=1)
    parser.add_argument("--query-refresh-shard-index", type=int, default=0)
    parser.add_argument("--query-refresh-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.5)

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
    args.env_file = str(common.resolve_path(args.env_file, module_root=SCRIPT_DIR))
    args.output_root = common.resolve_path(args.output_root, module_root=SCRIPT_DIR)
    args.detect_output_dir = common.resolve_path(args.detect_output_dir, module_root=SCRIPT_DIR)
    args.point_output_dir = common.resolve_path(args.point_output_dir, module_root=SCRIPT_DIR)
    args.query_output_dir = common.resolve_path(args.query_output_dir, module_root=SCRIPT_DIR)
    args.query_proposal_output_dir = common.resolve_path(args.query_proposal_output_dir, module_root=SCRIPT_DIR) if str(args.query_proposal_output_dir or "").strip() else Path("")
    args.query_finding_output_dir = common.resolve_path(args.query_finding_output_dir, module_root=SCRIPT_DIR) if str(args.query_finding_output_dir or "").strip() else Path("")
    args.query_reasoning_output_dir = common.resolve_path(args.query_reasoning_output_dir, module_root=SCRIPT_DIR) if str(args.query_reasoning_output_dir or "").strip() else Path("")
    args.query_text_cache_jsonl = common.resolve_path(args.query_text_cache_jsonl, module_root=SCRIPT_DIR)
    args.source_manifest = str(common.resolve_path(args.source_manifest, module_root=SCRIPT_DIR)) if args.source_manifest else ""
    args.detect_exclude_source_datasets = [str(item).strip() for item in list(args.detect_exclude_source_datasets or []) if str(item).strip()]
    if args.progress is None:
        args.progress = bool(getattr(os.sys.stderr, "isatty", lambda: False)())
    if int(args.query_refresh_max_concurrency) <= 0:
        raise ValueError("--query-refresh-max-concurrency must be >= 1")
    if int(args.query_refresh_retries) < 0:
        raise ValueError("--query-refresh-retries must be >= 0")
    if float(args.query_refresh_retry_backoff_s) <= 0.0:
        raise ValueError("--query-refresh-retry-backoff-s must be > 0")
    if int(args.query_refresh_cache_flush_every) <= 0:
        raise ValueError("--query-refresh-cache-flush-every must be >= 1")
    if int(args.query_refresh_num_shards) <= 0:
        raise ValueError("--query-refresh-num-shards must be >= 1")
    if int(args.query_refresh_shard_index) < 0 or int(args.query_refresh_shard_index) >= int(args.query_refresh_num_shards):
        raise ValueError("--query-refresh-shard-index must be in [0, --query-refresh-num-shards)")
    if int(args.query_refresh_num_shards) > 1 and not bool(args.query_refresh_only):
        raise ValueError("--query-refresh-num-shards > 1 requires --query-refresh-only")
    return args


def _features() -> Features:
    return Features(
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
    )


def _write_seed_image(path: Path, *, color: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (96, 64), color).save(path)
    return str(path)


def _seed_records(output_root: Path) -> list[dict[str, Any]]:
    seed_dir = output_root / "seed_images"
    return [
        {
            "row_id": "seed-1",
            "split": "train",
            "image_path": _write_seed_image(seed_dir / "seed-1.png", color="lightgray"),
            "inspection_request": prompt_library.CANONICAL_QUERY_INSPECTION_REQUEST,
            "asset_context": "Two-story residential exterior",
            "hard_example": False,
            "expected_findings": [
                {
                    "issue_code": "roof_cover_damage",
                    "box": {"x_min": 0.08, "y_min": 0.05, "x_max": 0.72, "y_max": 0.34},
                    "title": "Roof Cover Damage",
                    "evidence": ["Missing shingle area is visible along the roof slope."],
                    "recommended_action": "Repair the damaged roof covering and inspect adjacent roofing.",
                    "cost_band": "high",
                    "possible_compliance_issue": True,
                    "insufficient_evidence": False,
                    "compliance_note": "Possible compliance issue related to water_intrusion_risk. This is not a legal determination and needs field review.",
                    "source_detect_labels": ["missing shingle"],
                    "spatial_ref_index": 0,
                }
            ],
            "expected_proposals": [{"issue_code": "roof_cover_damage", "evidence": "Missing shingle area is visible along the roof slope."}],
        },
        {
            "row_id": "seed-2",
            "split": "validation",
            "image_path": _write_seed_image(seed_dir / "seed-2.png", color="white"),
            "inspection_request": prompt_library.CANONICAL_QUERY_INSPECTION_REQUEST,
            "asset_context": "Exterior entrance walkway",
            "hard_example": True,
            "expected_findings": [
                {
                    "issue_code": "blocked_egress_or_access",
                    "box": {"x_min": 0.35, "y_min": 0.42, "x_max": 0.91, "y_max": 0.92},
                    "title": "Blocked Egress or Access",
                    "evidence": ["Stored material appears to block the primary access path."],
                    "recommended_action": "Clear the blocked route and maintain unobstructed access or egress.",
                    "cost_band": "medium",
                    "possible_compliance_issue": True,
                    "insufficient_evidence": False,
                    "compliance_note": "Possible compliance issue related to egress_access. This is not a legal determination and needs field review.",
                    "source_detect_labels": ["blocked access route"],
                    "spatial_ref_index": 0,
                }
            ],
            "expected_proposals": [{"issue_code": "blocked_egress_or_access", "evidence": "Stored material appears to block the primary access path."}],
        },
        {
            "row_id": "seed-3",
            "split": "test",
            "image_path": _write_seed_image(seed_dir / "seed-3.png", color="silver"),
            "inspection_request": prompt_library.CANONICAL_QUERY_INSPECTION_REQUEST,
            "asset_context": "Commercial exterior facade",
            "hard_example": True,
            "expected_findings": [
                {
                    "issue_code": "corrosion_rust",
                    "box": {"x_min": 0.14, "y_min": 0.22, "x_max": 0.46, "y_max": 0.68},
                    "title": "Corrosion or Rust",
                    "evidence": ["Rust staining is visible on the exposed metal component."],
                    "recommended_action": "Remove corrosion where feasible and repair or replace compromised metal components.",
                    "cost_band": "medium",
                    "possible_compliance_issue": True,
                    "insufficient_evidence": False,
                    "compliance_note": "Possible compliance issue related to surface_stability. This is not a legal determination and needs field review.",
                    "source_detect_labels": ["rusted metal"],
                    "spatial_ref_index": 0,
                }
            ],
            "expected_proposals": [{"issue_code": "corrosion_rust", "evidence": "Rust staining is visible on the exposed metal component."}],
        },
    ]


def _require_keys(payload: dict[str, Any], required_keys: tuple[str, ...], *, context: str) -> None:
    missing = [key for key in required_keys if key not in payload]
    if missing:
        raise ValueError(f"{context} missing required key(s): {missing}")


def _normalize_split_name(raw_split: Any) -> str:
    split = str(raw_split or "").strip().lower()
    if split in {"val", "validation"}:
        return "validation"
    if split in {"train", "test"}:
        return split
    raise ValueError(f"Unsupported split: {raw_split!r}")


def _normalize_manifest_proposals(raw_proposals: Any, *, row_id: str) -> list[dict[str, Any]]:
    if not isinstance(raw_proposals, list):
        raise ValueError(f"row={row_id} expected_proposals must be a list")
    proposals: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw_item in enumerate(raw_proposals, start=1):
        if not isinstance(raw_item, dict):
            raise ValueError(f"row={row_id} proposal[{index}] must be an object")
        _require_keys(raw_item, REQUIRED_PROPOSAL_FIELDS, context=f"row={row_id} proposal[{index}]")
        proposal = task_schema.IssueProposal.from_payload(raw_item)
        if proposal.issue_code in seen:
            continue
        seen.add(proposal.issue_code)
        proposals.append(proposal.to_payload())
    return proposals


def _normalize_manifest_findings(raw_findings: Any, *, row_id: str) -> list[dict[str, Any]]:
    if not isinstance(raw_findings, list):
        raise ValueError(f"row={row_id} expected_findings must be a list")
    findings: list[dict[str, Any]] = []
    for index, raw_item in enumerate(raw_findings, start=1):
        if not isinstance(raw_item, dict):
            raise ValueError(f"row={row_id} finding[{index}] must be an object")
        _require_keys(raw_item, REQUIRED_FINDING_FIELDS, context=f"row={row_id} finding[{index}]")
        finding = task_schema.Finding.from_payload(raw_item)
        if not finding.source_detect_labels:
            raise ValueError(f"row={row_id} finding[{index}] must include at least one source_detect_labels entry")
        for label in finding.source_detect_labels:
            if ontology.is_generic_detect_label(label):
                raise ValueError(f"row={row_id} finding[{index}] uses generic detect label: {label!r}")
            detected_issue = ontology.issue_code_for_detect_label(label)
            if detected_issue != finding.issue_code:
                raise ValueError(
                    f"row={row_id} finding[{index}] detect label {label!r} does not map to issue_code={finding.issue_code!r}"
                )
        payload = finding.to_payload()
        if not payload["finding_id"]:
            payload["finding_id"] = f"{row_id}_finding_{index:03d}"
        findings.append(payload)
    return findings


def normalize_source_manifest(payload: Any, *, manifest_dir: Path, show_progress: bool = False) -> list[dict[str, Any]]:
    if not isinstance(payload, list):
        raise ValueError("source manifest must be a JSON array")
    normalized_records: list[dict[str, Any]] = []
    seen_row_ids: set[str] = set()
    progress = _make_progress_bar(total=len(payload), desc="normalize manifest", enabled=show_progress)
    for index, raw_record in enumerate(payload, start=1):
        if not isinstance(raw_record, dict):
            raise ValueError(f"manifest row {index} must be an object")
        _require_keys(raw_record, REQUIRED_MANIFEST_FIELDS, context=f"manifest row {index}")
        row_id = str(raw_record.get("row_id") or "").strip()
        if not row_id:
            raise ValueError(f"manifest row {index} row_id is required")
        if row_id in seen_row_ids:
            raise ValueError(f"Duplicate row_id: {row_id!r}")
        seen_row_ids.add(row_id)
        raw_image_path = Path(str(raw_record.get("image_path") or "")).expanduser()
        image_path = raw_image_path.resolve() if raw_image_path.is_absolute() else (manifest_dir / raw_image_path).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"row={row_id} image_path not found: {image_path}")
        request = task_schema.InspectionRequest.from_payload(
            {
                "image_path": str(image_path),
                "inspection_request": str(raw_record.get("inspection_request") or ""),
                "asset_context": str(raw_record.get("asset_context") or ""),
            }
        )
        proposals = _normalize_manifest_proposals(raw_record.get("expected_proposals"), row_id=row_id)
        findings = _normalize_manifest_findings(raw_record.get("expected_findings"), row_id=row_id)
        source_metadata = dict(raw_record.get("source_metadata") or {})
        annotation_type = str(source_metadata.get("annotation_type") or raw_record.get("annotation_type") or "").strip()
        if annotation_type in CROP_ANNOTATION_TYPES:
            issue_codes = {
                str(item.get("issue_code") or "").strip()
                for item in findings
                if str(item.get("issue_code") or "").strip()
            }
            if not (issue_codes & QUERY_V2_CROP_ISSUE_CODE_EXCEPTIONS):
                progress.update(1)
                continue
        normalized_records.append(
            {
                "row_id": row_id,
                "split": _normalize_split_name(raw_record.get("split")),
                "image_path": request.image_path,
                "inspection_request": request.inspection_request,
                "asset_context": request.asset_context,
                "hard_example": bool(raw_record.get("hard_example", False)),
                "expected_proposals": proposals,
                "expected_findings": findings,
                "source_dataset": str(raw_record.get("source_dataset") or "inspector_md"),
                "source_metadata": source_metadata,
            }
        )
        progress.update(1)
    progress.close()
    return normalized_records


def _split_records(records: list[dict[str, Any]], *, seed: int, val_fraction: float, test_fraction: float) -> dict[str, list[dict[str, Any]]]:
    if any(str(row.get("split") or "").strip() for row in records):
        out = {"train": [], "validation": [], "test": []}
        for row in records:
            split = str(row.get("split") or "").strip().lower()
            if split in {"val", "validation"}:
                out["validation"].append(row)
            elif split == "test":
                out["test"].append(row)
            else:
                out["train"].append(row)
        return out
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    holdout_count = max(1, int(round(len(shuffled) * float(val_fraction))))
    holdout = shuffled[:holdout_count]
    train = shuffled[holdout_count:]
    test_count = max(1, int(round(len(holdout) * float(test_fraction)))) if holdout else 0
    return {
        "train": train,
        "validation": holdout[test_count:],
        "test": holdout[:test_count],
    }


def _load_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.source_manifest:
        manifest_path = Path(args.source_manifest)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        return normalize_source_manifest(payload, manifest_dir=manifest_path.parent, show_progress=bool(args.progress))
    return normalize_source_manifest(
        _seed_records(Path(args.output_root)),
        manifest_dir=Path(args.output_root),
        show_progress=bool(args.progress),
    )


def _detect_rows(
    records: list[dict[str, Any]],
    *,
    excluded_source_datasets: set[str],
    show_progress: bool = False,
    progress_desc: str = "detect rows",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    progress = _make_progress_bar(total=len(records), desc=progress_desc, enabled=show_progress)
    for record in records:
        source_dataset = str(record.get("source_dataset") or "")
        if source_dataset in excluded_source_datasets:
            progress.update(1)
            continue
        answer_boxes: list[dict[str, Any]] = []
        for finding in list(record.get("expected_findings") or []):
            issue_code = ontology.normalize_issue_code(finding.get("issue_code"))
            box = finding.get("box") or {}
            labels = list(finding.get("source_detect_labels") or []) or list(ontology.detect_labels_for_issue(issue_code))
            for label in labels[:1]:
                answer_boxes.append(
                    {
                        "class_uid": "inspector_md:" + str(label).lower().replace(" ", "_"),
                        "class_name": label,
                        "x_min": float(box["x_min"]),
                        "y_min": float(box["y_min"]),
                        "x_max": float(box["x_max"]),
                        "y_max": float(box["y_max"]),
                    }
                )
        rows.append(
            {
                "image": str(record["image_path"]),
                "answer_boxes": json.dumps(answer_boxes, ensure_ascii=False),
                "source_dataset": source_dataset or "inspector_md",
                "source_collection": source_dataset or "inspector_md",
                "source_variant": str(record.get("source_metadata", {}).get("original_split") or "building_inspection"),
                "source_is_synthetic": False,
                "source_split": str(record.get("split") or ""),
                "source_image_id": str(record.get("row_id") or ""),
                "source_base_id": str(record.get("row_id") or ""),
                "split_group_id": str(record.get("row_id") or ""),
                "class_count": len(answer_boxes),
            }
        )
        progress.update(1)
    progress.close()
    return rows


def _query_issue_rows(
    records: list[dict[str, Any]],
    *,
    query_bundles: dict[str, dict[str, Any]],
    show_progress: bool = False,
    progress_desc: str = "issue rows",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    progress = _make_progress_bar(total=len(records), desc=progress_desc, enabled=show_progress)
    for record in records:
        bundle = query_bundles.get(str(record.get("row_id") or ""), _template_query_bundle(record))
        inspection_request = prompt_library.CANONICAL_QUERY_INSPECTION_REQUEST
        asset_context = str(bundle.get("asset_context") or record.get("asset_context") or "").strip()
        proposals = task_schema.normalize_issue_proposals({"issues": list(bundle.get("issues") or [])})
        final_answer_payload = task_schema.issue_list_payload(proposals)
        final_answer_json = json.dumps(final_answer_payload, ensure_ascii=False, sort_keys=True)
        source_metadata = dict(record.get("source_metadata") or {})
        spatial_refs = _record_spatial_refs(record)
        rows.append(
            {
                "row_id": str(record.get("row_id") or ""),
                "split": str(record.get("split") or ""),
                "task_type": "issues",
                "question": prompt_library.build_visible_issue_question_with_style(
                    inspection_request=inspection_request,
                    asset_context=asset_context,
                    prompt_style="request_only",
                    variation_key=str(record.get("row_id") or record.get("image_path") or ""),
                ),
                "image_path": str(record["image_path"]),
                "inspection_request": inspection_request,
                "asset_context": asset_context,
                "spatial_refs_json": json.dumps(spatial_refs, ensure_ascii=False),
                "reasoning_text": str(bundle.get("reasoning_text") or "").strip(),
                "hard_example": bool(record.get("hard_example", False)),
                "target_text": final_answer_json,
                "target_format": query_compact.TARGET_FORMAT_JSON_ISSUE_LIST,
                "final_answer_json": final_answer_json,
                "query_text_refresh_mode": str(bundle.get("query_text_refresh_mode") or ""),
                "issue_count": len(proposals),
                "is_multi_issue": len(proposals) > 1,
                "source_dataset": str(record.get("source_dataset") or ""),
                "source_annotation_type": str(source_metadata.get("annotation_type") or ""),
                "crop_derived": str(source_metadata.get("annotation_type") or "").strip() in CROP_ANNOTATION_TYPES,
            }
        )
        progress.update(1)
    progress.close()
    return rows


def _record_spatial_refs(record: dict[str, Any]) -> list[dict[str, float]]:
    findings = list(record.get("expected_findings") or [])
    refs_with_index: list[tuple[int, dict[str, float]]] = []
    for finding in findings:
        raw_box = (finding or {}).get("box")
        if raw_box is None:
            continue
        try:
            box = task_schema.Box.from_payload(raw_box).to_payload()
        except ValueError:
            continue
        spatial_ref_index = finding.get("spatial_ref_index", len(refs_with_index))
        try:
            order_index = int(spatial_ref_index)
        except (TypeError, ValueError):
            order_index = len(refs_with_index)
        refs_with_index.append((max(0, order_index), box))
    refs_with_index.sort(key=lambda item: item[0])
    refs: list[dict[str, float]] = []
    seen: set[tuple[float, float, float, float]] = set()
    for _index, box in refs_with_index:
        key = (
            float(box["x_min"]),
            float(box["y_min"]),
            float(box["x_max"]),
            float(box["y_max"]),
        )
        if key in seen:
            continue
        seen.add(key)
        refs.append(box)
    return refs


def _issue_box_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        for finding in list(record.get("expected_findings") or []):
            issue_code = str(finding.get("issue_code") or "").strip()
            if issue_code:
                counts[issue_code] += 1
    return {code: int(counts.get(code, 0)) for code in ontology.all_issue_codes()}


def _query_spatial_ref_summary(query_rows: dict[str, list[dict[str, Any]]]) -> tuple[dict[str, int], int]:
    by_split: dict[str, int] = {}
    total = 0
    for split_name, rows in query_rows.items():
        non_empty = sum(1 for row in rows if str(row.get("spatial_refs_json") or "").strip() not in {"", "[]"})
        by_split[split_name] = int(non_empty)
        total += int(non_empty)
    return by_split, total


def _make_dataset(rows: list[dict[str, Any]]) -> Dataset:
    return Dataset.from_list(rows, features=_features())


def _num_shards_by_split(split_rows: dict[str, list[dict[str, Any]]], *, target_rows_per_shard: int = 2048) -> dict[str, int]:
    shards: dict[str, int] = {}
    for split_name, rows in split_rows.items():
        row_count = len(rows)
        if row_count <= 0:
            continue
        shards[split_name] = max(1, (row_count + int(target_rows_per_shard) - 1) // int(target_rows_per_shard))
    return shards


def _query_cache_key_for_record(record: dict[str, Any], *, model_id: str) -> str:
    payload = {
        "row_id": str(record.get("row_id") or ""),
        "asset_context": str(record.get("asset_context") or ""),
        "expected_proposals": list(record.get("expected_proposals") or []),
        "expected_findings": list(record.get("expected_findings") or []),
        "model_id": str(model_id or "").strip(),
        "prompt_version": QUERY_REFRESH_PROMPT_VERSION,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()


def _query_refresh_shard_for_record(record: dict[str, Any], *, num_shards: int) -> int:
    if int(num_shards) <= 1:
        return 0
    row_id = str(record.get("row_id") or "").strip()
    digest = hashlib.sha1(row_id.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % int(num_shards)


def _select_query_refresh_records(
    records: list[dict[str, Any]],
    *,
    num_shards: int,
    shard_index: int,
) -> list[dict[str, Any]]:
    if int(num_shards) <= 1:
        return list(records)
    return [
        dict(record)
        for record in records
        if _query_refresh_shard_for_record(record, num_shards=int(num_shards)) == int(shard_index)
    ]


def _load_query_text_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    cache: dict[str, dict[str, Any]] = {}
    for row in common.load_jsonl(path):
        key = str(row.get("cache_key") or "").strip()
        if not key:
            continue
        cache[key] = dict(row)
    return cache


def _write_query_text_cache(path: Path, cache: dict[str, dict[str, Any]]) -> None:
    rows = [cache[key] for key in sorted(cache)]
    common.write_jsonl(path, rows)


def _template_query_bundle(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "inspection_request": prompt_library.CANONICAL_QUERY_INSPECTION_REQUEST,
        "asset_context": str(record.get("asset_context") or "").strip(),
        "reasoning_text": "",
        "issues": [
            {
                "type": str(item.get("issue_code") or "").strip(),
                "reasoning": str(item.get("evidence") or "").strip(),
            }
            for item in list(record.get("expected_proposals") or [])
        ],
    }


def _query_refresh_system_prompt() -> str:
    return (
        "You rewrite building-inspection dataset text for compact prompt training. "
        "Preserve the canonical defect labels and visible evidence intent. "
        "Do not write or rewrite inspection_request. "
        "Do not add hidden causes, legal conclusions, or exact dollar amounts. "
        "Output JSON only."
    )


def _query_refresh_user_prompt(record: dict[str, Any]) -> str:
    payload = {
        "task": "refresh_query_text",
        "prompt_version": QUERY_REFRESH_PROMPT_VERSION,
        "record": {
            "row_id": str(record.get("row_id") or ""),
            "source_dataset": str(record.get("source_dataset") or ""),
            "asset_context": str(record.get("asset_context") or ""),
            "hard_example": bool(record.get("hard_example", False)),
            "expected_issues": task_schema.issue_list_payload(
                task_schema.normalize_issue_proposals({"proposals": list(record.get("expected_proposals") or [])})
            ),
        },
        "return_schema": {
            "asset_context": "Short visual context sentence.",
            "reasoning_text": "Optional concise dataset note.",
            "issues": [{"type": "<issue_code>", "reasoning": "<short visible evidence>"}],
        },
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _normalize_query_refresh_bundle(record: dict[str, Any], payload: Any) -> dict[str, Any]:
    fallback = _template_query_bundle(record)
    if not isinstance(payload, dict):
        return fallback
    issue_by_code = {
        str(item.get("type") or item.get("issue_code") or "").strip(): {
            "type": str(item.get("type") or item.get("issue_code") or "").strip(),
            "reasoning": str(item.get("reasoning") or item.get("evidence") or "").strip(),
        }
        for item in list(payload.get("issues") or payload.get("proposals") or [])
        if str(item.get("type") or item.get("issue_code") or "").strip()
    }
    normalized_issues: list[dict[str, Any]] = []
    for proposal in list(record.get("expected_proposals") or []):
        issue_code = str(proposal.get("issue_code") or "").strip()
        override = issue_by_code.get(issue_code, {})
        normalized_issues.append(
            {
                "type": issue_code,
                "reasoning": str(override.get("reasoning") or proposal.get("evidence") or "").strip(),
            }
        )
    return {
        "inspection_request": str(fallback["inspection_request"]).strip(),
        "asset_context": str(payload.get("asset_context") or fallback["asset_context"]).strip(),
        "reasoning_text": str(payload.get("reasoning_text") or "").strip(),
        "issues": normalized_issues,
    }


def _refresh_single_query_bundle(
    *,
    record: dict[str, Any],
    teacher: openrouter_grader.OpenRouterGrader,
) -> tuple[dict[str, Any], dict[str, Any], float]:
    payload, raw_payload, latency_ms = teacher.chat_json(
        system_prompt=_query_refresh_system_prompt(),
        user_prompt=_query_refresh_user_prompt(record),
    )
    return _normalize_query_refresh_bundle(record, payload), raw_payload, latency_ms


def _refresh_single_query_bundle_with_retry(
    *,
    record: dict[str, Any],
    teacher: openrouter_grader.OpenRouterGrader,
    retries: int,
    backoff_s: float,
) -> tuple[dict[str, Any], dict[str, Any], float]:
    max_attempts = max(1, int(retries) + 1)
    last_error: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return _refresh_single_query_bundle(record=record, teacher=teacher)
        except Exception as exc:  # pragma: no cover - exercised via retry-capable integration paths
            last_error = exc
            if attempt + 1 >= max_attempts:
                break
            time.sleep(float(backoff_s) * float(2**attempt))
    if last_error is not None:
        raise last_error
    raise RuntimeError("query refresh retry loop exited without a result")


def _build_query_refresh_bundles(
    *,
    records: list[dict[str, Any]],
    args: argparse.Namespace,
    call_openrouter_fn: Optional[Callable[..., tuple[str, dict[str, Any], float]]] = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    mode = str(args.query_text_refresh_mode or "template_only").strip().lower()
    if mode not in {"template_only", "openrouter"}:
        raise ValueError(f"Unsupported query_text_refresh_mode={args.query_text_refresh_mode!r}")
    selected_records = _select_query_refresh_records(
        records,
        num_shards=int(args.query_refresh_num_shards),
        shard_index=int(args.query_refresh_shard_index),
    )
    if mode == "template_only":
        bundles = {str(record.get("row_id") or ""): _template_query_bundle(record) for record in selected_records}
        return bundles, {
            "query_text_refresh_mode": mode,
            "query_teacher_model_id": "",
            "query_text_cache_jsonl": str(args.query_text_cache_jsonl),
            "refreshed_record_count": 0,
            "used_cache_count": 0,
            "selected_record_count": len(selected_records),
            "total_record_count": len(records),
            "query_refresh_num_shards": int(args.query_refresh_num_shards),
            "query_refresh_shard_index": int(args.query_refresh_shard_index),
            "query_refresh_max_concurrency": int(args.query_refresh_max_concurrency),
            "query_refresh_retries": int(args.query_refresh_retries),
            "query_refresh_retry_backoff_s": float(args.query_refresh_retry_backoff_s),
            "query_refresh_cache_flush_every": int(args.query_refresh_cache_flush_every),
            "query_refresh_only": bool(args.query_refresh_only),
        }

    teacher_model_id = str(args.query_teacher_model_id or "").strip()
    if not teacher_model_id:
        raise ValueError("--query-teacher-model-id is required when query_text_refresh_mode=openrouter")
    api_key = openrouter_grader.resolve_openrouter_api_key(
        explicit_api_key=str(args.query_teacher_api_key or ""),
        api_key_env_var=str(args.query_teacher_api_key_env_var or openrouter_grader.DEFAULT_OPENROUTER_ENV_VAR),
    )
    teacher = openrouter_grader.OpenRouterGrader(
        api_key=api_key,
        model_id=teacher_model_id,
        api_base=str(args.query_teacher_api_base or openrouter_grader.DEFAULT_OPENROUTER_API_BASE),
        timeout=float(args.query_teacher_timeout),
        call_api_fn=call_openrouter_fn,
    )
    cache_path = Path(args.query_text_cache_jsonl)
    cache = _load_query_text_cache(cache_path)
    used_cache_count = 0
    refreshed_count = 0
    bundles: dict[str, dict[str, Any]] = {}
    pending_cache_writes = 0
    progress = _make_progress_bar(
        total=len(selected_records),
        desc="query text refresh" if mode == "openrouter" else "query text",
        enabled=bool(args.progress),
    )
    refresh_jobs: list[tuple[dict[str, Any], str, str]] = []
    for record in selected_records:
        row_id = str(record.get("row_id") or "")
        cache_key = _query_cache_key_for_record(record, model_id=teacher_model_id)
        cached = cache.get(cache_key)
        if cached is not None:
            used_cache_count += 1
            bundles[row_id] = _normalize_query_refresh_bundle(record, cached.get("bundle"))
            progress.set_postfix(cache_hits=used_cache_count, refreshed=refreshed_count)
            progress.update(1)
            continue
        refresh_jobs.append((record, row_id, cache_key))
    def _flush_cache(*, force: bool = False) -> None:
        nonlocal pending_cache_writes
        if (not force) and pending_cache_writes < int(args.query_refresh_cache_flush_every):
            return
        _write_query_text_cache(cache_path, cache)
        pending_cache_writes = 0
    max_workers = max(1, min(int(args.query_refresh_max_concurrency), len(refresh_jobs) or 1))
    try:
        if max_workers == 1:
            for record, row_id, cache_key in refresh_jobs:
                bundle, raw_payload, latency_ms = _refresh_single_query_bundle_with_retry(
                    record=record,
                    teacher=teacher,
                    retries=int(args.query_refresh_retries),
                    backoff_s=float(args.query_refresh_retry_backoff_s),
                )
                bundles[row_id] = bundle
                cache[cache_key] = {
                    "cache_key": cache_key,
                    "row_id": row_id,
                    "model_id": teacher_model_id,
                    "prompt_version": QUERY_REFRESH_PROMPT_VERSION,
                    "bundle": bundle,
                    "raw_payload": raw_payload,
                    "latency_ms": latency_ms,
                }
                refreshed_count += 1
                pending_cache_writes += 1
                _flush_cache()
                progress.set_postfix(cache_hits=used_cache_count, refreshed=refreshed_count)
                progress.update(1)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_meta = {
                    executor.submit(
                        _refresh_single_query_bundle_with_retry,
                        record=record,
                        teacher=teacher,
                        retries=int(args.query_refresh_retries),
                        backoff_s=float(args.query_refresh_retry_backoff_s),
                    ): (row_id, cache_key)
                    for record, row_id, cache_key in refresh_jobs
                }
                for future in concurrent.futures.as_completed(future_to_meta):
                    row_id, cache_key = future_to_meta[future]
                    bundle, raw_payload, latency_ms = future.result()
                    bundles[row_id] = bundle
                    cache[cache_key] = {
                        "cache_key": cache_key,
                        "row_id": row_id,
                        "model_id": teacher_model_id,
                        "prompt_version": QUERY_REFRESH_PROMPT_VERSION,
                        "bundle": bundle,
                        "raw_payload": raw_payload,
                        "latency_ms": latency_ms,
                    }
                    refreshed_count += 1
                    pending_cache_writes += 1
                    _flush_cache()
                    progress.set_postfix(cache_hits=used_cache_count, refreshed=refreshed_count)
                    progress.update(1)
    finally:
        progress.close()
        _flush_cache(force=True)
    return bundles, {
        "query_text_refresh_mode": mode,
        "query_teacher_model_id": teacher_model_id,
        "query_text_cache_jsonl": str(args.query_text_cache_jsonl),
        "refreshed_record_count": refreshed_count,
        "used_cache_count": used_cache_count,
        "selected_record_count": len(selected_records),
        "total_record_count": len(records),
        "query_refresh_num_shards": int(args.query_refresh_num_shards),
        "query_refresh_shard_index": int(args.query_refresh_shard_index),
        "query_refresh_max_concurrency": int(args.query_refresh_max_concurrency),
        "query_refresh_retries": int(args.query_refresh_retries),
        "query_refresh_retry_backoff_s": float(args.query_refresh_retry_backoff_s),
        "query_refresh_cache_flush_every": int(args.query_refresh_cache_flush_every),
        "query_refresh_only": bool(args.query_refresh_only),
    }


def build_dataset(
    args: argparse.Namespace,
    *,
    call_openrouter_fn: Optional[Callable[..., tuple[str, dict[str, Any], float]]] = None,
) -> dict[str, Any]:
    stage_total = 3 if bool(args.query_refresh_only) else 7
    stage_progress = _make_progress_bar(total=stage_total, desc="build dataset", enabled=bool(args.progress))
    records = _load_records(args)
    stage_progress.set_postfix(stage="split records")
    stage_progress.update(1)
    split_records = _split_records(records, seed=args.seed, val_fraction=args.val_fraction, test_fraction=args.test_fraction)
    stage_progress.set_postfix(stage="refresh query text")
    stage_progress.update(1)
    query_bundles, refresh_summary = _build_query_refresh_bundles(
        records=records,
        args=args,
        call_openrouter_fn=call_openrouter_fn,
    )
    for row_id, bundle in query_bundles.items():
        bundle["query_text_refresh_mode"] = refresh_summary["query_text_refresh_mode"]
    if bool(args.query_refresh_only):
        stage_progress.set_postfix(stage="write refresh summary")
        stage_progress.update(1)
        summary_name = "query_refresh_summary.json"
        if int(args.query_refresh_num_shards) > 1:
            summary_name = (
                f"query_refresh_summary.shard_{int(args.query_refresh_shard_index):03d}"
                f"_of_{int(args.query_refresh_num_shards):03d}.json"
            )
        summary = {
            "source_manifest": args.source_manifest or "<seed_records>",
            "query_text_refresh_mode": refresh_summary["query_text_refresh_mode"],
            "query_teacher_model_id": refresh_summary["query_teacher_model_id"],
            "query_text_cache_jsonl": refresh_summary["query_text_cache_jsonl"],
            "query_refreshed_record_count": refresh_summary["refreshed_record_count"],
            "query_cache_hits": refresh_summary["used_cache_count"],
            "selected_record_count": refresh_summary["selected_record_count"],
            "total_record_count": refresh_summary["total_record_count"],
            "query_refresh_num_shards": refresh_summary["query_refresh_num_shards"],
            "query_refresh_shard_index": refresh_summary["query_refresh_shard_index"],
            "query_refresh_max_concurrency": refresh_summary["query_refresh_max_concurrency"],
            "query_refresh_retries": refresh_summary["query_refresh_retries"],
            "query_refresh_retry_backoff_s": refresh_summary["query_refresh_retry_backoff_s"],
            "query_refresh_cache_flush_every": refresh_summary["query_refresh_cache_flush_every"],
            "query_refresh_only": True,
        }
        Path(args.output_root).mkdir(parents=True, exist_ok=True)
        common.write_json(Path(args.output_root) / summary_name, summary)
        stage_progress.close()
        return summary
    stage_progress.set_postfix(stage="detect rows")
    stage_progress.update(1)
    excluded_source_datasets = {str(item) for item in list(args.detect_exclude_source_datasets or []) if str(item)}
    detect_split_rows = {
        split_name: _detect_rows(
            rows,
            excluded_source_datasets=excluded_source_datasets,
            show_progress=bool(args.progress),
            progress_desc=f"detect rows [{split_name}]",
        )
        for split_name, rows in split_records.items()
    }
    stage_progress.set_postfix(stage="save detect/point")
    stage_progress.update(1)
    detect_dict = DatasetDict({split_name: _make_dataset(rows) for split_name, rows in detect_split_rows.items() if rows})
    point_dict = DatasetDict({split_name: _make_dataset(rows) for split_name, rows in detect_split_rows.items() if rows})
    Path(args.detect_output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.point_output_dir).mkdir(parents=True, exist_ok=True)
    num_shards = _num_shards_by_split(detect_split_rows)
    detect_dict.save_to_disk(str(args.detect_output_dir), num_shards=num_shards)
    point_dict.save_to_disk(str(args.point_output_dir), num_shards=num_shards)

    stage_progress.set_postfix(stage="issue rows")
    stage_progress.update(1)
    query_rows = {
        split_name: _query_issue_rows(
            rows,
            query_bundles=query_bundles,
            show_progress=bool(args.progress),
            progress_desc=f"issue rows [{split_name}]",
        )
        for split_name, rows in split_records.items()
    }

    stage_progress.set_postfix(stage="write query jsonl")
    stage_progress.update(1)
    Path(args.query_output_dir).mkdir(parents=True, exist_ok=True)
    for split_name, rows in query_rows.items():
        common.write_jsonl(Path(args.query_output_dir) / "jsonl" / f"{split_name}.jsonl", rows)

    stage_progress.set_postfix(stage="write metadata")
    stage_progress.update(1)
    issue_box_counts = _issue_box_counts(records)
    missing_issue_codes = [code for code, count in issue_box_counts.items() if int(count) <= 0]
    query_spatial_refs_nonempty_by_split, query_spatial_refs_nonempty_count = _query_spatial_ref_summary(query_rows)
    metadata = {
        "source_manifest": args.source_manifest or "<seed_records>",
        "split_counts": {split_name: len(rows) for split_name, rows in split_records.items()},
        "detect_split_counts": {split_name: len(rows) for split_name, rows in detect_split_rows.items()},
        "point_split_counts": {split_name: len(rows) for split_name, rows in detect_split_rows.items()},
        "query_split_counts": {split_name: len(rows) for split_name, rows in query_rows.items()},
        "detect_exclude_source_datasets": sorted(excluded_source_datasets),
        "detect_class_catalog": ontology.detect_class_catalog(),
        "issue_codes": ontology.all_issue_codes(),
        "issue_box_counts": issue_box_counts,
        "missing_issue_codes": missing_issue_codes,
        "query_text_refresh_mode": refresh_summary["query_text_refresh_mode"],
        "query_teacher_model_id": refresh_summary["query_teacher_model_id"],
        "query_text_cache_jsonl": refresh_summary["query_text_cache_jsonl"],
        "query_target_format": str(args.query_sft_format),
        "query_prompt_styles": {"issues": "request_only"},
        "query_spatial_refs_nonempty_by_split": query_spatial_refs_nonempty_by_split,
        "query_spatial_refs_nonempty_count": query_spatial_refs_nonempty_count,
        "query_refreshed_record_count": refresh_summary["refreshed_record_count"],
        "query_cache_hits": refresh_summary["used_cache_count"],
        "selected_record_count": refresh_summary["selected_record_count"],
        "total_record_count": refresh_summary["total_record_count"],
        "query_refresh_num_shards": refresh_summary["query_refresh_num_shards"],
        "query_refresh_shard_index": refresh_summary["query_refresh_shard_index"],
        "query_refresh_max_concurrency": refresh_summary["query_refresh_max_concurrency"],
        "query_refresh_retries": refresh_summary["query_refresh_retries"],
        "query_refresh_retry_backoff_s": refresh_summary["query_refresh_retry_backoff_s"],
        "query_refresh_cache_flush_every": refresh_summary["query_refresh_cache_flush_every"],
        "query_refresh_only": refresh_summary["query_refresh_only"],
    }
    for output_dir in (
        Path(args.detect_output_dir),
        Path(args.point_output_dir),
        Path(args.query_output_dir),
    ):
        common.write_json(output_dir / "metadata.json", metadata)
    common.write_json(Path(args.output_root) / "source_manifest.normalized.json", records)
    summary = {
        "metadata": metadata,
        "detect_output_dir": str(args.detect_output_dir),
        "point_output_dir": str(args.point_output_dir),
        "query_output_dir": str(args.query_output_dir),
    }
    common.write_json(Path(args.output_root) / "build_summary.json", summary)
    stage_progress.set_postfix(stage="done")
    stage_progress.close()
    return summary


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    common.maybe_load_env_file(args.env_file, override=False)
    summary = build_dataset(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
