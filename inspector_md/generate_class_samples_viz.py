#!/usr/bin/env python3
"""Render representative per-class sample visualizations for Inspector MD."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import textwrap
from pathlib import Path
from typing import Any, Optional

from PIL import Image, ImageDraw, ImageFont

from inspector_md import common, ontology

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = common.repo_relative("configs", "generate_class_samples_viz_default.json")

PALETTE = (
    (220, 38, 38),
    (37, 99, 235),
    (22, 163, 74),
    (217, 119, 6),
    (147, 51, 234),
    (8, 145, 178),
    (190, 24, 93),
    (75, 85, 99),
    (234, 88, 12),
    (5, 150, 105),
)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    raw_argv = list(argv) if argv is not None else list(os.sys.argv[1:])
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    pre_args, _ = pre_parser.parse_known_args(raw_argv)
    config_path = common.resolve_config_path(pre_args.config, script_dir=SCRIPT_DIR)
    config = common.load_json_config(config_path, default_path=DEFAULT_CONFIG_PATH)

    parser = argparse.ArgumentParser(description="Generate per-class annotated sample visualizations.")
    parser.add_argument("--config", default=str(config_path))
    parser.add_argument(
        "--source-manifest",
        default=str(common.repo_relative("dataset", "merged_synth_v1", "synthetic_manifest.json")),
    )
    parser.add_argument(
        "--output-dir",
        default=str(common.repo_relative("outputs", "class_samples_viz_merged_synth_v1")),
    )
    parser.add_argument("--samples-per-class", type=int, default=6)
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--tile-width", type=int, default=480)
    parser.add_argument("--tile-height", type=int, default=420)
    parser.add_argument("--margin", type=int, default=16)
    parser.add_argument("--max-evidence-chars", type=int, default=120)
    parser.add_argument("--issue-codes", nargs="*", default=[])

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
    args.source_manifest = str(common.resolve_path(args.source_manifest, module_root=SCRIPT_DIR))
    args.output_dir = common.resolve_path(args.output_dir, module_root=SCRIPT_DIR)
    args.issue_codes = [ontology.normalize_issue_code(item) for item in list(args.issue_codes or [])]
    if int(args.samples_per_class) <= 0:
        raise ValueError("--samples-per-class must be > 0")
    return args


def _slug(value: Any) -> str:
    text = "".join(ch if ch.isalnum() else "_" for ch in str(value or "").strip().lower())
    text = "_".join(part for part in text.split("_") if part)
    return text or "sample"


def _stable_key(value: Any) -> str:
    return hashlib.sha1(str(value).encode("utf-8")).hexdigest()


def _issue_color(issue_code: str) -> tuple[int, int, int]:
    index = int(hashlib.sha1(issue_code.encode("utf-8")).hexdigest()[:8], 16) % len(PALETTE)
    return PALETTE[index]


def _load_font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "Arial.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("source manifest must be a JSON array")
    return [item for item in payload if isinstance(item, dict)]


def _extract_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row in rows:
        findings = [item for item in list(row.get("expected_findings") or []) if isinstance(item, dict)]
        if not findings:
            continue
        for finding in findings:
            issue_code = ontology.normalize_issue_code(finding.get("issue_code"))
            candidates.append(
                {
                    "issue_code": issue_code,
                    "row": row,
                    "finding": finding,
                    "row_id": str(row.get("row_id") or ""),
                    "image_path": str(row.get("image_path") or ""),
                    "source_dataset": str(row.get("source_dataset") or row.get("source_metadata", {}).get("source_dataset") or "unknown"),
                    "split": str(row.get("split") or ""),
                    "spatial_ref_index": int(finding.get("spatial_ref_index", 0)),
                }
            )
    return candidates


def _select_samples_for_issue(issue_candidates: list[dict[str, Any]], *, samples_per_class: int) -> list[dict[str, Any]]:
    ordered = sorted(
        issue_candidates,
        key=lambda item: (
            _stable_key(f"{item['issue_code']}::{item['row_id']}::{item['spatial_ref_index']}"),
            item["row_id"],
        ),
    )
    selected: list[dict[str, Any]] = []
    selected_row_ids: set[str] = set()
    seen_sources: set[str] = set()
    seen_splits: set[str] = set()

    def try_append(candidate: dict[str, Any], *, require_new_source: bool = False, require_new_split: bool = False) -> bool:
        if candidate["row_id"] in selected_row_ids:
            return False
        if require_new_source and candidate["source_dataset"] in seen_sources:
            return False
        if require_new_split and candidate["split"] in seen_splits:
            return False
        selected.append(candidate)
        selected_row_ids.add(candidate["row_id"])
        seen_sources.add(candidate["source_dataset"])
        seen_splits.add(candidate["split"])
        return True

    for candidate in ordered:
        if len(selected) >= samples_per_class:
            break
        try_append(candidate, require_new_source=True)
    for candidate in ordered:
        if len(selected) >= samples_per_class:
            break
        try_append(candidate, require_new_split=True)
    for candidate in ordered:
        if len(selected) >= samples_per_class:
            break
        try_append(candidate)
    return selected


def _fit_image(image: Image.Image, *, max_width: int, max_height: int) -> tuple[Image.Image, float]:
    scale = min(max_width / float(image.width), max_height / float(image.height))
    scale = min(1.0, max(scale, 0.01))
    new_size = (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale))))
    return image.resize(new_size, Image.Resampling.LANCZOS), scale


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    text = str(text or "").strip()
    if not text:
        return []
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join(current + [word])
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if bbox[2] - bbox[0] <= max_width or not current:
            current.append(word)
            continue
        lines.append(" ".join(current))
        current = [word]
    if current:
        lines.append(" ".join(current))
    if not lines:
        for piece in textwrap.wrap(text, width=24):
            lines.append(piece)
    return lines


def _draw_labeled_box(
    draw: ImageDraw.ImageDraw,
    box_px: tuple[float, float, float, float],
    *,
    label: str,
    color: tuple[int, int, int],
    font: ImageFont.ImageFont,
    width: int,
) -> None:
    x_min, y_min, x_max, y_max = box_px
    draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=width)
    if not label:
        return
    padding = 4
    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    label_x = max(0, x_min)
    label_y = max(0, y_min - text_height - (padding * 2) - 2)
    draw.rectangle(
        [label_x, label_y, label_x + text_width + (padding * 2), label_y + text_height + (padding * 2)],
        fill=color,
    )
    draw.text((label_x + padding, label_y + padding), label, fill=(255, 255, 255), font=font)


def _render_sample_tile(
    sample: dict[str, Any],
    *,
    tile_width: int,
    tile_height: int,
    margin: int,
    max_evidence_chars: int,
    title_font: ImageFont.ImageFont,
    body_font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
) -> Image.Image:
    header_height = 68
    footer_height = 86
    image_area_height = tile_height - header_height - footer_height - (margin * 2)
    image_area_width = tile_width - (margin * 2)
    canvas = Image.new("RGB", (tile_width, tile_height), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    issue_code = sample["issue_code"]
    issue = ontology.get_issue(issue_code)
    color = _issue_color(issue_code)

    with Image.open(Path(sample["image_path"])) as image:
        image = image.convert("RGB")
        fitted, scale = _fit_image(image, max_width=image_area_width, max_height=image_area_height)

    image_x = (tile_width - fitted.width) // 2
    image_y = header_height + margin + max(0, (image_area_height - fitted.height) // 2)
    canvas.paste(fitted, (image_x, image_y))

    overlay = ImageDraw.Draw(canvas)
    all_findings = [item for item in list(sample["row"].get("expected_findings") or []) if isinstance(item, dict)]
    for finding in all_findings:
        box = finding.get("box") or {}
        box_px = (
            image_x + float(box["x_min"]) * fitted.width,
            image_y + float(box["y_min"]) * fitted.height,
            image_x + float(box["x_max"]) * fitted.width,
            image_y + float(box["y_max"]) * fitted.height,
        )
        is_target = int(finding.get("spatial_ref_index", 0)) == int(sample["spatial_ref_index"]) and ontology.normalize_issue_code(
            finding.get("issue_code")
        ) == issue_code
        line_color = color if is_target else (160, 160, 160)
        line_width = 4 if is_target else 2
        label = issue.title if is_target else ""
        _draw_labeled_box(overlay, box_px, label=label, color=line_color, font=small_font, width=line_width)

    draw.rectangle([0, 0, tile_width, header_height], fill=(255, 255, 255))
    draw.line([(0, header_height), (tile_width, header_height)], fill=(225, 225, 225), width=1)
    draw.text((margin, 10), issue.title, fill=color, font=title_font)
    draw.text(
        (margin, 36),
        f"{sample['source_dataset']} | {sample['split']} | {sample['row_id']}",
        fill=(70, 70, 70),
        font=small_font,
    )

    footer_top = tile_height - footer_height
    draw.rectangle([0, footer_top, tile_width, tile_height], fill=(255, 255, 255))
    draw.line([(0, footer_top), (tile_width, footer_top)], fill=(225, 225, 225), width=1)
    evidence_list = list(sample["finding"].get("evidence") or [])
    evidence = common.truncate(evidence_list[0] if evidence_list else "", limit=max_evidence_chars)
    detail_text = f"Detect: {', '.join(sample['finding'].get('source_detect_labels') or [])}"
    evidence_lines = _wrap_text(draw, evidence, body_font, tile_width - (margin * 2))
    y = footer_top + 8
    for line in evidence_lines[:2]:
        draw.text((margin, y), line, fill=(40, 40, 40), font=body_font)
        y += 18
    draw.text((margin, tile_height - 26), common.truncate(detail_text, limit=80), fill=(85, 85, 85), font=small_font)
    draw.rectangle([0, 0, tile_width - 1, tile_height - 1], outline=(220, 220, 220), width=1)
    return canvas


def _make_contact_sheet(tiles: list[Image.Image], *, columns: int, background: tuple[int, int, int] = (245, 245, 245)) -> Image.Image:
    if not tiles:
        raise ValueError("At least one tile is required.")
    tile_width, tile_height = tiles[0].size
    columns = max(1, int(columns))
    rows = int(math.ceil(len(tiles) / float(columns)))
    sheet = Image.new("RGB", (tile_width * columns, tile_height * rows), background)
    for index, tile in enumerate(tiles):
        x = (index % columns) * tile_width
        y = (index // columns) * tile_height
        sheet.paste(tile, (x, y))
    return sheet


def generate_visualizations(args: argparse.Namespace) -> dict[str, Any]:
    manifest_rows = _load_manifest(Path(args.source_manifest))
    candidates = _extract_candidates(manifest_rows)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate["issue_code"], []).append(candidate)

    issue_codes = list(args.issue_codes or sorted(grouped))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    title_font = _load_font(20)
    body_font = _load_font(14)
    small_font = _load_font(12)

    class_summaries: dict[str, Any] = {}
    overview_tiles: list[Image.Image] = []
    overview_issue_codes: list[str] = []

    for issue_code in issue_codes:
        issue_candidates = list(grouped.get(issue_code) or [])
        if not issue_candidates:
            continue
        selected = _select_samples_for_issue(issue_candidates, samples_per_class=int(args.samples_per_class))
        class_dir = output_dir / issue_code
        class_dir.mkdir(parents=True, exist_ok=True)
        tile_paths: list[str] = []
        sample_rows: list[dict[str, Any]] = []
        tiles: list[Image.Image] = []
        for index, sample in enumerate(selected, start=1):
            tile = _render_sample_tile(
                sample,
                tile_width=int(args.tile_width),
                tile_height=int(args.tile_height),
                margin=int(args.margin),
                max_evidence_chars=int(args.max_evidence_chars),
                title_font=title_font,
                body_font=body_font,
                small_font=small_font,
            )
            tile_name = f"{index:02d}_{_slug(sample['row_id'])}.png"
            tile_path = class_dir / tile_name
            tile.save(tile_path)
            tile_paths.append(str(tile_path))
            sample_rows.append(
                {
                    "row_id": sample["row_id"],
                    "source_dataset": sample["source_dataset"],
                    "split": sample["split"],
                    "spatial_ref_index": sample["spatial_ref_index"],
                    "image_path": sample["image_path"],
                    "tile_path": str(tile_path),
                }
            )
            tiles.append(tile)
        contact_sheet = _make_contact_sheet(tiles, columns=int(args.columns))
        sheet_path = class_dir / "contact_sheet.png"
        contact_sheet.save(sheet_path)
        if tiles:
            overview_tiles.append(tiles[0])
            overview_issue_codes.append(issue_code)
        class_summaries[issue_code] = {
            "title": ontology.get_issue(issue_code).title,
            "candidate_count": len(issue_candidates),
            "selected_count": len(selected),
            "contact_sheet_path": str(sheet_path),
            "tile_paths": tile_paths,
            "samples": sample_rows,
        }

    summary = {
        "source_manifest": str(Path(args.source_manifest).resolve()),
        "output_dir": str(output_dir.resolve()),
        "issue_codes": issue_codes,
        "rendered_issue_codes": sorted(class_summaries),
        "samples_per_class": int(args.samples_per_class),
        "class_summaries": class_summaries,
    }
    if overview_tiles:
        overview_sheet = _make_contact_sheet(overview_tiles, columns=int(args.columns))
        overview_path = output_dir / "overview.png"
        overview_sheet.save(overview_path)
        summary["overview_path"] = str(overview_path)
        summary["overview_issue_codes"] = overview_issue_codes
    common.write_json(output_dir / "summary.json", summary)
    return summary


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    summary = generate_visualizations(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
