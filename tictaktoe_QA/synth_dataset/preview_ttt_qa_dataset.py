#!/usr/bin/env python3
"""Create quick preview artifacts for TicTacToe QA dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PIL import Image, ImageOps


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _load_rows_by_split(jsonl_dir: Path) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(jsonl_dir.glob("*.jsonl")):
        rows[path.stem] = _load_jsonl(path)
    return rows


def _build_grid(image_paths: list[Path], out_path: Path, *, cols: int = 4, thumb: int = 192) -> None:
    if not image_paths:
        return
    rows = (len(image_paths) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * thumb, rows * thumb), color=(24, 24, 24))

    for i, path in enumerate(image_paths):
        img = Image.open(path).convert("RGB")
        tile = ImageOps.fit(img, (thumb, thumb), method=Image.Resampling.BICUBIC)
        r = i // cols
        c = i % cols
        canvas.paste(tile, (c * thumb, r * thumb))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="PNG")


def _pick_examples(rows: list[dict[str, Any]], task_type: str, n: int) -> list[dict[str, Any]]:
    picks = [row for row in rows if row.get("task_type") == task_type]
    return picks[:n]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview TicTacToe QA dataset")
    parser.add_argument("--dataset-dir", default=str(Path(__file__).resolve().parent / "outputs" / "v2"))
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--samples-per-task", type=int, default=3)
    parser.add_argument("--samples-per-colorway", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else dataset_dir / "preview"

    rows_by_split = _load_rows_by_split(dataset_dir / "jsonl")
    out_dir.mkdir(parents=True, exist_ok=True)

    split_counts = {s: len(rows) for s, rows in rows_by_split.items()}
    task_counts = {s: dict(Counter(row["task_type"] for row in rows)) for s, rows in rows_by_split.items()}

    colorway_to_paths: dict[str, list[Path]] = defaultdict(list)
    for rows in rows_by_split.values():
        for row in rows:
            colorway = row["colorway"]
            path = Path(row["image_path"])
            if path.exists() and path not in colorway_to_paths[colorway]:
                colorway_to_paths[colorway].append(path)

    grid_rel_paths: dict[str, str] = {}
    for colorway, paths in sorted(colorway_to_paths.items()):
        sample_paths = paths[: args.samples_per_colorway]
        grid_path = out_dir / f"grid_{colorway}.png"
        _build_grid(sample_paths, grid_path)
        grid_rel_paths[colorway] = grid_path.name

    summary = {
        "split_counts": split_counts,
        "task_counts": task_counts,
        "colorways": sorted(colorway_to_paths),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    md_lines: list[str] = ["# TicTacToe QA Dataset Preview", "", "## Split Counts"]
    for split_name, count in sorted(split_counts.items()):
        md_lines.append(f"- `{split_name}`: {count}")

    md_lines.extend(["", "## Colorway Grids"])
    for colorway, rel in sorted(grid_rel_paths.items()):
        md_lines.append(f"### `{colorway}`")
        md_lines.append(f"![{colorway}]({rel})")
        md_lines.append("")

    md_lines.extend(["## Example Rows (test split)"])
    test_rows = rows_by_split.get("test", [])
    task_types = sorted({row["task_type"] for row in test_rows})
    for task_type in task_types:
        md_lines.append(f"### `{task_type}`")
        for row in _pick_examples(test_rows, task_type, args.samples_per_task):
            md_lines.append(f"- image: `{row['image_path']}`")
            md_lines.append(f"- question: {row['question']}")
            md_lines.append(f"- answer: {row['answer_text']}")
        md_lines.append("")

    (out_dir / "preview.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(f"preview written to {out_dir}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
