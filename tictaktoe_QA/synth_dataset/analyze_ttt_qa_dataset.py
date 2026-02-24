#!/usr/bin/env python3
"""Analyze TicTacToe QA dataset distributions and quality signals."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for ln, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{ln}: {exc}") from exc
    return rows


def _load_rows_by_split(jsonl_dir: Path) -> dict[str, list[dict[str, Any]]]:
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(jsonl_dir.glob("*.jsonl")):
        rows_by_split[path.stem] = _load_jsonl(path)
    if not rows_by_split:
        raise ValueError(f"No .jsonl files found in {jsonl_dir}")
    return rows_by_split


def _parse_final_answer(row: dict[str, Any]) -> dict[str, Any] | None:
    raw = row.get("final_answer_json")
    if not isinstance(raw, str):
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _distribution_for_task(rows: list[dict[str, Any]], task_type: str) -> dict[str, int]:
    cnt: Counter[str] = Counter()
    for row in rows:
        if row.get("task_type") != task_type:
            continue
        ans = _parse_final_answer(row)
        if ans is None:
            cnt["<invalid_json>"] += 1
            continue

        if task_type == "winner":
            cnt[str(ans.get("winner"))] += 1
        elif task_type == "is_terminal":
            cnt[str(bool(ans.get("is_terminal")))] += 1
        elif task_type == "has_winning_move":
            cnt[str(bool(ans.get("has_winning_move")))] += 1
        elif task_type == "turn_player":
            cnt[str(ans.get("player"))] += 1
        elif task_type == "best_move":
            row_v = ans.get("row")
            col_v = ans.get("col")
            cnt[f"({row_v},{col_v})"] += 1
        elif task_type == "legal_moves_count":
            cnt[str(ans.get("legal_move_count"))] += 1
        elif task_type == "legal_moves_list":
            moves = ans.get("legal_moves")
            if isinstance(moves, list):
                cnt[f"len={len(moves)}"] += 1
            else:
                cnt["len=<invalid>"] += 1
        else:
            cnt["<unknown_task>"] += 1
    return dict(sorted(cnt.items(), key=lambda kv: kv[0]))


def _missing_expected_labels(dist: dict[str, int], expected: set[str]) -> list[str]:
    present = set(dist.keys())
    return sorted(expected - present)


def _summarize_rows(rows_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    splits = sorted(rows_by_split)

    split_sizes = {split: len(rows) for split, rows in rows_by_split.items()}
    task_counts_by_split = {
        split: dict(sorted(Counter(row.get("task_type", "") for row in rows).items()))
        for split, rows in rows_by_split.items()
    }

    label_dists: dict[str, dict[str, dict[str, int]]] = {}
    for split, rows in rows_by_split.items():
        label_dists[split] = {
            "winner": _distribution_for_task(rows, "winner"),
            "is_terminal": _distribution_for_task(rows, "is_terminal"),
            "has_winning_move": _distribution_for_task(rows, "has_winning_move"),
            "turn_player": _distribution_for_task(rows, "turn_player"),
            "best_move": _distribution_for_task(rows, "best_move"),
            "legal_moves_count": _distribution_for_task(rows, "legal_moves_count"),
            "legal_moves_list": _distribution_for_task(rows, "legal_moves_list"),
        }

    expected = {
        "winner": {"X", "O", "draw", "in_progress"},
        "is_terminal": {"True", "False"},
        "has_winning_move": {"True", "False"},
        "turn_player": {"X", "O"},
    }

    coverage_gaps: dict[str, dict[str, list[str]]] = {}
    warnings: list[str] = []

    for split in splits:
        coverage_gaps[split] = {}
        for task, exp in expected.items():
            missing = _missing_expected_labels(label_dists[split].get(task, {}), exp)
            if missing:
                coverage_gaps[split][task] = missing

                # Warn if degenerate labels (<=1 present) on main splits, or all in_progress/False on benchmark.
                present_count = len(label_dists[split].get(task, {}))
                if split in {"train", "val", "test"} and present_count <= 1:
                    warnings.append(
                        f"split={split} task={task} appears degenerate; present={sorted(label_dists[split].get(task, {}).keys())}"
                    )
                if split.startswith("benchmark_top50_") and task in {"winner", "is_terminal", "has_winning_move"}:
                    warnings.append(
                        f"benchmark split={split} task={task} missing labels={missing}; present={sorted(label_dists[split].get(task, {}).keys())}"
                    )

    # Symmetry leakage across train/val/test
    train_groups = {row.get("symmetry_group", "") for row in rows_by_split.get("train", [])}
    val_groups = {row.get("symmetry_group", "") for row in rows_by_split.get("val", [])}
    test_groups = {row.get("symmetry_group", "") for row in rows_by_split.get("test", [])}
    overlaps = {
        "train_val": len(train_groups & val_groups),
        "train_test": len(train_groups & test_groups),
        "val_test": len(val_groups & test_groups),
    }

    if overlaps["train_val"] or overlaps["train_test"] or overlaps["val_test"]:
        warnings.append(f"symmetry leakage detected: {overlaps}")

    # Image dedup/reuse and existence
    image_rows: list[dict[str, Any]] = [row for rows in rows_by_split.values() for row in rows]
    image_usage = Counter(str(row.get("image_path", "")) for row in image_rows)
    image_paths = [Path(p) for p in image_usage if p]
    missing_images = [str(p) for p in image_paths if not p.exists()]

    reuse_values = list(image_usage.values())
    image_reuse = {
        "unique_images": len(image_usage),
        "rows_per_image_min": min(reuse_values) if reuse_values else 0,
        "rows_per_image_max": max(reuse_values) if reuse_values else 0,
        "rows_per_image_mean": round(mean(reuse_values), 4) if reuse_values else 0.0,
        "missing_image_count": len(missing_images),
        "missing_images_sample": sorted(missing_images)[:10],
    }
    if missing_images:
        warnings.append(f"missing image files detected: count={len(missing_images)}")

    # State reuse summary
    state_usage = Counter(str(row.get("state_key", "")) for row in image_rows)
    state_vals = list(state_usage.values())
    state_reuse = {
        "unique_states": len(state_usage),
        "rows_per_state_min": min(state_vals) if state_vals else 0,
        "rows_per_state_max": max(state_vals) if state_vals else 0,
        "rows_per_state_mean": round(mean(state_vals), 4) if state_vals else 0.0,
    }

    # Explicit vs implicit move prompts
    move_rows = [r for r in image_rows if r.get("task_type") in {"best_move", "has_winning_move"}]
    explicit = sum(1 for r in move_rows if ":explicit" in str(r.get("prompt_variant_id", "")))
    implicit = sum(1 for r in move_rows if ":implicit" in str(r.get("prompt_variant_id", "")))
    move_prompt_mix = {
        "move_rows": len(move_rows),
        "explicit_rows": explicit,
        "implicit_rows": implicit,
        "explicit_ratio": round(explicit / max(1, explicit + implicit), 4),
    }

    # Basic parseability checks
    parse_fail_final = 0
    parse_fail_messages = 0
    for row in image_rows:
        if _parse_final_answer(row) is None:
            parse_fail_final += 1
        try:
            _ = json.loads(str(row.get("messages_json", "")))
        except json.JSONDecodeError:
            parse_fail_messages += 1

    parse_health = {
        "final_answer_json_parse_fail": parse_fail_final,
        "messages_json_parse_fail": parse_fail_messages,
    }
    if parse_fail_final or parse_fail_messages:
        warnings.append(
            f"parse failures detected: final={parse_fail_final}, messages={parse_fail_messages}"
        )

    return {
        "split_sizes": split_sizes,
        "task_counts_by_split": task_counts_by_split,
        "label_distributions": label_dists,
        "label_coverage_gaps": coverage_gaps,
        "symmetry_overlap_counts": overlaps,
        "image_reuse": image_reuse,
        "state_reuse": state_reuse,
        "move_prompt_mix": move_prompt_mix,
        "parse_health": parse_health,
        "warnings": warnings,
    }


def _print_report(report: dict[str, Any]) -> None:
    print("Dataset Analysis")
    print("split sizes:")
    for split, count in sorted(report["split_sizes"].items()):
        print(f"  {split}: {count}")

    print("task counts by split:")
    for split, counts in sorted(report["task_counts_by_split"].items()):
        print(f"  {split}: {counts}")

    print("symmetry overlaps (train/val/test):", report["symmetry_overlap_counts"])
    print("image reuse:", report["image_reuse"])
    print("state reuse:", report["state_reuse"])
    print("move prompt mix:", report["move_prompt_mix"])
    print("parse health:", report["parse_health"])

    gaps = report["label_coverage_gaps"]
    print("label coverage gaps:")
    for split, task_missing in sorted(gaps.items()):
        if not task_missing:
            print(f"  {split}: none")
        else:
            print(f"  {split}: {task_missing}")

    warnings = report["warnings"]
    if warnings:
        print("warnings:")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("warnings: none")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze TicTacToe QA dataset")
    parser.add_argument(
        "--dataset-dir",
        default=str(Path(__file__).resolve().parent / "outputs" / "v1"),
        help="Dataset directory that contains jsonl/*.jsonl",
    )
    parser.add_argument(
        "--out-json",
        default="",
        help="Optional output JSON report path. Defaults to <dataset-dir>/analysis_report.json",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    jsonl_dir = dataset_dir / "jsonl"

    rows_by_split = _load_rows_by_split(jsonl_dir)
    report = _summarize_rows(rows_by_split)

    out_json = Path(args.out_json).expanduser().resolve() if args.out_json else dataset_dir / "analysis_report.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    _print_report(report)
    print(f"analysis report written: {out_json}")


if __name__ == "__main__":
    main()
