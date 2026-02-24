#!/usr/bin/env python3
"""Validate generated TicTacToe QA dataset artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tictaktoe_QA.synth_dataset.src.exporters import REQUIRED_COLUMNS
from tictaktoe_QA.synth_dataset.src.label_engine import build_board_record, move_to_row_col, row_col_to_move
from tictaktoe_QA.synth_dataset.src.rationale import build_final_answer
from tictaktoe_QA.synth_dataset.src.sampler import MAIN_TASK_QUOTAS
from tictaktoe_QA.synth_dataset.src.state_source import load_cloudwalk_data


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
                raise ValueError(f"invalid JSON at {path}:{ln}: {exc}") from exc
    return rows


def _load_rows_by_split(jsonl_dir: Path) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(jsonl_dir.glob("*.jsonl")):
        rows[path.stem] = _load_jsonl(path)
    if not rows:
        raise ValueError(f"No JSONL files found in {jsonl_dir}")
    return rows


def _check_required_columns(rows_by_split: dict[str, list[dict[str, Any]]]) -> None:
    for split_name, rows in rows_by_split.items():
        for idx, row in enumerate(rows):
            missing = [col for col in REQUIRED_COLUMNS if col not in row]
            if missing:
                raise ValueError(f"split={split_name} row={idx} missing columns: {missing}")


def _check_main_counts(rows_by_split: dict[str, list[dict[str, Any]]], expected_total: int) -> None:
    total = sum(len(rows_by_split.get(split, [])) for split in ("train", "val", "test"))
    if total != expected_total:
        raise ValueError(f"Main split total rows mismatch: expected {expected_total}, got {total}")

    for split_name, expected_task_quota in MAIN_TASK_QUOTAS.items():
        rows = rows_by_split.get(split_name, [])
        obs = Counter(row["task_type"] for row in rows)
        for task_type, expected in expected_task_quota.items():
            got = obs.get(task_type, 0)
            if got != expected:
                raise ValueError(
                    f"Task quota mismatch split={split_name} task={task_type}: expected {expected}, got {got}"
                )


def _check_leakage(rows_by_split: dict[str, list[dict[str, Any]]]) -> None:
    groups = {}
    for split_name in ("train", "val", "test"):
        groups[split_name] = {row["symmetry_group"] for row in rows_by_split.get(split_name, [])}

    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = groups[a] & groups[b]
        if overlap:
            sample = sorted(overlap)[:5]
            raise ValueError(f"symmetry_group leakage between {a} and {b}: {sample}")


def _check_top50_exclusion(rows_by_split: dict[str, list[dict[str, Any]]], top50_keys: set[str]) -> None:
    for split_name in ("train", "val", "test"):
        keys = {row["state_key"] for row in rows_by_split.get(split_name, [])}
        overlap = keys & top50_keys
        if overlap:
            sample = sorted(overlap)[:5]
            raise ValueError(f"top50 leakage into {split_name}: {sample}")


def _as_move_set(payload_json: str) -> set[int]:
    arr = json.loads(payload_json)
    out: set[int] = set()
    if not isinstance(arr, list):
        return out
    for item in arr:
        if not isinstance(item, dict):
            continue
        if "move" in item:
            out.add(int(item["move"]))
        elif "row" in item and "col" in item:
            out.add(row_col_to_move(int(item["row"]), int(item["col"])))
    return out


def _best_move_from_json(payload_json: str) -> int | None:
    payload = json.loads(payload_json)
    if payload is None:
        return None
    if not isinstance(payload, dict):
        return None
    if "move" in payload:
        return int(payload["move"])
    if "row" in payload and "col" in payload:
        return row_col_to_move(int(payload["row"]), int(payload["col"]))
    return None


def _check_label_consistency(rows_by_split: dict[str, list[dict[str, Any]]], source_records: dict[str, Any]) -> None:
    for split_name, rows in rows_by_split.items():
        for idx, row in enumerate(rows):
            state_key = row["state_key"]
            if state_key not in source_records:
                raise ValueError(f"unknown state_key {state_key} in split {split_name}")
            rec = source_records[state_key]

            if row["player_to_move"] != rec.player_to_move:
                raise ValueError(f"player mismatch split={split_name} row={idx}")
            if row["winner_label"] != rec.winner_label:
                raise ValueError(f"winner mismatch split={split_name} row={idx}")
            if bool(row["is_terminal"]) != bool(rec.is_terminal):
                raise ValueError(f"terminal mismatch split={split_name} row={idx}")
            if row["symmetry_group"] != rec.symmetry_group:
                raise ValueError(f"symmetry_group mismatch split={split_name} row={idx}")

            legal_set_row = _as_move_set(row["legal_moves_json"])
            legal_set_gt = set(rec.legal_moves)
            if legal_set_row != legal_set_gt:
                raise ValueError(f"legal moves mismatch split={split_name} row={idx}")

            best_set_row = _as_move_set(row["best_move_optimal_set_json"])
            if best_set_row != set(rec.best_move_optimal_set):
                raise ValueError(f"best_move_optimal_set mismatch split={split_name} row={idx}")

            best_canon = _best_move_from_json(row["best_move_canonical_json"])
            if best_canon != rec.best_move_canonical:
                raise ValueError(f"best_move_canonical mismatch split={split_name} row={idx}")

            expected_final = build_final_answer(row["task_type"], rec)
            observed_final = json.loads(row["final_answer_json"])
            if expected_final != observed_final:
                raise ValueError(f"final_answer mismatch split={split_name} row={idx} task={row['task_type']}")

            _ = json.loads(row["messages_json"])


def _check_images(rows_by_split: dict[str, list[dict[str, Any]]]) -> None:
    signature_by_path: dict[str, tuple[str, str, str]] = {}
    for split_name, rows in rows_by_split.items():
        for idx, row in enumerate(rows):
            path = Path(row["image_path"])
            if not path.exists():
                raise ValueError(f"missing image split={split_name} row={idx}: {path}")
            sig = (row["state_key"], row["colorway"], row["augmentation_profile"])
            prev = signature_by_path.get(str(path))
            if prev is None:
                signature_by_path[str(path)] = sig
            elif prev != sig:
                raise ValueError(f"image dedup conflict for {path}: {prev} vs {sig}")


def _label_coverage_report(rows_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, list[str]]]:
    report: dict[str, dict[str, list[str]]] = {}
    for split_name, rows in rows_by_split.items():
        task_labels: dict[str, set[str]] = {
            "winner": set(),
            "is_terminal": set(),
            "has_winning_move": set(),
            "turn_player": set(),
        }
        for row in rows:
            task = row["task_type"]
            if task not in task_labels:
                continue
            ans = json.loads(row["final_answer_json"])
            if task == "winner":
                task_labels[task].add(str(ans["winner"]))
            elif task == "is_terminal":
                task_labels[task].add(str(bool(ans["is_terminal"])))
            elif task == "has_winning_move":
                task_labels[task].add(str(bool(ans["has_winning_move"])))
            elif task == "turn_player":
                task_labels[task].add(str(ans["player"]))
        report[split_name] = {task: sorted(values) for task, values in task_labels.items()}
    return report


def _score_predictions(rows_by_split: dict[str, list[dict[str, Any]]], predictions_path: Path) -> dict[str, Any]:
    preds: dict[str, dict[str, Any]] = {}
    for row in _load_jsonl(predictions_path):
        rid = str(row.get("row_id", "")).strip()
        if rid:
            preds[rid] = row

    counters = Counter()
    by_task = Counter()

    for rows in rows_by_split.values():
        for row in rows:
            rid = row["row_id"]
            if rid not in preds:
                continue
            pred = preds[rid]
            pred_json_raw = pred.get("final_answer_json") or pred.get("prediction") or pred.get("answer")
            if not isinstance(pred_json_raw, str):
                continue
            counters["evaluated"] += 1
            task = row["task_type"]
            by_task[f"{task}:count"] += 1

            try:
                gt = json.loads(row["final_answer_json"])
                guess = json.loads(pred_json_raw)
            except json.JSONDecodeError:
                counters["json_parse_fail"] += 1
                continue

            if task == "best_move":
                guess_move = None
                if isinstance(guess, dict):
                    if "move" in guess:
                        guess_move = int(guess["move"])
                    elif "row" in guess and "col" in guess:
                        guess_move = row_col_to_move(int(guess["row"]), int(guess["col"]))

                opt_set = _as_move_set(row["best_move_optimal_set_json"])
                canon_move = _best_move_from_json(row["best_move_canonical_json"])

                if guess_move in opt_set:
                    counters["best_move_set_correct"] += 1
                    by_task["best_move:set_correct"] += 1
                if guess_move == canon_move:
                    counters["best_move_canonical_correct"] += 1
                    by_task["best_move:canonical_correct"] += 1
            else:
                if guess == gt:
                    counters["exact_correct"] += 1
                    by_task[f"{task}:exact_correct"] += 1

    result = {
        "evaluated_rows": counters["evaluated"],
        "json_parse_fail": counters["json_parse_fail"],
        "exact_accuracy_non_best_move": (
            counters["exact_correct"] / max(1, counters["evaluated"])
        ),
        "best_move_set_accuracy": (
            counters["best_move_set_correct"] / max(1, by_task["best_move:count"])
        ),
        "best_move_canonical_accuracy": (
            counters["best_move_canonical_correct"] / max(1, by_task["best_move:count"])
        ),
        "by_task": dict(by_task),
    }
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate TicTacToe QA dataset")
    parser.add_argument("--dataset-dir", default=str(Path(__file__).resolve().parent / "outputs" / "v1"))
    parser.add_argument("--cache-dir", default=str(Path(__file__).resolve().parent / "cache" / "cloudwalk"))
    parser.add_argument("--allow-network", action="store_true", default=True)
    parser.add_argument("--no-network", dest="allow_network", action="store_false")
    parser.add_argument("--expected-main-rows", type=int, default=50000)
    parser.add_argument("--skip-main-counts", action="store_true")
    parser.add_argument("--predictions-jsonl", default="")
    parser.add_argument("--strict-label-diversity", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    jsonl_dir = dataset_dir / "jsonl"

    rows_by_split = _load_rows_by_split(jsonl_dir)
    _check_required_columns(rows_by_split)

    if not args.skip_main_counts:
        _check_main_counts(rows_by_split, expected_total=args.expected_main_rows)

    _check_leakage(rows_by_split)

    cloudwalk = load_cloudwalk_data(
        cache_dir=Path(args.cache_dir).expanduser().resolve(),
        allow_network=args.allow_network,
    )
    records = {k: build_board_record(k, p) for k, p in cloudwalk.main_boards.items()}
    top50_keys = set(cloudwalk.top50_boards.keys())

    _check_top50_exclusion(rows_by_split, top50_keys)
    _check_label_consistency(rows_by_split, records)
    _check_images(rows_by_split)
    coverage = _label_coverage_report(rows_by_split)

    print("validation: OK")
    for split_name, rows in sorted(rows_by_split.items()):
        print(f"  {split_name}: {len(rows)} rows")
    print("label_coverage:")
    print(json.dumps(coverage, indent=2, sort_keys=True))

    if args.strict_label_diversity:
        for split_name, task_labels in coverage.items():
            for task_name in ("winner", "is_terminal", "has_winning_move", "turn_player"):
                values = task_labels.get(task_name, [])
                if split_name.startswith("benchmark_top50_"):
                    # benchmark tracks can be intentionally narrow for some tasks.
                    continue
                if task_name in {"winner", "is_terminal", "has_winning_move"} and len(values) <= 1:
                    raise ValueError(
                        f"low label diversity: split={split_name} task={task_name} labels={values}"
                    )

    if args.predictions_jsonl:
        metrics = _score_predictions(rows_by_split, Path(args.predictions_jsonl).expanduser().resolve())
        print("prediction_metrics:")
        print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
