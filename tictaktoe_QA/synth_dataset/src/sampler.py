"""State and QA row sampling for TicTacToe QA dataset."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Iterable

from .label_engine import BoardRecord
from .renderer import COLORWAYS
from .templates import TASK_TYPES

MAIN_TASK_QUOTAS: dict[str, dict[str, int]] = {
    "train": {
        "best_move": 8800,
        "has_winning_move": 5600,
        "turn_player": 4800,
        "winner": 5600,
        "is_terminal": 5600,
        "legal_moves_count": 4800,
        "legal_moves_list": 4800,
    },
    "val": {
        "best_move": 1100,
        "has_winning_move": 700,
        "turn_player": 600,
        "winner": 700,
        "is_terminal": 700,
        "legal_moves_count": 600,
        "legal_moves_list": 600,
    },
    "test": {
        "best_move": 1100,
        "has_winning_move": 700,
        "turn_player": 600,
        "winner": 700,
        "is_terminal": 700,
        "legal_moves_count": 600,
        "legal_moves_list": 600,
    },
}

MAIN_TOTAL_ROWS = sum(sum(task_counts.values()) for task_counts in MAIN_TASK_QUOTAS.values())


@dataclass(frozen=True)
class RowPlan:
    split: str
    task_type: str
    state_key: str
    benchmark_track: str | None
    explicit_player_override: bool | None
    colorway: str | None


@dataclass(frozen=True)
class SamplingOutput:
    selected_main_state_keys: tuple[str, ...]
    split_state_keys: dict[str, tuple[str, ...]]
    main_rows: tuple[RowPlan, ...]
    benchmark_rows: tuple[RowPlan, ...]


def _eligible(record: BoardRecord, task_type: str) -> bool:
    if task_type in {"best_move", "has_winning_move", "turn_player"}:
        return not record.is_terminal
    return True


def _stratified_state_sample(
    records: dict[str, BoardRecord],
    *,
    target_states: int,
    rng: random.Random,
) -> list[str]:
    by_depth: dict[int, list[str]] = defaultdict(list)
    for key, rec in records.items():
        by_depth[rec.depth_complexity].append(key)

    for arr in by_depth.values():
        rng.shuffle(arr)

    bins = sorted(by_depth)
    if not bins:
        return []

    target = min(target_states, sum(len(by_depth[b]) for b in bins))
    quotas = {b: 0 for b in bins}
    base = target // len(bins)

    for b in bins:
        quotas[b] = min(base, len(by_depth[b]))

    assigned = sum(quotas.values())
    remaining = target - assigned

    def priority_order() -> list[int]:
        # Prefer harder bins first, then bins with more remaining capacity.
        return sorted(
            bins,
            key=lambda d: (-d, -(len(by_depth[d]) - quotas[d]), d),
        )

    while remaining > 0:
        progressed = False
        for b in priority_order():
            if quotas[b] < len(by_depth[b]):
                quotas[b] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            break

    selected: list[str] = []
    for b in bins:
        selected.extend(by_depth[b][: quotas[b]])

    rng.shuffle(selected)
    return selected


def _split_groups(
    records: dict[str, BoardRecord],
    selected_keys: Iterable[str],
    *,
    rng: random.Random,
) -> dict[str, set[str]]:
    group_to_keys: dict[str, list[str]] = defaultdict(list)
    group_depth: dict[str, int] = {}

    for key in selected_keys:
        rec = records[key]
        group_to_keys[rec.symmetry_group].append(key)
        group_depth.setdefault(rec.symmetry_group, rec.depth_complexity)

    groups_by_depth: dict[int, list[str]] = defaultdict(list)
    for group, depth in group_depth.items():
        groups_by_depth[depth].append(group)

    split_groups = {"train": set(), "val": set(), "test": set()}

    for depth, groups in sorted(groups_by_depth.items()):
        rng.shuffle(groups)
        n = len(groups)
        n_train = int(round(n * 0.8))
        n_val = int(round(n * 0.1))
        n_test = n - n_train - n_val

        if n >= 3:
            if n_train <= 0:
                n_train = 1
            if n_val <= 0:
                n_val = 1
            if n_test <= 0:
                n_test = 1
            while n_train + n_val + n_test > n:
                if n_train > n_val and n_train > n_test and n_train > 1:
                    n_train -= 1
                elif n_val > 1:
                    n_val -= 1
                elif n_test > 1:
                    n_test -= 1
                else:
                    break

        split_groups["train"].update(groups[:n_train])
        split_groups["val"].update(groups[n_train : n_train + n_val])
        split_groups["test"].update(groups[n_train + n_val :])

    # Ensure every split has at least one group.
    for split_name in ("val", "test"):
        if not split_groups[split_name] and split_groups["train"]:
            moved = sorted(split_groups["train"])[0]
            split_groups["train"].remove(moved)
            split_groups[split_name].add(moved)

    if not split_groups["train"]:
        raise ValueError("No train groups after split.")

    return split_groups


def _state_keys_by_split(records: dict[str, BoardRecord], selected_keys: list[str], split_groups: dict[str, set[str]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for key in selected_keys:
        group = records[key].symmetry_group
        if group in split_groups["train"]:
            out["train"].append(key)
        elif group in split_groups["val"]:
            out["val"].append(key)
        elif group in split_groups["test"]:
            out["test"].append(key)
        else:
            out["train"].append(key)

    for split_name in out:
        out[split_name].sort()
    return out


def derive_main_task_quotas(target_rows: int) -> dict[str, dict[str, int]]:
    if target_rows <= 0:
        raise ValueError("target_rows must be > 0")
    if target_rows == MAIN_TOTAL_ROWS:
        return {split: dict(task_counts) for split, task_counts in MAIN_TASK_QUOTAS.items()}

    cells: list[tuple[str, str, int, float]] = []
    base_total = 0
    for split_name, task_counts in MAIN_TASK_QUOTAS.items():
        for task_type, baseline in task_counts.items():
            raw = (target_rows * baseline) / MAIN_TOTAL_ROWS
            base = int(math.floor(raw))
            cells.append((split_name, task_type, base, raw - base))
            base_total += base

    remainder = target_rows - base_total
    cells.sort(key=lambda item: (-item[3], item[0], item[1]))
    bumped: set[tuple[str, str]] = set()
    for split_name, task_type, _, _ in cells:
        if remainder <= 0:
            break
        bumped.add((split_name, task_type))
        remainder -= 1

    out: dict[str, dict[str, int]] = {split: {} for split in MAIN_TASK_QUOTAS}
    for split_name, task_type, base, _ in cells:
        out[split_name][task_type] = base + (1 if (split_name, task_type) in bumped else 0)
    return out


def _build_main_rows(
    records: dict[str, BoardRecord],
    split_state_keys: dict[str, list[str]],
    *,
    main_task_quotas: dict[str, dict[str, int]],
    seed: int,
) -> list[RowPlan]:
    rows: list[RowPlan] = []
    for split_name, task_quotas in main_task_quotas.items():
        rng = random.Random(f"{seed}:{split_name}:rows")
        for task_type in TASK_TYPES:
            target = task_quotas.get(task_type, 0)
            if target <= 0:
                continue

            eligible = [key for key in split_state_keys[split_name] if _eligible(records[key], task_type)]
            if not eligible:
                raise ValueError(f"No eligible states for split={split_name}, task={task_type}")
            rng.shuffle(eligible)

            explicit_cutoff = int(round(target * 0.7))
            for i in range(target):
                state_key = eligible[i % len(eligible)]
                explicit_override: bool | None = None
                if task_type in {"best_move", "has_winning_move"}:
                    explicit_override = i < explicit_cutoff
                rows.append(
                    RowPlan(
                        split=split_name,
                        task_type=task_type,
                        state_key=state_key,
                        benchmark_track=None,
                        explicit_player_override=explicit_override,
                        colorway=None,
                    )
                )
    return rows


def _build_benchmark_rows(top50_keys: list[str], records: dict[str, BoardRecord]) -> list[RowPlan]:
    def _pick_n(keys: list[str], n: int) -> list[str]:
        if not keys:
            return []
        out: list[str] = []
        for i in range(n):
            out.append(keys[i % len(keys)])
        return out

    top50_sorted = sorted(top50_keys)
    top50_set = set(top50_sorted)

    terminal_x = sorted(
        k for k, r in records.items() if k not in top50_set and r.is_terminal and r.winner_label == "X"
    )
    terminal_o = sorted(
        k for k, r in records.items() if k not in top50_set and r.is_terminal and r.winner_label == "O"
    )
    terminal_draw = sorted(
        k for k, r in records.items() if k not in top50_set and r.is_terminal and r.winner_label == "draw"
    )
    nonterminal_win_true = sorted(
        k for k, r in records.items() if k not in top50_set and (not r.is_terminal) and bool(r.immediate_winning_moves)
    )
    top50_win_false = sorted(k for k in top50_sorted if not records[k].immediate_winning_moves)

    # Keep total benchmark rows per task stable (50 states * 4 colorways = 200 rows).
    benchmark_states_by_task: dict[str, list[str]] = {
        "best_move": top50_sorted,
        "turn_player": top50_sorted,
        "legal_moves_count": top50_sorted,
        "legal_moves_list": top50_sorted,
        # Include probe states so winner/is_terminal are not all in_progress/False.
        "winner": (
            _pick_n(top50_sorted, 20)
            + _pick_n(terminal_x, 15)
            + _pick_n(terminal_o, 10)
            + _pick_n(terminal_draw, 5)
        ),
        "is_terminal": (
            _pick_n(top50_sorted, 20)
            + _pick_n(terminal_x, 15)
            + _pick_n(terminal_o, 10)
            + _pick_n(terminal_draw, 5)
        ),
        # Inject immediate-win positives so this task is not all-false.
        "has_winning_move": _pick_n(top50_win_false or top50_sorted, 25) + _pick_n(nonterminal_win_true, 25),
    }

    rows: list[RowPlan] = []
    for track in ("canonical", "paraphrase"):
        split_name = f"benchmark_top50_{track}"
        for task_type in TASK_TYPES:
            state_keys = benchmark_states_by_task.get(task_type, top50_sorted)
            if not state_keys:
                state_keys = top50_sorted
            for state_key in state_keys:
                rec = records[state_key]
                if not _eligible(rec, task_type):
                    continue
                for colorway in COLORWAYS:
                    rows.append(
                        RowPlan(
                            split=split_name,
                            task_type=task_type,
                            state_key=state_key,
                            benchmark_track=track,
                            explicit_player_override=True,
                            colorway=colorway,
                        )
                    )
    return rows


def sample_dataset_plan(
    records: dict[str, BoardRecord],
    top50_keys: set[str],
    *,
    target_states: int,
    main_task_quotas: dict[str, dict[str, int]] | None = None,
    seed: int,
) -> SamplingOutput:
    rng = random.Random(seed)

    main_pool = {k: v for k, v in records.items() if k not in top50_keys}
    selected_main = _stratified_state_sample(main_pool, target_states=target_states, rng=rng)

    split_groups = _split_groups(main_pool, selected_main, rng=rng)
    split_state_keys = _state_keys_by_split(main_pool, selected_main, split_groups)

    quotas = main_task_quotas or MAIN_TASK_QUOTAS
    main_rows = _build_main_rows(main_pool, split_state_keys, main_task_quotas=quotas, seed=seed)
    benchmark_rows = _build_benchmark_rows(sorted(top50_keys), records)

    return SamplingOutput(
        selected_main_state_keys=tuple(selected_main),
        split_state_keys={k: tuple(v) for k, v in split_state_keys.items()},
        main_rows=tuple(main_rows),
        benchmark_rows=tuple(benchmark_rows),
    )
