#!/usr/bin/env python3
"""Build synthetic TicTacToe QA dataset for VLM training."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any
import sys

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tictaktoe_QA.synth_dataset.src.exporters import build_stats, write_hf_dataset, write_json, write_jsonl
from tictaktoe_QA.synth_dataset.src.label_engine import BoardRecord, build_board_record, move_to_row_col
from tictaktoe_QA.synth_dataset.src.rationale import OpenAIParaphraser, build_answer_payload
from tictaktoe_QA.synth_dataset.src.renderer import choose_colorway_for_state, render_board_image
from tictaktoe_QA.synth_dataset.src.sampler import (
    MAIN_TOTAL_ROWS,
    RowPlan,
    derive_main_task_quotas,
    sample_dataset_plan,
)
from tictaktoe_QA.synth_dataset.src.state_source import MAIN_URL, TOP50_URL, load_cloudwalk_data
from tictaktoe_QA.synth_dataset.src.templates import choose_prompt

SOURCE_NAME = "cloudwalk/tictactoe-dataset"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_output_dir() -> Path:
    return Path(__file__).resolve().parent / "outputs" / "v1"


def _default_cache_dir() -> Path:
    return Path(__file__).resolve().parent / "cache" / "cloudwalk"


def _default_env_file() -> Path:
    return Path(__file__).resolve().parents[1] / ".env"


def _best_move_canonical_json(record: BoardRecord) -> str:
    if record.best_move_canonical is None:
        return "null"
    row, col = move_to_row_col(record.best_move_canonical)
    payload = {"move": record.best_move_canonical, "row": row, "col": col}
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def _best_move_optimal_set_json(record: BoardRecord) -> str:
    payload = []
    for move in record.best_move_optimal_set:
        row, col = move_to_row_col(move)
        payload.append({"move": move, "row": row, "col": col})
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def _legal_moves_json(record: BoardRecord) -> str:
    payload = []
    for move in record.legal_moves:
        row, col = move_to_row_col(move)
        payload.append({"move": move, "row": row, "col": col})
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def _build_row(
    *,
    row_id: str,
    split: str,
    task_type: str,
    record: BoardRecord,
    image_path: Path,
    colorway: str,
    augmentation_profile: str,
    prompt_question: str,
    prompt_variant_id: str,
    answer_text: str,
    final_answer_json: str,
    rationale_source: str,
) -> dict[str, Any]:
    messages = [
        {"role": "user", "content": prompt_question},
        {"role": "assistant", "content": answer_text},
    ]

    return {
        "row_id": row_id,
        "image": str(image_path),
        "image_path": str(image_path),
        "split": split,
        "task_type": task_type,
        "question": prompt_question,
        "answer_text": answer_text,
        "final_answer_json": final_answer_json,
        "messages_json": json.dumps(messages, separators=(",", ":"), ensure_ascii=False),
        "state_key": record.state_key,
        "symmetry_group": record.symmetry_group,
        "player_to_move": record.player_to_move,
        "winner_label": record.winner_label,
        "is_terminal": bool(record.is_terminal),
        "legal_moves_json": _legal_moves_json(record),
        "best_move_canonical_json": _best_move_canonical_json(record),
        "best_move_optimal_set_json": _best_move_optimal_set_json(record),
        "depth_complexity": int(record.depth_complexity),
        "choice_complexity_num": int(record.choice_complexity_num),
        "choice_complexity_den": int(record.choice_complexity_den),
        "colorway": colorway,
        "augmentation_profile": augmentation_profile,
        "prompt_variant_id": prompt_variant_id,
        "source_name": SOURCE_NAME,
        "rationale_source": rationale_source,
        "scores_by_move_json": record.scores_by_move_json,
    }


def _render_main_images(
    *,
    records: dict[str, BoardRecord],
    state_keys: list[str],
    images_dir: Path,
    seed: int,
) -> dict[str, Path]:
    by_state: dict[str, Path] = {}
    for state_key in state_keys:
        record = records[state_key]
        colorway = choose_colorway_for_state(state_key)
        path = render_board_image(
            record=record,
            colorway=colorway,
            out_dir=images_dir,
            augmentation_profile="low",
            aug_seed=seed,
        )
        by_state[state_key] = path
    return by_state


def _render_benchmark_images(
    *,
    records: dict[str, BoardRecord],
    rows: list[RowPlan],
    images_dir: Path,
) -> dict[tuple[str, str], Path]:
    out: dict[tuple[str, str], Path] = {}
    needed_pairs = sorted({(plan.state_key, str(plan.colorway)) for plan in rows if plan.colorway})
    for state_key, colorway in needed_pairs:
        rec = records[state_key]
        path = render_board_image(
            record=rec,
            colorway=colorway,
            out_dir=images_dir,
            augmentation_profile="none",
            aug_seed=0,
        )
        out[(state_key, colorway)] = path
    return out


def _build_rows(
    *,
    records: dict[str, BoardRecord],
    plans: list[RowPlan],
    split: str,
    seed: int,
    llm_ratio: float,
    llm_enabled: bool,
    paraphraser: OpenAIParaphraser | None,
    main_state_images: dict[str, Path],
    benchmark_images: dict[tuple[str, str], Path],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rng = random.Random(f"rows:{split}:{seed}")

    for idx, plan in enumerate(plans):
        record = records[plan.state_key]

        if plan.benchmark_track is None:
            colorway = choose_colorway_for_state(plan.state_key)
            image_path = main_state_images[plan.state_key]
            augmentation_profile = "low"
        else:
            if plan.colorway is None:
                raise ValueError("benchmark row missing colorway")
            colorway = plan.colorway
            image_path = benchmark_images[(plan.state_key, colorway)]
            augmentation_profile = "none"

        prompt = choose_prompt(
            task_type=plan.task_type,
            record=record,
            rng=rng,
            benchmark_track=plan.benchmark_track,
            explicit_player_override=plan.explicit_player_override,
        )

        answer = build_answer_payload(
            task_type=plan.task_type,
            record=record,
            rng=rng,
            split=split,
            llm_ratio=llm_ratio,
            paraphraser=paraphraser,
            llm_enabled=llm_enabled,
        )

        row = _build_row(
            row_id=f"{split}_{idx:06d}",
            split=split,
            task_type=plan.task_type,
            record=record,
            image_path=image_path,
            colorway=colorway,
            augmentation_profile=augmentation_profile,
            prompt_question=prompt.question,
            prompt_variant_id=prompt.prompt_variant_id,
            answer_text=answer.answer_text,
            final_answer_json=answer.final_answer_json,
            rationale_source=answer.rationale_source,
        )
        rows.append(row)

    return rows


def _validate_main_totals(
    rows_by_split: dict[str, list[dict[str, Any]]],
    *,
    expected_total: int,
    expected_task_quotas: dict[str, dict[str, int]],
) -> None:
    main_splits = ("train", "val", "test")
    total = sum(len(rows_by_split[s]) for s in main_splits)
    if total != expected_total:
        raise ValueError(f"main rows total mismatch: expected {expected_total}, got {total}")

    for split_name, expected_task_counts in expected_task_quotas.items():
        rows = rows_by_split[split_name]
        observed: dict[str, int] = {}
        for row in rows:
            observed[row["task_type"]] = observed.get(row["task_type"], 0) + 1
        for task_type, expected in expected_task_counts.items():
            got = observed.get(task_type, 0)
            if got != expected:
                raise ValueError(
                    f"task quota mismatch split={split_name} task={task_type}: expected {expected}, got {got}"
                )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build TicTacToe synthetic QA dataset")
    parser.add_argument("--output-dir", default=str(_default_output_dir()))
    parser.add_argument("--cache-dir", default=str(_default_cache_dir()))
    parser.add_argument("--env-file", default=str(_default_env_file()))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-states", type=int, default=3000)
    parser.add_argument("--target-rows", type=int, default=MAIN_TOTAL_ROWS)
    parser.add_argument("--allow-network", action="store_true", default=True)
    parser.add_argument("--no-network", dest="allow_network", action="store_false")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--llm-ratio", type=float, default=0.2)
    parser.add_argument("--openai-model", default="gpt-4o-mini")
    parser.add_argument("--max-main-rows", type=int, default=0, help="Optional smoke cap for main rows.")
    parser.add_argument("--max-benchmark-rows", type=int, default=0, help="Optional smoke cap for benchmark rows.")
    parser.add_argument("--skip-jsonl-export", action="store_true")
    parser.add_argument("--skip-hf-export", action="store_true")
    parser.add_argument(
        "--hf-repo-id",
        default="",
        help="Hugging Face dataset repo id, e.g. username/tictactoe-qa-v1. "
        "If omitted, reads HF_DATASET_REPO_ID from env/.env.",
    )
    parser.add_argument(
        "--hf-token",
        default="",
        help="Optional Hugging Face token override. If omitted, uses HF_TOKEN or HUGGINGFACE_HUB_TOKEN.",
    )
    parser.add_argument("--hf-private", action="store_true", help="Push dataset to a private Hugging Face repo.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    load_dotenv(args.env_file, override=False)

    if args.target_rows <= 0:
        raise ValueError("--target-rows must be > 0")
    if args.llm_ratio < 0.0 or args.llm_ratio > 1.0:
        raise ValueError("--llm-ratio must be within [0,1]")
    if args.skip_jsonl_export and args.skip_hf_export:
        raise ValueError("Cannot skip both exports; enable at least one of JSONL or HF export.")

    hf_repo_id = args.hf_repo_id.strip() or os.environ.get("HF_DATASET_REPO_ID", "").strip()
    hf_token = (
        args.hf_token.strip()
        or os.environ.get("HF_TOKEN", "").strip()
        or os.environ.get("HUGGINGFACE_HUB_TOKEN", "").strip()
    )
    if not args.skip_hf_export and not hf_repo_id:
        raise ValueError("HF export is enabled; provide --hf-repo-id or set HF_DATASET_REPO_ID in env/.env.")

    main_task_quotas = derive_main_task_quotas(args.target_rows)

    output_dir = Path(args.output_dir).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    images_dir = output_dir / "images"
    jsonl_dir = output_dir / "jsonl"
    hf_dir = output_dir / "hf_dataset"

    cloudwalk = load_cloudwalk_data(cache_dir=cache_dir, allow_network=args.allow_network)

    records: dict[str, BoardRecord] = {}
    for state_key, payload in cloudwalk.main_boards.items():
        records[state_key] = build_board_record(state_key, payload)

    top50_keys = set(cloudwalk.top50_boards.keys())
    missing_top50 = sorted(k for k in top50_keys if k not in records)
    if missing_top50:
        raise ValueError(f"top50 keys not present in main boards: {missing_top50[:5]}")

    sampling = sample_dataset_plan(
        records,
        top50_keys,
        target_states=args.target_states,
        main_task_quotas=main_task_quotas,
        seed=args.seed,
    )

    llm_enabled = not args.no_llm
    paraphraser = OpenAIParaphraser(model=args.openai_model) if llm_enabled else None

    main_plans_by_split: dict[str, list[RowPlan]] = {"train": [], "val": [], "test": []}
    for plan in sampling.main_rows:
        main_plans_by_split[plan.split].append(plan)

    bench_plans_by_split: dict[str, list[RowPlan]] = {
        "benchmark_top50_canonical": [],
        "benchmark_top50_paraphrase": [],
    }
    for plan in sampling.benchmark_rows:
        bench_plans_by_split[plan.split].append(plan)

    if args.max_main_rows > 0:
        for split_name in main_plans_by_split:
            main_plans_by_split[split_name] = main_plans_by_split[split_name][: args.max_main_rows]
    if args.max_benchmark_rows > 0:
        for split_name in bench_plans_by_split:
            bench_plans_by_split[split_name] = bench_plans_by_split[split_name][: args.max_benchmark_rows]

    selected_main_keys = sorted({plan.state_key for plans in main_plans_by_split.values() for plan in plans})
    main_images = _render_main_images(records=records, state_keys=selected_main_keys, images_dir=images_dir, seed=args.seed)
    benchmark_plan_rows = [plan for plans in bench_plans_by_split.values() for plan in plans]
    bench_images = _render_benchmark_images(records=records, rows=benchmark_plan_rows, images_dir=images_dir)

    rows_by_split: dict[str, list[dict[str, Any]]] = {}

    for split_name, plans in main_plans_by_split.items():
        rows = _build_rows(
            records=records,
            plans=plans,
            split=split_name,
            seed=args.seed,
            llm_ratio=args.llm_ratio,
            llm_enabled=llm_enabled,
            paraphraser=paraphraser,
            main_state_images=main_images,
            benchmark_images=bench_images,
        )
        rows_by_split[split_name] = rows

    for split_name, plans in bench_plans_by_split.items():
        rows = _build_rows(
            records=records,
            plans=plans,
            split=split_name,
            seed=args.seed,
            llm_ratio=0.0,
            llm_enabled=False,
            paraphraser=None,
            main_state_images=main_images,
            benchmark_images=bench_images,
        )
        rows_by_split[split_name] = rows

    if args.max_main_rows <= 0:
        _validate_main_totals(
            rows_by_split,
            expected_total=args.target_rows,
            expected_task_quotas=main_task_quotas,
        )

    if not args.skip_jsonl_export:
        write_jsonl(rows_by_split, jsonl_dir)
    if not args.skip_hf_export:
        try:
            write_hf_dataset(
                rows_by_split,
                hf_dir,
                push_to_hub=True,
                repo_id=hf_repo_id,
                token=hf_token,
                private=bool(args.hf_private),
            )
        except OSError as exc:
            raise OSError(
                f"HF export failed (likely disk space issue): {exc}. "
                "Retry with --skip-hf-export to generate JSONL-only artifacts."
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "HF upload failed. Check your HF token permissions and repo id, "
                "or rerun with --skip-hf-export for local-only artifacts."
            ) from exc

    stats = build_stats(rows_by_split)
    metadata = {
        "seed": args.seed,
        "target_states": args.target_states,
        "target_rows": args.target_rows,
        "allow_network": bool(args.allow_network),
        "llm_enabled": bool(llm_enabled),
        "llm_ratio": args.llm_ratio,
        "openai_model": args.openai_model,
        "source": {
            "name": SOURCE_NAME,
            "main_url": MAIN_URL,
            "top50_url": TOP50_URL,
            "cache_dir": str(cache_dir),
        },
        "selected_main_state_count": len(selected_main_keys),
        "top50_state_count": len(top50_keys),
        "top50_state_keys": sorted(top50_keys),
        "split_state_counts": {k: len(v) for k, v in sampling.split_state_keys.items()},
        "main_task_quotas": main_task_quotas,
        "exports": {
            "jsonl_enabled": not args.skip_jsonl_export,
            "hf_enabled": not args.skip_hf_export,
            "hf_repo_id": hf_repo_id,
            "hf_upload_enabled": not args.skip_hf_export,
        },
    }

    write_json(output_dir / "stats.json", stats)
    write_json(output_dir / "metadata.json", metadata)

    print(f"saved dataset to {output_dir}")
    for split_name, rows in rows_by_split.items():
        print(f"  {split_name}: {len(rows)} rows")
    if not args.skip_hf_export:
        print(f"uploaded hf dataset to {hf_repo_id}")


if __name__ == "__main__":
    main()
