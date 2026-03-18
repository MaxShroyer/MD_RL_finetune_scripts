#!/usr/bin/env python3
"""Run a small deterministic LR sweep for best-move-only RL configs."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

DEFAULT_BASE_CONFIG = (
    Path(__file__).resolve().parent / "configs" / "query_rl_best_move_only.json"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "best_move_lr_sweep"
DEFAULT_LRS = (1e-4, 2e-4, 5e-4)


def _parse_lrs(raw_value: str) -> list[float]:
    out: list[float] = []
    for piece in str(raw_value).split(","):
        text = piece.strip().lower()
        if not text:
            continue
        out.append(float(text))
    if not out:
        raise ValueError("No valid LR values provided")
    return out


def _lr_tag(lr_value: float) -> str:
    text = f"{float(lr_value):.8f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def _build_run_command(
    *,
    base_config: Path,
    lr_value: float,
    seed: int,
    group_size: int,
    ranking_output: Path,
    benchmark_output_json: Path,
    benchmark_predictions_jsonl: Path,
    num_steps: Optional[int],
    no_progress: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        str((Path(__file__).resolve().parent / "train_ttt_query_rl.py").resolve()),
        "--config",
        str(base_config),
        "--seed",
        str(int(seed)),
        "--lr",
        str(float(lr_value)),
        "--group-size",
        str(int(group_size)),
        "--checkpoint-ranking-output",
        str(ranking_output),
        "--auto-benchmark-output-json",
        str(benchmark_output_json),
        "--auto-benchmark-predictions-jsonl",
        str(benchmark_predictions_jsonl),
    ]
    if num_steps is not None and int(num_steps) > 0:
        cmd.extend(["--num-steps", str(int(num_steps))])
    if no_progress:
        cmd.append("--no-progress")
    return cmd


def _load_json_if_exists(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _collect_run_result(
    *,
    lr_value: float,
    trial_dir: Path,
    return_code: int,
) -> dict[str, Any]:
    ranking_path = trial_dir / "checkpoint_ranking.json"
    benchmark_path = trial_dir / "benchmark_auto.json"
    ranking = _load_json_if_exists(ranking_path) or {}
    benchmark = _load_json_if_exists(benchmark_path) or {}

    return {
        "lr": float(lr_value),
        "trial_dir": str(trial_dir),
        "status": "completed" if int(return_code) == 0 else "failed",
        "return_code": int(return_code),
        "best_avg_checkpoint_metric": float(ranking.get("best_avg_checkpoint_metric", 0.0)),
        "best_avg_checkpoint_metric_step": int(ranking.get("best_avg_checkpoint_metric_step", -1)),
        "auto_eval_best_move_set_accuracy": float(benchmark.get("eval_best_move_set_accuracy", 0.0)),
        "auto_eval_json_parse_rate": float(benchmark.get("eval_json_parse_rate", 0.0)),
        "ranking_file": str(ranking_path),
        "benchmark_file": str(benchmark_path),
    }


def _rank_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def _key(item: dict[str, Any]) -> tuple[float, float, float]:
        if str(item.get("status")) != "completed":
            return (float("-inf"), float("-inf"), float("-inf"))
        return (
            float(item.get("best_avg_checkpoint_metric", 0.0)),
            float(item.get("auto_eval_best_move_set_accuracy", 0.0)),
            float(item.get("auto_eval_json_parse_rate", 0.0)),
        )

    return sorted(results, key=_key, reverse=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a 3-point LR sweep for best-move-only RL.")
    parser.add_argument("--base-config", default=str(DEFAULT_BASE_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group-size", type=int, default=6)
    parser.add_argument("--lrs", default="1e-4,2e-4,5e-4")
    parser.add_argument("--num-steps", type=int, default=0, help="<=0 keeps config num_steps.")
    parser.add_argument("--no-progress", action="store_true", default=False)
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    base_config = Path(args.base_config).expanduser().resolve()
    if not base_config.exists():
        raise FileNotFoundError(f"--base-config not found: {base_config}")
    if int(args.group_size) <= 0:
        raise ValueError("--group-size must be > 0")
    lrs = _parse_lrs(args.lrs)

    output_dir = Path(args.output_dir).expanduser().resolve()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_root = output_dir / f"sweep_{timestamp}"
    run_root.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for lr_value in lrs:
        trial_dir = run_root / f"lr_{_lr_tag(lr_value)}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        command = _build_run_command(
            base_config=base_config,
            lr_value=float(lr_value),
            seed=int(args.seed),
            group_size=int(args.group_size),
            ranking_output=trial_dir / "checkpoint_ranking.json",
            benchmark_output_json=trial_dir / "benchmark_auto.json",
            benchmark_predictions_jsonl=trial_dir / "benchmark_auto_predictions.jsonl",
            num_steps=(None if int(args.num_steps) <= 0 else int(args.num_steps)),
            no_progress=bool(args.no_progress),
        )
        print(f"running lr={lr_value}: {' '.join(command)}")
        proc = subprocess.run(command, text=True, capture_output=True)
        (trial_dir / "stdout.log").write_text(proc.stdout, encoding="utf-8")
        (trial_dir / "stderr.log").write_text(proc.stderr, encoding="utf-8")
        if proc.returncode != 0:
            print(f"trial failed lr={lr_value} return_code={proc.returncode}")
        results.append(
            _collect_run_result(
                lr_value=float(lr_value),
                trial_dir=trial_dir,
                return_code=int(proc.returncode),
            )
        )

    ranked = _rank_results(results)
    summary_payload = {
        "base_config": str(base_config),
        "seed": int(args.seed),
        "group_size": int(args.group_size),
        "lrs": [float(v) for v in lrs],
        "run_root": str(run_root),
        "results": ranked,
    }
    summary_json_path = run_root / "summary.json"
    summary_csv_path = run_root / "summary.csv"
    summary_json_path.write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    with summary_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "lr",
                "status",
                "return_code",
                "best_avg_checkpoint_metric",
                "best_avg_checkpoint_metric_step",
                "auto_eval_best_move_set_accuracy",
                "auto_eval_json_parse_rate",
                "trial_dir",
            ],
        )
        writer.writeheader()
        for row in ranked:
            writer.writerow(
                {
                    "lr": row["lr"],
                    "status": row["status"],
                    "return_code": row["return_code"],
                    "best_avg_checkpoint_metric": row["best_avg_checkpoint_metric"],
                    "best_avg_checkpoint_metric_step": row["best_avg_checkpoint_metric_step"],
                    "auto_eval_best_move_set_accuracy": row["auto_eval_best_move_set_accuracy"],
                    "auto_eval_json_parse_rate": row["auto_eval_json_parse_rate"],
                    "trial_dir": row["trial_dir"],
                }
            )

    print(f"wrote summary: {summary_json_path}")
    print(f"wrote summary CSV: {summary_csv_path}")


if __name__ == "__main__":
    main()
