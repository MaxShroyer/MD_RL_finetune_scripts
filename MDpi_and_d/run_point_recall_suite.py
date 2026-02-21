#!/usr/bin/env python3
"""Run and compare PI&D point-training experiments for recall stabilization."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class Experiment:
    name: str
    extra_args: list[str]


@dataclass
class RunResult:
    name: str
    cmd: list[str]
    exit_code: int
    run_dir: Optional[Path]
    finetune_id: Optional[str]
    baseline_eval_f1: Optional[float]
    best_eval_f1: Optional[float]
    final_eval_f1: Optional[float]
    recall_gate_pass: Optional[int]
    f1_target_pass: Optional[int]

    @property
    def best_delta_vs_baseline(self) -> Optional[float]:
        if self.best_eval_f1 is None or self.baseline_eval_f1 is None:
            return None
        return self.best_eval_f1 - self.baseline_eval_f1


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _latest_new_run(wandb_dir: Path, before: set[Path]) -> Optional[Path]:
    current = {p.resolve() for p in wandb_dir.glob("run-*")}
    new_runs = list(current - before)
    candidates = new_runs if new_runs else list(current)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _read_run_result(name: str, cmd: list[str], exit_code: int, run_dir: Optional[Path]) -> RunResult:
    summary: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    finetune_id: Optional[str] = None

    if run_dir is not None:
        files_dir = run_dir / "files"
        summary_path = files_dir / "wandb-summary.json"
        metadata_path = files_dir / "wandb-metadata.json"
        config_path = files_dir / "config.yaml"
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                summary = {}
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                metadata = {}
        if config_path.exists():
            text = config_path.read_text(encoding="utf-8", errors="replace")
            marker = "finetune_id:\n    value: "
            idx = text.find(marker)
            if idx >= 0:
                start = idx + len(marker)
                end = text.find("\n", start)
                finetune_id = text[start:end].strip() if end > start else None

    if not finetune_id:
        finetune_id = summary.get("finetune_id") or metadata.get("finetune_id")

    return RunResult(
        name=name,
        cmd=cmd,
        exit_code=exit_code,
        run_dir=run_dir,
        finetune_id=finetune_id,
        baseline_eval_f1=_coerce_float(summary.get("baseline_eval_f1")),
        best_eval_f1=_coerce_float(summary.get("best_eval_f1")) or _coerce_float(summary.get("eval_f1")),
        final_eval_f1=_coerce_float(summary.get("eval_f1")),
        recall_gate_pass=_coerce_int(summary.get("recall_gate_pass")),
        f1_target_pass=_coerce_int(summary.get("f1_target_pass")),
    )


def _build_experiments(include_prompt_ab: bool) -> list[Experiment]:
    exps = [
        Experiment(name="control", extra_args=[]),
        Experiment(name="recall_primary_detect_phrase", extra_args=["--use-recall-first-preset"]),
        Experiment(
            name="recall_offpolicy_detect_phrase",
            extra_args=["--use-recall-first-preset", "--off-policy"],
        ),
    ]
    if include_prompt_ab:
        exps.extend(
            [
                Experiment(
                    name="recall_primary_class_name",
                    extra_args=["--use-recall-first-preset", "--point-prompt-style", "class_name"],
                ),
                Experiment(
                    name="recall_offpolicy_class_name",
                    extra_args=[
                        "--use-recall-first-preset",
                        "--off-policy",
                        "--point-prompt-style",
                        "class_name",
                    ],
                ),
            ]
        )
    return exps


def _score(result: RunResult) -> float:
    if result.best_eval_f1 is None:
        return -1.0
    return float(result.best_eval_f1)


def _pick_winner(results: list[RunResult]) -> Optional[RunResult]:
    gated = [
        r
        for r in results
        if r.exit_code == 0 and r.recall_gate_pass == 1 and r.f1_target_pass == 1 and r.best_eval_f1 is not None
    ]
    if gated:
        return max(gated, key=_score)

    successful = [r for r in results if r.exit_code == 0 and r.best_eval_f1 is not None]
    if successful:
        return max(successful, key=_score)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PI&D point recall stabilization experiment suite.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--train-script", default="train_pid_icons.py")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--dataset-path", default="outputs/pandid_dataset_v2")
    parser.add_argument("--split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--neg-prompts-per-empty", type=int, default=2)
    parser.add_argument("--neg-prompts-per-nonempty", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--eval-max-samples", type=int, default=400)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--rollout-retries", type=int, default=2)
    parser.add_argument("--rollout-retry-backoff-s", type=float, default=1.0)
    parser.add_argument("--recall-gate-step", type=int, default=40)
    parser.add_argument("--recall-drop-threshold", type=float, default=0.25)
    parser.add_argument("--f1-improvement-target", type=float, default=0.01)
    parser.add_argument("--wandb-project", default="moondream-pid-icons-rl")
    parser.add_argument("--run-prefix", default="pid-point-recall")
    parser.add_argument("--suite-output", default="")
    parser.add_argument("--include-prompt-ab", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    script_path = Path(args.train_script).resolve()
    if not script_path.exists():
        raise FileNotFoundError(f"train script not found: {script_path}")

    workdir = script_path.parent
    wandb_dir = workdir / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    experiments = _build_experiments(include_prompt_ab=args.include_prompt_ab)

    base_args = [
        "--env-file",
        args.env_file,
        "--dataset-path",
        args.dataset_path,
        "--split",
        args.split,
        "--val-split",
        args.val_split,
        "--num-steps",
        str(args.num_steps),
        "--batch-size",
        str(args.batch_size),
        "--group-size",
        str(args.group_size),
        "--neg-prompts-per-empty",
        str(args.neg_prompts_per_empty),
        "--neg-prompts-per-nonempty",
        str(args.neg_prompts_per_nonempty),
        "--eval-every",
        str(args.eval_every),
        "--save-every",
        str(args.save_every),
        "--eval-max-samples",
        str(args.eval_max_samples),
        "--max-workers",
        str(args.max_workers),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--rollout-retries",
        str(args.rollout_retries),
        "--rollout-retry-backoff-s",
        str(args.rollout_retry_backoff_s),
        "--recall-gate-step",
        str(args.recall_gate_step),
        "--recall-drop-threshold",
        str(args.recall_drop_threshold),
        "--f1-improvement-target",
        str(args.f1_improvement_target),
        "--skill",
        "point",
        "--wandb-project",
        args.wandb_project,
    ]

    results: list[RunResult] = []
    for exp in experiments:
        run_name = f"{args.run_prefix}-{exp.name}-{timestamp}"
        cmd = [args.python, str(script_path), *base_args, "--wandb-run-name", run_name, *exp.extra_args]
        if args.extra_args:
            passthrough = args.extra_args[1:] if args.extra_args[0] == "--" else args.extra_args
            cmd.extend(passthrough)

        print(f"\n=== running: {exp.name} ===")
        print(" ".join(cmd))
        if args.dry_run:
            continue

        before = {p.resolve() for p in wandb_dir.glob("run-*")}
        proc = subprocess.run(cmd, cwd=str(workdir), check=False)
        run_dir = _latest_new_run(wandb_dir=wandb_dir, before=before)
        result = _read_run_result(name=exp.name, cmd=cmd, exit_code=proc.returncode, run_dir=run_dir)
        results.append(result)
        print(
            f"{exp.name}: exit={result.exit_code} baseline_f1={result.baseline_eval_f1} "
            f"best_f1={result.best_eval_f1} delta={result.best_delta_vs_baseline} "
            f"recall_gate_pass={result.recall_gate_pass} f1_target_pass={result.f1_target_pass} "
            f"finetune_id={result.finetune_id}"
        )

    if args.dry_run:
        return

    winner = _pick_winner(results)
    print("\n=== suite summary ===")
    for r in results:
        print(
            f"{r.name}: exit={r.exit_code} baseline_f1={r.baseline_eval_f1} best_f1={r.best_eval_f1} "
            f"delta={r.best_delta_vs_baseline} recall_gate_pass={r.recall_gate_pass} "
            f"f1_target_pass={r.f1_target_pass} finetune_id={r.finetune_id}"
        )
    if winner is not None:
        print(
            f"winner={winner.name} best_f1={winner.best_eval_f1} delta={winner.best_delta_vs_baseline} "
            f"finetune_id={winner.finetune_id}"
        )
    else:
        print("winner=none")

    output_path = (
        Path(args.suite_output).expanduser().resolve()
        if args.suite_output
        else (workdir / "outputs" / f"point_recall_suite_{timestamp}.json")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": timestamp,
        "winner": winner.name if winner is not None else None,
        "results": [
            {
                "name": r.name,
                "exit_code": r.exit_code,
                "run_dir": str(r.run_dir) if r.run_dir else None,
                "finetune_id": r.finetune_id,
                "baseline_eval_f1": r.baseline_eval_f1,
                "best_eval_f1": r.best_eval_f1,
                "final_eval_f1": r.final_eval_f1,
                "best_delta_vs_baseline": r.best_delta_vs_baseline,
                "recall_gate_pass": r.recall_gate_pass,
                "f1_target_pass": r.f1_target_pass,
                "cmd": r.cmd,
            }
            for r in results
        ],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote suite report: {output_path}")


if __name__ == "__main__":
    main()
