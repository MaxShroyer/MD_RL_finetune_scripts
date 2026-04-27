#!/usr/bin/env python3
"""Launch or materialize DisasterM3 RL-only sweep commands."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from disaster_m3 import common  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST_PATH = common.repo_relative("outputs", "disaster_m3_rl_sweep_manifest.json")
DEFAULT_KEY_SLOT_ENV_VARS = [
    "CICID_GPUB_MOONDREAM_API_KEY_1",
    "CICID_GPUB_MOONDREAM_API_KEY_1",
    "CICID_GPUB_MOONDREAM_API_KEY_2",
    "CICID_GPUB_MOONDREAM_API_KEY_2",
    "CICID_GPUB_MOONDREAM_API_KEY_3",
    "CICID_GPUB_MOONDREAM_API_KEY_3",
    "CICID_GPUB_MOONDREAM_API_KEY_4",
    "CICID_GPUB_MOONDREAM_API_KEY_4",
]


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch DisasterM3 RL-only sweep runs.")
    parser.add_argument("--env-file", default=str(common.repo_relative(".env.staging")))
    parser.add_argument("--dataset-dir", default=str(common.DEFAULT_OUTPUT_DIR))
    parser.add_argument("--sweep", choices=("off_policy", "reasoning", "both"), default="both")
    parser.add_argument("--training-regime", choices=("rl_only", "warmup_200", "both"), default="rl_only")
    parser.add_argument("--ranks", nargs="+", type=int, default=[32, 24, 16])
    parser.add_argument("--lrs", nargs="+", type=float, default=[2e-4, 5e-4, 5e-5, 1e-5])
    parser.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--run-output-dir", default=str(common.repo_relative("outputs", "runs")))
    parser.add_argument("--max-parallel", type=int, default=8)
    parser.add_argument("--key-slot-env-vars", nargs="+", default=list(DEFAULT_KEY_SLOT_ENV_VARS))
    parser.add_argument("--launcher-log-dir", default=str(common.repo_relative("outputs", "sweep_launcher_logs", "disaster_m3")))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    args.env_file = str(common.resolve_path(args.env_file, repo_root=REPO_ROOT, module_root=SCRIPT_DIR))
    args.dataset_dir = common.resolve_path(args.dataset_dir, repo_root=REPO_ROOT, module_root=SCRIPT_DIR)
    args.manifest_path = common.resolve_path(args.manifest_path, repo_root=REPO_ROOT, module_root=SCRIPT_DIR)
    args.run_output_dir = common.resolve_path(args.run_output_dir, repo_root=REPO_ROOT, module_root=SCRIPT_DIR)
    args.launcher_log_dir = common.resolve_path(args.launcher_log_dir, repo_root=REPO_ROOT, module_root=SCRIPT_DIR)
    return args


def _lr_slug(lr: float) -> str:
    text = f"{lr:.0e}".replace("+0", "").replace("+", "")
    return text.replace("-", "m").replace(".", "p")


def build_sweep_runs(args: argparse.Namespace) -> list[dict[str, Any]]:
    sweep_modes = ["off_policy", "reasoning"] if args.sweep == "both" else [str(args.sweep)]
    regime_modes = ["rl_only", "warmup_200"] if args.training_regime == "both" else [str(args.training_regime)]
    config_by_regime_and_mode = {
        ("rl_only", "off_policy"): SCRIPT_DIR / "configs" / "train_disaster_m3_mixed_offpolicy_base.json",
        ("rl_only", "reasoning"): SCRIPT_DIR / "configs" / "train_disaster_m3_mixed_reasoning_base.json",
        ("warmup_200", "off_policy"): SCRIPT_DIR / "configs" / "train_disaster_m3_mixed_warmup200_offpolicy_base.json",
        ("warmup_200", "reasoning"): SCRIPT_DIR / "configs" / "train_disaster_m3_mixed_warmup200_reasoning_base.json",
    }
    runs: list[dict[str, Any]] = []
    for regime in regime_modes:
        for sweep_mode in sweep_modes:
            config_path = config_by_regime_and_mode[(regime, sweep_mode)]
            for rank in list(args.ranks):
                for lr in list(args.lrs):
                    regime_prefix = "warmup200" if regime == "warmup_200" else "rlonly"
                    name = f"disaster-m3-{regime_prefix}-{sweep_mode}-r{int(rank)}-lr{_lr_slug(float(lr))}"
                    command = [
                        sys.executable,
                        str((SCRIPT_DIR / "train_disaster_m3_mixed.py").resolve()),
                        "--config",
                        str(config_path),
                        "--env-file",
                        str(args.env_file),
                        "--dataset-dir",
                        str(args.dataset_dir),
                        "--run-output-dir",
                        str(args.run_output_dir),
                        "--api-key-env-var",
                        "__KEY_SLOT_ENV_VAR__",
                        "--rank",
                        str(int(rank)),
                        "--rl-lr",
                        str(float(lr)),
                        "--finetune-name",
                        name,
                    ]
                    runs.append(
                        {
                            "training_regime": regime,
                            "sweep_mode": sweep_mode,
                            "rank": int(rank),
                            "rl_lr": float(lr),
                            "finetune_name": name,
                            "command": command,
                        }
                    )
    return runs


def _validate_key_slots(args: argparse.Namespace) -> list[str]:
    key_slot_env_vars = [str(item).strip() for item in list(args.key_slot_env_vars) if str(item).strip()]
    if not key_slot_env_vars:
        raise ValueError("--key-slot-env-vars must provide at least one env var name")
    if int(args.max_parallel) <= 0:
        raise ValueError("--max-parallel must be > 0")
    if int(args.max_parallel) > len(key_slot_env_vars):
        raise ValueError("--max-parallel cannot exceed the number of configured key slots")
    common.maybe_load_env_file(str(args.env_file))
    missing = [name for name in key_slot_env_vars if not str(os.environ.get(name) or "").strip()]
    if missing:
        raise ValueError(f"Missing key env var(s): {sorted(set(missing))}")
    return key_slot_env_vars[: int(args.max_parallel)]


def _slot_log_path(*, log_dir: Path, finetune_name: str) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"{finetune_name}.log"


def _launch_parallel_runs(args: argparse.Namespace, runs: list[dict[str, Any]], key_slots: list[str]) -> None:
    pending = [dict(run) for run in runs]
    active: list[dict[str, Any]] = []
    completed = 0
    failure: Optional[tuple[str, int, Path]] = None

    def start_next(slot_index: int, key_env_var: str) -> None:
        nonlocal pending, active
        if not pending:
            return
        run = pending.pop(0)
        command = [key_env_var if part == "__KEY_SLOT_ENV_VAR__" else part for part in list(run["command"])]
        log_path = _slot_log_path(log_dir=Path(args.launcher_log_dir), finetune_name=str(run["finetune_name"]))
        env = dict(os.environ)
        env["MOONDREAM_SWEEP_SLOT_KEY_ENV_VAR"] = str(key_env_var)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "finetune_name": run["finetune_name"],
                        "slot_index": int(slot_index),
                        "key_env_var": key_env_var,
                        "command": command,
                        "started_at": time.time(),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        log_handle = log_path.open("a", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        active.append(
            {
                "slot_index": int(slot_index),
                "key_env_var": key_env_var,
                "run": run,
                "command": command,
                "log_path": log_path,
                "log_handle": log_handle,
                "process": process,
                "started_at": time.time(),
            }
        )
        print(
            f"launched slot={slot_index + 1}/{len(key_slots)} key={key_env_var} "
            f"run={run['finetune_name']} remaining={len(pending)} log={log_path}"
        )

    for slot_index, key_env_var in enumerate(key_slots):
        start_next(slot_index, key_env_var)

    while active:
        time.sleep(2.0)
        next_active: list[dict[str, Any]] = []
        finished: list[dict[str, Any]] = []
        for item in active:
            process = item["process"]
            returncode = process.poll()
            if returncode is None:
                next_active.append(item)
                continue
            item["returncode"] = int(returncode)
            item["finished_at"] = time.time()
            item["log_handle"].close()
            finished.append(item)
        active = next_active
        for item in finished:
            completed += 1
            run = item["run"]
            if int(item["returncode"]) != 0 and failure is None:
                failure = (str(run["finetune_name"]), int(item["returncode"]), Path(item["log_path"]))
            print(
                f"completed {completed}/{len(runs)} slot={item['slot_index'] + 1} "
                f"key={item['key_env_var']} run={run['finetune_name']} rc={item['returncode']}"
            )
            if failure is None:
                start_next(int(item["slot_index"]), str(item["key_env_var"]))
        if failure is not None:
            break

    if failure is not None:
        failed_name, returncode, log_path = failure
        for item in active:
            item["process"].terminate()
            item["log_handle"].close()
        raise subprocess.CalledProcessError(returncode=returncode, cmd=failed_name, output=str(log_path))


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    key_slots = _validate_key_slots(args)
    runs = build_sweep_runs(args)
    manifest = {
        "dataset_dir": str(args.dataset_dir),
        "env_file": str(args.env_file),
        "run_output_dir": str(args.run_output_dir),
        "sweep": str(args.sweep),
        "training_regime": str(args.training_regime),
        "max_parallel": int(args.max_parallel),
        "key_slot_env_vars": list(key_slots),
        "run_count": len(runs),
        "runs": runs,
    }
    common.write_json(Path(args.manifest_path), manifest)
    print(f"saved sweep manifest: {args.manifest_path} runs={len(runs)}")
    if args.dry_run:
        for index, run in enumerate(runs):
            slot_index = index % len(key_slots)
            key_env_var = key_slots[slot_index]
            command = [key_env_var if part == "__KEY_SLOT_ENV_VAR__" else part for part in run["command"]]
            print(f"# slot={slot_index + 1} key={key_env_var}")
            print(" ".join(command))
        return
    _launch_parallel_runs(args, runs, key_slots)


if __name__ == "__main__":
    main()
