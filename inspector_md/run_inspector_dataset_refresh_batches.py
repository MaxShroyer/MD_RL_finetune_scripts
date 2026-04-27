#!/usr/bin/env python3
"""Run sharded query-text refresh jobs in parallel, then finalize the dataset build."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

from inspector_md import common

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = common.repo_relative("configs", "run_inspector_dataset_refresh_batches_default.json")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    raw_argv = list(argv) if argv is not None else list(os.sys.argv[1:])
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    pre_args, _ = pre_parser.parse_known_args(raw_argv)
    config_path = common.resolve_config_path(pre_args.config, script_dir=SCRIPT_DIR)
    config = common.load_json_config(config_path, default_path=DEFAULT_CONFIG_PATH)

    parser = argparse.ArgumentParser(description="Run sharded OpenRouter query-text refresh jobs for Inspector MD.")
    parser.add_argument("--config", default=str(config_path))
    parser.add_argument("--builder-config", default=str(common.repo_relative("configs", "build_inspector_dataset_default.json")))
    parser.add_argument("--output-root", default=str(common.repo_relative("outputs")))
    parser.add_argument("--query-text-cache-jsonl", default=str(common.repo_relative("outputs", "inspector_query_text_cache.jsonl")))
    parser.add_argument(
        "--shard-cache-dir",
        default=str(common.repo_relative("outputs", "inspector_query_text_cache_shards")),
    )
    parser.add_argument(
        "--log-dir",
        default=str(common.repo_relative("outputs", "inspector_query_refresh_logs")),
    )
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--max-parallel-jobs", type=int, default=4)
    parser.add_argument("--per-worker-concurrency", type=int, default=1)
    parser.add_argument("--max-shard-attempts", type=int, default=2)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--finalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")

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
    args.builder_config = str(common.resolve_config_path(args.builder_config, script_dir=SCRIPT_DIR))
    args.output_root = common.resolve_path(args.output_root, module_root=SCRIPT_DIR)
    args.query_text_cache_jsonl = common.resolve_path(args.query_text_cache_jsonl, module_root=SCRIPT_DIR)
    args.shard_cache_dir = common.resolve_path(args.shard_cache_dir, module_root=SCRIPT_DIR)
    args.log_dir = common.resolve_path(args.log_dir, module_root=SCRIPT_DIR)
    if int(args.num_shards) <= 0:
        raise ValueError("--num-shards must be >= 1")
    if int(args.max_parallel_jobs) <= 0:
        raise ValueError("--max-parallel-jobs must be >= 1")
    if int(args.per_worker_concurrency) <= 0:
        raise ValueError("--per-worker-concurrency must be >= 1")
    if int(args.max_shard_attempts) <= 0:
        raise ValueError("--max-shard-attempts must be >= 1")
    return args


def _shard_cache_path(args: argparse.Namespace, shard_index: int) -> Path:
    return Path(args.shard_cache_dir) / f"query_text_cache.shard_{int(shard_index):03d}_of_{int(args.num_shards):03d}.jsonl"


def _shard_summary_path(args: argparse.Namespace, shard_index: int) -> Path:
    return Path(args.output_root) / (
        f"query_refresh_summary.shard_{int(shard_index):03d}_of_{int(args.num_shards):03d}.json"
    )


def _shard_log_path(args: argparse.Namespace, shard_index: int) -> Path:
    return Path(args.log_dir) / f"query_refresh.shard_{int(shard_index):03d}_of_{int(args.num_shards):03d}.log"


def _build_shard_command(args: argparse.Namespace, shard_index: int) -> list[str]:
    command = [
        os.sys.executable,
        "-m",
        "inspector_md.build_inspector_dataset",
        "--config",
        str(args.builder_config),
        "--query-refresh-only",
        "--query-refresh-num-shards",
        str(int(args.num_shards)),
        "--query-refresh-shard-index",
        str(int(shard_index)),
        "--query-refresh-max-concurrency",
        str(int(args.per_worker_concurrency)),
        "--query-text-cache-jsonl",
        str(_shard_cache_path(args, shard_index)),
        "--output-root",
        str(args.output_root),
    ]
    command.append("--progress" if bool(args.progress) else "--no-progress")
    return command


def _shard_is_complete(args: argparse.Namespace, shard_index: int) -> bool:
    summary_path = _shard_summary_path(args, shard_index)
    cache_path = _shard_cache_path(args, shard_index)
    if not summary_path.exists() or not cache_path.exists():
        return False
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(payload.get("query_refresh_only"))


def _merge_caches(args: argparse.Namespace) -> dict[str, int]:
    merged: dict[str, dict[str, object]] = {}
    shard_cache_count = 0
    for shard_index in range(int(args.num_shards)):
        cache_path = _shard_cache_path(args, shard_index)
        if not cache_path.exists():
            continue
        shard_cache_count += 1
        for row in common.load_jsonl(cache_path):
            cache_key = str(row.get("cache_key") or "").strip()
            if not cache_key:
                continue
            merged[cache_key] = dict(row)
    common.write_jsonl(Path(args.query_text_cache_jsonl), [merged[key] for key in sorted(merged)])
    return {"merged_cache_rows": len(merged), "shard_cache_count": shard_cache_count}


def _run_finalize_build(args: argparse.Namespace) -> subprocess.CompletedProcess[str]:
    command = [
        os.sys.executable,
        "-m",
        "inspector_md.build_inspector_dataset",
        "--config",
        str(args.builder_config),
        "--query-text-cache-jsonl",
        str(args.query_text_cache_jsonl),
        "--query-refresh-max-concurrency",
        str(int(args.per_worker_concurrency)),
    ]
    command.append("--progress" if bool(args.progress) else "--no-progress")
    return subprocess.run(command, cwd=str(common.REPO_ROOT), text=True, check=False)


def run_batches(args: argparse.Namespace) -> dict[str, object]:
    Path(args.shard_cache_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    shard_commands = {
        shard_index: _build_shard_command(args, shard_index) for shard_index in range(int(args.num_shards))
    }
    if bool(args.dry_run):
        return {
            "dry_run": True,
            "num_shards": int(args.num_shards),
            "max_parallel_jobs": int(args.max_parallel_jobs),
            "per_worker_concurrency": int(args.per_worker_concurrency),
            "finalize": bool(args.finalize),
            "shard_commands": shard_commands,
        }

    pending: list[int] = []
    shard_results: list[dict[str, object]] = []
    for shard_index in range(int(args.num_shards)):
        if bool(args.resume) and _shard_is_complete(args, shard_index):
            shard_results.append(
                {
                    "shard_index": shard_index,
                    "status": "skipped_complete",
                    "attempt_count": 0,
                    "exit_code": 0,
                    "log_path": str(_shard_log_path(args, shard_index)),
                    "cache_path": str(_shard_cache_path(args, shard_index)),
                    "summary_path": str(_shard_summary_path(args, shard_index)),
                }
            )
            print(f"[refresh] skipping shard {shard_index + 1}/{int(args.num_shards)} already complete")
            continue
        pending.append(shard_index)
    running: dict[int, tuple[subprocess.Popen[str], object, int]] = {}
    attempts: defaultdict[int, int] = defaultdict(int)
    while pending or running:
        while pending and len(running) < int(args.max_parallel_jobs):
            shard_index = pending.pop(0)
            log_path = _shard_log_path(args, shard_index)
            attempts[shard_index] += 1
            attempt_count = int(attempts[shard_index])
            log_handle = log_path.open("a", encoding="utf-8")
            if attempt_count > 1:
                log_handle.write(f"\n[retry] attempt={attempt_count}\n")
                log_handle.flush()
            proc = subprocess.Popen(
                shard_commands[shard_index],
                cwd=str(common.REPO_ROOT),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            running[shard_index] = (proc, log_handle, attempt_count)
            print(
                f"[refresh] started shard {shard_index + 1}/{int(args.num_shards)} "
                f"attempt={attempt_count} pid={proc.pid} log={log_path}"
            )
        if not running:
            continue
        finished: list[int] = []
        for shard_index, (proc, log_handle, attempt_count) in running.items():
            exit_code = proc.poll()
            if exit_code is None:
                continue
            log_handle.close()
            finished.append(shard_index)
            if exit_code != 0 and attempt_count < int(args.max_shard_attempts):
                pending.append(shard_index)
                print(
                    f"[refresh] shard {shard_index + 1}/{int(args.num_shards)} failed exit={exit_code}; "
                    f"retrying attempt {attempt_count + 1}/{int(args.max_shard_attempts)}"
                )
                continue
            shard_results.append(
                {
                    "shard_index": shard_index,
                    "status": "succeeded" if int(exit_code) == 0 else "failed",
                    "attempt_count": attempt_count,
                    "exit_code": int(exit_code),
                    "log_path": str(_shard_log_path(args, shard_index)),
                    "cache_path": str(_shard_cache_path(args, shard_index)),
                    "summary_path": str(_shard_summary_path(args, shard_index)),
                }
            )
            print(
                f"[refresh] finished shard {shard_index + 1}/{int(args.num_shards)} "
                f"attempt={attempt_count} exit={exit_code}"
            )
        for shard_index in finished:
            running.pop(shard_index, None)
        if running:
            time.sleep(1.0)

    failures = [item for item in shard_results if str(item.get("status")) == "failed"]
    summary = {
        "dry_run": False,
        "num_shards": int(args.num_shards),
        "max_parallel_jobs": int(args.max_parallel_jobs),
        "per_worker_concurrency": int(args.per_worker_concurrency),
        "max_shard_attempts": int(args.max_shard_attempts),
        "resume": bool(args.resume),
        "finalize": bool(args.finalize),
        "query_text_cache_jsonl": str(args.query_text_cache_jsonl),
        "shard_cache_dir": str(args.shard_cache_dir),
        "log_dir": str(args.log_dir),
        "shards": sorted(shard_results, key=lambda item: int(item["shard_index"])),
        "failures": failures,
    }
    if failures:
        common.write_json(Path(args.output_root) / "query_refresh_batches.summary.json", summary)
        raise RuntimeError(f"One or more shard refresh jobs failed: {failures}")

    merge_summary = _merge_caches(args)
    finalize_result: dict[str, object] = {"finalize": False}
    if bool(args.finalize):
        proc = _run_finalize_build(args)
        finalize_result = {
            "finalize": True,
            "exit_code": int(proc.returncode),
        }
        if proc.returncode != 0:
            summary.update(merge_summary)
            summary["finalize_result"] = finalize_result
            common.write_json(Path(args.output_root) / "query_refresh_batches.summary.json", summary)
            raise RuntimeError(f"Finalize build failed with exit_code={proc.returncode}")

    summary.update(merge_summary)
    summary["finalize_result"] = finalize_result
    common.write_json(Path(args.output_root) / "query_refresh_batches.summary.json", summary)
    return summary


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    summary = run_batches(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
