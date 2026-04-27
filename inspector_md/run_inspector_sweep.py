#!/usr/bin/env python3
"""Build and optionally launch staged Inspector MD SFT -> RL sweep runs."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from inspector_md import common
from tuna_sdk.retry import compute_backoff_delay

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = common.repo_relative("configs", "run_inspector_sweep_default.json")
DEFAULT_MANIFEST_PATH = common.repo_relative("outputs", "inspector_sweep_manifest.json")
DEFAULT_LOG_DIR = common.repo_relative("outputs", "inspector_sweep_logs")
DEFAULT_SUMMARY_PATH = common.repo_relative("outputs", "inspector_sweep_launch.summary.json")
DEFAULT_READINESS_CONFIG = common.repo_relative("configs", "check_inspector_finetune_readiness_default.json")
DEFAULT_SWEEP_FAMILIES = [
    "detect_sft",
    "detect_rl",
    "point_sft",
    "point_rl",
    "query_proposal_sft",
    "query_proposal_rl",
    "query_finding_sft",
    "query_finding_rl",
    "query_reasoning_sft",
    "query_reasoning_rl",
]
ALL_SWEEP_FAMILIES = [*DEFAULT_SWEEP_FAMILIES, "query_offpolicy_rl"]
_FINETUNE_ID_PATTERN = re.compile(r"resolved_finetune_id=([A-Za-z0-9_-]+)")
_STATUS_CODE_PATTERN = re.compile(r"status=(\d+)")
_ERROR_CODE_PATTERN = re.compile(r"error code:\s*(\d+)", re.IGNORECASE)
_TRANSIENT_QUERY_FAILURE_STATUS_CODES = frozenset({500, 502, 503, 504, 520, 524})


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _slug_float(value: float) -> str:
    text = f"{float(value):.0e}".replace("+", "")
    return text.replace(".", "p")


def _family_spec(family: str) -> dict[str, Any]:
    specs: dict[str, dict[str, Any]] = {
        "detect_sft": {
            "stage": "sft",
            "trainer": "detect",
            "module": "inspector_md.train_inspector_detect",
            "script": SCRIPT_DIR / "train_inspector_detect.py",
            "config": common.repo_relative("configs", "train_inspector_detect_sft_default.json"),
            "depends_on_family": "",
            "reasoning": False,
            "off_policy": False,
        },
        "detect_rl": {
            "stage": "rl",
            "trainer": "detect",
            "module": "inspector_md.train_inspector_detect",
            "script": SCRIPT_DIR / "train_inspector_detect.py",
            "config": common.repo_relative("configs", "train_inspector_detect_default.json"),
            "depends_on_family": "detect_sft",
            "reasoning": False,
            "off_policy": False,
        },
        "point_sft": {
            "stage": "sft",
            "trainer": "point",
            "module": "inspector_md.train_inspector_point",
            "script": SCRIPT_DIR / "train_inspector_point.py",
            "config": common.repo_relative("configs", "train_inspector_point_sft_default.json"),
            "depends_on_family": "",
            "reasoning": False,
            "off_policy": False,
        },
        "point_rl": {
            "stage": "rl",
            "trainer": "point",
            "module": "inspector_md.train_inspector_point",
            "script": SCRIPT_DIR / "train_inspector_point.py",
            "config": common.repo_relative("configs", "train_inspector_point_default.json"),
            "depends_on_family": "point_sft",
            "reasoning": False,
            "off_policy": False,
        },
        "query_proposal_sft": {
            "stage": "sft",
            "trainer": "query",
            "module": "inspector_md.train_inspector_query",
            "script": SCRIPT_DIR / "train_inspector_query.py",
            "config": common.repo_relative("configs", "train_inspector_query_proposal_default.json"),
            "depends_on_family": "",
            "reasoning": False,
            "off_policy": False,
            "mode": "sft",
        },
        "query_proposal_rl": {
            "stage": "rl",
            "trainer": "query",
            "module": "inspector_md.train_inspector_query",
            "script": SCRIPT_DIR / "train_inspector_query.py",
            "config": common.repo_relative("configs", "train_inspector_query_proposal_default.json"),
            "depends_on_family": "query_proposal_sft",
            "reasoning": False,
            "off_policy": False,
            "mode": "rl",
        },
        "query_finding_sft": {
            "stage": "sft",
            "trainer": "query",
            "module": "inspector_md.train_inspector_query",
            "script": SCRIPT_DIR / "train_inspector_query.py",
            "config": common.repo_relative("configs", "train_inspector_query_default.json"),
            "depends_on_family": "",
            "reasoning": False,
            "off_policy": False,
            "mode": "sft",
        },
        "query_finding_rl": {
            "stage": "rl",
            "trainer": "query",
            "module": "inspector_md.train_inspector_query",
            "script": SCRIPT_DIR / "train_inspector_query.py",
            "config": common.repo_relative("configs", "train_inspector_query_default.json"),
            "depends_on_family": "query_finding_sft",
            "reasoning": False,
            "off_policy": False,
            "mode": "rl",
        },
        "query_reasoning_sft": {
            "stage": "sft",
            "trainer": "query",
            "module": "inspector_md.train_inspector_query",
            "script": SCRIPT_DIR / "train_inspector_query.py",
            "config": common.repo_relative("configs", "train_inspector_query_reasoning_hard.json"),
            "depends_on_family": "",
            "reasoning": True,
            "off_policy": False,
            "mode": "sft",
        },
        "query_reasoning_rl": {
            "stage": "rl",
            "trainer": "query",
            "module": "inspector_md.train_inspector_query",
            "script": SCRIPT_DIR / "train_inspector_query.py",
            "config": common.repo_relative("configs", "train_inspector_query_reasoning_hard.json"),
            "depends_on_family": "query_reasoning_sft",
            "reasoning": True,
            "off_policy": False,
            "mode": "rl",
        },
        "query_offpolicy_rl": {
            "stage": "rl",
            "trainer": "query",
            "module": "inspector_md.train_inspector_query",
            "script": SCRIPT_DIR / "train_inspector_query.py",
            "config": common.repo_relative("configs", "train_inspector_query_default.json"),
            "depends_on_family": "query_finding_sft",
            "reasoning": False,
            "off_policy": True,
            "mode": "rl",
        },
    }
    if family not in specs:
        raise ValueError(f"Unsupported sweep family: {family}")
    return dict(specs[family])


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    raw_argv = list(argv) if argv is not None else list(sys.argv[1:])
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    pre_args, _ = pre_parser.parse_known_args(raw_argv)
    config_path = common.resolve_config_path(pre_args.config, script_dir=SCRIPT_DIR)
    config = common.load_json_config(config_path, default_path=DEFAULT_CONFIG_PATH)

    parser = argparse.ArgumentParser(description="Build and launch the approved Inspector MD SFT -> RL sweep.")
    parser.add_argument("--config", default=str(config_path))
    parser.add_argument("--env-file", default=str(common.repo_relative(".env.staging")))
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--base-url", default=common.DEFAULT_BASE_URL)
    parser.add_argument("--api-key-env-vars", nargs="+", default=list(common.DEFAULT_API_KEY_ENV_VARS))
    parser.add_argument("--ranks", nargs="+", type=int, default=[24, 32])
    parser.add_argument("--lrs", nargs="+", type=float, default=[2e-4, 5e-5])
    parser.add_argument("--groups-per-step", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--families", nargs="+", default=list(DEFAULT_SWEEP_FAMILIES))
    parser.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    parser.add_argument("--summary-path", default=str(DEFAULT_SUMMARY_PATH))
    parser.add_argument("--readiness-config", default=str(DEFAULT_READINESS_CONFIG))
    parser.add_argument("--skip-readiness-check", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-parallel-jobs", type=int, default=4)
    parser.add_argument("--query-max-parallel-jobs", type=int, default=1)
    parser.add_argument("--poll-interval-s", type=float, default=5.0)
    parser.add_argument("--query-run-max-retries", type=int, default=2)
    parser.add_argument("--query-run-retry-backoff-base-s", type=float, default=30.0)
    parser.add_argument("--query-run-retry-backoff-max-s", type=float, default=180.0)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--launch", action=argparse.BooleanOptionalAction, default=False)

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
    args.env_file = str(common.resolve_path(args.env_file, module_root=SCRIPT_DIR))
    python_executable = str(args.python_executable or "").strip()
    if python_executable in {"", "python", "python3"}:
        args.python_executable = sys.executable
    else:
        args.python_executable = str(Path(python_executable).expanduser())
    args.api_key_env_vars = common.normalize_api_key_env_vars(args.api_key_env_vars)
    args.families = [str(item).strip() for item in list(args.families or []) if str(item).strip()]
    if not args.families:
        raise ValueError("At least one sweep family is required.")
    invalid_families = sorted({family for family in args.families if family not in ALL_SWEEP_FAMILIES})
    if invalid_families:
        raise ValueError(f"Unsupported sweep family(s): {invalid_families}")
    args.manifest_path = common.resolve_path(args.manifest_path, module_root=SCRIPT_DIR)
    args.log_dir = common.resolve_path(args.log_dir, module_root=SCRIPT_DIR)
    args.summary_path = common.resolve_path(args.summary_path, module_root=SCRIPT_DIR)
    args.readiness_config = str(common.resolve_config_path(args.readiness_config, script_dir=SCRIPT_DIR))
    if any(int(rank) <= 0 for rank in list(args.ranks or [])):
        raise ValueError("--ranks must be positive integers.")
    if any(float(lr) <= 0 for lr in list(args.lrs or [])):
        raise ValueError("--lrs must be positive.")
    if any(int(group) not in {4, 8} for group in list(args.groups_per_step or [])):
        raise ValueError("--groups-per-step must use the approved values {4, 8}.")
    if int(args.max_parallel_jobs) <= 0:
        raise ValueError("--max-parallel-jobs must be >= 1.")
    if int(args.query_max_parallel_jobs) <= 0:
        raise ValueError("--query-max-parallel-jobs must be >= 1.")
    if float(args.poll_interval_s) <= 0:
        raise ValueError("--poll-interval-s must be > 0.")
    if int(args.query_run_max_retries) < 0:
        raise ValueError("--query-run-max-retries must be >= 0.")
    if float(args.query_run_retry_backoff_base_s) <= 0:
        raise ValueError("--query-run-retry-backoff-base-s must be > 0.")
    if float(args.query_run_retry_backoff_max_s) < float(args.query_run_retry_backoff_base_s):
        raise ValueError("--query-run-retry-backoff-max-s must be >= --query-run-retry-backoff-base-s.")
    return args


def _build_base_command(
    *,
    args: argparse.Namespace,
    family: str,
    rank: int,
    lr: float,
    groups_per_step: int,
    assigned_api_key_env_var: str,
    run_name: str,
) -> list[str]:
    spec = _family_spec(family)
    command = [
        str(args.python_executable),
        "-m",
        str(spec["module"]),
        "--config",
        str(Path(spec["config"]).resolve()),
        "--env-file",
        str(args.env_file),
        "--base-url",
        str(args.base_url),
        "--wandb-run-name",
        str(run_name),
        "--rank",
        str(int(rank)),
    ]
    trainer = str(spec["trainer"])
    if trainer in {"detect", "point"}:
        command.extend(
            [
                "--api-key-env-var",
                str(assigned_api_key_env_var),
                "--lr",
                str(float(lr)),
                "--batch-size",
                str(int(groups_per_step)),
                "--group-size",
                "8",
            ]
        )
        if spec["stage"] == "rl":
            command.extend(["--sft-bootstrap-steps", "0"])
    elif trainer == "query":
        command.extend(
            [
                "--api-key-env-var",
                str(assigned_api_key_env_var),
                "--mode",
                str(spec["mode"]),
                "--batch-size",
                str(int(groups_per_step)),
                "--num-rollouts",
                "8",
                "--rollout-stream-max-concurrency",
                "4",
                "--rollout-stream-buffer-size",
                "8",
            ]
        )
        if spec["stage"] == "sft":
            command.extend(["--sft-lr", str(float(lr)), "--rl-steps", "0"])
        else:
            command.extend(["--rl-lr", str(float(lr)), "--sft-steps", "0"])
        if bool(spec.get("reasoning")):
            command.append("--reasoning")
        if bool(spec.get("off_policy")):
            command.append("--off-policy")
    else:
        raise ValueError(f"Unsupported trainer: {trainer}")
    return command


def build_sweep_runs(args: argparse.Namespace) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    dependency_lookup: dict[tuple[str, int, float, int], str] = {}
    ordered_families = [family for family in DEFAULT_SWEEP_FAMILIES if family in args.families] + [
        family for family in args.families if family not in DEFAULT_SWEEP_FAMILIES
    ]
    stage_order = {"sft": 0, "rl": 1}
    family_specs = {family: _family_spec(family) for family in ordered_families}
    ordered_families = sorted(
        ordered_families,
        key=lambda family: (stage_order[family_specs[family]["stage"]], ordered_families.index(family)),
    )
    assignment_index = 0
    stage_to_families: dict[str, list[str]] = {
        "sft": [family for family in ordered_families if family_specs[family]["stage"] == "sft"],
        "rl": [family for family in ordered_families if family_specs[family]["stage"] == "rl"],
    }
    combinations = [
        (int(rank), float(lr), int(groups_per_step))
        for rank in list(args.ranks or [])
        for lr in list(args.lrs or [])
        for groups_per_step in list(args.groups_per_step or [])
    ]
    for stage_name in ("sft", "rl"):
        for rank, lr, groups_per_step in combinations:
            for family in stage_to_families[stage_name]:
                spec = family_specs[family]
                combo_key = (family, int(rank), float(lr), int(groups_per_step))
                run_name = f"{family}-r{int(rank)}-lr{_slug_float(float(lr))}-g{int(groups_per_step)}"
                depends_on_run_name = ""
                depends_on_family = str(spec.get("depends_on_family") or "")
                if depends_on_family:
                    depends_on_run_name = dependency_lookup[(depends_on_family, int(rank), float(lr), int(groups_per_step))]
                assigned_api_key_env_var = str(args.api_key_env_vars[assignment_index % len(args.api_key_env_vars)])
                assignment_index += 1
                command_template = _build_base_command(
                    args=args,
                    family=family,
                    rank=int(rank),
                    lr=float(lr),
                    groups_per_step=int(groups_per_step),
                    assigned_api_key_env_var=assigned_api_key_env_var,
                    run_name=run_name,
                )
                runs.append(
                    {
                        "run_name": run_name,
                        "family": family,
                        "stage": str(spec["stage"]),
                        "trainer": str(spec["trainer"]),
                        "script_path": str(Path(spec["script"]).resolve()),
                        "config_path": str(Path(spec["config"]).resolve()),
                        "depends_on_run_name": depends_on_run_name,
                        "assigned_api_key_env_var": assigned_api_key_env_var,
                        "base_url": str(args.base_url),
                        "rank": int(rank),
                        "lr": float(lr),
                        "groups_per_step": int(groups_per_step),
                        "batch_size": int(groups_per_step),
                        "num_rollouts": 8,
                        "rollout_stream_max_concurrency": 4,
                        "rollout_stream_buffer_size": 8,
                        "reasoning": bool(spec.get("reasoning", False)),
                        "off_policy": bool(spec.get("off_policy", False)),
                        "finetune_name": run_name if str(spec["stage"]) == "sft" else "",
                        "command_template": command_template,
                    }
                )
                dependency_lookup[combo_key] = run_name
    return runs


def _manifest_payload(args: argparse.Namespace, runs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "generated_at": _utc_now_iso(),
        "base_url": str(args.base_url),
        "api_key_env_vars": list(args.api_key_env_vars),
        "max_parallel_jobs": int(args.max_parallel_jobs),
        "query_max_parallel_jobs": int(args.query_max_parallel_jobs),
        "reserved_trainer_slots": _reserved_trainer_slots(
            runs,
            max_parallel=min(int(args.max_parallel_jobs), len(args.api_key_env_vars)),
        ),
        "families": list(args.families),
        "runs": runs,
    }


def _check_readiness(args: argparse.Namespace) -> dict[str, Any]:
    command = [
        str(args.python_executable),
        "-m",
        "inspector_md.check_inspector_finetune_readiness",
        "--config",
        str(args.readiness_config),
    ]
    if not any(str(family).startswith("query_") for family in list(args.families or [])):
        command.append("--no-require-query-text-refresh")
    proc = subprocess.run(
        command,
        cwd=str(common.REPO_ROOT),
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        check=False,
    )
    payload: dict[str, Any]
    try:
        payload = json.loads(proc.stdout.strip() or "{}")
    except json.JSONDecodeError:
        payload = {"raw_stdout": proc.stdout, "raw_stderr": proc.stderr}
    return {
        "command": command,
        "returncode": int(proc.returncode),
        "summary": payload,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _materialize_launch_command(run: dict[str, Any], *, dependency_finetune_id: str = "") -> list[str]:
    command = list(run["command_template"])
    if str(dependency_finetune_id or "").strip():
        command.extend(["--finetune-id", str(dependency_finetune_id).strip()])
    else:
        finetune_name = str(run.get("finetune_name") or "").strip()
        if finetune_name:
            command.extend(["--finetune-name", finetune_name])
    return command


def _extract_finetune_id_from_log(log_path: Path) -> str:
    if not log_path.is_file():
        return ""
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = _FINETUNE_ID_PATTERN.findall(text)
    if matches:
        return str(matches[-1]).strip()
    return ""


def _sanitize_log_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    return safe or "run"


def _reserved_trainer_slots(runs: list[dict[str, Any]], *, max_parallel: int) -> dict[str, int]:
    first_wave = [dict(run) for run in runs[: max(0, int(max_parallel))]]
    counts = Counter(str(run.get("trainer") or "").strip() for run in first_wave if str(run.get("trainer") or "").strip())
    return {trainer: int(count) for trainer, count in counts.items()}


def _pending_trainers_needing_slots(
    *,
    pending: list[dict[str, Any]],
    running: dict[str, dict[str, Any]],
    reserved_slots: dict[str, int],
) -> set[str]:
    pending_counts = Counter(str(run.get("trainer") or "").strip() for run in pending)
    running_counts = Counter(str(state["run"].get("trainer") or "").strip() for state in running.values())
    return {
        trainer
        for trainer, reserved_count in reserved_slots.items()
        if int(reserved_count) > 0
        and int(running_counts.get(trainer, 0)) < int(reserved_count)
        and int(pending_counts.get(trainer, 0)) > 0
    }


def _query_failure_details(log_path: Path) -> dict[str, Any]:
    if not log_path.is_file():
        return {"status_code": None, "transient": False, "summary": ""}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    tail = text[-8000:]
    status_codes = [int(value) for value in _STATUS_CODE_PATTERN.findall(tail)]
    if not status_codes:
        status_codes = [int(value) for value in _ERROR_CODE_PATTERN.findall(tail)]
    status_code = int(status_codes[-1]) if status_codes else None
    if any(marker in tail for marker in ("TunaNetworkError", "Network error while calling API")):
        return {
            "status_code": status_code,
            "transient": True,
            "summary": "network_error",
        }
    transient = status_code in _TRANSIENT_QUERY_FAILURE_STATUS_CODES
    summary = f"status={status_code}" if status_code is not None else ""
    return {
        "status_code": status_code,
        "transient": bool(transient),
        "summary": summary,
    }


def _launch_runs(args: argparse.Namespace, runs: list[dict[str, Any]]) -> dict[str, Any]:
    common.maybe_load_env_file(args.env_file, override=False)
    common.resolve_api_key_pool(api_key_env_vars=args.api_key_env_vars)
    if not bool(args.skip_readiness_check):
        readiness = _check_readiness(args)
        if int(readiness.get("returncode", 1)) != 0:
            common.write_json(
                Path(args.summary_path),
                {
                    "generated_at": _utc_now_iso(),
                    "launch_attempted": False,
                    "readiness": readiness,
                    "reason": "readiness_check_failed",
                },
            )
            raise RuntimeError("Inspector MD readiness check failed; refusing to launch the sweep.")
    else:
        readiness = {"skipped": True}

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    pending = [{**dict(run), "retry_count": 0} for run in runs]
    running: dict[str, dict[str, Any]] = {}
    completed: dict[str, dict[str, Any]] = {}
    max_parallel = min(int(args.max_parallel_jobs), len(args.api_key_env_vars))
    reserved_slots = _reserved_trainer_slots(runs, max_parallel=max_parallel)

    while pending or running:
        started_any = False
        in_use_keys = {str(item["run"]["assigned_api_key_env_var"]) for item in running.values()}
        while pending and len(running) < max_parallel:
            reserved_trainers = _pending_trainers_needing_slots(
                pending=pending,
                running=running,
                reserved_slots=reserved_slots,
            )
            now_monotonic = time.monotonic()
            running_counts = Counter(str(state["run"].get("trainer") or "").strip() for state in running.values())
            started_this_round = False
            for index, run in enumerate(list(pending)):
                if reserved_trainers and str(run.get("trainer") or "").strip() not in reserved_trainers:
                    continue
                if str(run.get("trainer") or "").strip() == "query" and int(running_counts.get("query", 0)) >= int(args.query_max_parallel_jobs):
                    continue
                retry_not_before = float(run.get("retry_not_before_monotonic") or 0.0)
                if retry_not_before > now_monotonic:
                    continue
                dependency_name = str(run.get("depends_on_run_name") or "").strip()
                dependency_finetune_id = ""
                if dependency_name:
                    dependency_state = completed.get(dependency_name)
                    if dependency_state is None:
                        continue
                    if str(dependency_state.get("status")) != "succeeded":
                        blocked = {
                            **run,
                            "status": "blocked_dependency",
                            "dependency_status": str(dependency_state.get("status") or ""),
                            "dependency_run_name": dependency_name,
                        }
                        completed[str(run["run_name"])] = blocked
                        pending.pop(index)
                        started_this_round = True
                        break
                    dependency_finetune_id = str(dependency_state.get("resolved_finetune_id") or "").strip()
                    if not dependency_finetune_id:
                        blocked = {
                            **run,
                            "status": "blocked_dependency",
                            "dependency_status": "missing_finetune_id",
                            "dependency_run_name": dependency_name,
                        }
                        completed[str(run["run_name"])] = blocked
                        pending.pop(index)
                        started_this_round = True
                        break
                assigned_key = str(run["assigned_api_key_env_var"])
                if assigned_key in in_use_keys:
                    continue
                log_path = Path(args.log_dir) / f"{_sanitize_log_name(run['run_name'])}.log"
                retry_count = int(run.get("retry_count", 0) or 0)
                if retry_count > 0:
                    log_path = Path(args.log_dir) / f"{_sanitize_log_name(run['run_name'])}.retry_{retry_count:02d}.log"
                command = _materialize_launch_command(run, dependency_finetune_id=dependency_finetune_id)
                log_handle = log_path.open("w", encoding="utf-8")
                proc = subprocess.Popen(
                    command,
                    cwd=str(common.REPO_ROOT),
                    env=os.environ.copy(),
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                running[str(run["run_name"])] = {
                    "run": run,
                    "proc": proc,
                    "log_handle": log_handle,
                    "log_path": str(log_path),
                    "command": command,
                    "started_at": _utc_now_iso(),
                }
                in_use_keys.add(assigned_key)
                pending.pop(index)
                print(
                    f"[sweep] started {run['run_name']} stage={run['stage']} key={assigned_key} "
                    f"pid={proc.pid} log={log_path}"
                )
                started_any = True
                started_this_round = True
                break
            if not started_this_round:
                break

        finished_names: list[str] = []
        for run_name, state in list(running.items()):
            proc = state["proc"]
            exit_code = proc.poll()
            if exit_code is None:
                continue
            state["log_handle"].close()
            log_path = Path(str(state["log_path"]))
            resolved_finetune_id = _extract_finetune_id_from_log(log_path)
            status = "succeeded" if int(exit_code) == 0 else "failed"
            if status == "succeeded" and str(state["run"]["stage"]) == "sft" and not resolved_finetune_id:
                status = "failed"
            retry_count = int(state["run"].get("retry_count", 0) or 0)
            if status == "failed" and str(state["run"].get("trainer") or "") == "query":
                failure_details = _query_failure_details(log_path)
                if bool(failure_details.get("transient")) and retry_count < int(args.query_run_max_retries):
                    delay_s = compute_backoff_delay(
                        retry_count,
                        base=float(args.query_run_retry_backoff_base_s),
                        max_delay=float(args.query_run_retry_backoff_max_s),
                        jitter=0.1,
                    )
                    pending.insert(
                        0,
                        {
                            **state["run"],
                            "retry_count": retry_count + 1,
                            "retry_not_before_monotonic": time.monotonic() + delay_s,
                            "previous_log_paths": [*list(state["run"].get("previous_log_paths") or []), str(log_path)],
                            "last_failure_status_code": failure_details.get("status_code"),
                            "last_failure_summary": str(failure_details.get("summary") or ""),
                        },
                    )
                    print(
                        f"[sweep] retrying {run_name} after transient query failure "
                        f"attempt={retry_count + 1}/{int(args.query_run_max_retries) + 1} "
                        f"delay={delay_s:.1f}s {failure_details.get('summary') or ''}".rstrip()
                    )
                    finished_names.append(run_name)
                    continue
            completed[run_name] = {
                **state["run"],
                "status": status,
                "returncode": int(exit_code),
                "started_at": state["started_at"],
                "finished_at": _utc_now_iso(),
                "resolved_finetune_id": resolved_finetune_id,
                "command": state["command"],
                "log_path": str(log_path),
            }
            print(
                f"[sweep] finished {run_name} exit={exit_code} status={status}"
                + (f" finetune_id={resolved_finetune_id}" if resolved_finetune_id else "")
            )
            finished_names.append(run_name)
        for run_name in finished_names:
            running.pop(run_name, None)

        if running or pending:
            if not started_any and not finished_names:
                time.sleep(float(args.poll_interval_s))

    ordered_results = [completed[str(run["run_name"])] for run in runs if str(run["run_name"]) in completed]
    summary = {
        "generated_at": _utc_now_iso(),
        "launch_attempted": True,
        "base_url": str(args.base_url),
        "api_key_env_vars": list(args.api_key_env_vars),
        "max_parallel_jobs": int(max_parallel),
        "query_max_parallel_jobs": int(args.query_max_parallel_jobs),
        "reserved_trainer_slots": reserved_slots,
        "readiness": readiness,
        "runs": ordered_results,
        "succeeded_count": sum(1 for item in ordered_results if str(item.get("status")) == "succeeded"),
        "failed_count": sum(1 for item in ordered_results if str(item.get("status")) == "failed"),
        "blocked_count": sum(1 for item in ordered_results if str(item.get("status")) == "blocked_dependency"),
    }
    common.write_json(Path(args.summary_path), summary)
    return summary


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    runs = build_sweep_runs(args)
    manifest_payload = _manifest_payload(args, runs)
    if bool(args.dry_run):
        print(json.dumps(manifest_payload, indent=2))
        return
    common.write_json(Path(args.manifest_path), manifest_payload)
    if not bool(args.launch):
        print(json.dumps({"manifest_path": str(args.manifest_path), "run_count": len(runs)}, indent=2))
        return
    summary = _launch_runs(args, runs)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
