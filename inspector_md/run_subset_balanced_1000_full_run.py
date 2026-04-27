#!/usr/bin/env python3
"""Sequential full-run orchestration for the Inspector MD 1k balanced subset."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inspector_md import common

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = common.repo_relative("configs", "run_subset_balanced_1000_full_run_default.json")
ALL_STAGES = ("build", "readiness", "train", "benchmark")
BUILD_DATASET_CONFIG = common.repo_relative("configs", "build_inspector_dataset_default.json")
READINESS_CONFIG = common.repo_relative("configs", "check_inspector_finetune_readiness_default.json")
TRAIN_CONFIGS = {
    "detect_sft": common.repo_relative("configs", "train_inspector_detect_sft_default.json"),
    "detect_rl": common.repo_relative("configs", "train_inspector_detect_default.json"),
    "point_sft": common.repo_relative("configs", "train_inspector_point_sft_default.json"),
    "point_rl": common.repo_relative("configs", "train_inspector_point_default.json"),
    "query": common.repo_relative("configs", "train_inspector_query_default.json"),
}
BENCHMARK_CONFIGS = {
    "detect": common.repo_relative("configs", "benchmark_inspector_detect_default.json"),
    "query": common.repo_relative("configs", "benchmark_inspector_query_default.json"),
    "pipeline": common.repo_relative("configs", "benchmark_inspector_pipeline_default.json"),
}
FINETUNE_ID_RE = re.compile(r"resolved_finetune_id=([A-Za-z0-9_-]+)")
DONE_RE = re.compile(
    r"done\. finetune_id=([A-Za-z0-9_-]+)\s+best_step=([^\s]+)\s+best_metric=([^\s]+)\s+"
    r"recall_gate_pass=([^\s]+)\s+f1_target_pass=([^\s]+)\s+stopped_early=([^\s]+)"
)
RUN_DIR_RE = re.compile(r"finished finetune=([A-Za-z0-9_-]+)\s+run_dir=(\S+)")


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _read_json(path: Path, *, default: Any) -> Any:
    if not path.is_file():
        return default
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload


def _write_json(path: Path, payload: Any) -> None:
    common.write_json(path, _json_safe(payload))


def _coerce_optional_int(value: Any) -> Optional[int]:
    text = str(value or "").strip()
    if not text or text.lower() == "none":
        return None
    try:
        return int(text)
    except (TypeError, ValueError):
        return None


def _coerce_optional_float(value: Any) -> Optional[float]:
    text = str(value or "").strip()
    if not text or text.lower() == "none":
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _coerce_optional_bool(value: Any) -> Optional[bool]:
    text = str(value or "").strip().lower()
    if not text or text == "none":
        return None
    if text in {"1", "true", "yes"}:
        return True
    if text in {"0", "false", "no"}:
        return False
    return None


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    raw_argv = list(argv) if argv is not None else list(sys.argv[1:])
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    pre_args, _ = pre_parser.parse_known_args(raw_argv)
    config_path = common.resolve_config_path(pre_args.config, script_dir=SCRIPT_DIR)
    config = common.load_json_config(config_path, default_path=DEFAULT_CONFIG_PATH)

    parser = argparse.ArgumentParser(description="Run the Inspector MD 1k balanced subset baseline sequence.")
    parser.add_argument("--config", default=str(config_path))
    parser.add_argument("--env-file", default=str(common.repo_relative(".env.staging")))
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument(
        "--source-manifest",
        default=str(common.repo_relative("dataset", "subset_balanced_1000", "synthetic_manifest.json")),
    )
    parser.add_argument(
        "--query-text-cache-jsonl",
        default=str(common.repo_relative("outputs", "inspector_query_text_cache.jsonl")),
    )
    parser.add_argument("--output-root", default=str(common.repo_relative("outputs", "subset_balanced_1000")))
    parser.add_argument(
        "--detect-output-dir",
        default=str(common.repo_relative("outputs", "subset_balanced_1000", "inspector_detect_v1")),
    )
    parser.add_argument(
        "--point-output-dir",
        default=str(common.repo_relative("outputs", "subset_balanced_1000", "inspector_point_v1")),
    )
    parser.add_argument(
        "--query-output-dir",
        default=str(common.repo_relative("outputs", "subset_balanced_1000", "inspector_query_issues_v2")),
    )
    parser.add_argument(
        "--readiness-output-json",
        default=str(common.repo_relative("outputs", "subset_balanced_1000", "inspector_finetune_readiness.json")),
    )
    parser.add_argument(
        "--log-dir",
        default=str(common.repo_relative("outputs", "train_logs", "subset_balanced_1000")),
    )
    parser.add_argument(
        "--run-output-root",
        default=str(common.repo_relative("outputs", "runs", "subset_balanced_1000")),
    )
    parser.add_argument(
        "--benchmark-output-root",
        default=str(common.repo_relative("outputs", "benchmarks", "subset_balanced_1000")),
    )
    parser.add_argument(
        "--summary-json",
        default=str(common.repo_relative("outputs", "subset_balanced_1000", "full_run_summary.json")),
    )
    parser.add_argument("--run-tag", default="subset1k-20260414")
    parser.add_argument("--stages", nargs="+", default=list(ALL_STAGES))
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--detect-sft-steps", type=int, default=40)
    parser.add_argument("--detect-rl-steps", type=int, default=80)
    parser.add_argument("--point-sft-steps", type=int, default=40)
    parser.add_argument("--point-rl-steps", type=int, default=80)
    parser.add_argument("--query-sft-steps", type=int, default=60)
    parser.add_argument("--query-rl-steps", type=int, default=120)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--detect-point-eval-max-samples", type=int, default=68)
    parser.add_argument("--query-eval-max-samples", type=int, default=64)

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
    args.python_executable = str(args.python_executable or "").strip() or sys.executable
    if args.python_executable in {"python", "python3"}:
        args.python_executable = sys.executable
    args.source_manifest = common.resolve_path(args.source_manifest, module_root=SCRIPT_DIR)
    args.query_text_cache_jsonl = common.resolve_path(args.query_text_cache_jsonl, module_root=SCRIPT_DIR)
    args.output_root = common.resolve_path(args.output_root, module_root=SCRIPT_DIR)
    args.detect_output_dir = common.resolve_path(args.detect_output_dir, module_root=SCRIPT_DIR)
    args.point_output_dir = common.resolve_path(args.point_output_dir, module_root=SCRIPT_DIR)
    args.query_output_dir = common.resolve_path(args.query_output_dir, module_root=SCRIPT_DIR)
    args.readiness_output_json = common.resolve_path(args.readiness_output_json, module_root=SCRIPT_DIR)
    args.log_dir = common.resolve_path(args.log_dir, module_root=SCRIPT_DIR)
    args.run_output_root = common.resolve_path(args.run_output_root, module_root=SCRIPT_DIR)
    args.benchmark_output_root = common.resolve_path(args.benchmark_output_root, module_root=SCRIPT_DIR)
    args.summary_json = common.resolve_path(args.summary_json, module_root=SCRIPT_DIR)
    normalized_stages = [str(item).strip().lower() for item in list(args.stages or []) if str(item).strip()]
    if "all" in normalized_stages:
        normalized_stages = list(ALL_STAGES)
    invalid = [stage for stage in normalized_stages if stage not in ALL_STAGES]
    if invalid:
        raise ValueError(f"Unsupported stage(s): {invalid}")
    args.stages = normalized_stages
    return args


def _load_summary(path: Path) -> dict[str, Any]:
    payload = _read_json(path, default={})
    return payload if isinstance(payload, dict) else {}


def _save_summary(path: Path, summary: dict[str, Any]) -> None:
    _write_json(path, summary)


def _ensure_summary_base(args: argparse.Namespace) -> dict[str, Any]:
    summary = _load_summary(Path(args.summary_json))
    summary.setdefault("created_at", _now_iso())
    summary["updated_at"] = _now_iso()
    summary["run_tag"] = str(args.run_tag)
    summary["source_manifest"] = str(Path(args.source_manifest))
    summary["paths"] = {
        "output_root": str(Path(args.output_root)),
        "detect_output_dir": str(Path(args.detect_output_dir)),
        "point_output_dir": str(Path(args.point_output_dir)),
        "query_output_dir": str(Path(args.query_output_dir)),
        "readiness_output_json": str(Path(args.readiness_output_json)),
        "log_dir": str(Path(args.log_dir)),
        "run_output_root": str(Path(args.run_output_root)),
        "benchmark_output_root": str(Path(args.benchmark_output_root)),
    }
    summary.setdefault("build", {})
    summary.setdefault("readiness", {})
    summary.setdefault("runs", {})
    summary.setdefault("candidates", {})
    summary.setdefault("benchmarks", {})
    return summary


def _stage_completed(section: MappingLike, *, status: str = "succeeded") -> bool:
    return str(section.get("status") or "") == str(status)


class MappingLike(dict):
    pass


def _run_logged_command(command: list[str], *, log_path: Path, cwd: Path) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    last_lines: deque[str] = deque(maxlen=60)
    finetune_id = ""
    started_at = _now_iso()
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"$ {' '.join(command)}\n")
        handle.flush()
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=os.environ.copy(),
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            handle.write(line)
            handle.flush()
            last_lines.append(line.rstrip("\n"))
            match = FINETUNE_ID_RE.search(line)
            if match:
                finetune_id = str(match.group(1)).strip()
        returncode = int(process.wait())
    return {
        "status": "succeeded" if returncode == 0 else "failed",
        "returncode": returncode,
        "command": command,
        "log_path": str(log_path),
        "started_at": started_at,
        "ended_at": _now_iso(),
        "resolved_finetune_id": finetune_id,
        "tail": list(last_lines),
    }


def _parse_detect_point_training_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
    finetune_match = None
    for match in FINETUNE_ID_RE.finditer(text):
        finetune_match = match
    done_match = None
    for match in DONE_RE.finditer(text):
        done_match = match
    result: dict[str, Any] = {
        "resolved_finetune_id": str(finetune_match.group(1)).strip() if finetune_match else "",
        "best_step": None,
        "best_metric": None,
        "recall_gate_pass": None,
        "f1_target_pass": None,
        "stopped_early": None,
    }
    if done_match is None:
        return result
    result.update(
        {
            "resolved_finetune_id": str(done_match.group(1)).strip(),
            "best_step": _coerce_optional_int(done_match.group(2)),
            "best_metric": _coerce_optional_float(done_match.group(3)),
            "recall_gate_pass": _coerce_optional_bool(done_match.group(4)),
            "f1_target_pass": _coerce_optional_bool(done_match.group(5)),
            "stopped_early": _coerce_optional_bool(done_match.group(6)),
        }
    )
    return result


def _load_query_train_summary(*, run_output_dir: Path, finetune_id: str) -> dict[str, Any]:
    if not str(finetune_id or "").strip():
        return {}
    train_summary_path = run_output_dir / str(finetune_id).strip() / "train_summary.json"
    payload = _read_json(train_summary_path, default={})
    if not isinstance(payload, dict):
        return {}
    payload["train_summary_path"] = str(train_summary_path)
    return payload


def _build_dataset_command(args: argparse.Namespace) -> list[str]:
    return [
        str(args.python_executable),
        "-m",
        "inspector_md.build_inspector_dataset",
        "--config",
        str(BUILD_DATASET_CONFIG.resolve()),
        "--source-manifest",
        str(args.source_manifest),
        "--output-root",
        str(args.output_root),
        "--detect-output-dir",
        str(args.detect_output_dir),
        "--point-output-dir",
        str(args.point_output_dir),
        "--query-output-dir",
        str(args.query_output_dir),
        "--query-text-cache-jsonl",
        str(args.query_text_cache_jsonl),
    ]


def _readiness_command(args: argparse.Namespace) -> list[str]:
    return [
        str(args.python_executable),
        "-m",
        "inspector_md.check_inspector_finetune_readiness",
        "--config",
        str(READINESS_CONFIG.resolve()),
        "--detect-dataset-path",
        str(args.detect_output_dir),
        "--point-dataset-path",
        str(args.point_output_dir),
        "--query-dataset-dir",
        str(args.query_output_dir),
        "--output-json",
        str(args.readiness_output_json),
    ]


def _train_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    return [
        {
            "key": "detect_sft",
            "module": "inspector_md.train_inspector_detect",
            "config": TRAIN_CONFIGS["detect_sft"],
            "log_name": "detect_sft.log",
            "wandb_run_name": f"inspector-detect-{args.run_tag}-sft",
            "finetune_name": f"inspector-detect-{args.run_tag}-sft",
            "dataset_path": str(args.detect_output_dir),
            "args": [
                "--dataset-path",
                str(args.detect_output_dir),
                "--num-steps",
                str(int(args.detect_sft_steps)),
                "--eval-every",
                str(int(args.eval_every)),
                "--save-every",
                str(int(args.save_every)),
                "--eval-max-samples",
                str(int(args.detect_point_eval_max_samples)),
                "--async-checkpoint-eval-dir",
                str(Path(args.output_root) / "async_checkpoint_eval_detect"),
            ],
        },
        {
            "key": "detect_rl",
            "module": "inspector_md.train_inspector_detect",
            "config": TRAIN_CONFIGS["detect_rl"],
            "log_name": "detect_rl.log",
            "wandb_run_name": f"inspector-detect-{args.run_tag}-rl",
            "dataset_path": str(args.detect_output_dir),
            "dependency": "detect_sft",
            "args": [
                "--dataset-path",
                str(args.detect_output_dir),
                "--num-steps",
                str(int(args.detect_rl_steps)),
                "--sft-bootstrap-steps",
                "0",
                "--eval-every",
                str(int(args.eval_every)),
                "--save-every",
                str(int(args.save_every)),
                "--eval-max-samples",
                str(int(args.detect_point_eval_max_samples)),
                "--async-checkpoint-eval-dir",
                str(Path(args.output_root) / "async_checkpoint_eval_detect"),
            ],
        },
        {
            "key": "point_sft",
            "module": "inspector_md.train_inspector_point",
            "config": TRAIN_CONFIGS["point_sft"],
            "log_name": "point_sft.log",
            "wandb_run_name": f"inspector-point-{args.run_tag}-sft",
            "finetune_name": f"inspector-point-{args.run_tag}-sft",
            "dataset_path": str(args.point_output_dir),
            "args": [
                "--dataset-path",
                str(args.point_output_dir),
                "--num-steps",
                str(int(args.point_sft_steps)),
                "--eval-every",
                str(int(args.eval_every)),
                "--save-every",
                str(int(args.save_every)),
                "--eval-max-samples",
                str(int(args.detect_point_eval_max_samples)),
                "--async-checkpoint-eval-dir",
                str(Path(args.output_root) / "async_checkpoint_eval_point"),
            ],
        },
        {
            "key": "point_rl",
            "module": "inspector_md.train_inspector_point",
            "config": TRAIN_CONFIGS["point_rl"],
            "log_name": "point_rl.log",
            "wandb_run_name": f"inspector-point-{args.run_tag}-rl",
            "dataset_path": str(args.point_output_dir),
            "dependency": "point_sft",
            "args": [
                "--dataset-path",
                str(args.point_output_dir),
                "--num-steps",
                str(int(args.point_rl_steps)),
                "--sft-bootstrap-steps",
                "0",
                "--eval-every",
                str(int(args.eval_every)),
                "--save-every",
                str(int(args.save_every)),
                "--eval-max-samples",
                str(int(args.detect_point_eval_max_samples)),
                "--async-checkpoint-eval-dir",
                str(Path(args.output_root) / "async_checkpoint_eval_point"),
            ],
        },
        {
            "key": "query",
            "module": "inspector_md.train_inspector_query",
            "config": TRAIN_CONFIGS["query"],
            "log_name": "query.log",
            "wandb_run_name": f"inspector-query-{args.run_tag}",
            "finetune_name": f"inspector-query-{args.run_tag}",
            "dataset_dir": str(args.query_output_dir),
            "run_output_dir": str(Path(args.run_output_root) / "query"),
            "args": [
                "--dataset-dir",
                str(args.query_output_dir),
                "--mode",
                "sft_then_rl",
                "--sft-steps",
                str(int(args.query_sft_steps)),
                "--rl-steps",
                str(int(args.query_rl_steps)),
                "--eval-every",
                str(int(args.eval_every)),
                "--save-every",
                str(int(args.save_every)),
                "--eval-max-samples",
                str(int(args.query_eval_max_samples)),
                "--run-output-dir",
                str(Path(args.run_output_root) / "query"),
                "--async-checkpoint-eval-dir",
                str(Path(args.output_root) / "async_checkpoint_eval_query"),
            ],
        },
    ]


def _build_train_command(spec: dict[str, Any], *, dependency_finetune_id: str, python_executable: str) -> list[str]:
    command = [
        str(python_executable),
        "-m",
        str(spec["module"]),
        "--config",
        str(Path(spec["config"]).resolve()),
        "--wandb-run-name",
        str(spec["wandb_run_name"]),
    ]
    if dependency_finetune_id:
        command.extend(["--finetune-id", str(dependency_finetune_id)])
    elif str(spec.get("finetune_name") or "").strip():
        command.extend(["--finetune-name", str(spec["finetune_name"])])
    command.extend(str(item) for item in list(spec["args"]))
    return command


def _pick_detect_or_point_candidate(
    *,
    sft_run: dict[str, Any],
    rl_run: dict[str, Any],
) -> dict[str, Any]:
    sft_metric = _coerce_optional_float(sft_run.get("best_metric"))
    rl_metric = _coerce_optional_float(rl_run.get("best_metric"))
    sft_step = _coerce_optional_int(sft_run.get("best_step"))
    rl_step = _coerce_optional_int(rl_run.get("best_step"))
    sft_id = str(sft_run.get("resolved_finetune_id") or "").strip()
    rl_id = str(rl_run.get("resolved_finetune_id") or "").strip()
    if rl_id and (sft_metric is None or rl_metric is None or rl_metric >= sft_metric):
        return {
            "preferred_stage": "rl",
            "finetune_id": rl_id,
            "preferred_checkpoint_step": rl_step,
            "selection_reason": "rl_best_metric_not_worse_than_sft",
        }
    if sft_id:
        return {
            "preferred_stage": "sft",
            "finetune_id": sft_id,
            "preferred_checkpoint_step": sft_step,
            "selection_reason": "sft_metric_exceeds_rl_or_rl_missing",
        }
    return {
        "preferred_stage": "",
        "finetune_id": "",
        "preferred_checkpoint_step": None,
        "selection_reason": "no_successful_run",
    }


def _build_detect_benchmark_command(
    args: argparse.Namespace,
    *,
    detect_finetune_id: str,
    checkpoint_step: Optional[int],
) -> list[str]:
    command = [
        str(args.python_executable),
        "-m",
        "inspector_md.benchmark_inspector_detect",
        "--config",
        str(BENCHMARK_CONFIGS["detect"].resolve()),
        "--dataset-manifest",
        str(args.source_manifest),
        "--split",
        "test",
        "--max-samples",
        "0",
        "--detect-finetune-id",
        str(detect_finetune_id),
        "--output-json",
        str(Path(args.benchmark_output_root) / "inspector_detect.metrics.json"),
        "--predictions-jsonl",
        str(Path(args.benchmark_output_root) / "inspector_detect.records.jsonl"),
    ]
    if checkpoint_step is not None:
        command.extend(["--checkpoint-step", str(int(checkpoint_step))])
    return command


def _build_query_benchmark_command(
    args: argparse.Namespace,
    *,
    dataset_dir: Path,
    finetune_id: str,
    output_stem: str,
    reasoning: bool,
) -> list[str]:
    command = [
        str(args.python_executable),
        "-m",
        "inspector_md.benchmark_inspector_query",
        "--config",
        str(BENCHMARK_CONFIGS["query"].resolve()),
        "--dataset-dir",
        str(dataset_dir),
        "--split",
        "test",
        "--max-samples",
        "0",
        "--finetune-id",
        str(finetune_id),
        "--output-json",
        str(Path(args.benchmark_output_root) / f"{output_stem}.metrics.json"),
        "--predictions-jsonl",
        str(Path(args.benchmark_output_root) / f"{output_stem}.predictions.jsonl"),
    ]
    if bool(reasoning):
        command.append("--reasoning")
    return command


def _build_pipeline_benchmark_command(
    args: argparse.Namespace,
    *,
    detect_finetune_id: str,
    query_finetune_id: str,
) -> list[str]:
    return [
        str(args.python_executable),
        "-m",
        "inspector_md.benchmark_inspector_pipeline",
        "--config",
        str(BENCHMARK_CONFIGS["pipeline"].resolve()),
        "--dataset-manifest",
        str(args.source_manifest),
        "--split",
        "test",
        "--max-samples",
        "0",
        "--detect-finetune-id",
        str(detect_finetune_id),
        "--query-finetune-id",
        str(query_finetune_id),
        "--output-json",
        str(Path(args.benchmark_output_root) / "inspector_pipeline.metrics.json"),
        "--predictions-jsonl",
        str(Path(args.benchmark_output_root) / "inspector_pipeline.predictions.jsonl"),
    ]


def _run_build_stage(args: argparse.Namespace, summary: dict[str, Any]) -> dict[str, Any]:
    command = _build_dataset_command(args)
    log_path = Path(args.log_dir) / "build_dataset.log"
    result = _run_logged_command(command, log_path=log_path, cwd=REPO_ROOT)
    build_summary_path = Path(args.output_root) / "build_summary.json"
    result["build_summary_path"] = str(build_summary_path)
    if result["returncode"] == 0 and build_summary_path.is_file():
        result["build_summary"] = _read_json(build_summary_path, default={})
    summary["build"] = result
    return summary


def _run_readiness_stage(args: argparse.Namespace, summary: dict[str, Any]) -> dict[str, Any]:
    command = _readiness_command(args)
    log_path = Path(args.log_dir) / "readiness.log"
    result = _run_logged_command(command, log_path=log_path, cwd=REPO_ROOT)
    readiness_json_path = Path(args.readiness_output_json)
    result["output_json"] = str(readiness_json_path)
    if readiness_json_path.is_file():
        result["summary"] = _read_json(readiness_json_path, default={})
    summary["readiness"] = result
    return summary


def _run_training_stage(args: argparse.Namespace, summary: dict[str, Any]) -> dict[str, Any]:
    runs_summary = dict(summary.get("runs") or {})
    for spec in _train_specs(args):
        run_key = str(spec["key"])
        existing = dict(runs_summary.get(run_key) or {})
        if bool(args.skip_existing) and _stage_completed(MappingLike(existing)):
            continue
        dependency_key = str(spec.get("dependency") or "")
        dependency_finetune_id = ""
        if dependency_key:
            dependency = dict(runs_summary.get(dependency_key) or {})
            dependency_finetune_id = str(dependency.get("resolved_finetune_id") or "").strip()
            if not dependency_finetune_id:
                raise RuntimeError(f"Missing dependency finetune ID for {run_key}: {dependency_key}")
        resume_finetune_id = str(existing.get("resolved_finetune_id") or "").strip()
        finetune_id = resume_finetune_id or dependency_finetune_id
        command = _build_train_command(
            spec,
            dependency_finetune_id=finetune_id,
            python_executable=args.python_executable,
        )
        log_path = Path(args.log_dir) / str(spec["log_name"])
        result = _run_logged_command(command, log_path=log_path, cwd=REPO_ROOT)
        if run_key.startswith("query_"):
            query_summary = _load_query_train_summary(
                run_output_dir=Path(spec["run_output_dir"]),
                finetune_id=str(result.get("resolved_finetune_id") or ""),
            )
            if query_summary:
                result["train_summary"] = query_summary
                result["best_metric"] = _coerce_optional_float(query_summary.get("best_metric_value"))
        else:
            result.update(_parse_detect_point_training_log(log_path))
        runs_summary[run_key] = result
        summary["runs"] = runs_summary
        summary["updated_at"] = _now_iso()
        _save_summary(Path(args.summary_json), summary)
        if result["returncode"] != 0:
            raise RuntimeError(f"Training run failed: {run_key}")

    detect_candidate = _pick_detect_or_point_candidate(
        sft_run=dict(runs_summary.get("detect_sft") or {}),
        rl_run=dict(runs_summary.get("detect_rl") or {}),
    )
    point_candidate = _pick_detect_or_point_candidate(
        sft_run=dict(runs_summary.get("point_sft") or {}),
        rl_run=dict(runs_summary.get("point_rl") or {}),
    )
    summary["candidates"] = {
        "detect": detect_candidate,
        "point": point_candidate,
        "query": {
            "finetune_id": str(dict(runs_summary.get("query") or {}).get("resolved_finetune_id") or "").strip(),
        },
    }
    return summary


def _run_benchmark_stage(args: argparse.Namespace, summary: dict[str, Any]) -> dict[str, Any]:
    candidates = dict(summary.get("candidates") or {})
    detect_candidate = dict(candidates.get("detect") or {})
    detect_finetune_id = str(detect_candidate.get("finetune_id") or "").strip()
    detect_checkpoint_step = _coerce_optional_int(detect_candidate.get("preferred_checkpoint_step"))
    query_finetune_id = str(dict(candidates.get("query") or {}).get("finetune_id") or "").strip()
    if not detect_finetune_id or not query_finetune_id:
        raise RuntimeError("Missing detect or query finetune ID; benchmarks cannot run.")
    benchmark_specs = [
        (
            "detect",
            _build_detect_benchmark_command(
                args,
                detect_finetune_id=detect_finetune_id,
                checkpoint_step=detect_checkpoint_step,
            ),
            Path(args.benchmark_output_root) / "detect.log",
            Path(args.benchmark_output_root) / "inspector_detect.metrics.json",
        ),
        (
            "query",
            _build_query_benchmark_command(
                args,
                dataset_dir=Path(args.query_output_dir),
                finetune_id=query_finetune_id,
                output_stem="inspector_query_issues",
                reasoning=False,
            ),
            Path(args.benchmark_output_root) / "query.log",
            Path(args.benchmark_output_root) / "inspector_query_issues.metrics.json",
        ),
        (
            "pipeline",
            _build_pipeline_benchmark_command(
                args,
                detect_finetune_id=detect_finetune_id,
                query_finetune_id=query_finetune_id,
            ),
            Path(args.benchmark_output_root) / "pipeline.log",
            Path(args.benchmark_output_root) / "inspector_pipeline.metrics.json",
        ),
    ]
    benchmarks = dict(summary.get("benchmarks") or {})
    for key, command, log_path, metrics_path in benchmark_specs:
        if key == "query" and not query_finetune_id:
            benchmarks[key] = {"status": "skipped", "reason": "missing_finetune_id"}
            continue
        existing = dict(benchmarks.get(key) or {})
        if bool(args.skip_existing) and _stage_completed(MappingLike(existing)) and Path(existing.get("output_json") or "").is_file():
            continue
        result = _run_logged_command(command, log_path=log_path, cwd=REPO_ROOT)
        result["output_json"] = str(metrics_path)
        if metrics_path.is_file():
            result["metrics"] = _read_json(metrics_path, default={})
        benchmarks[key] = result
        summary["benchmarks"] = benchmarks
        summary["updated_at"] = _now_iso()
        _save_summary(Path(args.summary_json), summary)
        if result["returncode"] != 0:
            raise RuntimeError(f"Benchmark run failed: {key}")
    summary["benchmarks"] = benchmarks
    return summary


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    common.maybe_load_env_file(args.env_file, override=False)
    summary = _ensure_summary_base(args)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    Path(args.run_output_root).mkdir(parents=True, exist_ok=True)
    Path(args.benchmark_output_root).mkdir(parents=True, exist_ok=True)
    _save_summary(Path(args.summary_json), summary)

    if "build" in args.stages:
        if not (bool(args.skip_existing) and _stage_completed(MappingLike(summary.get("build") or {}))):
            summary = _run_build_stage(args, summary)
            _save_summary(Path(args.summary_json), summary)
            if not _stage_completed(MappingLike(summary.get("build") or {})):
                raise SystemExit(1)

    if "readiness" in args.stages:
        if not (bool(args.skip_existing) and _stage_completed(MappingLike(summary.get("readiness") or {}))):
            summary = _run_readiness_stage(args, summary)
            _save_summary(Path(args.summary_json), summary)
            if not _stage_completed(MappingLike(summary.get("readiness") or {})):
                raise SystemExit(1)

    if "train" in args.stages:
        summary = _run_training_stage(args, summary)
        _save_summary(Path(args.summary_json), summary)

    if "benchmark" in args.stages:
        summary = _run_benchmark_stage(args, summary)
        _save_summary(Path(args.summary_json), summary)

    summary["updated_at"] = _now_iso()
    _save_summary(Path(args.summary_json), summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
