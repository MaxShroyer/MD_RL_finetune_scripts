#!/usr/bin/env python3
"""Check whether Inspector MD datasets and configs are ready for finetuning."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

from inspector_md import common, openrouter_grader

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = common.repo_relative("configs", "check_inspector_finetune_readiness_default.json")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    raw_argv = list(argv) if argv is not None else list(os.sys.argv[1:])
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    pre_args, _ = pre_parser.parse_known_args(raw_argv)
    config_path = common.resolve_config_path(pre_args.config, script_dir=SCRIPT_DIR)
    config = common.load_json_config(config_path, default_path=DEFAULT_CONFIG_PATH)

    parser = argparse.ArgumentParser(description="Check Inspector MD finetuning readiness.")
    parser.add_argument("--config", default=str(config_path))
    parser.add_argument("--env-file", default=str(common.repo_relative(".env.staging")))
    parser.add_argument("--detect-dataset-path", default=str(common.repo_relative("outputs", "inspector_detect_v1")))
    parser.add_argument("--point-dataset-path", default=str(common.repo_relative("outputs", "inspector_point_v1")))
    parser.add_argument("--query-dataset-dir", default=str(common.repo_relative("outputs", "inspector_query_issues_v2")))
    parser.add_argument("--require-query-text-refresh", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verify-openrouter-judge", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grader-api-key", default="")
    parser.add_argument("--grader-api-key-env-var", default=openrouter_grader.DEFAULT_OPENROUTER_ENV_VAR)
    parser.add_argument("--grader-api-base", default=openrouter_grader.DEFAULT_OPENROUTER_API_BASE)
    parser.add_argument("--grader-model-id", default=openrouter_grader.DEFAULT_GRADER_MODEL)
    parser.add_argument("--grader-profile", default=openrouter_grader.DEFAULT_GRADER_PROFILE)
    parser.add_argument("--grader-rubric-version", default=openrouter_grader.DEFAULT_GRADER_RUBRIC_VERSION)
    parser.add_argument("--output-json", default=str(common.repo_relative("outputs", "inspector_finetune_readiness.json")))
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--emit-smoke-commands", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--launch-smoke", action="store_true")
    parser.add_argument("--smoke-detect-config", default=str(common.repo_relative("configs", "train_inspector_detect_sft_default.json")))
    parser.add_argument("--smoke-query-config", default=str(common.repo_relative("configs", "train_inspector_query_default.json")))
    parser.add_argument("--smoke-output-root", default=str(common.repo_relative("outputs", "runs", "inspector_smoke")))

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
    args.detect_dataset_path = common.resolve_inspector_dataset_path(
        args.detect_dataset_path,
        task="detect",
        module_root=SCRIPT_DIR,
    )[0]
    args.point_dataset_path = common.resolve_inspector_dataset_path(
        args.point_dataset_path,
        task="point",
        module_root=SCRIPT_DIR,
    )[0]
    args.query_dataset_dir = common.resolve_path(args.query_dataset_dir, module_root=SCRIPT_DIR)
    args.output_json = common.resolve_path(args.output_json, module_root=SCRIPT_DIR)
    args.smoke_detect_config = common.resolve_path(args.smoke_detect_config, module_root=SCRIPT_DIR)
    args.smoke_query_config = common.resolve_path(args.smoke_query_config, module_root=SCRIPT_DIR)
    args.smoke_output_root = common.resolve_path(args.smoke_output_root, module_root=SCRIPT_DIR)
    return args


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl_paths(dataset_dir: Path) -> list[Path]:
    jsonl_dir = dataset_dir / "jsonl"
    if not jsonl_dir.is_dir():
        return []
    return sorted(path for path in jsonl_dir.glob("*.jsonl") if path.is_file())


def _query_rows(dataset_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in _jsonl_paths(dataset_dir):
        rows.extend(common.load_jsonl(path))
    return rows


def _dataset_exists(path: Path) -> bool:
    return path.is_dir() and (path / "metadata.json").is_file()


def _hf_dataset_nonempty(path: Path, *, metadata_key: str) -> tuple[bool, dict[str, int]]:
    if not _dataset_exists(path):
        return False, {}
    metadata = _load_json(path / "metadata.json")
    counts = {
        str(split): int(count)
        for split, count in dict(metadata.get(metadata_key) or {}).items()
    }
    return bool(sum(counts.values()) > 0), counts


def _query_dataset_nonempty(path: Path, *, metadata_key: str) -> tuple[bool, dict[str, int], list[dict[str, Any]], dict[str, Any]]:
    if not _dataset_exists(path):
        return False, {}, [], {}
    metadata = _load_json(path / "metadata.json")
    counts = {
        str(split): int(count)
        for split, count in dict(metadata.get(metadata_key) or {}).items()
    }
    rows = _query_rows(path)
    return bool(sum(counts.values()) > 0 and rows), counts, rows, metadata


def _check_query_rows(rows: list[dict[str, Any]]) -> tuple[bool, dict[str, int]]:
    missing_question = 0
    missing_target_text = 0
    missing_target_format = 0
    missing_final_answer_json = 0
    non_empty_spatial_refs = 0
    for row in rows:
        if not str(row.get("question") or "").strip():
            missing_question += 1
        if not str(row.get("target_text") or "").strip():
            missing_target_text += 1
        if str(row.get("target_format") or "").strip() != "json_issue_list":
            missing_target_format += 1
        if not str(row.get("final_answer_json") or "").strip():
            missing_final_answer_json += 1
        if str(row.get("spatial_refs_json") or "").strip() not in {"", "[]"}:
            non_empty_spatial_refs += 1
    summary = {
        "missing_question_count": missing_question,
        "missing_target_text_count": missing_target_text,
        "missing_target_format_count": missing_target_format,
        "missing_final_answer_json_count": missing_final_answer_json,
        "non_empty_spatial_refs_count": non_empty_spatial_refs,
    }
    return not any(summary.values()), summary


def _check_teacher_cache(metadata: dict[str, Any], *, require_query_text_refresh: bool) -> tuple[bool, dict[str, Any]]:
    refresh_mode = str(metadata.get("query_text_refresh_mode") or "").strip().lower()
    cache_path_raw = str(metadata.get("query_text_cache_jsonl") or "").strip()
    cache_path = common.resolve_path(cache_path_raw, module_root=SCRIPT_DIR) if cache_path_raw else Path("")
    cache_exists = bool(cache_path and cache_path.is_file())
    cache_row_count = len(common.load_jsonl(cache_path)) if cache_exists else 0
    ok = True
    if require_query_text_refresh and refresh_mode != "openrouter":
        ok = False
    if require_query_text_refresh and (not cache_exists or cache_row_count <= 0):
        ok = False
    return ok, {
        "query_text_refresh_mode": refresh_mode,
        "query_text_cache_jsonl": str(cache_path) if cache_path_raw else "",
        "cache_exists": cache_exists,
        "cache_row_count": cache_row_count,
    }


def _check_openrouter_judge(args: argparse.Namespace) -> tuple[bool, dict[str, Any]]:
    if not bool(args.verify_openrouter_judge):
        return True, {"verified": False, "skipped": True}
    try:
        api_key = openrouter_grader.resolve_openrouter_api_key(
            explicit_api_key=args.grader_api_key,
            api_key_env_var=args.grader_api_key_env_var,
        )
    except Exception as exc:
        return False, {
            "verified": False,
            "error": str(exc),
            "grader_model_id": str(args.grader_model_id),
        }
    return True, {
        "verified": True,
        "grader_model_id": str(args.grader_model_id),
        "grader_profile": str(args.grader_profile),
        "grader_rubric_version": str(args.grader_rubric_version),
        "grader_api_base": str(args.grader_api_base),
        "api_key_present": bool(api_key),
    }


def _smoke_commands(args: argparse.Namespace) -> list[dict[str, Any]]:
    python_executable = str(args.python_executable or sys.executable)
    commands = [
        {
            "name": "detect_sft_smoke",
            "command": [
                python_executable,
                str((SCRIPT_DIR / "train_inspector_detect.py").resolve()),
                "--config",
                str(args.smoke_detect_config),
                "--num-steps",
                "2",
                "--sft-bootstrap-steps",
                "2",
                "--eval-every",
                "1",
                "--save-every",
                "1",
                "--eval-max-samples",
                "4",
                "--finetune-name",
                "inspector-detect-smoke",
            ],
            "expected_artifacts": ["train_summary.json", "eval_predictions"],
        },
        {
            "name": "query_sft_smoke",
            "command": [
                python_executable,
                str((SCRIPT_DIR / "train_inspector_query.py").resolve()),
                "--config",
                str(args.smoke_query_config),
                "--mode",
                "sft",
                "--sft-steps",
                "2",
                "--rl-steps",
                "0",
                "--eval-every",
                "1",
                "--save-every",
                "1",
                "--eval-max-samples",
                "4",
                "--finetune-name",
                "inspector-query-sft-smoke",
            ],
            "expected_artifacts": ["train_summary.json", "eval_predictions"],
        },
        {
            "name": "query_rl_smoke",
            "command": [
                python_executable,
                str((SCRIPT_DIR / "train_inspector_query.py").resolve()),
                "--config",
                str(args.smoke_query_config),
                "--mode",
                "rl",
                "--sft-steps",
                "0",
                "--rl-steps",
                "1",
                "--eval-every",
                "1",
                "--save-every",
                "1",
                "--eval-max-samples",
                "4",
                "--finetune-name",
                "inspector-query-rl-smoke",
            ],
            "expected_artifacts": ["train_summary.json", "eval_predictions", "judge_cache.jsonl"],
        },
    ]
    return commands


def evaluate_readiness(args: argparse.Namespace) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    detect_ok, detect_counts = _hf_dataset_nonempty(Path(args.detect_dataset_path), metadata_key="detect_split_counts")
    checks.append({"name": "detect_dataset", "ok": detect_ok, "details": {"path": str(args.detect_dataset_path), "split_counts": detect_counts}})

    point_ok, point_counts = _hf_dataset_nonempty(Path(args.point_dataset_path), metadata_key="point_split_counts")
    checks.append({"name": "point_dataset", "ok": point_ok, "details": {"path": str(args.point_dataset_path), "split_counts": point_counts}})

    query_ok, query_counts, query_rows, query_meta = _query_dataset_nonempty(
        Path(args.query_dataset_dir),
        metadata_key="query_split_counts",
    )
    query_rows_ok, query_row_summary = _check_query_rows(query_rows)
    checks.append(
        {
            "name": "query_dataset",
            "ok": query_ok and query_rows_ok,
            "details": {
                "path": str(args.query_dataset_dir),
                "split_counts": query_counts,
                "issue_box_counts": dict(query_meta.get("issue_box_counts") or {}),
                "missing_issue_codes": list(query_meta.get("missing_issue_codes") or []),
                "query_spatial_refs_nonempty_by_split": dict(query_meta.get("query_spatial_refs_nonempty_by_split") or {}),
                **query_row_summary,
            },
        }
    )

    teacher_cache_ok, teacher_cache_details = _check_teacher_cache(
        query_meta,
        require_query_text_refresh=bool(args.require_query_text_refresh),
    )
    checks.append({"name": "query_teacher_cache", "ok": teacher_cache_ok, "details": teacher_cache_details})

    judge_ok, judge_details = _check_openrouter_judge(args)
    checks.append({"name": "openrouter_judge", "ok": judge_ok, "details": judge_details})

    smoke_commands = _smoke_commands(args) if bool(args.emit_smoke_commands) else []
    ready = all(bool(item["ok"]) for item in checks)
    summary = {
        "ready": ready,
        "checks": checks,
        "smoke_commands": smoke_commands,
    }
    return summary


def launch_smoke(summary: dict[str, Any]) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for item in list(summary.get("smoke_commands") or []):
        command = list(item.get("command") or [])
        proc = subprocess.run(
            command,
            cwd=str(common.REPO_ROOT),
            env=os.environ.copy(),
            text=True,
            capture_output=True,
            check=False,
        )
        runs.append(
            {
                "name": str(item.get("name") or ""),
                "command": command,
                "returncode": int(proc.returncode),
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }
        )
        if proc.returncode != 0:
            break
    return runs


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    common.maybe_load_env_file(args.env_file, override=False)
    summary = evaluate_readiness(args)
    if bool(args.launch_smoke):
        if not bool(summary.get("ready")):
            common.write_json(Path(args.output_json), summary)
            raise SystemExit("Inspector MD is not ready for smoke finetune launches.")
        summary["smoke_runs"] = launch_smoke(summary)
        summary["ready"] = bool(summary.get("ready")) and all(
            int(item.get("returncode", 1)) == 0 for item in list(summary.get("smoke_runs") or [])
        )
    common.write_json(Path(args.output_json), summary)
    print(json.dumps(summary, indent=2))
    if not bool(summary.get("ready")):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
