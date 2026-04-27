#!/usr/bin/env python3
"""Run the staged Inspector MD runtime pipeline on one image."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

from inspector_md import common
from inspector_md.moondream_client import MoondreamInspectorClient
from inspector_md import openrouter_grader
from inspector_md.pipeline import InspectorPipeline, PipelineModels, PipelineSettings, report_to_punch_list_text
from inspector_md.task_schema import InspectionRequest

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = common.repo_relative("configs", "run_inspector_pipeline_default.json")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    raw_argv = list(argv) if argv is not None else list(os.sys.argv[1:])
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    pre_args, _ = pre_parser.parse_known_args(raw_argv)
    config_path = common.resolve_config_path(pre_args.config, script_dir=SCRIPT_DIR)
    config = common.load_json_config(config_path, default_path=DEFAULT_CONFIG_PATH)
    single_key_requested = "--api-key-env-var" in raw_argv or (
        "api_key_env_var" in config and "--api-key-env-vars" not in raw_argv
    )
    multi_key_requested = "--api-key-env-vars" in raw_argv or (
        "api_key_env_vars" in config and "--api-key-env-var" not in raw_argv
    )

    parser = argparse.ArgumentParser(description="Run the Inspector MD staged pipeline.")
    parser.add_argument("--config", default=str(config_path))
    parser.add_argument("--env-file", default=str(common.repo_relative(".env.staging")))
    parser.add_argument("--api-key", default="")
    parser.add_argument("--api-key-env-var", default=common.DEFAULT_API_KEY_ENV_VAR)
    parser.add_argument("--api-key-env-vars", nargs="+", default=list(common.DEFAULT_API_KEY_ENV_VARS))
    parser.add_argument("--base-url", default=common.DEFAULT_BASE_URL)
    parser.add_argument("--image-path", default="")
    parser.add_argument(
        "--inspection-request",
        default="Inspect this building image for visible exterior and site issues.",
    )
    parser.add_argument("--asset-context", default="")
    parser.add_argument("--detect-finetune-id", default="")
    parser.add_argument("--query-finetune-id", default="")
    parser.add_argument("--base-model", default=common.DEFAULT_BASE_MODEL)
    parser.add_argument("--proposal-temperature", type=float, default=0.0)
    parser.add_argument("--proposal-top-p", type=float, default=1.0)
    parser.add_argument("--proposal-max-tokens", type=int, default=512)
    parser.add_argument("--finding-temperature", type=float, default=0.0)
    parser.add_argument("--finding-top-p", type=float, default=1.0)
    parser.add_argument("--finding-max-tokens", type=int, default=384)
    parser.add_argument("--detect-temperature", type=float, default=0.0)
    parser.add_argument("--detect-top-p", type=float, default=1.0)
    parser.add_argument("--detect-max-tokens", type=int, default=256)
    parser.add_argument("--detect-max-objects", type=int, default=24)
    parser.add_argument("--iou-merge-threshold", type=float, default=0.5)
    parser.add_argument("--reasoning", action="store_true")
    parser.add_argument("--proposal-prompt-style", choices=("structured", "request_only"), default="request_only")
    parser.add_argument("--finding-prompt-style", choices=("structured", "minimal"), default="minimal")
    parser.add_argument("--query-normalization-mode", choices=("local_only", "openrouter_fallback"), default="local_only")
    parser.add_argument("--grader-api-key", default="")
    parser.add_argument("--grader-api-key-env-var", default=openrouter_grader.DEFAULT_OPENROUTER_ENV_VAR)
    parser.add_argument("--grader-api-base", default=openrouter_grader.DEFAULT_OPENROUTER_API_BASE)
    parser.add_argument("--grader-model-id", default=openrouter_grader.DEFAULT_GRADER_MODEL)
    parser.add_argument("--grader-profile", default=openrouter_grader.DEFAULT_GRADER_PROFILE)
    parser.add_argument("--grader-rubric-version", default=openrouter_grader.DEFAULT_GRADER_RUBRIC_VERSION)
    parser.add_argument("--grader-timeout", type=float, default=60.0)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--output-json", default=str(common.repo_relative("outputs", "reports", "last_report.json")))
    parser.add_argument("--output-text", default=str(common.repo_relative("outputs", "reports", "last_punch_list.txt")))
    parser.add_argument("--trace-json", default=str(common.repo_relative("outputs", "reports", "last_trace.json")))

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
    args.output_json = common.resolve_path(args.output_json, module_root=SCRIPT_DIR)
    args.output_text = common.resolve_path(args.output_text, module_root=SCRIPT_DIR)
    args.trace_json = common.resolve_path(args.trace_json, module_root=SCRIPT_DIR)
    if single_key_requested and not multi_key_requested:
        args.api_key_env_vars = [args.api_key_env_var]
    else:
        args.api_key_env_vars = common.normalize_api_key_env_vars(args.api_key_env_vars)
    if not str(args.image_path or "").strip():
        raise SystemExit("--image-path is required")
    args.image_path = str(common.resolve_path(args.image_path, module_root=SCRIPT_DIR))
    return args


def _resolve_model(base_model: str, finetune_id: str) -> str:
    finetune = str(finetune_id or "").strip()
    return f"{base_model}/{finetune}" if finetune else str(base_model).strip()


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    common.maybe_load_env_file(args.env_file, override=False)
    api_key_pool = common.resolve_api_key_pool(explicit_api_key=args.api_key, api_key_env_vars=args.api_key_env_vars)
    print(f"preflight base_url={args.base_url} key_slots={api_key_pool.env_var_names or ['<explicit>']}")
    client = MoondreamInspectorClient(api_key_pool=api_key_pool, base_url=args.base_url, timeout=args.timeout)
    normalizer = None
    if str(args.query_normalization_mode).strip().lower() == "openrouter_fallback":
        grader_api_key = openrouter_grader.resolve_openrouter_api_key(
            explicit_api_key=args.grader_api_key,
            api_key_env_var=args.grader_api_key_env_var,
        )
        normalizer = openrouter_grader.OpenRouterGrader(
            api_key=grader_api_key,
            model_id=args.grader_model_id,
            api_base=args.grader_api_base,
            timeout=float(args.grader_timeout),
            profile=args.grader_profile,
            rubric_version=args.grader_rubric_version,
        )
    pipeline = InspectorPipeline(
        client=client,
        models=PipelineModels(
            detect_model=_resolve_model(args.base_model, args.detect_finetune_id),
            query_model=_resolve_model(args.base_model, args.query_finetune_id),
            point_model=_resolve_model(args.base_model, args.detect_finetune_id),
        ),
        settings=PipelineSettings(
            proposal_temperature=args.proposal_temperature,
            proposal_top_p=args.proposal_top_p,
            proposal_max_tokens=args.proposal_max_tokens,
            finding_temperature=args.finding_temperature,
            finding_top_p=args.finding_top_p,
            finding_max_tokens=args.finding_max_tokens,
            detect_temperature=args.detect_temperature,
            detect_top_p=args.detect_top_p,
            detect_max_tokens=args.detect_max_tokens,
            detect_max_objects=args.detect_max_objects,
            iou_merge_threshold=args.iou_merge_threshold,
            reasoning=bool(args.reasoning),
            proposal_prompt_style=str(args.proposal_prompt_style),
            finding_prompt_style=str(args.finding_prompt_style),
        ),
        normalizer=normalizer,
    )
    report = pipeline.run(
        InspectionRequest(
            image_path=args.image_path,
            inspection_request=args.inspection_request,
            asset_context=args.asset_context,
        )
    )
    report.summary.update(
        {
            "base_url": args.base_url,
            "api_key_env_vars": api_key_pool.env_var_names,
            "api_key_slot_count": len(api_key_pool.slots),
            "query_normalization_mode": str(args.query_normalization_mode),
            "grader_model_id": normalizer.model_id if normalizer is not None else "",
        }
    )
    report.trace["runtime"] = {
        "base_url": args.base_url,
        "api_key_env_vars": api_key_pool.env_var_names,
        "api_key_slot_count": len(api_key_pool.slots),
        "query_normalization_mode": str(args.query_normalization_mode),
        "grader_model_id": normalizer.model_id if normalizer is not None else "",
    }
    common.write_json(Path(args.output_json), report.to_payload())
    common.write_text(Path(args.output_text), report_to_punch_list_text(report))
    common.write_json(Path(args.trace_json), report.trace)
    print(f"saved report: {args.output_json}")
    print(f"saved punch list: {args.output_text}")
    print(f"saved trace: {args.trace_json}")


if __name__ == "__main__":
    main()
