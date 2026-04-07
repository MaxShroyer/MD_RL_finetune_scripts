#!/usr/bin/env python3
"""Class-conditional RL finetuning for bone fracture detection."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from football_detect import train_football_detect as _base

from bone_fracture.common import (
    DEFAULT_DETECT_HF_DATASET_NAME,
    DEFAULT_DETECT_WANDB_PROJECT,
    DEFAULT_STAGING_API_BASE,
    config_to_cli_args,
    default_prompt_for_class,
    discover_class_names,
    load_json_config,
    normalize_class_name,
    parse_box_element_annotations,
    repo_relative,
    resolve_config_path,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = repo_relative("configs", "train_bone_fracture_detect_default.json")

_base.default_prompt_for_class = default_prompt_for_class
_base.discover_class_names = discover_class_names
_base.normalize_class_name = normalize_class_name
_base.parse_box_element_annotations = parse_box_element_annotations

ClassBox = _base.ClassBox
BaseSample = _base.BaseSample
TaskSample = _base.TaskSample
AugmentConfig = _base.AugmentConfig
UsageStats = _base.UsageStats
parse_box_element_annotations = parse_box_element_annotations
augment_task_sample = _base.augment_task_sample
tasks_from_base_sample = _base.tasks_from_base_sample
_box_from_normalized = _base._box_from_normalized
_to_base_sample = _base._to_base_sample


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    raw_argv = list(argv) if argv is not None else list(os.sys.argv[1:])
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    pre_args, _ = pre_parser.parse_known_args(raw_argv)
    config_path = resolve_config_path(pre_args.config, script_dir=SCRIPT_DIR)
    config = load_json_config(config_path, default_path=DEFAULT_CONFIG_PATH)

    parser = argparse.ArgumentParser(description="RL finetune Moondream for bone fracture detection.")
    parser.add_argument("--config", default=str(config_path))
    parser.add_argument("--env-file", default=str(repo_relative(".env")))
    parser.add_argument("--api-key", default="")
    parser.add_argument("--api-key-env-var", default="MOONDREAM_API_KEY")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"))
    parser.add_argument("--base-url", default="")

    parser.add_argument(
        "--dataset-path",
        default=str(repo_relative("outputs", "maxs-m87_bone_fracture_detect_v1")),
    )
    parser.add_argument("--dataset-name", default=DEFAULT_DETECT_HF_DATASET_NAME)
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--val-split",
        default="validation",
        help="Validation split name. If omitted, uses validation/val/dev/test/post_val when present.",
    )
    parser.add_argument(
        "--test-split",
        default="test",
        help="Held-out test split name. If omitted, uses test/post_val when present.",
    )
    parser.add_argument("--class-names-file", default="")
    parser.add_argument("--include-classes", nargs="*", default=None)
    parser.add_argument("--exclude-classes", nargs="*", default=None)
    parser.add_argument("--prompt-overrides-json", default="{}")

    parser.add_argument("--finetune-id", default="")
    parser.add_argument("--finetune-name", default="")
    parser.add_argument("--rank", type=int, default=16)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--buffer-size", type=int, default=100)
    parser.add_argument("--num-steps", type=int, default=300)
    parser.add_argument("--resume-step", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--rollout-retries", type=int, default=2)
    parser.add_argument("--rollout-retry-backoff-s", type=float, default=1.0)

    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-objects", type=int, default=50)
    parser.add_argument(
        "--reward-metric",
        choices=["f1", "miou"],
        default="f1",
        help="Training reward metric for detect rollouts.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=_base.SELECTION_METRIC_CHOICES,
        default="f1",
        help="Validation metric used to select the best checkpoint. Defaults to --reward-metric.",
    )

    reasoning_group = parser.add_mutually_exclusive_group()
    reasoning_group.add_argument(
        "--reasoning",
        dest="reasoning",
        action="store_true",
        help="Enable Moondream reasoning for training rollout requests.",
    )
    reasoning_group.add_argument(
        "--no-reasoning",
        dest="reasoning",
        action="store_false",
        help="Disable Moondream reasoning for training rollout requests.",
    )
    parser.set_defaults(reasoning=False)

    eval_reasoning_group = parser.add_mutually_exclusive_group()
    eval_reasoning_group.add_argument(
        "--eval-reasoning",
        dest="eval_reasoning",
        action="store_true",
        help="Force reasoning=true for eval requests.",
    )
    eval_reasoning_group.add_argument(
        "--no-eval-reasoning",
        dest="eval_reasoning",
        action="store_false",
        help="Force reasoning=false for eval requests.",
    )
    eval_reasoning_group.add_argument(
        "--eval-reasoning-inherit",
        dest="eval_reasoning",
        action="store_const",
        const=None,
        help="Inherit eval reasoning from --reasoning.",
    )
    parser.set_defaults(eval_reasoning=None)

    parser.add_argument("--off-policy", action="store_true")
    parser.add_argument(
        "--allow-off-policy-with-reasoning",
        action="store_true",
        help="Allow off-policy GT injection even when --reasoning is enabled.",
    )
    parser.add_argument("--off-policy-std-thresh", type=float, default=0.02)
    parser.add_argument("--off-policy-max-reward", type=float, default=0.15)
    parser.add_argument("--off-policy-min-reward", type=float, default=0.15)
    parser.add_argument("--off-policy-reward-scale", type=float, default=2.0)
    parser.add_argument(
        "--fn-penalty-exponent",
        type=float,
        default=1.0,
        help="Exponent for false negatives in reward denominator via FN^exp.",
    )
    parser.add_argument(
        "--fp-penalty-exponent",
        type=float,
        default=1.0,
        help="Exponent for false positives in reward denominator via FP^exp.",
    )

    parser.add_argument("--neg-prompts-per-empty", type=int, default=0)
    parser.add_argument("--neg-prompts-per-nonempty", type=int, default=1)
    parser.add_argument(
        "--pos-task-prob",
        type=float,
        default=0.95,
        help="When an image has positive tasks, choose a positive task with this probability.",
    )
    parser.add_argument(
        "--neg-reward-weight",
        type=float,
        default=0.5,
        help="Scale factor applied to rewards for negative tasks (no GT boxes).",
    )
    parser.add_argument("--augment-prob", type=float, default=0.5)
    parser.add_argument("--eval-temperature", type=float, default=0.0)
    parser.add_argument("--eval-top-p", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-max-samples", type=int, default=200)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument(
        "--kl-warning-threshold",
        type=float,
        default=0.0,
        help="Log a warning when a train step KL reaches this threshold. <=0 disables warnings.",
    )
    parser.add_argument(
        "--kl-stop-threshold",
        type=float,
        default=0.0,
        help="Stop training early when train-step KL reaches this threshold for N consecutive updates. <=0 disables stopping.",
    )
    parser.add_argument(
        "--kl-stop-consecutive",
        type=int,
        default=1,
        help="How many consecutive KL threshold hits are required before early stop.",
    )
    parser.add_argument(
        "--run-final-test",
        action="store_true",
        help="Evaluate the best validation checkpoint once on the held-out test split after training.",
    )

    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument(
        "--usage-report-every",
        type=int,
        default=30,
        help="Print running dataset/task usage stats every N steps (<=0 disables periodic usage logs).",
    )
    parser.add_argument(
        "--usage-top-k",
        type=int,
        default=30,
        help="How many sources/classes to include in usage summaries.",
    )
    parser.add_argument("--wandb-project", default=DEFAULT_DETECT_WANDB_PROJECT)
    parser.add_argument("--wandb-run-name", default="")

    option_to_dest: dict[str, str] = {}
    for action in parser._actions:
        if not action.option_strings:
            continue
        for opt in action.option_strings:
            option_to_dest[opt] = action.dest
    overridden_dests = {option_to_dest[arg] for arg in raw_argv if arg in option_to_dest}
    config_cli_args = config_to_cli_args(
        parser,
        config,
        config_path=config_path,
        overridden_dests=overridden_dests,
    )
    args = parser.parse_args(config_cli_args + raw_argv)
    args.config = str(resolve_config_path(args.config, script_dir=SCRIPT_DIR))
    if not str(args.base_url).strip():
        args.base_url = DEFAULT_STAGING_API_BASE
    args.include_classes = list(args.include_classes or [])
    args.exclude_classes = list(args.exclude_classes or [])
    args.prompt_overrides = _base._parse_prompt_overrides_json(args.prompt_overrides_json)
    if not args.finetune_id and not args.finetune_name:
        args.finetune_name = f"bone-fracture-detect-{_base._random_suffix()}"
    args.async_checkpoint_eval_benchmark_script = str(
        (SCRIPT_DIR / "benchmark_bone_fracture_detect.py").resolve()
    )
    return args


_base.parse_args = parse_args


def main(argv: Optional[list[str]] = None) -> None:
    _base.main(argv)


if __name__ == "__main__":
    main()
