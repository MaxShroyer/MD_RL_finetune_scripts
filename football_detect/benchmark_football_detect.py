#!/usr/bin/env python3
"""Benchmark football detect checkpoints via the public detect API."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finetune_checkpoints import resolve_checkpoint_step  # noqa: E402
from football_detect import train_football_detect as train_utils  # noqa: E402

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "benchmark_football_detect_best_general.json"
DEFAULT_BASE_URL = "https://api.moondream.ai/v1"


def _load_json_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        if path == DEFAULT_CONFIG_PATH:
            return {}
        raise FileNotFoundError(f"Config file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a JSON object: {path}")
    return payload


def _cfg_str(config: dict[str, Any], key: str, fallback: str) -> str:
    value = config.get(key, fallback)
    return str(value) if value is not None else fallback


def _cfg_int(config: dict[str, Any], key: str, fallback: int) -> int:
    value = config.get(key, fallback)
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _cfg_float(config: dict[str, Any], key: str, fallback: float) -> float:
    value = config.get(key, fallback)
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _cfg_optional_int(config: dict[str, Any], key: str, fallback: Optional[int]) -> Optional[int]:
    value = config.get(key, fallback)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _resolve_config_path(raw_path: str) -> Path:
    path = Path(str(raw_path or "")).expanduser()
    if path.is_absolute():
        return path
    from_cwd = (Path.cwd() / path).resolve()
    if from_cwd.exists():
        return from_cwd
    from_repo = (REPO_ROOT / path).resolve()
    if from_repo.exists():
        return from_repo
    from_script = (Path(__file__).resolve().parent / path).resolve()
    if from_script.exists():
        return from_script
    return from_cwd


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    raw_argv = list(argv) if argv is not None else list(sys.argv[1:])
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    pre_args, _ = pre_parser.parse_known_args(raw_argv)
    config_path = _resolve_config_path(pre_args.config)
    config = _load_json_config(config_path)

    parser = argparse.ArgumentParser(description="Benchmark football detect checkpoints.")
    parser.add_argument("--config", default=str(config_path))
    parser.add_argument("--env-file", default=_cfg_str(config, "env_file", ""))
    parser.add_argument("--api-key", default=_cfg_str(config, "api_key", ""))
    parser.add_argument("--api-key-env-var", default=_cfg_str(config, "api_key_env_var", "MOONDREAM_API_KEY"))
    parser.add_argument("--hf-token", default=_cfg_str(config, "hf_token", ""))
    parser.add_argument("--base-url", "--api-base", dest="base_url", default=_cfg_str(config, "base_url", _cfg_str(config, "api_base", "")))
    parser.add_argument("--dataset-path", default=_cfg_str(config, "dataset_path", ""))
    parser.add_argument("--dataset-name", default=_cfg_str(config, "dataset_name", train_utils.DEFAULT_SPLIT_DATASET_NAME))
    parser.add_argument("--split", default=_cfg_str(config, "split", "post_val"))
    parser.add_argument("--class-names-file", default=_cfg_str(config, "class_names_file", ""))
    parser.add_argument("--include-classes", nargs="*", default=None)
    parser.add_argument("--exclude-classes", nargs="*", default=None)
    parser.add_argument("--prompt-overrides-json", default=json.dumps(config.get("prompt_overrides_json", {})))
    parser.add_argument("--max-samples", type=_cfg_optional_int.__func__ if hasattr(_cfg_optional_int, "__func__") else int, default=None)
    parser.add_argument("--model", default=_cfg_str(config, "model", ""))
    parser.add_argument("--finetune-id", default=_cfg_str(config, "finetune_id", ""))
    parser.add_argument("--checkpoint-step", type=int, default=_cfg_int(config, "checkpoint_step", -1))
    parser.add_argument(
        "--checkpoint-fallback-policy",
        choices=["nearest_saved", "exact"],
        default=_cfg_str(config, "checkpoint_fallback_policy", "nearest_saved"),
    )
    parser.add_argument("--checkpoint-ready-max-wait-s", type=float, default=_cfg_float(config, "checkpoint_ready_max_wait_s", 0.0))
    parser.add_argument("--checkpoint-ready-poll-interval-s", type=float, default=_cfg_float(config, "checkpoint_ready_poll_interval_s", 5.0))
    parser.add_argument("--base-model", default=_cfg_str(config, "base_model", "moondream3-preview"))
    parser.add_argument("--temperature", type=float, default=_cfg_float(config, "temperature", 0.0))
    parser.add_argument("--top-p", type=float, default=_cfg_float(config, "top_p", 1.0))
    parser.add_argument("--max-tokens", type=int, default=_cfg_int(config, "max_tokens", 256))
    parser.add_argument("--max-objects", type=int, default=_cfg_int(config, "max_objects", 50))
    parser.add_argument("--timeout", type=float, default=_cfg_float(config, "timeout", 120.0))
    parser.add_argument("--output-json", "--out-json", dest="output_json", default=_cfg_str(config, "output_json", _cfg_str(config, "out_json", "")))
    parser.add_argument("--predictions-jsonl", default=_cfg_str(config, "predictions_jsonl", ""))
    parser.add_argument("--neg-prompts-per-empty", type=int, default=_cfg_int(config, "neg_prompts_per_empty", 0))
    parser.add_argument("--neg-prompts-per-nonempty", type=int, default=_cfg_int(config, "neg_prompts_per_nonempty", 1))
    parser.add_argument("--seed", type=int, default=_cfg_int(config, "seed", 42))
    args = parser.parse_args(raw_argv)

    args.config = str(_resolve_config_path(args.config))
    args.env_file = str(_resolve_config_path(args.env_file)) if str(args.env_file).strip() else ""
    args.prompt_overrides = train_utils._parse_prompt_overrides_json(args.prompt_overrides_json)
    if args.max_samples is None:
        raw_max_samples = config.get("max_samples", None)
        args.max_samples = _cfg_optional_int({"value": raw_max_samples}, "value", None)
    if args.max_samples is not None and int(args.max_samples) <= 0:
        args.max_samples = None
    if int(args.checkpoint_step) < 0:
        args.checkpoint_step = None
    else:
        args.checkpoint_step = int(args.checkpoint_step)
    args.include_classes = list(args.include_classes or [])
    args.exclude_classes = list(args.exclude_classes or [])
    return args


def _resolve_model(args: argparse.Namespace) -> str:
    if str(args.model).strip():
        return str(args.model).strip()
    if not str(args.finetune_id).strip():
        return str(args.base_model).strip() or "moondream3-preview"
    if args.checkpoint_step is None:
        return f"{str(args.base_model).rstrip('/')}/{str(args.finetune_id).strip()}"
    resolved_step, used_fallback = resolve_checkpoint_step(
        api_base=str(args.base_url),
        api_key=str(args.api_key),
        finetune_id=str(args.finetune_id).strip(),
        requested_step=int(args.checkpoint_step),
        fallback_policy=str(args.checkpoint_fallback_policy),
        ready_max_wait_s=float(args.checkpoint_ready_max_wait_s),
        ready_poll_interval_s=float(args.checkpoint_ready_poll_interval_s),
    )
    if used_fallback:
        print(
            f"warning: requested checkpoint step={int(args.checkpoint_step)} not available; "
            f"using nearest saved step={resolved_step}"
        )
    return f"{str(args.base_model).rstrip('/')}/{str(args.finetune_id).strip()}@{int(resolved_step)}"


def _resolve_eval_rows(
    *,
    dataset_path: str,
    dataset_name: str,
    split: str,
    hf_token: str,
) -> tuple[Iterable[dict], Iterable[Mapping[str, Any]]]:
    if str(dataset_path).strip():
        dataset_obj = train_utils._load_local_dataset_dict(str(dataset_path))
        if split not in dataset_obj:
            raise ValueError(f"split '{split}' not found in local dataset: {list(dataset_obj.keys())}")
        rows = list(train_utils._iter_local_rows_once(dataset_obj, split))
        return iter(rows), rows
    rows = train_utils._materialize_rows(train_utils._iter_hf_rows_once(str(dataset_name), split, hf_token))
    return iter(rows), rows


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    args = train_utils._resolve_runtime_env(args)
    if not args.base_url:
        args.base_url = DEFAULT_BASE_URL
    if not args.api_key:
        raise ValueError("MOONDREAM_API_KEY is required")

    use_local = bool(str(args.dataset_path or "").strip())
    if not use_local and not str(args.dataset_name or "").strip():
        raise ValueError("Provide --dataset-path or --dataset-name")

    eval_rows, class_source_rows = _resolve_eval_rows(
        dataset_path=str(args.dataset_path or ""),
        dataset_name=str(args.dataset_name or ""),
        split=str(args.split),
        hf_token=str(args.hf_token or ""),
    )
    all_class_names = train_utils._extract_class_names_from_file(str(args.class_names_file or ""))
    if not all_class_names:
        all_class_names = train_utils.discover_class_names(class_source_rows)
    if args.include_classes:
        include_set = set(args.include_classes)
        all_class_names = [name for name in all_class_names if name in include_set]
    if args.exclude_classes:
        exclude_set = set(args.exclude_classes)
        all_class_names = [name for name in all_class_names if name not in exclude_set]
    if not all_class_names:
        raise ValueError("No class names remain after applying include/exclude filters.")

    model = _resolve_model(args)
    metrics = train_utils._evaluate_api(
        model=model,
        eval_rows=eval_rows,
        all_class_names=all_class_names,
        prompt_overrides=args.prompt_overrides,
        rng=train_utils.random.Random(int(args.seed)),
        neg_prompts_per_empty=int(args.neg_prompts_per_empty),
        neg_prompts_per_nonempty=int(args.neg_prompts_per_nonempty),
        max_samples=int(args.max_samples or 0),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        max_tokens=int(args.max_tokens),
        max_objects=int(args.max_objects),
        api_base=str(args.base_url),
        api_key=str(args.api_key),
        timeout=float(args.timeout),
    )

    if str(args.predictions_jsonl or "").strip():
        predictions_path = Path(str(args.predictions_jsonl)).expanduser().resolve()
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        predictions_path.write_text("", encoding="utf-8")

    if str(args.output_json or "").strip():
        output_path = Path(str(args.output_json)).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    print(
        f"benchmark split={args.split} tasks={int(metrics.get('eval_tasks', 0))} "
        f"miou={float(metrics.get('eval_miou', 0.0)):.4f} "
        f"f1={float(metrics.get('eval_f1', 0.0)):.4f} "
        f"macro_f1={float(metrics.get('eval_f1_macro', 0.0)):.4f}"
    )


if __name__ == "__main__":
    main()
