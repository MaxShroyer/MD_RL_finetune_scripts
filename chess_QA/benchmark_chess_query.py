#!/usr/bin/env python3
"""Benchmark chess QA query checkpoints via the public query API."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover
    def load_dotenv(*_args: Any, **_kwargs: Any) -> bool:
        return False

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from construction_site import query_common as shared_query_common  # noqa: E402
from chess_QA import data_loader as dataset_loader  # noqa: E402
from chess_QA import train_chess_query_rl as train_utils  # noqa: E402

DEFAULT_BASE_URL = "https://api.moondream.ai/v1"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "bench" / "benchmark_chess_default.json"


def _load_json_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        if config_path == DEFAULT_CONFIG_PATH:
            return {}
        raise FileNotFoundError(f"Config file not found: {config_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a JSON object: {config_path}")
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


def _cfg_bool(config: dict[str, Any], key: str, fallback: bool) -> bool:
    value = config.get(key, fallback)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y", "on"}:
            return True
        if lowered in {"false", "0", "no", "n", "off"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
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


def _normalize_task_types(raw_values: Optional[list[str]]) -> list[str]:
    if not raw_values:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for raw_value in raw_values:
        for piece in str(raw_value).split(","):
            task_type = piece.strip()
            if not task_type:
                continue
            task_type = train_utils.normalize_task_type(task_type)
            if task_type in seen:
                continue
            seen.add(task_type)
            out.append(task_type)
    return out


def _progress_enabled(no_progress: bool) -> bool:
    return (not bool(no_progress)) and sys.stderr.isatty()


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    raw_argv = list(argv) if argv is not None else list(sys.argv[1:])
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    pre_args, _ = pre_parser.parse_known_args(raw_argv)
    config_path = _resolve_config_path(pre_args.config)
    config = _load_json_config(config_path)

    parser = argparse.ArgumentParser(description="Benchmark chess QA query models.")
    parser.add_argument("--config", default=str(config_path))
    parser.add_argument("--env-file", default=_cfg_str(config, "env_file", str(Path(__file__).resolve().parent / ".env")))
    parser.add_argument("--api-key", default=_cfg_str(config, "api_key", ""))
    parser.add_argument("--base-url", default=_cfg_str(config, "base_url", ""))
    parser.add_argument(
        "--dataset-source",
        choices=sorted(dataset_loader.SUPPORTED_DATASET_SOURCES),
        default=_cfg_str(config, "dataset_source", dataset_loader.DEFAULT_DATASET_SOURCE),
    )
    parser.add_argument("--dataset-variant-tag", default=_cfg_str(config, "dataset_variant_tag", dataset_loader.DEFAULT_DATASET_VARIANT_TAG))
    parser.add_argument("--dataset-dir", default=_cfg_str(config, "dataset_dir", str(Path(__file__).resolve().parent / "synth-chess-dataset" / "outputs")))
    parser.add_argument("--hf-dataset-repo-id", default=_cfg_str(config, "hf_dataset_repo_id", dataset_loader.DEFAULT_HF_DATASET_REPO_ID))
    parser.add_argument("--hf-dataset-revision", default=_cfg_str(config, "hf_dataset_revision", dataset_loader.DEFAULT_HF_DATASET_REVISION))
    parser.add_argument("--hf-token", default=_cfg_str(config, "hf_token", ""))
    parser.add_argument("--hf-cache-dir", default=_cfg_str(config, "hf_cache_dir", ""))
    parser.add_argument("--split", default=_cfg_str(config, "split", "test"))
    parser.add_argument("--seed", type=int, default=_cfg_int(config, "seed", 42))
    parser.add_argument("--max-samples", type=int, default=_cfg_int(config, "max_samples", 0))
    parser.add_argument("--task-types", nargs="*", default=None)
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
    parser.add_argument("--temperature", type=float, default=_cfg_float(config, "temperature", 0.0))
    parser.add_argument("--top-p", type=float, default=_cfg_float(config, "top_p", 1.0))
    parser.add_argument("--max-tokens", type=int, default=_cfg_int(config, "max_tokens", 256))
    parser.add_argument("--timeout", type=float, default=_cfg_float(config, "timeout", 60.0))
    reasoning_group = parser.add_mutually_exclusive_group()
    reasoning_group.add_argument("--reasoning", dest="reasoning", action="store_true")
    reasoning_group.add_argument("--no-reasoning", dest="reasoning", action="store_false")
    parser.set_defaults(reasoning=_cfg_bool(config, "reasoning", False))
    parser.add_argument("--retry-429-max-retries", type=int, default=_cfg_int(config, "retry_429_max_retries", 2))
    parser.add_argument("--retry-429-backoff-s", type=float, default=_cfg_float(config, "retry_429_backoff_s", 1.0))
    parser.add_argument("--retry-429-max-backoff-s", type=float, default=_cfg_float(config, "retry_429_max_backoff_s", 8.0))
    parser.add_argument("--retry-5xx-max-retries", type=int, default=_cfg_int(config, "retry_5xx_max_retries", 2))
    parser.add_argument("--retry-5xx-backoff-s", type=float, default=_cfg_float(config, "retry_5xx_backoff_s", 2.0))
    parser.add_argument("--retry-5xx-max-backoff-s", type=float, default=_cfg_float(config, "retry_5xx_max_backoff_s", 16.0))
    parser.add_argument("--image-jpeg-quality", type=int, default=_cfg_int(config, "image_jpeg_quality", 92))
    parser.add_argument("--list-piece-reward-mode", choices=list(train_utils.LIST_PIECE_REWARD_MODES), default=_cfg_str(config, "list_piece_reward_mode", train_utils.DEFAULT_LIST_PIECE_REWARD_MODE))
    parser.add_argument("--output-json", default=_cfg_str(config, "output_json", ""))
    parser.add_argument("--predictions-jsonl", default=_cfg_str(config, "predictions_jsonl", ""))
    parser.add_argument("--skip-preflight", action="store_true", default=_cfg_bool(config, "skip_preflight", False))
    parser.add_argument("--no-progress", action="store_true", default=_cfg_bool(config, "no_progress", False))
    args = parser.parse_args(raw_argv)

    args.config = str(_resolve_config_path(args.config))
    args.env_file = str(_resolve_config_path(args.env_file))
    args.task_types = _normalize_task_types(args.task_types)
    if int(args.checkpoint_step) < 0:
        args.checkpoint_step = None
    else:
        args.checkpoint_step = int(args.checkpoint_step)
    if args.max_samples <= 0:
        args.max_samples = None
    return args


def _validate_args(args: argparse.Namespace) -> None:
    args.dataset_source = dataset_loader.normalize_dataset_source(args.dataset_source)
    args.dataset_variant_tag = dataset_loader.normalize_dataset_variant_tag(args.dataset_variant_tag)
    if not (0.0 <= args.temperature <= 2.0):
        raise ValueError("--temperature must be in [0,2]")
    if not (0.0 < args.top_p <= 1.0):
        raise ValueError("--top-p must be in (0,1]")
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be > 0")
    if args.timeout <= 0.0:
        raise ValueError("--timeout must be > 0")
    if args.dataset_source == "local_jsonl" and not str(args.dataset_dir).strip():
        raise ValueError("--dataset-dir is required when --dataset-source=local_jsonl")
    if args.dataset_source == "hf_hub" and not str(args.hf_dataset_repo_id).strip():
        raise ValueError("--hf-dataset-repo-id is required when --dataset-source=hf_hub")


@dataclass(frozen=True)
class _FakeOutput:
    answer: str


@dataclass(frozen=True)
class _FakeRollout:
    output: _FakeOutput


@dataclass(frozen=True)
class _FakeResult:
    rollouts: list[_FakeRollout]


class _APIFinetune:
    def __init__(self, *, args: argparse.Namespace, model: str) -> None:
        self.args = args
        self.model = model

    def _query_request(self, request: Any) -> _FakeResult:
        try:
            answer_text, _, _ = shared_query_common.call_query_api(
                api_base=self.args.base_url,
                api_key=self.args.api_key,
                model=self.model,
                question=str(getattr(request, "question", "")),
                image_url=str(getattr(request, "image_url", "")),
                temperature=float(getattr(getattr(request, "settings", None), "temperature", self.args.temperature)),
                top_p=float(getattr(getattr(request, "settings", None), "top_p", self.args.top_p)),
                max_tokens=int(getattr(getattr(request, "settings", None), "max_tokens", self.args.max_tokens)),
                reasoning=bool(getattr(request, "reasoning", self.args.reasoning)),
                timeout=float(self.args.timeout),
                retry_429_max_retries=int(self.args.retry_429_max_retries),
                retry_429_backoff_s=float(self.args.retry_429_backoff_s),
                retry_429_max_backoff_s=float(self.args.retry_429_max_backoff_s),
                retry_5xx_max_retries=int(self.args.retry_5xx_max_retries),
                retry_5xx_backoff_s=float(self.args.retry_5xx_backoff_s),
                retry_5xx_max_backoff_s=float(self.args.retry_5xx_max_backoff_s),
            )
        except Exception as exc:
            print(f"benchmark request failed: {shared_query_common.error_message(exc)}")
            return _FakeResult(rollouts=[])
        return _FakeResult(rollouts=[_FakeRollout(output=_FakeOutput(answer=str(answer_text or "")))])

    def rollouts_batch(self, *, requests: list[Any], num_rollouts: int, max_workers: int) -> list[_FakeResult]:
        del num_rollouts
        if not requests:
            return []
        results: list[Optional[_FakeResult]] = [None] * len(requests)
        worker_count = max(1, min(int(max_workers), len(requests)))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(self._query_request, request): idx
                for idx, request in enumerate(requests)
            }
            for future in as_completed(future_map):
                idx = future_map[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:  # pragma: no cover
                    print(f"benchmark request worker failed: {type(exc).__name__}: {exc}")
                    results[idx] = _FakeResult(rollouts=[])
        return [result if result is not None else _FakeResult(rollouts=[]) for result in results]


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    load_dotenv(args.env_file, override=False)
    if not args.api_key:
        args.api_key = os.environ.get("MOONDREAM_API_KEY", "")
    if not args.base_url:
        args.base_url = os.environ.get("TUNA_BASE_URL") or DEFAULT_BASE_URL
    _validate_args(args)
    if not args.api_key:
        raise ValueError("MOONDREAM_API_KEY is required")
    args.hf_token = dataset_loader.resolve_hf_token(args.hf_token)

    try:
        model_resolution = shared_query_common.resolve_query_inference_model(
            api_base=args.base_url,
            api_key=args.api_key,
            model=args.model,
            finetune_id=args.finetune_id,
            checkpoint_step=args.checkpoint_step,
            timeout=args.timeout,
            fallback_policy=str(args.checkpoint_fallback_policy),
            checkpoint_ready_max_wait_s=float(args.checkpoint_ready_max_wait_s),
            checkpoint_ready_poll_interval_s=float(args.checkpoint_ready_poll_interval_s),
        )
    except ValueError as exc:
        raise SystemExit(str(exc))
    model = model_resolution.model

    if not args.skip_preflight:
        try:
            shared_query_common.preflight_query_api(
                api_base=args.base_url,
                api_key=args.api_key,
                model=model,
                timeout=args.timeout,
                reasoning=args.reasoning,
                retry_429_max_retries=args.retry_429_max_retries,
                retry_429_backoff_s=args.retry_429_backoff_s,
                retry_429_max_backoff_s=args.retry_429_max_backoff_s,
                retry_5xx_max_retries=args.retry_5xx_max_retries,
                retry_5xx_backoff_s=args.retry_5xx_backoff_s,
                retry_5xx_max_backoff_s=args.retry_5xx_max_backoff_s,
            )
        except Exception as exc:
            raise SystemExit(
                f"query preflight failed for model={model}. details={shared_query_common.error_message(exc)}"
            )

    dataset_dir: Optional[Path] = None
    if args.dataset_source == "local_jsonl":
        dataset_dir = Path(args.dataset_dir).expanduser().resolve()

    examples = train_utils._load_split_examples(
        split_name=args.split,
        dataset_source=args.dataset_source,
        dataset_variant_tag=args.dataset_variant_tag,
        dataset_dir=dataset_dir,
        hf_dataset_repo_id=args.hf_dataset_repo_id,
        hf_dataset_revision=args.hf_dataset_revision,
        hf_token=args.hf_token,
        hf_cache_dir=args.hf_cache_dir,
    )
    if args.task_types:
        allowed_tasks = set(args.task_types)
        examples = [example for example in examples if example.task_type in allowed_tasks]
    if not examples:
        raise ValueError("No benchmark examples remain after task filtering.")

    predictions_path = Path(args.predictions_jsonl).expanduser().resolve() if args.predictions_jsonl else None
    predictions_output_dir = predictions_path.parent if predictions_path is not None else None
    checkpoint_tag = int(args.checkpoint_step or 0)
    fake_finetune = _APIFinetune(args=args, model=model)

    metrics = train_utils._evaluate_split(
        finetune=fake_finetune,
        examples=examples,
        split_name=args.split,
        seed=int(args.seed),
        batch_size=1,
        max_workers=8,
        max_samples=args.max_samples,
        rollout_retries=0,
        rollout_retry_backoff_s=1.0,
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        max_tokens=int(args.max_tokens),
        max_tokens_by_task=train_utils.DEFAULT_MAX_TOKENS_BY_TASK,
        reasoning=bool(args.reasoning),
        image_jpeg_quality=int(args.image_jpeg_quality),
        show_progress=_progress_enabled(args.no_progress),
        list_piece_reward_mode=str(args.list_piece_reward_mode),
        list_piece_reward_weights=None,
        fixed_indices=None,
        save_predictions=predictions_output_dir is not None,
        predictions_output_dir=predictions_output_dir,
        step_for_log=checkpoint_tag,
        stage_label="benchmark",
    )

    if predictions_path is not None:
        generated_predictions_path = train_utils._eval_predictions_path(
            output_dir=predictions_output_dir,
            stage_label="benchmark",
            split_name=args.split,
            step_for_log=checkpoint_tag,
        )
        if generated_predictions_path.exists() and generated_predictions_path != predictions_path:
            predictions_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(generated_predictions_path), str(predictions_path))
        elif not predictions_path.exists():
            predictions_path.parent.mkdir(parents=True, exist_ok=True)
            predictions_path.touch()

    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    print(
        f"benchmark split={args.split} samples={int(metrics.get('eval_samples', 0.0))} "
        f"reward={float(metrics.get('eval_reward_mean', 0.0)):.4f} "
        f"parse={float(metrics.get('eval_json_parse_rate', 0.0)):.4f} "
        f"exact={float(metrics.get('eval_exact_accuracy', 0.0)):.4f}"
    )


if __name__ == "__main__":
    main()
