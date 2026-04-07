#!/usr/bin/env python3
"""Benchmark omega detect checkpoints via the public detect API."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finetune_checkpoints import resolve_checkpoint_step  # noqa: E402
import train_omega2 as train_utils  # noqa: E402
from football_detect import train_football_detect as detect_utils  # noqa: E402
from tuna_sdk import DetectAnnotation  # noqa: E402

DEFAULT_BASE_URL = "https://api.moondream.ai/v1"


def _extract_boxes(payload: dict[str, Any]) -> list[DetectAnnotation]:
    raw_boxes = payload.get("objects")
    if raw_boxes is None and isinstance(payload.get("output"), dict):
        raw_boxes = payload["output"].get("objects")
    boxes: list[DetectAnnotation] = []
    for item in raw_boxes or []:
        if not isinstance(item, dict):
            continue
        try:
            boxes.append(
                DetectAnnotation(
                    x_min=float(item["x_min"]),
                    y_min=float(item["y_min"]),
                    x_max=float(item["x_max"]),
                    y_max=float(item["y_max"]),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return boxes


def _call_detect_api(
    *,
    api_base: str,
    api_key: str,
    model: str,
    image_url: str,
    object_name: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_objects: int,
    timeout: float,
) -> list[DetectAnnotation]:
    payload = {
        "model": model,
        "object": object_name,
        "image_url": image_url,
        "settings": {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "max_objects": max_objects,
        },
    }
    req = urllib.request.Request(
        api_base.rstrip("/") + "/detect",
        data=json.dumps(payload).encode("utf-8"),
        headers=detect_utils._build_auth_headers(api_key),
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=float(timeout)) as response:
        body = response.read().decode("utf-8", errors="replace")
    parsed = json.loads(body) if body else {}
    if not isinstance(parsed, dict):
        return []
    return _extract_boxes(parsed)


@dataclass(frozen=True)
class _FakeOutput:
    objects: list[DetectAnnotation]


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

    def _run_request(self, request: Any) -> _FakeResult:
        try:
            boxes = _call_detect_api(
                api_base=str(self.args.base_url),
                api_key=str(self.args.api_key),
                model=self.model,
                image_url=str(getattr(request, "image_url", "")),
                object_name=str(getattr(request, "object_name", "omega logo")),
                temperature=float(getattr(getattr(request, "settings", None), "temperature", self.args.temperature)),
                top_p=float(getattr(getattr(request, "settings", None), "top_p", self.args.top_p)),
                max_tokens=int(getattr(getattr(request, "settings", None), "max_tokens", self.args.max_tokens)),
                max_objects=int(getattr(getattr(request, "settings", None), "max_objects", self.args.max_objects)),
                timeout=float(self.args.timeout),
            )
        except Exception as exc:
            print(f"benchmark detect failed: {type(exc).__name__}: {exc}")
            return _FakeResult(rollouts=[])
        return _FakeResult(rollouts=[_FakeRollout(output=_FakeOutput(objects=boxes))])

    def rollouts_batch(self, *, requests: list[Any], num_rollouts: int, max_workers: int) -> list[_FakeResult]:
        del num_rollouts
        results: list[Optional[_FakeResult]] = [None] * len(requests)
        worker_count = max(1, min(int(max_workers), len(requests)))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {executor.submit(self._run_request, request): idx for idx, request in enumerate(requests)}
            for future in as_completed(future_map):
                idx = future_map[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:  # pragma: no cover
                    print(f"benchmark worker failed: {type(exc).__name__}: {exc}")
                    results[idx] = _FakeResult(rollouts=[])
        return [result if result is not None else _FakeResult(rollouts=[]) for result in results]


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark omega detect checkpoints.")
    parser.add_argument("--api-key", default=os.environ.get("MOONDREAM_API_KEY", ""))
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN", ""))
    parser.add_argument("--base-url", default=os.environ.get("TUNA_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--dataset-name", default=train_utils.TEST_DATASET)
    parser.add_argument("--split", default="train")
    parser.add_argument("--model", default="")
    parser.add_argument("--finetune-id", default="")
    parser.add_argument("--checkpoint-step", type=int, default=-1)
    parser.add_argument(
        "--checkpoint-fallback-policy",
        choices=["nearest_saved", "exact"],
        default="nearest_saved",
    )
    parser.add_argument("--checkpoint-ready-max-wait-s", type=float, default=0.0)
    parser.add_argument("--checkpoint-ready-poll-interval-s", type=float, default=5.0)
    parser.add_argument("--base-model", default="moondream3-preview")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-objects", type=int, default=50)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--max-boxes", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-max-samples", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--predictions-jsonl", default="")
    args = parser.parse_args(argv)
    if int(args.checkpoint_step) < 0:
        args.checkpoint_step = None
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


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    if not args.api_key:
        raise ValueError("MOONDREAM_API_KEY is required")
    if not args.hf_token:
        raise ValueError("HF_TOKEN or HUGGINGFACE_HUB_TOKEN is required")
    model = _resolve_model(args)
    fake_finetune = _APIFinetune(args=args, model=model)
    metrics = train_utils._evaluate(
        finetune=fake_finetune,
        dataset_name=str(args.dataset_name),
        split=str(args.split),
        token=str(args.hf_token),
        batch_size=int(args.eval_batch_size),
        max_boxes=args.max_boxes,
        max_samples=args.eval_max_samples,
        rng=train_utils.random.Random(int(args.seed)),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        max_tokens=int(args.max_tokens),
        max_objects=int(args.max_objects),
        max_workers=int(args.max_workers),
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
        f"benchmark split={args.split} f1={float(metrics.get('eval_f1', 0.0)):.4f} "
        f"macro_f1={float(metrics.get('eval_f1_macro', 0.0)):.4f} "
        f"miou={float(metrics.get('eval_miou', 0.0)):.4f}"
    )


if __name__ == "__main__":
    main()
