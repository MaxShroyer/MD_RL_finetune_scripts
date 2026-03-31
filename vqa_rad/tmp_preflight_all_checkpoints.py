#!/usr/bin/env python3
"""Preflight /query against every saved checkpoint for a finetune."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from construction_site import query_common  # noqa: E402
from vqa_rad.common import DEFAULT_STAGING_API_BASE, repo_relative  # noqa: E402


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight finetuned /query checkpoints.")
    parser.add_argument("--env-file", default=str(repo_relative(".env.staging")))
    parser.add_argument("--api-key", default="")
    parser.add_argument("--api-key-env-var", default="CICID_GPUB_MOONDREAM_API_KEY_1")
    parser.add_argument("--base-url", default=DEFAULT_STAGING_API_BASE)
    parser.add_argument("--finetune-id", required=True)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--reasoning", action="store_true", help="Enable reasoning for preflight requests.")
    parser.add_argument("--retry-429-max-retries", type=int, default=2)
    parser.add_argument("--retry-429-backoff-s", type=float, default=1.0)
    parser.add_argument("--retry-429-max-backoff-s", type=float, default=8.0)
    parser.add_argument("--retry-5xx-max-retries", type=int, default=1)
    parser.add_argument("--retry-5xx-backoff-s", type=float, default=1.0)
    parser.add_argument("--retry-5xx-max-backoff-s", type=float, default=8.0)
    parser.add_argument("--reverse", action="store_true", help="Check newest checkpoints first.")
    parser.add_argument("--max-checkpoints", type=int, default=0, help="Optional cap; <=0 means all.")
    parser.add_argument("--output-json", default="", help="Optional summary JSON output path.")
    args = parser.parse_args(argv)
    args.env_file = query_common.resolve_env_file(
        args.env_file,
        repo_root=REPO_ROOT,
        module_root=Path(__file__).resolve().parent,
    )
    return args


def resolve_api_key(args: argparse.Namespace) -> str:
    if str(args.api_key or "").strip():
        return str(args.api_key).strip()
    value = os.environ.get(str(args.api_key_env_var or "MOONDREAM_API_KEY"), "")
    if str(value or "").strip():
        return str(value).strip()
    value = os.environ.get("MOONDREAM_API_KEY", "")
    if str(value or "").strip():
        return str(value).strip()
    raise ValueError("MOONDREAM_API_KEY is required")


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    query_common.load_dotenv(args.env_file, override=False)
    api_key = resolve_api_key(args)
    finetune_id = str(args.finetune_id or "").strip()
    if not finetune_id:
        raise SystemExit("--finetune-id is required")

    try:
        saved_steps = query_common.list_saved_checkpoint_steps(
            api_base=args.base_url,
            api_key=api_key,
            finetune_id=finetune_id,
            timeout=args.timeout,
        )
    except Exception as exc:
        raise SystemExit(query_common.error_message(exc))

    if args.reverse:
        saved_steps = list(reversed(saved_steps))
    if int(args.max_checkpoints) > 0:
        saved_steps = saved_steps[: int(args.max_checkpoints)]

    print(
        f"finetune_id={finetune_id} checkpoints={len(saved_steps)} "
        f"steps={saved_steps}"
    )

    results: list[dict[str, object]] = []
    ok_count = 0
    fail_count = 0

    for step in saved_steps:
        model = f"moondream3-preview/{finetune_id}@{int(step)}"
        try:
            answer_text, raw_response, latency_ms = query_common.preflight_query_api(
                api_base=args.base_url,
                api_key=api_key,
                model=model,
                timeout=args.timeout,
                reasoning=bool(args.reasoning),
                retry_429_max_retries=args.retry_429_max_retries,
                retry_429_backoff_s=args.retry_429_backoff_s,
                retry_429_max_backoff_s=args.retry_429_max_backoff_s,
                retry_5xx_max_retries=args.retry_5xx_max_retries,
                retry_5xx_backoff_s=args.retry_5xx_backoff_s,
                retry_5xx_max_backoff_s=args.retry_5xx_max_backoff_s,
            )
            ok_count += 1
            record: dict[str, object] = {
                "step": int(step),
                "model": model,
                "status": "ok",
                "latency_ms": float(latency_ms),
                "answer": str(answer_text or ""),
                "raw_response": raw_response,
            }
            print(
                f"step={step} status=ok latency_ms={latency_ms:.1f} "
                f"answer={json.dumps(str(answer_text or ''))}"
            )
        except Exception as exc:
            fail_count += 1
            record = {
                "step": int(step),
                "model": model,
                "status": "fail",
                "error": query_common.error_message(exc),
            }
            print(f"step={step} status=fail error={query_common.error_message(exc)}")
        results.append(record)

    summary = {
        "finetune_id": finetune_id,
        "base_url": args.base_url,
        "checked_steps": [int(step) for step in saved_steps],
        "ok_count": ok_count,
        "fail_count": fail_count,
        "results": results,
    }

    if str(args.output_json or "").strip():
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote summary JSON: {output_path}")


if __name__ == "__main__":
    main()
