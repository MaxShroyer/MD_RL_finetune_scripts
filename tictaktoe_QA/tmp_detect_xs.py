#!/usr/bin/env python3
"""Small helper to test Moondream detect on TicTacToe X marks."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_URL = "https://api.moondream.ai/v1"
DEFAULT_ENV_FILE = Path(__file__).resolve().parent / ".env"
DEFAULT_IMAGE = (
    REPO_ROOT
    / ".cache"
    / "tictaktoe_QA"
    / "hf_images"
    / "maxs-m87_tictactoe-qa-v1"
    / "main"
    / "train"
    / "003436a674eba3264e6f5d8e72c05d776ea3bd63.png"
)


def _resolve_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
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


def _build_auth_headers(api_key: str) -> dict[str, str]:
    header_name = os.environ.get("MOONDREAM_AUTH_HEADER", "X-Moondream-Auth")
    key = api_key.strip()
    if header_name.lower() == "authorization" and not key.lower().startswith("bearer "):
        key = f"Bearer {key}"
    return {
        "Accept": "application/json",
        "Content-Type": "application/json",
        header_name: key,
        "User-Agent": "md-ttt-detect-xs/0.1",
    }


def _to_data_url(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(image_path))
    if not mime_type:
        mime_type = "application/octet-stream"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test /detect on TicTacToe X marks.")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE))
    parser.add_argument("--api-key", default="")
    parser.add_argument("--base-url", default=os.environ.get("TUNA_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--image", default=str(DEFAULT_IMAGE))
    parser.add_argument("--object", default="X")
    parser.add_argument("--model", default="moondream3-preview")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-objects", type=int, default=9)
    parser.add_argument("--timeout", type=float, default=60.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    env_file = _resolve_path(args.env_file)
    load_dotenv(env_file, override=False)

    api_key = str(args.api_key or os.environ.get("MOONDREAM_API_KEY", "")).strip()
    if not api_key:
        raise ValueError("MOONDREAM_API_KEY is required")

    image_path = _resolve_path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"image not found: {image_path}")

    payload: dict[str, Any] = {
        "model": str(args.model),
        "image_url": _to_data_url(image_path),
        "object": str(args.object),
        "settings": {
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "max_tokens": int(args.max_tokens),
            "max_objects": int(args.max_objects),
        },
    }
    endpoint = str(args.base_url).rstrip("/") + "/detect"

    print(f"POST {endpoint}")
    print(f"image={image_path}")
    print(f"object={args.object}")

    with httpx.Client(timeout=float(args.timeout), headers=_build_auth_headers(api_key)) as client:
        response = client.post(endpoint, json=payload)

    request_id = response.headers.get("x-request-id") or response.headers.get("X-Request-Id") or ""
    print(f"status={response.status_code}")
    if request_id:
        print(f"request_id={request_id}")

    if not response.is_success:
        body = response.text.strip()
        raise RuntimeError(f"detect failed: status={response.status_code} body={body}")

    try:
        parsed = response.json()
    except ValueError:
        print(response.text)
        return

    print(json.dumps(parsed, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
