#!/usr/bin/env python3
"""Small helper to probe the Moondream staging /query API."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

try:
    from dotenv import load_dotenv as _load_dotenv
except ModuleNotFoundError:  # pragma: no cover
    _load_dotenv = None

DEFAULT_STAGING_BASE_URL = "https://api-staging.moondream.ai/v1"
DEFAULT_ENV_FILE = Path(__file__).resolve().parent / ".env.staging"
DEFAULT_API_KEY_ENV_VAR = "CICID_GPUB_MOONDREAM_API_KEY_1"
FALLBACK_API_KEY_ENV_VARS = (
    "MOONDREAM_API_KEY",
    "CICID_GPUB_MOONDREAM_API_KEY_2",
    "CICID_GPUB_MOONDREAM_API_KEY_3",
    "CICID_GPUB_MOONDREAM_API_KEY_4",
)
DEFAULT_MODEL = "moondream3-preview"
DEFAULT_TEST_IMAGE_DATA_URL = (
    "data:image/jpeg;base64,"
    "/9j//gAQTGF2YzYxLjE5LjEwMQD/2wBDAAg+Pkk+SVVVVVVVVWRdZGhoaGRkZGRoaGhwcHCDg4NwcHBoaHBwfHyDg4+Tj4eHg4eTk5ubm7q6srLZ2eD/////xABZAAADAQEBAQAAAAAAAAAAAAAABgcFCAECAQEAAAAAAAAAAAAAAAAAAAAAEAADAAMBAQEBAAAAAAAAAAAAAQIDIREEURKBEQEAAAAAAAAAAAAAAAAAAAAA/8AAEQgAGQAZAwESAAISAAMSAP/aAAwDAQACEQMRAD8A5/PQAAABirHyVS2mUip/Pm4/vQAih9ABuRUrVLqMEALVNead7/pFgAfc+d5NLSEEAAAA/9k="
)


def _resolve_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path

    module_root = Path(__file__).resolve().parent
    repo_root = module_root.parent
    for base in (Path.cwd(), repo_root, module_root):
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    return (Path.cwd() / path).resolve()


def _maybe_load_dotenv(env_file: str) -> None:
    if _load_dotenv is None:
        return

    path = _resolve_path(env_file)
    if not path.exists():
        return
    _load_dotenv(path, override=False)


def _build_auth_headers(api_key: str, *, auth_header: str) -> dict[str, str]:
    key = api_key.strip()
    if auth_header.lower() == "authorization" and not key.lower().startswith("bearer "):
        key = f"Bearer {key}"
    return {
        "Accept": "application/json",
        "Content-Type": "application/json",
        auth_header: key,
        "User-Agent": "md-ttt-staging-check/0.1",
    }


def _image_path_to_data_url(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(image_path))
    if not mime_type or not mime_type.startswith("image/"):
        mime_type = "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _resolve_api_key(explicit_api_key: str, api_key_env_var: str) -> tuple[str, str]:
    key = str(explicit_api_key or "").strip()
    if key:
        return key, "cli"

    env_var_name = str(api_key_env_var or "").strip()
    if env_var_name:
        key = os.environ.get(env_var_name, "").strip()
        if key:
            return key, env_var_name

    for fallback_name in FALLBACK_API_KEY_ENV_VARS:
        key = os.environ.get(fallback_name, "").strip()
        if key:
            return key, fallback_name

    return "", env_var_name or DEFAULT_API_KEY_ENV_VAR


def _build_query_payload(
    *,
    model: str,
    question: str,
    image_url: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    reasoning: Optional[bool],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "question": question,
        "image_url": image_url,
        "settings": {
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": int(max_tokens),
        },
    }
    if reasoning is not None:
        payload["reasoning"] = bool(reasoning)
    return payload


def _extract_answer_text(payload: Any) -> str:
    if isinstance(payload, dict):
        answer = payload.get("answer")
        if isinstance(answer, str):
            return answer
        output = payload.get("output")
        if isinstance(output, dict):
            nested_answer = output.get("answer")
            if isinstance(nested_answer, str):
                return nested_answer
    return ""


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe the Moondream staging /query API.")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE))
    parser.add_argument("--base-url", default=DEFAULT_STAGING_BASE_URL)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--api-key-env-var", default=DEFAULT_API_KEY_ENV_VAR)
    parser.add_argument("--auth-header", default=os.environ.get("MOONDREAM_AUTH_HEADER", "X-Moondream-Auth"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--question", default="Describe this image briefly.")
    parser.add_argument("--image", default="")
    parser.add_argument("--image-url", default="")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--reasoning",
        dest="reasoning",
        action="store_true",
        help="Include reasoning=true in the /query payload.",
    )
    parser.add_argument(
        "--no-reasoning",
        dest="reasoning",
        action="store_false",
        help="Include reasoning=false in the /query payload.",
    )
    parser.set_defaults(reasoning=None)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    _maybe_load_dotenv(args.env_file)

    api_key, api_key_source = _resolve_api_key(args.api_key, args.api_key_env_var)
    if not api_key:
        print(
            "Missing Moondream API key. Pass --api-key or set "
            f"{args.api_key_env_var or DEFAULT_API_KEY_ENV_VAR} in {args.env_file}.",
            file=sys.stderr,
        )
        return 2

    if args.image_url:
        image_url = str(args.image_url).strip()
        image_label = "image_url"
    elif args.image:
        image_path = _resolve_path(args.image)
        if not image_path.exists():
            print(f"Image not found: {image_path}", file=sys.stderr)
            return 2
        image_url = _image_path_to_data_url(image_path)
        image_label = str(image_path)
    else:
        image_url = DEFAULT_TEST_IMAGE_DATA_URL
        image_label = "builtin_test_image"

    payload = _build_query_payload(
        model=str(args.model).strip(),
        question=str(args.question),
        image_url=image_url,
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        max_tokens=int(args.max_tokens),
        reasoning=args.reasoning,
    )
    endpoint = str(args.base_url).rstrip("/") + "/query"
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=_build_auth_headers(api_key, auth_header=str(args.auth_header)),
        method="POST",
    )

    print(f"POST {endpoint}")
    print(f"model={payload['model']}")
    print(f"api_key_source={api_key_source}")
    print(f"image={image_label}")

    try:
        with urllib.request.urlopen(request, timeout=float(args.timeout)) as response:
            status_code = getattr(response, "status", 200)
            request_id = str(
                response.headers.get("x-request-id") or response.headers.get("X-Request-Id") or ""
            )
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        request_id = str(exc.headers.get("x-request-id") or exc.headers.get("X-Request-Id") or "")
        body = exc.read().decode("utf-8", errors="replace")
        print(f"status={exc.code}", file=sys.stderr)
        if request_id:
            print(f"request_id={request_id}", file=sys.stderr)
        print(body or str(exc), file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f"Network error: {exc}", file=sys.stderr)
        return 1

    print(f"status={status_code}")
    if request_id:
        print(f"request_id={request_id}")

    try:
        parsed = json.loads(body) if body else {}
    except ValueError:
        print(body)
        return 0

    answer_text = _extract_answer_text(parsed)
    if answer_text:
        print(f"answer={answer_text}")
    print(json.dumps(parsed, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
