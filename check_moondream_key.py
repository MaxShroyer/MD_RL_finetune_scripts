#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import mimetypes
import os
import sys
from pathlib import Path


DEFAULT_TEST_IMAGE_DATA_URL = (
    "data:image/jpeg;base64,"
    "/9j//gAQTGF2YzYxLjE5LjEwMQD/2wBDAAg+Pkk+SVVVVVVVVWRdZGhoaGRkZGRoaGhwcHCDg4NwcHBoaHBwfHyDg4+Tj4eHg4eTk5ubm7q6srLZ2eD/////xABZAAADAQEBAQAAAAAAAAAAAAAABgcFCAECAQEAAAAAAAAAAAAAAAAAAAAAEAADAAMBAQEBAAAAAAAAAAAAAQIDIREEURKBEQEAAAAAAAAAAAAAAAAAAAAA/8AAEQgAGQAZAwESAAISAAMSAP/aAAwDAQACEQMRAD8A5/PQAAABirHyVS2mUip/Pm4/vQAih9ABuRUrVLqMEALVNead7/pFgAfc+d5NLSEEAAAA/9k="
)


def _default_env_file(repo_root: Path) -> str | None:
    for candidate in ("MDBallHolder/.env", "MDpi_and_d/.env", ".env"):
        if (repo_root / candidate).exists():
            return candidate
    return None


def _load_dotenv(repo_root: Path, env_file: str | None) -> None:
    if not env_file:
        return

    env_path = Path(env_file)
    if not env_path.is_absolute():
        env_path = repo_root / env_path
    if not env_path.exists():
        return

    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return

    load_dotenv(dotenv_path=str(env_path), override=False)

def _image_path_to_data_url(image_path: Path) -> str:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    mime, _ = mimetypes.guess_type(str(image_path))
    if not mime or not mime.startswith("image/"):
        mime = "image/jpeg"

    data = image_path.read_bytes()
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def main() -> int:
    repo_root = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Simple Moondream API key check (makes an authenticated /v1/query inference request)."
    )
    parser.add_argument(
        "--env-file",
        default=_default_env_file(repo_root),
        help="Optional path to a dotenv file containing MOONDREAM_API_KEY (defaults to a detected .env in this repo).",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("MOONDREAM_BASE_URL", "https://api.moondream.ai/v1"),
        help="Moondream API base URL.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Request timeout in seconds.",
    )
    parser.add_argument(
        "--question",
        default="What is in this image?",
        help="Question to ask via /v1/query.",
    )
    parser.add_argument(
        "--image",
        help="Path to an image file to send (defaults to a tiny built-in test image).",
    )
    parser.add_argument(
        "--image-url",
        help="Optional prebuilt data URL (data:image/...;base64,...) to send instead of --image.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("MOONDREAM_MODEL"),
        help="Optional model override (e.g. moondream3-preview/<finetune_id>@<step>).",
    )
    parser.add_argument(
        "--auth-header",
        default=os.environ.get("MOONDREAM_AUTH_HEADER", "X-Moondream-Auth"),
        help="Auth header name (default: X-Moondream-Auth).",
    )
    args = parser.parse_args()

    _load_dotenv(repo_root, args.env_file)

    api_key = os.environ.get("MOONDREAM_API_KEY")
    if not api_key:
        print(
            "Missing MOONDREAM_API_KEY. Export it in your shell or put it in a dotenv file and pass --env-file.",
            file=sys.stderr,
        )
        return 2

    if args.image_url:
        image_url = args.image_url
    elif args.image:
        try:
            image_url = _image_path_to_data_url(Path(args.image))
        except Exception as exc:
            print(str(exc), file=sys.stderr)
            return 2
    else:
        image_url = DEFAULT_TEST_IMAGE_DATA_URL

    import httpx

    url = f"{args.base_url.rstrip('/')}/query"
    headers = {args.auth_header: api_key}
    payload: dict[str, object] = {"image_url": image_url, "question": args.question}
    if args.model:
        payload["model"] = args.model

    try:
        with httpx.Client(timeout=args.timeout) as client:
            resp = client.post(url, headers=headers, json=payload)
    except httpx.RequestError as exc:
        print(f"Network error while calling Moondream API: {exc}", file=sys.stderr)
        return 1

    if resp.status_code in (401, 403):
        print(
            f"Unauthorized ({resp.status_code}): API key is invalid or not permitted for /v1/query.",
            file=sys.stderr,
        )
        request_id = resp.headers.get("x-request-id") or resp.headers.get("X-Request-Id")
        if request_id:
            print(f"request_id: {request_id}", file=sys.stderr)
        return 1

    if not resp.is_success:
        print(f"API error ({resp.status_code}).", file=sys.stderr)
        request_id = resp.headers.get("x-request-id") or resp.headers.get("X-Request-Id")
        if request_id:
            print(f"request_id: {request_id}", file=sys.stderr)
        try:
            print(resp.json(), file=sys.stderr)
        except Exception:
            print(resp.text, file=sys.stderr)
        return 1

    try:
        data = resp.json()
    except Exception:
        print("OK: authenticated request succeeded, but response was not JSON.", file=sys.stderr)
        print(resp.text)
        return 0

    answer = data.get("answer")
    request_id = data.get("request_id")
    print("OK: inference request succeeded.")
    if request_id:
        print(f"request_id={request_id}")
    if isinstance(answer, str):
        print(answer)
    else:
        print(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
