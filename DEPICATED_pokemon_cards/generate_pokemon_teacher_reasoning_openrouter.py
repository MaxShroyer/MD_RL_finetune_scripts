#!/usr/bin/env python3
"""Generate teacher rationale labels for PokemonCards rows via OpenRouter."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

try:
    from tqdm.auto import tqdm  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    class _SimpleTqdm:
        def __init__(self, iterable, *args: Any, **kwargs: Any) -> None:
            self._iterable = iterable

        def __iter__(self):
            return iter(self._iterable)

        def set_postfix(self, *args: Any, **kwargs: Any) -> None:
            return

    def tqdm(iterable=None, *args, **kwargs):  # type: ignore
        if iterable is None:
            iterable = []
        return _SimpleTqdm(iterable, *args, **kwargs)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from DEPICATED_pokemon_cards import data_loader  # noqa: E402
from DEPICATED_pokemon_cards.common import (  # noqa: E402
    cfg_bool,
    cfg_float,
    cfg_int,
    cfg_list_str,
    cfg_str,
    ensure_parent_dir,
    image_path_to_data_url,
    load_dotenv_if_available,
    load_json_object,
    progress_enabled,
    resolve_path,
    truncate_text,
    validate_config_keys,
    write_json,
)
from DEPICATED_pokemon_cards.scoring import answer_reward_for_task, parse_prediction_json  # noqa: E402
from DEPICATED_pokemon_cards.task_schema import (  # noqa: E402
    CANONICAL_TASK_TYPES,
    canonicalize_rationale_text,
    normalize_answer_for_task,
    parse_rationale_text,
    normalize_task_type,
)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "teacher_openrouter_default.json"
DEFAULT_API_BASE = "https://openrouter.ai/api/v1"
DEFAULT_OUTPUT_DIR_SUFFIX = "_teacher"
DEFAULT_TIMEOUT = 90.0
CARD_SUMMARY_ACCEPTANCE_THRESHOLD = 0.85

CONFIG_ALLOWED_KEYS = {
    "api_base",
    "api_key",
    "dataset_dir",
    "dataset_source",
    "env_file",
    "fallback_to_ground_truth_rationale",
    "hf_cache_dir",
    "hf_dataset_repo_id",
    "hf_dataset_revision",
    "hf_token",
    "max_rows_per_split",
    "no_progress",
    "output_dir",
    "request_overrides",
    "seed",
    "splits",
    "task_types",
    "teacher_model_id",
    "temperature",
    "timeout",
    "top_p",
    "max_tokens",
}


class TeacherAPIError(Exception):
    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        request_id: str = "",
        response_body: str = "",
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.request_id = request_id
        self.response_body = response_body


def _build_auth_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "HTTP-Referer": "https://openrouter.ai",
        "X-Title": "pokemon_cards_teacher_generation",
        "User-Agent": "pokemon-cards-teacher-generator/0.1",
    }


def _extract_openrouter_message(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return {}
    first = choices[0]
    if not isinstance(first, dict):
        return {}
    message = first.get("message")
    return message if isinstance(message, dict) else {}


def extract_openrouter_answer_text(payload: Any) -> str:
    message = _extract_openrouter_message(payload)
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
            elif isinstance(item, str) and item:
                parts.append(item)
        return "\n".join(parts)
    return ""


def _schema_hint_for_task(task_type: str) -> dict[str, Any]:
    if task_type == "card_identity":
        return {"name": "string", "hp": 0, "set_name": "string"}
    if task_type == "card_core":
        return {
            "name": "string",
            "hp": 0,
            "set_name": "string",
            "stage": "string_or_null",
            "pokemon_types": ["string"],
            "rarity": "string_or_null",
            "evolves_from": "string_or_null",
        }
    if task_type == "attack_overview":
        return {"attack_names": ["string"], "attack_count": 0}
    if task_type == "card_summary":
        return {"summary": "string"}
    raise ValueError(f"unsupported task_type: {task_type}")


def _rationale_instruction_for_task(task_type: str) -> str:
    if task_type == "card_identity":
        return 'Use rationale format: "name=<value>; hp=<value>; set=<value>"'
    if task_type == "card_core":
        return (
            'Use rationale format: "name=<value>; hp=<value>; set=<value>; stage=<value>; '
            'types=<comma-list>; rarity=<value>; evolves_from=<value>"'
        )
    if task_type == "attack_overview":
        return 'Use rationale format: "name=<value>; attacks=<comma-list>; attack_count=<value>"'
    if task_type == "card_summary":
        return 'Use rationale format: "name=<value>; hp=<value>; set=<value>; stage=<value>; types=<comma-list>; attacks=<comma-list>"'
    raise ValueError(f"unsupported task_type: {task_type}")


def _build_prompt(*, task_type: str, question: str) -> str:
    schema_hint = json.dumps(_schema_hint_for_task(task_type), sort_keys=True)
    return (
        "You are extracting structured visual information from a Pokemon trading card image.\n"
        f"Question: {question}\n"
        "Return exactly one JSON object with exactly two top-level keys: answer and rationale.\n"
        f"answer schema: {schema_hint}\n"
        f"{_rationale_instruction_for_task(task_type)}\n"
        "Keep rationale under 96 tokens. Do not include markdown. Do not include extra keys."
    )


def _build_chat_payload(
    *,
    model_id: str,
    prompt: str,
    image_url: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    request_overrides: Optional[dict[str, Any]],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": int(max_tokens),
    }
    if request_overrides:
        payload.update(request_overrides)
    return payload


def _call_openrouter_chat_api(
    *,
    api_base: str,
    api_key: str,
    model_id: str,
    prompt: str,
    image_url: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout: float,
    request_overrides: Optional[dict[str, Any]] = None,
) -> tuple[str, dict[str, Any], float]:
    payload = _build_chat_payload(
        model_id=model_id,
        prompt=prompt,
        image_url=image_url,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        request_overrides=request_overrides,
    )
    endpoint = api_base.rstrip("/") + "/chat/completions"
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=_build_auth_headers(api_key),
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=float(timeout)) as response:
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        request_id = exc.headers.get("x-request-id", "")
        body_text = exc.read().decode("utf-8", errors="replace")
        raise TeacherAPIError(
            f"HTTP {exc.code} {exc.reason}",
            status_code=exc.code,
            request_id=request_id,
            response_body=body_text,
        ) from exc
    except urllib.error.URLError as exc:
        raise TeacherAPIError(f"Network error: {exc}") from exc

    latency_ms = (time.monotonic() - started) * 1000.0
    data = json.loads(body) if body else {}
    if not isinstance(data, dict):
        data = {}
    answer_text = extract_openrouter_answer_text(data)
    return answer_text, data, latency_ms


def _resolve_api_key(cli_api_key: str) -> str:
    key = str(cli_api_key or "").strip()
    if key:
        return key
    env_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if env_key:
        return env_key
    raise ValueError("OPENROUTER_API_KEY is required")


def _parse_teacher_payload(answer_text: str) -> Optional[dict[str, Any]]:
    return parse_prediction_json(answer_text)


def _teacher_meta_from_response(
    *,
    payload: dict[str, Any],
    latency_ms: float,
    teacher_model_id: str,
    normalized_teacher_answer: Optional[dict[str, Any]],
    answer_reward: float,
    answer_matches_ground_truth: bool,
    rationale_text: str,
    error: str = "",
) -> dict[str, Any]:
    message = _extract_openrouter_message(payload)
    return {
        "teacher_model_id": teacher_model_id,
        "latency_ms": latency_ms,
        "answer_reward": float(answer_reward),
        "answer_matches_ground_truth": answer_matches_ground_truth,
        "teacher_answer": normalized_teacher_answer,
        "teacher_rationale_raw": rationale_text,
        "provider_reasoning": message.get("reasoning"),
        "provider_reasoning_details": message.get("reasoning_details"),
        "error": error,
    }


def _copy_dataset_images(source_dir: Path, target_dir: Path) -> None:
    source_images = source_dir / "images"
    if not source_images.exists():
        return
    target_images = target_dir / "images"
    if target_images.exists():
        return
    shutil.copytree(source_images, target_images)


def _load_rows_for_split(
    *,
    dataset_dir: Path,
    split_name: str,
    hf_dataset_repo_id: str,
    hf_dataset_revision: str,
    hf_token: str,
    hf_cache_dir: str,
) -> list[dict[str, Any]]:
    return data_loader.load_split_rows(
        dataset_source="local_jsonl",
        split_name=split_name,
        dataset_dir=dataset_dir,
        hf_dataset_repo_id=hf_dataset_repo_id,
        hf_dataset_revision=hf_dataset_revision,
        hf_token=hf_token,
        hf_cache_dir=hf_cache_dir,
    )


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    pre_args, _ = pre_parser.parse_known_args(argv)

    config_path = resolve_path(pre_args.config, search_roots=(REPO_ROOT, Path(__file__).resolve().parent))
    config = load_json_object(config_path) if config_path.exists() else {}
    validate_config_keys(config, allowed_keys=CONFIG_ALLOWED_KEYS, config_path=config_path)

    parser = argparse.ArgumentParser(description="Generate teacher rationale labels for PokemonCards rows.")
    parser.add_argument("--config", default=str(config_path))
    parser.add_argument("--env-file", default=cfg_str(config, "env_file", ".env"))
    parser.add_argument("--api-key", default=cfg_str(config, "api_key", ""))
    parser.add_argument("--api-base", default=cfg_str(config, "api_base", DEFAULT_API_BASE))
    parser.add_argument("--dataset-dir", default=cfg_str(config, "dataset_dir", "pokemon_cards/outputs/thefusion21_pokemoncards_v1"))
    parser.add_argument("--output-dir", default=cfg_str(config, "output_dir", ""))
    parser.add_argument("--teacher-model-id", default=cfg_str(config, "teacher_model_id", ""))
    parser.add_argument("--splits", nargs="*", default=None)
    parser.add_argument("--task-types", nargs="*", default=None)
    parser.add_argument("--temperature", type=float, default=cfg_float(config, "temperature", 0.0))
    parser.add_argument("--top-p", type=float, default=cfg_float(config, "top_p", 1.0))
    parser.add_argument("--max-tokens", type=int, default=cfg_int(config, "max_tokens", 256))
    parser.add_argument("--timeout", type=float, default=cfg_float(config, "timeout", DEFAULT_TIMEOUT))
    parser.add_argument("--max-rows-per-split", type=int, default=cfg_int(config, "max_rows_per_split", 0))
    parser.add_argument("--request-overrides-json", default="")
    parser.add_argument(
        "--fallback-to-ground-truth-rationale",
        action="store_true",
        default=cfg_bool(config, "fallback_to_ground_truth_rationale", False),
    )
    parser.add_argument("--no-progress", action="store_true", default=cfg_bool(config, "no_progress", False))
    args = parser.parse_args(argv)

    args.dataset_dir = resolve_path(args.dataset_dir, search_roots=(REPO_ROOT,))
    if args.output_dir:
        args.output_dir = resolve_path(args.output_dir, search_roots=(REPO_ROOT,))
    else:
        args.output_dir = args.dataset_dir.parent / f"{args.dataset_dir.name}{DEFAULT_OUTPUT_DIR_SUFFIX}"
    args.env_file = resolve_path(args.env_file, search_roots=(REPO_ROOT, Path(__file__).resolve().parent))
    args.request_overrides = {}
    if isinstance(config.get("request_overrides"), dict):
        args.request_overrides.update(dict(config["request_overrides"]))
    if args.request_overrides_json:
        payload = json.loads(args.request_overrides_json)
        if not isinstance(payload, dict):
            raise ValueError("--request-overrides-json must decode to a JSON object")
        args.request_overrides.update(payload)
    if args.splits is None:
        args.splits = cfg_list_str(config, "splits", ["train", "val", "test"])
    if args.task_types is None:
        args.task_types = cfg_list_str(config, "task_types", list(CANONICAL_TASK_TYPES))
    args.task_types = [normalize_task_type(item) for item in args.task_types]
    if not str(args.teacher_model_id or "").strip():
        raise ValueError("--teacher-model-id is required")
    return args


def generate_teacher_labels(
    *,
    dataset_dir: Path,
    output_dir: Path,
    api_base: str,
    api_key: str,
    teacher_model_id: str,
    splits: list[str],
    task_types: list[str],
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout: float,
    request_overrides: dict[str, Any],
    fallback_to_ground_truth_rationale: bool,
    max_rows_per_split: int,
    no_progress: bool,
) -> dict[str, Any]:
    _copy_dataset_images(dataset_dir, output_dir)
    summary: dict[str, Any] = {
        "teacher_model_id": teacher_model_id,
        "splits": {},
    }

    for split_name in splits:
        rows = _load_rows_for_split(
            dataset_dir=dataset_dir,
            split_name=split_name,
            hf_dataset_repo_id="",
            hf_dataset_revision="main",
            hf_token="",
            hf_cache_dir="",
        )
        updated_rows: list[dict[str, Any]] = []
        processed = 0
        accepted = 0
        fallback_count = 0
        errors = 0

        iterable = rows[: max_rows_per_split or None] if max_rows_per_split > 0 else rows
        progress = tqdm(iterable, desc=f"teacher:{split_name}", disable=not progress_enabled(no_progress))
        for row in progress:
            task_type = normalize_task_type(str(row.get("task_type", "")))
            if task_type not in task_types:
                updated_rows.append(dict(row))
                continue

            processed += 1
            image_path = (dataset_dir / str(row["image_path"])).resolve()
            prompt = _build_prompt(task_type=task_type, question=str(row.get("question", "")))
            expected_answer = normalize_answer_for_task(
                task_type,
                json.loads(str(row.get("final_answer_json", "{}"))),
            )
            if expected_answer is None:
                raise ValueError(f"invalid final_answer_json for row_id={row.get('row_id')}")
            source_metadata_raw = str(row.get("source_metadata_json", "") or "")
            if source_metadata_raw:
                try:
                    source_metadata = json.loads(source_metadata_raw)
                except json.JSONDecodeError:
                    source_metadata = {}
            else:
                source_metadata = {}
            if not isinstance(source_metadata, dict):
                source_metadata = {}

            teacher_rationale_text = ""
            teacher_meta: dict[str, Any] = {}
            try:
                image_url = image_path_to_data_url(image_path)
                answer_text, raw_payload, latency_ms = _call_openrouter_chat_api(
                    api_base=api_base,
                    api_key=api_key,
                    model_id=teacher_model_id,
                    prompt=prompt,
                    image_url=image_url,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    request_overrides=request_overrides,
                )
                teacher_payload = _parse_teacher_payload(answer_text)
                if not isinstance(teacher_payload, dict):
                    raise ValueError("teacher response did not decode to a JSON object")
                normalized_teacher_answer = normalize_answer_for_task(task_type, teacher_payload.get("answer"))
                raw_rationale = str(teacher_payload.get("rationale", "") or "")
                answer_reward = (
                    answer_reward_for_task(
                        task_type,
                        teacher_payload.get("answer"),
                        expected_answer,
                        expected_metadata=source_metadata,
                    )
                    if normalized_teacher_answer is not None
                    else 0.0
                )
                acceptance_threshold = CARD_SUMMARY_ACCEPTANCE_THRESHOLD if task_type == "card_summary" else 1.0
                answer_matches = float(answer_reward) >= float(acceptance_threshold)
                if answer_matches and raw_rationale.strip():
                    teacher_rationale_text = canonicalize_rationale_text(task_type, raw_rationale)
                    if parse_rationale_text(task_type, teacher_rationale_text):
                        accepted += 1
                    else:
                        teacher_rationale_text = ""
                elif fallback_to_ground_truth_rationale:
                    teacher_rationale_text = canonicalize_rationale_text(
                        task_type,
                        str(row.get("ground_truth_rationale_text", "") or ""),
                    )
                    fallback_count += 1
                teacher_meta = _teacher_meta_from_response(
                    payload=raw_payload,
                    latency_ms=latency_ms,
                    teacher_model_id=teacher_model_id,
                    normalized_teacher_answer=normalized_teacher_answer,
                    answer_reward=float(answer_reward),
                    answer_matches_ground_truth=answer_matches,
                    rationale_text=raw_rationale,
                )
            except (TeacherAPIError, OSError, ValueError, json.JSONDecodeError) as exc:
                errors += 1
                if fallback_to_ground_truth_rationale:
                    teacher_rationale_text = canonicalize_rationale_text(
                        task_type,
                        str(row.get("ground_truth_rationale_text", "") or ""),
                    )
                    fallback_count += 1
                teacher_meta = {
                    "teacher_model_id": teacher_model_id,
                    "answer_reward": 0.0,
                    "error": truncate_text(str(exc)),
                }

            updated = dict(row)
            updated["teacher_rationale_text"] = teacher_rationale_text
            updated["teacher_model_meta_json"] = json.dumps(teacher_meta, ensure_ascii=False, sort_keys=True)
            updated_rows.append(updated)

        out_path = output_dir / "jsonl" / f"{split_name}.jsonl"
        ensure_parent_dir(out_path)
        with out_path.open("w", encoding="utf-8") as handle:
            for row in updated_rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        summary["splits"][split_name] = {
            "rows_total": len(updated_rows),
            "rows_processed": processed,
            "teacher_rationale_accepted": accepted,
            "teacher_rationale_fallback": fallback_count,
            "errors": errors,
        }

    write_json(output_dir / "metadata" / "teacher_generation_summary.json", summary)
    return summary


def main() -> None:
    args = _parse_args()
    load_dotenv_if_available(args.env_file)
    api_key = _resolve_api_key(args.api_key)
    summary = generate_teacher_labels(
        dataset_dir=Path(args.dataset_dir),
        output_dir=Path(args.output_dir),
        api_base=str(args.api_base),
        api_key=api_key,
        teacher_model_id=str(args.teacher_model_id),
        splits=list(args.splits),
        task_types=list(args.task_types),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        max_tokens=int(args.max_tokens),
        timeout=float(args.timeout),
        request_overrides=dict(args.request_overrides),
        fallback_to_ground_truth_rationale=bool(args.fallback_to_ground_truth_rationale),
        max_rows_per_split=int(args.max_rows_per_split),
        no_progress=bool(args.no_progress),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
