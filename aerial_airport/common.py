from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_ROOT = Path(__file__).resolve().parent

DEFAULT_RAW_DATASET_DIR = MODULE_ROOT / "raw_dataset" / "Aerial Airport.coco"
DEFAULT_HF_DATASET_NAME = "maxs-m87/aerial_airport_point_v2"
DEFAULT_WANDB_PROJECT = "moondream-aerial-airport-point-rl"
DEFAULT_STAGING_API_BASE = "https://api-staging.moondream.ai/v1"
DEFAULT_CLASS_NAME = "airplane"
DEFAULT_CLASS_UID = "aerial_airport:airplane"
DEFAULT_SKILL = "point"
DEFAULT_POINT_PROMPT_STYLE = "class_name"
DEFAULT_REWARD_METRIC = "f1"


def repo_relative(*parts: str) -> Path:
    return MODULE_ROOT.joinpath(*parts)


def resolve_config_path(raw_path: str, *, script_dir: Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    from_cwd = (Path.cwd() / path).resolve()
    if from_cwd.exists():
        return from_cwd
    from_repo = (REPO_ROOT / path).resolve()
    if from_repo.exists():
        return from_repo
    from_script = (script_dir / path).resolve()
    if from_script.exists():
        return from_script
    return from_cwd


def load_json_config(
    config_path: Path,
    *,
    default_path: Optional[Path] = None,
) -> dict[str, Any]:
    if not config_path.exists():
        if default_path is not None and config_path == default_path:
            return {}
        raise FileNotFoundError(f"Config file not found: {config_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a JSON object: {config_path}")
    return payload


def option_for_action(action: argparse.Action) -> str:
    for opt in action.option_strings:
        if opt.startswith("--"):
            return opt
    return action.option_strings[0]


def config_to_cli_args(
    parser: argparse.ArgumentParser,
    config: dict[str, Any],
    *,
    config_path: Path,
    overridden_dests: Optional[set[str]] = None,
) -> list[str]:
    overridden = set(overridden_dests or set())
    by_dest: dict[str, list[argparse.Action]] = {}
    for action in parser._actions:
        if not action.option_strings or action.dest == "help":
            continue
        by_dest.setdefault(action.dest, []).append(action)

    unknown = sorted(key for key in config if key not in by_dest)
    if unknown:
        raise ValueError(
            f"Unknown config key(s) in {config_path}: {unknown}. "
            "Remove typos or update script support."
        )

    cli_args: list[str] = []
    for key, raw_value in config.items():
        if key in overridden:
            continue
        actions = by_dest[key]
        const_actions = [a for a in actions if isinstance(a, argparse._StoreConstAction)]
        store_actions = [a for a in actions if not isinstance(a, argparse._StoreConstAction)]

        if raw_value is None:
            matched = next((a for a in const_actions if getattr(a, "const", object()) is None), None)
            if matched is not None:
                cli_args.append(option_for_action(matched))
            continue

        if isinstance(raw_value, bool):
            matched = next((a for a in const_actions if getattr(a, "const", object()) is raw_value), None)
            if matched is not None:
                cli_args.append(option_for_action(matched))
            continue

        if not store_actions:
            continue

        action = store_actions[0]
        cli_args.append(option_for_action(action))
        if isinstance(raw_value, list):
            cli_args.extend(str(item) for item in raw_value)
        elif isinstance(raw_value, dict):
            cli_args.append(json.dumps(raw_value))
        else:
            cli_args.append(str(raw_value))
    return cli_args


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def normalize_class_name(value: Any) -> str:
    text = " ".join(str(value or "").strip().replace("_", " ").split()).lower()
    if text in {"plane", "planes"}:
        return DEFAULT_CLASS_NAME
    return text


def default_prompt_for_class(class_name: str) -> str:
    normalized = normalize_class_name(class_name)
    return normalized or DEFAULT_CLASS_NAME


def class_uid_for_name(class_name: str) -> str:
    normalized = normalize_class_name(class_name)
    if normalized == DEFAULT_CLASS_NAME:
        return DEFAULT_CLASS_UID
    safe = normalized.replace("/", "_").replace(" ", "_")
    return f"aerial_airport:{safe}"


def build_class_catalog(class_names: Iterable[str]) -> list[dict[str, str]]:
    normalized_names = sorted({normalize_class_name(name) for name in class_names if normalize_class_name(name)})
    return [
        {
            "class_uid": class_uid_for_name(class_name),
            "class_name": class_name,
            "prompt": default_prompt_for_class(class_name),
        }
        for class_name in normalized_names
    ]


def _load_jsonish(value: Any) -> Any:
    raw = value
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        raw = json.loads(text)
    return raw


def extract_all_class_names(answer_boxes: Any) -> list[str]:
    raw = _load_jsonish(answer_boxes)
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return []

    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        class_name = normalize_class_name(item.get("class_name") or item.get("source_class_name"))
        if not class_name or class_name in seen:
            continue
        seen.add(class_name)
        out.append(class_name)
    return out


def discover_class_names(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    discovered: set[str] = set()
    for row in rows:
        discovered.update(extract_all_class_names(row.get("answer_boxes")))
    return sorted(discovered)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
