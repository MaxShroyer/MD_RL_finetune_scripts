from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_ROOT = Path(__file__).resolve().parent

DEFAULT_BASE_URL = "https://api-staging.moondream.ai/v1"
DEFAULT_BASE_MODEL = "moondream3-preview"
DEFAULT_API_KEY_ENV_VAR = "CICID_GPUB_MOONDREAM_API_KEY_1"
DEFAULT_API_KEY_ENV_VARS = (
    "CICID_GPUB_MOONDREAM_API_KEY_1",
    "CICID_GPUB_MOONDREAM_API_KEY_2",
    "CICID_GPUB_MOONDREAM_API_KEY_3",
    "CICID_GPUB_MOONDREAM_API_KEY_4",
)
DEFAULT_OUTPUT_ROOT = MODULE_ROOT / "outputs"

try:
    from dotenv import load_dotenv as _load_dotenv
except ModuleNotFoundError:  # pragma: no cover
    _load_dotenv = None

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


@dataclass(frozen=True)
class ApiKeySlot:
    index: int
    env_var: str
    api_key: str


class ApiKeyPool:
    def __init__(self, slots: Iterable[ApiKeySlot]) -> None:
        self._slots = tuple(slots)
        if not self._slots:
            raise ValueError("ApiKeyPool requires at least one slot.")
        self._cursor = 0

    @property
    def slots(self) -> tuple[ApiKeySlot, ...]:
        return self._slots

    @property
    def env_var_names(self) -> list[str]:
        return [slot.env_var for slot in self._slots if slot.env_var and not slot.env_var.startswith("<")]

    def next_slot(self) -> ApiKeySlot:
        slot = self._slots[self._cursor % len(self._slots)]
        self._cursor += 1
        return slot

    def next_api_key(self) -> str:
        return self.next_slot().api_key

    def describe(self) -> dict[str, Any]:
        return {
            "slot_count": len(self._slots),
            "env_vars": self.env_var_names,
        }


def repo_relative(*parts: str) -> Path:
    return MODULE_ROOT.joinpath(*parts)


def resolve_config_path(raw_path: str, *, script_dir: Path) -> Path:
    path = Path(str(raw_path or "")).expanduser()
    if path.is_absolute():
        return path
    for base in (Path.cwd(), REPO_ROOT, script_dir):
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    return (Path.cwd() / path).resolve()


def resolve_path(raw_path: str, *, repo_root: Path = REPO_ROOT, module_root: Path = MODULE_ROOT) -> Path:
    path = Path(str(raw_path or "")).expanduser()
    if path.is_absolute():
        return path.resolve()
    for base in (Path.cwd(), repo_root, module_root):
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    return (Path.cwd() / path).resolve()


def _is_hf_dataset_dir(path: Path) -> bool:
    return path.is_dir() and (path / "dataset_dict.json").is_file() and (path / "metadata.json").is_file()


def _looks_like_default_inspector_dataset(raw_path: str, *, task: str) -> bool:
    path = Path(str(raw_path or "").strip())
    return path.name.startswith(f"inspector_{task}_v")


def _discover_inspector_dataset_candidates(
    *,
    task: str,
    repo_root: Path = REPO_ROOT,
    module_root: Path = MODULE_ROOT,
) -> list[Path]:
    task_name = str(task or "").strip()
    if task_name not in {"detect", "point"}:
        raise ValueError("task must be 'detect' or 'point'")

    output_root = (module_root / "outputs").resolve()
    if not output_root.is_dir():
        return []

    preferred = [
        output_root / "openrouter_refresh_smoke" / task_name,
        output_root / "smoke_merged_synth_v1" / task_name,
    ]
    discovered: list[Path] = []
    seen: set[Path] = set()

    def _add_candidate(path: Path) -> None:
        resolved = path.resolve()
        if resolved in seen or not _is_hf_dataset_dir(resolved):
            return
        seen.add(resolved)
        discovered.append(resolved)

    for candidate in preferred:
        _add_candidate(candidate)
    for candidate in sorted(output_root.glob(f"*/{task_name}")):
        _add_candidate(candidate)
    for candidate in sorted(output_root.glob(f"inspector_{task_name}_v*")):
        _add_candidate(candidate)
    return discovered


def resolve_inspector_dataset_path(
    raw_path: str,
    *,
    task: str,
    repo_root: Path = REPO_ROOT,
    module_root: Path = MODULE_ROOT,
) -> tuple[Optional[Path], str]:
    dataset_path = str(raw_path or "").strip()
    if not dataset_path:
        return None, ""

    resolved = resolve_path(dataset_path, repo_root=repo_root, module_root=module_root)
    if resolved.exists():
        return resolved, ""

    candidates = _discover_inspector_dataset_candidates(
        task=task,
        repo_root=repo_root,
        module_root=module_root,
    )
    if _looks_like_default_inspector_dataset(dataset_path, task=task) and candidates:
        chosen = candidates[0]
        extras = [str(path) for path in candidates[1:]]
        detail = f"configured {task} dataset path '{resolved}' was not found; using '{chosen}' instead"
        if extras:
            detail = f"{detail}; other local candidates: {', '.join(extras)}"
        return chosen, detail

    message = f"{task} dataset path not found: {resolved}"
    if candidates:
        message = f"{message}. Available local candidates: {', '.join(str(path) for path in candidates)}"
    raise FileNotFoundError(message)


def load_json_config(config_path: Path, *, default_path: Optional[Path] = None) -> dict[str, Any]:
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
        raise ValueError(f"Unknown config key(s) in {config_path}: {unknown}")

    cli_args: list[str] = []
    for key, raw_value in config.items():
        if key in overridden:
            continue
        actions = by_dest[key]
        boolean_optional_actions = [a for a in actions if isinstance(a, argparse.BooleanOptionalAction)]
        const_actions = [a for a in actions if isinstance(a, argparse._StoreConstAction)]
        store_actions = [a for a in actions if not isinstance(a, argparse._StoreConstAction)]
        if raw_value is None:
            matched = next((a for a in const_actions if getattr(a, "const", object()) is None), None)
            if matched is not None:
                cli_args.append(option_for_action(matched))
            continue
        if isinstance(raw_value, bool) and boolean_optional_actions:
            action = boolean_optional_actions[0]
            if raw_value:
                positive = next(
                    (opt for opt in action.option_strings if opt.startswith("--") and not opt.startswith("--no-")),
                    option_for_action(action),
                )
                cli_args.append(positive)
            else:
                negative = next((opt for opt in action.option_strings if opt.startswith("--no-")), "")
                if negative:
                    cli_args.append(negative)
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


def maybe_load_env_file(path: str, *, override: bool = False) -> bool:
    if _load_dotenv is None:
        return False
    return bool(_load_dotenv(path, override=override))


def normalize_api_key_env_vars(raw_value: Any) -> list[str]:
    if raw_value is None:
        return list(DEFAULT_API_KEY_ENV_VARS)
    if isinstance(raw_value, str):
        text = raw_value.strip()
        if not text:
            return list(DEFAULT_API_KEY_ENV_VARS)
        if text.startswith("["):
            try:
                raw_value = json.loads(text)
            except json.JSONDecodeError:
                raw_value = [part for part in text.replace(",", " ").split() if part.strip()]
        else:
            raw_value = [part for part in text.replace(",", " ").split() if part.strip()]
    if not isinstance(raw_value, (list, tuple)):
        raise ValueError("api_key_env_vars must be a list of environment variable names.")
    env_vars: list[str] = []
    seen: set[str] = set()
    for item in raw_value:
        name = str(item or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        env_vars.append(name)
    if not env_vars:
        raise ValueError("At least one api_key_env_var is required.")
    return env_vars


def resolve_api_key(explicit_api_key: str, api_key_env_var: str) -> str:
    explicit = str(explicit_api_key or "").strip()
    if explicit:
        return explicit
    for name in (str(api_key_env_var or "").strip(), DEFAULT_API_KEY_ENV_VAR, "MOONDREAM_API_KEY"):
        if not name:
            continue
        value = str(os.environ.get(name) or "").strip()
        if value:
            return value
    raise ValueError("Moondream API key is required.")


def resolve_api_key_pool(
    *,
    explicit_api_key: str = "",
    api_key_env_vars: Optional[Iterable[str]] = None,
) -> ApiKeyPool:
    explicit = str(explicit_api_key or "").strip()
    if explicit:
        return ApiKeyPool([ApiKeySlot(index=0, env_var="<explicit>", api_key=explicit)])
    env_vars = normalize_api_key_env_vars(api_key_env_vars)
    slots: list[ApiKeySlot] = []
    missing: list[str] = []
    for index, env_var in enumerate(env_vars):
        api_key = str(os.environ.get(env_var) or "").strip()
        if not api_key:
            missing.append(env_var)
            continue
        slots.append(ApiKeySlot(index=index, env_var=env_var, api_key=api_key))
    if missing:
        raise ValueError(f"Missing Moondream API key(s) for env vars: {missing}")
    return ApiKeyPool(slots)


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def tokenize_text(value: Any) -> list[str]:
    return [match.group(0).lower() for match in _TOKEN_PATTERN.finditer(str(value or ""))]


def token_f1(reference: Any, prediction: Any) -> float:
    ref_tokens = tokenize_text(reference)
    pred_tokens = tokenize_text(prediction)
    if not ref_tokens and not pred_tokens:
        return 1.0
    if not ref_tokens or not pred_tokens:
        return 0.0
    ref_counts: dict[str, int] = {}
    pred_counts: dict[str, int] = {}
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    overlap = 0
    for token, count in ref_counts.items():
        overlap += min(count, pred_counts.get(token, 0))
    if overlap <= 0:
        return 0.0
    precision = overlap / float(len(pred_tokens))
    recall = overlap / float(len(ref_tokens))
    denom = precision + recall
    return 0.0 if denom <= 0.0 else (2.0 * precision * recall) / denom


def set_f1(reference: Iterable[Any], prediction: Iterable[Any]) -> float:
    ref = {normalize_text(item) for item in reference if normalize_text(item)}
    pred = {normalize_text(item) for item in prediction if normalize_text(item)}
    if not ref and not pred:
        return 1.0
    if not ref or not pred:
        return 0.0
    overlap = len(ref & pred)
    precision = overlap / float(len(pred))
    recall = overlap / float(len(ref))
    denom = precision + recall
    return 0.0 if denom <= 0.0 else (2.0 * precision * recall) / denom


def json_object_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    n = len(text)
    for start in range(n):
        if text[start] != "{":
            continue
        depth = 0
        in_string = False
        escaped = False
        for idx in range(start, n):
            char = text[idx]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
                continue
            if char == "{":
                depth += 1
                continue
            if char == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : idx + 1])
                    break
                if depth < 0:
                    break
    return candidates


def parse_prediction_json(answer_text: str) -> Optional[dict[str, Any]]:
    text = str(answer_text or "").strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    for candidate in json_object_candidates(text):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def to_data_url(image: Image.Image, *, quality: int = 92) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=max(1, min(100, int(quality))))
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def build_auth_headers(api_key: str) -> dict[str, str]:
    header_name = os.environ.get("MOONDREAM_AUTH_HEADER", "X-Moondream-Auth")
    user_agent = os.environ.get("MOONDREAM_USER_AGENT") or "inspector-md-client/0.1"
    key = str(api_key).strip()
    if header_name.lower() == "authorization" and not key.lower().startswith("bearer "):
        key = f"Bearer {key}"
    return {
        "Content-Type": "application/json",
        "Accept": "application/json",
        header_name: key,
        "User-Agent": user_agent,
    }


def truncate(value: Any, *, limit: int = 240) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(text), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(payload)
    return rows
