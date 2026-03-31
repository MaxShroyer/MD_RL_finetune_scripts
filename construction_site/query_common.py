from __future__ import annotations

import base64
import io
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from PIL import Image

os.environ.setdefault("WANDB_START_METHOD", "thread")
os.environ.setdefault("WANDB__SERVICE_WAIT", "300")

try:
    from dotenv import load_dotenv as _third_party_load_dotenv
except ModuleNotFoundError:  # pragma: no cover
    _third_party_load_dotenv = None

try:
    from tqdm.auto import tqdm  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    class _SimpleTqdm:
        def __init__(self, iterable=None, *_args: Any, **_kwargs: Any) -> None:
            self._iterable = iterable or []

        def __iter__(self):
            return iter(self._iterable)

        def set_postfix(self, *_args: Any, **_kwargs: Any) -> None:
            return

    def tqdm(iterable=None, *args, **kwargs):  # type: ignore
        return _SimpleTqdm(iterable, *args, **kwargs)

try:
    import wandb  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    class _WandbRun:
        def __init__(self) -> None:
            self.summary: dict[str, Any] = {}

        def finish(self) -> None:
            return

    class _WandbShim:
        @staticmethod
        def init(*_args: Any, **_kwargs: Any) -> _WandbRun:
            print("wandb not installed; continuing without remote logging.")
            return _WandbRun()

        @staticmethod
        def log(*_args: Any, **_kwargs: Any) -> None:
            return

        @staticmethod
        def finish() -> None:
            return

    wandb = _WandbShim()

from tuna_sdk import QueryRequest, QuerySettings, TunaClient  # noqa: E402
from tuna_sdk.errors import TunaAPIError, TunaNetworkError  # noqa: E402

DEFAULT_PREFLIGHT_QUERY_IMAGE_DATA_URL = (
    "data:image/jpeg;base64,"
    "/9j//gAQTGF2YzYxLjE5LjEwMQD/2wBDAAg+Pkk+SVVVVVVVVWRdZGhoaGRkZGRoaGhwcHCDg4NwcHBoaHBwfHyDg4+Tj4eHg4eTk5ubm7q6srLZ2eD/////xABZAAADAQEBAQAAAAAAAAAAAAAABgcFCAECAQEAAAAAAAAAAAAAAAAAAAAAEAADAAMBAQEBAAAAAAAAAAAAAQIDIREEURKBEQEAAAAAAAAAAAAAAAAAAAAA/8AAEQgAGQAZAwESAAISAAMSAP/aAAwDAQACEQMRAD8A5/PQAAABirHyVS2mUip/Pm4/vQAih9ABuRUrVLqMEALVNead7/pFgAfc+d5NLSEEAAAA/9k="
)
FINETUNED_MODEL_RE = re.compile(
    r"^(?P<base_model>moondream3-preview)/(?P<finetune_id>[0-9A-Za-z_-]+)(?:@(?P<checkpoint_step>\d+))?$"
)


def load_dotenv(path: str, *, override: bool = False) -> bool:
    if _third_party_load_dotenv is None:
        return False
    return bool(_third_party_load_dotenv(path, override=override))


@dataclass(frozen=True)
class ScoreOutcome:
    reward: float
    parse_success: bool
    task_correct: bool
    json_object_parsed: bool


@dataclass(frozen=True)
class QueryModelResolution:
    model: str
    finetune_id: str
    requested_checkpoint_step: Optional[int]
    resolved_checkpoint_step: Optional[int]
    used_checkpoint_fallback: bool = False


class QueryAPIError(Exception):
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


def resolve_env_file(env_file: str, *, repo_root: Path, module_root: Path) -> str:
    path = Path(env_file).expanduser()
    if path.is_absolute():
        return str(path)
    from_cwd = (Path.cwd() / path).resolve()
    if from_cwd.exists():
        return str(from_cwd)
    from_repo = (repo_root / path).resolve()
    if from_repo.exists():
        return str(from_repo)
    from_module = (module_root / path).resolve()
    if from_module.exists():
        return str(from_module)
    return str(from_cwd)


def resolve_path(raw_path: str, *, repo_root: Path, module_root: Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    for base in (Path.cwd(), repo_root, module_root):
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    return (Path.cwd() / path).resolve()


def to_data_url(image: Image.Image, *, quality: int = 92) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=max(1, min(100, int(quality))))
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def load_local_jsonl_rows(*, dataset_dir: Path, split_name: str) -> list[dict[str, Any]]:
    path = dataset_dir / "jsonl" / f"{split_name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"split JSONL not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            rows.append(payload)
    if not rows:
        raise ValueError(f"split={split_name} contains no rows")
    return rows


def existing_path(raw_path: str, *, dataset_dir: Path) -> Optional[Path]:
    text = str(raw_path or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if path.is_file():
        return path.resolve()
    if not path.is_absolute():
        joined = (dataset_dir / path).resolve()
        if joined.is_file():
            return joined
    return None


def prepare_requests(
    examples: list[Any],
    *,
    temperature: float,
    top_p: float,
    max_tokens: int,
    reasoning: bool,
    question_getter: Callable[[Any], str] = lambda item: str(item.question),
    image_path_getter: Callable[[Any], Path] = lambda item: item.image_path,
) -> tuple[list[QueryRequest], list[Any]]:
    requests: list[QueryRequest] = []
    active_examples: list[Any] = []
    for example in examples:
        image_path = image_path_getter(example)
        try:
            with Image.open(image_path) as image:
                image_url = to_data_url(image)
        except (FileNotFoundError, OSError) as exc:
            print(f"image load failed for {image_path}: {exc}; skipping")
            continue
        requests.append(
            QueryRequest(
                question=question_getter(example),
                image_url=image_url,
                reasoning=bool(reasoning),
                settings=QuerySettings(
                    temperature=float(temperature),
                    top_p=float(top_p),
                    max_tokens=int(max_tokens),
                ),
            )
        )
        active_examples.append(example)
    return requests, active_examples


def error_message(exc: Exception) -> str:
    if isinstance(exc, TunaAPIError):
        request_id = f" request_id={exc.request_id}" if exc.request_id else ""
        return f"TunaAPIError status={exc.status_code}{request_id} message={exc}"
    if isinstance(exc, TunaNetworkError):
        cause = getattr(exc, "cause", None)
        if cause is not None:
            return f"TunaNetworkError message={exc} cause={type(cause).__name__}: {cause}"
    if isinstance(exc, QueryAPIError):
        request_id = f" request_id={exc.request_id}" if exc.request_id else ""
        response_body = f" body={truncate(exc.response_body)}" if exc.response_body else ""
        return f"QueryAPIError status={exc.status_code}{request_id} message={exc}{response_body}"
    return f"{type(exc).__name__}: {exc}"


def rollouts_batch_with_retry(
    *,
    finetune: Any,
    requests: list[QueryRequest],
    num_rollouts: int,
    max_workers: int,
    retries: int,
    backoff_s: float,
    context: str,
) -> Any:
    worker_count = max(1, min(max_workers, len(requests)))
    for attempt in range(max(0, int(retries)) + 1):
        try:
            return finetune.rollouts_batch(
                requests=requests,
                num_rollouts=num_rollouts,
                max_workers=worker_count,
            )
        except (TunaAPIError, TunaNetworkError) as exc:
            should_retry = isinstance(exc, TunaNetworkError) or (
                isinstance(exc, TunaAPIError) and exc.status_code == 429
            ) or "too many requests" in str(exc).lower()
            if not should_retry or attempt >= int(retries):
                print(
                    f"{context}: rollouts_batch failed with no further retries. "
                    f"attempt={attempt + 1}/{int(retries) + 1} workers={worker_count} "
                    f"details={error_message(exc)}"
                )
                raise
            delay = max(0.1, float(backoff_s)) * (2**attempt)
            next_workers = max(1, worker_count // 2)
            print(
                f"{context}: retrying rollouts_batch attempt={attempt + 1}/{int(retries) + 1} "
                f"workers={worker_count}->{next_workers} sleep={delay:.2f}s details={error_message(exc)}"
            )
            time.sleep(delay)
            worker_count = next_workers


def _checkpoint_step_from_save_result(saved_checkpoint: Any) -> Optional[int]:
    checkpoint = getattr(saved_checkpoint, "checkpoint", None)
    raw_step = getattr(checkpoint, "step", None)
    if raw_step is None:
        return None
    try:
        return int(raw_step)
    except (TypeError, ValueError):
        return None


def save_checkpoint(*, finetune: Any, context: str) -> Optional[int]:
    try:
        saved_checkpoint = finetune.save_checkpoint()
        return _checkpoint_step_from_save_result(saved_checkpoint)
    except (TunaAPIError, TunaNetworkError) as exc:
        print(f"{context}: checkpoint save failed; continuing. details={error_message(exc)}")
        return None


def compose_train_groups(
    *,
    on_policy_groups: list[Any],
    replay_groups: deque[Any],
    off_policy: bool,
    off_policy_mix_ratio: float,
    off_policy_warmup_steps: int,
    off_policy_min_buffer_groups: int,
    global_step: int,
    rng: Any,
) -> tuple[list[Any], int]:
    if (
        not on_policy_groups
        or not off_policy
        or off_policy_mix_ratio <= 0.0
        or global_step < off_policy_warmup_steps
        or len(replay_groups) < off_policy_min_buffer_groups
    ):
        return list(on_policy_groups), 0
    off_policy_count = min(
        max(1, int(round(len(on_policy_groups) * off_policy_mix_ratio))),
        len(on_policy_groups),
        len(replay_groups),
    )
    keep_count = max(0, len(on_policy_groups) - off_policy_count)
    selected_on_policy = (
        list(on_policy_groups)
        if keep_count >= len(on_policy_groups)
        else rng.sample(list(on_policy_groups), k=keep_count)
    )
    mixed = selected_on_policy + rng.sample(list(replay_groups), k=off_policy_count)
    rng.shuffle(mixed)
    return mixed, off_policy_count


def progress_enabled(no_progress: bool) -> bool:
    if no_progress:
        return False
    return os.isatty(2)


def truncate(text: str, limit: int = 600) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "...<truncated>"


def build_auth_headers(api_key: str) -> dict[str, str]:
    header_name = os.environ.get("MOONDREAM_AUTH_HEADER", "X-Moondream-Auth")
    user_agent = os.environ.get("MOONDREAM_USER_AGENT") or (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
    key = api_key.strip()
    if header_name.lower() == "authorization" and not key.lower().startswith("bearer "):
        key = f"Bearer {key}"
    return {
        "Content-Type": "application/json",
        "Accept": "application/json",
        header_name: key,
        "User-Agent": user_agent,
    }


def resolve_model_identifier(
    *,
    model: str,
    finetune_id: str,
    checkpoint_step: Optional[int],
) -> str:
    if model.strip():
        return model.strip()
    ftid = finetune_id.strip()
    if ftid:
        if checkpoint_step is not None:
            return f"moondream3-preview/{ftid}@{checkpoint_step}"
        return f"moondream3-preview/{ftid}"
    return "moondream3-preview"


def _parse_finetuned_model(model: str) -> Optional[tuple[str, Optional[int]]]:
    match = FINETUNED_MODEL_RE.fullmatch(str(model or "").strip())
    if not match:
        return None
    finetune_id = str(match.group("finetune_id") or "").strip()
    checkpoint_raw = match.group("checkpoint_step")
    checkpoint_step = None if checkpoint_raw is None else int(checkpoint_raw)
    return finetune_id, checkpoint_step


def _format_checkpoint_steps(steps: list[int], *, limit: int = 20) -> str:
    if not steps:
        return "(none)"
    shown = [str(int(step)) for step in steps[:limit]]
    if len(steps) > limit:
        shown.append("...")
    return ", ".join(shown)


def list_saved_checkpoint_steps(
    *,
    api_base: str,
    api_key: str,
    finetune_id: str,
    timeout: float = 180.0,
) -> list[int]:
    client = TunaClient(api_key=api_key, base_url=api_base, timeout=float(timeout))
    try:
        finetune = client.get_finetune(finetune_id)
        cursor: Optional[str] = None
        steps: set[int] = set()
        while True:
            page = finetune.list_checkpoints(limit=100, cursor=cursor)
            for checkpoint in page.checkpoints:
                steps.add(int(checkpoint.step))
            if not page.has_more or not page.next_cursor:
                break
            cursor = page.next_cursor
        return sorted(steps)
    except (TunaAPIError, TunaNetworkError) as exc:
        raise ValueError(
            f"unable to list saved checkpoints for finetune_id={finetune_id}. "
            f"details={error_message(exc)}"
        ) from exc
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()


def resolve_query_inference_model(
    *,
    api_base: str,
    api_key: str,
    model: str,
    finetune_id: str,
    checkpoint_step: Optional[int],
    timeout: float = 180.0,
    fallback_policy: str = "nearest_saved",
) -> QueryModelResolution:
    raw_model = str(model or "").strip()
    raw_finetune_id = str(finetune_id or "").strip()
    requested_step = None if checkpoint_step is None else int(checkpoint_step)
    parsed_model = _parse_finetuned_model(raw_model) if raw_model else None

    if raw_model:
        if parsed_model is None:
            return QueryModelResolution(
                model=raw_model,
                finetune_id="",
                requested_checkpoint_step=None,
                resolved_checkpoint_step=None,
            )
        parsed_finetune_id, parsed_step = parsed_model
        if parsed_step is None:
            raise ValueError(
                "finetuned model strings must include a checkpoint step. "
                "Use moondream3-preview/{finetune_id}@{step}."
            )
        raw_finetune_id = parsed_finetune_id
        requested_step = int(parsed_step)
    elif raw_finetune_id:
        if requested_step is None:
            raise ValueError("--finetune-id requires --checkpoint-step for finetuned inference.")
    else:
        return QueryModelResolution(
            model="moondream3-preview",
            finetune_id="",
            requested_checkpoint_step=None,
            resolved_checkpoint_step=None,
        )

    if requested_step is None:
        raise ValueError("checkpoint step is required for finetuned inference.")

    saved_steps = list_saved_checkpoint_steps(
        api_base=api_base,
        api_key=api_key,
        finetune_id=raw_finetune_id,
        timeout=timeout,
    )
    if not saved_steps:
        raise ValueError(
            f"finetune_id={raw_finetune_id} has no saved checkpoints available for inference."
        )
    if requested_step in saved_steps:
        resolved_step = int(requested_step)
        used_fallback = False
    else:
        if fallback_policy != "nearest_saved":
            raise ValueError(f"unsupported checkpoint fallback policy: {fallback_policy}")
        eligible_steps = [int(step) for step in saved_steps if int(step) <= requested_step]
        if not eligible_steps:
            raise ValueError(
                f"requested checkpoint step={requested_step} is earlier than all saved checkpoints for "
                f"finetune_id={raw_finetune_id}. saved_steps={_format_checkpoint_steps(saved_steps)}"
            )
        resolved_step = int(eligible_steps[-1])
        used_fallback = True
        print(
            "checkpoint resolution: "
            f"finetune_id={raw_finetune_id} requested_step={requested_step} "
            f"resolved_step={resolved_step} policy={fallback_policy}"
        )
    return QueryModelResolution(
        model=f"moondream3-preview/{raw_finetune_id}@{resolved_step}",
        finetune_id=raw_finetune_id,
        requested_checkpoint_step=int(requested_step),
        resolved_checkpoint_step=int(resolved_step),
        used_checkpoint_fallback=used_fallback,
    )


def build_query_payload(
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
            nested = output.get("answer")
            if isinstance(nested, str):
                return nested
    return ""


def _http_error_details(exc: urllib.error.HTTPError) -> tuple[str, str]:
    request_id = str(exc.headers.get("x-request-id") or exc.headers.get("X-Request-Id") or "")
    body_text = ""
    try:
        body_text = exc.read().decode("utf-8", errors="replace")
    except Exception:
        body_text = ""
    return request_id, body_text


def call_query_api(
    *,
    api_base: str,
    api_key: str,
    model: str,
    question: str,
    image_url: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    reasoning: Optional[bool],
    timeout: float,
    retry_429_max_retries: int,
    retry_429_backoff_s: float,
    retry_429_max_backoff_s: float,
    retry_5xx_max_retries: int,
    retry_5xx_backoff_s: float,
    retry_5xx_max_backoff_s: float,
) -> tuple[str, dict[str, Any], float]:
    payload = build_query_payload(
        model=model,
        question=question,
        image_url=image_url,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        reasoning=reasoning,
    )
    endpoint = api_base.rstrip("/") + "/query"
    retry_429_limit = max(0, int(retry_429_max_retries))
    retry_5xx_limit = max(0, int(retry_5xx_max_retries))
    retry_429_attempt = 0
    retry_5xx_attempt = 0
    while True:
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=build_auth_headers(api_key),
            method="POST",
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=float(timeout)) as response:
                body = response.read().decode("utf-8", errors="replace")
            latency_ms = (time.monotonic() - started) * 1000.0
            data = json.loads(body) if body else {}
            if not isinstance(data, dict):
                data = {}
            return _extract_answer_text(data), data, latency_ms
        except urllib.error.HTTPError as exc:
            latency_ms = (time.monotonic() - started) * 1000.0
            request_id, body_text = _http_error_details(exc)
            retry_after_s = 0.0
            if exc.code == 429:
                header = (exc.headers.get("Retry-After") or "").strip()
                if header:
                    try:
                        retry_after_s = max(0.0, float(header))
                    except (TypeError, ValueError):
                        retry_after_s = 0.0
            if exc.code == 429 and retry_429_attempt < retry_429_limit:
                exp_backoff = max(0.0, float(retry_429_backoff_s)) * (2.0**retry_429_attempt)
                capped_backoff = min(max(0.0, float(retry_429_max_backoff_s)), exp_backoff)
                sleep_s = max(retry_after_s, capped_backoff)
                print(
                    "query retry: "
                    f"status=429 attempt={retry_429_attempt + 1}/{retry_429_limit + 1} "
                    f"sleep={sleep_s:.2f}s latency_ms={latency_ms:.1f} "
                    f"request_id={request_id or '-'} body={truncate(body_text)}"
                )
                if sleep_s > 0.0:
                    time.sleep(sleep_s)
                retry_429_attempt += 1
                continue
            if 500 <= exc.code <= 599 and retry_5xx_attempt < retry_5xx_limit:
                exp_backoff = max(0.0, float(retry_5xx_backoff_s)) * (2.0**retry_5xx_attempt)
                capped_backoff = min(max(0.0, float(retry_5xx_max_backoff_s)), exp_backoff)
                sleep_s = max(retry_after_s, capped_backoff)
                print(
                    "query retry: "
                    f"status={exc.code} attempt={retry_5xx_attempt + 1}/{retry_5xx_limit + 1} "
                    f"sleep={sleep_s:.2f}s latency_ms={latency_ms:.1f} "
                    f"request_id={request_id or '-'} body={truncate(body_text)}"
                )
                if sleep_s > 0.0:
                    time.sleep(sleep_s)
                retry_5xx_attempt += 1
                continue
            raise QueryAPIError(
                f"HTTP {exc.code} {exc.reason}",
                status_code=exc.code,
                request_id=request_id,
                response_body=body_text,
            )
        except urllib.error.URLError as exc:
            raise QueryAPIError(f"Network error: {exc}") from exc


def preflight_query_api(
    *,
    api_base: str,
    api_key: str,
    model: str,
    timeout: float,
    reasoning: Optional[bool],
    retry_429_max_retries: int,
    retry_429_backoff_s: float,
    retry_429_max_backoff_s: float,
    retry_5xx_max_retries: int,
    retry_5xx_backoff_s: float,
    retry_5xx_max_backoff_s: float,
) -> tuple[str, dict[str, Any], float]:
    return call_query_api(
        api_base=api_base,
        api_key=api_key,
        model=model,
        question="What is in this image? Reply briefly.",
        image_url=DEFAULT_PREFLIGHT_QUERY_IMAGE_DATA_URL,
        temperature=0.0,
        top_p=1.0,
        max_tokens=32,
        reasoning=reasoning,
        timeout=timeout,
        retry_429_max_retries=retry_429_max_retries,
        retry_429_backoff_s=retry_429_backoff_s,
        retry_429_max_backoff_s=retry_429_max_backoff_s,
        retry_5xx_max_retries=retry_5xx_max_retries,
        retry_5xx_backoff_s=retry_5xx_backoff_s,
        retry_5xx_max_backoff_s=retry_5xx_max_backoff_s,
    )
