from __future__ import annotations

import time
from typing import Any, Callable, Optional

from tuna_sdk import TunaClient
from tuna_sdk.errors import TunaAPIError, TunaNetworkError


ErrorFormatter = Callable[[Exception], str]


def default_error_message(exc: Exception) -> str:
    if isinstance(exc, TunaAPIError):
        request_id = f" request_id={exc.request_id}" if getattr(exc, "request_id", "") else ""
        return f"TunaAPIError status={exc.status_code}{request_id} message={exc}"
    if isinstance(exc, TunaNetworkError):
        cause = getattr(exc, "cause", None)
        if cause is not None:
            return f"TunaNetworkError message={exc} cause={type(cause).__name__}: {cause}"
    return f"{type(exc).__name__}: {exc}"


def checkpoint_step_from_save_result(saved_checkpoint: Any) -> Optional[int]:
    checkpoint = getattr(saved_checkpoint, "checkpoint", None)
    raw_step = getattr(checkpoint, "step", None)
    if raw_step is None:
        return None
    try:
        return int(raw_step)
    except (TypeError, ValueError):
        return None


def save_checkpoint_step(
    *,
    finetune: Any,
    context: str,
    error_formatter: Optional[ErrorFormatter] = None,
) -> Optional[int]:
    format_error = error_formatter or default_error_message
    try:
        saved_checkpoint = finetune.save_checkpoint()
        return checkpoint_step_from_save_result(saved_checkpoint)
    except (TunaAPIError, TunaNetworkError) as exc:
        print(f"{context}: checkpoint save failed; continuing. details={format_error(exc)}")
        return None


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
            f"details={default_error_message(exc)}"
        ) from exc
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()


def wait_for_exact_checkpoint(
    *,
    api_base: str,
    api_key: str,
    finetune_id: str,
    checkpoint_step: int,
    timeout: float = 180.0,
    ready_max_wait_s: float = 0.0,
    ready_poll_interval_s: float = 5.0,
) -> list[int]:
    deadline = time.monotonic() + max(0.0, float(ready_max_wait_s))
    last_steps: list[int] = []
    while True:
        last_steps = list_saved_checkpoint_steps(
            api_base=api_base,
            api_key=api_key,
            finetune_id=finetune_id,
            timeout=timeout,
        )
        if int(checkpoint_step) in last_steps:
            return last_steps
        if time.monotonic() >= deadline:
            return last_steps
        time.sleep(max(0.1, float(ready_poll_interval_s)))


def format_checkpoint_steps(steps: list[int], *, limit: int = 20) -> str:
    if not steps:
        return "(none)"
    shown = [str(int(step)) for step in steps[:limit]]
    if len(steps) > limit:
        shown.append("...")
    return ", ".join(shown)


def resolve_checkpoint_step(
    *,
    api_base: str,
    api_key: str,
    finetune_id: str,
    requested_step: int,
    timeout: float = 180.0,
    fallback_policy: str = "nearest_saved",
    ready_max_wait_s: float = 0.0,
    ready_poll_interval_s: float = 5.0,
) -> tuple[int, bool]:
    requested = int(requested_step)
    if fallback_policy == "exact":
        saved_steps = wait_for_exact_checkpoint(
            api_base=api_base,
            api_key=api_key,
            finetune_id=finetune_id,
            checkpoint_step=requested,
            timeout=timeout,
            ready_max_wait_s=ready_max_wait_s,
            ready_poll_interval_s=ready_poll_interval_s,
        )
        if requested not in saved_steps:
            raise ValueError(
                f"requested checkpoint step={requested} is not available for "
                f"finetune_id={finetune_id}. saved_steps={format_checkpoint_steps(saved_steps)}"
            )
        return requested, False

    saved_steps = list_saved_checkpoint_steps(
        api_base=api_base,
        api_key=api_key,
        finetune_id=finetune_id,
        timeout=timeout,
    )
    if not saved_steps:
        raise ValueError(
            f"finetune_id={finetune_id} has no saved checkpoints available for inference."
        )
    if requested in saved_steps:
        return requested, False
    if fallback_policy != "nearest_saved":
        raise ValueError(f"unsupported checkpoint fallback policy: {fallback_policy}")
    eligible_steps = [int(step) for step in saved_steps if int(step) <= requested]
    if not eligible_steps:
        raise ValueError(
            f"requested checkpoint step={requested} is earlier than all saved checkpoints for "
            f"finetune_id={finetune_id}. saved_steps={format_checkpoint_steps(saved_steps)}"
        )
    return int(eligible_steps[-1]), True
