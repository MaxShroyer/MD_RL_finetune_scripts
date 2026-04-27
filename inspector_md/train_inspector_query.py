#!/usr/bin/env python3
"""Train Inspector MD query finetunes for full-image JSON issue-list generation."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import string
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from types import SimpleNamespace
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from async_checkpoint_eval import dispatch_checkpoint_eval, drain_checkpoint_eval_jobs, poll_checkpoint_eval_jobs


def _prime_site_wandb() -> None:
    if "wandb" in sys.modules:
        return
    original_sys_path = list(sys.path)
    repo_root = REPO_ROOT.resolve()
    cwd_root = Path.cwd().resolve()

    def _is_repo_or_cwd_path(entry: str) -> bool:
        try:
            resolved = Path(entry).resolve()
        except Exception:
            return False
        return (
            resolved == repo_root
            or resolved == cwd_root
            or repo_root in resolved.parents
            or cwd_root in resolved.parents
        )

    try:
        sys.path[:] = [
            entry
            for entry in original_sys_path
            if entry and not _is_repo_or_cwd_path(entry)
        ]
        try:
            importlib.import_module("wandb")
        except ModuleNotFoundError:
            return
    finally:
        sys.path[:] = original_sys_path


_prime_site_wandb()

from inspector_md import common, ontology, openrouter_grader, prompt_library, query_compact, task_schema
from inspector_md.moondream_client import MoondreamInferenceError, MoondreamInspectorClient
from tuna_sdk import QueryRequest, QuerySFTTarget, QuerySettings, TrainStepGroup, TunaClient
from tuna_sdk.errors import TunaAPIError, TunaNetworkError
from tuna_sdk.retry import RetryConfig, compute_backoff_delay

try:
    from finetune_checkpoints import resolve_checkpoint_step
except ModuleNotFoundError:  # pragma: no cover
    from pathlib import Path as _PathForImport

    sys.path.append(str(_PathForImport(__file__).resolve().parents[1]))
    from finetune_checkpoints import resolve_checkpoint_step

class _WandbRun:
    def __init__(self) -> None:
        self.summary: dict[str, Any] = {}

    def finish(self) -> None:
        return


class _WandbShim:
    @staticmethod
    def init(*args: Any, **kwargs: Any) -> _WandbRun:
        print("wandb not installed; continuing without remote logging.")
        return _WandbRun()

    @staticmethod
    def log(*args: Any, **kwargs: Any) -> None:
        return


try:
    import wandb  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    wandb = _WandbShim()
else:  # pragma: no branch
    if not hasattr(wandb, "init") or not hasattr(wandb, "log"):
        # Local `wandb/` run directories can shadow the package as an empty namespace.
        wandb = _WandbShim()

try:
    from tqdm.auto import tqdm as _tqdm
except ModuleNotFoundError:  # pragma: no cover
    _tqdm = None

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = common.repo_relative("configs", "train_inspector_query_default.json")
TRANSIENT_QUERY_TRAIN_STEP_STATUS_CODES = frozenset({500, 502, 503, 504, 520, 524})
SYNC_EVAL_CHECKPOINT_READY_MAX_WAIT_S = 300.0
SYNC_EVAL_CHECKPOINT_READY_POLL_INTERVAL_S = 5.0
SYNC_EVAL_SUBPROCESS_TIMEOUT_S = 900.0


@dataclass(frozen=True)
class QueryExample:
    row_id: str
    split: str
    task_type: str
    image_path: Path
    inspection_request: str
    asset_context: str
    spatial_refs: list[list[float]]
    question: str
    target_text: str
    target_format: str
    final_answer_json: str
    reasoning_text: str
    hard_example: bool
    query_text_refresh_mode: str
    issue_count: int
    is_multi_issue: bool
    source_dataset: str
    source_annotation_type: str
    crop_derived: bool


@dataclass(frozen=True)
class QueryScoreOutcome:
    reward: float
    parse_success: bool
    task_correct: bool
    json_object_parsed: bool
    issue_precision: float
    issue_recall: float
    issue_f1: float
    reasoning_f1: float
    extra_issue_rate: float
    empty_list_accuracy: float
    predicted_issue_count: int
    verbosity_penalty: float


@dataclass(frozen=True)
class QueryParseOutcome:
    payload: dict[str, Any]
    method: str


class _NullProgressBar:
    def update(self, n: int = 1) -> None:
        return

    def set_postfix(self, *args: Any, **kwargs: Any) -> None:
        return

    def close(self) -> None:
        return


def _make_progress_bar(*, total: int, desc: str):
    if _tqdm is None:
        return _NullProgressBar()
    return _tqdm(total=max(0, int(total)), desc=desc, dynamic_ncols=True, mininterval=1.0)


def _namespaced_wandb_payload(payload: dict[str, Any], *, namespace: str) -> dict[str, Any]:
    output = dict(payload)
    normalized_namespace = str(namespace or "").strip().strip("/")
    if not normalized_namespace:
        return output
    for key, value in payload.items():
        metric_key = str(key)
        if "/" in metric_key:
            continue
        if metric_key.startswith(f"{normalized_namespace}_"):
            metric_key = metric_key[len(normalized_namespace) + 1 :]
        elif metric_key.startswith("eval_") and normalized_namespace in {"eval", "async_eval", "test"}:
            metric_key = metric_key[len("eval_") :]
        elif metric_key.startswith("train_") and normalized_namespace == "train":
            metric_key = metric_key[len("train_") :]
        output[f"{normalized_namespace}/{metric_key}"] = value
    return output


def _wandb_log(payload: dict[str, Any], *, step: int, namespace: str) -> None:
    wandb.log(_namespaced_wandb_payload(payload, namespace=namespace), step=int(step))


def _format_tuna_error(exc: Exception) -> str:
    parts = [f"{type(exc).__name__}: {exc}"]
    if isinstance(exc, TunaAPIError):
        if exc.status_code is not None:
            parts.append(f"status={exc.status_code}")
        if exc.request_id:
            parts.append(f"request_id={exc.request_id}")
        if exc.response_body is not None:
            body = exc.response_body
            if not isinstance(body, str):
                try:
                    body = json.dumps(body, ensure_ascii=True)
                except Exception:
                    body = str(body)
            body_text = str(body).strip().replace("\n", " ")
            if len(body_text) > 800:
                body_text = body_text[:800] + "..."
            if body_text:
                parts.append(f"body={body_text}")
    elif isinstance(exc, TunaNetworkError) and getattr(exc, "cause", None) is not None:
        parts.append(f"cause={exc.cause}")
    return " | ".join(parts)


def _is_transient_query_train_step_error(exc: Exception) -> bool:
    if isinstance(exc, TunaNetworkError):
        return True
    if isinstance(exc, TunaAPIError):
        return int(exc.status_code or 0) in TRANSIENT_QUERY_TRAIN_STEP_STATUS_CODES
    return False


def _is_query_scheduler_capacity_error(exc: Exception) -> bool:
    if not isinstance(exc, TunaAPIError):
        return False
    body = exc.response_body
    if isinstance(body, dict):
        try:
            body_text = json.dumps(body, ensure_ascii=True)
        except Exception:
            body_text = str(body)
    else:
        body_text = str(body or "")
    return "Error scheduling request" in body_text


def _query_train_step_with_retry(
    *,
    invoke: Any,
    context: str,
    max_retries: int,
    backoff_base_s: float,
    backoff_max_s: float,
) -> Any:
    retry_budget = max(0, int(max_retries))
    for attempt in range(retry_budget + 1):
        try:
            return invoke()
        except (TunaAPIError, TunaNetworkError) as exc:
            if (not _is_transient_query_train_step_error(exc)) or attempt >= retry_budget:
                raise
            delay_s = compute_backoff_delay(
                attempt,
                base=max(0.1, float(backoff_base_s)),
                max_delay=max(float(backoff_base_s), float(backoff_max_s)),
                jitter=0.1,
            )
            if _is_query_scheduler_capacity_error(exc):
                delay_s = max(delay_s, min(max(float(backoff_base_s), float(backoff_max_s)), 30.0))
            print(
                f"{context} transient failure attempt {attempt + 1}/{retry_budget + 1}: "
                f"{_format_tuna_error(exc)}; retrying in {delay_s:.1f}s"
            )
            time.sleep(delay_s)
    raise RuntimeError(f"{context} exhausted retry loop")


def _train_query_sft_groups(
    *,
    finetune: Any,
    groups: list[TrainStepGroup],
    lr: float,
    train_step_max_retries: int,
    train_step_retry_backoff_base_s: float,
    train_step_retry_backoff_max_s: float,
    step_label: str,
) -> Any:
    if not groups:
        raise ValueError("query SFT requires at least one train step group")
    responses: list[Any] = []
    failed_microbatch_count = 0
    last_transient_exc: Exception | None = None
    total_groups = len(groups)
    for group_index, group in enumerate(groups, start=1):
        print(f"{step_label} microbatch {group_index}/{total_groups} request started", flush=True)
        try:
            responses.append(
                _query_train_step_with_retry(
                    invoke=lambda current_group=group: finetune.train_step(groups=[current_group], lr=float(lr)),
                    context=f"{step_label} microbatch {group_index}/{total_groups}",
                    max_retries=int(train_step_max_retries),
                    backoff_base_s=float(train_step_retry_backoff_base_s),
                    backoff_max_s=float(train_step_retry_backoff_max_s),
                )
            )
        except (TunaAPIError, TunaNetworkError) as exc:
            if not _is_transient_query_train_step_error(exc):
                raise
            failed_microbatch_count += 1
            last_transient_exc = exc
            print(
                f"{step_label} microbatch {group_index}/{total_groups} dropped after retries: "
                f"{_format_tuna_error(exc)}"
            )
            continue

    def _avg(name: str) -> float:
        values = [float(getattr(item, name, 0.0) or 0.0) for item in responses]
        return fmean(values) if values else 0.0

    return SimpleNamespace(
        step=getattr(responses[-1], "step", None) if responses else None,
        applied=bool(responses) and all(bool(getattr(item, "applied", True)) for item in responses),
        sft_loss=_avg("sft_loss"),
        kl=_avg("kl"),
        router_kl=_avg("router_kl"),
        grad_norm=_avg("grad_norm"),
        reward_mean=_avg("reward_mean"),
        reward_std=_avg("reward_std"),
        microbatch_count=len(responses),
        successful_microbatch_count=len(responses),
        failed_microbatch_count=int(failed_microbatch_count),
        exhausted_transient_failure=last_transient_exc is not None and not responses,
        last_failure=_format_tuna_error(last_transient_exc) if last_transient_exc is not None else "",
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _random_suffix(length: int = 6) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return "".join(random.choices(alphabet, k=length))


def _resolve_question(task_type: str, *, row_id: str, question: str, inspection_request: str, asset_context: str, final_answer_json: str, spatial_refs: list[list[float]]) -> str:
    explicit = str(question or "").strip()
    if explicit:
        return explicit
    if task_type in {"proposal", "issues"}:
        return prompt_library.build_visible_issue_question_with_style(
            inspection_request=inspection_request,
            asset_context=asset_context,
            prompt_style="request_only",
            variation_key=row_id,
        )
    payload = json.loads(final_answer_json)
    finding_payload = payload.get("finding") if isinstance(payload, dict) else {}
    issue_code = ontology.normalize_issue_code((finding_payload or {}).get("issue_code"))
    localized_issue = task_schema.LocalizedIssue(
        issue_code=issue_code,
        box=task_schema.Box.from_payload((spatial_refs or [[0.0, 0.0, 1.0, 1.0]])[0]),
        evidence=[],
        source_detect_labels=[ontology.detect_labels_for_issue(issue_code)[0]],
    )
    return prompt_library.build_finding_question_with_style(
        localized_issue=localized_issue,
        inspection_request=inspection_request,
        asset_context=asset_context,
        prompt_style="minimal",
    )


def _resolve_target_text(task_type: str, *, target_text: str, final_answer_json: str) -> str:
    explicit = str(target_text or "").strip()
    if explicit:
        return explicit
    payload = json.loads(final_answer_json)
    if task_type in {"proposal", "issues"}:
        return query_compact.format_issue_list_target(task_schema.normalize_issue_proposals(payload))
    finding_payload = payload.get("finding") if isinstance(payload, dict) else payload
    if not isinstance(finding_payload, dict):
        return ""
    finding = task_schema.Finding.from_payload(
        {
            "finding_id": "target",
            "issue_code": finding_payload.get("issue_code"),
            "title": finding_payload.get("title"),
            "box": {"x_min": 0.0, "y_min": 0.0, "x_max": 1.0, "y_max": 1.0},
            "evidence": list(finding_payload.get("evidence") or []),
            "severity": finding_payload.get("severity"),
            "recommended_action": finding_payload.get("recommended_action"),
            "cost_band": "unknown",
            "possible_compliance_issue": False,
            "insufficient_evidence": finding_payload.get("insufficient_evidence", False),
            "compliance_note": "",
            "source_detect_labels": ontology.detect_labels_for_issue(finding_payload.get("issue_code"))[:1],
            "spatial_ref_index": 0,
        }
    )
    return query_compact.format_finding_target(finding)


def _parse_spatial_refs(raw_value: Any) -> list[list[float]]:
    if isinstance(raw_value, str):
        text = raw_value.strip()
        raw_value = json.loads(text) if text else []
    if not isinstance(raw_value, list):
        return []
    refs: list[list[float]] = []
    for item in raw_value:
        if isinstance(item, dict):
            box = task_schema.Box.from_payload(item)
            refs.append(prompt_library.box_to_spatial_ref(box))
        elif isinstance(item, list) and len(item) == 4:
            box = task_schema.Box.from_payload(item)
            refs.append(prompt_library.box_to_spatial_ref(box))
    return refs


def _build_example(row: dict[str, Any], *, dataset_dir: Path, split_name: str, line_number: int) -> QueryExample:
    image_path = common.resolve_path(str(row.get("image_path") or ""), module_root=dataset_dir)
    if not image_path.is_file():
        raise FileNotFoundError(f"split={split_name} line={line_number} image_path not found: {image_path}")
    task_type = str(row.get("task_type") or "").strip().lower()
    if task_type not in {"proposal", "finding", "issues"}:
        raise ValueError(f"split={split_name} line={line_number} unsupported task_type={task_type!r}")
    final_answer_json = str(row.get("final_answer_json") or "").strip()
    if not final_answer_json:
        raise ValueError(f"split={split_name} line={line_number} missing final_answer_json")
    spatial_refs = _parse_spatial_refs(row.get("spatial_refs_json"))
    inspection_request = str(row.get("inspection_request") or "").strip()
    asset_context = str(row.get("asset_context") or "").strip()
    target_text = _resolve_target_text(
        task_type,
        target_text=str(row.get("target_text") or ""),
        final_answer_json=final_answer_json,
    )
    if not target_text:
        raise ValueError(f"split={split_name} line={line_number} missing target_text")
    question = _resolve_question(
        task_type,
        row_id=str(row.get("row_id") or f"{split_name}_{line_number:06d}"),
        question=str(row.get("question") or ""),
        inspection_request=inspection_request,
        asset_context=asset_context,
        final_answer_json=final_answer_json,
        spatial_refs=spatial_refs,
    )
    return QueryExample(
        row_id=str(row.get("row_id") or f"{split_name}_{line_number:06d}"),
        split=str(row.get("split") or split_name),
        task_type=task_type,
        image_path=image_path,
        inspection_request=inspection_request,
        asset_context=asset_context,
        spatial_refs=spatial_refs,
        question=question,
        target_text=target_text,
        target_format=str(row.get("target_format") or query_compact.TARGET_FORMAT_JSON_ISSUE_LIST).strip() or query_compact.TARGET_FORMAT_JSON_ISSUE_LIST,
        final_answer_json=final_answer_json,
        reasoning_text=str(row.get("reasoning_text") or "").strip(),
        hard_example=bool(row.get("hard_example", False)),
        query_text_refresh_mode=str(row.get("query_text_refresh_mode") or "").strip(),
        issue_count=int(row.get("issue_count", 0) or 0),
        is_multi_issue=bool(row.get("is_multi_issue", False)),
        source_dataset=str(row.get("source_dataset") or "").strip(),
        source_annotation_type=str(row.get("source_annotation_type") or "").strip(),
        crop_derived=bool(row.get("crop_derived", False)),
    )


def _load_split_examples(*, split_name: str, dataset_dir: Path) -> list[QueryExample]:
    rows = common.load_jsonl(dataset_dir / "jsonl" / f"{split_name}.jsonl")
    examples = [
        _build_example(row, dataset_dir=dataset_dir, split_name=split_name, line_number=index)
        for index, row in enumerate(rows, start=1)
    ]
    if not examples:
        raise ValueError(f"split={split_name} contains no usable rows")
    return examples


def _parse_issue_payload(payload: Any) -> tuple[list[task_schema.IssueProposal], bool]:
    try:
        proposals = task_schema.normalize_issue_proposals(payload, loose_issue_codes=True)
    except Exception:
        return [], isinstance(payload, dict)
    return proposals, isinstance(payload, dict)


def _target_issue_proposals(example: QueryExample) -> list[task_schema.IssueProposal]:
    payload = json.loads(example.final_answer_json)
    proposals = task_schema.normalize_issue_proposals(payload)
    if proposals:
        return proposals
    if example.task_type not in {"proposal", "issues"}:
        finding_payload = payload.get("finding") if isinstance(payload, dict) else payload
        if isinstance(finding_payload, dict):
            issue_code = str(finding_payload.get("issue_code") or "").strip()
            evidence = " ".join(
                str(item).strip()
                for item in list(finding_payload.get("evidence") or [])
                if str(item).strip()
            )
            if issue_code:
                return [
                    task_schema.IssueProposal.from_payload(
                        {
                            "type": issue_code,
                            "reasoning": evidence,
                        }
                    )
                ]
    return []


def _set_precision_recall(reference: list[str], prediction: list[str]) -> tuple[float, float]:
    ref = {common.normalize_text(item) for item in reference if common.normalize_text(item)}
    pred = {common.normalize_text(item) for item in prediction if common.normalize_text(item)}
    if not ref and not pred:
        return 1.0, 1.0
    if not pred:
        return 1.0 if not ref else 0.0, 0.0 if ref else 1.0
    if not ref:
        return 0.0, 1.0
    overlap = len(ref & pred)
    return overlap / float(len(pred)), overlap / float(len(ref))


def _reasoning_f1_for_matches(
    *,
    target: list[task_schema.IssueProposal],
    predicted: list[task_schema.IssueProposal],
) -> float:
    if not target and not predicted:
        return 1.0
    predicted_by_code = {item.issue_code: str(item.evidence or "").strip() for item in predicted}
    values = [
        common.token_f1(str(item.evidence or "").strip(), predicted_by_code.get(item.issue_code, ""))
        for item in target
    ]
    return fmean(values) if values else 0.0


def _issue_list_verbosity_penalty(
    *,
    target: list[task_schema.IssueProposal],
    predicted: list[task_schema.IssueProposal],
) -> float:
    if not predicted:
        return 0.0
    target_count = len(target)
    predicted_count = len(predicted)
    count_penalty = min(1.0, max(0, predicted_count - max(1, target_count + 1)) / float(max(1, target_count + 1)))
    long_reason_count = 0
    for item in predicted:
        if len(common.tokenize_text(item.evidence)) > 24:
            long_reason_count += 1
    reason_penalty = long_reason_count / float(max(1, predicted_count))
    return min(1.0, (0.6 * count_penalty) + (0.4 * reason_penalty))


def _score_issue_payload(example: QueryExample, payload: Any) -> QueryScoreOutcome:
    predicted, json_object_parsed = _parse_issue_payload(payload)
    target = _target_issue_proposals(example)
    predicted_codes = [item.issue_code for item in predicted]
    target_codes = [item.issue_code for item in target]
    issue_precision, issue_recall = _set_precision_recall(target_codes, predicted_codes)
    issue_f1 = common.set_f1(target_codes, predicted_codes)
    reasoning_f1 = _reasoning_f1_for_matches(target=target, predicted=predicted)
    predicted_code_set = set(predicted_codes)
    target_code_set = set(target_codes)
    extra_issue_count = len(predicted_code_set - target_code_set)
    extra_issue_rate = extra_issue_count / float(max(1, len(predicted_code_set))) if predicted_code_set else 0.0
    target_empty = not target_code_set
    predicted_empty = not predicted_code_set
    empty_list_accuracy = 1.0 if target_empty == predicted_empty else 0.0
    verbosity_penalty = _issue_list_verbosity_penalty(target=target, predicted=predicted)
    parse_success = bool(json_object_parsed or predicted or (target_empty and isinstance(payload, dict)))
    reward = common.clamp(
        (0.45 * issue_recall)
        + (0.20 * issue_precision)
        + (0.20 * reasoning_f1)
        + (0.15 * empty_list_accuracy)
        - (0.10 * extra_issue_rate)
        - (0.05 * verbosity_penalty)
    ) if parse_success else 0.0
    return QueryScoreOutcome(
        reward=reward,
        parse_success=parse_success,
        task_correct=target_code_set == predicted_code_set and reasoning_f1 >= 0.999,
        json_object_parsed=json_object_parsed,
        issue_precision=issue_precision,
        issue_recall=issue_recall,
        issue_f1=issue_f1,
        reasoning_f1=reasoning_f1,
        extra_issue_rate=extra_issue_rate,
        empty_list_accuracy=empty_list_accuracy,
        predicted_issue_count=len(predicted),
        verbosity_penalty=verbosity_penalty,
    )


def _score_payload_for_example(example: QueryExample, payload: Any) -> QueryScoreOutcome:
    if example.task_type in {"proposal", "issues"}:
        return _score_issue_payload(example, payload)
    finding_payload = payload.get("finding") if isinstance(payload, dict) and isinstance(payload.get("finding"), dict) else payload
    if not isinstance(finding_payload, dict):
        return QueryScoreOutcome(
            reward=0.0,
            parse_success=False,
            task_correct=False,
            json_object_parsed=isinstance(payload, dict),
            issue_precision=0.0,
            issue_recall=0.0,
            issue_f1=0.0,
            reasoning_f1=0.0,
            extra_issue_rate=1.0,
            empty_list_accuracy=0.0,
            predicted_issue_count=0,
            verbosity_penalty=0.0,
        )
    legacy_issue = str(finding_payload.get("issue_code") or "").strip()
    legacy_evidence = " ".join(str(item).strip() for item in list(finding_payload.get("evidence") or []) if str(item).strip())
    return _score_issue_payload(
        example,
        {"issues": [{"type": legacy_issue, "reasoning": legacy_evidence}] if legacy_issue else []},
    )


def _finding_defaults_from_example(example: QueryExample) -> tuple[str, str]:
    payload = json.loads(example.final_answer_json)
    finding_payload = payload.get("finding") if isinstance(payload, dict) else payload
    if not isinstance(finding_payload, dict):
        return "", ""
    issue_code = ontology.normalize_issue_code(finding_payload.get("issue_code"))
    title = str(finding_payload.get("title") or ontology.get_issue(issue_code).title).strip()
    return issue_code, title


def _parse_answer_payload(
    example: QueryExample,
    answer_text: str,
    *,
    grader: Optional[openrouter_grader.OpenRouterGrader] = None,
) -> QueryParseOutcome:
    text = str(answer_text or "").strip()
    if example.task_type in {"proposal", "issues"}:
        issue_payload, issue_method = query_compact.parse_issue_list_answer_detailed(text)
        if issue_payload is not None:
            return QueryParseOutcome(payload=issue_payload, method=issue_method)
    else:
        default_issue_code, default_title = _finding_defaults_from_example(example)
        compact_payload = query_compact.parse_finding_answer(
            text,
            default_issue_code=default_issue_code,
            default_title=default_title,
        )
        if compact_payload is not None:
            return QueryParseOutcome(payload=compact_payload, method="compact_text")
    json_payload = common.parse_prediction_json(text)
    if isinstance(json_payload, dict):
        return QueryParseOutcome(payload=json_payload, method="json")
    if grader is not None and text:
        if example.task_type in {"proposal", "issues"}:
            normalized = grader.normalize_issues(
                inspection_request=example.inspection_request,
                asset_context=example.asset_context,
                answer_text=text,
            )
            return QueryParseOutcome(payload={"issues": list(normalized.get("issues") or [])}, method="openrouter_normalize")
        default_issue_code, _default_title = _finding_defaults_from_example(example)
        localized_issue = task_schema.LocalizedIssue(
            issue_code=default_issue_code,
            box=task_schema.Box.from_payload((example.spatial_refs or [[0.0, 0.0, 1.0, 1.0]])[0]),
            evidence=[],
            source_detect_labels=ontology.detect_labels_for_issue(default_issue_code)[:1],
        )
        normalized = grader.normalize_finding(
            localized_issue=localized_issue,
            inspection_request=example.inspection_request,
            asset_context=example.asset_context,
            answer_text=text,
        )
        return QueryParseOutcome(payload={"finding": dict(normalized.get("finding") or {})}, method="openrouter_normalize")
    return QueryParseOutcome(payload={}, method="unparsed")


def _score_answer_text(
    example: QueryExample,
    answer_text: str,
    *,
    grader: Optional[openrouter_grader.OpenRouterGrader] = None,
) -> tuple[QueryScoreOutcome, QueryParseOutcome]:
    parse_outcome = _parse_answer_payload(example, answer_text, grader=grader)
    return _score_payload_for_example(example, parse_outcome.payload), parse_outcome


def _build_request(example: QueryExample, *, reasoning: bool, temperature: float, top_p: float, max_tokens: int) -> QueryRequest:
    from PIL import Image

    with Image.open(example.image_path) as image:
        image_url = common.to_data_url(image.convert("RGB"))
    return QueryRequest(
        question=example.question,
        image_url=image_url,
        spatial_refs=example.spatial_refs or None,
        reasoning=bool(reasoning),
        settings=QuerySettings(
            temperature=float(temperature),
            top_p=float(top_p),
            max_tokens=int(max_tokens),
        ),
    )


def _sft_target(example: QueryExample, *, include_reasoning: bool = False) -> QuerySFTTarget:
    reasoning_text = str(example.reasoning_text or "").strip()
    if include_reasoning and reasoning_text:
        return QuerySFTTarget(answer=example.target_text, reasoning=reasoning_text)
    return QuerySFTTarget(answer=example.target_text)


def _evaluate_split(
    *,
    inference_client: MoondreamInspectorClient,
    model: str,
    examples: list[QueryExample],
    split_name: str,
    seed: int,
    max_samples: Optional[int],
    reasoning: bool,
    temperature: float,
    top_p: float,
    max_tokens: int,
    grader: Optional[openrouter_grader.OpenRouterGrader] = None,
    judge_cache: Optional[dict[str, dict[str, Any]]] = None,
    judge_cache_path: Optional[Path] = None,
    predictions_path: Optional[Path] = None,
    answer_parse_mode: str = "grader_normalize",
) -> dict[str, float]:
    parse_mode = str(answer_parse_mode or "grader_normalize").strip().lower()
    if parse_mode not in {"grader_normalize", "strict_raw"}:
        raise ValueError(f"Unsupported answer_parse_mode: {answer_parse_mode!r}")
    parse_grader = grader if parse_mode == "grader_normalize" else None
    indices = list(range(len(examples)))
    random.Random(seed).shuffle(indices)
    if max_samples is not None:
        indices = indices[: max_samples]
    requested_count = len(indices)
    rewards: list[float] = []
    parse_success_count = 0
    task_correct_count = 0
    issue_precision_values: list[float] = []
    issue_recall_values: list[float] = []
    issue_f1_values: list[float] = []
    reasoning_f1_values: list[float] = []
    extra_issue_rate_values: list[float] = []
    empty_list_accuracy_values: list[float] = []
    predicted_issue_count_values: list[float] = []
    verbosity_penalty_values: list[float] = []
    judge_scores: list[float] = []
    local_rewards: list[float] = []
    degraded_count = 0
    inference_error_count = 0
    logged_inference_errors = 0
    handle = None
    if predictions_path is not None:
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        handle = predictions_path.open("w", encoding="utf-8")
    try:
        for index in indices:
            example = examples[index]
            try:
                result = inference_client.query_raw(
                    model=model,
                    request=_build_request(
                        example,
                        reasoning=reasoning,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                    ),
                )
            except MoondreamInferenceError as exc:
                inference_error_count += 1
                if logged_inference_errors < 5:
                    print(
                        f"eval query skipped row_id={example.row_id} split={split_name} "
                        f"due_to={type(exc).__name__}: {exc}"
                    )
                    logged_inference_errors += 1
                elif logged_inference_errors == 5:
                    print("eval query skipping additional inference error logs after first 5 rows")
                    logged_inference_errors += 1
                continue
            outcome, parse_outcome = _score_answer_text(example, result.answer_text, grader=parse_grader)
            local_rewards.append(outcome.reward)
            judge = _judge_answer_with_cache(
                grader=grader,
                cache=judge_cache or {},
                cache_path=judge_cache_path or (predictions_path.parent / "judge_cache.jsonl" if predictions_path is not None else Path("judge_cache.jsonl")),
                example=example,
                answer_text=result.answer_text,
                parse_outcome=parse_outcome,
            ) if grader is not None else {"score": 0.0, "degraded": True}
            if bool(judge.get("degraded", False)):
                degraded_count += 1
            judge_score = float(judge.get("score", 0.0))
            judge_scores.append(judge_score)
            reward = (0.7 * judge_score) + (0.3 * float(outcome.reward)) if grader is not None else float(outcome.reward)
            rewards.append(reward)
            parse_success_count += int(outcome.parse_success)
            task_correct_count += int(outcome.task_correct)
            issue_precision_values.append(outcome.issue_precision)
            issue_recall_values.append(outcome.issue_recall)
            issue_f1_values.append(outcome.issue_f1)
            reasoning_f1_values.append(outcome.reasoning_f1)
            extra_issue_rate_values.append(outcome.extra_issue_rate)
            empty_list_accuracy_values.append(outcome.empty_list_accuracy)
            predicted_issue_count_values.append(float(outcome.predicted_issue_count))
            verbosity_penalty_values.append(outcome.verbosity_penalty)
            if handle is not None:
                handle.write(
                    json.dumps(
                        {
                            "row_id": example.row_id,
                            "split": split_name,
                            "task_type": example.task_type,
                            "answer_text": result.answer_text,
                            "prediction": parse_outcome.payload,
                            "parse_method": parse_outcome.method,
                            "answer_parse_mode": parse_mode,
                            "local_reward": outcome.reward,
                            "judge_score": judge_score,
                            "reward": reward,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                handle.flush()
    finally:
        if handle is not None:
            handle.close()
    completed_count = len(rewards)
    total = max(1, completed_count)
    return {
        "requested_count": float(requested_count),
        "count": float(completed_count),
        "inference_error_count": float(inference_error_count),
        "inference_error_rate": (float(inference_error_count) / float(max(1, requested_count))),
        "reward_mean": fmean(rewards) if rewards else 0.0,
        "local_reward_mean": fmean(local_rewards) if local_rewards else 0.0,
        "judge_score_mean": fmean(judge_scores) if judge_scores else 0.0,
        "judge_degraded_rate": degraded_count / float(total),
        "parse_rate": parse_success_count / float(total),
        "task_correct_rate": task_correct_count / float(total),
        "issue_precision": fmean(issue_precision_values) if issue_precision_values else 0.0,
        "issue_recall": fmean(issue_recall_values) if issue_recall_values else 0.0,
        "issue_f1": fmean(issue_f1_values) if issue_f1_values else 0.0,
        "reasoning_f1": fmean(reasoning_f1_values) if reasoning_f1_values else 0.0,
        "extra_issue_rate": fmean(extra_issue_rate_values) if extra_issue_rate_values else 0.0,
        "empty_list_accuracy": fmean(empty_list_accuracy_values) if empty_list_accuracy_values else 0.0,
        "predicted_issue_count_mean": fmean(predicted_issue_count_values) if predicted_issue_count_values else 0.0,
        "verbosity_penalty_mean": fmean(verbosity_penalty_values) if verbosity_penalty_values else 0.0,
    }


def _resolve_sync_eval_model(
    *,
    base_url: str,
    tuna_api_key: str,
    finetune_id: str,
    checkpoint_step: int,
) -> Optional[str]:
    try:
        resolved_checkpoint_step, _ = resolve_checkpoint_step(
            api_base=str(base_url),
            api_key=str(tuna_api_key),
            finetune_id=str(finetune_id),
            requested_step=int(checkpoint_step),
            fallback_policy="exact",
            ready_max_wait_s=float(SYNC_EVAL_CHECKPOINT_READY_MAX_WAIT_S),
            ready_poll_interval_s=float(SYNC_EVAL_CHECKPOINT_READY_POLL_INTERVAL_S),
        )
    except Exception as exc:
        print(
            f"sync eval skipped checkpoint_step={int(checkpoint_step)} "
            f"because checkpoint readiness check failed: {type(exc).__name__}: {exc}"
        )
        return None
    return f"{common.DEFAULT_BASE_MODEL}/{str(finetune_id)}@{int(resolved_checkpoint_step)}"


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    raw_argv = list(argv) if argv is not None else list(os.sys.argv[1:])
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    pre_args, _ = pre_parser.parse_known_args(raw_argv)
    config_path = common.resolve_config_path(pre_args.config, script_dir=SCRIPT_DIR)
    config = common.load_json_config(config_path, default_path=DEFAULT_CONFIG_PATH)
    single_key_requested = "--api-key-env-var" in raw_argv or (
        "api_key_env_var" in config and "--api-key-env-vars" not in raw_argv
    )
    multi_key_requested = "--api-key-env-vars" in raw_argv or (
        "api_key_env_vars" in config and "--api-key-env-var" not in raw_argv
    )

    parser = argparse.ArgumentParser(description="Train Inspector MD query finetunes.")
    parser.add_argument("--config", default=str(config_path))
    parser.add_argument("--env-file", default=str(common.repo_relative(".env.staging")))
    parser.add_argument("--api-key", default="")
    parser.add_argument("--api-key-env-var", default=common.DEFAULT_API_KEY_ENV_VAR)
    parser.add_argument("--api-key-env-vars", nargs="+", default=list(common.DEFAULT_API_KEY_ENV_VARS))
    parser.add_argument("--base-url", default=common.DEFAULT_BASE_URL)
    parser.add_argument("--dataset-dir", default=str(common.repo_relative("outputs", "inspector_query_issues_v2")))
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="validation")
    parser.add_argument("--final-eval-splits", nargs="+", default=["validation", "test"])
    parser.add_argument("--mode", choices=("sft", "rl", "sft_then_rl", "both"), default="sft_then_rl")
    parser.add_argument("--finetune-id", default="")
    parser.add_argument("--finetune-name", default="")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sft-steps", type=int, default=80)
    parser.add_argument("--rl-steps", type=int, default=160)
    parser.add_argument("--sft-lr", type=float, default=5e-5)
    parser.add_argument("--rl-lr", type=float, default=5e-5)
    parser.add_argument("--num-rollouts", type=int, default=8)
    parser.add_argument("--rollout-stream-max-concurrency", type=int, default=4)
    parser.add_argument("--rollout-stream-buffer-size", type=int, default=8)
    parser.add_argument("--sft-temperature", type=float, default=0.0)
    parser.add_argument("--sft-top-p", type=float, default=1.0)
    parser.add_argument("--sft-max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--eval-temperature", type=float, default=0.0)
    parser.add_argument("--eval-top-p", type=float, default=1.0)
    parser.add_argument("--eval-max-tokens", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--eval-max-samples", type=int, default=64)
    parser.add_argument("--final-eval-max-samples", type=int, default=0)
    parser.add_argument("--sync-eval-subprocess-timeout-s", type=float, default=SYNC_EVAL_SUBPROCESS_TIMEOUT_S)
    parser.add_argument("--resume-step-offset", type=int, default=0)
    parser.add_argument("--multi-issue-sample-multiplier", type=float, default=2.0)
    parser.add_argument("--multi-issue-sample-max-share", type=float, default=0.6)
    parser.add_argument("--reasoning", action="store_true")
    parser.add_argument("--off-policy", action="store_true")
    parser.add_argument("--off-policy-mix-ratio", type=float, default=0.25)
    parser.add_argument("--off-policy-buffer-size", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--run-output-dir", default=str(common.repo_relative("outputs", "runs", "inspector_query")))
    parser.add_argument("--best-metric", default="eval_reward_mean")
    parser.add_argument("--async-checkpoint-eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--async-checkpoint-eval-dir",
        default=str(common.repo_relative("outputs", "async_checkpoint_eval_query")),
    )
    parser.add_argument("--async-checkpoint-eval-max-inflight", type=int, default=1)
    parser.add_argument("--async-checkpoint-eval-drain-on-exit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-project", default="moondream-inspector-query-rl")
    parser.add_argument("--wandb-run-name", default="")
    parser.add_argument("--grader-api-key", default="")
    parser.add_argument("--grader-api-key-env-var", default=openrouter_grader.DEFAULT_OPENROUTER_ENV_VAR)
    parser.add_argument("--grader-api-base", default=openrouter_grader.DEFAULT_OPENROUTER_API_BASE)
    parser.add_argument("--grader-model-id", default=openrouter_grader.DEFAULT_GRADER_MODEL)
    parser.add_argument("--grader-profile", default=openrouter_grader.DEFAULT_GRADER_PROFILE)
    parser.add_argument("--grader-rubric-version", default=openrouter_grader.DEFAULT_GRADER_RUBRIC_VERSION)
    parser.add_argument("--grader-timeout", type=float, default=60.0)
    parser.add_argument("--require-query-text-refresh", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--client-max-retries", type=int, default=1)
    parser.add_argument("--client-backoff-base-s", type=float, default=1.0)
    parser.add_argument("--client-backoff-max-s", type=float, default=5.0)
    parser.add_argument("--train-step-max-retries", type=int, default=3)
    parser.add_argument("--train-step-retry-backoff-base-s", type=float, default=10.0)
    parser.add_argument("--train-step-retry-backoff-max-s", type=float, default=60.0)
    parser.add_argument("--max-consecutive-train-step-failures", type=int, default=4)
    parser.add_argument("--transient-train-step-failure-cooldown-s", type=float, default=30.0)
    parser.add_argument("--post-create-warmup-s", type=float, default=10.0)

    option_to_dest: dict[str, str] = {}
    for action in parser._actions:
        if not action.option_strings:
            continue
        for opt in action.option_strings:
            option_to_dest[opt] = action.dest
    overridden = {option_to_dest[arg] for arg in raw_argv if arg in option_to_dest}
    config_cli_args = common.config_to_cli_args(parser, config, config_path=config_path, overridden_dests=overridden)
    args = parser.parse_args(config_cli_args + raw_argv)
    args.config = str(common.resolve_config_path(args.config, script_dir=SCRIPT_DIR))
    args.dataset_dir = common.resolve_path(args.dataset_dir, module_root=SCRIPT_DIR)
    args.env_file = str(common.resolve_path(args.env_file, module_root=SCRIPT_DIR))
    args.run_output_dir = common.resolve_path(args.run_output_dir, module_root=SCRIPT_DIR)
    args.async_checkpoint_eval_dir = common.resolve_path(args.async_checkpoint_eval_dir, module_root=SCRIPT_DIR)
    if single_key_requested and not multi_key_requested:
        args.api_key_env_vars = [args.api_key_env_var]
    else:
        args.api_key_env_vars = common.normalize_api_key_env_vars(args.api_key_env_vars)
    args.mode = _normalize_query_mode(args.mode)
    if int(args.batch_size) not in {4, 8}:
        raise ValueError("--batch-size must be 4 or 8 for the approved rollout policy")
    if int(args.num_rollouts) != 8:
        raise ValueError("--num-rollouts must be 8 for the approved rollout policy")
    if bool(args.reasoning) and bool(args.off_policy):
        raise ValueError("Reasoning and off-policy cannot be enabled in the same run.")
    if int(args.rollout_stream_max_concurrency) != 4:
        raise ValueError("--rollout-stream-max-concurrency must be 4")
    if int(args.rollout_stream_buffer_size) != 8:
        raise ValueError("--rollout-stream-buffer-size must be 8")
    if int(args.async_checkpoint_eval_max_inflight) <= 0:
        raise ValueError("--async-checkpoint-eval-max-inflight must be >= 1")
    if int(args.final_eval_max_samples) < 0:
        raise ValueError("--final-eval-max-samples must be >= 0")
    if float(args.sync_eval_subprocess_timeout_s) <= 0:
        raise ValueError("--sync-eval-subprocess-timeout-s must be > 0")
    if int(args.resume_step_offset) < 0:
        raise ValueError("--resume-step-offset must be >= 0")
    if float(args.multi_issue_sample_multiplier) < 1.0:
        raise ValueError("--multi-issue-sample-multiplier must be >= 1.0")
    if not 0.0 < float(args.multi_issue_sample_max_share) <= 1.0:
        raise ValueError("--multi-issue-sample-max-share must be in (0, 1]")
    if int(args.client_max_retries) < 0:
        raise ValueError("--client-max-retries must be >= 0")
    if float(args.client_backoff_base_s) <= 0:
        raise ValueError("--client-backoff-base-s must be > 0")
    if float(args.client_backoff_max_s) < float(args.client_backoff_base_s):
        raise ValueError("--client-backoff-max-s must be >= --client-backoff-base-s")
    if int(args.train_step_max_retries) < 0:
        raise ValueError("--train-step-max-retries must be >= 0")
    if float(args.train_step_retry_backoff_base_s) <= 0:
        raise ValueError("--train-step-retry-backoff-base-s must be > 0")
    if float(args.train_step_retry_backoff_max_s) < float(args.train_step_retry_backoff_base_s):
        raise ValueError("--train-step-retry-backoff-max-s must be >= --train-step-retry-backoff-base-s")
    if int(args.max_consecutive_train_step_failures) < 0:
        raise ValueError("--max-consecutive-train-step-failures must be >= 0")
    if float(args.transient_train_step_failure_cooldown_s) < 0:
        raise ValueError("--transient-train-step-failure-cooldown-s must be >= 0")
    if float(args.post_create_warmup_s) < 0:
        raise ValueError("--post-create-warmup-s must be >= 0")
    if not str(args.finetune_name or "").strip():
        args.finetune_name = f"inspector-query-{_random_suffix()}"
    return args


def _normalize_query_mode(mode: str) -> str:
    value = str(mode or "").strip().lower()
    return "sft_then_rl" if value == "both" else value


def _query_mode_status_line(*, mode: str, sft_steps: int, rl_steps: int) -> str:
    normalized = _normalize_query_mode(mode)
    prefix = (
        f"query training plan: mode={normalized} "
        f"sft_steps={int(sft_steps)} rl_steps={int(rl_steps)}"
    )
    if normalized == "sft_then_rl":
        return prefix + " | RL starts automatically after SFT."
    if normalized == "sft" and int(rl_steps) > 0:
        return prefix + " | RL is configured but will not run in mode=sft; use --mode sft_then_rl or --mode both to chain phases."
    return prefix


def _format_query_sft_step_log(
    *,
    step: int,
    batch_size: int,
    sft_loss: float,
    kl: float,
    successful_microbatches: int,
    failed_microbatches: int,
) -> str:
    return (
        f"step {int(step)} sft loss={float(sft_loss):.4f} kl={float(kl):.4f} "
        f"batch={int(batch_size)} microbatches={int(successful_microbatches)} "
        f"failed_microbatches={int(failed_microbatches)}"
    )


def _format_query_rl_step_log(
    *,
    step: int,
    batch_size: int,
    reward_mean: float,
    local_reward_mean: float,
    judge_score_mean: float,
    kl: float,
    off_policy_injected: int,
    group_count: int,
) -> str:
    return (
        f"step {int(step)} rl reward={float(reward_mean):.4f} "
        f"local_reward={float(local_reward_mean):.4f} judge={float(judge_score_mean):.4f} "
        f"kl={float(kl):.4f} batch={int(batch_size)} "
        f"offp={int(off_policy_injected)}/{int(group_count)}"
    )


def _sample_batch(
    examples: list[QueryExample],
    *,
    batch_size: int,
    rng: random.Random,
    multi_issue_multiplier: float,
    multi_issue_max_share: float,
) -> list[QueryExample]:
    batch_size = max(1, int(batch_size))
    multi_issue = [item for item in examples if bool(item.is_multi_issue)]
    single_issue = [item for item in examples if not bool(item.is_multi_issue)]
    if not multi_issue or float(multi_issue_multiplier) <= 1.0:
        return [rng.choice(examples) for _ in range(batch_size)]
    if not single_issue:
        return [rng.choice(multi_issue) for _ in range(batch_size)]
    weighted_multi_mass = float(multi_issue_multiplier) * float(len(multi_issue))
    total_mass = weighted_multi_mass + float(len(single_issue))
    multi_count = int(round(batch_size * (weighted_multi_mass / float(total_mass)))) if total_mass > 0.0 else 0
    multi_count = min(multi_count, int(round(batch_size * float(multi_issue_max_share))))
    multi_count = max(0, min(batch_size, multi_count))
    single_count = max(0, batch_size - multi_count)
    batch = [rng.choice(multi_issue) for _ in range(multi_count)] + [rng.choice(single_issue) for _ in range(single_count)]
    rng.shuffle(batch)
    return batch


def _record_eval(
    *,
    path: Path,
    step: int,
    stage: str,
    split_name: str,
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "step": int(step),
                    "stage": stage,
                    "split": split_name,
                    "metrics": metrics,
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def _selection_metric_value(metrics: dict[str, Any], best_metric_name: str) -> float:
    metric_key = str(best_metric_name or "eval_reward_mean").removeprefix("eval_")
    try:
        return float(metrics.get(metric_key, metrics.get("reward_mean", 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _build_async_query_eval_command(
    *,
    args: argparse.Namespace,
    finetune_id: str,
    checkpoint_step: int,
    metrics_json_path: Path,
    predictions_jsonl_path: Path,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "inspector_md.benchmark_inspector_query",
        "--env-file",
        str(args.env_file),
        "--base-url",
        str(args.base_url),
        "--dataset-dir",
        str(args.dataset_dir),
        "--split",
        str(args.val_split),
        "--finetune-id",
        str(finetune_id),
        "--checkpoint-step",
        str(int(checkpoint_step)),
        "--checkpoint-fallback-policy",
        "exact",
        "--checkpoint-ready-max-wait-s",
        "300",
        "--checkpoint-ready-poll-interval-s",
        "5",
        "--temperature",
        str(float(args.eval_temperature)),
        "--top-p",
        str(float(args.eval_top_p)),
        "--max-tokens",
        str(int(args.eval_max_tokens)),
        "--timeout",
        str(float(args.timeout)),
        "--client-max-retries",
        str(int(args.client_max_retries)),
        "--client-backoff-base-s",
        str(float(args.client_backoff_base_s)),
        "--client-backoff-max-s",
        str(float(args.client_backoff_max_s)),
        "--output-json",
        str(metrics_json_path),
        "--predictions-jsonl",
        str(predictions_jsonl_path),
        "--grader-api-base",
        str(args.grader_api_base),
        "--grader-model-id",
        str(args.grader_model_id),
        "--grader-profile",
        str(args.grader_profile),
        "--grader-rubric-version",
        str(args.grader_rubric_version),
        "--grader-timeout",
        str(float(args.grader_timeout)),
    ]
    if str(args.api_key or "").strip():
        command.extend(["--api-key", str(args.api_key)])
    else:
        command.extend(["--api-key-env-vars", *list(args.api_key_env_vars or [])])
    if str(args.grader_api_key or "").strip():
        command.extend(["--grader-api-key", str(args.grader_api_key)])
    else:
        command.extend(["--grader-api-key-env-var", str(args.grader_api_key_env_var)])
    if int(args.eval_max_samples) > 0:
        command.extend(["--max-samples", str(int(args.eval_max_samples))])
    if bool(args.reasoning):
        command.append("--reasoning")
    return command


def _build_sync_query_eval_command(
    *,
    args: argparse.Namespace,
    finetune_id: str,
    checkpoint_step: int,
    split_name: str,
    max_samples: int,
    metrics_json_path: Path,
    predictions_jsonl_path: Path,
    judge_cache_jsonl_path: Path,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "inspector_md.benchmark_inspector_query",
        "--env-file",
        str(args.env_file),
        "--base-url",
        str(args.base_url),
        "--dataset-dir",
        str(args.dataset_dir),
        "--split",
        str(split_name),
        "--finetune-id",
        str(finetune_id),
        "--checkpoint-step",
        str(int(checkpoint_step)),
        "--checkpoint-fallback-policy",
        "exact",
        "--checkpoint-ready-max-wait-s",
        str(float(SYNC_EVAL_CHECKPOINT_READY_MAX_WAIT_S)),
        "--checkpoint-ready-poll-interval-s",
        str(float(SYNC_EVAL_CHECKPOINT_READY_POLL_INTERVAL_S)),
        "--temperature",
        str(float(args.eval_temperature)),
        "--top-p",
        str(float(args.eval_top_p)),
        "--max-tokens",
        str(int(args.eval_max_tokens)),
        "--timeout",
        str(float(args.timeout)),
        "--client-max-retries",
        str(int(args.client_max_retries)),
        "--client-backoff-base-s",
        str(float(args.client_backoff_base_s)),
        "--client-backoff-max-s",
        str(float(args.client_backoff_max_s)),
        "--grader-api-base",
        str(args.grader_api_base),
        "--grader-model-id",
        str(args.grader_model_id),
        "--grader-profile",
        str(args.grader_profile),
        "--grader-rubric-version",
        str(args.grader_rubric_version),
        "--grader-timeout",
        str(float(args.grader_timeout)),
        "--output-json",
        str(metrics_json_path),
        "--predictions-jsonl",
        str(predictions_jsonl_path),
        "--judge-cache-jsonl",
        str(judge_cache_jsonl_path),
        "--reuse-judge-cache",
    ]
    if str(args.api_key or "").strip():
        command.extend(["--api-key", str(args.api_key)])
    else:
        command.extend(["--api-key-env-vars", *list(args.api_key_env_vars or [])])
    if str(args.grader_api_key or "").strip():
        command.extend(["--grader-api-key", str(args.grader_api_key)])
    else:
        command.extend(["--grader-api-key-env-var", str(args.grader_api_key_env_var)])
    if int(max_samples) > 0:
        command.extend(["--max-samples", str(int(max_samples))])
    if bool(args.reasoning):
        command.append("--reasoning")
    return command


def _ingest_async_query_eval_results(
    *,
    results: list[Any],
    eval_history_path: Path,
    log_step: int,
    best_metric_name: str,
    best_metric_value: float,
) -> tuple[float, int]:
    success_count = 0
    for result in results:
        source_step = int(result.metadata.get("step_for_log", result.checkpoint_step))
        stage = str(result.metadata.get("stage") or "async")
        split_name = str(result.metadata.get("split_name") or "validation")
        if result.status != "succeeded" or result.metrics_payload is None:
            print(
                f"async query eval failed step={source_step} "
                f"checkpoint_step={result.checkpoint_step} log={result.stdout_log_path}"
            )
            continue
        metrics = dict(result.metrics_payload)
        _record_eval(path=eval_history_path, step=source_step, stage=f"async_{stage}", split_name=split_name, metrics=metrics)
        _wandb_log(
            {
                **metrics,
                "source_step": int(source_step),
                "checkpoint_step": int(result.checkpoint_step),
                "split": split_name,
                "stage": stage,
            },
            step=int(log_step),
            namespace="async_eval",
        )
        best_metric_value = max(best_metric_value, _selection_metric_value(metrics, best_metric_name))
        success_count += 1
    return best_metric_value, success_count


def _load_metadata(dataset_dir: Path) -> dict[str, Any]:
    metadata_path = dataset_dir / "metadata.json"
    if not metadata_path.exists():
        return {}
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _require_query_dataset_ready(dataset_dir: Path, *, require_query_text_refresh: bool) -> dict[str, Any]:
    metadata = _load_metadata(dataset_dir)
    if not metadata:
        raise ValueError(f"Query dataset metadata not found at {dataset_dir / 'metadata.json'}")
    refresh_mode = str(metadata.get("query_text_refresh_mode") or "").strip().lower()
    if bool(require_query_text_refresh) and refresh_mode != "openrouter":
        raise ValueError(
            f"Query finetuning requires refreshed query text. dataset_dir={dataset_dir} query_text_refresh_mode={refresh_mode!r}"
        )
    if str(metadata.get("query_target_format") or "").strip() != query_compact.TARGET_FORMAT_JSON_ISSUE_LIST:
        raise ValueError(
            f"Query finetuning requires target_format={query_compact.TARGET_FORMAT_JSON_ISSUE_LIST!r}. "
            f"dataset_dir={dataset_dir}"
        )
    return metadata


def _load_judge_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    cache: dict[str, dict[str, Any]] = {}
    for row in common.load_jsonl(path):
        key = str(row.get("cache_key") or "").strip()
        if key:
            cache[key] = dict(row)
    return cache


def _append_judge_cache_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _expected_finding_for_example(example: QueryExample) -> task_schema.Finding:
    payload = json.loads(example.final_answer_json)
    finding_payload = payload.get("finding") if isinstance(payload, dict) else payload
    if not isinstance(finding_payload, dict):
        raise ValueError(f"row={example.row_id} invalid final_answer_json finding payload")
    issue_code = ontology.normalize_issue_code(finding_payload.get("issue_code"))
    return task_schema.Finding.from_payload(
        {
            "finding_id": f"{example.row_id}_expected",
            "issue_code": issue_code,
            "title": finding_payload.get("title") or ontology.get_issue(issue_code).title,
            "box": (example.spatial_refs or [[0.0, 0.0, 1.0, 1.0]])[0],
            "evidence": list(finding_payload.get("evidence") or []),
            "severity": finding_payload.get("severity") or "unknown",
            "recommended_action": finding_payload.get("recommended_action") or ontology.get_issue(issue_code).default_recommended_action,
            "cost_band": ontology.get_issue(issue_code).default_cost_band,
            "possible_compliance_issue": False,
            "insufficient_evidence": finding_payload.get("insufficient_evidence", False),
            "compliance_note": "",
            "source_detect_labels": ontology.detect_labels_for_issue(issue_code)[:1],
            "spatial_ref_index": 0,
        }
    )


def _predicted_finding_for_example(example: QueryExample, payload: dict[str, Any]) -> task_schema.Finding:
    finding_payload = payload.get("finding") if isinstance(payload.get("finding"), dict) else payload
    if not isinstance(finding_payload, dict):
        expected = _expected_finding_for_example(example)
        return task_schema.Finding.from_payload(
            {
                **expected.to_payload(),
                "finding_id": f"{example.row_id}_predicted",
                "title": "",
                "evidence": [],
                "severity": "unknown",
                "recommended_action": "",
                "insufficient_evidence": True,
            }
        )
    expected_issue_code = _expected_finding_for_example(example).issue_code
    raw_issue_code = str(finding_payload.get("issue_code") or "").strip()
    try:
        issue_code = ontology.normalize_issue_code(raw_issue_code or expected_issue_code)
    except ValueError:
        issue_code = expected_issue_code
    issue = ontology.get_issue(issue_code)
    return task_schema.Finding.from_payload(
        {
            "finding_id": f"{example.row_id}_predicted",
            "issue_code": issue_code,
            "title": finding_payload.get("title") or issue.title,
            "box": (example.spatial_refs or [[0.0, 0.0, 1.0, 1.0]])[0],
            "evidence": list(finding_payload.get("evidence") or []),
            "severity": finding_payload.get("severity") or "unknown",
            "recommended_action": finding_payload.get("recommended_action") or issue.default_recommended_action,
            "cost_band": issue.default_cost_band,
            "possible_compliance_issue": False,
            "insufficient_evidence": finding_payload.get("insufficient_evidence", False),
            "compliance_note": "",
            "source_detect_labels": ontology.detect_labels_for_issue(issue_code)[:1],
            "spatial_ref_index": 0,
        }
    )


def _judge_answer_with_cache(
    *,
    grader: Optional[openrouter_grader.OpenRouterGrader],
    cache: dict[str, dict[str, Any]],
    cache_path: Path,
    example: QueryExample,
    answer_text: str,
    parse_outcome: QueryParseOutcome,
) -> dict[str, Any]:
    if grader is None:
        return {"score": 0.0, "degraded": True, "cache_hit": False, "reason": "grader_unavailable"}
    cache_key = openrouter_grader.cache_key_for_judgement(
        task_type=example.task_type,
        prompt_text=example.question,
        answer_text=answer_text,
        canonical_gt=json.loads(example.final_answer_json),
        model_id=grader.model_id,
        rubric_version=grader.rubric_version,
        profile=grader.profile,
    )
    cached = cache.get(cache_key)
    if cached is not None:
        judgement = dict(cached.get("judgement") or {})
        judgement["cache_hit"] = True
        return judgement
    judgement: dict[str, Any]
    degraded = False
    for attempt in range(2):
        try:
            if example.task_type in {"proposal", "issues"}:
                judgement = grader.grade_issues(
                    inspection_request=example.inspection_request,
                    asset_context=example.asset_context,
                    expected_issues=task_schema.normalize_issue_proposals(json.loads(example.final_answer_json)),
                    answer_text=answer_text,
                )
            else:
                judgement = grader.grade_finding(
                    inspection_request=example.inspection_request,
                    asset_context=example.asset_context,
                    predicted_finding=_predicted_finding_for_example(example, parse_outcome.payload),
                    expected_finding=_expected_finding_for_example(example),
                    answer_text=answer_text,
                )
            break
        except (TimeoutError, openrouter_grader.OpenRouterGradingError):
            judgement = {}
            degraded = True
            if attempt >= 1:
                break
    if not judgement:
        judgement = {"score": 0.0, "degraded": True, "cache_hit": False, "reason": "judge_fallback_to_local"}
    else:
        judgement = dict(judgement)
        judgement["degraded"] = degraded
        judgement["cache_hit"] = False
    cache_row = {"cache_key": cache_key, "judgement": judgement}
    cache[cache_key] = cache_row
    _append_judge_cache_row(cache_path, cache_row)
    return judgement


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    common.maybe_load_env_file(args.env_file, override=False)
    if args.mode in {"sft", "sft_then_rl"} and bool(args.reasoning):
        raise ValueError(
            "Query SFT with --reasoning is currently unsupported by the staging Tuna backend. "
            "Direct probes show /tuning/train_step returns HTTP 500 when query SFT requests set "
            "request.reasoning=true or include target.reasoning. Use --mode rl --sft-steps 0 for "
            "reasoning runs, or disable reasoning for SFT."
        )
    dataset_metadata = _require_query_dataset_ready(
        Path(args.dataset_dir),
        require_query_text_refresh=bool(args.require_query_text_refresh),
    )
    api_key_pool = common.resolve_api_key_pool(explicit_api_key=args.api_key, api_key_env_vars=args.api_key_env_vars)
    tuna_api_key = api_key_pool.slots[0].api_key
    train_examples = _load_split_examples(split_name=args.train_split, dataset_dir=Path(args.dataset_dir))
    val_examples = _load_split_examples(split_name=args.val_split, dataset_dir=Path(args.dataset_dir))
    grader_api_key = openrouter_grader.resolve_openrouter_api_key(
        explicit_api_key=args.grader_api_key,
        api_key_env_var=args.grader_api_key_env_var,
    )
    grader = openrouter_grader.OpenRouterGrader(
        api_key=grader_api_key,
        model_id=args.grader_model_id,
        api_base=args.grader_api_base,
        timeout=float(args.grader_timeout),
        profile=args.grader_profile,
        rubric_version=args.grader_rubric_version,
    )
    print(
        f"preflight base_url={args.base_url} key_slots={api_key_pool.env_var_names or ['<explicit>']} "
        f"grader_model={grader.model_id} refresh_mode={dataset_metadata.get('query_text_refresh_mode', '')}"
    )
    tuna_client = TunaClient(
        api_key=tuna_api_key,
        base_url=args.base_url,
        timeout=args.timeout,
        retry=RetryConfig(
            max_retries=int(args.client_max_retries),
            backoff_base=float(args.client_backoff_base_s),
            backoff_max=float(args.client_backoff_max_s),
        ),
    )
    inference_client = MoondreamInspectorClient(api_key_pool=api_key_pool, base_url=args.base_url, timeout=args.timeout)
    if str(args.finetune_id or "").strip():
        finetune = tuna_client.get_finetune(str(args.finetune_id).strip())
    else:
        finetune = tuna_client.create_finetune(name=args.finetune_name, rank=int(args.rank))
    print(f"resolved_finetune_id={finetune.finetune_id}")
    if float(args.post_create_warmup_s) > 0:
        warmup_s = float(args.post_create_warmup_s)
        print(f"warming query finetune for {warmup_s:.1f}s before first train_step")
        time.sleep(warmup_s)

    run_config_payload = _json_safe(
        {
            **vars(args),
            "dataset_dir": str(args.dataset_dir),
            "run_output_dir": str(args.run_output_dir),
            "resolved_finetune_id": finetune.finetune_id,
            "api_key_env_vars": list(api_key_pool.env_var_names),
            "api_key_slot_count": len(api_key_pool.slots),
            "dataset_metadata": dataset_metadata,
        }
    )
    run = wandb.init(
        project=str(args.wandb_project),
        name=str(args.wandb_run_name or "").strip() or None,
        config=run_config_payload,
    )
    run_dir = Path(args.run_output_dir) / finetune.finetune_id
    run_dir.mkdir(parents=True, exist_ok=True)
    common.write_json(
        run_dir / "run_config.json",
        {
            **run_config_payload,
            "base_url": str(args.base_url),
        },
    )
    eval_history_path = run_dir / "eval_history.jsonl"
    judge_cache_path = run_dir / "judge_cache.jsonl"
    judge_cache = _load_judge_cache(judge_cache_path)
    rng = random.Random(args.seed)
    off_policy_buffer: deque[TrainStepGroup] = deque(maxlen=max(1, int(args.off_policy_buffer_size)))
    best_metric_value = float("-inf")
    best_metric_name = str(args.best_metric)
    total_train_steps = int(args.sft_steps) + int(args.rl_steps)
    resume_step_offset = int(args.resume_step_offset)
    async_eval_jobs: list[Any] = []
    async_eval_success_count = 0
    latest_checkpoint_step: Optional[int] = None
    sft_consecutive_failures = 0
    rl_consecutive_failures = 0
    print(
        _query_mode_status_line(
            mode=str(args.mode),
            sft_steps=int(args.sft_steps),
            rl_steps=int(args.rl_steps),
        )
    )

    def save_checkpoint(*, context: str) -> Optional[int]:
        nonlocal latest_checkpoint_step
        try:
            saved = finetune.save_checkpoint()
        except Exception as exc:
            print(f"{context}: checkpoint save failed; continuing. details={type(exc).__name__}: {exc}")
            return None
        checkpoint_step = getattr(getattr(saved, "checkpoint", None), "step", None)
        try:
            if checkpoint_step is None:
                return None
            latest_checkpoint_step = int(checkpoint_step)
            run.summary["latest_checkpoint_step"] = int(latest_checkpoint_step)
            return int(latest_checkpoint_step)
        except (TypeError, ValueError):
            return None

    def run_sync_eval(
        *,
        step: int,
        stage: str,
        split_name: str,
        max_samples: int,
        checkpoint_step: Optional[int] = None,
        checkpoint_context: str,
        namespace: str,
    ) -> Optional[dict[str, float]]:
        eval_checkpoint_step = checkpoint_step
        if eval_checkpoint_step is None:
            eval_checkpoint_step = save_checkpoint(context=checkpoint_context)
        if eval_checkpoint_step is None:
            print(f"{stage} eval skipped step={step}: checkpoint save unavailable")
            return None
        metrics_json_path = run_dir / "eval_metrics" / f"{stage}_step_{step:04d}_{split_name}.metrics.json"
        predictions_path = run_dir / "eval_predictions" / f"{stage}_step_{step:04d}_{split_name}.jsonl"
        stdout_log_path = run_dir / "eval_logs" / f"{stage}_step_{step:04d}_{split_name}.stdout.log"
        metrics_json_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_log_path.parent.mkdir(parents=True, exist_ok=True)
        command = _build_sync_query_eval_command(
            args=args,
            finetune_id=str(finetune.finetune_id),
            checkpoint_step=int(eval_checkpoint_step),
            split_name=str(split_name),
            max_samples=int(max_samples),
            metrics_json_path=metrics_json_path,
            predictions_jsonl_path=predictions_path,
            judge_cache_jsonl_path=judge_cache_path,
        )
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=float(args.sync_eval_subprocess_timeout_s),
                cwd=str(REPO_ROOT),
                env=dict(os.environ),
            )
        except subprocess.TimeoutExpired as exc:
            stdout_log_path.write_text(
                (str(exc.stdout or "") + ("\n" if exc.stdout else ""))
                + (str(exc.stderr or "") if exc.stderr else ""),
                encoding="utf-8",
            )
            print(
                f"{stage} eval timed out step={step} split={split_name} "
                f"checkpoint_step={int(eval_checkpoint_step)} timeout_s={float(args.sync_eval_subprocess_timeout_s):.1f}"
            )
            _wandb_log(
                {
                    "split": str(split_name),
                    "stage": str(stage),
                    "checkpoint_step": int(eval_checkpoint_step),
                    "timed_out": 1.0,
                },
                step=int(step),
                namespace=f"{namespace}_timeout",
            )
            return None
        stdout_log_path.write_text(
            str(completed.stdout or "") + ("\n" if completed.stdout and completed.stderr else "") + str(completed.stderr or ""),
            encoding="utf-8",
        )
        if completed.returncode != 0:
            print(
                f"{stage} eval failed step={step} split={split_name} "
                f"checkpoint_step={int(eval_checkpoint_step)} returncode={int(completed.returncode)} "
                f"log={stdout_log_path}"
            )
            _wandb_log(
                {
                    "split": str(split_name),
                    "stage": str(stage),
                    "checkpoint_step": int(eval_checkpoint_step),
                    "returncode": float(completed.returncode),
                },
                step=int(step),
                namespace=f"{namespace}_failure",
            )
            return None
        if not metrics_json_path.exists():
            print(
                f"{stage} eval failed step={step} split={split_name}: "
                f"missing metrics output at {metrics_json_path}"
            )
            return None
        payload = json.loads(metrics_json_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            print(f"{stage} eval failed step={step} split={split_name}: metrics payload was not a JSON object")
            return None
        return dict(payload)

    def evaluate(step: int, stage: str) -> bool:
        nonlocal best_metric_value
        split_metrics = run_sync_eval(
            step=int(step),
            stage=str(stage),
            split_name=str(args.val_split),
            max_samples=int(args.eval_max_samples),
            checkpoint_context=f"{stage} eval step={step}",
            namespace="eval",
        )
        if split_metrics is None:
            return False
        _record_eval(path=eval_history_path, step=step, stage=stage, split_name=args.val_split, metrics=split_metrics)
        _wandb_log(
            {
                **split_metrics,
                "split": str(args.val_split),
                "stage": str(stage),
            },
            step=int(step),
            namespace="eval",
        )
        metric_value = _selection_metric_value(split_metrics, best_metric_name)
        best_metric_value = max(best_metric_value, metric_value)
        return True

    if args.mode in {"sft", "sft_then_rl"}:
        if int(args.sft_steps) > 0:
            print(
                f"starting query sft phase at step {resume_step_offset + 1} "
                f"(local_steps={int(args.sft_steps)})"
            )
        sft_progress = _make_progress_bar(total=int(args.sft_steps), desc="query sft")
        for step in range(1, int(args.sft_steps) + 1):
            log_step = resume_step_offset + int(step)
            if bool(args.async_checkpoint_eval):
                async_eval_jobs, completed_async_results = poll_checkpoint_eval_jobs(async_eval_jobs)
                best_metric_value, completed_successes = _ingest_async_query_eval_results(
                    results=completed_async_results,
                    eval_history_path=eval_history_path,
                    log_step=log_step,
                    best_metric_name=best_metric_name,
                    best_metric_value=best_metric_value,
                )
                async_eval_success_count += int(completed_successes)
            batch = _sample_batch(
                train_examples,
                batch_size=args.batch_size,
                rng=rng,
                multi_issue_multiplier=float(args.multi_issue_sample_multiplier),
                multi_issue_max_share=float(args.multi_issue_sample_max_share),
            )
            groups = [
                TrainStepGroup.from_sft(
                    request=_build_request(
                        example,
                        reasoning=bool(args.reasoning),
                        temperature=float(args.sft_temperature),
                        top_p=float(args.sft_top_p),
                        max_tokens=int(args.sft_max_tokens),
                    ),
                    targets=[_sft_target(example, include_reasoning=bool(args.reasoning))],
                )
                for example in batch
            ]
            try:
                train_out = _train_query_sft_groups(
                    finetune=finetune,
                    groups=groups,
                    lr=float(args.sft_lr),
                    train_step_max_retries=int(args.train_step_max_retries),
                    train_step_retry_backoff_base_s=float(args.train_step_retry_backoff_base_s),
                    train_step_retry_backoff_max_s=float(args.train_step_retry_backoff_max_s),
                    step_label=f"query sft step {step}",
                )
            except (TunaAPIError, TunaNetworkError) as exc:
                print(f"query sft train_step failed at step {step}: {_format_tuna_error(exc)}")
                if _is_transient_query_train_step_error(exc) and sft_consecutive_failures < int(args.max_consecutive_train_step_failures):
                    sft_consecutive_failures += 1
                    _wandb_log(
                        {
                            "stage": "sft",
                            "train_step_failed": 1.0,
                            "train_step_applied": 0.0,
                            "consecutive_train_step_failures": float(sft_consecutive_failures),
                        },
                        step=log_step,
                        namespace="train",
                    )
                    cooldown_s = float(args.transient_train_step_failure_cooldown_s)
                    if cooldown_s > 0:
                        print(
                            f"query sft skipping step {step} after transient backend failure; "
                            f"cooldown {cooldown_s:.1f}s before continuing"
                        )
                        time.sleep(cooldown_s)
                    sft_progress.set_postfix({"step": step, "status": "retry_skip"}, refresh=False)
                    sft_progress.update(1)
                    continue
                raise
            if not bool(getattr(train_out, "applied", False)):
                sft_consecutive_failures += 1
                last_failure = str(getattr(train_out, "last_failure", "") or "transient_query_failure")
                print(f"query sft step {step} produced no successful microbatches; details={last_failure}")
                _wandb_log(
                    {
                        "stage": "sft",
                        "train_step_failed": 1.0,
                        "train_step_applied": 0.0,
                        "consecutive_train_step_failures": float(sft_consecutive_failures),
                        "failed_microbatch_count": float(getattr(train_out, "failed_microbatch_count", len(groups))),
                        "successful_microbatch_count": 0.0,
                    },
                    step=log_step,
                    namespace="train",
                )
                cooldown_s = float(args.transient_train_step_failure_cooldown_s)
                if cooldown_s > 0:
                    print(
                        f"query sft skipping step {step} after zero successful microbatches; "
                        f"cooldown {cooldown_s:.1f}s before continuing"
                    )
                    time.sleep(cooldown_s)
                sft_progress.set_postfix({"step": step, "status": "retry_skip"}, refresh=False)
                sft_progress.update(1)
                continue
            sft_consecutive_failures = 0
            _wandb_log(
                {
                    "stage": "sft",
                    "train_step_failed": 0.0,
                    "train_step_applied": 1.0,
                    "consecutive_train_step_failures": 0.0,
                    "group_count": len(groups),
                    "microbatch_count": int(getattr(train_out, "microbatch_count", len(groups)) or len(groups)),
                    "successful_microbatch_count": int(getattr(train_out, "successful_microbatch_count", len(groups)) or 0),
                    "failed_microbatch_count": int(getattr(train_out, "failed_microbatch_count", 0) or 0),
                    "batch_size": len(batch),
                    "lr": float(args.sft_lr),
                    "sft_loss": float(getattr(train_out, "sft_loss", 0.0) or 0.0),
                    "kl": float(getattr(train_out, "kl", 0.0) or 0.0),
                    "router_kl": float(getattr(train_out, "router_kl", 0.0) or 0.0),
                    "grad_norm": float(getattr(train_out, "grad_norm", 0.0) or 0.0),
                },
                step=log_step,
                namespace="train",
            )
            print(
                _format_query_sft_step_log(
                    step=log_step,
                    batch_size=len(batch),
                    sft_loss=float(getattr(train_out, "sft_loss", 0.0) or 0.0),
                    kl=float(getattr(train_out, "kl", 0.0) or 0.0),
                    successful_microbatches=int(getattr(train_out, "successful_microbatch_count", len(groups)) or 0),
                    failed_microbatches=int(getattr(train_out, "failed_microbatch_count", 0) or 0),
                ),
                flush=True,
            )
            checkpoint_saved_for_eval = False
            if int(args.eval_every) > 0 and log_step % int(args.eval_every) == 0:
                if bool(args.async_checkpoint_eval):
                    checkpoint_step = save_checkpoint(context=f"sft async eval step={log_step}")
                    if checkpoint_step is not None:
                        checkpoint_saved_for_eval = True
                        job = dispatch_checkpoint_eval(
                            trainer="inspector_query",
                            finetune_id=str(finetune.finetune_id),
                            checkpoint_step=int(checkpoint_step),
                            selection_metric=str(best_metric_name),
                            base_dir=str(args.async_checkpoint_eval_dir),
                            command_builder=lambda metrics_json_path, predictions_jsonl_path, _stdout_log_path: _build_async_query_eval_command(
                                args=args,
                                finetune_id=str(finetune.finetune_id),
                                checkpoint_step=int(checkpoint_step),
                                metrics_json_path=metrics_json_path,
                                predictions_jsonl_path=predictions_jsonl_path,
                            ),
                            metadata={
                                "step_for_log": log_step,
                                "stage": "sft",
                                "split_name": str(args.val_split),
                            },
                            max_inflight=int(args.async_checkpoint_eval_max_inflight),
                            inflight_jobs=async_eval_jobs,
                        )
                        if job is not None:
                            async_eval_jobs.append(job)
                    else:
                        print(f"async query eval skipped step={step}: checkpoint save unavailable")
                else:
                    checkpoint_saved_for_eval = bool(evaluate(log_step, "sft"))
            if int(args.save_every) > 0 and log_step % int(args.save_every) == 0:
                if not checkpoint_saved_for_eval:
                    save_checkpoint(context=f"sft save_every step={log_step}")
            sft_progress.set_postfix(
                {
                    "step": step,
                    "loss": f"{float(getattr(train_out, 'sft_loss', 0.0) or 0.0):.3f}",
                    "kl": f"{float(getattr(train_out, 'kl', 0.0) or 0.0):.3f}",
                },
                refresh=False,
            )
            sft_progress.update(1)
        sft_progress.close()
        if args.mode == "sft_then_rl" and int(args.rl_steps) > 0:
            print(
                f"query sft phase complete at step {resume_step_offset + int(args.sft_steps)}; "
                f"starting rl phase at step {resume_step_offset + int(args.sft_steps) + 1}"
            )
        elif args.mode == "sft":
            print(f"query sft phase complete at step {resume_step_offset + int(args.sft_steps)}")

    if args.mode in {"rl", "sft_then_rl"}:
        if int(args.rl_steps) > 0 and (args.mode == "rl" or int(args.sft_steps) <= 0):
            print(
                f"starting query rl phase at step "
                f"{resume_step_offset + (1 if args.mode == 'rl' else int(args.sft_steps) + 1)} "
                f"(local_steps={int(args.rl_steps)})"
            )
        rl_progress = _make_progress_bar(total=int(args.rl_steps), desc="query rl")
        for step in range(1, int(args.rl_steps) + 1):
            log_step = resume_step_offset + int(args.sft_steps) + int(step)
            if bool(args.async_checkpoint_eval):
                async_eval_jobs, completed_async_results = poll_checkpoint_eval_jobs(async_eval_jobs)
                best_metric_value, completed_successes = _ingest_async_query_eval_results(
                    results=completed_async_results,
                    eval_history_path=eval_history_path,
                    log_step=log_step,
                    best_metric_name=best_metric_name,
                    best_metric_value=best_metric_value,
                )
                async_eval_success_count += int(completed_successes)
            batch = _sample_batch(
                train_examples,
                batch_size=args.batch_size,
                rng=rng,
                multi_issue_multiplier=float(args.multi_issue_sample_multiplier),
                multi_issue_max_share=float(args.multi_issue_sample_max_share),
            )
            requests = [
                _build_request(
                    example,
                    reasoning=bool(args.reasoning),
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                    max_tokens=int(args.max_tokens),
                )
                for example in batch
            ]
            try:
                results = _query_train_step_with_retry(
                    invoke=lambda: finetune.rollouts_batch(
                        requests=requests,
                        num_rollouts=int(args.num_rollouts),
                        max_workers=int(args.rollout_stream_max_concurrency),
                    ),
                    context=f"query rl rollout step {step}",
                    max_retries=int(args.train_step_max_retries),
                    backoff_base_s=float(args.train_step_retry_backoff_base_s),
                    backoff_max_s=float(args.train_step_retry_backoff_max_s),
                )
            except (TunaAPIError, TunaNetworkError) as exc:
                print(f"query rl rollouts failed at step {step}: {_format_tuna_error(exc)}")
                if _is_transient_query_train_step_error(exc) and rl_consecutive_failures < int(args.max_consecutive_train_step_failures):
                    rl_consecutive_failures += 1
                    _wandb_log(
                        {
                            "stage": "rl",
                            "rollout_failed": 1.0,
                            "train_step_failed": 1.0,
                            "train_step_applied": 0.0,
                            "consecutive_train_step_failures": float(rl_consecutive_failures),
                        },
                        step=log_step,
                        namespace="train",
                    )
                    cooldown_s = float(args.transient_train_step_failure_cooldown_s)
                    if cooldown_s > 0:
                        print(
                            f"query rl skipping step {step} after transient rollout failure; "
                            f"cooldown {cooldown_s:.1f}s before continuing"
                        )
                        time.sleep(cooldown_s)
                    rl_progress.set_postfix({"step": step, "status": "retry_skip"}, refresh=False)
                    rl_progress.update(1)
                    continue
                raise
            groups: list[TrainStepGroup] = []
            local_reward_values: list[float] = []
            judge_score_values: list[float] = []
            parse_success_values: list[float] = []
            task_correct_values: list[float] = []
            degraded_values: list[float] = []
            for example, request, result in zip(batch, requests, results):
                rewards = []
                for rollout in result.rollouts:
                    answer_text = str(getattr(rollout.output, "answer", "") or "")
                    local_outcome, parse_outcome = _score_answer_text(example, answer_text, grader=grader)
                    judgement = _judge_answer_with_cache(
                        grader=grader,
                        cache=judge_cache,
                        cache_path=judge_cache_path,
                        example=example,
                        answer_text=answer_text,
                        parse_outcome=parse_outcome,
                    )
                    judge_score = float(judgement.get("score", 0.0))
                    reward = float(local_outcome.reward) if bool(judgement.get("degraded", False)) else (0.7 * judge_score) + (0.3 * float(local_outcome.reward))
                    local_reward_values.append(float(local_outcome.reward))
                    judge_score_values.append(judge_score)
                    parse_success_values.append(float(bool(local_outcome.parse_success)))
                    task_correct_values.append(float(bool(local_outcome.task_correct)))
                    degraded_values.append(float(bool(judgement.get("degraded", False))))
                    rewards.append(reward)
                groups.append(result.to_group(rewards=rewards))
                if bool(args.off_policy):
                    off_policy_buffer.append(
                        TrainStepGroup.from_sft(
                            request=request,
                            targets=[_sft_target(example, include_reasoning=bool(args.reasoning))],
                        )
                    )
            off_policy_injected = 0
            if bool(args.off_policy) and off_policy_buffer:
                inject_count = min(len(off_policy_buffer), max(1, int(round(len(groups) * float(args.off_policy_mix_ratio)))))
                rl_keep_count = max(0, len(groups) - inject_count)
                selected_off_policy = random.sample(list(off_policy_buffer), k=inject_count)
                groups = groups[:rl_keep_count] + selected_off_policy
                off_policy_injected = int(inject_count)
            try:
                train_out = _query_train_step_with_retry(
                    invoke=lambda: finetune.train_step(groups=groups, lr=float(args.rl_lr)),
                    context=f"query rl step {step}",
                    max_retries=int(args.train_step_max_retries),
                    backoff_base_s=float(args.train_step_retry_backoff_base_s),
                    backoff_max_s=float(args.train_step_retry_backoff_max_s),
                )
            except (TunaAPIError, TunaNetworkError) as exc:
                print(f"query rl train_step failed at step {step}: {_format_tuna_error(exc)}")
                if _is_transient_query_train_step_error(exc) and rl_consecutive_failures < int(args.max_consecutive_train_step_failures):
                    rl_consecutive_failures += 1
                    _wandb_log(
                        {
                            "stage": "rl",
                            "train_step_failed": 1.0,
                            "train_step_applied": 0.0,
                            "consecutive_train_step_failures": float(rl_consecutive_failures),
                        },
                        step=log_step,
                        namespace="train",
                    )
                    cooldown_s = float(args.transient_train_step_failure_cooldown_s)
                    if cooldown_s > 0:
                        print(
                            f"query rl skipping step {step} after transient backend failure; "
                            f"cooldown {cooldown_s:.1f}s before continuing"
                        )
                        time.sleep(cooldown_s)
                    rl_progress.set_postfix({"step": step, "status": "retry_skip"}, refresh=False)
                    rl_progress.update(1)
                    continue
                raise
            rl_consecutive_failures = 0
            reward_mean_value = fmean([float(value) for group in groups for value in list(group.rewards or [])]) if groups else 0.0
            local_reward_mean_value = fmean(local_reward_values) if local_reward_values else 0.0
            judge_score_mean_value = fmean(judge_score_values) if judge_score_values else 0.0
            _wandb_log(
                {
                    "stage": "rl",
                    "train_step_failed": 0.0,
                    "train_step_applied": 1.0,
                    "consecutive_train_step_failures": 0.0,
                    "group_count": len(groups),
                    "batch_size": len(batch),
                    "lr": float(args.rl_lr),
                    "reward_mean": reward_mean_value,
                    "local_reward_mean": local_reward_mean_value,
                    "judge_score_mean": judge_score_mean_value,
                    "parse_success_rate": fmean(parse_success_values) if parse_success_values else 0.0,
                    "task_correct_rate": fmean(task_correct_values) if task_correct_values else 0.0,
                    "judge_degraded_rate": fmean(degraded_values) if degraded_values else 0.0,
                    "off_policy_injected": int(off_policy_injected),
                    "sft_loss": float(getattr(train_out, "sft_loss", 0.0) or 0.0),
                    "kl": float(getattr(train_out, "kl", 0.0) or 0.0),
                    "router_kl": float(getattr(train_out, "router_kl", 0.0) or 0.0),
                    "grad_norm": float(getattr(train_out, "grad_norm", 0.0) or 0.0),
                },
                step=log_step,
                namespace="train",
            )
            print(
                _format_query_rl_step_log(
                    step=log_step,
                    batch_size=len(batch),
                    reward_mean=reward_mean_value,
                    local_reward_mean=local_reward_mean_value,
                    judge_score_mean=judge_score_mean_value,
                    kl=float(getattr(train_out, "kl", 0.0) or 0.0),
                    off_policy_injected=int(off_policy_injected),
                    group_count=len(groups),
                ),
                flush=True,
            )
            checkpoint_saved_for_eval = False
            if int(args.eval_every) > 0 and log_step % int(args.eval_every) == 0:
                if bool(args.async_checkpoint_eval):
                    checkpoint_step = save_checkpoint(context=f"rl async eval step={log_step}")
                    if checkpoint_step is not None:
                        checkpoint_saved_for_eval = True
                        job = dispatch_checkpoint_eval(
                            trainer="inspector_query",
                            finetune_id=str(finetune.finetune_id),
                            checkpoint_step=int(checkpoint_step),
                            selection_metric=str(best_metric_name),
                            base_dir=str(args.async_checkpoint_eval_dir),
                            command_builder=lambda metrics_json_path, predictions_jsonl_path, _stdout_log_path: _build_async_query_eval_command(
                                args=args,
                                finetune_id=str(finetune.finetune_id),
                                checkpoint_step=int(checkpoint_step),
                                metrics_json_path=metrics_json_path,
                                predictions_jsonl_path=predictions_jsonl_path,
                            ),
                            metadata={
                                "step_for_log": log_step,
                                "stage": "rl",
                                "split_name": str(args.val_split),
                            },
                            max_inflight=int(args.async_checkpoint_eval_max_inflight),
                            inflight_jobs=async_eval_jobs,
                        )
                        if job is not None:
                            async_eval_jobs.append(job)
                    else:
                        print(f"async query eval skipped step={step}: checkpoint save unavailable")
                else:
                    checkpoint_saved_for_eval = bool(evaluate(log_step, "rl"))
            if int(args.save_every) > 0 and log_step % int(args.save_every) == 0:
                if not checkpoint_saved_for_eval:
                    save_checkpoint(context=f"rl save_every step={log_step}")
            rl_progress.set_postfix(
                {
                    "step": step,
                    "reward": f"{reward_mean_value:.3f}",
                    "judge": f"{judge_score_mean_value:.3f}",
                    "kl": f"{float(getattr(train_out, 'kl', 0.0) or 0.0):.3f}",
                },
                refresh=False,
            )
            rl_progress.update(1)
        rl_progress.close()

    if bool(args.async_checkpoint_eval) and bool(args.async_checkpoint_eval_drain_on_exit):
        completed_async_results = drain_checkpoint_eval_jobs(async_eval_jobs)
        best_metric_value, completed_successes = _ingest_async_query_eval_results(
            results=completed_async_results,
            eval_history_path=eval_history_path,
            log_step=resume_step_offset + total_train_steps,
            best_metric_name=best_metric_name,
            best_metric_value=best_metric_value,
        )
        async_eval_success_count += int(completed_successes)

    final_eval: dict[str, Any] = {}
    final_checkpoint_step = save_checkpoint(context="final eval checkpoint save")
    if final_checkpoint_step is None:
        print("final query eval skipped: checkpoint save unavailable")
    else:
        final_eval_max_samples = (
            int(args.final_eval_max_samples)
            if int(args.final_eval_max_samples) > 0
            else int(args.eval_max_samples)
        )
        for split_name in list(args.final_eval_splits):
            split_metrics = run_sync_eval(
                step=resume_step_offset + total_train_steps,
                stage="final",
                split_name=str(split_name),
                max_samples=int(final_eval_max_samples),
                checkpoint_step=int(final_checkpoint_step),
                checkpoint_context="final eval checkpoint save",
                namespace="test" if str(split_name) == "test" else "eval",
            )
            if split_metrics is None:
                continue
            final_eval[split_name] = split_metrics
            _record_eval(
                path=eval_history_path,
                step=resume_step_offset + total_train_steps,
                stage="final",
                split_name=split_name,
                metrics=final_eval[split_name],
            )
            _wandb_log(
                {
                    **final_eval[split_name],
                    "split": str(split_name),
                    "stage": "final",
                },
                step=resume_step_offset + total_train_steps,
                namespace="test" if str(split_name) == "test" else "eval",
            )
    common.write_json(
        run_dir / "train_summary.json",
        {
            "finetune_id": finetune.finetune_id,
            "best_metric": best_metric_name,
            "best_metric_value": best_metric_value,
            "final_eval": final_eval,
            "mode": args.mode,
            "reasoning": bool(args.reasoning),
            "off_policy": bool(args.off_policy),
            "num_rollouts": int(args.num_rollouts),
            "rollout_stream_max_concurrency": int(args.rollout_stream_max_concurrency),
            "rollout_stream_buffer_size": int(args.rollout_stream_buffer_size),
            "async_checkpoint_eval": bool(args.async_checkpoint_eval),
            "async_checkpoint_eval_dir": str(args.async_checkpoint_eval_dir),
            "async_checkpoint_eval_max_inflight": int(args.async_checkpoint_eval_max_inflight),
            "async_checkpoint_eval_success_count": int(async_eval_success_count),
            "judge_cache_jsonl": str(judge_cache_path),
            "grader_model_id": grader.model_id,
        },
    )
    run.summary["best_metric"] = str(best_metric_name)
    run.summary["best_metric_value"] = float(best_metric_value)
    run.summary["async_checkpoint_eval_enabled"] = int(bool(args.async_checkpoint_eval))
    run.summary["async_checkpoint_eval_success_count"] = int(async_eval_success_count)
    if latest_checkpoint_step is not None:
        run.summary["latest_checkpoint_step"] = int(latest_checkpoint_step)
    run.finish()
    print(f"finished finetune={finetune.finetune_id} run_dir={run_dir}")


if __name__ == "__main__":
    main()
