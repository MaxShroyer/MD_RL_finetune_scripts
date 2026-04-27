from __future__ import annotations

import json
import random
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from PIL import Image

from inspector_md import ontology, openrouter_grader, prompt_library, query_compact, task_schema
from inspector_md.common import (
    ApiKeyPool,
    ApiKeySlot,
    DEFAULT_BASE_MODEL,
    DEFAULT_BASE_URL,
    build_auth_headers,
    clamp,
    parse_prediction_json,
    to_data_url,
    truncate,
)
from tuna_sdk import DetectAnnotation, DetectRequest, DetectSettings, PointAnnotation, PointRequest, PointSettings, QueryRequest, QuerySettings


TRANSIENT_INFERENCE_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504, 520, 522, 524})


class MoondreamInferenceError(Exception):
    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        request_id: str = "",
        body: str = "",
        transient: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = None if status_code is None else int(status_code)
        self.request_id = str(request_id or "").strip()
        self.body = str(body or "")
        self.transient = bool(transient)


def is_transient_inference_error(exc: Exception) -> bool:
    return isinstance(exc, MoondreamInferenceError) and bool(exc.transient)


@dataclass(frozen=True)
class QueryResult:
    payload: dict[str, Any]
    raw_response: dict[str, Any]
    latency_ms: float
    answer_text: str = ""


def _extract_answer_text(data: dict[str, Any]) -> str:
    if isinstance(data.get("answer"), str):
        return str(data["answer"])
    output = data.get("output")
    if isinstance(output, dict):
        if isinstance(output.get("answer"), str):
            return str(output["answer"])
        content = output.get("content")
        if isinstance(content, list):
            text_parts = [str(item.get("text") or "") for item in content if isinstance(item, dict)]
            if text_parts:
                return "\n".join(text_parts)
    return ""


def _http_error_details(exc: urllib.error.HTTPError) -> tuple[str, str]:
    request_id = str(exc.headers.get("x-request-id") or "").strip()
    try:
        body = exc.read().decode("utf-8", errors="replace")
    except Exception:  # pragma: no cover
        body = ""
    return request_id, body


def _maybe_attach_model(payload: dict[str, Any], model: str) -> None:
    normalized = str(model or "").strip()
    if not normalized or normalized == DEFAULT_BASE_MODEL:
        return
    payload["model"] = normalized


def _post_json(
    *,
    api_base: str,
    api_key: str,
    endpoint: str,
    payloads: list[dict[str, Any]],
    timeout: float,
    max_retries: int,
    backoff_base_s: float,
    backoff_max_s: float,
    retry_jitter_s: float,
) -> tuple[dict[str, Any], float]:
    first_error: Optional[Exception] = None
    last_latency_ms = 0.0
    for payload in payloads:
        for attempt in range(max(0, int(max_retries)) + 1):
            request = urllib.request.Request(
                api_base.rstrip("/") + endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers=build_auth_headers(api_key),
                method="POST",
            )
            started = time.monotonic()
            try:
                with urllib.request.urlopen(request, timeout=float(timeout)) as response:
                    body = response.read().decode("utf-8", errors="replace")
                last_latency_ms = (time.monotonic() - started) * 1000.0
                data = json.loads(body) if body else {}
                if not isinstance(data, dict):
                    data = {}
                return data, last_latency_ms
            except urllib.error.HTTPError as exc:
                last_latency_ms = (time.monotonic() - started) * 1000.0
                request_id, body_text = _http_error_details(exc)
                first_error = MoondreamInferenceError(
                    f"HTTP {exc.code} request_id={request_id or '-'} body={truncate(body_text)}",
                    status_code=int(exc.code),
                    request_id=request_id,
                    body=body_text,
                    transient=int(exc.code) in TRANSIENT_INFERENCE_STATUS_CODES,
                )
            except (TimeoutError, socket.timeout, urllib.error.URLError) as exc:
                first_error = MoondreamInferenceError(
                    f"Network error: {exc}",
                    transient=True,
                )
            if not is_transient_inference_error(first_error) or attempt >= int(max_retries):
                break
            sleep_s = min(
                float(backoff_max_s),
                float(backoff_base_s) * (2 ** attempt),
            ) + random.uniform(0.0, max(0.0, float(retry_jitter_s)))
            time.sleep(max(0.0, sleep_s))
    if first_error is not None:
        raise first_error
    raise MoondreamInferenceError("No payloads were provided.")


class MoondreamInspectorClient:
    def __init__(
        self,
        *,
        api_key: str = "",
        api_key_pool: Optional[ApiKeyPool] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 60.0,
        max_retries: int = 4,
        backoff_base_s: float = 2.0,
        backoff_max_s: float = 20.0,
        retry_jitter_s: float = 0.5,
        post_json: Optional[Callable[..., tuple[dict[str, Any], float]]] = None,
    ) -> None:
        explicit_key = str(api_key or "").strip()
        if api_key_pool is not None:
            self.api_key_pool = api_key_pool
        elif explicit_key:
            self.api_key_pool = ApiKeyPool([ApiKeySlot(index=0, env_var="<explicit>", api_key=explicit_key)])
        else:
            raise ValueError("MoondreamInspectorClient requires api_key or api_key_pool.")
        self.base_url = str(base_url).strip().rstrip("/")
        self.timeout = float(timeout)
        self.max_retries = max(0, int(max_retries))
        self.backoff_base_s = float(backoff_base_s)
        self.backoff_max_s = float(backoff_max_s)
        self.retry_jitter_s = float(retry_jitter_s)
        self._post_json = post_json or _post_json

    def _load_image_url(self, image_path: str | Path) -> str:
        with Image.open(Path(image_path)) as image:
            return to_data_url(image.convert("RGB"))

    @property
    def active_api_key_env_vars(self) -> list[str]:
        return self.api_key_pool.env_var_names

    def describe_key_pool(self) -> dict[str, Any]:
        return self.api_key_pool.describe()

    def query_raw(self, *, model: str, request: QueryRequest) -> QueryResult:
        payload = request.to_payload()
        _maybe_attach_model(payload, model)
        slot = self.api_key_pool.next_slot()
        data, latency_ms = self._post_json(
            api_base=self.base_url,
            api_key=slot.api_key,
            endpoint="/query",
            payloads=[payload],
            timeout=self.timeout,
            max_retries=self.max_retries,
            backoff_base_s=self.backoff_base_s,
            backoff_max_s=self.backoff_max_s,
            retry_jitter_s=self.retry_jitter_s,
        )
        answer_text = _extract_answer_text(data)
        parsed = parse_prediction_json(answer_text) or {}
        return QueryResult(payload=parsed, raw_response=data, latency_ms=latency_ms, answer_text=answer_text)

    def detect_boxes(
        self,
        *,
        model: str,
        image_path: str | Path,
        detect_label: str,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 256,
        max_objects: int = 24,
    ) -> list[DetectAnnotation]:
        request = DetectRequest(
            object_name=str(detect_label).strip(),
            image_url=self._load_image_url(image_path),
            settings=DetectSettings(
                temperature=float(temperature),
                top_p=float(top_p),
                max_tokens=int(max_tokens),
                max_objects=int(max_objects),
            ),
        )
        payload = request.to_payload()
        _maybe_attach_model(payload, model)
        fallback_payload = dict(payload)
        fallback_payload.pop("skill", None)
        slot = self.api_key_pool.next_slot()
        data, _ = self._post_json(
            api_base=self.base_url,
            api_key=slot.api_key,
            endpoint="/detect",
            payloads=[payload, fallback_payload],
            timeout=self.timeout,
            max_retries=self.max_retries,
            backoff_base_s=self.backoff_base_s,
            backoff_max_s=self.backoff_max_s,
            retry_jitter_s=self.retry_jitter_s,
        )
        raw_objects = data.get("objects")
        if raw_objects is None and isinstance(data.get("output"), dict):
            raw_objects = data["output"].get("objects")
        boxes: list[DetectAnnotation] = []
        for item in list(raw_objects or []):
            if not isinstance(item, dict):
                continue
            try:
                box = DetectAnnotation(
                    x_min=clamp(float(item["x_min"])),
                    y_min=clamp(float(item["y_min"])),
                    x_max=clamp(float(item["x_max"])),
                    y_max=clamp(float(item["y_max"])),
                )
            except (KeyError, TypeError, ValueError):
                continue
            if box.x_max <= box.x_min or box.y_max <= box.y_min:
                continue
            boxes.append(box)
        return boxes

    def point_locations(
        self,
        *,
        model: str,
        image_path: str | Path,
        point_label: str,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 128,
    ) -> list[PointAnnotation]:
        request = PointRequest(
            object_name=str(point_label).strip(),
            image_url=self._load_image_url(image_path),
            settings=PointSettings(
                temperature=float(temperature),
                top_p=float(top_p),
                max_tokens=int(max_tokens),
            ),
        )
        payload = request.to_payload()
        _maybe_attach_model(payload, model)
        fallback_payload = dict(payload)
        fallback_payload.pop("skill", None)
        slot = self.api_key_pool.next_slot()
        data, _ = self._post_json(
            api_base=self.base_url,
            api_key=slot.api_key,
            endpoint="/point",
            payloads=[payload, fallback_payload],
            timeout=self.timeout,
            max_retries=self.max_retries,
            backoff_base_s=self.backoff_base_s,
            backoff_max_s=self.backoff_max_s,
            retry_jitter_s=self.retry_jitter_s,
        )
        points_raw: Any = data.get("points")
        if points_raw is None and isinstance(data.get("output"), dict):
            points_raw = data["output"].get("points")
        if isinstance(points_raw, dict):
            points_raw = [points_raw]
        points: list[PointAnnotation] = []
        for item in list(points_raw or []):
            if not isinstance(item, dict):
                continue
            try:
                points.append(PointAnnotation(x=clamp(float(item["x"])), y=clamp(float(item["y"]))))
            except (KeyError, TypeError, ValueError):
                continue
        return points

    def propose_issues(
        self,
        *,
        model: str,
        image_path: str | Path,
        inspection_request: str,
        asset_context: str = "",
        prompt_style: str = "request_only",
        normalizer: Optional[openrouter_grader.OpenRouterGrader] = None,
        reasoning: bool = False,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 512,
    ) -> tuple[list[task_schema.IssueProposal], QueryResult]:
        request = QueryRequest(
            question=prompt_library.build_visible_issue_question_with_style(
                inspection_request=inspection_request,
                asset_context=asset_context,
                prompt_style=prompt_style,
                variation_key=str(image_path),
            ),
            image_url=self._load_image_url(image_path),
            reasoning=bool(reasoning),
            settings=QuerySettings(
                temperature=float(temperature),
                top_p=float(top_p),
                max_tokens=int(max_tokens),
            ),
        )
        result = self.query_raw(model=model, request=request)
        proposals = task_schema.normalize_issue_proposals(result.payload, loose_issue_codes=True)
        payload = result.payload
        if not proposals and result.answer_text:
            issue_payload = query_compact.parse_issue_list_answer(result.answer_text)
            if issue_payload is not None:
                payload = issue_payload
                proposals = task_schema.normalize_issue_proposals(issue_payload, loose_issue_codes=True)
        if not proposals and normalizer is not None and result.answer_text:
            normalized = normalizer.normalize_issues(
                inspection_request=inspection_request,
                asset_context=asset_context,
                answer_text=result.answer_text,
            )
            payload = {"issues": list(normalized.get("issues") or [])}
            proposals = task_schema.normalize_issue_proposals(payload, loose_issue_codes=True)
        return proposals, QueryResult(payload=payload, raw_response=result.raw_response, latency_ms=result.latency_ms, answer_text=result.answer_text)

    def query_finding_fields(
        self,
        *,
        model: str,
        image_path: str | Path,
        localized_issue: task_schema.LocalizedIssue,
        inspection_request: str,
        asset_context: str = "",
        prompt_style: str = "minimal",
        normalizer: Optional[openrouter_grader.OpenRouterGrader] = None,
        reasoning: bool = False,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 384,
    ) -> QueryResult:
        request = QueryRequest(
            question=prompt_library.build_finding_question_with_style(
                localized_issue=localized_issue,
                inspection_request=inspection_request,
                asset_context=asset_context,
                prompt_style=prompt_style,
            ),
            image_url=self._load_image_url(image_path),
            spatial_refs=[prompt_library.box_to_spatial_ref(localized_issue.box)],
            reasoning=bool(reasoning),
            settings=QuerySettings(
                temperature=float(temperature),
                top_p=float(top_p),
                max_tokens=int(max_tokens),
            ),
        )
        result = self.query_raw(model=model, request=request)
        payload = result.payload
        finding_payload = payload.get("finding") if isinstance(payload.get("finding"), dict) else payload
        if not isinstance(finding_payload, dict) and result.answer_text:
            compact_payload = query_compact.parse_finding_answer(
                result.answer_text,
                default_issue_code=localized_issue.issue_code,
                default_title=ontology.get_issue(localized_issue.issue_code).title,
            )
            if compact_payload is not None:
                payload = compact_payload
        if not isinstance((payload.get("finding") if isinstance(payload.get("finding"), dict) else payload), dict) and normalizer is not None and result.answer_text:
            normalized = normalizer.normalize_finding(
                localized_issue=localized_issue,
                inspection_request=inspection_request,
                asset_context=asset_context,
                answer_text=result.answer_text,
            )
            payload = {"finding": dict(normalized.get("finding") or {})}
        return QueryResult(payload=payload, raw_response=result.raw_response, latency_ms=result.latency_ms, answer_text=result.answer_text)
