from __future__ import annotations

import json
import os
import hashlib
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Optional

from inspector_md import common, ontology, task_schema

DEFAULT_OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_ENV_VAR = "OPENROUTER_API_KEY"
DEFAULT_GRADER_MODEL = "openai/gpt-4.1-mini"
DEFAULT_GRADER_PROFILE = "balanced"
DEFAULT_GRADER_RUBRIC_VERSION = "query_issues_v2"

YES_PARTIAL_NO = {"yes", "partial", "no"}
UNSUPPORTED_LEVELS = {"none", "minor", "major"}
VERBOSITY_LEVELS = {"concise", "acceptable", "verbose"}
ACTION_QUALITY_LEVELS = {"good", "partial", "poor"}

PROPOSAL_BASE_SCORES = {"yes": 1.0, "partial": 0.5, "no": 0.0}
PROPOSAL_GROUNDED_PENALTIES = {"yes": 0.0, "partial": 0.1, "no": 0.25}
PROPOSAL_UNSUPPORTED_PENALTIES = {"none": 0.0, "minor": 0.1, "major": 0.25}
PROPOSAL_EXTRA_ISSUE_PENALTIES = {"none": 0.0, "minor": 0.05, "major": 0.15}
PROPOSAL_VERBOSITY_PENALTIES = {"concise": 0.0, "acceptable": 0.0, "verbose": 0.1}

FINDING_BASE_SCORES = {"yes": 1.0, "partial": 0.5, "no": 0.0}
FINDING_GROUNDED_PENALTIES = {"yes": 0.0, "partial": 0.1, "no": 0.25}
FINDING_ACTION_QUALITY_PENALTIES = {"good": 0.0, "partial": 0.05, "poor": 0.1}
FINDING_UNSUPPORTED_PENALTIES = {"none": 0.0, "minor": 0.1, "major": 0.25}
FINDING_VERBOSITY_PENALTIES = {"concise": 0.0, "acceptable": 0.0, "verbose": 0.1}


class OpenRouterGradingError(Exception):
    pass


def resolve_openrouter_api_key(*, explicit_api_key: str = "", api_key_env_var: str = DEFAULT_OPENROUTER_ENV_VAR) -> str:
    explicit = str(explicit_api_key or "").strip()
    if explicit:
        return explicit
    preferred = str(api_key_env_var or "").strip()
    if preferred:
        preferred_value = str(os.environ.get(preferred) or "").strip()
        if preferred_value:
            return preferred_value
    default_value = str(os.environ.get(DEFAULT_OPENROUTER_ENV_VAR) or "").strip()
    if default_value:
        return default_value
    raise ValueError(f"{DEFAULT_OPENROUTER_ENV_VAR} is required for OpenRouter grading.")


def _build_openrouter_headers(api_key: str) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {str(api_key).strip()}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    referer = str(os.environ.get("OPENROUTER_HTTP_REFERER") or "").strip()
    if referer:
        headers["HTTP-Referer"] = referer
    title = str(os.environ.get("OPENROUTER_APP_NAME") or "").strip()
    if title:
        headers["X-Title"] = title
    return headers


def _extract_openrouter_answer_text(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    choice = choices[0]
    if not isinstance(choice, dict):
        return ""
    message = choice.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if isinstance(item, dict) and isinstance(item.get("text"), str):
            parts.append(str(item["text"]))
    return "\n".join(parts)


def _http_error_details(exc: urllib.error.HTTPError) -> tuple[str, str]:
    request_id = str(exc.headers.get("x-request-id") or "").strip()
    try:
        body = exc.read().decode("utf-8", errors="replace")
    except Exception:  # pragma: no cover
        body = ""
    return request_id, body


def _call_openrouter_chat_api(
    *,
    api_base: str,
    api_key: str,
    model_id: str,
    messages: list[dict[str, Any]],
    timeout: float,
) -> tuple[str, dict[str, Any], float]:
    payload = {
        "model": str(model_id).strip(),
        "messages": list(messages),
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 800,
        "response_format": {"type": "json_object"},
    }
    endpoint = str(api_base).rstrip("/") + "/chat/completions"
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=_build_openrouter_headers(api_key),
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
        return _extract_openrouter_answer_text(data), data, latency_ms
    except urllib.error.HTTPError as exc:
        latency_ms = (time.monotonic() - started) * 1000.0
        request_id, body = _http_error_details(exc)
        raise OpenRouterGradingError(
            f"HTTP {exc.code} request_id={request_id or '-'} latency_ms={latency_ms:.1f} body={common.truncate(body, limit=400)}"
        ) from exc
    except urllib.error.URLError as exc:
        raise OpenRouterGradingError(f"Network error: {exc}") from exc


def _issue_family_notes() -> str:
    family_map: dict[str, list[str]] = {}
    for record in ontology.ISSUE_CATALOG:
        family_map.setdefault(record.evaluation_family, []).append(record.issue_code)
    parts: list[str] = []
    for family, codes in sorted(family_map.items()):
        parts.append(f"{family}: {', '.join(sorted(codes))}")
    return "\n".join(parts)


def cache_key_for_judgement(
    *,
    task_type: str,
    prompt_text: str,
    answer_text: str,
    canonical_gt: Any,
    model_id: str,
    rubric_version: str,
    profile: str,
) -> str:
    payload = {
        "task_type": str(task_type or "").strip().lower(),
        "prompt_text": str(prompt_text or "").strip(),
        "answer_text": str(answer_text or "").strip(),
        "canonical_gt": canonical_gt,
        "model_id": str(model_id or "").strip(),
        "rubric_version": str(rubric_version or "").strip(),
        "profile": str(profile or "").strip().lower(),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()


def _enum_value(value: Any, *, allowed: set[str], default: str) -> str:
    text = str(value or "").strip().lower()
    return text if text in allowed else default


def normalize_proposal_judgement_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        payload = {}
    return {
        "satisfies_gt": _enum_value(payload.get("satisfies_gt"), allowed=YES_PARTIAL_NO, default="no"),
        "grounded_visible_evidence": _enum_value(payload.get("grounded_visible_evidence"), allowed=YES_PARTIAL_NO, default="no"),
        "unsupported_claims": _enum_value(payload.get("unsupported_claims"), allowed=UNSUPPORTED_LEVELS, default="none"),
        "extra_issue_claims": _enum_value(payload.get("extra_issue_claims"), allowed=UNSUPPORTED_LEVELS, default="none"),
        "verbosity": _enum_value(payload.get("verbosity"), allowed=VERBOSITY_LEVELS, default="acceptable"),
        "reason": str(payload.get("reason") or "").strip(),
    }


def normalize_finding_judgement_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        payload = {}
    return {
        "satisfies_gt": _enum_value(payload.get("satisfies_gt"), allowed=YES_PARTIAL_NO, default="no"),
        "grounded_visible_evidence": _enum_value(payload.get("grounded_visible_evidence"), allowed=YES_PARTIAL_NO, default="no"),
        "recommended_action_quality": _enum_value(payload.get("recommended_action_quality"), allowed=ACTION_QUALITY_LEVELS, default="poor"),
        "unsupported_claims": _enum_value(payload.get("unsupported_claims"), allowed=UNSUPPORTED_LEVELS, default="none"),
        "verbosity": _enum_value(payload.get("verbosity"), allowed=VERBOSITY_LEVELS, default="acceptable"),
        "reason": str(payload.get("reason") or "").strip(),
    }


def score_proposal_judgement(payload: Any, *, profile: str = DEFAULT_GRADER_PROFILE) -> dict[str, Any]:
    if str(profile or DEFAULT_GRADER_PROFILE).strip().lower() != DEFAULT_GRADER_PROFILE:
        raise ValueError(f"Unsupported grader profile: {profile!r}")
    normalized = normalize_proposal_judgement_payload(payload)
    score = PROPOSAL_BASE_SCORES[normalized["satisfies_gt"]]
    score -= PROPOSAL_GROUNDED_PENALTIES[normalized["grounded_visible_evidence"]]
    score -= PROPOSAL_UNSUPPORTED_PENALTIES[normalized["unsupported_claims"]]
    score -= PROPOSAL_EXTRA_ISSUE_PENALTIES[normalized["extra_issue_claims"]]
    score -= PROPOSAL_VERBOSITY_PENALTIES[normalized["verbosity"]]
    normalized["score"] = common.clamp(score)
    return normalized


def score_issue_judgement(payload: Any, *, profile: str = DEFAULT_GRADER_PROFILE) -> dict[str, Any]:
    return score_proposal_judgement(payload, profile=profile)


def score_finding_judgement(payload: Any, *, profile: str = DEFAULT_GRADER_PROFILE) -> dict[str, Any]:
    if str(profile or DEFAULT_GRADER_PROFILE).strip().lower() != DEFAULT_GRADER_PROFILE:
        raise ValueError(f"Unsupported grader profile: {profile!r}")
    normalized = normalize_finding_judgement_payload(payload)
    score = FINDING_BASE_SCORES[normalized["satisfies_gt"]]
    score -= FINDING_GROUNDED_PENALTIES[normalized["grounded_visible_evidence"]]
    score -= FINDING_ACTION_QUALITY_PENALTIES[normalized["recommended_action_quality"]]
    score -= FINDING_UNSUPPORTED_PENALTIES[normalized["unsupported_claims"]]
    score -= FINDING_VERBOSITY_PENALTIES[normalized["verbosity"]]
    normalized["score"] = common.clamp(score)
    return normalized


def unmatched_finding_judgement(*, reason: str) -> dict[str, Any]:
    return score_finding_judgement(
        {
            "satisfies_gt": "no",
            "grounded_visible_evidence": "no",
            "recommended_action_quality": "poor",
            "unsupported_claims": "major",
            "verbosity": "acceptable",
            "reason": reason,
        }
    )


class OpenRouterGrader:
    def __init__(
        self,
        *,
        api_key: str,
        model_id: str = DEFAULT_GRADER_MODEL,
        api_base: str = DEFAULT_OPENROUTER_API_BASE,
        timeout: float = 60.0,
        profile: str = DEFAULT_GRADER_PROFILE,
        rubric_version: str = DEFAULT_GRADER_RUBRIC_VERSION,
        call_api_fn: Optional[Callable[..., tuple[str, dict[str, Any], float]]] = None,
    ) -> None:
        self.api_key = str(api_key).strip()
        self.model_id = str(model_id or DEFAULT_GRADER_MODEL).strip()
        self.api_base = str(api_base or DEFAULT_OPENROUTER_API_BASE).strip()
        self.timeout = float(timeout)
        self.profile = str(profile or DEFAULT_GRADER_PROFILE).strip().lower()
        self.rubric_version = str(rubric_version or DEFAULT_GRADER_RUBRIC_VERSION).strip()
        self._call_api = call_api_fn or _call_openrouter_chat_api

    def _chat_json(self, *, system_prompt: str, user_prompt: str) -> tuple[dict[str, Any], dict[str, Any], float]:
        answer_text, raw_payload, latency_ms = self._call_api(
            api_base=self.api_base,
            api_key=self.api_key,
            model_id=self.model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            timeout=self.timeout,
        )
        payload = common.parse_prediction_json(answer_text) or {}
        return payload, raw_payload, latency_ms

    def chat_json(self, *, system_prompt: str, user_prompt: str) -> tuple[dict[str, Any], dict[str, Any], float]:
        return self._chat_json(system_prompt=system_prompt, user_prompt=user_prompt)

    def normalize_issues(
        self,
        *,
        inspection_request: str,
        asset_context: str,
        answer_text: str,
    ) -> dict[str, Any]:
        if not str(answer_text or "").strip():
            return {"issues": [], "notes": "empty_answer", "raw_payload": {}, "latency_ms": 0.0}
        system_prompt = (
            "You normalize a full-image building inspection answer into canonical Inspector MD issues. "
            "Output JSON only."
        )
        user_prompt = "\n".join(
            [
                f"Request: {inspection_request.strip()}",
                f"Context: {asset_context.strip()}",
                f"Allowed issue_codes: {', '.join(ontology.all_issue_codes())}",
                "Return JSON with schema:",
                json.dumps(
                    {"issues": [{"type": "<allowed_issue_code>", "reasoning": "<short visible evidence>"}], "notes": "<brief note>"},
                    ensure_ascii=False,
                ),
                "If the answer contains no supported visible issue, return an empty issues list.",
                "Model answer:",
                str(answer_text).strip(),
            ]
        )
        payload, raw_payload, latency_ms = self._chat_json(system_prompt=system_prompt, user_prompt=user_prompt)
        try:
            issues = task_schema.issue_list_payload(task_schema.normalize_issue_proposals(payload)).get("issues", [])
        except Exception:
            issues = []
        return {
            "issues": issues,
            "notes": str(payload.get("notes") or "").strip(),
            "raw_payload": raw_payload,
            "latency_ms": latency_ms,
        }

    def normalize_proposals(
        self,
        *,
        inspection_request: str,
        asset_context: str,
        answer_text: str,
    ) -> dict[str, Any]:
        normalized = self.normalize_issues(
            inspection_request=inspection_request,
            asset_context=asset_context,
            answer_text=answer_text,
        )
        issues = list(normalized.get("issues") or [])
        return {
            "proposals": [
                {
                    "issue_code": str(item.get("type") or "").strip(),
                    "evidence": str(item.get("reasoning") or "").strip(),
                }
                for item in issues
                if str(item.get("type") or "").strip()
            ],
            "notes": str(normalized.get("notes") or "").strip(),
            "raw_payload": dict(normalized.get("raw_payload") or {}),
            "latency_ms": float(normalized.get("latency_ms", 0.0) or 0.0),
        }

    def grade_issues(
        self,
        *,
        inspection_request: str,
        asset_context: str,
        expected_issues: list[task_schema.IssueProposal],
        answer_text: str,
    ) -> dict[str, Any]:
        if not str(answer_text or "").strip():
            judgement = score_issue_judgement(
                {
                    "satisfies_gt": "no",
                    "grounded_visible_evidence": "no",
                    "unsupported_claims": "none",
                    "extra_issue_claims": "none",
                    "verbosity": "acceptable",
                    "reason": "empty_answer",
                },
                profile=self.profile,
            )
            judgement.update({"raw_payload": {}, "latency_ms": 0.0})
            return judgement
        system_prompt = (
            "You grade a full-image building-inspection issue-list answer against the ground truth using rubric labels only. "
            "Do not output any numeric scores. "
            f"Rubric version: {self.rubric_version}. "
            "Use yes when the answer satisfies the ground truth issue list. "
            "Use partial for same-family matches, incomplete satisfaction, or mixed correct/incorrect content. "
            "Use no when the answer fails to satisfy the ground truth. "
            "Judge only visible evidence and avoid rewarding unsupported causes, legal claims, or cost claims. "
            "Use extra_issue_claims for additional issue categories, speculative diagnoses, or overlisted defect claims. "
            "Treat overly long issue lists or long reason strings as verbosity problems."
        )
        expected_payload = task_schema.issue_list_payload(expected_issues)
        user_prompt = "\n".join(
            [
                f"Request: {inspection_request.strip()}",
                f"Context: {asset_context.strip()}",
                "Evaluation families:",
                _issue_family_notes(),
                "Expected issues:",
                json.dumps(expected_payload, ensure_ascii=False, sort_keys=True),
                "Model answer:",
                str(answer_text).strip(),
                "Return JSON with schema:",
                json.dumps(
                    {
                        "satisfies_gt": "yes|partial|no",
                        "grounded_visible_evidence": "yes|partial|no",
                        "unsupported_claims": "none|minor|major",
                        "extra_issue_claims": "none|minor|major",
                        "verbosity": "concise|acceptable|verbose",
                        "reason": "<brief explanation>",
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            ]
        )
        payload, raw_payload, latency_ms = self._chat_json(system_prompt=system_prompt, user_prompt=user_prompt)
        judgement = score_issue_judgement(payload, profile=self.profile)
        judgement.update({"raw_payload": raw_payload, "latency_ms": latency_ms})
        return judgement

    def grade_proposals(
        self,
        *,
        inspection_request: str,
        asset_context: str,
        expected_proposals: list[task_schema.IssueProposal],
        answer_text: str,
    ) -> dict[str, Any]:
        return self.grade_issues(
            inspection_request=inspection_request,
            asset_context=asset_context,
            expected_issues=expected_proposals,
            answer_text=answer_text,
        )

    def normalize_finding(
        self,
        *,
        localized_issue: task_schema.LocalizedIssue,
        inspection_request: str,
        asset_context: str,
        answer_text: str,
    ) -> dict[str, Any]:
        issue = ontology.get_issue(localized_issue.issue_code)
        if not str(answer_text or "").strip():
            return {
                "finding": {
                    "issue_code": issue.issue_code,
                    "title": issue.title,
                    "evidence": list(localized_issue.evidence),
                    "severity": "unknown",
                    "recommended_action": issue.default_recommended_action,
                    "insufficient_evidence": True,
                },
                "notes": "empty_answer",
                "raw_payload": {},
                "latency_ms": 0.0,
            }
        system_prompt = (
            "You convert a free-form building inspection answer about a highlighted image region into a concise structured finding. "
            "Output JSON only."
        )
        user_prompt = "\n".join(
            [
                f"Request: {inspection_request.strip()}",
                f"Context: {asset_context.strip()}",
                f"Allowed issue_code: {issue.issue_code}",
                f"Default title: {issue.title}",
                f"Detect labels: {', '.join(localized_issue.source_detect_labels)}",
                "Return JSON with schema:",
                json.dumps(
                    {
                        "finding": {
                            "issue_code": issue.issue_code,
                            "title": issue.title,
                            "evidence": ["<short visible evidence>"],
                            "severity": "minor|moderate|major|unknown",
                            "recommended_action": "<short action>",
                            "insufficient_evidence": False,
                        },
                        "notes": "<brief note>",
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "Model answer:",
                str(answer_text).strip(),
            ]
        )
        payload, raw_payload, latency_ms = self._chat_json(system_prompt=system_prompt, user_prompt=user_prompt)
        finding_payload = payload.get("finding") if isinstance(payload.get("finding"), dict) else payload
        evidence = finding_payload.get("evidence") if isinstance(finding_payload, dict) else None
        if isinstance(evidence, str):
            evidence = [evidence]
        normalized_finding = {
            "issue_code": issue.issue_code,
            "title": str((finding_payload or {}).get("title") or issue.title).strip(),
            "evidence": evidence if isinstance(evidence, list) else list(localized_issue.evidence),
            "severity": str((finding_payload or {}).get("severity") or "unknown").strip().lower() or "unknown",
            "recommended_action": str(
                (finding_payload or {}).get("recommended_action") or issue.default_recommended_action
            ).strip(),
            "insufficient_evidence": bool((finding_payload or {}).get("insufficient_evidence", False)),
        }
        return {
            "finding": normalized_finding,
            "notes": str(payload.get("notes") or "").strip(),
            "raw_payload": raw_payload,
            "latency_ms": latency_ms,
        }

    def grade_finding(
        self,
        *,
        inspection_request: str,
        asset_context: str,
        predicted_finding: task_schema.Finding,
        expected_finding: task_schema.Finding,
        answer_text: str,
    ) -> dict[str, Any]:
        if not str(answer_text or "").strip():
            judgement = score_finding_judgement(
                {
                    "satisfies_gt": "no",
                    "grounded_visible_evidence": "no",
                    "recommended_action_quality": "poor",
                    "unsupported_claims": "none",
                    "verbosity": "acceptable",
                    "reason": "empty_answer",
                },
                profile=self.profile,
            )
            judgement.update({"raw_payload": {}, "latency_ms": 0.0})
            return judgement
        system_prompt = (
            "You grade a highlighted-region building inspection answer against the ground truth finding using rubric labels only. "
            "Do not output any numeric scores. "
            f"Rubric version: {self.rubric_version}. "
            "Judge whether the answer satisfies the ground truth, stays grounded in visible evidence, and gives a usable recommended action. "
            "Penalize unsupported causes, legal claims, exact costs, and unnecessary verbosity. "
            "Do not judge box geometry, spatial precision, or localization quality."
        )
        expected_semantic = {
            "issue_code": expected_finding.issue_code,
            "title": expected_finding.title,
            "evidence": list(expected_finding.evidence),
            "severity": expected_finding.severity,
            "recommended_action": expected_finding.recommended_action,
            "insufficient_evidence": bool(expected_finding.insufficient_evidence),
        }
        predicted_semantic = {
            "issue_code": predicted_finding.issue_code,
            "title": predicted_finding.title,
            "evidence": list(predicted_finding.evidence),
            "severity": predicted_finding.severity,
            "recommended_action": predicted_finding.recommended_action,
            "insufficient_evidence": bool(predicted_finding.insufficient_evidence),
        }
        user_prompt = "\n".join(
            [
                f"Request: {inspection_request.strip()}",
                f"Context: {asset_context.strip()}",
                "Expected finding:",
                json.dumps(expected_semantic, ensure_ascii=False, sort_keys=True),
                "Predicted semantic finding:",
                json.dumps(predicted_semantic, ensure_ascii=False, sort_keys=True),
                "Model answer:",
                str(answer_text).strip(),
                "Return JSON with schema:",
                json.dumps(
                    {
                        "satisfies_gt": "yes|partial|no",
                        "grounded_visible_evidence": "yes|partial|no",
                        "recommended_action_quality": "good|partial|poor",
                        "unsupported_claims": "none|minor|major",
                        "verbosity": "concise|acceptable|verbose",
                        "reason": "<brief explanation>",
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            ]
        )
        payload, raw_payload, latency_ms = self._chat_json(system_prompt=system_prompt, user_prompt=user_prompt)
        judgement = score_finding_judgement(payload, profile=self.profile)
        judgement.update({"raw_payload": raw_payload, "latency_ms": latency_ms})
        return judgement
