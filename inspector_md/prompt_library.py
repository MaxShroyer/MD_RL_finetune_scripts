from __future__ import annotations

import hashlib
import json

from inspector_md import ontology
from inspector_md.task_schema import Box, LocalizedIssue

CANONICAL_QUERY_INSPECTION_REQUEST = (
    "Inspect this image for visible building or site issues. Use only what is directly visible."
)
CANONICAL_REGION_ONLY_QUERY_QUESTION = CANONICAL_QUERY_INSPECTION_REQUEST
VISIBLE_ISSUE_PROMPT_VARIANTS = (
    '{request} Return JSON only in the schema: {schema}\nIf none: {{"issues":[]}}',
    '{request} Respond with JSON only in the schema: {schema}\nIf none: {{"issues":[]}}',
    '{request} Output JSON only using the schema: {schema}\nIf none: {{"issues":[]}}',
    '{request} Use JSON only in the schema: {schema}\nIf none: {{"issues":[]}}',
)


def _issue_code_list() -> str:
    return ", ".join(ontology.all_issue_codes())


def proposal_schema_hint() -> dict[str, object]:
    return {"issues": [{"type": "<allowed_issue_code>", "reasoning": "<short visible evidence>"}]}


def finding_schema_hint(*, issue_code: str = "<issue_code>", title: str = "<issue_title>") -> dict[str, object]:
    return {
        "finding": {
            "issue_code": issue_code,
            "title": title,
            "evidence": ["<short visible evidence>"],
            "recommended_action": "<short recommended action>",
            "insufficient_evidence": False,
        }
    }


def build_visible_issue_question(*, inspection_request: str, asset_context: str = "") -> str:
    return build_visible_issue_question_with_style(
        inspection_request=inspection_request,
        asset_context=asset_context,
        prompt_style="structured",
    )


def build_visible_issue_question_with_style(
    *,
    inspection_request: str,
    asset_context: str = "",
    prompt_style: str = "structured",
    variation_key: str = "",
) -> str:
    style = str(prompt_style or "structured").strip().lower()
    request_text = inspection_request.strip() or CANONICAL_QUERY_INSPECTION_REQUEST
    schema = json.dumps(proposal_schema_hint(), sort_keys=True)
    key = str(variation_key or f"{request_text}|{style}").strip()
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    index = int(digest[:8], 16) % len(VISIBLE_ISSUE_PROMPT_VARIANTS)
    return VISIBLE_ISSUE_PROMPT_VARIANTS[index].format(request=request_text, schema=schema)


def build_finding_question(
    *,
    localized_issue: LocalizedIssue,
    inspection_request: str,
    asset_context: str = "",
) -> str:
    return build_finding_question_with_style(
        localized_issue=localized_issue,
        inspection_request=inspection_request,
        asset_context=asset_context,
        prompt_style="structured",
    )


def build_finding_question_with_style(
    *,
    localized_issue: LocalizedIssue,
    inspection_request: str,
    asset_context: str = "",
    prompt_style: str = "structured",
) -> str:
    issue = ontology.get_issue(localized_issue.issue_code)
    style = str(prompt_style or "structured").strip().lower()
    request_text = inspection_request.strip()
    lines = [
        "Return JSON only.",
        "Write one issue object for this image.",
        f"Schema: {json.dumps({'issues': [{'type': issue.issue_code, 'reasoning': '<short visible evidence>'}]}, sort_keys=True)}",
        f"Allowed type: {issue.issue_code}",
        "Use only visible evidence. No hidden causes, legal claims, or costs.",
        f"Request: {request_text}",
    ]
    if asset_context.strip():
        lines.append(f"Context: {asset_context.strip()}")
    if localized_issue.source_detect_labels and style not in {"request_only", "terse", "minimal", "freeform"}:
        lines.append("Detect labels: " + ", ".join(localized_issue.source_detect_labels))
    return "\n".join(lines)


def box_to_spatial_ref(box: Box) -> list[float]:
    return [box.x_min, box.y_min, box.x_max, box.y_max]
