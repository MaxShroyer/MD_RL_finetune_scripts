from __future__ import annotations

import json
import re
from typing import Any, Optional

from inspector_md import common, ontology, task_schema

PROPOSAL_NONE_TOKEN = "none"
TARGET_FORMAT_COMPACT_TEXT = "compact_text"
TARGET_FORMAT_JSON_ISSUE_LIST = "json_issue_list"
_SENTENCE_SPLIT_RE = re.compile(r"(?:\n+|(?<=[.!?;])\s+)")
_NO_ISSUE_MARKERS = (
    "no visible building or site issues",
    "no visible building issues",
    "no visible site issues",
    "no visible issues",
    "no issues visible",
    "no obvious issues",
    "no visible defects",
    "no defects visible",
    "no building or site issues",
)


def format_proposal_target(proposals: list[task_schema.IssueProposal]) -> str:
    if not proposals:
        return PROPOSAL_NONE_TOKEN
    lines: list[str] = []
    for proposal in proposals:
        evidence = str(proposal.evidence or "").strip()
        lines.append(f"{proposal.issue_code} | {evidence}")
    return "\n".join(lines)


def format_finding_target(finding: task_schema.Finding) -> str:
    evidence = " ".join(str(item).strip() for item in list(finding.evidence or []) if str(item).strip())
    insufficient = "true" if bool(finding.insufficient_evidence) else "false"
    return " | ".join(
        [
            finding.issue_code,
            str(finding.title or "").strip(),
            str(finding.severity or "unknown").strip().lower() or "unknown",
            insufficient,
            evidence,
            str(finding.recommended_action or "").strip(),
        ]
    )


def format_issue_list_target(proposals: list[task_schema.IssueProposal]) -> str:
    return json.dumps(task_schema.issue_list_payload(proposals), ensure_ascii=False, sort_keys=True)


def parse_bool_token(value: Any, *, default: bool = False) -> bool:
    text = str(value or "").strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return bool(default)


def _token_regex(token: str) -> str:
    normalized = common.normalize_text(token)
    if not normalized:
        return ""
    escaped = re.escape(normalized)
    if " " in normalized:
        return escaped.replace(r"\ ", r"\s+")
    if normalized.endswith("e") and len(normalized) > 3:
        return rf"{escaped}(?:s|d|ing)?"
    return rf"{escaped}(?:s|es|ed|ing)?"


def _phrase_regex(phrase: str) -> Optional[re.Pattern[str]]:
    normalized = common.normalize_text(phrase)
    if not normalized:
        return None
    tokens = normalized.split()
    if not tokens:
        return None
    token_patterns = [_token_regex(token) for token in tokens]
    body = r"\s+".join(part for part in token_patterns if part)
    if not body:
        return None
    return re.compile(rf"\b{body}\b")


def _proposal_evidence_text(answer_text: str, pattern: re.Pattern[str]) -> str:
    sentences = [str(item or "").strip() for item in _SENTENCE_SPLIT_RE.split(str(answer_text or "").strip())]
    for sentence in sentences:
        if sentence and pattern.search(common.normalize_text(sentence)):
            return sentence
    return str(answer_text or "").strip()


def _candidate_phrase_texts(record: ontology.IssueOntologyRecord) -> list[str]:
    candidates = [
        record.issue_code.replace("_", " "),
        record.title,
        *record.aliases,
        *record.detect_labels,
    ]
    seen: set[str] = set()
    output: list[str] = []
    for item in candidates:
        normalized = common.normalize_text(item)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(normalized)
    return output


def _heuristic_parse_proposal_answer(answer_text: str) -> Optional[list[dict[str, Any]]]:
    normalized_text = common.normalize_text(answer_text)
    if not normalized_text:
        return None
    matches: list[dict[str, Any]] = []
    for record in ontology.ISSUE_CATALOG:
        best_match: Optional[dict[str, Any]] = None
        for phrase in _candidate_phrase_texts(record):
            pattern = _phrase_regex(phrase)
            if pattern is None:
                continue
            match = pattern.search(normalized_text)
            if match is None:
                continue
            candidate = {
                "issue_code": record.issue_code,
                "family": record.evaluation_family,
                "start": int(match.start()),
                "end": int(match.end()),
                "specificity": len(phrase.split()),
                "evidence": _proposal_evidence_text(answer_text, pattern),
            }
            if best_match is None or (
                candidate["start"],
                -candidate["specificity"],
                -(candidate["end"] - candidate["start"]),
            ) < (
                best_match["start"],
                -best_match["specificity"],
                -(best_match["end"] - best_match["start"]),
            ):
                best_match = candidate
        if best_match is not None:
            matches.append(best_match)
    if not matches:
        return None
    matches.sort(key=lambda item: (item["start"], -item["specificity"], item["issue_code"]))
    kept: list[dict[str, Any]] = []
    for candidate in matches:
        candidate_start = int(candidate["start"])
        candidate_end = int(candidate["end"])
        candidate_family = str(candidate["family"])
        candidate_specificity = int(candidate["specificity"])
        suppressed = False
        for existing in kept:
            if str(existing["family"]) != candidate_family:
                continue
            existing_start = int(existing["start"])
            existing_end = int(existing["end"])
            overlaps = not (candidate_end <= existing_start or existing_end <= candidate_start)
            if overlaps and int(existing["specificity"]) >= candidate_specificity:
                suppressed = True
                break
        if suppressed:
            continue
        kept = [
            existing
            for existing in kept
            if not (
                str(existing["family"]) == candidate_family
                and not (int(existing["end"]) <= candidate_start or candidate_end <= int(existing["start"]))
                and candidate_specificity > int(existing["specificity"])
            )
        ]
        kept.append(candidate)
    kept.sort(key=lambda item: (item["start"], item["issue_code"]))
    proposals: list[dict[str, Any]] = []
    seen_codes: set[str] = set()
    for item in kept:
        issue_code = str(item["issue_code"])
        if issue_code in seen_codes:
            continue
        seen_codes.add(issue_code)
        proposals.append(
            {
                "issue_code": issue_code,
                "evidence": str(item["evidence"] or "").strip(),
            }
        )
    return proposals or None


def _looks_like_no_issue_answer(answer_text: str) -> bool:
    normalized = common.normalize_text(answer_text)
    return any(marker in normalized for marker in _NO_ISSUE_MARKERS)


def parse_proposal_answer_detailed(answer_text: Any) -> tuple[Optional[dict[str, Any]], str]:
    text = str(answer_text or "").strip()
    if not text:
        return None, "unparsed"
    if text.lower() == PROPOSAL_NONE_TOKEN:
        return {"proposals": []}, "compact_text"
    lines = [str(raw_line or "").strip() for raw_line in text.splitlines() if str(raw_line or "").strip()]
    compact_candidate = bool(lines)
    proposals: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line in lines:
        issue_text, sep, evidence_text = line.partition("|")
        if not sep:
            compact_candidate = False
            break
        try:
            issue_code = ontology.normalize_issue_code(issue_text, loose=True)
        except ValueError:
            compact_candidate = False
            break
        if issue_code in seen:
            continue
        seen.add(issue_code)
        proposals.append(
            {
                "issue_code": issue_code,
                "evidence": str(evidence_text or "").strip(),
            }
        )
    if compact_candidate and proposals:
        return {"proposals": proposals}, "compact_text"
    heuristic_proposals = _heuristic_parse_proposal_answer(text)
    if heuristic_proposals is not None:
        return {"proposals": heuristic_proposals}, "heuristic_text"
    if _looks_like_no_issue_answer(text):
        return {"proposals": []}, "heuristic_none"
    return None, "unparsed"


def parse_proposal_answer(answer_text: Any) -> Optional[dict[str, Any]]:
    payload, _method = parse_proposal_answer_detailed(answer_text)
    return payload


def parse_issue_list_answer_detailed(answer_text: Any) -> tuple[Optional[dict[str, Any]], str]:
    text = str(answer_text or "").strip()
    if not text:
        return None, "unparsed"
    json_payload = common.parse_prediction_json(text)
    if isinstance(json_payload, dict):
        try:
            proposals = task_schema.normalize_issue_proposals(json_payload, loose_issue_codes=True)
        except Exception:
            proposals = None
        if proposals is not None:
            return task_schema.issue_list_payload(proposals), "json"
    compact_payload, method = parse_proposal_answer_detailed(text)
    if compact_payload is None:
        return None, "unparsed"
    try:
        proposals = task_schema.normalize_issue_proposals(compact_payload, loose_issue_codes=True)
    except Exception:
        return None, "unparsed"
    return task_schema.issue_list_payload(proposals), method


def parse_issue_list_answer(answer_text: Any) -> Optional[dict[str, Any]]:
    payload, _method = parse_issue_list_answer_detailed(answer_text)
    return payload


def parse_finding_answer(
    answer_text: Any,
    *,
    default_issue_code: str = "",
    default_title: str = "",
) -> Optional[dict[str, Any]]:
    text = str(answer_text or "").strip()
    if not text:
        return None
    line = next((str(item).strip() for item in text.splitlines() if str(item).strip()), "")
    if not line:
        return None
    parts = [part.strip() for part in line.split("|")]
    if len(parts) < 6:
        return None
    issue_token = parts[0] or default_issue_code
    try:
        issue_code = ontology.normalize_issue_code(issue_token)
    except ValueError:
        if not default_issue_code:
            return None
        issue_code = ontology.normalize_issue_code(default_issue_code)
    title = parts[1] or default_title or ontology.get_issue(issue_code).title
    severity = str(parts[2] or "unknown").strip().lower() or "unknown"
    insufficient = parse_bool_token(parts[3], default=False)
    evidence_text = str(parts[4] or "").strip()
    action_text = str("|".join(parts[5:]) or "").strip()
    return {
        "finding": {
            "issue_code": issue_code,
            "title": title,
            "evidence": [evidence_text] if evidence_text else [],
            "severity": severity,
            "recommended_action": action_text,
            "insufficient_evidence": insufficient,
        }
    }
