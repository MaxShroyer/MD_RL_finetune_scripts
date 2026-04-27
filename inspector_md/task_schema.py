"""Canonical schema and normalization helpers for Inspector MD."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from inspector_md import ontology
from inspector_md.common import clamp

COST_BANDS = {"low", "medium", "high", "very_high", "unknown"}
SEVERITY_LEVELS = {"minor", "moderate", "major", "unknown"}


def _bool(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return default


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        text = str(value).strip()
        return [text] if text else []
    if not isinstance(value, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


@dataclass(frozen=True)
class Box:
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @classmethod
    def from_payload(cls, payload: Any) -> "Box":
        if isinstance(payload, (list, tuple)) and len(payload) == 4:
            x_min, y_min, x_max, y_max = payload
        elif isinstance(payload, dict):
            x_min = payload.get("x_min")
            y_min = payload.get("y_min")
            x_max = payload.get("x_max")
            y_max = payload.get("y_max")
        else:
            raise ValueError(f"Invalid box payload: {payload!r}")
        try:
            normalized = cls(
                x_min=clamp(float(x_min)),
                y_min=clamp(float(y_min)),
                x_max=clamp(float(x_max)),
                y_max=clamp(float(y_max)),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid box coordinates: {payload!r}") from exc
        if normalized.x_max <= normalized.x_min or normalized.y_max <= normalized.y_min:
            raise ValueError(f"Degenerate box: {payload!r}")
        return normalized

    def to_payload(self) -> dict[str, float]:
        return {
            "x_min": self.x_min,
            "y_min": self.y_min,
            "x_max": self.x_max,
            "y_max": self.y_max,
        }


@dataclass(frozen=True)
class InspectionRequest:
    image_path: str
    inspection_request: str
    asset_context: str = ""

    @classmethod
    def from_payload(cls, payload: Any) -> "InspectionRequest":
        if not isinstance(payload, dict):
            raise ValueError("InspectionRequest must be a JSON object.")
        image_path = str(payload.get("image_path") or "").strip()
        inspection_request = str(payload.get("inspection_request") or "").strip()
        if not image_path:
            raise ValueError("InspectionRequest.image_path is required.")
        if not inspection_request:
            raise ValueError("InspectionRequest.inspection_request is required.")
        return cls(
            image_path=image_path,
            inspection_request=inspection_request,
            asset_context=str(payload.get("asset_context") or "").strip(),
        )

    def to_payload(self) -> dict[str, str]:
        return {
            "image_path": self.image_path,
            "inspection_request": self.inspection_request,
            "asset_context": self.asset_context,
        }


@dataclass(frozen=True)
class IssueProposal:
    issue_code: str
    evidence: str
    confidence: float = 0.5

    @classmethod
    def from_payload(cls, payload: Any, *, loose_issue_code: bool = False) -> "IssueProposal":
        if not isinstance(payload, dict):
            raise ValueError("IssueProposal must be a JSON object.")
        raw_issue_code = payload.get("issue_code", payload.get("type"))
        raw_evidence = payload.get("evidence", payload.get("reasoning"))
        return cls(
            issue_code=ontology.normalize_issue_code(raw_issue_code, loose=bool(loose_issue_code)),
            evidence=str(raw_evidence or "").strip(),
            confidence=clamp(float(payload.get("confidence", 0.5))),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "issue_code": self.issue_code,
            "evidence": self.evidence,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class LocalizedIssue:
    issue_code: str
    box: Box
    evidence: list[str] = field(default_factory=list)
    source_detect_labels: list[str] = field(default_factory=list)

    @classmethod
    def from_payload(cls, payload: Any) -> "LocalizedIssue":
        if not isinstance(payload, dict):
            raise ValueError("LocalizedIssue must be a JSON object.")
        return cls(
            issue_code=ontology.normalize_issue_code(payload.get("issue_code")),
            box=Box.from_payload(payload.get("box")),
            evidence=_string_list(payload.get("evidence")),
            source_detect_labels=_string_list(payload.get("source_detect_labels")),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "issue_code": self.issue_code,
            "box": self.box.to_payload(),
            "evidence": list(self.evidence),
            "source_detect_labels": list(self.source_detect_labels),
        }


@dataclass(frozen=True)
class Finding:
    finding_id: str
    issue_code: str
    title: str
    box: Box
    evidence: list[str]
    severity: str
    recommended_action: str
    cost_band: str
    possible_compliance_issue: bool
    insufficient_evidence: bool
    compliance_note: str
    source_detect_labels: list[str]
    spatial_ref_index: int

    @classmethod
    def from_payload(cls, payload: Any) -> "Finding":
        if not isinstance(payload, dict):
            raise ValueError("Finding must be a JSON object.")
        severity = str(payload.get("severity") or "unknown").strip().lower() or "unknown"
        if severity not in SEVERITY_LEVELS:
            severity = "unknown"
        cost_band = str(payload.get("cost_band") or "unknown").strip().lower() or "unknown"
        if cost_band not in COST_BANDS:
            cost_band = "unknown"
        spatial_ref_index = int(payload.get("spatial_ref_index", 0))
        if spatial_ref_index < 0:
            raise ValueError("spatial_ref_index must be >= 0")
        return cls(
            finding_id=str(payload.get("finding_id") or "").strip(),
            issue_code=ontology.normalize_issue_code(payload.get("issue_code")),
            title=str(payload.get("title") or "").strip(),
            box=Box.from_payload(payload.get("box")),
            evidence=_string_list(payload.get("evidence")),
            severity=severity,
            recommended_action=str(payload.get("recommended_action") or "").strip(),
            cost_band=cost_band,
            possible_compliance_issue=_bool(payload.get("possible_compliance_issue")),
            insufficient_evidence=_bool(payload.get("insufficient_evidence")),
            compliance_note=str(payload.get("compliance_note") or "").strip(),
            source_detect_labels=_string_list(payload.get("source_detect_labels")),
            spatial_ref_index=spatial_ref_index,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "issue_code": self.issue_code,
            "title": self.title,
            "box": self.box.to_payload(),
            "evidence": list(self.evidence),
            "recommended_action": self.recommended_action,
            "cost_band": self.cost_band,
            "possible_compliance_issue": self.possible_compliance_issue,
            "insufficient_evidence": self.insufficient_evidence,
            "compliance_note": self.compliance_note,
            "source_detect_labels": list(self.source_detect_labels),
            "spatial_ref_index": self.spatial_ref_index,
        }


@dataclass(frozen=True)
class InspectionReport:
    request: InspectionRequest
    detect_model: str
    query_model: str
    findings: list[Finding]
    summary: dict[str, Any]
    trace: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Any) -> "InspectionReport":
        if not isinstance(payload, dict):
            raise ValueError("InspectionReport must be a JSON object.")
        request = InspectionRequest.from_payload(payload.get("request"))
        findings = [Finding.from_payload(item) for item in list(payload.get("findings") or [])]
        summary = payload.get("summary")
        if not isinstance(summary, dict):
            summary = {}
        trace = payload.get("trace")
        if not isinstance(trace, dict):
            trace = {}
        return cls(
            request=request,
            detect_model=str(payload.get("detect_model") or "").strip(),
            query_model=str(payload.get("query_model") or "").strip(),
            findings=findings,
            summary=dict(summary),
            trace=dict(trace),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "request": self.request.to_payload(),
            "detect_model": self.detect_model,
            "query_model": self.query_model,
            "findings": [item.to_payload() for item in self.findings],
            "summary": dict(self.summary),
            "trace": dict(self.trace),
        }


def normalize_issue_proposals(value: Any, *, loose_issue_codes: bool = False) -> list[IssueProposal]:
    if isinstance(value, dict):
        if isinstance(value.get("issues"), list):
            value = value.get("issues")
        elif isinstance(value.get("proposals"), list):
            value = value.get("proposals")
    if not isinstance(value, list):
        return []
    proposals: list[IssueProposal] = []
    seen: set[str] = set()
    for item in value:
        try:
            proposal = IssueProposal.from_payload(item, loose_issue_code=bool(loose_issue_codes))
        except ValueError:
            continue
        if proposal.issue_code in seen:
            continue
        seen.add(proposal.issue_code)
        proposals.append(proposal)
    return proposals


def normalize_findings(value: Any) -> list[Finding]:
    if not isinstance(value, list):
        return []
    findings: list[Finding] = []
    for item in value:
        findings.append(Finding.from_payload(item))
    return findings


def issue_list_payload(proposals: list[IssueProposal]) -> dict[str, Any]:
    return {
        "issues": [
            {
                "type": item.issue_code,
                "reasoning": item.evidence,
            }
            for item in proposals
        ]
    }
