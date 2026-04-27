from __future__ import annotations

from inspector_md import ontology
from inspector_md.task_schema import Finding


_SEVERITY_TO_COST = {
    ("low", "minor"): "low",
    ("low", "moderate"): "medium",
    ("low", "major"): "medium",
    ("medium", "minor"): "medium",
    ("medium", "moderate"): "medium",
    ("medium", "major"): "high",
    ("high", "minor"): "medium",
    ("high", "moderate"): "high",
    ("high", "major"): "very_high",
    ("very_high", "minor"): "high",
    ("very_high", "moderate"): "very_high",
    ("very_high", "major"): "very_high",
    ("unknown", "minor"): "unknown",
    ("unknown", "moderate"): "unknown",
    ("unknown", "major"): "unknown",
    ("unknown", "unknown"): "unknown",
}


def estimate_cost_band(finding: Finding) -> str:
    if finding.insufficient_evidence:
        return "unknown"
    issue = ontology.get_issue(finding.issue_code)
    if str(finding.severity or "").strip().lower() in {"", "unknown"}:
        return issue.default_cost_band
    return _SEVERITY_TO_COST.get((issue.default_cost_band, finding.severity), issue.default_cost_band)


def possible_compliance_flag(finding: Finding) -> bool:
    if finding.insufficient_evidence:
        return False
    issue = ontology.get_issue(finding.issue_code)
    return bool(issue.compliance_tags)


def compliance_note_for_finding(finding: Finding) -> str:
    issue = ontology.get_issue(finding.issue_code)
    if finding.insufficient_evidence:
        return "Insufficient visual evidence for a possible compliance flag. Field verification is needed."
    if not issue.compliance_tags:
        return ""
    tags = ", ".join(issue.compliance_tags)
    return f"Possible compliance issue related to {tags}. This is not a legal determination and needs field review."


def apply_rule_stubs(finding: Finding) -> Finding:
    return Finding(
        finding_id=finding.finding_id,
        issue_code=finding.issue_code,
        title=finding.title,
        box=finding.box,
        evidence=list(finding.evidence),
        severity=finding.severity,
        recommended_action=finding.recommended_action,
        cost_band=estimate_cost_band(finding),
        possible_compliance_issue=possible_compliance_flag(finding),
        insufficient_evidence=finding.insufficient_evidence,
        compliance_note=compliance_note_for_finding(finding),
        source_detect_labels=list(finding.source_detect_labels),
        spatial_ref_index=finding.spatial_ref_index,
    )
