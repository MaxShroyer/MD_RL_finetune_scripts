from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from inspector_md import ontology, openrouter_grader, rule_stubs, task_schema
from inspector_md.moondream_client import MoondreamInspectorClient


@dataclass(frozen=True)
class PipelineModels:
    detect_model: str
    query_model: str
    point_model: str = ""


@dataclass(frozen=True)
class PipelineSettings:
    proposal_temperature: float = 0.0
    proposal_top_p: float = 1.0
    proposal_max_tokens: int = 512
    finding_temperature: float = 0.0
    finding_top_p: float = 1.0
    finding_max_tokens: int = 384
    detect_temperature: float = 0.0
    detect_top_p: float = 1.0
    detect_max_tokens: int = 256
    detect_max_objects: int = 24
    iou_merge_threshold: float = 0.5
    reasoning: bool = False
    proposal_prompt_style: str = "request_only"
    finding_prompt_style: str = "minimal"


def box_iou(box_a: task_schema.Box, box_b: task_schema.Box) -> float:
    inter_x_min = max(box_a.x_min, box_b.x_min)
    inter_y_min = max(box_a.y_min, box_b.y_min)
    inter_x_max = min(box_a.x_max, box_b.x_max)
    inter_y_max = min(box_a.y_max, box_b.y_max)
    inter_w = max(0.0, inter_x_max - inter_x_min)
    inter_h = max(0.0, inter_y_max - inter_y_min)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0
    area_a = (box_a.x_max - box_a.x_min) * (box_a.y_max - box_a.y_min)
    area_b = (box_b.x_max - box_b.x_min) * (box_b.y_max - box_b.y_min)
    union = max(1e-8, area_a + area_b - inter_area)
    return inter_area / union


def _merge_box_cluster(cluster: list[task_schema.LocalizedIssue]) -> task_schema.LocalizedIssue:
    issue_code = cluster[0].issue_code
    x_min = min(item.box.x_min for item in cluster)
    y_min = min(item.box.y_min for item in cluster)
    x_max = max(item.box.x_max for item in cluster)
    y_max = max(item.box.y_max for item in cluster)
    evidence: list[str] = []
    seen_evidence: set[str] = set()
    labels: list[str] = []
    seen_labels: set[str] = set()
    for item in cluster:
        for text in item.evidence:
            if text and text not in seen_evidence:
                seen_evidence.add(text)
                evidence.append(text)
        for label in item.source_detect_labels:
            if label and label not in seen_labels:
                seen_labels.add(label)
                labels.append(label)
    return task_schema.LocalizedIssue(
        issue_code=issue_code,
        box=task_schema.Box(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max),
        evidence=evidence,
        source_detect_labels=labels,
    )


def merge_localized_issues(
    issues: list[task_schema.LocalizedIssue],
    *,
    iou_threshold: float,
) -> list[task_schema.LocalizedIssue]:
    merged: list[task_schema.LocalizedIssue] = []
    pending = list(issues)
    while pending:
        head = pending.pop(0)
        cluster = [head]
        rest: list[task_schema.LocalizedIssue] = []
        for item in pending:
            if item.issue_code == head.issue_code and box_iou(item.box, head.box) >= float(iou_threshold):
                cluster.append(item)
            else:
                rest.append(item)
        merged.append(_merge_box_cluster(cluster))
        pending = rest
    return merged


def finding_to_text(finding: task_schema.Finding, *, index: int) -> str:
    evidence = "; ".join(finding.evidence) if finding.evidence else "No concise evidence string was returned."
    compliance_line = finding.compliance_note or "No possible compliance flag from rule stubs."
    return "\n".join(
        [
            f"{index}. {finding.title}",
            f"Issue code: {finding.issue_code}",
            f"Cost band: {finding.cost_band}",
            f"Evidence: {evidence}",
            f"Recommended action: {finding.recommended_action}",
            f"Possible compliance issue: {'yes' if finding.possible_compliance_issue else 'no'}",
            f"Insufficient evidence: {'yes' if finding.insufficient_evidence else 'no'}",
            f"Compliance note: {compliance_line}",
        ]
    )


def report_to_punch_list_text(report: task_schema.InspectionReport) -> str:
    header = [
        "Inspector MD Punch List",
        f"Image: {report.request.image_path}",
        f"Request: {report.request.inspection_request}",
    ]
    if report.request.asset_context:
        header.append(f"Asset context: {report.request.asset_context}")
    if not report.findings:
        return "\n".join(header + ["", "No visible punch-list findings identified from the supplied image."])
    sections = header + [""]
    for index, finding in enumerate(report.findings, start=1):
        sections.append(finding_to_text(finding, index=index))
        sections.append("")
    return "\n".join(sections).rstrip() + "\n"


class InspectorPipeline:
    def __init__(
        self,
        *,
        client: MoondreamInspectorClient,
        models: PipelineModels,
        settings: Optional[PipelineSettings] = None,
        normalizer: Optional[openrouter_grader.OpenRouterGrader] = None,
    ) -> None:
        self.client = client
        self.models = models
        self.settings = settings or PipelineSettings()
        self.normalizer = normalizer

    def _localize_proposals(self, request: task_schema.InspectionRequest, proposals: list[task_schema.IssueProposal]) -> list[task_schema.LocalizedIssue]:
        localized: list[task_schema.LocalizedIssue] = []
        for proposal in proposals:
            for detect_label in ontology.detect_labels_for_issue(proposal.issue_code):
                boxes = self.client.detect_boxes(
                    model=self.models.detect_model,
                    image_path=request.image_path,
                    detect_label=detect_label,
                    temperature=self.settings.detect_temperature,
                    top_p=self.settings.detect_top_p,
                    max_tokens=self.settings.detect_max_tokens,
                    max_objects=self.settings.detect_max_objects,
                )
                for box in boxes:
                    localized.append(
                        task_schema.LocalizedIssue(
                            issue_code=proposal.issue_code,
                            box=task_schema.Box.from_payload(
                                {
                                    "x_min": box.x_min,
                                    "y_min": box.y_min,
                                    "x_max": box.x_max,
                                    "y_max": box.y_max,
                                }
                            ),
                            evidence=[proposal.evidence] if proposal.evidence else [],
                            source_detect_labels=[detect_label],
                        )
                    )
        return merge_localized_issues(localized, iou_threshold=self.settings.iou_merge_threshold)

    def _build_finding(self, *, request: task_schema.InspectionRequest, localized_issue: task_schema.LocalizedIssue, index: int) -> task_schema.Finding:
        issue = ontology.get_issue(localized_issue.issue_code)
        evidence = [str(item).strip() for item in list(localized_issue.evidence) if str(item).strip()]
        payload = {
            "finding_id": f"finding_{index:03d}",
            "issue_code": localized_issue.issue_code,
            "title": issue.title,
            "box": localized_issue.box.to_payload(),
            "evidence": evidence,
            "recommended_action": issue.default_recommended_action,
            "cost_band": issue.default_cost_band,
            "possible_compliance_issue": False,
            "insufficient_evidence": False,
            "compliance_note": "",
            "source_detect_labels": list(localized_issue.source_detect_labels),
            "spatial_ref_index": 0,
        }
        finding = task_schema.Finding.from_payload(payload)
        return rule_stubs.apply_rule_stubs(finding)

    def run(self, request_payload: task_schema.InspectionRequest | dict[str, Any]) -> task_schema.InspectionReport:
        request = request_payload if isinstance(request_payload, task_schema.InspectionRequest) else task_schema.InspectionRequest.from_payload(request_payload)
        proposals, proposal_result = self.client.propose_issues(
            model=self.models.query_model,
            image_path=request.image_path,
            inspection_request=request.inspection_request,
            asset_context=request.asset_context,
            prompt_style=self.settings.proposal_prompt_style,
            normalizer=self.normalizer,
            reasoning=self.settings.reasoning,
            temperature=self.settings.proposal_temperature,
            top_p=self.settings.proposal_top_p,
            max_tokens=self.settings.proposal_max_tokens,
        )
        localized = self._localize_proposals(request, proposals)
        findings = [self._build_finding(request=request, localized_issue=item, index=index) for index, item in enumerate(localized, start=1)]
        summary = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "proposal_count": len(proposals),
            "localized_issue_count": len(localized),
            "finding_count": len(findings),
        }
        trace = {
            "proposals": [item.to_payload() for item in proposals],
            "localized_issues": [item.to_payload() for item in localized],
            "proposal_raw_response": proposal_result.raw_response,
            "query_contract": "issues_v2",
        }
        return task_schema.InspectionReport(
            request=request,
            detect_model=self.models.detect_model,
            query_model=self.models.query_model,
            findings=findings,
            summary=summary,
            trace=trace,
        )
