from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Optional

from inspector_md.common import normalize_text, tokenize_text


@dataclass(frozen=True)
class IssueOntologyRecord:
    issue_code: str
    title: str
    category: str
    evaluation_family: str
    aliases: tuple[str, ...]
    detect_labels: tuple[str, ...]
    compliance_tags: tuple[str, ...]
    default_cost_band: str
    default_recommended_action: str


ISSUE_CATALOG: tuple[IssueOntologyRecord, ...] = (
    IssueOntologyRecord(
        issue_code="crack_defect",
        title="Crack Defect",
        category="surface",
        evaluation_family="crack",
        aliases=("crack", "surface crack", "concrete crack", "structural crack"),
        detect_labels=("surface crack", "concrete crack"),
        compliance_tags=("surface_stability", "water_intrusion_risk"),
        default_cost_band="medium",
        default_recommended_action="Inspect the crack pattern and repair the cracked surface where appropriate.",
    ),
    IssueOntologyRecord(
        issue_code="roof_cover_damage",
        title="Roof Cover Damage",
        category="roofing",
        evaluation_family="roof",
        aliases=("roof damage", "missing shingles", "roof puncture"),
        detect_labels=("missing shingle", "roof puncture", "roof edge lifting"),
        compliance_tags=("water_intrusion_risk",),
        default_cost_band="high",
        default_recommended_action="Inspect the roof covering and schedule repair of damaged roof areas.",
    ),
    IssueOntologyRecord(
        issue_code="gutter_downspout_damage",
        title="Gutter or Downspout Damage",
        category="drainage",
        evaluation_family="drainage",
        aliases=("gutter damage", "downspout damage", "detached gutter"),
        detect_labels=("detached gutter", "damaged downspout"),
        compliance_tags=("drainage_maintenance", "water_intrusion_risk"),
        default_cost_band="medium",
        default_recommended_action="Repair or resecure the drainage components and confirm proper runoff.",
    ),
    IssueOntologyRecord(
        issue_code="window_glazing_damage",
        title="Window or Glazing Damage",
        category="fenestration",
        evaluation_family="window",
        aliases=("broken window", "cracked window", "damaged glazing"),
        detect_labels=("broken window pane", "cracked glazing"),
        compliance_tags=("water_intrusion_risk", "egress_access"),
        default_cost_band="medium",
        default_recommended_action="Replace damaged glazing and inspect the surrounding frame and seals.",
    ),
    IssueOntologyRecord(
        issue_code="siding_facade_crack",
        title="Siding or Facade Crack",
        category="facade",
        evaluation_family="crack",
        aliases=("facade crack", "siding crack", "exterior wall crack"),
        detect_labels=("siding crack", "facade crack"),
        compliance_tags=("surface_stability", "water_intrusion_risk"),
        default_cost_band="medium",
        default_recommended_action="Inspect facade materials and repair cracked exterior surfaces.",
    ),
    IssueOntologyRecord(
        issue_code="surface_spalling",
        title="Surface Spalling",
        category="masonry",
        evaluation_family="surface_loss",
        aliases=("spalling", "spallation", "concrete spall", "masonry spall"),
        detect_labels=("concrete spalling", "masonry spalling", "surface spalling"),
        compliance_tags=("surface_stability",),
        default_cost_band="high",
        default_recommended_action="Inspect the affected masonry or concrete and repair loose or failing material.",
    ),
    IssueOntologyRecord(
        issue_code="water_leakage",
        title="Water Leakage",
        category="moisture",
        evaluation_family="moisture",
        aliases=("leakage", "water leakage", "active leak"),
        detect_labels=("water leakage", "active leakage trace"),
        compliance_tags=("water_intrusion_risk",),
        default_cost_band="high",
        default_recommended_action="Identify the source of leakage and repair the affected assembly after drying and inspection.",
    ),
    IssueOntologyRecord(
        issue_code="moisture_intrusion",
        title="Moisture Intrusion",
        category="moisture",
        evaluation_family="moisture",
        aliases=("moisture", "moisture intrusion", "dampness", "damp surface"),
        detect_labels=("moisture intrusion", "damp surface"),
        compliance_tags=("water_intrusion_risk",),
        default_cost_band="medium",
        default_recommended_action="Investigate the moisture source and repair affected materials after drying.",
    ),
    IssueOntologyRecord(
        issue_code="water_staining",
        title="Water Staining",
        category="moisture",
        evaluation_family="moisture",
        aliases=("water stain", "moisture stain", "water intrusion staining"),
        detect_labels=("water stain", "moisture stain"),
        compliance_tags=("water_intrusion_risk",),
        default_cost_band="medium",
        default_recommended_action="Investigate the moisture source and repair the affected area after drying.",
    ),
    IssueOntologyRecord(
        issue_code="corrosion_rust",
        title="Corrosion or Rust",
        category="metal",
        evaluation_family="corrosion",
        aliases=("rust", "corrosion", "rusted metal", "corrosion stain", "corrosionstain"),
        detect_labels=("rusted metal", "corroded railing", "corrosion stain"),
        compliance_tags=("surface_stability",),
        default_cost_band="medium",
        default_recommended_action="Remove corrosion where feasible and repair or replace compromised metal components.",
    ),
    IssueOntologyRecord(
        issue_code="efflorescence_deposit",
        title="Efflorescence Deposit",
        category="moisture",
        evaluation_family="moisture",
        aliases=("efflorescence", "efflorescence deposit"),
        detect_labels=("efflorescence deposit", "white mineral deposit"),
        compliance_tags=("water_intrusion_risk",),
        default_cost_band="low",
        default_recommended_action="Investigate the moisture path and clean or repair the affected surface after addressing the source.",
    ),
    IssueOntologyRecord(
        issue_code="exposed_rebar",
        title="Exposed Rebar",
        category="structure",
        evaluation_family="surface_loss",
        aliases=("exposed bars", "exposed rebar", "exposed reinforcement"),
        detect_labels=("exposed rebar", "exposed reinforcement"),
        compliance_tags=("surface_stability",),
        default_cost_band="high",
        default_recommended_action="Inspect the exposed reinforcement and repair surrounding concrete or protective cover.",
    ),
    IssueOntologyRecord(
        issue_code="material_abscission",
        title="Material Abscission",
        category="surface",
        evaluation_family="surface_loss",
        aliases=("abscission", "material abscission", "surface delamination"),
        detect_labels=("material abscission", "surface delamination"),
        compliance_tags=("surface_stability",),
        default_cost_band="high",
        default_recommended_action="Inspect the affected material and repair or replace delaminated surface sections.",
    ),
    IssueOntologyRecord(
        issue_code="surface_bulge",
        title="Surface Bulge",
        category="surface",
        evaluation_family="surface_deformation",
        aliases=("bulge", "surface bulge", "wall bulge"),
        detect_labels=("surface bulge", "wall bulge"),
        compliance_tags=("surface_stability",),
        default_cost_band="high",
        default_recommended_action="Inspect the bulged surface for underlying instability and repair the affected assembly.",
    ),
    IssueOntologyRecord(
        issue_code="foundation_settlement_sign",
        title="Foundation Settlement Sign",
        category="structure",
        evaluation_family="crack",
        aliases=("foundation crack", "settlement crack", "settlement sign"),
        detect_labels=("foundation crack", "settlement crack"),
        compliance_tags=("surface_stability",),
        default_cost_band="very_high",
        default_recommended_action="Arrange further structural evaluation for visible settlement indicators.",
    ),
    IssueOntologyRecord(
        issue_code="trip_hazard_obstruction",
        title="Trip Hazard or Walkway Obstruction",
        category="site",
        evaluation_family="site_access",
        aliases=("trip hazard", "walkway obstruction", "uneven walking surface"),
        detect_labels=("trip hazard", "walkway obstruction"),
        compliance_tags=("egress_access", "site_housekeeping"),
        default_cost_band="low",
        default_recommended_action="Remove the obstruction or repair the walking surface to restore safe access.",
    ),
    IssueOntologyRecord(
        issue_code="debris_accumulation",
        title="Debris Accumulation",
        category="site",
        evaluation_family="site_access",
        aliases=("debris pile", "site debris", "debris buildup"),
        detect_labels=("debris pile", "construction debris"),
        compliance_tags=("site_housekeeping",),
        default_cost_band="low",
        default_recommended_action="Remove accumulated debris and restore housekeeping in the affected area.",
    ),
    IssueOntologyRecord(
        issue_code="guardrail_handrail_missing_or_damaged",
        title="Guardrail or Handrail Missing or Damaged",
        category="life_safety",
        evaluation_family="guardrail",
        aliases=("missing handrail", "damaged guardrail", "broken handrail"),
        detect_labels=("missing handrail", "damaged guardrail"),
        compliance_tags=("fall_protection", "egress_access"),
        default_cost_band="high",
        default_recommended_action="Repair or replace the damaged guard or handrail and verify secure attachment.",
    ),
    IssueOntologyRecord(
        issue_code="blocked_egress_or_access",
        title="Blocked Egress or Access",
        category="life_safety",
        evaluation_family="site_access",
        aliases=("blocked exit", "blocked access", "blocked path of egress"),
        detect_labels=("blocked exit path", "blocked access route"),
        compliance_tags=("egress_access",),
        default_cost_band="medium",
        default_recommended_action="Clear the blocked route and maintain unobstructed access or egress.",
    ),
)

ISSUE_BY_CODE = {record.issue_code: record for record in ISSUE_CATALOG}
_ALIASES: dict[str, str] = {}
_LOOSE_IGNORED_TOKENS = frozenset(
    {
        "visible",
        "definition",
        "defined",
        "issue",
        "issues",
        "problem",
        "problems",
        "finding",
        "findings",
        "label",
        "labels",
        "type",
        "category",
        "annotated",
        "annotation",
        "area",
        "defect",
    }
)
_LOOSE_MANUAL_ALIASES = {
    "material abef": "material_abscission",
    "exposed reid": "exposed_rebar",
    "subrosion": "corrosion_rust",
}
for record in ISSUE_CATALOG:
    _ALIASES[normalize_text(record.issue_code)] = record.issue_code
    _ALIASES[normalize_text(record.title)] = record.issue_code
    for alias in record.aliases:
        _ALIASES[normalize_text(alias)] = record.issue_code


def _candidate_phrase_texts(record: IssueOntologyRecord) -> list[str]:
    candidates = [
        record.issue_code.replace("_", " "),
        record.title,
        *record.aliases,
        *record.detect_labels,
    ]
    seen: set[str] = set()
    phrases: list[str] = []
    for item in candidates:
        normalized = normalize_text(item)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        phrases.append(normalized)
    return phrases


_LOOSE_CANDIDATES: tuple[tuple[str, str, tuple[str, ...]], ...] = tuple(
    (phrase, record.issue_code, tuple(tokenize_text(phrase)))
    for record in ISSUE_CATALOG
    for phrase in _candidate_phrase_texts(record)
)


def _try_exact_normalize_issue_code(value: Any) -> Optional[str]:
    key = normalize_text(value)
    if not key:
        return None
    normalized = _ALIASES.get(key, key.replace(" ", "_"))
    if normalized in ISSUE_BY_CODE:
        return normalized
    return None


def _squash_duplicate_tokens(tokens: list[str]) -> list[str]:
    output: list[str] = []
    for token in tokens:
        if output and output[-1] == token:
            continue
        output.append(token)
    return output


def _loose_issue_code_variants(tokens: list[str]) -> list[str]:
    variants: list[str] = []

    def _add_variant(parts: list[str]) -> None:
        compact = [str(part).strip().lower() for part in parts if str(part).strip()]
        if not compact:
            return
        for candidate in (" ".join(compact), "_".join(compact)):
            if candidate and candidate not in variants:
                variants.append(candidate)

    deduped = _squash_duplicate_tokens(tokens)
    filtered = [token for token in deduped if token not in _LOOSE_IGNORED_TOKENS]
    _add_variant(tokens)
    _add_variant(deduped)
    _add_variant(filtered)
    if filtered:
        _add_variant(sorted(set(filtered), key=filtered.index))
    return variants


def _normalize_issue_code_loose(value: Any) -> Optional[str]:
    raw_text = str(value or "").strip()
    if not raw_text:
        return None
    raw_key = normalize_text(raw_text)
    if raw_key in _LOOSE_MANUAL_ALIASES:
        return _LOOSE_MANUAL_ALIASES[raw_key]
    tokens = tokenize_text(raw_text)
    if not tokens:
        return None
    for candidate in _loose_issue_code_variants(tokens):
        exact = _try_exact_normalize_issue_code(candidate)
        if exact is not None:
            return exact
        manual = _LOOSE_MANUAL_ALIASES.get(candidate)
        if manual is not None:
            return manual

    normalized_tokens = [
        token for token in _squash_duplicate_tokens(tokens)
        if token not in _LOOSE_IGNORED_TOKENS
    ] or _squash_duplicate_tokens(tokens)
    input_phrase = " ".join(normalized_tokens)
    if not input_phrase:
        return None
    input_token_set = set(normalized_tokens)
    best_by_code: dict[str, float] = {}
    for phrase, issue_code, phrase_tokens in _LOOSE_CANDIDATES:
        phrase_token_set = set(phrase_tokens)
        shared_tokens = input_token_set & phrase_token_set
        ratio = SequenceMatcher(None, input_phrase, phrase).ratio()
        if len(normalized_tokens) == 1:
            if len(phrase_tokens) != 1 and ratio < 0.9:
                continue
            score = ratio
        else:
            if not shared_tokens and ratio < 0.88:
                continue
            score = ratio + (0.04 * min(2, len(shared_tokens)))
        if score > best_by_code.get(issue_code, 0.0):
            best_by_code[issue_code] = score
    if not best_by_code:
        return None
    ranked = sorted(best_by_code.items(), key=lambda item: item[1], reverse=True)
    best_issue_code, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    min_score = 0.78 if len(normalized_tokens) == 1 else 0.76
    min_margin = 0.06 if len(normalized_tokens) == 1 else 0.05
    if best_score >= min_score and (best_score - second_score) >= min_margin:
        return best_issue_code
    return None


def all_issue_codes() -> list[str]:
    return [record.issue_code for record in ISSUE_CATALOG]


def normalize_issue_code(value: Any, *, allow_unknown: bool = False, loose: bool = False) -> str:
    normalized = _try_exact_normalize_issue_code(value)
    if normalized is None and bool(loose):
        normalized = _normalize_issue_code_loose(value)
    if normalized is not None:
        return normalized
    key = normalize_text(value)
    normalized = key.replace(" ", "_")
    if allow_unknown:
        return normalized
    raise ValueError(f"Unknown issue_code: {value!r}")


def get_issue(value: Any) -> IssueOntologyRecord:
    return ISSUE_BY_CODE[normalize_issue_code(value)]


def issue_match_score(expected: Any, predicted: Any) -> float:
    expected_issue = get_issue(expected)
    predicted_issue = get_issue(predicted)
    if expected_issue.issue_code == predicted_issue.issue_code:
        return 1.0
    if expected_issue.evaluation_family == predicted_issue.evaluation_family:
        return 0.5
    return 0.0


def issues_match(expected: Any, predicted: Any) -> bool:
    return issue_match_score(expected, predicted) > 0.0


def detect_labels_for_issue(value: Any) -> tuple[str, ...]:
    return get_issue(value).detect_labels


def detect_class_catalog() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for record in ISSUE_CATALOG:
        for label in record.detect_labels:
            class_name = str(label).strip()
            if not class_name or class_name in seen:
                continue
            seen.add(class_name)
            class_uid = "inspector_md:" + class_name.lower().replace("/", "_").replace(" ", "_")
            rows.append(
                {
                    "class_uid": class_uid,
                    "class_name": class_name,
                    "prompt": class_name,
                }
            )
    return rows


def issue_code_for_detect_label(label: str) -> Optional[str]:
    normalized = normalize_text(label)
    for record in ISSUE_CATALOG:
        for detect_label in record.detect_labels:
            if normalize_text(detect_label) == normalized:
                return record.issue_code
    return None


def is_generic_detect_label(label: str) -> bool:
    normalized = normalize_text(label)
    return normalized in {"damage", "defect", "issue", "problem"}
