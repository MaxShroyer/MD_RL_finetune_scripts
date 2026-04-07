"""Canonical task and normalization helpers for PokemonCards."""

from __future__ import annotations

import json
import re
from typing import Any, Optional

CANONICAL_TASK_TYPES: tuple[str, ...] = (
    "card_identity",
    "card_core",
    "attack_overview",
    "card_summary",
)

TASK_TYPE_SET = set(CANONICAL_TASK_TYPES)
TASK_TYPE_ALIASES: dict[str, str] = {}

CARD_IDENTITY_KEYS: tuple[str, ...] = ("name", "hp", "set_name")
CARD_CORE_KEYS: tuple[str, ...] = (
    "name",
    "hp",
    "set_name",
    "stage",
    "pokemon_types",
    "rarity",
    "evolves_from",
)
ATTACK_OVERVIEW_KEYS: tuple[str, ...] = ("attack_names", "attack_count")
POKEMON_TYPE_VOCAB: tuple[str, ...] = (
    "Grass",
    "Fire",
    "Water",
    "Lightning",
    "Psychic",
    "Fighting",
    "Darkness",
    "Metal",
    "Dragon",
    "Fairy",
    "Colorless",
)

RATIONALE_KEYS_BY_TASK: dict[str, tuple[str, ...]] = {
    "card_identity": ("name", "hp", "set"),
    "card_core": ("name", "hp", "set", "stage", "types", "rarity", "evolves_from"),
    "attack_overview": ("name", "attacks", "attack_count"),
    "card_summary": ("name", "hp", "set", "stage", "types", "attacks"),
}


def normalize_task_type(task_type: str, *, allow_unknown: bool = False) -> str:
    normalized = TASK_TYPE_ALIASES.get(str(task_type or "").strip(), str(task_type or "").strip())
    if normalized in TASK_TYPE_SET:
        return normalized
    if allow_unknown:
        return normalized
    raise ValueError(f"unknown task_type: {task_type!r}")


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def normalize_text_lower(value: Any) -> str:
    return normalize_text(value).lower()


def _normalize_nullable_text(value: Any) -> Optional[str]:
    text = normalize_text(value)
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"none", "null", "n/a", "unknown"}:
        return None
    return text


def _normalize_hp(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    text = normalize_text(value)
    if not text:
        return None
    match = re.search(r"\d+", text)
    if not match:
        return None
    return int(match.group(0))


def _split_csvish_items(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = re.split(r"\s*,\s*|\s*/\s*|\s+\|\s+|\s+and\s+", value.strip())
    elif isinstance(value, (list, tuple, set)):
        raw_items = [str(item) for item in value]
    else:
        raw_items = [str(value)]
    out: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        text = normalize_text(item)
        if not text:
            continue
        lowered = text.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        out.append(text)
    return out


def _normalize_string_list(value: Any) -> list[str]:
    items = _split_csvish_items(value)
    items.sort(key=lambda item: item.lower())
    return items


def _normalize_attack_names(value: Any) -> list[str]:
    items = _split_csvish_items(value)
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        cleaned = item.strip('"')
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
    return out


def normalize_summary_text(value: Any) -> str:
    return normalize_text(value)


def _join_with_and(items: list[str]) -> str:
    cleaned = [normalize_text(item) for item in items if normalize_text(item)]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f"{cleaned[0]} and {cleaned[1]}"
    return f"{', '.join(cleaned[:-1])}, and {cleaned[-1]}"


def summary_from_answer(answer: dict[str, Any]) -> str:
    name = normalize_text(answer.get("name"))
    hp = _normalize_hp(answer.get("hp"))
    set_name = normalize_text(answer.get("set_name"))
    stage = _normalize_nullable_text(answer.get("stage"))
    types = _normalize_string_list(answer.get("pokemon_types"))
    attacks = _normalize_attack_names(answer.get("attack_names"))
    subject = name or "This card"
    stage_prefix = f"{stage} " if stage else ""
    hp_text = f"{hp} HP" if hp is not None else "unknown HP"
    set_text = set_name or "an unknown set"
    first_sentence = f"{subject} is a {stage_prefix}Pokemon card from {set_text} with {hp_text}."

    detail_parts: list[str] = []
    if types:
        type_phrase = _join_with_and(types)
        if type_phrase:
            detail_parts.append(f"its type is {type_phrase}")
    if attacks:
        attack_phrase = _join_with_and(attacks)
        if attack_phrase:
            attack_label = "attack" if len(attacks) == 1 else "attacks"
            detail_parts.append(f"it has the {attack_label} {attack_phrase}")

    if not detail_parts:
        return first_sentence
    detail_sentence = " and ".join(detail_parts)
    detail_sentence = detail_sentence[:1].upper() + detail_sentence[1:]
    return f"{first_sentence} {detail_sentence}."


def normalize_answer_for_task(task_type: str, payload: Any) -> Optional[dict[str, Any]]:
    canonical_task = normalize_task_type(task_type)
    if not isinstance(payload, dict):
        return None

    if canonical_task == "card_identity":
        name = normalize_text(payload.get("name"))
        hp = _normalize_hp(payload.get("hp"))
        set_name = normalize_text(payload.get("set_name"))
        if not name or hp is None or not set_name:
            return None
        return {"name": name, "hp": hp, "set_name": set_name}

    if canonical_task == "card_core":
        name = normalize_text(payload.get("name"))
        hp = _normalize_hp(payload.get("hp"))
        set_name = normalize_text(payload.get("set_name"))
        if not name or hp is None or not set_name:
            return None
        return {
            "name": name,
            "hp": hp,
            "set_name": set_name,
            "stage": _normalize_nullable_text(payload.get("stage")),
            "pokemon_types": _normalize_string_list(payload.get("pokemon_types")),
            "rarity": _normalize_nullable_text(payload.get("rarity")),
            "evolves_from": _normalize_nullable_text(payload.get("evolves_from")),
        }

    if canonical_task == "attack_overview":
        attack_names = _normalize_attack_names(payload.get("attack_names"))
        attack_count = payload.get("attack_count")
        if attack_count is None:
            attack_count = len(attack_names)
        hp_count = _normalize_hp(attack_count)
        if hp_count is None:
            return None
        return {"attack_names": attack_names, "attack_count": hp_count}

    if canonical_task == "card_summary":
        if "summary" in payload:
            summary = normalize_summary_text(payload.get("summary"))
            if not summary:
                return None
            return {"summary": summary}
        core = normalize_answer_for_task(
            "card_core",
            {
                "name": payload.get("name"),
                "hp": payload.get("hp"),
                "set_name": payload.get("set_name"),
                "stage": payload.get("stage"),
                "pokemon_types": payload.get("pokemon_types"),
                "rarity": payload.get("rarity"),
                "evolves_from": payload.get("evolves_from"),
            },
        )
        if core is None:
            return None
        attack_names = _normalize_attack_names(payload.get("attack_names"))
        summary = summary_from_answer({**core, "attack_names": attack_names})
        return {"summary": summary}

    return None


def rationale_slots_from_answer(task_type: str, answer: dict[str, Any]) -> dict[str, str]:
    canonical_task = normalize_task_type(task_type)
    slots: dict[str, str] = {}
    if canonical_task == "card_summary" and isinstance(answer, dict):
        core = normalize_answer_for_task(
            "card_core",
            {
                "name": answer.get("name"),
                "hp": answer.get("hp"),
                "set_name": answer.get("set_name"),
                "stage": answer.get("stage"),
                "pokemon_types": answer.get("pokemon_types"),
                "rarity": answer.get("rarity"),
                "evolves_from": answer.get("evolves_from"),
            },
        )
        if core is not None:
            slots["name"] = normalize_text_lower(core.get("name"))
            slots["hp"] = str(core.get("hp"))
            slots["set"] = normalize_text_lower(core.get("set_name"))
            slots["stage"] = normalize_text_lower(core.get("stage") or "none")
            types = core.get("pokemon_types") or []
            slots["types"] = ",".join(item.lower() for item in types) if types else "none"
            attacks = _normalize_attack_names(answer.get("attack_names"))
            slots["attacks"] = ",".join(item.lower() for item in attacks) if attacks else "none"
            return {key: slots[key] for key in RATIONALE_KEYS_BY_TASK[canonical_task] if key in slots}

    normalized = normalize_answer_for_task(canonical_task, answer)
    if normalized is None:
        return {}

    if canonical_task in {"card_identity", "card_core"}:
        slots["name"] = normalize_text_lower(normalized.get("name"))
        slots["hp"] = str(normalized.get("hp"))
        slots["set"] = normalize_text_lower(normalized.get("set_name"))
    if canonical_task == "card_core":
        slots["stage"] = normalize_text_lower(normalized.get("stage") or "none")
        types = normalized.get("pokemon_types") or []
        slots["types"] = ",".join(item.lower() for item in types) if types else "none"
    if canonical_task == "card_core":
        slots["rarity"] = normalize_text_lower(normalized.get("rarity") or "none")
        slots["evolves_from"] = normalize_text_lower(normalized.get("evolves_from") or "none")
    if canonical_task == "attack_overview":
        slots["attacks"] = ",".join(item.lower() for item in normalized.get("attack_names", [])) or "none"
        slots["attack_count"] = str(normalized.get("attack_count", 0))
    if canonical_task == "card_summary":
        summary_support = summary_support_from_payload(normalized)
        if summary_support is not None:
            slots["name"] = normalize_text_lower(summary_support.get("name"))
            slots["hp"] = str(summary_support.get("hp"))
            slots["set"] = normalize_text_lower(summary_support.get("set_name"))
            slots["stage"] = normalize_text_lower(summary_support.get("stage") or "none")
            types = summary_support.get("pokemon_types") or []
            attacks = summary_support.get("attack_names") or []
            slots["types"] = ",".join(item.lower() for item in types) if types else "none"
            slots["attacks"] = ",".join(item.lower() for item in attacks) if attacks else "none"
    return {key: slots[key] for key in RATIONALE_KEYS_BY_TASK[canonical_task] if key in slots}


def rationale_text_from_answer(task_type: str, answer: dict[str, Any]) -> str:
    slots = rationale_slots_from_answer(task_type, answer)
    return "; ".join(f"{key}={value}" for key, value in slots.items())


def _normalize_rationale_slot_value(key: str, value: str) -> str:
    text = normalize_text_lower(value)
    if not text:
        return "none"
    if key in {"types", "attacks"}:
        parts = [part for part in _split_csvish_items(text) if part]
        lowered = sorted({part.lower() for part in parts})
        return ",".join(lowered) if lowered else "none"
    if key in {"hp", "attack_count"}:
        parsed = _normalize_hp(text)
        return str(parsed) if parsed is not None else "0"
    return text


def parse_rationale_text(task_type: str, text: str) -> dict[str, str]:
    canonical_task = normalize_task_type(task_type)
    expected_keys = set(RATIONALE_KEYS_BY_TASK[canonical_task])
    out: dict[str, str] = {}

    for raw_part in str(text or "").split(";"):
        part = raw_part.strip()
        if not part or "=" not in part:
            continue
        key, _, raw_value = part.partition("=")
        normalized_key = normalize_text_lower(key).replace("set_name", "set")
        if normalized_key not in expected_keys:
            continue
        out[normalized_key] = _normalize_rationale_slot_value(normalized_key, raw_value)

    return out


def rationale_text_to_json(task_type: str, text: str) -> str:
    parsed = parse_rationale_text(task_type, text)
    ordered = {key: parsed.get(key, "none") for key in RATIONALE_KEYS_BY_TASK[normalize_task_type(task_type)]}
    return json.dumps(ordered, sort_keys=True)


def canonicalize_rationale_text(task_type: str, text: str) -> str:
    canonical_task = normalize_task_type(task_type)
    parsed = parse_rationale_text(canonical_task, text)
    ordered_keys = RATIONALE_KEYS_BY_TASK[canonical_task]
    return "; ".join(f"{key}={parsed.get(key, 'none')}" for key in ordered_keys)


def summary_support_from_payload(payload: Any) -> Optional[dict[str, Any]]:
    if not isinstance(payload, dict):
        return None

    core = normalize_answer_for_task(
        "card_core",
        {
            "name": payload.get("name"),
            "hp": payload.get("hp"),
            "set_name": payload.get("set_name"),
            "stage": payload.get("stage"),
            "pokemon_types": payload.get("pokemon_types"),
            "rarity": payload.get("rarity"),
            "evolves_from": payload.get("evolves_from"),
        },
    )
    if core is not None:
        return {
            "name": core.get("name"),
            "hp": core.get("hp"),
            "set_name": core.get("set_name"),
            "stage": core.get("stage"),
            "pokemon_types": list(core.get("pokemon_types", [])),
            "attack_names": _normalize_attack_names(payload.get("attack_names")),
        }

    summary = normalize_summary_text(payload.get("summary"))
    if not summary:
        return None

    parsed = parse_rationale_text("card_summary", summary)
    if not parsed:
        return None
    hp = _normalize_hp(parsed.get("hp"))
    if hp is None:
        return None
    set_name = parsed.get("set", "")
    name = parsed.get("name", "")
    if not name or not set_name:
        return None
    stage = parsed.get("stage", "none")
    types = parsed.get("types", "none")
    attacks = parsed.get("attacks", "none")
    return {
        "name": name,
        "hp": hp,
        "set_name": set_name,
        "stage": None if stage in {"", "none"} else stage,
        "pokemon_types": [] if types in {"", "none"} else [item for item in types.split(",") if item],
        "attack_names": [] if attacks in {"", "none"} else [item for item in attacks.split(",") if item],
    }
