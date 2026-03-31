#!/usr/bin/env python3
"""Build a local PokemonCards task dataset from TheFusion21/PokemonCards."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from PIL import Image

try:
    from tqdm.auto import tqdm  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    class _SimpleTqdm:
        def __init__(self, iterable=None, *args: Any, **kwargs: Any) -> None:
            self._iterable = [] if iterable is None else iterable

        def __iter__(self):
            return iter(self._iterable)

        def set_postfix(self, *args: Any, **kwargs: Any) -> None:
            return

    def tqdm(iterable=None, *args, **kwargs):  # type: ignore
        return _SimpleTqdm(iterable, *args, **kwargs)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from DEPICATED_pokemon_cards.common import ensure_parent_dir, write_json
from DEPICATED_pokemon_cards.task_schema import (
    normalize_answer_for_task,
    rationale_text_from_answer,
    summary_from_answer,
)

DEFAULT_DATASET_NAME = "TheFusion21/PokemonCards"
DEFAULT_OUTPUT_DIR = Path("pokemon_cards/outputs/thefusion21_pokemoncards_v1")
DEFAULT_SPLIT = "train"
DEFAULT_SEED = 42
DEFAULT_TRAIN_FRACTION = 0.80
DEFAULT_VAL_FRACTION = 0.10
DEFAULT_TEST_FRACTION = 0.10
DEFAULT_IMAGE_TIMEOUT = 60.0
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

QUESTION_BY_TASK: dict[str, str] = {
    "card_identity": (
        'Return a JSON object with keys "name", "hp", and "set_name" for this Pokemon card.'
    ),
    "card_core": (
        'Return a JSON object with keys "name", "hp", "set_name", "stage", "pokemon_types", '
        '"rarity", and "evolves_from" for this Pokemon card.'
    ),
    "attack_overview": (
        'Return a JSON object with keys "attack_names" and "attack_count" for this Pokemon card.'
    ),
    "card_summary": (
        'Return a JSON object with key "summary" containing a concise natural-language summary of this '
        'Pokemon card. Mention the card name, HP, set, stage, type(s), and main attack(s) when visible.'
    ),
}


@dataclass(frozen=True)
class ParsedCaption:
    title: Optional[str]
    hp: Optional[int]
    set_name: Optional[str]
    stage: Optional[str]
    pokemon_types: list[str]
    rarity: Optional[str]
    evolves_from: Optional[str]
    attack_names: list[str]


@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    image_url: str
    image_relpath: str
    image_sha1: str
    name: str
    hp: int
    set_name: str
    parsed_caption: ParsedCaption
    noisy_identity: bool
    summary_text: str


def _normalize_spaces(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _normalize_nullable(value: Any) -> Optional[str]:
    text = _normalize_spaces(value)
    if not text:
        return None
    if text.lower() in {"none", "null", "n/a"}:
        return None
    return text


def _parse_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    match = re.search(r"\d+", _normalize_spaces(value))
    if match is None:
        return None
    return int(match.group(0))


def _parse_type_list(raw_types: str) -> list[str]:
    parts = [_normalize_spaces(item) for item in raw_types.split(",")]
    out: list[str] = []
    seen: set[str] = set()
    for part in parts:
        if not part:
            continue
        key = part.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(part)
    out.sort(key=lambda item: item.lower())
    return out


def parse_caption(caption: str) -> ParsedCaption:
    text = _normalize_spaces(caption)
    stage_match = re.search(r"^A (?P<stage>.+?) Pokemon Card of type ", text)
    types_match = re.search(r" of type (?P<types>.+?) with the title ", text)
    title_match = re.search(r" with the title \"?(?P<title>.+?)\"? and (?P<hp>\d+) HP", text)
    rarity_match = re.search(r" of rarity \"?(?P<rarity>.+?)\"?(?: evolved from| from the set)", text)
    evolves_match = re.search(r" evolved from (?P<evolves_from>.+?) from the set ", text)
    set_match = re.search(r" from the set (?P<set_name>.+?)(?: and the flavor text:|\. It has the attack|\. It has weakness)", text)

    attack_names = [
        _normalize_spaces(match.group(1)).strip('"')
        for match in re.finditer(r'It has the attack "?(.+?)"? with the cost ', text)
    ]
    deduped_attacks: list[str] = []
    seen_attacks: set[str] = set()
    for attack_name in attack_names:
        key = attack_name.lower()
        if not attack_name or key in seen_attacks:
            continue
        seen_attacks.add(key)
        deduped_attacks.append(attack_name)

    return ParsedCaption(
        title=_normalize_nullable(title_match.group("title")) if title_match else None,
        hp=_parse_int(title_match.group("hp")) if title_match else None,
        set_name=_normalize_nullable(set_match.group("set_name")) if set_match else None,
        stage=_normalize_nullable(stage_match.group("stage")) if stage_match else None,
        pokemon_types=_parse_type_list(types_match.group("types")) if types_match else [],
        rarity=_normalize_nullable(rarity_match.group("rarity")) if rarity_match else None,
        evolves_from=_normalize_nullable(evolves_match.group("evolves_from")) if evolves_match else None,
        attack_names=deduped_attacks,
    )


def _caption_identity_conflicts(
    *,
    parsed_caption: ParsedCaption,
    name: str,
    hp: int,
    set_name: str,
) -> bool:
    if parsed_caption.title and parsed_caption.title.lower() != name.lower():
        return True
    if parsed_caption.hp is not None and int(parsed_caption.hp) != int(hp):
        return True
    if parsed_caption.set_name and parsed_caption.set_name.lower() != set_name.lower():
        return True
    return False


def _build_card_summary_answer(
    *,
    name: str,
    hp: int,
    set_name: str,
    parsed_caption: ParsedCaption,
) -> dict[str, Any]:
    core_payload = {
        "name": name,
        "hp": hp,
        "set_name": set_name,
        "stage": parsed_caption.stage,
        "pokemon_types": parsed_caption.pokemon_types,
        "rarity": parsed_caption.rarity,
        "evolves_from": parsed_caption.evolves_from,
    }
    normalized_core = normalize_answer_for_task("card_core", core_payload)
    if normalized_core is None:
        raise ValueError("unable to normalize core payload")
    summary = summary_from_answer({**normalized_core, "attack_names": list(parsed_caption.attack_names)})
    return {"summary": summary}


def _hash_bytes(payload: bytes) -> str:
    return hashlib.sha1(payload).hexdigest()  # noqa: S324


def _download_image_bytes(image_url: str, *, timeout: float) -> bytes:
    req = urllib.request.Request(
        image_url,
        headers={"User-Agent": DEFAULT_USER_AGENT},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read()


def _write_png_from_bytes(image_bytes: bytes, *, image_sha1: str, images_dir: Path) -> str:
    out_path = images_dir / f"{image_sha1}.png"
    if out_path.exists():
        return str(Path("images") / out_path.name)
    ensure_parent_dir(out_path)
    from io import BytesIO

    with Image.open(BytesIO(image_bytes)) as image:
        image.convert("RGB").save(out_path, format="PNG")
    return str(Path("images") / out_path.name)


def build_source_record(
    row: dict[str, Any],
    *,
    images_dir: Path,
    timeout: float,
) -> SourceRecord:
    source_id = _normalize_spaces(row.get("id"))
    image_url = _normalize_spaces(row.get("image_url"))
    name = _normalize_spaces(row.get("name"))
    hp = _parse_int(row.get("hp"))
    set_name = _normalize_spaces(row.get("set_name"))
    caption = _normalize_spaces(row.get("caption"))

    if not source_id or not image_url or not name or hp is None or not set_name or not caption:
        raise ValueError(f"row missing required fields: {row}")

    parsed_caption = parse_caption(caption)
    noisy_identity = _caption_identity_conflicts(
        parsed_caption=parsed_caption,
        name=name,
        hp=hp,
        set_name=set_name,
    )

    image_bytes = _download_image_bytes(image_url, timeout=timeout)
    image_sha1 = _hash_bytes(image_bytes)
    image_relpath = _write_png_from_bytes(image_bytes, image_sha1=image_sha1, images_dir=images_dir)

    summary_answer = _build_card_summary_answer(
        name=name,
        hp=hp,
        set_name=set_name,
        parsed_caption=parsed_caption,
    )

    return SourceRecord(
        source_id=source_id,
        image_url=image_url,
        image_relpath=image_relpath,
        image_sha1=image_sha1,
        name=name,
        hp=hp,
        set_name=set_name,
        parsed_caption=parsed_caption,
        noisy_identity=noisy_identity,
        summary_text=str(summary_answer["summary"]),
    )


def task_rows_for_record(record: SourceRecord, *, split_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    identity_answer = {"name": record.name, "hp": record.hp, "set_name": record.set_name}
    base_metadata = {
        "source_id": record.source_id,
        "image_url": record.image_url,
        "image_sha1": record.image_sha1,
        "noisy_identity": record.noisy_identity,
    }

    def add_task(
        task_type: str,
        answer_payload: dict[str, Any],
        *,
        metadata_extra: Optional[dict[str, Any]] = None,
        rationale_answer_payload: Optional[dict[str, Any]] = None,
    ) -> None:
        metadata = dict(base_metadata)
        if metadata_extra:
            metadata.update(metadata_extra)
        rows.append(
            {
                "row_id": f"{record.source_id}__{task_type}",
                "split": split_name,
                "task_type": task_type,
                "question": QUESTION_BY_TASK[task_type],
                "image_path": record.image_relpath,
                "final_answer_json": json.dumps(answer_payload, ensure_ascii=False, sort_keys=True),
                "teacher_rationale_text": "",
                "teacher_model_meta_json": "{}",
                "source_metadata_json": json.dumps(metadata, ensure_ascii=False, sort_keys=True),
                "ground_truth_rationale_text": rationale_text_from_answer(
                    task_type,
                    rationale_answer_payload if rationale_answer_payload is not None else answer_payload,
                ),
            }
        )

    add_task("card_identity", identity_answer)

    if record.noisy_identity:
        return rows

    core_answer = {
        "name": record.name,
        "hp": record.hp,
        "set_name": record.set_name,
        "stage": record.parsed_caption.stage,
        "pokemon_types": record.parsed_caption.pokemon_types,
        "rarity": record.parsed_caption.rarity,
        "evolves_from": record.parsed_caption.evolves_from,
    }
    add_task("card_core", core_answer)

    attack_answer = {
        "attack_names": list(record.parsed_caption.attack_names),
        "attack_count": len(record.parsed_caption.attack_names),
    }
    add_task("attack_overview", attack_answer)

    summary_answer = {"summary": record.summary_text}
    summary_rationale_payload = {
        "name": record.name,
        "hp": record.hp,
        "set_name": record.set_name,
        "stage": record.parsed_caption.stage,
        "pokemon_types": record.parsed_caption.pokemon_types,
        "attack_names": list(record.parsed_caption.attack_names),
    }
    add_task(
        "card_summary",
        summary_answer,
        metadata_extra={
            "summary_source": "natural_template_v2",
            "name": record.name,
            "hp": record.hp,
            "set_name": record.set_name,
            "stage": record.parsed_caption.stage,
            "pokemon_types": list(record.parsed_caption.pokemon_types),
            "attack_names": list(record.parsed_caption.attack_names),
        },
        rationale_answer_payload=summary_rationale_payload,
    )
    return rows


def _assign_splits(records: list[SourceRecord], *, seed: int) -> dict[str, list[SourceRecord]]:
    grouped: dict[str, list[SourceRecord]] = {}
    for record in records:
        grouped.setdefault(record.image_sha1, []).append(record)

    groups = list(grouped.values())
    rng = random.Random(seed)
    rng.shuffle(groups)

    total = sum(len(group) for group in groups)
    train_target = int(round(total * DEFAULT_TRAIN_FRACTION))
    val_target = int(round(total * DEFAULT_VAL_FRACTION))

    splits: dict[str, list[SourceRecord]] = {"train": [], "val": [], "test": []}
    train_count = 0
    val_count = 0

    for group in groups:
        if train_count < train_target:
            bucket = "train"
            train_count += len(group)
        elif val_count < val_target:
            bucket = "val"
            val_count += len(group)
        else:
            bucket = "test"
        splits[bucket].extend(group)
    return splits


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _push_rows_to_hub(
    *,
    split_rows: dict[str, list[dict[str, Any]]],
    output_dir: Path,
    repo_id: str,
    hf_token: str,
    private: bool,
) -> None:
    try:
        from datasets import Dataset, DatasetDict, Features, Image as HFImage, Value  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError("datasets is required for --push-to-hub") from exc

    feature_map = Features(
        {
            "row_id": Value("string"),
            "split": Value("string"),
            "task_type": Value("string"),
            "question": Value("string"),
            "image": HFImage(),
            "image_path": Value("string"),
            "final_answer_json": Value("string"),
            "teacher_rationale_text": Value("string"),
            "teacher_model_meta_json": Value("string"),
            "source_metadata_json": Value("string"),
            "ground_truth_rationale_text": Value("string"),
        }
    )

    dataset_splits: dict[str, Dataset] = {}
    for split_name, rows in split_rows.items():
        normalized_rows: list[dict[str, Any]] = []
        for row in rows:
            image_path = (output_dir / str(row["image_path"])).resolve()
            payload = dict(row)
            payload["image"] = str(image_path)
            normalized_rows.append(payload)
        dataset_splits[split_name] = Dataset.from_list(normalized_rows, features=feature_map)

    DatasetDict(dataset_splits).push_to_hub(repo_id, token=hf_token, private=private)


def _load_raw_rows(dataset_name: str, *, split_name: str, hf_token: str, max_rows: int) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError("datasets is required to build the PokemonCards dataset") from exc

    load_kwargs: dict[str, Any] = {"split": split_name}
    if hf_token:
        load_kwargs["token"] = hf_token
    try:
        ds = load_dataset(dataset_name, **load_kwargs)
    except TypeError:
        token_value = load_kwargs.pop("token", None)
        if token_value:
            load_kwargs["use_auth_token"] = token_value
        ds = load_dataset(dataset_name, **load_kwargs)
    rows = [dict(row) for row in ds]
    if max_rows > 0:
        rows = rows[:max_rows]
    return rows


def _build_summary(split_rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    split_counts = {split_name: len(rows) for split_name, rows in split_rows.items()}
    task_counts_by_split: dict[str, dict[str, int]] = {}
    noisy_rows = 0
    unique_source_ids: set[str] = set()
    for split_name, rows in split_rows.items():
        task_counts: dict[str, int] = {}
        for row in rows:
            task_type = str(row["task_type"])
            task_counts[task_type] = task_counts.get(task_type, 0) + 1
            source_meta = json.loads(str(row["source_metadata_json"]))
            unique_source_ids.add(str(source_meta.get("source_id", "")))
            if bool(source_meta.get("noisy_identity")):
                noisy_rows += 1
        task_counts_by_split[split_name] = dict(sorted(task_counts.items()))
    return {
        "split_counts": split_counts,
        "task_counts_by_split": task_counts_by_split,
        "unique_source_ids": len(unique_source_ids),
        "noisy_task_rows": noisy_rows,
    }


def build_dataset(
    *,
    dataset_name: str,
    split_name: str,
    output_dir: Path,
    seed: int,
    hf_token: str,
    max_rows: int,
    image_timeout: float,
    push_to_hub: str,
    hub_private: bool,
    no_progress: bool,
) -> dict[str, Any]:
    raw_rows = _load_raw_rows(dataset_name, split_name=split_name, hf_token=hf_token, max_rows=max_rows)
    images_dir = output_dir / "images"

    source_records: list[SourceRecord] = []
    failures: list[dict[str, Any]] = []
    progress = tqdm(
        raw_rows,
        total=len(raw_rows),
        desc="Downloading and parsing card rows",
        disable=bool(no_progress),
    )
    for index, row in enumerate(progress, start=1):
        try:
            source_records.append(build_source_record(row, images_dir=images_dir, timeout=image_timeout))
        except (OSError, urllib.error.URLError, urllib.error.HTTPError, ValueError) as exc:
            failures.append({"row_id": row.get("id"), "error": str(exc)})
        if index == 1 or index % 25 == 0 or index == len(raw_rows):
            progress.set_postfix(ok=len(source_records), failed=len(failures))

    split_records = _assign_splits(source_records, seed=seed)
    split_rows: dict[str, list[dict[str, Any]]] = {}
    for split_bucket, records in split_records.items():
        rows: list[dict[str, Any]] = []
        for record in records:
            rows.extend(task_rows_for_record(record, split_name=split_bucket))
        split_rows[split_bucket] = rows
        _write_jsonl(output_dir / "jsonl" / f"{split_bucket}.jsonl", rows)

    summary = _build_summary(split_rows)
    summary.update(
        {
            "dataset_name": dataset_name,
            "source_split": split_name,
            "download_failures": failures,
            "source_record_count": len(source_records),
        }
    )
    write_json(output_dir / "metadata" / "build_summary.json", summary)

    if push_to_hub:
        _push_rows_to_hub(
            split_rows=split_rows,
            output_dir=output_dir,
            repo_id=push_to_hub,
            hf_token=hf_token,
            private=hub_private,
        )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the PokemonCards JSONL task dataset.")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--hf-token", default="")
    parser.add_argument("--max-rows", type=int, default=0, help="0 means all rows.")
    parser.add_argument("--image-timeout", type=float, default=DEFAULT_IMAGE_TIMEOUT)
    parser.add_argument("--push-to-hub", default="")
    parser.add_argument("--hub-private", action="store_true", default=False)
    parser.add_argument("--no-progress", action="store_true", default=False)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    summary = build_dataset(
        dataset_name=args.dataset_name,
        split_name=args.split,
        output_dir=output_dir,
        seed=args.seed,
        hf_token=str(args.hf_token or "").strip(),
        max_rows=int(args.max_rows),
        image_timeout=float(args.image_timeout),
        push_to_hub=str(args.push_to_hub or "").strip(),
        hub_private=bool(args.hub_private),
        no_progress=bool(args.no_progress),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
