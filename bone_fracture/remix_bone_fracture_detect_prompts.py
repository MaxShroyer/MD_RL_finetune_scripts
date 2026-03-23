#!/usr/bin/env python3
"""Clone a saved detect dataset and rewrite prompts to generic broken-bone variants."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from datasets import Dataset, DatasetDict, load_from_disk
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bone_fracture.common import DEFAULT_DETECT_HF_DATASET_NAME, repo_relative, write_json

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = repo_relative("outputs", "maxs-m87_bone_fracture_detect_v1")
DEFAULT_OUTPUT_DIR = repo_relative("outputs", "maxs-m87_bone_fracture_detect_v1_prompt_variants")
DEFAULT_PROMPT_VARIANTS = [
    "broken bone",
    "fractured bone",
    "bone fracture",
    "fractureed bone",
    "bone is broken",
    "fracture in the bone",
    "bone looks fractured",
    "xray of a broken bone",
]


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rewrite bone-fracture detect prompts to generic variants.")
    parser.add_argument("--env-file", default=str(repo_relative(".env.staging")))
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prompt-variants-json",
        default="",
        help="Optional JSON list of prompt variants. Defaults to a built-in broken-bone variant pool.",
    )
    parser.add_argument("--push-to-hub", default="")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"))
    return parser.parse_args(argv)


def _resolve_dir(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path.resolve()
    parts = path.parts
    if parts and parts[0] == SCRIPT_DIR.name:
        return (REPO_ROOT / path).resolve()
    return path.resolve()


def _load_prompt_variants(raw_value: str) -> list[str]:
    if not str(raw_value or "").strip():
        return list(DEFAULT_PROMPT_VARIANTS)
    payload = json.loads(raw_value)
    if not isinstance(payload, list):
        raise ValueError("--prompt-variants-json must decode to a JSON list.")
    variants = [str(item or "").strip() for item in payload]
    variants = [item for item in variants if item]
    if not variants:
        raise ValueError("Prompt variant list must contain at least one non-empty string.")
    return variants


def _prompt_schedule(num_rows: int, *, prompt_variants: list[str], seed: int) -> list[str]:
    prompts = [prompt_variants[idx % len(prompt_variants)] for idx in range(num_rows)]
    random.Random(seed).shuffle(prompts)
    return prompts


def _rewrite_split_prompts(split_dataset: Dataset, *, prompts: list[str], split_name: str) -> Dataset:
    if len(split_dataset) != len(prompts):
        raise ValueError(
            f"Prompt schedule length mismatch for {split_name}: {len(prompts)} != {len(split_dataset)}"
        )

    def _assign_prompts(_batch: dict[str, list[Any]], indices: list[int]) -> dict[str, list[str]]:
        return {"prompt": [prompts[idx] for idx in indices]}

    return split_dataset.map(
        _assign_prompts,
        batched=True,
        with_indices=True,
        load_from_cache_file=False,
        desc=f"rewrite {split_name} prompts",
    )


def _rewrite_class_catalog(class_catalog: Any, *, prompt_variants: list[str]) -> list[dict[str, Any]]:
    if not isinstance(class_catalog, list):
        return []
    rewritten: list[dict[str, Any]] = []
    for idx, item in enumerate(class_catalog):
        if not isinstance(item, dict):
            continue
        payload = dict(item)
        payload["prompt"] = prompt_variants[idx % len(prompt_variants)]
        rewritten.append(payload)
    return rewritten


def _prompt_counts_by_split(dataset_dict: DatasetDict) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for split_name, split_dataset in dataset_dict.items():
        counts[split_name] = dict(sorted(Counter(split_dataset["prompt"]).items()))
    return counts


def _load_json_if_present(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _build_metadata(
    *,
    source_metadata: dict[str, Any],
    input_dir: Path,
    output_dir: Path,
    prompt_variants: list[str],
    prompt_counts: dict[str, dict[str, int]],
    seed: int,
    push_to_hub: str,
) -> dict[str, Any]:
    metadata = dict(source_metadata)
    metadata["source_output_dir"] = str(input_dir)
    metadata["output_dir"] = str(output_dir)
    metadata["prompt_rewrite_seed"] = seed
    metadata["prompt_rewrite_strategy"] = "split_balanced_cycle_then_shuffle"
    metadata["prompt_variants"] = list(prompt_variants)
    metadata["prompt_counts"] = prompt_counts
    metadata["push_to_hub"] = bool(push_to_hub)
    metadata["hub_repo_id"] = push_to_hub or str(metadata.get("hub_repo_id") or "")
    rewritten_class_catalog = _rewrite_class_catalog(metadata.get("class_catalog"), prompt_variants=prompt_variants)
    if rewritten_class_catalog:
        metadata["class_catalog"] = rewritten_class_catalog
    return metadata


def _build_stats(
    *,
    source_stats: dict[str, Any],
    prompt_counts: dict[str, dict[str, int]],
) -> dict[str, Any]:
    stats = dict(source_stats)
    stats["prompt_counts"] = prompt_counts
    return stats


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    load_dotenv(args.env_file, override=False)
    if not args.hf_token:
        args.hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    input_dir = _resolve_dir(args.input_dir)
    output_dir = _resolve_dir(args.output_dir)
    if input_dir == output_dir:
        raise ValueError("--output-dir must differ from --input-dir so the source dataset stays intact.")

    prompt_variants = _load_prompt_variants(args.prompt_variants_json)
    source_dataset = load_from_disk(str(input_dir))
    if not isinstance(source_dataset, DatasetDict):
        raise ValueError(f"Expected a DatasetDict at {input_dir}, found {type(source_dataset)!r}")

    split_names = list(source_dataset.keys())
    rewritten_splits: dict[str, Dataset] = {}
    for split_index, split_name in enumerate(split_names):
        prompts = _prompt_schedule(
            len(source_dataset[split_name]),
            prompt_variants=prompt_variants,
            seed=args.seed + split_index + 1,
        )
        rewritten_splits[split_name] = _rewrite_split_prompts(
            source_dataset[split_name],
            prompts=prompts,
            split_name=split_name,
        )
    rewritten_dataset = DatasetDict(rewritten_splits)

    prompt_counts = _prompt_counts_by_split(rewritten_dataset)
    source_metadata = _load_json_if_present(input_dir / "metadata.json")
    source_stats = _load_json_if_present(input_dir / "stats.json")
    metadata = _build_metadata(
        source_metadata=source_metadata,
        input_dir=input_dir,
        output_dir=output_dir,
        prompt_variants=prompt_variants,
        prompt_counts=prompt_counts,
        seed=args.seed,
        push_to_hub=str(args.push_to_hub or "").strip(),
    )
    stats = _build_stats(source_stats=source_stats, prompt_counts=prompt_counts)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    rewritten_dataset.save_to_disk(str(output_dir))
    write_json(output_dir / "metadata.json", metadata)
    write_json(output_dir / "stats.json", stats)

    print(
        f"saved prompt-remixed dataset to {output_dir} "
        f"(train={len(rewritten_dataset['train'])}, validation={len(rewritten_dataset['validation'])}, test={len(rewritten_dataset['test'])})"
    )
    print(f"prompt variants: {json.dumps(prompt_variants)}")

    if args.push_to_hub:
        if not args.hf_token:
            raise ValueError("HF token required to push to hub.")
        rewritten_dataset.push_to_hub(args.push_to_hub, token=args.hf_token)
        print(f"pushed dataset to {args.push_to_hub}")


if __name__ == "__main__":
    main()
