#!/usr/bin/env python3
"""Build a local JSONL VQA-RAD query dataset from Hugging Face."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Optional

from datasets import load_dataset
from PIL import Image

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover
    def load_dotenv(*_args: Any, **_kwargs: Any) -> bool:
        return False

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vqa_rad.common import (
    ANSWER_TYPE_CLOSE,
    DEFAULT_DATASET_NAME,
    DEFAULT_QUERY_OUTPUT_DIR,
    DEFAULT_SHARED_IMAGE_DIR,
    PROMPT_STYLE_CHOICES,
    PROMPT_STYLE_LEGACY_JSON,
    QUERY_TASK_TYPE,
    config_to_cli_args,
    infer_answer_type,
    infer_question_family,
    load_json_config,
    make_prompt,
    normalize_open_answer,
    normalize_question,
    relative_path,
    repo_relative,
    resolve_config_path,
    write_json,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = repo_relative("configs", "build_vqa_rad_hf_dataset_default.json")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    raw_argv = list(argv) if argv is not None else list(os.sys.argv[1:])
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    pre_args, _ = pre_parser.parse_known_args(raw_argv)
    config_path = resolve_config_path(pre_args.config, script_dir=SCRIPT_DIR)
    config = load_json_config(config_path, default_path=DEFAULT_CONFIG_PATH)

    parser = argparse.ArgumentParser(description="Build a local VQA-RAD query dataset.")
    parser.add_argument("--config", default=str(config_path))
    parser.add_argument("--env-file", default=str(repo_relative(".env.staging")))
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"))
    parser.add_argument("--hf-cache-dir", default="")
    parser.add_argument("--output-dir", default=str(DEFAULT_QUERY_OUTPUT_DIR))
    parser.add_argument("--shared-image-dir", default=str(DEFAULT_SHARED_IMAGE_DIR))
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt-style", choices=PROMPT_STYLE_CHOICES, default=PROMPT_STYLE_LEGACY_JSON)

    option_to_dest: dict[str, str] = {}
    for action in parser._actions:
        if not action.option_strings:
            continue
        for opt in action.option_strings:
            option_to_dest[opt] = action.dest
    overridden_dests = {option_to_dest[arg] for arg in raw_argv if arg in option_to_dest}
    config_cli_args = config_to_cli_args(
        parser,
        config,
        config_path=config_path,
        overridden_dests=overridden_dests,
    )
    args = parser.parse_args(config_cli_args + raw_argv)
    args.config = str(resolve_config_path(args.config, script_dir=SCRIPT_DIR))
    return args


def _resolved_output(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path.cwd() / path).resolve()


def _iter_source_rows(
    *,
    dataset_name: str,
    split_name: str,
    hf_token: str,
    hf_cache_dir: str,
) -> Iterable[dict[str, Any]]:
    kwargs: dict[str, Any] = {"split": split_name, "streaming": True}
    if hf_token:
        kwargs["token"] = hf_token
    if hf_cache_dir.strip():
        kwargs["cache_dir"] = hf_cache_dir
    try:
        dataset = load_dataset(dataset_name, **kwargs)
    except TypeError:
        token = kwargs.pop("token", None)
        if token:
            kwargs["use_auth_token"] = token
        dataset = load_dataset(dataset_name, **kwargs)
    for row in dataset:
        if isinstance(row, dict):
            yield dict(row)


def _to_png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def _image_group_id(image: Image.Image) -> str:
    return hashlib.sha1(_to_png_bytes(image)).hexdigest()  # noqa: S324


def _persist_image(image: Image.Image, *, shared_image_dir: Path) -> tuple[str, Path]:
    image_bytes = _to_png_bytes(image)
    digest = hashlib.sha1(image_bytes).hexdigest()  # noqa: S324
    output_path = shared_image_dir / f"{digest}.png"
    if not output_path.exists():
        shared_image_dir.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(image_bytes)
    return digest, output_path.resolve()


def _assign_local_split(*, seed: int, image_group_id: str, val_fraction: float) -> str:
    digest = hashlib.sha1(f"{seed}:{image_group_id}".encode("utf-8")).hexdigest()  # noqa: S324
    bucket = int(digest[:12], 16) / float(16 ** 12)
    return "validation" if bucket < float(val_fraction) else "train"


def _normalized_answer_key(answer_text: str) -> str:
    answer_type = infer_answer_type(answer_text)
    if answer_type == ANSWER_TYPE_CLOSE:
        return str(answer_text or "").strip().lower()
    return normalize_open_answer(answer_text)


def _triplet_key(*, image_group_id: str, question: str, answer_text: str) -> tuple[str, str, str]:
    return (
        str(image_group_id),
        normalize_question(question),
        _normalized_answer_key(answer_text),
    )


def _build_row(
    *,
    question: str,
    answer_text: str,
    split_name: str,
    source_split: str,
    output_dir: Path,
    image_path: Path,
    image_group_id: str,
    row_index: int,
    prompt_style: str,
) -> dict[str, Any]:
    answer_type = infer_answer_type(answer_text)
    question_family = infer_question_family(question)
    return {
        "row_id": f"{split_name}_{image_group_id[:12]}_{row_index:06d}",
        "split": split_name,
        "source_split": source_split,
        "task_type": QUERY_TASK_TYPE,
        "question": make_prompt(question, prompt_style=prompt_style),
        "question_family": question_family,
        "answer_type": answer_type,
        "answer_text": str(answer_text or "").strip(),
        "final_answer_json": json.dumps({"answer": str(answer_text or "").strip()}, ensure_ascii=False),
        "image_path": relative_path(output_dir, image_path),
        "image_group_id": image_group_id,
    }


def _open_handles(output_dir: Path) -> dict[str, Any]:
    jsonl_dir = output_dir / "jsonl"
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    return {
        split_name: (jsonl_dir / f"{split_name}.jsonl").open("w", encoding="utf-8")
        for split_name in ("train", "validation", "test")
    }


def _write_metadata_and_stats(
    *,
    output_dir: Path,
    shared_image_dir: Path,
    dataset_name: str,
    val_fraction: float,
    seed: int,
    prompt_style: str,
    split_counts: dict[str, Counter[str]],
    unique_images: dict[str, set[str]],
    family_counts: dict[str, Counter[str]],
    skipped_counts: Counter[str],
) -> None:
    write_json(
        output_dir / "metadata.json",
        {
            "dataset_name": dataset_name,
            "task_type": QUERY_TASK_TYPE,
            "question_template": make_prompt("{question}", prompt_style=prompt_style),
            "prompt_style": prompt_style,
            "prediction_formats": ["plain_text", "json_object"],
            "json_schema": {"answer": "..."},
            "shared_image_dir": str(shared_image_dir),
            "val_fraction": float(val_fraction),
            "seed": int(seed),
        },
    )
    write_json(
        output_dir / "stats.json",
        {
            "dataset_name": dataset_name,
            "task_type": QUERY_TASK_TYPE,
            "splits": {
                split_name: {
                    "rows": int(split_counts[split_name]["rows"]),
                    "unique_images": len(unique_images[split_name]),
                    "close_ended_rows": int(split_counts[split_name]["close_ended_rows"]),
                    "open_ended_rows": int(split_counts[split_name]["open_ended_rows"]),
                    "question_family_counts": dict(sorted(family_counts[split_name].items())),
                }
                for split_name in ("train", "validation", "test")
            },
            "skipped": dict(sorted(skipped_counts.items())),
        },
    )


def _build_query_jsonl_dataset_streaming(
    *,
    dataset_name: str,
    hf_token: str,
    hf_cache_dir: str,
    output_dir: Path,
    shared_image_dir: Path,
    val_fraction: float,
    seed: int,
    prompt_style: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    shared_image_dir.mkdir(parents=True, exist_ok=True)
    handles = _open_handles(output_dir)
    split_counts = {split_name: Counter() for split_name in ("train", "validation", "test")}
    family_counts = {split_name: Counter() for split_name in ("train", "validation", "test")}
    unique_images = {split_name: set() for split_name in ("train", "validation", "test")}
    skipped_counts: Counter[str] = Counter()
    seen_triplets: dict[str, set[tuple[str, str, str]]] = {
        split_name: set() for split_name in ("train", "validation", "test")
    }
    test_image_groups: set[str] = set()
    test_triplets: set[tuple[str, str, str]] = set()
    row_index = 0

    try:
        for source_split in ("test", "train"):
            for row in _iter_source_rows(
                dataset_name=dataset_name,
                split_name=source_split,
                hf_token=hf_token,
                hf_cache_dir=hf_cache_dir,
            ):
                image = row.get("image")
                if image is None:
                    skipped_counts["missing_image"] += 1
                    continue
                if not isinstance(image, Image.Image):
                    skipped_counts["invalid_image"] += 1
                    continue
                question = str(row.get("question") or "").strip()
                answer_text = str(row.get("answer") or "").strip()
                if not question or not answer_text:
                    skipped_counts["missing_text"] += 1
                    continue
                image_group_id, image_path = _persist_image(image, shared_image_dir=shared_image_dir)
                split_name = "test" if source_split == "test" else _assign_local_split(
                    seed=seed,
                    image_group_id=image_group_id,
                    val_fraction=val_fraction,
                )
                triplet_key = _triplet_key(
                    image_group_id=image_group_id,
                    question=question,
                    answer_text=answer_text,
                )
                if source_split == "test":
                    if triplet_key in test_triplets:
                        skipped_counts["duplicate_test_triplet"] += 1
                        continue
                    test_image_groups.add(image_group_id)
                    test_triplets.add(triplet_key)
                else:
                    if image_group_id in test_image_groups:
                        skipped_counts["train_test_image_overlap"] += 1
                        continue
                    if triplet_key in test_triplets:
                        skipped_counts["train_test_triplet_overlap"] += 1
                        continue
                if triplet_key in seen_triplets[split_name]:
                    skipped_counts[f"duplicate_{split_name}_triplet"] += 1
                    continue
                seen_triplets[split_name].add(triplet_key)
                normalized_row = _build_row(
                    question=question,
                    answer_text=answer_text,
                    split_name=split_name,
                    source_split=source_split,
                    output_dir=output_dir,
                    image_path=image_path,
                    image_group_id=image_group_id,
                    row_index=row_index,
                    prompt_style=prompt_style,
                )
                row_index += 1
                handles[split_name].write(json.dumps(normalized_row, ensure_ascii=False) + "\n")
                split_counts[split_name]["rows"] += 1
                if normalized_row["answer_type"] == ANSWER_TYPE_CLOSE:
                    split_counts[split_name]["close_ended_rows"] += 1
                else:
                    split_counts[split_name]["open_ended_rows"] += 1
                unique_images[split_name].add(image_group_id)
                family_counts[split_name][str(normalized_row["question_family"])] += 1
    finally:
        for handle in handles.values():
            handle.close()

    _write_metadata_and_stats(
        output_dir=output_dir,
        shared_image_dir=shared_image_dir,
        dataset_name=dataset_name,
        val_fraction=val_fraction,
        seed=seed,
        prompt_style=prompt_style,
        split_counts=split_counts,
        unique_images=unique_images,
        family_counts=family_counts,
        skipped_counts=skipped_counts,
    )


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    load_dotenv(args.env_file, override=False)
    if not args.hf_token:
        args.hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or ""
    if not (0.0 < float(args.val_fraction) < 1.0):
        raise ValueError("--val-fraction must be in (0, 1)")
    output_dir = _resolved_output(args.output_dir)
    shared_image_dir = _resolved_output(args.shared_image_dir)
    _build_query_jsonl_dataset_streaming(
        dataset_name=str(args.dataset_name),
        hf_token=str(args.hf_token or ""),
        hf_cache_dir=str(args.hf_cache_dir or ""),
        output_dir=output_dir,
        shared_image_dir=shared_image_dir,
        val_fraction=float(args.val_fraction),
        seed=int(args.seed),
        prompt_style=str(args.prompt_style),
    )
    print(f"saved vqa_rad dataset: {output_dir}")


if __name__ == "__main__":
    main()
