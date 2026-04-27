#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

import pyarrow as pa
from datasets import Dataset, DatasetDict, Features, Image as HFImage, Value, load_from_disk

from inspector_md import common

try:
    from huggingface_hub import HfApi
except ModuleNotFoundError:  # pragma: no cover
    HfApi = None  # type: ignore[assignment]

try:
    from datasets import config as datasets_config
    from datasets.download.download_config import DownloadConfig
    from datasets.table import array_cast
    from datasets.utils.file_utils import xopen
    from datasets.utils.py_utils import string_to_dict
except ModuleNotFoundError:  # pragma: no cover
    datasets_config = None  # type: ignore[assignment]
    DownloadConfig = None  # type: ignore[assignment]
    string_to_dict = None  # type: ignore[assignment]
    array_cast = None  # type: ignore[assignment]
    xopen = None  # type: ignore[assignment]

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = common.repo_relative("configs", "publish_inspector_md_hf_dataset_default.json")
DEFAULT_REPO_BASENAME = "inspector_md_full_v1"


def _apply_hf_image_embed_storage_compat_patch() -> None:
    if any(dep is None for dep in (datasets_config, DownloadConfig, string_to_dict, array_cast, xopen)):
        return

    def _ensure_arrow_array(value: pa.Array | pa.ChunkedArray) -> pa.Array:
        if isinstance(value, pa.ChunkedArray):
            return value.combine_chunks()
        return value

    def _embed_storage_compat(self: HFImage, storage: pa.StructArray, token_per_repo_id=None) -> pa.StructArray:
        if token_per_repo_id is None:
            token_per_repo_id = {}

        def path_to_bytes(path: str | None) -> bytes | None:
            if path is None:
                return None
            source_url = path.split("::")[-1]
            pattern = (
                datasets_config.HUB_DATASETS_URL
                if source_url.startswith(datasets_config.HF_ENDPOINT)
                else datasets_config.HUB_DATASETS_HFFS_URL
            )
            source_url_fields = string_to_dict(source_url, pattern)
            token = token_per_repo_id.get(source_url_fields["repo_id"]) if source_url_fields is not None else None
            download_config = DownloadConfig(token=token)
            with xopen(path, "rb", download_config=download_config) as handle:
                return handle.read()

        rows = storage.to_pylist()
        bytes_array = pa.array(
            [
                (path_to_bytes(item["path"]) if item["bytes"] is None else item["bytes"]) if item is not None else None
                for item in rows
            ],
            type=pa.binary(),
        )
        path_array = pa.array(
            [
                os.path.basename(item["path"]) if item is not None and item["path"] is not None else None
                for item in rows
            ],
            type=pa.string(),
        )
        bytes_array = _ensure_arrow_array(bytes_array)
        path_array = _ensure_arrow_array(path_array)
        mask = bytes_array.is_null()
        if not isinstance(mask, pa.Array) or getattr(mask, "type", None) != pa.bool_():
            mask = pa.array(mask.to_pylist(), type=pa.bool_())
        mask = _ensure_arrow_array(mask)
        storage = pa.StructArray.from_arrays([bytes_array, path_array], ["bytes", "path"], mask=mask)
        return array_cast(storage, self.pa_type)

    HFImage.embed_storage = _embed_storage_compat  # type: ignore[assignment]


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    raw_argv = list(argv) if argv is not None else list(os.sys.argv[1:])
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    pre_args, _ = pre_parser.parse_known_args(raw_argv)
    config_path = common.resolve_config_path(pre_args.config, script_dir=SCRIPT_DIR)
    config = common.load_json_config(config_path, default_path=DEFAULT_CONFIG_PATH)

    parser = argparse.ArgumentParser(description="Export and publish the Inspector MD datasets to Hugging Face.")
    parser.add_argument("--config", default=str(config_path))
    parser.add_argument("--env-file", default=str(common.repo_relative(".env.staging")))
    parser.add_argument("--detect-input-dir", default=str(common.repo_relative("outputs", "inspector_detect_v1")))
    parser.add_argument("--point-input-dir", default=str(common.repo_relative("outputs", "inspector_point_v1")))
    parser.add_argument("--query-input-dir", default=str(common.repo_relative("outputs", "inspector_query_issues_v2")))
    parser.add_argument("--output-dir", default=str(common.repo_relative("outputs", "hf_export", DEFAULT_REPO_BASENAME)))
    parser.add_argument("--repo-id", default="")
    parser.add_argument("--hf-token", default="")
    parser.add_argument("--hf-token-env-var", default="HUGGINGFACE_HUB_TOKEN")
    parser.add_argument(
        "--configs",
        nargs="*",
        default=[],
        help="Optional subset of configs to export/push. Choices: detect, point, query_issues",
    )
    parser.add_argument("--push", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--private", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-shard-size", default="500MB")
    parser.add_argument("--embed-external-files", action=argparse.BooleanOptionalAction, default=True)

    option_to_dest: dict[str, str] = {}
    for action in parser._actions:
        if not action.option_strings:
            continue
        for opt in action.option_strings:
            option_to_dest[opt] = action.dest
    overridden = {option_to_dest[arg] for arg in raw_argv if arg in option_to_dest}
    config_cli_args = common.config_to_cli_args(
        parser,
        config,
        config_path=config_path,
        overridden_dests=overridden,
    )
    args = parser.parse_args(config_cli_args + raw_argv)
    args.config = str(common.resolve_config_path(args.config, script_dir=SCRIPT_DIR))
    args.env_file = str(common.resolve_path(args.env_file, module_root=SCRIPT_DIR))
    args.detect_input_dir = common.resolve_path(args.detect_input_dir, module_root=SCRIPT_DIR)
    args.point_input_dir = common.resolve_path(args.point_input_dir, module_root=SCRIPT_DIR)
    args.query_input_dir = common.resolve_path(args.query_input_dir, module_root=SCRIPT_DIR)
    args.output_dir = common.resolve_path(args.output_dir, module_root=SCRIPT_DIR)
    allowed_configs = {"detect", "point", "query_issues"}
    args.configs = [str(item).strip() for item in list(args.configs or []) if str(item).strip()]
    unknown = sorted(set(args.configs) - allowed_configs)
    if unknown:
        raise ValueError(f"Unknown --configs value(s): {unknown}")
    return args


def _resolve_hf_token(explicit_token: str, env_var_name: str) -> str:
    explicit = str(explicit_token or "").strip()
    if explicit:
        return explicit
    for name in (str(env_var_name or "").strip(), "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if not name:
            continue
        value = str(os.environ.get(name) or "").strip()
        if value:
            return value
    raise ValueError("Hugging Face token is required.")


def _resolve_repo_id(api: Any, explicit_repo_id: str) -> str:
    explicit = str(explicit_repo_id or "").strip()
    if explicit:
        return explicit
    profile = api.whoami()
    username = str(profile.get("name") or "").strip()
    if not username:
        raise ValueError("Could not determine Hugging Face username. Pass --repo-id explicitly.")
    return f"{username}/{DEFAULT_REPO_BASENAME}"


def _task_metadata_path(task_dir: Path) -> Path:
    return task_dir / "metadata.json"


def _query_jsonl_dir(task_dir: Path) -> Path:
    path = task_dir / "jsonl"
    if not path.is_dir():
        raise FileNotFoundError(f"Missing JSONL directory: {path}")
    return path


def _query_features() -> Features:
    return Features(
        {
            "row_id": Value("string"),
            "split": Value("string"),
            "task_type": Value("string"),
            "question": Value("string"),
            "image": Value("string"),
            "inspection_request": Value("string"),
            "asset_context": Value("string"),
            "spatial_refs_json": Value("string"),
            "reasoning_text": Value("string"),
            "hard_example": Value("bool"),
            "target_text": Value("string"),
            "target_format": Value("string"),
            "final_answer_json": Value("string"),
            "query_text_refresh_mode": Value("string"),
            "issue_count": Value("int32"),
            "is_multi_issue": Value("bool"),
            "source_dataset": Value("string"),
            "source_annotation_type": Value("string"),
            "crop_derived": Value("bool"),
        }
    )


def _load_query_split(jsonl_path: Path) -> Dataset:
    rows: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            image_path = str(payload.pop("image_path", "")).strip()
            if not image_path:
                raise ValueError(f"Missing image_path in {jsonl_path}")
            payload["image"] = image_path
            rows.append(payload)
    dataset = Dataset.from_list(rows, features=_query_features())
    return dataset.cast_column("image", HFImage())


def _load_query_dataset_dict(task_dir: Path) -> DatasetDict:
    jsonl_dir = _query_jsonl_dir(task_dir)
    splits: dict[str, Dataset] = {}
    for split_name in ("train", "validation", "test"):
        jsonl_path = jsonl_dir / f"{split_name}.jsonl"
        if not jsonl_path.is_file():
            continue
        splits[split_name] = _load_query_split(jsonl_path)
    if not splits:
        raise FileNotFoundError(f"No query JSONL splits found under {jsonl_dir}")
    return DatasetDict(splits)


def _load_image_dataset_dict(path: Path) -> DatasetDict:
    dataset_dict = load_from_disk(str(path))
    if not isinstance(dataset_dict, DatasetDict):
        raise TypeError(f"Expected DatasetDict at {path}")
    return dataset_dict.cast_column("image", HFImage())


def _split_counts(dataset_dict: DatasetDict) -> dict[str, int]:
    return {split_name: int(len(dataset)) for split_name, dataset in dataset_dict.items()}


def _load_json_if_exists(path: Path) -> Any:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _build_dataset_card(
    *,
    repo_id: str,
    private: bool,
    config_summaries: dict[str, dict[str, Any]],
    build_summary: Optional[dict[str, Any]],
) -> str:
    lines: list[str] = []
    lines.append("---")
    lines.append("pretty_name: Inspector MD Full Dataset")
    lines.append("task_categories:")
    lines.append("- object-detection")
    lines.append("- visual-question-answering")
    lines.append("size_categories:")
    lines.append("- 100K<n<1M")
    lines.append("---")
    lines.append("")
    lines.append(f"# {repo_id}")
    lines.append("")
    lines.append("Multi-config Hugging Face export of the Inspector MD training datasets.")
    lines.append("")
    lines.append("## Configs")
    lines.append("")
    for config_name, summary in config_summaries.items():
        lines.append(f"- `{config_name}`: {summary['task_label']} with splits {summary['split_counts']}")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Images are embedded into the published dataset artifacts; local training paths are not preserved.")
    lines.append("- Query configs keep the canonical serialized `issues[]` payloads in both `final_answer_json` and `target_text`.")
    lines.append("- This repo is private by default because source-license redistribution metadata is not normalized in the local merged-dataset outputs.")
    if not private:
        lines.append("- Public publication was requested explicitly. Verify redistribution rights for all merged source datasets.")
    if build_summary:
        lines.append("")
        lines.append("## Build Summary")
        lines.append("")
        if "detect_output_dir" in build_summary:
            lines.append(f"- detect output: `{build_summary['detect_output_dir']}`")
        query_modes = sorted(
            {
                str(summary.get("metadata", {}).get("query_text_refresh_mode") or "")
                for summary in config_summaries.values()
                if isinstance(summary.get("metadata"), dict)
            }
        )
        if query_modes:
            lines.append(f"- query refresh modes: `{', '.join(mode for mode in query_modes if mode)}`")
    lines.append("")
    return "\n".join(lines).strip() + "\n"


def _write_export_artifacts(
    *,
    output_dir: Path,
    repo_id: str,
    private: bool,
    config_summaries: dict[str, dict[str, Any]],
    build_summary: Optional[dict[str, Any]],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    readme_path = output_dir / "README.md"
    summary_path = output_dir / "export_summary.json"
    readme_path.write_text(
        _build_dataset_card(
            repo_id=repo_id,
            private=private,
            config_summaries=config_summaries,
            build_summary=build_summary,
        ),
        encoding="utf-8",
    )
    common.write_json(
        summary_path,
        {
            "repo_id": repo_id,
            "private": bool(private),
            "configs": config_summaries,
            "build_summary": build_summary,
        },
    )
    return readme_path, summary_path


def _upload_file(api: Any, *, repo_id: str, local_path: Path, path_in_repo: str) -> None:
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
    )


def _looks_like_arrow_offset_overflow(exc: Exception) -> bool:
    text = str(exc or "").lower()
    return "offset overflow" in text or "consider casting input from `binary` to `large_binary`" in text


def _push_dataset_dict_with_shard_fallback(
    dataset_dict: DatasetDict,
    *,
    repo_id: str,
    config_name: str,
    set_default: bool,
    private: bool,
    token: str,
    commit_message: str,
    embed_external_files: bool,
    initial_max_shard_size: str,
) -> None:
    shard_sizes = [str(initial_max_shard_size), "25MB", "10MB", "5MB"]
    seen: set[str] = set()
    ordered_sizes = [size for size in shard_sizes if not (size in seen or seen.add(size))]
    last_exc: Exception | None = None
    for index, shard_size in enumerate(ordered_sizes, start=1):
        try:
            if index > 1:
                print(
                    f"retrying {config_name} push with smaller max_shard_size={shard_size} "
                    f"after Arrow overflow on previous attempt"
                )
            dataset_dict.push_to_hub(
                repo_id=repo_id,
                config_name=config_name,
                set_default=set_default,
                private=bool(private),
                token=token,
                commit_message=commit_message,
                embed_external_files=bool(embed_external_files),
                max_shard_size=shard_size,
            )
            return
        except Exception as exc:  # pragma: no cover - exercised against live HF runtime
            last_exc = exc
            if not bool(embed_external_files) or not _looks_like_arrow_offset_overflow(exc) or index == len(ordered_sizes):
                raise
    if last_exc is not None:  # pragma: no cover
        raise last_exc


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    common.maybe_load_env_file(args.env_file, override=False)
    _apply_hf_image_embed_storage_compat_patch()

    if HfApi is None:
        raise ModuleNotFoundError("huggingface_hub is required to publish Inspector MD datasets.")

    hf_token = _resolve_hf_token(args.hf_token, args.hf_token_env_var)
    api = HfApi(token=hf_token or None)
    repo_id = _resolve_repo_id(api, args.repo_id)

    datasets_by_config_all: dict[str, DatasetDict] = {
        "detect": _load_image_dataset_dict(args.detect_input_dir),
        "point": _load_image_dataset_dict(args.point_input_dir),
        "query_issues": _load_query_dataset_dict(args.query_input_dir),
    }
    metadata_by_config_all: dict[str, Any] = {
        "detect": _load_json_if_exists(_task_metadata_path(args.detect_input_dir)),
        "point": _load_json_if_exists(_task_metadata_path(args.point_input_dir)),
        "query_issues": _load_json_if_exists(_task_metadata_path(args.query_input_dir)),
    }
    selected_configs = list(args.configs) if args.configs else list(datasets_by_config_all.keys())
    datasets_by_config = {config_name: datasets_by_config_all[config_name] for config_name in selected_configs}
    metadata_by_config = {config_name: metadata_by_config_all.get(config_name) for config_name in selected_configs}
    build_summary = _load_json_if_exists(common.resolve_path("outputs/build_summary.json", module_root=SCRIPT_DIR))

    config_summaries: dict[str, dict[str, Any]] = {}
    task_labels = {
        "detect": "Detect",
        "point": "Point",
        "query_issues": "Query issues",
    }
    for config_name, dataset_dict in datasets_by_config.items():
        config_summaries[config_name] = {
            "task_label": task_labels[config_name],
            "split_counts": _split_counts(dataset_dict),
            "metadata": metadata_by_config.get(config_name),
        }

    readme_path, summary_path = _write_export_artifacts(
        output_dir=args.output_dir,
        repo_id=repo_id,
        private=bool(args.private),
        config_summaries=config_summaries,
        build_summary=build_summary,
    )

    print(f"prepared HF export at {args.output_dir}")
    for config_name, summary in config_summaries.items():
        print(f"- {config_name}: {summary['split_counts']}")

    if not bool(args.push):
        return

    api.create_repo(repo_id=repo_id, repo_type="dataset", private=bool(args.private), exist_ok=True)
    default_config_name = "detect" if "detect" in datasets_by_config else selected_configs[0]
    for config_name, dataset_dict in datasets_by_config.items():
        _push_dataset_dict_with_shard_fallback(
            dataset_dict,
            repo_id=repo_id,
            config_name=config_name,
            set_default=(config_name == default_config_name),
            private=bool(args.private),
            token=hf_token,
            commit_message=f"Add {config_name} config",
            embed_external_files=bool(args.embed_external_files),
            initial_max_shard_size=str(args.max_shard_size),
        )
        print(f"pushed config {config_name} to {repo_id}")

    _upload_file(api, repo_id=repo_id, local_path=readme_path, path_in_repo="README.md")
    _upload_file(api, repo_id=repo_id, local_path=summary_path, path_in_repo="metadata/export_summary.json")
    for config_name, metadata in metadata_by_config.items():
        if metadata is None:
            continue
        metadata_path = args.output_dir / f"{config_name}.metadata.json"
        common.write_json(metadata_path, metadata)
        _upload_file(
            api,
            repo_id=repo_id,
            local_path=metadata_path,
            path_in_repo=f"metadata/{config_name}.metadata.json",
        )
    if build_summary is not None:
        build_summary_path = args.output_dir / "build_summary.json"
        common.write_json(build_summary_path, build_summary)
        _upload_file(api, repo_id=repo_id, local_path=build_summary_path, path_in_repo="metadata/build_summary.json")

    print(f"pushed dataset repo to {repo_id}")


if __name__ == "__main__":
    main()
