from __future__ import annotations

import io
import json
import re
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from datasets import load_from_disk
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = REPO_ROOT / "outputs" / "task_sample_packets"


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return slug or "sample"


def _normalize_id(value: str) -> str:
    return (value or "").strip().lower()


def _parse_boxes(value: Any) -> list[dict[str, Any]]:
    if value in (None, "", []):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return [value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return [parsed]
    return []


def _latest_snapshot_path(dataset_id: str) -> Path:
    hub_dir = Path.home() / ".cache" / "huggingface" / "hub" / f"datasets--{dataset_id.replace('/', '--')}"
    ref_path = hub_dir / "refs" / "main"
    if ref_path.exists():
        snapshot_id = ref_path.read_text(encoding="utf-8").strip()
        return hub_dir / "snapshots" / snapshot_id
    snapshots = sorted((hub_dir / "snapshots").iterdir())
    if not snapshots:
        raise FileNotFoundError(f"No cached snapshots found for {dataset_id}")
    return snapshots[-1]


def _read_parquet_rows(parquet_path: Path) -> list[dict[str, Any]]:
    return pq.read_table(parquet_path).to_pylist()


def _image_from_bytes(image_payload: dict[str, Any]) -> Image.Image:
    raw_bytes = image_payload.get("bytes")
    if raw_bytes is None:
        raise ValueError("Missing image bytes in parquet row")
    return Image.open(io.BytesIO(raw_bytes)).convert("RGB")


def _save_image(image: Image.Image, image_path: Path) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(image_path)


def _find_wandb_summary(root: Path, finetune_id: str) -> tuple[Path, dict[str, Any]]:
    for summary_path in sorted(root.glob("run-*/files/wandb-summary.json")):
        payload = _load_json(summary_path)
        if payload.get("finetune_id") == finetune_id:
            return summary_path, payload
    raise FileNotFoundError(f"Unable to find W&B summary for finetune {finetune_id}")


def _slim_metrics(payload: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    return {key: payload.get(key) for key in keys}


def _render_task_readme(
    *,
    title: str,
    task_text: str,
    dataset_id: str,
    dataset_url: str,
    export_note: str,
    source_details: list[str],
    before_after_lines: list[str],
) -> str:
    lines = [
        f"# {title}",
        "",
        "## Task",
        task_text,
        "",
        "## Dataset",
        f"- Hugging Face: `{dataset_id}`",
        f"- Link: {dataset_url}",
        "",
        "## Export",
        export_note,
    ]
    if source_details:
        lines.extend(["", "## Source Details", *[f"- {line}" for line in source_details]])
    if before_after_lines:
        lines.extend(["", "## Before / After", *[f"- {line}" for line in before_after_lines]])
    lines.extend(
        [
            "",
            "## Contents",
            "- `images/`: 20 source images for this task.",
            "- `samples.before_after.json`: sample metadata, image filenames, annotations, and before/after references.",
            "",
        ]
    )
    return "\n".join(lines)


def _build_state_farm_packet() -> dict[str, Any]:
    dataset_id = "maxs-m87/NBA_StateFarm_Splits_01"
    dataset_url = "https://huggingface.co/datasets/maxs-m87/NBA_StateFarm_Splits_01"
    snapshot_path = _latest_snapshot_path(dataset_id)
    parquet_path = snapshot_path / "data" / "validation-00000-of-00001.parquet"
    rows = _read_parquet_rows(parquet_path)[:20]

    summary_path, summary = _find_wandb_summary(
        REPO_ROOT / "_DEPICATED_MDstatefarmRL" / "wandb",
        finetune_id="01KGJ70G90A12SKSKW0579RCT0",
    )
    packet_dir = OUTPUT_ROOT / "state_farm"
    images_dir = packet_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    before_metrics = {
        "scope": "task_level_metrics",
        "source": str(summary_path.relative_to(REPO_ROOT)),
        "model": "moondream3-preview",
        "metrics": {
            "eval_f1": summary.get("baseline_eval_f1"),
            "eval_f1_macro": summary.get("baseline_eval_f1_macro"),
            "eval_miou": summary.get("baseline_eval_miou"),
            "eval_true_pos": summary.get("baseline_eval_true_pos"),
            "eval_false_pos": summary.get("baseline_eval_false_pos"),
            "eval_false_neg": summary.get("baseline_eval_false_neg"),
        },
        "note": "Local artifacts did not preserve per-sample predictions for this task, only run-level metrics.",
    }
    after_metrics = {
        "scope": "task_level_metrics",
        "source": str(summary_path.relative_to(REPO_ROOT)),
        "model": "finetune 01KGJ70G90A12SKSKW0579RCT0",
        "metrics": {
            "eval_f1": summary.get("eval_f1"),
            "eval_f1_macro": summary.get("eval_f1_macro"),
            "eval_miou": summary.get("eval_miou"),
            "eval_true_pos": summary.get("eval_true_pos"),
            "eval_false_pos": summary.get("eval_false_pos"),
            "eval_false_neg": summary.get("eval_false_neg"),
        },
        "note": "This comes from the saved training-run evaluation summary rather than a retained per-sample export.",
    }

    samples: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        image_name = row["image"].get("path") or f"state_farm_{index:03d}.jpg"
        sample_id = Path(image_name).stem
        image_file = f"{index:03d}_{_safe_slug(sample_id)}.png"
        image_path = images_dir / image_file
        _save_image(_image_from_bytes(row["image"]), image_path)
        samples.append(
            {
                "sample_index": index,
                "sample_id": sample_id,
                "image_file": f"images/{image_file}",
                "source_image_path": image_name,
                "prompt": row.get("prompt"),
                "task_type": row.get("type"),
                "ground_truth_boxes": _parse_boxes(row.get("answer_boxes")),
                "notes": row.get("notes"),
                "timestamp": row.get("timestamp"),
                "before": before_metrics,
                "after": after_metrics,
            }
        )

    readme_text = _render_task_readme(
        title="State Farm",
        task_text="Detect the State Farm logo in NBA broadcast frames.",
        dataset_id=dataset_id,
        dataset_url=dataset_url,
        export_note=(
            "This packet uses the first 20 cached validation samples from the local Hugging Face snapshot. "
            "The sample JSON includes exact image annotations and task-level before/after metrics because no local "
            "per-sample prediction log was preserved."
        ),
        source_details=[
            f"cached snapshot: `{snapshot_path.relative_to(Path.home())}`",
            f"parquet split: `{parquet_path.name}`",
            "selected samples: first 20 validation rows",
        ],
        before_after_lines=[
            "before: baseline metrics from the saved W&B training summary for finetune `01KGJ70G90A12SKSKW0579RCT0`",
            "after: finetuned metrics from the same W&B training summary",
        ],
    )
    (packet_dir / "README.md").write_text(readme_text, encoding="utf-8")
    _write_json(packet_dir / "samples.before_after.json", samples)
    return {
        "title": "State Farm",
        "folder": str(packet_dir.relative_to(REPO_ROOT)),
        "sample_count": len(samples),
    }


def _best_ballholder_candidate() -> tuple[Path, dict[str, Any]]:
    candidate_files = sorted((REPO_ROOT / "_DEPICATED_MDBallHolder" / "outputs").glob("staging_*.json"))
    candidates = [(path, _load_json(path)) for path in candidate_files]
    if not candidates:
        raise FileNotFoundError("No ball-holder candidate benchmark files found")
    return max(candidates, key=lambda item: (item[1].get("eval_miou", 0.0), item[1].get("eval_f1", 0.0)))


def _build_ballholder_packet() -> dict[str, Any]:
    dataset_id = "maxs-m87/Ball-Holder-splits-v1"
    dataset_url = "https://huggingface.co/datasets/maxs-m87/Ball-Holder-splits-v1"
    snapshot_path = _latest_snapshot_path(dataset_id)
    parquet_path = snapshot_path / "data" / "test-00000-of-00001.parquet"
    rows = _read_parquet_rows(parquet_path)[:20]

    baseline_path = REPO_ROOT / "_DEPICATED_MDBallHolder" / "outputs" / "benchmark_metrics.json"
    baseline_metrics = _load_json(baseline_path)
    candidate_path, candidate_metrics = _best_ballholder_candidate()

    packet_dir = OUTPUT_ROOT / "player_with_ball"
    images_dir = packet_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    before_metrics = {
        "scope": "task_level_metrics",
        "source": str(baseline_path.relative_to(REPO_ROOT)),
        "model": baseline_metrics.get("model"),
        "metrics": _slim_metrics(
            baseline_metrics,
            ["eval_f1", "eval_f1_macro", "eval_miou", "tp", "fp", "fn", "samples"],
        ),
        "note": "Local artifacts retained only task-level benchmark metrics for this test split.",
    }
    after_metrics = {
        "scope": "task_level_metrics",
        "source": str(candidate_path.relative_to(REPO_ROOT)),
        "model": candidate_metrics.get("model"),
        "metrics": _slim_metrics(
            candidate_metrics,
            ["eval_f1", "eval_f1_macro", "eval_miou", "tp", "fp", "fn", "samples"],
        ),
        "note": "This is the best locally saved candidate benchmark by mIoU.",
    }

    samples: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        image_name = row["image"].get("path") or f"player_with_ball_{index:03d}.jpg"
        sample_id = Path(image_name).stem
        image_file = f"{index:03d}_{_safe_slug(sample_id)}.png"
        image_path = images_dir / image_file
        _save_image(_image_from_bytes(row["image"]), image_path)
        samples.append(
            {
                "sample_index": index,
                "sample_id": sample_id,
                "image_file": f"images/{image_file}",
                "source_image_path": image_name,
                "prompt": row.get("prompt"),
                "task_type": row.get("type"),
                "ground_truth_boxes": _parse_boxes(row.get("answer_boxes")),
                "notes": row.get("notes"),
                "timestamp": row.get("timestamp"),
                "before": before_metrics,
                "after": after_metrics,
            }
        )

    readme_text = _render_task_readme(
        title="Player With Ball",
        task_text="Detect the player currently holding the basketball.",
        dataset_id=dataset_id,
        dataset_url=dataset_url,
        export_note=(
            "This packet uses all 20 cached test samples from the local Hugging Face snapshot. "
            "The sample JSON includes exact image annotations and task-level before/after metrics because no local "
            "per-sample prediction log was preserved."
        ),
        source_details=[
            f"cached snapshot: `{snapshot_path.relative_to(Path.home())}`",
            f"parquet split: `{parquet_path.name}`",
            "selected samples: all 20 test rows",
        ],
        before_after_lines=[
            f"before: baseline benchmark from `{baseline_path.relative_to(REPO_ROOT)}`",
            f"after: best saved candidate benchmark from `{candidate_path.relative_to(REPO_ROOT)}`",
        ],
    )
    (packet_dir / "README.md").write_text(readme_text, encoding="utf-8")
    _write_json(packet_dir / "samples.before_after.json", samples)
    return {
        "title": "Player With Ball",
        "folder": str(packet_dir.relative_to(REPO_ROOT)),
        "sample_count": len(samples),
    }


def _record_map(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {_normalize_id(record["sample_id"]): record for record in records}


def _slim_aerial_record(record: dict[str, Any] | None) -> dict[str, Any] | None:
    if record is None:
        return None
    return {
        "label": record.get("label"),
        "model": record.get("model"),
        "finetune_id": record.get("finetune_id"),
        "checkpoint_step": record.get("checkpoint_step"),
        "prompt": record.get("prompt"),
        "task_f1": record.get("task_f1"),
        "task_miou": record.get("task_miou"),
        "tp": record.get("tp"),
        "fp": record.get("fp"),
        "fn": record.get("fn"),
        "pred_count": record.get("pred_count"),
        "gt_count": record.get("gt_count"),
        "run_id": record.get("run_id"),
    }


def _build_aerial_packet() -> dict[str, Any]:
    dataset_id = "maxs-m87/aerial_airport_point_v2"
    dataset_url = "https://huggingface.co/datasets/maxs-m87/aerial_airport_point_v2"
    subset_dir = REPO_ROOT / "outputs" / "advertising_subsets" / "aerial_manual_top100"
    dataset_path = subset_dir / "subset" / "dataset"
    dataset = load_from_disk(str(dataset_path))["test"]

    selected_manifest = _load_jsonl(subset_dir / "selected_samples.jsonl")[:20]
    delta_map = {
        _normalize_id(item["sample_id"]): item
        for item in _load_jsonl(subset_dir / "combined_sample_deltas.jsonl")
    }
    detect_before = _record_map(_load_jsonl(subset_dir / "detect" / "subset" / "baseline.records.jsonl"))
    detect_after = _record_map(_load_jsonl(subset_dir / "detect" / "subset" / "candidate.records.jsonl"))
    point_before = _record_map(_load_jsonl(subset_dir / "point" / "subset" / "baseline.records.jsonl"))
    point_after = _record_map(_load_jsonl(subset_dir / "point" / "subset" / "candidate.records.jsonl"))

    row_index: dict[str, dict[str, Any]] = {}
    for row in dataset:
        row_index[_normalize_id(row["source_image_id"])] = row

    packet_dir = OUTPUT_ROOT / "aerial"
    images_dir = packet_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    detect_before_metrics = _load_json(subset_dir / "detect" / "subset" / "baseline.metrics.json")
    detect_after_metrics = _load_json(subset_dir / "detect" / "subset" / "candidate.metrics.json")
    point_before_metrics = _load_json(subset_dir / "point" / "subset" / "baseline.metrics.json")
    point_after_metrics = _load_json(subset_dir / "point" / "subset" / "candidate.metrics.json")

    samples: list[dict[str, Any]] = []
    for manifest in selected_manifest:
        sample_id = manifest["sample_id"]
        normalized_id = _normalize_id(sample_id)
        row = row_index.get(normalized_id)
        if row is None:
            raise KeyError(f"Aerial subset sample not found in local dataset: {sample_id}")
        image_file = f"{manifest['selection_rank']:03d}_{_safe_slug(sample_id)}.png"
        image_path = images_dir / image_file
        _save_image(row["image"].convert("RGB"), image_path)
        samples.append(
            {
                "selection_rank": manifest["selection_rank"],
                "sample_id": sample_id,
                "image_file": f"images/{image_file}",
                "source_image_id": row.get("source_image_id"),
                "source_split": row.get("source_split"),
                "source_dataset": row.get("source_dataset"),
                "source_collection": row.get("source_collection"),
                "source_variant": row.get("source_variant"),
                "source_is_synthetic": row.get("source_is_synthetic"),
                "class_count": row.get("class_count"),
                "ground_truth_boxes": _parse_boxes(row.get("answer_boxes")),
                "selection": manifest,
                "delta_summary": delta_map.get(normalized_id),
                "before": {
                    "detect": _slim_aerial_record(detect_before.get(normalized_id)),
                    "point": _slim_aerial_record(point_before.get(normalized_id)),
                },
                "after": {
                    "detect": _slim_aerial_record(detect_after.get(normalized_id)),
                    "point": _slim_aerial_record(point_after.get(normalized_id)),
                },
            }
        )

    readme_text = _render_task_readme(
        title="Aerial",
        task_text="Detect airplanes in aerial airport imagery.",
        dataset_id=dataset_id,
        dataset_url=dataset_url,
        export_note=(
            "This packet uses the curated aerial advertising subset in "
            "`outputs/advertising_subsets/aerial_manual_top100/subset/dataset`. "
            "The 20 images here are the first 20 ranked samples from `selected_samples.jsonl`, and the sample JSON "
            "contains true sample-level before/after records for both the detect and point tools."
        ),
        source_details=[
            f"subset root: `{subset_dir.relative_to(REPO_ROOT)}`",
            f"local subset dataset: `{dataset_path.relative_to(REPO_ROOT)}`",
            "selected samples: first 20 rows from `selected_samples.jsonl`",
        ],
        before_after_lines=[
            f"detect subset before/after: f1 {detect_before_metrics.get('eval_f1'):.6f} -> {detect_after_metrics.get('eval_f1'):.6f}, "
            f"miou {detect_before_metrics.get('eval_miou'):.6f} -> {detect_after_metrics.get('eval_miou'):.6f}",
            f"point subset before/after: f1 {point_before_metrics.get('eval_f1'):.6f} -> {point_after_metrics.get('eval_f1'):.6f}, "
            f"miou {point_before_metrics.get('eval_miou'):.6f} -> {point_after_metrics.get('eval_miou'):.6f}",
        ],
    )
    (packet_dir / "README.md").write_text(readme_text, encoding="utf-8")
    _write_json(packet_dir / "samples.before_after.json", samples)
    return {
        "title": "Aerial",
        "folder": str(packet_dir.relative_to(REPO_ROOT)),
        "sample_count": len(samples),
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary = {
        "output_root": str(OUTPUT_ROOT.relative_to(REPO_ROOT)),
        "exports": [
            _build_state_farm_packet(),
            _build_ballholder_packet(),
            _build_aerial_packet(),
        ],
    }
    top_level_readme = "\n".join(
        [
            "# Task Sample Packets",
            "",
            "Generated exports:",
            *[f"- {item['title']}: `{item['folder']}` ({item['sample_count']} images)" for item in summary["exports"]],
            "",
            "State Farm and Player With Ball only retained task-level benchmark summaries locally, so their sample JSON files "
            "repeat those before/after metrics per sample alongside exact annotations.",
            "Aerial uses the local aerial advertising subset and includes sample-level before/after records from the saved subset benchmarks.",
            "",
        ]
    )
    (OUTPUT_ROOT / "README.md").write_text(top_level_readme, encoding="utf-8")
    _write_json(OUTPUT_ROOT / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
