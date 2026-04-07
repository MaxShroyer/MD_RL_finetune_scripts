from __future__ import annotations

import io
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from dotenv import load_dotenv
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import benchmark_statefarm_val as mod
from tuna_sdk import TunaClient

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_ID = "maxs-m87/NBA_StateFarm_Splits_01"
DEFAULT_SPLIT = "validation"
DEFAULT_BASELINE_MODEL = mod.DEFAULT_MODEL
DEFAULT_API_BASE = "https://api.moondream.ai/v1"
DEFAULT_IOU_THRESHOLD = 0.4


def _latest_snapshot_path(dataset_id: str) -> Path:
    hub_dir = Path.home() / ".cache" / "huggingface" / "hub" / f"datasets--{dataset_id.replace('/', '--')}"
    ref_path = hub_dir / "refs" / "main"
    if ref_path.exists():
        snapshot_id = ref_path.read_text(encoding="utf-8").strip()
        return hub_dir / "snapshots" / snapshot_id
    snapshots_dir = hub_dir / "snapshots"
    snapshots = sorted(snapshots_dir.iterdir())
    if not snapshots:
        raise FileNotFoundError(f"No cached snapshots found for {dataset_id}")
    return snapshots[-1]


def _load_split_rows(dataset_id: str, split: str) -> tuple[Path, list[dict[str, Any]]]:
    snapshot_path = _latest_snapshot_path(dataset_id)
    parquet_path = snapshot_path / "data" / f"{split}-00000-of-00001.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Unable to find cached parquet for split {split}: {parquet_path}")
    return snapshot_path, pq.read_table(parquet_path).to_pylist()


def _image_from_payload(image_payload: dict[str, Any]) -> Image.Image:
    raw_bytes = image_payload.get("bytes")
    if raw_bytes is None:
        raise ValueError("Image bytes missing from dataset row")
    return Image.open(io.BytesIO(raw_bytes)).convert("RGB")


def _sanitize_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in value).strip("._") or "sample"


def _serialize_boxes(boxes: list[mod.Box]) -> list[dict[str, float]]:
    return [
        {
            "x_min": box.x_min,
            "y_min": box.y_min,
            "x_max": box.x_max,
            "y_max": box.y_max,
        }
        for box in boxes
    ]


def _candidate_api_keys(env_path: Path) -> list[str]:
    keys: list[str] = []
    active = os.environ.get("MOONDREAM_API_KEY")
    if active:
        keys.append(active.strip())
    text = env_path.read_text(encoding="utf-8")
    for line in text.splitlines():
        match = re.match(r"\s*#?\s*MOONDREAM_API_KEY=(.+)", line)
        if not match:
            continue
        key = match.group(1).strip()
        if key and key not in keys:
            keys.append(key)
    return keys


def _resolve_api_key_for_finetune(*, env_path: Path, finetune_id: str, api_base: str) -> str:
    last_error: str | None = None
    for index, key in enumerate(_candidate_api_keys(env_path), start=1):
        client = TunaClient(api_key=key, base_url=api_base)
        try:
            finetune = client.get_finetune(finetune_id)
            print(f"using Moondream key candidate {index} for finetune {getattr(finetune, 'finetune_id', finetune_id)}", flush=True)
            return key
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        finally:
            client.close()
    raise RuntimeError(f"Unable to resolve a Moondream API key for finetune {finetune_id}. Last error: {last_error}")


def _evaluate_prediction(pred_boxes: list[mod.Box], gt_boxes: list[mod.Box], iou_threshold: float) -> dict[str, Any]:
    tp, fp, fn = mod._count_tp_fp_fn(pred_boxes, gt_boxes, iou_threshold=iou_threshold)
    return {
        "pred_boxes": _serialize_boxes(pred_boxes),
        "task_f1": mod._reward_f1(pred_boxes, gt_boxes, iou_threshold=iou_threshold),
        "task_miou": mod._reward_miou(pred_boxes, gt_boxes),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def _aggregate_metrics(samples: list[dict[str, Any]], key: str) -> dict[str, Any]:
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_f1 = 0.0
    total_miou = 0.0
    failures = 0
    count = 0
    for sample in samples:
        record = sample[key]
        if record.get("error"):
            failures += 1
            continue
        total_tp += int(record["tp"])
        total_fp += int(record["fp"])
        total_fn += int(record["fn"])
        total_f1 += float(record["task_f1"])
        total_miou += float(record["task_miou"])
        count += 1
    if count == 0:
        return {
            "eval_f1": 0.0,
            "eval_f1_macro": 0.0,
            "eval_miou": 0.0,
            "eval_true_pos": 0,
            "eval_false_pos": 0,
            "eval_false_neg": 0,
            "eval_failures": failures,
            "eval_samples": 0,
        }
    denom = 2 * total_tp + total_fp + total_fn
    micro_f1 = 1.0 if denom == 0 else (2 * total_tp) / denom
    return {
        "eval_f1": micro_f1,
        "eval_f1_macro": total_f1 / count,
        "eval_miou": total_miou / count,
        "eval_true_pos": total_tp,
        "eval_false_pos": total_fp,
        "eval_false_neg": total_fn,
        "eval_failures": failures,
        "eval_samples": count,
    }


def main() -> None:
    env_path = SCRIPT_DIR / ".env"
    load_dotenv(env_path, override=False)

    finetune_id = os.environ.get("STATEFARM_FINETUNE_ID") or "01KFYJ3T93RST3147ANRCJ8VA2"
    checkpoint_step = int(os.environ.get("STATEFARM_CHECKPOINT_STEP") or "139")
    dataset_id = os.environ.get("STATEFARM_DATASET_ID") or DEFAULT_DATASET_ID
    split = os.environ.get("STATEFARM_SPLIT") or DEFAULT_SPLIT
    viz_limit = int(os.environ.get("STATEFARM_VIZ_LIMIT") or "20")
    api_base = os.environ.get("MOONDREAM_BASE_URL") or DEFAULT_API_BASE
    iou_threshold = float(os.environ.get("STATEFARM_IOU_THRESHOLD") or str(DEFAULT_IOU_THRESHOLD))
    timeout = float(os.environ.get("STATEFARM_TIMEOUT_SEC") or "60.0")

    api_key = _resolve_api_key_for_finetune(env_path=env_path, finetune_id=finetune_id, api_base=api_base)

    snapshot_path, rows = _load_split_rows(dataset_id=dataset_id, split=split)
    output_dir = SCRIPT_DIR / "outputs" / f"compare_{finetune_id}_step{checkpoint_step}_{split}"
    baseline_viz_dir = output_dir / "viz" / "baseline"
    checkpoint_viz_dir = output_dir / "viz" / "checkpoint" / finetune_id
    baseline_viz_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_viz_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_model = f"{DEFAULT_BASELINE_MODEL}/{finetune_id}@{checkpoint_step}"

    samples: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        image = _image_from_payload(row["image"])
        width, height = image.size
        gt_boxes = mod._parse_statefarm_boxes(row.get("answer_boxes") or row.get("answer"), width=width, height=height)
        source_name = row["image"].get("path") or f"{split}_{index:04d}.jpg"
        sample_id = Path(source_name).stem
        safe_name = _sanitize_name(sample_id)

        sample_payload: dict[str, Any] = {
            "sample_index": index,
            "sample_id": sample_id,
            "source_image_path": source_name,
            "prompt": row.get("prompt"),
            "task_type": row.get("type"),
            "ground_truth_boxes": _serialize_boxes(gt_boxes),
            "notes": row.get("notes"),
            "timestamp": row.get("timestamp"),
            "baseline": {"model": DEFAULT_BASELINE_MODEL},
            "after": {"model": checkpoint_model, "finetune_id": finetune_id, "checkpoint_step": checkpoint_step},
        }

        for label, model_name, viz_dir_key, result_key in (
            ("baseline", DEFAULT_BASELINE_MODEL, baseline_viz_dir, "baseline"),
            ("checkpoint", checkpoint_model, checkpoint_viz_dir, "after"),
        ):
            try:
                pred_boxes, _, raw_response = mod._call_detect(
                    api_base=api_base,
                    api_key=api_key,
                    model=model_name,
                    image=image,
                    object_name=mod.OBJECT_NAME,
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=256,
                    max_objects=50,
                    timeout=timeout,
                )
                metrics = _evaluate_prediction(pred_boxes, gt_boxes, iou_threshold=iou_threshold)
                metrics["raw_response"] = raw_response
                sample_payload[result_key].update(metrics)
                if index <= viz_limit:
                    viz_name = f"{index:03d}_{safe_name}_{label}.png"
                    viz_path = viz_dir_key / viz_name
                    mod._save_viz(
                        image=image,
                        gt_boxes=gt_boxes,
                        pred_boxes=pred_boxes,
                        output_path=str(viz_path),
                        iou_threshold=iou_threshold,
                    )
                    sample_payload[result_key]["viz_path"] = str(viz_path.relative_to(REPO_ROOT))
            except Exception as exc:
                sample_payload[result_key]["error"] = str(exc)
        samples.append(sample_payload)
        print(f"processed {index}/{len(rows)}: {sample_id}", flush=True)

    metrics_payload = {
        "baseline": _aggregate_metrics(samples, "baseline"),
        "checkpoint": _aggregate_metrics(samples, "after"),
        "config": {
            "dataset_id": dataset_id,
            "split": split,
            "cached_snapshot": str(snapshot_path),
            "finetune_id": finetune_id,
            "checkpoint_step": checkpoint_step,
            "baseline_model": DEFAULT_BASELINE_MODEL,
            "checkpoint_model": checkpoint_model,
            "iou_threshold": iou_threshold,
            "viz_limit": viz_limit,
            "baseline_viz_dir": str(baseline_viz_dir.relative_to(REPO_ROOT)),
            "checkpoint_viz_dir": str(checkpoint_viz_dir.relative_to(REPO_ROOT)),
        },
    }

    (output_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2) + "\n", encoding="utf-8")
    (output_dir / "sample_before_after.json").write_text(json.dumps(samples, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir.relative_to(REPO_ROOT)), **metrics_payload}, indent=2))


if __name__ == "__main__":
    main()
