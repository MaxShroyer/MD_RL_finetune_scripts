#!/usr/bin/env python3
"""
hf_to_roboflow.py

Convert one or multiple Hugging Face datasets into a YOLO-style directory and upload to Roboflow.

NEW (requested):
- roboflow project auto-population:
  * If --roboflow-project is provided => ALL datasets upload into that single Roboflow project.
  * If --roboflow-project is NOT provided => each HF dataset uploads to its OWN Roboflow project,
    auto-named from the dataset config name (cfg.name) with optional --project-prefix.
  * Project names are slugified and made unique (dataset_id hash suffix) to avoid collisions.

Assumptions / Supported annotation formats:
- Image column contains PIL.Image, dict with path/bytes, or HF Image feature.
- BBoxes are either:
  A) COCO-style absolute pixels: [x, y, width, height]          => bbox_format="coco_xywh"
  B) Pascal VOC absolute pixels: [xmin, ymin, xmax, ymax]       => bbox_format="voc_xyxy"
  C) Normalized xywh in [0..1] (top-left origin)                => bbox_format="xywh_norm"
- Labels are integers mapping to a class list, or strings.

Usage examples:

# Auto-create one Roboflow project per dataset (recommended for multiple datasets)
export ROBOFLOW_API_KEY="YOUR_KEY"
python hf_to_roboflow.py \
  --dataset "org/dsA::dsA" \
  --dataset "org/dsB::dsB" \
  --roboflow-workspace "your_workspace"

# Force everything into one Roboflow project
python hf_to_roboflow.py \
  --dataset "org/dsA::dsA" \
  --dataset "org/dsB::dsB" \
  --roboflow-workspace "your_workspace" \
  --roboflow-project "combined-project"

# Custom schema via JSON
python hf_to_roboflow.py --config-json configs.json --roboflow-workspace "your_workspace"
"""

import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from roboflow import Roboflow
from roboflow.adapters.rfapi import RoboflowError

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None


# ----------------------------
# Config
# ----------------------------

@dataclass
class DatasetConfig:
    # Required
    dataset_id: str  # e.g. "voxel51/coco-2017" or local path
    name: str        # used for output folder + roboflow naming

    # HF load_dataset args
    subset: Optional[str] = None
    revision: Optional[str] = None
    splits: Optional[List[str]] = None  # if None, auto-detect

    # Column names
    image_col: str = "image"
    ann_col: str = "annotations"
    bboxes_key: str = "bbox"
    label_key: str = "category_id"

    # If bboxes/labels are separate top-level columns (instead of ann_col)
    bboxes_col: Optional[str] = None
    labels_col: Optional[str] = None

    # bbox format: "coco_xywh" (absolute), "voc_xyxy" (absolute),
    # "xywh_norm" (normalized xywh), "voc_xyxy_norm" (normalized xyxy)
    bbox_format: str = "coco_xywh"

    # Optional class names (index -> name). If not given, will try to infer.
    class_names: Optional[List[str]] = None

    # Optional: keep only these class ids (ints)
    keep_class_ids: Optional[List[int]] = None


# ----------------------------
# Helpers
# ----------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def reset_dir(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def infer_splits(ds_dict) -> List[str]:
    keys = list(ds_dict.keys())
    preferred = [k for k in ["train", "validation", "valid", "test"] if k in keys]
    return preferred if preferred else keys


def load_image(example_img: Any) -> Image.Image:
    """
    HF 'Image' feature usually returns PIL.Image already.
    Sometimes it's a dict with path/bytes.
    """
    if isinstance(example_img, Image.Image):
        return example_img.convert("RGB")

    if isinstance(example_img, dict):
        if "path" in example_img and example_img["path"]:
            return Image.open(example_img["path"]).convert("RGB")
        if "bytes" in example_img and example_img["bytes"]:
            from io import BytesIO
            return Image.open(BytesIO(example_img["bytes"])).convert("RGB")

    raise ValueError(f"Unsupported image type: {type(example_img)}")


def coco_xywh_to_yolo(x: float, y: float, w: float, h: float, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    x_center = (x + w / 2.0) / img_w
    y_center = (y + h / 2.0) / img_h
    w_norm = w / img_w
    h_norm = h / img_h
    return x_center, y_center, w_norm, h_norm


def voc_xyxy_to_yolo(xmin: float, ymin: float, xmax: float, ymax: float, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    w = max(0.0, xmax - xmin)
    h = max(0.0, ymax - ymin)
    x_center = (xmin + w / 2.0) / img_w
    y_center = (ymin + h / 2.0) / img_h
    return x_center, y_center, w / img_w, h / img_h


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def to_yolo_bbox(bbox: List[float], fmt: str, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    if fmt == "coco_xywh":
        x, y, w, h = bbox
        xc, yc, wn, hn = coco_xywh_to_yolo(x, y, w, h, img_w, img_h)
        return clamp01(xc), clamp01(yc), clamp01(wn), clamp01(hn)
    elif fmt == "voc_xyxy":
        xmin, ymin, xmax, ymax = bbox
        xc, yc, wn, hn = voc_xyxy_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h)
        return clamp01(xc), clamp01(yc), clamp01(wn), clamp01(hn)
    elif fmt == "xywh_norm":
        x, y, w, h = bbox
        xc = x + w / 2.0
        yc = y + h / 2.0
        return clamp01(xc), clamp01(yc), clamp01(w), clamp01(h)
    elif fmt == "voc_xyxy_norm":
        xmin, ymin, xmax, ymax = bbox
        w = max(0.0, xmax - xmin)
        h = max(0.0, ymax - ymin)
        xc = xmin + w / 2.0
        yc = ymin + h / 2.0
        return clamp01(xc), clamp01(yc), clamp01(w), clamp01(h)
    else:
        raise ValueError(f"Unknown bbox_format: {fmt}")


def maybe_parse_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text.lower() in {"null", "none", "nan"}:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def extract_anns(example: Dict[str, Any], cfg: DatasetConfig) -> Tuple[List[List[float]], List[Union[int, str]], Optional[str]]:
    """
    Returns (bboxes, labels, bbox_format_override).
    Supports:
    - Separate top-level columns (bboxes_col, labels_col)
    - Single ann_col containing list[dict] or dict of arrays
    - StateFarm-style `answer_boxes` JSON strings with normalized xyxy coords
    """
    if cfg.bboxes_col and cfg.labels_col:
        return example.get(cfg.bboxes_col, []) or [], example.get(cfg.labels_col, []) or [], None

    anns = example.get(cfg.ann_col, None)
    anns = maybe_parse_json(anns)
    if anns is None:
        anns = None

    if isinstance(anns, dict):
        b = anns.get(cfg.bboxes_key, []) or []
        l = anns.get(cfg.label_key, []) or []
        return b, l, None

    if isinstance(anns, list):
        bboxes, labels = [], []
        for a in anns:
            if not isinstance(a, dict):
                continue
            if cfg.bboxes_key not in a or cfg.label_key not in a:
                continue
            bboxes.append(a[cfg.bboxes_key])
            labels.append(a[cfg.label_key])
        return bboxes, labels, None

    # Heuristic fallback for datasets with "answer_boxes" json and prompt labels.
    answer_boxes = maybe_parse_json(example.get("answer_boxes"))
    if isinstance(answer_boxes, dict):
        answer_boxes = [answer_boxes]
    if isinstance(answer_boxes, list):
        bboxes: List[List[float]] = []
        labels: List[Union[int, str]] = []
        label_value = (
            example.get("prompt")
            or example.get("label")
            or example.get("type")
            or cfg.name
            or "object"
        )
        label_text = str(label_value).strip() if label_value is not None else "object"
        if not label_text:
            label_text = "object"

        for box in answer_boxes:
            if not isinstance(box, dict):
                continue
            x_min = box.get("x_min", box.get("xmin"))
            y_min = box.get("y_min", box.get("ymin"))
            x_max = box.get("x_max", box.get("xmax"))
            y_max = box.get("y_max", box.get("ymax"))
            if x_min is None or y_min is None or x_max is None or y_max is None:
                continue
            try:
                bboxes.append([float(x_min), float(y_min), float(x_max), float(y_max)])
                labels.append(label_text)
            except (TypeError, ValueError):
                continue
        return bboxes, labels, "voc_xyxy_norm"

    return [], [], None


def write_yolo_label_file(label_path: str, lines: List[str]) -> None:
    with open(label_path, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln.rstrip() + "\n")


def infer_class_names(ds_dict, splits: List[str]) -> Optional[List[str]]:
    """
    Heuristic: find a ClassLabel feature anywhere.
    """
    try:
        for sp in splits:
            features = ds_dict[sp].features
            for _, feat in features.items():
                if hasattr(feat, "names") and isinstance(feat.names, list) and feat.names:
                    return list(feat.names)
    except Exception:
        pass
    return None


def export_dataset_to_yolo(cfg: DatasetConfig, out_root: str) -> Dict[str, Any]:
    """
    Export HF dataset into:
      out_root/cfg.name/{split}/images/*.jpg
      out_root/cfg.name/{split}/labels/*.txt
    """
    ds = load_dataset(cfg.dataset_id, cfg.subset, revision=cfg.revision)
    splits = cfg.splits or infer_splits(ds)

    dataset_out = os.path.join(out_root, cfg.name)
    ensure_dir(dataset_out)

    class_names = cfg.class_names or infer_class_names(ds, splits) or []

    split_dirs = {}
    for sp in splits:
        d = ds[sp]
        images_dir = os.path.join(dataset_out, sp, "images")
        labels_dir = os.path.join(dataset_out, sp, "labels")
        reset_dir(images_dir)
        reset_dir(labels_dir)
        split_dirs[sp] = os.path.join(dataset_out, sp)

        for i, ex in enumerate(tqdm(d, desc=f"{cfg.name}:{sp}", unit="ex")):
            img = load_image(ex[cfg.image_col])
            img_w, img_h = img.size

            img_filename = f"{cfg.name}_{sp}_{i:07d}.jpg"
            img_path = os.path.join(images_dir, img_filename)
            img.save(img_path, quality=95)

            bboxes, labels, bbox_format_override = extract_anns(ex, cfg)
            yolo_lines = []
            bbox_format = bbox_format_override or cfg.bbox_format

            if bboxes and labels and len(bboxes) == len(labels):
                for bb, lab in zip(bboxes, labels):
                    if bb is None or lab is None:
                        continue

                    # Optional filter
                    if cfg.keep_class_ids is not None and isinstance(lab, int) and lab not in cfg.keep_class_ids:
                        continue

                    # Map string labels to indices
                    if isinstance(lab, str):
                        if lab not in class_names:
                            class_names.append(lab)
                        cls_id = class_names.index(lab)
                    else:
                        cls_id = int(lab)

                    xc, yc, wn, hn = to_yolo_bbox(bb, bbox_format, img_w, img_h)
                    if wn <= 0 or hn <= 0:
                        continue
                    yolo_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

            if yolo_lines:
                label_filename = os.path.splitext(img_filename)[0] + ".txt"
                label_path = os.path.join(labels_dir, label_filename)
                write_yolo_label_file(label_path, yolo_lines)

    # Write YOLO yaml for convenience
    yaml_path = os.path.join(dataset_out, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {dataset_out}\n")
        if "train" in splits:
            f.write("train: train/images\n")
        if "validation" in splits:
            f.write("val: validation/images\n")
        if "valid" in splits:
            f.write("val: valid/images\n")
        if "test" in splits:
            f.write("test: test/images\n")
        f.write(f"nc: {len(class_names)}\n")
        f.write("names:\n")
        for n in class_names:
            f.write(f"  - {n}\n")

    return {
        "dataset_out": dataset_out,
        "splits": splits,
        "split_dirs": split_dirs,
        "class_names": class_names,
        "yaml_path": yaml_path,
    }


def slugify_project_name(name: str, max_len: int = 50) -> str:
    """
    Make a safe project slug from arbitrary names.
    """
    s = name.strip().lower()
    s = s.replace("/", "-").replace(":", "-").replace("_", "-")
    s = re.sub(r"[^a-z0-9\-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    if not s:
        s = "hf-dataset"
    return s[:max_len]


def short_hash(text: str, n: int = 8) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def get_or_create_project(
    ws,
    project_name: str,
    project_type: str = "object-detection",
    project_license: str = "MIT",
    annotation: Optional[str] = None,
):
    """
    Get existing Roboflow project or create it.
    """
    try:
        return ws.project(project_name)
    except RoboflowError:
        return ws.create_project(
            project_name=project_name,
            project_type=project_type,
            project_license=project_license,
            annotation=annotation or project_name,
        )


def upload_dataset(
    rf_api_key: str,
    workspace: str,
    project_name: str,
    dataset_dir: str,
    version_name: str,
    project_type: str,
    project_license: str,
    annotation: Optional[str],
) -> None:
    rf = Roboflow(api_key=rf_api_key)
    ws = rf.workspace(workspace)

    # Ensure project exists with desired metadata before dataset upload.
    proj = get_or_create_project(
        ws,
        project_name,
        project_type=project_type,
        project_license=project_license,
        annotation=annotation,
    )
    actual_project_name = project_name
    project_id = getattr(proj, "id", "")
    if isinstance(project_id, str) and project_id:
        actual_project_name = project_id.split("/", 1)[-1]

    print(f"Uploading '{dataset_dir}' -> {workspace}/{actual_project_name} (version '{version_name}')")
    # Use workspace-level dataset upload for recursive image/annotation folder parsing.
    ws.upload_dataset(
        dataset_path=dataset_dir,
        project_name=actual_project_name,
        num_workers=8,
        project_license=project_license,
        project_type=project_type,
        batch_name=version_name,
        num_retries=0,
        is_prediction=False,
    )
    print("Upload complete.")


def parse_configs_from_args(args) -> List[DatasetConfig]:
    cfgs: List[DatasetConfig] = []
    if args.config_json:
        with open(args.config_json, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            raw = [raw]
        for item in raw:
            cfgs.append(DatasetConfig(**item))
        return cfgs

    for d in args.dataset:
        if "::" in d:
            dataset_id, name = d.split("::", 1)
        else:
            dataset_id = d
            name = d.replace("/", "_").replace(":", "_")
        cfgs.append(DatasetConfig(dataset_id=dataset_id, name=name))
    return cfgs


# ----------------------------
# Main
# ----------------------------

def main():
    if load_dotenv is not None:
        # Prefer a .env adjacent to this script, but also allow running from within MDstatefarmRL/.
        candidates = [
            Path(__file__).resolve().parent / ".env",
            Path.cwd() / ".env",
        ]
        for env_path in candidates:
            if env_path.exists():
                load_dotenv(dotenv_path=env_path, override=False)

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        action="append",
        default=[
            "maxs-m87/Ball-Holder-splits-v1"
            # "maxs-m87/NBA_StateFarm_splits_half",
            # "maxs-m87/NBA_StateFarm_splits_p1",
            # "maxs-m87/NBA_StateFarm_Splits_01"
            ],#"maxs-m87/NBA_StateFarm_splits_1-4th",
            #"maxs-m87/NBA_StateFarm_splits_1-8th",
            #"maxs-m87/NBA_StateFarm_splits_2-3rds",

        help='Repeatable. "dataset_id" or "dataset_id::name". Example: --dataset "voxel51/coco-2017::coco17"',
    )
    ap.add_argument(
        "--config-json",
        type=str,
        default=None,
        help="Path to a JSON file defining one or more DatasetConfig objects for custom schemas.",
    )
    ap.add_argument("--out", type=str, default="out_hf_yolo", help="Output root folder.")
    ap.add_argument("--roboflow-workspace", type=str, required=True)

    # OPTIONAL (auto-populate when omitted)
    ap.add_argument(
        "--roboflow-project",
        type=str,
        default=None,
        help="Optional. If set, uploads ALL datasets into this single project.",
    )

    ap.add_argument(
        "--project-prefix",
        type=str,
        default="",
        help="Optional prefix for auto-created project names (e.g. 'hf-').",
    )

    ap.add_argument(
        "--project-type",
        type=str,
        default="object-detection",
        help='Roboflow project type: "object-detection", "classification", "instance-segmentation"',
    )

    ap.add_argument(
        "--project-license",
        type=str,
        default="MIT",
        help='Roboflow project license (e.g. "MIT", "Private").',
    )
    ap.add_argument(
        "--project-annotation",
        type=str,
        default=None,
        help="Roboflow project 'annotation' field used on creation (defaults to the project name).",
    )

    ap.add_argument("--version-prefix", type=str, default="hf_import", help="Prefix for Roboflow version/batch name.")
    ap.add_argument("--skip-upload", action="store_true", help="Export only; do not upload.")
    args = ap.parse_args()

    rf_key = os.environ.get("ROBOFLOW_API_KEY", "")
    if not args.skip_upload and not rf_key:
        raise SystemExit("ROBOFLOW_API_KEY is not set in environment.")

    cfgs = parse_configs_from_args(args)
    if not cfgs:
        raise SystemExit("No datasets provided. Use --dataset or --config-json.")

    ensure_dir(args.out)

    for cfg in cfgs:
        meta = export_dataset_to_yolo(cfg, args.out)

        # Version name per dataset (uploaded version/batch)
        version_name = f"{args.version_prefix}_{slugify_project_name(cfg.name)}"

        # Auto project naming:
        # - if user forces a single project name, use it
        # - else create one project per dataset, with uniqueness suffix
        if args.roboflow_project:
            project_name = slugify_project_name(args.roboflow_project)
        else:
            # avoid collisions by appending short hash from dataset_id
            base = slugify_project_name(f"{args.project_prefix}{cfg.name}")
            project_name = f"{base}-{short_hash(cfg.dataset_id)}"

        print("\n==============================")
        print(f"HF dataset_id : {cfg.dataset_id}")
        print(f"Local export  : {meta['dataset_out']}")
        print(f"Roboflow proj : {args.roboflow_workspace}/{project_name}")
        print(f"Upload version: {version_name}")
        print(f"Classes       : {len(meta['class_names'])}")
        print("==============================\n")

        if not args.skip_upload:
            upload_dataset(
                rf_api_key=rf_key,
                workspace=args.roboflow_workspace,
                project_name=project_name,
                dataset_dir=meta["dataset_out"],
                version_name=version_name,
                project_type=args.project_type,
                project_license=args.project_license,
                annotation=args.project_annotation,
            )


if __name__ == "__main__":
    main()
