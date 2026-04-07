from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import random
import socket
import string
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional, Sequence

import cv2
import numpy as np
from datasets import Dataset, load_dataset
from PIL import Image
from scipy.optimize import linear_sum_assignment

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover
    load_dotenv = None  # type: ignore[assignment]

from tuna_sdk import DetectAnnotation
from tuna_sdk.errors import TunaAPIError

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_ROOT = Path(__file__).resolve().parent
DEFAULT_BASE_MODEL = "moondream3-preview"
DEFAULT_STAGING_API_BASE = "https://api-staging.moondream.ai/v1"
DEFAULT_API_KEY_ENV_VAR = "CICID_GPUB_MOONDREAM_API_KEY_1"
DEFAULT_PROMPT = "crowns of a trees"
DEFAULT_HF_DATASET_REPO_ID = "maxs-m87/neon_tree_detect_v1"
DEFAULT_HF_DATASET_REVISION = "main"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
IGNORED_MODALITY_TOKENS = ("lidar", "hyperspectral", "hsi", "chm", "pointcloud")
RGB_HINT_TOKENS = ("rgb", "_image", "_image_crop", "image_crop", "image")


def repo_relative(*parts: str) -> Path:
    return MODULE_ROOT.joinpath(*parts)


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def random_suffix(length: int = 6) -> str:
    chars = string.ascii_lowercase + string.digits
    return "".join(random.choices(chars, k=length))


def resolve_config_path(raw_path: str, *, script_dir: Path) -> Path:
    path = Path(str(raw_path or "")).expanduser()
    if path.is_absolute():
        return path
    from_cwd = (Path.cwd() / path).resolve()
    if from_cwd.exists():
        return from_cwd
    from_repo = (REPO_ROOT / path).resolve()
    if from_repo.exists():
        return from_repo
    from_script = (script_dir / path).resolve()
    if from_script.exists():
        return from_script
    return from_cwd


def load_json_config(config_path: Path, *, default_path: Optional[Path] = None) -> dict[str, Any]:
    if not config_path.exists():
        if default_path is not None and config_path == default_path:
            return {}
        raise FileNotFoundError(f"Config file not found: {config_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a JSON object: {config_path}")
    return payload


def option_for_action(action: argparse.Action) -> str:
    for opt in action.option_strings:
        if opt.startswith("--"):
            return opt
    return action.option_strings[0]


def config_to_cli_args(
    parser: argparse.ArgumentParser,
    config: dict[str, Any],
    *,
    config_path: Path,
    overridden_dests: Optional[set[str]] = None,
) -> list[str]:
    overridden = set(overridden_dests or set())
    by_dest: dict[str, list[argparse.Action]] = {}
    for action in parser._actions:
        if not action.option_strings or action.dest == "help":
            continue
        by_dest.setdefault(action.dest, []).append(action)

    unknown = sorted(key for key in config if key not in by_dest)
    if unknown:
        raise ValueError(f"Unknown config key(s) in {config_path}: {unknown}")

    cli_args: list[str] = []
    for key, raw_value in config.items():
        if key in overridden:
            continue
        actions = by_dest[key]
        const_actions = [a for a in actions if isinstance(a, argparse._StoreConstAction)]
        store_actions = [a for a in actions if not isinstance(a, argparse._StoreConstAction)]

        if raw_value is None:
            matched = next((a for a in const_actions if getattr(a, "const", object()) is None), None)
            if matched is not None:
                cli_args.append(option_for_action(matched))
            continue

        if isinstance(raw_value, bool):
            matched = next((a for a in const_actions if getattr(a, "const", object()) is raw_value), None)
            if matched is not None:
                cli_args.append(option_for_action(matched))
            continue

        if not store_actions:
            continue

        action = store_actions[0]
        cli_args.append(option_for_action(action))
        if isinstance(raw_value, list):
            cli_args.extend(str(item) for item in raw_value)
        elif isinstance(raw_value, dict):
            cli_args.append(json.dumps(raw_value))
        else:
            cli_args.append(str(raw_value))
    return cli_args


def maybe_load_env_file(env_file: str) -> None:
    text = str(env_file or "").strip()
    if not text or load_dotenv is None:
        return
    load_dotenv(text, override=False)


def resolve_api_key(
    *,
    api_key: str,
    api_key_env_var: str,
    env_file: str,
) -> str:
    explicit = str(api_key or "").strip()
    if explicit:
        return explicit
    maybe_load_env_file(env_file)
    env_name = str(api_key_env_var or DEFAULT_API_KEY_ENV_VAR).strip() or DEFAULT_API_KEY_ENV_VAR
    value = str(os.environ.get(env_name) or "").strip()
    if value:
        return value
    fallback = str(os.environ.get("MOONDREAM_API_KEY") or "").strip()
    if fallback:
        return fallback
    raise ValueError(f"Missing API key. Checked explicit value, {env_name}, and MOONDREAM_API_KEY.")


def resolve_hf_token(hf_token: str, *, env_file: str) -> str:
    explicit = str(hf_token or "").strip()
    if explicit:
        return explicit
    maybe_load_env_file(env_file)
    return str(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or "").strip()


def to_data_url(image: Image.Image, *, quality: int = 92) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=max(1, min(100, int(quality))))
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def build_auth_headers(api_key: str) -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "neon-tree/0.1",
        "X-Moondream-Auth": str(api_key).strip(),
    }


@dataclass(frozen=True)
class TileWindow:
    tile_id: str
    left: int
    top: int
    right: int
    bottom: int
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width(self) -> int:
        return int(self.right - self.left)

    @property
    def height(self) -> int:
        return int(self.bottom - self.top)


@dataclass(frozen=True)
class TilingConfig:
    enabled: bool = False
    tile_width: int = 1024
    tile_height: int = 1024
    overlap_x: int = 128
    overlap_y: int = 128
    merge_iou_threshold: float = 0.5


@dataclass(frozen=True)
class DetectionTask:
    image: Image.Image
    gt_boxes: list[DetectAnnotation]
    source_image_id: str
    tile_id: str


def tiling_config_from_args(args: argparse.Namespace) -> TilingConfig:
    return TilingConfig(
        enabled=bool(getattr(args, "tiling_enabled", False)),
        tile_width=max(1, int(getattr(args, "tile_width", 1024))),
        tile_height=max(1, int(getattr(args, "tile_height", 1024))),
        overlap_x=max(0, int(getattr(args, "tile_overlap_x", 128))),
        overlap_y=max(0, int(getattr(args, "tile_overlap_y", 128))),
        merge_iou_threshold=float(getattr(args, "merge_iou_threshold", 0.5)),
    )


def _axis_starts(size: int, *, tile_size: int, overlap: int) -> list[int]:
    if tile_size <= 0:
        raise ValueError("tile_size must be > 0")
    if size <= tile_size:
        return [0]
    step = max(1, int(tile_size - overlap))
    starts = list(range(0, max(1, size - tile_size + 1), step))
    if starts[-1] != size - tile_size:
        starts.append(size - tile_size)
    return sorted(set(starts))


def _axis_scan_starts(size: int, *, window_size: int, step: int) -> list[int]:
    if window_size <= 0:
        raise ValueError("window_size must be > 0")
    if step <= 0:
        raise ValueError("step must be > 0")
    if size < window_size:
        return []
    if size == window_size:
        return [0]
    starts = list(range(0, max(1, size - window_size + 1), int(step)))
    if starts[-1] != size - window_size:
        starts.append(size - window_size)
    return sorted(set(starts))


def build_tile_windows(
    *,
    width: int,
    height: int,
    tiling: TilingConfig,
) -> list[TileWindow]:
    if width <= 0 or height <= 0:
        raise ValueError("image size must be > 0")
    if not tiling.enabled:
        return [
            TileWindow(
                tile_id="full",
                left=0,
                top=0,
                right=width,
                bottom=height,
                x_min=0.0,
                y_min=0.0,
                x_max=1.0,
                y_max=1.0,
            )
        ]
    x_starts = _axis_starts(width, tile_size=tiling.tile_width, overlap=tiling.overlap_x)
    y_starts = _axis_starts(height, tile_size=tiling.tile_height, overlap=tiling.overlap_y)
    windows: list[TileWindow] = []
    for row, top in enumerate(y_starts):
        for col, left in enumerate(x_starts):
            right = min(width, left + tiling.tile_width)
            bottom = min(height, top + tiling.tile_height)
            windows.append(
                TileWindow(
                    tile_id=f"r{row:02d}_c{col:02d}",
                    left=left,
                    top=top,
                    right=right,
                    bottom=bottom,
                    x_min=float(left) / float(width),
                    y_min=float(top) / float(height),
                    x_max=float(right) / float(width),
                    y_max=float(bottom) / float(height),
                )
            )
    return windows


def build_flyover_windows(
    *,
    width: int,
    height: int,
    window_width: int,
    window_height: int,
    step_x: int,
    step_y: int,
    path_style: str = "serpentine",
) -> list[TileWindow]:
    if width <= 0 or height <= 0:
        raise ValueError("image size must be > 0")
    x_starts = _axis_scan_starts(width, window_size=max(1, int(window_width)), step=max(1, int(step_x)))
    y_starts = _axis_scan_starts(height, window_size=max(1, int(window_height)), step=max(1, int(step_y)))
    if not x_starts or not y_starts:
        return []
    windows: list[TileWindow] = []
    frame_index = 0
    for row_index, top in enumerate(y_starts):
        ordered_x = list(x_starts)
        if str(path_style).strip() == "serpentine" and row_index % 2 == 1:
            ordered_x = list(reversed(ordered_x))
        for left in ordered_x:
            right = min(width, left + int(window_width))
            bottom = min(height, top + int(window_height))
            windows.append(
                TileWindow(
                    tile_id=f"f{frame_index:06d}",
                    left=int(left),
                    top=int(top),
                    right=int(right),
                    bottom=int(bottom),
                    x_min=float(left) / float(width),
                    y_min=float(top) / float(height),
                    x_max=float(right) / float(width),
                    y_max=float(bottom) / float(height),
                )
            )
            frame_index += 1
    return windows


def crop_image_to_tiles(image: Image.Image, *, tiling: TilingConfig) -> list[tuple[TileWindow, Image.Image]]:
    width, height = image.size
    return [(window, image.crop((window.left, window.top, window.right, window.bottom))) for window in build_tile_windows(width=width, height=height, tiling=tiling)]


def clip_box_to_window(box: DetectAnnotation, *, window: TileWindow) -> Optional[DetectAnnotation]:
    x_min = max(clamp(box.x_min), window.x_min)
    y_min = max(clamp(box.y_min), window.y_min)
    x_max = min(clamp(box.x_max), window.x_max)
    y_max = min(clamp(box.y_max), window.y_max)
    if x_max <= x_min or y_max <= y_min:
        return None
    width_norm = max(window.x_max - window.x_min, 1e-12)
    height_norm = max(window.y_max - window.y_min, 1e-12)
    return DetectAnnotation(
        x_min=clamp((x_min - window.x_min) / width_norm),
        y_min=clamp((y_min - window.y_min) / height_norm),
        x_max=clamp((x_max - window.x_min) / width_norm),
        y_max=clamp((y_max - window.y_min) / height_norm),
    )


def map_box_from_tile(box: DetectAnnotation, *, window: TileWindow) -> Optional[DetectAnnotation]:
    width_norm = max(window.x_max - window.x_min, 1e-12)
    height_norm = max(window.y_max - window.y_min, 1e-12)
    x_min = clamp(window.x_min + clamp(box.x_min) * width_norm)
    y_min = clamp(window.y_min + clamp(box.y_min) * height_norm)
    x_max = clamp(window.x_min + clamp(box.x_max) * width_norm)
    y_max = clamp(window.y_min + clamp(box.y_max) * height_norm)
    if x_max <= x_min or y_max <= y_min:
        return None
    return DetectAnnotation(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)


def serialize_answer_boxes(boxes: Sequence[DetectAnnotation]) -> str:
    payload = [
        {
            "class_name": DEFAULT_PROMPT,
            "source_class_name": "tree crown",
            "x_min": float(box.x_min),
            "y_min": float(box.y_min),
            "x_max": float(box.x_max),
            "y_max": float(box.y_max),
        }
        for box in boxes
    ]
    return json.dumps(payload, separators=(",", ":"))


def boxes_to_payload(boxes: Sequence[DetectAnnotation]) -> list[dict[str, Any]]:
    return [
        {
            "class_name": DEFAULT_PROMPT,
            "source_class_name": "tree crown",
            "x_min": float(box.x_min),
            "y_min": float(box.y_min),
            "x_max": float(box.x_max),
            "y_max": float(box.y_max),
        }
        for box in boxes
    ]


def parse_answer_boxes(value: Any) -> list[DetectAnnotation]:
    raw = value
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        raw = json.loads(text)
    if isinstance(raw, Mapping):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    boxes: list[DetectAnnotation] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        try:
            x_min = clamp(float(item["x_min"]))
            y_min = clamp(float(item["y_min"]))
            x_max = clamp(float(item["x_max"]))
            y_max = clamp(float(item["y_max"]))
        except (KeyError, TypeError, ValueError):
            continue
        if x_max <= x_min or y_max <= y_min:
            continue
        boxes.append(DetectAnnotation(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max))
    return boxes


def load_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, Mapping):
        path = value.get("path")
        if path:
            return Image.open(path).convert("RGB")
        bytes_value = value.get("bytes")
        if bytes_value is not None:
            return Image.open(io.BytesIO(bytes_value)).convert("RGB")
    if isinstance(value, (str, os.PathLike)):
        return Image.open(value).convert("RGB")
    raise ValueError(f"Unsupported image value: {type(value)!r}")


def synthetic_track_ids(source_image_id: str, count: int) -> list[str]:
    base = str(source_image_id or "").strip() or "source"
    return [f"{base}:{index}" for index in range(max(0, int(count)))]


def project_boxes_to_window(
    boxes: Sequence[DetectAnnotation],
    *,
    window: TileWindow,
    track_ids: Optional[Sequence[str]] = None,
) -> tuple[list[DetectAnnotation], list[str]]:
    ids = list(track_ids or [])
    projected_boxes: list[DetectAnnotation] = []
    projected_ids: list[str] = []
    for index, box in enumerate(boxes):
        clipped = clip_box_to_window(box, window=window)
        if clipped is None:
            continue
        projected_boxes.append(clipped)
        projected_ids.append(ids[index] if index < len(ids) else f"track_{index}")
    return projected_boxes, projected_ids


def hf_split_rows(
    *,
    repo_id: str,
    split: str,
    revision: str,
    hf_token: str,
    cache_dir: str,
) -> Dataset:
    resolved_cache_dir = str(cache_dir or "").strip()
    if not resolved_cache_dir:
        resolved_cache_dir = str(repo_relative("outputs", "hf_cache"))
    cache_root = Path(resolved_cache_dir).expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_root))
    os.environ.setdefault("HF_DATASETS_CACHE", str((cache_root / "datasets").resolve()))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str((cache_root / "hub").resolve()))
    kwargs: dict[str, Any] = {"split": split}
    if repo_id.strip() == "":
        raise ValueError("hf_dataset_repo_id is required")
    if revision.strip():
        kwargs["revision"] = revision.strip()
    if hf_token.strip():
        kwargs["token"] = hf_token.strip()
    kwargs["cache_dir"] = str(cache_root)
    try:
        return load_dataset(repo_id, **kwargs)  # type: ignore[return-value]
    except TypeError:
        token = kwargs.pop("token", None)
        if token:
            kwargs["use_auth_token"] = token
        return load_dataset(repo_id, **kwargs)  # type: ignore[return-value]


def load_synthetic_rows(
    *,
    clip_manifest: str,
    synthetic_gt_jsonl: str,
    synthetic_video: str,
) -> list[dict[str, Any]]:
    manifest_path = Path(str(clip_manifest or "")).expanduser().resolve() if str(clip_manifest or "").strip() else None
    manifest: dict[str, Any] = {}
    if manifest_path is not None:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        synthetic_gt_jsonl = synthetic_gt_jsonl or str(manifest.get("gt_jsonl_path") or "")
        synthetic_video = synthetic_video or str(manifest.get("clean_video_path") or "")
    gt_path = Path(str(synthetic_gt_jsonl or "")).expanduser().resolve()
    video_path = Path(str(synthetic_video or "")).expanduser().resolve()
    if not gt_path.exists():
        raise FileNotFoundError(f"Synthetic GT JSONL not found: {gt_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"Synthetic video not found: {video_path}")

    records = [json.loads(line) for line in gt_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open synthetic video: {video_path}")
    rows: list[dict[str, Any]] = []
    try:
        for record in records:
            ok, frame = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            clip_id = str(record.get("clip_id") or manifest.get("clip_id") or video_path.stem)
            frame_index = int(record.get("frame_index", len(rows)))
            rows.append(
                {
                    "image": Image.fromarray(frame_rgb),
                    "answer_boxes": json.dumps(record.get("gt_boxes") or [], separators=(",", ":")),
                    "source_image_id": f"{clip_id}:{frame_index:06d}",
                    "source_base_id": str(record.get("source_image_id") or clip_id),
                    "split_group_id": clip_id,
                    "synthetic_track_ids": json.dumps(record.get("gt_track_ids") or [], separators=(",", ":")),
                }
            )
    finally:
        cap.release()
    if len(rows) != len(records):
        raise RuntimeError(
            f"Synthetic video / GT mismatch: video_frames={len(rows)} gt_records={len(records)} path={video_path}"
        )
    return rows


def load_detection_rows(
    *,
    dataset_source: str,
    split: str,
    hf_dataset_repo_id: str,
    hf_dataset_revision: str,
    hf_token: str,
    hf_cache_dir: str,
    clip_manifest: str = "",
    synthetic_gt_jsonl: str = "",
    synthetic_video: str = "",
) -> list[dict[str, Any]]:
    source = str(dataset_source or "hf_hub").strip() or "hf_hub"
    if source == "hf_hub":
        dataset = hf_split_rows(
            repo_id=hf_dataset_repo_id,
            split=split,
            revision=hf_dataset_revision,
            hf_token=hf_token,
            cache_dir=hf_cache_dir,
        )
        return [dict(row) for row in iter_rows(dataset)]
    if source == "synthetic_flyover":
        return load_synthetic_rows(
            clip_manifest=clip_manifest,
            synthetic_gt_jsonl=synthetic_gt_jsonl,
            synthetic_video=synthetic_video,
        )
    raise ValueError(f"Unsupported dataset_source: {source}")


def source_image_id_from_row(row: Mapping[str, Any], *, fallback: str) -> str:
    for key in ("source_image_id", "source_base_id", "split_group_id", "row_id"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return fallback


def _rgb_image_candidate(path: Path) -> bool:
    lower = path.as_posix().lower()
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        return False
    if any(token in lower for token in IGNORED_MODALITY_TOKENS):
        return False
    return any(token in lower for token in RGB_HINT_TOKENS)


def find_rgb_images(root: Path) -> dict[str, Path]:
    candidates: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or not _rgb_image_candidate(path):
            continue
        candidates.setdefault(path.stem.lower(), path.resolve())
    return candidates


def detect_site_code(name: str) -> str:
    text = str(name or "").strip()
    if not text:
        return "unknown"
    return text.split("_", 1)[0]


@dataclass(frozen=True)
class BoxMatch:
    predicted_index: int
    ground_truth_index: int
    predicted_box: DetectAnnotation
    ground_truth_box: DetectAnnotation
    iou: float


@dataclass(frozen=True)
class MatchResult:
    matched_pairs: tuple[BoxMatch, ...]
    unmatched_pred_indices: tuple[int, ...]
    unmatched_ground_truth_indices: tuple[int, ...]
    assigned_ious: np.ndarray


@dataclass(frozen=True)
class HybridRewardBreakdown:
    reward: float
    soft_tp: float
    soft_fp: float
    soft_fn: float
    soft_fbeta: float
    loc_term: float
    matched_oversize_penalty: float
    absolute_huge_penalty: float
    collapse_penalty: float
    giant_box_rate: float
    collapse_box_rate: float


def _box_area(box: DetectAnnotation) -> float:
    return max(0.0, box.x_max - box.x_min) * max(0.0, box.y_max - box.y_min)


def _box_center(box: DetectAnnotation) -> tuple[float, float]:
    return ((box.x_min + box.x_max) / 2.0, (box.y_min + box.y_max) / 2.0)


def _box_contains_point(box: DetectAnnotation, *, x: float, y: float) -> bool:
    return box.x_min <= x <= box.x_max and box.y_min <= y <= box.y_max


def _box_iou(a: DetectAnnotation, b: DetectAnnotation) -> float:
    inter_x_min = max(a.x_min, b.x_min)
    inter_y_min = max(a.y_min, b.y_min)
    inter_x_max = min(a.x_max, b.x_max)
    inter_y_max = min(a.y_max, b.y_max)
    inter_w = max(0.0, inter_x_max - inter_x_min)
    inter_h = max(0.0, inter_y_max - inter_y_min)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0
    area_a = max(0.0, a.x_max - a.x_min) * max(0.0, a.y_max - a.y_min)
    area_b = max(0.0, b.x_max - b.x_min) * max(0.0, b.y_max - b.y_min)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def _box_ciou(a: DetectAnnotation, b: DetectAnnotation) -> float:
    iou = _box_iou(a, b)
    width_a = max(0.0, a.x_max - a.x_min)
    height_a = max(0.0, a.y_max - a.y_min)
    width_b = max(0.0, b.x_max - b.x_min)
    height_b = max(0.0, b.y_max - b.y_min)
    if width_a <= 0.0 or height_a <= 0.0 or width_b <= 0.0 or height_b <= 0.0:
        return float(iou)

    center_a_x, center_a_y = _box_center(a)
    center_b_x, center_b_y = _box_center(b)
    rho_squared = ((center_a_x - center_b_x) ** 2) + ((center_a_y - center_b_y) ** 2)

    enc_x_min = min(a.x_min, b.x_min)
    enc_y_min = min(a.y_min, b.y_min)
    enc_x_max = max(a.x_max, b.x_max)
    enc_y_max = max(a.y_max, b.y_max)
    enc_width = max(0.0, enc_x_max - enc_x_min)
    enc_height = max(0.0, enc_y_max - enc_y_min)
    c_squared = (enc_width**2) + (enc_height**2)
    if c_squared <= 0.0:
        return float(iou)

    v = (4.0 / (math.pi**2)) * (math.atan2(width_b, height_b) - math.atan2(width_a, height_a)) ** 2
    alpha_denom = 1.0 - iou + v
    alpha = 0.0 if alpha_denom <= 0.0 else v / alpha_denom
    return float(iou - ((rho_squared / c_squared) + (alpha * v)))


def match_boxes(predicted: Sequence[DetectAnnotation], ground_truth: Sequence[DetectAnnotation]) -> MatchResult:
    pred_count = len(predicted)
    gt_count = len(ground_truth)
    size = max(pred_count, gt_count)
    if size <= 0:
        return MatchResult(
            matched_pairs=(),
            unmatched_pred_indices=(),
            unmatched_ground_truth_indices=(),
            assigned_ious=np.zeros((0,), dtype=np.float32),
        )

    iou_matrix = np.zeros((size, size), dtype=np.float32)
    for pred_index, pred_box in enumerate(predicted):
        for gt_index, gt_box in enumerate(ground_truth):
            iou_matrix[pred_index, gt_index] = _box_iou(pred_box, gt_box)

    cost = 1.0 - iou_matrix
    row_idx, col_idx = linear_sum_assignment(cost)
    assigned_ious = iou_matrix[row_idx, col_idx]

    matched_pairs: list[BoxMatch] = []
    matched_pred_indices: set[int] = set()
    matched_ground_truth_indices: set[int] = set()
    for row_value, col_value in zip(row_idx.tolist(), col_idx.tolist()):
        if row_value >= pred_count or col_value >= gt_count:
            continue
        matched_pairs.append(
            BoxMatch(
                predicted_index=int(row_value),
                ground_truth_index=int(col_value),
                predicted_box=predicted[row_value],
                ground_truth_box=ground_truth[col_value],
                iou=float(iou_matrix[row_value, col_value]),
            )
        )
        matched_pred_indices.add(int(row_value))
        matched_ground_truth_indices.add(int(col_value))

    return MatchResult(
        matched_pairs=tuple(matched_pairs),
        unmatched_pred_indices=tuple(index for index in range(pred_count) if index not in matched_pred_indices),
        unmatched_ground_truth_indices=tuple(index for index in range(gt_count) if index not in matched_ground_truth_indices),
        assigned_ious=assigned_ious,
    )


def match_ious(predicted: Sequence[DetectAnnotation], ground_truth: Sequence[DetectAnnotation]) -> np.ndarray:
    return match_boxes(predicted, ground_truth).assigned_ious


def count_tp_fp_fn(
    predicted: Sequence[DetectAnnotation],
    ground_truth: Sequence[DetectAnnotation],
    *,
    iou_threshold: float = 0.5,
) -> tuple[int, int, int]:
    matches = match_ious(predicted, ground_truth)
    true_pos = int((matches >= iou_threshold).sum())
    false_pos = max(0, len(predicted) - true_pos)
    false_neg = max(0, len(ground_truth) - true_pos)
    return true_pos, false_pos, false_neg


def reward_miou(predicted: Sequence[DetectAnnotation], ground_truth: Sequence[DetectAnnotation]) -> float:
    if not predicted and not ground_truth:
        return 1.0
    if not predicted or not ground_truth:
        return 0.0
    matches = match_ious(predicted, ground_truth)
    return float(matches.mean()) if matches.size else 0.0


def reward_f1(
    predicted: Sequence[DetectAnnotation],
    ground_truth: Sequence[DetectAnnotation],
    *,
    iou_threshold: float = 0.5,
) -> float:
    tp, fp, fn = count_tp_fp_fn(predicted, ground_truth, iou_threshold=iou_threshold)
    denom = (2 * tp) + fp + fn
    return 1.0 if denom == 0 else (2 * tp) / float(denom)


def hybrid_reward_breakdown(
    predicted: Sequence[DetectAnnotation],
    ground_truth: Sequence[DetectAnnotation],
    *,
    fn_beta: float = 2.0,
    oversize_ratio_cap: float = 4.0,
    huge_area_cap: float = 0.15,
    collapse_centers_cap: float = 3.0,
) -> HybridRewardBreakdown:
    eps = 1e-9
    beta_squared = max(float(fn_beta), eps) ** 2
    ratio_cap = max(float(oversize_ratio_cap), eps)
    area_cap = max(float(huge_area_cap), eps)
    collapse_cap = max(float(collapse_centers_cap), eps)

    absolute_huge_values = [
        min(1.0, max(0.0, (_box_area(pred_box) - area_cap) / area_cap))
        for pred_box in predicted
    ]
    absolute_huge_penalty = float(sum(absolute_huge_values) / len(absolute_huge_values)) if absolute_huge_values else 0.0
    giant_box_rate = (
        float(sum(1.0 for pred_box in predicted if _box_area(pred_box) > area_cap) / len(predicted))
        if predicted
        else 0.0
    )

    if not ground_truth:
        empty_tile_reward = (
            1.0
            if not predicted
            else clamp(1.0 - (0.25 * min(len(predicted), 3)) - (0.75 * absolute_huge_penalty), 0.0, 1.0)
        )
        return HybridRewardBreakdown(
            reward=float(empty_tile_reward),
            soft_tp=0.0,
            soft_fp=float(len(predicted)),
            soft_fn=0.0,
            soft_fbeta=1.0 if not predicted else 0.0,
            loc_term=0.0,
            matched_oversize_penalty=0.0,
            absolute_huge_penalty=absolute_huge_penalty,
            collapse_penalty=0.0,
            giant_box_rate=giant_box_rate,
            collapse_box_rate=0.0,
        )

    match_result = match_boxes(predicted, ground_truth)
    soft_tp = float(sum(match.iou for match in match_result.matched_pairs))
    soft_fp = max(0.0, float(len(predicted)) - soft_tp)
    soft_fn = max(0.0, float(len(ground_truth)) - soft_tp)
    soft_fbeta_denom = ((1.0 + beta_squared) * soft_tp) + (beta_squared * soft_fn) + soft_fp
    soft_fbeta = 0.0 if soft_fbeta_denom <= 0.0 else ((1.0 + beta_squared) * soft_tp) / soft_fbeta_denom

    ciou_values = [max(_box_ciou(match.predicted_box, match.ground_truth_box), 0.0) for match in match_result.matched_pairs]
    loc_term = float(sum(ciou_values) / len(ciou_values)) if ciou_values else 0.0

    matched_oversize_values = [
        min(
            1.0,
            max(0.0, math.log2(_box_area(match.predicted_box) / ((ratio_cap * _box_area(match.ground_truth_box)) + eps))),
        )
        for match in match_result.matched_pairs
    ]
    matched_oversize_penalty = (
        float(sum(matched_oversize_values) / len(matched_oversize_values))
        if matched_oversize_values
        else 0.0
    )

    matched_ground_truth_by_pred_index = {
        match.predicted_index: match.ground_truth_index for match in match_result.matched_pairs
    }
    collapse_values: list[float] = []
    collapse_hits = 0
    for pred_index, pred_box in enumerate(predicted):
        contained_gt_centers = 0
        for gt_box in ground_truth:
            center_x, center_y = _box_center(gt_box)
            if _box_contains_point(pred_box, x=center_x, y=center_y):
                contained_gt_centers += 1

        matched_ground_truth_index = matched_ground_truth_by_pred_index.get(pred_index)
        if matched_ground_truth_index is not None:
            matched_center_x, matched_center_y = _box_center(ground_truth[matched_ground_truth_index])
            if _box_contains_point(pred_box, x=matched_center_x, y=matched_center_y):
                contained_gt_centers = max(0, contained_gt_centers - 1)

        if contained_gt_centers >= 1:
            collapse_hits += 1
        collapse_values.append(min(1.0, float(contained_gt_centers) / collapse_cap))

    collapse_penalty = float(sum(collapse_values) / len(collapse_values)) if collapse_values else 0.0
    collapse_box_rate = float(collapse_hits / len(predicted)) if predicted else 0.0

    base = (0.75 * soft_fbeta) + (0.25 * loc_term)
    reward = clamp(
        base
        * (1.0 - (0.35 * matched_oversize_penalty))
        * (1.0 - (0.35 * absolute_huge_penalty))
        * (1.0 - (0.50 * collapse_penalty)),
        0.0,
        1.0,
    )
    return HybridRewardBreakdown(
        reward=float(reward),
        soft_tp=soft_tp,
        soft_fp=soft_fp,
        soft_fn=soft_fn,
        soft_fbeta=float(soft_fbeta),
        loc_term=float(loc_term),
        matched_oversize_penalty=float(matched_oversize_penalty),
        absolute_huge_penalty=float(absolute_huge_penalty),
        collapse_penalty=float(collapse_penalty),
        giant_box_rate=float(giant_box_rate),
        collapse_box_rate=float(collapse_box_rate),
    )


def reward_hybrid(
    predicted: Sequence[DetectAnnotation],
    ground_truth: Sequence[DetectAnnotation],
    *,
    fn_beta: float = 2.0,
    oversize_ratio_cap: float = 4.0,
    huge_area_cap: float = 0.15,
    collapse_centers_cap: float = 3.0,
) -> float:
    return hybrid_reward_breakdown(
        predicted,
        ground_truth,
        fn_beta=fn_beta,
        oversize_ratio_cap=oversize_ratio_cap,
        huge_area_cap=huge_area_cap,
        collapse_centers_cap=collapse_centers_cap,
    ).reward


def merge_boxes(boxes: Sequence[DetectAnnotation], *, iou_threshold: float) -> list[DetectAnnotation]:
    if not boxes:
        return []
    parents = list(range(len(boxes)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(a_idx: int, b_idx: int) -> None:
        a_root = find(a_idx)
        b_root = find(b_idx)
        if a_root != b_root:
            parents[b_root] = a_root

    for left in range(len(boxes)):
        for right in range(left + 1, len(boxes)):
            if _box_iou(boxes[left], boxes[right]) >= iou_threshold:
                union(left, right)

    grouped: dict[int, list[DetectAnnotation]] = {}
    for index, box in enumerate(boxes):
        grouped.setdefault(find(index), []).append(box)

    merged: list[DetectAnnotation] = []
    for cluster in grouped.values():
        merged.append(
            DetectAnnotation(
                x_min=clamp(sum(item.x_min for item in cluster) / float(len(cluster))),
                y_min=clamp(sum(item.y_min for item in cluster) / float(len(cluster))),
                x_max=clamp(sum(item.x_max for item in cluster) / float(len(cluster))),
                y_max=clamp(sum(item.y_max for item in cluster) / float(len(cluster))),
            )
        )
    return [box for box in merged if box.x_max > box.x_min and box.y_max > box.y_min]


def extract_boxes_from_api_payload(payload: Mapping[str, Any]) -> list[DetectAnnotation]:
    raw_boxes = payload.get("objects")
    if raw_boxes is None and isinstance(payload.get("output"), Mapping):
        raw_boxes = payload["output"].get("objects")
    boxes: list[DetectAnnotation] = []
    for item in raw_boxes or []:
        if not isinstance(item, Mapping):
            continue
        try:
            x_min = clamp(float(item["x_min"]))
            y_min = clamp(float(item["y_min"]))
            x_max = clamp(float(item["x_max"]))
            y_max = clamp(float(item["y_max"]))
        except (KeyError, TypeError, ValueError):
            continue
        if x_max <= x_min or y_max <= y_min:
            continue
        boxes.append(DetectAnnotation(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max))
    return boxes


def call_detect_api(
    *,
    api_base: str,
    api_key: str,
    model: str,
    image: Image.Image,
    object_name: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_objects: int,
    timeout: float,
    retries: int = 2,
    retry_backoff_s: float = 5.0,
) -> list[DetectAnnotation]:
    url = api_base.rstrip("/") + "/detect"
    payload = {
        "model": model,
        "object": object_name,
        "image_url": to_data_url(image),
        "settings": {
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": int(max_tokens),
            "max_objects": int(max_objects),
        },
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=build_auth_headers(api_key),
        method="POST",
    )
    attempts = max(0, int(retries)) + 1
    last_error: Optional[Exception] = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8")
            payload_out = json.loads(body) if body else {}
            if not isinstance(payload_out, Mapping):
                return []
            return extract_boxes_from_api_payload(payload_out)
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8") if exc.fp else ""
            request_id = exc.headers.get("x-request-id") if exc.headers else None
            suffix = f" (x-request-id={request_id})" if request_id else ""
            raise RuntimeError(f"HTTP {exc.code} from {url}: {(error_body or exc.reason).strip()}{suffix}") from exc
        except (TimeoutError, socket.timeout, urllib.error.URLError) as exc:
            last_error = exc
            if attempt >= attempts - 1:
                break
            delay = max(0.0, float(retry_backoff_s)) * float(attempt + 1)
            print(
                f"detect request retry {attempt + 1}/{attempts - 1} "
                f"model={model} object={object_name} delay_s={delay:.1f}"
            )
            time.sleep(delay)
    raise RuntimeError(f"detect request failed after retries: {last_error}") from last_error


def maybe_resolve_tuna_error(exc: Exception) -> str:
    parts = [f"{type(exc).__name__}: {exc}"]
    if isinstance(exc, TunaAPIError):
        if exc.status_code is not None:
            parts.append(f"status={exc.status_code}")
        if exc.request_id:
            parts.append(f"request_id={exc.request_id}")
    return " ".join(parts)


def task_from_row(
    row: Mapping[str, Any],
    *,
    tiling: TilingConfig,
    rng: random.Random,
    include_empty_tiles: bool,
    positive_tile_probability: float = 0.8,
) -> DetectionTask:
    image = load_image(row.get("image"))
    gt_boxes = parse_answer_boxes(row.get("answer_boxes"))
    image_id = source_image_id_from_row(row, fallback="row")
    windows = build_tile_windows(width=image.width, height=image.height, tiling=tiling)
    if len(windows) == 1:
        return DetectionTask(image=image, gt_boxes=gt_boxes, source_image_id=image_id, tile_id=windows[0].tile_id)

    positive_candidates: list[DetectionTask] = []
    empty_candidates: list[DetectionTask] = []
    for window in windows:
        tile_boxes = [clipped for box in gt_boxes if (clipped := clip_box_to_window(box, window=window)) is not None]
        if tile_boxes:
            positive_candidates.append(
                DetectionTask(
                    image=image.crop((window.left, window.top, window.right, window.bottom)),
                    gt_boxes=tile_boxes,
                    source_image_id=image_id,
                    tile_id=window.tile_id,
                )
            )
        elif include_empty_tiles:
            empty_candidates.append(
                DetectionTask(
                    image=image.crop((window.left, window.top, window.right, window.bottom)),
                    gt_boxes=[],
                    source_image_id=image_id,
                    tile_id=window.tile_id,
                )
            )
    if positive_candidates and (not empty_candidates or rng.random() < positive_tile_probability):
        return rng.choice(positive_candidates)
    if empty_candidates:
        return rng.choice(empty_candidates)
    return DetectionTask(image=image, gt_boxes=[], source_image_id=image_id, tile_id="full")


def runtime_detect(
    *,
    image: Image.Image,
    model: str,
    api_base: str,
    api_key: str,
    prompt: str,
    tiling: TilingConfig,
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_objects: int,
    timeout: float,
    retries: int = 2,
    retry_backoff_s: float = 5.0,
) -> list[DetectAnnotation]:
    if not tiling.enabled:
        return call_detect_api(
            api_base=api_base,
            api_key=api_key,
            model=model,
            image=image,
            object_name=prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            max_objects=max_objects,
            timeout=timeout,
            retries=retries,
            retry_backoff_s=retry_backoff_s,
        )
    merged_boxes: list[DetectAnnotation] = []
    for window, tile_image in crop_image_to_tiles(image, tiling=tiling):
        tile_boxes = call_detect_api(
            api_base=api_base,
            api_key=api_key,
            model=model,
            image=tile_image,
            object_name=prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            max_objects=max_objects,
            timeout=timeout,
            retries=retries,
            retry_backoff_s=retry_backoff_s,
        )
        for box in tile_boxes:
            global_box = map_box_from_tile(box, window=window)
            if global_box is not None:
                merged_boxes.append(global_box)
    return merge_boxes(merged_boxes, iou_threshold=tiling.merge_iou_threshold)


def build_dataset_card(
    *,
    repo_id: str,
    metadata: Mapping[str, Any],
    stats: Mapping[str, Any],
) -> str:
    return "\n".join(
        [
            f"# {repo_id}",
            "",
            "Cleaned RGB-only derivative of the NEON TreeEvaluation benchmark for single-class tree detection.",
            "",
            "## Labeling",
            "",
            "- Public class label: `tree`",
            "- Source annotations represent tree crowns in airborne RGB imagery.",
            "",
            "## Source",
            "",
            "- Zenodo DOI: [10.5281/zenodo.5914554](https://zenodo.org/records/5914554)",
            "- Benchmark repo: [weecology/NeonTreeEvaluation](https://github.com/weecology/NeonTreeEvaluation)",
            "- License: CC BY 4.0",
            "",
            "## Splits",
            "",
            f"- Train rows: {int(stats.get('train_rows', 0))}",
            f"- Validation rows: {int(stats.get('validation_rows', 0))}",
            "",
            "## Metadata",
            "",
            "```json",
            json.dumps(dict(metadata), indent=2, sort_keys=True),
            "```",
        ]
    )


def timestamp_slug() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")


def iter_rows(dataset: Dataset) -> Iterator[dict[str, Any]]:
    for row in dataset:
        if isinstance(row, dict):
            yield dict(row)
