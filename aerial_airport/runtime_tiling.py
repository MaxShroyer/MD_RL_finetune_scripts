from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional

from PIL import Image


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


@dataclass(frozen=True)
class TileWindow:
    row: int
    col: int
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width_norm(self) -> float:
        return float(self.x_max - self.x_min)

    @property
    def height_norm(self) -> float:
        return float(self.y_max - self.y_min)


@dataclass(frozen=True)
class Point2D:
    x: float
    y: float


@dataclass(frozen=True)
class Box2D:
    x_min: float
    y_min: float
    x_max: float
    y_max: float


def _axis_windows(size: int, *, grid_size: int, overlap: float) -> list[tuple[float, float, int, int]]:
    if grid_size <= 0:
        raise ValueError("grid_size must be > 0")
    if not (0.0 <= overlap < 1.0):
        raise ValueError("overlap must be in [0, 1)")

    tile_size_norm = 1.0 / (float(grid_size) - float(grid_size - 1) * float(overlap))
    step_norm = tile_size_norm * (1.0 - float(overlap))

    windows: list[tuple[float, float, int, int]] = []
    for index in range(grid_size):
        start_norm = step_norm * float(index)
        end_norm = start_norm + tile_size_norm
        if index == grid_size - 1:
            end_norm = 1.0
            start_norm = max(0.0, 1.0 - tile_size_norm)
        start_norm = clamp(start_norm)
        end_norm = clamp(end_norm)
        start_px = max(0, min(size, int(math.floor(start_norm * float(size)))))
        end_px = max(0, min(size, int(math.ceil(end_norm * float(size)))))
        if end_px <= start_px:
            end_px = min(size, start_px + 1)
        windows.append((start_norm, end_norm, start_px, end_px))

    if windows:
        first = windows[0]
        last = windows[-1]
        windows[0] = (0.0, first[1], 0, first[3])
        windows[-1] = (last[0], 1.0, last[2], size)
    return windows


def build_tile_windows(
    *,
    width: int,
    height: int,
    grid_size: int = 3,
    overlap: float = 0.1,
) -> list[TileWindow]:
    if width <= 0 or height <= 0:
        raise ValueError("image size must be > 0")
    x_windows = _axis_windows(width, grid_size=grid_size, overlap=overlap)
    y_windows = _axis_windows(height, grid_size=grid_size, overlap=overlap)
    windows: list[TileWindow] = []
    for row, (y_min, y_max, top, bottom) in enumerate(y_windows):
        for col, (x_min, x_max, left, right) in enumerate(x_windows):
            windows.append(
                TileWindow(
                    row=row,
                    col=col,
                    x_min=x_min,
                    y_min=y_min,
                    x_max=x_max,
                    y_max=y_max,
                    left=left,
                    top=top,
                    right=right,
                    bottom=bottom,
                )
            )
    return windows


def crop_image_to_tiles(
    image: Image.Image,
    *,
    grid_size: int = 3,
    overlap: float = 0.1,
) -> list[tuple[TileWindow, Image.Image]]:
    width, height = image.size
    out: list[tuple[TileWindow, Image.Image]] = []
    for window in build_tile_windows(width=width, height=height, grid_size=grid_size, overlap=overlap):
        out.append((window, image.crop((window.left, window.top, window.right, window.bottom))))
    return out


def map_point_from_tile(point: Point2D, window: TileWindow) -> Point2D:
    return Point2D(
        x=clamp(window.x_min + (clamp(point.x) * window.width_norm)),
        y=clamp(window.y_min + (clamp(point.y) * window.height_norm)),
    )


def map_box_from_tile(box: Box2D, window: TileWindow) -> Optional[Box2D]:
    x_min = window.x_min + (clamp(box.x_min) * window.width_norm)
    y_min = window.y_min + (clamp(box.y_min) * window.height_norm)
    x_max = window.x_min + (clamp(box.x_max) * window.width_norm)
    y_max = window.y_min + (clamp(box.y_max) * window.height_norm)
    x_min = clamp(x_min)
    y_min = clamp(y_min)
    x_max = clamp(x_max)
    y_max = clamp(y_max)
    if x_max <= x_min or y_max <= y_min:
        return None
    return Box2D(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)


def clip_box_to_window(box: Box2D, window: TileWindow) -> Optional[Box2D]:
    x_min = max(clamp(box.x_min), window.x_min)
    y_min = max(clamp(box.y_min), window.y_min)
    x_max = min(clamp(box.x_max), window.x_max)
    y_max = min(clamp(box.y_max), window.y_max)
    if x_max <= x_min or y_max <= y_min:
        return None
    return Box2D(
        x_min=clamp((x_min - window.x_min) / max(window.width_norm, 1e-12)),
        y_min=clamp((y_min - window.y_min) / max(window.height_norm, 1e-12)),
        x_max=clamp((x_max - window.x_min) / max(window.width_norm, 1e-12)),
        y_max=clamp((y_max - window.y_min) / max(window.height_norm, 1e-12)),
    )


def _point_distance(a: Point2D, b: Point2D) -> float:
    return math.hypot(float(a.x) - float(b.x), float(a.y) - float(b.y))


def merge_points(points: Iterable[Point2D], *, radius: float) -> list[Point2D]:
    if radius < 0.0:
        raise ValueError("radius must be >= 0")
    clusters: list[list[Point2D]] = []
    for point in sorted(points, key=lambda item: (float(item.x), float(item.y))):
        matched_cluster: Optional[list[Point2D]] = None
        matched_distance: Optional[float] = None
        for cluster in clusters:
            center = Point2D(
                x=sum(member.x for member in cluster) / float(len(cluster)),
                y=sum(member.y for member in cluster) / float(len(cluster)),
            )
            distance = _point_distance(point, center)
            if distance <= radius and (matched_distance is None or distance < matched_distance):
                matched_cluster = cluster
                matched_distance = distance
        if matched_cluster is None:
            clusters.append([point])
        else:
            matched_cluster.append(point)

    merged: list[Point2D] = []
    for cluster in clusters:
        merged.append(
            Point2D(
                x=clamp(sum(member.x for member in cluster) / float(len(cluster))),
                y=clamp(sum(member.y for member in cluster) / float(len(cluster))),
            )
        )
    return merged


def box_iou(a: Box2D, b: Box2D) -> float:
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


def merge_boxes(boxes: Iterable[Box2D], *, iou_threshold: float) -> list[Box2D]:
    if not (0.0 <= iou_threshold <= 1.0):
        raise ValueError("iou_threshold must be in [0, 1]")
    box_list = list(boxes)
    if not box_list:
        return []

    parents = list(range(len(box_list)))

    def _find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def _union(a_index: int, b_index: int) -> None:
        a_root = _find(a_index)
        b_root = _find(b_index)
        if a_root != b_root:
            parents[b_root] = a_root

    for left in range(len(box_list)):
        for right in range(left + 1, len(box_list)):
            if box_iou(box_list[left], box_list[right]) >= iou_threshold:
                _union(left, right)

    grouped: dict[int, list[Box2D]] = {}
    for index, box in enumerate(box_list):
        grouped.setdefault(_find(index), []).append(box)

    merged: list[Box2D] = []
    for _, cluster in sorted(grouped.items()):
        x_min = clamp(sum(box.x_min for box in cluster) / float(len(cluster)))
        y_min = clamp(sum(box.y_min for box in cluster) / float(len(cluster)))
        x_max = clamp(sum(box.x_max for box in cluster) / float(len(cluster)))
        y_max = clamp(sum(box.y_max for box in cluster) / float(len(cluster)))
        if x_max <= x_min or y_max <= y_min:
            continue
        merged.append(Box2D(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max))
    return merged
