"""Board rendering for synthetic tic-tac-toe images."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .label_engine import BoardRecord

IMAGE_SIZE = 768

COLORWAYS = (
    "mono_black_bg",
    "mono_white_bg",
    "two_color_black_bg",
    "two_color_white_bg",
)


@dataclass(frozen=True)
class Palette:
    bg: tuple[int, int, int]
    grid: tuple[int, int, int]
    x: tuple[int, int, int]
    o: tuple[int, int, int]


PALETTES: dict[str, Palette] = {
    "mono_black_bg": Palette(bg=(0, 0, 0), grid=(235, 235, 235), x=(235, 235, 235), o=(235, 235, 235)),
    "mono_white_bg": Palette(bg=(255, 255, 255), grid=(20, 20, 20), x=(20, 20, 20), o=(20, 20, 20)),
    "two_color_black_bg": Palette(bg=(0, 0, 0), grid=(235, 235, 235), x=(220, 60, 60), o=(60, 120, 235)),
    "two_color_white_bg": Palette(bg=(255, 255, 255), grid=(20, 20, 20), x=(220, 60, 60), o=(60, 120, 235)),
}


def choose_colorway_for_state(state_key: str) -> str:
    digest = hashlib.sha1(state_key.encode("utf-8")).hexdigest()  # noqa: S324
    idx = int(digest[:8], 16) % len(COLORWAYS)
    return COLORWAYS[idx]


def _seed_int(key: str) -> int:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()  # noqa: S324
    return int(digest[:16], 16)


def _center_crop(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    if w == size and h == size:
        return img
    left = max(0, (w - size) // 2)
    top = max(0, (h - size) // 2)
    return img.crop((left, top, left + size, top + size))


def _apply_hue_shift(img: Image.Image, delta: float) -> Image.Image:
    if abs(delta) < 1e-6:
        return img
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    # RGB to HSV
    maxc = arr.max(axis=2)
    minc = arr.min(axis=2)
    v = maxc
    s = np.where(maxc == 0, 0.0, (maxc - minc) / np.clip(maxc, 1e-8, None))

    rc = (maxc - arr[:, :, 0]) / np.clip(maxc - minc, 1e-8, None)
    gc = (maxc - arr[:, :, 1]) / np.clip(maxc - minc, 1e-8, None)
    bc = (maxc - arr[:, :, 2]) / np.clip(maxc - minc, 1e-8, None)

    h = np.zeros_like(maxc)
    rmax = arr[:, :, 0] == maxc
    gmax = arr[:, :, 1] == maxc
    bmax = arr[:, :, 2] == maxc

    h = np.where(rmax, bc - gc, h)
    h = np.where(gmax, 2.0 + rc - bc, h)
    h = np.where(bmax, 4.0 + gc - rc, h)
    h = (h / 6.0) % 1.0
    h = (h + delta) % 1.0

    i = np.floor(h * 6.0).astype(np.int32)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)

    r = np.choose(i % 6, [v, q, p, p, t, v])
    g = np.choose(i % 6, [t, v, v, q, p, p])
    b = np.choose(i % 6, [p, p, t, v, v, q])

    out = np.stack([r, g, b], axis=2)
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def _augment_low(img: Image.Image, *, seed: int) -> Image.Image:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    # Mild random crop then resize back.
    scale = rng.uniform(0.965, 1.0)
    crop = int(round(IMAGE_SIZE * scale))
    max_offset = IMAGE_SIZE - crop
    ox = rng.randint(0, max(0, max_offset))
    oy = rng.randint(0, max(0, max_offset))
    img = img.crop((ox, oy, ox + crop, oy + crop)).resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.Resampling.BICUBIC)

    # Tiny rotation around center.
    angle = rng.uniform(-1.5, 1.5)
    bg = tuple(int(v) for v in np.asarray(img)[0, 0].tolist())
    img = img.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=bg)
    img = _center_crop(img, IMAGE_SIZE)

    # Tiny hue shift.
    hue_delta = rng.uniform(-0.015, 0.015)
    img = _apply_hue_shift(img, hue_delta)

    # Mild Gaussian noise.
    arr = np.asarray(img, dtype=np.float32)
    noise = np_rng.normal(loc=0.0, scale=2.2, size=arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _draw_base(record: BoardRecord, palette: Palette) -> Image.Image:
    img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=palette.bg)
    draw = ImageDraw.Draw(img)

    margin = 72
    board_min = margin
    board_max = IMAGE_SIZE - margin
    span = board_max - board_min
    step = span / 3.0

    line_w = 10
    mark_w = 24
    o_pad = 36
    x_pad = 54

    # Grid lines
    for k in (1, 2):
        x = board_min + int(round(step * k))
        y = board_min + int(round(step * k))
        draw.line((x, board_min, x, board_max), fill=palette.grid, width=line_w)
        draw.line((board_min, y, board_max, y), fill=palette.grid, width=line_w)

    for idx, cell in enumerate(record.state):
        row = idx // 3
        col = idx % 3
        x0 = int(round(board_min + col * step))
        y0 = int(round(board_min + row * step))
        x1 = int(round(board_min + (col + 1) * step))
        y1 = int(round(board_min + (row + 1) * step))

        if cell == "X":
            draw.line((x0 + x_pad, y0 + x_pad, x1 - x_pad, y1 - x_pad), fill=palette.x, width=mark_w)
            draw.line((x1 - x_pad, y0 + x_pad, x0 + x_pad, y1 - x_pad), fill=palette.x, width=mark_w)
        elif cell == "O":
            draw.ellipse((x0 + o_pad, y0 + o_pad, x1 - o_pad, y1 - o_pad), outline=palette.o, width=mark_w)

    return img


def render_board_image(
    *,
    record: BoardRecord,
    colorway: str,
    out_dir: Path,
    augmentation_profile: str,
    aug_seed: int,
) -> Path:
    if colorway not in PALETTES:
        raise ValueError(f"unknown colorway: {colorway}")

    out_dir.mkdir(parents=True, exist_ok=True)
    image_id = f"{record.state_key}_{colorway}_{augmentation_profile}_{aug_seed}"
    digest = hashlib.sha1(image_id.encode("utf-8")).hexdigest()[:12]  # noqa: S324
    filename = f"{record.state_key}_{colorway}_{digest}.png"
    out_path = out_dir / filename

    if out_path.exists():
        return out_path

    base = _draw_base(record, PALETTES[colorway])
    if augmentation_profile == "low":
        final = _augment_low(base, seed=_seed_int(f"{image_id}|low"))
    else:
        final = base

    final.save(out_path, format="PNG")
    return out_path
