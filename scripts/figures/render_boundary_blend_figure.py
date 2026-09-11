#!/usr/bin/env python3
"""Render a boundary-aware latent blending explainer from a Carnator mask bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = ROOT / "artifacts/static-prompt-defaults-results/ceramic-mug/frames/four_images/start_original.png"
DEFAULT_MASK = ROOT / "artifacts/static-prompt-defaults/ceramic-mug/cache/source_dense/fusion_masks/masks.pt"
DEFAULT_OUTPUT = ROOT / "artifacts/static-prompt-defaults-results/ceramic-mug/boundary_blend_principle.png"
DEFAULT_MASK_OVERLAP_OUTPUT = ROOT / "artifacts/static-prompt-defaults-results/ceramic-mug/boundary_blend_mask_overlap.png"
DEFAULT_TRANSITION_GRID_OUTPUT = ROOT / "artifacts/static-prompt-defaults-results/ceramic-mug/boundary_blend_transition_pixels.png"

BACKGROUND = (248, 250, 252)
PANEL = (255, 255, 255)
INK = (22, 32, 43)
MUTED = (89, 103, 120)
BORDER = (215, 223, 232)
TARGET = (224, 72, 77)
SOURCE = (29, 141, 154)
BAND = (245, 158, 11)
EMPTY = (233, 238, 243)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--mask", type=Path, default=DEFAULT_MASK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mask-overlap-output", type=Path, default=DEFAULT_MASK_OVERLAP_OUTPUT)
    parser.add_argument("--transition-grid-output", type=Path, default=DEFAULT_TRANSITION_GRID_OUTPUT)
    parser.add_argument("--time-index", type=int, default=0)
    return parser.parse_args()


def font(size: int, *, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    if mono:
        name = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
    elif bold:
        name = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    else:
        name = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    return ImageFont.truetype(name, size)


def rounded_panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    draw.rounded_rectangle(box, radius=18, fill=PANEL, outline=BORDER, width=3)


def text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, size: int, *, color=INK, bold=False, mono=False) -> None:
    draw.text(xy, value, fill=color, font=font(size, bold=bold, mono=mono))


def fit_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    scale = max(size[0] / image.width, size[1] / image.height)
    resized = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
    left = (resized.width - size[0]) // 2
    top = (resized.height - size[1]) // 2
    return resized.crop((left, top, left + size[0], top + size[1]))


def expand_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    height, width = size[1], size[0]
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    return np.asarray(image.resize((width, height), Image.Resampling.NEAREST)) > 0


def blend(base: tuple[int, int, int], overlay: tuple[int, int, int], alpha: float) -> tuple[int, int, int]:
    return tuple(round((1.0 - alpha) * base[index] + alpha * overlay[index]) for index in range(3))


def smoothstep_alpha(mbase: np.ndarray, medit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Match Carnator's per-frame Chebyshev-distance source-weight calculation."""
    band = medit & ~mbase
    distance = np.zeros_like(mbase, dtype=np.float32)
    remaining = band.copy()
    frontier = mbase.copy()
    height, width = mbase.shape
    for step in range(1, max(height, width) + 1):
        padded = np.pad(frontier, 1, mode="constant")
        next_frontier = np.zeros_like(frontier)
        for dy in range(3):
            for dx in range(3):
                next_frontier |= padded[dy : dy + height, dx : dx + width]
        frontier = next_frontier
        newly = frontier & remaining
        distance[newly] = step
        remaining &= ~newly
        if not remaining.any():
            break
    max_distance = float(distance[band].max()) if band.any() else 1.0
    normalized = (distance - 1.0) / max(max_distance - 1.0, 1.0)
    normalized = np.clip(normalized, 0.0, 1.0)
    if max_distance == 1.0:
        normalized[band] = 0.5
    alpha = normalized * normalized * (3.0 - 2.0 * normalized)
    alpha[~band] = 0.0
    return alpha, band


def overlay_masks(frame: Image.Image, mbase: np.ndarray, band: np.ndarray) -> Image.Image:
    canvas = np.asarray(frame.convert("RGB")).copy()
    base_pixels = expand_mask(mbase, frame.size)
    band_pixels = expand_mask(band, frame.size)
    canvas[band_pixels] = np.asarray([blend(tuple(pixel), BAND, 0.48) for pixel in canvas[band_pixels]], dtype=np.uint8)
    canvas[base_pixels] = np.asarray([blend(tuple(pixel), TARGET, 0.50) for pixel in canvas[base_pixels]], dtype=np.uint8)
    return Image.fromarray(canvas)


def overlay_mask_overlap(frame: Image.Image, mbase: np.ndarray, medit: np.ndarray) -> Image.Image:
    """Render the two canonical masks in their native nested order, without labels."""
    canvas = np.asarray(frame.convert("RGB")).copy()
    medit_pixels = expand_mask(medit, frame.size)
    mbase_pixels = expand_mask(mbase, frame.size)
    canvas[medit_pixels] = np.asarray([blend(tuple(pixel), BAND, 0.42) for pixel in canvas[medit_pixels]], dtype=np.uint8)
    canvas[mbase_pixels] = np.asarray([blend(tuple(pixel), TARGET, 0.55) for pixel in canvas[mbase_pixels]], dtype=np.uint8)
    return Image.fromarray(canvas)


def render_transition_pixels(mbase: np.ndarray, medit: np.ndarray, _alpha: np.ndarray, size: tuple[int, int]) -> Image.Image:
    """Render a smooth, grid-aligned visualization of the real Mbase--Medit band.

    The masks retain their exact 22 x 40 grid placement.  The color field is
    evaluated between their fixed raster boundaries, so it makes the narrow
    transition visible without expanding either mask.
    """
    from scipy.ndimage import distance_transform_cdt

    width, height = size
    band = medit & ~mbase
    mbase_pixels = expand_mask(mbase, size)
    medit_pixels = expand_mask(medit, size)
    band_pixels = medit_pixels & ~mbase_pixels
    # At a point in Mband, the ratio of the chessboard distances to its inner
    # and outer boundaries gives a continuous 0-to-1 coordinate through band.
    to_base = distance_transform_cdt(~mbase_pixels, metric="chessboard").astype(np.float32)
    to_exterior = distance_transform_cdt(medit_pixels, metric="chessboard").astype(np.float32)
    progress = (to_base - 0.5) / np.maximum(to_base + to_exterior - 1.0, 1.0)
    progress = np.clip(progress, 0.0, 1.0)
    smooth_weight = progress * progress * (3.0 - 2.0 * progress)

    pixels = np.full((height, width, 3), EMPTY, dtype=np.uint8)
    # The amber midpoint prevents the target/source colors from turning gray
    # and makes the boundary's interpolation visually legible at a glance.
    lower = smooth_weight[..., None] * 2.0
    upper = (smooth_weight[..., None] - 0.5) * 2.0
    target_to_band = np.asarray(TARGET, dtype=np.float32)[None, None, :] * (1.0 - lower) + np.asarray(BAND, dtype=np.float32)[None, None, :] * lower
    band_to_source = np.asarray(BAND, dtype=np.float32)[None, None, :] * (1.0 - upper) + np.asarray(SOURCE, dtype=np.float32)[None, None, :] * upper
    interpolated = np.where((smooth_weight[..., None] <= 0.5), target_to_band, band_to_source).round().astype(np.uint8)
    pixels[band_pixels] = interpolated[band_pixels]
    pixels[mbase_pixels] = TARGET

    grid = Image.fromarray(pixels)
    draw = ImageDraw.Draw(grid, "RGBA")
    token_height, token_width = mbase.shape
    for x in range(0, width + 1, width // token_width):
        draw.line((x, 0, x, height), fill=(255, 255, 255, 116), width=1)
    for y in range(0, height + 1, height // token_height):
        draw.line((0, y, width, y), fill=(255, 255, 255, 116), width=1)
    return grid.convert("RGB")


def paste_image_with_border(canvas: Image.Image, image: Image.Image, xy: tuple[int, int]) -> None:
    canvas.paste(image, xy)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((xy[0], xy[1], xy[0] + image.width, xy[1] + image.height), outline=BORDER, width=2)


def draw_legend(draw: ImageDraw.ImageDraw, x: int, y: int, color: tuple[int, int, int], title: str, detail: str) -> None:
    draw.rounded_rectangle((x, y + 3, x + 26, y + 29), radius=5, fill=color)
    text(draw, (x + 42, y), title, 25, bold=True)
    text(draw, (x + 42, y + 33), detail, 20, color=MUTED)


def draw_grid(draw: ImageDraw.ImageDraw, x: int, y: int, mbase: np.ndarray, band: np.ndarray, alpha: np.ndarray) -> tuple[int, int]:
    ys, xs = np.where(mbase | band)
    crop = (slice(max(0, ys.min() - 2), min(mbase.shape[0], ys.max() + 3)), slice(max(0, xs.min() - 2), min(mbase.shape[1], xs.max() + 3)))
    core = mbase[crop]
    transition = band[crop]
    source_weight = alpha[crop]
    cell = 31
    rows, cols = core.shape
    for row in range(rows):
        for col in range(cols):
            if core[row, col]:
                color = TARGET
            elif transition[row, col]:
                color = blend(TARGET, SOURCE, float(source_weight[row, col]))
            else:
                color = EMPTY
            left = x + col * cell
            top = y + row * cell
            draw.rectangle((left, top, left + cell - 2, top + cell - 2), fill=color, outline=(255, 255, 255), width=1)
    return cols * cell, rows * cell


def draw_alpha_strip(draw: ImageDraw.ImageDraw, x: int, y: int, width: int) -> None:
    text(draw, (x, y), "Source weight inside Mband", 24, bold=True)
    y += 45
    steps = 7
    step_width = width // steps
    for index in range(steps):
        alpha = index / (steps - 1)
        color = blend(TARGET, SOURCE, alpha)
        left = x + index * step_width
        draw.rounded_rectangle((left, y, left + step_width - 8, y + 52), radius=8, fill=color)
        text(draw, (left + 4, y + 64), f"{alpha:.1f}", 17, color=MUTED)
    draw.line((x + 10, y + 130, x + width - 10, y + 130), fill=BORDER, width=3)
    points = []
    for index in range(61):
        value = index / 60
        smooth = value * value * (3 - 2 * value)
        px = x + 10 + round((width - 20) * value)
        py = y + 130 - round(80 * smooth)
        points.append((px, py))
    draw.line(points, fill=SOURCE, width=5, joint="curve")
    text(draw, (x, y + 148), "alpha = smoothstep(distance to Mbase)", 20, color=MUTED)


def draw_latent_stack(draw: ImageDraw.ImageDraw, x: int, y: int, color: tuple[int, int, int], label: str, symbol: str) -> None:
    for offset in (28, 14, 0):
        shade = blend(color, (255, 255, 255), offset / 85)
        draw.rounded_rectangle((x + offset, y + offset, x + offset + 260, y + offset + 150), radius=10, fill=shade, outline=blend(color, INK, 0.15), width=3)
        for row in range(3):
            for col in range(6):
                left = x + offset + 22 + col * 36
                top = y + offset + 25 + row * 34
                draw.rounded_rectangle((left, top, left + 21, top + 19), radius=3, fill=blend(shade, (255, 255, 255), 0.40))
    text(draw, (x + 20, y + 205), symbol, 32, bold=True, color=color)
    text(draw, (x + 20, y + 246), label, 22, color=MUTED)


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], color: tuple[int, int, int] = MUTED) -> None:
    draw.line((start, end), fill=color, width=5)
    x, y = end
    draw.polygon([(x, y), (x - 15, y - 8), (x - 15, y + 8)], fill=color)


def draw_region_table(draw: ImageDraw.ImageDraw, x: int, y: int, width: int) -> None:
    rows = [
        (TARGET, "Mbase", "Use target latent", "alpha_source = 0"),
        (BAND, "Mband", "Blend target and source", "0 < alpha_source < 1"),
        (SOURCE, "outside Medit", "Reuse source cache", "alpha_source = 1"),
    ]
    row_height = 96
    draw.rounded_rectangle((x, y, x + width, y + row_height * 3), radius=12, outline=BORDER, width=2, fill=(252, 253, 254))
    for index, (color, region, action, weight) in enumerate(rows):
        top = y + index * row_height
        if index:
            draw.line((x, top, x + width, top), fill=BORDER, width=2)
        draw.rounded_rectangle((x + 18, top + 29, x + 44, top + 55), radius=5, fill=color)
        text(draw, (x + 62, top + 16), region, 21, bold=True)
        text(draw, (x + 310, top + 16), action, 22)
        text(draw, (x + 310, top + 49), weight, 19, color=MUTED)


def render(args: argparse.Namespace) -> None:
    frame = Image.open(args.image).convert("RGB")
    masks = torch.load(args.mask, map_location="cpu", weights_only=True)
    mbase_all = masks["Mbase"].bool().numpy()
    medit_all = masks["Medit"].bool().numpy()
    if not 0 <= args.time_index < mbase_all.shape[0]:
        raise ValueError(f"time index must be in [0, {mbase_all.shape[0] - 1}]")
    mbase = mbase_all[args.time_index]
    medit = medit_all[args.time_index]
    alpha, band = smoothstep_alpha(mbase, medit)

    canvas = Image.new("RGB", (3200, 1660), BACKGROUND)
    draw = ImageDraw.Draw(canvas)

    text(draw, (92, 64), "Boundary-aware Latent Blending", 58, bold=True)
    text(draw, (95, 143), "Carnator illustration: preserve stable source structure, re-generate the edit core, and blend only at the boundary.", 27, color=MUTED)

    left = (70, 245, 1050, 1550)
    middle = (1090, 245, 2040, 1550)
    right = (2080, 245, 3130, 1550)
    for panel in (left, middle, right):
        rounded_panel(draw, panel)

    # Panel A: real frame plus the exact core/band regions.
    text(draw, (110, 290), "a", 33, bold=True, color=SOURCE)
    text(draw, (160, 294), "Masks define three latent regions", 32, bold=True)
    text(draw, (110, 342), "Real mask geometry projected onto the first video frame", 22, color=MUTED)
    image_size = (900, 495)
    image_x, image_y = 110, 410
    mask_overlay = overlay_masks(fit_image(frame, image_size), mbase, band)
    paste_image_with_border(canvas, mask_overlay, (image_x, image_y))
    draw.line((650, 705, 780, 615), fill=TARGET, width=4)
    draw.rounded_rectangle((770, 570, 985, 632), radius=10, fill=(255, 255, 255), outline=TARGET, width=2)
    text(draw, (790, 584), "Mbase: edit core", 19, color=TARGET, bold=True)
    draw.line((650, 695, 785, 792), fill=BAND, width=4)
    draw.rounded_rectangle((775, 765, 992, 827), radius=10, fill=(255, 255, 255), outline=BAND, width=2)
    text(draw, (792, 779), "Mband: transition", 19, color=(174, 111, 0), bold=True)
    draw_legend(draw, 135, 1000, TARGET, "Mbase", "Target latent only")
    draw_legend(draw, 135, 1112, BAND, "Mband = Medit - Mbase", "Distance-weighted target/source blend")
    draw_legend(draw, 135, 1224, SOURCE, "1 - Medit", "Source cache only")
    text(draw, (135, 1360), "The displayed masks are binary.  The blend weight is", 22, color=MUTED)
    text(draw, (135, 1394), "computed later on the latent grid, only inside Mband.", 22, color=MUTED)
    text(draw, (135, 1467), f"t = 0: Mbase {mbase.mean() * 100:.2f}%   |   Medit {medit.mean() * 100:.2f}%   |   Mband {band.mean() * 100:.2f}%", 20, color=INK, mono=True)

    # Panel B: true token-grid crop and smoothstep source weights.
    text(draw, (1130, 290), "b", 33, bold=True, color=SOURCE)
    text(draw, (1180, 294), "Distance-weighted boundary band", 32, bold=True)
    text(draw, (1130, 342), "A crop of the actual 22 x 40 token grid at t = 0", 22, color=MUTED)
    grid_width, grid_height = draw_grid(draw, 1195, 410, mbase, band, alpha)
    text(draw, (1195, 430 + grid_height), "Red: target latent   |   Teal: source contribution grows outward", 20, color=MUTED)
    draw_alpha_strip(draw, 1195, 1075, 690)
    text(draw, (1195, 1370), "Chebyshev distance is measured from Mbase on the latent grid.", 21, color=MUTED)
    text(draw, (1195, 1406), "The step-5 mask and its alpha field remain fixed for each injection.", 21, color=MUTED)

    # Panel C: source and target latent inputs mapped to the three output rules.
    text(draw, (2120, 290), "c", 33, bold=True, color=SOURCE)
    text(draw, (2170, 294), "Per-element latent injection", 32, bold=True)
    text(draw, (2120, 342), "Every selected denoising step uses the same spatial policy", 22, color=MUTED)
    draw_latent_stack(draw, 2180, 430, TARGET, "current target trajectory", "L_target")
    draw_latent_stack(draw, 2690, 430, SOURCE, "cached source trajectory", "L_source")
    arrow(draw, (2445, 735), (2590, 835), TARGET)
    arrow(draw, (2945, 735), (2725, 835), SOURCE)
    text(draw, (2474, 798), "select by region", 21, color=MUTED)
    draw_region_table(draw, 2160, 900, 870)
    text(draw, (2160, 1232), "L_new = Mbase * L_target", 25, mono=True)
    text(draw, (2160, 1272), "      + Mband * ((1 - alpha) * L_target + alpha * L_source)", 25, mono=True)
    text(draw, (2160, 1312), "      + (1 - Medit) * L_source", 25, mono=True)
    draw.rounded_rectangle((2160, 1410, 3030, 1490), radius=12, fill=(240, 248, 249), outline=(188, 218, 221), width=2)
    text(draw, (2190, 1430), "Only Mband blends.  Mbase is fully target-generated; outside Medit is fully reused.", 20, color=INK)

    text(draw, (92, 1590), "Mask source: completed step 5 | alpha: smoothstep(Chebyshev distance to Mbase) | illustrated on the ceramic-mug run", 20, color=MUTED)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, quality=95)
    args.mask_overlap_output.parent.mkdir(parents=True, exist_ok=True)
    overlay_mask_overlap(frame, mbase, medit).save(args.mask_overlap_output, quality=95)
    args.transition_grid_output.parent.mkdir(parents=True, exist_ok=True)
    render_transition_pixels(mbase, medit, alpha, frame.size).save(args.transition_grid_output, quality=95)
    print(args.output)
    print(args.mask_overlap_output)
    print(args.transition_grid_output)


if __name__ == "__main__":
    render(parse_args())
