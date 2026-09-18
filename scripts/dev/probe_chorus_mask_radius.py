#!/usr/bin/env python3
"""Sweep Chorus's Medit halo radius and report the resulting mask geometry.

``methods/chorus/mask.py`` builds Mbase from Grounded-SAM-2, dilates it by
``mask_dilation_edit`` to get Medit and by ``mask_dilation_see`` to get Msee, and
projects both onto the Wan token grid.  This probe runs that same segmentation
and the same projection math once, then reports the masks for every radius in
``--radii`` so a radius can be chosen from measurements instead of guessed.

It is read-only with respect to the rest of the repository: it writes nothing
into an artifact root and never modifies the Chorus defaults in
``configs/methods/default.yaml``.  The defaults of this script describe the P1
counterexample run (Wan2.2-TI2V-5B, 1280x704, 121 frames, patch (1,2,2)).

Radii are counted in VAE-latent cells, not pixels: the dilation runs on the
latent grid after ``_resize_mask``, with a ``(1, 2r+1, 2r+1)`` max-pool kernel,
so a radius of r extends the mask r cells in H/W and does not extend it in time.
For Wan2.2's vae_stride (4, 16, 16) one latent cell is 16 pixels.

Example::

    python scripts/dev/probe_chorus_mask_radius.py \\
      --video artifacts/chorus_counterexample_p1_v2_halo2/pair_B/source_dense/videos/0000_p1B-source-barbell-gym-floor.mp4 \\
      --phrase barbell --radii 0,1,2,3,4,6,8 --out /tmp/probe_pairB.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from methods.chorus.grounded_sam import GroundedSam2Segmenter  # noqa: E402
from methods.chorus.mask import _resize_mask  # noqa: E402

# Wan2.2-TI2V-5B at 1280x704 / 121 frames: vae_stride (4, 16, 16) gives a
# 31x44x80 latent, which patch_size (1, 2, 2) turns into a 31x22x40 token grid.
DEFAULT_LATENT_SHAPE = (31, 44, 80)
DEFAULT_PATCH = (1, 2, 2)
DEFAULT_RADII = "0,1,2,3,4,6,8,10,14"


def _parse_shape(value: str, length: int, name: str) -> tuple[int, ...]:
    parts = tuple(int(item) for item in value.replace("x", ",").split(",") if item)
    if len(parts) != length:
        raise SystemExit(f"{name} expects {length} comma-separated integers, got {value!r}")
    return parts


def dilate(source: torch.Tensor, radius: int) -> torch.Tensor:
    """Same helper as ``methods/chorus/mask.py``: spatial-only max pooling."""
    if radius <= 0:
        return source.clamp(0, 1)
    kernel = 2 * radius + 1
    return torch.nn.functional.max_pool3d(
        source, kernel_size=(1, kernel, kernel), stride=1, padding=(0, radius, radius)
    ).clamp(0, 1)


def project(source: torch.Tensor, latent_shape: tuple[int, int, int],
            patch: tuple[int, int, int]) -> torch.Tensor:
    """Same token projection as ``methods/chorus/mask.py``, kept as a 3D grid."""
    frames, height, width = latent_shape
    patch_frames, patch_height, patch_width = patch
    token_height = max(1, height // patch_height)
    token_width = max(1, width // patch_width)
    pooled = torch.nn.functional.max_pool3d(
        source, kernel_size=patch, stride=patch
    )
    if pooled.shape[-2:] != (token_height, token_width):
        pooled = torch.nn.functional.interpolate(
            pooled, size=(frames, token_height, token_width), mode="nearest"
        )
    return pooled[0, 0].gt(0)


def frame_components(frame: np.ndarray) -> int:
    """4-connected component count of one token frame."""
    height, width = frame.shape
    seen = np.zeros_like(frame, dtype=bool)
    count = 0
    for row in range(height):
        for column in range(width):
            if not frame[row, column] or seen[row, column]:
                continue
            count += 1
            stack = [(row, column)]
            seen[row, column] = True
            while stack:
                y, x = stack.pop()
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width and frame[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
    return count


def measure(token_edit: np.ndarray) -> dict[str, object]:
    """Shape statistics of a token mask, per frame.

    ``bbox_*_median`` is the median per-frame extent of the region in token
    cells, so a full-width region reports the grid's width.
    """
    components = [frame_components(token_edit[index]) for index in range(token_edit.shape[0])]
    union = token_edit.any(axis=0)
    bbox_h, bbox_w = [], []
    for index in range(token_edit.shape[0]):
        rows, columns = np.nonzero(token_edit[index])
        if rows.size:
            bbox_h.append(int(rows.max() - rows.min() + 1))
            bbox_w.append(int(columns.max() - columns.min() + 1))
    return {
        "comp_median": int(np.median(components)),
        "comp_max": int(max(components)),
        "bbox_h_median": float(np.median(bbox_h)) if bbox_h else 0.0,
        "bbox_w_median": float(np.median(bbox_w)) if bbox_w else 0.0,
        "touches_left_and_right": bool(union[:, 0].any() and union[:, -1].any()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path, required=True, help="Source video to segment.")
    parser.add_argument("--phrase", required=True, help="Grounded-SAM-2 phrase, e.g. 'barbell'.")
    parser.add_argument("--radii", default=DEFAULT_RADII,
                        help=f"Medit halo radii to sweep (default {DEFAULT_RADII}).")
    parser.add_argument("--see-radius", type=int, default=14,
                        help="Msee radius, held fixed (default 14, Chorus's default).")
    parser.add_argument("--latent-shape", default=",".join(str(item) for item in DEFAULT_LATENT_SHAPE),
                        help="Latent (frames,height,width); default is Wan2.2-TI2V-5B at 1280x704/121f.")
    parser.add_argument("--patch", default=",".join(str(item) for item in DEFAULT_PATCH),
                        help="Wan patch_size (frames,height,width); default 1,2,2.")
    parser.add_argument("--device", default="cuda:0", help="Device for Grounded-SAM-2.")
    parser.add_argument("--mask-cache", type=Path,
                        help="Optional .npy for the pixel mask: reused if present, written otherwise.")
    parser.add_argument("--out", type=Path, help="Write the full result as JSON here.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    latent_shape = _parse_shape(args.latent_shape, 3, "--latent-shape")
    patch = _parse_shape(args.patch, 3, "--patch")
    radii = [int(item) for item in str(args.radii).split(",") if item.strip()]

    cache = args.mask_cache.expanduser().resolve() if args.mask_cache else None
    if cache is not None and cache.is_file():
        pixel_mask = np.load(cache)
        metadata: dict[str, object] = {"cached": True}
    else:
        chorus_cfg = yaml.safe_load(
            (ROOT / "configs/methods/default.yaml").read_text(encoding="utf-8")
        )["chorus"]
        segmenter = GroundedSam2Segmenter({**chorus_cfg["grounded_sam2"], "device": args.device})
        pixel_mask, metadata = segmenter.segment(args.video.expanduser().resolve(), args.phrase)
        pixel_mask = np.asarray(pixel_mask)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, pixel_mask)

    latent_np = _resize_mask(pixel_mask, latent_shape)
    base = torch.from_numpy(latent_np.astype(np.float32))[None, None]
    token_base = project(base, latent_shape, patch).numpy()
    pixels_per_cell = pixel_mask.shape[1] / latent_shape[1]

    rows: list[dict[str, object]] = []
    for radius in radii:
        edit = dilate(base, radius)
        see = torch.maximum(dilate(base, args.see_radius), edit)
        token_edit = project(edit, latent_shape, patch).numpy()
        token_see = project(see, latent_shape, patch).numpy()
        base_fraction = float(token_base.mean())
        edit_fraction = float(token_edit.mean())
        row: dict[str, object] = {
            "radius": radius,
            # One latent cell covers pixels_per_cell source pixels (16 for
            # Wan2.2-TI2V-5B at 704p, i.e. vae_stride 16).
            "pixel_radius": round(radius * pixels_per_cell, 1),
            "base_token_fraction": round(base_fraction, 4),
            "edit_token_fraction": round(edit_fraction, 4),
            "see_token_fraction": round(float(token_see.mean()), 4),
            "edit_pixel_fraction": round(float(edit.mean()), 4),
            "edit_over_base": round(edit_fraction / max(base_fraction, 1e-9), 2),
            "edit_extra_tokens": int(token_edit.sum() - token_base.sum()),
            **measure(token_edit),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    print(json.dumps({
        "video": str(args.video),
        "phrase": args.phrase,
        "latent_shape": list(latent_shape),
        "patch": list(patch),
        "grid": [latent_shape[0], max(1, latent_shape[1] // patch[1]),
                 max(1, latent_shape[2] // patch[2])],
        "see_radius": args.see_radius,
        "sam_metadata": metadata,
        "rows": rows,
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
