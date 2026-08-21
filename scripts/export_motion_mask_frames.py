#!/usr/bin/env python3
"""Export text-free beginning/middle/end frames with Carnator mask overlays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image


COLORS = {
    "original": None,
    "mbase": ((239, 68, 68), 152),
    "medit": ((245, 158, 11), 120),
    "msee": ((14, 165, 166), 92),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Artifact run directory containing videos and cache.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--video", default="videos/source_dense.mp4")
    parser.add_argument("--mask", default="cache/source_dense/fusion_masks/masks.pt")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--positive", required=True)
    parser.add_argument(
        "--positions",
        nargs="+",
        choices=("start", "middle", "end"),
        default=("start", "middle", "end"),
        help="Temporal positions to export; defaults to all three.",
    )
    return parser.parse_args()


def load_masks(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    masks = {}
    for name in ("Mbase", "Medit", "Msee"):
        value = payload[name]
        if value.ndim != 3:
            raise ValueError(f"{name} must be [token_time, token_height, token_width], got {tuple(value.shape)}")
        masks[name.lower()] = value.bool()
    return masks


def expand_token_mask(mask: torch.Tensor, *, frames: int, height: int, width: int) -> np.ndarray:
    """Follow Wan's token-to-latent convention, then expand to RGB frames."""
    token_t, token_h, token_w = mask.shape
    latent_h = height // 16
    latent_w = width // 16
    scale_h = max(1, (latent_h + token_h - 1) // token_h)
    scale_w = max(1, (latent_w + token_w - 1) // token_w)
    latent = mask.repeat_interleave(scale_h, dim=1).repeat_interleave(scale_w, dim=2)
    latent = latent[:, :latent_h, :latent_w]
    latent = torch.nn.functional.interpolate(
        latent[None, None].float(), size=(frames, latent_h, latent_w), mode="nearest"
    )[0, 0, :, :, :]
    pixel = torch.nn.functional.interpolate(
        latent[None, None], size=(frames, height, width), mode="nearest"
    )[0, 0]
    return pixel.bool().numpy()


def overlay(frame: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: int) -> Image.Image:
    base = Image.fromarray(frame[..., :3].astype(np.uint8), mode="RGB").convert("RGBA")
    layer = Image.new("RGBA", base.size, color + (0,))
    mask_image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    layer.putalpha(mask_image.point(lambda value: value * alpha // 255))
    return Image.alpha_composite(base, layer).convert("RGB")


def main() -> None:
    args = parse_args()
    video_path = args.run_dir / args.video
    mask_path = args.run_dir / args.mask
    frames = iio.imread(video_path)
    if frames.ndim != 4:
        raise ValueError(f"expected a video tensor [T,H,W,C], got {frames.shape}")
    frame_count, height, width = frames.shape[:3]
    indices = {"start": 0, "middle": frame_count // 2, "end": frame_count - 1}
    masks = load_masks(mask_path)
    expanded = {name: expand_token_mask(mask, frames=frame_count, height=height, width=width) for name, mask in masks.items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected_indices = {position: indices[position] for position in args.positions}
    for position, index in selected_indices.items():
        frame = frames[index]
        Image.fromarray(frame[..., :3].astype(np.uint8), mode="RGB").save(args.output_dir / f"{position}_original.png")
        for name, (color, alpha) in ((key, value) for key, value in COLORS.items() if value is not None):
            overlay(frame, expanded[name][index], color, alpha).save(args.output_dir / f"{position}_{name}.png")

    metadata = {
        "prompt": args.prompt,
        "positive_phrase": args.positive,
        "video": str(video_path),
        "mask_bundle": str(mask_path),
        "frame_count": frame_count,
        "resolution": [width, height],
        "selected_frames": selected_indices,
        "mask_areas": {name: float(mask.float().mean()) for name, mask in masks.items()},
        "files_per_position": ["original", "mbase", "medit", "msee"],
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
