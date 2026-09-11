from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .grounded_sam import GroundedSam2Segmenter


def build_cogvideo_edit_mask(
    video_path: Path,
    phrase: str,
    *,
    latent_shape: tuple[int, int, int],
    patch_size: tuple[int, int, int],
    config: dict[str, Any],
    segmenter: GroundedSam2Segmenter | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch
    import torch.nn.functional as functional

    pixel_mask, metadata = (segmenter or GroundedSam2Segmenter(config)).segment(video_path, phrase)
    frames, height, width = latent_shape
    source = torch.from_numpy(np.asarray(pixel_mask, dtype=np.float32))[None, None]
    base = functional.interpolate(source, size=latent_shape, mode="nearest")

    def dilate(mask: Any, radius: int) -> Any:
        if radius <= 0:
            return mask.bool()
        kernel = 2 * radius + 1
        return functional.max_pool3d(
            mask, kernel_size=(1, kernel, kernel), stride=1, padding=(0, radius, radius)
        ).bool()

    edit = dilate(base, int(config.get("mask_dilation_edit", 8)))
    see = dilate(base, int(config.get("mask_dilation_see", 14))) | edit
    patch_t, patch_h, patch_w = patch_size
    padded_frames = ((frames + patch_t - 1) // patch_t) * patch_t

    def project(mask: Any) -> Any:
        if padded_frames != frames:
            mask = functional.pad(mask.float(), (0, 0, 0, 0, 0, padded_frames - frames), mode="replicate")
        pooled = functional.max_pool3d(
            mask.float(), kernel_size=patch_size, stride=patch_size
        )
        return pooled[0, 0].bool().contiguous()

    token_base, token_edit, token_see = project(base), project(edit), project(see)
    if not bool(token_base.any()) or not bool(token_edit.any()):
        raise RuntimeError("Grounded-SAM-2 produced an empty CogVideo token mask")

    def expand(mask: Any) -> Any:
        value = mask.repeat_interleave(patch_t, 0).repeat_interleave(patch_h, 1).repeat_interleave(patch_w, 2)
        return value[:frames, :height, :width].contiguous()

    metadata.update({
        "latent_grid": [frames, height, width],
        "token_grid": list(token_base.shape),
        "patch_size": list(patch_size),
        "base_fraction": float(token_base.float().mean().item()),
        "edit_fraction": float(token_edit.float().mean().item()),
        "see_fraction": float(token_see.float().mean().item()),
        "pooling": "max",
    })
    return {
        "token_base": token_base,
        "token_edit": token_edit,
        "token_see": token_see,
        "latent_base": expand(token_base),
        "latent_edit": expand(token_edit),
        "latent_see": expand(token_see),
    }, metadata
