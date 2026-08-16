from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .grounded_sam import GroundedSam2Segmenter


def _resize_mask(mask: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    frames, height, width = shape
    tensor = torch.from_numpy(mask.astype(np.float32))[None, None]
    resized = F.interpolate(tensor, size=(frames, height, width), mode="nearest")
    return resized[0, 0].numpy()


def _token_mask_to_latent_mask(
    token_mask: Any,
    latent_shape: tuple[int, int, int],
    token_shape: tuple[int, int, int],
) -> Any:
    """Expand a token-grid mask using the same convention as Wan unpatchify."""
    latent_frames, latent_height, latent_width = latent_shape
    token_frames, token_height, token_width = token_shape
    if tuple(token_mask.shape) != (token_frames * token_height * token_width,):
        raise ValueError(
            "Chorus token mask does not match the Wan token grid: "
            f"mask={tuple(token_mask.shape)}, grid={token_shape}"
        )
    mask = token_mask.reshape(token_frames, token_height, token_width).bool()
    scale_frames = max(1, (latent_frames + token_frames - 1) // token_frames)
    scale_height = max(1, (latent_height + token_height - 1) // token_height)
    scale_width = max(1, (latent_width + token_width - 1) // token_width)
    expanded = mask.repeat_interleave(scale_frames, dim=0)
    expanded = expanded.repeat_interleave(scale_height, dim=1)
    expanded = expanded.repeat_interleave(scale_width, dim=2)
    return expanded[:latent_frames, :latent_height, :latent_width]


def build_edit_mask(
    video_path: Path,
    phrase: str,
    *,
    latent_shape: tuple[int, int, int],
    patch_size: tuple[int, int],
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the SRD base/edit/visible masks from Grounded-SAM-2 output.

    ``Medit`` selects tokens whose features are recomputed. ``Msee`` is the
    larger context window used as the self-attention K/V set. Both masks are
    projected with max pooling so a token is retained when any covered latent
    cell belongs to the corresponding pixel-space region. The returned latent
    edit mask is then expanded back through the same patch grid for injection.
    """
    import torch
    import torch.nn.functional as F

    pixel_mask, metadata = GroundedSam2Segmenter(config).segment(video_path, phrase)
    latent_frames, latent_height, latent_width = latent_shape
    latent_mask_np = _resize_mask(pixel_mask, latent_shape)
    radius_edit = int(config.get("mask_dilation_edit", 8))
    radius_see = int(config.get("mask_dilation_see", 14))
    base = torch.from_numpy(latent_mask_np.astype(np.float32))[None, None]

    def dilate(source: Any, radius: int) -> Any:
        if radius <= 0:
            return source.clamp(0, 1)
        kernel = 2 * radius + 1
        return F.max_pool3d(source, kernel_size=(1, kernel, kernel), stride=1,
                            padding=(0, radius, radius)).clamp(0, 1)

    edit = dilate(base, radius_edit)
    see = dilate(base, radius_see)
    see = torch.maximum(see, edit)
    token_height = max(1, latent_height // int(patch_size[0]))
    token_width = max(1, latent_width // int(patch_size[1]))

    def project(source: Any) -> Any:
        pooled = F.max_pool3d(
            source,
            kernel_size=(1, int(patch_size[0]), int(patch_size[1])),
            stride=(1, int(patch_size[0]), int(patch_size[1])),
        )
        if pooled.shape[-2:] != (token_height, token_width):
            pooled = F.interpolate(pooled, size=(latent_frames, token_height, token_width), mode="nearest")
        return pooled[0, 0].reshape(-1).gt(0)

    token_edit = project(edit)
    token_see = project(see)
    token_see |= token_edit
    if not bool(token_edit.any()):
        raise RuntimeError("Grounded-SAM-2 produced an empty SRD edit mask")
    metadata.update({
        "grid": [latent_frames, token_height, token_width],
        "base_fraction": round(float(base.mean().item()), 4),
        "edit_fraction": round(float(token_edit.float().mean().item()), 4),
        "see_fraction": round(float(token_see.float().mean().item()), 4),
        "dilation": {"edit": radius_edit, "see": radius_see},
        "pooling": "max",
    })
    # Latent injection must use the same patch-aligned mask as SRD token
    # selection. A pixel mask would otherwise split a Wan (1, 2, 2) patch and
    # leave the scheduler/model operating on different regions.
    latent_edit = _token_mask_to_latent_mask(
        token_edit,
        (latent_frames, latent_height, latent_width),
        (latent_frames, token_height, token_width),
    )
    return {
        "latent_edit": latent_edit,
        "pixel_edit": edit.to(dtype=torch.float32)[0, 0],
        "token_edit": token_edit,
        "token_see": token_see,
    }, metadata
