from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SelfContrastMaskConfig:
    mask_steps: tuple[int, ...] = (5,)
    completed_step: int = 5
    noun: Optional[str] = None
    top_p: float = 0.02
    background_p: float = 0.02
    high_threshold: float = 0.45
    low_threshold: float = 0.35
    spatial_medit_radius: int = 2
    spatial_msee_radius: int = 3
    keep_isolated: bool = False
    layer: int = 15


@dataclass(frozen=True)
class FusionMaskConfig:
    mask_steps: tuple[int, ...] = (5,)
    completed_step: int = 5
    eps: float = 0.2
    gamma: float = 1.0
    high_threshold: float = 0.45
    low_threshold: float = 0.35
    spatial_medit_radius: int = 2
    spatial_msee_radius: int = 3
    keep_isolated: bool = False


def minmax_norm(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    lo = x.amin()
    hi = x.amax()
    return (x - lo) / (hi - lo + 1e-8)


def robust_norm(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    lo = torch.quantile(x.flatten(), 0.01)
    hi = torch.quantile(x.flatten(), 0.99)
    return ((x - lo) / (hi - lo).clamp_min(1e-6)).clamp(0.0, 1.0)


def top_fraction_mask(score: torch.Tensor, top_frac: float, largest: bool = True) -> torch.Tensor:
    """Exact top-k mask used by the self-contrast scripts."""

    flat = score.flatten()
    k = max(1, int(round(flat.numel() * float(top_frac))))
    indices = torch.topk(flat, k, largest=largest).indices
    out = torch.zeros_like(flat, dtype=torch.bool)
    out[indices] = True
    return out.reshape_as(score)


def build_cross_attention_seed_scores(
    records: list[dict[str, Any]],
    *,
    completed_step: int,
) -> dict[str, Any]:
    """Normalize completed-step cross-attention scores for self-attention seeds.

    This deliberately returns scores only. Cross attention is never a
    standalone Mbase source.
    """

    selected = [item for item in records if int(item["step_index"]) + 1 == int(completed_step)]
    if not selected:
        raise RuntimeError(f"missing probe records for completed_step={completed_step}")
    score = torch.stack([(item.get("noun_score", item.get("positive"))).float() for item in selected]).mean(dim=0)
    return {
        "seed_score_mode": "single-step",
        "seed_variant": "cross_attention_seed_scores",
        "source_completed_step": int(completed_step),
        "score_normalization": "minmax",
        "attention_layer_fusion_rule": "mean_across_probe_records",
        "score": minmax_norm(score),
        "records": [
            {
                "step_index": int(item["step_index"]),
                "completed_step": int(item["step_index"]) + 1,
                "layer_index": int(item["layer_index"]),
            }
            for item in selected
        ],
    }


def dilate_spatial(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask.bool()
    x = mask.bool().float()[:, None]
    return F.max_pool2d(x, kernel_size=2 * radius + 1, stride=1, padding=radius)[:, 0].bool()


def token_mask_to_latent_mask(mask: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
    """Expand a `[T,H,W]` token mask to a Wan latent tensor shape."""

    if mask.ndim != 3:
        raise ValueError(f"token mask must have shape [T,H,W], got {tuple(mask.shape)}")
    if latents.ndim != 5:
        raise ValueError(f"latents must have shape [B,C,T,H,W], got {tuple(latents.shape)}")

    mask = mask.bool().to(device=latents.device)
    _, _, latent_t, latent_h, latent_w = latents.shape
    scale_t = max(1, math.ceil(latent_t / mask.shape[0]))
    scale_h = max(1, math.ceil(latent_h / mask.shape[1]))
    scale_w = max(1, math.ceil(latent_w / mask.shape[2]))
    expanded = mask.repeat_interleave(scale_t, dim=0).repeat_interleave(scale_h, dim=1).repeat_interleave(scale_w, dim=2)
    expanded = expanded[:latent_t, :latent_h, :latent_w]
    return expanded[None, None].expand(latents.shape[0], latents.shape[1], -1, -1, -1)


def _mask_area(mbase: torch.Tensor, medit: torch.Tensor, msee: torch.Tensor) -> dict[str, float]:
    return {
        "mbase": float(mbase.float().mean().item()),
        "medit": float(medit.float().mean().item()),
        "msee": float(msee.float().mean().item()),
    }


def add_canonical_latent_masks(bundle: dict[str, Any], latents: torch.Tensor) -> dict[str, Any]:
    """Materialize canonical Wan region masks in latent space."""

    if "Medit" not in bundle:
        raise KeyError("missing canonical token mask Medit")
    mbase = bundle.get("Mbase", bundle["Medit"])
    medit = bundle["Medit"]
    msee = bundle.get("Msee", medit)
    bundle["latent_Mbase"] = token_mask_to_latent_mask(mbase, latents)
    bundle["latent_Medit"] = token_mask_to_latent_mask(medit, latents)
    bundle["latent_Msee"] = token_mask_to_latent_mask(msee, latents)
    bundle["latent_area"] = {
        "mbase": float(bundle["latent_Mbase"].float().mean().item()),
        "medit": float(bundle["latent_Medit"].float().mean().item()),
        "msee": float(bundle["latent_Msee"].float().mean().item()),
        "mband": float((bundle["latent_Medit"] & ~bundle["latent_Mbase"]).float().mean().item()),
    }
    return bundle


def get_canonical_latent_masks(bundle: dict[str, Any], latents: torch.Tensor) -> dict[str, Any]:
    """Return the fixed source-calibrated mask for any native Wan target step."""

    if "latent_Mbase" in bundle:
        latent_mbase = bundle["latent_Mbase"].to(device=latents.device, dtype=torch.bool)
    elif "Mbase" in bundle:
        latent_mbase = token_mask_to_latent_mask(bundle["Mbase"], latents)
    elif "Medit" in bundle:
        # Older bundles did not serialize Mbase.  Treat their edit region as
        # the core so the legacy hard-injection behavior remains unchanged.
        latent_mbase = token_mask_to_latent_mask(bundle["Medit"], latents)
    else:
        raise KeyError("missing canonical Mbase mask")
    if "latent_Medit" in bundle:
        latent_medit = bundle["latent_Medit"].to(device=latents.device, dtype=torch.bool)
    elif "Medit" in bundle:
        latent_medit = token_mask_to_latent_mask(bundle["Medit"], latents)
    else:
        raise KeyError("missing canonical Medit mask")
    if "latent_Msee" in bundle:
        latent_msee = bundle["latent_Msee"].to(device=latents.device, dtype=torch.bool)
    elif "Msee" in bundle:
        latent_msee = token_mask_to_latent_mask(bundle["Msee"], latents)
    else:
        raise KeyError("missing canonical Msee mask")
    return {
        "latent_mbase": latent_mbase,
        "latent_medit": latent_medit,
        "latent_msee": latent_msee,
        "source_completed_step": int(bundle["source_completed_step"]),
    }


def _distance_alpha(
    mbase: torch.Tensor,
    medit: torch.Tensor,
    *,
    function: str = "smoothstep",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a smooth source weight and the transition band.

    The distance is an exact Chebyshev distance on the latent grid, computed
    by iterative 3x3 dilation.  This matches the spatial max-pool dilation
    used to construct ``Medit`` while avoiding a scipy dependency.
    """

    if mbase.shape != medit.shape:
        raise ValueError(f"Mbase shape {tuple(mbase.shape)} != Medit shape {tuple(medit.shape)}")
    if function not in {"smoothstep", "cosine"}:
        raise ValueError("boundary alpha function must be 'smoothstep' or 'cosine'")
    if mbase.ndim != 5:
        raise ValueError(f"latent masks must have shape [B,C,T,H,W], got {tuple(mbase.shape)}")

    # The masks are channel-expanded copies.  Compute one spatial mask per
    # sample/frame and broadcast the resulting weights back to all channels.
    base = mbase[:, 0].bool()
    edit = (medit[:, 0].bool() | base)
    band = edit & ~base
    batch, frames, height, width = band.shape
    flat_base = base.reshape(batch * frames, 1, height, width)
    flat_band = band.reshape(batch * frames, 1, height, width)
    remaining = flat_band.clone()
    frontier = flat_base.clone()
    distance = torch.zeros_like(flat_band, dtype=torch.float32)

    # In normal Carnator masks this exits after only a few iterations because
    # Medit is a small dilation of Mbase.  The bound also handles malformed or
    # hand-authored bundles without an unbounded loop.
    for step in range(1, max(height, width) + 1):
        frontier = F.max_pool2d(frontier.float(), kernel_size=3, stride=1, padding=1).gt(0)
        newly = frontier & remaining
        distance = torch.where(newly, torch.as_tensor(float(step), device=distance.device), distance)
        remaining &= ~newly
        if not bool(remaining.any()):
            break
    if bool(remaining.any()):
        fallback = distance.amax(dim=(-2, -1), keepdim=True).clamp_min(1.0) + 1.0
        distance = torch.where(remaining, fallback, distance)

    band_flat = flat_band.flatten(1)
    max_distance = distance.flatten(1).masked_fill(~band_flat, 0.0).amax(dim=1).view(batch * frames, 1, 1, 1)
    denominator = (max_distance - 1.0).clamp_min(1.0)
    normalized = ((distance - 1.0) / denominator).clamp(0.0, 1.0)
    # A one-cell transition cannot expose both endpoints.  Keep it genuinely
    # blended instead of collapsing it to the target-side endpoint.
    normalized = torch.where(max_distance == 1.0, torch.full_like(normalized, 0.5), normalized)
    if function == "smoothstep":
        alpha = normalized.square() * (3.0 - 2.0 * normalized)
    else:
        alpha = 0.5 - 0.5 * torch.cos(torch.pi * normalized)
    alpha = torch.where(flat_band, alpha, torch.zeros_like(alpha))
    alpha = alpha.reshape(batch, frames, height, width)[:, None]
    alpha = alpha.expand_as(mbase)
    return alpha, band[:, None].expand_as(mbase)


def boundary_blend_weights(
    masks: dict[str, torch.Tensor],
    *,
    function: str = "smoothstep",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cached-friendly ``(alpha, M_band)`` tensors for latent blending."""

    return _distance_alpha(masks["latent_mbase"], masks["latent_medit"], function=function)


def boundary_blend_latent(
    latent: torch.Tensor,
    source_latent: torch.Tensor,
    masks: dict[str, torch.Tensor],
    *,
    function: str = "smoothstep",
    weights: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Blend source latent through the Mbase--Medit transition band.

    ``Mbase`` stays entirely target-generated, ``Medit minus Mbase`` receives a
    distance-weighted blend, and the stable region outside ``Medit`` keeps the
    source latent used by the original Carnator baseline.
    """

    if latent.shape != source_latent.shape:
        raise ValueError(f"target latent {tuple(latent.shape)} != source latent {tuple(source_latent.shape)}")
    alpha, band = weights if weights is not None else boundary_blend_weights(masks, function=function)
    target = latent.unsqueeze(0)
    source = source_latent.to(device=latent.device, dtype=latent.dtype).unsqueeze(0)
    mbase = masks["latent_mbase"].to(device=latent.device, dtype=torch.bool)
    band = band.to(device=latent.device, dtype=torch.bool)
    alpha = alpha.to(device=latent.device, dtype=latent.dtype)
    blended = target * (1.0 - alpha) + source * alpha
    output = torch.where(mbase, target, torch.where(band, blended, source))[0]
    band_alpha = alpha[band].float()
    stats = {
        "base_fraction": float(mbase.float().mean().item()),
        "band_fraction": float(band.float().mean().item()),
        "edit_fraction": float((mbase | band).float().mean().item()),
        "alpha_mean": float(band_alpha.mean().item()) if band_alpha.numel() else 0.0,
        "alpha_min": float(band_alpha.min().item()) if band_alpha.numel() else 0.0,
        "alpha_max": float(band_alpha.max().item()) if band_alpha.numel() else 0.0,
    }
    return output, stats


def remove_isolated_pixels(mask: torch.Tensor) -> torch.Tensor:
    mask = mask.bool()
    height, width = mask.shape
    out = mask.clone()
    for y in range(height):
        for x in range(width):
            if not bool(mask[y, x]):
                continue
            neighbor_count = 0
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= yy < height and 0 <= xx < width and bool(mask[yy, xx]):
                    neighbor_count += 1
            if neighbor_count == 0:
                out[y, x] = False
    return out


def clean_per_slice(mask: torch.Tensor) -> torch.Tensor:
    return torch.stack([remove_isolated_pixels(frame) for frame in mask.bool()], dim=0).bool()


def temporal_smooth_score(score: torch.Tensor) -> torch.Tensor:
    prev_score = torch.cat([score[:1], score[:-1]], dim=0)
    next_score = torch.cat([score[1:], score[-1:]], dim=0)
    return 0.25 * prev_score + 0.5 * score + 0.25 * next_score


def temporal_dilate(mask: torch.Tensor) -> torch.Tensor:
    prev_mask = torch.cat([mask[:1], mask[:-1]], dim=0)
    next_mask = torch.cat([mask[1:], mask[-1:]], dim=0)
    return mask.bool() | prev_mask.bool() | next_mask.bool()


def temporal_close(mask: torch.Tensor) -> torch.Tensor:
    prev_mask = torch.cat([mask[:1], mask[:-1]], dim=0)
    next_mask = torch.cat([mask[1:], mask[-1:]], dim=0)
    return mask.bool() | (prev_mask.bool() & next_mask.bool())


def build_mbase_from_hysteresis_score(
    score: torch.Tensor,
    *,
    high_threshold: float,
    low_threshold: float,
    keep_isolated: bool,
) -> tuple[torch.Tensor, dict[str, Any]]:
    smoothed = temporal_smooth_score(score)
    core = smoothed >= high_threshold
    support = smoothed >= low_threshold
    hysteresis = support & temporal_dilate(core)
    closed = temporal_close(hysteresis)
    mbase = closed if keep_isolated else clean_per_slice(closed)
    return mbase, {
        "mode": "smooth-hysteresis",
        "high_threshold": high_threshold,
        "low_threshold": low_threshold,
        "core_area": float(core.float().mean().item()),
        "support_area": float(support.float().mean().item()),
        "hysteresis_area": float(hysteresis.float().mean().item()),
        "closed_area": float(closed.float().mean().item()),
    }


def ensure_tail_mbase_support(mask: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    """Keep the signature token covered without changing fusion thresholds."""

    if mask.ndim != 3:
        raise ValueError(f"Mbase must have shape [T,H,W], got {tuple(mask.shape)}")
    nonempty = mask.bool().flatten(1).any(dim=1)
    if not bool(nonempty.any()):
        raise ValueError("fused Mbase is empty at every time token")
    trailing_empty = 0
    for present in reversed(nonempty.tolist()):
        if present:
            break
        trailing_empty += 1
    if trailing_empty == 0:
        return mask.bool(), {"applied": False, "mode": "none", "filled_slices": 0}
    source_index = int(torch.where(nonempty)[0][-1].item())
    repaired = mask.bool().clone()
    repaired[-trailing_empty:] = repaired[source_index]
    return repaired, {
        "applied": True,
        "mode": "nearest_nonempty_tail_slice",
        "filled_slices": trailing_empty,
        "source_slice": source_index,
    }


def variation_metrics(mask: torch.Tensor) -> dict[str, Any]:
    flat = mask.bool().flatten(1)
    areas = flat.float().mean(dim=1)
    diffs = (flat[1:] != flat[:-1]).float().mean(dim=1) if flat.shape[0] > 1 else torch.empty(0)
    unique_masks = torch.unique(flat.to(torch.uint8), dim=0).shape[0]
    return {
        "area_min": float(areas.min().item()),
        "area_max": float(areas.max().item()),
        "area_mean": float(areas.mean().item()),
        "area_per_token_t": [float(v) for v in areas.tolist()],
        "adjacent_hamming_min": float(diffs.min().item()) if diffs.numel() else 0.0,
        "adjacent_hamming_max": float(diffs.max().item()) if diffs.numel() else 0.0,
        "adjacent_hamming_mean": float(diffs.mean().item()) if diffs.numel() else 0.0,
        "unique_token_time_masks": int(unique_masks),
    }


def score_metrics(score: torch.Tensor) -> dict[str, Any]:
    quantiles = [0.0, 0.01, 0.05, 0.1, 0.2, 0.35, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
    q_tensor = torch.quantile(score.flatten(), torch.tensor(quantiles, dtype=torch.float32))
    return {
        "min": float(score.min().item()),
        "max": float(score.max().item()),
        "mean": float(score.mean().item()),
        "quantiles": {str(q): float(v) for q, v in zip(quantiles, q_tensor.tolist())},
    }


def seed_indices_from_cross_score(
    cross_score: torch.Tensor,
    *,
    top_p: float = 0.02,
    background_p: float = 0.02,
) -> tuple[torch.Tensor, torch.Tensor]:
    noun_seed = top_fraction_mask(cross_score, top_p, largest=True).flatten()
    background_seed = top_fraction_mask(cross_score, background_p, largest=False).flatten()
    noun_idx = torch.nonzero(noun_seed, as_tuple=False).flatten()
    background_idx = torch.nonzero(background_seed, as_tuple=False).flatten()
    return noun_idx, background_idx


def _fill_repeated_steps(
    bundle: dict[str, Any],
    *,
    mask_steps: tuple[int, ...],
    score: torch.Tensor,
    mbase: torch.Tensor,
    medit: torch.Tensor,
    msee: torch.Tensor,
    source_completed_step: int,
) -> None:
    for completed_step in mask_steps:
        step_key = f"step{completed_step:03d}"
        bundle[f"{step_key}_score"] = score
        bundle[f"{step_key}_raw_coverage"] = mbase
        bundle[f"{step_key}_Mbase"] = mbase
        bundle[f"{step_key}_Medit"] = medit
        bundle[f"{step_key}_Msee"] = msee
        bundle["step_metadata"][str(completed_step)] = {
            "source_completed_step": int(source_completed_step),
            "token_area": {
                "mbase": float(mbase.float().mean().item()),
                "medit": float(medit.float().mean().item()),
                "msee": float(msee.float().mean().item()),
            },
        }


def build_self_contrast_masks_from_score(
    self_contrast_score: torch.Tensor,
    config: SelfContrastMaskConfig,
    *,
    noun_seed_count: Optional[int] = None,
    background_seed_count: Optional[int] = None,
) -> dict[str, Any]:
    """Replicate `self_contrast_3d_per_token_time` mask post-processing."""

    if self_contrast_score.ndim != 3:
        raise ValueError(f"self_contrast_score must have shape [T,H,W], got {tuple(self_contrast_score.shape)}")

    self_contrast = robust_norm(self_contrast_score).cpu()
    mbase, postprocess_meta = build_mbase_from_hysteresis_score(
        self_contrast,
        high_threshold=config.high_threshold,
        low_threshold=config.low_threshold,
        keep_isolated=config.keep_isolated,
    )
    medit = dilate_spatial(mbase, config.spatial_medit_radius)
    msee = dilate_spatial(medit, config.spatial_msee_radius)

    bundle: dict[str, Any] = {
        "attention_mask_mode": "attention-stepwise-coarse",
        "mask_variant": "self_contrast_3d_per_token_time",
        "attention_noun": config.noun,
        "mask_top_p": config.top_p,
        "background_p": config.background_p,
        "attention_probe_layers": [config.layer],
        "attention_layer_fusion_rule": "single_layer_self_attention_qk_probe",
        "mask_steps": list(config.mask_steps),
        "mask_spatial_medit_radius": config.spatial_medit_radius,
        "mask_spatial_msee_radius": config.spatial_msee_radius,
        "source_completed_step": config.completed_step,
        "temporal_postprocess": postprocess_meta,
        "removed_isolated_pixels_per_slice": not config.keep_isolated,
        "score_normalization": "self_contrast_robust_quantile_0.01_0.99",
        "temporal_shape_rule": "3d_per_token_time_smooth_hysteresis",
        "token_shape": list(mbase.shape),
        "noun_seed_count": noun_seed_count,
        "background_seed_count": background_seed_count,
        "step_metadata": {},
    }
    _fill_repeated_steps(
        bundle,
        mask_steps=config.mask_steps,
        score=self_contrast,
        mbase=mbase,
        medit=medit,
        msee=msee,
        source_completed_step=config.completed_step,
    )
    bundle["variation"] = {
        "mbase": variation_metrics(mbase),
        "medit": variation_metrics(medit),
        "msee": variation_metrics(msee),
    }
    return bundle


def build_fusion_score(self_score: torch.Tensor, cross_score: torch.Tensor, eps: float, gamma: float) -> torch.Tensor:
    if not 0.0 <= eps <= 1.0:
        raise ValueError(f"eps must be in [0, 1], got {eps}")
    if gamma <= 0.0 or not math.isfinite(gamma):
        raise ValueError(f"gamma must be positive and finite, got {gamma}")
    self_score = self_score.float().clamp(0.0, 1.0)
    cross_score = cross_score.float().clamp(0.0, 1.0)
    gate = eps + (1.0 - eps) * cross_score.pow(gamma)
    return robust_norm(self_score * gate)


def build_self_contrast_fusion_masks(
    self_score: torch.Tensor,
    cross_score: torch.Tensor,
    config: FusionMaskConfig,
    *,
    original_mbase: Optional[torch.Tensor] = None,
) -> dict[str, Any]:
    """Replicate `self_contrast_cross_attention_fusion` mask post-processing."""

    if self_score.shape != cross_score.shape:
        raise ValueError(f"self score shape {tuple(self_score.shape)} != cross score shape {tuple(cross_score.shape)}")

    fusion_score = build_fusion_score(self_score, cross_score, config.eps, config.gamma)
    fused_mbase, postprocess_meta = build_mbase_from_hysteresis_score(
        fusion_score,
        high_threshold=config.high_threshold,
        low_threshold=config.low_threshold,
        keep_isolated=config.keep_isolated,
    )
    fused_medit = dilate_spatial(fused_mbase, config.spatial_medit_radius)
    fused_msee = dilate_spatial(fused_medit, config.spatial_msee_radius)

    bundle: dict[str, Any] = {
        "attention_mask_mode": "attention-stepwise-coarse",
        "mask_variant": "self_contrast_cross_attention_fusion",
        "formula": "F = robust_norm(S * (eps + (1 - eps) * C ** gamma))",
        "eps": config.eps,
        "gamma": config.gamma,
        "source_completed_step": config.completed_step,
        "score": {
            "self_contrast": score_metrics(self_score),
            "cross_attention": score_metrics(cross_score),
            "fusion": score_metrics(fusion_score),
        },
        "temporal_postprocess": postprocess_meta,
        "removed_isolated_pixels_per_slice": not config.keep_isolated,
        "mask_spatial_medit_radius": config.spatial_medit_radius,
        "mask_spatial_msee_radius": config.spatial_msee_radius,
        "mask_steps": list(config.mask_steps),
        "token_shape": list(fusion_score.shape),
        "step_metadata": {},
    }
    _fill_repeated_steps(
        bundle,
        mask_steps=config.mask_steps,
        score=fusion_score,
        mbase=fused_mbase,
        medit=fused_medit,
        msee=fused_msee,
        source_completed_step=config.completed_step,
    )
    token_area = {
        "fusion_mbase": float(fused_mbase.float().mean().item()),
        "fusion_medit": float(fused_medit.float().mean().item()),
        "fusion_msee": float(fused_msee.float().mean().item()),
    }
    variation = {
        "fusion_mbase": variation_metrics(fused_mbase),
        "fusion_medit": variation_metrics(fused_medit),
        "fusion_msee": variation_metrics(fused_msee),
    }
    if original_mbase is not None:
        token_area["original_mbase"] = float(original_mbase.float().mean().item())
        variation["original_mbase"] = variation_metrics(original_mbase)
    bundle["token_area"] = token_area
    bundle["variation"] = variation
    return bundle


def build_canonical_self_contrast_mask(
    self_contrast_score: torch.Tensor,
    config: SelfContrastMaskConfig,
    *,
    noun_seed_count: Optional[int] = None,
    background_seed_count: Optional[int] = None,
) -> dict[str, Any]:
    """Build the fixed native-Wan mask from one calibrated self-attention probe."""

    if self_contrast_score.ndim != 3:
        raise ValueError(f"self_contrast_score must have shape [T,H,W], got {tuple(self_contrast_score.shape)}")
    self_contrast = robust_norm(self_contrast_score).cpu()
    mbase, postprocess_meta = build_mbase_from_hysteresis_score(
        self_contrast,
        high_threshold=config.high_threshold,
        low_threshold=config.low_threshold,
        keep_isolated=config.keep_isolated,
    )
    medit = dilate_spatial(mbase, config.spatial_medit_radius)
    msee = dilate_spatial(medit, config.spatial_msee_radius)
    return {
        "attention_mask_mode": "single-step",
        "mask_variant": "self_contrast_3d_per_token_time",
        "source_completed_step": int(config.completed_step),
        "attention_noun": config.noun,
        "mask_top_p": config.top_p,
        "background_p": config.background_p,
        "attention_probe_layers": [config.layer],
        "attention_layer_fusion_rule": "single_layer_self_attention_qk_probe",
        "mask_spatial_medit_radius": config.spatial_medit_radius,
        "mask_spatial_msee_radius": config.spatial_msee_radius,
        "temporal_postprocess": postprocess_meta,
        "removed_isolated_pixels_per_slice": not config.keep_isolated,
        "score_normalization": "self_contrast_robust_quantile_0.01_0.99",
        "temporal_shape_rule": "3d_per_token_time_smooth_hysteresis",
        "token_shape": list(mbase.shape),
        "noun_seed_count": noun_seed_count,
        "background_seed_count": background_seed_count,
        "score": self_contrast,
        "raw_coverage": mbase,
        "Mbase": mbase,
        "Medit": medit,
        "Msee": msee,
        "token_area": _mask_area(mbase, medit, msee),
        "variation": {
            "mbase": variation_metrics(mbase),
            "medit": variation_metrics(medit),
            "msee": variation_metrics(msee),
        },
    }


def build_canonical_fusion_mask(
    self_score: torch.Tensor,
    cross_score: torch.Tensor,
    config: FusionMaskConfig,
    *,
    original_mbase: Optional[torch.Tensor] = None,
) -> dict[str, Any]:
    """Fuse one calibrated self/cross score pair into the native-Wan mask."""

    if self_score.shape != cross_score.shape:
        raise ValueError(f"self score shape {tuple(self_score.shape)} != cross score shape {tuple(cross_score.shape)}")
    fusion_score = build_fusion_score(self_score, cross_score, config.eps, config.gamma)
    mbase, postprocess_meta = build_mbase_from_hysteresis_score(
        fusion_score,
        high_threshold=config.high_threshold,
        low_threshold=config.low_threshold,
        keep_isolated=config.keep_isolated,
    )
    mbase, tail_support_meta = ensure_tail_mbase_support(mbase)
    postprocess_meta = dict(postprocess_meta)
    postprocess_meta["tail_support_fallback"] = tail_support_meta
    medit = dilate_spatial(mbase, config.spatial_medit_radius)
    msee = dilate_spatial(medit, config.spatial_msee_radius)
    token_area = _mask_area(mbase, medit, msee)
    variation = {
        "mbase": variation_metrics(mbase),
        "medit": variation_metrics(medit),
        "msee": variation_metrics(msee),
    }
    if original_mbase is not None:
        token_area["original_mbase"] = float(original_mbase.float().mean().item())
        variation["original_mbase"] = variation_metrics(original_mbase)
    return {
        "attention_mask_mode": "single-step",
        "mask_variant": "self_contrast_cross_attention_fusion",
        "source_completed_step": int(config.completed_step),
        "formula": "F = robust_norm(S * (eps + (1 - eps) * C ** gamma))",
        "eps": config.eps,
        "gamma": config.gamma,
        "score": fusion_score,
        "score_metrics": {
            "self_contrast": score_metrics(self_score),
            "cross_attention": score_metrics(cross_score),
            "fusion": score_metrics(fusion_score),
        },
        "raw_coverage": mbase,
        "Mbase": mbase,
        "Medit": medit,
        "Msee": msee,
        "token_area": token_area,
        "variation": variation,
        "temporal_postprocess": postprocess_meta,
        "removed_isolated_pixels_per_slice": not config.keep_isolated,
        "mask_spatial_medit_radius": config.spatial_medit_radius,
        "mask_spatial_msee_radius": config.spatial_msee_radius,
        "token_shape": list(fusion_score.shape),
    }


def build_per_noun_mbase_bundle(
    mbase: torch.Tensor,
    *,
    source_completed_step: int,
    spatial_medit_radius: int,
    spatial_msee_radius: int,
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a request-private canonical bundle from one source noun Mbase."""

    if not isinstance(mbase, torch.Tensor) or mbase.dtype != torch.bool or mbase.ndim != 3:
        raise ValueError("per-noun Mbase must be a bool [T,H,W] tensor")
    mbase, tail_support = ensure_tail_mbase_support(mbase.detach().cpu().contiguous())
    medit = dilate_spatial(mbase, spatial_medit_radius)
    msee = dilate_spatial(medit, spatial_msee_radius)
    return {
        "attention_mask_mode": "per-noun-early-signature",
        "mask_variant": "source_per_noun_mbase",
        "source_completed_step": int(source_completed_step),
        "Mbase": mbase,
        "Medit": medit,
        "Msee": msee,
        "raw_coverage": mbase,
        "token_shape": list(mbase.shape),
        "mask_spatial_medit_radius": int(spatial_medit_radius),
        "mask_spatial_msee_radius": int(spatial_msee_radius),
        "temporal_postprocess": {"tail_support_fallback": tail_support},
        "selection": dict(selection),
        "token_area": _mask_area(mbase, medit, msee),
        "variation": {
            "mbase": variation_metrics(mbase),
            "medit": variation_metrics(medit),
            "msee": variation_metrics(msee),
        },
    }


def save_mask_bundle(mask_dir: Path, bundle: dict[str, Any], save_cpu: bool = True) -> str:
    mask_dir.mkdir(parents=True, exist_ok=True)
    tensor_payload = {}
    metadata = {}
    for key, value in bundle.items():
        if isinstance(value, torch.Tensor):
            tensor_payload[key] = value.detach().cpu() if save_cpu else value.detach()
        else:
            metadata[key] = value

    payload_path = mask_dir / "masks.pt"
    meta_path = mask_dir / "masks.json"
    torch.save(tensor_payload, payload_path)
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return str(payload_path)


def load_mask_bundle(mask_dir: Path, device: Optional[str] = None) -> dict[str, Any]:
    payload_path = mask_dir / "masks.pt"
    meta_path = mask_dir / "masks.json"
    tensors = torch.load(payload_path, map_location="cpu")
    metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    out = dict(metadata)
    for key, value in tensors.items():
        out[key] = value.to(device=device) if device is not None and isinstance(value, torch.Tensor) else value
    return out
