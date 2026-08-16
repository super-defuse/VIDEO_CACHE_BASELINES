from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class CoarseMaskConfig:
    mask_steps: tuple[int, ...] = (5,)
    top_p: float = 0.02
    coarse_shape: str = "capsule"
    coarse_frame_top_p: float = 0.05
    coarse_min_seed_tokens: int = 4
    coarse_expand_w: float = 2.1
    coarse_expand_h: float = 1.6
    coarse_min_width_ratio: float = 0.16
    coarse_min_height_ratio: float = 0.16
    coarse_min_area_ratio: float = 0.10
    coarse_max_area_ratio: float = 0.55
    coarse_size_rule: str = "max"
    coarse_smooth_radius: int = 1
    coarse_temporal_union_radius: int = 1
    spatial_medit_radius: int = 2
    spatial_msee_radius: int = 3
    score_normalization: str = "minmax"


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


def coarse_top_fraction_mask(score: torch.Tensor, top_frac: float) -> torch.Tensor:
    """Threshold-style top fraction used by the coarse cross-attention mask."""

    flat = score.flatten()
    k = max(1, int(round(flat.numel() * float(top_frac))))
    threshold = torch.topk(flat, k).values.min()
    mask = score >= threshold
    if int(mask.sum().item()) <= k:
        return mask.bool()

    top_idx = torch.topk(flat, k).indices
    out = torch.zeros_like(flat, dtype=torch.bool)
    out[top_idx] = True
    return out.reshape_as(score)


def _fill_missing_centers(
    centers: list[Optional[tuple[float, float]]],
    fallback: tuple[float, float],
) -> list[tuple[float, float]]:
    filled = list(centers)
    prev = None
    for idx, center in enumerate(filled):
        if center is None and prev is not None:
            filled[idx] = prev
        elif center is not None:
            prev = center

    nxt = None
    for idx in range(len(filled) - 1, -1, -1):
        center = filled[idx]
        if center is None and nxt is not None:
            filled[idx] = nxt
        elif center is not None:
            nxt = center

    return [center if center is not None else fallback for center in filled]


def _filled_centers(centers: list[Optional[tuple[float, float]]]) -> list[tuple[float, float]]:
    valid = [center for center in centers if center is not None]
    if not valid:
        raise ValueError("cannot build attention mask: no valid attention center")
    return _fill_missing_centers(centers, valid[len(valid) // 2])


def _smooth(values: list[float], radius: int) -> list[float]:
    if radius <= 0:
        return values
    out = []
    for idx in range(len(values)):
        lo = max(0, idx - radius)
        hi = min(len(values), idx + radius + 1)
        out.append(sum(values[lo:hi]) / max(1, hi - lo))
    return out


def _smooth_boxes(
    boxes: list[tuple[float, float, float, float]],
    radius: int,
) -> list[tuple[float, float, float, float]]:
    if radius <= 0:
        return boxes
    cy = _smooth([(box[0] + box[2]) * 0.5 for box in boxes], radius)
    cx = _smooth([(box[1] + box[3]) * 0.5 for box in boxes], radius)
    hh = _smooth([(box[2] - box[0]) * 0.5 for box in boxes], radius)
    hw = _smooth([(box[3] - box[1]) * 0.5 for box in boxes], radius)
    return [(y - h, x - w, y + h, x + w) for y, x, h, w in zip(cy, cx, hh, hw)]


def _union_boxes(
    boxes: list[tuple[float, float, float, float]],
    radius: int,
) -> list[tuple[float, float, float, float]]:
    if radius <= 0:
        return boxes
    out = []
    for idx in range(len(boxes)):
        lo = max(0, idx - radius)
        hi = min(len(boxes), idx + radius + 1)
        out.append(
            (
                min(box[0] for box in boxes[lo:hi]),
                min(box[1] for box in boxes[lo:hi]),
                max(box[2] for box in boxes[lo:hi]),
                max(box[3] for box in boxes[lo:hi]),
            )
        )
    return out


def _clamp_box(
    box: tuple[float, float, float, float],
    height: int,
    width: int,
) -> tuple[float, float, float, float]:
    y0, x0, y1, x1 = box
    box_h = max(1.0, y1 - y0)
    box_w = max(1.0, x1 - x0)
    cy = (y0 + y1) * 0.5
    cx = (x0 + x1) * 0.5
    y0, y1 = cy - box_h * 0.5, cy + box_h * 0.5
    x0, x1 = cx - box_w * 0.5, cx + box_w * 0.5

    if y0 < 0:
        y1 -= y0
        y0 = 0.0
    if x0 < 0:
        x1 -= x0
        x0 = 0.0
    if y1 > height:
        y0 -= y1 - height
        y1 = float(height)
    if x1 > width:
        x0 -= x1 - width
        x1 = float(width)
    return max(0.0, y0), max(0.0, x0), min(float(height), y1), min(float(width), x1)


def _box_mask(
    box: tuple[float, float, float, float],
    height: int,
    width: int,
    shape: str,
    device: torch.device,
) -> torch.Tensor:
    y0, x0, y1, x1 = _clamp_box(box, height, width)
    yy, xx = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
    inside = (yy.float() >= y0) & (yy.float() <= y1) & (xx.float() >= x0) & (xx.float() <= x1)
    if shape == "bbox":
        return inside
    if shape != "capsule":
        raise ValueError("coarse_shape must be 'bbox' or 'capsule'")

    cy = (y0 + y1) * 0.5
    cx = (x0 + x1) * 0.5
    ry = max((y1 - y0) * 0.5, 0.5)
    rx = max((x1 - x0) * 0.5, 0.5)
    abs_y = (yy.float() - cy).abs()
    abs_x = (xx.float() - cx).abs()
    if rx >= ry:
        core = max(rx - ry, 0.0)
        return inside & ((abs_x <= core) | ((abs_x - core).clamp_min(0.0).square() + abs_y.square() <= ry * ry))
    core = max(ry - rx, 0.0)
    return inside & ((abs_y <= core) | ((abs_y - core).clamp_min(0.0).square() + abs_x.square() <= rx * rx))


def _frame_box(
    frame_score: torch.Tensor,
    global_seed_frame: torch.Tensor,
    config: CoarseMaskConfig,
) -> tuple[tuple[float, float, float, float], Optional[tuple[float, float]], int, int]:
    height, width = frame_score.shape
    seed = global_seed_frame.bool()
    if int(seed.sum().item()) < config.coarse_min_seed_tokens:
        seed = coarse_top_fraction_mask(frame_score, config.coarse_frame_top_p)

    ys, xs = torch.where(seed)
    if ys.numel() == 0:
        return (0.0, 0.0, float(height), float(width)), None, 0, 0

    weights = frame_score[ys, xs].float().clamp_min(1e-6)
    cy = float((ys.float() * weights).sum().item() / weights.sum().item())
    cx = float((xs.float() * weights).sum().item() / weights.sum().item())
    seed_h = float(ys.max().item() - ys.min().item() + 1)
    seed_w = float(xs.max().item() - xs.min().item() + 1)

    box_h = max(seed_h * config.coarse_expand_h, height * config.coarse_min_height_ratio)
    box_w = max(seed_w * config.coarse_expand_w, width * config.coarse_min_width_ratio)
    area = box_h * box_w
    min_area = height * width * config.coarse_min_area_ratio
    max_area = height * width * config.coarse_max_area_ratio
    if area < min_area:
        scale = math.sqrt(min_area / max(area, 1.0))
        box_h *= scale
        box_w *= scale
    if box_h * box_w > max_area:
        scale = math.sqrt(max_area / max(box_h * box_w, 1.0))
        box_h *= scale
        box_w *= scale

    box_h = min(box_h, float(height))
    box_w = min(box_w, float(width))
    box = (cy - box_h * 0.5, cx - box_w * 0.5, cy + box_h * 0.5, cx + box_w * 0.5)
    return _clamp_box(box, height, width), (cy, cx), int(seed.sum().item()), int(global_seed_frame.sum().item())


def _center_boxes(
    boxes: list[tuple[float, float, float, float]],
    centers: list[tuple[float, float]],
    height: int,
    width: int,
) -> list[tuple[float, float, float, float]]:
    out = []
    for box, (cy, cx) in zip(boxes, centers):
        box_h = box[2] - box[0]
        box_w = box[3] - box[1]
        out.append(_clamp_box((cy - box_h * 0.5, cx - box_w * 0.5, cy + box_h * 0.5, cx + box_w * 0.5), height, width))
    return out


def _apply_size_rule(
    boxes: list[tuple[float, float, float, float]],
    centers: list[tuple[float, float]],
    height: int,
    width: int,
    size_rule: str,
) -> list[tuple[float, float, float, float]]:
    if size_rule == "per-frame":
        return boxes
    if size_rule != "max":
        raise ValueError("coarse_size_rule must be 'per-frame' or 'max'")
    max_h = max(box[2] - box[0] for box in boxes)
    max_w = max(box[3] - box[1] for box in boxes)
    return [_clamp_box((cy - max_h * 0.5, cx - max_w * 0.5, cy + max_h * 0.5, cx + max_w * 0.5), height, width) for cy, cx in centers]


def build_attention_step_mask(score: torch.Tensor, config: CoarseMaskConfig) -> dict[str, Any]:
    """Build one completed-step attention mask token mask from a `[T, H, W]` score tensor."""

    if score.ndim != 3:
        raise ValueError(f"score must have shape [T,H,W], got {tuple(score.shape)}")

    score = minmax_norm(score) if config.score_normalization == "minmax" else score.float()
    seed = coarse_top_fraction_mask(score, config.top_p)
    height, width = int(score.shape[1]), int(score.shape[2])

    boxes = []
    centers = []
    seed_counts = []
    global_seed_counts = []
    for t in range(score.shape[0]):
        box, center, seed_count, global_seed_count = _frame_box(score[t], seed[t], config)
        boxes.append(box)
        centers.append(center)
        seed_counts.append(seed_count)
        global_seed_counts.append(global_seed_count)

    filled_centers = _filled_centers(centers)
    boxes = _center_boxes(boxes, filled_centers, height, width)
    boxes = _apply_size_rule(boxes, filled_centers, height, width, config.coarse_size_rule)
    boxes = _smooth_boxes(boxes, config.coarse_smooth_radius)
    boxes = _union_boxes(boxes, config.coarse_temporal_union_radius)
    mbase = torch.stack(
        [_box_mask(box, height, width, config.coarse_shape, score.device) for box in boxes],
        dim=0,
    ).bool()

    areas = [float(mask.float().mean().item()) for mask in mbase]
    best_frame = int(max(range(len(areas)), key=lambda idx: areas[idx]))
    return {
        "score": score,
        "raw_coverage": seed,
        "Mbase": mbase,
        "best_frame": best_frame,
        "boxes": [[float(v) for v in box] for box in boxes],
        "centers": [[float(y), float(x)] for y, x in filled_centers],
        "seed_counts": seed_counts,
        "global_seed_counts": global_seed_counts,
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


def build_canonical_cross_attention_mask(
    records: list[dict[str, Any]],
    config: CoarseMaskConfig,
    *,
    completed_step: int,
) -> dict[str, Any]:
    """Build the one source-calibrated cross-attention mask used by native Wan."""

    selected = [item for item in records if int(item["step_index"]) + 1 == int(completed_step)]
    if not selected:
        raise RuntimeError(f"missing probe records for completed_step={completed_step}")
    score = torch.stack([(item.get("noun_score", item.get("positive"))).float() for item in selected]).mean(dim=0)
    built = build_attention_step_mask(score, config)
    mbase = built["Mbase"]
    medit = dilate_spatial(mbase, config.spatial_medit_radius)
    msee = dilate_spatial(medit, config.spatial_msee_radius)
    attention_config = asdict(config)
    attention_config.pop("mask_steps", None)
    return {
        "attention_mask_mode": "single-step",
        "mask_variant": "offline_cross_attention_probe",
        "source_completed_step": int(completed_step),
        "attention_config": attention_config,
        "attention_layer_fusion_rule": "mean_across_probe_records",
        "score": built["score"],
        "raw_coverage": built["raw_coverage"],
        "Mbase": mbase,
        "Medit": medit,
        "Msee": msee,
        "token_area": _mask_area(mbase, medit, msee),
        "coarse": {
            "shape": config.coarse_shape,
            "best_frame_for_mbase": int(built["best_frame"]),
            "boxes": built["boxes"],
            "centers": built["centers"],
            "seed_counts": built["seed_counts"],
            "global_seed_counts": built["global_seed_counts"],
        },
        "records": [
            {
                "step_index": int(item["step_index"]),
                "completed_step": int(item["step_index"]) + 1,
                "layer_index": int(item["layer_index"]),
            }
            for item in selected
        ],
    }


def add_canonical_latent_masks(bundle: dict[str, Any], latents: torch.Tensor) -> dict[str, Any]:
    """Materialize the sole native Wan region mask in latent space."""

    if "Medit" not in bundle:
        raise KeyError("missing canonical token mask Medit")
    medit = bundle["Medit"]
    msee = bundle.get("Msee", medit)
    bundle["latent_Medit"] = token_mask_to_latent_mask(medit, latents)
    bundle["latent_Msee"] = token_mask_to_latent_mask(msee, latents)
    bundle["latent_area"] = {
        "medit": float(bundle["latent_Medit"].float().mean().item()),
        "msee": float(bundle["latent_Msee"].float().mean().item()),
    }
    return bundle


def get_canonical_latent_masks(bundle: dict[str, Any], latents: torch.Tensor) -> dict[str, Any]:
    """Return the fixed source-calibrated mask for any native Wan target step."""

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
        "latent_medit": latent_medit,
        "latent_msee": latent_msee,
        "source_completed_step": int(bundle["source_completed_step"]),
    }


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


def build_masks_from_records(
    records: list[dict[str, Any]],
    config: CoarseMaskConfig,
) -> dict[str, Any]:
    if not records:
        raise ValueError("records is empty")

    by_step: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        completed_step = int(record["step_index"]) + 1
        by_step.setdefault(completed_step, []).append(record)

    bundle: dict[str, Any] = {
        "attention_mask_mode": "attention-stepwise-coarse",
        "mask_variant": "offline_cross_attention_probe",
        "attention_config": asdict(config),
        "attention_layer_fusion_rule": "mean_across_probe_records",
        "mask_steps": list(config.mask_steps),
        "step_metadata": {},
    }

    for completed_step in config.mask_steps:
        step_records = by_step.get(int(completed_step))
        if not step_records:
            raise RuntimeError(f"missing probe records for completed_step={completed_step}")

        score = torch.stack([(item.get("noun_score", item.get("positive"))).float() for item in step_records]).mean(dim=0)
        built = build_attention_step_mask(score, config)
        mbase = built["Mbase"]
        medit = dilate_spatial(mbase, config.spatial_medit_radius)
        msee = dilate_spatial(medit, config.spatial_msee_radius)

        step_key = f"step{completed_step:03d}"
        bundle[f"{step_key}_score"] = built["score"]
        bundle[f"{step_key}_raw_coverage"] = built["raw_coverage"]
        bundle[f"{step_key}_Mbase"] = mbase
        bundle[f"{step_key}_Medit"] = medit
        bundle[f"{step_key}_Msee"] = msee

        layers = sorted({int(item["layer_index"]) for item in step_records})
        bundle["step_metadata"][str(completed_step)] = {
            "record_count": len(step_records),
            "probe_layers": layers,
            "best_frame_for_mbase": int(built["best_frame"]),
            "token_area": {
                "mbase": float(mbase.float().mean().item()),
                "medit": float(medit.float().mean().item()),
                "msee": float(msee.float().mean().item()),
            },
            "coarse": {
                "shape": config.coarse_shape,
                "boxes": built["boxes"],
                "centers": built["centers"],
                "seed_counts": built["seed_counts"],
                "global_seed_counts": built["global_seed_counts"],
            },
        }

    bundle["records"] = [
        {
            "step_index": int(item["step_index"]),
            "completed_step": int(item["step_index"]) + 1,
            "layer_index": int(item["layer_index"]),
        }
        for item in records
    ]
    return bundle


def build_coarse_masks_from_records(records: list[dict[str, Any]], config: CoarseMaskConfig) -> dict[str, Any]:
    return build_masks_from_records(records, config)


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
