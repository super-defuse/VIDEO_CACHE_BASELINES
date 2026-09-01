"""Carnator Early Signatures captured at completed step 5.

The native collector decodes the step-5 latent once, persists its tail frame,
and records mean CIELAB colour under each noun's native ``Mbase`` mask. Shared
nouns use maximum CIEDE2000 distance; a single noun substitution additionally
uses area, normalized shape, and colour distance.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image
from safetensors.torch import load_file, save_file
from skimage.color import deltaE_ciede2000, rgb2lab


EARLY_SIGNATURE_SCHEMA = "carnator-online-color-signature"
EARLY_SIGNATURE_PROFILE_ID = "carnator-step5-tail-lab-ciede2000"
PER_NOUN_INFERENCE_MASK_CONTRACT = "carnator-source-per-noun-mbase/v1"
COLOR_EXTRACTION_POLICY = "srgb-d65-2deg-per-pixel-lab-mean"
COLOR_SHAPE_POLICY = "shape_norm"
_COMPATIBILITY_FIELDS = (
    "model",
    "scheduler",
    "scheduler_config",
    "cfg",
    "probe_seed",
    "completed_step",
    "latent_shape",
    "latent_token_grid",
    "noun_extractor",
    "mask_policy",
)


class ColorSignatureError(ValueError):
    """Raised when a colour signature cannot safely authorize reuse."""


def _without_version_suffix(value: object) -> object:
    if isinstance(value, str):
        return re.sub(r"(?:/|_)v[0-9]+$", "", value)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_noun(noun: str) -> str:
    value = " ".join(str(noun).strip().lower().split())
    if not value:
        raise ColorSignatureError("noun must be non-empty")
    return value


def _require_step5_latent(latent: torch.Tensor) -> torch.Tensor:
    if not isinstance(latent, torch.Tensor) or latent.ndim != 4 or not latent.is_floating_point():
        raise ColorSignatureError("step5_latent must be a floating [C,T,H,W] tensor")
    value = latent.detach().float().cpu().contiguous()
    if not torch.isfinite(value).all():
        raise ColorSignatureError("step5_latent contains non-finite values")
    return value


def _validate_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(metadata)
    missing = [field for field in _COMPATIBILITY_FIELDS if field not in result]
    if missing:
        raise ColorSignatureError(f"signature metadata is missing {missing}")
    if result["completed_step"] != 5:
        raise ColorSignatureError("colour online signature must be captured at completed step 5")
    for field in ("latent_shape", "latent_token_grid"):
        value = result[field]
        if not isinstance(value, list) or len(value) != (4 if field == "latent_shape" else 3):
            raise ColorSignatureError(f"metadata.{field} has an invalid shape")
        result[field] = [int(item) for item in value]
    if not isinstance(result["scheduler_config"], Mapping):
        raise ColorSignatureError("metadata.scheduler_config must be an object")
    if isinstance(result["cfg"], bool) or not isinstance(result["cfg"], (int, float)):
        raise ColorSignatureError("metadata.cfg must be numeric")
    if isinstance(result["probe_seed"], bool) or not isinstance(result["probe_seed"], int):
        raise ColorSignatureError("metadata.probe_seed must be an integer")
    for field in ("model", "scheduler", "noun_extractor", "mask_policy"):
        if not isinstance(result[field], str) or not result[field].strip():
            raise ColorSignatureError(f"metadata.{field} must be a non-empty string")
    return result


def _tail_frame_to_rgb(frame: torch.Tensor) -> np.ndarray:
    if not isinstance(frame, torch.Tensor) or frame.ndim != 3 or frame.shape[0] < 3 or not frame.is_floating_point():
        raise ColorSignatureError("decoded_tail_frame must be a floating [C,H,W] tensor with at least three channels")
    value = frame.detach().float().cpu().contiguous()
    if not torch.isfinite(value).all():
        raise ColorSignatureError("decoded_tail_frame contains non-finite values")
    rgb = ((value[:3].clamp(-1.0, 1.0) + 1.0) * 127.5).round().to(torch.uint8)
    return rgb.permute(1, 2, 0).numpy()


def _read_rgb_frame(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"))
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ColorSignatureError("decoded tail frame is not RGB")
    return rgb


def _tail_mask(mask: torch.Tensor) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor) or mask.ndim != 3 or mask.dtype != torch.bool:
        raise ColorSignatureError("Mbase must be a bool [T,H,W] tensor")
    value = mask.detach().cpu().contiguous()
    if not bool(value.any()):
        raise ColorSignatureError("Mbase is empty")
    tail = value[-1]
    if not bool(tail.any()):
        raise ColorSignatureError("Mbase is empty at the tail signature token")
    return tail


def _pixel_mask(mask: torch.Tensor, height: int, width: int) -> np.ndarray:
    resized = functional.interpolate(
        mask.unsqueeze(0).unsqueeze(0).float(), size=(height, width), mode="nearest"
    )[0, 0]
    return resized.bool().numpy()


def _mask_record(mask: torch.Tensor, noun: str, tensor_key: str, rgb: np.ndarray) -> dict[str, Any]:
    tail = _tail_mask(mask)
    ys, xs = torch.where(tail)
    pixel_mask = _pixel_mask(tail, rgb.shape[0], rgb.shape[1])
    pixels = rgb[pixel_mask]
    if pixels.size == 0:
        raise ColorSignatureError("Mbase projects to an empty decoded tail-frame region")
    lab = rgb2lab(pixels.astype(np.float32) / 255.0, illuminant="D65", observer="2")
    mean_lab = lab.mean(axis=0)
    mean_rgb = pixels.astype(np.float32).mean(axis=0)
    return {
        "noun": noun,
        "tensor_key": tensor_key,
        "shape": [int(item) for item in mask.shape],
        "dtype": "bool",
        "area_frac": float(tail.float().mean().item()),
        "centroid_yx": [float(ys.float().mean().item()), float(xs.float().mean().item())],
        "signature_token_t": int(mask.shape[0] - 1),
        "color_signature": {
            "space": "CIELAB",
            "input_space": "sRGB",
            "illuminant": "D65",
            "observer": "2deg",
            "aggregation": "mean_of_per_pixel_lab",
            "mean_lab": [float(item) for item in mean_lab],
            "mean_rgb": [float(item) for item in mean_rgb],
            "valid_pixel_count": int(pixels.shape[0]),
            "source": "step5_decoded_tail_frame_projected_from_mbase",
        },
    }


def write_early_signature(
    directory: Path,
    *,
    step5_latent: torch.Tensor,
    decoded_tail_frame: torch.Tensor,
    metadata: Mapping[str, Any],
    mbase_by_noun: Mapping[str, torch.Tensor],
    cache: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist the colour-aware signature for a source or query probe."""

    step5 = _require_step5_latent(step5_latent)
    validated_metadata = _validate_metadata(metadata)
    if validated_metadata["latent_shape"] != list(step5.shape):
        raise ColorSignatureError("metadata latent_shape does not match step5_latent")
    if not mbase_by_noun:
        raise ColorSignatureError("a colour signature requires one Mbase per canonical noun")
    rgb = _tail_frame_to_rgb(decoded_tail_frame)
    directory = directory.expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)

    frame_path = directory / "step5_tail_frame.png"
    Image.fromarray(rgb).save(frame_path)

    masks: dict[str, torch.Tensor] = {}
    objects: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, (raw_noun, mask) in enumerate(sorted(mbase_by_noun.items())):
        noun = _canonical_noun(raw_noun)
        if noun in seen:
            raise ColorSignatureError(f"duplicate canonical noun {noun!r}")
        seen.add(noun)
        tensor_key = f"mbase_{index:03d}"
        record = _mask_record(mask, noun, tensor_key, rgb)
        if record["shape"] != validated_metadata["latent_token_grid"]:
            raise ColorSignatureError(f"Mbase for {noun!r} does not match latent_token_grid")
        masks[tensor_key] = mask.detach().cpu().contiguous()
        objects.append(record)
    mask_path = directory / "mbase.safetensors"
    save_file(masks, str(mask_path), metadata={"policy": str(validated_metadata["mask_policy"])})

    payload = {
        "signature_version": EARLY_SIGNATURE_SCHEMA,
        "feature_profile": EARLY_SIGNATURE_PROFILE_ID,
        "prompt": str(metadata.get("prompt", "")),
        "compatibility": {field: validated_metadata[field] for field in _COMPATIBILITY_FIELDS},
        "frame": {
            "path": frame_path.name,
            "sha256": _sha256(frame_path),
            "shape": [int(item) for item in rgb.shape],
            "frame_policy": "tail_frame",
            "signature_token_t": int(validated_metadata["latent_token_grid"][0] - 1),
            "decoded_completed_step": 5,
            "step5_latent_sha256": hashlib.sha256(step5.numpy().tobytes()).hexdigest(),
        },
        "color_extraction_policy": COLOR_EXTRACTION_POLICY,
        "shape_normalization_policy": COLOR_SHAPE_POLICY,
        "inference_mask_contract": PER_NOUN_INFERENCE_MASK_CONTRACT,
        "mbase": {"path": mask_path.name, "sha256": _sha256(mask_path)},
        "objects": objects,
        "cache": dict(cache) if cache is not None else None,
    }
    signature_path = directory / "online_color_signature.json"
    signature_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"signature_path": str(signature_path)}


def _signature_path(signature: Mapping[str, Any]) -> Path:
    value = signature.get("signature_path")
    if not isinstance(value, str) or not value.strip():
        raise ColorSignatureError("signature needs a readable signature_path")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ColorSignatureError(f"signature sidecar is missing: {path}")
    return path


def _load_signature(
    signature: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, torch.Tensor], dict[str, tuple[Mapping[str, Any], torch.Tensor]]]:
    path = _signature_path(signature)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ColorSignatureError(f"cannot read signature sidecar: {path}") from exc
    if _without_version_suffix(payload.get("signature_version")) != EARLY_SIGNATURE_SCHEMA:
        raise ColorSignatureError("colour signature version mismatch")
    if _without_version_suffix(payload.get("feature_profile")) != EARLY_SIGNATURE_PROFILE_ID:
        raise ColorSignatureError("colour signature profile mismatch")
    compatibility = payload.get("compatibility")
    if not isinstance(compatibility, Mapping):
        raise ColorSignatureError("signature compatibility metadata is missing")
    _validate_metadata(compatibility)
    if _without_version_suffix(payload.get("color_extraction_policy")) != COLOR_EXTRACTION_POLICY:
        raise ColorSignatureError("colour extraction policy mismatch")
    if _without_version_suffix(payload.get("shape_normalization_policy")) != COLOR_SHAPE_POLICY:
        raise ColorSignatureError("shape normalization policy mismatch")

    frame_info = payload.get("frame")
    if not isinstance(frame_info, Mapping) or frame_info.get("decoded_completed_step") != 5:
        raise ColorSignatureError("step-5 decoded tail frame metadata is missing")
    if frame_info.get("frame_policy") != "tail_frame":
        raise ColorSignatureError("colour signature must use the tail frame")
    if frame_info.get("signature_token_t") != compatibility["latent_token_grid"][0] - 1:
        raise ColorSignatureError("tail frame token index does not match latent_token_grid")
    frame_path = path.parent / str(frame_info.get("path", ""))
    if not frame_path.is_file() or _sha256(frame_path) != frame_info.get("sha256"):
        raise ColorSignatureError("decoded tail frame is missing or its hash changed")
    rgb = _read_rgb_frame(frame_path)
    if frame_info.get("shape") != list(rgb.shape):
        raise ColorSignatureError("decoded tail frame metadata mismatch")

    mask_info = payload.get("mbase")
    if not isinstance(mask_info, Mapping):
        raise ColorSignatureError("Mbase metadata is missing")
    mask_path = path.parent / str(mask_info.get("path", ""))
    if not mask_path.is_file() or _sha256(mask_path) != mask_info.get("sha256"):
        raise ColorSignatureError("Mbase is missing or its hash changed")
    masks = load_file(str(mask_path), device="cpu")
    raw_objects = payload.get("objects")
    if not isinstance(raw_objects, list) or not raw_objects:
        raise ColorSignatureError("signature objects are missing")
    objects: dict[str, tuple[Mapping[str, Any], torch.Tensor]] = {}
    for item in raw_objects:
        if not isinstance(item, Mapping):
            raise ColorSignatureError("signature object is invalid")
        noun = _canonical_noun(str(item.get("noun", "")))
        if noun in objects:
            raise ColorSignatureError(f"duplicate noun {noun!r} in signature")
        key = item.get("tensor_key")
        mask = masks.get(key) if isinstance(key, str) else None
        if mask is None or mask.dtype != torch.bool or list(mask.shape) != item.get("shape"):
            raise ColorSignatureError("Mbase tensor metadata mismatch")
        if list(mask.shape) != compatibility["latent_token_grid"]:
            raise ColorSignatureError("Mbase tensor does not match latent_token_grid")
        record = _mask_record(mask, noun, str(key), rgb)
        for field in ("shape", "dtype", "area_frac", "centroid_yx", "signature_token_t", "color_signature"):
            if item.get(field) != record[field]:
                raise ColorSignatureError("Mbase or colour derived metadata mismatch")
        objects[noun] = (item, mask[-1].detach().cpu().contiguous())
    return payload, masks, objects


def load_per_noun_mbase(signature: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Load the source-side Mbase tensors authorized for per-noun reuse."""

    payload, masks, objects = _load_signature(signature)
    if payload.get("inference_mask_contract") != PER_NOUN_INFERENCE_MASK_CONTRACT:
        raise ColorSignatureError("signature lacks the per-noun inference-mask contract")
    result: dict[str, torch.Tensor] = {}
    for noun, (record, _tail_mask) in objects.items():
        key = record.get("tensor_key")
        mask = masks.get(key) if isinstance(key, str) else None
        if mask is None:
            raise ColorSignatureError(f"Mbase tensor is missing for {noun!r}")
        result[noun] = mask.detach().cpu().bool().contiguous()
    return result


def per_noun_mbase_is_valid(
    signature: Mapping[str, Any], *, expected_nouns: Sequence[str] | None = None
) -> bool:
    """Return whether a sidecar can safely supply every expected source noun."""

    try:
        masks = load_per_noun_mbase(signature)
        if expected_nouns is not None:
            expected = {_canonical_noun(noun) for noun in expected_nouns}
            if set(masks) != expected:
                return False
        return bool(masks)
    except (OSError, RuntimeError, ColorSignatureError, ValueError, KeyError, TypeError):
        return False


def select_per_noun_inference_mbase(
    signature: Mapping[str, Any], *, target_nouns: Sequence[str]
) -> tuple[dict[str, Any], torch.Tensor]:
    """Choose the source mask for a compatible Carnator cache hit.

    A one-noun substitution regenerates the source-only object region. When
    noun sets are equal, the smallest valid source mask is the deterministic,
    conservative fallback.
    """

    masks = load_per_noun_mbase(signature)
    source_names = set(masks)
    target_names = {_canonical_noun(noun) for noun in target_nouns}
    source_only = sorted(source_names - target_names)
    target_only = sorted(target_names - source_names)
    if len(source_only) == len(target_only) == 1:
        selected = source_only[0]
        reason = "single_noun_substitution_source_only"
    elif not source_only and not target_only:
        selected = min(masks, key=lambda noun: (int(masks[noun].sum().item()), noun))
        reason = "same_noun_set_smallest_area"
    else:
        raise ColorSignatureError(
            "per-noun inference requires equal noun sets or exactly one noun substitution"
        )
    mask = masks[selected]
    area_tokens = int(mask.sum().item())
    if area_tokens <= 0:
        raise ColorSignatureError(f"selected Mbase is empty for {selected!r}")
    return {
        "contract": PER_NOUN_INFERENCE_MASK_CONTRACT,
        "selected_source_noun": selected,
        "selection_reason": reason,
        "source_nouns": sorted(source_names),
        "target_nouns": sorted(target_names),
        "source_only_nouns": source_only,
        "target_only_nouns": target_only,
        "selected_mask_area_tokens": area_tokens,
        "selected_mask_area_fraction": float(mask.float().mean().item()),
    }, mask


def materialize_per_noun_inference_mask_bundle(
    signature: Mapping[str, Any],
    *,
    target_nouns: Sequence[str],
    source_fusion_mask_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Write the selected source noun mask into a request-private bundle."""

    from carnator.masks import (
        build_per_noun_mbase_bundle,
        load_mask_bundle,
        save_mask_bundle,
    )

    selection, mbase = select_per_noun_inference_mbase(signature, target_nouns=target_nouns)
    template = load_mask_bundle(source_fusion_mask_dir, None)
    try:
        source_completed_step = int(template["source_completed_step"])
        medit_radius = int(template["mask_spatial_medit_radius"])
        msee_radius = int(template["mask_spatial_msee_radius"])
        token_shape = [int(item) for item in template["token_shape"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise ColorSignatureError("source fusion-mask metadata is incomplete") from exc
    if list(mbase.shape) != token_shape:
        raise ColorSignatureError("selected Mbase does not match source fusion-mask token shape")
    bundle = build_per_noun_mbase_bundle(
        mbase,
        source_completed_step=source_completed_step,
        spatial_medit_radius=medit_radius,
        spatial_msee_radius=msee_radius,
        selection=selection,
    )
    output_dir = output_dir.expanduser().resolve()
    save_mask_bundle(output_dir, bundle, save_cpu=True)
    return {**selection, "mask_bundle_dir": str(output_dir)}


def _normalized_extent(mask: np.ndarray) -> tuple[float, float, float, float, float, float, float]:
    ys, xs = np.where(mask)
    area = float(len(xs))
    if area <= 0:
        raise ColorSignatureError("tail Mbase shape is empty")
    centroid_y = float(np.mean(ys + 0.5))
    centroid_x = float(np.mean(xs + 0.5))
    scale = math.sqrt(area)
    return (
        (float(ys.min()) - centroid_y) / scale,
        (float(ys.max() + 1) - centroid_y) / scale,
        (float(xs.min()) - centroid_x) / scale,
        (float(xs.max() + 1) - centroid_x) / scale,
        centroid_y,
        centroid_x,
        scale,
    )


def _rasterize_normalized_mask(
    mask: np.ndarray,
    y_pixels: np.ndarray,
    x_pixels: np.ndarray,
    *,
    factor: int,
    centroid_y: float,
    centroid_x: float,
    scale: float,
) -> np.ndarray:
    source_y = ((y_pixels.astype(np.float64) + 0.5) / factor) * scale + centroid_y
    source_x = ((x_pixels.astype(np.float64) + 0.5) / factor) * scale + centroid_x
    source_y = np.floor(source_y).astype(np.int64)
    source_x = np.floor(source_x).astype(np.int64)
    valid_y = (source_y >= 0) & (source_y < mask.shape[0])
    valid_x = (source_x >= 0) & (source_x < mask.shape[1])
    result = np.zeros((len(y_pixels), len(x_pixels)), dtype=bool)
    if valid_y.any() and valid_x.any():
        result[np.ix_(valid_y, valid_x)] = mask[np.ix_(source_y[valid_y], source_x[valid_x])]
    return result


def _shape_distance(query: torch.Tensor, candidate: torch.Tensor) -> float:
    query_mask = query.numpy().astype(bool, copy=False)
    candidate_mask = candidate.numpy().astype(bool, copy=False)
    query_extent = _normalized_extent(query_mask)
    candidate_extent = _normalized_extent(candidate_mask)
    factor, padding = 8, 8
    y_pixels = np.arange(math.floor(min(query_extent[0], candidate_extent[0]) * factor) - padding,
                         math.ceil(max(query_extent[1], candidate_extent[1]) * factor) + padding,
                         dtype=np.int64)
    x_pixels = np.arange(math.floor(min(query_extent[2], candidate_extent[2]) * factor) - padding,
                         math.ceil(max(query_extent[3], candidate_extent[3]) * factor) + padding,
                         dtype=np.int64)
    if not len(y_pixels) or not len(x_pixels):
        raise ColorSignatureError("normalized tail Mbase canvas is empty")
    left = _rasterize_normalized_mask(
        query_mask, y_pixels, x_pixels, factor=factor, centroid_y=query_extent[4], centroid_x=query_extent[5], scale=query_extent[6]
    )
    right = _rasterize_normalized_mask(
        candidate_mask, y_pixels, x_pixels, factor=factor, centroid_y=candidate_extent[4], centroid_x=candidate_extent[5], scale=candidate_extent[6]
    )
    denominator = int(left.sum() + right.sum())
    if denominator <= 0:
        raise ColorSignatureError("normalized tail Mbase shape is empty")
    return 1.0 - (2.0 * int(np.logical_and(left, right).sum()) / denominator)


def _ciede2000(left: Sequence[float], right: Sequence[float]) -> float:
    value = deltaE_ciede2000(
        np.asarray(left, dtype=np.float64)[np.newaxis, :],
        np.asarray(right, dtype=np.float64)[np.newaxis, :],
        kL=1,
        kC=1,
        kH=1,
    )
    result = float(np.asarray(value).reshape(-1)[0])
    if not math.isfinite(result):
        raise ColorSignatureError("CIEDE2000 returned a non-finite value")
    return result


def _cache_ready(payload: Mapping[str, Any], *, check_artifacts: bool) -> tuple[bool, str | None]:
    cache = payload.get("cache")
    if not isinstance(cache, Mapping) or cache.get("anchor_step") != 10:
        return False, "missing_step10_artifacts"
    compatibility = payload.get("compatibility")
    if not isinstance(compatibility, Mapping) or (
        cache.get("model") != compatibility.get("model")
        or cache.get("latent_token_grid") != compatibility.get("latent_token_grid")
    ):
        return False, "cache_metadata_mismatch"
    for key in ("latent_path", "latent_sha256", "sparse_attention_path", "sparse_attention_sha256"):
        if not isinstance(cache.get(key), str) or not str(cache[key]).strip():
            return False, "missing_step10_artifacts"
    if not check_artifacts:
        return True, None
    for path_key, hash_key in (("latent_path", "latent_sha256"), ("sparse_attention_path", "sparse_attention_sha256")):
        artifact = Path(str(cache[path_key])).expanduser()
        if not artifact.is_absolute() or not artifact.is_file() or _sha256(artifact) != cache[hash_key]:
            return False, "step10_artifact_unreadable_or_changed"
    return True, None


def compare_early_signatures(
    query_signature: Mapping[str, Any],
    candidate_signature: Mapping[str, Any],
    *,
    query_changed_noun: str | None = None,
    candidate_changed_noun: str | None = None,
    max_context_color_distance: float | None = None,
    max_log_area_ratio: float | None = None,
    max_shape_distance: float | None = None,
    max_changed_color_distance: float | None = None,
    check_cache_artifacts: bool = True,
) -> dict[str, Any]:
    """Compare step-5 colour signatures and fail closed on errors."""

    result: dict[str, Any] = {
        "profile": EARLY_SIGNATURE_PROFILE_ID,
        "metadata_compatible": False,
        "distances": {
            "context_color_ciede2000": None,
            "log_area_ratio": None,
            "shape_one_minus_dice": None,
            "changed_color_ciede2000": None,
        },
        "per_same_noun_context_color_ciede2000": {},
        "gates": {
            "context_color_ciede2000": None,
            "log_area_ratio": None,
            "shape_one_minus_dice": None,
            "changed_color_ciede2000": None,
        },
        "signature_compatible": False,
        "cache_ready": False,
        "reusable": False,
        "reason": None,
    }
    try:
        query, _query_masks, query_objects = _load_signature(query_signature)
        candidate, _candidate_masks, candidate_objects = _load_signature(candidate_signature)
        if query["compatibility"] != candidate["compatibility"]:
            result["reason"] = "metadata_mismatch"
            return result
        result["metadata_compatible"] = True
        query_names, candidate_names = set(query_objects), set(candidate_objects)
        query_only, candidate_only = query_names - candidate_names, candidate_names - query_names
        substitution = query_changed_noun is not None or candidate_changed_noun is not None
        if substitution:
            if not query_changed_noun or not candidate_changed_noun:
                result["reason"] = "invalid_noun_substitution"
                return result
            if query_only != {_canonical_noun(query_changed_noun)} or candidate_only != {_canonical_noun(candidate_changed_noun)}:
                result["reason"] = "noun_substitution_mismatch"
                return result
        elif query_only or candidate_only:
            result["reason"] = "noun_set_mismatch"
            return result

        same_nouns = sorted(query_names & candidate_names)
        per_same = {
            noun: _ciede2000(
                query_objects[noun][0]["color_signature"]["mean_lab"],
                candidate_objects[noun][0]["color_signature"]["mean_lab"],
            )
            for noun in same_nouns
        }
        context_distance = max(per_same.values(), default=0.0)
        result["per_same_noun_context_color_ciede2000"] = per_same
        result["distances"]["context_color_ciede2000"] = context_distance
        result["gates"]["context_color_ciede2000"] = (
            max_context_color_distance is not None and context_distance <= float(max_context_color_distance)
        )

        if substitution:
            query_object = query_objects[_canonical_noun(str(query_changed_noun))]
            candidate_object = candidate_objects[_canonical_noun(str(candidate_changed_noun))]
            query_metadata, query_mask = query_object
            candidate_metadata, candidate_mask = candidate_object
            area_distance = abs(math.log(float(query_metadata["area_frac"]) / float(candidate_metadata["area_frac"])))
            shape_distance = _shape_distance(query_mask, candidate_mask)
            changed_color = _ciede2000(
                query_metadata["color_signature"]["mean_lab"],
                candidate_metadata["color_signature"]["mean_lab"],
            )
            result["distances"].update({
                "log_area_ratio": area_distance,
                "shape_one_minus_dice": shape_distance,
                "changed_color_ciede2000": changed_color,
            })
            result["gates"].update({
                "log_area_ratio": max_log_area_ratio is not None and area_distance <= float(max_log_area_ratio),
                "shape_one_minus_dice": max_shape_distance is not None and shape_distance <= float(max_shape_distance),
                "changed_color_ciede2000": (
                    max_changed_color_distance is not None and changed_color <= float(max_changed_color_distance)
                ),
            })

        applicable = ["context_color_ciede2000"]
        if substitution:
            applicable.extend(("log_area_ratio", "shape_one_minus_dice", "changed_color_ciede2000"))
        result["signature_compatible"] = all(result["gates"][name] is True for name in applicable)
        cache_ready, cache_reason = _cache_ready(candidate, check_artifacts=check_cache_artifacts)
        result["cache_ready"] = cache_ready
        result["reusable"] = result["signature_compatible"] and cache_ready
        result["reason"] = "accepted" if result["reusable"] else (cache_reason or "gate_failed")
        return result
    except (OSError, RuntimeError, ColorSignatureError, ValueError, KeyError, TypeError) as exc:
        result["reason"] = f"invalid_signature:{type(exc).__name__}"
        return result


__all__ = [
    "COLOR_EXTRACTION_POLICY",
    "COLOR_SHAPE_POLICY",
    "EARLY_SIGNATURE_PROFILE_ID",
    "PER_NOUN_INFERENCE_MASK_CONTRACT",
    "ColorSignatureError",
    "EARLY_SIGNATURE_SCHEMA",
    "compare_early_signatures",
    "load_per_noun_mbase",
    "materialize_per_noun_inference_mask_bundle",
    "per_noun_mbase_is_valid",
    "select_per_noun_inference_mbase",
    "write_early_signature",
]
