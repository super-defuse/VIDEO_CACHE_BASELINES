"""Carnator's native step-5 online-retrieval signature.

The signature intentionally contains no decoded video or colour feature.  It
stores one compact feature of the complete denoising state plus one native
``Mbase`` tensor per canonical noun.  All values required to make a reuse
decision are versioned and integrity checked here.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as functional
from safetensors.torch import load_file, save_file


ONLINE_SIGNATURE_VERSION = "carnator-online-signature/v1"
LATENT_FEATURE_CODEC = "latent-residual-channel-rms-multiscale-pool/v1"
LATENT_FEATURE_POOL_SHAPES = ((4, 6, 10), (8, 11, 20))
LATENT_FEATURE_PROFILE_ID = "carnator-latent-feature-v1"
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
_EPSILON = 1.0e-6


class SignatureError(ValueError):
    """Raised when a signature is malformed or incompatible."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_noun(noun: str) -> str:
    value = " ".join(str(noun).strip().lower().split())
    if not value:
        raise SignatureError("noun must be non-empty")
    return value


def _require_latent(latent: torch.Tensor, label: str) -> torch.Tensor:
    if not isinstance(latent, torch.Tensor) or latent.ndim != 4:
        raise SignatureError(f"{label} must have shape [C,T,H,W]")
    if not latent.is_floating_point():
        raise SignatureError(f"{label} must be floating point")
    if not torch.isfinite(latent).all():
        raise SignatureError(f"{label} contains non-finite values")
    return latent.detach().float().cpu().contiguous()


def encode_latent_feature(step5_latent: torch.Tensor, initial_latent: torch.Tensor) -> torch.Tensor:
    """Encode the full step-5 state into a compact, unit-length feature.

    The prompt-independent initial Gaussian field is removed before pooling.
    Per-channel RMS normalization avoids a few high-energy latent channels
    dominating the spatial-temporal comparison.
    """

    step5 = _require_latent(step5_latent, "step5_latent")
    initial = _require_latent(initial_latent, "initial_latent")
    if tuple(step5.shape) != tuple(initial.shape):
        raise SignatureError("step5_latent and initial_latent shapes differ")
    residual = step5 - initial
    rms = residual.square().mean(dim=(1, 2, 3), keepdim=True).sqrt().clamp_min(_EPSILON)
    normalized = residual / rms
    parts = [
        functional.adaptive_avg_pool3d(normalized.unsqueeze(0), output_size=shape).reshape(-1)
        for shape in LATENT_FEATURE_POOL_SHAPES
    ]
    feature = torch.cat(parts).contiguous()
    norm = torch.linalg.vector_norm(feature)
    if not torch.isfinite(norm) or float(norm) <= _EPSILON:
        raise SignatureError("latent feature has zero or non-finite norm")
    return (feature / norm).contiguous()


def cosine_distance(query: torch.Tensor, candidate: torch.Tensor) -> float:
    """Return cosine distance for two persisted, unit-length features."""

    if tuple(query.shape) != tuple(candidate.shape):
        raise SignatureError("latent feature dimensions differ")
    left, right = query.float().reshape(-1), candidate.float().reshape(-1)
    if not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise SignatureError("latent feature contains non-finite values")
    left_norm, right_norm = torch.linalg.vector_norm(left), torch.linalg.vector_norm(right)
    if float(left_norm) <= _EPSILON or float(right_norm) <= _EPSILON:
        raise SignatureError("latent feature has zero norm")
    similarity = torch.dot(left, right) / (left_norm * right_norm)
    return float((1.0 - similarity.clamp(-1.0, 1.0)).item())


def _validate_metadata(metadata: Mapping[str, Any], latent_shape: Sequence[int]) -> dict[str, Any]:
    result = dict(metadata)
    missing = [field for field in _COMPATIBILITY_FIELDS if field not in result]
    if missing:
        raise SignatureError(f"signature metadata is missing {missing}")
    if result["completed_step"] != 5:
        raise SignatureError("online signature must be captured at completed step 5")
    if list(result["latent_shape"]) != list(latent_shape):
        raise SignatureError("metadata latent_shape does not match step-5 latent")
    if not isinstance(result["latent_token_grid"], list) or len(result["latent_token_grid"]) != 3:
        raise SignatureError("latent_token_grid must be [T,H,W]")
    for field in ("model", "scheduler", "noun_extractor", "mask_policy"):
        if not isinstance(result[field], str) or not result[field].strip():
            raise SignatureError(f"metadata.{field} must be a non-empty string")
    if not isinstance(result["scheduler_config"], Mapping):
        raise SignatureError("metadata.scheduler_config must be an object")
    if isinstance(result["cfg"], bool) or not isinstance(result["cfg"], (int, float)):
        raise SignatureError("metadata.cfg must be numeric")
    if isinstance(result["probe_seed"], bool) or not isinstance(result["probe_seed"], int):
        raise SignatureError("metadata.probe_seed must be an integer")
    result["latent_shape"] = [int(value) for value in result["latent_shape"]]
    result["latent_token_grid"] = [int(value) for value in result["latent_token_grid"]]
    return result


def _mask_record(mask: torch.Tensor, noun: str, tensor_key: str) -> dict[str, Any]:
    if not isinstance(mask, torch.Tensor) or mask.ndim != 3 or mask.dtype != torch.bool:
        raise SignatureError(f"Mbase for {noun!r} must be a bool tensor with shape [T,H,W]")
    value = mask.detach().cpu().contiguous()
    if not bool(value.any()):
        raise SignatureError(f"Mbase for {noun!r} is empty")
    projection = value.any(dim=0)
    ys, xs = torch.where(projection)
    return {
        "noun": noun,
        "tensor_key": tensor_key,
        "shape": list(value.shape),
        "dtype": "bool",
        "area_frac": float(value.float().mean().item()),
        "centroid_yx": [float(ys.float().mean().item()), float(xs.float().mean().item())],
        "shape_projection": "any_over_t",
    }


def write_online_signature(
    directory: Path,
    *,
    step5_latent: torch.Tensor,
    initial_latent: torch.Tensor,
    metadata: Mapping[str, Any],
    mbase_by_noun: Mapping[str, torch.Tensor],
    cache: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist a complete source/query Early Signature and return its sidecar."""

    step5 = _require_latent(step5_latent, "step5_latent")
    initial = _require_latent(initial_latent, "initial_latent")
    if tuple(step5.shape) != tuple(initial.shape):
        raise SignatureError("step5_latent and initial_latent shapes differ")
    if not mbase_by_noun:
        raise SignatureError("an online signature requires one Mbase per canonical noun")
    directory = directory.expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    feature = encode_latent_feature(step5, initial)
    feature_path = directory / "latent_feature.safetensors"
    save_file({"latent_feature": feature.to(torch.float16)}, str(feature_path), metadata={"codec": LATENT_FEATURE_CODEC})

    mask_tensors: dict[str, torch.Tensor] = {}
    objects: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, (raw_noun, mask) in enumerate(sorted(mbase_by_noun.items())):
        noun = _canonical_noun(raw_noun)
        if noun in seen:
            raise SignatureError(f"duplicate canonical noun {noun!r}")
        seen.add(noun)
        tensor_key = f"mbase_{index:03d}"
        record = _mask_record(mask, noun, tensor_key)
        if record["shape"] != list(metadata["latent_token_grid"]):
            raise SignatureError(f"Mbase for {noun!r} does not match latent_token_grid")
        mask_tensors[tensor_key] = mask.detach().cpu().contiguous()
        objects.append(record)
    mask_path = directory / "mbase.safetensors"
    save_file(mask_tensors, str(mask_path), metadata={"policy": str(metadata["mask_policy"])})

    validated_metadata = _validate_metadata(metadata, step5.shape)
    compatibility = {field: validated_metadata[field] for field in _COMPATIBILITY_FIELDS}
    payload = {
        "signature_version": ONLINE_SIGNATURE_VERSION,
        "feature_profile": LATENT_FEATURE_PROFILE_ID,
        "prompt": str(metadata.get("prompt", "")),
        "compatibility": compatibility,
        "feature": {
            "codec": LATENT_FEATURE_CODEC,
            "pool_shapes": [list(shape) for shape in LATENT_FEATURE_POOL_SHAPES],
            "normalization": "residual_to_initial_then_per_channel_rms_then_l2",
            "path": feature_path.name,
            "sha256": _sha256(feature_path),
            "shape": list(feature.shape),
            "dtype": "float16",
            "initial_latent_sha256": hashlib.sha256(initial.numpy().tobytes()).hexdigest(),
            "step5_latent_sha256": hashlib.sha256(step5.numpy().tobytes()).hexdigest(),
        },
        "mbase": {"path": mask_path.name, "sha256": _sha256(mask_path)},
        "objects": objects,
        "cache": dict(cache) if cache is not None else None,
    }
    signature_path = directory / "online_signature.json"
    signature_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"signature_path": str(signature_path)}


def _signature_path(signature: Mapping[str, Any]) -> Path:
    value = signature.get("signature_path")
    if not isinstance(value, str) or not value.strip():
        raise SignatureError("signature needs a readable signature_path")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise SignatureError(f"signature sidecar is missing: {path}")
    return path


def _load_signature(signature: Mapping[str, Any]) -> tuple[Path, dict[str, Any], torch.Tensor, dict[str, torch.Tensor]]:
    path = _signature_path(signature)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SignatureError(f"cannot read signature sidecar: {path}") from exc
    if payload.get("signature_version") != ONLINE_SIGNATURE_VERSION:
        raise SignatureError("signature version mismatch")
    if payload.get("feature_profile") != LATENT_FEATURE_PROFILE_ID:
        raise SignatureError("latent feature profile mismatch")
    compatibility = payload.get("compatibility")
    if not isinstance(compatibility, Mapping):
        raise SignatureError("signature compatibility metadata is missing")
    _validate_metadata(compatibility, compatibility.get("latent_shape", []))
    feature_info = payload.get("feature")
    if not isinstance(feature_info, Mapping) or feature_info.get("codec") != LATENT_FEATURE_CODEC:
        raise SignatureError("latent feature codec mismatch")
    feature_path = path.parent / str(feature_info.get("path", ""))
    if not feature_path.is_file() or _sha256(feature_path) != feature_info.get("sha256"):
        raise SignatureError("latent feature is missing or its hash changed")
    feature = load_file(str(feature_path), device="cpu").get("latent_feature")
    if feature is None or list(feature.shape) != feature_info.get("shape"):
        raise SignatureError("latent feature tensor metadata mismatch")
    if not feature.is_floating_point():
        raise SignatureError("latent feature tensor must be floating point")
    mask_info = payload.get("mbase")
    if not isinstance(mask_info, Mapping):
        raise SignatureError("Mbase metadata is missing")
    mask_path = path.parent / str(mask_info.get("path", ""))
    if not mask_path.is_file() or _sha256(mask_path) != mask_info.get("sha256"):
        raise SignatureError("Mbase is missing or its hash changed")
    masks = load_file(str(mask_path), device="cpu")
    objects = payload.get("objects")
    if not isinstance(objects, list) or not objects:
        raise SignatureError("signature objects are missing")
    for item in objects:
        if not isinstance(item, Mapping):
            raise SignatureError("signature object is invalid")
        key = item.get("tensor_key")
        mask = masks.get(key) if isinstance(key, str) else None
        if mask is None or mask.dtype != torch.bool or list(mask.shape) != item.get("shape"):
            raise SignatureError("Mbase tensor metadata mismatch")
        record = _mask_record(mask, _canonical_noun(str(item.get("noun", ""))), key)
        for field in ("dtype", "shape", "area_frac", "centroid_yx", "shape_projection"):
            if item.get(field) != record[field]:
                raise SignatureError("Mbase derived metadata mismatch")
    return path, payload, feature, masks


def _objects(payload: Mapping[str, Any], masks: Mapping[str, torch.Tensor]) -> dict[str, tuple[Mapping[str, Any], torch.Tensor]]:
    result: dict[str, tuple[Mapping[str, Any], torch.Tensor]] = {}
    for item in payload["objects"]:
        assert isinstance(item, Mapping)
        noun = _canonical_noun(str(item.get("noun", "")))
        if noun in result:
            raise SignatureError(f"duplicate noun {noun!r} in signature")
        result[noun] = (item, masks[str(item["tensor_key"])])
    return result


def _normalized_shape(mask: torch.Tensor, *, resolution: int = 128, support: float = 4.0) -> torch.Tensor:
    projection = mask.any(dim=0)
    ys, xs = torch.where(projection)
    if not len(xs):
        raise SignatureError("Mbase shape projection is empty")
    scale = math.sqrt(float(len(xs)))
    y = (ys.float() - ys.float().mean()) / scale
    x = (xs.float() - xs.float().mean()) / scale
    rows = ((y + support) * (resolution - 1) / (2.0 * support)).round().long()
    cols = ((x + support) * (resolution - 1) / (2.0 * support)).round().long()
    valid = (rows >= 0) & (rows < resolution) & (cols >= 0) & (cols < resolution)
    out = torch.zeros((resolution, resolution), dtype=torch.bool)
    out[rows[valid], cols[valid]] = True
    if not bool(out.any()):
        raise SignatureError("normalized Mbase shape is empty")
    return out


def normalized_shape_distance(query: torch.Tensor, candidate: torch.Tensor) -> float:
    left, right = _normalized_shape(query), _normalized_shape(candidate)
    intersection = int((left & right).sum().item())
    denominator = int(left.sum().item() + right.sum().item())
    return 1.0 - (2.0 * intersection / denominator)


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
        if not artifact.is_absolute():
            return False, "cache_artifact_path_not_absolute"
        if not artifact.is_file() or _sha256(artifact) != cache[hash_key]:
            return False, "step10_artifact_unreadable_or_changed"
    return True, None


def validate_online_signature(
    signature: Mapping[str, Any],
    *,
    require_cache: bool = False,
    check_cache_artifacts: bool = False,
) -> dict[str, Any]:
    """Validate a persisted sidecar before it enters online retrieval."""

    _path, payload, _feature, _masks = _load_signature(signature)
    cache_ready, reason = _cache_ready(payload, check_artifacts=check_cache_artifacts)
    if require_cache and not cache_ready:
        raise SignatureError(reason or "online signature is not cache-ready")
    return payload


def compare_online_signatures(
    query_signature: Mapping[str, Any],
    candidate_signature: Mapping[str, Any],
    *,
    query_changed_noun: str | None = None,
    candidate_changed_noun: str | None = None,
    max_latent_distance: float | None = None,
    max_log_area_ratio: float | None = None,
    max_shape_distance: float | None = None,
    check_cache_artifacts: bool = True,
) -> dict[str, Any]:
    """Apply the plan's feature and optional single-substitution gates."""

    result: dict[str, Any] = {
        "profile": LATENT_FEATURE_PROFILE_ID,
        "metadata_compatible": False,
        "distances": {"latent_feature": None, "log_area_ratio": None, "shape_one_minus_dice": None},
        "gates": {"latent_feature": None, "log_area_ratio": None, "shape_one_minus_dice": None},
        "signature_compatible": False,
        "cache_ready": False,
        "reusable": False,
        "reason": None,
    }
    try:
        _query_path, query, query_feature, query_masks = _load_signature(query_signature)
        _candidate_path, candidate, candidate_feature, candidate_masks = _load_signature(candidate_signature)
        if query["compatibility"] != candidate["compatibility"]:
            result["reason"] = "metadata_mismatch"
            return result
        result["metadata_compatible"] = True
        distance = cosine_distance(query_feature, candidate_feature)
        result["distances"]["latent_feature"] = distance
        result["gates"]["latent_feature"] = (
            max_latent_distance is not None and distance <= float(max_latent_distance)
        )
        substitution = query_changed_noun is not None or candidate_changed_noun is not None
        if substitution:
            if not query_changed_noun or not candidate_changed_noun:
                result["reason"] = "invalid_noun_substitution"
                return result
            query_object = _objects(query, query_masks).get(_canonical_noun(query_changed_noun))
            candidate_object = _objects(candidate, candidate_masks).get(_canonical_noun(candidate_changed_noun))
            if query_object is None or candidate_object is None:
                result["reason"] = "substitution_mbase_missing"
                return result
            query_meta, query_mask = query_object
            candidate_meta, candidate_mask = candidate_object
            area_distance = abs(math.log(float(query_meta["area_frac"]) / float(candidate_meta["area_frac"])))
            shape_distance = normalized_shape_distance(query_mask, candidate_mask)
            result["distances"].update({"log_area_ratio": area_distance, "shape_one_minus_dice": shape_distance})
            result["gates"].update({
                "log_area_ratio": max_log_area_ratio is not None and area_distance <= float(max_log_area_ratio),
                "shape_one_minus_dice": max_shape_distance is not None and shape_distance <= float(max_shape_distance),
            })
        applicable = ["latent_feature"] + (["log_area_ratio", "shape_one_minus_dice"] if substitution else [])
        result["signature_compatible"] = all(result["gates"][name] is True for name in applicable)
        cache_ready, cache_reason = _cache_ready(candidate, check_artifacts=check_cache_artifacts)
        result["cache_ready"] = cache_ready
        result["reusable"] = result["signature_compatible"] and cache_ready
        result["reason"] = "accepted" if result["reusable"] else (cache_reason or "gate_failed")
        return result
    except (OSError, RuntimeError, SignatureError, ValueError) as exc:
        result["reason"] = f"invalid_signature:{type(exc).__name__}"
        return result


__all__ = [
    "LATENT_FEATURE_CODEC",
    "LATENT_FEATURE_POOL_SHAPES",
    "LATENT_FEATURE_PROFILE_ID",
    "ONLINE_SIGNATURE_VERSION",
    "SignatureError",
    "compare_online_signatures",
    "cosine_distance",
    "encode_latent_feature",
    "normalized_shape_distance",
    "validate_online_signature",
    "write_online_signature",
]
