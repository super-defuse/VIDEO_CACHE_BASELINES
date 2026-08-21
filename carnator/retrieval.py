"""Carnator's repository-local two-stage cross-request retrieval.

The native Wan runner owns denoising and attention execution.  This module
only makes the authorization decision for reusing a source: a whole-prompt
vector shortlist followed by the ETS signature and artifact checks described
in ``docs/CARNATOR_CROSS_REQUEST_RETRIEVAL_DESIGN.md``.  It deliberately has
no dependency on the external early-signature project.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from core.retrieval import RetrievalStrategy, TextEmbedder, VectorRecord
from core.types import CacheMatch, Request


CARNATOR_IMPLEMENTATION_ID = "video-cache-baselines.carnator-two-stage.v1"
CARNATOR_SIGNATURE_PROFILE_ID = "ets-v1"
CARNATOR_RETRIEVER_ID = "carnator_two_stage"
_NOUN_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", re.IGNORECASE)
_NOUN_STOP_WORDS = frozenset({
    "a", "an", "and", "as", "at", "by", "for", "from", "in", "into", "is", "of",
    "on", "or", "the", "to", "with", "without", "near", "over", "under", "while",
    "this", "that", "these", "those", "very", "featuring", "feature", "style",
})


def _finite_number(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("non-finite numeric value")
    return result


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _normalise(vector: Iterable[float]) -> list[float]:
    values = [_finite_number(value) for value in vector]
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 1e-12:
        raise ValueError("zero-norm signature vector")
    return [value / norm for value in values]


def _cosine(left: Iterable[float], right: Iterable[float]) -> float:
    lhs = list(left)
    rhs = list(right)
    if len(lhs) != len(rhs) or not lhs:
        raise ValueError("signature vectors have incompatible dimensions")
    denominator = math.sqrt(sum(a * a for a in lhs)) * math.sqrt(sum(b * b for b in rhs))
    if denominator <= 1e-12:
        raise ValueError("zero-norm signature vector")
    return sum(a * b for a, b in zip(lhs, rhs, strict=True)) / denominator


def _flatten_mask(value: Any) -> list[float]:
    if isinstance(value, dict):
        if value.get("encoding") != "bitset-base64":
            raise ValueError("unsupported mask encoding")
        shape = value.get("shape")
        if not isinstance(shape, list) or not shape or any(int(item) <= 0 for item in shape):
            raise ValueError("encoded mask needs a non-empty positive shape")
        count = math.prod(int(item) for item in shape)
        if int(value.get("count", count)) != count:
            raise ValueError("encoded mask count does not match shape")
        try:
            payload = base64.b64decode(str(value["data"]), validate=True)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid encoded mask payload") from exc
        if len(payload) != math.ceil(count / 8):
            raise ValueError("encoded mask byte length does not match shape")
        flattened = [float((payload[index // 8] >> (7 - index % 8)) & 1) for index in range(count)]
        if count % 8 and payload[-1] & ((1 << (8 - count % 8)) - 1):
            raise ValueError("encoded mask has non-zero padding bits")
        return flattened
    if isinstance(value, (str, bytes, bytearray)) or value is None:
        raise ValueError("mask must be a non-empty numeric array")
    if isinstance(value, (int, float)):
        number = _finite_number(value)
        if not 0.0 <= number <= 1.0:
            raise ValueError("mask values must be in [0, 1]")
        return [number]
    if not isinstance(value, (list, tuple)):
        raise ValueError("mask must be a numeric array")
    flattened: list[float] = []
    for item in value:
        flattened.extend(_flatten_mask(item))
    if not flattened:
        raise ValueError("mask must not be empty")
    if any(value < 0.0 or value > 1.0 for value in flattened):
        raise ValueError("mask values must be in [0, 1]")
    return flattened


def _iou(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("mask dimensions differ")
    lhs = [value >= 0.5 for value in left]
    rhs = [value >= 0.5 for value in right]
    union = sum(a or b for a, b in zip(lhs, rhs, strict=True))
    intersection = sum(a and b for a, b in zip(lhs, rhs, strict=True))
    return 1.0 if union == 0 else intersection / union


def _shape_and_values(value: Any) -> tuple[list[int], list[float]]:
    """Convert a CPU tensor or nested numeric list into a rectangular shape/value pair."""

    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "tolist"):
        value = value.detach().cpu().tolist()

    def visit(item: Any) -> tuple[list[int], list[float]]:
        if isinstance(item, (int, float, bool)):
            return [], [_finite_number(item)]
        if not isinstance(item, (list, tuple)) or not item:
            raise ValueError("native ETS tensor must be a non-empty numeric array")
        child_shape, child_values = visit(item[0])
        values = list(child_values)
        for child in item[1:]:
            shape, child_values = visit(child)
            if shape != child_shape:
                raise ValueError("native ETS tensor must be rectangular")
            values.extend(child_values)
        return [len(item), *child_shape], values

    return visit(value)


def _encode_binary_mask(value: Any) -> dict[str, Any]:
    shape, values = _shape_and_values(value)
    if not shape:
        raise ValueError("native ETS mask must not be scalar")
    bits = [number >= 0.5 for number in values]
    packed = bytearray(math.ceil(len(bits) / 8))
    for index, enabled in enumerate(bits):
        if enabled:
            packed[index // 8] |= 1 << (7 - index % 8)
    return {
        "encoding": "bitset-base64",
        "shape": shape,
        "count": len(bits),
        "data": base64.b64encode(bytes(packed)).decode("ascii"),
    }


def _pooled_ets_vector(value: Any, *, pool_shape: tuple[int, int, int] = (4, 8, 8)) -> list[float]:
    """Pool a real [T,H,W] fusion score into a compact, stable ETS vector."""

    shape, values = _shape_and_values(value)
    if len(shape) != 3:
        raise ValueError(f"native ETS score must have [T,H,W] shape, got {shape}")
    if any(size <= 0 for size in pool_shape):
        raise ValueError("native ETS pool shape must be positive")
    effective_shape = tuple(min(int(pool), int(source)) for pool, source in zip(pool_shape, shape, strict=True))
    pooled_count = math.prod(effective_shape)
    sums = [0.0] * pooled_count
    counts = [0] * pooled_count
    source_t, source_h, source_w = shape
    output_h, output_w = effective_shape[1:]
    for index, number in enumerate(values):
        t = index // (source_h * source_w)
        rem = index % (source_h * source_w)
        h, w = divmod(rem, source_w)
        target_t = min(effective_shape[0] - 1, t * effective_shape[0] // source_t)
        target_h = min(output_h - 1, h * output_h // source_h)
        target_w = min(output_w - 1, w * output_w // source_w)
        target = (target_t * output_h + target_h) * output_w + target_w
        sums[target] += number
        counts[target] += 1
    if not all(counts):
        raise ValueError("native ETS pooling produced an empty bin")
    return _normalise(total / count for total, count in zip(sums, counts, strict=True))


def build_native_prompt_ets_signature(
    prompt: str,
    nouns: Iterable[str],
    mask_bundle: dict[str, Any],
    *,
    model_id: str = "Wan2.2-TI2V-5B",
    pool_shape: tuple[int, int, int] = (4, 8, 8),
    request_seed: int | None = None,
) -> dict[str, Any]:
    """Build a real step-5 prompt ETS from native Wan fusion-mask tensors.

    The cross-request noun set remains a cheap stage-1 filter.  The ETS itself
    is prompt-level because the native source path captures one calibrated
    fusion mask per prompt; no noun-only fixture is used here.
    """

    noun_set = canonical_nouns(nouns)
    if not noun_set:
        raise ValueError("native ETS needs at least one annotated noun")
    if not isinstance(mask_bundle, dict):
        raise ValueError("native ETS needs a mask bundle")
    if int(mask_bundle.get("source_completed_step", -1)) != 5:
        raise ValueError("native ETS must come from completed step 5")
    captured_prompt = mask_bundle.get("prompt")
    if captured_prompt is not None and str(captured_prompt) != prompt:
        raise ValueError("native ETS prompt does not match the captured mask bundle")
    required = ("Mbase", "Medit", "Msee", "score")
    if any(mask_bundle.get(key) is None for key in required):
        raise ValueError("native ETS bundle is missing canonical masks or fusion score")
    mask_shapes = [_shape_and_values(mask_bundle[key])[0] for key in required]
    if any(shape != mask_shapes[0] for shape in mask_shapes) or len(mask_shapes[0]) != 3:
        raise ValueError("native ETS masks and fusion score must share a [T,H,W] shape")
    effective_pool_shape = [min(int(pool), int(source)) for pool, source in zip(pool_shape, mask_shapes[0], strict=True)]
    return {
        "implementation_id": CARNATOR_IMPLEMENTATION_ID,
        "signature_profile_id": CARNATOR_SIGNATURE_PROFILE_ID,
        "completed_step": 5,
        "signature_scope": "prompt",
        "signature_source": "native_wan_step5_fusion_mask",
        "prompt_sha256": prompt_sha256(prompt),
        "trf_nouns": list(noun_set),
        "prompt_signature": {
            "mbase": _encode_binary_mask(mask_bundle["Mbase"]),
            "medit": _encode_binary_mask(mask_bundle["Medit"]),
            "msee": _encode_binary_mask(mask_bundle["Msee"]),
            "ets": _pooled_ets_vector(mask_bundle["score"], pool_shape=pool_shape),
        },
        "native_capture": {
            "model_id": model_id,
            "mask_variant": mask_bundle.get("mask_variant"),
            "token_shape": mask_shapes[0],
            "score_pool_shape": effective_pool_shape,
            "mask_positive": mask_bundle.get("mask_positive", ""),
            "request_seed": int(request_seed) if request_seed is not None else None,
        },
    }


def build_native_prompt_ets_signature_from_dir(
    mask_dir: str | Path,
    prompt: str,
    nouns: Iterable[str],
    *,
    model_id: str = "Wan2.2-TI2V-5B",
    request_seed: int | None = None,
) -> dict[str, Any]:
    """Load a native fusion-mask bundle and serialize its real ETS."""

    from carnator.masks import load_mask_bundle

    return build_native_prompt_ets_signature(
        prompt,
        nouns,
        load_mask_bundle(Path(mask_dir), None),
        model_id=model_id,
        request_seed=request_seed,
    )


def canonical_nouns(values: Iterable[str]) -> tuple[str, ...]:
    """Return the canonical, lower-case, de-duplicated noun set."""

    if isinstance(values, str):
        values = re.split(r"[;,|]", values)
    result = {
        " ".join(_NOUN_TOKEN_RE.findall(str(value).lower())).strip()
        for value in values
    }
    return tuple(sorted(value for value in result if value))


def heuristic_nouns(prompt: str) -> tuple[str, ...]:
    """Small local fallback used only when a prompt is outside the VBench CSV."""

    values = [token.lower() for token in _NOUN_TOKEN_RE.findall(prompt)]
    nouns = [token for token in values if token not in _NOUN_STOP_WORDS and len(token) > 1]
    return canonical_nouns(nouns) or ("prompt",)


class CarnatorNounAnnotations:
    """Load the repository-owned ``trf_nouns`` annotation CSV."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self._by_uuid: dict[str, tuple[str, ...]] = {}
        self._by_prompt: dict[str, tuple[str, ...]] = {}
        if self.path is not None:
            self._load()

    def _load(self) -> None:
        if not self.path or not self.path.is_file():
            raise FileNotFoundError(f"Carnator noun annotations are missing: {self.path}")
        with self.path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        required = {"request_index", "uuid", "prompt", "dimension", "trf_nouns"}
        if not rows or required - set(rows[0]):
            raise ValueError(f"noun annotation CSV must contain {sorted(required)}")
        for row in rows:
            nouns = canonical_nouns(str(row["trf_nouns"]).replace("|", ";").split(";"))
            if not nouns:
                raise ValueError(f"empty trf_nouns for {row.get('uuid')}")
            uuid = str(row["uuid"])
            prompt = str(row["prompt"])
            if uuid in self._by_uuid:
                raise ValueError(f"duplicate noun annotation: {uuid}")
            self._by_uuid[uuid] = nouns
            previous = self._by_prompt.get(prompt)
            if previous is not None and previous != nouns:
                raise ValueError(f"conflicting noun annotation for prompt: {prompt}")
            self._by_prompt[prompt] = nouns

    def for_request(self, request: Request) -> tuple[str, ...]:
        return self._by_uuid.get(request.uuid, self._by_prompt.get(request.prompt, heuristic_nouns(request.prompt)))

    def for_entry(self, entry: Any) -> tuple[str, ...]:
        return self._by_uuid.get(entry.uuid, self._by_prompt.get(entry.prompt, heuristic_nouns(entry.prompt)))


@dataclass(frozen=True)
class CarnatorThresholdConfig:
    threshold_config_id: str
    implementation_id: str
    signature_profile_id: str
    thresholds: dict[str, float]
    frozen: bool
    authenticated: bool

    REQUIRED_METRICS = ("mbase_iou", "medit_iou", "msee_iou", "ets_cosine")

    @classmethod
    def load(cls, path: str | Path) -> "CarnatorThresholdConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Carnator threshold config must be an object")
        raw = payload.get("thresholds", payload.get("metric_thresholds", payload.get("metrics", {})))
        if not isinstance(raw, dict):
            raise ValueError("Carnator threshold config thresholds must be an object")
        thresholds: dict[str, float] = {}
        for key, value in raw.items():
            if isinstance(value, dict):
                value = value.get("min", value.get("threshold"))
            thresholds[str(key)] = _finite_number(value)
        return cls(
            str(payload.get("threshold_config_id", "")),
            str(payload.get("implementation_id", payload.get("local_implementation_id", ""))),
            str(payload.get("signature_profile_id", payload.get("profile_id", ""))),
            thresholds,
            bool(payload.get("frozen", payload.get("status") == "frozen")),
            bool(payload.get("authenticated", payload.get("authentication") in {"authenticated", "verified"})),
        )

    @property
    def valid(self) -> bool:
        try:
            values_valid = all(0.0 <= _finite_number(value) <= 1.0 for value in self.thresholds.values())
        except (TypeError, ValueError):
            values_valid = False
        return bool(
            self.threshold_config_id
            and self.implementation_id == CARNATOR_IMPLEMENTATION_ID
            and self.signature_profile_id == CARNATOR_SIGNATURE_PROFILE_ID
            and self.frozen
            and self.authenticated
            and all(metric in self.thresholds for metric in self.REQUIRED_METRICS)
            and values_valid
        )


class CarnatorSignatureBuilder:
    """Deterministic local ETS fixture builder.

    Native Wan adapters can replace this builder with their step-5 probe.  The
    default is intentionally noun-only, so dry-run and retrieval-only tests
    exercise the protocol without loading a model.
    """

    def __init__(self, *, implementation_id: str = CARNATOR_IMPLEMENTATION_ID,
                 signature_profile_id: str = CARNATOR_SIGNATURE_PROFILE_ID) -> None:
        self.implementation_id = implementation_id
        self.signature_profile_id = signature_profile_id

    @staticmethod
    def _mask(noun: str, region: str) -> list[float]:
        digest = hashlib.sha256(f"{region}:{noun}".encode("utf-8")).digest()
        # A compact binary mask keeps fixtures portable while retaining exact
        # shape checks and deterministic ETS metrics.
        return [1.0 if digest[index] & 1 else 0.0 for index in range(16)]

    @staticmethod
    def _vector(noun: str) -> list[float]:
        digest = hashlib.sha256(f"ets:{noun}".encode("utf-8")).digest()
        return _normalise(((-1.0 if byte & 1 else 1.0) for byte in digest[:16]))

    def build(self, prompt: str, nouns: Iterable[str]) -> dict[str, Any]:
        del prompt
        noun_set = canonical_nouns(nouns)
        if not noun_set:
            raise ValueError("ETS signature needs at least one noun")
        return {
            "implementation_id": self.implementation_id,
            "signature_profile_id": self.signature_profile_id,
            "completed_step": 5,
            "signature_scope": "noun",
            "signature_source": "fixture",
            "trf_nouns": list(noun_set),
            "nouns": {
                noun: {
                    "mbase": self._mask(noun, "mbase"),
                    "medit": self._mask(noun, "medit"),
                    "msee": self._mask(noun, "msee"),
                    "ets": self._vector(noun),
                }
                for noun in noun_set
            },
        }


class CarnatorSignatureComparator:
    """Validate and compare source/target step-5 ETS signatures."""

    def __init__(self, threshold_config: CarnatorThresholdConfig) -> None:
        self.threshold_config = threshold_config or CarnatorThresholdConfig("", "", "", {}, False, False)

    @staticmethod
    def noun_mapping(source: Iterable[str], target: Iterable[str]) -> dict[str, str]:
        source_set, target_set = set(canonical_nouns(source)), set(canonical_nouns(target))
        if source_set == target_set:
            return {noun: noun for noun in sorted(source_set)}
        source_only, target_only = sorted(source_set - target_set), sorted(target_set - source_set)
        if len(source_only) == len(target_only) == 1 and len(source_set) == len(target_set):
            return {noun: noun for noun in sorted(source_set & target_set)} | {source_only[0]: target_only[0]}
        raise ValueError("noun sets differ by more than one substitution")

    @staticmethod
    def _validate(signature: dict[str, Any]) -> tuple[tuple[str, ...], dict[str, dict[str, list[float]]], str]:
        if not isinstance(signature, dict):
            raise ValueError("ETS signature must be an object")
        implementation_id = signature.get("implementation_id", signature.get("local_implementation_id"))
        profile_id = signature.get("signature_profile_id", signature.get("profile_id"))
        if implementation_id != CARNATOR_IMPLEMENTATION_ID:
            raise ValueError("ETS implementation id mismatch")
        if profile_id != CARNATOR_SIGNATURE_PROFILE_ID:
            raise ValueError("ETS signature profile mismatch")
        if int(signature.get("completed_step", signature.get("step", -1))) != 5:
            raise ValueError("ETS signature must be from completed step 5")
        nouns = canonical_nouns(signature.get("trf_nouns", []))
        scope = str(signature.get("signature_scope", "noun"))
        if scope not in {"noun", "prompt"}:
            raise ValueError("unsupported ETS signature scope")
        raw_nouns = signature.get("nouns", signature.get("noun_signatures"))
        direct_metrics = signature.get("metrics")
        if raw_nouns is None and isinstance(direct_metrics, dict):
            for metric in CarnatorThresholdConfig.REQUIRED_METRICS:
                value = direct_metrics.get(metric)
                if value is None:
                    value = direct_metrics.get({
                        "mbase_iou": "base_iou", "medit_iou": "edit_iou",
                        "msee_iou": "see_iou", "ets_cosine": "signature_cosine",
                    }[metric])
                value = _finite_number(value)
                if not 0.0 <= value <= 1.0:
                    raise ValueError("ETS metric must be in [0, 1]")
            return nouns, {}, scope

        if scope == "prompt":
            raw_prompt = signature.get("prompt_signature")
            prompt_sha256 = signature.get("prompt_sha256")
            if (not nouns or not isinstance(raw_prompt, dict)
                    or not isinstance(prompt_sha256, str)
                    or not re.fullmatch(r"[0-9a-f]{64}", prompt_sha256)):
                raise ValueError("prompt ETS needs prompt_signature, noun metadata and prompt hash")
            raw_nouns = {"__prompt__": raw_prompt}
            expected_nouns = ("__prompt__",)
        else:
            expected_nouns = nouns
        if not nouns or not isinstance(raw_nouns, dict) or set(raw_nouns) != set(expected_nouns):
            raise ValueError("ETS noun metadata does not match signature entries")
        entries: dict[str, dict[str, list[float]]] = {}
        for noun in expected_nouns:
            raw = raw_nouns[noun]
            if not isinstance(raw, dict):
                raise ValueError("ETS noun entry must be an object")
            def _field(name: str):
                return raw.get(name, raw.get(name.capitalize(), raw.get(name.upper())))

            entries[noun] = {
                "mbase": _flatten_mask(_field("mbase")),
                "medit": _flatten_mask(_field("medit")),
                "msee": _flatten_mask(_field("msee")),
                "ets": [_finite_number(value) for value in (_field("ets") or _field("vector"))],
            }
            if not entries[noun]["ets"]:
                raise ValueError("ETS vector must not be empty")
            if not (len(entries[noun]["mbase"]) == len(entries[noun]["medit"]) == len(entries[noun]["msee"])):
                raise ValueError("ETS masks have inconsistent dimensions")
        return nouns, entries, scope

    def compare(self, target: dict[str, Any], source: dict[str, Any]) -> dict[str, float]:
        if not self.threshold_config.valid:
            raise ValueError("threshold config is not frozen and authenticated")
        target_nouns, target_entries, target_scope = self._validate(target)
        source_nouns, source_entries, source_scope = self._validate(source)
        if target_scope != source_scope:
            raise ValueError("ETS signature scopes differ")
        if not target_entries or not source_entries:
            aliases = {
                "mbase_iou": "base_iou", "medit_iou": "edit_iou",
                "msee_iou": "see_iou", "ets_cosine": "signature_cosine",
            }
            metrics: dict[str, float] = {}
            for metric in CarnatorThresholdConfig.REQUIRED_METRICS:
                values = []
                for signature in (target, source):
                    direct = signature.get("metrics", {})
                    if isinstance(direct, dict) and (metric in direct or aliases[metric] in direct):
                        values.append(_finite_number(direct.get(metric, direct.get(aliases[metric]))))
                if not values:
                    raise ValueError("missing direct ETS metric")
                metrics[metric] = min(values)
            return metrics
        if target_scope == "prompt":
            paired = [(source_entries["__prompt__"], target_entries["__prompt__"])]
        else:
            mapping = self.noun_mapping(source_nouns, target_nouns)
            paired = [
                (source_entries[source_noun], target_entries[target_noun])
                for source_noun, target_noun in sorted(mapping.items())
            ]
        metrics = {
            "mbase_iou": sum(_iou(lhs["mbase"], rhs["mbase"]) for lhs, rhs in paired) / len(paired),
            "medit_iou": sum(_iou(lhs["medit"], rhs["medit"]) for lhs, rhs in paired) / len(paired),
            "msee_iou": sum(_iou(lhs["msee"], rhs["msee"]) for lhs, rhs in paired) / len(paired),
            "ets_cosine": sum(_cosine(lhs["ets"], rhs["ets"]) for lhs, rhs in paired) / len(paired),
        }
        if any(not math.isfinite(value) for value in metrics.values()):
            raise ValueError("ETS metric is non-finite")
        return metrics

    def accepts(self, metrics: dict[str, float]) -> bool:
        try:
            return all(
                metric in metrics and _finite_number(metrics[metric]) >= _finite_number(self.threshold_config.thresholds[metric])
                for metric in CarnatorThresholdConfig.REQUIRED_METRICS
            )
        except (TypeError, ValueError):
            return False


class CarnatorArtifactValidator:
    """Fail-closed validation for the exact step-10 and sparse artifacts."""

    def __init__(self, *, retrieval_only: bool = False) -> None:
        self.retrieval_only = retrieval_only
        self.threshold_config_id: str | None = None

    @staticmethod
    def _inside(root: Path, value: str) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as exc:
            raise ValueError("artifact path escapes the Carnator namespace") from exc
        return path

    @staticmethod
    def _hash(path: Path) -> str:
        digest = hashlib.sha256()
        if path.is_dir():
            for child in sorted(item for item in path.rglob("*") if item.is_file()):
                digest.update(str(child.relative_to(path)).encode("utf-8"))
                with child.open("rb") as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(block)
        else:
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
        return digest.hexdigest()

    def validate(self, metadata: dict[str, Any] | None, *, expected_uuid: str,
                 namespace_root: Path | None) -> tuple[bool, str]:
        if not isinstance(metadata, dict):
            return False, "missing_artifact_metadata"
        if metadata.get("cache_ready") is False:
            return False, "cache_not_ready"
        manifest = metadata.get("manifest")
        manifest_path_value = metadata.get("manifest_path")
        # Retrieval-only fixtures carry an inline readiness manifest.  Do not
        # resolve or read artifact paths in this mode.
        if manifest is None and manifest_path_value and not self.retrieval_only:
            if namespace_root is None:
                return False, "missing_namespace"
            try:
                manifest_path = self._inside(namespace_root, str(manifest_path_value))
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                return False, "invalid_manifest"
        if self.retrieval_only and metadata.get("cache_ready") is True:
            try:
                if (self.threshold_config_id
                        and isinstance(manifest, dict)
                        and manifest.get("threshold_config_id") != self.threshold_config_id):
                    return False, "threshold_config_mismatch"
                return self._validate_manifest_metadata(manifest, expected_uuid)
            except (TypeError, ValueError, KeyError):
                return False, "invalid_artifact_metadata"
        if not isinstance(manifest, dict):
            return False, "missing_manifest"
        if self.threshold_config_id and manifest.get("threshold_config_id") != self.threshold_config_id:
            return False, "threshold_config_mismatch"
        try:
            valid_manifest, manifest_reason = self._validate_manifest_metadata(manifest, expected_uuid)
            if not valid_manifest:
                return False, manifest_reason
            if namespace_root is None:
                return False, "missing_namespace"
            if str(manifest.get("namespace")) != namespace_root.name:
                return False, "namespace_identity_mismatch"
            artifact_items = {
                "step_10_latent": manifest.get("step_10_latent", manifest.get("latent_step_10")),
                "sparse_attention": manifest.get("sparse_attention", manifest.get("sparse_attention_pattern")),
            }
            for key, item in artifact_items.items():
                if not isinstance(item, dict) or not item.get("path") or not item.get("sha256"):
                    return False, f"missing_{key}"
                if key == "step_10_latent" and int(item.get("step", 10)) != 10:
                    return False, "invalid_step_10_latent_step"
                path = self._inside(namespace_root, str(item["path"]))
                if not path.is_file() and not path.is_dir():
                    return False, f"missing_{key}"
                if path.is_file() and path.stat().st_size == 0:
                    return False, f"missing_{key}"
                if path.is_dir() and not any(item.is_file() for item in path.rglob("*")):
                    return False, f"missing_{key}"
                if self._hash(path) != str(item["sha256"]):
                    return False, f"invalid_{key}_hash"
            signature_path_value = manifest.get("signature_path")
            signature_hash = manifest.get("signature_sha256")
            if not signature_path_value or not signature_hash:
                return False, "missing_signature_artifact"
            signature_path = self._inside(namespace_root, str(signature_path_value))
            if not signature_path.is_file() or signature_path.stat().st_size == 0:
                return False, "missing_signature_artifact"
            if self._hash(signature_path) != str(signature_hash):
                return False, "invalid_signature_artifact_hash"
        except (OSError, ValueError, TypeError, KeyError):
            return False, "invalid_artifact_metadata"
        return True, "cache_ready"

    @staticmethod
    def _validate_manifest_metadata(manifest: Any, expected_uuid: str) -> tuple[bool, str]:
        if not isinstance(manifest, dict):
            return False, "missing_manifest"
        if manifest.get("implementation_id", manifest.get("local_implementation_id")) != CARNATOR_IMPLEMENTATION_ID:
            return False, "implementation_id_mismatch"
        if manifest.get("signature_profile_id", manifest.get("profile_id")) != CARNATOR_SIGNATURE_PROFILE_ID:
            return False, "signature_profile_id_mismatch"
        if str(manifest.get("uuid")) != expected_uuid:
            return False, "manifest_uuid_mismatch"
        if not manifest.get("namespace"):
            return False, "missing_namespace_identity"
        if int(manifest.get("completed_step", -1)) != 5:
            return False, "completed_step_mismatch"
        if int(manifest.get("reuse_anchor_step", -1)) != 10:
            return False, "reuse_anchor_step_mismatch"
        return True, "manifest_valid"


class CarnatorRetriever(RetrievalStrategy):
    """Two-stage Carnator retrieval with deterministic candidate ordering."""

    kind = CARNATOR_RETRIEVER_ID
    eviction_policy = "lru"

    def __init__(self, embedder: TextEmbedder, noun_annotations: CarnatorNounAnnotations,
                 threshold_config: CarnatorThresholdConfig | None, *, candidate_top_k: int = 20,
                 retrieval_only: bool = False,
                 allow_fixture_signatures: bool = False,
                 signature_builder: CarnatorSignatureBuilder | None = None) -> None:
        if candidate_top_k <= 0:
            raise ValueError("Carnator candidate_top_k must be positive")
        super().__init__(embedder, -1.0, vector_backend="memory")
        self.noun_annotations = noun_annotations
        self.threshold_config = threshold_config or CarnatorThresholdConfig("", "", "", {}, False, False)
        self.candidate_top_k = int(candidate_top_k)
        self.retrieval_only = retrieval_only
        self.allow_fixture_signatures = bool(allow_fixture_signatures or retrieval_only)
        self.signature_builder = signature_builder or CarnatorSignatureBuilder()
        self.comparator = CarnatorSignatureComparator(self.threshold_config)
        self.validator = CarnatorArtifactValidator(retrieval_only=retrieval_only)
        self.validator.threshold_config_id = self.threshold_config.threshold_config_id
        self._namespace_root: Path | None = None

    @property
    def signature(self) -> str:
        return (
            f"{self.kind}|{self.embedder.signature}|top_k={self.candidate_top_k}"
            f"|threshold_config_id={self.threshold_config.threshold_config_id}"
            f"|implementation_id={CARNATOR_IMPLEMENTATION_ID}|profile={CARNATOR_SIGNATURE_PROFILE_ID}"
        )

    def bind_vector_store(self, root: Path) -> None:
        super().bind_vector_store(root)
        self._namespace_root = root.parent

    def records_for(self, entry: Any) -> list[VectorRecord]:
        nouns = self.noun_annotations.for_entry(entry)
        metadata = dict(getattr(entry, "metadata", None) or {})
        metadata.setdefault("source_request_index", int(entry.request_index))
        metadata.setdefault("trf_nouns", list(nouns))
        if "signature" not in metadata and self.allow_fixture_signatures:
            metadata["signature"] = self.signature_builder.build(entry.prompt, nouns)
        if "signature" not in metadata:
            # A production source must provide the native step-5 signature.
            # Do not silently turn a real request into the lexical fixture.
            metadata["cache_ready"] = False
            metadata.setdefault("signature_missing_reason", "native_step5_signature_required")
        # Dataset replay uses inline, retrieval-only manifests and does not
        # materialize a latent path.  Native/cache Runner entries still carry
        # the on-disk manifest path as before.
        latent_path = getattr(entry, "latent_path", None)
        if latent_path is not None:
            metadata.setdefault("manifest_path", str(Path(latent_path).with_suffix(".manifest.json")))
        return [VectorRecord(entry.uuid, entry.prompt, self.embedder.encode([entry.prompt])[0], metadata,
                             int(entry.request_index))]

    def record_is_valid(self, record: VectorRecord) -> bool:
        """Allow NamespaceCache to prune vectors whose artifacts disappeared."""
        metadata = record.metadata if isinstance(record.metadata, dict) else None
        ready, _ = self.validator.validate(metadata, expected_uuid=record.uuid,
                                           namespace_root=self._namespace_root)
        return ready

    @staticmethod
    def _prefilter(source: Iterable[str], target: Iterable[str]) -> tuple[bool, str]:
        try:
            source_set, target_set = set(canonical_nouns(source)), set(canonical_nouns(target))
        except (TypeError, ValueError):
            return False, "invalid_noun_metadata"
        if source_set == target_set:
            return True, "same_noun_set"
        if len(source_set - target_set) == len(target_set - source_set) == 1:
            return True, "single_noun_substitution"
        return False, "noun_set_difference_gt_one"

    def _top_k(self, request: Request, records: list[VectorRecord]) -> list[tuple[float, VectorRecord]]:
        # Carnator retrieval is defined over the complete target prompt;
        # ``Request.cache_prompt`` is an explicit source override for legacy
        # methods and must not bypass this stage.
        query = self.embedder.encode([request.prompt])[0]
        candidates: list[tuple[float, VectorRecord]] = []
        for record in records:
            try:
                dimensions_match = len(query) == len(record.vector)
            except TypeError:
                continue
            if not dimensions_match:
                continue
            try:
                score = _finite_number(sum(a * b for a, b in zip(query, record.vector, strict=True)))
            except (TypeError, ValueError):
                continue
            candidates.append((score, record))
        def _sort_key(item: tuple[float, VectorRecord]):
            raw_index = item[1].source_request_index
            if raw_index is None and isinstance(item[1].metadata, dict):
                raw_index = item[1].metadata.get("source_request_index")
            try:
                source_index = int(raw_index)
            except (TypeError, ValueError):
                source_index = 2**63 - 1
            return -item[0], source_index, item[1].uuid

        candidates.sort(key=_sort_key)
        return candidates[: self.candidate_top_k]

    def _source_signature(self, record: VectorRecord) -> dict[str, Any]:
        metadata = record.metadata if isinstance(record.metadata, dict) else {}
        signature = metadata.get("signature")
        if signature is None and metadata.get("signature_path"):
            path = CarnatorArtifactValidator._inside(self._namespace_root or Path.cwd(), str(metadata["signature_path"]))
            signature = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(signature, dict):
            raise ValueError("missing ETS signature")
        return signature

    def search(
        self,
        request: Request,
        records: Iterable[VectorRecord],
        *,
        target_signature: dict[str, Any] | None = None,
    ) -> CacheMatch:
        metadata: dict[str, Any] = {
            "retriever": self.kind,
            "embedding": self.embedder.signature,
            "candidate_top_k": self.candidate_top_k,
            "threshold_config_id": self.threshold_config.threshold_config_id,
            "cache_ready": False,
            "step_2_probe_run": False,
            "probe_required": False,
            "top_k": [],
            "candidates": [],
            "noun_prefilter": [],
            "stage_2": [],
        }
        target_nouns = self.noun_annotations.for_request(request)
        candidates = self._top_k(request, list(records))
        metadata["top_k"] = [
            {"uuid": record.uuid, "rank": rank, "score": score}
            for rank, (score, record) in enumerate(candidates, start=1)
        ]
        metadata["candidates"] = metadata["top_k"]
        passed: list[tuple[float, VectorRecord, str]] = []
        for score, record in candidates:
            source_nouns = (record.metadata or {}).get("trf_nouns", []) if isinstance(record.metadata, dict) else []
            ok, reason = self._prefilter(source_nouns, target_nouns)
            metadata["noun_prefilter"].append({
                "uuid": record.uuid,
                "source_nouns": list(canonical_nouns(source_nouns)) if isinstance(source_nouns, (list, tuple, str)) else [],
                "target_nouns": list(target_nouns),
                "passed": ok,
                "reason": reason,
            })
            if ok:
                passed.append((score, record, reason))
        if not self.threshold_config.valid:
            metadata["reason"] = "threshold_config_unauthenticated"
            return CacheMatch(metadata=metadata, kind="no_reuse")
        if not passed:
            metadata["reason"] = "no_noun_compatible_candidate"
            return CacheMatch(metadata=metadata, kind="no_reuse")

        if target_signature is None and not self.allow_fixture_signatures:
            # Stage 1 is intentionally non-authoritative. The native Wan
            # request must run to completed step 5 and call this method again
            # with its real fusion-mask ETS before any latent is loaded.
            metadata["reason"] = "target_signature_required"
            metadata["probe_required"] = True
            return CacheMatch(metadata=metadata, kind="probe_required")

        # The probe is intentionally performed once, after noun prefiltering.
        metadata["step_2_probe_run"] = True
        if target_signature is None:
            target_signature = self.signature_builder.build(request.prompt, target_nouns)
        elif target_signature.get("signature_scope") == "prompt":
            if target_signature.get("prompt_sha256") != prompt_sha256(request.prompt):
                metadata["reason"] = "target_signature_prompt_mismatch"
                return CacheMatch(metadata=metadata, kind="no_reuse")
        for score, record, _ in passed:
            candidate_result: dict[str, Any] = {"uuid": record.uuid, "score": score}
            try:
                source_signature = self._source_signature(record)
                if (source_signature.get("signature_scope") == "prompt"
                        and source_signature.get("prompt_sha256") != prompt_sha256(record.term)):
                    raise ValueError("source ETS prompt hash does not match source prompt")
                source_metadata_nouns = canonical_nouns(
                    (record.metadata or {}).get("trf_nouns", []) if isinstance(record.metadata, dict) else []
                )
                signature_nouns = canonical_nouns(source_signature.get("trf_nouns", []))
                if source_metadata_nouns != signature_nouns:
                    raise ValueError("source noun metadata does not match ETS signature")
                metrics = self.comparator.compare(target_signature, source_signature)
                candidate_result["metrics"] = metrics
                if not self.comparator.accepts(metrics):
                    candidate_result["reason"] = "signature_incompatible"
                    metadata["stage_2"].append(candidate_result)
                    continue
            except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                candidate_result["reason"] = f"invalid_signature:{type(exc).__name__}"
                metadata["stage_2"].append(candidate_result)
                continue
            ready, reason = self.validator.validate(record.metadata, expected_uuid=record.uuid,
                                                   namespace_root=self._namespace_root)
            candidate_result["cache_ready"] = ready
            if not ready:
                candidate_result["reason"] = reason
                metadata["stage_2"].append(candidate_result)
                continue
            candidate_result["reason"] = "accepted"
            metadata["stage_2"].append(candidate_result)
            metadata["source_uuid"] = record.uuid
            metadata["cache_ready"] = True
            metadata["reuse_mode"] = "joint_latent_sparse_attention"
            metadata["selected_metrics"] = metrics
            return CacheMatch(True, record.uuid, None, self.kind, score, metadata)
        metadata["reason"] = "no_compatible_cache"
        metadata["cache_ready"] = False
        return CacheMatch(metadata=metadata, kind="no_reuse")

    def prepare_cache_entry(self, request: Request, latent_path: Path, latent_step: int) -> dict[str, Any]:
        """Create portable synthetic artifacts for the protocol Runner.

        Native Wan source generation writes its own artifacts.  The generic
        Runner uses this hook for dry-run fixtures so cache readiness follows
        exactly the same manifest contract.
        """

        namespace_root = latent_path.parent.parent
        signature = self.signature_builder.build(request.prompt, self.noun_annotations.for_request(request))
        signature_path = latent_path.with_suffix(".carnator_ets.json")
        signature_path.write_text(json.dumps(signature, sort_keys=True) + "\n", encoding="utf-8")
        sparse_path = latent_path.with_suffix(".sparse_attention")
        sparse_path.write_bytes(hashlib.sha256(request.uuid.encode("utf-8")).digest())
        latent_rel = str(latent_path.resolve().relative_to(namespace_root.resolve()))
        sparse_rel = str(sparse_path.resolve().relative_to(namespace_root.resolve()))
        manifest = {
            "implementation_id": CARNATOR_IMPLEMENTATION_ID,
            "signature_profile_id": CARNATOR_SIGNATURE_PROFILE_ID,
            "threshold_config_id": self.threshold_config.threshold_config_id,
            "namespace": namespace_root.name,
            "uuid": request.uuid,
            "completed_step": 5,
            "reuse_anchor_step": 10,
            "step_10_latent": {"step": int(latent_step), "path": latent_rel, "sha256": self._hash(latent_path)},
            "sparse_attention": {"path": sparse_rel, "sha256": self._hash(sparse_path)},
            "signature_path": str(signature_path.resolve().relative_to(namespace_root.resolve())),
            "signature_sha256": self._hash(signature_path),
        }
        manifest_path = latent_path.with_suffix(".manifest.json")
        manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
        return {"trf_nouns": list(signature["trf_nouns"]), "signature": signature,
                "manifest": manifest, "manifest_path": str(manifest_path), "cache_ready": self.threshold_config.valid,
                "latent_step": int(latent_step)}

    @staticmethod
    def _hash(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()


def make_carnator_retriever(config: dict[str, Any], *, dry_run: bool,
                            retrieval_only: bool = False) -> CarnatorRetriever:
    """Build Carnator retrieval from local configuration only."""

    from core.retrieval import _make_embedder

    annotation_path = config.get("noun_annotations")
    annotations = CarnatorNounAnnotations(annotation_path)
    threshold_path = config.get("threshold_config")
    if threshold_path:
        try:
            thresholds = CarnatorThresholdConfig.load(threshold_path)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            thresholds = CarnatorThresholdConfig("", "", "", {}, False, False)
    else:
        thresholds = CarnatorThresholdConfig("", "", "", {}, False, False)
    return CarnatorRetriever(
        _make_embedder(config, dry_run=dry_run), annotations, thresholds,
        candidate_top_k=int(config.get("candidate_top_k", 20)),
        retrieval_only=retrieval_only,
        allow_fixture_signatures=bool(dry_run),
    )


# Descriptive aliases used by retrieval-only callers.
CarnatorETSComparator = CarnatorSignatureComparator
CarnatorETSBuilder = CarnatorSignatureBuilder


__all__ = [
    "CARNATOR_IMPLEMENTATION_ID", "CARNATOR_SIGNATURE_PROFILE_ID", "CARNATOR_RETRIEVER_ID",
    "CarnatorArtifactValidator", "CarnatorNounAnnotations", "CarnatorRetriever",
    "CarnatorSignatureBuilder", "CarnatorSignatureComparator", "CarnatorThresholdConfig",
    "CarnatorETSBuilder", "CarnatorETSComparator",
    "build_native_prompt_ets_signature", "build_native_prompt_ets_signature_from_dir",
    "canonical_nouns", "heuristic_nouns", "make_carnator_retriever", "prompt_sha256",
]
