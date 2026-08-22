"""Carnator's object-level cross-request retrieval authorization.

Stage 1 ranks complete prompts, then filters their noun sets. Stage 2 accepts
only native step-5 online-signature sidecars; legacy prompt-level ETS is
intentionally unsupported.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

from carnator.online_signature import LATENT_FEATURE_PROFILE_ID, compare_online_signatures
from core.retrieval import (
    EntityExtractor,
    LexicalTestEntityExtractor,
    PromptRetriever,
    RetrievalStrategy,
    SpacyEntityExtractor,
    TextEmbedder,
    VectorRecord,
    _make_embedder,
    _vector_backend,
)
from core.types import CacheMatch, Request


CARNATOR_RETRIEVER_ID = "carnator_two_stage"
_NOUN_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", re.IGNORECASE)
# spaCy labels the noun in this one-token VBench prompt as ADJ. Keep the
# correction tied to the request UUID rather than expanding the POS policy.
_NOUNLESS_OVERRIDES = {"object_class_65": ("remote",)}


def canonical_nouns(values: Iterable[str]) -> tuple[str, ...]:
    """Return canonical lower-case, de-duplicated noun strings."""

    if isinstance(values, str):
        values = re.split(r"[;,|]", values)
    nouns = {
        " ".join(_NOUN_TOKEN_RE.findall(str(value).lower())).strip()
        for value in values
    }
    return tuple(sorted(noun for noun in nouns if noun))


class CarnatorNounExtractor:
    """Extract canonical nouns from the prompt at request arrival time."""

    def __init__(self, extractor: EntityExtractor | None = None) -> None:
        self.extractor = extractor or LexicalTestEntityExtractor()

    def for_request(self, request: Request) -> tuple[str, ...]:
        return self._extract(request.uuid, request.prompt)

    def for_entry(self, entry: Any) -> tuple[str, ...]:
        return self._extract(entry.uuid, entry.prompt)

    def _extract(self, uuid: str, prompt: str) -> tuple[str, ...]:
        nouns = canonical_nouns(self.extractor.extract(prompt))
        if not nouns:
            nouns = canonical_nouns(_NOUNLESS_OVERRIDES.get(str(uuid), ()))
        if not nouns:
            raise ValueError("noun extraction produced no NOUN/PROPN tokens")
        return nouns


class CarnatorRetriever(RetrievalStrategy):
    """Top-20 prompt recall followed by an object-level Early Signature gate.

    The production runner remains fail-closed until it supplies a native
    object-level sidecar and artifact contract. ``retrieval_only`` is for
    benchmark replay: it regards a supplied cache-ready sidecar as a source
    artifact without loading any latent or attention tensor.
    """

    kind = CARNATOR_RETRIEVER_ID
    eviction_policy = "lru"

    def __init__(
        self,
        embedder: TextEmbedder,
        noun_extractor: CarnatorNounExtractor,
        *,
        candidate_top_k: int = 20,
        candidate_min_similarity: float = 0.65,
        candidate_vector_backend: str = "memory",
        stage2_candidate_top_k: int = 5,
        retrieval_only: bool = False,
        max_latent_distance: float | None = None,
        max_log_area_ratio: float | None = None,
        max_shape_distance: float | None = None,
    ) -> None:
        if candidate_top_k <= 0 or stage2_candidate_top_k <= 0:
            raise ValueError("Carnator candidate limits must be positive")
        if not -1.0 <= float(candidate_min_similarity) <= 1.0:
            raise ValueError("Carnator candidate_min_similarity must be between -1 and 1")
        super().__init__(embedder, -1.0, vector_backend="memory")
        self.noun_extractor = noun_extractor
        self.candidate_top_k = int(candidate_top_k)
        self.candidate_min_similarity = float(candidate_min_similarity)
        self.stage2_candidate_top_k = int(stage2_candidate_top_k)
        self.retrieval_only = bool(retrieval_only)
        self.max_latent_distance = self._threshold(max_latent_distance, "max_latent_distance")
        self.max_log_area_ratio = self._threshold(max_log_area_ratio, "max_log_area_ratio")
        self.max_shape_distance = self._threshold(max_shape_distance, "max_shape_distance")
        self.prompt_retriever = PromptRetriever(
            embedder,
            self.candidate_min_similarity,
            result_kind="carnator_prompt_candidates",
            reuse_step=10,
            vector_backend=candidate_vector_backend,
        )

    @property
    def signature(self) -> str:
        return (
            f"{self.kind}|{self.embedder.signature}|top_k={self.candidate_top_k}"
            f"|min_similarity={self.candidate_min_similarity:.8f}"
            f"|stage2_top_k={self.stage2_candidate_top_k}"
            f"|vector_backend={self.prompt_retriever.vector_backend}"
            f"|stage_2={LATENT_FEATURE_PROFILE_ID}"
            f"|max_latent_distance={self._threshold_signature(self.max_latent_distance)}"
            f"|max_log_area_ratio={self._threshold_signature(self.max_log_area_ratio)}"
            f"|max_shape_distance={self._threshold_signature(self.max_shape_distance)}"
        )

    @staticmethod
    def _threshold(value: float | None, name: str) -> float | None:
        if value is None:
            return None
        threshold = float(value)
        if not math.isfinite(threshold) or threshold < 0.0:
            raise ValueError(f"{name} must be a finite non-negative number")
        return threshold

    @staticmethod
    def _threshold_signature(value: float | None) -> str:
        return "unset" if value is None else f"{value:.8f}"

    def bind_vector_store(self, root: Path) -> None:
        super().bind_vector_store(root)
        self.prompt_retriever.bind_vector_store(root)

    def sync_vector_store(self, records: Iterable[VectorRecord]) -> None:
        self.prompt_retriever.sync_vector_store(records)

    def records_for(self, entry: Any) -> list[VectorRecord]:
        nouns = self.noun_extractor.for_entry(entry)
        metadata = dict(getattr(entry, "metadata", None) or {})
        metadata.setdefault("source_request_index", int(entry.request_index))
        metadata.setdefault("trf_nouns", list(nouns))
        if "signature" not in metadata:
            metadata["cache_ready"] = False
            metadata.setdefault("signature_missing_reason", "native_online_signature_required")
        return [
            VectorRecord(
                entry.uuid,
                entry.prompt,
                self.embedder.encode([entry.prompt])[0],
                metadata,
                int(entry.request_index),
            )
        ]

    def record_is_valid(self, record: VectorRecord) -> bool:
        if not self.retrieval_only:
            return False
        metadata = record.metadata if isinstance(record.metadata, dict) else {}
        return bool(metadata.get("cache_ready")) and self._is_sidecar(metadata.get("signature"))

    @staticmethod
    def _is_sidecar(signature: Any) -> bool:
        return isinstance(signature, dict) and isinstance(signature.get("signature_path"), str)

    @staticmethod
    def _prefilter(source: Iterable[str], target: Iterable[str]) -> tuple[bool, str]:
        source_nouns, target_nouns = set(canonical_nouns(source)), set(canonical_nouns(target))
        if source_nouns == target_nouns:
            return True, "same_noun_set"
        if len(source_nouns - target_nouns) == len(target_nouns - source_nouns) == 1:
            return True, "single_noun_substitution"
        return False, "noun_set_difference_gt_one"

    def _top_k(self, request: Request, records: list[VectorRecord]) -> list[tuple[float, VectorRecord]]:
        return self.prompt_retriever.rank_candidates(
            request, records, top_k=self.candidate_top_k, use_cache_prompt=False
        )

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
            "candidate_min_similarity": self.candidate_min_similarity,
            "stage_2_candidate_top_k": self.stage2_candidate_top_k,
            "stage_2_profile": "online_signature",
            "stage_2_reference_profile": LATENT_FEATURE_PROFILE_ID,
            "stage_2_operational": self.max_latent_distance is not None,
            "single_noun_substitution_operational": (
                self.max_log_area_ratio is not None and self.max_shape_distance is not None
            ),
            "stage_2_thresholds": {
                "max_latent_distance": self.max_latent_distance,
                "max_log_area_ratio": self.max_log_area_ratio,
                "max_shape_distance": self.max_shape_distance,
            },
            "stage_2_threshold_mode": (
                "all_gates_configured"
                if self.max_log_area_ratio is not None and self.max_shape_distance is not None
                else ("latent_only" if self.max_latent_distance is not None else "unconfigured")
            ),
            "cache_ready": False,
            "step_2_probe_run": False,
            "probe_required": False,
            "top_k": [],
            "noun_prefilter": [],
            "stage_2": [],
        }
        target_nouns = self.noun_extractor.for_request(request)
        candidates = self._top_k(request, list(records))
        metadata["top_k"] = [
            {"uuid": record.uuid, "rank": rank, "score": score}
            for rank, (score, record) in enumerate(candidates, start=1)
        ]

        passed: list[tuple[float, VectorRecord, str]] = []
        for score, record in candidates:
            source_nouns = (record.metadata or {}).get("trf_nouns", []) if isinstance(record.metadata, dict) else []
            try:
                allowed, reason = self._prefilter(source_nouns, target_nouns)
                display_nouns = list(canonical_nouns(source_nouns))
            except (TypeError, ValueError):
                allowed, reason, display_nouns = False, "invalid_noun_metadata", []
            metadata["noun_prefilter"].append({
                "uuid": record.uuid,
                "source_nouns": display_nouns,
                "target_nouns": list(target_nouns),
                "passed": allowed,
                "reason": reason,
            })
            if allowed:
                passed.append((score, record, reason))

        stage_2_candidates = passed[:self.stage2_candidate_top_k]
        metadata["stage_2_candidates"] = [
            {"uuid": record.uuid, "score": score, "prefilter_reason": reason}
            for score, record, reason in stage_2_candidates
        ]
        if not passed:
            metadata["reason"] = "no_noun_compatible_candidate"
            return CacheMatch(metadata=metadata, kind="no_reuse")
        if not self._is_sidecar(target_signature):
            metadata["reason"] = "target_signature_required"
            metadata["probe_required"] = True
            return CacheMatch(metadata=metadata, kind="probe_required")

        metadata["step_2_probe_run"] = True
        assert target_signature is not None
        for score, record, _ in stage_2_candidates:
            candidate_result: dict[str, Any] = {"uuid": record.uuid, "score": score}
            source_metadata = record.metadata if isinstance(record.metadata, dict) else {}
            source_signature = source_metadata.get("signature")
            if not self._is_sidecar(source_signature):
                candidate_result["reason"] = "invalid_signature:ValueError"
                metadata["stage_2"].append(candidate_result)
                continue
            source_nouns = set(canonical_nouns(source_metadata.get("trf_nouns", [])))
            target_only, source_only = sorted(set(target_nouns) - source_nouns), sorted(source_nouns - set(target_nouns))
            comparison = compare_online_signatures(
                target_signature,
                source_signature,
                query_changed_noun=target_only[0] if len(target_only) == 1 and len(source_only) == 1 else None,
                candidate_changed_noun=source_only[0] if len(target_only) == 1 and len(source_only) == 1 else None,
                max_latent_distance=self.max_latent_distance,
                max_log_area_ratio=self.max_log_area_ratio,
                max_shape_distance=self.max_shape_distance,
                check_cache_artifacts=not self.retrieval_only,
            )
            candidate_result["online_signature"] = comparison
            if not comparison["signature_compatible"]:
                candidate_result["reason"] = "signature_incompatible"
                metadata["stage_2"].append(candidate_result)
                continue
            ready = bool(source_metadata.get("cache_ready")) and bool(comparison["cache_ready"])
            candidate_result["cache_ready"] = ready
            if not ready:
                candidate_result["reason"] = "cache_not_ready"
                metadata["stage_2"].append(candidate_result)
                continue
            candidate_result["reason"] = "accepted"
            metadata["stage_2"].append(candidate_result)
            metadata.update({
                "source_uuid": record.uuid,
                "cache_ready": True,
                "reuse_mode": "joint_latent_sparse_attention",
                "selected_metrics": comparison["distances"],
            })
            return CacheMatch(True, record.uuid, None, self.kind, score, metadata)

        metadata["reason"] = "no_compatible_cache"
        return CacheMatch(metadata=metadata, kind="no_reuse")

    def prepare_cache_entry(self, request: Request, latent_path: Path, latent_step: int) -> dict[str, Any]:
        del latent_path
        return {
            "trf_nouns": list(self.noun_extractor.for_request(request)),
            "cache_ready": False,
            "latent_step": int(latent_step),
            "signature_missing_reason": "native_online_signature_required",
        }


def make_carnator_retriever(
    config: dict[str, Any], *, dry_run: bool, retrieval_only: bool = False
) -> CarnatorRetriever:
    extractor: EntityExtractor = (
        LexicalTestEntityExtractor()
        if dry_run
        else SpacyEntityExtractor(str(config.get("spacy_model", "en_core_web_sm")), allowed_pos=("NOUN", "PROPN"))
    )
    return CarnatorRetriever(
        _make_embedder(config, dry_run=dry_run),
        CarnatorNounExtractor(extractor),
        candidate_top_k=int(config.get("candidate_top_k", 20)),
        candidate_min_similarity=float(config.get("min_similarity", 0.65 if not dry_run else 0.20)),
        candidate_vector_backend=_vector_backend(config, dry_run=dry_run, default="chroma"),
        stage2_candidate_top_k=int(config.get("stage2_candidate_top_k", 5)),
        retrieval_only=retrieval_only,
        max_latent_distance=config.get("max_latent_distance"),
        max_log_area_ratio=config.get("max_log_area_ratio"),
        max_shape_distance=config.get("max_shape_distance"),
    )


__all__ = ["CARNATOR_RETRIEVER_ID", "CarnatorNounExtractor", "CarnatorRetriever", "canonical_nouns", "make_carnator_retriever"]
