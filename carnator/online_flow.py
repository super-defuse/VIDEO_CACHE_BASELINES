"""The request-local orchestration for Carnator online retrieval.

The native Wan collector is supplied as a callback so this module owns the
retrieval order without coupling it to a particular model runtime.  A target
signature is collected exactly once, and only after stage 1 has found a
noun-compatible historical candidate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

from carnator.retrieval import CarnatorRetriever
from core.retrieval import VectorRecord
from core.types import CacheMatch, Request


@dataclass(frozen=True)
class OnlineRetrievalDecision:
    """The hand-off from retrieval authorization to native generation."""

    action: str
    match: CacheMatch
    target_signature: dict[str, Any] | None
    probe_collected: bool


TargetSignatureCollector = Callable[[], Mapping[str, Any]]


def resolve_online_request(
    retriever: CarnatorRetriever,
    request: Request,
    records: Iterable[VectorRecord],
    *,
    collect_target_signature: TargetSignatureCollector,
) -> OnlineRetrievalDecision:
    """Run the plan's recall, one-probe, signature-gate decision sequence.

    ``collect_target_signature`` is responsible for running the target's
    unchanged scheduler through completed step 5 and writing a native
    ``online_signature`` sidecar.  It is never called when stage 1 has no
    noun-compatible candidate.
    """

    candidates = list(records)
    preflight = retriever.search(request, candidates)
    if preflight.kind != "probe_required":
        return OnlineRetrievalDecision(
            action="dense_generation",
            match=preflight,
            target_signature=None,
            probe_collected=False,
        )

    target_signature = dict(collect_target_signature())
    if not isinstance(target_signature.get("signature_path"), str):
        raise ValueError("target signature collector must return a signature_path sidecar")
    match = retriever.search(request, candidates, target_signature=target_signature)
    return OnlineRetrievalDecision(
        action="joint_reuse" if match.hit else "dense_generation",
        match=match,
        target_signature=target_signature,
        probe_collected=True,
    )


__all__ = ["OnlineRetrievalDecision", "TargetSignatureCollector", "resolve_online_request"]
