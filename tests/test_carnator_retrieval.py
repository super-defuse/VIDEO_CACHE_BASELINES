from __future__ import annotations

from carnator.retrieval import (
    CARNATOR_IMPLEMENTATION_ID,
    CARNATOR_SIGNATURE_PROFILE_ID,
    CarnatorNounAnnotations,
    CarnatorRetriever,
    CarnatorSignatureBuilder,
    CarnatorThresholdConfig,
)
from core.retrieval import LexicalTestEmbedder, VectorRecord
from core.types import Request


def _retriever(*, retrieval_only: bool = True, threshold_config=None):
    return CarnatorRetriever(
        LexicalTestEmbedder(), CarnatorNounAnnotations(),
        threshold_config or CarnatorThresholdConfig(
            "test-frozen", CARNATOR_IMPLEMENTATION_ID, CARNATOR_SIGNATURE_PROFILE_ID,
            {"mbase_iou": 0.5, "medit_iou": 0.5, "msee_iou": 0.5, "ets_cosine": 0.8}, True, True,
        ),
        retrieval_only=retrieval_only,
    )


def _record(uuid: str, request_index: int, prompt: str, nouns: list[str], *, ready: bool = True):
    builder = CarnatorSignatureBuilder()
    manifest = {
        "implementation_id": CARNATOR_IMPLEMENTATION_ID,
        "signature_profile_id": CARNATOR_SIGNATURE_PROFILE_ID,
        "threshold_config_id": "test-frozen",
        "uuid": uuid,
        "namespace": "fixture",
        "completed_step": 5,
        "reuse_anchor_step": 10,
    }
    metadata = {
        "source_request_index": request_index,
        "trf_nouns": nouns,
        "signature": builder.build(prompt, nouns),
        "cache_ready": ready,
        "manifest": manifest,
    }
    return VectorRecord(uuid, prompt, LexicalTestEmbedder().encode([prompt])[0], metadata, request_index)


def test_carnator_accepts_same_nouns_only_when_fixture_is_cache_ready():
    retriever = _retriever()
    source = _record("source", 0, "train", ["train"])
    result = retriever.search(Request(1, "target", "train", 2), [source])
    assert result.hit
    assert result.source_uuid == "source"
    assert result.metadata["reuse_mode"] == "joint_latent_sparse_attention"

    rejected = retriever.search(Request(1, "target", "train", 2),
                                [_record("source", 0, "train", ["train"], ready=False)])
    assert not rejected.hit
    assert rejected.metadata["stage_2"][0]["reason"] == "cache_not_ready"


def test_carnator_noun_prefilter_skips_step_five_probe_for_multiple_differences():
    retriever = _retriever()
    source = _record("source", 0, "a red train", ["train", "station"])
    result = retriever.search(Request(1, "target", "a blue dog", 2), [source])
    assert not result.hit
    assert result.metadata["step_2_probe_run"] is False
    assert result.metadata["reason"] == "no_noun_compatible_candidate"


def test_carnator_rejects_unauthenticated_threshold_config_before_probe():
    config = CarnatorThresholdConfig("untrusted", "wrong", "wrong", {}, False, False)
    result = _retriever(threshold_config=config).search(
        Request(1, "target", "train", 2), [_record("source", 0, "train", ["train"])])
    assert not result.hit
    assert result.metadata["reason"] == "threshold_config_unauthenticated"
    assert result.metadata["step_2_probe_run"] is False
