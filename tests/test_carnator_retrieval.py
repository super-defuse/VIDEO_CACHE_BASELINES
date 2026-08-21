from __future__ import annotations

import csv
import json

from carnator.retrieval import (
    CARNATOR_IMPLEMENTATION_ID,
    CARNATOR_SIGNATURE_PROFILE_ID,
    CarnatorNounAnnotations,
    CarnatorRetriever,
    CarnatorSignatureBuilder,
    CarnatorThresholdConfig,
    build_native_prompt_ets_signature,
)
from core.retrieval import LexicalTestEmbedder, VectorRecord
from core.types import Request
from scripts.run import carnator_dataset_ets
from scripts.run import cross_request_dataset


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


def _native_signature(
    prompt: str,
    nouns: list[str],
    *,
    offset: float = 0.0,
    request_seed: int | None = None,
):
    return build_native_prompt_ets_signature(
        prompt,
        nouns,
        {
            "source_completed_step": 5,
            "prompt": prompt,
            "mask_variant": "self_contrast_cross_attention_fusion",
            "mask_positive": "",
            "Mbase": [[[1, 0], [0, 1]], [[1, 0], [0, 1]]],
            "Medit": [[[1, 1], [0, 1]], [[1, 1], [0, 1]]],
            "Msee": [[[1, 1], [1, 1]], [[1, 1], [1, 1]]],
            "score": [
                [[0.1 + offset, 0.3], [0.2, 0.4]],
                [[0.2, 0.4 + offset], [0.3, 0.5]],
            ],
        },
        request_seed=request_seed,
    )


def test_native_prompt_ets_requires_a_real_target_probe_before_authorization():
    source = _record("source", 0, "a train", ["train"])
    source.metadata["signature"] = _native_signature("a train", ["train"])
    retriever = CarnatorRetriever(
        LexicalTestEmbedder(), CarnatorNounAnnotations(),
        CarnatorThresholdConfig(
            "test-frozen", CARNATOR_IMPLEMENTATION_ID, CARNATOR_SIGNATURE_PROFILE_ID,
            {"mbase_iou": 0.5, "medit_iou": 0.5, "msee_iou": 0.5, "ets_cosine": 0.8}, True, True,
        ),
    )
    request = Request(1, "target", "a train", 2)

    pending = retriever.search(request, [source])

    assert not pending.hit
    assert pending.kind == "probe_required"
    assert pending.metadata["reason"] == "target_signature_required"
    assert pending.metadata["probe_required"] is True
    assert pending.metadata["step_2_probe_run"] is False


def test_native_prompt_ets_compares_compressed_masks_after_the_probe():
    source = _record("source", 0, "a train", ["train"])
    source.metadata["signature"] = _native_signature("a train", ["train"])
    target_signature = _native_signature("a train", ["train"])
    retriever = _retriever(retrieval_only=True)

    result = retriever.search(
        Request(1, "target", "a train", 2), [source], target_signature=target_signature
    )

    assert result.hit
    assert result.metadata["step_2_probe_run"] is True
    assert result.metadata["selected_metrics"]["mbase_iou"] == 1.0
    assert result.metadata["selected_metrics"]["ets_cosine"] > 0.99


def test_native_prompt_ets_rejects_a_signature_from_another_prompt():
    source = _record("source", 0, "a train", ["train"])
    source.metadata["signature"] = _native_signature("a train", ["train"])
    retriever = _retriever(retrieval_only=True)

    result = retriever.search(
        Request(1, "target", "a train", 2),
        [source],
        target_signature=_native_signature("a different train", ["train"]),
    )

    assert not result.hit
    assert result.metadata["reason"] == "target_signature_prompt_mismatch"


def test_dataset_replay_uses_native_ets_and_writes_comparison_outputs(tmp_path):
    workload = tmp_path / "workload.csv"
    workload.write_text(
        "request_index,uuid,prompt\n"
        "0,first,a train\n"
        "1,second,a train\n"
        "2,third,a train\n",
        encoding="utf-8",
    )
    signatures = tmp_path / "native_ets.jsonl"
    signatures.write_text(
        "".join(
            json.dumps({
                "uuid": uuid,
                "signature": _native_signature("a train", ["train"], request_seed=20260218 + index),
            }) + "\n"
            for index, uuid in enumerate(("first", "second", "third"))
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "replay"

    assert cross_request_dataset.main([
        "--workload", str(workload),
        "--native-ets-jsonl", str(signatures),
        "--methods", "nirvana_vid,carnator",
        "--output-dir", str(output_dir),
        "--group-size", "2",
        "--shuffle-seed", "0",
        "--dry-run",
    ]) == 0

    with (output_dir / "selection_comparison.csv").open(newline="", encoding="utf-8") as handle:
        selected = list(csv.DictReader(handle))
    assert selected[0]["carnator_hit"] == "False"
    assert selected[1]["carnator_source_uuid"] == "first"
    assert selected[1]["nirvana_vid_source_uuid"] == "first"
    with (output_dir / "hit_rate_by_group.csv").open(newline="", encoding="utf-8") as handle:
        groups = list(csv.DictReader(handle))
    assert len(groups) == 4
    assert (output_dir / "hit_rate_by_group.svg").is_file()


def test_dataset_replay_can_resolve_organized_run_layout(tmp_path):
    workload = tmp_path / "workload.csv"
    workload.write_text(
        "request_index,uuid,prompt\n"
        "0,first,a train\n"
        "1,second,a train\n",
        encoding="utf-8",
    )
    assert cross_request_dataset.main([
        "--workload", str(workload),
        "--methods", "chai",
        "--artifact-root", str(tmp_path / "artifacts"),
        "--run-id", "test-run",
        "--run-name", "chai",
        "--shuffle-seed", "0",
        "--dry-run",
    ]) == 0
    output_dir = tmp_path / "artifacts" / "test-run" / "runs" / "chai"
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifact_layout"] == "cross-request-retrieval/v1"
    assert manifest["run_id"] == "test-run"
    assert manifest["run_name"] == "chai"


def test_native_dataset_ets_collector_forces_a_step_five_source_probe(tmp_path):
    args = carnator_dataset_ets.build_parser().parse_args(["--output", str(tmp_path / "ets.jsonl")])

    carnator_dataset_ets._validate_args(args)

    assert args.source_only is True
    assert args.signature_only is True
    assert args.steps == args.mask_probe_step == 5
