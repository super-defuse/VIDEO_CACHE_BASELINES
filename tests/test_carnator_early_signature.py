from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from carnator.early_signature import (
    EARLY_SIGNATURE_PROFILE_ID,
    compare_early_signatures,
    materialize_per_noun_inference_mask_bundle,
    per_noun_mbase_is_valid,
    select_per_noun_inference_mbase,
    write_early_signature,
)
from carnator.masks import (
    build_cross_attention_seed_scores,
    ensure_tail_mbase_support,
    load_mask_bundle,
    save_mask_bundle,
)
from carnator.retrieval import CarnatorNounExtractor, CarnatorRetriever, CarnatorRetrieverV2
from core.retrieval import LexicalTestEmbedder, VectorRecord
from core.types import Request
from scripts.benchmark import cross_request_retrieval


def _metadata(prompt: str) -> dict[str, object]:
    return {
        "prompt": prompt,
        "model": "test-wan",
        "scheduler": "unipc",
        "scheduler_config": {"flow_shift": 5.0, "solver_order": 2, "total_steps": 50},
        "cfg": 5.0,
        "probe_seed": 20260218,
        "completed_step": 5,
        "latent_shape": [2, 3, 4, 5],
        "latent_token_grid": [3, 4, 5],
        "noun_extractor": "test-nouns",
        "mask_policy": "test-mbase",
    }


def _sidecar(tmp_path, name: str, prompt: str, noun: str, colour: tuple[float, float, float]):
    directory = tmp_path / name
    directory.mkdir()
    mask = torch.zeros((3, 4, 5), dtype=torch.bool)
    mask[:, 1:3, 1:4] = True
    step10 = directory / "step10.pt"
    sparse = directory / "sparse.json"
    step10.write_bytes(b"step10")
    sparse.write_bytes(b"sparse")
    frame = torch.tensor(colour, dtype=torch.float32).view(3, 1, 1).expand(3, 8, 10).contiguous()
    return write_early_signature(
        directory,
        step5_latent=torch.ones((2, 3, 4, 5), dtype=torch.float32),
        decoded_tail_frame=frame,
        metadata=_metadata(prompt),
        mbase_by_noun={noun: mask},
        cache={
            "anchor_step": 10,
            "model": "test-wan",
            "latent_token_grid": [3, 4, 5],
            "latent_path": str(step10),
            "latent_sha256": hashlib.sha256(step10.read_bytes()).hexdigest(),
            "sparse_attention_path": str(sparse),
            "sparse_attention_sha256": hashlib.sha256(sparse.read_bytes()).hexdigest(),
        },
    )


def _multi_sidecar(tmp_path, name: str, prompt: str, masks: dict[str, torch.Tensor]):
    directory = tmp_path / name
    directory.mkdir()
    step10 = directory / "step10.pt"
    sparse = directory / "sparse.json"
    step10.write_bytes(b"step10")
    sparse.write_bytes(b"sparse")
    return write_early_signature(
        directory,
        step5_latent=torch.ones((2, 3, 4, 5), dtype=torch.float32),
        decoded_tail_frame=torch.ones((3, 8, 10), dtype=torch.float32),
        metadata=_metadata(prompt),
        mbase_by_noun=masks,
        cache={
            "anchor_step": 10,
            "model": "test-wan",
            "latent_token_grid": [3, 4, 5],
            "latent_path": str(step10),
            "latent_sha256": hashlib.sha256(step10.read_bytes()).hexdigest(),
            "sparse_attention_path": str(sparse),
            "sparse_attention_sha256": hashlib.sha256(sparse.read_bytes()).hexdigest(),
        },
    )


def _mask(*, rows: int, columns: int) -> torch.Tensor:
    value = torch.zeros((3, 4, 5), dtype=torch.bool)
    value[:, :rows, :columns] = True
    return value


def test_cross_attention_seed_scores_do_not_construct_a_mask():
    records = [
        {"step_index": 4, "layer_index": 0, "noun_score": torch.tensor([[[0.0, 1.0]]])},
        {"step_index": 4, "layer_index": 1, "noun_score": torch.tensor([[[1.0, 3.0]]])},
    ]

    bundle = build_cross_attention_seed_scores(records, completed_step=5)

    assert bundle["seed_variant"] == "cross_attention_seed_scores"
    assert set(bundle).isdisjoint({"Mbase", "Medit", "Msee", "raw_coverage"})
    assert torch.equal(bundle["score"], torch.tensor([[[0.0, 1.0]]]))


def test_fused_mbase_tail_support_copies_only_trailing_empty_slices():
    mask = torch.zeros((4, 3, 3), dtype=torch.bool)
    mask[1, 1, 1] = True

    repaired, metadata = ensure_tail_mbase_support(mask)

    assert repaired[0].sum() == 0
    assert repaired[1].equal(repaired[2])
    assert repaired[2].equal(repaired[3])
    assert metadata == {
        "applied": True,
        "mode": "nearest_nonempty_tail_slice",
        "filled_slices": 2,
        "source_slice": 1,
    }


def test_early_signature_persists_step5_tail_frame_and_accepts_matching_colour(tmp_path):
    source = _sidecar(tmp_path, "source", "a red train", "train", (1.0, -1.0, -1.0))
    target = _sidecar(tmp_path, "target", "a red train", "train", (1.0, -1.0, -1.0))

    result = compare_early_signatures(
        target, source, max_context_color_distance=0.01, check_cache_artifacts=True
    )

    assert result["reusable"] is True
    assert result["profile"] == EARLY_SIGNATURE_PROFILE_ID
    assert result["distances"]["context_color_ciede2000"] == 0.0
    assert (tmp_path / "source" / "step5_tail_frame.png").is_file()
    assert (tmp_path / "source" / "online_color_signature.json").is_file()


def test_per_noun_inference_selects_source_only_mask_for_substitution(tmp_path):
    source = _multi_sidecar(
        tmp_path,
        "source",
        "a cat and dog",
        {"cat": _mask(rows=1, columns=1), "dog": _mask(rows=2, columns=2)},
    )

    selection, mask = select_per_noun_inference_mbase(source, target_nouns=("dog", "beagle"))

    assert selection["selected_source_noun"] == "cat"
    assert selection["selection_reason"] == "single_noun_substitution_source_only"
    assert selection["source_only_nouns"] == ["cat"]
    assert selection["target_only_nouns"] == ["beagle"]
    assert torch.equal(mask, _mask(rows=1, columns=1))


def test_per_noun_inference_uses_smallest_mask_for_equal_noun_sets(tmp_path):
    source = _multi_sidecar(
        tmp_path,
        "source",
        "a cat and dog",
        {"dog": _mask(rows=2, columns=2), "cat": _mask(rows=1, columns=1)},
    )

    selection, mask = select_per_noun_inference_mbase(source, target_nouns=("dog", "cat"))

    assert selection["selected_source_noun"] == "cat"
    assert selection["selection_reason"] == "same_noun_set_smallest_area"
    assert selection["selected_mask_area_tokens"] == 3
    assert torch.equal(mask, _mask(rows=1, columns=1))


def test_per_noun_inference_materializes_private_bundle_without_mutating_source(tmp_path):
    source = _multi_sidecar(
        tmp_path,
        "source",
        "a cat and dog",
        {"cat": _mask(rows=1, columns=1), "dog": _mask(rows=2, columns=2)},
    )
    shared = tmp_path / "shared-fusion"
    save_mask_bundle(
        shared,
        {
            "source_completed_step": 5,
            "mask_spatial_medit_radius": 1,
            "mask_spatial_msee_radius": 1,
            "token_shape": [3, 4, 5],
            "Mbase": _mask(rows=3, columns=3),
            "Medit": _mask(rows=3, columns=3),
            "Msee": _mask(rows=3, columns=3),
        },
    )
    source_bytes = (shared / "masks.pt").read_bytes()

    selection = materialize_per_noun_inference_mask_bundle(
        source,
        target_nouns=("dog", "beagle"),
        source_fusion_mask_dir=shared,
        output_dir=tmp_path / "private",
    )
    private = load_mask_bundle(Path(selection["mask_bundle_dir"]))

    assert selection["selected_source_noun"] == "cat"
    assert private["mask_variant"] == "source_per_noun_mbase"
    assert torch.equal(private["Mbase"], _mask(rows=1, columns=1))
    assert (shared / "masks.pt").read_bytes() == source_bytes


def test_per_noun_inference_rejects_sidecar_without_its_contract(tmp_path):
    source = _sidecar(tmp_path, "source", "a red train", "train", (1.0, -1.0, -1.0))
    path = Path(source["signature_path"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("inference_mask_contract")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    assert not per_noun_mbase_is_valid(source)


def test_early_signature_rejects_different_same_noun_colour(tmp_path):
    source = _sidecar(tmp_path, "source", "a red train", "train", (1.0, -1.0, -1.0))
    target = _sidecar(tmp_path, "target", "a blue train", "train", (-1.0, -1.0, 1.0))

    result = compare_early_signatures(
        target, source, max_context_color_distance=1.0, check_cache_artifacts=True
    )

    assert result["reusable"] is False
    assert result["gates"]["context_color_ciede2000"] is False
    assert result["distances"]["context_color_ciede2000"] > 1.0


def test_early_signature_reads_historical_versioned_sidecars(tmp_path):
    source = _sidecar(tmp_path, "source", "a red train", "train", (1.0, -1.0, -1.0))
    target = _sidecar(tmp_path, "target", "a red train", "train", (1.0, -1.0, -1.0))
    for signature in (source, target):
        path = Path(signature["signature_path"])
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.update({
            "signature_version": "carnator-online-color-signature/v1",
            "feature_profile": "carnator-step5-tail-lab-ciede2000/v1",
            "color_extraction_policy": "srgb-d65-2deg-per-pixel-lab-mean/v1",
            "shape_normalization_policy": "shape_norm_v1",
        })
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result = compare_early_signatures(
        target, source, max_context_color_distance=0.01, check_cache_artifacts=True
    )

    assert result["reusable"] is True


def test_early_signature_uses_area_shape_and_changed_colour_gates(tmp_path):
    source = _sidecar(tmp_path, "source", "a red cat", "cat", (1.0, -1.0, -1.0))
    target = _sidecar(tmp_path, "target", "a red dog", "dog", (1.0, -1.0, -1.0))

    accepted = compare_early_signatures(
        target,
        source,
        query_changed_noun="dog",
        candidate_changed_noun="cat",
        max_context_color_distance=0.01,
        max_log_area_ratio=0.01,
        max_shape_distance=0.01,
        max_changed_color_distance=0.01,
        check_cache_artifacts=True,
    )

    assert accepted["reusable"] is True
    assert accepted["gates"] == {
        "context_color_ciede2000": True,
        "log_area_ratio": True,
        "shape_one_minus_dice": True,
        "changed_color_ciede2000": True,
    }

    blue_target = _sidecar(tmp_path, "blue-target", "a blue dog", "dog", (-1.0, -1.0, 1.0))
    rejected = compare_early_signatures(
        blue_target,
        source,
        query_changed_noun="dog",
        candidate_changed_noun="cat",
        max_context_color_distance=0.01,
        max_log_area_ratio=0.01,
        max_shape_distance=0.01,
        max_changed_color_distance=1.0,
        check_cache_artifacts=True,
    )

    assert rejected["reusable"] is False
    assert rejected["gates"]["changed_color_ciede2000"] is False


def test_early_signature_routes_ciede2000_gate_through_carnator_retriever(tmp_path):
    source_signature = _sidecar(tmp_path, "source", "a red train", "train", (1.0, -1.0, -1.0))
    target_signature = _sidecar(tmp_path, "target", "a red train", "train", (1.0, -1.0, -1.0))
    embedder = LexicalTestEmbedder()
    retriever = CarnatorRetriever(
        embedder,
        CarnatorNounExtractor(),
        candidate_min_similarity=-1.0,
        retrieval_only=True,
        max_context_color_distance=0.01,
    )
    source = VectorRecord(
        "source",
        "train",
        embedder.encode(["train"])[0],
        {
            "source_request_index": 0,
            "trf_nouns": ["train"],
            "cache_ready": True,
            "signature": source_signature,
        },
        0,
    )

    result = retriever.search(
        Request(1, "target", "train", 2), [source], target_signature=target_signature
    )

    assert result.hit
    assert result.metadata["stage_2_reference_profile"] == EARLY_SIGNATURE_PROFILE_ID
    assert result.metadata["stage_2_profile"] == "early_signature"
    assert result.metadata["stage_2"][0]["early_signature"]["reusable"] is True
    assert result.metadata["inference_mask_selection"]["selected_source_noun"] == "train"


def test_v2_retriever_propagates_selected_per_noun_mask(tmp_path):
    source_signature = _sidecar(tmp_path, "source", "a red train", "train", (1.0, -1.0, -1.0))
    target_signature = _sidecar(tmp_path, "target", "a red train", "train", (1.0, -1.0, -1.0))
    embedder = LexicalTestEmbedder()
    retriever = CarnatorRetrieverV2(
        embedder,
        CarnatorNounExtractor(),
        candidate_min_similarity=-1.0,
        retrieval_only=True,
        max_context_color_distance=0.01,
    )
    source = VectorRecord(
        "source",
        "train",
        embedder.encode(["train"])[0],
        {
            "source_request_index": 0,
            "trf_nouns": ["train"],
            "cache_ready": True,
            "signature": source_signature,
        },
        0,
    )

    result = retriever.search(
        Request(1, "target", "train", 2), [source], target_signature=target_signature
    )

    assert result.hit
    assert result.metadata["selection_order"] == ["noun_prefilter", "early_signature", "clip_top_1"]
    assert result.metadata["inference_mask_selection"]["selected_source_noun"] == "train"
    assert result.metadata["stage_2"][0]["inference_mask_selection"]["selection_reason"] == (
        "same_noun_set_smallest_area"
    )


def test_dry_run_benchmark_writes_early_signature_tail_frame_sidecars(tmp_path):
    workload = tmp_path / "workload.csv"
    workload.write_text(
        "request_index,uuid,prompt\n"
        "0,first,train\n"
        "1,second,train\n",
        encoding="utf-8",
    )
    artifact_dir = tmp_path / "artifacts"

    assert cross_request_retrieval.main([
        "--workload", str(workload),
        "--artifact-dir", str(artifact_dir),
        "--shuffle-seed", "0",
        "--dry-run",
        "--assume-step10-cache",
    ]) == 0

    benchmark = json.loads((artifact_dir / "benchmark.json").read_text(encoding="utf-8"))
    assert benchmark["carnator"]["early_signature"] == "step5-tail-lab-ciede2000"
    assert benchmark["carnator"]["video_decoded"] is True
    source_dirs = [path for path in (artifact_dir / "carnator_sources").iterdir() if path.is_dir()]
    assert source_dirs
    assert all((path / "online_color_signature.json").is_file() for path in source_dirs)
    assert all((path / "step5_tail_frame.png").is_file() for path in source_dirs)
