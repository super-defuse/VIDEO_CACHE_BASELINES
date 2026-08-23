from __future__ import annotations

import csv
import hashlib
import json

import torch

from carnator.adapters.wan_ti2v_5b import find_token_indices
from carnator.early_signature import compare_early_signatures, write_early_signature
from carnator.online_flow import resolve_online_request
from carnator.retrieval import CarnatorNounExtractor, CarnatorRetriever
from core.retrieval import LexicalTestEmbedder, VectorRecord
from core.types import Request
from scripts.benchmark import cross_request_retrieval


def _sidecar(
    tmp_path,
    name: str,
    prompt: str,
    nouns: list[str],
    *,
    colour: tuple[float, float, float] = (1.0, -1.0, -1.0),
):
    work = tmp_path / name
    work.mkdir()
    masks = {}
    for index, noun in enumerate(nouns):
        mask = torch.zeros((3, 4, 5), dtype=torch.bool)
        mask[:, 1:3, 1 + index:3 + index] = True
        masks[noun] = mask
    step10 = work / "step10.pt"
    sparse = work / "sparse.bin"
    step10.write_bytes(b"step10")
    sparse.write_bytes(b"sparse")
    frame = torch.tensor(colour, dtype=torch.float32).view(3, 1, 1).expand(3, 8, 10).contiguous()
    return write_early_signature(
        work,
        step5_latent=torch.ones((2, 3, 4, 5), dtype=torch.float32),
        decoded_tail_frame=frame,
        metadata={
            "prompt": prompt,
            "model": "test-wan",
            "scheduler": "unipc",
            "scheduler_config": {"flow_shift": 5.0, "solver_order": 2},
            "cfg": 5.0,
            "probe_seed": 20260218,
            "completed_step": 5,
            "latent_shape": [2, 3, 4, 5],
            "latent_token_grid": [3, 4, 5],
            "noun_extractor": "test-nouns",
            "mask_policy": "test-mbase",
        },
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


def _retriever(*, retrieval_only: bool = True):
    return CarnatorRetriever(
        LexicalTestEmbedder(),
        CarnatorNounExtractor(),
        retrieval_only=retrieval_only,
        max_context_color_distance=0.01,
        max_log_area_ratio=0.01,
        max_shape_distance=0.01,
        max_changed_color_distance=0.01,
    )


def _record(uuid: str, request_index: int, prompt: str, nouns: list[str], signature, *, ready: bool = True):
    metadata = {
        "source_request_index": request_index,
        "trf_nouns": nouns,
        "signature": signature,
        "cache_ready": ready,
    }
    return VectorRecord(uuid, prompt, LexicalTestEmbedder().encode([prompt])[0], metadata, request_index)


def test_carnator_accepts_matching_early_signature_when_fixture_is_cache_ready(tmp_path):
    source_signature = _sidecar(tmp_path, "source", "train", ["train"])
    target_signature = _sidecar(tmp_path, "target", "train", ["train"])
    source = _record("source", 0, "train", ["train"], source_signature)

    result = _retriever().search(Request(1, "target", "train", 2), [source], target_signature=target_signature)

    assert result.hit
    assert result.source_uuid == "source"
    assert result.metadata["reuse_mode"] == "joint_latent_sparse_attention"
    assert result.metadata["stage_2_profile"] == "early_signature"
    assert result.metadata["stage_2"][0]["early_signature"]["reusable"] is True


def test_carnator_rejects_when_signature_gate_passes_but_cache_is_not_ready(tmp_path):
    source_signature = _sidecar(tmp_path, "source", "train", ["train"])
    target_signature = _sidecar(tmp_path, "target", "train", ["train"])
    result = _retriever().search(
        Request(1, "target", "train", 2),
        [_record("source", 0, "train", ["train"], source_signature, ready=False)],
        target_signature=target_signature,
    )

    assert not result.hit
    assert result.metadata["stage_2"][0]["reason"] == "cache_not_ready"


def test_early_signature_rejects_a_changed_step10_artifact(tmp_path):
    source_signature = _sidecar(tmp_path, "source", "train", ["train"])
    target_signature = _sidecar(tmp_path, "target", "train", ["train"])
    accepted = compare_early_signatures(
        target_signature, source_signature, max_context_color_distance=0.01, check_cache_artifacts=True
    )
    assert accepted["reusable"] is True
    (tmp_path / "source" / "step10.pt").write_bytes(b"changed")
    rejected = compare_early_signatures(
        target_signature, source_signature, max_context_color_distance=0.01, check_cache_artifacts=True
    )
    assert rejected["reusable"] is False
    assert rejected["reason"] == "step10_artifact_unreadable_or_changed"


def test_carnator_uses_colour_area_shape_gates_for_one_noun_change(tmp_path):
    source_signature = _sidecar(tmp_path, "source", "a ball", ["ball"])
    target_signature = _sidecar(tmp_path, "target", "a cube", ["cube"])
    retriever = CarnatorRetriever(
        LexicalTestEmbedder(),
        CarnatorNounExtractor(),
        candidate_min_similarity=-1.0,
        retrieval_only=True,
        max_context_color_distance=0.01,
        max_log_area_ratio=0.01,
        max_shape_distance=0.01,
        max_changed_color_distance=0.01,
    )
    result = retriever.search(
        Request(1, "target", "a cube", 2),
        [_record("source", 0, "a ball", ["ball"], source_signature)],
        target_signature=target_signature,
    )

    assert result.hit
    comparison = result.metadata["stage_2"][0]["early_signature"]
    assert comparison["gates"]["context_color_ciede2000"] is True
    assert comparison["gates"]["log_area_ratio"] is True
    assert comparison["gates"]["shape_one_minus_dice"] is True
    assert comparison["gates"]["changed_color_ciede2000"] is True


def test_carnator_noun_prefilter_skips_early_signature_for_multiple_differences():
    source = VectorRecord(
        "source",
        "a red train",
        LexicalTestEmbedder().encode(["a red train"])[0],
        {"trf_nouns": ["train", "station"]},
        0,
    )

    result = _retriever().search(Request(1, "target", "a blue dog", 2), [source])

    assert not result.hit
    assert result.metadata["step_2_probe_run"] is False
    assert result.metadata["reason"] == "no_noun_compatible_candidate"


def test_carnator_requires_a_target_sidecar_after_stage_one(tmp_path):
    source_signature = _sidecar(tmp_path, "source", "a train", ["train"])
    result = _retriever(retrieval_only=False).search(
        Request(1, "target", "a train", 2),
        [_record("source", 0, "a train", ["train"], source_signature)],
    )

    assert not result.hit
    assert result.kind == "probe_required"
    assert result.metadata["reason"] == "target_signature_required"


def test_online_flow_collects_one_target_signature_then_authorizes_joint_reuse(tmp_path):
    source_signature = _sidecar(tmp_path, "source", "a train", ["train"])
    target_signature = _sidecar(tmp_path, "target", "a train", ["train"])
    calls = []

    decision = resolve_online_request(
        _retriever(retrieval_only=False),
        Request(1, "target", "a train", 2),
        [_record("source", 0, "a train", ["train"], source_signature)],
        collect_target_signature=lambda: (calls.append("probe") or target_signature),
    )

    assert calls == ["probe"]
    assert decision.probe_collected is True
    assert decision.action == "joint_reuse"
    assert decision.match.source_uuid == "source"


def test_online_flow_skips_probe_without_a_noun_compatible_candidate(tmp_path):
    source_signature = _sidecar(tmp_path, "source", "a train", ["train"])
    calls = []

    decision = resolve_online_request(
        _retriever(retrieval_only=False),
        Request(1, "target", "a dog and a cat", 2),
        [_record("source", 0, "a train", ["train"], source_signature)],
        collect_target_signature=lambda: (calls.append("probe") or source_signature),
    )

    assert calls == []
    assert decision.probe_collected is False
    assert decision.action == "dense_generation"


def test_online_flow_falls_back_to_dense_after_an_incompatible_signature(tmp_path):
    source_signature = _sidecar(tmp_path, "source", "a train", ["train"])
    target_signature = _sidecar(
        tmp_path, "target", "a train", ["train"], colour=(-1.0, -1.0, 1.0)
    )
    calls = []

    decision = resolve_online_request(
        _retriever(retrieval_only=False),
        Request(1, "target", "a train", 2),
        [_record("source", 0, "a train", ["train"], source_signature)],
        collect_target_signature=lambda: (calls.append("probe") or target_signature),
    )

    assert calls == ["probe"]
    assert decision.probe_collected is True
    assert decision.action == "dense_generation"
    assert decision.match.metadata["stage_2"][0]["early_signature"]["gates"]["context_color_ciede2000"] is False


def test_carnator_uses_top_twenty_and_limits_stage_two_to_five(tmp_path):
    class FixedEmbedder:
        @property
        def signature(self) -> str:
            return "fixed-clip"

        def encode(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0] for _ in texts]

    source_signature = _sidecar(tmp_path, "source", "train", ["train"])
    target_signature = _sidecar(tmp_path, "target", "train", ["train"])
    retriever = CarnatorRetriever(
        FixedEmbedder(),
        CarnatorNounExtractor(),
        candidate_top_k=20,
        candidate_min_similarity=0.65,
        stage2_candidate_top_k=5,
        retrieval_only=True,
        max_context_color_distance=0.01,
        max_log_area_ratio=0.01,
        max_shape_distance=0.01,
        max_changed_color_distance=0.01,
    )
    records = []
    for index, score in enumerate((0.99, 0.98, 0.97, 0.96, 0.95, 0.94), start=1):
        record = _record(f"source-{index}", index, "train", ["train"], source_signature, ready=index == 6)
        records.append(VectorRecord(record.uuid, record.term, [score, (1.0 - score * score) ** 0.5], record.metadata, record.source_request_index))
    records.append(VectorRecord("below-floor", "train", [0.64, 0.0], records[0].metadata, 7))

    result = retriever.search(Request(8, "target", "train", 8), records, target_signature=target_signature)

    assert not result.hit
    assert [candidate["uuid"] for candidate in result.metadata["top_k"]] == [
        "source-1", "source-2", "source-3", "source-4", "source-5", "source-6",
    ]
    assert [candidate["uuid"] for candidate in result.metadata["stage_2_candidates"]] == [
        "source-1", "source-2", "source-3", "source-4", "source-5",
    ]
    assert len(result.metadata["stage_2"]) == 5
    assert all(candidate["reason"] == "cache_not_ready" for candidate in result.metadata["stage_2"])


def test_carnator_runtime_extractor_preserves_chai_noun_tokens():
    class ChaiLikeExtractor:
        @property
        def signature(self) -> str:
            return "chai-like"

        def extract(self, prompt: str) -> list[str]:
            assert prompt == "A Corgi beside facades"
            return ["Corgi", "Corgi", "façades"]

    extractor = CarnatorNounExtractor(ChaiLikeExtractor())
    assert extractor.for_request(Request(0, "prompt", "A Corgi beside facades", 1)) == (
        "Corgi", "Corgi", "façades",
    )


def test_carnator_rejects_nounless_requests():
    class EmptyExtractor:
        @property
        def signature(self) -> str:
            return "empty"

        def extract(self, prompt: str) -> list[str]:
            return []

    retriever = CarnatorRetriever(LexicalTestEmbedder(), CarnatorNounExtractor(EmptyExtractor()))
    match = retriever.search(Request(0, "nounless", "a remote", 1), [])
    assert match.kind == "no_reuse"
    assert match.metadata["reason"] == "nounless_prompt"


def test_carnator_native_token_lookup_folds_accents():
    class Tokenizer:
        def __call__(self, prompt, **kwargs):
            del kwargs
            return {
                "offset_mapping": torch.tensor([[[0, len(prompt)]]]),
                "attention_mask": torch.tensor([[1]]),
            }

    class Wrapper:
        pass

    class Encoder:
        pass

    wrapper = Wrapper()
    wrapper.tokenizer = Tokenizer()
    encoder = Encoder()
    encoder.tokenizer = wrapper

    assert find_token_indices(encoder, "ancient façades", "façades", 8) == [0]


def test_cross_request_benchmark_writes_all_four_reports(tmp_path):
    workload = tmp_path / "workload.csv"
    workload.write_text(
        "request_index,uuid,prompt\n"
        "0,first,a train\n"
        "1,second,a train\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "artifacts"

    assert cross_request_retrieval.main([
        "--workload", str(workload), "--artifact-dir", str(output_dir), "--shuffle-seed", "0", "--dry-run",
    ]) == 0

    with (output_dir / "reports" / "request_matches.csv").open(newline="", encoding="utf-8") as handle:
        selected = list(csv.DictReader(handle))
    assert selected[0]["carnator_hit"] == "False"
    assert selected[1]["carnator_source_uuid"] == ""
    assert selected[1]["nirvana_vid_source_uuid"] == selected[0]["uuid"]
    benchmark = json.loads((output_dir / "benchmark.json").read_text(encoding="utf-8"))
    assert benchmark["schema_version"] == "cross-request-retrieval/v3-progressive-online"
    assert benchmark["carnator"]["sidecar_mode"] == "arrival_order_miss_only"
    assert benchmark["methods"]["carnator"]["stage_2"] == "progressive_early_signature"
    assert benchmark["methods"]["carnator"]["early_probe_requests"] == 1
    assert benchmark["methods"]["carnator"]["source_cache_entries"] == 2
    carnator_rows = [
        json.loads(line)
        for line in (output_dir / "results" / "carnator.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert carnator_rows[0]["probe_collected"] is False
    assert carnator_rows[0]["source_cached"] is True
    assert carnator_rows[1]["probe_collected"] is True
    assert carnator_rows[1]["source_cached"] is True
    assert len([path for path in (output_dir / "carnator_sources").iterdir() if path.is_dir()]) == 2
    assert (output_dir / "reports" / "hit_rate_over_time.csv").is_file()
    assert (output_dir / "reports" / "retrieval_summary.csv").is_file()
    assert (output_dir / "reports" / "cache_growth.csv").is_file()
    assert set(path.name for path in (output_dir / "results").iterdir()) == {
        "nirvana_vid.jsonl", "chai.jsonl", "chorus.jsonl", "carnator.jsonl",
    }


def test_cross_request_benchmark_can_assume_step10_cache(tmp_path):
    workload = tmp_path / "workload.csv"
    workload.write_text(
        "request_index,uuid,prompt\n"
        "0,first,a train\n"
        "1,second,a train\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "artifacts"

    assert cross_request_retrieval.main([
        "--workload", str(workload), "--artifact-dir", str(output_dir), "--shuffle-seed", "0", "--dry-run",
        "--assume-step10-cache",
    ]) == 0

    benchmark = json.loads((output_dir / "benchmark.json").read_text(encoding="utf-8"))
    assert benchmark["schema_version"] == "cross-request-retrieval/v4-retrieval-only-assumed-step10"
    assert benchmark["carnator"]["step10_cache_assumed"] is True
    assert benchmark["carnator"]["collector_max_completed_step"] == 5
    assert benchmark["methods"]["carnator"]["step10_cache_assumed"] is True
    assert benchmark["methods"]["carnator"]["source_cache_total_s"] == 0.0

    carnator_rows = [
        json.loads(line)
        for line in (output_dir / "results" / "carnator.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert all(row["completed_steps"] == 5 for row in carnator_rows)
    assert carnator_rows[0]["step10_cache_assumed"] is True
    assert carnator_rows[1]["step10_cache_assumed"] is True
    source_dirs = [path for path in (output_dir / "carnator_sources").iterdir() if path.is_dir()]
    assert len(source_dirs) == 2
    assert all(not (source_dir / "step10.pt").exists() for source_dir in source_dirs)
    signatures = [json.loads((source_dir / "online_color_signature.json").read_text(encoding="utf-8")) for source_dir in source_dirs]
    assert all(signature["cache"]["assumption"] == "step10_cache_exists" for signature in signatures)
