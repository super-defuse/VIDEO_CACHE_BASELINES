from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adapters import make_adapter
from core.cache import NamespaceCache
from core.retrieval import (
    CacheSelectorObservation,
    CacheSelectorProfile,
    ChaiRetriever,
    ChorusRetriever,
    LexicalTestEmbedder,
    LexicalTestEntityExtractor,
)
from core import CacheMatch, Request, Runner
from core.registry import make_method


WORKLOAD = ROOT / "datasets" / "vbench_cross_request_prompts.csv"


def _config() -> dict:
    return {
        "dry_run": True,
        "seed": 3,
        "cache_root": "cache",
        "max_cache_bytes": 1024 * 1024,
        "nirvana_vid": {"reuse_start_step": 10, "cache_steps": [5, 10, 15, 20, 25]},
        "retrieval": {
            "nirvana_vid": {
                "backend": "test_lexical",
                "similarity_to_step": [
                    {"step": 5, "min_similarity": 0.10},
                    {"step": 10, "min_similarity": 0.20},
                    {"step": 15, "min_similarity": 0.30},
                    {"step": 20, "min_similarity": 0.40},
                    {"step": 25, "min_similarity": 0.50},
                ],
            },
            "chai": {"backend": "test_lexical", "min_entity_similarity": 0.20},
            "chorus": {
                "backend": "test_lexical",
                "min_similarity": 0.20,
                "stage_scheduler": {"min_k1": 10, "min_k2": 40, "max_k1": 30, "max_k2": 45},
            },
        },
    }


def _requests(limit: int = 18) -> list[Request]:
    with WORKLOAD.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))[:limit]
    return [Request(int(row["request_index"]), row["uuid"], row["prompt"], 11 + index)
            for index, row in enumerate(rows)]


def _run_workload(tmp_path: Path, method_name: str):
    config = _config()
    adapter = make_adapter("synthetic", config)
    method = make_method(method_name, config)
    runner = Runner(adapter, method, model="synthetic", output_dir=tmp_path / method_name,
                    cache_root=tmp_path / "cache")
    return runner, [runner.process(request) for request in _requests()]


def test_nirvana_selects_the_profiled_restart_latent():
    method = make_method("nirvana_vid", _config())
    selected = method.select_cached_latent(
        {5: b"step-5", 15: b"step-15"},
        CacheMatch(hit=True, metadata={"reuse_step": 15}),
    )
    assert selected == b"step-15"


def test_vbench_workload_retrieval_reaches_fast_generation_for_all_methods(tmp_path):
    for method_name, retriever in (
        ("nirvana_vid", "nirvana_prompt"),
        ("chai", "chai_entity"),
        ("chorus", "chorus_prompt"),
    ):
        runner, results = _run_workload(tmp_path, method_name)
        hits = [result for result in results if result.cache_match.hit]
        assert results[0].status == "ok"
        assert not results[0].cache_match.hit
        assert hits, method_name
        assert all(result.status == "ok" for result in results)
        assert all(result.executed_steps == 8 for result in hits)
        assert all(result.cache_match.metadata["retriever"] == retriever for result in hits)
        assert (runner.cache.retrieval_path).is_file()


def test_similarity_index_survives_a_runner_restart(tmp_path):
    first_runner, first_results = _run_workload(tmp_path, "chai")
    assert any(result.cache_match.hit for result in first_results)

    config = _config()
    adapter = make_adapter("synthetic", config)
    method = make_method("chai", config)
    restarted = Runner(adapter, method, model="synthetic", output_dir=tmp_path / "chai-restarted",
                       cache_root=tmp_path / "cache")
    result = restarted.process(_requests()[1])
    assert result.status == "ok"
    assert result.cache_match.hit
    assert result.cache_match.source_uuid == _requests()[0].uuid
    assert first_runner.cache.retrieval_path.read_text() == restarted.cache.retrieval_path.read_text()


def _save_bytes(value, path: Path) -> None:
    path.write_bytes(value)


def test_chai_uses_faiss_inner_product_index_when_installed(tmp_path):
    if importlib.util.find_spec("faiss") is None:
        return
    strategy = ChaiRetriever(
        LexicalTestEmbedder(), LexicalTestEntityExtractor(), 0.10, vector_backend="faiss"
    )
    cache = NamespaceCache(tmp_path, "chai-faiss")
    source = Request(0, "source", "a red train at night", 1)
    cache.put(source, b"latent", 5, _save_bytes, strategy=strategy)
    match = cache.find(Request(1, "target", "a red train", 1), strategy)
    assert match.hit
    assert match.source_uuid == source.uuid


def test_chorus_uses_persistent_chromadb_index_when_installed(tmp_path):
    if importlib.util.find_spec("chromadb") is None:
        return
    schedule = {"min_k1": 10, "min_k2": 40, "max_k1": 30, "max_k2": 45}
    source = Request(0, "source", "a red train at night", 1)
    first = NamespaceCache(tmp_path, "chorus-chroma")
    first.put(
        source,
        b"latent",
        5,
        _save_bytes,
        strategy=ChorusRetriever(LexicalTestEmbedder(), 0.10, schedule, vector_backend="chroma"),
    )
    restarted = NamespaceCache(tmp_path, "chorus-chroma")
    match = restarted.find(
        Request(1, "target", "a red train", 1),
        ChorusRetriever(LexicalTestEmbedder(), 0.10, schedule, vector_backend="chroma"),
    )
    assert match.hit
    assert match.source_uuid == source.uuid
    assert (restarted.vector_store / "chromadb").is_dir()


def test_nirvana_cache_selector_profile_uses_quality_safe_thresholds(tmp_path):
    profile = CacheSelectorProfile.from_observations(
        [
            CacheSelectorObservation(0.20, 5, 0.95),
            CacheSelectorObservation(0.40, 5, 0.92),
            CacheSelectorObservation(0.20, 10, 0.83),
            CacheSelectorObservation(0.40, 10, 0.91),
            CacheSelectorObservation(0.60, 10, 0.94),
        ],
        quality_threshold=0.90,
    )
    assert profile.thresholds == {5: 0.20, 10: 0.40}
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile.to_payload()), encoding="utf-8")
    assert CacheSelectorProfile.load(path).thresholds == profile.thresholds
