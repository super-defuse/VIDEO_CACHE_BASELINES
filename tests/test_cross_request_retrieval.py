from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adapters import make_adapter
from core.cache import NamespaceCache
from core.retrieval import (
    ChaiRetriever,
    LexicalTestEmbedder,
    LexicalTestEntityExtractor,
    PromptRetriever,
    SpacyEntityExtractor,
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
                "min_similarity": 0.20,
                "reuse_step": 10,
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


def test_nirvana_selects_the_fixed_restart_latent():
    method = make_method("nirvana_vid", _config())
    selected = method.select_cached_latent(
        {10: b"step-10"},
        CacheMatch(hit=True, metadata={"reuse_step": 10}),
    )
    assert selected == b"step-10"


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


def test_chai_spacy_extractor_matches_reference_token_filter():
    class Token:
        def __init__(self, text: str, pos: str) -> None:
            self.text = text
            self.pos_ = pos

    class FakeNlp:
        def __init__(self) -> None:
            self.seen_prompt = None

        def __call__(self, prompt: str):
            self.seen_prompt = prompt
            return [
                Token("Shanghai", "PROPN"),
                Token("bund", "NOUN"),
                Token("steady", "ADJ"),
                Token("perspective", "NOUN"),
            ]

    extractor = SpacyEntityExtractor("fake")
    nlp = FakeNlp()
    extractor._nlp = nlp
    assert extractor.extract("Shanghai bund aesthetic score 0.9") == ["bund", "perspective"]
    assert nlp.seen_prompt == "Shanghai bund "

    carnator_extractor = SpacyEntityExtractor("fake", allowed_pos=("NOUN", "PROPN"))
    carnator_extractor._nlp = nlp
    assert carnator_extractor.extract("Shanghai bund aesthetic score 0.9") == [
        "Shanghai", "bund", "perspective",
    ]
    assert carnator_extractor.signature == "spacy:fake|pos=NOUN,PROPN"


def test_chai_write_threshold_deduplicates_entity_batch():
    class Embedder:
        @property
        def signature(self) -> str:
            return "fixed"

        def encode(self, texts: list[str]) -> list[list[float]]:
            vectors = {
                "red": [1.0, 0.0],
                "blue": [0.0, 1.0],
                "green": [-1.0, 0.0],
            }
            return [vectors[text] for text in texts]

    class Extractor:
        @property
        def signature(self) -> str:
            return "fixed-entities"

        def extract(self, prompt: str) -> list[str]:
            return prompt.split()

    strategy = ChaiRetriever(Embedder(), Extractor(), 0.90, add_threshold=0.95)
    source = Request(0, "source", "red blue", 1)
    target = Request(1, "target", "red green", 1)
    source_records = strategy.records_for_add(source, [])
    target_records = strategy.records_for_add(target, source_records)
    assert [record.term for record in source_records] == ["red", "blue"]
    assert [record.term for record in target_records] == ["green"]


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
        strategy=PromptRetriever(
            LexicalTestEmbedder(), 0.10, result_kind="chorus_prompt", schedule=schedule, vector_backend="chroma"
        ),
    )
    restarted = NamespaceCache(tmp_path, "chorus-chroma")
    match = restarted.find(
        Request(1, "target", "a red train", 1),
        PromptRetriever(
            LexicalTestEmbedder(), 0.10, result_kind="chorus_prompt", schedule=schedule, vector_backend="chroma"
        ),
    )
    assert match.hit
    assert match.source_uuid == source.uuid
    assert (restarted.vector_store / "chromadb").is_dir()
