from __future__ import annotations

import hashlib
import importlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol

from .types import CacheMatch, Request


_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", re.IGNORECASE)
_TEST_STOP_WORDS = frozenset({
    "a", "an", "and", "at", "by", "for", "from", "in", "of", "on", "the", "to", "with",
})


@dataclass(frozen=True)
class VectorRecord:
    uuid: str
    term: str
    vector: list[float]
    # Optional method-specific recovery metadata.  Existing manifests only
    # contain the first three fields and remain valid after this extension.
    metadata: dict[str, Any] | None = None
    source_request_index: int | None = None


class EntryLike(Protocol):
    uuid: str
    prompt: str


class TextEmbedder(Protocol):
    @property
    def signature(self) -> str: ...

    def encode(self, texts: list[str]) -> list[list[float]]: ...


class LexicalTestEmbedder:
    """Deterministic encoder used only by dry-run tests.

    It deliberately has a distinct signature from paper embedders so a cache
    populated by this encoder cannot be reused during a real benchmark.
    """

    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions

    @property
    def signature(self) -> str:
        return f"test-lexical-v1:{self.dimensions}"

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dimensions
            for token in _tokens(text):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                value = int.from_bytes(digest, "big")
                index = value % self.dimensions
                vector[index] += -1.0 if value & 1 else 1.0
            vectors.append(_normalize(vector))
        return vectors


class TransformersClipEmbedder:
    """CLIP text embedding with the projected, normalized text feature."""

    def __init__(self, model_id: str, device: str) -> None:
        self.model_id = model_id
        self.device = device
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._torch: Any | None = None

    @property
    def signature(self) -> str:
        return f"clip:{self.model_id}"

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import CLIPTextModelWithProjection, CLIPTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "CLIP retrieval requires torch and transformers; install the model environment first"
            ) from exc
        self._torch = torch
        self._tokenizer = CLIPTokenizer.from_pretrained(self.model_id)
        self._model = CLIPTextModelWithProjection.from_pretrained(self.model_id).to(self.device).eval()

    def encode(self, texts: list[str]) -> list[list[float]]:
        self._load()
        assert self._model is not None and self._tokenizer is not None and self._torch is not None
        encoded = self._tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with self._torch.no_grad():
            features = self._model(**encoded).text_embeds.float()
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return features.cpu().tolist()


class LongClipEmbedder:
    """Official LongCLIP module wrapper used by CHAI's entity index."""

    def __init__(self, checkpoint: str, device: str, module_name: str = "longclip") -> None:
        self.checkpoint = checkpoint
        self.device = device
        self.module_name = module_name
        self._module: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None

    @property
    def signature(self) -> str:
        return f"longclip:{self.checkpoint}"

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            module = importlib.import_module(self.module_name)
        except ImportError as exc:
            raise RuntimeError(
                "CHAI retrieval requires the official LongCLIP module. Add it to PYTHONPATH and set "
                "retrieval.chai.longclip_checkpoint."
            ) from exc
        self._torch = torch
        self._module = module
        self._model, _ = module.load(self.checkpoint, device=self.device)
        self._model.eval()

    def encode(self, texts: list[str]) -> list[list[float]]:
        self._load()
        assert self._module is not None and self._model is not None and self._torch is not None
        tokens = self._module.tokenize(texts).to(self.device)
        with self._torch.no_grad():
            features = self._model.encode_text(tokens).float()
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return features.cpu().tolist()


class EntityExtractor(Protocol):
    @property
    def signature(self) -> str: ...

    def extract(self, prompt: str) -> list[str]: ...


class SpacyEntityExtractor:
    def __init__(self, model: str) -> None:
        self.model = model
        self._nlp: Any | None = None

    @property
    def signature(self) -> str:
        return f"spacy:{self.model}"

    def _load(self) -> None:
        if self._nlp is not None:
            return
        try:
            import spacy
        except ImportError as exc:
            raise RuntimeError("CHAI retrieval requires spacy==3.8.11 and its English model") from exc
        try:
            self._nlp = spacy.load(self.model)
        except OSError as exc:
            raise RuntimeError(
                f"CHAI retrieval requires the spaCy model {self.model!r}; install it before running"
            ) from exc

    def extract(self, prompt: str) -> list[str]:
        self._load()
        assert self._nlp is not None
        # CHAI's reference implementation ignores the optional suffix used by
        # VidProM prompts and indexes only spaCy tokens tagged as NOUN. It
        # does not expand noun chunks or include PROPN tokens.
        cleaned_prompt = prompt.split("aesthetic score", 1)[0]
        doc = self._nlp(cleaned_prompt)
        return [token.text for token in doc if token.pos_ == "NOUN"]


class LexicalTestEntityExtractor:
    @property
    def signature(self) -> str:
        return "test-lexical-entities-v1"

    def extract(self, prompt: str) -> list[str]:
        return _unique_nonempty(token for token in _tokens(prompt) if token not in _TEST_STOP_WORDS)


class RetrievalStrategy:
    """Method-specific mapping between requests and persistent vector records."""

    kind = "similarity"
    eviction_policy = "lru"

    def __init__(self, embedder: TextEmbedder, threshold: float, *, vector_backend: str = "memory") -> None:
        if vector_backend not in {"memory", "faiss", "chroma"}:
            raise ValueError(f"unsupported retrieval vector backend: {vector_backend}")
        self.embedder = embedder
        self.threshold = float(threshold)
        self.vector_backend = vector_backend
        self._vector_store_root: Path | None = None

    @property
    def signature(self) -> str:
        return (
            f"{self.kind}|{self.embedder.signature}|threshold={self.threshold:.8f}"
            f"|vector_backend={self.vector_backend}"
        )

    def bind_vector_store(self, root: Path) -> None:
        """Bind an optional backend-specific persistent index to a cache namespace."""

        self._vector_store_root = root

    def sync_vector_store(self, records: Iterable[VectorRecord]) -> None:
        """Synchronize an external vector index from the JSON recovery manifest.

        Most strategies query the manifest directly.  Persistent backends
        override this method; `NamespaceCache` invokes it after additions,
        evictions, and process restarts.
        """

        del records

    def index_terms(self, prompt: str) -> list[str]:
        return [prompt]

    def query_terms(self, request: Request) -> list[str]:
        return [request.cache_prompt or request.prompt]

    def records_for(self, entry: EntryLike) -> list[VectorRecord]:
        terms = self.index_terms(entry.prompt)
        vectors = self.embedder.encode(terms)
        return [VectorRecord(entry.uuid, term, vector) for term, vector in zip(terms, vectors, strict=True)]

    def records_for_add(
        self,
        entry: EntryLike,
        existing_records: Iterable[VectorRecord],
    ) -> list[VectorRecord]:
        """Return the records accepted by the method's cache-write policy."""

        del existing_records
        return self.records_for(entry)

    def search(self, request: Request, records: Iterable[VectorRecord]) -> CacheMatch:
        query_terms = self.query_terms(request)
        if not query_terms:
            return CacheMatch()
        query_vectors = self.embedder.encode(query_terms)
        record_list = list(records)
        best = self._top1(query_terms, query_vectors, record_list)
        if best is None or best[0] < self.threshold:
            return CacheMatch()
        score, record, query_term = best
        metadata = {
            "retriever": self.kind,
            "embedding": self.embedder.signature,
            "query_term": query_term,
            "matched_term": record.term,
        }
        kind = "source_prompt" if request.cache_prompt and request.cache_prompt != request.prompt else self.kind
        return CacheMatch(True, record.uuid, None, kind, score, metadata)

    def _top1(
        self,
        query_terms: list[str],
        query_vectors: list[list[float]],
        records: list[VectorRecord],
    ) -> tuple[float, VectorRecord, str] | None:
        if not records:
            return None
        if self.vector_backend == "memory":
            return _memory_top1(query_terms, query_vectors, records)
        if self.vector_backend == "faiss":
            return _faiss_top1(query_terms, query_vectors, records)
        raise RuntimeError(
            f"{self.kind} uses the {self.vector_backend!r} backend, but does not implement its query path"
        )


class ChaiRetriever(RetrievalStrategy):
    kind = "chai_entity"

    def __init__(
        self,
        embedder: TextEmbedder,
        extractor: EntityExtractor,
        threshold: float,
        *,
        add_threshold: float = 0.95,
        vector_backend: str = "memory",
    ) -> None:
        super().__init__(embedder, threshold, vector_backend=vector_backend)
        if not -1.0 <= float(add_threshold) <= 1.0:
            raise ValueError("CHAI add threshold must be between -1 and 1")
        self.extractor = extractor
        self.add_threshold = float(add_threshold)
        self._faiss_index: Any | None = None
        self._faiss_record_ids: tuple[str, ...] = ()

    @property
    def signature(self) -> str:
        return (
            f"{self.kind}|{self.embedder.signature}|{self.extractor.signature}"
            f"|threshold={self.threshold:.8f}|add_threshold={self.add_threshold:.8f}"
            f"|vector_backend={self.vector_backend}"
        )

    def index_terms(self, prompt: str) -> list[str]:
        return self.extractor.extract(prompt)

    def query_terms(self, request: Request) -> list[str]:
        return self.extractor.extract(request.cache_prompt or request.prompt)

    def records_for_add(
        self,
        entry: EntryLike,
        existing_records: Iterable[VectorRecord],
    ) -> list[VectorRecord]:
        """Apply CHAI's similarity gate before adding a prompt's entity group.

        The reference implementation searches the index built before this
        batch and adds all nouns whose nearest existing entity is below
        ``sim_thres_for_add``. Delaying the append until every noun has been
        scored preserves that batch behavior for repeated nouns in one prompt.
        """

        candidates = list(existing_records)
        records = self.records_for(entry)
        accepted: list[VectorRecord] = []
        for record in records:
            nearest = self._top1([record.term], [record.vector], candidates)
            if nearest is None or nearest[0] < self.add_threshold:
                accepted.append(record)
        return accepted

    def sync_vector_store(self, records: Iterable[VectorRecord]) -> None:
        if self.vector_backend != "faiss":
            return
        record_list = list(records)
        record_ids = tuple(_record_id(index, record) for index, record in enumerate(record_list))
        if record_ids == self._faiss_record_ids:
            return
        self._faiss_index = _make_faiss_index(record_list) if record_list else None
        self._faiss_record_ids = record_ids

    def _top1(
        self,
        query_terms: list[str],
        query_vectors: list[list[float]],
        records: list[VectorRecord],
    ) -> tuple[float, VectorRecord, str] | None:
        if self.vector_backend != "faiss":
            return super()._top1(query_terms, query_vectors, records)
        record_ids = tuple(_record_id(index, record) for index, record in enumerate(records))
        if record_ids != self._faiss_record_ids:
            self.sync_vector_store(records)
        if self._faiss_index is None:
            return None
        return _faiss_index_top1(self._faiss_index, query_terms, query_vectors, records)


class PromptRetriever(RetrievalStrategy):
    """Shared whole-prompt CLIP/Chroma retrieval for NIRVANA and Chorus."""

    kind = "prompt"

    def __init__(
        self,
        embedder: TextEmbedder,
        threshold: float,
        *,
        result_kind: str,
        reuse_step: int | None = None,
        schedule: dict[str, int] | None = None,
        vector_backend: str = "chroma",
    ) -> None:
        super().__init__(embedder, threshold, vector_backend=vector_backend)
        self.result_kind = result_kind
        self.reuse_step = int(reuse_step) if reuse_step is not None else None
        self.schedule = {key: int(value) for key, value in (schedule or {}).items()}
        self._chroma_client: Any | None = None
        self._chroma_collection: Any | None = None
        if self.reuse_step is None:
            required = {"min_k1", "min_k2", "max_k1", "max_k2"}
            if required - self.schedule.keys():
                raise ValueError(f"prompt stage scheduler needs {sorted(required)}")
        elif self.reuse_step <= 0:
            raise ValueError("prompt reuse_step must be positive")

    @property
    def signature(self) -> str:
        return f"{self.kind}|{self.embedder.signature}|threshold={self.threshold:.8f}|vector_backend={self.vector_backend}"

    def bind_vector_store(self, root: Path) -> None:
        if self._vector_store_root == root:
            return
        super().bind_vector_store(root)
        self._chroma_client = None
        self._chroma_collection = None

    def sync_vector_store(self, records: Iterable[VectorRecord]) -> None:
        if self.vector_backend != "chroma":
            return
        records = list(records)
        collection = self._get_chroma_collection()
        expected_ids = [_record_id(index, record) for index, record in enumerate(records)]
        existing = collection.get(include=[])
        existing_ids = list(existing.get("ids", []))
        if len(existing_ids) == len(expected_ids) and set(existing_ids) == set(expected_ids):
            return
        if existing_ids:
            collection.delete(ids=existing_ids)
        if records:
            collection.add(
                ids=expected_ids,
                embeddings=[record.vector for record in records],
                documents=[record.term for record in records],
                metadatas=[{"uuid": record.uuid, "term": record.term} for record in records],
            )

    def _top1(
        self,
        query_terms: list[str],
        query_vectors: list[list[float]],
        records: list[VectorRecord],
    ) -> tuple[float, VectorRecord, str] | None:
        if self.vector_backend != "chroma":
            return super()._top1(query_terms, query_vectors, records)
        if not records:
            return None
        by_id = {_record_id(index, record): record for index, record in enumerate(records)}
        record_order = {record.uuid: index for index, record in enumerate(records)}
        result = self._get_chroma_collection().query(
            query_embeddings=query_vectors,
            n_results=len(records),
            include=["distances"],
        )
        best: tuple[float, VectorRecord, str] | None = None
        result_ids = result.get("ids", [])
        result_distances = result.get("distances", [])
        for query_term, query_vector, ids, _distances in zip(
            query_terms, query_vectors, result_ids, result_distances, strict=True
        ):
            # Chroma's HNSW result order is approximate. Rerank the complete
            # synchronized candidate set with the shared exact cosine rule so
            # NIRVANA and Chorus cannot diverge before their reuse policies run.
            if len(ids) != len(by_id):
                return _memory_top1(query_terms, query_vectors, records)
            for record_id in ids:
                record = by_id.get(record_id)
                if record is None:
                    return _memory_top1(query_terms, query_vectors, records)
                score = _cosine(query_vector, record.vector)
                candidate = (score, record, query_term)
                if best is None or candidate[0] > best[0] or (
                    candidate[0] == best[0]
                    and record_order[record.uuid] < record_order[best[1].uuid]
                ):
                    best = candidate
        return best

    def _get_chroma_collection(self):
        if self._chroma_collection is not None:
            return self._chroma_collection
        if self._vector_store_root is None:
            raise RuntimeError("Chorus ChromaDB index is not bound to a cache namespace")
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError(
                "Chorus retrieval requires chromadb; install requirements/retrieval-wan.txt"
            ) from exc
        path = self._vector_store_root / "chromadb"
        path.mkdir(parents=True, exist_ok=True)
        self._chroma_client = chromadb.PersistentClient(path=str(path))
        self._chroma_collection = self._chroma_client.get_or_create_collection(
            name="prompt_embeddings",
            metadata={"hnsw:space": "cosine"},
        )
        return self._chroma_collection

    def search(self, request: Request, records: Iterable[VectorRecord]) -> CacheMatch:
        match = super().search(request, records)
        if not match.hit or match.score is None:
            return match
        match.metadata["retriever"] = self.result_kind
        if self.reuse_step is not None:
            match.metadata["reuse_step"] = self.reuse_step
            match.metadata["selection"] = "fixed_reuse_step"
        else:
            ratio = min(1.0, max(0.0, (match.score - self.threshold) / (1.0 - self.threshold)))
            k1 = _interpolate(self.schedule["min_k1"], self.schedule["max_k1"], ratio)
            k2 = _interpolate(self.schedule["min_k2"], self.schedule["max_k2"], ratio)
            match.metadata["stage_boundaries"] = {"k1": k1, "k2": k2}
            match.metadata["stage_scheduler"] = "linear_similarity_approximation"
        return match


def make_nirvana_retriever(config: dict[str, Any], *, dry_run: bool) -> PromptRetriever:
    embedder = _make_embedder(config, dry_run=dry_run)
    return PromptRetriever(
        embedder,
        float(config.get("min_similarity", 0.65 if not dry_run else 0.20)),
        result_kind="nirvana_prompt",
        reuse_step=int(config.get("reuse_step", 10)),
        vector_backend=_vector_backend(config, dry_run=dry_run, default="chroma"),
    )


def make_chai_retriever(config: dict[str, Any], *, dry_run: bool) -> ChaiRetriever:
    threshold = float(config.get(
        "sim_thres_for_search",
        config.get("min_entity_similarity", 0.90 if not dry_run else 0.20),
    ))
    add_threshold = float(config.get("sim_thres_for_add", 0.95))
    if dry_run:
        extractor: EntityExtractor = LexicalTestEntityExtractor()
    else:
        extractor = SpacyEntityExtractor(str(config.get("spacy_model", "en_core_web_sm")))
    return ChaiRetriever(
        _make_embedder(config, dry_run=dry_run),
        extractor,
        threshold,
        add_threshold=add_threshold,
        vector_backend=_vector_backend(config, dry_run=dry_run, default="faiss"),
    )


def make_chorus_retriever(config: dict[str, Any], *, dry_run: bool) -> PromptRetriever:
    threshold = float(config.get("min_similarity", 0.65 if not dry_run else 0.20))
    schedule = config.get("stage_scheduler", {
        "min_k1": 10, "min_k2": 40, "max_k1": 30, "max_k2": 45,
    })
    return PromptRetriever(
        _make_embedder(config, dry_run=dry_run),
        threshold,
        result_kind="chorus_prompt",
        schedule=schedule,
        vector_backend=_vector_backend(config, dry_run=dry_run, default="chroma"),
    )


def make_carnator_retriever(config: dict[str, Any], *, dry_run: bool, retrieval_only: bool = False):
    """Build Carnator's independent two-stage retriever lazily.

    The implementation lives in ``carnator.retrieval`` so importing the
    common retrieval module never pulls in model-specific code.
    """

    from carnator.retrieval import make_carnator_retriever as _make

    return _make(config, dry_run=dry_run, retrieval_only=retrieval_only)


def _vector_backend(config: dict[str, Any], *, dry_run: bool, default: str) -> str:
    # Synthetic tests should exercise the protocol without importing optional
    # native/vector-service dependencies.
    if dry_run:
        return "memory"
    return str(config.get("vector_backend", default))


def _make_embedder(config: dict[str, Any], *, dry_run: bool) -> TextEmbedder:
    backend = str(config.get("backend", "test_lexical" if dry_run else "clip"))
    if dry_run:
        return LexicalTestEmbedder(int(config.get("test_dimensions", 256)))
    if backend == "test_lexical":
        raise RuntimeError("test_lexical is limited to dry-run retrieval tests")
    if backend == "clip":
        return TransformersClipEmbedder(
            str(config.get("model_id", "openai/clip-vit-large-patch14")),
            str(config.get("device", "cuda")),
        )
    if backend == "longclip":
        checkpoint = config.get("longclip_checkpoint")
        if not checkpoint:
            raise RuntimeError("LongCLIP retrieval requires retrieval.chai.longclip_checkpoint")
        return LongClipEmbedder(str(checkpoint), str(config.get("device", "cuda")),
                                str(config.get("longclip_module", "longclip")))
    raise ValueError(f"unsupported retrieval embedding backend: {backend}")


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text)]


def _normalize_term(value: str) -> str:
    return " ".join(_tokens(value))


def _unique_nonempty(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError(f"embedding dimensions differ: {len(left)} != {len(right)}")
    return sum(a * b for a, b in zip(left, right, strict=True))


def _memory_top1(
    query_terms: list[str],
    query_vectors: list[list[float]],
    records: list[VectorRecord],
) -> tuple[float, VectorRecord, str] | None:
    best: tuple[float, VectorRecord, str] | None = None
    for query_term, query_vector in zip(query_terms, query_vectors, strict=True):
        for record in records:
            score = _cosine(query_vector, record.vector)
            if best is None or score > best[0]:
                best = (score, record, query_term)
    return best


def _faiss_top1(
    query_terms: list[str],
    query_vectors: list[list[float]],
    records: list[VectorRecord],
) -> tuple[float, VectorRecord, str] | None:
    """Paper-aligned FAISS exact inner-product search over normalized vectors."""

    return _faiss_index_top1(_make_faiss_index(records), query_terms, query_vectors, records)


def _make_faiss_index(records: list[VectorRecord]):
    if not records:
        return None

    try:
        import faiss
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "CHAI retrieval requires faiss-cpu==1.13.0; install requirements/retrieval-common.txt"
        ) from exc
    dimensions = len(records[0].vector)
    if not dimensions or any(len(record.vector) != dimensions for record in records):
        raise ValueError("FAISS index received inconsistent embedding dimensions")
    index = faiss.IndexFlatIP(dimensions)
    index.add(np.asarray([record.vector for record in records], dtype="float32"))
    return index


def _faiss_index_top1(
    index: Any,
    query_terms: list[str],
    query_vectors: list[list[float]],
    records: list[VectorRecord],
) -> tuple[float, VectorRecord, str] | None:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("CHAI retrieval requires numpy") from exc
    scores, indices = index.search(np.asarray(query_vectors, dtype="float32"), 1)
    best: tuple[float, VectorRecord, str] | None = None
    for query_term, score, index_value in zip(query_terms, scores, indices, strict=True):
        record_index = int(index_value[0])
        if record_index < 0:
            continue
        candidate = (float(score[0]), records[record_index], query_term)
        if best is None or candidate[0] > best[0]:
            best = candidate
    return best


def _record_id(index: int, record: VectorRecord) -> str:
    # Chroma IDs only need to be stable within the recovery manifest. Hashing
    # avoids assumptions about UUID formats or entity text characters.
    value = f"{index}\0{record.uuid}\0{record.term}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _interpolate(lower: int, upper: int, ratio: float) -> int:
    return int(round(lower + (upper - lower) * ratio))
