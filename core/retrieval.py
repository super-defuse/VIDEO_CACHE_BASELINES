from __future__ import annotations

import hashlib
import importlib
import json
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


@dataclass(frozen=True)
class CacheSelectorObservation:
    """A measured quality result for one NIRVANA restart step."""

    similarity: float
    reuse_step: int
    quality: float


@dataclass(frozen=True)
class CacheSelectorProfile:
    """Offline NIRVANA similarity-to-restart-step profile.

    The profile contains measured quality observations condensed into the
    minimum similarity accepted for each cached restart step.  It does not
    select model parameters or change them at runtime.
    """

    thresholds: dict[int, float]
    quality_metric: str = "CLIPScore"
    quality_threshold: float = 0.90
    source: str = "offline_observations"

    def __post_init__(self) -> None:
        if not self.thresholds:
            raise ValueError("CacheSelector profile needs at least one restart step")
        for step, score in self.thresholds.items():
            if int(step) <= 0 or not -1.0 <= float(score) <= 1.0:
                raise ValueError(f"invalid CacheSelector threshold: step={step}, similarity={score}")

    @classmethod
    def from_observations(
        cls,
        observations: Iterable[CacheSelectorObservation],
        *,
        quality_threshold: float = 0.90,
        quality_metric: str = "CLIPScore",
    ) -> "CacheSelectorProfile":
        """Derive the first safe similarity threshold for every measured step.

        For a candidate threshold, every observed reuse at or above that
        similarity must meet the requested quality.  This is the offline
        CacheSelector construction described by NIRVANA; it intentionally
        does not tune a threshold against a workload hit-rate objective.
        """

        grouped: dict[int, list[CacheSelectorObservation]] = {}
        for observation in observations:
            grouped.setdefault(int(observation.reuse_step), []).append(observation)
        thresholds: dict[int, float] = {}
        for step, samples in grouped.items():
            for candidate in sorted({float(sample.similarity) for sample in samples}):
                selected = [sample for sample in samples if sample.similarity >= candidate]
                if selected and all(sample.quality >= quality_threshold for sample in selected):
                    thresholds[step] = candidate
                    break
        if not thresholds:
            raise ValueError("no restart step satisfies the requested CacheSelector quality threshold")
        return cls(dict(sorted(thresholds.items())), quality_metric, quality_threshold)

    @classmethod
    def load(cls, path: str | Path) -> "CacheSelectorProfile":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        raw_steps = payload.get("steps", payload.get("thresholds", []))
        if isinstance(raw_steps, dict):
            thresholds = {int(step): float(score) for step, score in raw_steps.items()}
        else:
            thresholds = {
                int(item["step"]): float(item["min_similarity"])
                for item in raw_steps
            }
        return cls(
            dict(sorted(thresholds.items())),
            str(payload.get("quality_metric", "CLIPScore")),
            float(payload.get("quality_threshold", 0.90)),
            str(payload.get("source", "offline_observations")),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "quality_metric": self.quality_metric,
            "quality_threshold": self.quality_threshold,
            "source": self.source,
            "steps": [
                {"step": step, "min_similarity": score}
                for step, score in sorted(self.thresholds.items())
            ],
        }


class MatchPredictor(Protocol):
    @property
    def signature(self) -> str: ...

    def accepts(self, vector: list[float]) -> bool: ...


class SgdOneClassMatchPredictor:
    """Optional persisted `SGDOneClassSVM` gate used by NIRVANA."""

    def __init__(self, model: Any, embedding_signature: str, source_path: str | None = None) -> None:
        self.model = model
        self.embedding_signature = embedding_signature
        self.source_path = source_path

    @property
    def signature(self) -> str:
        return f"sgd-one-class-svm:{self.embedding_signature}"

    @classmethod
    def load(cls, path: str | Path, embedding_signature: str) -> "SgdOneClassMatchPredictor":
        try:
            import joblib
        except ImportError as exc:
            raise RuntimeError("NIRVANA match predictor requires scikit-learn and joblib") from exc
        payload = joblib.load(path)
        saved_signature = str(payload["embedding_signature"])
        if saved_signature != embedding_signature:
            raise RuntimeError(
                "NIRVANA match predictor embedding differs from the active retriever: "
                f"{saved_signature!r} != {embedding_signature!r}"
            )
        return cls(payload["model"], saved_signature, str(path))

    @classmethod
    def fit(cls, vectors: list[list[float]], embedding_signature: str, *, nu: float = 0.05) -> "SgdOneClassMatchPredictor":
        if not vectors:
            raise ValueError("NIRVANA match predictor needs at least one inlier embedding")
        try:
            import numpy as np
            from sklearn.linear_model import SGDOneClassSVM
        except ImportError as exc:
            raise RuntimeError("NIRVANA match predictor requires numpy and scikit-learn") from exc
        model = SGDOneClassSVM(nu=nu, random_state=0, shuffle=False).fit(np.asarray(vectors, dtype="float32"))
        return cls(model, embedding_signature)

    def save(self, path: str | Path) -> None:
        try:
            import joblib
        except ImportError as exc:
            raise RuntimeError("NIRVANA match predictor requires scikit-learn and joblib") from exc
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"embedding_signature": self.embedding_signature, "model": self.model}, output)

    def accepts(self, vector: list[float]) -> bool:
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("NIRVANA match predictor requires numpy") from exc
        return bool(self.model.predict(np.asarray([vector], dtype="float32"))[0] == 1)


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
        doc = self._nlp(prompt)
        entities: list[str] = []
        for chunk in doc.noun_chunks:
            if chunk.root.pos_ in {"NOUN", "PROPN"}:
                entities.append(_normalize_term(chunk.text))
        entities.extend(_normalize_term(token.text) for token in doc if token.pos_ in {"NOUN", "PROPN"})
        return _unique_nonempty(entities)


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


class NirvanaRetriever(RetrievalStrategy):
    kind = "nirvana_prompt"
    eviction_policy = "lcbfu"

    def __init__(
        self,
        embedder: TextEmbedder,
        thresholds: dict[int, float],
        *,
        match_predictor: MatchPredictor | None = None,
    ) -> None:
        if not thresholds:
            raise ValueError("NIRVANA retrieval needs at least one similarity-to-step threshold")
        self.thresholds = dict(sorted((int(step), float(score)) for step, score in thresholds.items()))
        self.match_predictor = match_predictor
        super().__init__(embedder, min(self.thresholds.values()))

    @property
    def signature(self) -> str:
        schedule = ",".join(f"{step}:{score:.8f}" for step, score in self.thresholds.items())
        predictor = self.match_predictor.signature if self.match_predictor else "disabled"
        return f"{self.kind}|{self.embedder.signature}|steps={schedule}|match_predictor={predictor}"

    def search(self, request: Request, records: Iterable[VectorRecord]) -> CacheMatch:
        if self.match_predictor is not None:
            query_term = request.cache_prompt or request.prompt
            query_vector = self.embedder.encode([query_term])[0]
            if not self.match_predictor.accepts(query_vector):
                return CacheMatch(
                    kind="predicted_miss",
                    metadata={
                        "retriever": self.kind,
                        "embedding": self.embedder.signature,
                        "match_predictor": self.match_predictor.signature,
                    },
                )
        match = super().search(request, records)
        if not match.hit or match.score is None:
            return match
        eligible = [step for step, threshold in self.thresholds.items() if match.score >= threshold]
        if not eligible:
            return CacheMatch()
        match.metadata["reuse_step"] = max(eligible)
        match.metadata["selection"] = "largest-profiled-step-at-score"
        return match


class ChaiRetriever(RetrievalStrategy):
    kind = "chai_entity"

    def __init__(
        self,
        embedder: TextEmbedder,
        extractor: EntityExtractor,
        threshold: float,
        *,
        vector_backend: str = "memory",
    ) -> None:
        super().__init__(embedder, threshold, vector_backend=vector_backend)
        self.extractor = extractor
        self._faiss_index: Any | None = None
        self._faiss_record_ids: tuple[str, ...] = ()

    @property
    def signature(self) -> str:
        return (
            f"{self.kind}|{self.embedder.signature}|{self.extractor.signature}"
            f"|threshold={self.threshold:.8f}|vector_backend={self.vector_backend}"
        )

    def index_terms(self, prompt: str) -> list[str]:
        return self.extractor.extract(prompt)

    def query_terms(self, request: Request) -> list[str]:
        return self.extractor.extract(request.cache_prompt or request.prompt)

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


class ChorusRetriever(RetrievalStrategy):
    kind = "chorus_prompt"

    def __init__(
        self,
        embedder: TextEmbedder,
        threshold: float,
        schedule: dict[str, int],
        *,
        vector_backend: str = "memory",
    ) -> None:
        super().__init__(embedder, threshold, vector_backend=vector_backend)
        self.schedule = {key: int(value) for key, value in schedule.items()}
        self._chroma_client: Any | None = None
        self._chroma_collection: Any | None = None
        required = {"min_k1", "min_k2", "max_k1", "max_k2"}
        if required - self.schedule.keys():
            raise ValueError(f"Chorus stage scheduler needs {sorted(required)}")

    @property
    def signature(self) -> str:
        values = ",".join(f"{key}:{value}" for key, value in sorted(self.schedule.items()))
        return (
            f"{self.kind}|{self.embedder.signature}|threshold={self.threshold:.8f}|{values}"
            f"|vector_backend={self.vector_backend}"
        )

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
        result = self._get_chroma_collection().query(
            query_embeddings=query_vectors,
            n_results=1,
            include=["distances", "metadatas"],
        )
        best: tuple[float, VectorRecord, str] | None = None
        result_ids = result.get("ids", [])
        result_distances = result.get("distances", [])
        for query_term, ids, distances in zip(query_terms, result_ids, result_distances, strict=True):
            if not ids:
                continue
            record = by_id.get(ids[0])
            if record is None:
                continue
            # ChromaDB's cosine HNSW distance is 1 - cosine similarity.
            score = 1.0 - float(distances[0])
            if best is None or score > best[0]:
                best = (score, record, query_term)
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
            name="chorus_prompt_embeddings",
            metadata={"hnsw:space": "cosine"},
        )
        return self._chroma_collection

    def search(self, request: Request, records: Iterable[VectorRecord]) -> CacheMatch:
        match = super().search(request, records)
        if not match.hit or match.score is None:
            return match
        ratio = min(1.0, max(0.0, (match.score - self.threshold) / (1.0 - self.threshold)))
        k1 = _interpolate(self.schedule["min_k1"], self.schedule["max_k1"], ratio)
        k2 = _interpolate(self.schedule["min_k2"], self.schedule["max_k2"], ratio)
        match.metadata["stage_boundaries"] = {"k1": k1, "k2": k2}
        match.metadata["stage_scheduler"] = "linear_similarity_approximation"
        return match


def make_nirvana_retriever(config: dict[str, Any], *, dry_run: bool) -> NirvanaRetriever:
    embedder = _make_embedder(config, dry_run=dry_run)
    profile_path = config.get("cache_selector_profile")
    if profile_path:
        thresholds = CacheSelectorProfile.load(str(profile_path)).thresholds
    else:
        thresholds = {
            int(item["step"]): float(item["min_similarity"])
            for item in config.get("similarity_to_step", [])
        }
    if not thresholds:
        thresholds = {5: 0.20, 10: 0.35, 15: 0.50, 20: 0.65, 25: 0.80}
    predictor_config = config.get("match_predictor", {})
    predictor: MatchPredictor | None = None
    if not dry_run and predictor_config.get("enabled", False):
        model_path = predictor_config.get("model_path")
        if not model_path:
            raise RuntimeError("enabled NIRVANA match predictor needs retrieval.nirvana_vid.match_predictor.model_path")
        predictor = SgdOneClassMatchPredictor.load(str(model_path), embedder.signature)
    return NirvanaRetriever(embedder, thresholds, match_predictor=predictor)


def make_chai_retriever(config: dict[str, Any], *, dry_run: bool) -> ChaiRetriever:
    threshold = float(config.get("min_entity_similarity", 0.55 if not dry_run else 0.20))
    if dry_run:
        extractor: EntityExtractor = LexicalTestEntityExtractor()
    else:
        extractor = SpacyEntityExtractor(str(config.get("spacy_model", "en_core_web_sm")))
    return ChaiRetriever(
        _make_embedder(config, dry_run=dry_run),
        extractor,
        threshold,
        vector_backend=_vector_backend(config, dry_run=dry_run, default="faiss"),
    )


def make_chorus_retriever(config: dict[str, Any], *, dry_run: bool) -> ChorusRetriever:
    threshold = float(config.get("min_similarity", 0.65 if not dry_run else 0.20))
    schedule = config.get("stage_scheduler", {
        "min_k1": 10, "min_k2": 40, "max_k1": 30, "max_k2": 45,
    })
    return ChorusRetriever(
        _make_embedder(config, dry_run=dry_run),
        threshold,
        schedule,
        vector_backend=_vector_backend(config, dry_run=dry_run, default="chroma"),
    )


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
