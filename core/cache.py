from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .retrieval import RetrievalStrategy, VectorRecord
from .types import CacheMatch, Request


@dataclass
class CacheEntry:
    uuid: str
    request_index: int
    prompt: str
    latent_path: str
    latent_step: int
    bytes: int
    accessed_at: float
    hit_count: int = 0
    created_at: float = 0.0


class NamespaceCache:
    """Persistent cache with a method-specific namespace."""

    def __init__(self, root: Path, namespace: str, max_bytes: int = 5 * 1024**3) -> None:
        self.root = root / namespace
        self.entries = self.root / "entries"
        self.index_path = self.root / "index.json"
        self.retrieval_path = self.root / "retrieval.json"
        # `retrieval.json` remains the portable recovery manifest. Backends
        # such as ChromaDB keep their persistent query index below this path.
        self.vector_store = self.root / "vector_store"
        self.max_bytes = max_bytes
        self.entries.mkdir(parents=True, exist_ok=True)
        self._entries: dict[str, CacheEntry] = self._load()
        self._retrieval_signature: str | None
        self._records: list[VectorRecord]
        self._retrieval_signature, self._records = self._load_retrieval()

    def _load(self) -> dict[str, CacheEntry]:
        if not self.index_path.is_file():
            return {}
        entries: dict[str, CacheEntry] = {}
        for key, payload in json.loads(self.index_path.read_text()).items():
            payload.setdefault("hit_count", 0)
            payload.setdefault("created_at", payload.get("accessed_at", 0.0))
            entries[key] = CacheEntry(**payload)
        return entries

    def _save(self) -> None:
        self.index_path.write_text(json.dumps({k: asdict(v) for k, v in self._entries.items()}, indent=2) + "\n")

    def _load_retrieval(self) -> tuple[str | None, list[VectorRecord]]:
        if not self.retrieval_path.is_file():
            return None, []
        try:
            payload = json.loads(self.retrieval_path.read_text())
            records = [VectorRecord(**record) for record in payload.get("records", [])]
        except (OSError, TypeError, ValueError):
            return None, []
        return payload.get("signature"), records

    def _save_retrieval(self) -> None:
        payload = {
            "signature": self._retrieval_signature,
            "records": [asdict(record) for record in self._records],
        }
        self.retrieval_path.write_text(json.dumps(payload, separators=(",", ":")) + "\n")

    def _ensure_retrieval_index(self, strategy: RetrievalStrategy) -> None:
        strategy.bind_vector_store(self.vector_store)
        if self._retrieval_signature != strategy.signature:
            self._records = []
            for entry in self._entries.values():
                if Path(entry.latent_path).is_file():
                    self._records.extend(strategy.records_for(entry))
            self._retrieval_signature = strategy.signature
            self._save_retrieval()
        strategy.sync_vector_store(self._records)

    @property
    def bytes_used(self) -> int:
        return sum(e.bytes for e in self._entries.values())

    @staticmethod
    def _payload_bytes(path: Path) -> int:
        if path.is_dir():
            return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
        return path.stat().st_size if path.is_file() else 0

    @staticmethod
    def _remove_payload(path: Path) -> None:
        path.unlink(missing_ok=True)
        # Chorus stores its per-step tensors beside the manifest. Other
        # methods have no sidecar and are unaffected by this cleanup.
        sidecar = path.with_suffix(".steps")
        if sidecar.is_dir():
            shutil.rmtree(sidecar)
        source_video = path.with_suffix(".source.mp4")
        source_video.unlink(missing_ok=True)

    def clear(self) -> None:
        shutil.rmtree(self.entries, ignore_errors=True)
        shutil.rmtree(self.vector_store, ignore_errors=True)
        self.entries.mkdir(parents=True, exist_ok=True)
        self._entries.clear()
        self._retrieval_signature = None
        self._records.clear()
        self._save()
        self._save_retrieval()

    def find(self, request: Request, strategy: RetrievalStrategy | None = None) -> CacheMatch:
        if strategy is not None:
            self._ensure_retrieval_index(strategy)
            valid_uuids = {entry.uuid for entry in self._entries.values() if Path(entry.latent_path).is_file()}
            valid_records = [record for record in self._records if record.uuid in valid_uuids]
            # A missing payload outside of this process must not remain a
            # Chroma candidate. Memory/FAISS backends intentionally no-op.
            strategy.sync_vector_store(valid_records)
            match = strategy.search(request, valid_records)
            if match.hit:
                entry = self._entries.get(match.source_uuid or "")
                if entry is None:
                    return CacheMatch()
                entry.accessed_at = time.time()
                entry.hit_count += 1
                self._save()
                match.source_request_index = entry.request_index
            return match
        lookup_prompt = request.cache_prompt or request.prompt
        exact = next((e for e in self._entries.values() if e.prompt == lookup_prompt), None)
        if exact is None or not Path(exact.latent_path).is_file():
            return CacheMatch()
        exact.accessed_at = time.time()
        exact.hit_count += 1
        self._save()
        kind = "source_prompt" if request.cache_prompt and request.cache_prompt != request.prompt else "prompt"
        return CacheMatch(True, exact.uuid, exact.request_index, kind, 1.0)

    def load(self, match: CacheMatch) -> tuple[object, CacheEntry] | None:
        if not match.hit or not match.source_uuid:
            return None
        entry = self._entries.get(match.source_uuid)
        if entry is None:
            return None
        return Path(entry.latent_path), entry

    def put(self, request: Request, latent: object, latent_step: int, save_latent,
            *, strategy: RetrievalStrategy | None = None, eviction_policy: str = "lru") -> CacheEntry:
        path = self.entries / f"{request.uuid}.latent"
        temporary = path.with_suffix(".tmp")
        save_latent(latent, temporary)
        temporary.replace(path)
        now = time.time()
        entry = CacheEntry(request.uuid, request.request_index, request.prompt, str(path), latent_step,
                           self._payload_bytes(path) + self._payload_bytes(path.with_suffix(".steps")), now,
                           created_at=now)
        self._entries[request.uuid] = entry
        if strategy is not None:
            self._ensure_retrieval_index(strategy)
            self._records = [record for record in self._records if record.uuid != entry.uuid]
            self._records.extend(strategy.records_for(entry))
        while self.bytes_used > self.max_bytes and self._entries:
            victim = self._eviction_victim(eviction_policy)
            self._remove_payload(Path(victim.latent_path))
            self._entries.pop(victim.uuid, None)
            self._records = [record for record in self._records if record.uuid != victim.uuid]
        self._save()
        if strategy is not None:
            self._save_retrieval()
            strategy.sync_vector_store(self._records)
        return entry

    def _eviction_victim(self, policy: str) -> CacheEntry:
        if policy == "lru":
            return min(self._entries.values(), key=lambda entry: entry.accessed_at)
        if policy == "lcbfu":
            # NIRVANA's LCBFU keeps entries that have both frequent reuse and
            # a large restart step. The paper does not disclose its exact
            # scoring formula; this product is the explicit local proxy.
            return min(self._entries.values(),
                       key=lambda entry: (entry.hit_count * max(entry.latent_step, 1), entry.accessed_at))
        raise ValueError(f"unsupported cache eviction policy: {policy}")
