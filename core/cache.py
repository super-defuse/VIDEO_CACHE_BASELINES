from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

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


class NamespaceCache:
    """Persistent cache with a method-specific namespace."""

    def __init__(self, root: Path, namespace: str, max_bytes: int = 5 * 1024**3) -> None:
        self.root = root / namespace
        self.entries = self.root / "entries"
        self.index_path = self.root / "index.json"
        self.max_bytes = max_bytes
        self.entries.mkdir(parents=True, exist_ok=True)
        self._entries: dict[str, CacheEntry] = self._load()

    def _load(self) -> dict[str, CacheEntry]:
        if not self.index_path.is_file():
            return {}
        return {k: CacheEntry(**v) for k, v in json.loads(self.index_path.read_text()).items()}

    def _save(self) -> None:
        self.index_path.write_text(json.dumps({k: asdict(v) for k, v in self._entries.items()}, indent=2) + "\n")

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
        self.entries.mkdir(parents=True, exist_ok=True)
        self._entries.clear()
        self._save()

    def find(self, request: Request) -> CacheMatch:
        lookup_prompt = request.cache_prompt or request.prompt
        exact = next((e for e in self._entries.values() if e.prompt == lookup_prompt), None)
        if exact is None or not Path(exact.latent_path).is_file():
            return CacheMatch()
        exact.accessed_at = time.time()
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

    def put(self, request: Request, latent: object, latent_step: int, save_latent) -> CacheEntry:
        path = self.entries / f"{request.uuid}.latent"
        temporary = path.with_suffix(".tmp")
        save_latent(latent, temporary)
        temporary.replace(path)
        entry = CacheEntry(request.uuid, request.request_index, request.prompt, str(path), latent_step,
                           self._payload_bytes(path) + self._payload_bytes(path.with_suffix(".steps")), time.time())
        self._entries[request.uuid] = entry
        while self.bytes_used > self.max_bytes and self._entries:
            victim = min(self._entries.values(), key=lambda e: e.accessed_at)
            self._remove_payload(Path(victim.latent_path))
            self._entries.pop(victim.uuid, None)
        self._save()
        return entry
