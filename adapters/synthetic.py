from __future__ import annotations

import hashlib
from pathlib import Path

from core.protocol import ModelAdapter
from core.types import Generation, Request


class SyntheticAdapter(ModelAdapter):
    model_name = "synthetic"

    def generate(self, request: Request, *, fast: bool = False, cached_latent=None,
                 capture: bool = False, method: str = "baseline") -> Generation:
        digest = hashlib.sha256(f"{request.seed}:{request.prompt}".encode()).digest()
        latent = digest if capture else None
        payload = cached_latent if fast and cached_latent is not None else digest
        return Generation(b"VIDEO-CACHE-BASELINES:" + bytes(payload), latent, 1, 8 if fast else 50)

    def save_latent(self, latent, path: Path) -> None:
        path.write_bytes(bytes(latent))

    def load_latent(self, path: Path):
        return path.read_bytes()

    def save_video(self, video, path: Path) -> None:
        path.write_bytes(bytes(video))
