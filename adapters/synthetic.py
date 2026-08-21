from __future__ import annotations

import hashlib
import base64
import json
from pathlib import Path

from core.protocol import ModelAdapter
from core.types import Generation, Request


class SyntheticAdapter(ModelAdapter):
    model_name = "synthetic"

    def generate(self, request: Request, *, fast: bool = False, cached_latent=None,
                 capture: bool = False, method: str = "baseline") -> Generation:
        digest = hashlib.sha256(f"{request.seed}:{request.prompt}".encode()).digest()
        if capture and method == "nirvana_vid":
            steps = self.config.get("nirvana_vid", {}).get("cache_steps", [5, 10, 15, 20, 25])
            latent = {int(step): digest for step in steps}
        else:
            latent = digest if capture else None
        payload = cached_latent if fast and cached_latent is not None else digest
        return Generation(b"VIDEO-CACHE-BASELINES:" + bytes(payload), latent, 1, 8 if fast else 50)

    def save_latent(self, latent, path: Path) -> None:
        if isinstance(latent, dict):
            payload = {str(key): base64.b64encode(bytes(value)).decode("ascii") for key, value in latent.items()}
            path.write_bytes(b"SYNTHETIC-NIRVANA-V1\n" + json.dumps(payload, sort_keys=True).encode("utf-8"))
            return
        path.write_bytes(bytes(latent))

    def load_latent(self, path: Path):
        payload = path.read_bytes()
        prefix = b"SYNTHETIC-NIRVANA-V1\n"
        if payload.startswith(prefix):
            decoded = json.loads(payload[len(prefix):].decode("utf-8"))
            return {int(key): base64.b64decode(value) for key, value in decoded.items()}
        return payload

    def save_video(self, video, path: Path) -> None:
        path.write_bytes(bytes(video))
