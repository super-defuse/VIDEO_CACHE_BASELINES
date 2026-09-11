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
            if self.config.get("model") == "cogvideox15":
                latent = {int(step): {"latents": digest, "scheduler": "CogVideoXDDIMScheduler",
                                      "timesteps": list(range(50, 0, -1)), "completed_step": int(step)}
                          for step in steps}
            else:
                latent = {int(step): digest for step in steps}
        elif capture and self.config.get("model") == "cogvideox15" and method in {
            "chorus", "chorus2", "carnator_boundary_b5"
        }:
            formats = {
                "chorus": "cogvideox15-chorus-trajectory/v1",
                "chorus2": "cogvideox15-chorus2-source-routing/v1",
                "carnator_boundary_b5": "cogvideox15-carnator-boundary-b5/v1",
            }
            latent = {"format": formats[method], "latents": digest, "scheduler": "CogVideoXDDIMScheduler"}
        else:
            latent = digest if capture else None
        payload = cached_latent if fast and cached_latent is not None else digest
        if isinstance(payload, dict) and "latents" in payload:
            payload = payload["latents"]
        anchor_steps = {"chorus": 5, "chorus2": 10, "carnator": 10, "carnator_boundary_b5": 10}
        latent_step = (max(int(step) for step in self.config.get("nirvana_vid", {}).get("cache_steps", [5]))
                       if capture and method == "nirvana_vid" else anchor_steps.get(method, 1))
        return Generation(b"VIDEO-CACHE-BASELINES:" + bytes(payload), latent, latent_step, 8 if fast else 50)

    def save_latent(self, latent, path: Path) -> None:
        if isinstance(latent, dict):
            def encode(value):
                if isinstance(value, (bytes, bytearray)):
                    return {"__bytes__": base64.b64encode(bytes(value)).decode("ascii")}
                if isinstance(value, dict):
                    return {str(key): encode(item) for key, item in value.items()}
                if isinstance(value, (list, tuple)):
                    return [encode(item) for item in value]
                return value
            kind = "payload" if "latents" in latent else "steps"
            payload = {"__kind__": kind, "value": {str(key): encode(value) for key, value in latent.items()}}
            path.write_bytes(b"SYNTHETIC-NIRVANA-V1\n" + json.dumps(payload, sort_keys=True).encode("utf-8"))
            return
        path.write_bytes(bytes(latent))

    def load_latent(self, path: Path):
        payload = path.read_bytes()
        prefix = b"SYNTHETIC-NIRVANA-V1\n"
        if payload.startswith(prefix):
            def decode(value):
                if isinstance(value, dict) and "__bytes__" in value:
                    return base64.b64decode(value["__bytes__"])
                if isinstance(value, dict):
                    return {key: decode(item) for key, item in value.items()}
                if isinstance(value, list):
                    return [decode(item) for item in value]
                return value
            decoded = json.loads(payload[len(prefix):].decode("utf-8"))
            value = {key: decode(item) for key, item in decoded["value"].items()}
            return value if decoded.get("__kind__") == "payload" else {int(key): item for key, item in value.items()}
        return payload

    def save_video(self, video, path: Path) -> None:
        path.write_bytes(bytes(video))
