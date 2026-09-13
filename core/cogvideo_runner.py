from __future__ import annotations

from pathlib import Path
from typing import Any

from adapters.cogvideo_source_cache import create_manifest, validate_manifest, write_manifest
from .wan21_runner import Wan21ReplayRunner


class CogVideoReplayRunner(Wan21ReplayRunner):
    """Replay fixed legacy decisions against the CogVideoX1.5 baselines.

    The execution protocol -- immutable hit/miss decisions, a per-request
    temporary cache that is removed in ``finally``, a miss that is skipped
    without generation, and the latency deduction rules -- is identical to
    Wan2.1/LTX-Video, so the whole loop is inherited rather than duplicated.
    (It used to be a copy, which is how its timing and error handling drifted
    away from ``core/wan21_runner.py``.)

    The one genuine model difference is the cache itself: a CogVideoX payload
    is a dict of latents plus scheduler state, and it is pinned by a manifest
    that records the model geometry and the replay source UUID.  That check
    runs through the ``_verify_cache`` hook below.
    """

    def _verify_cache(self, tmp_cache: Path, source_uuid: str, cache_payload: Any, cached: Any,
                      source_result: Any) -> None:
        manifest = create_manifest(
            self.config,
            source_uuid=source_uuid,
            scheduler=str(source_result.extra.get("scheduler", "CogVideoXDDIMScheduler")),
            reuse_step=int(source_result.latent_step
                           or self.config.get("nirvana_vid", {}).get("reuse_start_step", 5)),
            latent=cache_payload,
        )
        write_manifest(tmp_cache / "manifest.json", manifest)
        validate_manifest(manifest, config=self.config, source_uuid=source_uuid, latent=cached)
