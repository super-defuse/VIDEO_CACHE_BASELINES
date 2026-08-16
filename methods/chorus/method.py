from __future__ import annotations

from core.protocol import BaselineMethod
from core.types import MethodCapabilities
import tempfile
from pathlib import Path


class Chorus(BaselineMethod):
    name = "chorus"
    capabilities = MethodCapabilities(
        cross_request_cache=True,
        step_skipping=True,
        cache_attention=True,
        sparse_attention=True,
        source_trajectory=True,
    )

    def generate(self, request, *, cached_latent=None, capture=False):
        assert self.adapter is not None
        if self.adapter.model_name == "CogVideoX1.5":
            raise NotImplementedError("Chorus is N/A for CogVideoX1.5")
        # The source request is always dense and refreshes the temporary video
        # consumed by Grounded-SAM-2. This also makes reruns independent of a
        # stale source latent left in the namespace cache.
        is_source = request.cache_prompt is None
        if is_source:
            cached_latent = None
            capture = True
        elif not getattr(self, "_source_video_path", None):
            # A source trajectory may already exist from a previous process;
            # keep its encoded source video alongside the latent cache.
            entries = getattr(self._cache, "entries", None)
            if entries is not None:
                for entry in getattr(self._cache, "_entries", {}).values():
                    if entry.prompt == request.cache_prompt:
                        candidate = Path(entry.latent_path).with_suffix(".source.mp4")
                        if candidate.is_file():
                            self._source_video_path = candidate
                            break
            if not getattr(self, "_source_video_path", None):
                raise RuntimeError("Chorus target requires the source video generated in this run")
        if not is_source:
            self.adapter.chorus_source_video_path = Path(self._source_video_path)
        result = self.adapter.generate(
            request, fast=cached_latent is not None, cached_latent=cached_latent,
            capture=capture, method=self.name,
        )
        if is_source:
            entries = getattr(self._cache, "entries", None)
            source_path = (Path(entries) / f"{request.uuid}.source.mp4" if entries is not None
                           else Path(tempfile.mkstemp(prefix="chorus-source-", suffix=".mp4")[1]))
            self.adapter.save_video(result.video, source_path)
            self._source_video_path = source_path
            self._source_prompt = request.prompt
        else:
            self._source_video_path = None
        return result


def create(config):
    return Chorus(config)
