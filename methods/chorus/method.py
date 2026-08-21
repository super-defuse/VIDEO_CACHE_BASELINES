from __future__ import annotations

from core.protocol import BaselineMethod
from core.retrieval import make_chorus_retriever
from core.types import MethodCapabilities
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

    def prepare(self, model_adapter, config=None) -> None:
        super().prepare(model_adapter, config)
        retrieval_config = self.config.get("retrieval", {}).get("chorus", {})
        self.retriever = make_chorus_retriever(retrieval_config, dry_run=bool(self.config.get("dry_run")))
        self.cache_policy = "lru"

    def generate(self, request, *, cached_latent=None, capture=False):
        assert self.adapter is not None
        if self.adapter.model_name == "CogVideoX1.5":
            raise NotImplementedError("Chorus is N/A for CogVideoX1.5")
        # A cache miss produces a dense source trajectory. A semantic hit is a
        # target request even though it has no manually supplied cache_prompt.
        is_source = cached_latent is None
        if is_source:
            capture = True
        else:
            entry = self._active_cache_entry
            if entry is None:
                raise RuntimeError("Chorus target is missing its matched cache entry")
            source_video_path = Path(entry.latent_path).with_suffix(".source.mp4")
            if not source_video_path.is_file():
                raise RuntimeError(f"Chorus target source video is missing: {source_video_path}")
        if not is_source:
            self.adapter.chorus_source_video_path = source_video_path
        boundaries = self._active_cache_match.metadata.get("stage_boundaries")
        previous_boundaries = getattr(self.adapter, "chorus_stage_boundaries", None)
        if boundaries:
            self.adapter.chorus_stage_boundaries = dict(boundaries)
        try:
            result = self.adapter.generate(
                request, fast=cached_latent is not None, cached_latent=cached_latent,
                capture=capture, method=self.name,
            )
        finally:
            if previous_boundaries is None:
                self.adapter.__dict__.pop("chorus_stage_boundaries", None)
            else:
                self.adapter.chorus_stage_boundaries = previous_boundaries
        if is_source:
            entries = getattr(self._cache, "entries", None)
            if entries is None:
                raise RuntimeError("Chorus source requires an attached namespace cache")
            source_path = Path(entries) / f"{request.uuid}.source.mp4"
            self.adapter.save_video(result.video, source_path)
        return result


def create(config):
    return Chorus(config)
