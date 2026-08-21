from __future__ import annotations

from core.protocol import BaselineMethod
from core.retrieval import make_nirvana_retriever
from core.types import MethodCapabilities


class NirvanaVID(BaselineMethod):
    name = "nirvana_vid"
    capabilities = MethodCapabilities(cross_request_cache=True, step_skipping=True, source_trajectory=True)

    def prepare(self, model_adapter, config=None) -> None:
        super().prepare(model_adapter, config)
        retrieval_config = self.config.get("retrieval", {}).get("nirvana_vid", {})
        self.retriever = make_nirvana_retriever(retrieval_config, dry_run=bool(self.config.get("dry_run")))
        self.cache_policy = "lcbfu"

    def select_cached_latent(self, latent, match):
        step = match.metadata.get("reuse_step")
        if step is None or not isinstance(latent, dict):
            return latent
        value = latent.get(step, latent.get(str(step)))
        if value is None:
            raise KeyError(f"NIRVANA cache payload has no latent for selected restart step {step}")
        return value

    def generate(self, request, *, cached_latent=None, capture=False):
        assert self.adapter is not None
        previous = getattr(self.adapter, "nirvana_reuse_start_step", None)
        selected_step = self._active_cache_match.metadata.get("reuse_step")
        if cached_latent is not None and selected_step is not None:
            self.adapter.nirvana_reuse_start_step = int(selected_step)
        try:
            return self.adapter.generate(request, fast=cached_latent is not None, cached_latent=cached_latent,
                                         capture=capture, method=self.name)
        finally:
            if previous is None:
                self.adapter.__dict__.pop("nirvana_reuse_start_step", None)
            else:
                self.adapter.nirvana_reuse_start_step = previous


def create(config):
    return NirvanaVID(config)
