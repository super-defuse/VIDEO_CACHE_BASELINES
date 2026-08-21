from __future__ import annotations

from pathlib import Path

from core.protocol import BaselineMethod
from core.retrieval import make_carnator_retriever
from core.types import MethodCapabilities


class Carnator(BaselineMethod):
    name = "carnator"
    capabilities = MethodCapabilities(cross_request_cache=True, step_skipping=True, sparse_attention=True,
                                      quantization=True, source_trajectory=True)

    def prepare(self, model_adapter, config=None) -> None:
        super().prepare(model_adapter, config)
        retrieval_config = self.config.get("retrieval", {}).get("carnator", {})
        if model_adapter.model_name != "CogVideoX1.5":
            self.retriever = make_carnator_retriever(
                retrieval_config,
                dry_run=bool(self.config.get("dry_run")),
            )

    def select_cached_latent(self, latent, match):
        if not match.metadata.get("cache_ready", False):
            raise RuntimeError("Carnator cache match is not artifact-ready")
        return latent

    def generate(self, request, *, cached_latent=None, capture=False):
        assert self.adapter is not None
        previous_sparse = getattr(self.adapter, "carnator_sparse_attention_artifact", None)
        if cached_latent is not None and self._active_cache_match.metadata.get("reuse_mode") == "joint_latent_sparse_attention":
            if self._active_cache_entry is not None:
                self.adapter.carnator_sparse_attention_artifact = str(
                    Path(self._active_cache_entry.latent_path).with_suffix(".sparse_attention")
                )
        try:
            if self.adapter.model_name == "CogVideoX1.5":
                return self.adapter.generate(request, fast=cached_latent is not None, cached_latent=cached_latent,
                                             capture=capture, method=self.name)
            if self.adapter.model_name != "synthetic":
                raise NotImplementedError("run scripts/run/carnator_wan_matrix.py for the native C0/B0-B5 matrix")
            return self.adapter.generate(request, fast=cached_latent is not None, cached_latent=cached_latent,
                                         capture=capture, method=self.name)
        finally:
            if previous_sparse is None:
                self.adapter.__dict__.pop("carnator_sparse_attention_artifact", None)
            else:
                self.adapter.carnator_sparse_attention_artifact = previous_sparse


def create(config):
    return Carnator(config)
