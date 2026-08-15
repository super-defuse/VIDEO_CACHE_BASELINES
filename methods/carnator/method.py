from __future__ import annotations

from core.protocol import BaselineMethod
from core.types import MethodCapabilities


class Carnator(BaselineMethod):
    name = "carnator"
    capabilities = MethodCapabilities(cross_request_cache=True, step_skipping=True, sparse_attention=True,
                                      quantization=True, source_trajectory=True)

    def generate(self, request, *, cached_latent=None, capture=False):
        assert self.adapter is not None
        if self.adapter.model_name == "CogVideoX1.5":
            return self.adapter.generate(request, fast=cached_latent is not None, cached_latent=cached_latent,
                                         capture=capture, method=self.name)
        if self.adapter.model_name != "synthetic":
            raise NotImplementedError("run scripts/run/carnator_wan_matrix.py for the native C0/B0-B5 matrix")
        return self.adapter.generate(request, fast=cached_latent is not None, cached_latent=cached_latent,
                                     capture=capture, method=self.name)


def create(config):
    return Carnator(config)
