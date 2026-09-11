from __future__ import annotations

from core.protocol import BaselineMethod
from core.types import MethodCapabilities


class CarnatorBoundaryB5(BaselineMethod):
    name = "carnator_boundary_b5"
    capabilities = MethodCapabilities(
        cross_request_cache=True,
        step_skipping=True,
        sparse_attention=True,
        quantization=True,
        source_trajectory=True,
    )

    def generate(self, request, *, cached_latent=None, capture=False):
        assert self.adapter is not None
        return self.adapter.generate(
            request,
            fast=cached_latent is not None,
            cached_latent=cached_latent,
            capture=capture,
            method=self.name,
        )


def create(config):
    return CarnatorBoundaryB5(config)
