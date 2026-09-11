from __future__ import annotations

from core.protocol import BaselineMethod
from core.types import MethodCapabilities


class Chorus2(BaselineMethod):
    name = "chorus2"
    capabilities = MethodCapabilities(cross_request_cache=True, sparse_attention=True, source_trajectory=True)

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
    return Chorus2(config)
