from __future__ import annotations

from core.protocol import BaselineMethod
from core.types import MethodCapabilities


class Chai(BaselineMethod):
    name = "chai"
    capabilities = MethodCapabilities(cross_request_cache=True, step_skipping=True, cache_attention=True)

    def generate(self, request, *, cached_latent=None, capture=False):
        assert self.adapter is not None
        return self.adapter.generate(request, fast=cached_latent is not None, cached_latent=cached_latent,
                                     capture=capture, method=self.name)


def create(config):
    return Chai(config)
