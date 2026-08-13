from __future__ import annotations

from core.protocol import BaselineMethod
from core.types import MethodCapabilities


class DenseBaseline(BaselineMethod):
    name = "baseline"
    capabilities = MethodCapabilities()

    def generate(self, request, *, cached_latent=None, capture=False):
        assert self.adapter is not None
        return self.adapter.generate(request, fast=False, cached_latent=None, capture=False, method=self.name)


def create(config):
    return DenseBaseline(config)
