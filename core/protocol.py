from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .types import Generation, MethodCapabilities, Request


class ModelAdapter(ABC):
    """Small model boundary shared by every method plugin."""

    model_name: str

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def prepare(self) -> None:
        pass

    def reset_hooks(self) -> None:
        pass

    @abstractmethod
    def generate(self, request: Request, *, fast: bool = False, cached_latent: Any = None,
                 capture: bool = False, method: str = "baseline") -> Generation:
        raise NotImplementedError

    def save_latent(self, latent: Any, path: Any) -> None:
        raise NotImplementedError(f"{self.model_name} does not expose latent serialization")

    def load_latent(self, path: Any) -> Any:
        raise NotImplementedError(f"{self.model_name} does not expose latent serialization")

    def save_video(self, video: Any, path: Any) -> None:
        raise NotImplementedError(f"{self.model_name} does not expose video serialization")


class BaselineMethod(ABC):
    """Method plugin contract; model code stays in an adapter."""

    name: str
    capabilities: MethodCapabilities = MethodCapabilities()

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.adapter: ModelAdapter | None = None
        self._cache: Any | None = None

    def prepare(self, model_adapter: ModelAdapter, config: dict[str, Any] | None = None) -> None:
        self.adapter = model_adapter

    def attach_cache(self, cache: Any) -> None:
        self._cache = cache

    def reset_cache(self) -> None:
        if self._cache is not None:
            self._cache.clear()

    @abstractmethod
    def generate(self, request: Request, *, cached_latent: Any = None, capture: bool = False):
        raise NotImplementedError
