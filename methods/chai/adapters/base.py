from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from core.types import Generation


class T2VAdapter(ABC):
    """Common model boundary for full and CHAI accelerated T2V inference."""

    latent_extension = ".pt"

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.chai_manager: Any | None = None

    def set_chai_manager(self, manager: Any) -> None:
        self.chai_manager = manager

    def prepare(self) -> None:
        """Load persistent model state before request-level timing starts."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        seed: int,
        *,
        fast: bool = False,
        cached_latent: Any | None = None,
        capture: bool = False,
    ) -> Generation:
        raise NotImplementedError

    @abstractmethod
    def save_latent(self, latent: Any, path: Path) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_latent(self, path: Path) -> Any:
        raise NotImplementedError

    @abstractmethod
    def save_video(self, video: Any, path: Path) -> None:
        raise NotImplementedError
