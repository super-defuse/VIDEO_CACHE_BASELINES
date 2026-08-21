from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Request:
    request_index: int
    uuid: str
    prompt: str
    seed: int
    # Optional lookup prompt used by source/target experiments.  Generation
    # always uses ``prompt``; caches may deliberately look up the source.
    cache_prompt: str | None = None


@dataclass(frozen=True)
class MethodCapabilities:
    cross_request_cache: bool = False
    step_skipping: bool = False
    cache_attention: bool = False
    sparse_attention: bool = False
    quantization: bool = False
    source_trajectory: bool = False


@dataclass
class Generation:
    video: Any | None
    latent: Any | None = None
    latent_step: int | None = None
    executed_steps: int | None = None
    video_path: Path | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheMatch:
    hit: bool = False
    source_uuid: str | None = None
    source_request_index: int | None = None
    kind: str = "miss"
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RequestResult:
    request_index: int
    uuid: str
    prompt: str
    model: str
    method: str
    seed: int
    cache_match: CacheMatch
    executed_steps: int | None
    latency_s: float
    status: str
    output_path: str | None = None
    quality_input_path: str | None = None
    output_hash: str | None = None
    cache_bytes: int = 0
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
