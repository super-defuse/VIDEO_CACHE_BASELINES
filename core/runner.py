from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from .cache import NamespaceCache
from .protocol import BaselineMethod, ModelAdapter
from .types import CacheMatch, Request, RequestResult


class Runner:
    def __init__(self, adapter: ModelAdapter, method: BaselineMethod, *, model: str,
                 output_dir: Path, cache_root: Path) -> None:
        self.adapter, self.method, self.model = adapter, method, model
        self.output_dir = output_dir
        self.video_dir = output_dir / "videos"
        self.video_dir.mkdir(parents=True, exist_ok=True)
        # A method's latent representation is model-specific. Keep both axes
        # in the on-disk namespace so identical prompt UUIDs cannot cross-hit.
        self.cache = NamespaceCache(cache_root, f"{model}/{method.name}")
        method.attach_cache(self.cache)
        method.prepare(adapter, method.config)

    def run(self, requests: Iterable[Request], results_path: Path) -> list[RequestResult]:
        self.adapter.prepare()
        results: list[RequestResult] = []
        with results_path.open("a", encoding="utf-8") as handle:
            for request in requests:
                result = self.process(request)
                handle.write(json.dumps(_jsonable(result), ensure_ascii=False) + "\n")
                handle.flush()
                results.append(result)
        return results

    def process(self, request: Request) -> RequestResult:
        start = time.perf_counter()
        match = CacheMatch()
        timing: dict[str, float] = {}
        try:
            lookup_start = time.perf_counter()
            match = self.cache.find(request) if self.method.capabilities.cross_request_cache else CacheMatch()
            timing["cache_lookup_s"] = time.perf_counter() - lookup_start
            cached = None
            cache_load_start = time.perf_counter()
            loaded = self.cache.load(match)
            if loaded:
                path, _entry = loaded
                cached = self.adapter.load_latent(path)
            timing["cache_load_s"] = time.perf_counter() - cache_load_start
            generation_start = time.perf_counter()
            generation = self.method.generate(request, cached_latent=cached, capture=not match.hit)
            timing["generation_s"] = time.perf_counter() - generation_start
            output_path = self.video_dir / f"{request.request_index:04d}_{request.uuid}.mp4"
            save_start = time.perf_counter()
            self.adapter.save_video(generation.video, output_path)
            timing["save_video_s"] = time.perf_counter() - save_start
            cache_save_start = time.perf_counter()
            if not match.hit and generation.latent is not None and self.method.capabilities.cross_request_cache:
                self.cache.put(request, generation.latent, generation.latent_step or -1, self.adapter.save_latent)
            timing["cache_save_s"] = time.perf_counter() - cache_save_start
            hash_start = time.perf_counter()
            output_hash = _hash(output_path)
            timing["hash_s"] = time.perf_counter() - hash_start
            elapsed = time.perf_counter() - start
            excluded = float(generation.extra.get("excluded_latency_s", 0.0))
            timing["excluded_preprocess_s"] = excluded
            timing["reported_latency_s"] = max(0.0, elapsed - excluded)
            extra = dict(generation.extra)
            extra["timing"] = timing
            return RequestResult(request.request_index, request.uuid, request.prompt, self.model, self.method.name,
                                 request.seed, match, generation.executed_steps, timing["reported_latency_s"],
                                 "ok", str(output_path), str(output_path), output_hash, self.cache.bytes_used,
                                 extra=extra)
        except Exception as exc:
            timing["reported_latency_s"] = time.perf_counter() - start
            return RequestResult(request.request_index, request.uuid, request.prompt, self.model, self.method.name,
                                 request.seed, match, None, timing["reported_latency_s"], "error",
                                 error=f"{type(exc).__name__}: {exc}", cache_bytes=self.cache.bytes_used,
                                 extra={"timing": timing})


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(result: RequestResult) -> dict:
    payload = asdict(result)
    return payload
