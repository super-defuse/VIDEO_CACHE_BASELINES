from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterable

from .types import CacheMatch, Request, RequestResult


class Wan21ReplayRunner:
    """Replay fixed legacy hit/miss decisions against the Wan2.1 baselines.

    ``replay`` is ``None`` for the baseline, which runs every request as a
    plain dense generation. Otherwise each request must have an immutable
    decision row: a miss is skipped without any generation, and a hit runs
    the source request into a per-request temporary cache that the target
    reuses before the cache is removed again.
    """

    def __init__(self, adapter: Any, method: Any, *, output_dir: Path,
                 replay: dict[int, dict[str, Any]] | None,
                 source_prompts: dict[str, dict[str, Any]], config: dict[str, Any],
                 model: str = "Wan2.1-T2V-1.3B") -> None:
        self.adapter = adapter
        self.method = method
        self.output_dir = output_dir
        self.replay = replay
        self.source_prompts = source_prompts
        self.config = config
        self.model = model
        self.video_dir = output_dir / "videos"
        self.source_dir = output_dir / "source_prepare"
        self.tmp_root = output_dir / "tmp_cache" / method.name
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.source_dir.mkdir(parents=True, exist_ok=True)
        if replay is not None:
            self.tmp_root.mkdir(parents=True, exist_ok=True)

    def run(self, requests: Iterable[Request], results_path: Path) -> list[RequestResult]:
        self.adapter.prepare()
        results: list[RequestResult] = []
        with results_path.open("a", encoding="utf-8") as stream:
            for request in requests:
                result = self.process(request)
                self._release_device_memory()
                stream.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
                stream.flush()
                results.append(result)
        return results

    @staticmethod
    def _release_device_memory() -> None:
        """Return cached allocator segments between requests.

        A failed request (CUDA OOM included) otherwise leaves dead frames
        holding the device through the allocator, and every later request
        fails instantly on allocation instead of running to completion.
        """
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def process(self, request: Request) -> RequestResult:
        started = time.perf_counter()
        if self.replay is None:
            return self._process_baseline(request, started)

        replay = self.replay.get(request.request_index)
        if replay is None or replay.get("uuid") != request.uuid or replay.get("prompt") != request.prompt:
            return self._error(request, CacheMatch(kind="replay_schema_error"), started,
                               "replay decision does not match request index, UUID, or prompt")
        decision = replay.get("decision")
        if decision == "hit":
            source_uuid = replay.get("source_uuid")
            source = self.source_prompts.get(str(source_uuid))
            if not source_uuid or source is None:
                return self._error(request, CacheMatch(kind="replay_hit", metadata={"replay": replay}), started,
                                   "replay hit has no source prompt in the read-only source view")
            match = CacheMatch(True, str(source_uuid), int(source.get("request_index", 0)), "replay_hit",
                               metadata={"replay_decision": "hit", "legacy_status": replay.get("legacy_status")})
        elif decision == "miss":
            match = CacheMatch(False, None, None, "replay_miss", metadata={"replay_decision": "miss"})
        else:
            return self._error(request, CacheMatch(kind="replay_schema_error"), started,
                               f"unsupported replay decision: {decision!r}")
        if not match.hit:
            return RequestResult(request.request_index, request.uuid, request.prompt, self.model, self.method.name,
                                 request.seed, match, None, 0.0, "cache_miss",
                                 extra={"replay": replay, "skipped": True})

        source_prepare_s = 0.0
        tmp_cache = self.tmp_root / f"{request.request_index:04d}_{match.source_uuid}"
        cleanup = {"status": "pending", "tmp_cache": str(tmp_cache), "tmp_cache_removed": False}
        previous_cache_dir = getattr(self.adapter, "request_cache_dir", None)
        try:
            tmp_cache.mkdir(parents=True, exist_ok=False)
            self.adapter.request_cache_dir = tmp_cache
            source_started = time.perf_counter()
            source_request = Request(int(self.source_prompts[match.source_uuid].get("request_index", 0)),
                                     match.source_uuid,
                                     str(self.source_prompts[match.source_uuid]["prompt"]),
                                     int(self.config.get("seed", 20260218)))
            source_result = self.method.generate(source_request, cached_latent=None, capture=True)
            if source_result.latent is None:
                raise RuntimeError("Wan2.1 source generation did not return a cache payload")
            cache_payload = source_result.latent
            source_video_path: Path | None = None
            if self.method.name == "nirvana_vid":
                reuse_step = int(self.config.get("nirvana_vid", {}).get("reuse_start_step", 5))
                if not isinstance(cache_payload, dict) or reuse_step not in cache_payload:
                    raise RuntimeError(f"Wan2.1 NIRVANA source cache has no completed step {reuse_step}")
                cache_payload = cache_payload[reuse_step]
            elif self.method.name == "chorus":
                if source_result.video is None:
                    raise RuntimeError("Wan2.1 Chorus source generation did not return a video tensor")
                source_video_path = tmp_cache / f"{match.source_uuid}.source.mp4"
                self.adapter.save_video(source_result.video, source_video_path)
            self.adapter.save_latent(cache_payload, tmp_cache / "latent.pt")
            source_prepare_s = time.perf_counter() - source_started
            # Cache loading and mask-cache GPU preparation stay outside the
            # reported latency, matching the Wan2.2 hit convention
            # (exclude_cache_prepare_latency / native prepare_s).
            load_started = time.perf_counter()
            cached = self.adapter.load_latent(tmp_cache / "latent.pt")
            cache_load_s = time.perf_counter() - load_started

            target_request = request
            if source_video_path is not None:
                self.adapter.chorus_source_video_path = source_video_path
                target_request = replace(request, cache_prompt=source_request.prompt)
            target_started = time.perf_counter()
            try:
                generation = self.method.generate(target_request, cached_latent=cached, capture=False)
            finally:
                if source_video_path is not None:
                    self.adapter.__dict__.pop("chorus_source_video_path", None)
            target_s = time.perf_counter() - target_started
            output_path = self.video_dir / f"{request.request_index:04d}_{request.uuid}.mp4"
            save_started = time.perf_counter()
            if generation.video_path is not None:
                output_path = Path(generation.video_path)
                if output_path.resolve().parent != self.video_dir.resolve():
                    shutil.move(str(output_path), self.video_dir / output_path.name)
                    output_path = self.video_dir / output_path.name
            else:
                self.adapter.save_video(generation.video, output_path)
            save_s = time.perf_counter() - save_started
            hash_started = time.perf_counter()
            output_hash = _hash(output_path)
            hash_s = time.perf_counter() - hash_started
            elapsed = time.perf_counter() - started

            denoising_s = generation.extra.get("dit_forward_s")
            cache_prepare_s = float(generation.extra.get("cache_prepare_s", 0.0) or 0.0)
            latency_s = max(0.0, elapsed - source_prepare_s - cache_load_s - cache_prepare_s - save_s - hash_s)
            timing = {
                "source_prepare_s": source_prepare_s,
                "cache_load_s": cache_load_s,
                "cache_prepare_s": cache_prepare_s,
                "target_e2e_s": latency_s,
                "target_generation_s": target_s,
                "denoising_s": denoising_s,
                "save_video_s": save_s,
                "hash_s": hash_s,
            }
            cleanup["status"] = "ready"
            return RequestResult(request.request_index, request.uuid, request.prompt, self.model, self.method.name,
                                 request.seed, match, generation.executed_steps, latency_s, "ok",
                                 str(output_path), str(output_path), output_hash, extra={
                                     "replay": replay,
                                     "source_prepare_s": source_prepare_s,
                                     "external_s": 0.0,
                                     "denoising_s": denoising_s,
                                     "hit_e2e_s": latency_s,
                                     "hit_denoising_s": denoising_s,
                                     "timing": timing,
                                     "cleanup": cleanup,
                                     **generation.extra,
                                 })
        except Exception as exc:
            return self._error(request, match, started, f"{type(exc).__name__}: {exc}", replay=replay,
                               source_prepare_s=source_prepare_s, cleanup=cleanup)
        finally:
            if previous_cache_dir is None:
                self.adapter.__dict__.pop("request_cache_dir", None)
            else:
                self.adapter.request_cache_dir = previous_cache_dir
            if tmp_cache.exists():
                shutil.rmtree(tmp_cache)
            cleanup["tmp_cache_removed"] = not tmp_cache.exists()
            cleanup["status"] = "cleaned" if cleanup["tmp_cache_removed"] else "cleanup_failed"
            event = {
                "request_index": request.request_index,
                "uuid": request.uuid,
                "source_uuid": match.source_uuid,
                "source_prepare_s": source_prepare_s,
                "cleanup": cleanup,
            }
            with (self.source_dir / "events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _process_baseline(self, request: Request, started: float) -> RequestResult:
        match = CacheMatch(kind="baseline_run", metadata={"replay_decision": "baseline_all_requests"})
        try:
            generation = self.method.generate(request, cached_latent=None, capture=False)
            output_path = self.video_dir / f"{request.request_index:04d}_{request.uuid}.mp4"
            save_started = time.perf_counter()
            if generation.video_path is not None:
                output_path = Path(generation.video_path)
                if output_path.resolve().parent != self.video_dir.resolve():
                    shutil.move(str(output_path), self.video_dir / output_path.name)
                    output_path = self.video_dir / output_path.name
            else:
                self.adapter.save_video(generation.video, output_path)
            save_s = time.perf_counter() - save_started
            hash_started = time.perf_counter()
            output_hash = _hash(output_path)
            hash_s = time.perf_counter() - hash_started
            elapsed = time.perf_counter() - started
            denoising_s = generation.extra.get("dit_forward_s")
            return RequestResult(request.request_index, request.uuid, request.prompt, self.model, self.method.name,
                                 request.seed, match, generation.executed_steps,
                                 max(0.0, elapsed - save_s - hash_s), "ok",
                                 str(output_path), str(output_path), output_hash, extra={
                                     "replay": None,
                                     "source_prepare_s": 0.0,
                                     "external_s": 0.0,
                                     "denoising_s": denoising_s,
                                     "timing": {
                                         "source_prepare_s": 0.0,
                                         "target_e2e_s": max(0.0, elapsed - save_s - hash_s),
                                         "denoising_s": denoising_s,
                                         "save_video_s": save_s,
                                         "hash_s": hash_s,
                                     },
                                     **generation.extra,
                                 })
        except Exception as exc:
            return self._error(request, match, started, f"{type(exc).__name__}: {exc}")

    def _error(self, request: Request, match: CacheMatch, started: float, message: str, *, replay: Any = None,
               source_prepare_s: float = 0.0, cleanup: dict[str, Any] | None = None) -> RequestResult:
        return RequestResult(request.request_index, request.uuid, request.prompt, self.model, self.method.name,
                             request.seed, match, None, time.perf_counter() - started, "error", error=message,
                             extra={"replay": replay, "source_prepare_s": source_prepare_s,
                                    "cleanup": cleanup or {"status": "not_required", "tmp_cache_removed": True}})


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
