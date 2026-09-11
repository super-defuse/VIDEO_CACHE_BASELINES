from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from adapters.cogvideo_source_cache import create_manifest, validate_directory, validate_manifest, write_manifest
from .types import CacheMatch, Request, RequestResult


class CogVideoReplayRunner:
    """Execute fixed legacy decisions without consulting the Wan cache."""

    def __init__(self, adapter: Any, method: Any, *, output_dir: Path, replay: dict[int, dict[str, Any]],
                 source_prompts: dict[str, dict[str, Any]], config: dict[str, Any], model: str = "CogVideoX1.5") -> None:
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
        self.tmp_root.mkdir(parents=True, exist_ok=True)

    def run(self, requests: Iterable[Request], results_path: Path) -> list[RequestResult]:
        self.adapter.prepare()
        results: list[RequestResult] = []
        with results_path.open("a", encoding="utf-8") as stream:
            for request in requests:
                result = self.process(request)
                stream.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
                stream.flush()
                results.append(result)
        return results

    def process(self, request: Request) -> RequestResult:
        started = time.perf_counter()
        replay = self.replay.get(request.request_index)
        if replay is None or replay.get("uuid") != request.uuid or replay.get("prompt") != request.prompt:
            return self._error(request, CacheMatch(kind="replay_schema_error"), started,
                               "replay decision does not match request index, UUID, or prompt")
        decision = replay.get("decision")
        if self.method.name == "teacache":
            match = CacheMatch(kind="target_only", metadata={"replay_decision": "target_only"})
        elif decision == "hit":
            source_uuid = replay.get("source_uuid")
            source = self.source_prompts.get(str(source_uuid))
            if not source_uuid or source is None:
                return self._error(request, CacheMatch(kind="replay_hit", metadata={"replay": replay}), started,
                                   "replay hit has no source prompt in the read-only source view")
            match = CacheMatch(True, str(source_uuid), int(source.get("request_index", 0)), "replay_hit",
                               metadata={"replay_decision": "hit", "legacy_status": replay.get("legacy_status")})
        elif decision == "miss":
            # A miss is skipped without any generation: no video, zero latency and
            # zero executed steps. This mirrors core/wan21_runner.py, core/runner.py
            # and the documented miss_policy in run_vidprom_300_experiment.py. The
            # teacache branch above deliberately does not reach here, because
            # CacheMatch(kind="target_only") also has hit=False.
            match = CacheMatch(False, None, None, "replay_miss", metadata={"replay_decision": "miss"})
            return RequestResult(request.request_index, request.uuid, request.prompt, self.model,
                                 self.method.name, request.seed, match, 0, 0.0, "cache_miss",
                                 extra={"replay": replay, "skipped": True})
        else:
            return self._error(request, CacheMatch(kind="replay_schema_error"), started,
                               f"unsupported replay decision: {decision!r}")

        source_prepare_s = 0.0
        tmp_cache = self.tmp_root / f"{request.request_index:04d}_{match.source_uuid or 'target-only'}"
        cached = None
        cleanup = {"status": "pending", "tmp_cache": str(tmp_cache), "tmp_cache_removed": False}
        previous_cache_dir = getattr(self.adapter, "request_cache_dir", None)
        try:
            tmp_cache.mkdir(parents=True, exist_ok=False)
            self.adapter.request_cache_dir = tmp_cache
            if match.hit:
                source_started = time.perf_counter()
                source_request = self.source_prompts[match.source_uuid]
                source = Request(int(source_request.get("request_index", 0)), match.source_uuid,
                                 str(source_request["prompt"]),
                                 int(self.config.get("seed", 20260218)))
                source_result = self.method.generate(source, cached_latent=None, capture=True)
                if source_result.latent is None:
                    raise RuntimeError("CogVideo source generation did not return a cache payload")
                source_latent = source_result.latent
                if self.method.name == "nirvana_vid":
                    reuse_step = int(self.config.get("nirvana_vid", {}).get("reuse_start_step", 5))
                    if not isinstance(source_latent, dict) or reuse_step not in source_latent:
                        raise RuntimeError(f"CogVideo source cache has no completed step {reuse_step}")
                    source_latent = source_latent[reuse_step]
                    if not isinstance(source_latent, dict) or "latents" not in source_latent:
                        raise RuntimeError("CogVideo NIRVANA cache entry must include latents and scheduler state")
                    cache_payload = source_latent
                else:
                    cache_payload = source_latent
                latent_path = tmp_cache / "latent.pt"
                self.adapter.save_latent(cache_payload, latent_path)
                write_manifest(tmp_cache / "manifest.json", create_manifest(
                    self.config,
                    source_uuid=match.source_uuid,
                    scheduler=str(source_result.extra.get("scheduler", "CogVideoXDDIMScheduler")),
                    reuse_step=int(source_result.latent_step or self.config.get("nirvana_vid", {}).get("reuse_start_step", 5)),
                    latent=cache_payload,
                ))
                source_prepare_s = time.perf_counter() - source_started
                manifest = validate_directory(tmp_cache, config=self.config, source_uuid=match.source_uuid)
                cached = self.adapter.load_latent(latent_path)
                validate_manifest(manifest, config=self.config, source_uuid=match.source_uuid, latent=cached)
            target_started = time.perf_counter()
            generation = self.method.generate(request, cached_latent=cached, capture=False)
            target_s = time.perf_counter() - target_started
            output_path = self.video_dir / f"{request.request_index:04d}_{request.uuid}.mp4"
            save_started = time.perf_counter()
            if generation.video_path is not None:
                output_path = Path(generation.video_path)
            else:
                self.adapter.save_video(generation.video, output_path)
            save_s = time.perf_counter() - save_started
            hash_started = time.perf_counter()
            output_hash = _hash(output_path)
            hash_s = time.perf_counter() - hash_started
            elapsed = time.perf_counter() - started

            # Align timing fields with early_signature_disabled experiment (wan version)
            denoising_s = generation.extra.get("dit_forward_s")
            # latency_s: request wall clock excluding the post-hit cache preparation
            # interval and the video/hash bookkeeping, matching the protocol in
            # docs/VIDPROM_1000CACHE_300INPUT_EXPERIMENT_PLAN.md and core/wan21_runner.py.
            # For a miss source_prepare_s is 0.0, so this is a no-op.
            latency_s = max(0.0, elapsed - source_prepare_s - save_s - hash_s)
            # hit_e2e_s and hit_denoising_s: same as latency_s and denoising_s for hit case
            hit_e2e_s = latency_s if match.hit else None
            hit_denoising_s = denoising_s if match.hit else None
            # external_s: external overhead (set to 0 for CogVideo, similar to nirvana_vid)
            external_s = 0.0

            timing = {
                "source_prepare_s": source_prepare_s,
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
                                     "external_s": external_s,
                                     "denoising_s": denoising_s,
                                     "hit_e2e_s": hit_e2e_s,
                                     "hit_denoising_s": hit_denoising_s,
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
            if match.hit:
                event = {
                    "request_index": request.request_index,
                    "uuid": request.uuid,
                    "source_uuid": match.source_uuid,
                    "source_prepare_s": source_prepare_s,
                    "cleanup": cleanup,
                }
                with (self.source_dir / "events.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(event, ensure_ascii=False) + "\n")

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
