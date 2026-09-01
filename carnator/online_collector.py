"""Native and deterministic collectors for progressive Carnator replay."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import torch

from carnator.early_signature import write_early_signature
from core.types import CacheMatch, Request


_SPARSE_ARTIFACT_FORMAT = "carnator-sparse-attention-artifact"


@dataclass(frozen=True)
class OnlineCollectionResult:
    """Artifacts and timing produced for one arrival-order request."""

    match: CacheMatch | None
    source_signature: dict[str, Any] | None
    probe_collected: bool
    source_cached: bool
    early_probe_s: float
    signature_gate_s: float
    source_cache_s: float
    completed_steps: int
    step10_cache_assumed: bool
    probe_latent_path: str | None = None
    dit_forward_s: float = 0.0


ProbeEvaluator = Callable[[dict[str, Any]], CacheMatch]


def _assumed_step10_cache(metadata: dict[str, Any], request_uuid: str) -> dict[str, Any]:
    """Describe a logical step-10 cache without materializing its artifacts."""

    prefix = f"assumed://{request_uuid}"
    return {
        "anchor_step": 10,
        "assumption": "step10_cache_exists",
        "model": metadata["model"],
        "latent_token_grid": metadata["latent_token_grid"],
        "latent_path": f"{prefix}/step10.pt",
        "latent_sha256": "assumed",
        "sparse_attention_path": f"{prefix}/sparse_attention.json",
        "sparse_attention_sha256": "assumed",
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sparse_artifact(path: Path, *, uuid: str, mbase_path: Path, mask_policy: str) -> str:
    """Persist the source-side mask state consumed by sparse joint reuse."""

    payload = {
        "format": _SPARSE_ARTIFACT_FORMAT,
        "source_uuid": uuid,
        "mask_policy": mask_policy,
        "mbase_path": str(mbase_path.resolve()),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return _sha256(path)


class SyntheticOnlineCollector:
    """CPU-only progressive collector used by protocol tests and dry runs."""

    def __init__(
        self,
        artifact_root: Path,
        *,
        probe_seed: int,
        noun_extractor: str = "test-nouns",
        assume_step10_cache: bool = False,
    ) -> None:
        self.artifact_root = artifact_root.resolve()
        self.probe_seed = int(probe_seed)
        self.noun_extractor = noun_extractor
        self.mask_policy = "synthetic-mbase"
        self.assume_step10_cache = bool(assume_step10_cache)

    def close(self) -> None:
        return None

    def _tail_frame(self, step5_latent: torch.Tensor) -> torch.Tensor:
        pooled = step5_latent.mean(dim=(1, 2, 3))
        channels = torch.stack((pooled[0], pooled[-1], pooled.mean())).view(3, 1, 1)
        return torch.tanh(channels).expand(3, 8, 10).contiguous()

    def _write_signature(
        self,
        directory: Path,
        *,
        step5_latent: torch.Tensor,
        metadata: dict[str, Any],
        masks: dict[str, torch.Tensor],
        decoded_tail_frame: torch.Tensor,
        cache: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return write_early_signature(
            directory,
            step5_latent=step5_latent,
            decoded_tail_frame=decoded_tail_frame,
            metadata=metadata,
            mbase_by_noun=masks,
            cache=cache,
        )

    def process(
        self,
        request: Request,
        nouns: Iterable[str],
        *,
        evaluate_probe: ProbeEvaluator | None,
    ) -> OnlineCollectionResult:
        noun_list = tuple(sorted(set(str(noun) for noun in nouns)))
        if not noun_list:
            raise ValueError("online collector needs at least one noun token")
        source_dir = self.artifact_root / request.uuid
        source_dir.mkdir(parents=True, exist_ok=False)
        prompt_digest = hashlib.sha256(request.prompt.encode("utf-8")).digest()
        generator = torch.Generator(device="cpu").manual_seed(int.from_bytes(prompt_digest[:8], "little"))
        step5 = torch.randn((2, 3, 4, 5), generator=generator, dtype=torch.float32)
        masks = {noun: torch.ones((3, 4, 5), dtype=torch.bool) for noun in noun_list}
        metadata = {
            "prompt": request.prompt,
            "model": "synthetic-wan",
            "scheduler": "unipc",
            "scheduler_config": {"flow_shift": 5.0, "solver_order": 2, "total_steps": 50},
            "cfg": 5.0,
            "probe_seed": self.probe_seed,
            "completed_step": 5,
            "latent_shape": [2, 3, 4, 5],
            "latent_token_grid": [3, 4, 5],
            "noun_extractor": self.noun_extractor,
            "mask_policy": self.mask_policy,
        }
        probe_started = time.perf_counter()
        signature = self._write_signature(
            source_dir,
            step5_latent=step5,
            metadata=metadata,
            masks=masks,
            decoded_tail_frame=self._tail_frame(step5),
        )
        early_probe_s = time.perf_counter() - probe_started
        match = None
        signature_gate_s = 0.0
        if evaluate_probe is not None:
            gate_started = time.perf_counter()
            match = evaluate_probe(signature)
            signature_gate_s = time.perf_counter() - gate_started
            if match.hit:
                shutil.rmtree(source_dir)
                return OnlineCollectionResult(
                    match, None, True, False, early_probe_s, signature_gate_s, 0.0, 5, False
                )

        if self.assume_step10_cache:
            cache = _assumed_step10_cache(metadata, request.uuid)
            source_signature = self._write_signature(
                source_dir,
                step5_latent=step5,
                metadata=metadata,
                masks=masks,
                decoded_tail_frame=self._tail_frame(step5),
                cache=cache,
            )
            return OnlineCollectionResult(
                match, source_signature, evaluate_probe is not None, True,
                early_probe_s if evaluate_probe is not None else 0.0,
                signature_gate_s, 0.0, 5, True,
            )

        cache_started = time.perf_counter()
        step10_path = source_dir / "step10.pt"
        torch.save(step5, step10_path)
        sparse_path = source_dir / "sparse_attention.json"
        sparse_hash = _sparse_artifact(
            sparse_path,
            uuid=request.uuid,
            mbase_path=source_dir / "mbase.safetensors",
            mask_policy=self.mask_policy,
        )
        cache = {
            "anchor_step": 10,
            "model": metadata["model"],
            "latent_token_grid": metadata["latent_token_grid"],
            "latent_path": str(step10_path.resolve()),
            "latent_sha256": _sha256(step10_path),
            "sparse_attention_path": str(sparse_path.resolve()),
            "sparse_attention_sha256": sparse_hash,
        }
        source_signature = self._write_signature(
            source_dir,
            step5_latent=step5,
            metadata=metadata,
            masks=masks,
            decoded_tail_frame=self._tail_frame(step5),
            cache=cache,
        )
        return OnlineCollectionResult(
            match,
            source_signature,
            evaluate_probe is not None,
            True,
            early_probe_s if evaluate_probe is not None else 0.0,
            signature_gate_s,
            time.perf_counter() - cache_started,
            10,
            False,
        )


class NativeOnlineCollector:
    """Run the native Wan request prefix once and materialize misses as sources."""

    def __init__(
        self,
        args: Any,
        artifact_root: Path,
        *,
        probe_seed: int,
        noun_extractor: str,
        assume_step10_cache: bool = False,
        retain_artifacts: bool = False,
        runtime: Any | None = None,
    ) -> None:
        from carnator.adapters import wan_ti2v_5b as native

        if int(args.steps) != 50:
            raise ValueError("online collector requires the complete 50-step scheduler")
        if int(args.mask_probe_step) != 5:
            raise ValueError("online collector requires completed step 5")
        if int(args.reuse_start_step) != 10:
            raise ValueError("online collector requires a step-10 source cache")
        self.native = native
        self.args = args
        self.artifact_root = artifact_root.resolve()
        self.probe_seed = int(probe_seed)
        self.noun_extractor = noun_extractor
        self.assume_step10_cache = bool(assume_step10_cache)
        self.retain_artifacts = bool(retain_artifacts)
        self.mask_policy = native.fused_mbase_policy(args)
        native.ensure_import_paths(args)
        native.validate_experiment_config(args)
        self._owns_runtime = runtime is None
        self.runtime = runtime or native.load_runtime(args)

    def close(self) -> None:
        if not self._owns_runtime:
            return
        self.runtime.profiler.close()
        if self.args.offload_model:
            self.runtime.pipe.model.cpu()
        self.runtime.torch.cuda.empty_cache()

    def _metadata(self, request: Request, latent, token_grid: tuple[int, int, int]) -> dict[str, Any]:
        return {
            "prompt": request.prompt,
            "model": "Wan2.2-TI2V-5B",
            "scheduler": self.args.scheduler,
            "scheduler_config": {
                "flow_shift": float(self.args.flow_shift),
                "solver_order": int(self.args.solver_order) if self.args.scheduler == "unipc" else None,
                "total_steps": int(self.args.steps),
            },
            "cfg": float(self.args.guidance_scale),
            "probe_seed": self.probe_seed,
            "completed_step": 5,
            "latent_shape": list(latent.shape),
            "latent_token_grid": list(token_grid),
            "noun_extractor": self.noun_extractor,
            "mask_policy": self.mask_policy,
        }

    def _decoded_tail_frame(self, latent: torch.Tensor) -> torch.Tensor:
        video = self.runtime.pipe.vae.decode([latent])[0]
        if not isinstance(video, torch.Tensor) or video.ndim != 4 or video.shape[0] < 3 or video.shape[1] <= 0:
            raise RuntimeError("Wan VAE did not return a [C,T,H,W] decoded video")
        return video[:, -1].detach()

    def _write_signature(
        self,
        directory: Path,
        *,
        step5_latent: torch.Tensor,
        metadata: dict[str, Any],
        masks: dict[str, torch.Tensor],
        decoded_tail_frame: torch.Tensor,
        cache: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return write_early_signature(
            directory,
            step5_latent=step5_latent,
            decoded_tail_frame=decoded_tail_frame,
            metadata=metadata,
            mbase_by_noun=masks,
            cache=cache,
        )

    def process(
        self,
        request: Request,
        nouns: Iterable[str],
        *,
        evaluate_probe: ProbeEvaluator | None,
    ) -> OnlineCollectionResult:
        noun_list = tuple(sorted(set(str(noun) for noun in nouns)))
        if not noun_list:
            raise ValueError("online collector needs at least one noun token")
        native = self.native
        args = copy.copy(self.args)
        args.source_prompt = request.prompt
        args.source_positive = ""
        args.seed = self.probe_seed
        source_dir = self.artifact_root / request.uuid
        source_dir.mkdir(parents=True, exist_ok=False)

        context, context_null = native.encode_prompts(self.runtime, args, request.prompt)
        initial_latent, generator = native.prepare_initial_latent(self.runtime, args)
        latent = initial_latent
        token_grid = native.token_grid_for(latent, self.runtime.pipe.patch_size)
        seq_len = native.seq_len_for(self.runtime, latent)
        scheduler = native.build_scheduler(self.runtime, args)
        index_by_noun = {
            noun: native.find_token_indices(
                self.runtime.pipe.text_encoder, request.prompt, noun, self.runtime.pipe.config.text_len
            )
            for noun in noun_list
        }
        probe = native.NativeMultiNounCrossAttentionProbe(
            self.runtime.pipe.model,
            native.parse_int_list(args.mask_layers),
            index_by_noun,
            token_grid,
        )
        model = self.runtime.pipe.model
        if args.offload_model or self.runtime.pipe.init_on_cpu:
            model.to(self.runtime.device)
            torch.cuda.empty_cache()

        match: CacheMatch | None = None
        source_signature: dict[str, Any] | None = None
        early_probe_s = 0.0
        signature_gate_s = 0.0
        source_started = time.perf_counter()
        completed_steps = 0
        source_cached = False
        probe_started = time.perf_counter()
        dit_forward_s = 0.0
        step5_latent: torch.Tensor | None = None
        mbase_by_noun: dict[str, torch.Tensor] | None = None
        decoded_tail_frame: torch.Tensor | None = None
        no_sync = getattr(model, "no_sync", nullcontext)
        with torch.amp.autocast("cuda", dtype=self.runtime.pipe.param_dtype), torch.no_grad(), no_sync(), probe:
            for step_index, timestep in enumerate(scheduler.timesteps):
                probe.enabled = (step_index + 1) == args.mask_probe_step
                probe.current_step_index = step_index
                timestep_input = native.timestep_for(self.runtime, latent, timestep, seq_len)
                torch.cuda.synchronize()
                dit_started = time.perf_counter()
                noise_pred_cond = model([latent], t=timestep_input, context=context, seq_len=seq_len)[0]
                torch.cuda.synchronize()
                dit_forward_s += time.perf_counter() - dit_started
                if args.guidance_scale == 1.0:
                    noise_pred = noise_pred_cond
                else:
                    torch.cuda.synchronize()
                    dit_started = time.perf_counter()
                    noise_pred_uncond = model([latent], t=timestep_input, context=context_null, seq_len=seq_len)[0]
                    torch.cuda.synchronize()
                    dit_forward_s += time.perf_counter() - dit_started
                    noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                latent = scheduler.step(
                    noise_pred.unsqueeze(0), timestep, latent.unsqueeze(0), return_dict=False, generator=generator
                )[0].squeeze(0)
                completed_steps = step_index + 1

                if completed_steps == args.mask_probe_step:
                    torch.cuda.synchronize()
                    step5_latent = latent.detach().float().cpu().contiguous()
                    metadata = self._metadata(request, latent, token_grid)
                    mbase_by_noun = native.build_fused_mbase_by_noun(
                        self.runtime,
                        args,
                        prompt=request.prompt,
                        token_grid=token_grid,
                        noun_token_indices=index_by_noun,
                        cross_records_by_noun=probe.records_by_noun,
                        self_attention_input=latent,
                    )
                    decoded_tail_frame = self._decoded_tail_frame(latent)
                    torch.cuda.synchronize()
                    prefix_to_step5_s = time.perf_counter() - probe_started
                    signature = self._write_signature(
                        source_dir,
                        step5_latent=step5_latent,
                        metadata=metadata,
                        masks=mbase_by_noun,
                        decoded_tail_frame=decoded_tail_frame,
                    )
                    # The target continuation consumes this exact step-5
                    # latent.  Persist it once so a gate miss can resume the
                    # same schedule without recomputing the prefix.
                    probe_latent_path = source_dir / "step5.pt"
                    native.save_latent(probe_latent_path, step5_latent)
                    if evaluate_probe is not None:
                        early_probe_s = prefix_to_step5_s
                        gate_started = time.perf_counter()
                        match = evaluate_probe(signature)
                        signature_gate_s = time.perf_counter() - gate_started
                        # Both outcomes leave the collector at step 5.  The
                        # caller either resumes the source at step 10 (hit)
                        # or continues dense from this target latent (miss).
                        break

                    if self.assume_step10_cache:
                        source_signature = self._write_signature(
                            source_dir,
                            step5_latent=step5_latent,
                            metadata=metadata,
                            masks=mbase_by_noun,
                            decoded_tail_frame=decoded_tail_frame,
                            cache=_assumed_step10_cache(metadata, request.uuid),
                        )
                        source_cached = True
                        break

                if completed_steps == args.reuse_start_step:
                    if step5_latent is None:
                        raise RuntimeError("source cache reached step 10 without a step-5 latent")
                    if mbase_by_noun is None:
                        raise RuntimeError("source cache reached step 10 without step-5 Mbase")
                    if decoded_tail_frame is None:
                        raise RuntimeError("source cache reached step 10 without a decoded step-5 tail frame")
                    step10_path = source_dir / "step10.pt"
                    native.save_latent(step10_path, latent)
                    sparse_path = source_dir / "sparse_attention.json"
                    sparse_hash = _sparse_artifact(
                        sparse_path,
                        uuid=request.uuid,
                        mbase_path=source_dir / "mbase.safetensors",
                        mask_policy=self.mask_policy,
                    )
                    metadata = self._metadata(request, latent, token_grid)
                    cache = {
                        "anchor_step": 10,
                        "model": metadata["model"],
                        "latent_token_grid": metadata["latent_token_grid"],
                        "latent_path": str(step10_path.resolve()),
                        "latent_sha256": _sha256(step10_path),
                        "sparse_attention_path": str(sparse_path.resolve()),
                        "sparse_attention_sha256": sparse_hash,
                    }
                    source_signature = self._write_signature(
                        source_dir,
                        step5_latent=step5_latent,
                        metadata=metadata,
                        masks=mbase_by_noun,
                        decoded_tail_frame=decoded_tail_frame,
                        cache=cache,
                    )
                    source_cached = True
                    break

        torch.cuda.synchronize()
        if match is not None and match.hit and not self.retain_artifacts:
            shutil.rmtree(source_dir)
            return OnlineCollectionResult(
                match, None, True, False, early_probe_s, signature_gate_s, 0.0, completed_steps, False,
                str(probe_latent_path) if 'probe_latent_path' in locals() else None,
                dit_forward_s,
            )
        if evaluate_probe is not None and match is not None:
            return OnlineCollectionResult(
                match, None, True, False, early_probe_s, signature_gate_s, 0.0,
                completed_steps, False, str(probe_latent_path), dit_forward_s,
            )
        if source_signature is None:
            raise RuntimeError("online source cache did not reach completed step 10")
        source_cache_s = 0.0 if self.assume_step10_cache else time.perf_counter() - source_started
        return OnlineCollectionResult(
            match,
            source_signature,
            evaluate_probe is not None,
            source_cached,
            early_probe_s,
            signature_gate_s,
            source_cache_s,
            completed_steps,
            self.assume_step10_cache,
            dit_forward_s=dit_forward_s,
        )


__all__ = ["NativeOnlineCollector", "OnlineCollectionResult", "SyntheticOnlineCollector"]
