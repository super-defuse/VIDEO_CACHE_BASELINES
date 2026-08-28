#!/usr/bin/env python3
"""Run the 60-pair Carnator Early Signature calibration workload on multiple GPUs.

The native Wan runner already owns the source trajectory and joint-reuse
implementation.  This experiment runner adds the calibration-specific
requirements around it: per-noun step-5 Early Signatures, one-to-one pair
mapping, resumable per-pair state, and one worker process per selected GPU.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import queue
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PAIRS = ROOT / "artifacts" / "early_signature_calibration_cluster_02_2170" / "pairs.csv"
DEFAULT_RUN_ROOT = (
    ROOT
    / "artifacts"
    / "early_signature_calibration_cluster_02_2170"
    / "runs"
    / "full_source_target"
)
DEFAULT_MODEL = Path("/home/tangxuebin/models/Wan2.2-TI2V-5B")
DEFAULT_GPUS = "0,1,2,3,5,6,7"
NOUN_EXTRACTOR = "spacy/en_core_web_sm:NOUN"


@dataclass(frozen=True)
class Pair:
    pair_id: str
    cached_uuid: str
    cached_prompt: str
    cached_nouns: tuple[str, ...]
    target_uuid: str
    target_prompt: str
    target_nouns: tuple[str, ...]

    @property
    def source_only_noun(self) -> str:
        values = sorted(set(self.cached_nouns) - set(self.target_nouns))
        if len(values) != 1:
            raise ValueError(f"{self.pair_id}: expected exactly one source-only noun, got {values}")
        return values[0]

    @property
    def target_only_noun(self) -> str:
        values = sorted(set(self.target_nouns) - set(self.cached_nouns))
        if len(values) != 1:
            raise ValueError(f"{self.pair_id}: expected exactly one target-only noun, got {values}")
        return values[0]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_pairs(path: Path) -> list[Pair]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "pair_id", "cached_uuid", "cached_prompt", "cached_nouns",
        "target_uuid", "target_prompt", "target_nouns",
    }
    if not rows or set(rows[0]) != required:
        raise ValueError(f"unexpected pairs.csv schema: {path}")
    pairs = [
        Pair(
            pair_id=str(row["pair_id"]),
            cached_uuid=str(row["cached_uuid"]),
            cached_prompt=str(row["cached_prompt"]),
            cached_nouns=tuple(json.loads(row["cached_nouns"])),
            target_uuid=str(row["target_uuid"]),
            target_prompt=str(row["target_prompt"]),
            target_nouns=tuple(json.loads(row["target_nouns"])),
        )
        for row in rows
    ]
    if len({pair.pair_id for pair in pairs}) != len(pairs):
        raise ValueError("pairs.csv has duplicate pair_id values")
    for pair in pairs:
        _ = pair.source_only_noun
        _ = pair.target_only_noun
        if len(set(pair.cached_nouns)) < 2 or len(set(pair.target_nouns)) < 2:
            raise ValueError(f"{pair.pair_id}: each side needs at least two nouns")
    return pairs


def parse_gpus(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(value < 0 for value in values) or len(set(values)) != len(values):
        raise ValueError("--gpus must be a non-empty, duplicate-free comma-separated GPU list")
    return values


def source_dir(run_root: Path, pair: Pair) -> Path:
    return run_root / "sources" / pair.pair_id


def target_dir(run_root: Path, pair: Pair) -> Path:
    return run_root / "targets" / pair.pair_id


def source_status_path(run_root: Path, pair: Pair) -> Path:
    return source_dir(run_root, pair) / "calibration_source.json"


def target_status_path(run_root: Path, pair: Pair) -> Path:
    return target_dir(run_root, pair) / "calibration_target.json"


def stage_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return read_json(path).get("status") == "completed"
    except (OSError, json.JSONDecodeError):
        return False


def native_args(args: argparse.Namespace, *, pair: Pair, phase: str, gpu: int):
    from carnator.adapters import wan_ti2v_5b as native

    native_args = native.build_parser().parse_args([])
    native_args.no_env_setup = True
    native_args.model_dir = args.model_dir.resolve()
    native_args.device = f"cuda:{gpu}"
    native_args.artifact_root = (args.run_root / ("sources" if phase == "source" else "targets")).resolve()
    native_args.results_root = (args.run_root / ("source_results" if phase == "source" else "target_results")).resolve()
    native_args.run_id = pair.pair_id
    native_args.source_prompt = pair.cached_prompt
    native_args.target_prompt = pair.target_prompt
    native_args.source_positive = pair.source_only_noun
    native_args.seed = args.seed
    native_args.steps = 50
    native_args.mask_probe_step = 5
    native_args.reuse_start_step = 10
    native_args.height = args.height
    native_args.width = args.width
    native_args.frames = args.frames
    native_args.fps = args.fps
    native_args.scheduler = args.scheduler
    native_args.solver_order = args.solver_order
    native_args.flow_shift = args.flow_shift
    native_args.guidance_scale = args.guidance_scale
    native_args.model_dtype = args.model_dtype
    native_args.convert_model_dtype = True
    native_args.offload_model = False
    native_args.t5_cpu = False
    native_args.disable_progress = True
    native_args.outer_attn_backend = "dense"
    native_args.ffn_backend = "fp"
    native_args.latent_injection_mode = "hard"
    native_args.capture_outer_block_masks = False
    native_args.source_only = phase == "source"
    native_args.source_run_dir = source_dir(args.run_root, pair) if phase == "target" else None
    return native_args


def signature_metadata(runtime, native_args: argparse.Namespace, prompt: str, latent, token_grid: tuple[int, int, int]) -> dict[str, Any]:
    from carnator.adapters import wan_ti2v_5b as native

    return {
        "prompt": prompt,
        "model": "Wan2.2-TI2V-5B",
        "scheduler": native_args.scheduler,
        "scheduler_config": {
            "flow_shift": float(native_args.flow_shift),
            "solver_order": int(native_args.solver_order) if native_args.scheduler == "unipc" else None,
            "total_steps": int(native_args.steps),
        },
        "cfg": float(native_args.guidance_scale),
        "probe_seed": int(native_args.seed),
        "completed_step": 5,
        "latent_shape": list(latent.shape),
        "latent_token_grid": list(token_grid),
        "noun_extractor": NOUN_EXTRACTOR,
        "mask_policy": native.fused_mbase_policy(native_args),
    }


def capture_early_signature(
    runtime,
    native_args: argparse.Namespace,
    *,
    prompt: str,
    nouns: tuple[str, ...],
    attention_input,
    step5_latent,
    directory: Path,
    cache: dict[str, Any] | None,
) -> Path:
    """Build a completed-step-5 signature from the native fused Mbase path."""

    import torch

    from carnator.adapters import wan_ti2v_5b as native
    from carnator.early_signature import write_early_signature

    if len(set(nouns)) != len(nouns):
        raise ValueError(f"duplicate signature nouns for prompt: {prompt!r}")
    token_grid = native.token_grid_for(step5_latent, runtime.pipe.patch_size)
    indices = {
        noun: native.find_token_indices(runtime.pipe.text_encoder, prompt, noun, runtime.pipe.config.text_len)
        for noun in sorted(nouns)
    }
    probe = native.NativeMultiNounCrossAttentionProbe(
        runtime.pipe.model, native.parse_int_list(native_args.mask_layers), indices, token_grid
    )
    context, _ = native.encode_prompts(runtime, native_args, prompt)
    scheduler = native.build_scheduler(runtime, native_args)
    step_index = native_args.mask_probe_step - 1
    timestep = scheduler.timesteps[step_index]
    seq_len = native.seq_len_for(runtime, attention_input)
    timestep_input = native.timestep_for(runtime, attention_input, timestep, seq_len)
    model = runtime.pipe.model
    if native_args.offload_model or runtime.pipe.init_on_cpu:
        model.to(runtime.device)
        torch.cuda.empty_cache()
    no_sync = getattr(model, "no_sync", __import__("contextlib").nullcontext)
    with torch.amp.autocast("cuda", dtype=runtime.pipe.param_dtype), torch.no_grad(), no_sync(), probe:
        probe.enabled = True
        probe.current_step_index = step_index
        _ = model([attention_input], t=timestep_input, context=context, seq_len=seq_len)[0]
        torch.cuda.synchronize()
    if not probe.records_by_noun or any(not records for records in probe.records_by_noun.values()):
        raise RuntimeError("step-5 multi-noun attention probe produced incomplete records")
    fused_mbase_by_noun = native.build_fused_mbase_by_noun(
        runtime,
        native_args,
        prompt=prompt,
        token_grid=token_grid,
        noun_token_indices=indices,
        cross_records_by_noun=probe.records_by_noun,
        self_attention_input=step5_latent,
    )
    with torch.amp.autocast("cuda", dtype=runtime.pipe.param_dtype), torch.no_grad():
        decoded = runtime.pipe.vae.decode([step5_latent])[0]
        torch.cuda.synchronize()
    if decoded.ndim != 4 or decoded.shape[0] < 3 or decoded.shape[1] < 1:
        raise RuntimeError("Wan VAE did not return a [C,T,H,W] video for the step-5 signature")
    result = write_early_signature(
        directory,
        step5_latent=step5_latent.detach().float().cpu().contiguous(),
        decoded_tail_frame=decoded[:, -1].detach(),
        metadata=signature_metadata(runtime, native_args, prompt, step5_latent, token_grid),
        mbase_by_noun=fused_mbase_by_noun,
        cache=cache,
    )
    return Path(result["signature_path"])


def run_target_probe(runtime, native_args: argparse.Namespace, pair: Pair, directory: Path) -> Path:
    """Run the target's five-step probe under the full 50-step scheduler."""

    import torch

    from carnator.adapters import wan_ti2v_5b as native

    prompt = pair.target_prompt
    context, context_null = native.encode_prompts(runtime, native_args, prompt)
    latent, generator = native.prepare_initial_latent(runtime, native_args)
    scheduler = native.build_scheduler(runtime, native_args)
    token_grid = native.token_grid_for(latent, runtime.pipe.patch_size)
    indices = {
        noun: native.find_token_indices(runtime.pipe.text_encoder, prompt, noun, runtime.pipe.config.text_len)
        for noun in sorted(pair.target_nouns)
    }
    probe = native.NativeMultiNounCrossAttentionProbe(
        runtime.pipe.model, native.parse_int_list(native_args.mask_layers), indices, token_grid
    )
    model = runtime.pipe.model
    if native_args.offload_model or runtime.pipe.init_on_cpu:
        model.to(runtime.device)
        torch.cuda.empty_cache()
    no_sync = getattr(model, "no_sync", __import__("contextlib").nullcontext)
    attention_input = None
    with torch.amp.autocast("cuda", dtype=runtime.pipe.param_dtype), torch.no_grad(), no_sync(), probe:
        for step_index, timestep in enumerate(scheduler.timesteps):
            if step_index >= native_args.mask_probe_step:
                break
            probe.enabled = (step_index + 1) == native_args.mask_probe_step
            probe.current_step_index = step_index
            if probe.enabled:
                attention_input = latent.detach().clone()
            seq_len = native.seq_len_for(runtime, latent)
            timestep_input = native.timestep_for(runtime, latent, timestep, seq_len)
            cond = model([latent], t=timestep_input, context=context, seq_len=seq_len)[0]
            if native_args.guidance_scale == 1.0:
                noise_pred = cond
            else:
                uncond = model([latent], t=timestep_input, context=context_null, seq_len=seq_len)[0]
                noise_pred = uncond + native_args.guidance_scale * (cond - uncond)
            latent = scheduler.step(
                noise_pred.unsqueeze(0), timestep, latent.unsqueeze(0), return_dict=False, generator=generator
            )[0].squeeze(0)
        torch.cuda.synchronize()
    if attention_input is None:
        raise RuntimeError("target step-5 probe did not capture its pre-step latent")
    # Cross scores were captured during step 5; the final Mbase is fused with
    # one self-attention contrast pass on the completed step-5 latent.
    from carnator.early_signature import write_early_signature

    if not probe.records_by_noun or any(not records for records in probe.records_by_noun.values()):
        raise RuntimeError("target step-5 multi-noun attention probe produced incomplete records")
    fused_mbase_by_noun = native.build_fused_mbase_by_noun(
        runtime,
        native_args,
        prompt=prompt,
        token_grid=token_grid,
        noun_token_indices=indices,
        cross_records_by_noun=probe.records_by_noun,
        self_attention_input=latent,
    )
    with torch.amp.autocast("cuda", dtype=runtime.pipe.param_dtype), torch.no_grad():
        decoded = runtime.pipe.vae.decode([latent])[0]
        torch.cuda.synchronize()
    result = write_early_signature(
        directory,
        step5_latent=latent.detach().float().cpu().contiguous(),
        decoded_tail_frame=decoded[:, -1].detach(),
        metadata=signature_metadata(runtime, native_args, pair.target_prompt, latent, token_grid),
        mbase_by_noun=fused_mbase_by_noun,
        cache=None,
    )
    return Path(result["signature_path"])


def source_cache_payload(native_args: argparse.Namespace, cache_dir: Path, token_grid: list[int]) -> dict[str, Any]:
    latent_path = cache_dir / f"latents_step_{native_args.reuse_start_step - 1:03d}.pt"
    sparse_path = cache_dir / "fusion_masks" / "masks.pt"
    for path in (latent_path, sparse_path):
        if not path.is_file():
            raise FileNotFoundError(f"source cache artifact is missing: {path}")
    return {
        "anchor_step": native_args.reuse_start_step,
        "model": "Wan2.2-TI2V-5B",
        "latent_token_grid": token_grid,
        "latent_path": str(latent_path.resolve()),
        "latent_sha256": sha256(latent_path),
        "sparse_attention_path": str(sparse_path.resolve()),
        "sparse_attention_sha256": sha256(sparse_path),
    }


def run_source_worker(args: argparse.Namespace, pair: Pair, gpu: int) -> dict[str, Any]:
    import torch

    from carnator.adapters import wan_ti2v_5b as native

    run_dir = source_dir(args.run_root, pair)
    status_path = source_status_path(args.run_root, pair)
    if stage_complete(status_path):
        return read_json(status_path)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"incomplete source directory exists: {run_dir}")
    native_args = native_args_for_worker(args, pair, "source", gpu)
    started_at = utc_now()
    started = time.perf_counter()
    native.ensure_import_paths(native_args)
    native.validate_experiment_config(native_args)
    runtime = native.load_runtime(native_args)
    try:
        paths = native.stage_paths(native_args)
        source = native.run_native_t2v_stage(
            runtime,
            native_args,
            label="source_dense",
            prompt=pair.cached_prompt,
            out_video=paths["source_video"],
            cache_dir=paths["source_cache"],
            capture_cross_seed_scores=True,
            save_trajectory=True,
            cross_seed_output_dir=paths["cross_seed_dir"],
            decode_video=True,
        )
        trajectory = {
            int(item["step_index"]): Path(item["path"])
            for item in source["trajectory"]
            if item.get("kind") == "post_scheduler"
        }
        self_bundle = native.build_native_self_contrast_mask_bundle(
            runtime,
            native_args,
            source_trajectory=trajectory,
            cross_seed_dir=paths["cross_seed_dir"],
            self_mask_dir=paths["self_mask_dir"],
        )
        native.build_native_fusion_mask_bundle(
            runtime,
            native_args,
            source_trajectory=trajectory,
            cross_seed_dir=paths["cross_seed_dir"],
            self_bundle=self_bundle,
            self_mask_dir=paths["self_mask_dir"],
            fusion_mask_dir=paths["mask_dir"],
        )
        probe_input = trajectory[native_args.mask_probe_step - 2]
        step5_path = trajectory[native_args.mask_probe_step - 1]
        step5_latent = native.load_latent(step5_path, runtime.device, torch.float32)
        attention_input = native.load_latent(probe_input, runtime.device, torch.float32)
        cache = source_cache_payload(native_args, paths["source_cache"], list(native.token_grid_for(step5_latent, runtime.pipe.patch_size)))
        signature_path = capture_early_signature(
            runtime,
            native_args,
            prompt=pair.cached_prompt,
            nouns=pair.cached_nouns,
            attention_input=attention_input,
            step5_latent=step5_latent,
            directory=paths["source_cache"],
            cache=cache,
        )
        result = {
            "status": "completed",
            "completed_at": utc_now(),
            "pair_id": pair.pair_id,
            "gpu": gpu,
            "source": {
                "uuid": pair.cached_uuid,
                "prompt": pair.cached_prompt,
                "nouns": list(pair.cached_nouns),
                "source_only_noun": pair.source_only_noun,
                "video": source["output"],
                "cache_dir": str(paths["source_cache"]),
                "manifest": source["manifest_path"],
                "signature": str(signature_path),
                "cache_artifacts": cache,
            },
            "timing": {
                "started_at": started_at,
                "wall_clock_seconds": time.perf_counter() - started,
                "source_stage": source["timing"],
            },
        }
        write_json(status_path, result)
        return result
    finally:
        runtime.profiler.close()
        torch.cuda.empty_cache()


def native_args_for_worker(args: argparse.Namespace, pair: Pair, phase: str, gpu: int):
    return native_args(args, pair=pair, phase=phase, gpu=gpu)


def run_target_worker(args: argparse.Namespace, pair: Pair, gpu: int) -> dict[str, Any]:
    import torch

    from contextlib import ExitStack

    from carnator.adapters import wan_ti2v_5b as native
    from carnator.masks import load_mask_bundle

    source_state = source_status_path(args.run_root, pair)
    if not stage_complete(source_state):
        raise RuntimeError(f"target cannot start before its source is complete: {source_state}")
    run_dir = target_dir(args.run_root, pair)
    status_path = target_status_path(args.run_root, pair)
    if stage_complete(status_path):
        return read_json(status_path)
    native_args = native_args_for_worker(args, pair, "target", gpu)
    started_at = utc_now()
    started = time.perf_counter()
    native.ensure_import_paths(native_args)
    native.validate_experiment_config(native_args)
    runtime = native.load_runtime(native_args)
    try:
        paths = native.stage_paths(native_args)
        existing_signature = paths["target_reuse_cache"] / "online_color_signature.json"
        target_manifest = paths["target_reuse_cache"] / "manifest.json"
        target_video = paths["target_reuse_video"]
        if run_dir.exists() and any(run_dir.iterdir()):
            # A failure after the five-step probe but before reuse is safe to
            # resume: the sidecar is self-contained and its cache is null.
            if not existing_signature.is_file() or target_manifest.exists() or target_video.exists():
                raise FileExistsError(f"incomplete target directory needs inspection: {run_dir}")
            target_signature = existing_signature
            target_probe_seconds = None
        else:
            probe_started = time.perf_counter()
            target_signature = run_target_probe(runtime, native_args, pair, paths["target_reuse_cache"])
            target_probe_seconds = time.perf_counter() - probe_started
        source, source_trajectory, source_mask_dir, _source_outer_mask_dir = native.source_assets_from_run(
            native_args, source_dir(args.run_root, pair)
        )
        expected_sampling = {
            "steps": int(native_args.steps),
            "scheduler": native_args.scheduler,
            "solver_order": int(native_args.solver_order) if native_args.scheduler == "unipc" else None,
            "flow_shift": float(native_args.flow_shift),
            "guidance_scale": float(native_args.guidance_scale),
        }
        actual_sampling = {key: source.get(key) for key in expected_sampling}
        if actual_sampling != expected_sampling:
            raise ValueError(
                f"{pair.pair_id}: source sampling configuration differs: "
                f"expected={expected_sampling}, actual={actual_sampling}"
            )
        mask_bundle = load_mask_bundle(source_mask_dir, None)
        sparse_state = native.NativeSparseState(
            model_module=runtime.wan["model_module"],
            mask_bundle=mask_bundle,
            layers=native.parse_int_ranges(native_args.sparse_layers),
        )
        resume_latent_path = source_trajectory[native_args.reuse_start_step - 1]
        with ExitStack() as stack:
            stack.enter_context(native.NativeSparseBlockHooks(runtime.pipe.model, sparse_state))
            target = native.run_native_t2v_stage(
                runtime,
                native_args,
                label="target_latent_reuse",
                prompt=pair.target_prompt,
                out_video=paths["target_reuse_video"],
                cache_dir=paths["target_reuse_cache"],
                resume_latent_path=resume_latent_path,
                resume_start_step=native_args.reuse_start_step,
                source_trajectory=source_trajectory,
                mask_bundle=mask_bundle,
                sparse_state=sparse_state,
            )
        source_state_payload = read_json(source_state)
        reuse_inputs = {
            "pair_id": pair.pair_id,
            "source_run_dir": str(source_dir(args.run_root, pair)),
            "source_cache": source_state_payload["source"]["cache_dir"],
            "source_signature": source_state_payload["source"]["signature"],
            "source_fusion_masks": str(source_mask_dir),
            "target_signature": str(target_signature),
            "reuse_start_step": native_args.reuse_start_step,
        }
        write_json(paths["target_reuse_cache"] / "reuse_inputs.json", reuse_inputs)
        result = {
            "status": "completed",
            "completed_at": utc_now(),
            "pair_id": pair.pair_id,
            "gpu": gpu,
            "target": {
                "uuid": pair.target_uuid,
                "prompt": pair.target_prompt,
                "nouns": list(pair.target_nouns),
                "target_only_noun": pair.target_only_noun,
                "video": target["output"],
                "cache_dir": str(paths["target_reuse_cache"]),
                "manifest": target["manifest_path"],
                "signature": str(target_signature),
                "reuse_inputs": str(paths["target_reuse_cache"] / "reuse_inputs.json"),
            },
            "timing": {
                "started_at": started_at,
                "wall_clock_seconds": time.perf_counter() - started,
                "target_early_probe_seconds": target_probe_seconds,
                "target_reuse_stage": target["timing"],
            },
        }
        write_json(status_path, result)
        return result
    finally:
        runtime.profiler.close()
        torch.cuda.empty_cache()


def worker_main(args: argparse.Namespace) -> int:
    pair = next(pair for pair in load_pairs(args.pairs) if pair.pair_id == args.pair_id)
    result = run_source_worker(args, pair, args.gpu) if args.worker == "source" else run_target_worker(args, pair, args.gpu)
    print(json.dumps({"pair_id": pair.pair_id, "phase": args.worker, "status": result["status"]}, ensure_ascii=False))
    return 0


def ensure_manifest(args: argparse.Namespace, pairs: list[Pair]) -> None:
    path = args.run_root / "experiment_manifest.json"
    pair_hash = sha256(args.pairs)
    current = {
        "schema_version": "carnator-early-signature-calibration-run",
        "created_at": utc_now(),
        "pairs_csv": str(args.pairs.resolve()),
        "pairs_sha256": pair_hash,
        "pair_count": len(pairs),
        "configuration": {
            "model_dir": str(args.model_dir.resolve()),
            "seed": args.seed,
            "scheduler": args.scheduler,
            "solver_order": args.solver_order,
            "flow_shift": args.flow_shift,
            "guidance_scale": args.guidance_scale,
            "steps": 50,
            "reuse_start_step": 10,
            "signature_completed_step": 5,
            "shape": {"height": args.height, "width": args.width, "frames": args.frames, "fps": args.fps},
            "early_signature": "step5-tail-lab-ciede2000",
            "noun_extractor": NOUN_EXTRACTOR,
        },
        "quality_thresholds": {
            "q_min": args.q_min,
            "clip_t_min": args.clip_t_min,
            "coverage_min": args.coverage_min,
            "confidence": args.confidence,
        },
    }
    if path.exists():
        previous = read_json(path)
        if previous.get("pairs_sha256") != pair_hash:
            raise ValueError(f"run root was initialized with a different pairs.csv: {path}")
        old_config = previous.get("configuration")
        if old_config != current["configuration"]:
            raise ValueError(f"run root was initialized with a different generation configuration: {path}")
        if any(value is not None for value in current["quality_thresholds"].values()):
            previous["quality_thresholds"] = current["quality_thresholds"]
            previous["updated_at"] = utc_now()
            write_json(path, previous)
        return
    write_json(path, current)


def require_target_thresholds(args: argparse.Namespace) -> None:
    values = (args.q_min, args.clip_t_min, args.coverage_min, args.confidence)
    if any(value is None for value in values):
        raise ValueError(
            "target phase requires --q-min, --clip-t-min, --coverage-min, and --confidence "
            "so quality criteria are frozen before target generation"
        )


def backfill_stage_timings(run_root: Path, pairs: list[Pair]) -> None:
    """Preserve native stage timings for workers completed before timing support."""

    for pair in pairs:
        for phase, status_path in (
            ("source", source_status_path(run_root, pair)),
            ("target", target_status_path(run_root, pair)),
        ):
            if not stage_complete(status_path):
                continue
            status = read_json(status_path)
            if "timing" in status:
                continue
            stage = status[phase]
            manifest_path = Path(stage["manifest"])
            if not manifest_path.is_file():
                continue
            native_manifest = read_json(manifest_path)
            status["timing"] = {
                "backfilled_at": utc_now(),
                "wall_clock_seconds": None,
                f"{phase}_stage": native_manifest.get("timing"),
                "note": "worker wall-clock timing was not recorded before timing support was added",
            }
            write_json(status_path, status)


def worker_command(args: argparse.Namespace, pair: Pair, phase: str, gpu: int) -> list[str]:
    command = [
        str(args.runtime_python), str(Path(__file__).resolve()), "--worker", phase,
        "--pair-id", pair.pair_id, "--gpu", str(gpu), "--pairs", str(args.pairs),
        "--run-root", str(args.run_root), "--model-dir", str(args.model_dir),
        "--seed", str(args.seed), "--height", str(args.height), "--width", str(args.width),
        "--frames", str(args.frames), "--fps", str(args.fps), "--scheduler", args.scheduler,
        "--solver-order", str(args.solver_order), "--flow-shift", str(args.flow_shift),
        "--guidance-scale", str(args.guidance_scale), "--model-dtype", args.model_dtype,
    ]
    return command


def controller_main(args: argparse.Namespace) -> int:
    pairs = load_pairs(args.pairs)
    if args.limit is not None:
        pairs = pairs[:args.limit]
    gpus = parse_gpus(args.gpus)
    args.run_root.mkdir(parents=True, exist_ok=True)
    ensure_manifest(args, load_pairs(args.pairs))
    backfill_stage_timings(args.run_root, pairs)
    phases = [args.phase] if args.phase != "all" else ["source", "target"]
    for phase in phases:
        if phase == "target":
            require_target_thresholds(args)
        status = source_status_path if phase == "source" else target_status_path
        pending = [pair for pair in pairs if not stage_complete(status(args.run_root, pair))]
        if not pending:
            print(f"phase={phase} pending=0")
            continue
        print(f"phase={phase} pending={len(pending)} gpus={gpus}", flush=True)
        jobs: queue.Queue[Pair] = queue.Queue()
        for pair in pending:
            jobs.put(pair)
        failures: list[tuple[str, int]] = []
        lock = threading.Lock()

        def gpu_worker(gpu: int) -> None:
            while True:
                try:
                    pair = jobs.get_nowait()
                except queue.Empty:
                    return
                log_path = args.run_root / "logs" / phase / f"{pair.pair_id}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                environment = os.environ.copy()
                environment["CARNATOR_ENV_READY"] = "1"
                environment["PYTHONPATH"] = os.pathsep.join(
                    (str(ROOT), str(ROOT / "external" / "wan2.2"), environment.get("PYTHONPATH", ""))
                )
                with log_path.open("w", encoding="utf-8") as log:
                    process = subprocess.run(
                        worker_command(args, pair, phase, gpu), cwd=ROOT, env=environment,
                        stdout=log, stderr=subprocess.STDOUT, check=False,
                    )
                jobs.task_done()
                if process.returncode:
                    with lock:
                        failures.append((pair.pair_id, process.returncode))
                    print(f"phase={phase} pair={pair.pair_id} gpu={gpu} failed log={log_path}", flush=True)
                else:
                    print(f"phase={phase} pair={pair.pair_id} gpu={gpu} completed", flush=True)

        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            futures = [pool.submit(gpu_worker, gpu) for gpu in gpus]
            for future in futures:
                future.result()
        if failures:
            pairs_text = ", ".join(f"{pair_id} (exit {code})" for pair_id, code in failures)
            raise RuntimeError(f"{phase} phase failed: {pairs_text}")
    return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    value.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    value.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    value.add_argument("--runtime-python", type=Path, default=Path(sys.executable))
    value.add_argument("--phase", choices=("source", "target", "all"), default="source")
    value.add_argument("--gpus", default=DEFAULT_GPUS)
    value.add_argument("--limit", type=int)
    value.add_argument("--seed", type=int, default=20260218)
    value.add_argument("--height", type=int, default=704)
    value.add_argument("--width", type=int, default=1280)
    value.add_argument("--frames", type=int, default=121)
    value.add_argument("--fps", type=int, default=24)
    value.add_argument("--scheduler", choices=("unipc", "dpm++"), default="unipc")
    value.add_argument("--solver-order", type=int, default=2)
    value.add_argument("--flow-shift", type=float, default=5.0)
    value.add_argument("--guidance-scale", type=float, default=5.0)
    value.add_argument("--model-dtype", choices=("fp16", "bf16"), default="fp16")
    value.add_argument("--q-min", type=float)
    value.add_argument("--clip-t-min", type=float)
    value.add_argument("--coverage-min", type=float)
    value.add_argument("--confidence", type=float)
    value.add_argument("--worker", choices=("source", "target"), help=argparse.SUPPRESS)
    value.add_argument("--pair-id", help=argparse.SUPPRESS)
    value.add_argument("--gpu", type=int, help=argparse.SUPPRESS)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    args.pairs = args.pairs.expanduser().resolve()
    args.run_root = args.run_root.expanduser().resolve()
    args.model_dir = args.model_dir.expanduser().resolve()
    args.runtime_python = args.runtime_python.expanduser().resolve()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if not args.pairs.is_file():
        raise FileNotFoundError(f"pairs.csv not found: {args.pairs}")
    if not args.model_dir.is_dir():
        raise FileNotFoundError(f"model directory not found: {args.model_dir}")
    if not args.runtime_python.is_file():
        raise FileNotFoundError(f"runtime Python not found: {args.runtime_python}")
    if args.worker:
        if not args.pair_id or args.gpu is None:
            raise ValueError("worker mode needs --pair-id and --gpu")
        return worker_main(args)
    return controller_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
