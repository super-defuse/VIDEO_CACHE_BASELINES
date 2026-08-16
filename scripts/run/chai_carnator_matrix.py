#!/usr/bin/env python3
"""Run the CHAI-Carnator hybrid baseline on native Wan2.2-TI2V-5B.

The source and target requests both use the eight-step express schedule.  The
source request stores the full eight-step latent trajectory, Carnator masks
probed at completed step five, and CHAI block-0 transformer inputs for local
steps 1--4.  Each target variant reuses the same source artifacts while
applying a Carnator B0--B5 backend to blocks outside block 0.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import types
from contextlib import AbstractContextManager, ExitStack
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
CARNATOR_ROOT = ROOT.parent / "CARNATOR_ICLR2027"
if str(CARNATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(CARNATOR_ROOT))


NAME = "CHAI-Carnator"
CC_CACHE_STEPS = (1, 2, 3, 4)
CC_STEPS = tuple(range(1, 9))
CC_MASK_PROBE_STEP = 5
CC_REGION_LAYERS = "10-20"
# Block 0 is reserved for CHAI.  The remaining outer layers can use the
# Carnator B1/B2/B4/B5 sparse backends without competing attention patches.
CC_OUTER_LAYERS = "1-9,21-29"


def _external():
    from carnator.adapters import wan_ti2v_5b as native

    return native


class ChaiBlock0Hooks(AbstractContextManager):
    """Capture/read CHAI block-0 tokens around the native Wan loop.

    CHAI's public Wan adapter identifies denoising calls from the pair of CFG
    forwards and uses zero-based local indices.  We keep that convention here:
    configured steps 1--4 are the same four calls used by the existing CHAI
    implementation, while Carnator's completed-step masks remain one-based.
    """

    def __init__(
        self,
        model: Any,
        model_module: Any,
        *,
        capture: bool = False,
        cached_tokens: dict[int, Any] | None = None,
        cache_steps: Iterable[int] = CC_CACHE_STEPS,
    ) -> None:
        self.model = model
        self.model_module = model_module
        self.capture = bool(capture)
        self.cached_tokens = cached_tokens or {}
        self.cache_steps = set(int(step) for step in cache_steps)
        self.captures: dict[int, Any] = {}
        self.calls = 0
        self.current_step = 0
        self.current_branch = "cond"
        self.active = False
        self.cache_attention_calls = 0
        self.cache_attention_steps: set[int] = set()
        self._model_original: Any = None
        self._attention_original: Any = None
        self._attention: Any = None

    def __enter__(self) -> "ChaiBlock0Hooks":
        import torch

        block = self.model.blocks[0]
        attention = block.self_attn
        self._attention = attention
        self._attention_original = attention.forward

        def patched(x, seq_lens, grid_sizes, freqs):
            step = self.current_step
            if self.capture and self.current_branch == "cond" and step in self.cache_steps:
                self.captures.setdefault(step, x.detach().cpu())
            if not self.active:
                return self._attention_original(x, seq_lens, grid_sizes, freqs)
            cached = self.cached_tokens.get(step)
            if cached is None:
                raise KeyError(f"{NAME} target cache has no block-0 tokens for local step {step}")
            self.cache_attention_calls += 1
            self.cache_attention_steps.add(step)
            cached = cached.to(device=x.device, dtype=x.dtype)
            batch, tokens = x.shape[:2]
            heads, head_dim = attention.num_heads, attention.head_dim
            query = attention.norm_q(attention.q(x)).view(batch, tokens, heads, head_dim)
            key = attention.norm_k(attention.k(x)).view(batch, tokens, heads, head_dim)
            value = attention.v(x).view(batch, tokens, heads, head_dim)
            cached_key = attention.norm_k(attention.k(cached)).view(batch, -1, heads, head_dim)
            cached_value = attention.v(cached).view(batch, -1, heads, head_dim)
            query = self.model_module.rope_apply(query, grid_sizes, freqs)
            key = self.model_module.rope_apply(key, grid_sizes, freqs)
            cached_key = self.model_module.rope_apply(cached_key, grid_sizes, freqs)
            output = self.model_module.flash_attention(
                query,
                torch.cat((key, cached_key), dim=1),
                torch.cat((value, cached_value), dim=1),
            )
            return attention.o(output.flatten(2))

        # An instance attribute is intentionally a plain callable, matching
        # CHAI's adapter and avoiding a second implicit ``self`` argument.
        attention.forward = patched
        self._model_original = self.model.forward

        def scheduled_forward(*args, **kwargs):
            self.current_step = self.calls // 2
            self.current_branch = "cond" if self.calls % 2 == 0 else "uncond"
            self.active = (not self.capture) and self.current_step in self.cache_steps
            try:
                return self._model_original(*args, **kwargs)
            finally:
                self.calls += 1

        self.model.forward = scheduled_forward
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._attention is not None:
            self._attention.forward = self._attention_original
        if self._model_original is not None:
            self.model.forward = self._model_original
        return False

    def summary(self) -> dict[str, Any]:
        return {
            "capture": self.capture,
            "cache_steps": sorted(self.cache_steps),
            "captured_steps": sorted(self.captures),
            "cache_attention_steps": sorted(self.cache_attention_steps),
            "cache_attention_calls": self.cache_attention_calls,
            "model_forward_calls": self.calls,
        }


def _load_tokens(path: Path) -> dict[int, Any]:
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"CHAI token cache must be a dict, got {type(payload).__name__}")
    return {int(key): value for key, value in payload.items()}


def _write_tokens(path: Path, tokens: dict[int, Any]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({int(step): value.cpu() for step, value in sorted(tokens.items())}, path)


def _parser() -> argparse.ArgumentParser:
    native = _external()
    parser = native.build_parser()
    parser.description = f"Run the {NAME} Wan2.2 B0-B5 matrix."
    parser.set_defaults(
        steps=8,
        reuse_start_step=0,
        mask_probe_step=CC_MASK_PROBE_STEP,
        sparse_layers=CC_REGION_LAYERS,
        outer_sparse_layers=CC_OUTER_LAYERS,
        capture_outer_block_masks=True,
        run_id=os.environ.get("CHAI_CARNATOR_RUN_ID", "real-wan-chai-carnator-b0-b5"),
    )
    parser.add_argument(
        "--matrix-output",
        type=Path,
        help="Write the completed matrix JSON here (defaults below --results-root).",
    )
    parser.add_argument(
        "--skip-dense-reference",
        action="store_true",
        help="Only for protocol smoke tests; a real matrix requires the dense reference.",
    )
    return parser


def _validate(args: argparse.Namespace) -> None:
    native = _external()
    if args.steps != 8:
        raise ValueError(f"{NAME} requires exactly 8 steps, got {args.steps}")
    if args.mask_probe_step != CC_MASK_PROBE_STEP:
        raise ValueError(f"{NAME} calibrates masks at completed step {CC_MASK_PROBE_STEP}")
    if args.reuse_start_step != 0:
        raise ValueError(f"{NAME} starts target from the source initial latent; reuse-start-step must be 0")
    if args.guidance_scale == 1.0:
        raise ValueError(f"{NAME} requires CFG so block-0 cache steps can be paired with conditional source tokens")
    if native.parse_int_ranges(args.sparse_layers) != native.parse_int_ranges(CC_REGION_LAYERS):
        raise ValueError(f"{NAME} reserves Carnator region layers {CC_REGION_LAYERS}")
    if native.parse_int_ranges(args.outer_sparse_layers) != native.parse_int_ranges(CC_OUTER_LAYERS):
        raise ValueError(f"{NAME} reserves outer layers {CC_OUTER_LAYERS}")
    native.validate_experiment_config(args)


def _variant_specs() -> list[dict[str, str]]:
    return [
        {"baseline": "CC-B0", "outer_attention": "dense", "ffn": "fp"},
        {"baseline": "CC-B1", "outer_attention": "flashinfer", "ffn": "fp"},
        {"baseline": "CC-B2", "outer_attention": "sparge-sage2", "ffn": "fp"},
        {"baseline": "CC-B3", "outer_attention": "dense", "ffn": "w8a8"},
        {"baseline": "CC-B4", "outer_attention": "flashinfer", "ffn": "w8a8"},
        {"baseline": "CC-B5", "outer_attention": "sparge-sage2", "ffn": "w8a8"},
    ]


def _run_target_variant(runtime: Any, args: argparse.Namespace, source: dict[str, Any], mask_bundle: dict[str, Any], tokens: dict[int, Any], source_outer_dir: Path, spec: dict[str, str], artifact_run: Path) -> dict[str, Any]:
    import torch

    native = _external()
    from carnator.adapters.wan_ti2v_5b import NativeSparseBlockHooks, NativeSparseState
    from carnator.outer_sparse import OuterSparseAttentionHooks, OuterSparseExecutor, OuterSparseState, SourceBlockMaskCache, cfg_branches
    from carnator.quantized_ffn import QuantizedFFNHooks

    baseline = spec["baseline"]
    output = artifact_run / "videos" / f"{baseline}.mp4"
    cache_dir = artifact_run / "cache" / baseline
    source_trajectory = {
        int(item["step_index"]): Path(item["path"])
        for item in source.get("trajectory", [])
        if item.get("kind") == "post_scheduler"
    }
    source_initial = Path(source["cache_dir"]) / "latents_initial.pt"
    sparse_state = NativeSparseState(
        model_module=runtime.wan["model_module"],
        mask_bundle=mask_bundle,
        layers=native.parse_int_ranges(CC_REGION_LAYERS),
    )
    outer_state = None
    outer_executor = None
    if spec["outer_attention"] != "dense":
        outer_cache = SourceBlockMaskCache(source_outer_dir, native.outer_mask_config(args))
        outer_cache.expect_cfg_branches(cfg_branches(args.guidance_scale))
        outer_state = OuterSparseState(cfg_branch_count=cfg_branches(args.guidance_scale), start_completed_step=1)
        outer_executor = OuterSparseExecutor(
            outer_cache,
            spec["outer_attention"],
            flashinfer_wrapper=args.flashinfer_wrapper,
            workspace_mib=args.outer_workspace_mib,
            execution_dtype=native.outer_sparse_execution_dtype(args),
            use_fp16_qk_reduction=args.use_fp16_qk_reduction,
            flashinfer_plan_cache=args.flashinfer_plan_cache,
            sparge_smooth_k=args.sparge_smooth_k,
            sparge_pv_threshold=args.sparge_pv_threshold,
        )

    with ExitStack() as stack:
        ffn_state = None
        if spec["ffn"] == "w8a8":
            ffn_hooks = QuantizedFFNHooks(runtime.pipe.model)
            ffn_state = stack.enter_context(ffn_hooks)
            ffn_hooks.attach_profiler(runtime.profiler)
            if args.w8a8_verify_tokens:
                ffn_hooks.verify(args.w8a8_verify_tokens)
        stack.enter_context(NativeSparseBlockHooks(runtime.pipe.model, sparse_state))
        if outer_state is not None and outer_executor is not None:
            stack.enter_context(
                OuterSparseAttentionHooks(
                    runtime.pipe.model,
                    runtime.wan["model_module"],
                    outer_state,
                    outer_executor,
                    native.parse_int_ranges(CC_OUTER_LAYERS),
                )
            )
        chai_hooks = stack.enter_context(
            ChaiBlock0Hooks(
                runtime.pipe.model,
                runtime.wan["model_module"],
                cached_tokens=tokens,
            )
        )
        manifest = native.run_native_t2v_stage(
            runtime,
            args,
            label=baseline,
            prompt=args.target_prompt,
            out_video=output,
            cache_dir=cache_dir,
            resume_latent_path=source_initial,
            resume_start_step=0,
            source_trajectory=source_trajectory,
            mask_bundle=mask_bundle,
            sparse_state=sparse_state,
            outer_sparse_state=outer_state,
            ffn_state=ffn_state,
        )
    manifest["hybrid"] = {
        "name": NAME,
        "baseline": baseline,
        "source_steps": 8,
        "target_steps": 8,
        "block0": "CHAI cache attention",
        "block0_cache_steps": list(CC_CACHE_STEPS),
        "carnator_region_layers": native.parse_int_ranges(CC_REGION_LAYERS),
        "outer_attention_backend": spec["outer_attention"],
        "ffn_backend": spec["ffn"],
        "fixed_mask_completed_step": CC_MASK_PROBE_STEP,
        "fixed_latent_mask_source": CC_MASK_PROBE_STEP,
        "latent_injection_calls_expected": 8,
        "block0_cache_attention": chai_hooks.summary(),
    }
    native.write_json(cache_dir / "manifest.json", manifest)
    return manifest


def run_matrix(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    native = _external()
    from carnator.masks import save_mask_bundle
    from carnator.outer_sparse import OuterSparseCaptureHooks, OuterSparseState, SourceBlockMaskCache, cfg_branches

    _validate(args)
    paths = native.stage_paths(args)
    artifact_run = paths["artifact_run"]
    result_run = paths["result_run"]
    if args.clean:
        if artifact_run.exists():
            shutil.rmtree(artifact_run)
        if result_run.exists():
            shutil.rmtree(result_run)
    artifact_run.mkdir(parents=True, exist_ok=True)
    result_run.mkdir(parents=True, exist_ok=True)
    matrix_output = (args.matrix_output or (result_run / "matrix.json")).expanduser().resolve()

    if args.dry_run:
        matrix = {
            "status": "dry_run",
            "name": NAME,
            "method": "chai-carnator",
            "run_id": args.run_id,
            "steps": 8,
            "protocol": {
                "source_steps": 8,
                "target_steps": 8,
                "chai_block0_cache_steps": list(CC_CACHE_STEPS),
                "carnator_mask_probe_step": CC_MASK_PROBE_STEP,
                "target_latent_mask_source_step": CC_MASK_PROBE_STEP,
                "target_region_layers": native.parse_int_ranges(CC_REGION_LAYERS),
                "target_outer_layers": native.parse_int_ranges(CC_OUTER_LAYERS),
            },
            "baselines": [{"baseline": item["baseline"], **item} for item in _variant_specs()],
        }
        native.write_json(matrix_output, matrix)
        return matrix

    started = time.perf_counter()
    runtime = native.load_runtime(args)
    try:
        source_outer_state = OuterSparseState(cfg_branch_count=cfg_branches(args.guidance_scale), start_completed_step=1)
        source_outer_cache = SourceBlockMaskCache(
            paths["outer_mask_dir"],
            native.outer_mask_config(args),
            source_run_id=args.run_id,
            cfg_branch_count=cfg_branches(args.guidance_scale),
            create=True,
        )
        with ExitStack() as source_stack:
            source_stack.enter_context(
                OuterSparseCaptureHooks(
                    runtime.pipe.model,
                    runtime.wan["model_module"],
                    source_outer_state,
                    source_outer_cache,
                    native.parse_int_ranges(CC_OUTER_LAYERS),
                    execution_dtype=native.outer_sparse_execution_dtype(args),
                )
            )
            chai_capture = ChaiBlock0Hooks(
                runtime.pipe.model,
                runtime.wan["model_module"],
                capture=True,
                cache_steps=CC_CACHE_STEPS,
            )
            source_stack.enter_context(chai_capture)
            source = native.run_native_t2v_stage(
                runtime,
                args,
                label="source_dense_8step",
                prompt=args.source_prompt,
                out_video=paths["source_video"],
                cache_dir=paths["source_cache"],
                capture_masks=True,
                save_trajectory=True,
                outer_sparse_state=source_outer_state,
                mask_output_dir=paths["coarse_mask_dir"],
            )
        if set(chai_capture.captures) != set(CC_CACHE_STEPS):
            missing = sorted(set(CC_CACHE_STEPS) - set(chai_capture.captures))
            raise RuntimeError(f"source did not capture CHAI block-0 tokens at local steps {missing}")
        chai_path = paths["source_cache"] / "chai_block0_tokens.pt"
        _write_tokens(chai_path, chai_capture.captures)
        source["hybrid"] = {
            "name": NAME,
            "executed_steps": 8,
            "saved_latent_steps": list(CC_STEPS),
            "chai_block0_cache_steps": list(CC_CACHE_STEPS),
            "chai_block0_cache_path": str(chai_path),
            "block0_cache_capture": chai_capture.summary(),
            "carnator_mask_probe_step": CC_MASK_PROBE_STEP,
            "outer_mask_layers": native.parse_int_ranges(CC_OUTER_LAYERS),
        }
        native.write_json(paths["source_cache"] / "manifest.json", source)

        source_trajectory = {
            int(item["step_index"]): Path(item["path"])
            for item in source.get("trajectory", [])
            if item.get("kind") == "post_scheduler"
        }
        self_bundle = native.build_native_self_contrast_mask_bundle(
            runtime,
            args,
            source_trajectory=source_trajectory,
            coarse_mask_dir=paths["coarse_mask_dir"],
            self_mask_dir=paths["self_mask_dir"],
        )
        mask_bundle = native.build_native_fusion_mask_bundle(
            runtime,
            args,
            source_trajectory=source_trajectory,
            coarse_mask_dir=paths["coarse_mask_dir"],
            self_bundle=self_bundle,
            self_mask_dir=paths["self_mask_dir"],
            fusion_mask_dir=paths["mask_dir"],
        )
        mask_bundle["hybrid"] = {
            "name": NAME,
            "calibration_completed_step": CC_MASK_PROBE_STEP,
            "target_mask_source_completed_step": CC_MASK_PROBE_STEP,
            "block0_cache_steps": list(CC_CACHE_STEPS),
        }
        save_mask_bundle(paths["mask_dir"], mask_bundle, save_cpu=True)

        if args.skip_dense_reference:
            raise ValueError("--skip-dense-reference is only allowed by protocol tests, not matrix execution")
        dense = native.run_native_t2v_stage(
            runtime,
            args,
            label="target_dense_8step",
            prompt=args.target_prompt,
            out_video=artifact_run / "videos" / "target_dense_8step.mp4",
            cache_dir=artifact_run / "cache" / "target_dense_8step",
            resume_latent_path=paths["source_cache"] / "latents_initial.pt",
            resume_start_step=0,
        )
        variants: list[dict[str, Any]] = []
        tokens = _load_tokens(chai_path)
        for spec in _variant_specs():
            print(f"[{NAME}] running {spec['baseline']}", flush=True)
            variants.append(_run_target_variant(runtime, args, source, mask_bundle, tokens, paths["outer_mask_dir"], spec, artifact_run))
    finally:
        del runtime
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    dense_output = dense["output"]
    entries = []
    for manifest in variants:
        entries.append({
            "baseline": manifest["label"],
            "prompt": args.target_prompt,
            "output": manifest["output"],
            "target": manifest,
        })
    matrix = {
        "status": "completed",
        "name": NAME,
        "method": "chai-carnator",
        "profile": "wan22-ti2v-5b",
        "model": "Wan2.2-TI2V-5B",
        "run_id": args.run_id,
        "source_prompt": args.source_prompt,
        "target_prompt": args.target_prompt,
        "seed": args.seed,
        "steps": 8,
        "source": source,
        "target_dense": dense,
        "target_dense_output": dense_output,
        "baselines": entries,
        "protocol": {
            "source_steps": 8,
            "target_steps": 8,
            "chai_block0_cache_steps": list(CC_CACHE_STEPS),
            "carnator_mask_probe_step": CC_MASK_PROBE_STEP,
            "target_latent_mask_source_step": CC_MASK_PROBE_STEP,
            "target_region_layers": native.parse_int_ranges(CC_REGION_LAYERS),
            "target_outer_layers": native.parse_int_ranges(CC_OUTER_LAYERS),
        },
        "timing": {"matrix_seconds": time.perf_counter() - started},
    }
    native.write_json(matrix_output, matrix)
    return matrix


def main() -> None:
    args = _parser().parse_args()
    matrix = run_matrix(args)
    print(json.dumps({"status": matrix["status"], "name": matrix["name"], "matrix": matrix["run_id"], "baselines": [item["baseline"] for item in matrix["baselines"]]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
