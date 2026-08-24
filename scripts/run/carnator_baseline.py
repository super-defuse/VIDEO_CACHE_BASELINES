#!/usr/bin/env python3
"""Run one C0/B0--B5 T2V baseline for a supported CARNATOR profile."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from carnator.baselines import BASELINES, parse_baseline
from carnator.runtime import default_checkpoint, load_profile, write_json


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--profile", choices=("t2v-14b-distilled", "ti2v-5b"), required=True)
    value.add_argument("--baseline", choices=tuple(BASELINES), required=True)
    value.add_argument("--source-prompt", required=True, help="Prompt used to prepare B0--B5 source assets.")
    value.add_argument("--target-prompt", required=True)
    value.add_argument("--source-positive", default="", help="Optional source phrase used by the mask probe.")
    value.add_argument("--seed", type=int, default=42)
    value.add_argument("--checkpoint", type=Path)
    value.add_argument("--artifact-root", type=Path, default=ROOT / "artifacts")
    value.add_argument("--results-root", type=Path, default=ROOT / "results")
    value.add_argument("--run-id", default="latest")
    value.add_argument(
        "--latent-injection-mode",
        choices=("hard", "boundary-blend"),
        default="hard",
        help="Native Wan latent reuse policy.",
    )
    value.add_argument(
        "--boundary-alpha-function",
        choices=("smoothstep", "cosine"),
        default="smoothstep",
        help="Distance-to-Mbase interpolation for boundary-blend mode.",
    )
    value.add_argument(
        "--middle-attention-mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply source block-mask selection to Medit/Msee in native Wan middle layers.",
    )
    value.add_argument("--source-run-dir", type=Path, help="Required by B0--B5 when source assets were prepared separately.")
    value.add_argument("--target-dense-manifest", type=Path, help="Optional matched C0 manifest for B0--B5 timing references.")
    value.add_argument("--check-only", action="store_true")
    value.add_argument("--cpu-offload", action="store_true", help="Allowed for C0 only on 14B; hooks need resident denoisers.")
    value.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    value.add_argument("--region-layers", default="10-30")
    value.add_argument("--outer-layers", default="0-9,31-39")
    value.add_argument("--mask-self-layer", type=int, default=20)
    value.add_argument("--q-block-size", type=int, default=64)
    value.add_argument("--k-block-size", type=int, default=128)
    value.add_argument("--top-p", type=float, default=0.95)
    value.add_argument("--min-top-k", type=float, default=0.1)
    value.add_argument("--workspace-mib", type=int, default=256)
    value.add_argument("--flashinfer-plan-cache", choices=("layer", "all", "none"), default="layer")
    value.add_argument("--flashinfer-wrapper", choices=("variable", "block"), default="variable")
    value.add_argument("--outer-dtype", choices=("fp16", "bf16"), default="fp16")
    value.add_argument("--model-dtype", choices=("fp16", "bf16"), default="fp16")
    value.add_argument(
        "--skip-w8a8-dense-diagnostic",
        action="store_true",
        help="Skip the optional dense W8A8 target diagnostic when using an existing C0 reference.",
    )
    value.add_argument("--guidance-scale", type=float, default=None)
    value.add_argument("--runtime-python", type=Path, default=Path(sys.executable))
    return value


def native_5b_command(args: argparse.Namespace, checkpoint: Path, spec: Any) -> list[str]:
    runtime_run_id = args.run_id if spec.identifier == "C0" else f"{args.run_id}/{spec.identifier}"
    # The 14B defaults are 40-layer ranges.  Map those defaults to the native
    # 30-layer TI2V topology while preserving explicit user choices.
    region_layers = "10-20" if args.region_layers == "10-30" else args.region_layers
    outer_layers = "0-9,21-29" if args.outer_layers == "0-9,31-39" else args.outer_layers
    command = [
        str(args.runtime_python), "-m", "carnator.adapters.wan_ti2v_5b", "--no-env-setup",
        "--model-dir", str(checkpoint), "--artifact-root", str(args.artifact_root / "ti2v-5b"),
        "--results-root", str(args.results_root / "ti2v-5b"), "--run-id", runtime_run_id,
        "--source-prompt", args.source_prompt, "--target-prompt", args.target_prompt,
        "--source-positive", args.source_positive, "--seed", str(args.seed),
        "--latent-injection-mode", args.latent_injection_mode,
        "--boundary-alpha-function", args.boundary_alpha_function,
        "--outer-attn-backend", spec.outer_attention, "--ffn-backend", spec.ffn,
        "--model-dtype", args.model_dtype, "--sparse-layers", region_layers,
        "--outer-sparse-layers", outer_layers, "--outer-mask-q-block", str(args.q_block_size),
        "--outer-mask-k-block", str(args.k_block_size), "--outer-mask-top-p", str(args.top_p),
        "--outer-mask-min-top-k", str(args.min_top_k), "--outer-workspace-mib", str(args.workspace_mib),
        "--outer-sparse-dtype", args.outer_dtype, "--flashinfer-plan-cache", args.flashinfer_plan_cache,
        "--flashinfer-wrapper", args.flashinfer_wrapper,
    ]
    if not args.middle_attention_mask:
        command.append("--no-middle-attention-mask")
    if args.guidance_scale is not None:
        command.extend(("--guidance-scale", str(args.guidance_scale)))
    if args.skip_w8a8_dense_diagnostic:
        command.append("--skip-w8a8-dense-diagnostic")
    if args.check_only:
        return [*command, "--check-only"]
    if spec.identifier == "C0":
        return [*command, "--target-dense-only"]
    if args.source_run_dir is not None:
        command.extend(("--source-run-dir", str(args.source_run_dir)))
        if not args.skip_w8a8_dense_diagnostic:
            command.append("--skip-w8a8-dense-diagnostic")
        if args.target_dense_manifest is not None:
            command.extend(("--target-dense-manifest", str(args.target_dense_manifest)))
    return command


def run_14b(args: argparse.Namespace, checkpoint: Path, spec: Any) -> dict[str, Any]:
    from carnator.adapters.lightx2v_t2v_14b import (
        LightX2V14BBackend,
        T2V14BEngine,
        T2VRequest,
        parse_layers,
        positive_token_indices,
    )
    from carnator.outer_sparse import BlockMaskConfig
    import torch

    if args.cpu_offload:
        raise ValueError("14B CPU offload is intentionally not exposed in this baseline suite; CARNATOR hooks require resident denoisers")
    if args.guidance_scale is not None:
        raise ValueError("Wan2.2-14B-T2V-distilled is fixed to no CFG; do not pass --guidance-scale")
    artifact_root = (args.artifact_root / "t2v-14b-distilled" / args.run_id).resolve()
    result_root = (args.results_root / "t2v-14b-distilled" / args.run_id).resolve()
    backend = LightX2V14BBackend(checkpoint, dtype=args.dtype, cpu_offload=args.cpu_offload)
    if args.check_only:
        payload = backend.preflight()
        write_json(result_root / "preflight.json", payload)
        return payload
    backend.load(artifact_root / "runtime")
    try:
        if spec.identifier == "C0":
            import time

            result_root.mkdir(parents=True, exist_ok=True)
            output = result_root / "C0.mp4"
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            started = time.perf_counter()
            denoising_started: float | None = None
            denoising_finished: float | None = None

            def before(_phase: str, step: int, _scheduler: Any) -> None:
                nonlocal denoising_started
                if step == 1:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    denoising_started = time.perf_counter()

            def after(_phase: str, step: int, _latents: Any, _scheduler: Any) -> None:
                nonlocal denoising_finished
                if step == 4:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    denoising_finished = time.perf_counter()

            backend.run(T2VRequest(args.target_prompt, args.seed, output), before_step=before, after_step=after)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            denoising = (
                denoising_finished - denoising_started
                if denoising_started is not None and denoising_finished is not None
                else elapsed
            )
            result = {
                "status": "completed",
                "baseline": "C0",
                "profile": "t2v-14b-distilled",
                "output": str(output),
                "prompt": args.target_prompt,
                "seed": args.seed,
                "configuration": {"enable_cfg": False, "steps": 4, "outer_attention": "dense", "ffn": "fp", "region_mode": "full"},
                "source_manifest": None,
                "timing": {"denoising_seconds": denoising, "end_to_end_seconds": elapsed},
            }
            write_json(result_root / "C0.json", result)
            return result
        engine = T2V14BEngine(
            backend,
            args.source_run_dir.resolve() if args.source_run_dir else artifact_root / "source",
            region_layers=parse_layers(args.region_layers),
            outer_layers=parse_layers(args.outer_layers),
            mask_self_layer=args.mask_self_layer,
            mask_config=BlockMaskConfig(args.q_block_size, args.k_block_size, args.top_p, args.min_top_k),
            outer_options={"workspace_mib": args.workspace_mib, "execution_dtype": torch.float16 if args.outer_dtype == "fp16" else torch.bfloat16, "flashinfer_plan_cache": args.flashinfer_plan_cache},
        )
        request = T2VRequest(args.target_prompt, args.seed)
        if spec.requires_source and not engine.source_manifest_path.exists():
            engine.capture_source(T2VRequest(args.source_prompt, args.seed), positive_token_indices(backend, args.source_prompt, args.source_positive))
        return engine.run_baseline(
            baseline=spec.identifier,
            outer_backend=spec.outer_attention,
            ffn_backend=spec.ffn,
            region_mode=spec.region_mode,
            output_root=result_root,
            request=request,
        )
    finally:
        backend.unload()


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    spec = parse_baseline(args.baseline)
    profile = load_profile(args.profile)
    if args.profile != "ti2v-5b" and args.latent_injection_mode != "hard":
        raise ValueError("boundary-blend is currently implemented only for native Wan2.2-TI2V-5B")
    checkpoint = (args.checkpoint or default_checkpoint(profile)).expanduser()
    if args.profile == "ti2v-5b":
        command = native_5b_command(args, checkpoint, spec)
        subprocess.run(command, cwd=ROOT, check=True)
        return 0
    result = run_14b(args, checkpoint, spec)
    print(f"result={result.get('output', 'preflight completed')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
