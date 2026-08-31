#!/usr/bin/env python3
"""Run Chorus2 on Wan2.2-TI2V-5B with SpargeAttn block-sparse attention."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from carnator.adapters import wan_ti2v_5b as native
from carnator.outer_sparse import (
    OuterSparseAttentionHooks,
    OuterSparseCaptureHooks,
    OuterSparseExecutor,
    OuterSparseState,
    SourceBlockMaskCache,
    cfg_branches,
)

ALL_BLOCKS = tuple(range(30))
SPARSE_START_STEP = 10


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model-dir", type=Path, default=native.DEFAULT_MODEL_DIR)
    parser.add_argument("--device", default=os.environ.get("CHORUS2_DEVICE", "cuda:0"))
    parser.add_argument("--artifact-root", type=Path, default=ROOT / "artifacts" / "chorus2")
    parser.add_argument("--results-root", type=Path, default=ROOT / "results" / "chorus2")
    parser.add_argument("--run-id", default="latest")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument(
        "--source-mask-cache",
        type=Path,
        help="Existing source block-mask cache for target steps after the sparse start step.",
    )
    parser.add_argument(
        "--source-latent-cache",
        type=Path,
        help="Existing source cache directory containing manifest.json and the post-scheduler latent trajectory.",
    )
    parser.add_argument(
        "--speedup-reference-manifest",
        type=Path,
        help="Shared 50-step target C0 Dense manifest or summary; used as the universal end-to-end speedup denominator.",
    )
    parser.add_argument("--target-dense-only", action="store_true", help="Run the full 50-step Dense target control.")

    parser.add_argument("--source-prompt", default=native.SOURCE_PROMPT)
    parser.add_argument("--target-prompt", default=native.TARGET_PROMPT)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--frames", type=int, default=121)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scheduler", choices=("unipc", "dpm++"), default="unipc")
    parser.add_argument("--solver-order", type=int, default=2)
    parser.add_argument("--flow-shift", type=float, default=5.0)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--offload-model", action="store_true")
    parser.add_argument("--t5-cpu", action="store_true")
    parser.add_argument("--model-dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--convert-model-dtype", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable-progress", action="store_true", default=True)

    parser.add_argument(
        "--reuse-start-step",
        type=int,
        default=0,
        help="Compatibility field; Chorus2 no longer restarts from a source latent and must remain zero.",
    )
    parser.add_argument("--outer-mask-q-block", type=int, default=64)
    parser.add_argument("--outer-mask-k-block", type=int, default=128)
    parser.add_argument("--outer-mask-top-p", type=float, default=0.95)
    parser.add_argument("--outer-mask-min-top-k", type=float, default=0.1)
    parser.add_argument("--layer-refresh-size", type=int, default=8)
    parser.add_argument("--outer-sparse-dtype", choices=("fp16", "bf16"), default="fp16")
    return parser


def run_paths(args: argparse.Namespace) -> dict[str, Path]:
    artifact_root = args.artifact_root.expanduser().resolve()
    results_root = args.results_root.expanduser().resolve()
    artifact_run = (artifact_root / args.run_id).resolve()
    result_run = (results_root / args.run_id).resolve()
    if not artifact_run.is_relative_to(artifact_root) or not result_run.is_relative_to(results_root):
        raise ValueError("--run-id must remain below the configured artifact and results roots")
    return {
        "artifact_run": artifact_run,
        "result_run": result_run,
        "source_video": artifact_run / "videos" / "source.mp4",
        "target_video": artifact_run / "videos" / "target.mp4",
        "source_cache": artifact_run / "cache" / "source",
        "target_cache": artifact_run / "cache" / "target",
        "mask_cache": artifact_run / "cache" / "source" / "block_masks",
        "summary": result_run / "summary.json",
        "report": result_run / "report.md",
    }


def validate(args: argparse.Namespace) -> None:
    if args.reuse_start_step != 0:
        raise ValueError("Chorus2 no longer supports source-latent restart; --reuse-start-step must be 0")
    if not 0 <= SPARSE_START_STEP < args.steps:
        raise ValueError("Chorus2 sparse start step must be in [0, steps)")
    if args.source_latent_cache is not None:
        raise ValueError("Chorus2 no longer accepts --source-latent-cache")
    if args.layer_refresh_size <= 0:
        raise ValueError("--layer-refresh-size must be positive")
    if not args.target_dense_only:
        if (args.outer_mask_q_block, args.outer_mask_k_block) != (64, 128):
            raise ValueError("Chorus2 SpargeAttn requires --outer-mask-q-block=64 and --outer-mask-k-block=128")
        native.outer_mask_config(args)


def compact_run(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "output": manifest.get("output"),
        "manifest_path": manifest.get("manifest_path"),
        "denoising_seconds": manifest.get("timing", {}).get("loop_denoise_seconds"),
        "dit_forward_seconds": manifest.get("timing", {}).get("dit_forward_seconds"),
        "end_to_end_seconds": manifest.get("timing", {}).get("end_to_end_seconds"),
        "completed_steps": manifest.get("completed_steps"),
        "outer_sparse": manifest.get("outer_sparse"),
        "resume": manifest.get("resume"),
        "masked_injection": manifest.get("masked_injection"),
        "cache_prepare": manifest.get("cache_prepare"),
        "peak_gpu_memory_gib": manifest.get("peak_gpu_memory_gib"),
        "hardware": manifest.get("hardware"),
    }


def load_speedup_reference(path: Path) -> dict[str, Any]:
    """Read the universal target C0 timing without accepting a restart control."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid --speedup-reference-manifest: {path}") from exc
    target = payload.get("runs", {}).get("target_dense", payload)
    timing = target.get("timing", {})
    end_to_end_seconds = timing.get("end_to_end_seconds", target.get("end_to_end_seconds"))
    if not isinstance(end_to_end_seconds, (int, float)) or end_to_end_seconds <= 0:
        raise ValueError("speedup reference has no positive target C0 end_to_end_seconds")
    if int(target.get("steps", payload.get("config", {}).get("steps", 50))) != 50:
        raise ValueError("Chorus2 speedup reference must be a full 50-step target C0 Dense run")
    resume_start = target.get("resume", {}).get("resume_start_step")
    if resume_start not in (None, 0):
        raise ValueError("Chorus2 speedup reference must not be a source-latent restart Dense control")
    return {"manifest_path": str(path), "end_to_end_seconds": float(end_to_end_seconds)}


def build_summary(
    args: argparse.Namespace,
    paths: dict[str, Path],
    *,
    status: str,
    source: dict[str, Any] | None = None,
    target: dict[str, Any] | None = None,
    speedup_reference: dict[str, Any] | None = None,
) -> dict[str, Any]:
    target_compact = compact_run(target) if target is not None else None
    speedup = (
        speedup_reference["end_to_end_seconds"] / target_compact["end_to_end_seconds"]
        if speedup_reference is not None and target_compact and target_compact.get("end_to_end_seconds")
        else None
    )
    return {
        "method": "chorus2",
        "status": status,
        "config": {
            "model": "Wan2.2-TI2V-5B",
            "source_prompt": args.source_prompt,
            "target_prompt": args.target_prompt,
            "seed": args.seed,
            "steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "source_mask_cache": {
                "mode": "not_used" if args.target_dense_only else ("external" if args.source_mask_cache is not None else "generated"),
                "path": (
                    None
                    if args.target_dense_only
                    else str(args.source_mask_cache) if args.source_mask_cache is not None else str(paths["mask_cache"])
                ),
            },
            "source_latent_cache": {
                "mode": "not_used",
                "path": None,
            },
            "sparse_attention": {
                "backend": "dense" if args.target_dense_only else "sparge-sage2",
                "layers": list(ALL_BLOCKS),
                "start_step": SPARSE_START_STEP + 1,
                "q_block_size": args.outer_mask_q_block,
                "k_block_size": args.outer_mask_k_block,
                "top_p": args.outer_mask_top_p,
                "min_top_k": args.outer_mask_min_top_k,
                "layer_refresh_size": args.layer_refresh_size,
            },
            "latent_reuse": False,
            "reuse_start_step": 0,
            "target_denoising_steps": args.steps,
            "speedup_reference": speedup_reference,
        },
        "paths": {key: str(value) for key, value in paths.items() if key not in {"summary", "report"}},
        "runs": {
            **({"source": compact_run(source)} if source is not None else {}),
            **({"target": target_compact} if target_compact is not None else {}),
        },
        "speedups": {
            "reference": "shared_target_c0_end_to_end" if speedup_reference is not None else None,
            "end_to_end_vs_reference": speedup,
        },
    }


def write_report(paths: dict[str, Path], summary: dict[str, Any]) -> None:
    config = summary["config"]
    target = summary["runs"].get("target", {})
    lines = [
        "# Chorus2",
        "",
        f"- Status: `{summary['status']}`",
        f"- Source latent restart: `disabled`; target denoises `{config['target_denoising_steps']}` steps",
        f"- Attention backend: `{config['sparse_attention']['backend']}` from target step `{config['sparse_attention']['start_step']}`",
        f"- Layer refresh: every `{config['sparse_attention']['layer_refresh_size']}` blocks",
        "- Latent reuse: `false`",
        f"- Target denoising: `{target.get('denoising_seconds')}` seconds",
        f"- Target end-to-end: `{target.get('end_to_end_seconds')}` seconds",
        f"- Universal C0 speedup: `{summary['speedups'].get('end_to_end_vs_reference')}`",
        f"- Target output: `{target.get('output')}`",
        "",
    ]
    paths["report"].parent.mkdir(parents=True, exist_ok=True)
    paths["report"].write_text("\n".join(lines), encoding="utf-8")


def write_summary(paths: dict[str, Path], summary: dict[str, Any]) -> None:
    native.write_json(paths["summary"], summary)
    write_report(paths, summary)


def verify_target_coverage(args: argparse.Namespace, target: dict[str, Any]) -> None:
    outer = target.get("outer_sparse", {})
    expected_runs = len(ALL_BLOCKS) * (args.steps - SPARSE_START_STEP) * cfg_branches(args.guidance_scale)
    expected_refresh_events = math.ceil(len(ALL_BLOCKS) / args.layer_refresh_size) * (
        args.steps - SPARSE_START_STEP
    ) * cfg_branches(args.guidance_scale)
    if (
        outer.get("backend") != "sparge-sage2"
        or outer.get("layers") != list(ALL_BLOCKS)
        or outer.get("runs") != expected_runs
        or outer.get("layer_refresh_size") != args.layer_refresh_size
        or outer.get("refresh_events") != expected_refresh_events
        or outer.get("dense_fallback_count") != 0
    ):
        raise RuntimeError(f"Chorus2 did not cover every configured sparse attention call: {outer}")
    resume = target.get("resume", {})
    if resume.get("resume_start_step") != 0 or resume.get("resume_latent_path") is not None:
        raise RuntimeError("Chorus2 target unexpectedly used a source-latent restart")
    if target.get("completed_steps") != args.steps:
        raise RuntimeError(f"Chorus2 target completed {target.get('completed_steps')} steps, expected {args.steps}")


def verify_source_mask_coverage(args: argparse.Namespace, cache: SourceBlockMaskCache) -> None:
    branches = ("cond",) if cfg_branches(args.guidance_scale) == 1 else ("cond", "uncond")
    missing = [
        (step, branch, layer)
        for step in range(SPARSE_START_STEP + 1, args.steps + 1)
        for branch in branches
        for layer in ALL_BLOCKS
        if (step, branch, layer) not in cache.entries
    ]
    if missing:
        sample = ", ".join(f"step={step}/{branch}/layer={layer}" for step, branch, layer in missing[:3])
        raise ValueError(
            "--source-mask-cache does not fully cover Chorus2; "
            f"missing {len(missing)} entries, including {sample}"
        )


def run_experiment(args: argparse.Namespace, *, runtime: native.NativeRuntime | None = None) -> dict[str, Any]:
    paths = run_paths(args)
    validate(args)
    speedup_reference = (
        load_speedup_reference(args.speedup_reference_manifest)
        if args.speedup_reference_manifest is not None
        else None
    )
    if args.dry_run:
        summary = build_summary(args, paths, status="dry_run", speedup_reference=speedup_reference)
        write_summary(paths, summary)
        return summary
    if args.clean:
        for path in (paths["artifact_run"], paths["result_run"]):
            if path.exists():
                shutil.rmtree(path)
    paths["artifact_run"].mkdir(parents=True, exist_ok=True)
    paths["result_run"].mkdir(parents=True, exist_ok=True)
    resume_latent_path = None

    owns_runtime = runtime is None
    runtime = runtime or native.load_runtime(args)
    try:
        branch_count = cfg_branches(args.guidance_scale)
        execution_dtype = native.outer_sparse_execution_dtype(args)
        source = None
        if args.source_mask_cache is None:
            if args.target_dense_only:
                source = None
            else:
                source_cache = SourceBlockMaskCache(
                    paths["mask_cache"],
                    native.outer_mask_config(args),
                    source_run_id=args.run_id,
                    cfg_branch_count=branch_count,
                    create=True,
                )
                source_state = OuterSparseState(
                    cfg_branch_count=branch_count,
                    start_completed_step=SPARSE_START_STEP + 1,
                )
                with OuterSparseCaptureHooks(
                    runtime.pipe.model,
                    runtime.wan["model_module"],
                    source_state,
                    source_cache,
                    list(ALL_BLOCKS),
                    execution_dtype=execution_dtype,
                ):
                    source = native.run_native_t2v_stage(
                        runtime,
                        args,
                        label="source",
                        prompt=args.source_prompt,
                        out_video=paths["source_video"],
                        cache_dir=paths["source_cache"],
                        outer_sparse_state=source_state,
                        save_trajectory=True,
                    )

        if args.target_dense_only:
            target = native.run_native_t2v_stage(
                runtime,
                args,
                label="target_dense",
                prompt=args.target_prompt,
                out_video=paths["target_video"].with_name("target_dense.mp4"),
                cache_dir=paths["target_cache"],
                resume_latent_path=None,
                resume_start_step=0,
            )
            resume = target.get("resume", {})
            if resume.get("resume_start_step") != 0 or resume.get("resume_latent_path") is not None:
                raise RuntimeError("Chorus2 Dense control unexpectedly used a source-latent restart")
        else:
            mask_cache_path = args.source_mask_cache if args.source_mask_cache is not None else paths["mask_cache"]
            target_cache = SourceBlockMaskCache(mask_cache_path, native.outer_mask_config(args))
            target_cache.expect_cfg_branches(branch_count)
            verify_source_mask_coverage(args, target_cache)
            target_state = OuterSparseState(
                cfg_branch_count=branch_count,
                start_completed_step=SPARSE_START_STEP + 1,
            )
            executor = OuterSparseExecutor(
                target_cache,
                "sparge-sage2",
                execution_dtype=execution_dtype,
                layer_refresh_size=args.layer_refresh_size,
            )
            cache_prepare = target_cache.prepare(device=runtime.device)
            with ExitStack() as stack:
                stack.enter_context(
                    OuterSparseAttentionHooks(
                        runtime.pipe.model,
                        runtime.wan["model_module"],
                        target_state,
                        executor,
                        list(ALL_BLOCKS),
                    )
                )
                target = native.run_native_t2v_stage(
                    runtime,
                    args,
                    label="target",
                    prompt=args.target_prompt,
                    out_video=paths["target_video"],
                    cache_dir=paths["target_cache"],
                    resume_latent_path=None,
                    resume_start_step=0,
                    outer_sparse_state=target_state,
                )
            target["cache_prepare"] = cache_prepare
            verify_target_coverage(args, target)
        summary = build_summary(
            args,
            paths,
            status="completed",
            source=source,
            target=target,
            speedup_reference=speedup_reference,
        )
        write_summary(paths, summary)
        return summary
    finally:
        if owns_runtime:
            runtime.profiler.close()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.model_dir = args.model_dir.expanduser()
    args.artifact_root = args.artifact_root.expanduser()
    args.results_root = args.results_root.expanduser()
    if args.source_mask_cache is not None:
        args.source_mask_cache = args.source_mask_cache.expanduser().resolve()
    if args.source_latent_cache is not None:
        args.source_latent_cache = args.source_latent_cache.expanduser().resolve()
    if args.speedup_reference_manifest is not None:
        args.speedup_reference_manifest = args.speedup_reference_manifest.expanduser().resolve()
    args.outer_attn_backend = "dense" if args.target_dense_only else "sparge-sage2"
    paths = run_paths(args)
    if args.check_only:
        status = native.env_status(args)
        native.write_json(paths["result_run"] / "preflight.json", status)
        print(f"preflight={paths['result_run'] / 'preflight.json'}")
        return 0
    summary = run_experiment(args)
    print(f"summary={paths['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
