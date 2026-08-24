#!/usr/bin/env python3
"""Run C0/B0--B5 for one T2V profile with one shared source-cache history."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from carnator.baselines import BASELINES
from carnator.runtime import default_checkpoint, load_profile, read_json, utc_now, write_json


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--profile", choices=("t2v-14b-distilled", "ti2v-5b"), required=True)
    value.add_argument("--source-prompt", required=True)
    value.add_argument("--target-prompt", required=True)
    value.add_argument("--source-positive", default="")
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
    value.add_argument("--baselines", default="C0,B0,B1,B2,B3,B4,B5")
    value.add_argument("--check-only", action="store_true")
    value.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    value.add_argument("--model-dtype", choices=("fp16", "bf16"), default="fp16")
    value.add_argument("--guidance-scale", type=float, default=None)
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
    value.add_argument("--runtime-python", type=Path, default=Path(sys.executable))
    return value


def selected(value: str) -> list[str]:
    names = [item.strip().upper() for item in value.split(",") if item.strip()]
    if not names or any(item not in BASELINES for item in names):
        raise ValueError("--baselines must be a non-empty subset of C0,B0,B1,B2,B3,B4,B5")
    return list(dict.fromkeys(names))


def common_command(args: argparse.Namespace, checkpoint: Path, baseline: str) -> list[str]:
    command = [
        str(args.runtime_python), str(ROOT / "scripts" / "run" / "carnator_baseline.py"),
        "--profile", args.profile, "--baseline", baseline,
        "--source-prompt", args.source_prompt, "--target-prompt", args.target_prompt,
        "--source-positive", args.source_positive, "--seed", str(args.seed),
        "--checkpoint", str(checkpoint), "--artifact-root", str(args.artifact_root),
        "--results-root", str(args.results_root), "--run-id", args.run_id,
        "--latent-injection-mode", args.latent_injection_mode,
        "--boundary-alpha-function", args.boundary_alpha_function,
        "--dtype", args.dtype, "--model-dtype", args.model_dtype,
        "--region-layers", args.region_layers, "--outer-layers", args.outer_layers,
        "--mask-self-layer", str(args.mask_self_layer), "--q-block-size", str(args.q_block_size),
        "--k-block-size", str(args.k_block_size), "--top-p", str(args.top_p),
        "--min-top-k", str(args.min_top_k), "--workspace-mib", str(args.workspace_mib),
        "--flashinfer-plan-cache", args.flashinfer_plan_cache, "--outer-dtype", args.outer_dtype,
        "--flashinfer-wrapper", args.flashinfer_wrapper,
    ]
    if not args.middle_attention_mask:
        command.append("--no-middle-attention-mask")
    if args.guidance_scale is not None:
        command.extend(("--guidance-scale", str(args.guidance_scale)))
    return command


def source_root(args: argparse.Namespace) -> Path:
    family = "t2v-14b-distilled" if args.profile == "t2v-14b-distilled" else "ti2v-5b"
    return (args.artifact_root / family / args.run_id / ("source" if args.profile == "t2v-14b-distilled" else "history")).resolve()


def record_14b(args: argparse.Namespace, baseline: str) -> dict[str, Any]:
    path = args.results_root / "t2v-14b-distilled" / args.run_id / f"{baseline}.json"
    return read_json(path)


def record_5b(args: argparse.Namespace, baseline: str) -> dict[str, Any]:
    result_root = args.results_root / "ti2v-5b"
    if baseline == "C0":
        summary = read_json(result_root / args.run_id / "summary.json")
        run = summary["runs"]["target_dense"]
    else:
        summary = read_json(result_root / f"{args.run_id}/{baseline}" / "summary.json")
        run = summary["runs"]["target_latent_reuse"]
    return {
        "status": summary["status"],
        "baseline": baseline,
        "profile": "ti2v-5b",
        "output": run.get("output"),
        "prompt": summary["config"]["target_prompt"],
        "seed": summary["config"]["seed"],
        "configuration": {
            "guidance_scale": summary["config"]["guidance_scale"],
            "outer_attention": BASELINES[baseline].outer_attention,
            "ffn": BASELINES[baseline].ffn,
            "latent_injection_mode": summary["config"].get("latent_injection_mode", "hard"),
            "boundary_alpha_function": summary["config"].get("boundary_alpha_function", "smoothstep"),
        },
        "timing": {"denoising_seconds": run.get("denoising_seconds"), "end_to_end_seconds": run.get("end_to_end_seconds")},
        "summary": str(result_root / (args.run_id if baseline == "C0" else f"{args.run_id}/{baseline}") / "summary.json"),
    }


def prepare_5b_source(args: argparse.Namespace, checkpoint: Path) -> Path:
    region_layers = "10-20" if args.region_layers == "10-30" else args.region_layers
    outer_layers = "0-9,21-29" if args.outer_layers == "0-9,31-39" else args.outer_layers
    command = [
        str(args.runtime_python), "-m", "carnator.adapters.wan_ti2v_5b", "--no-env-setup",
        "--model-dir", str(checkpoint), "--artifact-root", str(args.artifact_root / "ti2v-5b"),
        "--results-root", str(args.results_root / "ti2v-5b"), "--run-id", f"{args.run_id}/history",
        "--source-prompt", args.source_prompt, "--target-prompt", args.target_prompt,
        "--source-positive", args.source_positive, "--seed", str(args.seed), "--source-only",
        "--capture-outer-block-masks", "--sparse-layers", region_layers,
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
    subprocess.run(command, cwd=ROOT, check=True)
    return (args.artifact_root / "ti2v-5b" / args.run_id / "history").resolve()


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    names = selected(args.baselines)
    if any(name != "C0" for name in names) and "C0" not in names:
        # B0--B5 report speed against the matched dense target.  A matrix
        # therefore always materializes C0 before a derived baseline.
        names.insert(0, "C0")
    profile = load_profile(args.profile)
    checkpoint = (args.checkpoint or default_checkpoint(profile)).expanduser()
    if args.check_only:
        subprocess.run([*common_command(args, checkpoint, "C0"), "--check-only"], cwd=ROOT, check=True)
        return 0

    source = None
    if any(name != "C0" for name in names):
        source = prepare_5b_source(args, checkpoint) if args.profile == "ti2v-5b" else source_root(args)
    records: list[dict[str, Any]] = []
    for baseline in names:
        command = common_command(args, checkpoint, baseline)
        if baseline != "C0":
            assert source is not None
            command.extend(("--source-run-dir", str(source)))
            if args.profile == "ti2v-5b":
                dense_manifest = args.artifact_root / "ti2v-5b" / args.run_id / "cache" / "target_dense" / "manifest.json"
                command.extend(("--target-dense-manifest", str(dense_manifest)))
        subprocess.run(command, cwd=ROOT, check=True)
        records.append(record_14b(args, baseline) if args.profile == "t2v-14b-distilled" else record_5b(args, baseline))

    matrix_path = (args.results_root / args.profile / args.run_id / "matrix.json").resolve()
    write_json(
        matrix_path,
        {
            "status": "completed",
            "created_at": utc_now(),
            "profile": args.profile,
            "source_cache": str(source) if source else None,
            "latent_injection_mode": args.latent_injection_mode,
            "boundary_alpha_function": args.boundary_alpha_function,
            "middle_attention_mask": args.middle_attention_mask,
            "baselines": records,
            "quality_metrics": ["clip_t", "vbench_q"],
        },
    )
    print(f"matrix={matrix_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
