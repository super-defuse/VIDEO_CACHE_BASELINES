#!/usr/bin/env python3
"""Run Carnator's native Wan2.2 TI2V-5B C0/B0--B5 matrix."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "models" / "Wan2.2-TI2V-5B"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-prompt", required=True)
    parser.add_argument("--target-prompt", required=True)
    parser.add_argument("--source-positive", default="dog")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--baselines", default="C0,B0,B1,B2,B3,B4,B5")
    parser.add_argument(
        "--latent-injection-mode",
        choices=("hard", "boundary-blend"),
        default="hard",
        help="Native Wan latent reuse policy.",
    )
    parser.add_argument(
        "--boundary-alpha-function",
        choices=("smoothstep", "cosine"),
        default="smoothstep",
        help="Distance-to-Mbase interpolation for boundary-blend mode.",
    )
    parser.add_argument(
        "--middle-attention-mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply source block-mask selection to Medit/Msee in native Wan middle layers.",
    )
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--runtime-python", type=Path, default=Path(sys.executable))
    args = parser.parse_args()

    wan_root = ROOT / "external" / "wan2.2"
    if not (wan_root / "wan").is_dir():
        raise SystemExit(f"Carnator Wan submodule is missing: {wan_root}; run git submodule update --init external/wan2.2")
    if not args.model_path.is_dir():
        raise SystemExit(f"Wan2.2 TI2V-5B checkpoint is missing: {args.model_path}")

    artifact_root = ROOT / "artifacts" / "carnator"
    results_root = ROOT / "artifacts" / "carnator-results"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run" / "carnator_matrix.py"),
        "--profile", "ti2v-5b",
        "--source-prompt", args.source_prompt,
        "--target-prompt", args.target_prompt,
        "--source-positive", args.source_positive,
        "--seed", str(args.seed),
        "--checkpoint", str(args.model_path.resolve()),
        "--artifact-root", str(artifact_root),
        "--results-root", str(results_root),
        "--run-id", args.run_id,
        "--baselines", args.baselines,
        "--latent-injection-mode", args.latent_injection_mode,
        "--boundary-alpha-function", args.boundary_alpha_function,
        "--runtime-python", str(args.runtime_python),
    ]
    if not args.middle_attention_mask:
        command.append("--no-middle-attention-mask")
    if args.check_only:
        command.append("--check-only")
    environment = os.environ.copy()
    environment.update({
        "CARNATOR_ENV_READY": "1",
        "CARNATOR_DEVICE": args.device,
        "CARNATOR_TI2V_5B_MODEL_DIR": str(args.model_path.resolve()),
        "PYTHONPATH": os.pathsep.join((str(ROOT), str(wan_root), environment.get("PYTHONPATH", ""))),
    })
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
