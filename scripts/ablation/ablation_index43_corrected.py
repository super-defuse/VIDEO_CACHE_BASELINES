#!/usr/bin/env python3
"""
Carnator Ablation Study - Corrected Version

运行4个变体的消融实验:
1. Full Computation - 完全计算，无任何优化 (基线)
2. Latent Reuse Only - 中间层(10-20)局部计算，头尾层正常，启用W8A8 FFN
3. Attention Reuse Only - 头尾层(0-9,21-29)稀疏attention，中间层正常，启用W8A8 FFN
4. Joint Latent-Attention Reuse - 中间层局部计算+头尾层稀疏attention，启用W8A8 FFN
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime
import multiprocessing as mp

ROOT = Path(__file__).resolve().parents[1]

# 使用之前生成的 source cache
REQUEST_PROMPT = "A cute little lion and a wobbly penguin appear"
CACHE_PROMPT = "a little dog and a penguin dancing"
SEED = 20260311
STEPS = 50

# 路径 - 使用之前生成的source cache
SOURCE_CACHE_DIR = ROOT / "artifacts/ablation_req16_20260909_032326/source_gen/source_gen"
MODEL_DIR = ROOT / "models/Wan2.2-TI2V-5B"

# Conda 环境
CONDA_ENV = "wan"

# 默认参数
DEFAULT_REUSE_START_STEP = 10
DEFAULT_MASK_LAYERS = "15"
DEFAULT_SPARSE_LAYERS = "10-20"
DEFAULT_OUTER_SPARSE_LAYERS = "0-9,21-29"


def run_with_conda(cmd: list, log_file: Path, gpu: int):
    """使用 conda 环境运行命令"""
    conda_cmd = [
        "conda", "run", "--no-capture-output", "-n", CONDA_ENV,
        "env", f"CUDA_VISIBLE_DEVICES={gpu}"
    ] + cmd

    with open(log_file, 'w') as f:
        f.write(f"Command: {' '.join(conda_cmd)}\n")
        f.write(f"Started: {datetime.now()}\n")
        f.write("="*80 + "\n\n")
        f.flush()

        start = time.time()
        result = subprocess.run(conda_cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
        elapsed = time.time() - start

        f.write(f"\n\n{'='*80}\n")
        f.write(f"Completed: {datetime.now()}\n")
        f.write(f"Elapsed: {elapsed:.2f}s\n")
        f.write(f"Exit code: {result.returncode}\n")

    return result.returncode, elapsed


def run_full_computation(gpu: int, output_dir: Path, log_file: Path):
    """变体1: Full Computation - 无任何优化"""
    cmd = [
        "python3",
        str(ROOT / "carnator/adapters/wan_ti2v_5b.py"),
        "--device", "cuda:0",
        "--model-dir", str(MODEL_DIR),
        "--target-prompt", REQUEST_PROMPT,
        "--seed", str(SEED),
        "--steps", str(STEPS),
        "--target-dense-only",
        "--ffn-backend", "fp",  # 基线用全精度
        "--artifact-root", str(output_dir),
        "--results-root", str(ROOT / "results/ti2v_5b"),
        "--run-id", "ablation",
    ]
    print(f"[Full Computation] Starting on GPU {gpu}...")
    return run_with_conda(cmd, log_file, gpu)


def run_latent_only(gpu: int, output_dir: Path, log_file: Path):
    """变体2: Latent Reuse Only - 中间层(10-20)局部计算，头尾层正常"""
    if not SOURCE_CACHE_DIR.exists():
        raise FileNotFoundError(f"Source cache not found: {SOURCE_CACHE_DIR}")

    cmd = [
        "python3",
        str(ROOT / "carnator/adapters/wan_ti2v_5b.py"),
        "--device", "cuda:0",
        "--model-dir", str(MODEL_DIR),
        "--source-prompt", CACHE_PROMPT,
        "--target-prompt", REQUEST_PROMPT,
        "--seed", str(SEED),
        "--steps", str(STEPS),
        "--source-run-dir", str(SOURCE_CACHE_DIR),
        "--reuse-start-step", str(DEFAULT_REUSE_START_STEP),
        "--mask-layers", DEFAULT_MASK_LAYERS,
        "--sparse-layers", DEFAULT_SPARSE_LAYERS,
        # 关键配置：
        "--no-middle-attention-mask",  # 中间层不用attention mask
        "--outer-attn-backend", "dense",  # 头尾层正常密集计算
        "--ffn-backend", "w8a8",  # 启用FFN量化
        "--artifact-root", str(output_dir),
        "--results-root", str(ROOT / "results/ti2v_5b"),
        "--run-id", "ablation",
    ]
    print(f"[Latent Only] Starting on GPU {gpu}...")
    return run_with_conda(cmd, log_file, gpu)


def run_attention_only(gpu: int, output_dir: Path, log_file: Path):
    """变体3: Attention Reuse Only - 头尾层(0-9,21-29)稀疏，中间层正常"""
    if not SOURCE_CACHE_DIR.exists():
        raise FileNotFoundError(f"Source cache not found: {SOURCE_CACHE_DIR}")

    cmd = [
        "python3",
        str(ROOT / "carnator/adapters/wan_ti2v_5b.py"),
        "--device", "cuda:0",
        "--model-dir", str(MODEL_DIR),
        "--source-prompt", CACHE_PROMPT,
        "--target-prompt", REQUEST_PROMPT,
        "--seed", str(SEED),
        "--steps", str(STEPS),
        "--source-run-dir", str(SOURCE_CACHE_DIR),
        # 关键配置：
        "--no-latent-injection",  # 禁用latent注入，中间层正常计算
        "--outer-attn-backend", "sparge-sage2",  # 头尾层用稀疏attention
        "--outer-sparse-layers", DEFAULT_OUTER_SPARSE_LAYERS,
        "--ffn-backend", "w8a8",  # 启用FFN量化
        "--artifact-root", str(output_dir),
        "--results-root", str(ROOT / "results/ti2v_5b"),
        "--run-id", "ablation",
    ]
    print(f"[Attention Only] Starting on GPU {gpu}...")
    return run_with_conda(cmd, log_file, gpu)


def run_joint(gpu: int, output_dir: Path, log_file: Path):
    """变体4: Joint - 中间层局部计算 + 头尾层稀疏"""
    if not SOURCE_CACHE_DIR.exists():
        raise FileNotFoundError(f"Source cache not found: {SOURCE_CACHE_DIR}")

    cmd = [
        "python3",
        str(ROOT / "carnator/adapters/wan_ti2v_5b.py"),
        "--device", "cuda:0",
        "--model-dir", str(MODEL_DIR),
        "--source-prompt", CACHE_PROMPT,
        "--target-prompt", REQUEST_PROMPT,
        "--seed", str(SEED),
        "--steps", str(STEPS),
        "--source-run-dir", str(SOURCE_CACHE_DIR),
        "--reuse-start-step", str(DEFAULT_REUSE_START_STEP),
        "--mask-layers", DEFAULT_MASK_LAYERS,
        "--sparse-layers", DEFAULT_SPARSE_LAYERS,
        # 关键配置：
        "--middle-attention-mask",  # 中间层用attention mask
        "--outer-attn-backend", "sparge-sage2",  # 头尾层用稀疏attention
        "--outer-sparse-layers", DEFAULT_OUTER_SPARSE_LAYERS,
        "--ffn-backend", "w8a8",  # 启用FFN量化
        "--artifact-root", str(output_dir),
        "--results-root", str(ROOT / "results/ti2v_5b"),
        "--run-id", "ablation",
    ]
    print(f"[Joint] Starting on GPU {gpu}...")
    return run_with_conda(cmd, log_file, gpu)


def run_variant_wrapper(args):
    """Wrapper for multiprocessing"""
    variant_name, func, gpu, output_dir, log_file = args
    try:
        exit_code, elapsed = func(gpu, output_dir, log_file)
        return {
            "variant": variant_name,
            "status": "success" if exit_code == 0 else "failed",
            "exit_code": exit_code,
            "elapsed": elapsed,
            "gpu": gpu,
            "output_dir": str(output_dir),
            "log_file": str(log_file),
        }
    except Exception as e:
        return {
            "variant": variant_name,
            "status": "error",
            "error": str(e),
            "gpu": gpu,
        }


def main():
    parser = argparse.ArgumentParser(description="Carnator Ablation Study - Corrected")
    parser.add_argument("--output-root", type=Path,
                       default=ROOT / f"artifacts/ablation_req16_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--gpus", nargs=4, type=int, default=[3, 4, 5, 6],
                       help="GPUs for [full-comp, latent-only, attention-only, joint]")
    parser.add_argument("--sequential", action="store_true", help="Run sequentially")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("Carnator Ablation Study - Corrected Version")
    print("="*80)
    print(f"Request prompt: {REQUEST_PROMPT}")
    print(f"Cache prompt:   {CACHE_PROMPT}")
    print(f"Source cache:   {SOURCE_CACHE_DIR}")
    print(f"Conda env: {CONDA_ENV}")
    print(f"Output root: {args.output_root}")
    print(f"Source cache exists: {SOURCE_CACHE_DIR.exists()}")
    print("="*80)
    print("\nVariant Configuration:")
    print("  1. Full Computation:      No optimization (baseline)")
    print("  2. Latent Only:           Middle(10-20) partial + W8A8 FFN")
    print("  3. Attention Only:        Outer(0-9,21-29) sparse + W8A8 FFN")
    print("  4. Joint:                 Middle partial + Outer sparse + W8A8 FFN")
    print("="*80)
    print()

    all_variants = [
        ("full_computation", run_full_computation, args.gpus[0]),
        ("latent_only", run_latent_only, args.gpus[1]),
        ("attention_only", run_attention_only, args.gpus[2]),
        ("joint", run_joint, args.gpus[3]),
    ]

    tasks = []
    for name, func, gpu in all_variants:
        variant_dir = args.output_root / name
        variant_dir.mkdir(parents=True, exist_ok=True)
        log_file = variant_dir / "run.log"
        tasks.append((name, func, gpu, variant_dir, log_file))

    results = []
    if args.sequential:
        print("Running variants sequentially...")
        for task in tasks:
            result = run_variant_wrapper(task)
            results.append(result)
            print(f"  {result['variant']}: {result['status']}")
    else:
        print(f"Running {len(tasks)} variants in parallel...")
        with mp.Pool(processes=len(tasks)) as pool:
            results = pool.map(run_variant_wrapper, tasks)

    summary = {
        "experiment": "Carnator Ablation Study - Corrected",
        "timestamp": datetime.now().isoformat(),
        "conda_env": CONDA_ENV,
        "request": {
            "prompt": REQUEST_PROMPT,
            "cache_prompt": CACHE_PROMPT,
            "source_cache_dir": str(SOURCE_CACHE_DIR),
            "seed": SEED,
            "steps": STEPS,
        },
        "configuration": {
            "full_computation": "No optimization, FFN=fp",
            "latent_only": "Middle(10-20) latent injection, no attention mask, outer dense, FFN=w8a8",
            "attention_only": "No latent injection, outer(0-9,21-29) sparse attention, FFN=w8a8",
            "joint": "Middle(10-20) latent+attention, outer(0-9,21-29) sparse, FFN=w8a8",
        },
        "variants": results,
    }

    summary_file = args.output_root / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*80)
    print("ABLATION STUDY SUMMARY")
    print("="*80)
    for r in results:
        status_icon = "✓" if r["status"] == "success" else "✗"
        elapsed_str = f"{r.get('elapsed', 0):.2f}s" if "elapsed" in r else "N/A"
        print(f"{status_icon} {r['variant']:25s} GPU {r['gpu']}  {elapsed_str:>10s}  {r['status']}")
    print("="*80)
    print(f"Summary: {summary_file}")
    print(f"Output: {args.output_root}")
    print("="*80)


if __name__ == "__main__":
    main()
