#!/usr/bin/env python3
"""Run the six Wan2.2 acceptance baselines for one or more prompt pairs.

The orchestrator deliberately launches every worker with a one-GPU
``CUDA_VISIBLE_DEVICES`` mask.  Physical GPUs outside the user supplied
0--2 set therefore cannot be touched by this experiment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
BASELINES = (
    "dense_baseline",
    "nirvana_vid",
    "teacache",
    "chorus",
    "chorus2",
    "carnator_boundary_b5",
)


@dataclass(frozen=True)
class Pair:
    pair_id: str
    source_prompt: str
    target_prompt: str
    seed: int


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-prompt")
    p.add_argument("--target-prompt")
    p.add_argument("--seed", type=int, default=20260218)
    p.add_argument("--pairs", type=Path, help="CSV or JSONL with pair_id, source_prompt, target_prompt, seed")
    p.add_argument("--model-path", type=Path, default=ROOT / "models" / "Wan2.2-TI2V-5B")
    p.add_argument("--run-root", type=Path, default=ROOT / "artifacts" / "acceptance")
    p.add_argument("--gpus", default="0,1,2", help="Physical target GPUs; only 0,1,2 are permitted")
    p.add_argument("--source-gpu", type=int, default=0)
    p.add_argument(
        "--source-positive",
        default=None,
        help="Carnator source-mask phrase; defaults to all non-special source prompt tokens",
    )
    p.add_argument("--quality-env", default="vbench")
    p.add_argument("--skip-quality", action="store_true")
    p.add_argument("--model-dtype", choices=("fp16", "bf16"), default="fp16")
    p.add_argument("--boundary-alpha-function", choices=("smoothstep", "cosine"), default="smoothstep")
    p.add_argument("--chorus-k1", type=int, default=10)
    p.add_argument("--chorus-k2", type=int, default=30)
    p.add_argument("--reference-root", type=Path, default=ROOT / "artifacts" / "acceptance" / "wan22-ti2v-5b-dog-to-cat")
    p.add_argument("--validate-reference", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def parse_gpus(value: str) -> list[int]:
    try:
        gpus = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("--gpus must be a comma-separated list of integers") from exc
    if not gpus or any(gpu not in (0, 1, 2) for gpu in gpus):
        raise ValueError("this acceptance runner is restricted to physical GPUs 0, 1 and 2")
    if len(set(gpus)) != len(gpus):
        raise ValueError("--gpus cannot contain duplicates")
    return gpus


def slug(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value.strip())
    return safe.strip("_") or "pair"


def load_pairs(args: argparse.Namespace) -> list[Pair]:
    if args.pairs:
        path = args.pairs.expanduser().resolve()
        if path.suffix.lower() == ".csv":
            rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
        else:
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        pairs = []
        for index, row in enumerate(rows):
            pairs.append(
                Pair(
                    str(row.get("pair_id") or f"pair-{index:03d}"),
                    str(row["source_prompt"]),
                    str(row["target_prompt"]),
                    int(row.get("seed", args.seed)),
                )
            )
        if not pairs:
            raise ValueError(f"no prompt pairs found in {path}")
        return pairs
    if not args.source_prompt or not args.target_prompt:
        raise ValueError("provide --source-prompt and --target-prompt, or --pairs")
    return [Pair("pair-000", args.source_prompt, args.target_prompt, args.seed)]


def make_run_root(base: Path) -> Path:
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    root = base.expanduser().resolve() / stamp
    index = 1
    while root.exists():
        root = base.expanduser().resolve() / f"{stamp}_{index:02d}"
        index += 1
    root.mkdir(parents=True)
    return root


def child_env(gpu: int, *, carnator: bool = False) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["PYTHONPATH"] = os.pathsep.join(
        item for item in (str(ROOT), str(ROOT / "external" / "wan2.2"), env.get("PYTHONPATH", "")) if item
    )
    if carnator:
        env["CARNATOR_ENV_READY"] = "1"
        env["CARNATOR_DEVICE"] = "cuda:0"
    return env


def run_command(name: str, command: list[str], *, gpu: int, log_path: Path, dry_run: bool, carnator: bool = False) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = " ".join(subprocess.list2cmdline([item]) for item in command)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"gpu={gpu}\ncommand={rendered}\n\n")
        log.flush()
        if dry_run:
            return
        subprocess.run(command, cwd=ROOT, env=child_env(gpu, carnator=carnator), stdout=log, stderr=subprocess.STDOUT, check=True)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def first_frame(video: Path, output: Path) -> None:
    import cv2

    reader = cv2.VideoCapture(str(video))
    try:
        ok, frame = reader.read()
    finally:
        reader.release()
    if not ok:
        raise RuntimeError(f"could not decode first frame: {video}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), frame):
        raise RuntimeError(f"could not write first frame: {output}")


def copy_video(source: Path, destination: Path) -> Path:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    return destination


def source_stage(pair_root: Path, pair: Pair, args: argparse.Namespace) -> Path:
    command = [
        sys.executable,
        "-m",
        "carnator.adapters.wan_ti2v_5b",
        "--no-env-setup",
        "--model-dir",
        str(args.model_path.expanduser().resolve()),
        "--artifact-root",
        str(pair_root),
        "--results-root",
        str(pair_root / "_native_results"),
        "--run-id",
        "shared_cache",
        "--source-prompt",
        pair.source_prompt,
        "--target-prompt",
        pair.target_prompt,
        "--seed",
        str(pair.seed),
        "--device",
        "cuda:0",
        "--source-only",
        "--prepare-shared-dense-cache",
        "--model-dtype",
        args.model_dtype,
    ]
    # Passing an empty phrase makes the native adapter select every real source
    # token.  It remains usable for arbitrary prompts rather than assuming "dog".
    command.extend(("--source-positive", args.source_positive if args.source_positive is not None else ""))
    run_command("source", command, gpu=args.source_gpu, log_path=pair_root / "source.log", dry_run=args.dry_run)
    manifest = pair_root / "shared_cache" / "cache" / "shared_dense" / "manifest.json"
    if not args.dry_run and not manifest.is_file():
        raise FileNotFoundError(f"source stage did not produce shared manifest: {manifest}")
    if not args.dry_run:
        shutil.copy2(manifest, pair_root / "shared_cache" / "manifest.json")
        source_video = pair_root / "shared_cache" / "videos" / "source_dense.mp4"
        if source_video.is_file():
            canonical_source = copy_video(source_video, pair_root / "shared_cache" / "source.mp4")
            first_frame(canonical_source, pair_root / "shared_cache" / "source_first_frame.png")
    return manifest


def dense_stage(pair_root: Path, pair: Pair, args: argparse.Namespace, gpu: int) -> dict[str, Any]:
    run_dir = pair_root / "dense_baseline"
    command = [
        sys.executable,
        "-m",
        "carnator.adapters.wan_ti2v_5b",
        "--no-env-setup",
        "--model-dir",
        str(args.model_path.expanduser().resolve()),
        "--artifact-root",
        str(pair_root),
        "--results-root",
        str(pair_root / "_native_results"),
        "--run-id",
        "dense_baseline",
        "--source-prompt",
        pair.source_prompt,
        "--target-prompt",
        pair.target_prompt,
        "--seed",
        str(pair.seed),
        "--device",
        "cuda:0",
        "--target-dense-only",
        "--model-dtype",
        args.model_dtype,
    ]
    run_command("dense_baseline", command, gpu=gpu, log_path=pair_root / "logs" / "dense_baseline.log", dry_run=args.dry_run)
    output = run_dir / "videos" / "target_dense.mp4"
    normalized = copy_video(output, run_dir / "videos" / "target.mp4") if not args.dry_run else run_dir / "videos" / "target.mp4"
    manifest = run_dir / "cache" / "target_dense" / "manifest.json"
    data = read_json(manifest) if not args.dry_run else {"timing": {}, "completed_steps": 50}
    timing = data.get("timing", {})
    row = {
        "request_index": 0,
        "uuid": f"{pair.pair_id}-target-000",
        "pair_id": pair.pair_id,
        "prompt": pair.target_prompt,
        "model": "Wan2.2",
        "method": "dense_baseline",
        "seed": pair.seed,
        "cache_match": {"hit": False, "kind": "not_applicable"},
        "executed_steps": data.get("completed_steps", 50),
        "latency_s": timing.get("end_to_end_seconds"),
        "denoising_seconds": timing.get("loop_denoise_seconds"),
        "status": "planned" if args.dry_run else "ok",
        "output_path": str(normalized) if not args.dry_run else None,
        "quality_input_path": str(normalized) if not args.dry_run else None,
        "actual_gpu": gpu,
    }
    write_jsonl(run_dir / "results.jsonl", [row])
    write_json(run_dir / "config.resolved.json", {
        "method": "dense_baseline",
        "source_prompt": pair.source_prompt,
        "target_prompt": pair.target_prompt,
        "seed": pair.seed,
        "model_path": str(args.model_path),
        "model_dtype": args.model_dtype,
    })
    if not args.dry_run:
        first_frame(normalized, run_dir / "frames" / "target_first_frame.png")
    return row


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def generic_stage(pair_root: Path, pair: Pair, args: argparse.Namespace, method: str, gpu: int, manifest: Path) -> dict[str, Any]:
    run_dir = pair_root / method
    command = [
        sys.executable,
        "scripts/run/shared_cache_wan_baseline.py",
        "--method",
        method,
        "--artifact-root",
        str(pair_root),
        "--results-root",
        str(pair_root / "_native_results"),
        "--run-id",
        method,
        "--shared-cache-manifest",
        str(manifest),
        "--source-prompt",
        pair.source_prompt,
        "--target-prompt",
        pair.target_prompt,
        "--seed",
        str(pair.seed),
        "--device",
        "cuda:0",
    ]
    if method == "chorus":
        command.extend((
            "--chorus-k1", str(args.chorus_k1), "--chorus-k2", str(args.chorus_k2),
            "--chorus-gamma-k", "1.5", "--chorus-gamma-o", "1.5",
        ))
    run_command(method, command, gpu=gpu, log_path=pair_root / "logs" / f"{method}.log", dry_run=args.dry_run)
    results_path = run_dir / "results.jsonl"
    if args.dry_run:
        row = {"method": method, "status": "planned", "actual_gpu": gpu}
        write_jsonl(results_path, [row])
        write_json(run_dir / "config.resolved.json", {
            "method": method,
            "source_prompt": pair.source_prompt,
            "target_prompt": pair.target_prompt,
            "seed": pair.seed,
            "shared_cache_manifest": str(manifest),
        })
        return row
    rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != 1:
        raise ValueError(f"expected one result row in {results_path}")
    row = rows[0]
    row["pair_id"] = pair.pair_id
    row["actual_gpu"] = gpu
    timing = row.get("extra", {}).get("timing", {})
    generation = timing.get("generation_s")
    excluded = timing.get("excluded_preprocess_s")
    if not excluded:
        excluded = timing.get("preprocess_breakdown_s", 0.0)
    row["denoising_seconds"] = (float(generation) - float(excluded)) if generation is not None else None
    write_jsonl(results_path, [row])
    output = Path(row["output_path"])
    first_frame(output, run_dir / "frames" / "target_first_frame.png")
    return row


def chorus2_stage(pair_root: Path, pair: Pair, args: argparse.Namespace, gpu: int, manifest: Path) -> dict[str, Any]:
    shared = read_json(manifest) if not args.dry_run else {"consumers": {"chorus2": {"source_mask_cache": "planned"}}}
    mask = Path(shared["consumers"]["chorus2"]["source_mask_cache"]) if not args.dry_run else pair_root / "shared_cache" / "cache" / "source_dense" / "outer_block_masks"
    dense_manifest = pair_root / "dense_baseline" / "cache" / "target_dense" / "manifest.json"
    run_dir = pair_root / "chorus2"
    command = [
        sys.executable,
        "scripts/run/chorus2_wan.py",
        "--model-dir",
        str(args.model_path.expanduser().resolve()),
        "--artifact-root",
        str(pair_root),
        "--results-root",
        str(pair_root / "_native_results"),
        "--run-id",
        "chorus2",
        "--source-mask-cache",
        str(mask),
        "--speedup-reference-manifest",
        str(dense_manifest),
        "--source-prompt",
        pair.source_prompt,
        "--target-prompt",
        pair.target_prompt,
        "--seed",
        str(pair.seed),
        "--device",
        "cuda:0",
        "--model-dtype",
        args.model_dtype,
    ]
    run_command("chorus2", command, gpu=gpu, log_path=pair_root / "logs" / "chorus2.log", dry_run=args.dry_run)
    summary_path = pair_root / "_native_results" / "chorus2" / "summary.json"
    if args.dry_run:
        write_json(run_dir / "config.resolved.json", {
            "method": "chorus2",
            "source_prompt": pair.source_prompt,
            "target_prompt": pair.target_prompt,
            "seed": pair.seed,
            "model_path": str(args.model_path),
            "source_mask_cache": str(mask),
            "speedup_reference_manifest": str(dense_manifest),
        })
        return {"method": "chorus2", "status": "planned", "actual_gpu": gpu}
    summary = read_json(summary_path)
    target = summary["runs"]["target"]
    output = Path(target["output"])
    row = {
        "request_index": 0,
        "uuid": f"{pair.pair_id}-target-000",
        "pair_id": pair.pair_id,
        "prompt": pair.target_prompt,
        "model": "Wan2.2",
        "method": "chorus2",
        "seed": pair.seed,
        "cache_match": {"hit": True, "kind": "shared_dense"},
        "executed_steps": target.get("completed_steps"),
        "latency_s": target.get("end_to_end_seconds"),
        "denoising_seconds": target.get("denoising_seconds"),
        "status": "ok",
        "output_path": str(output),
        "quality_input_path": str(output),
        "actual_gpu": gpu,
    }
    write_jsonl(run_dir / "results.jsonl", [row])
    write_json(run_dir / "config.resolved.json", {
        "method": "chorus2",
        "source_prompt": pair.source_prompt,
        "target_prompt": pair.target_prompt,
        "seed": pair.seed,
        "model_path": str(args.model_path),
        "source_mask_cache": str(mask),
        "speedup_reference_manifest": str(dense_manifest),
    })
    first_frame(output, run_dir / "frames" / "target_first_frame.png")
    return row


def carnator_stage(pair_root: Path, pair: Pair, args: argparse.Namespace, gpu: int) -> dict[str, Any]:
    run_dir = pair_root / "carnator_boundary_b5"
    command = [
        sys.executable,
        "scripts/run/carnator_baseline.py",
        "--profile",
        "ti2v-5b",
        "--baseline",
        "B5",
        "--source-prompt",
        pair.source_prompt,
        "--target-prompt",
        pair.target_prompt,
        "--seed",
        str(pair.seed),
        "--checkpoint",
        str(args.model_path.expanduser().resolve()),
        "--artifact-root",
        str(pair_root),
        "--results-root",
        str(pair_root / "_native_results"),
        "--run-id",
        "carnator_boundary_b5",
        "--latent-injection-mode",
        "boundary-blend",
        "--boundary-alpha-function",
        args.boundary_alpha_function,
        "--source-run-dir",
        str(pair_root / "shared_cache"),
        "--target-dense-manifest",
        str(pair_root / "dense_baseline" / "cache" / "target_dense" / "manifest.json"),
        "--skip-w8a8-dense-diagnostic",
        "--runtime-python",
        sys.executable,
    ]
    command.extend(("--source-positive", args.source_positive if args.source_positive is not None else ""))
    run_command("carnator_boundary_b5", command, gpu=gpu, log_path=pair_root / "logs" / "carnator_boundary_b5.log", dry_run=args.dry_run, carnator=True)
    summary_path = pair_root / "_native_results" / "ti2v-5b" / "carnator_boundary_b5" / "B5" / "summary.json"
    if args.dry_run:
        write_json(run_dir / "config.resolved.json", {
            "method": "carnator_boundary_b5",
            "source_prompt": pair.source_prompt,
            "target_prompt": pair.target_prompt,
            "seed": pair.seed,
            "model_path": str(args.model_path),
            "latent_injection_mode": "boundary-blend",
            "baseline": "B5",
            "boundary_alpha_function": args.boundary_alpha_function,
            "source_run_dir": str(pair_root / "shared_cache"),
            "target_dense_manifest": str(pair_root / "dense_baseline" / "cache" / "target_dense" / "manifest.json"),
        })
        return {"method": "carnator_boundary_b5", "status": "planned", "actual_gpu": gpu}
    summary = read_json(summary_path)
    target = summary["runs"]["target_latent_reuse"]
    output = Path(target["output"])
    normalized = copy_video(output, run_dir / "videos" / "target.mp4")
    row = {
        "request_index": 0,
        "uuid": f"{pair.pair_id}-target-000",
        "pair_id": pair.pair_id,
        "prompt": pair.target_prompt,
        "model": "Wan2.2",
        "method": "carnator_boundary_b5",
        "seed": pair.seed,
        "cache_match": {"hit": True, "kind": "shared_dense"},
        "executed_steps": target.get("completed_steps"),
        "latency_s": target.get("end_to_end_seconds"),
        "denoising_seconds": target.get("denoising_seconds"),
        "status": "ok",
        "output_path": str(normalized),
        "quality_input_path": str(normalized),
        "actual_gpu": gpu,
    }
    write_jsonl(run_dir / "results.jsonl", [row])
    write_json(run_dir / "config.resolved.json", {
        "method": "carnator_boundary_b5",
        "source_prompt": pair.source_prompt,
        "target_prompt": pair.target_prompt,
        "seed": pair.seed,
        "model_path": str(args.model_path),
        "latent_injection_mode": "boundary-blend",
        "baseline": "B5",
        "boundary_alpha_function": args.boundary_alpha_function,
        "source_run_dir": str(pair_root / "shared_cache"),
        "target_dense_manifest": str(pair_root / "dense_baseline" / "cache" / "target_dense" / "manifest.json"),
    })
    first_frame(normalized, run_dir / "frames" / "target_first_frame.png")
    return row


def quality_stage(pair_root: Path, args: argparse.Namespace, baseline: str) -> None:
    command = [sys.executable, "scripts/benchmark/quality.py", str(pair_root / baseline), "--quality-env", args.quality_env]
    run_command(f"quality:{baseline}", command, gpu=0, log_path=pair_root / "logs" / f"quality_{baseline}.log", dry_run=args.dry_run)


def reference_scores(reference_root: Path) -> dict[str, dict[str, float]]:
    path = reference_root / "quality" / "summary.json"
    if not path.is_file():
        return {}
    payload = read_json(path)
    return {str(key): value for key, value in payload.get("scores", {}).items() if isinstance(value, dict)}


def reference_table_scores() -> dict[str, dict[str, float]]:
    """Use the frozen acceptance table for baselines absent from quality/summary."""
    path = ROOT / "docs" / "ACCEPTANCE.md"
    aliases = {"Dense C0": "dense_baseline", "NIRVANA-VID": "nirvana_vid", "TeaCache": "teacache", "Chorus": "chorus", "Chorus2": "chorus2", "Carnator Boundary B5": "carnator_boundary_b5"}
    values: dict[str, dict[str, float]] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 7 or cells[0] not in aliases:
            continue
        try:
            values[aliases[cells[0]]] = {"clip_t": float(cells[5]), "vbench_q": float(cells[6])}
        except ValueError:
            continue
    return values


def reference_timing(reference_root: Path) -> dict[str, float]:
    """Read denoising timings from the existing native/generic manifests."""
    values: dict[str, float] = {}
    dense = reference_root / "shared-dense-gpu3" / "cache" / "target_dense" / "manifest.json"
    if dense.is_file():
        timing = read_json(dense).get("timing", {})
        if timing.get("loop_denoise_seconds") is not None:
            values["dense_baseline"] = float(timing["loop_denoise_seconds"])
    for method, directory in {
        "nirvana_vid": "nirvana-vid-shared-cache-gpu5",
        "teacache": "teacache-shared-cache-gpu7",
        "chorus": "chorus-shared-cache-gpu7",
    }.items():
        summary = reference_root / "results" / directory / "summary.json"
        if summary.is_file():
            timing = read_json(summary).get("target", {}).get("extra", {}).get("timing", {})
            generation = timing.get("generation_s")
            excluded = timing.get("excluded_preprocess_s", 0.0) or 0.0
            if generation is not None:
                values[method] = float(generation) - float(excluded)
    chorus2 = reference_root / "results" / "chorus2-shared-cache-gpu6" / "summary.json"
    if chorus2.is_file():
        value = read_json(chorus2).get("runs", {}).get("target", {}).get("denoising_seconds")
        if value is not None:
            values["chorus2"] = float(value)
    carnator = reference_root / "results" / "carnator-boundary-B5-gpu3" / "summary.json"
    if carnator.is_file():
        value = read_json(carnator).get("runs", {}).get("target_latent_reuse", {}).get("denoising_seconds")
        if value is not None:
            values["carnator_boundary_b5"] = float(value)
    return values


def validate_reference(rows: list[dict[str, Any]], reference_root: Path) -> dict[str, Any]:
    scores = {**reference_table_scores(), **reference_scores(reference_root)}
    timings = reference_timing(reference_root)
    checks = []
    for row in rows:
        ref = scores.get(row.get("method"), {})
        historical_chorus2 = row.get("method") == "chorus2" and row.get("executed_steps") != 40
        check = {
            "baseline": row.get("method"),
            "denoising_reference": timings.get(row.get("method")),
            "denoising_actual": row.get("denoising_seconds"),
            "clip_t_reference": ref.get("clip_t"),
            "clip_t_actual": row.get("clip_t"),
            "vbench_q_reference": ref.get("vbench_q"),
            "vbench_q_actual": row.get("vbench_q"),
            "status": "protocol_changed" if historical_chorus2 else ("not_scored" if not ref else "compared"),
        }
        if historical_chorus2:
            check["note"] = (
                "current Chorus2 is the 50-step sparse-mask protocol; "
                "the historical acceptance artifact used a 40-step source-latent restart"
            )
        for actual_key, reference_key, delta_key in (
            ("denoising_actual", "denoising_reference", "denoising_delta"),
            ("clip_t_actual", "clip_t_reference", "clip_t_delta"),
            ("vbench_q_actual", "vbench_q_reference", "vbench_q_delta"),
        ):
            if not historical_chorus2 and check[actual_key] is not None and check[reference_key] is not None:
                check[delta_key] = float(check[actual_key]) - float(check[reference_key])
        checks.append(check)
    return {"reference_root": str(reference_root), "checks": checks}


def run_pair(pair_root: Path, pair: Pair, args: argparse.Namespace, gpus: list[int]) -> list[dict[str, Any]]:
    manifest = source_stage(pair_root, pair, args)
    rows: list[dict[str, Any]] = []
    def run_one(baseline: str, gpu: int) -> dict[str, Any]:
        if baseline == "dense_baseline":
            return dense_stage(pair_root, pair, args, gpu)
        if baseline in {"nirvana_vid", "teacache", "chorus"}:
            return generic_stage(pair_root, pair, args, baseline, gpu, manifest)
        if baseline == "chorus2":
            return chorus2_stage(pair_root, pair, args, gpu, manifest)
        return carnator_stage(pair_root, pair, args, gpu)

    # Submit at most one task per physical GPU.  The next batch starts only
    # after all tasks in the current batch have exited, preserving timing
    # isolation even when fewer than six baselines are requested in future.
    for offset in range(0, len(BASELINES), len(gpus)):
        batch = list(zip(BASELINES[offset : offset + len(gpus)], gpus))
        with ThreadPoolExecutor(max_workers=len(batch)) as pool:
            futures = {pool.submit(run_one, baseline, gpu): (baseline, gpu) for baseline, gpu in batch}
            for future in as_completed(futures):
                baseline, gpu = futures[future]
                try:
                    row = future.result()
                except Exception as exc:
                    row = {
                        "pair_id": pair.pair_id,
                        "method": baseline,
                        "actual_gpu": gpu,
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "log_path": str(pair_root / "logs" / f"{baseline}.log"),
                    }
                rows.append(row)
    rows.sort(key=lambda row: BASELINES.index(row.get("method", "")) if row.get("method") in BASELINES else 999)
    if not args.skip_quality:
        for baseline in BASELINES:
            if (pair_root / baseline / "results.jsonl").is_file() and not any(row.get("status") == "error" and row.get("method") == baseline for row in rows):
                try:
                    quality_stage(pair_root, args, baseline)
                    scored = [json.loads(line) for line in (pair_root / baseline / "results.jsonl").read_text(encoding="utf-8").splitlines() if line]
                    if scored:
                        matching = next(row for row in rows if row.get("method") == baseline)
                        matching.update({key: scored[0].get(key) for key in ("clip_t", "vbench_q", "vbench_dimensions")})
                except Exception as exc:
                    failed_row = next(row for row in rows if row.get("method") == baseline)
                    failed_row["quality_error"] = f"{type(exc).__name__}: {exc}"
                    failed_row["quality_log_path"] = str(pair_root / "logs" / f"quality_{baseline}.log")
    return rows


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    args.model_path = args.model_path.expanduser().resolve()
    gpus = parse_gpus(args.gpus)
    if args.source_gpu != 0:
        raise ValueError("--source-gpu is fixed to physical GPU0 by the acceptance protocol")
    if not 0 < args.chorus_k1 < args.chorus_k2 < 50:
        raise ValueError("Chorus boundaries must satisfy 0 < --chorus-k1 < --chorus-k2 < 50")
    pairs = load_pairs(args)
    run_root = make_run_root(args.run_root)
    run_manifest = {
        "status": "running",
        "created_at": datetime.now().astimezone().isoformat(),
        "root": str(run_root),
        "pairs": [pair.__dict__ for pair in pairs],
        "baselines": list(BASELINES),
        "source_gpu": args.source_gpu,
        "target_gpus": gpus,
        "cuda_policy": "CUDA_VISIBLE_DEVICES restricted to physical GPUs 0,1,2",
        "model_path": str(args.model_path),
        "model_dtype": args.model_dtype,
        "source_positive": args.source_positive if args.source_positive is not None else "<all-source-tokens>",
        "carnator": {"mode": "boundary-blend", "baseline": "B5", "alpha_function": args.boundary_alpha_function},
        "chorus": {"k1": args.chorus_k1, "k2": args.chorus_k2, "gamma_k": 1.5, "gamma_o": 1.5},
    }
    write_json(run_root / "run_manifest.json", run_manifest)
    (run_root / "prompts.jsonl").write_text("".join(json.dumps(pair.__dict__, ensure_ascii=False) + "\n" for pair in pairs), encoding="utf-8")

    all_rows: list[dict[str, Any]] = []
    multi = len(pairs) > 1
    for pair in pairs:
        pair_root = run_root / slug(pair.pair_id) if multi else run_root
        pair_root.mkdir(parents=True, exist_ok=True)
        try:
            rows = run_pair(pair_root, pair, args, gpus)
        except Exception as exc:
            # Source preparation is shared by all six targets.  Preserve a
            # complete failure record when it prevents target execution.
            rows = [
                {
                    "pair_id": pair.pair_id,
                    "method": baseline,
                    "actual_gpu": args.source_gpu,
                    "status": "error",
                    "error": f"source stage: {type(exc).__name__}: {exc}",
                    "log_path": str(pair_root / "source.log"),
                }
                for baseline in BASELINES
            ]
        manifest_path = pair_root / "shared_cache" / "cache" / "shared_dense" / "manifest.json"
        if manifest_path.is_file():
            run_manifest.setdefault("cache_manifests", {})[pair.pair_id] = {
                "path": str(manifest_path),
                "sha256": file_sha256(manifest_path),
            }
        for row in rows:
            row["pair_root"] = str(pair_root)
        all_rows.extend(rows)

    has_failure = any(row.get("status") == "error" or row.get("quality_error") for row in all_rows)
    all_planned = bool(all_rows) and all(row.get("status") == "planned" for row in all_rows)
    run_status = "dry_run" if all_planned else ("partial" if has_failure else "completed")
    write_json(run_root / "summary.json", {"status": run_status, "rows": all_rows})
    fieldnames = [
        "pair_id", "method", "status", "actual_gpu", "denoising_seconds", "clip_t", "vbench_q",
        "output_path", "quality_input_path", "error", "log_path", "quality_error", "quality_log_path",
    ]
    with (run_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    validation = validate_reference(all_rows, args.reference_root) if args.validate_reference else None
    report = ["# Acceptance experiment", "", f"- Run root: `{run_root}`", f"- Target GPUs: `{','.join(map(str, gpus))}`"]
    for pair_id, manifest in run_manifest.get("cache_manifests", {}).items():
        report.append(f"- Shared cache `{pair_id}`: `{manifest['sha256']}` ({manifest['path']})")
    report.extend(["", "| Baseline | Status | GPU | Denoising (s) | CLIP-T | VBench-Q |", "| --- | --- | ---: | ---: | ---: | ---: |"])
    for row in all_rows:
        report.append(f"| {row.get('method')} | {row.get('status')} | {row.get('actual_gpu', '')} | {row.get('denoising_seconds', '')} | {row.get('clip_t', '')} | {row.get('vbench_q', '')} |")
    if validation is not None:
        report.extend([
            "", "## Reference validation", "",
            "| Baseline | Status | Denoising ref | Denoising actual | Delta | CLIP-T ref | CLIP-T actual | VBench-Q ref | VBench-Q actual |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for check in validation["checks"]:
            report.append(
                "| {baseline} | {status} | {denoising_reference} | {denoising_actual} | {denoising_delta} | "
                "{clip_t_reference} | {clip_t_actual} | {vbench_q_reference} | {vbench_q_actual} |".format(**{
                    key: check.get(key, "") for key in (
                        "baseline", "status", "denoising_reference", "denoising_actual", "denoising_delta",
                        "clip_t_reference", "clip_t_actual", "vbench_q_reference", "vbench_q_actual",
                    )
                })
            )
        report.extend(["", "```json", json.dumps(validation, ensure_ascii=False, indent=2), "```"])
    (run_root / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    run_manifest["status"] = run_status
    write_json(run_root / "run_manifest.json", run_manifest)
    print(json.dumps({"run_root": str(run_root), "summary": str(run_root / "summary.json"), "status": run_status}, ensure_ascii=False))
    return 0 if run_status in {"completed", "dry_run"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
