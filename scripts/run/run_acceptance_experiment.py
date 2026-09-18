#!/usr/bin/env python3
"""Run the five Wan2.2 acceptance baselines for one or more prompt pairs.

Each baseline owns one physical GPU in the 3--7 range and runs a
self-contained pipeline: prepare its own source cache, run its target against
that cache, then drop the cache so it cannot accumulate across prompt pairs.
The orchestrator launches every worker with a one-GPU ``CUDA_VISIBLE_DEVICES``
mask, so physical GPUs outside the configured 3--7 set cannot be touched.

``dense_baseline`` is target-only (it consumes no source cache) so it skips the
source stage entirely and doubles as the shared 50-step speedup denominator.
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
    "chorus",
    "chorus2",
    "carnator_boundary_b5",
)
# Mirrors scripts/benchmark/run_vidprom_300_experiment.py:39 (WAN21_GPU_BY_METHOD),
# whose "baseline" key is this file's "dense_baseline".  One worker per baseline,
# one physical GPU each; GPUs 0--2 stay untouched by this experiment.
GPU_BY_METHOD = {
    "dense_baseline": 3,
    "nirvana_vid": 4,
    "chorus": 5,
    "chorus2": 6,
    "carnator_boundary_b5": 7,
}
# Every baseline except dense consumes a source cache, so every one of these
# prepares (and later prunes) its own independent source run.
SOURCE_METHODS = ("nirvana_vid", "chorus", "chorus2", "carnator_boundary_b5")
# Nested run id, so the source stage lands in a sibling tree instead of
# pre-creating the target's own run directory.  A source run id of "<method>"
# would make shared_cache_wan_baseline's bare artifact_run.mkdir(parents=True)
# raise FileExistsError before the target could ever start.
SOURCE_RUN_ID = "_sources/{method}"
# Source files hashed before the cache is pruned.  These are hashed instead of
# the manifests because both manifests embed absolute per-baseline paths, so
# their digests would differ by construction rather than by content.
SOURCE_FINGERPRINT_FILES = (
    "cache/source_dense/latents_initial.pt",
    "cache/source_dense/latents_step_009.pt",
    "cache/source_dense/latents_step_049.pt",
    "cache/source_dense/fusion_masks/masks.pt",
    "cache/source_dense/outer_block_masks/metadata.json",
    "videos/source_dense.mp4",
)
# metadata.json is the one fingerprint target whose JSON body carries run
# identity, so it is hashed through normalized_json_digest rather than raw.
SOURCE_FINGERPRINT_DROP_KEYS = ("source_run_id", "created_at")


@dataclass(frozen=True)
class Pair:
    pair_id: str
    source_prompt: str
    target_prompt: str
    seed: int
    # Carnator's source-mask probe phrase.  It selects which source tokens the
    # fusion mask is built from, so it changes the region the target is steered
    # into: "dog" pins the region to the dog, while empty means every source
    # token and therefore the whole frame.  Per pair, because the subject noun
    # changes with the prompt.
    source_positive: str | None = None


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-prompt")
    p.add_argument("--target-prompt")
    p.add_argument("--seed", type=int, default=20260218)
    p.add_argument("--pairs", type=Path, help="CSV or JSONL with pair_id, source_prompt, target_prompt, seed")
    p.add_argument("--model-path", type=Path, default=ROOT / "models" / "Wan2.2-TI2V-5B")
    p.add_argument("--run-root", type=Path, default=ROOT / "artifacts" / "acceptance")
    p.add_argument("--gpus", default="3,4,5,6,7", help="Physical GPUs this experiment may occupy; 3-7 only")
    p.add_argument(
        "--allow-gpu-remap",
        action="store_true",
        help="Assign baselines to --gpus in the order given instead of the fixed 3-7 plan",
    )
    p.add_argument(
        "--keep-source-cache",
        action="store_true",
        help="Never prune a source cache; by default only failed baselines retain theirs",
    )
    p.add_argument(
        "--source-positive",
        default=None,
        help=(
            "Carnator source-mask phrase for every pair, overriding --pairs. When unset, each "
            "pair's own source_positive is required; an empty phrase means all source tokens"
        ),
    )
    p.add_argument("--quality-env", default="vbench")
    p.add_argument(
        "--quality-gpu",
        type=int,
        default=None,
        help="Pin every VBench job to one GPU; by default each baseline scores on its own",
    )
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
    if not gpus or any(gpu not in (3, 4, 5, 6, 7) for gpu in gpus):
        raise ValueError("this acceptance runner is restricted to physical GPUs 3, 4, 5, 6 and 7")
    if len(set(gpus)) != len(gpus):
        raise ValueError("--gpus cannot contain duplicates")
    return gpus


def gpu_assignment(gpus: list[int], *, allow_remap: bool) -> dict[str, int]:
    """Map each baseline onto a physical GPU, defaulting to the fixed 3-7 plan."""

    if allow_remap:
        if len(gpus) < len(BASELINES):
            raise ValueError(f"--gpus needs at least {len(BASELINES)} devices to remap onto")
        return {method: gpus[index] for index, method in enumerate(BASELINES)}
    missing = sorted(set(GPU_BY_METHOD.values()) - set(gpus))
    if missing:
        raise ValueError(
            f"--gpus is missing the planned device(s) {missing}; "
            "pass the full 3,4,5,6,7 set or use --allow-gpu-remap"
        )
    return dict(GPU_BY_METHOD)


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
            pair_id = str(row.get("pair_id") or f"pair-{index:03d}")
            phrase = str(row.get("source_positive") or "").strip()
            # An empty phrase is not neutral: it selects every source token,
            # which builds a whole-frame mask and measurably degrades Carnator.
            # Requiring it unless the CLI overrides every pair keeps that from
            # arriving as a silent quality change.
            if not phrase and args.source_positive is None:
                raise ValueError(
                    f"{path}: pair {pair_id!r} has no source_positive; set it to the subject noun "
                    'or pass --source-positive "" to explicitly select all source tokens'
                )
            pairs.append(
                Pair(
                    pair_id,
                    str(row["source_prompt"]),
                    str(row["target_prompt"]),
                    int(row.get("seed", args.seed)),
                    phrase or None,
                )
            )
        if not pairs:
            raise ValueError(f"no prompt pairs found in {path}")
        return pairs
    if not args.source_prompt or not args.target_prompt:
        raise ValueError("provide --source-prompt and --target-prompt, or --pairs")
    return [Pair("pair-000", args.source_prompt, args.target_prompt, args.seed, args.source_positive)]


def resolve_source_positive(pair: Pair, args: argparse.Namespace) -> str:
    """The mask phrase for a pair: the global flag wins, then the pair, then all tokens."""

    if args.source_positive is not None:
        return args.source_positive
    return pair.source_positive if pair.source_positive is not None else ""


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


FRAME_INDICES: tuple[int, ...] = (0, 60, 120)


def sample_frames(
    video: Path, output_dir: Path, stem: str, indices: tuple[int, ...] = FRAME_INDICES
) -> list[Path]:
    """Write `indices` of `video` as `<stem>_frame_<index>.png`.

    Frames are read sequentially rather than seeking: `cv2.CAP_PROP_POS_FRAMES` is
    unreliable on the H.264 files the runners emit, and 121 frames decode in well
    under a second.
    """
    import cv2

    reader = cv2.VideoCapture(str(video))
    written: list[Path] = []
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        for index in range(max(indices) + 1):
            ok, frame = reader.read()
            if not ok:
                raise RuntimeError(f"could not decode frame {index} of {video}")
            if index not in indices:
                continue
            output = output_dir / f"{stem}_frame_{index:03d}.png"
            if not cv2.imwrite(str(output), frame):
                raise RuntimeError(f"could not write frame: {output}")
            written.append(output)
    finally:
        reader.release()
    return written


def copy_video(source: Path, destination: Path) -> Path:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    return destination


def source_run_dir(pair_root: Path, method: str) -> Path:
    return pair_root / "_sources" / method


def source_manifest_path(source_run: Path) -> Path:
    return source_run / "cache" / "shared_dense" / "manifest.json"


def dense_manifest_path(pair_root: Path) -> Path:
    """The 50-step C0 target manifest every baseline speeds up against.

    Produced by dense_baseline in the first scheduling phase, so it is on disk
    before any consumer of it starts.
    """

    return pair_root / "dense_baseline" / "cache" / "target_dense" / "manifest.json"


def source_stage(pair_root: Path, pair: Pair, args: argparse.Namespace, method: str, gpu: int) -> Path:
    """Prepare one baseline's private source cache and return its run directory."""

    source_run = source_run_dir(pair_root, method)
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
        SOURCE_RUN_ID.format(method=method),
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
        # Only ever removes this baseline's own _sources/<method> and its
        # matching _native_results run, so a retry inside the same run root
        # does not trip export_shared_dense_cache's "already exists" guard.
        "--clean",
    ]
    # Passing an empty phrase makes the native adapter select every real source
    # token.  It remains usable for arbitrary prompts rather than assuming "dog".
    command.extend(("--source-positive", resolve_source_positive(pair, args)))
    run_command(
        f"source:{method}",
        command,
        gpu=gpu,
        log_path=pair_root / "logs" / f"source_{method}.log",
        dry_run=args.dry_run,
    )
    manifest = source_manifest_path(source_run)
    if not args.dry_run:
        if not manifest.is_file():
            raise FileNotFoundError(f"source stage for {method} did not produce a shared manifest: {manifest}")
        shutil.copy2(manifest, source_run / "manifest.json")
        source_video = source_run / "videos" / "source_dense.mp4"
        if source_video.is_file():
            # The video stays at its native path: the retained manifest's
            # consumers.chorus.source_video points at it and quality/figures
            # read it after the cache is pruned.
            sample_frames(source_video, source_run, "source")
    return source_run


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _strip_run_identity(value: Any) -> Any:
    """Erase the fields that differ between two runs of the same source."""

    if isinstance(value, dict):
        return {
            key: _strip_run_identity(item)
            for key, item in sorted(value.items())
            if key not in SOURCE_FINGERPRINT_DROP_KEYS
        }
    if isinstance(value, list):
        return [_strip_run_identity(item) for item in value]
    if isinstance(value, str) and value.startswith("/"):
        return "<abs-path>"
    return value


def normalized_json_digest(path: Path) -> str | None:
    """Digest a JSON sidecar with its run identity and absolute paths removed.

    `outer_block_masks/metadata.json` records `source_run_id` and the absolute
    cache paths, so two bit-identical sources still hash differently.  Stripping
    those keeps the payload that has to match -- the per-mask `sha256` entries --
    and drops the fields that are guaranteed to differ by construction.
    """

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    canonical = json.dumps(_strip_run_identity(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def source_fingerprint(source_run: Path) -> dict[str, str | None]:
    """Digest the source artifacts, normalising the JSON sidecars first."""

    fingerprint: dict[str, str | None] = {}
    for name in SOURCE_FINGERPRINT_FILES:
        path = source_run / name
        if not path.is_file():
            fingerprint[name] = None
        elif path.suffix == ".json":
            fingerprint[name] = normalized_json_digest(path)
        else:
            fingerprint[name] = file_sha256(path)
    return fingerprint


def prune_source_cache(source_run: Path, *, keep: bool, dry_run: bool) -> dict[str, Any]:
    """Drop a baseline's source cache, keeping its video, first frame and manifest."""

    cache = source_run / "cache"
    if keep or dry_run or not cache.is_dir():
        return {"source_cache_pruned": False, "source_cache_bytes_freed": None, "source_cache_path": str(cache)}
    freed = directory_bytes(cache)
    shutil.rmtree(cache)
    return {"source_cache_pruned": True, "source_cache_bytes_freed": freed, "source_cache_path": str(cache)}


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
        "end_to_end_seconds": timing.get("end_to_end_seconds"),
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
        sample_frames(normalized, run_dir / "frames", "target")
    return row


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def generic_stage(
    pair_root: Path, pair: Pair, args: argparse.Namespace, method: str, gpu: int, source_run: Path
) -> dict[str, Any]:
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
        str(source_manifest_path(source_run)),
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
            "shared_cache_manifest": str(source_manifest_path(source_run)),
        })
        return row
    rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != 1:
        raise ValueError(f"expected one result row in {results_path}")
    row = rows[0]
    row["pair_id"] = pair.pair_id
    row["actual_gpu"] = gpu
    row["end_to_end_seconds"] = row.get("latency_s")
    timing = row.get("extra", {}).get("timing", {})
    generation = timing.get("generation_s")
    excluded = timing.get("excluded_preprocess_s")
    if not excluded:
        excluded = timing.get("preprocess_breakdown_s", 0.0)
    row["denoising_seconds"] = (float(generation) - float(excluded)) if generation is not None else None
    write_jsonl(results_path, [row])
    output = Path(row["output_path"])
    sample_frames(output, run_dir / "frames", "target")
    return row


def chorus2_stage(
    pair_root: Path, pair: Pair, args: argparse.Namespace, gpu: int, source_run: Path
) -> dict[str, Any]:
    shared = (
        read_json(source_manifest_path(source_run))
        if not args.dry_run
        else {"consumers": {"chorus2": {"source_mask_cache": "planned"}}}
    )
    mask = (
        Path(shared["consumers"]["chorus2"]["source_mask_cache"])
        if not args.dry_run
        else source_run / "cache" / "source_dense" / "outer_block_masks"
    )
    dense_manifest = dense_manifest_path(pair_root)
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
    ]
    if dense_manifest.is_file() or args.dry_run:
        command.extend(("--speedup-reference-manifest", str(dense_manifest)))
    command.extend([
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
    ])
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
        "end_to_end_seconds": target.get("end_to_end_seconds"),
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
    sample_frames(output, run_dir / "frames", "target")
    return row


def carnator_stage(
    pair_root: Path, pair: Pair, args: argparse.Namespace, gpu: int, source_run: Path
) -> dict[str, Any]:
    run_dir = pair_root / "carnator_boundary_b5"
    dense_manifest = dense_manifest_path(pair_root)
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
        str(source_run),
        "--skip-w8a8-dense-diagnostic",
        "--runtime-python",
        sys.executable,
    ]
    if dense_manifest.is_file() or args.dry_run:
        command.extend(("--target-dense-manifest", str(dense_manifest)))
    command.extend(("--source-positive", resolve_source_positive(pair, args)))
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
            "source_run_dir": str(source_run),
            "target_dense_manifest": str(dense_manifest),
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
        "end_to_end_seconds": target.get("end_to_end_seconds"),
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
        "source_run_dir": str(source_run),
        "target_dense_manifest": str(dense_manifest),
    })
    sample_frames(normalized, run_dir / "frames", "target")
    return row


def quality_stage(pair_root: Path, args: argparse.Namespace, baseline: str, gpu: int) -> None:
    command = [sys.executable, "scripts/benchmark/quality.py", str(pair_root / baseline), "--quality-env", args.quality_env]
    run_command(
        f"quality:{baseline}",
        command,
        gpu=gpu,
        log_path=pair_root / "logs" / f"quality_{baseline}.log",
        dry_run=args.dry_run,
    )


def quality_gpu_for(args: argparse.Namespace, gpu_of: dict[str, int], baseline: str) -> int:
    """Each baseline scores on its own card unless --quality-gpu pins them all to one."""

    return args.quality_gpu if args.quality_gpu is not None else gpu_of[baseline]


def reference_scores(reference_root: Path) -> dict[str, dict[str, float]]:
    path = reference_root / "quality" / "summary.json"
    if not path.is_file():
        return {}
    payload = read_json(path)
    return {str(key): value for key, value in payload.get("scores", {}).items() if isinstance(value, dict)}


def reference_table_scores() -> dict[str, dict[str, float]]:
    """Use the frozen acceptance table for baselines absent from quality/summary."""
    path = ROOT / "docs" / "ACCEPTANCE.md"
    aliases = {"Dense C0": "dense_baseline", "NIRVANA-VID": "nirvana_vid", "Chorus": "chorus", "Chorus2": "chorus2", "Carnator Boundary B5": "carnator_boundary_b5"}
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


class StageError(RuntimeError):
    """A baseline stage failed, optionally carrying a source cache kept behind."""

    def __init__(self, message: str, *, source_run: Path | None = None) -> None:
        super().__init__(message)
        self.source_run = source_run


def error_row(pair_root: Path, pair: Pair, method: str, gpu: int, exc: BaseException) -> dict[str, Any]:
    row = {
        "pair_id": pair.pair_id,
        "method": method,
        "actual_gpu": gpu,
        "status": "error",
        "error": f"{type(exc).__name__}: {exc}",
        "log_path": str(pair_root / "logs" / f"{method}.log"),
    }
    if isinstance(exc, StageError) and exc.source_run is not None:
        row["source_run_dir"] = str(exc.source_run)
    return row


def attach_speedups(rows: list[dict[str, Any]]) -> None:
    """Derive every speedup from the collected dense row rather than a child process."""

    dense = next(
        (row for row in rows if row.get("method") == "dense_baseline" and row.get("status") != "error"),
        None,
    )
    base_denoise = (dense or {}).get("denoising_seconds")
    base_end_to_end = (dense or {}).get("end_to_end_seconds")
    for row in rows:
        row.setdefault("speedup_vs_dense", None)
        row.setdefault("speedup_end_to_end_vs_dense", None)
        if dense is None or row.get("status") == "error":
            continue
        denoising = row.get("denoising_seconds")
        end_to_end = row.get("end_to_end_seconds")
        if base_denoise and denoising:
            row["speedup_vs_dense"] = float(base_denoise) / float(denoising)
        if base_end_to_end and end_to_end:
            row["speedup_end_to_end_vs_dense"] = float(base_end_to_end) / float(end_to_end)


def mean_by_baseline(rows: list[dict[str, Any]]) -> list[str]:
    """Average each baseline over the pairs it completed.

    A single pair cannot separate a baseline's quality from its luck on that
    prompt, so the multi-pair report carries the mean and the spread next to the
    per-pair rows instead of leaving the reader to average them by hand.
    """

    report = [
        "",
        "## Mean across pairs",
        "",
        "| Baseline | Pairs | Mean denoising (s) | Mean speedup (denoise) | Mean CLIP-T | CLIP-T range | Mean VBench-Q | VBench-Q range |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in BASELINES:
        scored = [row for row in rows if row.get("method") == method and row.get("status") != "error"]
        if not scored:
            continue
        denoise = [float(row["denoising_seconds"]) for row in scored if row.get("denoising_seconds") is not None]
        speedup = [float(row["speedup_vs_dense"]) for row in scored if row.get("speedup_vs_dense") is not None]
        clip_t = [float(row["clip_t"]) for row in scored if row.get("clip_t") is not None]
        vbench_q = [float(row["vbench_q"]) for row in scored if row.get("vbench_q") is not None]

        def mean(values: list[float]) -> str:
            return f"{sum(values) / len(values):.4f}" if values else ""

        def spread(values: list[float]) -> str:
            return f"{min(values):.4f} - {max(values):.4f}" if values else ""

        report.append(
            f"| {method} | {len(scored)} | {mean(denoise) if denoise else ''} | {mean(speedup)} | "
            f"{mean(clip_t) if clip_t else ''} | {spread(clip_t)} | "
            f"{mean(vbench_q) if vbench_q else ''} | {spread(vbench_q)} |"
        )
    return report


def source_consistency_report(rows: list[dict[str, Any]]) -> list[str]:
    """Report whether each pair's independent source runs stayed identical.

    Every cache-consuming baseline now prepares its own source trajectory, so a
    cross-baseline comparison is only clean while those trajectories agree.
    Drift here is the one failure mode the parallel layout introduces, so it is
    surfaced explicitly rather than assumed away.
    """

    by_pair: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        method = row.get("method")
        fingerprint = row.get("source_fingerprint")
        if method in SOURCE_METHODS and fingerprint:
            by_pair.setdefault(str(row.get("pair_id")), {})[method] = fingerprint
    if not by_pair:
        return []
    report = ["", "## Source consistency", ""]
    for pair_id, per_method in sorted(by_pair.items()):
        consistent = len({json.dumps(item, sort_keys=True) for item in per_method.values()}) == 1
        report.append(
            f"- `{pair_id}`: {'**PASS**' if consistent else '**WARN**'} - "
            f"{len(per_method)} independent source runs ({', '.join(f'`{name}`' for name in sorted(per_method))})"
        )
        if not consistent:
            report.append("  - source runs differ, so cross-baseline comparison carries source drift")
        for name in SOURCE_FINGERPRINT_FILES:
            values = {method: fingerprint.get(name) for method, fingerprint in sorted(per_method.items())}
            if len({json.dumps(value) for value in values.values()}) > 1:
                report.append(f"  - `{name}` differs:")
                report.extend(f"    - `{method}`: `{value}`" for method, value in values.items())
    return report


def run_pair(pair_root: Path, pair: Pair, args: argparse.Namespace, gpu_of: dict[str, int]) -> list[dict[str, Any]]:
    """Run one prompt pair across the configured GPUs.

    Phase 1 runs the dense 50-step control alongside one private source run per
    cache-consuming baseline.  Every target in phase 2 therefore finds a
    complete ``dense_baseline/cache/target_dense/manifest.json`` to speed up
    against, so chorus2 and carnator keep their reference instead of each
    re-deriving a 50-step dense control of their own.
    """

    rows: list[dict[str, Any]] = []
    sources: dict[str, Path] = {}

    def record(method: str, exc: BaseException) -> None:
        rows.append(error_row(pair_root, pair, method, gpu_of[method], exc))

    # Phase 1: one worker per physical GPU, so nothing contends for a device.
    with ThreadPoolExecutor(max_workers=len(BASELINES)) as pool:
        phase_one = {
            pool.submit(dense_stage, pair_root, pair, args, gpu_of["dense_baseline"]): "dense_baseline"
        }
        for method in SOURCE_METHODS:
            phase_one[pool.submit(source_stage, pair_root, pair, args, method, gpu_of[method])] = method
        for future in as_completed(phase_one):
            method = phase_one[future]
            try:
                result = future.result()
            except Exception as exc:
                record(method, exc)
            else:
                if method == "dense_baseline":
                    rows.append(result)
                else:
                    sources[method] = result

    # Phase 2: each surviving source cache feeds its own target on its own GPU.
    def run_target(method: str) -> dict[str, Any]:
        source_run = sources[method]
        try:
            if method in {"nirvana_vid", "chorus"}:
                row = generic_stage(pair_root, pair, args, method, gpu_of[method], source_run)
            elif method == "chorus2":
                row = chorus2_stage(pair_root, pair, args, gpu_of[method], source_run)
            else:
                row = carnator_stage(pair_root, pair, args, gpu_of[method], source_run)
        except Exception as exc:
            # Keep the source cache so the target can be retried without paying
            # for the source stage again; the row records where it was left.
            raise StageError(str(exc), source_run=source_run) from exc
        fingerprint = {} if args.dry_run else source_fingerprint(source_run)
        row.update(prune_source_cache(source_run, keep=args.keep_source_cache, dry_run=args.dry_run))
        row["source_run_dir"] = str(source_run)
        row["source_fingerprint"] = fingerprint
        return row

    with ThreadPoolExecutor(max_workers=len(SOURCE_METHODS)) as pool:
        phase_two = {pool.submit(run_target, method): method for method in sources}
        for future in as_completed(phase_two):
            method = phase_two[future]
            try:
                rows.append(future.result())
            except Exception as exc:
                record(method, exc)

    rows.sort(key=lambda row: BASELINES.index(row["method"]) if row.get("method") in BASELINES else 999)
    attach_speedups(rows)
    if not args.skip_quality:
        failed = {row.get("method") for row in rows if row.get("status") == "error"}
        eligible = [
            baseline for baseline in BASELINES
            if (pair_root / baseline / "results.jsonl").is_file() and baseline not in failed
        ]

        def score(baseline: str) -> dict[str, Any]:
            quality_stage(pair_root, args, baseline, quality_gpu_for(args, gpu_of, baseline))
            lines = (pair_root / baseline / "results.jsonl").read_text(encoding="utf-8").splitlines()
            scored = [json.loads(line) for line in lines if line]
            if not scored:
                return {}
            return {key: scored[0].get(key) for key in ("clip_t", "vbench_q", "vbench_dimensions")}

        # Every target has already exited, so the whole 3-7 band is free.
        # Scoring each baseline on its own card runs the slowest step of the
        # pair five-wide instead of queueing it behind a single GPU.
        with ThreadPoolExecutor(max_workers=len(eligible)) as pool:
            futures = {pool.submit(score, baseline): baseline for baseline in eligible}
            for future in as_completed(futures):
                baseline = futures[future]
                target_row = next(row for row in rows if row.get("method") == baseline)
                try:
                    target_row.update(future.result())
                except Exception as exc:
                    target_row["quality_error"] = f"{type(exc).__name__}: {exc}"
                    target_row["quality_log_path"] = str(pair_root / "logs" / f"quality_{baseline}.log")
    return rows


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    args.model_path = args.model_path.expanduser().resolve()
    gpus = parse_gpus(args.gpus)
    gpu_of = gpu_assignment(gpus, allow_remap=args.allow_gpu_remap)
    if args.quality_gpu is not None and args.quality_gpu not in gpus:
        raise ValueError(f"--quality-gpu {args.quality_gpu} is outside the configured --gpus set {gpus}")
    if not 0 < args.chorus_k1 < args.chorus_k2 < 50:
        raise ValueError("Chorus boundaries must satisfy 0 < --chorus-k1 < --chorus-k2 < 50")
    pairs = load_pairs(args)
    if args.validate_reference and len(pairs) > 1:
        # The reference table is the frozen dog-to-cat run; comparing other
        # pairs against it would report every baseline as drifted.
        raise ValueError("--validate-reference only applies to the single dog->cat pair")
    run_root = make_run_root(args.run_root)
    run_manifest = {
        "status": "running",
        "created_at": datetime.now().astimezone().isoformat(),
        "root": str(run_root),
        "layout": "per-baseline-source-v1",
        "pairs": [pair.__dict__ for pair in pairs],
        "baselines": list(BASELINES),
        "gpu_by_method": gpu_of,
        "quality_gpu": args.quality_gpu if args.quality_gpu is not None else "per-baseline",
        "cuda_policy": "CUDA_VISIBLE_DEVICES restricted to physical GPUs 3,4,5,6,7",
        "source_mode": "one independent source run per cache-consuming baseline, pruned after its target",
        "cache_pruning": not args.keep_source_cache,
        "model_path": str(args.model_path),
        "model_dtype": args.model_dtype,
        "source_positive": args.source_positive if args.source_positive is not None else "<per-pair>",
        "source_positive_by_pair": {pair.pair_id: resolve_source_positive(pair, args) for pair in pairs},
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
            rows = run_pair(pair_root, pair, args, gpu_of)
        except Exception as exc:
            # run_pair records per-baseline failures itself, so reaching this
            # handler means the orchestration died before any worker reported.
            rows = [error_row(pair_root, pair, baseline, gpu_of[baseline], exc) for baseline in BASELINES]
        for method in SOURCE_METHODS:
            # Read the retained copy, not cache/shared_dense/manifest.json:
            # the latter is pruned along with the rest of the source cache.
            manifest_path = source_run_dir(pair_root, method) / "manifest.json"
            if manifest_path.is_file():
                run_manifest.setdefault("cache_manifests", {}).setdefault(pair.pair_id, {})[method] = {
                    "path": str(manifest_path),
                    "sha256": file_sha256(manifest_path),
                }
        retained = {row["method"]: row["source_run_dir"] for row in rows if row.get("source_run_dir") and row.get("status") == "error"}
        if retained:
            run_manifest.setdefault("retained_source_caches", {})[pair.pair_id] = retained
        for row in rows:
            row["pair_root"] = str(pair_root)
        all_rows.extend(rows)

    has_failure = any(row.get("status") == "error" or row.get("quality_error") for row in all_rows)
    all_planned = bool(all_rows) and all(row.get("status") == "planned" for row in all_rows)
    run_status = "dry_run" if all_planned else ("partial" if has_failure else "completed")
    write_json(run_root / "summary.json", {"status": run_status, "rows": all_rows})
    fieldnames = [
        "pair_id", "method", "status", "actual_gpu", "denoising_seconds", "end_to_end_seconds",
        "speedup_vs_dense", "clip_t", "vbench_q", "output_path", "quality_input_path",
        "error", "log_path", "quality_error", "quality_log_path",
    ]
    with (run_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    validation = validate_reference(all_rows, args.reference_root) if args.validate_reference else None
    report = [
        "# Acceptance experiment",
        "",
        f"- Run root: `{run_root}`",
        "- Layout: `per-baseline-source-v1`",
        "- GPU plan: " + ", ".join(f"`{method}`={gpu}" for method, gpu in sorted(gpu_of.items(), key=lambda item: item[1])),
        "- Quality GPU: "
        + (
            f"`{args.quality_gpu}` (all baselines pinned to one card)"
            if args.quality_gpu is not None
            else "one per baseline, concurrent on " + ", ".join(str(gpu_of[method]) for method in BASELINES)
        ),
    ]
    if args.keep_source_cache:
        report.append("- Source caches: **retained** (`--keep-source-cache`)")
    else:
        report.append("- Source caches: one independent run per baseline, pruned once its target succeeded")
    for pair in pairs:
        manifests = run_manifest.get("cache_manifests", {}).get(pair.pair_id, {})
        for method, entry in sorted(manifests.items()):
            report.append(f"- Source cache `{pair.pair_id}`/`{method}`: `{entry['sha256']}` ({entry['path']})")
    retained_caches = run_manifest.get("retained_source_caches", {})
    if retained_caches:
        report.extend(["", "## Retained source caches", ""])
        for pair_id, entries in retained_caches.items():
            for method, path in sorted(entries.items()):
                report.append(f"- `{pair_id}`/`{method}` kept for debugging: `{path}`")
    columns = ["Baseline", "Status", "GPU", "Denoising (s)", "End-to-end (s)",
               "Speedup (denoise)", "Speedup (end-to-end)", "CLIP-T", "VBench-Q"]
    alignment = ["---", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]
    if multi:
        columns.insert(0, "Pair")
        alignment.insert(0, "---")
    report.extend(["", "| " + " | ".join(columns) + " |", "| " + " | ".join(alignment) + " |"])
    for row in all_rows:
        denoise_speedup = row.get("speedup_vs_dense")
        end_to_end_speedup = row.get("speedup_end_to_end_vs_dense")
        cells = [str(row.get("method")), str(row.get("status")), str(row.get("actual_gpu", "")),
                 str(row.get("denoising_seconds", "")), str(row.get("end_to_end_seconds", "")),
                 f"{denoise_speedup:.2f}x" if denoise_speedup else "",
                 f"{end_to_end_speedup:.2f}x" if end_to_end_speedup else "",
                 str(row.get("clip_t", "")), str(row.get("vbench_q", ""))]
        if multi:
            # Only the first row of each pair repeats the id, so the group reads
            # as one block instead of five identical cells.
            cells.insert(0, str(row.get("pair_id", "")) if row.get("method") == BASELINES[0] else "")
        report.append("| " + " | ".join(cells) + " |")
    if multi:
        report.extend(mean_by_baseline(all_rows))
    report.extend(source_consistency_report(all_rows))
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
