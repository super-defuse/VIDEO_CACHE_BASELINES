#!/usr/bin/env python3
"""Carnator layer-window ablation on one representative VidProM hit pair.

The 300-request run fixes Carnator's region-update window at ``10-20`` (11 of
Wan2.2-TI2V-5B's 30 layers).  This script sweeps that window on a single hit
pair whose Carnator CLIP-T / VBench-Q sit closest to the 300-request means, so
the layer count is the only variable:

    arm      --sparse-layers   --outer-sparse-layers   layers
    dense    (no reuse at all, 50-step dense target)    --
    l05      13-17             0-12,18-29                5
    l11      10-20             0-9,21-29                11   <- 300-run config
    l15      8-22              0-7,23-29                15
    l20      6-25              0-5,26-29                20
    l30      0-29              (empty)                  30

The windows are nested around layer 15 (the centre of the recorded 10-20
window), with the extra layer of an even count taken on the higher-index side.
``--outer-sparse-layers`` is the complement because the two hook families both
rewrite ``block.forward`` and must stay disjoint; widening the Carnator window
therefore also shrinks outer ``sparge-sage2`` coverage, which is inherent to
the partition rather than a configuration mistake.

Carnator arms go through ``run_vidprom_300_experiment.py`` -- the same harness
the 300-request table was produced with -- so ``latency_s`` / ``denoising_s``
keep that run's accounting (CUDA-synchronised DiT forward, B5 probe counted,
cache prepare excluded).  The layer window reaches the native adapter through
``VCB_CARNATOR_SPARSE_LAYERS`` / ``VCB_CARNATOR_OUTER_SPARSE_LAYERS``.  The
dense arm uses the harness's own ``native_dense_argv`` flag set.

Usage:
    python scripts/ablation/carnator_layer_window_ablation.py            # run + quality + aggregate
    python scripts/ablation/carnator_layer_window_ablation.py --skip-run # re-aggregate only
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# The pair whose Carnator CLIP-T / VBench-Q are closest to the 300-request
# means (0.261571 / 0.640110).  Both values come from the recorded run, not
# from this sweep.
REQUEST_INDEX = 16
LIMIT = 300
TARGET_UUID = "ed46710a-2ea6-5a63-87f9-7e67a9f41743"
TARGET_PROMPT = "A cute little lion and a wobbly penguin appear"
SOURCE_UUID = "e9b92745-1872-59b5-9de5-c839a0196d1b"
SOURCE_PROMPT = "A tiger and a lion are fighting"
SEED_BASE = 20260218
TARGET_SEED = SEED_BASE + REQUEST_INDEX

REFERENCE_300 = {
    "clip_t": 0.2615714518229167,
    "vbench_q": 0.6401098799308389,
    "latency_s": 201.315425576875,
    "denoising_s": 155.31189188570715,
}

ARMS: list[tuple[str, str | None, str | None, int]] = [
    ("dense", None, None, 0),
    ("l05", "13-17", "0-12,18-29", 5),
    ("l11", "10-20", "0-9,21-29", 11),
    ("l15", "8-22", "0-7,23-29", 15),
    ("l20", "6-25", "0-5,26-29", 20),
    ("l25", "3-27", "0-2,28-29", 25),
    ("l30", "0-29", "", 30),
]

QUALITY_DIMENSIONS = [
    "subject_consistency", "background_consistency", "motion_smoothness",
    "dynamic_degree", "aesthetic_quality", "imaging_quality",
]

CONDA = "/data/shared/miniconda/bin/conda"
WAN_ENV = "wan"
QUALITY_ENV = "vbench"


def model_path() -> Path:
    """Resolve the Wan2.2-TI2V-5B checkpoint the 300-request run used."""
    recorded = (
        ROOT / "artifacts" / "vidprom_1000cache_300input_noun_match"
        / "early_signature_disabled_20260903_100305" / "carnator_boundary_b5" / "run_config.json"
    )
    if recorded.is_file():
        config = json.loads(recorded.read_text(encoding="utf-8"))
        path = Path(config["resolved_config"]["model_path"])
        if path.is_dir():
            return path
    fallback = Path("/home/tangxuebin/models/Wan2.2-TI2V-5B")
    if fallback.is_dir():
        return fallback
    raise FileNotFoundError("cannot locate the Wan2.2-TI2V-5B checkpoint")


def run_logged(cmd: list[str], log_path: Path, env: dict[str, str] | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as stream:
        stream.write(f"Command: {' '.join(cmd)}\nStarted: {datetime.now().isoformat()}\n")
        stream.write("=" * 80 + "\n\n")
        stream.flush()
        started = time.perf_counter()
        result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, text=True)
        stream.write(
            f"\n\n{'=' * 80}\nExit code: {result.returncode}\n"
            f"Elapsed: {time.perf_counter() - started:.2f}s\n"
        )
    return result.returncode


def base_env(**extra: str) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPATH"] = str(ROOT)
    env.update(extra)
    return env


def arm_root(output_root: Path, arm: str) -> Path:
    return output_root / arm


def carnator_output(output_root: Path, arm: str) -> Path:
    return arm_root(output_root, arm) / "out" / "carnator_boundary_b5"


def dense_summary_path(output_root: Path, arm: str) -> Path:
    return arm_root(output_root, arm) / "results" / "run" / "summary.json"


def launch_carnator(arm: str, sparse: str, outer: str, gpu: int, output_root: Path) -> tuple[list[str], dict[str, str]]:
    out = arm_root(output_root, arm) / "out"
    cmd = [
        CONDA, "run", "--no-capture-output", "-n", WAN_ENV, "env",
        "PYTHONNOUSERSITE=1", f"PYTHONPATH={ROOT}",
        f"VCB_CARNATOR_SPARSE_LAYERS={sparse}",
        f"VCB_CARNATOR_OUTER_SPARSE_LAYERS={outer}",
        "python", str(ROOT / "scripts" / "benchmark" / "run_vidprom_300_experiment.py"),
        "--method", "carnator_boundary_b5", "--gpu", str(gpu),
        "--limit", str(LIMIT), "--request-indices", str(REQUEST_INDEX),
        "--early-signature-disabled",
        # The plan pins Carnator to GPU 7; this sweep needs one arm per GPU.
        "--allow-gpu-remap",
        "--output-root", str(out), "--overwrite", "--skip-quality",
    ]
    return cmd, base_env()


def launch_dense(arm: str, gpu: int, output_root: Path) -> tuple[list[str], dict[str, str]]:
    root = arm_root(output_root, arm)
    artifacts = root / "artifacts"
    results = root / "results"
    artifacts.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    # Flag set copied from native_dense_argv() in run_vidprom_300_experiment.py.
    cmd = [
        CONDA, "run", "--no-capture-output", "-n", WAN_ENV, "env",
        "PYTHONNOUSERSITE=1", f"PYTHONPATH={ROOT}", "CARNATOR_ENV_READY=1",
        "python", "-m", "carnator.adapters.wan_ti2v_5b", "--no-env-setup",
        "--model-dir", str(model_path()), "--device", f"cuda:{gpu}",
        "--artifact-root", str(artifacts), "--results-root", str(results), "--run-id", "run",
        "--target-prompt", TARGET_PROMPT, "--seed", str(TARGET_SEED),
        "--height", "704", "--width", "1280", "--frames", "121", "--fps", "24",
        "--steps", "50", "--scheduler", "unipc", "--flow-shift", "5.0", "--guidance-scale", "5.0",
        "--target-dense-only", "--outer-attn-backend", "dense", "--ffn-backend", "fp",
        "--skip-w8a8-dense-diagnostic", "--disable-progress",
    ]
    return cmd, base_env()


def collect(output_root: Path, arm: str, layers: int, sparse: str | None) -> dict:
    """Read one arm's harness record plus its native summary."""
    record: dict = {"arm": arm, "layers": layers, "sparse_layers": sparse, "status": "missing"}
    if arm == "dense":
        summary_path = dense_summary_path(output_root, arm)
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            run = summary.get("runs", {}).get("target_dense", {})
            record.update({
                "status": summary.get("status", "unknown"),
                "native_video": run.get("output"),
                "dit_s": run.get("dit_forward_seconds"),
                "native_denoising_s": run.get("denoising_seconds"),
                "native_e2e_s": run.get("end_to_end_seconds"),
                "executed_steps": run.get("completed_steps"),
                "harness_denoising_s": run.get("dit_forward_seconds"),
            })
        return record

    out = carnator_output(output_root, arm)
    results_path = out / "results.jsonl"
    if not results_path.is_file():
        return record
    rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line]
    row = rows[0]
    record.update({
        "status": row.get("status"),
        "source_uuid": row.get("source_uuid"),
        "source_matches_expected": row.get("source_uuid") == SOURCE_UUID,
        "latency_s": row.get("latency_s"),
        "denoising_s": row.get("denoising_s"),
        "executed_steps": row.get("executed_steps"),
        "harness_video": row.get("video_path"),
    })
    timing = (row.get("extra") or {}).get("denoising_timing") or {}
    record["target_dit_forward_s"] = timing.get("target_dit_forward_s")
    record["probe_dit_forward_s"] = timing.get("probe_dit_forward_s")
    record["dit_s"] = timing.get("target_dit_forward_s")
    record["harness_denoising_s"] = row.get("denoising_s")

    summary_path = out / "native" / "request-0000" / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        run = summary.get("runs", {}).get("target_latent_reuse", {})
        config = summary.get("config", {})
        record.update({
            "native_video": run.get("output"),
            "native_e2e_s": run.get("end_to_end_seconds"),
            "native_denoising_s": run.get("denoising_seconds"),
            "native_dit_s": run.get("dit_forward_seconds"),
            "native_executed_steps": run.get("completed_steps"),
            "recorded_sparse_layers": config.get("sparse_layers"),
            "recorded_outer_sparse_layers": config.get("outer_sparse_layers"),
            "recorded_region_update_layers": config.get("region_update_layers"),
        })
        calls = ((run.get("timing") or {}).get("sparse_blocks") or {}).get("calls_by_layer")
        if calls is None:
            calls = (run.get("sparse_blocks") or {}).get("calls_by_layer")
        if calls:
            record["sparse_layers_called"] = [int(k) for k in calls]
            record["sparse_layers_called_count"] = len(calls)
    return record


def run_quality(output_root: Path, records: list[dict], gpu: int) -> dict[str, dict]:
    """Score every arm's video in one VBench pass, matching the 300-run口径."""
    staging = output_root / "quality"
    videos = staging / "videos"
    if staging.is_dir():
        shutil.rmtree(staging)
    videos.mkdir(parents=True)

    rows = []
    for index, record in enumerate(records):
        native_video = record.get("native_video")
        if not native_video:
            continue
        source = Path(native_video)
        if not source.is_file():
            # The harness removes the native artifact tree; the copied video is
            # the one the recorded run shipped quality against.
            fallback = record.get("harness_video")
            if not fallback or not Path(fallback).is_file():
                print(f"[quality] {record['arm']}: no video found, skipping", file=sys.stderr)
                continue
            source = Path(fallback)
        name = f"{index:04d}_{record['arm']}.mp4"
        (videos / name).write_bytes(source.read_bytes())
        rows.append({
            "request_index": index,
            "uuid": f"{record['arm']}",
            "status": "ok",
            "prompt": TARGET_PROMPT,
            "output_path": str(videos / name),
            "video_path": str(videos / name),
            "arm": record["arm"],
        })
    if not rows:
        return {}
    (staging / "results.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    cmd = [
        CONDA, "run", "--no-capture-output", "-n", QUALITY_ENV,
        "python", str(ROOT / "scripts" / "benchmark" / "quality.py"), str(staging),
        "--dimension", *QUALITY_DIMENSIONS,
    ]
    code = run_logged(cmd, output_root / "logs" / "quality.log", base_env(CUDA_VISIBLE_DEVICES=str(gpu)))
    if code != 0:
        raise RuntimeError(f"quality evaluation failed, see {output_root / 'logs' / 'quality.log'}")
    scored = {}
    for row in (json.loads(line) for line in (staging / "results.jsonl").read_text(encoding="utf-8").splitlines() if line):
        scored[row["arm"]] = row
    return scored


def fmt(value, digits: int = 3, dash: str = "n/a") -> str:
    return dash if value is None else f"{value:.{digits}f}"


def write_report(output_root: Path, records: list[dict]) -> None:
    ordered = {record["arm"]: record for record in records}
    dense = ordered["dense"]
    # The dense arm has no probe and no cache prepare, so its target-stage DiT
    # forward and end-to-end are already in the 300-request convention.
    dense_dit = dense.get("dit_s")
    dense_denoising = dense.get("denoising_s") or dense.get("harness_denoising_s")
    dense_e2e = dense.get("latency_s") or dense.get("native_e2e_s")

    def ratio(numerator, denominator):
        return (numerator / denominator) if numerator and denominator else None

    rows = []
    for arm, _sparse, _outer, layers in ARMS:
        record = ordered[arm]
        denoising = record.get("denoising_s") if arm != "dense" else dense_denoising
        e2e = record.get("latency_s") if arm != "dense" else dense_e2e
        dit = record.get("dit_s")
        target_e2e = record.get("native_e2e_s")
        rows.append({
            "arm": arm,
            "layers": layers,
            "sparse_layers": record.get("sparse_layers"),
            "outer_sparse_layers": record.get("recorded_outer_sparse_layers") if arm != "dense" else None,
            "denoising_s": denoising,
            "denoising_speedup": ratio(dense_denoising, denoising),
            "e2e_s": e2e,
            "e2e_speedup": ratio(dense_e2e, e2e),
            "dit_s": dit,
            "dit_speedup": ratio(dense_dit, dit),
            "target_stage_e2e_s": target_e2e,
            "executed_steps": record.get("executed_steps"),
            "sparse_layers_called_count": record.get("sparse_layers_called_count"),
            "status": record.get("status"),
            "clip_t": record.get("clip_t"),
            "vbench_q": record.get("vbench_q"),
        })

    csv_path = output_root / "layer_window_ablation.csv"
    fields = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8") as stream:
        stream.write(",".join(fields) + "\n")
        for row in rows:
            stream.write(",".join("" if row[f] is None else str(row[f]) for f in fields) + "\n")

    (output_root / "layer_window_ablation.json").write_text(
        json.dumps({"records": records, "table": rows}, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# Carnator layer-window ablation (pair: arrival 16)",
        "",
        f"- target: `{TARGET_PROMPT}` (`{TARGET_UUID}`, seed {TARGET_SEED})",
        f"- source: `{SOURCE_PROMPT}` (`{SOURCE_UUID}`)",
        f"- output: `{output_root}`",
        "",
        "Primary columns follow the 300-request table. `Denoising s` is "
        "`denoising_s` (CUDA-synchronised DiT forward, probe counted, cache "
        "prepare excluded) and `E2E s` is `latency_s` (request wall time, cache "
        "prepare excluded); both are directly comparable to the recorded "
        f"arrival-16 row (denoising {REFERENCE_300['denoising_s']:.2f} s, "
        f"E2E {REFERENCE_300['latency_s']:.2f} s) and to the 300-request "
        f"Carnator means (CLIP-T {REFERENCE_300['clip_t']:.6f}, VBench-Q "
        f"{REFERENCE_300['vbench_q']:.6f}). Speedups divide the dense arm's own "
        "numbers by each arm's.",
        "",
        "Secondary columns are the native target-stage values. `DiT s` is "
        "`dit_forward_seconds` for the reuse stage alone: dense runs 50 dense "
        "steps against each arm's 40 reuse steps, so its speedup is an "
        "equal-work DiT comparison. The probe runs in its own stage, which is "
        "why the target-stage `E2E s` sits below `Denoising s`.",
        "",
        "| arm | layers | `--sparse-layers` | Denoising s | Denoising speedup | E2E s | E2E speedup | DiT s | DiT speedup | target-stage E2E s | CLIP-T | VBench-Q |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['arm']} | {row['layers']} | `{row['sparse_layers'] or '—'}` | "
            f"{fmt(row['denoising_s'])} | {fmt(row['denoising_speedup'], 2)}x | "
            f"{fmt(row['e2e_s'])} | {fmt(row['e2e_speedup'], 2)}x | "
            f"{fmt(row['dit_s'])} | {fmt(row['dit_speedup'], 2)}x | "
            f"{fmt(row['target_stage_e2e_s'])} | {fmt(row['clip_t'], 4)} | {fmt(row['vbench_q'], 4)} |"
        )
    lines += [
        "",
        "## Reference",
        "",
        f"300-request Carnator means: CLIP-T {REFERENCE_300['clip_t']:.6f}, "
        f"VBench-Q {REFERENCE_300['vbench_q']:.6f}; recorded `arrival 16` row: "
        f"CLIP-T 0.267578, VBench-Q 0.641478, E2E {REFERENCE_300['latency_s']:.3f} s, "
        f"denoising {REFERENCE_300['denoising_s']:.3f} s, executed_steps 45.",
        "",
    ]
    (output_root / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "artifacts" / f"pair16_carnator_layer_window_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--gpus", nargs="+", type=int, default=[3, 4, 5, 6, 7])
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument("--skip-quality", action="store_true")
    parser.add_argument("--dense-gpu", type=int, default=None, help="Defaults to the first --gpus entry.")
    args = parser.parse_args()

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    gpus = list(args.gpus)
    dense_gpu = args.dense_gpu if args.dense_gpu is not None else gpus[0]

    print(f"output root: {output_root}")
    print(f"gpus: {gpus} (dense on {dense_gpu})")

    if not args.skip_run:
        print("\n[1/4] dense 50-step target (1.00x reference)")
        cmd, env = launch_dense("dense", dense_gpu, output_root)
        code = run_logged(cmd, output_root / "logs" / "dense.log", env)
        print(f"      exit {code}")

        print("\n[2/4] Carnator layer arms in parallel")
        carnator_arms = [arm for arm in ARMS if arm[0] != "dense"]
        if len(carnator_arms) > len(gpus):
            raise SystemExit(f"{len(carnator_arms)} arms exceed {len(gpus)} GPUs; widen --gpus")
        procs = []
        for (arm, sparse, outer, _layers), gpu in zip(carnator_arms, gpus, strict=False):
            cmd, env = launch_carnator(arm, sparse, outer, gpu, output_root)
            log_path = output_root / "logs" / f"{arm}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            stream = log_path.open("w", encoding="utf-8")
            stream.write(f"Command: {' '.join(cmd)}\nGPU: {gpu}\nStarted: {datetime.now().isoformat()}\n")
            stream.write("=" * 80 + "\n\n")
            stream.flush()
            procs.append((arm, subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, text=True), stream, gpu))
            print(f"      {arm}: gpu {gpu}, log {log_path}")
        for arm, proc, stream, gpu in procs:
            code = proc.wait()
            stream.write(f"\n{'=' * 80}\nExit code: {code}\n")
            stream.close()
            print(f"      {arm}: exit {code}")

    print("\n[3/4] collecting metrics")
    records = [collect(output_root, arm, layers, sparse) for arm, sparse, _outer, layers in ARMS]
    for record in records:
        notes = []
        if record.get("status") != "ok" and record.get("status") != "completed":
            notes.append(f"status={record.get('status')}")
        if record.get("status") == "ok" and record.get("source_matches_expected") is False:
            notes.append(f"SOURCE MISMATCH: {record.get('source_uuid')}")
        print(f"      {record['arm']:>6}: {record.get('status')} {' '.join(notes)}")

    if not args.skip_quality:
        print("\n[4/4] quality evaluation (CLIP-T + 6 VBench dimensions)")
        scored = run_quality(output_root, records, dense_gpu)
        for record in records:
            row = scored.get(record["arm"])
            if row:
                record["clip_t"] = row.get("clip_t")
                record["vbench_q"] = row.get("vbench_q")
                record["vbench_dimensions"] = row.get("vbench_dimensions")
            print(f"      {record['arm']:>6}: clip_t={fmt(record.get('clip_t'), 4)} vbench_q={fmt(record.get('vbench_q'), 4)}")
    else:
        print("\n[4/4] quality skipped")

    write_report(output_root, records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
