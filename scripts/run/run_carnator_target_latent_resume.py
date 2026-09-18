#!/usr/bin/env python3
"""Run the Carnator target-prefix latent-resume variant on an existing acceptance run.

The accepted ``carnator_boundary_b5`` baseline starts its target from the *source*
trajectory's latent at step ``reuse_start_step - 1``: the object is inherited from
the source and the mask only reshapes it afterwards.  This script runs a second
baseline that changes that one starting point and nothing else.  Per pair it:

1. prepares Carnator's own source cache with the acceptance runner's own
   ``source_stage`` (same command, same ``--source-only
   --prepare-shared-dense-cache`` flags), so the cache is the one the accepted
   baseline would have built;
2. runs the *target* prompt densely for ``reuse_start_step`` steps and keeps that
   last step's latent;
3. runs the ordinary boundary-blend reuse stage from that target latent, after
   which the source cache is injected through the mask exactly as before.

Injection formula, alpha function, sparse attention, FFN and scheduler are all
untouched, so the variant differs from the accepted baseline by one variable.

Results land beside the accepted ones, under the same five pair directories:
``<root>/<pair>/<method>/`` with its source under ``<root>/<pair>/_sources/``.
The accepted ``summary.json``, ``report.md`` and ``run_manifest.json`` are never
written to; this script writes ``*.carnator_boundary_b5_targetresume.json`` /
``.md`` siblings instead.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reused wholesale so the source stage is provably the same command the accepted
# run issued, rather than a re-implementation that could drift from it.
from scripts.run.run_acceptance_experiment import (  # noqa: E402
    Pair,
    copy_video,
    dense_manifest_path,
    prune_source_cache,
    read_json,
    resolve_source_positive,
    run_command,
    sample_frames,
    source_fingerprint,
    source_stage,
    write_json,
    write_jsonl,
)


METHOD = "carnator_boundary_b5_targetresume"
# The accepted baseline every delta in the report is measured against.
BASELINE_METHOD = "carnator_boundary_b5"
DEFAULT_ROOT = ROOT / "artifacts" / "acceptance" / "20260918_183809"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--acceptance-root", type=Path, default=DEFAULT_ROOT, help="Existing acceptance run root to extend.")
    p.add_argument("--method", default=METHOD, help="Name of the new baseline's result directory.")
    p.add_argument("--gpus", default="3,4,5,6,7", help="Physical GPUs this experiment may occupy; 3-7 only")
    p.add_argument("--pairs", default=None, help="Comma-separated pair ids to run; by default every pair in the run")
    p.add_argument("--model-path", type=Path, default=ROOT / "models" / "Wan2.2-TI2V-5B")
    p.add_argument("--model-dtype", choices=("fp16", "bf16"), default="fp16")
    p.add_argument("--boundary-alpha-function", choices=("smoothstep", "cosine"), default="smoothstep")
    p.add_argument(
        "--source-positive",
        default=None,
        help="Carnator source-mask phrase for every pair, overriding the run's per-pair value.",
    )
    p.add_argument("--quality-env", default="vbench")
    p.add_argument("--skip-quality", action="store_true")
    p.add_argument("--keep-source-cache", action="store_true", help="Never prune a source cache, even after success.")
    p.add_argument("--clean", action="store_true", help="Delete this variant's previous run dirs before starting.")
    p.add_argument("--dry-run", action="store_true")
    return p


def parse_gpus(value: str) -> list[int]:
    try:
        gpus = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("--gpus must be a comma-separated list of integers") from exc
    if not gpus or any(gpu not in (3, 4, 5, 6, 7) for gpu in gpus):
        raise ValueError("this runner is restricted to physical GPUs 3, 4, 5, 6 and 7")
    if len(set(gpus)) != len(gpus):
        raise ValueError("--gpus cannot contain duplicates")
    return gpus


def load_pairs(root: Path, args: argparse.Namespace) -> list[Pair]:
    """Read the pairs the accepted run already ran, from its own prompts.jsonl."""

    path = root / "prompts.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"acceptance root has no prompts.jsonl: {path}")
    selected = {item.strip() for item in (args.pairs or "").split(",") if item.strip()}
    pairs: list[Pair] = []
    for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()):
        pair_id = str(row["pair_id"])
        if selected and pair_id not in selected:
            continue
        phrase = str(row.get("source_positive") or "").strip()
        if not phrase and args.source_positive is None:
            raise ValueError(f"{path}: pair {pair_id!r} has no source_positive and no --source-positive override")
        pairs.append(
            Pair(
                pair_id,
                str(row["source_prompt"]),
                str(row["target_prompt"]),
                int(row.get("seed", 20260218)),
                phrase or None,
            )
        )
    if not pairs:
        raise ValueError(f"no prompt pairs selected from {path}")
    if selected - {pair.pair_id for pair in pairs}:
        raise ValueError(f"--pairs names unknown pair(s): {sorted(selected - {pair.pair_id for pair in pairs})}")
    return pairs


def pair_root_for(root: Path, pair_id: str) -> Path:
    """Where a pair's baselines live inside an acceptance root.

    Multi-pair runs nest each pair under its own directory; a single-pair run
    writes its baselines straight into the root, because the acceptance runner
    only applies the ``<pair_id>/`` prefix when there is more than one pair.
    """

    nested = root / pair_id
    return nested if nested.is_dir() else root


def stage_paths(pair_root: Path, method: str) -> dict[str, Path]:
    artifact_run = pair_root / "ti2v-5b" / method / "B5"
    return {
        "artifact_run": artifact_run,
        "native_summary": pair_root / "_native_results" / "ti2v-5b" / method / "B5" / "summary.json",
        "target_prefix_manifest": artifact_run / "cache" / "target_prefix" / "manifest.json",
    }


def target_stage(
    pair_root: Path, pair: Pair, args: argparse.Namespace, gpu: int, source_run: Path, method: str
) -> dict[str, Any]:
    """Run the variant's target: dense target prefix, then the usual reuse stage."""

    run_dir = pair_root / method
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
        method,
        "--latent-injection-mode",
        "boundary-blend",
        "--boundary-alpha-function",
        args.boundary_alpha_function,
        "--source-run-dir",
        str(source_run),
        # The one behavioural difference from the accepted baseline.
        "--target-resume-from-target-prefix",
        "--skip-w8a8-dense-diagnostic",
        "--runtime-python",
        sys.executable,
    ]
    if dense_manifest.is_file() or args.dry_run:
        # Reuses the accepted run's own 50-step dense control, so the variant is
        # timed against the same C0 reference as every other baseline.
        command.extend(("--target-dense-manifest", str(dense_manifest)))
    if args.clean:
        command.append("--clean")
    command.extend(("--source-positive", resolve_source_positive(pair, args)))

    started = time.perf_counter()
    run_command(
        method,
        command,
        gpu=gpu,
        log_path=pair_root / "logs" / f"{method}.log",
        dry_run=args.dry_run,
        carnator=True,
    )
    wall_seconds = time.perf_counter() - started
    paths = stage_paths(pair_root, method)
    if args.dry_run:
        write_json(run_dir / "config.resolved.json", {
            "method": method,
            "source_prompt": pair.source_prompt,
            "target_prompt": pair.target_prompt,
            "seed": pair.seed,
            "model_path": str(args.model_path),
            "latent_injection_mode": "boundary-blend",
            "baseline": "B5",
            "boundary_alpha_function": args.boundary_alpha_function,
            "source_run_dir": str(source_run),
            "target_dense_manifest": str(dense_manifest),
            "target_resume_from_target_prefix": True,
        })
        return {"pair_id": pair.pair_id, "method": method, "status": "planned", "actual_gpu": gpu}

    summary = read_json(paths["native_summary"])
    target = summary["runs"]["target_latent_reuse"]
    output = Path(target["output"])
    normalized = copy_video(output, run_dir / "videos" / "target.mp4")

    # The stage timings above exclude model load; the wall clock includes it.
    # Recording both keeps the variant's honest cost visible next to the
    # accepted baseline's number, which was measured the same stage-only way.
    prefix_seconds = None
    if paths["target_prefix_manifest"].is_file():
        prefix_seconds = read_json(paths["target_prefix_manifest"]).get("timing", {}).get("end_to_end_seconds")
    reuse_manifest = read_json(Path(target["manifest_path"]))

    row: dict[str, Any] = {
        "request_index": 0,
        "uuid": f"{pair.pair_id}-target-000",
        "pair_id": pair.pair_id,
        "prompt": pair.target_prompt,
        "model": "Wan2.2",
        "method": method,
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
        "target_prefix_end_to_end_seconds": prefix_seconds,
        "variant_end_to_end_seconds": (
            float(prefix_seconds) + float(target["end_to_end_seconds"]) if prefix_seconds is not None else None
        ),
        "target_stage_wall_seconds": wall_seconds,
        "resume_latent_path": (reuse_manifest.get("resume") or {}).get("resume_latent_path"),
        "resume_start_step": (reuse_manifest.get("resume") or {}).get("resume_start_step"),
    }
    write_jsonl(run_dir / "results.jsonl", [row])
    write_json(run_dir / "config.resolved.json", {
        "method": method,
        "source_prompt": pair.source_prompt,
        "target_prompt": pair.target_prompt,
        "seed": pair.seed,
        "model_path": str(args.model_path),
        "latent_injection_mode": "boundary-blend",
        "baseline": "B5",
        "boundary_alpha_function": args.boundary_alpha_function,
        "source_run_dir": str(source_run),
        "target_dense_manifest": str(dense_manifest),
        "target_resume_from_target_prefix": True,
    })
    sample_frames(normalized, run_dir / "frames", "target")
    return row


def quality_stage(pair_root: Path, args: argparse.Namespace, method: str, gpu: int) -> None:
    command = [sys.executable, "scripts/benchmark/quality.py", str(pair_root / method), "--quality-env", args.quality_env]
    run_command(
        f"quality:{method}",
        command,
        gpu=gpu,
        log_path=pair_root / "logs" / f"quality_{method}.log",
        dry_run=args.dry_run,
    )


def run_pair(
    pair_root: Path, pair: Pair, args: argparse.Namespace, gpu: int, method: str
) -> dict[str, Any]:
    """Prepare the source cache, run the variant's target, then score it."""

    started = time.perf_counter()
    source_run = source_stage(pair_root, pair, args, method, gpu)
    source_seconds = time.perf_counter() - started

    row = target_stage(pair_root, pair, args, gpu, source_run, method)

    fingerprint = {} if args.dry_run else source_fingerprint(source_run)
    row.update(prune_source_cache(source_run, keep=args.keep_source_cache, dry_run=args.dry_run))
    row["source_run_dir"] = str(source_run)
    row["source_fingerprint"] = fingerprint
    row["source_stage_seconds"] = source_seconds
    row["pair_wall_seconds"] = time.perf_counter() - started

    if not args.skip_quality and not args.dry_run:
        try:
            quality_stage(pair_root, args, method, gpu)
            scored = [
                json.loads(line)
                for line in (pair_root / method / "results.jsonl").read_text(encoding="utf-8").splitlines()
                if line
            ]
            if scored:
                row.update({key: scored[0].get(key) for key in ("clip_t", "vbench_q", "vbench_dimensions")})
        except Exception as exc:
            row["quality_error"] = f"{type(exc).__name__}: {exc}"
            row["quality_log_path"] = str(pair_root / "logs" / f"quality_{method}.log")
    return row


def error_row(pair_root: Path, pair: Pair, method: str, gpu: int, exc: BaseException) -> dict[str, Any]:
    return {
        "pair_id": pair.pair_id,
        "method": method,
        "actual_gpu": gpu,
        "status": "error",
        "error": f"{type(exc).__name__}: {exc}",
        "log_path": str(pair_root / "logs" / f"{method}.log"),
    }


def accepted_reference(root: Path) -> dict[str, dict[str, Any]]:
    """The accepted run's own rows, keyed by method then pair, for the delta columns."""

    path = root / "summary.json"
    if not path.is_file():
        return {}
    payload = read_json(path)
    table: dict[str, dict[str, Any]] = {}
    for row in payload.get("rows", []):
        table.setdefault(str(row.get("method")), {})[str(row.get("pair_id"))] = row
    return table


def source_consistency(rows: list[dict[str, Any]], reference: dict[str, dict[str, Any]]) -> list[str]:
    """Compare each fresh source cache against the accepted run's recorded fingerprint.

    The new source runs are independent, so this is the check that the variant
    really is being fed the same source trajectory the accepted baseline used.
    """

    accepted = reference.get(BASELINE_METHOD, {})
    lines: list[str] = []
    for row in rows:
        fingerprint = row.get("source_fingerprint")
        if row.get("status") == "error" or not fingerprint:
            continue
        expected = (accepted.get(str(row.get("pair_id"))) or {}).get("source_fingerprint")
        if not expected:
            lines.append(f"- `{row.get('pair_id')}`: **SKIP** - no recorded fingerprint in the accepted summary")
            continue
        differing = sorted(name for name in expected if fingerprint.get(name) != expected.get(name))
        if differing:
            lines.append(f"- `{row.get('pair_id')}`: **WARN** - source artifacts differ from the accepted run")
            for name in differing:
                lines.append(f"  - `{name}`: new `{fingerprint.get(name)}` vs accepted `{expected.get(name)}`")
        else:
            lines.append(f"- `{row.get('pair_id')}`: **PASS** - source cache matches the accepted run byte for byte")
    return lines


# Columns that identify a row rather than measure it, so no mean is meaningful.
NON_AVERAGED_COLUMNS = {"pair_id", "actual_gpu"}


def mean_row(rows: list[dict[str, Any]], cells: list[tuple[str, str, int]]) -> str:
    """Average the columns a table declared, so a mean row cannot drift out of alignment.

    Each cell is ``(key, format, column index)``; columns without a key keep their slot blank.
    """

    scored = [row for row in rows if row.get("status") != "error"]
    if not scored:
        return ""
    rendered = [""] * (1 + max(index for _, _, index in cells))
    rendered[0] = "mean"
    for key, fmt, index in cells:
        if key == "#pairs":
            rendered[index] = str(len(scored))
            continue
        if key in NON_AVERAGED_COLUMNS:
            # Labels rather than measurements: averaging a pair id or a GPU
            # index would print a number that means nothing.
            continue
        values = [float(row[key]) for row in scored if row.get(key) is not None]
        rendered[index] = fmt.format(sum(values) / len(values)) if values else ""
    return "| " + " | ".join(rendered) + " |"


QUALITY_COLUMNS = [
    ("pair_id", "{}", 0),
    ("#pairs", "{}", 1),
    ("actual_gpu", "{}", 2),
    ("clip_t", "{}", 3),
    ("baseline_clip_t", "{}", 4),
    ("clip_t_delta", "{:+.4f}", 5),
    ("vbench_q", "{}", 6),
    ("baseline_vbench_q", "{}", 7),
    ("vbench_q_delta", "{:+.4f}", 8),
]
SPEED_COLUMNS = [
    ("pair_id", "{}", 0),
    ("#pairs", "{}", 1),
    ("actual_gpu", "{}", 2),
    ("source_stage_seconds", "{:.1f}", 3),
    ("target_prefix_end_to_end_seconds", "{:.1f}", 4),
    ("end_to_end_seconds", "{:.1f}", 5),
    ("variant_end_to_end_seconds", "{:.1f}", 6),
    ("speedup_reuse_vs_dense", "{:.2f}x", 7),
    ("speedup_variant_vs_dense", "{:.2f}x", 8),
    ("baseline_speedup_end_to_end", "{:.2f}x", 9),
]


def build_report(root: Path, rows: list[dict[str, Any]], args: argparse.Namespace, gpu_of: dict[str, int]) -> str:
    def render(columns: list[tuple[str, str, int]], header: list[str], alignment: list[str]) -> list[str]:
        table = ["| " + " | ".join(header) + " |", "| " + " | ".join(alignment) + " |"]
        cells = {index: (key, fmt) for key, fmt, index in columns}
        for row in rows:
            line = []
            for index in range(len(header)):
                key, fmt = cells.get(index, ("", "{}"))
                value = row.get(key) if key else None
                line.append("" if value is None else fmt.format(value))
            table.append("| " + " | ".join(line) + " |")
        table.append(mean_row(rows, columns))
        return table

    lines = [
        "# Carnator target-prefix latent-resume variant",
        "",
        f"- Acceptance root: `{root}`",
        f"- Baseline: `{METHOD}`; the accepted `{BASELINE_METHOD}` is the reference column",
        "- Difference: the reuse stage resumes from the **target** prompt's own latent at "
        "step `reuse_start_step - 1` instead of the source trajectory's. Injection formula, alpha "
        "function, sparse attention, FFN and scheduler are unchanged.",
        "- Source stage: the acceptance runner's own `source_stage`, so the cache is built by the same command.",
        "- Pair -> GPU: " + ", ".join(f"`{pair}`={gpu}" for pair, gpu in sorted(gpu_of.items())),
        "- Quality: "
        + ("skipped (`--skip-quality`)" if args.skip_quality else f"`quality.py` per pair, `--quality-env {args.quality_env}`"),
        "",
        "## Quality",
        "",
        *render(
            QUALITY_COLUMNS,
            ["Pair", "Pairs", "GPU", "CLIP-T", "CLIP-T (accepted)", "Δ CLIP-T",
             "VBench-Q", "VBench-Q (accepted)", "Δ VBench-Q"],
            ["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"],
        ),
        "",
        "Positive Δ means the variant scores higher than the accepted baseline on that pair.",
        "",
        "## Speed",
        "",
        *render(
            SPEED_COLUMNS,
            ["Pair", "Pairs", "GPU", "Source (s)", "Prefix (s)", "Reuse E2E (s)", "Variant E2E (s)",
             "Speedup vs dense (reuse)", "Speedup vs dense (variant)", "Speedup (accepted)"],
            ["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"],
        ),
        "",
        "`Reuse E2E` is the reuse stage alone, measured exactly like the accepted baseline's "
        "`end_to_end_seconds`, so `Speedup (reuse)` is directly comparable with `Speedup (accepted)`. "
        "`Variant E2E` adds the dense target-prefix run this variant needs to obtain its starting "
        "latent, so `Speedup (variant)` is the honest cost of the variant as a whole. `Prefix (s)` "
        "is that run's own end-to-end time; the source stage is shared with the accepted baseline "
        "and is therefore excluded from both speedups, as it is in the acceptance table.",
        "",
        "## Source consistency",
        "",
        "Each pair's source cache was rebuilt from scratch and compared against the fingerprint the "
        "accepted run recorded, so a PASS means the variant really was fed the same source trajectory:",
        "",
        *source_consistency(rows, accepted_reference(root)),
    ]
    errors = [row for row in rows if row.get("status") == "error"]
    if errors:
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- `{row['pair_id']}`: {row.get('error')} (see `{row.get('log_path')}`)" for row in errors)
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    args.model_path = args.model_path.expanduser().resolve()
    args.acceptance_root = args.acceptance_root.expanduser().resolve()
    method = args.method
    gpus = parse_gpus(args.gpus)
    root = args.acceptance_root
    if not root.is_dir():
        raise ValueError(f"acceptance root does not exist: {root}")

    pairs = load_pairs(root, args)
    if len(pairs) > len(gpus):
        raise ValueError(f"{len(pairs)} pairs but only {len(gpus)} GPU(s); pairs run one GPU each")

    reference = accepted_reference(root)
    gpu_of = {pair.pair_id: gpus[index] for index, pair in enumerate(pairs)}
    pair_roots = {pair.pair_id: pair_root_for(root, pair.pair_id) for pair in pairs}
    for pair in pairs:
        # A flat single-pair root always exists; a nested one has to be there.
        if pair_roots[pair.pair_id] == root and not (root / "dense_baseline").is_dir():
            raise ValueError(f"{root} is neither a nested pair directory nor a single-pair root")

    print(json.dumps({
        "root": str(root),
        "method": method,
        "pairs": {pair.pair_id: gpu_of[pair.pair_id] for pair in pairs},
        "pair_roots": {pair.pair_id: str(pair_roots[pair.pair_id]) for pair in pairs},
        "started_at": datetime.now().astimezone().isoformat(),
    }, ensure_ascii=False), flush=True)

    def worker(pair: Pair) -> dict[str, Any]:
        pair_root = pair_roots[pair.pair_id]
        try:
            return run_pair(pair_root, pair, args, gpu_of[pair.pair_id], method)
        except Exception as exc:
            return error_row(pair_root, pair, method, gpu_of[pair.pair_id], exc)

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(pairs)) as pool:
        futures = {pool.submit(worker, pair): pair for pair in pairs}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"[{row.get('pair_id')}] {row.get('status')} "
                f"denoise={row.get('denoising_seconds')} variant_e2e={row.get('variant_end_to_end_seconds')} "
                f"clip_t={row.get('clip_t')} vbench_q={row.get('vbench_q')}",
                flush=True,
            )
    order = [pair.pair_id for pair in pairs]
    rows.sort(key=lambda row: order.index(str(row.get("pair_id"))) if str(row.get("pair_id")) in order else len(order))

    # The accepted run's own numbers, attached per row so the report can show deltas
    # against it without a second file to read.
    for row in rows:
        pair_id = str(row.get("pair_id"))
        baseline_row = reference.get(BASELINE_METHOD, {}).get(pair_id, {})
        row["baseline_clip_t"] = baseline_row.get("clip_t")
        row["baseline_vbench_q"] = baseline_row.get("vbench_q")
        row["baseline_speedup_end_to_end"] = baseline_row.get("speedup_end_to_end_vs_dense")
        for key in ("clip_t", "vbench_q"):
            if row.get(key) is not None and baseline_row.get(key) is not None:
                row[f"{key}_delta"] = float(row[key]) - float(baseline_row[key])
        dense_end_to_end = (reference.get("dense_baseline", {}).get(pair_id) or {}).get("end_to_end_seconds")
        if dense_end_to_end:
            for key, out in (
                ("end_to_end_seconds", "speedup_reuse_vs_dense"),
                ("variant_end_to_end_seconds", "speedup_variant_vs_dense"),
            ):
                if row.get(key):
                    row[out] = float(dense_end_to_end) / float(row[key])

    status = "error" if any(row.get("status") == "error" for row in rows) else "completed"
    write_json(root / f"summary.{method}.json", {"status": status, "rows": rows})
    (root / f"report.{method}.md").write_text(build_report(root, rows, args, gpu_of), encoding="utf-8")
    fieldnames = [
        "pair_id", "method", "status", "actual_gpu", "source_stage_seconds", "denoising_seconds",
        "end_to_end_seconds", "target_prefix_end_to_end_seconds", "variant_end_to_end_seconds",
        "target_stage_wall_seconds", "pair_wall_seconds", "speedup_reuse_vs_dense",
        "speedup_variant_vs_dense", "baseline_speedup_end_to_end", "clip_t", "baseline_clip_t",
        "clip_t_delta", "vbench_q", "baseline_vbench_q", "vbench_q_delta", "resume_latent_path",
        "output_path", "error", "log_path", "quality_error", "quality_log_path",
    ]
    with (root / f"summary.{method}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({
        "root": str(root),
        "status": status,
        "summary": str(root / f"summary.{method}.json"),
        "report": str(root / f"report.{method}.md"),
    }, ensure_ascii=False), flush=True)
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
