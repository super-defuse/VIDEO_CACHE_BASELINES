#!/usr/bin/env python3
"""Wan2.1 Carnator step-5 Early Signature dataset for offline hit-rate sweeps.

The formal Wan2.1 run (``run_wan21_all_formal.sh``) needs ~10 days for 300
requests because every hit re-runs the whole Carnator source: a dense 50-step
trajectory, the fusion-mask template, the outer routing masks from completed
step 11, and the source video.  None of that is needed to *fit the four Early
Signature gates* -- only the step-5 signature is.

This script therefore runs, per arrival, exactly two five-step dense prefixes:

* the **source** prompt the legacy replay selected for that arrival (seeded with
  the fixed experiment seed, which is also what the formal source branch uses);
* the arrival's **target** prompt (the formal target probe is seeded with the
  fixed experiment seed as well, not the arrival seed);

and writes both Early Signatures.  Everything else is skipped: no trajectory,
no fusion masks, no outer block masks, no 40-step target reuse, no video.  Both
probes go through ``carnator.adapters.wan21_native.run_wan21_target_probe``,
which performs the *same* step-5 capture the formal source branch does (probe
enabled on completed step 5, post-scheduler latent, identical fused-Mbase
inputs, identical compatibility metadata), so a signature written here is the
signature the full run would have written.

Consequence to keep in mind when reading the output: this dataset cannot report
a real cache hit.  It has no reused latent and no video, so ``cache_ready`` is
never true and the legacy replay file stays the only source of hit/miss truth.
What it does give is the four gate distances for each arrival, which is enough
to recompute the hit rate for *any* hyperparameter tuple offline:

    run ... --summarize

``arrivals/<arrival>_<uuid>/pair.json`` holds the raw distances; the summarize
pass turns them into ``quality/step5_pair_distances.{jsonl,csv}`` and the
threshold sweep ``quality/step5_hit_rate_sweep.csv``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# The formal experiment module owns the input reader, the replay reader and the
# resolved model config; reusing them is what keeps this dataset's arrival
# ordering, source selection and generation geometry identical to the full run.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_vidprom_300_experiment as formal  # noqa: E402

from adapters import make_adapter  # noqa: E402
from core import Request  # noqa: E402

METHOD = "carnator_boundary_b5"
DEFAULT_SEED = 20260218
DEFAULT_GPUS = "3,4,5,6,7"
DEFAULT_REPLAY_ROOT = (
    ROOT / "artifacts" / "vidprom_1000cache_300input_noun_match" / "cogvideox15_replay"
)
DEFAULT_RUN_ROOT = (
    ROOT / "artifacts" / "vidprom_1000cache_300input_noun_match" / "wan21_carnator_step5"
)
DEFAULT_INPUT = Path("/home/tangxuebin/datasets/1000cache_300input_noun_match/cluster_02_2170_random300_input.csv")

# The probe step is the whole point of the dataset; both the parser default and
# the formal run's contract pin it at 5.  A config drift silently changing the
# capture step would invalidate every signature, so it is checked at startup.
EXPECTED_PROBE_STEP = 5
EXPECTED_TOTAL_STEPS = 50

# ``compare_early_signatures`` distance -> the gate hyperparameter it is compared
# against.  The three non-context distances only exist for a one-noun
# substitution; for a same-noun pair they are undefined and are carried as 0.0
# so a gate on them can never reject the pair (the convention the Wan2.2 forced
# pair study uses).
GATE_BY_DISTANCE = {
    "context_color_ciede2000": "max_context_color_distance",
    "log_area_ratio": "max_log_area_ratio",
    "shape_one_minus_dice": "max_shape_distance",
    "changed_color_ciede2000": "max_changed_color_distance",
}
DISTANCE_FIELDS = tuple(GATE_BY_DISTANCE)
GATE_FIELDS = tuple(GATE_BY_DISTANCE[field] for field in DISTANCE_FIELDS)
INFINITE_GATES = {gate: float("inf") for gate in GATE_FIELDS}
# A pair the sweep can score: the noun sets match, or exactly one noun was
# substituted.  Anything else has no defined gate tuple to compare.
COMPARABLE = ("same_nouns", "one_noun_substitution")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()


def parse_gpus(value: str) -> list[int]:
    gpus = [int(item) for item in str(value).replace(",", " ").split() if item.strip()]
    if not gpus:
        raise ValueError("at least one GPU index is required")
    return gpus


def arrival_dir(run_root: Path, arrival: dict[str, Any]) -> Path:
    return run_root / "arrivals" / f"{int(arrival['arrival_index']):04d}_{arrival['uuid']}"


def stage_dir(run_root: Path, arrival: dict[str, Any], stage: str) -> Path:
    return arrival_dir(run_root, arrival) / stage


def stage_complete(directory: Path) -> bool:
    status = directory / "status.json"
    if not status.is_file():
        return False
    try:
        return read_json(status).get("status") == "completed"
    except (OSError, json.JSONDecodeError):
        return False


def carnator_extra_flags(config: dict[str, Any]) -> list[str]:
    """The constants the formal Carnator target passes explicitly.

    They do not change the step-5 probe numerically (the probe runs dense fp
    before any quantized FFN or sparse hook is installed), but passing them
    keeps the probe's args -- and therefore its recorded ``mask_policy`` --
    identical to the formal run's.
    """

    section = config.get("carnator_b5", {})
    return [
        "--ffn-backend", str(section.get("ffn_backend", "w8a8")),
        "--outer-attn-backend", str(section.get("outer_backend", "sparge-sage2")),
        "--latent-injection-mode", str(section.get("latent_injection_mode", "boundary-blend")),
        "--boundary-alpha-function", str(section.get("boundary_alpha_function", "smoothstep")),
    ]


def build_probe_context(config: dict[str, Any], seed: int):
    """Load the resident Wan2.1 pipeline and the shared probe args once."""

    from carnator.adapters import wan21_native

    adapter = make_adapter("wan21", config)
    adapter.prepare()
    runtime = adapter._wan21_runtime()
    args = wan21_native.build_wan21_args(config, seed=seed, extra_flags=carnator_extra_flags(config))
    if int(args.mask_probe_step) != EXPECTED_PROBE_STEP:
        raise RuntimeError(
            f"Carnator probe step changed to {args.mask_probe_step}; the dataset contract is "
            f"{EXPECTED_PROBE_STEP}"
        )
    if int(args.steps) != EXPECTED_TOTAL_STEPS:
        raise RuntimeError(
            f"Carnator total steps changed to {args.steps}; the probes must run the "
            f"{EXPECTED_TOTAL_STEPS}-step schedule so the step-{EXPECTED_PROBE_STEP} timestep matches "
            "the formal run"
        )
    return runtime, args


def run_step5_probe(runtime, args, *, prompt: str, nouns, seed: int, output_dir: Path) -> dict[str, Any]:
    """One five-step dense prefix, writing the Early Signature into ``output_dir``.

    ``run_wan21_target_probe`` is the generic capture: the formal target calls
    it as-is and the formal source inlines the identical code.  Using it for
    both sides here is what makes the source and target signatures
    compatibility-equal (same model, scheduler config, CFG, probe seed,
    completed step, latent geometry, noun extractor and mask policy).
    """

    from carnator.adapters import wan21_native

    started = time.perf_counter()
    result = wan21_native.run_wan21_target_probe(
        runtime, args, prompt=prompt, nouns=nouns, output_dir=output_dir, probe_seed=seed,
    )
    result["wall_seconds"] = time.perf_counter() - started
    return result


def compare_pair(source_dir: Path, target_dir: Path) -> dict[str, Any]:
    """Four gate distances for one (target, source) signature pair.

    ``check_cache_artifacts`` is off because this dataset deliberately has none:
    the signatures are real, the caches they would describe are not.  Only
    ``metadata_compatible`` and the distances are meaningful here.
    """

    from carnator.early_signature import compare_early_signatures

    source_path = source_dir / "online_color_signature.json"
    target_path = target_dir / "online_color_signature.json"
    payload: dict[str, Any] = {
        "metadata_compatible": False,
        "comparability": None,
        "distances": {field: None for field in DISTANCE_FIELDS},
        "gate_distances": {field: None for field in DISTANCE_FIELDS},
        "applicable_gates": [],
        "reason": None,
        "source_only_nouns": [],
        "target_only_nouns": [],
    }
    if not source_path.is_file() or not target_path.is_file():
        payload["reason"] = "missing_signature"
        return payload
    try:
        source_objects = {str(item["noun"]) for item in read_json(source_path).get("objects", [])}
        target_objects = {str(item["noun"]) for item in read_json(target_path).get("objects", [])}
        source_only = sorted(source_objects - target_objects)
        target_only = sorted(target_objects - source_objects)
        payload["source_only_nouns"] = source_only
        payload["target_only_nouns"] = target_only

        kwargs: dict[str, Any] = {**INFINITE_GATES, "check_cache_artifacts": False}
        if source_only or target_only:
            if len(source_only) > 1 or len(target_only) > 1:
                payload["comparability"] = "noun_difference_gt_one"
                payload["reason"] = "noun_difference_gt_one"
                return payload
            payload["comparability"] = "one_noun_substitution"
            kwargs["query_changed_noun"] = target_only[0]
            kwargs["candidate_changed_noun"] = source_only[0]
        else:
            payload["comparability"] = "same_nouns"

        comparison = compare_early_signatures(
            {"signature_path": str(target_path)}, {"signature_path": str(source_path)}, **kwargs
        )
        payload["metadata_compatible"] = bool(comparison["metadata_compatible"])
        payload["reason"] = comparison.get("reason")
        if not comparison["metadata_compatible"]:
            return payload
        distances = comparison["distances"]
        payload["distances"] = {field: distances.get(field) for field in DISTANCE_FIELDS}
        payload["applicable_gates"] = ["context_color_ciede2000"] + (
            ["log_area_ratio", "shape_one_minus_dice", "changed_color_ciede2000"]
            if payload["comparability"] == "one_noun_substitution" else []
        )
        # Undefined distances are zero-filled only for the gate copy; the raw
        # ``distances`` above keeps the null so the record stays honest.
        payload["gate_distances"] = {
            field: float(distances.get(field) or 0.0) for field in DISTANCE_FIELDS
        }
        if any(not math.isfinite(payload["gate_distances"][field]) for field in payload["applicable_gates"]):
            # A non-finite distance cannot be compared against any threshold, so
            # the pair is excluded from the sweep rather than silently rejected
            # (the forced-pair study makes the same call).
            payload["comparability"] = "non_finite_distance"
            payload["reason"] = "non_finite_distance"
        return payload
    except Exception as exc:  # a broken pair must not abort the shard
        payload["reason"] = f"comparison_error:{type(exc).__name__}"
        return payload


def process_arrival(runtime, args, *, arrival: dict[str, Any], replay_row: dict[str, Any],
                    source_row: dict[str, Any], run_root: Path, seed: int, phase: str) -> dict[str, Any]:
    """Run the selected probes for one arrival and write its ``pair.json``."""

    from carnator.adapters import wan21_native

    base = arrival_dir(run_root, arrival)
    base.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "arrival_index": int(arrival["arrival_index"]),
        "request_index": int(arrival["request_index"]),
        "target_uuid": arrival["uuid"],
        "target_prompt": arrival["prompt"],
        "source_uuid": str(replay_row["source_uuid"]),
        "source_prompt": source_row["prompt"],
        "replay_decision": replay_row.get("decision"),
        "legacy_status": replay_row.get("legacy_status"),
        "probe_seed": int(seed),
        "probe_completed_step": int(args.mask_probe_step),
        "probe_total_steps": int(args.steps),
        "phase": phase,
        "status": "pending",
        "timing": {},
        "error": None,
    }
    if str(replay_row.get("decision")) != "hit":
        # A replay miss executes nothing in the formal run either.  The 300-row
        # Carnator replay is all hits, so this is a guard, not a path.
        record["status"] = "skipped_miss"
        write_json(base / "pair.json", record)
        return record

    try:
        source_request = Request(int(source_row.get("request_index", 0)), record["source_uuid"],
                                 source_row["prompt"], seed)
        target_request = Request(int(arrival["request_index"]), arrival["uuid"], arrival["prompt"], seed)
        record["source_nouns"] = list(wan21_native.extract_request_nouns(source_request))
        record["target_nouns"] = list(wan21_native.extract_request_nouns(target_request))

        for stage, prompt, nouns in (("source", source_row["prompt"], record["source_nouns"]),
                                     ("target", arrival["prompt"], record["target_nouns"])):
            if phase not in (stage, "all"):
                continue
            directory = stage_dir(run_root, arrival, stage)
            if stage_complete(directory):
                # A resumed arrival keeps the timings its earlier run recorded,
                # so the per-arrival cost report stays complete.
                previous = read_json(directory / "status.json")
                record["timing"][f"{stage}_probe_s"] = previous.get("wall_seconds")
                record["timing"][f"{stage}_dit_forward_s"] = previous.get("dit_forward_seconds")
                record["timing"][f"{stage}_resumed"] = True
                continue
            result = run_step5_probe(runtime, args, prompt=prompt, nouns=nouns, seed=seed,
                                     output_dir=directory)
            write_json(directory / "status.json", {
                "status": "completed", "stage": stage, "completed_at": utc_now(),
                "completed_steps": result["completed_steps"],
                "wall_seconds": result["wall_seconds"],
                "dit_forward_seconds": result["dit_forward_seconds"],
            })
            record["timing"][f"{stage}_probe_s"] = result["wall_seconds"]
            record["timing"][f"{stage}_dit_forward_s"] = result["dit_forward_seconds"]

        comparison = compare_pair(stage_dir(run_root, arrival, "source"),
                                 stage_dir(run_root, arrival, "target"))
        record.update(comparison)
        record["status"] = "completed" if comparison["comparability"] in COMPARABLE else "incomparable"
        record["timing"]["total_s"] = sum(
            float(record["timing"].get(key) or 0.0)
            for key in ("source_probe_s", "target_probe_s")
        )
    except Exception as exc:  # keep the shard alive; the arrival stays resumable
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"
    write_json(base / "pair.json", record)
    return record


def worker_main(arguments: argparse.Namespace) -> int:
    rows = formal.read_rows(arguments.input, arguments.limit)
    replay = formal.load_replay_decisions(arguments.replay_root, METHOD, limit=arguments.limit)
    view_rows = {row["uuid"]: row for row in formal.view_rows_for_preload(arguments.view)}
    config = formal.resolved_config(f"cuda:{arguments.gpu}", False, model_name="wan21")
    seed = int(config.get("seed", DEFAULT_SEED))

    shard = [row for row in rows if row["arrival_index"] % arguments.shard_count == arguments.shard_index]
    print(f"[gpu {arguments.gpu}] shard {arguments.shard_index}/{arguments.shard_count}: "
          f"{len(shard)} arrivals, seed {seed}", flush=True)

    runtime, native_args = build_probe_context(config, seed)
    print(f"[gpu {arguments.gpu}] models resident; probe step {native_args.mask_probe_step}/"
          f"{native_args.steps}", flush=True)

    signatures_path = arguments.run_root / "signatures.jsonl"
    done = 0
    for arrival in shard:
        index = int(arrival["arrival_index"])
        replay_row = replay.get(int(arrival["request_index"]))
        if replay_row is None:
            print(f"[gpu {arguments.gpu}] arrival {index}: no replay row; skipped", flush=True)
            continue
        if replay_row.get("uuid") != arrival["uuid"] or replay_row.get("prompt") != arrival["prompt"]:
            print(f"[gpu {arguments.gpu}] arrival {index}: replay row does not match the input row; "
                  "skipped", flush=True)
            continue
        source_row = view_rows.get(str(replay_row.get("source_uuid")))
        if source_row is None:
            print(f"[gpu {arguments.gpu}] arrival {index}: source "
                  f"{replay_row.get('source_uuid')} is absent from the source view; skipped", flush=True)
            continue
        if arguments.resume and (arrival_dir(arguments.run_root, arrival) / "pair.json").is_file():
            previous = read_json(arrival_dir(arguments.run_root, arrival) / "pair.json")
            if previous.get("status") in ("completed", "incomparable", "skipped_miss"):
                done += 1
                continue

        started = time.perf_counter()
        record = process_arrival(runtime, native_args, arrival=arrival, replay_row=replay_row,
                                 source_row=source_row, run_root=arguments.run_root, seed=seed,
                                 phase=arguments.phase)
        append_jsonl(signatures_path, record)
        done += 1
        print(f"[gpu {arguments.gpu}] {done}/{len(shard)} arrival {index} "
              f"status={record['status']} comparability={record.get('comparability')} "
              f"source={record['timing'].get('source_probe_s')}s "
              f"target={record['timing'].get('target_probe_s')}s "
              f"e2e={time.perf_counter() - started:.1f}s", flush=True)
    print(f"[gpu {arguments.gpu}] shard complete: {done} arrivals", flush=True)
    return 0


def _candidate_values(values: list[float], default: float, max_values: int) -> list[float]:
    """Observed distances plus the production default, thinned to ``max_values``."""

    candidates = sorted({float(value) for value in values} | {float(default)})
    if len(candidates) <= max_values:
        return candidates
    indices = {round(index * (len(candidates) - 1) / (max_values - 1)) for index in range(max_values)}
    return [candidates[index] for index in sorted(indices)]


def load_pair_records(run_root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted((run_root / "arrivals").glob("*/pair.json")):
        records.append(read_json(path))
    records.sort(key=lambda row: int(row["arrival_index"]))
    return records


def summarize(arguments: argparse.Namespace) -> int:
    """Turn the collected distances into per-pair tables and a threshold sweep."""

    run_root = arguments.run_root
    records = load_pair_records(run_root)
    if not records:
        raise FileNotFoundError(f"no arrival records under {run_root / 'arrivals'}")
    quality_root = run_root / "quality"
    quality_root.mkdir(parents=True, exist_ok=True)
    arrivals = len(formal.read_rows(arguments.input, arguments.limit))
    completed = [row for row in records if row["status"] in ("completed", "incomparable")]
    comparable = [row for row in completed if row.get("comparability") in COMPARABLE]

    fields = [
        "arrival_index", "request_index", "target_uuid", "source_uuid", "status", "comparability",
        "replay_decision", *DISTANCE_FIELDS, "applicable_gates", "source_only_nouns", "target_only_nouns",
    ]
    with (quality_root / "step5_pair_distances.csv.tmp").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            distances = row.get("gate_distances") or {}
            writer.writerow({
                "arrival_index": row["arrival_index"], "request_index": row["request_index"],
                "target_uuid": row["target_uuid"], "source_uuid": row["source_uuid"],
                "status": row["status"], "comparability": row.get("comparability"),
                "replay_decision": row.get("replay_decision"),
                **{field: distances.get(field) for field in DISTANCE_FIELDS},
                "applicable_gates": "|".join(row.get("applicable_gates") or []),
                "source_only_nouns": "|".join(row.get("source_only_nouns") or []),
                "target_only_nouns": "|".join(row.get("target_only_nouns") or []),
            })
    (quality_root / "step5_pair_distances.csv.tmp").replace(quality_root / "step5_pair_distances.csv")
    # The raw records stay as JSON too: the CSV flattens the noun sets and the
    # applicability list, and any later re-scoring wants those intact.
    (quality_root / "step5_pair_distances.jsonl.tmp").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
    )
    (quality_root / "step5_pair_distances.jsonl.tmp").replace(quality_root / "step5_pair_distances.jsonl")

    sweep: list[dict[str, Any]] = []
    if comparable and arguments.max_threshold_values >= 2:
        candidates = {
            field: _candidate_values([float(row["gate_distances"][field]) for row in comparable],
                                     arguments.default_thresholds[GATE_BY_DISTANCE[field]],
                                     arguments.max_threshold_values)
            for field in DISTANCE_FIELDS
        }
        for values in itertools.product(*(candidates[field] for field in DISTANCE_FIELDS)):
            thresholds = dict(zip(GATE_FIELDS, values))
            accepted = [
                row for row in comparable
                if all(float(row["gate_distances"][field]) <= thresholds[GATE_BY_DISTANCE[field]]
                       for field in row["applicable_gates"])
            ]
            n = len(accepted)
            sweep.append({
                **thresholds,
                "accepted": n,
                "rejected_by_gates": len(comparable) - n,
                "unscorable": len(records) - len(comparable),
                # The denominator the user cares about: every arrival of the
                # formal 300-request experiment.  A pair the gates reject is a
                # miss, and so is a pair the gates cannot evaluate at all.
                "hit_rate_all_arrivals": n / arrivals,
                "hit_rate_comparable": n / len(comparable),
            })
    sweep.sort(key=lambda row: (-row["hit_rate_all_arrivals"],
                                tuple(row[gate] for gate in GATE_FIELDS)))
    with (quality_root / "step5_hit_rate_sweep.csv.tmp").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*GATE_FIELDS, "accepted", "rejected_by_gates",
                                                    "unscorable", "hit_rate_all_arrivals",
                                                    "hit_rate_comparable"])
        writer.writeheader()
        for row in sweep:
            writer.writerow(row)
    (quality_root / "step5_hit_rate_sweep.csv.tmp").replace(quality_root / "step5_hit_rate_sweep.csv")

    default_row = next(
        (row for row in sweep
         if all(abs(row[gate] - arguments.default_thresholds[gate]) < 1e-12 for gate in GATE_FIELDS)),
        None,
    )
    manifest = {
        "schema_version": "wan21-carnator-step5-signatures/v1",
        "summarized_at": utc_now(),
        "arrivals": arrivals,
        "records": len(records),
        "completed": len(completed),
        "comparable": len(comparable),
        "unscorable": len(records) - len(comparable),
        "errors": len([row for row in records if row["status"] == "error"]),
        "sweep_tuples": len(sweep),
        "max_threshold_values": arguments.max_threshold_values,
        "default_thresholds": arguments.default_thresholds,
        "default_thresholds_in_sweep": default_row,
        "best_hit_rate": sweep[0] if sweep else None,
    }
    write_json(quality_root / "step5_summary.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


def run_manifest(arguments: argparse.Namespace, gpus: list[int]) -> None:
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"
    replay_path = arguments.replay_root / f"{METHOD}.jsonl"
    write_json(arguments.run_root / "run_manifest.json", {
        "schema_version": "wan21-carnator-step5-signatures/v1",
        "created_at": utc_now(),
        "model": "Wan2.1-T2V-1.3B",
        "method": METHOD,
        "purpose": "collect step-5 Early Signatures only, so the four gate thresholds can be "
                   "swept offline without the 50-step source or the 40-step reuse",
        "input": str(arguments.input),
        "input_sha256": sha256_file(arguments.input),
        "source_view": str(arguments.view),
        "replay_root": str(arguments.replay_root),
        "replay_sha256": sha256_file(replay_path) if replay_path.is_file() else None,
        "replay_policy": "the legacy decision file still owns hit/miss and the source UUID; every "
                         "arrival in it is a hit, so this dataset reports gate acceptance only",
        "probe_contract": {
            "completed_step": EXPECTED_PROBE_STEP,
            "total_steps": EXPECTED_TOTAL_STEPS,
            "seed": "the fixed experiment seed for both sides, matching the formal source and the "
                    "formal target probe",
            "capture": "carnator.adapters.wan21_native.run_wan21_target_probe",
            "skipped": ["50-step source trajectory", "fusion mask template",
                        "outer routing masks", "40-step target reuse", "videos"],
        },
        "git_revision": revision,
        "limit": arguments.limit,
        "gpus": gpus,
        "phase": arguments.phase,
        "resume": arguments.resume,
    })


def controller_main(arguments: argparse.Namespace) -> int:
    gpus = parse_gpus(arguments.gpus)
    arguments.run_root.mkdir(parents=True, exist_ok=True)
    run_manifest(arguments, gpus)
    logs = arguments.run_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    def gpu_worker(position: int, gpu: int) -> int:
        command = [
            sys.executable, str(Path(__file__).resolve()), "--worker",
            "--gpu", str(gpu), "--shard-index", str(position), "--shard-count", str(len(gpus)),
            "--input", str(arguments.input), "--view", str(arguments.view),
            "--replay-root", str(arguments.replay_root), "--run-root", str(arguments.run_root),
            "--limit", str(arguments.limit), "--phase", arguments.phase,
        ]
        # The worker defaults to resuming; only the opt-out needs forwarding, and
        # the parser spells it --no-resume.
        if not arguments.resume:
            command.append("--no-resume")
        environment = os.environ.copy()
        environment["CARNATOR_ENV_READY"] = "1"
        log_path = logs / f"gpu{gpu}.log"
        print(f"[controller] gpu {gpu} shard {position}/{len(gpus)} -> {log_path}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            return subprocess.run(command, cwd=ROOT, env=environment, stdout=log,
                                  stderr=subprocess.STDOUT, check=False).returncode

    failures: list[tuple[int, int]] = []
    lock = threading.Lock()

    def run(gpu: int) -> None:
        code = gpu_worker(gpus.index(gpu), gpu)
        if code:
            with lock:
                failures.append((gpu, code))

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        for future in [pool.submit(run, gpu) for gpu in gpus]:
            future.result()
    if failures:
        raise RuntimeError("worker failures: " + ", ".join(f"gpu {gpu} (exit {code})"
                                                           for gpu, code in failures))
    return summarize(arguments) if arguments.summarize else 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    value.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    value.add_argument("--view", type=Path, default=formal.DEFAULT_VIEW)
    value.add_argument("--replay-root", type=Path, default=DEFAULT_REPLAY_ROOT)
    value.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    value.add_argument("--gpus", default=DEFAULT_GPUS)
    value.add_argument("--limit", type=int, default=300)
    value.add_argument("--phase", choices=("source", "target", "all"), default="all")
    value.add_argument("--no-resume", dest="resume", action="store_false",
                       help="re-run arrivals that already reached a terminal status")
    value.add_argument("--summarize", action="store_true",
                       help="after the workers finish, write the distance tables and threshold sweep")
    value.add_argument("--summarize-only", action="store_true",
                       help="skip generation and only re-run the offline summary over run-root")
    value.add_argument("--max-threshold-values", type=int, default=12,
                       help="candidate values per gate dimension in the sweep")
    value.add_argument("--default-thresholds", type=Path, dest="default_thresholds_path",
                       help="JSON gate tuple to always include in the sweep; defaults to the "
                            "experiment's production thresholds")
    value.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    value.add_argument("--gpu", type=int, help=argparse.SUPPRESS)
    value.add_argument("--shard-index", type=int, help=argparse.SUPPRESS)
    value.add_argument("--shard-count", type=int, help=argparse.SUPPRESS)
    value.set_defaults(resume=True)
    return value


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    thresholds = formal.EARLY_SIGNATURE_THRESHOLDS
    if arguments.default_thresholds_path is not None:
        thresholds = {**thresholds, **read_json(arguments.default_thresholds_path)}
    arguments.default_thresholds = {gate: float(thresholds[gate]) for gate in GATE_FIELDS}
    if arguments.summarize_only:
        return summarize(arguments)
    if arguments.worker:
        if arguments.gpu is None or arguments.shard_index is None or arguments.shard_count is None:
            raise SystemExit("--worker requires --gpu, --shard-index and --shard-count")
        return worker_main(arguments)
    return controller_main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
