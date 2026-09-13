#!/usr/bin/env python3
"""Run one VidProM 300-request baseline against the immutable source view.

The script is intentionally one-method/one-GPU.  It can therefore be started
five times in parallel without sharing model state or writing to the canonical
source cache.  ``--limit 1`` is the supported smoke-test mode.
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
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters import make_adapter
from core import CacheMatch, Request, Runner
from core.cogvideo_runner import CogVideoReplayRunner
from core.wan21_runner import Wan21ReplayRunner
from core.cache import CacheEntry
from core.registry import ensure_supported, make_method
from core.retrieval import VectorRecord

METHODS = ("baseline", "nirvana_vid", "teacache", "chorus", "chorus2", "carnator_boundary_b5")
GPU_BY_METHOD = {"baseline": 3, "nirvana_vid": 3, "teacache": 4, "chorus": 5, "chorus2": 6, "carnator_boundary_b5": 7}
# Wan2.1 runs one method per GPU; nirvana_vid takes teacache's idle slot.
WAN21_GPU_BY_METHOD = {"baseline": 3, "nirvana_vid": 4, "chorus": 5, "chorus2": 6, "carnator_boundary_b5": 7}
# LTX-Video has no TeaCache entry in the matrix, so it uses the Wan2.1 assignment.
LTX_GPU_BY_METHOD = {"baseline": 3, "nirvana_vid": 4, "chorus": 5, "chorus2": 6, "carnator_boundary_b5": 7}
# CogVideoX1.5 has no TeaCache baseline either; one method per GPU, same slots.
COGVIDEOX_GPU_BY_METHOD = {"baseline": 3, "nirvana_vid": 4, "chorus": 5, "chorus2": 6, "carnator_boundary_b5": 7}
GPU_COUNT = 8
EARLY_SIGNATURE_THRESHOLDS = {
    # Representative noun-match calibration tuple: 270/300 pairs pass;
    # 1,092 qualified tuples remain in forced_scan_qualified.json.
    "max_context_color_distance": 2.5460641772840114,
    "max_log_area_ratio": 1.6755377126043391,
    "max_shape_distance": 1.0,
    "max_changed_color_distance": 2.2003341853972085,
}
DISABLED_EARLY_SIGNATURE_THRESHOLDS = {
    "max_context_color_distance": 1_000_000.0,
    "max_log_area_ratio": 1_000_000.0,
    "max_shape_distance": 1_000_000.0,
    "max_changed_color_distance": 1_000_000.0,
}
FUSED_MBASE_POLICY_PREFIX = "native-self-cross-fusion-mbase|"
DEFAULT_VIEW = ROOT / "artifacts" / "vidprom_1000cache_300input_noun_match" / "source_view"
DEFAULT_INPUT = Path("/home/tangxuebin/datasets/1000cache_300input_noun_match/cluster_02_2170_random300_input.csv")
DEFAULT_OUTPUT = ROOT / "artifacts" / "vidprom_1000cache_300input_noun_match" / "results"
DENSE_C0_VERSION = "wan22-dense-c0-v1"
WARMUP_SOURCE_PROMPT = "a dog is sleeping on a grassland, focus on the dog"
WARMUP_TARGET_PROMPT = "a cat is sleeping on a grassland, focus on the cat"
SMOKE_FORCED_REQUEST_INDICES = (0, 1)
SMOKE_FORCED_HIT_SOURCE_UUID = "4f8357f4-3305-51d5-95d3-30fd8067ec44"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return payload


def resolved_config(
    device: str,
    dry_run: bool,
    *,
    model_name: str = "wan22",
    early_signature_thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    model = {"model": "synthetic", "dry_run": True, "seed": 20260218} if dry_run else load_yaml(ROOT / "configs" / "models" / f"{model_name}.yaml")
    methods = {} if dry_run else load_yaml(ROOT / "configs" / "methods" / "default.yaml")
    config = {**model, **methods, "dry_run": dry_run, "device": device, "cache_root": "cache"}
    # ``t5_weights`` is LTX-Video's; the other two are Wan's.
    for key in ("model_path", "wan_source", "t5_weights"):
        value = config.get(key)
        if value and not Path(value).is_absolute():
            config[key] = str((ROOT / value).resolve())
    thresholds = dict(early_signature_thresholds or EARLY_SIGNATURE_THRESHOLDS)
    for section in ("nirvana_vid", "chai", "chorus", "carnator"):
        values = config.get("retrieval", {}).get(section, {})
        if section == "carnator":
            values.update(thresholds)
        checkpoint = values.get("longclip_checkpoint")
        if checkpoint and not Path(checkpoint).is_absolute():
            values["longclip_checkpoint"] = str((ROOT / checkpoint).resolve())
        if values:
            values["device"] = device
    return config


def read_rows(path: Path, limit: int, *, arrival_start: int = 0, arrival_stop: int | None = None,
              request_indices: list[int] | None = None) -> list[dict[str, Any]]:
    with path.expanduser().open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    required = {"uuid", "prompt"}
    if not rows or required - set(rows[0]):
        raise ValueError(f"input CSV must contain {sorted(required)}: {path}")
    selected = rows[:limit]
    indexed = list(enumerate(selected))
    if request_indices is not None:
        indexed = [(index, selected[index]) for index in request_indices]
    selected = [row for _, row in indexed]
    stop = len(selected) if arrival_stop is None else arrival_stop
    if not 0 <= arrival_start < stop <= len(selected):
        raise ValueError(f"arrival range must satisfy 0 <= start < stop <= {len(selected)}")
    seen: set[str] = set()
    result = []
    for arrival_index, row in enumerate(selected):
        if not arrival_start <= arrival_index < stop:
            continue
        uuid = str(row["uuid"]).strip()
        prompt = str(row["prompt"]).strip()
        if not uuid or Path(uuid).name != uuid or uuid in seen or not prompt:
            raise ValueError(f"invalid or duplicate request row at arrival {arrival_index}: {uuid!r}")
        seen.add(uuid)
        original_index = request_indices[arrival_index] if request_indices is not None else arrival_index
        result.append({"arrival_index": arrival_index, "request_index": original_index, "uuid": uuid, "prompt": prompt})
    if not result:
        raise ValueError("input CSV is empty")
    return result


def forced_smoke_decision(args: argparse.Namespace, request_index: int) -> str | None:
    """Return the explicitly requested cross-request outcome for the two-row smoke."""
    if not args.smoke_force_outcomes:
        return None
    if request_index == SMOKE_FORCED_REQUEST_INDICES[0]:
        return "miss"
    if request_index == SMOKE_FORCED_REQUEST_INDICES[1]:
        return "hit"
    raise ValueError(f"forced smoke has no decision for request_index={request_index}")


def forced_smoke_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "enabled": bool(args.smoke_force_outcomes),
        "request_0": "miss" if args.smoke_force_outcomes else None,
        "request_1": "hit" if args.smoke_force_outcomes else None,
        "hit_source_uuid": SMOKE_FORCED_HIT_SOURCE_UUID if args.smoke_force_outcomes else None,
        "tea_cache": "not_applicable_target_only",
    }


def validate_forced_smoke(args: argparse.Namespace, rows: list[dict[str, Any]], view_rows: list[dict[str, Any]]) -> None:
    if not args.smoke_force_outcomes:
        return
    request_indices = tuple(row["request_index"] for row in rows)
    if request_indices != SMOKE_FORCED_REQUEST_INDICES:
        raise ValueError(
            "--smoke-force-outcomes requires exactly --request-indices 0,1 "
            "with both arrivals in this worker"
        )
    if SMOKE_FORCED_HIT_SOURCE_UUID not in {row["uuid"] for row in view_rows}:
        raise ValueError(
            "--smoke-force-outcomes requires its fixed hit source UUID in the supplied cache view: "
            f"{SMOKE_FORCED_HIT_SOURCE_UUID}"
        )


def cache_match_audit(match: CacheMatch) -> dict[str, Any]:
    return {
        "hit": match.hit,
        "source_uuid": match.source_uuid,
        "source_request_index": match.source_request_index,
        "kind": match.kind,
        "score": match.score,
    }


def force_cache_match(
    observed: CacheMatch,
    *,
    outcome: str,
    request_index: int,
    method_name: str,
    source_by_uuid: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> CacheMatch:
    """Override one smoke decision while retaining the real lookup as audit metadata."""
    if outcome not in {"hit", "miss"}:
        raise ValueError(f"unsupported forced outcome: {outcome}")
    metadata: dict[str, Any] = {
        "retriever": "forced_smoke",
        "forced_smoke": {
            "outcome": outcome,
            "request_index": request_index,
            "observed_match": cache_match_audit(observed),
        },
    }
    if outcome == "miss":
        return CacheMatch(kind="forced_smoke_miss", metadata=metadata)

    source = source_by_uuid.get(SMOKE_FORCED_HIT_SOURCE_UUID)
    if source is None:
        raise ValueError(f"forced smoke source is unavailable: {SMOKE_FORCED_HIT_SOURCE_UUID}")
    metadata["forced_smoke"]["source_uuid"] = SMOKE_FORCED_HIT_SOURCE_UUID
    if method_name == "nirvana_vid":
        metadata["reuse_step"] = int(config.get("retrieval", {}).get("nirvana_vid", {}).get("reuse_step", 5))
    elif method_name == "chorus":
        stages = config.get("chorus", {}).get("stages", {})
        metadata["stage_boundaries"] = {"k1": int(stages.get("k1", 5)), "k2": int(stages.get("k2", 30))}
    return CacheMatch(
        True,
        SMOKE_FORCED_HIT_SOURCE_UUID,
        int(source.get("request_index", 0)),
        "forced_smoke_hit",
        metadata=metadata,
    )


def attach_forced_carnator_mask_selection(
    match: CacheMatch, *, records: list[VectorRecord], target_nouns: tuple[str, ...]
) -> CacheMatch:
    """Attach selection for the smoke's fixed source rather than its natural hit."""

    if not match.hit or match.source_uuid is None:
        return match
    source = next((record for record in records if record.uuid == match.source_uuid), None)
    if source is None or not isinstance(source.metadata, dict):
        raise ValueError(f"forced Carnator source is unavailable: {match.source_uuid}")
    from carnator.early_signature import select_per_noun_inference_mbase

    selection, _mask = select_per_noun_inference_mbase(
        source.metadata.get("signature", {}), target_nouns=target_nouns
    )
    metadata = dict(match.metadata or {})
    metadata["inference_mask_selection"] = selection
    return CacheMatch(
        match.hit,
        match.source_uuid,
        match.source_request_index,
        match.kind,
        match.score,
        metadata,
    )


def forced_carnator_probe_match(observed: CacheMatch, *, request_index: int, outcome: str) -> CacheMatch:
    """Keep the target-side probe in the smoke before overriding its final decision."""
    return CacheMatch(
        kind="probe_required",
        metadata={
            "retriever": "forced_smoke",
            "probe_required": True,
            "forced_smoke": {
                "outcome": outcome,
                "request_index": request_index,
                "observed_match": cache_match_audit(observed),
            },
        },
    )


def install_forced_generic_decisions(
    method: Any,
    args: argparse.Namespace,
    *,
    source_by_uuid: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> None:
    """Apply the smoke override after the normal lookup has completed."""
    if not args.smoke_force_outcomes or not method.capabilities.cross_request_cache:
        return
    find_cache = method.find_cache

    def forced_find_cache(request: Request) -> CacheMatch:
        observed = find_cache(request)
        outcome = forced_smoke_decision(args, request.request_index)
        assert outcome is not None
        return force_cache_match(
            observed,
            outcome=outcome,
            request_index=request.request_index,
            method_name=method.name,
            source_by_uuid=source_by_uuid,
            config=config,
        )

    method.find_cache = forced_find_cache


def load_view(view: Path) -> list[dict[str, Any]]:
    manifest = view / "view_manifest.json"
    index = view / "index.jsonl"
    if not manifest.is_file() or not index.is_file():
        raise FileNotFoundError(f"cache view is incomplete: {view}")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    selected = int(payload.get("selected_requests", -1))
    if selected <= 0:
        raise ValueError(f"source view has invalid selected_requests={selected}")
    rows = [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != selected or len({row.get("uuid") for row in rows}) != selected:
        raise ValueError(f"cache view index must contain {selected} unique UUIDs")
    for row in rows:
        entry = view / "entries" / str(row["uuid"])
        if not entry.is_dir() or not (entry / "manifest.json").is_file() or not (entry / "source.mp4").is_file():
            raise FileNotFoundError(f"cache view entry is incomplete: {entry}")
    return rows


def has_fused_mbase_signature(signature_path: Path) -> bool:
    try:
        payload = json.loads(signature_path.read_text(encoding="utf-8"))
        policy = payload["compatibility"]["mask_policy"]
    except (KeyError, OSError, TypeError, json.JSONDecodeError):
        return False
    return isinstance(policy, str) and policy.startswith(FUSED_MBASE_POLICY_PREFIX)


def has_per_noun_inference_signature(signature_path: Path) -> bool:
    from carnator.early_signature import per_noun_mbase_is_valid

    return has_fused_mbase_signature(signature_path) and per_noun_mbase_is_valid(
        {"signature_path": str(signature_path)}
    )


def validate_carnator_signature_view(view: Path, rows: list[dict[str, Any]]) -> None:
    invalid: list[str] = []
    for row in rows:
        cache_dir = view / "entries" / str(row["uuid"]) / "carnator" / "cache" / "source_dense"
        signature = cache_dir / "online_color_signature.json"
        if not has_per_noun_inference_signature(signature) or not (cache_dir / "fusion_masks" / "masks.pt").is_file():
            invalid.append(str(row["uuid"]))
            if len(invalid) == 5:
                break
    if invalid:
        raise ValueError(f"Carnator requires v4 per-noun Early Signature artifacts; invalid source UUIDs: {invalid}")


def check_only(args: argparse.Namespace) -> dict[str, Any]:
    if args.gpu not in range(3, 8) and not args.allow_gpu_remap:
        raise ValueError("--gpu must be one of 3,4,5,6,7")
    view_rows = load_view(args.cache_view.expanduser().resolve())
    if args.smoke_force_outcomes:
        rows = read_rows(
            args.input,
            args.limit,
            arrival_start=args.arrival_start,
            arrival_stop=args.arrival_stop,
            request_indices=args.request_indices,
        )
        validate_forced_smoke(args, rows, view_rows)
    if (args.method == "carnator_boundary_b5" and getattr(args, "model", "wan22") not in {"cogvideox15", "wan21"}):
        validate_carnator_signature_view(args.cache_view.expanduser().resolve(), view_rows)
    config = resolved_config(
        f"cuda:{args.gpu}",
        args.dry_run,
        model_name=getattr(args, "model", "wan22"),
        early_signature_thresholds=getattr(args, "early_signature_thresholds", None),
    )
    if not args.dry_run:
        model_path = Path(config["model_path"])
        # Carnator's native TI2V runner is intentionally separate from the
        # generic adapter registry; Chorus2 has the same native split.
        if getattr(args, "model", "wan22") == "cogvideox15":
            # Same rule as Wan2.1/LTX-Video: four methods replay the immutable
            # decisions, the baseline runs every arrival as a dense generation.
            if args.method == "teacache":
                raise ValueError("teacache is not part of the CogVideoX1.5 baseline set")
            if args.method != "baseline":
                if args.replay_root is None:
                    raise ValueError("CogVideoX1.5 replay methods require --replay-root built by "
                                     "build_cogvideo_hit_replay.py")
                load_replay_decisions(args.replay_root, args.method, limit=args.limit)
            required = {
                "model_index.json", "scheduler/scheduler_config.json", "tokenizer/tokenizer_config.json",
                "text_encoder/config.json", "transformer/config.json", "vae/config.json",
            }
            missing = sorted(str(model_path / relative) for relative in required if not (model_path / relative).is_file())
            if missing:
                raise FileNotFoundError(f"CogVideoX1.5 checkpoint is incomplete: {missing}")
        elif getattr(args, "model", "wan22") == "wan21":
            if args.method == "teacache":
                raise ValueError("teacache is not part of the Wan2.1 baseline set")
            if args.method != "baseline":
                if args.replay_root is None:
                    raise ValueError("Wan2.1 replay methods require --replay-root from the legacy experiment")
                load_replay_decisions(args.replay_root, args.method, limit=args.limit)
        elif args.method != "carnator_boundary_b5":
            ensure_supported("wan22", "chorus" if args.method == "chorus2" else args.method)
        if not model_path.is_dir():
            raise FileNotFoundError(f"{getattr(args, 'model', 'wan22')} checkpoint is missing: {model_path}")
    return {
        "status": "ready",
        "method": args.method,
        "model": getattr(args, "model", "wan22"),
        "gpu": args.gpu,
        "source_entries": len(view_rows),
        "input": str(args.input.expanduser().resolve()),
        "input_sha256": sha256_file(args.input.expanduser().resolve()),
        "view_manifest_sha256": sha256_file((args.cache_view / "view_manifest.json").resolve()),
        "config_fingerprint": hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest(),
        "forced_smoke": forced_smoke_config(args),
        "replay_root": str(args.replay_root) if args.replay_root else None,
    }


def write_cogvideo_run_manifest(output: Path, *, args: argparse.Namespace, config: dict[str, Any],
                                replay_root: Path | None, replay_path: Path | None, view: Path) -> None:
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"
    model_path = Path(config["model_path"])
    model_index = model_path / "model_index.json"
    manifest = {
        "schema_version": "cogvideox15-experiment/v1",
        "model": "CogVideoX1.5",
        "model_path": str(model_path),
        "model_index_sha256": sha256_file(model_index) if model_index.is_file() else None,
        "legacy_source_view": str(view),
        "legacy_source_view_manifest_sha256": sha256_file(view / "view_manifest.json"),
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "replay_root": str(replay_root) if replay_root is not None else None,
        "replay_sha256": sha256_file(replay_path) if replay_path is not None else None,
        "git_revision": revision,
        "resolved_config": config,
        "timing": {
            "source_prepare_s": "source generation plus cache serialization; reported separately",
            "target_e2e_s": "request wall clock excluding source preparation, cache loading, "
                            "method preprocessing and output bookkeeping",
            "denoising_s": "CUDA-synchronized CogVideo transformer forward time",
        },
        "replay_policy": "legacy hit/miss and source UUID are immutable ground truth; baseline runs every request",
        "miss_policy": "skipped without generation (status cache_miss, latency 0)",
        "temporary_cache_policy": "tmp_cache/<method>/<arrival_index>_<source_uuid>, removed in finally",
        "request_error_policy": "record error without changing replay decision",
        # Carnator's target runs a dense prefix before reuse and counts it in the
        # request's dit_forward_s, so the manifest has to say so or the denoising
        # column looks like a regression against a run that skipped the prefix.
        "target_probe_enabled": args.method == "carnator_boundary_b5",
        "target_probe_completed_step": 5 if args.method == "carnator_boundary_b5" else None,
        "target_probe_seed": args.seed if args.method == "carnator_boundary_b5" else None,
        "args": {"method": args.method, "gpu": args.gpu, "limit": args.limit, "seed": args.seed},
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def cogvideo_run(args: argparse.Namespace, rows: list[dict[str, Any]], view: Path, output: Path) -> list[dict[str, Any]]:
    """CogVideoX1.5 replay run, structurally identical to the Wan2.1/LTX-Video ones.

    The same fixed decisions drive the same five baselines through the same
    replay runner; only the adapter, the method plugins' model branch and the
    manifest's model identity differ.
    """
    config = resolved_config(f"cuda:{args.gpu}", args.dry_run, model_name="cogvideox15",
                             early_signature_thresholds=getattr(args, "early_signature_thresholds", None))
    replay = None
    replay_path = None
    if args.method != "baseline":
        if args.replay_root is None:
            raise ValueError("CogVideoX1.5 replay methods require --replay-root")
        replay_path = args.replay_root / f"{args.method}.jsonl"
        replay = load_replay_decisions(args.replay_root, args.method, limit=args.limit)
    source_by_uuid = {row["uuid"]: row for row in view_rows_for_preload(view)}
    if args.smoke_force_outcomes:
        if replay is None:
            raise ValueError("--smoke-force-outcomes requires a replay decision file")
        for row in rows:
            decision = forced_smoke_decision(args, row["request_index"])
            replay_row = dict(replay[row["request_index"]])
            replay_row.update({
                "decision": decision,
                "hit": decision == "hit",
                "source_uuid": SMOKE_FORCED_HIT_SOURCE_UUID if decision == "hit" else None,
                "source_request_index": (
                    int(source_by_uuid[SMOKE_FORCED_HIT_SOURCE_UUID].get("request_index", 0))
                    if decision == "hit" else None
                ),
                "forced_smoke": True,
            })
            replay[row["request_index"]] = replay_row
    if args.dry_run:
        config = {**config, **load_yaml(ROOT / "configs" / "models" / "cogvideox15.yaml"),
                  "model": "cogvideox15", "dry_run": True, "device": f"cuda:{args.gpu}"}
    adapter = make_adapter("synthetic" if args.dry_run else "cogvideox15", config)
    method = make_method(args.method, config)
    method.adapter = adapter
    if not args.dry_run and not args.resume:
        # The resident model plus Grounded-SAM-2 are the two loads that would
        # otherwise land inside the first timed request.
        warmup_paths = ["model_load"]
        if args.method == "chorus":
            from methods.chorus.grounded_sam import GroundedSam2Segmenter

            sam_config = {**config.get("chorus", {}).get("grounded_sam2", {}),
                          "device": str(config.get("device", "cuda"))}
            GroundedSam2Segmenter(sam_config)._load()
            warmup_paths.append("chorus_sam_model_loaded")
        write_warmup_record(output, paths=warmup_paths)
    runner = CogVideoReplayRunner(adapter, method, output_dir=output, replay=replay,
                                  source_prompts=source_by_uuid, config=config,
                                  # Without this the inherited default labels every
                                  # CogVideoX row "Wan2.1-T2V-1.3B", which is how the
                                  # old smoke_two_requests_* results came to be read as
                                  # CogVideoX evidence.
                                  model="CogVideoX1.5-5B")
    write_cogvideo_run_manifest(output, args=args, config=config, replay_root=args.replay_root,
                                replay_path=replay_path, view=view)
    # The runner streams its raw RequestResult rows here; the canonical
    # requests.jsonl / request_records.jsonl are written below from the
    # flattened payloads, exactly like the Wan2.1 and LTX-Video paths.
    raw_path = output / "requests.raw.jsonl"
    run_results = runner.run(
        (Request(row["request_index"], row["uuid"], row["prompt"], args.seed + row["request_index"])
         for row in rows),
        raw_path,
    )
    payloads = flatten_replay_payloads(args, rows, run_results, output)
    raw_path.unlink(missing_ok=True)
    return payloads


def load_replay_decisions(root: Path, method: str, *, limit: int) -> dict[int, dict[str, Any]]:
    """Read the first ``limit`` legacy replay rows for a method.

    Same schema as the CogVideo replay files; a short test run simply uses the
    leading arrivals of the 300-row decision file.
    """
    path = root / f"{method}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"replay decisions are missing: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    result: dict[int, dict[str, Any]] = {}
    for expected, row in enumerate(rows):
        if row.get("schema_version") != "cogvideox15-hit-replay/v1" or int(row.get("arrival_index", -1)) != expected:
            raise ValueError(f"invalid replay row {expected}: {path}")
        if expected < limit:
            previous = result.setdefault(int(row["request_index"]), row)
            if previous is not row:
                raise ValueError(f"replay request_index values must be unique: {path}")
    if len(result) != min(limit, len(rows)):
        raise ValueError(f"replay has fewer than {limit} usable rows: {path}")
    return result


def wan21_run(args: argparse.Namespace, rows: list[dict[str, Any]], view: Path, output: Path) -> list[dict[str, Any]]:
    config = resolved_config(f"cuda:{args.gpu}", args.dry_run, model_name="wan21",
                             early_signature_thresholds=getattr(args, "early_signature_thresholds", None))
    replay = None
    replay_path = None
    if args.method != "baseline":
        if args.replay_root is None:
            raise ValueError("Wan2.1 replay methods require --replay-root from the legacy experiment")
        replay_path = args.replay_root / f"{args.method}.jsonl"
        replay = load_replay_decisions(args.replay_root, args.method, limit=args.limit)
    if args.dry_run:
        # Keep the method plugins on the wan21 branch even with the synthetic
        # adapter behind it (same pattern as the CogVideo dry run).
        config = {**config, **load_yaml(ROOT / "configs" / "models" / "wan21.yaml"),
                  "model": "wan21", "dry_run": True, "device": f"cuda:{args.gpu}"}
    adapter = make_adapter("synthetic" if args.dry_run else "wan21", config)
    method = make_method(args.method, config)
    method.adapter = adapter
    if not args.dry_run and not args.resume:
        # A full warm-up generation costs ~18 minutes on Wan2.1; only preload
        # the resident models that would otherwise pollute the first request.
        warmup_paths = ["model_load"]
        if args.method == "chorus":
            from methods.chorus.grounded_sam import GroundedSam2Segmenter

            sam_config = {**config.get("chorus", {}).get("grounded_sam2", {}),
                          "device": str(config.get("device", "cuda"))}
            GroundedSam2Segmenter(sam_config)._load()
            warmup_paths.append("chorus_sam_model_loaded")
        write_warmup_record(output, paths=warmup_paths)
    runner = Wan21ReplayRunner(adapter, method, output_dir=output, replay=replay,
                               source_prompts={row["uuid"]: row for row in view_rows_for_preload(view)},
                               config=config)
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"
    (output / "run_manifest.json").write_text(json.dumps({
        "schema_version": "wan21-experiment/v1",
        "model": "Wan2.1-T2V-1.3B",
        "model_path": (str(Path(config["model_path"]).resolve()) if config.get("model_path") else None),
        "legacy_source_view": str(view),
        "legacy_source_view_manifest_sha256": sha256_file(view / "view_manifest.json"),
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "replay_root": str(args.replay_root) if args.replay_root else None,
        "replay_sha256": sha256_file(replay_path) if replay_path is not None else None,
        "git_revision": revision,
        "resolved_config": config,
        "replay_policy": "legacy hit/miss and source UUID are immutable ground truth; baseline runs every request",
        "miss_policy": "skipped without generation (status cache_miss, latency 0)",
        "temporary_cache_policy": "tmp_cache/<method>/<arrival_index>_<source_uuid>, removed in finally",
        "request_error_policy": "record error without changing replay decision",
        # Carnator's target runs a dense prefix before reuse and counts it in the
        # request's dit_forward_s, so the manifest has to say so or the denoising
        # column looks like a regression against a run that skipped the prefix.
        "target_probe_enabled": args.method == "carnator_boundary_b5",
        "target_probe_completed_step": 5 if args.method == "carnator_boundary_b5" else None,
        "target_probe_seed": args.seed if args.method == "carnator_boundary_b5" else None,
        "args": {"method": args.method, "gpu": args.gpu, "limit": args.limit, "seed": args.seed},
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    # The runner streams its raw RequestResult rows here; the canonical
    # requests.jsonl / request_records.jsonl are written below from the
    # flattened payloads, exactly like the generic and native paths.
    raw_path = output / "requests.raw.jsonl"
    run_results = runner.run(
        (Request(row["request_index"], row["uuid"], row["prompt"], args.seed + row["request_index"])
         for row in rows),
        raw_path,
    )
    payloads = flatten_replay_payloads(args, rows, run_results, output)
    raw_path.unlink(missing_ok=True)
    return payloads


def flatten_replay_payloads(args: argparse.Namespace, rows: list[dict[str, Any]], run_results: list[Any],
                            output: Path) -> list[dict[str, Any]]:
    """Turn a replay runner's raw results into the aggregation contract rows.

    Shared by the Wan2.1 and LTX-Video runs: both drive the same replay runner
    and must emit identical ``requests.jsonl`` / ``request_records.jsonl`` rows
    for the aggregator to compare them.
    """
    payloads = []
    for row, result in zip(rows, run_results, strict=True):
        payload = asdict(result)
        hit = bool(payload.get("cache_match", {}).get("hit"))
        extra = payload.get("extra", {})
        if payload.get("status") == "error":
            payload.update({"arrival_index": row["arrival_index"], "video_path": None, "quality_input_path": None,
                            "output_path": None, "denoising_s": None, "external_s": None, "hit_e2e_s": None,
                            "hit_denoising_s": None, "hit": hit,
                            "source_uuid": payload.get("cache_match", {}).get("source_uuid")})
        elif payload.get("status") == "cache_miss":
            payload.update({"arrival_index": row["arrival_index"], "video_path": None, "quality_input_path": None,
                            "output_path": None, "denoising_s": None, "external_s": 0.0, "hit_e2e_s": None,
                            "hit_denoising_s": None, "hit": False, "source_uuid": None, "latency_s": 0.0,
                            "executed_steps": 0})
        else:
            # Flatten the timing fields the aggregation contract reads; the
            # nested extra copy stays for the detailed audit trail.
            denoising_s = extra.get("dit_forward_s")
            if denoising_s is None and not args.dry_run:
                raise RuntimeError(f"{args.method} did not report CUDA-synchronized DiT forward time")
            denoising_s = float(denoising_s or 0.0)
            external_s = (float(extra.get("chorus_llm_s", 0.0)) + float(extra.get("chorus_sam_s", 0.0))
                          if args.method == "chorus" else 0.0)
            payload.update({"arrival_index": row["arrival_index"], "video_path": payload.get("output_path"),
                            "quality_input_path": payload.get("output_path"), "denoising_s": denoising_s,
                            "external_s": external_s, "hit": hit,
                            "hit_e2e_s": payload.get("latency_s") if hit else None,
                            "hit_denoising_s": denoising_s if hit else None,
                            "extra": {**extra, "denoising_timing": {
                                "definition": "cuda_synchronized_dit_forward",
                                "target_dit_forward_s": denoising_s - float(extra.get("probe_dit_forward_s", 0.0)),
                                "probe_dit_forward_s": float(extra.get("probe_dit_forward_s", 0.0)),
                                "total_dit_forward_s": denoising_s,
                                "probe_steps": extra.get("probe_steps", 0)}}})
        payloads.append(payload)
    for row, payload in zip(rows, payloads, strict=True):
        append_result_and_record(output, payload, row, args.method)
    return payloads


def ltxvideo_run(args: argparse.Namespace, rows: list[dict[str, Any]], view: Path, output: Path) -> list[dict[str, Any]]:
    """LTX-Video replay run, structurally identical to the Wan2.1 one.

    The same fixed legacy decisions drive the same five baselines through the
    same replay runner; only the adapter and the method plugins' model branch
    differ, plus the manifest's model identity.
    """
    config = resolved_config(f"cuda:{args.gpu}", args.dry_run, model_name="ltxvideo",
                             early_signature_thresholds=getattr(args, "early_signature_thresholds", None))
    replay = None
    replay_path = None
    if args.method != "baseline":
        if args.replay_root is None:
            raise ValueError("LTX-Video replay methods require --replay-root from the legacy experiment")
        replay_path = args.replay_root / f"{args.method}.jsonl"
        replay = load_replay_decisions(args.replay_root, args.method, limit=args.limit)
    if args.dry_run:
        config = {**config, **load_yaml(ROOT / "configs" / "models" / "ltxvideo.yaml"),
                  "model": "ltxvideo", "dry_run": True, "device": f"cuda:{args.gpu}"}
    adapter = make_adapter("synthetic" if args.dry_run else "ltxvideo", config)
    method = make_method(args.method, config)
    method.adapter = adapter
    if not args.dry_run and not args.resume:
        # The resident model plus Grounded-SAM-2 are the two loads that would
        # otherwise land inside the first timed request.
        warmup_paths = ["model_load"]
        if args.method == "chorus":
            from methods.chorus.grounded_sam import GroundedSam2Segmenter

            sam_config = {**config.get("chorus", {}).get("grounded_sam2", {}),
                          "device": str(config.get("device", "cuda"))}
            GroundedSam2Segmenter(sam_config)._load()
            warmup_paths.append("chorus_sam_model_loaded")
        write_warmup_record(output, paths=warmup_paths)
    runner = Wan21ReplayRunner(adapter, method, output_dir=output, replay=replay,
                               source_prompts={row["uuid"]: row for row in view_rows_for_preload(view)},
                               config=config, model="LTX-Video-0.9.7-dev")
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"
    (output / "run_manifest.json").write_text(json.dumps({
        "schema_version": "ltxvideo-experiment/v1",
        "model": "LTX-Video-0.9.7-dev",
        "model_path": (str(Path(config["model_path"]).resolve()) if config.get("model_path") else None),
        "legacy_source_view": str(view),
        "legacy_source_view_manifest_sha256": sha256_file(view / "view_manifest.json"),
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "replay_root": str(args.replay_root) if args.replay_root else None,
        "replay_sha256": sha256_file(replay_path) if replay_path is not None else None,
        "git_revision": revision,
        "resolved_config": config,
        "replay_policy": "legacy hit/miss and source UUID are immutable ground truth; baseline runs every request",
        "miss_policy": "skipped without generation (status cache_miss, latency 0)",
        "temporary_cache_policy": "tmp_cache/<method>/<arrival_index>_<source_uuid>, removed in finally",
        "request_error_policy": "record error without changing replay decision",
        # Carnator's target runs a dense prefix before reuse and counts it in the
        # request's dit_forward_s, so the manifest has to say so or the denoising
        # column looks like a regression against a run that skipped the prefix.
        "target_probe_enabled": args.method == "carnator_boundary_b5",
        "target_probe_completed_step": 5 if args.method == "carnator_boundary_b5" else None,
        "target_probe_seed": args.seed if args.method == "carnator_boundary_b5" else None,
        "args": {"method": args.method, "gpu": args.gpu, "limit": args.limit, "seed": args.seed},
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    raw_path = output / "requests.raw.jsonl"
    run_results = runner.run(
        (Request(row["request_index"], row["uuid"], row["prompt"], args.seed + row["request_index"])
         for row in rows),
        raw_path,
    )
    payloads = flatten_replay_payloads(args, rows, run_results, output)
    raw_path.unlink(missing_ok=True)
    return payloads


def source_path(view: Path, row: dict[str, Any], method: str) -> Path:
    root = view / "entries" / str(row["uuid"])
    source = root / "carnator" / "cache" / "source_dense"
    path = source / "latents_step_004.pt" if method == "nirvana_vid" else source
    if not path.exists():
        raise FileNotFoundError(f"source payload is missing: {path}")
    return path


def preload_view(runner: Runner, view: Path, rows: list[dict[str, Any]], method_name: str) -> None:
    """Install view entries into a method-local NamespaceCache.

    Latent files remain symlinks (or direct read-only paths) into the view; the
    NamespaceCache can still add target misses below the run directory.
    """
    if method_name not in {"nirvana_vid", "chorus"}:
        return
    retriever = runner.method.retriever
    if retriever is None:
        return
    entries: dict[str, CacheEntry] = {}
    prompts: list[str] = []
    records_meta: list[dict[str, Any]] = []
    for row in rows:
        uuid = str(row["uuid"])
        prompt = str(row["prompt"])
        path = source_path(view, row, method_name)
        if method_name == "chorus":
            # Chorus derives the source video from <uuid>.source.mp4.
            alias = runner.cache.entries / f"{uuid}.latent"
            video_alias = runner.cache.entries / f"{uuid}.source.mp4"
            alias.parent.mkdir(parents=True, exist_ok=True)
            if not alias.exists():
                alias.symlink_to(path, target_is_directory=True)
            source_video = view / "entries" / uuid / "source.mp4"
            if not video_alias.exists():
                video_alias.symlink_to(source_video)
            path = alias
        entries[uuid] = CacheEntry(uuid, int(row.get("request_index", 0)), prompt, str(path), 5, 1, time.time(), metadata={"source_view": True})
        prompts.append(prompt)
        records_meta.append({"uuid": uuid, "term": prompt})
    vectors = retriever.embedder.encode(prompts)
    runner.cache._entries.update(entries)
    runner.cache._records = [VectorRecord(item["uuid"], item["term"], vector) for item, vector in zip(records_meta, vectors, strict=True)]
    runner.cache._retrieval_signature = retriever.signature
    runner.cache._save()
    runner.cache._save_retrieval()
    retriever.bind_vector_store(runner.cache.vector_store)
    retriever.sync_vector_store(runner.cache._records)


def write_warmup_record(output: Path, *, paths: list[str], status: str = "completed") -> None:
    (output / "warmup.json").write_text(json.dumps({
        "status": status,
        "source_prompt": WARMUP_SOURCE_PROMPT,
        "target_prompt": WARMUP_TARGET_PROMPT,
        "seed": 20260218,
        "paths": paths,
        "timed": False,
        "shared_cache_written": False,
    }, indent=2) + "\n", encoding="utf-8")


def synchronize_cuda() -> None:
    import torch

    torch.cuda.synchronize()


def warmup_generic(adapter: Any, method_name: str, output: Path, view: Path) -> None:
    """Exercise each generic path without adding a benchmark request."""
    root = output / "warmup-work"
    root.mkdir(parents=True, exist_ok=True)
    source = Request(-1, "warmup-dog", WARMUP_SOURCE_PROMPT, 20260218)
    target = Request(-2, "warmup-cat", WARMUP_TARGET_PROMPT, 20260218, cache_prompt=WARMUP_SOURCE_PROMPT)
    if method_name == "baseline":
        adapter.generate(target, capture=False, method="baseline")
        paths = ["baseline_warmup"]
    elif method_name == "teacache":
        adapter.generate(target, capture=False, method="teacache")
        paths = ["target_only"]
    elif method_name == "nirvana_vid":
        source_result = adapter.generate(source, capture=True, method="nirvana_vid")
        adapter.generate(target, cached_latent=source_result.latent[5], capture=False, method="nirvana_vid")
        paths = ["source_capture", "latent_reuse"]
    elif method_name == "chorus":
        adapter.generate(source, capture=True, method="chorus")
        # Load the external models and their kernels outside the timed
        # requests. Detection itself runs on the real source video, where a
        # synthetic warm-up clip can otherwise produce a detector miss.
        from methods.chorus.grounded_sam import GroundedSam2Segmenter
        chorus_cfg = adapter.config.get("chorus", {})
        sam_config = {
            **chorus_cfg.get("grounded_sam2", {}),
            "device": str(adapter.config.get("device", "cuda")),
        }
        GroundedSam2Segmenter(sam_config)._load()
        paths = ["source_capture", "chorus_sam_model_loaded"]
    elif method_name == "chorus2":
        adapter.generate(target, capture=False, method="chorus2")
        paths = ["chorus2_warmup"]
    else:
        raise ValueError(f"unsupported generic warm-up method: {method_name}")
    synchronize_cuda()
    shutil.rmtree(root)
    write_warmup_record(output, paths=paths)


def generic_run(args: argparse.Namespace, rows: list[dict[str, Any]], view: Path, output: Path) -> list[dict[str, Any]]:
    method_name = args.method
    config = resolved_config(
        f"cuda:{args.gpu}",
        args.dry_run,
        model_name=getattr(args, "model", "wan22"),
        early_signature_thresholds=getattr(args, "early_signature_thresholds", None),
    )
    adapter = make_adapter("synthetic" if args.dry_run else getattr(args, "model", "wan22"), config)
    method = make_method(method_name, config)
    method.config["persist_target_cache"] = False
    runner = Runner(
        adapter,
        method,
        model=adapter.model_name,
        output_dir=output,
        cache_root=output / "cache",
        # The immutable source view is the only cache input for this
        # experiment.  A target miss produces a video and audit record, but
        # must not add a latent to the method-local cache.
        persist_target_cache=False,
        # End-to-end latency is the complete request wall-clock.  Chorus's
        # external models are retained as a separate breakdown below.
        exclude_preprocess_latency=False,
        exclude_cache_prepare_latency=True,
        skip_generation_on_miss=True,
    )
    source_rows = view_rows_for_preload(view)
    if not args.dry_run:
        preload_view(runner, view, source_rows, method_name)
    install_forced_generic_decisions(
        method,
        args,
        source_by_uuid={row["uuid"]: row for row in source_rows},
        config=config,
    )
    if args.dry_run:
        write_warmup_record(output, paths=[], status="skipped_dry_run")
    elif not args.resume:
        warmup_generic(adapter, method_name, output, view)
    results: list[dict[str, Any]] = []
    for row in rows:
        request = Request(row["request_index"], row["uuid"], row["prompt"], args.seed + row["request_index"])
        result = runner.process(request)
        payload = asdict(result)
        timing = payload.get("extra", {}).get("timing", {})
        external_s = (
            float(payload.get("extra", {}).get("chorus_llm_s", 0.0))
            + float(payload.get("extra", {}).get("chorus_sam_s", 0.0))
            if method_name == "chorus" else 0.0
        )
        hit = bool(payload.get("cache_match", {}).get("hit"))
        if payload.get("status") == "error":
            # ``Runner.process`` contains adapter failures within a result.
            # Preserve that result as a terminal audit event and let the
            # remaining independent requests run.  The error result has no
            # reliable DiT or external breakdown, so those fields are null.
            payload.update({
                "arrival_index": row["arrival_index"],
                "video_path": None,
                "quality_input_path": None,
                "output_path": None,
                "denoising_s": None,
                "external_s": None,
                "hit_e2e_s": None,
                "hit_denoising_s": None,
                "hit": hit,
                "source_uuid": payload.get("cache_match", {}).get("source_uuid"),
                "target_cache_persisted": False,
                "extra": {**payload.get("extra", {}), "denoising_timing": None},
            })
            append_result_and_record(output, payload, row, method_name)
            results.append(payload)
            continue
        if payload.get("status") == "cache_miss":
            payload.update({
                "arrival_index": row["arrival_index"],
                "video_path": None,
                "quality_input_path": None,
                "output_path": None,
                "denoising_s": None,
                "external_s": 0.0,
                "hit_e2e_s": None,
                "hit_denoising_s": None,
                "hit": False,
                "source_uuid": None,
                "latency_s": 0.0,
                "executed_steps": 0,
                "target_cache_persisted": False,
                "extra": {**payload.get("extra", {}), "denoising_timing": None},
            })
            append_result_and_record(output, payload, row, method_name)
            results.append(payload)
            continue
        raw_dit_forward_s = payload.get("extra", {}).get("dit_forward_s")
        if raw_dit_forward_s is None and not args.dry_run:
            raise RuntimeError(f"{method_name} did not report CUDA-synchronized DiT forward time")
        denoising_s = float(raw_dit_forward_s or 0.0)
        payload.update({
            "arrival_index": row["arrival_index"],
            "video_path": payload.get("output_path"),
            # External is an E2E component for Chorus and remains a separate
            # breakdown; Denoising itself is DiT forward time only.
            "denoising_s": denoising_s,
            "external_s": external_s,
            "hit": hit,
            "hit_e2e_s": payload.get("latency_s") if hit else None,
            "hit_denoising_s": denoising_s if hit else None,
            "target_cache_persisted": False,
            "extra": {
                **payload.get("extra", {}),
                "denoising_timing": {
                    "definition": "cuda_synchronized_dit_forward",
                    "target_dit_forward_s": denoising_s,
                    "probe_dit_forward_s": 0.0,
                    "total_dit_forward_s": denoising_s,
                },
            },
        })
        append_result_and_record(output, payload, row, method_name)
        results.append(payload)
    return results


def view_rows_for_preload(view: Path) -> list[dict[str, Any]]:
    return [{"request_index": int(item.get("request_index", i)), "uuid": item["uuid"], "prompt": item["prompt"]}
            for i, item in enumerate(load_view(view))]


def build_request_record(
    payload: dict[str, Any],
    row: dict[str, Any],
    method_name: str,
) -> dict[str, Any]:
    """Return the stable per-arrival retrieval audit projection."""
    cache_match = payload.get("cache_match") if isinstance(payload.get("cache_match"), dict) else {}
    metadata = cache_match.get("metadata") if isinstance(cache_match.get("metadata"), dict) else {}
    top_k = metadata.get("top_k") if isinstance(metadata.get("top_k"), list) else []
    similarity = cache_match.get("score")
    if similarity is None and top_k and isinstance(top_k[0], dict):
        similarity = top_k[0].get("score")
    record: dict[str, Any] = {
        "arrival_index": row["arrival_index"],
        "request_index": row["request_index"],
        "uuid": row["uuid"],
        "prompt": row["prompt"],
        "method": method_name,
        "status": payload.get("status"),
        "error": payload.get("error"),
        "hit": bool(cache_match.get("hit", payload.get("hit", False))),
        "hit_object": cache_match.get("source_uuid", payload.get("source_uuid")),
        "similarity": similarity,
        "retrieval_kind": cache_match.get("kind") or metadata.get("retriever"),
        "external_s": (
            float(payload["external_s"])
            if payload.get("external_s") is not None else None
        ),
        "latency_s": payload.get("latency_s"),
        "denoising_s": payload.get("denoising_s"),
        "hit_e2e_s": payload.get("hit_e2e_s"),
        "hit_denoising_s": payload.get("hit_denoising_s"),
    }
    if method_name == "carnator_boundary_b5":
        record["retrieval"] = {
            "stage_1": {
                "top_k": metadata.get("top_k", []),
                "candidate_top_k": metadata.get("candidate_top_k"),
                "candidate_min_similarity": metadata.get("candidate_min_similarity"),
            },
            "noun_prefilter": metadata.get("noun_prefilter", []),
            "stage_2": {
                "profile": metadata.get("stage_2_profile"),
                "reference_profile": metadata.get("stage_2_reference_profile"),
                "operational": metadata.get("stage_2_operational"),
                "thresholds": metadata.get("stage_2_thresholds", EARLY_SIGNATURE_THRESHOLDS),
                "threshold_mode": metadata.get("stage_2_threshold_mode"),
                "candidates": metadata.get("stage_2_candidates", []),
                "comparisons": metadata.get("stage_2", []),
                "probe_required": metadata.get("probe_required", False),
                "probe_run": metadata.get("step_2_probe_run", False),
                "target_probe": metadata.get("target_probe"),
                "inference_mask_selection": metadata.get("inference_mask_selection"),
            },
            "reason": metadata.get("reason"),
        }
    else:
        record["retrieval"] = {"metadata": metadata}
    return record


def append_result_and_record(
    output: Path,
    payload: dict[str, Any],
    row: dict[str, Any],
    method_name: str,
) -> None:
    record = build_request_record(payload, row, method_name)
    with (output / "requests.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
    with (output / "request_records.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def pending_rows_for_resume(
    output: Path,
    rows: list[dict[str, Any]],
    *,
    resume: bool,
) -> tuple[list[dict[str, Any]], int]:
    """Return only rows with a complete, matching terminal record on resume.

    A result is considered durable only after both JSONL audit files contain a
    matching terminal entry.  ``error`` is terminal so a resume cannot silently
    re-run a request that has already been audited as failed.
    """
    if not resume:
        return rows, 0

    results_path = output / "requests.jsonl"
    records_path = output / "request_records.jsonl"
    if not results_path.exists() and not records_path.exists():
        return rows, 0
    if not results_path.exists() or not records_path.exists():
        raise RuntimeError("cannot resume: requests.jsonl and request_records.jsonl must either both exist or both be absent")

    expected = {int(row["arrival_index"]): row for row in rows}

    def load_completed(path: Path, *, record: bool) -> set[int]:
        completed: set[int] = set()
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise RuntimeError(f"cannot resume: failed to read {path}") from exc
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                arrival = int(payload["arrival_index"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"cannot resume: invalid JSONL record at {path}:{line_number}") from exc
            row = expected.get(arrival)
            if row is None:
                raise RuntimeError(f"cannot resume: {path}:{line_number} is outside the requested input slice")
            for field in ("request_index", "uuid", "prompt"):
                if payload.get(field) != row[field]:
                    raise RuntimeError(
                        f"cannot resume: {path}:{line_number} does not match input {field} for arrival {arrival}"
                    )
            if record or payload.get("status") in {"ok", "cache_miss", "error"}:
                if arrival in completed:
                    raise RuntimeError(f"cannot resume: duplicate completed arrival {arrival} in {path}")
                completed.add(arrival)
        return completed

    completed_results = load_completed(results_path, record=False)
    completed_records = load_completed(records_path, record=True)
    if completed_results != completed_records:
        raise RuntimeError(
            "cannot resume: requests.jsonl and request_records.jsonl disagree on completed arrivals "
            f"({sorted(completed_results)} != {sorted(completed_records)})"
        )
    return [row for row in rows if int(row["arrival_index"]) not in completed_results], len(completed_results)


def prompt_matcher(
    view: Path,
    config: dict[str, Any],
    dry_run: bool,
    method_name: str,
    *,
    early_signature_thresholds: dict[str, float] | None = None,
):
    from core.retrieval import make_chorus_retriever

    # Native target workers only need the ranked candidate.  Keep the
    # experiment's CLIP/threshold settings but use the in-process ranking
    # path here; the generic Chorus runner persists Chroma separately.
    retrieval = {**config.get("retrieval", {}).get("chorus", {}), "vector_backend": "memory"}
    retriever = make_chorus_retriever(retrieval, dry_run=dry_run)
    source_rows = view_rows_for_preload(view)
    prompts = [row["prompt"] for row in source_rows]
    vectors = retriever.embedder.encode(prompts)
    records = [VectorRecord(row["uuid"], row["prompt"], vector) for row, vector in zip(source_rows, vectors, strict=True)]
    source_by_uuid = {row["uuid"]: row for row in source_rows}
    if method_name != "carnator_boundary_b5":
        return retriever, records, source_by_uuid, None

    from carnator.retrieval import make_carnator_retriever

    carnator_retrieval = {
        **config.get("retrieval", {}).get("carnator", {}),
        **(early_signature_thresholds or EARLY_SIGNATURE_THRESHOLDS),
        "vector_backend": "memory",
    }
    carnator = make_carnator_retriever(
        carnator_retrieval,
        dry_run=False,
        retrieval_only=True,
    )
    carnator_vectors = carnator.embedder.encode(prompts)
    carnator_records: list[VectorRecord] = []
    for row, vector in zip(source_rows, carnator_vectors, strict=True):
        cache_dir = view / "entries" / row["uuid"] / "carnator" / "cache" / "source_dense"
        signature_path = cache_dir / "online_color_signature.json"
        valid_signature = has_per_noun_inference_signature(signature_path)
        if valid_signature:
            from carnator.early_signature import load_per_noun_mbase

            nouns = tuple(sorted(load_per_noun_mbase({"signature_path": str(signature_path)})))
        else:
            nouns = ()
        metadata = {
            "source_request_index": int(row.get("request_index", 0)),
            "trf_nouns": list(nouns),
            "cache_ready": valid_signature and (cache_dir / "fusion_masks" / "masks.pt").is_file(),
            "signature": {"signature_path": str(signature_path)} if valid_signature else None,
        }
        carnator_records.append(VectorRecord(row["uuid"], row["prompt"], vector, metadata, int(row.get("request_index", 0))))
    return retriever, records, source_by_uuid, (carnator, carnator_records)


def write_fake_dense_manifest(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"mode": "T2V", "steps": 50, "output": str(path.with_name("target_dense.mp4")),
                                "timing": {"end_to_end_seconds": 1.0}}) + "\n", encoding="utf-8")


class NativeWanWorker:
    """One resident native Wan runtime for one benchmark worker/GPU."""

    def __init__(self, args: argparse.Namespace, config: dict[str, Any]) -> None:
        from carnator.adapters import wan_ti2v_5b as native

        self.native = native
        self.args = native.build_parser().parse_args([
            "--no-env-setup",
            "--model-dir", str(config["model_path"]),
            "--device", f"cuda:{args.gpu}",
            "--steps", "50",
            "--mask-probe-step", "5",
            "--reuse-start-step", "10",
            "--mask-layers", "15",
            "--disable-progress",
        ])
        native.ensure_import_paths(self.args)
        native.validate_experiment_config(self.args)
        self.runtime = native.load_runtime(self.args)
        self._closed = False

    def run_carnator(self, argv: list[str]) -> dict[str, Any]:
        native_args = self.native.build_parser().parse_args(argv)
        native_args.clean = True
        return self.native.run_experiment(native_args, runtime=self.runtime)

    def run_chorus2(self, argv: list[str]) -> dict[str, Any]:
        from chorus2 import wan_ti2v_5b as chorus2

        chorus2_args = chorus2.build_parser().parse_args(argv)
        chorus2_args.clean = True
        chorus2_args.outer_attn_backend = "dense" if chorus2_args.target_dense_only else "sparge-sage2"
        return chorus2.run_experiment(chorus2_args, runtime=self.runtime)

    def close(self) -> None:
        if self._closed:
            return
        self.runtime.profiler.close()
        if self.args.offload_model:
            self.runtime.pipe.model.cpu()
        self.runtime.torch.cuda.empty_cache()
        self._closed = True


def native_dense_argv(
    args: argparse.Namespace,
    config: dict[str, Any],
    artifact_root: Path,
    results_root: Path,
    prompt: str,
    seed: int,
    *,
    resume_latent: Path | None = None,
) -> list[str]:
    argv = [
        "--no-env-setup", "--model-dir", str(config["model_path"]), "--device", f"cuda:{args.gpu}",
        "--artifact-root", str(artifact_root), "--results-root", str(results_root),
        "--run-id", "run", "--target-prompt", prompt, "--seed", str(seed),
        "--target-dense-only", "--outer-attn-backend", "dense", "--ffn-backend", "fp",
        "--skip-w8a8-dense-diagnostic",
    ]
    if resume_latent is not None:
        argv.extend(("--target-resume-latent", str(resume_latent), "--target-resume-step", "5"))
    return argv


def warmup_native(
    args: argparse.Namespace,
    config: dict[str, Any],
    view: Path,
    output: Path,
    worker: NativeWanWorker,
) -> None:
    """Warm the resident runtime without producing a scored request."""
    root = output / "warmup-work"
    root.mkdir(parents=True, exist_ok=True)
    worker.run_carnator(native_dense_argv(
        args, config, root / "dense-artifacts", root / "dense-results", WARMUP_TARGET_PROMPT, 20260218,
    ))
    source = view_rows_for_preload(view)[0]
    entry = view / "entries" / source["uuid"]
    if args.method == "chorus2":
        worker.run_chorus2([
            "--model-dir", str(config["model_path"]), "--device", f"cuda:{args.gpu}",
            "--artifact-root", str(root / "reuse-artifacts"), "--results-root", str(root / "reuse-results"),
            "--run-id", "run", "--source-prompt", source["prompt"], "--target-prompt", WARMUP_TARGET_PROMPT,
            "--seed", "20260218",
            "--source-mask-cache", str(entry / "carnator" / "cache" / "source_dense" / "outer_block_masks"),
        ])
        paths = ["dense_c0", "chorus2_reuse"]
    else:
        collector = make_native_target_collector(args, config, root / "probe", 20260218, runtime=worker.runtime)
        try:
            collector.process(
                Request(-2, "warmup-cat", WARMUP_TARGET_PROMPT, 20260218), ["cat"],
                evaluate_probe=lambda _: CacheMatch(kind="no_reuse"),
            )
        finally:
            collector.close()
        manifest = root / "target_dense_manifest.json"
        write_fake_dense_manifest(manifest)
        worker.run_carnator([
            "--no-env-setup", "--model-dir", str(config["model_path"]), "--device", f"cuda:{args.gpu}",
            "--artifact-root", str(root / "reuse-artifacts"), "--results-root", str(root / "reuse-results"),
            "--run-id", "run", "--source-prompt", source["prompt"], "--target-prompt", WARMUP_TARGET_PROMPT,
            "--seed", "20260218", "--source-run-dir", str(entry / "carnator"),
            "--target-dense-manifest", str(manifest), "--outer-attn-backend", "sparge-sage2",
            "--ffn-backend", "w8a8", "--skip-w8a8-dense-diagnostic",
        ])
        paths = ["dense_c0", "probe_b5", "carnator_reuse"]
    synchronize_cuda()
    shutil.rmtree(root)
    write_warmup_record(output, paths=paths)


def make_native_target_collector(
    args: argparse.Namespace,
    config: dict[str, Any],
    artifact_root: Path,
    probe_seed: int,
    *,
    runtime: Any | None = None,
):
    from carnator.online_collector import NativeOnlineCollector
    from carnator.adapters import wan_ti2v_5b as native

    collector_args = native.build_parser().parse_args([
        "--no-env-setup",
        "--model-dir", str(config["model_path"]),
        "--device", f"cuda:{args.gpu}",
        "--steps", "50",
        "--mask-probe-step", "5",
        "--reuse-start-step", "10",
        "--mask-layers", "15",
        "--disable-progress",
    ])
    return NativeOnlineCollector(
        collector_args,
        artifact_root,
        probe_seed=probe_seed,
        noun_extractor="spacy/en_core_web_sm:NOUN",
        retain_artifacts=args.retain_native_artifacts,
        runtime=runtime,
    )


def compact_native_run(req_root: Path, summary_path: Path) -> Path:
    """Keep the native summary/report and top-level video only.

    Native adapters use an artifact tree for intermediate videos, frames and
    cache manifests.  The experiment-level JSONL is the source of truth, so
    retaining that tree would duplicate every video and grow the run output
    unnecessarily.
    """
    compact_summary = req_root / "summary.json"
    shutil.copy2(summary_path, compact_summary)
    report = req_root / "results" / "run" / "report.md"
    if report.is_file():
        shutil.copy2(report, req_root / "report.md")
    artifacts = req_root / "artifacts"
    if artifacts.is_dir():
        shutil.rmtree(artifacts)
    results = req_root / "results"
    if results.is_dir():
        shutil.rmtree(results)
    manifest = req_root / "target_dense_manifest.json"
    if manifest.is_file():
        manifest.unlink()
    probe_root = req_root / "target_probe"
    if probe_root.is_dir():
        shutil.rmtree(probe_root)
    return compact_summary


def native_run(args: argparse.Namespace, rows: list[dict[str, Any]], view: Path, output: Path) -> list[dict[str, Any]]:
    config = resolved_config(
        f"cuda:{args.gpu}",
        args.dry_run,
        model_name=getattr(args, "model", "wan22"),
        early_signature_thresholds=getattr(args, "early_signature_thresholds", None),
    )
    matcher = None if args.dry_run else prompt_matcher(
        view,
        config,
        False,
        args.method,
        early_signature_thresholds=getattr(args, "early_signature_thresholds", None),
    )
    output.mkdir(parents=True, exist_ok=True)
    resident_collector = None
    native_worker: NativeWanWorker | None = None

    def worker() -> NativeWanWorker:
        nonlocal native_worker
        if native_worker is None:
            native_worker = NativeWanWorker(args, config)
        return native_worker

    if args.dry_run:
        write_warmup_record(output, paths=[], status="skipped_dry_run")
    elif not args.resume:
        warmup_native(args, config, view, output, worker())

    for row in rows:
        arrival = row["arrival_index"]
        target = Request(row["request_index"], row["uuid"], row["prompt"], args.seed + row["request_index"])
        if args.dry_run:
            video = output / "videos" / f"{arrival:04d}_{row['uuid']}.mp4"
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(b"CHAI-DRY-RUN:vidprom\n" + row["prompt"].encode())
            payload = {"arrival_index": arrival, "request_index": target.request_index, "uuid": target.uuid, "prompt": target.prompt,
                       "model": "synthetic", "method": args.method, "seed": target.seed, "cache_match": {"hit": False},
                       "hit": False, "executed_steps": 50, "latency_s": 0.0, "denoising_s": 0.0,
                       "hit_e2e_s": None, "hit_denoising_s": None,
                       "status": "ok", "target_cache_persisted": False, "output_path": str(video),
                       "video_path": str(video), "quality_input_path": str(video)}
        else:
            retriever, records, source_by_uuid, carnator_matcher = matcher
            request_started = time.perf_counter()
            forced_outcome = forced_smoke_decision(args, target.request_index)
            observed_match = retriever.search(target, records)
            match = (
                force_cache_match(
                    observed_match,
                    outcome=forced_outcome,
                    request_index=target.request_index,
                    method_name=args.method,
                    source_by_uuid=source_by_uuid,
                    config=config,
                )
                if forced_outcome is not None
                else observed_match
            )
            carnator_match = None
            target_nouns: tuple[str, ...] = ()
            if carnator_matcher is not None:
                carnator_retriever, carnator_records = carnator_matcher
                target_nouns = carnator_retriever.noun_extractor.for_request(target)
                observed_carnator_match = carnator_retriever.search(target, carnator_records)
                carnator_match = (
                    forced_carnator_probe_match(
                        observed_carnator_match,
                        request_index=target.request_index,
                        outcome=forced_outcome,
                    )
                    if forced_outcome is not None
                    else observed_carnator_match
                )
            req_root = output / "native" / f"request-{arrival:04d}"
            req_root.mkdir(parents=True, exist_ok=True)
            probe_result = None
            probe_failure = None
            if carnator_matcher is not None and carnator_match is not None and carnator_match.kind == "probe_required":
                probe_root = req_root / "target_probe"
                if resident_collector is None:
                    resident_collector = make_native_target_collector(
                        args, config, probe_root, args.seed, runtime=worker().runtime
                    )
                else:
                    resident_collector.artifact_root = probe_root.resolve()
                try:
                    def evaluate_probe(signature: dict[str, Any]) -> CacheMatch:
                        observed_probe_match = carnator_retriever.search(
                            target, carnator_records, target_signature=signature
                        )
                        if forced_outcome is None:
                            return observed_probe_match
                        forced_match = force_cache_match(
                            observed_probe_match,
                            outcome=forced_outcome,
                            request_index=target.request_index,
                            method_name=args.method,
                            source_by_uuid=source_by_uuid,
                            config=config,
                        )
                        return attach_forced_carnator_mask_selection(
                            forced_match,
                            records=carnator_records,
                            target_nouns=target_nouns,
                        )

                    probe_result = resident_collector.process(
                        target,
                        target_nouns,
                        evaluate_probe=evaluate_probe,
                    )
                except ValueError as exc:
                    if str(exc) != "fused Mbase is empty at every time token":
                        raise
                    probe_failure = "fused_mbase_empty_at_all_time_tokens"
                    if not args.retain_native_artifacts:
                        shutil.rmtree(probe_root, ignore_errors=True)
                if probe_result is not None:
                    carnator_match = probe_result.match or carnator_match
            execution_match = carnator_match if carnator_match is not None else match
            hit = bool(execution_match.hit)
            retrieval_metadata = dict(execution_match.metadata or {})
            if carnator_match is not None:
                retrieval_metadata["execution_candidate"] = {
                    "source_uuid": execution_match.source_uuid,
                    "score": execution_match.score,
                    "retriever": execution_match.metadata.get("retriever"),
                }
                if probe_result is not None:
                    retrieval_metadata["target_probe"] = {
                        "early_probe_s": probe_result.early_probe_s,
                        "early_probe_wall_s": probe_result.early_probe_wall_s,
                        "precision_fallback": probe_result.probe_precision_fallback,
                        "signature_gate_s": probe_result.signature_gate_s,
                        "completed_steps": probe_result.completed_steps,
                        "probe_collected": probe_result.probe_collected,
                    }
                if probe_failure is not None:
                    retrieval_metadata["target_probe_failure"] = probe_failure
            if not hit:
                retrieval_metadata["execution_reason"] = "cache_miss"
                payload = {"arrival_index": arrival, "request_index": target.request_index, "uuid": target.uuid,
                           "prompt": target.prompt, "model": "Wan2.2", "method": args.method, "seed": target.seed,
                           "cache_match": {"hit": False, "source_uuid": None, "score": None, "kind": execution_match.kind, "metadata": retrieval_metadata},
                           "hit": False, "source_uuid": None, "executed_steps": 0, "latency_s": 0.0,
                           "denoising_s": None, "hit_e2e_s": None, "hit_denoising_s": None,
                           "external_s": 0.0, "status": "cache_miss", "target_cache_persisted": False,
                           "output_path": None, "video_path": None, "quality_input_path": None,
                           "extra": {"retrieval": retrieval_metadata, "denoising_timing": None}}
                append_result_and_record(output, payload, row, args.method)
                continue
            if execution_match.source_uuid is None:
                raise RuntimeError("reuse execution has no selected source UUID")
            source = source_by_uuid[execution_match.source_uuid]
            entry = view / "entries" / execution_match.source_uuid
            if args.method == "chorus2":
                command = [sys.executable, "-m", "chorus2.wan_ti2v_5b", "--model-dir", config["model_path"], "--device", f"cuda:{args.gpu}",
                           "--artifact-root", str(req_root / "artifacts"), "--results-root", str(req_root / "results"), "--run-id", "run",
                           "--source-prompt", source["prompt"], "--target-prompt", target.prompt, "--seed", str(target.seed),
                           "--source-mask-cache", str(entry / "carnator" / "cache" / "source_dense" / "outer_block_masks")]
            else:
                from carnator.early_signature import materialize_per_noun_inference_mask_bundle

                selection = retrieval_metadata.get("inference_mask_selection")
                if not isinstance(selection, dict):
                    raise RuntimeError("Carnator cache hit has no per-noun inference-mask selection")
                source_cache = entry / "carnator" / "cache" / "source_dense"
                selected_mask = materialize_per_noun_inference_mask_bundle(
                    {"signature_path": str(source_cache / "online_color_signature.json")},
                    target_nouns=target_nouns,
                    source_fusion_mask_dir=source_cache / "fusion_masks",
                    output_dir=req_root / "selected_source_mask",
                )
                if (
                    selected_mask["selected_source_noun"] != selection.get("selected_source_noun")
                    or selected_mask["selection_reason"] != selection.get("selection_reason")
                ):
                    raise RuntimeError("Carnator inference-mask selection changed after retrieval")
                retrieval_metadata["inference_mask_selection"] = selected_mask
                fake_manifest = req_root / "target_dense_manifest.json"
                write_fake_dense_manifest(fake_manifest)
                command = [sys.executable, "-m", "carnator.adapters.wan_ti2v_5b", "--no-env-setup", "--model-dir", config["model_path"],
                           "--device", f"cuda:{args.gpu}", "--artifact-root", str(req_root / "artifacts"), "--results-root", str(req_root / "results"),
                           "--run-id", "run", "--source-prompt", source["prompt"], "--target-prompt", target.prompt, "--seed", str(target.seed),
                           "--source-run-dir", str(entry / "carnator"), "--target-dense-manifest", str(fake_manifest),
                           "--source-mask-dir", selected_mask["mask_bundle_dir"],
                           "--outer-attn-backend", "sparge-sage2", "--ffn-backend", "w8a8", "--skip-w8a8-dense-diagnostic"]
            summary_path = req_root / "results" / "run" / "summary.json"
            reusable_summary = False
            if summary_path.is_file():
                try:
                    existing = json.loads(summary_path.read_text(encoding="utf-8"))
                    existing_target = existing.get("runs", {}).get("target", existing.get("runs", {}).get("target_latent_reuse", {}))
                    reusable_summary = (
                        existing.get("status") == "completed"
                        and isinstance(existing_target, dict)
                        and existing_target.get("dit_forward_seconds") is not None
                    )
                except (OSError, json.JSONDecodeError):
                    reusable_summary = False
            if not reusable_summary:
                if args.method == "carnator_boundary_b5":
                    worker().run_carnator(command[3:])
                else:
                    worker().run_chorus2(command[3:])
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            target_run = summary["runs"].get("target", summary["runs"].get("target_latent_reuse", {}))
            native_video = Path(target_run["output"])
            video = output / "videos" / f"{arrival:04d}_{target.uuid}.mp4"
            video.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(native_video, video)
            compact_summary = (
                summary_path if args.retain_native_artifacts
                else compact_native_run(req_root, summary_path)
            )
            probe_denoising_s = 0.0
            if probe_result is not None:
                probe_denoising_s = float(getattr(probe_result, "dit_forward_s", 0.0))
            cache_prepare = target_run.get("cache_prepare") or {}
            prepare_s = float(cache_prepare.get("elapsed_seconds", 0.0)) if isinstance(cache_prepare, dict) else 0.0
            latency_s = max(0.0, time.perf_counter() - request_started - prepare_s)
            target_dit_forward_s = target_run.get("dit_forward_seconds")
            if target_dit_forward_s is None:
                raise RuntimeError("native target summary did not report CUDA-synchronized DiT forward time")
            target_dit_forward_s = float(target_dit_forward_s)
            denoising_s = target_dit_forward_s + probe_denoising_s
            probe_steps = int(getattr(probe_result, "completed_steps", 0)) if probe_result is not None else 0
            target_steps = int(target_run.get("completed_steps") or 40)
            payload = {"arrival_index": arrival, "request_index": target.request_index, "uuid": target.uuid, "prompt": target.prompt,
                       "model": "Wan2.2", "method": args.method, "seed": target.seed,
                       "cache_match": {"hit": True, "source_uuid": execution_match.source_uuid, "score": execution_match.score, "kind": execution_match.kind, "metadata": retrieval_metadata},
                       "hit": True, "source_uuid": execution_match.source_uuid, "executed_steps": target_steps + probe_steps, "latency_s": latency_s,
                       "denoising_s": denoising_s,
                       "hit_e2e_s": latency_s,
                       "hit_denoising_s": denoising_s,
                       "status": "ok",
                       "target_cache_persisted": False,
                       "output_path": str(video), "video_path": str(video), "quality_input_path": str(video),
                       "extra": {
                           "native_summary": str(compact_summary),
                           "retrieval": retrieval_metadata,
                           "cache_prepare": cache_prepare,
                           "denoising_timing": {
                               "definition": "cuda_synchronized_dit_forward",
                               "target_dit_forward_s": target_dit_forward_s,
                               "probe_dit_forward_s": probe_denoising_s,
                               "total_dit_forward_s": denoising_s,
                               "target_steps": target_steps,
                               "probe_steps": probe_steps,
                           },
                       }}
        append_result_and_record(output, payload, row, args.method)
    if resident_collector is not None:
        resident_collector.close()
    if native_worker is not None:
        native_worker.close()
    return [json.loads(line) for line in (output / "requests.jsonl").read_text(encoding="utf-8").splitlines() if line]


def run_quality_evaluation(output: Path, gpu: int, dry_run: bool) -> dict[str, Any]:
    """Run quality evaluation and return summary statistics."""
    results_path = output / "results.jsonl"
    if not results_path.exists():
        return {"status": "no_results", "clip_t": None, "vbench_q": None}

    rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line]
    ok_rows = [row for row in rows if row.get("status") == "ok" and row.get("video_path")]

    if not ok_rows:
        return {"status": "no_videos", "clip_t": None, "vbench_q": None}

    quality_script = ROOT / "scripts" / "benchmark" / "quality.py"
    quality_env = os.environ.get("VCB_QUALITY_ENV", "vbench")

    try:
        cmd = [
            "conda", "run", "--no-capture-output", "-n", quality_env,
            "python", str(quality_script), str(output),
            "--dimension", "subject_consistency", "background_consistency", "motion_smoothness",
            "dynamic_degree", "aesthetic_quality", "imaging_quality"
        ]
        print(f"Running quality evaluation: {' '.join(cmd)}")
        # quality.py -> quality_runner.py default to cuda:0, but --gpu names the
        # physical device the method itself ran on.  Without this remap the
        # quality stage always lands on physical GPU 0, which on a shared machine
        # is routinely occupied by another tenant's job and OOMs the CLIP image
        # encoding (see docs/WAN21_AGGREGATOR_AND_SOURCE_OPTIMIZATION_PLAN.md).
        quality_env_vars = dict(os.environ)
        quality_env_vars["CUDA_VISIBLE_DEVICES"] = str(gpu)
        subprocess.run(cmd, check=True, cwd=ROOT, env=quality_env_vars)

        # Read updated results
        rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line]
        ok_rows = [row for row in rows if row.get("status") == "ok" and row.get("video_path")]

        clip_scores = [row["clip_t"] for row in ok_rows if "clip_t" in row]
        vbench_scores = [row["vbench_q"] for row in ok_rows if "vbench_q" in row]

        return {
            "status": "completed",
            "clip_t": sum(clip_scores) / len(clip_scores) if clip_scores else None,
            "vbench_q": sum(vbench_scores) / len(vbench_scores) if vbench_scores else None,
            "evaluated_count": len(ok_rows),
        }
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        # The run still writes results.jsonl and reports "completed"; make the
        # degraded quality stage loud here so the aggregate table's blank cells
        # are traceable to a real failure instead of looking like "not measured".
        print(f"Quality evaluation failed: {e}", file=sys.stderr, flush=True)
        print(
            f"WARNING: {output.name} continued without CLIP-T / VBench-Q; "
            f"results.jsonl will have empty quality columns and the aggregator will refuse it "
            f"unless run with --allow-incomplete-quality.",
            file=sys.stderr, flush=True,
        )
        return {"status": "failed", "error": str(e), "clip_t": None, "vbench_q": None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--model", choices=("wan21", "wan22", "cogvideox15", "ltxvideo"), default="wan22")
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--cache-view", type=Path, default=DEFAULT_VIEW)
    parser.add_argument("--replay-root", type=Path,
                        help="CogVideo replay directory produced by build_cogvideo_hit_replay.py.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-dir", type=Path, help="Write this worker's results directly here instead of <output-root>/<method>.")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--arrival-start", type=int, default=0, help="Inclusive global CSV arrival index for this worker.")
    parser.add_argument("--arrival-stop", type=int, help="Exclusive global CSV arrival index for this worker.")
    parser.add_argument("--request-indices", help="Comma-separated source row indices for a focused smoke run.")
    parser.add_argument(
        "--smoke-force-outcomes",
        action="store_true",
        help=(
            "Only for the two-row --request-indices 0,1 smoke: force request 0 to miss and request 1 to hit "
            "the fixed cache-view source. TeaCache remains target-only."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260218)
    parser.add_argument(
        "--early-signature-disabled",
        action="store_true",
        help="Use 1e6 for all four Early Signature gates, matching the disabled canonical run.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Replace this method's output directory before running.")
    parser.add_argument(
        "--retain-native-artifacts",
        action="store_true",
        help="Keep native request and target-probe artifacts for a focused diagnostic run.",
    )
    parser.add_argument("--allow-gpu-remap", action="store_true", help="Permit a temporary GPU other than the plan's 3..7 assignment.")
    parser.add_argument("--skip-quality", action="store_true", help="Skip quality evaluation after experiment.")
    args = parser.parse_args()
    if args.gpu < 0 or args.gpu >= GPU_COUNT:
        raise SystemExit(f"--gpu must be in [0, {GPU_COUNT - 1}]")
    if args.gpu not in range(3, 8) and not args.allow_gpu_remap:
        raise SystemExit("--gpu outside the plan's 3..7 range requires --allow-gpu-remap")
    assigned_gpu = (
        WAN21_GPU_BY_METHOD if args.model == "wan21"
        else LTX_GPU_BY_METHOD if args.model == "ltxvideo"
        else COGVIDEOX_GPU_BY_METHOD if args.model == "cogvideox15"
        else GPU_BY_METHOD
    )[args.method]
    if args.gpu != assigned_gpu and not args.allow_gpu_remap:
        raise SystemExit(f"{args.method} on {args.model} is assigned to GPU {assigned_gpu}, got {args.gpu}")
    args.input = args.input.expanduser().resolve()
    args.cache_view = args.cache_view.expanduser().resolve()
    args.replay_root = args.replay_root.expanduser().resolve() if args.replay_root else None
    args.output_root = args.output_root.expanduser().resolve()
    args.run_dir = args.run_dir.expanduser().resolve() if args.run_dir else None
    if args.limit < 1 or args.limit > 300:
        raise ValueError("--limit must be in [1, 300]")
    args.request_indices = ([int(item) for item in args.request_indices.split(",") if item.strip()]
                            if args.request_indices else None)
    args.early_signature_thresholds = dict(
        DISABLED_EARLY_SIGNATURE_THRESHOLDS if args.early_signature_disabled else EARLY_SIGNATURE_THRESHOLDS
    )
    if args.request_indices and any(index < 0 or index >= args.limit for index in args.request_indices):
        raise ValueError("--request-indices must refer to rows below --limit")
    selected_count = len(args.request_indices) if args.request_indices else args.limit
    args.arrival_stop = selected_count if args.arrival_stop is None else args.arrival_stop
    if not 0 <= args.arrival_start < args.arrival_stop <= args.limit:
        raise ValueError(f"arrival range must satisfy 0 <= --arrival-start < --arrival-stop <= --limit ({args.limit})")
    if args.check_only:
        print(json.dumps(check_only(args), indent=2, ensure_ascii=False))
        return 0
    rows = read_rows(args.input, args.limit, arrival_start=args.arrival_start, arrival_stop=args.arrival_stop,
                     request_indices=args.request_indices)
    view_rows = load_view(args.cache_view)
    validate_forced_smoke(args, rows, view_rows)
    # LTX-Video's Carnator source builds its own Early Signature inside the
    # request, so unlike Wan2.1 it needs nothing from the cache view here.
    if args.method == "carnator_boundary_b5" and args.model not in {"cogvideox15", "ltxvideo"}:
        validate_carnator_signature_view(args.cache_view, view_rows)
    output = args.run_dir or args.output_root / args.method
    if output.exists() and args.overwrite and not args.resume:
        shutil.rmtree(output)
    elif output.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite existing run: {output}; use --overwrite or --resume")
    output.mkdir(parents=True, exist_ok=True)
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"
    resolved = resolved_config(
        f"cuda:{args.gpu}",
        args.dry_run,
        model_name=args.model,
        early_signature_thresholds=getattr(args, "early_signature_thresholds", None),
    )
    (output / "run_config.json").write_text(json.dumps({"method": args.method, "gpu": args.gpu, "seed": args.seed, "limit": args.limit,
                                                          "arrival_start": args.arrival_start, "arrival_stop": args.arrival_stop,
                                                          "request_indices": args.request_indices,
                                                          "forced_smoke": forced_smoke_config(args),
                                                          "early_signature_disabled": args.early_signature_disabled,
                                                          "early_signature_thresholds": args.early_signature_thresholds,
                                                          "input": str(args.input), "cache_view": str(args.cache_view),
                                                          "input_sha256": sha256_file(args.input),
                                                          "view_manifest_sha256": sha256_file(args.cache_view / "view_manifest.json"),
                                                          "persist_target_cache": False,
                                                          "request_error_policy": {
                                                              "mode": "record_and_continue",
                                                              "terminal_statuses": ["ok", "cache_miss", "error"],
                                                              "quality_statuses": ["ok"],
                                                          },
                                                          "warmup": {"enabled": not args.dry_run, "timed": False,
                                                                     "source_prompt": WARMUP_SOURCE_PROMPT,
                                                                     "target_prompt": WARMUP_TARGET_PROMPT},
                                                          "native_worker_policy": "one_resident_worker_per_gpu" if args.method in {"chorus2", "carnator_boundary_b5"} else None,
                                                          "dense_c0": {"version": DENSE_C0_VERSION, "steps": 50},
                                                          "denoising_timing": "cuda_synchronized_dit_forward; Carnator adds B5 probe",
                                                          "native_artifact_policy": (
                                                              "retain_for_diagnostic" if args.retain_native_artifacts
                                                              else "top_level_video_and_records_only"
                                                          ),
                                                          # Overwritten below for wan21 and ltxvideo, whose
                                                          # runs write their own model-specific manifest.
                                                          "target_probe_enabled": args.method == "carnator_boundary_b5",
                                                          "target_probe_completed_step": 5 if args.method == "carnator_boundary_b5" else None,
                                                          "target_probe_seed": args.seed if args.method == "carnator_boundary_b5" else None,
                                                          "git_revision": revision,
                                                          "config_fingerprint": hashlib.sha256(json.dumps(resolved, sort_keys=True, default=str).encode()).hexdigest(),
                                                          "resolved_config": resolved}, indent=2) + "\n", encoding="utf-8")
    pending_rows, resumed_count = pending_rows_for_resume(output, rows, resume=args.resume)
    if pending_rows:
        if args.model == "wan21":
            # Every Wan2.1 method replays the legacy decisions through the
            # dedicated runner; there is no generic/native split here.
            wan21_run(args, pending_rows, args.cache_view, output)
        elif args.model == "ltxvideo":
            # Same shape as Wan2.1: one replay runner for all five baselines.
            ltxvideo_run(args, pending_rows, args.cache_view, output)
        elif args.model == "cogvideox15":
            # Same shape again: the CogVideoX runner is the Wan2.1 replay
            # runner plus its cache manifest check.
            cogvideo_run(args, pending_rows, args.cache_view, output)
        elif args.method in {"nirvana_vid", "teacache", "chorus"}:
            generic_run(args, pending_rows, args.cache_view, output)
        else:
            native_run(args, pending_rows, args.cache_view, output)
    # Keep the generic quality.py contract alongside the experiment-specific
    # requests.jsonl name used by the VidProM aggregation scripts.
    shutil.copyfile(output / "requests.jsonl", output / "results.jsonl")
    completed = [json.loads(line).get("status") for line in (output / "requests.jsonl").read_text(encoding="utf-8").splitlines()]
    terminal = sum(status in {"ok", "cache_miss", "error"} for status in completed)
    if terminal != len(rows):
        raise RuntimeError(f"expected {len(rows)} terminal requests, found {terminal}")
    ok = sum(status == "ok" for status in completed)
    misses = sum(status == "cache_miss" for status in completed)
    errors = sum(status == "error" for status in completed)

    quality_result = {"clip_t": None, "vbench_q": None}
    if not args.skip_quality and ok > 0:
        quality_result = run_quality_evaluation(output, args.gpu, args.dry_run)

    print(json.dumps({"status": "completed_with_errors" if errors else "completed", "method": args.method, "gpu": args.gpu,
                      "requests": len(rows), "ok": ok, "cache_miss": misses, "errors": errors,
                      "resumed": resumed_count, "results": str(output / "requests.jsonl"),
                      "clip_t": quality_result.get("clip_t"), "vbench_q": quality_result.get("vbench_q")}, ensure_ascii=False))
    return 2 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
