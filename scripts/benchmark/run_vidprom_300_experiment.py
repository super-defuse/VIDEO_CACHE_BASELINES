#!/usr/bin/env python3
"""Run one VidProM 300-request baseline against the immutable 923-entry view.

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
from core import Request, Runner
from core.cache import CacheEntry
from core.registry import ensure_supported, make_method
from core.retrieval import VectorRecord

METHODS = ("nirvana_vid", "teacache", "chorus", "chorus2", "carnator_boundary_b5")
GPU_BY_METHOD = {"nirvana_vid": 3, "teacache": 4, "chorus": 5, "chorus2": 6, "carnator_boundary_b5": 7}
GPU_COUNT = 8
EARLY_SIGNATURE_THRESHOLDS = {
    # Frozen from the forced-pair calibration: 84/93 comparable pairs pass
    # the gate, with a one-sided 95% CP quality lower bound of 0.8943.
    "max_context_color_distance": 2.2635538336347514,
    "max_log_area_ratio": 2.7357336654774373,
    "max_shape_distance": 0.924812030075188,
    "max_changed_color_distance": 2.540308037689532,
}
FUSED_MBASE_POLICY_PREFIX = "native-self-cross-fusion-mbase|"
DEFAULT_VIEW = ROOT / "artifacts" / "shared_cross_request_cache" / "vidprom_cluster_02_2170_first1000" / "cached_requests_923"
DEFAULT_INPUT = Path("/home/tangxuebin/datasets/1000cache_300input/cluster_02_2170_random300_input.csv")
DEFAULT_OUTPUT = ROOT / "artifacts" / "vidprom_1000cache_300input" / "results"


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


def resolved_config(device: str, dry_run: bool) -> dict[str, Any]:
    model = {"model": "synthetic", "dry_run": True, "seed": 20260218} if dry_run else load_yaml(ROOT / "configs" / "models" / "wan22.yaml")
    methods = {} if dry_run else load_yaml(ROOT / "configs" / "methods" / "default.yaml")
    config = {**model, **methods, "dry_run": dry_run, "device": device, "cache_root": "cache"}
    for key in ("model_path", "wan_source"):
        value = config.get(key)
        if value and not Path(value).is_absolute():
            config[key] = str((ROOT / value).resolve())
    for section in ("nirvana_vid", "chai", "chorus", "carnator"):
        values = config.get("retrieval", {}).get(section, {})
        if section == "carnator":
            values.update(EARLY_SIGNATURE_THRESHOLDS)
        checkpoint = values.get("longclip_checkpoint")
        if checkpoint and not Path(checkpoint).is_absolute():
            values["longclip_checkpoint"] = str((ROOT / checkpoint).resolve())
        if values:
            values["device"] = device
    return config


def read_rows(path: Path, limit: int, *, arrival_start: int = 0, arrival_stop: int | None = None) -> list[dict[str, Any]]:
    with path.expanduser().open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    required = {"uuid", "prompt"}
    if not rows or required - set(rows[0]):
        raise ValueError(f"input CSV must contain {sorted(required)}: {path}")
    selected = rows[:limit]
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
        result.append({"arrival_index": arrival_index, "request_index": arrival_index, "uuid": uuid, "prompt": prompt})
    if not result:
        raise ValueError("input CSV is empty")
    return result


def load_view(view: Path) -> list[dict[str, Any]]:
    manifest = view / "view_manifest.json"
    index = view / "index.jsonl"
    if not manifest.is_file() or not index.is_file():
        raise FileNotFoundError(f"cache view is incomplete: {view}")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if int(payload.get("selected_requests", -1)) != 923:
        raise ValueError(f"expected 923 source entries, got {payload.get('selected_requests')}")
    rows = [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != 923 or len({row.get("uuid") for row in rows}) != 923:
        raise ValueError("cache view index must contain 923 unique UUIDs")
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


def validate_carnator_signature_view(view: Path, rows: list[dict[str, Any]]) -> None:
    invalid: list[str] = []
    for row in rows:
        cache_dir = view / "entries" / str(row["uuid"]) / "carnator" / "cache" / "source_dense"
        signature = cache_dir / "online_color_signature.json"
        if not has_fused_mbase_signature(signature) or not (cache_dir / "fusion_masks" / "masks.pt").is_file():
            invalid.append(str(row["uuid"]))
            if len(invalid) == 5:
                break
    if invalid:
        raise ValueError(f"Carnator requires fused self/cross Mbase signatures; invalid source UUIDs: {invalid}")


def check_only(args: argparse.Namespace) -> dict[str, Any]:
    if args.gpu not in range(3, 8):
        raise ValueError("--gpu must be one of 3,4,5,6,7")
    view_rows = load_view(args.cache_view.expanduser().resolve())
    if args.method == "carnator_boundary_b5":
        validate_carnator_signature_view(args.cache_view.expanduser().resolve(), view_rows)
    config = resolved_config(f"cuda:{args.gpu}", args.dry_run)
    if not args.dry_run:
        # Carnator's native TI2V runner is intentionally separate from the
        # generic adapter registry; Chorus2 has the same native split.
        if args.method != "carnator_boundary_b5":
            ensure_supported("wan22", "chorus" if args.method == "chorus2" else args.method)
        model_path = Path(config["model_path"])
        if not model_path.is_dir():
            raise FileNotFoundError(f"Wan checkpoint is missing: {model_path}")
    return {
        "status": "ready",
        "method": args.method,
        "gpu": args.gpu,
        "source_entries": len(view_rows),
        "input": str(args.input.expanduser().resolve()),
        "input_sha256": sha256_file(args.input.expanduser().resolve()),
        "view_manifest_sha256": sha256_file((args.cache_view / "view_manifest.json").resolve()),
        "config_fingerprint": hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest(),
    }


def source_path(view: Path, row: dict[str, Any], method: str) -> Path:
    root = view / "entries" / str(row["uuid"])
    source = root / "carnator" / "cache" / "source_dense"
    if method == "nirvana_vid":
        path = source / "latents_step_009.pt"
    else:
        path = source
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
        entries[uuid] = CacheEntry(uuid, int(row.get("request_index", 0)), prompt, str(path), 10, 1, time.time(), metadata={"source_view": True})
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


def generic_run(args: argparse.Namespace, rows: list[dict[str, Any]], view: Path, output: Path) -> list[dict[str, Any]]:
    method_name = args.method
    config = resolved_config(f"cuda:{args.gpu}", args.dry_run)
    adapter = make_adapter("synthetic" if args.dry_run else "wan22", config)
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
    )
    preload_view(runner, view, view_rows_for_preload(view), method_name)
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
        payload.update({
            "arrival_index": row["arrival_index"],
            "video_path": payload.get("output_path"),
            # TeaCache is target-only; protocol reports no cross-request
            # retrieval even though the common timer has measurable overhead.
            "retrieval_s": 0.0 if method_name == "teacache" else timing.get("cache_lookup_s", 0.0),
            # Chorus generation_s includes its external LLM/SAM calls; keep
            # those calls in External and report only native denoising here.
            "denoising_s": max(0.0, timing.get("generation_s", payload.get("latency_s", 0.0)) - external_s),
            "external_s": external_s,
            "hit": bool(payload.get("cache_match", {}).get("hit")),
            "target_cache_persisted": False,
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
        "hit": bool(cache_match.get("hit", payload.get("hit", False))),
        "hit_object": cache_match.get("source_uuid", payload.get("source_uuid")),
        "similarity": similarity,
        "retrieval_kind": cache_match.get("kind") or metadata.get("retriever"),
        "external_s": float(payload.get("external_s", 0.0)),
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


def prompt_matcher(view: Path, config: dict[str, Any], dry_run: bool, method_name: str):
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
        **EARLY_SIGNATURE_THRESHOLDS,
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
        entry = CacheEntry(
            row["uuid"], int(row.get("request_index", 0)), row["prompt"], "", 10, 1, 0.0
        )
        nouns = carnator.noun_extractor.for_entry(entry)
        cache_dir = view / "entries" / row["uuid"] / "carnator" / "cache" / "source_dense"
        signature_path = cache_dir / "online_color_signature.json"
        metadata = {
            "source_request_index": int(row.get("request_index", 0)),
            "trf_nouns": list(nouns),
            "cache_ready": has_fused_mbase_signature(signature_path) and (cache_dir / "fusion_masks" / "masks.pt").is_file(),
            "signature": {"signature_path": str(signature_path)} if has_fused_mbase_signature(signature_path) else None,
        }
        carnator_records.append(VectorRecord(row["uuid"], row["prompt"], vector, metadata, int(row.get("request_index", 0))))
    return retriever, records, source_by_uuid, (carnator, carnator_records)


def write_fake_dense_manifest(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"mode": "T2V", "steps": 50, "output": str(path.with_name("target_dense.mp4")),
                                "timing": {"end_to_end_seconds": 1.0}}) + "\n", encoding="utf-8")


def make_native_target_collector(args: argparse.Namespace, config: dict[str, Any], artifact_root: Path, probe_seed: int):
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
    config = resolved_config(f"cuda:{args.gpu}", args.dry_run)
    matcher = None if args.dry_run else prompt_matcher(view, config, False, args.method)
    output.mkdir(parents=True, exist_ok=True)
    for row in rows:
        arrival = row["arrival_index"]
        target = Request(row["request_index"], row["uuid"], row["prompt"], args.seed + row["request_index"])
        if args.dry_run:
            video = output / "videos" / f"{arrival:04d}_{row['uuid']}.mp4"
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(b"CHAI-DRY-RUN:vidprom\n" + row["prompt"].encode())
            payload = {"arrival_index": arrival, "request_index": target.request_index, "uuid": target.uuid, "prompt": target.prompt,
                       "model": "synthetic", "method": args.method, "seed": target.seed, "cache_match": {"hit": False},
                       "hit": False, "executed_steps": 50, "latency_s": 0.0, "retrieval_s": 0.0, "denoising_s": 0.0,
                       "status": "ok", "target_cache_persisted": False, "output_path": str(video),
                       "video_path": str(video), "quality_input_path": str(video)}
        else:
            retriever, records, source_by_uuid, carnator_matcher = matcher
            request_started = time.perf_counter()
            retrieval_started = time.perf_counter()
            match = retriever.search(target, records)
            carnator_match = None
            if carnator_matcher is not None:
                carnator_retriever, carnator_records = carnator_matcher
                carnator_match = carnator_retriever.search(target, carnator_records)
            retrieval_s = time.perf_counter() - retrieval_started
            req_root = output / "native" / f"request-{arrival:04d}"
            req_root.mkdir(parents=True, exist_ok=True)
            probe_result = None
            probe_failure = None
            if carnator_matcher is not None and carnator_match is not None and carnator_match.kind == "probe_required":
                target_nouns = carnator_retriever.noun_extractor.for_request(target)
                probe_root = req_root / "target_probe"
                collector = make_native_target_collector(args, config, probe_root, args.seed)
                try:
                    try:
                        probe_result = collector.process(
                            target,
                            target_nouns,
                            evaluate_probe=lambda signature: carnator_retriever.search(
                                target, carnator_records, target_signature=signature
                            ),
                        )
                    except ValueError as exc:
                        if str(exc) != "fused Mbase is empty at every time token":
                            raise
                        probe_failure = "fused_mbase_empty_at_all_time_tokens"
                        shutil.rmtree(probe_root, ignore_errors=True)
                finally:
                    collector.close()
                if probe_result is not None:
                    carnator_match = probe_result.match or carnator_match
                    retrieval_s += probe_result.signature_gate_s
            carnator_dense_miss = (
                args.method == "carnator_boundary_b5"
                and (carnator_match is None or not carnator_match.hit or carnator_match.kind == "probe_required")
            )
            dense_miss = (not match.hit) or carnator_dense_miss
            execution_match = (
                carnator_match
                if carnator_match is not None and carnator_match.hit
                else match
            )
            if dense_miss:
                dense_seed = (
                    args.seed
                    if probe_result is not None and probe_result.source_signature is not None
                    else target.seed
                )
                command = [
                    sys.executable, "-m", "carnator.adapters.wan_ti2v_5b", "--no-env-setup",
                    "--model-dir", config["model_path"], "--device", f"cuda:{args.gpu}",
                    "--artifact-root", str(req_root / "artifacts"), "--results-root", str(req_root / "results"),
                    "--run-id", "run", "--target-prompt", target.prompt, "--seed", str(dense_seed),
                    "--target-dense-only", "--outer-attn-backend", "sparge-sage2", "--ffn-backend", "w8a8",
                    "--skip-w8a8-dense-diagnostic",
                ]
                if probe_result is not None and probe_result.source_signature is not None:
                    cache = probe_result.source_signature.get("cache")
                    if isinstance(cache, dict) and cache.get("latent_path"):
                        command.extend(("--target-resume-latent", str(cache["latent_path"]), "--target-resume-step", "10"))
                summary_path = req_root / "results" / "run" / "summary.json"
            else:
                if execution_match.source_uuid is None:
                    raise RuntimeError("reuse execution has no selected source UUID")
                source = source_by_uuid[execution_match.source_uuid]
                entry = view / "entries" / execution_match.source_uuid
                if args.method == "chorus2":
                    command = [sys.executable, "-m", "chorus2.wan_ti2v_5b", "--model-dir", config["model_path"], "--device", f"cuda:{args.gpu}",
                               "--artifact-root", str(req_root / "artifacts"), "--results-root", str(req_root / "results"), "--run-id", "run",
                               "--source-prompt", source["prompt"], "--target-prompt", target.prompt, "--seed", str(target.seed),
                               "--source-latent-cache", str(entry / "carnator" / "cache" / "source_dense"),
                               "--source-mask-cache", str(entry / "carnator" / "cache" / "source_dense" / "outer_block_masks")]
                    summary_path = req_root / "results" / "run" / "summary.json"
                else:
                    fake_manifest = req_root / "target_dense_manifest.json"
                    write_fake_dense_manifest(fake_manifest)
                    command = [sys.executable, "-m", "carnator.adapters.wan_ti2v_5b", "--no-env-setup", "--model-dir", config["model_path"],
                               "--device", f"cuda:{args.gpu}", "--artifact-root", str(req_root / "artifacts"), "--results-root", str(req_root / "results"),
                               "--run-id", "run", "--source-prompt", source["prompt"], "--target-prompt", target.prompt, "--seed", str(target.seed),
                               "--source-run-dir", str(entry / "carnator"), "--target-dense-manifest", str(fake_manifest),
                               "--outer-attn-backend", "sparge-sage2", "--ffn-backend", "w8a8", "--skip-w8a8-dense-diagnostic"]
                    summary_path = req_root / "results" / "run" / "summary.json"
            reusable_summary = False
            if summary_path.is_file():
                try:
                    existing = json.loads(summary_path.read_text(encoding="utf-8"))
                    reusable_summary = existing.get("status") == "completed"
                except (OSError, json.JSONDecodeError):
                    reusable_summary = False
            if not reusable_summary:
                try:
                    subprocess.run(command, cwd=ROOT, check=True)
                except subprocess.CalledProcessError:
                    # Older native Carnator builds can finish the target and
                    # write summary.json before failing while compacting an
                    # optional C0 manifest. Accept only that completed target.
                    if not summary_path.is_file():
                        raise
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            target_run = summary["runs"].get("target_dense" if dense_miss else "target", summary["runs"].get("target_latent_reuse", {}))
            native_video = Path(target_run["output"])
            video = output / "videos" / f"{arrival:04d}_{target.uuid}.mp4"
            video.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(native_video, video)
            compact_summary = compact_native_run(req_root, summary_path)
            retrieval_metadata = carnator_match.metadata if carnator_match is not None else match.metadata
            if carnator_match is not None:
                retrieval_metadata = dict(retrieval_metadata)
                retrieval_metadata["execution_candidate"] = {
                    "source_uuid": execution_match.source_uuid,
                    "score": execution_match.score,
                    "retriever": execution_match.metadata.get("retriever"),
                }
                if probe_result is not None:
                    retrieval_metadata["target_probe"] = {
                        "early_probe_s": probe_result.early_probe_s,
                        "signature_gate_s": probe_result.signature_gate_s,
                        "completed_steps": probe_result.completed_steps,
                        "probe_collected": probe_result.probe_collected,
                    }
                if probe_failure is not None:
                    retrieval_metadata["target_probe_failure"] = probe_failure
            if not match.hit:
                retrieval_metadata = dict(retrieval_metadata)
                retrieval_metadata["execution_reason"] = "no_stage_1_candidate"
            probe_denoising_s = 0.0
            if probe_result is not None:
                # A gate hit stops after step 5; a gate miss continues the
                # same target prefix through step 10 before dense resumes.
                probe_denoising_s = (
                    probe_result.early_probe_s
                    if probe_result.match is not None and probe_result.match.hit
                    else probe_result.source_cache_s
                )
            payload = {"arrival_index": arrival, "request_index": target.request_index, "uuid": target.uuid, "prompt": target.prompt,
                       "model": "Wan2.2", "method": args.method, "seed": target.seed,
                       "cache_match": {"hit": not dense_miss, "source_uuid": None if dense_miss else execution_match.source_uuid, "score": None if dense_miss else execution_match.score, "metadata": retrieval_metadata},
                       "hit": not dense_miss, "source_uuid": None if dense_miss else execution_match.source_uuid, "executed_steps": 50 if dense_miss else 40, "latency_s": time.perf_counter() - request_started,
                       "retrieval_s": retrieval_s, "external_s": 0.0,
                       "denoising_s": (target_run.get("denoising_seconds") or 0.0) + probe_denoising_s,
                       "status": "ok",
                       "target_cache_persisted": False,
                       "output_path": str(video), "video_path": str(video), "quality_input_path": str(video),
                       "extra": {"native_summary": str(compact_summary), "retrieval": retrieval_metadata}}
        append_result_and_record(output, payload, row, args.method)
    return [json.loads(line) for line in (output / "requests.jsonl").read_text(encoding="utf-8").splitlines() if line]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--cache-view", type=Path, default=DEFAULT_VIEW)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-dir", type=Path, help="Write this worker's results directly here instead of <output-root>/<method>.")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--arrival-start", type=int, default=0, help="Inclusive global CSV arrival index for this worker.")
    parser.add_argument("--arrival-stop", type=int, help="Exclusive global CSV arrival index for this worker.")
    parser.add_argument("--seed", type=int, default=20260218)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-gpu-remap", action="store_true", help="Permit a temporary GPU other than the plan's 3..7 assignment.")
    args = parser.parse_args()
    if args.gpu < 0 or args.gpu >= GPU_COUNT:
        raise SystemExit(f"--gpu must be in [0, {GPU_COUNT - 1}]")
    if args.gpu not in range(3, 8) and not args.allow_gpu_remap:
        raise SystemExit("--gpu outside the plan's 3..7 range requires --allow-gpu-remap")
    if args.gpu != GPU_BY_METHOD[args.method] and not args.allow_gpu_remap:
        raise SystemExit(f"{args.method} is assigned to GPU {GPU_BY_METHOD[args.method]}, got {args.gpu}")
    args.input = args.input.expanduser().resolve()
    args.cache_view = args.cache_view.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.run_dir = args.run_dir.expanduser().resolve() if args.run_dir else None
    if args.limit < 1 or args.limit > 300:
        raise ValueError("--limit must be in [1, 300]")
    args.arrival_stop = args.limit if args.arrival_stop is None else args.arrival_stop
    if not 0 <= args.arrival_start < args.arrival_stop <= args.limit:
        raise ValueError(f"arrival range must satisfy 0 <= --arrival-start < --arrival-stop <= --limit ({args.limit})")
    if args.check_only:
        print(json.dumps(check_only(args), indent=2, ensure_ascii=False))
        return 0
    rows = read_rows(args.input, args.limit, arrival_start=args.arrival_start, arrival_stop=args.arrival_stop)
    view_rows = load_view(args.cache_view)
    if args.method == "carnator_boundary_b5":
        validate_carnator_signature_view(args.cache_view, view_rows)
    output = args.run_dir or args.output_root / args.method
    if output.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite existing run: {output}; use --resume")
    output.mkdir(parents=True, exist_ok=True)
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"
    resolved = resolved_config(f"cuda:{args.gpu}", args.dry_run)
    (output / "run_config.json").write_text(json.dumps({"method": args.method, "gpu": args.gpu, "seed": args.seed, "limit": args.limit,
                                                          "arrival_start": args.arrival_start, "arrival_stop": args.arrival_stop,
                                                          "input": str(args.input), "cache_view": str(args.cache_view),
                                                          "input_sha256": sha256_file(args.input),
                                                          "view_manifest_sha256": sha256_file(args.cache_view / "view_manifest.json"),
                                                          "persist_target_cache": False,
                                                          "native_artifact_policy": "top_level_video_and_records_only",
                                                          "target_probe_enabled": args.method == "carnator_boundary_b5",
                                                          "target_probe_completed_step": 5 if args.method == "carnator_boundary_b5" else None,
                                                          "target_probe_seed": args.seed if args.method == "carnator_boundary_b5" else None,
                                                          "git_revision": revision,
                                                          "config_fingerprint": hashlib.sha256(json.dumps(resolved, sort_keys=True, default=str).encode()).hexdigest(),
                                                          "resolved_config": resolved}, indent=2) + "\n", encoding="utf-8")
    if args.method in {"nirvana_vid", "teacache", "chorus"}:
        generic_run(args, rows, args.cache_view, output)
    else:
        native_run(args, rows, args.cache_view, output)
    # Keep the generic quality.py contract alongside the experiment-specific
    # requests.jsonl name used by the VidProM aggregation scripts.
    shutil.copyfile(output / "requests.jsonl", output / "results.jsonl")
    ok = sum(1 for line in (output / "requests.jsonl").read_text(encoding="utf-8").splitlines() if json.loads(line).get("status") == "ok")
    print(json.dumps({"status": "completed", "method": args.method, "gpu": args.gpu, "requests": len(rows), "ok": ok,
                      "results": str(output / "requests.jsonl")}, ensure_ascii=False))
    return 0 if ok == len(rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
