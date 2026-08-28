#!/usr/bin/env python3
"""Calibrate Carnator Early Signature on CLIP-forced VidProM pairs.

This is intentionally separate from the ordinary 300-request benchmark.  It
first pairs every target with its highest-CLIP source from the read-only 923
entry view (excluding the target UUID itself), then optionally runs only the
target five-step signature probe plus forced latent reuse.  Quality scoring
and the existing offline four-gate sweep are exposed as subcommands.

No command is run when this module is imported.  The ``pair`` subcommand is
CPU/GPU embedding work; ``run`` is the only subcommand that starts Wan workers.
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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_INPUT = Path("/home/tangxuebin/datasets/1000cache_300input/cluster_02_2170_random300_input.csv")
DEFAULT_SOURCE_VIEW = ROOT / "artifacts" / "shared_cross_request_cache" / "vidprom_cluster_02_2170_first1000" / "cached_requests_923"
DEFAULT_RUN_ROOT = ROOT / "artifacts" / "vidprom_1000cache_300input" / "forced_early_signature_calibration"
DEFAULT_MODEL = Path(os.environ.get("CARNATOR_TI2V_5B_MODEL_DIR", "/home/tangxuebin/models/Wan2.2-TI2V-5B"))
DEFAULT_GPUS = "3,4,5,6,7"
FUSED_POLICY_PREFIX = "native-self-cross-fusion-mbase|"
DISTANCE_FIELDS = (
    "context_color_ciede2000", "log_area_ratio", "shape_one_minus_dice", "changed_color_ciede2000",
)
PAIR_FIELDS = (
    "pair_id", "request_index", "target_uuid", "target_prompt", "target_nouns",
    "source_uuid", "source_prompt", "cached_prompt", "source_nouns", "clip_similarity",
    "pair_source", "source_run_dir", "source_cache_dir", "source_signature_path",
)


@dataclass(frozen=True)
class PromptRow:
    request_index: int
    uuid: str
    prompt: str


@dataclass(frozen=True)
class SourceRow:
    uuid: str
    prompt: str
    source_run_dir: Path
    source_cache_dir: Path
    source_signature_path: Path
    source_nouns: tuple[str, ...]


@dataclass(frozen=True)
class ForcedPair:
    pair_id: str
    request_index: int
    target_uuid: str
    target_prompt: str
    target_nouns: tuple[str, ...]
    source_uuid: str
    source_prompt: str
    source_nouns: tuple[str, ...]
    clip_similarity: float
    pair_source: str
    source_run_dir: Path
    source_cache_dir: Path
    source_signature_path: Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_prompt_rows(path: Path, limit: int | None = None) -> list[PromptRow]:
    rows: list[PromptRow] = []
    seen: set[str] = set()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        if not {"uuid", "prompt"}.issubset(fields):
            raise ValueError(f"input CSV must contain uuid,prompt: {path}")
        for request_index, row in enumerate(reader):
            if limit is not None and len(rows) >= limit:
                break
            uuid = str(row.get("uuid", "")).strip()
            prompt = str(row.get("prompt", "")).strip()
            if not uuid or not prompt:
                raise ValueError(f"input row {request_index + 2} has an empty uuid or prompt")
            if uuid in seen:
                raise ValueError(f"input CSV contains duplicate UUID: {uuid}")
            seen.add(uuid)
            rows.append(PromptRow(request_index, uuid, prompt))
    if not rows:
        raise ValueError(f"input CSV has no rows: {path}")
    return rows


def _canonical_nouns(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted({" ".join(str(value).strip().lower().split()) for value in values if str(value).strip()}))


def _extract_nouns(prompts: Sequence[str], extractor_name: str) -> list[tuple[str, ...]]:
    if extractor_name == "none":
        return [tuple() for _ in prompts]
    from core.retrieval import LexicalTestEntityExtractor, SpacyEntityExtractor

    extractor = LexicalTestEntityExtractor() if extractor_name == "lexical" else SpacyEntityExtractor("en_core_web_sm")
    return [_canonical_nouns(extractor.extract(prompt)) for prompt in prompts]


def load_source_rows(view: Path, *, require_fused_signature: bool = True) -> list[SourceRow]:
    manifest = read_json(view / "view_manifest.json")
    selected = int(manifest.get("selected_requests", -1))
    index_rows = [json.loads(line) for line in (view / "index.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if selected != len(index_rows) or not index_rows:
        raise ValueError(f"source view count mismatch: manifest={selected}, index={len(index_rows)}")
    result: list[SourceRow] = []
    for row in index_rows:
        uuid, prompt = str(row.get("uuid", "")), str(row.get("prompt", ""))
        consumer = row.get("consumers", {}).get("carnator", {})
        source_run_dir = Path(str(consumer.get("source_run_dir", ""))).expanduser().resolve()
        if not source_run_dir.is_dir():
            entry = row.get("entry")
            source_run_dir = (view / str(entry) / "carnator").resolve()
        cache_dir = source_run_dir / "cache" / "source_dense"
        signature = cache_dir / "online_color_signature.json"
        required = (
            cache_dir / "manifest.json", cache_dir / "latents_step_009.pt",
            cache_dir / "fusion_masks" / "masks.pt", signature,
            source_run_dir.parent / "source.mp4",
        )
        if not all(path.is_file() for path in required):
            raise FileNotFoundError(f"source cache is incomplete for {uuid}: {cache_dir}")
        if require_fused_signature:
            payload = read_json(signature)
            policy = str(payload.get("compatibility", {}).get("mask_policy", ""))
            if not policy.startswith(FUSED_POLICY_PREFIX):
                raise ValueError(f"source signature is not fused self/cross Mbase: {signature}")
        nouns = _canonical_nouns(item.get("noun", "") for item in read_json(signature).get("objects", []))
        result.append(SourceRow(uuid, prompt, source_run_dir, cache_dir, signature, nouns))
    if len({row.uuid for row in result}) != len(result):
        raise ValueError("source view contains duplicate UUIDs")
    return result


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("embedding dimensions do not match")
    numerator = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = sum(float(a) * float(a) for a in left) ** 0.5
    right_norm = sum(float(b) * float(b) for b in right) ** 0.5
    if left_norm == 0.0 or right_norm == 0.0:
        raise ValueError("zero-norm embedding")
    return numerator / (left_norm * right_norm)


def select_nearest_pairs(
    targets: Sequence[PromptRow],
    sources: Sequence[SourceRow],
    target_vectors: Sequence[Sequence[float]],
    source_vectors: Sequence[Sequence[float]],
) -> list[ForcedPair]:
    if len(targets) != len(target_vectors) or len(sources) != len(source_vectors):
        raise ValueError("embedding count does not match prompt count")
    if not sources:
        raise ValueError("source pool is empty")
    try:
        import numpy as np

        target_matrix = np.asarray(target_vectors, dtype=np.float32)
        source_matrix = np.asarray(source_vectors, dtype=np.float32)
        if target_matrix.ndim != 2 or source_matrix.ndim != 2 or target_matrix.shape[1] != source_matrix.shape[1]:
            raise ValueError("embedding dimensions do not match")
        target_norm = np.linalg.norm(target_matrix, axis=1, keepdims=True)
        source_norm = np.linalg.norm(source_matrix, axis=1, keepdims=True)
        if np.any(target_norm == 0.0) or np.any(source_norm == 0.0):
            raise ValueError("zero-norm embedding")
        scores = (target_matrix / target_norm) @ (source_matrix / source_norm).T
    except ImportError:
        scores = None
    pairs: list[ForcedPair] = []
    for target_index, (target, target_vector) in enumerate(zip(targets, target_vectors)):
        if scores is not None:
            candidates = [(source, float(scores[target_index, source_index])) for source_index, source in enumerate(sources) if source.uuid != target.uuid]
        else:
            candidates = [(source, _cosine(target_vector, vector)) for source, vector in zip(sources, source_vectors) if source.uuid != target.uuid]
        if not candidates:
            raise ValueError(f"no non-self source candidate for target {target.uuid}")
        source, score = max(candidates, key=lambda item: item[1])
        pairs.append(ForcedPair(
            pair_id=f"clip_forced_{target.request_index + 1:04d}",
            request_index=target.request_index,
            target_uuid=target.uuid,
            target_prompt=target.prompt,
            target_nouns=tuple(),
            source_uuid=source.uuid,
            source_prompt=source.prompt,
            source_nouns=source.source_nouns,
            clip_similarity=float(score),
            pair_source="clip_nearest_excluding_self",
            source_run_dir=source.source_run_dir,
            source_cache_dir=source.source_cache_dir,
            source_signature_path=source.source_signature_path,
        ))
    return pairs


def write_pairs(path: Path, pairs: Sequence[ForcedPair]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PAIR_FIELDS), lineterminator="\n")
        writer.writeheader()
        for pair in pairs:
            writer.writerow({
                "pair_id": pair.pair_id,
                "request_index": pair.request_index,
                "target_uuid": pair.target_uuid,
                "target_prompt": pair.target_prompt,
                "target_nouns": json.dumps(pair.target_nouns, ensure_ascii=False),
                "source_uuid": pair.source_uuid,
                "source_prompt": pair.source_prompt,
                "cached_prompt": pair.source_prompt,
                "source_nouns": json.dumps(pair.source_nouns, ensure_ascii=False),
                "clip_similarity": f"{pair.clip_similarity:.12g}",
                "pair_source": pair.pair_source,
                "source_run_dir": str(pair.source_run_dir),
                "source_cache_dir": str(pair.source_cache_dir),
                "source_signature_path": str(pair.source_signature_path),
            })
    temporary.replace(path)


def read_pairs(path: Path) -> list[ForcedPair]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or set(rows[0]) != set(PAIR_FIELDS):
        raise ValueError(f"unexpected forced-pair CSV schema: {path}")
    result: list[ForcedPair] = []
    for row in rows:
        result.append(ForcedPair(
            pair_id=str(row["pair_id"]),
            request_index=int(row["request_index"]),
            target_uuid=str(row["target_uuid"]),
            target_prompt=str(row["target_prompt"]),
            target_nouns=tuple(json.loads(row["target_nouns"])),
            source_uuid=str(row["source_uuid"]),
            source_prompt=str(row["source_prompt"]),
            source_nouns=tuple(json.loads(row["source_nouns"])),
            clip_similarity=float(row["clip_similarity"]),
            pair_source=str(row["pair_source"]),
            source_run_dir=Path(row["source_run_dir"]).expanduser().resolve(),
            source_cache_dir=Path(row["source_cache_dir"]).expanduser().resolve(),
            source_signature_path=Path(row["source_signature_path"]).expanduser().resolve(),
        ))
    if len({pair.pair_id for pair in result}) != len(result):
        raise ValueError("forced-pair CSV contains duplicate pair IDs")
    return sorted(result, key=lambda pair: pair.request_index)


def pair_command(args: argparse.Namespace) -> int:
    input_path = args.input.expanduser().resolve()
    view = args.source_view.expanduser().resolve()
    run_root = args.run_root.expanduser().resolve()
    targets = read_prompt_rows(input_path, args.limit)
    sources = load_source_rows(view)
    from core.retrieval import TransformersClipEmbedder

    embedder = TransformersClipEmbedder(args.clip_model_id, args.clip_device)
    target_vectors = embedder.encode([row.prompt for row in targets])
    source_vectors = embedder.encode([row.prompt for row in sources])
    pairs = select_nearest_pairs(targets, sources, target_vectors, source_vectors)
    target_nouns = _extract_nouns([pair.target_prompt for pair in pairs], args.noun_extractor)
    if any(not nouns for nouns in target_nouns):
        empty = [pair.target_uuid for pair, nouns in zip(pairs, target_nouns) if not nouns]
        raise ValueError(f"noun extraction produced no nouns for target UUIDs: {empty[:5]}")
    pairs = [ForcedPair(**{**pair.__dict__, "target_nouns": nouns}) for pair, nouns in zip(pairs, target_nouns)]
    source_link_root = run_root / "sources"
    source_link_root.mkdir(parents=True, exist_ok=True)
    for pair in pairs:
        # Keep the quality/scanner layout local while pointing every heavy
        # source artifact back to the immutable shared entry.
        link = source_link_root / pair.pair_id
        link.mkdir(parents=True, exist_ok=True)
        cache_link = link / "cache" / "source_dense"
        cache_link.parent.mkdir(parents=True, exist_ok=True)
        if cache_link.exists() or cache_link.is_symlink():
            if not cache_link.is_symlink() or cache_link.resolve() != pair.source_cache_dir:
                raise FileExistsError(f"source cache link exists with a different target: {cache_link}")
        else:
            cache_link.symlink_to(pair.source_cache_dir, target_is_directory=True)
        source_video = pair.source_run_dir.parent / "source.mp4"
        video_link = link / "videos" / "source_dense.mp4"
        video_link.parent.mkdir(parents=True, exist_ok=True)
        if video_link.exists() or video_link.is_symlink():
            if not video_link.is_symlink() or video_link.resolve() != source_video:
                raise FileExistsError(f"source video link exists with a different target: {video_link}")
        elif source_video.is_file():
            video_link.symlink_to(source_video)
    pairs_path = run_root / "pairs.csv"
    write_pairs(pairs_path, pairs)
    write_json(run_root / "experiment_manifest.json", {
        "schema_version": "vidprom-forced-early-signature-calibration/v1",
        "input": str(input_path),
        "input_sha256": sha256_file(input_path),
        "source_view": str(view),
        "source_view_manifest_sha256": sha256_file(view / "view_manifest.json"),
        "pair_count": len(pairs),
        "pair_source": "CLIP nearest source excluding identical target UUID",
        "clip": {"model_id": args.clip_model_id, "device": args.clip_device},
        "generation": {"steps": 50, "mask_probe_step": 5, "reuse_start_step": 10, "signature": "fused-self-cross"},
        "quality_thresholds": {"q_min": None, "clip_t_min": None, "coverage_min": args.coverage_min, "confidence": args.confidence},
        "pairs_csv": {"path": str(pairs_path), "sha256": sha256_file(pairs_path)},
    })
    print(json.dumps({"status": "completed", "pairs": len(pairs), "pairs_csv": str(pairs_path)}, ensure_ascii=False))
    return 0


def native_args_for_pair(args: argparse.Namespace, pair: ForcedPair, gpu: int):
    from carnator.adapters import wan_ti2v_5b as native

    value = native.build_parser().parse_args([])
    value.no_env_setup = True
    value.model_dir = args.model_dir.resolve()
    value.device = f"cuda:{gpu}"
    value.artifact_root = (args.run_root / "targets").resolve()
    value.results_root = (args.run_root / "target_results").resolve()
    value.run_id = pair.pair_id
    value.source_prompt = pair.source_prompt
    value.target_prompt = pair.target_prompt
    value.source_positive = pair.source_nouns[0] if pair.source_nouns else "object"
    value.seed = args.seed
    value.steps = 50
    value.mask_probe_step = 5
    value.reuse_start_step = 10
    value.height, value.width, value.frames, value.fps = args.height, args.width, args.frames, args.fps
    value.scheduler, value.solver_order = args.scheduler, args.solver_order
    value.flow_shift, value.guidance_scale = args.flow_shift, args.guidance_scale
    value.model_dtype = args.model_dtype
    value.convert_model_dtype = True
    value.offload_model = False
    value.t5_cpu = False
    value.disable_progress = True
    value.outer_attn_backend = "dense"
    value.ffn_backend = "fp"
    value.latent_injection_mode = args.latent_injection_mode
    value.boundary_alpha_function = args.boundary_alpha_function
    value.capture_outer_block_masks = False
    value.source_run_dir = pair.source_run_dir
    value.source_only = False
    return value


def run_one_target(
    args: argparse.Namespace,
    pair: ForcedPair,
    gpu: int,
    *,
    runtime=None,
    native_args=None,
) -> dict[str, Any]:
    import torch
    from contextlib import ExitStack

    from carnator.adapters import wan_ti2v_5b as native
    from carnator.masks import load_mask_bundle
    from scripts.benchmark.run_carnator_early_signature_calibration import run_target_probe

    target_cache = args.run_root / "targets" / pair.pair_id / "cache" / "target_latent_reuse"
    target_video = args.run_root / "targets" / pair.pair_id / "videos" / "target_latent_reuse.mp4"
    status_path = args.run_root / "targets" / pair.pair_id / "calibration_target.json"
    if status_path.is_file() and read_json(status_path).get("status") == "completed":
        return read_json(status_path)
    owns_runtime = runtime is None
    if native_args is None:
        native_args = native_args_for_pair(args, pair, gpu)
    if owns_runtime:
        native.ensure_import_paths(native_args)
        native.validate_experiment_config(native_args)
        runtime = native.load_runtime(native_args)
    started = time.perf_counter()
    try:
        target_cache.mkdir(parents=True, exist_ok=True)
        signature_path = target_cache / "online_color_signature.json"
        if not signature_path.is_file():
            run_target_probe(runtime, native_args, pair, target_cache)
        _source, source_trajectory, source_mask_dir, _outer = native.source_assets_from_run(native_args, pair.source_run_dir)
        mask_bundle = load_mask_bundle(source_mask_dir, None)
        sparse_state = native.NativeSparseState(
            model_module=runtime.wan["model_module"],
            mask_bundle=mask_bundle,
            layers=native.parse_int_ranges(native_args.sparse_layers),
        )
        with ExitStack() as stack:
            stack.enter_context(native.NativeSparseBlockHooks(runtime.pipe.model, sparse_state))
            reuse = native.run_native_t2v_stage(
                runtime, native_args, label="target_latent_reuse", prompt=pair.target_prompt,
                out_video=target_video, cache_dir=target_cache,
                resume_latent_path=source_trajectory[native_args.reuse_start_step - 1],
                resume_start_step=native_args.reuse_start_step,
                source_trajectory=source_trajectory, mask_bundle=mask_bundle, sparse_state=sparse_state,
            )
        result = {
            "status": "completed", "pair_id": pair.pair_id, "gpu": gpu,
            "target_uuid": pair.target_uuid, "source_uuid": pair.source_uuid,
            "target_prompt": pair.target_prompt, "source_prompt": pair.source_prompt,
            "clip_similarity": pair.clip_similarity,
            "target_video": str(target_video), "target_cache": str(target_cache),
            "target_signature": str(signature_path), "source_run_dir": str(pair.source_run_dir),
            "reuse_manifest": reuse.get("manifest_path"),
            "wall_clock_seconds": time.perf_counter() - started,
        }
        write_json(status_path, result)
        return result
    finally:
        if owns_runtime:
            runtime.profiler.close()
            torch.cuda.empty_cache()


def parse_gpus(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result or len(result) != len(set(result)) or any(item < 0 for item in result):
        raise ValueError("--gpus must be a non-empty list of distinct non-negative IDs")
    return result


def run_gpu_worker(args: argparse.Namespace, pairs: list[ForcedPair], gpu: int) -> int:
    """Run one disjoint shard in one process with one long-lived Wan runtime."""
    import torch

    from carnator.adapters import wan_ti2v_5b as native

    pending = [
        pair for pair in pairs
        if not (
            (args.run_root / "targets" / pair.pair_id / "calibration_target.json").is_file()
            and read_json(args.run_root / "targets" / pair.pair_id / "calibration_target.json").get("status") == "completed"
        )
    ]
    if not pending:
        print(json.dumps({"gpu": gpu, "status": "already_completed", "pairs": len(pairs)}), flush=True)
        return 0
    first_args = native_args_for_pair(args, pending[0], gpu)
    native.ensure_import_paths(first_args)
    native.validate_experiment_config(first_args)
    runtime = native.load_runtime(first_args)
    failures: list[str] = []
    try:
        for pair in pairs:
            try:
                pair_args = native_args_for_pair(args, pair, gpu)
                result = run_one_target(args, pair, gpu, runtime=runtime, native_args=pair_args)
                print(json.dumps({"gpu": gpu, "pair_id": pair.pair_id, "status": result["status"]}), flush=True)
            except Exception as exc:
                failures.append(f"{pair.pair_id}: {type(exc).__name__}: {exc}")
                print(f"gpu={gpu} pair={pair.pair_id} failed: {exc}", file=sys.stderr, flush=True)
    finally:
        runtime.profiler.close()
        torch.cuda.empty_cache()
    if failures:
        raise RuntimeError("target worker failed: " + "; ".join(failures))
    return 0


def _worker_command(args: argparse.Namespace, gpu: int, pairs: list[ForcedPair]) -> list[str]:
    command = [
        sys.executable, str(Path(__file__).resolve()), "run",
        "--pairs", str(args.pairs), "--run-root", str(args.run_root),
        "--model-dir", str(args.model_dir), "--gpus", str(gpu),
        "--seed", str(args.seed), "--height", str(args.height), "--width", str(args.width),
        "--frames", str(args.frames), "--fps", str(args.fps), "--scheduler", args.scheduler,
        "--solver-order", str(args.solver_order), "--flow-shift", str(args.flow_shift),
        "--guidance-scale", str(args.guidance_scale), "--model-dtype", args.model_dtype,
        "--latent-injection-mode", args.latent_injection_mode,
        "--boundary-alpha-function", args.boundary_alpha_function,
        "--worker-gpu", str(gpu), "--worker-pair-ids", ",".join(pair.pair_id for pair in pairs),
    ]
    return command


def run_command(args: argparse.Namespace) -> int:
    pairs = read_pairs(args.pairs.expanduser().resolve())
    run_root = args.run_root.expanduser().resolve()
    if args.worker_gpu is not None:
        if not args.worker_pair_ids:
            raise ValueError("--worker-pair-ids is required with --worker-gpu")
        pair_map = {pair.pair_id: pair for pair in pairs}
        worker_ids = [item for item in args.worker_pair_ids.split(",") if item]
        missing = [item for item in worker_ids if item not in pair_map]
        if missing:
            raise ValueError(f"unknown worker pair IDs: {missing[:3]}")
        return run_gpu_worker(args, [pair_map[item] for item in worker_ids], args.worker_gpu)
    gpus = parse_gpus(args.gpus)
    run_root.mkdir(parents=True, exist_ok=True)
    shards = [pairs[index::len(gpus)] for index in range(len(gpus))]
    processes = []
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(ROOT), environment.get("PYTHONPATH", "")))
    for gpu, shard in zip(gpus, shards):
        if not shard:
            continue
        processes.append((gpu, subprocess.Popen(_worker_command(args, gpu, shard), cwd=ROOT, env=environment)))
    failures = []
    for gpu, process in processes:
        return_code = process.wait()
        if return_code:
            failures.append(f"gpu {gpu} exited with status {return_code}")
    if failures:
        raise RuntimeError("target workers failed: " + "; ".join(failures))
    return 0


def delegated_command(script_name: str, args: argparse.Namespace, extra: list[str]) -> int:
    command = [sys.executable, str(ROOT / "scripts" / "benchmark" / script_name), *extra]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(ROOT), environment.get("PYTHONPATH", "")))
    return subprocess.call(command, cwd=ROOT, env=environment)


def score_command(args: argparse.Namespace) -> int:
    return delegated_command("score_carnator_early_signature_calibration.py", args, [
        "--run-root", str(args.run_root.resolve()), "--pairs", str(args.pairs.resolve()),
        "--gpus", args.gpus, "--clip-weights", str(args.clip_weights), "--vbench-cache", str(args.vbench_cache),
    ])


def _cp_lower(k: int, n: int, confidence: float) -> float:
    if n <= 0:
        return 0.0
    if k <= 0:
        return 0.0
    from scipy.stats import beta

    return float(beta.ppf(1.0 - confidence, k, n - k + 1))


def _pair_gate_distances(run_root: Path, pair_id: str) -> tuple[dict[str, float] | None, str | None]:
    from carnator.early_signature import compare_early_signatures

    source_path = run_root / "sources" / pair_id / "cache" / "source_dense" / "online_color_signature.json"
    target_path = run_root / "targets" / pair_id / "cache" / "target_latent_reuse" / "online_color_signature.json"
    if not source_path.is_file() or not target_path.is_file():
        return None, "missing_signature"
    try:
        source_objects = {str(item["noun"]) for item in read_json(source_path).get("objects", [])}
        target_objects = {str(item["noun"]) for item in read_json(target_path).get("objects", [])}
        source_only = source_objects - target_objects
        target_only = target_objects - source_objects
        if len(source_only) > 1 or len(target_only) > 1:
            return None, "noun_difference_gt_one"
        kwargs: dict[str, Any] = {
            "max_context_color_distance": float("inf"),
            "max_log_area_ratio": float("inf"),
            "max_shape_distance": float("inf"),
            "max_changed_color_distance": float("inf"),
            "check_cache_artifacts": True,
        }
        if source_only or target_only:
            kwargs["query_changed_noun"] = next(iter(target_only))
            kwargs["candidate_changed_noun"] = next(iter(source_only))
        comparison = compare_early_signatures({"signature_path": str(target_path)}, {"signature_path": str(source_path)}, **kwargs)
        if not comparison["metadata_compatible"] or not comparison["cache_ready"] or not comparison["signature_compatible"]:
            return None, str(comparison.get("reason", "signature_incompatible"))
        distances = comparison["distances"]
        # Same-noun pairs have no changed object. Treat the three inapplicable
        # distances as zero so they do not artificially reject the pair.
        values = {field: float(distances.get(field) or 0.0) for field in DISTANCE_FIELDS}
        if any(not math.isfinite(value) for value in values.values()):
            return None, "non_finite_distance"
        return values, None
    except Exception as exc:
        return None, f"comparison_error:{type(exc).__name__}"


def _threshold_grid(values: list[float], minimum_accepted: int, max_values: int) -> list[float]:
    ordered = sorted(set(values))
    if not ordered or minimum_accepted < 1 or minimum_accepted > len(values):
        return []
    candidates = ordered[max(0, len(ordered) - (len(values) - minimum_accepted + 1)):]
    if len(candidates) <= max_values:
        return candidates
    indices = {round(index * (len(candidates) - 1) / (max_values - 1)) for index in range(max_values)}
    return [candidates[index] for index in sorted(indices)]


def _write_scan_report(run_root: Path, records: list[dict[str, Any]], sweep: list[dict[str, Any]], args: argparse.Namespace) -> None:
    quality_root = run_root / "quality"
    quality_root.mkdir(parents=True, exist_ok=True)
    suffix = "" if getattr(args, "coverage_denominator", "all") == "all" else "_comparable"
    with (quality_root / f"forced_threshold_sweep{suffix}.csv.tmp").open("w", newline="", encoding="utf-8") as handle:
        fields = ["max_context_color_distance", "max_log_area_ratio", "max_shape_distance", "max_changed_color_distance", "n", "k", "coverage", "success_rate", "one_sided_cp_lower", "quality_floor_pass"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sweep:
            writer.writerow({field: row[field] for field in fields})
    (quality_root / f"forced_threshold_sweep{suffix}.csv.tmp").replace(quality_root / f"forced_threshold_sweep{suffix}.csv")
    write_jsonl(quality_root / f"forced_threshold_sweep{suffix}_details.jsonl", sweep)
    valid = len(records)
    invalid = args.pair_count - valid if hasattr(args, "pair_count") else 0
    denominator = valid if getattr(args, "coverage_denominator", "all") == "comparable" else args.pair_count
    report = [
        "# Forced Early Signature threshold scan", "",
        f"- Quality floors: VBench-Q >= {args.q_min:.6f}; CLIP-T >= {args.clip_t_min:.6f}",
        f"- Required coverage: {args.coverage_min:.3f}; confidence: {args.confidence:.3f}",
        f"- Comparable pairs: {valid}; incomparable pairs: {invalid}",
        f"- Coverage denominator: {getattr(args, 'coverage_denominator', 'all')} ({denominator})",
        f"- Threshold tuples evaluated: {len(sweep)}", "",
        f"Pairs with noun-set difference greater than one are retained in `forced_scan_pairs{suffix}.jsonl` but excluded from four-distance gate fitting.",
    ]
    (quality_root / f"forced_scan_report{suffix}.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def scan_forced_pairs(args: argparse.Namespace) -> int:
    run_root = args.run_root
    pairs = read_pairs(args.pairs)
    quality_path = run_root / "quality" / "pair_results.jsonl"
    if not quality_path.is_file():
        raise FileNotFoundError(f"run score first: {quality_path}")
    quality_rows = [json.loads(line) for line in quality_path.read_text(encoding="utf-8").splitlines() if line]
    quality_by_pair = {str(row["pair_id"]): row for row in quality_rows}
    if set(quality_by_pair) != {pair.pair_id for pair in pairs}:
        raise RuntimeError("pair_results.jsonl does not cover exactly the forced pairs")
    records: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for pair in pairs:
        distances, reason = _pair_gate_distances(run_root, pair.pair_id)
        base = dict(quality_by_pair[pair.pair_id])
        base["quality_pass"] = {
            "vbench_q_pass": float(base["vbench_q_i"]) >= args.q_min,
            "clip_t_pass": float(base["clip_t"]) >= args.clip_t_min,
        }
        base["quality_pass"]["success"] = base["quality_pass"]["vbench_q_pass"] and base["quality_pass"]["clip_t_pass"]
        if distances is None:
            base["gate_status"] = "incomparable"
            base["gate_failure_reason"] = reason
            invalid.append(base)
        else:
            base["gate_distances"] = distances
            base["gate_status"] = "comparable"
            records.append(base)
    all_records = records + invalid
    suffix = "" if getattr(args, "coverage_denominator", "all") == "all" else "_comparable"
    write_jsonl(run_root / "quality" / f"forced_scan_pairs{suffix}.jsonl", all_records)
    denominator = len(records) if getattr(args, "coverage_denominator", "all") == "comparable" else len(pairs)
    minimum_accepted = math.ceil(denominator * args.coverage_min)
    threshold_lists = {
        field: _threshold_grid([row["gate_distances"][field] for row in records], minimum_accepted, args.max_threshold_values)
        for field in DISTANCE_FIELDS
    }
    sweep: list[dict[str, Any]] = []
    if all(threshold_lists.values()):
        for values in itertools.product(*(threshold_lists[field] for field in DISTANCE_FIELDS)):
            thresholds = dict(zip(DISTANCE_FIELDS, values))
            accepted = [row for row in records if all(row["gate_distances"][field] <= thresholds[field] for field in DISTANCE_FIELDS)]
            success = sum(bool(row["quality_pass"]["success"]) for row in accepted)
            n = len(accepted)
            coverage = n / denominator
            lower = _cp_lower(success, n, args.confidence)
            sweep.append({
                "max_context_color_distance": thresholds[DISTANCE_FIELDS[0]],
                "max_log_area_ratio": thresholds[DISTANCE_FIELDS[1]],
                "max_shape_distance": thresholds[DISTANCE_FIELDS[2]],
                "max_changed_color_distance": thresholds[DISTANCE_FIELDS[3]],
                "n": n, "k": success, "coverage": coverage,
                "success_rate": success / n if n else 0.0,
                "one_sided_cp_lower": lower,
                "quality_floor_pass": coverage >= args.coverage_min and lower >= args.selection_min_lower_bound,
                "accepted_pair_ids": [row["pair_id"] for row in accepted],
            })
    args.pair_count = len(pairs)
    _write_scan_report(run_root, records, sweep, args)
    write_json(run_root / "quality" / f"forced_scan_manifest{suffix}.json", {
        "pair_count": len(pairs), "comparable_pairs": len(records), "incomparable_pairs": len(invalid),
        "quality_floors": {"q_min": args.q_min, "clip_t_min": args.clip_t_min},
        "coverage_denominator": getattr(args, "coverage_denominator", "all"),
        "coverage_min": args.coverage_min, "confidence": args.confidence,
        "selection_min_lower_bound": args.selection_min_lower_bound,
        "max_threshold_values": args.max_threshold_values, "sweep_count": len(sweep),
    })
    print(json.dumps({"status": "completed", "pair_count": len(pairs), "comparable_pairs": len(records), "sweep_count": len(sweep)}))
    return 0


def scan_command(args: argparse.Namespace) -> int:
    manifest_path = args.run_root / "experiment_manifest.json"
    manifest = read_json(manifest_path)
    quality = dict(manifest.get("quality_thresholds", {}))
    quality.update({"q_min": args.q_min, "clip_t_min": args.clip_t_min, "coverage_min": args.coverage_min, "confidence": args.confidence, "coverage_denominator": args.coverage_denominator})
    manifest["quality_thresholds"] = quality
    write_json(manifest_path, manifest)
    return scan_forced_pairs(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    pair = sub.add_parser("pair", help="CLIP-pair each target with the nearest non-self source")
    pair.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    pair.add_argument("--source-view", type=Path, default=DEFAULT_SOURCE_VIEW)
    pair.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    pair.add_argument("--limit", type=int, default=300)
    pair.add_argument("--clip-model-id", default="openai/clip-vit-large-patch14")
    pair.add_argument("--clip-device", default="cuda:3")
    pair.add_argument("--noun-extractor", choices=("spacy", "lexical", "none"), default="spacy")
    pair.add_argument("--coverage-min", type=float, default=0.90)
    pair.add_argument("--confidence", type=float, default=0.95)

    run = sub.add_parser("run", help="run forced target probe and latent reuse on GPU workers")
    run.add_argument("--pairs", type=Path, default=DEFAULT_RUN_ROOT / "pairs.csv")
    run.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    run.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    run.add_argument("--gpus", default=DEFAULT_GPUS)
    run.add_argument("--seed", type=int, default=20260218)
    run.add_argument("--height", type=int, default=704)
    run.add_argument("--width", type=int, default=1280)
    run.add_argument("--frames", type=int, default=121)
    run.add_argument("--fps", type=int, default=24)
    run.add_argument("--scheduler", choices=("unipc", "dpm++"), default="unipc")
    run.add_argument("--solver-order", type=int, default=2)
    run.add_argument("--flow-shift", type=float, default=5.0)
    run.add_argument("--guidance-scale", type=float, default=5.0)
    run.add_argument("--model-dtype", choices=("fp16", "bf16"), default="fp16")
    run.add_argument("--latent-injection-mode", choices=("hard", "boundary-blend"), default="hard")
    run.add_argument("--boundary-alpha-function", choices=("smoothstep", "cosine"), default="smoothstep")
    run.add_argument("--worker-gpu", type=int, default=None, help=argparse.SUPPRESS)
    run.add_argument("--worker-pair-ids", default=None, help=argparse.SUPPRESS)

    score = sub.add_parser("score", help="delegate per-video CLIP-T/VBench-Q scoring")
    score.add_argument("--pairs", type=Path, default=DEFAULT_RUN_ROOT / "pairs.csv")
    score.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    score.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    score.add_argument("--clip-weights", type=Path, default=ROOT / "artifacts" / "vbench-cache" / "clip_model" / "ViT-B-32.pt")
    score.add_argument("--vbench-cache", type=Path, default=ROOT / "artifacts" / "vbench-cache")

    scan = sub.add_parser("scan", help="scan quality floors and the four Early Signature thresholds")
    scan.add_argument("--pairs", type=Path, default=DEFAULT_RUN_ROOT / "pairs.csv")
    scan.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    scan.add_argument("--q-min", type=float, required=True)
    scan.add_argument("--clip-t-min", type=float, required=True)
    scan.add_argument("--coverage-min", type=float, default=0.90)
    scan.add_argument("--confidence", type=float, default=0.95)
    scan.add_argument("--selection-min-lower-bound", type=float, default=0.80)
    scan.add_argument("--max-threshold-values", type=int, default=12, help="maximum candidate values per gate dimension")
    scan.add_argument("--coverage-denominator", choices=("all", "comparable"), default="all", help="coverage over all requests or only noun-comparable pairs")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "pair":
        if args.limit is not None and args.limit < 1:
            raise ValueError("--limit must be positive")
        if not 0 < args.coverage_min <= 1 or not 0 < args.confidence < 1:
            raise ValueError("coverage/confidence must be in (0,1]")
        return pair_command(args)
    if args.command == "run":
        args.pairs = args.pairs.expanduser().resolve()
        args.run_root = args.run_root.expanduser().resolve()
        args.model_dir = args.model_dir.expanduser().resolve()
        if not args.pairs.is_file():
            raise FileNotFoundError(args.pairs)
        if not args.model_dir.is_dir():
            raise FileNotFoundError(args.model_dir)
        return run_command(args)
    args.pairs = args.pairs.expanduser().resolve()
    args.run_root = args.run_root.expanduser().resolve()
    if args.command == "score":
        return score_command(args)
    if not 0 < args.coverage_min <= 1 or not 0 < args.confidence < 1 or not 0 <= args.selection_min_lower_bound <= 1 or args.max_threshold_values < 2:
        raise ValueError("invalid scan statistical parameters")
    return scan_command(args)


if __name__ == "__main__":
    raise SystemExit(main())
