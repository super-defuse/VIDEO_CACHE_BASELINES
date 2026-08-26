#!/usr/bin/env python3
"""Run one Wan target from a canonical shared source cache.

NIRVANA-VID and Chorus consume the same native Carnator trajectory directly;
no baseline-specific source payload is persisted. TeaCache remains target-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters import make_adapter
from core import Request
from core.registry import ensure_supported


METHODS = ("nirvana_vid", "chai", "teacache", "chorus")
SOURCE_PROMPT = "a dog is sleeping on a grassland, focus on the dog"
TARGET_PROMPT = "a cat is sleeping on a grassland, focus on the cat"


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--method", choices=METHODS, required=True)
    value.add_argument("--artifact-root", type=Path, required=True)
    value.add_argument("--results-root", type=Path, required=True)
    value.add_argument("--run-id", required=True)
    value.add_argument("--shared-cache-manifest", type=Path, required=True)
    value.add_argument("--source-prompt", default=SOURCE_PROMPT)
    value.add_argument("--target-prompt", default=TARGET_PROMPT)
    value.add_argument("--seed", type=int, default=20260218)
    value.add_argument("--device", default="cuda:0")
    value.add_argument("--chorus-gamma-k", type=float)
    value.add_argument("--chorus-gamma-o", type=float)
    return value


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def resolved_config(
    device: str, chorus_gamma_k: float | None = None, chorus_gamma_o: float | None = None
) -> dict[str, Any]:
    model = load_yaml(ROOT / "configs" / "models" / "wan22.yaml")
    methods = load_yaml(ROOT / "configs" / "methods" / "default.yaml")
    config = {**model, **methods, "dry_run": False, "device": device}
    for key in ("model_path", "wan_source"):
        value = config.get(key)
        if value and not Path(value).is_absolute():
            config[key] = str((ROOT / value).resolve())
    chai = config.get("retrieval", {}).get("chai", {})
    checkpoint = chai.get("longclip_checkpoint")
    if checkpoint and not Path(checkpoint).is_absolute():
        chai["longclip_checkpoint"] = str((ROOT / checkpoint).resolve())
    if chorus_gamma_k is not None:
        config["chorus"] = {
            **config.get("chorus", {}),
            "gamma_k": chorus_gamma_k,
            "gamma_o": chorus_gamma_o,
        }
    return config


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def child_path(root: Path, run_id: str) -> Path:
    root = root.expanduser().resolve()
    child = (root / run_id).resolve()
    if not child.is_relative_to(root):
        raise ValueError("--run-id must remain below its configured root")
    return child


def build_target_request(source_prompt: str, target_prompt: str, seed: int) -> Request:
    """Build a target request that retains the source prompt for Chorus diffing."""

    return Request(0, "target-000", target_prompt, seed, cache_prompt=source_prompt)


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_torch_payload(path: Path) -> Any:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def cache_input(
    method: str, manifest: dict[str, Any], adapter: Any
) -> tuple[Any | None, dict[str, Any]]:
    consumers = manifest.get("consumers")
    if not isinstance(consumers, dict):
        raise ValueError("shared cache manifest has no consumers section")
    if method == "teacache":
        return None, {"mode": "not_applicable", "reason": "TeaCache is target-only"}
    consumer = consumers.get(method)
    if not isinstance(consumer, dict):
        raise ValueError(f"shared cache manifest has no {method} consumer")
    if method == "nirvana_vid" and consumer.get("latent"):
        restart = int(consumer.get("reuse_start_step", 10))
        latent_path = Path(str(consumer["latent"])).expanduser()
        if not latent_path.is_file():
            raise FileNotFoundError(f"shared NIRVANA-VID latent is missing: {latent_path}")
        payload = {restart: load_torch_payload(latent_path)}
        payload_path = latent_path
    elif method == "chorus" and consumer.get("source_cache"):
        payload_path = Path(str(consumer["source_cache"])).expanduser()
        if not payload_path.is_dir():
            raise FileNotFoundError(f"shared Chorus source cache is missing: {payload_path}")
        source_video = Path(str(consumer.get("source_video", ""))).expanduser()
        if not source_video.is_file():
            raise FileNotFoundError(f"shared Chorus source video is missing: {source_video}")
        adapter.chorus_source_video_path = str(source_video)
        payload = adapter.load_latent(payload_path)
    else:
        payload_path = Path(str(consumer.get("payload", ""))).expanduser()
        if not payload_path.is_file():
            raise FileNotFoundError(f"shared cache payload is missing: {payload_path}")
        if method == "chorus":
            source_video = Path(str(consumer.get("source_video", ""))).expanduser()
            if not source_video.is_file():
                raise FileNotFoundError(f"shared Chorus source video is missing: {source_video}")
            adapter.chorus_source_video_path = str(source_video)
            payload = adapter.load_latent(payload_path)
        else:
            payload = load_torch_payload(payload_path)
    if method == "nirvana_vid":
        restart = int(consumer.get("reuse_start_step", 10))
        if not isinstance(payload, dict) or payload.get(restart, payload.get(str(restart))) is None:
            raise ValueError(f"NIRVANA-VID payload lacks completed step {restart}")
    if method == "chai":
        expected = {1, 2, 3, 4}
        if not isinstance(payload, dict) or {int(key) for key in payload} != expected:
            raise ValueError("CHAI payload must contain only source token steps 1, 2, 3, and 4")
    return payload, {"mode": "shared_dense", "payload": str(payload_path), **consumer}


def target_generation(method: str, adapter: Any, request: Request, payload: Any | None) -> Any:
    if method == "teacache":
        return adapter.generate(request, fast=False, cached_latent=None, capture=False, method=method)
    if method == "nirvana_vid":
        restart_step = int(adapter.config["nirvana_vid"]["reuse_start_step"])
        assert isinstance(payload, dict)
        payload = payload.get(restart_step, payload.get(str(restart_step)))
        if payload is None:
            raise KeyError(f"NIRVANA-VID payload lacks completed step {restart_step}")
    return adapter.generate(request, fast=True, cached_latent=payload, capture=False, method=method)


def hardware(device: str) -> dict[str, Any]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"device": device, "cuda_available": False}
        index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        return {
            "device": device,
            "visible_device": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_name": props.name,
            "total_memory_gib": round(props.total_memory / 1024**3, 3),
        }
    except Exception as exc:
        return {"device": device, "hardware_error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    args = parser().parse_args()
    ensure_supported("wan22", args.method)
    if (args.chorus_gamma_k is None) != (args.chorus_gamma_o is None):
        raise ValueError("--chorus-gamma-k and --chorus-gamma-o must be provided together")
    if args.method != "chorus" and args.chorus_gamma_k is not None:
        raise ValueError("Chorus gamma overrides are only valid with --method chorus")
    artifact_run = child_path(args.artifact_root, args.run_id)
    result_run = child_path(args.results_root, args.run_id)
    if artifact_run.exists() or result_run.exists():
        raise FileExistsError(f"refusing to overwrite run: {args.run_id}")

    shared_manifest_path = args.shared_cache_manifest.expanduser().resolve()
    shared_manifest = read_json(shared_manifest_path)
    source = shared_manifest.get("source", {})
    if source.get("prompt") != args.source_prompt:
        raise ValueError("shared cache source prompt does not match --source-prompt")
    if int(source.get("seed", -1)) != args.seed or int(source.get("steps", -1)) != 50:
        raise ValueError("shared cache must originate from the matching 50-step source run and seed")

    artifact_run.mkdir(parents=True)
    videos = artifact_run / "videos"
    videos.mkdir()
    result_run.mkdir(parents=True)
    request = build_target_request(args.source_prompt, args.target_prompt, args.seed)
    adapter = make_adapter(
        "wan22", resolved_config(args.device, args.chorus_gamma_k, args.chorus_gamma_o)
    )
    output = videos / "target.mp4"
    row: dict[str, Any]
    summary: dict[str, Any]
    try:
        # Mirror Runner: model construction happens before the target E2E timer.
        adapter.prepare()
        started = time.perf_counter()
        lookup_started = time.perf_counter()
        timed_manifest = read_json(shared_manifest_path)
        lookup_seconds = time.perf_counter() - lookup_started
        load_started = time.perf_counter()
        payload, input_info = cache_input(args.method, timed_manifest, adapter)
        load_seconds = time.perf_counter() - load_started
        generation_started = time.perf_counter()
        generation = target_generation(args.method, adapter, request, payload)
        generation_seconds = time.perf_counter() - generation_started
        save_started = time.perf_counter()
        adapter.save_video(generation.video, output)
        save_seconds = time.perf_counter() - save_started
        hash_started = time.perf_counter()
        output_hash = hash_file(output)
        hash_seconds = time.perf_counter() - hash_started
        elapsed = time.perf_counter() - started
        excluded = float(generation.extra.get("excluded_latency_s", 0.0))
        timing = {
            "cache_lookup_s": lookup_seconds,
            "cache_load_s": load_seconds,
            "generation_s": generation_seconds,
            "save_video_s": save_seconds,
            "hash_s": hash_seconds,
            "excluded_preprocess_s": excluded,
            "reported_latency_s": max(0.0, elapsed - excluded),
        }
        reference = Path(str(shared_manifest.get("source", {}).get("cache_dir", ""))).parent / "target_dense" / "manifest.json"
        dense_seconds = None
        if reference.is_file():
            dense_seconds = float(read_json(reference).get("timing", {}).get("end_to_end_seconds", 0.0)) or None
        row = {
            "request_index": request.request_index,
            "uuid": request.uuid,
            "prompt": request.prompt,
            "model": adapter.model_name,
            "method": args.method,
            "seed": request.seed,
            "cache_match": {"hit": args.method != "teacache", "kind": input_info["mode"]},
            "executed_steps": generation.executed_steps,
            "latency_s": timing["reported_latency_s"],
            "status": "ok",
            "output_path": str(output),
            "quality_input_path": str(output),
            "output_hash": output_hash,
            "cache_bytes": output.stat().st_size,
            "extra": {**generation.extra, "timing": timing, "shared_cache": input_info},
        }
        summary = {
            "method": args.method,
            "status": "completed",
            "config": {"source_prompt": args.source_prompt, "target_prompt": args.target_prompt, "seed": args.seed},
            "shared_cache_manifest": str(shared_manifest_path),
            "cache_input": input_info,
            "target": row,
            "speedup_reference": {
                "target_dense_manifest": str(reference),
                "target_dense_end_to_end_s": dense_seconds,
                "speedup": dense_seconds / row["latency_s"] if dense_seconds else None,
            },
            "hardware": hardware(args.device),
        }
    except Exception as exc:
        row = {
            "request_index": request.request_index,
            "uuid": request.uuid,
            "prompt": request.prompt,
            "model": "Wan2.2",
            "method": args.method,
            "seed": request.seed,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }
        summary = {"method": args.method, "status": "error", "target": row, "hardware": hardware(args.device)}
    finally:
        try:
            adapter.reset_hooks()
        except Exception:
            pass

    results = artifact_run / "results.jsonl"
    results.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    (artifact_run / "config.resolved.json").write_text(
        json.dumps({"method": args.method, "source_prompt": args.source_prompt, "target_prompt": args.target_prompt,
                    "seed": args.seed, "shared_cache_manifest": str(shared_manifest_path),
                    "chorus_gamma_k": args.chorus_gamma_k, "chorus_gamma_o": args.chorus_gamma_o}, indent=2) + "\n",
        encoding="utf-8",
    )
    (result_run / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (result_run / "report.md").write_text(
        "# Shared Dense Baseline\n\n"
        f"- Method: `{args.method}`\n"
        f"- Status: `{summary['status']}`\n"
        f"- Target video: `{output}`\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": summary["status"], "summary": str(result_run / "summary.json")}, ensure_ascii=False))
    if summary["status"] != "completed":
        raise RuntimeError(row["error"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
