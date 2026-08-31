#!/usr/bin/env python3
"""Generate fused-Mbase Carnator step-5 signatures for the source view.

Only the artifacts consumed by ``compare_early_signatures`` are written into
each existing source cache directory: ``online_color_signature.json``,
``mbase.safetensors`` and ``step5_tail_frame.png``.  The step-5 latent is
loaded from the immutable trajectory, used to compute its content hash, and
never written by this script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_VIEW = ROOT / "artifacts" / "vidprom_1000cache_300input_noun_match" / "source_view"
DEFAULT_MODEL = Path(os.environ.get("CARNATOR_TI2V_5B_MODEL_DIR", "/home/tangxuebin/models/Wan2.2-TI2V-5B"))
DEFAULT_OUTPUT = ROOT / "artifacts" / "vidprom_1000cache_300input_noun_match" / "early_signature_generation"
NOUN_EXTRACTOR = "spacy:en_core_web_sm"
FUSED_MBASE_POLICY_PREFIX = "native-self-cross-fusion-mbase|"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(view: Path) -> list[dict[str, Any]]:
    manifest = json.loads((view / "view_manifest.json").read_text(encoding="utf-8"))
    selected = int(manifest.get("selected_requests", -1))
    if selected <= 0:
        raise ValueError(f"invalid selected_requests={selected}")
    rows = [json.loads(line) for line in (view / "index.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != selected or len({row.get("uuid") for row in rows}) != selected:
        raise ValueError(f"view index must contain {selected} unique UUIDs")
    for row in rows:
        if not row.get("uuid") or not row.get("prompt"):
            raise ValueError(f"invalid source row: {row}")
    return rows


def native_args(model_dir: Path, device: str, *, seed: int, height: int, width: int, frames: int, fps: int):
    from carnator.adapters import wan_ti2v_5b as native

    args = native.build_parser().parse_args([])
    args.no_env_setup = True
    args.model_dir = model_dir.resolve()
    args.device = device
    args.seed = int(seed)
    args.height = int(height)
    args.width = int(width)
    args.frames = int(frames)
    args.fps = int(fps)
    args.steps = 50
    args.mask_probe_step = 5
    args.reuse_start_step = 10
    args.model_dtype = "fp16"
    args.convert_model_dtype = True
    args.offload_model = False
    args.t5_cpu = False
    args.disable_progress = True
    args.outer_attn_backend = "dense"
    args.ffn_backend = "fp"
    args.capture_outer_block_masks = False
    return args


def cache_metadata(cache_dir: Path, token_grid: tuple[int, int, int], uuid: str) -> dict[str, Any]:
    latent = cache_dir / "latents_step_009.pt"
    sparse = cache_dir / "fusion_masks" / "masks.pt"
    for path in (latent, sparse):
        if not path.is_file():
            raise FileNotFoundError(f"required source cache artifact is missing: {path}")
    return {
        "anchor_step": 10,
        "model": "Wan2.2-TI2V-5B",
        "latent_token_grid": list(token_grid),
        "latent_path": str(latent.resolve()),
        "latent_sha256": sha256_file(latent),
        "sparse_attention_path": str(sparse.resolve()),
        "sparse_attention_sha256": sha256_file(sparse),
        "source_uuid": uuid,
    }


def sidecar_complete(cache_dir: Path) -> bool:
    required = all(
        (cache_dir / name).is_file() and (cache_dir / name).stat().st_size > 0
        for name in ("online_color_signature.json", "mbase.safetensors", "step5_tail_frame.png")
    )
    if not required:
        return False
    try:
        signature = json.loads((cache_dir / "online_color_signature.json").read_text(encoding="utf-8"))
        policy = signature["compatibility"]["mask_policy"]
    except (KeyError, OSError, TypeError, json.JSONDecodeError):
        return False
    return isinstance(policy, str) and policy.startswith(FUSED_MBASE_POLICY_PREFIX)


def canonical_nouns(values: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Match early_signature's lower-case, whitespace-normalized noun keys."""

    return tuple(sorted({" ".join(str(value).strip().lower().split()) for value in values if str(value).strip()}))


def generate_one(runtime, args, row: dict[str, Any], source_root: Path, nouns: tuple[str, ...]) -> dict[str, Any]:
    import torch

    from carnator.adapters import wan_ti2v_5b as native
    from scripts.benchmark.run_carnator_early_signature_calibration import capture_early_signature

    uuid = str(row["uuid"])
    cache_dir = source_root / uuid / "carnator" / "cache" / "source_dense"
    if sidecar_complete(cache_dir):
        return {"uuid": uuid, "status": "skipped_existing", "nouns": list(nouns)}
    if not nouns:
        return {"uuid": uuid, "status": "skipped_no_nouns", "nouns": []}
    pre_step5 = cache_dir / "latents_step_003.pt"
    post_step5 = cache_dir / "latents_step_004.pt"
    if not pre_step5.is_file() or not post_step5.is_file():
        raise FileNotFoundError(f"step-5 trajectory is incomplete for {uuid}")
    attention_input = native.load_latent(pre_step5, runtime.device, torch.float32)
    step5_latent = native.load_latent(post_step5, runtime.device, torch.float32)
    token_grid = native.token_grid_for(step5_latent, runtime.pipe.patch_size)
    cache = cache_metadata(cache_dir, token_grid, uuid)
    signature_path = capture_early_signature(
        runtime,
        args,
        prompt=str(row["prompt"]),
        nouns=nouns,
        attention_input=attention_input,
        step5_latent=step5_latent,
        directory=cache_dir,
        cache=cache,
    )
    del attention_input, step5_latent
    torch.cuda.empty_cache()
    return {
        "uuid": uuid,
        "status": "completed",
        "nouns": list(nouns),
        "signature": str(signature_path),
    }


def worker(args: argparse.Namespace, rows: list[dict[str, Any]], source_root: Path) -> int:
    import torch

    from carnator.adapters import wan_ti2v_5b as native
    from carnator.retrieval import SpacyEntityExtractor

    start = args.shard
    shard_rows = rows[start::args.shards]
    extractor = SpacyEntityExtractor("en_core_web_sm")
    runtime_args = native_args(
        args.model_dir, f"cuda:{args.gpu}", seed=args.seed,
        height=args.height, width=args.width, frames=args.frames, fps=args.fps,
    )
    native.ensure_import_paths(runtime_args)
    native.validate_experiment_config(runtime_args)
    runtime = native.load_runtime(runtime_args)
    log_path = args.output / f"worker-gpu{args.gpu}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    failed = False
    try:
        with log_path.open("a", encoding="utf-8") as log:
            for row in shard_rows:
                uuid = str(row["uuid"])
                try:
                    nouns = canonical_nouns(extractor.extract(str(row["prompt"])))
                    result = generate_one(runtime, runtime_args, row, source_root, nouns)
                except Exception as exc:
                    result = {"uuid": uuid, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
                log.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                log.flush()
                print(json.dumps({"gpu": args.gpu, **result}, ensure_ascii=False), flush=True)
                if result.get("status") == "failed":
                    failed = True
    finally:
        runtime.profiler.close()
        torch.cuda.empty_cache()
    return 2 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", type=Path, default=DEFAULT_VIEW)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260218)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--frames", type=int, default=121)
    parser.add_argument("--fps", type=int, default=24)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.shards < 1 or not 0 <= args.shard < args.shards:
        raise SystemExit("invalid shard/shards")
    if args.gpu < 0:
        raise SystemExit("GPU id must be non-negative")
    view = args.view.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    rows = read_rows(view)
    manifest = json.loads((view / "view_manifest.json").read_text(encoding="utf-8"))
    source_root = Path(str(manifest["source_cache_root"])).expanduser().resolve() / "entries"
    if not source_root.is_dir():
        raise SystemExit(f"source entries root is missing: {source_root}")
    return worker(args, rows, source_root)


if __name__ == "__main__":
    raise SystemExit(main())
