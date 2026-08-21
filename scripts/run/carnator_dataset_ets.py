#!/usr/bin/env python3
"""Collect real native Wan step-5 ETS signatures for a prompt dataset.

The script is intentionally a source-probe job: it runs exactly the first
``--mask-probe-step`` denoising steps for each prompt, captures the native
cross/self/fusion masks, writes a compact prompt ETS JSONL record, then removes
the transient mask and latent files unless ``--keep-probe-artifacts`` is set.
It never renders a video or runs a target continuation.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from carnator.adapters import wan_ti2v_5b as native  # noqa: E402
from carnator.retrieval import (  # noqa: E402
    CarnatorNounAnnotations,
    build_native_prompt_ets_signature_from_dir,
)
from core.types import Request  # noqa: E402


def _ensure_runtime_python(args: argparse.Namespace) -> None:
    """Mirror the native runner's environment handoff without changing entrypoint."""

    if args.no_env_setup or not args.runtime_python or os.environ.get("CARNATOR_ENV_READY") == "1":
        return
    target = Path(args.runtime_python).expanduser()
    if Path(sys.executable).resolve() == target.resolve():
        os.environ["CARNATOR_ENV_READY"] = "1"
        return
    if not target.exists():
        raise SystemExit(f"Missing runtime Python: {target}")
    environment = os.environ.copy()
    environment["CARNATOR_ENV_READY"] = "1"
    environment["PYTHONPATH"] = os.pathsep.join((str(ROOT), str(native.DEFAULT_WAN_ROOT), environment.get("PYTHONPATH", "")))
    os.execvpe(str(target), [str(target), str(Path(__file__).resolve()), *sys.argv[1:]], environment)


def _load_workload(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"request_index", "uuid", "prompt"}
    if not rows or required - set(rows[0]):
        raise ValueError(f"workload must contain {sorted(required)}")
    for row in rows:
        uuid = str(row["uuid"])
        if not uuid or Path(uuid).name != uuid:
            raise ValueError(f"unsafe or empty workload UUID: {uuid!r}")
    return rows


def _existing_uuids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    existing: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            uuid = str(row.get("uuid", ""))
            if not uuid or not isinstance(row.get("signature"), dict):
                raise ValueError(f"invalid ETS record in {path}:{line_number}")
            if uuid in existing:
                raise ValueError(f"duplicate ETS UUID in {path}: {uuid}")
            existing.add(uuid)
    return existing


def _summary_path(output: Path) -> Path:
    return output.with_suffix(output.suffix + ".manifest.json")


def _write_summary(output: Path, payload: dict[str, Any]) -> None:
    _summary_path(output).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = native.build_parser()
    parser.set_defaults(seed=20260218)
    parser.description = __doc__
    parser.add_argument("--workload", type=Path, default=ROOT / "datasets" / "vbench_cross_request_prompts.csv")
    parser.add_argument("--noun-annotations", type=Path, default=ROOT / "datasets" / "carnator_vbench_trf_nouns.csv")
    parser.add_argument("--output", type=Path, required=True,
                        help="JSONL output consumed by cross_request_dataset.py.")
    parser.add_argument("--start", type=int, default=0, help="Zero-based row offset into --workload.")
    parser.add_argument("--limit", type=int, help="Maximum number of workload rows to probe.")
    parser.add_argument("--resume", action="store_true", help="Skip UUIDs already present in --output.")
    parser.add_argument("--keep-probe-artifacts", action="store_true",
                        help="Keep transient step-5 latent and mask directories for inspection.")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.start < 0:
        raise ValueError("--start must be non-negative")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"output exists; pass --resume to append safely: {args.output}")
    # The native validation also checks layer and outer-attention options.  The
    # dataset collector always runs as a source-only, completed-step probe.
    args.source_only = True
    args.signature_only = True
    args.source_run_dir = None
    args.steps = args.mask_probe_step
    args.ets_nouns = "dataset"
    native.validate_experiment_config(args)


def _probe_one(
    runtime: native.NativeRuntime,
    args: argparse.Namespace,
    row: dict[str, str],
    nouns: tuple[str, ...],
    work_dir: Path,
) -> dict[str, Any]:
    probe_args = copy.copy(args)
    probe_args.source_prompt = str(row["prompt"])
    probe_args.seed = int(args.seed) + int(row["request_index"])
    # Dataset ETS represents the complete prompt, not a hand-picked noun in a
    # one-off acceptance pair. Empty phrase selects every non-special token.
    probe_args.source_positive = ""
    cache_dir = work_dir / "source_cache"
    coarse_mask_dir = cache_dir / "probe_masks"
    self_mask_dir = cache_dir / "self_contrast_masks"
    fusion_mask_dir = cache_dir / "fusion_masks"
    source = native.run_native_t2v_stage(
        runtime,
        probe_args,
        label=f"ets_{row['uuid']}",
        prompt=probe_args.source_prompt,
        out_video=work_dir / "no_video.mp4",
        cache_dir=cache_dir,
        capture_masks=True,
        save_trajectory=True,
        mask_output_dir=coarse_mask_dir,
        decode_video=False,
    )
    trajectory = {
        int(item["step_index"]): Path(item["path"])
        for item in source["trajectory"]
        if item.get("kind") == "post_scheduler"
    }
    self_bundle = native.build_native_self_contrast_mask_bundle(
        runtime,
        probe_args,
        source_trajectory=trajectory,
        coarse_mask_dir=coarse_mask_dir,
        self_mask_dir=self_mask_dir,
    )
    native.build_native_fusion_mask_bundle(
        runtime,
        probe_args,
        source_trajectory=trajectory,
        coarse_mask_dir=coarse_mask_dir,
        self_bundle=self_bundle,
        self_mask_dir=self_mask_dir,
        fusion_mask_dir=fusion_mask_dir,
    )
    signature = build_native_prompt_ets_signature_from_dir(
        fusion_mask_dir,
        probe_args.source_prompt,
        nouns,
        request_seed=probe_args.seed,
    )
    return {
        "request_index": int(row["request_index"]),
        "uuid": str(row["uuid"]),
        "prompt": probe_args.source_prompt,
        "seed": probe_args.seed,
        "signature": signature,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.model_dir = args.model_dir.expanduser()
    args.artifact_root = args.artifact_root.expanduser()
    args.results_root = args.results_root.expanduser()
    args.output = args.output.expanduser().resolve()
    _ensure_runtime_python(args)
    native.ensure_import_paths(args)
    _validate_args(args)
    native.validate_native_model(args)

    rows = _load_workload(args.workload.expanduser())
    selected = rows[args.start:args.start + args.limit if args.limit is not None else None]
    annotations = CarnatorNounAnnotations(args.noun_annotations.expanduser())
    existing = _existing_uuids(args.output) if args.resume else set()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    work_root = args.artifact_root.resolve() / args.run_id / "dataset_ets"
    runtime = native.load_runtime(args)
    completed = 0
    skipped = 0
    try:
        with args.output.open("a", encoding="utf-8") as output:
            for position, row in enumerate(selected, start=args.start):
                uuid = str(row["uuid"])
                if uuid in existing:
                    skipped += 1
                    continue
                work_dir = work_root / uuid
                if work_dir.exists():
                    shutil.rmtree(work_dir)
                try:
                    request = Request(int(row["request_index"]), uuid, str(row["prompt"]), args.seed)
                    record = _probe_one(runtime, args, row, annotations.for_request(request), work_dir)
                    output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                    output.flush()
                    completed += 1
                    print(f"[{position + 1}/{args.start + len(selected)}] ETS {uuid}", flush=True)
                finally:
                    if not args.keep_probe_artifacts:
                        shutil.rmtree(work_dir, ignore_errors=True)
                    gc.collect()
                    runtime.torch.cuda.empty_cache()
    finally:
        runtime.profiler.close()

    summary = {
        "mode": "native-wan-step5-prompt-ets",
        "workload": str(args.workload.resolve()),
        "output": str(args.output),
        "run_id": args.run_id,
        "requested_rows": len(selected),
        "completed": completed,
        "skipped_existing": skipped,
        "mask_probe_step": args.mask_probe_step,
        "steps_per_prompt": args.steps,
        "source_positive": "all_prompt_tokens",
        "kept_probe_artifacts": args.keep_probe_artifacts,
    }
    _write_summary(args.output, summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
