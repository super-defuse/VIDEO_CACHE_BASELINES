#!/usr/bin/env python3
"""Run an isolated Chorus update-mask background-mismatch experiment.

The experiment owns everything below ``artifacts/chorus_mbase_failure_case_v5``:
source dense capture, target dense control, and a Chorus update-mask target,
prompt similarity, logs, and the method-local cache.  The parent process uses
only the explicitly assigned physical GPUs and remaps each child to cuda:0.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "chorus_mbase_failure_case_v5"
ALLOWED_GPUS = frozenset({0, 1, 2})
SOURCE_GPU = 0
TARGET_DENSE_GPU = 1
SCORE_GPU = 1
CHORUS_GPU = 2
SOURCE_UUID = "source-apple-snow"
TARGET_UUID = "target-peach-mountain"
SOURCE_PROMPT = (
    "a red apple rolling down a snow mountain, camera focus on the apple"
)
TARGET_PROMPT = (
    "a yellow peach rolling down a mountain, camera focus on the peach"
)
PROMPT_DIFF_OVERRIDE = {
    "source_entities": ["red apple"],
    "target_entities": ["yellow peach"],
    "removed_tokens": ["red", "apple"],
    "added_tokens": ["yellow", "peach"],
    "edit_summary": "replace red apple with yellow peach; preserve the snow background difference",
    "background_tokens_ignored": ["snow"],
}
EXPERIMENT_STAGE_BOUNDARIES = {"k1": 0, "k2": 30}


def _chorus_method(update_mask: str) -> str:
    return "chorus_mbase_only" if update_mask == "Mbase" else "chorus"


def _chorus_stage_dir(update_mask: str) -> str:
    return "chorus_mbase_only" if update_mask == "Mbase" else "chorus_medit"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _load_config(output_root: Path, device: str) -> dict[str, Any]:
    import yaml

    model_config = yaml.safe_load((ROOT / "configs/models/wan22.yaml").read_text(encoding="utf-8"))
    method_config = yaml.safe_load((ROOT / "configs/methods/default.yaml").read_text(encoding="utf-8"))
    config = {**model_config, **method_config, "dry_run": False, "device": device}
    config["cache_root"] = str((output_root / "cache").resolve())
    for key in ("model_path", "wan_source"):
        value = config.get(key)
        if value and not Path(value).is_absolute():
            config[key] = str((ROOT / value).resolve())
    retrieval = config.setdefault("retrieval", {})
    chorus_retrieval = retrieval.setdefault("chorus", {})
    chorus_retrieval["device"] = device
    chorus_retrieval["stage_scheduler"] = {
        "min_k1": 0,
        "max_k1": 0,
        "min_k2": 30,
        "max_k2": 30,
    }
    experiment_chorus = config.setdefault("chorus", {})
    experiment_chorus["stages"] = dict(EXPERIMENT_STAGE_BOUNDARIES)
    experiment_chorus["prompt_diff_override"] = dict(PROMPT_DIFF_OVERRIDE)
    return config


def _write_stage_status(output_root: Path, stage: str, payload: dict[str, Any]) -> None:
    _write_json(output_root / stage / "status.json", {"stage": stage, **payload})


def _run_source(
    output_root: Path,
    device: str,
    seed: int,
    update_mask: str,
    physical_gpu: int | None = None,
) -> None:
    from adapters import make_adapter
    from core import Request, Runner
    from core.registry import make_method

    config = _load_config(output_root, device)
    run_dir = output_root / "source_dense"
    adapter = make_adapter("wan22", config)
    method_name = _chorus_method(update_mask)
    method = make_method(method_name, config)
    runner = Runner(adapter, method, model=adapter.model_name, output_dir=run_dir,
                    cache_root=Path(config["cache_root"]))
    try:
        results = runner.run(
            [Request(0, SOURCE_UUID, SOURCE_PROMPT, seed)],
            run_dir / "results.jsonl",
        )
        result = results[-1]
        _write_json(run_dir / "config.resolved.json", {
            **config,
            "experiment": "chorus_update_mask_failure_case",
            "role": "source_dense_capture",
            "stage2_update_mask": update_mask,
            "source_prompt": SOURCE_PROMPT,
            "target_prompt": TARGET_PROMPT,
            "physical_gpu": physical_gpu,
        })
        if result.status != "ok":
            raise RuntimeError(result.error or "source dense capture failed")
        _write_stage_status(output_root, "source_dense", {
            "status": "completed",
            "method": method_name,
            "dense_capture": True,
            "result": result.extra,
            "output_path": result.output_path,
        })
    finally:
        adapter.reset_hooks()


def _run_target_dense(output_root: Path, device: str, seed: int, physical_gpu: int | None = None) -> None:
    from adapters import make_adapter
    from core import Request, Runner
    from core.registry import make_method

    config = _load_config(output_root, device)
    run_dir = output_root / "target_dense"
    adapter = make_adapter("wan22", config)
    method = make_method("baseline", config)
    runner = Runner(adapter, method, model=adapter.model_name, output_dir=run_dir,
                    cache_root=Path(config["cache_root"]))
    try:
        results = runner.run(
            [Request(1, TARGET_UUID, TARGET_PROMPT, seed)],
            run_dir / "results.jsonl",
        )
        result = results[-1]
        _write_json(run_dir / "config.resolved.json", {
            **config,
            "experiment": "chorus_mbase_failure_case",
            "role": "target_dense_control",
            "source_prompt": SOURCE_PROMPT,
            "target_prompt": TARGET_PROMPT,
            "physical_gpu": physical_gpu,
        })
        if result.status != "ok":
            raise RuntimeError(result.error or "target dense control failed")
        _write_stage_status(output_root, "target_dense", {
            "status": "completed",
            "method": "baseline",
            "result": result.extra,
            "output_path": result.output_path,
        })
    finally:
        adapter.reset_hooks()


def _run_chorus_reuse(
    output_root: Path,
    device: str,
    seed: int,
    update_mask: str,
    physical_gpu: int | None = None,
) -> None:
    from adapters import make_adapter
    from core import Request, Runner
    from core.registry import make_method

    config = _load_config(output_root, device)
    run_dir = output_root / _chorus_stage_dir(update_mask)
    adapter = make_adapter("wan22", config)
    method_name = _chorus_method(update_mask)
    method = make_method(method_name, config)
    runner = Runner(adapter, method, model=adapter.model_name, output_dir=run_dir,
                    cache_root=Path(config["cache_root"]))
    try:
        # cache_prompt pins the controlled source entry while the CLIP score is
        # measured independently below. Generation still uses TARGET_PROMPT.
        results = runner.run(
            [Request(1, TARGET_UUID, TARGET_PROMPT, seed, cache_prompt=SOURCE_PROMPT)],
            run_dir / "results.jsonl",
        )
        result = results[-1]
        _write_json(run_dir / "config.resolved.json", {
            **config,
            "experiment": "chorus_update_mask_failure_case",
            "role": f"target_chorus_{update_mask.lower()}",
            "stage2_update_mask": update_mask,
            "source_prompt": SOURCE_PROMPT,
            "target_prompt": TARGET_PROMPT,
            "cache_lookup": "controlled source prompt via cache_prompt",
            "physical_gpu": physical_gpu,
        })
        if result.status != "ok":
            raise RuntimeError(result.error or f"Chorus {update_mask} target failed")
        reported_update_mask = result.extra.get("chorus_mask", {}).get("update_mask")
        if reported_update_mask != update_mask:
            raise RuntimeError(
                f"target did not report {update_mask} as its update mask: {reported_update_mask!r}"
            )
        if not result.cache_match.hit:
            raise RuntimeError("target did not hit the controlled source cache")
        _write_stage_status(output_root, _chorus_stage_dir(update_mask), {
            "status": "completed",
            "method": method_name,
            "result": result.extra,
            "cache_match": {
                "hit": result.cache_match.hit,
                "kind": result.cache_match.kind,
                "score": result.cache_match.score,
                "source_uuid": result.cache_match.source_uuid,
            },
            "output_path": result.output_path,
        })
    finally:
        adapter.reset_hooks()


def _run_clip_score(output_root: Path, device: str, physical_gpu: int | None = None) -> None:
    from core.retrieval import TransformersClipEmbedder

    config = _load_config(output_root, device)
    retrieval = config["retrieval"]["chorus"]
    embedder = TransformersClipEmbedder(str(retrieval["model_id"]), device)
    source, target = embedder.encode([SOURCE_PROMPT, TARGET_PROMPT])
    similarity = sum(float(left) * float(right) for left, right in zip(source, target, strict=True))
    threshold = float(retrieval.get("min_similarity", 0.65))
    payload = {
        "status": "completed",
        "embedder": embedder.signature,
        "source_prompt": SOURCE_PROMPT,
        "target_prompt": TARGET_PROMPT,
        "cosine_similarity": similarity,
        "chorus_threshold": threshold,
        "passes_threshold": similarity >= threshold,
        "physical_gpu": physical_gpu,
    }
    _write_json(output_root / "clip_similarity.json", payload)
    _write_stage_status(output_root, "clip_score", payload)


def _last_result(path: Path) -> dict[str, Any]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"no result rows: {path}")
    value = json.loads(lines[-1])
    if not isinstance(value, dict):
        raise RuntimeError(f"invalid result row: {path}")
    return value


def _write_report(output_root: Path, seed: int, gpu_map: dict[str, int], requested_update_mask: str) -> None:
    source = _last_result(output_root / "source_dense" / "results.jsonl")
    dense = _last_result(output_root / "target_dense" / "results.jsonl")
    chorus_stage_dir = _chorus_stage_dir(requested_update_mask)
    chorus = _last_result(output_root / chorus_stage_dir / "results.jsonl")
    score = _read_json(output_root / "clip_similarity.json")
    chorus_extra = chorus.get("extra", {})
    update_mask = chorus_extra.get("chorus_mask", {}).get("update_mask")
    chorus_llm = chorus_extra.get("chorus_llm", {})
    chorus_mask = chorus_extra.get("chorus_mask", {})
    report = (
        f"# Chorus {requested_update_mask} failure case\n\n"
        "This is an isolated experiment. The source dense capture, target dense "
        "control, Chorus target, CLIP score, logs, and cache are all below this directory.\n\n"
        "## Prompts\n\n"
        f"- Source/cache: `{SOURCE_PROMPT}`\n"
        f"- Target: `{TARGET_PROMPT}`\n"
        f"- CLIP cosine: `{score.get('cosine_similarity')}` (threshold `{score.get('chorus_threshold')}`)\n\n"
        "- Cache policy: forced source cache reuse; the CLIP threshold is intentionally waived.\n\n"
        "## Expected comparison\n\n"
        "The target dense control should contain a yellow peach on a mountain. "
        f"The target runs all 50 steps: it updates only {requested_update_mask} in steps 0-29, then runs dense "
        f"in steps 30-49. Stage-2 source latent features remain outside {requested_update_mask}, so a snow "
        "background is the expected failure signal. The visual mismatch must be confirmed from "
        "the saved videos.\n\n"
        "## Integrity checks\n\n"
        f"- Source status: `{source.get('status')}`; target dense status: `{dense.get('status')}`; "
        f"Chorus status: `{chorus.get('status')}`\n"
        f"- Chorus cache hit: `{chorus.get('cache_match', {}).get('hit')}`; "
        f"source UUID: `{chorus.get('cache_match', {}).get('source_uuid')}`\n"
        f"- Stage-2 update mask reported by the adapter: `{update_mask}`\n"
        f"- Prompt-diff backend: `{chorus_llm.get('backend')}`; SAM phrase: `{chorus_mask.get('phrase')}`\n"
        f"- Mask fractions (base/edit/see): `{chorus_mask.get('base_token_fraction')}` / "
        f"`{chorus_mask.get('edit_fraction')}` / `{chorus_mask.get('see_fraction')}`\n"
        f"- Physical GPU allocation: `{json.dumps(gpu_map, sort_keys=True)}`\n\n"
        "## Outputs\n\n"
        "- `source_dense/videos/`: source dense video\n"
        "- `target_dense/videos/`: target dense control\n"
        f"- `{chorus_stage_dir}/videos/`: target with {requested_update_mask}-only Chorus reuse\n"
        "- `cache/`: experiment-local Chorus trajectory and source video\n"
    )
    (output_root / "report.md").write_text(report, encoding="utf-8")
    _write_json(output_root / "experiment_manifest.json", {
        "experiment": "chorus_update_mask_failure_case",
        "seed": seed,
        "source_prompt": SOURCE_PROMPT,
        "target_prompt": TARGET_PROMPT,
        "retrieval_policy": "forced_source_cache; CLIP threshold intentionally waived",
        "target_schedule": {f"stage2_{requested_update_mask.lower()}": [0, 30], "stage3_dense": [30, 50]},
        "gpu_allocation": gpu_map,
        "source_dense": source,
        "target_dense": dense,
        chorus_stage_dir: chorus,
        "clip_similarity": score,
        "stage2_update_mask": update_mask,
    })


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "source", "target-dense", "chorus-reuse", "score", "report"), default="all")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--update-mask", choices=("Mbase", "Medit"), default="Mbase")
    parser.add_argument("--seed", type=int, default=20260218)
    parser.add_argument("--source-gpu", type=int, default=SOURCE_GPU)
    parser.add_argument("--target-dense-gpu", type=int, default=TARGET_DENSE_GPU)
    parser.add_argument("--score-gpu", type=int, default=SCORE_GPU)
    parser.add_argument("--chorus-gpu", type=int, default=CHORUS_GPU)
    parser.add_argument("--device", default="cuda:0", help=argparse.SUPPRESS)
    parser.add_argument("--physical-gpu", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--resume", action="store_true")
    return parser


def _validate_gpus(args: argparse.Namespace) -> dict[str, int]:
    gpu_map = {
        "source_dense": args.source_gpu,
        "target_dense": args.target_dense_gpu,
        "clip_score": args.score_gpu,
        "chorus_reuse": args.chorus_gpu,
    }
    invalid = {gpu for gpu in gpu_map.values() if gpu not in ALLOWED_GPUS}
    if invalid:
        raise SystemExit(f"only physical GPUs {sorted(ALLOWED_GPUS)} are allowed; got {sorted(invalid)}")
    return gpu_map


def _run_child(args: argparse.Namespace, stage: str, physical_gpu: int) -> None:
    output_root = args.output_root.expanduser().resolve()
    logs = output_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--stage",
        stage,
        "--output-root",
        str(output_root),
        "--seed",
        str(args.seed),
        "--update-mask",
        args.update_mask,
        "--device",
        "cuda:0",
        "--physical-gpu",
        str(physical_gpu),
    ]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    log_path = logs / f"{stage}.log"
    with log_path.open("a", encoding="utf-8") as handle:
        completed = subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError(f"{stage} failed on physical GPU {physical_gpu}; see {log_path}")


def _stage_done(output_root: Path, stage_dir: str) -> bool:
    return (output_root / stage_dir / "status.json").is_file()


def main() -> int:
    args = _parser().parse_args()
    gpu_map = _validate_gpus(args)
    output_root = args.output_root.expanduser().resolve()
    if args.stage == "all" and output_root.exists() and not args.resume:
        raise SystemExit(f"refusing to overwrite existing experiment directory: {output_root}; use --resume")
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "prompts.json", {
        "source_prompt": SOURCE_PROMPT,
        "target_prompt": TARGET_PROMPT,
        "seed": args.seed,
        "target_schedule": {f"stage2_{args.update_mask.lower()}": [0, 30], "stage3_dense": [30, 50]},
        "stage2_update_mask": args.update_mask,
        "gpu_allocation": gpu_map,
    })

    if args.stage == "source":
        _run_source(output_root, args.device, args.seed, args.update_mask, args.physical_gpu)
    elif args.stage == "target-dense":
        _run_target_dense(output_root, args.device, args.seed, args.physical_gpu)
    elif args.stage == "chorus-reuse":
        _run_chorus_reuse(output_root, args.device, args.seed, args.update_mask, args.physical_gpu)
    elif args.stage == "score":
        _run_clip_score(output_root, args.device, args.physical_gpu)
    elif args.stage == "report":
        _write_report(output_root, args.seed, gpu_map, args.update_mask)
    else:
        if not (args.resume and _stage_done(output_root, "source_dense")):
            _run_child(args, "source", gpu_map["source_dense"])
        chorus_stage_dir = _chorus_stage_dir(args.update_mask)
        stages = (
            ("target-dense", gpu_map["target_dense"], "target_dense"),
            ("chorus-reuse", gpu_map["chorus_reuse"], chorus_stage_dir),
        )
        # A shared physical GPU is valid for constrained hosts, but the two
        # model-heavy stages must then be serialized to avoid an OOM.
        if gpu_map["target_dense"] == gpu_map["chorus_reuse"]:
            for stage, gpu, stage_dir in stages:
                if args.resume and _stage_done(output_root, stage_dir):
                    continue
                _run_child(args, stage, gpu)
            parallel = []
        else:
            parallel = []
            for stage, gpu, stage_dir in stages:
                if args.resume and _stage_done(output_root, stage_dir):
                    continue
                log_path = output_root / "logs" / f"{stage}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                handle = log_path.open("a", encoding="utf-8")
                command = [sys.executable, str(Path(__file__).resolve()), "--stage", stage,
                           "--output-root", str(output_root), "--seed", str(args.seed),
                           "--update-mask", args.update_mask, "--device", "cuda:0",
                           "--physical-gpu", str(gpu)]
                env = dict(os.environ)
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
                parallel.append((subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT), handle, stage, gpu, log_path))
        for process, handle, stage, gpu, log_path in parallel:
            return_code = process.wait()
            handle.close()
            if return_code:
                raise RuntimeError(f"{stage} failed on physical GPU {gpu}; see {log_path}")
        if not (args.resume and _stage_done(output_root, "clip_score")):
            _run_child(args, "score", gpu_map["clip_score"])
        _write_report(output_root, args.seed, gpu_map, args.update_mask)
    print(json.dumps({"status": "completed", "output_root": str(output_root), "gpu_allocation": gpu_map}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
