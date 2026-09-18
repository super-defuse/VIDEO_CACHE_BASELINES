#!/usr/bin/env python3
"""Run one Chorus counterexample pair for the P1 qualitative figure.

See ``docs/CHORUS_COUNTEREXAMPLE_P1.md``.  Each pair owns everything below its
output root: the source trajectory cache, a forced-reuse Chorus target, an
optional dense control, extracted frames, and a report.

The target deliberately pins ``cache_prompt`` to the pair's source, so the
reuse decision never passes through Chorus's CLIP similarity gate.  The CLIP
cosine is still measured and recorded as an audit field.

Stages run serially on a single explicitly assigned physical GPU.
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
# The published P1 artifact.  Its Medit halo is deliberately smaller than
# Chorus's own default (see DEFAULT_MASK_DILATION_EDIT); the superseded v1/v2
# roots were removed once this run became the pair of record.
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "chorus_counterexample_p1_v2_halo2"
ALLOWED_GPUS = frozenset({3, 4})
DEFAULT_GPU = 3
# Chorus's own default is 8 (``configs/methods/default.yaml``, untouched).  This
# experiment uses 2 so the reused region stays close to the source object's own
# footprint, which is what makes the reuse-scope failure attributable to the
# region's shape rather than to its size.
DEFAULT_MASK_DILATION_EDIT = 2
FRAME_INDICES = (0, 30, 60, 90, 120)
EXPERIMENT_STAGE_BOUNDARIES = {"k1": 5, "k2": 30}

PAIRS: dict[str, dict[str, Any]] = {
    "A": {
        "role": "validity",
        "source_uuid": "p1A-source-apple-snow-mountain",
        "target_uuid": "p1A-target-apple-mountain",
        # Both prompts carry the "camera focus" suffix: it improves Grounded-SAM-2
        # segmentation of the edit region, and the suffix must appear on *both*
        # sides or the CLIP cosine collapses below the 0.65 gate (measured 0.61
        # for either one-sided variant, vs 0.9149 for the symmetric pair).
        "source_prompt": "a red apple rolling down a snow mountain, camera focus on the apple",
        "target_prompt": "a red apple rolling down a mountain, camera focus on the apple",
        "expected_sam_phrase": "red apple",
        "prompt_diff_override": {
            "source_entities": ["red apple"],
            "target_entities": ["red apple"],
            "removed_tokens": ["snow"],
            "added_tokens": [],
            "edit_summary": "remove the snow from the historical mountain background",
        },
    },
    "B": {
        "role": "scope",
        "source_uuid": "p1B-source-barbell-gym-floor",
        "target_uuid": "p1B-target-football-gym-floor",
        # The planned "on a playground" pair was dropped: the model degenerates that
        # scene (a playground with no barbell at all), so Grounding DINO has nothing
        # to segment.  "on a gym floor" renders an unambiguous barbell with no people,
        # keeps one identical background term across both prompts (so the background
        # is genuinely reusable and the failure isolates to the update region), and
        # keeps CLIP cosine at 0.7303 -- comfortably above the 0.65 gate, so the pair
        # is a case Chorus's own gate would accept.  Both prompts carry the "camera
        # focus" suffix, which sharpens the segmentation of the edit region; the
        # suffix must be symmetric, since adding it to only one side drops CLIP to
        # 0.61 and the pair would no longer be a gate counterexample.  Longer, more
        # descriptive wording was tried and rejected: it emphasises the barbell and
        # pushes CLIP *below* the gate (down to 0.5273), turning the pair into an
        # artefact of forced reuse.
        "source_prompt": "a barbell on a gym floor, camera focus on the barbell",
        "target_prompt": "a football on a gym floor, camera focus on the football",
        "expected_sam_phrase": "barbell",
        "prompt_diff_override": {
            "source_entities": ["barbell"],
            "target_entities": ["football"],
            "removed_tokens": ["barbell"],
            "added_tokens": ["football"],
            "edit_summary": (
                "replace the barbell with a football, including in the camera focus clause"
            ),
        },
    },
}

_OVERRIDABLE_KEYS = (
    "source_prompt",
    "target_prompt",
    "expected_sam_phrase",
    "prompt_diff_override",
)


def _pair_spec(pair: str, prompts_json: Path | None) -> dict[str, Any]:
    spec = json.loads(json.dumps(PAIRS[pair]))
    if prompts_json is None:
        return spec
    override = json.loads(prompts_json.read_text(encoding="utf-8"))
    if not isinstance(override, dict):
        raise SystemExit(f"prompts override must be a JSON object: {prompts_json}")
    for key in _OVERRIDABLE_KEYS:
        if key in override:
            spec[key] = override[key]
    return spec


def _chorus_stage_dir() -> str:
    return "chorus_medit"


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


def _load_config(output_root: Path, device: str, spec: dict[str, Any], prompt_diff: str) -> dict[str, Any]:
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
        "min_k1": EXPERIMENT_STAGE_BOUNDARIES["k1"],
        "max_k1": EXPERIMENT_STAGE_BOUNDARIES["k1"],
        "min_k2": EXPERIMENT_STAGE_BOUNDARIES["k2"],
        "max_k2": EXPERIMENT_STAGE_BOUNDARIES["k2"],
    }
    experiment_chorus = config.setdefault("chorus", {})
    experiment_chorus["stages"] = dict(EXPERIMENT_STAGE_BOUNDARIES)
    if prompt_diff == "manual" and spec.get("prompt_diff_override") is not None:
        experiment_chorus["prompt_diff_override"] = dict(spec["prompt_diff_override"])
    if spec.get("mask_dilation_edit") is not None:
        # Experiment-local halo override only.  ``configs/methods/default.yaml``
        # keeps Chorus's own ``mask_dilation_edit`` untouched; the effective
        # value lands in config.resolved.json and is cross-checked against the
        # Grounded-SAM-2 mask metadata in the report.
        experiment_chorus["mask_dilation_edit"] = int(spec["mask_dilation_edit"])
    return config


def _write_stage_status(output_root: Path, stage: str, payload: dict[str, Any]) -> None:
    _write_json(output_root / stage / "status.json", {"stage": stage, **payload})


def _run_source(
    output_root: Path,
    device: str,
    seed: int,
    spec: dict[str, Any],
    prompt_diff: str,
    physical_gpu: int | None = None,
) -> None:
    from adapters import make_adapter
    from core import Request, Runner
    from core.registry import make_method

    config = _load_config(output_root, device, spec, prompt_diff)
    run_dir = output_root / "source_dense"
    adapter = make_adapter("wan22", config)
    method = make_method("chorus", config)
    runner = Runner(adapter, method, model=adapter.model_name, output_dir=run_dir,
                    cache_root=Path(config["cache_root"]))
    try:
        results = runner.run(
            [Request(0, spec["source_uuid"], spec["source_prompt"], seed)],
            run_dir / "results.jsonl",
        )
        result = results[-1]
        _write_json(run_dir / "config.resolved.json", {
            **config,
            "experiment": "chorus_counterexample_p1",
            "role": "source_dense_capture",
            "source_prompt": spec["source_prompt"],
            "target_prompt": spec["target_prompt"],
            "physical_gpu": physical_gpu,
        })
        if result.status != "ok":
            raise RuntimeError(result.error or "source dense capture failed")
        _write_stage_status(output_root, "source_dense", {
            "status": "completed",
            "method": "chorus",
            "dense_capture": True,
            "result": result.extra,
            "output_path": result.output_path,
        })
    finally:
        adapter.reset_hooks()


def _run_target_dense(
    output_root: Path,
    device: str,
    seed: int,
    spec: dict[str, Any],
    prompt_diff: str,
    physical_gpu: int | None = None,
) -> None:
    from adapters import make_adapter
    from core import Request, Runner
    from core.registry import make_method

    config = _load_config(output_root, device, spec, prompt_diff)
    run_dir = output_root / "target_dense"
    adapter = make_adapter("wan22", config)
    method = make_method("baseline", config)
    runner = Runner(adapter, method, model=adapter.model_name, output_dir=run_dir,
                    cache_root=Path(config["cache_root"]))
    try:
        results = runner.run(
            [Request(1, spec["target_uuid"], spec["target_prompt"], seed)],
            run_dir / "results.jsonl",
        )
        result = results[-1]
        _write_json(run_dir / "config.resolved.json", {
            **config,
            "experiment": "chorus_counterexample_p1",
            "role": "target_dense_control",
            "source_prompt": spec["source_prompt"],
            "target_prompt": spec["target_prompt"],
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
    spec: dict[str, Any],
    prompt_diff: str,
    physical_gpu: int | None = None,
) -> None:
    from adapters import make_adapter
    from core import Request, Runner
    from core.registry import make_method

    config = _load_config(output_root, device, spec, prompt_diff)
    run_dir = output_root / _chorus_stage_dir()
    adapter = make_adapter("wan22", config)
    method = make_method("chorus", config)
    runner = Runner(adapter, method, model=adapter.model_name, output_dir=run_dir,
                    cache_root=Path(config["cache_root"]))
    try:
        # cache_prompt pins the pair's source so the reuse decision bypasses the
        # CLIP similarity gate; generation still uses the target prompt.
        results = runner.run(
            [Request(1, spec["target_uuid"], spec["target_prompt"], seed,
                     cache_prompt=spec["source_prompt"])],
            run_dir / "results.jsonl",
        )
        result = results[-1]
        _write_json(run_dir / "config.resolved.json", {
            **config,
            "experiment": "chorus_counterexample_p1",
            "role": "target_chorus_medit",
            "source_prompt": spec["source_prompt"],
            "target_prompt": spec["target_prompt"],
            "cache_lookup": "forced source prompt via cache_prompt",
            "physical_gpu": physical_gpu,
        })
        if result.status != "ok":
            raise RuntimeError(result.error or "Chorus target failed")
        reported_update_mask = result.extra.get("chorus_mask", {}).get("update_mask")
        if reported_update_mask != "Medit":
            raise RuntimeError(
                f"target did not report Medit as its update mask: {reported_update_mask!r}"
            )
        if not result.cache_match.hit:
            raise RuntimeError("target did not hit the forced source cache")
        _write_stage_status(output_root, _chorus_stage_dir(), {
            "status": "completed",
            "method": "chorus",
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


def _run_clip_score(
    output_root: Path,
    device: str,
    spec: dict[str, Any],
    prompt_diff: str,
    physical_gpu: int | None = None,
) -> None:
    from core.retrieval import TransformersClipEmbedder

    config = _load_config(output_root, device, spec, prompt_diff)
    retrieval = config["retrieval"]["chorus"]
    embedder = TransformersClipEmbedder(str(retrieval["model_id"]), device)
    source, target = embedder.encode([spec["source_prompt"], spec["target_prompt"]])
    similarity = sum(float(left) * float(right) for left, right in zip(source, target, strict=True))
    threshold = float(retrieval.get("min_similarity", 0.65))
    payload = {
        "status": "completed",
        "embedder": embedder.signature,
        "source_prompt": spec["source_prompt"],
        "target_prompt": spec["target_prompt"],
        "cosine_similarity": similarity,
        "chorus_threshold": threshold,
        "passes_threshold": similarity >= threshold,
        "policy": "audit only; forced cache reuse does not consult this score",
        "physical_gpu": physical_gpu,
    }
    _write_json(output_root / "clip_similarity.json", payload)
    _write_stage_status(output_root, "clip_score", payload)


def _find_video(stage_dir: Path) -> Path:
    videos = sorted((stage_dir / "videos").glob("*.mp4"))
    if len(videos) != 1:
        raise RuntimeError(f"expected exactly one video under {stage_dir / 'videos'}, found {len(videos)}")
    return videos[0]


def _extract_stage_frames(video: Path, role: str, frames_dir: Path) -> list[str]:
    import cv2

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    follow_up = getattr(capture, "grab", None)
    written: list[str] = []
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        for index in FRAME_INDICES:
            if index >= total:
                raise RuntimeError(f"{video} has {total} frames; cannot read index {index}")
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"failed to read frame {index} from {video}")
            path = frames_dir / f"{role}_{index:03d}.png"
            if not cv2.imwrite(str(path), frame):
                raise RuntimeError(f"failed to write frame: {path}")
            written.append(path.name)
    finally:
        capture.release()
    return written


def _run_frames(output_root: Path, with_dense: bool) -> None:
    frames_dir = output_root / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    stages = [("source", output_root / "source_dense"), ("chorus", output_root / _chorus_stage_dir())]
    if with_dense:
        stages.insert(1, ("dense", output_root / "target_dense"))
    payload: dict[str, Any] = {"frame_indices": list(FRAME_INDICES), "videos": {}}
    for role, stage_dir in stages:
        video = _find_video(stage_dir)
        payload["videos"][role] = {
            "video": str(video),
            "frames": _extract_stage_frames(video, role, frames_dir),
        }
    _write_json(frames_dir / "frames.json", payload)
    _write_stage_status(output_root, "frames", {"status": "completed", **payload})


def _last_result(path: Path) -> dict[str, Any]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"no result rows: {path}")
    value = json.loads(lines[-1])
    if not isinstance(value, dict):
        raise RuntimeError(f"invalid result row: {path}")
    return value


def _write_report(
    output_root: Path,
    seed: int,
    physical_gpu: int | None,
    spec: dict[str, Any],
    prompt_diff: str,
    with_dense: bool,
    pair: str,
) -> None:
    chorus_stage_dir = _chorus_stage_dir()
    chorus = _last_result(output_root / chorus_stage_dir / "results.jsonl")
    score = _read_json(output_root / "clip_similarity.json")
    chorus_extra = chorus.get("extra", {})
    chorus_llm = chorus_extra.get("chorus_llm", {})
    chorus_mask = chorus_extra.get("chorus_mask", {})
    observed_dilation = (chorus_mask.get("dilation") or {}).get("edit")

    checks = {
        "forced_cache_hit": bool(chorus.get("cache_match", {}).get("hit")),
        "source_uuid_matches": chorus.get("cache_match", {}).get("source_uuid") == spec["source_uuid"],
        "update_mask_is_medit": chorus_mask.get("update_mask") == "Medit",
        "sam_phrase_matches": chorus_mask.get("phrase") == spec["expected_sam_phrase"],
        "mask_fractions_recorded": all(
            isinstance(chorus_mask.get(key), (int, float))
            for key in ("base_token_fraction", "edit_fraction", "see_fraction")
        ),
    }
    requested_dilation = spec.get("mask_dilation_edit")
    if requested_dilation is not None:
        # Only meaningful for runs that shrink/raise the halo; the default-halo
        # pairs keep their original five-criterion check set.
        checks["mask_dilation_matches"] = observed_dilation == int(requested_dilation)
    # Always report what the mask metadata actually says; the override flag only
    # decides whether the Chorus default is named as the source of the change.
    halo_note = "- Mask halo: `mask_dilation_edit=%s` (see `%s`)%s\n" % (
        observed_dilation,
        (chorus_mask.get("dilation") or {}).get("see"),
        ", overridden from the Chorus default `8`" if requested_dilation is not None else "",
    )
    report = (
        f"# Chorus counterexample P1 — Pair {pair} ({spec['role']})\n\n"
        "Forced cache reuse: the target pins `cache_prompt` to this pair's source, so the reuse\n"
        "decision never consults Chorus's CLIP similarity gate. The figure must be labelled\n"
        "`forced cache reuse` and must report the measured CLIP cosine below.\n\n"
        "## Prompts\n\n"
        f"- Source/cache: `{spec['source_prompt']}`\n"
        f"- Target: `{spec['target_prompt']}`\n"
        f"- Expected SAM phrase: `{spec['expected_sam_phrase']}`\n\n"
        "## Audit fields (not pass/fail)\n\n"
        f"- CLIP cosine: `{score.get('cosine_similarity')}` (threshold `{score.get('chorus_threshold')}`, "
        f"passes `{score.get('passes_threshold')}`)\n"
        f"- Prompt-diff backend: `{chorus_llm.get('backend')}` (requested `{prompt_diff}`)\n"
        f"- Update mask: `{chorus_mask.get('update_mask')}`\n"
        f"- Mask fractions (base/edit/see): `{chorus_mask.get('base_token_fraction')}` / "
        f"`{chorus_mask.get('edit_fraction')}` / `{chorus_mask.get('see_fraction')}`\n"
        + halo_note
        + f"- Seed: `{seed}`; Chorus stages: `{EXPERIMENT_STAGE_BOUNDARIES}`\n"
        f"- Physical GPU: `{physical_gpu}`\n\n"
        "## Validity checks\n\n"
        + "".join(f"- `{name}`: `{value}`\n" for name, value in checks.items())
        + "\nThe remaining validity criterion is visual: the target video must show the intended\n"
        "failure (Pair A keeps the snow background; Pair B shows a shape/texture mismatch).\n"
        "Confirm it by inspecting `frames/` before using this pair in the figure.\n\n"
        "## Outputs\n\n"
        "- `source_dense/videos/`: source dense video\n"
        f"- `{chorus_stage_dir}/videos/`: forced-reuse Chorus target video\n"
        + ("- `target_dense/videos/`: dense control (audit only, not in the figure)\n" if with_dense else "")
        + "- `frames/`: 5 extracted frames per video\n"
        "- `cache/`: pair-local Chorus trajectory\n"
    )
    (output_root / "report.md").write_text(report, encoding="utf-8")

    manifest: dict[str, Any] = {
        "experiment": "chorus_counterexample_p1",
        "pair": pair,
        "role": spec["role"],
        "seed": seed,
        "source_prompt": spec["source_prompt"],
        "target_prompt": spec["target_prompt"],
        "expected_sam_phrase": spec["expected_sam_phrase"],
        "retrieval_policy": "forced_source_cache; CLIP gate bypassed",
        "prompt_diff": prompt_diff,
        "prompt_diff_override": spec.get("prompt_diff_override") if prompt_diff == "manual" else None,
        "chorus_stages": dict(EXPERIMENT_STAGE_BOUNDARIES),
        "update_mask": "Medit",
        "mask_dilation": chorus_mask.get("dilation"),
        "mask_dilation_override": requested_dilation,
        "frame_indices": list(FRAME_INDICES),
        "with_dense": with_dense,
        "physical_gpu": physical_gpu,
        "validity_checks": checks,
        "clip_similarity": score,
        "chorus_target": chorus,
    }
    _write_json(output_root / "experiment_manifest.json", manifest)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", choices=sorted(PAIRS), required=True)
    parser.add_argument(
        "--stage",
        choices=("all", "source", "target-dense", "chorus-reuse", "score", "frames", "report"),
        default="all",
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--gpu", type=int, default=DEFAULT_GPU)
    parser.add_argument("--seed", type=int, default=20260218)
    parser.add_argument("--with-dense", action="store_true")
    parser.add_argument("--prompt-diff", choices=("manual", "auto"), default="manual")
    parser.add_argument("--prompts-json", type=Path,
                        help="Optional per-pair prompt override (see _OVERRIDABLE_KEYS).")
    parser.add_argument(
        "--mask-dilation-edit",
        type=int,
        default=DEFAULT_MASK_DILATION_EDIT,
        help="Medit halo radius for this experiment (default 2, the published P1 "
             "setting).  Chorus's own default in configs/methods/default.yaml stays "
             "at 8 and is never modified; pass 8 to reproduce the wider halo.",
    )
    parser.add_argument("--device", default="cuda:0", help=argparse.SUPPRESS)
    parser.add_argument("--physical-gpu", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--resume", action="store_true")
    return parser


def _run_child(args: argparse.Namespace, stage: str, physical_gpu: int) -> None:
    output_root = args.output_root.expanduser().resolve()
    logs = output_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--pair", args.pair,
        "--stage", stage,
        "--output-root", str(output_root),
        "--seed", str(args.seed),
        "--prompt-diff", args.prompt_diff,
        "--device", "cuda:0",
        "--physical-gpu", str(physical_gpu),
        # Without this the child falls back to DEFAULT_GPU when it rewrites
        # prompts.json, so a Pair B run on card 4 would record card 3 there
        # while config.resolved.json (written from --physical-gpu) said 4.
        "--gpu", str(physical_gpu),
    ]
    if args.with_dense:
        command.append("--with-dense")
    if args.mask_dilation_edit is not None:
        command += ["--mask-dilation-edit", str(args.mask_dilation_edit)]
    if args.prompts_json is not None:
        command += ["--prompts-json", str(args.prompts_json.expanduser().resolve())]
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
    if args.gpu not in ALLOWED_GPUS:
        raise SystemExit(f"only physical GPUs {sorted(ALLOWED_GPUS)} are allowed; got {args.gpu}")
    if args.physical_gpu is None:
        # A directly invoked stage is not spawned by _run_child, which is what
        # normally sets CUDA_VISIBLE_DEVICES.  Honour --gpu here too, otherwise
        # every direct invocation would silently land on physical GPU 0.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if args.output_root is None:
        args.output_root = DEFAULT_OUTPUT_ROOT / f"pair_{args.pair}"
    output_root = args.output_root.expanduser().resolve()
    spec = _pair_spec(args.pair, args.prompts_json)
    # Carried in the spec so every stage (including the GPU child processes)
    # resolves the same halo through _load_config.
    spec["mask_dilation_edit"] = args.mask_dilation_edit

    if args.stage == "all" and output_root.exists() and not args.resume:
        raise SystemExit(f"refusing to overwrite existing experiment directory: {output_root}; use --resume")
    output_root.mkdir(parents=True, exist_ok=True)

    _write_json(output_root / "prompts.json", {
        "pair": args.pair,
        "role": spec["role"],
        "source_prompt": spec["source_prompt"],
        "target_prompt": spec["target_prompt"],
        "expected_sam_phrase": spec["expected_sam_phrase"],
        "seed": args.seed,
        "target_schedule": {"stage2_medit": [EXPERIMENT_STAGE_BOUNDARIES["k1"],
                                             EXPERIMENT_STAGE_BOUNDARIES["k2"]],
                            "stage3_dense": [EXPERIMENT_STAGE_BOUNDARIES["k2"], 50]},
        "retrieval_policy": "forced_source_cache; CLIP gate bypassed",
        "prompt_diff": args.prompt_diff,
        "with_dense": args.with_dense,
        "mask_dilation_edit_override": args.mask_dilation_edit,
        "physical_gpu": args.gpu,
    })

    def stage_pending(stage_dir: str) -> bool:
        return not (args.resume and _stage_done(output_root, stage_dir))

    if args.stage == "all":
        # Every GPU stage runs as its own child process on the assigned card;
        # frames and report are CPU-only and run in process.
        if stage_pending("source_dense"):
            _run_child(args, "source", args.gpu)
        if args.with_dense and stage_pending("target_dense"):
            _run_child(args, "target-dense", args.gpu)
        if stage_pending(_chorus_stage_dir()):
            _run_child(args, "chorus-reuse", args.gpu)
        if stage_pending("clip_score"):
            _run_child(args, "score", args.gpu)
        _run_frames(output_root, args.with_dense)
        _write_report(output_root, args.seed, args.gpu, spec, args.prompt_diff, args.with_dense, args.pair)
    elif args.stage == "source":
        _run_source(output_root, args.device, args.seed, spec, args.prompt_diff, args.physical_gpu)
    elif args.stage == "target-dense":
        _run_target_dense(output_root, args.device, args.seed, spec, args.prompt_diff, args.physical_gpu)
    elif args.stage == "chorus-reuse":
        _run_chorus_reuse(output_root, args.device, args.seed, spec, args.prompt_diff, args.physical_gpu)
    elif args.stage == "score":
        _run_clip_score(output_root, args.device, spec, args.prompt_diff, args.physical_gpu)
    elif args.stage == "frames":
        _run_frames(output_root, args.with_dense)
    elif args.stage == "report":
        _write_report(output_root, args.seed, args.gpu, spec, args.prompt_diff, args.with_dense, args.pair)

    print(json.dumps({
        "status": "completed",
        "pair": args.pair,
        "output_root": str(output_root),
        "physical_gpu": args.gpu,
        "with_dense": args.with_dense,
        "prompt_diff": args.prompt_diff,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
