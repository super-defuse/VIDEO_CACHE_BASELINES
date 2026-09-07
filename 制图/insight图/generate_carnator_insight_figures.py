#!/usr/bin/env python3
"""Generate the standalone Carnator insight figures described in the local plan.

The experiment owns its complete artifact root below this directory.  Generation
is optional so an interrupted GPU run can be resumed with ``--render-only``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = HERE / "carnator_insight_experiment"
DEFAULT_MODEL = ROOT / "models" / "Wan2.2-TI2V-5B"
FONT_PATH = Path("/home/tangxuebin/font/arial.ttf")
if not FONT_PATH.is_file():
    raise FileNotFoundError(f"Arial font file not found: {FONT_PATH}")
font_manager.fontManager.addfont(str(FONT_PATH))
ARIAL_FAMILY = font_manager.FontProperties(fname=str(FONT_PATH)).get_name()
matplotlib.rcParams.update({"font.family": ARIAL_FAMILY, "font.sans-serif": [ARIAL_FAMILY]})
DEFAULT_GPUS = "3,4,5,6,7"
PROMPT_NEGATIVE = "text, watermark, logo, captions, people, extra objects, distorted object"
BLOCK_CONFIG = {"q_block_size": 64, "k_block_size": 128, "top_p": 0.95, "min_top_k": 0.10}
TOKEN_GRID = (31, 22, 40)


@dataclass(frozen=True)
class Request:
    request_id: str
    group: str
    role: str
    prompt: str
    positive: str
    seed: int
    capture_mass: bool = False
    capture_residual: bool = False


FIG1_REQUESTS = (
    Request("fig1_source_cat", "fig1", "source", "a cat is sleeping on a grassland, focus on the cat", "cat", 20260218, True),
    Request("fig1_target_dog", "fig1", "target", "a dog is sleeping on a grassland, focus on the dog", "dog", 20260218, True),
)
FIG2_PAIRS = {
    "similar": (
        "a cat is sleeping on a grassland, focus on the cat",
        "a dog is sleeping on a grassland, focus on the dog",
    ),
    "hard_negative": (
        "a dog is sleeping on a grassland, focus on the dog",
        "a dog is running through a snowy forest, focus on the dog",
    ),
    "easy_negative": (
        "a red sports car driving through a rainy city at night",
        "a close-up of a yellow sunflower swaying in a sunlit field",
    ),
}


def fig2_requests() -> tuple[Request, ...]:
    requests: list[Request] = []
    for pair_index, (pair, prompts) in enumerate(FIG2_PAIRS.items()):
        for role, prompt in zip(("source", "target"), prompts):
            positive = next((word for word in ("cat", "dog", "car", "sunflower") if word in prompt), "")
            requests.append(Request(f"fig2_{pair}_{role}", f"fig2_{pair}", role, prompt, positive, 20260218, False, True))
    return tuple(requests)


def all_requests() -> tuple[Request, ...]:
    return FIG1_REQUESTS + fig2_requests()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def request_paths(output_root: Path, request: Request) -> dict[str, Path]:
    artifact = output_root / "runs" / "artifacts" / request.request_id
    cache = artifact / "cache" / "source_dense"
    return {
        "artifact": artifact,
        "cache": cache,
        "video": artifact / "videos" / "source_dense.mp4",
        "initial": cache / "latents_initial.pt",
        "step5": cache / "latents_step_004.pt",
        "residual": cache / "cfg_residual_step_005.pt",
        "outer": cache / "outer_block_masks",
        "summary": output_root / "runs" / "results" / request.request_id / "summary.json",
    }


def native_command(args: argparse.Namespace, request: Request) -> list[str]:
    paths = request_paths(args.output_root, request)
    command = [
        str(args.runtime_python), "-m", "carnator.adapters.wan_ti2v_5b", "--no-env-setup",
        "--model-dir", str(args.model_dir), "--artifact-root", str(paths["artifact"].parent),
        "--results-root", str(paths["summary"].parents[1]), "--run-id", request.request_id,
        "--source-prompt", request.prompt, "--target-prompt", request.prompt,
        "--source-positive", request.positive, "--seed", str(request.seed),
        "--height", str(args.height), "--width", str(args.width), "--frames", str(args.frames),
        "--fps", str(args.fps), "--steps", str(args.steps), "--guidance-scale", str(args.guidance_scale),
        "--negative-prompt", PROMPT_NEGATIVE, "--source-only", "--outer-attn-backend", "dense",
        "--ffn-backend", "fp", "--sparse-layers", "", "--outer-sparse-layers", "15",
        "--no-middle-attention-mask", "--mask-layers", "15", "--mask-self-layer", "15",
        "--outer-mask-q-block", str(BLOCK_CONFIG["q_block_size"]),
        "--outer-mask-k-block", str(BLOCK_CONFIG["k_block_size"]),
        "--outer-mask-top-p", str(BLOCK_CONFIG["top_p"]),
        "--outer-mask-min-top-k", str(BLOCK_CONFIG["min_top_k"]), "--disable-progress",
    ]
    if request.capture_mass:
        command.append("--capture-outer-block-masks")
        command.append("--capture-block-mass")
    if request.capture_residual:
        command.extend(("--capture-cfg-residual-step", "5", "--capture-cfg-residual-path", str(paths["residual"])))
    if args.overwrite:
        command.append("--clean")
    return command


def write_request_manifest(args: argparse.Namespace, request: Request) -> None:
    paths = request_paths(args.output_root, request)
    files = {key: sha256(path) for key, path in paths.items() if key not in {"artifact", "cache", "summary", "outer"} and path.is_file()}
    outer_metadata = paths["outer"] / "metadata.json"
    if outer_metadata.is_file():
        files["outer_metadata"] = sha256(outer_metadata)
    json_dump(paths["artifact"] / "manifest.json", {
        "schema": "carnator-insight-request-v1",
        "request": asdict(request),
        "model": "Wan2.2-TI2V-5B",
        "steps": args.steps,
        "completed_step": 20 if request.capture_mass else 5,
        "attention_layer": 15 if request.capture_mass else None,
        "mask": BLOCK_CONFIG if request.capture_mass else None,
        "seed": request.seed,
        "artifact_hashes": files,
    })


def generate(args: argparse.Namespace) -> None:
    if not args.model_dir.is_dir():
        raise FileNotFoundError(f"Wan checkpoint is missing: {args.model_dir}")
    wan_root = ROOT / "external" / "wan2.2"
    env_base = os.environ.copy()
    env_base.update({
        "CARNATOR_ENV_READY": "1", "CARNATOR_DEVICE": "cuda:0",
        "CARNATOR_TI2V_5B_MODEL_DIR": str(args.model_dir),
        "PYTHONPATH": os.pathsep.join((str(ROOT), str(wan_root), env_base.get("PYTHONPATH", ""))),
    })
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU index")
    requests = all_requests()
    for start in range(0, len(requests), len(gpus)):
        batch = requests[start : start + len(gpus)]
        processes: list[tuple[Request, subprocess.Popen[bytes]]] = []
        for request, gpu in zip(batch, gpus):
            paths = request_paths(args.output_root, request)
            if paths["artifact"].exists() and not args.overwrite:
                write_request_manifest(args, request)
                continue
            env = env_base | {"CUDA_VISIBLE_DEVICES": gpu}
            print(f"[insight] GPU {gpu}: {request.request_id}", flush=True)
            process = subprocess.Popen(native_command(args, request), cwd=ROOT, env=env)
            processes.append((request, process))
        for request, process in processes:
            code = process.wait()
            if code:
                raise subprocess.CalledProcessError(code, native_command(args, request))
            write_request_manifest(args, request)
    json_dump(args.output_root / "request_manifest.json", {
        "schema": "carnator-insight-request-set-v1",
        "requests": [asdict(request) for request in requests],
        "gpus": gpus,
        "independent_artifact_root": str((args.output_root / "runs").resolve()),
    })


def load_latent(path: Path) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(f"missing latent artifact: {path}")
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, torch.Tensor) or value.ndim != 4:
        raise ValueError(f"expected [C,T,H,W] latent at {path}")
    return value.float().contiguous()


def token_vectors(latent: torch.Tensor) -> torch.Tensor:
    c, t, h, w = latent.shape
    if (t, h, w) != (31, 44, 80):
        raise ValueError(f"expected latent spatial shape [C,31,44,80], got {tuple(latent.shape)}")
    return latent.reshape(c, t, 22, 2, 40, 2).permute(1, 2, 4, 0, 3, 5).reshape(t, 22, 40, c * 4)


def cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (a * b).sum(dim=-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(eps)


def video_frame(path: Path, fraction: float = 0.5) -> Image.Image:
    metadata = iio.immeta(path)
    frames = int(round(float(metadata.get("fps", 24)) * float(metadata.get("duration", 5.04))))
    frames = max(1, frames)
    index = min(frames - 1, max(0, round((frames - 1) * fraction)))
    frame = iio.imread(path, index=index)
    return Image.fromarray(np.asarray(frame)[..., :3].astype(np.uint8))


def cache_entry(paths: dict[str, Path], step: int = 20, layer: int = 15) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    metadata = json.loads((paths["outer"] / "metadata.json").read_text(encoding="utf-8"))
    entry = next(item for item in metadata["entries"] if item["step"] == step and item["layer"] == layer and item["cfg_branch"] == "cond")
    mask = torch.load(paths["outer"] / entry["path"], map_location="cpu", weights_only=True).bool()
    mass_path = paths["outer"] / entry.get("block_mass_path", "")
    if not mass_path.is_file():
        raise FileNotFoundError(f"block mass was not captured for {paths['artifact']}")
    mass = torch.load(mass_path, map_location="cpu", weights_only=True).float()
    return mask, mass, entry


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.ndim == 3 and b.ndim == 3:
        return float(np.mean([spearman(left, right) for left, right in zip(a, b)]))
    x, y = a.detach().cpu().numpy().reshape(-1), b.detach().cpu().numpy().reshape(-1)
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    rx, ry = rankdata(x), rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def render_fig1(args: argparse.Namespace) -> None:
    fig_dir = args.output_root / "fig1"
    fig_dir.mkdir(parents=True, exist_ok=True)
    records: dict[str, Any] = {}
    masks: dict[str, torch.Tensor] = {}
    masses: dict[str, torch.Tensor] = {}
    for request in FIG1_REQUESTS:
        paths = request_paths(args.output_root, request)
        frame = video_frame(paths["video"])
        frame.save(fig_dir / f"{'source' if request.role == 'source' else 'target'}_frame.png")
        mask, mass, entry = cache_entry(paths)
        masks[request.role], masses[request.role] = mask, mass
        records[request.role] = {"request": asdict(request), "entry": entry, "frame": str(fig_dir / f"{request.role}_frame.png")}
    vmin, vmax = np.percentile(torch.cat([masses["source"].flatten(), masses["target"].flatten()]).numpy(), [1, 99])
    for role in ("source", "target"):
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)
        axes[0].imshow(masses[role].mean(dim=0), cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
        active = masks[role].float().mean(dim=0).numpy() >= 0.5
        axes[1].imshow(active, cmap="Greys", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
        fig.savefig(fig_dir / f"{role}_block_mass_and_sparse_mask.png", dpi=180)
        plt.close(fig)
    source_mask, target_mask = masks["source"], masks["target"]
    inter = (source_mask & target_mask).float().sum(dim=(1, 2))
    union = (source_mask | target_mask).float().sum(dim=(1, 2)).clamp_min(1)
    mask_iou = float((inter / union).mean())
    retained = float((masses["target"] * source_mask).sum() / masses["target"].sum().clamp_min(1e-8))
    metrics = [{"sample": "fig1_cat_dog", "pair_type": "compatible", "mask_iou_all_heads": mask_iou,
                "block_mass_spearman_all_heads": spearman(masses["source"], masses["target"]),
                "source_mask_retained_mass_on_target": retained}]
    with (fig_dir / "routing_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0])); writer.writeheader(); writer.writerows(metrics)
    json_dump(fig_dir / "manifest.json", {"schema": "carnator-insight-fig1-v1", "fixed_sample": records,
        "render": {"step": 20, "layer": 15, "branch": "cond", "head_consensus": 0.5, "mass_color_limits": [float(vmin), float(vmax)]},
        "outputs": sorted(str(path) for path in fig_dir.glob("*.png")) + [str(fig_dir / "routing_metrics.csv")]})


def pair_feature(source: Request, target: Request, args: argparse.Namespace) -> dict[str, float]:
    source_paths, target_paths = request_paths(args.output_root, source), request_paths(args.output_root, target)
    source_delta, target_delta = load_latent(source_paths["step5"]) - load_latent(source_paths["initial"]), load_latent(target_paths["step5"]) - load_latent(target_paths["initial"])
    source_vec, target_vec = token_vectors(source_delta), token_vectors(target_delta)
    delta_affinity = cosine(source_vec, target_vec)
    source_residual, target_residual = token_vectors(load_latent(source_paths["residual"])), token_vectors(load_latent(target_paths["residual"]))
    residual_affinity = cosine(source_residual, target_residual)
    energy_source = source_delta.norm(dim=0).mean(dim=(1, 2)); energy_target = target_delta.norm(dim=0).mean(dim=(1, 2))
    return {"delta_latent_global_cosine": float(delta_affinity.mean()), "delta_latent_cosine_distance": float(1 - delta_affinity.mean()),
            "energy_mae": float((energy_source - energy_target).abs().mean()), "cfg_residual_global_cosine": float(residual_affinity.mean()),
            "cfg_residual_cosine_distance": float(1 - residual_affinity.mean()),
            "cfg_residual_norm_mae": float((source_residual.norm(dim=-1) - target_residual.norm(dim=-1)).abs().mean())}


def render_fig2(args: argparse.Namespace) -> None:
    fig_dir = args.output_root / "fig2"; fig_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    all_residual_norms: list[torch.Tensor] = []
    all_residual_diffs: list[torch.Tensor] = []
    for pair in FIG2_PAIRS:
        source = next(request for request in fig2_requests() if request.group == f"fig2_{pair}" and request.role == "source")
        target = next(request for request in fig2_requests() if request.group == f"fig2_{pair}" and request.role == "target")
        source_residual = token_vectors(load_latent(request_paths(args.output_root, source)["residual"]))
        target_residual = token_vectors(load_latent(request_paths(args.output_root, target)["residual"]))
        all_residual_norms.extend((source_residual.norm(dim=-1).flatten(), target_residual.norm(dim=-1).flatten()))
        all_residual_diffs.append((source_residual.norm(dim=-1) - target_residual.norm(dim=-1)).abs().flatten())
    residual_vmax = float(torch.cat(all_residual_norms).quantile(0.99).item())
    residual_diff_vmax = float(torch.cat(all_residual_diffs).quantile(0.99).item())
    for pair, prompts in FIG2_PAIRS.items():
        source = next(request for request in fig2_requests() if request.group == f"fig2_{pair}" and request.role == "source")
        target = next(request for request in fig2_requests() if request.group == f"fig2_{pair}" and request.role == "target")
        source_paths, target_paths = request_paths(args.output_root, source), request_paths(args.output_root, target)
        video_frame(source_paths["video"]).save(fig_dir / f"pair_{pair}_source_frame.png")
        video_frame(target_paths["video"]).save(fig_dir / f"pair_{pair}_target_frame.png")
        source_delta, target_delta = load_latent(source_paths["step5"]) - load_latent(source_paths["initial"]), load_latent(target_paths["step5"]) - load_latent(target_paths["initial"])
        source_vec, target_vec = token_vectors(source_delta), token_vectors(target_delta)
        affinity = cosine(source_vec, target_vec)
        times = [round((TOKEN_GRID[0] - 1) * fraction) for fraction in (0.1, 0.5, 0.9)]
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), constrained_layout=True)
        for axis, time_index in zip(axes, times):
            image = affinity[time_index].numpy(); axis.imshow(image, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        fig.savefig(fig_dir / f"pair_{pair}_latent_affinity.png", dpi=180); plt.close(fig)
        energy_source, energy_target = source_delta.norm(dim=0).mean(dim=(1, 2)), target_delta.norm(dim=0).mean(dim=(1, 2))
        fig, axis = plt.subplots(figsize=(6.5, 2.8), constrained_layout=True)
        frames = np.arange(TOKEN_GRID[0])
        display_pair = pair.replace("_", " ")
        axis.plot(frames, energy_source, label="source")
        axis.plot(frames, energy_target, label="target")
        axis.fill_between(frames, 0, (energy_source - energy_target).abs(), alpha=0.16, label="absolute difference")
        axis.set(xlabel="frame index", ylabel="mean ||z5-z0||")
        axis.tick_params(axis="both", labelsize=12)
        axis.legend(frameon=False, ncol=3, fontsize=12)
        axis.set_title(f"{display_pair}: latent energy", fontsize=16, pad=10)
        axis.xaxis.label.set_size(14)
        axis.yaxis.label.set_size(14)
        fig.savefig(fig_dir / f"pair_{pair}_latent_energy.png", dpi=180)
        plt.close(fig)
        source_residual, target_residual = token_vectors(load_latent(source_paths["residual"])), token_vectors(load_latent(target_paths["residual"]))
        norm_source, norm_target = source_residual.norm(dim=-1), target_residual.norm(dim=-1)
        time_index = times[1]; difference = (norm_source[time_index] - norm_target[time_index]).abs()
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), constrained_layout=True)
        for axis, image, limits in ((axes[0], norm_source[time_index], (0, residual_vmax)), (axes[1], norm_target[time_index], (0, residual_vmax)), (axes[2], difference, (0, residual_diff_vmax))):
            axis.imshow(image.numpy(), cmap="magma", vmin=limits[0], vmax=max(limits[1], 1e-8), aspect="auto")
        fig.savefig(fig_dir / f"pair_{pair}_cfg_residual.png", dpi=180); plt.close(fig)
        summary_rows.append({"pair": pair, "source_prompt": source.prompt, "target_prompt": target.prompt, **pair_feature(source, target, args)})
    with (fig_dir / "pair_feature_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0])); writer.writeheader(); writer.writerows(summary_rows)
    json_dump(fig_dir / "manifest.json", {"schema": "carnator-insight-fig2-v1", "completed_step": 5, "token_grid": list(TOKEN_GRID), "cfg_residual_color_limits": {"norm_vmax": residual_vmax, "difference_vmax": residual_diff_vmax}, "pairs": summary_rows,
        "outputs": sorted(str(path) for path in fig_dir.glob("*.png")) + [str(fig_dir / "pair_feature_summary.csv")]})


def build_request_pool(args: argparse.Namespace) -> None:
    source_csv = args.dataset
    rows: list[dict[str, Any]] = []
    with source_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try: nouns = set(json.loads(row.get("carnator_nouns", "[]")))
            except json.JSONDecodeError: nouns = set()
            rows.append({"prompt": row["prompt"], "uuid": row.get("uuid", ""), "nouns": nouns})
    compatible, incompatible = [], []
    shuffled = list(rows)
    random.Random(args.pool_seed).shuffle(shuffled)
    for left, right in itertools.combinations(shuffled, 2):
        shared = sorted(left["nouns"] & right["nouns"])
        if shared and len(compatible) < args.pool_size:
            compatible.append({"pair_type": "noun-compatible", "source": left["prompt"], "target": right["prompt"], "source_uuid": left["uuid"], "target_uuid": right["uuid"], "shared_nouns": shared})
        if not shared and len(incompatible) < args.pool_size:
            incompatible.append({"pair_type": "noun-incompatible-control", "source": left["prompt"], "target": right["prompt"], "source_uuid": left["uuid"], "target_uuid": right["uuid"], "shared_nouns": []})
        if len(compatible) >= args.pool_size and len(incompatible) >= args.pool_size: break
    if len(compatible) < args.pool_size or len(incompatible) < args.pool_size:
        raise RuntimeError(f"dataset could not produce {args.pool_size} pairs of each type")
    output = args.output_root / "request_pool.jsonl"; output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(compatible + incompatible):
            record.update({"pool_index": index, "seed": args.pool_seed})
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    default_runtime = os.environ.get("CARNATOR_PYTHON", "/data/shared/conda_envs/wan/bin/python")
    parser.add_argument("--runtime-python", type=Path, default=Path(default_runtime))
    parser.add_argument("--gpus", default=DEFAULT_GPUS, help="Comma-separated physical GPU indices, default 3,4,5,6,7")
    parser.add_argument("--height", type=int, default=704); parser.add_argument("--width", type=int, default=1280); parser.add_argument("--frames", type=int, default=121); parser.add_argument("--fps", type=int, default=24); parser.add_argument("--steps", type=int, default=50); parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets" / "vbench_cross_request_prompts_carnator_nouns.csv")
    parser.add_argument("--pool-size", type=int, default=100); parser.add_argument("--pool-seed", type=int, default=20260218)
    parser.add_argument("--generate", action="store_true"); parser.add_argument("--render-only", action="store_true"); parser.add_argument("--pool-only", action="store_true"); parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args(); args.output_root = args.output_root.expanduser().resolve(); args.model_dir = args.model_dir.expanduser().resolve(); args.dataset = args.dataset.expanduser().resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    build_request_pool(args)
    if args.pool_only:
        return 0
    if args.generate and not args.render_only: generate(args)
    render_fig1(args); render_fig2(args)
    json_dump(args.output_root / "manifest.json", {"schema": "carnator-insight-experiment-v1", "plan": str(HERE / "CARNATOR_INSIGHT_FIGURES_PLAN.md"), "model": "Wan2.2-TI2V-5B", "seed": 20260218, "steps": 50, "resolution": [704, 1280], "frames": 121, "cfg": 5.0, "outputs": {"fig1": str(args.output_root / "fig1"), "fig2": str(args.output_root / "fig2"), "request_pool": str(args.output_root / "request_pool.jsonl")}})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
