from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DIMENSIONS = (
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "aesthetic_quality",
    "imaging_quality",
    "dynamic_degree",
)
DIM_WEIGHT = {
    "subject_consistency": 1.0,
    "background_consistency": 1.0,
    "motion_smoothness": 1.0,
    "aesthetic_quality": 1.0,
    "imaging_quality": 1.0,
    "dynamic_degree": 0.5,
}
NORMALIZE_DIC = {
    "subject_consistency": (0.1462, 1.0),
    "background_consistency": (0.2615, 1.0),
    "motion_smoothness": (0.706, 0.9975),
    "aesthetic_quality": (0.0, 1.0),
    "imaging_quality": (0.0, 1.0),
    "dynamic_degree": (0.0, 1.0),
}


@dataclass(frozen=True)
class Pair:
    pair_id: str
    cached_prompt: str
    target_prompt: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_pairs(path: Path) -> list[Pair]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"pair_id", "cached_prompt", "target_prompt"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"pairs CSV must include {sorted(required)}: {path}")
    pairs = [Pair(row["pair_id"], row["cached_prompt"], row["target_prompt"]) for row in rows]
    if len({pair.pair_id for pair in pairs}) != len(pairs):
        raise ValueError(f"duplicate pair_id in {path}")
    return pairs


def target_video(run_root: Path, pair_id: str) -> Path:
    return run_root / "targets" / pair_id / "videos" / "target_latent_reuse.mp4"


def target_cache(run_root: Path, pair_id: str) -> Path:
    return run_root / "targets" / pair_id / "cache" / "target_latent_reuse"


def source_cache(run_root: Path, pair_id: str) -> Path:
    return run_root / "sources" / pair_id / "cache" / "source_dense"


def source_video(run_root: Path, pair_id: str) -> Path:
    return run_root / "sources" / pair_id / "videos" / "source_dense.mp4"


def quality_root(run_root: Path) -> Path:
    return run_root / "quality"


def staged_video_path(run_root: Path, pair_id: str) -> Path:
    return quality_root(run_root) / "input_videos" / f"{pair_id}.mp4"


def make_staged_video(source: Path, staged: Path) -> None:
    source = source.absolute()
    if staged.is_symlink():
        if staged.resolve() == source.resolve():
            return
        raise RuntimeError(f"staged video points to a different target: {staged} -> {staged.resolve()}")
    if staged.exists():
        raise RuntimeError(f"refusing to replace non-symlink staged video: {staged}")
    staged.parent.mkdir(parents=True, exist_ok=True)
    staged.symlink_to(source)


def build_rows(run_root: Path, pairs: list[Pair]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for request_index, pair in enumerate(pairs):
        video = target_video(run_root, pair.pair_id)
        cache = target_cache(run_root, pair.pair_id)
        signature = cache / "online_color_signature.json"
        manifest = cache / "manifest.json"
        source_manifest = source_cache(run_root, pair.pair_id) / "manifest.json"
        if not video.is_file() or video.stat().st_size == 0:
            raise FileNotFoundError(f"missing target video for {pair.pair_id}: {video}")
        for required in (signature, manifest, source_manifest, source_video(run_root, pair.pair_id)):
            if not required.is_file():
                raise FileNotFoundError(f"missing completed artifact for {pair.pair_id}: {required}")
        staged = staged_video_path(run_root, pair.pair_id)
        make_staged_video(video, staged)
        rows.append(
            {
                "request_index": request_index,
                "pair_id": pair.pair_id,
                "cached_prompt": pair.cached_prompt,
                "prompt": pair.target_prompt,
                "target_prompt": pair.target_prompt,
                "source_video_path": str(source_video(run_root, pair.pair_id)),
                "source_cache_dir": str(source_cache(run_root, pair.pair_id)),
                "target_video_path": str(video),
                "target_cache_dir": str(cache),
                "video_path": str(staged.absolute()),
            }
        )
    return rows


def prepare(args: argparse.Namespace) -> list[dict[str, Any]]:
    pairs = load_pairs(args.pairs)
    rows = build_rows(args.run_root, pairs)
    root = quality_root(args.run_root)
    write_jsonl(root / "input_manifest.jsonl", rows)
    write_json(
        root / "scoring_manifest.json",
        {
            "created_at": utc_now(),
            "pair_count": len(rows),
            "sample_frames": args.sample_frames,
            "clip_weights": str(args.clip_weights),
            "vbench_cache": str(args.vbench_cache),
            "dimensions": list(DIMENSIONS),
            "protocol": "per-video six-dimension VBench-Q; CLIP-T is an eight-frame mean cosine similarity",
        },
    )
    return rows


def load_prepared_rows(run_root: Path) -> list[dict[str, Any]]:
    path = quality_root(run_root) / "input_manifest.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"run --worker prepare first: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise RuntimeError(f"empty scoring input: {path}")
    return rows


def clip_scores(rows: list[dict[str, Any]], sample_frames: int, weights: Path, device: str) -> dict[str, float]:
    import clip
    import cv2
    import torch
    from PIL import Image

    if not weights.is_file():
        raise FileNotFoundError(f"missing CLIP weights: {weights}")
    model, preprocess = clip.load(str(weights), device=device, jit=False)
    model.eval()
    values: dict[str, float] = {}
    with torch.inference_mode():
        for row in rows:
            reader = cv2.VideoCapture(row["video_path"])
            try:
                frame_count = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
                if frame_count <= 0:
                    raise RuntimeError(f"CLIP-T cannot read frames: {row['video_path']}")
                positions = [round(index * (frame_count - 1) / max(sample_frames - 1, 1)) for index in range(sample_frames)]
                frames = []
                for position in positions:
                    reader.set(cv2.CAP_PROP_POS_FRAMES, position)
                    ok, frame = reader.read()
                    if ok:
                        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            finally:
                reader.release()
            if not frames:
                raise RuntimeError(f"CLIP-T found no readable frames: {row['video_path']}")
            image_tensor = torch.stack([preprocess(Image.fromarray(frame)) for frame in frames]).to(device)
            text_tokens = clip.tokenize([row["prompt"]]).to(device)
            text = torch.nn.functional.normalize(model.encode_text(text_tokens), dim=-1)
            image = torch.nn.functional.normalize(model.encode_image(image_tensor), dim=-1)
            values[row["pair_id"]] = float((image @ text.T).mean().cpu())
    return values


def run_clip(args: argparse.Namespace) -> None:
    rows = load_prepared_rows(args.run_root)
    if not 0 <= args.clip_shard_index < args.clip_shard_count:
        raise ValueError("invalid CLIP-T shard index/count")
    selected_rows = rows[args.clip_shard_index :: args.clip_shard_count]
    if not selected_rows:
        raise ValueError("CLIP-T shard has no rows")
    output_name = "clip_t.json" if args.clip_shard_count == 1 else f"clip_t_shard_{args.clip_shard_index:02d}_of_{args.clip_shard_count:02d}.json"
    output = quality_root(args.run_root) / output_name
    existing = read_json(output) if output.is_file() else None
    expected_pairs = {row["pair_id"] for row in selected_rows}
    if isinstance(existing, dict) and set(existing.get("scores", {})) == expected_pairs:
        print(json.dumps({"worker": "clip", "status": "already-complete", "count": len(selected_rows), "shard": args.clip_shard_index}))
        return
    started = time.perf_counter()
    scores = clip_scores(selected_rows, args.sample_frames, args.clip_weights, args.device)
    if set(scores) != expected_pairs:
        raise RuntimeError(f"CLIP-T score coverage mismatch: got={len(scores)} expected={len(selected_rows)}")
    write_json(
        output,
        {
            "completed_at": utc_now(),
            "device": args.device,
            "sample_frames": args.sample_frames,
            "shard_index": args.clip_shard_index,
            "shard_count": args.clip_shard_count,
            "wall_clock_seconds": time.perf_counter() - started,
            "scores": scores,
        },
    )
    print(json.dumps({"worker": "clip", "status": "completed", "count": len(scores), "shard": args.clip_shard_index}))


def install_opencv_vbench_video_loader() -> None:
    import cv2
    import numpy as np
    import torch
    import vbench.aesthetic_quality as aesthetic_quality
    import vbench.background_consistency as background_consistency
    import vbench.imaging_quality as imaging_quality
    import vbench.subject_consistency as subject_consistency
    from vbench.utils import get_frame_indices

    def load_video(video_path: str, num_frames: int | None = None, return_tensor: bool = True, data_transform: Any = None, width: int | None = None, height: int | None = None) -> Any:
        if not video_path.endswith(".mp4"):
            raise NotImplementedError(f"OpenCV compatibility loader only supports MP4: {video_path}")
        reader = cv2.VideoCapture(video_path)
        frames: list[Any] = []
        try:
            while True:
                ok, frame = reader.read()
                if not ok:
                    break
                if width is not None or height is not None:
                    frame = cv2.resize(frame, (width or frame.shape[1], height or frame.shape[0]))
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        finally:
            reader.release()
        if not frames:
            raise RuntimeError(f"OpenCV could not decode video: {video_path}")
        buffer = np.stack(frames).astype(np.uint8, copy=False)
        if num_frames:
            buffer = buffer[get_frame_indices(num_frames, len(buffer), sample="middle")]
        if data_transform:
            return data_transform(buffer)
        if return_tensor:
            return torch.Tensor(buffer).permute(0, 3, 1, 2)
        return buffer

    subject_consistency.load_video = load_video
    background_consistency.load_video = load_video
    aesthetic_quality.load_video = load_video
    imaging_quality.load_video = load_video


def run_vbench_dimension(args: argparse.Namespace) -> None:
    if args.dimension not in DIMENSIONS:
        raise ValueError(f"unsupported VBench dimension: {args.dimension}")
    rows = load_prepared_rows(args.run_root)
    output_dir = quality_root(args.run_root) / "vbench" / args.dimension
    output_file = output_dir / f"target_{args.dimension}_eval_results.json"
    if output_file.is_file():
        payload = read_json(output_file)
        result = payload.get(args.dimension)
        if isinstance(result, list) and len(result) > 1 and isinstance(result[1], list) and len(result[1]) == len(rows):
            print(json.dumps({"worker": "vbench", "dimension": args.dimension, "status": "already-complete", "count": len(rows)}))
            return
    os.environ["VBENCH_CACHE_DIR"] = str(args.vbench_cache)
    sys.path.insert(0, str(ROOT / "external" / "vbench"))
    from vbench import VBench

    install_opencv_vbench_video_loader()
    prompts = {str(Path(row["video_path"]).absolute()): row["prompt"] for row in rows}
    staged_dir = quality_root(args.run_root) / "input_videos"
    if len(list(staged_dir.glob("*.mp4"))) != len(rows):
        raise RuntimeError(f"staged video count mismatch in {staged_dir}")
    started = time.perf_counter()
    runner = VBench(args.device, str(ROOT / "external" / "vbench" / "vbench" / "VBench_full_info.json"), str(output_dir))
    runner.evaluate(
        videos_path=str(staged_dir),
        name=f"target_{args.dimension}",
        prompt_list=prompts,
        dimension_list=[args.dimension],
        local=True,
        mode="custom_input",
    )
    payload = read_json(output_file)
    result = payload.get(args.dimension)
    if not isinstance(result, list) or len(result) < 2 or not isinstance(result[1], list) or len(result[1]) != len(rows):
        raise RuntimeError(f"VBench {args.dimension} returned incomplete results")
    write_json(
        output_dir / "worker_timing.json",
        {"completed_at": utc_now(), "device": args.device, "wall_clock_seconds": time.perf_counter() - started, "video_count": len(rows)},
    )
    print(json.dumps({"worker": "vbench", "dimension": args.dimension, "status": "completed", "count": len(rows)}))


def canonical_dimension_value(dimension: str, value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    result = float(value)
    if dimension == "imaging_quality":
        result /= 100.0
    if not math.isfinite(result):
        raise ValueError(f"non-finite {dimension} score: {value}")
    return result


def aggregate_vbench_scores(dimensions: dict[str, float]) -> float:
    missing = [dimension for dimension in DIMENSIONS if dimension not in dimensions]
    if missing:
        raise ValueError(f"missing VBench dimensions: {missing}")
    rounded = {dimension: round(float(dimensions[dimension]), 4) for dimension in DIMENSIONS}
    normalized = {
        dimension: ((rounded[dimension] - NORMALIZE_DIC[dimension][0]) / (NORMALIZE_DIC[dimension][1] - NORMALIZE_DIC[dimension][0])) * DIM_WEIGHT[dimension]
        for dimension in DIMENSIONS
    }
    return (sum(normalized.values()) / sum(DIM_WEIGHT.values())) * (4.0 / 5.0)


def load_vbench_results(run_root: Path, rows: list[dict[str, Any]]) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    staged_to_pair = {str(Path(row["video_path"]).absolute()): row["pair_id"] for row in rows}
    dimensions_by_pair = {row["pair_id"]: {} for row in rows}
    official_averages: dict[str, float] = {}
    for dimension in DIMENSIONS:
        path = quality_root(run_root) / "vbench" / dimension / f"target_{dimension}_eval_results.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing VBench output for {dimension}: {path}")
        result = read_json(path).get(dimension)
        if not isinstance(result, list) or len(result) < 2 or not isinstance(result[1], list):
            raise RuntimeError(f"invalid VBench output for {dimension}: {path}")
        official_averages[dimension] = float(result[0])
        seen: set[str] = set()
        for item in result[1]:
            video_path = str(Path(item.get("video_path", "")).absolute())
            pair_id = staged_to_pair.get(video_path)
            if pair_id is None:
                raise RuntimeError(f"VBench {dimension} returned an unknown video: {video_path}")
            dimensions_by_pair[pair_id][dimension] = canonical_dimension_value(dimension, item.get("video_results"))
            seen.add(pair_id)
        missing = set(dimensions_by_pair) - seen
        if missing:
            raise RuntimeError(f"VBench {dimension} omitted pairs: {sorted(missing)}")
    return dimensions_by_pair, official_averages


def aggregate(args: argparse.Namespace) -> None:
    rows = load_prepared_rows(args.run_root)
    root = quality_root(args.run_root)
    clip_paths = [root / "clip_t.json"] if (root / "clip_t.json").is_file() else sorted(root.glob("clip_t_shard_*_of_*.json"))
    if not clip_paths:
        raise FileNotFoundError(f"missing CLIP-T output in {root}")
    clip_by_pair: dict[str, float] = {}
    for clip_path in clip_paths:
        scores = read_json(clip_path).get("scores")
        if not isinstance(scores, dict) or set(scores) & set(clip_by_pair):
            raise RuntimeError(f"invalid or overlapping CLIP-T shard: {clip_path}")
        clip_by_pair.update({pair_id: float(score) for pair_id, score in scores.items()})
    if set(clip_by_pair) != {row["pair_id"] for row in rows}:
        raise RuntimeError("CLIP-T score coverage mismatch")
    dimensions_by_pair, official_averages = load_vbench_results(args.run_root, rows)
    results: list[dict[str, Any]] = []
    for row in rows:
        dimensions = dimensions_by_pair[row["pair_id"]]
        results.append(
            {
                **row,
                "clip_t": float(clip_by_pair[row["pair_id"]]),
                "vbench_dimensions": dimensions,
                "vbench_q_i": aggregate_vbench_scores(dimensions),
            }
        )
    write_jsonl(root / "pair_results.jsonl", results)
    summary = {
        "completed_at": utc_now(),
        "pair_count": len(results),
        "clip_t_mean": statistics.fmean(row["clip_t"] for row in results),
        "vbench_q_mean": statistics.fmean(row["vbench_q_i"] for row in results),
        "per_video_dimension_means": {dimension: statistics.fmean(row["vbench_dimensions"][dimension] for row in results) for dimension in DIMENSIONS},
        "official_vbench_batch_averages": official_averages,
        "vbench_q_formula": "per-video six-dimension normalization and 4/5 quality weighting, matching scripts/benchmark/quality_runner.py",
    }
    write_json(root / "quality_summary.json", summary)
    print(json.dumps({"worker": "aggregate", "status": "completed", "pair_count": len(results), "vbench_q_mean": summary["vbench_q_mean"], "clip_t_mean": summary["clip_t_mean"]}))


def parse_gpus(value: str) -> list[int]:
    gpus = [int(item.strip()) for item in value.split(",") if item.strip()]
    if len(gpus) != len(set(gpus)) or any(gpu < 0 for gpu in gpus):
        raise ValueError(f"invalid GPU list: {value}")
    return gpus


def run_all(args: argparse.Namespace) -> None:
    prepare(args)
    gpus = parse_gpus(args.gpus)
    if len(gpus) < len(DIMENSIONS) + 1:
        raise ValueError(f"--worker all needs at least {len(DIMENSIONS) + 1} GPUs")
    log_dir = quality_root(args.run_root) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    tasks: list[tuple[str, int, list[str]]] = []
    for dimension, gpu in zip(DIMENSIONS, gpus):
        tasks.append((f"vbench_{dimension}", gpu, ["--worker", "vbench", "--dimension", dimension, "--device", f"cuda:{gpu}"]))
    clip_gpus = gpus[len(DIMENSIONS) :]
    for shard_index, gpu in enumerate(clip_gpus):
        tasks.append((f"clip_t_{shard_index}", gpu, ["--worker", "clip", "--device", f"cuda:{gpu}", "--clip-shard-index", str(shard_index), "--clip-shard-count", str(len(clip_gpus))]))
    processes: list[tuple[str, subprocess.Popen[Any], Any]] = []
    for name, _gpu, worker_args in tasks:
        log_handle = (log_dir / f"{name}.log").open("w", encoding="utf-8")
        command = [sys.executable, str(Path(__file__).resolve()), "--run-root", str(args.run_root), "--pairs", str(args.pairs), "--sample-frames", str(args.sample_frames), "--clip-weights", str(args.clip_weights), "--vbench-cache", str(args.vbench_cache), *worker_args]
        processes.append((name, subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT, cwd=ROOT), log_handle))
    failures: list[str] = []
    for name, process, log_handle in processes:
        exit_code = process.wait()
        log_handle.close()
        if exit_code:
            failures.append(f"{name} (exit={exit_code})")
    if failures:
        raise RuntimeError(f"scoring workers failed: {', '.join(failures)}")
    aggregate(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score Early Signature calibration targets per video with VBench-Q and CLIP-T.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--worker", choices=("prepare", "clip", "vbench", "aggregate", "all"), default="all")
    parser.add_argument("--dimension", choices=DIMENSIONS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--sample-frames", type=int, default=8)
    parser.add_argument("--clip-shard-index", type=int, default=0)
    parser.add_argument("--clip-shard-count", type=int, default=1)
    parser.add_argument("--clip-weights", type=Path, default=ROOT / "artifacts" / "vbench-cache" / "clip_model" / "ViT-B-32.pt")
    parser.add_argument("--vbench-cache", type=Path, default=ROOT / "artifacts" / "vbench-cache")
    args = parser.parse_args()
    if args.sample_frames <= 0:
        parser.error("--sample-frames must be positive")
    if args.worker == "vbench" and args.dimension is None:
        parser.error("--worker vbench requires --dimension")
    return args


def main() -> int:
    args = parse_args()
    if args.worker == "prepare":
        rows = prepare(args)
        print(json.dumps({"worker": "prepare", "status": "completed", "count": len(rows)}))
    elif args.worker == "clip":
        run_clip(args)
    elif args.worker == "vbench":
        run_vbench_dimension(args)
    elif args.worker == "aggregate":
        aggregate(args)
    else:
        run_all(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
