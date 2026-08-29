from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


TASK_INFO = (
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
QUALITY_WEIGHT = 4.0
SEMANTIC_WEIGHT = 1.0


def aggregate_vbench_scores(averages: dict[str, float]) -> float:
    """Match ``benchmarking/scripts/bench-custom.py``'s full_vbench_score."""
    missing = [dimension for dimension in TASK_INFO if dimension not in averages]
    if missing:
        raise ValueError(f"missing VBench dimensions: {missing}")
    rounded = {dimension: round(float(averages[dimension]), 4) for dimension in TASK_INFO}
    normalized = {
        dimension: ((rounded[dimension] - NORMALIZE_DIC[dimension][0]) / (NORMALIZE_DIC[dimension][1] - NORMALIZE_DIC[dimension][0])) * DIM_WEIGHT[dimension]
        for dimension in TASK_INFO
    }
    quality_score = sum(normalized.values()) / sum(DIM_WEIGHT.values())
    # SEMANTIC_LIST is empty in the original bench-custom protocol.
    return (quality_score * QUALITY_WEIGHT) / (QUALITY_WEIGHT + SEMANTIC_WEIGHT)


def _successful_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if any(row.get("split") == "evaluation" for row in rows):
        rows = [row for row in rows if row.get("split") == "evaluation"]
    return [row for row in rows if row.get("status") == "ok" and row.get("video_path")]


def _is_finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def has_complete_quality(rows: list[dict[str, Any]]) -> bool:
    """Return whether every generated video already has the full quality record."""

    if not rows:
        return False
    for row in rows:
        dimensions = row.get("vbench_dimensions")
        if not _is_finite_number(row.get("clip_t")) or not _is_finite_number(row.get("vbench_q")):
            return False
        if not isinstance(dimensions, dict) or any(not _is_finite_number(dimensions.get(name)) for name in TASK_INFO):
            return False
    return True


def has_complete_vbench_quality(row: dict[str, Any]) -> bool:
    dimensions = row.get("vbench_dimensions")
    return (
        _is_finite_number(row.get("vbench_q"))
        and isinstance(dimensions, dict)
        and all(_is_finite_number(dimensions.get(name)) for name in TASK_INFO)
    )


def stage_videos(rows: list[dict[str, Any]], output_dir: Path) -> tuple[Path, dict[str, str]]:
    """Expose videos from multiple worker directories under one VBench input root."""

    sources = [(row, Path(str(row["video_path"])).expanduser().resolve()) for row in rows]
    fingerprint = hashlib.sha256(
        "\n".join(f"{row['request_index']}\t{row['uuid']}\t{source}" for row, source in sources).encode("utf-8")
    ).hexdigest()[:16]
    staged_dir = output_dir / f"input_videos_{fingerprint}"
    staged_dir.mkdir(parents=True, exist_ok=True)
    staged_paths: dict[str, str] = {}
    for row, source in sources:
        if not source.is_file():
            raise FileNotFoundError(f"quality video is missing: {source}")
        suffix = source.suffix or ".mp4"
        destination = staged_dir / f"{int(row['request_index']):04d}_{row['uuid']}{suffix}"
        if destination.is_symlink():
            if destination.resolve() != source:
                raise RuntimeError(f"staged quality video points to a different source: {destination}")
        elif destination.exists():
            raise FileExistsError(f"refusing to replace non-symlink quality input: {destination}")
        else:
            destination.symlink_to(source)
        staged_paths[str(source)] = str(destination.absolute())
    return staged_dir, staged_paths


def clip_t(rows: list[dict[str, Any]], sample_frames: int, model_path: Path, device: str) -> dict[int, float]:
    """Compute per-request frame-mean CLIP text/image cosine similarity."""
    import clip
    import cv2
    import torch
    from PIL import Image

    model, preprocess = clip.load(str(model_path), device=device, jit=False)
    model.eval()
    scores: dict[int, float] = {}
    with torch.inference_mode():
        for row in rows:
            reader = cv2.VideoCapture(row["video_path"])
            try:
                frame_count = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
                if frame_count <= 0:
                    continue
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
                continue
            images = [Image.fromarray(frame) for frame in frames]
            image_tensor = torch.stack([preprocess(image) for image in images]).to(device)
            text_tokens = clip.tokenize([row["prompt"]]).to(device)
            text = model.encode_text(text_tokens)
            image = model.encode_image(image_tensor)
            text = torch.nn.functional.normalize(text, dim=-1)
            image = torch.nn.functional.normalize(image, dim=-1)
            scores[row["request_index"]] = float((image @ text.T).mean().cpu())
    return scores


def install_opencv_vbench_video_loader() -> None:
    """Avoid the Decord/FFmpeg incompatibility in the shared VBench env."""
    import cv2
    import numpy as np
    import torch
    import vbench.aesthetic_quality as aesthetic_quality
    import vbench.background_consistency as background_consistency
    import vbench.imaging_quality as imaging_quality
    import vbench.subject_consistency as subject_consistency
    from vbench.utils import get_frame_indices

    def load_video(video_path: str, num_frames: int | None = None, return_tensor: bool = True, data_transform: Any = None, width: int | None = None, height: int | None = None) -> Any:
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
        return torch.Tensor(buffer).permute(0, 3, 1, 2) if return_tensor else buffer

    subject_consistency.load_video = load_video
    background_consistency.load_video = load_video
    aesthetic_quality.load_video = load_video
    imaging_quality.load_video = load_video


def vbench_quality(
    rows: list[dict[str, Any]], output_dir: Path, dimensions: list[str], device: str
) -> tuple[float | None, dict[int, dict[str, float | None]]]:
    """Run VBench and apply the original CHAI custom-input aggregation."""
    if not dimensions:
        dimensions = list(TASK_INFO)
    unsupported = sorted(set(dimensions) - set(TASK_INFO))
    missing = [dimension for dimension in TASK_INFO if dimension not in dimensions]
    if unsupported or missing:
        raise ValueError(
            "CHAI VBench-Q requires exactly the six dimensions "
            f"{list(TASK_INFO)}; unsupported={unsupported}, missing={missing}"
        )
    root = Path(__file__).resolve().parents[2]
    vbench_root = root / "external" / "vbench"
    sys.path.insert(0, str(vbench_root))
    from vbench import VBench

    install_opencv_vbench_video_loader()
    output_dir.mkdir(parents=True, exist_ok=True)
    staged_videos, staged_paths = stage_videos(rows, output_dir)
    prompts: dict[str, str] = {}
    for row in rows:
        video_path = Path(row["video_path"])
        staged_path = Path(staged_paths[str(video_path.resolve())])
        # VBench enumerates staged symlink paths, while result manifests may
        # record the resolved source path. Support both representations.
        prompts[str(video_path.absolute())] = row["prompt"]
        prompts[str(video_path.resolve())] = row["prompt"]
        prompts[str(staged_path)] = row["prompt"]
        prompts[str(staged_path.resolve())] = row["prompt"]
    runner = VBench(
        device,
        str(vbench_root / "vbench" / "VBench_full_info.json"),
        str(output_dir),
    )
    runner.evaluate(
        videos_path=str(staged_videos),
        name="custom_vidprom",
        prompt_list=prompts,
        dimension_list=dimensions,
        local=True,
        mode="custom_input",
    )
    result_file = output_dir / "custom_vidprom_eval_results.json"
    payload = json.loads(result_file.read_text())
    averages: dict[str, float] = {}
    per_request: dict[int, dict[str, float | None]] = {row["request_index"]: {} for row in rows}
    path_to_index: dict[str, int] = {}
    for row in rows:
        video_path = Path(row["video_path"])
        staged_path = Path(staged_paths[str(video_path.resolve())])
        path_to_index[str(video_path.absolute())] = row["request_index"]
        path_to_index[str(video_path.resolve())] = row["request_index"]
        path_to_index[str(staged_path.absolute())] = row["request_index"]
        path_to_index[str(staged_path.resolve())] = row["request_index"]
    for dimension in dimensions:
        result = payload.get(dimension)
        if not isinstance(result, list) or not result:
            continue
        averages[dimension] = float(result[0])
        for video_result in result[1] if len(result) > 1 and isinstance(result[1], list) else []:
            video_path = str(Path(video_result.get("video_path", "")).resolve())
            request_index = path_to_index.get(video_path)
            if request_index is not None:
                value = video_result.get("video_results")
                if value is None:
                    per_request[request_index][dimension] = None
                elif isinstance(value, bool):
                    per_request[request_index][dimension] = float(value)
                else:
                    score = float(value)
                    per_request[request_index][dimension] = score / 100.0 if dimension == "imaging_quality" else score

    missing = [dimension for dimension in TASK_INFO if dimension not in averages]
    if missing:
        raise RuntimeError(f"VBench did not return required CHAI dimensions: {missing}")
    final_score = aggregate_vbench_scores(averages)
    for request_index, values in per_request.items():
        if all(values.get(dimension) is not None for dimension in TASK_INFO):
            values["vbench_q"] = aggregate_vbench_scores({dimension: float(values[dimension]) for dimension in TASK_INFO})
    return final_score, per_request


def _rewrite_results(
    path: Path,
    clip_scores: dict[int, float],
    vbench_score: float | None,
    evaluated_indices: set[int],
    vbench_dimensions: dict[int, dict[str, float | None]] | None = None,
) -> None:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    for row in rows:
        if row["request_index"] in clip_scores:
            row["clip_t"] = clip_scores[row["request_index"]]
        if vbench_dimensions and row["request_index"] in vbench_dimensions:
            dimensions = vbench_dimensions[row["request_index"]]
            row["vbench_dimensions"] = dimensions
            if dimensions.get("vbench_q") is not None:
                row["vbench_q"] = dimensions["vbench_q"]
        elif row["request_index"] in evaluated_indices and vbench_score is not None:
            row["vbench_q"] = vbench_score
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Attach CLIP-T and VBench-Q to CHAI result JSONL.")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--sample-frames", type=int, default=8)
    parser.add_argument("--clip-weights", type=Path)
    parser.add_argument("--vbench-dimensions", nargs="*", default=list(TASK_INFO))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-if-complete",
        action="store_true",
        help="Exit successfully when all successful requests already have CLIP-T and six-dimensional VBench-Q.",
    )
    parser.add_argument(
        "--reuse-existing-metrics",
        action="store_true",
        help="Keep existing CLIP-T and per-request VBench metrics, evaluating only requests with missing values.",
    )
    args = parser.parse_args()
    rows = _successful_rows(args.results)
    if args.skip_if_complete and has_complete_quality(rows):
        print(json.dumps({"status": "skipped_existing_quality", "requests": len(rows)}, ensure_ascii=False))
        return
    dry_media = bool(rows) and Path(rows[0]["video_path"]).read_bytes().startswith(b"CHAI-DRY-RUN:")
    if args.dry_run or dry_media:
        # Dry-run media is protocol-only, so expose deterministic placeholders
        # without pretending that binary smoke-test payloads are real videos.
        _rewrite_results(
            args.results,
            {row["request_index"]: 0.0 for row in rows},
            0.0,
            {row["request_index"] for row in rows},
            {row["request_index"]: {**{dimension: 0.0 for dimension in TASK_INFO}, "vbench_q": 0.0} for row in rows},
        )
        return
    root = Path(__file__).resolve().parents[2]
    clip_weights = args.clip_weights or root / "artifacts" / "vbench-cache" / "clip_model" / "ViT-B-32.pt"
    existing_clip_scores = {
        row["request_index"]: float(row["clip_t"])
        for row in rows
        if args.reuse_existing_metrics and _is_finite_number(row.get("clip_t"))
    }
    clip_rows = [row for row in rows if row["request_index"] not in existing_clip_scores]
    scores = {
        **existing_clip_scores,
        **(clip_t(clip_rows, args.sample_frames, clip_weights, args.device) if clip_rows else {}),
    }
    vbench_rows = [row for row in rows if not args.reuse_existing_metrics or not has_complete_vbench_quality(row)]
    score, dimensions = (
        vbench_quality(vbench_rows, args.results.parent / "vbench", args.vbench_dimensions, args.device)
        if vbench_rows
        else (None, {})
    )
    _rewrite_results(args.results, scores, score, {row["request_index"] for row in vbench_rows}, dimensions)


if __name__ == "__main__":
    main()
