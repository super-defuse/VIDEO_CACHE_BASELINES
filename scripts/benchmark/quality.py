from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def main() -> None:
    parser = argparse.ArgumentParser(description="Run VBench in the shared vbench environment.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--quality-env", default=os.environ.get("VCB_QUALITY_ENV", "vbench"))
    parser.add_argument("--vbench-root", type=Path, default=ROOT / "external" / "vbench")
    parser.add_argument("--dimension", nargs="+", default=[
        "subject_consistency", "background_consistency", "motion_smoothness",
        "dynamic_degree", "aesthetic_quality", "imaging_quality",
    ])
    args = parser.parse_args()
    videos = args.run_dir / "videos"
    if not videos.is_dir():
        raise SystemExit(f"missing videos directory: {videos}")
    results = args.run_dir / "results.jsonl"
    if not results.is_file():
        raise SystemExit(f"missing request manifest: {results}")
    prompts = {
        Path(row["output_path"]).name: row["prompt"]
        for row in (json.loads(line) for line in results.read_text(encoding="utf-8").splitlines() if line)
        if row["status"] == "ok" and row.get("output_path")
    }
    if not prompts:
        raise SystemExit("no successful videos to evaluate")
    quality_input = args.run_dir / "quality_input.jsonl"
    source_rows = [json.loads(line) for line in results.read_text(encoding="utf-8").splitlines() if line]
    quality_rows = []
    for row in source_rows:
        copied = dict(row)
        copied["video_path"] = copied.get("output_path")
        quality_rows.append(copied)
    quality_input.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in quality_rows), encoding="utf-8"
    )
    if not (args.vbench_root / "vbench").is_dir():
        raise SystemExit(f"VBench source is missing: {args.vbench_root}; see external/README.md for setup.")
    vbench_cache = ROOT / "artifacts" / "vbench-cache"
    command = [
        "conda", "run", "--no-capture-output", "-n", args.quality_env, "env",
        f"PYTHONPATH={ROOT}{os.pathsep}{args.vbench_root}",
        f"VBENCH_CACHE_DIR={vbench_cache}",
        "python", "-m", "scripts.benchmark.quality_runner", "--results", str(quality_input),
        "--clip-weights", str(vbench_cache / "clip_model" / "ViT-B-32.pt"),
        "--vbench-dimensions", *args.dimension,
    ]
    print(" ".join(command))
    subprocess.run(command, check=True)
    scored = {row["request_index"]: row for row in (
        json.loads(line) for line in quality_input.read_text(encoding="utf-8").splitlines() if line
    )}
    for row in source_rows:
        for key in ("clip_t", "vbench_q", "vbench_dimensions"):
            if key in scored[row["request_index"]]:
                row[key] = scored[row["request_index"]][key]
    results.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in source_rows), encoding="utf-8")


if __name__ == "__main__":
    main()
