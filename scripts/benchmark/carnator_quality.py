#!/usr/bin/env python3
"""Evaluate a completed Carnator matrix through the shared VBench workflow."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--quality-env", default="vbench")
    args = parser.parse_args()

    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    if matrix.get("status") != "completed":
        raise SystemExit(f"matrix is not completed: {args.matrix}")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, entry in enumerate(matrix.get("baselines", [])):
        target = entry.get("target", {})
        output = Path(entry.get("output") or target["output"])
        if not output.is_file():
            raise SystemExit(f"missing generated video: {output}")
        run_dir = output_dir / entry["baseline"]
        videos_dir = run_dir / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)
        staged = videos_dir / "video.mp4"
        if staged.is_symlink():
            if staged.resolve() != output.resolve():
                raise SystemExit(f"refusing to replace unexpected staged video: {staged}")
            staged.unlink()
        shutil.copyfile(output, staged)
        row = {
            "request_index": 0,
            "uuid": entry["baseline"],
            "prompt": entry.get("prompt") or matrix.get("target_prompt", ""),
            "model": "Wan2.2",
            "method": matrix.get("method", "carnator"),
            "status": "ok",
            "output_path": str(staged),
            "baseline": entry["baseline"],
        }
        manifest = run_dir / "results.jsonl"
        manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
        subprocess.run([
            sys.executable, str(ROOT / "scripts" / "benchmark" / "quality.py"), str(run_dir),
            "--quality-env", args.quality_env,
        ], check=True)
        scored = json.loads(manifest.read_text(encoding="utf-8"))
        records.append({
            "profile": matrix.get("profile", matrix.get("model", "unknown")), "baseline": entry["baseline"], "video": str(output),
            **{key: scored.get(key) for key in ("clip_t", "vbench_q", "vbench_dimensions")},
        })
    if not records:
        raise SystemExit("matrix has no completed baselines")
    (output_dir / "metrics.json").write_text(json.dumps({
        "status": "completed", "matrix": str(args.matrix.resolve()),
        "quality_metrics": ["clip_t", "vbench_q"], "records": records,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
