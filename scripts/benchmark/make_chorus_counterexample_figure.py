#!/usr/bin/env python3
"""Compose the P1 four-row Chorus counterexample figure.

Rows are ``(role, prompt, video)``.  Each row renders the prompt in the leftmost
column and ``--frames`` evenly spaced frames of that row's video on the right.

See ``docs/CHORUS_COUNTEREXAMPLE_P1.md``.  The default input is the two pair
artifact roots produced by ``run_chorus_counterexample_p1.py``; ``--rows-json``
overrides the row list entirely so the figure can be re-laid-out without
touching the generation artifacts.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "chorus_counterexample_p1_v2_halo2"
FONT_PATH = Path("/home/tangxuebin/font/arial.ttf")
if not FONT_PATH.is_file():
    raise FileNotFoundError(f"Arial font file not found: {FONT_PATH}")
font_manager.fontManager.addfont(str(FONT_PATH))
ARIAL_FAMILY = font_manager.FontProperties(fname=str(FONT_PATH)).get_name()
matplotlib.rcParams.update({"font.family": ARIAL_FAMILY, "font.sans-serif": [ARIAL_FAMILY]})

DEFAULT_FRAMES = (0, 30, 60, 90, 120)
FRAME_WIDTH_IN = 1.35
PROMPT_WIDTH_IN = 1.60
ROLE_COLOR = "#8a8a8a"
PROMPT_COLOR = "#1a1a1a"


def _find_video(stage_dir: Path) -> Path:
    videos = sorted((stage_dir / "videos").glob("*.mp4"))
    if len(videos) != 1:
        raise RuntimeError(f"expected exactly one video under {stage_dir / 'videos'}, found {len(videos)}")
    return videos[0]


def _prompt_of(root: Path, key: str) -> str:
    payload = json.loads((root / "prompts.json").read_text(encoding="utf-8"))
    return str(payload[key])


def _rows_from_artifacts(pair_a: Path, pair_b: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for name, root in (("A", pair_a), ("B", pair_b)):
        rows.append({
            "role": f"Pair {name} · cache",
            "prompt": _prompt_of(root, "source_prompt"),
            "video": str(_find_video(root / "source_dense")),
        })
        rows.append({
            "role": f"Pair {name} · Chorus output",
            "prompt": _prompt_of(root, "target_prompt"),
            "video": str(_find_video(root / "chorus_medit")),
        })
    return rows


def _read_frames(video: Path, indices: tuple[int, ...]) -> list[Any]:
    import cv2

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    images: list[Any] = []
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        for index in indices:
            if index >= total:
                raise RuntimeError(f"{video} has {total} frames; cannot read index {index}")
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"failed to read frame {index} from {video}")
            images.append(Image.fromarray(frame[:, :, ::-1]))
    finally:
        capture.release()
    return images


def render(rows: list[dict[str, str]], frames: tuple[int, ...], output: Path, dpi: int) -> None:
    if not rows:
        raise SystemExit("no rows to render")
    n_rows = len(rows)
    n_cols = 1 + len(frames)
    width_ratios = [PROMPT_WIDTH_IN] + [FRAME_WIDTH_IN] * len(frames)
    figure_width = sum(width_ratios)
    frame_height = FRAME_WIDTH_IN * 704 / 1280
    figure_height = n_rows * frame_height + 0.18 * (n_rows + 1)

    fig = plt.figure(figsize=(figure_width, figure_height))
    grid = fig.add_gridspec(
        n_rows, n_cols, width_ratios=width_ratios,
        left=0.004, right=0.996, top=0.98, bottom=0.02,
        wspace=0.02, hspace=0.10,
    )

    for row_index, row in enumerate(rows):
        prompt_axes = fig.add_subplot(grid[row_index, 0])
        prompt_axes.axis("off")
        prompt_axes.text(
            0.0, 0.5,
            f"{row['role']}\n{textwrap.fill(row['prompt'], width=26)}",
            ha="left", va="center", transform=prompt_axes.transAxes,
            fontsize=7.5, color=PROMPT_COLOR, linespacing=1.45,
        )
        for column_index, image in enumerate(_read_frames(Path(row["video"]), frames), start=1):
            axes = fig.add_subplot(grid[row_index, column_index])
            axes.imshow(image)
            axes.set_axis_off()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=dpi)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-a-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "pair_A")
    parser.add_argument("--pair-b-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "pair_B")
    parser.add_argument("--rows-json", type=Path,
                        help='Explicit row list: [{"role":..., "prompt":..., "video":...}, ...]')
    parser.add_argument("--frames", type=int, nargs="+", default=list(DEFAULT_FRAMES))
    parser.add_argument("--output", type=Path,
                        default=DEFAULT_OUTPUT_ROOT / "figures" / "chorus_counterexample_p1.png")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.rows_json is not None:
        rows = json.loads(args.rows_json.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise SystemExit(f"rows JSON must be a list: {args.rows_json}")
    else:
        rows = _rows_from_artifacts(args.pair_a_root.expanduser().resolve(),
                                    args.pair_b_root.expanduser().resolve())
    render(rows, tuple(int(item) for item in args.frames), args.output.expanduser().resolve(), args.dpi)
    print(json.dumps({"status": "completed", "output": str(args.output), "rows": len(rows),
                      "frames": list(args.frames)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
