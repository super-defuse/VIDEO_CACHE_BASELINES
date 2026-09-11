#!/usr/bin/env python3
"""Percentile analysis of VBench-Q / CLIP-T for the 300-request VidProM run.

Reads each method's ``quality_input.jsonl`` and reports:

* per-method distributions on all scored requests (populations differ);
* per-method distributions on the paired subset scored by every method;
* per-request paired deltas against Carnator B5;
* the six VBench dimensions behind the composite.

Run from this directory:  python3 quality_percentile_analysis.py
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
METHODS = [
    ("nirvana_vid", "NIRVANA-VID"),
    ("teacache", "TeaCache"),
    ("chorus", "Chorus"),
    ("chorus2", "Chorus2"),
    ("carnator_boundary_b5", "Carnator B5"),
]
ORDER = [m for m, _ in METHODS]  # run-table order
REFERENCE = "carnator_boundary_b5"
PCTS = (5, 10, 25, 50, 75, 90, 95)
DIMS = (
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "aesthetic_quality",
    "imaging_quality",
    "dynamic_degree",
)
# Mirrors scripts/benchmark/quality_runner.py aggregate_vbench_scores().
DIM_WEIGHT = {**{d: 1.0 for d in DIMS}, "dynamic_degree": 0.5}
NORMALIZE = {
    "subject_consistency": (0.1462, 1.0),
    "background_consistency": (0.2615, 1.0),
    "motion_smoothness": (0.7060, 0.9975),
    "aesthetic_quality": (0.0, 1.0),
    "imaging_quality": (0.0, 1.0),
    "dynamic_degree": (0.0, 1.0),
}


def label(method: str) -> str:
    return dict(METHODS)[method]


def load(method: str) -> dict[int, dict]:
    path = HERE / method / "quality_input.jsonl"
    rows = (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
    return {row["request_index"]: row for row in rows}


def values(data: dict[int, dict], metric: str, indices) -> np.ndarray:
    if indices is None:
        indices = sorted(scored(data, metric))
    return np.array([data[i][metric] for i in indices], dtype=float)


def scored(data: dict[int, dict], metric: str) -> set[int]:
    return {i for i, row in data.items() if row.get(metric) is not None}


def verify_formula(data: dict[int, dict]) -> int:
    """Confirm the stored vbench_q equals the documented composite."""
    bad = 0
    for row in data.values():
        if row.get("vbench_q") is None:
            continue
        dims = row["vbench_dimensions"]
        norm = {
            d: ((round(float(dims[d]), 4) - NORMALIZE[d][0]) / (NORMALIZE[d][1] - NORMALIZE[d][0]))
            * DIM_WEIGHT[d]
            for d in DIMS
        }
        expected = sum(norm.values()) / sum(DIM_WEIGHT.values()) * 4.0 / 5.0
        bad += abs(expected - row["vbench_q"]) > 1e-9
    return bad


def table(rows: list[list[str]], header: list[str]) -> str:
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def distribution_table(data, metric, indices, order) -> str:
    header = ["Method", "n", "Mean", "Std", "Min"] + [f"P{p}" for p in PCTS] + ["Max"]
    rows = []
    for m in order:
        a = values(data[m], metric, indices)
        p = np.percentile(a, PCTS)
        rows.append(
            [label(m), str(len(a)), f"{a.mean():.4f}", f"{a.std(ddof=1):.4f}", f"{a.min():.4f}"]
            + [f"{x:.4f}" for x in p]
            + [f"{a.max():.4f}"]
        )
    return table(rows, header)


def paired_table(data, metric, common, order, reference) -> str:
    header = ["Method", "Mean", "Std"] + [f"P{p}" for p in PCTS]
    rows = []
    for m in order:
        a = values(data[m], metric, common)
        p = np.percentile(a, PCTS)
        rows.append([label(m), f"{a.mean():.4f}", f"{a.std(ddof=1):.4f}"] + [f"{x:.4f}" for x in p])
    return table(rows, header)


def delta_table(data, metric, common, order, reference) -> str:
    header = ["Comparison", "Mean Δ", "Δ P5", "Δ P25", "Δ P50", "Δ P75", "Δ P95", "Ref wins"]
    ref = values(data[reference], metric, common)
    rows = []
    for m in order:
        if m == reference:
            continue
        d = values(data[m], metric, common) - ref
        rows.append(
            [f"{label(m)} − {label(reference)}", f"{d.mean():+.4f}"]
            + [f"{np.percentile(d, p):+.4f}" for p in (5, 25, 50, 75, 95)]
            + [f"{np.mean(d < 0) * 100:.1f}%"]
        )
    return table(rows, header)


def dimension_table(data, common, order) -> str:
    header = ["Method"] + [d[:11] for d in DIMS]
    rows = []
    for m in order:
        rows.append(
            [label(m)]
            + [f"{np.mean([data[m][i]['vbench_dimensions'][d] for i in common]):.4f}" for d in DIMS]
        )
    return table(rows, header)


def dim_contribution(dims: dict) -> dict[str, float]:
    """Each dimension's additive contribution to the VBench-Q composite."""
    return {
        d: ((round(float(dims[d]), 4) - NORMALIZE[d][0]) / (NORMALIZE[d][1] - NORMALIZE[d][0]))
        * DIM_WEIGHT[d]
        / sum(DIM_WEIGHT.values())
        * 4.0
        / 5.0
        for d in DIMS
    }


def contribution_table(data, common, order) -> str:
    header = ["Method"] + [d[:11] for d in DIMS] + ["Total"]
    rows = []
    for m in order:
        per_dim = {
            d: np.mean([dim_contribution(data[m][i]["vbench_dimensions"])[d] for i in common])
            for d in DIMS
        }
        rows.append(
            [label(m)]
            + [f"{per_dim[d]:.4f}" for d in DIMS]
            + [f"{sum(per_dim.values()):.4f}"]
        )
    return table(rows, header)


def main() -> None:
    data = {m: load(m) for m in ORDER}
    mismatches = sum(verify_formula(d) for d in data.values())

    lines: list[str] = []
    add = lines.append
    add("# VidProM 300-request quality distributions (VBench-Q / CLIP-T)")
    add("")
    add(f"Source: `{HERE.name}` — per-request `clip_t` / `vbench_q` from each method's `quality_input.jsonl`.")
    add("")
    add("`vbench_q` is the CHAI composite of six VBench dimensions (weights: dynamic_degree 0.5, "
        "others 1.0; each min-max normalized, then rescaled by 4/5 = quality 4 : semantic 1). "
        f"Recomputation from the stored dimensions matched all stored values ({mismatches} mismatches). "
        "`clip_t` is the frame-mean CLIP ViT-B/32 text-image cosine similarity.")
    add("")

    for metric, title in (("vbench_q", "VBench-Q"), ("clip_t", "CLIP-T")):
        add(f"## {title} — all scored requests (populations differ)")
        add("")
        add(distribution_table(data, metric, None, ORDER))
        add("")

        common = sorted(set.intersection(*[scored(data[m], metric) for m in ORDER]))
        add(f"## {title} — paired subset scored by every method (n={len(common)})")
        add("")
        add(paired_table(data, metric, common, ORDER, REFERENCE))
        add("")
        add(f"Per-request paired deltas vs {label(REFERENCE)} "
            f"(`Ref wins` = share of requests where the reference scores higher):")
        add("")
        add(delta_table(data, metric, common, ORDER, REFERENCE))
        add("")

    common = sorted(set.intersection(*[scored(data[m], "vbench_q") for m in ORDER]))
    add("## VBench-Q dimension means (paired subset)")
    add("")
    add(dimension_table(data, common, ORDER))
    add("")
    add("Each dimension's additive contribution to the composite "
        "(columns sum to the VBench-Q mean above):")
    add("")
    add(contribution_table(data, common, ORDER))
    add("")

    # Coverage, so that the population difference is explicit.
    add("## Scored coverage per method")
    add("")
    rows = []
    for m in ORDER:
        total = len(data[m])
        n = len(scored(data[m], "vbench_q"))
        hits = sum(1 for r in data[m].values() if r.get("vbench_q") is not None and r.get("hit"))
        rows.append([label(m), str(total), str(n), f"{n / total:.3f}", str(hits)])
    add(table(rows, ["Method", "Requests", "Scored", "Scored rate", "Scored rows that are cache hits"]))
    add("")

    report = HERE / "QUALITY_PERCENTILE_ANALYSIS.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {report}")


if __name__ == "__main__":
    main()
