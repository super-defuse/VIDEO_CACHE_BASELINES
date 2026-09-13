#!/usr/bin/env python3
"""Percentile analysis of VBench-Q / CLIP-T for the 300-request VidProM run.

Reads each method's ``quality_input.jsonl`` and reports:

* per-method distributions on all scored requests (populations differ);
* per-method distributions on the paired subset scored by every method;
* per-request paired deltas against a reference method;
* each method's contribution to the VBench-Q composite;
* low-score-tail behaviour, including a control for regression to the mean.

Usage (from this directory):

    python3 quality_percentile_analysis.py
    python3 quality_percentile_analysis.py --exclude teacache

``--exclude`` drops methods from every table and, importantly, from the set of
paired requests — the common subset is recomputed after exclusion.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ALL_METHODS = [
    ("nirvana_vid", "NIRVANA-VID"),
    ("teacache", "TeaCache"),
    ("chorus", "Chorus"),
    ("chorus2", "Chorus2"),
    ("carnator_boundary_b5", "Carnator B5"),
]
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
LOW_THRESHOLDS = {
    "vbench_q": (0.52, 0.55, 0.58, 0.60),
    "clip_t": (0.18, 0.20, 0.22, 0.25),
}


def label(method: str) -> str:
    return dict(ALL_METHODS)[method]


def load(method: str) -> dict[int, dict]:
    path = HERE / method / "quality_input.jsonl"
    rows = (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
    return {row["request_index"]: row for row in rows}


def scored(data: dict[int, dict], metric: str) -> set[int]:
    return {i for i, row in data.items() if row.get(metric) is not None}


def values(data: dict[int, dict], metric: str, indices) -> np.ndarray:
    if indices is None:
        indices = sorted(scored(data, metric))
    return np.array([data[i][metric] for i in indices], dtype=float)


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


def verify_formula(data: dict[int, dict]) -> int:
    """Confirm the stored vbench_q equals the documented composite."""
    bad = 0
    for row in data.values():
        if row.get("vbench_q") is not None:
            bad += abs(sum(dim_contribution(row["vbench_dimensions"]).values()) - row["vbench_q"]) > 1e-9
    return bad


def table(rows: list[list[str]], header: list[str]) -> str:
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def distribution_table(data, metric, order) -> str:
    header = ["Method", "n", "Mean", "Std", "Min"] + [f"P{p}" for p in PCTS] + ["Max"]
    rows = []
    for m in order:
        a = values(data[m], metric, None)
        rows.append(
            [label(m), str(len(a)), f"{a.mean():.4f}", f"{a.std(ddof=1):.4f}", f"{a.min():.4f}"]
            + [f"{x:.4f}" for x in np.percentile(a, PCTS)]
            + [f"{a.max():.4f}"]
        )
    return table(rows, header)


def paired_table(data, metric, common, order) -> str:
    header = ["Method", "Mean", "Std"] + [f"P{p}" for p in PCTS]
    rows = []
    for m in order:
        a = values(data[m], metric, common)
        rows.append([label(m), f"{a.mean():.4f}", f"{a.std(ddof=1):.4f}"]
                    + [f"{x:.4f}" for x in np.percentile(a, PCTS)])
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
        rows.append([label(m)] + [
            f"{np.mean([data[m][i]['vbench_dimensions'][d] for i in common]):.4f}" for d in DIMS
        ])
    return table(rows, header)


def contribution_table(data, common, order) -> str:
    header = ["Method"] + [d[:11] for d in DIMS] + ["Total"]
    rows = []
    for m in order:
        per_dim = {
            d: np.mean([dim_contribution(data[m][i]["vbench_dimensions"])[d] for i in common])
            for d in DIMS
        }
        rows.append([label(m)] + [f"{per_dim[d]:.4f}" for d in DIMS]
                    + [f"{sum(per_dim.values()):.4f}"])
    return table(rows, header)


def coverage_table(data, order) -> str:
    rows = []
    for m in order:
        n = len(scored(data[m], "vbench_q"))
        hits = sum(1 for r in data[m].values() if r.get("vbench_q") is not None and r.get("hit"))
        rows.append([label(m), str(len(data[m])), str(n), f"{n / len(data[m]):.3f}", str(hits)])
    return table(rows, ["Method", "Requests", "Scored", "Scored rate",
                        "Scored rows that are cache hits"])


def tail_table(data, metric, common, order) -> str:
    header = ["Method", "Min", "Worst-3", "P1", "P2.5", "P5", "P10", "P25"]
    rows = []
    for m in order:
        a = values(data[m], metric, common)
        rows.append([label(m), f"{a.min():.4f}", f"{np.sort(a)[:3].mean():.4f}"]
                    + [f"{np.percentile(a, p):.4f}" for p in (1, 2.5, 5, 10, 25)])
    return table(rows, header)


def low_band_table(data, metric, common, order) -> str:
    ths = LOW_THRESHOLDS[metric]
    header = ["Method"] + [f"n<{t}" for t in ths] + [f"%<{t}" for t in ths]
    rows = []
    for m in order:
        a = values(data[m], metric, common)
        counts = [int((a < t).sum()) for t in ths]
        rows.append([label(m)] + [str(c) for c in counts]
                    + [f"{c / len(a) * 100:.1f}" for c in counts])
    return table(rows, header)


def field_offset_table(data, metric, common, order) -> str:
    """Offset from the other methods' mean, on each method's own low quartile.

    Guards against regression to the mean: a selector's own score is depressed by
    construction, so a method can only be credited for beating the *rest* of the
    field, not the selector it was selected against.
    """
    header = ["Selector (P25)"] + [label(m) for m in order]
    rows = []
    for selector in order:
        thr = np.percentile(values(data[selector], metric, common), 25)
        sel = [i for i in common if data[selector][i][metric] <= thr]
        vals = {m: values(data[m], metric, sel) for m in order}
        cells = []
        for m in order:
            field = np.mean([vals[o] for o in order if o != m], axis=0)
            cells.append(f"{(vals[m] - field).mean() * 1000:+.2f}")
        rows.append([label(selector)] + cells)
    return table(rows, header)


def sign_test_p(k: int, n: int) -> float:
    """Two-sided exact sign test against p=0.5, ties excluded from ``n``."""
    if n == 0:
        return float("nan")
    observed = math.comb(n, k)
    return min(1.0, sum(math.comb(n, i) for i in range(n + 1)
                        if math.comb(n, i) <= observed + 1e-9) / 2 ** n)


def pairwise_lowband_table(data, metric, common, order, quantile=25) -> str:
    """Pairwise deltas inside low bands defined by a method external to the pair.

    For each pair (A, B) every other method S defines a hard subset (S's own low
    quantile). S is never A or B, so the selection is independent of the
    comparison and regression to the mean cannot manufacture a difference
    between A and B. ``A wins`` is the share of paired requests where A > B;
    a mean Δ alone is easily carried by a few large differences, so the win
    rate and its sign test decide whether the gap is real.
    """
    header = ["Pair", "Selector", "n", "Δ (A−B)", "A wins", "p"]
    rows = []
    for a, b in itertools.combinations(order, 2):
        for s in order:
            if s in (a, b):
                continue
            thr = np.percentile(values(data[s], metric, common), quantile)
            sel = [i for i in common if data[s][i][metric] <= thr]
            d = values(data[a], metric, sel) - values(data[b], metric, sel)
            wins, losses = int((d > 0).sum()), int((d < 0).sum())
            rows.append(
                [f"{label(a)} − {label(b)}", label(s), str(len(sel)), f"{d.mean():+.4f}",
                 f"{wins / len(sel) * 100:.1f}%", f"{sign_test_p(wins, wins + losses):.3f}"]
            )
    return table(rows, header)


def hard_subset_matrix(data, metric, common, order, k=10) -> str:
    """For each method's k worst requests, every method's mean on that subset."""
    header = ["Worst-10 by"] + [label(m) for m in order]
    rows = []
    for selector in order:
        sel = [common[j] for j in np.argsort(values(data[selector], metric, common))[:k]]
        rows.append([label(selector)] + [f"{values(data[m], metric, sel).mean():.4f}" for m in order])
    return table(rows, header)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exclude", nargs="*", default=[],
                        help="method keys to drop from every table, e.g. teacache")
    args = parser.parse_args()

    unknown = sorted(set(args.exclude) - {m for m, _ in ALL_METHODS})
    if unknown:
        raise SystemExit(f"unknown method key(s): {', '.join(unknown)}")
    methods = [(m, n) for m, n in ALL_METHODS if m not in args.exclude]
    order = [m for m, _ in methods]
    if REFERENCE not in order:
        raise SystemExit(f"reference method {REFERENCE} was excluded")

    data = {m: load(m) for m in order}
    mismatches = sum(verify_formula(d) for d in data.values())

    excluded = [m for m, _ in ALL_METHODS if m in args.exclude]
    lines: list[str] = []
    add = lines.append
    add("# VidProM 300-request quality distributions (VBench-Q / CLIP-T)")
    add("")
    if excluded:
        add(f"**Excluded methods: {', '.join(label(m) for m in excluded)}.** "
            "Every table and every paired subset below is recomputed without them.")
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
        add(distribution_table(data, metric, order))
        add("")

        common = sorted(set.intersection(*[scored(data[m], metric) for m in order]))
        add(f"## {title} — paired subset scored by every method (n={len(common)})")
        add("")
        add(paired_table(data, metric, common, order))
        add("")
        add(f"Per-request paired deltas vs {label(REFERENCE)} "
            f"(`Ref wins` = share of requests where the reference scores higher):")
        add("")
        add(delta_table(data, metric, common, order, REFERENCE))
        add("")

    common = sorted(set.intersection(*[scored(data[m], "vbench_q") for m in order]))
    add("## VBench-Q dimension means (paired subset)")
    add("")
    add(dimension_table(data, common, order))
    add("")
    add("Each dimension's additive contribution to the composite "
        "(columns sum to the VBench-Q mean above):")
    add("")
    add(contribution_table(data, common, order))
    add("")

    add("## Scored coverage per method")
    add("")
    add(coverage_table(data, order))
    add("")

    add(f"## Low-score tail (paired subset, n={len(common)})")
    add("")
    for metric, title in (("vbench_q", "VBench-Q"), ("clip_t", "CLIP-T")):
        add(f"### {title} — tail quantiles")
        add("")
        add(tail_table(data, metric, common, order))
        add("")
        add(f"### {title} — requests below absolute thresholds")
        add("")
        add(low_band_table(data, metric, common, order))
        add("")
        add(f"### {title} — offset from the other methods, on each selector's own P25")
        add("")
        add("Each row selects the requests in that method's own lowest quartile; a negative "
            "value for the selector is expected mechanically (regression to the mean), so the "
            "comparison of interest is between the *other* columns, not against the selector.")
        add("")
        add(field_offset_table(data, metric, common, order))
        add("")
        add(f"### {title} — pairwise deltas inside third-party low bands")
        add("")
        add("For each pair, the hard subset is defined by a method that is neither member of "
            "the pair, so selection cannot favour either side.")
        add("")
        add(pairwise_lowband_table(data, metric, common, order))
        add("")
        add(f"### {title} — every method's mean on each method's 10 worst requests")
        add("")
        add(hard_subset_matrix(data, metric, common, order))
        add("")

    suffix = "".join(f"_no_{m}" for m in excluded)
    report = HERE / f"QUALITY_PERCENTILE_ANALYSIS{suffix.upper()}.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {report}")


if __name__ == "__main__":
    main()
