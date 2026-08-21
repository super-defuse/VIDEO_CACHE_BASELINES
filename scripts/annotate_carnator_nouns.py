#!/usr/bin/env python3
"""Generate Carnator's local canonical ``trf_nouns`` annotations.

This tool is intentionally self-contained and uses no sibling-project files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", re.IGNORECASE)
STOP = frozenset({
    "a", "an", "and", "as", "at", "by", "for", "from", "in", "into", "is", "of",
    "on", "or", "the", "to", "with", "without", "near", "over", "under", "while",
    "this", "that", "these", "those", "very", "featuring", "feature", "style",
})
CONFIG = {"version": "carnator-trf-nouns-v1", "tokenizer": "ascii-alnum-apostrophe",
          "stop_words": sorted(STOP), "separator": ";"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def annotate(prompt: str) -> str:
    values = sorted({token.lower() for token in TOKEN_RE.findall(prompt)
                     if token.lower() not in STOP and len(token) > 1})
    return ";".join(values or ["prompt"])


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "datasets" / "vbench_cross_request_prompts.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "datasets" / "carnator_vbench_trf_nouns.csv")
    parser.add_argument("--provenance", type=Path)
    args = parser.parse_args()
    provenance = args.provenance or args.output.with_suffix(".provenance.json")
    with args.input.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 694:
        raise SystemExit(f"expected 694 source requests, found {len(rows)}")
    required = {"request_index", "uuid", "prompt", "dimension"}
    if required - set(rows[0]):
        raise SystemExit(f"input CSV missing columns: {sorted(required - set(rows[0]))}")
    output_rows = [{**{key: row[key] for key in required}, "trf_nouns": annotate(row["prompt"])} for row in rows]
    output_rows.sort(key=lambda row: int(row["request_index"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["request_index", "uuid", "prompt", "dimension", "trf_nouns"],
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    if len(output_rows) != 694 or any(not row["trf_nouns"] for row in output_rows):
        raise SystemExit("annotation coverage/non-empty assertion failed")
    config_bytes = json.dumps(CONFIG, sort_keys=True, separators=(",", ":")).encode("utf-8")
    provenance.write_text(json.dumps({
        "tool": "scripts/annotate_carnator_nouns.py",
        "config": CONFIG,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "input": display_path(args.input),
        "input_sha256": sha256(args.input),
        "output": display_path(args.output),
        "output_sha256": sha256(args.output),
        "rows": len(output_rows),
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(output_rows), "output": str(args.output), "provenance": str(provenance)}))


if __name__ == "__main__":
    main()
