#!/usr/bin/env python3
"""Generate Carnator's local spaCy ``NOUN``/``PROPN`` annotations.

This tool is intentionally self-contained and uses no sibling-project files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from carnator.retrieval import canonical_nouns  # noqa: E402
from core.retrieval import SpacyEntityExtractor  # noqa: E402


# spaCy tags the noun in the one-token prompt "a remote" as ADJ. Keep this
# dataset correction explicit instead of broadening Carnator's POS policy.
NOUNLESS_OVERRIDES = {"object_class_65": ("remote",)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def annotate(uuid: str, prompt: str, extractor: SpacyEntityExtractor) -> str:
    values = canonical_nouns(extractor.extract(prompt))
    if not values:
        values = canonical_nouns(NOUNLESS_OVERRIDES.get(uuid, ()))
    if not values:
        raise ValueError("spaCy produced no NOUN/PROPN tokens")
    return ";".join(values)


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
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    args = parser.parse_args()
    provenance = args.provenance or args.output.with_suffix(".provenance.json")
    with args.input.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 694:
        raise SystemExit(f"expected 694 source requests, found {len(rows)}")
    required = {"request_index", "uuid", "prompt", "dimension"}
    if required - set(rows[0]):
        raise SystemExit(f"input CSV missing columns: {sorted(required - set(rows[0]))}")
    extractor = SpacyEntityExtractor(args.spacy_model, allowed_pos=("NOUN", "PROPN"))
    config = {
        "version": "carnator-trf-entities-v2",
        "extractor": extractor.signature,
        "nounless_overrides": NOUNLESS_OVERRIDES,
        "separator": ";",
    }
    output_rows = [
        {**{key: row[key] for key in required}, "trf_nouns": annotate(row["uuid"], row["prompt"], extractor)}
        for row in rows
    ]
    output_rows.sort(key=lambda row: int(row["request_index"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["request_index", "uuid", "prompt", "dimension", "trf_nouns"],
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    if len(output_rows) != 694 or any(not row["trf_nouns"] for row in output_rows):
        raise SystemExit("annotation coverage/non-empty assertion failed")
    config_bytes = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    provenance.write_text(json.dumps({
        "tool": "scripts/annotate_carnator_nouns.py",
        "config": config,
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
