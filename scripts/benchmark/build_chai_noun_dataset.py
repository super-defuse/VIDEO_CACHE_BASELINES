#!/usr/bin/env python3
"""Create a CHAI noun-annotation CSV from a UUID/prompt CSV."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.retrieval import SpacyEntityExtractor  # noqa: E402


DEFAULT_INPUT = ROOT.parents[1] / "project" / "VidProM" / "clusters" / "cluster_02_2170.csv"
DEFAULT_OUTPUT = ROOT / "datasets" / "vidprom_cluster_02_2170_chai_nouns.csv"


def _load_rows(path: Path) -> list[dict[str, str]]:
    csv.field_size_limit(16 * 1024 * 1024)
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = {"uuid", "prompt"} - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"input CSV is missing required columns: {', '.join(sorted(missing))}")
        rows = [{"uuid": str(row.get("uuid", "")).strip(), "prompt": str(row.get("prompt", ""))} for row in reader]
    seen: set[str] = set()
    for index, row in enumerate(rows, start=2):
        if not row["uuid"]:
            raise ValueError(f"input CSV row {index} has an empty UUID")
        if row["uuid"] in seen:
            raise ValueError(f"input CSV has duplicate UUID: {row['uuid']}")
        seen.add(row["uuid"])
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dimension", default="cluster_02")
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--processes", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing output CSV: {output_path}")
    if not args.dimension:
        raise ValueError("dimension must not be empty")
    if args.batch_size <= 0 or args.processes < 1:
        raise ValueError("batch size must be positive and processes must be at least one")

    rows = _load_rows(input_path)
    extractor = SpacyEntityExtractor(args.spacy_model)
    extractor._load()
    assert extractor._nlp is not None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", newline="", encoding="utf-8", dir=output_path.parent,
        prefix=f".{output_path.name}.", suffix=".tmp", delete=False,
    ) as handle:
        staging_path = Path(handle.name)
        writer = csv.DictWriter(
            handle,
            fieldnames=("request_index", "uuid", "prompt", "dimension", "chai_nouns"),
            lineterminator="\n",
        )
        writer.writeheader()
        # This is CHAI's exact extractor: noun tokens only, original spelling,
        # original order, duplicate tokens retained, and no noun-chunk expansion.
        prompts = (row["prompt"].split("aesthetic score", 1)[0] for row in rows)
        for request_index, (row, doc) in enumerate(
            zip(rows, extractor._nlp.pipe(prompts, batch_size=args.batch_size, n_process=args.processes), strict=True)
        ):
            nouns = [token.text for token in doc if token.pos_ == "NOUN"]
            writer.writerow({
                "request_index": request_index,
                "uuid": row["uuid"],
                "prompt": row["prompt"],
                "dimension": args.dimension,
                "chai_nouns": json.dumps(nouns, ensure_ascii=False, separators=(",", ":")),
            })
    staging_path.replace(output_path)
    print(f"wrote {len(rows)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
