#!/usr/bin/env python3
"""Sample large VidProM clusters and audit Carnator's pre-signature pairs.

For each top-level ``cluster_*.csv`` with more than ``--sample-size`` rows,
this command samples a reproducible dataset and runs CHAI-compatible
``NOUN`` extraction plus Carnator's CLIP Top-K and noun prefilter.
It intentionally stops before the step-5 signature gate, so the resulting
pairs are candidates for an Early Signature comparison rather than confirmed
cache-reuse hits.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import random
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.retrieval import SpacyEntityExtractor, TransformersClipEmbedder  # noqa: E402


DEFAULT_INPUT_DIR = ROOT.parents[1] / "project" / "VidProM" / "clusters"
DEFAULT_OUTPUT = ROOT / "artifacts" / "vidprom_cluster_carnator_noun_pairs"
SCHEMA_VERSION = "vidprom-cluster-carnator-noun-pairs/v2"
PAIR_TYPES = ("same_noun_set", "single_noun_substitution")


@dataclass(frozen=True)
class PromptRow:
    uuid: str
    prompt: str
    source_row_index: int


@dataclass(frozen=True)
class PromptCandidate:
    uuid: str
    prompt: str
    sample_index: int
    source_row_index: int
    nouns: tuple[str, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _dataset_seed(seed: int, filename: str) -> int:
    material = f"{seed}:{filename}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _read_cluster(path: Path) -> list[PromptRow]:
    csv.field_size_limit(16 * 1024 * 1024)
    rows: list[PromptRow] = []
    seen_uuids: set[str] = set()
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        headers = set(reader.fieldnames or ())
        missing = {"uuid", "prompt"} - headers
        if missing:
            raise ValueError(f"{path} is missing required columns: {', '.join(sorted(missing))}")
        for source_row_index, row in enumerate(reader):
            uuid = str(row.get("uuid", "")).strip()
            if not uuid:
                raise ValueError(f"{path} row {source_row_index + 2} has an empty UUID")
            if uuid in seen_uuids:
                raise ValueError(f"{path} has duplicate UUID: {uuid}")
            seen_uuids.add(uuid)
            rows.append(PromptRow(uuid, str(row.get("prompt", "")), source_row_index))
    return rows


def _sample_rows(rows: list[PromptRow], *, sample_size: int, seed: int) -> list[PromptRow]:
    if len(rows) <= sample_size:
        raise ValueError(f"cluster has {len(rows)} rows, which is not greater than sample size {sample_size}")
    selected = random.Random(seed).sample(range(len(rows)), sample_size)
    # Preserve the original order after sampling. It becomes the deterministic
    # record order used to resolve the exceptionally rare equal-score tie.
    return [rows[index] for index in sorted(selected)]


def _write_csv(path: Path, fieldnames: Iterable[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _noun_relation(source: Iterable[str], target: Iterable[str]) -> tuple[str, str, str] | None:
    source_nouns, target_nouns = set(source), set(target)
    if source_nouns == target_nouns:
        return "same_noun_set", "", ""
    source_only = sorted(source_nouns - target_nouns)
    target_only = sorted(target_nouns - source_nouns)
    if len(source_only) == len(target_only) == 1:
        return "single_noun_substitution", source_only[0], target_only[0]
    return None


def _extract_nouns(
    rows: list[PromptRow],
    *,
    extractor: SpacyEntityExtractor,
    batch_size: int,
    processes: int,
) -> tuple[list[PromptCandidate], list[dict[str, Any]]]:
    extractor._load()
    assert extractor._nlp is not None
    candidates: list[PromptCandidate] = []
    nounless_rows: list[dict[str, Any]] = []
    # This is the same prompt cleanup and POS policy as SpacyEntityExtractor.extract.
    prompts = (row.prompt.split("aesthetic score", 1)[0] for row in rows)
    for sample_index, (row, doc) in enumerate(
        zip(rows, extractor._nlp.pipe(prompts, batch_size=batch_size, n_process=processes), strict=True)
    ):
        nouns = tuple(token.text for token in doc if token.pos_ in extractor.allowed_pos)
        if not nouns:
            nounless_rows.append({
                "sample_index": sample_index,
                "source_row_index": row.source_row_index,
                "uuid": row.uuid,
                "prompt": row.prompt,
                "reason": "noun_extraction_produced_no_NOUN",
            })
            continue
        candidates.append(PromptCandidate(row.uuid, row.prompt, sample_index, row.source_row_index, nouns))
    return candidates, nounless_rows


def _embed_candidates(
    candidates: list[PromptCandidate],
    *,
    embedder: TransformersClipEmbedder,
    batch_size: int,
) -> np.ndarray:
    if not candidates:
        return np.empty((0, 0), dtype=np.float32)
    batches: list[np.ndarray] = []
    for start in range(0, len(candidates), batch_size):
        end = min(start + batch_size, len(candidates))
        features = np.asarray(embedder.encode([candidate.prompt for candidate in candidates[start:end]]), dtype=np.float32)
        if features.ndim != 2 or features.shape[0] != end - start:
            raise RuntimeError(f"unexpected CLIP feature shape {features.shape} for candidates {start}:{end}")
        batches.append(features)
    return np.concatenate(batches, axis=0)


def _empty_target_audit(row: PromptRow, sample_index: int) -> dict[str, Any]:
    return {
        "sample_index": sample_index,
        "source_row_index": row.source_row_index,
        "target_uuid": row.uuid,
        "target_prompt": row.prompt,
        "target_nouns": "",
        "noun_extraction_status": "rejected_no_noun",
        "stage1_recalled_candidates": 0,
        "noun_compatible_candidates_before_cap": 0,
        "stage2_candidate_pairs": 0,
        "has_stage2_candidate": False,
        "same_noun_candidates_before_cap": 0,
        "single_noun_substitution_candidates_before_cap": 0,
        "selected_source_uuid": "",
        "selected_recall_score": "",
        "selected_pair_type": "",
    }


def _analyze_candidates(
    rows: list[PromptRow],
    candidates: list[PromptCandidate],
    embeddings: np.ndarray,
    *,
    device: str,
    top_k: int,
    min_similarity: float,
    stage2_top_k: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    import torch

    by_sample_index = {candidate.sample_index: candidate for candidate in candidates}
    audits = [_empty_target_audit(row, sample_index) for sample_index, row in enumerate(rows)]
    pairs: list[dict[str, Any]] = []
    statistics = {
        "stage1_recalled_candidate_pairs": 0,
        "noun_compatible_candidate_pairs_before_cap": 0,
        "stage2_candidate_pairs": 0,
        "stage2_same_noun_set_pairs": 0,
        "stage2_single_noun_substitution_pairs": 0,
        "targets_with_stage2_candidate": 0,
        "targets_with_same_noun_candidate": 0,
        "targets_with_single_noun_substitution_candidate": 0,
    }
    if len(candidates) < 2:
        return audits, pairs, statistics

    candidate_matrix = torch.from_numpy(embeddings).to(device=device, dtype=torch.float32)
    scores = candidate_matrix @ candidate_matrix.T
    scores.fill_diagonal_(float("-inf"))
    # stable=True retains sample order as the tie break, matching Carnator's
    # manifest record order for equal cosine scores.
    ranked_indices = torch.argsort(scores, dim=1, descending=True, stable=True)[:, : min(top_k, len(candidates) - 1)]
    ranked_scores = scores.gather(1, ranked_indices)

    for target_index, target in enumerate(candidates):
        audit = audits[target.sample_index]
        audit.update({
            "target_nouns": ";".join(target.nouns),
            "noun_extraction_status": "accepted",
        })
        passed: list[tuple[int, float, PromptCandidate, tuple[str, str, str]]] = []
        stage1_count = 0
        type_counts = {pair_type: 0 for pair_type in PAIR_TYPES}
        for rank, (source_index, score) in enumerate(
            zip(ranked_indices[target_index].tolist(), ranked_scores[target_index].tolist(), strict=True), start=1
        ):
            if not math.isfinite(score) or score < min_similarity:
                continue
            stage1_count += 1
            source = candidates[source_index]
            relation = _noun_relation(source.nouns, target.nouns)
            if relation is None:
                continue
            passed.append((rank, float(score), source, relation))
            type_counts[relation[0]] += 1

        statistics["stage1_recalled_candidate_pairs"] += stage1_count
        statistics["noun_compatible_candidate_pairs_before_cap"] += len(passed)
        stage2_candidates = passed[:stage2_top_k]
        audit.update({
            "stage1_recalled_candidates": stage1_count,
            "noun_compatible_candidates_before_cap": len(passed),
            "stage2_candidate_pairs": len(stage2_candidates),
            "has_stage2_candidate": bool(stage2_candidates),
            "same_noun_candidates_before_cap": type_counts["same_noun_set"],
            "single_noun_substitution_candidates_before_cap": type_counts["single_noun_substitution"],
        })
        if stage2_candidates:
            statistics["targets_with_stage2_candidate"] += 1
        if type_counts["same_noun_set"]:
            statistics["targets_with_same_noun_candidate"] += 1
        if type_counts["single_noun_substitution"]:
            statistics["targets_with_single_noun_substitution_candidate"] += 1

        for stage2_rank, (stage1_rank, score, source, relation) in enumerate(stage2_candidates, start=1):
            pair_type, source_changed_noun, target_changed_noun = relation
            statistics["stage2_candidate_pairs"] += 1
            statistics[f"stage2_{pair_type}_pairs"] += 1
            pairs.append({
                "target_sample_index": target.sample_index,
                "target_source_row_index": target.source_row_index,
                "target_uuid": target.uuid,
                "target_prompt": target.prompt,
                "target_nouns": ";".join(target.nouns),
                "stage1_rank": stage1_rank,
                "stage2_rank": stage2_rank,
                "source_sample_index": source.sample_index,
                "source_source_row_index": source.source_row_index,
                "source_uuid": source.uuid,
                "source_prompt": source.prompt,
                "source_nouns": ";".join(source.nouns),
                "recall_score": f"{score:.8f}",
                "pair_type": pair_type,
                "source_changed_noun": source_changed_noun,
                "target_changed_noun": target_changed_noun,
            })
            if stage2_rank == 1:
                audit.update({
                    "selected_source_uuid": source.uuid,
                    "selected_recall_score": f"{score:.8f}",
                    "selected_pair_type": pair_type,
                })
    del ranked_scores, ranked_indices, scores, candidate_matrix
    return audits, pairs, statistics


def _write_report(path: Path, datasets: list[dict[str, Any]], *, args: argparse.Namespace) -> None:
    lines = [
        "# VidProM Cluster Carnator Noun-Pair Audit",
        "",
        "`stage2_candidate_pairs` is the usable noun-pair count: CLIP Top-20 candidates "
        f"with cosine similarity >= {args.min_similarity:.2f}, followed by Carnator's noun prefilter and "
        f"per-target Top-{args.stage2_top_k} cap. These are pre-signature candidates only; no step-5 "
        "signature comparison or video/cache reuse was run.",
        "",
        "| dataset | sampled prompts | noun-valid prompts | stage-1 recalled pairs | noun-compatible before cap | usable noun pairs | targets with a pair | same-noun pairs | one-noun-substitution pairs |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset in datasets:
        stats = dataset["statistics"]
        lines.append(
            f"| {dataset['dataset']} | {dataset['sampled_rows']} | {dataset['noun_valid_rows']} | "
            f"{stats['stage1_recalled_candidate_pairs']} | {stats['noun_compatible_candidate_pairs_before_cap']} | "
            f"{stats['stage2_candidate_pairs']} | {stats['targets_with_stage2_candidate']} | "
            f"{stats['stage2_same_noun_set_pairs']} | {stats['stage2_single_noun_substitution_pairs']} |"
        )
    lines.extend([
        "",
        "The samples are in `samples/`, the exact capped candidate pairs are in `pairs/`, and every target's "
        "retrieval outcome is in `target_audits/`. Prompts with no `NOUN`/`PROPN` are in `nounless_prompts/` "
        "and are excluded from both source and target retrieval because Carnator cannot form their noun set.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument("--sample-seed", type=int, default=20260823)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    parser.add_argument("--clip-model", default="openai/clip-vit-large-patch14")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-similarity", type=float, default=0.65)
    parser.add_argument("--stage2-top-k", type=int, default=5)
    parser.add_argument("--spacy-batch-size", type=int, default=512)
    parser.add_argument("--spacy-processes", type=int, default=1)
    parser.add_argument("--clip-batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output}")
    if args.sample_size <= 0 or args.top_k <= 0 or args.stage2_top_k <= 0:
        raise ValueError("sample size and Top-K values must be positive")
    if args.spacy_batch_size <= 0 or args.clip_batch_size <= 0 or args.spacy_processes < 1:
        raise ValueError("batch sizes must be positive and spaCy processes must be at least one")
    if not -1.0 <= args.min_similarity <= 1.0:
        raise ValueError("min similarity must be between -1 and 1")

    clusters = []
    for path in sorted(input_dir.glob("cluster_*.csv")):
        rows = _read_cluster(path)
        if len(rows) > args.sample_size:
            clusters.append((path, rows))
    if not clusters:
        raise ValueError(f"no cluster CSV under {input_dir} contains more than {args.sample_size} rows")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        for directory in ("samples", "pairs", "target_audits", "nounless_prompts"):
            (staging / directory).mkdir()

        extractor = SpacyEntityExtractor(args.spacy_model)
        embedder = TransformersClipEmbedder(args.clip_model, args.device)
        reports: list[dict[str, Any]] = []
        for input_path, source_rows in clusters:
            dataset_name = input_path.stem
            dataset_seed = _dataset_seed(args.sample_seed, input_path.name)
            sampled = _sample_rows(source_rows, sample_size=args.sample_size, seed=dataset_seed)
            print(f"{dataset_name}: sampled {len(sampled):,} of {len(source_rows):,} rows", flush=True)
            _write_csv(
                staging / "samples" / f"{dataset_name}_sample_{args.sample_size}.csv",
                ("uuid", "prompt"),
                ({"uuid": row.uuid, "prompt": row.prompt} for row in sampled),
            )
            candidates, nounless = _extract_nouns(
                sampled,
                extractor=extractor,
                batch_size=args.spacy_batch_size,
                processes=args.spacy_processes,
            )
            print(f"{dataset_name}: noun-valid prompts {len(candidates):,}", flush=True)
            embeddings = _embed_candidates(candidates, embedder=embedder, batch_size=args.clip_batch_size)
            audits, pairs, statistics = _analyze_candidates(
                sampled,
                candidates,
                embeddings,
                device=args.device,
                top_k=args.top_k,
                min_similarity=args.min_similarity,
                stage2_top_k=args.stage2_top_k,
            )
            del embeddings
            _write_csv(
                staging / "pairs" / f"{dataset_name}_stage2_candidates.csv",
                (
                    "target_sample_index", "target_source_row_index", "target_uuid", "target_prompt", "target_nouns",
                    "stage1_rank", "stage2_rank", "source_sample_index", "source_source_row_index", "source_uuid",
                    "source_prompt", "source_nouns", "recall_score", "pair_type", "source_changed_noun",
                    "target_changed_noun",
                ),
                pairs,
            )
            _write_csv(
                staging / "target_audits" / f"{dataset_name}_targets.csv",
                (
                    "sample_index", "source_row_index", "target_uuid", "target_prompt", "target_nouns",
                    "noun_extraction_status", "stage1_recalled_candidates", "noun_compatible_candidates_before_cap",
                    "stage2_candidate_pairs", "has_stage2_candidate", "same_noun_candidates_before_cap",
                    "single_noun_substitution_candidates_before_cap", "selected_source_uuid", "selected_recall_score",
                    "selected_pair_type",
                ),
                audits,
            )
            _write_csv(
                staging / "nounless_prompts" / f"{dataset_name}_rejected.csv",
                ("sample_index", "source_row_index", "uuid", "prompt", "reason"),
                nounless,
            )
            reports.append({
                "dataset": dataset_name,
                "input_file": str(input_path),
                "input_sha256": _sha256(input_path),
                "source_rows": len(source_rows),
                "sampled_rows": len(sampled),
                "sample_seed": dataset_seed,
                "noun_valid_rows": len(candidates),
                "nounless_rows": len(nounless),
                "statistics": statistics,
            })
            print(
                f"{dataset_name}: {statistics['stage2_candidate_pairs']:,} stage-2 noun pairs across "
                f"{statistics['targets_with_stage2_candidate']:,} targets",
                flush=True,
            )

        extractor._load()
        assert extractor._nlp is not None
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "Static pre-signature Carnator candidate-pair audit; not an online replay or reuse-hit result.",
            "sampling": {
                "rule": f"top-level cluster_*.csv with rows > {args.sample_size}",
                "sample_size": args.sample_size,
                "base_seed": args.sample_seed,
                "per_dataset_seed": "sha256(f'{base_seed}:{filename}')[:8] interpreted as big-endian integer",
                "output_order": "original source CSV row order after random selection",
            },
            "noun_extractor": {
                "implementation": "core.retrieval.SpacyEntityExtractor",
                "model": args.spacy_model,
                "model_version": extractor._nlp.meta.get("version", "unknown"),
                "spacy_version": _distribution_version("spacy"),
                "allowed_pos": ["NOUN"],
                "token_policy": "CHAI-compatible raw NOUN tokens",
                "prompt_cleanup": "split before first literal 'aesthetic score'",
            },
            "retrieval": {
                "embedding": f"clip:{args.clip_model}",
                "device": args.device,
                "candidate_top_k": args.top_k,
                "candidate_min_similarity": args.min_similarity,
                "stage2_candidate_top_k": args.stage2_top_k,
                "noun_prefilter": "same NOUN token set OR exactly one noun substitution",
                "remaining_gate": "Early Signature and cache-artifact checks were not run",
            },
            "datasets": reports,
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        _write_csv(
            staging / "summary.csv",
            (
                "dataset", "source_rows", "sampled_rows", "noun_valid_rows", "nounless_rows",
                "stage1_recalled_candidate_pairs", "noun_compatible_candidate_pairs_before_cap",
                "stage2_candidate_pairs", "targets_with_stage2_candidate", "stage2_same_noun_set_pairs",
                "stage2_single_noun_substitution_pairs", "targets_with_same_noun_candidate",
                "targets_with_single_noun_substitution_candidate",
            ),
            ({
                "dataset": report["dataset"],
                "source_rows": report["source_rows"],
                "sampled_rows": report["sampled_rows"],
                "noun_valid_rows": report["noun_valid_rows"],
                "nounless_rows": report["nounless_rows"],
                **report["statistics"],
            } for report in reports),
        )
        _write_report(staging / "report.md", reports, args=args)
        shutil.move(str(staging), str(output))
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f"wrote {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
