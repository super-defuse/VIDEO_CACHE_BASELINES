#!/usr/bin/env python3
"""Freeze 60 single-noun-substitution Carnator pairs from one VidProM cluster."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
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


DEFAULT_INPUT = ROOT.parent / "VidProM" / "clusters" / "cluster_02_2170.csv"
DEFAULT_OUTPUT = ROOT / "artifacts" / "early_signature_calibration_cluster_02_2170"
SCHEMA_VERSION = "carnator-early-signature-cluster-calibration-dataset"


@dataclass(frozen=True)
class PromptRow:
    uuid: str
    prompt: str
    source_row_index: int


@dataclass(frozen=True)
class PromptCandidate:
    uuid: str
    prompt: str
    source_row_index: int
    noun_tokens: tuple[str, ...]

    @property
    def noun_set(self) -> frozenset[str]:
        # Carnator's current prefilter calls set() over CHAI's raw NOUN tokens.
        return frozenset(self.noun_tokens)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _read_rows(path: Path) -> list[PromptRow]:
    csv.field_size_limit(16 * 1024 * 1024)
    rows: list[PromptRow] = []
    seen_uuids: set[str] = set()
    seen_prompts: set[str] = set()
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = {"uuid", "prompt"} - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"input CSV is missing required columns: {', '.join(sorted(missing))}")
        for source_row_index, row in enumerate(reader):
            uuid = str(row.get("uuid", "")).strip()
            prompt = str(row.get("prompt", "")).strip()
            if not uuid:
                raise ValueError(f"input row {source_row_index + 2} has an empty UUID")
            if uuid in seen_uuids:
                raise ValueError(f"input CSV has duplicate UUID: {uuid}")
            seen_uuids.add(uuid)
            if prompt in seen_prompts:
                # The calibration protocol requires 120 distinct prompts. It
                # is cleaner to keep one deterministic copy out of the pool.
                continue
            seen_prompts.add(prompt)
            rows.append(PromptRow(uuid, prompt, source_row_index))
    return rows


def _extract_candidates(
    rows: list[PromptRow],
    *,
    model: str,
    batch_size: int,
    processes: int,
    min_unique_nouns: int,
) -> tuple[list[PromptCandidate], list[dict[str, Any]], dict[str, Any]]:
    extractor = SpacyEntityExtractor(model)
    extractor._load()
    assert extractor._nlp is not None
    candidates: list[PromptCandidate] = []
    rejected: list[dict[str, Any]] = []
    prompts = (row.prompt.split("aesthetic score", 1)[0] for row in rows)
    for row, doc in zip(rows, extractor._nlp.pipe(prompts, batch_size=batch_size, n_process=processes), strict=True):
        # Match CarnatorNounExtractor exactly: CHAI's NOUN-only raw tokens,
        # retaining original spelling, order, and duplicate tokens.
        noun_tokens = tuple(token.text for token in doc if token.pos_ == "NOUN")
        unique_nouns = frozenset(noun_tokens)
        if not noun_tokens:
            reason = "nounless_prompt"
        elif len(unique_nouns) < min_unique_nouns:
            reason = "unique_noun_count_below_minimum"
        else:
            candidates.append(PromptCandidate(row.uuid, row.prompt, row.source_row_index, noun_tokens))
            continue
        rejected.append({
            "uuid": row.uuid,
            "prompt": row.prompt,
            "source_row_index": row.source_row_index,
            "noun_tokens": list(noun_tokens),
            "noun_set": sorted(unique_nouns),
            "reason": reason,
        })
    candidates.sort(key=lambda candidate: candidate.uuid)
    return candidates, rejected, dict(extractor._nlp.meta)


def _embed_candidates(
    candidates: list[PromptCandidate],
    *,
    model_id: str,
    device: str,
    batch_size: int,
) -> np.ndarray:
    embedder = TransformersClipEmbedder(model_id, device)
    batches: list[np.ndarray] = []
    for start in range(0, len(candidates), batch_size):
        end = min(start + batch_size, len(candidates))
        vectors = np.asarray(
            embedder.encode([candidate.prompt for candidate in candidates[start:end]]), dtype=np.float32
        )
        if vectors.ndim != 2 or vectors.shape[0] != end - start:
            raise RuntimeError(f"unexpected CLIP feature shape {vectors.shape} for candidates {start}:{end}")
        batches.append(vectors)
    if not batches:
        raise ValueError("no candidates remain after noun filtering")
    return np.concatenate(batches, axis=0)


def _single_substitution(
    source: PromptCandidate, target: PromptCandidate,
) -> tuple[str, str] | None:
    source_set, target_set = source.noun_set, target.noun_set
    if len(source_set) != len(target_set) or source_set == target_set:
        return None
    source_only = sorted(source_set - target_set)
    target_only = sorted(target_set - source_set)
    if len(source_only) == len(target_only) == 1:
        return source_only[0], target_only[0]
    return None


def _noun_record(candidate: PromptCandidate) -> dict[str, Any]:
    return {
        "noun_tokens": list(candidate.noun_tokens),
        "noun_set": sorted(candidate.noun_set),
        "unique_noun_count": len(candidate.noun_set),
    }


def _select_pairs(
    candidates: list[PromptCandidate],
    embeddings: np.ndarray,
    *,
    seed: int,
    pair_count: int,
    top_k: int,
    min_similarity: float,
    device: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import torch

    if len(candidates) < pair_count * 2:
        raise ValueError(f"need at least {pair_count * 2} candidates, found {len(candidates)}")
    candidate_matrix = torch.from_numpy(embeddings).to(device=device, dtype=torch.float32)
    available = torch.ones(len(candidates), device=device, dtype=torch.bool)
    target_order = list(range(len(candidates)))
    random.Random(seed).shuffle(target_order)
    pairs: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []

    for attempt, target_index in enumerate(target_order, start=1):
        if len(pairs) == pair_count:
            break
        if not bool(available[target_index].item()):
            continue
        target = candidates[target_index]
        scores = candidate_matrix @ candidate_matrix[target_index]
        scores = scores.masked_fill(~available, float("-inf"))
        scores[target_index] = float("-inf")
        # A stable sort preserves the UUID-sorted candidate order on equal
        # scores, matching Carnator's deterministic vector-record ordering.
        ranked_indices = torch.argsort(scores, descending=True, stable=True)[:top_k]
        recalled: list[tuple[float, int]] = []
        top_20: list[dict[str, Any]] = []
        for rank, source_index in enumerate(ranked_indices.cpu().tolist(), start=1):
            score = float(scores[source_index].item())
            if not math.isfinite(score) or score < min_similarity:
                continue
            source = candidates[source_index]
            changed = _single_substitution(source, target)
            top_20.append({
                "rank": rank,
                "uuid": source.uuid,
                "prompt": source.prompt,
                "source_row_index": source.source_row_index,
                "recall_score": score,
                **_noun_record(source),
                "noun_prefilter": {
                    "passed": changed is not None,
                    "reason": "single_noun_substitution_equal_cardinality" if changed is not None else "rejected_by_cluster_pair_constraint",
                    "source_changed_noun": changed[0] if changed is not None else None,
                    "target_changed_noun": changed[1] if changed is not None else None,
                },
            })
            if changed is not None:
                recalled.append((score, source_index))
        audit: dict[str, Any] = {
            "attempt": attempt,
            "target": {
                "uuid": target.uuid,
                "prompt": target.prompt,
                "source_row_index": target.source_row_index,
                **_noun_record(target),
            },
            "available_candidates_before_selection": int(available.sum().item()),
            "top_20_recall": top_20,
        }
        if not recalled:
            audit.update({
                "outcome": "skipped",
                "reason": "no_single_noun_substitution_equal_cardinality_candidate_in_top_20",
            })
            audits.append(audit)
            continue

        score, source_index = recalled[0]
        source = candidates[source_index]
        source_changed_noun, target_changed_noun = _single_substitution(source, target) or (None, None)
        assert source_changed_noun is not None and target_changed_noun is not None
        pair_id = f"cluster_02_early_signature_{len(pairs) + 1:03d}"
        pairs.append({
            "pair_id": pair_id,
            "cached_uuid": source.uuid,
            "cached_prompt": source.prompt,
            "cached_source_row_index": source.source_row_index,
            "target_uuid": target.uuid,
            "target_prompt": target.prompt,
            "target_source_row_index": target.source_row_index,
            "recall_score": score,
            "pair_type": "single_noun_substitution",
            "source_nouns": json.dumps(list(source.noun_tokens), ensure_ascii=True, separators=(",", ":")),
            "target_nouns": json.dumps(list(target.noun_tokens), ensure_ascii=True, separators=(",", ":")),
            "source_noun_set": json.dumps(sorted(source.noun_set), ensure_ascii=True, separators=(",", ":")),
            "target_noun_set": json.dumps(sorted(target.noun_set), ensure_ascii=True, separators=(",", ":")),
            "source_changed_noun": source_changed_noun,
            "target_changed_noun": target_changed_noun,
        })
        available[source_index] = False
        available[target_index] = False
        audit.update({
            "outcome": "accepted",
            "pair_id": pair_id,
            "selected_source_uuid": source.uuid,
            "selected_recall_score": score,
            "source_changed_noun": source_changed_noun,
            "target_changed_noun": target_changed_noun,
        })
        audits.append(audit)

    if len(pairs) != pair_count:
        raise RuntimeError(f"only selected {len(pairs)} of requested {pair_count} pairs")
    all_uuids = [pair[key] for pair in pairs for key in ("cached_uuid", "target_uuid")]
    all_prompts = [pair[key] for pair in pairs for key in ("cached_prompt", "target_prompt")]
    if len(set(all_uuids)) != pair_count * 2 or len(set(all_prompts)) != pair_count * 2:
        raise AssertionError("selected pairs do not contain 120 distinct source and target prompts")
    return pairs, audits


def _write_csv(path: Path, fieldnames: Iterable[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            materialized = dict(row)
            if "recall_score" in materialized:
                materialized["recall_score"] = f"{float(materialized['recall_score']):.8f}"
            for key, value in tuple(materialized.items()):
                if isinstance(value, (list, dict)):
                    materialized[key] = json.dumps(value, ensure_ascii=True, separators=(",", ":"))
            writer.writerow(materialized)


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:5")
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    parser.add_argument("--clip-model", default="openai/clip-vit-large-patch14")
    parser.add_argument("--pair-selection-seed", type=int, default=20260823)
    parser.add_argument("--pair-count", type=int, default=60)
    parser.add_argument("--min-unique-nouns", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-similarity", type=float, default=0.65)
    parser.add_argument("--spacy-batch-size", type=int, default=512)
    parser.add_argument("--spacy-processes", type=int, default=1)
    parser.add_argument("--clip-batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_path}")
    if args.pair_count <= 0 or args.min_unique_nouns < 2 or args.top_k <= 0:
        raise ValueError("pair count/top-k must be positive and the minimum unique noun count must be at least two")
    if args.spacy_batch_size <= 0 or args.spacy_processes < 1 or args.clip_batch_size <= 0:
        raise ValueError("batch sizes must be positive and spaCy processes must be at least one")
    if not -1.0 <= args.min_similarity <= 1.0:
        raise ValueError("min similarity must be between -1 and 1")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging_path = Path(tempfile.mkdtemp(prefix=f".{output_path.name}.", dir=output_path.parent))
    try:
        print(f"reading {input_path}", flush=True)
        rows = _read_rows(input_path)
        print(f"unique input prompts: {len(rows):,}", flush=True)
        candidates, rejected, spacy_model_meta = _extract_candidates(
            rows,
            model=args.spacy_model,
            batch_size=args.spacy_batch_size,
            processes=args.spacy_processes,
            min_unique_nouns=args.min_unique_nouns,
        )
        print(f"candidates after noun filter: {len(candidates):,}", flush=True)
        embeddings = _embed_candidates(
            candidates,
            model_id=args.clip_model,
            device=args.device,
            batch_size=args.clip_batch_size,
        )
        print("selecting frozen pairs", flush=True)
        pairs, audits = _select_pairs(
            candidates,
            embeddings,
            seed=args.pair_selection_seed,
            pair_count=args.pair_count,
            top_k=args.top_k,
            min_similarity=args.min_similarity,
            device=args.device,
        )
        del embeddings

        pairs_path = staging_path / "pairs.csv"
        rejected_path = staging_path / "noun_filter_rejections.jsonl"
        audit_path = staging_path / "selection_audit.jsonl"
        noun_path = staging_path / "noun_annotations.csv"
        _write_csv(pairs_path, pairs[0].keys(), pairs)
        _write_jsonl(rejected_path, rejected)
        _write_jsonl(audit_path, audits)
        _write_csv(
            noun_path,
            (
                "uuid", "prompt", "source_row_index", "noun_tokens", "noun_set", "unique_noun_count", "eligible",
            ),
            [
                {
                    "uuid": candidate.uuid,
                    "prompt": candidate.prompt,
                    "source_row_index": candidate.source_row_index,
                    **_noun_record(candidate),
                    "eligible": True,
                }
                for candidate in candidates
            ]
            + [
                {
                    "uuid": record["uuid"],
                    "prompt": record["prompt"],
                    "source_row_index": record["source_row_index"],
                    "noun_tokens": record["noun_tokens"],
                    "noun_set": record["noun_set"],
                    "unique_noun_count": len(record["noun_set"]),
                    "eligible": False,
                }
                for record in rejected
            ],
        )
        rejection_counts: dict[str, int] = {}
        for record in rejected:
            reason = str(record["reason"])
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "frozen_dataset_only",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "input": {
                "path": str(input_path),
                "sha256": _sha256(input_path),
                "source_rows": len(rows),
            },
            "noun_extractor": {
                "implementation": "carnator.retrieval.CarnatorNounExtractor via core.retrieval.SpacyEntityExtractor",
                "spacy_model": args.spacy_model,
                "spacy_model_version": spacy_model_meta.get("version", "unknown"),
                "spacy_version": _package_version("spacy"),
                "allowed_pos": ["NOUN"],
                "token_policy": "raw CHAI token spelling/order/duplicates retained",
                "comparison_policy": "Carnator _noun_set: raw-token set, case-sensitive",
            },
            "candidate_filter": {
                "min_unique_noun_count": args.min_unique_nouns,
                "exact_same_noun_set": "rejected",
                "eligible_pair": "equal set cardinality with exactly one source-only and one target-only noun",
                "rejection_counts": rejection_counts,
                "eligible_candidate_count": len(candidates),
            },
            "retrieval": {
                "implementation": "core.retrieval.TransformersClipEmbedder and Carnator PromptRetriever ranking",
                "model_id": args.clip_model,
                "embedding_signature": f"clip:{args.clip_model}",
                "transformers_version": _package_version("transformers"),
                "candidate_top_k": args.top_k,
                "min_similarity": args.min_similarity,
                "ranking": "descending exact whole-prompt cosine; UUID order breaks exact score ties",
            },
            "selection": {
                "pair_selection_seed": args.pair_selection_seed,
                "target_order": "eligible candidates sorted by UUID, then random.Random(seed).shuffle",
                "source_selection": "highest-ranked eligible source in unused Top-20 recall",
                "requested_pair_count": args.pair_count,
                "selected_pair_count": len(pairs),
                "source_target_unique_prompt_count": len(pairs) * 2,
                "pair_type_counts": {"single_noun_substitution": len(pairs), "same_noun_set": 0},
                "skipped_target_count": sum(audit["outcome"] == "skipped" for audit in audits),
            },
            "artifacts": {
                "pairs_csv": {"path": pairs_path.name, "sha256": _sha256(pairs_path)},
                "noun_annotations_csv": {"path": noun_path.name, "sha256": _sha256(noun_path)},
                "noun_filter_rejections_jsonl": {"path": rejected_path.name, "sha256": _sha256(rejected_path)},
                "selection_audit_jsonl": {"path": audit_path.name, "sha256": _sha256(audit_path)},
            },
            "target_selection_audit": audits,
        }
        manifest_path = staging_path / "dataset_manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(staging_path, output_path)
        print(f"wrote {output_path}", flush=True)
    except BaseException:
        shutil.rmtree(staging_path, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
