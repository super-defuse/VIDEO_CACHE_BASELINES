#!/usr/bin/env python3
"""Freeze the prompt pairs used to calibrate Carnator Early Signature.

The selection protocol lives in ``docs/CARNATOR_EARLY_SIGNATURE_CALIBRATION_PLAN.md``.
This command intentionally only constructs the dataset; it does not generate
source caches, target videos, or quality labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import random
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from core.retrieval import SpacyEntityExtractor, TransformersClipEmbedder


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT.parents[1] / "datasets" / "VidProM-CHAI-100-1000" / "VidProM_unique.csv"
DEFAULT_OUTPUT = ROOT / "artifacts" / "early_signature_calibration"
TOXICITY_COLUMNS = (
    "toxicity",
    "obscene",
    "identity_attack",
    "insult",
    "threat",
    "sexual_explicit",
)
URL_RE = re.compile(r"(?:https?://|www\.)", re.IGNORECASE)
HTML_ENTITY_RE = re.compile(r"&(?:#[0-9]+|#x[0-9a-f]+|[a-z][a-z0-9]+);", re.IGNORECASE)
ASPECT_RATIO_RE = re.compile(r"--ar(?:\s|$)", re.IGNORECASE)
SCHEMA_VERSION = "carnator-early-signature-calibration-dataset"


@dataclass(frozen=True)
class PromptCandidate:
    uuid: str
    prompt: str
    nouns: tuple[str, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normal_prompt_rejection(row: dict[str, str]) -> str | None:
    prompt = str(row.get("prompt", "")).strip()
    if not prompt:
        return "empty_prompt"
    if not 30 <= len(prompt) <= 300:
        return "prompt_length_out_of_range"
    if URL_RE.search(prompt):
        return "contains_url"
    if HTML_ENTITY_RE.search(prompt):
        return "contains_html_entity"
    if ASPECT_RATIO_RE.search(prompt):
        return "contains_aspect_ratio_instruction"
    try:
        scores = [float(row[column]) for column in TOXICITY_COLUMNS]
    except (KeyError, TypeError, ValueError):
        return "invalid_toxicity_score"
    if any(not math.isfinite(score) or score > 0.05 for score in scores):
        return "toxicity_score_above_limit"
    return None


def _load_basic_candidates(input_path: Path) -> tuple[list[tuple[str, str]], dict[str, int], int]:
    required_columns = {"uuid", "prompt", *TOXICITY_COLUMNS}
    candidates: list[tuple[str, str]] = []
    reject_counts: dict[str, int] = {}
    seen_uuids: set[str] = set()
    seen_prompts: set[str] = set()
    source_rows = 0
    csv.field_size_limit(16 * 1024 * 1024)
    with input_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        headers = set(reader.fieldnames or ())
        missing = sorted(required_columns - headers)
        if missing:
            raise ValueError(f"input CSV is missing required columns: {', '.join(missing)}")
        for row in reader:
            source_rows += 1
            uuid = str(row.get("uuid", "")).strip()
            if not uuid:
                reject_counts["empty_uuid"] = reject_counts.get("empty_uuid", 0) + 1
                continue
            if uuid in seen_uuids:
                raise ValueError(f"duplicate UUID in input CSV: {uuid}")
            seen_uuids.add(uuid)
            reason = _normal_prompt_rejection(row)
            if reason is not None:
                reject_counts[reason] = reject_counts.get(reason, 0) + 1
                continue
            prompt = str(row["prompt"]).strip()
            if prompt in seen_prompts:
                reject_counts["duplicate_prompt"] = reject_counts.get("duplicate_prompt", 0) + 1
                continue
            seen_prompts.add(prompt)
            candidates.append((uuid, prompt))
    return candidates, reject_counts, source_rows


def _extract_candidates(
    basic_candidates: list[tuple[str, str]],
    *,
    model: str,
    batch_size: int,
    processes: int,
) -> tuple[list[PromptCandidate], int, dict[str, object]]:
    extractor = SpacyEntityExtractor(model)
    extractor._load()
    assert extractor._nlp is not None
    prompts = (prompt.split("aesthetic score", 1)[0] for _, prompt in basic_candidates)
    candidates: list[PromptCandidate] = []
    invalid_noun_count = 0
    for (uuid, prompt), doc in zip(
        basic_candidates,
        extractor._nlp.pipe(prompts, batch_size=batch_size, n_process=processes),
        strict=True,
    ):
        nouns = tuple(token.text for token in doc if token.pos_ in extractor.allowed_pos)
        if not 1 <= len(nouns) <= 3:
            invalid_noun_count += 1
            continue
        candidates.append(PromptCandidate(uuid, prompt, nouns))
    candidates.sort(key=lambda candidate: candidate.uuid)
    return candidates, invalid_noun_count, dict(extractor._nlp.meta)


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _embed_candidates(
    candidates: list[PromptCandidate],
    output_dir: Path,
    *,
    model_id: str,
    device: str,
    batch_size: int,
) -> np.memmap:
    if not candidates:
        raise ValueError("no normal prompt candidates remain after noun extraction")
    embedder = TransformersClipEmbedder(model_id, device)
    first_batch = embedder.encode([candidate.prompt for candidate in candidates[:batch_size]])
    first_features = np.asarray(first_batch, dtype=np.float32)
    if first_features.ndim != 2 or first_features.shape[0] == 0:
        raise RuntimeError("CLIP did not return a two-dimensional feature matrix")
    embeddings_path = output_dir / "candidate_embeddings.float32.mmap"
    embeddings = np.memmap(
        embeddings_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(candidates), first_features.shape[1]),
    )
    embeddings[: len(first_features)] = first_features
    for start in range(len(first_features), len(candidates), batch_size):
        end = min(start + batch_size, len(candidates))
        features = np.asarray(
            embedder.encode([candidate.prompt for candidate in candidates[start:end]]), dtype=np.float32
        )
        if features.shape != (end - start, embeddings.shape[1]):
            raise RuntimeError(f"unexpected CLIP feature shape {features.shape} for rows {start}:{end}")
        embeddings[start:end] = features
    embeddings.flush()
    return embeddings


def _pair_type(source: PromptCandidate, target: PromptCandidate) -> tuple[str, str, str] | None:
    source_nouns, target_nouns = set(source.nouns), set(target.nouns)
    if source_nouns == target_nouns:
        return "same_noun_set", "", ""
    source_only = sorted(source_nouns - target_nouns)
    target_only = sorted(target_nouns - source_nouns)
    if len(source_only) == len(target_only) == 1:
        return "single_noun_substitution", source_only[0], target_only[0]
    return None


def _candidate_audit(
    candidate: PromptCandidate,
    score: float,
    rank: int,
    target: PromptCandidate,
) -> dict[str, object]:
    relation = _pair_type(candidate, target)
    return {
        "rank": rank,
        "uuid": candidate.uuid,
        "prompt": candidate.prompt,
        "recall_score": score,
        "nouns": list(candidate.nouns),
        "noun_prefilter": {
            "passed": relation is not None,
            "pair_type": relation[0] if relation is not None else None,
            "source_changed_noun": relation[1] if relation is not None else None,
            "target_changed_noun": relation[2] if relation is not None else None,
        },
    }


def _select_pairs(
    candidates: list[PromptCandidate],
    embeddings: np.memmap,
    *,
    seed: int,
    pair_count: int,
    top_k: int,
    min_similarity: float,
    device: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    import torch

    if len(candidates) < pair_count * 2:
        raise ValueError(f"need at least {pair_count * 2} candidates, found {len(candidates)}")
    candidate_matrix = torch.from_numpy(np.asarray(embeddings)).to(device=device, dtype=torch.float32)
    available = torch.ones(len(candidates), device=device, dtype=torch.bool)
    ordered_targets = list(range(len(candidates)))
    random.Random(seed).shuffle(ordered_targets)
    pairs: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []

    for attempt, target_index in enumerate(ordered_targets, start=1):
        if len(pairs) == pair_count:
            break
        if not bool(available[target_index].item()):
            continue
        target = candidates[target_index]
        scores = candidate_matrix @ candidate_matrix[target_index]
        scores = scores.masked_fill(~available, float("-inf"))
        scores[target_index] = float("-inf")
        recall_size = min(top_k, len(candidates) - 1)
        top_scores, top_indices = torch.topk(scores, k=recall_size, largest=True, sorted=True)
        recalled: list[tuple[float, int]] = []
        for score, source_index in zip(top_scores.cpu().tolist(), top_indices.cpu().tolist(), strict=True):
            if not math.isfinite(score) or score < min_similarity:
                continue
            recalled.append((float(score), int(source_index)))
        # CLIP's score ties are uncommon, but ordering them by UUID keeps the
        # frozen audit stable across equivalent GPU top-k implementations.
        recalled.sort(key=lambda item: (-item[0], candidates[item[1]].uuid))
        audit_top_k = [
            _candidate_audit(candidates[source_index], score, rank, target)
            for rank, (score, source_index) in enumerate(recalled, start=1)
        ]
        selected: tuple[float, int, tuple[str, str, str]] | None = None
        for score, source_index in recalled:
            relation = _pair_type(candidates[source_index], target)
            if relation is not None:
                selected = score, source_index, relation
                break
        audit: dict[str, object] = {
            "attempt": attempt,
            "target": {"uuid": target.uuid, "prompt": target.prompt, "nouns": list(target.nouns)},
            "available_candidates_before_selection": int(available.sum().item()),
            "top_20_recall": audit_top_k,
        }
        if selected is None:
            audit["outcome"] = "skipped"
            audit["reason"] = (
                "no_recall_candidate_at_or_above_min_similarity"
                if not recalled
                else "no_noun_compatible_candidate_in_top_20"
            )
            audits.append(audit)
            continue

        score, source_index, relation = selected
        source = candidates[source_index]
        pair_id = f"early_signature_{len(pairs) + 1:03d}"
        pair = {
            "pair_id": pair_id,
            "cached_uuid": source.uuid,
            "cached_prompt": source.prompt,
            "target_uuid": target.uuid,
            "target_prompt": target.prompt,
            "recall_score": score,
            "pair_type": relation[0],
            "source_nouns": ";".join(source.nouns),
            "target_nouns": ";".join(target.nouns),
            "source_changed_noun": relation[1],
            "target_changed_noun": relation[2],
        }
        pairs.append(pair)
        available[source_index] = False
        available[target_index] = False
        audit.update({
            "outcome": "accepted",
            "pair_id": pair_id,
            "selected_source_uuid": source.uuid,
            "selected_recall_score": score,
            "pair_type": relation[0],
            "source_changed_noun": relation[1],
            "target_changed_noun": relation[2],
        })
        audits.append(audit)

    if len(pairs) != pair_count:
        raise RuntimeError(f"only selected {len(pairs)} of requested {pair_count} pairs")
    prompt_uuids = {str(pair["cached_uuid"]) for pair in pairs} | {str(pair["target_uuid"]) for pair in pairs}
    if len(prompt_uuids) != pair_count * 2:
        raise AssertionError("pair selection did not produce distinct source and target UUIDs")
    prompt_texts = {str(pair["cached_prompt"]) for pair in pairs} | {str(pair["target_prompt"]) for pair in pairs}
    if len(prompt_texts) != pair_count * 2:
        raise AssertionError("pair selection did not produce distinct source and target prompts")
    return pairs, audits


def _write_pairs(path: Path, pairs: Iterable[dict[str, object]]) -> None:
    fieldnames = (
        "pair_id",
        "cached_uuid",
        "cached_prompt",
        "target_uuid",
        "target_prompt",
        "recall_score",
        "pair_type",
        "source_nouns",
        "target_nouns",
        "source_changed_noun",
        "target_changed_noun",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for pair in pairs:
            row = dict(pair)
            row["recall_score"] = f"{float(pair['recall_score']):.8f}"
            writer.writerow(row)


def _write_audit(path: Path, audits: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for audit in audits:
            handle.write(json.dumps(audit, ensure_ascii=True, sort_keys=True) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:5")
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    parser.add_argument("--clip-model", default="openai/clip-vit-large-patch14")
    parser.add_argument("--pair-selection-seed", type=int, default=20260823)
    parser.add_argument("--pair-count", type=int, default=60)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-similarity", type=float, default=0.65)
    parser.add_argument("--spacy-batch-size", type=int, default=512)
    parser.add_argument("--spacy-processes", type=int, default=1)
    parser.add_argument("--clip-batch-size", type=int, default=256)
    parser.add_argument(
        "--q-min",
        type=float,
        default=None,
        help="Optional VBench-Q success threshold. It must be set before target generation.",
    )
    parser.add_argument("--c-min", type=float, default=59 / 60)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_path}")
    if args.pair_count <= 0 or args.top_k <= 0 or args.clip_batch_size <= 0 or args.spacy_batch_size <= 0:
        raise ValueError("pair count and batch sizes must be positive")
    if args.spacy_processes < 1:
        raise ValueError("spaCy process count must be at least one")
    if not -1.0 <= args.min_similarity <= 1.0:
        raise ValueError("min similarity must be between -1 and 1")
    if not 59 / 60 <= args.c_min <= 1.0:
        raise ValueError("C_min must be in [59/60, 1] for the stated 95% one-sided CP criterion")
    if args.q_min is not None and not math.isfinite(args.q_min):
        raise ValueError("Q_min must be finite when supplied")

    staging_path = Path(tempfile.mkdtemp(prefix=f".{output_path.name}.", dir=output_path.parent))
    try:
        print(f"reading and filtering {input_path}", flush=True)
        basic_candidates, reject_counts, source_rows = _load_basic_candidates(input_path)
        print(f"basic candidates: {len(basic_candidates):,}", flush=True)
        candidates, noun_rejections, spacy_model_meta = _extract_candidates(
            basic_candidates,
            model=args.spacy_model,
            batch_size=args.spacy_batch_size,
            processes=args.spacy_processes,
        )
        del basic_candidates
        print(f"normal candidates after noun extraction: {len(candidates):,}", flush=True)
        embeddings = _embed_candidates(
            candidates,
            staging_path,
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
        (staging_path / "candidate_embeddings.float32.mmap").unlink()
        pairs_path = staging_path / "pairs.csv"
        audit_path = staging_path / "selection_audit.jsonl"
        _write_pairs(pairs_path, pairs)
        _write_audit(audit_path, audits)
        pair_type_counts = {
            pair_type: sum(pair["pair_type"] == pair_type for pair in pairs)
            for pair_type in ("same_noun_set", "single_noun_substitution")
        }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "frozen_dataset_only",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "input": {
                "path": str(input_path),
                "sha256": _sha256(input_path),
                "source_rows": source_rows,
            },
            "normal_prompt_filter": {
                "trimmed_character_length": [30, 300],
                "excluded_patterns": ["URL", "HTML entity", "--ar format instruction"],
                "toxicity_columns": list(TOXICITY_COLUMNS),
                "max_toxicity_score": 0.05,
                "noun_count_range": [1, 3],
                "rejection_counts_before_noun_extraction": reject_counts,
                "noun_count_rejections": noun_rejections,
            },
            "noun_extractor": {
                "implementation": "core.retrieval.SpacyEntityExtractor",
                "model": args.spacy_model,
                "model_version": spacy_model_meta.get("version", "unknown"),
                "spacy_version": _distribution_version("spacy"),
                "allowed_pos": ["NOUN"],
                "signature": f"spacy:{args.spacy_model}",
                "token_policy": "CHAI-compatible raw NOUN tokens",
            },
            "retrieval": {
                "implementation": "core.retrieval.TransformersClipEmbedder",
                "model_id": args.clip_model,
                "embedding_signature": f"clip:{args.clip_model}",
                "transformers_version": _distribution_version("transformers"),
                "candidate_top_k": args.top_k,
                "min_similarity": args.min_similarity,
                "ranking": "descending exact whole-prompt cosine; UUID tiebreak for equal scores",
            },
            "selection": {
                "pair_selection_seed": args.pair_selection_seed,
                "requested_pair_count": args.pair_count,
                "selected_pair_count": len(pairs),
                "normal_candidate_count": len(candidates),
                "source_target_unique_prompt_count": len(pairs) * 2,
                "target_order": "normal candidates sorted by UUID, then random.Random(seed).shuffle",
                "source_selection": "highest-scoring noun-compatible result within unused Top-20 recall",
                "pair_type_counts": pair_type_counts,
            },
            "target_quality_policy": {
                "q_min": args.q_min,
                "q_min_status": "configured" if args.q_min is not None else "must_be_set_before_target_generation",
                "c_min": args.c_min,
                "confidence": 0.95,
                "primary_interval": "one-sided Clopper-Pearson lower bound",
            },
            "artifacts": {
                "pairs_csv": {"path": "pairs.csv", "sha256": _sha256(pairs_path)},
                "selection_audit_jsonl": {"path": "selection_audit.jsonl", "sha256": _sha256(audit_path)},
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
