# Carnator disabled-early-signature recheck — 2026-09-06

## Scope

This run replaces the 65 cache_miss rows in the canonical
early_signature_disabled_20260903_100305 Carnator result. It reran exactly the
original 64 failed indices plus case-normalization case request_index=23 on
GPUs 3–7 (13 requests per GPU).

## Protocol and validation

- All five worker run_config.json files set early_signature_disabled: true.
- All four Early Signature gate thresholds were set to 1,000,000.0, matching
  the canonical disabled-early-signature protocol.
- The five shards contain exactly 65 distinct request_index values, exactly
  matching the prior canonical miss set; each shard has 13 request and 13
  request-record rows.
- All 65 rows completed with status=ok and hit=true.
- All 65 rows retained a step-5 target-probe signature and selected source mask.
  Probe collection completed at step 5 for every row; the finite-precision
  fallback was used for all 65 probes.
- CLIP-T and the six-dimensional VBench-Q record are present and finite for all
  65 generated videos. The focused recheck means are CLIP-T 0.258139 and
  VBench-Q 0.641770.

## Canonical merge

The 65 rows were atomically substituted by global request_index into
early_signature_disabled_20260903_100305/carnator_boundary_b5 after creating
pre_recheck_20260906/, which preserves the original four JSONL files and run
configuration. The canonical Carnator aggregate is now 300 requests, 300
completed, 0 errors, 300 hits, and a 1.000000 hit rate.

recheck_merge_manifest.json records the input shard paths, the exact target
index set, the disabled thresholds, scored-results source, and backup location.
