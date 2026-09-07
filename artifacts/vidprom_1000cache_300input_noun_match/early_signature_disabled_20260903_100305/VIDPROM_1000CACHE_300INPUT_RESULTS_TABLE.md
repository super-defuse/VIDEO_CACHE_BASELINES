# VidProM results (300 requests)

Denoising is CUDA-synchronized DiT forward time; Carnator also includes its B5 probe. Source cache preparation is outside E2E and denoising. External is Chorus-only LLM + Grounded-SAM2 time and remains in E2E. Warm-up is unscored. Misses are recorded without generation; timing and quality use generated requests only. Errors are terminal audit records, excluded from hit-rate denominators and quality scoring.

| Baseline | GPU | Requests | Completed | Errors | Hits | Hit rate | E2E s/request | Hit E2E s/hit | Denoising s/request | Hit denoising s/hit | External s/request | CLIP-T | VBench-Q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NIRVANA-VID | 3 | 300 | 300 | 0 | 242 | 0.806667 | 256.087258 | 256.087258 | 204.211476 | 204.211476 | 0.000000 | 0.285484 | 0.648716 |
| TeaCache | 4 | 300 | 300 | 0 | N/A | N/A | 135.382503 | N/A | 118.459225 | N/A | 0.000000 | 0.293771 | 0.604909 |
| Chorus | 5 | 300 | 259 | 41 | 201 | 0.776062 | 264.467974 | 264.467974 | 175.484454 | 175.484454 | 56.607013 | 0.299310 | 0.640661 |
| Chorus2 | 6 | 300 | 300 | 0 | 242 | 0.806667 | 222.440040 | 222.440040 | 201.577447 | 201.577447 | 0.000000 | 0.301749 | 0.641706 |
| Carnator Boundary B5 | 7 | 300 | 300 | 0 | 300 | 1.000000 | 205.178984 | 205.178984 | 160.258860 | 160.258860 | 0.000000 | 0.261571 | 0.640110 |

## Carnator disabled-early-signature recheck (2026-09-06)

- The 65 prior Carnator misses (64 mask-diagnostic cases plus request_index=23) were rerun on GPUs 3–7 under the canonical disabled rule: all four Early Signature thresholds equal 1e6.
- All 65 completed as cache hits after the finite-precision T5 probe fallback; each has retained step-5 probe and selected-mask diagnostics, plus CLIP-T and six-dimensional VBench-Q quality values.
- The 65 rows were substituted by global request_index; the pre-merge JSONL/config backup is carnator_boundary_b5/pre_recheck_20260906/. Detailed validation is in [RECHECK_SUMMARY.md](../carnator_disabled_recheck_20260906/RECHECK_SUMMARY.md).
