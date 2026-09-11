# VidProM 300-request quality distributions (VBench-Q / CLIP-T)

Source: `early_signature_disabled_20260903_100305` — per-request `clip_t` / `vbench_q` from each method's `quality_input.jsonl`.

`vbench_q` is the CHAI composite of six VBench dimensions (weights: dynamic_degree 0.5, others 1.0; each min-max normalized, then rescaled by 4/5 = quality 4 : semantic 1). Recomputation from the stored dimensions matched all stored values (0 mismatches). `clip_t` is the frame-mean CLIP ViT-B/32 text-image cosine similarity.

## VBench-Q — all scored requests (populations differ)

| Method | n | Mean | Std | Min | P5 | P10 | P25 | P50 | P75 | P90 | P95 | Max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NIRVANA-VID | 242 | 0.6487 | 0.0314 | 0.5692 | 0.6002 | 0.6055 | 0.6205 | 0.6549 | 0.6760 | 0.6868 | 0.6916 | 0.7065 |
| TeaCache | 300 | 0.6049 | 0.0440 | 0.4512 | 0.5262 | 0.5546 | 0.5834 | 0.6040 | 0.6347 | 0.6597 | 0.6759 | 0.7008 |
| Chorus | 201 | 0.6407 | 0.0318 | 0.5835 | 0.5973 | 0.6025 | 0.6116 | 0.6361 | 0.6713 | 0.6800 | 0.6906 | 0.7006 |
| Chorus2 | 242 | 0.6417 | 0.0356 | 0.4592 | 0.5883 | 0.5995 | 0.6141 | 0.6463 | 0.6717 | 0.6824 | 0.6878 | 0.7038 |
| Carnator B5 | 300 | 0.6401 | 0.0339 | 0.5557 | 0.5836 | 0.5932 | 0.6113 | 0.6454 | 0.6677 | 0.6817 | 0.6863 | 0.7139 |

## VBench-Q — paired subset scored by every method (n=201)

| Method | Mean | Std | P5 | P10 | P25 | P50 | P75 | P90 | P95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NIRVANA-VID | 0.6494 | 0.0312 | 0.6007 | 0.6056 | 0.6206 | 0.6554 | 0.6759 | 0.6868 | 0.6920 |
| TeaCache | 0.6041 | 0.0452 | 0.5182 | 0.5542 | 0.5829 | 0.6033 | 0.6341 | 0.6630 | 0.6759 |
| Chorus | 0.6407 | 0.0318 | 0.5973 | 0.6025 | 0.6116 | 0.6361 | 0.6713 | 0.6800 | 0.6906 |
| Chorus2 | 0.6432 | 0.0363 | 0.5882 | 0.6000 | 0.6156 | 0.6511 | 0.6732 | 0.6828 | 0.6880 |
| Carnator B5 | 0.6399 | 0.0343 | 0.5819 | 0.5910 | 0.6115 | 0.6452 | 0.6670 | 0.6816 | 0.6883 |

Per-request paired deltas vs Carnator B5 (`Ref wins` = share of requests where the reference scores higher):

| Comparison | Mean Δ | Δ P5 | Δ P25 | Δ P50 | Δ P75 | Δ P95 | Ref wins |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NIRVANA-VID − Carnator B5 | +0.0095 | -0.0601 | -0.0103 | +0.0056 | +0.0313 | +0.0781 | 39.8% |
| TeaCache − Carnator B5 | -0.0358 | -0.1267 | -0.0676 | -0.0381 | +0.0062 | +0.0364 | 73.1% |
| Chorus − Carnator B5 | +0.0008 | -0.0675 | -0.0230 | +0.0033 | +0.0222 | +0.0703 | 47.3% |
| Chorus2 − Carnator B5 | +0.0033 | -0.0695 | -0.0318 | +0.0057 | +0.0347 | +0.0798 | 43.3% |

## CLIP-T — all scored requests (populations differ)

| Method | n | Mean | Std | Min | P5 | P10 | P25 | P50 | P75 | P90 | P95 | Max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NIRVANA-VID | 242 | 0.2855 | 0.0340 | 0.1748 | 0.2273 | 0.2422 | 0.2636 | 0.2865 | 0.3087 | 0.3261 | 0.3392 | 0.3691 |
| TeaCache | 300 | 0.2938 | 0.0376 | 0.1785 | 0.2283 | 0.2462 | 0.2695 | 0.2942 | 0.3209 | 0.3423 | 0.3521 | 0.3936 |
| Chorus | 201 | 0.2993 | 0.0315 | 0.1964 | 0.2491 | 0.2571 | 0.2791 | 0.3010 | 0.3198 | 0.3386 | 0.3496 | 0.3887 |
| Chorus2 | 242 | 0.3017 | 0.0331 | 0.1748 | 0.2466 | 0.2625 | 0.2827 | 0.3032 | 0.3251 | 0.3403 | 0.3505 | 0.3792 |
| Carnator B5 | 300 | 0.2616 | 0.0380 | 0.1335 | 0.1958 | 0.2115 | 0.2395 | 0.2654 | 0.2891 | 0.3008 | 0.3116 | 0.3789 |

## CLIP-T — paired subset scored by every method (n=201)

| Method | Mean | Std | P5 | P10 | P25 | P50 | P75 | P90 | P95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NIRVANA-VID | 0.2863 | 0.0343 | 0.2294 | 0.2428 | 0.2661 | 0.2869 | 0.3098 | 0.3269 | 0.3396 |
| TeaCache | 0.2963 | 0.0362 | 0.2405 | 0.2515 | 0.2739 | 0.2979 | 0.3213 | 0.3428 | 0.3486 |
| Chorus | 0.2993 | 0.0315 | 0.2491 | 0.2571 | 0.2791 | 0.3010 | 0.3198 | 0.3386 | 0.3496 |
| Chorus2 | 0.3024 | 0.0328 | 0.2463 | 0.2646 | 0.2832 | 0.3042 | 0.3252 | 0.3401 | 0.3484 |
| Carnator B5 | 0.2653 | 0.0390 | 0.1965 | 0.2181 | 0.2410 | 0.2700 | 0.2915 | 0.3064 | 0.3145 |

Per-request paired deltas vs Carnator B5 (`Ref wins` = share of requests where the reference scores higher):

| Comparison | Mean Δ | Δ P5 | Δ P25 | Δ P50 | Δ P75 | Δ P95 | Ref wins |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NIRVANA-VID − Carnator B5 | +0.0210 | -0.0320 | +0.0007 | +0.0149 | +0.0348 | +0.0900 | 23.4% |
| TeaCache − Carnator B5 | +0.0310 | -0.0332 | -0.0012 | +0.0249 | +0.0596 | +0.1255 | 26.4% |
| Chorus − Carnator B5 | +0.0340 | -0.0164 | +0.0056 | +0.0302 | +0.0535 | +0.1163 | 18.4% |
| Chorus2 − Carnator B5 | +0.0371 | -0.0288 | +0.0083 | +0.0339 | +0.0593 | +0.1222 | 16.9% |

## VBench-Q dimension means (paired subset)

| Method | subject_con | background_ | motion_smoo | aesthetic_q | imaging_qua | dynamic_deg |
| --- | --- | --- | --- | --- | --- | --- |
| NIRVANA-VID | 0.9323 | 0.9598 | 0.9816 | 0.5935 | 0.7136 | 0.6915 |
| TeaCache | 0.9234 | 0.9536 | 0.9850 | 0.5174 | 0.6447 | 0.3731 |
| Chorus | 0.9415 | 0.9598 | 0.9830 | 0.5827 | 0.7209 | 0.5473 |
| Chorus2 | 0.9333 | 0.9553 | 0.9850 | 0.5726 | 0.7049 | 0.6517 |
| Carnator B5 | 0.9223 | 0.9503 | 0.9813 | 0.5688 | 0.7060 | 0.6766 |

Each dimension's additive contribution to the composite (columns sum to the VBench-Q mean above):

| Method | subject_con | background_ | motion_smoo | aesthetic_q | imaging_qua | dynamic_deg | Total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NIRVANA-VID | 0.1339 | 0.1375 | 0.1375 | 0.0863 | 0.1038 | 0.0503 | 0.6494 |
| TeaCache | 0.1324 | 0.1363 | 0.1392 | 0.0753 | 0.0938 | 0.0271 | 0.6041 |
| Chorus | 0.1355 | 0.1375 | 0.1382 | 0.0848 | 0.1049 | 0.0398 | 0.6407 |
| Chorus2 | 0.1341 | 0.1366 | 0.1392 | 0.0833 | 0.1025 | 0.0474 | 0.6432 |
| Carnator B5 | 0.1322 | 0.1357 | 0.1373 | 0.0827 | 0.1027 | 0.0492 | 0.6399 |

## Scored coverage per method

| Method | Requests | Scored | Scored rate | Scored rows that are cache hits |
| --- | --- | --- | --- | --- |
| NIRVANA-VID | 300 | 242 | 0.807 | 242 |
| TeaCache | 300 | 300 | 1.000 | 0 |
| Chorus | 300 | 201 | 0.670 | 201 |
| Chorus2 | 300 | 242 | 0.807 | 242 |
| Carnator B5 | 300 | 300 | 1.000 | 300 |

