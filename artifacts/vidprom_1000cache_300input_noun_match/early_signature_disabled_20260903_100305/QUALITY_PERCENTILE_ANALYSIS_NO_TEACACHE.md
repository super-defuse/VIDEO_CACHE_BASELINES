# VidProM 300-request quality distributions (VBench-Q / CLIP-T)

**Excluded methods: TeaCache.** Every table and every paired subset below is recomputed without them.

Source: `early_signature_disabled_20260903_100305` — per-request `clip_t` / `vbench_q` from each method's `quality_input.jsonl`.

`vbench_q` is the CHAI composite of six VBench dimensions (weights: dynamic_degree 0.5, others 1.0; each min-max normalized, then rescaled by 4/5 = quality 4 : semantic 1). Recomputation from the stored dimensions matched all stored values (0 mismatches). `clip_t` is the frame-mean CLIP ViT-B/32 text-image cosine similarity.

## VBench-Q — all scored requests (populations differ)

| Method | n | Mean | Std | Min | P5 | P10 | P25 | P50 | P75 | P90 | P95 | Max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NIRVANA-VID | 242 | 0.6487 | 0.0314 | 0.5692 | 0.6002 | 0.6055 | 0.6205 | 0.6549 | 0.6760 | 0.6868 | 0.6916 | 0.7065 |
| Chorus | 201 | 0.6407 | 0.0318 | 0.5835 | 0.5973 | 0.6025 | 0.6116 | 0.6361 | 0.6713 | 0.6800 | 0.6906 | 0.7006 |
| Chorus2 | 242 | 0.6417 | 0.0356 | 0.4592 | 0.5883 | 0.5995 | 0.6141 | 0.6463 | 0.6717 | 0.6824 | 0.6878 | 0.7038 |
| Carnator B5 | 300 | 0.6401 | 0.0339 | 0.5557 | 0.5836 | 0.5932 | 0.6113 | 0.6454 | 0.6677 | 0.6817 | 0.6863 | 0.7139 |

## VBench-Q — paired subset scored by every method (n=201)

| Method | Mean | Std | P5 | P10 | P25 | P50 | P75 | P90 | P95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NIRVANA-VID | 0.6494 | 0.0312 | 0.6007 | 0.6056 | 0.6206 | 0.6554 | 0.6759 | 0.6868 | 0.6920 |
| Chorus | 0.6407 | 0.0318 | 0.5973 | 0.6025 | 0.6116 | 0.6361 | 0.6713 | 0.6800 | 0.6906 |
| Chorus2 | 0.6432 | 0.0363 | 0.5882 | 0.6000 | 0.6156 | 0.6511 | 0.6732 | 0.6828 | 0.6880 |
| Carnator B5 | 0.6399 | 0.0343 | 0.5819 | 0.5910 | 0.6115 | 0.6452 | 0.6670 | 0.6816 | 0.6883 |

Per-request paired deltas vs Carnator B5 (`Ref wins` = share of requests where the reference scores higher):

| Comparison | Mean Δ | Δ P5 | Δ P25 | Δ P50 | Δ P75 | Δ P95 | Ref wins |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NIRVANA-VID − Carnator B5 | +0.0095 | -0.0601 | -0.0103 | +0.0056 | +0.0313 | +0.0781 | 39.8% |
| Chorus − Carnator B5 | +0.0008 | -0.0675 | -0.0230 | +0.0033 | +0.0222 | +0.0703 | 47.3% |
| Chorus2 − Carnator B5 | +0.0033 | -0.0695 | -0.0318 | +0.0057 | +0.0347 | +0.0798 | 43.3% |

## CLIP-T — all scored requests (populations differ)

| Method | n | Mean | Std | Min | P5 | P10 | P25 | P50 | P75 | P90 | P95 | Max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NIRVANA-VID | 242 | 0.2855 | 0.0340 | 0.1748 | 0.2273 | 0.2422 | 0.2636 | 0.2865 | 0.3087 | 0.3261 | 0.3392 | 0.3691 |
| Chorus | 201 | 0.2993 | 0.0315 | 0.1964 | 0.2491 | 0.2571 | 0.2791 | 0.3010 | 0.3198 | 0.3386 | 0.3496 | 0.3887 |
| Chorus2 | 242 | 0.3017 | 0.0331 | 0.1748 | 0.2466 | 0.2625 | 0.2827 | 0.3032 | 0.3251 | 0.3403 | 0.3505 | 0.3792 |
| Carnator B5 | 300 | 0.2616 | 0.0380 | 0.1335 | 0.1958 | 0.2115 | 0.2395 | 0.2654 | 0.2891 | 0.3008 | 0.3116 | 0.3789 |

## CLIP-T — paired subset scored by every method (n=201)

| Method | Mean | Std | P5 | P10 | P25 | P50 | P75 | P90 | P95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NIRVANA-VID | 0.2863 | 0.0343 | 0.2294 | 0.2428 | 0.2661 | 0.2869 | 0.3098 | 0.3269 | 0.3396 |
| Chorus | 0.2993 | 0.0315 | 0.2491 | 0.2571 | 0.2791 | 0.3010 | 0.3198 | 0.3386 | 0.3496 |
| Chorus2 | 0.3024 | 0.0328 | 0.2463 | 0.2646 | 0.2832 | 0.3042 | 0.3252 | 0.3401 | 0.3484 |
| Carnator B5 | 0.2653 | 0.0390 | 0.1965 | 0.2181 | 0.2410 | 0.2700 | 0.2915 | 0.3064 | 0.3145 |

Per-request paired deltas vs Carnator B5 (`Ref wins` = share of requests where the reference scores higher):

| Comparison | Mean Δ | Δ P5 | Δ P25 | Δ P50 | Δ P75 | Δ P95 | Ref wins |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NIRVANA-VID − Carnator B5 | +0.0210 | -0.0320 | +0.0007 | +0.0149 | +0.0348 | +0.0900 | 23.4% |
| Chorus − Carnator B5 | +0.0340 | -0.0164 | +0.0056 | +0.0302 | +0.0535 | +0.1163 | 18.4% |
| Chorus2 − Carnator B5 | +0.0371 | -0.0288 | +0.0083 | +0.0339 | +0.0593 | +0.1222 | 16.9% |

## VBench-Q dimension means (paired subset)

| Method | subject_con | background_ | motion_smoo | aesthetic_q | imaging_qua | dynamic_deg |
| --- | --- | --- | --- | --- | --- | --- |
| NIRVANA-VID | 0.9323 | 0.9598 | 0.9816 | 0.5935 | 0.7136 | 0.6915 |
| Chorus | 0.9415 | 0.9598 | 0.9830 | 0.5827 | 0.7209 | 0.5473 |
| Chorus2 | 0.9333 | 0.9553 | 0.9850 | 0.5726 | 0.7049 | 0.6517 |
| Carnator B5 | 0.9223 | 0.9503 | 0.9813 | 0.5688 | 0.7060 | 0.6766 |

Each dimension's additive contribution to the composite (columns sum to the VBench-Q mean above):

| Method | subject_con | background_ | motion_smoo | aesthetic_q | imaging_qua | dynamic_deg | Total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NIRVANA-VID | 0.1339 | 0.1375 | 0.1375 | 0.0863 | 0.1038 | 0.0503 | 0.6494 |
| Chorus | 0.1355 | 0.1375 | 0.1382 | 0.0848 | 0.1049 | 0.0398 | 0.6407 |
| Chorus2 | 0.1341 | 0.1366 | 0.1392 | 0.0833 | 0.1025 | 0.0474 | 0.6432 |
| Carnator B5 | 0.1322 | 0.1357 | 0.1373 | 0.0827 | 0.1027 | 0.0492 | 0.6399 |

## Scored coverage per method

| Method | Requests | Scored | Scored rate | Scored rows that are cache hits |
| --- | --- | --- | --- | --- |
| NIRVANA-VID | 300 | 242 | 0.807 | 242 |
| Chorus | 300 | 201 | 0.670 | 201 |
| Chorus2 | 300 | 242 | 0.807 | 242 |
| Carnator B5 | 300 | 300 | 1.000 | 300 |

## Low-score tail (paired subset, n=201)

### VBench-Q — tail quantiles

| Method | Min | Worst-3 | P1 | P2.5 | P5 | P10 | P25 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NIRVANA-VID | 0.5692 | 0.5833 | 0.5915 | 0.5938 | 0.6007 | 0.6056 | 0.6206 |
| Chorus | 0.5835 | 0.5849 | 0.5860 | 0.5905 | 0.5973 | 0.6025 | 0.6116 |
| Chorus2 | 0.4592 | 0.5267 | 0.5644 | 0.5728 | 0.5882 | 0.6000 | 0.6156 |
| Carnator B5 | 0.5557 | 0.5656 | 0.5729 | 0.5783 | 0.5819 | 0.5910 | 0.6115 |

### VBench-Q — requests below absolute thresholds

| Method | n<0.52 | n<0.55 | n<0.58 | n<0.6 | %<0.52 | %<0.55 | %<0.58 | %<0.6 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NIRVANA-VID | 0 | 0 | 1 | 9 | 0.0 | 0.0 | 0.5 | 4.5 |
| Chorus | 0 | 0 | 0 | 13 | 0.0 | 0.0 | 0.0 | 6.5 |
| Chorus2 | 1 | 1 | 7 | 20 | 0.5 | 0.5 | 3.5 | 10.0 |
| Carnator B5 | 0 | 0 | 9 | 33 | 0.0 | 0.0 | 4.5 | 16.4 |

### VBench-Q — offset from the other methods, on each selector's own P25

Each row selects the requests in that method's own lowest quartile; a negative value for the selector is expected mechanically (regression to the mean), so the comparison of interest is between the *other* columns, not against the selector.

| Selector (P25) | NIRVANA-VID | Chorus | Chorus2 | Carnator B5 |
| --- | --- | --- | --- | --- |
| NIRVANA-VID | -16.82 | -9.91 | +19.87 | +6.86 |
| Chorus | +7.58 | -28.14 | +16.28 | +4.29 |
| Chorus2 | +18.20 | +8.15 | -35.80 | +9.46 |
| Carnator B5 | +19.74 | +5.87 | +15.63 | -41.24 |

### VBench-Q — pairwise deltas inside third-party low bands

For each pair, the hard subset is defined by a method that is neither member of the pair, so selection cannot favour either side.

| Pair | Selector | n | Δ (A−B) | A wins | p |
| --- | --- | --- | --- | --- | --- |
| NIRVANA-VID − Chorus | Chorus2 | 51 | +0.0075 | 51.0% | 1.000 |
| NIRVANA-VID − Chorus | Carnator B5 | 51 | +0.0104 | 60.8% | 0.161 |
| NIRVANA-VID − Chorus2 | Chorus | 51 | -0.0065 | 41.2% | 0.262 |
| NIRVANA-VID − Chorus2 | Carnator B5 | 51 | +0.0031 | 52.9% | 0.780 |
| NIRVANA-VID − Carnator B5 | Chorus | 51 | +0.0025 | 52.9% | 0.780 |
| NIRVANA-VID − Carnator B5 | Chorus2 | 51 | +0.0066 | 49.0% | 1.000 |
| Chorus − Chorus2 | NIRVANA-VID | 51 | -0.0223 | 33.3% | 0.024 |
| Chorus − Chorus2 | Carnator B5 | 51 | -0.0073 | 43.1% | 0.401 |
| Chorus − Carnator B5 | NIRVANA-VID | 51 | -0.0126 | 41.2% | 0.262 |
| Chorus − Carnator B5 | Chorus2 | 51 | -0.0010 | 49.0% | 1.000 |
| Chorus2 − Carnator B5 | NIRVANA-VID | 51 | +0.0098 | 64.7% | 0.049 |
| Chorus2 − Carnator B5 | Chorus | 51 | +0.0090 | 66.7% | 0.024 |

### VBench-Q — every method's mean on each method's 10 worst requests

| Worst-10 by | NIRVANA-VID | Chorus | Chorus2 | Carnator B5 |
| --- | --- | --- | --- | --- |
| NIRVANA-VID | 0.5924 | 0.5992 | 0.6152 | 0.6213 |
| Chorus | 0.6195 | 0.5892 | 0.6335 | 0.6211 |
| Chorus2 | 0.6351 | 0.6385 | 0.5621 | 0.6369 |
| Carnator B5 | 0.6491 | 0.6395 | 0.6359 | 0.5747 |

### CLIP-T — tail quantiles

| Method | Min | Worst-3 | P1 | P2.5 | P5 | P10 | P25 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NIRVANA-VID | 0.1748 | 0.1838 | 0.1891 | 0.2095 | 0.2294 | 0.2428 | 0.2661 |
| Chorus | 0.1964 | 0.2120 | 0.2214 | 0.2422 | 0.2491 | 0.2571 | 0.2791 |
| Chorus2 | 0.1748 | 0.1900 | 0.2043 | 0.2296 | 0.2463 | 0.2646 | 0.2832 |
| Carnator B5 | 0.1335 | 0.1428 | 0.1604 | 0.1819 | 0.1965 | 0.2181 | 0.2410 |

### CLIP-T — requests below absolute thresholds

| Method | n<0.18 | n<0.2 | n<0.22 | n<0.25 | %<0.18 | %<0.2 | %<0.22 | %<0.25 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NIRVANA-VID | 1 | 4 | 7 | 29 | 0.5 | 2.0 | 3.5 | 14.4 |
| Chorus | 0 | 1 | 2 | 13 | 0.0 | 0.5 | 1.0 | 6.5 |
| Chorus2 | 1 | 2 | 5 | 11 | 0.5 | 1.0 | 2.5 | 5.5 |
| Carnator B5 | 5 | 14 | 23 | 64 | 2.5 | 7.0 | 11.4 | 31.8 |

### CLIP-T — offset from the other methods, on each selector's own P25

Each row selects the requests in that method's own lowest quartile; a negative value for the selector is expected mechanically (regression to the mean), so the comparison of interest is between the *other* columns, not against the selector.

| Selector (P25) | NIRVANA-VID | Chorus | Chorus2 | Carnator B5 |
| --- | --- | --- | --- | --- |
| NIRVANA-VID | -26.36 | +15.45 | +36.63 | -25.73 |
| Chorus | -12.24 | -2.73 | +34.55 | -19.59 |
| Chorus2 | +2.25 | +26.62 | -9.97 | -18.91 |
| Carnator B5 | +0.25 | +29.67 | +38.58 | -68.50 |

### CLIP-T — pairwise deltas inside third-party low bands

For each pair, the hard subset is defined by a method that is neither member of the pair, so selection cannot favour either side.

| Pair | Selector | n | Δ (A−B) | A wins | p |
| --- | --- | --- | --- | --- | --- |
| NIRVANA-VID − Chorus | Chorus2 | 52 | -0.0183 | 21.2% | 0.000 |
| NIRVANA-VID − Chorus | Carnator B5 | 51 | -0.0221 | 17.6% | 0.000 |
| NIRVANA-VID − Chorus2 | Chorus | 51 | -0.0351 | 13.7% | 0.000 |
| NIRVANA-VID − Chorus2 | Carnator B5 | 51 | -0.0287 | 25.5% | 0.001 |
| NIRVANA-VID − Carnator B5 | Chorus | 51 | +0.0055 | 68.6% | 0.011 |
| NIRVANA-VID − Carnator B5 | Chorus2 | 52 | +0.0159 | 67.3% | 0.018 |
| Chorus − Chorus2 | NIRVANA-VID | 51 | -0.0159 | 33.3% | 0.024 |
| Chorus − Chorus2 | Carnator B5 | 51 | -0.0067 | 41.2% | 0.262 |
| Chorus − Carnator B5 | NIRVANA-VID | 51 | +0.0309 | 74.5% | 0.001 |
| Chorus − Carnator B5 | Chorus2 | 52 | +0.0341 | 84.6% | 0.000 |
| Chorus2 − Carnator B5 | NIRVANA-VID | 51 | +0.0468 | 86.3% | 0.000 |
| Chorus2 − Carnator B5 | Chorus | 51 | +0.0406 | 84.3% | 0.000 |

### CLIP-T — every method's mean on each method's 10 worst requests

| Worst-10 by | NIRVANA-VID | Chorus | Chorus2 | Carnator B5 |
| --- | --- | --- | --- | --- |
| NIRVANA-VID | 0.2044 | 0.2594 | 0.2744 | 0.2150 |
| Chorus | 0.2444 | 0.2333 | 0.2788 | 0.2484 |
| Chorus2 | 0.2605 | 0.2769 | 0.2187 | 0.2369 |
| Carnator B5 | 0.2541 | 0.2865 | 0.2812 | 0.1690 |

