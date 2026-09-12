# VidProM results (10 requests)

Denoising is CUDA-synchronized DiT forward time; Carnator also includes its B5 probe. Source cache preparation is outside E2E and denoising. External is Chorus-only LLM + Grounded-SAM2 time and remains in E2E. Warm-up is unscored. Misses are recorded without generation; timing and quality use generated requests only. Errors are terminal audit records, excluded from hit-rate denominators and quality scoring. Speedups divide the dense-baseline mean (per-request over generated requests; hit columns compare the method's hit subset against the baseline mean).

| Baseline | GPU | Requests | Completed | Errors | Hits | Hit rate | E2E s/request | E2E speedup | Denoising s/request | Denoising speedup | Hit E2E s/hit | Hit E2E speedup | Hit denoising s/hit | Hit denoising speedup | External s/request | CLIP-T | VBench-Q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense Baseline | 3 | 10 | 10 | 0 | N/A | N/A | 81.420689 | N/A | 79.367488 | N/A | N/A | N/A | N/A | N/A | 0.000000 | 0.260779 | 0.589608 |
| NIRVANA-VID | 4 | 10 | 10 | 0 | 8 | 0.800000 | 72.823070 | 1.118062 | 70.777384 | 1.121368 | 72.823070 | 1.118062 | 70.777384 | 1.121368 | 0.000000 | 0.258118 | 0.609897 |
| Chorus | 5 | 10 | 10 | 0 | 8 | 0.800000 | 136.236925 | 0.597640 | 70.378952 | 1.127716 | 136.236925 | 0.597640 | 70.378952 | 1.127716 | 63.864084 | 0.256607 | 0.601782 |
| Chorus2 | 6 | 10 | 10 | 0 | 8 | 0.800000 | 77.686015 | 1.048074 | 75.256135 | 1.054631 | 77.686015 | 1.048074 | 75.256135 | 1.054631 | 0.000000 | 0.277054 | 0.632814 |
| Carnator Boundary B5 | 7 | 10 | 10 | 0 | 10 | 1.000000 | 68.154582 | 1.194647 | 62.531660 | 1.269237 | 68.154582 | 1.194647 | 62.531660 | 1.269237 | 0.000000 | 0.257727 | 0.592239 |
