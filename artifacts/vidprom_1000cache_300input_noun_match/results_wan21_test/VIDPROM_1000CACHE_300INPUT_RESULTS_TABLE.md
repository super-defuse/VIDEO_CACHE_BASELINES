# VidProM results (10 requests)

Denoising is CUDA-synchronized DiT forward time; Carnator also includes its B5 probe. Source cache preparation is outside E2E and denoising. External is Chorus-only LLM + Grounded-SAM2 time and remains in E2E. Warm-up is unscored. Misses are recorded without generation; timing and quality use generated requests only. Errors are terminal audit records, excluded from hit-rate denominators and quality scoring. Speedups divide the dense-baseline mean (per-request over generated requests; hit columns compare the method's hit subset against the baseline mean).

| Baseline | GPU | Requests | Completed | Errors | Hits | Hit rate | E2E s/request | E2E speedup | Denoising s/request | Denoising speedup | Hit E2E s/hit | Hit E2E speedup | Hit denoising s/hit | Hit denoising speedup | External s/request | CLIP-T | VBench-Q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense Baseline | 3 | 10 | 10 | 0 | N/A | N/A | 1138.642229 | N/A | 974.156171 | N/A | N/A | N/A | N/A | N/A | 0.000000 | 0.298438 | 0.637282 |
| NIRVANA-VID | 4 | 10 | 10 | 0 | 8 | 0.800000 | 1035.934165 | 1.099145 | 884.471668 | 1.101399 | 1035.934165 | 1.099145 | 884.471668 | 1.101399 | 0.000000 | 0.293365 | 0.616879 |
| Chorus | 5 | 10 | 10 | 0 | 8 | 0.800000 | 719.834631 | 1.581811 | 582.443441 | 1.672534 | 719.834631 | 1.581811 | 582.443441 | 1.672534 | 63.312982 | 0.289886 | 0.581901 |
| Chorus2 | 6 | 10 | 10 | 0 | 8 | 0.800000 | 577.065239 | 1.973160 | 556.885907 | 1.749292 | 577.065239 | 1.973160 | 556.885907 | 1.749292 | 0.000000 | 0.296936 | 0.646070 |
| Carnator Boundary B5 | 7 | 10 | 10 | 0 | 10 | 1.000000 | 547.247633 | 2.080671 | 496.019180 | 1.963949 | 547.247633 | 2.080671 | 496.019180 | 1.963949 | 0.000000 | 0.253345 | 0.608552 |
