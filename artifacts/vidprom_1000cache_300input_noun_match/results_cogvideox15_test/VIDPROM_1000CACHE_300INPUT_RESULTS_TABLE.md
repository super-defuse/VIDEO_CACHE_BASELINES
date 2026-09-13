# VidProM results (10 requests)

Denoising is CUDA-synchronized DiT forward time; Carnator also includes its B5 probe. Source cache preparation is outside E2E and denoising. External is Chorus-only LLM + Grounded-SAM2 time and remains in E2E. Warm-up is unscored. Misses are recorded without generation; timing and quality use generated requests only. Errors are terminal audit records, excluded from hit-rate denominators and quality scoring. Speedups divide the dense-baseline mean (per-request over generated requests; hit columns compare the method's hit subset against the baseline mean).

| Baseline | GPU | Requests | Completed | Errors | Hits | Hit rate | E2E s/request | E2E speedup | Denoising s/request | Denoising speedup | Hit E2E s/hit | Hit E2E speedup | Hit denoising s/hit | Hit denoising speedup | External s/request | CLIP-T | VBench-Q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense Baseline | 3 | 10 | 10 | 0 | N/A | N/A | 785.108083 | N/A | 760.731574 | N/A | N/A | N/A | N/A | N/A | 0.000000 | 0.311084 | 0.628999 |
| NIRVANA-VID | 4 | 10 | 10 | 0 | 8 | 0.800000 | 709.038179 | 1.107286 | 684.493487 | 1.111379 | 709.038179 | 1.107286 | 684.493487 | 1.111379 | 0.000000 | 0.289307 | 0.639609 |
| Chorus | 5 | 10 | 10 | 0 | 8 | 0.800000 | 502.651154 | 1.561934 | 428.550172 | 1.775128 | 502.651154 | 1.561934 | 428.550172 | 1.775128 | 49.738619 | 0.237732 | 0.569421 |
| Chorus2 | 6 | 10 | 10 | 0 | 8 | 0.800000 | 560.930952 | 1.399652 | 536.247551 | 1.418620 | 560.930952 | 1.399652 | 536.247551 | 1.418620 | 0.000000 | 0.309998 | 0.633450 |
| Carnator Boundary B5 | 7 | 10 | 10 | 0 | 10 | 1.000000 | 471.539954 | 1.664987 | 424.704092 | 1.791204 | 471.539954 | 1.664987 | 424.704092 | 1.791204 | 0.000000 | 0.270178 | 0.578636 |
