# VidProM results (300 requests)

Denoising is CUDA-synchronized DiT forward time; Carnator also includes its B5 probe. Source cache preparation is outside E2E and denoising. External is Chorus-only LLM + Grounded-SAM2 time and remains in E2E. Warm-up is unscored. Misses are recorded without generation; timing and quality use generated requests only. Errors are terminal audit records, excluded from hit-rate denominators and quality scoring. Speedups divide the dense-baseline mean (per-request over generated requests; hit columns compare the method's hit subset against the baseline mean).

| Baseline | GPU | Requests | Completed | Errors | Hits | Hit rate | E2E s/request | E2E speedup | Denoising s/request | Denoising speedup | Hit E2E s/hit | Hit E2E speedup | Hit denoising s/hit | Hit denoising speedup | External s/request | CLIP-T | VBench-Q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense Baseline | 3 | 300 | 300 | 0 | N/A | N/A | 80.365435 | N/A | 78.461967 | N/A | N/A | N/A | N/A | N/A | 0.000000 | 0.265015 | 0.578201 |
| NIRVANA-VID | 4 | 300 | 300 | 0 | 242 | 0.806667 | 72.425166 | 1.109634 | 70.563457 | 1.111935 | 72.425166 | 1.109634 | 70.563457 | 1.111935 | 0.000000 | 0.263592 | 0.601987 |
| Chorus | 5 | 300 | 293 | 7 | 235 | 0.802048 | 132.682578 | 0.605697 | 69.049680 | 1.136312 | 132.682578 | 0.605697 | 69.049680 | 1.136312 | 61.748206 | 0.263719 | 0.592555 |
| Chorus2 | 6 | 300 | 300 | 0 | 242 | 0.806667 | 78.108267 | 1.028898 | 75.712998 | 1.036308 | 78.108267 | 1.028898 | 75.712998 | 1.036308 | 0.000000 | 0.285969 | 0.611711 |
| Carnator Boundary B5 | 7 | 300 | 300 | 0 | 300 | 1.000000 | 67.750684 | 1.186194 | 62.163494 | 1.262187 | 67.750684 | 1.186194 | 62.163494 | 1.262187 | 0.000000 | 0.249987 | 0.589048 |
