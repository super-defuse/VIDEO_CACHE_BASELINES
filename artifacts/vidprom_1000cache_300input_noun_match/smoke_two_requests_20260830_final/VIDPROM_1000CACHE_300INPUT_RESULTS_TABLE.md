# VidProM results (2 requests)

Denoising is CUDA-synchronized DiT forward time; Carnator also includes its B5 probe. Source cache preparation is outside E2E and denoising. External is Chorus-only LLM + Grounded-SAM2 time and remains in E2E. Warm-up is unscored. Misses are recorded without generation; timing means use generated requests only.

| Baseline | GPU | Requests | Hits | Hit rate | E2E s/request | Hit E2E s/hit | Denoising s/request | Hit denoising s/hit | External s/request | CLIP-T | VBench-Q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NIRVANA-VID | 3 | 2 | 1 | 0.500000 | 257.298805 | 257.298805 | 204.406239 | 204.406239 | 0.000000 | N/A | N/A |
| TeaCache | 4 | 2 | N/A | N/A | 136.276040 | N/A | 119.819310 | N/A | 0.000000 | N/A | N/A |
| Chorus | 5 | 2 | 1 | 0.500000 | 238.252918 | 238.252918 | 141.852685 | 141.852685 | 64.996481 | N/A | N/A |
| Chorus2 | 6 | 2 | 1 | 0.500000 | 221.184858 | 221.184858 | 201.317301 | 201.317301 | 0.000000 | N/A | N/A |
| Carnator Boundary B5 | 7 | 2 | 1 | 0.500000 | 231.683165 | 231.683165 | 188.261614 | 188.261614 | 0.000000 | N/A | N/A |
