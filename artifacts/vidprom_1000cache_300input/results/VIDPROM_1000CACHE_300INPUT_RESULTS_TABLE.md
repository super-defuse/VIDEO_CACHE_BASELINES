# VidProM results (300 requests)

Carnator records Stage-1 candidates, noun prefilter, Stage-2 Early Signature probe/gate, and the selected source. The source view supplies precomputed sidecars; External is Chorus-only LLM + Grounded-SAM2 time and is zero for other baselines.

| Baseline | GPU | Requests | Hits | Hit rate | E2E s/request | Retrieval ms/request | Denoising s/request | External s/request | CLIP-T | VBench-Q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NIRVANA-VID | 3 | 300 | 214 | 0.713333 | 248.018446 | 102.525036 | 244.827778 | 0.000000 | 0.277944 | 0.649438 |
| TeaCache | 4 | 300 | N/A | N/A | 137.071327 | 0.000000 | 134.231460 | 0.000000 | 0.291166 | 0.608910 |
| Chorus | 5 | 300 | 190 | 0.688406 | 253.205998 | 110.051860 | 214.922190 | 35.393867 | 0.285775 | 0.645655 |
| Chorus2 | 6 | 300 | 214 | 0.713333 | 274.542191 | 39.281762 | 186.674758 | 0.000000 | 0.277558 | 0.647873 |
| Carnator Boundary B5 | 7 | 300 | 104 | 0.346667 | 377.349484 | 202.964816 | 251.788476 | 0.000000 | 0.287384 | 0.647938 |
