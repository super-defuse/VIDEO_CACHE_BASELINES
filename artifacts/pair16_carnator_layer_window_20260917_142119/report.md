# Carnator layer-window ablation (pair: arrival 16)

- target: `A cute little lion and a wobbly penguin appear` (`ed46710a-2ea6-5a63-87f9-7e67a9f41743`, seed 20260234)
- source: `A tiger and a lion are fighting` (`e9b92745-1872-59b5-9de5-c839a0196d1b`)
- output: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/pair16_carnator_layer_window_20260917_142119`

Primary columns follow the 300-request table. `Denoising s` is `denoising_s` (CUDA-synchronised DiT forward, probe counted, cache prepare excluded) and `E2E s` is `latency_s` (request wall time, cache prepare excluded); both are directly comparable to the recorded arrival-16 row (denoising 155.31 s, E2E 201.32 s) and to the 300-request Carnator means (CLIP-T 0.261571, VBench-Q 0.640110). Speedups divide the dense arm's own numbers by each arm's.

Secondary columns are the native target-stage values. `DiT s` is `dit_forward_seconds` for the reuse stage alone: dense runs 50 dense steps against each arm's 40 reuse steps, so its speedup is an equal-work DiT comparison. The probe runs in its own stage, which is why the target-stage `E2E s` sits below `Denoising s`.

| arm | layers | `--sparse-layers` | Denoising s | Denoising speedup | E2E s | E2E speedup | DiT s | DiT speedup | target-stage E2E s | CLIP-T | VBench-Q |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | 0 | `—` | 273.606 | 1.00x | 296.988 | 1.00x | 273.606 | 1.00x | 296.988 | 0.3330 | 0.6699 |
| l05 | 5 | `13-17` | 160.670 | 1.70x | 206.269 | 1.44x | 133.181 | 2.05x | 153.589 | 0.2681 | 0.6447 |
| l11 | 11 | `10-20` | 156.672 | 1.75x | 200.537 | 1.48x | 129.469 | 2.11x | 149.142 | 0.2676 | 0.6415 |
| l15 | 15 | `8-22` | 150.581 | 1.82x | 193.624 | 1.53x | 124.014 | 2.21x | 143.422 | 0.2700 | 0.6440 |
| l20 | 20 | `6-25` | 150.966 | 1.81x | 196.736 | 1.51x | 123.708 | 2.21x | 143.484 | 0.2803 | 0.6497 |
| l25 | 25 | `3-27` | 148.973 | 1.84x | 194.322 | 1.53x | 121.487 | 2.25x | 141.744 | 0.2749 | 0.6268 |
| l30 | 30 | `0-29` | 145.094 | 1.89x | 189.855 | 1.56x | 117.925 | 2.32x | 137.893 | 0.2617 | 0.6092 |

## Reference

300-request Carnator means: CLIP-T 0.261571, VBench-Q 0.640110; recorded `arrival 16` row: CLIP-T 0.267578, VBench-Q 0.641478, E2E 201.315 s, denoising 155.312 s, executed_steps 45.
