# Carnator target-prefix latent-resume variant

- Acceptance root: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/20260919_114625`
- Baseline: `carnator_boundary_b5_targetresume`; the accepted `carnator_boundary_b5` is the reference column
- Difference: the reuse stage resumes from the **target** prompt's own latent at step `reuse_start_step - 1` instead of the source trajectory's. Injection formula, alpha function, sparse attention, FFN and scheduler are unchanged.
- Source stage: the acceptance runner's own `source_stage`, so the cache is built by the same command.
- Pair -> GPU: `airplane-to-helicopter`=4, `car-to-truck`=3, `hamburger-to-pizza`=5, `lighthouse-to-windmill`=6, `sunflower-to-tulip`=7
- Quality: `quality.py` per pair, `--quality-env vbench`

## Quality

| Pair | Pairs | GPU | CLIP-T | CLIP-T (accepted) | Δ CLIP-T | VBench-Q | VBench-Q (accepted) | Δ VBench-Q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| car-to-truck |  | 3 | 0.32568359375 | 0.2939453125 | +0.0317 | 0.6682027049287982 | 0.6715572623866465 | -0.0034 |
| airplane-to-helicopter |  | 4 | 0.3359375 | 0.3486328125 | -0.0127 | 0.6487856444915102 | 0.663494614707442 | -0.0147 |
| hamburger-to-pizza |  | 5 | 0.30078125 | 0.1943359375 | +0.1064 | 0.6284090332606436 | 0.6431210571983172 | -0.0147 |
| lighthouse-to-windmill |  | 6 | 0.28759765625 | 0.264404296875 | +0.0232 | 0.6156295104156186 | 0.6356186786968439 | -0.0200 |
| sunflower-to-tulip |  | 7 | 0.318115234375 | 0.210205078125 | +0.1079 | 0.6232512208142744 | 0.624612208206399 | -0.0014 |
| mean | 5 |  | 0.313623046875 | 0.2623046875 | +0.0513 | 0.636855622782169 | 0.6476807642391297 | -0.0108 |

Positive Δ means the variant scores higher than the accepted baseline on that pair.

## Speed

| Pair | Pairs | GPU | Source (s) | Prefix (s) | Reuse E2E (s) | Variant E2E (s) | Speedup vs dense (reuse) | Speedup vs dense (variant) | Speedup (accepted) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| car-to-truck |  | 3 | 523.7 | 59.4 | 163.4 | 222.7 | 1.86x | 1.37x | 1.81x |
| airplane-to-helicopter |  | 4 | 515.5 | 63.2 | 148.6 | 211.7 | 2.05x | 1.44x | 1.95x |
| hamburger-to-pizza |  | 5 | 508.8 | 62.4 | 153.2 | 215.6 | 2.00x | 1.42x | 1.86x |
| lighthouse-to-windmill |  | 6 | 517.5 | 61.2 | 164.0 | 225.2 | 1.87x | 1.36x | 1.76x |
| sunflower-to-tulip |  | 7 | 520.5 | 60.7 | 176.8 | 237.5 | 1.74x | 1.29x | 1.64x |
| mean | 5 |  | 517.2 | 61.4 | 161.2 | 222.5 | 1.91x | 1.38x | 1.81x |

`Reuse E2E` is the reuse stage alone, measured exactly like the accepted baseline's `end_to_end_seconds`, so `Speedup (reuse)` is directly comparable with `Speedup (accepted)`. `Variant E2E` adds the dense target-prefix run this variant needs to obtain its starting latent, so `Speedup (variant)` is the honest cost of the variant as a whole. `Prefix (s)` is that run's own end-to-end time; the source stage is shared with the accepted baseline and is therefore excluded from both speedups, as it is in the acceptance table.

## Source consistency

Each pair's source cache was rebuilt from scratch and compared against the fingerprint the accepted run recorded, so a PASS means the variant really was fed the same source trajectory:

- `car-to-truck`: **PASS** - source cache matches the accepted run byte for byte
- `airplane-to-helicopter`: **PASS** - source cache matches the accepted run byte for byte
- `hamburger-to-pizza`: **PASS** - source cache matches the accepted run byte for byte
- `lighthouse-to-windmill`: **PASS** - source cache matches the accepted run byte for byte
- `sunflower-to-tulip`: **PASS** - source cache matches the accepted run byte for byte
