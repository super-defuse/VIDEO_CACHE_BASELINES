# Carnator target-prefix latent-resume variant

- Acceptance root: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/20260918_183809`
- Baseline: `carnator_boundary_b5_targetresume`; the accepted `carnator_boundary_b5` is the reference column
- Difference: the reuse stage resumes from the **target** prompt's own latent at step `reuse_start_step - 1` instead of the source trajectory's. Injection formula, alpha function, sparse attention, FFN and scheduler are unchanged.
- Source stage: the acceptance runner's own `source_stage`, so the cache is built by the same command.
- Pair -> GPU: `apple-to-orange`=7, `bear-to-panda`=6, `duck-to-swan`=5, `horse-to-zebra`=3, `lion-to-tiger`=4
- Quality: `quality.py` per pair, `--quality-env vbench`

## Quality

| Pair | Pairs | GPU | CLIP-T | CLIP-T (accepted) | Δ CLIP-T | VBench-Q | VBench-Q (accepted) | Δ VBench-Q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| horse-to-zebra |  | 3 | 0.319091796875 | 0.2451171875 | +0.0740 | 0.6548187452040574 | 0.6600969710577858 | -0.0053 |
| lion-to-tiger |  | 4 | 0.3232421875 | 0.304443359375 | +0.0188 | 0.6682691784670716 | 0.6734560760574073 | -0.0052 |
| duck-to-swan |  | 5 | 0.3408203125 | 0.306884765625 | +0.0339 | 0.610766954558758 | 0.6665886456138366 | -0.0558 |
| bear-to-panda |  | 6 | 0.3271484375 | 0.284912109375 | +0.0422 | 0.6209506611466116 | 0.6253879384474221 | -0.0044 |
| apple-to-orange |  | 7 | 0.354248046875 | 0.31884765625 | +0.0354 | 0.6318323377541794 | 0.6414832718833469 | -0.0097 |
| mean | 5 |  | 0.33291015625 | 0.292041015625 | +0.0409 | 0.6373275754261356 | 0.6534025806119598 | -0.0161 |

Positive Δ means the variant scores higher than the accepted baseline on that pair.

## Speed

| Pair | Pairs | GPU | Source (s) | Prefix (s) | Reuse E2E (s) | Variant E2E (s) | Speedup vs dense (reuse) | Speedup vs dense (variant) | Speedup (accepted) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| horse-to-zebra |  | 3 | 516.7 | 59.2 | 154.7 | 213.9 | 1.97x | 1.42x | 1.85x |
| lion-to-tiger |  | 4 | 515.6 | 66.4 | 167.8 | 234.2 | 1.80x | 1.29x | 1.71x |
| duck-to-swan |  | 5 | 505.2 | 60.0 | 179.6 | 239.5 | 1.69x | 1.26x | 1.60x |
| bear-to-panda |  | 6 | 508.7 | 65.2 | 162.3 | 227.5 | 1.87x | 1.33x | 1.71x |
| apple-to-orange |  | 7 | 516.1 | 62.1 | 147.8 | 209.9 | 2.06x | 1.45x | 1.92x |
| mean | 5 |  | 512.4 | 62.6 | 162.4 | 225.0 | 1.88x | 1.35x | 1.76x |

`Reuse E2E` is the reuse stage alone, measured exactly like the accepted baseline's `end_to_end_seconds`, so `Speedup (reuse)` is directly comparable with `Speedup (accepted)`. `Variant E2E` adds the dense target-prefix run this variant needs to obtain its starting latent, so `Speedup (variant)` is the honest cost of the variant as a whole. `Prefix (s)` is that run's own end-to-end time; the source stage is shared with the accepted baseline and is therefore excluded from both speedups, as it is in the acceptance table.

## Source consistency

Each pair's source cache was rebuilt from scratch and compared against the fingerprint the accepted run recorded, so a PASS means the variant really was fed the same source trajectory:

- `horse-to-zebra`: **PASS** - source cache matches the accepted run byte for byte
- `lion-to-tiger`: **PASS** - source cache matches the accepted run byte for byte
- `duck-to-swan`: **PASS** - source cache matches the accepted run byte for byte
- `bear-to-panda`: **PASS** - source cache matches the accepted run byte for byte
- `apple-to-orange`: **PASS** - source cache matches the accepted run byte for byte
