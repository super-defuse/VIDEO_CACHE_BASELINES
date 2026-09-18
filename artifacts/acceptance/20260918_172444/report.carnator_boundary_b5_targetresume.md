# Carnator target-prefix latent-resume variant

- Acceptance root: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/20260918_172444`
- Baseline: `carnator_boundary_b5_targetresume`; the accepted `carnator_boundary_b5` is the reference column
- Difference: the reuse stage resumes from the **target** prompt's own latent at step `reuse_start_step - 1` instead of the source trajectory's. Injection formula, alpha function, sparse attention, FFN and scheduler are unchanged.
- Source stage: the acceptance runner's own `source_stage`, so the cache is built by the same command.
- Pair -> GPU: `pair-000`=3
- Quality: `quality.py` per pair, `--quality-env vbench`

## Quality

| Pair | Pairs | GPU | CLIP-T | CLIP-T (accepted) | Δ CLIP-T | VBench-Q | VBench-Q (accepted) | Δ VBench-Q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pair-000 |  | 3 | 0.342529296875 | 0.354736328125 | -0.0122 | 0.6315007305077583 | 0.6326169480667867 | -0.0011 |
| mean | 1 |  | 0.342529296875 | 0.354736328125 | -0.0122 | 0.6315007305077583 | 0.6326169480667867 | -0.0011 |

Positive Δ means the variant scores higher than the accepted baseline on that pair.

## Speed

| Pair | Pairs | GPU | Source (s) | Prefix (s) | Reuse E2E (s) | Variant E2E (s) | Speedup vs dense (reuse) | Speedup vs dense (variant) | Speedup (accepted) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pair-000 |  | 3 | 497.0 | 58.6 | 143.1 | 201.8 | 2.15x | 1.53x | 2.03x |
| mean | 1 |  | 497.0 | 58.6 | 143.1 | 201.8 | 2.15x | 1.53x | 2.03x |

`Reuse E2E` is the reuse stage alone, measured exactly like the accepted baseline's `end_to_end_seconds`, so `Speedup (reuse)` is directly comparable with `Speedup (accepted)`. `Variant E2E` adds the dense target-prefix run this variant needs to obtain its starting latent, so `Speedup (variant)` is the honest cost of the variant as a whole. `Prefix (s)` is that run's own end-to-end time; the source stage is shared with the accepted baseline and is therefore excluded from both speedups, as it is in the acceptance table.

## Source consistency

Each pair's source cache was rebuilt from scratch and compared against the fingerprint the accepted run recorded, so a PASS means the variant really was fed the same source trajectory:

- `pair-000`: **PASS** - source cache matches the accepted run byte for byte
