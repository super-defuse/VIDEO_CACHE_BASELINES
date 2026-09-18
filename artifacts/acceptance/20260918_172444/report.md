# Acceptance experiment

- Run root: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/20260918_172444`
- Layout: `per-baseline-source-v1`
- GPU plan: `dense_baseline`=3, `nirvana_vid`=4, `chorus`=5, `chorus2`=6, `carnator_boundary_b5`=7
- Quality GPU: `3`
- Source caches: one independent run per baseline, pruned once its target succeeded
- Source cache `pair-000`/`carnator_boundary_b5`: `9636ea6642b9429499880e6226b2452d50533ffb6064ea07962a1c4fee7e6fcb` (/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/20260918_172444/_sources/carnator_boundary_b5/manifest.json)
- Source cache `pair-000`/`chorus`: `0a66e8176b40e224e862677071ceb0c081ca9d77cc7037ce0db186ff57a187ca` (/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/20260918_172444/_sources/chorus/manifest.json)
- Source cache `pair-000`/`chorus2`: `34437465dfdd23c3bd4ce0672a07a290167c3aee6746a4e80d3761cfd82c97d7` (/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/20260918_172444/_sources/chorus2/manifest.json)
- Source cache `pair-000`/`nirvana_vid`: `7c92e08c296d0080b33b7d4a8955802de6ef682910ff7fceee3e58ba91b7f060` (/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/20260918_172444/_sources/nirvana_vid/manifest.json)

| Baseline | Status | GPU | Denoising (s) | End-to-end (s) | Speedup (denoise) | Speedup (end-to-end) | CLIP-T | VBench-Q |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense_baseline | ok | 3 | 275.45753143634647 | 308.11188800120726 | 1.00x | 1.00x | 0.337158203125 | 0.6330714069844214 |
| nirvana_vid | ok | 4 | 235.2758046220988 | 237.9354821019806 | 1.17x | 1.29x | 0.355712890625 | 0.6362940693916224 |
| chorus | ok | 5 | 196.03220169106498 | 254.79314185399562 | 1.41x | 1.21x | 0.35546875 | 0.6112559992677009 |
| chorus2 | ok | 6 | 201.82858257181942 | 227.02440326707438 | 1.36x | 1.36x | 0.33447265625 | 0.6269709150927067 |
| carnator_boundary_b5 | ok | 7 | 124.10527424886823 | 151.80145765608177 | 2.22x | 2.03x | 0.354736328125 | 0.6326169480667867 |

## Source consistency

- `pair-000`: **PASS** - 4 independent source runs (`carnator_boundary_b5`, `chorus`, `chorus2`, `nirvana_vid`)
