# Chorus2

Chorus2 is a native Wan2.2-TI2V-5B baseline. Source generation captures per-step,
per-block self-attention masks. Target starts from its own initial latent and runs
all 50 denoising steps; steps 1--10 use the normal attention path and steps 11--50
use the cached masks through SpargeAttn on all 30 blocks.
The final block of every eight-layer group also receives blocks not selected by
earlier layers in that group, matching Chorus2's mask refresh behavior.

SpargeAttn is the default sparse backend. A complete existing Chorus2 source-mask
cache can be passed to skip source generation; no source latent cache is required.
The unified Dense preparation entry creates the masks from one Carnator-style source
run. Chorus2 directly reads Carnator's complete raw block-mask cache; block 10--20
remain unfiltered normal sparse masks in that one cache. Carnator applies its
target-only `Medit`/`Msee` token selection at execution time, so it never mutates
the shared files.

Prepare the source cache and the universal C0 Dense denominator on GPU 3--7:

```bash
CUDA_VISIBLE_DEVICES=3 conda run -n wan env PYTHONNOUSERSITE=1 PYTHONPATH=. CARNATOR_ENV_READY=1 \
  python -m carnator.adapters.wan_ti2v_5b --no-env-setup --device cuda:0 \
  --artifact-root artifacts/acceptance/wan22-ti2v-5b-dog-to-cat \
  --results-root artifacts/acceptance/wan22-ti2v-5b-dog-to-cat/results \
  --prepare-shared-dense-cache --run-id shared-dense-gpu3
```

Then point Chorus2 at the exported cache:

```bash
conda run -n wan env PYTHONNOUSERSITE=1 PYTHONPATH=. python scripts/run/chorus2_wan.py \
  --artifact-root artifacts/acceptance/wan22-ti2v-5b-dog-to-cat \
  --results-root artifacts/acceptance/wan22-ti2v-5b-dog-to-cat/results \
  --source-prompt "a dog is sleeping on a grassland, focus on the dog" \
  --target-prompt "a cat is sleeping on a grassland, focus on the cat" \
  --source-mask-cache artifacts/acceptance/wan22-ti2v-5b-dog-to-cat/shared-dense-gpu3/cache/source_dense/outer_block_masks \
  --speedup-reference-manifest artifacts/acceptance/wan22-ti2v-5b-dog-to-cat/shared-dense-gpu3/cache/target_dense/manifest.json \
  --seed 20260218 --run-id chorus2-shared-cache-gpu6
```

The result summary verifies the 50-step target, SpargeAttn coverage from step 11,
layer refresh, and no dense fallback; the stage manifest records the GPU used for
each run. Its speedup uses the shared run's target C0 end-to-end time.
