# CARNATOR Latent Reuse Report

Updated: `2026-08-24 19:01:48 UTC`
Status: `completed`

## Config

- Backend: `native Wan2.2`
- Model: `Wan2.2-TI2V-5B`
- Task: `T2V`
- Prompts: source `a dog is sleeping on a grassland, focus on the dog`, target `a cat is sleeping on a grassland, focus on the cat`
- Shape: `1280x704`, frames `121`, fps `24`
- Sampling: `50` steps, `unipc`, solver order `2`, flow shift `5.0`
- Guidance: `5.0` (CFG), region-update layers `10-20` with middle attention-mask `True`, outer `sparge-sage2` (n/a) on `0-9,21-29`, FFN `w8a8`, reuse start step `10`
- latent injection: `hard` (alpha `smoothstep`)
- attention mask: `fusion`, completed probe step `5`, self layer `15`

## Results

| Run | Self-attn | Cross-attn | FFN | Outer sparse density | Outer mask mass coverage | Latency denoising | Speed Up denoising | Latency End2End | Speed Up End2End | Peak reserved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Source dense | 350.773 s | 16.214 s | 36.745 s | n/a | n/a | 447.488 s | n/a | 474.806 s | n/a | 52.93 GiB |
| C0 full dense target | 178.285 s | 16.519 s | 34.998 s | 100.000% | 100.000% | 274.479 s | 1.00x | 294.078 s | 1.00x | 52.90 GiB |
| Target latent reuse | 41.314 s | 9.958 s | 19.167 s | 32.042% | 96.324% | 124.785 s | 2.20x | 157.343 s | 1.87x | 50.47 GiB |

## Artifacts

- Source dense video: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/wan22-ti2v-5b-dog-to-cat/shared-dense-gpu3/videos/source_dense.mp4`
- Source first frame: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/wan22-ti2v-5b-dog-to-cat/results/carnator-hard-B5-gpu4/frames/source_dense_first_frame.png`
- Target dense video: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/wan22-ti2v-5b-dog-to-cat/shared-dense-gpu3/videos/target_dense.mp4`
- Target first frame: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/wan22-ti2v-5b-dog-to-cat/results/carnator-hard-B5-gpu4/frames/target_dense_first_frame.png`
- Target latent reuse video: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/wan22-ti2v-5b-dog-to-cat/carnator-hard-B5-gpu4/videos/target_latent_reuse.mp4`
- Summary JSON: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/wan22-ti2v-5b-dog-to-cat/results/carnator-hard-B5-gpu4/summary.json`

## Reuse

- Sparse calls: `880`
- Dense fallback calls: `0`
- Avg Medit query ratio: `35.205%`
- Avg Msee context ratio: `67.881%`
- Middle attention-mask: `True`; masked Medit query ratio `35.205%`; masked Msee context ratio `67.881%`
- Latent injection: `{'enabled': True, 'mode': 'hard', 'alpha_function': None, 'formula': 'L_new = Medit * L_target + (1 - Medit) * L_source_cache', 'calls': 40, 'total_seconds': 0.16574354807380587, 'avg_ms': 4.143588701845147, 'mask_source_completed_step': 5, 'mask_statistics': {'mode': 'hard', 'base_fraction': 0.16942815482616425, 'band_fraction': 0.0, 'edit_fraction': 0.3520527780056}}`
- attention mask variant: `self_contrast_cross_attention_fusion`
- attention mask dir: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/wan22-ti2v-5b-dog-to-cat/shared-dense-gpu3/cache/source_dense/fusion_masks`
- attention mask token area: `{'mbase': 0.16942815482616425, 'medit': 0.3520527780056, 'msee': 0.6788123250007629, 'original_mbase': 0.2219574749469757}`
- attention mask latent area: `{'mbase': 0.16942815482616425, 'medit': 0.3520527780056, 'msee': 0.6788123250007629, 'mband': 0.18262463808059692}`

## Environment

- Python: `/data/shared/conda_envs/wan/bin/python`
- Model dir: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/models/Wan2.2-TI2V-5B`
- Wan root: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/external/wan2.2`
- CUDA available: `True`
- Flash attention: `{'fa2': True, 'fa3': False}`
- Model layout: `{'exists': True, 'native_config_json': True, 'vae_checkpoint': True, 't5_checkpoint': True, 't5_tokenizer': True}`
