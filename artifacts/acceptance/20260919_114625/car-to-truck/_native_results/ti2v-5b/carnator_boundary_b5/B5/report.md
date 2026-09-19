# CARNATOR Latent Reuse Report

Updated: `2026-09-19 03:59:14 UTC`
Status: `completed`

## Config

- Backend: `native Wan2.2`
- Model: `Wan2.2-TI2V-5B`
- Task: `T2V`
- Prompts: source `a car is driving along a coastal highway, focus on the car`, target `a truck is driving along a coastal highway, focus on the truck`
- Shape: `1280x704`, frames `121`, fps `24`
- Sampling: `50` steps, `unipc`, solver order `2`, flow shift `5.0`
- Guidance: `5.0` (CFG), region-update layers `10-20` with middle attention-mask `True`, outer `sparge-sage2` (n/a) on `0-9,21-29`, FFN `w8a8`, reuse start step `10`
- latent injection: `boundary-blend` (alpha `smoothstep`)
- attention mask: `fusion`, completed probe step `5`, self layer `15`

## Results

| Run | Self-attn | Cross-attn | FFN | Outer sparse density | Outer mask mass coverage | Latency denoising | Speed Up denoising | Latency End2End | Speed Up End2End | Peak reserved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Source dense | 321.173 s | 16.299 s | 35.056 s | n/a | n/a | 417.165 s | n/a | 449.240 s | n/a | 52.93 GiB |
| C0 full dense target | 177.820 s | 16.634 s | 35.464 s | 100.000% | 100.000% | 274.657 s | 1.00x | 304.598 s | 1.00x | 52.93 GiB |
| Target latent reuse | 40.876 s | 11.239 s | 21.273 s | 32.344% | 95.892% | 142.876 s | 1.92x | 167.952 s | 1.81x | 55.87 GiB |

## Artifacts

- Source dense video: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/20260919_114625/car-to-truck/_sources/carnator_boundary_b5/videos/source_dense.mp4`
- Source first frame: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/20260919_114625/car-to-truck/_native_results/ti2v-5b/carnator_boundary_b5/B5/frames/source_dense_first_frame.png`
- Target dense video: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/20260919_114625/car-to-truck/dense_baseline/videos/target_dense.mp4`
- Target first frame: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/20260919_114625/car-to-truck/_native_results/ti2v-5b/carnator_boundary_b5/B5/frames/target_dense_first_frame.png`
- Target latent reuse video: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/20260919_114625/car-to-truck/ti2v-5b/carnator_boundary_b5/B5/videos/target_latent_reuse.mp4`
- Summary JSON: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/20260919_114625/car-to-truck/_native_results/ti2v-5b/carnator_boundary_b5/B5/summary.json`

## Reuse

- Sparse calls: `880`
- Dense fallback calls: `0`
- Avg Medit query ratio: `62.287%`
- Avg Msee context ratio: `89.666%`
- Middle attention-mask: `True`; masked Medit query ratio `62.287%`; masked Msee context ratio `89.666%`
- Latent injection: `{'enabled': True, 'mode': 'boundary-blend', 'alpha_function': 'smoothstep', 'formula': 'L_new = Mbase * L_target + Mband * ((1 - alpha) * L_target + alpha * L_source) + (1 - Medit) * L_source', 'calls': 40, 'total_seconds': 0.20693135168403387, 'avg_ms': 5.173283792100847, 'mask_source_completed_step': 5, 'mask_statistics': {'base_fraction': 0.2952346205711365, 'band_fraction': 0.32763931155204773, 'edit_fraction': 0.6228739023208618, 'alpha_mean': 0.43977752327919006, 'alpha_min': 0.0, 'alpha_max': 1.0}}`
- Source mask prepare: `{'entries': 2400, 'decoded_entries': 2400, 'file_bytes': 5267354400, 'device': 'cuda:0', 'elapsed_seconds': 2.3939749309793115}`
- attention mask variant: `self_contrast_cross_attention_fusion`
- attention mask dir: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/20260919_114625/car-to-truck/_sources/carnator_boundary_b5/cache/source_dense/fusion_masks`
- attention mask token area: `{'mbase': 0.2952345907688141, 'medit': 0.6228739023208618, 'msee': 0.8966642022132874, 'original_mbase': 0.4707111418247223}`
- attention mask latent area: `{'mbase': 0.2952345907688141, 'medit': 0.6228739023208618, 'msee': 0.8966642022132874, 'mband': 0.32763928174972534}`

## Environment

- Python: `/data/shared/conda_envs/wan/bin/python`
- Model dir: `/home/tangxuebin/models/Wan2.2-TI2V-5B`
- Wan root: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/external/wan2.2`
- CUDA available: `True`
- Flash attention: `{'fa2': True, 'fa3': False}`
- Model layout: `{'exists': True, 'native_config_json': True, 'vae_checkpoint': True, 't5_checkpoint': True, 't5_tokenizer': True}`
