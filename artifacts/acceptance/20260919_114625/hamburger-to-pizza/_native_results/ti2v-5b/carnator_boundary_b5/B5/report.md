# CARNATOR Latent Reuse Report

Updated: `2026-09-19 04:28:44 UTC`
Status: `completed`

## Config

- Backend: `native Wan2.2`
- Model: `Wan2.2-TI2V-5B`
- Task: `T2V`
- Prompts: source `a hamburger is resting on a white plate, focus on the hamburger`, target `a pizza is resting on a white plate, focus on the pizza`
- Shape: `1280x704`, frames `121`, fps `24`
- Sampling: `50` steps, `unipc`, solver order `2`, flow shift `5.0`
- Guidance: `5.0` (CFG), region-update layers `10-20` with middle attention-mask `True`, outer `sparge-sage2` (n/a) on `0-9,21-29`, FFN `w8a8`, reuse start step `10`
- latent injection: `boundary-blend` (alpha `smoothstep`)
- attention mask: `fusion`, completed probe step `5`, self layer `15`

## Results

| Run | Self-attn | Cross-attn | FFN | Outer sparse density | Outer mask mass coverage | Latency denoising | Speed Up denoising | Latency End2End | Speed Up End2End | Peak reserved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Source dense | 321.932 s | 16.200 s | 35.037 s | n/a | n/a | 417.808 s | n/a | 446.615 s | n/a | 52.93 GiB |
| C0 full dense target | 177.898 s | 16.623 s | 35.473 s | 100.000% | 100.000% | 274.782 s | 1.00x | 306.641 s | 1.00x | 52.93 GiB |
| Target latent reuse | 42.884 s | 10.766 s | 20.416 s | 38.942% | 95.567% | 137.258 s | 2.00x | 164.773 s | 1.86x | 55.67 GiB |

## Artifacts

- Source dense video: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/20260919_114625/hamburger-to-pizza/_sources/carnator_boundary_b5/videos/source_dense.mp4`
- Source first frame: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/20260919_114625/hamburger-to-pizza/_native_results/ti2v-5b/carnator_boundary_b5/B5/frames/source_dense_first_frame.png`
- Target dense video: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/20260919_114625/hamburger-to-pizza/dense_baseline/videos/target_dense.mp4`
- Target first frame: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/20260919_114625/hamburger-to-pizza/_native_results/ti2v-5b/carnator_boundary_b5/B5/frames/target_dense_first_frame.png`
- Target latent reuse video: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/20260919_114625/hamburger-to-pizza/ti2v-5b/carnator_boundary_b5/B5/videos/target_latent_reuse.mp4`
- Summary JSON: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/20260919_114625/hamburger-to-pizza/_native_results/ti2v-5b/carnator_boundary_b5/B5/summary.json`

## Reuse

- Sparse calls: `880`
- Dense fallback calls: `0`
- Avg Medit query ratio: `53.292%`
- Avg Msee context ratio: `80.029%`
- Middle attention-mask: `True`; masked Medit query ratio `53.292%`; masked Msee context ratio `80.029%`
- Latent injection: `{'enabled': True, 'mode': 'boundary-blend', 'alpha_function': 'smoothstep', 'formula': 'L_new = Mbase * L_target + Mband * ((1 - alpha) * L_target + alpha * L_source) + (1 - Medit) * L_source', 'calls': 40, 'total_seconds': 0.19202661886811256, 'avg_ms': 4.800665471702814, 'mask_source_completed_step': 5, 'mask_statistics': {'base_fraction': 0.32917889952659607, 'band_fraction': 0.2037390023469925, 'edit_fraction': 0.5329179167747498, 'alpha_mean': 0.511335015296936, 'alpha_min': 0.0, 'alpha_max': 1.0}}`
- Source mask prepare: `{'entries': 2400, 'decoded_entries': 2400, 'file_bytes': 5267354400, 'device': 'cuda:0', 'elapsed_seconds': 2.4407073808833957}`
- attention mask variant: `self_contrast_cross_attention_fusion`
- attention mask dir: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/20260919_114625/hamburger-to-pizza/_sources/carnator_boundary_b5/cache/source_dense/fusion_masks`
- attention mask token area: `{'mbase': 0.32917889952659607, 'medit': 0.5329179167747498, 'msee': 0.8002932667732239, 'original_mbase': 0.40538856387138367}`
- attention mask latent area: `{'mbase': 0.32917889952659607, 'medit': 0.5329179167747498, 'msee': 0.8002932667732239, 'mband': 0.2037390023469925}`

## Environment

- Python: `/data/shared/conda_envs/wan/bin/python`
- Model dir: `/home/tangxuebin/models/Wan2.2-TI2V-5B`
- Wan root: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/external/wan2.2`
- CUDA available: `True`
- Flash attention: `{'fa2': True, 'fa3': False}`
- Model layout: `{'exists': True, 'native_config_json': True, 'vae_checkpoint': True, 't5_checkpoint': True, 't5_tokenizer': True}`
