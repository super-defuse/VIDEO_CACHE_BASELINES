# CARNATOR Latent Reuse Report

Updated: `2026-09-19 04:59:41 UTC`
Status: `completed`

## Config

- Backend: `native Wan2.2`
- Model: `Wan2.2-TI2V-5B`
- Task: `T2V`
- Prompts: source `a sunflower is swaying in a garden, focus on the sunflower`, target `a tulip is swaying in a garden, focus on the tulip`
- Shape: `1280x704`, frames `121`, fps `24`
- Sampling: `50` steps, `unipc`, solver order `2`, flow shift `5.0`
- Guidance: `5.0` (CFG), region-update layers `10-20` with middle attention-mask `True`, outer `sparge-sage2` (n/a) on `0-9,21-29`, FFN `w8a8`, reuse start step `10`
- latent injection: `boundary-blend` (alpha `smoothstep`)
- attention mask: `fusion`, completed probe step `5`, self layer `15`

## Results

| Run | Self-attn | Cross-attn | FFN | Outer sparse density | Outer mask mass coverage | Latency denoising | Speed Up denoising | Latency End2End | Speed Up End2End | Peak reserved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Source dense | 321.587 s | 16.231 s | 35.206 s | n/a | n/a | 417.618 s | n/a | 446.893 s | n/a | 52.93 GiB |
| C0 full dense target | 178.286 s | 16.683 s | 35.637 s | 100.000% | 100.000% | 275.447 s | 1.00x | 307.234 s | 1.00x | 52.93 GiB |
| Target latent reuse | 41.714 s | 12.181 s | 23.017 s | 34.897% | 96.062% | 157.213 s | 1.75x | 187.473 s | 1.64x | 55.87 GiB |

## Artifacts

- Source dense video: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/20260919_114625/sunflower-to-tulip/_sources/carnator_boundary_b5/videos/source_dense.mp4`
- Source first frame: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/20260919_114625/sunflower-to-tulip/_native_results/ti2v-5b/carnator_boundary_b5/B5/frames/source_dense_first_frame.png`
- Target dense video: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/20260919_114625/sunflower-to-tulip/dense_baseline/videos/target_dense.mp4`
- Target first frame: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/20260919_114625/sunflower-to-tulip/_native_results/ti2v-5b/carnator_boundary_b5/B5/frames/target_dense_first_frame.png`
- Target latent reuse video: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/20260919_114625/sunflower-to-tulip/ti2v-5b/carnator_boundary_b5/B5/videos/target_latent_reuse.mp4`
- Summary JSON: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/20260919_114625/sunflower-to-tulip/_native_results/ti2v-5b/carnator_boundary_b5/B5/summary.json`

## Reuse

- Sparse calls: `880`
- Dense fallback calls: `0`
- Avg Medit query ratio: `83.490%`
- Avg Msee context ratio: `94.824%`
- Middle attention-mask: `True`; masked Medit query ratio `83.490%`; masked Msee context ratio `94.824%`
- Latent injection: `{'enabled': True, 'mode': 'boundary-blend', 'alpha_function': 'smoothstep', 'formula': 'L_new = Mbase * L_target + Mband * ((1 - alpha) * L_target + alpha * L_source) + (1 - Medit) * L_source', 'calls': 40, 'total_seconds': 0.2212014445103705, 'avg_ms': 5.530036112759262, 'mask_source_completed_step': 5, 'mask_statistics': {'base_fraction': 0.5300220251083374, 'band_fraction': 0.30487537384033203, 'edit_fraction': 0.8348973989486694, 'alpha_mean': 0.2803572714328766, 'alpha_min': 0.0, 'alpha_max': 1.0}}`
- Source mask prepare: `{'entries': 2400, 'decoded_entries': 2400, 'file_bytes': 5267354400, 'device': 'cuda:0', 'elapsed_seconds': 2.1731679728254676}`
- attention mask variant: `self_contrast_cross_attention_fusion`
- attention mask dir: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/20260919_114625/sunflower-to-tulip/_sources/carnator_boundary_b5/cache/source_dense/fusion_masks`
- attention mask token area: `{'mbase': 0.5300219655036926, 'medit': 0.8348973393440247, 'msee': 0.9482404589653015, 'original_mbase': 0.7266862392425537}`
- attention mask latent area: `{'mbase': 0.5300219655036926, 'medit': 0.8348973393440247, 'msee': 0.9482404589653015, 'mband': 0.30487537384033203}`

## Environment

- Python: `/data/shared/conda_envs/wan/bin/python`
- Model dir: `/home/tangxuebin/models/Wan2.2-TI2V-5B`
- Wan root: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/external/wan2.2`
- CUDA available: `True`
- Flash attention: `{'fa2': True, 'fa3': False}`
- Model layout: `{'exists': True, 'native_config_json': True, 'vae_checkpoint': True, 't5_checkpoint': True, 't5_tokenizer': True}`
