# CARNATOR Latent Reuse Report

Updated: `2026-09-19 05:14:58 UTC`
Status: `completed`

## Config

- Backend: `native Wan2.2`
- Model: `Wan2.2-TI2V-5B`
- Task: `T2V`
- Prompts: source `a lighthouse is standing on a rocky coast, focus on the lighthouse`, target `a windmill is standing on a rocky coast, focus on the windmill`
- Shape: `1280x704`, frames `121`, fps `24`
- Sampling: `50` steps, `unipc`, solver order `2`, flow shift `5.0`
- Guidance: `5.0` (CFG), region-update layers `10-20` with middle attention-mask `True`, outer `sparge-sage2` (n/a) on `0-9,21-29`, FFN `w8a8`, reuse start step `10`
- latent injection: `boundary-blend` (alpha `smoothstep`)
- attention mask: `fusion`, completed probe step `5`, self layer `15`

## Results

| Run | Self-attn | Cross-attn | FFN | Outer sparse density | Outer mask mass coverage | Latency denoising | Speed Up denoising | Latency End2End | Speed Up End2End | Peak reserved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Source dense | 317.831 s | 16.444 s | 36.020 s | n/a | n/a | 414.481 s | n/a | 443.557 s | n/a | 52.93 GiB |
| C0 full dense target | 177.877 s | 16.585 s | 35.375 s | 100.000% | 100.000% | 274.606 s | 1.00x | 307.397 s | 1.00x | 52.93 GiB |
| Target latent reuse | 40.814 s | 11.571 s | 21.594 s | 32.490% | 95.918% | 144.633 s | 1.90x | 163.988 s | 1.87x | 59.67 GiB |

## Artifacts

- Source dense video: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/20260919_114625/lighthouse-to-windmill/_sources/carnator_boundary_b5_targetresume/videos/source_dense.mp4`
- Source first frame: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/20260919_114625/lighthouse-to-windmill/_native_results/ti2v-5b/carnator_boundary_b5_targetresume/B5/frames/source_dense_first_frame.png`
- Target dense video: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/20260919_114625/lighthouse-to-windmill/dense_baseline/videos/target_dense.mp4`
- Target first frame: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/20260919_114625/lighthouse-to-windmill/_native_results/ti2v-5b/carnator_boundary_b5_targetresume/B5/frames/target_dense_first_frame.png`
- Target latent reuse video: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/20260919_114625/lighthouse-to-windmill/ti2v-5b/carnator_boundary_b5_targetresume/B5/videos/target_latent_reuse.mp4`
- Summary JSON: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/20260919_114625/lighthouse-to-windmill/_native_results/ti2v-5b/carnator_boundary_b5_targetresume/B5/summary.json`

## Reuse

- Sparse calls: `880`
- Dense fallback calls: `0`
- Avg Medit query ratio: `67.709%`
- Avg Msee context ratio: `86.811%`
- Middle attention-mask: `True`; masked Medit query ratio `67.709%`; masked Msee context ratio `86.811%`
- Latent injection: `{'enabled': True, 'mode': 'boundary-blend', 'alpha_function': 'smoothstep', 'formula': 'L_new = Mbase * L_target + Mband * ((1 - alpha) * L_target + alpha * L_source) + (1 - Medit) * L_source', 'calls': 40, 'total_seconds': 0.18959806812927127, 'avg_ms': 4.739951703231782, 'mask_source_completed_step': 5, 'mask_statistics': {'base_fraction': 0.3442448675632477, 'band_fraction': 0.3328445851802826, 'edit_fraction': 0.6770894527435303, 'alpha_mean': 0.35678333044052124, 'alpha_min': 0.0, 'alpha_max': 1.0}}`
- Source mask prepare: `{'entries': 2400, 'decoded_entries': 2400, 'file_bytes': 5267354400, 'device': 'cuda:0', 'elapsed_seconds': 2.0526601308956742}`
- attention mask variant: `self_contrast_cross_attention_fusion`
- attention mask dir: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/20260919_114625/lighthouse-to-windmill/_sources/carnator_boundary_b5_targetresume/cache/source_dense/fusion_masks`
- attention mask token area: `{'mbase': 0.3442448675632477, 'medit': 0.6770894527435303, 'msee': 0.8681085109710693, 'original_mbase': 0.5215175747871399}`
- attention mask latent area: `{'mbase': 0.3442448675632477, 'medit': 0.6770894527435303, 'msee': 0.8681085109710693, 'mband': 0.3328445851802826}`

## Environment

- Python: `/data/shared/conda_envs/wan/bin/python`
- Model dir: `/home/tangxuebin/models/Wan2.2-TI2V-5B`
- Wan root: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/external/wan2.2`
- CUDA available: `True`
- Flash attention: `{'fa2': True, 'fa3': False}`
- Model layout: `{'exists': True, 'native_config_json': True, 'vae_checkpoint': True, 't5_checkpoint': True, 't5_tokenizer': True}`
