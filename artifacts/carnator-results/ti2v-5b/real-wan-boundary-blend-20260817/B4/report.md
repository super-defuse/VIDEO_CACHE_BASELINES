# CARNATOR Latent Reuse Report

Updated: `2026-08-17 07:24:45 UTC`
Status: `completed`

## Config

- Backend: `native Wan2.2`
- Model: `Wan2.2-TI2V-5B`
- Task: `T2V`
- Prompts: source `a dog is sleeping on a grassland, focus on the dog`, target `a cat is sleeping on a grassland, focus on the cat`
- Shape: `1280x704`, frames `121`, fps `24`
- Sampling: `50` steps, `unipc`, solver order `2`, flow shift `5.0`
- Guidance: `5.0` (CFG), region-update layers `10-20`, outer `flashinfer` (variable) on `0-9,21-29`, FFN `w8a8`, reuse start step `10`
- latent injection: `boundary-blend` (alpha `smoothstep`)
- attention mask: `fusion`, completed probe step `5`, self layer `15`

## Results

| Run | Self-attn | Cross-attn | FFN | Outer sparse density | Outer mask mass coverage | Latency denoising | Speed Up denoising | Latency End2End | Speed Up End2End | Peak reserved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Source dense | 280.320 s | 16.420 s | 36.245 s | n/a | n/a | 377.105 s | n/a | 401.148 s | n/a | 52.93 GiB |
| C0 full dense target | 182.491 s | 16.573 s | 35.352 s | 100.000% | 100.000% | 278.967 s | 1.00x | 303.581 s | 1.00x | 52.93 GiB |
| Target latent reuse | 88.436 s | 10.105 s | 18.883 s | 32.003% | 96.328% | 170.604 s | 1.64x | 196.456 s | 1.55x | 63.57 GiB |

## Artifacts

- Source dense video: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/carnator/ti2v-5b/real-wan-boundary-blend-20260817/history/videos/source_dense.mp4`
- Source first frame: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/carnator-results/ti2v-5b/real-wan-boundary-blend-20260817/B4/frames/source_dense_first_frame.png`
- Target dense video: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/carnator/ti2v-5b/real-wan-boundary-blend-20260817/videos/target_dense.mp4`
- Target first frame: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/carnator-results/ti2v-5b/real-wan-boundary-blend-20260817/B4/frames/target_dense_first_frame.png`
- Target latent reuse video: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/carnator/ti2v-5b/real-wan-boundary-blend-20260817/B4/videos/target_latent_reuse.mp4`
- Summary JSON: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/carnator-results/ti2v-5b/real-wan-boundary-blend-20260817/B4/summary.json`

## Reuse

- Sparse calls: `880`
- Dense fallback calls: `0`
- Avg Medit query ratio: `35.128%`
- Avg Msee context ratio: `67.753%`
- Latent injection: `{'enabled': True, 'mode': 'boundary-blend', 'alpha_function': 'smoothstep', 'formula': 'L_new = Mbase * L_target + Mband * ((1 - alpha) * L_target + alpha * L_source) + (1 - Medit) * L_source', 'calls': 40, 'total_seconds': 0.2261733920313418, 'avg_ms': 5.654334800783545, 'mask_source_completed_step': 5, 'mask_statistics': {'base_fraction': 0.1690615862607956, 'band_fraction': 0.1822214126586914, 'edit_fraction': 0.3512829840183258, 'alpha_mean': 0.5070818066596985, 'alpha_min': 0.0, 'alpha_max': 1.0}}`
- attention mask variant: `self_contrast_cross_attention_fusion`
- attention mask dir: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/carnator/ti2v-5b/real-wan-boundary-blend-20260817/history/cache/source_dense/fusion_masks`
- attention mask token area: `{'mbase': 0.1690615862607956, 'medit': 0.3512829840183258, 'msee': 0.6775293350219727, 'original_mbase': 0.22192081809043884}`
- attention mask latent area: `{'mbase': 0.1690615862607956, 'medit': 0.3512829840183258, 'msee': 0.6775293350219727, 'mband': 0.1822214126586914}`

## Environment

- Python: `/data/shared/conda_envs/wan/bin/python`
- Model dir: `/home/tangxuebin/models/Wan2.2-TI2V-5B`
- Wan root: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/external/wan2.2`
- CUDA available: `True`
- Flash attention: `{'fa2': True, 'fa3': False}`
- Model layout: `{'exists': True, 'native_config_json': True, 'vae_checkpoint': True, 't5_checkpoint': True, 't5_tokenizer': True}`
