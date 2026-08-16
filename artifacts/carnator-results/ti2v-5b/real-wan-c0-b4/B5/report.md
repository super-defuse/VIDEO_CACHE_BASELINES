# CARNATOR Latent Reuse Report

Updated: `2026-08-15 07:29:57 UTC`
Status: `completed`

## Config

- Backend: `native Wan2.2`
- Model: `Wan2.2-TI2V-5B`
- Task: `T2V`
- Prompts: source `a dog is sleeping on a grassland, focus on the dog`, target `a cat is sleeping on a grassland, focus on the cat`
- Shape: `1280x704`, frames `121`, fps `24`
- Sampling: `50` steps, `unipc`, solver order `2`, flow shift `5.0`
- Guidance: `5.0` (CFG), region-update layers `10-20`, outer `sparge-sage2` (n/a) on `0-9,21-29`, FFN `w8a8`, reuse start step `10`
- attention mask: `fusion`, completed probe step `5`, self layer `15`

## Results

| Run | Self-attn | Cross-attn | FFN | Outer sparse density | Outer mask mass coverage | Latency denoising | Speed Up denoising | Latency End2End | Speed Up End2End | Peak reserved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Source dense | 279.671 s | 16.407 s | 36.134 s | n/a | n/a | 376.208 s | n/a | 398.931 s | n/a | 52.93 GiB |
| C0 full dense target | 181.546 s | 16.459 s | 34.977 s | 100.000% | 100.000% | 277.363 s | 1.00x | 300.641 s | 1.00x | 52.93 GiB |
| Target latent reuse | 40.940 s | 10.149 s | 18.573 s | 32.003% | 96.328% | 123.213 s | 2.25x | 149.092 s | 2.02x | 50.48 GiB |

## Artifacts

- Source dense video: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/carnator/ti2v-5b/real-wan-c0-b4/videos/source_dense.mp4`
- Source first frame: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/carnator-results/ti2v-5b/real-wan-c0-b4/B5/frames/source_dense_first_frame.png`
- Target dense video: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/carnator/ti2v-5b/real-wan-c0-b4/videos/target_dense.mp4`
- Target first frame: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/carnator-results/ti2v-5b/real-wan-c0-b4/B5/frames/target_dense_first_frame.png`
- Target latent reuse video: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/carnator/ti2v-5b/real-wan-c0-b4/B5/videos/target_latent_reuse.mp4`
- Summary JSON: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/carnator-results/ti2v-5b/real-wan-c0-b4/B5/summary.json`

## Reuse

- Sparse calls: `880`
- Dense fallback calls: `0`
- Avg Medit query ratio: `35.128%`
- Avg Msee context ratio: `67.753%`
- Latent injection: `{'enabled': True, 'formula': 'L_new = Medit * L_target + (1 - Medit) * L_source_cache', 'calls': 40, 'total_seconds': 0.11033530486747622, 'avg_ms': 2.7583826216869056, 'mask_source_completed_step': 5}`
- attention mask variant: `self_contrast_cross_attention_fusion`
- attention mask dir: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/carnator/ti2v-5b/real-wan-c0-b4/cache/source_dense/fusion_masks`
- attention mask token area: `{'mbase': 0.1690615862607956, 'medit': 0.3512829840183258, 'msee': 0.6775293350219727, 'original_mbase': 0.22192081809043884}`
- attention mask latent area: `{'medit': 0.3512829840183258, 'msee': 0.6775293350219727}`

## Environment

- Python: `/data/shared/conda_envs/wan/bin/python`
- Model dir: `/home/tangxuebin/models/Wan2.2-TI2V-5B`
- Wan root: `/home/tangxuebin/project/CARNATOR_ICLR2027/external/Wan2.2`
- CUDA available: `True`
- Flash attention: `{'fa2': True, 'fa3': False}`
- Model layout: `{'exists': True, 'native_config_json': True, 'vae_checkpoint': True, 't5_checkpoint': True, 't5_tokenizer': True}`
