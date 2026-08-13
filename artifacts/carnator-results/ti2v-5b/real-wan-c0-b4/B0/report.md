# CARNATOR Latent Reuse Report

Updated: `2026-08-13 10:11:28 UTC`
Status: `completed`

## Config

- Backend: `native Wan2.2`
- Model: `Wan2.2-TI2V-5B`
- Task: `T2V`
- Prompts: source `a dog is sleeping on a grassland, focus on the dog`, target `a cat is sleeping on a grassland, focus on the cat`
- Shape: `1280x704`, frames `121`, fps `24`
- Sampling: `50` steps, `unipc`, solver order `2`, flow shift `5.0`
- Guidance: `5.0` (CFG), region-update layers `10-20`, outer `dense` (n/a) on `0-9,21-29`, FFN `fp`, reuse start step `10`
- attention mask: `fusion`, completed probe step `5`, self layer `15`

## Results

| Run | Self-attn | Cross-attn | FFN | Outer sparse density | Outer mask mass coverage | Latency denoising | Speed Up denoising | Latency End2End | Speed Up End2End | Peak reserved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Source dense | 280.248 s | 16.463 s | 35.997 s | n/a | n/a | 376.712 s | n/a | 399.631 s | n/a | 52.94 GiB |
| C0 full dense target | 182.564 s | 16.553 s | 35.334 s | 100.000% | 100.000% | 278.898 s | 1.00x | 302.089 s | 1.00x | 52.93 GiB |
| Target latent reuse | 92.638 s | 10.256 s | 21.662 s | 100.000% | 100.000% | 178.662 s | 1.56x | 201.280 s | 1.50x | 52.94 GiB |

## Artifacts

- Source dense video: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/carnator/ti2v-5b/real-wan-c0-b4/history/videos/source_dense.mp4`
- Source first frame: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/carnator-results/ti2v-5b/real-wan-c0-b4/B0/frames/source_dense_first_frame.png`
- Target dense video: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/carnator/ti2v-5b/real-wan-c0-b4/videos/target_dense.mp4`
- Target first frame: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/carnator-results/ti2v-5b/real-wan-c0-b4/B0/frames/target_dense_first_frame.png`
- Target latent reuse video: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/carnator/ti2v-5b/real-wan-c0-b4/B0/videos/target_latent_reuse.mp4`
- Summary JSON: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/carnator-results/ti2v-5b/real-wan-c0-b4/B0/summary.json`

## Reuse

- Sparse calls: `880`
- Dense fallback calls: `0`
- Avg Medit query ratio: `35.128%`
- Avg Msee context ratio: `67.753%`
- Latent injection: `{'enabled': True, 'formula': 'L_new = Medit * L_target + (1 - Medit) * L_source_cache', 'calls': 40, 'total_seconds': 0.1260375170968473, 'avg_ms': 3.1509379274211824, 'mask_steps': {'10': 4, '15': 5, '20': 5, '25': 26}}`
- attention mask variant: `self_contrast_cross_attention_fusion`
- attention mask dir: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/carnator/ti2v-5b/real-wan-c0-b4/history/cache/source_dense/fusion_masks`
- attention mask avg token area: `{'mbase': 0.1690615862607956, 'medit': 0.3512829840183258, 'msee': 0.6775293350219727}`
- attention mask avg latent area: `{'medit': 0.3512829840183258, 'msee': 0.6775293350219727}`

## Environment

- Python: `/data/shared/conda_envs/chai-wan/bin/python`
- Model dir: `/home/tangxuebin/models/Wan2.2-TI2V-5B`
- Wan root: `/home/tangxuebin/project/CARNATOR_ICLR2027/external/Wan2.2`
- CUDA available: `True`
- Flash attention: `{'fa2': True, 'fa3': False}`
- Model layout: `{'exists': True, 'native_config_json': True, 'vae_checkpoint': True, 't5_checkpoint': True, 't5_tokenizer': True}`
