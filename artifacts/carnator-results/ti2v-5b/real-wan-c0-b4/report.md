# CARNATOR Latent Reuse Report

Updated: `2026-08-15 07:20:07 UTC`
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
| Source dense | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| C0 full dense target | 181.546 s | 16.459 s | 34.977 s | 100.000% | 100.000% | 277.363 s | 1.00x | 300.641 s | 1.00x | 52.93 GiB |
| Target latent reuse | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

## Artifacts

- Source dense video: `None`
- Source first frame: `None`
- Target dense video: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/carnator/ti2v-5b/real-wan-c0-b4/videos/target_dense.mp4`
- Target first frame: `None`
- Target latent reuse video: `None`
- Summary JSON: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/carnator-results/ti2v-5b/real-wan-c0-b4/summary.json`

## Reuse

- Sparse calls: `None`
- Dense fallback calls: `None`
- Avg Medit query ratio: `n/a`
- Avg Msee context ratio: `n/a`
- Latent injection: `None`
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
