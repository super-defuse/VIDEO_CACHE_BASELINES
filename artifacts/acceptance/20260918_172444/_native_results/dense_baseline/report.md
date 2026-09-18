# CARNATOR Latent Reuse Report

Updated: `2026-09-18 09:31:02 UTC`
Status: `completed`

## Config

- Backend: `native Wan2.2`
- Model: `Wan2.2-TI2V-5B`
- Task: `T2V`
- Prompts: source `a dog is sleeping on a grassland, focus on the dog`, target `a cat is sleeping on a grassland, focus on the cat`
- Shape: `1280x704`, frames `121`, fps `24`
- Sampling: `50` steps, `unipc`, solver order `2`, flow shift `5.0`
- Guidance: `5.0` (CFG), region-update layers `10-20` with middle attention-mask `True`, outer `dense` (n/a) on `0-9,21-29`, FFN `fp`, reuse start step `10`
- latent injection: `hard` (alpha `smoothstep`)
- attention mask: `fusion`, completed probe step `5`, self layer `15`

## Results

| Run | Self-attn | Cross-attn | FFN | Outer sparse density | Outer mask mass coverage | Latency denoising | Speed Up denoising | Latency End2End | Speed Up End2End | Peak reserved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Source dense | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| C0 full dense target | 178.386 s | 16.643 s | 35.530 s | 100.000% | 100.000% | 275.458 s | 1.00x | 308.112 s | 1.00x | 52.93 GiB |
| Target latent reuse | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

## Artifacts

- Source dense video: `None`
- Source first frame: `None`
- Target dense video: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/20260918_172444/dense_baseline/videos/target_dense.mp4`
- Target first frame: `None`
- Target latent reuse video: `None`
- Summary JSON: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/20260918_172444/_native_results/dense_baseline/summary.json`

## Reuse

- Sparse calls: `None`
- Dense fallback calls: `None`
- Avg Medit query ratio: `n/a`
- Avg Msee context ratio: `n/a`
- Middle attention-mask: `None`; masked Medit query ratio `n/a`; masked Msee context ratio `n/a`
- Latent injection: `None`
- Source mask prepare: `None`
- attention mask variant: `None`
- attention mask dir: `None`
- attention mask token area: `None`
- attention mask latent area: `None`

## Environment

- Python: `/data/shared/conda_envs/wan/bin/python`
- Model dir: `/home/tangxuebin/models/Wan2.2-TI2V-5B`
- Wan root: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/external/wan2.2`
- CUDA available: `True`
- Flash attention: `{'fa2': True, 'fa3': False}`
- Model layout: `{'exists': True, 'native_config_json': True, 'vae_checkpoint': True, 't5_checkpoint': True, 't5_tokenizer': True}`
