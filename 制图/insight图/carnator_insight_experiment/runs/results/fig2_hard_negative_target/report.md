# CARNATOR Latent Reuse Report

Updated: `2026-08-29 07:28:48 UTC`
Status: `source_prepared`

## Config

- Backend: `native Wan2.2`
- Model: `Wan2.2-TI2V-5B`
- Task: `T2V`
- Prompts: source `a dog is running through a snowy forest, focus on the dog`, target `a dog is running through a snowy forest, focus on the dog`
- Shape: `1280x704`, frames `121`, fps `24`
- Sampling: `50` steps, `unipc`, solver order `2`, flow shift `5.0`
- Guidance: `5.0` (CFG), region-update layers `` with middle attention-mask `False`, outer `dense` (n/a) on `15`, FFN `fp`, reuse start step `10`
- latent injection: `hard` (alpha `smoothstep`)
- attention mask: `fusion`, completed probe step `5`, self layer `15`

## Results

| Run | Self-attn | Cross-attn | FFN | Outer sparse density | Outer mask mass coverage | Latency denoising | Speed Up denoising | Latency End2End | Speed Up End2End | Peak reserved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Source dense | 177.520 s | 16.525 s | 35.067 s | n/a | n/a | 274.358 s | n/a | 306.752 s | n/a | 52.93 GiB |
| C0 full dense target | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| Target latent reuse | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

## Artifacts

- Source dense video: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/制图/insight图/carnator_insight_experiment/runs/artifacts/fig2_hard_negative_target/videos/source_dense.mp4`
- Source first frame: `None`
- Target dense video: `None`
- Target first frame: `None`
- Target latent reuse video: `None`
- Summary JSON: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/制图/insight图/carnator_insight_experiment/runs/results/fig2_hard_negative_target/summary.json`

## Reuse

- Sparse calls: `None`
- Dense fallback calls: `None`
- Avg Medit query ratio: `n/a`
- Avg Msee context ratio: `n/a`
- Middle attention-mask: `None`; masked Medit query ratio `n/a`; masked Msee context ratio `n/a`
- Latent injection: `None`
- attention mask variant: `self_contrast_cross_attention_fusion`
- attention mask dir: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/制图/insight图/carnator_insight_experiment/runs/artifacts/fig2_hard_negative_target/cache/source_dense/fusion_masks`
- attention mask token area: `{'mbase': 0.3245234489440918, 'medit': 0.6895894408226013, 'msee': 0.8962609767913818, 'original_mbase': 0.5387830138206482}`
- attention mask latent area: `{'mbase': 0.3245234489440918, 'medit': 0.6895894408226013, 'msee': 0.8962609767913818, 'mband': 0.3650659918785095}`

## Environment

- Python: `/data/shared/conda_envs/wan/bin/python`
- Model dir: `/home/tangxuebin/models/Wan2.2-TI2V-5B`
- Wan root: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/external/wan2.2`
- CUDA available: `True`
- Flash attention: `{'fa2': True, 'fa3': False}`
- Model layout: `{'exists': True, 'native_config_json': True, 'vae_checkpoint': True, 't5_checkpoint': True, 't5_tokenizer': True}`
