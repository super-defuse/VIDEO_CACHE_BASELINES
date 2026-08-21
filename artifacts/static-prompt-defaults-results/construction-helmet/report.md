# CARNATOR Latent Reuse Report

Updated: `2026-08-20 15:53:23 UTC`
Status: `source_prepared`

## Config

- Backend: `native Wan2.2`
- Model: `Wan2.2-TI2V-5B`
- Task: `T2V`
- Prompts: source `a single yellow construction helmet on a plain wooden workbench, photorealistic, static camera`, target `a single orange construction helmet on a plain wooden workbench, photorealistic, static camera`
- Shape: `1280x704`, frames `121`, fps `24`
- Sampling: `50` steps, `unipc`, solver order `2`, flow shift `5.0`
- Guidance: `5.0` (CFG), region-update layers `10-20`, outer `dense` (n/a) on `0-9,21-29`, FFN `fp`, reuse start step `10`
- latent injection: `hard` (alpha `smoothstep`)
- attention mask: `fusion`, completed probe step `5`, self layer `15`

## Results

| Run | Self-attn | Cross-attn | FFN | Outer sparse density | Outer mask mass coverage | Latency denoising | Speed Up denoising | Latency End2End | Speed Up End2End | Peak reserved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Source dense | 177.820 s | 16.522 s | 34.807 s | n/a | n/a | 274.435 s | n/a | 296.791 s | n/a | 52.93 GiB |
| C0 full dense target | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| Target latent reuse | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

## Artifacts

- Source dense video: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/static-prompt-defaults/construction-helmet/videos/source_dense.mp4`
- Source first frame: `None`
- Target dense video: `None`
- Target first frame: `None`
- Target latent reuse video: `None`
- Summary JSON: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/static-prompt-defaults-results/construction-helmet/summary.json`

## Reuse

- Sparse calls: `None`
- Dense fallback calls: `None`
- Avg Medit query ratio: `n/a`
- Avg Msee context ratio: `n/a`
- Latent injection: `None`
- attention mask variant: `self_contrast_cross_attention_fusion`
- attention mask dir: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/static-prompt-defaults/construction-helmet/cache/source_dense/fusion_masks`
- attention mask token area: `{'mbase': 0.37177419662475586, 'medit': 0.6922653913497925, 'msee': 0.9020161032676697, 'original_mbase': 0.557331383228302}`
- attention mask latent area: `{'mbase': 0.37177419662475586, 'medit': 0.6922653913497925, 'msee': 0.9020161032676697, 'mband': 0.3204911947250366}`

## Environment

- Python: `/data/shared/conda_envs/wan/bin/python`
- Model dir: `/home/tangxuebin/models/Wan2.2-TI2V-5B`
- Wan root: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/external/wan2.2`
- CUDA available: `True`
- Flash attention: `{'fa2': True, 'fa3': False}`
- Model layout: `{'exists': True, 'native_config_json': True, 'vae_checkpoint': True, 't5_checkpoint': True, 't5_tokenizer': True}`
