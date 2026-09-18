# CARNATOR Latent Reuse Report

Updated: `2026-09-15 12:52:01 UTC`
Status: `completed`

## Config

- Backend: `native Wan2.2`
- Model: `Wan2.2-TI2V-5B`
- Task: `T2V`
- Prompts: source `a red bus driving through a city`, target `a red car driving through a city`
- Shape: `1280x704`, frames `121`, fps `24`
- Sampling: `50` steps, `unipc`, solver order `2`, flow shift `5.0`
- Guidance: `5.0` (CFG), region-update layers `10-20` with middle attention-mask `True`, outer `sparge-sage2` (n/a) on `0-9,21-29`, FFN `w8a8`, reuse start step `10`
- latent injection: `boundary-blend` (alpha `smoothstep`)
- attention mask: `fusion`, completed probe step `5`, self layer `15`

## Results

| Run | Self-attn | Cross-attn | FFN | Outer sparse density | Outer mask mass coverage | Latency denoising | Speed Up denoising | Latency End2End | Speed Up End2End | Peak reserved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Source dense | 321.576 s | 16.105 s | 35.010 s | n/a | n/a | 417.218 s | n/a | 440.144 s | n/a | 52.93 GiB |
| C0 full dense target | n/a | n/a | n/a | 100.000% | 100.000% | n/a | 1.00x | 279.255 s | 1.00x | n/a |
| Target latent reuse | 39.964 s | 11.901 s | 22.219 s | 31.721% | 96.018% | 151.935 s | n/a | 176.530 s | 1.58x | 55.87 GiB |

## Artifacts

- Source dense video: `/home/tangxuebin/project/CHAI/artifacts/runs/pair1-wan-carnator/carnator/artifacts/source/videos/source_dense.mp4`
- Source first frame: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/pair1_carnator_ablation_20260915/e3_self_contrast_mask/results/run/frames/source_dense_first_frame.png`
- Target dense video: `/home/tangxuebin/project/CHAI/artifacts/runs/pair1-wan-carnator/carnator/target_dense.mp4`
- Target first frame: `None`
- Target latent reuse video: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/pair1_carnator_ablation_20260915/e3_self_contrast_mask/artifacts/run/videos/target_latent_reuse.mp4`
- Summary JSON: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/pair1_carnator_ablation_20260915/e3_self_contrast_mask/results/run/summary.json`

## Reuse

- Sparse calls: `880`
- Dense fallback calls: `0`
- Avg Medit query ratio: `80.425%`
- Avg Msee context ratio: `95.942%`
- Middle attention-mask: `True`; masked Medit query ratio `80.425%`; masked Msee context ratio `95.942%`
- Latent injection: `{'enabled': True, 'mode': 'boundary-blend', 'alpha_function': 'smoothstep', 'formula': 'L_new = Mbase * L_target + Mband * ((1 - alpha) * L_target + alpha * L_source) + (1 - Medit) * L_source', 'calls': 40, 'total_seconds': 0.1859335470944643, 'avg_ms': 4.6483386773616076, 'mask_source_completed_step': 5, 'mask_statistics': {'base_fraction': 0.5960777401924133, 'band_fraction': 0.20817449688911438, 'edit_fraction': 0.8042522072792053, 'alpha_mean': 0.3632127642631531, 'alpha_min': 0.0, 'alpha_max': 1.0}}`
- Source mask prepare: `{'entries': 2400, 'decoded_entries': 2400, 'file_bytes': 5267354400, 'device': 'cuda:0', 'elapsed_seconds': 2.011585800908506}`
- attention mask variant: `self_contrast_3d_per_token_time`
- attention mask dir: `/home/tangxuebin/project/CHAI/artifacts/runs/pair1-wan-carnator/carnator/artifacts/source/cache/source_dense/self_contrast_masks`
- attention mask token area: `{'mbase': 0.5960777401924133, 'medit': 0.8042522072792053, 'msee': 0.9594208002090454}`
- attention mask latent area: `{'mbase': 0.5960777401924133, 'medit': 0.8042522072792053, 'msee': 0.9594208002090454, 'mband': 0.20817448198795319}`

## Environment

- Python: `/data/shared/conda_envs/wan/bin/python`
- Model dir: `/home/tangxuebin/models/Wan2.2-TI2V-5B`
- Wan root: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/external/wan2.2`
- CUDA available: `True`
- Flash attention: `{'fa2': True, 'fa3': False}`
- Model layout: `{'exists': True, 'native_config_json': True, 'vae_checkpoint': True, 't5_checkpoint': True, 't5_tokenizer': True}`
