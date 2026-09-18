# CARNATOR Latent Reuse Report

Updated: `2026-09-15 12:46:03 UTC`
Status: `completed`

## Config

- Backend: `native Wan2.2`
- Model: `Wan2.2-TI2V-5B`
- Task: `T2V`
- Prompts: source `a red bus driving through a city`, target `a red car driving through a city`
- Shape: `1280x704`, frames `121`, fps `24`
- Sampling: `50` steps, `unipc`, solver order `2`, flow shift `5.0`
- Guidance: `5.0` (CFG), region-update layers `10-20` with middle attention-mask `True`, outer `sparge-sage2` (n/a) on `0-9,21-29`, FFN `w8a8`, reuse start step `10`
- latent injection: `hard` (alpha `smoothstep`)
- attention mask: `fusion`, completed probe step `5`, self layer `15`

## Results

| Run | Self-attn | Cross-attn | FFN | Outer sparse density | Outer mask mass coverage | Latency denoising | Speed Up denoising | Latency End2End | Speed Up End2End | Peak reserved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Source dense | 321.576 s | 16.105 s | 35.010 s | n/a | n/a | 417.218 s | n/a | 440.144 s | n/a | 52.93 GiB |
| C0 full dense target | n/a | n/a | n/a | 100.000% | 100.000% | n/a | 1.00x | 279.255 s | 1.00x | n/a |
| Target latent reuse | 40.578 s | 11.178 s | 21.368 s | 31.721% | 96.018% | 142.646 s | n/a | 170.583 s | 1.64x | 55.87 GiB |

## Artifacts

- Source dense video: `/home/tangxuebin/project/CHAI/artifacts/runs/pair1-wan-carnator/carnator/artifacts/source/videos/source_dense.mp4`
- Source first frame: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/pair1_carnator_ablation_20260915/e2_hard_injection/results/run/frames/source_dense_first_frame.png`
- Target dense video: `/home/tangxuebin/project/CHAI/artifacts/runs/pair1-wan-carnator/carnator/target_dense.mp4`
- Target first frame: `None`
- Target latent reuse video: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/pair1_carnator_ablation_20260915/e2_hard_injection/artifacts/run/videos/target_latent_reuse.mp4`
- Summary JSON: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/pair1_carnator_ablation_20260915/e2_hard_injection/results/run/summary.json`

## Reuse

- Sparse calls: `880`
- Dense fallback calls: `0`
- Avg Medit query ratio: `62.852%`
- Avg Msee context ratio: `86.499%`
- Middle attention-mask: `True`; masked Medit query ratio `62.852%`; masked Msee context ratio `86.499%`
- Latent injection: `{'enabled': True, 'mode': 'hard', 'alpha_function': None, 'formula': 'L_new = Medit * L_target + (1 - Medit) * L_source_cache', 'calls': 40, 'total_seconds': 0.14723499957472086, 'avg_ms': 3.6808749893680215, 'mask_source_completed_step': 5, 'mask_statistics': {'mode': 'hard', 'base_fraction': 0.31909823417663574, 'band_fraction': 0.0, 'edit_fraction': 0.6285190582275391}}`
- Source mask prepare: `{'entries': 2400, 'decoded_entries': 2400, 'file_bytes': 5267354400, 'device': 'cuda:0', 'elapsed_seconds': 5.528174943756312}`
- attention mask variant: `self_contrast_cross_attention_fusion`
- attention mask dir: `/home/tangxuebin/project/CHAI/artifacts/runs/pair1-wan-carnator/carnator/artifacts/source/cache/source_dense/fusion_masks`
- attention mask token area: `{'mbase': 0.31909823417663574, 'medit': 0.6285190582275391, 'msee': 0.8649926781654358, 'original_mbase': 0.5960777401924133}`
- attention mask latent area: `{'mbase': 0.31909823417663574, 'medit': 0.6285190582275391, 'msee': 0.8649926781654358, 'mband': 0.3094208240509033}`

## Environment

- Python: `/data/shared/conda_envs/wan/bin/python`
- Model dir: `/home/tangxuebin/models/Wan2.2-TI2V-5B`
- Wan root: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/external/wan2.2`
- CUDA available: `True`
- Flash attention: `{'fa2': True, 'fa3': False}`
- Model layout: `{'exists': True, 'native_config_json': True, 'vae_checkpoint': True, 't5_checkpoint': True, 't5_tokenizer': True}`
