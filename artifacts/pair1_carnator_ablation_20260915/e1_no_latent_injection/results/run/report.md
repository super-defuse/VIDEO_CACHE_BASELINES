# CARNATOR Latent Reuse Report

Updated: `2026-09-15 12:45:55 UTC`
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
| Target latent reuse | 40.749 s | 11.400 s | 21.175 s | 31.721% | 96.018% | 142.029 s | n/a | 176.450 s | 1.58x | 55.87 GiB |

## Artifacts

- Source dense video: `/home/tangxuebin/project/CHAI/artifacts/runs/pair1-wan-carnator/carnator/artifacts/source/videos/source_dense.mp4`
- Source first frame: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/pair1_carnator_ablation_20260915/e1_no_latent_injection/results/run/frames/source_dense_first_frame.png`
- Target dense video: `/home/tangxuebin/project/CHAI/artifacts/runs/pair1-wan-carnator/carnator/target_dense.mp4`
- Target first frame: `None`
- Target latent reuse video: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/pair1_carnator_ablation_20260915/e1_no_latent_injection/artifacts/run/videos/target_latent_reuse.mp4`
- Summary JSON: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/pair1_carnator_ablation_20260915/e1_no_latent_injection/results/run/summary.json`

## Reuse

- Sparse calls: `880`
- Dense fallback calls: `0`
- Avg Medit query ratio: `62.852%`
- Avg Msee context ratio: `86.499%`
- Middle attention-mask: `True`; masked Medit query ratio `62.852%`; masked Msee context ratio `86.499%`
- Latent injection: `{'enabled': False, 'mode': 'boundary-blend', 'alpha_function': 'smoothstep', 'formula': 'L_new = Mbase * L_target + Mband * ((1 - alpha) * L_target + alpha * L_source) + (1 - Medit) * L_source', 'calls': 0, 'total_seconds': 0.0, 'avg_ms': 0.0, 'mask_source_completed_step': None, 'mask_statistics': {}}`
- Source mask prepare: `{'entries': 2400, 'decoded_entries': 2400, 'file_bytes': 5267354400, 'device': 'cuda:0', 'elapsed_seconds': 2.328215762041509}`
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
