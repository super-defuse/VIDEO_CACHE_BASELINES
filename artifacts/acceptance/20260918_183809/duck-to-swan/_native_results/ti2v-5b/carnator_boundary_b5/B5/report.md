# CARNATOR Latent Reuse Report

Updated: `2026-09-18 11:21:37 UTC`
Status: `completed`

## Config

- Backend: `native Wan2.2`
- Model: `Wan2.2-TI2V-5B`
- Task: `T2V`
- Prompts: source `a duck is swimming on a calm lake, focus on the duck`, target `a swan is swimming on a calm lake, focus on the swan`
- Shape: `1280x704`, frames `121`, fps `24`
- Sampling: `50` steps, `unipc`, solver order `2`, flow shift `5.0`
- Guidance: `5.0` (CFG), region-update layers `10-20` with middle attention-mask `True`, outer `sparge-sage2` (n/a) on `0-9,21-29`, FFN `w8a8`, reuse start step `10`
- latent injection: `boundary-blend` (alpha `smoothstep`)
- attention mask: `fusion`, completed probe step `5`, self layer `15`

## Results

| Run | Self-attn | Cross-attn | FFN | Outer sparse density | Outer mask mass coverage | Latency denoising | Speed Up denoising | Latency End2End | Speed Up End2End | Peak reserved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Source dense | 320.623 s | 16.162 s | 34.805 s | n/a | n/a | 416.076 s | n/a | 439.038 s | n/a | 52.93 GiB |
| C0 full dense target | 177.766 s | 16.586 s | 35.169 s | 100.000% | 100.000% | 274.264 s | 1.00x | 302.935 s | 1.00x | 52.93 GiB |
| Target latent reuse | 40.144 s | 12.710 s | 24.068 s | 29.934% | 96.064% | 164.082 s | 1.67x | 189.188 s | 1.60x | 55.87 GiB |

## Artifacts

- Source dense video: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/20260918_183809/duck-to-swan/_sources/carnator_boundary_b5/videos/source_dense.mp4`
- Source first frame: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/20260918_183809/duck-to-swan/_native_results/ti2v-5b/carnator_boundary_b5/B5/frames/source_dense_first_frame.png`
- Target dense video: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/20260918_183809/duck-to-swan/dense_baseline/videos/target_dense.mp4`
- Target first frame: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/20260918_183809/duck-to-swan/_native_results/ti2v-5b/carnator_boundary_b5/B5/frames/target_dense_first_frame.png`
- Target latent reuse video: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/20260918_183809/duck-to-swan/ti2v-5b/carnator_boundary_b5/B5/videos/target_latent_reuse.mp4`
- Summary JSON: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/20260918_183809/duck-to-swan/_native_results/ti2v-5b/carnator_boundary_b5/B5/summary.json`

## Reuse

- Sparse calls: `880`
- Dense fallback calls: `0`
- Avg Medit query ratio: `95.407%`
- Avg Msee context ratio: `99.982%`
- Middle attention-mask: `True`; masked Medit query ratio `95.407%`; masked Msee context ratio `99.982%`
- Latent injection: `{'enabled': True, 'mode': 'boundary-blend', 'alpha_function': 'smoothstep', 'formula': 'L_new = Mbase * L_target + Mband * ((1 - alpha) * L_target + alpha * L_source) + (1 - Medit) * L_source', 'calls': 40, 'total_seconds': 0.18994311522692442, 'avg_ms': 4.7485778806731105, 'mask_source_completed_step': 5, 'mask_statistics': {'base_fraction': 0.6447947025299072, 'band_fraction': 0.30927419662475586, 'edit_fraction': 0.9540688991546631, 'alpha_mean': 0.23367200791835785, 'alpha_min': 0.0, 'alpha_max': 1.0}}`
- Source mask prepare: `{'entries': 2400, 'decoded_entries': 2400, 'file_bytes': 5267354400, 'device': 'cuda:0', 'elapsed_seconds': 1.842820652294904}`
- attention mask variant: `self_contrast_cross_attention_fusion`
- attention mask dir: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/20260918_183809/duck-to-swan/_sources/carnator_boundary_b5/cache/source_dense/fusion_masks`
- attention mask token area: `{'mbase': 0.6447947025299072, 'medit': 0.9540688991546631, 'msee': 0.9998167157173157, 'original_mbase': 0.7865102887153625}`
- attention mask latent area: `{'mbase': 0.6447947025299072, 'medit': 0.9540688991546631, 'msee': 0.9998167157173157, 'mband': 0.30927419662475586}`

## Environment

- Python: `/data/shared/conda_envs/wan/bin/python`
- Model dir: `/home/tangxuebin/models/Wan2.2-TI2V-5B`
- Wan root: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/external/wan2.2`
- CUDA available: `True`
- Flash attention: `{'fa2': True, 'fa3': False}`
- Model layout: `{'exists': True, 'native_config_json': True, 'vae_checkpoint': True, 't5_checkpoint': True, 't5_tokenizer': True}`
