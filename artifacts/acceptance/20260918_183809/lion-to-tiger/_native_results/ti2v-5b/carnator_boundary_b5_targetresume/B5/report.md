# CARNATOR Latent Reuse Report

Updated: `2026-09-18 13:10:04 UTC`
Status: `completed`

## Config

- Backend: `native Wan2.2`
- Model: `Wan2.2-TI2V-5B`
- Task: `T2V`
- Prompts: source `a lion is walking through tall grass, focus on the lion`, target `a tiger is walking through tall grass, focus on the tiger`
- Shape: `1280x704`, frames `121`, fps `24`
- Sampling: `50` steps, `unipc`, solver order `2`, flow shift `5.0`
- Guidance: `5.0` (CFG), region-update layers `10-20` with middle attention-mask `True`, outer `sparge-sage2` (n/a) on `0-9,21-29`, FFN `w8a8`, reuse start step `10`
- latent injection: `boundary-blend` (alpha `smoothstep`)
- attention mask: `fusion`, completed probe step `5`, self layer `15`

## Results

| Run | Self-attn | Cross-attn | FFN | Outer sparse density | Outer mask mass coverage | Latency denoising | Speed Up denoising | Latency End2End | Speed Up End2End | Peak reserved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Source dense | 320.761 s | 16.096 s | 34.267 s | n/a | n/a | 415.717 s | n/a | 444.040 s | n/a | 52.93 GiB |
| C0 full dense target | 179.100 s | 16.680 s | 35.474 s | 100.000% | 100.000% | 276.110 s | 1.00x | 302.398 s | 1.00x | 52.93 GiB |
| Target latent reuse | 40.479 s | 11.606 s | 22.229 s | 31.683% | 96.279% | 148.203 s | 1.86x | 167.774 s | 1.80x | 59.67 GiB |

## Artifacts

- Source dense video: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/20260918_183809/lion-to-tiger/_sources/carnator_boundary_b5_targetresume/videos/source_dense.mp4`
- Source first frame: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/20260918_183809/lion-to-tiger/_native_results/ti2v-5b/carnator_boundary_b5_targetresume/B5/frames/source_dense_first_frame.png`
- Target dense video: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/20260918_183809/lion-to-tiger/dense_baseline/videos/target_dense.mp4`
- Target first frame: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/20260918_183809/lion-to-tiger/_native_results/ti2v-5b/carnator_boundary_b5_targetresume/B5/frames/target_dense_first_frame.png`
- Target latent reuse video: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/20260918_183809/lion-to-tiger/ti2v-5b/carnator_boundary_b5_targetresume/B5/videos/target_latent_reuse.mp4`
- Summary JSON: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/20260918_183809/lion-to-tiger/_native_results/ti2v-5b/carnator_boundary_b5_targetresume/B5/summary.json`

## Reuse

- Sparse calls: `880`
- Dense fallback calls: `0`
- Avg Medit query ratio: `71.880%`
- Avg Msee context ratio: `89.359%`
- Middle attention-mask: `True`; masked Medit query ratio `71.880%`; masked Msee context ratio `89.359%`
- Latent injection: `{'enabled': True, 'mode': 'boundary-blend', 'alpha_function': 'smoothstep', 'formula': 'L_new = Mbase * L_target + Mband * ((1 - alpha) * L_target + alpha * L_source) + (1 - Medit) * L_source', 'calls': 40, 'total_seconds': 0.1960069565102458, 'avg_ms': 4.900173912756145, 'mask_source_completed_step': 5, 'mask_statistics': {'base_fraction': 0.5153592228889465, 'band_fraction': 0.20344574749469757, 'edit_fraction': 0.7188050150871277, 'alpha_mean': 0.40993162989616394, 'alpha_min': 0.0, 'alpha_max': 1.0}}`
- Source mask prepare: `{'entries': 2400, 'decoded_entries': 2400, 'file_bytes': 5267354400, 'device': 'cuda:0', 'elapsed_seconds': 2.0657696672715247}`
- attention mask variant: `self_contrast_cross_attention_fusion`
- attention mask dir: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/20260918_183809/lion-to-tiger/_sources/carnator_boundary_b5_targetresume/cache/source_dense/fusion_masks`
- attention mask token area: `{'mbase': 0.5153592228889465, 'medit': 0.7188050150871277, 'msee': 0.8935850262641907, 'original_mbase': 0.6236070394515991}`
- attention mask latent area: `{'mbase': 0.5153592228889465, 'medit': 0.7188050150871277, 'msee': 0.8935850262641907, 'mband': 0.20344574749469757}`

## Environment

- Python: `/data/shared/conda_envs/wan/bin/python`
- Model dir: `/home/tangxuebin/models/Wan2.2-TI2V-5B`
- Wan root: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/external/wan2.2`
- CUDA available: `True`
- Flash attention: `{'fa2': True, 'fa3': False}`
- Model layout: `{'exists': True, 'native_config_json': True, 'vae_checkpoint': True, 't5_checkpoint': True, 't5_tokenizer': True}`
