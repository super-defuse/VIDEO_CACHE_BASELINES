# CARNATOR Latent Reuse Report

Updated: `2026-09-19 05:14:51 UTC`
Status: `completed`

## Config

- Backend: `native Wan2.2`
- Model: `Wan2.2-TI2V-5B`
- Task: `T2V`
- Prompts: source `an airplane is flying through a clear blue sky, focus on the airplane`, target `a helicopter is flying through a clear blue sky, focus on the helicopter`
- Shape: `1280x704`, frames `121`, fps `24`
- Sampling: `50` steps, `unipc`, solver order `2`, flow shift `5.0`
- Guidance: `5.0` (CFG), region-update layers `10-20` with middle attention-mask `True`, outer `sparge-sage2` (n/a) on `0-9,21-29`, FFN `w8a8`, reuse start step `10`
- latent injection: `boundary-blend` (alpha `smoothstep`)
- attention mask: `fusion`, completed probe step `5`, self layer `15`

## Results

| Run | Self-attn | Cross-attn | FFN | Outer sparse density | Outer mask mass coverage | Latency denoising | Speed Up denoising | Latency End2End | Speed Up End2End | Peak reserved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Source dense | 322.010 s | 16.201 s | 34.360 s | n/a | n/a | 417.260 s | n/a | 446.356 s | n/a | 52.93 GiB |
| C0 full dense target | 177.681 s | 16.575 s | 35.299 s | 100.000% | 100.000% | 274.286 s | 1.00x | 304.302 s | 1.00x | 52.93 GiB |
| Target latent reuse | 42.186 s | 10.281 s | 19.801 s | 36.983% | 95.996% | 129.551 s | 2.12x | 148.576 s | 2.05x | 60.11 GiB |

## Artifacts

- Source dense video: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/20260919_114625/airplane-to-helicopter/_sources/carnator_boundary_b5_targetresume/videos/source_dense.mp4`
- Source first frame: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/20260919_114625/airplane-to-helicopter/_native_results/ti2v-5b/carnator_boundary_b5_targetresume/B5/frames/source_dense_first_frame.png`
- Target dense video: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/20260919_114625/airplane-to-helicopter/dense_baseline/videos/target_dense.mp4`
- Target first frame: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/20260919_114625/airplane-to-helicopter/_native_results/ti2v-5b/carnator_boundary_b5_targetresume/B5/frames/target_dense_first_frame.png`
- Target latent reuse video: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/20260919_114625/airplane-to-helicopter/ti2v-5b/carnator_boundary_b5_targetresume/B5/videos/target_latent_reuse.mp4`
- Summary JSON: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/20260919_114625/airplane-to-helicopter/_native_results/ti2v-5b/carnator_boundary_b5_targetresume/B5/summary.json`

## Reuse

- Sparse calls: `880`
- Dense fallback calls: `0`
- Avg Medit query ratio: `42.845%`
- Avg Msee context ratio: `68.394%`
- Middle attention-mask: `True`; masked Medit query ratio `42.845%`; masked Msee context ratio `68.394%`
- Latent injection: `{'enabled': True, 'mode': 'boundary-blend', 'alpha_function': 'smoothstep', 'formula': 'L_new = Mbase * L_target + Mband * ((1 - alpha) * L_target + alpha * L_source) + (1 - Medit) * L_source', 'calls': 40, 'total_seconds': 0.207532346714288, 'avg_ms': 5.1883086678572, 'mask_source_completed_step': 5, 'mask_statistics': {'base_fraction': 0.165139302611351, 'band_fraction': 0.26330646872520447, 'edit_fraction': 0.4284457564353943, 'alpha_mean': 0.4084605574607849, 'alpha_min': 0.0, 'alpha_max': 1.0}}`
- Source mask prepare: `{'entries': 2400, 'decoded_entries': 2400, 'file_bytes': 5267354400, 'device': 'cuda:0', 'elapsed_seconds': 2.2198749873787165}`
- attention mask variant: `self_contrast_cross_attention_fusion`
- attention mask dir: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/20260919_114625/airplane-to-helicopter/_sources/carnator_boundary_b5_targetresume/cache/source_dense/fusion_masks`
- attention mask token area: `{'mbase': 0.165139302611351, 'medit': 0.4284457564353943, 'msee': 0.6839442849159241, 'original_mbase': 0.397140771150589}`
- attention mask latent area: `{'mbase': 0.165139302611351, 'medit': 0.4284457564353943, 'msee': 0.6839442849159241, 'mband': 0.2633064389228821}`

## Environment

- Python: `/data/shared/conda_envs/wan/bin/python`
- Model dir: `/home/tangxuebin/models/Wan2.2-TI2V-5B`
- Wan root: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/external/wan2.2`
- CUDA available: `True`
- Flash attention: `{'fa2': True, 'fa3': False}`
- Model layout: `{'exists': True, 'native_config_json': True, 'vae_checkpoint': True, 't5_checkpoint': True, 't5_tokenizer': True}`
