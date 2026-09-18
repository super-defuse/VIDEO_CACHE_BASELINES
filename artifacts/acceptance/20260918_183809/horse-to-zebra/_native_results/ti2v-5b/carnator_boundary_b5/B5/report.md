# CARNATOR Latent Reuse Report

Updated: `2026-09-18 10:51:02 UTC`
Status: `completed`

## Config

- Backend: `native Wan2.2`
- Model: `Wan2.2-TI2V-5B`
- Task: `T2V`
- Prompts: source `a horse is galloping across a dusty plain, focus on the horse`, target `a zebra is galloping across a dusty plain, focus on the zebra`
- Shape: `1280x704`, frames `121`, fps `24`
- Sampling: `50` steps, `unipc`, solver order `2`, flow shift `5.0`
- Guidance: `5.0` (CFG), region-update layers `10-20` with middle attention-mask `True`, outer `sparge-sage2` (n/a) on `0-9,21-29`, FFN `w8a8`, reuse start step `10`
- latent injection: `boundary-blend` (alpha `smoothstep`)
- attention mask: `fusion`, completed probe step `5`, self layer `15`

## Results

| Run | Self-attn | Cross-attn | FFN | Outer sparse density | Outer mask mass coverage | Latency denoising | Speed Up denoising | Latency End2End | Speed Up End2End | Peak reserved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Source dense | 322.748 s | 16.313 s | 35.074 s | n/a | n/a | 418.769 s | n/a | 450.785 s | n/a | 52.93 GiB |
| C0 full dense target | 178.537 s | 16.680 s | 35.516 s | 100.000% | 100.000% | 275.606 s | 1.00x | 304.555 s | 1.00x | 52.93 GiB |
| Target latent reuse | 40.221 s | 10.969 s | 20.665 s | 29.912% | 96.178% | 134.774 s | 2.04x | 164.930 s | 1.85x | 55.67 GiB |

## Artifacts

- Source dense video: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/20260918_183809/horse-to-zebra/_sources/carnator_boundary_b5/videos/source_dense.mp4`
- Source first frame: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/20260918_183809/horse-to-zebra/_native_results/ti2v-5b/carnator_boundary_b5/B5/frames/source_dense_first_frame.png`
- Target dense video: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/20260918_183809/horse-to-zebra/dense_baseline/videos/target_dense.mp4`
- Target first frame: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/20260918_183809/horse-to-zebra/_native_results/ti2v-5b/carnator_boundary_b5/B5/frames/target_dense_first_frame.png`
- Target latent reuse video: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/20260918_183809/horse-to-zebra/ti2v-5b/carnator_boundary_b5/B5/videos/target_latent_reuse.mp4`
- Summary JSON: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/20260918_183809/horse-to-zebra/_native_results/ti2v-5b/carnator_boundary_b5/B5/summary.json`

## Reuse

- Sparse calls: `880`
- Dense fallback calls: `0`
- Avg Medit query ratio: `55.367%`
- Avg Msee context ratio: `74.340%`
- Middle attention-mask: `True`; masked Medit query ratio `55.367%`; masked Msee context ratio `74.340%`
- Latent injection: `{'enabled': True, 'mode': 'boundary-blend', 'alpha_function': 'smoothstep', 'formula': 'L_new = Mbase * L_target + Mband * ((1 - alpha) * L_target + alpha * L_source) + (1 - Medit) * L_source', 'calls': 40, 'total_seconds': 0.21795516647398472, 'avg_ms': 5.448879161849618, 'mask_source_completed_step': 5, 'mask_statistics': {'base_fraction': 0.37587976455688477, 'band_fraction': 0.1777859330177307, 'edit_fraction': 0.5536656975746155, 'alpha_mean': 0.408356249332428, 'alpha_min': 0.0, 'alpha_max': 1.0}}`
- Source mask prepare: `{'entries': 2400, 'decoded_entries': 2400, 'file_bytes': 5267354400, 'device': 'cuda:0', 'elapsed_seconds': 2.6885737190023065}`
- attention mask variant: `self_contrast_cross_attention_fusion`
- attention mask dir: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/20260918_183809/horse-to-zebra/_sources/carnator_boundary_b5/cache/source_dense/fusion_masks`
- attention mask token area: `{'mbase': 0.37587976455688477, 'medit': 0.5536656975746155, 'msee': 0.7434017658233643, 'original_mbase': 0.49277859926223755}`
- attention mask latent area: `{'mbase': 0.37587976455688477, 'medit': 0.5536656975746155, 'msee': 0.7434017658233643, 'mband': 0.17778591811656952}`

## Environment

- Python: `/data/shared/conda_envs/wan/bin/python`
- Model dir: `/home/tangxuebin/models/Wan2.2-TI2V-5B`
- Wan root: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/external/wan2.2`
- CUDA available: `True`
- Flash attention: `{'fa2': True, 'fa3': False}`
- Model layout: `{'exists': True, 'native_config_json': True, 'vae_checkpoint': True, 't5_checkpoint': True, 't5_tokenizer': True}`
