# Chorus Wan2.2 TI2V-5B

This is the current route-one Chorus reproduction for the fixed dog -> cat
protocol, seed `20260218`, 121 frames at `1280x704`, and 50 UniPC steps.

- Source: 50 steps, `295.77 s`; target: 40 executed steps, `179.06 s` reported.
- Stages: `K1=10`, `K2=40`; Stage 2 uses SRD, all 30 TGAA blocks, and source
  latent fusion; Stage 3 is dense with no TGAA, SRD, or latent reuse.
- Stage 2 keeps the native Wan head dense, so non-edit tokens produce a full
  scheduler output. At Stage 3, UniPC derivative history is cleared at step 40
  while the global timestep index is preserved.
- Mask: Grounded-SAM-2 v1.0; `Medit=53.24%`, `Msee=84.06%` of the token grid.
- TGAA: 1,800 calls, using the documented step-only decay approximation.
- Quality: target `CLIP-T=0.3491`, `VBench-Q=0.6218`.

Files:

- `videos/0000_source-000.mp4`
- `videos/0001_target-000.mp4`
- `results.jsonl`
- `summary.json`
- `vbench/custom_vidprom_eval_results.json`
