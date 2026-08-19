# Chorus Wan2.2 TI2V-5B: no-TGAA ablation

This is the fixed dog -> cat Chorus ablation with only Stage-2 TGAA disabled.
The source trajectory, Grounded-SAM-2 mask, SRD, latent fusion, scheduler reset,
and Stage-3 dense refinement are unchanged.

- Source: 50 steps, `296.47 s`; source video hash matches the Chorus source.
- Target: 40 steps, `179.36 s` reported; source-prompt cache hit.
- Mask: `phrase=dog`, `Medit=53.24%`, `Msee=84.06%`.
- TGAA: `enabled=false`, `calls=0`.
- Quality: target `CLIP-T=0.3491`, `VBench-Q=0.6332`.

Files:

- `videos/0000_source-000.mp4`
- `videos/0001_target-000.mp4`
- `results.jsonl`
- `summary.json`
- `vbench/custom_vidprom_eval_results.json`
