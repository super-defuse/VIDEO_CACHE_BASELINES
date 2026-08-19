# Chorus Wan2.2 TI2V-5B: TGAA key-only ablation

Fixed dog -> cat Chorus reproduction with `gamma_k=1.15` and `gamma_o=1.0`.
All other settings remain unchanged: 50-step source, `K1=10/K2=40`, Stage-2
SRD/latent fusion, all 30 TGAA blocks, and Stage-3 dense/no-reuse refinement.

- Source: 50 steps, `296.51 s`; source hash matches the other Chorus variants.
- Target: 40 steps, `179.88 s` reported; source-prompt cache hit.
- Mask: `phrase=dog`, `Medit=53.24%`, `Msee=84.06%`.
- TGAA: `gamma_k=1.15`, `gamma_o=1.0`, 1,800 calls.
- Quality: target `CLIP-T=0.3159`, `VBench-Q=0.6189`.

Files:

- `videos/0000_source-000.mp4`
- `videos/0001_target-000.mp4`
- `results.jsonl`
- `vbench/custom_vidprom_eval_results.json`
