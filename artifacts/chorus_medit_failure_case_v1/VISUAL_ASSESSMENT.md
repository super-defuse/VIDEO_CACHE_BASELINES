# Medit Visual Assessment

This qualitative Medit ablation uses the same prompts, seed, forced source-cache
reuse, 50-step schedule, and manual prompt diff as the Mbase v5 experiment.
Only the Stage-2 update mask changes: steps 0-29 update `Medit`; steps 30-49
use dense inference. The run completed on physical gpu0 only.

## Integrity

- Chorus method: `chorus`
- Cache hit: `true`, source UUID `source-apple-snow`
- Reported update mask: `Medit`
- Stage-2 token fraction: `0.2287` (`Mbase` is `0.0428`)
- Stage-3: dense steps 30-49

## Visual Result

`target_dense` produces the expected yellow peach on a green mountain. The
Medit result retains the source snow mountain across the global frame, which
preserves the intended background-mismatch failure. Expanding the update area
does not make a coherent replacement: sampled frames show a large rectangular
green-mountain patch within the snowy scene, while the yellow peach is absent
or displaced to the frame edge late in the video.

The Mbase comparison is visually more localized: it retains a peach-like
foreground but leaves apple residuals. Medit removes neither the snow
background mismatch nor the local discontinuity; it expands the discontinuity
to the larger `Medit` region.

## Frames

- `frames/compare_01.png`, `compare_02.png`, `compare_03.png`: source / target
  dense / Medit at corresponding start, middle, and late frames.
- `frames/mbase_vs_medit_01.png`, `mbase_vs_medit_02.png`,
  `mbase_vs_medit_03.png`: Mbase v5 on the left and Medit on the right.
