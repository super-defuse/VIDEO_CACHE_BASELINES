# Chorus counterexample P1 — Pair A (validity)

Forced cache reuse: the target pins `cache_prompt` to this pair's source, so the reuse
decision never consults Chorus's CLIP similarity gate. The figure must be labelled
`forced cache reuse` and must report the measured CLIP cosine below.

## Prompts

- Source/cache: `a red apple rolling down a snow mountain, camera focus on the apple`
- Target: `a red apple rolling down a mountain, camera focus on the apple`
- Expected SAM phrase: `red apple`

## Audit fields (not pass/fail)

- CLIP cosine: `0.9148925164950095` (threshold `0.65`, passes `True`)
- Prompt-diff backend: `manual_override` (requested `manual`)
- Update mask: `Medit`
- Mask fractions (base/edit/see): `0.0428` / `0.0779` / `0.4111`
- Mask halo: `mask_dilation_edit=2` (see `14`), overridden from the Chorus default `8`
- Seed: `20260218`; Chorus stages: `{'k1': 5, 'k2': 30}`
- Physical GPU: `3`

## Validity checks

- `forced_cache_hit`: `True`
- `source_uuid_matches`: `True`
- `update_mask_is_medit`: `True`
- `sam_phrase_matches`: `True`
- `mask_fractions_recorded`: `True`
- `mask_dilation_matches`: `True`

The remaining validity criterion is visual: the target video must show the intended
failure (Pair A keeps the snow background; Pair B shows a shape/texture mismatch).
Confirm it by inspecting `frames/` before using this pair in the figure.

## Outputs

- `source_dense/videos/`: source dense video
- `chorus_medit/videos/`: forced-reuse Chorus target video
- `target_dense/videos/`: dense control (audit only, not in the figure)
- `frames/`: 5 extracted frames per video
- `cache/`: pair-local Chorus trajectory
