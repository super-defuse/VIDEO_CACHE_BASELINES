# Chorus Mbase-only failure case

This is an isolated experiment. The source dense capture, target dense control, Chorus target, CLIP score, logs, and cache are all below this directory.

## Prompts

- Source/cache: `a red apple rolling down a snow mountain, camera focus on the apple`
- Target: `a yellow peach rolling down a mountain, camera focus on the peach`
- CLIP cosine: `0.627954887281304` (threshold `0.65`)

- Cache policy: forced source cache reuse; the CLIP threshold is intentionally waived.

## Expected comparison

The target dense control should contain a yellow peach on a mountain. The target runs all 50 steps: it updates only Mbase in steps 0-29, then runs dense in steps 30-49. Stage-2 source latent features remain outside Mbase, so a snow background is the expected failure signal. The visual mismatch must be confirmed from the saved videos.

## Integrity checks

- Source status: `ok`; target dense status: `ok`; Chorus status: `ok`
- Chorus cache hit: `True`; source UUID: `source-apple-snow`
- Stage-2 update mask reported by the adapter: `Mbase`
- Prompt-diff backend: `manual_override`; SAM phrase: `red apple`
- Mask fractions (base/edit/see): `0.0428` / `0.2287` / `0.4111`
- Physical GPU allocation: `{"chorus_mbase_only": 0, "clip_score": 0, "source_dense": 0, "target_dense": 0}`

## Outputs

- `source_dense/videos/`: source dense video
- `target_dense/videos/`: target dense control
- `chorus_mbase_only/videos/`: target with Mbase-only Chorus reuse
- `cache/`: experiment-local Chorus trajectory and source video
