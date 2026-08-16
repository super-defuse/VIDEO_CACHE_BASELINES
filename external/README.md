# External Sources

`external/` contains only third-party source trees. Baseline logic is kept in
`adapters/`, `methods/` and `carnator/`. Initialize the locked submodules
after cloning:

```bash
git submodule update --init --recursive
```

The required revisions and upstream URLs are recorded in `lock.json`. Model
weights remain outside this directory and are configured with `model_path`.

| Directory | Purpose |
| --- | --- |
| `wan2.2` | Wan2.2 model runtime |
| `cogvideo` | CogVideo model runtime |
| `lightx2v` | LightX2V distilled-Wan runtime |
| `spargeattn` | SpargeAttn sparse-attention backend |
| `grounded-sam2` | Grounded-SAM-2 mask backend |
| `vbench` | Optional VBench checkout, installed separately when quality evaluation is needed |

VBench is not a Git submodule because the accepted evaluator was originally a
snapshot in the CHAI source tree. Clone the revision recorded in `lock.json`
into `external/vbench` before running a real quality evaluation. It remains
ignored by the parent repository so its large model and cache files cannot be
committed accidentally.
