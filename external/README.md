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
| `LTX-Video` | LTXV model runtime (official Lightricks source) |
| `spargeattn` | SpargeAttn sparse-attention backend |
| `grounded-sam2` | Grounded-SAM-2 mask backend |
| `Long-CLIP` | Official LongCLIP module used by CHAI entity retrieval |
| `vbench` | Optional VBench checkout, installed separately when quality evaluation is needed |

VBench is not a Git submodule because the accepted evaluator was originally a
snapshot in the CHAI source tree. Install the locked checkout with:

```bash
git clone https://github.com/Vchitect/VBench.git external/vbench
git -C external/vbench checkout c2dd0593fb913a24bc4f4ebd8803fd8c49c7db2d
```

It remains ignored by the parent repository so its large model and cache files
cannot be committed accidentally. The VBench Python dependencies are installed
by `scripts/create_environments.sh` into the `vbench` environment.

For Chorus, install the pinned Grounded-SAM-2 checkout into `wan` after the
base environments exist:

```bash
conda run -n wan python -m pip install --no-build-isolation -e external/grounded-sam2
conda run -n wan python -m pip install addict yapf timm "supervision>=0.22.0" pycocotools
conda run -n wan python -m pip install --no-build-isolation -e external/grounded-sam2/grounding_dino
```

Set `CUDA_HOME` to the matching toolkit before these commands. Place the SAM
and Grounding-DINO checkpoints under `models/Grounded-SAM-2-v1.0/` as described
in the root README, and cache a local `bert-base-uncased` snapshot before using
the mask backend.
