# Video Cache Baselines

一个独立的 T2V 缓存加速基线仓库，包含 Baseline、CHAI、TeaCache、NIRVANA-VID、Carnator、CHAI-Carnator 和 Chorus。方法实现位于本仓库；Wan2.2、CogVideo、SpargeAttn 等上游源码只位于 `external/`。

## Environments

| Conda 环境 | 用途 |
| --- | --- |
| `wan` | 所有 Wan2.2 方法，包括 Carnator、CHAI-Carnator 和 Chorus |
| `cogvideo` | CogVideoX1.5 方法 |
| `vbench` | CLIP-T 与 VBench-Q 质量评测 |

## Install

需要 Linux、Conda、Git、NVIDIA 驱动和与 PyTorch wheel 匹配的 CUDA toolkit。

```bash
git submodule update --init --recursive
# Replace cu128 with the CUDA wheel index matching the machine.
PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 scripts/create_environments.sh
PYTHONPATH=. python3 scripts/check_environment.py
```

模型权重不提交。模板默认从仓库内的 `models/` 读取；将权重放入对应目录，或修改
`configs/models/*.yaml` 中的 `model_path`。

启用 VBench 质量评测：

```bash
git clone https://github.com/Vchitect/VBench.git external/vbench
git -C external/vbench checkout c2dd0593fb913a24bc4f4ebd8803fd8c49c7db2d
mkdir -p artifacts/vbench-cache/clip_model
conda run -n vbench python -c \
  'import clip; clip.load("ViT-B/32", download_root="artifacts/vbench-cache/clip_model")'
```

启用 Chorus 的 Grounded-SAM-2（需要匹配 PyTorch 的 `CUDA_HOME`）：

```bash
conda run -n wan python -m pip install --no-build-isolation -e external/grounded-sam2
conda run -n wan python -m pip install addict yapf timm "supervision>=0.22.0" pycocotools
conda run -n wan python -m pip install --no-build-isolation -e external/grounded-sam2/grounding_dino
```

将 `sam2.1_hiera_large.pt` 放到
`models/Grounded-SAM-2-v1.0/checkpoints/`，将 `groundingdino_swint_ogc.pth`
放到 `models/Grounded-SAM-2-v1.0/gdino_checkpoints/`，并预先缓存
`bert-base-uncased`，例如：

```bash
HF_ENDPOINT=https://hf-mirror.com conda run -n wan python -c \
  'from huggingface_hub import snapshot_download; snapshot_download("bert-base-uncased")'
```

Chorus 如需调用 DMXAPI prompt diff，设置 `LLM_API_KEY`；未设置时使用内置 lexical fallback。

Carnator 的加速后端：

```bash
conda run -n wan python -m pip install --no-build-isolation flash-attn==2.8.3.post1
conda run -n wan python -m pip install --no-build-isolation -e external/spargeattn
```

上述两个后端需要本机 CUDA 编译工具链；只运行 Baseline、CHAI、TeaCache 或 NIRVANA-VID 时可以跳过。
VBench、Grounded-SAM-2 的来源和固定版本见 [external/README.md](external/README.md)。

## Run

```bash
conda run -n wan env PYTHONPATH=. python scripts/run/run.py \
  --model synthetic --method chai --prompt "a red car driving through a city" \
  --run-name smoke --dry-run

conda run -n wan env PYTHONPATH=. python scripts/run/carnator_wan_matrix.py \
  --source-prompt "a dog is sleeping on grass" --target-prompt "a cat is sleeping on grass" \
  --source-positive dog --run-id wan-matrix

conda run -n vbench env PYTHONPATH=. python scripts/benchmark/quality.py artifacts/runs/smoke
```

## Support

| 方法 | Wan2.2 | CogVideoX1.5 |
| --- | --- | --- |
| Baseline / CHAI / TeaCache / NIRVANA-VID | 支持 | 支持 |
| Carnator | 原生 C0/B0--B5 matrix | source-latent-restart adapter |
| CHAI-Carnator | 原生 CC-B0--CC-B5 matrix | N/A |
| Chorus / TGAA and no-TGAA ablations | source/target matrix | N/A |
| Chorus TGAA joint-gamma probes (`1.30`, `1.6`, `2.0`) | source/target matrix | N/A |

## Layout

- `core/`：协议、cache namespace、runner 和外部依赖解析。
- `adapters/`、`methods/`、`carnator/`：全部基线实现与模型 hook。
- `external/`：锁定的第三方源码；提交见 `external/lock.json`。
- `configs/`、`requirements/`、`environments/`：运行配置与三个 Conda 环境。
- `docs/`：方法、adapter、评测、验收和来源记录。

当前仅覆盖 T2V。运行结果与验收数据见 [docs/ACCEPTANCE.md](docs/ACCEPTANCE.md)。
