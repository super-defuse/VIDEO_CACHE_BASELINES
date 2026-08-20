# Video Cache Baselines

一个独立的 T2V 缓存加速基线仓库，包含 Baseline、CHAI、TeaCache、NIRVANA-VID、Carnator、CHAI-Carnator 和 Chorus。方法实现位于本仓库；Wan2.2、CogVideo、SpargeAttn 等上游源码只位于 `external/`。

## Environments

| Conda 环境 | 用途 |
| --- | --- |
| `wan` | 所有 Wan2.2 方法，包括 Carnator、CHAI-Carnator 和 Chorus |
| `cogvideo` | CogVideoX1.5 方法 |
| `vbench` | CLIP-T 与 VBench-Q 质量评测 |

## Install

```bash
git submodule update --init --recursive
# Replace cu128 with the CUDA wheel index matching the machine.
PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 scripts/create_environments.sh
PYTHONPATH=. python3 scripts/check_environment.py
```

模型权重不提交；在 `configs/models/*.yaml` 中设置 `model_path`。真实质量评测还需要把锁定的 VBench checkout 放到 `external/vbench`，详见 [external/README.md](external/README.md)。Carnator B1--B5 需要在 `wan` 中编译 FlashAttention；B2/B5 还需要 SpargeAttn：

```bash
conda run -n wan python -m pip install --no-build-isolation flash-attn==2.8.3.post1
conda run -n wan python -m pip install --no-build-isolation -e external/spargeattn
```

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
| Chorus TGAA joint-gamma probes (`1.30`, `1.6`) | source/target matrix | N/A |

## Layout

- `core/`：协议、cache namespace、runner 和外部依赖解析。
- `adapters/`、`methods/`、`carnator/`：全部基线实现与模型 hook。
- `external/`：锁定的第三方源码；提交见 `external/lock.json`。
- `configs/`、`requirements/`、`environments/`：运行配置与三个 Conda 环境。
- `docs/`：方法、adapter、评测、验收和来源记录。

当前仅覆盖 T2V。运行结果与验收数据见 [docs/ACCEPTANCE.md](docs/ACCEPTANCE.md)。
