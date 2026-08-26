# Video Cache Baselines

一个独立的 T2V 缓存加速基线仓库，包含 Baseline、CHAI、TeaCache、NIRVANA-VID、Carnator、Chorus2 和 Chorus。方法实现位于本仓库；Wan2.2、CogVideo、SpargeAttn 等上游源码只位于 `external/`。

## Environments

| Conda 环境 | 用途 |
| --- | --- |
| `wan` | 所有 Wan2.2 方法，包括 Carnator 和 Chorus |
| `cogvideo` | CogVideoX1.5 方法 |
| `vbench` | CLIP-T 与 VBench-Q 质量评测 |

## Install

需要 Linux、Conda、Git、NVIDIA 驱动和与 PyTorch wheel 匹配的 CUDA toolkit。

```bash
git submodule update --init --recursive
# Replace cu128 with the CUDA wheel index matching the machine. The default
# torch 2.8/torchvision 0.23 pair matches the pinned Wan Triton dependency.
PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 scripts/create_environments.sh
PYTHONPATH=. python3 scripts/check_environment.py
```

如需使用同一 CUDA index 下的其他已验证 PyTorch 版本，显式传入匹配版本：
`PYTORCH_VERSION=... TORCHVISION_VERSION=...`。

`scripts/create_environments.sh` 会创建或校验 `wan`、`cogvideo` 和 `vbench`。
如果发现目录是错误的 venv、存在 `pyvenv.cfg`，或 `bin/python` 指向其他环境，
脚本会先将原目录改名为带时间戳的 `.invalid.*` 备份，再创建真正隔离的 Conda 环境，
不会直接删除原目录。

检索依赖按运行时拆分：

- `wan`：CHAI 的 spaCy/FAISS，以及 Chorus 的 ChromaDB。
- `cogvideo`：CHAI 的 spaCy/FAISS；Chorus 对 CogVideoX1.5 不支持，因此不装 ChromaDB。
- `wan`、`vbench`：分别为 Wan 上游导入链和 VBench 评测安装 Conda-forge 的
  `decord=0.6.0` Python 3.10 构建，避免 PyPI wheel 的平台标签问题。
- `wan`、`cogvideo` 使用 `numpy==2.2.6`，以匹配论文指定的 `faiss-cpu==1.13.0`；
  `vbench` 保持独立的 NumPy 1.26 评测依赖。

脚本会自动安装 `requirements/retrieval-wan.txt` 和
`requirements/retrieval-cogvideo.txt`。手动安装或在已有的正确 Conda 环境中补装时，使用：

```bash
conda run -n wan env PYTHONNOUSERSITE=1 python -m pip install -r requirements/wan.txt
conda run -n wan env PYTHONNOUSERSITE=1 python -m pip install -r requirements/retrieval-wan.txt
conda run -n cogvideo env PYTHONNOUSERSITE=1 python -m pip install -r requirements/cogvideo.txt
conda run -n cogvideo env PYTHONNOUSERSITE=1 python -m pip install -r requirements/retrieval-cogvideo.txt
```

Wan 基础推理需要 FlashAttention。若服务器无法从 PyPI 取得预编译包，下载官方
`flash_attn-2.8.3.post1+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl`
后传到服务器，并在 `wan` 中安装：

```bash
sha256sum flash_attn-*.whl  # 93f44772f2d63814086666b30dfa1808b96ca5cbd2efe292a96cafa7e6e7df2e
conda run -n wan env PYTHONNOUSERSITE=1 python -m pip install --no-deps ./flash_attn-*.whl
```

该 wheel 匹配 Python 3.10、Torch 2.8/cu128、Linux x86_64；使用预编译包不需要本机
`nvcc` 参与 FlashAttention 编译。官方文件位于
`Dao-AILab/flash-attention` 的 `v2.8.3.post1` release。

不要使用 `pip install --user`，也不要使用系统 `python3` 安装运行时依赖。
安装后检查实际解释器和检索依赖：

```bash
conda run -n wan env PYTHONNOUSERSITE=1 python scripts/check_retrieval_environment.py --runtime wan
conda run -n cogvideo env PYTHONNOUSERSITE=1 python scripts/check_retrieval_environment.py --runtime cogvideo
```

CHAI 还需要官方 LongCLIP Python 模块、`longclip-L.pt` checkpoint，以及 spaCy 模型。LongCLIP
源码已锁定在 `external/Long-CLIP` submodule；首次 clone 后执行 `git submodule update --init --recursive`，
并将该目录加入 `PYTHONPATH`：

```bash
conda run -n wan env PYTHONNOUSERSITE=1 python -m spacy download en_core_web_sm
conda run -n cogvideo env PYTHONNOUSERSITE=1 python -m spacy download en_core_web_sm
```

LongCLIP 不在稳定 PyPI 依赖中。将官方模块放入 `PYTHONPATH`，并在
`configs/methods/default.yaml` 的 `retrieval.chai.longclip_checkpoint` 填写 checkpoint
路径。完成安装后再运行 synthetic smoke test；真实模型运行还需要对应的模型权重。

安装完成后可清理下载缓存（不会删除环境内已安装的包）：

```bash
conda clean --yes --tarballs --index-cache --logfiles
conda run -n wan env PYTHONNOUSERSITE=1 python -m pip cache purge
```

若网络访问 PyPI 较慢，可临时给脚本设置镜像，例如：
`PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple`；PyTorch 仍需使用与 CUDA
匹配的 `PYTORCH_INDEX_URL`。

模型权重不提交。模板默认从仓库内的 `models/` 读取；将权重放入对应目录，或修改
`configs/models/*.yaml` 中的 `model_path`。

启用 VBench 质量评测：

```bash
git clone https://github.com/Vchitect/VBench.git external/vbench
git -C external/vbench checkout c2dd0593fb913a24bc4f4ebd8803fd8c49c7db2d
mkdir -p artifacts/vbench-cache/clip_model
conda run -n vbench env PYTHONNOUSERSITE=1 python -c \
  'import clip; clip.load("ViT-B/32", download_root="artifacts/vbench-cache/clip_model")'
```

启用 Chorus 的 Grounded-SAM-2（需要匹配 PyTorch 的 `CUDA_HOME`）：

```bash
conda run -n wan env PYTHONNOUSERSITE=1 python -m pip install --no-build-isolation -e external/grounded-sam2
conda run -n wan env PYTHONNOUSERSITE=1 python -m pip install addict yapf timm "supervision>=0.22.0" pycocotools
conda run -n wan env PYTHONNOUSERSITE=1 python -m pip install --no-build-isolation -e external/grounded-sam2/grounding_dino
```

将 `sam2.1_hiera_large.pt` 放到
`models/Grounded-SAM-2-v1.0/checkpoints/`，将 `groundingdino_swint_ogc.pth`
放到 `models/Grounded-SAM-2-v1.0/gdino_checkpoints/`，并预先缓存
`bert-base-uncased`，例如：

```bash
HF_ENDPOINT=https://hf-mirror.com conda run -n wan env PYTHONNOUSERSITE=1 python -c \
  'from huggingface_hub import snapshot_download; snapshot_download("bert-base-uncased")'
```

Chorus 如需调用 DMXAPI prompt diff，设置 `LLM_API_KEY`；未设置时使用内置 lexical fallback。

Carnator 的额外加速后端（Wan 基础推理已由 `requirements/wan.txt` 安装 FlashAttention）：

```bash
conda run -n wan env PYTHONNOUSERSITE=1 python -m pip install --no-build-isolation -e external/spargeattn
```

SpargeAttn 需要本机 CUDA 编译工具链；只运行 Baseline、CHAI、TeaCache、NIRVANA-VID
或 Chorus 时可以跳过；Carnator 的 SpargeAttn 变体和 Chorus2 均需要安装。
VBench、Grounded-SAM-2 的来源和固定版本见 [external/README.md](external/README.md)。

## Run

```bash
conda run -n wan env PYTHONPATH=. python scripts/run/run.py \
  --model synthetic --method chai --prompt "a red car driving through a city" \
  --run-name smoke --dry-run

# 使用真实模型配置验证 Runner、method、cache 和结果序列化链路；不会加载权重
conda run -n wan env PYTHONPATH=. python scripts/run/run.py \
  --model wan22 --method nirvana_vid --prompt "a red car driving through a city" \
  --run-name wan-dry-run --dry-run
conda run -n cogvideo env PYTHONPATH=. python scripts/run/run.py \
  --model cogvideox15 --method baseline --prompt "a red car driving through a city" \
  --run-name cogvideo-dry-run --dry-run

conda run -n wan env PYTHONPATH=. python scripts/run/carnator_wan_matrix.py \
  --source-prompt "a dog is sleeping on grass" --target-prompt "a cat is sleeping on grass" \
  --source-positive dog --run-id wan-matrix

conda run -n wan env PYTHONNOUSERSITE=1 PYTHONPATH=. python scripts/run/chorus2_wan.py \
  --source-prompt "a dog is sleeping on a grassland, focus on the dog" \
  --target-prompt "a cat is sleeping on a grassland, focus on the cat" \
  --seed 20260218 --run-id dog-to-cat

# One canonical source Dense cache for NIRVANA-VID, Chorus, Chorus2 and Carnator,
# followed by the single universal target C0 Dense timing. Use only GPU 3--7.
CUDA_VISIBLE_DEVICES=3 conda run -n wan env PYTHONNOUSERSITE=1 PYTHONPATH=. CARNATOR_ENV_READY=1 \
  python -m carnator.adapters.wan_ti2v_5b --no-env-setup --device cuda:0 \
  --prepare-shared-dense-cache --run-id shared-dense

conda run -n vbench env PYTHONPATH=. python scripts/benchmark/quality.py artifacts/runs/smoke

# Cross-request retrieval benchmark for NIRVANA-VID, CHAI, Chorus, and Carnator.
# Carnator creates sidecars progressively in arrival order and stops at step 10.
PYTHONPATH=. python3 scripts/benchmark/cross_request_retrieval.py

# Retrieval-only replay: assume the logical step-10 cache exists and stop at step 5.
PYTHONPATH=. python3 scripts/benchmark/cross_request_retrieval.py --assume-step10-cache
```

`--dry-run` 会走对应模型配置和原有推理编排，但使用确定性的测试 adapter，不执行 CUDA
去噪，也不代表视频质量或实际吞吐。要执行真实生成，需先按上文放置模型权重，再去掉
`--dry-run`；当前仓库不提交任何权重。

当前环境已用 Wan2.2-TI2V-5B 做最小真实 smoke（256×256、17 帧、2 步）验证：
FlashAttention CUDA 调用、Runner 落盘，以及 Baseline、TeaCache、NIRVANA-VID、CHAI、Chorus
的接线均通过；这不替代完整实验。

### VidProM 300-request experiment

正式实验的唯一入口是 [`scripts/benchmark/run_vidprom_300_experiment.py`](scripts/benchmark/run_vidprom_300_experiment.py)。
它按 CSV 到达顺序在 GPU 3--7 分别运行五个基线，使用 923-entry source view；Carnator
仅在有候选时在线执行 step-5 Early Signature probe，所有新请求不写入 cache。

```bash
# 每个基线先执行一次 --check-only，再在独立终端运行 --limit 300
conda run -n wan env PYTHONNOUSERSITE=1 PYTHONPATH=. python scripts/benchmark/run_vidprom_300_experiment.py \
  --method nirvana_vid --gpu 3 --limit 300

# 生成质量和汇总结果（全部基线完成后）
conda run -n vbench env PYTHONPATH=. python scripts/benchmark/quality_runner.py \
  --results artifacts/vidprom_1000cache_300input/results/<method>/results.jsonl
PYTHONPATH=. python scripts/benchmark/aggregate_vidprom_300_results.py
```

GPU 分配、固定阈值、日志字段和产物清理规则见 [`docs/VIDPROM_1000CACHE_300INPUT_EXPERIMENT_PLAN.md`](docs/VIDPROM_1000CACHE_300INPUT_EXPERIMENT_PLAN.md)。

## Support

| 方法 | Wan2.2 | CogVideoX1.5 |
| --- | --- | --- |
| Baseline / CHAI / TeaCache / NIRVANA-VID | 支持 | 支持 |
| Carnator | 原生 C0/B0--B5 matrix | source-latent-restart adapter |
| Chorus2 | 原生 SpargeAttn sparse-attention CLI | N/A |
| Chorus / TGAA and no-TGAA ablations | source/target matrix | N/A |
| Chorus TGAA joint-gamma probes (`1.30`, `1.6`, `2.0`) | source/target matrix | N/A |

## Layout

- `core/`：协议、cache namespace、runner 和外部依赖解析。
- `adapters/`、`methods/`、`carnator/`：全部基线实现与模型 hook。
- `external/`：锁定的第三方源码；提交见 `external/lock.json`。
- `configs/`、`requirements/`、`environments/`：运行配置与三个 Conda 环境。
- `docs/`：方法、adapter、评测、验收和来源记录。

当前仅覆盖 T2V。文档职责和入口见 [docs/README.md](docs/README.md)；运行结果与验收数据见
[docs/ACCEPTANCE.md](docs/ACCEPTANCE.md)。
