# Video Cache Baselines

统一承载 NIRVANA-VID、TeaCache、CHAI 和 Carnator 的 T2V 集成骨架，模型入口为 Wan2.2 与 CogVideoX1.5。Chorus 暂不实现，也不把 Carnator 对 Chorus2 的引用当作 Chorus。

## 运行

代码只保存算法插件和协议；模型源码、权重、视频和缓存通过外部路径提供。模型执行环境统一命名为 `wan`、`cogvideo`，质量评估环境为共享的 `vbench`；方法环境仍可按 `environments/` 中的模板独立创建：

```bash
scripts/create_environments.sh
PYTHONPATH=. python scripts/run/run.py --model synthetic --method chai \
  --prompt "a red car driving through a city" --run-name smoke --dry-run
```

每个模型/方法使用独立缓存 namespace（`artifacts/cache/<model>/<method>/`）。baseline 不读写持久化缓存；方法 hook 关闭时由同一 adapter 运行 dense 路径。质量评估统一通过 `vbench` conda 环境执行：

```bash
python scripts/benchmark/quality.py artifacts/runs/smoke
```

安装脚本会将旧名称 `chai-wan`、`chai-cogvideo`、`chai-vbench` 分别迁移为 `wan`、`cogvideo`、`vbench`。方法环境名为 `vcb-baseline`、`vcb-chai`、`vcb-teacache`、`vcb-nirvana-vid` 和 `vcb-carnator`，可独立安装或替换为兼容环境。

## 支持矩阵

| 方法 | Wan2.2 | CogVideoX1.5 | 状态 |
| --- | --- | --- | --- |
| CHAI | 支持，复用 `CHAI/chai` adapter | 支持，复用 `CHAI/chai` adapter | 已真实验收 |
| TeaCache | 支持，adapter 内 transformer 输出复用 | 支持，adapter 内 transformer 输出复用 | 已真实验收 |
| NIRVANA-VID | 支持，原生扩散 latent restart | 支持，Diffusers latent restart | 已真实验收 |
| Carnator | 原生 C0/B0--B5 矩阵入口 | source-latent-restart adapter | 已真实验收 |
| Carnator + CHAI 新基线 | Wan block-0 Cache Attention + B0--B5 后端组合 | N/A（Wan 专用） | 已真实验收，见 `docs/ACCEPTANCE.md` |
| Chorus | N/A | N/A | 本阶段不实现 |

N/A 会保留在结果汇总中，不静默跳过。所有结果逐请求写入 JSONL，再由 `scripts/aggregate/results.py` 自动汇总。

## 目录

- `core/`：协议、方法注册、namespace cache、runner 和结果类型。
- `adapters/`：Wan2.2、CogVideoX1.5 和 CPU synthetic adapter。
- `methods/`：方法插件；不复制模型源码。
- `configs/`：模型、方法、支持矩阵和 smoke 配置。
- `environments/`：每方法独立环境及共享 `vbench` 环境。
- `external.lock.json`：来源仓库与固定 commit 记录。

当前范围仅为 T2V；不包含 I2V、训练或 OpenSora 适配。

Carnator 需要初始化三个固定 submodule。B2/B5 使用 SpargeAttn 扩展；运行前应确认该扩展与 `wan` 环境的 PyTorch/CUDA 版本匹配。

```bash
git -C ../CARNATOR_ICLR2027 submodule update --init external/Wan2.2 external/LightX2V external/SpargeAttn
conda run -n wan env PYTHONPATH=. python scripts/run/carnator_wan_matrix.py \
  --source-prompt "a dog is sleeping on grass" --target-prompt "a cat is sleeping on grass" --run-id wan-matrix
```

来源 commit 可用 `PYTHONPATH=. /usr/bin/python3 scripts/check_environment.py` 校验。

真实运行与质量验收结果见 `docs/ACCEPTANCE.md`。Carnator 矩阵可用
`scripts/benchmark/carnator_quality.py <matrix.json> --output-dir <dir>` 评测。

跨请求 source/target 运行 CHAI、NIRVANA-VID 和 Carnator：

```bash
PYTHONPATH=. python scripts/run/source_target_matrix.py --model cogvideox15 \
  --source-prompt "a dog is sleeping on a grassland, focus on the dog" \
  --target-prompt "a cat is sleeping on a grassland, focus on the cat" \
  --methods chai,nirvana_vid,carnator --run-prefix cog-source-target
```

baseline 和 TeaCache 是 target-only 方法，在同一实验中只传 target prompt；不要为它们制造 source cache 命中。
