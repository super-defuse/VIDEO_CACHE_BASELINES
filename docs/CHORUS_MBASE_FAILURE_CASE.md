# Chorus Mbase/Medit 失败案例

这是一个独立的定性实验，不修改 VidProM 300-request 结果目录。实验固定使用 Wan2.2-TI2V-5B、50 steps、`seed=20260218`，并把全部产物写入：

```text
artifacts/chorus_mbase_failure_case_v5/
  source_dense/          # source 的完整 dense trajectory 和视频
  target_dense/          # target dense control
  chorus_mbase_only/     # target 的 Mbase-only Chorus 结果
  cache/                  # 本实验私有的 Chorus trajectory
  clip_similarity.json    # source/target prompt 的 CLIP cosine
  report.md
  experiment_manifest.json
```

## Prompt

```text
Source/cache:
a red apple rolling down a snow mountain, camera focus on the apple

Target:
a yellow peach rolling down a mountain, camera focus on the peach
```

两条 prompt 保持动作、构图和镜头一致，只改变前景对象及背景材质。实验脚本显式固定人工 prompt-diff：`red apple -> yellow peach`，并把 source 的 `snow` 标记为保留的背景差异，不交给 Grounded-SAM-2 分割。target 不跳过前 10 步：第 0-29 步只更新 `Mbase`，第 30-49 步恢复 dense 推理。实验预期是 target dense 生成“黄桃 + 普通山地”，而 Mbase-only Chorus 保留 source latent 的雪山背景，因而出现“黄桃 + 雪山”。最终是否保留错配必须以保存的视频为准。

本次按用户要求不检查 CLIP 阈值；结果会记录实际 cosine，并通过 `cache_prompt` 强制复用 source cache。还需检查 Chorus 结果的 `chorus_mask.update_mask` 为 `Mbase`、prompt-diff backend 为 `manual_override`，以及 Grounded-SAM-2 phrase 为 `red apple`。

## Run

脚本只允许物理 GPU `0`、`1`、`2`。本实验的当前命令将 source dense、target dense、CLIP 和 Chorus 均显式固定到 gpu0，串行运行；不会启动 gpu3-7。

```bash
cd project/VIDEO_CACHE_BASELINES
/data/shared/miniconda/bin/conda run -n wan env PYTHONNOUSERSITE=1 PYTHONPATH=. \
  python scripts/benchmark/run_chorus_mbase_failure_case.py \
  --source-gpu 0 --target-dense-gpu 0 --score-gpu 0 --chorus-gpu 0
```

若上一次实验已完成部分阶段，可加 `--resume`。脚本会拒绝覆盖已有实验目录；不会启动 `gpu3-7` 上的进程。

需要单独重跑某个阶段时，使用相同的 `--output-root` 和 `--stage source|target-dense|chorus-reuse|score|report`；子进程内部的 `cuda:0` 会由 `CUDA_VISIBLE_DEVICES` 映射到命令行指定的物理卡。

注意：source dense 必须保存 Chorus 所需的 `chorus-trajectory-v2` 逐步 latent；普通 baseline 的单个 latent 不能直接作为 Chorus cache。这里的 `source_dense/` 是 Chorus source capture 的完整 dense 路径，`target_dense/` 是独立的普通 dense control。

## Medit 对照

脚本的 `--update-mask` 仅影响本独立实验：`Mbase` 选择紧致的对象区域，`Medit` 选择扩张后的对象区域。Medit 会使用正常 `chorus` 方法及独立的缓存命名空间，不改变正常 Chorus 代码或覆盖 v5 的 Mbase 结果。

只使用 gpu0 运行 Medit 对照：

```bash
cd project/VIDEO_CACHE_BASELINES
/data/shared/miniconda/bin/conda run -n wan env PYTHONNOUSERSITE=1 PYTHONPATH=. \
  python scripts/benchmark/run_chorus_mbase_failure_case.py \
  --output-root artifacts/chorus_medit_failure_case_v1 --update-mask Medit \
  --source-gpu 0 --target-dense-gpu 0 --score-gpu 0 --chorus-gpu 0
```
