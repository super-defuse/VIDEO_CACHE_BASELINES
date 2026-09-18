# 五基线多 Prompt 验收脚本说明

## 目标

`scripts/run/run_acceptance_experiment.py` 是唯一的编排入口：输入一组或多组
source→target prompt pair，让五条基线各自在 GPU3--7 上完成「准备自己的 source → 跑
target → 删掉自己的 source 缓存」，最后统一做质量评测并保留每个视频产物的三帧图。

五个基线固定为，各自独占一张物理卡：

| 基线 | GPU | target runner | source 输入 |
| --- | ---: | --- | --- |
| `dense_baseline` | 3 | `carnator.adapters.wan_ti2v_5b`（target-only 50-step） | **不跑 source**，不读任何缓存 |
| `nirvana_vid` | 4 | `scripts/run/shared_cache_wan_baseline.py --method nirvana_vid` | 自己的 `shared_dense` manifest，从 step-10 latent 重启 |
| `chorus` | 5 | `scripts/run/shared_cache_wan_baseline.py --method chorus` | 自己的 `shared_dense` manifest（K1/K2 默认 10/30） |
| `chorus2` | 6 | `scripts/run/chorus2_wan.py` | 自己的 `outer_block_masks`（50-step sparse-mask 协议） |
| `carnator_boundary_b5` | 7 | `scripts/run/carnator_baseline.py --baseline B5` | 自己的 `source_dense`（boundary-blend + smoothstep） |

`dense_baseline` 是 target-only（`cache_match` 标 `not_applicable`），但它同时充当其余四条
基线的 50-step 加速比分母，所以仍然每 pair 跑一次。

GPU 映射沿用仓库既有约定（`scripts/benchmark/run_vidprom_300_experiment.py` 的
`WAN21_GPU_BY_METHOD`），GPU0--2 不在本实验的可见范围内。

## 入口参数

```text
--source-prompt TEXT       单 pair 的 source prompt
--target-prompt TEXT       单 pair 的 target prompt
--seed INT                 默认 20260218
--pairs PATH               多 pair 的 CSV/JSONL，含 pair_id / source_prompt / target_prompt /
                           seed / source_positive
--source-positive TEXT     覆盖所有 pair 的 Carnator source mask 短语；不传则逐 pair 取值
--gpus 3,4,5,6,7           物理卡集合，仅允许 3--7
--allow-gpu-remap          按 --gpus 顺序重新分配，而不是用固定的 3--7 计划
--quality-gpu INT          显式指定则把所有质量评测压在这一张卡上串行；默认每条基线各用自己那张
--quality-env vbench       质量评测环境
--keep-source-cache        不删 source 缓存（默认只在 target 失败时保留）
--skip-quality             跳过 CLIP-T / VBench-Q
--boundary-alpha-function smoothstep | cosine
--chorus-k1 / --chorus-k2  Chorus 边界，默认 10 / 30
--reference-root PATH      参考产物目录，默认 artifacts/acceptance/wan22-ti2v-5b-dog-to-cat
--model-path / --run-root / --model-dtype / --dry-run / --validate-reference
```

`--source-positive` 语义很重要，见下文「source-positive 与区域掩码」。

## 执行流程

每个 pair 是一个完整的批次，多个 pair **串行**执行（一个 pair 正好占满 5 张卡，没有并行的
空间）。每个 pair 内部分三个阶段：

```
阶段 1（5 个 worker，一卡一个，并发）
  GPU3  dense_baseline      跑 target-only 的 50-step 控制组
  GPU4  source(nirvana_vid) 跑自己的 50-step source，导出 shared_dense
  GPU5  source(chorus)      同上
  GPU6  source(chorus2)     同上
  GPU7  source(carnator)    同上

阶段 2（4 个 worker，一卡一个，并发）
  GPU4  nirvana_vid target
  GPU5  chorus     target
  GPU6  chorus2    target
  GPU7  carnator   target

阶段 3（5 个 worker，一卡一个，并发）
  GPU3  quality(dense_baseline)
  GPU4  quality(nirvana_vid)
  …
  GPU7  quality(carnator_boundary_b5)
```

两阶段而非一起跑，是因为 chorus2 和 carnator 需要
`dense_baseline/cache/target_dense/manifest.json` 作为加速比参照
（`--speedup-reference-manifest` / `--target-dense-manifest`）。如果五个 worker 同时起跑，
chorus2 / carnator 启动时这份 manifest 还不存在，两者就会各自重新跑一遍 50-step dense
控制组（`carnator/adapters/wan_ti2v_5b.py` 里约 4.5 分钟）。

阶段 3 开始时所有 target 进程都已退出，3--7 整段是空的，所以质量评测按基线各占一张卡并发
跑（实测单条基线完整跑一遍 VBench 六个维度约 29 秒），而不是全部压在 GPU3 上串行。只有
`--quality-gpu` 显式指定时才会退回「一张卡串行跑五条」的旧行为。

质量阶段唯一被共享的资源是 `artifacts/vbench-cache/`（`VBENCH_CACHE_DIR`），里面是预下载的
六个模型权重目录，只读；每个基线自己的视频暂存目录 `input_videos_<fingerprint>/` 和结果文件
都写在各自的 `<baseline>/vbench/` 下，互不干扰。

## 目录结构

```text
artifacts/acceptance/<timestamp>/
  run_manifest.json                # GPU 计划、逐 pair 的 source_positive、cache manifest 摘要
  prompts.jsonl
  summary.json  summary.csv  report.md
  logs/
    source_<baseline>.log  <baseline>.log  quality_<baseline>.log
  _native_results/
    _sources/<baseline>/summary.json       # source 阶段的原生 summary
    <baseline>/summary.json                # target 阶段
    ti2v-5b/carnator_boundary_b5/B5/summary.json
  _sources/<baseline>/                     # 仅 4 条读缓存的基线有
    cache/                                 # ← target 成功后整棵删除
    videos/source_dense.mp4                # 保留
    manifest.json                          # 保留（shared_dense manifest 的副本）
    source_frame_{000,060,120}.png         # 保留（三帧）
  <baseline>/
    videos/target.mp4
    frames/target_frame_{000,060,120}.png  # 保留（三帧）
    results.jsonl  config.resolved.json  quality_input.jsonl  vbench/
```

多 pair 时每个 pair 再套一层 `run_root/<pair_id>/`，`pair_id` 经 `slug()` 规整。
单 pair 直接落在 `run_root/` 下（与历史产物同构）。

单挑出 source 的 run-id 是 `<baseline>/source` 放在 `_sources/` 这个**兄弟树**里，而不是
`<baseline>/source/`：`shared_cache_wan_baseline.py` 建产物目录用的是不带 `exist_ok` 的
`artifact_run.mkdir(parents=True)`，把 source 放在 target 目录里面会让 target 一启动就
`FileExistsError`。

## 视频截帧

每个视频产物在生成后立刻抽三帧存成 PNG，用第 0 / 60 / 120 帧（视频共 121 帧，即首、中、末）：

```text
<baseline>/frames/target_frame_{000,060,120}.png      target 视频
_sources/<baseline>/source_frame_{000,060,120}.png    source 视频
```

口径统一由 `sample_frames()` 负责，target 与 source 两条路径共用。它顺序读满 121 帧而不
用 `CAP_PROP_POS_FRAMES` 跳帧——H.264 上 seek 不可靠，顺序解码 121 帧不到一秒。

这些 PNG 只是给人看的旁证：不进 `summary.csv`，不被质量评测或加速比读取。四条读缓存基线
的 source 视频逐字节相同，所以同一 pair 下 4 组 `source_frame_*` 也是同一画面，看一组即可。

## source 缓存清理

- target 成功 → 删 `_sources/<baseline>/cache/`（单份实测 6.0 GB，其中
  `outer_block_masks/` 占 5.0 GB）。
- target 失败 → **保留**该 basline 的 source 缓存，在 `run_manifest.json` 的
  `retained_source_caches` 里登记，并在 `report.md` 列出，方便直接重试 target 而不用重跑
  source。
- `--keep-source-cache` 可全局关闭清理。

source 视频、三帧图和 manifest 副本不在 `cache/` 下，清理后仍然保留。

## source 一致性校验

每条读缓存的基线都会独立跑一遍 source。删缓存之前，编排器对这六个文件取指纹写进
对应 row 和 `run_manifest.json`：

```
cache/source_dense/latents_initial.pt
cache/source_dense/latents_step_009.pt
cache/source_dense/latents_step_049.pt
cache/source_dense/fusion_masks/masks.pt
cache/source_dense/outer_block_masks/metadata.json
videos/source_dense.mp4
```

`report.md` 的 `## Source consistency` 逐 pair 给出 PASS / WARN；不一致时列出具体文件和各
基线的摘要值。不一致意味着跨基线对比带上了 source 漂移，必须当成混淆处理。

两个实现细节：

- **不要**直接哈希 `manifest.json`（无论 `shared_dense` 还是 `source_dense`）。它们内嵌
  `cache_dir`、`source_video` 这类绝对路径，每条基线的路径必然不同，比出来的差异是假的。
- `outer_block_masks/metadata.json` 内嵌 `source_run_id`（值是 `_sources/nirvana_vid` 这类
  run id），所以它走 `normalized_json_digest()`：剥掉 `source_run_id`、`created_at` 和绝对
  路径字符串后再哈希，保留每个 mask 的 `density` / `coverage` / `head_density_*` 字段。

实测（2026-09-18，GPU3--7 各跑一份 dog→cat source）：四个文件的指纹与 2026-09-02 旧
`shared_cache` 逐字节相同，跨卡、跨 16 天完全可复现。

## source-positive 与区域掩码

`--source-positive` 是 Carnator 的 source mask 探针短语，必须是 source prompt 的子串
（大小写和空白归一化后匹配，取**第一次**出现的位置；匹配不到会抛 `ValueError`）。

它**只影响 fusion mask，不影响 source 轨迹**：dog→cat 在 `dog` 和空串两种取值下，
`latents_*.pt` 和 `source_dense.mp4` 逐字节相同，只有
`cache/source_dense/fusion_masks/masks.pt` 变了（`2264c5ac3b…` vs `60b980201a…`）。

后果不小：空串等于「全部 source token」→ 区域覆盖整帧 → target 被整帧拉向 source，
Carnator B5 的 CLIP-T 从 `0.3547` 掉到 `0.2866`，去噪从 122.78s 涨到 165.22s。其余四条
基线质量指标逐位不变（它们读的是 `outer_block_masks`，不是 fusion mask）。

所以**每对 prompt 都要显式给出主体词**。`--pairs` 文件里的 `source_positive` 字段是逐 pair
的取值；命令行 `--source-positive` 则覆盖所有 pair。

`--pairs` 里缺 `source_positive` 或留空会直接报错，而不是悄悄退化成整帧区域——想真的走
「全部 token」必须显式写 `--source-positive ""`。这个默认值曾经让 Carnator 的 CLIP-T 掉
0.07 而整个 run 依旧报 `completed`，所以这里选择让它响。

## prompt pair 配置

现成的一份在 `configs/experiments/acceptance_prompt_pairs.jsonl`，五对同场景换主体：

| pair_id | source | target | source_positive |
| --- | --- | --- | --- |
| `horse-to-zebra` | a horse is galloping across a dusty plain, focus on the horse | a zebra is galloping across a dusty plain, focus on the zebra | `horse` |
| `lion-to-tiger` | a lion is walking through tall grass, focus on the lion | a tiger is walking through tall grass, focus on the tiger | `lion` |
| `duck-to-swan` | a duck is swimming on a calm lake, focus on the duck | a swan is swimming on a calm lake, focus on the swan | `duck` |
| `bear-to-panda` | a bear is standing in a snowy forest, focus on the bear | a panda is standing in a snowy forest, focus on the panda | `bear` |
| `apple-to-orange` | an apple is resting on a wooden table, focus on the apple | an orange is resting on a wooden table, focus on the orange | `apple` |

选取原则：source 和 target 只换主体、场景和姿态保持不变，这样跨请求缓存才有意义，也
才落在 Carnator 热启动（target 从 source 的 step-(reuse_start_step-1) latent 起步，不是
从噪声起步）能覆盖的范围内——形状差异过大的替换做不出来。

## 结果协议

`summary.csv` 每 pair/baseline 一行：`pair_id`、`method`、`status`、`actual_gpu`、
`denoising_seconds`、`end_to_end_seconds`、`speedup_vs_dense`、`clip_t`、`vbench_q`、
`output_path`、`quality_input_path`、`error`、`log_path`、`quality_error`、`quality_log_path`。

加速比由编排器自己算：分母取本 pair 的 `dense_baseline` 行，`speedup_vs_dense` 用去噪
时长，`speedup_end_to_end_vs_dense` 用子进程上报的 `reported_latency_s`。不依赖 target
子进程里的 `speedup` 字段，因为各基线的取值位置不同。

`report.md` 单 pair 时是一张 9 列的基线表；多 pair 时表格多一列 `Pair`，并追加一节
`## Mean across pairs`，给出每条基线在全部 pair 上的均值与 CLIP-T / VBench-Q 的极差——
单个 prompt 分不开基线质量和这条 prompt 的运气，均值才看得出趋势。

denoising 计时不含模型加载、source cache 准备、质量评测和文件复制。

## 参考锚定

`--validate-reference` 把结果与 `docs/ACCEPTANCE.md` 的冻结表和
`artifacts/acceptance/wan22-ti2v-5b-dog-to-cat/` 逐项比较，输出 reference / actual / delta
和协议状态。该参考表只对应 dog→cat，因此多 pair 时这个开关会直接报错，避免把其它 pair
误判成漂移。

Chorus2 当前是完整 50-step sparse-mask 协议，而旧验收目录是 40-step latent-restart
协议，报告会标 `protocol_changed` 而不把差异当成回归。

## 失败处理

- 单个 target 失败 → 该基线的 row 记 `status: error` 和错误文本，其余基线照常落盘，整轮
  `status: partial`；source 缓存保留。
- source 阶段失败 → 该基线的 target 不会启动，同样记 error row。
- 质量评测失败 → 记 `quality_error` 和 `quality_log_path`，不影响已经生成的视频。
- 编排器整体崩溃（还没轮到任何 worker 上报）→ 五条基线各写一条 error row。

## 当前命令

单 pair（dog→cat，锚定参考表）：

```bash
/data/shared/conda_envs/wan/bin/python scripts/run/run_acceptance_experiment.py \
  --source-prompt "a dog is sleeping on a grassland, focus on the dog" \
  --target-prompt "a cat is sleeping on a grassland, focus on the cat" \
  --source-positive dog \
  --validate-reference
```

五个新 pair：

```bash
/data/shared/conda_envs/wan/bin/python scripts/run/run_acceptance_experiment.py \
  --pairs configs/experiments/acceptance_prompt_pairs.jsonl
```

必须先 `--dry-run` 一次，确认 GPU 计划、命令构造、逐 pair 的 `source_positive` 和目录布局，
不加载模型。dry-run 写入的 `status` 是 `dry_run`，产物可以随便删。

时间量级（20260918_172444 的 dog→cat 实测）：单 pair **16.5 分钟**——阶段 1 约 8.5 分钟、
阶段 2 约 5 分钟、质量评测约 2 分钟（改成五卡并发后约 0.5 分钟）。五个 pair 串行约 1.5 小时。
峰值磁盘：4 份并发 source 缓存约 24 GB，清理前短暂存在；跑完每个 pair 的净产物约 170 MB
（对比保留共享缓存的旧布局 6.1 GB）。

时间几乎全花在 source 上：四条基线各自跑一遍完全相同的 50 步推理（约 8.5 分钟/次），
产出的 latent 和视频逐字节相同——这正是 source 一致性校验要证明的东西。每对 prompt 约有
34 GPU·分钟花在这份重复计算上，是这套「各基线独立 source、用完即删」设计的固有代价。

解释器必须用 `/data/shared/conda_envs/wan/bin/python`：系统 `python3` 缺 FlashAttention-2，
`external/wan2.2/wan/modules/attention.py` 会直接 `assert FLASH_ATTN_2_AVAILABLE` 失败，
五条基线会一起报错——这个失败模式看起来像脚本坏了，其实是解释器选错。
