# 六基线多 Prompt 验收脚本规划

## 目标

提供一个统一编排脚本：输入 `source_prompt`、`target_prompt` 和可选的 seed，先在 GPU0 生成一次 source 及跨请求缓存，再在 GPU0--2 以单卡方式运行六个 target 基线，最后统一评测并汇总结果。

六个基线固定为：

`Dense Baseline`、`NIRVANA-VID`、`TeaCache`、`Chorus`、`Chorus2`、`Carnator`。

Carnator 默认配置为 `boundary-blend B5`（`smoothstep`），同时允许命令行显式覆盖。

## 入口设计

编排入口为 `scripts/run/run_acceptance_experiment.py`，只负责参数校验、目录管理、子进程调度和汇总，不复制各 baseline 的推理实现。

实际参数：

```text
--source-prompt TEXT       必填
--target-prompt TEXT       必填
--seed INT                 默认 20260218
--model-path PATH          默认 Wan2.2-TI2V-5B 配置
--gpus 0,1,2               target 调度设备，默认 0,1,2
--boundary-alpha-function smoothstep
--pairs PATH               可选，多组 prompt 的 CSV/JSONL
--run-root PATH            默认 artifacts/acceptance
--quality-env vbench       质量评测环境
--source-positive TEXT     可选，覆盖 Carnator source mask 的正向短语（默认整个 source prompt）
--chorus-k1 INT            Chorus Stage-1 边界，默认 10
--chorus-k2 INT            Chorus Stage-2 边界，默认 30
--dry-run                  只构造命令和目录，不加载模型
--skip-quality             跳过 CLIP-T/VBench-Q 评测
```

单组 prompt 直接使用两个 prompt 参数；多组实验使用 `pairs`，每行至少包含 `pair_id`、`source_prompt`、`target_prompt`，并可覆盖 seed。

## 执行流程

1. 创建 `artifacts/acceptance/<timestamp>/`，写入不可变的 `run_manifest.json`（prompt、seed、模型指纹、采样参数、代码版本、GPU 分配和 cache fingerprint）。时间戳使用 `YYYYMMDD_HHMMSS`，冲突时追加序号。
2. GPU0 启动 source worker。source 只运行一次 50-step dense Wan2.2 推理，生成统一的 canonical cache 和 `shared_cache_manifest.json`。target 阶段只读该 manifest，不重新检索、不复制方法私有 cache。
3. 依次启动六个 target 子进程。每个进程设置独立的 `CUDA_VISIBLE_DEVICES`，只占一张卡；同一 GPU 上不并发，计时前完成 CUDA 同步和必要 warm-up。
4. 跨请求基线直接注入已匹配的 source cache，强制记录 `cache_match.hit=true`，不执行检索：NIRVANA-VID、Chorus、Chorus2、Carnator。Dense 和 TeaCache 不伪造跨请求命中。
5. 每个 target 完成后立即校验视频、`results.jsonl` 和 denoising 计时字段，再提取 target 首帧。
6. 六个 target 都成功后，在独立质量阶段调用现有 `scripts/benchmark/quality.py`/`quality_runner.py`，只保留 `CLIP-T` 和 `VBench-Q` 两个汇总指标。
7. 生成顶层 `summary.json`、`summary.csv` 和 `report.md`；失败的 baseline 写入错误、命令和日志路径，不能阻止其他 baseline 的可复现结果落盘。

## 现有实现的复用边界

编排器通过 subprocess 或明确的 Python API 调用现有入口：

| 基线 | target runner | source/cache 输入 |
| --- | --- | --- |
| Dense Baseline | 现有 Wan2.2 C0/dense runner | 不读取跨请求 cache |
| NIRVANA-VID | `scripts/run/shared_cache_wan_baseline.py --method nirvana_vid` | `latents_step_009.pt` |
| TeaCache | `scripts/run/shared_cache_wan_baseline.py --method teacache` | target-only，无 source cache |
| Chorus | `scripts/run/shared_cache_wan_baseline.py --method chorus` | 完整 source trajectory + source video |
| Chorus2 | `scripts/run/chorus2_wan.py` | `outer_block_masks`（完整 50-step target，不做 source latent restart） |
| Carnator | `scripts/run/carnator_baseline.py`（Wan2.2 TI2V-5B） | source trajectory、mask 和 B5 boundary 配置 |

Source worker 应复用当前 Carnator/Wan2.2 的 source preparation 能力，统一导出 NIRVANA-VID、Chorus、Chorus2、Carnator 所需的 payload；不能让四个 target runner 各自重新生成 source。

## GPU 调度

```text
阶段 A：GPU0
  source preparation -> canonical cache

阶段 B：GPU0、GPU1、GPU2
  dense       -> 任一空闲 GPU
  nirvana     -> 任一空闲 GPU
  teacache    -> 任一空闲 GPU
  chorus      -> 任一空闲 GPU
  chorus2     -> 任一空闲 GPU
  carnator B5 -> 任一空闲 GPU
```

调度器维护 GPU 锁和子进程表；最多同时运行三个 target。GPU0 的 source 进程必须退出并释放显存后才能接收 target。单个 target 的 denoising 计时不包含模型加载、source cache preparation、质量评测和文件复制。

## 目录结构

```text
artifacts/acceptance/<timestamp>/
  run_manifest.json
  prompts.jsonl
  shared_cache/
    source.mp4
    source_first_frame.png
    cache/                       # canonical source payload，target 只读
    manifest.json
  dense_baseline/
    videos/target.mp4
    frames/target_first_frame.png
    results.jsonl
    config.resolved.json
  nirvana_vid/
    videos/target.mp4
    frames/target_first_frame.png
    results.jsonl
    config.resolved.json
  teacache/
    ...
  chorus/
    ...
  chorus2/
    ...
  carnator_boundary_b5/
    videos/target.mp4
    frames/target_first_frame.png
    results.jsonl
    config.resolved.json
  quality/
    results.jsonl
    raw/                         # 可选，保留评测原始维度
  logs/
    source.log
    dense_baseline.log
    nirvana_vid.log
    teacache.log
    chorus.log
    chorus2.log
    carnator_boundary_b5.log
    quality_<baseline>.log
  summary.json
  summary.csv
  report.md
```

每个 baseline 子目录只保存自己的 target 产物；共享 source 视频和 cache 不复制到各子目录。所有路径写绝对路径或相对于本次 timestamp 根目录的规范化路径，避免汇总时发生路径串接错误。

## 结果协议

每个 baseline 的 `results.jsonl` 至少记录：`pair_id`、prompt、seed、method、cache hit/mode、output path、`denoising_seconds`、`clip_t`、`vbench_q`、status、error 和运行命令。顶层汇总每个 prompt/baseline 一行，只展示：

`denoising_seconds`、`CLIP-T`、`VBench-Q`。

`report.md` 额外列出 cache manifest fingerprint、实际 GPU、Carnator 配置、视频和首帧路径，便于审计；不把 source preparation 或 quality 时间混入 denoising 指标。

## 猫狗反向验证

脚本完成后先用现有验收协议运行一组：

```text
seed   = 20260218
source = a dog is sleeping on a grassland, focus on the dog
target = a cat is sleeping on a grassland, focus on the cat
Carnator = boundary-blend B5, smoothstep
```

验证脚本从 `docs/ACCEPTANCE.md` 及其对应 `results/*/summary.json` 读取参考值，与新 timestamp 目录逐项比较：

- 六个 baseline 都有成功视频、首帧和结果记录；
- NIRVANA-VID、Chorus、Chorus2、Carnator 的 cache 命中和 payload 路径符合 shared manifest；TeaCache 明确为 target-only；
- denoising 时间、CLIP-T、VBench-Q 使用同一字段和同一评测环境；
- 结果在可配置容差内接近 `wan22-ti2v-5b-dog-to-cat`，差异记录为环境/硬件差异，不手工改写结果；Chorus2 当前为完整 50-step sparse-mask 协议，而旧验收目录为 40-step latent-restart 协议，报告会标为 `protocol_changed`，不将数值差异误判为回归；
- `report.md` 输出逐 baseline 的 reference、actual、delta 和协议状态。

该验证只用于反向确认编排、缓存注入、计时和质量汇总正确；不改变原验收目录，也不覆盖已有产物。

## 断点与失败处理

每个阶段的状态、错误和日志路径都会写入 baseline `results.jsonl` 与顶层汇总；target 失败不会覆盖其他 baseline 的产物。source manifest 缺失或 prompt/seed 不匹配时，target 不会启动，并将原因写入六个 baseline 的错误记录。

## 当前命令

实现中的 runner 明确拒绝 GPU3--7；子进程均以 `CUDA_VISIBLE_DEVICES` 绑定到一个物理 GPU，且调度按三卡分批，避免同一张卡同时运行两个 target。

```bash
conda run --no-capture-output -n wan python scripts/run/run_acceptance_experiment.py \
  --source-prompt "a dog is sleeping on a grassland, focus on the dog" \
  --target-prompt "a cat is sleeping on a grassland, focus on the cat" \
  --source-positive dog \
  --seed 20260218 \
  --gpus 0,1,2 \
  --source-gpu 0 \
  --validate-reference
```

使用 `--dry-run --skip-quality` 只验证目录、命令构造和 GPU 隔离，不加载模型。多 prompt 使用 `--pairs <csv-or-jsonl>`，输入含 `pair_id`、`source_prompt`、`target_prompt`，可选 `seed`。
