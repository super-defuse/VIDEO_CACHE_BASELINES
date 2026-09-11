# CogVideoX 适配：验证状态与已知问题

> 状态截至 2026-09-11。CogVideoX 适配**尚未完成 300 请求正式实验**，本文档明确区分
> 「已验证」「未验证/缺口」和「已修复的历史问题」，避免把计划当作结果。

## 1. 已验证的部分

| 项目 | 证据 |
| --- | --- |
| 五个基线 `--check-only` 全部 `ready` | `run_vidprom_300_experiment.py --model cogvideox15 --check-only` |
| 全仓单元测试通过 | 159 passed（2026-09-11） |
| Chorus / Chorus2 / Carnator 通过 256x256、9 帧、2 步 dense smoke 与小尺寸 backend smoke | backend smoke 日志 |
| 五基线 10 样本 replay 部分跑通，miss 正确记为 `cache_miss` | `artifacts/cogvideox15_5baselines_10samples/` |
| 3 条真实请求跑通（2 命中 1 miss） | `artifacts/vidprom_1000cache_300input_noun_match/cogvideox15_3req_test_20260908_151456/` |
| replay 生成器可用 | `scripts/benchmark/build_cogvideo_hit_replay.py`，从旧 Wan 结果提取 hit/miss 与 source UUID |

## 2. 未验证 / 缺口

| 缺口 | 说明 |
| --- | --- |
| **300 请求正式实验未运行** | 本轮未启动任何 CogVideo 300 条正式请求，明确停在正式运行之前 |
| **质量评测只在 2 条命中上跑过** | 3 请求测试中 `nirvana_vid` 有 CLIP-T 0.302 / VBench-Q 0.632（两条命中均值，与 `EXPERIMENT_PLAN.md` 一致）；其余基线与 300 条规模均无质量数据 |
| **真实请求覆盖不足** | 只有 `nirvana_vid` 跑过 3 条真实请求（2 命中 1 miss）。`EXPERIMENT_PLAN.md` 建议每基线先做 5 条真实请求，尚未执行 |
| **无 CogVideoX 性能对照数据** | 仅 3 条请求，不足以做方法间比较；见下方关于 `smoke_two_requests_*` 的说明 |
| **CogVideoX 版 dog-cat fusion mask 无实例** | 代码路径已实现，但未找到用 CogVideoX 模型实际生成并可视化的 mask bundle |

> 仓内 `smoke_two_requests_20260830_final/` 与 `smoke_two_requests_20260830_final_pre_forced_smoke/`
> 两个目录的 `results.jsonl` 里 `model` 均为 **Wan2.2**（字段名也是 Wan runner 的
> `reported_latency_s` / `excluded_cache_prepare_s`），**不是 CogVideoX**。它们曾被当作
> CogVideoX smoke 性能引用，已更正；需要 Wan2.2 的 smoke 数据时可直接读这两个目录。
>
> `artifacts/vidprom_1000cache_300input_noun_match/carnator_mask_probe_gpu012_20260901/` 的
> manifest 显示 `model_dir` 为 **Wan2.2-TI2V-5B**，同样不能作为 CogVideoX mask 的证据。

## 3. 已修复的历史问题

### 3.1 `KeyError: 'chorus2'` / `KeyError: 'chorus'`（10 样本实验批量失败）

**症状**：`artifacts/cogvideox15_5baselines_10samples` 中，chorus2 与 carnator 的所有命中请求
立即报错，产出 0 个视频；但脚本把"进程未崩溃"记为 `completed successfully`，因此看起来
"chorus2 只用了 26.5s"，实为假象（每个请求 ~0.004s 就返回了）。

| 基线 | 真实状态 | 现象 |
| --- | --- | --- |
| baseline | ✓ 正常 | ~77s/请求，10 个视频 |
| nirvana_vid | ✓ 正常 | ~77s/请求，8 个视频（2 miss） |
| chorus | ✓ 正常 | ~180s/请求，8 个视频（2 miss） |
| chorus2 | ❌ 全部立即失败 | `KeyError: 'chorus2'` |
| carnator | ❌ 全部失败 | `KeyError: 'chorus'` |

**根因**：`scripts/run_5baselines_10samples*.py` 的配置拼装错误——
① 对 chorus2 误用 `"chorus"` 作为配置键；② 为 carnator 漏配其当时依赖的 `"chorus"` 键。

**已修复**：配置键改用 `method` 变量本身。同时 Carnator 不再依赖 `chorus` 配置
（见 §3.2）。

### 3.2 Carnator 使用外部分割，偏离 Wan2.2 算法

**症状**：早期 `carnator/cogvideox15.py` 通过 `adapter.config["chorus"]` 读取配置，调用
`GroundedSam2Segmenter` 与 `build_cogvideo_edit_mask()` 做外部分割——这是 Chorus 的做法，
不是 Carnator 的 fusion mask 算法。

**已修复**：重构为纯 fusion mask，`_mask()` 从 payload 加载预计算的 mask bundle，
不再依赖外部分割模型：

- 移除 `adapter.config["chorus"]` 依赖
- 移除 `GroundedSam2Segmenter` / `build_cogvideo_edit_mask()`
- `mask_policy` 由 `cogvideo-grounded-sam2-mbase/v1` 改为 `cogvideo-fusion-mask-v1`
- mask backend 标记为 `fusion_mask`

设计与参数见 [FUSION_MASK.md](FUSION_MASK.md)。

### 3.3 命中请求的 `latency_s` 计入了 source 生成成本（2026-09-11 修复）

**症状**：`test_cogvideox15_adapter.py::test_replay_runner_separates_source_time_and_always_cleans_tmp_cache`
稳定失败——`target_e2e_s != latency_s`，差值恰好是 `source_prepare_s`。

**根因**：`core/cogvideo_runner.py` 的 `latency_s` 只扣除 save/hash，没有扣除命中后的
source 准备区间。同仓库另外两个 runner 与协议文档都不是这个口径：

- [VIDPROM_1000CACHE_300INPUT_EXPERIMENT_PLAN.md](../VIDPROM_1000CACHE_300INPUT_EXPERIMENT_PLAN.md)
  第 93 行：`latency_s` / `hit_e2e_s`「仅扣除命中后的 cache prepare 区间，以及视频/缓存写盘和结果 hash」
- `core/wan21_runner.py`：`latency_s = elapsed - source_prepare_s - cache_load_s - cache_prepare_s - save_s - hash_s`
- `core/runner.py`（Wan2.2）：同口径

**已修复**：`latency_s` 扣除 `source_prepare_s`，`target_e2e_s` 直接取 `latency_s`
（与 `wan21_runner.py` 一致）。miss 的 `source_prepare_s` 为 0，不受影响。

**影响**：命中请求的 `latency_s` 会减少 `source_prepare_s`（实测约 74s/请求）。
`cogvideox15_3req_test_20260908_151456` 的数值与修复后口径一致，可以引用；但该次运行的
`denoising_s` 为 0（DiT 计时当时未接线），不要引用其去噪时间。

### 3.4 miss 请求未短路，会跑完整 dense 生成（2026-09-11 修复）

**症状**：`CogVideoReplayRunner.process()` 对 replay 决策为 `miss` 的请求返回
`status="ok"` 而非 `cache_miss`；请求会完整执行 dense 生成、落盘视频并计入耗时。

**根因**：`core/cogvideo_runner.py` 的 `decision == "miss"` 分支只构造了 `CacheMatch`，
没有提前返回，代码继续落到 target 生成段。

> ⚠️ 不能用 `if not match.hit` 短路：TeaCache 的 `CacheMatch(kind="target_only")`
> 同样是 `hit=False`，会被误判成 miss。

**后果**：若直接跑 300 条，所有 miss 会被 `aggregate_vidprom_300_results.py` 计入
`completed`（`status in {"ok","cache_miss"}`）并算作命中，污染 hit rate 与命中耗时均值。
这与 §3.1 是同一类陷阱：表象正常，结论错误。

**已修复**：`miss` 分支直接返回 `status="cache_miss"`、`latency_s=0`、`executed_steps=0`、
`extra={"replay": ..., "skipped": True}`，与 `core/wan21_runner.py`、`core/runner.py` 及
`run_vidprom_300_experiment.py` 中 `miss_policy = "skipped without generation (status cache_miss, latency 0)"`
一致。

> §3.3 与 §3.4 都是 `core/cogvideo_runner.py`（未提交的 WIP 文件，不在 Git 历史中）里的回归：
> `cogvideox15_5baselines_10samples` 当时就能正确产出 `cache_miss`，说明短路口径曾经存在。

## 4. 重构时的建议

1. 先在 `cogvideo` 环境确认模型权重、tokenizer、VAE、text encoder、transformer、scheduler 完整，
   再跑 `--check-only` → dry-run → 单请求 dense smoke。
2. 每个方法先做 forced miss/hit 两条 smoke，再做 5 条真实请求；检查 latent/token shape、
   scheduler step、seed determinism、GPU 隔离、清理后目录为空。
3. **不要依赖"脚本退出码"判断成功**——§3.1 的教训是进程正常退出但所有请求都在 error。
   必须检查 `results.jsonl` 的 `status` 字段和 `videos/` 是否非空。
4. 旧 replay 的 hit/miss 是 ground truth，执行失败应记为 `error`，不得改写成 `miss`，
   也不得因 probe 结果改变旧 hit rate。
5. 不得用 Wan 实现、source-latent-restart 兼容版或伪造 hit 结果替代独立的 CogVideo backend。
6. `core/cogvideo_runner.py` 目前不在 Git 历史中，改动容易丢——参考 §3.3 / §3.4 的回归，
   重构前先把它纳入版本管理。
