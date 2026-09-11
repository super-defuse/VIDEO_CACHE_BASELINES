# CogVideoX 适配：验证状态与已知问题

> 状态截至 2026-09-09。CogVideoX 适配**尚未完成 300 请求正式实验**，本文档明确区分
> 「已验证」「未验证/缺口」和「已修复的历史问题」，避免把计划当作结果。

## 1. 已验证的部分

| 项目 | 证据 |
| --- | --- |
| 五个基线 `--check-only` 全部 `ready` | `run_vidprom_300_experiment.py --model cogvideox15 --check-only` |
| 全仓单元测试通过 | 140 passed |
| Chorus / Chorus2 / Carnator 通过 256x256、9 帧、2 步 dense smoke 与小尺寸 backend smoke | backend smoke 日志 |
| forced miss / hit smoke 通过，临时 cache 清理成功 | `smoke_two_requests_20260830_final/` |
| 2 条 forced smoke 的真实性能 | 见下表 |
| replay 生成器可用 | `scripts/benchmark/build_cogvideo_hit_replay.py`，从旧 Wan 结果提取 hit/miss 与 source UUID |

### 2 条 forced smoke 性能（`smoke_two_requests_20260830_final/`）

| 方法 | Req #0 E2E | Req #0 去噪 | Req #1 | 平均 E2E | 平均去噪 |
| --- | --- | --- | --- | --- | --- |
| NIRVANA-VID | 257.30s | 204.41s | miss | 257.30s | 204.41s |
| TeaCache | 134.08s | 117.20s | 138.47s / 122.44s | **136.28s** | **119.82s** ⭐ |
| Chorus | 238.25s | 141.85s | miss | 238.25s | 141.85s |
| Chorus2 | 221.18s | 201.32s | miss | 221.18s | 201.32s |
| Carnator B5 | miss（仅 probe） | — | 231.68s / 188.26s | 231.68s | 188.26s |

> 这是 forced smoke，Req #0 / #1 的 hit/miss 是**预设**的，不代表真实命中率。
> Chorus 的外部处理（LLM+SAM2）用时 64.99s。

与 Wan 300 请求结果的对照（方法间差异基本在 ±5%，属样本差异）：
NIRVANA 256.09s / TeaCache 135.38s / Chorus 264.47s / Chorus2 222.44s / Carnator 205.18s。

## 2. 未验证 / 缺口

| 缺口 | 说明 |
| --- | --- |
| **300 请求正式实验未运行** | 本轮未启动任何 CogVideo 300 条正式请求，明确停在正式运行之前 |
| **质量评测未运行** | CogVideoX 结果没有 CLIP-T / VBench-Q 数据 |
| **"前 5 条真实请求"无 artifacts** | `EXPERIMENT_PLAN.md` 提到 25 条请求 0 error，但 artifacts 中只找到 2 条 forced smoke 记录 |
| **CogVideoX 版 dog-cat fusion mask 无实例** | 代码路径已实现，但未找到用 CogVideoX 模型实际生成并可视化的 mask bundle |
| **1 条测试常驻失败** | `test_cogvideox15_adapter.py::test_replay_runner_separates_source_time_and_always_cleans_tmp_cache`，计时口径不一致，见 §2.1 |

### 2.1 `target_e2e_s` 与 `latency_s` 的计时口径不一致（未修复）

`core/cogvideo_runner.py` 内两处口径不同：

- `latency_s = elapsed - save_s - hash_s`（第 127 行）——**计入** `source_prepare_s`
- `target_e2e_s = elapsed - source_prepare_s - save_s - hash_s`（第 136 行）——**不计入**
- `hit_e2e_s = latency_s`（第 129 行）——同样计入

上述测试断言 `target_e2e_s == latency_s`，差值恰好是 `source_prepare_s`，因此**稳定失败而非随机**。
[WORKFLOW.md](WORKFLOW.md) 的口径是「source cache preparation 不计入 target E2E 与 denoising，
但必须单独报告 `source_prepare_s`」，即第 136 行符合文档，`latency_s` 与 `hit_e2e_s` 与之相悖。

这是**口径选择问题**：先定下 `latency_s` 是否包含 source 生成成本，再统一三处代码与文档。
本轮只清理结构，未改动该逻辑。

`artifacts/vidprom_1000cache_300input_noun_match/carnator_mask_probe_gpu012_20260901/` 的
manifest 显示 `model_dir` 为 **Wan2.2-TI2V-5B**，不是 CogVideoX，不能作为 CogVideoX mask 的证据。

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
