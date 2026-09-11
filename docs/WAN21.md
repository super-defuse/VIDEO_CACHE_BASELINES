# Wan2.1 (T2V-1.3B) 基线实验

在 Wan2.1-T2V-1.3B 上复现 Wan2.2 的五个基线（baseline / nirvana_vid / chorus / chorus2 / carnator_boundary_b5）。
算法逻辑与 Wan2.2 版本完全一致，只有驱动 DiT 的代码按 Wan2.1 的接口差异改写。

## 快速开始

```bash
cd /home/tangxuebin/project/VIDEO_CACHE_BASELINES

# 10 请求测试（5 个基线并行，GPU 3-7，跑完自动聚合报告）
bash scripts/benchmark/run_wan21_all_test.sh

# 300 请求正式实验（支持中断后续跑：直接重跑同一脚本）
bash scripts/benchmark/run_wan21_all_formal.sh
```

两个脚本内部调用 `run_vidprom_300_experiment.py --model wan21`，每个方法跑完后自动执行
质量评估（`conda run -n vbench quality.py`，CLIP-T + VBench 六维度），全部完成后调用
`aggregate_vidprom_300_results.py --model wan21` 生成 summary.csv / hit_rate 曲线 / 汇总表。

### 聚合报告指标

汇总表 `VIDPROM_1000CACHE_300INPUT_RESULTS_TABLE.md` 每个方法一行，含：

| 指标 | 含义 |
|------|------|
| Requests / Completed / Errors | 请求总数、终态数（ok+cache_miss）、错误数 |
| Hits / Hit rate | 回放命中数与命中率（baseline 无缓存，两列均为 N/A） |
| E2E s/request | 端到端时延均值（排除存视频与哈希，含 source 准备） |
| E2E speedup | **加速比 = dense baseline 的 E2E 均值 ÷ 该方法 E2E 均值** |
| Denoising s/request | 去噪时延均值（CUDA 同步的 DiT 前向，Carnator 含 B5 探针） |
| Denoising speedup | 加速比 = baseline 去噪均值 ÷ 该方法去噪均值 |
| Hit E2E s/hit、Hit denoising s/hit | 仅命中子集的端到端 / 去噪均值 |
| Hit E2E speedup、Hit denoising speedup | 上述命中均值相对 dense baseline 均值的加速比 |
| External s/request | 仅 Chorus 的 LLM + Grounded-SAM2 耗时（已计入 E2E） |
| CLIP-T | 文本-视频一致性（VBench） |
| VBench-Q | 六维度均值（subject/background consistency、motion smoothness、dynamic degree、aesthetic、imaging quality） |

baseline 自身作为加速比参照，其 speedup 列恒为 N/A；`cache_miss` 行不计入时延与质量统计。

## 实验流程（回放模式）

命中/未命中决策固定复用 CogVideoX1.5 实验产生的回放文件（不可变真值）：
`artifacts/vidprom_1000cache_300input_noun_match/cogvideox15_replay/{method}.jsonl`

- **hit**：跑 source 请求 → 存缓存（临时目录）→ target 复用缓存出视频 → 记录指标 → 清理缓存
- **miss**：直接跳过，不生成（status=cache_miss，latency 0）
- **baseline**：无回放文件，全部 300 个请求都完整生成
- 临时缓存在 `results_wan21*/<method>/tmp_cache/`，每个请求的 finally 中删除

## GPU 分配

| 基线 | GPU | Wan2.2 等效实现要点 |
|------|-----|---------------------|
| baseline | 3 | 原生 50 步全量推理 |
| nirvana_vid | 4 | 复用 step-5 latent，`reuse_start_step: 5` |
| chorus | 5 | SRD + TGAA，K1=5 / K2=30，target 执行 45 步，latent 融合区 [5, 30) |
| chorus2 | 6 | 全 30 层 SpargeAttn（sparge-sage2），step 11 起，target 执行 50 步 |
| carnator_boundary_b5 | 7 | B5 探针 / 软 latent 融合（boundary-blend + smoothstep）/ 区域稀疏 + w8a8 FFN |

## 与 Wan2.2 参考实验的参数对齐

算法常量全部取自 Wan2.2 参考实验 `early_signature_disabled_20260903_100305`，Wan2.1 只替换驱动 DiT 的代码。

| 基线 | 对齐后的常量 |
|------|--------------|
| baseline | 50 步 dense |
| nirvana_vid | `reuse_start_step=5`，target 执行 45 步 |
| chorus | `k1=5, k2=30`，target 执行 `50-K1=45` 步，Stage2 latent 融合覆盖 completed step 5→29，Stage3 于 step 30 重置 UniPC 历史 |
| chorus2 | `sparse_start_step=10`（掩码自 completed step 11 起），全 30 层，`q_block=64/k_block=128/top_p=0.95/min_top_k=0.1`，`layer_refresh_size=8`，source latent cache `not_used`，target 执行 50 步 |
| carnator_boundary_b5 | `mask_probe_step=5`、`reuse_start_step=10`、区域稀疏 10-20、外层稀疏 0-9/21-29、`middle_attention_mask=True`、`ffn_backend=w8a8`、`outer_attn_backend=sparge-sage2`、软 latent 融合 `boundary-blend`（alpha=`smoothstep`）；source 产出**逐名词** Early Signature + 融合掩码模板 + 完整 latent trajectory + 外层路由掩码，target 按 `select_per_noun_inference_mbase` 取“source 独有名词”的 mask 后复用推理 |

> **Carnator 软/硬 latent 融合**：Wan2.2 的 300 请求产物记录的是 `hard`（`band_fraction = 0`），但同名 acceptance 实验（`carnator-boundary-B5`）与算法定义都是软融合，此处按软融合对齐。如需与旧产物逐位一致，把 `configs/models/wan21.yaml` 的 `latent_injection_mode` 改回 `hard` 即可。
>
> **Carnator 探针口径**：Wan2.1 的 target 同样跑 5 步 dense target probe（写 target Early Signature，其 DiT 时间计入 denoising），故 `executed_steps = probe 5 + reuse 40 = 45`，与 Wan2.2 一致。区别在于：命中判定在 Wan2.1 由回放文件事先固定，probe 的 signature 对比结果只写进 `extra`（`probe_comparison`）供审计，**不参与任何选源或命中决策**。

## Wan2.1 与 Wan2.2 的代码差异

| 差异 | 处理方式 |
|------|----------|
| timestep：Wan2.1 标量 `[B]` vs Wan2.2 每token `[1,seq]` | `wan_ti2v_5b.timestep_for` 按 `runtime.scalar_timesteps` 分发；chorus/nirvana 直接 `torch.stack([t])`（此前误用 wan2.2 写法导致 68TB 假 OOM，已修复） |
| modulation：`[B,1,C]` chunk(dim=1) vs `[1,T,C]` | `wan21_native.wan21_sparse_block_forward` 与 `wan21._native_srd_block` 按 Wan2.1 布局改写 |
| token 网格：109120 vs 27280 tokens | `runtime.expected_token_grid` 由 `wan21_token_grid()` 计算 |
| VAE 16 通道 vs 48 通道 | chai delegate 按配置加载，解码路径不变 |
| utils 只有 `cache_video`（无 `save_video`） | `Wan21Adapter.save_video` 直接用 `cache_video` |

## 文件结构

```
configs/models/wan21.yaml            # 模型 + 各基线算法常量
adapters/wan21.py                    # 五个基线的 Wan2.1 适配器
carnator/adapters/wan21_native.py    # carnator/chorus2 共用的 Wan2.1 native 驱动
core/wan21_runner.py                 # 回放 runner（hit/miss/baseline）
scripts/benchmark/
  run_vidprom_300_experiment.py      # --model wan21 入口
  aggregate_vidprom_300_results.py   # --model wan21 聚合
  run_wan21_all_test.sh              # 10 请求测试
  run_wan21_all_formal.sh            # 300 请求正式实验
external/Wan2.1/                     # Wan2.1 源码
models/Wan2.1-T2V-1.3B/              # 权重（VAE/DiT/T5）
```

## 监控与排查

```bash
tail -f artifacts/wan21_experiment_logs/*_$(date +%Y%m%d)*.log
ls artifacts/vidprom_1000cache_300input_noun_match/results_wan21_test/*/results.jsonl
```

- 单方法重跑：`run_vidprom_300_experiment.py --model wan21 --method <m> --gpu <g> --limit 10 --replay-root <replay> --output-root <root> [--resume|--overwrite]`
- 依赖：conda env `wan`（sparge-sage2 需 H100 sm90）；质量评估用 env `vbench`；carnator 需要 `pip install scikit-image`
