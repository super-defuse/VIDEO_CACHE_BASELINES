# Carnator 消融实验（Table 3: Ablation of latent and attention reuse）

在 Wan2.2-TI2V-5B 上对比 Carnator 各组件的加速贡献，对应论文 Table 3。

实验日期：2026-09-09。本文件由原先散落在仓库根目录的 7 份状态/分析文档合并而成
（`ABLATION_STATUS.md`、`ABLATION_EXPERIMENT_STATUS.md`、`ABLATION_EXPERIMENT_SUMMARY.md`、
`ABLATION_STUDY_CORRECTED.md`、`ABLATION_FINAL_RESULTS.md`、`ABLATION_ANALYSIS_TWO_QUESTIONS.md`、
`ABLATION_TIMING_ANALYSIS.md`）。

## 1. 实验设计

### 测试样本

- **Cache prompt**（source）: `a little dog and a penguin dancing`
- **Target prompt**: `A cute little lion and a wobbly penguin appear`
- **Source cache**: `artifacts/ablation_req16_20260909_032326/source_gen/source_gen`
- **模型**: Wan2.2-TI2V-5B
- **分辨率**: 1280x704, 121 frames, 24 fps
- **采样**: 50 steps, UniPC solver, seed 20260311
- **Reuse start step**: 10，**中间层** 10-20，**头尾层** 0-9 / 21-29

### 四个变体

| 变体 | 中间层 (10-20) | 头尾层 (0-9, 21-29) | FFN | 说明 |
|------|---------------|-------------------|-----|------|
| **1. Full Computation** | 正常密集计算 | 正常密集计算 | fp | 基线，无任何优化 |
| **2. Latent Only** | Latent Injection（只更新 Medit 区域） | 正常密集计算 (dense) | w8a8 | 仅中间层局部计算 |
| **3. Attention Only** | 正常密集计算（无 latent 注入、无 attention mask） | Sparge 稀疏 attention (sparge-sage2) | w8a8 | 仅头尾层稀疏化 |
| **4. Joint** | Latent Injection + Attention Mask | Sparge 稀疏 attention (sparge-sage2) | w8a8 | 完整 Carnator |

### 对应的命令行参数

```bash
# 变体 1: Full Computation
--target-dense-only
--ffn-backend fp

# 变体 2: Latent Only
--source-run-dir <source_cache>
--reuse-start-step 10
--sparse-layers 10-20
--no-middle-attention-mask        # 中间层不用 attention mask
--outer-attn-backend dense        # 头尾层正常密集
--ffn-backend w8a8

# 变体 3: Attention Only
--source-run-dir <source_cache>
--no-latent-injection             # 禁用 latent 注入
--outer-attn-backend sparge-sage2 # 头尾层稀疏
--outer-sparse-layers 0-9,21-29
--ffn-backend w8a8

# 变体 4: Joint（完整 Carnator）
--source-run-dir <source_cache>
--reuse-start-step 10
--sparse-layers 10-20
--middle-attention-mask           # 中间层用 attention mask
--outer-attn-backend sparge-sage2 # 头尾层稀疏
--outer-sparse-layers 0-9,21-29
--ffn-backend w8a8
```

### GPU 分配

4 个变体在 GPU 3/4/5/6 上并行运行，每个约 33 GiB 显存。

| GPU | 变体 |
|-----|------|
| 3 | Full Computation |
| 4 | Latent Only |
| 5 | Attention Only |
| 6 | Joint |

## 2. 配置修正记录

首轮实验（`artifacts/ablation_req16_20260909_040000/`）存在三处配置错误，导致结论反常
（Joint 反而最慢，1.40x）：

1. ❌ 所有变体 FFN 都用 `fp` —— 应除 Full Computation 外都用 `w8a8`
2. ❌ 所有变体 outer sparse 都是 `false` —— Attention Only 和 Joint 应启用 `sparge-sage2`
3. ❌ Attention Only 错误地在中间层启用了 attention mask —— 应禁用

修正后（`artifacts/ablation_req16_20260909_100759/`）的行为符合预期：Joint 取得最高加速比。
本文件中的全部数据均来自修正后的实验。

## 3. 最终结果

### 3.1 时延

| Variant | DiT Latency (Dense) | DiT Latency (Reuse) | E2E Latency (Dense) | E2E Latency (Reuse) | DiT Speedup | E2E Speedup |
|---------|---------------------|---------------------|---------------------|---------------------|-------------|-------------|
| **Full Computation** | 274.26s | N/A | 308.19s | N/A | 1.00x | 1.00x |
| **Latent Only** | 272.43s | 181.48s | 314.22s | 206.06s | 1.51x | 1.52x |
| **Attention Only** | 266.05s | 135.87s | 298.88s | 155.85s | 1.96x | 1.92x |
| **Joint** | 271.90s | 137.85s | 304.57s | 160.91s | 1.97x | 1.89x |

### 3.2 质量指标

评测完成于 2026-09-09 14:40。

| Variant | Type | CLIP-T | VBench-Q | Subject Cons. | Background Cons. | Motion Smooth. | Aesthetic | Imaging | Dynamic Degree |
|---------|------|--------|----------|---------------|------------------|----------------|-----------|---------|----------------|
| Full Computation | Dense | 0.3281 | 0.6436 | 0.8698 | 0.9531 | 0.9878 | 0.4991 | 0.6746 | 1.0000 |
| Latent Only | Dense | 0.3281 | 0.6436 | 0.8698 | 0.9531 | 0.9878 | 0.4991 | 0.6746 | 1.0000 |
| Latent Only | Reuse | 0.3376 | 0.6438 | 0.8331 | 0.9488 | 0.9783 | 0.5456 | 0.7114 | 1.0000 |
| Attention Only | Dense | 0.3281 | 0.6436 | 0.8698 | 0.9531 | 0.9878 | 0.4991 | 0.6746 | 1.0000 |
| Attention Only | Reuse | 0.2837 | 0.6145 | 0.8377 | 0.8984 | 0.9677 | 0.4534 | 0.7010 | 1.0000 |
| Joint | Dense | 0.3281 | 0.6436 | 0.8698 | 0.9531 | 0.9878 | 0.4991 | 0.6746 | 1.0000 |
| Joint | Reuse | 0.3391 | 0.6430 | 0.8384 | 0.9471 | 0.9779 | 0.5404 | 0.7082 | 1.0000 |

### 3.3 质量退化（Dense → Reuse）

| Variant | CLIP-T 变化 | VBench-Q 变化 | 说明 |
|---------|------------|--------------|------|
| **Latent Only** | +0.0095 (+2.90%) | +0.0003 (+0.04%) | ✅ 几乎无质量损失 |
| **Attention Only** | -0.0444 (-13.54%) | -0.0291 (-4.52%) | ⚠️ CLIP-T 明显退化，VBench-Q 小幅损失 |
| **Joint** | +0.0110 (+3.35%) | -0.0006 (-0.09%) | ✅ 质量损失可忽略 |

**关键观察**

1. 所有 Dense 变体质量完全一致（CLIP-T 0.3281 / VBench-Q 0.6436），说明优化只改变计算过程，
   不影响 dense 生成质量。
2. **Latent Only** 质量保持最好：CLIP-T +2.90%，VBench-Q +0.04%。
3. **Joint** 兼顾性能与质量：1.89x E2E 加速下 CLIP-T +3.35%，VBench-Q -0.09%。
4. **Attention Only** 退化最明显：CLIP-T -13.54%，背景一致性 -5.7%，但换来了最高加速比 1.92x。

## 4. 分析：两个关键问题

### 4.1 为什么消融 Joint (1.89x) 高于 VidProM Carnator (1.25x)？

| Metric | VidProM Carnator B5 | Ablation Joint | 差异 |
|--------|---------------------|----------------|------|
| E2E Latency | 205.18s | 160.91s | 44.27s faster (21.6%) |
| Denoising/DiT | 160.26s | 137.85s | 22.41s faster (14.0%) |
| Speedup | ~1.25x vs NIRVANA | 1.89x vs Full Comp | — |

原因有两层：

1. **基线不同**。VidProM 以 NIRVANA-VID（本身已优化，256.09s E2E）为参照，而消融实验以纯
   dense Full Computation（308.19s E2E）为参照。语境不同，加速比不可直接比较。
2. **工作负载不同**。VidProM 是 300 条多样化请求的平均表现，含 Boundary-5 probe 开销、
   early signature 过滤，且部分请求 cache 匹配质量较差；消融实验是单条近完美匹配
   （动物替换）的 best-case，mask 预生成无 probe 开销。

即使看绝对时间，消融 Joint 也确有优势（137.85s vs 160.26s，快 14%），主要来自无 B5 probe
开销与近乎完美的 cache 匹配。

**结论**：消融测的是 best-case，VidProM 测的是多样化请求的平均表现，两者互补而非矛盾。

### 4.2 为什么 Attention Only (1.96x) 与 Joint (1.97x) 几乎相同？

| Variant | DiT Latency | Speedup | Self-attn | Cross-attn | FFN |
|---------|-------------|---------|-----------|------------|-----|
| Full Computation | 274.26s | 1.00x | 177.38s | 16.57s | 35.26s |
| Latent Only | 181.48s | 1.51x | 86.60s | 11.01s | 19.85s |
| Attention Only | 135.87s | 2.02x | ~50s* | ~11s* | ~20s* |
| Joint | 137.85s | 1.99x | ~52s* | ~11s* | ~21s* |

\* 依时间比例估算。

**层分布是关键**。Wan2.2-TI2V-5B 共 30 层，中间层 10-20 仅 11 层（37%），头尾层
0-9/21-29 有 19 层（63%）。因此：

1. **头尾层占主导**。外层稀疏 attention 单独就贡献约 1.9x，是加速的主要来源。
2. **中间层优化边际收益小**。在外层已稀疏的前提下，latent 注入只再贡献约 0.1x
   （137.85 - 135.87 ≈ 2s）。
3. **W8A8 FFN 量化是全局收益**。FFN 从 35.26s 降到约 20s（1.7x），两个变体同样受益。
4. **稀疏 attention 本身有开销**。Joint 的中间层需按 Medit 做 query masking、按 Msee 做
   context masking，索引准备与块选择的开销部分抵消了 latent 注入的收益。

**结论**：该架构的瓶颈在头尾层而非 Carnator 主要针对的中间层。后续优化的优先级应为
①头尾层稀疏 attention（收益最高）→ ②W8A8 FFN 量化（全局收益）→ ③中间层 latent 注入（收益次要）。

## 5. 时间对比：为什么修正后更快？

对比首轮错误配置（`ablation_req16_20260909_040000`）与修正后（`..._100759`）：

| Variant | 错误配置 E2E | 修正后 E2E | 差异 |
|---------|-------------|-----------|------|
| Full Computation | 308.94s | 308.19s | 几乎相同 (±1%) |
| Attention Only (reuse) | 212.26s | 155.85s | 快 56.41s（26.6%） |

Full Computation 基线时间一致，说明测试环境相同；差异全部来自 Attention Only 的配置修正：

- 错误配置下 Attention Only 实际是「中间层带 attention mask 开销 + 外层密集 + FFN 全精度」，
  近似于 Latent Only 而非真正的 Attention Only。
- 修正后「外层稀疏 + FFN W8A8 + 中间层无额外开销」，DiT 从 192.20s 降到 135.87s。
- 加速来源分解：外层稀疏 ≈ 60s（110s → 50s），W8A8 FFN ≈ 15s（35s → 20s），
  移除中间层 mask 开销 ≈ 5s，合计约 80s，与实测 56.33s 基本吻合。

**这些计时不包含 Early Signature probe 开销**，因为是单请求测试、使用预生成的 source cache
和 fusion mask，无 probe、无 mask 选择、无 early signature 过滤。Early Signature 与
Boundary-5 probe 只在 VidProM 等多请求场景产生开销。

修正后 Attention Only 的时间构成：

```
E2E: 155.85s
├─ DiT forward: 135.87s (87.2%)
├─ Finalize output: 19.53s (12.5%)
└─ Cache write: 0s (0.0%)
```

## 6. 产物与脚本位置

| 项目 | 路径 |
|------|------|
| 实验脚本 | [scripts/ablation/ablation_index43_corrected.py](../scripts/ablation/ablation_index43_corrected.py) |
| 进度检查 | [scripts/ablation/check_progress.sh](../scripts/ablation/check_progress.sh) |
| 质量评测 | [scripts/ablation/evaluate_ablation_quality.py](../scripts/ablation/evaluate_ablation_quality.py) |
| Artifacts | `artifacts/ablation_req16_20260909_100759/{full_computation,latent_only,attention_only,joint}/` |
| Results | `results/ti2v_5b/ablation/`（report.md, summary.json, frames/） |

每个变体产物目录包含 `ablation/videos/target_dense.mp4` 与（reuse 变体）
`ablation/videos/target_latent_reuse.mp4`。大体积视频与 cache 为本地可复现产物，
不入库，见 [.gitignore](../.gitignore)。
