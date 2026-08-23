# Chorus 适配计划

## 目标与范围

将 Chorus 的三阶段逻辑适配到本仓库现有模型：

- **优先实现：Wan2.2-TI2V-5B**（原生 hook 和完整验收）。
- **CogVideoX1.5：暂标 `N/A`**，待 Wan2.2 路径稳定后再评估适配成本。
- 不实现 CLIP/Chroma 语义检索；直接使用现有 source/target prompt 测试协议：
  `a dog is sleeping on a grassland, focus on the dog` ->
  `a cat is sleeping on a grassland, focus on the cat`，seed `20260218`。

这是一项面向 Wan2.2 的工程适配，不声称复现论文的 Wan2.1-T2V 实验。

## 固定依赖

| 组件 | 选择 |
| --- | --- |
| Wan2.2 配置 | 现有 `configs/models/wan22.yaml`：121 帧、1280x704、50-step dense、CFG 5、shift 5、UniPC；Chorus 的 target 与其他基线使用相同配置。 |
| LLM | [`qwen3.5-plus-2026-02-15`](https://www.dmxapi.cn/rmb)，通过 DMXAPI 的 OpenAI-compatible `/v1/chat/completions` 调用。使用 `temperature=0` 和 JSON 输出，返回实体差异与 token 差异；记录模型 ID 和 API 请求耗时。 |
| 视频分割 | 官方 [Grounded-SAM-2 v1.0](https://github.com/IDEA-Research/Grounded-SAM-2/releases/tag/v1.0)，固定 commit `dd4c514`。下载 `sam2.1_hiera_large.pt` + `configs/sam2.1/sam2.1_hiera_l.yaml`，以及本地 `groundingdino_swint_ogc.pth` + `GroundingDINO_SwinT_OGC.py`。使用 Grounding DINO 文本框 -> SAM 2.1 mask -> 视频传播；不依赖 Grounding DINO 1.5/1.6 云 API。 |
| 评测 | 沿用 `docs/EVALUATION.md` 和 `docs/ACCEPTANCE.md`：CLIP-T、VBench-Q、target E2E latency、hit latency、speedup、executed steps、峰值显存和 cache bytes。 |

Grounded-SAM-2 选择 Hiera-Large 是为了使用官方 v1.0 视频 tracking 示例；其 checkpoint 较大，若显存或预处理时延不可接受，再以 Hiera-Base+ 做单独消融，不替换默认配置。

## 实现步骤

1. 使用锁定的 `external/grounded-sam2` v1.0（commit `dd4c514`）及与当前 PyTorch ABI 匹配的 GroundingDINO CUDA 扩展；加载器优先使用这份 checkout，不使用工作站上其他可编辑源码。固定使用 `sam2.1_hiera_large.pt`、Grounding DINO `groundingdino_swint_ogc.pth`，不使用其他 mask 方法。
2. source 原生运行 50 步并缓存完整 latent trajectory；target 固定读取 source，不经过向量检索。LLM 使用 DMXAPI `qwen3.5-plus-2026-02-15` 提取实体差分。
3. target 使用三阶段 50-step schedule：`K1=10`、`K2=40`（工程取值，论文未披露数值）。Stage 1 从 source 完成第 10 步的 latent 恢复，Stage 2 在 10--39 步执行真正的 SRD（`Medit` token 重算，`Msee` token 提供 self-attention K/V，cross-attention/FFN 仅处理 `Medit`），同时启用 TGAA 和 Stage-2 latent fusion；Stage 2 保留 native Wan head 的 dense 输出，避免非编辑 token 的 scheduler derivative 被置零。Stage 3 在 40--49 步执行 dense full computation，关闭 TGAA、SRD 和 latent reuse，并在 step 40 清空 UniPC derivative history、保留全局 timestep index，以修复区域复用产生的 discontinuity。论文未公开 `F_k(t,m)`/`F_o(t,m)` 的具体形式，当前保留 step-only linear decay，并在运行元数据中标记为 approximation；TGAA 配置覆盖全部 Wan blocks。差异 token 无法映射时直接报错，不使用 token-prefix fallback。

   ```text
   Stage 1: 复用 source 的早期 latent
   Stage 2: LLM 找出 dog -> cat 的差异
            Grounded-SAM-2 生成关键帧 mask，SAM2 传播到视频帧
            max-pool 到 latent，膨胀得到 Mbase/Medit/Msee
            TGAA 放大差异 token；SRD 只计算 Medit/Msee 对应区域
            用同一步 source latent 保留 Medit 之外的区域
   Stage 3: 关闭 TGAA/SRD/latent reuse，重置 UniPC history 后 dense 细化剩余步骤以修复 discontinuity
   ```

   阶段边界、mask 膨胀半径和 token 计算比例放入 YAML；先以少量固定值跑通，再做消融。所有 hook 必须在 `finally` 中清理，关闭 Chorus 时回退到同一 Wan2.2 dense adapter。
4. 已将 `chorus` 注册到 CLI 和方法矩阵：`wan22: supported`；`cogvideox15: N/A`。LLM 模型、SAM 版本配置和实际 mask 后端写入运行配置/结果。

## 验收

- 使用上述 dog -> cat、seed、分辨率、帧数和 GPU，与现有 Wan2.2 dense baseline 对比；不引入 VidProM、CLIP 检索或论文的 1K warm-cache 流程。
- 分别报告 source 预处理和 target 请求；target 表格使用现有基线的 E2E、hit latency 与 speedup 口径。
- 质量评测接入现有 VBench 流程，报告 CLIP-T、VBench-Q；同时记录 LLM、Grounded-SAM-2、mask 投影和 sparse kernel 的耗时。
- 测试覆盖：source trajectory/cache 读写、JSON 输出校验、Grounded-SAM-2 mask 传播、TGAA token 映射失败即报错、Stage-2 TGAA/SRD/fusion 门控、Stage-3 dense/no-reuse 回归和 hook 清理。
- 在 Wan2.2 真实验收完成前，CogVideoX1.5 所有 Chorus 指标保持 `N/A`，不得用其他方法结果代填。

## 结果归档

实跑结果、质量指标、计时口径和产物路径统一记录在
[ACCEPTANCE.md](ACCEPTANCE.md#结果汇总)。本文件只保留复现目标、固定依赖和实现约束，
避免与验收记录重复。
