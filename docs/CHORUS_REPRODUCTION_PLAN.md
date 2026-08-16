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

1. 已在 `wan` 环境安装 Grounded-SAM-2 v1.0（commit `dd4c514`）及 CUDA 扩展；固定使用 `sam2.1_hiera_large.pt`、Grounding DINO `groundingdino_swint_ogc.pth`，不使用其他 mask 方法。
2. source 原生运行 50 步并缓存完整 latent trajectory；target 固定读取 source，不经过向量检索。LLM 使用 DMXAPI `qwen3.5-plus-2026-02-15` 提取实体差分。
3. target 使用三阶段 50-step schedule：`K1=10`、`K2=40`（工程取值，论文未披露数值）。Stage 1 从 source 完成第 10 步的 latent 恢复，Stage 2 在 10--39 步执行真正的 SRD（`Medit` token 重算，`Msee` token 提供 self-attention K/V，cross-attention/FFN 仅处理 `Medit`），Stage 3 在 40--49 步恢复 dense 计算。自 `K1` 起每一步都按同一步 source latent 做 `Medit`/非 `Medit` 区域融合，SRD 只在 Stage 2 生效；无几何 mask fallback。

   ```text
   Stage 1: 复用 source 的早期 latent
   Stage 2: LLM 找出 dog -> cat 的差异
            Grounded-SAM-2 生成关键帧 mask，SAM2 传播到视频帧
            max-pool 到 latent，膨胀得到 Mbase/Medit/Msee
            TGAA 放大差异 token；SRD 只计算 Medit/Msee 对应区域
            用同一步 source latent 保留 Medit 之外的区域
   Stage 3: 关闭 SRD，dense 细化剩余步骤，同时继续保留 Medit 之外的 source latent
   ```

   阶段边界、mask 膨胀半径和 token 计算比例放入 YAML；先以少量固定值跑通，再做消融。所有 hook 必须在 `finally` 中清理，关闭 Chorus 时回退到同一 Wan2.2 dense adapter。
4. 已将 `chorus` 注册到 CLI 和方法矩阵：`wan22: supported`；`cogvideox15: N/A`。LLM 模型、SAM 版本配置和实际 mask 后端写入运行配置/结果。

## 验收

- 使用上述 dog -> cat、seed、分辨率、帧数和 GPU，与现有 Wan2.2 dense baseline 对比；不引入 VidProM、CLIP 检索或论文的 1K warm-cache 流程。
- 分别报告 source 预处理和 target 请求；target 表格使用现有基线的 E2E、hit latency 与 speedup 口径。
- 质量评测接入现有 VBench 流程，报告 CLIP-T、VBench-Q；同时记录 LLM、Grounded-SAM-2、mask 投影和 sparse kernel 的耗时。
- 测试覆盖：source trajectory/cache 读写、JSON 输出校验、Grounded-SAM-2 mask 传播、TGAA token 映射、hook 清理、fingerprint 不匹配回退，以及关闭 Chorus 的 dense 回归。
- 在 Wan2.2 真实验收完成前，CogVideoX1.5 所有 Chorus 指标保持 `N/A`，不得用其他方法结果代填。

## 当前实跑（2026-08-16）

- 当前产物：`artifacts/runs/wan22-ti2v-5b/dog-to-cat/chorus/`，摘要：
  `artifacts/runs/wan22-ti2v-5b/dog-to-cat/chorus/summary.json`。source 50 步 `294.19s`；target
  使用同一 source cache 重新生成 40 步修复版，命中 `source_prompt`，cache `1.07GB`。从 `K1=10`
  到最终 step 都执行 source latent fusion，SRD 仅在 Stage 2 (`10 <= step < 40`) 生效；旧版
  `chorus-wan22-final` 产物已删除。
- fixed target 是 target-only replacement，未单独输出新的 Runner timing；摘要中的 `178.27s`、
  `generation_s=227.83s` 保留为同配置完整 source/target 运行参考，并在 `summary.json` 的
  provenance 字段中标注。
- cache 已与 Carnator 对齐：manifest 只保存索引，source trajectory 以 `latents_initial.pt` + `latents_step_000.pt`--`latents_step_049.pt` 逐步读取；target 的逐步读取包含在 `generation_s=227.83s` 内，不做预加载或从计时中扣除。
- LLM：DMXAPI `qwen3.5-plus-2026-02-15`，请求耗时 `25.729s`；Grounded-SAM-2 检出 1 个 `dog` 框并传播 121 帧，耗时 `26.769s`。两者合计 `52.499s`，作为预处理单独记录并从 target 报告时延剔除；其余 cache lookup/load、生成、视频保存和 hash 均正常计时。
- SRD：`Medit=53.24%`、`Msee=84.06%`；self-attention 使用 `Medit` queries/`Msee` keys-values，cross-attention 与 FFN 只计算 `Medit`，Stage 3 恢复 dense。
- 以现有 50-step dense target `300.64s` 为分母，target speedup `1.69x`（若把 LLM+SAM 也计入则为 `1.30x`）。本轮未重新执行 VBench，因此质量指标不填旧结果。
- CogVideoX1.5：保持 `N/A`。
