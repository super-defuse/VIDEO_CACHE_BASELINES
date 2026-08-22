# Carnator 跨请求检索

范围：Wan2.2-TI2V-5B 的原生 Carnator 路径；不改变采样、稀疏 attention 或量化。

## 流程

1. 与 NIRVANA/Chorus 共享完整 prompt 的 CLIP/Chroma 精确 cosine 排名，保留 `score >= 0.65` 的前 20 条。
2. 使用 spaCy `NOUN+PROPN` 的 `trf_nouns` 过滤：noun set 相同，或双方各仅一个不同 noun。
3. 过滤后的前 5 条按 CLIP 顺序进行 step-5 prompt ETS 校验；第一个同时通过 ETS、冻结阈值和 artifact 校验的 source 获选。

不足 20 条时直接使用已有候选；没有通过 noun 过滤的候选时不执行 probe。CLIP 只用于候选排序，不能授权复用。

## ETS 范围

阶段 2 为本仓库的 prompt-level `ets-v1`：压缩 `Mbase/Medit/Msee` 并池化 `score`。它不是
`early_signature` V3 的对象级 mask、背景/tail-frame 和质量校准实现；后者需要独立采集与阈值验证。

任何签名、阈值或 artifact 校验失败均为 `No Reuse`，不得注入 source latent 或 attention。

## 数据与验收

`datasets/carnator_vbench_trf_nouns.csv` 由本地 spaCy 标注工具生成，并记录 SHA-256 provenance。实体集变化时只同步 ETS metadata；retrieval-only 重跑不重新采集原生 mask。

`results.jsonl.cache_match` 记录 Top-20、noun 过滤、最多 5 条 ETS 候选和最终结果。验收覆盖共享排序、Top-20/Top-5、实体过滤及 fail-closed 分支。
