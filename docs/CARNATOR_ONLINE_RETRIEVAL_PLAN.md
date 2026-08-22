# Carnator Online Retrieval Plan

状态：v1 codec、线上决策入口和同 noun set 的临时距离阈值已就绪；单 noun substitution 的 mask 阈值留待正式标注集校准。

## 目标

只回答当前 request 是否可以复用一个 historical cache。决策只有 `Reuse` 和 `No Reuse`。`Reuse` 后走固定的 step-10 latent 与 sparse-attention joint reuse；没有通过任何 gate 时继续 target 的 dense generation。

## 在线流程

1. 新 request 用完整 prompt 做 CLIP recall，保留最多 20 个且分数达到下限的历史 cache。
2. 比较 canonical noun set。只保留 noun set 相同、或恰好一个 noun substitution 的候选，最多 5 条，保持 CLIP 排名。
3. 没有候选时，不执行 Early Probe，直接 dense generation。
4. 有候选时，target 只运行到 completed step 5，采集一次 target Early Signature。
5. 顺序读取候选已持久化的 Early Signature，逐个比较；第一个通过的 cache 被选中，其余候选不再比较。
6. 命中时从 source 的 step-10 artifact 执行 joint reuse。全部拒绝时，target 继续 dense generation；该完整生成的 target 生成后成为未来可召回的 source cache。

候选 signature 在其原始 request 成为 source cache 时采集一次，当前 request 绝不重新生成候选的 signature。已命中并复用的 target 没有独立的 step-10 source artifact，不作为新的可复用 cache entry。

## Early Probe 保存内容

一个 signature 绑定固定的 model、scheduler、CFG、completed step、latent token grid、noun extractor 和 mask policy。签名包含：

- 全局 `latent_feature`：第 5 步的整个 latent state 的紧凑、归一化特征；不保存完整 latent。
- 每个 canonical `NOUN/PROPN` 的独立 binary `Mbase`，以及从该 mask 导出的 `area_frac` 和 `centroid_yx`。
- 对 source cache：指向 step-10 latent 与 sparse-attention artifact 的路径、内容 hash 和兼容性元数据。

不保存 VAE decoded frame，也不使用颜色。`Medit` 和 `Msee` 是命中后 joint reuse 的执行数据，不参与 retrieval gate。

`latent_feature` 固定为 `latent-residual-channel-rms-multiscale-pool/v1`：完整 step-5 latent
相对同 seed 初始 latent 的残差，经 channel RMS 和两档固定池化（`[4,6,10]`、`[8,11,20]`）后
L2 归一化。sidecar 只保存约 192 KiB 的 `float16` feature、Mbase 和兼容性/hash，不保存完整 latent；比较使用 cosine distance。

probe 必须沿用完整 50-step scheduler，只在第 5 个 completed step 采集，不得将 scheduler 的
总步数改为 5。线上入口 `carnator.online_flow.resolve_online_request` 只在 Stage 1 有合格候选时采集一次 target sidecar。相同 noun set 先使用临时 `latent_feature` 距离阈值 `0.15`；正式验证可替换该配置。单 noun substitution 的面积和形状阈值尚未配置，必须 `No Reuse`。

## Signature 比较

先要求 model、scheduler、CFG、probe seed、completed step、latent grid、feature codec 和 mask policy 完全一致。noun set 不相同且不是单一替换时直接拒绝。

- noun set 完全相同：比较整个 `latent_feature` 的距离；临时阈值为 `0.15`。
- 恰好一个 noun substitution：同样比较整个 `latent_feature`；再比较替换 noun 的 `abs(log(area_q / area_s))` 和面积归一化、重心对齐后的 `1 - Dice`。三项均通过才 signature-compatible。

线上 `Reuse` 还要求 source 的 step-10 latent、sparse-attention artifact 及其 hash 都可读取。任意缺失、metadata mismatch、不可计算的距离或失败 gate 都返回 `No Reuse`。

## 正式测试

正式测试的唯一 workload 是原始的
`datasets/vbench_cross_request_prompts.csv`（694 条）。开始回放前固定执行
`random.Random(20260821).shuffle(rows)`；该顺序是模拟真实 request arrival 的唯一顺序，并写入每条结果的 `arrival_index`。四种方法各自从空逻辑 cache 开始，按同一到达顺序处理；只有 miss 会写入该方法自己的 source cache。

Carnator 在每个 request 到达时，以固定的 `spaCy en_core_web_sm` `NOUN+PROPN` extractor 直接从 prompt 得到 canonical noun set，不消费预生成的 VBench noun CSV。`object_class_65` 的 `a remote` 是 spaCy 已知的 POS 误标，保留为 UUID 绑定的 `remote` 纠正。

## 评测边界

`scripts/benchmark/cross_request_retrieval.py` 比较 NIRVANA-VID、CHAI、Chorus 和 Carnator，并固定产出：累计 hit rate、每 request 的命中及 source、平均检索时间和逻辑 cache 增长。它不读取任何预先收集的 694 条 sidecar。

Carnator 的 collector 与 arrival replay 同步运行：无 Stage-1 候选时直接将该 request 作为 future source 准备到 step 10；有候选时先运行到 completed step 5，并只在 signature gate 拒绝后继续到 step 10。scheduler 始终为完整 50 steps，不执行 VAE decode、也不运行 step 10 之后的去噪。命中的 target 不保留 source sidecar。报告将 Stage-1 retrieval、signature gate、Early Probe 和 source-cache preparation 分别计时，避免将 Carnator 的模型 probe 混为纯检索耗时。
