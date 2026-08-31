# 跨请求检索

本文件是 NIRVANA-VID、CHAI、Chorus 和 Carnator 的跨请求检索规范、回放基准及
Carnator 在线 gate 的唯一说明。它只测 cache recall、gate 和 source-cache 准备；不执行
命中后的完整视频生成。

## 方法对齐

所有方法在同一份 694 条 VBench prompt 上独立维护逻辑 cache，并以相同 arrival order
回放。

| 方法 | 检索 | 命中后的复用 |
| --- | --- | --- |
| NIRVANA-VID | 完整 prompt 的 CLIP embedding，ChromaDB Top-1 cosine，阈值 `0.65` | 固定使用第 10 步 latent，走 restart 路径 |
| CHAI | 清洗 `aesthetic score` 后缀，仅提取 spaCy `NOUN` token，LongCLIP 编码，归一化向量的 FAISS 内积 Top-1；搜索阈值 `0.90`，写入去重阈值 `0.95` | 通过 Cache Attention 复用实体相关的缓存信息 |
| Chorus | 与 NIRVANA-VID 共用完整 prompt 的 CLIP/ChromaDB 检索和阈值 | 使用匹配 trajectory 和 source video，按分数计算 `K1/K2` |
| Carnator v1 | 完整 prompt CLIP/ChromaDB 排名与 `0.65` 下限；Top-20 后按 spaCy `NOUN` 过滤，最多 Top-5 做对象级 Early Signature | step-5 decoded-tail-frame 的 Lab/CIEDE2000、面积和形状 gate；要求 artifact 校验 |
| Carnator v2 | 先在全部缓存中筛选 noun set 相同或单 noun substitution 的请求，再比较全部 Early Signature；剩余候选按 CLIP 取 Top-1 | 同 v1；任一前置阶段没有候选即 dense |

NIRVANA-VID 和 Chorus 的 Top-1 应逐请求一致；Carnator v1 消费同一完整排序的
Top-20，v2 则把 CLIP 放在 signature gate 之后。CHAI 保持实体级检索和写入去重。固定第 10 步和阈值 `0.65` 仅用于受控对比。
Chorus 调度是工程近似，当前 VBench 结果不能与论文 VidProM 结果直接比较。

## 回放基准

workload 为原始 `datasets/vbench_cross_request_prompts.csv` 的 694 条请求。开始回放前固定执行
`random.Random(20260821).shuffle(rows)`；该顺序是唯一的 request arrival order，并写入每条
结果的 `arrival_index`。四种方法均从空 cache 开始，只有 miss 写入各自的 source cache。
Carnator 的 source signature 与 step-10 cache 仅在对应 miss 到达时创建，绝不预先生成
694 条 sidecar。

唯一入口：

```bash
PYTHONPATH=. python scripts/benchmark/cross_request_retrieval.py
```

不使用 run ID。每次执行覆盖 `artifacts/cross_request_retrieval/` 下由脚本管理的
`results/`、`indices/`、`reports/` 和 `benchmark.json`；sidecar 输入可以位于其他路径。
输出固定为：

- `reports/hit_rate_over_time.csv`：随 request 到来变化的累计 hit rate。
- `reports/request_matches.csv`：每个 request 的四方法命中状态、source UUID/prompt 与分数。
- `reports/retrieval_summary.csv`：每个方法的总/平均检索时间、hit rate 和最终 cache 大小。
- `reports/cache_growth.csv`：每次 request 后的逻辑 cache records。
- `results/<method>.jsonl`：完整检索审计记录。

## Carnator 在线决策

Early Signature 是唯一实现：在 step 5 VAE decode 一次 tail frame，以对象级 CIELAB、面积和
形状 gate 决定是否复用。每个 Mbase 由 noun-specific cross-attention seed 与 self-attention
contrast 融合生成，禁止将 cross-attention mask 单独作为 Mbase。决策只有 `Reuse` 和 `No Reuse`：
命中后使用 source 的 step-10 latent 和融合 Mbase/sparse-attention artifact 执行 joint reuse；
未通过任一 gate 时继续 target 的 dense generation。

v1 的顺序是 CLIP Top-20、noun filter、最多 Top-5 的 Early Signature；v2 的顺序是：

1. 在全部缓存条目中比较 `NOUN` token set，仅保留相同集合或恰好一个 noun substitution 的候选。
2. 没有名词候选时，不运行 Early Probe，直接 dense；否则 target 运行至 completed step 5 并比较所有候选 Early Signature。
3. signature 阶段没有 artifact-ready 候选时继续 dense；否则只在通过 signature 的候选中按 CLIP 文本相似度取 Top-1。
4. 命中时从 source 的 step-10 artifact 执行 joint reuse。

候选 signature 只在其原始 request 成为 source cache 时采集。已命中并复用的 target 没有独立
step-10 source artifact，不作为新的 cache entry。无 noun 的 request fail-closed，不采集对象
signature 或写入 source cache。

### Early Signature

signature 绑定 model、scheduler、CFG、completed step、latent token grid、noun extractor 和
融合 mask policy，并保存每个 `NOUN` token 的 binary `Mbase`。第 5 个 completed step 会先用
cross-attention 定位 seed，再以 self-attention contrast 与 cross score 融合得到 Mbase，随后持久化
decoded tail frame、内容 hash 和每个 noun 的平均 CIELAB。`Mbase` 的 tail token 通过
nearest-neighbor 投影到该帧；source sidecar 还包含 step-10 latent 与 sparse-attention artifact
的路径、内容 hash 和兼容性元数据。`Medit` 和 `Msee` 是命中后的执行数据，不参与 retrieval
gate。

若融合后的 noun-specific `Mbase` 尾部时间切片为空，只复制最近的非空切片并记录元数据；不降低融合阈值。若整个时间轴均为空，probe
fail-closed，放弃 Stage-2 命中并回退完整 dense generation，不写入空 mask。

probe 必须沿用完整 50-step scheduler，只在第 5 个 completed step 采集，不能把总步数改为 5。
线上入口只在 Stage 1 有合格候选时采集一次 target sidecar。必须显式配置
`max_context_color_distance`；单 noun substitution 还要求 `max_log_area_ratio`、
`max_shape_distance` 和 `max_changed_color_distance`，缺失任一阈值即 `No Reuse`。正式阈值
选择流程见 [Carnator Early Signature 标定与统计流程](CARNATOR_EARLY_SIGNATURE_CALIBRATION_PLAN.md)。

### Gate 与 artifact 校验

首先要求 model、scheduler、CFG、probe seed、completed step、latent grid 和 mask
policy 完全一致。noun set 不相同且不是单一替换时直接拒绝。

- 同 noun set 比较共享 noun 的最大 CIEDE2000。
- 单 noun substitution 额外比较 `abs(log(area_q / area_s))`、面积归一化且重心对齐后的
  `1 - Dice`，以及 changed-noun CIEDE2000。

线上 `Reuse` 还要求 source 的 step-10 latent、sparse-attention artifact 及其 hash 都可读取。
任何 metadata mismatch、缺失 artifact、不可计算的距离或失败 gate 都返回 `No Reuse`。

## 执行边界

Carnator 的 collector 与 arrival replay 同步运行。没有 Stage-1 候选时，当前 request 会直接
准备为 future source 至 step 10；有候选时，先运行到 completed step 5，只有 signature gate
拒绝才继续至 step 10。命中的 target 会删除临时 probe sidecar，且不进入 cache。报告分别统计
Stage-1 retrieval、signature gate、Early Probe 和 source-cache preparation 的时间。

Early Signature 仅在 step 5 decode 一次以保存 tail-frame 颜色，不编码视频，也不运行 step 10
后的去噪；因此此基准不能代表命中后的完整视频质量或端到端生成时间。

只评估检索命中效果时，使用：

```bash
PYTHONPATH=. python scripts/benchmark/cross_request_retrieval.py --assume-step10-cache
```

该模式仍按 arrival order 运行到 step 5 并执行候选 gate；miss source 不再运行 step 6--10，
也不写入实际 step-10 latent 或 sparse-attention artifact，只在逻辑 cache 标记 artifact 已存在。
输出 schema 会记录 `step10_cache_assumed=true`，结果不能用于验证实际 latent/sparse-attention
复用。
