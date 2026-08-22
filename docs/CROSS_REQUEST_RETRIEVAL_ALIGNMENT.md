# 跨请求检索对齐

同一份 694 条 VBench prompt 的 retrieval-only 对比；不执行命中后的模型推理。

| 方法 | 检索 | 命中后的复用 |
| --- | --- | --- |
| NIRVANA-VID | 完整 prompt 的 CLIP embedding，ChromaDB Top-1 cosine，阈值 `0.65` | 固定使用第 10 步 latent，走 restart 路径 |
| CHAI | 清洗 `aesthetic score` 后缀，仅提取 spaCy `NOUN` token，LongCLIP 编码，归一化向量的 FAISS 内积 Top-1；搜索阈值 `0.90`，写入去重阈值 `0.95` | 通过 Cache Attention 复用实体相关的缓存信息 |
| Chorus | 与 NIRVANA 共用完整 prompt 的 CLIP/ChromaDB 检索和阈值 | 使用匹配 trajectory 和 source video，按分数计算 `K1/K2` |
| Carnator | 与 NIRVANA/Chorus 共享完整 prompt CLIP/Chroma 排名与 `0.65` 下限；Top-20 后按 spaCy `NOUN+PROPN` 过滤，最多 Top-5 做 step-5 ETS | 只有通过 ETS 与 artifact 校验才复用 |

NIRVANA 和 Chorus 的 Top-1 应逐请求一致；Carnator 复用同一完整排序，但消费 Top-20。CHAI 保持实体级检索和写入去重。

固定第 10 步和阈值 `0.65` 仅用于受控对比。Chorus 调度是工程近似；当前 VBench 结果不能与论文 VidProM 结果直接比较。Carnator 阶段 2 是 prompt ETS `ets-v1`，不是 `early_signature` V3。
