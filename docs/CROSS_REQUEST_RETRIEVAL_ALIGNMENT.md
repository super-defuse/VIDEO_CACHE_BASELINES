# 跨请求检索对齐

## 范围

实验使用 `datasets/vbench_cross_request_prompts.csv` 的 694 条 VBench prompt，比较
NIRVANA-VID、CHAI、Chorus 和 Carnator 的跨请求检索。当前实验只记录检索结果，不执行命中后的模型推理。

## 方法

| 方法 | 检索 | 命中后的复用 |
| --- | --- | --- |
| NIRVANA-VID | 完整 prompt 的 CLIP embedding，ChromaDB Top-1 cosine，阈值 `0.65` | 固定使用第 10 步 latent，走 restart 路径 |
| CHAI | 清洗 `aesthetic score` 后缀，仅提取 spaCy `NOUN` token，LongCLIP 编码，归一化向量的 FAISS 内积 Top-1；搜索阈值 `0.90`，写入去重阈值 `0.95` | 通过 Cache Attention 复用实体相关的缓存信息 |
| Chorus | 与 NIRVANA 共用完整 prompt 的 CLIP/ChromaDB 检索和阈值 | 使用匹配 trajectory 和 source video，按分数计算 `K1/K2` |
| Carnator | CLIP Top-K 召回，再做 noun 过滤和 step-5 ETS 校验 | 只有通过 ETS 与 artifact 校验才复用 |

NIRVANA 和 Chorus 的检索结果应逐请求一致；差异只来自命中后的复用方式。CHAI 保持官方仓库中的实体级检索和写入去重，Carnator 保持自己的 ETS 二阶段校验。

## 实现位置

- 共享 prompt 检索：`core/retrieval.py`
- CHAI 检索：`core/retrieval.py`
- Carnator 检索：`carnator/retrieval.py`
- 数据集回放：`scripts/run/cross_request_dataset.py`
- 当前四方法结果：`artifacts/cross_request_retrieval/20260821/runs/benchmark/`
- 协议测试：`tests/test_cross_request_retrieval.py`

## 说明

- 固定第 10 步和阈值 `0.65` 是为了与 Chorus 做受控对比，不是原始 NIRVANA 的分级 CacheSelector profile。
- Chorus 的 `K1/K2` 调度是论文信息不足时采用的工程近似。
- 当前 workload 是 VBench，不是 Chorus 论文使用的 VidProM，因此结果不能直接与论文命中率比较。
