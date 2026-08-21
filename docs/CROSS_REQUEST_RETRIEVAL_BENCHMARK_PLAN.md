# 跨请求检索对比方案

在同一份 `datasets/vbench_cross_request_prompts.csv` 上对 NIRVANA-VID、CHAI、Chorus、Carnator
进行 retrieval-only 对比。实验只比较跨请求检索，不执行命中后的 Wan 推理、不加载 latent、不生成视频。

## 请求回放

694 条 prompt 以固定 seed `20260821` 全局打乱后，按该顺序逐条到达。四种方法均从空的独立 cache 开始；当前请求先查询历史记录，只有 miss 后才将它作为新的 source record 写入自己的 cache。四种方法不共享 cache、向量索引或召回结果。

Carnator 的第二阶段需要真实 Early Signature。预先对数据集每条 prompt 运行原生 Wan 的 completed step-5 probe，生成按 UUID 对齐的 ETS JSONL；该 probe 不 decode 视频，也不运行后续 target 推理。
回放时，Carnator 使用同 prompt、同 `seed + request_index` 的 ETS；缺失或 prompt/seed 不一致时直接失败，不使用 fixture。

## 记录内容

### Hit Rate 变化

每处理 50 条请求记录一次该组的 `hits` 和 `hit_rate`；最后一组记录剩余请求。输出一张包含四条线的
对比图：横轴为已处理请求数，纵轴为该 50 条内的 Hit Rate。同时保存原始 CSV，字段为：

```text
method, group_start, end_request, requests_in_group, hits, hit_rate
```

### 相同 Prompt 的最终选择

按输入 query 对齐输出一张 CSV。每行记录同一个 prompt 在四种方法下最终选中的历史 source prompt：

```text
request_index, uuid, prompt,
<method>_hit, <method>_source_uuid, <method>_source_prompt, <method>_score
```

miss 的 source 字段为空。

### 检索耗时

每种方法统计：

```text
requests, hits, retrieval_total_s, retrieval_mean_ms
```

计时包含单条 query 的文本/实体编码、向量查询和方法决策；不包含 encoder 首次加载、source record
编码/写入、ETS 离线采集以及任何模型推理时间。

### Cache 峰值

记录每种方法的 `peak_cache_records` 和 `final_cache_records`。由于本实验不保存真实 latent、trajectory
或 attention tensor，逻辑 record 数是唯一可比的 cache 峰值；不将其解释为 GPU 或磁盘字节数。

## 输出

一个 run 目录包含：

- `arrival_order.csv`：固定打乱后的到达顺序。
- `<method>/results.jsonl` 与 `<method>/summary.json`：逐请求结果及方法汇总。
- `selection_comparison.csv`：四种方法对相同 prompt 的最终选择。
- `hit_rate_by_group.csv` 与 `hit_rate_by_group.svg`：每 50 条的 Hit Rate 对比。
- `manifest.json`：workload、seed、方法配置和输出路径。

实现入口为 `scripts/run/cross_request_dataset.py`；Carnator ETS 的离线采集入口为
`scripts/run/carnator_dataset_ets.py`。
