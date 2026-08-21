# 跨请求检索基准计划

## 目标

在同一份 `datasets/vbench_cross_request_prompts.csv` 上比较 NIRVANA-VID、CHAI、Chorus 和 Carnator 的检索命中率、检索耗时和 cache 规模。
实验是 retrieval-only：不执行命中后的 Wan 推理、不加载 latent、不生成视频。

## 回放规则

1. 使用固定 shuffle seed `20260821`，按 694 条 prompt 的到达顺序逐条回放。
2. 四种方法使用独立的空 cache，不共享向量索引或召回结果。
3. 当前请求先查询历史 source；只有 miss 才写入新的 source record。
4. Carnator 需要预先生成覆盖全部 UUID 的 native step-5 ETS JSONL。

## 指标

- `hit_rate = hits / requests`
- `retrieval_mean_ms`：单条 query 的编码、向量查询和方法决策时间，不含模型首次加载。
- `peak_cache_records`、`final_cache_records`：逻辑 source record 数，不代表 GPU 或磁盘字节数。
- NIRVANA 和 Chorus 还应检查逐请求的 hit、source UUID 和 score 是否一致。

## 输出

每次 retrieval-only 运行使用统一目录：

`artifacts/cross_request_retrieval/<run-id>/runs/<run-name>/`

运行命令示例：

```bash
conda run -n wan env PYTHONNOUSERSITE=1 PYTHONPATH=.:external/Long-CLIP \
  python scripts/run/cross_request_dataset.py \
  --methods chai \
  --run-id 20260821 \
  --run-name chai-official-real
```

运行目录包含：

- `arrival_order.csv`：固定到达顺序。
- `<method>/results.jsonl`：逐请求检索结果。
- `<method>/summary.json`：方法汇总。
- `selection_comparison.csv`：同一 prompt 在各方法下的选择。
- `hit_rate_by_group.csv`、`hit_rate_by_group.svg`：每 50 条请求的命中率。
- `manifest.json`：workload、seed、方法和输出路径。

同一日期目录下的其他内容按用途分区：`inputs/` 保存 workload/ETS 输入，`work/` 保存可删除的
中间探针文件，`logs/` 保存运行日志；当前四方法汇总结果固定在
`runs/benchmark/`，每个方法只保留一个 `<method>/results.jsonl`。

## 入口

- 数据集回放：`scripts/run/cross_request_dataset.py`
- Carnator ETS 采集：`scripts/run/carnator_dataset_ets.py`
- 协议测试：`tests/test_cross_request_retrieval.py`
