# 跨请求检索基准

在 `datasets/vbench_cross_request_prompts.csv` 的 694 条请求上，以固定 shuffle seed `20260821`
回放 NIRVANA-VID、CHAI、Chorus 和 Carnator 的独立逻辑 cache。该基准只测检索授权，不执行
Wan 推理或命中后的复用。

唯一入口是：

```bash
PYTHONPATH=. python scripts/benchmark/cross_request_retrieval.py \
  --online-signatures /path/to/carnator-sidecars.jsonl
```

不使用 run ID。每次执行覆盖 `artifacts/cross_request_retrieval/` 下由该脚本管理的
`results/`、`indices/`、`reports/` 和 `benchmark.json`；sidecar 输入可以放在其他位置。
输出固定为：

- `reports/hit_rate_over_time.csv`：随着 request 到来变化的累计 hit rate；
- `reports/request_matches.csv`：每个 request 的四方法命中状态、source UUID/prompt 与分数；
- `reports/retrieval_summary.csv`：每个方法的总/平均检索时间、hit rate 和最终 cache 大小；
- `reports/cache_growth.csv`：每次 request 后的逻辑 cache records；
- `results/<method>.jsonl`：完整检索审计记录。

Carnator 需要为 workload 中每个 UUID 提供可读的对象级 signature sidecar。临时检查回退行为时，
可显式加 `--allow-missing-carnator-signatures`；该模式的 Carnator 零命中不是方法效果评估。
在线 Early Probe 的最终定义见 [CARNATOR_ONLINE_RETRIEVAL_PLAN.md](CARNATOR_ONLINE_RETRIEVAL_PLAN.md)。
