# 跨请求检索基准

在原始 `datasets/vbench_cross_request_prompts.csv` 的 694 条请求上，以固定 shuffle seed
`20260821` 打乱后按 arrival order 回放 NIRVANA-VID、CHAI、Chorus 和 Carnator 的独立逻辑
cache，模拟真实请求到达。四种方法从空 cache 开始，只有 miss 写入各自的 cache。Carnator 的
source signature 与 step-10 cache 在对应 miss 到达时创建，绝不预先生成全部 694 条 sidecar。

唯一入口是：

```bash
PYTHONPATH=. python scripts/benchmark/cross_request_retrieval.py
```

不使用 run ID。每次执行覆盖 `artifacts/cross_request_retrieval/` 下由该脚本管理的
`results/`、`indices/`、`reports/` 和 `benchmark.json`；sidecar 输入可以放在其他位置。
输出固定为：

- `reports/hit_rate_over_time.csv`：随着 request 到来变化的累计 hit rate；
- `reports/request_matches.csv`：每个 request 的四方法命中状态、source UUID/prompt 与分数；
- `reports/retrieval_summary.csv`：每个方法的总/平均检索时间、hit rate 和最终 cache 大小；
- `reports/cache_growth.csv`：每次 request 后的逻辑 cache records；
- `results/<method>.jsonl`：完整检索审计记录。

Carnator 在 Stage 1 有 noun-compatible historical candidate 时才运行 target Early Probe，到 completed
step 5 写入临时 object-level signature 并执行 gate。无候选的 request 不运行 target probe；它在
成为 miss source 时从相同 50-step scheduler 运行至 step 10，写入未来请求所需的 sidecar 与 source
artifact。命中的 target 会删除临时 probe sidecar，且不进入 cache。测试不解码视频，也不会执行
step 10 后的去噪。在线 Early Probe 的最终定义见
[CARNATOR_ONLINE_RETRIEVAL_PLAN.md](CARNATOR_ONLINE_RETRIEVAL_PLAN.md)。
