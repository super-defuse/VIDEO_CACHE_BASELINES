# Evaluation

每条请求的 `results.jsonl` 记录 model、method、seed、cache match/source、executed steps、端到端 latency、output hash、quality input path、cache bytes 和 error。汇总只读取这些记录，不手填均值。

性能至少报告 latency、命中 latency、executed steps、Hit Rate、Speedup、`Speedup(hit)`、峰值显存和 cache bytes。质量环境固定为共享 conda 环境 `vbench`；需要时可用 `VCB_QUALITY_ENV` 覆盖名称。

baseline 和方法必须使用同一 prompt、resolution、frames、seed 和 GPU。方法关闭 hook 时必须回退到同一 adapter 的 dense 路径；不兼容组合记录 `N/A`。
