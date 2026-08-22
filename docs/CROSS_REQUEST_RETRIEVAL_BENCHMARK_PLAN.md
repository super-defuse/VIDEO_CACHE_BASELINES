# 跨请求检索基准计划

在 `datasets/vbench_cross_request_prompts.csv` 的 694 条请求上比较命中率、检索时间和逻辑 cache 规模；只做检索，不执行 Wan 推理。

固定 shuffle seed 为 `20260821`。每种方法使用独立空 cache，miss 才写 source。NIRVANA/Chorus 取共享 CLIP 排名的 Top-1；Carnator 取 Top-20，经 spaCy `NOUN+PROPN` 过滤后最多 5 条进入 ETS。Carnator 需要覆盖全部 UUID 的 step-5 ETS JSONL。

输出位于 `artifacts/cross_request_retrieval/<run-id>/runs/<run-name>/`：逐方法 `results.jsonl` 与 `summary.json`，以及到达顺序、四方法 comparison、分组命中率和 manifest。入口为 `scripts/run/cross_request_dataset.py`；ETS 采集入口为 `scripts/run/carnator_dataset_ets.py`。
