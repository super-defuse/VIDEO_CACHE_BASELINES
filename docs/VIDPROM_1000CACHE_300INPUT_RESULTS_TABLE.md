# VidProM 300-request 结果表

实验 seed 固定为 shared cache 使用的 `20260218`。时间单位为秒；端到端时延从请求进入 runner 开始，包含检索、缓存读取和推理直到视频解码完成，忽略视频/缓存写盘和结果哈希。去噪时延只计 DiT Blocks forward。检索仅保留决策元数据，不单列耗时。当前阶段先记录时延、
命中率和质量，speedup 暂填 `TBD`；待同一批 300 条请求的 Dense C0 control 完成后，
再用 `Dense C0 平均时间 / 基线平均时间` 计算。共享 cache preparation 时间不作为分母。
hit rate 只统计跨请求检索命中。TeaCache 没有跨请求检索，相关列填 `N/A`。质量指标在全部
视频生成结束后统一评估。

## 汇总表

| 基线 | GPU | 检索配置 / 版本 | 请求数 | 命中数 | hit rate | 端到端时延 (s/request) | 命中端到端时延 (s/hit) | 去噪时延 (s/request) | 命中去噪时延 (s/hit) | External (s/request) | 端到端加速比 | 去噪加速比 | CLIP-T | VBench-Q | 产物 / 日志 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Dense C0 control | reserved | none | 300 | N/A | N/A | TBD | N/A | TBD | N/A | 0 | 1.00x | 1.00x | TBD | TBD | `results/dense-c0/` |
| NIRVANA-VID | 3 | CLIP/Chroma Top-1, threshold 0.65, restart step 10 | 300 | TBD | TBD | TBD | TBD | TBD | TBD | 0 | TBD | TBD | TBD | TBD | `results/nirvana_vid/` |
| TeaCache | 4 | target-only TeaCache, relative L1 threshold 0.08 | 300 | N/A | N/A | TBD | N/A | TBD | N/A | 0 | TBD | TBD | TBD | TBD | `results/teacache/` |
| Chorus | 5 | CLIP/Chroma Top-1, threshold 0.65, K1/K2 10/30, TGAA | 300 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | `results/chorus/` |
| Chorus2 | 6 | shared NIRVANA/Chorus CLIP/Chroma Top-1, threshold 0.65; SpargeAttn | 300 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | `results/chorus2/` |
| Carnator Boundary B5 | 3, 4, 5, 7 | v2 noun filter + step-5 self/cross-fusion Early Signature + CLIP top-1 (`2.5460641772840114`, `1.6755377126043391`, `1.0`, `2.2003341853972085`); Boundary B5 | 300 | TBD | TBD | TBD | TBD | TBD | 0 | TBD | TBD | TBD | TBD | TBD | `results/carnator_boundary_b5/` |

### 字段定义

| 字段 | 定义 |
| --- | --- |
| External | Chorus 的 LLM prompt diff + Grounded-SAM2 分割耗时；其他基线固定为 `0`。该时间是端到端时延中的独立 breakdown，不重复加到去噪时延 |
| 去噪时延 | 仅 DiT Blocks forward；CUDA 前后同步并剔除缓存保存/写盘，不能使用 shared cache preparation 时间 |
| 命中端到端/去噪时延 | 仅对 `hit=true` 请求取均值；未命中请求填 `null`，避免 dense fallback 混入命中耗时 |
| 端到端/去噪加速比 | 暂填 `TBD`；Dense C0 control 完成后为其平均时延 / 基线平均时延，端到端包含检索和 External，去噪只含 DiT Blocks |
| hit rate | `命中请求数 / 300`；按方法独立检索状态计算。TeaCache 为 N/A |
| CLIP-T | 保留视频按 prompt 的 CLIP text/image cosine，报告 300 条平均值（同时保留逐请求值） |
| VBench-Q | 六维 VBench 聚合分数，按现有 `quality_runner.py` 协议；报告平均值与逐请求值 |

## 分组命中率表

每行对应到达后的累计点；`window_*` 是最近 50 条请求的窗口值，`cumulative_*` 是从第 1
条到当前请求的累计值。TeaCache 行保留为 `N/A`。

| 基线 | 到达请求数 | 窗口请求范围 | window hits | window hit rate | cumulative hits | cumulative hit rate |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| NIRVANA-VID | 50 | 1--50 | TBD | TBD | TBD | TBD |
| NIRVANA-VID | 100 | 51--100 | TBD | TBD | TBD | TBD |
| NIRVANA-VID | 150 | 101--150 | TBD | TBD | TBD | TBD |
| NIRVANA-VID | 200 | 151--200 | TBD | TBD | TBD | TBD |
| NIRVANA-VID | 250 | 201--250 | TBD | TBD | TBD | TBD |
| NIRVANA-VID | 300 | 251--300 | TBD | TBD | TBD | TBD |
| Chorus | 50 | 1--50 | TBD | TBD | TBD | TBD |
| Chorus | 100 | 51--100 | TBD | TBD | TBD | TBD |
| Chorus | 150 | 101--150 | TBD | TBD | TBD | TBD |
| Chorus | 200 | 151--200 | TBD | TBD | TBD | TBD |
| Chorus | 250 | 201--250 | TBD | TBD | TBD | TBD |
| Chorus | 300 | 251--300 | TBD | TBD | TBD | TBD |
| Chorus2 | 50 | 1--50 | TBD | TBD | TBD | TBD |
| Chorus2 | 100 | 51--100 | TBD | TBD | TBD | TBD |
| Chorus2 | 150 | 101--150 | TBD | TBD | TBD | TBD |
| Chorus2 | 200 | 151--200 | TBD | TBD | TBD | TBD |
| Chorus2 | 250 | 201--250 | TBD | TBD | TBD | TBD |
| Chorus2 | 300 | 251--300 | TBD | TBD | TBD | TBD |
| Carnator Boundary B5 | 50 | 1--50 | TBD | TBD | TBD | TBD |
| Carnator Boundary B5 | 100 | 51--100 | TBD | TBD | TBD | TBD |
| Carnator Boundary B5 | 150 | 101--150 | TBD | TBD | TBD | TBD |
| Carnator Boundary B5 | 200 | 151--200 | TBD | TBD | TBD | TBD |
| Carnator Boundary B5 | 250 | 201--250 | TBD | TBD | TBD | TBD |
| Carnator Boundary B5 | 300 | 251--300 | TBD | TBD | TBD | TBD |
| TeaCache | 50 | N/A | N/A | N/A | N/A | N/A |
| TeaCache | 100 | N/A | N/A | N/A | N/A | N/A |
| TeaCache | 150 | N/A | N/A | N/A | N/A | N/A |
| TeaCache | 200 | N/A | N/A | N/A | N/A | N/A |
| TeaCache | 250 | N/A | N/A | N/A | N/A | N/A |
| TeaCache | 300 | N/A | N/A | N/A | N/A | N/A |

最终图文件：`results/hit_rate_over_arrivals.png`。横轴为到达请求数，纵轴为累计 hit rate，
每个跨请求基线一条线；TeaCache 不绘制或在图例中注明 N/A。
