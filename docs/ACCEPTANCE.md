# Acceptance

## 协议与口径

所有跨请求实验都按同一个 dog -> cat 协议执行：source 为
`a dog is sleeping on a grassland, focus on the dog`，target 为
`a cat is sleeping on a grassland, focus on the cat`，seed 为 `20260218`。
source 先写入方法自己的缓存 namespace，target 再以 source prompt 命中；表中的
时延只统计 target 请求。C0/dense target 是同模型、同 prompt、同 seed 的 target-only
参考。speedup 为 `C0 target latency / 当前 target latency`；`N/A` 表示该指标没有实测，
不是零收益。

TeaCache 是单请求方法，只运行 target prompt，不制造 source 命中。

## 结果汇总

| 模型 | 方法 / 变体 | 步数 | target E2E (s) | speedup | CLIP-T | VBench-Q | 状态 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Wan2.2-TI2V-5B | Carnator C0 / dense target reference | 50 | 302.09 | 1.00x | 0.3374 | 0.6334 | 跨请求协议中的 dense 参考 |
| Wan2.2-TI2V-5B | NIRVANA-VID | 40 | 237.08 | 1.27x | 0.3555 | 0.6337 | source_prompt hit；本次补跑完成 |
| Wan2.2-TI2V-5B | CHAI | 8 | 63.49 | 4.76x | 0.3601 | 0.6178 | source_prompt hit；block 0 逐步读取 source steps 1/2/3/4；本次重跑 |
| Wan2.2-TI2V-5B | TeaCache（target-only） | 22 | 137.98 | 2.19x | 0.3335 | 0.6209 | 单请求；22 个实际执行步；本次补跑完成 |
| Wan2.2-TI2V-5B | Carnator B0 | 50 | 201.28 | 1.50x | 0.3511 | 0.6381 | dense outer + FP FFN |
| Wan2.2-TI2V-5B | Carnator B1 | 50 | 197.78 | 1.53x | 0.3533 | 0.6366 | VariableBlock/FlashInfer + FP FFN；本次补跑完成 |
| Wan2.2-TI2V-5B | Carnator B2 | 50 | 153.12 | 1.97x | 0.3523 | 0.6346 | SpargeAttn + FP FFN |
| Wan2.2-TI2V-5B | Carnator B3 | 50 | 192.53 | 1.57x | 0.3513 | 0.6376 | dense outer + W8A8 FFN；本次补跑完成 |
| Wan2.2-TI2V-5B | Carnator B4 | 50 | 195.66 | 1.54x | 0.3516 | 0.6363 | VariableBlock/FlashInfer + W8A8 FFN；本次补跑完成 |
| Wan2.2-TI2V-5B | Carnator B5 | 50 | 146.88 | 2.06x | 0.3516 | 0.6360 | SpargeAttn + W8A8 FFN |
| Wan2.2-TI2V-5B | 新基线 C0 / dense reference | 50 | 302.09 | 1.00x | 0.3374 | 0.6334 | 与 Carnator C0 相同的 target reference |
| Wan2.2-TI2V-5B | 新基线 B0 | 8 | 56.03 | 5.39x | 0.2262 | 0.5857 | block 0 Cache Attention + 其余层 Carnator B0；source/target Cache Attention steps 11--14；本次重跑 |
| Wan2.2-TI2V-5B | 新基线 B1 | 8 | 60.45 | 5.00x | 0.2264 | 0.5858 | block 0 + B1 VariableBlock；source/target Cache Attention steps 11--14；本次重跑 |
| Wan2.2-TI2V-5B | 新基线 B2 | 8 | 50.90 | 5.94x | 0.2278 | 0.5859 | block 0 + B2 SpargeAttn；source/target Cache Attention steps 11--14；本次重跑 |
| Wan2.2-TI2V-5B | 新基线 B3 | 8 | 60.58 | 4.99x | 0.2261 | 0.5859 | block 0 + B3 W8A8 FFN；source/target Cache Attention steps 11--14；本次重跑 |
| Wan2.2-TI2V-5B | 新基线 B4 | 8 | 62.24 | 4.85x | 0.2272 | 0.5858 | block 0 + B4 VariableBlock/W8A8；source/target Cache Attention steps 11--14；本次重跑 |
| Wan2.2-TI2V-5B | 新基线 B5 | 8 | 52.46 | 5.76x | 0.2269 | 0.5859 | block 0 + B5 SpargeAttn/W8A8；source/target Cache Attention steps 11--14；本次重跑 |
| CogVideoX1.5 | Carnator C0 / dense target reference | N/A | N/A | N/A | N/A | N/A | 本轮按要求留空 |
| CogVideoX1.5 | NIRVANA-VID | N/A | N/A | N/A | N/A | N/A | 本轮按要求留空 |
| CogVideoX1.5 | CHAI | N/A | N/A | N/A | N/A | N/A | 本轮按要求留空；旧版产物已清理 |
| CogVideoX1.5 | TeaCache（target-only） | N/A | N/A | N/A | N/A | N/A | 本轮按要求留空 |
| CogVideoX1.5 | Carnator adapter（source-latent restart） | N/A | N/A | N/A | N/A | N/A | 本轮按要求留空 |
| CogVideoX1.5 | Carnator 原生 B0/B1/B2/B3/B4/B5 | N/A | N/A | N/A | N/A | N/A | 上游没有 CogVideoX1.5 的 Wan B-series 实现 |
| CogVideoX1.5 | 新基线 C0/B0/B1/B2/B3/B4/B5 | N/A | N/A | N/A | N/A | N/A | 新基线依赖 Wan2.2 block/mask/FFN 拓扑，不宣称 CogVideo 等价实现 |

### 新基线定义

新基线从 Carnator source trajectory 的 post-step-10 latent 恢复，只计算原 scheduler 的
indices `10..17` 共八步。每个 B 变体都在 target 请求中执行 Carnator 的 latent/mask
复用；与旧 Carnator 的差异只有：

- block 0 从 outer sparse layer 列表中移除，使用 CHAI Cache Attention；local steps
  `1/2/3/4` 激活，其他四个 local steps 对 block 0 走 dense attention。
- 其余层保持原 B-series 策略：B1 = FlashInfer VariableBlock，B2 = SpargeAttn，
  B3 = dense outer，B4 = FlashInfer，B5 = SpargeAttn；outer sparse layers 为
  `1-9,21-29`，region-update layers 仍为 `10-20`。
- B3/B4/B5 的全局 W8A8 FFN 保留；B0/B1/B2 使用 FP FFN。

每个新基线结果的 JSON 都审计了 `source_model_steps=[11,12,13,14]`、
`target_global_steps=[11,12,13,14]`、`activated_local_steps=[1,2,3,4]`、outer layer
列表、FFN backend 和 8 次 masked injection。

Wan 新基线 B0--B5、Wan CHAI 以及缺失的 Wan 质量评估均已实际完成；CogVideoX1.5
本轮按要求不运行，验收结果和产物均留空。

## 产物索引

- Wan NIRVANA-VID：`artifacts/runs/real-wan-source-target-20260814-nirvana_vid-nirvana_vid/`
- Wan TeaCache：`artifacts/runs/real-wan-source-target-20260814-teacache-teacache/`
- Wan 新基线 source trajectory、mask 和 latent：
  `artifacts/carnator/real-wan-new-baseline-20260814/source/`
- Wan 新基线 B0--B5 summaries、reports、matrix JSON/Markdown：
  `artifacts/carnator-results/ti2v-5b/real-wan-new-baseline-20260814/`
- Wan 新基线 source block-0 token（source 全局 steps 11--14）：
  `artifacts/carnator/real-wan-new-baseline-20260814/source/cache/source_dense/chai_block0_tokens/`
- Wan 新基线 Cache Attention 映射清单：
  `artifacts/carnator/real-wan-new-baseline-20260814/source/cache/source_dense/chai_cache_attention.json`
- Wan 新基线质量评测（六个 B 变体）：
  `artifacts/runs/real-wan-new-baseline-20260814-quality/metrics.json`
- Wan 缺失质量指标补跑（原生 B1/B3/B4、NIRVANA-VID、TeaCache）：
  `artifacts/runs/real-wan-baselines-quality-20260814/metrics.json`
- Wan dog -> cat 全部 target 视频统一索引：
  `artifacts/videos/wan2.2-ti2v-5b/dog-to-cat/README.md`
- Wan CHAI 四步缓存载荷（source steps 1--4）：
  `artifacts/cache/Wan2.2/chai/entries/source-000.latent`
- Wan 原生 Carnator C0/B0/B2/B5：`artifacts/carnator-results/ti2v-5b/real-wan-c0-b4/`；
  B1/B4：`artifacts/carnator-results/ti2v-5b/real-wan-c0-b4-flashinfer-20260813/`；
  B3：`artifacts/carnator-results/ti2v-5b/real-wan-c0-b3-20260814/`
- CogVideoX1.5：本轮不运行，未保留旧版运行目录和缓存。

旧版组合基线的代码、配置和实验目录已清理；本次新基线和原生 Carnator 结果共享同一份
source-only 产物，`real-wan-c0-b4/history` 仅作为兼容符号链接保留，不再复制 source cache。
