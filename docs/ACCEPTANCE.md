# Acceptance

## 协议与口径

所有跨请求实验都按同一个 dog -> cat 协议执行：source 为
`a dog is sleeping on a grassland, focus on the dog`，target 为
`a cat is sleeping on a grassland, focus on the cat`，seed 为 `20260218`。
source 先写入方法自己的缓存 namespace，target 再以 source prompt 命中；表中的
时延只统计 target 请求。Dense baseline（Carnator C0）是所有 Wan target 加速结果的共同
参考：同模型、同 prompt、同 seed、50-step、无 Carnator cache、mask 注入、稀疏 attention
或 W8A8 FFN hook。speedup 为 `dense baseline target latency / 当前 target latency`；`N/A`
表示该指标没有实测，不是零收益。

TeaCache 是单请求方法，只运行 target prompt，不制造 source 命中。

## 结果汇总

| 模型 | 方法 / 变体 | 步数 | target E2E (s) | speedup | CLIP-T | VBench-Q | 状态 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Wan2.2-TI2V-5B | Dense baseline（Carnator C0） | 50 | 302.09 | 1.00x | 0.3374 | 0.6334 | 无方法级加速的 native dense target 参考 |
| Wan2.2-TI2V-5B | NIRVANA-VID | 40 | 237.08 | 1.27x | 0.3555 | 0.6337 | source_prompt hit；本次补跑完成 |
| Wan2.2-TI2V-5B | CHAI | 8 | 63.49 | 4.76x | 0.3601 | 0.6178 | source_prompt hit；block 0 逐步读取 source steps 1/2/3/4；本次重跑 |
| Wan2.2-TI2V-5B | TeaCache（target-only） | 22 | 137.98 | 2.19x | 0.3335 | 0.6209 | 单请求；22 个实际执行步；本次补跑完成 |
| Wan2.2-TI2V-5B | Carnator B0 | 50 | 201.28 | 1.50x | 0.3511 | 0.6381 | dense outer + FP FFN |
| Wan2.2-TI2V-5B | Carnator B1 | 50 | 197.78 | 1.53x | 0.3533 | 0.6366 | VariableBlock/FlashInfer + FP FFN；本次补跑完成 |
| Wan2.2-TI2V-5B | Carnator B2 | 50 | 153.12 | 1.97x | 0.3523 | 0.6346 | SpargeAttn + FP FFN |
| Wan2.2-TI2V-5B | Carnator B3 | 50 | 192.53 | 1.57x | 0.3513 | 0.6376 | dense outer + W8A8 FFN；本次补跑完成 |
| Wan2.2-TI2V-5B | Carnator B4 | 50 | 195.66 | 1.54x | 0.3516 | 0.6363 | VariableBlock/FlashInfer + W8A8 FFN；本次补跑完成 |
| Wan2.2-TI2V-5B | Carnator B5 | 50 | 146.88 | 2.06x | 0.3516 | 0.6360 | SpargeAttn + W8A8 FFN |
| Wan2.2-TI2V-5B | CHAI-Carnator dense target control | 8 | 63.90 | N/A | N/A | N/A | 同 seed/source initial latent 的未修改 8-step target；仅用于分离 CC-B 后端收益 |
| Wan2.2-TI2V-5B | CHAI-Carnator CC-B0 | 8 | 59.18 | 5.10x | 0.3457 | 0.5547 | block 0 CHAI cache steps 1/2/3/4；其余 block Carnator dense outer + FP FFN |
| Wan2.2-TI2V-5B | CHAI-Carnator CC-B1 | 8 | 60.57 | 4.99x | 0.3508 | 0.5594 | block 0 CHAI；其余 block FlashInfer VariableBlock + FP FFN |
| Wan2.2-TI2V-5B | CHAI-Carnator CC-B2 | 8 | 50.10 | 6.03x | 0.3494 | 0.5589 | block 0 CHAI；其余 block SpargeAttn + FP FFN |
| Wan2.2-TI2V-5B | CHAI-Carnator CC-B3 | 8 | 59.51 | 5.08x | 0.3367 | 0.5500 | block 0 CHAI；其余 block dense outer + W8A8 FFN |
| Wan2.2-TI2V-5B | CHAI-Carnator CC-B4 | 8 | 58.49 | 5.16x | 0.3506 | 0.5556 | block 0 CHAI；其余 block FlashInfer VariableBlock + W8A8 FFN |
| Wan2.2-TI2V-5B | CHAI-Carnator CC-B5 | 8 | 48.18 | 6.27x | 0.3484 | 0.5578 | block 0 CHAI；其余 block SpargeAttn + W8A8 FFN |
| CogVideoX1.5 | Dense baseline（Carnator C0） | N/A | N/A | N/A | N/A | N/A | 本轮按要求留空 |
| CogVideoX1.5 | NIRVANA-VID | N/A | N/A | N/A | N/A | N/A | 本轮按要求留空 |
| CogVideoX1.5 | CHAI | N/A | N/A | N/A | N/A | N/A | 本轮按要求留空；旧版产物已清理 |
| CogVideoX1.5 | TeaCache（target-only） | N/A | N/A | N/A | N/A | N/A | 本轮按要求留空 |
| CogVideoX1.5 | Carnator adapter（source-latent restart） | N/A | N/A | N/A | N/A | N/A | 本轮按要求留空 |
| CogVideoX1.5 | Carnator 原生 B0/B1/B2/B3/B4/B5 | N/A | N/A | N/A | N/A | N/A | 上游没有 CogVideoX1.5 的 Wan B-series 实现 |
| CogVideoX1.5 | CHAI-Carnator CC-B0/B1/B2/B3/B4/B5 | N/A | N/A | N/A | N/A | N/A | 依赖 Wan block/mask/FFN 拓扑，不宣称 CogVideo 等价实现 |

### CHAI-Carnator 定义

`CHAI-Carnator` 不是从 50-step Carnator trajectory 中截取八步 continuation。source 和
target 都使用同一个 8-step scheduler：source 保存 initial latent 和 post-scheduler
steps 1--8 的 latent，保存 block 0 的 source steps 1--4 token，并在 completed step 5
从 cross/self attention 生成融合 `Mbase/Medit/Msee`。这一个 completed-step-5 mask 固定
用于 target 的全部八步。

target 在 block 0 的 local steps 1--4 读取对应 source token 做 CHAI Cache Attention；
另外四步走原生 dense block-0 attention。block 10--20 在所有八步进行 Carnator region
update；block 1--9、21--29 按 CC-B1/CC-B2/CC-B4/CC-B5 使用对应 outer sparse backend，
CC-B0/CC-B3 保持 dense outer；CC-B3/CC-B4/CC-B5 使用 W8A8 FFN。每个 CC-B target 都
审计到 8 次 latent injection，且 8 次的 mask source 均为 completed step 5。

上述 CC-B0--CC-B5 在 H100 PCIe 上实际完成，source preparation（含缓存、mask）不计入
target E2E 与 speedup。CC-B 的 speedup 以 50-step dense baseline（Carnator C0，302.09s）为共同分母；
63.90s 的 8-step dense target 仅保留为 CC 后端消融对照。CLIP-T 和 VBench-Q 使用同一套
六维 VBench 流程评测。

## 产物索引

- Wan NIRVANA-VID：`artifacts/runs/real-wan-source-target-20260814-nirvana_vid-nirvana_vid/`
- Wan TeaCache：`artifacts/runs/real-wan-source-target-20260814-teacache-teacache/`
- Wan CHAI-Carnator source trajectory、block-0 token、step-5 mask 和 target 视频：
  `artifacts/chai-carnator/real-wan-chai-carnator-20260814/`
- Wan CHAI-Carnator CC-B0--CC-B5 matrix：
  `artifacts/chai-carnator-results/real-wan-chai-carnator-20260814/matrix.json`
- Wan CHAI-Carnator 质量评测（六个 CC-B 变体）：
  `artifacts/runs/real-wan-chai-carnator-20260814-quality/metrics.json`
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

旧版未定型组合基线仍保持清理状态；当前 `CHAI-Carnator` 是独立实现和独立实验目录，
不复用旧 source continuation 或旧验收数值。
