# Acceptance

## 协议与口径

所有跨请求实验都按同一个 dog -> cat 协议执行：source 为
`a dog is sleeping on a grassland, focus on the dog`，target 为
`a cat is sleeping on a grassland, focus on the cat`，seed 为 `20260218`。
Wan native Carnator/CHAI-Carnator 的 source positive prompt 为 `dog`。
source 先写入方法自己的缓存 namespace，target 再以 source prompt 命中；表中的
时延只统计 target 请求。Dense baseline（Carnator C0）是所有 Wan target 加速结果的共同
参考：同模型、同 prompt、同 seed、50-step、无 Carnator cache、mask 注入、稀疏 attention
或 W8A8 FFN hook。speedup 为 `dense baseline target latency / 当前 target latency`；`N/A`
表示该指标没有实测，不是零收益。历史 hard-injection 行沿用 `300.64s` 的 C0；
boundary-blend 组使用同一 run 内匹配的 `303.58s` C0，避免把不同运行条件的计时混作一个分母。

TeaCache 是单请求方法，只运行 target prompt，不制造 source 命中。

Chorus 的 target 计时沿用同一 Runner 口径：cache lookup、manifest 和逐步 trajectory
读取、模型生成、视频保存与 hash 都计入；只有 target 内部的 LLM prompt diff 与
Grounded-SAM-2 预处理通过 `excluded_latency_s` 单独列出并从报告时延剔除。原始墙钟和
各阶段明细保存在 Chorus 运行结果的 `extra.timing` 中。

## 结果汇总

| 模型 | 方法 / 变体 | 执行步数 | target E2E (s) | speedup | CLIP-T | VBench-Q | 状态 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Wan2.2-TI2V-5B | Dense baseline（Carnator C0） | 50 | 300.64 | 1.00x | 0.3374 | 0.6334 | 无方法级加速的 native dense target 参考；本轮重跑 |
| Wan2.2-TI2V-5B | NIRVANA-VID | 40 | 237.08 | 1.27x | 0.3555 | 0.6337 | source_prompt hit；本次补跑完成 |
| Wan2.2-TI2V-5B | CHAI | 8 | 63.49 | 4.76x | 0.3601 | 0.6178 | source_prompt hit；block 0 逐步读取 source steps 1/2/3/4；本次重跑 |
| Wan2.2-TI2V-5B | TeaCache（target-only） | 22 | 137.98 | 2.19x | 0.3335 | 0.6209 | 单请求；22 个实际执行步；本次补跑完成 |
| Wan2.2-TI2V-5B | Carnator B0 | 40 | 208.88 | 1.44x | 0.3511 | 0.6381 | dense outer + FP FFN；本轮重跑 |
| Wan2.2-TI2V-5B | Carnator B1 | 40 | 206.61 | 1.46x | 0.3533 | 0.6366 | VariableBlock/FlashInfer + FP FFN；本轮重跑 |
| Wan2.2-TI2V-5B | Carnator B2 | 40 | 162.67 | 1.85x | 0.3523 | 0.6346 | SpargeAttn + FP FFN；本轮重跑 |
| Wan2.2-TI2V-5B | Carnator B3 | 40 | 193.65 | 1.55x | 0.3513 | 0.6376 | dense outer + W8A8 FFN；本轮重跑 |
| Wan2.2-TI2V-5B | Carnator B4 | 40 | 196.01 | 1.53x | 0.3516 | 0.6363 | VariableBlock/FlashInfer + W8A8 FFN；本轮重跑 |
| Wan2.2-TI2V-5B | Carnator B5 | 40 | 149.09 | 2.02x | 0.3516 | 0.6360 | SpargeAttn + W8A8 FFN；本轮重跑 |
| Wan2.2-TI2V-5B | Carnator boundary-blend dense target control（C0） | 50 | 303.58 | 1.00x | 0.3374 | 0.6334 | 同 run 的 dense target 参考；smoothstep；CLIP/VBench 本轮完成 |
| Wan2.2-TI2V-5B | Carnator boundary-blend B0 | 40 | 201.93 | 1.50x | 0.3540 | 0.6348 | dense outer + FP FFN；smoothstep；CLIP/VBench 本轮完成 |
| Wan2.2-TI2V-5B | Carnator boundary-blend B1 | 40 | 198.43 | 1.53x | 0.3542 | 0.6314 | VariableBlock/FlashInfer + FP FFN；smoothstep；CLIP/VBench 本轮完成 |
| Wan2.2-TI2V-5B | Carnator boundary-blend B2 | 40 | 153.02 | 1.98x | 0.3542 | 0.6304 | SpargeAttn + FP FFN；smoothstep；CLIP/VBench 本轮完成 |
| Wan2.2-TI2V-5B | Carnator boundary-blend B3 | 40 | 197.17 | 1.54x | 0.3533 | 0.6344 | dense outer + W8A8 FFN；smoothstep；CLIP/VBench 本轮完成 |
| Wan2.2-TI2V-5B | Carnator boundary-blend B4 | 40 | 196.46 | 1.55x | 0.3550 | 0.6320 | VariableBlock/FlashInfer + W8A8 FFN；smoothstep；CLIP/VBench 本轮完成 |
| Wan2.2-TI2V-5B | Carnator boundary-blend B5 | 40 | 148.16 | 2.05x | 0.3545 | 0.6307 | SpargeAttn + W8A8 FFN；smoothstep；CLIP/VBench 本轮完成 |
| Wan2.2-TI2V-5B | CHAI-Carnator dense target control | 8 | 63.92 | N/A | N/A | N/A | 同 seed/source initial latent 的未修改 8-step target；本轮重跑 |
| Wan2.2-TI2V-5B | CHAI-Carnator CC-B0 | 8 | 58.92 | 5.10x | 0.3457 | 0.5547 | block 0 CHAI cache steps 1/2/3/4；其余 block Carnator dense outer + FP FFN；本轮重跑 |
| Wan2.2-TI2V-5B | CHAI-Carnator CC-B1 | 8 | 60.52 | 4.97x | 0.3508 | 0.5594 | block 0 CHAI；其余 block FlashInfer VariableBlock + FP FFN；本轮重跑 |
| Wan2.2-TI2V-5B | CHAI-Carnator CC-B2 | 8 | 50.19 | 5.99x | 0.3494 | 0.5589 | block 0 CHAI；其余 block SpargeAttn + FP FFN；本轮重跑 |
| Wan2.2-TI2V-5B | CHAI-Carnator CC-B3 | 8 | 59.45 | 5.06x | 0.3367 | 0.5500 | block 0 CHAI；其余 block dense outer + W8A8 FFN；本轮重跑 |
| Wan2.2-TI2V-5B | CHAI-Carnator CC-B4 | 8 | 58.48 | 5.14x | 0.3506 | 0.5556 | block 0 CHAI；其余 block FlashInfer VariableBlock + W8A8 FFN；本轮重跑 |
| Wan2.2-TI2V-5B | CHAI-Carnator CC-B5 | 8 | 48.30 | 6.23x | 0.3484 | 0.5578 | block 0 CHAI；其余 block SpargeAttn + W8A8 FFN；本轮重跑 |
| Wan2.2-TI2V-5B | Chorus | 40 | 179.06 | 1.68x | 0.3491 | 0.6218 | source `295.77s`；K1=10/K2=40；Stage 2 使用 SRD、dense head、全部 30 个 blocks 的 TGAA（1800 calls）和 latent fusion；Stage 3 在 step 40 重置 UniPC history 后 dense/no-reuse；LLM+SAM `60.28s` 单独剔除；VBench 六维度已完成 |
| CogVideoX1.5 | Dense baseline（Carnator C0） | N/A | N/A | N/A | N/A | N/A | 本轮按要求留空 |
| CogVideoX1.5 | NIRVANA-VID | N/A | N/A | N/A | N/A | N/A | 本轮按要求留空 |
| CogVideoX1.5 | CHAI | N/A | N/A | N/A | N/A | N/A | 本轮按要求留空；旧版产物已清理 |
| CogVideoX1.5 | TeaCache（target-only） | N/A | N/A | N/A | N/A | N/A | 本轮按要求留空 |
| CogVideoX1.5 | Carnator adapter（source-latent restart） | N/A | N/A | N/A | N/A | N/A | 本轮按要求留空 |
| CogVideoX1.5 | Carnator 原生 B0/B1/B2/B3/B4/B5 | N/A | N/A | N/A | N/A | N/A | 上游没有 CogVideoX1.5 的 Wan B-series 实现 |
| CogVideoX1.5 | CHAI-Carnator CC-B0/B1/B2/B3/B4/B5 | N/A | N/A | N/A | N/A | N/A | 依赖 Wan block/mask/FFN 拓扑，不宣称 CogVideo 等价实现 |

### Carnator / CHAI-Carnator 定义

原生 Carnator source 仍完整运行 50 步并保存每一步 latent trajectory，但只在 completed
step 5 采集一次 cross-attention，并在同一步执行 self-attention probe，生成一份 canonical
`Mbase/Medit/Msee` 及 latent mask。target 从 reuse 开始的所有去噪步都使用这份 step-5 mask；
不再保存或复制 per-step region-mask。B1/B2/B4/B5 的 outer sparse attention 仍按各自的
逐步 block mask 工作，它们与 Carnator region mask 是两类独立缓存。

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
target E2E 与 speedup。CC-B 的 speedup 以本轮 50-step dense baseline（Carnator C0，300.64s）为共同分母；
63.92s 的 8-step dense target 仅保留为 CC 后端消融对照。表中的 CLIP-T/VBench-Q 沿用此前
协议等价运行的质量结果，本轮重点重跑 source、C0/B0--B5 与 CC-B0--CC-B5 的缓存和时延，
未重新执行 VBench；boundary-blend 组的 CLIP-T/VBench-Q 已在本轮单独完成。

### Boundary-blend 变体

Boundary-blend 保留原 Carnator 的 canonical `Mbase/Medit/Msee`，新增
`Mband = Medit - Mbase`。在 latent 注入时使用
`L_new = Mbase * L_target + Mband * ((1 - alpha) * L_target + alpha * L_source) + (1 - Medit) * L_source`。
`alpha` 由 token-grid 上到 `Mbase` 的 Chebyshev 距离计算，在 `Mband` 内使用 smoothstep
（本轮配置），因此 `Mbase` 内完全采用 target latent，`Medit` 外完全采用 source latent，
只在中间带平滑过渡。旧的 `latent-injection-mode=hard` 行为保持不变；该变体不增加模型
forward，只增加 mask/latent 的逐元素融合。

本轮固定 step-5 mask 的 latent 面积为 `Mbase=16.91%`、`Mband=18.22%`、
`Medit=35.13%`；alpha 的运行时均值为 `0.5071`（范围 `[0, 1]`），每个 B 变体执行
40 次注入。表中 boundary-blend 的 speedup 使用同一个 run 内的 303.58s dense C0，
与上方历史 hard-injection 结果的 300.64s 仅作参考，不能直接混算。

## 产物索引

- Wan NIRVANA-VID：`artifacts/runs/wan22-ti2v-5b/dog-to-cat/nirvana-vid/`
- Wan TeaCache：`artifacts/runs/wan22-ti2v-5b/dog-to-cat/teacache/`
- Wan CHAI-Carnator source trajectory、block-0 token、step-5 mask 和 target 视频：
  `artifacts/chai-carnator/real-wan-chai-carnator-20260814/`
- Wan CHAI-Carnator CC-B0--CC-B5 matrix：
  `artifacts/chai-carnator-results/real-wan-chai-carnator-20260814/matrix.json`
- Wan 原生 Carnator source、C0/B0--B5 matrix：
  `artifacts/carnator-results/ti2v-5b/real-wan-c0-b4/matrix.json`
- Wan 原生 Carnator boundary-blend source、C0/B0--B5 matrix：
  `artifacts/carnator-results/ti2v-5b/real-wan-boundary-blend-20260817/matrix.json`
- Wan 原生 Carnator boundary-blend source、C0/B0--B5 视频与缓存：
  `artifacts/carnator/ti2v-5b/real-wan-boundary-blend-20260817/`
- Wan 原生 Carnator boundary-blend 质量评测（C0、B0--B5，CLIP-T + 六维 VBench-Q）：
  `artifacts/runs/wan22-ti2v-5b/dog-to-cat/carnator-boundary-blend-quality/metrics.json`
- 本项目独立 VBench 模型缓存：
  `artifacts/vbench-cache/`
- Wan CHAI-Carnator 质量评测（六个 CC-B 变体）：
  `artifacts/runs/wan22-ti2v-5b/dog-to-cat/chai-carnator-quality/metrics.json`
- Wan 缺失质量指标补跑（原生 B1/B3/B4、NIRVANA-VID、TeaCache）：
  `artifacts/runs/wan22-ti2v-5b/dog-to-cat/carnator-quality/metrics.json`
- Wan dog -> cat 全部 target 视频统一索引：
  `artifacts/videos/wan2.2-ti2v-5b/dog-to-cat/README.md`
- Wan CHAI 四步缓存载荷（source steps 1--4）：
  `artifacts/cache/Wan2.2/chai/entries/source-000.latent`
- Wan 原生 Carnator source、C0/B0--B5：`artifacts/carnator/ti2v-5b/real-wan-c0-b4/`；
  结果汇总：`artifacts/carnator-results/ti2v-5b/real-wan-c0-b4/`
- CogVideoX1.5：本轮不运行，未保留旧版运行目录和缓存。

- Wan Chorus source/target 视频、结果、质量评估与分阶段计时：
  `artifacts/runs/wan22-ti2v-5b/dog-to-cat/chorus/`；本轮 target `179.06s`、source_prompt
  hit，target `CLIP-T=0.3491`、`VBench-Q=0.6218`；VBench 原始六维度见同目录下的
  `vbench/custom_vidprom_eval_results.json`；cache：`artifacts/cache/Wan2.2/chorus/`。

旧版未定型组合基线仍保持清理状态；当前 `CHAI-Carnator` 是独立实现和独立实验目录，
不复用旧 source continuation 或旧验收数值。
