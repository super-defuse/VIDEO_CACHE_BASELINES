# Wan2.2-TI2V-5B 验收

协议：seed `20260218`；source 为 `a dog is sleeping on a grassland, focus on the dog`；target
为 `a cat is sleeping on a grassland, focus on the cat`。所有行的加速比统一使用本轮 50-step
target C0：端到端 `293.466s`，去噪 `274.013s`。

## 结果

| 基线 | 配置 | 步数 | 时延 (s) | 加速比 | CLIP-T | VBench-Q |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Dense C0 | 50-step target | 50 | 293.47 | 1.00x | 0.3372 | 0.6331 |
| CHAI | 共享 steps 1--4 token | 8 | 64.78 | 4.53x | 0.3555 | 0.6046 |
| TeaCache | 仅目标请求 | 22 | 138.43 | 2.12x | 0.3330 | 0.6219 |
| NIRVANA-VID | source latent restart step 10 | 40 | 239.90 | 1.22x | 0.3557 | 0.6363 |
| Chorus | TGAA K1/K2=`10/30`；`gamma_k=1.5`，`gamma_o=1.5` | 40 | 198.38 | 1.48x | 0.3555 | 0.6113 |
| Chorus2 | SpargeAttn + source latent restart | 40 | 173.30 | 1.69x | 0.3560 | 0.6353 |
| Carnator Hard B0 | Dense / FP | 40 | 198.02 | 1.48x | 0.3508 | 0.6381 |
| Carnator Hard B1 | FlashInfer / FP | 40 | 201.93 | 1.45x | 0.3538 | 0.6299 |
| Carnator Hard B2 | SpargeAttn / FP | 40 | 150.73 | 1.95x | 0.3530 | 0.6304 |
| Carnator Hard B3 | Dense / W8A8 | 40 | 195.48 | 1.50x | 0.3508 | 0.6371 |
| Carnator Hard B4 | FlashInfer / W8A8 | 40 | 203.90 | 1.44x | 0.3535 | 0.6311 |
| Carnator Hard B5 | SpargeAttn / W8A8 | 40 | 157.34 | 1.87x | 0.3530 | 0.6297 |
| Carnator Boundary B0 | Dense / FP | 40 | 210.23 | 1.40x | 0.3535 | 0.6352 |
| Carnator Boundary B1 | FlashInfer / FP | 40 | 204.44 | 1.44x | 0.3552 | 0.6338 |
| Carnator Boundary B2 | SpargeAttn / FP | 40 | 152.78 | 1.92x | 0.3552 | 0.6348 |
| Carnator Boundary B3 | Dense / W8A8 | 40 | 199.33 | 1.47x | 0.3538 | 0.6350 |
| Carnator Boundary B4 | FlashInfer / W8A8 | 40 | 199.82 | 1.47x | 0.3552 | 0.6335 |
| Carnator Boundary B5 | SpargeAttn / W8A8 | 40 | 149.19 | 1.97x | 0.3547 | 0.6326 |

保留基线的质量评测清单和原始 VBench
维度结果见 [quality/results.jsonl](/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/wan22-ti2v-5b-dog-to-cat/quality/results.jsonl:1)
和 [quality/summary.json](/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/wan22-ti2v-5b-dog-to-cat/quality/summary.json:1)。

## 核验

- Chorus 已保留 source prompt，日志确认 `dog -> cat`，视频出现边界伪影；当前 TGAA 为 `gamma_k=1.5`、`gamma_o=1.5`。
- NIRVANA 和 Chorus2 均读取 source step-10 latent；Chorus2 为 2,400/2,400 命中、0 次 Dense fallback。身份保留来自 latent restart，不是缓存串错。

## 产物

所有验收产物统一位于：

```text
artifacts/acceptance/wan22-ti2v-5b-dog-to-cat/
  shared-dense-gpu3/                    # Dense C0 + shared cache
  chai-shared-cache-gpu7/
  teacache-shared-cache-gpu7/
  nirvana-vid-shared-cache-gpu5/
  chorus-shared-cache-gpu7/
  chorus2-shared-cache-gpu6/
  carnator-hard-B0..B5-gpu*/
  carnator-boundary-B0..B5-gpu*/
  results/                              # summaries/reports
  quality/                              # CLIP-T/VBench-Q
```
