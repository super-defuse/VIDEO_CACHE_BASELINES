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

## 多 Prompt 验收（2026-09-18）

[`scripts/run/run_acceptance_experiment.py`](../scripts/run/run_acceptance_experiment.py) 的
五基线多 prompt 验收在本轮跑通，协议和目录布局见
[ACCEPTANCE_MULTI_PROMPT_RUNNER_PLAN.md](ACCEPTANCE_MULTI_PROMPT_RUNNER_PLAN.md)。上表的
CHAI / TeaCache 和 carnator 的 Hard/Boundary matrix 不在本轮范围内，本轮固定为
`dense_baseline` / `nirvana_vid` / `chorus` / `chorus2` / `carnator_boundary_b5`。

| 运行 | Prompt 对 | 产物 |
| --- | ---: | --- |
| `20260918_172444` | 1（dog→cat） | 单 pair 冒烟，同时校验编排器和旧参考表的对齐 |
| `20260918_183809` | 5 | horse→zebra、lion→tiger、duck→swan、bear→panda、apple→orange |

五个 pair 的均值（`20260918_183809` 的 `## Mean across pairs`，去噪计时不含模型加载、
source 准备和质量评测）：

| 基线 | 均值去噪 (s) | 均值加速比 | 均值 CLIP-T | CLIP-T 极差 | 均值 VBench-Q | VBench-Q 极差 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense | 274.63 | 1.00x | 0.3385 | 0.3232 - 0.3477 | 0.6426 | 0.6080 - 0.6730 |
| NIRVANA-VID | 237.98 | 1.15x | 0.2988 | 0.2554 - 0.3171 | 0.6571 | 0.6273 - 0.6754 |
| Chorus | 183.57 | 1.50x | 0.3195 | 0.3062 - 0.3345 | 0.6530 | 0.6235 - 0.6774 |
| Chorus2 | 201.47 | 1.36x | 0.3382 | 0.3235 - 0.3459 | 0.6396 | 0.6074 - 0.6689 |
| Carnator Boundary B5 | 143.98 | 1.92x | 0.2920 | 0.2451 - 0.3188 | 0.6534 | 0.6254 - 0.6735 |

五对 prompt 上 4 条走缓存的基线各自的 source 轨迹逐字节一致（`## Source consistency`
全部 PASS）。CLIP-T 的极差说明单 pair 结果（含上表的 dog→cat）分不开基线和 prompt 运气，
均值才看得出趋势：NIRVANA-VID 和 Carnator B5 的 VBench-Q 略高于 Dense，但 CLIP-T 明显更低。

### Carnator target 前缀变体

两轮都附带了 `carnator_boundary_b5_targetresume`，由
[`scripts/run/run_carnator_target_latent_resume.py`](../scripts/run/run_carnator_target_latent_resume.py)
在已有验收根目录上补跑（它复用验收 runner 的 `source_stage`，只写
`*.carnator_boundary_b5_targetresume.json` / `.md` 兄弟文件，不改写被接受的 summary）。
对应 adapter 开关 `--target-resume-from-target-prefix`：先让 target prompt 自己跑
`reuse_start_step` 步 dense，再从该前缀的 latent 恢复复用阶段，只有被掩码区域的 source
注入从缓存继承。五对均值
Δ CLIP-T `+0.0409`、Δ VBench-Q `-0.0161`，比被接受的 B5 高；但 dog→cat 单对 Δ CLIP-T
为 `-0.0122`，方向不一致。

代价是这个变体必须真跑一段 dense 前缀（均值 `62.6s`），所以诚实的端到端加速比是
`1.35x`（`Speedup (variant)`），而不是只算复用阶段的 `1.88x`。相对被接受的 B5 的 `1.76x`
是净损失，因此该开关保持实验性、默认关闭。
