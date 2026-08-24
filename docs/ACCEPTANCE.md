# 验收结果

## 协议

Wan2.2-TI2V-5B，seed `20260218`，source prompt 为
`a dog is sleeping on a grassland, focus on the dog`，target prompt 为
`a cat is sleeping on a grassland, focus on the cat`。

时延为目标端到端时间：prompt encoding、cache lookup/load（适用时）、去噪、VAE decode
和视频写出。source Dense、mask 计算与缓存落盘单列为 `cache_preparation`，不计入时延。
所有加速比均以同一次共享运行的 50-step target C0 `294.078s` 为分母，Chorus2 没有额外
Dense 对照。实验只使用 GPU 3--7；GPU 0--2 互连带宽较低，不参与生成或计时。

## 共享缓存

一次 source Dense 导出 Carnator trajectory/fusion mask/block mask、NIRVANA-VID step-10
latent、CHAI block-0 steps 1--4 token 和 Chorus trajectory。Chorus2 使用完整普通
30-layer block mask：0--9、21--29 硬链接 Carnator 原始文件，10--20 单独保存未经过
`Medit`/`Msee` 筛选的 raw sparse mask。Carnator 仅在 target 端将这份 raw mask 与
`Medit`/`Msee` 相交。

```bash
CUDA_VISIBLE_DEVICES=3 conda run -n wan env PYTHONNOUSERSITE=1 PYTHONPATH=. CARNATOR_ENV_READY=1 \
  python -m carnator.adapters.wan_ti2v_5b --no-env-setup --device cuda:0 \
  --artifact-root artifacts/acceptance/wan22-ti2v-5b-dog-to-cat-20260825 \
  --results-root artifacts/acceptance/wan22-ti2v-5b-dog-to-cat-20260825/results \
  --prepare-shared-dense-cache --run-id shared-dense-gpu3
```

## 结果

本次重跑：共享 source/C0、NIRVANA-VID、CHAI、TeaCache、Chorus、Chorus2、Carnator B0--B5
和 Carnator Boundary-Blend B0--B5。质量列沿用既有评测，未将其标为本次重算结果。

| 方法 | 配置 | 步数 | 时延 (s) | 加速比 | CLIP-T | VBench-Q |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Dense baseline | C0 | 50 | 294.08 | 1.00x | 0.3372 | 0.6331 |
| NIRVANA-VID | 默认 | 40 | 233.83 | 1.26x | 0.3555 | 0.6354 |
| CHAI | 默认 | 8 | 64.78 | 4.54x | 0.3572 | 0.6047 |
| TeaCache | 仅目标请求 | 22 | 138.43 | 2.12x | 0.3330 | 0.6219 |
| Chorus | TGAA；K1/K2=`10/30` | 40 | 240.58 | 1.22x | 0.3528 | 0.6104 |
| Chorus2 | SpargeAttn；共享 step-10 latent | 40 | 175.79 | 1.67x | 0.3560 | 0.6353 |

Chorus2 在 GPU 4 完成，2,400/2,400 mask 命中、零 Dense fallback；其 `1.67x` 直接使用
GPU 3 的共享 C0 分母。

### Carnator

| 变体 | 外层注意力 / FFN | 时延 (s) | 加速比 | CLIP-T | VBench-Q |
| --- | --- | ---: | ---: | ---: | ---: |
| B0 | Dense / FP | 198.02 | 1.49x | 0.3508 | 0.6381 |
| B1 | FlashInfer / FP | 201.93 | 1.46x | 0.3538 | 0.6299 |
| B2 | SpargeAttn / FP | 150.73 | 1.95x | 0.3530 | 0.6304 |
| B3 | Dense / W8A8 | 195.48 | 1.50x | 0.3508 | 0.6371 |
| B4 | FlashInfer / W8A8 | 203.90 | 1.44x | 0.3535 | 0.6311 |
| B5 | SpargeAttn / W8A8 | 157.34 | 1.87x | 0.3530 | 0.6297 |

### Carnator Boundary-Blend

| 变体 | 外层注意力 / FFN | 时延 (s) | 加速比 | CLIP-T | VBench-Q |
| --- | --- | ---: | ---: | ---: | ---: |
| B0 | Dense / FP | 210.23 | 1.40x | 0.3535 | 0.6352 |
| B1 | FlashInfer / FP | 204.44 | 1.44x | 0.3552 | 0.6338 |
| B2 | SpargeAttn / FP | 152.78 | 1.92x | 0.3552 | 0.6348 |
| B3 | Dense / W8A8 | 199.33 | 1.48x | 0.3538 | 0.6350 |
| B4 | FlashInfer / W8A8 | 199.82 | 1.47x | 0.3552 | 0.6335 |
| B5 | SpargeAttn / W8A8 | 149.19 | 1.97x | 0.3547 | 0.6326 |

## 产物

- 视频、Dense 轨迹和全部共享缓存：
  `artifacts/acceptance/wan22-ti2v-5b-dog-to-cat-20260825/`
  （`shared-dense-gpu3`、各方法的 `*-shared-cache-gpu*` 目录，以及
  `carnator-hard-B*-gpu*`、`carnator-boundary-B*-gpu*`）。

旧的单方法 source cache、Chorus2 restart Dense 对照、smoke 和中断产物已清理。
