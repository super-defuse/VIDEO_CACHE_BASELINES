# Wan2.2-TI2V-5B dog -> cat videos

这些是 target prompt `a cat is sleeping on a grassland, focus on the cat` 的视频入口。
目录中的 `.mp4` 是指向原始实验产物的相对符号链接，不重复占用视频空间。

| 方法 | 视频 |
| --- | --- |
| Carnator C0 / dense reference | `carnator-c0.mp4` |
| 新基线 C0 / dense reference | `carnator-c0.mp4`（与 C0 共用同一 target reference） |
| NIRVANA-VID | `nirvana-vid.mp4` |
| CHAI | `chai.mp4` |
| TeaCache | `teacache.mp4` |
| Carnator B0 | `carnator-b0.mp4` |
| Carnator B1 | `carnator-b1.mp4` |
| Carnator B2 | `carnator-b2.mp4` |
| Carnator B3 | `carnator-b3.mp4` |
| Carnator B4 | `carnator-b4.mp4` |
| Carnator B5 | `carnator-b5.mp4` |
| 新基线 B0 | `new-baseline-b0.mp4` |
| 新基线 B1 | `new-baseline-b1.mp4` |
| 新基线 B2 | `new-baseline-b2.mp4` |
| 新基线 B3 | `new-baseline-b3.mp4` |
| 新基线 B4 | `new-baseline-b4.mp4` |
| 新基线 B5 | `new-baseline-b5.mp4` |

CHAI 的历史请求缓存现在保存 source steps 1--4 的四份 block-0 token；新基线保存
source steps 11--14，并在 target 全局 steps 11--14（local 1--4）逐步读取。

原始路径和运行元数据仍保留在 `artifacts/carnator/`、`artifacts/runs/` 以及
`artifacts/carnator-results/ti2v-5b/` 下；质量指标汇总见
`artifacts/runs/real-wan-baselines-quality-20260814/metrics.json`，新基线六项见
`artifacts/runs/real-wan-new-baseline-20260814-quality/metrics.json`。
