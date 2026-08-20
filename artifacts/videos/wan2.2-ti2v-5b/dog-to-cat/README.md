# Wan2.2-TI2V-5B dog -> cat videos

这些是 target prompt `a cat is sleeping on a grassland, focus on the cat` 的视频入口。
目录中的 `.mp4` 是指向原始实验产物的相对符号链接，不重复占用视频空间。

| 方法 | 视频 |
| --- | --- |
| Dense baseline（Carnator C0） | `carnator-c0.mp4` |
| NIRVANA-VID | `nirvana-vid.mp4` |
| CHAI | `chai.mp4` |
| TeaCache | `teacache.mp4` |
| Chorus | `chorus.mp4` |
| Chorus（仅去掉 TGAA） | `chorus-no-tgaa.mp4` |
| Chorus TGAA joint-gamma probe（`gamma_k=gamma_o=1.30`） | `chorus-tgaa-joint-gamma-1-3.mp4` |
| Chorus TGAA joint-gamma probe（`gamma_k=gamma_o=1.6`） | `chorus-tgaa-joint-gamma-1-6.mp4` |
| Carnator B0 | `carnator-b0.mp4` |
| Carnator B1 | `carnator-b1.mp4` |
| Carnator B2 | `carnator-b2.mp4` |
| Carnator B3 | `carnator-b3.mp4` |
| Carnator B4 | `carnator-b4.mp4` |
| Carnator B5 | `carnator-b5.mp4` |
| Carnator boundary-blend C0 | `carnator-boundary-blend-c0.mp4` |
| Carnator boundary-blend B0 | `carnator-boundary-blend-b0.mp4` |
| Carnator boundary-blend B1 | `carnator-boundary-blend-b1.mp4` |
| Carnator boundary-blend B2 | `carnator-boundary-blend-b2.mp4` |
| Carnator boundary-blend B3 | `carnator-boundary-blend-b3.mp4` |
| Carnator boundary-blend B4 | `carnator-boundary-blend-b4.mp4` |
| Carnator boundary-blend B5 | `carnator-boundary-blend-b5.mp4` |
| CHAI-Carnator CC-B0 | `chai-carnator-b0.mp4` |
| CHAI-Carnator CC-B1 | `chai-carnator-b1.mp4` |
| CHAI-Carnator CC-B2 | `chai-carnator-b2.mp4` |
| CHAI-Carnator CC-B3 | `chai-carnator-b3.mp4` |
| CHAI-Carnator CC-B4 | `chai-carnator-b4.mp4` |
| CHAI-Carnator CC-B5 | `chai-carnator-b5.mp4` |

原始路径和运行元数据仍保留在 `artifacts/carnator/`、`artifacts/chai-carnator/`、
`artifacts/runs/wan22-ti2v-5b/dog-to-cat/`、`artifacts/carnator-results/ti2v-5b/`
以及 `artifacts/chai-carnator-results/` 下；boundary-blend 的质量指标见
`artifacts/runs/wan22-ti2v-5b/dog-to-cat/carnator-boundary-blend-quality/metrics.json`，
CHAI-Carnator 质量指标见
`artifacts/runs/wan22-ti2v-5b/dog-to-cat/chai-carnator-quality/metrics.json`。
