# Artifact Layout

当前验收只保留 Wan2.2-TI2V-5B 的 dog -> cat 产物。目录职责固定如下：

- `videos/wan2.2-ti2v-5b/dog-to-cat/`: 所有 target 视频的统一符号链接入口。
- `runs/`: CHAI、TeaCache、NIRVANA-VID 和质量评估运行目录。
- `carnator/`: Wan source trajectory、mask、latent 和 target 视频。
- `carnator-results/`: Carnator summaries、reports、matrix 和质量补跑记录。
- `cache/Wan2.2/`: 当前 Wan 方法的持久缓存；CHAI 载荷为 source steps 1--4。

`carnator/ti2v-5b/real-wan-c0-b4/history` 是指向
`carnator/real-wan-new-baseline-20260814/source` 的兼容符号链接，供原生
Carnator 结果中的历史 manifest 继续解析；source cache 只保留一份。

CogVideoX1.5 本轮按要求不运行，因此不在 `artifacts` 中保留 CogVideo 运行目录或缓存。
新的运行应复用以上目录和现有 run name，不要在 `artifacts` 根目录创建临时结果目录。
