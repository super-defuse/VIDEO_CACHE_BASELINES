# Artifact Layout

当前验收只保留 Wan2.2-TI2V-5B 的 dog -> cat 产物。目录职责固定如下：

- `videos/wan2.2-ti2v-5b/dog-to-cat/`: 所有 target 视频的统一符号链接入口。
- `runs/`: CHAI、TeaCache、NIRVANA-VID 和质量评估运行目录。
- `carnator/`: Wan source trajectory、mask、latent 和 target 视频。
- `carnator-results/`: Carnator summaries、reports、matrix 和质量补跑记录。
- `chai-carnator/`: CHAI-Carnator 的 8-step source trajectory、block-0 token、step-5 mask 和 CC-B0--CC-B5 视频。
- `chai-carnator-results/`: CHAI-Carnator matrix JSON。
- `cache/Wan2.2/`: 当前 Wan 方法的持久缓存；CHAI 载荷为 source steps 1--4。

CogVideoX1.5 本轮按要求不运行，因此不在 `artifacts` 中保留 CogVideo 运行目录或缓存。
新运行应使用上面的稳定方法目录和明确 run name；不要在 `artifacts` 根目录创建无归属的临时结果目录。
