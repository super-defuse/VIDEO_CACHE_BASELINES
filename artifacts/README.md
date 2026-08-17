# Artifact Layout

所有本轮 Wan2.2 TI2V-5B dog -> cat 结果按同一实验根目录归档：

```text
artifacts/
├── runs/wan22-ti2v-5b/dog-to-cat/
│   ├── chai/
│   ├── chorus/
│   ├── nirvana-vid/
│   ├── teacache/
│   ├── carnator-quality/
│   └── chai-carnator-quality/
├── carnator/ti2v-5b/<run-id>/          # Carnator 原生 source/mask/latent/video
├── carnator-results/ti2v-5b/<run-id>/  # Carnator matrix 和质量结果
├── chai-carnator/<run-id>/              # CHAI-Carnator source/cache/video
├── chai-carnator-results/<run-id>/      # CHAI-Carnator matrix
├── videos/wan2.2-ti2v-5b/dog-to-cat/   # 面向验收的统一视频符号链接
└── cache/Wan2.2/<method>/               # 本地持久缓存，不提交 Git
```

## 命名约定

- 运行目录使用 `<model>-<task>/<method>`，模型名和方法名统一使用小写 kebab-case。
- 每个方法目录包含 `config.resolved.json`、`results.jsonl`；跨请求运行的摘要统一为该目录内的 `summary.json`。
- 质量评测目录使用 `<method>-quality`，指标文件固定为 `metrics.json`。
- 视频使用 `videos/0000_source-000.mp4` 和 `videos/0001_target-000.mp4`；对外入口只放在 `videos/wan2.2-ti2v-5b/dog-to-cat/`。
- `cache/`、latent、模型权重和其他大体积运行缓存保留在本机，受 `.gitignore` 保护，不作为仓库产物推送。

## 当前验收产物

- `runs/wan22-ti2v-5b/dog-to-cat/chorus/` 是当前 Chorus source/target 结果；target 只在 Stage 2 (`10 <= step < 40`) 使用 source latent fusion、SRD 和 TGAA，Stage 3 为 dense full computation，不复用 latent。Stage 2 保留 native Wan head dense，Stage 3 在 step 40 清空 UniPC derivative history 并保留全局 timestep index。
- Chorus 质量结果位于 `runs/wan22-ti2v-5b/dog-to-cat/chorus/vbench/`，target `CLIP-T=0.3491`、`VBench-Q=0.6218`；逐请求字段写回该目录的 `results.jsonl`。
- `runs/wan22-ti2v-5b/dog-to-cat/carnator-quality/` 包含原生 Carnator B1/B3/B4、NIRVANA-VID 和 TeaCache 的质量补跑。
- `runs/wan22-ti2v-5b/dog-to-cat/chai-carnator-quality/` 包含 CC-B0--CC-B5 的质量结果。
- `videos/wan2.2-ti2v-5b/dog-to-cat/README.md` 是全部 target 视频的稳定入口索引。

CogVideoX1.5 本轮按要求不运行，因此不在 `artifacts` 中保留 CogVideo 运行目录或缓存。新运行应通过
`scripts/run/source_target_matrix.py --run-prefix wan22-ti2v-5b/dog-to-cat` 写入上述稳定目录，不要在 `artifacts/runs` 根目录创建临时结果。
