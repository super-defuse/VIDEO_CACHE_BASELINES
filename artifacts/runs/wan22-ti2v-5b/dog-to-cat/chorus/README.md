# Chorus Wan2.2 dog -> cat

这是修复 latent 拼接后的 Chorus 产物。source 轨迹沿用已验证的 50-step capture；target 使用同一 source cache 重新生成，替换了旧版在 mask 外出现噪声的 target 视频。

| 文件 | 说明 |
| --- | --- |
| `videos/0000_source-000.mp4` | source: `a dog is sleeping on a grassland, focus on the dog` |
| `videos/0001_target-000.mp4` | fixed target: `a cat is sleeping on a grassland, focus on the cat` |
| `results.jsonl` | source/target 请求级结果和实际输出 hash |
| `summary.json` | 本次 Chorus 汇总及 latent fusion 语义 |
| `config.resolved.json` | 运行时解析后的 Wan2.2/Chorus 配置 |

修复后的 target 规则是：从 `K1=10` 开始，source latent 在每个 target denoising step（包括 Stage 3，直到最终 step）参与 fusion；SRD 的 edit/see attention 只在 `10 <= step < 40` 的 Stage 2 生效。这样 mask 外的背景保持 source latent，而 mask 内仍由 target prompt 编辑。

fixed target 是基于现有 source cache 的 target-only 重跑，未单独持久化 Runner timing；`summary.json` 中的 `178.27s`/`227.83s` 是同配置完整 source/target 运行的参考计时，并已在 provenance 字段中明确标注。

旧目录 `artifacts/runs/chorus-wan22-final/` 及其摘要已删除；本目录是唯一保留的 Chorus 运行产物。
