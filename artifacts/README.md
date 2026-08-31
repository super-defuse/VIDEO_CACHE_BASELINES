# Artifact Layout

`artifacts/` 只保留当前验收、仍在使用的共享缓存和明确可复现的实验产物。

## 当前验收

2026-08-24 的 Wan2.2 TI2V-5B dog -> cat 验收入口：

- `videos/wan2.2-ti2v-5b/dog-to-cat/`：18 个当前 target 视频的符号链接和视频索引。
- `acceptance/wan22-ti2v-5b/20260824/`：运行目录、质量指标与视频入口的总索引。
- `runs/wan22-ti2v-5b/rerun-20260824/`：NIRVANA-VID、CHAI、TeaCache 的原始结果和质量评测。
- `runs/wan22-ti2v-5b/rerun-20260824/chorus-gamma-1-5-fixed-k1-10-k2-30/`：当前 Chorus 原始结果和质量评测。
- `carnator/ti2v-5b/dog-to-cat-fast-refresh-20260824/`：Dense baseline 与 Carnator B0--B1 的原始缓存和视频。
- `ti2v-5b/dog-to-cat-fast-refresh-20260824*/`：Carnator B2--B5 及 boundary-blend 的原始缓存和视频。
- `chorus2/dog-to-cat-fast-refresh-20260824-gpu3/`：Chorus2 的原始缓存和视频。

## 长期保留

- `shared_cross_request_cache/`：当前缓存基于 `datasets/1000cache_300input_noun_match/` 的 1000 条预置请求，包含 Early Signature。
- `vidprom_1000cache_300input_noun_match/`：当前 300-request 实验的配对、生成、质量评估和汇总结果，Carnator 使用 v2 的 noun -> Early Signature -> CLIP 检索顺序。
- `early_signature_color_v1_calibration_cluster_02_2170/`：当前校准实验，不能按常规验收缓存清理。
- `cross_request_retrieval/`：检索产物。
- `vbench-cache/`：质量评测模型缓存。
- `cache/synthetic/`：测试夹具，不用于验收结果。

历史 Wan 基线缓存、历史 Carnator 结果以及早期重复 Chorus2 运行已移动到系统回收站的 `VIDEO_CACHE_BASELINES-cleanup-20260824/`，未永久删除。
