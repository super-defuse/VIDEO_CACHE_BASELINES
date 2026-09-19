# Artifact Layout

`artifacts/` 只保留当前验收、仍在使用的共享缓存和明确可复现的实验产物。哪些文件入库由
[.gitignore](../.gitignore) 决定，里面每个区块都写明「保留什么、为什么排除」，本文件只
记录目录职责，不重复排除规则。

## 当前验收

`acceptance/` 下的五轮 Wan2.2-TI2V-5B 验收：

| 运行 | Prompt 对 | 协议 | 口径 |
| --- | ---: | --- | --- |
| `wan22-ti2v-5b-dog-to-cat/` | 1（dog→cat） | 共享 source cache、单卡串行 | 18 条 baseline 的冻结参考表，见 [docs/ACCEPTANCE.md](../docs/ACCEPTANCE.md) |
| `20260902_174245/` | 1（dog→cat） | 共享 source cache、单卡（GPU0） | 六基线，含 TeaCache；旧布局 |
| `20260918_172444/` | 1（dog→cat） | 每条基线各自 source、GPU3--7 并发 | 五基线；新布局的单 pair 对齐 |
| `20260918_183809/` | 5（动物） | 每条基线各自 source、GPU3--7 并发 | 五基线；均值表见 [docs/ACCEPTANCE.md](../docs/ACCEPTANCE.md) |
| `20260919_114625/` | 5（非动物） | 每条基线各自 source、GPU3--7 并发 | 五基线；prompt 为 `configs/experiments/acceptance_prompt_pairs_nonanimal.jsonl` |

编排入口是 [scripts/run/run_acceptance_experiment.py](../scripts/run/run_acceptance_experiment.py)，
协议见 [docs/ACCEPTANCE_MULTI_PROMPT_RUNNER_PLAN.md](../docs/ACCEPTANCE_MULTI_PROMPT_RUNNER_PLAN.md)。
`*.carnator_boundary_b5_targetresume.{json,md}` 兄弟文件来自
[scripts/run/run_carnator_target_latent_resume.py](../scripts/run/run_carnator_target_latent_resume.py)
补跑的 target 前缀变体，不是第二轮验收。

## Carnator 消融

| 目录 | 内容 |
| --- | --- |
| `ablation_req16_20260909_100759/` | 论文 Table 3 的四变体（latent / attention / joint），见 [docs/CARNATOR_ABLATION_STUDY.md](../docs/CARNATOR_ABLATION_STUDY.md) |
| `pair1_carnator_ablation_20260915/` | pair-1 单请求消融，产物含 `FINDINGS.md` 与 `run_all.sh` |
| `pair16_carnator_layer_window_20260917_142119/` | sparse-attention 层窗口扫描（论文附录），聚合表在 `layer_window_ablation.{csv,json}` |

## 失败案例与图表

| 目录 | 内容 |
| --- | --- |
| `chorus_mbase_failure_case_v5/` | Chorus 苹果/桃子背景错配失败案例，见 [docs/CHORUS_MBASE_FAILURE_CASE.md](../docs/CHORUS_MBASE_FAILURE_CASE.md) |
| `chorus_medit_failure_case_v1/` | 同一失败案例的 `--update-mask Medit` 变体 |
| `chorus_counterexample_p1_v2_halo2/` | 强制复用下的两对反例 prompt 与四行对比图，见 [docs/CHORUS_COUNTEREXAMPLE_P1.md](../docs/CHORUS_COUNTEREXAMPLE_P1.md) |
| `seedream-comparison-images/` | 定性对比图 |
| `fusion_mask_experiment_20260909_221704/` | fusion mask 重构的验证产物 |

## 长期保留

- `shared_cross_request_cache/`：当前缓存基于 `datasets/1000cache_300input_noun_match/` 的 1000 条预置请求，包含 Early Signature。
- `vidprom_1000cache_300input_noun_match/`：当前 300-request 实验的配对、生成、质量评估和汇总结果，Carnator 使用 v2 的 noun -> Early Signature -> CLIP 检索顺序。
- `early_signature_color_v1_calibration_cluster_02_2170/`：当前校准实验，不能按常规验收缓存清理。
- `static-prompt-defaults-results/`：静态 prompt 默认配置的结果；对应的 17GB 生成产物 `static-prompt-defaults/` 与 `static-prompt-defaults-seed42/` 只留本地。
- `cross_request_retrieval/`：检索产物。
- `cogvideox15_5baselines_10samples/`：CogVideoX1.5 的 10 样本适配产物。
- `vbench-cache/`：质量评测模型缓存。
- `cache/synthetic/`：测试夹具，不用于验收结果。

`*_experiment_logs/`、`logs/`、`mask_visualization_carnator/` 和空的 `runs/`、`ti2v_5b/`
是过程日志与诊断渲染，不入库。

历史 Wan 基线缓存、历史 Carnator 结果以及早期重复 Chorus2 运行已移动到系统回收站的
`VIDEO_CACHE_BASELINES-cleanup-20260824/`，未永久删除。
