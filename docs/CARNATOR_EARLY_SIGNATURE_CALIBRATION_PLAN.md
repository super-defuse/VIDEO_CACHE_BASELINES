# Carnator Early Signature 标定与统计流程

## 目的

在冻结的 60 个 `(cached, target)` prompt 对上，先准备完整的 historical source cache，再让
每个 target 强制复用其对应 source cache。对 60 个 target 视频逐条计算 `VBench-Q` 和 `CLIP-T`，
最后离线选择 Early Signature gate 超参数。阈值搜索不重新生成任何视频。

使用冻结的 cluster 02 `pairs.csv`，不得重新采样或替换 pair。它含 60 对、120 条不同 prompt；
所有 pair 均为单 noun substitution。

所有运行固定同一模型 checkpoint、50-step scheduler、CFG、seed、分辨率、帧数、mask policy 和
`reuse_start_step=10`。Early Signature 在第 5 个 completed step 采集，scheduler 总步数不得缩短。

## 1. Historical Source（60 条）

对每个 `cached_prompt` 完整运行 50-step scheduler：

- 第 5 步保存 Early Signature、`Mbase` 和 tail frame；
- 第 10 步保存 joint reuse 所需 latent、sparse-attention artifact 及 hash；
- 继续运行至视频输出，并持久化 source 视频、可执行 cache 和运行 manifest。

在开始 target 前，逐条验证 source 视频和 cache 存在、cache 可读、artifact hash 正确且 metadata
兼容。未通过校验的 source 不得被 target 使用。

## 2. Target Reuse 与质量评分（60 条）

每个 `target_prompt` 必须使用 `pairs.csv` 中同一 `pair_id` 指定的、已通过校验的 source cache：跳过
在线 recall 和 gate，从 source 的 step-10 artifact 执行真实 joint latent/sparse-attention reuse 至视频输出。

每条 target 必须持久化：

- target 视频；
- target 复用执行 cache（所用 source cache 的引用、`Medit`、`Msee` 和运行 manifest）；
- 第 5 步的 `online_color_signature.json`、`step5_tail_frame.png`、`mbase.safetensors`。

对每条 target 视频及其 target prompt 单独计算并记录：

- `CLIP-T`；
- 六个 VBench 维度 `subject_consistency`、`background_consistency`、`motion_smoothness`、
  `aesthetic_quality`、`imaging_quality`、`dynamic_degree`；
- 基于该条视频六维度的 `vbench_q_i`。

不得将批量平均 `VBench-Q` 写入全部 60 条记录。

## 3. 超参数判定

在跑 target 前将 `Q_min`、`T_min`、最小 coverage `C_min`、置信度和单侧 CP 下界目标 `L_min` 写入 manifest。对每个
source/target sidecar 先无阈值运行一次 `compare_early_signatures`，保存完整距离；此后仅离线
搜索下列 gate 上界，不再运行生成：

| 参数 | 距离 |
| --- | --- |
| `max_context_color_distance` | 共享 noun 的最大 CIEDE2000 |
| `max_log_area_ratio` | `abs(log(area_q / area_s))` |
| `max_shape_distance` | `1 - Dice` |
| `max_changed_color_distance` | changed noun CIEDE2000 |

对每个阈值元组 `h`，计算接受集合 `A_h`、`n_h=|A_h|`、coverage `n_h/60`，以及成功数
`k_h`。成功定义为该 pair 同时满足 `vbench_q_i >= Q_min` 和 `clip_t_i >= T_min`。报告成功率的
单侧 Clopper-Pearson 下界及双侧 95% 区间；只有 coverage 不低于 `C_min` 且单侧下界达到预先
冻结的 `L_min` 的候选可入选。未在 target 生成前冻结 `L_min` 时，只能报告诊断性 sweep，不得据此正式选择
gate。在合格候选中选择 coverage 最大者，coverage 相同时选择更严格的阈值。

这 60 对仅用于 calibration。选定阈值后，须在独立、预先冻结的配对集上重新计算统计区间，才可
报告泛化结论。

## 产物

- 60 个完整 source 运行目录：视频、cache、early signature、step-10 artifact 和 hash；
- 60 个 target 运行目录：视频、复用执行 cache、early signature 和运行 manifest；
- `pair_results.jsonl`：每 pair 的 source/target artifact 路径、四个距离、六维 VBench、`vbench_q_i`、
  `clip_t_i` 和成功标签；
- `threshold_sweep.csv`：每组超参数的 `n,k,coverage,CP`、接受/拒绝 pair 与失败 gate；
- `report.md`：冻结配置和阈值、统计结果、失败样本、环境与 artifact hash。
