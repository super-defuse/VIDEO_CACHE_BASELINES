# Carnator Early Signature gate 标定

四个 gate 上界（context color CIEDE2000、`log(area_q/area_s)`、`1 - Dice`、changed color
CIEDE2000）全部离线标定：先采集 signature 与质量指标，再扫描阈值元组，扫描不重新生成任何视频。
下面三套数据集共用这一口径，差别在 pair 来源、采集内容和代价。

| 数据集 | pair 来源 | 采集内容 | 状态 |
| --- | --- | --- | --- |
| cluster 02 60 对 | `artifacts/early_signature_color_v1_calibration_cluster_02_2170/pairs.csv`（冻结，不得重采样） | 完整 50-step source + 强制复用 target 视频 + VBench/CLIP-T | 已执行；未选出正式 gate |
| 300 arrival step-5 signature | `cluster_02_2170_random300_input.csv`，回放 `cogvideox15_replay` 的命中决策 | 每条只跑 5 步 dense probe 的 signature，无视频、无 latent reuse | 已执行；20736 组扫描 |
| forced reuse v2（300） | 300 条输入经 noun filter + CLIP 最近邻 | 完整 50-step 强制复用 + VBench/CLIP-T | 仅协议，未执行 |

## 1. cluster 02 60 对

冻结 `pairs.csv` 含 60 对、120 条不同 prompt，全部为单 noun substitution。所有运行固定同一
checkpoint、50-step scheduler、CFG、seed、分辨率、帧数、mask policy 和 `reuse_start_step=10`；
Early Signature 在**第 5 个 completed step** 采集，scheduler 总步数不得缩短。

- **source（60 条）**：完整跑完 50 步。第 5 步用 cross-attention seed 与 self-attention contrast
  融合出 `Mbase` 并保存 signature 与 tail frame；第 10 步保存 joint reuse 所需 latent、sparse-attention
  artifact 及 hash。尾部空切片只补最近非空切片，不降阈值；整条时间轴为空则 fail-closed，不写空 mask。
  进入 target 前逐条校验视频、cache、hash 与 metadata，未通过者不得被使用。
- **target（60 条）**：按同一 `pair_id` 取已校验 source cache，**跳过在线 recall 与 gate**，从
  step-10 artifact 执行真实 joint latent/sparse-attention reuse 至视频输出。每条单独计算 `CLIP-T`
  和六个 VBench 维度（`subject_consistency`、`background_consistency`、`motion_smoothness`、
  `aesthetic_quality`、`imaging_quality`、`dynamic_degree`）及 `vbench_q_i`；不得把批量平均
  `VBench-Q` 写进全部 60 条记录。

```bash
python3 scripts/benchmark/run_carnator_early_signature_calibration.py --pairs <pairs.csv> --run-root <run-root> --gpus <gpu 列表>
python3 scripts/benchmark/score_carnator_early_signature_calibration.py --run-root <run-root> --pairs <pairs.csv> --gpus <gpu 列表>
```

实测（`runs/color_v1_full_source_target_20260823/quality/`）：`Q_min=0.60`、`CLIP-T_min=0.30`、
`C_min=0.90`、confidence `0.95`，`L_min` 未预注册。VBench-Q 均值 `0.6457`、CLIP-T 均值 `0.2722`，
质量成功 pair 10/60；扫描 1764 组，其中 368 组满足 coverage，coverage ≥ 0.9 时最好的单侧 95%
Clopper-Pearson 下界为 `0.104`（10/54）。**未选出正式 gate**：manifest 未在生成前冻结 `L_min`，
事后补选会破坏既定选择规则，`report.md` 只给出并列最优的诊断性代表组。

这 60 对仅用于 calibration。选定阈值后须在独立、预先冻结的配对集上重算统计区间，才可报告泛化结论。

## 2. 300 arrival step-5 signature 数据集

正式实验每次命中都要重跑整个 Carnator source（50 步轨迹、融合 mask 模板、外层路由 mask、
source 视频），成本极高；而拟合 gate 只需要第 5 步的 signature。该数据集因此对每条 arrival 只跑
两个 5 步 dense prefix——回放选中的 source prompt 和 arrival 的 target prompt——用与正式
source/target 相同的 `carnator.adapters.wan21_native.run_wan21_target_probe` 采集。

```bash
# 300 条，GPU 3-7 分片，跑完自动 --summarize；可中断续跑
bash scripts/benchmark/run_wan21_carnator_step5_signatures.sh
```

产物在 `artifacts/vidprom_1000cache_300input_noun_match/wan21_carnator_step5/`：`arrivals/` 下每个
`pair.json` 记录四条 gate 距离与名词替换情况，`quality/` 汇总为 `step5_pair_distances.{csv,jsonl}`
和 `step5_hit_rate_sweep.csv`（20736 组）。实测 300 条全部 `completed`，`one_noun_substitution` 297、
`same_nouns` 3，因此全部计入可比口径。

口径边界：该数据集没有复用 latent、也不产出视频，`cache_ready` 恒为 false，**不能报告真实 cache
hit**；命中/未命中真值仍只由回放文件提供。它给出的是任意阈值元组下的 gate 接受率，足以离线重算
命中率而不重新生成。`run_manifest.json` 记录其采集模型为 **Wan2.1-T2V-1.3B**、回放文件为
`cogvideox15_replay`；把它当作 Wan2.2-TI2V-5B 实验的阈值证据时须显式声明这一跨模型口径。

## 3. 300 条强制复用标定（协议，未执行）

独立于正式 300-request baseline，使用 noun-match 数据集和 Carnator v2 的名词过滤顺序，输出默认位于
`artifacts/vidprom_1000cache_300input_noun_match/forced_early_signature_calibration_v2/`（该目录尚未
产生）。协议：

1. 逐条读取 300 条输入 prompt，在只读 source view 中先保留相同名词集合或单名词替换的 source，
   再用 CLIP 取最高相似度候选，排除 UUID 相同的自匹配；每个 target 只形成一个 pair。
2. source 使用已有融合 cache，且必须有每个 source noun 的 step-5 Mbase；新目录只建 symlink，不改写共享 cache。
3. target 完整跑 50 步 scheduler：第 5 步保存 fused Early Signature；单 noun 替换取 source-only
   Mbase，同 noun 集合取 token 面积最小的 Mbase，写入 target 私有 `selected_source_mask/`；
   第 10 步起用该 bundle 完成 latent reuse。不保存前五步的其它中间缓存。
4. 统一计算每条复用视频的 CLIP-T 与 VBench-Q，之后离线扫描质量下限与四个距离阈值。

```bash
cd /home/tangxuebin/project/VIDEO_CACHE_BASELINES
S=scripts/benchmark/run_vidprom_forced_early_signature_calibration.py
R=artifacts/vidprom_1000cache_300input_noun_match/forced_early_signature_calibration_v2

python3 $S pair --run-root $R --noun-filter-first --clip-device cuda:3   # pairs.csv + manifest + symlink
python3 $S run  --run-root $R --gpus 3,4,5,6,7                          # 300 个 forced target probe + reuse
python3 $S score --run-root $R --gpus 0,1,2,3,4,5,6,7                   # CLIP-T + VBench-Q
python3 $S scan --run-root $R --q-min <VBench-Q 下限> --clip-t-min <CLIP-T 下限>
```

`scan` 沿用同一套 Clopper–Pearson 规则，写入 `quality/forced_threshold_sweep.csv`、
`forced_threshold_sweep_details.jsonl` 和 `forced_scan_report.md`。`coverage-min` 默认 `0.90`，质量
下限必须在看到结果后由实验负责人指定。same-noun pair 的不适用距离按 0 处理；noun-set 差异大于一个的
pair 保留在审计文件中但不参与四维 gate 拟合。默认 `--coverage-denominator all` 要求覆盖全部请求；
`--coverage-denominator comparable` 只对可比 pair 做条件统计，结果带 `_comparable` 后缀，不能解释为
全部请求的覆盖率。最近邻强制配对测量的是 Early Signature 的候选上限，不等价于正式 Carnator 检索命中率。

## 统计判定规则（三套共用）

对每个阈值元组 `h` 计算接受集合 `A_h`、`n_h=|A_h|`、coverage 与成功数 `k_h`；成功定义为该 pair
同时满足 `vbench_q_i >= Q_min` 且 `clip_t_i >= T_min`。报告成功率的单侧 Clopper-Pearson 下界与双侧
95% 区间；只有 coverage 不低于 `C_min` 且单侧下界达到**生成前**冻结的 `L_min` 的候选可入选。未预注册
`L_min` 时只能报告诊断性 sweep，不得据此正式选择 gate。合格候选中取 coverage 最大者，coverage 相同时
取更严格的阈值。

## 现行阈值及其来源

`configs/methods/default.yaml` 的 carnator 段冻结了
`2.5460641772840114` / `1.6755377126043391` / `1.0` / `2.2003341853972085`，各 run 的
`run_config.json` 与请求元数据都回显该元组。其可复现的旁证只有第 2 节的 300 条 signature 数据集：
按该元组逐条重算 `quality/step5_pair_distances.csv`，300 对中通过 272 对（changed color 拒 13、
context color 拒 12、log area 拒 3），coverage `0.9067`。注意 `step5_summary.json` 记录
`default_thresholds_in_sweep: null`，即该元组不在 20736 组扫描网格内，标定时的原始扫描输出未随
artifact 保留；cluster 02 的 60 对标定则明确 withheld。引用该元组时应连同这两点一并说明。
