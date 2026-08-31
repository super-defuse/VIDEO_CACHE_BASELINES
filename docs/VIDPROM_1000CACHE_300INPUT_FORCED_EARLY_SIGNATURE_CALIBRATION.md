# VidProM 300 条强制复用 Early Signature 标定

这是一个独立于正式 300-request baseline 的标定实验，使用 noun-match 数据集和 Carnator v2 的名词过滤顺序。输出默认位于：

```text
artifacts/vidprom_1000cache_300input_noun_match/forced_early_signature_calibration/
```

## 协议

1. 从随机 300 条输入逐条读取 prompt，在只读的 923-entry source view 中先保留相同名词集合或单名词替换的 source，再用 CLIP 选择最高相似度候选；排除 UUID 相同的自匹配。每个 target 只形成一个 pair，记录相似度和 `pair_source`。
2. source 使用已有的、经过 self-attention/cross-attention 融合的 step-10 cache。新目录只建立 source symlink，不复制或改写共享 cache。
3. 对每个 pair 运行 target 的完整 50-step scheduler：第 5 步保存 fused Early Signature，第 10 步开始直接使用对应 source cache 完成 latent reuse。target 不保存前五步的其它中间缓存。
4. 统一计算每条复用视频的 CLIP-T 和 VBench-Q；之后离线扫描质量下限和四个 Early Signature 距离阈值。扫描不重新生成视频。

## 运行顺序

以下命令只作为正式启动时的入口；本次准备阶段不会自动执行它们。

```bash
cd /home/tangxuebin/project/VIDEO_CACHE_BASELINES

# 1. 生成 pairs.csv、实验 manifest 和 source symlink
python3 scripts/benchmark/run_vidprom_forced_early_signature_calibration.py pair \
  --run-root artifacts/vidprom_1000cache_300input_noun_match/forced_early_signature_calibration \
  --noun-filter-first \
  --clip-device cuda:3

# 2. GPU3--7 并行执行 300 个 forced target probe + reuse
python3 scripts/benchmark/run_vidprom_forced_early_signature_calibration.py run \
  --run-root artifacts/vidprom_1000cache_300input_noun_match/forced_early_signature_calibration \
  --gpus 3,4,5,6,7

# 3. 统一质量评估（按机器上可用 GPU 调整 --gpus）
python3 scripts/benchmark/run_vidprom_forced_early_signature_calibration.py score \
  --run-root artifacts/vidprom_1000cache_300input_noun_match/forced_early_signature_calibration \
  --gpus 0,1,2,3,4,5,6,7

# 4. 给定 CLIP-T/VBench-Q 下限后，扫描四维 gate
python3 scripts/benchmark/run_vidprom_forced_early_signature_calibration.py scan \
  --run-root artifacts/vidprom_1000cache_300input_noun_match/forced_early_signature_calibration \
  --q-min <VBench-Q下限> --clip-t-min <CLIP-T下限>
```

`scan` 使用同一套 Clopper–Pearson 统计规则，并把结果写入 `quality/forced_threshold_sweep.csv`、`quality/forced_threshold_sweep_details.jsonl` 和 `quality/forced_scan_report.md`。`coverage-min` 默认 0.90，质量下限必须在看到结果后由实验负责人指定。same-noun pair 的不适用距离按 0 处理；noun-set 差异大于一个的 pair 会保留在审计文件中，但不参与四维 gate 拟合。默认 `--coverage-denominator all` 要求覆盖全部请求；若最近邻 pair 中存在较多多 noun 替换，可用 `--coverage-denominator comparable` 对可比 pair 做条件统计，结果带 `_comparable` 后缀，不能解释为全部请求的覆盖率。最近邻强制配对用于测量 Early Signature 的候选上限，不等价于正式 Carnator 检索命中率。
