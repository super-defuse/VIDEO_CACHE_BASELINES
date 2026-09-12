# 文档索引

| 文档 | 职责 |
| --- | --- |
| [METHODS.md](METHODS.md) | 已实现方法、能力边界与模型支持矩阵。 |
| [ADAPTERS.md](ADAPTERS.md) | Adapter 统一接口与模型接线。 |
| [CROSS_REQUEST_RETRIEVAL.md](CROSS_REQUEST_RETRIEVAL.md) | 跨请求检索方法对齐、回放基准和 Carnator 在线 gate 的规范。 |
| [CROSS_REQUEST_CACHE_PREPARATION_PLAN.md](CROSS_REQUEST_CACHE_PREPARATION_PLAN.md) | VidProM 前 1000 条 source 的八卡共享缓存准备脚本规划。 |
| [CARNATOR_EARLY_SIGNATURE_CALIBRATION_PLAN.md](CARNATOR_EARLY_SIGNATURE_CALIBRATION_PLAN.md) | Early Signature gate 的离线标定、质量评分和统计判定。 |
| [VIDPROM_1000CACHE_300INPUT_EXPERIMENT_PLAN.md](VIDPROM_1000CACHE_300INPUT_EXPERIMENT_PLAN.md) | 923 条 source cache、300 条新请求、五 GPU 基线实验的唯一执行计划。 |
| [VIDPROM_1000CACHE_300INPUT_RESULTS_TABLE.md](VIDPROM_1000CACHE_300INPUT_RESULTS_TABLE.md) | 300 条请求的汇总表字段和时延/质量口径。 |
| [VIDPROM_1000CACHE_300INPUT_FORCED_EARLY_SIGNATURE_CALIBRATION.md](VIDPROM_1000CACHE_300INPUT_FORCED_EARLY_SIGNATURE_CALIBRATION.md) | 独立于正式 300 条的强制复用 Early Signature 标定实验协议。 |
| [CHORUS2.md](CHORUS2.md) | Chorus2 的 SpargeAttn 稀疏 attention 基线与运行入口。 |
| [CHORUS_REPRODUCTION_PLAN.md](CHORUS_REPRODUCTION_PLAN.md) | Chorus 的实现范围、固定依赖和复现约束。 |
| [CHORUS_MBASE_FAILURE_CASE.md](CHORUS_MBASE_FAILURE_CASE.md) | 独立的苹果/桃子背景错配失败案例实验和运行产物约定。 |
| [EVALUATION.md](EVALUATION.md) | 结果记录和性能/质量评测口径。 |
| [ACCEPTANCE.md](ACCEPTANCE.md) | 已完成运行的验收结果、产物索引及定性分析。 |
| [ACCEPTANCE_MULTI_PROMPT_RUNNER_PLAN.md](ACCEPTANCE_MULTI_PROMPT_RUNNER_PLAN.md) | 六基线多 Prompt 统一编排验收脚本的规划。 |
| [PROVENANCE.md](PROVENANCE.md) | 仓库内实现的冻结来源。 |
| [WAN21.md](WAN21.md) | Wan2.1-T2V-1.3B 上五个基线的适配与运行入口。 |
| [WAN21_AGGREGATOR_AND_SOURCE_OPTIMIZATION_PLAN.md](WAN21_AGGREGATOR_AND_SOURCE_OPTIMIZATION_PLAN.md) | Wan2.1 聚合器修正与 Chorus2/Carnator source 优化的实施与验收记录。 |
| [LTXVIDEO.md](LTXVIDEO.md) | LTX-Video 0.9.7-dev 上五个基线的适配规划、与 Wan2.1 的第二轮对齐修正，以及验证结果。 |
| [CARNATOR_ABLATION_STUDY.md](CARNATOR_ABLATION_STUDY.md) | Carnator 消融实验（论文 Table 3）：设计、结果与分析。 |
| [cogvideox/](cogvideox/README.md) | CogVideoX1.5 适配：实验计划、工作流、验证状态与 fusion mask 重构。 |

`Chorus.pdf` 和 `CHAI.pdf` 是参考论文，不是本仓库的规范性说明。新文档应更新对应的
单一职责文件，避免重复记录实现细节或实跑结果。

## 实验结果文档的写法

一份实验结果只在一处记录：设计口径写进计划文档，实跑数据写进结果文档或 `artifacts/`
下的 `summary.json` / `report.md`。中间过程状态（"正在运行"、"已启动"、"等待确认"）
不单独成文——完成后直接并入最终结果，或写入提交信息。历史上 `ABLATION_*.md` 曾有 7 份
同内容迭代文档，已合并为 [CARNATOR_ABLATION_STUDY.md](CARNATOR_ABLATION_STUDY.md)。
