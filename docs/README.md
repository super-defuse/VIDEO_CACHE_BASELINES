# 文档索引

| 文档 | 职责 |
| --- | --- |
| [METHODS.md](METHODS.md) | 已实现方法、能力边界与模型支持矩阵。 |
| [ADAPTERS.md](ADAPTERS.md) | Adapter 统一接口与模型接线。 |
| [CROSS_REQUEST_RETRIEVAL.md](CROSS_REQUEST_RETRIEVAL.md) | 跨请求检索方法对齐、回放基准和 Carnator 在线 gate 的规范。 |
| [CROSS_REQUEST_CACHE_PREPARATION_PLAN.md](CROSS_REQUEST_CACHE_PREPARATION_PLAN.md) | VidProM 前 1000 条 source 的八卡共享缓存准备脚本规划。 |
| [CARNATOR_EARLY_SIGNATURE_CALIBRATION_PLAN.md](CARNATOR_EARLY_SIGNATURE_CALIBRATION_PLAN.md) | Early Signature gate 的离线标定、质量评分和统计判定。 |
| [CHORUS_REPRODUCTION_PLAN.md](CHORUS_REPRODUCTION_PLAN.md) | Chorus 的实现范围、固定依赖和复现约束。 |
| [EVALUATION.md](EVALUATION.md) | 结果记录和性能/质量评测口径。 |
| [ACCEPTANCE.md](ACCEPTANCE.md) | 已完成运行的验收结果、产物索引及定性分析。 |
| [PROVENANCE.md](PROVENANCE.md) | 仓库内实现的冻结来源。 |

`Chorus.pdf` 和 `CHAI.pdf` 是参考论文，不是本仓库的规范性说明。新文档应更新对应的
单一职责文件，避免重复记录实现细节或实跑结果。
