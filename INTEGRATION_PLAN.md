# Video Cache Baselines 集成计划

## 结论

可以新建一个独立的开源仓库，统一承载 NIRVANA-VID、TeaCache、CHAI、Chorus 和 Carnator，并以 Wan2.2、CogVideoX1.5 为共同模型入口。

当前本地已有的是四类实现：`project/CHAI` 中的 NIRVANA-VID、TeaCache、CHAI，以及 `project/CARNATOR_ICLR2027` 中的 Carnator。Chorus 在当前两个目录中没有可直接迁移的完整实现，不能把 Carnator 中对 Chorus2 的引用当作 Chorus 实现。

集成应采用“统一运行协议 + 方法插件 + 模型适配器”，而不是把六套代码直接复制到一个 Python 包中。原始实现、上游提交和许可证信息必须保留可追溯记录。

## 目标目录

建议新建 `project/VIDEO_CACHE_BASELINES/`，作为最终 Git 仓库：

```text
VIDEO_CACHE_BASELINES/
├── README.md                         # 安装、快速开始、支持矩阵
├── pyproject.toml                    # 公共工具与 CLI
├── external.lock.json               # 模型/方法/评测工具的固定提交
├── methods/
│   ├── nirvana_vid/
│   ├── teacache/
│   ├── chai/
│   ├── chorus/
│   └── carnator/
├── adapters/
│   ├── wan22.py                     # Wan2.2 T2V/TI2V 统一边界
│   └── cogvideox15.py               # CogVideoX1.5 T2V 统一边界
├── core/                             # 方法注册、请求、缓存、日志、结果协议
├── configs/{models,methods,experiments}/
├── scripts/{run,benchmark,aggregate}/
├── tests/
├── docs/{METHODS,ADAPTERS,EVALUATION}.md
└── artifacts/                        # 只提交 manifest/汇总表，不提交权重和视频
```

`methods/` 只保存算法逻辑；模型调用、采样循环、latent/attention hook 放在 `adapters/`。模型源代码和大型评测依赖使用 submodule 或外部安装，不复制进仓库。

## 统一协议

实现一个最小的 `BaselineMethod` 接口，至少包含：

- `prepare(model_adapter, config)`、`reset_cache()`、`generate(request)`；
- 方法能力声明：跨请求缓存、步跳过、Cache Attention、稀疏 attention、量化、是否需要 source trajectory；
- 统一的 `RequestResult`：prompt UUID、model、method、seed、命中类型/来源、执行步数、端到端延迟、缓存大小、输出路径、质量输入路径和错误状态。

每个方法使用独立的缓存 namespace、配置和运行环境，禁止共享 latent、索引或临时文件。关闭方法 hook 时，必须能回退到同一模型的 dense baseline。

## 适配矩阵与优先级

最终目标是 6 个方法 × 2 个模型的显式支持矩阵；暂时不支持的单元格必须在 README 和结果表中标为 `N/A`，不能静默跳过。

| 方法 | 当前来源 | Wan2.2 | CogVideoX1.5 | 主要工作 |
| --- | --- | --- | --- | --- |
| NIRVANA-VID | `CHAI/nirvana-strawman` | 已验收 | 已验收 | adapter 内跨请求 latent restart |
| TeaCache | `CHAI/teacache` | 已验收 | 已验收 | adapter 内 transformer 输出复用与固定阈值 |
| CHAI | `CHAI/chai` | 已验收 | 已验收 | 迁移 `chai/runtime`、cache、retrieval 和验收脚本 |
| Chorus | 外部上游/论文 | 待实现 | 待实现 | 根据 Chorus 论文进行复现 |
| Carnator | `CARNATOR_ICLR2027/carnator` | C0/B0--B5 已验收 | source-latent-restart adapter 已验收 | Wan 保留原生 C0/B0--B5；CogVideoX1.5 明确标注为 adapter-level source trajectory，不宣称 Wan 专用 mask/FFN 等价路径 |

建议顺序：先迁移 CHAI 的公共运行骨架和两个模型 adapter，再迁移 Carnator；随后完成 TeaCache、NIRVANA-VID，最后引入 Chorus。每接入一个方法，先通过 synthetic adapter 和 CPU 协议测试，再运行真实模型。

## 迁移阶段

1. **冻结来源**：记录四个现有实现的 Git commit；确认 Chorus 的官方仓库、版本、许可证和论文配置；建立 `external.lock.json`。
2. **建立骨架**：创建公共 CLI、配置 schema、方法注册表、模型 adapter 接口、结果 manifest 和缓存隔离规则；迁移现有 Wan/CogVideo CHAI acceptance run。
3. **迁移已存在实现**：将 NIRVANA-VID、TeaCache、CHAI、Carnator 逐个放入 `methods/`，去除相对路径和 OpenSora 专用的隐式全局状态；保留原仓库 attribution 和必要测试。
4. **完成模型适配**：为每个方法实现 Wan2.2 与 CogVideoX1.5 的 hook；统一 CFG、时间步、latent shape、attention block 编号和 seed 语义；对不兼容的实现记录明确限制。
5. **补齐 Chorus**：优先复用官方代码；若只有论文描述，则单独实现并在文档中标明 reimplementation，不与官方结果混称。
6. **统一评测**：用同一批 prompt、分辨率、帧数、seed 和硬件运行 dense baseline 及六种方法；冷缓存写入和热缓存命中至少各跑一轮。
7. **开源整理**：补齐 README、方法卡片、环境安装、许可证汇总、故障排查和最小可复现实验；确认权重、视频、缓存和私有路径均被 `.gitignore` 排除。

## 评测与验收

- **性能**：端到端 latency、命中 latency、实际执行步数、Hit Rate、整体 Speedup、`Speedup(hit)`、峰值显存和缓存存储开销。
- **质量**：CLIP-T、VBench-Q；同一请求的 baseline/加速输出保留 hash，并按模型定义的容差检查。
- **可复现**：每条结果记录代码 commit、上游 commit、模型标识/校验和、数据集版本、resolved config、GPU、seed 和错误状态；汇总表只能由逐请求 JSONL 自动生成。
- **正确性**：两个模型各自可运行 dense baseline；每个已声明支持的方法均可冷启动、写缓存、热命中和清空缓存；关闭 hook 时输出与 baseline 一致；方法之间缓存完全隔离。
- **回归**：现有 CHAI OpenSora 路径和 Carnator Wan 路径不回归，CPU 协议测试与 adapter smoke test 在 CI 中通过。

## 注意事项

1. 先不用实现 Chorus。
2. Carnator 的 Wan 只声明上游已有的 C0/B0--B5；CogVideoX1.5 使用单独的 source-latent-restart adapter，不能与 Wan C0/B0--B5 结果混称。
4. 只考虑支持 T2V；I2V、OpenSora 等相关适配不要出现在仓库中。
