# VidProM 1000-cache / 300-request 实验计划

本文描述当前 `1000cache_300input_noun_match` 实验，结果位于
`artifacts/vidprom_1000cache_300input_noun_match/`，Carnator 使用 v2（noun ->
Early Signature -> CLIP top-1），并以 source per-noun Mbase 进行复用。

本轮实验固定使用 Wan2.2-TI2V-5B、50-step、`1280x704`、121 帧和同一 seed/config。
输入缓存源是 `/home/tangxuebin/datasets/1000cache_300input_noun_match/cluster_02_2170_first1000_cached.csv`（机器本地数据集，不入库；脚本默认值即该绝对路径）中的 1000 条请求；新请求是 `cluster_02_2170_random300_input.csv`，按 CSV 顺序到达。GPU0--2 保留不用，五个基线各占一张 GPU：

| GPU | 基线 | 说明 |
| ---: | --- | --- |
| 3 | NIRVANA-VID | 完整 prompt CLIP/Chroma Top-1，阈值 `0.65`，命中后从 completed step 5 latent restart，target 执行后 45 步 |
| 4 | TeaCache | 仅目标请求的 intra-request step skipping，不读取共享跨请求 cache；hit rate 记为 N/A |
| 5 | Chorus | 完整 prompt CLIP/Chroma Top-1，阈值 `0.65`，使用 source trajectory、K1/K2=`5/30` 与 TGAA，target 执行后 45 步 |
| 6 | Chorus2 | 暂无独立检索器；与 NIRVANA-VID/Chorus 共用完整 prompt CLIP/Chroma Top-1（同一排序与阈值），target 完整执行 50 步；第 11--50 步使用 source raw block mask 的普通 SpargeAttn，取消 source latent restart |
| 7 | Carnator Boundary B5 | noun prefilter、step-5 Early Signature gate、CLIP top-1；命中后选择 source-only noun mask，或同 noun 集合中的最小 Mbase，再执行固定 Boundary B5（SpargeAttn + W8A8） |

五个基线各自使用一张 GPU，按全局 `arrival_index=0..299` 顺序处理；如为并行执行拆分 arrival shard，必须保留原始索引并在聚合前合并 JSONL，不改变请求协议。每条 Carnator Early Signature 的 Mbase 固定为
step-5 cross-attention seed 与 self-attention contrast 的融合结果，不能使用仅 cross-attention mask；source
为每个 noun 保存独立 Mbase。尾部空切片仅复制最近非空融合切片，不降低阈值；整个时间轴为空、probe
未通过或 source 无有效 per-noun mask 时 fail-closed，记录 miss，不执行 dense generation。

实验固定使用与 shared cache `run_manifest.config.seed` 相同的 `seed=20260218`。Carnator
Early Signature 四个 gate 阈值（依次为 context color、log area、shape、changed color）固定为
`2.5460641772840114`、`1.6755377126043391`、`1.0`、`2.2003341853972085`；按该元组重算 300 条
step-5 signature 可比 pair，通过 272 条（coverage `0.9067`）。该元组不在 20736 组扫描网格内，
标定原始输出未随 artifact 保留；阈值来源、跨模型口径与统计规则见
[CARNATOR_EARLY_SIGNATURE_CALIBRATION.md](CARNATOR_EARLY_SIGNATURE_CALIBRATION.md)。

## 1. 缓存与数据边界

原始 1000-entry cache 位于
`artifacts/shared_cross_request_cache/vidprom_cluster_02_2170_first1000/`，不删除、不改写。
根据 1000-row CSV 已生成只读视图：

```text
artifacts/vidprom_1000cache_300input_noun_match/source_view/
  entries/<uuid> -> ../../entries/<uuid> # relative symlink, no tensor/video copy
  index.jsonl                             # only the 1000 selected UUIDs
  source_requests.csv                     # exact selection CSV snapshot
  excluded_requests.csv                  # 77 screened-out UUIDs (for audit only)
  view_manifest.json                      # counts, hashes, provenance
```

`source_requests.csv` 和 `excluded_requests.csv` 是可重建的本地审计快照，不纳入版本库；选择集哈希保存在 `view_manifest.json`。

实验 runner 只把该 view 的 `index.jsonl` 建成可检索的 source metadata，并把命中的 UUID
解析到 `entries/<uuid>`；绝不把新请求生成结果写回 shared cache。300 条新请求的产物、逐请求
JSONL 和视频统一写到各自的 run directory。TeaCache 使用独立的 target-only run directory；Carnator
命中的 `selected_source_mask/` 也仅在该请求目录中创建。

每个方法从同一份 1000-entry source pool 开始、各自维护检索状态；300 条请求的到达顺序不能
shuffle。为便于复现实验，runner 应保存 resolved config、GPU、git revision、输入 hash、cache
view manifest hash 和完整命中决策。

## 2. 当前脚本的问题

- generic runner 和 native runner 的计时起点不同；native 请求计时覆盖检索、probe 和部分 cache 准备，导致同名 `latency_s` 的组成不完全一致。
- worker warm-up、Carnator probe、source mask `prewarm` 和请求内缓存读取分散在不同函数中，缺少统一的准备阶段和排除规则。
- forced-smoke 覆盖正常检索结果，且与正式 300-request 运行的自然命中规则混在同一 runner 中，容易把协议验证和正式 benchmark 混淆。
- 未命中时回退到 Dense C0，既增加不必要的生成时间，也使“命中基线耗时”和“全请求耗时”难以比较。
- 复用步数和总步数分散在配置、runner 和 native adapter 中，容易出现文档、命令和结果不一致。

## 3. 统一请求生命周期

所有基线使用同一生命周期；指标名称保持现有定义不变。

1. **Worker warm-up（一次/worker）**：加载模型、编译 kernel、建立 CUDA allocator 和必要的 baseline runtime。使用固定 dog/cat prompt，不写入 shared cache，不进入任何请求时间。
2. **请求到达与检索**：请求到达即启动 wall-clock，记录 `arrival_index`、UUID 和 prompt，执行该基线的自然检索并保存候选、阈值、命中对象和 miss 原因。检索时间属于端到端时间；正式实验不覆盖自然命中结果，forced smoke 仅作为独立协议检查。
3. **请求准备（命中后）**：根据 `source_uuid` 准备 source latent、trajectory、attention mask 和 sparse runtime。Carnator 依据 source/target noun 集合选择 source-only Mbase；集合相同时按完整 token 面积选最小者，并生成请求私有 bundle。mask 按 step 预解码并放入 RAM/pinned memory/GPU；该区间在 wall-clock 中单独标记，并从报告的端到端时间中排除，但可在 audit metadata 中记录。
4. **被测生成**：沿用同一请求计时，命中请求执行对应的 reuse 路径；未命中不启动 Dense C0，只写一行 miss 记录并继续下一个请求。
5. **结果与清理**：保留生成视频和 JSONL；视频写盘、cache 写盘和结果 hash 仍不计入既有端到端指标。所有请求结束后再统一运行 CLIP-T/VBench-Q。

### 3.1 命中与未命中

- NIRVANA-VID、Chorus、Chorus2：完整 prompt CLIP/Chroma Top-1，阈值 `0.65`；只有检索命中才准备 source cache 并执行 reuse。
- Carnator Boundary B5：noun prefilter、step-5 Early Signature gate 后以 CLIP 选择 top-1；probe/gate fail、per-noun mask 缺失或选择不唯一都是 miss，不回退 dense。
- TeaCache：无跨请求 cache 命中，hit rate 保持 N/A；每条请求只执行 target-only TeaCache。
- miss 仍写入 `requests.jsonl` 和 `request_records.jsonl`，至少包含 `hit=false`、`source_uuid=null`、`executed_steps=0`、`status=cache_miss`、完整 `cache_match` 和 miss reason；不生成视频，不计入命中耗时均值。

### 3.2 各基线算法路径

| 基线 | 命中后的算法流程 | 未命中 | `denoising_s` 组成 |
| --- | --- | --- | --- |
| NIRVANA-VID | source latent completed step 5 -> target 45-step restart | 只记录 miss | target 实际 45 步 DiT |
| TeaCache | target-only intra-request step skipping，50-step scheduler | 不适用 | target 实际执行的 DiT steps |
| Chorus | source trajectory + TGAA，K1/K2=`5/30`，target 45 步 | 只记录 miss | target 实际 45 步 DiT |
| Chorus2 | target 完整 50-step；第 11--50 步使用 source raw block mask + SpargeAttn，不做 latent restart | 只记录 miss | target 实际 50 步 DiT |
| Carnator Boundary B5 | step-5 probe -> gate 通过 -> CLIP top-1 -> 选择 source-only Mbase；同集合选最小面积 Mbase -> 请求私有 mask bundle -> source step-10 latent + Boundary B5 reuse，target 40 步 | probe/gate/mask fail 只记录 miss | probe 前 5 步 DiT + target 40 步 DiT |

## 4. 时间指标定义

保留现有字段和含义：

- `latency_s` / 命中请求的 `hit_e2e_s`：从请求到达到视频解码/Finalization 完成，包含检索、Carnator 实际 probe、生成和准备阶段之外的其它请求工作；仅扣除命中后的 cache prepare 区间，以及视频/缓存写盘和结果 hash。
- Chorus 的 LLM prompt diff 和 Grounded-SAM2 包含在端到端时间中，`external_s` 继续单独记录；其他基线为 0。
- `denoising_s` / `hit_denoising_s`：CUDA 前后同步包围的实际 DiT forward。NIRVANA-VID、TeaCache、Chorus、Chorus2 只统计 target 实际执行的 steps；Carnator 额外加 B5 probe 的前五步。
- 命中耗时均值只对 `hit=true` 请求计算；hit rate 的分母仍是全部 300 条请求。miss 不参与耗时均值，但必须保留完整检索审计记录。

本阶段先不计算 speedup：先记录 300 条请求的端到端、去噪、命中率和质量指标。之后再用同一批 300 条请求、同一配置、完全不读取 cache 的 Dense C0 control 平均时间作为端到端
和去噪 speedup 的统一分母，结果表中的两类 speedup 先填 TBD。

## 5. 汇总与曲线

每个基线完成后生成：

- `results/<method>/requests.jsonl`：逐请求命中、去噪/端到端/External 时间、命中请求专用的 `hit_e2e_s`/`hit_denoising_s`、输出路径、错误和原始 `cache_match` 元数据；
- `results/<method>/request_records.jsonl`：每条新请求的检索决策元数据，包含 `hit`、命中对象、相似度；Carnator 额外保存 noun prefilter、Stage-2 候选、Early Signature gate/probe 状态，以及选中 source noun、选择原因、面积和私有 mask bundle 路径；
- `results/summary.csv`：大表的一行/基线汇总，包含全请求及命中请求耗时均值；
- `results/VIDPROM_1000CACHE_300INPUT_RESULTS_TABLE.md`：汇总表（由聚合脚本生成）；
- `results/hit_rate_over_arrivals.csv`：每 50 条请求一组（1--50、51--100、…、251--300），
  记录该组命中数、组内 hit rate、累计命中数、累计 hit rate；
- `results/hit_rate_over_arrivals.png`：五基线同图对比。TeaCache 的曲线标为 N/A，不伪造跨请求命中。

图的横轴是已到达请求数（50、100、…、300），纵轴是累计 hit rate；另可在 CSV 中保留
50-request window hit rate 供审计。质量评估读取保留的视频和 prompt manifest，输出 CLIP-T、
VBench-Q 及每请求分数，最后合并到大表。

实验脚本默认不持久化新请求的 target cache；native 方法只保留顶层视频、summary/report 和 JSONL
记录，删除每请求的中间 artifact。Carnator 对有候选的目标请求在线执行 step-5 Early Signature
probe：cross-attention 仅提供 noun seed，Mbase 必须融合 self-attention contrast；只有 probe 实际
完成且通过 Stage-2 gate 后才允许 reuse。命中推理不生成 target 执行 mask；命中后仅从候选 source 的
per-noun Early Signature 读取 Mbase，生成私有 bundle，绝不把
共享 generic `fusion_masks` 作为 noun mask 或改写它。probe 不通过时记录 miss、probe 时间、比较结果和执行原因，不生成 dense 视频。

## 6. 推荐脚本拆分

- `scripts/benchmark/materialize_cached_request_view.py`：生成/校验本文件定义的 1000-entry view；
- `scripts/benchmark/run_vidprom_300_experiment.py`：单 GPU、单基线、按 arrival order 运行，
  只读 view，不保存新 cache；统一编排 worker warm-up、retrieval、request prepare、计时和 miss 记录；支持 `--gpu 3..7`、`--method`、resume 和 dry-run；
- `scripts/benchmark/aggregate_vidprom_300_results.py`：从五个 JSONL 计算大表、speedup 和
  50-request hit-rate CSV/PNG；
- `scripts/benchmark/quality_runner.py`：在所有生成结束后统一计算 CLIP-T/VBench-Q。

每个 baseline adapter 只负责声明自己的 retrieval、`prepare_request` 和 hit generation 逻辑；runner 负责统一的请求生命周期、计时起止和 miss 记录。脚本实现前先做 `--check-only`：检查模型/config fingerprint、view 中 1000 个 UUID 的 manifest/hash、各 baseline 的步数与复用步数、GPU 只在 3--7，以及输出目录为空或可安全 resume。
