# VidProM 923-cache / 300-request 实验计划

本轮实验固定使用 Wan2.2-TI2V-5B、50-step、`1280x704`、121 帧和同一 seed/config。
输入缓存源是 `datasets/1000cache_300input/cluster_02_2170_first1000_cached.csv` 中的 923 条请求；新请求是 `cluster_02_2170_random300_input.csv`，按 CSV 顺序到达。GPU0--2 保留不用，五个基线各占一张 GPU：

| GPU | 基线 | 说明 |
| ---: | --- | --- |
| 3 | NIRVANA-VID | 完整 prompt CLIP/Chroma Top-1，阈值 `0.65`，命中后从 completed step 10 latent restart |
| 4 | TeaCache | 仅目标请求的 intra-request step skipping，不读取共享跨请求 cache；hit rate 记为 N/A |
| 5 | Chorus | 完整 prompt CLIP/Chroma Top-1，阈值 `0.65`，使用 source trajectory、K1/K2=`10/30` 与 TGAA |
| 6 | Chorus2 | 暂无独立检索器；与 NIRVANA-VID/Chorus 共用完整 prompt CLIP/Chroma Top-1（同一排序与阈值），命中后使用 source step-10 latent + 普通 SpargeAttn mask |
| 7 | Carnator Boundary B5 | CLIP/Chroma 候选排序后执行 Carnator noun/step-5 Early Signature gate；target 端固定 Boundary B5（SpargeAttn + W8A8） |

TeaCache 完成后，Carnator 正式实验拆分为四个互不重叠的 arrival shard：GPU3/4/5/7 分别处理
`0--74`、`75--149`、`150--224`、`225--299`；合并 JSONL 后再进入质量评估。每条 Carnator Early Signature 的 Mbase 固定为
step-5 cross-attention seed 与 self-attention contrast 的融合结果，不能使用仅 cross-attention mask；尾部空切片仅复制最近非空融合切片，不降低阈值；整个时间轴为空时 probe fail-closed 并回退完整 dense generation。

实验固定使用与 shared cache `run_manifest.config.seed` 相同的 `seed=20260218`。Carnator
Early Signature 四个 gate 阈值（依次为 context color、log area、shape、changed color）固定为
`0.8569418522546745`、`0.7643009491051245`、`0.3015873015873016`、`1.0714081281330827`；
runner 将其写入 resolved config 和请求 metadata。

## 1. 缓存与数据边界

原始 1000-entry cache 位于
`artifacts/shared_cross_request_cache/vidprom_cluster_02_2170_first1000/`，不删除、不改写。
根据 923-row CSV 已生成只读视图：

```text
artifacts/shared_cross_request_cache/vidprom_cluster_02_2170_first1000/cached_requests_923/
  entries/<uuid> -> ../../entries/<uuid> # relative symlink, no tensor/video copy
  index.jsonl                             # only the 923 selected UUIDs
  source_requests.csv                     # exact selection CSV snapshot
  excluded_requests.csv                  # 77 screened-out UUIDs (for audit only)
  view_manifest.json                      # counts, hashes, provenance
```

`source_requests.csv` 和 `excluded_requests.csv` 是可重建的本地审计快照，不纳入版本库；选择集哈希保存在 `view_manifest.json`。

实验 runner 只把该 view 的 `index.jsonl` 建成可检索的 source metadata，并把命中的 UUID
解析到 `entries/<uuid>`；绝不把新请求生成结果写回 shared cache。300 条新请求的产物、逐请求
JSONL 和视频统一写到各自的 run directory。TeaCache 使用独立的 target-only run directory。

每个方法从同一份 923-entry source pool 开始、各自维护检索状态；300 条请求的到达顺序不能
shuffle。为便于复现实验，runner 应保存 resolved config、GPU、git revision、输入 hash、cache
view manifest hash 和完整命中决策。

## 2. 每条请求的执行协议

1. 记录 `arrival_index`、request UUID/prompt，并启动端到端 wall-clock 计时。
2. 在计时范围内执行该基线的检索；记录 `retrieval_s` 和候选/阈值/命中 source UUID。
3. 命中时只读 shared view 的缓存产物并执行 target reuse；未命中时执行完整 target dense
   （Carnator B5 按 Boundary B5 配置）。命中或未命中都不把 target 写入 shared view。
4. 单独记录 denoising loop 时间、视频编码/保存时间和端到端时间；端到端包含第 2 步的检索和
   Chorus External，所有 CUDA timing 前后同步。External 作为独立 breakdown，不重复计入去噪。
5. 每条请求写一行结果 JSONL，视频保留到该方法的 run directory。全部 300 条结束后，再对
   五个方法的输出统一运行 CLIP-T 与 VBench-Q，不能在生成计时期间运行质量模型。

本阶段先不计算 speedup：先记录 300 条请求的端到端、检索、去噪、命中率和质量指标。之后再用同一批 300 条请求、同一配置、完全不读取 cache 的 Dense C0 control 平均时间作为端到端
和去噪 speedup 的统一分母，结果表中的两类 speedup 先填 TBD。

## 3. 汇总与曲线

每个基线完成后生成：

- `results/<method>/requests.jsonl`：逐请求命中、检索/去噪/端到端/External 时间、输出路径、错误和原始 `cache_match` 元数据；
- `results/<method>/request_records.jsonl`：每条新请求的独立检索记录，包含 `hit`、命中对象、相似度；Carnator 额外保存 Stage-1 top-k、noun prefilter、Stage-2 候选和 Early Signature gate/probe 状态；
- `results/summary.csv`：大表的一行/基线汇总；
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
完成且通过 Stage-2 gate 后才允许 reuse。命中推理只读取候选 source 的 fusion Mbase，不生成 target
执行 mask；probe 不通过时继续 dense，并在 request record 中记录 probe 时间、比较结果和执行原因。

## 4. 推荐脚本拆分

- `scripts/benchmark/materialize_cached_request_view.py`：生成/校验本文件定义的 923-entry view；
- `scripts/benchmark/run_vidprom_300_experiment.py`：单 GPU、单基线、按 arrival order 运行，
  只读 view，不保存新 cache；支持 `--gpu 3..7`、`--method`、resume 和 dry-run；
- `scripts/benchmark/aggregate_vidprom_300_results.py`：从五个 JSONL 计算大表、speedup 和
  50-request hit-rate CSV/PNG；
- `scripts/benchmark/quality_runner.py`：在所有生成结束后统一计算 CLIP-T/VBench-Q。

脚本实现前先做 `--check-only`：检查模型/config fingerprint、view 中 923 个 UUID 的 manifest/hash、Chorus2 复用 `chorus` retriever 的配置、GPU 只在 3--7，以及输出目录为空或可安全 resume。
