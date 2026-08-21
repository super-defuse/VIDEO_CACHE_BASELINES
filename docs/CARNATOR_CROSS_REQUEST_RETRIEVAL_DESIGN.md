# Carnator 跨请求两阶段检索设计

状态：已实现（2026-08-21）

适用范围为 Wan2.2-TI2V-5B 的原生 Carnator C0/B0--B5 路径；不改变其采样、稀疏 attention
或量化逻辑，CogVideoX1.5 不在范围内。

## 流程

```text
prompt
  -> 阶段 1：整条 prompt 的 CLIP Top-K 召回
  -> trf_nouns noun-set 预过滤
  -> target 正常执行至 completed step 5，构造 ETS signature
  -> 阶段 2：按召回顺序比较 signature
  -> 首个通过者：step-10 latent + sparse-attention joint reuse
  -> 无通过者：沿当前轨迹完成 dense generation
```

阶段 1 使用归一化 `openai/clip-vit-large-patch14` embedding 的精确内积查询。默认
`candidate_top_k=20`，按 `(score DESC, source_request_index ASC, uuid ASC)` 排序；它只是
候选上限，CLIP 分数不参与最终授权，也不设未经校准的接受阈值。

source 和 query 均使用 `trf_nouns`。只保留 noun set 相同，或双方各恰有一个不同 noun 的候选；
没有候选时不运行 step-5 probe。对保留候选，target 在 step 5 生成每个 noun 的 `Mbase` 和临时
`ets` signature；依次执行 metadata/noun mapping、四项 ETS metric、冻结阈值和精确 cache
artifact 检查。第一个同时满足 `signature_compatible && cache_ready` 的候选获选，不得融合或重排
signature 分数。

无冻结且认证的 `threshold_config_id`、任一 metadata/mask/数值错误，或缺少精确 step-10 latent 和 sparse-attention artifact 时一律 `No Reuse`。失败候选的 latent 或 attention 不得注入 target。

## 独立实现与重跑

检索、signature builder/comparator、shape policy、threshold loader、配置和测试全部在本仓库实现。外部设计只能供人工理解算法语义，禁止导入、执行、读取、软链接或复制 `../early_signature/` 下的
任何代码或文件；不得使用其 noun CSV、mask、signature、cache、阈值、质量观测或 fixture。CI 应在
该目录不可访问时通过。

实现完成后，从 `datasets/vbench_cross_request_prompts.csv` 的 694 条请求重新生成本仓库的
`datasets/carnator_vbench_trf_nouns.csv`：

```text
request_index,uuid,prompt,dimension,trf_nouns
```

本地标注工具独立生成非空、去重的 canonical `trf_nouns`；记录输入、输出和配置的 SHA-256，并断言
覆盖 694 条。随后在空的 Carnator namespace 中重新运行 source，生成新的 `Mbase`、
`carnator_ets.json`、step-10 latent 和 sparse-attention artifacts；只用这批新 pair 选阈值并在独立新生成的 split 上做 Clopper-Pearson certification。旧 namespace、仓库外路径或缺少本地
implementation/profile ID 的 manifest 一律拒绝。

## 落盘与配置

每个 cache-ready source 保存 `carnator_ets.json`、精确 step-10 latent、
sparse-attention pattern、含 artifact hash 的 `manifest.json`，以及 prompt vector 到 UUID/signature/manifest 的 `retrieval.json`。淘汰或 artifact 失效时同步删除 vector record。`results.jsonl.cache_match` 记录 `retriever=carnator_two_stage`、Top-K `(uuid, rank, score)`、noun 预过滤、阶段 2 拒绝 reason/metric、获选 UUID、`threshold_config_id` 与 `cache_ready`；只有最终 joint reuse 才是 `hit=true`。

```yaml
retrieval:
  carnator:
    backend: clip
    model_id: openai/clip-vit-large-patch14
    candidate_top_k: 20
    noun_annotations: datasets/carnator_vbench_trf_nouns.csv
    signature_profile: ets
    completed_step: 5
    reuse_anchor_step: 10
    threshold_config: configs/retrieval/carnator_ets_thresholds.json
```

## 检索验收

验收使用 `retrieval-only` 模式，不加载 Wan、不执行去噪或 CTAP、不生成视频，也不读取真实 cache
artifact。该模式以本仓库生成的 prompt vector、ETS signature fixture 和 cache-ready fixture 驱动两阶段逻辑，输出 `retrieval_results.jsonl`；每行至少包含 `request_index`、`uuid`、Top-K candidates、noun prefilter、最终 `hit`、`source_uuid` 和拒绝 reason。fixture 中的 cache-ready 仅用于验证决策，不会加载 latent 或 attention tensor。

验收覆盖：694 条本地 noun 数据、目录隔离、Top-K 稳定排序与重启恢复、相同/单 noun substitution/
多 noun 差异、无效 metadata/空 mask/缺 artifact/未认证 threshold 的 fail-closed。真正的 source 生成、artifact 生成和质量校准是独立流程，不属于检索验收。
