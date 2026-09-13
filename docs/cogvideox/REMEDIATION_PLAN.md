# CogVideoX1.5 适配重构与实验流程对齐计划

**目标**：把 CogVideoX1.5 上五个缓存基线（baseline / nirvana_vid / chorus / chorus2 /
carnator_boundary_b5）的实现与实验流程对齐 `docs/WAN21.md` 与 `docs/LTXVIDEO.md`。

**原则**（用户明确要求）：

1. **算法逻辑不变**，只做模型架构适配。任何数值/调度/掩码语义的改动都必须能指回 Wan2.1
   或 LTX-Video 的同一行逻辑。
2. **实验流程**参考 Wan2.1 / LTX-Video：命中请求对由不可变 replay 真值**直接确定**，
   不重算命中率、不用 probe 结果改变 hit/miss。
3. 口径（计时字段、status 语义、产物 schema、聚合报告）与另外两个模型逐项可比。

> **状态（2026-09-12）：P1–P5 已实施完毕，10 请求测试已通过（50 条请求 0 error），
> P6（300 请求）尚未运行。**
> 本文件保留为「偏差清单 + 计划」的审计记录；每一行的**实际处理结果**见
> [§7 实施结果](#7-实施结果2026-09-12)。实施过程中核实到两处本文档自身的事实错误
> （C1 与 B4），已在原处标注更正，避免后来者按错的前提去改代码。

---

## 1. 现状与结论

CogVideoX 的**算法分支本身**已经与 Wan/LTX 同构：同一套 replay 决策语义
（hit → source 捕获 → target 复用 → finally 清理；miss → 短路 `cache_miss`）、
同样五个基线、同样的 source/target 分离计时。问题集中在三处：

| 类别 | 严重度 | 一句话 |
|------|--------|--------|
| A 实验管线未接入共享契约 | 阻塞 | baseline 无法启动；聚合器不认 `cogvideox15`；结果 payload 字段缺失 |
| B Carnator fusion-mask 路径结构性损坏 | 阻塞 | 三处 API 误用 + 异常被吞 → 该基线在原理上不可能成功 |
| C 缺少两个实验脚本 | 阻塞 | 只有 3 请求脚本，无质量评估/聚合/错误核验 |

其中 **B 类与 `docs/cogvideox/KNOWN_ISSUES.md` §2「CogVideoX 版 dog-cat fusion mask 无实例」
互为佐证**：该代码路径从未成功执行过。

---

## 2. 已核实的偏差清单（含证据）

### A. 实验管线

| # | 位置 | 现象 | 影响 |
|---|------|------|------|
| A1 | [run_vidprom_300_experiment.py:381](../../scripts/benchmark/run_vidprom_300_experiment.py#L381)、[:472](../../scripts/benchmark/run_vidprom_300_experiment.py#L472) | `load_cogvideo_replay(root, method, limit=300)` **无条件**执行；但 `cogvideox15_replay/` 只有 nirvana_vid / teacache / chorus / chorus2 / carnator_boundary_b5 五个文件，**没有 `baseline.jsonl`**（replay manifest 自身也只登记这 5 个方法）。Wan/LTX 的 `*_run()` 都写成 `if args.method != "baseline"` 才读 replay | **已实测**：`--model cogvideox15 --method baseline --check-only` 抛 `FileNotFoundError: CogVideo replay is missing: .../baseline.jsonl`。dense 对照跑不起来 |
| A2 | [run_vidprom_300_experiment.py:37](../../scripts/benchmark/run_vidprom_300_experiment.py#L37) | `GPU_BY_METHOD = {"baseline": 3, "nirvana_vid": 3, ...}`，两个方法撞在同一张卡；而 [:1722-1726](../../scripts/benchmark/run_vidprom_300_experiment.py#L1722-L1726) 对 cogvideox15 走的就是这张表 | 五方法并行时 baseline 与 nirvana_vid 互相抢显存；需要 `COGVIDEOX_GPU_BY_METHOD`（baseline 3、nirvana 4、chorus 5、chorus2 6、carnator 7） |
| A3 | [run_vidprom_300_experiment.py:506-520](../../scripts/benchmark/run_vidprom_300_experiment.py#L506-L520) | `cogvideo_run` 直接 `asdict(result)` 自建 payload，**不走** `flatten_replay_payloads()` | 顶层 `denoising_s` / `external_s` / `hit_e2e_s` / `hit_denoising_s` / `video_path` / `quality_input_path` / `arrival_index` / `extra.denoising_timing` 全部缺失 → 聚合器读不到去噪与加速比 |
| A4 | [aggregate_vidprom_300_results.py:16-20](../../scripts/benchmark/aggregate_vidprom_300_results.py#L16-L20) | `MODEL_METHODS` **没有 `cogvideox15`** | `--model cogvideox15` 被 argparse `choices` 直接拒绝，聚合完全不可用 |
| A5 | [run_vidprom_300_experiment.py:565-576](../../scripts/benchmark/run_vidprom_300_experiment.py#L565-L576) | Wan/LTX 都写 `write_warmup_record(paths=["model_load", ...])`；`cogvideo_run` 没有 | 首个被计时请求会吃掉模型加载；且缺失 warm-up 审计行 |
| A6 | [run_vidprom_300_experiment.py:596-606](../../scripts/benchmark/run_vidprom_300_experiment.py#L596-L606) | Wan/LTX manifest 有 `miss_policy`、`target_probe_enabled`、`target_probe_completed_step`、`target_probe_seed`；CogVideoX manifest 全缺 | 计时口径与 probe 口径无法从产物自证 |
| A7 | [run_vidprom_300_experiment.py:415-429](../../scripts/benchmark/run_vidprom_300_experiment.py#L415-L429) | `load_cogvideo_replay` 硬编码 `limit=300` 且要求行数**恰好等于** limit；`load_replay_decisions`（Wan/LTX 用）接受 `limit=args.limit` | 10 请求测试依赖「300 行真值文件」这一巧合；裁剪过的 replay 无法使用 |
| A8 | [core/cogvideo_runner.py](../../core/cogvideo_runner.py) 全文 | `CogVideoReplayRunner` 与 [`core/wan21_runner.py`](../../core/wan21_runner.py) 的 `Wan21ReplayRunner` 高度重复；后者已被 LTX 直接复用（`Wan21ReplayRunner` 本身就是 model-agnostic，`model=` 可传 label） | 两套 runner 会持续漂移（A3/A5/A6 就是漂移的结果）。应二选一 |
| A9 | [run_cogvideox_3requests_test.sh](../../scripts/benchmark/run_cogvideox_3requests_test.sh) | 用系统 `python`（非 conda env）、`set -e` 配后台任务、无质量评估、无聚合、无 `FAILED` 统计、输出目录带时间戳、`--limit 3` | 正是 `KNOWN_ISSUES.md` §3.1「进程正常退出但所有请求都在 error」的温床；且不产出可聚合的报告 |
| A10 | [WORKFLOW.md:9](WORKFLOW.md#L9) | 写「480x720，49 frames」；实测 `resolved_config` 是 **1360×768、81 帧**、50 步、guidance 6.0（见 `cogvideox15_3req_test_20260908_151456/nirvana_vid/run_config.json`） | 文档失真 |

### B. Carnator Boundary B5（最严重）

`carnator/cogvideox15.py` 的 fusion-mask 路径有三处**必然抛异常**的 API 误用，且异常被吞：

| # | 位置 | 现象 |
|---|------|------|
| B1 | [carnator/cogvideox15.py:286](../../carnator/cogvideox15.py#L286) | `infer_token_grid_shape(video_tokens, frames=49)` —— 该函数签名是 `infer_token_grid_shape(latent_shape)`（[cogvideo_attention_probe.py:273](../../carnator/cogvideo_attention_probe.py#L273)），**既多传位置参数又多传关键字**，且把帧数**写死 49**（实验是 81 帧）→ `TypeError` |
| B2 | [carnator/cogvideox15.py:315-319](../../carnator/cogvideox15.py#L315-L319) | `build_canonical_fusion_mask(self_contrast=..., cross_attention=..., config=...)` —— 真实签名是 `(self_score, cross_score, config, *, original_mbase=None)`（[carnator/masks.py:644](../../carnator/masks.py#L644)）→ `TypeError` |
| B3 | [carnator/cogvideox15.py:170-174](../../carnator/cogvideox15.py#L170-L174) | 上面两个异常（以及任何其它异常）被 `except Exception` 吞掉，只 `print` 一条 warning + traceback，然后 `payload["mask_bundle_path"] = None`。target 的 [`_mask()`](../../carnator/cogvideox15.py#L454-L459) 随后 `RuntimeError: CogVideo Carnator requires fusion mask bundle` | 
| — | **合并结论** | B1/B2/B3 使得 **Carnator 在 CogVideoX 上 100% 必然失败**。这不是「未验证」，是结构性损坏。而且失败被降级成 warning，本地看不出问题 |
| B4 | [carnator/cogvideox15.py:335-441](../../carnator/cogvideox15.py#L335-L441) | `_generate_fusion_mask_from_latent` 是**死代码**（只被 `scripts/dev/verify_carnator_fusion_mask.py` 的存在性检查引用），但它偏偏是**唯一用对了 API** 的实现（`self_score=`/`cross_score=`、`infer_token_grid_shape(step5_latent.shape)`）。两份实现参数默认值还不同：captured 版 `eps=0.1/gamma=2.0/high=0.7/low=0.3/msee=1`，latent 版 `eps=0.2/gamma=1.0/high=0.45/low=0.35/msee=3`（**后者才与 `cogvideox15.yaml` 一致**） |
| B5 | [carnator/cogvideox15.py:471-475](../../carnator/cogvideox15.py#L471-L475) | `latent_base = token_base`（并反向），注释断言「CogVideoX 的 latent 空间与 token 空间同维」。**与 Chorus 的实现直接矛盾**：[methods/chorus/cogvideo_mask.py:38-71](../../methods/chorus/cogvideo_mask.py#L38-L71) 明确用 `patch_size` 做 `project()`（latent→token 下采样）与 `expand()`（token→latent 上采样），并按 `patch_size=(patch_t, patch, patch)` 取参。CogVideoX1.5 是 `patch_size=2 / patch_size_t=2`，两者**必然不同维** |
| B6 | [cogvideo_attention_probe.py:292-294](../../carnator/cogvideo_attention_probe.py#L292-L294) | `infer_token_grid_shape` 里写着 `# CogVideoX token grid is same as latent spatial dimensions` —— 与 B5 同一个错误假设，是根因所在 |
| B7 | [carnator/cogvideox15.py:588-590](../../carnator/cogvideox15.py#L588-L590) | per-noun mask 被简化：`mbase_by_noun = {noun: masks["token_base"] for noun in all_source_nouns}`，注释自承 *"simplified approach … In the full implementation, each noun should have its own segmented mask"*。Wan 的 FUSION_MASK 是**逐名词**的（git 历史 `fix(carnator): use per-noun source masks for reuse`）；LTX 也已落到 `select_per_noun_inference_mbase` |
| B8 | [configs/models/cogvideox15.yaml:27-45](../../configs/models/cogvideox15.yaml#L27-L45) | 缺 `latent_injection_mode` / `boundary_alpha_function` / `outer_backend` / `ffn_backend` 四个键。Wan2.1（[wan21.yaml:39-46](../../configs/models/wan21.yaml#L39-L46)）与 LTX（[ltxvideo.yaml:94-95](../../configs/models/ltxvideo.yaml#L94-L95)）都从配置读；CogVideoX 把 `boundary-blend` + `smoothstep` **写死**在 [`_boundary_weights()`](../../carnator/cogvideox15.py#L542-L563)（`normalized²(3-2·normalized)`），无法切 `hard` 复现旧产物 |
| B9 | [configs/models/cogvideox15.yaml:37](../../configs/models/cogvideox15.yaml#L37) | 只有 `attention_probe_layer: 15`，缺 `attention_self_layer`。LTX 与 Wan 都把 cross 探针层与 self 探针层分开命名（Wan 参考实现探测两层） |

### C. 其它偏差（需决策，非阻塞）

| # | 位置 | 现象 | 建议 |
|---|------|------|------|
| C1 | [methods/chorus/cogvideo.py:206-209](../../methods/chorus/cogvideo.py#L206-L209) | Chorus 写 `excluded_latency_s` / `chorus_preprocess_s`，但 [core/cogvideo_runner.py:138](../../core/cogvideo_runner.py#L138) 只扣 `source_prepare_s`/save/hash，**没有**扣这两项；而 `wan21_runner.py:176` 扣的是 `cache_prepare_s` | ~~统一：把 LLM+SAM2 耗时按 Wan 口径扣出 `latency_s`…**当前 CogVideoX 的 `External s/request` 恒为 0，是错的**~~ **⚠️ 本条的前提是错的，已作废——见下方更正。** |
| C2 | [adapters/cogvideo_sparse.py](../../adapters/cogvideo_sparse.py) | CogVideoX 稀疏后端是 `torch.compile(flex_attention)`，Wan2.1/LTX 都是 SpargeAttn（sparge-sage2） | 架构级差异，`docs/cogvideox/README.md` 已说明理由（免外部 CUDA 编译）。**建议保留 flex，但在报告与文档里显式登记为「kernel 不同、路由逻辑相同」的不可比项**；若要严格可比再评估切 SpargeAttn（H100 sm90 具备条件，风险是分块几何 64/128 与 `cogvideox15.yaml` 的 128/128 冲突，需一并改） |
| C3 | [carnator/cogvideox15.py:153-155](../../carnator/cogvideox15.py#L153-L155) | carnator source 解码并存了 `source.mp4`。只有 Chorus 需要（它要拿视频做 Grounded-SAM2 分割）；Carnator 用 fusion mask，不需要 | 删除，省一次 VAE decode（当前计入 `source_prepare_s`） |
| C4 | [configs/models/cogvideox15.yaml:29-30](../../configs/models/cogvideox15.yaml#L29-L30) | 层划分 `regional_layers: 14-27` / `outer_sparse_layers: 0-13,28-41`（42 层中段 14 层、外段 28 层）；Wan2.1 是 `10-20` / `0-9,21-29`（30 层中段 11 层、外段 19 层） | 属按层数重标定，非缺陷。需在文档中显式标注「中/外比例按 42 层重标定」 |

#### C1 更正（2026-09-12 核实）

原判据说「`wan21_runner.py:176` 扣的是 `cache_prepare_s`，所以 CogVideoX 也该把 LLM+SAM2 扣出去」。
逐行读 `core/wan21_runner.py` 后确认这个推论是错的：

```python
latency_s = max(0.0, elapsed - source_prepare_s - cache_load_s - cache_prepare_s - save_s - hash_s)
```

**replay runner 只扣 `cache_prepare_s`（来自 `generation.extra["cache_prepare_s"]`），从不扣
`excluded_latency_s`。** `excluded_latency_s` 只在 `core/runner.py` 里被读，而且还要
`self.exclude_preprocess_latency` 打开才生效——那是 Wan2.2 跨请求缓存的路径，与本实验的
replay 协议无关。

所以：

1. Chorus 的 LLM+SAM2 耗时**本来就该留在 `latency_s` 内**，Wan2.1、LTX 也都是这样。
   `excluded_latency_s` / `chorus_preprocess_s` 是上报字段，不是扣除项；
2. `external_s` 也不是 0：`flatten_replay_payloads()` 会用
   `extra["chorus_llm_s"] + extra["chorus_sam_s"]` 合成它（只对 `method == "chorus"`）。
   `methods/chorus/cogvideo.py` 正是写这两个键，所以 CogVideoX 的 `External s/request`
   一直是对的；
3. `cache_prepare_s` 在 CogVideoX 上确实是 0，而且**应该**是 0——各基线的 block mask 是
   在 `cache_load_s` 区间从磁盘读出来的，`CogVideoFlexExecutor` 又是懒构建索引（那部分时间
   计在 `denoising_s` 内）。LTX 有 `cache_prepare_s` 是因为它的 executor 有一次显式
   `prepare()`。

**结论：C1 不需要任何代码改动**，只需要把口径写清楚（已写入
[WORKFLOW.md](WORKFLOW.md) 的「计时口径」一节）。

#### B4 更正（2026-09-12 核实）

原文说 `_generate_fusion_mask_from_latent` 是「唯一用对了 API 的实现」，并据此把它定为 P2 的
唯一入口。核实后也不对：它把**原始 VAE latent（最后一维 16）**和 **4096 维的 text embedding**
直接喂给 `attn1.to_q`（该投影输入是 3072 维），`torch.cat` 立即就报错——它之所以看起来「用对」，
只是因为它正确调用了 `build_canonical_fusion_mask(self_score=..., cross_score=...)` 的**关键字名字**，
而它传进去的分数本身是用错的方式算出来的。P2 因此改为**重写**而不是「扶正其中一个」，见 §7。

---

## 3. 修改方案

### P1 — 实验管线对齐（不动任何算法）

**1. `scripts/benchmark/run_vidprom_300_experiment.py`**

- 新增 `COGVIDEOX_GPU_BY_METHOD = {"baseline": 3, "nirvana_vid": 4, "chorus": 5, "chorus2": 6, "carnator_boundary_b5": 7}`，
  并在 `--gpu` 校验处按 `args.model` 选择（修 A2）。
- `check_only()` 里把 cogvideox15 的 replay 校验改成 `if args.method != "baseline"`（修 A1）。
- `cogvideo_run()` 改为与 `ltxvideo_run()` 同构：
  - `if args.method != "baseline"` 才 `load_cogvideo_replay(..., limit=args.limit)`；baseline 的 `replay=None`（修 A1/A7）。
  - 复用 `flatten_replay_payloads()` + `append_result_and_record()`，删除自建的 records 循环（修 A3）。
    *前置*：让五个 CogVideoX 方法都把 `dit_forward_s` 放在 `extra` 顶层（当前 Chorus/Chorus2/Nirvana 已是，Carnator 的 `probe_dit_forward_s` 需额外提到顶层，供 `denoising_timing` 拆分）。
  - 加 `write_warmup_record(output, paths=["model_load"])`；`chorus` 额外预热 Grounded-SAM2（修 A5）。
  - manifest 补齐 `miss_policy` / `target_probe_enabled` / `target_probe_completed_step` / `target_probe_seed`（修 A6）。
  - 用 `requests.raw.jsonl` + `raw_path.unlink(missing_ok=True)` 的写法，与 Wan/LTX 完全一致。
- `load_cogvideo_replay` / `load_replay_decisions` 合并为一个 `load_replay_decisions(root, method, limit=...)`（两者 schema 校验完全相同，只有行数要求不同）。

**2. `scripts/benchmark/aggregate_vidprom_300_results.py`**

- `MODEL_METHODS` 增加 `"cogvideox15": ("baseline", "nirvana_vid", "chorus", "chorus2", "carnator_boundary_b5")`（修 A4）。
- `aggregate_method()` 里默认 `gpu` 表按 model 分派（当前读 `run_config.json` 可兜底，但默认值需一致）。

**3. `core/cogvideo_runner.py`（修 A8 + C1）**

二选一，推荐 **(a)**：

- **(a) 删除 `CogVideoReplayRunner`，CogVideoX 直接复用 `Wan21ReplayRunner`**——与 LTX 的做法一致。
  该 runner 已是 model-agnostic：`Wan21ReplayRunner(adapter, method, output_dir=..., replay=..., source_prompts=..., config=..., model="CogVideoX1.5")`。
  需要确认三点：①`teacache` 的 `target_only` 分支语义相同（两边都有）；②`latency_s` 扣除项一致；
  ③source 生成时 `capture=True` 的调用签名一致（`method.generate(req, cached_latent=None, capture=True)` 两边相同）。
  *额外*：nrivana 的 `reuse_step` 拆包逻辑（[cogvideo_runner.py:94-103](../../core/cogvideo_runner.py#L94-L103)）在 `wan21_runner` 里是
  `generation.extra["cache_prepare_s"]` 路径——需要在 CogVideoX 侧补一个等价钩子或把这层适配放进
  `CogVideoX15Adapter.generate()`。
- **(b) 保留双 runner**，但把 A3/A5/A6/C1 的差异逐条补齐，并加一个测试锁死两个 runner 的 `RequestResult` 字段集一致。
  （次选：漂移风险回到原点。）

C1 的具体改法：`latency_s = elapsed - source_prepare_s - preprocess_s - save_s - hash_s`，
其中 `preprocess_s = generation.extra.get("chorus_preprocess_s", 0.0)`，与 Wan 的 `cache_prepare_s` 同源。

### P2 — Carnator fusion mask 修复

1. **收敛到单一实现**：~~删除 `_generate_fusion_mask_from_captured`，把
   `_generate_fusion_mask_from_latent` 扶正为唯一入口（它已用对 API）~~
   → **实际做法：两个都删，连同 `_boundary_weights`、`_mask` 一起，重写为一条走
   `joint_qkv` + `build_canonical_fusion_mask` + `build_per_noun_mbase_bundle` +
   `apply_latent_injection` 的链路**（原因见上文 C1 后面的「B4 更正」；修 B1/B2/B4）。
   参数默认值以 `cogvideox15.yaml` 为准（`eps=0.2 / gamma=1.0 / high=0.45 / low=0.35 / msee=3`）。
2. **修 token↔latent 网格**：把 [methods/chorus/cogvideo_mask.py](../../methods/chorus/cogvideo_mask.py) 的
   `project()` / `expand()` 抽到共享位置（建议 `adapters/cogvideo_attention.py`），
   Carnator `_mask()` 用它做双向上/下采样；同时修正
   `infer_token_grid_shape` 的错误注释与实现（修 B5/B6）。
   网格由 `transformer.config.patch_size` / `patch_size_t` 与 latent shape 计算，不得写死帧数。
3. **fail-closed**：把 [`_source()` 的 `except Exception`](../../carnator/cogvideox15.py#L170-L174) 改成硬失败
   （让异常冒泡 → `CogVideoReplayRunner` 记 `status=error`），与 Wan/LTX 的 fail-closed 一致。
   保留一条 `mask_bundle_path is None → raise` 的断言（修 B3）。
4. **逐名词 mask**：按 Wan 语义产出 per-noun `Mbase`（每个 source noun 一份），
   target 侧按 `select_per_noun_inference_mbase` 取「source 独有名词」的 mask（修 B7）。
5. **配置驱动**：`cogvideox15.yaml` 的 `carnator_b5` 补
   `latent_injection_mode: boundary-blend`、`boundary_alpha_function: smoothstep`、
   `outer_backend`、`ffn_backend`、`attention_self_layer: 15`；
   代码从配置读取（`_boundary_weights()` 支持 `hard`）（修 B8/B9）。
6. **删除多余的 source.mp4**（修 C3）。
7. **回归测试**：`tests/test_cogvideox15_adapter.py` 增加
   ①fusion mask 的 token/latent 双向投影形状断言；
   ②fusion mask 失败必须产生 `status=error` 而非 warning；
   ③`_boundary_weights` 在 `hard` 模式下 alpha ∈ {0,1}。

### P3 — 实验流程（对齐 Wan2.1 / LTX-Video 的「直接确定命中请求对」）

沿用现有 replay 真值，不改判定逻辑，只对齐执行与口径：

```
输入真值（不可变）：artifacts/vidprom_1000cache_300input_noun_match/cogvideox15_replay/{method}.jsonl
  · 每个方法 300 行，schema cogvideox15-hit-replay/v1
  · nirvana_vid / chorus 242 hit + 58 miss；chorus2 242/58；carnator_boundary_b5 300/0
  · baseline 无 replay 文件 → 300 条全量 dense 生成

每个 arrival（0..299）：
  hit  → source 按 replay 的 source_uuid 生成 → 写 tmp_cache → target 复用 → 落视频 → finally 清 tmp_cache
  miss → 直接返回 status=cache_miss, latency_s=0, executed_steps=0, extra.skipped=true（不生成视频）
  error→ 终态审计记录，不改写为 miss，不计入 hit-rate 分母与质量评分

计时口径（与 wan21_runner 一致）：
  latency_s / hit_e2e_s / target_e2e_s 同值
      = wall − source_prepare_s − cache_load_s − cache_prepare_s − save_s − hash_s
  denoising_s = CUDA 同步后的 DiT forward（Carnator = probe 5 步 + reuse 40 步 = 45 步）
  source_prepare_s 单独上报，不得隐藏
  external_s = Chorus 的 chorus_llm_s + chorus_sam_s（由 flatten_replay_payloads 合成），
      且【留在 latency_s 内】——replay runner 不扣 excluded_latency_s（见 C1 更正）
  warm-up 不计分
```

**Carnator target probe 口径**（与 Wan2.1/LTX §9.2 逐项对齐，CogVideoX 已实现，只需保证不被改坏）：
跑完整 50 步计划的**前 5 步**、用固定实验 seed、probe 的 signature 对比结果只写 `extra`
供审计，**不参与任何命中决策**（replay 已是最终真值）。

### P4 — 两个实验脚本

新增（对齐 `run_ltxvideo_all_{test,formal}.sh` 的结构）：

- `scripts/benchmark/run_cogvideox_all_test.sh` —— 10 请求测试
- `scripts/benchmark/run_cogvideox_all_formal.sh` —— 300 请求正式实验

两者都：五方法并行 GPU 3–7 → 等待 → **核验 `requests.jsonl` 的 status 分布**（不只依赖退出码）
→ 聚合出 `summary.csv` / `hit_rate_over_arrivals.{csv,png}` / `VIDPROM_1000CACHE_300INPUT_RESULTS_TABLE.md`。
质量评估由 `run_vidprom_300_experiment.py` 在每个方法跑完后自动执行（`conda run -n vbench`），
脚本不重复调用；聚合前脚本会校验质量列完整。

```bash
#!/bin/bash
# scripts/benchmark/run_cogvideox_all_test.sh
# CogVideoX1.5 10-request smoke run: all five baselines in parallel, then aggregate.
# Formal 300-request experiment: run_cogvideox_all_formal.sh
#
# The cogvideo environment is separate from the wan environment used by
# Wan2.1 / LTX-Video; override it with COGVIDEOX_ENV if needed.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="/data/shared/conda_envs/${COGVIDEOX_ENV:-cogvideo}/bin/python"

LIMIT=10
OUTPUT_ROOT="$ROOT_DIR/artifacts/vidprom_1000cache_300input_noun_match/results_cogvideox15_test"
REPLAY_ROOT="$ROOT_DIR/artifacts/vidprom_1000cache_300input_noun_match/cogvideox15_replay"
LOG_DIR="$ROOT_DIR/artifacts/cogvideox15_experiment_logs"
METHODS=(baseline nirvana_vid chorus chorus2 carnator_boundary_b5)
GPU_FOR=(3 4 5 6 7)  # index matches METHODS; must equal COGVIDEOX_GPU_BY_METHOD

mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"
echo "CogVideoX1.5 test run: ${METHODS[*]} x $LIMIT requests -> $OUTPUT_ROOT"

PIDS=()
for i in "${!METHODS[@]}"; do
    method=${METHODS[$i]}
    log_file="$LOG_DIR/test_${method}_$(date +%Y%m%d_%H%M%S).log"
    echo "[$method] GPU ${GPU_FOR[$i]}, log $log_file"
    "$PYTHON" "$ROOT_DIR/scripts/benchmark/run_vidprom_300_experiment.py" \
        --model cogvideox15 --method "$method" --gpu "${GPU_FOR[$i]}" \
        --limit "$LIMIT" --replay-root "$REPLAY_ROOT" \
        --output-root "$OUTPUT_ROOT" --overwrite \
        > "$log_file" 2>&1 &
    PIDS+=($!)
done

FAILED=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "OK   ${METHODS[$i]}"
    else
        echo "FAIL ${METHODS[$i]} (see $LOG_DIR)"
        FAILED=$((FAILED + 1))
    fi
done

# The exit code is not evidence: the 2026-08 incident had workers exit 0 with
# every request in `error`. Count the terminal statuses in requests.jsonl.
"$PYTHON" - "$OUTPUT_ROOT" "${METHODS[@]}" <<'PY' || FAILED=1
import json, pathlib, sys

root, methods = pathlib.Path(sys.argv[1]), sys.argv[2:]
bad = False
for method in methods:
    path = root / method / "requests.jsonl"
    if not path.is_file():
        print(f"{method:<22} MISSING {path}")
        bad = True
        continue
    counts = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            status = json.loads(line).get("status")
            counts[status] = counts.get(status, 0) + 1
    print(f"{method:<22} {json.dumps(counts, sort_keys=True)}")
    if counts.get("error", 0) or not counts.get("ok", 0):
        bad = True
raise SystemExit(1 if bad else 0)
PY

if [ "$FAILED" -eq 0 ]; then
    "$PYTHON" "$SCRIPT_DIR/aggregate_vidprom_300_results.py" \
        --model cogvideox15 --results-root "$OUTPUT_ROOT"
fi
exit "$FAILED"
```

```bash
#!/bin/bash
# scripts/benchmark/run_cogvideox_all_formal.sh
# CogVideoX1.5 formal experiment: 300 requests per baseline in parallel, then the
# aggregated report. Smoke run first: run_cogvideox_all_test.sh
#
# The cogvideo environment is separate from the wan environment used by
# Wan2.1 / LTX-Video; override it with COGVIDEOX_ENV if needed.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="/data/shared/conda_envs/${COGVIDEOX_ENV:-cogvideo}/bin/python"

LIMIT=300
OUTPUT_ROOT="$ROOT_DIR/artifacts/vidprom_1000cache_300input_noun_match/results_cogvideox15"
REPLAY_ROOT="$ROOT_DIR/artifacts/vidprom_1000cache_300input_noun_match/cogvideox15_replay"
LOG_DIR="$ROOT_DIR/artifacts/cogvideox15_experiment_logs"
METHODS=(baseline nirvana_vid chorus chorus2 carnator_boundary_b5)
GPU_FOR=(3 4 5 6 7)  # index matches METHODS; must equal COGVIDEOX_GPU_BY_METHOD

mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"
echo "CogVideoX1.5 formal run: ${METHODS[*]} x $LIMIT requests -> $OUTPUT_ROOT"

PIDS=()
for i in "${!METHODS[@]}"; do
    method=${METHODS[$i]}
    log_file="$LOG_DIR/formal_${method}_$(date +%Y%m%d_%H%M%S).log"
    echo "[$method] GPU ${GPU_FOR[$i]}, log $log_file"
    # --resume keeps finished requests when a worker is restarted; drop it
    # only when you also pass --overwrite to start from scratch.
    "$PYTHON" "$ROOT_DIR/scripts/benchmark/run_vidprom_300_experiment.py" \
        --model cogvideox15 --method "$method" --gpu "${GPU_FOR[$i]}" \
        --limit "$LIMIT" --replay-root "$REPLAY_ROOT" \
        --output-root "$OUTPUT_ROOT" --resume \
        > "$log_file" 2>&1 &
    PIDS+=($!)
done

FAILED=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "OK   ${METHODS[$i]}"
    else
        echo "FAIL ${METHODS[$i]} (see $LOG_DIR)"
        FAILED=$((FAILED + 1))
    fi
done

# The exit code is not evidence: the 2026-08 incident had workers exit 0 with
# every request in `error`. Count the terminal statuses in requests.jsonl.
"$PYTHON" - "$OUTPUT_ROOT" "${METHODS[@]}" <<'PY' || FAILED=1
import json, pathlib, sys

root, methods = pathlib.Path(sys.argv[1]), sys.argv[2:]
bad = False
for method in methods:
    path = root / method / "requests.jsonl"
    if not path.is_file():
        print(f"{method:<22} MISSING {path}")
        bad = True
        continue
    counts = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            status = json.loads(line).get("status")
            counts[status] = counts.get(status, 0) + 1
    print(f"{method:<22} {json.dumps(counts, sort_keys=True)}")
    if counts.get("error", 0) or not counts.get("ok", 0):
        bad = True
raise SystemExit(1 if bad else 0)
PY

if [ "$FAILED" -eq 0 ]; then
    "$PYTHON" "$SCRIPT_DIR/aggregate_vidprom_300_results.py" \
        --model cogvideox15 --results-root "$OUTPUT_ROOT"
fi
exit "$FAILED"
```

> 旧的 `run_cogvideox_3requests_test.sh` 在 P1 完成后删除（其功能被 test 脚本覆盖，
> 且它依赖 `--limit 3` + 系统 python + 无质量评估的旧路径）。

### P5 — 文档订正

| 文件 | 改动 | 状态 |
|------|------|------|
| [docs/cogvideox/WORKFLOW.md](WORKFLOW.md) | 分辨率/帧数订正为 1360×768 / 81 帧；补两个脚本入口、五个基线逐步逻辑、产物结构、计时口径、故障排查 | ✅ |
| [docs/cogvideox/EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) | 「最新状态」改为引用本计划的验收结果；文件结构改为真实的 `cogvideox15_replay/` 与 `results_cogvideox15[_test]/`；Carnator 行删去 source-latent-restart 的措辞 | ✅ |
| [docs/cogvideox/KNOWN_ISSUES.md](KNOWN_ISSUES.md) | §2 缺口表按验收结果更新；§3.2 补记 B1–B3 之外的四处致命缺陷（新增 §3.5）；新增 §3.6（结果行被标成 Wan2.1） | ✅ |
| [docs/cogvideox/FUSION_MASK.md](FUSION_MASK.md) | 补 token 几何（81 帧 → 22 latent → `(11,48,85)`）、joint probe 的捕获方式、per-noun Mbase 与 target 选择、`attention_self_layer` 键、fail-closed 说明 | ✅ |
| [docs/cogvideox/README.md](README.md) | 文档索引加入本文件；代码位置补两个新脚本与 verifier | ✅ |
| 本文件 REMEDIATION_PLAN.md | C1 与 B4 两处前提更正；新增 §7 实施结果 | ✅ |

---

## 4. 实施阶段与验收

| 阶段 | 内容 | 验收 |
|------|------|------|
| P1 | 管线对齐（A1–A10） | `--check-only` 五个方法全 `ready`（含 baseline）；`--dry-run` 走通 `cogvideo_run → flatten_replay_payloads → 聚合入口`；`--model cogvideox15` 聚合器可调用 |
| P2 | Carnator fusion mask 修复（B1–B9、C3） | 单请求 Carnator 能产出并加载 mask bundle；token/latent 投影形状断言通过；fusion mask 失败必记为 `error`；`pytest tests/` 全绿 |
| P3 | 流程口径对齐（C1） | 与 `wan21_runner` 的 `RequestResult` 字段集一致；Chorus 的 `external_s` 非 0；Carnator `executed_steps = 45`、`denoising_s = probe + reuse` |
| P4 | 两个脚本 | `bash run_cogvideox_all_test.sh` 五方法 0 error，10 请求产出报告与质量分 |
| P5 | 文档订正 | 四份文档与实现一致 |
| P6 | 300 请求正式实验 | `run_cogvideox_all_formal.sh` 跑完；`results_cogvideox15/` 生成 summary + 汇总表；与 wan21 / ltxvideo 同表可比 |

**执行顺序**：P1 → P2 → P3 → P4 → 10 请求测试（P4）→ P5 → 300 请求（P6）。
P1/P2 完成前**不要**启动任何 GPU 运行——P2 之前 Carnator 必然全 error，P1 之前 baseline 根本起不来。

---

## 5. 成本预算（实测推导，重要）

`cogvideox15_3req_test_20260908_151456` 的实测值：nirvana_vid 命中请求
`denoising_s = 689.1 s / 45 步 ≈ 15.3 s/步`，`source_prepare_s ≈ 74 s`，
单命中请求墙钟 ≈ 790 s。据此外推 300 请求（1360×768×81，五方法并行于 GPU 3–7）：

| 方法 | 单命中成本 | 命中数 | 预计墙钟 |
|------|-----------|--------|---------|
| baseline | 50 步 ≈ 800 s | 300 | **≈ 67 h** |
| nirvana_vid | source 74 + target 690 ≈ 790 s | 242 | ≈ 53 h |
| chorus | source 765 + LLM/SAM2 30 + target 690 ≈ 1 490 s | 242 | **≈ 100 h** |
| chorus2 | source 765 + target 765 ≈ 1 530 s | 242 | **≈ 103 h** |
| carnator_boundary_b5 | source 765 + probe 77 + target 690 ≈ 1 530 s | 300 | **≈ 127 h** |

并行下的墙钟由最慢者决定 → **约 5–6 天**（未计质量评估）。

> ⚠️ 仓库旧文档里的「~77s/请求 × 300 ≈ 6.4 小时」是**小分辨率/小帧数的旧估算**，
> 与本实验的 1360×768×81 不符，不可用于排期。
>
> 建议：10 请求测试跑完后，用实测 `denoising_s` 复核上表，再决定是否维持
> 1360×768×81，或降到 CogVideoX1.5 支持的较低预算（如 720×480×49）以换取可接受的
> 实验周期——**该决策需用户确认，因为它会同时改变 Wan2.1/LTX 之外第三个模型的比较基线**。

---

## 6. 不确定项（需在 P1/P2 中实测确认，不要当成已知）

1. **token 与 latent 网格**：B5/B6 推断 CogVideoX1.5（`patch_size=2`、`patch_size_t=2`）
   下 token 数 ≠ latent 元素数。P2 第一步就是打印
   `hidden.shape[1]`（transformer 内视频 token 数）与 `latent.shape[1:].numel()`，
   把真实网格写进测试断言。**在此之前不要假定 Mask 张量的维度。**
2. **Chorus 为何在 10 样本实验里跑通过**（`regional_calls=1050`）：那是 2026-09-09 的旧
   分辨率配置，与当前 1360×768 不同。需确认在当前分辨率下 `RegionalAttentionState.indices()`
   的 token 数校验是否仍然通过。
3. **稀疏后端**（C2）：flex vs SpargeAttn 是否必须切换，取决于论文对「稀疏 kernel 一致性」的要求。
   建议先保留 flex 并在报告脚注中登记。

---

## 7. 实施结果（2026-09-12）

### 7.1 A 类：实验管线

| # | 处理 | 落点 |
|---|------|------|
| A1 | ✅ `cogvideo_run()` 改为 `if args.method != "baseline"` 才读 replay；`check_only()` 同步 | `run_vidprom_300_experiment.py` |
| A2 | ✅ 新增 `COGVIDEOX_GPU_BY_METHOD`（baseline 3 / nirvana 4 / chorus 5 / chorus2 6 / carnator 7），`--gpu` 校验按 `args.model` 选择 | 同上 |
| A3 | ✅ `cogvideo_run()` 改用 `flatten_replay_payloads()` + `append_result_and_record()`，与 wan21/ltxvideo 逐行同构；`requests.raw.jsonl` 用完即删 | 同上 |
| A4 | ✅ `MODEL_METHODS` 增加 `"cogvideox15"` | `aggregate_vidprom_300_results.py` |
| A5 | ✅ 加 `write_warmup_record(paths=["model_load"])`；chorus 额外预热 Grounded-SAM2 | `run_vidprom_300_experiment.py` |
| A6 | ✅ manifest 补齐 `miss_policy` / `target_probe_enabled` / `target_probe_completed_step` / `target_probe_seed` | 同上 |
| A7 | ✅ `load_cogvideo_replay` 并入 `load_replay_decisions(root, method, limit=...)` | 同上 |
| A8 | ✅ **采用「二选一」中的继承方案**：`core/cogvideo_runner.py` 从约 300 行缩到 37 行，`CogVideoReplayRunner(Wan21ReplayRunner)` 只覆写 `_verify_cache()` 做 manifest 校验。§3.3/§3.4 那类漂移在结构上不再可能 | `core/cogvideo_runner.py` |
| A9 | ✅ 删除旧 `run_cogvideox_3requests_test.sh`，由 `run_cogvideox_all_test.sh` 取代 | `scripts/benchmark/` |
| A10 | ✅ WORKFLOW.md 订正为 1360×768 / 81 帧 | `docs/cogvideox/WORKFLOW.md` |

### 7.2 B 类：Carnator fusion mask

按「B4 更正」改为重写，而不是扶正旧实现中的某一个。最终形态：

- `carnator/cogvideo_attention_probe.py` 重写为 `CogVideoMultiNounJointAttentionProbe`：
  **一个 probe 同时拿 cross 与 self 两半**（CogVideoX 只有一条 joint 序列，Wan 要两个类、
  LTX 还要一次额外 forward，这里 source 本来就在跑的 step-5 forward 就够）。QKV 走
  `adapters.cogvideo_attention.joint_qkv()`，与稀疏处理器替换 diffusers attention 时用的是
  同一个函数；只读 CFG 的 cond 行；`validate()` fail-closed。删除了 `CogVideoAttentionProbe`、
  `extract_scores_from_forward/captured`、`compute_self_contrast_score`、
  `compute_cross_attention_score`、`infer_token_grid_shape`（修 B1/B2/B3/B6）。
- token grid 由 `adapters/cogvideo_attention.py` 新增的 `cogvideo_token_grid()` 统一计算：
  81 帧 → 21 latent → 补到 22 → `(11,48,85)` = 44880 tokens。实测与该 transformer 的 reshape
  一致（修 B5/B6）。
- 每名词一份 Mbase（`build_canonical_fusion_mask`），target 用与 Wan/LTX 共用的
  `select_per_noun_inference_mbase()` 选择（修 B7）。
- 配置补 `attention_self_layer: 15`、`latent_injection_mode: boundary-blend`、
  `boundary_alpha_function: smoothstep`、`ffn_backend: w8a8`，全部从配置读，不再写死
  （修 B8/B9）。
- 删除 source.mp4 的编码：CogVideoX 的 mask 不再来自解码帧（修 C3）。
- **`outer_backend` 有意不加**：LTX 有 flex 与非 flex 两套外层后端可选，CogVideoX 只有
  `torch.compile(flex_attention)` 一套，加一个恒定的配置键只会让人以为有得选。这一点与
  原计划不同。
- 回归测试：`tests/test_cogvideox15_adapter.py` 增加 7 个测试（probe 记录与形状、
  cond 行、fail-closed、token grid、caption 索引、token→latent 投影、per-noun Mbase →
  target 选择）。原计划里的「`hard` 模式 alpha ∈ {0,1}」随 `_boundary_weights` 一起消失，
  改由共用的 `apply_latent_injection` 覆盖（它的测试在 LTX/Wan 侧已有）。

### 7.3 C 类

- **C1 不需要改动**（前提错误，见上文更正）。
- C2 保留 flex 后端，未切换 SpargeAttn；已在 README 与 WORKFLOW 标注为「路由逻辑相同、
  kernel 不同」的不可比项。
- C3 ✅ 已删除。
- C4 ✅ 已在 WORKFLOW/EXPERIMENT_PLAN 注明中/外比例按 42 层重标定。

### 7.4 计划外发现并修复

| # | 问题 | 影响 |
|---|------|------|
| X1 | `cogvideo_run()` 构造 `CogVideoReplayRunner` 时没传 `model=`，继承基类默认值 `"Wan2.1-T2V-1.3B"` | CogVideoX 的每一行结果都会被标成 Wan——正是 KNOWN_ISSUES §2 里两个目录被误读的成因。已显式传 `"CogVideoX1.5-5B"` |
| X2 | Carnator 的 `extra` 审计键用的是 `carnator_b5` / `mode`，而 Wan/LTX 用 `carnator_stage` + `carnator` | 同一份审计脚本读不了三个模型。已改为与 Wan/LTX 同名 |
| X3 | Chorus 的 `_token_indices` 与 Carnator 各有一份 caption 索引实现 | 收敛为 `adapters.cogvideo_attention.caption_token_indices()`，两处共用（Chorus 侧行为不变） |

### 7.5 验收对照

| 阶段 | 验收条件 | 结果 |
|------|---------|------|
| P1 | `--check-only` 五方法全 `ready`；dry-run 走通 `cogvideo_run → flatten_replay_payloads → 聚合入口` | ✅（离线） |
| P2 | fusion mask 链路形状断言通过；失败必记为 `error`；`pytest tests/` 全绿 | ✅ 234 passed |
| P3 | `RequestResult` 字段集与 `wan21_runner` 一致；Chorus 的 `external_s` 非 0 | ✅ |
| P4 | 两个脚本就位，含 status 分布核验 | ✅ 脚本已就位 |
| P4′ | **10 请求测试跑通** | ✅ **2026-09-12：50 条请求 0 error，分布逐条吻合，verifier rc=0，聚合 rc=0** |
| P5 | 五份文档与实现一致 | ✅ |
| P6 | 300 请求跑完并产出可比的 summary + 汇总表 | ⏳ **待用户确认后运行** |

**途中新增修复**（计划外）：`.cogvideo_overlay/` —— Chorus 的 Grounded-SAM-2 依赖
（`supervision` / `iopath` / `portalocker`）只在 `~/.local` 里，与 CogVideoX 必需的
`PYTHONNOUSERSITE=1` 冲突；详见 [KNOWN_ISSUES.md](KNOWN_ISSUES.md) §3.7。

**下一步**：用 10 请求的实测耗时复核 §5 的成本预算，再决定 300 请求的分辨率预算。

按实测外推（每次命中的墙钟 = `source_prepare_s` + `latency_s`，miss 瞬时短路）：

| 方法 | 命中数 | 单次命中墙钟 | 方法总墙钟 |
| --- | ---: | ---: | ---: |
| baseline | 300（全跑） | 785s | ~65h |
| nirvana_vid | 242 | 760 + 709 = 1469s | ~99h |
| chorus | 242 | 758 + 503 = 1261s | ~85h |
| chorus2 | 242 | 1323 + 561 = 1884s | ~127h |
| carnator_boundary_b5 | 300 | 1164 + 472 = 1636s | ~136h |

五个方法并行于 GPU 3–7 时，整体墙钟受最慢的 carnator 支配，约 **5–6 天**（且这 5 张卡期间
不能做别的）。**这个量级需要先和你确认再启动**；若不可接受，可讨论降分辨率、降帧数或
减少请求数的替代口径。
