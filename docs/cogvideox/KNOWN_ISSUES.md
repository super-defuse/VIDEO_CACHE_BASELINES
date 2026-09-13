# CogVideoX 适配：验证状态与已知问题

> 状态截至 2026-09-12。CogVideoX 适配**尚未完成 300 请求正式实验**，本文档明确区分
> 「已验证」「未验证/缺口」和「已修复的历史问题」，避免把计划当作结果。

## 1. 已验证的部分

| 项目 | 证据 |
| --- | --- |
| 五个基线 `--check-only` 全部 `ready` | `run_vidprom_300_experiment.py --model cogvideox15 --check-only` |
| 全仓单元测试通过 | **234 passed**（2026-09-12；`PYTHONNOUSERSITE=1 /data/shared/conda_envs/wan/bin/python -m pytest tests/ -q`） |
| CogVideoX token 几何由实测确定 | 44880 video tokens（grid `(11,48,85)`）+ 226 text tokens = 45106，与 transformer 实际 reshape 一致；见 [FUSION_MASK.md](FUSION_MASK.md) |
| fusion mask 全链路可用 | `tests/test_cogvideox15_adapter.py` 的 7 个新回归测试：probe 记录、cond 行、fail-closed、token grid、caption 索引、token→latent 投影、per-noun Mbase → target 选择 |
| 命中/未命中的 mock 协议正确 | `test_replay_runner_separates_source_time_and_always_cleans_tmp_cache`（miss 的 `executed_steps == 0`）、`test_replay_runner_preserves_hit_when_source_execution_errors` |
| Chorus / Chorus2 / Carnator 通过 256x256、9 帧、2 步 dense smoke 与小尺寸 backend smoke | backend smoke 日志 |
| 五基线 10 样本 replay 部分跑通，miss 正确记为 `cache_miss` | `artifacts/cogvideox15_5baselines_10samples/` |
| **10 请求测试整体通过（2026-09-12）** | `results_cogvideox15_test/`：五方法 50 条请求 0 error，状态分布与 replay 逐条吻合，`verify_replay_status_distribution.py` rc=0，聚合器 rc=0 |
| Carnator 在真实权重上产出 fusion mask | 10 请求测试里 `carnator_boundary_b5` 10/10 命中并产出视频，走完整 probe → 每名词 Mbase → target 选择链路（此前只有 mock 覆盖） |
| 3 条真实请求跑通（2 命中 1 miss） | `artifacts/vidprom_1000cache_300input_noun_match/cogvideox15_3req_test_20260908_151456/` |
| replay 生成器可用 | `scripts/benchmark/build_cogvideo_hit_replay.py`，从旧 Wan 结果提取 hit/miss 与 source UUID |

## 2. 未验证 / 缺口

| 缺口 | 说明 |
| --- | --- |
| **300 请求正式实验未运行** | 本轮未启动任何 CogVideo 300 条正式请求，明确停在正式运行之前 |
| **10 请求的结论不能当方法排名** | 每方法只有 10 条请求（命中 8 或 10 条），质量分噪声极大；性能排序只作趋势参考，正式结论以 300 请求为准 |
| **fusion mask 未做可视化核验** | 真实权重下已跑通并产出 `carnator_fusion_masks/` bundle，但尚未渲染叠加图人工核对区域选择是否正确 |
| **质量评测仍属小样本** | 五基线 10 请求的 CLIP-T 0.238–0.310 / VBench-Q 0.569–0.640，与 3 请求测试同一量级；300 条规模无数据 |
| **CogVideoX 绝对耗时远高于旧评估** | 1360×768 下 50 步 dense 实测 ~761s（nirvana 命中 ~684s@45 步），比旧 smoke 数据慢一个量级；下文 §5 的成本预算要按此重估 |

> 仓内 `smoke_two_requests_20260830_final/` 与 `smoke_two_requests_20260830_final_pre_forced_smoke/`
> 两个目录的 `results.jsonl` 里 `model` 均为 **Wan2.2**（字段名也是 Wan runner 的
> `reported_latency_s` / `excluded_cache_prepare_s`），**不是 CogVideoX**。它们曾被当作
> CogVideoX smoke 性能引用，已更正；需要 Wan2.2 的 smoke 数据时可直接读这两个目录。
>
> `artifacts/vidprom_1000cache_300input_noun_match/carnator_mask_probe_gpu012_20260901/` 的
> manifest 显示 `model_dir` 为 **Wan2.2-TI2V-5B**，同样不能作为 CogVideoX mask 的证据。
>
> 这类误读有一个结构性根因，已修复：`cogvideo_run()` 构造 `CogVideoReplayRunner` 时没有传
> `model=`，于是继承了基类的默认值 `Wan2.1-T2V-1.3B`，CogVideoX 的每一行结果都会被标成 Wan。
> 现在显式传 `model="CogVideoX1.5-5B"`（见 §3.6）。

## 3. 已修复的历史问题

### 3.1 `KeyError: 'chorus2'` / `KeyError: 'chorus'`（10 样本实验批量失败）

**症状**：`artifacts/cogvideox15_5baselines_10samples` 中，chorus2 与 carnator 的所有命中请求
立即报错，产出 0 个视频；但脚本把"进程未崩溃"记为 `completed successfully`，因此看起来
"chorus2 只用了 26.5s"，实为假象（每个请求 ~0.004s 就返回了）。

| 基线 | 真实状态 | 现象 |
| --- | --- | --- |
| baseline | ✓ 正常 | ~77s/请求，10 个视频 |
| nirvana_vid | ✓ 正常 | ~77s/请求，8 个视频（2 miss） |
| chorus | ✓ 正常 | ~180s/请求，8 个视频（2 miss） |
| chorus2 | ❌ 全部立即失败 | `KeyError: 'chorus2'` |
| carnator | ❌ 全部失败 | `KeyError: 'chorus'` |

**根因**：`scripts/run_5baselines_10samples*.py` 的配置拼装错误——
① 对 chorus2 误用 `"chorus"` 作为配置键；② 为 carnator 漏配其当时依赖的 `"chorus"` 键。

**已修复**：配置键改用 `method` 变量本身。同时 Carnator 不再依赖 `chorus` 配置
（见 §3.2）。

### 3.2 Carnator 曾用外部分割，偏离 Wan2.2 算法

**症状**：早期 `carnator/cogvideox15.py` 通过 `adapter.config["chorus"]` 读取配置，调用
`GroundedSam2Segmenter` 与 `build_cogvideo_edit_mask()` 做外部分割——这是 Chorus 的做法，
不是 Carnator 的 fusion mask 算法。

**已修复**：改为纯 fusion mask，不再依赖外部分割模型，`mask_policy` 也从
`cogvideo-fusion-mask-v1` 改为 `cogvideo-native-self-cross-fusion-mbase|...`。

> ⚠️ **但这一版仍然没有真正对齐算法**：它保留了 Grounded-SAM2 之外的错误结构——probe 把
> video token 当 text 喂进 3072 维的 `to_q` 投影、token grid 直接取原始 latent 维度
> `[22,96,170]`、所有名词共用同一个 mask、融合失败的写法是 `print("Warning: ...")` 之后
> 继续产出空 mask。上下文见 §3.5。

### 3.3 命中请求的 `latency_s` 计入了 source 生成成本（2026-09-11 修复）

**症状**：`test_cogvideox15_adapter.py::test_replay_runner_separates_source_time_and_always_cleans_tmp_cache`
稳定失败——`target_e2e_s != latency_s`，差值恰好是 `source_prepare_s`。

**根因**：`core/cogvideo_runner.py` 的 `latency_s` 只扣除 save/hash，没有扣除命中后的
source 准备区间。同仓库另外两个 runner 与协议文档都不是这个口径：

- [VIDPROM_1000CACHE_300INPUT_EXPERIMENT_PLAN.md](../VIDPROM_1000CACHE_300INPUT_EXPERIMENT_PLAN.md)
  第 93 行：`latency_s` / `hit_e2e_s`「仅扣除命中后的 cache prepare 区间，以及视频/缓存写盘和结果 hash」
- `core/wan21_runner.py`：`latency_s = elapsed - source_prepare_s - cache_load_s - cache_prepare_s - save_s - hash_s`
- `core/runner.py`（Wan2.2）：同口径

**已修复**：`latency_s` 扣除 `source_prepare_s`，`target_e2e_s` 直接取 `latency_s`
（与 `wan21_runner.py` 一致）。miss 的 `source_prepare_s` 为 0，不受影响。

**影响**：命中请求的 `latency_s` 会减少 `source_prepare_s`（实测约 74s/请求）。
`cogvideox15_3req_test_20260908_151456` 的数值与修复后口径一致，可以引用；但该次运行的
`denoising_s` 为 0（DiT 计时当时未接线），不要引用其去噪时间。

### 3.4 miss 请求未短路，会跑完整 dense 生成（2026-09-11 修复）

**症状**：`CogVideoReplayRunner.process()` 对 replay 决策为 `miss` 的请求返回
`status="ok"` 而非 `cache_miss`；请求会完整执行 dense 生成、落盘视频并计入耗时。

**根因**：`core/cogvideo_runner.py` 的 `decision == "miss"` 分支只构造了 `CacheMatch`，
没有提前返回，代码继续落到 target 生成段。

> ⚠️ 不能用 `if not match.hit` 短路：TeaCache 的 `CacheMatch(kind="target_only")`
> 同样是 `hit=False`，会被误判成 miss。

**后果**：若直接跑 300 条，所有 miss 会被 `aggregate_vidprom_300_results.py` 计入
`completed`（`status in {"ok","cache_miss"}`）并算作命中，污染 hit rate 与命中耗时均值。
这与 §3.1 是同一类陷阱：表象正常，结论错误。

**已修复**：`miss` 分支直接返回 `status="cache_miss"`、`latency_s=0`、`executed_steps=0`、
`extra={"replay": ..., "skipped": True}`，与 `core/wan21_runner.py`、`core/runner.py` 及
`run_vidprom_300_experiment.py` 中 `miss_policy = "skipped without generation (status cache_miss, latency 0)"`
一致。

> §3.3 与 §3.4 都是 `core/cogvideo_runner.py`（未提交的 WIP 文件，不在 Git 历史中）里的回归：
> `cogvideox15_5baselines_10samples` 当时就能正确产出 `cache_miss`，说明短路口径曾经存在。
> 现在该文件已缩到 37 行——只剩一句继承 `Wan21ReplayRunner` 加一个 cache manifest 校验钩子，
> §3.3/§3.4 那种"两份 runner 各自漂移"的回归在结构上不再可能。

### 3.5 Carnator fusion mask 的四处致命缺陷（2026-09-12 修复）

**症状**：CogVideoX 的 Carnator B5 只在 mock 下跑过，从未在真实模型上产出过一份 mask。
逐行对照 Wan/LTX 后确认旧实现有四处各自足以致命的缺陷：

| # | 缺陷 | 后果 |
| --- | --- | --- |
| 1 | probe 自行拼 `to_q`，把 video token 当 text 传进去（该投影输入是 3072 维，video 是 4096/16 维） | `torch.cat` 立即报错，路径根本跑不通 |
| 2 | `infer_token_grid_shape()` 返回原始 latent 维度 `[22, 96, 170]` | 即使跑通，mask 的空间轴与 token 轴也不对应 |
| 3 | 所有名词共用同一个 mask | 单名词替换（如 cat→dog）时区域选择完全错 |
| 4 | 失败路径是 `print("Warning: ...")` 后继续 | 产出空 mask 静默进入 target，污染结果且不报错 |

**已修复**（细节见 [FUSION_MASK.md](FUSION_MASK.md)）：

1. QKV 改走 `adapters.cogvideo_attention.joint_qkv()`——与 Chorus2/Carnator 稀疏处理器
   替换 diffusers attention 时用的是同一个函数（含 RoPE 与 `norm_q/norm_k`），probe 不可能
   与它要测量的注意力漂移。
2. token grid 由 `cogvideo_token_grid()` 统一算出：81 帧 → 21 latent → 补到 22 → `(11,48,85)`。
3. 每名词一个 Mbase，target 用与 Wan/LTX 共用的 `select_per_noun_inference_mbase()` 选择。
4. `probe.validate()` fail-closed：名词缺 cross/self 记录、token 数与 grid 不符、名词索引越过
   caption 长度、`attention_self_layer < attention_probe_layer`，全部直接抛异常。

同时把旧路径里四个独立实现（`_generate_fusion_mask_from_captured`、
`_generate_fusion_mask_from_latent`、`_boundary_weights`、`_mask`）删掉，收敛成一条走
`build_canonical_fusion_mask()` / `build_per_noun_mbase_bundle()` / `apply_latent_injection()`
的链路；并删掉每次命中都要跑的 81 帧 source 视频编码（CogVideoX 的 mask 不再来自解码帧）。

### 3.6 结果行被标成 Wan2.1（2026-09-12 修复）

**症状**：`cogvideo_run()` 构造 `CogVideoReplayRunner` 时没有传 `model=`，继承基类默认值
`"Wan2.1-T2V-1.3B"`。按 §2 的记载，仓里已经有两个目录因此被误读为 CogVideoX 证据过。

**已修复**：显式传 `model="CogVideoX1.5-5B"`，与 wan21 传 `Wan2.1-...`、ltxvideo 传
`LTX-Video-0.9.7-dev` 的写法一致。

### 3.7 Chorus 在 cogvideo 环境缺 Grounded-SAM-2 依赖（2026-09-12 修复）

**症状**：10 请求测试里 chorus 唯一一路启动即崩：

```
File ".../external/grounded-sam2/grounding_dino/groundingdino/util/inference.py", line 5
    import supervision as sv
ModuleNotFoundError: No module named 'supervision'
```

**根因**：CogVideoX 必须 `PYTHONNOUSERSITE=1`——`~/.local` 里的 diffusers 0.40.0 会盖掉
cogvideo 环境的 0.35.2，而 0.40.0 在该环境连 pipeline 都 import 不了。但 Grounded-SAM-2 的
`supervision` / `iopath` / `portalocker` 三个包**只存在于 `~/.local`**，`PYTHONNOUSERSITE=1`
正好把这条路关掉了。Wan/LTX 没有这个问题，因为它们的整条链路都跑在 `wan` 环境里，而该环境
自带 `supervision` 0.22.0。

> 9 月 9 日那版 `run_5baselines_10samples` 里 chorus 能跑通，是因为当时**没有**设
> `PYTHONNOUSERSITE`，`supervision` 从 `~/.local` 溜了进来（旧日志里能同时看到
> `~/.local/.../timm/` 和 `conda_envs/cogvideo/.../torch/` 两条路径）。换成正式的
> `PYTHONNOUSERSITE=1` 口径后这个隐式依赖才暴露出来。

**已修复**：新增仓库内 overlay 目录 `.cogvideo_overlay/`，只放 cogvideo 环境缺失的那几个包的
符号链接，两个实验脚本通过 `PYTHONPATH` 注入：

```bash
OVERLAY="$ROOT_DIR/.cogvideo_overlay"
export PYTHONPATH="$OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
```

overlay 里的包与 cogvideo 环境**没有任何重叠**（已逐个核对），因此不存在版本遮蔽：
`diffusers` 仍是 0.35.2、`torch` 仍来自 conda 环境。目前收录
`supervision deprecate defusedxml addict yapf yapf_third_party tomli matplotlib pyparsing
cycler dateutil kiwisolver pycocotools timm hydra omegaconf antlr4 iopath portalocker`。

重建方式（换机器或 `~/.local` 变动后）：在 `PYTHONNOUSERSITE=1 PYTHONPATH=.cogvideo_overlay`
下反复 import `groundingdino.util.inference` 与 `sam2.build_sam`，每次把报出的
`ModuleNotFoundError` 顶层包从 `~/.local/lib/python3.10/site-packages` 软链进 overlay，直到
"ALL IMPORTS OK"。目录已进 `.gitignore`（软链指向用户目录，不进版本库）。

## 4. 重构时的建议

1. 先在 `cogvideo` 环境确认模型权重、tokenizer、VAE、text encoder、transformer、scheduler 完整，
   再跑 `--check-only` → dry-run → 单请求 dense smoke。
2. 每个方法先做 forced miss/hit 两条 smoke，再做 5 条真实请求；检查 latent/token shape、
   scheduler step、seed determinism、GPU 隔离、清理后目录为空。
3. **不要依赖"脚本退出码"判断成功**——§3.1 的教训是进程正常退出但所有请求都在 error。
   两个实验脚本已内置 `verify_replay_status_distribution.py`：逐方法比对 `ok`/`cache_miss`
   与 replay 决策，并拒绝任何 error。
4. 旧 replay 的 hit/miss 是 ground truth，执行失败应记为 `error`，不得改写成 `miss`，
   也不得因 probe 结果改变旧 hit rate。
5. 不得用 Wan 实现、source-latent-restart 兼容版或伪造 hit 结果替代独立的 CogVideo backend。
6. `core/cogvideo_runner.py` 目前不在 Git 历史中，改动容易丢——参考 §3.3 / §3.4 的回归，
   重构前先把它纳入版本管理。

## 5. 10 请求测试实测（2026-09-12，仅作趋势参考）

`results_cogvideox15_test/`，五方法各 10 条请求、50 条全部 0 error。质量分只算生成出来的
视频，miss 不计入。

| 基线 | 命中 | Denoising s/request | Denoising 加速比 | E2E s/request | E2E 加速比 | External s/request | CLIP-T | VBench-Q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense Baseline | N/A | 760.73 | N/A | 785.11 | N/A | 0 | 0.311 | 0.629 |
| NIRVANA-VID | 8/10 | 684.49 | **1.111** | 709.04 | 1.107 | 0 | 0.289 | 0.640 |
| Chorus | 8/10 | 428.55 | 1.775 | 502.65 | 1.562 | 49.74 | 0.238 | 0.569 |
| Chorus2 | 8/10 | 536.25 | 1.419 | 560.93 | 1.400 | 0 | 0.310 | 0.633 |
| Carnator Boundary B5 | 10/10 | 424.70 | 1.791 | 471.54 | 1.665 | 0 | 0.270 | 0.579 |

三条必须在正式实验前先想清楚的读数：

1. **NIRVANA-VID 的加速比 1.111 精确等于 50/45**，即缓存复用除了少跑那 5 步之外没有任何
   额外收益。这是 `reuse_start_step=5` 的协议上限，不是实现缺陷；但把它和 Chorus/Carnator
   放在同一张加速比表里会严重误导，报告里必须说明它的上限来自协议而非缓存质量。
2. **Chorus 的 E2E 反而低于 NIRVANA-VID**（502.65 vs 709.04），虽然它多跑了 LLM + SAM2
   外部推理。原因是 regional attention 把每步的注意力范围收窄到 `Medit/Msee`，单步成本远低于
   dense；省下的钱远多于外部推理的开销。它的 `external_s = 49.74s/request` 是所有方法里唯一
   非零的，且**留在 `latency_s` 内**。
3. **命中请求的真实墙钟成本是 target 的两倍左右**：实测 `source_prepare_s`
   chorus ~758s、chorus2 ~1323s、carnator ~1164s，全部被计时契约排除在 `latency_s` 之外。
   这不违反协议（Wan/LTX 同口径），但 300 请求的预算必须按 `source_prepare + target` 估，
   否则会低估到两倍以上。

> 10 条请求下质量分噪声极大（CLIP-T 极差 0.07、VBench-Q 极差 0.07，方法间排序与命中子集
> 不同强相关），**不要据此判断哪个方法质量更好**。
