# CogVideoX 实验工作流

本文档描述各基线的逐步执行逻辑与产物格式。实验口径与验证顺序见
[EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)；当前完成度见 [KNOWN_ISSUES.md](KNOWN_ISSUES.md)。

## 实验配置

- **模型**：CogVideoX1.5-5B（`models/CogVideoX1.5-5B`）
- **分辨率**：1360x768，**81 frames**，fps 16
- **推理步数**：50，**Guidance scale**：6.0，**Seed**：20260218
- **latent**：`[1, 16, 22, 96, 170]`（21 帧补到 22）；**token grid** `(11, 48, 85)` = 44880 video
  tokens + 226 text tokens，几何推导见 [FUSION_MASK.md](FUSION_MASK.md)
- **输入 replay decisions**：`artifacts/vidprom_1000cache_300input_noun_match/cogvideox15_replay/{method}.jsonl`
  （每个方法一份预先决定的 hit/miss 决策，300 行；与 wan21、ltxvideo 共用同一份）
- **Source prompts**：`artifacts/vidprom_1000cache_300input_noun_match/source_view/index.jsonl`

五个基线分别是 `baseline` / `nirvana_vid` / `chorus` / `chorus2` / `carnator_boundary_b5`。
`baseline` 没有 replay 决策文件，它按定义跑满每一条请求。

## 运行脚本

| 脚本 | 用途 |
| --- | --- |
| `scripts/benchmark/run_cogvideox_all_test.sh` | 10 请求 smoke：五个基线并行（GPU 3–7），跑完做状态分布校验与聚合 |
| `scripts/benchmark/run_cogvideox_all_formal.sh` | 300 请求正式实验：同上，`--resume` 支持断点续跑 |

```bash
# 10 请求测试
bash scripts/benchmark/run_cogvideox_all_test.sh
# 300 请求正式实验
bash scripts/benchmark/run_cogvideox_all_formal.sh
```

两者都 `export PYTHONNOUSERSITE=1`（用户的 `~/.local` 有 diffusers 0.40.0，会盖掉
cogvideo 环境里的 0.35.2，而 0.40.0 在该环境连 pipeline 都 import 不了）。
只用一个方法时可直接调 worker：

```bash
PYTHONNOUSERSITE=1 /data/shared/conda_envs/cogvideo/bin/python \
    scripts/benchmark/run_vidprom_300_experiment.py \
    --model cogvideox15 --method chorus2 --gpu 5 --limit 10 \
    --replay-root artifacts/vidprom_1000cache_300input_noun_match/cogvideox15_replay \
    --output-root artifacts/vidprom_1000cache_300input_noun_match/results_cogvideox15_test \
    --overwrite
```

## 各基线执行逻辑

`decision` 来自 replay 文件。`miss` 一律短路：`status="cache_miss"`、`latency_s=0`、
`executed_steps=0`、无视频、不做任何推理（见 KNOWN_ISSUES §3.4）。下表都只描述 `hit`。

### Baseline (Dense)

```
for each request (300 requests):
    generate(prompt, seed, steps=50)                   # 无检索、无缓存
    save_video(videos/{index}_{uuid}.mp4)
```

`executed_steps=50`，`latency_s = elapsed − save − hash`。

### NIRVANA-VID（`reuse_start_step=5`）

```
if decision == "hit":
    # source：直接跑，在第 5 步完成后中断
    source = generate(source_prompt, steps=50, capture=True)     # _interrupt at completed step 5
    cache  = {step5_latent, scheduler, full_50_step_timesteps}   # 落 tmp_cache/latent.pt
    # target：注入 step-5 latent，在【原 50 步 scheduler 的后缀】上继续
    output = generate(target_prompt, latents=cache.step5_latent,
                      resume original timestep suffix from step 5)  # 45 步
elif decision == "miss":
    skip(status="cache_miss")
```

`_resume_original_schedule()` 把 scheduler 的 `set_timesteps` 换回原 50 步序列并只跑后缀；
cache payload 会校验 `completed_step`、scheduler 名和 timestep 序列长度，**不允许**用
`50 − reuse_step` 新建短 scheduler 冒充 restart。target `executed_steps=45`。

### Chorus（K1=5 / K2=30，取自 `configs/methods/default.yaml`）

```
if decision == "hit":
    # source：完整 50 步，回调保存 completed step 5..30 的 latent
    source = generate(source_prompt, steps=50, capture trajectory[5..30])
    save_video(source) -> tmp_cache/source.mp4
    # target
    prompt_diff = llm_extract_prompt_diff(source_prompt, target_prompt)     # 外部 LLM
    masks = grounded_sam2_segment(source.mp4, removed_entities)            # 外部 SAM2
    install regional attention on ALL 42 joint blocks, active for steps 5..30
      - query = text tokens + Medit tokens；context = text tokens + Msee tokens
      - TGAA：对差异文本 token 的 K 乘 gamma_k，输出乘 gamma_o（线性衰减近似）
    output = generate(target_prompt, latents=source.trajectory[5],
                      every completed step in [5,30]: latents = where(latent_edit, target, source))
                      # step 30 之后恢复 dense
```

命中 ≈ source(50 步) + LLM/SAM2 + target(45 步)。`executed_steps=45`。
两个外部模型是仅有的外部依赖；它们的耗时**留在 `latency_s` 内**，同时以
`chorus_llm_s + chorus_sam_s` 报成扁平字段 `external_s`（与 Wan/LTX 同一口径）。

### Chorus2（`sparse_start_step=10` → 首个稀疏步 11）

```
if decision == "hit":
    # source：完整 50 步，从 completed step 11 起、每个 (step, CFG branch, layer) 捕获 RoPE 后 Q/K
    masks[(completed_step, branch, layer)] = top_p_block_mask(q, k, block=128, top_p=0.95)
    masks 落盘为 bit-packed + SHA256，并校验覆盖度
    # target：仍跑完整 50 步
    steps 1..10   : dense
    steps 11..50  : torch.compile(flex_attention) block-sparse，按 step/branch/layer 取 source routing
                    每 8 层 refresh 一次 block mask
```

`executed_steps=50`，无外部依赖。target 结束时校验 `executor.runs` 等于
`42 × 稀疏步数 × CFG branch 数`，覆盖不全直接报错。

### Carnator Boundary B5（`probe_step=5` / `reuse_step=10`）

```
if decision == "hit":
    # ---- source：一次 dense 50 步，同时采集四样东西
    probe armed at completed step 5 : cross_layer=15 提种子；self_layer=15 出对比度
    trajectory[10..50]              : 每步 latent
    outer masks[0-13, 28-41]        : 从 completed step 11 起的 block-sparse routing
    step5_latent + decoded tail frame
    # 每名词一个 Mbase -- build_canonical_fusion_mask；落 carnator_fusion_masks/
    # 每名词一个 CIELAB 颜色签名 -- write_early_signature；落 source_signature/

    # ---- target
    selection = select one source noun's Mbase     # 名词集合相同→面积最小；单名词替换→被替换者
    probe 5 steps densely (interrupted)            # 与 source 同样的 step-5 probe
    compare early signatures                       # 仅审计，绝不改写固定 replay 决策
    resume from completed step 10:
      regional attention on layers 14-27  (Medit query / Msee context)
      outer sparse on layers 0-13, 28-41  (source routing)
      w8a8 quantized FFN on all 42 blocks
      every step: latent = boundary-blend(target, source_trajectory[step], alpha=smoothstep)
```

`executed_steps = probe_step + (steps − reuse_step) = 45`，`denoising_s` 是 probe 的 DiT
forward 加 reuse 段的 DiT forward（扁平字段 `probe_dit_forward_s` / `probe_steps` 供聚合器
把两者拆开）。无外部模型依赖。

Carnator 的 mask 细节（token 几何、双 hook probe、每名词 Mbase、fail-closed 策略）见
[FUSION_MASK.md](FUSION_MASK.md)。

## 输出结构

```text
artifacts/vidprom_1000cache_300input_noun_match/
  cogvideox15_replay/                    # 只读，与 wan21/ltxvideo 共用
    manifest.json
    {nirvana_vid,teacache,chorus,chorus2,carnator_boundary_b5}.jsonl
  results_cogvideox15_test/              # 10 请求 smoke（run_cogvideox_all_test.sh）
    <method>/{requests.jsonl,request_records.jsonl,results.jsonl,videos/}
    <method>/source_prepare/events.jsonl
    <method>/tmp_cache/<request_index>_<source_uuid>/   # 每次请求后清空
  results_cogvideox15/                   # 300 请求正式实验
```

聚合产物写在 `results_cogvideox15[_test]/` 根下：`summary.csv`、
`hit_rate_over_arrivals.csv`、`hit_rate_over_arrivals.png`、
`VIDPROM_1000CACHE_300INPUT_RESULTS_TABLE.md`。聚合器会先校验五个方法的质量字段是否齐全，
任一方法缺 CLIP-T / VBench-Q 就整体拒绝、不落任何摘要（`--allow-incomplete-quality` 可放宽）。

`tmp_cache` 在 worker 启动时创建，每个请求跑完（成功或失败）都在 `finally` 里删除，
`source_prepare/events.jsonl` 记录 `tmp_cache_removed` 与 `cleanup.status`。

## results.jsonl schema

`requests.jsonl` 与 `results.jsonl` 内容相同（后者是前者的副本，沿用 Wan 的
`quality.py` 契约）。以下是命中请求的示例：

```json
{
  "request_index": 0,
  "uuid": "3384bf4f-88a8-5a0d-9487-f8d853c76d49",
  "prompt": "a puppy dog crying outside the cave in winter",
  "model": "CogVideoX1.5-5B",
  "method": "chorus2",
  "seed": 20260218,
  "cache_match": {
    "hit": true,
    "source_uuid": "3a3813cf-bfb3-5c2a-98a2-fcfb7a6b6eea",
    "source_request_index": 310,
    "kind": "replay_hit"
  },
  "executed_steps": 50,
  "latency_s": 220.844,
  "status": "ok",
  "output_path": "/path/to/videos/0000_{uuid}.mp4",
  "video_path": "/path/to/videos/0000_{uuid}.mp4",
  "quality_input_path": "/path/to/videos/0000_{uuid}.mp4",
  "arrival_index": 0,
  "hit": true,
  "source_uuid": "3a3813cf-bfb3-5c2a-98a2-fcfb7a6b6eea",
  "denoising_s": 201.047,
  "hit_e2e_s": 220.844,
  "hit_denoising_s": 201.047,
  "external_s": 0.0,
  "extra": {
    "source_prepare_s": 110.5,
    "cache_load_s": 1.2,
    "cache_prepare_s": 0.0,
    "denoising_s": 201.047,
    "timing": {"source_prepare_s": 110.5, "cache_load_s": 1.2, "cache_prepare_s": 0.0,
               "target_e2e_s": 220.844, "target_generation_s": 219.6,
               "denoising_s": 201.047, "save_video_s": 0.9, "hash_s": 0.3},
    "cleanup": {"status": "cleaned", "tmp_cache_removed": true}
  }
}
```

顶层扁平字段由 `flatten_replay_payloads()` 从 `extra` 摊平而来，聚合器只读这些：

- `denoising_s` / `hit_denoising_s` — CUDA 同步后的 DiT forward 时间
- `external_s` — Chorus 的 `chorus_llm_s + chorus_sam_s`，其他方法恒为 0
- `hit_e2e_s` — 命中请求的 `latency_s`
- `hit` — 命中标志；miss 为 `false` 且 `denoising_s=null`、`latency_s=0`

`status` 取值：`ok` / `cache_miss` / `error`。error 是终态审计记录，不改写为 miss，
也不计入 hit-rate 分母与质量评分。**miss 不生成视频**。聚合器把 `ok` 与 `cache_miss`
都算作已完成，命中率只数 `ok`，所以 miss 必须走短路而不是回退 dense。

## 计时口径

命中请求：

```
latency_s = elapsed − source_prepare_s − cache_load_s − cache_prepare_s − save_video_s − hash_s
```

即扣除 source 准备区间、cache 反序列化、GPU 侧 cache 准备、视频写盘和结果 hash。
`source_prepare_s` 必须单独报告，不得把源生成成本隐藏在"预置缓存"中；warm-up 不计分
（`write_warmup_record()` 在首个计时请求前把模型加载和 Chorus 的 SAM2 权重加载做掉）。

三点容易误解，都是**跨模型一致**的口径：

1. `excluded_latency_s` 只是上报字段，**replay runner 不扣它**。Chorus 的 LLM+SAM2 时间因此
   留在 `latency_s` 内，同时以 `external_s` 单独可见。（只有 `core/runner.py` 的
   `exclude_preprocess_latency` 开关会真正扣除它，那是 Wan2.2 跨请求缓存的路径。）
2. `cache_load_s`（把 `latent.pt` 读回内存）已扣除；CogVideoX 各基线的 block mask 也在这一步
   从磁盘读出，所以没有额外的 `cache_prepare_s`——LTX 的 Chorus2/Carnator 之所以有，是因为它的
   executor 有一次显式的 GPU `prepare()`。CogVideoX 的 `CogVideoFlexExecutor` 是懒构建索引，
   那部分时间算在 `denoising_s` 里。
3. `latency_s`、`hit_e2e_s`、`target_e2e_s` 三者同值。

## 质量评估与聚合

复用 Wan 的两个通用工具，无需 CogVideo 专用实现，两个实验脚本已自动调用：

```bash
# 阶段 2：质量评估（CLIP-T + VBench-Q，写回 results.jsonl 的 clip_t / vbench_q 字段）
# 由 worker 内部经 conda run -n vbench 触发；CUDA_VISIBLE_DEVICES 会重映射到该方法的 GPU
# 阶段 3：聚合（summary.csv / 结果表 / 命中率曲线）
PYTHONNOUSERSITE=1 /data/shared/conda_envs/cogvideo/bin/python \
    scripts/benchmark/aggregate_vidprom_300_results.py \
    --model cogvideox15 --results-root artifacts/.../results_cogvideox15
```

需要报告的指标：E2E s/request、Hit E2E s/hit、Denoising s/request、
Hit denoising s/hit、External s/request（Chorus 的 LLM+SAM2）、CLIP-T、VBench-Q、Speedup。

## 故障排查

| 症状 | 检查 |
| --- | --- |
| 生成阶段失败 | `nvidia-smi`；模型路径；`artifacts/cogvideox15_experiment_logs/`；results.jsonl 的 `error` 字段 |
| 退出码为 0 但结果不对 | 跑 `verify_replay_status_distribution.py`（两个脚本已自动调用）：逐方法比对 `ok`/`cache_miss` 与 replay 决策，并拒绝任何 error |
| 质量评估失败 | 视频文件是否存在；`vbench` 环境与 CLIP / VBench 权重可用性 |
| 聚合失败 | 所有 results.jsonl 是否已含质量分数；五个方法的输出目录是否齐全 |
| GPU 与计划不符 | `--allow-gpu-remap` 仅在 GPU 属于 3..7 时放行；单方法调试时用它 |

## 用户侧配置依赖

| 基线 | 需要的配置键 | 外部依赖 | 状态 |
| --- | --- | --- | --- |
| baseline | — | 无 | ✅ |
| nirvana_vid | `nirvana_vid` | 无 | ✅ |
| chorus | `chorus`（含 `stages`，取自 `configs/methods/default.yaml`） | LLM + SAM2 | ✅ |
| chorus2 | `chorus2` | 无 | ✅ |
| carnator_boundary_b5 | `carnator_b5` | 无 | ✅ |

**用户自己编写的配置文件被覆盖**：`--check-only` 与 dry-run 会打印最终 resolved config；
`run_manifest.json` 里存有 `config_fingerprint` 与完整 `resolved_config`，出问题时以它为准。
