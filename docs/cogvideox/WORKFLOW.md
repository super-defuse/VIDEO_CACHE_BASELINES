# CogVideoX 实验工作流

本文档描述各基线的逐步执行逻辑与产物格式。实验口径与验证顺序见
[EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)；当前完成度见 [KNOWN_ISSUES.md](KNOWN_ISSUES.md)。

## 实验配置

- **模型**：CogVideoX1.5-5B（`/home/tangxuebin/models/CogVideoX1.5-5B`）
- **分辨率**：480x720，49 frames
- **推理步数**：50，**Guidance scale**：6.0，**Seed**：20260218
- **输入 replay decisions**：`artifacts/vidprom_1000cache_300input_noun_match/cogvideox15_replay/{method}.jsonl`
  （每个方法一份预先决定的 hit/miss 决策）
- **Source prompts**：`artifacts/vidprom_1000cache_300input_noun_match/source_view/index.jsonl`

## 各基线执行逻辑

### Baseline (Dense)

```
For each request (300 requests):
    generate(prompt, seed, steps=50)
    save_video(output_dir/videos/{index}_{uuid}.mp4)
    write_result(results.jsonl)
```

预期 ~77s/请求 × 300 ≈ 6.4 小时。

### NIRVANA-VID

```
decision = replay_decisions[request_index]
if decision == "hit":
    source_latent = generate_until_step(source_prompt, step=5, capture=True)
    save_latent(source_latent)
    output = generate_from_step(target_prompt, start_step=6,
                                cached_latent=source_latent, steps=50)
elif decision == "miss":
    skip(status="cache_miss")
```

命中请求 ≈ Source(5步) + Target(45步) ≈ 77s。注意 source 必须在 CogVideo 原 50-step
scheduler 中捕获 latent，target 在**同一 scheduler 的原始后缀**上继续剩余步骤，不能用
`50-reuse_step` 新建短 scheduler 冒充 restart。

### Chorus

```
if decision == "hit":
    source_result = generate(source_prompt, steps=50, capture_trajectory=True)
    save_trajectory(source_result.latent_trajectory[5:50])
    prompt_diff = extract_prompt_diff(source_prompt, target_prompt)
    mask = segment_video(source_video, removed_entities=prompt_diff.removed)
    save_mask(M_edit, M_see)
    install_regional_attention(layers=14-27, masks=(M_edit, M_see))
    output = generate_with_regional_attention(target_prompt,
                                              source_trajectory=trajectory,
                                              start_step=5)
elif decision == "miss":
    skip(status="cache_miss")
```

命中请求 ≈ Source(50步 ~77s) + LLM/SAM2(~30s) + Target(45步 ~70s) ≈ 180s。
外部依赖：LLM（prompt diff）+ Grounded-SAM2（分割）。

### Chorus2

```
if decision == "hit":
    for step in range(10, 51):        # sparse_start_step=10
        for layer in range(42):       # all layers
            masks[(step, layer)] = capture_top_p_attention_mask(
                q_block=128, k_block=128, top_p=0.95)
    save_attention_masks(masks)
    install_sparse_attention(layers=0-41, masks=masks)
    output = generate_with_sparse_attention(target_prompt, steps=50)
elif decision == "miss":
    skip(status="cache_miss")
```

命中请求 ≈ Source(50步 ~77s) + Target(稀疏50步 ~70s) ≈ 150s，无外部依赖。

### Carnator Boundary B5

```
if decision == "hit":
    trajectory = {step: generate_step_latent(step) for step in range(10, 51)}
    step5_latent = capture_probe(step=5)
    save_trajectory(trajectory); save_probe(step5_latent)

    # mask：纯 fusion mask，由 source 的 joint attention 分数生成，见 FUSION_MASK.md
    self_scores, cross_scores = capture_joint_attention(source, layer=attention_probe_layer)
    mask = build_canonical_fusion_mask(self_scores, cross_scores)
    save_mask(M_base, M_edit, M_see)

    target_probe = generate_until_step(target_prompt, step=5)
    signature_pass = compare_signatures(source_probe, target_probe)
    if signature_pass:
        install_boundary_blend(layers=10-30, masks=(M_base, M_edit, M_see),
                               source_trajectory=trajectory)
        install_sparse_attention(layers=[0-9, 31-39], masks=outer_masks)
        output = generate_from_step(target_prompt, start_step=10)
```

命中请求 ≈ Source(~77s) + Target probe(~40s) + Target gen(~70s) ≈ 190s，无外部模型依赖。

> 早期实现用外部分割（同 Chorus），已重构为纯 fusion mask，见
> [FUSION_MASK.md](FUSION_MASK.md) 与 [KNOWN_ISSUES.md](KNOWN_ISSUES.md) §3.2。

## 输出结构

```text
artifacts/vidprom_1000cache_300input_noun_match/cogvideox15_<run_id>/
├── replay/{nirvana_vid,teacache,chorus,chorus2,carnator_boundary_b5}.jsonl
├── results/<method>/{requests.jsonl,request_records.jsonl,videos/}
├── source_prepare/
├── tmp_cache/<method>/<arrival_index>_<source_uuid>/
└── run_manifest.json
```

`run_manifest.json` 保存旧实验目录与输入文件 hash、replay manifest hash、CogVideo 模型
路径/hash、git revision、resolved config 及计时口径。Wan 目录和共享 cache 只读，
CogVideo 结果写入新的 sibling 目录。

## results.jsonl schema

```json
{
  "request_index": 0,
  "uuid": "3384bf4f-88a8-5a0d-9487-f8d853c76d49",
  "prompt": "a puppy dog crying outside the cave in winter",
  "model": "CogVideoX1.5",
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
  "extra": {
    "denoising_s": 201.047,
    "hit_e2e_s": 220.844,
    "hit_denoising_s": 201.047,
    "external_s": 0.0,
    "source_prepare_s": 110.5
  }
}
```

`status` 取值：`ok` / `cache_miss` / `error`。error 是终态审计记录，不改写为 miss，
也不计入 hit-rate 分母与质量评分。**miss 不生成视频**，返回 `cache_miss`、`latency_s=0`、
`executed_steps=0`、`extra.skipped=true`。聚合器把 `ok` 与 `cache_miss` 都算作已完成，
命中率只数 `ok`，所以 miss 必须走短路而不是回退 dense。

## 质量评估与聚合

复用 Wan 的两个通用工具，无需 CogVideo 专用实现：

```bash
# 阶段 2：质量评估（CLIP-T + VBench-Q，写回 results.jsonl 的 clip_t / vbench_q 字段）
conda run -n vbench env PYTHONPATH=. python scripts/benchmark/quality_runner.py \
    --results <run_root>/<method>/results.jsonl --output-dir <run_root>/<method>

# 阶段 3：聚合（summary.csv / 结果表 / 命中率曲线）
PYTHONPATH=. python scripts/benchmark/aggregate_vidprom_300_results.py \
    --results-root <run_root>
```

需要报告的指标：E2E s/request、Hit E2E s/hit、Denoising s/request、
Hit denoising s/hit、External s/request（Chorus 的 LLM+SAM2）、CLIP-T、VBench-Q、Speedup。

计时口径：denoising 是 CUDA 同步后的 DiT forward 时间。**`latency_s`、`hit_e2e_s` 与
`target_e2e_s` 三者同值，都扣除命中后的 source preparation 区间与视频/hash 写盘开销**
（见 [KNOWN_ISSUES.md](KNOWN_ISSUES.md) §3.3）；source 成本必须单独报告
`source_prepare_s`，不得把源生成成本隐藏在"预置缓存"中；warm-up 不计分。

## 故障排查

| 症状 | 检查 |
| --- | --- |
| 生成阶段失败 | `nvidia-smi`；模型路径；`artifacts/.../<baseline>_gpu{N}.log`；results.jsonl 的 error 字段 |
| 质量评估失败 | 视频文件是否存在；CLIP / VBench 模型可用性；`quality_*.log` |
| 聚合失败 | 所有 results.jsonl 是否已含质量分数；基线输出是否齐全 |

## 用户侧配置依赖

| 基线 | 需要的配置键 | 外部依赖 | 状态 |
| --- | --- | --- | --- |
| baseline | — | 无 | ✅ |
| nirvana_vid | `nirvana_vid` | 无 | ✅ |
| chorus | `chorus` | LLM + SAM2 | ✅ |
| chorus2 | `chorus2` | 无 | ✅ |
| carnator_boundary_b5 | `carnator_b5` | 无 | ✅ |

> 历史上 `scripts/run_5baselines_10samples*.py` 曾对 chorus2 误用 `chorus` 配置键，
> 并为 carnator 漏配 `chorus` 键，导致 `KeyError`；carnator 当时还依赖外部分割。
> 两者都已修复。详见 [KNOWN_ISSUES.md](KNOWN_ISSUES.md) §3.1 / §3.2。
