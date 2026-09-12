# LTX-Video (0.9.7-dev) 基线适配规划

在 LTX-Video-0.9.7-dev 上复现 Wan2.1 已有的五个基线（baseline / nirvana_vid / chorus /
chorus2 / carnator_boundary_b5），**算法逻辑完全不变**，只重写驱动 DiT 的代码与实验脚本。
实验流程、质量评估、聚合报告与 10/300 请求两个脚本全部对齐 `docs/WAN21.md`。

## 1. 已完成的可行性核查

| 项 | 结论 |
|----|------|
| 推理后端 | `models/LTX-Video-0.9.7-dev` 是 **diffusers 格式**（`LTXConditionPipeline` + `LTXVideoTransformer3DModel` + `AutoencoderKLLTXVideo` + `FlowMatchEulerDiscreteScheduler`），**不走** `external/LTX-Video`（该子模块是 0.9.8/2B 旧仓库，无 0.9.7 配置） |
| 权重兼容性 | `wan` env 的 diffusers 0.35.2 从 config 建出的模型 **1215/1215 权重键与 checkpoint 完全一致**，13.04 B 参数 ✅ |
| 环境阻塞 | `~/.local/lib/python3.10/site-packages/diffusers`（0.40.0）会 shadow `wan` env 的 0.35.2，且与 env 内 `huggingface_hub` 0.36.2 不兼容 → **已定：复用 `wan` env，所有 LTX 入口脚本与 subprocess 调用固定加 `PYTHONNOUSERSITE=1`** |
| T5 权重缺失 | `models/LTX-Video-0.9.7-dev/text_encoder/` **只有 config.json，没有权重**；T5-v1.1-XXL(fp16, 220 keys) 在 `models/t5xxl_fp16.safetensors`，需在 `prepare()` 中单独加载 |
| GPU | 8× H100 80GB；沿用 Wan2.1 的 GPU 3–7 一方法一卡 |

## 2. 与 Wan 的差异清单（适配点）

只列需要改代码的差异，其余沿用。

| # | 差异 | 处理 |
|---|------|------|
| D1 | 采样器：`FlowMatchEulerDiscreteScheduler`（一阶流匹配）vs Wan UniPC 多步 | nirvana 从 cached latent 续跑只需对齐 sigmas；**Chorus Stage3「step 30 重置 UniPC 历史」在 LTX 上是空操作**，需在文档和 `extra` 中显式标注 |
| D2 | 引导：0.9.7 需要 **CFG + STG**（`stg_mode: attention_values` + `skip_block_list` + `guidance_timesteps` 分段映射），diffusers 的 `LTXConditionPipeline.__call__` **只有 CFG、没有 STG** | 所有方法（含 dense baseline）统一走自写的采样循环，循环内实现 CFG+STG；这是保证 baseline 与各方法同口径比较的前提 |
| D3 | latent：128 通道 / 48 层 / 3D RoPE patchifier / VAE 时空压缩 | 各方法里写死的 Wan token 网格、`vae_stride`、`patch_size` 换成由配置计算；`save_latent`/`load_latent` 按新 shape 序列化 |
| D4 | 注意力 hook 点：diffusers `LTXVideoTransformerBlock` 的 `attn1`(self)/`attn2`(cross)，`LTXVideoAttnProcessor2_0` 调 `F.scaled_dot_product_attention`；签名无 Wan 的 `context_lens/freqs` | chorus2 稀疏路由 / carnator 区域稀疏与 TGAA 的 hook 改写到 LTX block forward |
| D5 | 文本编码器：T5-v1.1-XXL(4096) vs Wan umt5，需 `prompt_attention_mask`，`max_sequence_length=256` | 文本侧写进 adapter，Chorus 的 prompt-diff/TGAA 改为基于 T5 token 序列 |
| D6 | VAE 解码带 `decode_timestep` 条件 + latent normalize | decode 路径按 `AutoencoderKLLTXVideo` 写 |
| D7 | 视频写出：diffusers `export_to_video`（Wan 用 `cache_video`） | `save_video` 直接调 `export_to_video`，帧率写进 `ltxvideo.yaml` |

## 3. 文件改动

新增：

```
configs/models/ltxvideo.yaml                    # 模型 + STG 调度 + 五个基线常量
adapters/ltxvideo.py                            # LTXConditionPipeline adapter + 五个基线
adapters/ltx_pipeline.py                        # 管道委托：编码 / 自写采样循环 / 解码 / pack-unpack
adapters/ltx_attention.py                       # LTX attention 处理器与 qkv 视图（chorus2 / carnator 共用）
adapters/ltx_sparse.py                          # LTX native block forward + 稀疏执行器（chorus2 / carnator 共用）
carnator/ltx_attention_probe.py                 # carnator 的两个探针（cross-attention 种子 / self-attention 对比）
carnator/ltx_quantized_ffn.py                   # carnator w8a8 FFN（LTX 版）
scripts/benchmark/run_ltxvideo_all_test.sh      # 10 请求测试（对齐 run_wan21_all_test.sh）
scripts/benchmark/run_ltxvideo_all_formal.sh    # 300 请求正式（对齐 run_wan21_all_formal.sh，支持 --resume）
docs/LTXVIDEO.md                                # 本文件
```

改动（全部为分发/登记，不动算法）：

| 文件 | 改动 |
|------|------|
| `scripts/benchmark/run_vidprom_300_experiment.py` | `--model` 增加 `ltxvideo`；新增 `LTX_GPU_BY_METHOD`；`ltxvideo_run()` 复用现有 replay runner 分支；`flatten_replay_payloads()` 从 `wan21_run` 中抽出共用；`resolved_config()` 的路径绝对化加入 `t5_weights`；`target_probe_*` 对 LTX 关闭（LTX 的 Early Signature 在 source 阶段自建，无独立 target probe） |
| `scripts/benchmark/aggregate_vidprom_300_results.py` | `MODEL_METHODS["ltxvideo"]` 增加一行 |
| `core/registry.py` | `REAL_SUPPORTED` 为五个方法登记 `ltxvideo` |
| `configs/methods/matrix.yaml` | 五个方法各增加 `ltxvideo` 行 |
| `adapters/__init__.py` | `make_adapter()` 增加 `ltxvideo` 分支 |
| `methods/chorus/method.py` | 直通分支的模型判断加入 `LtxVideo` / `ltxvideo` |
| `core/wan21_runner.py` | **不改算法**；`Wan21ReplayRunner` 本身与模型无关（已参数化 `model`），LTX 直接复用并把 label 传成 `LTX-Video-0.9.7-dev`。若觉得命名别扭，可原地改名为 `ReplayRunner` 并留别名 |

## 4. 实施阶段

| 阶段 | 内容 | 验收 |
|------|------|------|
| P0 | `PYTHONNOUSERSITE=1` 下确认 diffusers 0.35.2 可用、T5 权重从 `t5xxl_fp16.safetensors` 加载成功、`export_to_video` 可用 | `--check-only` 通过 |
| P1 | `LtxVideoAdapter` dense baseline（自写 CFG+STG 循环，768×512×121@50 步） | 2 请求 smoke，人眼确认视频质量、记录 `dit_forward_s` |
| P2 | `nirvana_vid`（复用 step-5 latent） | 命中/未命中各 1 条 |
| P3 | `chorus`（SRD + TGAA + latent 融合；Stage3 scheduler 重置标注为空操作） | 同上 |
| P4 | `chorus2`（全层稀疏注意力） | 同上 |
| P5 | `carnator_boundary_b5`（B5 探针 / 软 latent 融合 / 区域稀疏 + w8a8 FFN） | 同上 |
| P6 | 接 `run_vidprom_300_experiment.py` + 聚合，写 `run_ltxvideo_all_test.sh` | 10 请求全跑通并出报告 |
| P7 | 300 请求正式实验 + 汇总表 | 生成 `results_ltxvideo/` 与报告 |

**当前状态**：P0–P6 代码与脚本已写完，**尚未在 GPU 上跑过任何真实请求**（10 请求测试待审核后由用户决定是否在 GPU 3–7 启动）。
已完成的离线验证：全部改动文件 `py_compile` 通过；五个方法在 `--dry-run`（synthetic adapter）下走通
`ltxvideo_run` → `flatten_replay_payloads` → 聚合入口（聚合器按设计拒绝 dry-run 结果）；GPU 固定校验
能按 `LTX_GPU_BY_METHOD` 正确报错。

**回放真值不变**：命中/未命中决策仍复用
`artifacts/vidprom_1000cache_300input_noun_match/cogvideox15_replay/{method}.jsonl`，
保证三个模型在同一批 source/target 上可比。

## 5. 已确定的实验参数

**推理预算：768×512 × 121 帧 × 50 步**（`configs/models/ltxvideo.yaml` 的 `width/height/frames/baseline_steps`）。
各基线的步数常量按同一调度对齐（nirvana `reuse_start_step=5`、chorus `k1=5/k2=30`、
chorus2 `sparse_start_step=10`、carnator `probe_step=5/reuse_step=10`），与 Wan2.1 保持可比的
相对位置。

**运行环境：`wan` env + `PYTHONNOUSERSITE=1`。** 该变量必须同时作用于：
入口脚本、各方法原生 CLI 的 subprocess 调用（`run_vidprom_300_experiment.py` 内的
`native_run`/chorus2/carnator 分支），以及质量评估的 `conda run -n vbench` 之外的
Python 调用链。遗漏任一处会重新踩到 0.40.0 shadow 导致的 import 崩溃。

## 6. 默认采用（如无异议）

1. **baseline 带 STG**（D2），保证 baseline 与各方法同口径。
2. **基线范围先只做 Wan2.1 的五个**；`chai` / `teacache` 待五个跑通后再评估是否补。

## 7. 实现记录（与本节前述规划不一致之处，以实现为准）

| 项 | 规划 | 实现 |
|----|------|------|
| 帧率 | 沿用 Wan 的 24 | **25**。`LTXConditionPipeline.__call__` 的 `frame_rate: int = 25` 会做 `video_coords[:, 0] *= 1/frame_rate`，改成 24 就偏离了 stock pipeline，因此 `ltxvideo.yaml: fps` 固定 25 |
| STG 实现方式 | 未定 | **每步单次 batched forward**：`num_conds = 1 + do_cfg + do_stg`，batch 顺序 `[neg, cond, cond_perturbed]`；跳过层用 `skip_layer_mask[block, ptb_index::num_conds] = 0`；在 `attn1` 的 `to_out[0]` 之前做 `hidden = hidden_a * mask + value_for_stg * (1 - mask)`。STG 开关在 `ltxvideo.yaml` 的 `stg.enabled` |
| LTX native block forward | 计划放 `carnator/adapters/ltxvideo_native.py` | 实际放在 `adapters/ltx_sparse.py`（`adapters/ltx_attention.py` 提供 qkv 视图），chorus2 与 carnator 共用 |
| Chorus Stage3 scheduler 重置 | 计划「标注为空操作」 | FlowMatchEuler 无历史状态，重置确实无操作，已在请求 `extra` 中记录 |
| Carnator Early Signature | 计划沿用 Wan 的 two-probe 结构 | 沿用，落到 `carnator/ltx_attention_probe.py`：cross-attention 探针出 noun 种子 → self-attention 对比探针出 `self_contrast_max` → `build_canonical_fusion_mask` 出 `Mbase`。Wan 的自对比探针按 query 分块（`chunk_size=512`）且只对 noun 的 ~2% 种子 key 取 max，单块约 8 MB，因此在 LTX 的 6144 token 上内存可行 |
| step-5 latent 落盘 | 未定 | `write_early_signature` 要求 `latent_shape` 为 4 元，故先把 packed latent unpack 成 `[128,16,16,24]` 再写，`latent_token_grid` 仍为 `(16,16,24)`；`apply_latent_injection` 无需改，直接吃 unpack 后的 `[C,F,H,W]` |

## 8. 首轮 10 请求排障记录

第一次在 GPU 3–7 上跑 10 请求测试时五个基线全部失败，定位并修复了四个问题（均为 LTX 适配层自身的 bug，未改动任何基线的算法逻辑）：

| # | 现象 | 根因 | 修复 |
|---|------|------|------|
| B1 | 五个基线全部 `OutOfMemoryError`，显存占满 79 GiB | `denoise` 循环没包 `torch.no_grad()`。采样器把每步输出喂回下一步，autograd 图跨 50 步串起来，所有中间激活被保留；单步前向只要 1 GiB，50 步累计到 78 GiB | `_forward` 内加 `torch.no_grad()`（Wan2.1 基线同样在循环外包裹）。修复后峰值 38.5 GiB |
| B2 | chorus2 target 全部 `NameError: name 'torch' is not defined` | `LtxFlexExecutor._block_mask` 是 staticmethod，只 import 了 `BlockMask`，却用了 `torch.int32` / `torch.int8` | 补 `import torch` |
| B3 | chorus2 target 全部 `InductorError: Q and KV block size must be divisible by BLOCK_M and BLOCK_N` | `q_block_size=64` 沿用了 `carnator/outer_sparse.py` 的 dataclass 默认值，而 flex attention kernel 的 BLOCK_M=128，64 不能被整除 | `ltxvideo.yaml` 的 chorus2 / carnator_b5 改为 `128/128`，与 `cogvideox15.yaml` 一致（6144 % 128 == 0） |
| B4 | carnator source 全部 `KeyError: 1` | `_carnator_source` 只按 `capture_steps=(probe_step,)` 捕获了 step 5，却要写 50 步完整 trajectory，`captured[1]` 不存在 | 改为 `capture_steps=tuple(range(1, steps + 1))`；Early Signature 仍从同一个 dict 里取 step 5 |
| B5 | carnator / chorus 跑通但 `regional_calls: 0`、`tgaa_calls: 0` | `LtxRegionalState.active` 只由 `state.set_step()` 置位，而两个 target 的 `begin_step_hook` 只喂了 `LtxStepTracker`，区域稀疏与 TGAA 整条路径从未激活 | chorus target 直接 `begin_step_hook=state.set_step`（与 CogVideoX 的 `on_begin=region.set_step` 同构）；carnator target 的 hook 同时驱动 tracker 与 state。窗口语义用 0-based step index，与 latent 融合窗口 `[K1, K2)` 对齐 |

修复后的 10 请求测试：五个基线 **0 error**，`carnator` 的 regional_calls / injection / outer-sparse 计数与 `chorus` 的 TGAA 计数均与理论值吻合（见第 5 节的窗口推导）。

调试辅助：`WAN21_RUNNER_TRACEBACK=1` 时 `Wan21ReplayRunner` 会把每个请求失败的完整 traceback 打到 stderr（默认关闭，不影响结果表中的一行式错误信息）。

## 9. 与 Wan 对齐修正（第二轮）

第二轮核查把 LTX 适配逐条对照 `configs/models/wan21.yaml` 与 Wan2.2 的同名基线，发现并修正了以下三点。三处都是**适配层自身的偏差**，算法逻辑未改动。

### 9.1 outer-sparse 后端：FlexAttention → SpargeAttn（修正 A/C）

| 项 | 修正前 | 修正后 |
|----|--------|--------|
| 后端 | `torch.compile(flex_attention)`（Triton，FP16） | `spas_sage_attn.block_sparse_sage2_attn_cuda`（INT8 QK + FP8 PV，sm90） |
| 分块 | `q_block_size: 128` / `k_block_size: 128` | `q_block_size: 64` / `k_block_size: 128` |
| 配置键 | 无 | 无——SpargeAttn 是 LTX replay 的唯一后端 |

chorus2 全 48 层、carnator 的外稀疏 32 层（`0-15,32-47`）此前都走 flex 路径，而 Wan2.1/Wan2.2 的同名基线走的是 SpargeAttn。这是 chorus2 在 LTX 上看不到加速的直接原因：flex 只有 FP16、拿不到量化收益，而且每次调用都要重建一次 `BlockMask`，在 6144 token 上比 dense 基线用的 cuDNN/flash SDPA 每 FLOP 更慢。此外 LTX 的 `attn1` 只占单步 FLOP 的约 16 %（Wan2.1 约 84 %，序列 6144 对 109120），这是结构性的收益上限。

**flex 已从 LTX 路径整个删除**，不是保留为消融选项。保留没有意义：Triton flex-attention 的 `BLOCK_M = BLOCK_N = 128` 是硬编码的，64 不能整除（B3 那条 `InductorError` 就是这么来的），所以它既跑不了 SpargeAttn 要求的 64/128 几何，也不能和 SpargeAttn 共用同一份 mask 缓存（缓存的 `metadata.json` 里记着 `mask` 几何，不匹配会拒绝加载）；而换成 128/128 后它又与 Wan 基线的分块不一致，结果本就不可比。`LtxSparseExecutor` 因此不再有 `backend` 参数、`_flex`/`_warm_flex`/`_block_mask` 与 `BACKENDS` 校验，`LtxBlockMaskConfig`／缓存名里的 "BlockMask" 指的是稀疏路由掩码本身（用 bit 打包存盘），与 torch 的 `BlockMask` 无关。

B3 那条「64 不能被 BLOCK_M=128 整除」是 flex 后端的要求，换成 sparge 后不再适用——sparge 恰恰要求 Q=64/K=128，所以这次把 `128/128` 改回 64/128，并删掉了 `cogvideox15.yaml` 的对齐理由。

`LtxFlexExecutor` 相应改名为 `LtxSparseExecutor`，并新增 `prepare()`：导入 SpargeAttn + sm90/分块校验，同时把 mask 缓存的解码搬到计时循环外。

配套修掉了一处**被计时的重复开销**：`LtxBlockMaskCache.get()` 原本每次调用都重新读文件、重算 sha256、重跑 `np.unpackbits`，而 target 阶段会按 (step, branch, layer) 调用 3840 次（carnator 2560 次），这些全部落在 `dit_forward_s` 里。Wan 的 `SourceBlockMaskCache` 不是这么做的：它用 `_prepared_masks` 记忆化解码结果，并由 `prepare(device=...)` 在计时循环外一次性解码并搬到目标设备。现在 `LtxBlockMaskCache` 采用同一协议（`_decode` + `prepared` 记忆化 + `prepare()`），`LtxSparseExecutor.prepare()` 会连带调用它。LTX 上一次解码 3840 张 mask 约 0.6 s，这是唯一一次。

> **重跑说明**：块几何变了，用 128/128 捕获的旧 mask 缓存与新配置不兼容，`LtxBlockMaskCache` 会报 `metadata does not match the target`。不过实际驱动方式下不会碰到：每个 worker 的源缓存在 `<output-root>/<method>/tmp_cache/` 下，阶段跑完即清空（实测 10 请求结果目录与本次 smoke 目录里该目录均为空），所以每次运行本来就会重新做完整源捕获。只有在自行保留缓存目录、或对旧目录用 `--resume` 时，才需要先删掉 `chorus2_source` / `carnator_source`。

### 9.2 carnator target 的 5 步 dense probe（修正 B）

Wan2.1 的 carnator target 在复用前会先跑 `mask_probe_step` 步 dense 前缀（`run_wan21_target_probe`，并把它写进 target 目录的 Early Signature），且这段时间和步数都计入 target 的 `dit_forward_s` / `executed_steps`。LTX 此前完全跳过了这段，于是两边的计时口径不可比。

现在 `_carnator_target` 增加 `_carnator_target_probe`，语义与 Wan 逐项对齐：

- 跑完整 50 步计划的**前 5 步**，不是把 5 步重新缩放成一条 5 步计划。为此 `LtxVideoPipelineDelegate.denoise` 新增 `schedule_steps` 参数：`steps` 是执行步数，`schedule_steps` 是该步数所属计划的总长，时间步与 `normalized_time`（STG 分段按它索引）都取自 50 步计划。Wan 的探针是同构的——枚举完整 `scheduler.timesteps` 并在 `mask_probe_step` 处 break。
- 用固定的实验 seed（`config.seed`）而非请求 seed，对应 Wan 探针的 `target_probe_seed`。
- 在 `TargetProbe` 目录写自己的 Early Signature，`compare_early_signatures` 的结果只作审计元数据，异常被吞掉，不影响复用决策——与 Wan 一致。

Early Signature 的构造逻辑（cross 种子 → self 对比 → `Mbase` → `write_early_signature`）此前内联在 `_carnator_source` 里，现提取为 `_carnator_early_signature`，源与探针共用同一份实现，避免两边漂移。

target 的计数改为与 Wan 相同：`executed_steps = probe_steps + reuse_steps`、`dit_forward_s = forward_s + probe_dit_forward_s`，并在请求 `extra` 中记录 `probe_steps` / `probe_dit_forward_s` / `probe_comparison` / `cache_prepare_s`。

### 9.3 复核确认无偏差的两项

- **latent / step 约定**：`captured[completed] = index + 1`、Chorus 的 `trajectory_paths[k1]`、Stage2 窗口 `k1 <= index < k2`、融合条件 `k1 <= completed_step-1 < k2`、Carnator 的 `trajectory_paths[reuse_step-1]`，均与 Wan2.1 逐条一致，无差一错误。
- **常量**：chorus 的 k1=5/k2=30 是 Wan2.1 的值（Wan2.2 用 10/40），已在 `ltxvideo.yaml` 注明「mirror configs/models/wan21.yaml」，属既有的 Wan2.1/Wan2.2 差异，不是 LTX 缺陷；nirvana、chorus2、carnator 的常量在 Wan2.1 与 Wan2.2 上相同。

## 10. 修正后的验证结果

### 10.1 10 请求正式结果（`--limit 10 --overwrite`，覆盖 `results_ltxvideo_test`）

两个方法在 GPU 6/7 并行重跑「源捕获 + target 复用 + 质量评估」，基线三方法未改动，因此 dense 对照与修正前同源（79.37 s）。日志见 `artifacts/ltxvideo_experiment_logs/test_{chorus2,carnator_boundary_b5}_20260912_114501.log`。

| 方法 | 去噪 s/request | 去噪加速比 | E2E s/request | E2E 加速比 | 命中 | CLIP-T | VBench-Q |
|------|---------------:|-----------:|--------------:|-----------:|-----:|-------:|---------:|
| Dense Baseline | 79.37 | — | 81.42 | — | — | 0.2608 | 0.5896 |
| chorus2 修正前 (n=8) | 78.47 | 1.011x | — | — | 8/10 | — | — |
| **chorus2 修正后** | **75.26** | **1.055x** | 77.69 | 1.048x | 8/10 | 0.2771 | 0.6328 |
| carnator 修正前 (n=10) | 57.48（复用 40 步，无探针） | 1.381x | — | — | 10/10 | — | — |
| **carnator 修正后** | **62.53**（复用 54.73 + 探针 7.80） | **1.269x** | 68.15 | 1.195x | 10/10 | 0.2577 | 0.5922 |

拆开看这次修正对 **target 复用阶段本身**的影响（把修正 B 新增的探针时间扣掉，保证两边口径一致）：

| 方法 | 复用阶段去噪（修正前 → 修正后） | 变化 |
|------|-------------------------------|------|
| chorus2 | 78.47 → 75.26 | −4.1 % |
| carnator | 57.48 → 54.73 | −4.8 % |

即 A/C（SpargeAttn + 64/128）与 mask 缓存记忆化合起来，在 target 阶段拿到约 4–5 % 的真实收益。carnator 表观变慢完全来自修正 B：它现在要付 Wan2.1 同样要付的 5 步 dense 探针（7.80 s），这是用户明确要求的计时对齐的代价。

chorus2 的 1.055x 已接近 §9.1/§10.2 算出的约 1.07x 结构性上限。质量指标没有退化：chorus2 的 CLIP-T / VBench-Q 是五个基线里最高的 0.2771 / 0.6328。

### 10.2 单请求冒烟（`--limit 1`，通路验证）

| 方法 | target `dit_forward_s` | 相对 dense | 后端 | runs | 运行时密度 |
|------|----------------------|-----------|------|------|-----------|
| chorus2 修正前 | 78.47 | 1.011x | torch-compiled-flex-attention | 3840 | 0.6132 |
| chorus2 修正后 | 75.23 | 1.055x | block-sparse-sage2-attn-cuda | 3840 | 0.6006 |
| carnator 修正前 | 57.48（= 复用 40 步，无探针） | 1.381x | torch-compiled-flex-attention | 2560 | — |
| carnator 修正后 | 63.70（= 复用 55.89 + 探针 7.81） | 1.246x | block-sparse-sage2-attn-cuda | 2560 | 0.6102 |

（同配置重复跑的单请求差值约 0.5 %；n=1 与 10 请求均值一致到 0.05 s 以内。）

两张计数都对得上理论值：chorus2 `runs = 40 × 2 × 48 = 3840`、`refresh_events = 40 × 2 × 6 = 480`；carnator 外稀疏 `runs = 40 × 2 × 32 = 2560`、`refresh_events = 40 × 2 × 4 = 320`、`executed_steps = 5 + 40 = 45`、注入 40 次、`regional_calls` 640（与修正前一致，区域路径未动）。

carnator 的 target 时间**变大**是预期的：修正 B 之前它没有跑那 5 步 dense 探针，现在跑了，7.80 s（单请求 7.81 s）计入 `dit_forward_s`——这正是与 Wan 拉齐计时口径的代价。复用阶段本身从 57.48 s 降到 54.73 s（单请求 55.89 s）。

单独微基准（`[1, 32, 6144, 128]` fp16，H100，均值 60 次）：dense SDPA 2.658 ms/次，sparge-sage2 在密度 0.60 时 1.120 ms/次（**2.37x**），密度 0.95 时 1.639 ms/次（1.62x）。`attn1` 一步 48 层 × 2 分支约 255 ms，而整步约 1550 ms，即 **attn1 只占单步 16 %**——所以即便 kernel 快 2.37x、且只在 40/50 步生效，chorus2 target 的结构性上限也只有约 1.07x。实测 1.055x（10 请求）已接近该上限，剩余差距来自真实掩码（top_p=0.95 的散乱分布）比微基准里的连续块掩码更难跑满。

## 11. 回归测试

LTX 适配的关键不变量由三个测试文件锁定，均不加载模型（`pytest tests/` 全绿，227 passed）：

| 文件 | 锁定内容 |
|------|---------|
| `tests/test_ltx_sparse.py` | block mask 打包格式与校验和、跨几何回放拒绝、`prepare()` 记忆化后热路径零 I/O、SpargeAttn 的 CUDA/sm90/64-128 三重守卫、`run()` 的 HND 批处理与 FP16 转换、层刷新语义、处理器按分支各跑一次 |
| `tests/test_ltx_attention.py` | `Medit ⊆ Msee` 的索引语义与缓存、TGAA 的差分 token 键缩放与输出缩放、`stg_mix` 只替换扰动行、区域 block 只回写 Medit 行（其余行逐位不变）、`install_regional_blocks` 的补丁与恢复 |
| `tests/test_ltxvideo_adapter.py` | 五个基线的注册与 TeaCache 的拒绝、`resolved_config` 的路径绝对化与 64/128 几何、pointer/trajectory 两种 latent 序列化、denoise 的 `schedule_steps` 前缀口径（§9.2 修正 B）、分支标签来自切片而非计数 |

其中 `schedule_steps` 那一项是修正 B 的回归保护：探针必须跑 50 步计划的前 5 步，而不是把 5 步重新缩放成一条短计划。
