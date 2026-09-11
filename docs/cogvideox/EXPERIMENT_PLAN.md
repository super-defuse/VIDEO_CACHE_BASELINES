# CogVideoX1.5-5B 300 请求实验计划

## 结论与实验口径

现有 1000-entry cache 是 Wan2.2-TI2V-5B 的中间状态，不能直接加载到 CogVideoX1.5：两者的 VAE、latent channel、帧数、token grid、transformer、scheduler 和 cache schema 都不同。可直接复用的只有 1000 条请求的 UUID、prompt、request index、到达顺序，以及旧实验已经作出的命中/未命中和 source UUID 决策；Wan latent、trajectory、CHAI token、Early Signature、mask、source.mp4 均不能充当 CogVideo cache。

新实验采用固定 replay 协议，不重新计算 CogVideo 命中率：

1. 从 `early_signature_disabled_20260903_100305/` 读取每个方法的 300 条 hit/miss 和 source UUID。NIRVANA-VID、Chorus、Chorus2 的 source UUID 从 `cache_match.source_uuid` 读取；TeaCache 没有跨请求 source hit；Carnator（旧实验禁用 Early Signature）固定为 300 hit。
2. 按 `arrival_index=0..299` 顺序处理。miss 只写 audit record，不运行 source/target；hit 才启动当前请求专属的 CogVideo source。
3. source 使用 source UUID 对应的 1000-entry prompt，在 CogVideoX1.5 上按该方法的原协议重新推理，生成临时 CogVideo cache；随后 target 读取该 cache 完成请求。
4. source 和 target 无论成功或失败都在 `finally` 中清理临时 cache，再处理下一个请求。临时 cache 不跨请求、不跨方法、不进入共享 Wan cache。
5. 记录 `source_prepare_s`、target E2E、denoising/DiT 时间和清理状态。若延续旧主表口径，source preparation 不计入 target E2E，但必须单独报告；不得把源生成成本隐藏在“预置缓存”中。
6. 旧 hit/miss 是 replay ground truth。新 CogVideo 执行失败单独记录为 error，不把 error 改写成 miss，也不因 CogVideo probe 改变旧 hit rate。

旧结果的决策摘要：NIRVANA-VID 242/58、Chorus 242/58、Chorus2 242/58、Carnator Boundary B5 300/0；TeaCache 为 target-only，hit rate 记 N/A。旧 Chorus 的 41 个 error（主要是 LLM prompt diff 或 Grounded-SAM-2 预处理失败）不是 miss；主实验仍按原 hit 决策运行 source 和 target，若新实现再次失败则记录新 error 并单独统计，不能因为历史 error 直接跳过或改写为 miss。若 CogVideo 所需外部依赖不可用，应将 Chorus 标为 blocked/N/A，而不是静默删掉这 41 条请求。

## 基线适配要求

所有方法先对齐旧实验的总步数、reuse step、seed、输入顺序、计时字段和错误语义；CogVideo 使用其支持的 81 帧、分辨率（计划值 `1360x768`）、VAE、scheduler 和 latent/token shape。不得把 Wan 的步数或 tensor 形状直接复制为 CogVideo 实现；每个 cache payload 都写入模型、dtype、尺寸、帧数、steps、scheduler 和 config fingerprint。

| 基线 | CogVideoX1.5 适配 |
| --- | --- |
| NIRVANA-VID | 在 CogVideo 的原 50-step scheduler 中按旧协议捕获 source latent；target 从同一 scheduler/timestep 状态继续剩余步骤。不能用 `50-reuse_step` 新建短 scheduler 冒充 restart；旧 300-request 配置的 `reuse_step=5` 必须显式记录。 |
| TeaCache | 保持 target-only、50-step 设置，不读 source cache。针对 `CogVideoXTransformer3DModel` 校验 forward 返回值、CFG branch、step counter、跳过/执行步数和 CUDA 同步 DiT 计时，并重新校准 threshold 与 `max_skip_steps`。 |
| Chorus | 独立实现 CogVideo source trajectory、`K1=5/K2=30` 阶段、TGAA、prompt diff、mask 到 CogVideo latent/token grid 的投影及 scheduler 状态；不能调用 Wan Chorus adapter，也不能复用 Wan trajectory/mask。 |
| Chorus2 | 独立适配 CogVideo 的 42 层/48 头 joint text-video attention、rotary embedding、token grid 和 sparse mask/backend；不能直接使用 `chorus2/wan_ti2v_5b.py`、Wan outer mask 或 SpargeAttn native loop。 |
| Carnator Boundary B5 | 只有完成 CogVideo retriever、step-5 probe、Early Signature、mask projection、sparse attention 和量化 FFN hook 后才可使用 B5 名称。现有 source-latent-restart 路径只能标为 `carnator_source_latent_restart`，不能宣称等价于 Wan B5；本次旧 replay 仍固定 300 hit。 |

## 五个基线的算法流程

| 基线 | CogVideoX1.5 实际流程 |
| --- | --- |
| NIRVANA-VID | hit 时 source 固定 seed 跑到 completed step 5，保存 latent、scheduler 类型和完整 50-step timestep 序列；target 注入 step-5 latent，并在同一个 `CogVideoXDDIMScheduler` 的原始后缀上执行 45 步。miss 不推理。 |
| TeaCache | 不做跨请求检索；target 保持 50 个 scheduler step，用相邻 transformer 输入的 relative-L1 决定复用上一输出，`max_skip_steps=2` 强制定期实算，并记录 50 次调用中的实算/跳过 step。 |
| Chorus | source 完整跑 50 步并保存 completed step 5--30 trajectory；target 从 step 5 恢复。step 5--30 在 `Medit` query、`Msee` context 上执行 joint regional attention，TGAA 放大差异文本 token 的 K/输出，且每步在 `Medit` 外恢复 source latent；step 30 后恢复 dense block。 |
| Chorus2 | source 从 completed step 11 起，按 step、CFG branch、42 层捕获 RoPE 后 Q/K，生成 top-p bit-packed block mask 与 SHA256；target 仍跑完整 50 步，前 10 步 dense，之后用 source routing 驱动 `torch.compile` 后的 FlexAttention block-sparse kernel，每 8 层 refresh。 |
| Carnator Boundary B5 | source 保存 step-5 latent/tail、step 10--50 trajectory，以及外层 28 层的 source routing；target 固定 seed 跑 step-5 probe并审计 CIELAB Early Signature，再从 step 10 恢复。中间 14 层使用 `Medit/Msee` regional attention，外层 28 层使用 compiled FlexAttention，42 层 FFN 使用 Triton W8A8；每步执行 Boundary B5 latent blend。固定 replay 始终是最终 hit/miss ground truth。 |

## 最新状态（2026-09-08）

**CogVideoX1.5 前3请求测试完成**：
- 实验脚本已添加自动质量评估功能，使用VBench计算CLIP-T和VBench-Q
- 构建了replay文件，从旧Wan实验提取hit/miss决策
- NIRVANA-VID完成3请求测试：2个ok、1个cache_miss，平均CLIP-T 0.302、VBench-Q 0.632
- 其他四个基线待测试

**旧验证状态**：
- 五个方法的 `--check-only` 均为 `ready`；全仓单元测试为 `159 passed`（2026-09-11）
- Chorus、Chorus2、Carnator 通过256x256、9帧、2步dense smoke和小尺寸backend smoke
- forced miss/hit测试通过，缓存清理成功

## 文件结构

```text
configs/models/cogvideox15.yaml
configs/methods/matrix.yaml
adapters/cogvideox15.py
adapters/cogvideo_source_cache.py
methods/teacache/                    # CogVideo forward/step 校准
methods/chorus/cogvideo*.py          # trajectory/TGAA/mask
methods/chorus2/cogvideo.py          # joint sparse attention
carnator/cogvideo*.py                # B5 probe/signature/mask/FFN
scripts/benchmark/build_cogvideo_hit_replay.py
scripts/benchmark/run_vidprom_300_experiment.py  # --model cogvideox15
tests/test_cogvideo_advanced_backends.py
tests/test_cogvideox15_adapter.py
tests/test_cogvideo_hit_replay.py

artifacts/vidprom_1000cache_300input_noun_match/
  cogvideox15_<run_id>/
    replay/{nirvana_vid,teacache,chorus,chorus2,carnator_boundary_b5}.jsonl
    results/<method>/{requests.jsonl,request_records.jsonl,videos/}
    source_prepare/
    tmp_cache/<method>/<arrival_index>_<source_uuid>/
    run_manifest.json
```

`run_manifest.json` 保存旧实验目录和输入文件 hash、replay manifest hash、CogVideo 模型路径/hash、git revision、resolved config 及计时口径。Wan 目录和共享 cache 只读，CogVideo 结果写入新的 sibling 目录。

## 实现与验证顺序

1. 在 `cogvideo` 环境确认模型权重、tokenizer、VAE、text encoder、transformer 和 scheduler 完整；完成 `--check-only`、dry-run、单请求 dense smoke。
2. 实现 replay manifest 生成器，统一读取 `hit`、`arrival_index`、`request_index`、`source_uuid`，并校验每个方法恰好 300 条 terminal decision。
3. 将 runner 改为 model-aware，加入 source 临时目录、`try/finally` 清理、resume 不改变 replay 决策、source/target 分离计时和 fail-closed schema 校验。
4. 每个方法先做 forced miss/hit 两条 smoke，再做 5 条真实请求；检查 latent/token shape、scheduler step、seed determinism、GPU 隔离和清理后目录为空。
5. 五个方法各运行完整 300 条，最后按旧字段生成 summary/results table，并额外列出 source preparation 总时间、每 hit 平均时间和 error 数。

Chorus、Chorus2 和 Carnator B5 的独立 CogVideo backend 已通过上述 pre-300 验证；不得用 Wan 实现、source-latent-restart 兼容版或伪造 hit 结果替代它们。
