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

所有方法先对齐旧实验的总步数、reuse step、seed、输入顺序、计时字段和错误语义；CogVideo 使用其支持的 81 帧、分辨率 `1360x768`、VAE、scheduler 和 latent/token shape。不得把 Wan 的步数或 tensor 形状直接复制为 CogVideo 实现；每个 cache payload 都写入模型、dtype、尺寸、帧数、steps、scheduler 和 config fingerprint。

| 基线 | CogVideoX1.5 适配 |
| --- | --- |
| Baseline (Dense) | 不读 replay，不需要 cache；每条请求跑满 50 步。作为所有加速比的分母。 |
| NIRVANA-VID | 在 CogVideo 的原 50-step scheduler 中按旧协议捕获 source latent；target 从同一 scheduler/timestep 状态继续剩余步骤。不能用 `50-reuse_step` 新建短 scheduler 冒充 restart；旧 300-request 配置的 `reuse_step=5` 必须显式记录。 |
| Chorus | 独立实现 CogVideo source trajectory、`K1=5/K2=30` 阶段、TGAA、prompt diff、mask 到 CogVideo latent/token grid 的投影及 scheduler 状态；不能调用 Wan Chorus adapter，也不能复用 Wan trajectory/mask。 |
| Chorus2 | 独立适配 CogVideo 的 42 层/48 头 joint text-video attention、rotary embedding、token grid 和 sparse mask/backend；不能直接使用 `chorus2/wan_ti2v_5b.py`、Wan outer mask 或 SpargeAttn native loop。 |
| Carnator Boundary B5 | step-5 probe、Early Signature、per-noun fusion mask、region + outer sparse attention 与 w8a8 量化 FFN 均已在 CogVideo 上独立实现，与 Wan/LTX 共用 `carnator/masks.py`、`carnator/early_signature.py` 的算法本体。不得用 Wan 实现或 source-latent-restart 兼容版替代。 |
| TeaCache（不在本轮五个基线内） | 方法插件存在且可运行（`--method teacache`），但不属于本实验的五个基线：它没有跨请求 source，旧 replay 里也没有它的决策文件。需要时可单独用它做 target-only 对照。 |

## 五个基线的算法流程

逐步逻辑与伪代码见 [WORKFLOW.md](WORKFLOW.md)，此处只给一句话摘要。

| 基线 | CogVideoX1.5 实际流程 |
| --- | --- |
| Baseline (Dense) | 无检索、无缓存，50 步 dense。 |
| NIRVANA-VID | hit 时 source 中断在 completed step 5，保存 latent、scheduler 类型和完整 50-step timestep 序列；target 注入 step-5 latent，并在同一个 `CogVideoXDDIMScheduler` 的原始后缀上执行 45 步。miss 不推理。 |
| Chorus | source 完整跑 50 步并保存 completed step 5--30 trajectory；target 从 step 5 恢复。step 5--30 在 `Medit` query、`Msee` context 上对全部 42 个 joint block 执行 regional attention，TGAA 放大差异文本 token 的 K/输出，且每步在 `Medit` 外恢复 source latent；step 30 后恢复 dense。 |
| Chorus2 | source 从 completed step 11 起，按 step、CFG branch、42 层捕获 RoPE 后 Q/K，生成 top-p bit-packed block mask 与 SHA256；target 仍跑完整 50 步，前 10 步 dense，之后用 source routing 驱动 `torch.compile` 后的 FlexAttention block-sparse kernel，每 8 层 refresh。 |
| Carnator Boundary B5 | source 保存 step-5 latent/tail、step 10--50 trajectory、外层 28 层（0-13、28-41）的 source routing，以及 step-5 joint probe 产出的每名词 Mbase 与 CIELAB 颜色签名；target 固定 seed 跑 step-5 probe（与 source 同一套双 hook，签名对比仅审计、不改写 replay 决策），再从 step 10 恢复。中间 14 层（14-27）使用 `Medit/Msee` regional attention，外层 28 层使用 compiled FlexAttention，42 层 FFN 使用 Triton W8A8；每步执行 Boundary B5 latent blend。固定 replay 始终是最终 hit/miss ground truth。 |

## 最新状态（2026-09-12）

**代码、离线验证与 10 请求测试均已完成；300 请求正式实验待启动**：

- 五个基线的实现已与 wan2.1 / ltx-video 对齐；`core/cogvideo_runner.py` 缩到 37 行，直接继承
  `Wan21ReplayRunner`，只保留 CogVideo 特有的 cache manifest 校验。
- Carnator fusion mask 重写为与 Wan/LTX 同一条链路（见 [FUSION_MASK.md](FUSION_MASK.md)），
  四处致命缺陷修复记录见 [KNOWN_ISSUES.md](KNOWN_ISSUES.md) §3.5；10 请求测试里它已在
  **真实权重**上 10/10 命中跑通，不再只有 mock 覆盖。
- 两个实验脚本已就绪：`run_cogvideox_all_test.sh`（10 请求）与
  `run_cogvideox_all_formal.sh`（300 请求），都带状态分布校验。
- 全仓单元测试 **234 passed**（2026-09-12）；五个方法 `--check-only` 均为 `ready`。
- **10 请求测试（2026-09-12）**：50 条请求 0 error，状态分布与 replay 逐条吻合，
  `verify_replay_status_distribution.py` rc=0，聚合器 rc=0。实测耗时与加速比见
  [KNOWN_ISSUES.md](KNOWN_ISSUES.md) §5——**注意 1360×768 下单次请求实测 ~761s，比旧
  smoke 数据慢一个量级，§5 的成本预算要按此重估**。

**历史验证状态**：
- 3 条真实请求跑通（`nirvana_vid`：2 命中 1 miss），平均 CLIP-T 0.302、VBench-Q 0.632
- Chorus、Chorus2、Carnator 通过 256x256、9 帧、2 步 dense smoke 与小尺寸 backend smoke
- forced miss/hit 测试通过，缓存清理成功

## 文件结构

```text
configs/models/cogvideox15.yaml
configs/methods/default.yaml          # chorus 的 K1/K2、Grounded-SAM2 等跨模型默认值
configs/methods/matrix.yaml
adapters/cogvideox15.py
adapters/cogvideo_attention.py        # joint_qkv / caption_token_indices / cogvideo_token_grid
adapters/cogvideo_source_cache.py
core/cogvideo_runner.py               # = Wan21ReplayRunner + cache manifest 校验
methods/teacache/method.py            # CogVideo forward/step 校准（不在五个基线内）
methods/chorus/cogvideo*.py           # trajectory/TGAA/mask
methods/chorus2/cogvideo.py           # joint sparse attention
carnator/cogvideox15.py               # B5 probe / signature / mask / FFN 驱动
carnator/cogvideo_attention_probe.py  # step-5 joint probe（cross + self 一次拿到）
carnator/cogvideo_quantized_ffn.py
scripts/benchmark/run_cogvideox_all_test.sh       # 10 请求测试
scripts/benchmark/run_cogvideox_all_formal.sh     # 300 请求正式实验
scripts/benchmark/verify_replay_status_distribution.py
scripts/benchmark/run_vidprom_300_experiment.py   # --model cogvideox15
scripts/benchmark/aggregate_vidprom_300_results.py # --model cogvideox15
scripts/benchmark/build_cogvideo_hit_replay.py
tests/test_cogvideo_advanced_backends.py
tests/test_cogvideox15_adapter.py
tests/test_cogvideo_hit_replay.py

artifacts/vidprom_1000cache_300input_noun_match/
  cogvideox15_replay/{nirvana_vid,teacache,chorus,chorus2,carnator_boundary_b5}.jsonl  # 只读
  results_cogvideox15_test/<method>/    # 10 请求
  results_cogvideox15/<method>/         # 300 请求
    {requests.jsonl,request_records.jsonl,results.jsonl,videos/}
    source_prepare/events.jsonl
    tmp_cache/<request_index>_<source_uuid>/
    run_manifest.json
```

`run_manifest.json` 保存旧实验目录和输入文件 hash、replay manifest hash、CogVideo 模型路径/hash、git revision、resolved config 及计时口径。Wan 目录和共享 cache 只读，CogVideo 结果写入新的 sibling 目录。

## 实现与验证顺序

1. ~~在 `cogvideo` 环境确认模型权重、tokenizer、VAE、text encoder、transformer 和 scheduler 完整；完成 `--check-only`、dry-run、单请求 dense smoke。~~ ✅
2. ~~实现 replay manifest 生成器，统一读取 `hit`、`arrival_index`、`request_index`、`source_uuid`，并校验每个方法恰好 300 条 terminal decision。~~ ✅ `build_cogvideo_hit_replay.py`
3. ~~将 runner 改为 model-aware，加入 source 临时目录、`try/finally` 清理、resume 不改变 replay 决策、source/target 分离计时和 fail-closed schema 校验。~~ ✅ `core/cogvideo_runner.py` 继承 `Wan21ReplayRunner`
4. ~~10 请求测试~~ ✅ **2026-09-12 通过**：`verify_replay_status_distribution.py` 判定
   baseline 10 ok、nirvana/chorus/chorus2 各 8 ok + 2 cache_miss、carnator 10 ok，
   全部 0 error；清理后 `tmp_cache` 为空。实测数据见
   [KNOWN_ISSUES.md](KNOWN_ISSUES.md) §5。
5. **300 请求正式实验**：`bash scripts/benchmark/run_cogvideox_all_formal.sh`，最后按旧字段生成
   summary/results table，并额外列出 source preparation 总时间、每 hit 平均时间和 error 数。

Chorus、Chorus2 和 Carnator B5 的独立 CogVideo backend 已通过上述 pre-300 验证；不得用 Wan 实现、source-latent-restart 兼容版或伪造 hit 结果替代它们。
