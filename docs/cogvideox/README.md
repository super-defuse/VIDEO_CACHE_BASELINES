# CogVideoX1.5 基线适配

CogVideoX1.5-5B 上五个缓存加速基线（baseline / nirvana_vid / chorus / chorus2 /
carnator_boundary_b5）的适配记录。**该适配尚未完成 300 请求正式实验，且已知存在若干问题**，
详见 [KNOWN_ISSUES.md](KNOWN_ISSUES.md)。如需重构，本文档集是当前状态的完整入口。

| 文档 | 内容 |
| --- | --- |
| [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) | 300 请求实验口径、基线适配要求、验证顺序（权威协议） |
| [WORKFLOW.md](WORKFLOW.md) | 三阶段流程、各基线逐步执行逻辑、产物结构与 results.jsonl schema |
| [KNOWN_ISSUES.md](KNOWN_ISSUES.md) | 已验证/未验证边界、配置键 bug、质量评测缺口 |
| [FUSION_MASK.md](FUSION_MASK.md) | Carnator 从外部分割重构为纯 fusion mask 的设计与进展 |

## 与 Wan 版本的架构差异

CogVideoX 不是 Wan 的简单移植，以下差异决定了 latent / trajectory / mask 全部无法复用：

1. **稀疏注意力后端：SpargeAttention → FlexAttention** ⭐ 最显著的改动
   - Wan：`external/spargeattn` + FlashInfer BSR 格式，需要本机 CUDA 编译
   - CogVideoX：PyTorch 原生 `torch.nn.attention.flex_attention` + `torch.compile`，
     不依赖外部 CUDA 编译（[adapters/cogvideo_sparse.py](../../adapters/cogvideo_sparse.py)）
2. **Transformer 结构**
   - Wan：30 层纯视频 transformer
   - CogVideoX：42 层 joint text-video transformer + RoPE，需处理 text/video token 拼接与索引选择
     （[adapters/cogvideo_attention.py](../../adapters/cogvideo_attention.py)）
3. **Regional Attention（Chorus / Carnator）**：Wan 直接在视频 token 上操作，CogVideoX 需联合处理
   text token 与 video token
4. **量化 FFN 分离**：Wan 用 `carnator/quantized_ffn.py`，CogVideoX 用独立的
   `carnator/cogvideo_quantized_ffn.py`（FFN 结构与归一化方式不同）
5. **Scheduler 与 latent 格式完全不兼容**：Wan 用自定义 scheduler，CogVideoX 用
   `CogVideoXDDIMScheduler`，5D tensor `[1,T,C,H,W]`。两者中间状态无法互通
6. **步数与阶段配置**：两者同为 50 步；Chorus 的 `K1=5/K2=30` 一致；Chorus2 Wan 从 step 10
   开始稀疏、CogVideoX 从 step 11；Carnator 的 `probe_step=5 / reuse_step=10` 一致

> 因上述差异，所有 trajectory 与 mask 都必须在 CogVideoX 上重新生成，不能沿用 Wan 产物。

## 代码位置

```text
configs/models/cogvideox15.yaml     # 模型 + 各基线算法常量
adapters/cogvideox15.py             # 模型适配器（薄分发层）
adapters/cogvideo_pipeline.py       # 推理管线代理
adapters/cogvideo_attention.py      # joint attention 与 regional attention
adapters/cogvideo_sparse.py         # FlexAttention 稀疏后端
adapters/cogvideo_source_cache.py   # source cache manifest 与校验
core/cogvideo_runner.py             # replay 运行器
methods/teacache/method.py          # CogVideo forward/step 校准
methods/chorus/cogvideo.py          # trajectory / TGAA
methods/chorus/cogvideo_mask.py     # mask 到 latent/token grid 的投影
methods/chorus2/cogvideo.py         # joint sparse attention
methods/carnator_boundary_b5/       # B5 方法入口
carnator/cogvideox15.py             # B5 probe / signature / mask / FFN 驱动
carnator/cogvideo_attention_probe.py
carnator/cogvideo_quantized_ffn.py
scripts/benchmark/build_cogvideo_hit_replay.py
scripts/benchmark/run_vidprom_300_experiment.py   # --model cogvideox15
tests/test_cogvideo_advanced_backends.py
tests/test_cogvideox15_adapter.py
tests/test_cogvideo_hit_replay.py
```
