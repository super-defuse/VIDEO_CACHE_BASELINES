# Adapters

所有 adapter 都提供 `prepare()`、`generate()`、latent 序列化和视频写出。配置只允许 T2V 参数：prompt、seed、CFG、时间步、latent shape 和输出路径。

- `Wan22Adapter`：CHAI Cache Attention 实现在 `methods/chai/adapters/wan_t2v.py`；TeaCache 在同一采样循环复用 transformer 输出；NIRVANA-VID 在第 10 步缓存/恢复原生扩散 latent。
- `Wan21Adapter`：Wan2.1-T2V-1.3B 上的五个基线，全部经 `core/wan21_runner.py` 回放固定的命中决策；Chorus 的 SRD/TGAA、Chorus2 的 block-sparse、Carnator 的 regional/outer sparse 与 B5 blend 分别实现在 `adapters/wan21.py` 与 `carnator/`。
- `CogVideoX15Adapter`：使用 CogVideoX1.5 原生 81 帧、50-step scheduler；TeaCache 在同一采样循环复用 transformer 输出，NIRVANA-VID 保存 step-5 latent 并在原 scheduler 后缀执行 45 步。
- `LtxVideoAdapter`：LTX-Video 0.9.7-dev（13B）上的同五个基线，驱动 `LtxVideoPipelineDelegate` 的单一采样循环而非 pipeline `__call__`——0.9.7-dev 的发布配方是 CFG + STG，而 diffusers 的 `LTXConditionPipeline` 只实现 CFG。除模型边界外算法与 Wan2.1 相同，改写见 [LTXVIDEO.md](LTXVIDEO.md)。
- CogVideoX 的各基线入口：Chorus 在 `methods/chorus/`（step 5--30 trajectory、regional attention、TGAA、latent fusion 后恢复 dense），Chorus2 在 `methods/chorus2/cogvideo.py` + `adapters/cogvideo_sparse.py`，Carnator 在 `carnator/cogvideox15.py`（step-5 probe/Early Signature、step-10 resume、regional/outer sparse、Triton W8A8 FFN、Boundary B5 blend）。
- CogVideo backend 只读取当前请求的临时 CogVideo cache；旧 Wan cache 不可加载，固定 replay 决策也不会被运行时 probe 改写。
- `SyntheticAdapter`：无依赖 CPU 协议测试，不代表模型质量或性能。

每个模型的 adapter 按同一分层拆分：`<model>.py`（五个基线与 replay 接线）、`<model>_pipeline.py`（采样循环）、`<model>_attention.py`（QKV/RoPE 等模型算术）、`<model>_sparse.py`（block mask 缓存与稀疏执行）。Wan2.1 的对应实现直接复用 Wan2.2 的模块，不另设一套。

稀疏 attention 的后端在模型间**有意不同**：CogVideoX1.5 的 Chorus2/Carnator 走 `CogVideoFlexExecutor`（`torch.compile(flex_attention)`，128/128），LTX-Video 走 `LtxSparseExecutor`（`spas_sage_attn` 的 `block_sparse_sage2_attn_cuda`，Q=64/K=128），与 Wan2.1 的基线一致。两者不可互换：flex 的 `BLOCK_M=BLOCK_N=128` 是硬编码的，跑不了 SpargeAttn 要求的 64/128 几何。

模型源码只从 `external/` 的锁定子模块加载；权重不提交，通过 `configs/models/*.yaml` 的路径提供。
