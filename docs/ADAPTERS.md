# Adapters

所有 adapter 都提供 `prepare()`、`generate()`、latent 序列化和视频写出。配置只允许 T2V 参数：prompt、seed、CFG、时间步、latent shape 和输出路径。

- `Wan22Adapter`：CHAI Cache Attention 实现在 `methods/chai/adapters/wan_t2v.py`；TeaCache 在同一采样循环复用 transformer 输出；NIRVANA-VID 在第 10 步缓存/恢复原生扩散 latent。
- `CogVideoX15Adapter`：使用 CogVideoX1.5 原生 81 帧、50-step scheduler；TeaCache 在同一采样循环复用 transformer 输出，NIRVANA-VID 保存 step-5 latent 并在原 scheduler 后缀执行 45 步。
- `ChorusCogVideoBackend`：保存 step 5--30 trajectory，target 结合 regional attention、TGAA 和 latent fusion 后恢复 dense。
- `Chorus2CogVideoBackend`：保存 RoPE 后 Q/K routing，target 前 10 步 dense，之后执行编译后的 FlexAttention block-sparse kernel。
- `CarnatorCogVideoBackend`：执行 step-5 probe/Early Signature、step-10 resume、regional/outer sparse attention、Triton W8A8 FFN 和 Boundary B5 blend。
- CogVideo backend 只读取当前请求的临时 CogVideo cache；旧 Wan cache 不可加载，固定 replay 决策也不会被运行时 probe 改写。
- `SyntheticAdapter`：无依赖 CPU 协议测试，不代表模型质量或性能。

模型源码只从 `external/` 的锁定子模块加载；权重不提交，通过 `configs/models/*.yaml` 的路径提供。
