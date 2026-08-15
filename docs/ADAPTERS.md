# Adapters

所有 adapter 都提供 `prepare()`、`generate()`、latent 序列化和视频写出。配置只允许 T2V 参数：prompt、seed、CFG、时间步、latent shape 和输出路径。

- `Wan22Adapter`：CHAI 复用 `CHAI/chai/adapters/wan_t2v.py`；TeaCache 在同一采样循环复用 transformer 输出；NIRVANA-VID 在第 10 步缓存/恢复原生扩散 latent。CHAI-Carnator 不走此 adapter，而由 `scripts/run/chai_carnator_matrix.py` 直接接管 Wan native 8-step loop：block 0 使用 CHAI Cache Attention，其余 block 使用 Carnator hook。
- `CogVideoX15Adapter`：CHAI 复用 `CHAI/chai/adapters/cogvideo_t2v.py`；TeaCache 在同一采样循环复用 transformer 输出；NIRVANA-VID 通过 Diffusers callback 缓存 latent 后重启短调度。
- `SyntheticAdapter`：无依赖 CPU 协议测试，不代表模型质量或性能。

模型源码和权重不复制到本仓库；通过 `configs/models/*.yaml` 的路径或外部安装提供。
