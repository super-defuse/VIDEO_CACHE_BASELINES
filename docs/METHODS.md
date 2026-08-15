# Methods

插件只实现 `BaselineMethod.generate()`，模型调用由 adapter 完成。

| 插件 | capability | 来源 |
| --- | --- | --- |
| `baseline` | dense fallback | 公共协议 |
| `chai` | 跨请求缓存、跳步、Cache Attention | `../CHAI/chai` |
| `teacache` | adapter 内跳步 | `../CHAI/teacache` |
| `nirvana_vid` | 跨请求扩散 latent、跳步、source trajectory | `../CHAI/nirvana-strawman` |
| `carnator` | Wan C0/B0--B5 矩阵、稀疏 attention、量化、source trajectory；CogVideoX1.5 使用 source latent restart 适配 | `../CARNATOR_ICLR2027/carnator` |
| `carnator_new_baseline` | Wan 专用新基线：复用 Carnator 第 10 步 source latent，执行索引 10--17 的 8 步；block 0 从 outer sparse layers 移除，在 target 全局步骤 11--14（local steps 1--4）读取历史 source 全局步骤 11--14 的 Cache Attention，其余层保留 B0/B1/B2/B3/B4/B5 后端，B3/B4/B5 保留 W8A8 FFN | `../CARNATOR_ICLR2027/carnator/adapters/wan_ti2v_5b_new_baseline.py` + `../CARNATOR_ICLR2027/scripts/run_t2v_new_baseline_matrix.py` |

CHAI 的 8-step express schedule 对齐 OpenSora：Cache Attention 在 block 0 的
历史请求 source 的 1、2、3、4 步 token 逐步保存，target fast steps 1、2、3、4
逐步读取对应缓存。TeaCache、NIRVANA-VID 和 Carnator 的 CogVideoX1.5 适配代码仍保留，
但本轮不生成 CogVideoX1.5 验收产物。Carnator 保留其原生 Wan C0/B0--B5 矩阵入口；
CogVideoX1.5 使用明确标注的 source-latent-restart adapter，不将 Wan 专用的
mask/FFN 路径标成 CogVideoX1.5 结果。synthetic adapter 只用于协议测试。

`carnator_new_baseline` 不是将 8-step scheduler 从头开始采样：它从 source 的
post-step-10 latent 恢复，保持原 50-step scheduler 的时间轴，仅计算索引 10--17。
因此 Cache Attention 的 `1/2/3/4` 指的是这段八步 continuation 的本地步号，分别
映射到 target 全局步骤 11/12/13/14；对应的历史 source block-0 token 也来自
全局步骤 11/12/13/14，而不是单个 capture step。

`carnator_new_baseline` 使用同一 continuation 协议，但对 B1/B2/B4/B5 将
outer sparse layers 从 `0-9,21-29` 改为 `1-9,21-29`，确保 block 0 不再被稀疏
attention hook 接管；block 0 在 Cache Attention 未激活的四步仍走 dense attention。
