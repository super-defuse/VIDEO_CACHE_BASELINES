# Methods

常规插件只实现 `BaselineMethod.generate()`，模型调用由 adapter 完成；Wan 原生矩阵
runner（Carnator）直接接管 native sampling loop，以便同时安装 attention、
latent 和 FFN hook。

| 插件 | capability | 来源 |
| --- | --- | --- |
| `baseline` | dense fallback | 公共协议 |
| `chai` | 跨请求缓存、跳步、Cache Attention | `methods/chai/` |
| `teacache` | adapter 内跳步 | `adapters/teacache.py` |
| `nirvana_vid` | 跨请求扩散 latent、跳步、source trajectory | `adapters/{wan22,cogvideox15}.py` |
| `carnator` | Wan C0/B0--B5 矩阵、稀疏 attention、量化、source trajectory；默认将 10--20 的原始 block mask 与 `Medit`/`Msee` 相交；可一次导出所有共享缓存 | `carnator/` + `scripts/run/carnator_matrix.py` |
| `chorus2` | Wan 专用：从共享 source 的第 10 步 latent 恢复，只计算后 40 步；全部 30 个 block 使用普通 SpargeAttn mask，每 8 层 refresh | `chorus2/` + `scripts/run/chorus2_wan.py` |
| `chorus` | Wan2.2 50-step source trajectory；`K1=10/K2=40` 三阶段 per-step cache、Stage-2 SRD/TGAA/latent fusion、Stage-3 dense no-reuse、LLM prompt diff、Grounded-SAM-2 v1.0 mask；CogVideoX1.5 为 N/A | `methods/chorus` + `adapters/wan22.py` |
| `chorus_no_tgaa` | Wan2.2 Chorus 消融；保留同一 source trajectory、SRD、latent fusion、Stage-3 dense/no-reuse 和 mask，仅关闭 Stage-2 TGAA；CogVideoX1.5 为 N/A | `methods/chorus_no_tgaa` + `adapters/wan22.py` |
| `chorus_tgaa_joint_gamma_1_6` | Wan2.2 Chorus TGAA 工程探针；同步设置 `gamma_k=gamma_o=1.30`，其余 Chorus 配置不变；CogVideoX1.5 为 N/A | `methods/chorus_tgaa_joint_gamma_1_6` + `adapters/wan22.py` |
| `chorus_tgaa_joint_gamma_2_1` | Wan2.2 Chorus TGAA 工程探针；同步设置 `gamma_k=gamma_o=1.6`，其余 Chorus 配置不变；CogVideoX1.5 为 N/A | `methods/chorus_tgaa_joint_gamma_2_1` + `adapters/wan22.py` |
| `chorus_tgaa_joint_gamma_2_0` | Wan2.2 Chorus TGAA 工程探针；同步设置 `gamma_k=gamma_o=2.0`，其余 Chorus 配置不变；CogVideoX1.5 为 N/A | `methods/chorus_tgaa_joint_gamma_2_0` + `adapters/wan22.py` |

Chorus 当前真实产物的 mask 后端为官方 Grounded-SAM-2 v1.0：Grounding DINO 在 source
首帧生成框，SAM 2.1 Hiera-Large 传播到 121 帧，再 max-pool 到 Wan latent/token 网格；缺失
模型或权重时直接报错，不切换到几何 mask。

Chorus 的 TGAA hook 只在 Stage 2 激活，并覆盖 Wan 的全部 transformer blocks。Stage 2 的 SRD
保留 native Wan head 的 dense 输出；Stage 3 起点清空 UniPC derivative history、保留全局 timestep
index，再执行 dense/no-reuse。论文没有公开
`F_k(t,m)`/`F_o(t,m)` 的具体函数，因此当前使用 step-only linear decay approximation，
运行元数据会显式标注该近似；差异 token 无法映射到 target tokenizer 时直接失败，不使用
任意 token-prefix fallback。Stage 3 关闭 TGAA、SRD 和 source latent fusion，执行完整 dense
计算以修复区域复用的 discontinuity。

CHAI 的 8-step express schedule 对齐 OpenSora：Cache Attention 在 block 0 的
历史请求 source 的 1、2、3、4 步 token 逐步保存，target fast steps 1、2、3、4
逐步读取对应缓存。TeaCache、NIRVANA-VID 和 Carnator 的 CogVideoX1.5 适配代码仍保留，
但本轮不生成 CogVideoX1.5 验收产物。Carnator 保留其原生 Wan C0/B0--B5 矩阵入口。
原生 source 完整运行 50 步并保存 latent trajectory，但只在 completed step 5 生成一次
cross/self attention canonical mask；所有 reuse step 共用 `Mbase/Medit/Msee` 及 latent
mask，不再保留 per-step region-mask 副本。
CogVideoX1.5 使用明确标注的 source-latent-restart adapter，不将 Wan 专用的
mask/FFN 路径标成 CogVideoX1.5 结果。synthetic adapter 只用于协议测试。

`python -m carnator.adapters.wan_ti2v_5b --prepare-shared-dense-cache` 是 Wan 的统一
source cache 入口。一次 source Dense 导出 Carnator 原始 trajectory/fusion mask/block masks、
NIRVANA-VID 的 step-10 latent、CHAI 的 block-0 source tokens、Chorus 的
`chorus-trajectory-v2` sidecar，以及 Chorus2 的完整 block-mask cache。Chorus2 的 10--20
层会额外保存未筛选的普通 sparse mask；Carnator 的更稀疏 `Medit`/`Msee` 交集只在 Carnator
target 运行时构造。缓存准备时延与 target C0 的端到端时延分开记录，后者是所有 speedup 的
唯一分母。
