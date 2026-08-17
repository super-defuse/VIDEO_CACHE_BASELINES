# Methods

常规插件只实现 `BaselineMethod.generate()`，模型调用由 adapter 完成；Wan 原生矩阵
runner（Carnator、CHAI-Carnator）直接接管 native sampling loop，以便同时安装 attention、
latent 和 FFN hook。

| 插件 | capability | 来源 |
| --- | --- | --- |
| `baseline` | dense fallback | 公共协议 |
| `chai` | 跨请求缓存、跳步、Cache Attention | `methods/chai/` |
| `teacache` | adapter 内跳步 | `adapters/teacache.py` |
| `nirvana_vid` | 跨请求扩散 latent、跳步、source trajectory | `adapters/{wan22,cogvideox15}.py` |
| `carnator` | Wan C0/B0--B5 矩阵、稀疏 attention、量化、source trajectory；CogVideoX1.5 使用 source latent restart 适配 | `carnator/` + `scripts/run/carnator_matrix.py` |
| `chai_carnator` | Wan 专用 `CHAI-Carnator`：8-step source/target；block 0 在 CHAI local steps 1--4 读取对应 source token，其余 block 按 CC-B0--CC-B5 使用 Carnator sparse attention、latent injection 和可选 W8A8 FFN | `scripts/run/chai_carnator_matrix.py` + `carnator/` |
| `chorus` | Wan2.2 50-step source trajectory；`K1=10/K2=40` 三阶段 per-step cache、Stage-2 SRD/TGAA/latent fusion、Stage-3 dense no-reuse、LLM prompt diff、Grounded-SAM-2 v1.0 mask；CogVideoX1.5 为 N/A | `methods/chorus` + `adapters/wan22.py` |

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

`CHAI-Carnator` 是独立的 Wan native matrix runner，不复用已清理的旧组合基线：
source 与 target 都从同一 8-step scheduler 的初始 latent 开始。source 保存 initial
latent 及每个 post-scheduler step 1--8 的 latent；同时在 block 0 保存 CHAI 约定的
local steps 1--4 条件分支 transformer 输入。source completed step 5 的 cross/self
attention 生成一份融合 `Mbase/Medit/Msee`，并固定供 target 全部八步使用；不会再复制为多个
去噪步的 region-mask key。

target 的 block 0 仅在 local steps 1--4 使用对应的 Cache Attention；其他四步为 dense
attention。block 1--9、21--29 按 B1/B2/B4/B5 启用 outer sparse backend（B0/B3 保持
dense），block 10--20 始终执行 Carnator region update，B3/B4/B5 另启用 W8A8 FFN。
每个 target step 都以固定的 source completed-step-5 `Medit` 执行 latent injection；
matrix manifest 审计 8 次注入、source completed step 5、cache steps、outer backend 和 FFN backend。
