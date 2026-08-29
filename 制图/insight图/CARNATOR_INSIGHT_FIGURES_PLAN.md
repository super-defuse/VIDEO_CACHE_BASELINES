# Carnator Insight 图计划

只生成独立子图，不在脚本中拼论文大图。所有样例固定 Wan2.2-TI2V-5B、50-step、同一 CFG、seed、分辨率和帧数。每次导出同时写 `manifest.json`，记录 prompt、seed、completed step、layer、mask 参数和输入 artifact hash。

## 图 1：相似请求的 attention routing 相似，可复用 sparse mask

**要论证的事。** 两个语义接近、仅主体不同的请求，在相同推理位置的 self-attention block mass 和由其生成的 sparse mask 上具有一致结构；因此 source sparse mask 可以作为 target 的路由先验。

**固定样例。**

- source：`a cat is sleeping on a grassland, focus on the cat`
- target：`a dog is sleeping on a grassland, focus on the dog`
- 两个请求各自独立 dense 生成，使用相同初始 seed；捕获 conditional branch、completed step `20`、layer `15`。现有视频/mask 根目录为 `artifacts/static-prompt-defaults-seed20260218/{cat,dog}-sleeping-grassland/`。

**需要的导出。** 在 `SourceBlockMaskCache.capture` 中旁路保存产生 top-p mask 的真实 `block_mass[head,q_block,k_block]`；现有 `outer_block_masks` 保存真实 `bool[head,q_block,k_block]`。不保存 token-level dense attention，也不把 source mask 画到 target 上。

```text
fig1/
  source_frame.png                         # source 视频 50% 时间位置截图
  target_frame.png                         # target 视频同一时间位置截图
  source_block_mass_and_sparse_mask.png    # 1 x 2: block mass | sparse mask
  target_block_mass_and_sparse_mask.png    # 1 x 2: block mass | sparse mask
  routing_metrics.csv
```

- 两张 `1 x 2` 图都使用相同的 query/key block 轴和色标：左图为跨 head 平均的连续 block mass 热图，右图为跨 head consensus sparse mask（固定阈值，例如 `>=0.5`）。
- `routing_metrics.csv` 额外记录全 head 的 mask IoU、block-mass Spearman、以及 source mask 在 target block mass 上的 retained mass。用 100 个 noun-compatible pair 与 100 个随机 noun-incompatible control pair 做统计；子图只选指标接近 compatible 组中位数的样例，不挑最优例。

## 图 2：相似请求在早期推理阶段已呈现相似特征

**要论证的事。** 对三对请求均独立运行到 completed step `5`。预期 Similar pair 的 `delta latent`、能量轨迹和 CFG residual 比两对 negative 更接近，从早期中间状态即可看出请求是否相似。

**候选中间特征和子图形式。**

| 特征 | 为什么值得看 | 生成的独立子图 |
| --- | --- | --- |
| `delta latent`：`z_5-z_0` 的 token-wise cosine | 去掉相同 seed 的初始噪声后，直接比较早期去噪方向；这是最直接的 latent 相似性证据。 | `latent_affinity_<pair>.png`：在 `10%/50%/90%` 三个 token 时间切片绘制 source/target 的同位置 latent affinity；统一发散色标，另存全局均值。 |
| `delta latent` 能量轨迹：每帧 `mean(||z_5-z_0||)` | 反映各帧何时、何处开始形成结构，适合检验运动/时序布局是否一致。 | `latent_energy_<pair>.png`：一条紧凑的 frame-index 曲线，source 与 target 同图；可在曲线下方附一条绝对差的细带。 |
| CFG semantic residual：`eps_cond-eps_uncond` 的 token norm / cosine | 表示 prompt 在第 5 步实际施加的语义引导，和 raw latent 互补。 | `cfg_residual_<pair>.png`：三个时间切片的空间热图，或 source/target/absolute-difference 三联图；只保留最易读的一种。 |

三种特征使用相同 completed step、token grid 和色标；每个子图标题注明该 pair 的全局相似度/距离，便于在 PPT 中直接比较 Similar、Hard negative 与 Easy negative。

**首轮固定三对请求。** 三对均以相同初始 seed 独立 dense 生成；两对负例分别检验“文本/主体相近但画面不同”和“完全不同画面”。

| Pair | Source prompt | Target prompt | 预期 |
| --- | --- | --- | --- |
| Similar | `a cat is sleeping on a grassland, focus on the cat` | `a dog is sleeping on a grassland, focus on the dog` | 相似：构图、动作、草地背景接近，仅主体替换。 |
| Hard negative | `a dog is sleeping on a grassland, focus on the dog` | `a dog is running through a snowy forest, focus on the dog` | 不相似：保留主体词，但动作、背景、颜色和时序均变化。 |
| Easy negative | `a red sports car driving through a rainy city at night` | `a close-up of a yellow sunflower swaying in a sunlit field` | 不相似：主体、构图、光照和运动模式均不同。 |

```text
fig2/
  pair_<id>_latent_affinity.png
  pair_<id>_latent_energy.png
  pair_<id>_cfg_residual.png
  pair_feature_summary.csv                  # 三对请求的三个 step-5 特征相似度/距离
```
