# Carnator Fusion Mask 在 CogVideoX 上的实现

CogVideoX 版 Carnator 曾用外部分割模型（Grounded-SAM2）生成 mask，与 Wan2.2 的纯 fusion mask
算法不一致；该路径已删除。本文档记录现在这条与 Wan/LTX 逐行同构的 fusion mask 实现。
历史见 [KNOWN_ISSUES.md](KNOWN_ISSUES.md) §3.2 / §3.5。

## 可行性结论

CogVideoX 的 **Joint Attention** 架构完全支持纯 fusion mask：attention score 矩阵可分解为四块，
其中 V→V 为自注意力、V→T 为交叉注意力，两者正是 fusion 算法所需的输入。

```
              Key (列)
         ┌──────────┬──────────┐
         │  Text    │  Video   │
    ┌────┼──────────┼──────────┤
  Q │Text│   T→T    │   T→V    │
  u ├────┼──────────┼──────────┤
  e │Vid │   V→T    │   V→V    │
  r │    │ (cross)  │ (self)   │
  y └────┴──────────┴──────────┘
```

与 Wan/LTX 不同的一点是**两半来自同一个模块**：CogVideoX 每个 block 只有一个 `attn1`，把
caption 与 video 拼成一条 joint 序列，所以 Wan 需要两个 probe 类、LTX 还需要一次额外 forward，
而 CogVideoX 只需要一个 hook，source 本来就在跑的 step-5 forward 就够。

## Token 几何（必须先确定，否则 mask 全是错的）

CogVideoX1.5 在 patchify **之前**把采样的 latent 帧数补到 `patch_size_t` 的整数倍
（`CogVideoXPipeline` 第 5 步），所以 temporal token 数不是 `frames // temporal_compression_ratio`。
配置的 81 帧下：

| 量 | 值 |
| --- | --- |
| pixel frames | 81（补到 85 解码，再丢掉第一帧） |
| latent 帧 | 21 → 补到 22 |
| token grid `[T,H,W]` | **(11, 48, 85)** |
| video tokens | 11 × 48 × 85 = **44880** |
| text tokens | 226（`max_text_seq_length`） |
| joint 序列长度 | 45106 |

token 顺序 t-major → h → w，token `t` 覆盖 latent 帧 `(2t, 2t+1)`。`token_mask_to_latent_mask`
的 ceil 缩放在这三个轴上恰好都是 2，投影正确；倒推时第一个 latent 帧是解码会丢掉的 padding，
因此不会被任何 token 覆盖。

这条几何只在 [adapters/cogvideo_attention.py](../../adapters/cogvideo_attention.py) 的
`cogvideo_token_grid()` 里算一次，probe、region mask 和 latent blend 共用它。

## 算法

融合分数（与 Wan/LTX 完全一致，复用 `carnator/masks.py`）：

```
F = robust_norm( S * (eps + (1 - eps) * C^gamma) )
```

- `S` = 自对比分数（noun seeds 注意力 − background seeds 注意力）
- `C` = 交叉注意力分数（与名词相关 caption token 的注意力）
- `eps` = 自对比权重，`gamma` = 交叉注意力指数

掩码三层，逐层空间膨胀：

1. **Mbase** — 由融合分数经滞后阈值（high/low）生成的核心掩码
2. **Medit** — Mbase + 空间膨胀（`mask_spatial_medit_radius`）
3. **Msee** — Medit + 额外膨胀（`mask_spatial_msee_radius`）

核心算法复用 `carnator/masks.py` 的 `build_canonical_fusion_mask()` / `robust_norm()`，
CogVideoX 侧只提供 self / cross 两个分数。

## 实现要点

### Attention 捕获

`CogVideoMultiNounJointAttentionProbe`（[carnator/cogvideo_attention_probe.py](../../carnator/cogvideo_attention_probe.py)）
在 source 的 `probe_step`（completed step，1-based）给 `attention_probe_layer` 与
`attention_self_layer` 两个 block 的 `attn1` 装 `forward_pre_hook`。每个 block 每次
transformer forward 只调用一次 `attn1`，所以该层的调用计数就是已完成的步数——probe 因此
不依赖外层 runner 的步数记账。

QKV 计算走 `adapters.cogvideo_attention.joint_qkv()`，**与 Chorus2/Carnator 稀疏处理器替换
diffusers `CogVideoXAttnProcessor2_0` 时用的是同一个函数**（含 RoPE、`norm_q`/`norm_k`），
所以 probe 测的就是它要测的那份注意力，不会随实现漂移：

```python
query, key, _value, text_tokens = joint_qkv(attention, hidden, text, rotary)
video_query = query[:, :, text_tokens:]        # [B, H, T_video, D]
scale = key.shape[-1] ** -0.5

# cross：cross_layer，每个名词一组
cross = einsum("bhqd,bhkd->bhqk", video_query, key[:, :, :text_tokens].index_select(2, noun_tokens))
cross = cross.mul(scale).mean(dim=(0, 1, 3)).reshape(*token_grid)

# self：self_layer，key 只取种子集合
noun_max = einsum("bhqd,bhkd->bhqk", q_chunk, video_key.index_select(2, noun_seeds)).mul(scale).max(dim=3).values
bg_max   = einsum("bhqd,bhkd->bhqk", q_chunk, video_key.index_select(2, bg_seeds)).mul(scale).max(dim=3).values
contrast = noun_max - bg_max                   # chunk 化的 [T,H,W]
```

- **只读 CFG 的 cond 行**：`CogVideoXPipeline` 用 `cat([latents]*2)` 配
  `prompt_embeds = cat([negative, positive])`，cond 行是最后一行；uncond 行的 caption token
  索引在那里指向的是负 prompt 的别的位置，两行平均会把 mask 抹到整个画面。
- **交叉分数**：先经 `build_cross_attention_seed_scores()` 归一化（`minmax`），再由
  `seed_indices_from_cross_score()` 取 noun seed（`mask_top_p`）与 background seed
  （`mask_background_p`）。交叉分数本身**不单独构成 Mbase**，只负责提出种子位置。
- **自对比分数**：key 只有种子，但 query 仍是 44880 个 token，逐 token logit 图会到 GB 级，
  因此按 `DEFAULT_CHUNK_SIZE = 512` 分块；产出的三个键
  （`self_noun_max` / `self_bg_max` / `self_contrast_max`）与 LTX probe 完全同名。
- **失败即抛**：`probe.validate()` 在 source 结束前检查每个名词都拿到了 cross 与 self 记录；
  token 数与 token grid 不符、名词索引越过 caption 长度、`attention_self_layer <
  attention_probe_layer` 都直接抛异常。旧实现用 `print("Warning: ...")` 继续跑并产出空 mask。

### 每名词一个 Mbase

source 对每个名词分别跑 `build_canonical_fusion_mask()`，得到 `{noun: Mbase[T,H,W]}`，
用 `build_per_noun_mbase_bundle()` 落盘到 `<tmp_cache>/carnator_fusion_masks/`，
再用 `write_early_signature()` 把每个名词的 tail 帧 CIELAB 颜色签名写到
`<tmp_cache>/source_signature/`。payload 里带 `fusion_mask_dir` 与 `signature_path`。

target 通过 `materialize_per_noun_inference_mask_bundle()` 选出一个 source 名词的 Mbase：
名词集合相同时取面积最小的，恰好一个名词替换时取被替换掉的那个 source-only 名词；
其余情况直接报错。这条选择规则与 Wan/LTX 共用同一份 `select_per_noun_inference_mbase()`。

### 掩码产物

`carnator_fusion_masks/` 下为 `masks.pt`（Mbase / Medit / Msee / raw_coverage 张量）与
`masks.json`（eps、gamma、阈值、膨胀半径、token_shape、selection 等元数据）。
target 自己再落一份 `inference_masks/`。token → latent 的投影由
`token_mask_to_latent_mask()` 完成。

`mask_policy`（写进 source signature，参与命中兼容性判定）：

```
cogvideo-native-self-cross-fusion-mbase|cross_layer=15|self_layer=15|completed_step=5
  |top_p=0.02000000|background_p=0.02000000|eps=0.20000000|gamma=1.00000000
```

## 配置

`configs/models/cogvideox15.yaml` 的 `carnator_b5` 段：

```yaml
carnator_b5:
  probe_step: 5                  # source 在这一 completed step 采集 probe
  reuse_step: 10                 # target 从这里恢复
  regional_layers: 14-27         # Medit/Msee regional attention 的层
  outer_sparse_layers: 0-13,28-41
  attention_probe_layer: 15      # cross：提出种子
  attention_self_layer: 15       # self：把种子变成对比度；必须 >= attention_probe_layer
  mask_top_p: 0.02               # noun seed 的 top-p 比例
  mask_background_p: 0.02        # background seed 的 top-p 比例
  fusion_eps: 0.2                # 自对比权重
  fusion_gamma: 1.0              # 交叉注意力指数
  fusion_high_threshold: 0.45    # 滞后阈值（高）
  fusion_low_threshold: 0.35     # 滞后阈值（低）
  mask_spatial_medit_radius: 2   # Medit 膨胀半径
  mask_spatial_msee_radius: 3    # Msee 膨胀半径
  latent_injection_mode: boundary-blend
  boundary_alpha_function: smoothstep
  ffn_backend: w8a8
```

`attention_self_layer` 必须 **≥** `attention_probe_layer`：self 对比读的是同一个 transformer
forward 里 cross 分数刚产出的种子，更早的层拿不到种子。

已移除的过时参数：`mask_dilation_edit`、`mask_dilation_see`（改用上面的 radius 参数）。

## 重构前后对照

| 方面 | 旧实现 | 现在 |
| --- | --- | --- |
| 依赖 | 需要外部分割模型 | 无外部依赖 |
| 掩码来源 | 视频分割 | joint attention 分数 |
| probe 的 QKV | 自行拼 `to_q`，把 video token 当 text 喂进 3072 维投影（必然报错） | `joint_qkv()`，与稀疏处理器同源 |
| token grid | 直接返回原始 latent 维度 `[22,96,170]` | `cogvideo_token_grid()` → `(11,48,85)` |
| 每名词掩码 | 所有名词共享同一个 mask | 每名词一个 Mbase，target 按替换规则选择 |
| 失败方式 | `print("Warning: ...")` 后产出空 mask | `validate()` fail-closed，直接抛 |
| 配置依赖 | 需读取 `carnator_b5` + `chorus` | 只需 `carnator_b5` |

代码落点：`carnator/cogvideox15.py` 的 `_source()`（dense 50 步 + 双 hook probe + 落盘
mask/signature）与 `_target()`（选择掩码、跑 probe、boundary-blend 融合、region + outer
sparse + w8a8 FFN）。
