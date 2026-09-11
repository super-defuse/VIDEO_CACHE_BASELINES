# Carnator Fusion Mask 在 CogVideoX 上的实现

CogVideoX 版 Carnator 曾用外部分割模型（Grounded-SAM2）生成 mask，与 Wan2.2 的纯 fusion mask
算法不一致。本文档记录可行的替代方案与重构结果。状态见 [KNOWN_ISSUES.md](KNOWN_ISSUES.md) §3.2。

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

## 算法

融合分数：

```
F = robust_norm( S * (eps + (1 - eps) * C^gamma) )
```

- `S` = 自对比分数（noun seeds 注意力 − background seeds 注意力）
- `C` = 交叉注意力分数（与名词相关 text token 的注意力）
- `eps` = 自对比权重，`gamma` = 交叉注意力指数

掩码三层，逐层空间膨胀：

1. **M_base** — 由融合分数经滞后阈值（high/low）生成的核心掩码
2. **M_edit** — M_base + 空间膨胀（`mask_spatial_medit_radius`）
3. **M_see** — M_edit + 额外膨胀（`mask_spatial_msee_radius`）

核心算法复用 `carnator/masks.py` 的 `build_canonical_fusion_mask()` / `robust_norm()`，
CogVideoX 侧只需提供 self / cross attention 分数。

## 实现要点

### Attention 捕获

在 source 的 `probe_step` 处 hook transformer block 的 attention 层（`attention_probe_layer`），
在 SDPA 之前按文本 token 数切分 score 矩阵：

```python
scores = (Q @ K.transpose(-2, -1)) / sqrt(D)      # [B, H, T_total, T_total]
video_q = scores[:, :, text_tokens:, :]           # [B, H, T_video, T_total]
self_scores  = video_q[:, :, :, text_tokens:]     # V→V
cross_scores = video_q[:, :, :, :text_tokens]     # V→T
```

实现位于 [carnator/cogvideo_attention_probe.py](../../carnator/cogvideo_attention_probe.py)，
安装与区域注意力状态由 [adapters/cogvideo_attention.py](../../adapters/cogvideo_attention.py) 提供。

### 分数计算

- **自对比**：对 head 取平均后，分别取 noun seeds 与 background seeds 上的最大注意力，相减，
  再 reshape 回 token grid `[T, H, W]`
- **交叉注意力**：对 head 取平均后，取名词对应 text token 索引上的最大注意力，reshape 到 `[T, H, W]`
- **noun seed 选择**：对 cross score 在时间维取平均得到 `[H, W]`，按 `mask_top_p` 取最大
  （noun seed）与最小（background seed）；CogVideoX 的 `mask_background_p` 控制背景种子比例

### 掩码产物

写入 `root / "carnator_fusion_mask"`，含 `masks.pt`（Mbase / Medit / Msee / score 张量）
与 `masks.json`（eps、gamma、阈值等元数据）。payload 中记录 `mask_bundle_path`，
target 阶段由 `_mask()` 加载，并从 token 空间投影到 latent 空间。

## 配置

`configs/models/cogvideox15.yaml` 的 `carnator_b5` 段：

```yaml
carnator_b5:
  attention_probe_layer: 15      # 探测哪一层 transformer
  mask_top_p: 0.02               # noun seed 的 top-p 比例
  mask_background_p: 0.02        # background seed 的 top-p 比例
  fusion_eps: 0.2                # 自对比权重
  fusion_gamma: 1.0              # 交叉注意力指数
  fusion_high_threshold: 0.45    # 滞后阈值（高）
  fusion_low_threshold: 0.35     # 滞后阈值（低）
  mask_spatial_medit_radius: 2   # M_edit 膨胀半径
  mask_spatial_msee_radius: 3    # M_see 膨胀半径
```

已移除的过时参数：`mask_dilation_edit`、`mask_dilation_see`（改用上面的 radius 参数）。

## 重构后的变化

| 方面 | 重构前（GroundedSAM2） | 重构后（fusion mask） |
| --- | --- | --- |
| 依赖 | 需要外部分割模型 | 无外部依赖 |
| 掩码来源 | 视频分割 | attention 分数 |
| 速度 | 慢（视频处理） | 快（无额外模型） |
| 一致性 | 依赖分割质量，可能不稳定 | 确定性 |
| 配置依赖 | 需读取 `carnator_b5` + `chorus` | 只需 `carnator_b5` |

代码落点：`carnator/cogvideox15.py` 的 `_source()`（捕获 attention 并生成 mask）、
`_generate_fusion_mask_from_latent()`、`_mask()`（从 payload 加载 mask bundle）；
`_signature_metadata()` 的 `mask_policy` 现为 `cogvideo-fusion-mask-v1`。

## 调参参考

- `fusion_eps` (0.1–0.3)：平衡自对比与交叉注意力
- `fusion_gamma` (0.8–1.2)：调节交叉注意力敏感度
- 高低阈值：控制掩码大小与滞后行为

## 参数说明（Wan 侧参考）

Wan2.2 的对应实现见 [carnator/adapters/wan_ti2v_5b.py](../../carnator/adapters/wan_ti2v_5b.py)，
可作为对齐算法语义的参照。
