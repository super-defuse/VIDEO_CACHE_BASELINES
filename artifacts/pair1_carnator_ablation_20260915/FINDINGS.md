# pair1-wan-carnator「target 仍是红 bus」归因（E1–E3）

2026-09-15。复现对象：`project/CHAI/artifacts/runs/pair1-wan-carnator`（source `a red bus driving
through a city` seed 20260218 → target `a red car driving through a city` seed 20260219）。
三个实验各自只改一个 flag，其余逐字复刻 `run.log` 里的 reuse 命令，产物在 `e1*/ e2*/ e3*/`。

## 结论

**三个实验都仍然输出 source 的红 bus（连车身黄字涂装都一样）。** 决定性的一条是 E1：
把整条 latent 注入路径关掉（`calls: 0`、`enabled: false`）之后，画面没有任何回归成车的迹象。

因此 mask（Mbase 太散）、注入模式（hard vs boundary-blend）都不是这个症状的成因。

真正的原因在这两行：

```python
# carnator/adapters/wan_ti2v_5b.py:2846-2849
if args.reuse_start_step == 0:
    resume_latent_path = source_initial_latent
else:
    resume_latent_path = source_trajectory[args.reuse_start_step - 1]
```

reuse 阶段 target 的**起始 latent 就是 source 在 completed step 9 的 latent**（等价于
`source_dense/latents_step_009.pt`），然后只跑 step 10–49 共 40 步。三个实验的 target cache
manifest 都写着 `completed_steps: 40 / steps: 50`，与 `reuse_start_step=10` 吻合。

也就是说 target 从来没有用自己的噪声起步——它是在 source 视频的第 9 步轨迹上**续写**。
在这个前提下，Mbase / Mband / Medit 能决定的只有「哪些像素允许从 source 漂移出去」，
决定不了「哪里是 target 自己的内容」。bus 确实落在 Mbase 里（允许漂移），但它漂移的起点
本身就是 source 那只 bus 的 latent，40 步 CFG 换一个名词不足以把它重绘成车。E1（完全没有
pin、完全自由漂移）就是这条的下界证明。

## 各实验实测

| 实验 | 改动 | PSNR vs source | PSNR vs target dense | 画面 |
| --- | --- | ---: | ---: | --- |
| published reuse | — | 23.02 | 7.60 | 红 bus |
| E1 | `--no-latent-injection` | 17.03 | 7.58 | 红 bus（+ 掩码边缘伪影） |
| E2 | `--latent-injection-mode hard` | 20.91 | 7.60 | 红 bus |
| E3 | `--source-mask-dir .../self_contrast_masks` | 22.87 | 7.60 | 红 bus |

（PSNR vs target dense 全部 ~7.6 dB，说明三者都和 target 自己的 dense 解无关。）

E3 用的名词门控 mask：`base_fraction 0.596 / band 0.208 / edit 0.804`，比 published 用的融合
mask（`0.319 / 0.309 / 0.629`）覆盖更大，结果没有区别——再次说明 mask 不是这个症状的杠杆。

## 这不是 CHAI bridge 特有的

VidProM 300 产物自己的表里，Carnator Boundary B5 的 CLIP-T 是五个方法里**最低**的：

| 方法 | CLIP-T | VBench-Q |
| --- | ---: | ---: |
| Chorus2 | 0.3017 | 0.6417 |
| Chorus | 0.2993 | 0.6407 |
| TeaCache | 0.2938 | 0.6049 |
| NIRVANA-VID | 0.2855 | 0.6487 |
| **Carnator Boundary B5** | **0.2616** | 0.6401 |

VBench-Q 与其他方法持平，CLIP-T 明显掉队——正是「输出更贴 source、不够贴 target prompt」的
聚合层签名。pair1 只是把这个现象放大到肉眼可见。

## 顺带记录：CHAI bridge 与 300 产物在两处本就不同

1. `--latent-injection-mode`：产物是 `hard`（argparse 默认值，`run_vidprom_300_experiment.py`
   从不传这个 flag），CHAI 传 `boundary-blend`。本 pair 下 boundary-blend 注入的 source 权重
   ≈49.6%，hard ≈37.2%（Mbase 0.319 / Medit 0.629 / band 0.309，band 内 ᾱ=0.403）。
2. `--source-mask-dir`：产物传 per-noun bundle（从 `online_color_signature.json` materialize），
   CHAI 不传，回落到粗的 `fusion_masks`。

两处都不影响本症状，但会影响那一行与论文表格的可比性。

## 还没做、但能一刀切开剩下问题的一个实验（E5，无需改代码）

让 target 从一个**干净 dense 续写**跑一遍，验证「从 source 的 x_9 出发、不套任何 mask/稀疏」是否
也留不住车：

```bash
/data/shared/conda_envs/wan/bin/python -m carnator.adapters.wan_ti2v_5b \
  --no-env-setup --model-dir /home/tangxuebin/models/Wan2.2-TI2V-5B --device cuda:0 \
  --artifact-root .../e5/artifacts --results-root .../e5/results --run-id run \
  --target-prompt "a red car driving through a city" --seed 20260219 \
  --height 704 --width 1280 --frames 121 --steps 50 \
  --target-dense-only --outer-attn-backend dense --ffn-backend fp \
  --skip-w8a8-dense-diagnostic --disable-progress \
  --target-resume-latent <source_dense>/latents_step_009.pt --target-resume-step 10
```

- 若仍然是红 bus → 问题完全在「从 source 轨迹续写」这条设计上，mask/attention 都是无辜的，
  要出车必须换起始点（例如从 target 自己的 probe latent 起步，只 pin 背景）。
- 若变成车 → mask/稀疏路由确实压制了改写，才值得回头调 mask。
