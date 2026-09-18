# Chorus 反例定性实验（P1）

用 Wan2.2 适配版 Chorus 跑两对 prompt，产出一张四行对比图：每行最左侧是 prompt，
右侧是该视频的 5 帧画面。目的是给出两个失败案例，分别对应 reuse validity 和 reuse scope
两个 gap。

**采用强制复用**：target 通过 `cache_prompt` 固定到本 pair 的 source cache，不经过 CLIP
相似度门控。CLIP cosine 仍然记录，但只作为审计字段，不参与判定。

## 1. 两对 prompt

| Pair | 角色 | Prompt | 差异 |
| --- | --- | --- | --- |
| **A** | source（缓存） | `a red apple rolling down a snow mountain, camera focus on the apple` | — |
| **A** | target | `a red apple rolling down a mountain, camera focus on the apple` | 删除 `snow` |
| **B** | source（缓存） | `a barbell on a gym floor, camera focus on the barbell` | — |
| **B** | target | `a football on a gym floor, camera focus on the football` | 单名词替换 `barbell` → `football`，背景词不变 |

两条 prompt 的 seed、步数、分辨率全部相同，只差上表列出的部分。

**`camera focus` 后缀必须两侧同时出现。** 它的作用是让 Grounded-SAM-2 更稳定地分割出编辑区域；
但只给一侧加会把 CLIP cosine 压到门控以下，该 pair 就不再是门控反例。实测（`clip-vit-large-patch14`）：

| 变体 | CLIP cos | 过 0.65 门控 |
| --- | --- | --- |
| Pair A 两侧都加 | `0.9149` | ✓ |
| Pair B 两侧都加 | `0.7303` | ✓ |
| Pair B 只给 barbell 加 | `0.6125` | ✗ |
| Pair B 只给 football 加 | `0.6000` | ✗ |

因此后缀成对出现不是为了改写 prompt 语义，而是保持 CLIP 相似度不失真。

Pair B 的背景词在两条 prompt 中**完全相同**（`a gym floor`），这是刻意的：背景复用本身是有效的，
失效点落在更新区域（Medit）的形状与球体不匹配上，这样两个 gap 才能被干净地分开。
注意"背景被原样保留"只对**编辑区域之外**成立（地垫格纹、杠铃片堆、墙面立柱都在），
编辑区域内部沿用的是源物体的位置与地面——详见 §1 失败机理里 Pair B 那段。

Pair B 还必须满足第三个条件：**CLIP cosine 过 0.65 门控**。否则审稿人会说这个失败是强制复用
造出来的，Chorus 本来就不会命中。prompt 用短句式正是为了这个——给杠铃加形容词描述会强化杠铃语义，
反而把 CLIP 压低（实测 `a black barbell with two round weight plates lying on a wooden gym floor`
只有 `0.5273`）。候选扫描与渲染检查见 §3 备注。

### 失败机理

- **Pair A（validity）**：已跑通。CLIP cosine `0.9149`（高于阈值 0.65），即**文本层面高度相似、
  Chorus 自己也会命中**，而输出仍是雪山苹果——这是本节最强的一个反例。
  （实测值见 `pair_A/clip_similarity.json`。）
- **Pair A 机理**：Chorus 的 Grounded-SAM-2 只分割 `red apple`，雪山背景不在任何 mask 内，
  于是 Stage-2 的 source latent 被原样复用到背景上 → 输出仍是**雪山上的苹果**，而 target 要求的
  是一座普通的山。即：历史计算对当前请求**无效**。
- **Pair B（scope）**：已跑通。CLIP cosine `0.7303`（**同样高于 0.65，Chorus 自己也会命中**）。
  背景词两边完全相同，背景复用本身是**有效**的；失效点在更新区域的形状。
  Chorus 分割 `barbell` 后向外扩 `mask_dilation_edit=2` 得到 Medit，而 Medit 的形状仍是
  "一根长杆 + 两片圆盘"、横贯整个画面的细长双叶形（base/edit/see =
  `0.1536` / `0.2851` / `0.9378`），与紧凑的球体严重不匹配。
  实测输出：小球出现在源杠铃**圆盘**所在的那个足迹上、尺寸贴着那片圆盘，源圆盘的高光残影
  仍留在区域内部，横杆那一段则没有生成任何物体，远小于 dense 对照里那个居中、横躺、
  占满画面的球（序列末帧球体发虚）。背景侧的证据是双向的：地垫格纹、
  左上角那摞杠铃片、墙面的深色立柱都在输出中原样保留（背景复用确实生效）；但编辑区域
  内部沿用的是源物体的几何布局，而不是生成一个符合新 prompt 的独立物体。
  也就是说，重建区域**沿用了源物体的几何布局**。即：该不该复用没问题，**复用哪一块**选错了。
  （实测值见 `pair_B/clip_similarity.json`、`pair_B/report.md`。）

  > **v1 → v2 的差异**：v1 的 source 渲染成绿片杠铃 + 深色地垫 + 木踏板的近景，`base` 为
  > `0.2159`；加上 `camera focus on the barbell` 后 source 被重新采样成黑片杠铃的宽景，
  > `base` 降到 `0.1536`，编辑区域覆盖的地面更多，"背景被原样保留"这一条因此**比 v1 弱**。
  > 失败性质不变（更新区域跟着源物体走），但正文若引用本 pair，描述应与上面这段一致，
  > 不要沿用 v1 关于"背景原样保留、仅残留杆状物"的措辞。

### 实测结果（现行产物）

两次运行（Pair A 在物理卡 3、Pair B 在物理卡 4）的实测值：

| 项 | Pair A | Pair B |
| --- | --- | --- |
| CLIP cosine | `0.9148925164950095` | `0.7302866210184978` |
| 过 0.65 门控 | ✓ | ✓ |
| `forced_cache_hit` | ✓ | ✓ |
| `source_uuid_matches` | ✓ | ✓ |
| `sam_phrase` | `red apple` | `barbell` |
| `update_mask` | `Medit` | `Medit` |
| base / edit / see | `0.0428` / `0.0779` / `0.4111` | `0.1536` / `0.2851` / `0.9378` |
| 执行步数 | 45 / 50 | 45 / 50 |

两条 pair 的六项自动判据全部通过（第六项 `mask_dilation_matches` 交叉验证 mask 元数据里记录的
膨胀半径确为 2，而不是回落到 Chorus 自身默认的 8）；`clip_score`、`frames`、`report` 三个阶段
均正常产出。`base_token_fraction` 两个 pair 都远离 0 与整帧，未触发判据 3 的退化条件。

## 2. 实验流程

两对**并行**：Pair A 在物理卡 3，Pair B 在物理卡 4。每对内部串行。

| 阶段 | 内容 | 说明 |
| --- | --- | --- |
| `source` | 用 `chorus` 方法跑 source prompt，完整 50 步 | 产出 source 视频 + `chorus-trajectory-v2` cache |
| `target-dense` | 用 `baseline` 方法跑 target prompt（同 seed） | 可选，仅 `--with-dense` 时执行（见下） |
| `chorus-reuse` | 用 `chorus` 方法跑 target prompt，`cache_prompt=SOURCE_PROMPT` 强制复用 | 命中由构造保证，不触发检索门控 |
| `score` | 记录 CLIP cosine（仅审计） | 可能低于 0.65，属预期，不阻断流程 |
| `frames` | 从各阶段视频抽 5 帧 | 帧号 `0, 30, 60, 90, 120`（共 121 帧）；source 与 chorus-reuse 必抽，`--with-dense` 时加 target-dense |
| `report` | 汇总产出 `report.md` 与 `experiment_manifest.json` | 跑 §3 的自动判据（判据 1–3、5–6；判据 4 的人眼确认不在此列） |

GPU 阶段（`source`/`target-dense`/`chorus-reuse`/`score`）各起一个子进程跑在指定卡上，`frames`
与 `report` 是 CPU 收尾，在主进程内联执行。四行对比图**不是** stage，由
`make_chorus_counterexample_figure.py` 单独生成（见 §4）。

`--with-dense` 可选：额外跑一条 target dense control（同 seed、同 prompt），用于说明"正确输出
应该长什么样"。**不计入四行图**，只作为质证材料保留。

默认参数：seed `20260218`、50 步 UniPC、`1280x704`、121 帧、Chorus 默认 `k1=5 / k2=30`、
`mask_dilation_edit=2`（Chorus 自身默认是 8，本实验刻意取更小的 halo，让复用区域贴近源物体
自身的足迹，从而把失效归因到区域的**形状**而不是**大小**；`mask_dilation_see` 仍是默认的 14），
`update_mask=Medit`（即真实的 `chorus` 方法，不是 `chorus_mbase_only`）。

```bash
cd project/VIDEO_CACHE_BASELINES
CONDA=/data/shared/miniconda/bin/conda
RUN="python scripts/benchmark/run_chorus_counterexample_p1.py --stage all --with-dense"

# Pair A → 物理卡 3；--mask-dilation-edit 已是脚本默认值 2，这里显式写出以便审计
$CONDA run -n wan env PYTHONNOUSERSITE=1 PYTHONPATH=. \
  $RUN --pair A --gpu 3 --mask-dilation-edit 2 \
  --output-root artifacts/chorus_counterexample_p1_v2_halo2/pair_A \
  > logs/chorus_p1_pair_A_halo2.log 2>&1 &

# Pair B → 物理卡 4
$CONDA run -n wan env PYTHONNOUSERSITE=1 PYTHONPATH=. \
  $RUN --pair B --gpu 4 --mask-dilation-edit 2 \
  --output-root artifacts/chorus_counterexample_p1_v2_halo2/pair_B \
  > logs/chorus_p1_pair_B_halo2.log 2>&1 &

wait
```

等价地直接跑 `bash scripts/benchmark/run_chorus_p1_parallel.sh`（两对并行、卡 3/4，默认就带
`--mask-dilation-edit 2`；`EXTRA_ARGS="--mask-dilation-edit 8"` 可复现 Chorus 默认 halo）。

输出根为 `_v2_halo2`，脚本的 `DEFAULT_OUTPUT_ROOT` 与作图脚本的默认输入都指向它。Chorus 自身
的默认配置 `configs/methods/default.yaml` 里的 `mask_dilation_edit: 8` **没有被改动**，实验值只
通过命令行覆盖写进每个 stage 的 `config.resolved.json`。
（`conda run` 会缓冲 stdout，外层的 `logs/*.log` 在进程结束前是空的；要看进度请读
`artifacts/chorus_counterexample_p1_v2_halo2/pair_{A,B}/logs/{stage}.log`。）

只启动 gpu3 和 gpu4，不动其他卡。脚本拒绝覆盖已有输出目录；重跑某阶段用相同的
`--output-root` 加 `--stage`（`all` / `source` / `target-dense` / `chorus-reuse` / `score` /
`frames` / `report`），子进程内部的 `cuda:0` 由 `CUDA_VISIBLE_DEVICES` 映射到命令行指定的物理卡。
图由独立的 `make_chorus_counterexample_figure.py` 生成，不在本脚本的 `--stage` 里。

脚本是 [run_chorus_mbase_failure_case.py](../../scripts/benchmark/run_chorus_mbase_failure_case.py)
的泛化：新增 `--pair` 与 prompt 常量表、`--gpu`、`--with-dense`、`--mask-dilation-edit` 与
`frames` 一个 stage；保留其 `cache_prompt` 强制复用与 `prompt_diff_override` 兜底逻辑。

## 3. 有效性判据（任一不满足则该 pair 作废，需改 prompt 重跑）

1. `results.jsonl` 中 target 行 `cache_match.hit == true`，且 `source_uuid` 指向本 pair 的 source；
2. `chorus_mask.phrase` 是期望的 source 实体（Pair A 为 `red apple`，Pair B 为 `barbell`）；
3. `chorus_mask` 的 `base_token_fraction / edit_fraction / see_fraction` 均已记录且非退化
   （`base_token_fraction` 不应接近 0 或覆盖整帧）；
4. **人眼确认失败可见**：Pair A 的输出保留了雪山（有雪的背景原样复用）；
   Pair B 的输出把球放在了源杠铃**圆盘**的位置上、尺寸贴着那片圆盘，横杆那一段没有物体，
   与 dense 对照那个居中横躺、占满画面的球明显不同。若 Pair B 没有复现，优先改 prompt，
   不要改配置。
5. **CLIP cosine ≥ 0.65**（对 Pair B 尤其关键）：低于门控则说明该 pair 只是强制复用的产物，
   Chorus 本来就不会命中，不能作为门控反例。Pair A 为 `0.9149`，Pair B 为 `0.7303`，均满足。

审计字段（写入 `report.md`，不参与判定）：CLIP cosine 及其 `passes_threshold`、
prompt-diff 后端、`update_mask`、各阶段是否命中。

**改 prompt 的注意事项**：`camera focus` 后缀一旦引入就必须两侧对称（见 §1 表），
只加一侧会把 CLIP 压到 `0.61` 左右、直接违反判据 5。其余改动仍遵循"一次只改一处"。

**后缀引入记录**（2026-09-14，v2）：四条 prompt 统一追加 `camera focus on the <object>`
（Pair A 为 `apple`，Pair B 为 `barbell` / `football`），目的是稳定 Grounded-SAM-2 的编辑区域分割。
四个 prompt 必须同时改，理由见 §1 的 CLIP 对照表——单侧加后缀会把 cosine 压到 `0.61` 量级。
（更早的无后缀 v1 产物已随本次定稿删除，prompt 本身记录在下条。）

**Pair B 的 prompt 修正记录**（2026-09-14，v1，无后缀）：

1. 原计划 `a barbell on a playground` / `an American football on a playground` 无法使用：
   Wan2.2 把这个组合渲染成"只有秋千和小孩的操场"，画面中没有杠铃，Grounded-SAM-2 直接报
   `Grounding DINO found no object for phrase: 'barbell'`。（该版本 CLIP 为 `0.7360`，过门控。）
2. 备选 `a barbell on the grass` 同样不可用：抽帧检查发现渲染结果是"一条链条吊着一个铃铛状金属物"，
   不是杠铃。
3. 给杠铃加描述性定语的版本渲染正常（一根带两片圆盘的标准杠铃），但 CLIP 掉到 `0.5273`，
   意味着 Chorus 的门控本来就会拒绝，该 pair 就不再是门控反例——**这是选用短句式的原因**。
4. 对 8 个短句候选做了 CLIP 扫描（`clip-vit-large-patch14`，与实验一致）：
   `a barbell on a gym floor` `0.7428` > `a gymnasium floor` `0.7396` > `a wooden gym floor` `0.7331`
   > …… > `a wooden floor` `0.6735`，全部过门控。
5. 对得分最高的两个候选做了 source 渲染检查：
   `a barbell on a gym floor` 渲染出干净的落地杠铃（无人物，横贯画面）；
   `a barbell on a wooden gym floor` 渲染成"一个女人劈叉举着杠铃"，画面有人且杠铃离地，排除。
6. v1 最终采用 `a barbell on a gym floor` / `an American football on a gym floor`：
   CLIP 过门控、分割目标清晰、背景词完全一致。
7. v2 把 target 的名词改为更短的 `a football`（与其他三条 prompt 的句式对齐），
   CLIP 为 `0.7303`，仍过门控。

## 4. 图表规格

一张图，4 行 × 6 列（1 列 prompt + 5 列画面）：

| 行 | 左侧 prompt 列 | 右侧 5 帧 |
| --- | --- | --- |
| 1 | `Pair A · cache` | Pair A source 视频 |
| 2 | `Pair A · Chorus output` | Pair A Chorus 输出 |
| 3 | `Pair B · cache` | Pair B source 视频 |
| 4 | `Pair B · Chorus output` | Pair B Chorus 输出 |

左侧两行文字即上方角色标注与 prompt 原文，由脚本从各 pair 的 `prompts.json` 直接读取，
改 prompt 后重跑 `figure` 阶段即可自动同步，不需要手改图注。

- 左侧列：字体 `/home/tangxuebin/font/arial.ttf`，随文字长度自动换行。
- 5 帧横向等距、无边框、行间留白，**跨行不加任何后期标注**（画面本身即证据）。
- 输出 `artifacts/chorus_counterexample_p1_v2_halo2/figures/chorus_counterexample_p1.{png,pdf}`，
  `dpi=300`，图宽约 8.3 英寸（每帧 1.35 英寸 + prompt 列 1.6 英寸），总高约 3 英寸，适配论文通栏。

图由 `scripts/benchmark/make_chorus_counterexample_figure.py` 生成，默认从
`artifacts/chorus_counterexample_p1_v2_halo2/pair_{A,B}` 读取 4 行（每 pair 的 `prompts.json`
提供左侧文字、`source_dense` / `chorus_medit` 两个视频提供画面），与生成流程解耦；
`--rows-json` 可整体替换行列表，便于单独重排。

## 5. 产物

```text
artifacts/chorus_counterexample_p1_v2_halo2/
  pair_A/
    source_dense/videos/            # source 视频
    chorus_medit/videos/            # target Chorus 输出
    target_dense/videos/            # 可选 dense control
    cache/                          # 本 pair 私有的 Chorus trajectory
    frames/                         # 每视频 5 帧 PNG
    clip_similarity.json            # 审计用，不参与判定
    report.md                       # 含 §3 的六条判据结果
    experiment_manifest.json
  pair_B/ ...
  figures/chorus_counterexample_p1.{png,pdf}
```

`_v2_halo2` 是定稿产物；更早的 v1/v2 目录（无后缀 / halo=8）已删除，正文与图中引用的一律是它。

## 6. 边界

本节只论证"CLIP 相似度不足以判断历史计算可否复用"这一**问题**的存在，不构成 Carnator 优于
Chorus 的证据；有效性证据由 P0 的 risk–coverage 曲线给出。

由于采用强制复用，图中每一行必须在正文或图注中显式标注 `forced cache reuse`，并同时给出
实测 CLIP cosine。**不得**表述为 Chorus 在默认配置下自行命中——这是本节与 P0 的分工所在，
也是唯一会让这张图失去可信度的写法。

两对 prompt 的 CLIP cosine（`0.9149` / `0.7303`）都高于 Chorus 的检索阈值 `0.65`，
所以准确的说法是：**Chorus 的门控在面对这两对 prompt 时会接受复用**，只是本实验用
`cache_prompt` 把这一接受结果固定在特定 source 上（以确保复用的确实是本 pair 的历史缓存）。
这个区分必须写清楚——它是这张图可信度的来源，也是它区别于"用低相似度 prompt 硬凑反例"的地方。
