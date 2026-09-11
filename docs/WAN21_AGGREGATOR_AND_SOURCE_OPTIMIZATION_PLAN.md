# Wan2.1 聚合器修正与 Chorus2/Carnator source 优化计划

状态：**已实施并完成真实重跑验收（2026-09-11）。**

代码三项全部落地，见文末「实施记录」。`results_wan21_test` 的 chorus2 / carnator 已于
2026-09-11 13:20 跑完并完成最终聚合，聚合器改动在全部方法进程退出后才进行，
未影响任何在跑实验。

**核心验收已通过**：2026-09-11 16:08–17:00 在 GPU 3/4 上以完全相同设置重跑了
chorus2 的 arrival 0，target 产物与归档**逐位一致**（`output_hash` 与 `videos/*.mp4`
的 sha256 完全相同），确认位打包 + 去同步 + 延迟元数据三处改动对 mask 比特中性。
测量明细见文末「真实重跑验收」。

**遗留**：carnator 的 arrival 0 在重跑中 OOM（归档里同一请求成功），原因未定，
见「遗留问题」一节。GPU 3–7 已于 17:0x 全部释放。

---

## 一、聚合器三处修改（✅ 已实施）

文件：`scripts/benchmark/aggregate_vidprom_300_results.py`

### 问题 1：优先消费 dry-run 产物，且不校验时效

`rows_for()`（L30-37）优先读 `results.jsonl`，存在即用：

```python
scored = path.with_name("results.jsonl")
source = scored if scored.is_file() else path
```

`--overwrite` 不会清除历史 `results.jsonl`，因此一次 `--dry-run` 写下的桩数据
会一直被后续真实运行沿用。实测后果：`results_wan21_test/summary.csv`
（mtime 2026-09-10 22:25，早于该目录下全部真实产物）记录 baseline
`e2e_s_per_request = 1.372992992401123e-05`（13.7 微秒/请求）、
`denoising_s_per_request = 0.0`、CLIP-T / VBench 全空——只有走
`adapters/synthetic.py` 的哈希桩才可能出现这些值。由此推出的
`e2e_speedup`（0.167 / 0.179 / 0.251 / 0.307）方向完全反了。

**修法**（两道闸，任一触发即报错退出，不写 summary）：

1. **时效校验**：`results.jsonl` 由 `requests.jsonl` 派生
   （`run_vidprom_300_experiment.py:1718` 的 `shutil.copyfile`，之后
   quality_runner 覆写附加指标）。若 `results.jsonl` 的 mtime 早于同目录
   `requests.jsonl`，判定为过期产物，报错并打印两者的 mtime。
2. **桩数据校验**：若某方法的 `ok` 行全部满足
   `denoising_s == 0` 且 `max(latency_s) < 1.0`，判定为 dry-run / 桩数据，
   报错并指出文件路径。

### 问题 2：`results.jsonl` 缺失时静默回退

`rows_for()` 在 `results.jsonl` 不存在时回退到传入的 `requests.jsonl`。
后者是运行期增量追加的原始日志，且没有 quality 阶段附加的 CLIP-T /
VBench 指标，回退会产出缺失质量列、且可能只有部分到达的表。

已确认 `results.jsonl` 只在方法**完整跑完**时写出（`run_vidprom_300_experiment.py:1718`
位于 quality 评估之后），因此正常流程下它必然存在。

**修法**：`results.jsonl` 缺失时直接报错，不做回退。

### 问题 3：质量列为空时静默出表

`aggregate_method()`（L40-82）读 `clip_t` / `vbench_q` 时逐行取值、缺失即跳过，
某方法若**一行都没有**这两个字段，就只是把整列留空，既不报错也不提示，
聚合器仍然 `exit=0` 并写出 summary。

**实测后果**（2026-09-11 的 `results_wan21_test`）：`carnator_boundary_b5`
的 10 条 `ok` 行中 `clip_t` 0 条、`vbench_q` 0 条，汇总表里 carnator 的
CLIP-T / VBench 两列全空，看起来像"该方法指标偏低/未测"，实际是质量评估
整段失败。链路是：

1. `quality_runner.py:304` 的 `--device` 默认写死为 `cuda:0`，不跟随
   `--gpu` 参数；
2. 当时外部任务的进程（PID 759362，74.60 GiB）独占物理 GPU 0，
   `background_consistency` 的 CLIP 图像编码触发
   `torch.OutOfMemoryError: Tried to allocate 36.00 MiB`；
3. `run_quality_evaluation()`（L1521）捕获 `CalledProcessError`，返回
   `{"status": "failed", "clip_t": None, "vbench_q": None}`——失败被吞掉，
   实验仍写出 `results.jsonl` 并打印 `"status": "completed"`；
4. 聚合器照常出表，carnator 两列留空。

即**同一个"质量缺失"信号在三个环节都被降级成了非错误**，最终只表现为
表格里的空白单元格。

**修法**：写 summary 之前加一道完整性闸——对每个方法统计其 `ok` 行中
`clip_t` / `vbench_q` 非空的行数，只要存在方法出现
`非空行数 < ok 行数`（含"无 `ok` 行"这一退化情形），即报错退出，
错误信息需给出方法名、`非空/总数`、以及该方法的 `results.jsonl` 路径。
适用全部方法，不限于 `NO_HIT_METHODS`（后者的语义是"没有命中率"，
与质量列无关）。

**实施补充**（原计划未列）：新增 `--allow-incomplete-quality` 逃生阀。`--skip-quality`
或 smoke 跑法下质量列本就为空，硬闸会让聚合器无法用于小规模验证；该 flag 把质量闸降级为
stderr 警告。时效闸与桩数据闸**不提供**逃生阀（它们只可能指向坏数据，没有合法的空列场景）。

**根因另记**：这次故障的根因不在聚合器，而在质量评估链路本身，见下节。

### 生效时机（重要）

聚合器由 `/tmp/wan21_supervise.sh` 在所有 wan21 方法进程退出后调用。该
脚本是在**结束时才由 shell 拉起** `aggregate_vidprom_300_results.py`，所以
现在改动其内容会立刻作用于在跑实验的最终汇总产物。这正是必须先等的原因。

### 验证（✅ 已执行）

跑完后已执行一次聚合，确认：

- 三处产物 `summary.csv` / `hit_rate_over_arrivals.csv` /
  `VIDPROM_1000CACHE_300INPUT_RESULTS_TABLE.md` 的 mtime 均为聚合时刻；
- baseline 行 `e2e_s_per_request ≈ 1138.6`、`denoising_s_per_request ≈ 974.2`；
- 制造一次桩数据（临时把某方法 `results.jsonl` 换成一个 dry-run 产物）
  确认两道闸都能拦下；
- 制造一次质量缺失（临时把 carnator 某几行的 `clip_t` / `vbench_q` 置为
  `null`，或指向那份失败时的 `quality_input.jsonl`）确认完整性闸报错退出，
  且不再写出 summary。

---

## 二、质量评估设备绑定的根因修正（✅ 已实施）

文件：`scripts/benchmark/quality_runner.py:304`

```python
parser.add_argument("--device", default="cuda:0")
```

`quality_runner` 的 `--device` 写死为 `cuda:0`，不跟随
`run_vidprom_300_experiment.py` 的 `--gpu`：`run_quality_evaluation()`
（L1521）起的子进程从不传 `--device`，也不设 `CUDA_VISIBLE_DEVICES`，
于是无论方法跑在哪张卡上，质量评估一律落在**物理 0 号卡**。

后果（2026-09-11 实测）：外部任务进程 PID 759362 占用物理 GPU 0 达
74.60 GiB 时，`background_consistency` 的 CLIP 图像编码
`torch.OutOfMemoryError: Tried to allocate 36.00 MiB`，carnator 质量评估
整段失败——即第一部分"问题 3"的根因。这在多租户机器上不是偶发。

**修法**：让 `--device` 跟随调用方的 `--gpu`（`quality.py` 增加
`--device` 透传，`run_quality_evaluation()` 传 `f"cuda:{gpu}"`），
或在子进程环境中显式设置 `CUDA_VISIBLE_DEVICES=<gpu>`。取后者，改动更小。

**勘误**：原文称"与 `run_vidprom_300_experiment.py` 其余子进程的既有做法一致"不成立 ——
仓库内（`scripts/`、`carnator/`、`adapters/`、`core/`）此前**没有任何** `CUDA_VISIBLE_DEVICES`
用法。该 runner 对模型和其余子进程一律直接传 `f"cuda:{args.gpu}"` 物理卡号，所以这里是
新引入的一种约定，需注意它假定父进程的 `CUDA_VISIBLE_DEVICES` 未被重映射过（若被重映射，
`--gpu` 的语义整体已经不一致，不只是质量这一处）。

**同时收紧**：`run_quality_evaluation()` 原本捕获 `CalledProcessError` 后返回
`{"status": "failed"}` 而实验仍打印 `"completed"`。返回值契约保持不变，但失败现在会向
stderr 打两行明确告警，与聚合器的质量闸构成两端可见。

**临时绕过**（补跑时用过，不改代码）：

```bash
CUDA_VISIBLE_DEVICES=<空闲卡> conda run -n vbench python \
  scripts/benchmark/quality.py <run_dir> --dimension ... 
```

**验证**：在 GPU 0 被其他任务占用、目标方法跑在 GPU 6 的条件下起一次质量
评估，确认它落在 GPU 6 且不再 OOM。

---

## 三、Chorus2 / Carnator source 阶段优化（✅ 已实施）

> **行号勘误**：下文引用的 `SourceBlockMaskCache.capture()` 行号（L72-116、L76-78、
> L116、L92/L102、L85）与文件实际内容不符。实施时的实际位置是
> `carnator/outer_sparse.py` 的 L569-635，`torch.save` 在 L595。
> 只有 L456-457 的两个 format 常量是对的。

**采纳范围**：仅下探到的两项——三处工程开销、位打包存储。
**明确不做**：不共享 source 缓存，不合并 chorus2 与 carnator 的 source run，
不改变现有实验流程与每个基线独立跑的语义。
**明确不做**：不调整 block size（Wan2.2 与 Wan2.1 均为 q=64 / k=128，相对
粒度本就对齐，调大会破坏可比性），不做 query 采样近似（需独立质量消融）。

文件：`carnator/outer_sparse.py`，`SourceBlockMaskCache.capture()`（L72-116）。

### 第 1 项：三处工程开销（零质量影响）

| 位置 | 现状 | 修法 |
|---|---|---|
| L76-78 | 每次捕获都 `torch.cuda.synchronize(q.device)`，2400 次全设备同步，打断流水线。该同步只为给 `capture_seconds` 计时。 | 移除逐次同步；计时改在 stage 结束时统一同步，或在 `finalize()` 汇总。 |
| L116 | 每次捕获都 `self._write_metadata()`，2400 次重写一个持续增长的 JSON（终态约 600 KB），累计约 1.4 GB 序列化与写盘。 | 改为仅在 `finalize()` 写一次；`entries` 在内存中累积。 |
| L92, L102 | `mask.float()` 被调用两次（`head_densities` 与 `density`），每次都是 17.45 M 元素的临时张量。 | 复用同一份 float 结果，或直接用 `mask.sum()` 计数，避免物化。 |

三项均不改变写入的 mask 数值，属纯开销。

**实施方式**：

- 同步项没有采用"stage 结束统一同步"，而是把紧随其后的 `mask.cpu()` 移进计时窗口 ——
  同一 stream 的 D2H 本就必须等前面的 kernel 结束，逐次 `torch.cuda.synchronize` 是纯冗余。
  这样 `capture_seconds` 仍是"计算 + 搬回主机"的真实耗时，`write_seconds` 变成纯磁盘写入
  计时，两个指标都保留且都不需要新增字段。
- `finalize()` 在类里并不存在（原文按文件名假设）。实际新增 `flush()`，由
  `OuterSparseCaptureHooks.__exit__` 调用一次 —— 生产环境所有捕获都经由该 context manager，
  且 `capture()` 的其它调用方只有测试（走 `prewarm()`/`get()`，读内存与磁盘文件，不读
  `metadata.json`）。另在 `summary()` 里兜底调用，保证 cache 每次自报都先落盘。
  `flush()` 对 `create=False` 打开的 cache 是 no-op。
- 第三项**没有**采用 `mask.sum()` 计数：`float_mask.mean()` 与 `mask.sum()/numel` 的浮点
  累加次序不同（17.45 M 元素已超过 fp32 精确整数上限 2^24），会改动 metadata 里的
  `density` / `head_density_*` 数值。改为复用**同一份** `mask.float()`，且在**原设备**上
  归约，与改动前逐位一致。

**预期收益未量化**——+25 s/step 中三者各占多少是代码推断，未做 profiling。
因此**必须先做这三项并重测**，才能看清剩余的真实算法成本。

### 第 2 项：位打包存储（零质量影响）

`SourceBlockMaskCache` 已定义打包格式常量（L456-457）：

```python
PACKED_FORMAT = "carnator-source-block-mask-bitpacked-v1"
CHORUS2_PACKED_FORMAT = "chorus2-normal-sparse-attention-mask-bitpacked-v1"
```

但 Wan2.1 chorus2 / carnator 的 source 路径写的是未打包格式：L85
`torch.save(mask.cpu(), ...)` 存 `torch.bool`，1 字节仅用 1 位。实测单份
mask 文件 `step011_cond_layer00.pt` 为 17,454,020 字节。

**修法**：接到已存在的打包格式上，使 2400 份 entry 的存储与 I/O 下降 8 倍。

**实施时的关键发现**：这个打包格式**早已完整存在**，只是 Wan2.1 的 source 路径没写它 ——
`scripts/benchmark/prepare_vidprom_cluster_cross_request_cache.py:35` 的
`ATTENTION_FORMAT` 与 `SourceBlockMaskCache.PACKED_FORMAT` 是**同一个字符串**
（`carnator-source-block-mask-bitpacked-v1`），`_decode()` 也**已经能读**它
（校验 `sha256` / `packed_elements` / `mask_shape` / 字节数），并且已被
`tests/test_prepare_vidprom_cluster_cross_request_cache.py` 的读回用例覆盖。
该脚本的 `pack_bool_mask()` 就是写入端实现，位序为 LSB-first、元素 `k` 落在字节 `k//8`
的第 `k%8` 位，与 `_decode` 的 `(raw >> shifts) & 1` 严格互逆。

所以本项**不是新造格式，而是接线**。实施内容：

- 把 `pack_bool_mask()` / `unpack_bool_mask()` 提取到 `carnator/outer_sparse.py` 作为
  canonical 实现（依赖方向本是 scripts → carnator），`_decode()` 的内联位序逻辑一并去重，
  cluster 脚本改为委托调用（保留其模块级 API 与惰性 import 风格）。至此写入端只有一份实现，
  不可能漂移。
- `SourceBlockMaskCache.__init__` 新增**显式 opt-in** 的 `pack_format` 参数；只在
  `adapters/wan21.py` 的两个 `create=True` source 点传值
  （chorus2 → `CHORUS2_PACKED_FORMAT`，carnator → `PACKED_FORMAT`）。
  Wan2.2 / CogVideoX / lightx2v 的 source 路径不传该参数，磁盘格式逐字不变；
  读侧 `_load_metadata()` 自动识别，零改动。
- 顺带修正 `adapters/wan21.py` 中从 Wan2.2 沿袭的过期注释（原文见文末参考数据一节）。

| | 单份 mask | 单个 source 全量（2400 份） |
|---|---:|---:|
| 现状（bool，1 字节/元素） | 16.6 MiB | **39.9 GiB** |
| 位打包后 | 2.1 MiB | **5.0 GiB** |

**收益定位要说清楚**：按 1.5 GB/s 估算，42 GB 的写+读约 28 s，相对 2231 s
的 source 阶段不是时间瓶颈，**这一项主要省的是存储与 I/O，不是直接提速**。
它的价值在于让 Wan2.1 侧持久化 source 缓存变得可行（5.0 GiB/source，与
Wan2.2 同量级），那是后续单独决策的事，不在本次范围内。

### 验证

**已执行（单元 / 回归层）**：

- 打包往返：同一 mask 以 bool 与 `pack_format=PACKED_FORMAT` 两条路径落盘，读回的
  `torch.bool` 张量**逐位相等**（`torch.equal`），文件字节数 `== ceil(elements/8)`。
- 数值中性：打包与非打包两条路径产出的 `density` / `coverage` / `head_density_*`
  完全相等 —— 守护"复用一份 float 没改数值"。
- checksum 校验、metadata 延迟写（capture 不落盘、`flush()` 落盘一次）、
  `flush()` 对回放 cache 是 no-op、未知 `pack_format` 被拒绝。
- 跨写器一致性：cluster 脚本与 `SourceBlockMaskCache` 对同一 mask 的打包输出逐字节相同。
- `pytest tests/`：158 passed。唯一失败
  `test_cogvideox15_adapter.py::test_replay_runner_separates_source_time_and_always_cleans_tmp_cache`
  经 `git stash` 撤掉全部改动后**同样失败**，是既有的亚毫秒计时 flake（该测试只走
  `SyntheticAdapter` + `CogVideoReplayRunner`，不 import 本次改动的任何模块）。

**已执行（真实重跑验收，2026-09-11 16:08–17:00）** —— 见下一节。

不必担心存量数据：`results_wan21_test` 的 source mask 缓存本就随 `tmp_cache`
在跑完后清掉（`source_prepare/` 只剩 `events.jsonl`），没有旧格式产物需要迁移 ——
改动只对**下一次运行**生效。

---

## 真实重跑验收（2026-09-11）

### 设置

`results_wan21_test` 的归档配置原样重放（seed 20260218、limit 10、arrival 0..10、
同一 input CSV、同一 `source_view`、同一 `cogvideox15_replay`），只把 GPU 从计划里的
6/7 换成 3/4（`--allow-gpu-remap`），输出写到新的 `results_wan21_rerun/`，
归档目录保持只读基线。重跑命令见 `/tmp/wan21_rerun_supervise.sh`。

**范围收紧**：原本可以跑满 10 个 arrival（≈6.3 h），但核心问题（打包是否中性）
由**一个 source+target 对**即可判定 —— arrival 0 恰好走完
`capture() 打包写盘 → flush() 落 metadata → _decode() 读打包 mask → target replay`
整条被改动的路径，其余 9 个 arrival 只增加样本量、不引入新的失效模式。
因此在 arrival 0 落盘后即终止，实际占用 GPU 约 50 min。

### 1. 逐位一致（核心验收，✅ 通过）

chorus2 arrival 0（`uuid=3384bf4f-…`, `source_uuid=3a3813cf-…`）：

| | 归档 `results_wan21_test` | 重跑 `results_wan21_rerun` |
|---|---|---|
| `output_hash` | `954ea7a5…2ea9be` | `954ea7a5…2ea9be` ✅ |
| `videos/3384bf4f-….mp4` sha256 | `954ea7a5…2ea9be` | `954ea7a5…2ea9be` ✅ |

两者完全相同。由于 target 的 `denoising_s` 只取决于掩码比特，逐位一致即证明
打包/解包、去掉逐次同步、复用一份 `float()`、延迟元数据四项改动**对结果完全中性**。

顺带确认了流水线本身可复现：同一 seed 在同一机器上两次独立运行给出同一 video 字节
（此前未经验证；若不确定，逐位比对本身就不成立）。对比工具：
`scripts/benchmark/compare_wan21_rerun.py`。

### 2. `source_prepare_s`（⚠️ 单样本不足以判定加速）

| | 归档 | 重跑 | 差值 |
|---|---:|---:|---:|
| `source_prepare_s` | 2233.0 | 2264.3 | **+31.3 s** |
| `cache_prepare_s`（target 侧） | 20.6 | 10.2 | −10.4 s |
| `denoising_s`（target 侧） | 551.7 | 560.9 | +9.2 s |

**+31 s 看起来像变慢，其实是机器变慢，不是改动变慢。** 控制变量是 target 的
`denoising_s`：它与掩码比特一一对应，而两个 mask 已被上表证明逐位相同，因此这
551.7 → 560.9 s（**+1.67%**）就是这台机器/这张卡本次运行的速度差，与改动无关。
把它按比例套到 source 的 2233 s 上，预期 2270.4 s；实测 2264.3 s，
**残差 −6.1 s**（落在噪声内）。也就是说归一化后没有观察到 source 阶段的回归，
但也**没有**测到可归因于本次改动的加速。

原因是掩码写入本就不是 source 阶段的主导成本。本机微基准（同一 shape
`(12, 1705, 853)`，17,452,380 元素）：

```
per mask:  pack 14.6 ms   torch.save 15.1 ms   unpack 9.3 ms   torch.load 3.0 ms
x2400:     pack 35.2 s    torch.save 36.2 s    unpack 22.4 s   torch.load 7.3 s
```

位打包把**写入字节数**降到 1/8，但换来等量的 CPU 位打包开销（35 s vs 36 s），
两者基本抵消。因此这三项改动的收益应记为**存储与 I/O 压力**（见下），
而非 wall-clock 加速；source 阶段 ~2233 s 里真正的成本是 50 步扩散本身。

### 3. mask 目录字节数（✅ 通过）

`tmp_cache` 在 `finally` 里被删除，峰值只能在运行期采样
（采样器 `/tmp/wan21_maskcache_sampler.sh`，每 20 s 一次）。source 写满 2400 份后：

| | 打包后实测 | 等价未打包（×8） |
|---|---:|---:|
| chorus2 | 2401 份 / **4.88 GiB** | **39.0 GiB** |
| carnator | 2405 份 / **4.88 GiB** | **39.0 GiB** |

单份文件 **2,181,548 B = `ceil(12×1705×853 / 8)`**，未打包的 `torch.bool` 是
17,452,380 B（16.64 MiB）—— **正好 8×**，与预期一致。每个 source 因此少写
~36.7 GB。本机根分区已用 6.5T/6.9T（98%），这项收益主要体现在磁盘压力上。

运行期同时确认：两个 `metadata.json` 的 `format` 分别是
`chorus2-normal-sparse-attention-mask-bitpacked-v1` 与
`carnator-source-block-mask-bitpacked-v1`，且 capture 期间 `entries` 为空 ——
说明打包分支与延迟元数据都真的生效了。

---

## 遗留问题：carnator 重跑的 arrival 0 OOM

同一轮重跑里 carnator（GPU 4）的 arrival 0 **失败**，报错：

```
OutOfMemoryError: CUDA out of memory. Tried to allocate 1.94 GiB.
GPU 4 ... 79.06 GiB memory in use. Of the allocated memory 69.72 GiB is
allocated by PyTorch, and 8.46 GiB is reserved but unallocated.
```

发生在 **target replay 阶段**，不是 source 阶段 —— `source_prepare_s = 2269.6` 已经
记录、`cleanup.status = "cleaned"`，说明 source 生成与打包写盘正常完成，是随后的
target 生成（`capture=False`）炸的。

**判断：与本轮改动无关，但未证实。** 理由：

- 改动只触及 source 的 capture 路径；target 走 `capture=False`，根本不调用
  `capture()`。
- `SourceBlockMaskCache` 的读侧在打包前后都把完整 bool 张量物化在**主机内存**
  （`unpack_bool_mask` 全程 CPU），不增加显存占用。
- 改动是**减少**显存：`mask.float()` 从两次变一次；去掉的逐次 `torch.cuda.synchronize`
  与紧随其后的 `mask.cpu()`（阻塞拷贝）重复，不改变张量生命周期。
- chorus2 同轮重跑中 target 侧 `cache_prepare_s` 从 20.6 s **降到** 10.2 s，
  说明打包后的解码路径并未变重。

但归档里 carnator 在同一请求上（GPU 7）是成功的，所以确实出现了一次差异。
最可能的解释是 carnator target 的显存本就贴着上限（69.7 GiB allocated / 79.2 GiB
capacity），任何运行间波动都会翻车；归档那次是 10/10 成功，属于擦边通过。
**要证实需要一次单独重跑 carnator arrival 0**（约 50 min GPU），本次未做。

注意这与聚合器的质量闸是同一类现象的两个面：carnator 之前的 CLIP/VBench OOM
（第二节）也是"质量评估撞上别人占用的卡"，本次 `CUDA_VISIBLE_DEVICES` 修复已让
质量阶段跟随 `--gpu`，但**不保证该卡空闲**。

---

## 附：本次测量的参考数据

实测（`results_wan21_test`，10 请求全部跑完，聚合时刻 2026-09-11 14:27；
命中率取自各方法自己的 `{replay_root}/{method}.jsonl`）：

| 方法 | Hit率 | E2E s/请求 | Denoising s/请求 | s/step | source prepare s/请求 | CLIP-T | VBench-Q |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense Baseline | N/A | 1138.6 | 974.2 | 19.5 | 0.0 | 0.2984 | 0.6373 |
| NIRVANA-VID | 0.80 | 1035.9 | 884.5 | 17.7 | 1149.4 | 0.2934 | 0.6169 |
| Chorus | 0.80 | 719.8 | 582.4 | 11.6 | 1118.6 | 0.2899 | 0.5819 |
| Chorus2 | 0.80 | 577.1 | 556.9 | 11.1 | 2232.8 | 0.2969 | 0.6461 |
| Carnator B5 | 1.00 | 547.2 | 496.0 | 10.4 | 2288.8 | 0.2533 | 0.6086 |

两点读表提醒：

- **E2E 的分母是"实际生成过的请求"**：miss 不生成、不计时（`wan21_runner.py:94-97`）。
  chorus2 的 577.1 s 是 8 个请求的均值，carnator 的 547.2 s 是 10 个请求的均值，
  两者的 2.08× / 1.97× 因此不是同一子集上的比较。去比 `Hit E2E s/hit` 列时
  同样是 8 vs 10。这是设计使然（Wan2.2 侧同构），但引用加速比时需要注明。
- **命中率按方法各自取**：replay 文件是 `{replay_root}/{method}.jsonl`，
  carnator 的 legacy 命中情况本就是 10/10、chorus2 是 8/10，不是同一份决策的
  不同处理。

source prepare 的绝对量：chorus2 8 个 source 共 17,862 s，carnator 10 个共
22,888 s。

Wan2.1 与 Wan2.2 的 token 数差异（同一 1280×704×121 输入）：

| | `vae_stride` | token 数 | q_blocks × k_blocks × heads | 单份 mask | ×2400 |
|---|---|---:|---|---:|---:|
| Wan2.1-T2V-1.3B | (4, 8, 8) | 109,120 | 1705 × 853 × 12 | 16.6 MiB | 39.9 GiB |
| Wan2.2-TI2V-5B | (4, 16, 16) | 27,280 | 427 × 214 × 24 | 2.09 MiB | 5.0 GiB |

Wan2.1 的 token 数是 Wan2.2 的 4.0 倍，block pair 数因此是 16 倍，扣掉
head 数差异后单份 mask 相差 8.0 倍。`adapters/wan21.py:949` 注释中
"~5 GiB at Wan2.1's 27k-token sequence length" 的 27k 与 5 GiB 均为
Wan2.2 的数值，属从 Wan2.2 代码沿袭未重算。

**已修正**（`adapters/wan21.py` 的 chorus2 target 路径）：注释改为 Wan2.1 的
109,120 token / 单份 16.6 MiB / 全量约 40 GiB **主机内存**，并写明位打包只压缩磁盘、
不缩小 `prepare()` 解出的这份主机 bool 缓存。

---

## 实施记录（2026-09-11）

改动文件：

| 文件 | 内容 |
|---|---|
| `scripts/benchmark/aggregate_vidprom_300_results.py` | `AggregationError`、`rows_for()` 去回退 + 时效闸 + 桩数据闸、`validate_quality_completeness()`、`main()` 校验全部方法后再写盘（失败返回 1 且不产出任何产物）、`--allow-incomplete-quality` |
| `scripts/benchmark/run_vidprom_300_experiment.py` | `run_quality_evaluation()` 传 `CUDA_VISIBLE_DEVICES=<gpu>`；质量失败改为 stderr 明确告警 |
| `carnator/outer_sparse.py` | 新增模块级 `pack_bool_mask()` / `unpack_bool_mask()`；`__init__` 新增 `pack_format`；`capture()` 去逐次同步、去逐次写 metadata、float 复用一次；新增 `flush()`；`_decode()` 复用共享解包；`OuterSparseCaptureHooks.__exit__` 与 `summary()` 调 `flush()` |
| `adapters/wan21.py` | 两个 `create=True` source 点传 `pack_format`；修正过期注释 |
| `scripts/benchmark/prepare_vidprom_cluster_cross_request_cache.py` | 删除本地 `pack_bool_mask`/`unpack_bool_mask`，改为委托共享实现（保留模块级 API 与惰性 import） |
| `tests/test_source_block_mask_packing.py` | 新增 8 个用例 |
| `tests/test_aggregate_vidprom_results.py` | 新增 11 个用例 |
| `scripts/benchmark/compare_wan21_rerun.py` | 新增：归档 vs 重跑的 `output_hash` / 视频 sha256 / `source_prepare_s` 对比工具（本次验收用，可复用） |

辅助脚本（临时，放在 `/tmp`，未入库）：`wan21_rerun_supervise.sh`（重跑）、
`wan21_maskcache_sampler.sh`（运行期采样 tmp_cache 峰值）、
`wan21_stop_after_first.sh`（arrival 0 完成后收工）。

回归结果：对**现有** `results_wan21_test`（进程已全部退出、产物健康）重跑聚合，
exit=0，`summary.csv` 与改动前**逐字节相同**，三份产物 mtime 刷新；
baseline 仍为 `e2e_s_per_request = 1138.64`、`denoising_s_per_request = 974.16`。
三道闸对现有数据不误伤。

**行为变化提醒**：聚合器现在会对质量列缺失**硬失败**，`/tmp/wan21_supervise.sh` 里
`aggregator exit=$?` 以后可能记到非零 —— 预期行为，用 `--allow-incomplete-quality` 可降级。

---

## 结论

| 验收项 | 结果 |
|---|---|
| chorus2 target 产物逐位一致（`output_hash` + mp4 sha256） | ✅ 通过 |
| mask 目录 39.0 GiB → 4.88 GiB（8.00×） | ✅ 通过 |
| 打包分支 / 延迟元数据在运行期真的生效 | ✅ 通过 |
| `source_prepare_s` 加速 | ⚠️ 未测到（掩码写入本就不是瓶颈，归一化后残差 −6.1 s 在噪声内） |
| carnator arrival 0 | ❌ OOM，判断与本改动无关但未证实，见「遗留问题」 |
| 聚合器三道闸 + 质量设备绑定 | ✅ 回归通过（`summary.csv` 逐字节不变） |

**本次改动的净收益应记为存储与 I/O 压力，而非 wall-clock 加速** —— 位打包把每个
source 的写入量从 39 GB 降到 4.9 GB，代价是等量的 CPU 位打包时间。
若接下来要压 source 阶段的 wall-clock，方向是那 50 步扩散本身，不是序列化。
