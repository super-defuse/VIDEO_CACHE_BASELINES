# CogVideoX 5-Baseline 10-Sample Experiment

## 概述

这个实验在 CogVideoX-1.5 模型上运行 5 个基线方法，每个方法处理前 10 个样本：

1. **baseline** - Dense baseline（无缓存，直接生成）
2. **carnator_boundary_b5** - Carnator 方法（边界缓存）
3. **chorus** - Chorus 方法
4. **chorus2** - Chorus2 方法
5. **nirvana_vid** - NIRVANA-VID 方法

## 前置条件

1. Replay decision 文件已生成：
   - `artifacts/vidprom_1000cache_300input_noun_match/cogvideox15_replay/*.jsonl`
   - `artifacts/vidprom_1000cache_300input_noun_match/cogvideox15_replay/manifest.json`

2. GPU 资源：
   - 需要 5 个 GPU（默认使用 GPU 3-7）
   - 每个 GPU 约需要 20-30GB 显存

3. Python 环境：
   - VIDEO_CACHE_BASELINES 环境已配置
   - 依赖包已安装

## 文件说明

### 主要脚本

- **`run_5baselines_10samples.py`** - 主实验脚本（Python）
  - 支持并行或顺序执行
  - 支持选择特定基线
  - 支持 dry-run 模式

- **`launch_5baselines_10samples.sh`** - 快速启动脚本（Bash）
  - 使用默认配置启动实验
  - 5 个基线并行运行在 GPU 3-7

- **`monitor_5baselines.sh`** - 监控脚本
  - 显示实验进度
  - 显示 GPU 使用情况
  - 检查错误日志

- **`test_5baselines_config.sh`** - 配置测试脚本
  - Dry-run 模式验证配置
  - 不实际运行实验

- **`clean_failed_10samples.sh`** - 清理脚本
  - 删除之前失败的实验数据

## 使用方法

### 1. 测试配置（推荐第一步）

```bash
cd /home/tangxuebin/project/VIDEO_CACHE_BASELINES
./scripts/dev/test_5baselines_config.sh
```

这会验证：
- Replay decision 文件存在
- Manifest 文件存在
- GPU 分配正确
- 不会实际运行实验

### 2. 清理旧的失败实验（可选）

```bash
./scripts/dev/clean_failed_10samples.sh
```

会提示确认是否删除 `artifacts/cogvideox15_10samples_test/` 目录。

### 3. 启动实验

#### 方式 A：使用快速启动脚本（推荐）

```bash
./scripts/dev/launch_5baselines_10samples.sh
```

#### 方式 B：使用 Python 脚本（更多选项）

```bash
# 所有基线，并行运行
python3 scripts/dev/run_5baselines_10samples_fixed.py \
    --gpus 3,4,5,6,7 \
    --limit 10

# 只运行特定基线
python3 scripts/dev/run_5baselines_10samples_fixed.py \
    --baselines baseline chorus \
    --gpus 3,4 \
    --limit 10

# 顺序运行（单 GPU）
python3 scripts/dev/run_5baselines_10samples_fixed.py \
    --sequential \
    --gpus 3 \
    --limit 10
```

### 4. 监控实验进度

```bash
# 单次检查
./scripts/dev/monitor_5baselines.sh

# 持续监控（每 5 秒刷新）
watch -n 5 ./scripts/dev/monitor_5baselines.sh
```

监控脚本会显示：
- 运行中的进程
- GPU 3-7 的使用情况
- 每个基线的完成进度
- 最近的错误（如果有）

### 5. 检查结果

实验完成后，结果保存在：

```
artifacts/cogvideox15_5baselines_10samples/
├── baseline_20260909_123456_gpu3/
│   ├── results.jsonl          # 结果记录
│   ├── videos/                 # 生成的视频
│   └── ...
├── carnator_boundary_b5_20260909_123456_gpu4/
├── chorus_20260909_123456_gpu5/
├── chorus2_20260909_123456_gpu6/
├── nirvana_vid_20260909_123456_gpu7/
├── baseline_gpu3.log           # 运行日志
├── carnator_boundary_b5_gpu4.log
├── chorus_gpu5.log
├── chorus2_gpu6.log
├── nirvana_vid_gpu7.log
└── launch.log                  # 总启动日志
```

每个实验目录包含：
- `results.jsonl` - 每个请求的详细结果（耗时、状态等）
- `videos/` - 生成的视频文件
- 其他方法特定的输出

## 参数说明

### `run_5baselines_10samples.py` 参数

```
--baselines BASELINE [BASELINE ...]
    要运行的基线（默认：全部 5 个）
    可选：baseline carnator_boundary_b5 chorus chorus2 nirvana_vid

--replay-dir PATH
    Replay decision 文件目录
    默认：artifacts/vidprom_1000cache_300input_noun_match/cogvideox15_replay

--manifest PATH
    Source prompts manifest 文件
    默认：artifacts/vidprom_1000cache_300input_noun_match/cogvideox15_replay/manifest.json

--output-root PATH
    输出根目录
    默认：artifacts/cogvideox15_5baselines_10samples

--gpus GPU,GPU,...
    GPU ID 列表（逗号分隔）
    默认：3,4,5,6,7

--limit N
    每个基线处理的样本数
    默认：10

--dry-run
    Dry-run 模式（不实际运行）

--sequential
    顺序运行（单 GPU），而不是并行
```

## 预期运行时间

基于 CogVideoX-1.5 的典型性能：
- Dense baseline: ~15-20 秒/样本 → 10 样本约 2.5-3 分钟
- Cache-based methods: ~8-12 秒/样本 → 10 样本约 1.5-2 分钟

并行运行全部 5 个基线：**约 2.5-3 分钟**（取决于最慢的那个）

## 故障排查

### 问题：Import 错误

```
ImportError: cannot import name 'load_method' from 'core.registry'
```

**解决方案**：确保使用新的脚本 `run_5baselines_10samples.py`，它使用正确的 `make_method` 函数。

### 问题：Replay file not found

```
FileNotFoundError: Replay file not found: .../chorus.jsonl
```

**解决方案**：
1. 检查 replay decision 文件是否存在
2. 运行 replay decision 生成脚本

### 问题：GPU out of memory

**解决方案**：
1. 减少并行运行的基线数量
2. 使用 `--sequential` 模式
3. 检查其他进程是否占用 GPU

### 问题：进程卡住

**解决方案**：
1. 检查日志文件：`artifacts/cogvideox15_5baselines_10samples/*.log`
2. 检查 GPU 状态：`nvidia-smi`
3. 如果需要，强制终止：`pkill -f run_5baselines_10samples.py`

## 下一步

实验完成后，可以运行：

1. **质量评估**
   ```bash
   python scripts/evaluate_quality.py --input artifacts/cogvideox15_5baselines_10samples
   ```

2. **指标聚合**
   ```bash
   python scripts/aggregate_metrics.py --input artifacts/cogvideox15_5baselines_10samples
   ```

3. **对比分析**
   ```bash
   python scripts/compare_baselines.py --input artifacts/cogvideox15_5baselines_10samples
   ```

## 注意事项

1. **Dense baseline 特殊性**：
   - Dense baseline（`baseline`）不使用 replay decisions
   - 直接使用 manifest 中的前 10 个 source prompts
   - 其他 4 个方法使用各自的 replay decisions

2. **GPU 分配**：
   - 默认使用 GPU 3-7（5 个 GPU）
   - 每个基线分配一个 GPU
   - 可以通过 `--gpus` 参数自定义

3. **输出目录**：
   - 每次运行创建带时间戳的新目录
   - 不会覆盖之前的实验结果
   - 便于对比不同运行

4. **日志文件**：
   - 每个基线有独立的日志文件
   - 便于调试特定基线的问题
   - 主日志 `launch.log` 记录总体情况
