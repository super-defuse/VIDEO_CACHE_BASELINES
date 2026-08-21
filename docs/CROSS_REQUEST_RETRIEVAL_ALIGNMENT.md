# 跨请求检索对齐说明

## 范围与验证

工作负载为 `datasets/vbench_cross_request_prompts.csv`，由八个 VBench 维度的 694 条非空提示词组成，已排除 `multiple_objects`、`scene` 和 `spatial_relationship`。CSV 包含 runner 所需的 `request_index`、`uuid` 和
`prompt` 字段，并增加 `dimension` 供统计使用。

`tests/test_cross_request_retrieval.py` 通过 synthetic adapter 对 NIRVANA-VID、CHAI 和 Chorus 依次运行前 18 条请求，验证冷启动未命中、后续相似命中、latent 加载、快速生成，以及重建 Runner 后持久化索引仍可复用。它还分别验证 CHAI 的 FAISS 精确内积查询、Chorus 的 ChromaDB 重启复用和 NIRVANA CacheSelector profile 的离线推导。
synthetic 路径始终使用确定性的词法编码器和内存向量查询，仅用于检验协议和推理接线；真实检索后端不会在 dry-run 中被静默替换使用。

## 已实现的对齐

| 方法 | 检索与缓存行为 |
| --- | --- |
| NIRVANA-VID | 使用整条 prompt 的 CLIP embedding、余弦相似度 Top-1 查询，并由离线 CacheSelector profile 选择满足条件的最大 restart step。source 生成保存第 5/10/15/20/25 步的 latent；命中后将选定 tensor 交给原生 restart 路径。提供了从 `similarity,reuse_step,quality` 观测生成 profile 的脚本。缓存淘汰采用 LCBFU 形式：优先淘汰 `hit_count * restart_step` 较小的条目。 |
| CHAI | 使用 spaCy 将名词短语和名词提取为对象、场景实体；每个实体由 LongCLIP 编码，并索引到 source latent UUID。真实运行使用 FAISS 1.13 的 `IndexFlatIP` 对归一化向量做精确余弦 Top-1 查询；余弦分数最高的实体匹配会提供已有的 Cache Attention payload，淘汰策略保持 LRU。 |
| Chorus | 使用整条 prompt 的 768 维 CLIP embedding 和余弦相似度 Top-1 查询；50-step 模型采用论文给出的阈值 `tau=0.65`。真实运行使用 namespace 下的 ChromaDB 持久化 cosine collection。命中后将匹配到的 trajectory 和 source video sidecar 交给已有的 Chorus 三阶段 adapter。分数会驱动 `K1/K2`，因此普通工作负载请求无需手工提供 `cache_prompt` 也能进入 target 路径。 |
| Carnator | 使用整条 prompt 的归一化 CLIP 精确内积 Top-K（默认 20）；按本地 `trf_nouns` 做相同/单 noun substitution 预过滤，再按召回顺序逐一比较 step-5 ETS signature 与冻结阈值，并验证精确 step-10 latent 和 sparse-attention manifest。只有最终 joint reuse 才报告 `hit=true`；淘汰或 artifact 失效时同步移除 vector record。 |

四种方法都将 `retrieval.json` 作为可恢复的向量与 UUID 映射清单，在缓存载荷被淘汰时同步删除记录。CHAI 的 FAISS 索引由此清单重建；Chorus 的 ChromaDB 数据存放在 namespace 的 `vector_store/chromadb` 下并同步更新。`results.jsonl` 的 `cache_match` 会记录 retriever、匹配词项、分数及方法特定的调度元数据。

## Carnator 检索设计

Carnator 已接入本仓库独立实现的两阶段检索：整条 prompt 使用归一化 CLIP 的精确内积 Top-K
召回，随后用本地 `trf_nouns` 做相同/单 noun substitution 预过滤，再以 completed step 5
原生 fusion mask 构造的 prompt ETS、冻结且认证的阈值配置和精确 step-10/sparse-attention
artifact manifest 作唯一复用决策。`scripts/run/carnator_dataset_ets.py` 可在一个原生 Wan
进程中为整个数据集采集真实 ETS JSONL，`scripts/run/cross_request_dataset.py` 用它回放四种方法
的 retrieval-only dataset 模式。`carnator/retrieval.py` 与
[`CARNATOR_CROSS_REQUEST_RETRIEVAL_DESIGN.md`](CARNATOR_CROSS_REQUEST_RETRIEVAL_DESIGN.md)
共同定义协议；`scripts/run/carnator_retrieval_only.py` 仍提供使用 fixture 的单元验收路径。

## 真实运行前提

NIRVANA-VID 和 Chorus 通过 `transformers` 使用
`openai/clip-vit-large-patch14`，其投影后的文本特征为 768 维。CHAI 还需要
`requirements/retrieval-common.txt` 中的 FAISS 1.13、`en_core_web_sm` spaCy 模型，以及官方 LongCLIP
Python 模块和 checkpoint。Wan 的 Chorus 额外使用 `requirements/retrieval-wan.txt` 中的 ChromaDB；
CogVideoX1.5 不支持 Chorus，因此不安装 ChromaDB。通过 `retrieval.chai.longclip_checkpoint` 配置
LongCLIP checkpoint；NIRVANA 的 profile 路径会相对仓库根目录解析。真实运行缺少这些论文依赖时会明确失败，不会静默回退到测试编码器。

FAISS 1.13 使用 NumPy 2 的 `_core` 布局，因此 `wan`、`cogvideo` 固定使用
`numpy==2.2.6`；`vbench` 评测环境仍单独使用 NumPy 1.26，不要混用三个环境的 Python。

从仓库根目录执行 `scripts/create_environments.sh` 会创建或修复三个 Conda 环境并安装模型与检索依赖。
脚本发现错误的 venv 或跨环境 Python 软链接时，会保留 `.invalid.<timestamp>` 备份后重建；不要用
`pip install --user` 替代环境内安装。安装完成后可运行：

脚本默认使用 CUDA 12.8 的 `torch==2.8.0+cu128` 与 `torchvision==0.23.0+cu128`，
因为 Wan 依赖固定了 `triton<3.5`。更换 CUDA wheel index 时，也必须显式指定同一 index
中已验证且与 Triton 匹配的 PyTorch/torchvision 版本。

```bash
conda run -n wan env PYTHONNOUSERSITE=1 python scripts/check_retrieval_environment.py --runtime wan
conda run -n cogvideo env PYTHONNOUSERSITE=1 python scripts/check_retrieval_environment.py --runtime cogvideo
```

当前 `wan` 环境已用 Wan2.2-TI2V-5B 完成最小真实验证（256×256、17 帧、2 步）：
FlashAttention CUDA 调用、Runner 视频落盘，以及 Baseline、TeaCache、NIRVANA-VID、CHAI、
Chorus 的源/目标接线均通过；该验证不替代完整实验。

NIRVANA profile 的默认文件是 `configs/retrieval/nirvana_cache_selector_profile.json`，其 `source` 明确标记为未校准。获得真实质量观测后可运行：

```bash
python scripts/benchmark/nirvana_cache_selector.py \
  --observations quality_observations.csv \
  --output configs/retrieval/nirvana_cache_selector_profile.json
```

## 尚未完全对齐的部分

| 方法 | 论文未充分披露、因而无法完全复现的细节 |
| --- | --- |
| NIRVANA-VID | 原始 CacheSelector 的质量 profile 曲线、profile 所用精确质量采样协议和 LCBFU 评分函数没有公开。默认 profile 与 `hit_count * restart_step` 淘汰效用均明确为本地近似，不应视为论文常量。 |
| CHAI | 已接入论文指定的 FAISS 1.13，但论文没有公开具体索引类型和实体命中阈值；这里以归一化向量的精确 `IndexFlatIP` 和既有的 `min_entity_similarity=0.55` 实现。生成部分适配的是 Wan2.2/CogVideoX，而非论文的 OpenSora 1.2 backbone。 |
| Chorus | 已接入论文指定的 ChromaDB，但其版本、collection/HNSW 参数未披露。用于计算 `K1/K2` 的单调函数 `F(score, N)` 以及精确的 TGAA 函数也未披露；当前配置的线性阶段调度与已有的线性 TGAA 衰减仍明确标记为近似。 |

该 CSV 有意作为 VBench 工作负载，而非论文使用的 VidProM 请求流，因此命中率、质量和时延数值不能与论文报告结果直接比较。
