# VidProM 前 1000 条共享缓存准备

新增 `scripts/benchmark/prepare_vidprom_cluster_cross_request_cache.py`，按 CSV 原始顺序处理
`/home/tangxuebin/project/VidProM/clusters/cluster_02_2170.csv` 的前 1000 条。每条请求只运行一次
50-step Wan source 去噪，不执行 retrieval 或 target 生成；产物供 NIRVANA-VID、Chorus、Chorus2、
Carnator 共同只读，禁止复制到各方法的私有 cache namespace。

`step s` 统一为 completed/post-scheduler latent。保存完整 Carnator/Chorus trajectory：
`latents_initial.pt` 和 `latents_step_000..049.pt`；attention 的 `from_step_010` 表示从
`after_step_010` 到 `after_step_011` 的 forward，范围 `010..049`。

```text
artifacts/shared_cross_request_cache/vidprom_cluster_02_2170_first1000/
  entries/<uuid>/
    manifest.json
    source.mp4
    carnator/cache/source_dense/
      manifest.json
      latents_initial.pt
      latents_step_000.pt ... latents_step_049.pt
      cross_seed_scores/ self_contrast_masks/ fusion_masks/
      outer_block_masks/metadata.json
      outer_block_masks/maskbits/from_step_010/{cond,uncond}/layer_00.bin ...
    status.json
  index.jsonl
  run_manifest.json
  workers/gpu-<id>.jsonl
```

- NIRVANA-VID 直接读取 canonical `latents_step_009.pt`；Chorus 直接读取同一份完整
  51-state native trajectory 和 `source.mp4`；Chorus2 直接读取同一份
  `outer_block_masks`；Carnator 以 `entries/<uuid>/carnator` 作为 `--source-run-dir`，
  从 native `cache/source_dense` 读取 trajectory、fusion mask 和 raw block mask。四者没有
  私有 cache 目录，也不创建硬链接或副本。
- 不为 CHAI 导出或保留 source token cache。
- `cross_seed_scores` 只保存用于 self-attention 的 cross-attention score；不包含 Mbase、Medit 或 Msee。
  Carnator 的 completed-step-5 Early Signature 属于 target 在线流程，不预采集、不保存。
- Carnator 保留完整 30-layer 的 bit-packed raw `bool[heads,q_blocks,k_blocks]`，覆盖
  `cond`/`uncond`。Chorus2 使用这同一份完整的普通 sparse attention mask；尤其 10--20 层
  保持未筛选的 raw mask。Carnator 仅在 target 端将其与 `Medit`/`Msee` 相交。metadata 固化
  geometry、block size、mask 参数和 SHA-256；reader 解包后再创建对应 backend 的运行时 view。

## 并行执行

默认 `--gpus 3,4,5,6,7`：五个常驻单卡 worker 各加载一次模型，从动态队列取任务；请求独立，
不使用 DDP。每条请求先写入 `entries/.staging/`，文件/hash 完整后原子 rename 提交。每卡写独立
JSONL，控制进程最终按 CSV 顺序合并索引。`--resume` 仅跳过 manifest、文件及 checksum 均有效的 entry。

mask 在每步立即转 CPU 并交给有界写盘队列，避免累积 GPU tensor；第 10 步后继续 dense source 至第
50 步，但不再跑第二次 FlashInfer/SpargeAttn pass。

实施前先运行无 GPU 的 `--estimate-only`。它按 Wan 的 token geometry、完整 30-layer raw mask、
共享的 10--20 层 raw mask 及实测的 compatible latent/video 文件大小估算全量空间，并要求可用空间至少
为估算的 `1.2x`。先验证 reader 的路径、shape、hash 和 raw-mask 语义，再启动全量。source Dense、
mask 计算和所有落盘归入 `cache_preparation`，不计入任何 target 端到端时延。

```bash
conda run -n wan env PYTHONPATH=. python scripts/benchmark/prepare_vidprom_cluster_cross_request_cache.py \
  --input /home/tangxuebin/project/VidProM/clusters/cluster_02_2170.csv \
  --limit 1000 --gpus 3,4,5,6,7 --estimate-only \
  --output artifacts/shared_cross_request_cache/vidprom_cluster_02_2170_first1000
```

空间预估通过且确认执行后，移除 `--estimate-only` 并增加 `--confirm-full-run` 启动全量。
