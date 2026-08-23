# VidProM 前 1000 条共享缓存准备

新增 `scripts/benchmark/prepare_vidprom_cluster_cross_request_cache.py`，按 CSV 原始顺序处理
`/home/tangxuebin/project/VidProM/clusters/cluster_02_2170.csv` 的前 1000 条。每条请求只运行一次
50-step Wan source 去噪，不执行 retrieval 或 target 生成；产物供 NIRVANA-VID、Chorus、Chorus2、
Carnator 共同只读，禁止复制到各方法的私有 cache namespace。

`step s` 统一为 completed/post-scheduler latent。保存 `after_step_010..050`；attention 的
`from_step_010` 表示从 `after_step_010` 到 `after_step_011` 的 forward，范围 `010..049`。

```text
artifacts/shared_cross_request_cache/vidprom_cluster_02_2170_first1000/
  entries/<uuid>/
    manifest.json
    source.mp4
    latents/after_step_010.safetensors ... after_step_050.safetensors
    attention/metadata.json
    attention/maskbits/from_step_010/{cond,uncond}/layer_00.bin ...
    status.json
  index.jsonl
  run_manifest.json
  workers/gpu-<id>.jsonl
```

- NIRVANA-VID/Carnator 读取 `after_step_010`；Chorus 读取 `10..40` 和唯一的 `source.mp4`。
- Carnator 的 completed-step-5 Early Signature 属于 target 在线流程，不预采集、不保存。
- attention 只保存一份 bit-packed canonical `bool[heads,q_blocks,k_blocks]`，覆盖实际稀疏 layer 和
  `cond`/`uncond`。metadata 固化 geometry、block size、mask 参数、配置 fingerprint 和 SHA-256。
  FlashInfer 解包后创建 plan，SpargeAttn 解包为 `mask_id`；不保存两份 mask 或任何运行时 wrapper。

## 八卡执行

默认 `--gpus 0,1,2,3,4,5,6,7`：八个常驻单卡 worker 各加载一次模型，从动态队列取任务；请求独立，
不使用 DDP。每条请求先写入 `entries/.staging/`，文件/hash 完整后原子 rename 提交。每卡写独立
JSONL，控制进程最终按 CSV 顺序合并索引。`--resume` 仅跳过 manifest、文件及 checksum 均有效的 entry。

mask 在每步立即转 CPU 并交给有界写盘队列，避免累积 GPU tensor；第 10 步后继续 dense source 至第
50 步，但不再跑第二次 FlashInfer/SpargeAttn pass。

实施前先完成一条真实 pilot，按实际文件大小估算全量空间，并要求可用空间至少为估算的 `1.2x`；当前
工作盘约剩 2.0 TiB。先验证四个 reader 的路径、shape、hash 和 FlashInfer/SpargeAttn mask parity，
再启动全量。

```bash
conda run -n wan env PYTHONPATH=. python scripts/benchmark/prepare_vidprom_cluster_cross_request_cache.py \
  --input /home/tangxuebin/project/VidProM/clusters/cluster_02_2170.csv \
  --limit 1000 --gpus 0,1,2,3,4,5,6,7 --pilot \
  --output artifacts/shared_cross_request_cache/vidprom_cluster_02_2170_first1000
```

pilot 通过并确认空间后，再移除 `--pilot` 并增加 `--confirm-full-run` 启动全量。
