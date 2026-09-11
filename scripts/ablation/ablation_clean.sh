#!/bin/bash
# 简洁的消融实验脚本 - Request 16
# Target: "A cute little lion and a wobbly penguin appear"
# 使用per-noun mask以获得正确的attention reuse效果

set -e
cd /home/tangxuebin/project/VIDEO_CACHE_BASELINES

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ARTIFACT_ROOT="artifacts/ablation_req16_${TIMESTAMP}"
TARGET_PROMPT="A cute little lion and a wobbly penguin appear"
SOURCE_PROMPT="a little dog and a penguin dancing"
SOURCE_NOUN="dog"  # 差异名词，将被替换为lion
SEED=20260311
STEPS=50

mkdir -p "$ARTIFACT_ROOT"

echo "Step 1: 生成source cache (完整运行一次生成cache)"
conda run --no-capture-output -n wan env CUDA_VISIBLE_DEVICES=3 python3 carnator/adapters/wan_ti2v_5b.py \
    --device cuda:0 \
    --model-dir models/Wan2.2-TI2V-5B \
    --source-prompt "$SOURCE_PROMPT" \
    --source-positive "$SOURCE_NOUN" \
    --target-prompt "$TARGET_PROMPT" \
    --seed $SEED \
    --steps $STEPS \
    --reuse-start-step 10 \
    --mask-layers 15 \
    --sparse-layers 10-20 \
    --middle-attention-mask \
    --outer-attn-backend dense \
    --artifact-root "${ARTIFACT_ROOT}/source_gen" \
    --run-id source_gen

SOURCE_CACHE="${ARTIFACT_ROOT}/source_gen/source_gen/carnator"

echo ""
echo "Step 2: 并行运行4个变体 (GPU 3-6)"
echo ""

# Variant 1: Dense Baseline (GPU 3)
(
conda run --no-capture-output -n wan env CUDA_VISIBLE_DEVICES=3 python3 carnator/adapters/wan_ti2v_5b.py \
    --device cuda:0 \
    --model-dir models/Wan2.2-TI2V-5B \
    --target-prompt "$TARGET_PROMPT" \
    --seed $SEED \
    --steps $STEPS \
    --target-dense-only \
    --artifact-root "${ARTIFACT_ROOT}/dense_baseline" \
    --run-id dense
echo "Dense Baseline 完成"
) &

# Variant 2: Carnator Full (GPU 4)
(
conda run --no-capture-output -n wan env CUDA_VISIBLE_DEVICES=4 python3 carnator/adapters/wan_ti2v_5b.py \
    --device cuda:0 \
    --model-dir models/Wan2.2-TI2V-5B \
    --source-prompt "$SOURCE_PROMPT" \
    --target-prompt "$TARGET_PROMPT" \
    --seed $SEED \
    --steps $STEPS \
    --source-run-dir "$SOURCE_CACHE" \
    --reuse-start-step 10 \
    --mask-layers 15 \
    --sparse-layers 10-20 \
    --middle-attention-mask \
    --outer-attn-backend dense \
    --artifact-root "${ARTIFACT_ROOT}/carnator_full" \
    --run-id full
echo "Carnator Full 完成"
) &

# Variant 3: Carnator Latent-Only (GPU 5)
(
conda run --no-capture-output -n wan env CUDA_VISIBLE_DEVICES=5 python3 carnator/adapters/wan_ti2v_5b.py \
    --device cuda:0 \
    --model-dir models/Wan2.2-TI2V-5B \
    --source-prompt "$SOURCE_PROMPT" \
    --target-prompt "$TARGET_PROMPT" \
    --seed $SEED \
    --steps $STEPS \
    --source-run-dir "$SOURCE_CACHE" \
    --reuse-start-step 10 \
    --no-middle-attention-mask \
    --outer-attn-backend dense \
    --artifact-root "${ARTIFACT_ROOT}/carnator_latent_only" \
    --run-id latent
echo "Carnator Latent-Only 完成"
) &

# Variant 4: Carnator Attention-Only (GPU 6)
(
conda run --no-capture-output -n wan env CUDA_VISIBLE_DEVICES=6 python3 carnator/adapters/wan_ti2v_5b.py \
    --device cuda:0 \
    --model-dir models/Wan2.2-TI2V-5B \
    --source-prompt "$SOURCE_PROMPT" \
    --target-prompt "$TARGET_PROMPT" \
    --seed $SEED \
    --steps $STEPS \
    --source-run-dir "$SOURCE_CACHE" \
    --reuse-start-step 10 \
    --no-latent-injection \
    --mask-layers 15 \
    --sparse-layers 10-20 \
    --middle-attention-mask \
    --outer-attn-backend dense \
    --artifact-root "${ARTIFACT_ROOT}/carnator_attention_only" \
    --run-id attention
echo "Carnator Attention-Only 完成"
) &

wait

echo ""
echo "=================================="
echo "所有变体已完成"
echo "=================================="
echo "结果保存在: $ARTIFACT_ROOT"
