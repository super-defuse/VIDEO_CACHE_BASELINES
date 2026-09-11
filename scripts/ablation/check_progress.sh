#!/bin/bash
# 检查消融实验进度

ABLATION_DIR="artifacts/ablation_req16_20260909_100759"

echo "================================================================================"
echo "Carnator Ablation Study Progress"
echo "================================================================================"
echo "Time: $(date)"
echo ""

# 检查GPU使用情况
echo "GPU Status (3-6):"
nvidia-smi --query-compute-apps=gpu_bus_id,pid,process_name,used_memory --format=csv,noheader | grep -E "9A|9B|9C|9D" | nl
echo ""

# 检查各个变体
for variant in full_computation latent_only attention_only joint; do
    echo "[$variant]"
    log_file="$ABLATION_DIR/$variant/run.log"

    if [ -f "$log_file" ]; then
        # 检查是否完成
        if grep -q "Completed:" "$log_file"; then
            elapsed=$(grep "Elapsed:" "$log_file" | tail -1)
            exit_code=$(grep "Exit code:" "$log_file" | tail -1)
            echo "  Status: ✓ Completed"
            echo "  $elapsed"
            echo "  $exit_code"
        else
            # 查找最后的进度信息
            last_line=$(tail -5 "$log_file" | grep -E "step|Loading|Sampling|Writing" | tail -1)
            if [ -n "$last_line" ]; then
                echo "  Status: Running - $last_line"
            else
                echo "  Status: Running..."
            fi
        fi
    else
        echo "  Status: Not started"
    fi
    echo ""
done

# 检查results目录
echo "Results Directory:"
if [ -d "results/ti2v_5b" ]; then
    ls -lh results/ti2v_5b/ | tail -n +2
else
    echo "  Not created yet"
fi

echo "================================================================================"
