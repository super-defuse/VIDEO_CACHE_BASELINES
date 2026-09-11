#!/bin/bash
# Monitor the 5-baseline 10-sample experiment progress

OUTPUT_ROOT="artifacts/cogvideox15_5baselines_10samples"

echo "============================================"
echo "CogVideoX 5-Baseline Experiment Monitor"
echo "============================================"
echo ""

# Check if experiments are running
echo "Running processes:"
ps aux | grep "run_5baselines_10samples.py" | grep -v grep | grep -v monitor
if [ $? -ne 0 ]; then
    echo "  No active experiment processes"
fi
echo ""

# Check GPU usage
echo "GPU Status:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader | grep -E "^[3-7]," | column -t -s ','
echo ""

# Check output directories
if [ -d "$OUTPUT_ROOT" ]; then
    echo "Progress:"
    echo "--------"

    for method in baseline carnator_boundary_b5 chorus chorus2 nirvana_vid; do
        latest_dir=$(ls -dt "$OUTPUT_ROOT"/${method}_* 2>/dev/null | head -1)
        if [ -n "$latest_dir" ]; then
            printf "%-25s" "[$method]"
            if [ -f "$latest_dir/results.jsonl" ]; then
                count=$(wc -l < "$latest_dir/results.jsonl")
                printf " %2d/10 completed" "$count"

                # Show last result timing if available
                if [ $count -gt 0 ]; then
                    last_time=$(tail -1 "$latest_dir/results.jsonl" | python3 -c "import json,sys; d=json.load(sys.stdin); print(f\"{d.get('extra',{}).get('hit_e2e_s','N/A'):.1f}s\")" 2>/dev/null || echo "N/A")
                    printf "  (last: %s)" "$last_time"
                fi
            else
                printf " %s" "No results yet"
            fi
            echo ""
        else
            printf "%-25s %s\n" "[$method]" "Not started"
        fi
    done
    echo ""

    # Check logs for errors
    echo "Recent errors (if any):"
    echo "----------------------"
    for logfile in "$OUTPUT_ROOT"/*.log; do
        if [ -f "$logfile" ]; then
            errors=$(grep -i "error\|traceback\|failed" "$logfile" 2>/dev/null | tail -3)
            if [ -n "$errors" ]; then
                echo "$(basename $logfile):"
                echo "$errors"
                echo ""
            fi
        fi
    done
else
    echo "Output directory not found: $OUTPUT_ROOT"
fi

echo "Refresh: watch -n 5 ./scripts/dev/monitor_5baselines.sh"
