#!/bin/bash
# Monitor the 10-sample experiment progress

echo "============================================"
echo "CogVideoX 10-Sample Experiment Monitor"
echo "============================================"
echo ""

# Check if experiments are running
echo "Running processes:"
ps aux | grep "run_cogvideox_10samples.py" | grep -v grep
echo ""

# Check GPU usage
echo "GPU Status:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
echo ""

# Check output directories
OUTPUT_ROOT="artifacts/cogvideox15_10samples_test"
if [ -d "$OUTPUT_ROOT" ]; then
    echo "Output directories:"
    ls -lht "$OUTPUT_ROOT" | head -20
    echo ""

    # Check results for each baseline
    for method in baseline carnator_boundary_b5 chorus chorus2 nirvana_vid; do
        latest_dir=$(ls -dt "$OUTPUT_ROOT"/${method}_* 2>/dev/null | head -1)
        if [ -n "$latest_dir" ]; then
            echo "[$method]"
            if [ -f "$latest_dir/results.jsonl" ]; then
                count=$(wc -l < "$latest_dir/results.jsonl")
                echo "  Results: $count/10 completed"

                # Show last result status
                if [ $count -gt 0 ]; then
                    last_status=$(tail -1 "$latest_dir/results.jsonl" | python3 -c "import json,sys; d=json.load(sys.stdin); print(f\"  Last: req{d['request_index']} - {d['status']} - {d.get('extra',{}).get('hit_e2e_s','N/A')}s\")")
                    echo "$last_status"
                fi
            else
                echo "  No results yet"
            fi
            echo ""
        fi
    done
else
    echo "Output directory not found: $OUTPUT_ROOT"
fi

# Check main log
if [ -f "run_10samples.log" ]; then
    echo "Last 20 lines of main log:"
    tail -20 run_10samples.log
fi
