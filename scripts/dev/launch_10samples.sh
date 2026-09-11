#!/bin/bash
# Launch 4 baselines in parallel on GPUs 3-6

cd /home/tangxuebin/project/VIDEO_CACHE_BASELINES

# Carnator on GPU 3
CUDA_VISIBLE_DEVICES=3 python3 scripts/dev/run_10samples_simple.py --method carnator_boundary_b5 --gpu 0 --limit 10 > artifacts/cogvideox15_10samples_test/carnator_gpu3.log 2>&1 &
PID1=$!

# Chorus on GPU 4  
CUDA_VISIBLE_DEVICES=4 python3 scripts/dev/run_10samples_simple.py --method chorus --gpu 0 --limit 10 > artifacts/cogvideox15_10samples_test/chorus_gpu4.log 2>&1 &
PID2=$!

# Chorus2 on GPU 5
CUDA_VISIBLE_DEVICES=5 python3 scripts/dev/run_10samples_simple.py --method chorus2 --gpu 0 --limit 10 > artifacts/cogvideox15_10samples_test/chorus2_gpu5.log 2>&1 &
PID3=$!

# NirvanaVID on GPU 6
CUDA_VISIBLE_DEVICES=6 python3 scripts/dev/run_10samples_simple.py --method nirvana_vid --gpu 0 --limit 10 > artifacts/cogvideox15_10samples_test/nirvana_gpu6.log 2>&1 &
PID4=$!

echo "Launched 4 baselines:"
echo "  Carnator (PID $PID1) on GPU 3"
echo "  Chorus (PID $PID2) on GPU 4"
echo "  Chorus2 (PID $PID3) on GPU 5"
echo "  NirvanaVID (PID $PID4) on GPU 6"

# Wait for all to complete
wait $PID1
echo "Carnator completed"
wait $PID2
echo "Chorus completed"
wait $PID3
echo "Chorus2 completed"
wait $PID4
echo "NirvanaVID completed"

echo "All experiments completed!"
