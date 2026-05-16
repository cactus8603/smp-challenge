#!/bin/bash
set -e

PYTHON=/home/linchiayu/miniconda3/envs/venv/bin/python3
WORKDIR=/ssd1/lchiayu/smp-challenge
TEST_V4_PID=2503910
GPU=3

echo "[$(date)] Waiting for test_v4 (PID $TEST_V4_PID) to finish..."
while kill -0 $TEST_V4_PID 2>/dev/null; do
    sleep 60
done
echo "[$(date)] test_v4 finished. Launching test_v5 on GPU $GPU..."

cd $WORKDIR
CUDA_VISIBLE_DEVICES=$GPU nohup $PYTHON scripts/train.py \
    --config configs/test_v5.yaml \
    --fold 0 \
    --n_folds 5 \
    > outputs/test_v5_fold0.log 2>&1 &
echo "[$(date)] test_v5 started, PID=$!, log=outputs/test_v5_fold0.log"
