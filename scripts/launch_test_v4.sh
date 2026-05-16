#!/bin/bash
# 等 build_dataset_v3 (PID 2610928) 完成 → merge captions → 啟動 test_v4 訓練
set -e

PYTHON=/home/linchiayu/miniconda3/envs/venv/bin/python3
WORKDIR=/ssd1/lchiayu/smp-challenge
BUILD_PID=2610928
GPU=2

echo "[$(date)] Waiting for build_dataset_v3 (PID $BUILD_PID) to finish..."
while kill -0 $BUILD_PID 2>/dev/null; do
    sleep 30
done
echo "[$(date)] build_dataset_v3 finished."

# 確認 parquet 產生
if [ ! -f "$WORKDIR/data/processed_v4/official_train.parquet" ]; then
    echo "[ERROR] official_train.parquet not found in processed_v4. Aborting."
    exit 1
fi

# Merge captions
echo "[$(date)] Merging captions into processed_v4..."
$PYTHON $WORKDIR/data/merge_captions_to_parquet.py \
    --captions $WORKDIR/data/processed_v2/captions_blip2.json \
    --parquet  $WORKDIR/data/processed_v4/official_train.parquet \
    --output   $WORKDIR/data/processed_v4/official_train_cap.parquet
echo "[$(date)] Caption merge done."

# 啟動訓練
echo "[$(date)] Launching test_v4 on GPU $GPU..."
cd $WORKDIR
CUDA_VISIBLE_DEVICES=$GPU nohup $PYTHON scripts/train.py \
    --config configs/test_v4.yaml \
    --fold 0 \
    --n_folds 5 \
    > outputs/test_v4_fold0.log 2>&1 &
echo "[$(date)] test_v4 started, PID=$!, log=outputs/test_v4_fold0.log"
