#!/bin/bash
# Auto-start BLIP-2 caption pipeline after training finishes.
# Usage: bash scripts/auto_caption_after_train.sh <TRAIN_PID>
#
# Example:
#   bash scripts/auto_caption_after_train.sh 247691

set -euo pipefail

TRAIN_PID="${1:-}"
PYTHON=/ssd1/lchiayu/smp-challenge/data/get_extra_data/venv/bin/python
REPO=/ssd1/lchiayu/smp-challenge
LOG=/tmp/auto_caption.log

IMAGE_DIR="$REPO/data/official_data/train_set/train"
CAPTIONS_JSON="$REPO/data/processed_v2/captions_blip2.json"
TRAIN_PARQUET="$REPO/data/processed_v2/official_train.parquet"
CAP_PARQUET="$REPO/data/processed_v2/official_train_cap.parquet"
BATCH_SIZE=16

exec > >(tee -a "$LOG") 2>&1
echo "=============================="
echo "[AUTO] Started at $(date)"
echo "=============================="

# ── Step 1: Wait for training to finish ──────────────────────────────────────
if [ -z "$TRAIN_PID" ]; then
    echo "[AUTO] No PID given. Detecting training process..."
    TRAIN_PID=$(pgrep -f "scripts/train.py" | head -1 || true)
fi

if [ -n "$TRAIN_PID" ] && kill -0 "$TRAIN_PID" 2>/dev/null; then
    echo "[AUTO] Waiting for training process PID=$TRAIN_PID to finish..."
    while kill -0 "$TRAIN_PID" 2>/dev/null; do
        sleep 60
        echo "[AUTO] $(date '+%H:%M') — training still running (PID=$TRAIN_PID)..."
    done
    echo "[AUTO] Training process $TRAIN_PID exited."
else
    echo "[AUTO] No active training process found. Proceeding immediately."
fi

# Give workers a moment to fully release GPU memory
sleep 30

# ── Step 2: Confirm GPU is free ───────────────────────────────────────────────
GPU_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
GPU_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
GPU_FREE=$(( GPU_TOTAL - GPU_USED ))
echo "[AUTO] GPU memory: used=${GPU_USED}MiB  free=${GPU_FREE}MiB  total=${GPU_TOTAL}MiB"

if [ "$GPU_FREE" -lt 8000 ]; then
    echo "[AUTO] WARNING: Only ${GPU_FREE}MiB free. BLIP-2 needs ~9GB."
    echo "[AUTO] Waiting up to 5 more minutes for memory to clear..."
    for i in $(seq 1 5); do
        sleep 60
        GPU_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
        GPU_FREE=$(( GPU_TOTAL - GPU_USED ))
        echo "[AUTO] $(date '+%H:%M') GPU free: ${GPU_FREE}MiB"
        [ "$GPU_FREE" -ge 8000 ] && break
    done
fi

# ── Step 3: Run BLIP-2 captioning ────────────────────────────────────────────
echo "[AUTO] Starting BLIP-2 captioning at $(date)..."
echo "[AUTO] Output: $CAPTIONS_JSON"
echo "[AUTO] Batch size: $BATCH_SIZE"

"$PYTHON" "$REPO/data/generate_captions_blip2.py" \
    --image_dir "$IMAGE_DIR" \
    --output    "$CAPTIONS_JSON" \
    --batch_size "$BATCH_SIZE" \
    --resume

echo "[AUTO] BLIP-2 captioning done at $(date)"

# ── Step 4: Merge captions into parquet ──────────────────────────────────────
echo "[AUTO] Merging captions into parquet..."
"$PYTHON" "$REPO/data/merge_captions_to_parquet.py" \
    --captions "$CAPTIONS_JSON" \
    --parquet  "$TRAIN_PARQUET" \
    --output   "$CAP_PARQUET"

echo "[AUTO] Merge done. Caption parquet: $CAP_PARQUET"

# ── Step 5: Start caption experiment training ─────────────────────────────────
echo "[AUTO] Starting caption_exp training at $(date)..."
TRAIN_LOG="$REPO/outputs/caption_exp/fold_0/nohup_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$TRAIN_LOG")"

nohup "$PYTHON" "$REPO/scripts/train.py" \
    --config "$REPO/configs/caption_exp.yaml" \
    --fold 0 --n_folds 5 \
    > "$TRAIN_LOG" 2>&1 &

CAPTION_TRAIN_PID=$!
echo "[AUTO] Caption training started: PID=$CAPTION_TRAIN_PID  Log=$TRAIN_LOG"
echo "[AUTO] Pipeline complete at $(date)"
echo "[AUTO] Full log: $LOG"
