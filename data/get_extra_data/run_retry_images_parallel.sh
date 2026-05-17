#!/usr/bin/env bash
# Launch one retry_download_images.py process per category in background.
# Logs go to logs/retry_images_<category>.log

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
PYTHON="${PYTHON:-python3}"

SLEEP_MIN="${SLEEP_MIN:-1.0}"
SLEEP_MAX="${SLEEP_MAX:-2.5}"

mkdir -p "$LOG_DIR"

CATEGORIES=(
    extra_data_Animal
    extra_data_Electronics
    extra_data_Entertainment
    extra_data_Family
    extra_data_Fashion
    extra_data_Food
    extra_data_Holiday_Celebrations
    extra_data_Social_People
    extra_data_Travel_Active_Sports
    extra_data_Urban
    extra_data_Whether_Season
)

echo "Launching ${#CATEGORIES[@]} parallel download workers..."
echo "sleep_min=$SLEEP_MIN  sleep_max=$SLEEP_MAX"
echo ""

PIDS=()
for CAT in "${CATEGORIES[@]}"; do
    LOG="$LOG_DIR/retry_images_${CAT}.log"
    $PYTHON "$SCRIPT_DIR/retry_download_images.py" \
        --category "$CAT" \
        --sleep_min "$SLEEP_MIN" \
        --sleep_max "$SLEEP_MAX" \
        > "$LOG" 2>&1 &
    PID=$!
    PIDS+=($PID)
    echo "  [$PID] $CAT  →  $LOG"
done

echo ""
echo "All workers started. Monitor with:"
echo "  tail -f $LOG_DIR/retry_images_*.log"
echo ""
echo "Check progress:"
echo "  grep -c 'Downloaded\|Downloading' $LOG_DIR/retry_images_*.log"
echo ""
echo "PIDs: ${PIDS[*]}"
