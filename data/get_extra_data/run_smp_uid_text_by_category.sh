#!/usr/bin/env bash
# run_smp_uid_text_by_category.sh
#
# Crawl Flickr photos using train user IDs + category text queries.
# Each user is assigned to their most frequent category in the train set.
#
# Usage:
#   ./run_smp_uid_text_by_category.sh
#
# Override any variable via env:
#   MAX_USERS_PER_CATEGORY=200 MAX_ITEMS_PER_UID=10 ./run_smp_uid_text_by_category.sh
#
# Dry run (no download, small scale):
#   MAX_USERS_PER_CATEGORY=5 MAX_ITEMS_PER_UID=2 DOWNLOAD_IMAGES=0 ./run_smp_uid_text_by_category.sh

set -euo pipefail

# -------------------------------------------------------
# Config
# -------------------------------------------------------
export PYTHON_BIN="${PYTHON_BIN:-python3}"
export CRAWLER_SCRIPT="${CRAWLER_SCRIPT:-./crawl_flickr_to_smp_by_uid_text.py}"
export TRAIN_USER_JSON="${TRAIN_USER_JSON:-/local/smp/data/train_allmetadata_json/train_user_data.json}"
export TRAIN_CATEGORY_JSON="${TRAIN_CATEGORY_JSON:-/local/smp/data/train_allmetadata_json/train_category.json}"
export BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-/local/smp/extra_data_by_uid}"
export MAX_USERS_PER_CATEGORY="${MAX_USERS_PER_CATEGORY:-0}"   # 0 = all users
export MAX_ITEMS_PER_UID="${MAX_ITEMS_PER_UID:-10}"
export LICENSES="${LICENSES:-4,5,7,8,9,10}"
export SLEEP_MIN="${SLEEP_MIN:-0.8}"
export SLEEP_MAX="${SLEEP_MAX:-2.0}"
export FLUSH_EVERY="${FLUSH_EVERY:-100}"
export PER_PAGE="${PER_PAGE:-100}"
export SORT="${SORT:-date-posted-desc}"
export DOWNLOAD_IMAGES="${DOWNLOAD_IMAGES:-1}"   # 1 = download, 0 = skip

ORCHESTRATOR="${ORCHESTRATOR:-./run_uid_text_crawl.py}"

# -------------------------------------------------------
# Pre-flight checks
# -------------------------------------------------------
if [ ! -f "$ORCHESTRATOR" ]; then
    echo "[ERROR] Orchestrator not found: $ORCHESTRATOR"
    exit 1
fi
if [ ! -f "$CRAWLER_SCRIPT" ]; then
    echo "[ERROR] Crawler script not found: $CRAWLER_SCRIPT"
    exit 1
fi
if [ ! -f "$TRAIN_USER_JSON" ]; then
    echo "[ERROR] Train user JSON not found: $TRAIN_USER_JSON"
    exit 1
fi
if [ ! -f "$TRAIN_CATEGORY_JSON" ]; then
    echo "[ERROR] Train category JSON not found: $TRAIN_CATEGORY_JSON"
    exit 1
fi

# -------------------------------------------------------
# Print config
# -------------------------------------------------------
echo "========================================"
echo "PYTHON_BIN             : $PYTHON_BIN"
echo "CRAWLER_SCRIPT         : $CRAWLER_SCRIPT"
echo "TRAIN_USER_JSON        : $TRAIN_USER_JSON"
echo "TRAIN_CATEGORY_JSON    : $TRAIN_CATEGORY_JSON"
echo "BASE_OUTPUT_DIR        : $BASE_OUTPUT_DIR"
echo "MAX_USERS_PER_CATEGORY : $MAX_USERS_PER_CATEGORY  (0 = all)"
echo "MAX_ITEMS_PER_UID      : $MAX_ITEMS_PER_UID"
echo "LICENSES               : $LICENSES"
echo "SORT                   : $SORT"
echo "DOWNLOAD_IMAGES        : $DOWNLOAD_IMAGES"
echo "========================================"
echo ""

mkdir -p "$BASE_OUTPUT_DIR"

# -------------------------------------------------------
# Run
# -------------------------------------------------------
"$PYTHON_BIN" "$ORCHESTRATOR"

echo ""
echo "Done."
