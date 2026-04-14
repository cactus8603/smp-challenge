#!/usr/bin/env bash
# run_smp_category_crawl.sh
# Linux version of run_smp_category_crawl.ps1
# Category distribution aligned to test set

set -euo pipefail

# -----------------------------
# Config (override via env vars)
# -----------------------------
PYTHON_BIN="${PYTHON_BIN:-python3}"
CRAWLER_SCRIPT="${CRAWLER_SCRIPT:-./crawl_flickr_to_smp.py}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-/local/smp/extra_data}"
TOTAL_ITEMS="${TOTAL_ITEMS:-480000}"
LICENSES="${LICENSES:-4,5,7,8,9,10}"
SLEEP_MIN="${SLEEP_MIN:-0.8}"
SLEEP_MAX="${SLEEP_MAX:-2.0}"
FLUSH_EVERY="${FLUSH_EVERY:-200}"
PER_PAGE="${PER_PAGE:-100}"
SORT="${SORT:-date-posted-desc}"
MAX_NO_NEW_PAGES="${MAX_NO_NEW_PAGES:-20}"

# -----------------------------
# Category distribution (test set)
# Family base_target raised to 3000 (ratio gives only 1120, too few)
# Format: "category|ratio|query|base_target"
# -----------------------------
CATEGORIES=(
    "Travel&Active&Sports|0.2518|travel|25180"
    "Holiday&Celebrations|0.1079|holiday|10790"
    "Animal|0.1023|animal|10230"
    "Entertainment|0.0995|concert|9950"
    "Fashion|0.0995|fashion|9950"
    "Whether&Season|0.0827|winter|8270"
    "Social&People|0.0800|people|8000"
    "Urban|0.0665|city|6650"
    "Food|0.0655|food|6550"
    "Electronics|0.0332|electronics|3320"
    "Family|0.0112|family|3000"
)

# -----------------------------
# Helpers
# -----------------------------
sanitize_name() {
    echo "$1" | tr '& /' '___'
}

get_scaled_target() {
    local base_target=$1
    local total=$2
    # round(base_target * total / 100000), minimum 1
    python3 -c "import math; print(max(1, round($base_target * $total / 100000.0)))"
}

# -----------------------------
# Main
# -----------------------------
mkdir -p "$BASE_OUTPUT_DIR"

echo "Base output dir : $BASE_OUTPUT_DIR"
echo "Total target    : $TOTAL_ITEMS"
echo "Crawler script  : $CRAWLER_SCRIPT"
echo "Sort            : $SORT"
echo ""

for entry in "${CATEGORIES[@]}"; do
    IFS='|' read -r category ratio query base_target <<< "$entry"

    target=$(get_scaled_target "$base_target" "$TOTAL_ITEMS")
    safe_category=$(sanitize_name "$category")
    output_dir="${BASE_OUTPUT_DIR}/extra_data_${safe_category}"

    echo "========================================"
    echo "Category : $category"
    echo "Ratio    : $ratio"
    echo "Query    : $query"
    echo "Target   : $target"
    echo "Output   : $output_dir"
    echo "========================================"

    "$PYTHON_BIN" "$CRAWLER_SCRIPT" \
        --output_dir "$output_dir" \
        --text "$query" \
        --max_items "$target" \
        --licenses "$LICENSES" \
        --download_images \
        --sleep_min "$SLEEP_MIN" \
        --sleep_max "$SLEEP_MAX" \
        --flush_every "$FLUSH_EVERY" \
        --per_page "$PER_PAGE" \
        --sort "$SORT" \
        --max_no_new_pages "$MAX_NO_NEW_PAGES" \
        --resume \
        --dedupe_on_image_path

    echo ""
done

echo "All category crawls finished."
