#!/usr/bin/env bash
set -euo pipefail

# Run with:
#   bash run_smp_category_crawl.sh
#
# Optional environment overrides:
#   PYTHON_BIN=python
#   CRAWLER_SCRIPT=./crawl_flickr_to_smp.py
#   BASE_OUTPUT_DIR=./category_crawls
#   TOTAL_ITEMS=100000
#   LICENSES=4,5,7,8,9,10
#   SLEEP_MIN=0.8
#   SLEEP_MAX=2.0

PYTHON_BIN="${PYTHON_BIN:-python}"
CRAWLER_SCRIPT="${CRAWLER_SCRIPT:-./crawl_flickr_to_smp.py}"

BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-./category_crawls}"
TOTAL_ITEMS="${TOTAL_ITEMS:-100000}"
LICENSES="${LICENSES:-4,5,7,8,9,10}"

SLEEP_MIN="${SLEEP_MIN:-0.8}"
SLEEP_MAX="${SLEEP_MAX:-2.0}"
FLUSH_EVERY="${FLUSH_EVERY:-200}"
PER_PAGE="${PER_PAGE:-100}"
SORT="${SORT:-relevance}"

COMMON_ARGS=(
  --licenses "$LICENSES"
  --download_images
  --sleep_min "$SLEEP_MIN"
  --sleep_max "$SLEEP_MAX"
  --flush_every "$FLUSH_EVERY"
  --per_page "$PER_PAGE"
  --sort "$SORT"
  --resume
  --dedupe_on_image_path
  --write_pretty_json
)

mkdir -p "$BASE_OUTPUT_DIR"

# Based on your SMP train category distribution.
# Format: category|ratio|query|base_target_for_100k
declare -a CATEGORIES=(
  "Travel&Active&Sports|0.2175|travel|21750"
  "Holiday&Celebrations|0.1792|holiday|17920"
  "Fashion|0.1613|fashion|16130"
  "Entertainment|0.0968|concert|9680"
  "Social&People|0.0764|people|7640"
  "Whether&Season|0.0664|winter|6640"
  "Animal|0.0654|animal|6540"
  "Food|0.0547|food|5470"
  "Urban|0.0500|city|5000"
  "Electronics|0.0184|electronics|1840"
  "Family|0.0140|family|1400"
)

sanitize_name() {
  local name="$1"
  name="${name//&/_}"
  name="${name// /_}"
  name="${name//\//_}"
  echo "$name"
}

scaled_target() {
  local base_target="$1"
  python - "$TOTAL_ITEMS" "$base_target" <<'PY'
import sys
total_items = int(sys.argv[1])
base_target = int(sys.argv[2])
base_total = 100000
print(max(1, round(base_target * total_items / base_total)))
PY
}

echo "Base output dir: $BASE_OUTPUT_DIR"
echo "Total target items: $TOTAL_ITEMS"
echo "Crawler script: $CRAWLER_SCRIPT"
echo

for item in "${CATEGORIES[@]}"; do
  IFS="|" read -r category ratio query base_target <<< "$item"

  target="$(scaled_target "$base_target")"
  safe_category="$(sanitize_name "$category")"
  output_dir="${BASE_OUTPUT_DIR}/extra_data_${safe_category}"

  echo "========================================"
  echo "Category : $category"
  echo "Ratio    : $ratio"
  echo "Query    : $query"
  echo "Target   : $target"
  echo "Output   : $output_dir"
  echo "========================================"

  "$PYTHON_BIN" "$CRAWLER_SCRIPT"     --output_dir "$output_dir"     --text "$query"     --max_items "$target"     "${COMMON_ARGS[@]}"

  echo
done

echo "All category crawls finished."
