#!/usr/bin/env bash
# run_smp_category_crawl_v2.sh
#
# Crawl Flickr photos for all 11 SMP categories.
# V2 changes:
#   1. Output to /local/smp/extra_data_v2 by default
#   2. Average crawl quota across primary + extra queries
#   3. Use cumulative target so every query contributes
#   4. Default MAX_PHOTOS_PER_USER=5 to avoid user over-dominance
#
# Usage:
#   ./run_smp_category_crawl_v2.sh
#
# Example:
#   BASE_OUTPUT_DIR=/local/smp/extra_data_v2 \
#   TOTAL_ITEMS=480000 \
#   MAX_PHOTOS_PER_USER=5 \
#   ./run_smp_category_crawl_v2.sh
#
# Dry run:
#   TOTAL_ITEMS=5000 DOWNLOAD_IMAGES=0 TIME_SLICE=0 MAX_PHOTOS_PER_USER=3 ./run_smp_category_crawl_v2.sh

set -euo pipefail

# ──────────────────────────────────────────────────────
# Config (override via env vars)
# ──────────────────────────────────────────────────────
PYTHON_BIN="${PYTHON_BIN:-python3}"
CRAWLER_SCRIPT="${CRAWLER_SCRIPT:-./crawl_flickr_to_smp.py}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-/local/smp/extra_data_v2}"
TOTAL_ITEMS="${TOTAL_ITEMS:-480000}"
LICENSES="${LICENSES:-4,5,7,8,9,10}"
SLEEP_MIN="${SLEEP_MIN:-0.8}"
SLEEP_MAX="${SLEEP_MAX:-2.0}"
FLUSH_EVERY="${FLUSH_EVERY:-200}"
PER_PAGE="${PER_PAGE:-100}"
SORT="${SORT:-date-posted-desc}"
MAX_NO_NEW_PAGES="${MAX_NO_NEW_PAGES:-20}"
DOWNLOAD_IMAGES="${DOWNLOAD_IMAGES:-1}"       # 1 = download, 0 = skip
TIME_SLICE="${TIME_SLICE:-1}"                 # 1 = enable time slicing
DATE_START="${DATE_START:-2004-01-01}"
DATE_END="${DATE_END:-2024-12-31}"
INITIAL_WINDOW_DAYS="${INITIAL_WINDOW_DAYS:-180}"
MAX_PHOTOS_PER_USER="${MAX_PHOTOS_PER_USER:-5}"
FETCH_PHOTO_DETAIL="${FETCH_PHOTO_DETAIL:-0}" # 1 = call getInfo; slower but richer

# ──────────────────────────────────────────────────────
# Category distribution + multi-keyword
# Format:
#   SMP_Category|primary_query|extra_queries(comma-separated)|base_target_at_100k
# ──────────────────────────────────────────────────────
CATEGORIES=(
    "Travel&Active&Sports|travel|hiking,landscape,mountain,backpacking,adventure,tourism,landmark,camping,cycling,running,skiing,surfing,trekking,climbing,kayaking,roadtrip,nationalpark,waterfall,canyon,glacier,safari,scuba,marathon,triathlon,motorbike,sailing,paragliding,crossfit,yoga,volleyball|25180"

    "Holiday&Celebrations|holiday|christmas,newyear,halloween,parade,carnival,fireworks,graduation,thanksgiving,easter,anniversary,diwali,hanukkah,ramadan,lunarnewyear,oktoberfest,mardigras,stpatrick,valentines,mothersday,fathersday,independence,reunion,prom,quinceanera,baptism,retirement|10790"

    "Animal|animal|cat,dog,wildlife,bird,pet,zoo,fish,horse,insect,butterfly,marine,elephant,deer,rabbit,lion,tiger,bear,wolf,fox,owl,penguin,crocodile,snake,parrot,hamster,turtle,whale,dolphin,shark,monkey|10230"

    "Entertainment|concert|music,performance,dance,show,stage,theater,band,guitar,piano,drama,cinema,comedy,magic,circus,festival,dj,hiphop,jazz,opera,ballet,puppet,standup,esports,boardgame,cosplay,anime,convention,karaoke,busking|9950"

    "Fashion|fashion|style,outfit,model,clothing,dress,shoes,bag,makeup,beauty,hair,runway,jewelry,sunglasses,menswear,streetwear,vintage,luxury,accessory,lingerie,swimwear,sneakers,coat,suit,scarf,watch,perfume,editorial,lookbook,designer|9950"

    "Whether&Season|winter|snow,rain,summer,autumn,fog,spring,cloud,storm,sunset,sunrise,flower,leaves,ice,weather,rainbow,lightning,hail,heatwave,blizzard,drizzle,mist,dewdrop,frost,puddle,overcast,tornado,typhoon,monsoon,drought|8270"

    "Social&People|people|portrait,street,crowd,friends,community,selfie,smile,group,event,volunteers,protest,gathering,wedding,elderly,teenager,student,worker,musician,athlete,activist,journalist,doctor,soldier,farmer,fisherman,chef|8000"

    "Urban|city|architecture,skyline,building,urban,night,bridge,road,traffic,downtown,skyscraper,neon,station,market,alley,graffiti,subway,rooftop,courtyard,harbor,suburb,chinatown,mosque,cathedral,temple,museum,library,plaza,park|6650"

    "Food|food|meal,restaurant,cooking,coffee,dessert,lunch,dinner,breakfast,cake,drink,tea,bread,fruit,chef,sushi,pizza,burger,ramen,pasta,tacos,curry,bbq,icecream,chocolate,smoothie,salad,streetfood,bakery,buffet,vegan|6550"

    "Electronics|electronics|technology,gadget,phone,computer,camera,laptop,tablet,headphones,robot,drone,gaming,console,keyboard,screen,device,smartwatch,speaker,charger,sensor,circuit,server,coding,programming,3dprinting,vr,ar,iot,selfiephone,dashcam|3320"

    "Family|family|baby,children,parenting,together,kids,mother,father,child,parents,house,newborn,toddler,grandparents,siblings,playdate,schoolbus,lunchbox,bedtime,bathtime,stroller,nursery,adoption,babysitter,kindergarten,homework,birthday|3000"
)

# ──────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────
sanitize_name() {
    echo "$1" | tr '& /' '___'
}

scale_target() {
    local base=$1
    python3 -c "print(max(1, round($base * $TOTAL_ITEMS / 100000.0)))"
}

download_flag() {
    [ "${DOWNLOAD_IMAGES}" = "1" ] && echo "--download_images" || echo ""
}

time_slice_flag() {
    [ "${TIME_SLICE}" = "1" ] && echo "--time_slice" || echo ""
}

fetch_detail_flag() {
    [ "${FETCH_PHOTO_DETAIL}" = "1" ] && echo "--fetch_photo_detail" || echo ""
}

count_existing_items() {
    local output_dir=$1
    local f="${output_dir}/extra_text.jsonl"
    if [ -f "$f" ]; then
        grep -cve '^\s*$' "$f" || true
    else
        echo 0
    fi
}

# ──────────────────────────────────────────────────────
# Pre-flight
# ──────────────────────────────────────────────────────
if [ -z "${FLICKR_API_KEY:-}" ]; then
    echo "[ERROR] FLICKR_API_KEY is not set."
    echo "  Export it: export FLICKR_API_KEY=your_key_here"
    exit 1
fi

if [ ! -f "$CRAWLER_SCRIPT" ]; then
    echo "[ERROR] Crawler not found: $CRAWLER_SCRIPT"
    exit 1
fi

mkdir -p "$BASE_OUTPUT_DIR"

cat <<INFO
========================================
SMP Flickr category crawl V2
========================================
Base output dir     : $BASE_OUTPUT_DIR
Total target        : $TOTAL_ITEMS
Licenses            : $LICENSES
Sort                : $SORT
Download images     : $DOWNLOAD_IMAGES
Time slice          : $TIME_SLICE
Max photos per user : $MAX_PHOTOS_PER_USER
Fetch photo detail  : $FETCH_PHOTO_DETAIL
========================================
INFO

# ──────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────
for entry in "${CATEGORIES[@]}"; do
    IFS='|' read -r category primary_query extra_queries base_target <<< "$entry"

    target=$(scale_target "$base_target")
    safe_category=$(sanitize_name "$category")
    output_dir="${BASE_OUTPUT_DIR}/extra_data_${safe_category}"

    mkdir -p "$output_dir"

    all_queries="${primary_query},${extra_queries}"
    IFS=',' read -ra query_list <<< "$all_queries"
    num_queries=${#query_list[@]}
    per_query_target=$(python3 -c "import math; print(math.ceil($target / $num_queries))")

    echo ""
    echo "========================================"
    echo "Category          : $category"
    echo "Output            : $output_dir"
    echo "Category target   : $target"
    echo "Query count       : $num_queries"
    echo "Per-query quota   : $per_query_target"
    echo "Existing items    : $(count_existing_items "$output_dir")"
    echo "========================================"

    dl_flag=$(download_flag)
    ts_flag=$(time_slice_flag)
    dt_flag=$(fetch_detail_flag)

    idx=0
    for q in "${query_list[@]}"; do
        q="$(echo "$q" | xargs)"
        [ -z "$q" ] && continue

        idx=$((idx + 1))
        current_target=$((per_query_target * idx))
        if [ "$current_target" -gt "$target" ]; then
            current_target="$target"
        fi

        existing=$(count_existing_items "$output_dir")
        if [ "$existing" -ge "$current_target" ]; then
            echo "----------------------------------------"
            echo "[$idx/$num_queries] Query skipped : $q"
            echo "Existing items >= cumulative target: $existing >= $current_target"
            echo "----------------------------------------"
            continue
        fi

        echo "----------------------------------------"
        echo "[$idx/$num_queries] Query       : $q"
        echo "Existing items                  : $existing"
        echo "Cumulative target               : $current_target / $target"
        echo "----------------------------------------"

        "$PYTHON_BIN" "$CRAWLER_SCRIPT" \
            --output_dir          "$output_dir"          \
            --text                "$q"                   \
            --max_items           "$current_target"      \
            --licenses            "$LICENSES"            \
            --sort                "$SORT"                \
            --sleep_min           "$SLEEP_MIN"           \
            --sleep_max           "$SLEEP_MAX"           \
            --flush_every         "$FLUSH_EVERY"         \
            --per_page            "$PER_PAGE"            \
            --max_no_new_pages    "$MAX_NO_NEW_PAGES"    \
            --seed_category       "$category"            \
            --date_start          "$DATE_START"          \
            --date_end            "$DATE_END"            \
            --initial_window_days "$INITIAL_WINDOW_DAYS" \
            --max_photos_per_user "$MAX_PHOTOS_PER_USER" \
            --resume                                     \
            --dedupe_on_image_path                       \
            ${dl_flag} ${ts_flag} ${dt_flag}

        echo "After query items: $(count_existing_items "$output_dir")"
        echo ""
    done

    echo "Finished category: $category"
    echo "Final items      : $(count_existing_items "$output_dir") / $target"
    echo ""
done

echo "All category crawls finished."
