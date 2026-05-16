#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────
# Config  (override via env vars)
# ──────────────────────────────────────────────
PYTHON_BIN="${PYTHON_BIN:-python3}"
CRAWLER_SCRIPT="${CRAWLER_SCRIPT:-./crawl_flickr_to_smp.py}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-/ssd1/lchiayu/smp-challenge/data/get_extra_data/output}"
LOG_DIR="${LOG_DIR:-/ssd1/lchiayu/smp-challenge/data/get_extra_data/logs}"
TOTAL_ITEMS="${TOTAL_ITEMS:-480000}"
LICENSES="${LICENSES:-4,5,7,8,9,10}"
SLEEP_MIN="${SLEEP_MIN:-1.5}"
SLEEP_MAX="${SLEEP_MAX:-3.0}"
FLUSH_EVERY="${FLUSH_EVERY:-200}"
PER_PAGE="${PER_PAGE:-100}"
SORT="${SORT:-date-posted-desc}"
MAX_NO_NEW_PAGES="${MAX_NO_NEW_PAGES:-20}"
# GEO_RATIO: fraction of photos per category with geo coords (~11.4% in official SMP)
GEO_RATIO="${GEO_RATIO:-0.114}"
# New parameters (from v3)
DOWNLOAD_IMAGES="${DOWNLOAD_IMAGES:-1}"          # 1=download images, 0=metadata only
DATE_START="${DATE_START:-2004-01-01}"            # Flickr min_upload_date
DATE_END="${DATE_END:-2024-12-31}"                # Flickr max_upload_date
MAX_PHOTOS_PER_USER="${MAX_PHOTOS_PER_USER:-5}"   # avoid single user dominating
# Path to geo lookup JSON {"lat,lon": {"city":..,"state":..,"country":..}}
# Get this file from classmate; leave empty to skip geo enrichment
GEO_LOOKUP_PATH="${GEO_LOOKUP_PATH:-}"

# ──────────────────────────────────────────────
# Category distribution
# Format: "Category|ratio|kw1,kw2,..."
# ──────────────────────────────────────────────
CATEGORIES=(
    # 48 kws — ratio reduced from 0.2518 → 0.0882 (target 42,336 < existing 42,415 → fully skipped)
    # Original 15 kept; 33 new added (no overlap with classmate's kw set)
    "Travel&Active&Sports|0.0882|travel,sports,adventure,vacation,fitness,cycling,running,swimming,yoga,marathon,wanderlust,tourism,journey,workout,training,tennis,golf,basketball,soccer,baseball,hockey,boxing,skateboarding,snowboarding,rowing,badminton,archery,equestrian,gymnastics,pilates,weightlifting,calisthenics,aerobics,spinning,bootcamp,expedition,hostel,sightseeing,cruise,nomad,resort,fjord,savannah,reef,pilgrimage,lagoon,peninsula,bungee"
    # 21 kws — ~2,466 photos/kw
    "Holiday&Celebrations|0.1079|celebration,holiday,party,festival,wedding,christmas,fireworks,newyear,thanksgiving,ceremony,gathering,cheers,festive,gifts,anniversary,toast,confetti,champagne,ribbon,balloon,countdown"
    # 20 kws — ~2,455 photos/kw
    "Animal|0.1023|animal,wildlife,pet,nature-photography,dog,cat,bird,butterfly,fish,horse,kitten,puppy,zoo,safari,reptile,hedgehog,squirrel,flamingo,cheetah,gorilla"
    # 19 kws — ~2,514 photos/kw
    "Entertainment|0.0995|music,movie,entertainment,gaming,video,concert,cinema,album,song,streaming,hiphop,rock,dj,playlist,series,podcast,vlog,remix,livestream"
    # 19 kws — ~2,514 photos/kw
    "Fashion|0.0995|fashion,style,outfit,clothes,model,makeup,jewelry,ootd,streetstyle,vintage,lookbook,runway,shoe,fashionista,hairstyle,capsule,upcycled,thrift,monochrome"
    # 16 kws — ~2,481 photos/kw
    "Whether&Season|0.0827|weather,season,winter,summer,autumn,snow,rain,sunshine,cloud,spring,fog,rainbow,storm,sunny,windy,haze"
    # 15 kws — already at target
    "Social&People|0.0800|people,friends,friendship,social,portrait,selfie,couple,lifestyle,smile,love,relationship,hangout,weekend,group,laugh"
    # 15 kws — slightly over target (13), keep for diversity
    "Urban|0.0665|city,urban,architecture,street,building,skyline,graffiti,downtown,cityscape,subway,night,landmark,bridge,streetart,metropolis"
    # 15 kws — slightly over target (13), keep for diversity
    "Food|0.0655|food,cooking,delicious,restaurant,cuisine,breakfast,coffee,dessert,foodie,lunch,dinner,vegan,bbq,brunch,recipe"
    # 15 kws — over target (8), extra kws just skip early via cumulative logic
    "Electronics|0.0332|technology,electronics,gadgets,device,mobile,smartphone,laptops,tablet,headphones,smartwatch,digital,android,keyboard,pc,tech"
    # 15 kws — over target (8), same as above
    "Family|0.0112|family,parenting,child,home,together,babies,mother,father,kids,toddler,siblings,infant,parents,familytime,relatives"
)

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
sanitize_name() {
    echo "$1" | tr '& /' '___'
}

ratio_to_target() {
    python3 -c "print(max(1, round($1 * $2)))"
}

count_existing_items() {
    local f="${1}/extra_text.jsonl"
    [ -f "$f" ] && grep -cve '^\s*$' "$f" 2>/dev/null || echo 0
}

download_flag() {
    [ "${DOWNLOAD_IMAGES}" = "1" ] && echo "--download_images" || echo ""
}

geo_lookup_flag() {
    [ -n "${GEO_LOOKUP_PATH}" ] && [ -f "${GEO_LOOKUP_PATH}" ] && echo "--geo_lookup_path ${GEO_LOOKUP_PATH}" || echo ""
}

# ──────────────────────────────────────────────
# Per-category crawl (runs as background job)
# ──────────────────────────────────────────────
crawl_category() {
    local entry="$1"
    IFS='|' read -r category ratio query_list <<< "$entry"

    local total_cat_target safe_category output_dir log_file
    total_cat_target=$(ratio_to_target "$ratio" "$TOTAL_ITEMS")
    safe_category=$(sanitize_name "$category")
    output_dir="${BASE_OUTPUT_DIR}/extra_data_${safe_category}"
    log_file="${LOG_DIR}/crawl_${safe_category}.log"

    mkdir -p "$output_dir"

    {
        echo "========================================"
        echo "Category    : $category  (ratio=${ratio})"
        echo "Total target: $total_cat_target"
        echo "Output      : $output_dir"
        echo "Started at  : $(date '+%Y-%m-%d %H:%M:%S')"
        echo "========================================"

        # Compute geo / no-geo split
        local geo_target no_geo_target
        geo_target=$(python3 -c "print(max(0, round($GEO_RATIO * $total_cat_target)))")
        no_geo_target=$((total_cat_target - geo_target))

        IFS=',' read -r -a kws <<< "$query_list"
        local num_kws=${#kws[@]}
        local per_kw_quota=$((no_geo_target / num_kws))

        echo "no-geo target : $no_geo_target  (${num_kws} kws × ~${per_kw_quota} each)"
        echo "geo target    : $geo_target"

        local dl_flag geo_flag
        dl_flag=$(download_flag)
        geo_flag=$(geo_lookup_flag)

        # --------------------------------------------------
        # Phase 1: no-geo, one keyword at a time
        # Uses cumulative target: if previous keywords already
        # filled the quota, later keywords are skipped.
        # --------------------------------------------------
        local idx=0
        for kw in "${kws[@]}"; do
            idx=$((idx + 1))
            local cumulative_target
            cumulative_target=$((per_kw_quota * idx))
            [ "$cumulative_target" -gt "$no_geo_target" ] && cumulative_target="$no_geo_target"

            local existing
            existing=$(count_existing_items "$output_dir")

            if [ "$existing" -ge "$cumulative_target" ]; then
                echo "[$idx/$num_kws] Skip [no-geo] $kw  (existing=$existing >= cumulative=$cumulative_target)"
                continue
            fi

            echo "[$idx/$num_kws] [no-geo] $kw  existing=$existing  cumulative_target=$cumulative_target"

            # shellcheck disable=SC2086
            "$PYTHON_BIN" "$CRAWLER_SCRIPT" \
                --output_dir          "$output_dir"          \
                --text                "$kw"                  \
                --seed_category       "$category"            \
                --max_items           "$cumulative_target"   \
                --licenses            "$LICENSES"            \
                --sleep_min           "$SLEEP_MIN"           \
                --sleep_max           "$SLEEP_MAX"           \
                --flush_every         "$FLUSH_EVERY"         \
                --per_page            "$PER_PAGE"            \
                --sort                "$SORT"                \
                --max_no_new_pages    "$MAX_NO_NEW_PAGES"    \
                --min_upload_date     "$DATE_START"          \
                --max_upload_date     "$DATE_END"            \
                --max_photos_per_user "$MAX_PHOTOS_PER_USER" \
                --has_geo             0                      \
                --resume                                     \
                --dedupe_on_image_path                       \
                $dl_flag $geo_flag

            echo "[$idx/$num_kws] Done: $(count_existing_items "$output_dir") items"
        done

        # --------------------------------------------------
        # Phase 2: geo (has_geo=1), first keyword only
        # --------------------------------------------------
        if [ "$geo_target" -gt 0 ]; then
            local existing
            existing=$(count_existing_items "$output_dir")

            if [ "$existing" -ge "$total_cat_target" ]; then
                echo "[geo] Skip: already $existing >= $total_cat_target items"
            else
                local geo_kw="${kws[0]}"
                echo "[geo] $geo_kw  existing=$existing  geo_target=$geo_target  total_target=$total_cat_target"

                # shellcheck disable=SC2086
                "$PYTHON_BIN" "$CRAWLER_SCRIPT" \
                    --output_dir          "$output_dir"          \
                    --text                "$geo_kw"              \
                    --seed_category       "$category"            \
                    --max_items           "$total_cat_target"    \
                    --licenses            "$LICENSES"            \
                    --sleep_min           "$SLEEP_MIN"           \
                    --sleep_max           "$SLEEP_MAX"           \
                    --flush_every         "$FLUSH_EVERY"         \
                    --per_page            "$PER_PAGE"            \
                    --sort                "$SORT"                \
                    --max_no_new_pages    "$MAX_NO_NEW_PAGES"    \
                    --min_upload_date     "$DATE_START"          \
                    --max_upload_date     "$DATE_END"            \
                    --max_photos_per_user "$MAX_PHOTOS_PER_USER" \
                    --has_geo             1                      \
                    --resume                                     \
                    --dedupe_on_image_path                       \
                    $dl_flag $geo_flag
            fi
        fi

        echo ""
        echo "Finished  : $category"
        echo "Final items: $(count_existing_items "$output_dir") / $total_cat_target"
        echo "Ended at  : $(date '+%Y-%m-%d %H:%M:%S')"

    } >> "$log_file" 2>&1
}

export -f crawl_category sanitize_name ratio_to_target count_existing_items download_flag geo_lookup_flag

# ──────────────────────────────────────────────
# Pre-flight checks
# ──────────────────────────────────────────────
if [ -z "${FLICKR_API_KEY:-}" ]; then
    echo "[ERROR] FLICKR_API_KEY is not set."
    echo "  Export it: export FLICKR_API_KEY=your_key_here"
    exit 1
fi

if [ ! -f "$CRAWLER_SCRIPT" ]; then
    echo "[ERROR] Crawler not found: $CRAWLER_SCRIPT"
    exit 1
fi

mkdir -p "$BASE_OUTPUT_DIR" "$LOG_DIR"

cat <<INFO
========================================
SMP Flickr category crawl (parallel)
========================================
Base output dir     : $BASE_OUTPUT_DIR
Log dir             : $LOG_DIR
Total target        : $TOTAL_ITEMS
Categories          : ${#CATEGORIES[@]}
Date range          : ${DATE_START} ~ ${DATE_END}
Sleep               : ${SLEEP_MIN}s ~ ${SLEEP_MAX}s
Max photos/user     : $MAX_PHOTOS_PER_USER
Geo ratio           : $GEO_RATIO
Download images     : $DOWNLOAD_IMAGES
Geo lookup file     : ${GEO_LOOKUP_PATH:-"(not set)"}
========================================

INFO

# ──────────────────────────────────────────────
# Run categories sequentially (one at a time)
# Avoids hitting Flickr rate limit with parallel requests on a single API key
# ──────────────────────────────────────────────
echo ""
echo "All ${#CATEGORIES[@]} categories will run sequentially."
echo "Monitor progress:"
echo "  tail -f ${LOG_DIR}/crawl_Animal.log"
echo "  tail -n 5 ${LOG_DIR}/crawl_*.log"
echo ""

FAILED=()
for entry in "${CATEGORIES[@]}"; do
    IFS='|' read -r category _ _ <<< "$entry"
    safe=$(sanitize_name "$category")
    echo "Starting: $category  →  logs/${safe}.log"
    crawl_category "$entry"
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "Done: $category"
    else
        echo "Failed: $category (exit code $exit_code)"
        FAILED+=("$category")
    fi
done

echo ""
echo "========================================"
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "All categories finished successfully."
else
    echo "Failed categories:"
    for c in "${FAILED[@]}"; do echo "  - $c"; done
fi
echo "========================================"
