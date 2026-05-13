#!/usr/bin/env bash
# run_smp_category_crawl_v4.sh
#
# V4: clean rewrite
#   - Parallel execution (all categories simultaneously)
#   - Full license range 1-10
#   - Expanded keywords per category
#   - Travel skipped (already done)
#   - resume support (continues from existing data)
#
# Usage:
#   ./run_smp_category_crawl_v4.sh
#
# Monitor:
#   tail -f /local/smp/logs/crawl_Animal.log
#   tail -n 3 /local/smp/logs/crawl_*.log
#
# Dry run:
#   TOTAL_ITEMS=5000 DOWNLOAD_IMAGES=0 TIME_SLICE=0 ./run_smp_category_crawl_v4.sh

set -euo pipefail

# ──────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────
PYTHON_BIN="${PYTHON_BIN:-python3}"
CRAWLER_SCRIPT="${CRAWLER_SCRIPT:-./crawl_flickr_to_smp.py}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-/local/smp/extra_data}"
LOG_DIR="${LOG_DIR:-/local/smp/logs}"
TOTAL_ITEMS="${TOTAL_ITEMS:-480000}"
LICENSES="${LICENSES:-1,2,3,4,5,6,7,8,9,10}"
SLEEP_MIN="${SLEEP_MIN:-1.5}"
SLEEP_MAX="${SLEEP_MAX:-3.0}"
FLUSH_EVERY="${FLUSH_EVERY:-200}"
PER_PAGE="${PER_PAGE:-100}"
SORT="${SORT:-date-posted-desc}"
MAX_NO_NEW_PAGES="${MAX_NO_NEW_PAGES:-20}"
DOWNLOAD_IMAGES="${DOWNLOAD_IMAGES:-1}"
TIME_SLICE="${TIME_SLICE:-1}"
DATE_START="${DATE_START:-2004-01-01}"
DATE_END="${DATE_END:-2024-12-31}"
INITIAL_WINDOW_DAYS="${INITIAL_WINDOW_DAYS:-180}"
MAX_PHOTOS_PER_USER="${MAX_PHOTOS_PER_USER:-5}"
FETCH_PHOTO_DETAIL="${FETCH_PHOTO_DETAIL:-0}"

# ──────────────────────────────────────────────────────
# Categories (Travel skipped — already at 147%)
# Format: "Category|primary_query|extra_queries|base_target_at_100k"
# ──────────────────────────────────────────────────────
CATEGORIES=(
    "Holiday&Celebrations|holiday|christmas,newyear,halloween,parade,carnival,fireworks,graduation,thanksgiving,easter,anniversary,diwali,hanukkah,ramadan,lunarnewyear,oktoberfest,mardigras,stpatrick,valentines,mothersday,fathersday,independence,reunion,prom,quinceanera,baptism,retirement,ceremony,celebration,festive,lantern,confetti,costume,masquerade,bonfire,wreath,decoration,ornament,champagne,toast,ribbon,candle,balloon,gift,procession,folklore,tradition,ritual|10790"

    "Animal|animal|cat,dog,wildlife,bird,pet,zoo,fish,horse,insect,butterfly,marine,elephant,deer,rabbit,lion,tiger,bear,wolf,fox,owl,penguin,crocodile,snake,parrot,hamster,turtle,whale,dolphin,shark,monkey,kitten,puppy,eagle,falcon,gecko,lizard,crab,coral,flamingo,peacock,panda,koala,hedgehog,goldfish,dragonfly,frog,squirrel,raccoon,otter,seal,walrus,moose,bison,leopard,cheetah,gorilla,chimpanzee,macaw,toucan,heron,pelican|10230"

    "Entertainment|concert|music,performance,dance,show,stage,theater,band,guitar,piano,drama,cinema,comedy,magic,circus,festival,dj,hiphop,jazz,opera,ballet,puppet,standup,esports,boardgame,cosplay,anime,convention,karaoke,busking,musician,singer,drummer,saxophone,violin,turntable,microphone,spotlight,audience,applause,backstage,rehearsal,choreography,acrobat,juggling,cabaret,nightclub,rave,acoustic,orchestra|9950"

    "Fashion|fashion|style,outfit,model,clothing,dress,shoes,bag,makeup,beauty,hair,runway,jewelry,sunglasses,menswear,streetwear,vintage,luxury,accessory,lingerie,swimwear,sneakers,coat,suit,scarf,watch,perfume,editorial,lookbook,designer,couture,haute,vogue,catwalk,trend,wardrobe,boutique,tailor,embroidery,textile,fabric,pattern,denim,leather,silk,velvet,lace,knit,hoodie,blazer,heels,boots,handbag,necklace,bracelet,earring,lipstick,eyeliner,nail,skincare|9950"

    "Whether&Season|winter|snow,rain,summer,autumn,fog,spring,cloud,storm,sunset,sunrise,flower,leaves,ice,weather,rainbow,lightning,hail,heatwave,blizzard,drizzle,mist,dewdrop,frost,puddle,overcast,tornado,typhoon,monsoon,drought,sunshine,breeze,thunder,haze,sleet,avalanche,dusk,dawn,twilight,solstice,petal,blossom,sakura,maple,willow,poppy,sunflower,lavender,tulip,snowflake,icicle,frozen,flood,hurricane|8270"

    "Social&People|people|portrait,street,crowd,friends,community,selfie,smile,group,event,volunteers,protest,gathering,wedding,elderly,teenager,student,worker,musician,athlete,activist,journalist,doctor,soldier,farmer,fisherman,chef,couple,children,woman,man,senior,youth,culture,diversity,lifestyle,candid,expression,emotion,laughter,tears,hug,handshake,conversation,market,neighborhood,village,rural,migration,refugee,celebration|8000"

    "Urban|city|architecture,skyline,building,urban,night,bridge,road,traffic,downtown,skyscraper,neon,station,market,alley,graffiti,subway,rooftop,courtyard,harbor,suburb,chinatown,mosque,cathedral,temple,museum,library,plaza,park,construction,demolition,renovation,facade,mural,fountain,escalator,corridor,lobby,staircase,ruin,industrial,warehouse,factory,tower,crane,scaffold,boulevard,intersection,crosswalk,lamppost,signage,billboard,storefront,district,quarter|6650"

    "Food|food|meal,restaurant,cooking,coffee,dessert,lunch,dinner,breakfast,cake,drink,tea,bread,fruit,chef,sushi,pizza,burger,ramen,pasta,tacos,curry,bbq,icecream,chocolate,smoothie,salad,streetfood,bakery,buffet,vegan,noodle,dumpling,stew,soup,grill,roast,steak,seafood,vegetable,spice,herb,sauce,dough,batter,ferment,pickle,cheese,wine,beer,cocktail,latte,espresso,matcha,boba,crepe,waffle,pudding,tart,macaron|6550"

    "Electronics|electronics|technology,gadget,phone,computer,camera,laptop,tablet,headphones,robot,drone,gaming,console,keyboard,screen,device,smartwatch,speaker,charger,sensor,circuit,server,coding,programming,3dprinting,vr,ar,iot,dashcam,semiconductor,processor,motherboard,gpu,microchip,soldering,oscilloscope,telescope,microscope,radar,satellite,antenna,fiber,battery,solar,electric,hybrid,autonomous,interface,hologram,wearable,biometric,scanner|3320"

    "Family|family|baby,children,parenting,together,kids,mother,father,child,parents,house,newborn,toddler,grandparents,siblings,playdate,schoolbus,lunchbox,bedtime,bathtime,stroller,nursery,adoption,babysitter,kindergarten,homework,birthday,pregnant,maternity,breastfeeding,cradle,crib,diaper,feeding,playground,sandbox,swing,picnic,camping,roadtrip,reunion,portrait,home,kitchen,garden,backyard,livingroom,bedroom,twins,cousins,grandchild,caregiver|3000"
)

# ──────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────
sanitize_name() {
    echo "$1" | tr '& /' '___'
}

scale_target() {
    python3 -c "print(max(1, round($1 * $TOTAL_ITEMS / 100000.0)))"
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
    local f="${1}/extra_text.jsonl"
    [ -f "$f" ] && grep -cve '^\s*$' "$f" 2>/dev/null || echo 0
}

# ──────────────────────────────────────────────────────
# Per-category crawl (runs in background)
# ──────────────────────────────────────────────────────
crawl_category() {
    local entry="$1"
    IFS='|' read -r category primary_query extra_queries base_target <<< "$entry"

    local target safe_category output_dir log_file
    target=$(scale_target "$base_target")
    safe_category=$(sanitize_name "$category")
    output_dir="${BASE_OUTPUT_DIR}/extra_data_${safe_category}"
    log_file="${LOG_DIR}/crawl_${safe_category}.log"

    mkdir -p "$output_dir"

    {
        echo "========================================"
        echo "Category    : $category"
        echo "Target      : $target"
        echo "Output      : $output_dir"
        echo "Licenses    : $LICENSES"
        echo "Started at  : $(date '+%Y-%m-%d %H:%M:%S')"
        echo "========================================"

        local all_queries="${primary_query},${extra_queries}"
        IFS=',' read -ra query_list <<< "$all_queries"
        local num_queries=${#query_list[@]}
        local per_query_target
        per_query_target=$(python3 -c "import math; print(math.ceil($target / $num_queries))")

        echo "Query count     : $num_queries"
        echo "Per-query quota : $per_query_target"
        echo ""

        local dl_flag ts_flag dt_flag
        dl_flag=$(download_flag)
        ts_flag=$(time_slice_flag)
        dt_flag=$(fetch_detail_flag)

        local idx=0
        for q in "${query_list[@]}"; do
            q="$(echo "$q" | xargs)"
            [ -z "$q" ] && continue

            idx=$((idx + 1))
            local current_target=$((per_query_target * idx))
            [ "$current_target" -gt "$target" ] && current_target="$target"

            local existing
            existing=$(count_existing_items "$output_dir")

            if [ "$existing" -ge "$current_target" ]; then
                echo "[$idx/$num_queries] Skip: $q (existing=$existing >= cumulative=$current_target)"
                continue
            fi

            echo "[$idx/$num_queries] Query: $q | existing=$existing | cumulative_target=$current_target"

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

            echo "[$idx/$num_queries] After: $(count_existing_items "$output_dir") items"
            echo ""
        done

        local final_count
        final_count=$(count_existing_items "$output_dir")
        echo "Finished: $category"
        echo "Final: $final_count / $target"
        echo "Ended at: $(date '+%Y-%m-%d %H:%M:%S')"

    } >> "$log_file" 2>&1
}

export -f crawl_category sanitize_name scale_target \
           download_flag time_slice_flag fetch_detail_flag \
           count_existing_items

# ──────────────────────────────────────────────────────
# Pre-flight
# ──────────────────────────────────────────────────────
if [ -z "${FLICKR_API_KEY:-}" ]; then
    echo "[ERROR] FLICKR_API_KEY is not set."
    exit 1
fi
if [ ! -f "$CRAWLER_SCRIPT" ]; then
    echo "[ERROR] Crawler not found: $CRAWLER_SCRIPT"
    exit 1
fi

mkdir -p "$BASE_OUTPUT_DIR" "$LOG_DIR"

cat <<INFO
========================================
SMP Flickr category crawl V4
========================================
Base output dir     : $BASE_OUTPUT_DIR
Log dir             : $LOG_DIR
Total target        : $TOTAL_ITEMS
Categories          : ${#CATEGORIES[@]} (Travel skipped)
Licenses            : $LICENSES
Sleep               : ${SLEEP_MIN}s ~ ${SLEEP_MAX}s
Max photos/user     : $MAX_PHOTOS_PER_USER
Download images     : $DOWNLOAD_IMAGES
Time slice          : $TIME_SLICE
========================================
INFO

# ──────────────────────────────────────────────────────
# Launch all categories in parallel
# ──────────────────────────────────────────────────────
PIDS=()

for entry in "${CATEGORIES[@]}"; do
    IFS='|' read -r category _ _ _ <<< "$entry"
    safe=$(echo "$category" | tr '& /' '___')
    echo "Launching: $category  →  ${LOG_DIR}/crawl_${safe}.log"
    crawl_category "$entry" &
    PIDS+=($!)
done

echo ""
echo "All ${#CATEGORIES[@]} categories launched in parallel."
echo "Monitor: tail -f ${LOG_DIR}/crawl_*.log"
echo "Status:  python3 crawl_status.py --base_dir ${BASE_OUTPUT_DIR} --skip_images"
echo ""

# ──────────────────────────────────────────────────────
# Wait and report
# ──────────────────────────────────────────────────────
FAILED=()
for i in "${!PIDS[@]}"; do
    pid=${PIDS[$i]}
    entry=${CATEGORIES[$i]}
    IFS='|' read -r category _ _ _ <<< "$entry"
    if wait "$pid"; then
        echo "✅  Done: $category"
    else
        echo "❌  Failed: $category (exit=$?)"
        FAILED+=("$category")
    fi
done

echo ""
echo "========================================"
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "All categories finished successfully."
else
    echo "Failed:"
    for c in "${FAILED[@]}"; do echo "  - $c"; done
fi
echo "========================================"
