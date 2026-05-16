#!/usr/bin/env bash
# run_smp_category_crawl_v5.sh
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
CRAWLER_SCRIPT="${CRAWLER_SCRIPT:-./crawl_flickr_to_smp.py}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-/local/smp/extra_data_v2}"
LOG_DIR="${LOG_DIR:-/local/smp/extra_data_v2/logs}"
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
DATE_START="${DATE_START:-2000-01-01}"
DATE_END="${DATE_END:-2024-12-31}"
INITIAL_WINDOW_DAYS="${INITIAL_WINDOW_DAYS:-180}"
MAX_PHOTOS_PER_USER="${MAX_PHOTOS_PER_USER:-0}"
FETCH_PHOTO_DETAIL="${FETCH_PHOTO_DETAIL:-0}"

CATEGORIES=(
    "Travel&Active&Sports|travel|hiking,landscape,mountain,backpacking,adventure,tourism,landmark,camping,cycling,running,skiing,surfing,trekking,climbing,kayaking,roadtrip,nationalpark,waterfall,canyon,glacier,safari,scuba,marathon,triathlon,motorbike,sailing,paragliding,crossfit,yoga,volleyball,skateboarding,football,soccer,hockey,fitness,sport,sports,car,cars,action,beach,vacation,bike|25180"
    "Holiday&Celebrations|holiday|christmas,halloween,wedding,party,pumpkin,event,merrychristmas,xmas,bride,santa,ornaments,happynewyear,jackolantern,santaclaus,trickortreat,pumpkinpatch,hauntedhouse,christmastree,newyearseve,carnival,fireworks,graduation,thanksgiving,easter,anniversary,diwali,lunarnewyear,oktoberfest,valentines,mothersday,independence,ceremony,celebration,festive,lantern,confetti,costume,bonfire,decoration,balloon,gift,tradition,ritual|10790"
    "Animal|animal|nature,fish,water,dog,horse,cat,aquarium,tropical,freshwater,fishtank,coral,ocean,reef,wildlife,pets,underwater,diving,tropicalfish,macro,hound,pony,furry,doglover,catlover,kitten,puppy,eagle,lizard,flamingo,peacock,panda,koala,hedgehog,dragonfly,squirrel,raccoon,otter,seal,leopard,cheetah,gorilla,macaw,heron,pelican,saltwater,naturelover,wild,birdwatching,safari|10230"
    "Entertainment|concert|music,hiphop,show,festival,remix,gamer,cosplay,film,cinema,party,rap,movies,beats,celebrity,book,dubstep,melody,live,artist,theater,band,guitar,piano,drama,comedy,magic,circus,dj,jazz,opera,ballet,standup,esports,boardgame,anime,convention,karaoke,busking,musician,singer,drummer,saxophone,violin,spotlight,audience,backstage,choreography,nightclub,orchestra|9950"
    "Fashion|fashion|hair,beauty,model,makeup,style,tattoo,hairstyle,outfit,tattoos,nails,shoes,haircut,nike,bodyart,dress,ink,inked,haircolor,sneakers,tattooed,hairstyles,streetwear,vintage,luxury,accessory,lingerie,swimwear,coat,suit,scarf,perfume,lookbook,designer,couture,catwalk,wardrobe,boutique,embroidery,textile,denim,leather,silk,velvet,lace,hoodie,blazer,heels,boots,handbag,necklace,lipstick|9950"
    "Whether&Season|winter|snow,clouds,rain,landscape,sunset,summer,snowfall,weather,cloud,raining,rainyday,downpour,umbrella,autumn,pouring,fog,spring,storm,sunrise,rainbow,lightning,hail,heatwave,blizzard,drizzle,mist,dewdrop,frost,overcast,tornado,typhoon,monsoon,drought,sunshine,thunder,haze,sleet,avalanche,dusk,dawn,twilight,sakura,maple,sunflower,lavender,snowflake,icicle,flood|8270"
    "Social&People|people|love,wedding,girl,friends,smile,fun,happy,portrait,mother,women,bride,groom,bridesmaid,dancing,happiness,dance,couple,bridesmaids,volunteers,protest,gathering,elderly,teenager,student,worker,athlete,activist,journalist,doctor,soldier,farmer,chef,children,senior,youth,culture,diversity,lifestyle,candid,emotion,laughter,tears,hug,handshake,neighborhood,community,selfie,crowd|8000"
    "Food|food|tea,coffee,caffeine,foodporn,delicious,beer,foodpics,desserts,drink,foodie,tasty,lunch,dessert,restaurant,dinner,eat,foodphotography,chocolate,cocktail,cafe,sweet,yummy,breakfast,drinks,sushi,pizza,burger,ramen,pasta,tacos,curry,bbq,icecream,smoothie,salad,streetfood,bakery,buffet,vegan,noodle,dumpling,stew,soup,grill,steak,seafood,latte,espresso,matcha,boba,crepe,waffle|6550"
    "Electronics|electronics|gadget,samsung,galaxy,technology,tech,iphone,computer,screen,mobile,smartphone,android,apple,macbook,dell,laptops,ipod,intel,camera,drone,gaming,console,keyboard,smartwatch,speaker,sensor,circuit,server,coding,programming,3dprinting,vr,ar,iot,semiconductor,processor,gpu,microchip,soldering,radar,satellite,antenna,battery,solar,electric,wearable,biometric,scanner,robot,headphones|3320"
    "Family|family|cute,baby,infant,child,newborn,portrait,love,dance,boy,girl,son,party,happiness,adorable,pets,pink,babies,kawaii,happy,children,parenting,together,kids,mother,father,parents,house,toddler,grandparents,siblings,playdate,schoolbus,bedtime,bathtime,stroller,nursery,kindergarten,homework,birthday,pregnant,maternity,playground,sandbox,swing,picnic,camping,roadtrip,reunion,backyard|3000"
)

sanitize_name()   { echo "$1" | tr '& /' '___'; }
scale_target()    { python3 -c "print(max(1, round($1 * $TOTAL_ITEMS / 100000.0)))"; }
download_flag()   { [ "${DOWNLOAD_IMAGES}" = "1" ] && echo "--download_images" || echo ""; }
time_slice_flag() { [ "${TIME_SLICE}" = "1" ] && echo "--time_slice" || echo ""; }
fetch_detail_flag() { [ "${FETCH_PHOTO_DETAIL}" = "1" ] && echo "--fetch_photo_detail" || echo ""; }
count_existing_items() {
    local f="${1}/extra_text.jsonl"
    [ -f "$f" ] && grep -cve '^\s*$' "$f" 2>/dev/null || echo 0
}

crawl_category() {
    local entry="$1"
    IFS='|' read -r category primary_query extra_queries base_target <<< "$entry"
    local target safe_category output_dir log_file
    target=$(scale_target "$base_target")
    safe_category=$(sanitize_name "$category")
    output_dir="${BASE_OUTPUT_DIR}/extra_data_${safe_category}"
    log_file="${LOG_DIR}/crawl_v5_${safe_category}.log"
    mkdir -p "$output_dir"
    {
        echo "=== $category | target=$target | sort=$SORT | $(date '+%Y-%m-%d %H:%M:%S') ==="
        local all_queries="${primary_query},${extra_queries}"
        IFS=',' read -ra query_list <<< "$all_queries"
        local num_queries=${#query_list[@]}
        local per_query_target
        per_query_target=$(python3 -c "import math; print(math.ceil($target / $num_queries))")
        local dl_flag ts_flag dt_flag
        dl_flag=$(download_flag); ts_flag=$(time_slice_flag); dt_flag=$(fetch_detail_flag)
        local idx=0
        for q in "${query_list[@]}"; do
            q="$(echo "$q" | xargs)"
            [ -z "$q" ] && continue
            idx=$((idx + 1))
            local current_target=$((per_query_target * idx))
            [ "$current_target" -gt "$target" ] && current_target="$target"
            local existing; existing=$(count_existing_items "$output_dir")
            if [ "$existing" -ge "$current_target" ]; then
                echo "[$idx/$num_queries] Skip: $q ($existing >= $current_target)"
                continue
            fi
            echo "[$idx/$num_queries] $q | existing=$existing | target=$current_target"
            "$PYTHON_BIN" "$CRAWLER_SCRIPT" \
                --output_dir "$output_dir" --text "$q" --max_items "$current_target" \
                --licenses "$LICENSES" --sort "$SORT" \
                --sleep_min "$SLEEP_MIN" --sleep_max "$SLEEP_MAX" \
                --flush_every "$FLUSH_EVERY" --per_page "$PER_PAGE" \
                --max_no_new_pages "$MAX_NO_NEW_PAGES" \
                --seed_category "$category" \
                --date_start "$DATE_START" --date_end "$DATE_END" \
                --initial_window_days "$INITIAL_WINDOW_DAYS" \
                --max_photos_per_user "$MAX_PHOTOS_PER_USER" \
                --resume --dedupe_on_image_path \
                ${dl_flag} ${ts_flag} ${dt_flag}
            echo "[$idx/$num_queries] After: $(count_existing_items "$output_dir")"
        done
        echo "Done: $category | Final: $(count_existing_items "$output_dir") / $target | $(date '+%Y-%m-%d %H:%M:%S')"
    } >> "$log_file" 2>&1
}

export -f crawl_category sanitize_name scale_target \
           download_flag time_slice_flag fetch_detail_flag count_existing_items

[ -z "${FLICKR_API_KEY:-}" ] && { echo "[ERROR] FLICKR_API_KEY not set"; exit 1; }
[ ! -f "$CRAWLER_SCRIPT" ]   && { echo "[ERROR] Crawler not found: $CRAWLER_SCRIPT"; exit 1; }
mkdir -p "$BASE_OUTPUT_DIR" "$LOG_DIR"

echo "=== V5: sort=$SORT | dates=$DATE_START~$DATE_END | categories=${#CATEGORIES[@]} ==="

PIDS=()
for entry in "${CATEGORIES[@]}"; do
    IFS='|' read -r category _ _ _ <<< "$entry"
    safe=$(echo "$category" | tr '& /' '___')
    echo "Launching: $category"
    crawl_category "$entry" &
    PIDS+=($!)
done

FAILED=()
for i in "${!PIDS[@]}"; do
    pid=${PIDS[$i]}; entry=${CATEGORIES[$i]}
    IFS='|' read -r category _ _ _ <<< "$entry"
    if wait "$pid"; then echo "✅ $category"
    else echo "❌ $category"; FAILED+=("$category")
    fi
done

[ ${#FAILED[@]} -eq 0 ] && echo "All done." || echo "Failed: ${FAILED[*]}"
