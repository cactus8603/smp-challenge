## try to get more keywords from official dataset

import json
import os

# 1. 設定路徑（請根據你的 Server 實際狀況微調）
base_path = '/ssd1/lchiayu/smp-challenge/data/official_data'
files = [
    f"{base_path}/train_set/train_allmetadata_json/train_category.json",
    f"{base_path}/test/test_allmetadata_json/test_category.json" # 假設 test 的路徑結構一致
]

# 2. 準備存放集合
mined_data = {}

def process_file(file_path):
    if not os.path.exists(file_path):
        print(f"警告：找不到檔案 {file_path}")
        return
    
    with open(file_path, 'r', encoding='utf-8') as f:
        # 如果 JSON 是 list 格式直接 load，如果是 jsonl 則需逐行讀取
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            data = [json.loads(line) for line in f]

        for item in data:
            cat = item.get('Category')
            if not cat: continue
            
            if cat not in mined_data:
                mined_data[cat] = set()
            
            # 抓取 Subcategory 和 Concept
            for key in ['Subcategory', 'Concept']:
                val = item.get(key)
                if val and val.lower() != 'none':
                    # 清洗：轉小寫、去空格，並只保留單字或短詞
                    clean_val = val.lower().strip()
                    mined_data[cat].add(clean_val)

# 執行讀取
for f in files:
    print(f"正在處理: {f}")
    process_file(f)

# 3. 輸出成你原本 Python 腳本需要的格式
print("\n" + "="*30)
print("提取完成！請將以下內容更新至 _SMP_CATEGORY_KEYWORDS：")
print("="*30 + "\n")

final_dict = {}
for cat, keywords in mined_data.items():
    # 將 set 轉回 list 並排序，方便閱讀
    final_dict[cat] = sorted(list(keywords))

print(json.dumps(final_dict, indent=4, ensure_ascii=False))

# [Fashion] 挖掘到的關鍵字清單：
# ['hairstyle', 'accessory', 'black', 'tall', 'tatts', 'gorgeous', 'tshirt', 'hairfashion', 'highheels', 'branco', 'powder', 'blonde', 'shoe', 'unhas', 'tats', 'clothes', 'whatiwore', 'gem', 'longhair', 'bracelet']
# ------------------------------
# [Travel&Active&Sports] 挖掘到的關鍵字清單：
# ['health', 'bat', 'court', 'marathon', 'trailrunner', 'sanfrancsico', 'quarterback', 'squat', 'fight', 'shredded', 'bronco', 'mitt', 'baltimore', 'dedication', 'icerink', 'highway', 'car', 'practice', 'driver', 'denverbroncos']
# ------------------------------
# [Entertainment] 挖掘到的關鍵字清單：
# ['kindle', 'plot', 'book', 'flicks', 'online', 'stories', 'author', 'music', 'nook', 'beat', 'videos', 'actor', 'repeat', 'moviestar', 'video&games', 'words', 'films', 'listentothis', 'hiphop', 'pages']
# ------------------------------
# [Holiday&Celebrations] 挖掘到的關鍵字清單：
# ['partying', 'congratulations', 'newyear', 'weddinggown', 'scary', 'hauntedhouse', 'new', 'xmas', 'jolly', 'parties', 'gifts', 'bride', 'unforgettable', 'thanksgiving', 'pumpkins', 'wedding', 'gift', 'christmas', 'carving', 'merrychristmas']
# ------------------------------
# [Food] 挖掘到的關鍵字清單：
# ['dessert', 'pub', 'dinner', 'bar', 'desserts', 'delish', 'thirst', 'beers', 'delicious', 'liquor', 'general', 'amazing', 'hungry', 'foodpics', 'teacup', 'drinks', 'beer', 'caffeine', 'wine', 'thirsty']
# ------------------------------
# [Whether&Season] 挖掘到的關鍵字清單：
# ['downpour', 'trees', 'frost', 'summertime', 'chilly', 'snowing', 'clearskies', 'cloud', 'leaves', 'fall', 'rainyday', 'springtime', 'beautifulday', 'overcast', 'holidayseason', 'horizon', 'skyporn', 'umbrella', 'bright', 'color']
# ------------------------------
# [Animal] 挖掘到的關鍵字清單：
# ['doglover', 'wildlife', 'cats', 'ponies', 'bugs', 'insects', 'pup', 'lovenature', 'dog', 'aquaria', 'fish', 'fishtank', 'farm', 'dogsitting', 'lovecats', 'hound', 'mane', 'bug', 'macrophotography', 'dogs']
# ------------------------------
# [Family] 挖掘到的關鍵字清單：
# ['cuddle', 'small', 'cuddly', 'kids', 'infant', 'toddler', 'child', 'lovely', 'tiny', 'babies']
# ------------------------------
# [Social&People] 挖掘到的關鍵字清單：
# ['ready', 'goingout', 'love', 'lady', 'knockout', 'snooze', 'kisses', 'morning', 'wakeup', 'couple', 'good,morning', 'early', 'sunrise', 'girls', 'boyfriend', 'lightsout', 'bestfriend', 'girlfriend', 'portrait', 'selfie']
# ------------------------------
# [Urban] 挖掘到的關鍵字清單：
# ['abstract', 'streetarteverywhere', 'streetart', 'archidaily', 'skyscraper', 'urbanart', 'graffiti', 'architecture', 'pattern', 'artwork', 'composition', 'architecturelovers', 'lines', 'pasteup', 'arts', 'city', 'minimal', 'wall', 'building', 'stencil']
# ------------------------------
# [Electronics] 挖掘到的關鍵字清單：
# ['laptops', 'samsunggalaxy', 'iphone', 'phone', 'device', 'smartphone', 'hack', 'computers', 'gadget', 'android', 'gadgets', 'mobile', 'electronic', 'screen', 'electronics']
# ------------------------------