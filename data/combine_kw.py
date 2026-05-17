import json

# 1. 你原本的舊字典
old_keywords = {
    "Travel&Active&Sports": ["travel", "trip", "vacation", "journey", "tourism", "tourist", "abroad", "explore", "backpacking", "adventure", "wanderlust", "sightseeing", "itinerary", "flight", "passport", "hotel", "resort", "holiday",
    "travel-photography", "solo-travel", "road-trip", "travel-blog", "travel-diary", "budget-travel", "luxury-travel", "island", "cruise", "airport", "train-journey", "destination", "world-tour", "getaway", "travel-life","sport", "sports",
    "running", "football", "soccer", "basketball", "baseball", "cycling", "swimming", "tennis", "gym", "fitness", "athlete", "competition", "training", "workout", "match", "team", "active",
    "marathon", "jogging", "yoga", "weightlifting", "crossfit", "exercise", "stadium", "coach", "tournament", "outdoor-sports", "indoor-sports"],

    "Animal": ["animal", "dog", "cat", "bird", "wildlife", "pet", "horse", "fish", "zoo", "insect", "butterfly", "mammal", "puppy", "kitten", "safari", "forest-animals", "farm", "domestic", "nature-photography",
    "wild-animals", "animal-portrait", "animal-behavior", "pet-photography", "exotic-animals", "reptile", "amphibian", "marine-life", "underwater", "birdwatching", "animal-closeup", "cute-animals", "endangered-species"],

    "Food": ["food", "restaurant", "cooking", "recipe", "meal", "dinner", "lunch", "breakfast", "cafe", "coffee", "eat", "delicious", "cuisine", "dessert", "snack", "gastronomy", "bakery", "street-food", "yummy",
    "food-photography", "home-cooking", "fine-dining", "fast-food", "vegan", "vegetarian", "grill", "bbq", "seafood", "homemade", "foodie", "brunch", "takeout", "food-blog"],

    "Urban": ["city", "urban", "street", "building", "architecture", "downtown", "skyline", "bridge", "night", "metropolis", "skyscraper", "pavement", "alley", "traffic", "construction", "modern", "landmark",
    "cityscape", "urban-life", "street-life", "neon", "city-lights", "public-transport", "subway", "bus", "intersection", "urban-exploration", "infrastructure"],

    "Whether&Season": ['downpour', 'trees', 'frost', 'summertime', 'chilly', 'snowing', 'clearskies', 'cloud', 'leaves', 'fall', 'rainyday', 'springtime', 'beautifulday', 'overcast', 'holidayseason', 'horizon', 'skyporn', 'umbrella', 'bright', 'color',
    'sunny', 'rain', 'snow', 'winter', 'autumn', 'summer', 'spring', 'windy', 'storm', 'fog', 'mist', 'rainbow', 'sunshine', 'blue-sky', 'weather', 'season', 'cold', 'hot', 'drizzle', 'thunderstorm'],

    "Fashion": ['hairstyle', 'accessory', 'black', 'tall', 'tatts', 'gorgeous', 'tshirt', 'hairfashion', 'highheels', 'branco', 'powder', 'blonde', 'shoe', 'unhas', 'tats', 'clothes', 'whatiwore', 'gem', 'longhair', 'bracelet',
    'fashion', 'style', 'outfit', 'ootd', 'streetstyle', 'model', 'runway', 'fashionblogger', 'stylish', 'casual', 'formal', 'vintage', 'denim', 'jewelry', 'makeup', 'fashionista', 'lookbook', 'trend', 'apparel', 'wardrobe'],

    "Entertainment": ['kindle', 'plot', 'book', 'flicks', 'online', 'stories', 'author', 'music', 'nook', 'beat', 'videos', 'actor', 'repeat', 'moviestar', 'video&games', 'words', 'films', 'listentothis', 'hiphop', 'pages',
    'movie', 'cinema', 'netflix', 'streaming', 'tv', 'series', 'showtime', 'gaming', 'videogame', 'gamer', 'concert', 'playlist', 'song', 'album', 'dj', 'pop', 'rock', 'entertainment', 'media', 'broadcast'],

    "Holiday&Celebrations": ['partying', 'congratulations', 'newyear', 'weddinggown', 'scary', 'hauntedhouse', 'new', 'xmas', 'jolly', 'parties', 'gifts', 'bride', 'unforgettable', 'thanksgiving', 'pumpkins', 'wedding', 'gift', 'christmas', 'carving', 'merrychristmas',
    'celebration', 'holiday', 'party', 'festival', 'birthdayparty', 'anniversary', 'fireworks', 'newyearseve', 'halloweenparty', 'christmasdecor', 'celebrate', 'specialday', 'event', 'festive', 'cheers', 'gathering', 'ceremony'],

    "Family": ['cuddle', 'small', 'cuddly', 'kids', 'infant', 'toddler', 'child', 'lovely', 'tiny', 'babies',
    'family', 'parents', 'mother', 'father', 'siblings', 'brother', 'sister', 'home', 'parenting', 'familytime', 'together', 'lovefamily', 'happyfamily', 'familylife', 'relatives'],

    "Social&People": ['ready', 'goingout', 'love', 'lady', 'knockout', 'snooze', 'kisses', 'morning', 'wakeup', 'couple', 'good,morning', 'early', 'sunrise', 'girls', 'boyfriend', 'lightsout', 'bestfriend', 'girlfriend', 'portrait', 'selfie',
    'friends', 'friendship', 'hangout', 'nightout', 'dating', 'relationship', 'smile', 'laugh', 'fun', 'partytime', 'group', 'social', 'people', 'lifestyle', 'weekend', 'vibes', 'togetherness'],

    "Electronics": ['laptops', 'samsunggalaxy', 'iphone', 'phone', 'device', 'smartphone', 'hack', 'computers', 'gadget', 'android', 'gadgets', 'mobile', 'electronic', 'screen', 'electronics',
    'tablet', 'pc', 'desktop', 'monitor', 'keyboard', 'mouse', 'tech', 'technology', 'wearable', 'smartwatch', 'charger', 'usb', 'headphones', 'earphones', 'camera', 'digital', 'device-tech'],
}


# 2. 從官方資料挖出的新關鍵字 (假設你已經存成變數)
mined_keywords = {
    "Travel&Active&Sports": ['health', 'bat', 'court', 'marathon', 'trailrunner', 'sanfrancsico', 'quarterback', 'squat', 'fight', 'shredded', 'bronco', 'mitt', 'baltimore', 'dedication', 'icerink', 'highway', 'car', 'practice', 'driver', 'denverbroncos'],
    "Animal":['doglover', 'wildlife', 'cats', 'ponies', 'bugs', 'insects', 'pup', 'lovenature', 'dog', 'aquaria', 'fish', 'fishtank', 'farm', 'dogsitting', 'lovecats', 'hound', 'mane', 'bug', 'macrophotography', 'dogs'],
    "Food":['dessert', 'pub', 'dinner', 'bar', 'desserts', 'delish', 'thirst', 'beers', 'delicious', 'liquor', 'general', 'amazing', 'hungry', 'foodpics', 'teacup', 'drinks', 'beer', 'caffeine', 'wine', 'thirsty'],
    "Urban":['abstract', 'streetarteverywhere', 'streetart', 'archidaily', 'skyscraper', 'urbanart', 'graffiti', 'architecture', 'pattern', 'artwork', 'composition', 'architecturelovers', 'lines', 'pasteup', 'arts', 'city', 'minimal', 'wall', 'building', 'stencil'],    
    "Whether&Season":['downpour', 'trees', 'frost', 'summertime', 'chilly', 'snowing', 'clearskies', 'cloud', 'leaves', 'fall', 'rainyday', 'springtime', 'beautifulday', 'overcast', 'holidayseason', 'horizon', 'skyporn', 'umbrella', 'bright', 'color'],
    "Fashion": ['hairstyle', 'accessory', 'black', 'tall', 'tatts', 'gorgeous', 'tshirt', 'hairfashion', 'highheels', 'branco', 'powder', 'blonde', 'shoe', 'unhas', 'tats', 'clothes', 'whatiwore', 'gem', 'longhair', 'bracelet'],
    "Entertainment":['kindle', 'plot', 'book', 'flicks', 'online', 'stories', 'author', 'music', 'nook', 'beat', 'videos', 'actor', 'repeat', 'moviestar', 'video&games', 'words', 'films', 'listentothis', 'hiphop', 'pages'],
    "Holiday&Celebrations":['partying', 'congratulations', 'newyear', 'weddinggown', 'scary', 'hauntedhouse', 'new', 'xmas', 'jolly', 'parties', 'gifts', 'bride', 'unforgettable', 'thanksgiving', 'pumpkins', 'wedding', 'gift', 'christmas', 'carving', 'merrychristmas'],
    "Family":['cuddle', 'small', 'cuddly', 'kids', 'infant', 'toddler', 'child', 'lovely', 'tiny', 'babies'],
    "Social&People":['ready', 'goingout', 'love', 'lady', 'knockout', 'snooze', 'kisses', 'morning', 'wakeup', 'couple', 'good,morning', 'early', 'sunrise', 'girls', 'boyfriend', 'lightsout', 'bestfriend', 'girlfriend', 'portrait', 'selfie'],
    "Electronics":['laptops', 'samsunggalaxy', 'iphone', 'phone', 'device', 'smartphone', 'hack', 'computers', 'gadget', 'android', 'gadgets', 'mobile', 'electronic', 'screen', 'electronics']
}
# 3. 合併邏輯
combined_keywords = {}

# 取得所有的類別名稱 (聯集)
all_categories = set(old_keywords.keys()) | set(mined_keywords.keys())

for cat in all_categories:
    # 取得舊的詞 (若無則空 list)
    old_list = old_keywords.get(cat, [])
    # 取得新的詞 (若無則空 list)
    new_list = mined_keywords.get(cat, [])
    
    # 利用 set 自動去重，並轉回 sorted list 保持美觀
    combined_keywords[cat] = sorted(list(set(old_list) | set(new_list)))

# 4. 印出結果，這就是可以直接貼回 Python 爬蟲的格式
#print(json.dumps(combined_keywords, indent=4, ensure_ascii=False))

# 假設 combined_keywords 是你合併好的字典
print("{")
for i, (cat, kws) in enumerate(combined_keywords.items()):
    # 將 list 拆成每 10 個一組
    formatted_kws = ""
    for j in range(0, len(kws), 10):
        chunk = kws[j:j+10]
        # 把這 10 個詞串起來，前後加引號
        line = ", ".join(f'"{kw}"' for kw in chunk)
        # 加上縮排與逗號
        formatted_kws += f"\n        {line}," if j + 10 < len(kws) else f"\n        {line}"
    
    # 處理最後一個類別不需要逗號的問題
    comma = "," if i < len(combined_keywords) - 1 else ""
    
    print(f'    "{cat}": [{formatted_kws}\n    ]{comma}')
print("}")