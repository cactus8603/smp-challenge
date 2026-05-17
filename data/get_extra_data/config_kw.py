from typing import Dict, List

# SMP category keyword mapping（根據 SMP category 分布設計）
_SMP_CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "Family": [
        "babies", "brother", "child", "cuddle", "cuddly", "family", "familylife", "familytime", "father", "happyfamily",
        "home", "infant", "kids", "lovefamily", "lovely", "mother", "parenting", "parents", "relatives", "siblings",
        "sister", "small", "tiny", "toddler", "together"
    ],
    "Animal": [
        "amphibian", "animal", "animal-behavior", "animal-closeup", "animal-portrait", "aquaria", "bird", "birdwatching", "bug", "bugs",
        "butterfly", "cat", "cats", "cute-animals", "dog", "doglover", "dogs", "dogsitting", "domestic", "endangered-species",
        "exotic-animals", "farm", "fish", "fishtank", "forest-animals", "horse", "hound", "insect", "insects", "kitten",
        "lovecats", "lovenature", "macrophotography", "mammal", "mane", "marine-life", "nature-photography", "pet", "pet-photography", "ponies",
        "pup", "puppy", "reptile", "safari", "underwater", "wild-animals", "wildlife", "zoo"
    ],
    "Fashion": [
        "accessory", "apparel", "black", "blonde", "bracelet", "branco", "casual", "clothes", "denim", "fashion",
        "fashionblogger", "fashionista", "formal", "gem", "gorgeous", "hairfashion", "hairstyle", "highheels", "jewelry", "longhair",
        "lookbook", "makeup", "model", "ootd", "outfit", "powder", "runway", "shoe", "streetstyle", "style",
        "stylish", "tall", "tats", "tatts", "trend", "tshirt", "unhas", "vintage", "wardrobe", "whatiwore"
    ],
    "Travel&Active&Sports": [
        "abroad", "active", "adventure", "airport", "athlete", "backpacking", "baltimore", "baseball", "basketball", "bat",
        "bronco", "budget-travel", "car", "coach", "competition", "court", "crossfit", "cruise", "cycling", "dedication",
        "denverbroncos", "destination", "driver", "exercise", "explore", "fight", "fitness", "flight", "football", "getaway",
        "gym", "health", "highway", "holiday", "hotel", "icerink", "indoor-sports", "island", "itinerary", "jogging",
        "journey", "luxury-travel", "marathon", "match", "mitt", "outdoor-sports", "passport", "practice", "quarterback", "resort",
        "road-trip", "running", "sanfrancsico", "shredded", "sightseeing", "soccer", "solo-travel", "sport", "sports", "squat",
        "stadium", "swimming", "team", "tennis", "tourism", "tourist", "tournament", "trailrunner", "train-journey", "training",
        "travel", "travel-blog", "travel-diary", "travel-life", "travel-photography", "trip", "vacation", "wanderlust", "weightlifting", "workout",
        "world-tour", "yoga"
    ],
    "Social&People": [
        "bestfriend", "boyfriend", "couple", "dating", "early", "friends", "friendship", "fun", "girlfriend", "girls",
        "goingout", "good,morning", "group", "hangout", "kisses", "knockout", "lady", "laugh", "lifestyle", "lightsout",
        "love", "morning", "nightout", "partytime", "people", "portrait", "ready", "relationship", "selfie", "smile",
        "snooze", "social", "sunrise", "togetherness", "vibes", "wakeup", "weekend"
    ],
    "Food": [
        "amazing", "bakery", "bar", "bbq", "beer", "beers", "breakfast", "brunch", "cafe", "caffeine",
        "coffee", "cooking", "cuisine", "delicious", "delish", "dessert", "desserts", "dinner", "drinks", "eat",
        "fast-food", "fine-dining", "food", "food-blog", "food-photography", "foodie", "foodpics", "gastronomy", "general", "grill",
        "home-cooking", "homemade", "hungry", "liquor", "lunch", "meal", "pub", "recipe", "restaurant", "seafood",
        "snack", "street-food", "takeout", "teacup", "thirst", "thirsty", "vegan", "vegetarian", "wine", "yummy"
    ],
    "Electronics": [
        "android", "camera", "charger", "computers", "desktop", "device", "device-tech", "digital", "earphones", "electronic",
        "electronics", "gadget", "gadgets", "hack", "headphones", "iphone", "keyboard", "laptops", "mobile", "monitor",
        "mouse", "pc", "phone", "samsunggalaxy", "screen", "smartphone", "smartwatch", "tablet", "tech", "technology",
        "usb", "wearable"
    ],
    "Urban": [
        "abstract", "alley", "archidaily", "architecture", "architecturelovers", "arts", "artwork", "bridge", "building", "bus",
        "city", "city-lights", "cityscape", "composition", "construction", "downtown", "graffiti", "infrastructure", "intersection", "landmark",
        "lines", "metropolis", "minimal", "modern", "neon", "night", "pasteup", "pattern", "pavement", "public-transport",
        "skyline", "skyscraper", "stencil", "street", "street-life", "streetart", "streetarteverywhere", "subway", "traffic", "urban",
        "urban-exploration", "urban-life", "urbanart", "wall"
    ],
    "Entertainment": [
        "actor", "album", "author", "beat", "book", "broadcast", "cinema", "concert", "dj", "entertainment",
        "films", "flicks", "gamer", "gaming", "hiphop", "kindle", "listentothis", "media", "movie", "moviestar",
        "music", "netflix", "nook", "online", "pages", "playlist", "plot", "pop", "repeat", "rock",
        "series", "showtime", "song", "stories", "streaming", "tv", "video&games", "videogame", "videos", "words"
    ],
    "Whether&Season": [
        "autumn", "beautifulday", "blue-sky", "bright", "chilly", "clearskies", "cloud", "cold", "color", "downpour",
        "drizzle", "fall", "fog", "frost", "holidayseason", "horizon", "hot", "leaves", "mist", "overcast",
        "rain", "rainbow", "rainyday", "season", "skyporn", "snow", "snowing", "spring", "springtime", "storm",
        "summer", "summertime", "sunny", "sunshine", "thunderstorm", "trees", "umbrella", "weather", "windy", "winter"
    ],
    "Holiday&Celebrations": [
        "anniversary", "birthdayparty", "bride", "carving", "celebrate", "celebration", "ceremony", "cheers", "christmas", "christmasdecor",
        "congratulations", "event", "festival", "festive", "fireworks", "gathering", "gift", "gifts", "halloweenparty", "hauntedhouse",
        "holiday", "jolly", "merrychristmas", "new", "newyear", "newyearseve", "parties", "party", "partying", "pumpkins",
        "scary", "specialday", "thanksgiving", "unforgettable", "wedding", "weddinggown", "xmas"
    ]
}

_EXCLUDE_TAGS = {
    # Camera brands / hardware
    "canon", "nikon", "sony", "fujifilm", "olympus", "panasonic", "iphone",
    # Photography meta
    "explore", "interestingness", "fstop", "iso", "shutter", "lens", "camera",
    # Platforms / apps
    "flickr", "flickriosapp", "instagram", "app", "square", "vsco", "snapseed",
    "photography", "photo", "image", "pic", "pics", "photographer",
    # Flickr auto-generated quality-assessment system tags
    "blurjudgementappropriate", "exposurejudgementappropriate", "facefocusdetectionappropriate",
    # Wallpaper / format tags (describe file use, not content)
    "wallpapers", "backgrounds", "animated",
    # Demographic-only tags (describe a person, not the photo's content)
    "male", "adult", "man", "person",
}

# Photos tagged with ANY of these are AI-generated and should be skipped entirely.
_AI_GENERATED_TAGS = {
    "ai", "aigenerated",
    "aiinspirations", "aiphotography", "ainature",
    "aieurope", "aieuropeanmelancholy", "aiarthistory",
    "aiportraits", "aimelancholiclandscapes",
    "chatgptimagecreator",
    "midjourney", "stablediffusion",
    "dalle", "dalle2", "xai",
}