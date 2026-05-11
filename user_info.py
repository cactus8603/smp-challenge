import json
from collections import Counter
from pathlib import Path

# 改成你的 train category.json 路徑
cats = json.load(open("/local/smp/data/train/train_category.json"))
uid_counts = Counter(row["Uid"] for row in cats)

counts = sorted(uid_counts.values())
n = len(counts)

print(f"總 user 數     : {n:,}")
print(f"中位數         : {counts[n//2]}")
print(f"平均           : {sum(counts)/n:.1f}")
print(f"只有 1 張      : {sum(1 for c in counts if c == 1):,}  ({100*sum(1 for c in counts if c==1)/n:.1f}%)")
print(f"< 5 張         : {sum(1 for c in counts if c < 5):,}  ({100*sum(1 for c in counts if c<5)/n:.1f}%)")
print(f"5–30 張        : {sum(1 for c in counts if 5 <= c <= 30):,}")
print(f"31–100 張      : {sum(1 for c in counts if 31 <= c <= 100):,}")
print(f"> 100 張       : {sum(1 for c in counts if c > 100):,}")


# 總 user 數     : 38,312
# 中位數         : 1
# 平均           : 8.0
# 只有 1 張      : 19,446  (50.8%)
# < 5 張         : 29,901  (78.0%)
# 5–30 張        : 6,645
# 31–100 張      : 1,255
# > 100 張       : 511 
