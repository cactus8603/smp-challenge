# # import json, math, numpy as np

# # # 官方 label 分布
# # labels = [float(l.strip()) for l in open("/local/smp/data/train/train_label.txt") if l.strip()]
# # print(f"官方 label: min={min(labels):.2f} max={max(labels):.2f} mean={np.mean(labels):.2f}")

# # # pseudo label 分布（取一個 category 的樣本）
# # rows = [json.loads(l) for l in open("/local/smp/extra_data/extra_data_Travel_Active_Sports/extra_pseudo_label.jsonl") if l.strip()]
# # pseudo = [r["PseudoLogViewScore"] for r in rows if r.get("PseudoLogViewScore")]
# # print(f"Pseudo label: min={min(pseudo):.2f} max={max(pseudo):.2f} mean={np.mean(pseudo):.2f}")

# import json
# from pathlib import Path

# base = Path("/local/smp/extra_data/extra_data_Travel_Active_Sports")  # 改成你的路徑

# total = 0
# kept = 0

# for jsonl in base.rglob("extra_pseudo_label.jsonl"):
#     for line in jsonl.open():
#         line = line.strip()
#         if not line:
#             continue
#         try:
#             r = json.loads(line)
#             score = r.get("PseudoLogViewScore")
#             if score is not None:
#                 total += 1
#                 if float(score) >= 1.0:
#                     kept += 1
#         except Exception:
#             pass

# print(f"總筆數  : {total:,}")
# print(f">= 1.0  : {kept:,}  ({100*kept/total:.1f}%)")
# print(f"丟掉    : {total-kept:,}  ({100*(total-kept)/total:.1f}%)")

import json
from pathlib import Path

path = Path("/local/smp/data/train/train_temporalspatial_information.json")
data = json.load(open(path))

total = len(data)
has_geo = sum(
    1 for r in data
    if r.get("Latitude") not in (None, "", "0.0", "0", 0)
    and r.get("Longitude") not in (None, "", "0.0", "0", 0)
)
print(f"總筆數    : {total:,}")
print(f"有 geo    : {has_geo:,}  ({100*has_geo/total:.1f}%)")
print(f"unique lat/lon 組合數量:")

coords = set()
for r in data:
    lat = r.get("Latitude")
    lon = r.get("Longitude")
    if lat not in (None, "", "0.0", "0", 0) and lon not in (None, "", "0.0", "0", 0):
        # round to 2 decimal places to group nearby coords
        coords.add((round(float(lat), 2), round(float(lon), 2)))
print(f"  精確到小數點2位: {len(coords):,} 組")