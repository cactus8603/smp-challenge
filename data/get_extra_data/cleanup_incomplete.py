#!/usr/bin/env python3
"""
刪除不完整的爬蟲資料：
  完整 = image 存在 + 所有 photo-level jsonl 都有此 Pid + user_data 有此 Uid
不完整的 entry 會從所有 jsonl 中移除，對應 image 也會刪除。
"""
import json
import os
import shutil
from pathlib import Path

BASE = Path("/ssd1/lchiayu/smp-challenge/data/get_extra_data/output")

CATEGORIES = [
    "extra_data_Animal",
    "extra_data_Electronics",
    "extra_data_Entertainment",
    "extra_data_Family",
    "extra_data_Fashion",
    "extra_data_Food",
    "extra_data_Holiday_Celebrations",
    "extra_data_Social_People",
    "extra_data_Travel_Active_Sports",
    "extra_data_Urban",
    "extra_data_Whether_Season",
]

# Photo-level jsonls (每筆都有 Pid)
PHOTO_JSONLS = [
    "extra_text.jsonl",
    "extra_category.jsonl",
    "extra_additional_information.jsonl",
    "extra_temporalspatial_information.jsonl",
    "extra_pseudo_label.jsonl",
]


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def process_category(cat_dir: Path) -> dict:
    print(f"\n{'='*60}")
    print(f"Category: {cat_dir.name}")

    manifest_path = cat_dir / "crawl_manifest.jsonl"
    if not manifest_path.exists():
        print("  [SKIP] crawl_manifest.jsonl not found")
        return {}

    manifest = read_jsonl(manifest_path)
    print(f"  Manifest entries: {len(manifest)}")

    # Step 1: manifest 裡 Pid → relative_image_path + Uid
    pid_to_info = {}  # Pid → {relative_image_path, Uid}
    for row in manifest:
        pid = str(row.get("Pid", ""))
        if pid:
            pid_to_info[pid] = {
                "rel_path": row.get("relative_image_path", ""),
                "uid": str(row.get("Uid", "")),
            }

    # Step 2: 哪些 Pid 圖片真的存在
    img_dir = cat_dir / "images"
    pids_with_image = set()
    for pid, info in pid_to_info.items():
        img_path = img_dir / info["rel_path"]
        if img_path.exists() and img_path.stat().st_size > 0:
            pids_with_image.add(pid)
    print(f"  Pids with image on disk: {len(pids_with_image)}")

    # Step 3: 各 photo-level jsonl 的 Pid 集合
    jsonl_pid_sets = {}
    for fname in PHOTO_JSONLS:
        fpath = cat_dir / fname
        if not fpath.exists():
            print(f"  [WARN] {fname} not found — skipping this check")
            continue
        rows = read_jsonl(fpath)
        pids = {str(r["Pid"]) for r in rows if "Pid" in r}
        jsonl_pid_sets[fname] = pids

    # Step 4: 完整 Pid = image 存在 + 出現在所有 photo jsonl
    complete_pids = set(pids_with_image)
    for fname, pids in jsonl_pid_sets.items():
        before = len(complete_pids)
        complete_pids &= pids
        after = len(complete_pids)
        if before != after:
            print(f"  After intersect {fname}: {after} (removed {before-after})")
    print(f"  Complete Pids: {len(complete_pids)}")

    # Step 5: 對應的 Uid 集合
    complete_uids = {pid_to_info[pid]["uid"] for pid in complete_pids if pid in pid_to_info}

    # ── 統計 ──────────────────────────────────────────────
    n_total = len(pid_to_info)
    n_complete = len(complete_pids)
    n_removed = n_total - n_complete
    print(f"  Will KEEP {n_complete} / {n_total} entries (remove {n_removed})")

    if n_removed == 0:
        print("  Nothing to clean.")
        return {"category": cat_dir.name, "total": n_total, "kept": n_complete, "removed": 0}

    # ── 改寫 crawl_manifest.jsonl ─────────────────────────
    new_manifest = [r for r in manifest if str(r.get("Pid", "")) in complete_pids]
    write_jsonl(manifest_path, new_manifest)
    print(f"  Rewrote {manifest_path.name}: {len(new_manifest)} entries")

    # ── 改寫各 photo-level jsonl ──────────────────────────
    for fname in PHOTO_JSONLS:
        fpath = cat_dir / fname
        if not fpath.exists():
            continue
        rows = read_jsonl(fpath)
        new_rows = [r for r in rows if str(r.get("Pid", "")) in complete_pids]
        write_jsonl(fpath, new_rows)
        print(f"  Rewrote {fname}: {len(new_rows)} entries (was {len(rows)})")

    # ── 改寫 extra_user_data.jsonl（按 Uid）─────────────
    user_path = cat_dir / "extra_user_data.jsonl"
    if user_path.exists():
        user_rows = read_jsonl(user_path)
        new_user = [r for r in user_rows if str(r.get("Uid", "")) in complete_uids]
        write_jsonl(user_path, new_user)
        print(f"  Rewrote extra_user_data.jsonl: {len(new_user)} entries (was {len(user_rows)})")

    # ── 改寫 extra_img_filepath.txt ───────────────────────
    filepath_txt = cat_dir / "extra_img_filepath.txt"
    if filepath_txt.exists():
        lines = filepath_txt.read_text().splitlines()
        # 從 relative_image_path 取出 Pid（最後一段去掉 .jpg）
        complete_rel_paths = {pid_to_info[pid]["rel_path"] for pid in complete_pids}
        new_lines = [l for l in lines if l.strip() in complete_rel_paths]
        filepath_txt.write_text("\n".join(new_lines) + ("\n" if new_lines else ""))
        print(f"  Rewrote extra_img_filepath.txt: {len(new_lines)} lines (was {len(lines)})")

    # ── 刪除孤立圖片 ──────────────────────────────────────
    complete_rel_paths = {pid_to_info[pid]["rel_path"] for pid in complete_pids}
    n_deleted_imgs = 0
    for img_path in img_dir.rglob("*.jpg"):
        rel = str(img_path.relative_to(img_dir))
        if rel not in complete_rel_paths:
            img_path.unlink()
            n_deleted_imgs += 1
    # 刪空目錄
    for d in sorted(img_dir.rglob("*"), reverse=True):
        if d.is_dir():
            try:
                d.rmdir()
            except OSError:
                pass
    print(f"  Deleted {n_deleted_imgs} orphan image files")

    return {
        "category": cat_dir.name,
        "total": n_total,
        "kept": n_complete,
        "removed": n_removed,
        "deleted_images": n_deleted_imgs,
    }


def main():
    results = []
    for cat in CATEGORIES:
        cat_dir = BASE / cat
        if not cat_dir.exists():
            print(f"[SKIP] {cat} — directory not found")
            continue
        r = process_category(cat_dir)
        if r:
            results.append(r)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    total_kept = total_removed = total_del_img = 0
    for r in results:
        total_kept += r.get("kept", 0)
        total_removed += r.get("removed", 0)
        total_del_img += r.get("deleted_images", 0)
        print(f"  {r['category']:40s}  kept={r.get('kept',0):6d}  removed={r.get('removed',0):6d}  del_imgs={r.get('deleted_images',0):6d}")
    print(f"\n  TOTAL  kept={total_kept}  removed={total_removed}  del_imgs={total_del_img}")


if __name__ == "__main__":
    main()
