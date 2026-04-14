#!/usr/bin/env python3
"""
check_crawl_output.py

檢查 run_smp_category_crawl.sh 的輸出是否完整。

檢查項目：
1. 每個 category 資料夾是否存在
2. 每個 category 的各檔案筆數
3. 各檔案的 Pid 是否一致（有沒有對不上的）
4. 圖片檔案有沒有下載
5. 目標筆數達成率
6. mature / label 有效率

用法：
    python3 check_crawl_output.py
    python3 check_crawl_output.py --base_dir /local/smp/extra_data
    python3 check_crawl_output.py --base_dir /local/smp/extra_data --total_items 480000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Set

# -------------------------------------------------------
# 對應 run_smp_category_crawl.sh 的設定
# -------------------------------------------------------
TOTAL_ITEMS_DEFAULT = 480000

CATEGORIES = [
    ("Travel&Active&Sports", "travel",      25180),
    ("Holiday&Celebrations", "holiday",     10790),
    ("Animal",               "animal",      10230),
    ("Entertainment",        "concert",      9950),
    ("Fashion",              "fashion",      9950),
    ("Whether&Season",       "winter",       8270),
    ("Social&People",        "people",       8000),
    ("Urban",                "city",         6650),
    ("Food",                 "food",         6550),
    ("Electronics",          "electronics",  3320),
    ("Family",               "family",       3000),
]

EXPECTED_FILES = [
    "extra_text.jsonl",
    "extra_temporalspatial_information.jsonl",
    "extra_user_data.jsonl",
    "extra_additional_information.jsonl",
    "extra_category.jsonl",
    "extra_pseudo_label.jsonl",
    "extra_img_filepath.txt",
    "crawl_manifest.jsonl",
]


# -------------------------------------------------------
# Helpers
# -------------------------------------------------------
def sanitize_name(name: str) -> str:
    return name.replace("&", "_").replace(" ", "_").replace("/", "_")


def scaled_target(base_target: int, total_items: int) -> int:
    return max(1, round(base_target * total_items / 100000.0))


def count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def count_txt_lines(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def load_pids_from_jsonl(path: Path, key: str = "Pid") -> Set[str]:
    pids = set()
    if not path.exists():
        return pids
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                pid = obj.get(key)
                if pid is not None:
                    pids.add(str(pid))
            except json.JSONDecodeError:
                pass
    return pids


def load_mature_stats(path: Path):
    """從 pseudo_label 檔案統計 is_mature 和有效 label 數量"""
    if not path.exists():
        return 0, 0, 0
    total = 0
    mature = 0
    has_label = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                total += 1
                if obj.get("is_mature"):
                    mature += 1
                if obj.get("PseudoLogViewScore") is not None:
                    has_label += 1
            except json.JSONDecodeError:
                pass
    return total, mature, has_label


def count_images(image_dir: Path) -> int:
    if not image_dir.exists():
        return 0
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
    return sum(
        1 for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def fmt(n: int) -> str:
    return f"{n:,}"


def pct(n: int, total: int) -> str:
    if total == 0:
        return "N/A"
    return f"{100 * n / total:.1f}%"


# -------------------------------------------------------
# Check one category
# -------------------------------------------------------
def check_category(
    cat_name: str,
    base_target: int,
    total_items: int,
    base_dir: Path,
) -> Dict:
    safe = sanitize_name(cat_name)
    cat_dir = base_dir / f"extra_data_{safe}"
    target = scaled_target(base_target, total_items)

    result = {
        "category": cat_name,
        "target": target,
        "dir_exists": cat_dir.exists(),
        "files": {},
        "pid_consistency": {},
        "pseudo_stats": {},
        "image_count": 0,
        "warnings": [],
        "ok": True,
    }

    if not cat_dir.exists():
        result["ok"] = False
        result["warnings"].append(f"Directory not found: {cat_dir}")
        return result

    # ---- file counts ----
    file_counts: Dict[str, int] = {}
    for fname in EXPECTED_FILES:
        fpath = cat_dir / fname
        if fname.endswith(".jsonl"):
            file_counts[fname] = count_jsonl_lines(fpath)
        elif fname.endswith(".txt"):
            file_counts[fname] = count_txt_lines(fpath)
        else:
            file_counts[fname] = 1 if fpath.exists() else 0

        if not fpath.exists():
            result["warnings"].append(f"Missing file: {fname}")
            result["ok"] = False

    result["files"] = file_counts

    # ---- pid consistency ----
    # 以 extra_text.jsonl 的 Pid 為基準
    ref_pids = load_pids_from_jsonl(cat_dir / "extra_text.jsonl", key="Pid")
    pid_checks = {}
    for fname in [
        "extra_temporalspatial_information.jsonl",
        "extra_additional_information.jsonl",
        "extra_category.jsonl",
        "extra_pseudo_label.jsonl",
    ]:
        fpath = cat_dir / fname
        other_pids = load_pids_from_jsonl(fpath, key="Pid")
        missing = ref_pids - other_pids
        extra   = other_pids - ref_pids
        pid_checks[fname] = {
            "count": len(other_pids),
            "missing_from_ref": len(missing),
            "extra_vs_ref": len(extra),
        }
        if missing or extra:
            result["warnings"].append(
                f"Pid mismatch in {fname}: missing={len(missing)}, extra={len(extra)}"
            )
            result["ok"] = False
    result["pid_consistency"] = pid_checks

    # ---- pseudo label stats ----
    total_p, mature_p, has_label_p = load_mature_stats(
        cat_dir / "extra_pseudo_label.jsonl"
    )
    result["pseudo_stats"] = {
        "total": total_p,
        "mature": mature_p,
        "has_label": has_label_p,
        "mature_rate": pct(mature_p, total_p),
        "label_rate": pct(has_label_p, total_p),
    }

    # ---- image count ----
    result["image_count"] = count_images(cat_dir / "images")

    # ---- target completion ----
    actual = len(ref_pids)
    result["actual_count"] = actual
    result["completion_rate"] = pct(actual, target)
    if actual < target * 0.9:
        result["warnings"].append(
            f"Under target: {fmt(actual)}/{fmt(target)} ({pct(actual, target)})"
        )

    return result


# -------------------------------------------------------
# Report
# -------------------------------------------------------
def print_report(results: list, total_items: int) -> None:
    print("=" * 80)
    print(f"{'CRAWL OUTPUT CHECK REPORT':^80}")
    print(f"{'Total target: ' + fmt(total_items):^80}")
    print("=" * 80)

    grand_actual = 0
    grand_target = 0
    grand_images = 0
    grand_label  = 0
    all_ok = True

    for r in results:
        status = "✅" if r["ok"] else "❌"
        target = r["target"]
        actual = r.get("actual_count", 0)
        grand_actual += actual
        grand_target += target
        grand_images += r.get("image_count", 0)
        ps = r.get("pseudo_stats", {})
        grand_label += ps.get("has_label", 0)
        if not r["ok"]:
            all_ok = False

        print(f"\n{status} {r['category']}")
        print(f"   Dir exists    : {r['dir_exists']}")
        print(f"   Target        : {fmt(target)}")
        print(f"   Actual (text) : {fmt(actual)}  ({r.get('completion_rate','N/A')})")
        print(f"   Images        : {fmt(r['image_count'])}")
        print(f"   Mature rate   : {ps.get('mature_rate','N/A')}")
        print(f"   Label rate    : {ps.get('label_rate','N/A')}")

        # file counts
        files = r.get("files", {})
        if files:
            print("   File counts   :")
            for fname, cnt in files.items():
                flag = "" if cnt == actual or fname in ("extra_user_data.jsonl",) else " ⚠"
                print(f"     {fname:<50} {fmt(cnt):>8}{flag}")

        # warnings
        if r["warnings"]:
            for w in r["warnings"]:
                print(f"   ⚠  {w}")

    print("\n" + "=" * 80)
    print(f"{'SUMMARY':^80}")
    print("=" * 80)
    overall_ok = "✅ All OK" if all_ok else "❌ Issues found"
    print(f"  Status          : {overall_ok}")
    print(f"  Grand total     : {fmt(grand_actual)} / {fmt(grand_target)} ({pct(grand_actual, grand_target)})")
    print(f"  Images on disk  : {fmt(grand_images)}")
    print(f"  Valid labels    : {fmt(grand_label)} ({pct(grand_label, grand_actual)})")
    print("=" * 80)


# -------------------------------------------------------
# Main
# -------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check crawl output completeness.")
    parser.add_argument(
        "--base_dir", type=str, default="/local/smp/extra_data",
        help="Base output directory (same as BASE_OUTPUT_DIR in the shell script)",
    )
    parser.add_argument(
        "--total_items", type=int, default=TOTAL_ITEMS_DEFAULT,
        help=f"Total target items (default: {TOTAL_ITEMS_DEFAULT})",
    )
    parser.add_argument(
        "--skip_images", action="store_true",
        help="Skip counting image files (faster)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir)

    if not base_dir.exists():
        print(f"❌ Base directory not found: {base_dir}")
        return

    print(f"Checking: {base_dir}")
    print(f"Total target: {fmt(args.total_items)}\n")

    results = []
    for cat_name, query, base_target in CATEGORIES:
        print(f"  Checking {cat_name}...", end="\r")
        r = check_category(cat_name, base_target, args.total_items, base_dir)
        if args.skip_images:
            r["image_count"] = -1
        results.append(r)

    print(" " * 60, end="\r")  # clear progress line
    print_report(results, args.total_items)


if __name__ == "__main__":
    main()
