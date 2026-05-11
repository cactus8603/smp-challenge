#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
crawl_status.py

一次看清楚爬蟲目前的狀態：
  - 每個 category 爬了幾筆 vs 目標（進度條）
  - 所有應有的檔案是否存在
  - Pid 在各檔案間是否一致
  - 內容品質：欄位覆蓋率、label 覆蓋率、geo 覆蓋率
  - User 數量

用法：
  python3 crawl_status.py
  python3 crawl_status.py --base_dir /local/smp/extra_data
  python3 crawl_status.py --base_dir /local/smp/extra_data_v2 --total_items 480000
  python3 crawl_status.py --base_dir /local/smp/extra_data_v2 --cat Travel
  python3 crawl_status.py --base_dir /local/smp/extra_data_v2 --skip_images --sample 200
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# ──────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────
TOTAL_ITEMS_DEFAULT = 480_000

CATEGORIES = [
    ("Travel&Active&Sports", 25180),
    ("Holiday&Celebrations", 10790),
    ("Animal",               10230),
    ("Entertainment",         9950),
    ("Fashion",               9950),
    ("Whether&Season",        8270),
    ("Social&People",         8000),
    ("Urban",                 6650),
    ("Food",                  6550),
    ("Electronics",           3320),
    ("Family",                3000),
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

PID_SYNC_FILES = [
    "extra_temporalspatial_information.jsonl",
    "extra_additional_information.jsonl",
    "extra_category.jsonl",
    "extra_pseudo_label.jsonl",
]

VALID_CATEGORIES = {c for c, _ in CATEGORIES}

# ──────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────
def sanitize(name: str) -> str:
    return name.replace("&", "_").replace(" ", "_").replace("/", "_")


def scaled_target(base: int, total: int) -> int:
    return max(1, round(base * total / 100_000.0))


def fmt(n: int) -> str:
    return f"{n:,}"


def pct(n: int, total: int) -> str:
    if total == 0:
        return "N/A"
    return f"{100 * n / total:.1f}%"


def progress_bar(current: int, target: int, width: int = 24) -> str:
    if target == 0:
        return "[" + "?" * width + "]"
    ratio = min(current / target, 1.0)
    filled = int(ratio * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {pct(current, target)}"


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    c = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                c += 1
    return c


def load_pids(path: Path) -> Set[str]:
    pids: Set[str] = set()
    if not path.exists():
        return pids
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                pid = obj.get("Pid")
                if pid is not None:
                    pids.add(str(pid))
            except Exception:
                pass
    return pids


def sample_jsonl(path: Path, n: int) -> List[Dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if len(rows) >= n:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def count_images(d: Path) -> int:
    if not d.exists():
        return 0
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return sum(1 for p in d.rglob("*") if p.is_file() and p.suffix.lower() in exts)


# ──────────────────────────────────────────────────────
# Per-category checks
# ──────────────────────────────────────────────────────
def check_files(cat_dir: Path) -> Tuple[Dict[str, int], List[str]]:
    """Returns file line counts and list of missing files."""
    counts = {}
    missing = []
    for fname in EXPECTED_FILES:
        p = cat_dir / fname
        if not p.exists():
            missing.append(fname)
            counts[fname] = 0
        elif fname.endswith(".txt"):
            counts[fname] = count_lines(p)
        else:
            counts[fname] = count_lines(p)
    return counts, missing


def check_pid_consistency(cat_dir: Path, ref_pids: Set[str]) -> List[str]:
    """Returns list of warning strings for Pid mismatches."""
    warnings = []
    for fname in PID_SYNC_FILES:
        other = load_pids(cat_dir / fname)
        missing = ref_pids - other
        extra   = other - ref_pids
        if missing:
            warnings.append(f"{fname}: {len(missing)} Pid missing")
        if extra:
            warnings.append(f"{fname}: {len(extra)} Pid extra")
    return warnings


def check_content(cat_dir: Path, sample: int) -> Dict:
    """Sample rows and compute coverage stats."""
    stats = {}

    # text
    rows = sample_jsonl(cat_dir / "extra_text.jsonl", sample)
    n = len(rows)
    if n > 0:
        stats["title_rate"]   = pct(sum(1 for r in rows if r.get("Title")), n)
        stats["tag_rate"]     = pct(sum(1 for r in rows if r.get("Alltags")), n)
        stats["desc_rate"]    = pct(sum(1 for r in rows if r.get("description")), n)
        tag_counts = [r.get("tag_count", 0) or 0 for r in rows]
        stats["avg_tags"]     = f"{sum(tag_counts)/n:.1f}"

    # temporal
    rows = sample_jsonl(cat_dir / "extra_temporalspatial_information.jsonl", sample)
    n = len(rows)
    if n > 0:
        has_geo = sum(
            1 for r in rows
            if r.get("Latitude") not in (None, "None", "0.0", "0")
            and r.get("Longitude") not in (None, "None", "0.0", "0")
        )
        stats["geo_rate"]  = pct(has_geo, n)
        stats["taken_rate"] = pct(sum(1 for r in rows if r.get("date_taken")), n)

    # pseudo label
    rows = sample_jsonl(cat_dir / "extra_pseudo_label.jsonl", sample)
    n = len(rows)
    if n > 0:
        stats["label_rate"]    = pct(sum(1 for r in rows if r.get("PseudoLogViewScore") is not None), n)
        stats["views_rate"]    = pct(sum(1 for r in rows if r.get("raw_views") is not None), n)
        stats["comments_rate"] = pct(sum(1 for r in rows if r.get("raw_comments") is not None), n)
        stats["mature_rate"]   = pct(sum(1 for r in rows if r.get("is_mature")), n)

    # category
    rows = sample_jsonl(cat_dir / "extra_category.jsonl", sample)
    n = len(rows)
    if n > 0:
        stats["invalid_cats"] = sum(
            1 for r in rows
            if r.get("Category") and r["Category"] not in VALID_CATEGORIES
        )
        stats["sub_rate"] = pct(sum(1 for r in rows if r.get("Subcategory")), n)

    # user
    rows = sample_jsonl(cat_dir / "extra_user_data.jsonl", sample)
    n = len(rows)
    if n > 0:
        stats["user_sample"]       = n
        stats["follower_rate"]     = pct(sum(1 for r in rows if r.get("follower_count") is not None), n)
        stats["total_views_rate"]  = pct(sum(1 for r in rows if r.get("total_views") is not None), n)

    return stats


def check_category(
    cat_name: str,
    base_target: int,
    total_items: int,
    base_dir: Path,
    sample: int,
    skip_images: bool,
) -> Dict:
    safe    = sanitize(cat_name)
    cat_dir = base_dir / f"extra_data_{safe}"
    target  = scaled_target(base_target, total_items)

    result = {
        "name":    cat_name,
        "target":  target,
        "dir":     cat_dir,
        "exists":  cat_dir.exists(),
        "errors":  [],
        "warnings": [],
    }

    if not cat_dir.exists():
        result["actual"] = 0
        result["errors"].append("Directory not found")
        return result

    # ── file existence + counts ──
    file_counts, missing = check_files(cat_dir)
    result["file_counts"] = file_counts
    for f in missing:
        result["errors"].append(f"Missing: {f}")

    # ── reference count from extra_text.jsonl ──
    actual = file_counts.get("extra_text.jsonl", 0)
    result["actual"] = actual

    # ── user count ──
    result["user_count"] = count_lines(cat_dir / "extra_user_data.jsonl")

    # ── Pid consistency (full scan) ──
    ref_pids = load_pids(cat_dir / "extra_text.jsonl")
    pid_warnings = check_pid_consistency(cat_dir, ref_pids)
    result["warnings"].extend(pid_warnings)

    # ── content quality (sampled) ──
    result["content"] = check_content(cat_dir, sample)

    # ── image count ──
    result["image_count"] = -1 if skip_images else count_images(cat_dir / "images")

    # ── completion warning ──
    if actual < target * 0.9:
        result["warnings"].append(
            f"Under target: {fmt(actual)}/{fmt(target)} ({pct(actual, target)})"
        )

    return result


# ──────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────
def print_report(results: List[Dict], total_items: int, skip_images: bool) -> int:
    W = 80
    print("=" * W)
    print(f"{'CRAWL STATUS REPORT':^{W}}")
    print(f"{'Total target: ' + fmt(total_items):^{W}}")
    print("=" * W)

    grand_actual = grand_target = grand_users = grand_images = 0
    total_errors = total_warnings = 0

    for r in results:
        actual  = r.get("actual", 0)
        target  = r["target"]
        exists  = r["exists"]
        errors  = r["errors"]
        warnings = r["warnings"]
        content = r.get("content", {})
        users   = r.get("user_count", 0)
        images  = r.get("image_count", -1)

        grand_actual  += actual
        grand_target  += target
        grand_users   += users
        if images >= 0:
            grand_images += images
        total_errors   += len(errors)
        total_warnings += len(warnings)

        # ── status icon ──
        if not exists or errors:
            icon = "❌"
        elif warnings:
            icon = "⚠️ "
        elif actual >= target * 0.9:
            icon = "✅"
        else:
            icon = "🔄"

        bar = progress_bar(actual, target)
        print(f"\n{icon} {r['name']}")
        print(f"   Progress  {bar}  {fmt(actual)}/{fmt(target)}")
        print(f"   Users     {fmt(users)}"
              + (f"   Images {fmt(images)}" if images >= 0 else ""))

        # content stats
        c = content
        if c:
            print(f"   Text      title={c.get('title_rate','?')}  "
                  f"tags={c.get('tag_rate','?')}  "
                  f"avg_tags={c.get('avg_tags','?')}  "
                  f"desc={c.get('desc_rate','?')}")
            if "geo_rate" in c:
                print(f"   Temporal  geo={c.get('geo_rate','?')}  "
                      f"taken={c.get('taken_rate','?')}")
            if "label_rate" in c:
                print(f"   Label     coverage={c.get('label_rate','?')}  "
                      f"views={c.get('views_rate','?')}  "
                      f"comments={c.get('comments_rate','?')}  "
                      f"mature={c.get('mature_rate','?')}")
            if "sub_rate" in c:
                inv = c.get("invalid_cats", 0)
                inv_str = f"  ⚠ {inv} invalid cat" if inv else ""
                print(f"   Category  subcategory={c.get('sub_rate','?')}{inv_str}")
            if "follower_rate" in c:
                print(f"   User      followers={c.get('follower_rate','?')}  "
                      f"total_views={c.get('total_views_rate','?')}")

        # file counts (compact)
        fc = r.get("file_counts", {})
        if fc:
            counts_str = "  ".join(
                f"{k.replace('extra_','').replace('.jsonl','').replace('.txt','')}={fmt(v)}"
                for k, v in fc.items()
                if v != actual and k not in ("extra_user_data.jsonl", "crawl_manifest.jsonl")
            )
            if counts_str:
                print(f"   ⚠ Count mismatch: {counts_str}")

        for e in errors:
            print(f"   ❌ {e}")
        for w in warnings:
            print(f"   ⚠  {w}")

    # ── summary ──
    print(f"\n{'=' * W}")
    print(f"{'SUMMARY':^{W}}")
    print("=" * W)
    overall = "✅ All OK" if total_errors == 0 and total_warnings == 0 else \
              f"{'❌' if total_errors else '⚠️ '} {total_errors} errors, {total_warnings} warnings"
    print(f"  Status        : {overall}")
    print(f"  Progress      : {progress_bar(grand_actual, grand_target, 30)}  "
          f"{fmt(grand_actual)} / {fmt(grand_target)}")
    print(f"  Total users   : {fmt(grand_users)}")
    if not skip_images:
        print(f"  Total images  : {fmt(grand_images)}")
    print("=" * W)

    return total_errors


# ──────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check crawl status and data quality.")
    p.add_argument("--base_dir",    default="/local/smp/extra_data")
    p.add_argument("--total_items", type=int, default=TOTAL_ITEMS_DEFAULT)
    p.add_argument("--cat",         default=None,
                   help="Only check this category (partial match OK, e.g. 'Travel')")
    p.add_argument("--sample",      type=int, default=500,
                   help="Rows to sample per file for content checks (default: 500)")
    p.add_argument("--skip_images", action="store_true",
                   help="Skip counting image files (faster)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir)

    if not base_dir.exists():
        print(f"❌ Base directory not found: {base_dir}")
        sys.exit(1)

    cats = CATEGORIES
    if args.cat:
        cats = [(c, b) for c, b in CATEGORIES if args.cat.lower() in c.lower()]
        if not cats:
            print(f"❌ No category matching '{args.cat}'")
            sys.exit(1)

    print(f"Base dir : {base_dir}")
    print(f"Target   : {fmt(args.total_items)}")
    print(f"Sample   : {args.sample} rows/file\n")

    results = []
    for cat_name, base_target in cats:
        print(f"  Checking {cat_name}...", end="\r", flush=True)
        r = check_category(
            cat_name, base_target, args.total_items,
            base_dir, args.sample, args.skip_images,
        )
        results.append(r)

    print(" " * 60, end="\r")
    n_errors = print_report(results, args.total_items, args.skip_images)
    sys.exit(0 if n_errors == 0 else 1)


if __name__ == "__main__":
    main()