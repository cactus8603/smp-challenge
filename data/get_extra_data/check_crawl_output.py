#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_crawl_output.py

Check completeness and consistency of crawl output produced by
run_smp_category_crawl.sh / crawl_flickr_to_smp.py.

Checks:
  1. Each category directory exists
  2. File line counts for all expected files
  3. Pid consistency across files (vs extra_text.jsonl as reference)
  4. Image file count (optional, can be slow)
  5. Target completion rate
  6. Pseudo-label stats (mature rate, label coverage)

Usage:
    python3 check_crawl_output.py
    python3 check_crawl_output.py --base_dir /local/smp/extra_data --total_items 480000
    python3 check_crawl_output.py --skip_images   # faster, skip image counting
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Set

# ──────────────────────────────────────────────────────
# Defaults — mirror run_smp_category_crawl.sh
# ──────────────────────────────────────────────────────
TOTAL_ITEMS_DEFAULT = 480_000

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

# These files are expected to have the same Pid count as extra_text.jsonl
PID_CHECK_FILES = [
    "extra_temporalspatial_information.jsonl",
    "extra_additional_information.jsonl",
    "extra_category.jsonl",
    "extra_pseudo_label.jsonl",
]


# ──────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────
def sanitize_name(name: str) -> str:
    return name.replace("&", "_").replace(" ", "_").replace("/", "_")


def scaled_target(base: int, total: int) -> int:
    return max(1, round(base * total / 100_000.0))


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    c = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                c += 1
    return c


def load_pids(path: Path, key: str = "Pid") -> Set[str]:
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
                pid = obj.get(key)
                if pid is not None:
                    pids.add(str(pid))
            except json.JSONDecodeError:
                pass
    return pids


def pseudo_stats(path: Path) -> tuple[int, int, int]:
    """Returns (total, mature_count, has_label_count)."""
    if not path.exists():
        return 0, 0, 0
    total = mature = has_label = 0
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
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return sum(1 for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def fmt(n: int) -> str:
    return f"{n:,}"


def pct(n: int, total: int) -> str:
    if total == 0:
        return "N/A"
    return f"{100 * n / total:.1f}%"


# ──────────────────────────────────────────────────────
# Per-category check
# ──────────────────────────────────────────────────────
def check_category(cat_name: str, base_target: int, total_items: int,
                   base_dir: Path, skip_images: bool) -> Dict:
    safe    = sanitize_name(cat_name)
    cat_dir = base_dir / f"extra_data_{safe}"
    target  = scaled_target(base_target, total_items)

    result = {
        "category": cat_name, "target": target,
        "dir_exists": cat_dir.exists(),
        "files": {}, "pid_consistency": {},
        "pseudo_stats": {}, "image_count": 0,
        "warnings": [], "ok": True,
    }

    if not cat_dir.exists():
        result["ok"] = False
        result["warnings"].append(f"Directory not found: {cat_dir}")
        return result

    # ── file counts ──
    file_counts: Dict[str, int] = {}
    for fname in EXPECTED_FILES:
        fpath = cat_dir / fname
        file_counts[fname] = count_lines(fpath)
        if not fpath.exists():
            result["warnings"].append(f"Missing file: {fname}")
            result["ok"] = False
    result["files"] = file_counts

    # ── Pid consistency ──
    ref_pids = load_pids(cat_dir / "extra_text.jsonl")
    pid_checks: Dict[str, Dict] = {}
    for fname in PID_CHECK_FILES:
        other_pids = load_pids(cat_dir / fname)
        missing = ref_pids - other_pids
        extra   = other_pids - ref_pids
        pid_checks[fname] = {
            "count":            len(other_pids),
            "missing_from_ref": len(missing),
            "extra_vs_ref":     len(extra),
        }
        if missing or extra:
            result["warnings"].append(
                f"Pid mismatch {fname}: missing={len(missing)}, extra={len(extra)}"
            )
            result["ok"] = False
    result["pid_consistency"] = pid_checks

    # ── pseudo label stats ──
    total_p, mature_p, label_p = pseudo_stats(cat_dir / "extra_pseudo_label.jsonl")
    result["pseudo_stats"] = {
        "total":       total_p,
        "mature":      mature_p,
        "has_label":   label_p,
        "mature_rate": pct(mature_p, total_p),
        "label_rate":  pct(label_p, total_p),
    }

    # ── image count ──
    result["image_count"] = -1 if skip_images else count_images(cat_dir / "images")

    # ── completion rate ──
    actual = len(ref_pids)
    result["actual_count"]    = actual
    result["completion_rate"] = pct(actual, target)
    if actual < target * 0.9:
        result["warnings"].append(
            f"Under target: {fmt(actual)}/{fmt(target)} ({pct(actual, target)})"
        )
        result["ok"] = False

    return result


# ──────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────
def print_report(results: list, total_items: int) -> None:
    W = 80
    print("=" * W)
    print(f"{'CRAWL OUTPUT CHECK REPORT':^{W}}")
    print(f"{'Total target: ' + fmt(total_items):^{W}}")
    print("=" * W)

    grand_actual = grand_target = grand_images = grand_label = 0
    all_ok = True

    for r in results:
        ok     = r["ok"]
        status = "✅" if ok else "❌"
        actual = r.get("actual_count", 0)
        target = r["target"]
        ps     = r.get("pseudo_stats", {})

        grand_actual += actual
        grand_target += target
        grand_label  += ps.get("has_label", 0)
        if r["image_count"] >= 0:
            grand_images += r["image_count"]
        if not ok:
            all_ok = False

        print(f"\n{status} {r['category']}")
        print(f"   Dir exists    : {r['dir_exists']}")
        print(f"   Target        : {fmt(target)}")
        print(f"   Actual (text) : {fmt(actual)}  ({r.get('completion_rate','N/A')})")
        img_str = fmt(r['image_count']) if r['image_count'] >= 0 else "(skipped)"
        print(f"   Images        : {img_str}")
        print(f"   Mature rate   : {ps.get('mature_rate','N/A')}")
        print(f"   Label rate    : {ps.get('label_rate','N/A')}")

        files = r.get("files", {})
        if files:
            print("   File counts:")
            for fname, cnt in files.items():
                flag = " ⚠" if (cnt != actual and fname not in ("extra_user_data.jsonl", "crawl_manifest.jsonl")) else ""
                print(f"     {fname:<54} {fmt(cnt):>8}{flag}")

        for w in r["warnings"]:
            print(f"   ⚠  {w}")

    print("\n" + "=" * W)
    print(f"{'SUMMARY':^{W}}")
    print("=" * W)
    status_str = "✅ All OK" if all_ok else "❌ Issues found"
    print(f"  Status          : {status_str}")
    print(f"  Grand total     : {fmt(grand_actual)} / {fmt(grand_target)} ({pct(grand_actual, grand_target)})")
    if grand_images >= 0:
        print(f"  Images on disk  : {fmt(grand_images)}")
    print(f"  Valid labels    : {fmt(grand_label)} ({pct(grand_label, grand_actual)})")
    print("=" * W)


# ──────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check crawl output completeness.")
    p.add_argument("--base_dir",    default="/local/smp/extra_data")
    p.add_argument("--total_items", type=int, default=TOTAL_ITEMS_DEFAULT)
    p.add_argument("--skip_images", action="store_true", help="Skip image file counting (faster)")
    return p.parse_args()


def main() -> None:
    args     = parse_args()
    base_dir = Path(args.base_dir)

    if not base_dir.exists():
        print(f"❌ Base directory not found: {base_dir}")
        return

    print(f"Checking: {base_dir}")
    print(f"Total target: {fmt(args.total_items)}\n")

    results = []
    for cat_name, _query, base_target in CATEGORIES:
        print(f"  Checking {cat_name}...", end="\r", flush=True)
        r = check_category(cat_name, base_target, args.total_items, base_dir, args.skip_images)
        results.append(r)

    print(" " * 60, end="\r")
    print_report(results, args.total_items)


if __name__ == "__main__":
    main()
