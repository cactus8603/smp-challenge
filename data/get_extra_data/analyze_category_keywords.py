#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_category_keywords.py

分析官方 train data 中每個 category 最常出現的 tag/title 詞，
輸出可直接貼進 shell script 的 extra_queries 格式。

用法：
    python3 analyze_category_keywords.py \
        --data_dir /local/smp/data/train \
        --output   /local/smp/category_keywords.json \
        --topk     50

    # 只看某個 category
    python3 analyze_category_keywords.py \
        --data_dir /local/smp/data/train \
        --cat Animal \
        --topk 30
"""

from __future__ import annotations

import argparse
import json
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set


# ──────────────────────────────────────────────────────
# Stopwords（過濾掉無意義的詞）
# ──────────────────────────────────────────────────────
STOPWORDS: Set[str] = {
    # 太泛的詞
    "photo", "picture", "image", "img", "pic", "photos", "pictures",
    "flickr", "nikon", "canon", "sony", "olympus", "fuji", "fujifilm",
    "dslr", "camera", "lens", "raw", "jpg", "jpeg", "png",
    # 太短或無意義
    "a", "an", "the", "in", "on", "at", "to", "of", "for", "by",
    "is", "are", "was", "be", "it", "my", "me", "i", "we",
    "this", "that", "with", "from", "and", "or", "not",
    # 數字
    "2004", "2005", "2006", "2007", "2008", "2009", "2010",
    "2011", "2012", "2013", "2014", "2015", "2016", "2017",
    "2018", "2019", "2020", "2021", "2022", "2023", "2024",
    # 常見但無區分度的詞
    "new", "old", "big", "small", "beautiful", "great", "good",
    "best", "nice", "cool", "amazing", "awesome", "life", "world",
    "day", "night", "time", "year", "people", "man", "woman",
    "color", "colour", "light", "dark", "white", "black",
    "red", "blue", "green", "yellow",
}


# ──────────────────────────────────────────────────────
# Text cleaning
# ──────────────────────────────────────────────────────
def clean_tag(tag: str) -> str:
    """Clean a single tag: lowercase, remove punctuation, strip."""
    tag = tag.lower().strip()
    tag = tag.strip('"').strip("'")
    # remove non-alphanumeric except hyphen
    tag = re.sub(r"[^\w\-]", "", tag)
    tag = tag.strip("-_")
    return tag


def parse_tags(alltags: Optional[str]) -> List[str]:
    """Parse SMP Alltags format: '"tag1" "tag2" ...' → ['tag1', 'tag2', ...]"""
    if not alltags or str(alltags).lower() in ("none", "nan", "null", ""):
        return []
    raw = str(alltags)
    # Try quoted format first
    tokens = re.findall(r'"([^"]*)"', raw)
    if not tokens:
        # fallback: space-separated
        tokens = raw.split()
    return [clean_tag(t) for t in tokens if clean_tag(t)]


def parse_title(title: Optional[str]) -> List[str]:
    """Tokenize title into words."""
    if not title or str(title).lower() in ("none", "nan", "null", ""):
        return []
    raw = str(title).lower()
    # remove punctuation
    raw = raw.translate(str.maketrans("", "", string.punctuation))
    return [w.strip() for w in raw.split() if len(w.strip()) > 2]


# ──────────────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────────────
def load_json(path: Path) -> list:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # column-oriented format
        keys = list(data.keys())
        if not keys:
            return []
        n = len(data[keys[0]])
        return [{k: data[k][str(i)] for k in keys} for i in range(n)]
    return []


# ──────────────────────────────────────────────────────
# Main analysis
# ──────────────────────────────────────────────────────
def analyze(
    data_dir: Path,
    topk: int,
    min_count: int,
    use_tags: bool,
    use_title: bool,
    filter_cat: Optional[str],
) -> Dict[str, List[str]]:

    # Load category data
    cat_path = data_dir / "train_category.json"
    text_path = data_dir / "train_text.json"

    print(f"Loading category: {cat_path}")
    cat_data = load_json(cat_path)

    print(f"Loading text: {text_path}")
    text_data = load_json(text_path)

    if not cat_data or not text_data:
        raise SystemExit("Could not load data. Check paths.")

    # Build Pid → category map
    pid_to_cat: Dict[str, str] = {}
    for row in cat_data:
        pid = str(row.get("Pid", ""))
        cat = str(row.get("Category", "")).strip()
        if pid and cat:
            pid_to_cat[pid] = cat

    # Build Pid → text data map
    pid_to_text: Dict[str, dict] = {}
    for row in text_data:
        pid = str(row.get("Pid", ""))
        if pid:
            pid_to_text[pid] = row

    print(f"Total PIDs with category: {len(pid_to_cat)}")
    print(f"Total PIDs with text: {len(pid_to_text)}")

    # Count words per category
    cat_counters: Dict[str, Counter] = defaultdict(Counter)
    cat_totals: Dict[str, int] = defaultdict(int)

    for pid, cat in pid_to_cat.items():
        if filter_cat and filter_cat.lower() not in cat.lower():
            continue

        row = pid_to_text.get(pid, {})
        words = []

        if use_tags:
            words += parse_tags(row.get("Alltags"))

        if use_title:
            words += parse_title(row.get("Title"))

        # filter stopwords and too-short words
        words = [
            w for w in words
            if w not in STOPWORDS and len(w) >= 3 and not w.isdigit()
        ]

        cat_counters[cat].update(words)
        cat_totals[cat] += 1

    # Build top-k keywords per category
    result: Dict[str, List[str]] = {}

    cats = sorted(cat_counters.keys())
    for cat in cats:
        counter = cat_counters[cat]
        total = cat_totals[cat]

        # Filter by min_count
        top = [
            word for word, cnt in counter.most_common(topk * 3)
            if cnt >= min_count
        ][:topk]

        result[cat] = top

        print(f"\n{'='*60}")
        print(f"Category: {cat}  (total posts: {total:,})")
        print(f"{'='*60}")
        for i, (word, cnt) in enumerate(counter.most_common(topk)):
            if cnt < min_count:
                break
            pct = 100 * cnt / max(total, 1)
            print(f"  {i+1:3d}. {word:<25s} {cnt:6,}  ({pct:.1f}%)")

    return result


# ──────────────────────────────────────────────────────
# Output
# ──────────────────────────────────────────────────────
def print_shell_format(result: Dict[str, List[str]]) -> None:
    """Print keywords in shell script CATEGORIES array format."""
    print("\n" + "=" * 70)
    print("SHELL SCRIPT FORMAT (extra_queries)")
    print("=" * 70)

    for cat, keywords in result.items():
        kw_str = ",".join(keywords)
        safe_cat = cat.replace("&", "_").replace(" ", "_")
        print(f"\n# {cat}")
        print(f'"{cat}|<primary_query>|{kw_str}|<base_target>"')


def main() -> None:
    p = argparse.ArgumentParser(description="Analyze category keywords from official train data.")
    p.add_argument("--data_dir",  default="/local/smp/data/train",
                   help="Path to train data directory")
    p.add_argument("--output",    default=None,
                   help="Save result to JSON file")
    p.add_argument("--topk",      type=int, default=50,
                   help="Top K keywords per category")
    p.add_argument("--min_count", type=int, default=10,
                   help="Minimum occurrence count to include a keyword")
    p.add_argument("--cat",       default=None,
                   help="Filter to one category (partial match)")
    p.add_argument("--no_tags",   action="store_true",
                   help="Skip Alltags field")
    p.add_argument("--no_title",  action="store_true",
                   help="Skip Title field")
    p.add_argument("--shell",     action="store_true",
                   help="Print in shell script format")
    args = p.parse_args()

    result = analyze(
        data_dir=Path(args.data_dir),
        topk=args.topk,
        min_count=args.min_count,
        use_tags=not args.no_tags,
        use_title=not args.no_title,
        filter_cat=args.cat,
    )

    if args.shell:
        print_shell_format(result)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
