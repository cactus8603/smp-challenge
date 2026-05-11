#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
crawl_flickr_to_smp_by_uid_text.py

Crawl Flickr photos for a specific user ID + text query combination.
Outputs the same SMP-format files as crawl_flickr_to_smp.py,
appending into a shared output_dir per category.

This script is called per-uid by run_uid_text_crawl.py (the orchestrator).
It is intentionally thin: all heavy logic lives in crawl_flickr_to_smp.py.

Usage (standalone):
  python3 crawl_flickr_to_smp_by_uid_text.py \\
      --output_dir /local/smp/extra_data_by_uid/extra_data_Travel_by_uid_text \\
      --user_id 12345678@N00 \\
      --text travel \\
      --seed_category "Travel&Active&Sports" \\
      --max_items 10 \\
      --resume \\
      --dedupe_on_image_path
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Re-use everything from the main crawler
from crawl_flickr_to_smp import FlickrClient, QUERY_TO_CATEGORY, crawl, setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Crawl Flickr by uid+text → SMP format.")
    p.add_argument("--output_dir",           required=True)
    p.add_argument("--user_id",              required=True,  help="Flickr numeric user ID")
    p.add_argument("--text",                 required=True,  help="Search query text")
    p.add_argument("--seed_category",        default=None)
    p.add_argument("--max_items",            type=int, default=10)
    p.add_argument("--licenses",             default="4,5,7,8,9,10")
    p.add_argument("--sort",                 default="date-posted-desc")
    p.add_argument("--per_page",             type=int, default=100)
    p.add_argument("--max_no_new_pages",     type=int, default=5)
    p.add_argument("--sleep_min",            type=float, default=0.8)
    p.add_argument("--sleep_max",            type=float, default=2.0)
    p.add_argument("--flush_every",          type=int, default=100)
    p.add_argument("--download_images",      action="store_true")
    p.add_argument("--resume",               action="store_true")
    p.add_argument("--dedupe_on_image_path", action="store_true")
    p.add_argument("--crawl_mode",           default="by_uid_text")
    p.add_argument("--api_key",              default=os.getenv("FLICKR_API_KEY"))
    p.add_argument("--verbose",              action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    if not args.api_key:
        raise SystemExit("ERROR: FLICKR_API_KEY not set.")

    seed_cat = args.seed_category or QUERY_TO_CATEGORY.get(args.text.lower())
    client = FlickrClient(api_key=args.api_key)

    crawl(
        client=client,
        output_dir=Path(args.output_dir),
        text=args.text,
        max_items=args.max_items,
        licenses=args.licenses,
        sort=args.sort,
        per_page=args.per_page,
        max_no_new_pages=args.max_no_new_pages,
        sleep_min=args.sleep_min,
        sleep_max=args.sleep_max,
        flush_every=args.flush_every,
        download_images=args.download_images,
        resume=args.resume,
        dedupe_on_image_path=args.dedupe_on_image_path,
        seed_category=seed_cat,
        user_id=args.user_id,
        crawl_mode=args.crawl_mode,
    )


if __name__ == "__main__":
    main()
