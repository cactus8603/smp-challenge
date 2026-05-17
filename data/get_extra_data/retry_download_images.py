#!/usr/bin/env python3
"""
retry_download_images.py

Scan all category output directories, find image paths listed in
extra_img_filepath.txt that don't have a file on disk, look up their
URLs from crawl_manifest.jsonl, and retry with exponential backoff.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path

import requests

USER_AGENT = "smp-like-flickr-crawler/1.0"


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_url_map(manifest_path: Path) -> dict[str, str]:
    """Build rel_path -> image_url map from crawl_manifest.jsonl."""
    url_map: dict[str, str] = {}
    if not manifest_path.exists():
        return url_map
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                rel = obj.get("relative_image_path")
                url = obj.get("image_url")
                if rel and url:
                    url_map[rel] = url
            except json.JSONDecodeError:
                continue
    return url_map


def download_with_retry(
    session: requests.Session,
    url: str,
    out_path: Path,
    sleep_min: float,
    sleep_max: float,
    timeout: int,
    max_retries: int = 5,
) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(max_retries):
        try:
            resp = session.get(
                url, stream=True, timeout=timeout, headers={"User-Agent": USER_AGENT}
            )

            if resp.status_code == 429:
                # Exponential backoff on rate limit
                wait = (2 ** attempt) * random.uniform(15, 30)
                logging.warning(
                    "429 Too Many Requests (attempt %d/%d). Waiting %.0f s...",
                    attempt + 1, max_retries, wait,
                )
                time.sleep(wait)
                continue

            resp.raise_for_status()

            with out_path.open("wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            time.sleep(random.uniform(sleep_min, sleep_max))
            return True

        except requests.HTTPError as e:
            if attempt < max_retries - 1:
                wait = (2 ** attempt) * random.uniform(5, 10)
                logging.warning(
                    "HTTP error %s (attempt %d/%d). Waiting %.0f s...",
                    e, attempt + 1, max_retries, wait,
                )
                time.sleep(wait)
            else:
                logging.error("Giving up after %d retries: %s", max_retries, url)
                return False

        except Exception as e:
            logging.error("Download error %s -> %s: %s", url, out_path, e)
            return False

    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retry missing image downloads.")
    parser.add_argument(
        "--base_output_dir",
        type=str,
        default="/ssd1/lchiayu/smp-challenge/data/get_extra_data/output",
        help="Base directory containing extra_data_* subdirs.",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Only process this category dir name (e.g. extra_data_Animal). "
             "If omitted, all categories are processed.",
    )
    parser.add_argument("--sleep_min", type=float, default=3.0)
    parser.add_argument("--sleep_max", type=float, default=6.0)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    base_dir = Path(args.base_output_dir)
    session = requests.Session()

    total_missing = 0
    total_success = 0
    total_fail = 0

    cat_dirs = sorted(
        d for d in base_dir.iterdir()
        if d.is_dir() and d.name.startswith("extra_data_")
        and (args.category is None or d.name == args.category)
    )

    for cat_dir in cat_dirs:
        img_txt = cat_dir / "extra_img_filepath.txt"
        manifest_jsonl = cat_dir / "crawl_manifest.jsonl"
        image_dir = cat_dir / "images"

        if not img_txt.exists():
            continue

        logging.info("=== %s ===", cat_dir.name)

        url_map = load_url_map(manifest_jsonl)
        if not url_map:
            logging.warning("Empty or missing manifest: %s", manifest_jsonl)

        # Collect missing paths
        missing: list[str] = []
        with img_txt.open("r", encoding="utf-8") as f:
            for line in f:
                rel_path = line.strip()
                if rel_path and not (image_dir / rel_path).exists():
                    missing.append(rel_path)

        total_missing += len(missing)
        logging.info("Missing: %d images", len(missing))

        for rel_path in missing:
            url = url_map.get(rel_path)
            if not url:
                logging.warning("No URL in manifest for: %s", rel_path)
                total_fail += 1
                continue

            logging.info("Downloading: %s", rel_path)
            ok = download_with_retry(
                session=session,
                url=url,
                out_path=image_dir / rel_path,
                sleep_min=args.sleep_min,
                sleep_max=args.sleep_max,
                timeout=args.timeout,
                max_retries=args.max_retries,
            )
            if ok:
                total_success += 1
            else:
                total_fail += 1

    logging.info(
        "Done. Total missing=%d | Downloaded=%d | Failed=%d",
        total_missing, total_success, total_fail,
    )


if __name__ == "__main__":
    main()
