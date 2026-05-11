#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_uid_text_crawl.py

Orchestrator: reads train_user_data.json + train_category.json,
groups users by their top category, then calls
crawl_flickr_to_smp_by_uid_text.py for each uid.

Called by run_smp_uid_text_by_category.sh — do not run directly unless
you set the required environment variables.

Environment variables (set by shell wrapper):
  PYTHON_BIN, CRAWLER_SCRIPT, TRAIN_USER_JSON, TRAIN_CATEGORY_JSON,
  BASE_OUTPUT_DIR, MAX_USERS_PER_CATEGORY, MAX_ITEMS_PER_UID,
  LICENSES, SLEEP_MIN, SLEEP_MAX, FLUSH_EVERY, PER_PAGE, SORT,
  DOWNLOAD_IMAGES
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Config from environment ───────────────────────────────────────────────
PYTHON_BIN           = os.environ.get("PYTHON_BIN", "python3")
CRAWLER_SCRIPT       = os.environ.get("CRAWLER_SCRIPT", "./crawl_flickr_to_smp_by_uid_text.py")
TRAIN_USER_JSON      = Path(os.environ.get("TRAIN_USER_JSON",
                             "/local/smp/data/train_allmetadata_json/train_user_data.json"))
TRAIN_CATEGORY_JSON  = Path(os.environ.get("TRAIN_CATEGORY_JSON",
                             "/local/smp/data/train_allmetadata_json/train_category.json"))
BASE_OUTPUT_DIR      = Path(os.environ.get("BASE_OUTPUT_DIR", "/local/smp/extra_data_by_uid"))
MAX_USERS_PER_CATEGORY = int(os.environ.get("MAX_USERS_PER_CATEGORY", "0"))
MAX_ITEMS_PER_UID    = int(os.environ.get("MAX_ITEMS_PER_UID", "10"))
LICENSES             = os.environ.get("LICENSES", "4,5,7,8,9,10")
SLEEP_MIN            = os.environ.get("SLEEP_MIN", "0.8")
SLEEP_MAX            = os.environ.get("SLEEP_MAX", "2.0")
FLUSH_EVERY          = os.environ.get("FLUSH_EVERY", "100")
PER_PAGE             = os.environ.get("PER_PAGE", "100")
SORT                 = os.environ.get("SORT", "date-posted-desc")
DOWNLOAD_IMAGES      = os.environ.get("DOWNLOAD_IMAGES", "1") == "1"

CATEGORY_TO_QUERY: dict[str, str] = {
    "Travel&Active&Sports": "travel",
    "Holiday&Celebrations": "holiday",
    "Fashion":              "fashion",
    "Entertainment":        "concert",
    "Social&People":        "people",
    "Whether&Season":       "winter",
    "Animal":               "animal",
    "Food":                 "food",
    "Urban":                "city",
    "Electronics":          "electronics",
    "Family":               "family",
}


def sanitize_name(name: str) -> str:
    return name.replace("&", "_").replace(" ", "_").replace("/", "_")


def load_json_or_jsonl(path: Path) -> list:
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    if path.suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for key in ("data", "items", "records", "results"):
            if key in data and isinstance(data[key], list):
                return data[key]
        first_val = next(iter(data.values()), None)
        if isinstance(first_val, dict):
            import pandas as pd
            df = pd.DataFrame(data)
            return df.to_dict(orient="records")
        if all(isinstance(v, dict) for v in data.values()):
            return list(data.values())
        raise ValueError(f"Unsupported JSON structure in {path}")
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported JSON structure in {path}")


def build_category_uid_map() -> dict[str, list[tuple[str, int]]]:
    """
    Returns { category: [(uid, train_post_count), ...] } sorted by post count desc.
    Each user is assigned to their most frequent category in the train set.
    """
    log.info("Loading train_user_data from %s", TRAIN_USER_JSON)
    users = load_json_or_jsonl(TRAIN_USER_JSON)

    log.info("Loading train_category from %s", TRAIN_CATEGORY_JSON)
    cats = load_json_or_jsonl(TRAIN_CATEGORY_JSON)

    uid_post_count: Counter = Counter()
    for row in users:
        uid = row.get("Uid")
        if uid is not None:
            uid_post_count[str(uid)] += 1

    uid_cat_counter: defaultdict = defaultdict(Counter)
    for row in cats:
        uid = row.get("Uid")
        cat = row.get("Category")
        if uid is not None and cat not in (None, ""):
            uid_cat_counter[str(uid)][str(cat)] += 1

    category_to_uids: defaultdict = defaultdict(list)
    for uid, count in uid_post_count.items():
        if uid not in uid_cat_counter or not uid_cat_counter[uid]:
            continue
        top_cat = uid_cat_counter[uid].most_common(1)[0][0]
        if top_cat not in CATEGORY_TO_QUERY:
            continue
        category_to_uids[top_cat].append((uid, count))

    result = {}
    for cat, uid_rows in category_to_uids.items():
        uid_rows.sort(key=lambda x: (-x[1], x[0]))
        if MAX_USERS_PER_CATEGORY > 0:
            uid_rows = uid_rows[:MAX_USERS_PER_CATEGORY]
        result[cat] = uid_rows

    return result


def save_summary(category_uid_map: dict, output_dir: Path) -> None:
    summary = {
        cat: [{"Uid": uid, "train_post_count": cnt} for uid, cnt in rows]
        for cat, rows in category_uid_map.items()
    }
    path = output_dir / "train_uid_text_category_summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Saved uid summary to %s", path)

    total_uids  = sum(len(v) for v in category_uid_map.values())
    total_items = total_uids * MAX_ITEMS_PER_UID
    log.info("Total users to crawl : %d", total_uids)
    log.info("Max items per uid    : %d", MAX_ITEMS_PER_UID)
    log.info("Estimated total items: %d", total_items)
    for cat, rows in sorted(category_uid_map.items()):
        log.info("  %-30s %d users → ~%d items", cat, len(rows), len(rows) * MAX_ITEMS_PER_UID)


def crawl_uid(uid: str, category: str, out_dir: Path) -> None:
    query = CATEGORY_TO_QUERY[category]
    cmd = [
        PYTHON_BIN, CRAWLER_SCRIPT,
        "--output_dir",    str(out_dir),
        "--user_id",       uid,
        "--text",          query,
        "--seed_category", category,
        "--max_items",     str(MAX_ITEMS_PER_UID),
        "--licenses",      LICENSES,
        "--sleep_min",     SLEEP_MIN,
        "--sleep_max",     SLEEP_MAX,
        "--flush_every",   FLUSH_EVERY,
        "--per_page",      PER_PAGE,
        "--sort",          SORT,
        "--resume",
        "--dedupe_on_image_path",
        "--crawl_mode",    "by_uid_text",
    ]
    if DOWNLOAD_IMAGES:
        cmd.append("--download_images")
    subprocess.run(cmd, check=True)


def main() -> int:
    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    category_uid_map = build_category_uid_map()
    save_summary(category_uid_map, BASE_OUTPUT_DIR)

    total_uids = sum(len(v) for v in category_uid_map.values())
    done = 0

    for category, uid_rows in category_uid_map.items():
        safe_cat = sanitize_name(category)
        out_dir  = BASE_OUTPUT_DIR / f"extra_data_{safe_cat}_by_uid_text"
        out_dir.mkdir(parents=True, exist_ok=True)

        for uid, train_count in uid_rows:
            done += 1
            log.info(
                "[%d/%d] category=%-30s uid=%-25s train_posts=%d",
                done, total_uids, category, uid, train_count,
            )
            try:
                crawl_uid(uid, category, out_dir)
            except subprocess.CalledProcessError as e:
                log.error("Crawler failed for uid=%s: %s — skipping", uid, e)
                continue

    log.info("All category-based uid+text crawls finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
