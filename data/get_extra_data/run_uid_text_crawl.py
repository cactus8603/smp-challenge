#!/usr/bin/env python3
"""
run_uid_text_crawl.py

Orchestrator: reads train_user_data.json + train_category.json,
groups users by their top category, then calls crawl_flickr_to_smp_by_uid_text.py
for each uid.

Called by run_smp_uid_text_by_category.sh — do not run directly unless you
set the required environment variables.
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

# -------------------------------------------------------
# Config from environment (set by the shell wrapper)
# -------------------------------------------------------
PYTHON_BIN          = os.environ["PYTHON_BIN"]
CRAWLER_SCRIPT      = os.environ["CRAWLER_SCRIPT"]
TRAIN_USER_JSON     = Path(os.environ["TRAIN_USER_JSON"])
TRAIN_CATEGORY_JSON = Path(os.environ["TRAIN_CATEGORY_JSON"])
BASE_OUTPUT_DIR     = Path(os.environ["BASE_OUTPUT_DIR"])
MAX_USERS_PER_CATEGORY = int(os.environ.get("MAX_USERS_PER_CATEGORY", "0"))
MAX_ITEMS_PER_UID   = int(os.environ.get("MAX_ITEMS_PER_UID", "2"))
LICENSES            = os.environ.get("LICENSES", "4,5,7,8,9,10")
SLEEP_MIN           = os.environ.get("SLEEP_MIN", "0.8")
SLEEP_MAX           = os.environ.get("SLEEP_MAX", "2.0")
FLUSH_EVERY         = os.environ.get("FLUSH_EVERY", "100")
PER_PAGE            = os.environ.get("PER_PAGE", "100")
SORT                = os.environ.get("SORT", "date-posted-desc")
DOWNLOAD_IMAGES     = os.environ.get("DOWNLOAD_IMAGES", "1") == "1"

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
        raise FileNotFoundError(f"File not found: {path}")
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

    # Handle common dict wrapper structures
    if isinstance(data, dict):
        # Case 1: {"data": [...]} / {"items": [...]} wrapper
        for key in ("data", "items", "records", "results"):
            if key in data and isinstance(data[key], list):
                logging.info("Unwrapped JSON key '%s' from %s", key, path.name)
                return data[key]

        # Case 2: pandas column-oriented format
        # {"col1": {"0": v, "1": v, ...}, "col2": {"0": v, "1": v, ...}}
        first_val = next(iter(data.values()), None)
        if isinstance(first_val, dict):
            import pandas as pd
            logging.info("Detected pandas column-oriented JSON in %s, converting to records...", path.name)
            df = pd.DataFrame(data)
            records = df.to_dict(orient="records")
            logging.info("Converted %d records from %s", len(records), path.name)
            return records

        # Case 3: dict of uid -> record (e.g. {"uid1": {...}, "uid2": {...}})
        if all(isinstance(v, dict) for v in data.values()):
            logging.info("Converted dict-of-records to list from %s (%d entries)", path.name, len(data))
            return list(data.values())

        raise ValueError(
            f"Unsupported JSON structure in {path}: top-level dict with keys {list(data.keys())[:5]}"
        )

    if isinstance(data, list):
        return data

    raise ValueError(f"Unsupported JSON structure in {path}: expected list or dict, got {type(data)}")


def build_category_uid_map() -> dict[str, list[tuple[str, int]]]:
    """
    Returns { category: [(uid, train_post_count), ...] }
    sorted by train_post_count desc.
    Users are assigned to their most frequent category in train.
    """
    logging.info("Loading train_user_data from %s", TRAIN_USER_JSON)
    users = load_json_or_jsonl(TRAIN_USER_JSON)

    logging.info("Loading train_category from %s", TRAIN_CATEGORY_JSON)
    cats = load_json_or_jsonl(TRAIN_CATEGORY_JSON)

    # Count posts per uid
    uid_post_count: Counter = Counter()
    for row in users:
        uid = row.get("Uid")
        if uid is not None:
            uid_post_count[str(uid)] += 1

    # Count category appearances per uid
    uid_cat_counter: defaultdict = defaultdict(Counter)
    for row in cats:
        uid = row.get("Uid")
        cat = row.get("Category")
        if uid is not None and cat not in (None, ""):
            uid_cat_counter[str(uid)][str(cat)] += 1

    # Assign each uid to its top category
    category_to_uids: defaultdict = defaultdict(list)
    for uid, count in uid_post_count.items():
        if uid not in uid_cat_counter or not uid_cat_counter[uid]:
            continue
        top_cat = uid_cat_counter[uid].most_common(1)[0][0]
        if top_cat not in CATEGORY_TO_QUERY:
            continue
        category_to_uids[top_cat].append((uid, count))

    # Sort by post count desc, apply cap
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
    logging.info("Saved uid summary to %s", path)

    # Print stats
    total_uids = sum(len(v) for v in category_uid_map.values())
    total_items = total_uids * MAX_ITEMS_PER_UID
    logging.info("Total users to crawl : %d", total_uids)
    logging.info("Max items per uid    : %d", MAX_ITEMS_PER_UID)
    logging.info("Estimated total items: %d", total_items)
    for cat, rows in sorted(category_uid_map.items()):
        logging.info("  %-30s %d users → ~%d items", cat, len(rows), len(rows) * MAX_ITEMS_PER_UID)


def crawl_uid(uid: str, category: str, out_dir: Path) -> None:
    query = CATEGORY_TO_QUERY[category]

    cmd = [
        PYTHON_BIN,
        CRAWLER_SCRIPT,
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
        out_dir = BASE_OUTPUT_DIR / f"extra_data_{safe_cat}_by_uid_text"
        out_dir.mkdir(parents=True, exist_ok=True)

        for uid, train_count in uid_rows:
            done += 1
            logging.info(
                "[%d/%d] category=%-30s uid=%-25s train_posts=%d",
                done, total_uids, category, uid, train_count,
            )
            try:
                crawl_uid(uid, category, out_dir)
            except subprocess.CalledProcessError as e:
                logging.error("Crawler failed for uid=%s: %s — skipping", uid, e)
                continue

    logging.info("All category-based uid+text crawls finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
