#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_flickr_uids.py

Pre-validate Flickr user IDs from train_user_data.json before crawling.
Filters out deleted/suspended accounts to avoid wasting API quota.

Usage:
    python3 validate_flickr_uids.py \\
        --train_user_json /local/smp/data/train_allmetadata_json/train_user_data.json \\
        --output /local/smp/extra_data_by_uid/valid_uids.json \\
        --sleep_min 0.3 \\
        --sleep_max 0.8

Output:
    valid_uids.json   — {"uid1": true, "uid2": false, ...}
    valid_uids.txt    — one valid uid per line (for easy reuse)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

FLICKR_REST_URL = "https://api.flickr.com/services/rest"
DEFAULT_TIMEOUT = 15
USER_AGENT = "smp-uid-validator/1.0"


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def random_sleep(min_s: float, max_s: float) -> None:
    time.sleep(random.uniform(min_s, max_s))


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
        raise ValueError(f"Unsupported JSON structure: keys={list(data.keys())[:5]}")
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported JSON structure: {type(data)}")


def extract_uids(records: list) -> list[str]:
    uids = []
    seen = set()
    for row in records:
        if not isinstance(row, dict):
            continue
        uid = row.get("Uid")
        if uid is not None:
            uid_str = str(uid)
            if uid_str not in seen:
                seen.add(uid_str)
                uids.append(uid_str)
    return uids


def load_existing_results(path: Path) -> Dict[str, bool]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


class FlickrClient:
    def __init__(self, api_key: str, timeout: int = DEFAULT_TIMEOUT) -> None:
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    def get_people_info(self, user_id: str) -> Optional[Dict[str, Any]]:
        payload = {
            "method":        "flickr.people.getInfo",
            "api_key":       self.api_key,
            "user_id":       user_id,
            "format":        "json",
            "nojsoncallback": 1,
        }
        try:
            resp = self.session.get(FLICKR_REST_URL, params=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            raise RuntimeError(f"Network error for uid={user_id}: {e}") from e

        if data.get("stat") == "ok":
            return data.get("person", {})

        code    = data.get("code")
        message = data.get("message", "")
        if code in (1, 2) or "unknown user" in message.lower():
            return None
        raise RuntimeError(f"Flickr API error code={code} message={message}")


def validate_uids(
    uids: list[str],
    client: FlickrClient,
    existing: Dict[str, bool],
    output_path: Path,
    sleep_min: float,
    sleep_max: float,
    save_every: int = 500,
) -> Dict[str, bool]:
    results = dict(existing)
    to_check = [uid for uid in uids if uid not in results]

    total       = len(uids)
    already     = len(existing)
    valid_count = sum(1 for v in results.values() if v)
    invalid_count = sum(1 for v in results.values() if not v)

    logging.info("Total UIDs       : %d", total)
    logging.info("Already validated: %d", already)
    logging.info("To check now     : %d", len(to_check))

    for i, uid in enumerate(to_check, start=1):
        try:
            person   = client.get_people_info(uid)
            is_valid = person is not None
        except RuntimeError as e:
            logging.warning("Unexpected error uid=%s: %s — treating as invalid", uid, e)
            is_valid = False

        results[uid] = is_valid
        if is_valid:
            valid_count += 1
        else:
            invalid_count += 1
            logging.info("[%d/%d] uid=%-25s INVALID", already + i, total, uid)

        if i % 100 == 0:
            logging.info(
                "Progress: %d/%d checked | valid=%d invalid=%d (%.1f%% valid)",
                already + i, total, valid_count, invalid_count,
                100 * valid_count / max(already + i, 1),
            )

        if i % save_every == 0:
            _save_results(results, output_path)

        random_sleep(sleep_min, sleep_max)

    _save_results(results, output_path)
    return results


def _save_results(results: Dict[str, bool], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


def save_valid_uid_list(results: Dict[str, bool], txt_path: Path) -> None:
    valid_uids = [uid for uid, ok in results.items() if ok]
    txt_path.write_text("\n".join(valid_uids) + "\n", encoding="utf-8")
    logging.info("Valid UID list saved to %s (%d UIDs)", txt_path, len(valid_uids))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pre-validate Flickr UIDs before crawling.")
    p.add_argument("--train_user_json", default="/local/smp/data/train_allmetadata_json/train_user_data.json")
    p.add_argument("--output",          default="/local/smp/extra_data_by_uid/valid_uids.json")
    p.add_argument("--sleep_min",  type=float, default=0.3)
    p.add_argument("--sleep_max",  type=float, default=0.8)
    p.add_argument("--save_every", type=int,   default=500)
    p.add_argument("--api_key",    default=os.getenv("FLICKR_API_KEY"))
    p.add_argument("--verbose",    action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    if not args.api_key:
        logging.error("Missing FLICKR_API_KEY.")
        return 1

    output_path = Path(args.output)
    txt_path    = output_path.with_suffix(".txt")

    records = load_json_or_jsonl(Path(args.train_user_json))
    uids    = extract_uids(records)
    logging.info("Extracted %d unique UIDs", len(uids))

    existing = load_existing_results(output_path)
    if existing:
        logging.info("Resuming from %d already validated", len(existing))

    client  = FlickrClient(api_key=args.api_key)
    results = validate_uids(
        uids=uids, client=client, existing=existing,
        output_path=output_path,
        sleep_min=args.sleep_min, sleep_max=args.sleep_max,
        save_every=args.save_every,
    )

    valid   = sum(1 for v in results.values() if v)
    invalid = sum(1 for v in results.values() if not v)
    logging.info("Done: %d valid, %d invalid (%.1f%% valid)",
                 valid, invalid, 100 * valid / max(len(results), 1))
    save_valid_uid_list(results, txt_path)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
