#!/usr/bin/env python3
"""
crawl_flickr_to_smp.py

Use Flickr public API to collect licensed public data and export it into
SMP-like files.

Features
--------
- Randomized sleep between requests to reduce request burst.
- Keep missing/unavailable fields in output instead of dropping them.
- Explicitly mark fields that are SMP-only annotations and cannot be
  directly obtained from Flickr API.
- Export files with names and structures aligned to SMP challenge style.

Notes
-----
1. This script uses Flickr's public API, not HTML scraping.
2. You must provide your own Flickr API key.
3. "Exactly identical" to SMP is not always possible because some SMP
   fields are dataset-specific annotations. Those fields are preserved as
   null and marked with metadata.
4. This script only fetches public content and can optionally filter by
   license IDs.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from datetime import datetime, timezone

import requests

from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("FLICKR_API_KEY")

"""
python crawl_flickr_to_smp.py --output_dir ./external_smp_data --text "street photography" --licenses "4,5,7,8,9,10" --max_items 400 --download_images --sleep_min 0.8 --sleep_max 2.0 --flush_every 200 --resume --dedupe_on_image_path
python crawl_flickr_to_smp.py --output_dir ./external_smp_data --text "street photography" --licenses "4,5,7,8,9,10" --max_items 200 --download_images --flush_every 100 --resume --dedupe_on_image_path --write_pretty_json
"""


FLICKR_REST_URL = "https://api.flickr.com/services/rest"
DEFAULT_TIMEOUT = 30
USER_AGENT = "smp-like-flickr-crawler/1.0"


# -----------------------------
# Utilities
# -----------------------------
def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def random_sleep(min_seconds: float, max_seconds: float) -> None:
    duration = random.uniform(min_seconds, max_seconds)
    logging.debug("Sleeping for %.3f seconds", duration)
    time.sleep(duration)


def safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def log1p_or_none(value: Any) -> Optional[float]:
    v = safe_float(value)
    if v is None:
        return None
    if v < 0:
        return None
    return math.log1p(v)


def dump_json_lines(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl_record(handle, row: Dict[str, Any], flush: bool = False) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if flush:
        handle.flush()


def append_text_line(handle, value: str, flush: bool = False) -> None:
    handle.write(f"{value}\n")
    if flush:
        handle.flush()


def load_existing_pids_from_jsonl(path: Path, key: str = "Pid") -> set[str]:
    existing: set[str] = set()
    if not path.exists():
        return existing

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            value = obj.get(key)
            if value is not None:
                existing.add(str(value))
    return existing


def dump_json_pretty(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_txt_lines(path: Path, lines: Iterable[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")


def sanitize_text(text: Any) -> str:
    if text is None:
        return ""
    return str(text).replace("\r", " ").replace("\n", " ").strip()


# -----------------------------
# Flickr client
# -----------------------------
@dataclass
class FlickrClient:
    api_key: str
    sleep_min: float = 0.5
    sleep_max: float = 1.5
    timeout: int = DEFAULT_TIMEOUT
    session: Optional[requests.Session] = None

    def __post_init__(self) -> None:
        if self.session is None:
            self.session = requests.Session()
            self.session.headers.update({"User-Agent": USER_AGENT})

    def call(self, method: str, **params: Any) -> Dict[str, Any]:
        payload = {
            "method": method,
            "api_key": self.api_key,
            "format": "json",
            "nojsoncallback": 1,
            **params,
        }
        resp = self.session.get(FLICKR_REST_URL, params=payload, timeout=self.timeout)
        random_sleep(self.sleep_min, self.sleep_max)
        resp.raise_for_status()
        data = resp.json()
        if data.get("stat") != "ok":
            raise RuntimeError(
                f"Flickr API error for method={method}: {data.get('message', 'unknown error')}"
            )
        return data

    def search_photos(
        self,
        text: Optional[str],
        tags: Optional[str],
        licenses: Optional[str],
        sort: str,
        per_page: int,
        page: int,
        min_upload_date: Optional[str] = None,
        max_upload_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        extras = ",".join(
            [
                "license",
                "date_upload",
                "date_taken",
                "owner_name",
                "icon_server",
                "original_format",
                "last_update",
                "geo",
                "tags",
                "machine_tags",
                "o_dims",
                "views",
                "media",
                "path_alias",
                "url_sq",
                "url_t",
                "url_s",
                "url_q",
                "url_m",
                "url_n",
                "url_z",
                "url_c",
                "url_l",
                "url_o",
                "description",
            ]
        )

        params: Dict[str, Any] = {
            "text": text,
            "tags": tags,
            "tag_mode": "all",
            "license": licenses,
            "sort": sort,
            "per_page": per_page,
            "page": page,
            "content_type": 1,        # photos only
            "media": "photos",
            "safe_search": 1,
            "privacy_filter": 1,      # public
            "has_geo": 0,             # allow all
            "extras": extras,
        }

        if min_upload_date:
            params["min_upload_date"] = min_upload_date
        if max_upload_date:
            params["max_upload_date"] = max_upload_date

        params = {k: v for k, v in params.items() if v not in (None, "")}
        return self.call("flickr.photos.search", **params)

    def get_photo_info(self, photo_id: str, secret: Optional[str] = None) -> Dict[str, Any]:
        params: Dict[str, Any] = {"photo_id": photo_id}
        if secret:
            params["secret"] = secret
        return self.call("flickr.photos.getInfo", **params)

    def get_people_info(self, user_id: str) -> Dict[str, Any]:
        return self.call("flickr.people.getInfo", user_id=user_id)

    def get_licenses(self) -> Dict[str, str]:
        data = self.call("flickr.photos.licenses.getInfo")
        licenses = {}
        for item in data.get("licenses", {}).get("license", []):
            licenses[str(item.get("id"))] = item.get("name", "")
        return licenses


# -----------------------------
# SMP alignment helpers
# -----------------------------
SMP_ONLY_FIELDS = {
    "Category": "SMP-specific annotation. Flickr API does not provide official SMP category labels.",
    "Subcategory": "SMP-specific annotation. Flickr API does not provide official SMP subcategory labels.",
    "Concept": "SMP-specific annotation. Flickr API does not provide official SMP concept labels.",
    "user_description_vector": "Potentially preprocessed/embedded field in SMP; not directly available from Flickr API.",
    "location_description_vector": "Potentially preprocessed/embedded field in SMP; not directly available from Flickr API.",
}

FIELD_PROVENANCE = {
    # Text
    "Uid": "from Flickr API",
    "Pid": "from Flickr API",
    "Title": "from Flickr API",
    "Tile": "kept for compatibility; populated from Title because SMP page appears to use 'Tile' in one schema block",
    "Mediatype": "from Flickr API",
    "Alltags": "from Flickr API",
    # Time / geo
    "Postdate": "from Flickr API",
    "Latitude": "from Flickr API",
    "Longitude": "from Flickr API",
    "Geoaccuracy": "from Flickr API",
    # User
    "photo_firstdate": "from Flickr API if available",
    "photo_count": "from Flickr API if available",
    "ispro": "from Flickr API if available",
    "canbuypro": "not directly exposed in standard public people.getInfo response; preserved as null if unavailable",
    "timezone_offset": "from Flickr API if available",
    "photo_firstdatetaken": "not directly exposed in standard Flickr API; preserved as null",
    "timezone_id": "from Flickr API if available",
    "user_description": "from Flickr API if available",
    "location_description": "from Flickr API if available",
    "user_description_vector": SMP_ONLY_FIELDS["user_description_vector"],
    "location_description_vector": SMP_ONLY_FIELDS["location_description_vector"],
    # Additional
    "Pathalias": "from Flickr API",
    "Ispublic": "inferred from public-only search",
    "Mediastatus": "not directly exposed in equivalent SMP form; preserved as null",
    "license_id": "from Flickr API",
    "license_name": "from Flickr API via flickr.photos.licenses.getInfo",
    # Category
    "Category": SMP_ONLY_FIELDS["Category"],
    "Subcategory": SMP_ONLY_FIELDS["Subcategory"],
    "Concept": SMP_ONLY_FIELDS["Concept"],
    # Pseudo label
    "PseudoPopularityScore": "derived from Flickr engagement fields; not official SMP label",
    "views": "from Flickr API",
    "faves": "from Flickr API via getInfo",
    "comments": "from Flickr API via getInfo",
}


def build_image_url(photo: Dict[str, Any]) -> Optional[str]:
    """
    Prefer a direct URL returned by extras; otherwise try a conservative fallback.
    """
    for key in ("url_o", "url_l", "url_c", "url_z", "url_n", "url_m", "url_q", "url_s", "url_t", "url_sq"):
        if photo.get(key):
            return str(photo[key])

    server = photo.get("server")
    photo_id = photo.get("id")
    secret = photo.get("secret")
    if server and photo_id and secret:
        # Conservative fallback for medium size "_z.jpg"
        return f"https://live.staticflickr.com/{server}/{photo_id}_{secret}_z.jpg"
    return None


def build_rel_image_path(owner: str, photo_id: str, ext: str = ".jpg") -> str:
    owner_sanitized = owner.replace("/", "_").replace("@", "_")
    return f"external_flickr/{owner_sanitized}/{photo_id}{ext}"


def maybe_download_image(
    session: requests.Session,
    url: str,
    out_path: Path,
    sleep_min: float,
    sleep_max: float,
    timeout: int,
) -> bool:
    try:
        ensure_dir(out_path.parent)
        resp = session.get(url, stream=True, timeout=timeout, headers={"User-Agent": USER_AGENT})
        random_sleep(sleep_min, sleep_max)
        resp.raise_for_status()
        with out_path.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return True
    except Exception as e:
        logging.warning("Failed to download image %s -> %s: %s", url, out_path, e)
        return False


def extract_text_record(photo: Dict[str, Any]) -> Dict[str, Any]:
    title = sanitize_text(photo.get("title"))
    tags = sanitize_text(photo.get("tags"))
    media = sanitize_text(photo.get("media") or "photo")

    return {
        "Uid": photo.get("owner"),
        "Pid": photo.get("id"),
        "Title": title,
        "Tile": title,  # compatibility placeholder
        "Mediatype": media,
        "Alltags": tags,
    }


def extract_temporal_spatial_record(photo: Dict[str, Any], info_photo: Dict[str, Any]) -> Dict[str, Any]:
    dates = info_photo.get("dates", {}) if info_photo else {}
    location = info_photo.get("location", {}) if info_photo else {}

    postdate = (
        dates.get("posted")
        or photo.get("dateupload")
        or None
    )

    latitude = location.get("latitude")
    longitude = location.get("longitude")
    accuracy = location.get("accuracy")

    if latitude in (None, "", 0, "0"):
        latitude = photo.get("latitude")
    if longitude in (None, "", 0, "0"):
        longitude = photo.get("longitude")
    if accuracy in (None, "", 0, "0"):
        accuracy = photo.get("accuracy")

    return {
        "Uid": photo.get("owner"),
        "Pid": photo.get("id"),
        "Postdate": postdate,
        "Latitude": safe_float(latitude),
        "Longitude": safe_float(longitude),
        "Geoaccuracy": safe_int(accuracy),
    }


def extract_user_record(photo: Dict[str, Any], people_info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    user_root = {}
    person = {}
    if people_info:
        person = people_info.get("person", {})
        user_root = person

    photos_count = None
    firstdate = None
    photos = user_root.get("photos", {})
    if isinstance(photos, dict):
        photos_count = safe_int((photos.get("count") or {}).get("_content") if isinstance(photos.get("count"), dict) else photos.get("count"))
        firstdate = (photos.get("firstdate") or {}).get("_content") if isinstance(photos.get("firstdate"), dict) else photos.get("firstdate")

    timezone = user_root.get("timezone", {})
    description = user_root.get("description", {})
    location = user_root.get("location", {})
    ispro = user_root.get("ispro")

    return {
        "Uid": photo.get("owner"),
        "photo_firstdate": firstdate,
        "photo_count": photos_count,
        "ispro": safe_int(ispro),
        "canbuypro": None,  # unavailable in standard public response
        "timezone_offset": safe_int(timezone.get("offset")) if isinstance(timezone, dict) else None,
        "photo_firstdatetaken": None,  # unavailable directly
        "timezone_id": timezone.get("label") if isinstance(timezone, dict) else None,
        "user_description": sanitize_text(description.get("_content") if isinstance(description, dict) else description),
        "location_description": sanitize_text(location.get("_content") if isinstance(location, dict) else location),
        "user_description_vector": None,
        "location_description_vector": None,
    }


def extract_additional_record(photo: Dict[str, Any], info_photo: Dict[str, Any], license_map: Dict[str, str]) -> Dict[str, Any]:
    license_id = str(photo.get("license")) if photo.get("license") is not None else None
    visibility = info_photo.get("visibility", {}) if info_photo else {}

    ispublic = visibility.get("ispublic")
    if ispublic is None:
        ispublic = 1  # inferred from public search

    return {
        "Uid": photo.get("owner"),
        "Pid": photo.get("id"),
        "Pathalias": photo.get("pathalias"),
        "Ispublic": safe_int(ispublic, default=1),
        "Mediastatus": None,
        "license_id": license_id,
        "license_name": license_map.get(license_id, None) if license_id is not None else None,
    }


def extract_category_record(photo: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "Uid": photo.get("owner"),
        "Pid": photo.get("id"),
        "Category": None,
        "Subcategory": None,
        "Concept": None,
        "_note": "These are preserved for SMP compatibility but are SMP-specific annotations and not directly available from Flickr API.",
    }




def extract_pseudo_label_record(photo: Dict[str, Any], info_photo: Dict[str, Any]) -> Dict[str, Any]:
    views = safe_int(photo.get("views"))

    dates = info_photo.get("dates", {}) if info_photo else {}
    postdate_raw = dates.get("posted") or photo.get("dateupload")

    post_age_days = None
    if postdate_raw is not None:
        try:
            posted_ts = int(postdate_raw)
            now_ts = int(datetime.now(timezone.utc).timestamp())
            post_age_days = max(0, (now_ts - posted_ts) // 86400)
        except (TypeError, ValueError):
            post_age_days = None

    pseudo_view = math.log1p(views) if views is not None and views >= 0 else None

    return {
        "Pid": photo.get("id"),
        "PseudoLogViewScore": pseudo_view,
        "views": views,
        "post_age_days": post_age_days,
        "_note": "PseudoLogViewScore is derived from log1p(views) and is NOT the official SMP label.",
    }

# -----------------------------
# Main crawl/export
# -----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crawl licensed Flickr data and export SMP-like files.")
    parser.add_argument("--api_key", type=str, default=os.getenv("FLICKR_API_KEY"), help="Flickr API key.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory.")
    parser.add_argument("--download_images", action="store_true", help="Download images locally.")
    parser.add_argument("--max_items", type=int, default=100, help="Maximum number of photos to collect.")
    parser.add_argument("--per_page", type=int, default=100, help="Flickr search page size.")
    parser.add_argument("--text", type=str, default=None, help="Search text.")
    parser.add_argument("--tags", type=str, default=None, help="Comma-separated tags.")
    parser.add_argument(
        "--licenses",
        type=str,
        default=None,
        help="Comma-separated Flickr license IDs to keep. Example: 4,5,7,8,9,10",
    )
    parser.add_argument(
        "--sort",
        type=str,
        default="relevance",
        choices=[
            "date-posted-asc",
            "date-posted-desc",
            "date-taken-asc",
            "date-taken-desc",
            "interestingness-desc",
            "interestingness-asc",
            "relevance",
        ],
        help="Sort order for flickr.photos.search.",
    )
    parser.add_argument("--min_upload_date", type=str, default=None, help="Minimum upload date (unix ts or MySQL datetime).")
    parser.add_argument("--max_upload_date", type=str, default=None, help="Maximum upload date (unix ts or MySQL datetime).")
    parser.add_argument("--sleep_min", type=float, default=0.8, help="Minimum random sleep seconds.")
    parser.add_argument("--sleep_max", type=float, default=2.0, help="Maximum random sleep seconds.")
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout seconds.")
    parser.add_argument("--max_owner_cache", type=int, default=5000, help="Maximum cached user profiles in memory.")
    parser.add_argument("--flush_every", type=int, default=100, help="Flush file handles every N newly written records.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing JSONL/TXT outputs and skip already written Pids.")
    parser.add_argument("--skip_duplicate_owners", action="store_true", help="Optional: skip duplicate owners after first kept photo.")
    parser.add_argument("--manifest_json", action="store_true", help="Also emit crawl_manifest.json array at the end. Default keeps only JSONL for scale.")
    parser.add_argument("--write_pretty_json", action="store_true", help="Also emit array-style .json files at the end. Not recommended for very large crawls.")
    parser.add_argument("--dedupe_on_image_path", action="store_true", help="Also skip items with duplicate relative image paths.")
    parser.add_argument("--max_duplicate_log", type=int, default=20, help="How many duplicate skip messages to log before silencing repetitive logs.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    if not args.api_key:
        logging.error("Missing Flickr API key. Pass --api_key or set FLICKR_API_KEY.")
        return 1

    if args.sleep_min < 0 or args.sleep_max < 0 or args.sleep_min > args.sleep_max:
        logging.error("Invalid sleep range: sleep_min must be <= sleep_max and both >= 0.")
        return 1

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    image_dir = output_dir / "images"
    if args.download_images:
        ensure_dir(image_dir)

    client = FlickrClient(
        api_key=args.api_key,
        sleep_min=args.sleep_min,
        sleep_max=args.sleep_max,
        timeout=args.timeout,
    )

    try:
        license_map = client.get_licenses()
        logging.info("Fetched %d Flickr license definitions", len(license_map))
    except Exception as e:
        logging.warning("Failed to fetch license definitions: %s", e)
        license_map = {}

    owner_cache: Dict[str, Optional[Dict[str, Any]]] = {}

    text_jsonl_path = output_dir / "extra_text.jsonl"
    temporal_jsonl_path = output_dir / "extra_temporalspatial_information.jsonl"
    user_jsonl_path = output_dir / "extra_user_data.jsonl"
    additional_jsonl_path = output_dir / "extra_additional_information.jsonl"
    category_jsonl_path = output_dir / "extra_category.jsonl"
    pseudo_jsonl_path = output_dir / "extra_pseudo_label.jsonl"
    img_txt_path = output_dir / "extra_img_filepath.txt"
    manifest_jsonl_path = output_dir / "crawl_manifest.jsonl"

    if args.resume:
        existing_pids = set()
        existing_pids |= load_existing_pids_from_jsonl(text_jsonl_path, key="Pid")
        existing_pids |= load_existing_pids_from_jsonl(temporal_jsonl_path, key="Pid")
        existing_pids |= load_existing_pids_from_jsonl(additional_jsonl_path, key="Pid")
        existing_pids |= load_existing_pids_from_jsonl(category_jsonl_path, key="Pid")
        existing_pids |= load_existing_pids_from_jsonl(pseudo_jsonl_path, key="Pid")
        logging.info("Resume mode: loaded %d existing Pids from output files", len(existing_pids))
    else:
        existing_pids = set()

    existing_rel_paths: set[str] = set()
    if args.resume and args.dedupe_on_image_path and img_txt_path.exists():
        with img_txt_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    existing_rel_paths.add(line)
        logging.info("Resume mode: loaded %d existing image paths", len(existing_rel_paths))

    seen_pids_this_run: set[str] = set()
    seen_rel_paths_this_run: set[str] = set()
    seen_owners_this_run: set[str] = set()

    collected = 0
    kept = 0
    skipped_duplicates = 0
    duplicate_log_count = 0
    page = 1

    with text_jsonl_path.open("a", encoding="utf-8") as f_text,          temporal_jsonl_path.open("a", encoding="utf-8") as f_temporal,          user_jsonl_path.open("a", encoding="utf-8") as f_user,          additional_jsonl_path.open("a", encoding="utf-8") as f_additional,          category_jsonl_path.open("a", encoding="utf-8") as f_category,          pseudo_jsonl_path.open("a", encoding="utf-8") as f_pseudo,          img_txt_path.open("a", encoding="utf-8") as f_img,          manifest_jsonl_path.open("a", encoding="utf-8") as f_manifest:

        while kept < args.max_items:
            logging.info("Searching page %d...", page)
            try:
                search_data = client.search_photos(
                    text=args.text,
                    tags=args.tags,
                    licenses=args.licenses,
                    sort=args.sort,
                    per_page=min(args.per_page, max(1, args.max_items - kept)),
                    page=page,
                    min_upload_date=args.min_upload_date,
                    max_upload_date=args.max_upload_date,
                )
            except Exception as e:
                logging.error("Search failed on page %d: %s", page, e)
                break

            photo_block = search_data.get("photos", {})
            photos = photo_block.get("photo", [])
            pages_total = safe_int(photo_block.get("pages"), default=1) or 1

            if not photos:
                logging.info("No more photos returned.")
                break

            for photo in photos:
                if kept >= args.max_items:
                    break

                collected += 1
                pid = str(photo.get("id"))
                uid = str(photo.get("owner"))
                rel_preview_path = build_rel_image_path(uid, pid, ext=".jpg")

                duplicate_reason = None
                if pid in existing_pids:
                    duplicate_reason = "already exists in previous outputs"
                elif pid in seen_pids_this_run:
                    duplicate_reason = "duplicate pid within current run"
                elif args.skip_duplicate_owners and uid in seen_owners_this_run:
                    duplicate_reason = "duplicate owner within current run"
                elif args.dedupe_on_image_path and (rel_preview_path in existing_rel_paths or rel_preview_path in seen_rel_paths_this_run):
                    duplicate_reason = "duplicate relative image path"

                if duplicate_reason:
                    skipped_duplicates += 1
                    if duplicate_log_count < args.max_duplicate_log:
                        logging.info("Skipping duplicate photo_id=%s owner=%s (%s)", pid, uid, duplicate_reason)
                        duplicate_log_count += 1
                        if duplicate_log_count == args.max_duplicate_log:
                            logging.info("Further duplicate skip logs will be suppressed.")
                    continue

                secret = photo.get("secret")
                logging.info("Keeping item %d/%d | photo_id=%s | owner=%s", kept + 1, args.max_items, pid, uid)

                info_photo: Dict[str, Any] = {}
                try:
                    info_data = client.get_photo_info(pid, secret=secret)
                    info_photo = info_data.get("photo", {})
                except Exception as e:
                    logging.warning("getInfo failed for %s: %s", pid, e)

                if uid not in owner_cache:
                    if len(owner_cache) >= args.max_owner_cache:
                        evict_key = next(iter(owner_cache.keys()))
                        owner_cache.pop(evict_key, None)
                    try:
                        people_data = client.get_people_info(uid)
                        owner_cache[uid] = people_data
                    except Exception as e:
                        logging.warning("people.getInfo failed for %s: %s", uid, e)
                        owner_cache[uid] = None

                people_info = owner_cache.get(uid)

                text_row = extract_text_record(photo)
                temporal_row = extract_temporal_spatial_record(photo, info_photo)
                user_row = extract_user_record(photo, people_info)
                additional_row = extract_additional_record(photo, info_photo, license_map)
                category_row = extract_category_record(photo)
                pseudo_label_row = extract_pseudo_label_record(photo, info_photo)

                image_url = build_image_url(photo)
                ext = ".jpg"
                if image_url:
                    lower_url = image_url.lower()
                    if lower_url.endswith(".png"):
                        ext = ".png"
                    elif lower_url.endswith(".webp"):
                        ext = ".webp"
                    elif lower_url.endswith(".jpeg"):
                        ext = ".jpeg"

                rel_path = build_rel_image_path(uid, pid, ext=ext)

                if args.dedupe_on_image_path and (rel_path in existing_rel_paths or rel_path in seen_rel_paths_this_run):
                    skipped_duplicates += 1
                    if duplicate_log_count < args.max_duplicate_log:
                        logging.info("Skipping duplicate image path for photo_id=%s owner=%s", pid, uid)
                        duplicate_log_count += 1
                        if duplicate_log_count == args.max_duplicate_log:
                            logging.info("Further duplicate skip logs will be suppressed.")
                    continue

                local_image_path = image_dir / rel_path if args.download_images else None
                image_downloaded = False
                if args.download_images and image_url and local_image_path is not None:
                    image_downloaded = maybe_download_image(
                        session=client.session,
                        url=image_url,
                        out_path=local_image_path,
                        sleep_min=args.sleep_min,
                        sleep_max=args.sleep_max,
                        timeout=args.timeout,
                    )

                manifest_row = {
                    "Uid": uid,
                    "Pid": pid,
                    "image_url": image_url,
                    "relative_image_path": rel_path,
                    "downloaded": image_downloaded,
                    "license_id": str(photo.get("license")) if photo.get("license") is not None else None,
                    "search_text": args.text,
                    "search_tags": args.tags,
                }

                should_flush = (kept + 1) % max(1, args.flush_every) == 0

                append_jsonl_record(f_text, text_row, flush=should_flush)
                append_jsonl_record(f_temporal, temporal_row, flush=should_flush)
                append_jsonl_record(f_user, user_row, flush=should_flush)
                append_jsonl_record(f_additional, additional_row, flush=should_flush)
                append_jsonl_record(f_category, category_row, flush=should_flush)
                append_jsonl_record(f_pseudo, pseudo_label_row, flush=should_flush)
                append_text_line(f_img, rel_path, flush=should_flush)
                append_jsonl_record(f_manifest, manifest_row, flush=should_flush)

                existing_pids.add(pid)
                seen_pids_this_run.add(pid)
                seen_owners_this_run.add(uid)
                existing_rel_paths.add(rel_path)
                seen_rel_paths_this_run.add(rel_path)
                kept += 1

            if page >= pages_total:
                break
            page += 1

    # Export metadata/reference files
    dump_json_pretty(output_dir / "field_provenance.json", FIELD_PROVENANCE)
    dump_json_pretty(output_dir / "smp_only_fields.json", SMP_ONLY_FIELDS)

    if args.write_pretty_json:
        logging.info("Generating array-style .json files from streamed JSONL outputs...")
        mapping = [
            ("extra_text.jsonl", "extra_text.json"),
            ("extra_temporalspatial_information.jsonl", "extra_temporalspatial_information.json"),
            ("extra_user_data.jsonl", "extra_user_data.json"),
            ("extra_additional_information.jsonl", "extra_additional_information.json"),
            ("extra_category.jsonl", "extra_category.json"),
            ("extra_pseudo_label.jsonl", "extra_pseudo_label.json"),
        ]
        for src_name, dst_name in mapping:
            src = output_dir / src_name
            rows = []
            if src.exists():
                with src.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            rows.append(json.loads(line))
            dump_json_pretty(output_dir / dst_name, rows)

    if args.manifest_json:
        rows = []
        if manifest_jsonl_path.exists():
            with manifest_jsonl_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
        dump_json_pretty(output_dir / "crawl_manifest.json", rows)

    # Human-readable README
    readme_lines = [
        "# External Flickr SMP-like Export",
        "",
        "This folder was generated by `crawl_flickr_to_smp.py`.",
        "",
        "## Streaming output files",
        "- `extra_img_filepath.txt`: image path list aligned by row",
        "- `extra_text.jsonl`: SMP-like text records (streaming append-friendly)",
        "- `extra_temporalspatial_information.jsonl`: time and geo records",
        "- `extra_user_data.jsonl`: user/profile records",
        "- `extra_additional_information.jsonl`: extra metadata records",
        "- `extra_category.jsonl`: category placeholders kept for SMP compatibility",
        "- `extra_pseudo_label.jsonl`: derived pseudo popularity labels",
        "- `crawl_manifest.jsonl`: per-sample crawl trace",
        "",
        "## Metadata files",
        "- `field_provenance.json`: explains source of each field",
        "- `smp_only_fields.json`: fields that are SMP-specific and not directly available from Flickr",
        "",
        "## Duplicate handling",
        "- Resume mode can skip Pids already written in prior outputs",
        "- Current run also skips duplicate Pids",
        "- Optional flags can skip duplicate owners or duplicate image paths",
        "",
        "## Official-vs-derived distinction",
        "- Official SMP target label is not available from Flickr API.",
        "- `PseudoPopularityScore` is derived and should be treated as an auxiliary/pretraining signal.",
    ]
    write_txt_lines(output_dir / "README_EXTERNAL_SMP.txt", readme_lines)

    logging.info("Done. Seen %d search results, kept %d items, skipped %d duplicates.", collected, kept, skipped_duplicates)
    logging.info("Output directory: %s", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
