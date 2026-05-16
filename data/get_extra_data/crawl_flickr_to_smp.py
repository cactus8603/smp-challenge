#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
crawl_flickr_to_smp.py

Crawl Flickr photos by text query and output files in the exact SMP-Image
dataset format consumed by build_dataset_v1.py.

Output files (all in --output_dir):
  extra_text.jsonl                       — Uid, Pid, Title, Mediatype, Alltags
  extra_temporalspatial_information.jsonl — Uid, Pid, Postdate, Latitude, Longitude, Geoaccuracy
  extra_user_data.jsonl                  — per-user profile (one row per unique Uid)
  extra_additional_information.jsonl     — Uid, Pid, Pathalias, Ispublic, Mediastatus
  extra_category.jsonl                   — Uid, Pid, Category, Subcategory, Concept
  extra_pseudo_label.jsonl               — Uid, Pid, PseudoLogViewScore, is_mature
  extra_img_filepath.txt                 — extra/Uid/Pid.jpg  (one path per line)
  crawl_manifest.jsonl                   — per-photo crawl metadata
  images/Uid/Pid.jpg                     — downloaded images (optional)

Usage:
  python3 crawl_flickr_to_smp.py \\
      --output_dir /local/smp/extra_data/extra_data_Travel \\
      --text travel \\
      --max_items 25000 \\
      --licenses 4,5,7,8,9,10 \\
      --download_images \\
      --resume \\
      --dedupe_on_image_path

nohup ./run_smp_category_crawl_v3.sh > /local/smp/logs/main_v3.log 2>&1 &
echo $!

Required env:
  FLICKR_API_KEY   — your Flickr API key
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import time
import calendar
import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
FLICKR_REST = "https://api.flickr.com/services/rest"
USER_AGENT  = "smp-flickr-crawler/2.0"
IMG_EXTS    = {".jpg", ".jpeg", ".png", ".webp"}

# SMP Category taxonomy (Category → {Subcategory: [Concept, ...]})
# Used to assign category/subcategory/concept from Flickr tags when the
# API doesn't return them directly.
SMP_CATEGORY_MAP: Dict[str, Dict[str, List[str]]] = {
    "Travel&Active&Sports": {
        "Tourism":   ["landmark", "cityscape", "monument", "tourism", "travel"],
        "Outdoors":  ["hiking", "camping", "mountain", "nature", "forest"],
        "Sports":    ["soccer", "football", "basketball", "tennis", "running"],
        "Adventure": ["surfing", "skiing", "climbing", "cycling", "kayaking"],
    },
    "Holiday&Celebrations": {
        "Festival":    ["christmas", "halloween", "new year", "carnival", "parade"],
        "Celebration": ["birthday", "wedding", "anniversary", "graduation", "party"],
    },
    "Animal": {
        "Pet":      ["cat", "dog", "rabbit", "hamster", "bird"],
        "Wildlife": ["lion", "tiger", "elephant", "bear", "deer"],
        "Marine":   ["fish", "dolphin", "whale", "shark", "jellyfish"],
    },
    "Entertainment": {
        "Music":    ["concert", "band", "guitar", "piano", "drum"],
        "Theater":  ["performance", "stage", "dance", "ballet", "opera"],
        "Festival": ["film", "movie", "cinema", "comedy", "show"],
    },
    "Fashion": {
        "Clothing":   ["dress", "suit", "jacket", "coat", "shirt"],
        "Accessory":  ["bag", "shoes", "hat", "sunglasses", "jewelry"],
        "Runway":     ["model", "fashion week", "designer", "collection", "lookbook"],
    },
    "Whether&Season": {
        "Winter": ["snow", "ice", "frost", "blizzard", "frozen"],
        "Summer": ["beach", "sun", "heat", "summer", "sunshine"],
        "Rain":   ["rain", "umbrella", "storm", "puddle", "wet"],
        "Autumn": ["autumn", "fall", "leaves", "maple", "harvest"],
        "Spring": ["spring", "flower", "blossom", "bloom", "garden"],
    },
    "Social&People": {
        "Portrait": ["portrait", "face", "smile", "selfie", "people"],
        "Family":   ["family", "children", "baby", "mother", "father"],
        "Friends":  ["friends", "group", "together", "social", "gathering"],
    },
    "Urban": {
        "Architecture": ["building", "skyscraper", "bridge", "church", "tower"],
        "Street":       ["street", "road", "city", "urban", "traffic"],
        "Night":        ["night", "neon", "lights", "city lights", "nightlife"],
    },
    "Food": {
        "Dish":     ["meal", "plate", "dinner", "lunch", "breakfast"],
        "Drink":    ["coffee", "tea", "wine", "beer", "juice"],
        "Dessert":  ["cake", "ice cream", "chocolate", "pastry", "cookie"],
        "Cooking":  ["kitchen", "chef", "cooking", "recipe", "baking"],
    },
    "Electronics": {
        "Device":   ["phone", "laptop", "tablet", "camera", "headphones"],
        "Gaming":   ["game", "console", "pc", "gaming", "controller"],
        "Gadget":   ["drone", "robot", "smartwatch", "vr", "speaker"],
    },
    "Family": {
        "Baby":     ["baby", "newborn", "infant", "toddler", "nursery"],
        "Home":     ["home", "house", "interior", "room", "living"],
        "Reunion":  ["reunion", "together", "holiday", "family", "gathering"],
    },
}

# query keyword → SMP Category  (used when --seed_category is not provided)
QUERY_TO_CATEGORY: Dict[str, str] = {
    "travel":      "Travel&Active&Sports",
    "sports":      "Travel&Active&Sports",
    "hiking":      "Travel&Active&Sports",
    "holiday":     "Holiday&Celebrations",
    "christmas":   "Holiday&Celebrations",
    "festival":    "Holiday&Celebrations",
    "animal":      "Animal",
    "cat":         "Animal",
    "dog":         "Animal",
    "concert":     "Entertainment",
    "music":       "Entertainment",
    "fashion":     "Fashion",
    "style":       "Fashion",
    "winter":      "Whether&Season",
    "snow":        "Whether&Season",
    "rain":        "Whether&Season",
    "people":      "Social&People",
    "portrait":    "Social&People",
    "city":        "Urban",
    "urban":       "Urban",
    "architecture":"Urban",
    "food":        "Food",
    "meal":        "Food",
    "electronics": "Electronics",
    "tech":        "Electronics",
    "family":      "Family",
    "baby":        "Family",
}


# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


log = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def random_sleep(min_s: float, max_s: float) -> None:
    time.sleep(random.uniform(min_s, max_s))


def empty_to_none(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, str) and v.strip() == "":
        return None
    return v


def safe_float(v: Any) -> Optional[float]:
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except Exception:
        return None


def safe_int(v: Any) -> Optional[int]:
    f = safe_float(v)
    return None if f is None else int(f)


def infer_category_from_tags(
    tags: List[str],
    seed_category: Optional[str],
) -> Tuple[str, str, str]:
    """Return (Category, Subcategory, Concept) from tag list."""
    if seed_category and seed_category in SMP_CATEGORY_MAP:
        category = seed_category
    else:
        category = "Travel&Active&Sports"   # default

    tags_lower = {t.lower() for t in tags}
    best_sub = next(iter(SMP_CATEGORY_MAP[category]))
    best_concept = ""
    best_score = -1

    for sub, concepts in SMP_CATEGORY_MAP[category].items():
        for concept in concepts:
            if concept.lower() in tags_lower:
                score = len(concept)
                if score > best_score:
                    best_score = score
                    best_sub = sub
                    best_concept = concept

    if not best_concept and tags:
        best_concept = tags[0]

    return category, best_sub, best_concept


def pseudo_log_view_score(views: Optional[int], favorites: Optional[int]) -> Optional[float]:
    """
    Approximate SMP label (log-views) from Flickr view count.
    SMP label = log(views+1).  We replicate that exactly when views is known.
    Favorites act as a small smoothing signal when views are 0.
    """
    v = views if views is not None else 0
    f = favorites if favorites is not None else 0
    if v == 0 and f == 0:
        return None
    score = math.log1p(v + 0.1 * f)
    return round(score, 6)


# ──────────────────────────────────────────────
# JSONL helpers
# ──────────────────────────────────────────────
def append_jsonl(path: Path, record: Dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_txt(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_seen_pids(path: Path) -> Set[str]:
    """Load already-crawled Pids from extra_text.jsonl for resume support."""
    seen: Set[str] = set()
    if not path.exists():
        return seen
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                pid = obj.get("Pid")
                if pid is not None:
                    seen.add(str(pid))
            except json.JSONDecodeError:
                pass
    return seen


def load_seen_image_paths(path: Path) -> Set[str]:
    seen: Set[str] = set()
    if not path.exists():
        return seen
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                seen.add(line)
    return seen


def count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    c = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                c += 1
    return c


# ──────────────────────────────────────────────
# Flickr API client
# ──────────────────────────────────────────────
class FlickrClient:
    def __init__(self, api_key: str, timeout: int = 20) -> None:
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    def _get(self, method: str, extra: Dict) -> Dict:
        params = {
            "method":       method,
            "api_key":      self.api_key,
            "format":       "json",
            "nojsoncallback": 1,
            **extra,
        }
        resp = self.session.get(FLICKR_REST, params=params, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        if data.get("stat") != "ok":
            raise RuntimeError(f"Flickr error: {data.get('message')} (code={data.get('code')})")
        return data

    def search(
        self,
        text: str,
        page: int,
        per_page: int,
        licenses: str,
        sort: str,
        user_id: Optional[str] = None,
        min_upload_date: Optional[int] = None,
        max_upload_date: Optional[int] = None,
        extras: str = (
            "url_sq,url_m,url_l,url_o,"
            "date_upload,date_taken,"
            "geo,tags,machine_tags,"
            "views,count_faves,count_comments,"
            "description,media,license,"
            "original_format,owner_name,path_alias,"
            "isfavorite,ispublic,isfriend,isfamily"
        ),
    ) -> Dict:
        params: Dict[str, Any] = {
            "text":     text,
            "page":     page,
            "per_page": per_page,
            "license":  licenses,
            "sort":     sort,
            "extras":   extras,
            "safe_search": 1,
            "content_type": 1,  # photos only
        }
        if user_id:
            params["user_id"] = user_id
        if min_upload_date is not None:
            params["min_upload_date"] = min_upload_date
        if max_upload_date is not None:
            params["max_upload_date"] = max_upload_date
        return self._get("flickr.photos.search", params)

    def get_info(self, photo_id: str) -> Dict:
        return self._get("flickr.photos.getInfo", {"photo_id": photo_id})

    def get_people_info(self, user_id: str) -> Optional[Dict]:
        try:
            data = self._get("flickr.people.getInfo", {"user_id": user_id})
            return data.get("person", {})
        except RuntimeError as e:
            if "User not found" in str(e) or "code=1" in str(e):
                return None
            raise

    def get_public_photos(
        self,
        user_id: str,
        page: int = 1,
        per_page: int = 500,
        extras: str = (
            "url_sq,url_m,url_l,url_o,"
            "date_upload,date_taken,"
            "geo,tags,machine_tags,"
            "views,count_faves,count_comments,"
            "description,media,license,"
            "original_format,owner_name,path_alias,"
            "ispublic"
        ),
    ) -> Dict:
        """flickr.people.getPublicPhotos — all public photos of a specific user."""
        return self._get("flickr.people.getPublicPhotos", {
            "user_id":  user_id,
            "page":     page,
            "per_page": per_page,
            "extras":   extras,
        })

    def enrich_photo_with_getinfo(self, photo: Dict) -> Dict:
        """
        Call flickr.photos.getInfo and merge extra fields into photo dict.
        Adds: _location (city/state/country), count_comments (from getInfo),
              notes_count, has_people, taken_granularity.
        Returns enriched photo dict (modifies in place and returns).
        """
        pid = str(photo.get("id", ""))
        if not pid:
            return photo
        try:
            info = self._get("flickr.photos.getInfo", {"photo_id": pid})
            p = info.get("photo", {})

            # location — city / state / country
            loc = p.get("location", {})
            if isinstance(loc, dict):
                def _loc(key: str) -> Optional[str]:
                    v = loc.get(key, {})
                    if isinstance(v, dict):
                        return empty_to_none(v.get("_content"))
                    return empty_to_none(v)
                photo["_location"] = {
                    "city":    _loc("locality") or _loc("city"),
                    "state":   _loc("region")   or _loc("state"),
                    "country": _loc("country"),
                }
            else:
                photo["_location"] = {}

            # comments count (getInfo is more reliable than search extras)
            comments = p.get("comments", {})
            if isinstance(comments, dict):
                photo["count_comments"] = safe_int(comments.get("_content"))
            elif comments is not None:
                photo["count_comments"] = safe_int(comments)

            # notes
            notes = p.get("notes", {})
            if isinstance(notes, dict):
                photo["notes_count"] = len(notes.get("note", []))

            # has_people (people tag in photo)
            people = p.get("people", {})
            if isinstance(people, dict):
                photo["has_people"] = safe_int(people.get("haspeople", 0))

            # date taken granularity (0=exact, 4=year only, etc.)
            dates = p.get("dates", {})
            if isinstance(dates, dict):
                photo["taken_granularity"] = safe_int(dates.get("takengranularity", 0))

        except Exception as e:
            log.debug("enrich_photo_with_getinfo failed pid=%s: %s", pid, e)
        return photo


# ──────────────────────────────────────────────
# Record builders  →  SMP format
# ──────────────────────────────────────────────
def _extract_tags(photo: Dict) -> List[str]:
    """Return clean tag list from photo dict (handles both str and dict forms)."""
    raw = photo.get("tags", "")
    if isinstance(raw, dict):
        # getInfo returns {"tag": [...]} structure
        items = raw.get("tag", [])
        return [t.get("raw", t.get("_content", "")) for t in items if isinstance(t, dict)]
    if isinstance(raw, str):
        return [t.strip().strip('"') for t in raw.split() if t.strip()]
    return []


def _extract_machine_tags(photo: Dict) -> List[str]:
    raw = photo.get("machine_tags", "")
    if isinstance(raw, str):
        return [t.strip() for t in raw.split() if t.strip()]
    return []


def build_text_record(photo: Dict, uid: str) -> Dict:
    tags_list = _extract_tags(photo)
    # SMP format: "tag1" "tag2" ...
    alltags = " ".join(f'"{t}"' for t in tags_list) if tags_list else None
    desc_raw = photo.get("description", "")
    description = desc_raw.get("_content", "") if isinstance(desc_raw, dict) else str(desc_raw or "")
    return {
        "Uid":         uid,
        "Pid":         str(photo["id"]),
        "Title":       empty_to_none(photo.get("title")),
        "Mediatype":   photo.get("media", "photo"),
        "Alltags":     alltags,
        # extra text fields (stored for downstream feature engineering)
        "description": empty_to_none(description),
        "machine_tags": " ".join(_extract_machine_tags(photo)) or None,
        "tag_count":   len(tags_list),
    }


def build_temporal_record(photo: Dict, uid: str) -> Dict:
    lat = safe_float(photo.get("latitude"))
    lon = safe_float(photo.get("longitude"))
    acc = safe_int(photo.get("accuracy"))
    if lat == 0.0 and lon == 0.0:
        lat = lon = None
    postdate    = safe_int(photo.get("dateupload"))
    date_taken  = empty_to_none(photo.get("datetaken"))   # "YYYY-MM-DD HH:MM:SS"
    return {
        "Uid":         uid,
        "Pid":         str(photo["id"]),
        "Postdate":    postdate,
        "Latitude":    str(lat) if lat is not None else None,
        "Longitude":   str(lon) if lon is not None else None,
        "Geoaccuracy": str(acc) if acc is not None else "16",
        # extra temporal (for cyclical encoding, taken vs posted gap, etc.)
        "date_taken":  date_taken,
    }


def build_additional_record(photo: Dict, uid: str) -> Dict:
    # City/State/Country come from getInfo, stored in photo if pre-fetched
    location = photo.get("_location", {}) or {}
    return {
        "Uid":           uid,
        "Pid":           str(photo["id"]),
        "Pathalias":     empty_to_none(photo.get("pathalias")) or empty_to_none(photo.get("ownername")),
        "Ispublic":      str(photo.get("ispublic", 1)),
        "Mediastatus":   "ready",
        # extra fields (GitHub repo uses City/State/Country)
        "license_id":    safe_int(photo.get("license")),
        "original_format": empty_to_none(photo.get("originalformat")),
        "city":          empty_to_none(location.get("city")),
        "state":         empty_to_none(location.get("state")),
        "country":       empty_to_none(location.get("country")),
    }


def build_category_record(
    photo: Dict,
    uid: str,
    seed_category: Optional[str],
) -> Dict:
    tags_list = _extract_tags(photo)
    category, subcategory, concept = infer_category_from_tags(tags_list, seed_category)
    return {
        "Uid":         uid,
        "Pid":         str(photo["id"]),
        "Category":    category,
        "Subcategory": subcategory,
        "Concept":     concept,
    }


def build_pseudo_label_record(photo: Dict, uid: str) -> Dict:
    views    = safe_int(photo.get("views"))
    faves    = safe_int(photo.get("count_faves"))
    comments = safe_int(photo.get("count_comments"))
    is_mature = int(bool(photo.get("isfamily", 0)) or bool(photo.get("isfriend", 0)))
    score = pseudo_log_view_score(views, faves)
    return {
        "Uid":                uid,
        "Pid":                str(photo["id"]),
        "PseudoLogViewScore": score,
        "is_mature":          bool(is_mature),
        # raw engagement signals (all used in feature engineering)
        "raw_views":          views,
        "raw_faves":          faves,
        "raw_comments":       comments,
        # engagement ratios (pre-compute here to avoid None division later)
        "fave_per_view":      safe_float(faves) / (views + 1) if views is not None and faves is not None else None,
        "comment_per_view":   safe_float(comments) / (views + 1) if views is not None and comments is not None else None,
    }


def _get_count(d: Any, key: str) -> Optional[int]:
    """Safely extract count from Flickr person dict (handles int and {'_content': N} forms)."""
    if not isinstance(d, dict):
        return None
    val = d.get(key)
    if val is None:
        return None
    if isinstance(val, dict):
        val = val.get("_content")
    return safe_int(val)


def build_user_record(person: Dict, uid: str) -> Dict:
    """
    Build user record aligned with:
    - SMP official User Profile fields
    - GitHub repo extra fields (totalImages, totalGeotagged, totalInGroup, totalTags, totalFaves, follower, following, totalViews)
    - build_dataset_v1.py standardize_user_table() alias map
    """
    photos   = person.get("photos", {}) if isinstance(person.get("photos"), dict) else {}
    counts   = person.get("counts",  {}) if isinstance(person.get("counts"),  dict) else {}
    tz       = person.get("timezone", {}) if isinstance(person.get("timezone"), dict) else {}

    # ── SMP official ──────────────────────────────────
    photo_firstdate      = empty_to_none(photos.get("firstdate"))
    photo_firstdatetaken = empty_to_none(photos.get("firstdatetaken"))
    photo_count          = _get_count(photos, "count")
    ispro                = safe_int(person.get("ispro", 0))
    canbuypro            = safe_int(person.get("canbuypro", 0))
    tz_offset            = empty_to_none(tz.get("offset"))
    tz_id                = empty_to_none(tz.get("id"))

    # user_description / location_description: SMP uses pre-computed embedding vectors.
    # We cannot reproduce them from Flickr API, so store None.
    # build_dataset_v1 will handle None as zero-vector.
    user_description     = None
    location_description = None

    # ── GitHub repo extra fields ──────────────────────
    # flickr.people.getInfo returns these under photos.* and counts.*
    total_views     = _get_count(photos, "views")
    total_favorites = None   # not directly in getInfo; Flickr doesn't expose total faves per user
    follower_count  = _get_count(counts, "followers")
    following_count = _get_count(counts, "contacts")   # 'contacts' = following in Flickr API

    # totalImages = same as photo_count
    # totalGeotagged — available as photos.geo
    total_geotagged = _get_count(photos, "geo") if "geo" in photos else None

    # Flickr doesn't expose totalInGroup, totalTags, totalFaves per-user without extra calls.
    # Store what we have; downstream can impute or drop.
    total_in_group  = None
    total_tags      = None

    # Location text (city-level, not the embedding vector)
    location_raw    = person.get("location", {})
    location_text   = location_raw.get("_content", "") if isinstance(location_raw, dict) else str(location_raw or "")

    return {
        # ── SMP official ──
        "Uid":                  uid,
        "photo_firstdate":      photo_firstdate,
        "photo_count":          photo_count,
        "ispro":                ispro,
        "canbuypro":            canbuypro,
        "timezone_offset":      tz_offset,
        "photo_firstdatetaken": photo_firstdatetaken,
        "timezone_id":          tz_id,
        "user_description":     user_description,        # None → zero-vector in build_dataset
        "location_description": location_description,    # None → zero-vector in build_dataset
        # ── GitHub repo / build_dataset alias map ──
        "follower_count":       follower_count,
        "following_count":      following_count,
        "total_views":          total_views,
        "total_favorites":      total_favorites,
        "total_geotagged":      total_geotagged,
        "total_in_group":       total_in_group,
        "total_tags":           total_tags,
        "mean_views":           None,   # cannot compute without all-photo stats
        "mean_favorites":       None,
        "mean_tags":            None,
        # ── extra for feature engineering ──
        "location_text":        empty_to_none(location_text),  # raw city/country string
        "is_deleted":           int(person.get("isdeleted", 0)),
    }


# ──────────────────────────────────────────────
# Image download
# ──────────────────────────────────────────────
def best_photo_url(photo: Dict) -> Optional[str]:
    for key in ("url_l", "url_m", "url_o"):
        url = photo.get(key)
        if url:
            return url
    return None


def download_image(url: str, dest: Path, session: requests.Session) -> bool:
    try:
        resp = session.get(url, stream=True, timeout=30)
        resp.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)
        return True
    except Exception as e:
        log.debug("Image download failed %s: %s", url, e)
        return False


# ──────────────────────────────────────────────
# Flush / rotate buffers
# ──────────────────────────────────────────────
def flush_buffers(
    buffers: Dict[str, List],
    paths: Dict[str, Path],
    img_path_buf: List[str],
    img_filepath_path: Path,
) -> None:
    for key, records in buffers.items():
        if not records:
            continue
        p = paths[key]
        with p.open("a", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        records.clear()

    if img_path_buf:
        with img_filepath_path.open("a", encoding="utf-8") as f:
            for line in img_path_buf:
                f.write(line + "\n")
        img_path_buf.clear()


# ──────────────────────────────────────────────
# Time-window slicing  (bypass 100-page limit)
# ──────────────────────────────────────────────
# Flickr's hard cap: 100 pages × 500 per_page = 50,000 results per query.
# But each (query + time window) is an independent result set.
# Strategy: slice [date_start, date_end] into windows; if a window still
# has too many results (> WINDOW_OVERFLOW_THRESHOLD), halve it recursively.

WINDOW_OVERFLOW_THRESHOLD = 40_000   # start splitting when total > this
WINDOW_MIN_DAYS = 1                  # never split below 1-day windows


def generate_time_windows(
    date_start: str,
    date_end:   str,
    initial_days: int = 180,
) -> List[Tuple[int, int]]:
    """
    Split [date_start, date_end] into non-overlapping (ts_min, ts_max) pairs.
    Returns list of (unix_ts_start, unix_ts_end).
    """
    start = dt.date.fromisoformat(date_start)
    end   = dt.date.fromisoformat(date_end)
    windows: List[Tuple[int, int]] = []
    cur = start
    delta = dt.timedelta(days=initial_days)
    while cur < end:
        nxt = min(cur + delta, end)
        ts_s = int(calendar.timegm(cur.timetuple()))
        ts_e = int(calendar.timegm(nxt.timetuple()))
        windows.append((ts_s, ts_e))
        cur = nxt
    return windows


def estimate_window_total(
    client: FlickrClient,
    text: str,
    licenses: str,
    ts_min: int,
    ts_max: int,
    user_id: Optional[str],
) -> int:
    """One cheap API call (per_page=1) to get total result count for a window."""
    try:
        result = client.search(
            text=text, page=1, per_page=1,
            licenses=licenses, sort="date-posted-desc",
            user_id=user_id,
            min_upload_date=ts_min, max_upload_date=ts_max,
        )
        return int(result.get("photos", {}).get("total", 0))
    except Exception:
        return 0


def split_window(ts_min: int, ts_max: int) -> List[Tuple[int, int]]:
    """Split a window in half."""
    mid = (ts_min + ts_max) // 2
    if mid <= ts_min:          # already 1-second window, can't split
        return [(ts_min, ts_max)]
    return [(ts_min, mid), (mid, ts_max)]


def adaptive_windows(
    client: FlickrClient,
    text: str,
    licenses: str,
    date_start: str,
    date_end: str,
    initial_days: int,
    user_id: Optional[str],
    sleep_min: float,
    sleep_max: float,
) -> List[Tuple[int, int]]:
    """
    Build a list of time windows sized so each has <= WINDOW_OVERFLOW_THRESHOLD results.
    Uses a queue; windows that are too large get split in half and re-checked.
    """
    raw = generate_time_windows(date_start, date_end, initial_days)
    queue = list(raw)
    final: List[Tuple[int, int]] = []
    min_secs = WINDOW_MIN_DAYS * 86400

    log.info("Adaptive windowing: %d initial windows (%d-day chunks)", len(raw), initial_days)

    while queue:
        ts_min, ts_max = queue.pop(0)
        span_days = (ts_max - ts_min) / 86400

        if span_days < WINDOW_MIN_DAYS:
            final.append((ts_min, ts_max))
            continue

        total = estimate_window_total(client, text, licenses, ts_min, ts_max, user_id)
        log.debug("Window %s→%s  total=%d  days=%.0f",
                  ts_min, ts_max, total, span_days)

        if total > WINDOW_OVERFLOW_THRESHOLD and (ts_max - ts_min) > min_secs:
            halves = split_window(ts_min, ts_max)
            log.debug("Splitting window (total=%d > %d)", total, WINDOW_OVERFLOW_THRESHOLD)
            queue = halves + queue   # depth-first: handle smaller slices first
        else:
            final.append((ts_min, ts_max))

        random_sleep(sleep_min * 0.3, sleep_max * 0.3)

    log.info("Adaptive windowing done: %d final windows", len(final))
    return final


# ──────────────────────────────────────────────
# Per-user photo count sampler
# ──────────────────────────────────────────────
def sample_photos_per_user(max_photos_per_user: int) -> int:
    """
    Randomly sample how many photos to fetch for a new uid.

    This is intentionally long-tail shaped, but it must also be safe when
    max_photos_per_user is small, e.g. 3 or 5. The previous version could call
    random.randint(20, max_photos_per_user), which crashes when max < 20.

    max_photos_per_user=0 means disabled (return 0).
    """
    if max_photos_per_user <= 0:
        return 0
    if max_photos_per_user == 1:
        return 1

    r = random.random()

    # Small caps: keep it sparse and never sample an invalid range.
    if max_photos_per_user <= 4:
        if r < 0.60:
            return 1
        return random.randint(2, max_photos_per_user)

    if max_photos_per_user <= 10:
        if r < 0.50:
            return 1
        elif r < 0.80:
            return random.randint(2, min(4, max_photos_per_user))
        else:
            return random.randint(5, max_photos_per_user)

    # Larger caps: approximate the SMP long-tail user distribution.
    if r < 0.50:
        return 1
    elif r < 0.78:
        return random.randint(2, 4)
    elif r < 0.95:
        return random.randint(5, min(20, max_photos_per_user))
    else:
        low = min(21, max_photos_per_user)
        return random.randint(low, max_photos_per_user)


# ──────────────────────────────────────────────
# Per-user deep crawl
# ──────────────────────────────────────────────
def crawl_user_photos(
    client: FlickrClient,
    uid: str,
    seed_category: Optional[str],
    licenses: str,
    max_per_user: int,
    fetch_photo_detail: bool,
    download_images: bool,
    images_dir: Path,
    seen_pids: Set[str],
    seen_paths: Set[str],
    buffers: Dict[str, List],
    img_path_buf: List[str],
    sleep_min: float,
    sleep_max: float,
) -> int:
    """
    Fetch all public photos of `uid` via flickr.people.getPublicPhotos.
    Appends records into shared buffers. Returns number of newly added photos.
    """
    gained = 0
    page   = 1
    allowed_licenses = {int(x) for x in licenses.split(",")}

    while gained < max_per_user:
        try:
            result = client.get_public_photos(uid, page=page, per_page=500)
        except Exception as e:
            log.debug("get_public_photos failed uid=%s page=%d: %s", uid, page, e)
            break

        photos_data = result.get("photos", {})
        photos      = photos_data.get("photo", [])
        total_pages = int(photos_data.get("pages", 1))

        if not photos:
            break

        for photo in photos:
            if gained >= max_per_user:
                break

            pid = str(photo["id"])
            if pid in seen_pids:
                continue

            license_id = safe_int(photo.get("license", 0)) or 0
            if license_id not in allowed_licenses:
                continue

            img_rel_path = f"extra/{uid}/{pid}.jpg"
            if img_rel_path in seen_paths:
                continue

            # optional per-photo enrichment
            if fetch_photo_detail:
                photo = client.enrich_photo_with_getinfo(photo)
                random_sleep(sleep_min * 0.3, sleep_max * 0.3)

            buffers["text"].append(build_text_record(photo, uid))
            buffers["temporal"].append(build_temporal_record(photo, uid))
            buffers["additional"].append(build_additional_record(photo, uid))
            buffers["category"].append(build_category_record(photo, uid, seed_category))
            buffers["pseudo"].append(build_pseudo_label_record(photo, uid))
            buffers["manifest"].append({
                "Uid":        uid,
                "Pid":        pid,
                "crawl_mode": "by_uid_deepcrawl",
                "query":      f"uid:{uid}",
                "page":       page,
                "img_path":   img_rel_path,
                "crawled_at": int(time.time()),
                "has_detail": fetch_photo_detail,
            })

            img_path_buf.append(img_rel_path)
            seen_paths.add(img_rel_path)
            seen_pids.add(pid)
            gained += 1

            if download_images:
                url = best_photo_url(photo)
                if url:
                    dest = images_dir / uid / f"{pid}.jpg"
                    if not dest.exists():
                        download_image(url, dest, client.session)

        if page >= total_pages:
            break
        page += 1
        random_sleep(sleep_min * 0.5, sleep_max * 0.5)

    return gained


# ──────────────────────────────────────────────
# Core crawl loop
# ──────────────────────────────────────────────
def crawl(
    client: FlickrClient,
    output_dir: Path,
    text: str,
    max_items: int,
    licenses: str,
    sort: str,
    per_page: int,
    max_no_new_pages: int,
    sleep_min: float,
    sleep_max: float,
    flush_every: int,
    download_images: bool,
    resume: bool,
    dedupe_on_image_path: bool,
    seed_category: Optional[str],
    user_id: Optional[str],
    crawl_mode: str,
    # time-window slicing (optional — if both None, no date filter applied)
    min_upload_date: Optional[int] = None,
    max_upload_date: Optional[int] = None,
    # per-photo getInfo enrichment (City/State/Country/comments/notes/has_people)
    fetch_photo_detail: bool = False,
    # per-user deep crawl
    max_photos_per_user: int = 0,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "text":       output_dir / "extra_text.jsonl",
        "temporal":   output_dir / "extra_temporalspatial_information.jsonl",
        "user":       output_dir / "extra_user_data.jsonl",
        "additional": output_dir / "extra_additional_information.jsonl",
        "category":   output_dir / "extra_category.jsonl",
        "pseudo":     output_dir / "extra_pseudo_label.jsonl",
        "manifest":   output_dir / "crawl_manifest.jsonl",
    }
    img_filepath_path = output_dir / "extra_img_filepath.txt"
    images_dir        = output_dir / "images"

    # Resume: load already-seen Pids
    seen_pids: Set[str] = load_seen_pids(paths["text"]) if resume else set()
    seen_paths: Set[str] = load_seen_image_paths(img_filepath_path) if (resume and dedupe_on_image_path) else set()
    already_done = len(seen_pids)
    log.info("Resume: %d already crawled", already_done)

    # Per-user cache to avoid duplicate user API calls
    seen_uids: Set[str] = set()

    # In-memory buffers
    buffers: Dict[str, List] = {k: [] for k in paths}
    img_path_buf: List[str] = []

    # Flickr caps results at page 100 × per_page regardless of total_pages.
    # When one sort order is exhausted we rotate through alternates before giving up.
    SORT_ROTATION = [
        sort,
        "interestingness-desc",
        "relevance",
        "date-taken-desc",
        "date-posted-asc",
    ]
    # De-duplicate so the user-supplied sort appears only once
    seen_sorts: list[str] = []
    for s in SORT_ROTATION:
        if s not in seen_sorts:
            seen_sorts.append(s)
    SORT_ROTATION = seen_sorts

    MAX_API_RETRIES = 5   # max consecutive network errors before aborting

    total_saved  = already_done
    no_new_pages = 0
    page         = 1
    sort_idx     = 0
    current_sort = SORT_ROTATION[sort_idx]
    api_errors   = 0

    while total_saved < max_items:
        need = max_items - total_saved
        this_per_page = min(per_page, 500, need + 20)   # slight over-fetch

        log.info("Page %d | sort=%s | saved=%d/%d", page, current_sort, total_saved, max_items)

        try:
            result = client.search(
                text=text,
                page=page,
                per_page=this_per_page,
                licenses=licenses,
                sort=current_sort,
                user_id=user_id,
                min_upload_date=min_upload_date,
                max_upload_date=max_upload_date,
            )
            api_errors = 0   # reset on success
        except Exception as e:
            api_errors += 1
            wait = min(5 * api_errors, 60)
            log.warning("Search failed page=%d (attempt %d/%d): %s — waiting %ds",
                        page, api_errors, MAX_API_RETRIES, e, wait)
            if api_errors >= MAX_API_RETRIES:
                log.error("Too many consecutive API errors — stopping crawl early.")
                break
            time.sleep(wait)
            continue

        photos_data = result.get("photos", {})
        photos      = photos_data.get("photo", [])
        total_pages = int(photos_data.get("pages", 1))

        # ── Flickr hard-caps at page 100 ──────────────────────────────────
        max_reachable_page = min(total_pages, 100)

        if not photos:
            log.info("No photos returned on page %d (sort=%s)", page, current_sort)
            # Try next sort order
            sort_idx += 1
            if sort_idx >= len(SORT_ROTATION):
                log.info("All sort orders exhausted. Stopping with %d/%d items.",
                         total_saved, max_items)
                break
            current_sort = SORT_ROTATION[sort_idx]
            page = 1
            no_new_pages = 0
            log.info("Switching sort → %s", current_sort)
            continue

        new_this_page = 0
        for photo in photos:
            pid = str(photo["id"])
            uid = photo.get("owner", "unknown")

            if pid in seen_pids:
                continue

            # Build image path (SMP format: extra/Uid/Pid.jpg)
            img_rel_path = f"extra/{uid}/{pid}.jpg"
            if dedupe_on_image_path and img_rel_path in seen_paths:
                continue

            # ── optional getInfo enrichment (City/State/Country/comments/notes/has_people) ──
            if fetch_photo_detail:
                photo = client.enrich_photo_with_getinfo(photo)
                random_sleep(sleep_min * 0.3, sleep_max * 0.3)

            # ── text ──
            buffers["text"].append(build_text_record(photo, uid))

            # ── temporal ──
            buffers["temporal"].append(build_temporal_record(photo, uid))

            # ── additional ──
            buffers["additional"].append(build_additional_record(photo, uid))

            # ── category ──
            buffers["category"].append(build_category_record(photo, uid, seed_category))

            # ── pseudo label ──
            buffers["pseudo"].append(build_pseudo_label_record(photo, uid))

            # ── manifest ──
            buffers["manifest"].append({
                "Uid":        uid,
                "Pid":        pid,
                "crawl_mode": crawl_mode,
                "query":      text,
                "page":       page,
                "img_path":   img_rel_path,
                "crawled_at": int(time.time()),
                "has_detail": fetch_photo_detail,
            })

            # ── img filepath ──
            img_path_buf.append(img_rel_path)
            seen_paths.add(img_rel_path)
            seen_pids.add(pid)

            # ── user (once per uid) ──
            if uid not in seen_uids:
                try:
                    person = client.get_people_info(uid)
                    if person:
                        buffers["user"].append(build_user_record(person, uid))
                        seen_uids.add(uid)
                    random_sleep(sleep_min * 0.5, sleep_max * 0.5)
                except Exception as e:
                    log.debug("get_people_info failed uid=%s: %s", uid, e)

                # ── deep crawl: fetch photos of this new user ──
                if max_photos_per_user > 0 and total_saved < max_items:
                    this_user_limit = sample_photos_per_user(max_photos_per_user)
                    gained = crawl_user_photos(
                        client=client,
                        uid=uid,
                        seed_category=seed_category,
                        licenses=licenses,
                        max_per_user=this_user_limit,
                        fetch_photo_detail=fetch_photo_detail,
                        download_images=download_images,
                        images_dir=images_dir,
                        seen_pids=seen_pids,
                        seen_paths=seen_paths,
                        buffers=buffers,
                        img_path_buf=img_path_buf,
                        sleep_min=sleep_min,
                        sleep_max=sleep_max,
                    )
                    total_saved += gained
                    new_this_page += gained
                    if gained > 0:
                        log.info("Deep crawl uid=%s: +%d photos (total=%d/%d)",
                                 uid, gained, total_saved, max_items)
                    if total_saved % flush_every < gained:
                        flush_buffers(buffers, paths, img_path_buf, img_filepath_path)

            # ── image download ──
            if download_images:
                url = best_photo_url(photo)
                if url:
                    dest = images_dir / uid / f"{pid}.jpg"
                    if not dest.exists():
                        ok = download_image(url, dest, client.session)
                        if ok:
                            log.debug("Downloaded %s", dest)

            total_saved += 1
            new_this_page += 1

            # flush
            if total_saved % flush_every == 0:
                flush_buffers(buffers, paths, img_path_buf, img_filepath_path)
                log.info("Flushed at %d items", total_saved)

            if total_saved >= max_items:
                break

        if new_this_page == 0:
            no_new_pages += 1
            log.info("No new items on page %d/%d (%d consecutive, sort=%s)",
                     page, max_reachable_page, no_new_pages, current_sort)
        else:
            no_new_pages = 0

        # Decide whether to advance page or rotate sort
        exhausted_pages = (page >= max_reachable_page)
        exhausted_dedup = (no_new_pages >= max_no_new_pages)

        if exhausted_pages or exhausted_dedup:
            reason = "last page reached" if exhausted_pages else f"{no_new_pages} consecutive empty pages"
            log.info("Sort '%s' exhausted (%s). Total so far: %d/%d",
                     current_sort, reason, total_saved, max_items)
            sort_idx += 1
            if sort_idx >= len(SORT_ROTATION):
                log.info("All sort orders exhausted. Stopping with %d/%d items.",
                         total_saved, max_items)
                break
            current_sort = SORT_ROTATION[sort_idx]
            page = 1
            no_new_pages = 0
            log.info("Switching sort → %s", current_sort)
        else:
            page += 1

        random_sleep(sleep_min, sleep_max)

    # Final flush
    flush_buffers(buffers, paths, img_path_buf, img_filepath_path)
    log.info("Crawl complete. Total saved: %d", total_saved)
    return total_saved


# ──────────────────────────────────────────────
# Time-sliced crawl  (high-volume entry point)
# ──────────────────────────────────────────────
def crawl_with_time_slicing(
    client: FlickrClient,
    output_dir: Path,
    text: str,
    max_items: int,
    licenses: str,
    sort: str,
    per_page: int,
    max_no_new_pages: int,
    sleep_min: float,
    sleep_max: float,
    flush_every: int,
    download_images: bool,
    resume: bool,
    dedupe_on_image_path: bool,
    seed_category: Optional[str],
    user_id: Optional[str],
    crawl_mode: str,
    date_start: str = "2004-01-01",
    date_end:   str = "2024-12-31",
    initial_window_days: int = 180,
    fetch_photo_detail: bool = False,
    max_photos_per_user: int = 0,
) -> int:
    """
    Crawl by splitting the date range into adaptive time windows,
    each yielding up to ~50,000 unique results from the Flickr API.

    With date_start=2004, date_end=2024, initial_window_days=180 you get
    ~40 initial windows. If each yields ~40k unique photos, that's ~1.6M
    potential results before deduplication — far beyond the 50k page cap.

    Progress is tracked via seen_pids so --resume works across windows.
    """
    # Build shared state (resume-aware, shared across all windows)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "text":       output_dir / "extra_text.jsonl",
        "temporal":   output_dir / "extra_temporalspatial_information.jsonl",
        "user":       output_dir / "extra_user_data.jsonl",
        "additional": output_dir / "extra_additional_information.jsonl",
        "category":   output_dir / "extra_category.jsonl",
        "pseudo":     output_dir / "extra_pseudo_label.jsonl",
        "manifest":   output_dir / "crawl_manifest.jsonl",
    }
    img_filepath_path = output_dir / "extra_img_filepath.txt"

    seen_pids  = load_seen_pids(paths["text"]) if resume else set()
    already    = len(seen_pids)
    total_saved = already
    log.info("Time-sliced crawl | already=%d / target=%d", already, max_items)

    if total_saved >= max_items:
        log.info("Already at target, nothing to do.")
        return total_saved

    # Save progress file so we can resume between windows
    progress_path = output_dir / "crawl_window_progress.json"
    completed_windows: set = set()
    if resume and progress_path.exists():
        try:
            completed_windows = set(
                tuple(w) for w in json.loads(progress_path.read_text())
            )
            log.info("Resuming: %d windows already completed", len(completed_windows))
        except Exception:
            pass

    # Build adaptive windows (cheap: only 1 API call per initial window)
    windows = adaptive_windows(
        client=client, text=text, licenses=licenses,
        date_start=date_start, date_end=date_end,
        initial_days=initial_window_days,
        user_id=user_id,
        sleep_min=sleep_min, sleep_max=sleep_max,
    )

    log.info("Starting crawl across %d time windows", len(windows))

    for i, (ts_min, ts_max) in enumerate(windows):
        if total_saved >= max_items:
            log.info("Reached target %d, stopping.", max_items)
            break

        win_key = (ts_min, ts_max)
        if win_key in completed_windows:
            log.info("[%d/%d] Window already done, skipping", i + 1, len(windows))
            continue

        window_target = max_items - total_saved
        log.info("[%d/%d] Window %s→%s | need=%d",
                 i + 1, len(windows), ts_min, ts_max, window_target)

        saved = crawl(
            client=client,
            output_dir=output_dir,
            text=text,
            max_items=total_saved + window_target,
            licenses=licenses,
            sort=sort,
            per_page=per_page,
            max_no_new_pages=max_no_new_pages,
            sleep_min=sleep_min,
            sleep_max=sleep_max,
            flush_every=flush_every,
            download_images=download_images,
            resume=True,              # always resume within shared output_dir
            dedupe_on_image_path=dedupe_on_image_path,
            seed_category=seed_category,
            user_id=user_id,
            crawl_mode=crawl_mode,
            min_upload_date=ts_min,
            max_upload_date=ts_max,
            fetch_photo_detail=fetch_photo_detail,
            max_photos_per_user=max_photos_per_user,
        )

        gained = saved - total_saved
        total_saved = saved
        log.info("[%d/%d] Window done | gained=%d | total=%d/%d",
                 i + 1, len(windows), gained, total_saved, max_items)

        # Mark window as done
        completed_windows.add(win_key)
        progress_path.write_text(
            json.dumps([list(w) for w in completed_windows]), encoding="utf-8"
        )

    log.info("Time-sliced crawl finished. Total: %d/%d", total_saved, max_items)
    return total_saved


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Crawl Flickr → SMP format.")
    p.add_argument("--output_dir",           required=True)
    p.add_argument("--text",                 required=True,  help="Flickr search query text")
    p.add_argument("--max_items",            type=int, default=10000)
    p.add_argument("--licenses",             default="4,5,7,8,9,10")
    p.add_argument("--sort",                 default="date-posted-desc")
    p.add_argument("--per_page",             type=int, default=500)
    p.add_argument("--max_no_new_pages",     type=int, default=20)
    p.add_argument("--sleep_min",            type=float, default=0.8)
    p.add_argument("--sleep_max",            type=float, default=2.0)
    p.add_argument("--flush_every",          type=int, default=200)
    p.add_argument("--download_images",      action="store_true")
    p.add_argument("--resume",               action="store_true")
    p.add_argument("--dedupe_on_image_path", action="store_true")
    p.add_argument("--seed_category",        default=None,   help="Override SMP Category label")
    p.add_argument("--user_id",              default=None,   help="Restrict search to this Flickr user")
    p.add_argument("--crawl_mode",           default="by_text")
    p.add_argument("--api_key",              default=os.getenv("FLICKR_API_KEY"))
    p.add_argument("--verbose",              action="store_true")
    # ── time-slicing ──────────────────────────────────────────────────────
    p.add_argument("--time_slice",           action="store_true",
                   help="Enable time-window slicing to bypass 50k API limit")
    p.add_argument("--date_start",           default="2004-01-01",
                   help="Earliest upload date (YYYY-MM-DD). Used with --time_slice")
    p.add_argument("--date_end",             default="2024-12-31",
                   help="Latest upload date (YYYY-MM-DD). Used with --time_slice")
    p.add_argument("--initial_window_days",  type=int, default=180,
                   help="Initial time window size in days (halved automatically if too large)")
    # ── per-photo detail ──────────────────────────────────────────────────
    p.add_argument("--fetch_photo_detail",   action="store_true",
                   help="Call getInfo per photo: City/State/Country/comments/notes/has_people. "
                        "Doubles API calls — disable for speed, enable for richer features.")
    # ── per-user deep crawl ───────────────────────────────────────────────
    p.add_argument("--max_photos_per_user",  type=int, default=30,
                   help="After finding a new uid, fetch up to N of their public photos. 0=disabled.")
    # ── multi-keyword ─────────────────────────────────────────────────────
    p.add_argument("--extra_texts",          type=str, default="",
                   help="Comma-separated extra search queries to rotate after --text.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    if not args.api_key:
        raise SystemExit("ERROR: FLICKR_API_KEY not set. Use --api_key or export FLICKR_API_KEY=...")

    # Build query list (primary + extras, deduplicated)
    queries = [args.text]
    if args.extra_texts:
        for t in args.extra_texts.split(","):
            t = t.strip()
            if t and t not in queries:
                queries.append(t)

    client = FlickrClient(api_key=args.api_key)

    for qi, query in enumerate(queries):
        seed_cat = args.seed_category or QUERY_TO_CATEGORY.get(query.lower())
        log.info("Query [%d/%d]: '%s'  seed_category=%s", qi + 1, len(queries), query, seed_cat)

        common = dict(
            client=client,
            output_dir=Path(args.output_dir),
            text=query,
            max_items=args.max_items,
            licenses=args.licenses,
            sort=args.sort,
            per_page=args.per_page,
            max_no_new_pages=args.max_no_new_pages,
            sleep_min=args.sleep_min,
            sleep_max=args.sleep_max,
            flush_every=args.flush_every,
            download_images=args.download_images,
            resume=True,
            dedupe_on_image_path=args.dedupe_on_image_path,
            seed_category=seed_cat,
            user_id=args.user_id,
            crawl_mode="by_text",
            fetch_photo_detail=args.fetch_photo_detail,
            max_photos_per_user=args.max_photos_per_user,
        )

        if args.time_slice:
            total = crawl_with_time_slicing(
                **common,
                date_start=args.date_start,
                date_end=args.date_end,
                initial_window_days=args.initial_window_days,
            )
        else:
            total = crawl(**common)

        log.info("Query '%s' done. Total in output_dir: %d", query, total)
        if total >= args.max_items:
            log.info("Reached max_items=%d, stopping.", args.max_items)
            break


if __name__ == "__main__":
    main()
