#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_dataset_v2.py

Build unified SMP-Image train/test tables from official raw files.

Key changes in v2:
- Clean title / tags / user_description / location_description before building full_text.
- Keep raw user/location descriptions, and add *_clean columns.
- Preserve city/state/country if they already exist in temporalspatial files.
- Build location_text from location_description_clean + city/state/country.
- Build full_text ONLY from clean fields.
- full_train mode does NOT precompute label-derived user aggregate features, avoiding KFold leakage.
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


LOGGER = logging.getLogger("build_dataset_v2")


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------
def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    s = str(value).strip()
    if s.lower() in {"none", "nan", "null", "<na>"}:
        return ""
    return s


def empty_to_none(value: Any) -> Any:
    s = safe_str(value)
    return s if s else None


def to_int(value: Any) -> Optional[int]:
    value = empty_to_none(value)
    if value is None:
        return None
    try:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            if pd.isna(value):
                return None
            return int(value)
        return int(float(str(value).strip()))
    except Exception:
        return None


def to_float(value: Any) -> Optional[float]:
    value = empty_to_none(value)
    if value is None:
        return None
    try:
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            if pd.isna(value):
                return None
            return float(value)
        return float(str(value).strip())
    except Exception:
        return None


def safe_div(a: Any, b: Any) -> Optional[float]:
    a = to_float(a)
    b = to_float(b)
    if a is None or b is None or b == 0:
        return None
    return a / b


def clean_description(x: Any) -> str:
    """Clean dirty Flickr/user text before it can enter full_text."""
    s = safe_str(x)
    if not s:
        return ""

    s = html.unescape(s)

    # remove full HTML tags, including href/src attributes
    s = re.sub(r"<[^>]*>", " ", s)

    # remove URLs / domains / mailto
    s = re.sub(r"\b(?:https?|ftp)://[^\s\"'<>()]+", " ", s, flags=re.I)
    s = re.sub(r"\bwww\.[^\s\"'<>()]+", " ", s, flags=re.I)
    s = re.sub(r"\bmailto:[^\s\"'<>()]+", " ", s, flags=re.I)
    s = re.sub(
        r"\b[a-zA-Z0-9.-]+\.(?:com|org|net|edu|gov|co|io|me|info|biz|jp|tw|uk|de|fr|it|es|ca|au|in|cn|hk|ph|nl|se|ru)(?:/[^\s\"'<>()]*)?",
        " ",
        s,
        flags=re.I,
    )

    # remove emails / obfuscated emails
    s = re.sub(r"\b[\w.+-]+@[\w.-]+\.\w+\b", " ", s)
    s = re.sub(
        r"\b[\w.+-]+\s*(?:\{AT\}|\[AT\]|\(AT\)| at )\s*[\w.-]+\b",
        " ",
        s,
        flags=re.I,
    )

    # remove filenames
    s = re.sub(
        r"\b\S+\.(?:jpg|jpeg|png|gif|webp|bmp|tiff|svg|html|htm|php|aspx|pdf|zip)\b",
        " ",
        s,
        flags=re.I,
    )

    # remove repeated separators: ------, =====, *****
    s = re.sub(r"[-_=*~#]{3,}", " ", s)

    # remove repeated punctuation: !!!!!, ......, ///////
    s = re.sub(r"([!?.,/\\|:;])\1{2,}", " ", s)

    # remove very long garbage tokens
    s = re.sub(r"\b\S{35,}\b", " ", s)

    # remove common HTML leftovers
    s = re.sub(
        r"\b(?:href|src|rel|nofollow|class|title|alt|img|target|blank|style|width|height)\b",
        " ",
        s,
        flags=re.I,
    )

    # remove weird brackets/symbols
    s = re.sub(r"[<>={}\[\]|\\]", " ", s)

    # normalize spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s


def truncate_words(text: Any, max_words: int) -> str:
    s = safe_str(text)
    if not s or max_words <= 0:
        return ""
    return " ".join(s.split()[:max_words])


def join_unique(parts: List[Any], sep: str = " ") -> str:
    seen = set()
    out: List[str] = []
    for p in parts:
        s = safe_str(p)
        if not s:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return sep.join(out).strip()



# -----------------------------------------------------------------------------
# Location parsing / enrichment
# -----------------------------------------------------------------------------
_COUNTRY_ALIASES = {
    "usa": "United States",
    "u.s.a": "United States",
    "u.s.a.": "United States",
    "us": "United States",
    "u.s": "United States",
    "u.s.": "United States",
    "united states of america": "United States",
    "america": "United States",
    "uk": "United Kingdom",
    "u.k": "United Kingdom",
    "u.k.": "United Kingdom",
    "gb": "United Kingdom",
    "g.b.": "United Kingdom",
    "great britain": "United Kingdom",
    "england": "United Kingdom",
    "scotland": "United Kingdom",
    "wales": "United Kingdom",
    "brasil": "Brazil",
    "deutschland": "Germany",
    "españa": "Spain",
    "espana": "Spain",
    "méxico": "Mexico",
    "mexico": "Mexico",
    "czech republic": "Czechia",
    "russian federation": "Russia",
    "republic of korea": "South Korea",
    "korea": "South Korea",
    "ru": "Russia",
    "russia": "Russia",
    "the netherlands": "Netherlands",
    "taiwan": "Taiwan",
    "taiwan, province of china": "Taiwan",
    "taiwan province of china": "Taiwan",
    "prc": "China",
    "peoples republic of china": "China",
}

_KNOWN_COUNTRY_NAMES = {
    "United States", "United Kingdom", "Canada", "Australia", "New Zealand",
    "France", "Germany", "Italy", "Spain", "Portugal", "Netherlands", "The Netherlands", "Belgium",
    "Switzerland", "Austria", "Ireland", "Sweden", "Norway", "Denmark", "Finland",
    "Poland", "Czechia", "Greece", "Turkey", "Russia", "Ukraine",
    "Japan", "South Korea", "China", "Taiwan", "Hong Kong", "Singapore",
    "Thailand", "Malaysia", "Indonesia", "Philippines", "Vietnam", "India",
    "Brazil", "Argentina", "Chile", "Mexico", "Colombia", "Peru",
    "South Africa", "Egypt", "Morocco", "Israel", "United Arab Emirates",
}

_US_STATE_NAMES = {
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado",
    "Connecticut", "Delaware", "Florida", "Georgia", "Hawaii", "Idaho",
    "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana",
    "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota",
    "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada",
    "New Hampshire", "New Jersey", "New Mexico", "New York",
    "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon",
    "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota",
    "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington",
    "West Virginia", "Wisconsin", "Wyoming", "District of Columbia",
}

_US_STATE_ABBR = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
    "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska",
    "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey",
    "NM": "New Mexico", "NY": "New York", "NC": "North Carolina",
    "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon",
    "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
    "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
    "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
    "DC": "District of Columbia",
}


def clean_location_part(value: Any) -> str:
    s = clean_description(value)
    if not s:
        return ""
    s = re.sub(r"\s+", " ", s).strip(" ,;/|")
    if s.lower() in {"none", "nan", "null", "<na>", "unknown", "unk"}:
        return ""
    return s


def is_empty_location(value: Any) -> bool:
    return clean_location_part(value) == ""


def normalize_country(value: Any) -> str:
    s = clean_location_part(value)
    if not s:
        return ""

    key = re.sub(r"[.]", "", s.lower()).strip()
    if key in _COUNTRY_ALIASES:
        return _COUNTRY_ALIASES[key]
    return s


def normalize_state(value: Any) -> str:
    s = clean_location_part(value)
    if not s:
        return ""

    upper = s.upper().replace(".", "")
    if upper in _US_STATE_ABBR:
        return _US_STATE_ABBR[upper]
    return s


def parse_location_text(value: Any) -> Tuple[str, str, str]:
    """
    Heuristically parse free-form Flickr location text into city/state/country.

    Examples:
        London, United Kingdom
            -> city=London, state="", country=United Kingdom

        Buffalo, New York, United States of America
            -> city=Buffalo, state=New York, country=United States

        France
            -> city="", state="", country=France
    """
    raw = clean_location_part(value)
    if not raw:
        return "", "", ""

    raw = raw.replace("|", ",").replace(";", ",")
    parts = [clean_location_part(p) for p in raw.split(",")]
    parts = [p for p in parts if p]

    if not parts:
        return "", "", ""

    if len(parts) == 1:
        original = clean_location_part(parts[0])
        only = normalize_country(original)
        key = re.sub(r"[.]", "", only.lower()).strip()

        if only in _KNOWN_COUNTRY_NAMES or key in _COUNTRY_ALIASES:
            return "", "", normalize_country(only)

        # Handle common space-separated forms without commas:
        #   "England UK" -> country=United Kingdom
        #   "Paris France" -> city=Paris, country=France
        tokens = original.split()
        if len(tokens) >= 2:
            last_token_country = normalize_country(tokens[-1])
            last_key = re.sub(r"[.]", "", last_token_country.lower()).strip()
            if last_token_country in _KNOWN_COUNTRY_NAMES or last_key in _COUNTRY_ALIASES:
                city_guess = clean_location_part(" ".join(tokens[:-1]))
                return city_guess, "", normalize_country(last_token_country)

        # Keep one-part unknown locations as city-ish information.
        return only, "", ""

    country = normalize_country(parts[-1])
    city = ""
    state = ""

    if len(parts) == 2:
        city = parts[0]
    else:
        city = parts[0]
        state = normalize_state(parts[-2])

    return clean_location_part(city), normalize_state(state), normalize_country(country)


def first_non_empty(*values: Any) -> str:
    for value in values:
        s = clean_location_part(value)
        if s:
            return s
    return ""


def enrich_location_from_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill city/state/country from location_description_clean/location_description/location_text
    while preserving existing non-empty city/state/country from official files.

    This should run BEFORE add_extra_features(), because add_extra_features()
    builds location_text and full_text from city/state/country.
    """
    out = df.copy()

    for c in [
        "city",
        "state",
        "country",
        "location_description",
        "location_description_clean",
        "location_text",
    ]:
        if c not in out.columns:
            out[c] = None

    # Prefer cleaned user location text; fall back to raw location_description
    # and then any pre-existing location_text.
    source_text = [
        first_non_empty(loc_clean, loc_raw, loc_text)
        for loc_clean, loc_raw, loc_text in zip(
            out["location_description_clean"],
            out["location_description"],
            out["location_text"],
        )
    ]

    parsed = [parse_location_text(x) for x in source_text]
    parsed_city = pd.Series([x[0] for x in parsed], index=out.index)
    parsed_state = pd.Series([x[1] for x in parsed], index=out.index)
    parsed_country = pd.Series([x[2] for x in parsed], index=out.index)

    def fill_existing(existing: pd.Series, parsed_values: pd.Series, normalizer) -> pd.Series:
        existing_clean = existing.map(clean_location_part)
        parsed_clean = parsed_values.map(normalizer)
        mask = existing_clean.map(lambda x: x == "")
        return existing_clean.where(~mask, parsed_clean)

    out["city"] = fill_existing(out["city"], parsed_city, clean_location_part)
    out["state"] = fill_existing(out["state"], parsed_state, normalize_state)
    out["country"] = fill_existing(out["country"], parsed_country, normalize_country)

    # Simple presence flags, useful for later metadata binary columns.
    out["has_city"] = out["city"].map(lambda x: int(bool(clean_location_part(x))))
    out["has_state"] = out["state"].map(lambda x: int(bool(clean_location_part(x))))
    out["has_country"] = out["country"].map(lambda x: int(bool(clean_location_part(x))))

    LOGGER.info(
        "Location enrichment: city=%d/%d, state=%d/%d, country=%d/%d",
        int(out["has_city"].sum()), len(out),
        int(out["has_state"].sum()), len(out),
        int(out["has_country"].sum()), len(out),
    )

    for col in ["city", "state", "country"]:
        top = (
            out[col]
            .map(clean_location_part)
            .replace("", pd.NA)
            .dropna()
            .value_counts()
            .head(5)
            .to_dict()
        )
        if top:
            LOGGER.info("Top %s values after enrichment: %s", col, top)

    return out

# -----------------------------------------------------------------------------
# JSON loading
# -----------------------------------------------------------------------------
def parse_json_file(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_record_keys(record: Dict[str, Any]) -> Dict[str, Any]:
    alias_map = {
        "uid": "Uid",
        "pid": "Pid",
        "tile": "Title",
        "title": "Title",
        "post_date": "Postdate",
        "postdate": "Postdate",
        "latitude": "Latitude",
        "longitude": "Longitude",
        "geoaccuracy": "Geoaccuracy",
        "category": "Category",
        "subcategory": "Subcategory",
        "concept": "Concept",
        "mediatype": "Mediatype",
        "alltags": "Alltags",
        "pathalias": "Pathalias",
        "ispublic": "Ispublic",
        "mediastatus": "Mediastatus",
        "city": "city",
        "state": "state",
        "country": "country",
    }
    out: Dict[str, Any] = {}
    for k, v in record.items():
        nk = alias_map.get(str(k).lower(), str(k))
        out[nk] = v
    return out


def load_json_records(path: Path) -> List[Dict[str, Any]]:
    data = parse_json_file(path)

    # Official SMP JSON often stores columns as dicts:
    # {"Uid": {"0": "..."}, "Pid": {"0": "..."}, ...}
    # Convert that directly through pandas and back to records.
    if isinstance(data, dict):
        if data and all(isinstance(v, dict) for v in data.values()):
            try:
                df = pd.DataFrame(data).reset_index(drop=True)
                return [normalize_record_keys(r) for r in df.to_dict(orient="records")]
            except Exception:
                pass
        if data and all(isinstance(v, list) for v in data.values()):
            try:
                df = pd.DataFrame(data).reset_index(drop=True)
                return [normalize_record_keys(r) for r in df.to_dict(orient="records")]
            except Exception:
                pass
        for key in ("data", "items", "results", "records"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break

    if isinstance(data, dict):
        data = [data]

    if not isinstance(data, list):
        raise ValueError(f"Unsupported JSON structure in {path}")

    return [normalize_record_keys(item) for item in data if isinstance(item, dict)]


def load_json_table(path: Optional[Path]) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    records = load_json_records(path)
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    LOGGER.info("Loaded %s -> %d rows, %d cols", path.name, len(df), len(df.columns))
    return df


# -----------------------------------------------------------------------------
# Table standardization
# -----------------------------------------------------------------------------
def make_post_id(uid: Any, pid: Any) -> Optional[str]:
    uid = empty_to_none(uid)
    pid = empty_to_none(pid)
    if uid is None or pid is None:
        return None
    return f"{uid}_{pid}"


def parse_img_filepath_line(line: str, split: str) -> Optional[Dict[str, Any]]:
    raw = line.strip()
    if not raw:
        return None

    normalized = raw.replace("\\", "/")
    parts = [p for p in normalized.split("/") if p not in ("", ".")]

    try:
        split_idx = parts.index(split)
        uid = parts[split_idx + 1]
        pid = Path(parts[split_idx + 2]).stem
    except Exception:
        if len(parts) < 2:
            return None
        uid = parts[-2]
        pid = Path(parts[-1]).stem

    return {
        "Uid": uid,
        "Pid": pid,
        "post_id": make_post_id(uid, pid),
        "image_path": normalized,
        "split": split,
    }


def load_img_filepath_table(path: Path, split: str) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = parse_img_filepath_line(line, split=split)
            if row is not None:
                rows.append(row)
    df = pd.DataFrame(rows)
    if len(df) == 0:
        LOGGER.warning("No image paths parsed from %s", path)
        return pd.DataFrame(columns=["Uid", "Pid", "post_id", "image_path", "split"])
    df = df.drop_duplicates(subset=["post_id"], keep="first").copy()
    LOGGER.info("Loaded image index: %s -> %d rows", path.name, len(df))
    return df


def load_label_txt(path: Path) -> pd.DataFrame:
    labels: List[Optional[float]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            lower = line.lower()
            if lower in {"label", "popularityscore", "popularity_score"}:
                continue
            labels.append(to_float(line))
    return pd.DataFrame({"label": labels})


def standardize_post_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "Uid" not in df.columns:
        df["Uid"] = None
    if "Pid" not in df.columns:
        df["Pid"] = None
    df["Uid"] = df["Uid"].map(empty_to_none).astype("string")
    df["Pid"] = df["Pid"].map(empty_to_none).astype("string")
    if "post_id" not in df.columns:
        df["post_id"] = [make_post_id(u, p) for u, p in zip(df["Uid"], df["Pid"])]
    df["post_id"] = df["post_id"].astype("string")
    return df


def standardize_user_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "Uid" not in df.columns:
        df["Uid"] = None
    df["Uid"] = df["Uid"].map(empty_to_none).astype("string")
    return df


def standardize_category_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = standardize_post_df(df)
    rename = {"Category": "category", "Subcategory": "subcategory", "Concept": "concept"}
    df = df.rename(columns=rename)
    for c in ["category", "subcategory", "concept"]:
        if c not in df.columns:
            df[c] = None
        df[c] = df[c].map(empty_to_none)
    cols = ["Uid", "Pid", "post_id", "category", "subcategory", "concept"]
    return df[cols].drop_duplicates(subset=["post_id"], keep="first").copy()


def standardize_text_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = standardize_post_df(df)
    rename = {"Title": "title", "Tile": "title", "Mediatype": "mediatype", "Alltags": "alltags"}
    df = df.rename(columns=rename)
    for c in ["title", "mediatype", "alltags"]:
        if c not in df.columns:
            df[c] = None
        df[c] = df[c].map(empty_to_none)
    cols = ["Uid", "Pid", "post_id", "title", "mediatype", "alltags"]
    return df[cols].drop_duplicates(subset=["post_id"], keep="first").copy()


def standardize_temporal_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = standardize_post_df(df)
    rename = {
        "Postdate": "postdate",
        "Latitude": "latitude",
        "Longitude": "longitude",
        "Geoaccuracy": "geoaccuracy",
    }
    df = df.rename(columns=rename)
    for c in ["postdate", "latitude", "longitude", "geoaccuracy", "city", "state", "country"]:
        if c not in df.columns:
            df[c] = None
    df["postdate"] = df["postdate"].map(to_int)
    df["latitude"] = df["latitude"].map(to_float)
    df["longitude"] = df["longitude"].map(to_float)
    df["geoaccuracy"] = df["geoaccuracy"].map(to_int)
    for c in ["city", "state", "country"]:
        df[c] = df[c].map(lambda x: clean_description(x) if empty_to_none(x) is not None else None)
    cols = [
        "Uid", "Pid", "post_id", "postdate", "latitude", "longitude", "geoaccuracy",
        "city", "state", "country",
    ]
    return df[cols].drop_duplicates(subset=["post_id"], keep="first").copy()


def standardize_additional_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = standardize_post_df(df)
    rename = {"Pathalias": "pathalias", "Ispublic": "ispublic", "Mediastatus": "mediastatus"}
    df = df.rename(columns=rename)
    for c in ["pathalias", "ispublic", "mediastatus"]:
        if c not in df.columns:
            df[c] = None
    df["pathalias"] = df["pathalias"].map(empty_to_none)
    df["ispublic"] = df["ispublic"].map(to_int)
    df["mediastatus"] = df["mediastatus"].map(empty_to_none)
    cols = ["Uid", "Pid", "post_id", "pathalias", "ispublic", "mediastatus"]
    return df[cols].drop_duplicates(subset=["post_id"], keep="first").copy()


def standardize_user_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = standardize_user_df(df)

    lower_to_actual = {str(c).lower(): c for c in df.columns}
    alias_candidates = {
        "photo_firstdate": ["photo_firstdate", "firstdate", "user_firstdate", "photofirstdate"],
        "photo_count": ["photo_count", "photocount", "photo_num", "photos", "post_count"],
        "ispro": ["ispro", "is_pro", "pro", "professional", "professional_status"],
        "canbuypro": ["canbuypro", "can_buy_pro"],
        "timezone_offset": ["timezone_offset", "timezoneoffset", "tz_offset"],
        "photo_firstdatetaken": ["photo_firstdatetaken", "firstdatetaken", "photo_first_date_taken"],
        "timezone_id": ["timezone_id", "timezoneid", "tz_id"],
        "user_description": ["user_description", "userdescription", "description", "bio"],
        "location_description": ["location_description", "locationdescription", "location", "hometown"],
        "profile_summary": ["profile_summary", "profile", "profile_keywords"],
        "follower_count": ["follower_count", "followers", "followers_count", "follower", "num_followers"],
        "following_count": ["following_count", "following", "following_count_num", "contacts", "contact_count", "num_following"],
        "total_views": ["total_views", "views", "view_count", "count_views", "totalviews"],
        "total_favorites": ["total_favorites", "favorites", "faves", "favourites", "favorite_count", "fave_count"],
        "mean_views": ["mean_views", "avg_views", "average_views", "mean_view"],
        "mean_favorites": ["mean_favorites", "avg_favorites", "average_favorites", "mean_faves", "mean_favorites_count"],
        "mean_tags": ["mean_tags", "avg_tags", "average_tags", "mean_tag"],
    }

    mapping: Dict[str, str] = {}
    for dst, aliases in alias_candidates.items():
        for src in aliases:
            if src in lower_to_actual:
                mapping[lower_to_actual[src]] = dst
                break
    df = df.rename(columns=mapping)

    keep_cols = [
        "Uid", "photo_firstdate", "photo_count", "ispro", "canbuypro",
        "timezone_offset", "photo_firstdatetaken", "timezone_id",
        "user_description", "location_description", "profile_summary",
        "follower_count", "following_count", "total_views", "total_favorites",
        "mean_views", "mean_favorites", "mean_tags",
    ]
    for c in keep_cols:
        if c not in df.columns:
            df[c] = None

    int_cols = [
        "photo_firstdate", "photo_count", "ispro", "canbuypro",
        "timezone_offset", "photo_firstdatetaken", "timezone_id",
        "follower_count", "following_count", "total_views", "total_favorites",
    ]
    float_cols = ["mean_views", "mean_favorites", "mean_tags"]

    for c in int_cols:
        df[c] = df[c].map(to_int)
    for c in float_cols:
        df[c] = df[c].map(to_float)

    # Keep raw text, but add clean versions. Do NOT parse as vector here.
    df["user_description"] = df["user_description"].map(empty_to_none)
    df["location_description"] = df["location_description"].map(empty_to_none)
    df["profile_summary"] = df["profile_summary"].map(lambda x: clean_description(x) if empty_to_none(x) is not None else None)
    df["user_description_clean"] = df["user_description"].map(clean_description)
    df["location_description_clean"] = df["location_description"].map(clean_description)

    keep_cols += ["user_description_clean", "location_description_clean"]
    return df[keep_cols].drop_duplicates(subset=["Uid"], keep="first").copy()


# -----------------------------------------------------------------------------
# Feature engineering
# -----------------------------------------------------------------------------
def _safe_nunique(series: pd.Series) -> int:
    return int(series.dropna().nunique())


def split_tags(value: Any) -> List[str]:
    text = safe_str(value)
    if not text:
        return []
    # official Alltags often looks like: "tag1" "tag2"
    quoted = re.findall(r'"([^"]+)"', text)
    if quoted:
        items = quoted
    else:
        items = re.split(r"[\s,;|]+", text)
    return [clean_description(x) for x in items if clean_description(x)]


def count_words(text: Any) -> int:
    text = safe_str(text)
    if not text:
        return 0
    return len(re.findall(r"\S+", text))


def ratio_by_pattern(text: Any, pattern: str) -> Optional[float]:
    text = safe_str(text)
    if not text:
        return None
    matches = re.findall(pattern, text)
    return len(matches) / len(text)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    if "postdate" not in df.columns:
        return df

    ts = pd.to_numeric(df["postdate"], errors="coerce")
    dts = pd.to_datetime(ts, unit="s", utc=True, errors="coerce")

    df["datetime_utc"] = dts.map(lambda x: x.to_pydatetime() if not pd.isna(x) else None).astype("object")
    df["year"] = dts.dt.year.where(dts.notna(), other=None).astype("Int64")
    df["month"] = dts.dt.month.where(dts.notna(), other=None).astype("Int64")
    df["day"] = dts.dt.day.where(dts.notna(), other=None).astype("Int64")
    df["hour"] = dts.dt.hour.where(dts.notna(), other=None).astype("Int64")
    df["weekday"] = dts.dt.dayofweek.where(dts.notna(), other=None).astype("Int64")
    df["weekofyear"] = dts.dt.isocalendar().week.where(dts.notna(), other=pd.NA).astype("Int64")
    df["is_weekend"] = (dts.dt.dayofweek >= 5).astype("Int64").where(dts.notna(), other=pd.NA)
    df["is_night"] = ((dts.dt.hour < 6) | (dts.dt.hour >= 22)).astype("Int64").where(dts.notna(), other=pd.NA)
    df["is_workhour"] = ((dts.dt.dayofweek < 5) & (dts.dt.hour >= 9) & (dts.dt.hour < 18)).astype("Int64").where(dts.notna(), other=pd.NA)
    return df


def add_cyclic_time_features(df: pd.DataFrame) -> pd.DataFrame:
    import numpy as np

    if "hour" in df.columns:
        h = pd.to_numeric(df["hour"], errors="coerce")
        df["hour_sin"] = np.sin(2 * np.pi * h / 24)
        df["hour_cos"] = np.cos(2 * np.pi * h / 24)
    if "weekday" in df.columns:
        w = pd.to_numeric(df["weekday"], errors="coerce")
        df["weekday_sin"] = np.sin(2 * np.pi * w / 7)
        df["weekday_cos"] = np.cos(2 * np.pi * w / 7)
    if "month" in df.columns:
        m = pd.to_numeric(df["month"], errors="coerce")
        df["month_sin"] = np.sin(2 * np.pi * m / 12)
        df["month_cos"] = np.cos(2 * np.pi * m / 12)
    return df


def add_extra_features(
    df: pd.DataFrame,
    max_tags_for_full_text: int = 24,
    max_user_words: int = 40,
    max_location_words: int = 24,
    max_profile_words: int = 24,
) -> pd.DataFrame:
    for c in ["latitude", "longitude", "title", "alltags", "category", "subcategory", "concept", "city", "state", "country"]:
        if c not in df.columns:
            df[c] = None
    for c in ["user_description", "location_description", "profile_summary"]:
        if c not in df.columns:
            df[c] = None

    def has_geo(lat: Any, lon: Any) -> int:
        lat = to_float(lat)
        lon = to_float(lon)
        if lat is None and lon is None:
            return 0
        if (lat is not None and lat != 0.0) or (lon is not None and lon != 0.0):
            return 1
        return 0

    df["has_geo"] = [has_geo(a, b) for a, b in zip(df["latitude"], df["longitude"])]

    df["title_clean"] = df["title"].map(clean_description)
    df["alltags_clean"] = df["alltags"].map(lambda x: " ".join(split_tags(x)))
    if "user_description_clean" not in df.columns:
        df["user_description_clean"] = df["user_description"].map(clean_description)
    if "location_description_clean" not in df.columns:
        df["location_description_clean"] = df["location_description"].map(clean_description)
    df["profile_summary_clean"] = df["profile_summary"].map(clean_description)

    # location_text should stay clean and non-duplicated.
    # Prefer the original cleaned free-form user location.
    # If it is missing, fall back to city/state/country.
    loc_texts = []
    for loc_desc, city, state, country in zip(
        df["location_description_clean"], df["city"], df["state"], df["country"]
    ):
        loc_desc = truncate_words(loc_desc, max_location_words)
        fallback = join_unique(
            [
                clean_location_part(city),
                normalize_state(state),
                normalize_country(country),
            ],
            sep=", ",
        )
        loc_texts.append(loc_desc or fallback)

    df["location_text"] = loc_texts
    df["has_location_text"] = df["location_text"].map(lambda x: int(bool(safe_str(x))))
    df["has_city"] = df["city"].map(lambda x: int(bool(clean_location_part(x))))
    df["has_state"] = df["state"].map(lambda x: int(bool(clean_location_part(x))))
    df["has_country"] = df["country"].map(lambda x: int(bool(clean_location_part(x))))

    # Full text only uses cleaned/truncated fields.
    full_texts = []
    for title, tags, category, subcategory, concept, location_text, profile, user_desc in zip(
        df["title_clean"],
        df["alltags_clean"],
        df["category"],
        df["subcategory"],
        df["concept"],
        df["location_text"],
        df["profile_summary_clean"],
        df["user_description_clean"],
    ):
        topic = join_unique([category, subcategory, concept], sep=" ")
        tag_text = truncate_words(tags, max_tags_for_full_text)
        # Prefer LLM profile summary if available; otherwise fall back to short clean bio.
        profile_text = truncate_words(profile, max_profile_words) or truncate_words(user_desc, max_user_words)

        parts = []
        if safe_str(title):
            parts.append(safe_str(title))
        if safe_str(tag_text):
            parts.append("tags: " + tag_text)
        if safe_str(topic):
            parts.append("topic: " + topic)
        if safe_str(location_text):
            parts.append("location: " + truncate_words(location_text, max_location_words))
        if safe_str(profile_text):
            parts.append("profile: " + profile_text)
        full_texts.append(" | ".join(parts).strip())

    df["full_text"] = full_texts
    return df


def add_text_stats_features(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["title", "alltags", "full_text", "title_clean", "alltags_clean", "user_description_clean", "location_description_clean", "location_text"]:
        if c not in df.columns:
            df[c] = None

    tag_lists = df["alltags_clean"].map(split_tags)

    df["has_title"] = df["title_clean"].map(lambda x: int(bool(safe_str(x))))
    df["has_tags"] = tag_lists.map(lambda x: int(len(x) > 0))
    df["has_user_description"] = df["user_description_clean"].map(lambda x: int(bool(safe_str(x))))
    df["has_location_description"] = df["location_description_clean"].map(lambda x: int(bool(safe_str(x))))

    df["title_len"] = df["title_clean"].map(lambda x: len(safe_str(x)))
    df["tags_len"] = df["alltags_clean"].map(lambda x: len(safe_str(x)))
    df["full_text_len"] = df["full_text"].map(lambda x: len(safe_str(x)))
    df["user_description_clean_len"] = df["user_description_clean"].map(lambda x: len(safe_str(x)))
    df["location_text_len"] = df["location_text"].map(lambda x: len(safe_str(x)))

    df["title_word_count"] = df["title_clean"].map(count_words)
    df["full_text_word_count"] = df["full_text"].map(count_words)
    df["user_description_clean_word_count"] = df["user_description_clean"].map(count_words)
    df["location_text_word_count"] = df["location_text"].map(count_words)
    df["tag_count"] = tag_lists.map(len)
    df["avg_tag_len"] = tag_lists.map(lambda tags: (sum(len(t) for t in tags) / len(tags)) if tags else None)

    df["title_digit_ratio"] = df["title_clean"].map(lambda x: ratio_by_pattern(x, r"\d"))
    df["title_upper_ratio"] = df["title_clean"].map(lambda x: ratio_by_pattern(x, r"[A-Z]"))
    df["title_punct_ratio"] = df["title_clean"].map(lambda x: ratio_by_pattern(x, r"[^\w\s]"))
    df["full_text_digit_ratio"] = df["full_text"].map(lambda x: ratio_by_pattern(x, r"\d"))
    df["full_text_punct_ratio"] = df["full_text"].map(lambda x: ratio_by_pattern(x, r"[^\w\s]"))
    return df


def add_category_combo_features(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["category", "subcategory", "concept"]:
        if c not in df.columns:
            df[c] = None

    def make_combo(a: Any, b: Any) -> Optional[str]:
        a = empty_to_none(a)
        b = empty_to_none(b)
        if a is None and b is None:
            return None
        return f"{a if a is not None else 'NA'}__{b if b is not None else 'NA'}"

    df["category_subcategory_combo"] = [make_combo(a, b) for a, b in zip(df["category"], df["subcategory"])]
    df["category_concept_combo"] = [make_combo(a, b) for a, b in zip(df["category"], df["concept"])]
    return df


def add_geo_bin_features(df: pd.DataFrame, n_bins: int = 20) -> pd.DataFrame:
    for c in ["latitude", "longitude"]:
        if c not in df.columns:
            df[c] = None

    lat_series = pd.to_numeric(df["latitude"], errors="coerce")
    lon_series = pd.to_numeric(df["longitude"], errors="coerce")

    try:
        lat_bin = pd.cut(lat_series, bins=n_bins, labels=False, duplicates="drop")
    except Exception:
        lat_bin = pd.Series([None] * len(df), index=df.index)
    try:
        lon_bin = pd.cut(lon_series, bins=n_bins, labels=False, duplicates="drop")
    except Exception:
        lon_bin = pd.Series([None] * len(df), index=df.index)

    df["lat_bin"] = lat_bin.astype("Int64")
    df["lon_bin"] = lon_bin.astype("Int64")

    def make_geo_bucket(has_geo: Any, lat_b: Any, lon_b: Any) -> Optional[str]:
        if to_int(has_geo) != 1:
            return None
        if pd.isna(lat_b) or pd.isna(lon_b):
            return None
        return f"{int(lat_b)}_{int(lon_b)}"

    df["geo_cluster"] = [make_geo_bucket(h, a, b) for h, a, b in zip(df["has_geo"], df["lat_bin"], df["lon_bin"])]
    return df


def add_label_features(df: pd.DataFrame) -> pd.DataFrame:
    if "label" not in df.columns:
        return df
    df["label_log1p"] = df["label"].map(lambda x: math.log1p(x) if x is not None and not pd.isna(x) and x >= 0 else None)
    return df


def add_account_age_features(df: pd.DataFrame) -> pd.DataFrame:
    def diff_days(a: Any, b: Any) -> Optional[float]:
        a = to_float(a)
        b = to_float(b)
        if a is None or b is None:
            return None
        return (a - b) / 86400.0

    for c in ["postdate", "photo_firstdate", "photo_firstdatetaken"]:
        if c not in df.columns:
            df[c] = None

    df["account_age_days"] = [diff_days(a, b) for a, b in zip(df["postdate"], df["photo_firstdate"])]
    df["camera_age_days"] = [diff_days(a, b) for a, b in zip(df["postdate"], df["photo_firstdatetaken"])]

    for col in ["account_age_days", "camera_age_days"]:
        df[col] = df[col].map(lambda x: max(x, 0.0) if x is not None else None)
    return df


def add_user_history_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    for c in ["follower_count", "following_count", "photo_count", "total_views", "total_favorites", "mean_views", "mean_favorites", "mean_tags"]:
        if c not in df.columns:
            df[c] = None

    df["follower_following_ratio"] = [safe_div(a, b) for a, b in zip(df["follower_count"], df["following_count"])]
    df["views_per_photo"] = [safe_div(a, b) for a, b in zip(df["total_views"], df["photo_count"])]
    df["favorites_per_photo"] = [safe_div(a, b) for a, b in zip(df["total_favorites"], df["photo_count"])]

    for src in [
        "photo_count", "follower_count", "following_count", "total_views",
        "total_favorites", "mean_views", "mean_favorites", "mean_tags",
        "account_age_days", "camera_age_days",
        "user_description_clean_word_count", "location_text_word_count",
    ]:
        if src in df.columns:
            df[f"{src}_log1p"] = df[src].map(
                lambda x: math.log1p(x) if x is not None and not pd.isna(x) and x >= 0 else None
            )
    return df


def add_user_aggregate_features(train_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    """Fit label-derived user aggregate stats on train_df only and apply to target_df."""
    if train_df.empty or "Uid" not in train_df.columns or "Uid" not in target_df.columns:
        return target_df

    work = train_df.copy()
    work["label"] = pd.to_numeric(work["label"] if "label" in work.columns else None, errors="coerce")
    work["hour"] = pd.to_numeric(work["hour"] if "hour" in work.columns else None, errors="coerce")

    agg = (
        work.groupby("Uid", dropna=True)
        .agg(
            user_prev_post_count=("post_id", "count"),
            user_mean_label=("label", "mean"),
            user_median_label=("label", "median"),
            user_std_label=("label", "std"),
            user_category_nunique=("category", _safe_nunique),
            user_active_hour_mean=("hour", "mean"),
        )
        .reset_index()
    )
    return target_df.merge(agg, on="Uid", how="left")


# -----------------------------------------------------------------------------
# Split loading/building
# -----------------------------------------------------------------------------
def load_split(input_dir: Path, split: str) -> pd.DataFrame:
    prefix = split
    split_dir = input_dir / split

    img_path_file = split_dir / f"{prefix}_img_filepath.txt"
    category_file = split_dir / f"{prefix}_category.json"
    text_file = split_dir / f"{prefix}_text.json"
    temporal_file = split_dir / f"{prefix}_temporalspatial_information.json"
    user_file = split_dir / f"{prefix}_user_data.json"
    additional_file = split_dir / f"{prefix}_additional_information.json"
    label_txt_file = split_dir / f"{prefix}_label.txt"

    if not img_path_file.exists():
        raise FileNotFoundError(f"Missing required file: {img_path_file}")

    image_df = load_img_filepath_table(img_path_file, split=split)
    category_df = standardize_category_table(load_json_table(category_file))
    text_df = standardize_text_table(load_json_table(text_file))
    temporal_df = standardize_temporal_table(load_json_table(temporal_file))
    user_df = standardize_user_table(load_json_table(user_file))
    additional_df = standardize_additional_table(load_json_table(additional_file))

    out = image_df.copy()

    if split == "train" and label_txt_file.exists():
        label_df = load_label_txt(label_txt_file)
        if len(label_df) != len(out):
            raise ValueError(
                f"Label count ({len(label_df)}) != image count ({len(out)}) for split={split}. "
                "train_label.txt must align row-by-row with train_img_filepath.txt"
            )
        out = out.reset_index(drop=True)
        label_df = label_df.reset_index(drop=True)
        out["label"] = label_df["label"]
        LOGGER.info("%s split: label loaded from txt with %d entries", split, len(label_df))
    else:
        out["label"] = None
        if split == "train":
            LOGGER.warning("%s split: %s missing, labels will be None", split, label_txt_file.name)

    for name, df in [
        ("category", category_df),
        ("text", text_df),
        ("temporal", temporal_df),
        ("additional", additional_df),
    ]:
        if df.empty:
            LOGGER.warning("%s split: %s table is empty or missing", split, name)
            continue
        join_df = df.drop(columns=[c for c in ["Uid", "Pid"] if c in df.columns], errors="ignore")
        out = out.merge(join_df, on="post_id", how="left")
        LOGGER.info("%s split: after join %s -> %d rows, %d cols", split, name, len(out), len(out.columns))

    if not user_df.empty:
        out = out.merge(user_df, on="Uid", how="left")
        LOGGER.info("%s split: after join user -> %d rows, %d cols", split, len(out), len(out.columns))
    else:
        LOGGER.warning("%s split: user table is empty or missing", split)

    # Fill city/state/country from user location text before building location_text/full_text.
    # This fixes processed parquet files where country/state/city are empty but
    # location_description has values like "London, United Kingdom".
    out = enrich_location_from_text(out)

    out = add_time_features(out)
    out = add_cyclic_time_features(out)
    out = add_extra_features(out)
    out = add_text_stats_features(out)
    out = add_category_combo_features(out)
    out = add_geo_bin_features(out)
    out = add_account_age_features(out)
    out = add_user_history_features(out)
    out = add_label_features(out)

    front = [
        "split", "post_id", "Uid", "Pid", "image_path", "label", "label_log1p",
        "category", "subcategory", "concept", "category_subcategory_combo", "category_concept_combo",
        "title", "title_clean", "mediatype", "alltags", "alltags_clean", "full_text",
        "has_title", "has_tags", "title_len", "tags_len", "full_text_len",
        "title_word_count", "full_text_word_count", "tag_count", "avg_tag_len",
        "title_digit_ratio", "title_upper_ratio", "title_punct_ratio",
        "full_text_digit_ratio", "full_text_punct_ratio",
        "postdate", "datetime_utc", "year", "month", "day", "hour", "weekday", "weekofyear",
        "is_weekend", "is_night", "is_workhour",
        "hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "month_sin", "month_cos",
        "latitude", "longitude", "geoaccuracy", "has_geo", "lat_bin", "lon_bin", "geo_cluster",
        "city", "state", "country", "has_city", "has_state", "has_country",
        "location_description", "location_description_clean", "location_text", "has_location_text",
        "pathalias", "ispublic", "mediastatus",
        "photo_firstdate", "photo_count", "ispro", "canbuypro",
        "timezone_offset", "photo_firstdatetaken", "timezone_id",
        "user_description", "user_description_clean", "profile_summary", "profile_summary_clean",
        "has_user_description", "has_location_description",
        "user_description_clean_len", "user_description_clean_word_count",
        "location_text_len", "location_text_word_count",
        "follower_count", "following_count", "total_views", "total_favorites",
        "mean_views", "mean_favorites", "mean_tags",
        "follower_following_ratio", "views_per_photo", "favorites_per_photo",
        "photo_count_log1p", "follower_count_log1p", "following_count_log1p",
        "total_views_log1p", "total_favorites_log1p",
        "mean_views_log1p", "mean_favorites_log1p", "mean_tags_log1p",
        "account_age_days", "camera_age_days",
        "account_age_days_log1p", "camera_age_days_log1p",
        "user_description_clean_word_count_log1p", "location_text_word_count_log1p",
        # label-derived user aggregates: not precomputed in full_train mode
        "user_prev_post_count", "user_mean_label", "user_median_label",
        "user_std_label", "user_category_nunique", "user_active_hour_mean",
    ]
    existing_front = [c for c in front if c in out.columns]
    remaining = [c for c in out.columns if c not in existing_front]
    out = out[existing_front + remaining]
    return out


def split_train_valid(
    train_df: pd.DataFrame,
    val_ratio: float = 0.1,
    split_seed: int = 42,
    split_by: str = "user",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if train_df.empty:
        raise ValueError("train_df is empty; cannot split train/valid.")
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")

    split_by = str(split_by).lower()
    if split_by == "user":
        if "Uid" not in train_df.columns:
            raise ValueError("split_by='user' requires 'Uid' column.")
        unique_users = train_df[["Uid"]].drop_duplicates().sample(frac=1.0, random_state=split_seed).reset_index(drop=True)
        n_val_users = max(1, int(round(len(unique_users) * val_ratio)))
        val_users = set(unique_users.iloc[:n_val_users]["Uid"].astype(str).tolist())
        val_mask = train_df["Uid"].astype(str).isin(val_users)
        val_df = train_df[val_mask].copy()
        train_sub_df = train_df[~val_mask].copy()
    elif split_by == "post":
        shuffled = train_df.sample(frac=1.0, random_state=split_seed)
        n_val = max(1, int(round(len(shuffled) * val_ratio)))
        val_df = shuffled.iloc[:n_val].copy()
        train_sub_df = shuffled.iloc[n_val:].copy()
    else:
        raise ValueError(f"Unsupported split_by: {split_by}. Use 'user' or 'post'.")

    train_sub_df = train_sub_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    train_sub_df["split"] = "train"
    val_df["split"] = "val"
    LOGGER.info("Created train/val split with split_by=%s -> train=%d, val=%d", split_by, len(train_sub_df), len(val_df))
    return train_sub_df, val_df


def align_columns(*dfs: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
    all_cols: List[str] = []
    for df in dfs:
        for c in df.columns:
            if c not in all_cols:
                all_cols.append(c)

    aligned = []
    for df in dfs:
        x = df.copy()
        for c in all_cols:
            if c not in x.columns:
                x[c] = None
        aligned.append(x[all_cols].copy())
    return tuple(aligned)


def save_split(df: pd.DataFrame, output_dir: Path, split: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / f"{split}.parquet"
    jsonl_path = output_dir / f"{split}.jsonl"
    csv_path = output_dir / f"{split}.csv"

    try:
        df.to_parquet(parquet_path, index=False)
        LOGGER.info("Saved %s", parquet_path)
    except Exception as e:
        LOGGER.warning("Failed to save parquet: %s", e)

    with jsonl_path.open("w", encoding="utf-8") as f:
        for rec in df.to_dict(orient="records"):
            if isinstance(rec.get("datetime_utc"), datetime):
                rec["datetime_utc"] = rec["datetime_utc"].isoformat()
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    LOGGER.info("Saved %s", jsonl_path)

    csv_df = df.copy()
    if "datetime_utc" in csv_df.columns:
        csv_df["datetime_utc"] = csv_df["datetime_utc"].map(
            lambda x: x.isoformat() if isinstance(x, datetime) else x
        )
    csv_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    LOGGER.info("Saved %s", csv_path)


def save_summary(output_dir: Path, summary: Dict[str, Any]) -> None:
    path = output_dir / "summary.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    LOGGER.info("Saved %s", path)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build unified SMP dataset tables, v2 clean text/location pipeline.")
    parser.add_argument("--input_dir", type=str, required=True, help="Folder containing raw official files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output folder.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--split_mode",
        type=str,
        default="full_train",
        choices=["full_train", "single_split"],
        help="full_train: save official_train/test for later GroupKFold. single_split: also create train/val once.",
    )
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--split_by", type=str, default="user", choices=["user", "post"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    LOGGER.info("Input dir : %s", input_dir)
    LOGGER.info("Output dir: %s", output_dir)
    LOGGER.info("Split mode: %s", args.split_mode)

    official_train_df = load_split(input_dir, "train")
    test_df = load_split(input_dir, "test")

    if args.split_mode == "full_train":
        # Do NOT compute label-derived user aggregate features here.
        official_train_df["split"] = "train"
        test_df["split"] = "test"
        official_train_df, test_df = align_columns(official_train_df, test_df)

        save_split(official_train_df, output_dir, "official_train")
        save_split(test_df, output_dir, "test")

        summary = {
            "mode": "full_train",
            "official_train_rows": int(len(official_train_df)),
            "test_rows": int(len(test_df)),
            "official_train_cols": list(official_train_df.columns),
            "test_cols": list(test_df.columns),
            "official_train_missing_label": int(official_train_df["label"].isna().sum()) if "label" in official_train_df.columns else None,
            "test_missing_label": int(test_df["label"].isna().sum()) if "label" in test_df.columns else None,
            "note": "label-derived user aggregate features are intentionally NOT precomputed in full_train mode to avoid KFold leakage",
        }
        save_summary(output_dir, summary)
        LOGGER.info("Done. Official train shape: %s | Test shape: %s", official_train_df.shape, test_df.shape)
        return

    train_df, val_df = split_train_valid(
        official_train_df,
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
        split_by=args.split_by,
    )
    train_df = add_user_aggregate_features(train_df, train_df)
    val_df = add_user_aggregate_features(train_df, val_df)
    test_df = add_user_aggregate_features(train_df, test_df)
    train_df, val_df, test_df = align_columns(train_df, val_df, test_df)

    save_split(train_df, output_dir, "train")
    save_split(val_df, output_dir, "val")
    save_split(test_df, output_dir, "test")

    summary = {
        "mode": "single_split",
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "train_cols": list(train_df.columns),
        "val_cols": list(val_df.columns),
        "test_cols": list(test_df.columns),
        "train_missing_label": int(train_df["label"].isna().sum()) if "label" in train_df.columns else None,
        "val_missing_label": int(val_df["label"].isna().sum()) if "label" in val_df.columns else None,
        "test_missing_label": int(test_df["label"].isna().sum()) if "label" in test_df.columns else None,
    }
    save_summary(output_dir, summary)
    LOGGER.info("Done. Train shape: %s | Val shape: %s | Test shape: %s", train_df.shape, val_df.shape, test_df.shape)


if __name__ == "__main__":
    main()

# Example:
# python3 build_dataset_v2.py --input_dir /local/smp/data --output_dir /local/smp/processed_v2 --split_mode full_train
