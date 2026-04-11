#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build unified SMP-Image train/val/test tables from raw official files.

Key points:
- train labels are loaded from train_label.txt by row order
- the i-th label corresponds to the i-th line in train_img_filepath.txt
- official train is split into train/val
- train/val/test feature columns are aligned
- only test has no true label
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


LOGGER = logging.getLogger("build_dataset")


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )


def empty_to_none(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        v = value.strip()
        if v == "":
            return None
        return v
    return value


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


def safe_div(a: float, b: float) -> Optional[float]:
    if a is None or b is None or b == 0:
        return None
    return a / b


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
    }
    out = {}
    for k, v in record.items():
        nk = alias_map.get(str(k).lower(), str(k))
        out[nk] = v
    return out


def load_json_records(path: Path) -> List[Dict[str, Any]]:
    data = parse_json_file(path)

    if isinstance(data, dict):
        for key in ("data", "items", "results", "records"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break

    if isinstance(data, dict):
        data = [data]

    if not isinstance(data, list):
        raise ValueError(f"Unsupported JSON structure in {path}")

    records = []
    for item in data:
        if isinstance(item, dict):
            records.append(normalize_record_keys(item))
    return records


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
    """
    Load label text file where each row is the popularity score of the
    corresponding post in train_img_filepath.txt.
    """
    labels: List[Optional[float]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue

            lower = line.lower()
            # Skip header-like lines
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


def load_json_table(path: Optional[Path]) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    records = load_json_records(path)
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    LOGGER.info("Loaded %s -> %d rows, %d cols", path.name, len(df), len(df.columns))
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
    rename = {"Postdate": "postdate", "Latitude": "latitude", "Longitude": "longitude", "Geoaccuracy": "geoaccuracy"}
    df = df.rename(columns=rename)
    for c in ["postdate", "latitude", "longitude", "geoaccuracy"]:
        if c not in df.columns:
            df[c] = None
    df["postdate"] = df["postdate"].map(to_int)
    df["latitude"] = df["latitude"].map(to_float)
    df["longitude"] = df["longitude"].map(to_float)
    df["geoaccuracy"] = df["geoaccuracy"].map(to_int)
    cols = ["Uid", "Pid", "post_id", "postdate", "latitude", "longitude", "geoaccuracy"]
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


def parse_vector_like(value: Any) -> Any:
    value = empty_to_none(value)
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                return json.loads(s)
            except Exception:
                return s
        return s
    return value


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
        "follower_count": ["follower_count", "followers", "followers_count", "follower", "num_followers"],
        "following_count": ["following_count", "following", "following_count_num", "contacts", "contact_count", "num_following"],
        "total_views": ["total_views", "views", "view_count", "count_views", "totalviews"],
        "total_favorites": ["total_favorites", "favorites", "faves", "favourites", "favorite_count", "fave_count"],
        "mean_views": ["mean_views", "avg_views", "average_views", "mean_view"],
        "mean_favorites": ["mean_favorites", "avg_favorites", "average_favorites", "mean_faves", "mean_favorites_count"],
        "mean_tags": ["mean_tags", "avg_tags", "average_tags", "mean_tag"],
    }

    mapping = {}
    for dst, aliases in alias_candidates.items():
        for src in aliases:
            if src in lower_to_actual:
                mapping[lower_to_actual[src]] = dst
                break

    df = df.rename(columns=mapping)

    keep_cols = [
        "Uid", "photo_firstdate", "photo_count", "ispro", "canbuypro",
        "timezone_offset", "photo_firstdatetaken", "timezone_id",
        "user_description", "location_description",
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

    for c in ["user_description", "location_description"]:
        df[c] = df[c].map(parse_vector_like)

    return df[keep_cols].drop_duplicates(subset=["Uid"], keep="first").copy()


def _safe_nunique(series: pd.Series) -> int:
    return int(series.dropna().nunique())


def add_user_history_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    for c in ["follower_count", "following_count", "photo_count", "total_views", "total_favorites", "mean_views", "mean_favorites", "mean_tags"]:
        if c not in df.columns:
            df[c] = None

    def add_ratio(num_col: str, den_col: str, out_col: str) -> None:
        df[out_col] = [safe_div(to_float(a), to_float(b)) for a, b in zip(df[num_col], df[den_col])]

    add_ratio("follower_count", "following_count", "follower_following_ratio")
    add_ratio("total_views", "photo_count", "views_per_photo")
    add_ratio("total_favorites", "photo_count", "favorites_per_photo")

    for src in [
        "photo_count", "follower_count", "following_count", "total_views",
        "total_favorites", "mean_views", "mean_favorites", "mean_tags",
        "account_age_days", "camera_age_days",
    ]:
        if src in df.columns:
            df[f"{src}_log1p"] = df[src].map(
                lambda x: math.log1p(x) if x is not None and not pd.isna(x) and x >= 0 else None
            )

    df["has_user_description"] = df["user_description"].map(lambda x: int(empty_to_none(x) is not None))
    df["has_location_description"] = df["location_description"].map(lambda x: int(empty_to_none(x) is not None))
    return df


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


def add_extra_features(df: pd.DataFrame) -> pd.DataFrame:
    if "latitude" not in df.columns:
        df["latitude"] = None
    if "longitude" not in df.columns:
        df["longitude"] = None

    def has_geo(lat: Any, lon: Any) -> int:
        lat = to_float(lat)
        lon = to_float(lon)
        if lat is None and lon is None:
            return 0
        if (lat is not None and lat != 0.0) or (lon is not None and lon != 0.0):
            return 1
        return 0

    df["has_geo"] = [has_geo(a, b) for a, b in zip(df["latitude"], df["longitude"])]

    if "title" not in df.columns:
        df["title"] = None
    if "alltags" not in df.columns:
        df["alltags"] = None

    def merge_text(a: Any, b: Any) -> str:
        parts = []
        if empty_to_none(a) is not None:
            parts.append(str(a).strip())
        if empty_to_none(b) is not None:
            parts.append(str(b).strip())
        return " ".join(parts).strip()

    df["full_text"] = [merge_text(a, b) for a, b in zip(df["title"], df["alltags"])]
    return df


def split_tags(value: Any) -> List[str]:
    value = empty_to_none(value)
    if value is None:
        return []
    if isinstance(value, list):
        items = value
    else:
        text = str(value).strip()
        items = re.split(r"[\s,;|]+", text)
    return [str(x).strip() for x in items if empty_to_none(x) is not None]


def count_words(text: Any) -> int:
    text = empty_to_none(text)
    if text is None:
        return 0
    return len(re.findall(r"\S+", str(text)))


def ratio_by_pattern(text: Any, pattern: str) -> Optional[float]:
    text = empty_to_none(text)
    if text is None:
        return None
    s = str(text)
    if len(s) == 0:
        return None
    matches = re.findall(pattern, s)
    return len(matches) / len(s)


def add_text_stats_features(df: pd.DataFrame) -> pd.DataFrame:
    if "title" not in df.columns:
        df["title"] = None
    if "alltags" not in df.columns:
        df["alltags"] = None
    if "full_text" not in df.columns:
        df["full_text"] = None

    tag_lists = df["alltags"].map(split_tags)

    df["has_title"] = df["title"].map(lambda x: int(empty_to_none(x) is not None))
    df["has_tags"] = tag_lists.map(lambda x: int(len(x) > 0))
    df["title_len"] = df["title"].map(lambda x: len(str(x)) if empty_to_none(x) is not None else 0)
    df["tags_len"] = df["alltags"].map(lambda x: len(str(x)) if empty_to_none(x) is not None else 0)
    df["full_text_len"] = df["full_text"].map(lambda x: len(str(x)) if empty_to_none(x) is not None else 0)
    df["title_word_count"] = df["title"].map(count_words)
    df["full_text_word_count"] = df["full_text"].map(count_words)
    df["tag_count"] = tag_lists.map(len)
    df["avg_tag_len"] = tag_lists.map(lambda tags: (sum(len(t) for t in tags) / len(tags)) if tags else None)
    df["title_digit_ratio"] = df["title"].map(lambda x: ratio_by_pattern(x, r"\d"))
    df["title_upper_ratio"] = df["title"].map(lambda x: ratio_by_pattern(x, r"[A-Z]"))
    df["title_punct_ratio"] = df["title"].map(lambda x: ratio_by_pattern(x, r"[^\w\s]"))
    df["full_text_digit_ratio"] = df["full_text"].map(lambda x: ratio_by_pattern(x, r"\d"))
    df["full_text_punct_ratio"] = df["full_text"].map(lambda x: ratio_by_pattern(x, r"[^\w\s]"))
    return df


def add_category_combo_features(df: pd.DataFrame) -> pd.DataFrame:
    if "category" not in df.columns:
        df["category"] = None
    if "subcategory" not in df.columns:
        df["subcategory"] = None
    if "concept" not in df.columns:
        df["concept"] = None

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
    if "latitude" not in df.columns:
        df["latitude"] = None
    if "longitude" not in df.columns:
        df["longitude"] = None

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

    if "postdate" not in df.columns:
        df["postdate"] = None
    if "photo_firstdate" not in df.columns:
        df["photo_firstdate"] = None
    if "photo_firstdatetaken" not in df.columns:
        df["photo_firstdatetaken"] = None

    df["account_age_days"] = [diff_days(a, b) for a, b in zip(df["postdate"], df["photo_firstdate"])]
    df["camera_age_days"] = [diff_days(a, b) for a, b in zip(df["postdate"], df["photo_firstdatetaken"])]

    for col in ["account_age_days", "camera_age_days"]:
        df[col] = df[col].map(lambda x: max(x, 0.0) if x is not None else None)
    return df


def add_user_aggregate_features(train_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep aligned output columns across train / val / test.
    """
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

    if target_df is train_df:
        sort_cols = [c for c in ["Uid", "postdate", "post_id"] if c in work.columns]
        work = work.sort_values(sort_cols, kind="mergesort", na_position="first").copy()
        g = work.groupby("Uid", dropna=False)

        work["user_prev_post_count"] = g.cumcount()
        work["_label_filled"] = work["label"].fillna(0.0)
        work["_label_seen"] = work["label"].notna().astype(int)
        work["_cum_label_sum"] = g["_label_filled"].cumsum() - work["_label_filled"]
        work["_cum_label_cnt"] = g["_label_seen"].cumsum() - work["_label_seen"]
        work["user_mean_label"] = work["_cum_label_sum"] / work["_cum_label_cnt"].replace(0, pd.NA)
        work["user_active_hour_mean"] = g["hour"].expanding().mean().reset_index(level=0, drop=True).shift(1)

        def _expanding_nunique(series: pd.Series) -> pd.Series:
            result = []
            seen = set()
            for val in series:
                result.append(len(seen))
                cat = empty_to_none(val)
                if cat is not None:
                    seen.add(cat)
            return pd.Series(result, index=series.index)

        work["user_category_nunique"] = (
            work.groupby("Uid", dropna=False)["category"].transform(_expanding_nunique)
            if "category" in work.columns
            else 0
        )

        out = target_df.merge(
            work[[
                "post_id",
                "user_prev_post_count",
                "user_mean_label",
                "user_active_hour_mean",
                "user_category_nunique",
            ]],
            on="post_id",
            how="left",
        ).merge(
            agg.rename(columns={
                "user_prev_post_count": "_user_prev_post_count_global",
                "user_mean_label": "_user_mean_label_global",
                "user_median_label": "_user_median_label_global",
                "user_std_label": "_user_std_label_global",
                "user_category_nunique": "_user_category_nunique_global",
                "user_active_hour_mean": "_user_active_hour_mean_global",
            }),
            on="Uid",
            how="left",
        )

        for col, fallback in [
            ("user_prev_post_count", "_user_prev_post_count_global"),
            ("user_mean_label", "_user_mean_label_global"),
            ("user_category_nunique", "_user_category_nunique_global"),
            ("user_active_hour_mean", "_user_active_hour_mean_global"),
        ]:
            out[col] = out[col].where(out[col].notna(), out[fallback])

        out["user_median_label"] = out["_user_median_label_global"]
        out["user_std_label"] = out["_user_std_label_global"]

        drop_cols = [c for c in out.columns if c.startswith("_user_")] + ["_label_filled", "_label_seen", "_cum_label_sum", "_cum_label_cnt"]
        out = out.drop(columns=[c for c in drop_cols if c in out.columns], errors="ignore")
        return out

    out = target_df.merge(agg, on="Uid", how="left")
    return out


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

    # Train labels are aligned by row order with train_img_filepath.txt
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
        "title", "mediatype", "alltags", "full_text",
        "has_title", "has_tags", "title_len", "tags_len", "full_text_len",
        "title_word_count", "full_text_word_count", "tag_count", "avg_tag_len",
        "title_digit_ratio", "title_upper_ratio", "title_punct_ratio",
        "full_text_digit_ratio", "full_text_punct_ratio",
        "postdate", "datetime_utc", "year", "month", "day", "hour", "weekday", "weekofyear",
        "is_weekend", "is_night", "is_workhour",
        "hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "month_sin", "month_cos",
        "latitude", "longitude", "geoaccuracy", "has_geo", "lat_bin", "lon_bin", "geo_cluster",
        "pathalias", "ispublic", "mediastatus",
        "photo_firstdate", "photo_count", "ispro", "canbuypro",
        "timezone_offset", "photo_firstdatetaken", "timezone_id",
        "follower_count", "following_count", "total_views", "total_favorites",
        "mean_views", "mean_favorites", "mean_tags",
        "follower_following_ratio", "views_per_photo", "favorites_per_photo",
        "photo_count_log1p", "follower_count_log1p", "following_count_log1p",
        "total_views_log1p", "total_favorites_log1p",
        "mean_views_log1p", "mean_favorites_log1p", "mean_tags_log1p",
        "account_age_days", "camera_age_days",
        "account_age_days_log1p", "camera_age_days_log1p",
        "has_user_description", "has_location_description",
        "user_description", "location_description",
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

    LOGGER.info(
        "Created train/val split with split_by=%s, val_ratio=%.4f -> train=%d, val=%d",
        split_by, val_ratio, len(train_sub_df), len(val_df)
    )
    return train_sub_df, val_df


def align_columns(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_cols: List[str] = []
    for df in [train_df, val_df, test_df]:
        for c in df.columns:
            if c not in all_cols:
                all_cols.append(c)

    for df in [train_df, val_df, test_df]:
        for c in all_cols:
            if c not in df.columns:
                df[c] = None

    train_df = train_df[all_cols].copy()
    val_df = val_df[all_cols].copy()
    test_df = test_df[all_cols].copy()
    return train_df, val_df, test_df


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


def save_summary(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: Path) -> None:
    summary = {
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
    path = output_dir / "summary.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    LOGGER.info("Saved %s", path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build unified SMP dataset tables with train/val split.")
    parser.add_argument("--input_dir", type=str, required=True, help="Folder containing raw official files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output folder.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation ratio split from train. Default: 0.1")
    parser.add_argument("--split_seed", type=int, default=42, help="Random seed for train/val split.")
    parser.add_argument("--split_by", type=str, default="user", choices=["user", "post"], help="Split validation by 'user' or 'post'. Default: user")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    LOGGER.info("Input dir: %s", input_dir)
    LOGGER.info("Output dir: %s", output_dir)

    official_train_df = load_split(input_dir, "train")
    test_df = load_split(input_dir, "test")

    train_df, val_df = split_train_valid(
        official_train_df,
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
        split_by=args.split_by,
    )

    # Fit user aggregate features on train only
    train_df = add_user_aggregate_features(train_df, train_df)
    val_df = add_user_aggregate_features(train_df, val_df)
    test_df = add_user_aggregate_features(train_df, test_df)

    # Keep schema aligned
    train_df, val_df, test_df = align_columns(train_df, val_df, test_df)

    save_split(train_df, output_dir, "train")
    save_split(val_df, output_dir, "val")
    save_split(test_df, output_dir, "test")
    save_summary(train_df, val_df, test_df, output_dir)

    LOGGER.info("Done.")
    LOGGER.info("Train shape: %s", train_df.shape)
    LOGGER.info("Val shape: %s", val_df.shape)
    LOGGER.info("Test shape: %s", test_df.shape)


if __name__ == "__main__":
    main()

# Example:
# python build_dataset.py --input_dir ./data/raw --output_dir ./data/processed --val_ratio 0.1 --split_by user
