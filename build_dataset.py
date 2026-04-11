#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build unified SMP-Image train/test tables from raw official files.

Expected raw files (example):
  train_label.json
  train_category.json
  train_img_filepath.txt
  train_temporalspatial_information.json
  train_text.json
  train_user_data.json
  train_additional_information.json

  test_category.json
  test_img_filepath.txt
  test_temporalspatial_information.json
  test_text.json
  test_user_data.json
  test_additional_information.json

Usage:
  python build_dataset.py --input_dir /path/to/raw_files --output_dir /path/to/output

Output:
  train.parquet / train.jsonl / train.csv
  test.parquet / test.jsonl / test.csv
  summary.json

Notes:
- Post-level join key: (Uid, Pid) -> post_id = f"{Uid}_{Pid}"
- User-level join key: Uid
- Main table is built from *_img_filepath.txt
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

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
        nk = alias_map.get(str(k), alias_map.get(str(k).lower(), str(k)))
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
    """
    Expected line examples:
      train/59@N75/775.jpg
      ./train/59@N75/775.jpg
      /abs/path/to/train/59@N75/775.jpg
    We only need Uid and Pid.
    """
    raw = line.strip()
    if not raw:
        return None

    normalized = raw.replace("\\", "/")
    parts = [p for p in normalized.split("/") if p not in ("", ".")]

    # Find ".../<split>/<Uid>/<Pid>.ext"
    try:
        split_idx = parts.index(split)
        uid = parts[split_idx + 1]
        pid = Path(parts[split_idx + 2]).stem
    except Exception:
        # fallback: just use last 2 segments
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


def standardize_label_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = standardize_post_df(df)
    label_col = None
    for c in ["Label", "label", "Popularity", "popularity", "Views", "views", "score"]:
        if c in df.columns:
            label_col = c
            break
    df["label"] = df[label_col].map(to_float) if label_col else None
    cols = ["Uid", "Pid", "post_id", "label"]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df[cols].drop_duplicates(subset=["post_id"], keep="first").copy()


def standardize_category_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = standardize_post_df(df)
    rename = {
        "Category": "category",
        "Subcategory": "subcategory",
        "Concept": "concept",
    }
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
    rename = {
        "Title": "title",
        "Tile": "title",
        "Mediatype": "mediatype",
        "Alltags": "alltags",
    }
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
    rename = {
        "Pathalias": "pathalias",
        "Ispublic": "ispublic",
        "Mediastatus": "mediastatus",
    }
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

    # normalize lower-case columns if needed
    lower_to_actual = {str(c).lower(): c for c in df.columns}
    mapping = {}
    for src, dst in [
        ("photo_firstdate", "photo_firstdate"),
        ("photo_count", "photo_count"),
        ("ispro", "ispro"),
        ("canbuypro", "canbuypro"),
        ("timezone_offset", "timezone_offset"),
        ("photo_firstdatetaken", "photo_firstdatetaken"),
        ("timezone_id", "timezone_id"),
        ("user_description", "user_description"),
        ("location_description", "location_description"),
    ]:
        if src in lower_to_actual:
            mapping[lower_to_actual[src]] = dst

    df = df.rename(columns=mapping)

    keep_cols = [
        "Uid",
        "photo_firstdate",
        "photo_count",
        "ispro",
        "canbuypro",
        "timezone_offset",
        "photo_firstdatetaken",
        "timezone_id",
        "user_description",
        "location_description",
    ]
    for c in keep_cols:
        if c not in df.columns:
            df[c] = None

    int_cols = [
        "photo_firstdate",
        "photo_count",
        "ispro",
        "canbuypro",
        "timezone_offset",
        "photo_firstdatetaken",
        "timezone_id",
    ]
    for c in int_cols:
        df[c] = df[c].map(to_int)

    for c in ["user_description", "location_description"]:
        df[c] = df[c].map(parse_vector_like)

    return df[keep_cols].drop_duplicates(subset=["Uid"], keep="first").copy()


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    if "postdate" not in df.columns:
        return df

    def ts_to_dt(ts: Any) -> Optional[datetime]:
        ts = to_int(ts)
        if ts is None:
            return None
        try:
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        except Exception:
            return None

    dts = df["postdate"].map(ts_to_dt)
    df["datetime_utc"] = dts.astype("object")
    df["year"] = dts.map(lambda x: x.year if x is not None else None)
    df["month"] = dts.map(lambda x: x.month if x is not None else None)
    df["day"] = dts.map(lambda x: x.day if x is not None else None)
    df["hour"] = dts.map(lambda x: x.hour if x is not None else None)
    df["weekday"] = dts.map(lambda x: x.weekday() if x is not None else None)
    df["is_weekend"] = dts.map(lambda x: int(x.weekday() >= 5) if x is not None else None)
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


def load_split(input_dir: Path, split: str) -> pd.DataFrame:
    prefix = split

    split_dir = input_dir / split

    img_path_file = split_dir / f"{prefix}_img_filepath.txt"
    category_file = split_dir / f"{prefix}_category.json"
    text_file = split_dir / f"{prefix}_text.json"
    temporal_file = split_dir / f"{prefix}_temporalspatial_information.json"
    user_file = split_dir / f"{prefix}_user_data.json"
    additional_file = split_dir / f"{prefix}_additional_information.json"
    label_file = split_dir / f"{prefix}_label.json"

    if not img_path_file.exists():
        raise FileNotFoundError(f"Missing required file: {img_path_file}")

    image_df = load_img_filepath_table(img_path_file, split=split)
    category_df = standardize_category_table(load_json_table(category_file))
    text_df = standardize_text_table(load_json_table(text_file))
    temporal_df = standardize_temporal_table(load_json_table(temporal_file))
    user_df = standardize_user_table(load_json_table(user_file))
    additional_df = standardize_additional_table(load_json_table(additional_file))
    label_df = standardize_label_table(load_json_table(label_file)) if label_file.exists() else pd.DataFrame()

    out = image_df.copy()

    for name, df in [
        ("label", label_df),
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
    out = add_extra_features(out)

    front = [
        "split", "post_id", "Uid", "Pid", "image_path", "label",
        "category", "subcategory", "concept",
        "title", "mediatype", "alltags", "full_text",
        "postdate", "datetime_utc", "year", "month", "day", "hour", "weekday", "is_weekend",
        "latitude", "longitude", "geoaccuracy", "has_geo",
        "pathalias", "ispublic", "mediastatus",
        "photo_firstdate", "photo_count", "ispro", "canbuypro",
        "timezone_offset", "photo_firstdatetaken", "timezone_id",
        "user_description", "location_description",
    ]
    existing_front = [c for c in front if c in out.columns]
    remaining = [c for c in out.columns if c not in existing_front]
    out = out[existing_front + remaining]

    return out


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


def save_summary(train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: Path) -> None:
    summary = {
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_cols": list(train_df.columns),
        "test_cols": list(test_df.columns),
        "train_missing_label": int(train_df["label"].isna().sum()) if "label" in train_df.columns else None,
        "test_missing_label": int(test_df["label"].isna().sum()) if "label" in test_df.columns else None,
    }
    path = output_dir / "summary.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    LOGGER.info("Saved %s", path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build unified SMP dataset tables.")
    parser.add_argument("--input_dir", type=str, required=True, help="Folder containing raw official files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output folder.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    LOGGER.info("Input dir: %s", input_dir)
    LOGGER.info("Output dir: %s", output_dir)

    train_df = load_split(input_dir, "train")
    test_df = load_split(input_dir, "test")

    save_split(train_df, output_dir, "train")
    save_split(test_df, output_dir, "test")
    save_summary(train_df, test_df, output_dir)

    LOGGER.info("Done.")
    LOGGER.info("Train shape: %s", train_df.shape)
    LOGGER.info("Test shape: %s", test_df.shape)


if __name__ == "__main__":
    main()

# python build_dataset.py --input_dir ./data/raw --output_dir ./data/processed