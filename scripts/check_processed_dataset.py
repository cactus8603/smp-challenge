#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Quick sanity check for processed SMP parquet/csv files.

This version avoids relying only on random rows, because fields like
location_text / city / state / country / user_description are sparse.
It reports coverage first, then samples rows where the relevant fields
are actually non-empty.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import pandas as pd


DATA_DIR = Path("/local/smp/processed_v2")


def non_empty_mask(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip().ne("")


def safe_words(x) -> int:
    if pd.isna(x):
        return 0
    return len(str(x).split())


def print_section(title: str, width: int = 100) -> None:
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def print_subsection(title: str, width: int = 80) -> None:
    print("\n" + "-" * width)
    print(title)
    print("-" * width)


def summarize_col(df: pd.DataFrame, col: str, topn: int = 10) -> None:
    if col not in df.columns:
        print(f"{col:35s} -> MISSING")
        return

    s = df[col]
    mask = non_empty_mask(s)
    non_empty = int(mask.sum())
    ratio = non_empty / len(df) * 100 if len(df) else 0.0
    nunique = int(s[mask].astype(str).nunique())

    print(f"{col:35s} -> non_empty={non_empty:7d}/{len(df):7d} ({ratio:6.2f}%) | nunique={nunique}")

    if non_empty > 0:
        top = s[mask].astype(str).value_counts().head(topn)
        print(f"{'':35s}    top: {top.to_dict()}")


def summarize_flag(df: pd.DataFrame, col: str) -> None:
    if col not in df.columns:
        print(f"{col:35s} -> MISSING")
        return

    vals = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    count = int(vals.sum())
    ratio = count / len(df) * 100 if len(df) else 0.0
    uniq = sorted(vals.unique().tolist())
    print(f"{col:35s} -> sum={count:7d}/{len(df):7d} ({ratio:6.2f}%) | unique={uniq}")


def show_samples(
    df: pd.DataFrame,
    title: str,
    filter_cols: Iterable[str],
    show_cols: List[str],
    n: int = 10,
    random_state: int = 42,
) -> None:
    print_section(title)

    mask = pd.Series(False, index=df.index)
    for col in filter_cols:
        if col in df.columns:
            mask = mask | non_empty_mask(df[col])

    subset = df[mask].copy()
    print(f"[INFO] matched rows: {len(subset)}/{len(df)}")

    if subset.empty:
        print("[WARN] no rows matched this sample filter")
        return

    available = [c for c in show_cols if c in subset.columns]
    sample = subset.sample(min(n, len(subset)), random_state=random_state)

    for idx, row in sample.iterrows():
        print("\n" + "-" * 100)
        print(f"row = {idx}")
        for c in available:
            val = row.get(c, "")
            if pd.isna(val):
                val = ""
            val = str(val)
            if len(val) > 500:
                val = val[:500] + " ..."
            print(f"[{c}] {val}")


def check_bad_patterns(df: pd.DataFrame) -> None:
    print_section("BAD PATTERN CHECK")

    bad_patterns = [
        "http://",
        "https://",
        "www.",
        "<a ",
        "href=",
        "mailto:",
        "nofollow",
        ".jpg",
        ".png",
    ]

    for c in [
        "user_description_clean",
        "location_description_clean",
        "full_text",
    ]:
        if c not in df.columns:
            continue

        vals = df[c].fillna("").astype(str)

        print(f"\n[{c}]")
        for p in bad_patterns:
            count = int(vals.str.contains(p, case=False, regex=False).sum())
            print(f"{p:15s} -> {count}")


def check_file(path: Path) -> None:
    print_section(path.name)

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        print("skip unsupported:", path)
        return

    print("shape:", df.shape)

    print_section("COLUMNS")
    print(df.columns.tolist())

    if "label" in df.columns:
        print_section("LABEL")
        print("missing label:", int(df["label"].isna().sum()))

    print_section("COLUMN EXISTENCE + COVERAGE")

    important_cols = [
        "title",
        "alltags",
        "full_text",
        "user_description",
        "user_description_clean",
        "location_description",
        "location_description_clean",
        "location_text",
        "city",
        "state",
        "country",
        "latitude",
        "longitude",
        "geoaccuracy",
        "image_path",
    ]

    for c in important_cols:
        summarize_col(df, c)

    print_section("VALID GEO COORDINATES")
    if "latitude" in df.columns and "longitude" in df.columns:
        lat = pd.to_numeric(df["latitude"], errors="coerce")
        lon = pd.to_numeric(df["longitude"], errors="coerce")
        valid = lat.notna() & lon.notna() & ((lat != 0.0) | (lon != 0.0))
        print(f"valid coordinate rows: {int(valid.sum())}/{len(df)} ({valid.mean()*100:.2f}%)")
    else:
        print("latitude/longitude missing")

    print_section("HAS_* FLAG COVERAGE")
    for c in [
        "has_title",
        "has_tags",
        "has_user_description",
        "has_location_description",
        "has_location_text",
        "has_city",
        "has_state",
        "has_country",
        "has_geo",
    ]:
        summarize_flag(df, c)

    print_section("TEXT LENGTH STATS")
    for c in [
        "user_description_clean",
        "location_description_clean",
        "location_text",
        "full_text",
    ]:
        if c not in df.columns:
            continue

        vals = df[c].fillna("").astype(str)
        mask = vals.str.strip().ne("")
        lengths = vals.map(safe_words)

        print_subsection(c)
        print("non-empty ratio :", f"{mask.sum() / len(df) * 100:.2f}%")
        print("mean words      :", round(float(lengths.mean()), 2))
        print("median words    :", round(float(lengths.median()), 2))
        print("max words       :", int(lengths.max()))

        if mask.any():
            nonempty_lengths = lengths[mask]
            print("non-empty mean  :", round(float(nonempty_lengths.mean()), 2))
            print("non-empty median:", round(float(nonempty_lengths.median()), 2))

    show_samples(
        df=df,
        title="SAMPLES WITH GEO TEXT / PARSED GEO",
        filter_cols=["location_text", "city", "state", "country"],
        show_cols=[
            "post_id",
            "Uid",
            "location_description_clean",
            "location_text",
            "city",
            "state",
            "country",
            "has_city",
            "has_state",
            "has_country",
            "latitude",
            "longitude",
        ],
        n=12,
    )

    show_samples(
        df=df,
        title="SAMPLES WITH USER DESCRIPTION",
        filter_cols=["user_description_clean", "user_description"],
        show_cols=[
            "post_id",
            "Uid",
            "title_clean",
            "user_description_clean",
            "full_text",
            "has_user_description",
        ],
        n=8,
    )

    check_bad_patterns(df)

    print("\nDONE:", path.name)


def main() -> None:
    files = sorted(DATA_DIR.glob("*.parquet"))

    if not files:
        print("No parquet files found.")
        return

    for f in files:
        check_file(f)


if __name__ == "__main__":
    main()
