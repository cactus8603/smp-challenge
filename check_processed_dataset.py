#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Quick sanity check for processed SMP parquet/jsonl files.
"""

from pathlib import Path

import pandas as pd


DATA_DIR = Path("/local/smp/processed_v2")


def check_file(path: Path):
    print("\n" + "=" * 100)
    print(path.name)
    print("=" * 100)

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        print("skip unsupported:", path)
        return

    print("shape:", df.shape)

    print("\ncolumns:")
    print(df.columns.tolist())

    print("\nmissing label:")
    if "label" in df.columns:
        print(df["label"].isna().sum())

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
    ]

    print("\ncolumn existence:")
    for c in important_cols:
        print(f"{c:35s} -> {c in df.columns}")

    # --------------------------------------------------
    # basic stats
    # --------------------------------------------------
    def safe_len(x):
        if pd.isna(x):
            return 0
        return len(str(x).split())

    for c in [
        "user_description_clean",
        "location_description_clean",
        "location_text",
        "full_text",
    ]:
        if c not in df.columns:
            continue

        vals = df[c].fillna("").astype(str)

        non_empty = (vals != "").sum()
        lengths = vals.map(safe_len)

        print("\n" + "-" * 80)
        print(c)
        print("-" * 80)
        print("non-empty ratio :", f"{non_empty / len(df) * 100:.2f}%")
        print("mean words      :", round(lengths.mean(), 2))
        print("median words    :", round(lengths.median(), 2))
        print("max words       :", lengths.max())

    # --------------------------------------------------
    # examples
    # --------------------------------------------------
    print("\n" + "=" * 100)
    print("RANDOM EXAMPLES")
    print("=" * 100)

    sample_df = df.sample(min(10, len(df)), random_state=42)

    for idx, row in sample_df.iterrows():
        print("\n" + "-" * 100)
        print(f"row = {idx}")

        for c in [
            "title",
            "alltags",
            "user_description_clean",
            "location_description_clean",
            "location_text",
            "full_text",
        ]:
            if c not in df.columns:
                continue

            val = row.get(c, "")

            if pd.isna(val):
                val = ""

            val = str(val)

            print(f"\n[{c}]")
            print(val[:1000])

    # --------------------------------------------------
    # check bad patterns
    # --------------------------------------------------
    print("\n" + "=" * 100)
    print("BAD PATTERN CHECK")
    print("=" * 100)

    bad_patterns = [
        "http://",
        "https://",
        "www.",
        "<a ",
        "href=",
        "mailto:",
        "nofollow",
        "jpg",
        "png",
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
            count = vals.str.contains(p, case=False, regex=False).sum()
            print(f"{p:15s} -> {count}")

    print("\nDONE:", path.name)


def main():
    files = sorted(DATA_DIR.glob("*.parquet"))

    if not files:
        print("No parquet files found.")
        return

    for f in files:
        check_file(f)


if __name__ == "__main__":
    main()