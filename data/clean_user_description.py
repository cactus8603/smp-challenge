#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import html
import json
import re
from pathlib import Path

import pandas as pd


def safe_str(x):
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    s = str(x).strip()
    if s.lower() in {"none", "nan", "null"}:
        return ""
    return s


def clean_description(x):
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


def load_user_json(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data).reset_index(drop=True)


def save_user_json(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.to_dict(orient="list")
    with path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


def process_file(input_path: Path, output_path: Path):
    df = load_user_json(input_path)

    if "user_description" not in df.columns:
        df["user_description"] = ""
    if "location_description" not in df.columns:
        df["location_description"] = ""

    df["user_description_clean"] = df["user_description"].map(clean_description)
    df["location_description_clean"] = df["location_description"].map(clean_description)

    df["user_description_raw_words"] = df["user_description"].map(lambda x: len(safe_str(x).split()))
    df["user_description_clean_words"] = df["user_description_clean"].map(lambda x: len(safe_str(x).split()))

    df["location_description_raw_words"] = df["location_description"].map(lambda x: len(safe_str(x).split()))
    df["location_description_clean_words"] = df["location_description_clean"].map(lambda x: len(safe_str(x).split()))

    save_user_json(df, output_path)

    print(f"[OK] saved: {output_path}")
    print("user_description:")
    print("  raw mean words  :", df["user_description_raw_words"].mean())
    print("  clean mean words:", df["user_description_clean_words"].mean())
    print("  clean non-empty :", (df["user_description_clean"] != "").sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="/local/smp/data")
    parser.add_argument("--splits", default="train,test")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    for split in splits:
        input_path = input_dir / split / f"{split}_user_data.json"
        output_path = input_dir / split / f"{split}_user_data_clean.json"

        if not input_path.exists():
            print(f"[WARN] missing: {input_path}")
            continue

        process_file(input_path, output_path)


if __name__ == "__main__":
    main()