#!/usr/bin/env python3
"""
Merge BLIP-2 captions into the processed parquet files.

Adds a 'caption' column (empty string if no caption available).
Appends caption text to full_text as: "<original> | caption: <caption>"

Usage:
  python merge_captions_to_parquet.py \
      --captions  /ssd1/lchiayu/smp-challenge/data/processed_v2/captions_blip2.json \
      --parquet   /ssd1/lchiayu/smp-challenge/data/processed_v2/official_train.parquet \
      --output    /ssd1/lchiayu/smp-challenge/data/processed_v2/official_train_cap.parquet
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--captions", required=True)
    parser.add_argument("--parquet",  required=True)
    parser.add_argument("--output",   required=True)
    args = parser.parse_args()

    print(f"[INFO] Loading captions from {args.captions} ...")
    with open(args.captions, "r", encoding="utf-8") as f:
        captions: dict = json.load(f)
    print(f"[INFO] Loaded {len(captions)} captions")

    print(f"[INFO] Loading parquet from {args.parquet} ...")
    df = pd.read_parquet(args.parquet)
    print(f"[INFO] Parquet shape: {df.shape}")

    # photo_id column: prefer Pid (numeric photo id matching caption filename stems),
    # fall back to post_id, then first column
    if "Pid" in df.columns:
        id_col = "Pid"
    elif "photo_id" in df.columns:
        id_col = "photo_id"
    elif "post_id" in df.columns:
        id_col = "post_id"
    else:
        id_col = df.columns[0]
    print(f"[INFO] Using id column: '{id_col}'")

    df["caption"] = df[id_col].astype(str).map(captions).fillna("")

    coverage = (df["caption"].str.len() > 0).sum()
    print(f"[INFO] Caption coverage: {coverage}/{len(df)} ({coverage/len(df)*100:.1f}%)")

    # Append caption to full_text for the text branch
    has_cap = df["caption"].str.len() > 0
    df["full_text_cap"] = df["full_text"].fillna("")
    df.loc[has_cap, "full_text_cap"] = (
        df.loc[has_cap, "full_text"].fillna("")
        + " | caption: "
        + df.loc[has_cap, "caption"]
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"[INFO] Saved to {output_path}")

    # Quick sanity check
    sample = df[has_cap][["full_text", "caption", "full_text_cap"]].head(2)
    for _, row in sample.iterrows():
        print(f"\nfull_text    : {row['full_text'][:80]}")
        print(f"caption      : {row['caption'][:80]}")
        print(f"full_text_cap: {row['full_text_cap'][:120]}")


if __name__ == "__main__":
    main()
