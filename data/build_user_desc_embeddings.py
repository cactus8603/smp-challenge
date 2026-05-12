#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_user_desc_embeddings.py

Encode user_description text → sentence embedding using sentence-transformers.
Reads from processed parquet, writes embeddings to .npy + uid index to .json.

Output:
    user_desc_embeddings.npy   — float32 [N_users, embed_dim]
    user_desc_uid_index.json   — {"uid1": 0, "uid2": 1, ...}  (uid → row index)

Usage:
    python3 build_user_desc_embeddings.py --parquet_path /local/smp/processed_v2/official_train.parquet --output_dir   /local/smp/processed_v2/embeddings --model_name  sentence-transformers/all-mpnet-base-v2 --batch_size 512 --device cuda

    # Extra data (no user_description → will be zero vectors, excluded from index)
    python3 build_user_desc_embeddings.py \
        --parquet_path /local/smp/extra_data_v2 \
        --output_dir   /local/smp/data/processed/embeddings \
        --model_name   sentence-transformers/all-mpnet-base-v2 \
        --append       # merge into existing index
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────
def load_parquet_or_dir(path: Path) -> pd.DataFrame:
    """Load a single parquet file or all parquet files in a directory."""
    if path.is_file():
        return pd.read_parquet(path)
    if path.is_dir():
        files = sorted(path.rglob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files found in {path}")
        log.info("Loading %d parquet files from %s", len(files), path)
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    raise FileNotFoundError(f"Path not found: {path}")


def extract_uid_texts(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Extract unique (Uid, user_description) pairs.
    Returns (uids, texts) — both lists have same length, texts may be empty string.
    """
    if "user_description" not in df.columns:
        raise ValueError("Column 'user_description' not found in parquet.")
    if "Uid" not in df.columns:
        raise ValueError("Column 'Uid' not found in parquet.")

    # One row per unique Uid, keep first occurrence
    uid_text = (
        df[["Uid", "user_description"]]
        .drop_duplicates(subset=["Uid"], keep="first")
        .copy()
    )

    uid_text["user_description"] = uid_text["user_description"].fillna("").astype(str)
    uid_text["user_description"] = uid_text["user_description"].str.strip()

    uids  = uid_text["Uid"].astype(str).tolist()
    texts = uid_text["user_description"].tolist()

    has_text = sum(1 for t in texts if t)
    log.info("Unique UIDs: %d | with non-empty description: %d", len(uids), has_text)
    return uids, texts


def load_existing_index(index_path: Path) -> Dict[str, int]:
    if not index_path.exists():
        return {}
    with index_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_existing_embeddings(emb_path: Path) -> Optional[np.ndarray]:
    if not emb_path.exists():
        return None
    return np.load(str(emb_path))


# ──────────────────────────────────────────────────────
# Encode
# ──────────────────────────────────────────────────────
def encode_texts(
    texts: List[str],
    model_name: str,
    batch_size: int,
    device: str,
    normalize: bool,
) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise SystemExit(
            "sentence-transformers not installed.\n"
            "Run: pip install sentence-transformers"
        )

    log.info("Loading model: %s", model_name)
    model = SentenceTransformer(model_name, device=device)

    # Replace empty strings with a neutral placeholder so the model
    # still produces a valid vector; we'll zero it out afterwards.
    PLACEHOLDER = "[PAD]"
    is_empty = [t == "" for t in texts]
    texts_to_encode = [t if t else PLACEHOLDER for t in texts]

    log.info("Encoding %d texts on %s (batch_size=%d)...", len(texts_to_encode), device, batch_size)
    embeddings = model.encode(
        texts_to_encode,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )  # [N, embed_dim]

    # Zero out vectors for users with no description
    for i, empty in enumerate(is_empty):
        if empty:
            embeddings[i] = 0.0

    log.info("Embedding shape: %s  dtype: %s", embeddings.shape, embeddings.dtype)
    return embeddings.astype(np.float32)


# ──────────────────────────────────────────────────────
# Save / merge
# ──────────────────────────────────────────────────────
def save_or_merge(
    uids: List[str],
    embeddings: np.ndarray,
    output_dir: Path,
    append: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    emb_path   = output_dir / "user_desc_embeddings.npy"
    index_path = output_dir / "user_desc_uid_index.json"

    if append:
        existing_index = load_existing_index(index_path)
        existing_emb   = load_existing_embeddings(emb_path)

        if existing_emb is None:
            log.warning("--append set but no existing embeddings found. Creating new.")
            existing_emb   = np.empty((0, embeddings.shape[1]), dtype=np.float32)
            existing_index = {}

        # Only add UIDs not already in the index
        new_uids: List[str] = []
        new_vecs: List[np.ndarray] = []
        for uid, vec in zip(uids, embeddings):
            if uid not in existing_index:
                existing_index[uid] = len(existing_emb) + len(new_uids)
                new_uids.append(uid)
                new_vecs.append(vec)

        if new_vecs:
            combined = np.vstack([existing_emb, np.stack(new_vecs)])
            log.info("Appended %d new UIDs (total=%d)", len(new_vecs), len(existing_index))
        else:
            combined = existing_emb
            log.info("No new UIDs to append.")

        np.save(str(emb_path), combined)
        index_path.write_text(json.dumps(existing_index, ensure_ascii=False), encoding="utf-8")

    else:
        uid_to_idx = {uid: i for i, uid in enumerate(uids)}
        np.save(str(emb_path), embeddings)
        index_path.write_text(json.dumps(uid_to_idx, ensure_ascii=False), encoding="utf-8")
        log.info("Saved %d embeddings to %s", len(uids), emb_path)

    log.info("Index saved to %s", index_path)
    log.info("Embedding file: %s  shape: %s", emb_path, np.load(str(emb_path)).shape)


# ──────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build user_description sentence embeddings.")
    p.add_argument("--parquet_path", required=True,
                   help="Path to processed parquet file or directory")
    p.add_argument("--output_dir",   required=True,
                   help="Directory to save embeddings and index")
    p.add_argument("--model_name",   default="sentence-transformers/all-mpnet-base-v2",
                   help="sentence-transformers model name")
    p.add_argument("--batch_size",   type=int, default=512)
    p.add_argument("--device",       default="cuda",
                   help="cuda / cpu / cuda:0")
    p.add_argument("--normalize",    action="store_true",
                   help="L2-normalize embeddings (good for cosine similarity)")
    p.add_argument("--append",       action="store_true",
                   help="Merge into existing index instead of overwriting")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    parquet_path = Path(args.parquet_path)
    output_dir   = Path(args.output_dir)

    log.info("Loading data from %s", parquet_path)
    df = load_parquet_or_dir(parquet_path)
    log.info("Loaded %d rows", len(df))

    uids, texts = extract_uid_texts(df)

    embeddings = encode_texts(
        texts=texts,
        model_name=args.model_name,
        batch_size=args.batch_size,
        device=args.device,
        normalize=args.normalize,
    )

    save_or_merge(
        uids=uids,
        embeddings=embeddings,
        output_dir=output_dir,
        append=args.append,
    )

    log.info("Done.")


if __name__ == "__main__":
    main()
