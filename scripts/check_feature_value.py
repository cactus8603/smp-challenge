#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_feature_value.py

Check the ACTUAL inputs used by the current SMP training pipeline.

This script mirrors scripts/train.py instead of inspecting raw parquet columns
in isolation.

Pipeline:
  load config with base
  -> load official_train
  -> GroupKFold split
  -> train.py feature engineering
  -> label_mean / label_std from fold train
  -> MetadataPreprocessor fit on fold train
  -> MetadataPreprocessor transform on train/val
  -> SMPDataset construction
  -> DataLoader batches
  -> metadata / label / text / tag / image sanity checks

Usage:
  python3 scripts/check_feature_value.py \
      --config configs/text_meta_image_v2_fasttext.yaml \
      --fold 0 \
      --n_folds 5 \
      --split train \
      --batch_size 8 \
      --num_batches 3 \
      --show_ok

Fast check without loading images:
  python3 scripts/check_feature_value.py \
      --config configs/text_meta_image_v2_fasttext.yaml \
      --fold 0 \
      --n_folds 5 \
      --split train \
      --no_image \
      --show_ok

Optional:
  --save_train_tf /path/train_tf.parquet
  --save_val_tf /path/val_tf.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Reuse train.py exactly, so config merge / split / feature engineering
# behavior stays aligned with training.
from scripts.train import (
    load_config_with_base,
    load_dataframe,
    make_fold_split,
    ensure_user_aggregate_columns,
    enrich_location_from_text,
    add_user_aggregate_features_fold,
    add_geo_encoding_features,
)

from src.datasets.metadata_preprocessor import MetadataPreprocessor
from src.datasets.smp_dataset import SMPDataset, smp_collate_fn
from src.utils.user_desc_embeddings import maybe_prepare_user_desc_embeddings


CONST_STD_THRESHOLD = 1e-6
LOW_COVERAGE_THRESHOLD = 0.01
BINARY_SKEW_THRESHOLD = 0.99


def print_section(title: str, width: int = 100) -> None:
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def print_subsection(title: str, width: int = 80) -> None:
    print("\n" + "-" * width)
    print(title)
    print("-" * width)


def save_df(df: pd.DataFrame, path: Optional[str]) -> None:
    if not path:
        return

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if p.suffix.lower() == ".parquet":
        df.to_parquet(p, index=False)
    elif p.suffix.lower() == ".csv":
        df.to_csv(p, index=False)
    else:
        raise ValueError("save path must end with .parquet or .csv")

    print(f"[SAVE] {p}")


def numeric_series_stats(s: pd.Series) -> dict:
    x = pd.to_numeric(s, errors="coerce")
    n = len(x)
    n_valid = int(x.notna().sum())
    coverage = n_valid / n if n else 0.0

    return {
        "coverage": coverage,
        "n_valid": n_valid,
        "n_total": n,
        "mean": float(x.mean()) if n_valid else float("nan"),
        "std": float(x.std()) if n_valid > 1 else 0.0,
        "min": float(x.min()) if n_valid else float("nan"),
        "max": float(x.max()) if n_valid else float("nan"),
        "n_unique": int(x.dropna().nunique()),
    }


def evaluate_numeric(s: pd.Series, col: str) -> dict:
    st = numeric_series_stats(s)
    coverage = st["coverage"]
    std = st["std"]
    n_unique = st["n_unique"]
    mean = st["mean"]

    if coverage < LOW_COVERAGE_THRESHOLD:
        status = "❌"
        reason = f"coverage={coverage:.1%} too low"
    elif std < CONST_STD_THRESHOLD:
        status = "❌"
        reason = f"std={std:.2e} → constant (value={mean:.4f})"
    elif n_unique <= 1:
        status = "❌"
        reason = f"only {n_unique} unique value"
    elif coverage < 0.1:
        status = "⚠️ "
        reason = f"coverage={coverage:.1%} low but non-zero std"
    elif n_unique <= 3:
        status = "⚠️ "
        reason = f"only {n_unique} unique values"
    else:
        status = "✅"
        reason = f"std={std:.4f} nunique={n_unique}"

    return {
        "col": col,
        "type": "num",
        "status": status,
        "reason": reason,
        **st,
    }


def evaluate_categorical_id(s: pd.Series, col: str) -> dict:
    x = pd.to_numeric(s, errors="coerce").fillna(0).astype(int)
    n = len(x)
    vc = x.value_counts(dropna=False)
    n_unique = int(x.nunique())
    top_val = int(vc.index[0]) if len(vc) else 0
    top_ratio = float(vc.iloc[0]) / n if n and len(vc) else 1.0

    if n_unique <= 1:
        status = "❌"
        reason = f"only one category id={top_val}"
    elif top_ratio > BINARY_SKEW_THRESHOLD:
        status = "⚠️ "
        reason = f"top id={top_val} covers {top_ratio:.1%}"
    else:
        status = "✅"
        reason = f"cardinality={n_unique}, top_ratio={top_ratio:.1%}"

    return {
        "col": col,
        "type": "cat",
        "status": status,
        "reason": reason,
        "n_unique": n_unique,
        "top_ratio": top_ratio,
    }


def evaluate_binary(s: pd.Series, col: str) -> dict:
    x = pd.to_numeric(s, errors="coerce").fillna(0)
    n = len(x)
    mean = float(x.mean()) if n else 0.0
    n_unique = int(x.nunique())

    if mean < 1e-6:
        status = "❌"
        reason = f"all zeros (mean={mean:.4f})"
    elif mean > 1 - 1e-6:
        status = "❌"
        reason = f"all ones (mean={mean:.4f})"
    elif mean < 0.01 or mean > 0.99:
        status = "⚠️ "
        reason = f"highly skewed mean={mean:.4f}"
    else:
        status = "✅"
        reason = f"mean={mean:.4f} ({mean * 100:.1f}% positive)"

    return {
        "col": col,
        "type": "bin",
        "status": status,
        "reason": reason,
        "mean": mean,
        "n_unique": n_unique,
    }


def print_results(title: str, results: list[dict], show_ok: bool) -> None:
    print_section(f"{title} ({len(results)} features)", width=80)

    ok = [r for r in results if r["status"] == "✅"]
    warn = [r for r in results if r["status"] == "⚠️ "]
    bad = [r for r in results if r["status"] == "❌"]

    print(f"  ✅ {len(ok)} good  |  ⚠️  {len(warn)} low-value  |  ❌ {len(bad)} bad")

    if bad:
        print("\n  ❌ SHOULD REMOVE / FIX:")
        for r in bad:
            print(f"     {r['col']:<42s} {r['reason']}")

    if warn:
        print("\n  ⚠️  LOW VALUE:")
        for r in warn:
            print(f"     {r['col']:<42s} {r['reason']}")

    if show_ok and ok:
        print("\n  ✅ GOOD:")
        for r in ok:
            print(f"     {r['col']:<42s} {r['reason']}")


def print_remove_recommendation(
    num_results: list[dict],
    cat_results: list[dict],
    bin_results: list[dict],
) -> None:
    bad_num = [r["col"] for r in num_results if r["status"] == "❌"]
    bad_cat = [r["col"] for r in cat_results if r["status"] == "❌"]
    bad_bin = [r["col"] for r in bin_results if r["status"] == "❌"]

    print_section("RECOMMENDED: REMOVE THESE FROM CONFIG COLS", width=80)

    if bad_num:
        print(f"\n  num_cols remove: {bad_num}")
    if bad_cat:
        print(f"\n  cat_cols remove: {bad_cat}")
    if bad_bin:
        print(f"\n  bin_cols remove: {bad_bin}")
    if not bad_num and not bad_cat and not bad_bin:
        print("  (nothing to remove)")


def print_special_warning(num_results: list[dict]) -> None:
    bad = {r["col"] for r in num_results if r["status"] == "❌"}
    suspicious = [
        "user_prev_post_count",
        "user_mean_label",
        "user_median_label",
        "user_std_label",
        "user_active_hour_mean",
        "user_category_nunique",
    ]
    hits = [c for c in suspicious if c in bad]

    if hits:
        print_section("SPECIAL WARNING: USER AGGREGATE FEATURES LOOK BROKEN", width=80)
        print(
            "These fold-safe user aggregate features are constant/empty:\n"
            f"  {hits}\n\n"
            "Because this script mirrors train.py, this usually means the training "
            "pipeline itself is not producing these columns correctly, or the config "
            "is reading placeholder columns instead of recomputed aggregate columns."
        )


def tensor_stats(name: str, x: torch.Tensor) -> None:
    x_cpu = x.detach().cpu()
    msg = f"{name:<24s} shape={tuple(x_cpu.shape)} dtype={x_cpu.dtype}"

    if x_cpu.numel() == 0:
        print(msg + " EMPTY")
        return

    if torch.is_floating_point(x_cpu):
        finite = torch.isfinite(x_cpu)
        nan = int(torch.isnan(x_cpu).sum().item())
        inf = int(torch.isinf(x_cpu).sum().item())
        vals = x_cpu[finite].float()

        if vals.numel() > 0:
            msg += (
                f" | mean={vals.mean().item():.4f}"
                f" std={vals.std(unbiased=False).item():.4f}"
                f" min={vals.min().item():.4f}"
                f" max={vals.max().item():.4f}"
            )

        msg += f" | nan={nan} inf={inf}"
    else:
        vals = x_cpu.reshape(-1)
        msg += f" | min={vals.min().item()} max={vals.max().item()}"

    print(msg)


def list_columns_by_keyword(cols: Iterable[str], keywords: list[str]) -> None:
    cols = list(cols)
    for kw in keywords:
        hits = [c for c in cols if kw in c]
        print(f"[COLS] contains {kw!r}: {len(hits)}")
        if hits:
            print("       " + ", ".join(hits[:50]) + (" ..." if len(hits) > 50 else ""))


def build_pipeline(args: argparse.Namespace):
    cfg = load_config_with_base(Path(args.config).resolve())

    data_cfg = cfg["data"]
    text_cfg = cfg["text"]
    image_cfg = cfg["image"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    preprocess_cfg = cfg["preprocess"]

    official_train_path = data_cfg.get("official_train_path")
    if official_train_path is None:
        raise ValueError("config.data.official_train_path is required.")

    print_section("1) LOAD + SPLIT + FEATURE ENGINEERING")
    raw_df = load_dataframe(official_train_path)
    print(f"[RAW] path={official_train_path}")
    print(f"[RAW] shape={raw_df.shape}")

    # Optional reproducible Hugging Face / sentence-transformers user_description embeddings.
    user_desc_emb_path, user_desc_idx_path = maybe_prepare_user_desc_embeddings(
        cfg=cfg,
        df=raw_df,
        project_root=ROOT,
    )
    data_cfg = cfg["data"]

    if "Uid" not in raw_df.columns:
        raise ValueError("official_train dataframe must contain Uid.")
    if "label" not in raw_df.columns:
        raise ValueError("official_train dataframe must contain label.")

    raw_df = ensure_user_aggregate_columns(raw_df)

    train_df, val_df = make_fold_split(
        raw_df,
        fold=args.fold,
        n_folds=args.n_folds,
        group_col=args.group_col,
    )

    # Same order as train.py.
    train_df = enrich_location_from_text(train_df)
    val_df = enrich_location_from_text(val_df)

    train_df = add_user_aggregate_features_fold(train_df, train_df)
    val_df = add_user_aggregate_features_fold(train_df, val_df)

    train_df = add_geo_encoding_features(train_df, train_df)
    val_df = add_geo_encoding_features(train_df, val_df)

    print(f"[SPLIT] train={train_df.shape} val={val_df.shape}")
    print(f"[SPLIT] train users={train_df['Uid'].nunique()} val users={val_df['Uid'].nunique()}")

    # Same timing as train.py: after FE, before preprocessor transform.
    label_mean = float(train_df["label"].mean())
    label_std = float(train_df["label"].std())
    print(f"[LABEL] label_mean={label_mean:.6f} label_std={label_std:.6f}")

    print_section("2) METADATA PREPROCESSOR FIT/TRANSFORM")
    preprocessor = MetadataPreprocessor(
        num_cols=preprocess_cfg.get("num_cols"),
        cat_cols=preprocess_cfg.get("cat_cols"),
        bin_cols=preprocess_cfg.get("bin_cols"),
        log1p_cols=preprocess_cfg.get("log1p_cols"),
        normalize_numeric=bool(preprocess_cfg.get("normalize_numeric", True)),
        use_user_desc_lite=bool(preprocess_cfg.get("use_user_desc_lite", True)),
        user_desc_lite_debug=bool(preprocess_cfg.get("user_desc_lite_debug", True)),
        user_desc_lite_max_keywords=int(preprocess_cfg.get("user_desc_lite_max_keywords", 5)),
    )

    train_tf = preprocessor.fit_transform(train_df)
    val_tf = preprocessor.transform(val_df)

    print(f"[META] num_dim={len(preprocessor.transformed_num_cols)}")
    print(f"[META] cat_dim={len(preprocessor.transformed_cat_cols)}")
    print(f"[META] bin_dim={len(preprocessor.transformed_bin_cols)}")
    print(f"[META] cat_cardinalities_count={len(preprocessor.cat_cardinalities)}")

    print_subsection("Selected transformed columns")
    list_columns_by_keyword(
        preprocessor.transformed_num_cols + preprocessor.transformed_cat_cols + preprocessor.transformed_bin_cols,
        ["user_", "location", "country", "city", "state", "target_enc", "freq", "has_", "kw_"],
    )

    save_df(train_tf, args.save_train_tf)
    save_df(val_tf, args.save_val_tf)

    print_section("3) SMPDataset + DataLoader")
    active_tf = train_tf if args.split == "train" else val_tf
    active_is_train = args.split == "train"

    use_text = bool(model_cfg["use_text"])
    use_meta = bool(model_cfg["use_meta"])
    use_image = bool(model_cfg["use_image"]) and not args.no_image

    dataset = SMPDataset(
        df=active_tf,
        preprocessor=preprocessor,
        text_model_name=text_cfg["model_name"],
        image_model_name=image_cfg["model_name"],
        user_desc_emb_path=user_desc_emb_path,
        user_desc_idx_path=user_desc_idx_path,
        normalize_label=True,
        label_mean=label_mean,
        label_std=label_std,
        max_length=int(text_cfg["max_length"]),
        use_text=use_text,
        use_meta=use_meta,
        use_image=use_image,
        image_path_col=image_cfg.get("path_col", "image_path"),
        image_root_dir=image_cfg.get("root_dir", None),
        is_train=active_is_train,
    )

    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size or train_cfg.get("batch_size", 8)),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=False,
        persistent_workers=False,
        collate_fn=smp_collate_fn,
        drop_last=False,
    )

    print(f"[DATASET] split={args.split} len={len(dataset)}")
    print(f"[DATASET] use_text={use_text} use_meta={use_meta} use_image={use_image}")
    print(f"[DATASET] user_desc_emb_dim={getattr(dataset, 'user_desc_emb_dim', None)}")
    print(f"[DATASET] loc_desc_emb_dim={getattr(dataset, 'loc_desc_emb_dim', None)}")

    return cfg, preprocessor, train_tf, val_tf, dataset, loader, label_mean, label_std


def check_transformed_metadata(
    df: pd.DataFrame,
    preprocessor: MetadataPreprocessor,
    show_ok: bool,
) -> None:
    print_section("4) TRANSFORMED METADATA FEATURE VALUE CHECK")

    num_results = []
    for raw, col in zip(preprocessor.num_cols, preprocessor.transformed_num_cols):
        s = df[col] if col in df.columns else pd.Series([np.nan] * len(df))
        num_results.append(evaluate_numeric(s, raw))

    cat_results = []
    for raw, col in zip(preprocessor.cat_cols, preprocessor.transformed_cat_cols):
        s = df[col] if col in df.columns else pd.Series([0] * len(df))
        cat_results.append(evaluate_categorical_id(s, raw))

    bin_results = []
    for raw, col in zip(preprocessor.bin_cols, preprocessor.transformed_bin_cols):
        s = df[col] if col in df.columns else pd.Series([0] * len(df))
        bin_results.append(evaluate_binary(s, raw))

    print_results("NUMERIC FEATURES", num_results, show_ok)
    print_results("CATEGORICAL FEATURES", cat_results, show_ok)
    print_results("BINARY FEATURES", bin_results, show_ok)

    print_remove_recommendation(num_results, cat_results, bin_results)
    print_special_warning(num_results)


def check_dataset_samples(dataset: SMPDataset, n: int = 5) -> None:
    print_section("5) DATASET SAMPLE CHECK")

    n = min(n, len(dataset))
    for i in range(n):
        try:
            debug = dataset.get_debug_row(i)
        except Exception as e:
            print(f"[SAMPLE {i}] ERROR in get_debug_row: {e}")
            continue

        print_subsection(f"sample {i}")

        keys = [
            "post_id",
            "uid",
            "pid",
            "label",
            "title",
            "clip_text_preview",
            "clip_token_count_raw",
            "clip_was_truncated",
            "tag_text_preview",
            "tag_token_count",
            "tag_tokens_preview",
            "meta_num_dim",
            "meta_num_mean",
            "meta_num_std",
            "meta_cat_dim",
            "meta_bin_dim",
            "meta_bin_sum",
            "image_path",
        ]

        for k in keys:
            if k in debug:
                v = debug[k]
                s = str(v)
                if len(s) > 260:
                    s = s[:260] + " ..."
                print(f"{k:<24s}: {s}")


def check_loader_batches(loader: DataLoader, num_batches: int = 3) -> None:
    print_section("6) DATALOADER BATCH CHECK")

    label_values = []
    label_raw_values = []
    clip_token_counts = []
    tag_token_counts = []

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= num_batches:
            break

        print_subsection(f"batch {batch_idx}")

        for key in [
            "input_ids",
            "attention_mask",
            "meta_num",
            "meta_cat",
            "meta_bin",
            "user_desc",
            "loc_desc",
            "image_tensor",
            "labels",
            "label_raw",
            "clip_token_count",
            "clip_token_count_raw",
            "clip_was_truncated",
            "tag_token_count",
        ]:
            if key in batch and isinstance(batch[key], torch.Tensor):
                tensor_stats(key, batch[key])

        if "labels" in batch:
            label_values.append(batch["labels"].detach().cpu())
        if "label_raw" in batch:
            label_raw_values.append(batch["label_raw"].detach().cpu())
        if "clip_token_count" in batch:
            clip_token_counts.append(batch["clip_token_count"].detach().cpu())
        if "tag_token_count" in batch:
            tag_token_counts.append(batch["tag_token_count"].detach().cpu())

        if "clip_text" in batch:
            print("[TEXT] clip_text samples:")
            for t in batch["clip_text"][:3]:
                t = str(t)
                print("  - " + (t[:220] + " ..." if len(t) > 220 else t))

        if "tag_tokens" in batch:
            print("[TAG] tag_tokens samples:")
            for toks in batch["tag_tokens"][:3]:
                print("  - " + str(toks[:30]))

    print_section("7) BATCH SUMMARY")
    if label_values:
        x = torch.cat(label_values).float()
        tensor_stats("labels_collected", x)
    if label_raw_values:
        x = torch.cat(label_raw_values).float()
        tensor_stats("label_raw_collected", x)
    if clip_token_counts:
        x = torch.cat(clip_token_counts).float()
        tensor_stats("clip_token_count_all", x)
    if tag_token_counts:
        x = torch.cat(tag_token_counts).float()
        tensor_stats("tag_token_count_all", x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--group_col", type=str, default="Uid")

    parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_batches", type=int, default=3)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--no_image", action="store_true", help="Skip image loading for faster checks.")

    parser.add_argument("--save_train_tf", type=str, default=None)
    parser.add_argument("--save_val_tf", type=str, default=None)

    parser.add_argument("--show_ok", action="store_true")
    parser.add_argument("--skip_samples", action="store_true")
    parser.add_argument("--skip_batches", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    (
        cfg,
        preprocessor,
        train_tf,
        val_tf,
        dataset,
        loader,
        label_mean,
        label_std,
    ) = build_pipeline(args)

    active_tf = train_tf if args.split == "train" else val_tf

    check_transformed_metadata(active_tf, preprocessor, show_ok=bool(args.show_ok))

    if not args.skip_samples:
        check_dataset_samples(dataset, n=int(args.num_samples))

    if not args.skip_batches:
        check_loader_batches(loader, num_batches=int(args.num_batches))

    print("\nDone.")


if __name__ == "__main__":
    main()
