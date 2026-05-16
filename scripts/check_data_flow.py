#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_data_flow.py

Check whether SMP data fields are correctly flowing through:

raw dataframe
→ fold split
→ geo/user aggregate feature engineering
→ MetadataPreprocessor
→ SMPDataset
→ DataLoader batch

This version does NOT rely only on random/head rows, because geography/profile
fields are sparse. It reports coverage, top values, and samples rows where the
relevant fields are non-empty.
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.datasets.metadata_preprocessor import MetadataPreprocessor
from src.datasets.smp_dataset import SMPDataset, smp_collate_fn
from src.utils.user_desc_embeddings import maybe_prepare_user_desc_embeddings


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a YAML mapping.")
    return data


def deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = deep_merge_dict(merged[k], v)
        else:
            merged[k] = deepcopy(v)
    return merged


def load_config_with_base(config_path: Path) -> Dict[str, Any]:
    cfg = load_yaml(config_path)
    base_key = cfg.pop("base", None)
    if base_key is None:
        return cfg

    base_path = Path(base_key)
    if not base_path.is_absolute():
        base_path = (config_path.parent / base_path).resolve()

    base_cfg = load_config_with_base(base_path)
    return deep_merge_dict(base_cfg, cfg)


def load_dataframe(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {p}")

    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    if p.suffix == ".csv":
        return pd.read_csv(p)
    if p.suffix == ".jsonl":
        return pd.read_json(p, lines=True)

    raise ValueError(f"Unsupported data file format: {p.suffix}")


def make_fold_split(
    df: pd.DataFrame,
    fold: int,
    n_folds: int,
    group_col: str = "Uid",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' not found in dataframe.")
    if "label" not in df.columns:
        raise ValueError("official_train dataframe must contain 'label' column.")

    groups = df[group_col].fillna("UNK_GROUP").astype(str).to_numpy()
    gkf = GroupKFold(n_splits=n_folds)
    splits = list(gkf.split(df, groups=groups))

    if fold < 0 or fold >= len(splits):
        raise ValueError(f"fold must be in [0, {len(splits) - 1}], got {fold}")

    train_idx, val_idx = splits[fold]
    train_df = df.iloc[train_idx].reset_index(drop=True).copy()
    val_df = df.iloc[val_idx].reset_index(drop=True).copy()
    train_df["split"] = "train"
    val_df["split"] = "val"
    return train_df, val_df


def _safe_nunique(series: pd.Series) -> int:
    return int(series.dropna().nunique())


def ensure_user_aggregate_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    required_cols = [
        "user_prev_post_count",
        "user_mean_label",
        "user_median_label",
        "user_std_label",
        "user_category_nunique",
        "user_active_hour_mean",
    ]
    for c in required_cols:
        if c not in out.columns:
            out[c] = None
    return out


def add_user_aggregate_features_fold(train_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    if train_df.empty or "Uid" not in train_df.columns or "Uid" not in target_df.columns:
        return target_df

    work = train_df.copy()
    work["label"] = pd.to_numeric(work["label"], errors="coerce")
    if "hour" in work.columns:
        work["hour"] = pd.to_numeric(work["hour"], errors="coerce")
    else:
        work["hour"] = np.nan

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

    out = target_df.drop(
        columns=[c for c in agg.columns if c != "Uid" and c in target_df.columns],
        errors="ignore",
    )
    out = out.merge(agg, on="Uid", how="left")
    return out


def add_geo_encoding_features(
    train_df: pd.DataFrame,
    target_df: pd.DataFrame,
) -> pd.DataFrame:
    out = target_df.copy()

    if "label" not in train_df.columns:
        return out

    train_label = pd.to_numeric(train_df["label"], errors="coerce")
    global_mean = float(train_label.mean()) if train_label.notna().any() else 0.0

    for col in ["country", "state", "city", "location_text"]:
        if col not in train_df.columns:
            continue
        if col not in out.columns:
            out[col] = None

        enc_col = f"{col}_target_enc"
        te = (
            train_df[[col]]
            .assign(_label=train_label)
            .groupby(col, dropna=True)["_label"]
            .mean()
            .rename(enc_col)
        )
        out[enc_col] = out[col].map(te).fillna(global_mean).astype(np.float32)

        freq_col = f"{col}_freq"
        freq = train_df[col].value_counts(dropna=True).rename(freq_col)
        out[freq_col] = np.log1p(out[col].map(freq).fillna(0)).astype(np.float32)

    return out


def print_section(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def non_empty_mask(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip().ne("")


def summarize_raw_column(df: pd.DataFrame, col: str, topn: int = 8) -> None:
    if col not in df.columns:
        print(f"[RAW] {col:<24} MISSING")
        return

    s = df[col]
    mask = non_empty_mask(s)
    non_null = int(s.notna().sum())
    non_empty = int(mask.sum())
    ratio = non_empty / len(df) * 100 if len(df) else 0.0
    nunique = int(s[mask].astype(str).nunique())

    print(
        f"[RAW] {col:<24} exists | non_null={non_null}/{len(df)} "
        f"| non_empty={non_empty}/{len(df)} ({ratio:.2f}%) | nunique={nunique}"
    )

    vc = s[mask].astype(str).value_counts().head(topn)
    if len(vc) > 0:
        print(f"      top values: {vc.to_dict()}")


def summarize_transformed_column(df: pd.DataFrame, col: str, topn: int = 8) -> None:
    if col not in df.columns:
        print(f"[TRANSFORMED] {col:<32} MISSING")
        return

    s = df[col]
    if pd.api.types.is_numeric_dtype(s):
        arr = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(arr)
        nan_count = int(np.isnan(arr).sum())
        if finite.any():
            msg = (
                f"[TRANSFORMED] {col:<32} exists | dtype={s.dtype} "
                f"| nan={nan_count} "
                f"| min={np.nanmin(arr):.4f} max={np.nanmax(arr):.4f} "
                f"| mean={np.nanmean(arr):.4f}"
            )
        else:
            msg = f"[TRANSFORMED] {col:<32} exists | dtype={s.dtype} | all non-finite"
        print(msg)
    else:
        mask = non_empty_mask(s)
        print(
            f"[TRANSFORMED] {col:<32} exists | dtype={s.dtype} "
            f"| non_empty={int(mask.sum())}/{len(df)} ({mask.mean()*100:.2f}%)"
        )
        vc = s[mask].astype(str).value_counts().head(topn)
        if len(vc) > 0:
            print(f"      top values: {vc.to_dict()}")


def tensor_stats(name: str, x: torch.Tensor) -> None:
    x_cpu = x.detach().cpu()
    if x_cpu.numel() == 0:
        print(f"[BATCH] {name:<24} shape={tuple(x_cpu.shape)} EMPTY")
        return

    if not torch.is_floating_point(x_cpu):
        x_float = x_cpu.float()
    else:
        x_float = x_cpu

    nan_count = int(torch.isnan(x_float).sum().item())
    inf_count = int(torch.isinf(x_float).sum().item())
    nonzero = int((x_float != 0).sum().item())
    total = x_float.numel()

    print(
        f"[BATCH] {name:<24} shape={tuple(x_cpu.shape)} dtype={x_cpu.dtype} "
        f"| nan={nan_count} inf={inf_count} nonzero={nonzero}/{total} "
        f"| mean={x_float.mean().item():.4f} std={x_float.std(unbiased=False).item():.4f}"
    )


def check_required_cols(df: pd.DataFrame, cols: Iterable[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"[WARN] {label} missing columns: {missing}")
    else:
        print(f"[OK] {label}: all required columns exist")


def sample_non_empty_rows(
    df: pd.DataFrame,
    filter_cols: Iterable[str],
    show_cols: List[str],
    title: str,
    n: int = 8,
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
        print("[WARN] no rows matched")
        return

    available = [c for c in show_cols if c in subset.columns]
    sample = subset.sample(min(n, len(subset)), random_state=random_state)

    print(sample[available].to_string(index=False, max_colwidth=180))


def print_geo_coverage(df: pd.DataFrame, title: str) -> None:
    print_section(title)
    for col in [
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
    ]:
        summarize_raw_column(df, col)

    sample_non_empty_rows(
        df=df,
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
        title=f"{title} | samples with geo info",
        n=10,
    )



def print_valid_geo_summary(df: pd.DataFrame, title: str) -> None:
    print_section(title)
    if "latitude" not in df.columns or "longitude" not in df.columns:
        print("[WARN] latitude/longitude missing")
        return

    lat = pd.to_numeric(df["latitude"], errors="coerce")
    lon = pd.to_numeric(df["longitude"], errors="coerce")
    valid = lat.notna() & lon.notna() & ((lat != 0.0) | (lon != 0.0))
    print(f"[RAW] valid coordinate rows: {int(valid.sum())}/{len(df)} ({valid.mean()*100:.2f}%)")
    if "has_geo" in df.columns:
        hg = pd.to_numeric(df["has_geo"], errors="coerce").fillna(0).astype(int)
        print(f"[RAW] has_geo sum: {int(hg.sum())}/{len(df)} ({hg.mean()*100:.2f}%)")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--use_saved_preprocessor",
        action="store_true",
        help="Load outputs/<exp_name>/fold_<fold>/metadata_preprocessor.json instead of fitting a fresh one.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val"],
        help="Which split to inspect with SMPDataset/DataLoader.",
    )
    args = parser.parse_args()

    cfg = load_config_with_base(Path(args.config).resolve())

    data_cfg = cfg["data"]
    text_cfg = cfg["text"]
    image_cfg = cfg["image"]
    model_cfg = cfg["model"]
    preprocess_cfg = cfg["preprocess"]

    exp_name = str(cfg["exp_name"])
    exp_dir = Path(cfg["output"]["root_dir"]) / exp_name / f"fold_{args.fold}"

    official_train_path = data_cfg.get("official_train_path")
    if official_train_path is None:
        raise ValueError("config.data.official_train_path is required.")

    print_section("1) Load raw official_train")
    df = load_dataframe(official_train_path)
    print(f"[INFO] raw dataframe shape: {df.shape}")
    print(f"[INFO] path: {official_train_path}")

    user_desc_emb_path, user_desc_idx_path = maybe_prepare_user_desc_embeddings(
        cfg=cfg,
        df=df,
        project_root=ROOT,
    )
    data_cfg = cfg["data"]

    raw_cols_to_check = [
        "post_id",
        "Uid",
        "Pid",
        "label",
        "title",
        "alltags",
        "full_text",
        "user_description",
        "user_description_clean",
        "location_description_clean",
        "location_text",
        "loc_desc_len",
        "loc_desc_word_count",
        "country",
        "state",
        "city",
        "has_country",
        "has_state",
        "has_city",
        "latitude",
        "longitude",
        "geoaccuracy",
        "image_path",
    ]
    for col in raw_cols_to_check:
        summarize_raw_column(df, col)

    print_geo_coverage(df, "1.5) Raw geo coverage + non-empty samples")
    print_valid_geo_summary(df, "1.6) Raw valid coordinate summary")

    print_section("2) Fold split + feature engineering")
    df = ensure_user_aggregate_columns(df)
    train_df, val_df = make_fold_split(df, fold=args.fold, n_folds=args.n_folds, group_col="Uid")

    train_df = add_user_aggregate_features_fold(train_df, train_df)
    val_df = add_user_aggregate_features_fold(train_df, val_df)

    train_df = add_geo_encoding_features(train_df, train_df)
    val_df = add_geo_encoding_features(train_df, val_df)

    print(f"[INFO] train shape: {train_df.shape} | val shape: {val_df.shape}")
    print(f"[INFO] train users: {train_df['Uid'].nunique()} | val users: {val_df['Uid'].nunique()}")

    target_df_raw = train_df if args.split == "train" else val_df
    print_geo_coverage(target_df_raw, f"2.5) {args.split} split geo coverage + non-empty samples")
    print_valid_geo_summary(target_df_raw, f"2.6) {args.split} split valid coordinate summary")

    geo_feature_cols = [
        "country_target_enc",
        "city_target_enc",
        "state_target_enc",
        "location_text_target_enc",
        "country_freq",
        "city_freq",
        "state_freq",
        "location_text_freq",
    ]
    for col in geo_feature_cols:
        summarize_transformed_column(target_df_raw, col)

    print_section("3) MetadataPreprocessor")
    if args.use_saved_preprocessor:
        preprocessor_path = exp_dir / "metadata_preprocessor.json"
        print(f"[INFO] loading saved preprocessor: {preprocessor_path}")
        preprocessor = MetadataPreprocessor.load(preprocessor_path)
        train_tf = preprocessor.transform(train_df)
        val_tf = preprocessor.transform(val_df)
    else:
        print("[INFO] fitting fresh preprocessor from fold train")
        preprocessor = MetadataPreprocessor(
            num_cols=preprocess_cfg.get("num_cols"),
            cat_cols=preprocess_cfg.get("cat_cols"),
            bin_cols=preprocess_cfg.get("bin_cols"),
            log1p_cols=preprocess_cfg.get("log1p_cols"),
        )
        train_tf = preprocessor.fit_transform(train_df)
        val_tf = preprocessor.transform(val_df)

    print(f"[INFO] num cols: {len(preprocessor.num_cols)}")
    print(f"[INFO] cat cols: {len(preprocessor.cat_cols)}")
    print(f"[INFO] bin cols: {len(preprocessor.bin_cols)}")
    print(f"[INFO] bin_cols = {preprocessor.bin_cols}")
    print(f"[INFO] cat_cols = {preprocessor.cat_cols}")

    for must_cat in ["country", "state", "city"]:
        if must_cat in target_df_raw.columns and must_cat not in preprocessor.cat_cols:
            print(
                f"[WARN] raw column '{must_cat}' exists, but '{must_cat}' is NOT in preprocessor.cat_cols. "
                f"It will not enter meta_cat unless you add it to cat_cols."
            )

    target_tf = train_tf if args.split == "train" else val_tf

    print("\n[CHECK] transformed geography/category columns:")
    transformed_checks = [
        "cat__country",
        "cat__city",
        "cat__state",
        "num__country_target_enc",
        "num__city_target_enc",
        "num__state_target_enc",
        "num__location_text_target_enc",
        "num__country_freq",
        "num__city_freq",
        "num__state_freq",
        "num__location_text_freq",
        "num__loc_desc_len",
        "num__loc_desc_word_count",
    ]
    for col in transformed_checks:
        summarize_transformed_column(target_tf, col)

    print("\n[CHECK] transformed has_* columns:")
    has_cols = [c for c in preprocessor.transformed_bin_cols if c.startswith("bin__has_")]
    if not has_cols:
        print("[WARN] No bin__has_* columns found. Check MetadataPreprocessor.bin_cols.")
    for col in has_cols:
        summarize_transformed_column(target_tf, col)

    sample_non_empty_rows(
        df=target_tf,
        filter_cols=["location_text", "city", "state", "country"],
        show_cols=[
            "post_id",
            "Uid",
            "location_text",
            "city",
            "state",
            "country",
            "cat__city",
            "cat__state",
            "cat__country",
            "bin__has_city",
            "bin__has_state",
            "bin__has_country",
            "num__city_freq",
            "num__state_freq",
            "num__country_freq",
        ],
        title="3.5) transformed samples with geo info",
        n=10,
    )

    print_section("4) User description embedding files")
    # Paths are prepared by maybe_prepare_user_desc_embeddings() above.
    print(f"[INFO] user_desc_emb_path: {user_desc_emb_path}")
    print(f"[INFO] user_desc_idx_path: {user_desc_idx_path}")

    if user_desc_emb_path and user_desc_idx_path:
        emb_path = Path(user_desc_emb_path)
        idx_path = Path(user_desc_idx_path)
        print(f"[INFO] embedding file exists: {emb_path.exists()}")
        print(f"[INFO] index file exists: {idx_path.exists()}")
        if emb_path.exists() and idx_path.exists():
            arr = np.load(str(emb_path), mmap_mode="r")
            idx = json.loads(idx_path.read_text(encoding="utf-8"))
            uid_series = target_tf["Uid"].astype(str)
            coverage = uid_series.isin(idx.keys()).sum()
            print(f"[INFO] user_desc embedding shape: {arr.shape}")
            print(f"[INFO] user_desc index size: {len(idx)}")
            print(f"[INFO] {args.split} UID coverage: {coverage}/{len(uid_series)} rows")

            sample_uids = uid_series[uid_series.isin(idx.keys())].head(5).tolist()
            print("[INFO] sample UID lookup with available embeddings:")
            for uid in sample_uids:
                vec = np.asarray(arr[idx[uid]])
                print(f"  UID={uid} idx={idx[uid]} vec_norm={float(np.linalg.norm(vec)):.4f}")
    else:
        print("[WARN] user_desc paths not set in config.data")

    print_section("5) SMPDataset + one DataLoader batch")
    label_mean = float(train_df["label"].mean())
    label_std = float(train_df["label"].std())

    dataset_df = target_tf
    dataset = SMPDataset(
        df=dataset_df,
        preprocessor=preprocessor,
        text_model_name=text_cfg["model_name"],
        image_model_name=image_cfg["model_name"],
        user_desc_emb_path=user_desc_emb_path,
        user_desc_idx_path=user_desc_idx_path,
        normalize_label=True,
        label_mean=label_mean,
        label_std=label_std,
        max_length=int(text_cfg["max_length"]),
        use_text=bool(model_cfg["use_text"]),
        use_meta=bool(model_cfg["use_meta"]),
        use_image=bool(model_cfg["use_image"]),
        image_path_col=image_cfg.get("path_col", "image_path"),
        image_root_dir=image_cfg.get("root_dir", None),
        is_train=(args.split == "train"),
    )

    print(f"[INFO] dataset length: {len(dataset)}")
    print(f"[INFO] dataset meta_num_dim: {dataset.meta_num_dim}")
    print(f"[INFO] dataset meta_cat_dim: {dataset.meta_cat_dim}")
    print(f"[INFO] dataset meta_bin_dim: {dataset.meta_bin_dim}")
    print(f"[INFO] dataset num_cols[:10]: {dataset.num_cols[:10]}")
    print(f"[INFO] dataset cat_cols: {dataset.cat_cols}")
    print(f"[INFO] dataset bin_cols: {dataset.bin_cols}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=False,
        collate_fn=smp_collate_fn,
        drop_last=False,
    )

    batch = next(iter(loader))
    print(f"[INFO] batch keys: {sorted(batch.keys())}")

    for key in [
        "input_ids",
        "attention_mask",
        "meta_num",
        "meta_cat",
        "meta_bin",
        "labels",
        "user_desc",
        "loc_desc",
        "image_tensor",
        "glove_token_count",
    ]:
        if key in batch and isinstance(batch[key], torch.Tensor):
            tensor_stats(key, batch[key])
        elif key in batch:
            print(f"[BATCH] {key:<24} type={type(batch[key])}")
        else:
            print(f"[BATCH] {key:<24} MISSING")

    print("\n[CHECK] meta_bin has_* values in first batch:")
    for raw_name in [
        "has_geo",
        "has_title",
        "has_tags",
        "has_full_text",
        "has_user_description",
        "has_location_text",
        "has_city",
        "has_state",
        "has_country",
        "has_image",
    ]:
        col = f"bin__{raw_name}"
        if col in dataset.bin_cols and "meta_bin" in batch:
            idx = dataset.bin_cols.index(col)
            values = batch["meta_bin"][:, idx].detach().cpu().numpy().astype(int).tolist()
            print(f"  {col:<28} idx={idx:<3} values={values}")
        else:
            print(f"  {col:<28} MISSING")

    sample_non_empty_rows(
        df=dataset.df,
        filter_cols=["location_text", "city", "state", "country"],
        show_cols=[
            "post_id",
            "Uid",
            "title",
            "country",
            "state",
            "city",
            "location_text",
            "latitude",
            "longitude",
            "image_path",
        ],
        title="5.5) dataset dataframe samples with geo info",
        n=args.batch_size,
    )

    print_section("DONE")
    print("[OK] Data flow check finished.")


if __name__ == "__main__":
    main()
