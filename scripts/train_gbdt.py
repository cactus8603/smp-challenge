#!/usr/bin/env python3
"""
test_v6: Standalone GBDT training (LightGBM + CatBoost) on tabular features only.
No deep model, no CLIP, no image.
Uses the same fold split and feature engineering as train.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


# ─── same helpers as train.py ──────────────────────────────────────────────

def _safe_nunique(series: pd.Series) -> int:
    return int(series.dropna().nunique())


def ensure_user_aggregate_columns(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["user_prev_post_count", "user_mean_label",
              "user_median_label", "user_std_label",
              "user_category_nunique", "user_active_hour_mean"]:
        if c not in df.columns:
            df[c] = None
    return df


def add_user_aggregate_features_fold(train_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    work = train_df.copy()
    work["label"] = pd.to_numeric(work.get("label"), errors="coerce")
    work["hour"]  = pd.to_numeric(work.get("hour"),  errors="coerce")
    agg = (
        work.groupby("Uid", dropna=True)
        .agg(
            user_prev_post_count  =("post_id",   "count"),
            user_mean_label       =("label",      "mean"),
            user_median_label     =("label",      "median"),
            user_std_label        =("label",      "std"),
            user_category_nunique =("category",   _safe_nunique),
            user_active_hour_mean =("hour",        "mean"),
        )
        .reset_index()
    )
    out = target_df.merge(agg, on="Uid", how="left")
    return out


def add_geo_encoding_features(train_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    out = target_df.copy()
    train_label = pd.to_numeric(train_df["label"], errors="coerce")
    global_mean = float(train_label.mean()) if train_label.notna().any() else 0.0
    for col in ["country", "city", "location_text"]:
        if col not in train_df.columns:
            continue
        enc_col  = f"{col}_target_enc"
        freq_col = f"{col}_freq"
        te   = train_df[[col]].assign(_label=train_label).groupby(col, dropna=True)["_label"].mean()
        freq = train_df[col].value_counts(dropna=True)
        out[enc_col]  = out[col].map(te).fillna(global_mean).astype(np.float32)
        out[freq_col] = np.log1p(out[col].map(freq).fillna(0)).astype(np.float32)
    return out


def make_fold_split(df, fold, n_folds, group_col="Uid"):
    groups = df[group_col].fillna("UNK").astype(str).to_numpy()
    splits = list(GroupKFold(n_splits=n_folds).split(df, groups=groups))
    train_idx, val_idx = splits[fold]
    return df.iloc[train_idx].reset_index(drop=True).copy(), \
           df.iloc[val_idx].reset_index(drop=True).copy()


# ─── feature building ──────────────────────────────────────────────────────

NUM_COLS = [
    "hour", "weekday", "year", "month", "day", "weekofyear",
    "hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "month_sin", "month_cos",
    "latitude", "longitude", "geoaccuracy",
    "title_len", "tags_len", "full_text_len",
    "title_word_count", "full_text_word_count",
    "tag_count", "avg_tag_len",
    "title_digit_ratio", "title_upper_ratio", "title_punct_ratio",
    "full_text_digit_ratio", "full_text_punct_ratio",
    "timezone_offset",
    "photo_count_log1p", "follower_count_log1p", "following_count_log1p",
    "total_views_log1p", "total_favorites_log1p",
    "mean_views_log1p", "mean_favorites_log1p", "mean_tags_log1p",
    "account_age_days_log1p", "camera_age_days_log1p",
    "follower_following_ratio", "views_per_photo", "favorites_per_photo",
    "user_prev_post_count", "user_mean_label", "user_active_hour_mean", "user_category_nunique",
    "user_description_sentiment", "user_description_clean_len", "user_description_clean_word_count",
    # geo target encoding (computed from fold-train)
    "country_target_enc", "city_target_enc", "location_text_target_enc",
    "country_freq", "city_freq", "location_text_freq",
]

CAT_COLS = [
    "category", "subcategory", "concept",
    "category_subcategory_combo", "category_concept_combo",
    "geo_cluster", "timezone_id", "mediastatus", "mediatype",
    "city", "country",
]

BIN_COLS = [
    "is_weekend", "is_night", "is_workhour",
    "ispro", "canbuypro", "ispublic",
    "has_geo", "has_title", "has_tags", "has_user_description",
    "has_location_text", "has_timezone", "has_city", "has_country",
]


def build_features(df: pd.DataFrame):
    """Return X (DataFrame) with all available tabular features."""
    feats = {}

    for c in NUM_COLS:
        if c in df.columns:
            feats[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(np.float32)
        else:
            feats[c] = np.zeros(len(df), dtype=np.float32)

    for c in BIN_COLS:
        if c in df.columns:
            feats[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).clip(0, 1).astype(np.int8)
        else:
            feats[c] = np.zeros(len(df), dtype=np.int8)

    for c in CAT_COLS:
        if c in df.columns:
            feats[c] = df[c].fillna("UNK").astype(str)
        else:
            feats[c] = pd.Series(["UNK"] * len(df))

    return pd.DataFrame(feats, index=df.index)


# ─── main ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",    required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--fold",    type=int, default=0)
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--gpu_id",  type=int, default=-1, help="-1 = CPU; >=0 = GPU id for CatBoost")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading {args.data}")
    df = pd.read_parquet(args.data)
    print(f"[INFO] Loaded {len(df)} rows, {len(df.columns)} cols")

    df = ensure_user_aggregate_columns(df)

    train_df, val_df = make_fold_split(df, fold=args.fold, n_folds=args.n_folds)
    print(f"[INFO] Fold {args.fold}: train={len(train_df)}, val={len(val_df)}")

    # Recompute user aggregates from fold-train only (avoid leakage)
    train_df = add_user_aggregate_features_fold(train_df, train_df)
    val_df   = add_user_aggregate_features_fold(train_df, val_df)

    # Geo target encoding from fold-train only
    train_df = add_geo_encoding_features(train_df, train_df)
    val_df   = add_geo_encoding_features(train_df, val_df)

    y_train = pd.to_numeric(train_df["label"], errors="coerce").fillna(0).to_numpy()
    y_val   = pd.to_numeric(val_df["label"],   errors="coerce").fillna(0).to_numpy()

    X_train = build_features(train_df)
    X_val   = build_features(val_df)

    print(f"[INFO] Feature matrix: {X_train.shape[1]} features")

    results = {}

    # ── LightGBM ──────────────────────────────────────────────────────────
    print("\n[LightGBM] Training...")
    import lightgbm as lgb

    cat_feat_names = [c for c in CAT_COLS if c in X_train.columns]
    for c in cat_feat_names:
        X_train[c] = X_train[c].astype("category")
        X_val[c]   = X_val[c].astype("category")

    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val   = lgb.Dataset(X_val,   label=y_val, reference=lgb_train)

    lgb_params = {
        "objective":       "regression",
        "metric":          "rmse",
        "num_leaves":      255,
        "learning_rate":   0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq":    5,
        "min_child_samples": 20,
        "lambda_l1":       0.1,
        "lambda_l2":       0.1,
        "verbose":         -1,
        "n_jobs":          -1,
    }

    lgb_model = lgb.train(
        lgb_params,
        lgb_train,
        num_boost_round=2000,
        valid_sets=[lgb_val],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=100),
        ],
    )

    lgb_pred = lgb_model.predict(X_val)
    lgb_spearman = float(spearmanr(y_val, lgb_pred).correlation)
    print(f"[LightGBM] val_spearman = {lgb_spearman:.4f}")
    results["lightgbm_spearman"] = lgb_spearman

    lgb_model.save_model(str(out_dir / "lightgbm.txt"))
    pd.DataFrame({"post_id": val_df["post_id"].values, "pred": lgb_pred, "label": y_val}) \
      .to_csv(out_dir / "lgb_val_preds.csv", index=False)

    # ── CatBoost ──────────────────────────────────────────────────────────
    print("\n[CatBoost] Training...")
    from catboost import CatBoostRegressor, Pool

    # Reset cat cols back to string for CatBoost
    X_train_cb = build_features(train_df)
    X_val_cb   = build_features(val_df)
    cat_feat_idx = [X_train_cb.columns.get_loc(c) for c in CAT_COLS if c in X_train_cb.columns]

    cb_train = Pool(X_train_cb, label=y_train, cat_features=cat_feat_idx)
    cb_val   = Pool(X_val_cb,   label=y_val,   cat_features=cat_feat_idx)

    task_type = "GPU" if args.gpu_id >= 0 else "CPU"
    devices   = str(args.gpu_id) if args.gpu_id >= 0 else None

    cb_model = CatBoostRegressor(
        iterations=3000,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=3,
        random_strength=1,
        bagging_temperature=0.5,
        eval_metric="RMSE",
        early_stopping_rounds=50,
        task_type=task_type,
        devices=devices,
        verbose=100,
    )
    cb_model.fit(cb_train, eval_set=cb_val, use_best_model=True)

    cb_pred = cb_model.predict(X_val_cb)
    cb_spearman = float(spearmanr(y_val, cb_pred).correlation)
    print(f"[CatBoost] val_spearman = {cb_spearman:.4f}")
    results["catboost_spearman"] = cb_spearman

    cb_model.save_model(str(out_dir / "catboost.cbm"))
    pd.DataFrame({"post_id": val_df["post_id"].values, "pred": cb_pred, "label": y_val}) \
      .to_csv(out_dir / "cb_val_preds.csv", index=False)

    # ── Feature importance ────────────────────────────────────────────────
    lgb_imp = pd.DataFrame({
        "feature": lgb_model.feature_name(),
        "importance": lgb_model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)
    lgb_imp.to_csv(out_dir / "lgb_feature_importance.csv", index=False)

    cb_imp = pd.DataFrame({
        "feature": X_train_cb.columns.tolist(),
        "importance": cb_model.get_feature_importance(),
    }).sort_values("importance", ascending=False)
    cb_imp.to_csv(out_dir / "cb_feature_importance.csv", index=False)

    # ── Summary ───────────────────────────────────────────────────────────
    results["fold"] = args.fold
    results["n_train"] = len(train_df)
    results["n_val"]   = len(val_df)
    results["n_features"] = X_train.shape[1]
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    print(f"\n{'='*50}")
    print(f"[RESULT] LightGBM val_spearman = {lgb_spearman:.4f}")
    print(f"[RESULT] CatBoost  val_spearman = {cb_spearman:.4f}")
    print(f"[RESULT] Saved to {out_dir}")


if __name__ == "__main__":
    main()
