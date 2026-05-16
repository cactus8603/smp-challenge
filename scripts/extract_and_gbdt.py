#!/usr/bin/env python3
"""
test_v7: Method B stacking.
Load a trained deep model, extract embeddings from the FULL train/val sets,
then train LightGBM + CatBoost on those embeddings.

The key improvement over the GBDT monitor in train.py is using ALL training
data (monitor is capped at gbdt_max_train_batches=80 ≈ 8% of train set).
"""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.datasets.metadata_preprocessor import MetadataPreprocessor
from src.datasets.smp_dataset import SMPDataset, smp_collate_fn
from src.engine.gbdt_monitor import extract_deep_features
from src.models.fusion_model import SMPFusionModel


# ─── helpers (copied from train.py) ────────────────────────────────────────

def _safe_nunique(s): return int(s.dropna().nunique())

def ensure_user_aggregate_columns(df):
    for c in ["user_prev_post_count","user_mean_label","user_median_label",
              "user_std_label","user_category_nunique","user_active_hour_mean"]:
        if c not in df.columns:
            df[c] = None
    return df

def add_user_aggregate_features_fold(train_df, target_df):
    work = train_df.copy()
    work["label"] = pd.to_numeric(work["label"], errors="coerce")
    work["hour"]  = pd.to_numeric(work["hour"],  errors="coerce")
    agg = (
        work.groupby("Uid", dropna=True)
        .agg(user_prev_post_count=("post_id","count"),
             user_mean_label=("label","mean"),
             user_median_label=("label","median"),
             user_std_label=("label","std"),
             user_category_nunique=("category", _safe_nunique),
             user_active_hour_mean=("hour","mean"))
        .reset_index()
    )
    return target_df.merge(agg, on="Uid", how="left")

def add_geo_encoding_features(train_df, target_df):
    out = target_df.copy()
    train_label = pd.to_numeric(train_df["label"], errors="coerce")
    global_mean = float(train_label.mean()) if train_label.notna().any() else 0.0
    for col in ["country","city","location_text"]:
        if col not in train_df.columns:
            continue
        te   = train_df[[col]].assign(_label=train_label).groupby(col, dropna=True)["_label"].mean()
        freq = train_df[col].value_counts(dropna=True)
        out[f"{col}_target_enc"] = out[col].map(te).fillna(global_mean).astype(np.float32)
        out[f"{col}_freq"] = np.log1p(out[col].map(freq).fillna(0)).astype(np.float32)
    return out

def make_fold_split(df, fold, n_folds, group_col="Uid"):
    groups = df[group_col].fillna("UNK").astype(str).to_numpy()
    splits = list(GroupKFold(n_splits=n_folds).split(df, groups=groups))
    ti, vi = splits[fold]
    return df.iloc[ti].reset_index(drop=True).copy(), df.iloc[vi].reset_index(drop=True).copy()

def _spearman(y_true, y_pred):
    c = spearmanr(y_true, y_pred).correlation
    return float(0.0 if np.isnan(c) else c)


# ─── main ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src_exp",  required=True,
                   help="Source experiment dir, e.g. outputs/test_v3/fold_0")
    p.add_argument("--out_dir",  required=True,
                   help="Output dir, e.g. outputs/test_v7/fold_0")
    p.add_argument("--fold",     type=int, default=0)
    p.add_argument("--n_folds",  type=int, default=5)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    src  = Path(args.src_exp)
    out  = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # ── load resolved config ──────────────────────────────────────────────
    cfg = json.loads((src / "resolved_config.json").read_text())
    data_cfg      = cfg["data"]
    model_cfg     = cfg["model"]
    text_cfg      = cfg["text"]
    image_cfg     = cfg["image"]
    meta_cfg      = cfg["meta"]
    fusion_cfg    = cfg["fusion"]
    preprocess_cfg = cfg["preprocess"]

    # ── data ─────────────────────────────────────────────────────────────
    data_path = data_cfg["official_train_path"]
    print(f"[INFO] Loading {data_path}")
    df = pd.read_parquet(data_path)
    df = ensure_user_aggregate_columns(df)

    train_df, val_df = make_fold_split(df, args.fold, args.n_folds)
    print(f"[INFO] Fold {args.fold}: train={len(train_df)}, val={len(val_df)}")

    train_df = add_user_aggregate_features_fold(train_df, train_df)
    val_df   = add_user_aggregate_features_fold(train_df, val_df)
    train_df = add_geo_encoding_features(train_df, train_df)
    val_df   = add_geo_encoding_features(train_df, val_df)

    label_mean = train_df["label"].mean()
    label_std  = train_df["label"].std()

    # ── preprocessor ─────────────────────────────────────────────────────
    prep_path = src / "metadata_preprocessor.json"
    preprocessor = MetadataPreprocessor.load(prep_path)
    print(f"[INFO] Preprocessor loaded from {prep_path}")

    train_df = preprocessor.transform(train_df)
    val_df   = preprocessor.transform(val_df)

    # ── datasets ─────────────────────────────────────────────────────────
    common = dict(
        preprocessor=preprocessor,
        text_model_name=text_cfg["model_name"],
        image_model_name=image_cfg["model_name"],
        user_desc_emb_path=data_cfg.get("user_desc_emb_path"),
        user_desc_idx_path=data_cfg.get("user_desc_idx_path"),
        loc_desc_emb_path=data_cfg.get("loc_desc_emb_path"),
        loc_desc_idx_path=data_cfg.get("loc_desc_idx_path"),
        caption_max_freq=data_cfg.get("caption_max_freq", 50),
        normalize_label=True,
        label_mean=label_mean,
        label_std=label_std,
        max_length=int(text_cfg["max_length"]),
        use_text=bool(model_cfg["use_text"]),
        use_meta=bool(model_cfg["use_meta"]),
        use_image=bool(model_cfg["use_image"]),
        image_path_col=image_cfg.get("path_col", "image_path"),
        image_root_dir=image_cfg.get("root_dir"),
    )

    train_ds = SMPDataset(df=train_df, is_train=False, **common)
    val_ds   = SMPDataset(df=val_df,   is_train=False, **common)

    loader_kw = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=smp_collate_fn,
        shuffle=False,
    )
    train_loader = DataLoader(train_ds, **loader_kw)
    val_loader   = DataLoader(val_ds,   **loader_kw)

    # ── model ─────────────────────────────────────────────────────────────
    cat_cardinalities = [int(preprocessor.cat_cardinalities[c]) for c in preprocessor.cat_cols]
    model = SMPFusionModel(
        text_model_name=text_cfg["model_name"],
        meta_num_dim=len(preprocessor.transformed_num_cols),
        meta_cat_cardinalities=cat_cardinalities,
        meta_bin_dim=len(preprocessor.transformed_bin_cols),
        hidden_dim=int(model_cfg["hidden_dim"]),
        dropout=float(model_cfg["dropout"]),
        use_text=bool(model_cfg["use_text"]),
        use_meta=bool(model_cfg["use_meta"]),
        use_image=bool(model_cfg["use_image"]),
        image_model_name=image_cfg["model_name"],
        text_pooling=text_cfg["pooling"],
        text_trainable=bool(text_cfg["trainable"]),
        image_pretrained=bool(image_cfg.get("pretrained", True)),
        image_trainable=bool(image_cfg["trainable"]),
        fusion_type=fusion_cfg["type"],
        use_clip_similarity=bool(fusion_cfg.get("use_clip_similarity", True)),
        meta_branch_dim=int(meta_cfg["branch_dim"]),
        use_user_desc=bool(meta_cfg.get("use_user_desc", True)),
        user_desc_dim=int(meta_cfg.get("user_desc_dim", train_ds.user_desc_emb_dim or 768)),
        use_loc_desc=bool(meta_cfg.get("use_loc_desc", False)),
        loc_desc_dim=int(meta_cfg.get("loc_desc_dim", 400)),
        desc_bottleneck_dim=int(meta_cfg.get("desc_bottleneck_dim", 64)),
        user_desc_scale=float(meta_cfg.get("user_desc_scale", 0.10)),
        loc_desc_scale=float(meta_cfg.get("loc_desc_scale", 0.05)),
    ).to(device)

    ckpt_path = src / "checkpoints" / "best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[INFO] Model loaded from {ckpt_path} (epoch {ckpt.get('epoch','?')}, "
          f"best_spearman={ckpt.get('best_score','?')})")

    # ── extract features (full dataset, no batch limit) ───────────────────
    print("\n[INFO] Extracting train features (full dataset)...")
    train_feat_df = extract_deep_features(model, train_loader, str(device), max_batches=None)
    print(f"[INFO] Train features: {train_feat_df.shape}")

    print("[INFO] Extracting val features...")
    val_feat_df = extract_deep_features(model, val_loader, str(device), max_batches=None)
    print(f"[INFO] Val features: {val_feat_df.shape}")

    # ── prepare X/y ───────────────────────────────────────────────────────
    drop_cols = {"label"}
    feat_cols = [c for c in train_feat_df.columns if c not in drop_cols]

    X_train = train_feat_df[feat_cols].select_dtypes(include=[np.number])
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_train = train_feat_df["label"].astype(float).values

    X_val = val_feat_df[X_train.columns]
    X_val = X_val.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_val = val_feat_df["label"].astype(float).values

    print(f"[INFO] Feature dim: {X_train.shape[1]}, train rows: {len(X_train)}, val rows: {len(X_val)}")
    (out / "feature_columns.json").write_text(json.dumps({"feature_columns": list(X_train.columns)}, indent=2))

    deep_spearman = _spearman(y_val, val_feat_df["deep_pred"].astype(float).values)
    print(f"[INFO] Deep model val_spearman = {deep_spearman:.4f}")

    results = {"fold": args.fold, "deep_spearman": deep_spearman,
               "n_train": len(X_train), "n_val": len(X_val),
               "feature_dim": X_train.shape[1]}

    # ── LightGBM ──────────────────────────────────────────────────────────
    print("\n[LightGBM] Training on full embeddings...")
    import lightgbm as lgb

    lgb_model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=127,
        min_child_samples=20,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.5,
        reg_lambda=3.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="mae",
        callbacks=[lgb.early_stopping(100, verbose=True), lgb.log_evaluation(200)],
    )
    lgb_pred = lgb_model.predict(X_val)
    lgb_spearman = _spearman(y_val, lgb_pred)
    print(f"[LightGBM] val_spearman = {lgb_spearman:.4f}")
    results["lightgbm_spearman"] = lgb_spearman
    lgb_model.booster_.save_model(str(out / "best_lightgbm.txt"))
    pd.DataFrame({"pred": lgb_pred, "label": y_val}).to_csv(out / "lgb_val_preds.csv", index=False)

    # ── CatBoost ──────────────────────────────────────────────────────────
    print("\n[CatBoost] Training on full embeddings...")
    from catboost import CatBoostRegressor

    cb_model = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="MAE",
        iterations=3000,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=5.0,
        random_seed=42,
        od_type="Iter",
        od_wait=100,
        task_type="GPU" if torch.cuda.is_available() else "CPU",
        verbose=200,
        allow_writing_files=False,
    )
    cb_model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
    cb_pred = cb_model.predict(X_val)
    cb_spearman = _spearman(y_val, cb_pred)
    print(f"[CatBoost] val_spearman = {cb_spearman:.4f}")
    results["catboost_spearman"] = cb_spearman
    cb_model.save_model(str(out / "best_catboost.cbm"))
    pd.DataFrame({"pred": cb_pred, "label": y_val}).to_csv(out / "cb_val_preds.csv", index=False)

    # ── summary ───────────────────────────────────────────────────────────
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\n{'='*55}")
    print(f"[RESULT] Deep model   val_spearman = {deep_spearman:.4f}")
    print(f"[RESULT] LightGBM     val_spearman = {lgb_spearman:.4f}")
    print(f"[RESULT] CatBoost     val_spearman = {cb_spearman:.4f}")
    print(f"[RESULT] Saved to {out}")


if __name__ == "__main__":
    main()
