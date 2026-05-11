from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr


@torch.no_grad()
def extract_deep_features(
    model: torch.nn.Module,
    loader,
    device: str,
    max_batches: Optional[int] = None,
) -> pd.DataFrame:
    model.eval()
    rows: List[pd.DataFrame] = []

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        meta_num = batch["meta_num"].to(device)
        meta_cat = batch["meta_cat"].to(device)
        meta_bin = batch["meta_bin"].to(device)
        labels = batch["labels"].detach().cpu().numpy()

        image_tensor = batch.get("image_tensor", None)
        if image_tensor is not None and image_tensor.numel() > 0:
            image_tensor = image_tensor.to(device)
        else:
            image_tensor = None

        glove_tokens = batch.get("glove_tokens", None)
        glove_text = batch.get("glove_text", None)
        glove_token_count = batch.get("glove_token_count", None)
        if glove_token_count is not None:
            glove_token_count = glove_token_count.to(device)

        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            meta_num=meta_num,
            meta_cat=meta_cat,
            meta_bin=meta_bin,
            image_tensor=image_tensor,
            glove_tokens=glove_tokens,
            glove_text=glove_text,
            glove_token_count=glove_token_count,
            return_features=True,
        )

        pred = out["output"].squeeze(-1).detach().cpu().numpy()
        fused = out["fused"].detach().cpu().numpy()

        df = pd.DataFrame({
            "label": labels,
            "deep_pred": pred,
        })

        for name in ["text", "meta", "image"]:
            feat = out["features"].get(name)
            if feat is None:
                continue
            arr = feat.detach().cpu().numpy()
            for i in range(arr.shape[1]):
                df[f"{name}_{i}"] = arr[:, i]

        for i in range(fused.shape[1]):
            df[f"fused_{i}"] = fused[:, i]

        clip_sim = out["features"].get("clip_sim")
        if clip_sim is not None:
            df["clip_sim"] = clip_sim.squeeze(-1).detach().cpu().numpy()

        rows.append(df)

    return pd.concat(rows, axis=0, ignore_index=True)


def _spearman(y_true, y_pred) -> float:
    corr = spearmanr(y_true, y_pred).correlation
    return float(0.0 if np.isnan(corr) else corr)


def _prepare_xy(train_df: pd.DataFrame, val_df: pd.DataFrame):
    drop_cols = {"label"}
    feature_cols = [c for c in train_df.columns if c not in drop_cols]

    X_train = train_df[feature_cols].copy()
    y_train = train_df["label"].astype(float).values

    X_val = val_df[feature_cols].copy()
    y_val = val_df["label"].astype(float).values

    return X_train, y_train, X_val, y_val


def run_gbdt_monitor(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    device: str,
    run_lightgbm: bool = True,
    run_catboost: bool = True,
    max_train_batches: Optional[int] = None,
    max_val_batches: Optional[int] = None,
    seed: int = 42,
) -> Dict[str, float]:
    train_df = extract_deep_features(
        model=model,
        loader=train_loader,
        device=device,
        max_batches=max_train_batches,
    )
    val_df = extract_deep_features(
        model=model,
        loader=val_loader,
        device=device,
        max_batches=max_val_batches,
    )

    X_train, y_train, X_val, y_val = _prepare_xy(train_df, val_df)

    scores: Dict[str, float] = {}

    scores["deep_spearman"] = _spearman(
        y_val,
        val_df["deep_pred"].astype(float).values,
    )

    if run_lightgbm:
        try:
            import lightgbm as lgb

            lgb_model = lgb.LGBMRegressor(
                objective="regression",
                n_estimators=1000,
                learning_rate=0.03,
                num_leaves=64,
                min_child_samples=30,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=1.0,
                reg_lambda=5.0,
                random_state=seed,
                n_jobs=-1,
                verbose=-1,
            )

            lgb_model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="mae",
                callbacks=[
                    lgb.early_stopping(80, verbose=False),
                    lgb.log_evaluation(0),
                ],
            )

            lgb_pred = lgb_model.predict(X_val)
            scores["lightgbm_spearman"] = _spearman(y_val, lgb_pred)

        except Exception as e:
            scores["lightgbm_spearman"] = float("nan")
            scores["lightgbm_error"] = str(e)

    if run_catboost:
        try:
            from catboost import CatBoostRegressor

            cat_model = CatBoostRegressor(
                loss_function="RMSE",
                eval_metric="MAE",
                iterations=1000,
                learning_rate=0.03,
                depth=8,
                l2_leaf_reg=8.0,
                random_seed=seed,
                od_type="Iter",
                od_wait=80,
                verbose=False,
                allow_writing_files=False,
            )

            cat_model.fit(
                X_train,
                y_train,
                eval_set=(X_val, y_val),
                use_best_model=True,
            )

            cat_pred = cat_model.predict(X_val)
            scores["catboost_spearman"] = _spearman(y_val, cat_pred)

        except Exception as e:
            scores["catboost_spearman"] = float("nan")
            scores["catboost_error"] = str(e)

    return scores