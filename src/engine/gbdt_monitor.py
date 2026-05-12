from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

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
            user_desc=batch["user_desc"].to(device) if "user_desc" in batch else None,
            loc_desc=batch["loc_desc"].to(device) if "loc_desc" in batch else None,
            return_features=True,
        )

        pred = out["output"].squeeze(-1).detach().cpu().numpy()
        fused = out["fused"].detach().cpu().numpy()

        col_dict: dict = {
            "label": labels,
            "deep_pred": pred,
        }

        for name in ["text", "meta", "image", "user_desc", "loc_desc"]:
            feat = out["features"].get(name)
            if feat is None:
                continue
            arr = feat.detach().cpu().numpy()
            for i in range(arr.shape[1]):
                col_dict[f"{name}_{i}"] = arr[:, i]

        for i in range(fused.shape[1]):
            col_dict[f"fused_{i}"] = fused[:, i]

        clip_sim = out["features"].get("clip_sim")
        if clip_sim is not None:
            col_dict["clip_sim"] = clip_sim.squeeze(-1).detach().cpu().numpy()

        rows.append(pd.DataFrame(col_dict))

    if not rows:
        raise ValueError("No batches were extracted for GBDT monitor.")

    return pd.concat(rows, axis=0, ignore_index=True)


def _spearman(y_true, y_pred) -> float:
    corr = spearmanr(y_true, y_pred).correlation
    return float(0.0 if np.isnan(corr) else corr)


def _prepare_xy(train_df: pd.DataFrame, val_df: pd.DataFrame):
    """
    Prepare numeric-only feature matrices for LightGBM/CatBoost.

    Keep the same feature columns/order between train and val, and sanitize
    NaN/inf so GBDT failures are not caused by deep feature extraction noise.
    """
    drop_cols = {"label"}
    feature_cols = [c for c in train_df.columns if c not in drop_cols]

    X_train = train_df[feature_cols].copy()
    y_train = train_df["label"].astype(float).values

    X_val = val_df[feature_cols].copy()
    y_val = val_df["label"].astype(float).values

    X_train = X_train.select_dtypes(include=[np.number])
    X_val = X_val[X_train.columns]

    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_val = X_val.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0)
    y_val = np.nan_to_num(y_val, nan=0.0, posinf=0.0, neginf=0.0)

    return X_train, y_train, X_val, y_val


def _save_json(data: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _is_better(score: float, best_score: float) -> bool:
    return bool(np.isfinite(score) and score > best_score)


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
    save_dir: Optional[Union[str, Path]] = None,
    best_lightgbm_spearman: float = float("-inf"),
    best_catboost_spearman: float = float("-inf"),
    epoch: Optional[int] = None,
) -> Dict[str, object]:
    """
    Train LightGBM/CatBoost on extracted deep features and optionally save
    only when each model beats its previous best Spearman.

    Saved files, when improved:
      save_dir/best_lightgbm.txt
      save_dir/best_catboost.cbm
      save_dir/best_gbdt_meta.json
      save_dir/feature_columns.json
    """
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

    scores: Dict[str, object] = {}
    scores["deep_spearman"] = _spearman(
        y_val,
        val_df["deep_pred"].astype(float).values,
    )
    scores["feature_dim"] = int(X_train.shape[1])
    scores["train_rows"] = int(X_train.shape[0])
    scores["val_rows"] = int(X_val.shape[0])

    save_path: Optional[Path] = Path(save_dir) if save_dir is not None else None
    if save_path is not None:
        save_path.mkdir(parents=True, exist_ok=True)
        _save_json({"feature_columns": list(X_train.columns)}, save_path / "feature_columns.json")

    meta_path = save_path / "best_gbdt_meta.json" if save_path is not None else None
    if meta_path is not None and meta_path.exists():
        try:
            best_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            best_meta = {}
    else:
        best_meta = {}

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
            lgb_score = _spearman(y_val, lgb_pred)
            scores["lightgbm_spearman"] = lgb_score
            scores["lightgbm_saved"] = False

            if save_path is not None and _is_better(lgb_score, best_lightgbm_spearman):
                model_path = save_path / "best_lightgbm.txt"
                lgb_model.booster_.save_model(str(model_path))
                scores["lightgbm_saved"] = True
                scores["lightgbm_model_path"] = str(model_path)

                best_meta["lightgbm"] = {
                    "epoch": epoch,
                    "spearman": float(lgb_score),
                    "model_path": str(model_path),
                    "feature_columns_path": str(save_path / "feature_columns.json"),
                    "best_iteration": int(getattr(lgb_model, "best_iteration_", 0) or 0),
                }

        except Exception as e:
            scores["lightgbm_spearman"] = float("nan")
            scores["lightgbm_saved"] = False
            scores["lightgbm_error"] = repr(e)

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
            cat_score = _spearman(y_val, cat_pred)
            scores["catboost_spearman"] = cat_score
            scores["catboost_saved"] = False

            if save_path is not None and _is_better(cat_score, best_catboost_spearman):
                model_path = save_path / "best_catboost.cbm"
                cat_model.save_model(str(model_path))
                scores["catboost_saved"] = True
                scores["catboost_model_path"] = str(model_path)

                best_meta["catboost"] = {
                    "epoch": epoch,
                    "spearman": float(cat_score),
                    "model_path": str(model_path),
                    "feature_columns_path": str(save_path / "feature_columns.json"),
                    "best_iteration": int(cat_model.get_best_iteration() or 0),
                }

        except Exception as e:
            scores["catboost_spearman"] = float("nan")
            scores["catboost_saved"] = False
            scores["catboost_error"] = repr(e)

    if meta_path is not None:
        _save_json(best_meta, meta_path)

    return scores
