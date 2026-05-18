from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def load_dataframe(path: str | Path) -> pd.DataFrame:
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


def add_user_aggregate_features_fold(
    train_df: pd.DataFrame,
    target_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build label-derived user aggregate features from fold-train only.

    Train rows use leave-one-out aggregation for the strongest aggregate means.
    Validation/test rows use fold-train group statistics.
    """
    if train_df.empty or "Uid" not in train_df.columns or "Uid" not in target_df.columns:
        return target_df

    work = train_df.copy()
    work["label"] = pd.to_numeric(work["label"], errors="coerce")

    if "hour" in work.columns:
        work["hour"] = pd.to_numeric(work["hour"], errors="coerce")
    else:
        work["hour"] = np.nan

    is_train_self = (target_df is train_df) or (
        len(target_df) == len(train_df)
        and "post_id" in target_df.columns
        and "post_id" in train_df.columns
        and target_df["post_id"].equals(train_df["post_id"])
    )

    agg_cols_to_drop = [
        "user_prev_post_count",
        "user_mean_label",
        "user_median_label",
        "user_std_label",
        "user_category_nunique",
        "user_active_hour_mean",
    ]

    if is_train_self:
        out = train_df.drop(
            columns=[c for c in agg_cols_to_drop if c in train_df.columns],
            errors="ignore",
        ).copy()

        grp = work.groupby("Uid", dropna=True)
        user_count = grp["post_id"].transform("count")
        label_sum = grp["label"].transform("sum")
        label_count_nonnull = grp["label"].transform(lambda s: s.notna().sum())
        hour_sum = grp["hour"].transform("sum")
        hour_count_nonnull = grp["hour"].transform(lambda s: s.notna().sum())

        own_label = work["label"]
        own_hour = work["hour"]

        loo_count = (user_count - 1).clip(lower=0)
        out["user_prev_post_count"] = loo_count.astype(np.float32)

        label_sum_loo = label_sum - own_label.fillna(0.0)
        label_count_loo = (label_count_nonnull - own_label.notna().astype(int)).clip(lower=0)
        out["user_mean_label"] = np.where(
            label_count_loo > 0,
            label_sum_loo / label_count_loo.replace(0, np.nan),
            np.nan,
        )

        hour_sum_loo = hour_sum - own_hour.fillna(0.0)
        hour_count_loo = (hour_count_nonnull - own_hour.notna().astype(int)).clip(lower=0)
        out["user_active_hour_mean"] = np.where(
            hour_count_loo > 0,
            hour_sum_loo / hour_count_loo.replace(0, np.nan),
            np.nan,
        )

        median_full = grp["label"].transform("median")
        std_full = grp["label"].transform("std")
        cat_nunique = (
            grp["category"].transform(lambda s: s.dropna().nunique())
            if "category" in work.columns
            else np.nan
        )
        out["user_median_label"] = median_full.astype(np.float32)
        out["user_std_label"] = std_full.astype(np.float32)
        out["user_category_nunique"] = pd.to_numeric(cat_nunique, errors="coerce").astype(np.float32)

        return out

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

    agg_cols = [c for c in agg.columns if c != "Uid"]
    out = target_df.drop(
        columns=[c for c in agg_cols if c in target_df.columns],
        errors="ignore",
    )
    return out.merge(agg, on="Uid", how="left")


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
    for col in required_cols:
        if col not in out.columns:
            out[col] = None
    return out


_LOCATION_MISSING_VALUES = {"", "nan", "none", "null", "unknown", "unk", "0", "0.0"}


def _clean_location_component(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    s = str(x).strip()
    if not s:
        return ""
    s = " ".join(s.split())
    if s.lower() in _LOCATION_MISSING_VALUES:
        return ""
    alias = {
        "usa": "United States",
        "u.s.a.": "United States",
        "u.s.a": "United States",
        "us": "United States",
        "u.s.": "United States",
        "united states of america": "United States",
        "uk": "United Kingdom",
        "u.k.": "United Kingdom",
        "brasil": "Brazil",
    }
    return alias.get(s.lower(), s)


def _is_missing_location_value(x: Any) -> bool:
    return _clean_location_component(x) == ""


def _parse_location_text_value(x: Any) -> tuple[str, str, str]:
    s = _clean_location_component(x)
    if not s:
        return "", "", ""

    parts = [_clean_location_component(p) for p in s.split(",")]
    parts = [p for p in parts if p]
    if not parts:
        return "", "", ""

    if len(parts) == 1:
        return "", "", parts[0]
    if len(parts) == 2:
        return parts[0], "", parts[1]

    city = parts[0]
    state = ", ".join(parts[1:-1])
    country = parts[-1]
    return city, state, country


def enrich_location_from_text(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "location_text" not in out.columns:
        out["location_text"] = ""
    for col in ["city", "state", "country"]:
        if col not in out.columns:
            out[col] = ""

    parsed = out["location_text"].map(_parse_location_text_value)
    parsed_city = parsed.map(lambda t: t[0])
    parsed_state = parsed.map(lambda t: t[1])
    parsed_country = parsed.map(lambda t: t[2])

    for col, parsed_col in [
        ("city", parsed_city),
        ("state", parsed_state),
        ("country", parsed_country),
    ]:
        missing = out[col].map(_is_missing_location_value)
        out.loc[missing, col] = parsed_col[missing]
        out[col] = out[col].map(_clean_location_component)

    for col in ["location_text", "city", "state", "country"]:
        non_empty = int(out[col].fillna("").astype(str).str.strip().ne("").sum())
        print(f"[GEO] {col}: non_empty={non_empty}/{len(out)}")

    return out


def add_geo_encoding_features(
    train_df: pd.DataFrame,
    target_df: pd.DataFrame,
    smoothing: float = 30.0,
) -> pd.DataFrame:
    out = target_df.copy()
    if "label" not in train_df.columns:
        raise ValueError("add_geo_encoding_features requires train_df['label'].")

    train_label = pd.to_numeric(train_df["label"], errors="coerce")
    global_mean = float(train_label.mean()) if train_label.notna().any() else 0.0
    smoothing = float(smoothing)

    is_train_self = (target_df is train_df) or (
        len(target_df) == len(train_df)
        and "post_id" in target_df.columns
        and "post_id" in train_df.columns
        and target_df["post_id"].equals(train_df["post_id"])
    )

    for col in ["country", "state", "city", "location_text"]:
        enc_col = f"{col}_target_enc"
        freq_col = f"{col}_freq"

        if col not in train_df.columns or col not in out.columns:
            out[enc_col] = np.float32(global_mean)
            out[freq_col] = np.float32(0.0)
            continue

        work = train_df[[col]].copy()
        work["_label"] = train_label

        freq = work[col].value_counts(dropna=True).rename(freq_col)
        out[freq_col] = np.log1p(out[col].map(freq).fillna(0)).astype(np.float32)

        if is_train_self:
            grp_sum = work.groupby(col, dropna=True)["_label"].transform("sum")
            grp_count = work.groupby(col, dropna=True)["_label"].transform("count")

            own_label = train_label.fillna(0.0)
            own_count = train_label.notna().astype(int)

            loo_sum = grp_sum - own_label
            loo_count = (grp_count - own_count).clip(lower=0)
            loo_enc = (loo_sum + global_mean * smoothing) / (loo_count + smoothing)
            out[enc_col] = pd.to_numeric(loo_enc, errors="coerce").fillna(global_mean).astype(np.float32)
        else:
            stats = work.groupby(col, dropna=True)["_label"].agg(["sum", "count"])
            smoothed = (stats["sum"] + global_mean * smoothing) / (stats["count"] + smoothing)
            out[enc_col] = out[col].map(smoothed).fillna(global_mean).astype(np.float32)

    return out
