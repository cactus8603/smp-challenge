from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

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
from src.models.fusion_model import SMPFusionModel
from src.utils.user_desc_embeddings import maybe_prepare_user_desc_embeddings


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


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
        base_path = (config_path.parent / base_key).resolve()
    return deep_merge_dict(load_config_with_base(base_path), cfg)


def load_dataframe(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    if p.suffix == ".csv":
        return pd.read_csv(p)
    if p.suffix == ".jsonl":
        return pd.read_json(p, lines=True)
    raise ValueError(f"Unsupported format: {p.suffix}")


def make_fold_split(df: pd.DataFrame, fold: int, n_folds: int, group_col: str = "Uid"):
    groups = df[group_col].fillna("UNK_GROUP").astype(str).to_numpy()
    gkf = GroupKFold(n_splits=n_folds)
    splits = list(gkf.split(df, groups=groups))
    tr_idx, va_idx = splits[fold]
    train_df = df.iloc[tr_idx].reset_index(drop=True).copy()
    val_df = df.iloc[va_idx].reset_index(drop=True).copy()
    train_df["split"] = "train"
    val_df["split"] = "val"
    return train_df, val_df


def _safe_nunique(series: pd.Series) -> int:
    return int(series.dropna().nunique())


def add_user_aggregate_features_fold(train_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build label-derived user aggregate features using ONLY the current fold train_df,
    then apply them to target_df. Drops stale placeholder columns before merge.
    """
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

    agg_cols = [c for c in agg.columns if c != "Uid"]
    out = target_df.drop(
        columns=[c for c in agg_cols if c in target_df.columns],
        errors="ignore",
    )
    return out.merge(agg, on="Uid", how="left")


def ensure_user_aggregate_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in [
        "user_prev_post_count",
        "user_mean_label",
        "user_median_label",
        "user_std_label",
        "user_category_nunique",
        "user_active_hour_mean",
    ]:
        if c not in out.columns:
            out[c] = None
    return out






# ---------------------------------------------------------------------
# Location text parsing
# ---------------------------------------------------------------------
_LOCATION_MISSING_VALUES = {"", "nan", "none", "null", "unknown", "unk", "0", "0.0"}


def _clean_location_component(x: Any) -> str:
    """Normalize one location component from raw location_text/city/state/country."""
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
    # Light canonicalization for common country aliases.
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
    """
    Heuristic parser for Flickr-style location_text.

    Examples:
      "London, United Kingdom" -> city=London, country=United Kingdom
      "Buffalo, New York, United States of America" -> city=Buffalo, state=New York, country=United States
      "France" -> country=France
    """
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
    """
    Fill empty city/state/country from location_text before geo features are built.

    The processed parquet currently has useful location_text but empty city/state/country.
    This function makes those fields usable for categorical features and fold-safe
    target/frequency encoding. Existing non-empty city/state/country values are kept.
    """
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

    # Compact debug summary; useful in logs/check script.
    for col in ["location_text", "city", "state", "country"]:
        non_empty = int(out[col].fillna("").astype(str).str.strip().ne("").sum())
        print(f"[GEO] {col}: non_empty={non_empty}/{len(out)}")

    return out


def add_geo_encoding_features(train_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    """Same fold-safe geo features as train.py, computed from fold-train only."""
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

def build_model(cfg, preprocessor, device):
    model_cfg = cfg["model"]
    text_cfg = cfg["text"]
    image_cfg = cfg["image"]
    meta_cfg = cfg["meta"]
    fusion_cfg = cfg["fusion"]

    cat_cardinalities = [
        int(preprocessor.cat_cardinalities[col])
        for col in preprocessor.cat_cols
    ]

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
        meta_branch_dim=int(meta_cfg["branch_dim"]),
        use_clip_similarity=bool(fusion_cfg.get("use_clip_similarity", False)),
        use_user_desc=bool(meta_cfg.get("use_user_desc", True)),
        user_desc_dim=int(meta_cfg.get("user_desc_dim", 768)),
        use_loc_desc=bool(meta_cfg.get("use_loc_desc", False)),
        loc_desc_dim=int(meta_cfg.get("loc_desc_dim", 400)),
        desc_bottleneck_dim=int(meta_cfg.get("desc_bottleneck_dim", 64)),
        user_desc_scale=float(meta_cfg.get("user_desc_scale", 0.10)),
        loc_desc_scale=float(meta_cfg.get("loc_desc_scale", 0.05)),
    ).to(device)

    return model


@torch.no_grad()
def export_loader(model, loader, device, label_mean, label_std) -> pd.DataFrame:
    model.eval()
    rows: List[pd.DataFrame] = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        meta_num = batch["meta_num"].to(device)
        meta_cat = batch["meta_cat"].to(device)
        meta_bin = batch["meta_bin"].to(device)

        image_tensor = batch.get("image_tensor", None)
        if image_tensor is not None and image_tensor.numel() > 0:
            image_tensor = image_tensor.to(device)
        else:
            image_tensor = None

        glove_token_count = batch.get("glove_token_count", None)
        if glove_token_count is not None:
            glove_token_count = glove_token_count.to(device)

        user_desc = batch.get("user_desc", None)
        if user_desc is not None:
            user_desc = user_desc.to(device)
        loc_desc = batch.get("loc_desc", None)
        if loc_desc is not None:
            loc_desc = loc_desc.to(device)

        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            meta_num=meta_num,
            meta_cat=meta_cat,
            meta_bin=meta_bin,
            image_tensor=image_tensor,
            glove_tokens=batch.get("glove_tokens", None),
            glove_text=batch.get("glove_text", None),
            glove_token_count=glove_token_count,
            user_desc=user_desc,
            loc_desc=loc_desc,
            return_features=True,
        )

        pred_norm = out["output"].squeeze(-1).detach().cpu().numpy()
        pred_raw = pred_norm * label_std + label_mean

        fused = out["fused"].detach().cpu().numpy()
        text = out["features"]["text"].detach().cpu().numpy()
        meta = out["features"]["meta"].detach().cpu().numpy()
        image = out["features"]["image"].detach().cpu().numpy()

        data = {
            "post_id": batch["post_id"],
            "Uid": batch.get("uid", batch.get("Uid")),
            "Pid": batch.get("pid", batch.get("Pid")),
            "label": batch["labels"].detach().cpu().numpy() * label_std + label_mean,
            "deep_pred": pred_raw,
            "deep_pred_norm": pred_norm,
        }

        clip_sim = out["features"].get("clip_sim")
        if clip_sim is not None:
            data["clip_sim"] = clip_sim.squeeze(-1).detach().cpu().numpy()

        feature_cols: Dict[str, np.ndarray] = {}
        for name, arr in [("fused", fused), ("text", text), ("meta", meta), ("image", image)]:
            for i in range(arr.shape[1]):
                feature_cols[f"{name}_{i}"] = arr[:, i]

        df = pd.concat(
            [pd.DataFrame(data), pd.DataFrame(feature_cols)],
            axis=1,
        )
        rows.append(df)

    return pd.concat(rows, axis=0, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config_with_base(Path(args.config).resolve())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg = cfg["data"]
    text_cfg = cfg["text"]
    image_cfg = cfg["image"]
    train_cfg = cfg["train"]
    model_cfg = cfg["model"]

    df = load_dataframe(data_cfg["official_train_path"])

    user_desc_emb_path, user_desc_idx_path = maybe_prepare_user_desc_embeddings(
        cfg=cfg,
        df=df,
        project_root=ROOT,
    )
    data_cfg = cfg["data"]

    df = ensure_user_aggregate_columns(df)

    train_df, val_df = make_fold_split(
        df,
        fold=args.fold,
        n_folds=args.n_folds,
        group_col="Uid",
    )

    # Fill empty city/state/country from location_text before geo encoding.
    train_df = enrich_location_from_text(train_df)
    val_df = enrich_location_from_text(val_df)

    train_df = add_user_aggregate_features_fold(train_df, train_df)
    val_df = add_user_aggregate_features_fold(train_df, val_df)

    train_df = add_geo_encoding_features(train_df, train_df)
    val_df = add_geo_encoding_features(train_df, val_df)

    label_mean = train_df["label"].mean()
    label_std = train_df["label"].std()

    exp_name = str(cfg["exp_name"])
    exp_dir = Path(cfg["output"]["root_dir"]) / exp_name / f"fold_{args.fold}"
    preprocessor = MetadataPreprocessor.load(exp_dir / "metadata_preprocessor.json")

    val_df = preprocessor.transform(val_df)

    dataset = SMPDataset(
        df=val_df,
        preprocessor=preprocessor,
        text_model_name=text_cfg["model_name"],
        image_model_name=image_cfg["model_name"],
        normalize_label=True,
        label_mean=label_mean,
        label_std=label_std,
        max_length=int(text_cfg["max_length"]),
        use_text=bool(model_cfg["use_text"]),
        use_meta=bool(model_cfg["use_meta"]),
        use_image=bool(model_cfg["use_image"]),
        image_path_col=image_cfg.get("path_col", "image_path"),
        image_root_dir=image_cfg.get("root_dir", None),
        user_desc_emb_path=user_desc_emb_path,
        user_desc_idx_path=user_desc_idx_path,
        loc_desc_emb_path=data_cfg.get("loc_desc_emb_path"),
        loc_desc_idx_path=data_cfg.get("loc_desc_idx_path"),
        is_train=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(train_cfg["num_workers"]),
        pin_memory=bool(train_cfg.get("pin_memory", True)),
        collate_fn=smp_collate_fn,
        drop_last=False,
    )

    model = build_model(cfg, preprocessor, device)

    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)

    out_df = export_loader(model, loader, device, label_mean, label_std)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    print(f"[SAVED] {output_csv} shape={out_df.shape}")


if __name__ == "__main__":
    main()