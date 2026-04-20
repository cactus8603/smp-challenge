from __future__ import annotations

import argparse
import json
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

# Allow scripts/ to import from src/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.datasets.metadata_preprocessor import MetadataPreprocessor
from src.datasets.smp_dataset import SMPDataset, smp_collate_fn
from src.engine.trainer import Trainer
from src.models.fusion_model import SMPFusionModel
from src.utils.criterion import HybridLoss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Which fold to train. Example: 0 ~ n_folds-1",
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        default=5,
        help="Number of GroupKFold splits.",
    )
    return parser.parse_args()


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
        base_path = (config_path.parent / base_key).resolve()

    base_cfg = load_config_with_base(base_path)
    return deep_merge_dict(base_cfg, cfg)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loss(loss_name: str):
    name = loss_name.lower()

    if name == "mse":
        return torch.nn.MSELoss()
    elif name in {"mae", "l1"}:
        return torch.nn.L1Loss()
    elif name == "smoothl1":
        return torch.nn.SmoothL1Loss()
    elif name == "ranking":
        return HybridLoss(
            alpha=1.0,
            beta=0.3,
            margin=0.0,
            min_target_diff=0.1,
            weight_by_target_diff=True,
            max_weight=3.0,
        )
    elif name == "hybrid":
        # return HybridLoss(
        #     alpha=1.0,
        #     beta=0.3,
        #     gamma=0.1,
        #     margin=0.0,
        #     min_target_diff=0.1,
        #     weight_by_target_diff=True,
        #     max_weight=5.0,
        #     high_target_scale=0.5,
        #     reg_max_weight=3.0,
        # )
        return HybridLoss(
            alpha=1.0,
            beta=0.45,
            gamma=0.15,
            delta=0.10,
            eta=0.08,
            zeta=0.20,

            hard_scale=1.2,
            high_target_scale=0.25,
            reg_max_weight=4.0,

            min_target_diff=0.15,
            rank_max_weight=4.0,

            variance_floor_ratio=0.60,
            focal_gamma=1.5,
        )
    else:
        raise ValueError(f"Unsupported loss: {loss_name}")


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


def add_user_aggregate_features_fold(train_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build label-derived user aggregate features using ONLY the current fold train_df,
    then apply them to target_df. This avoids KFold leakage.
    """
    if train_df.empty or "Uid" not in train_df.columns or "Uid" not in target_df.columns:
        return target_df

    work = train_df.copy()
    work["label"] = pd.to_numeric(work["label"], errors="coerce")
    work["hour"] = pd.to_numeric(work["hour"] if "hour" in work.columns else None, errors="coerce")

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

    out = target_df.merge(agg, on="Uid", how="left")
    return out


def ensure_user_aggregate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make sure aggregate columns exist even if they are absent in official_train.
    This keeps schema stable for downstream preprocessing.
    """
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


def main() -> None:
    args = parse_args()
    cfg = load_config_with_base(Path(args.config).resolve())

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # paths / experiment dirs
    # -------------------------
    exp_name = str(cfg["exp_name"])
    output_root = Path(cfg["output"]["root_dir"])

    fold_name = f"fold_{args.fold}"
    exp_dir = output_root / exp_name / fold_name
    ckpt_dir = exp_dir / "checkpoints"
    tb_dir = exp_dir / "tensorboard"
    exp_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    resolved_cfg = deepcopy(cfg)
    resolved_cfg["runtime"] = {
        "fold": args.fold,
        "n_folds": args.n_folds,
    }

    with (exp_dir / "resolved_config.json").open("w", encoding="utf-8") as f:
        json.dump(resolved_cfg, f, ensure_ascii=False, indent=2)

    # -------------------------
    # config shortcuts
    # -------------------------
    model_cfg = cfg["model"]
    text_cfg = cfg["text"]
    image_cfg = cfg["image"]
    meta_cfg = cfg["meta"]
    train_cfg = cfg["train"]
    preprocess_cfg = cfg["preprocess"]
    fusion_cfg = cfg["fusion"]
    loss_cfg = cfg["loss"]
    data_cfg = cfg["data"]

    use_text = bool(model_cfg["use_text"])
    use_meta = bool(model_cfg["use_meta"])
    use_image = bool(model_cfg["use_image"])

    text_model_name = text_cfg["model_name"]
    image_model_name = image_cfg["model_name"]
    image_path_col = image_cfg.get("path_col", "image_path")

    batch_size = int(train_cfg["batch_size"])
    num_workers = int(train_cfg["num_workers"])
    pin_memory = bool(train_cfg.get("pin_memory", True))
    persistent_workers = bool(train_cfg.get("persistent_workers", num_workers > 0))
    drop_last = bool(train_cfg.get("drop_last", False))

    # -------------------------
    # data loading: KFold from official_train only
    # -------------------------
    official_train_path = data_cfg.get("official_train_path")
    if official_train_path is None:
        raise ValueError(
            "For KFold training, config.data.official_train_path is required."
        )

    official_train_df = load_dataframe(official_train_path)

    if "Uid" not in official_train_df.columns:
        raise ValueError("official_train dataframe must contain 'Uid' for GroupKFold.")

    # make sure missing aggregate cols exist; they will be recomputed after fold split
    official_train_df = ensure_user_aggregate_columns(official_train_df)

    train_df, val_df = make_fold_split(
        official_train_df,
        fold=args.fold,
        n_folds=args.n_folds,
        group_col="Uid",
    )

    # ---------------------------------------------------------
    # IMPORTANT: recompute label-derived user aggregates
    # using ONLY fold-train, then apply to train/val
    # ---------------------------------------------------------
    train_df = add_user_aggregate_features_fold(train_df, train_df)
    val_df = add_user_aggregate_features_fold(train_df, val_df)

    print(f"[INFO] Fold {args.fold}/{args.n_folds}")
    print(f"[INFO] Train rows: {len(train_df)} | Val rows: {len(val_df)}")
    print(f"[INFO] Train users: {train_df['Uid'].nunique()} | Val users: {val_df['Uid'].nunique()}")

    label_mean = train_df["label"].mean()
    label_std = train_df["label"].std()

    if use_image:
        if image_path_col not in train_df.columns:
            raise ValueError(f"train_df missing image path column: {image_path_col}")
        if image_path_col not in val_df.columns:
            raise ValueError(f"val_df missing image path column: {image_path_col}")

    # -------------------------
    # preprocessor: fit on fold-train only
    # -------------------------
    preprocessor = MetadataPreprocessor(
        num_cols=preprocess_cfg.get("num_cols"),
        cat_cols=preprocess_cfg.get("cat_cols"),
        bin_cols=preprocess_cfg.get("bin_cols"),
        log1p_cols=preprocess_cfg.get("log1p_cols"),
    )

    train_df = preprocessor.fit_transform(train_df)
    val_df = preprocessor.transform(val_df)

    preprocessor.save(exp_dir / "metadata_preprocessor.json")

    image_root_dir = image_cfg.get("root_dir", None)

    # -------------------------
    # datasets
    # -------------------------
    train_dataset = SMPDataset(
        df=train_df,
        preprocessor=preprocessor,
        text_model_name=text_model_name,
        image_model_name=image_model_name,
        normalize_label=True,
        label_mean=label_mean,
        label_std=label_std,
        max_length=int(text_cfg["max_length"]),
        use_text=use_text,
        use_meta=use_meta,
        use_image=use_image,
        image_path_col=image_path_col,
        image_root_dir=image_root_dir,
        is_train=True,
    )

    val_dataset = SMPDataset(
        df=val_df,
        preprocessor=preprocessor,
        text_model_name=text_model_name,
        image_model_name=image_model_name,
        normalize_label=True,
        label_mean=label_mean,
        label_std=label_std,
        max_length=int(text_cfg["max_length"]),
        use_text=use_text,
        use_meta=use_meta,
        use_image=use_image,
        image_path_col=image_path_col,
        image_root_dir=image_root_dir,
        is_train=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=smp_collate_fn,
        drop_last=drop_last,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=smp_collate_fn,
        drop_last=False,
    )

    # -------------------------
    # model
    # -------------------------
    cat_cardinalities = [
        int(preprocessor.cat_cardinalities[col])
        for col in preprocessor.cat_cols
    ]

    model = SMPFusionModel(
        text_model_name=text_model_name,
        meta_num_dim=len(preprocessor.transformed_num_cols),
        meta_cat_cardinalities=cat_cardinalities,
        meta_bin_dim=len(preprocessor.transformed_bin_cols),
        hidden_dim=int(model_cfg["hidden_dim"]),
        dropout=float(model_cfg["dropout"]),
        use_text=use_text,
        use_meta=use_meta,
        use_image=use_image,
        image_model_name=image_model_name,
        text_pooling=text_cfg["pooling"],
        text_trainable=bool(text_cfg["trainable"]),
        image_pretrained=bool(image_cfg.get("pretrained", True)),
        image_trainable=bool(image_cfg["trainable"]),
        fusion_type=fusion_cfg["type"],
        meta_branch_dim=int(meta_cfg["branch_dim"]),
    ).to(device)

    # -------------------------
    # optimization
    # -------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )

    criterion = build_loss(loss_cfg["name"])

    total_steps = len(train_loader) * int(train_cfg["epochs"])
    warmup_steps = int(total_steps * train_cfg["warmup_ratio"])

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=str(device),
        exp_name=f"{exp_name}_{fold_name}",
        exp_dir=exp_dir,
        ckpt_dir=ckpt_dir,
        tb_dir=tb_dir,
        grad_clip_norm=train_cfg.get("grad_clip_norm"),
    )

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=int(train_cfg["epochs"]),
    )


if __name__ == "__main__":
    main()

# Example:
# python3 scripts/train.py --config configs/text_meta_image_v2.yaml --fold 0 --n_folds 5
# python3 scripts/train.py --config configs/text_meta_image_v2.yaml --fold 1 --n_folds 5
# nohup python3 scripts/train.py --config configs/text_meta_image_v2.yaml --fold 1 --n_folds 5 > kfold.log 2>&1 &
