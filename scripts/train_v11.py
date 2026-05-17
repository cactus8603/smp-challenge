#!/usr/bin/env python3
"""
train_v11.py

Frozen encoder stacking: take pre-trained v8 (metadata) and v9 (text+image)
encoders, extract their intermediate representations, concatenate, then train
a small MLP fusion head on top.

Concatenation:
    [v8.meta_repr (256) | v9.text_repr (256) | v9.image_repr (256)] = 768-dim

Usage:
    CUDA_VISIBLE_DEVICES=6 python3 scripts/train_v11.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import spearmanr
from tqdm import tqdm

from scripts.train import (
    load_config_with_base,
    make_fold_split,
    add_user_aggregate_features_fold,
    add_geo_encoding_features,
    ensure_user_aggregate_columns,
    load_dataframe,
)
from src.datasets.smp_dataset import SMPDataset, smp_collate_fn
from src.models.fusion_model import SMPFusionModel
from src.datasets.metadata_preprocessor_v2 import MetadataPreprocessorV2 as MetadataPreprocessor


# ── config ────────────────────────────────────────────────────────────────────
FOLD       = 0
N_FOLDS    = 5
DEVICE     = "cuda:0"   # mapped via CUDA_VISIBLE_DEVICES=6

V8_CONFIG  = "configs/test_v8.yaml"
V9_CONFIG  = "configs/test_v9.yaml"
V8_CKPT    = "outputs/test_v8/fold_0/checkpoints/best.pt"
V9_CKPT    = "outputs/test_v9/fold_0/checkpoints/best.pt"

EXTRACT_BATCH = 256
NUM_WORKERS   = 4

MLP_EPOCHS    = 60
MLP_LR        = 1e-3
MLP_BATCH     = 2048
MLP_DROPOUT   = 0.2
MLP_HIDDEN    = [512, 256]
WEIGHT_DECAY  = 1e-3

OUT_DIR = Path("outputs/test_v11/fold_0")
# ─────────────────────────────────────────────────────────────────────────────


def build_datasets(config_path: str, fold: int, n_folds: int):
    cfg = load_config_with_base(Path(config_path).resolve())
    data_cfg      = cfg.get("data", {})
    text_cfg      = cfg.get("text", {})
    image_cfg     = cfg.get("image", {})
    preprocess_cfg = cfg.get("preprocess", {})
    model_cfg     = cfg.get("model", {})

    parquet_path = data_cfg["official_train_path"]
    full_df = load_dataframe(parquet_path)
    full_df = ensure_user_aggregate_columns(full_df)

    train_df, val_df = make_fold_split(full_df, fold=fold, n_folds=n_folds)
    train_df = add_user_aggregate_features_fold(train_df, train_df)
    val_df   = add_user_aggregate_features_fold(train_df, val_df)
    train_df = add_geo_encoding_features(train_df, train_df)
    val_df   = add_geo_encoding_features(train_df, val_df)

    preprocessor = MetadataPreprocessor(
        num_cols=preprocess_cfg.get("num_cols"),
        cat_cols=preprocess_cfg.get("cat_cols"),
        bin_cols=preprocess_cfg.get("bin_cols"),
        log1p_cols=preprocess_cfg.get("log1p_cols"),
    )
    train_df = preprocessor.fit_transform(train_df)
    val_df   = preprocessor.transform(val_df)

    label_mean = float(train_df["label"].mean())
    label_std  = float(train_df["label"].std())

    shared_kwargs = dict(
        preprocessor=preprocessor,
        text_model_name=text_cfg.get("model_name", "openai/clip-vit-base-patch32"),
        image_model_name=image_cfg.get("model_name", "openai/clip-vit-base-patch32"),
        user_desc_emb_path=data_cfg.get("user_desc_emb_path"),
        user_desc_idx_path=data_cfg.get("user_desc_idx_path"),
        loc_desc_emb_path=data_cfg.get("loc_desc_emb_path"),
        loc_desc_idx_path=data_cfg.get("loc_desc_idx_path"),
        caption_max_freq=data_cfg.get("caption_max_freq", 50),
        use_caption=bool(data_cfg.get("use_caption", True)),
        normalize_label=True,
        label_mean=label_mean,
        label_std=label_std,
        max_length=int(text_cfg.get("max_length", 77)),
        use_text=bool(model_cfg.get("use_text", True)),
        use_meta=bool(model_cfg.get("use_meta", True)),
        use_image=bool(model_cfg.get("use_image", True)),
        image_path_col=image_cfg.get("path_col", "image_path"),
        image_root_dir=image_cfg.get("root_dir", None),
    )

    train_ds = SMPDataset(df=train_df, is_train=False, **shared_kwargs)
    val_ds   = SMPDataset(df=val_df,   is_train=False, **shared_kwargs)

    return train_ds, val_ds, preprocessor, label_mean, label_std


def build_model(config_path: str, checkpoint_path: str,
                preprocessor, train_ds, device: str) -> SMPFusionModel:
    cfg = load_config_with_base(Path(config_path).resolve())
    model_cfg  = cfg.get("model", {})
    text_cfg   = cfg.get("text", {})
    image_cfg  = cfg.get("image", {})
    meta_cfg   = cfg.get("meta", {})
    fusion_cfg = cfg.get("fusion", {})

    cat_cardinalities = [
        int(preprocessor.cat_cardinalities[col])
        for col in preprocessor.cat_cols
    ]

    model = SMPFusionModel(
        text_model_name=text_cfg.get("model_name", "openai/clip-vit-base-patch32"),
        meta_num_dim=len(preprocessor.transformed_num_cols),
        meta_cat_cardinalities=cat_cardinalities,
        meta_bin_dim=len(preprocessor.transformed_bin_cols),
        hidden_dim=int(model_cfg.get("hidden_dim", 256)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        use_text=bool(model_cfg.get("use_text", True)),
        use_meta=bool(model_cfg.get("use_meta", True)),
        use_image=bool(model_cfg.get("use_image", True)),
        image_model_name=image_cfg.get("model_name", "openai/clip-vit-base-patch32"),
        text_pooling=text_cfg.get("pooling", "clip"),
        text_trainable=bool(text_cfg.get("trainable", False)),
        image_pretrained=bool(image_cfg.get("pretrained", True)),
        image_trainable=bool(image_cfg.get("trainable", False)),
        fusion_type=fusion_cfg.get("type", "concat"),
        use_clip_similarity=bool(fusion_cfg.get("use_clip_similarity", False)),
        meta_branch_dim=int(meta_cfg.get("branch_dim", 128)),
        use_user_desc=bool(meta_cfg.get("use_user_desc", True)),
        user_desc_dim=int(meta_cfg.get("user_desc_dim", train_ds.user_desc_emb_dim or 768)),
        use_loc_desc=bool(meta_cfg.get("use_loc_desc", False)),
        loc_desc_dim=int(meta_cfg.get("loc_desc_dim", train_ds.loc_desc_emb_dim or 400)),
        desc_bottleneck_dim=int(meta_cfg.get("desc_bottleneck_dim", 64)),
        user_desc_scale=float(meta_cfg.get("user_desc_scale", 0.10)),
        loc_desc_scale=float(meta_cfg.get("loc_desc_scale", 0.05)),
        feature_gate_config=preprocessor.feature_gate_config,
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    return model


@torch.no_grad()
def extract_features(model: SMPFusionModel, dataset: SMPDataset,
                     device: str, keys: list[str]) -> dict[str, np.ndarray]:
    loader = DataLoader(
        dataset, batch_size=EXTRACT_BATCH, shuffle=False,
        num_workers=NUM_WORKERS, collate_fn=smp_collate_fn, pin_memory=True,
    )

    buffers = {k: [] for k in keys}
    labels  = []

    for batch in tqdm(loader, desc="Extracting", leave=False):
        result = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            meta_num=batch["meta_num"].to(device),
            meta_cat=batch["meta_cat"].to(device),
            meta_bin=batch["meta_bin"].to(device),
            image_tensor=(
                batch["image_tensor"].to(device)
                if batch["image_tensor"].numel() > 0 else None
            ),
            user_desc=(
                batch["user_desc"].to(device)
                if "user_desc" in batch else None
            ),
            return_features=True,
        )
        feats = result["features"]
        for k in keys:
            if feats.get(k) is not None:
                buffers[k].append(feats[k].cpu().float().numpy())

        labels.append(batch["labels"].numpy())

    return (
        {k: np.concatenate(v, axis=0) for k, v in buffers.items()},
        np.concatenate(labels, axis=0),
    )


class FusionMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: list[int], dropout: float):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(X_train, y_train, X_val, y_val,
              label_mean: float, label_std: float, device: str):
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
    y_val_t   = torch.tensor(y_val,   dtype=torch.float32)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_dl = DataLoader(train_ds, batch_size=MLP_BATCH, shuffle=True)

    model = FusionMLP(X_train.shape[1], MLP_HIDDEN, MLP_DROPOUT).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=MLP_LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    best_sp   = -1.0
    best_epoch = 0
    best_state = None

    for epoch in range(1, MLP_EPOCHS + 1):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            preds_norm = model(X_val_t.to(device)).cpu().numpy()

        preds_raw  = preds_norm  * label_std + label_mean
        labels_raw = y_val_t.numpy() * label_std + label_mean
        sp = spearmanr(labels_raw, preds_raw).correlation

        marker = ""
        if sp > best_sp:
            best_sp    = sp
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            marker = " ← best"

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{MLP_EPOCHS}  val_spearman={sp:.4f}{marker}")

    print(f"\n[RESULT] Best epoch={best_epoch}, val_spearman={best_sp:.4f}")
    model.load_state_dict(best_state)
    return model, best_sp


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("test_v11: Frozen encoder stacking (v8 meta + v9 text+image)")
    print("=" * 60)

    # ── Phase 1: load datasets and models ────────────────────────
    print("\n[1/4] Building datasets and loading encoders...")

    train_ds_v8, val_ds_v8, pre_v8, lm8, ls8 = build_datasets(V8_CONFIG, FOLD, N_FOLDS)
    model_v8 = build_model(V8_CONFIG, V8_CKPT, pre_v8, train_ds_v8, DEVICE)
    print(f"  v8 loaded: meta_num={len(pre_v8.transformed_num_cols)}")

    train_ds_v9, val_ds_v9, pre_v9, lm9, ls9 = build_datasets(V9_CONFIG, FOLD, N_FOLDS)
    model_v9 = build_model(V9_CONFIG, V9_CKPT, pre_v9, train_ds_v9, DEVICE)
    print(f"  v9 loaded: meta_num={len(pre_v9.transformed_num_cols)}")

    # ── Phase 2: extract embeddings ───────────────────────────────
    print("\n[2/4] Extracting embeddings from v8 (meta branch)...")
    v8_train_feats, y_train_v8 = extract_features(model_v8, train_ds_v8, DEVICE, ["meta"])
    v8_val_feats,   y_val_v8   = extract_features(model_v8, val_ds_v8,   DEVICE, ["meta"])

    del model_v8
    torch.cuda.empty_cache()

    print("[2/4] Extracting embeddings from v9 (text + image branches)...")
    v9_train_feats, y_train = extract_features(model_v9, train_ds_v9, DEVICE, ["text", "image"])
    v9_val_feats,   y_val   = extract_features(model_v9, val_ds_v9,   DEVICE, ["text", "image"])

    del model_v9
    torch.cuda.empty_cache()

    # ── Phase 3: concatenate ──────────────────────────────────────
    print("\n[3/4] Concatenating: [v8.meta | v9.text | v9.image]")
    X_train = np.concatenate([
        v8_train_feats["meta"],   # 256-dim
        v9_train_feats["text"],   # 256-dim
        v9_train_feats["image"],  # 256-dim
    ], axis=1)

    X_val = np.concatenate([
        v8_val_feats["meta"],
        v9_val_feats["text"],
        v9_val_feats["image"],
    ], axis=1)

    print(f"  X_train shape: {X_train.shape}  X_val shape: {X_val.shape}")

    assert len(y_train_v8) == len(y_train), "Train label length mismatch"
    assert len(y_val_v8)   == len(y_val),   "Val label length mismatch"

    # use v9's label normalization (same fold, same split)
    label_mean, label_std = lm9, ls9

    # ── Phase 4: train fusion MLP ─────────────────────────────────
    print("\n[4/4] Training fusion MLP head...")
    mlp, best_sp = train_mlp(X_train, y_train, X_val, y_val, label_mean, label_std, DEVICE)

    # ── Save summary ──────────────────────────────────────────────
    summary = {
        "exp_name": "test_v11",
        "description": "Frozen encoder stacking: v8 meta + v9 text+image",
        "concat_dims": {"v8_meta": 256, "v9_text": 256, "v9_image": 256, "total": 768},
        "mlp_hidden": MLP_HIDDEN,
        "mlp_epochs": MLP_EPOCHS,
        "best_val_spearman": best_sp,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    torch.save(mlp.state_dict(), OUT_DIR / "mlp_head.pt")

    print("\n" + "=" * 60)
    print(f"[DONE] test_v11 best val_spearman = {best_sp:.4f}")
    print(f"       Comparison: v8={0.5104:.4f} | v9={0.5975:.4f} | ensemble={0.6156:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
