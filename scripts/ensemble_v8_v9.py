#!/usr/bin/env python3
"""
ensemble_v8_v9.py

Run inference with best checkpoints from test_v8 and test_v9 on the same
fold-0 val set, then search for the optimal blending weight alpha:
    final_pred = alpha * v9_pred + (1 - alpha) * v8_pred

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 scripts/ensemble_v8_v9.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.stats import spearmanr
from tqdm import tqdm
from pathlib import Path

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


FOLD = 0
N_FOLDS = 5
BATCH_SIZE = 256
NUM_WORKERS = 4
DEVICE = "cuda:0"


def load_model_and_val_dataset(config_path: str, checkpoint_path: str, device: str):
    cfg = load_config_with_base(Path(config_path).resolve())

    data_cfg     = cfg.get("data", {})
    model_cfg    = cfg.get("model", {})
    text_cfg     = cfg.get("text", {})
    image_cfg    = cfg.get("image", {})
    meta_cfg     = cfg.get("meta", {})
    fusion_cfg   = cfg.get("fusion", {})
    preprocess_cfg = cfg.get("preprocess", {})

    use_text  = bool(model_cfg.get("use_text", True))
    use_meta  = bool(model_cfg.get("use_meta", True))
    use_image = bool(model_cfg.get("use_image", True))

    text_model_name  = text_cfg.get("model_name", "openai/clip-vit-base-patch32")
    image_model_name = image_cfg.get("model_name", "openai/clip-vit-base-patch32")
    image_root_dir   = image_cfg.get("root_dir", None)
    image_path_col   = image_cfg.get("path_col", "image_path")

    parquet_path     = data_cfg["official_train_path"]
    caption_max_freq = data_cfg.get("caption_max_freq", 50)
    use_caption      = bool(data_cfg.get("use_caption", True))

    user_desc_emb_path = data_cfg.get("user_desc_emb_path", None)
    user_desc_idx_path = data_cfg.get("user_desc_idx_path", None)
    loc_desc_emb_path  = data_cfg.get("loc_desc_emb_path", None)
    loc_desc_idx_path  = data_cfg.get("loc_desc_idx_path", None)

    full_df = load_dataframe(parquet_path)
    full_df = ensure_user_aggregate_columns(full_df)

    train_df, val_df = make_fold_split(full_df, fold=FOLD, n_folds=N_FOLDS)

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

    val_dataset = SMPDataset(
        df=val_df,
        preprocessor=preprocessor,
        text_model_name=text_model_name,
        image_model_name=image_model_name,
        user_desc_emb_path=user_desc_emb_path,
        user_desc_idx_path=user_desc_idx_path,
        loc_desc_emb_path=loc_desc_emb_path,
        loc_desc_idx_path=loc_desc_idx_path,
        caption_max_freq=caption_max_freq,
        use_caption=use_caption,
        normalize_label=True,
        label_mean=label_mean,
        label_std=label_std,
        max_length=int(text_cfg.get("max_length", 77)),
        use_text=use_text,
        use_meta=use_meta,
        use_image=use_image,
        image_path_col=image_path_col,
        image_root_dir=image_root_dir,
        is_train=False,
    )

    cat_cardinalities = [
        int(preprocessor.cat_cardinalities[col])
        for col in preprocessor.cat_cols
    ]

    model = SMPFusionModel(
        text_model_name=text_model_name,
        meta_num_dim=len(preprocessor.transformed_num_cols),
        meta_cat_cardinalities=cat_cardinalities,
        meta_bin_dim=len(preprocessor.transformed_bin_cols),
        hidden_dim=int(model_cfg.get("hidden_dim", 256)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        use_text=use_text,
        use_meta=use_meta,
        use_image=use_image,
        image_model_name=image_model_name,
        text_pooling=text_cfg.get("pooling", "clip"),
        text_trainable=bool(text_cfg.get("trainable", False)),
        image_pretrained=bool(image_cfg.get("pretrained", True)),
        image_trainable=bool(image_cfg.get("trainable", False)),
        fusion_type=fusion_cfg.get("type", "pairwise_gated"),
        use_clip_similarity=bool(fusion_cfg.get("use_clip_similarity", True)),
        meta_branch_dim=int(meta_cfg.get("branch_dim", 128)),
        use_user_desc=bool(meta_cfg.get("use_user_desc", True)),
        user_desc_dim=int(meta_cfg.get("user_desc_dim", val_dataset.user_desc_emb_dim or 768)),
        use_loc_desc=bool(meta_cfg.get("use_loc_desc", False)),
        loc_desc_dim=int(meta_cfg.get("loc_desc_dim", val_dataset.loc_desc_emb_dim or 400)),
        desc_bottleneck_dim=int(meta_cfg.get("desc_bottleneck_dim", 64)),
        user_desc_scale=float(meta_cfg.get("user_desc_scale", 0.10)),
        loc_desc_scale=float(meta_cfg.get("loc_desc_scale", 0.05)),
        feature_gate_config=preprocessor.feature_gate_config,
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    return model, val_dataset, label_mean, label_std


@torch.no_grad()
def get_predictions(model, dataset, device, label_mean, label_std):
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=smp_collate_fn,
        pin_memory=True,
    )

    all_preds  = []
    all_labels = []

    for batch in tqdm(loader, desc="Inference", leave=False):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        meta_num       = batch["meta_num"].to(device)
        meta_cat       = batch["meta_cat"].to(device)
        meta_bin       = batch["meta_bin"].to(device)
        labels         = batch["labels"]

        image_tensor = batch.get("image_tensor", None)
        if image_tensor is not None and image_tensor.numel() > 0:
            image_tensor = image_tensor.to(device)
        else:
            image_tensor = None

        user_desc = batch.get("user_desc", None)
        if user_desc is not None:
            user_desc = user_desc.to(device)

        preds = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            meta_num=meta_num,
            meta_cat=meta_cat,
            meta_bin=meta_bin,
            image_tensor=image_tensor,
            user_desc=user_desc,
        )

        preds_raw  = preds.squeeze(-1).cpu().numpy() * label_std + label_mean
        labels_raw = labels.numpy() * label_std + label_mean

        all_preds.append(preds_raw)
        all_labels.append(labels_raw)

    return np.concatenate(all_preds), np.concatenate(all_labels)


def main():
    v8_config = "configs/test_v8.yaml"
    v9_config = "configs/test_v9.yaml"
    v8_ckpt   = "outputs/test_v8/fold_0/checkpoints/best.pt"
    v9_ckpt   = "outputs/test_v9/fold_0/checkpoints/best.pt"

    print(f"[INFO] Loading v8 model from {v8_ckpt} ...")
    model_v8, val_ds_v8, lm8, ls8 = load_model_and_val_dataset(v8_config, v8_ckpt, DEVICE)
    preds_v8, labels_v8 = get_predictions(model_v8, val_ds_v8, DEVICE, lm8, ls8)
    sp_v8 = spearmanr(labels_v8, preds_v8).correlation
    print(f"  v8 standalone spearman: {sp_v8:.4f}")

    del model_v8
    torch.cuda.empty_cache()

    print(f"\n[INFO] Loading v9 model from {v9_ckpt} ...")
    model_v9, val_ds_v9, lm9, ls9 = load_model_and_val_dataset(v9_config, v9_ckpt, DEVICE)
    preds_v9, labels = get_predictions(model_v9, val_ds_v9, DEVICE, lm9, ls9)
    sp_v9 = spearmanr(labels, preds_v9).correlation
    print(f"  v9 standalone spearman: {sp_v9:.4f}")

    del model_v9
    torch.cuda.empty_cache()

    # both models were trained on same fold-0 split, so labels align
    assert len(preds_v8) == len(preds_v9), (
        f"Prediction length mismatch: v8={len(preds_v8)}, v9={len(preds_v9)}"
    )

    print(f"\n[INFO] Searching best alpha (v9 weight) in [0, 1] step 0.05 ...")
    best_alpha, best_sp = 1.0, sp_v9
    for alpha in np.arange(0.0, 1.01, 0.05):
        blend = alpha * preds_v9 + (1 - alpha) * preds_v8
        sp = spearmanr(labels, blend).correlation
        print(f"  alpha={alpha:.2f} → spearman={sp:.4f}")
        if sp > best_sp:
            best_sp = sp
            best_alpha = alpha

    print(f"\n[RESULT] Best alpha={best_alpha:.2f}, spearman={best_sp:.4f}")
    print(f"         v9 alone: {sp_v9:.4f}  |  v8 alone: {sp_v8:.4f}")
    print(f"         Gain from ensemble: {best_sp - sp_v9:+.4f}")


if __name__ == "__main__":
    main()
