#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
check_current_data_flow.py

A quick debug script for the current SMP pipeline.

It checks:
1. YAML config values that control experiments
2. raw dataframe columns / coverage
3. fold split
4. user/location feature engineering from scripts.train
5. MetadataPreprocessor output dims and selected columns
6. SMPDataset sample keys
7. DataLoader batch keys / tensor shapes / NaN/Inf
8. Optional fastText tag encoder status
9. Optional model forward sanity check

Usage:
  python3 scripts/check_current_data_flow.py --config configs/text_meta_image_v2.yaml --fold 0 --n_folds 5

Optional model forward:
  python3 scripts/check_current_data_flow.py --config configs/text_meta_image_v2.yaml --fold 0 --n_folds 5 --forward
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Reuse the exact functions used by training.
from scripts.train import (
    load_config_with_base,
    load_dataframe,
    make_fold_split,
    ensure_user_aggregate_columns,
    enrich_location_from_text,
    add_user_aggregate_features_fold,
    add_geo_encoding_features,
    build_fasttext_tag_encoder,
)

from src.datasets.metadata_preprocessor import MetadataPreprocessor
from src.datasets.smp_dataset import SMPDataset, smp_collate_fn
from src.models.fusion_model import SMPFusionModel
from src.utils.user_desc_embeddings import maybe_prepare_user_desc_embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--forward", action="store_true", help="Build model and run one forward pass.")
    parser.add_argument("--no_image", action="store_true", help="Skip image loading for faster data sanity check.")
    return parser.parse_args()


def print_section(title: str, width: int = 100) -> None:
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def print_subsection(title: str, width: int = 80) -> None:
    print("\n" + "-" * width)
    print(title)
    print("-" * width)


def non_empty_mask(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip().ne("")


def summarize_raw_col(df: pd.DataFrame, col: str, topn: int = 5) -> None:
    if col not in df.columns:
        print(f"{col:35s} -> MISSING")
        return
    s = df[col]
    non_null = int(s.notna().sum())
    non_empty = int(non_empty_mask(s).sum())
    ratio = non_empty / len(df) * 100 if len(df) else 0.0
    nunique = int(s[non_empty_mask(s)].astype(str).nunique())
    print(f"{col:35s} -> non_null={non_null:7d}/{len(df):7d} | non_empty={non_empty:7d}/{len(df):7d} ({ratio:6.2f}%) | nunique={nunique}")
    if non_empty > 0:
        print(f"{'':35s}    top={s[non_empty_mask(s)].astype(str).value_counts().head(topn).to_dict()}")


def summarize_tensor(name: str, x: Any) -> None:
    if isinstance(x, torch.Tensor):
        shape = tuple(x.shape)
        dtype = str(x.dtype)
        device = str(x.device)
        msg = f"{name:28s} tensor shape={shape} dtype={dtype} device={device}"
        if x.is_floating_point():
            finite = torch.isfinite(x)
            nan_count = int(torch.isnan(x).sum().item())
            inf_count = int(torch.isinf(x).sum().item())
            if finite.any():
                vals = x[finite].detach().float().cpu()
                msg += f" | min={vals.min().item():.4f} max={vals.max().item():.4f} mean={vals.mean().item():.4f}"
            msg += f" | nan={nan_count} inf={inf_count}"
        else:
            if x.numel() > 0:
                vals = x.detach().cpu().reshape(-1)
                msg += f" | min={vals.min().item()} max={vals.max().item()}"
        print(msg)
    elif isinstance(x, list):
        lengths = [len(v) if hasattr(v, "__len__") else -1 for v in x]
        print(f"{name:28s} list len={len(x)} | item_lengths_head={lengths[:10]}")
        for i, item in enumerate(x[:3]):
            s = str(item)
            if len(s) > 180:
                s = s[:180] + " ..."
            print(f"{'':28s} sample[{i}]={s}")
    else:
        s = str(x)
        if len(s) > 220:
            s = s[:220] + " ..."
        print(f"{name:28s} {type(x).__name__}: {s}")


def check_no_bad_tensor(name: str, x: Any) -> None:
    if isinstance(x, torch.Tensor) and x.is_floating_point():
        if torch.isnan(x).any():
            print(f"[WARN] {name} contains NaN")
        if torch.isinf(x).any():
            print(f"[WARN] {name} contains Inf")


def show_available_cols(cols: Sequence[str], keywords: Iterable[str]) -> None:
    for kw in keywords:
        hits = [c for c in cols if kw in c]
        print(f"[COLS] contains '{kw}': {len(hits)}")
        if hits:
            print("       " + ", ".join(hits[:30]) + (" ..." if len(hits) > 30 else ""))


def main() -> None:
    args = parse_args()

    cfg = load_config_with_base(Path(args.config).resolve())

    print_section("CONFIG SUMMARY")
    for key in ["exp_name", "model", "fusion", "meta", "preprocess", "tag_embedding", "loss", "monitor", "code_snapshot"]:
        print(f"[CONFIG] {key}: {cfg.get(key)}")

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    text_cfg = cfg["text"]
    image_cfg = cfg["image"]
    meta_cfg = cfg["meta"]
    train_cfg = cfg["train"]
    preprocess_cfg = cfg["preprocess"]
    fusion_cfg = cfg["fusion"]
    tag_embedding_cfg = cfg.get("tag_embedding", {}) or {}

    official_train_path = data_cfg.get("official_train_path")
    if official_train_path is None:
        raise ValueError("config.data.official_train_path is required")

    print_section("RAW DATAFRAME")
    df = load_dataframe(official_train_path)
    print(f"[DF] path={official_train_path}")
    print(f"[DF] shape={df.shape}")

    # Prepare/reuse Hugging Face sentence-transformer user_description embeddings.
    # This updates cfg["data"] with generated cache paths so SMPDataset can keep
    # using its existing .npy/.json interface without requiring config.data paths.
    user_desc_emb_path, user_desc_idx_path = maybe_prepare_user_desc_embeddings(
        cfg=cfg,
        df=df,
        project_root=ROOT,
    )
    data_cfg = cfg["data"]
    print(f"[USER_DESC_EMB] emb_path={user_desc_emb_path}")
    print(f"[USER_DESC_EMB] idx_path={user_desc_idx_path}")

    important_raw_cols = [
        "post_id", "Pid", "Uid", "label",
        "title", "alltags", "full_text",
        "user_description", "user_description_clean",
        "location_text", "location_description", "location_description_clean",
        "city", "state", "country",
        "latitude", "longitude", "image_path",
    ]
    for col in important_raw_cols:
        summarize_raw_col(df, col)

    print_section("FOLD SPLIT + FEATURE ENGINEERING")
    df = ensure_user_aggregate_columns(df)
    train_df, val_df = make_fold_split(df, fold=args.fold, n_folds=args.n_folds, group_col="Uid")

    train_df = enrich_location_from_text(train_df)
    val_df = enrich_location_from_text(val_df)

    train_df = add_user_aggregate_features_fold(train_df, train_df)
    val_df = add_user_aggregate_features_fold(train_df, val_df)

    train_df = add_geo_encoding_features(train_df, train_df)
    val_df = add_geo_encoding_features(train_df, val_df)

    active_df = train_df if args.split == "train" else val_df
    print(f"[SPLIT] train={train_df.shape} val={val_df.shape} active={args.split}:{active_df.shape}")
    print(f"[SPLIT] train users={train_df['Uid'].nunique()} val users={val_df['Uid'].nunique()}")

    for col in [
        "user_prev_post_count", "user_mean_label", "user_median_label", "user_std_label",
        "location_text_target_enc", "location_text_freq",
        "country_target_enc", "state_target_enc", "city_target_enc",
    ]:
        summarize_raw_col(active_df, col)

    label_mean = float(train_df["label"].mean())
    label_std = float(train_df["label"].std())
    print(f"[LABEL] label_mean={label_mean:.6f} label_std={label_std:.6f}")

    print_section("METADATA PREPROCESSOR")
    preprocessor = MetadataPreprocessor(
        num_cols=preprocess_cfg.get("num_cols"),
        cat_cols=preprocess_cfg.get("cat_cols"),
        bin_cols=preprocess_cfg.get("bin_cols"),
        log1p_cols=preprocess_cfg.get("log1p_cols"),
        normalize_numeric=bool(preprocess_cfg.get("normalize_numeric", True)),
        use_user_desc_lite=bool(preprocess_cfg.get("use_user_desc_lite", True)),
        user_desc_lite_debug=bool(preprocess_cfg.get("user_desc_lite_debug", True)),
        user_desc_lite_max_keywords=int(preprocess_cfg.get("user_desc_lite_max_keywords", 5)),
    )

    train_tf = preprocessor.fit_transform(train_df)
    val_tf = preprocessor.transform(val_df)
    active_tf = train_tf if args.split == "train" else val_tf

    print(f"[META] num_dim={len(preprocessor.transformed_num_cols)}")
    print(f"[META] cat_dim={len(preprocessor.transformed_cat_cols)}")
    print(f"[META] bin_dim={len(preprocessor.transformed_bin_cols)}")
    print(f"[META] cat_cardinalities_count={len(preprocessor.cat_cardinalities)}")

    print_subsection("Transformed metadata column groups")
    show_available_cols(preprocessor.transformed_num_cols, ["user_desc", "loc_desc", "location", "country", "city", "state", "target_enc", "freq"])
    show_available_cols(preprocessor.transformed_bin_cols, ["has_", "user_desc_kw"])
    show_available_cols(preprocessor.transformed_cat_cols, ["city", "state", "country", "category", "geo"])

    print_subsection("Selected transformed column summaries")
    selected_transformed = [
        "num__user_desc_len",
        "num__user_desc_word_count",
        "num__user_desc_keyword_count",
        "bin__has_user_description",
        "bin__user_desc_kw_photo",
        "bin__user_desc_kw_travel",
        "bin__user_desc_kw_camera",
        "num__location_text_target_enc",
        "num__location_text_freq",
    ]
    for col in selected_transformed:
        summarize_raw_col(active_tf, col)

    print_section("DATASET + DATALOADER")
    use_text = bool(model_cfg["use_text"])
    use_meta = bool(model_cfg["use_meta"])
    use_image = bool(model_cfg["use_image"]) and not args.no_image
    use_tag_embedding = bool(tag_embedding_cfg.get("use", tag_embedding_cfg.get("enabled", False)))

    image_path_col = image_cfg.get("path_col", "image_path")
    image_root_dir = image_cfg.get("root_dir", None)

    ds = SMPDataset(
        df=active_tf,
        preprocessor=preprocessor,
        text_model_name=text_cfg["model_name"],
        image_model_name=image_cfg["model_name"],
        user_desc_emb_path=user_desc_emb_path,
        user_desc_idx_path=user_desc_idx_path,
        loc_desc_emb_path=data_cfg.get("loc_desc_emb_path"),
        loc_desc_idx_path=data_cfg.get("loc_desc_idx_path"),
        normalize_label=True,
        label_mean=label_mean,
        label_std=label_std,
        max_length=int(text_cfg["max_length"]),
        use_text=use_text,
        use_meta=use_meta,
        use_image=use_image,
        image_path_col=image_path_col,
        image_root_dir=image_root_dir,
        is_train=(args.split == "train"),
    )

    print(f"[DATASET] len={len(ds)} use_text={use_text} use_meta={use_meta} use_image={use_image} use_tag_embedding={use_tag_embedding}")
    print(f"[DATASET] user_desc_emb_dim={getattr(ds, 'user_desc_emb_dim', None)} loc_desc_emb_dim={getattr(ds, 'loc_desc_emb_dim', None)}")

    print_subsection("Dataset samples")
    for i in range(min(args.num_samples, len(ds))):
        sample = ds[i]
        print(f"\n[SAMPLE {i}] keys={sorted(sample.keys())}")
        for k in [
            "post_id", "Pid", "Uid", "label",
            "input_ids", "attention_mask",
            "meta_num", "meta_cat", "meta_bin",
            "tag_tokens", "tag_token_count", "tag_text",
            "user_desc", "loc_desc",
            "image_tensor",
        ]:
            if k in sample:
                summarize_tensor(k, sample[k])

    loader = DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=False,
        persistent_workers=False,
        collate_fn=smp_collate_fn,
        drop_last=False,
    )

    batch = next(iter(loader))

    print_section("BATCH CHECK")
    print(f"[BATCH] keys={sorted(batch.keys())}")
    for k, v in batch.items():
        summarize_tensor(k, v)
        check_no_bad_tensor(k, v)

    # Dimension checks
    print_subsection("Dimension checks")
    if "meta_num" in batch:
        print(f"[CHECK] meta_num dim batch={batch['meta_num'].shape[-1]} expected={len(preprocessor.transformed_num_cols)}")
    if "meta_cat" in batch:
        print(f"[CHECK] meta_cat dim batch={batch['meta_cat'].shape[-1]} expected={len(preprocessor.transformed_cat_cols)}")
    if "meta_bin" in batch:
        print(f"[CHECK] meta_bin dim batch={batch['meta_bin'].shape[-1]} expected={len(preprocessor.transformed_bin_cols)}")
    if use_tag_embedding:
        has_tokens = "tag_tokens" in batch and isinstance(batch["tag_tokens"], list)
        print(f"[CHECK] tag_embedding.use=true | batch has tag_tokens={has_tokens}")

    print_section("OPTIONAL TAG ENCODER CHECK")
    tag_encoder = build_fasttext_tag_encoder(
        tag_cfg=tag_embedding_cfg,
        data_cfg=data_cfg,
        use_tag_embedding=use_tag_embedding,
        hidden_dim=int(model_cfg["hidden_dim"]),
        dropout=float(model_cfg["dropout"]),
    )
    print(f"[TAG_EMB] tag_encoder={'yes' if tag_encoder is not None else 'no'}")
    if tag_encoder is not None:
        with torch.no_grad():
            tag_feat = tag_encoder(batch["tag_tokens"])
        summarize_tensor("tag_feat", tag_feat)

    if args.forward:
        print_section("MODEL FORWARD CHECK")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        cat_cardinalities = [
            int(preprocessor.cat_cardinalities[col])
            for col in preprocessor.cat_cols
        ]

        model = SMPFusionModel(
            text_model_name=text_cfg["model_name"],
            meta_num_dim=len(preprocessor.transformed_num_cols),
            meta_cat_cardinalities=cat_cardinalities,
            meta_bin_dim=len(preprocessor.transformed_bin_cols),
            meta_num_cols=preprocessor.transformed_num_cols,
            meta_cat_cols=preprocessor.transformed_cat_cols,
            meta_bin_cols=preprocessor.transformed_bin_cols,
            hidden_dim=int(model_cfg["hidden_dim"]),
            dropout=float(model_cfg["dropout"]),
            use_text=use_text,
            use_tag_embedding=use_tag_embedding,
            tag_encoder=tag_encoder,
            use_meta=use_meta,
            use_image=use_image,
            image_model_name=image_cfg["model_name"],
            text_pooling=text_cfg["pooling"],
            text_trainable=bool(text_cfg["trainable"]),
            image_pretrained=bool(image_cfg.get("pretrained", True)),
            image_trainable=bool(image_cfg["trainable"]),
            fusion_type=str(fusion_cfg.get("type", "pairwise_gated")),
            use_clip_similarity=bool(fusion_cfg.get("use_clip_similarity", True)),
            clip_similarity_mode=str(fusion_cfg.get("clip_similarity_mode", "raw")),
            head_type=str(model_cfg.get("head_type", "moe")),
            normalize_image_feature=bool(image_cfg.get("normalize_feature", False)),
            meta_branch_dim=int(meta_cfg["branch_dim"]),
            use_semantic_meta_groups=bool(meta_cfg.get("use_semantic_groups", False)),
            use_user_desc=bool(meta_cfg.get("use_user_desc", False)),
            user_desc_dim=int(meta_cfg.get("user_desc_dim", getattr(ds, "user_desc_emb_dim", None) or 768)),
            use_loc_desc=bool(meta_cfg.get("use_loc_desc", False)),
            loc_desc_dim=int(meta_cfg.get("loc_desc_dim", getattr(ds, "loc_desc_emb_dim", None) or 400)),
            desc_bottleneck_dim=int(meta_cfg.get("desc_bottleneck_dim", 32)),
            user_desc_scale=float(meta_cfg.get("user_desc_scale", 0.0)),
            loc_desc_scale=float(meta_cfg.get("loc_desc_scale", 0.0)),
        ).to(device)

        model.eval()

        model_inputs = {
            "input_ids": batch.get("input_ids"),
            "attention_mask": batch.get("attention_mask"),
            "meta_num": batch.get("meta_num"),
            "meta_cat": batch.get("meta_cat"),
            "meta_bin": batch.get("meta_bin"),
            "image_tensor": batch.get("image_tensor"),
            "tag_tokens": batch.get("tag_tokens"),
            "tag_text": batch.get("tag_text"),
            "tag_token_count": batch.get("tag_token_count"),
            "user_desc": batch.get("user_desc"),
            "loc_desc": batch.get("loc_desc"),
            "return_features": True,
        }

        for k, v in list(model_inputs.items()):
            if isinstance(v, torch.Tensor):
                model_inputs[k] = v.to(device)

        with torch.no_grad():
            out = model(**model_inputs)

        summarize_tensor("model.output", out["output"])
        summarize_tensor("model.fused", out["fused"])
        print("[MODEL FEATURES]")
        for k, v in out["features"].items():
            if v is not None:
                summarize_tensor(f"feature.{k}", v)

    print_section("DONE")
    print("[OK] Data flow check completed.")


if __name__ == "__main__":
    main()
