from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.datasets.fold_features import (
    add_geo_encoding_features,
    add_user_aggregate_features_fold,
    enrich_location_from_text,
    ensure_user_aggregate_columns,
    load_dataframe,
    make_fold_split,
)
from src.datasets.metadata_preprocessor import MetadataPreprocessor
from src.datasets.smp_dataset import SMPDataset, smp_collate_fn
from src.engine.experiment_factory import build_fasttext_tag_encoder
from src.models.factory import build_smp_fusion_model
from src.utils.config import load_config_with_base
from src.utils.user_desc_embeddings import maybe_prepare_user_desc_embeddings


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

        tag_tokens = batch.get("tag_tokens", None)
        tag_text = batch.get("tag_text", None)
        tag_token_count = batch.get("tag_token_count", None)
        if tag_token_count is not None:
            tag_token_count = tag_token_count.to(device)

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
            tag_tokens=tag_tokens,
            tag_text=tag_text,
            tag_token_count=tag_token_count,
            user_desc=user_desc,
            loc_desc=loc_desc,
            return_features=True,
        )

        pred_norm = out["output"].squeeze(-1).detach().cpu().numpy()
        pred_raw = pred_norm * label_std + label_mean

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
            clip_sim_arr = clip_sim.detach().cpu().numpy()
            if clip_sim_arr.shape[1] == 1:
                data["clip_sim"] = clip_sim_arr[:, 0]
            else:
                for i in range(clip_sim_arr.shape[1]):
                    data[f"clip_sim_{i}"] = clip_sim_arr[:, i]

        feature_cols: Dict[str, np.ndarray] = {}
        feature_tensors = {
            "fused": out["fused"],
            "text": out["features"].get("text"),
            "meta": out["features"].get("meta"),
            "image": out["features"].get("image"),
        }
        for name, tensor in feature_tensors.items():
            if tensor is None:
                continue
            arr = tensor.detach().cpu().numpy()
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
    preprocess_cfg = cfg["preprocess"]
    tag_embedding_cfg = cfg.get("tag_embedding", {}) or {}

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

    if bool(preprocess_cfg.get("use_label_user_aggregates", False)):
        train_df = add_user_aggregate_features_fold(train_df, train_df)
        val_df = add_user_aggregate_features_fold(train_df, val_df)

    target_encoding_smoothing = float(preprocess_cfg.get("target_encoding_smoothing", 30.0))
    train_df = add_geo_encoding_features(train_df, train_df, smoothing=target_encoding_smoothing)
    val_df = add_geo_encoding_features(train_df, val_df, smoothing=target_encoding_smoothing)

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
        use_caption=bool(text_cfg.get("use_caption", False)),
        caption_max_freq=text_cfg.get("caption_max_freq", 50),
        image_path_col=image_cfg.get("path_col", "image_path"),
        image_root_dir=image_cfg.get("root_dir", None),
        user_desc_emb_path=user_desc_emb_path,
        user_desc_idx_path=user_desc_idx_path,
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

    use_tag_embedding = bool(tag_embedding_cfg.get("use", tag_embedding_cfg.get("enabled", False)))
    tag_encoder = build_fasttext_tag_encoder(
        tag_cfg=tag_embedding_cfg,
        data_cfg=data_cfg,
        use_tag_embedding=use_tag_embedding,
        hidden_dim=int(model_cfg["hidden_dim"]),
        dropout=float(model_cfg["dropout"]),
    )

    model = build_smp_fusion_model(
        cfg=cfg,
        preprocessor=preprocessor,
        tag_encoder=tag_encoder,
        dataset_user_desc_dim=getattr(dataset, "user_desc_emb_dim", 0),
        dataset_loc_desc_dim=getattr(dataset, "loc_desc_emb_dim", 0),
    ).to(device)

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
