from __future__ import annotations

import argparse
import json
import random
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

# Allow scripts/ to import from src/
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
from src.engine.experiment_factory import (
    build_fasttext_tag_encoder,
    build_loss,
    snapshot_python_code,
)
from src.engine.trainer import Trainer
from src.models.factory import build_smp_fusion_model
from src.utils.config import load_config_with_base
from src.utils.user_desc_embeddings import maybe_prepare_user_desc_embeddings


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
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
    tag_embedding_cfg = cfg.get("tag_embedding", {}) or {}
    code_snapshot_cfg = cfg.get("code_snapshot", {}) or {}
    diagnostics_cfg = cfg.get("diagnostics", {}) or {}
    
    # user_desc paths are prepared below from cfg.user_desc_encoder.
    user_desc_emb_path = None
    user_desc_idx_path = None

    use_text = bool(model_cfg["use_text"])
    use_meta = bool(model_cfg["use_meta"])
    use_image = bool(model_cfg["use_image"])
    use_tag_embedding = bool(tag_embedding_cfg.get("use", tag_embedding_cfg.get("enabled", False)))

    text_model_name = text_cfg["model_name"]
    image_model_name = image_cfg["model_name"]
    image_path_col = image_cfg.get("path_col", "image_path")
    cache_static_fields = bool(preprocess_cfg.get("cache_static_fields", False))

    batch_size = int(train_cfg["batch_size"])
    num_workers = int(train_cfg["num_workers"])
    pin_memory = bool(train_cfg.get("pin_memory", True))
    persistent_workers = bool(train_cfg.get("persistent_workers", num_workers > 0))
    drop_last = bool(train_cfg.get("drop_last", False))

    if bool(code_snapshot_cfg.get("enabled", True)):
        snapshot_python_code(ROOT, exp_dir)

    # -------------------------
    # data loading: KFold from official_train only
    # -------------------------
    official_train_path = data_cfg.get("official_train_path")
    if official_train_path is None:
        raise ValueError(
            "For KFold training, config.data.official_train_path is required."
        )

    official_train_df = load_dataframe(official_train_path)

    # Optional reproducible Hugging Face / sentence-transformers user_description embeddings.
    # If cfg.user_desc_encoder.enabled=true, this builds/reuses a cache and updates
    # user_desc_emb_path / user_desc_idx_path for SMPDataset below.
    user_desc_emb_path, user_desc_idx_path = maybe_prepare_user_desc_embeddings(
        cfg=cfg,
        df=official_train_df,
        project_root=ROOT,
    )
    data_cfg = cfg["data"]

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

    # Fill empty city/state/country from location_text before geo encoding.
    train_df = enrich_location_from_text(train_df)
    val_df = enrich_location_from_text(val_df)

    # ---------------------------------------------------------
    # Optional label-derived user aggregates.
    # With GroupKFold by Uid, validation users are unseen, so these train-only
    # aggregates can create a train/val distribution mismatch. Keep them behind
    # a config flag for experiments where user overlap is expected.
    # ---------------------------------------------------------
    if bool(preprocess_cfg.get("use_label_user_aggregates", False)):
        train_df = add_user_aggregate_features_fold(train_df, train_df)
        val_df = add_user_aggregate_features_fold(train_df, val_df)

    # geo target encoding + frequency (computed from fold-train, applied to train/val)
    # Train rows use leave-one-out target encoding to avoid seeing their own label.
    target_encoding_smoothing = float(preprocess_cfg.get("target_encoding_smoothing", 30.0))
    train_df = add_geo_encoding_features(train_df, train_df, smoothing=target_encoding_smoothing)
    val_df   = add_geo_encoding_features(train_df, val_df, smoothing=target_encoding_smoothing)

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
        normalize_numeric=bool(preprocess_cfg.get("normalize_numeric", True)),
        use_user_desc_lite=bool(preprocess_cfg.get("use_user_desc_lite", True)),
        user_desc_lite_debug=bool(preprocess_cfg.get("user_desc_lite_debug", True)),
        user_desc_lite_max_keywords=int(preprocess_cfg.get("user_desc_lite_max_keywords", 5)),
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
        user_desc_emb_path=user_desc_emb_path,
        user_desc_idx_path=user_desc_idx_path,
        normalize_label=True,
        label_mean=label_mean,
        label_std=label_std,
        max_length=int(text_cfg["max_length"]),
        use_text=use_text,
        use_meta=use_meta,
        use_caption=bool(text_cfg.get("use_caption", False)),
        caption_max_freq=text_cfg.get("caption_max_freq", 50),
        use_image=use_image,
        image_path_col=image_path_col,
        image_root_dir=image_root_dir,
        is_train=True,
        cache_static_fields=cache_static_fields,
    )

    val_dataset = SMPDataset(
        df=val_df,
        preprocessor=preprocessor,
        text_model_name=text_model_name,
        image_model_name=image_model_name,
        user_desc_emb_path=user_desc_emb_path,
        user_desc_idx_path=user_desc_idx_path,
        normalize_label=True,
        label_mean=label_mean,
        label_std=label_std,
        max_length=int(text_cfg["max_length"]),
        use_text=use_text,
        use_meta=use_meta,
        use_caption=bool(text_cfg.get("use_caption", False)),
        caption_max_freq=text_cfg.get("caption_max_freq", 50),
        use_image=use_image,
        image_path_col=image_path_col,
        image_root_dir=image_root_dir,
        is_train=False,
        cache_static_fields=cache_static_fields,
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
    # optional fastText tag encoder
    # -------------------------
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
        dataset_user_desc_dim=getattr(
            train_dataset,
            "user_desc_feature_dim",
            getattr(train_dataset, "user_desc_emb_dim", 0),
        ),
        dataset_loc_desc_dim=getattr(train_dataset, "loc_desc_emb_dim", 0),
    ).to(device)

    print(
        f"[TAG_EMB] config.use_tag_embedding={use_tag_embedding} | "
        f"model.use_tag_embedding={getattr(model, 'use_tag_embedding', None)} | "
        f"text_branch_enabled={getattr(model, 'text_branch_enabled', None)} | "
        f"tag_encoder={'yes' if getattr(model, 'tag_encoder', None) is not None else 'no'}"
    )

    # -------------------------
    # optimization
    # -------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )

    criterion = build_loss(loss_cfg)

    print(f"[CONFIG] text={text_cfg}")
    print(f"[CONFIG] fusion={fusion_cfg}")
    print(f"[CONFIG] model.head_type={model_cfg.get('head_type', 'moe')}")
    print(f"[CONFIG] model.use_meta_aux_head={model_cfg.get('use_meta_aux_head', False)} | meta_aux_scale={model_cfg.get('meta_aux_scale', 0.0)}")
    print(f"[CONFIG] fusion.type={fusion_cfg.get('type', 'pairwise_gated')} | num_heads={fusion_cfg.get('num_heads', 4)}")
    print(f"[CONFIG] meta={meta_cfg}")
    print(f"[CONFIG] preprocess={preprocess_cfg}")
    print(f"[CONFIG] loss={loss_cfg}")
    print(f"[CONFIG] tag_embedding={tag_embedding_cfg}")
    print(f"[CONFIG] diagnostics={diagnostics_cfg}")

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
        ablation_spearman_threshold=float(
            diagnostics_cfg.get("ablation_spearman_threshold", 0.58)
        ),
        ablation_on_best_only=bool(
            diagnostics_cfg.get("ablation_on_best_only", True)
        ),
        meta_warmup_epochs=int(train_cfg.get("meta_warmup_epochs", 0)),
        early_stop_patience=train_cfg.get("early_stop_patience"),
        early_stop_min_delta=float(train_cfg.get("early_stop_min_delta", 0.0)),
    )

    monitor_cfg = cfg.get("monitor", {})

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=int(train_cfg["epochs"]),
        monitor_gbdt=bool(monitor_cfg.get("gbdt", False)),
        gbdt_interval=int(monitor_cfg.get("gbdt_interval", 1)),
        gbdt_max_train_batches=monitor_cfg.get("gbdt_max_train_batches", None),
        gbdt_max_val_batches=monitor_cfg.get("gbdt_max_val_batches", None),
    )


if __name__ == "__main__":
    main()

# Example:
# python3 scripts/train.py --config configs/text_meta_image_v2.yaml --fold 0 --n_folds 5
# python3 scripts/train.py --config configs/text_meta_image_v2.yaml --fold 1 --n_folds 5
# nohup python3 scripts/train.py --config configs/cross_attention_v2_fix.yaml --fold 0 --n_folds 5 > cross_attention_v2_fix.log 2>&1 &

"""
nohup python3 scripts/train.py --config configs/xattn_udcap.yaml --fold 0 --n_folds 5 > xattn_udcap_fold0.log 2>&1 &

=== Modality Ablation Summary ===
      full | loss=0.9288 | mae=1.7273 | spearman=0.5963
 mask_text | loss=0.9809 | mae=1.7514 | spearman=0.5506
 mask_meta | loss=1.0550 | mae=1.8767 | spearman=0.4321
mask_image | loss=0.9371 | mae=1.7072 | spearman=0.5688

=== Spearman Drop From Full ===
      text: 0.0458
      meta: 0.1642
     image: 0.0276
"""
