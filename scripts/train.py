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
from torch.utils.data import DataLoader

# Allow scripts/ to import from src/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.datasets.metadata_preprocessor import MetadataPreprocessor
from src.datasets.smp_datasets import SMPDataset, smp_collate_fn
from src.engine.trainer import Trainer
from src.models.fusion_model import SMPFusionModel
from src.utils.criterion import PairwiseRankingLoss, HybridLoss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
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
        base_loss = torch.nn.MSELoss()
    elif name == "mae" or name == "l1":
        base_loss = torch.nn.L1Loss()
    elif name == "smoothl1":
        base_loss = torch.nn.SmoothL1Loss()
    elif name == "ranking":
        return PairwiseRankingLoss()
    elif name == "hybrid":
        return HybridLoss()
    else:
        raise ValueError(f"Unsupported loss: {loss_name}")

    return base_loss


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
    exp_dir = output_root / exp_name
    ckpt_dir = exp_dir / "checkpoints"
    tb_dir = exp_dir / "tensorboard"
    exp_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    # Save the resolved config for reproducibility
    with (exp_dir / "resolved_config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    # -------------------------
    # data loading
    # -------------------------
    train_df = load_dataframe(cfg["data"]["train_path"])
    val_df = load_dataframe(cfg["data"]["val_path"])

    preprocessor = MetadataPreprocessor(
        num_cols=cfg["preprocess"].get("num_cols"),
        cat_cols=cfg["preprocess"].get("cat_cols"),
        bin_cols=cfg["preprocess"].get("bin_cols"),
        log1p_cols=cfg["preprocess"].get("log1p_cols"),
    )

    train_df = preprocessor.fit_transform(train_df)
    val_df = preprocessor.transform(val_df)

    # Save metadata preprocessor state
    preprocessor.save(exp_dir / "metadata_preprocessor.json")

    train_dataset = SMPDataset(
        df=train_df,
        preprocessor=preprocessor,
        text_model_name=cfg["text"]["model_name"],
        max_length=int(cfg["text"]["max_length"]),
        use_text=bool(cfg["model"]["use_text"]),
        use_meta=bool(cfg["model"]["use_meta"]),
        use_image=bool(cfg["model"]["use_image"]),
        is_train=True,
    )

    val_dataset = SMPDataset(
        df=val_df,
        preprocessor=preprocessor,
        text_model_name=cfg["text"]["model_name"],
        max_length=int(cfg["text"]["max_length"]),
        use_text=bool(cfg["model"]["use_text"]),
        use_meta=bool(cfg["model"]["use_meta"]),
        use_image=bool(cfg["model"]["use_image"]),
        is_train=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["train"]["num_workers"]),
        pin_memory=bool(cfg["train"].get("pin_memory", True)),
        collate_fn=smp_collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["train"]["num_workers"]),
        pin_memory=bool(cfg["train"].get("pin_memory", True)),
        collate_fn=smp_collate_fn,
    )

    # -------------------------
    # model
    # -------------------------
    cat_cardinalities = [
        int(preprocessor.cat_cardinalities[col])
        for col in preprocessor.cat_cols
    ]

    model = SMPFusionModel(
        text_model_name=cfg["text"]["model_name"],
        meta_num_dim=len(preprocessor.transformed_num_cols),
        meta_cat_cardinalities=cat_cardinalities,
        meta_bin_dim=len(preprocessor.transformed_bin_cols),
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        dropout=float(cfg["model"]["dropout"]),
        use_text=bool(cfg["model"]["use_text"]),
        use_meta=bool(cfg["model"]["use_meta"]),
        use_image=bool(cfg["model"]["use_image"]),
        image_model_name=cfg["image"]["model_name"],
        text_pooling=cfg["text"]["pooling"],
        text_trainable=bool(cfg["text"]["trainable"]),
        image_pretrained=bool(cfg["image"]["pretrained"]),
        image_trainable=bool(cfg["image"]["trainable"]),
        fusion_type=cfg["fusion"]["type"],
        meta_branch_dim=int(cfg["meta"]["branch_dim"]),
    ).to(device)

    # -------------------------
    # optimization
    # -------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"].get("weight_decay", 1e-4)),
    )

    criterion = build_loss(cfg["loss"]["name"])

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=str(device),
        exp_name=exp_name,
        exp_dir=exp_dir,
        ckpt_dir=ckpt_dir,
        tb_dir=tb_dir,
        grad_clip_norm=cfg["train"].get("grad_clip_norm"),
    )

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=int(cfg["train"]["epochs"]),
    )


if __name__ == "__main__":
    main()

# python3 scripts/train.py --config configs/text_meta_v1.yaml