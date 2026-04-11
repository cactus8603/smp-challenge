from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.datasets.smp_dataset import SMPDataset, smp_collate_fn
from src.models.fusion_model import SMPFusionModel
from src.utils.config import load_config
from src.utils.metrics import compute_mae, compute_spearman


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SMP model with YAML config.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment config yaml.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_loss(loss_name: str) -> torch.nn.Module:
    name = loss_name.lower()
    if name == "mse":
        return torch.nn.MSELoss()
    if name == "mae":
        return torch.nn.L1Loss()
    if name == "smoothl1":
        return torch.nn.SmoothL1Loss()
    raise ValueError(f"Unsupported loss: {loss_name}")


def save_json(data: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_score: float,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_score": best_score,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def setup_logger(log_dir: Path, exp_name: str) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(exp_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_dir / "train.log", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: str,
) -> float:
    model.train()
    total_loss = 0.0

    progress = tqdm(loader, desc="Train", leave=False)

    for batch in progress:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        meta_features = batch["meta_features"].to(device)
        labels = batch["labels"].to(device)

        image_tensor = batch.get("image_tensor", None)
        if image_tensor is not None:
            image_tensor = image_tensor.to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            meta_features=meta_features,
            image_tensor=image_tensor,
        )

        preds = outputs.squeeze(-1)
        loss = criterion(preds, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: str,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0

    all_preds = []
    all_labels = []

    progress = tqdm(loader, desc="Valid", leave=False)

    for batch in progress:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        meta_features = batch["meta_features"].to(device)
        labels = batch["labels"].to(device)

        image_tensor = batch.get("image_tensor", None)
        if image_tensor is not None:
            image_tensor = image_tensor.to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            meta_features=meta_features,
            image_tensor=image_tensor,
        )

        preds = outputs.squeeze(-1)
        loss = criterion(preds, labels)

        total_loss += loss.item()
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())

        progress.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / max(len(loader), 1)
    mae = compute_mae(all_labels, all_preds)
    spearman = compute_spearman(all_labels, all_preds)

    return avg_loss, mae, spearman


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    set_seed(cfg.seed)
    device = get_device()

    output_root = Path(cfg.output.root_dir)
    exp_dir = output_root / cfg.exp_name
    ckpt_dir = Path(cfg.output.save_dir) / cfg.exp_name
    tb_dir = Path(cfg.output.tensorboard_dir) / cfg.exp_name

    exp_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(exp_dir, cfg.exp_name)
    writer = SummaryWriter(log_dir=str(tb_dir))

    logger.info(f"Using device: {device}")
    logger.info(f"Experiment: {cfg.exp_name}")
    logger.info(f"Config: {args.config}")

    save_json(cfg.to_dict(), exp_dir / "config.json")

    # -------------------------
    # data
    # -------------------------
    train_df = pd.read_parquet(cfg.data.train_path)
    val_df = pd.read_parquet(cfg.data.val_path)

    logger.info(f"Train shape: {train_df.shape}")
    logger.info(f"Val shape: {val_df.shape}")

    train_dataset = SMPDataset(
        df=train_df,
        text_model_name=cfg.text.model_name,
        max_length=cfg.text.max_length,
        use_text=cfg.model.use_text,
        use_meta=cfg.model.use_meta,
        use_image=cfg.model.use_image,
        is_train=True,
    )
    val_dataset = SMPDataset(
        df=val_df,
        text_model_name=cfg.text.model_name,
        max_length=cfg.text.max_length,
        use_text=cfg.model.use_text,
        use_meta=cfg.model.use_meta,
        use_image=cfg.model.use_image,
        is_train=False,
    )

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Val dataset size: {len(val_dataset)}")
    logger.info(f"Metadata dim: {train_dataset.meta_dim}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        collate_fn=smp_collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        collate_fn=smp_collate_fn,
        pin_memory=True,
    )

    # -------------------------
    # model
    # -------------------------
    model = SMPFusionModel(
        text_model_name=cfg.text.model_name,
        meta_dim=train_dataset.meta_dim,
        hidden_dim=cfg.model.hidden_dim,
        dropout=cfg.model.dropout,
        use_text=cfg.model.use_text,
        use_meta=cfg.model.use_meta,
        use_image=cfg.model.use_image,
        image_model_name=cfg.image.model_name,
        text_pooling=cfg.text.pooling,
        text_trainable=cfg.text.trainable,
        image_pretrained=cfg.image.pretrained,
        image_trainable=cfg.image.trainable,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )

    criterion = build_loss(cfg.loss.name)

    logger.info("Model initialized successfully.")
    logger.info(f"Loss: {cfg.loss.name}")
    logger.info(f"Batch size: {cfg.train.batch_size}")
    logger.info(f"Epochs: {cfg.train.epochs}")
    logger.info(f"Learning rate: {cfg.train.lr}")

    # -------------------------
    # train loop
    # -------------------------
    best_spearman = -1.0
    best_epoch = -1
    history = []

    for epoch in range(1, cfg.train.epochs + 1):
        epoch_start = time.time()
        logger.info(f"Epoch {epoch}/{cfg.train.epochs}")

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        val_loss, val_mae, val_spearman = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        epoch_time = time.time() - epoch_start

        logger.info(
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_mae={val_mae:.4f} | "
            f"val_spearman={val_spearman:.4f} | "
            f"epoch_time={epoch_time:.2f}s"
        )

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("metric/mae", val_mae, epoch)
        writer.add_scalar("metric/spearman", val_spearman, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
        writer.add_scalar("time/epoch_seconds", epoch_time, epoch)

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_mae": float(val_mae),
                "val_spearman": float(val_spearman),
                "epoch_time_sec": float(epoch_time),
            }
        )
        save_json({"history": history}, exp_dir / "history.json")

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_score=best_spearman,
            path=ckpt_dir / "latest.pt",
        )

        if val_spearman > best_spearman:
            best_spearman = val_spearman
            best_epoch = epoch

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_score=best_spearman,
                path=ckpt_dir / "best.pt",
            )
            logger.info(f"Saved best model to {ckpt_dir / 'best.pt'}")

    writer.close()

    summary = {
        "exp_name": cfg.exp_name,
        "best_epoch": best_epoch,
        "best_val_spearman": float(best_spearman),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
    }
    save_json(summary, exp_dir / "summary.json")

    logger.info("Training finished.")
    logger.info(f"Best epoch: {best_epoch}")
    logger.info(f"Best val spearman: {best_spearman:.4f}")


if __name__ == "__main__":
    main()