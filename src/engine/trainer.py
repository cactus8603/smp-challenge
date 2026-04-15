from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.engine.evaluator import validate


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

    console_handler = logging.StreamHandler()
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
    scheduler: torch.optim.lr_scheduler,
    device: str,
    grad_clip_norm: Optional[float] = None,
    writer: Optional[SummaryWriter] = None,
    epoch: int = 0,
) -> float:
    model.train()
    total_loss = 0.0
    total_reg_loss = 0.0
    total_rank_loss = 0.0

    progress = tqdm(loader, desc="Train", leave=False)
    step = 0

    for batch in progress:
        # --------------------------
        # debugging: check batch contents and shapes
        # # --------------------------
        # print(batch["input_ids"].shape)          # CLIP
        # print(batch["attention_mask"].shape)
        # print(batch["clip_text"][:2])

        # print(batch["glove_token_count"][:5])    # GloVe
        # print(batch["glove_tokens"][:2])
        # print(batch["glove_text"][:2])

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        meta_num = batch["meta_num"].to(device)
        meta_cat = batch["meta_cat"].to(device)
        meta_bin = batch["meta_bin"].to(device)
        labels = batch["labels"].to(device)

        image_tensor = batch.get("image_tensor", None)
        if image_tensor is not None and image_tensor.numel() > 0:
            image_tensor = image_tensor.to(device)
        else:
            image_tensor = None

        optimizer.zero_grad(set_to_none=True)

        # GloVe branch inputs
        glove_tokens = batch.get("glove_tokens", None)
        glove_text = batch.get("glove_text", None)
        glove_token_count = batch.get("glove_token_count", None)
        if glove_token_count is not None:
            glove_token_count = glove_token_count.to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            meta_num=meta_num,
            meta_cat=meta_cat,
            meta_bin=meta_bin,
            image_tensor=image_tensor,
            glove_tokens=glove_tokens,
            glove_text=glove_text,
            glove_token_count=glove_token_count,
        )

        preds = outputs.squeeze(-1)
        loss = criterion(preds, labels)

        loss.backward()

        if grad_clip_norm is not None and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        # 記錄各損失
        if hasattr(criterion, "last_reg_loss"):
            total_reg_loss += criterion.last_reg_loss
            total_rank_loss += criterion.last_rank_loss

            if writer is not None:
                global_step = epoch * len(loader) + step
                writer.add_scalar("loss/train_reg", criterion.last_reg_loss, global_step)
                writer.add_scalar("loss/train_rank", criterion.last_rank_loss, global_step)

        step += 1
        progress.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / max(len(loader), 1)
    avg_reg_loss = total_reg_loss / max(len(loader), 1)
    avg_rank_loss = total_rank_loss / max(len(loader), 1)

    return avg_loss


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        scheduler: torch.optim.lr_scheduler,
        device: str,
        exp_name: str,
        exp_dir: Path,
        ckpt_dir: Path,
        tb_dir: Path,
        logger: Optional[logging.Logger] = None,
        grad_clip_norm: Optional[float] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.exp_name = exp_name
        self.exp_dir = exp_dir
        self.ckpt_dir = ckpt_dir
        self.tb_dir = tb_dir
        self.grad_clip_norm = grad_clip_norm

        self.logger = logger or setup_logger(exp_dir, exp_name)
        self.writer = SummaryWriter(log_dir=str(tb_dir))

        self.best_spearman = -1.0
        self.best_epoch = -1
        self.history: List[Dict] = []

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
    ) -> Dict:
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            self.logger.info(f"Epoch {epoch}/{epochs}")

            train_loss = train_one_epoch(
                model=self.model,
                loader=train_loader,
                optimizer=self.optimizer,
                criterion=self.criterion,
                scheduler=self.scheduler,
                device=self.device,
                grad_clip_norm=self.grad_clip_norm,
                writer=self.writer,
                epoch=epoch,
            )

            val_loss, val_mae, val_spearman = validate(
                model=self.model,
                loader=val_loader,
                criterion=self.criterion,
                device=self.device,
            )

            epoch_time = time.time() - epoch_start

            self.logger.info(
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_mae={val_mae:.4f} | "
                f"val_spearman={val_spearman:.4f} | "
                f"epoch_time={epoch_time:.2f}s"
            )

            self.writer.add_scalar("loss/train", train_loss, epoch)
            self.writer.add_scalar("loss/val", val_loss, epoch)
            self.writer.add_scalar("metric/mae", val_mae, epoch)
            self.writer.add_scalar("metric/spearman", val_spearman, epoch)
            self.writer.add_scalar("lr", self.optimizer.param_groups[0]["lr"], epoch)
            self.writer.add_scalar("time/epoch_seconds", epoch_time, epoch)

            self.history.append(
                {
                    "epoch": epoch,
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                    "val_mae": float(val_mae),
                    "val_spearman": float(val_spearman),
                    "epoch_time_sec": float(epoch_time),
                }
            )

            save_json({"history": self.history}, self.exp_dir / "history.json")

            save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                best_score=self.best_spearman,
                path=self.ckpt_dir / "latest.pt",
            )

            if val_spearman > self.best_spearman:
                self.best_spearman = val_spearman
                self.best_epoch = epoch

                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    best_score=self.best_spearman,
                    path=self.ckpt_dir / "best.pt",
                )
                self.logger.info(f"Saved best model to {self.ckpt_dir / 'best.pt'}")

        self.writer.close()

        summary = {
            "exp_name": self.exp_name,
            "best_epoch": self.best_epoch,
            "best_val_spearman": float(self.best_spearman),
        }
        save_json(summary, self.exp_dir / "summary.json")

        self.logger.info("Training finished.")
        self.logger.info(f"Best epoch: {self.best_epoch}")
        self.logger.info(f"Best val spearman: {self.best_spearman:.4f}")

        return summary
