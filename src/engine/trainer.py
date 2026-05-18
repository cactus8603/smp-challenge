from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.engine.evaluator import validate, validate_modality_ablation
from src.engine.gbdt_monitor import run_gbdt_monitor


def save_json(data: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    return value


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
    meta_only: bool = False,
) -> float:
    model.train()
    total_loss = 0.0
    total_reg_loss = 0.0
    total_rank_loss = 0.0

    progress = tqdm(loader, desc="Train", leave=False)
    step = 0

    for batch in progress:
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

        # tag branch inputs
        tag_tokens = batch.get("tag_tokens", None)
        tag_text = batch.get("tag_text", None)
        tag_token_count = batch.get("tag_token_count", None)
        if tag_token_count is not None:
            tag_token_count = tag_token_count.to(device)

        # user_description / location_description
        user_desc = batch["user_desc"].to(device) if "user_desc" in batch else None
        loc_desc = batch["loc_desc"].to(device) if "loc_desc" in batch else None

        outputs = model(
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
            meta_only=meta_only,
        )

        preds = outputs.squeeze(-1)
        loss = criterion(preds, labels)

        loss.backward()

        if grad_clip_norm is not None and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        # record component losses if criterion exposes them
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
        ablation_spearman_threshold: float = 0.58,
        ablation_on_best_only: bool = True,
        meta_warmup_epochs: int = 0,
        early_stop_patience: Optional[int] = None,
        early_stop_min_delta: float = 0.0,
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
        self.ablation_spearman_threshold = float(ablation_spearman_threshold)
        self.ablation_on_best_only = bool(ablation_on_best_only)
        self.meta_warmup_epochs = int(meta_warmup_epochs)
        self.early_stop_patience = (
            int(early_stop_patience)
            if early_stop_patience is not None and int(early_stop_patience) > 0
            else None
        )
        self.early_stop_min_delta = float(early_stop_min_delta)
        self._original_requires_grad = {name: p.requires_grad for name, p in model.named_parameters()}

        self.logger = logger or setup_logger(exp_dir, exp_name)
        self.writer = SummaryWriter(log_dir=str(tb_dir))

        self.best_spearman = -1.0
        self.best_epoch = -1
        self.best_ablation: Optional[Dict] = None
        self._early_stop_best = -1.0
        self._epochs_without_improvement = 0
        self.stopped_early = False
        self.stop_epoch: Optional[int] = None
        self.history: List[Dict] = []


    def _set_meta_warmup_trainable(self, enabled: bool) -> None:
        """Freeze everything except meta_encoder and meta_aux_head during warmup."""
        if enabled:
            for name, p in self.model.named_parameters():
                p.requires_grad = ("meta_encoder" in name) or ("meta_aux_head" in name)
        else:
            for name, p in self.model.named_parameters():
                if name in self._original_requires_grad:
                    p.requires_grad = self._original_requires_grad[name]

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        monitor_gbdt: bool = False,
        gbdt_interval: int = 1,
        gbdt_max_train_batches: Optional[int] = None,
        gbdt_max_val_batches: Optional[int] = None,
    ) -> Dict:
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            self.logger.info(f"Epoch {epoch}/{epochs}")

            meta_only_epoch = epoch <= self.meta_warmup_epochs
            if meta_only_epoch:
                self._set_meta_warmup_trainable(True)
                self.logger.info(
                    f"Meta-only warmup epoch {epoch}/{self.meta_warmup_epochs}: "
                    "training meta_encoder + meta_aux_head only."
                )
            elif self.meta_warmup_epochs > 0 and epoch == self.meta_warmup_epochs + 1:
                self._set_meta_warmup_trainable(False)
                self.logger.info("Meta warmup finished; restored original trainable parameters.")

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
                meta_only=meta_only_epoch,
            )

            val_loss, val_mae, val_spearman = validate(
                model=self.model,
                loader=val_loader,
                criterion=self.criterion,
                device=self.device,
                meta_only=meta_only_epoch,
                verbose_debug=False,
            )

            is_new_best = (not meta_only_epoch) and val_spearman > self.best_spearman
            is_early_stop_improvement = (
                (not meta_only_epoch)
                and val_spearman > self._early_stop_best + self.early_stop_min_delta
            )

            gbdt_scores = {}

            if monitor_gbdt and epoch % gbdt_interval == 0:
                gbdt_scores = run_gbdt_monitor(
                    model=self.model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=self.device,
                    run_lightgbm=True,
                    run_catboost=True,
                    max_train_batches=gbdt_max_train_batches,
                    max_val_batches=gbdt_max_val_batches,
                    save_dir=self.exp_dir / "gbdt_monitor",
                    best_lightgbm_spearman=float(getattr(self, "best_lightgbm_spearman", float("-inf"))),
                    best_catboost_spearman=float(getattr(self, "best_catboost_spearman", float("-inf"))),
                    epoch=epoch,
                )

                lgb_score = gbdt_scores.get("lightgbm_spearman", float("nan"))
                cat_score = gbdt_scores.get("catboost_spearman", float("nan"))
                if isinstance(lgb_score, (float, int)) and lgb_score == lgb_score:
                    self.best_lightgbm_spearman = max(
                        float(getattr(self, "best_lightgbm_spearman", float("-inf"))),
                        float(lgb_score),
                    )
                if isinstance(cat_score, (float, int)) and cat_score == cat_score:
                    self.best_catboost_spearman = max(
                        float(getattr(self, "best_catboost_spearman", float("-inf"))),
                        float(cat_score),
                    )

                self.logger.info(
                    "[GBDT Monitor] "
                    f"deep={gbdt_scores.get('deep_spearman', float('nan')):.4f} | "
                    f"lightgbm={gbdt_scores.get('lightgbm_spearman', float('nan')):.4f} | "
                    f"catboost={gbdt_scores.get('catboost_spearman', float('nan')):.4f}"
                )

                self.writer.add_scalar(
                    "metric_gbdt/deep_spearman",
                    gbdt_scores.get("deep_spearman", float("nan")),
                    epoch,
                )
                self.writer.add_scalar(
                    "metric_gbdt/lightgbm_spearman",
                    gbdt_scores.get("lightgbm_spearman", float("nan")),
                    epoch,
                )
                self.writer.add_scalar(
                    "metric_gbdt/catboost_spearman",
                    gbdt_scores.get("catboost_spearman", float("nan")),
                    epoch,
                )

            # ---------------------------------------------------------
            # Expensive modality ablation is only worth running when the
            # full validation score is already promising.
            # This avoids extra Valid[text/meta/image/user_desc_masked] passes
            # during weak early epochs or failed runs.
            # ---------------------------------------------------------
            ablation_results = None
            should_run_ablation = (
                (not meta_only_epoch)
                and val_spearman > self.ablation_spearman_threshold
                and ((not self.ablation_on_best_only) or is_new_best)
            )

            if should_run_ablation:
                self.logger.info(
                    f"val_spearman={val_spearman:.4f} > "
                    f"{self.ablation_spearman_threshold:.4f}; running modality ablation."
                )
                ablation_results = validate_modality_ablation(
                    model=self.model,
                    loader=val_loader,
                    criterion=self.criterion,
                    device=self.device,
                    verbose_debug=False,
                )
            else:
                if meta_only_epoch:
                    self.logger.info("Skip modality ablation during meta-only warmup.")
                elif self.ablation_on_best_only and not is_new_best:
                    self.logger.info("Skip modality ablation: current epoch is not a new best.")
                else:
                    self.logger.info(
                        f"Skip modality ablation: val_spearman={val_spearman:.4f} <= "
                        f"{self.ablation_spearman_threshold:.4f}"
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

            epoch_record = {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_mae": float(val_mae),
                "val_spearman": float(val_spearman),
                "epoch_time_sec": float(epoch_time),
                "ablation_ran": bool(ablation_results is not None),
                "meta_only_epoch": bool(meta_only_epoch),
            }

            if ablation_results is not None:
                epoch_record["modality_ablation"] = ablation_results
            if gbdt_scores:
                epoch_record["gbdt_monitor"] = to_jsonable(gbdt_scores)

            self.history.append(epoch_record)

            save_json({"history": self.history}, self.exp_dir / "history.json")

            save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                best_score=self.best_spearman,
                path=self.ckpt_dir / "latest.pt",
            )

            # Do not compare/save meta-only warmup epochs as best full-model checkpoints.
            # meta_only uses a different inference path (meta_feat -> meta_aux_head),
            # so its score is not directly comparable with full multimodal validation.
            if meta_only_epoch:
                self.logger.info("Skip saving best.pt during meta-only warmup.")
            elif is_new_best:
                self.best_spearman = val_spearman
                self.best_epoch = epoch
                self.best_ablation = ablation_results

                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    best_score=self.best_spearman,
                    path=self.ckpt_dir / "best.pt",
                )
                self.logger.info(f"Saved best model to {self.ckpt_dir / 'best.pt'}")

            if not meta_only_epoch and self.early_stop_patience is not None:
                if is_early_stop_improvement:
                    self._early_stop_best = val_spearman
                    self._epochs_without_improvement = 0
                else:
                    self._epochs_without_improvement += 1
                    self.logger.info(
                        "Early-stop patience: "
                        f"{self._epochs_without_improvement}/{self.early_stop_patience}"
                    )

                if self._epochs_without_improvement >= self.early_stop_patience:
                    self.stopped_early = True
                    self.stop_epoch = epoch
                    self.logger.info(
                        f"Early stopping at epoch {epoch}; "
                        f"best_val_spearman={self.best_spearman:.4f} "
                        f"at epoch {self.best_epoch}."
                    )
                    break

        self.writer.close()

        summary = {
            "exp_name": self.exp_name,
            "best_epoch": self.best_epoch,
            "best_val_spearman": float(self.best_spearman),
            "stopped_early": bool(self.stopped_early),
            "stop_epoch": self.stop_epoch,
            "best_modality_ablation": self.best_ablation,
        }
        save_json(summary, self.exp_dir / "summary.json")

        self.logger.info("Training finished.")
        self.logger.info(f"Best epoch: {self.best_epoch}")
        self.logger.info(f"Best val spearman: {self.best_spearman:.4f}")

        return summary
