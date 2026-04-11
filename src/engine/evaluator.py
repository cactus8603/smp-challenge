from __future__ import annotations

from typing import Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.metrics import compute_mae, compute_spearman


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: str,
) -> Tuple[float, float, float]:
    model.eval()
    total_loss = 0.0

    all_preds = []
    all_labels = []

    progress = tqdm(loader, desc="Valid", leave=False)

    for batch in progress:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        meta_num = batch["meta_num"].to(device)
        meta_cat = batch["meta_cat"].to(device)
        meta_bin = batch["meta_bin"].to(device)
        labels = batch["labels"].to(device)

        image_tensor = batch.get("image_tensor", None)
        # Keep placeholder image disabled unless it has real content
        if image_tensor is not None and image_tensor.numel() > 0:
            image_tensor = image_tensor.to(device)
        else:
            image_tensor = None

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            meta_num=meta_num,
            meta_cat=meta_cat,
            meta_bin=meta_bin,
            image_tensor=image_tensor,
        )

        preds = outputs.squeeze(-1)
        loss = criterion(preds, labels)

        total_loss += loss.item()
        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())

        progress.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / max(len(loader), 1)
    mae = compute_mae(all_labels, all_preds)
    spearman = compute_spearman(all_labels, all_preds)

    return avg_loss, mae, spearman
