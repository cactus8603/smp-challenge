from __future__ import annotations

from typing import Tuple

import torch
import numpy as np
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

    # raw scale for metrics / debug
    all_preds_raw = []
    all_labels_raw = []

    # try to read label normalization stats from dataset
    dataset = loader.dataset
    label_mean = getattr(dataset, "label_mean", None)
    label_std = getattr(dataset, "label_std", None)
    normalize_label = getattr(dataset, "normalize_label", False)

    progress = tqdm(loader, desc="Valid", leave=False)

    for batch in progress:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        meta_num = batch["meta_num"].to(device)
        meta_cat = batch["meta_cat"].to(device)
        meta_bin = batch["meta_bin"].to(device)
        labels = batch["labels"].to(device)   # normalized labels if enabled

        image_tensor = batch.get("image_tensor", None)
        if image_tensor is not None and image_tensor.numel() > 0:
            image_tensor = image_tensor.to(device)
        else:
            image_tensor = None

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

        preds = outputs.squeeze(-1)  # normalized prediction space
        loss = criterion(preds, labels)

        total_loss += loss.item()

        # keep normalized copies if you still want to inspect them
        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())

        # inverse transform for real-scale metrics
        preds_cpu = preds.detach().cpu()
        labels_cpu = labels.detach().cpu()

        if normalize_label:
            if label_mean is None or label_std is None:
                raise ValueError(
                    "Dataset has normalize_label=True but label_mean/label_std is missing."
                )
            preds_raw = preds_cpu * label_std + label_mean
            labels_raw = labels_cpu * label_std + label_mean
        else:
            preds_raw = preds_cpu
            labels_raw = labels_cpu

        all_preds_raw.extend(preds_raw.tolist())
        all_labels_raw.extend(labels_raw.tolist())

        progress.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / max(len(loader), 1)

    # use raw-scale values for reporting metrics
    mae = compute_mae(all_labels_raw, all_preds_raw)
    spearman = compute_spearman(all_labels_raw, all_preds_raw)

    print("=== normalized space ===")
    print("pred std:", np.std(all_preds))
    print("label std:", np.std(all_labels))
    print("pred sample:", all_preds[:10])
    print("label sample:", all_labels[:10])
    print("pred mean:", np.mean(all_preds))
    print("label mean:", np.mean(all_labels))
    print("pred min/max:", np.min(all_preds), np.max(all_preds))

    print("=== raw space ===")
    print("pred std:", np.std(all_preds_raw))
    print("label std:", np.std(all_labels_raw))
    print("pred sample:", all_preds_raw[:10])
    print("label sample:", all_labels_raw[:10])
    print("pred mean:", np.mean(all_preds_raw))
    print("label mean:", np.mean(all_labels_raw))
    print("pred min/max:", np.min(all_preds_raw), np.max(all_preds_raw))

    return avg_loss, mae, spearman