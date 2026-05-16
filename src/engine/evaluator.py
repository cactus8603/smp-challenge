from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.metrics import compute_mae, compute_spearman


def _run_model_with_mask(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    meta_num: torch.Tensor,
    meta_cat: torch.Tensor,
    meta_bin: torch.Tensor,
    image_tensor: Optional[torch.Tensor],
    tag_tokens,
    tag_text,
    tag_token_count: Optional[torch.Tensor],
    user_desc: Optional[torch.Tensor] = None,
    loc_desc: Optional[torch.Tensor] = None,
    modality_mask: Optional[Dict[str, bool]] = None,
    meta_only: bool = False,
):
    return model(
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
        modality_mask=modality_mask,
        meta_only=meta_only,
    )


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: str,
    modality_mask: Optional[Dict[str, bool]] = None,
    verbose_debug: bool = True,
    meta_only: bool = False,
) -> Tuple[float, float, float]:
    """
    Validate model on one dataloader.

    Args:
        model: PyTorch model
        loader: validation dataloader
        criterion: loss function
        device: device string
        modality_mask:
            None -> full model
            {"text": True} -> zero out text feature after extraction
            {"meta": True} -> zero out meta feature after extraction
            {"image": True} -> zero out image feature after extraction
            {"user_desc": True} -> zero out user_desc auxiliary modality
        verbose_debug: whether to print prediction statistics

    Returns:
        avg_loss, mae, spearman
    """
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

    desc = "Valid[meta_only]" if meta_only else "Valid"
    if modality_mask is not None:
        masked_names = [k for k, v in modality_mask.items() if v]
        if len(masked_names) > 0:
            desc = f"Valid[{'+'.join(masked_names)}_masked]"

    progress = tqdm(loader, desc=desc, leave=False)

    for batch in progress:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        meta_num = batch["meta_num"].to(device)
        meta_cat = batch["meta_cat"].to(device)
        meta_bin = batch["meta_bin"].to(device)
        labels = batch["labels"].to(device)  # normalized labels if enabled

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

        user_desc     = batch["user_desc"].to(device) if "user_desc" in batch else None
        loc_desc      = batch["loc_desc"].to(device)  if "loc_desc"  in batch else None

        outputs = _run_model_with_mask(
            model=model,
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
            modality_mask=modality_mask,
            meta_only=meta_only,
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

    if verbose_debug:
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


@torch.no_grad()
def validate_modality_ablation(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: str,
    verbose_debug: bool = False,
):
    """
    Run validation multiple times:
    - full
    - mask_text
    - mask_meta
    - mask_image
    - mask_user_desc

    Returns:
        results dict with:
        {
            "full": {...},
            "mask_text": {...},
            "mask_meta": {...},
            "mask_image": {...},
            "mask_user_desc": {...},
            "drop": {
                "text": ...,
                "meta": ...,
                "image": ...,
                "user_desc": ...
            }
        }
    """
    settings = {
        "full": None,
        "mask_text": {"text": True},
        "mask_meta": {"meta": True},
        "mask_image": {"image": True},
        "mask_user_desc": {"user_desc": True},
    }

    results = {}

    for name, mask in settings.items():
        val_loss, val_mae, val_spearman = validate(
            model=model,
            loader=loader,
            criterion=criterion,
            device=device,
            modality_mask=mask,
            verbose_debug=verbose_debug,
        )
        results[name] = {
            "loss": val_loss,
            "mae": val_mae,
            "spearman": val_spearman,
        }

    base_spearman = results["full"]["spearman"]
    results["drop"] = {
        "text": base_spearman - results["mask_text"]["spearman"],
        "meta": base_spearman - results["mask_meta"]["spearman"],
        "image": base_spearman - results["mask_image"]["spearman"],
        "user_desc": base_spearman - results["mask_user_desc"]["spearman"],
    }

    print("\n=== Modality Ablation Summary ===")
    for k in ["full", "mask_text", "mask_meta", "mask_image", "mask_user_desc"]:
        print(
            f"{k:>10s} | "
            f"loss={results[k]['loss']:.4f} | "
            f"mae={results[k]['mae']:.4f} | "
            f"spearman={results[k]['spearman']:.4f}"
        )

    print("\n=== Spearman Drop From Full ===")
    print(f"{'text':>10s}: {results['drop']['text']:.4f}")
    print(f"{'meta':>10s}: {results['drop']['meta']:.4f}")
    print(f"{'image':>10s}: {results['drop']['image']:.4f}")
    print(f"{'user_desc':>10s}: {results['drop']['user_desc']:.4f}")

    return results