from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


def safe_float(x: Any, default: float = 0.0) -> float:
    if x is None:
        return default
    try:
        if pd.isna(x):
            return default
    except Exception:
        pass
    try:
        return float(x)
    except Exception:
        return default


def safe_int(x: Any, default: int = 0) -> int:
    if x is None:
        return default
    try:
        if pd.isna(x):
            return default
    except Exception:
        pass
    try:
        return int(x)
    except Exception:
        return default


def safe_str(x: Any, default: str = "") -> str:
    if x is None:
        return default
    try:
        if pd.isna(x):
            return default
    except Exception:
        pass
    x = str(x).strip()
    return x if x else default


class SMPDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        text_model_name: str = "distilbert-base-uncased",
        max_length: int = 128,
        use_text: bool = True,
        use_meta: bool = True,
        use_image: bool = False,
        is_train: bool = True,
    ) -> None:
        self.df = df.reset_index(drop=True).copy()
        self.max_length = max_length
        self.use_text = use_text
        self.use_meta = use_meta
        self.use_image = use_image
        self.is_train = is_train

        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name) if use_text else None

        # 你目前先用這些 metadata
        self.meta_columns = [
            "hour",
            "weekday",
            "is_weekend",
            "has_geo",
            "latitude",
            "longitude",
            "geoaccuracy",
            "photo_count",
            "ispro",
            "canbuypro",
            "timezone_offset",
            "timezone_id",
            "ispublic",
            "year",
            "month",
            "day",
        ]

        # 確保欄位存在
        for col in self.meta_columns:
            if col not in self.df.columns:
                self.df[col] = 0

        # 文本欄位預處理
        for col in ["title", "alltags", "category", "subcategory", "concept", "full_text"]:
            if col not in self.df.columns:
                self.df[col] = ""

        if "label" not in self.df.columns:
            self.df["label"] = 0.0

        self.meta_dim = len(self.meta_columns)

    def __len__(self) -> int:
        return len(self.df)

    def build_text(self, row: pd.Series) -> str:
        """
        把可用的文字欄位組起來
        """
        parts = []

        full_text = safe_str(row.get("full_text", ""))
        if full_text:
            parts.append(full_text)
        else:
            title = safe_str(row.get("title", ""))
            alltags = safe_str(row.get("alltags", ""))
            if title:
                parts.append(title)
            if alltags:
                parts.append(alltags)

        category = safe_str(row.get("category", ""))
        subcategory = safe_str(row.get("subcategory", ""))
        concept = safe_str(row.get("concept", ""))

        if category:
            parts.append(f"category: {category}")
        if subcategory:
            parts.append(f"subcategory: {subcategory}")
        if concept:
            parts.append(f"concept: {concept}")

        return " [SEP] ".join(parts).strip()

    def build_meta_features(self, row: pd.Series) -> np.ndarray:
        """
        先做最基本的 numeric metadata features
        之後可以再加 normalize / log transform
        """
        feats = []

        for col in self.meta_columns:
            value = row.get(col, 0)

            # 幾個簡單處理
            if col == "photo_count":
                # 大數值常常建議做 log1p
                value = np.log1p(max(safe_float(value, 0.0), 0.0))
            else:
                value = safe_float(value, 0.0)

            feats.append(value)

        return np.array(feats, dtype=np.float32)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]

        item: Dict[str, Any] = {}

        # --------
        # text
        # --------
        if self.use_text:
            text = self.build_text(row)
            encoded = self.tokenizer(
                text,
                truncation=True,
                padding=False,
                max_length=self.max_length,
                return_tensors=None,
            )
            item["input_ids"] = torch.tensor(encoded["input_ids"], dtype=torch.long)
            item["attention_mask"] = torch.tensor(encoded["attention_mask"], dtype=torch.long)
        else:
            item["input_ids"] = torch.tensor([], dtype=torch.long)
            item["attention_mask"] = torch.tensor([], dtype=torch.long)

        # --------
        # metadata
        # --------
        if self.use_meta:
            meta_features = self.build_meta_features(row)
            item["meta_features"] = torch.tensor(meta_features, dtype=torch.float32)
        else:
            item["meta_features"] = torch.zeros(self.meta_dim, dtype=torch.float32)

        # --------
        # image placeholder
        # --------
        # 現在沒有圖片，先統一回傳 None
        if self.use_image:
            item["image_tensor"] = None
        else:
            item["image_tensor"] = None

        # --------
        # label
        # --------
        label = safe_float(row.get("label", 0.0), 0.0)
        item["labels"] = torch.tensor(label, dtype=torch.float32)

        # --------
        # optional debug info
        # --------
        item["post_id"] = safe_str(row.get("post_id", ""))
        item["uid"] = safe_str(row.get("Uid", ""))
        item["pid"] = safe_str(row.get("Pid", ""))

        return item


def smp_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    專門處理 variable-length token sequence 的 batch
    """
    output: Dict[str, Any] = {}

    # -----------------
    # text padding
    # -----------------
    input_ids_list = [x["input_ids"] for x in batch]
    attention_mask_list = [x["attention_mask"] for x in batch]

    if len(input_ids_list) > 0 and input_ids_list[0].numel() > 0:
        max_len = max(x.size(0) for x in input_ids_list)

        padded_input_ids = []
        padded_attention_mask = []

        for input_ids, attention_mask in zip(input_ids_list, attention_mask_list):
            pad_len = max_len - input_ids.size(0)

            if pad_len > 0:
                input_ids = torch.cat(
                    [input_ids, torch.zeros(pad_len, dtype=torch.long)],
                    dim=0,
                )
                attention_mask = torch.cat(
                    [attention_mask, torch.zeros(pad_len, dtype=torch.long)],
                    dim=0,
                )

            padded_input_ids.append(input_ids)
            padded_attention_mask.append(attention_mask)

        output["input_ids"] = torch.stack(padded_input_ids, dim=0)
        output["attention_mask"] = torch.stack(padded_attention_mask, dim=0)
    else:
        batch_size = len(batch)
        output["input_ids"] = torch.zeros((batch_size, 1), dtype=torch.long)
        output["attention_mask"] = torch.zeros((batch_size, 1), dtype=torch.long)

    # -----------------
    # meta
    # -----------------
    output["meta_features"] = torch.stack([x["meta_features"] for x in batch], dim=0)

    # -----------------
    # image
    # -----------------
    # 目前先統一 None
    output["image_tensor"] = None

    # -----------------
    # labels
    # -----------------
    output["labels"] = torch.stack([x["labels"] for x in batch], dim=0)

    # -----------------
    # debug info
    # -----------------
    output["post_id"] = [x["post_id"] for x in batch]
    output["uid"] = [x["uid"] for x in batch]
    output["pid"] = [x["pid"] for x in batch]

    return output