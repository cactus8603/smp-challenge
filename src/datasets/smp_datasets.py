from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoTokenizer, CLIPImageProcessor

from src.datasets.metadata_preprocessor import MetadataPreprocessor


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
    """
    SMP dataset aligned with:
    - build_dataset outputs
    - MetadataPreprocessor transformed columns
    - CLIP-based text encoder pipeline
    - CLIP-based image encoder pipeline

    Expected workflow:
        pre = MetadataPreprocessor()
        train_df = pre.fit_transform(train_df)
        valid_df = pre.transform(valid_df)

        train_ds = SMPDataset(train_df, preprocessor=pre, use_image=True)
        valid_ds = SMPDataset(valid_df, preprocessor=pre, use_image=True)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        preprocessor: MetadataPreprocessor,
        text_model_name: str = "openai/clip-vit-base-patch32",
        image_model_name: str = "openai/clip-vit-base-patch32",
        max_length: int = 77,
        use_text: bool = True,
        use_meta: bool = True,
        use_image: bool = False,
        image_path_col: str = "image_path",
        image_root_dir: str | None = None,
        is_train: bool = True,
    ) -> None:
        if not preprocessor.fitted:
            raise ValueError("preprocessor must be fitted before building SMPDataset.")

        self.df = df.reset_index(drop=True).copy()
        self.image_root_dir = Path(image_root_dir) if image_root_dir else None
        self.preprocessor = preprocessor
        self.max_length = max_length
        self.use_text = use_text
        self.use_meta = use_meta
        self.use_image = use_image
        self.image_path_col = image_path_col
        self.is_train = is_train

        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name) if use_text else None
        self.image_processor = CLIPImageProcessor.from_pretrained(image_model_name) if use_image else None

        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer is not None else 0
        if self.pad_token_id is None and self.tokenizer is not None:
            self.pad_token_id = self.tokenizer.eos_token_id or 0

        self.num_cols = preprocessor.transformed_num_cols
        self.cat_cols = preprocessor.transformed_cat_cols
        self.bin_cols = preprocessor.transformed_bin_cols

        self.meta_num_dim = len(self.num_cols)
        self.meta_cat_dim = len(self.cat_cols)
        self.meta_bin_dim = len(self.bin_cols)

        required_cols = self.num_cols + self.cat_cols + self.bin_cols + [
            "title",
            "alltags",
            "full_text",
            "category",
            "subcategory",
            "concept",
            "label",
            "post_id",
            "Uid",
            "Pid",
        ]

        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(
                    f"Missing required column '{col}'. Did you call preprocessor.transform(df)?"
                )

        if self.use_image and self.image_path_col not in self.df.columns:
            raise ValueError(
                f"use_image=True but image path column '{self.image_path_col}' is missing."
            )

    def __len__(self) -> int:
        return len(self.df)

    def build_text(self, row: pd.Series) -> str:
        """
        Build text input for CLIP-style text encoding.

        Differences from the old BERT-style version:
        - avoid artificial [SEP] tokens
        - keep the text natural and short
        - still preserve useful metadata words (category / concept)
        """
        parts: List[str] = []

        title = safe_str(row.get("title", ""))
        alltags = safe_str(row.get("alltags", ""))
        full_text = safe_str(row.get("full_text", ""))
        category = safe_str(row.get("category", ""))
        subcategory = safe_str(row.get("subcategory", ""))
        concept = safe_str(row.get("concept", ""))

        if full_text:
            parts.append(full_text)
        else:
            if title:
                parts.append(title)
            if alltags:
                parts.append(f"tags: {alltags}")

        meta_parts: List[str] = []
        if category:
            meta_parts.append(category)
        if subcategory:
            meta_parts.append(subcategory)
        if concept:
            meta_parts.append(concept)

        if meta_parts:
            parts.append("topic: " + ", ".join(meta_parts))

        text = " | ".join([p for p in parts if p]).strip()
        return text

    def load_image(self, image_path: str) -> torch.Tensor:
        """
        Load and preprocess image for CLIP vision encoder.

        Returns:
            image_tensor: [3, H, W]
        """
        if self.image_processor is None:
            raise RuntimeError("image_processor is None but use_image=True.")

        image_path = safe_str(image_path, "")
        if not image_path:
            raise ValueError("Empty image path encountered.")

        path = Path(image_path)

        if self.image_root_dir is not None:
            path = self.image_root_dir / path
            
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")

        image = Image.open(path).convert("RGB")
        pixel_values = self.image_processor(images=image, return_tensors="pt")["pixel_values"]
        return pixel_values.squeeze(0)  # [3, H, W]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        item: Dict[str, Any] = {}

        # -------------------------
        # text
        # -------------------------
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

        # -------------------------
        # metadata
        # -------------------------
        if self.use_meta:
            item["meta_num"] = torch.tensor(
                row[self.num_cols].to_numpy(dtype="float32"),
                dtype=torch.float32,
            )
            item["meta_cat"] = torch.tensor(
                row[self.cat_cols].to_numpy(dtype="int64"),
                dtype=torch.long,
            )
            item["meta_bin"] = torch.tensor(
                row[self.bin_cols].to_numpy(dtype="float32"),
                dtype=torch.float32,
            )
        else:
            item["meta_num"] = torch.zeros(self.meta_num_dim, dtype=torch.float32)
            item["meta_cat"] = torch.zeros(self.meta_cat_dim, dtype=torch.long)
            item["meta_bin"] = torch.zeros(self.meta_bin_dim, dtype=torch.float32)

        # -------------------------
        # image
        # -------------------------
        if self.use_image:
            image_path = safe_str(row.get(self.image_path_col, ""), "")
            item["image_tensor"] = self.load_image(image_path)
            item["image_path"] = image_path
        else:
            item["image_tensor"] = torch.zeros(0, dtype=torch.float32)
            item["image_path"] = ""

        # -------------------------
        # labels / ids
        # -------------------------
        item["labels"] = torch.tensor(float(row.get("label", 0.0)), dtype=torch.float32)
        item["pad_token_id"] = self.pad_token_id
        item["post_id"] = safe_str(row.get("post_id", ""))
        item["uid"] = safe_str(row.get("Uid", ""))
        item["pid"] = safe_str(row.get("Pid", ""))

        return item


def smp_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    output: Dict[str, Any] = {}

    # -------------------------
    # text padding
    # -------------------------
    input_ids_list = [x["input_ids"] for x in batch]
    attention_mask_list = [x["attention_mask"] for x in batch]

    if len(input_ids_list) > 0 and input_ids_list[0].numel() > 0:
        pad_token_id = batch[0].get("pad_token_id", 0)
        max_len = max(x.size(0) for x in input_ids_list)

        padded_input_ids = []
        padded_attention_mask = []

        for input_ids, attention_mask in zip(input_ids_list, attention_mask_list):
            pad_len = max_len - input_ids.size(0)
            if pad_len > 0:
                input_ids = torch.cat(
                    [input_ids, torch.full((pad_len,), pad_token_id, dtype=torch.long)],
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

    # -------------------------
    # metadata
    # -------------------------
    output["meta_num"] = torch.stack([x["meta_num"] for x in batch], dim=0)
    output["meta_cat"] = torch.stack([x["meta_cat"] for x in batch], dim=0)
    output["meta_bin"] = torch.stack([x["meta_bin"] for x in batch], dim=0)

    # -------------------------
    # image
    # -------------------------
    if len(batch) > 0 and batch[0]["image_tensor"].numel() > 0:
        output["image_tensor"] = torch.stack([x["image_tensor"] for x in batch], dim=0)
    else:
        output["image_tensor"] = torch.zeros(0, dtype=torch.float32)

    # -------------------------
    # labels / ids
    # -------------------------
    output["labels"] = torch.stack([x["labels"] for x in batch], dim=0)
    output["post_id"] = [x["post_id"] for x in batch]
    output["uid"] = [x["uid"] for x in batch]
    output["pid"] = [x["pid"] for x in batch]
    output["image_path"] = [x.get("image_path", "") for x in batch]

    return output