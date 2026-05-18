from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Dict, List, Sequence

import numpy as np
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


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(x):
            return default
    except Exception:
        pass
    try:
        return float(x)
    except Exception:
        return default


class SMPDataset(Dataset):
    """
    SMP dataset aligned with:
    - build_dataset outputs
    - MetadataPreprocessor transformed columns
    - CLIP-based text encoder pipeline
    - CLIP-based image encoder pipeline

    This version explicitly separates:
    - CLIP text path: short, token-budgeted text
    - tag text path: cleaned lexical tokens
    """

    def __init__(
        self,
        df: pd.DataFrame,
        preprocessor: MetadataPreprocessor,
        text_model_name: str = "openai/clip-vit-base-patch32",
        image_model_name: str = "openai/clip-vit-base-patch32",
        label_mean: float | None = None,
        label_std: float | None = None,
        normalize_label: bool = False,
        max_length: int = 77,
        use_text: bool = True,
        use_meta: bool = True,
        use_image: bool = False,
        image_path_col: str = "image_path",
        image_root_dir: str | None = None,
        is_train: bool = True,
        # caption handling for CLIP text
        use_caption: bool = False,
        caption_max_freq: int | None = 50,
        # user_description sentence embeddings
        user_desc_emb_path: str | None = None,   # path to .npy file
        user_desc_idx_path: str | None = None,   # path to uid_index .json
        cache_static_fields: bool = False,
    ) -> None:
        if not preprocessor.fitted:
            raise ValueError("preprocessor must be fitted before building SMPDataset.")

        self.df = df.reset_index(drop=True).copy()
        self.image_root_dir = Path(image_root_dir) if image_root_dir else None
        self.preprocessor = preprocessor
        self.label_mean = label_mean
        self.label_std = label_std
        self.normalize_label = normalize_label
        self.max_length = max_length
        self.use_text = use_text
        self.use_meta = use_meta
        self.use_image = use_image
        self.image_path_col = image_path_col
        self.is_train = is_train
        self.use_caption = bool(use_caption)
        self.caption_max_freq = caption_max_freq
        self.cache_static_fields = bool(cache_static_fields)

        if use_text:
            print(f"[SMPDataset] loading text tokenizer: {text_model_name}", flush=True)
            self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        else:
            self.tokenizer = None

        if use_image:
            print(f"[SMPDataset] loading image processor: {image_model_name}", flush=True)
            self.image_processor = CLIPImageProcessor.from_pretrained(image_model_name)
        else:
            self.image_processor = None

        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer is not None else 0
        if self.pad_token_id is None and self.tokenizer is not None:
            self.pad_token_id = self.tokenizer.eos_token_id or 0

        self.num_cols = preprocessor.transformed_num_cols
        self.cat_cols = preprocessor.transformed_cat_cols
        self.bin_cols = preprocessor.transformed_bin_cols

        self.meta_num_dim = len(self.num_cols)
        self.meta_cat_dim = len(self.cat_cols)
        self.meta_bin_dim = len(self.bin_cols)

        # ── optional caption filter for CLIP text ─────────────────────
        self._generic_captions: set[str] = set()
        if self.use_caption and self.caption_max_freq is not None and "caption" in self.df.columns:
            cap = self.df["caption"].map(lambda x: safe_str(x, ""))
            freq = cap[cap != ""].value_counts()
            self._generic_captions = set(freq[freq > int(self.caption_max_freq)].index.tolist())
            print(
                f"[SMPDataset] Caption filter: {len(self._generic_captions)} generic captions "
                f"(freq > {self.caption_max_freq}) will be skipped."
            )

        # ── user_description sentence embeddings ──────────────────────
        self.user_desc_embeddings: np.ndarray | None = None
        self.user_desc_uid_index: dict | None = None
        self.user_desc_emb_dim: int = 0
        self.user_desc_feature_dim: int = 0

        if user_desc_emb_path is not None and user_desc_idx_path is not None:
            import numpy as np
            import json
            emb_path = Path(user_desc_emb_path)
            idx_path = Path(user_desc_idx_path)
            if emb_path.exists() and idx_path.exists():
                self.user_desc_embeddings = np.load(str(emb_path))
                self.user_desc_uid_index  = json.loads(idx_path.read_text(encoding="utf-8"))
                self.user_desc_emb_dim    = self.user_desc_embeddings.shape[1]
                n_with_emb = sum(
                    1 for uid in self.df["Uid"].astype(str)
                    if uid in self.user_desc_uid_index
                )
                print(f"[SMPDataset] Loaded user_desc embeddings: "
                      f"shape={self.user_desc_embeddings.shape}, "
                      f"coverage={n_with_emb}/{len(self.df)} rows")
            else:
                print(f"[SMPDataset] WARNING: user_desc_emb_path or idx_path not found, "
                      f"user_desc embeddings disabled.")

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

        self.user_desc_feature_dim = int(self.user_desc_emb_dim or self.preprocessor.user_desc_dim)

        if self.cache_static_fields:
            self._init_cached_fields()
        else:
            print(
                "[SMPDataset] static field RAM cache disabled; "
                "items will be built on demand.",
                flush=True,
            )

    def __len__(self) -> int:
        return len(self.df)

    # ------------------------------------------------------------------
    # text helpers
    # ------------------------------------------------------------------
    def split_tags(self, value: Any) -> List[str]:
        value = safe_str(value, "")
        if not value:
            return []
        items = re.split(r"[\s,;|]+", value)
        return [x.strip() for x in items if x and x.strip()]

    def simple_tokenize(self, text: Any) -> List[str]:
        text = safe_str(text, "").lower()
        if not text:
            return []
        return re.findall(r"[a-z0-9_@#'\-]+", text)

    def dedup_preserve_order(self, items: List[str], lowercase_key: bool = True) -> List[str]:
        seen = set()
        out: List[str] = []
        for x in items:
            key = x.lower() if lowercase_key else x
            if key in seen:
                continue
            seen.add(key)
            out.append(x)
        return out

    def select_tags(
        self,
        alltags: Any,
        max_tags: int = 10,
        min_len: int = 2,
    ) -> List[str]:
        raw_tags = self.split_tags(alltags)
        raw_tags = self.dedup_preserve_order(raw_tags, lowercase_key=True)

        selected: List[str] = []
        for tag in raw_tags:
            t = tag.strip()
            if len(t) < min_len:
                continue
            if t.isdigit():
                continue
            selected.append(t)
            if len(selected) >= max_tags:
                break
        return selected

    def is_valid_tag_token(self, token: str) -> bool:
        token = safe_str(token, "").lower()
        if not token:
            return False
        if len(token) < 2:
            return False
        if token in {"-", "_", "|", "by"}:
            return False
        if token.isdigit():
            return False
        if not re.fullmatch(r"[a-z0-9#@_\-]+", token):
            return False
        return True

    def count_clip_tokens(self, text: str) -> int:
        if self.tokenizer is None:
            return 0
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding=False,
            max_length=77,
            return_tensors=None,
        )
        return len(encoded["input_ids"])

    def _get_caption_text(self, row: pd.Series) -> str:
        """Return a usable caption string, or empty if disabled/generic/missing."""
        if not self.use_caption:
            return ""
        caption = safe_str(row.get("caption", ""))
        if not caption:
            return ""
        if caption in self._generic_captions:
            return ""
        return caption

    def _append_part_if_fits(
        self,
        parts: List[str],
        new_part: str,
        max_tokens: int,
    ) -> List[str]:
        """Append a CLIP text part only if it stays inside the token budget."""
        if not new_part:
            return parts
        candidate = " | ".join([p for p in parts + [new_part] if p]).strip()
        if self.count_clip_tokens(candidate) <= max_tokens:
            return parts + [new_part]
        return parts

    def build_clip_text(
        self,
        row: pd.Series,
        max_tags: int = 10,
        max_tokens: int = 77,
    ) -> str:
        """
        Build a short, information-dense text for CLIP.
        Explicitly keep the final text within CLIP's 77-token limit.
        """
        title = safe_str(row.get("title", ""))
        category = safe_str(row.get("category", ""))
        subcategory = safe_str(row.get("subcategory", ""))
        concept = safe_str(row.get("concept", ""))
        caption = self._get_caption_text(row)

        tags = self.select_tags(row.get("alltags", ""), max_tags=max_tags)

        topic_parts = [x for x in [category, subcategory, concept] if x]
        topic_parts = self.dedup_preserve_order(topic_parts, lowercase_key=False)
        topic_text = "topic: " + ", ".join(topic_parts) if topic_parts else ""

        kept_tags: List[str] = []
        for tag in tags:
            candidate_tags = kept_tags + [tag]
            parts: List[str] = []
            if title:
                parts.append(title)
            if candidate_tags:
                parts.append("tags: " + " ".join(candidate_tags))
            if topic_text:
                parts.append(topic_text)

            candidate_text = " | ".join([p for p in parts if p]).strip()
            if self.count_clip_tokens(candidate_text) <= max_tokens:
                kept_tags = candidate_tags
            else:
                break

        final_parts: List[str] = []
        if title:
            final_parts.append(title)
        if kept_tags:
            final_parts.append("tags: " + " ".join(kept_tags))
        if topic_text:
            final_parts.append(topic_text)
        final_parts = self._append_part_if_fits(final_parts, f"caption: {caption}" if caption else "", max_tokens)

        final_text = " | ".join([p for p in final_parts if p]).strip()

        # If still too long, shrink title from the end while keeping tags/topic.
        if self.count_clip_tokens(final_text) > max_tokens and title:
            title_words = title.split()
            while len(title_words) > 1:
                title_words = title_words[:-1]
                shrink_parts: List[str] = [" ".join(title_words)]
                if kept_tags:
                    shrink_parts.append("tags: " + " ".join(kept_tags))
                if topic_text:
                    shrink_parts.append(topic_text)
                shrink_parts = self._append_part_if_fits(shrink_parts, f"caption: {caption}" if caption else "", max_tokens)
                candidate_text = " | ".join([p for p in shrink_parts if p]).strip()
                if self.count_clip_tokens(candidate_text) <= max_tokens:
                    final_text = candidate_text
                    break

        # Final safety fallback.
        if self.tokenizer is not None and self.count_clip_tokens(final_text) > max_tokens:
            encoded = self.tokenizer(
                final_text,
                truncation=True,
                padding=False,
                max_length=77,
                return_tensors=None,
            )
            final_text = self.tokenizer.decode(encoded["input_ids"], skip_special_tokens=True).strip()

        return final_text

    def build_tag_tokens(
        self,
        row: pd.Series,
        max_title_tokens: int = 24,
        max_tag_tokens: int = 20,
        max_topic_tokens: int = 8,
    ) -> List[str]:
        """
        Build a cleaned token list for future tag / lexical features.
        This is separate from CLIP text.
        """
        title = safe_str(row.get("title", ""))
        category = safe_str(row.get("category", ""))
        subcategory = safe_str(row.get("subcategory", ""))
        concept = safe_str(row.get("concept", ""))

        title_tokens = [t for t in self.simple_tokenize(title) if self.is_valid_tag_token(t)][:max_title_tokens]
        tag_tokens = [
            t.lower()
            for t in self.select_tags(row.get("alltags", ""), max_tags=max_tag_tokens)
            if self.is_valid_tag_token(t.lower())
        ]
        topic_text = " ".join(
            self.dedup_preserve_order([x for x in [category, subcategory, concept] if x], lowercase_key=False)
        )
        topic_tokens = [t for t in self.simple_tokenize(topic_text) if self.is_valid_tag_token(t)][:max_topic_tokens]

        tokens = self.dedup_preserve_order(title_tokens + tag_tokens + topic_tokens, lowercase_key=True)
        return tokens

    def build_tag_text(self, row: pd.Series) -> str:
        return " ".join(self.build_tag_tokens(row))

    def build_text(self, row: pd.Series) -> str:
        """
        Backward-compatible alias for the CLIP text path.
        """
        return self.build_clip_text(row)

    # ------------------------------------------------------------------
    # image
    # ------------------------------------------------------------------
    def load_image(self, image_path: str) -> torch.Tensor:
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
        return pixel_values.squeeze(0)

    def _init_cached_fields(self) -> None:
        """
        Precompute deterministic per-row work once per dataset/worker.

        The old __getitem__ rebuilt CLIP text, tag tokens, metadata tensors, and
        user-desc vectors for every epoch. Those values do not change during
        training, so caching them cuts a large CPU/DataLoader bottleneck.
        """
        n = len(self.df)
        print(
            f"[SMPDataset] caching static fields: rows={n} | "
            f"text={self.use_text} | meta={self.use_meta} | image={self.use_image}",
            flush=True,
        )
        self._clip_text_cache: List[str] = [""] * n
        self._tag_tokens_cache: List[List[str]] = [[] for _ in range(n)]
        self._tag_text_cache: List[str] = [""] * n
        self._input_ids_cache: List[torch.Tensor] = [torch.empty(0, dtype=torch.long) for _ in range(n)]
        self._attention_mask_cache: List[torch.Tensor] = [torch.empty(0, dtype=torch.long) for _ in range(n)]
        self._clip_token_count_cache = torch.zeros(n, dtype=torch.long)
        self._clip_was_truncated_cache = torch.zeros(n, dtype=torch.long)

        progress_every = 50000
        for idx, row in self.df.iterrows():
            if idx > 0 and idx % progress_every == 0:
                print(f"[SMPDataset] cached static fields: {idx}/{n}", flush=True)

            clip_text = self.build_clip_text(row)
            tag_tokens = self.build_tag_tokens(row)
            self._clip_text_cache[idx] = clip_text
            self._tag_tokens_cache[idx] = tag_tokens
            self._tag_text_cache[idx] = " ".join(tag_tokens)

            if self.use_text:
                encoded = self.tokenizer(
                    clip_text,
                    truncation=True,
                    padding=False,
                    max_length=77,
                    return_tensors=None,
                )
                input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long)
                attention_mask = torch.tensor(encoded["attention_mask"], dtype=torch.long)
                self._input_ids_cache[idx] = input_ids
                self._attention_mask_cache[idx] = attention_mask
                self._clip_token_count_cache[idx] = input_ids.numel()
                self._clip_was_truncated_cache[idx] = int(input_ids.numel() >= 77)

        if self.use_meta:
            self._meta_num_cache = torch.tensor(
                self.df[self.num_cols].to_numpy(dtype="float32"),
                dtype=torch.float32,
            )
            self._meta_cat_cache = torch.tensor(
                self.df[self.cat_cols].to_numpy(dtype="int64"),
                dtype=torch.long,
            )
            self._meta_bin_cache = torch.tensor(
                self.df[self.bin_cols].to_numpy(dtype="float32"),
                dtype=torch.float32,
            )
        else:
            self._meta_num_cache = torch.zeros((n, self.meta_num_dim), dtype=torch.float32)
            self._meta_cat_cache = torch.zeros((n, self.meta_cat_dim), dtype=torch.long)
            self._meta_bin_cache = torch.zeros((n, self.meta_bin_dim), dtype=torch.float32)

        user_desc_dim = self.user_desc_feature_dim
        self._user_desc_cache = torch.zeros((n, user_desc_dim), dtype=torch.float32)
        if self.user_desc_embeddings is not None and self.user_desc_uid_index is not None and user_desc_dim > 0:
            for idx, uid in enumerate(self.df["Uid"].astype(str)):
                emb_idx = self.user_desc_uid_index.get(uid)
                if emb_idx is not None:
                    self._user_desc_cache[idx] = torch.from_numpy(
                        np.asarray(self.user_desc_embeddings[emb_idx], dtype=np.float32)
                    )

        self._loc_desc_cache = torch.zeros((n, self.preprocessor.loc_desc_dim), dtype=torch.float32)
        labels_raw = pd.to_numeric(self.df["label"], errors="coerce").fillna(0.0).to_numpy(dtype="float32")
        self._label_raw_cache = torch.tensor(labels_raw, dtype=torch.float32)
        if self.normalize_label:
            if self.label_mean is None or self.label_std is None:
                raise ValueError("normalize_label=True but label_mean/std is None")
            labels = (labels_raw - float(self.label_mean)) / (float(self.label_std) + 1e-8)
        else:
            labels = labels_raw
        self._labels_cache = torch.tensor(labels, dtype=torch.float32)
        print(f"[SMPDataset] cached static fields: done ({n}/{n})", flush=True)

    # ------------------------------------------------------------------
    # debug
    # ------------------------------------------------------------------
    def get_debug_row(self, idx: int, text_preview_chars: int = 240) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        clip_text = self.build_clip_text(row)
        clip_token_count_raw = self.count_clip_tokens(clip_text)
        tag_tokens = self.build_tag_tokens(row)
        tag_text = " ".join(tag_tokens)

        debug_row: Dict[str, Any] = {
            "idx": idx,
            "post_id": safe_str(row.get("post_id", "")),
            "uid": safe_str(row.get("Uid", "")),
            "pid": safe_str(row.get("Pid", "")),
            "label": _safe_float(row.get("label", 0.0)),
            "title": safe_str(row.get("title", "")),
            "alltags": safe_str(row.get("alltags", "")),
            "caption_preview": safe_str(row.get("caption", ""))[:text_preview_chars],
            "caption_used": int(bool(self._get_caption_text(row))),
            "full_text_preview": safe_str(row.get("full_text", ""))[:text_preview_chars],
            "clip_text_preview": clip_text[:text_preview_chars],
            "clip_text_len_chars": len(clip_text),
            "clip_token_count_raw": clip_token_count_raw,
            "clip_was_truncated": int(clip_token_count_raw > 77),
            "tag_text_preview": tag_text[:text_preview_chars],
            "tag_token_count": len(tag_tokens),
            "tag_tokens_preview": tag_tokens[:20],
        }

        if self.use_meta:
            num_values = row[self.num_cols].to_numpy(dtype="float32")
            cat_values = row[self.cat_cols].to_numpy(dtype="int64")
            bin_values = row[self.bin_cols].to_numpy(dtype="float32")

            debug_row.update(
                {
                    "meta_num_dim": int(num_values.shape[0]),
                    "meta_num_mean": float(num_values.mean()) if num_values.size else 0.0,
                    "meta_num_std": float(num_values.std()) if num_values.size else 0.0,
                    "meta_num_min": float(num_values.min()) if num_values.size else 0.0,
                    "meta_num_max": float(num_values.max()) if num_values.size else 0.0,
                    "meta_cat_dim": int(cat_values.shape[0]),
                    "meta_cat_unique_count": int(len(set(cat_values.tolist()))) if cat_values.size else 0,
                    "meta_bin_dim": int(bin_values.shape[0]),
                    "meta_bin_sum": float(bin_values.sum()) if bin_values.size else 0.0,
                }
            )

        if self.use_image:
            debug_row["image_path"] = safe_str(row.get(self.image_path_col, ""), "")

        return debug_row

    # ------------------------------------------------------------------
    # dataset item
    # ------------------------------------------------------------------
    def _encode_text_item(self, clip_text: str) -> Dict[str, Any]:
        if not self.use_text:
            return {
                "input_ids": torch.tensor([], dtype=torch.long),
                "attention_mask": torch.tensor([], dtype=torch.long),
                "raw_text": "",
                "clip_text": "",
                "clip_token_count_raw": torch.tensor(0, dtype=torch.long),
                "clip_token_count": torch.tensor(0, dtype=torch.long),
                "clip_was_truncated": torch.tensor(0, dtype=torch.long),
            }

        encoded = self.tokenizer(
            clip_text,
            truncation=True,
            padding=False,
            max_length=77,
            return_tensors=None,
        )
        input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(encoded["attention_mask"], dtype=torch.long)
        token_count = torch.tensor(input_ids.numel(), dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "raw_text": clip_text,
            "clip_text": clip_text,
            "clip_token_count_raw": token_count,
            "clip_token_count": token_count,
            "clip_was_truncated": torch.tensor(int(input_ids.numel() >= 77), dtype=torch.long),
        }

    def _get_user_desc_tensor(self, uid: Any) -> torch.Tensor:
        x = torch.zeros(self.user_desc_feature_dim, dtype=torch.float32)
        if self.user_desc_embeddings is None or self.user_desc_uid_index is None or self.user_desc_feature_dim <= 0:
            return x

        emb_idx = self.user_desc_uid_index.get(safe_str(uid, ""))
        if emb_idx is None:
            return x
        return torch.from_numpy(np.asarray(self.user_desc_embeddings[emb_idx], dtype=np.float32))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        item: Dict[str, Any] = {}

        if self.cache_static_fields:
            if self.use_text:
                item["input_ids"] = self._input_ids_cache[idx]
                item["attention_mask"] = self._attention_mask_cache[idx]
                item["raw_text"] = self._clip_text_cache[idx]
                item["clip_text"] = self._clip_text_cache[idx]
                item["clip_token_count_raw"] = self._clip_token_count_cache[idx]
                item["clip_token_count"] = self._clip_token_count_cache[idx]
                item["clip_was_truncated"] = self._clip_was_truncated_cache[idx]
            else:
                item.update(self._encode_text_item(""))
        elif self.use_text:
            item.update(self._encode_text_item(self.build_clip_text(row)))
        else:
            item.update(self._encode_text_item(""))

        if self.cache_static_fields:
            tag_tokens = self._tag_tokens_cache[idx]
            tag_text = self._tag_text_cache[idx]
        else:
            tag_tokens = self.build_tag_tokens(row)
            tag_text = " ".join(tag_tokens)

        item["tag_text"] = tag_text
        item["tag_tokens"] = tag_tokens
        item["tag_token_count"] = torch.tensor(len(tag_tokens), dtype=torch.long)

        # -------------------------
        # metadata
        # -------------------------
        if self.cache_static_fields:
            item["meta_num"] = self._meta_num_cache[idx]
            item["meta_cat"] = self._meta_cat_cache[idx]
            item["meta_bin"] = self._meta_bin_cache[idx]
            item["user_desc"] = self._user_desc_cache[idx]
            item["loc_desc"] = self._loc_desc_cache[idx]
        elif self.use_meta:
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
            item["user_desc"] = self._get_user_desc_tensor(row.get("Uid", ""))
            item["loc_desc"] = torch.zeros(self.preprocessor.loc_desc_dim, dtype=torch.float32)
        else:
            item["meta_num"] = torch.zeros(self.meta_num_dim, dtype=torch.float32)
            item["meta_cat"] = torch.zeros(self.meta_cat_dim, dtype=torch.long)
            item["meta_bin"] = torch.zeros(self.meta_bin_dim, dtype=torch.float32)
            item["user_desc"] = torch.zeros(self.user_desc_feature_dim, dtype=torch.float32)
            item["loc_desc"] = torch.zeros(self.preprocessor.loc_desc_dim, dtype=torch.float32)

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
        if self.cache_static_fields:
            item["label_raw"] = self._label_raw_cache[idx]
            item["labels"] = self._labels_cache[idx]
        else:
            label_raw = torch.tensor(_safe_float(row.get("label", 0.0)), dtype=torch.float32)
            item["label_raw"] = label_raw
            if self.normalize_label:
                if self.label_mean is None or self.label_std is None:
                    raise ValueError("normalize_label=True but label_mean/std is None")
                item["labels"] = (label_raw - float(self.label_mean)) / (float(self.label_std) + 1e-8)
            else:
                item["labels"] = label_raw
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
    output["user_desc"] = torch.stack([x["user_desc"] for x in batch], dim=0)
    output["loc_desc"]  = torch.stack([x["loc_desc"]  for x in batch], dim=0)

    # -------------------------
    # image
    # -------------------------
    if len(batch) > 0 and batch[0]["image_tensor"].numel() > 0:
        output["image_tensor"] = torch.stack([x["image_tensor"] for x in batch], dim=0)
    else:
        output["image_tensor"] = torch.zeros(0, dtype=torch.float32)

    # -------------------------
    # labels / ids / texts
    # -------------------------
    output["labels"] = torch.stack([x["labels"] for x in batch], dim=0)
    output["label_raw"] = torch.stack([x["label_raw"] for x in batch], dim=0)
    output["post_id"] = [x["post_id"] for x in batch]
    output["uid"] = [x["uid"] for x in batch]
    output["pid"] = [x["pid"] for x in batch]
    output["image_path"] = [x.get("image_path", "") for x in batch]
    output["raw_text"] = [x.get("raw_text", "") for x in batch]
    output["clip_text"] = [x.get("clip_text", "") for x in batch]
    output["clip_token_count_raw"] = torch.stack([x["clip_token_count_raw"] for x in batch], dim=0)
    output["clip_token_count"] = torch.stack([x["clip_token_count"] for x in batch], dim=0)
    output["clip_was_truncated"] = torch.stack([x["clip_was_truncated"] for x in batch], dim=0)
    output["tag_text"] = [x.get("tag_text", "") for x in batch]
    output["tag_tokens"] = [x.get("tag_tokens", []) for x in batch]
    output["tag_token_count"] = torch.stack([x["tag_token_count"] for x in batch], dim=0)

    return output


def debug_id_alignment(
    df: pd.DataFrame,
    post_id_col: str = "post_id",
    pid_col: str = "Pid",
    uid_col: str = "Uid",
    label_col: str = "label",
    max_rows: int = 10,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "num_rows": int(len(df)),
        "post_id_col": post_id_col,
        "pid_col": pid_col,
        "uid_col": uid_col,
        "label_col": label_col,
    }

    if post_id_col in df.columns and pid_col in df.columns:
        post_id_series = df[post_id_col].astype(str)
        pid_series = df[pid_col].astype(str)
        result["post_id_equals_pid_ratio"] = float((post_id_series == pid_series).mean())
        mismatch_df = df.loc[
            post_id_series != pid_series,
            [post_id_col, pid_col, uid_col, label_col],
        ].head(max_rows)
        result["post_id_pid_mismatch_preview"] = mismatch_df.to_dict(orient="records")
    else:
        result["post_id_equals_pid_ratio"] = None
        result["post_id_pid_mismatch_preview"] = []

    for col in [post_id_col, pid_col, uid_col]:
        if col in df.columns:
            result[f"{col}_nunique"] = int(df[col].nunique(dropna=False))

    if label_col in df.columns:
        result["label_mean"] = float(df[label_col].mean())
        result["label_std"] = float(df[label_col].std())
        result["label_min"] = float(df[label_col].min())
        result["label_max"] = float(df[label_col].max())

    return result


def debug_dataset_samples(
    dataset: SMPDataset,
    indices: Sequence[int] | None = None,
    num_samples: int = 5,
    text_preview_chars: int = 240,
) -> List[Dict[str, Any]]:
    if len(dataset) == 0:
        return []

    if indices is None:
        upper = min(len(dataset), num_samples)
        indices = list(range(upper))

    results: List[Dict[str, Any]] = []
    for idx in indices:
        if idx < 0 or idx >= len(dataset):
            continue
        results.append(dataset.get_debug_row(idx=idx, text_preview_chars=text_preview_chars))
    return results


def debug_batch(
    batch: Dict[str, Any],
    model: torch.nn.Module | None = None,
    device: str | torch.device | None = None,
    max_items: int = 10,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    meta_num = batch["meta_num"]
    meta_cat = batch["meta_cat"]
    meta_bin = batch["meta_bin"]
    labels = batch["labels"]

    result["input_ids_shape"] = tuple(input_ids.shape)
    result["attention_mask_shape"] = tuple(attention_mask.shape)
    result["meta_num_shape"] = tuple(meta_num.shape)
    result["meta_cat_shape"] = tuple(meta_cat.shape)
    result["meta_bin_shape"] = tuple(meta_bin.shape)
    result["labels_shape"] = tuple(labels.shape)

    result["meta_num_mean"] = float(meta_num.mean().item()) if meta_num.numel() > 0 else 0.0
    result["meta_num_std"] = float(meta_num.std().item()) if meta_num.numel() > 1 else 0.0
    result["meta_num_min"] = float(meta_num.min().item()) if meta_num.numel() > 0 else 0.0
    result["meta_num_max"] = float(meta_num.max().item()) if meta_num.numel() > 0 else 0.0

    result["meta_bin_mean"] = float(meta_bin.mean().item()) if meta_bin.numel() > 0 else 0.0
    result["meta_bin_std"] = float(meta_bin.std().item()) if meta_bin.numel() > 1 else 0.0
    result["meta_cat_min"] = int(meta_cat.min().item()) if meta_cat.numel() > 0 else 0
    result["meta_cat_max"] = int(meta_cat.max().item()) if meta_cat.numel() > 0 else 0
    result["meta_cat_unique_count"] = int(torch.unique(meta_cat).numel()) if meta_cat.numel() > 0 else 0

    result["labels_mean"] = float(labels.mean().item()) if labels.numel() > 0 else 0.0
    result["labels_std"] = float(labels.std().item()) if labels.numel() > 1 else 0.0
    result["labels_min"] = float(labels.min().item()) if labels.numel() > 0 else 0.0
    result["labels_max"] = float(labels.max().item()) if labels.numel() > 0 else 0.0

    max_items = min(max_items, labels.shape[0])
    sample_rows: List[Dict[str, Any]] = []
    for i in range(max_items):
        row: Dict[str, Any] = {
            "i": i,
            "post_id": batch["post_id"][i] if "post_id" in batch else "",
            "uid": batch["uid"][i] if "uid" in batch else "",
            "pid": batch["pid"][i] if "pid" in batch else "",
            "label": float(labels[i].item()),
            "text_preview": batch["raw_text"][i][:180] if "raw_text" in batch else "",
            "clip_token_count_raw": int(batch["clip_token_count_raw"][i].item()) if "clip_token_count_raw" in batch else 0,
            "clip_token_count": int(batch["clip_token_count"][i].item()) if "clip_token_count" in batch else 0,
            "clip_was_truncated": int(batch["clip_was_truncated"][i].item()) if "clip_was_truncated" in batch else 0,
            "tag_text_preview": batch["tag_text"][i][:180] if "tag_text" in batch else "",
            "tag_token_count": int(batch["tag_token_count"][i].item()) if "tag_token_count" in batch else 0,
            "seq_len": int(attention_mask[i].sum().item()) if attention_mask.ndim == 2 else 0,
            "meta_num_mean": float(meta_num[i].mean().item()) if meta_num.ndim >= 2 and meta_num.shape[1] > 0 else 0.0,
            "meta_num_std": float(meta_num[i].std().item()) if meta_num.ndim >= 2 and meta_num.shape[1] > 1 else 0.0,
            "meta_bin_sum": float(meta_bin[i].sum().item()) if meta_bin.ndim >= 2 else 0.0,
        }
        sample_rows.append(row)
    result["sample_rows"] = sample_rows

    if model is not None:
        model_was_training = model.training
        if device is not None:
            model = model.to(device)

        def _move_tensor(x: Any) -> Any:
            if isinstance(x, torch.Tensor):
                return x.to(device) if device is not None else x
            return x

        batch_for_model = {k: _move_tensor(v) for k, v in batch.items()}

        with torch.no_grad():
            model.eval()
            pred = model(batch_for_model)
            if isinstance(pred, dict):
                pred = pred.get("logits", pred.get("pred", pred.get("outputs", pred)))
            if isinstance(pred, (list, tuple)):
                pred = pred[0]
            if not isinstance(pred, torch.Tensor):
                raise TypeError(f"Model output must be a torch.Tensor, got {type(pred)}")

            pred = pred.squeeze(-1).detach().float().cpu()

        if model_was_training:
            model.train()

        result["pred_shape"] = tuple(pred.shape)
        result["pred_mean"] = float(pred.mean().item()) if pred.numel() > 0 else 0.0
        result["pred_std"] = float(pred.std().item()) if pred.numel() > 1 else 0.0
        result["pred_min"] = float(pred.min().item()) if pred.numel() > 0 else 0.0
        result["pred_max"] = float(pred.max().item()) if pred.numel() > 0 else 0.0
        result["pred_sample"] = pred[:max_items].tolist()

        for i in range(min(max_items, len(result["sample_rows"]))):
            result["sample_rows"][i]["pred"] = float(pred[i].item())

    return result
