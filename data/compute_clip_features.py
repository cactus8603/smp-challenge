#!/usr/bin/env python3
"""
compute_clip_features.py

Pre-compute CLIP-derived similarity features and write processed_v5.

New columns added:
  clip_text_image_sim   : cos_sim(CLIP(user_text), CLIP(image))
                          user_text = title | tags: … | topic: …  (NO BLIP-2 caption)
                          Measures how well the user's own description matches the photo.
  clip_text_caption_sim : cos_sim(CLIP(user_text), CLIP(blip2_caption))
                          Measures how compatible the user's description is with the
                          image's actual content (as described by BLIP-2).
                          0.0 when caption is absent (test split).

Usage:
  python3 data/compute_clip_features.py \
      --input_dir  data/processed_v4 \
      --output_dir data/processed_v5 \
      --image_root data/official_data/train_set \
      --device     cuda:0 \
      --batch_size 256
"""

from __future__ import annotations

import argparse
import re
import html
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from transformers import AutoTokenizer, CLIPTextModelWithProjection, CLIPVisionModelWithProjection, CLIPImageProcessor
from tqdm import tqdm


MODEL_NAME = "openai/clip-vit-base-patch32"
MAX_TOKENS = 77


# ---------------------------------------------------------------------------
# Text helpers (mirrors smp_dataset.build_clip_text logic, caption excluded)
# ---------------------------------------------------------------------------

def safe_str(x) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    s = str(x).strip()
    return "" if s.lower() in {"none", "nan", "null", "<na>"} else s


def _clean_tag(tag: str) -> str:
    s = html.unescape(tag)
    s = re.sub(r"<[^>]*>", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def split_tags(value) -> List[str]:
    text = safe_str(value)
    if not text:
        return []
    quoted = re.findall(r'"([^"]+)"', text)
    items = quoted if quoted else re.split(r"[\s,;|]+", text)
    return [_clean_tag(x) for x in items if _clean_tag(x)]


def build_user_text(row: pd.Series, tokenizer, max_tags: int = 10) -> str:
    """Build CLIP text from user-provided fields only (NO BLIP-2 caption)."""
    title = safe_str(row.get("title", ""))
    category = safe_str(row.get("category", ""))
    subcategory = safe_str(row.get("subcategory", ""))
    concept = safe_str(row.get("concept", ""))

    topic_parts = [x for x in [category, subcategory, concept] if x]
    # dedup preserving order
    seen = set()
    topic_parts = [p for p in topic_parts if not (p.lower() in seen or seen.add(p.lower()))]
    topic_text = "topic: " + ", ".join(topic_parts) if topic_parts else ""

    tags = split_tags(row.get("alltags", ""))[:max_tags]

    parts = []
    if title:
        parts.append(title)
    if tags:
        parts.append("tags: " + " ".join(tags))
    if topic_text:
        parts.append(topic_text)

    return " | ".join(parts).strip()


# ---------------------------------------------------------------------------
# CLIP embedding helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_texts(texts: List[str], tokenizer, text_model, device: str) -> torch.Tensor:
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_TOKENS,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    out = text_model(input_ids=input_ids, attention_mask=attention_mask)
    emb = out.text_embeds  # [B, D]
    return F.normalize(emb, dim=-1).cpu()


@torch.no_grad()
def encode_images(
    paths: List[Optional[str]],
    image_processor: CLIPImageProcessor,
    vision_model,
    device: str,
    image_root: Optional[Path],
) -> torch.Tensor:
    images = []
    for p in paths:
        img = None
        if p:
            full = Path(p) if Path(p).is_absolute() else (image_root / p if image_root else Path(p))
            try:
                img = Image.open(full).convert("RGB")
            except (FileNotFoundError, UnidentifiedImageError, Exception):
                pass
        if img is None:
            img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        images.append(img)

    pixel_values = image_processor(images=images, return_tensors="pt")["pixel_values"].to(device)
    out = vision_model(pixel_values=pixel_values)
    emb = out.image_embeds  # [B, D]
    return F.normalize(emb, dim=-1).cpu()


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_parquet(
    input_path: Path,
    output_path: Path,
    tokenizer,
    text_model,
    vision_model,
    image_processor,
    device: str,
    batch_size: int,
    image_root: Optional[Path],
) -> None:
    print(f"\n[INFO] Loading {input_path} ...")
    df = pd.read_parquet(input_path)
    print(f"  Rows: {len(df)}, has caption: {'caption' in df.columns}")

    n = len(df)
    clip_text_image_sims = np.zeros(n, dtype=np.float32)
    clip_text_caption_sims = np.zeros(n, dtype=np.float32)
    clip_caption_image_sims = np.zeros(n, dtype=np.float32)

    has_images = "image_path" in df.columns
    has_captions = "caption" in df.columns

    rows_list = df.to_dict(orient="records")

    for start in tqdm(range(0, n, batch_size), desc=input_path.name):
        end = min(start + batch_size, n)
        batch_rows = rows_list[start:end]

        # --- user text embeddings ---
        user_texts = [build_user_text(pd.Series(r), tokenizer) or "photo" for r in batch_rows]
        user_text_emb = encode_texts(user_texts, tokenizer, text_model, device)  # [B, D]

        # --- image embeddings ---
        if has_images:
            img_paths = [safe_str(r.get("image_path", "")) or None for r in batch_rows]
            image_emb = encode_images(img_paths, image_processor, vision_model, device, image_root)
        else:
            image_emb = torch.zeros_like(user_text_emb)

        # --- caption embeddings ---
        if has_captions:
            captions = [safe_str(r.get("caption", "")) for r in batch_rows]
            has_cap = [bool(c) for c in captions]
            # only encode non-empty captions
            non_empty_idx = [i for i, h in enumerate(has_cap) if h]
            cap_emb = torch.zeros_like(user_text_emb)
            if non_empty_idx:
                cap_texts = [captions[i] for i in non_empty_idx]
                cap_emb_sub = encode_texts(cap_texts, tokenizer, text_model, device)
                for j, i in enumerate(non_empty_idx):
                    cap_emb[i] = cap_emb_sub[j]

        # --- cosine similarities ---
        text_image_sim = (user_text_emb * image_emb).sum(dim=-1).numpy()  # [B]
        clip_text_image_sims[start:end] = text_image_sim

        if has_captions:
            text_cap_sim = (user_text_emb * cap_emb).sum(dim=-1).numpy()
            cap_image_sim = (cap_emb * image_emb).sum(dim=-1).numpy()
            # zero out rows without captions
            for i, h in enumerate(has_cap):
                if not h:
                    text_cap_sim[i] = 0.0
                    cap_image_sim[i] = 0.0
            clip_text_caption_sims[start:end] = text_cap_sim
            clip_caption_image_sims[start:end] = cap_image_sim

    df["clip_text_image_sim"] = clip_text_image_sims
    df["clip_text_caption_sim"] = clip_text_caption_sims
    df["clip_caption_image_sim"] = clip_caption_image_sims

    print(f"  clip_text_image_sim:    mean={clip_text_image_sims.mean():.4f}, std={clip_text_image_sims.std():.4f}")
    print(f"  clip_text_caption_sim:  mean={clip_text_caption_sims.mean():.4f}, std={clip_text_caption_sims.std():.4f}")
    print(f"  clip_caption_image_sim: mean={clip_caption_image_sims.mean():.4f}, std={clip_caption_image_sims.std():.4f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"  Saved → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--image_root", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    image_root = Path(args.image_root) if args.image_root else None
    device = args.device

    print(f"[INFO] Loading CLIP model: {MODEL_NAME} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    text_model = CLIPTextModelWithProjection.from_pretrained(MODEL_NAME).to(device).eval()
    vision_model = CLIPVisionModelWithProjection.from_pretrained(MODEL_NAME).to(device).eval()
    image_processor = CLIPImageProcessor.from_pretrained(MODEL_NAME)
    print("[INFO] Model loaded.")

    targets = [
        ("official_train_cap.parquet", "official_train_cap.parquet"),
        ("official_train.parquet",     "official_train.parquet"),
        ("test.parquet",               "test.parquet"),
    ]

    for src_name, dst_name in targets:
        src = input_dir / src_name
        dst = output_dir / dst_name
        if not src.exists():
            print(f"[SKIP] {src} not found.")
            continue
        process_parquet(src, dst, tokenizer, text_model, vision_model,
                        image_processor, device, args.batch_size, image_root)

    print("\n[DONE] processed_v5 ready.")


if __name__ == "__main__":
    main()
