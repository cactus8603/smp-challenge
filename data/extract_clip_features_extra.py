#!/usr/bin/env python3
"""
Frozen CLIP ViT-B/32 backbone feature extraction for extra crawled images.

Saves two arrays per chunk:
  - global_feat  : (N, 512)    float16  — image_embeds from CLIP projection head
  - patch_tokens : (N, 49, 768) float16 — last_hidden_state patch tokens (excl. CLS)
                                           ViT-B/32: 7×7=49 patches, hidden_size=768

Output layout:
  <output_dir>/
    chunk_0000.npz   ← {photo_ids: [...], global: (N,512), patches: (N,196,768)}
    chunk_0001.npz
    ...
    photo_id_index.json  ← {photo_id: {"chunk": 0, "row": 42}}  (written at the end)

Usage (single GPU):
    CUDA_VISIBLE_DEVICES=0 python3 data/extract_clip_features_extra.py

Usage (specify custom paths):
    CUDA_VISIBLE_DEVICES=0 python3 data/extract_clip_features_extra.py \\
        --image_root /path/to/extra_data/output \\
        --output_dir /path/to/clip_features_extra \\
        --batch_size 128
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPProcessor, CLIPVisionModelWithProjection

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_IMAGE_ROOT = Path("/ssd1/lchiayu/smp-challenge/data/get_extra_data/output")
DEFAULT_OUTPUT_DIR = Path("/ssd1/lchiayu/smp-challenge/data/get_extra_data/clip_features_extra")
MODEL_NAME   = "openai/clip-vit-base-patch32"
CHUNK_SIZE   = 5_000   # images per npz file


# ── Image collection ──────────────────────────────────────────────────────────
def collect_images(image_root: Path) -> list[tuple[str, Path]]:
    """Scan all extra_data_*/images/ subdirs, return sorted (photo_id, path) list."""
    entries: list[tuple[str, Path]] = []
    for cat_dir in sorted(image_root.iterdir()):
        img_dir = cat_dir / "images"
        if not img_dir.exists():
            continue
        for p in img_dir.rglob("*.jpg"):
            entries.append((p.stem, p))
    entries.sort(key=lambda x: x[0])
    return entries


# ── Resume helpers ────────────────────────────────────────────────────────────
def load_done_ids(output_dir: Path) -> set[str]:
    """Return photo_ids already saved in existing chunk files."""
    done: set[str] = set()
    for npz in sorted(output_dir.glob("chunk_*.npz")):
        try:
            data = np.load(npz, allow_pickle=True)
            for pid in data["photo_ids"]:
                done.add(str(pid))
        except Exception:
            pass
    return done


# ── Preprocessing ─────────────────────────────────────────────────────────────
def load_pil(path: Path) -> Image.Image | None:
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frozen CLIP features for extra data.")
    parser.add_argument("--image_root", type=Path, default=DEFAULT_IMAGE_ROOT,
                        help="Root dir containing extra_data_*/images/ subdirs.")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Output directory for chunk npz files.")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE,
                        help="Number of images per npz file.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"[INFO] Loading {MODEL_NAME} on {device} ...")
    t0 = time.time()
    model = CLIPVisionModelWithProjection.from_pretrained(
        MODEL_NAME, use_safetensors=False,
    ).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    print(f"[INFO] Model ready in {time.time()-t0:.1f}s  |  "
          f"VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB" if device == "cuda" else "")

    # ── Collect images & filter already-done ──────────────────────────────────
    all_entries = collect_images(args.image_root)
    print(f"[INFO] Found {len(all_entries):,} images under {args.image_root}")

    done_ids = load_done_ids(args.output_dir)
    if done_ids:
        print(f"[INFO] Resuming: {len(done_ids):,} already done")
    todo = [(pid, p) for pid, p in all_entries if pid not in done_ids]
    print(f"[INFO] To process: {len(todo):,} images")

    if not todo:
        print("[INFO] Nothing to do — all images already extracted.")
    else:
        # ── Extraction loop ───────────────────────────────────────────────────
        # Buffer for current chunk
        buf_pids:    list[str]         = []
        buf_global:  list[np.ndarray]  = []   # (512,) float16
        buf_patches: list[np.ndarray]  = []   # (196, 768) float16

        chunk_idx   = len(list(args.output_dir.glob("chunk_*.npz")))
        errors      = 0
        t_start     = time.time()

        def flush_chunk() -> None:
            nonlocal chunk_idx
            path = args.output_dir / f"chunk_{chunk_idx:04d}.npz"
            np.savez_compressed(
                path,
                photo_ids = np.array(buf_pids),
                global_feat   = np.stack(buf_global,  axis=0),  # (N, 512)
                patch_tokens  = np.stack(buf_patches, axis=0),  # (N, 196, 768)
            )
            print(f"  → Saved {path.name}  ({len(buf_pids)} images)")
            chunk_idx += 1
            buf_pids.clear(); buf_global.clear(); buf_patches.clear()

        # Process in batches
        i = 0
        while i < len(todo):
            batch_pairs = todo[i: i + args.batch_size]
            pids   = [p for p, _ in batch_pairs]
            paths  = [p for _, p in batch_pairs]

            # Load images; mark failures
            pil_imgs = [load_pil(p) for p in paths]
            valid_mask = [img is not None for img in pil_imgs]
            valid_pils = [img for img in pil_imgs if img is not None]

            if valid_pils:
                try:
                    inputs = processor(images=valid_pils, return_tensors="pt").to(device)
                    with torch.no_grad():
                        outputs = model(**inputs)
                    # global embedding: (N, 512)
                    global_f = outputs.image_embeds.half().cpu().numpy()
                    # patch tokens: (N, 49, 768) for ViT-B/32 — skip CLS token at index 0
                    patch_f  = outputs.last_hidden_state[:, 1:, :].half().cpu().numpy()

                    vi = 0  # valid index
                    for ok, pid in zip(valid_mask, pids):
                        if ok:
                            buf_pids.append(pid)
                            buf_global.append(global_f[vi])
                            buf_patches.append(patch_f[vi])
                            vi += 1
                        else:
                            errors += 1
                except Exception as e:
                    print(f"  [WARN] Batch failed: {e}")
                    errors += len(valid_pils)
            else:
                errors += len(pids)

            # Flush when chunk buffer is full
            if len(buf_pids) >= args.chunk_size:
                flush_chunk()

            i += len(batch_pairs)
            processed = i
            if processed % (args.batch_size * 10) == 0 or processed == len(todo):
                elapsed = time.time() - t_start
                rate    = processed / elapsed
                eta_min = (len(todo) - processed) / rate / 60
                print(f"  [{processed:6d}/{len(todo)}]  "
                      f"{rate:.1f} img/s  |  ETA {eta_min:.0f} min  |  errors={errors}")

        # Flush remaining
        if buf_pids:
            flush_chunk()

    # ── Build photo_id → (chunk, row) index ──────────────────────────────────
    print("[INFO] Building photo_id index ...")
    index: dict[str, dict] = {}
    for npz_path in sorted(args.output_dir.glob("chunk_*.npz")):
        chunk_num = int(npz_path.stem.split("_")[1])
        data = np.load(npz_path, allow_pickle=True)
        for row, pid in enumerate(data["photo_ids"]):
            index[str(pid)] = {"chunk": chunk_num, "row": int(row)}

    index_path = args.output_dir / "photo_id_index.json"
    with open(index_path, "w") as f:
        json.dump(index, f)

    total = time.time() - t_start if todo else 0
    print(f"\n[DONE] {len(index):,} images indexed in {len(list(args.output_dir.glob('chunk_*.npz')))} chunks")
    print(f"[DONE] Output: {args.output_dir}")
    print(f"[DONE] Index:  {index_path}")
    if todo:
        print(f"[DONE] Time:   {total/3600:.2f}h  |  errors={errors}")

    # ── Size report ───────────────────────────────────────────────────────────
    total_bytes = sum(p.stat().st_size for p in args.output_dir.glob("chunk_*.npz"))
    print(f"[DONE] Total size: {total_bytes/1e9:.1f} GB")


if __name__ == "__main__":
    main()
