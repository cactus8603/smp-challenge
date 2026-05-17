#!/usr/bin/env python3
"""
Batch-generate image captions for all Flickr training images using BLIP-2.

Output: JSON file mapping photo_id -> caption string
Supports resume: skips already-processed photo_ids.

Usage:
  python generate_captions_blip2.py \
      --image_dir /ssd1/lchiayu/smp-challenge/data/official_data/train_set/train \
      --output    /ssd1/lchiayu/smp-challenge/data/processed_v2/captions_blip2.json \
      --batch_size 16 --resume

For test set (flat dir or user-subdir layout):
  python generate_captions_blip2.py \
      --image_dir /ssd1/lchiayu/smp-challenge/data/official_data/test_set/test \
      --output    /ssd1/lchiayu/smp-challenge/data/processed_v2/captions_blip2_test.json \
      --batch_size 16 --resume
"""

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Blip2ForConditionalGeneration


MODEL_ID = "Salesforce/blip2-opt-2.7b"
SAVE_EVERY = 2000   # checkpoint frequency


def load_checkpoint(path: Path) -> dict:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_checkpoint(captions: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(captions, f, ensure_ascii=False)
    tmp.replace(path)


def collect_images(image_dir: Path) -> list[tuple[str, Path]]:
    """Return (photo_id, path) for every jpg under image_dir.
    Handles both flat layout and user-subdir layout (user_id/photo_id.jpg).
    """
    pairs = []
    for p in image_dir.rglob("*.jpg"):
        photo_id = p.stem
        pairs.append((photo_id, p))
    pairs.sort(key=lambda x: x[0])
    return pairs


def load_model(device: str):
    print(f"[INFO] Loading {MODEL_ID} ...")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Blip2ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    if device == "cpu":
        model = model.to(device)
    model.eval()
    print(f"[INFO] Model loaded in {time.time()-t0:.1f}s | "
          f"VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")
    return processor, model


@torch.inference_mode()
def generate_batch(
    processor,
    model,
    images: list[Image.Image],
    device: str,
) -> list[str]:
    inputs = processor(images=images, return_tensors="pt").to(
        device, torch.float16 if device == "cuda" else torch.float32
    )
    output_ids = model.generate(
        **inputs,
        max_new_tokens=60,
        num_beams=1,      # greedy — fastest
        length_penalty=1.0,
    )
    captions = processor.batch_decode(output_ids, skip_special_tokens=True)
    return [c.strip() for c in captions]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output",    required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--resume", action="store_true",
                        help="Skip photo_ids already in output file")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N images (for testing)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device: {device}")

    output_path = Path(args.output)
    captions: dict = {}
    if args.resume:
        captions = load_checkpoint(output_path)
        print(f"[INFO] Resumed {len(captions)} existing captions from {output_path}")

    image_dir = Path(args.image_dir)
    all_pairs = collect_images(image_dir)
    if args.limit:
        all_pairs = all_pairs[: args.limit]

    # Filter already-done
    todo = [(pid, p) for pid, p in all_pairs if pid not in captions]
    print(f"[INFO] Total images: {len(all_pairs)} | To process: {len(todo)}")

    if not todo:
        print("[INFO] Nothing to do.")
        return

    processor, model = load_model(device)

    bs = args.batch_size
    t_start = time.time()
    processed = 0

    with tqdm(total=len(todo), desc="Captioning", unit="img") as pbar:
        for i in range(0, len(todo), bs):
            batch_pairs = todo[i : i + bs]
            images, pids = [], []
            for pid, p in batch_pairs:
                try:
                    img = Image.open(p).convert("RGB")
                    images.append(img)
                    pids.append(pid)
                except Exception as e:
                    print(f"\n[WARN] Cannot open {p}: {e}")

            if not images:
                pbar.update(len(batch_pairs))
                continue

            try:
                batch_captions = generate_batch(processor, model, images, device)
            except Exception as e:
                print(f"\n[WARN] batch {i} failed: {e}; writing empty captions")
                batch_captions = [""] * len(images)

            for pid, cap in zip(pids, batch_captions):
                captions[pid] = cap

            processed += len(images)
            pbar.update(len(batch_pairs))

            # ETA
            elapsed = time.time() - t_start
            rate = processed / elapsed
            remaining = (len(todo) - processed) / max(rate, 1e-6)
            pbar.set_postfix(
                rate=f"{rate:.1f}img/s",
                eta=f"{remaining/3600:.1f}h",
            )

            if processed % SAVE_EVERY < bs:
                save_checkpoint(captions, output_path)
                pbar.write(f"[INFO] Checkpoint saved ({len(captions)} captions)")

    save_checkpoint(captions, output_path)

    elapsed = time.time() - t_start
    print(f"\n[INFO] Done. {len(captions)} captions saved to {output_path}")
    print(f"[INFO] Total time: {elapsed/3600:.2f}h | "
          f"Avg: {elapsed/max(processed,1):.3f}s/img")


if __name__ == "__main__":
    main()
