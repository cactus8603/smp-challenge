#!/usr/bin/env python3
"""Benchmark Qwen3-VL-4B inference speed on sample Flickr images."""

import time
import torch
from pathlib import Path
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

MODEL_PATH = "/ssd1/chihyi/VLM_Storge/Qwen3-VL-4B-Instruct"
IMAGE_DIR = "/ssd1/lchiayu/smp-challenge/data/official_data/train_set/train"
N_WARMUP = 2
N_BENCH = 10

PROMPT = (
    "This is a photo shared on Flickr. "
    "Describe the main subject, visual style, and likely popularity appeal in one concise sentence."
)

def main():
    print(f"Loading model from {MODEL_PATH} ...")
    t0 = time.time()

    processor = Qwen3VLProcessor.from_pretrained(MODEL_PATH)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()
    print(f"Model loaded in {time.time()-t0:.1f}s")

    # pick sample images (structure: train/<user_id>/<photo_id>.jpg)
    image_paths = sorted(Path(IMAGE_DIR).glob("*/*.jpg"))[:N_WARMUP + N_BENCH]
    print(f"Found {len(image_paths)} sample images")

    def run_one(img_path):
        image = Image.open(img_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=False,
            )
        gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        return processor.decode(gen_ids, skip_special_tokens=True)

    # warmup
    print(f"\nWarming up ({N_WARMUP} images)...")
    for p in image_paths[:N_WARMUP]:
        run_one(p)

    # benchmark
    print(f"Benchmarking ({N_BENCH} images)...")
    times = []
    for p in image_paths[N_WARMUP:N_WARMUP + N_BENCH]:
        t = time.time()
        caption = run_one(p)
        elapsed = time.time() - t
        times.append(elapsed)
        print(f"  {p.name}: {elapsed:.2f}s — {caption[:80]}")

    avg = sum(times) / len(times)
    total_images = 305_613
    total_hours = avg * total_images / 3600

    print(f"\n--- Results ---")
    print(f"Avg per image : {avg:.2f}s")
    print(f"Min           : {min(times):.2f}s")
    print(f"Max           : {max(times):.2f}s")
    print(f"Total images  : {total_images:,}")
    print(f"Est. total    : {total_hours:.1f} hours  ({total_hours/24:.1f} days)")
    print(f"GPU memory    : {torch.cuda.max_memory_allocated()/1e9:.1f} GB peak")

if __name__ == "__main__":
    main()
