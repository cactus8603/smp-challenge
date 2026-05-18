#!/usr/bin/env python3
"""
Qwen2-VL-2B 對 extra crawled data (79k 張) 生成 MLLM 特徵。
圖片來自 data/get_extra_data/output/extra_data_*/images/ 下所有 .jpg。
Join key: photo_id (= Pid, 檔名去掉副檔名)。

多 GPU 並行用法：
    CUDA_VISIBLE_DEVICES=0 python3 data/generate_mllm_features_extra.py --chunk 0 --n_chunks 3 &
    CUDA_VISIBLE_DEVICES=1 python3 data/generate_mllm_features_extra.py --chunk 1 --n_chunks 3 &
    CUDA_VISIBLE_DEVICES=2 python3 data/generate_mllm_features_extra.py --chunk 2 --n_chunks 3 &
"""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_ID   = "Qwen/Qwen2-VL-2B-Instruct"
EXTRA_ROOT = Path("/ssd1/lchiayu/smp-challenge/data/get_extra_data/output")
OUTPUT_DIR = Path("/ssd1/lchiayu/smp-challenge/data/get_extra_data/mllm_features")

CATEGORY_CHOICES = (
    "Animal, Electronics, Entertainment, Family, Fashion, Food, "
    "Holiday_Celebrations, Social_People, Travel_Active_Sports, Urban, Weather_Season"
)

CLOSED_PROMPTS = {
    "content_category": (
        f"What is the main topic of this photo? "
        f"Reply with EXACTLY ONE word or phrase from this list: {CATEGORY_CHOICES}. "
        f"Do not explain, just output the category name."
    ),
    "dominant_color_tone": (
        "What is the dominant color tone of this photo? "
        "Definitions — warm: reds/oranges/yellows dominate; "
        "cool: blues/greens/purples dominate; "
        "neutral: grays/whites/blacks/browns dominate; "
        "vibrant: high saturation with multiple vivid colors; "
        "monochrome: black-and-white or single-hue. "
        "Reply with exactly one word from: warm, cool, neutral, vibrant, monochrome"
    ),
    "scene_complexity": (
        "How visually complex is this photo? "
        "Reply with exactly one word from: simple, moderate, complex"
    ),
    "estimated_quality": (
        "How would you rate the technical photography quality? "
        "Definitions — professional: high image quality and well composed; "
        "amateur: acceptable quality but imperfect composition; "
        "snapshot: low quality or poorly composed. "
        "Reply with exactly one word from: professional, amateur, snapshot"
    ),
    "emotion_trigger": (
        "What dominant emotion is this photo designed to evoke? "
        "Reply with exactly one word from: "
        "joy, surprise, curiosity, inspiration, humor, awful, disgust, angry, sad, neutral"
    ),
}


def collect_images() -> list[tuple[str, Path]]:
    """收集所有 extra data 圖片，回傳 (photo_id, abs_path) 列表。"""
    entries = []
    for cat_dir in sorted(EXTRA_ROOT.iterdir()):
        img_dir = cat_dir / "images"
        if not img_dir.exists():
            continue
        for p in img_dir.rglob("*.jpg"):
            entries.append((p.stem, p))
    entries.sort(key=lambda x: x[0])
    return entries


def build_messages(image_path: str, question: str) -> list:
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": f"file://{image_path}"},
            {"type": "text",  "text": question},
        ],
    }]


def ask_one(model, processor, image_path: str, question: str, device: str) -> str:
    messages = build_messages(image_path, question)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=12, do_sample=False)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    response = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()
    return response.split()[0].rstrip(".,;:").lower() if response else "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk",    type=int, default=0)
    parser.add_argument("--n_chunks", type=int, default=3)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"chunk_{args.chunk:02d}_of_{args.n_chunks:02d}.jsonl"

    # 斷點續跑
    done_ids: set[str] = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["photo_id"])
                except Exception:
                    pass
        print(f"[INFO] Resuming: {len(done_ids)} already done")

    # 收集全部圖片 → 按 chunk 分
    all_entries = collect_images()
    chunk_entries = [e for i, e in enumerate(all_entries) if i % args.n_chunks == args.chunk]
    todo = [(pid, p) for pid, p in chunk_entries if pid not in done_ids]
    print(f"[INFO] Chunk {args.chunk}/{args.n_chunks} | total={len(chunk_entries)} | todo={len(todo)}")

    # 載入模型
    print(f"[INFO] Loading {MODEL_ID} on {device}...")
    t0 = time.time()
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation="eager",
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    print(f"[INFO] Model ready in {time.time()-t0:.1f}s")

    t_start = time.time()
    errors = 0
    with open(output_path, "a") as fout:
        for i, (photo_id, img_path) in enumerate(todo):
            try:
                row = {"photo_id": photo_id}
                for feat, prompt in CLOSED_PROMPTS.items():
                    row[feat] = ask_one(model, processor, str(img_path), prompt, device)
                fout.write(json.dumps(row) + "\n")
                fout.flush()
            except Exception as e:
                errors += 1
                fout.write(json.dumps({"photo_id": photo_id, "error": str(e)}) + "\n")
                fout.flush()

            if (i + 1) % 200 == 0:
                elapsed = time.time() - t_start
                per_img = elapsed / (i + 1)
                eta_min = per_img * (len(todo) - i - 1) / 60
                print(f"  [{i+1:6d}/{len(todo)}] {per_img:.2f}s/img | ETA {eta_min:.0f}min | errors={errors}")

    total = time.time() - t_start
    print(f"\n[DONE] chunk {args.chunk} | {len(todo)} images | {total/3600:.1f}h | errors={errors}")
    print(f"[DONE] Saved to {output_path}")


if __name__ == "__main__":
    main()
