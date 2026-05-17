#!/usr/bin/env python3
"""
Qwen2-VL-2B-Instruct benchmark: 1000 張圖片的速度測試 + 結構化 category 特徵提取
用法：
    CUDA_VISIBLE_DEVICES=4 python3 data/benchmark_internvl2.py
"""

import json
import os
import random
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

# ── 設定 ──────────────────────────────────────────────────────────────────
MODEL_ID    = "Qwen/Qwen2-VL-2B-Instruct"
IMAGE_ROOT  = Path("/ssd1/lchiayu/smp-challenge/data/official_data/train_set/train")
FILEPATH_TXT = Path("/ssd1/lchiayu/smp-challenge/data/official_data/smp_input/train/train_img_filepath.txt")
OUTPUT_JSON = Path("/ssd1/lchiayu/smp-challenge/data/internvl2_benchmark_1000_v2.json")
N_SAMPLES   = 1000
SEED        = 42

# content_category: 固定選項（對應 SMP 官方 category）
CATEGORY_CHOICES = (
    "Animal, Electronics, Entertainment, Family, Fashion, Food, "
    "Holiday_Celebrations, Social_People, Travel_Active_Sports, Urban, Weather_Season"
)

# closed-ended prompts（固定選項，只取第一個 token 解析）
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

OPEN_PROMPTS: dict = {}  # content_subcategory 已移除

def build_messages(image_path: str, question: str) -> list:
    """建立 Qwen2-VL 所需的 messages 格式"""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text",  "text": question},
            ],
        }
    ]


def ask_one(model, processor, image_path: str, question: str, device: str,
            max_new_tokens: int = 12, keep_full: bool = False) -> str:
    """對一張圖片問一個問題，回傳模型的回答。
    keep_full=True: 保留完整回答（open-ended）；False: 只取第一個 token（closed）
    """
    messages = build_messages(image_path, question)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated_ids_trimmed = [
        out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

    if not response:
        return "unknown"
    if keep_full:
        return response.lower()
    # closed: 取第一個 word（處理 "Holiday_Celebrations" 等含底線的類別）
    return response.split()[0].rstrip(".,;:").lower()


def main():
    random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    # ── 載入模型 ─────────────────────────────────────────────────────────
    print(f"[INFO] Loading {MODEL_ID} ...")
    t0 = time.time()
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    load_time = time.time() - t0
    print(f"[INFO] Model loaded in {load_time:.1f}s")

    # ── 採樣 1000 張圖 ───────────────────────────────────────────────────
    all_paths = []
    with open(FILEPATH_TXT) as f:
        for line in f:
            rel = line.strip()           # e.g. train/59@N75/775.jpg
            uid_pid = "/".join(rel.split("/")[1:])   # e.g. 59@N75/775.jpg
            abs_path = IMAGE_ROOT / uid_pid
            if abs_path.exists():
                all_paths.append((uid_pid, abs_path))

    sampled = random.sample(all_paths, min(N_SAMPLES, len(all_paths)))
    print(f"[INFO] Sampled {len(sampled)} images from {len(all_paths)} total")

    # ── 推論 ─────────────────────────────────────────────────────────────
    results = []
    errors  = 0
    t_start = time.time()

    for i, (uid_pid, img_path) in enumerate(sampled):
        try:
            row = {"id": uid_pid}
            for feat_name, prompt in CLOSED_PROMPTS.items():
                row[feat_name] = ask_one(model, processor, str(img_path), prompt, device,
                                         max_new_tokens=12, keep_full=False)
            for feat_name, prompt in OPEN_PROMPTS.items():
                row[feat_name] = ask_one(model, processor, str(img_path), prompt, device,
                                         max_new_tokens=20, keep_full=True)
            results.append(row)
        except Exception as e:
            errors += 1
            results.append({"id": uid_pid, "error": str(e)})

        # 進度 + 速度
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t_start
            per_img = elapsed / (i + 1)
            remaining = per_img * (len(sampled) - i - 1)
            print(
                f"  [{i+1:4d}/{len(sampled)}] "
                f"{per_img:.2f}s/img | "
                f"ETA {remaining/60:.1f}min | "
                f"errors={errors}"
            )

    total_time = time.time() - t_start
    per_img    = total_time / len(sampled)

    # ── 結果 ─────────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"Total: {total_time:.0f}s for {len(sampled)} images")
    print(f"Speed: {per_img:.2f}s/image  ({1/per_img:.2f} img/s)")
    print(f"Errors: {errors}")
    print(f"\nEstimated time for 305,613 images:")
    full_305k = per_img * 305613
    print(f"  1 GPU : {full_305k/3600:.1f} hours")
    print(f"  3 GPUs: {full_305k/3/3600:.1f} hours")
    print(f"  5 GPUs: {full_305k/5/3600:.1f} hours")

    # 答案分布
    print("\nAnswer distribution (from successful results):")
    ok = [r for r in results if "error" not in r]
    from collections import Counter
    for feat in list(CLOSED_PROMPTS) + list(OPEN_PROMPTS):
        cnt = Counter(r.get(feat, "missing") for r in ok)
        print(f"  {feat}: {dict(cnt.most_common(8))}")

    # 樣本輸出
    print("\nSample outputs (first 3):")
    for r in ok[:3]:
        print(f"  {r}")

    # 儲存
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": MODEL_ID,
        "n_sampled": len(sampled),
        "n_success": len(ok),
        "errors": errors,
        "total_time_s": round(total_time, 1),
        "per_image_s": round(per_img, 3),
        "est_305k_1gpu_hours": round(full_305k / 3600, 1),
        "est_305k_3gpu_hours": round(full_305k / 3 / 3600, 1),
        "est_305k_5gpu_hours": round(full_305k / 5 / 3600, 1),
        "results": results,
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[INFO] Saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
