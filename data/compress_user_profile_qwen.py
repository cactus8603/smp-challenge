#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import html
import json
import re
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def safe_str(x) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    s = str(x).strip()
    if s.lower() in {"none", "nan", "null"}:
        return ""
    return s


def clean_description(x) -> str:
    s = safe_str(x)
    if not s:
        return ""

    s = html.unescape(s)
    s = re.sub(r"<[^>]*>", " ", s)

    s = re.sub(r"\b(?:https?|ftp)://[^\s\"'<>()]+", " ", s, flags=re.I)
    s = re.sub(r"\bwww\.[^\s\"'<>()]+", " ", s, flags=re.I)
    s = re.sub(r"\bmailto:[^\s\"'<>()]+", " ", s, flags=re.I)

    s = re.sub(
        r"\b[a-zA-Z0-9.-]+\.(?:com|org|net|edu|gov|co|io|me|info|biz|jp|tw|uk|de|fr|it|es|ca|au|in|cn|hk|ph|nl|se|ru)(?:/[^\s\"'<>()]*)?",
        " ",
        s,
        flags=re.I,
    )

    s = re.sub(r"\b[\w.+-]+@[\w.-]+\.\w+\b", " ", s)
    s = re.sub(
        r"\b[\w.+-]+\s*(?:\{AT\}|\[AT\]|\(AT\)| at )\s*[\w.-]+\b",
        " ",
        s,
        flags=re.I,
    )

    s = re.sub(
        r"\b\S+\.(?:jpg|jpeg|png|gif|webp|bmp|tiff|svg|html|htm|php|aspx|pdf|zip)\b",
        " ",
        s,
        flags=re.I,
    )

    s = re.sub(r"[-_=*~#]{3,}", " ", s)
    s = re.sub(r"([!?.,/\\|:;])\1{2,}", " ", s)
    s = re.sub(r"\b\S{35,}\b", " ", s)
    s = re.sub(r"[<>={}\[\]|\\]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def shorten_text(x: str, max_words: int) -> str:
    return " ".join(safe_str(x).split()[:max_words])


def build_prompt(user_desc: str, loc_desc: str) -> str:
    text = user_desc
    if loc_desc:
        text += f"\nLocation text: {loc_desc}"

    return f"""Extract useful visual/social profile keywords from this Flickr user profile.

Rules:
- Return only 3 to 12 lowercase English keywords or short phrases.
- Focus on identity, photography style, photo topics, profession, location clues, and visual interests.
- Remove URLs, emails, HTML, copyright notices, awards, camera-selling links, and unrelated spam.
- Do not write a sentence.
- Do not explain.
- Separate items by commas.

Profile:
{text}

Output:"""


def clean_model_output(x: str) -> str:
    x = safe_str(x).lower()
    x = re.sub(r"^[\"'`]+|[\"'`]+$", "", x)
    x = re.sub(r"^(keywords|summary|profile)\s*:\s*", "", x)
    x = re.sub(r"[\n\r]+", ", ", x)
    x = re.sub(r"\s*,\s*", ", ", x)
    x = re.sub(r"\s+", " ", x)
    x = x.strip(" ,.;")

    parts = [p.strip(" ,.;") for p in x.split(",") if p.strip(" ,.;")]
    parts = parts[:12]
    parts = [p for p in parts if len(p.split()) <= 5 and len(p) <= 60]

    return ", ".join(parts)


@torch.inference_mode()
def generate_summary(tokenizer, model, user_desc, loc_desc, device) -> str:
    user_desc = shorten_text(clean_description(user_desc), 180)
    loc_desc = shorten_text(clean_description(loc_desc), 60)

    if not user_desc and not loc_desc:
        return ""

    prompt = build_prompt(user_desc, loc_desc)

    messages = [
        {"role": "system", "content": "You are a precise information extraction engine."},
        {"role": "user", "content": prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    ).to(device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,
        repetition_penalty=1.05,
        pad_token_id=tokenizer.eos_token_id,
    )

    gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    out = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return clean_model_output(out)


def load_user_json(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data).reset_index(drop=True)
    return df


def save_user_json(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.to_dict(orient="list")
    with path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


def process_one_file(
    input_path: Path,
    output_path: Path,
    tokenizer,
    model,
    device: str,
    resume: bool = True,
    save_every: int = 500,
    limit: int | None = None,
) -> None:
    if resume and output_path.exists():
        print(f"[INFO] resume from output: {output_path}")
        df = load_user_json(output_path)
    else:
        print(f"[INFO] load input: {input_path}")
        df = load_user_json(input_path)

    if "profile_summary" not in df.columns:
        df["profile_summary"] = ""
    if "user_description" not in df.columns:
        df["user_description"] = ""
    if "location_description" not in df.columns:
        df["location_description"] = ""

    n = len(df)
    if limit is not None:
        n = min(n, limit)

    print(f"[INFO] processing: {input_path}")
    print(f"[INFO] output    : {output_path}")
    print(f"[INFO] rows      : {n}")

    profile_col = df.columns.get_loc("profile_summary")
    user_col = df.columns.get_loc("user_description")
    loc_col = df.columns.get_loc("location_description")

    for i in tqdm(range(n), desc=input_path.parent.name):
        old_summary = safe_str(df.iat[i, profile_col])
        if resume and old_summary:
            continue

        user_desc = df.iat[i, user_col]
        loc_desc = df.iat[i, loc_col]

        try:
            summary = generate_summary(
                tokenizer=tokenizer,
                model=model,
                user_desc=user_desc,
                loc_desc=loc_desc,
                device=device,
            )
        except Exception as e:
            print(f"[WARN] row={i} failed: {e}")
            summary = ""

        df.iat[i, profile_col] = summary

        if save_every > 0 and (i + 1) % save_every == 0:
            save_user_json(df, output_path)
            print(f"[INFO] checkpoint saved: {output_path}")

    save_user_json(df, output_path)
    print(f"[INFO] saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", required=True, help="official data dir, e.g. /local/smp/data")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--limit", type=int, default=None, help="debug limit per split")
    parser.add_argument("--splits", default="train,test", help="comma separated, default=train,test")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device: {device}")
    print(f"[INFO] loading model: {args.model}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )

    if device == "cpu":
        model = model.to(device)

    model.eval()

    for split in splits:
        input_path = input_dir / split / f"{split}_user_data.json"
        output_path = input_dir / split / f"{split}_user_data_with_profile.json"

        if not input_path.exists():
            print(f"[WARN] missing file, skip: {input_path}")
            continue

        process_one_file(
            input_path=input_path,
            output_path=output_path,
            tokenizer=tokenizer,
            model=model,
            device=device,
            resume=args.resume,
            save_every=args.save_every,
            limit=args.limit,
        )

    print("[INFO] all done.")


if __name__ == "__main__":
    main()

# nohup python3 compress_user_profile_qwen.py --input_dir /local/smp/data --resume --save_every 500 > compress_profile.log 2>&1 &
# echo $!

# python3 compress_user_profile_qwen.py --input_dir /local/smp/data --limit 100 --resume