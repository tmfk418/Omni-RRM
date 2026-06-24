#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Image MCQ Inference (Recursive Parquet · Multi‑vote Version)
-----------------------------------------------------------------
• Traverse all parquet files in a directory (including subdirectories)
• Swift full‑parameter model → Single run + Majority voting
• Output predictions.jsonl
"""

import os, json, gc, torch, logging, base64
from pathlib import Path
from tqdm import tqdm
from collections import Counter
import pandas as pd
import jsonlines
from ast import literal_eval
from io import BytesIO
from PIL import Image

from swift.llm import (
    PtEngine, RequestConfig, InferRequest,
    get_model_tokenizer, get_template
)

# --------------------------------------------------------------------
# 1. Configuration
# --------------------------------------------------------------------
INPUT_DIR      = r"/path/to/mmmu"   
OUTPUT_FILE    = r"/path/to/mmmu/predictions.jsonl"
FULL_MODEL_DIR = r"/path/to/Qwen2.5-Omni-7B"
SAVE_INTERVAL  = 5   # Save every N samples

os.environ["CUDA_VISIBLE_DEVICES"]        = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]     = "expandable_segments:True"
os.environ["MAX_PIXELS"]                  = "1003520"
os.environ["VIDEO_MAX_PIXELS"]            = "50176"
os.environ["FPS_MAX_FRAMES"]              = "20"

# --------------------------------------------------------------------
# 2. Load Model
# --------------------------------------------------------------------
logging.info("🚀 Loading fully‑finetuned model …")
model, processor = get_model_tokenizer(FULL_MODEL_DIR)
model.to("cuda:0")

logging.info("⚙️ Building PtEngine …")
template = get_template(model.config.model_type, processor)
engine   = PtEngine.from_model_template(model=model,
                                        template=template,
                                        max_batch_size=1)

# --------------------------------------------------------------------
# 3. Prompt Construction
# --------------------------------------------------------------------
def build_mcq_prompt(question: str, options: list[str]) -> str:
    opts_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
    return f"""
You are a knowledgeable AI assistant. 
Please carefully analyze the given image(s) and the multiple-choice question.
Select the single most accurate answer letter.

### Question
{question}

### Options
{opts_text}

### Output Format
{{"answer": "A"|"B"|"C"|"D"}}

### IMPORTANT
- Only output a valid JSON object.
- Do not include any explanation, reasoning, or extra text.
""".strip()

# --------------------------------------------------------------------
# 4. Extract Images from Parquet
# --------------------------------------------------------------------
def extract_images_from_parquet(row) -> list:
    imgs = []
    for i in range(1, 8):
        key = f"image_{i}"
        img_data = row.get(key)
        if isinstance(img_data, dict) and img_data.get("bytes"):
            try:
                img_bytes = img_data["bytes"]
                if isinstance(img_bytes, str):
                    img_bytes = base64.b64decode(img_bytes)
                img = Image.open(BytesIO(img_bytes)).convert("RGB")
                imgs.append(img)
            except Exception as e:
                logging.warning(f"Image decode failed for {row.get('id')} {key}: {e}")
    return imgs

# --------------------------------------------------------------------
# 5. Single Inference
# --------------------------------------------------------------------
def evaluate_once(images, question: str, options: list[str]) -> str:
    prompt = build_mcq_prompt(question, options)
    req = InferRequest(
        messages=[{"role": "user", "content": "<image>\n" + prompt}],
        images=images,
        videos=None
    )
    cfg = RequestConfig(max_tokens=128, temperature=0.7)
    try:
        resp = engine.infer([req], cfg)[0]
        content = resp.choices[0].message.content.strip()
        obj = json.loads(content)
        ans = obj.get("answer", "error").strip().upper()
        if ans not in ["A", "B", "C", "D"]:
            return "error"
        return ans
    except Exception as e:
        logging.warning(f"Parse failed: {str(e)} | raw: {content[:100] if 'content' in locals() else 'no output'}")
        return "error"
    finally:
        torch.cuda.empty_cache()
        gc.collect()

# --------------------------------------------------------------------
# 6. Multiple Inference + Voting
# --------------------------------------------------------------------
def evaluate_majority(images, question, options, runs=5):
    answers = []
    for _ in range(runs):
        ans = evaluate_once(images, question, options)
        if ans != "error":
            answers.append(ans)
    if not answers:
        return {"single_result": "error", "majority_result": "error", "votes": {}}
    counts = Counter(answers)
    majority_ans = counts.most_common(1)[0][0]
    return {
        "single_result": answers[0],
        "majority_result": majority_ans,
        "votes": dict(counts)
    }

# --------------------------------------------------------------------
# 7. Main Process (Recursive Scan)
# --------------------------------------------------------------------
def main():
    parquet_files = list(Path(INPUT_DIR).rglob("*.parquet"))
    logging.info(f"Found {len(parquet_files)} parquet files")

    buf = []
    with jsonlines.open(OUTPUT_FILE, "w") as w:
        for pq_file in parquet_files:
            df = pd.read_parquet(pq_file)
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{pq_file.name}"):
                try:
                    options = literal_eval(row["options"]) if isinstance(row["options"], str) else row["options"]
                except Exception:
                    options = []
                images = extract_images_from_parquet(row)
                if not images:
                    res = {"single_result": "error", "majority_result": "error", "votes": {}}
                else:
                    res = evaluate_majority(images, row["question"], options)

                buf.append({
                    "id": row["id"],
                    "question": row["question"],
                    "options": options,
                    "answer": row["answer"],
                    "images": [f"image_{i}" for i in range(1,8) if row.get(f"image_{i}")],
                    "qwen_single": res["single_result"],
                    "qwen_majority": res["majority_result"],
                    "qwen_votes": res["votes"]
                })

                if len(buf) >= SAVE_INTERVAL:
                    w.write_all(buf)
                    buf.clear()
        if buf:
            w.write_all(buf)
    logging.info(f"Finished! → {OUTPUT_FILE}")

# --------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    main()
