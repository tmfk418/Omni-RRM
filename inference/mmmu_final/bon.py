#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pairwise_best_from_qwen_runs.py
────────────────────────────────────────────────────────────────────────────
Based on a LoRA fine-tuned Qwen2.5-Omni model, this script performs pairwise 
elimination comparisons among candidate outputs in `qwen_all_runs` 
to select the best answer.

Inputs:
    • Parquet folder (provides question, images, options)
    • JSONL file (provides qwen_all_runs, answer)

Output JSONL fields:
    id, question, qwen_all_runs, selected_best, gt_answer
"""

import os, sys, json, base64, logging, argparse, tempfile
from pathlib import Path
from io import BytesIO
from typing import List, Dict, Any

import pandas as pd
import jsonlines
import torch
from PIL import Image
from peft import PeftModel
from swift.llm import PtEngine, RequestConfig, InferRequest, get_model_tokenizer, get_template

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

# ───────────── Image Handling ─────────────
def _decode_img(b64: str) -> str | None:
    """Convert base64 to a temporary file path"""
    try:
        img = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        img.save(tmp, format="JPEG")
        tmp.close()
        return tmp.name
    except Exception:
        return None

def extract_images_from_row(row: Dict[str, Any]) -> List[str]:
    imgs = []
    if isinstance(row.get("image"), dict):
        b64 = row["image"].get("bytes")
        if b64 and (path := _decode_img(b64)):
            imgs.append(path)
    for i in range(1, 8):
        key = f"image_{i}"
        if isinstance(row.get(key), dict):
            b64 = row[key].get("bytes")
            if b64 and (path := _decode_img(b64)):
                imgs.append(path)
    return imgs

# ───────────── Prompt Construction ─────────────
def build_prompt(question: str, ans_a: str, ans_b: str) -> str:
    return (
        "You are an expert judge for multimodal QA comparison.\n"
        "Compare Answer A and Answer B for the given question and image(s).\n"
        "Strictly output JSON: {\"better\": \"A\"|\"B\"|\"equal\"}\n\n"
        f"[Question] {question}\n"
        f"[Answer A] {ans_a}\n"
        f"[Answer B] {ans_b}"
    )

# ───────────── Model Comparison ─────────────
def compare_answers(engine, tokenizer, template, request_cfg,
                    question: str, images: List[str],
                    ans_a: str, ans_b: str) -> str:
    prompt = build_prompt(question, ans_a, ans_b)
    try:
        infer_request = InferRequest(
            messages=[{"role": "user", "content": prompt}],
            images=images
        )
        responses = engine.infer([infer_request], request_cfg)
        resp_text = responses[0].choices[0].message.content.strip()
        out_json = json.loads(resp_text)
        return out_json.get("better", "equal")
    except Exception as e:
        logging.warning(f"[compare] JSON parse failed → {e}")
        return "equal"

# ───────────── Pairwise Elimination ─────────────
def select_best(engine, tokenizer, template, request_cfg,
                sample_id, question, images, runs: List[str], gt_answer: str):
    labels = [chr(ord('A') + i) for i in range(len(runs))]
    best_lbl, best_ans = labels[0], runs[0]

    for lbl, ans in zip(labels[1:], runs[1:]):
        res = compare_answers(engine, tokenizer, template, request_cfg,
                              question, images, best_ans, ans)
        if res == "B":
            best_lbl, best_ans = lbl, ans

    return {
        "id": sample_id,
        "question": question,
        "qwen_all_runs": runs,
        "selected_best": best_ans,   # Output the selected answer
        "gt_answer": gt_answer       # Ground truth answer
    }

# ───────────── Main Process ─────────────
def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.cuda.set_device(0)

    logging.info("🚀 Loading base model …")
    base_model, tokenizer = get_model_tokenizer(args.base_model)

    logging.info("🔗 Merging LoRA …")
    base_model = PeftModel.from_pretrained(
        base_model, args.lora_ckpt, is_trainable=False
    ).merge_and_unload()
    base_model.to("cuda:0")

    template = get_template(base_model.config.model_type, tokenizer)
    engine   = PtEngine.from_model_template(
        model=base_model, template=template, max_batch_size=1
    )

    request_cfg = RequestConfig(max_tokens=args.max_new_tokens,
                                temperature=args.temp,
                                stream=False)

    # Load parquet files → Build id index
    parquet_files = list(Path(args.input_dir).rglob("*.parquet"))
    if not parquet_files:
        logging.error("No parquet files found!"); sys.exit(1)
    logging.info(f"Found {len(parquet_files)} parquet files")

    id2row = {}
    for pq in parquet_files:
        df = pd.read_parquet(pq)
        for _, row in df.iterrows():
            if "id" in row:
                id2row[row["id"]] = row

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    writer = jsonlines.open(args.output, "w")

    buf, processed = [], 0
    with jsonlines.open(args.jsonl_input, "r") as reader:
        for entry in reader:
            sid = entry["id"]
            if sid not in id2row:
                logging.warning(f"[{sid}] not found in parquet")
                continue
            row = id2row[sid]
            runs = entry.get("qwen_all_runs", [])
            gt_answer = entry.get("answer", None)   # Extract ground truth
            if not runs or len(runs) < 2:
                continue

            imgs = extract_images_from_row(row)
            result = select_best(engine, tokenizer, template, request_cfg,
                                 sample_id=sid,
                                 question=row.get("question", ""),
                                 images=imgs,
                                 runs=runs,
                                 gt_answer=gt_answer)
            buf.append(result)
            processed += 1

            if len(buf) >= args.save_interval:
                writer.write_all(buf)
                buf.clear()
                logging.info(f"Saved {processed} samples …")

    if buf:
        writer.write_all(buf)
    writer.close()
    logging.info(f"✅ Finished. Total processed: {processed}")
    logging.info(f"Results saved to {args.output}")

# ───────────── CLI ─────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pairwise best selector from qwen_all_runs")
    parser.add_argument("--base_model", default="/path/to/Qwen2.5-Omni-3B")
    parser.add_argument("--lora_ckpt",  default="/path/to/lora_checkpoint")
    parser.add_argument("--input_dir",  default="/path/to/data", help="Parquet files directory")
    parser.add_argument("--jsonl_input", default="/path/to/final.jsonl", help="JSONL file containing qwen_all_runs")
    parser.add_argument("--output",     default="best_from_runs.jsonl")
    parser.add_argument("--gpu",        default="0")
    parser.add_argument("--temp",       type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--save_interval",  type=int, default=20)
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()
 
    main(args)
