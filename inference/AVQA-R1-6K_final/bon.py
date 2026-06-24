#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pairwise_best_from_qwen_audio_image_relpath.py
────────────────────────────────────────────────────────────────────────────
Based on a LoRA fine-tuned Qwen2.5-Omni model.  
Performs pairwise elimination comparison on qwen_all_runs or qwen_votes.  
Inputs include fixed audio and image files (supports relative paths → automatically joined with root).
"""

import os, sys, json, logging, argparse
from pathlib import Path
from typing import Dict, Any, List

import jsonlines
import torch
from peft import PeftModel
from swift.llm import PtEngine, RequestConfig, InferRequest, get_model_tokenizer, get_template

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

# ───────────── Path Fixing ─────────────
def fix_path(path: str, root: str) -> str:
    """If path is relative, join with root"""
    if not path:
        return ""
    p = Path(path)
    if not p.is_absolute():
        return str(Path(root) / p)
    return str(p)

# ───────────── Prompt Construction ─────────────
def build_prompt(question: str, ans_a: str, ans_b: str) -> str:
    return (
        "You are an expert judge for multimodal QA comparison.\n"
        "Compare Answer A and Answer B for the given question using both the image and audio.\n"
        "You must ONLY output a JSON object in the form: {\"better\": \"A\"|\"B\"|\"equal\"}.\n"
        "Do not output anything else.\n\n"
        f"[Question] {question}\n"
        f"[Answer A] {ans_a}\n"
        f"[Answer B] {ans_b}"
    )

# ───────────── Model Comparison ─────────────
def compare_answers(engine, tokenizer, template, request_cfg,
                    question: str, image_path: str, audio_path: str,
                    ans_a: str, ans_b: str) -> str:
    prompt = build_prompt(question, ans_a, ans_b)
    try:
        infer_request = InferRequest(
            messages=[{"role": "user", "content": prompt}],
            images=[image_path] if image_path else None,
            audios=[audio_path] if audio_path else None
        )
        responses = engine.infer([infer_request], request_cfg)
        resp_text = responses[0].choices[0].message.content.strip()
        try:
            out_json = json.loads(resp_text)
            return out_json.get("better", "equal")
        except Exception:
            logging.warning(f"[compare] JSON parse failed → {resp_text}")
            return "equal"
    except Exception as e:
        logging.warning(f"[compare] Model call error → {e}")
        return "equal"

# ───────────── Pairwise Elimination ─────────────
def select_best(engine, tokenizer, template, request_cfg,
                sample_id, question, image_path, audio_path,
                runs: List[str], gt_answer: str):
    if len(runs) < 2:
        return None

    best_ans = runs[0]

    for ans in runs[1:]:
        res = compare_answers(engine, tokenizer, template, request_cfg,
                              question, image_path, audio_path,
                              best_ans, ans)
        if res == "B":
            best_ans = ans

    return {
        "id": sample_id,
        "question": question,
        "qwen_all_runs": runs,
        "selected_best": best_ans,
        "gt_answer": gt_answer
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

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    writer = jsonlines.open(args.output, "w")

    buf, processed = [], 0
    with jsonlines.open(args.jsonl_input, "r") as reader:
        for entry in reader:
            sid = entry["id"]
            runs = entry.get("qwen_all_runs") or list(entry.get("qwen_votes", {}).keys())
            gt_answer = entry.get("answer", None)

            image_path = fix_path(entry.get("image_path", ""), args.image_root)
            audio_path = fix_path(entry.get("audio_path", ""), args.audio_root)

            if not runs or len(runs) < 2 or not (image_path and audio_path):
                continue

            result = select_best(engine, tokenizer, template, request_cfg,
                                 sample_id=sid,
                                 question=entry.get("question", ""),
                                 image_path=image_path,
                                 audio_path=audio_path,
                                 runs=runs,
                                 gt_answer=gt_answer)
            if result:
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
    parser = argparse.ArgumentParser(description="Pair-wise best selector (Audio+Image with relative paths)")
    parser.add_argument("--base_model", default="/path/to/Qwen2.5-Omni-3B")
    parser.add_argument("--lora_ckpt",  default="/path/to/lora_checkpoint")
    parser.add_argument("--jsonl_input", default="/path/to/data_final.jsonl", 
                        help="JSONL file containing audio_path and image_path")
    parser.add_argument("--output",     default="best_from_audio_image.jsonl")
    parser.add_argument("--gpu",        default="0")
    parser.add_argument("--temp",       type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--save_interval",  type=int, default=20)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--audio_root", default="/path/to/audios",
                        help="Root directory for audio files")
    parser.add_argument("--image_root", default="/path/to/images",
                        help="Root directory for image files")
    args = parser.parse_args()

    main(args)
