#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pairwise_best_from_qwen_votes_video_uniform.py
────────────────────────────────────────────────────────────────────────────
Based on a LoRA fine-tuned Qwen2.5-Omni model, perform pairwise elimination 
comparison among candidate outputs in qwen_votes to select the best answer, 
using uniformly sampled video frames.

Input JSONL fields:
    video_path, question, options, answer, id, qwen_votes
Output JSONL fields:
    id, question, qwen_votes, selected_best, gt_answer
"""

import os, sys, json, logging, argparse, tempfile
from pathlib import Path
from typing import Dict, Any, List

import jsonlines
import torch
import cv2
import numpy as np
from peft import PeftModel
from swift.llm import PtEngine, RequestConfig, InferRequest, get_model_tokenizer, get_template

os.environ["VIDEO_MAX_PIXELS"] = "50176"

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

# ───────────── Uniform Frame Extraction ─────────────
def extract_uniform_frames(video_path: str, max_frames: int) -> List[str]:
    """Extract max_frames frames uniformly from a video and save them as temp files. Returns list of paths."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.warning(f"⚠️ Failed to open video: {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        return []

    step = max(1, total_frames // max_frames)
    frame_indices = list(range(0, total_frames, step))[:max_frames]

    temp_paths = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        cv2.imwrite(temp_file.name, frame)
        temp_paths.append(temp_file.name)

    cap.release()
    return temp_paths

# ───────────── Prompt Construction ─────────────
def build_prompt(question: str, ans_a: str, ans_b: str) -> str:
    return (
        "You are an expert judge for multimodal QA comparison.\n"
        "Compare Answer A and Answer B for the given question and uniformly sampled video frames.\n"
        "You must ONLY output a JSON object in the form: {\"better\": \"A\"|\"B\"|\"equal\"}.\n"
        "Do not output anything else.\n\n"
        f"[Question] {question}\n"
        f"[Answer A] {ans_a}\n"
        f"[Answer B] {ans_b}"
    )

# ───────────── Fix Video Path ─────────────
def fix_video_path(path: str, video_root: str) -> str:
    if not path:
        return ""
    if path.startswith("/videos") and video_root:
        return path.replace("/videos", video_root, 1)
    return path

# ───────────── Model Comparison ─────────────
def compare_answers(engine, tokenizer, template, request_cfg,
                    question: str, video_path: str, max_frames: int,
                    ans_a: str, ans_b: str) -> str:
    frames = extract_uniform_frames(video_path, max_frames)
    if not frames:
        logging.warning(f"⚠️ No frames extracted from {video_path}")
        return "equal"

    prompt = build_prompt(question, ans_a, ans_b)
    try:
        infer_request = InferRequest(
            messages=[{"role": "user", "content": prompt}],
            images=frames
        )
        responses = engine.infer([infer_request], request_cfg)
        resp_text = responses[0].choices[0].message.content.strip()
        try:
            out_json = json.loads(resp_text)
            return out_json.get("better", "equal")
        except Exception:
            logging.warning(f"[compare] JSON parse fail → {resp_text}")
            return "equal"
    except Exception as e:
        logging.warning(f"[compare] Model call error → {e}")
        return "equal"
    finally:
        for f in frames:
            try:
                os.remove(f)
            except:
                pass

# ───────────── Pairwise Elimination ─────────────
def select_best(engine, tokenizer, template, request_cfg,
                sample_id, question, video_path, votes: Dict[str, int],
                gt_answer: str, max_frames: int):
    runs = list(votes.keys())

    if len(runs) == 1:
        only_ans = runs[0]
        return {
            "id": sample_id,
            "question": question,
            "qwen_votes": votes,
            "selected_best": only_ans,
            "gt_answer": gt_answer
        }

    best_lbl, best_ans = runs[0], runs[0]
    for lbl in runs[1:]:
        ans = lbl
        res = compare_answers(engine, tokenizer, template, request_cfg,
                              question, video_path, max_frames,
                              best_ans, ans)
        if res == "B":
            best_lbl, best_ans = lbl, ans

    return {
        "id": sample_id,
        "question": question,
        "qwen_votes": votes,
        "selected_best": best_ans,
        "gt_answer": gt_answer
    }

# ───────────── Main Pipeline ─────────────
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
            votes = entry.get("qwen_votes", {})
            gt_answer = entry.get("answer", None)
            video_path = fix_video_path(entry.get("video_path", ""), args.video_root)

            if not votes or not video_path:
                continue

            result = select_best(engine, tokenizer, template, request_cfg,
                                 sample_id=sid,
                                 question=entry.get("question", ""),
                                 video_path=video_path,
                                 votes=votes,
                                 gt_answer=gt_answer,
                                 max_frames=args.max_frames)
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
    parser = argparse.ArgumentParser(description="Pair-wise best selector from qwen_votes (Video, Uniform Sampling)")
    parser.add_argument("--base_model", default="/path/to/Qwen2.5-Omni-3B")
    parser.add_argument("--lora_ckpt",  default="/path/to/lora_checkpoint")
    parser.add_argument("--jsonl_input", default="/path/to/input.jsonl", help="JSONL file containing video_path")
    parser.add_argument("--output",     default="best_from_votes.jsonl")
    parser.add_argument("--gpu",        default="0")
    parser.add_argument("--temp",       type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--save_interval",  type=int, default=20)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--video_root", default="/path/to/videos",
                        help="Local path to video files, will replace the /videos prefix in JSONL")
    parser.add_argument("--max_frames", type=int, default=8,
                        help="Number of uniformly sampled video frames")
    args = parser.parse_args()

    main(args)
