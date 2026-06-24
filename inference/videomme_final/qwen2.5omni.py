#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Video Multiple-Choice Inference (Single GPU · Multi-Vote Version)
-----------------------------------------------------------------------
• Based on Swift PtEngine + fully fine-tuned model
• Input: JSONL (with video_path / question / options / answer)
• Output: Appends fields qwen_single / qwen_majority / qwen_votes
"""

# --------------------------------------------------------------------
# 1. Environment and Dependencies
# --------------------------------------------------------------------
import os
os.environ["CUDA_VISIBLE_DEVICES"]        = "0"          # Use GPU 0
os.environ["PYTORCH_CUDA_ALLOC_CONF"]     = "expandable_segments:True"
os.environ["MAX_PIXELS"]                  = "1003520"
os.environ["VIDEO_MAX_PIXELS"]            = "50176"
os.environ["FPS_MAX_FRAMES"]              = "20"

import json, jsonlines, random, gc, torch, logging
from pathlib import Path
from tqdm import tqdm
from collections import Counter

from swift.llm import (
    PtEngine, RequestConfig, InferRequest,
    get_model_tokenizer, get_template
)

# --------------------------------------------------------------------
# 2. File and Model Paths
# --------------------------------------------------------------------
INPUT_FILE    = "/path/to/input.jsonl"
OUTPUT_FILE   = "/path/to/output.jsonl"
SAVE_INTERVAL = 1    # Save every N processed samples
FULL_MODEL_DIR = "/path/to/full_model_dir"

# --------------------------------------------------------------------
# 3. Load Model
# --------------------------------------------------------------------
logging.info("🚀 Loading fully fine-tuned model …")
model, processor = get_model_tokenizer(FULL_MODEL_DIR)
model.to("cuda:0")

logging.info("⚙️ Building PtEngine …")
template = get_template(model.config.model_type, processor)
engine   = PtEngine.from_model_template(model=model,
                                        template=template,
                                        max_batch_size=1)

# --------------------------------------------------------------------
# 4. Prompt Construction
# --------------------------------------------------------------------
def build_mcq_prompt(question: str, options: list[str]) -> str:
    opts_text = "\n".join(options)
    return f"""
You are a knowledgeable AI assistant. 
Please carefully analyze the given video and the multiple-choice question.
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
# 5. Single Inference
# --------------------------------------------------------------------
def evaluate_once(video: str, question: str, options: list[str]) -> str:
    prompt = build_mcq_prompt(question, options)
    req = InferRequest(
        messages=[{"role": "user", "content": "<video>\n" + prompt}],
        videos=[video],
        images=None
    )
    cfg = RequestConfig(max_tokens=256, temperature=0.7)  # temperature > 0 allows diversity
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
# 6. Multi-Inference with Majority Voting
# --------------------------------------------------------------------
def evaluate_majority(video: str, question: str, options: list[str], runs: int = 5):
    answers = []
    for _ in range(runs):
        ans = evaluate_once(video, question, options)
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
# 7. Main Process
# --------------------------------------------------------------------
def main():
    processed_ids = set()
    if Path(OUTPUT_FILE).exists():
        with jsonlines.open(OUTPUT_FILE) as r:
            processed_ids = {line.get("id") for line in r if "id" in line}
        logging.info(f"Loaded {len(processed_ids)} existing rows")

    todo = []
    with jsonlines.open(INPUT_FILE) as r:
        for idx, row in enumerate(r):
            row_id = row.get("id", f"row_{idx}")
            if row_id not in processed_ids:
                row["id"] = row_id
                row["video_path"] = row["video_path"].replace("\\", "/").strip()
                todo.append(row)

    logging.info(f"Need to process {len(todo)} samples")

    buf = []
    pbar = tqdm(total=len(todo), desc="Infer")
    for row in todo:
        if not os.path.exists(row["video_path"]):
            res = {"single_result": "error", "majority_result": "error", "votes": {}}
            logging.warning(f"video not found: {row['video_path']}")
        else:
            res = evaluate_majority(row["video_path"], row["question"], row["options"])

        buf.append({
            **row,
            "qwen_single": res["single_result"],
            "qwen_majority": res["majority_result"],
            "qwen_votes": res["votes"]
        })

        if len(buf) >= SAVE_INTERVAL:
            with jsonlines.open(OUTPUT_FILE, "a") as w:
                w.write_all(buf)
            buf.clear()

        pbar.update(1)

    if buf:
        with jsonlines.open(OUTPUT_FILE, "a") as w:
            w.write_all(buf)
    pbar.close()
    logging.info(f"Finished! → {OUTPUT_FILE}")

# --------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)
    main()
