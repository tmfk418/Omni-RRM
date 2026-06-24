#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch Video Preference Evaluation (Single GPU · Nested JSON Prompt)
-------------------------------------------------------------------
• Based on Swift PtEngine + LoRA merged full-parameter model
• Input: JSONL (containing video / query / answer_a / answer_b)
• Output: Appends fields qwen_better / qwen_reason (nested JSON string)
"""

# --------------------------------------------------------------------
# 1. Environment and Dependencies
# --------------------------------------------------------------------
import os
os.environ["CUDA_VISIBLE_DEVICES"]        = "0"          # Use only GPU 0
os.environ["PYTORCH_CUDA_ALLOC_CONF"]     = "expandable_segments:True"
os.environ["MAX_PIXELS"]                  = "1003520"
os.environ["VIDEO_MAX_PIXELS"]            = "50176"
os.environ["FPS_MAX_FRAMES"]              = "20"

import json, re, jsonlines, random, gc, torch, textwrap, logging
from pathlib import Path
from tqdm import tqdm
from peft import PeftModel
from swift.llm import (
    PtEngine, RequestConfig, InferRequest,
    get_model_tokenizer, get_template
)

# --------------------------------------------------------------------
# 2. File and Model Paths
# --------------------------------------------------------------------
INPUT_FILE   = "/path/to/input.jsonl"
OUTPUT_FILE  = "/path/to/output.jsonl"
SAVE_INTERVAL = 1                                    # Save every N samples

BASE_MODEL = "/path/to/base_model"
LORA_CKPT  = "/path/to/lora_checkpoint"

# --------------------------------------------------------------------
# 3. Load Base Model + Merge LoRA
# --------------------------------------------------------------------
print("🚀 Loading base model …")
base_model, processor = get_model_tokenizer(BASE_MODEL)

print("🔗 Merging LoRA …")
base_model = PeftModel.from_pretrained(base_model, LORA_CKPT).merge_and_unload()
base_model.to("cuda:0")                                # Explicitly move to GPU 0

print("⚙️  Building PtEngine …")
template = get_template(base_model.config.model_type, processor)
engine   = PtEngine.from_model_template(
    model=base_model,
    template=template,
    max_batch_size=1,                                  # Adjust based on GPU memory
)

# --------------------------------------------------------------------
# 4. Prompt Construction
# --------------------------------------------------------------------
def build_prompt(q: str, a: str, b: str) -> str:
    """Return evaluation prompt requiring a single-layer JSON output"""
    example = (
        '{'
        '"score_A": 9, '
        '"score_B": 7, '
        '"better": "A", '
        '"reasoning": "Part 1 …", '
        '"final_verdict": "<answer>[[A]]</answer>"'
        '}'
    )
    return textwrap.dedent(f"""
        You are a helpful and thoughtful AI assistant with experience in multimodal reasoning.

        ### Task
        Two candidate answers (Model A & Model B) are provided for a question related to a video.
        Evaluate them based on FIVE dimensions:
        1. Fluency and Coherence
        2. Relevance to the Question and Video
        3. Accuracy and Completeness
        4. Reasoning Quality
        5. Safety and Ethical Alignment

        Assign each model an integer score (0–10) on overall quality,
        decide which answer is better ("A", "B", "equal"),
        and provide detailed reasoning covering all five dimensions.

        ### STRICT Output Requirements
        - Output **one valid JSON object**. Do NOT include markdown, code fences, or extra keys.
        - Keys (exact names): score_A, score_B, better, reasoning, final_verdict
        - better ∈ {{ "A", "B", "equal" }}
        - final_verdict must repeat the better answer: "<answer>[[A]]</answer>" / "<answer>[[B]]</answer>" / "<answer>[[equal]]</answer>"
        - No additional keys, no comments.

        ### Example
        {example}

        ### Question
        {q}

        ### Model A Answer
        {a}

        ### Model B Answer
        {b}
    """).strip()

# --------------------------------------------------------------------
# 5. Parsing and Validation
# --------------------------------------------------------------------
_single_json_re = re.compile(r"\{.*\}", re.S)

def _safe_int(val, default=None):
    try:
        return int(val)
    except Exception:
        return default

def validate_result(obj: dict, outer_better: str):
    """
    Check score ranges, key completeness, and consistency between final_verdict and better.
    Returns status: ok / invalid
    """
    keys = {"score_A", "score_B", "better", "reasoning", "final_verdict"}
    if not keys.issubset(obj):
        return "invalid"

    score_a = _safe_int(obj["score_A"])
    score_b = _safe_int(obj["score_B"])
    if score_a is None or score_b is None or not (0 <= score_a <= 10) or not (0 <= score_b <= 10):
        return "invalid"

    if obj["better"] not in ("A", "B", "equal"):
        return "invalid"

    expected_tag = f"<answer>[[{obj['better']}]]</answer>"
    if obj["final_verdict"].strip() != expected_tag:
        return "invalid"

    # Ensure outer better (legacy format) matches if present
    if outer_better and outer_better != obj["better"]:
        return "invalid"

    return "ok"

def parse_model_output(text: str):
    """
    Supports two formats:
      1) Single-layer JSON   => Direct parsing
      2) Legacy format {better, reasoning: json-string} => Extract and parse
    Returns dict: {better, score_A, score_B, final_verdict, reasoning, status}
    """
    m = _single_json_re.search(text)
    if not m:
        logging.warning("No JSON found in output")
        return {"status": "invalid"}

    seg = m.group(0)
    try:
        obj = json.loads(seg)
        if isinstance(obj, dict) and {"score_A", "score_B"}.issubset(obj):
            status = validate_result(obj, outer_better="")
            obj["status"] = status
            return obj
        if "better" in obj and "reasoning" in obj and isinstance(obj["reasoning"], str):
            inner_match = _single_json_re.search(obj["reasoning"])
            if inner_match:
                inner = json.loads(inner_match.group(0))
                status = validate_result(inner, outer_better=obj.get("better"))
                inner["status"] = status
                return inner
    except Exception as e:
        logging.warning(f"JSON parse error: {e}")

    return {"status": "invalid"}

# --------------------------------------------------------------------
# 6. Single Sample Evaluation
# --------------------------------------------------------------------
def evaluate(video: str, q: str, a: str, b: str):
    prompt = build_prompt(q, a, b)
    req = InferRequest(
        messages=[{"role": "user", "content": "<video>\n" + prompt}],
        videos=[video]
    )
    cfg   = RequestConfig(max_tokens=2048, temperature=0)
    try:
        resp = engine.infer([req], cfg)[0]
        content = resp.choices[0].message.content.strip()
        result = parse_model_output(content)
        return result
    except Exception as e:
        return {"status": "error", "reasoning": str(e)}
    finally:
        torch.cuda.empty_cache(); gc.collect()

# --------------------------------------------------------------------
# 7. Main Process
# --------------------------------------------------------------------
def main():
    processed_ids = set()
    if Path(OUTPUT_FILE).exists():
        with jsonlines.open(OUTPUT_FILE) as r:
            processed_ids = {line["id"] for line in r}
        logging.info(f"Loaded {len(processed_ids)} existing rows")

    todo = []
    with jsonlines.open(INPUT_FILE) as r:
        for row in r:
            if row["id"] not in processed_ids:
                row["video"] = row["video"].replace("\\", "/").strip()
                todo.append(row)
    logging.info(f"Need to process {len(todo)} samples")

    buf = []
    pbar = tqdm(total=len(todo), desc="Infer")
    for row in todo:
        if not os.path.exists(row["video"]):
            res = {"status": "error", "reasoning": f"video not found: {row['video']}"}
        else:
            res = evaluate(row["video"], row["query"], row["answer_a"], row["answer_b"])

        if res.get("status") != "ok":
            logging.warning(f"id={row['id']} invalid result: {res}")

        buf.append({**row,
                    "qwen_status":         res.get("status", "error"),
                    "qwen_better":         res.get("better", "error"),
                    "qwen_score_A":        res.get("score_A"),
                    "qwen_score_B":        res.get("score_B"),
                    "qwen_final_verdict":  res.get("final_verdict"),
                    "qwen_reason":         res.get("reasoning")
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
