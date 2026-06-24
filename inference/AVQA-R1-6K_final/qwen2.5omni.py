#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Audio + Image Multiple-Choice QA Inference
(JSON input · Multi-vote mode + JSON fault-tolerant parsing + retry for errors)
--------------------------------------------------------------------
• Input file: omni_rl_format_valid.json
• Uses fully fine-tuned Swift model → single run + majority voting
• Output: predictions.jsonl
• Feature: Retry error cases and overwrite results in the original JSONL
"""

import os, json, gc, torch, logging, re
from tqdm import tqdm
from collections import Counter
import jsonlines
import argparse

from swift.llm import (
    PtEngine, RequestConfig, InferRequest,
    get_model_tokenizer, get_template
)

# --------------------------------------------------------------------
# 1. Configuration
# --------------------------------------------------------------------
BASE_DIR       = "/path/to/AVQA-R1-6K"
INPUT_FILE     = os.path.join(BASE_DIR, "omni_rl_format_valid.json")
OUTPUT_FILE    = os.path.join(BASE_DIR, "predictions.jsonl")
FULL_MODEL_DIR = "/path/to/Qwen2.5-Omni-3B"
SAVE_INTERVAL  = 5

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# --------------------------------------------------------------------
# 2. Load Model
# --------------------------------------------------------------------
logging.info("🚀 Loading fully fine-tuned model …")
model, processor = get_model_tokenizer(
    FULL_MODEL_DIR,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

logging.info("⚙️ Building PtEngine …")
template = get_template(model.config.model_type, processor)
engine   = PtEngine.from_model_template(model=model,
                                        template=template,
                                        max_batch_size=1)

# --------------------------------------------------------------------
# 3. Prompt Construction
# --------------------------------------------------------------------
def build_av_prompt(question: str, options: list[str]) -> str:
    opts_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
    return f"""
You are a knowledgeable AI assistant.
Please carefully analyze the given audio and image content to answer the multiple-choice question.
Select the single most accurate answer letter.

### Question
{question}

### Options
{opts_text}

### Output Format
{{"answer": "A"|"B"|"C"|"D"}}

### IMPORTANT
- Only output a valid JSON object.
- Do not include explanation or extra text.
""".strip()

# --------------------------------------------------------------------
# 4. Safe JSON Parsing (Fault Tolerant)
# --------------------------------------------------------------------
def safe_parse_json(content: str):
    try:
        return json.loads(content)
    except:
        pass

    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except:
            pass

    m = re.search(r"\{.*\}", content, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except:
            pass

    m = re.search(r"\b([ABCD])\b", content)
    if m:
        return {"answer": m.group(1)}

    return {"answer": "error"}

# --------------------------------------------------------------------
# 5. Single Inference (Audio + Image)
# --------------------------------------------------------------------
def evaluate_once_av(audio_path, image_path, question: str, options: list[str]) -> str:
    prompt = build_av_prompt(question, options)
    req = InferRequest(
        messages=[{"role": "user", "content": "<audio><image>\n" + prompt}],
        audios=[audio_path] if audio_path else None,
        images=[image_path] if image_path else None
    )
    cfg = RequestConfig(max_tokens=32, temperature=0.7)
    try:
        resp = engine.infer([req], cfg)[0]
        content = resp.choices[0].message.content.strip()
        obj = safe_parse_json(content)
        ans = obj.get("answer", "error").strip().upper()
        if ans not in ["A", "B", "C", "D"]:
            return "error"
        return ans
    except Exception as e:
        logging.warning(f"Parse failed: {str(e)}")
        return "error"
    finally:
        torch.cuda.empty_cache()
        gc.collect()

# --------------------------------------------------------------------
# 6. Multi-Run Inference + Majority Voting
# --------------------------------------------------------------------
def evaluate_majority_av(audio_path, image_path, question, options, runs=5):
    answers = []
    for i in range(runs):
        ans = evaluate_once_av(audio_path, image_path, question, options)
        print(f"[Run {i+1}] Answer: {ans}")
        if ans != "error":
            answers.append(ans)
    if not answers:
        return {"all_runs": [], "majority_result": "error", "votes": {}}
    counts = Counter(answers)
    majority_ans = counts.most_common(1)[0][0]
    print(f"=== Majority Result: {majority_ans} | Votes: {dict(counts)} ===")
    return {
        "all_runs": answers,
        "majority_result": majority_ans,
        "votes": dict(counts)
    }

# --------------------------------------------------------------------
# 7. Main Inference Process
# --------------------------------------------------------------------
def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    buf = []
    with jsonlines.open(OUTPUT_FILE, "w") as w:
        for row in tqdm(data, desc="Processing AVQA"):
            options = row.get("options", [])
            audio_path = os.path.join(BASE_DIR, row["path"]["audio"])
            image_path = os.path.join(BASE_DIR, row["path"]["image"])

            if not os.path.exists(audio_path):
                logging.warning(f"⚠️ Missing audio: {audio_path}")
                audio_path = None
            if not os.path.exists(image_path):
                logging.warning(f"⚠️ Missing image: {image_path}")
                image_path = None

            res = evaluate_majority_av(audio_path, image_path, row["problem"], options)

            buf.append({
                "id": row["problem_id"],
                "question": row["problem"],
                "options": options,
                "answer": row.get("solution"),
                "audio_path": audio_path,
                "image_path": image_path,
                "qwen_all_runs": res["all_runs"],
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
# 8. Retry Error Cases and Overwrite Original JSONL
# --------------------------------------------------------------------
def retry_errors(jsonl_path: str, runs=5):
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = {row["problem_id"]: row for row in json.load(f)}

    updated_records = []
    with jsonlines.open(jsonl_path, "r") as reader:
        for record in reader:
            if record.get("qwen_majority") == "error" and record["id"] in data:
                print(f"[Retry] Re-evaluating id={record['id']}")
                row = data[record["id"]]
                options = row.get("options", [])
                audio_path = os.path.join(BASE_DIR, row["path"]["audio"])
                image_path = os.path.join(BASE_DIR, row["path"]["image"])

                res = evaluate_majority_av(audio_path, image_path, row["problem"], options, runs=runs)
                record["qwen_all_runs"] = res["all_runs"]
                record["qwen_majority"] = res["majority_result"]
                record["qwen_votes"] = res["votes"]
            updated_records.append(record)

    with jsonlines.open(jsonl_path, "w") as writer:
        writer.write_all(updated_records)

    print(f"✅ Retry finished, results overwritten in {jsonl_path}")

# --------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retry", action="store_true", help="Retry only error items in predictions.jsonl")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs per item")
    args = parser.parse_args()

    torch.manual_seed(42)

    if args.retry:
        retry_errors(OUTPUT_FILE, runs=args.runs)
    else:
        main()
