#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio Preference Evaluation — Local Qwen2.5‑Omni (LoRA) Version
---------------------------------------------------------------
* Inputs  : JSONL with `audio_prompt`, `answer_a`, `answer_b`
* Outputs : same JSONL + `better`, `reasoning`
* Model   : LoRA‑merged Qwen2.5‑Omni‑3B, inference with Swift PtEngine
"""

import os, json, textwrap, re, io, base64
from typing import Dict, Any, List

# ───────────────────────────────────────────────────────────────────
# 0. Environment Variables & Paths (adjust as needed)
# ───────────────────────────────────────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["MAX_PIXELS"] = "1003520"
os.environ["VIDEO_MAX_PIXELS"] = "50176"
os.environ["FPS_MAX_FRAMES"] = "20"

BASE_MODEL = "/path/to/Qwen2.5-Omni-3B"
LORA_CKPT  = "/path/to/lora_checkpoint"

SRC_JSONL = "/path/to/input.jsonl"
DST_JSONL = "/path/to/output.jsonl"

# ───────────────────────────────────────────────────────────────────
# 1. Swift / PtEngine Initialization (auto-detect LoRA merge method)
# ───────────────────────────────────────────────────────────────────
from swift.llm import (
    PtEngine, RequestConfig, InferRequest,
    get_model_tokenizer, get_template
)
from swift.tuners import Swift
from peft import PeftModel  # fallback

def build_engine() -> "PtEngine":
    """Return a PtEngine instance with merged LoRA weights (auto-detect method)."""
    model, processor = get_model_tokenizer(BASE_MODEL)

    # Route A: Swift.merge_lora
    if hasattr(Swift, "merge_lora"):
        try:
            model = Swift.merge_lora(model, LORA_CKPT)
            print("[INFO] LoRA merged via Swift.merge_lora")
        except Exception:
            pass

    # Route B: PtEngine.from_pretrained(lora_path=…)
    if not isinstance(model, PtEngine):
        if hasattr(PtEngine, "from_pretrained"):
            try:
                eng = PtEngine.from_pretrained(model=BASE_MODEL,
                                               lora_path=LORA_CKPT,
                                               max_batch_size=4)
                print("[INFO] LoRA merged via PtEngine.from_pretrained")
                return eng
            except TypeError:
                pass

    # Route C: Manual PEFT merge
    if not hasattr(model, "peft_config"):
        model = PeftModel.from_pretrained(model, LORA_CKPT)
        model = model.merge_and_unload()
        print("[INFO] LoRA merged via PEFT")

    template = get_template(model.config.model_type, processor)
    engine = PtEngine.from_model_template(model=model,
                                          template=template,
                                          max_batch_size=4)
    return engine

ENGINE = build_engine()
REQ_CFG = RequestConfig(max_tokens=2048, temperature=0)

# ───────────────────────────────────────────────────────────────────
# 2. Prompt Construction & JSON Extraction
# ───────────────────────────────────────────────────────────────────
def build_prompt(question: str, a1: str, a2: str) -> str:
    return textwrap.dedent(f"""
    You are a helpful and thoughtful AI assistant with experience in multimodal reasoning.
    ### Task
    Two candidate answers (Model A & Model B) are provided for a question related to an audio.
    Your task is to analyze and give a comparative evaluation of their quality and accuracy based on FIVE key dimensions.

    **Evaluation Dimensions**
    1. Fluency and Coherence
    2. Relevance to the Question and Audio
    3. Accuracy and Completeness
    4. Reasoning Quality
    5. Safety and Ethical Alignment

    **Evaluation Process**
    1. First, imagine the most ideal and factually accurate answer to the question based on the audio and question context. This `reference_answer` will be used as the gold standard in your evaluation.
    2. Evaluate both answers across all five dimensions.
    3. Assign each model an integer score from 0 to 10 based on the dimensional analysis.
    4. Determine which model performed better overall ("A", "B", or "equal").
    5. Provide detailed reasoning covering all five dimensions.

    **Output Instructions**
    - Your output must be a strictly valid JSON object.
    - Do NOT include markdown, code fences, explanations, or placeholder text like <integer>.
    - All field names and string values must be enclosed in double quotes.
    - Make sure the reasoning is enclosed in a single string under the "reasoning" key.
    - The final verdict should match the better model inside: "<answer>[[A]]</answer>", "<answer>[[B]]</answer>", or "<answer>[[equal]]</answer>".

    ### Required Output Keys
    {{
      "score_A": [integer between 0 and 10],
      "score_B": [integer between 0 and 10],
      "better": "A" or "B" or "equal",
      "reasoning": "<think>...</think>",
      "final_verdict": "<answer>[[A]]</answer>"
    }}

    ### Input Data
    [Question]: {question}
    [Answer A]: {a1}
    [Answer B]: {a2}
    """).strip()


_JSON_RE = re.compile(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", re.DOTALL)
def extract_json(txt: str) -> str:
    m = _JSON_RE.search(txt)
    if m:
        candidate = m.group(1)
    else:
        s, e = txt.find('{'), txt.rfind('}')
        candidate = txt[s:e+1] if s != -1 and e != -1 and e > s else txt
    return re.sub(r"\bNone\b", "", candidate)

# ───────────────────────────────────────────────────────────────────
# 3. Inference: Call Local Qwen
# ───────────────────────────────────────────────────────────────────
def run_inference(item: Dict[str, Any]) -> Dict[str, Any]:
    path, a1, a2 = item["audio_prompt"], item["answer_a"], item["answer_b"]
    q_txt = item.get("question", "What is the audio talking about?")
    prompt = build_prompt(q_txt, a1, a2)

    messages = [{"role": "user", "content": f"<audio>{prompt}"}]
    req = InferRequest(messages=messages, audios=[path])

    try:
        rsp = ENGINE.infer([req], REQ_CFG)[0]
        txt = rsp.choices[0].message.content
        result = json.loads(extract_json(txt))   # parse into dict
        return result
    except Exception as e:
        print(f"[ERROR] {os.path.basename(path)}: {e}")
        return {"better": "error", "reasoning": str(e)}

# ───────────────────────────────────────────────────────────────────
# 4. JSONL I/O with Resume Capability
# ───────────────────────────────────────────────────────────────────
def convert_bytes(x):
    if isinstance(x, dict):
        return {k: convert_bytes(v) for k, v in x.items()}
    if isinstance(x, list):
        return [convert_bytes(i) for i in x]
    if isinstance(x, bytes):
        return x.decode("utf-8", "ignore")
    return x

def safe_dumps(obj):
    return json.dumps(convert_bytes(obj), ensure_ascii=False)

def process_jsonl(src: str, dst: str):
    done = set()
    if os.path.exists(dst):
        with open(dst, encoding="utf-8") as f:
            for l in f:
                try:
                    done.add(json.loads(l)["audio_prompt"])
                except Exception:
                    pass
    print("[INFO] already processed:", len(done))

    with open(src, encoding="utf-8") as fin, open(dst, "a", encoding="utf-8") as fout:
        for line in fin:
            item = json.loads(line)
            if item["audio_prompt"] in done:
                continue
            res = run_inference(item)
            item.update(res)
            fout.write(safe_dumps(item) + "\n")
            fout.flush()
            print("[✓]", os.path.basename(item["audio_prompt"]), "→", res.get("better"))

# ───────────────────────────────────────────────────────────────────
# 5. Entry Point
# ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    process_jsonl(SRC_JSONL, DST_JSONL)
