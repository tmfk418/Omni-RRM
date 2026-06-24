#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Local Qwen2.5-Omni Inference Script (Fixed Version)

Fixes：
1. Added json_serial function to handle non-serializable numpy arrays
2. Improved error handling logic
3. Optimized model loading process
"""
from typing import Any, Dict, Optional, Tuple, List  
import os
import re
import io
import json
import base64
import random
import argparse
import tempfile
import logging
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from datetime import datetime

from swift.llm import PtEngine, RequestConfig, get_model_tokenizer, get_template, InferRequest
from swift.tuners import Swift
import torch

# ✅ Multimodal resource limits: Prevent OOM
os.environ['MAX_PIXELS'] = '1003520'         # Max total image pixels
os.environ['VIDEO_MAX_PIXELS'] = '50176'     # Max pixels per video frame
os.environ['FPS_MAX_FRAMES'] = '20'          # Max number of video frames to sample

model_engine = None
request_config = None

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def prompt(data_obj, random_number):
    answers = [data_obj["response"][0], data_obj["response"][1]] if random_number == 0 else [data_obj["response"][1], data_obj["response"][0]]
    
    prompt_str = f"""
    You are a helpful and thoughtful AI assistant with experience in multimodal reasoning.
    ### Task
    Two candidate answers (Model A & Model B) are provided for a question related to a image.
    Your task is to analyze and give a comparative evaluation of their quality and accuracy based on FIVE key dimensions.

    **Evaluation Dimensions**
    1. Fluency and Coherence 
    2. Relevance to the Question and Image 
    3. Accuracy and Completeness 
    4. Reasoning Quality 
    5. Safety and Ethical Alignment 

    **Scoring Guidelines**
    - 9-10: Excellent in all dimensions
    - 6-8: Good overall with minor issues in 1-2 dimensions
    - 3-5: Deficient in 2-3 dimensions
    - 0-2: Poor in 4-5 dimensions

    **Evaluation Process**
    1. First, imagine the most ideal and factually accurate answer to the question based on the image and question context. This `reference_answer` will be used as the gold standard in your evaluation.
    2. Evaluate both answers across all five dimensions.
    3. Assign each model an integer score from 0 to 10 based on the dimensional analysis.
    4. Determine which model performed better overall ("A", "B", or "equal").
    5. Provide detailed reasoning covering all five dimensions.

    **Output Instructions**
    - Your output must be a **strictly valid JSON object**.
    - **Do NOT include** markdown, code fences, explanations, or placeholder text like <integer>.
    - All field names and string values must be enclosed in **double quotes**.
    - Make sure the reasoning is enclosed in a single string under the "reasoning" key.
    - The final verdict should match the better model inside: "<answer>[[A]]</answer>", "<answer>[[B]]</answer>", or "<answer>[[equal]]</answer>".

    ### Required Output Keys
    {{
      "score_A": [integer between 0 and 10],
      "score_B": [integer between 0 and 10],
      "better": "A" or "B" or "equal",
      "reasoning": "<think>Part 1: In terms of Fluency and Coherence, …  
       For Relevance to the Question and Image, …  
       Regarding Accuracy and Completeness, …  
       In terms of Reasoning Quality, …  
       Part 2: In terms of Safety and Ethical Alignment, …</think>",
      "final_verdict": "<answer>[[A]]</answer>"
    }}

    ### Evaluation Context
    Problem Statement: {data_obj["query"]}

    Candidate Solution 1:
    {answers[0]}

    Candidate Solution 2:
    {answers[1]}

    ### Critical Requirements
    ⚠️ OUTPUT MUST BE VALID JSON
    ⚠️ Include ALL specified fields exactly
    ⚠️ final_verdict MUST match better field
    ⚠️ Scores MUST be integers 0-10
    ⚠️ Provide CONCRETE examples from answers in reasoning

    ### Prohibited Content
    ❌ Do NOT include markdown/code formatting
    ❌ Do NOT add explanations outside JSON
    ❌ Do NOT modify field names/structures

    ### Validation Checklist
    1. All dimensional analyses are present
    2. Scores reflect rubric standards
    3. final_verdict format is exact
    4. JSON passes standard validation
    """
    return prompt_str
# ---------------------------------------------------------------------------
# JSON serialization helper function
# ---------------------------------------------------------------------------
def json_serial(obj):
    """Recursively handle non-serializable JSON objects"""
    if isinstance(obj, (np.ndarray, np.generic)):
        return obj.tolist()  # Convert numpy array to Python list
    if hasattr(obj, '__dict__'):
        return obj.__dict__  # Handle custom objects
    try:
        return str(obj)  # Finally try converting to string
    except Exception:
        return "unserializable"  # Ultimate fallback

# ---------------------------------------------------------------------------
# Helpers: image extraction
# ---------------------------------------------------------------------------
def _extract_image_bytes(image_field: Any) -> Optional[bytes]:
    """Best-effort extraction of raw image bytes from common parquet encodings."""
    if image_field is None:
        return None
    
    # HF datasets style dict
    if isinstance(image_field, dict):
        if "bytes" in image_field and image_field["bytes"] is not None:
            b = image_field["bytes"]
            if isinstance(b, bytes):
                return b
            if isinstance(b, bytearray):
                return bytes(b)
            if isinstance(b, str):
                try:
                    return base64.b64decode(b)
                except Exception:
                    return b.encode("utf-8")
        # path fallback
        if "path" in image_field and isinstance(image_field["path"], str) and os.path.exists(image_field["path"]):
            with open(image_field["path"], "rb") as f:
                return f.read()
        return None
    
    # raw bytes-like
    if isinstance(image_field, (bytes, bytearray, memoryview)):
        return bytes(image_field)
    
    # filesystem path string
    if isinstance(image_field, str) and os.path.exists(image_field):
        with open(image_field, "rb") as f:
            return f.read()
    return None

# ---------------------------------------------------------------------------
# Initialize local model
# ---------------------------------------------------------------------------
def initialize_model():
    """Initialize local model"""
    global model_engine, request_config
    
    logging.info("🚀 Initializing model......")
    
    try:
        base_model_path = '/path/to/resource'
        
        lora_checkpoint_path = '/path/to/resource'
        
        model, tokenizer = get_model_tokenizer(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        model = Swift.from_pretrained(model, lora_checkpoint_path)
        
        template_type = model.model_meta.template
        template = get_template(template_type, tokenizer)
        
        model_engine = PtEngine.from_model_template(
            model, 
            template, 
            max_batch_size=1
        )
        
        request_config = RequestConfig(
            max_tokens=4096, 
            temperature=0.7, 
            top_p=0.9
        )
        logging.info("✅ Model initialization complete")
    except Exception as e:
        logging.error(f"Model initialization failed: {str(e)}")
        raise

def call_local_model(prompt_text: str, image_bytes: Optional[bytes]) -> str:
    """
    Perform inference using local model
    Return: Model-generated text response
    """
    image_path = None
    
    try:
        if image_bytes:
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_img:
                temp_img.write(image_bytes)
                image_path = temp_img.name
        
        messages = [{"role": "user", "content": prompt_text}]
        images = [image_path] if image_path else []
        
        logging.debug(f"Sending inference request: prompt={prompt_text[:50]}... images={len(images)}")
        infer_request = InferRequest(messages=messages, images=images)
        
        responses = model_engine.infer([infer_request], request_config)
        
        return responses[0].choices[0].message.content.strip()
    
    except Exception as e:
        logging.error(f"Model inference failed: {str(e)}")
        return f"[LOCAL_ERROR] {str(e)}"
    
    finally:
        if image_path and os.path.exists(image_path):
            try:
                os.remove(image_path)
            except Exception as e:
                logging.warning(f"Failed to clean up temporary file: {str(e)}")

# ---------------------------------------------------------------------------
# Parse model output -> Answer# / flag_status (regex compatible w/ GPT script)

_ANS_PATTERN   = re.compile(
    r"(?:Overall\s*Judgment|Final\s*Decision|Therefore)?[^\n]*?Answer\s*([12])\s*"
    r"(?:is\s*)?(?:the\s*)?(?:slightly\s*)?(?:better|superior)",
    re.IGNORECASE
)
_BETTER_FIELD  = re.compile(r'"better"\s*:\s*"?(1|2)"?', re.I)          
_CODE_FENCE    = re.compile(r"```(?:json)?\s*(\{.*?})\s*```", re.S)    

_JSON_BLOCK    = re.compile(r"\{[^{}]*\"better\"[^{}]*\}", re.S)        
_BETTER_AB     = re.compile(r'"better"\s*:\s*"?([AaBb]|equal)"?', re.I) 

def _parse_flag(response_text: str, rejected_response_choice: int) -> Tuple[int, str]:

    raw = response_text.strip()

    m = _JSON_BLOCK.search(raw)
    if m:
        try:
            data = json.loads(m.group(0))
            better_val = str(data.get("better", "")).lower()
            if better_val in ("a", "1"):
                ans_num = 1
            elif better_val in ("b", "2"):
                ans_num = 2
            else:                             
                return -1, "doesntMatch"
            status = "reject" if ans_num - 1 == rejected_response_choice else "agree"
            return ans_num, status
        except Exception as e:
            logging.debug(f"[parse_flag] fail: {e}")

    try:
        data = json.loads(raw)
        better = str(data.get("better", "")).lower()
        if better in ("1", "a"):
            ans_num = 1
        elif better in ("2", "b"):
            ans_num = 2
        else:
            raise ValueError
        status = "reject" if ans_num - 1 == rejected_response_choice else "agree"
        return ans_num, status
    except Exception:
        pass

    m = _CODE_FENCE.search(raw)
    if m:
        try:
            data = json.loads(m.group(1))
            better = str(data.get("better", "")).lower()
            if better in ("1", "a"):
                ans_num = 1
            elif better in ("2", "b"):
                ans_num = 2
            else:
                raise ValueError
            status = "reject" if ans_num - 1 == rejected_response_choice else "agree"
            return ans_num, status
        except Exception as e:
            logging.debug(f"[parse_flag] fail: {e}")

    m = _BETTER_AB.search(raw)
    if m:
        val = m.group(1).lower()
        ans_num = 1 if val == "a" else 2       
        status = "reject" if ans_num - 1 == rejected_response_choice else "agree"
        return ans_num, status


    m = _BETTER_FIELD.search(raw)
    if m:
        ans_num = int(m.group(1))
        status  = "reject" if ans_num - 1 == rejected_response_choice else "agree"
        return ans_num, status


    m = _ANS_PATTERN.search(raw)
    if m:
        ans_num = int(m.group(1))
        status  = "reject" if ans_num - 1 == rejected_response_choice else "agree"
        return ans_num, status


    logging.warning(f"[parse_flag] fail: {raw[:120]}...")
    return -1, "doesntMatch"

# ---------------------------------------------------------------------------
# Per-row processing (SYNC version)
# ---------------------------------------------------------------------------
def process_item(row: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single data row and return the result record"""
    try:
        row_id = row.get("id", "unknown_id")
        random_number = random.choice([0, 1])  

        # Construct user prompt
        question = prompt(row, random_number).replace("<image>\n", "")


        img_bytes = None
        if "image" in row:
            try:
                img_bytes = _extract_image_bytes(row["image"])
            except Exception as e:
                logging.warning(f"Failed to extract image (id={row_id}): {str(e)}")


        response_text = call_local_model(question, img_bytes)

        try:
            ranking = list(row["human_ranking"])

            bad_idx = int(np.argmax(ranking))   
            good_idx = 1 - bad_idx
        except Exception as e:
            logging.warning(f"Failed to parse human_ranking (id={row_id}): {str(e)}")
            ranking, bad_idx, good_idx = [], -1, -1


        if response_text.startswith("[LOCAL_ERROR]"):
            flag_num, status = -1, "doesntMatch"
        else:

            flag_num, status = _parse_flag(response_text, bad_idx)

        if flag_num in (1, 2):
            cand_idx = flag_num - 1
            if random_number == 1:
                cand_idx = 1 - cand_idx
        else:
            cand_idx = -1


        if cand_idx == -1 or status == "doesntMatch":
            fstatus = "doesntMatch"
        elif cand_idx == bad_idx:
            fstatus = "reject"
        elif cand_idx == good_idx:
            fstatus = "agree"
        else:
            fstatus = "doesntMatch"


        meta = {
            "filtering model": "Qwen2.5-Omni-LoRA",
            "filter_choice": response_text,
            "filter_prompt": question,
            "filter_number": flag_num,
            "random_number": random_number,
            "flag_status": fstatus,
        }


        result = {
            "id": row_id,
            "query": row.get("query", ""),
            "response": row.get("response", ""),
            "ranking": json_serial(ranking),
            "meta": meta,
        }

        for field in ["models", "judge", "rationale", "query_source"]:
            if field in row:
                result[field] = json_serial(row[field])

        return result

    except Exception as e:
        logging.error(f"Failed to process entry {row.get('id', 'unknown')} 失败: {str(e)}")
        return {
            "id": row.get("id", "error_id"),
            "query": f"[ERROR] {str(e)}",
            "response": "",
            "ranking": [],
            "meta": {
                "filtering model": "Qwen2.5-Omni-LoRA",
                "filter_choice": f"PROCESSING_ERROR: {str(e)}",
                "filter_prompt": "",
                "filter_number": -1,
                "random_number": -1,
                "flag_status": "error"
            }
        }


def run_sync(
    df: pd.DataFrame,
    output_path: str,
    k: int = 1,
    resume: bool = False,
):
    """Run synchronous inference and save results"""

    done_ids = set()
    if resume and os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as rf:
                for line in rf:
                    try:
                        rec = json.loads(line)
                        done_ids.add(str(rec["id"]))
                    except Exception:
                        pass
            logging.info(f"⏩ Resume from checkpoint: skipped {len(done_ids)}")
        except Exception as e:
            logging.error(f"Failed to read existing results: {str(e)}")


    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    counters = {
        "total": 0,
        "processed": 0,
        "agree": 0,
        "reject": 0,
        "doesntMatch": 0,
        "errors": 0
    }
    
    total_items = len(df) * k
    pbar = tqdm(total=total_items, desc="Inference progress", unit="samples")
    

    with open(output_path, "a", encoding="utf-8") as fout:

        for idx, row in df.iterrows():
            row_id = str(row.get("id", f"row_{idx}"))
            

            if resume and row_id in done_ids:
                pbar.update(k)  
                counters["processed"] += k
                continue
                
            row_dict = row.to_dict()
            

            for i in range(k):
                try:
                    
                    result = process_item(row_dict)
                    counters["total"] += 1
                
                    safe_result = json.loads(json.dumps(result, default=json_serial))
                    
                    
                    fout.write(json.dumps(safe_result, ensure_ascii=False) + "\n")
                    fout.flush()  

                    status = result["meta"]["flag_status"]
                    if status == "agree":
                        counters["agree"] += 1
                    elif status == "reject":
                        counters["reject"] += 1
                    elif status == "doesntMatch":
                        counters["doesntMatch"] += 1
                    else:  
                        counters["errors"] += 1
                    
                    counters["processed"] += 1
                    
                except Exception as e:
                    counters["errors"] += 1
                    logging.error(f"Processing {row_id} Error during processing: {str(e)}")
                    
                pbar.update(1)
                pbar.set_postfix({
                    "Agree": counters["agree"], 
                    "Reject": counters["reject"],
                    "Does not match": counters["doesntMatch"],
                    "Errors": counters["errors"]
                })
                

            if counters["processed"] % 1 == 0:
                fout.flush()
    
    pbar.close()
    

    logging.info("\n✅ Inference complete!")
    logging.info(f"Results saved to: {output_path}")
    logging.info(f"  Processing: {counters['total']}")
    logging.info(f"  Agree (agree): {counters['agree']}")
    logging.info(f"  Reject (reject): {counters['reject']}")
    logging.info(f"  Does not match (doesntMatch): {counters['doesntMatch']}")
    logging.info(f"  Errors: {counters['errors']}")
    logging.info(f"  Processing samples: {counters['processed']}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="use Qwen2.5-Omni modl Processing VL-RewardBench")
    p.add_argument(
        "--data_path",
        type=str,
        default="/path/to/resource",
        help="Path to Parquet dataset"
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="/path/to/resource",
        help="Output directory (default ./results)"
    )
    p.add_argument(
        "--k",
        type=int,
        default=1,
        help="Number of repetitions per record (default 1)"
    )
    p.add_argument(
        "--max_rows",
        type=int,
        default=None
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last results"
    )
    return p.parse_args()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    np.set_printoptions(threshold=10, suppress=True)
    
    args = parse_args()
    
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file does not exist: {args.data_path}")
    
    initialize_model()
    
    # Loading data
    logging.info(f"📂 Loading data: {args.data_path}")
    df = pd.read_parquet(args.data_path)
    
    logging.info(f"Loaded, total {len(df)}")
    logging.info(f"Columns: {list(df.columns)}")
    logging.info(f"First row data: {json.dumps(df.iloc[0].to_dict(), default=json_serial, indent=2)}")
    
    if args.max_rows is not None:
        df = df.head(args.max_rows)
        logging.info(f"Using first {args.max_rows} rows for testing")
    
    if "meta" in df.columns:
        df = df.drop("meta", axis=1)
        logging.info("Removed 'meta' column")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(
        args.output_dir, 
        f"qwen-omni-results-{timestamp}.jsonl"
    )
    
    logging.info(f"🔄 Start inference（{len(df)}）...")
    run_sync(
        df=df,
        output_path=output_path,
        k=args.k,
        resume=args.resume
    )