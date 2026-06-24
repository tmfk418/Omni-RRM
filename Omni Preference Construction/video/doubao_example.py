#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Video Preference Evaluation with Doubao API
--------------------------------------------
* Loads a dataset of pre-evaluated video QA samples.
* Extracts frames from each video.
* Sends frames + evaluation prompt to Doubao for scoring.
* Includes caching, retries, robust JSON parsing, and resumable saving.
* Saves both detailed results and statistical analysis.
"""

import os
import json
import time
import shutil
import subprocess
import textwrap
import re
from tqdm import tqdm
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

# ========== Model Initialization ==========
from model import Doubao
model = Doubao(api_key="YOUR_API_KEY_HERE")  # Replace with your Doubao API key

# ========== Path Configuration ==========
video_root = "/path/to/videos"                          # Change to your video dataset path
frame_tmp_root = "./tmp_frames"
os.makedirs(frame_tmp_root, exist_ok=True)

# Output files
output_file = "doubao_output.json"
output_analysis = "doubao_output_analysis.txt"

# Performance parameters
MAX_WORKERS = 8                # Number of concurrent threads
FRAME_CACHE_SIZE = 30          # Frame cache size
MIN_FRAME_QUALITY = 3          # FFmpeg quality (1=best, 31=worst)
THUMBNAIL_SIZE = (512, 512)    # Thumbnail dimensions
API_TIMEOUT = 30               # API timeout (seconds)
MAX_RETRIES = 2                # Max retries for each operation

# ========== Load Dataset ==========
print("⏳ Loading pre-evaluated input samples...")

with open("/path/to/filtered_output.json", "r", encoding="utf-8") as f:
    records = json.load(f)

filtered_keys = [(r["video_path"], r["question"]) for r in records]
index_records = {(r["video_path"], r["question"]): r for r in records}

# ========== Core Functions ==========

def extract_frames(video_path, out_dir, fps=1, max_frames=35):
    """Extract frames from a video using ffmpeg."""
    os.makedirs(out_dir, exist_ok=True)
    video_path = video_path.replace("\\", "/")
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", f"fps={fps}",
        "-q:v", str(MIN_FRAME_QUALITY),
        os.path.join(out_dir, "frame_%04d.jpg")
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        frames = sorted([os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith(".jpg")])
        return frames[:max_frames] if max_frames else frames
    except Exception as e:
        print(f"❌ Frame extraction failed for {video_path}: {e}")
        return None

@lru_cache(maxsize=FRAME_CACHE_SIZE)
def load_frame_cached(frame_path):
    """Load and resize a frame, with caching."""
    try:
        img = Image.open(frame_path).convert("RGB")
        img.thumbnail(THUMBNAIL_SIZE)
        return img
    except Exception as e:
        print(f"⚠️ Corrupted frame: {frame_path} - {e}")
        return None

def build_prompt(video_path, question, answer_a, answer_b):
    """Construct evaluation prompt for Doubao."""
    return textwrap.dedent(f"""
    You are a helpful and thoughtful AI assistant with experience in multimodal reasoning.
    ### Task
    Two candidate answers (Model A & Model B) are provided for a question related to a video.
    Your task is to analyze and give a comparative evaluation of their quality and accuracy based on FIVE key dimensions.

    **Evaluation Dimensions**
    1. Fluency and Coherence
    2. Relevance to the Question and Video
    3. Accuracy and Completeness
    4. Reasoning Quality
    5. Safety and Ethical Alignment

    **Scoring Guidelines**
    - 9-10: Excellent in all dimensions
    - 6-8: Good overall with minor issues
    - 3-5: Deficient in multiple dimensions
    - 0-2: Poor in most dimensions

    **Evaluation Process**
    1. Imagine the most ideal and factually accurate answer.
    2. Evaluate both answers across all five dimensions.
    3. Assign each model a score from 0 to 10.
    4. Decide which model performed better overall ("A", "B", or "equal").
    5. Provide detailed reasoning.

    **Output Instructions**
    - Strictly valid JSON only.
    - Do NOT include markdown, code fences, or placeholders.
    - Use double quotes for all field names and string values.
    - Reasoning must be enclosed in "<think>...</think>".
    - The final verdict must be: "<answer>[[A]]</answer>", "<answer>[[B]]</answer>", or "<answer>[[equal]]</answer>".

    ### Required Output Keys
    {{
      "score_A": [0-10],
      "score_B": [0-10],
      "better": "A" or "B" or "equal",
      "reasoning": "<think>...</think>",
      "final_verdict": "<answer>[[A]]</answer>"
    }}

    ### Context
    Video file: {video_path}
    Question: {question}
    Candidate A: {answer_a}
    Candidate B: {answer_b}
    """).strip()

def extract_json(text):
    """Robust JSON parser with repair and fallback."""
    try:
        text = text.strip().strip("```json").strip("```").strip()
        match = re.search(r"\{[\s\S]+?\}", text)
        if not match:
            raise ValueError("No JSON object found")
        json_str = match.group(0)
        json_str = re.sub(r'(?<!")\b([a-zA-Z_]+)\b(?!")(?=\s*:)', r'"\1"', json_str)
        json_str = re.sub(r",\s*}", "}", json_str)
        json_str = re.sub(r",\s*]", "]", json_str)
        json_str = re.sub(r"[\x00-\x1F\x7F]", "", json_str)
        json_str = json_str.replace("<integer>", "0")
        parsed = json.loads(json_str)
        parsed.setdefault("score_A", None)
        parsed.setdefault("score_B", None)
        parsed.setdefault("better", "UNKNOWN")
        parsed.setdefault("reasoning", "<think>Missing reasoning</think>")
        parsed.setdefault("final_verdict", "<answer>[[equal]]</answer>")
        if "reasoning" in parsed and not parsed["reasoning"].startswith("<think>"):
            parsed["reasoning"] = f"<think>{parsed['reasoning']}</think>"
        return parsed
    except Exception as e:
        return {
            "score_A": None,
            "score_B": None,
            "better": "UNKNOWN",
            "reasoning": f"<think>JSON parse error: {e}</think>",
            "final_verdict": "<answer>[[equal]]</answer>"
        }

def call_doubao_with_frames(image_paths, prompt, max_retry=MAX_RETRIES):
    """Call Doubao API with extracted frames."""
    import base64
    def encode_image_to_base64(image_path):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    image_parts = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image_to_base64(p)}"}}
        for p in image_paths
    ]
    messages = [{"role": "user", "content": image_parts + [{"type": "text", "text": prompt}]}]

    last_error = None
    for attempt in range(max_retry):
        try:
            response = model.client.chat.completions.create(model=model.model_name, messages=messages)
            return extract_json(response.choices[0].message.content)
        except Exception as e:
            last_error = str(e)
            wait = min(5, 1.5 * (attempt + 1))
            print(f"⚠️ Doubao attempt {attempt + 1} failed, retrying in {wait:.1f}s: {last_error}")
            time.sleep(wait)

    return {
        "score_A": None,
        "score_B": None,
        "better": "UNKNOWN",
        "reasoning": f"<think>API call failed after {max_retry} retries: {last_error}</think>",
        "final_verdict": "<answer>[[equal]]</answer>"
    }

def retry_call(fn, args=(), kwargs=None, retries=3, delay=2, backoff=2):
    """Generic retry wrapper."""
    if kwargs is None:
        kwargs = {}
    for i in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if i == retries - 1:
                raise
            wait = delay * (backoff ** i)
            print(f"⚠️ [retry_call] Attempt {i+1} failed: {e}, retrying in {wait:.1f}s...")
            time.sleep(wait)

def safe_process_sample(key):
    return retry_call(process_sample, args=(key,), retries=3, delay=2)

def process_sample(key):
    """Process one video sample with retries."""
    rel_path, question = key
    abs_video_path = os.path.join(video_root, rel_path)
    frame_dir = os.path.join(frame_tmp_root, os.path.basename(rel_path).replace(".mp4", ""))

    # Check video existence
    if not os.path.exists(abs_video_path):
        return {
            "video_path": rel_path,
            "question": question,
            "error": "video_missing",
            "error_detail": f"File not found at {abs_video_path}",
            "score_A": None,
            "score_B": None,
            "better": "UNKNOWN",
            "reasoning": "<think>Video file missing</think>",
            "final_verdict": "<answer>[[equal]]</answer>"
        }

    # Frame extraction
    frames = None
    last_frame_error = None
    for attempt in range(MAX_RETRIES):
        try:
            frames = extract_frames(abs_video_path, frame_dir, fps=1, max_frames=35)
            if frames is not None:
                break
        except Exception as e:
            last_frame_error = str(e)
            wait = min(5, 1.5 * (attempt + 1))
            print(f"⚠️ Frame extraction attempt {attempt+1} failed: {last_frame_error}")
            time.sleep(wait)

    if frames is None:
        return {
            "video_path": rel_path,
            "question": question,
            "error": "frame_extraction_failed",
            "error_detail": f"Failed after {MAX_RETRIES} retries: {last_frame_error}",
            "score_A": None,
            "score_B": None,
            "better": "UNKNOWN",
            "reasoning": "<think>Frame extraction failed</think>",
            "final_verdict": "<answer>[[equal]]</answer>"
        }

    if not frames:
        return {
            "video_path": rel_path,
            "question": question,
            "error": "no_valid_frames",
            "error_detail": "Extraction succeeded but no frames were generated",
            "score_A": None,
            "score_B": None,
            "better": "UNKNOWN",
            "reasoning": "<think>No valid frames extracted</think>",
            "final_verdict": "<answer>[[equal]]</answer>"
        }

    # Build prompt and call Doubao
    prompt = build_prompt(
        rel_path,
        question,
        index_records[key]["qwen2_5_vl_3b_output"],
        index_records[key]["qwen2_5_vl_7b_output"]
    )
    result = call_doubao_with_frames(frames, prompt)

    shutil.rmtree(frame_dir, ignore_errors=True)

    return {
        "video_path": rel_path,
        "question": question,
        "qwen2_5_vl_3b_output": index_records[key]["qwen2_5_vl_3b_output"],
        "qwen2_5_vl_7b_output": index_records[key]["qwen2_5_vl_7b_output"],
        **result,
        "identical": index_records[key]["qwen2_5_vl_3b_output"].strip()
                     == index_records[key]["qwen2_5_vl_7b_output"].strip()
    }

# ========== Main Execution ==========
def main():
    completed = set()
    results_dict = {}

    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                saved = json.load(f)
                completed = {(x["video_path"], x["question"]) for x in saved}
                for r in saved:
                    results_dict[(r["video_path"], r["question"])] = r
                print(f"⏩ Resumed {len(completed)} existing results")
        except Exception as e:
            print(f"⚠️ Failed to load previous results: {e}")

    pending_keys = [k for k in filtered_keys if k not in completed]
    total_count = len(filtered_keys)
    completed_count = len(completed)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(safe_process_sample, k): k for k in pending_keys}
        with tqdm(total=total_count, initial=completed_count, desc="Processing") as pbar:
            write_buffer = []
            for future in as_completed(futures):
                try:
                    result = future.result()
                    key = (result["video_path"], result["question"])
                    results_dict[key] = result
                    write_buffer.append(key)

                    if len(write_buffer) >= 10:
                        ordered_results = [results_dict[k] for k in filtered_keys if k in results_dict]
                        with open(output_file, "w", encoding="utf-8") as f:
                            json.dump(ordered_results, f, ensure_ascii=False, indent=2)
                        write_buffer.clear()
                except Exception as e:
                    print(f"❌ Processing failed: {e}")
                finally:
                    pbar.update(1)

    ordered_results = [results_dict[k] for k in filtered_keys if k in results_dict]
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(ordered_results, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved {len(ordered_results)} results to {output_file}")

    # Statistical Analysis
    def is_valid_record(r):
        return all(k in r for k in ["score_A", "score_B", "better"]) and \
               isinstance(r["score_A"], (int, float)) and \
               isinstance(r["score_B"], (int, float)) and \
               r["better"] in ["A", "B", "equal"]

    valid_results = [r for r in ordered_results if is_valid_record(r)]
    total = len(ordered_results)
    valid = len(valid_results)

    if valid == 0:
        analysis = "⚠️ No valid evaluation results"
    else:
        stats = {
            "total_samples": total,
            "valid_samples": f"{valid} ({valid/total*100:.1f}%)",
            "model_a_wins": sum(1 for r in valid_results if r["better"] == "A"),
            "model_b_wins": sum(1 for r in valid_results if r["better"] == "B"),
            "equal": sum(1 for r in valid_results if r["better"] == "equal"),
            "identical": sum(1 for r in valid_results if r.get("identical")),
            "avg_score_a": sum(r["score_A"] for r in valid_results) / valid,
            "avg_score_b": sum(r["score_B"] for r in valid_results) / valid,
        }

        analysis = textwrap.dedent(f"""
        ===== Statistical Analysis =====
        Total samples: {stats['total_samples']}
        Valid samples: {stats['valid_samples']}
        Model A wins: {stats['model_a_wins']} ({stats['model_a_wins']/valid*100:.1f}%)
        Model B wins: {stats['model_b_wins']} ({stats['model_b_wins']/valid*100:.1f}%)
        Equal preference: {stats['equal']} ({stats['equal']/valid*100:.1f}%)
        Identical outputs: {stats['identical']} ({stats['identical']/valid*100:.1f}%)
        Average score A: {stats['avg_score_a']:.2f}
        Average score B: {stats['avg_score_b']:.2f}
        ================================
        """).strip()

    with open(output_analysis, "w", encoding="utf-8") as f:
        f.write(analysis)
    print(f"✅ Saved analysis to {output_analysis}")

if __name__ == "__main__":
    main()
