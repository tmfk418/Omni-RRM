import os
import json
import time
import textwrap
import base64
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_from_disk
from model import Doubao

# ========== Configuration ==========
DATASET_PATH = "/path/to/dataset/rlaif-v-dataset/train"
INPUT_JSON = "/path/to/json/gemini_comparison_3b_7b.json"
OUTPUT_FILE = "doubao_image_output.json"
OUTPUT_ANALYSIS = "doubao_image_analysis.txt"
MAX_WORKERS = 8
MAX_RETRIES = 3

# Initialize model client (recommend using environment variables for API key)
model = Doubao(api_key=os.getenv("DOUBAO_API_KEY", "YOUR_API_KEY"))

# ========== Load dataset ==========
print("📦 Loading dataset using HuggingFace datasets (metadata only)...")
dataset = load_from_disk(DATASET_PATH)
meta_dataset = dataset.remove_columns(['image'])  # Remove unused columns

print("📥 Loading filtered input from JSON...")
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    input_records = json.load(f)

idx_lookup = {(row["image_path"], row["question"]): i for i, row in enumerate(meta_dataset)}

final_records = []
for item in input_records:
    key = (item["image_path"], item["question"])
    if key in idx_lookup:
        idx = idx_lookup[key]
        record = dataset[idx]  # Direct reference to reduce memory usage
        record["qwen2_5_vl_3b_output"] = item["qwen2_5_vl_3b_output"]
        record["qwen2_5_vl_7b_output"] = item["qwen2_5_vl_7b_output"]
        final_records.append(record)
    else:
        print(f"⚠️ Warning: Not found in dataset: {key}")

filtered_keys = [(r["image_path"], r["question"]) for r in final_records]
index_records = {(r["image_path"], r["question"]): r for r in final_records}

# ========== Prompt Construction ==========
def build_prompt(image_path, question, answer_a, answer_b):
    return textwrap.dedent(f"""
    You are a helpful and thoughtful AI assistant with experience in multimodal reasoning.
    ### Task
    Two candidate answers (Model A & Model B) are provided for a question related to an image.
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
    1. First, imagine the most ideal and factually accurate answer to the question based on the image and question context. 
       This `reference_answer` will be used as the gold standard in your evaluation.
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
      "reasoning": "<think>Analysis covering all five dimensions...</think>",
      "final_verdict": "<answer>[[A]]</answer>"
    }}

    ### Context
    Image file: {image_path}  
    Question: {question}  
    Candidate A: {answer_a}  
    Candidate B: {answer_b}
    """).strip()

# ========== JSON Parsing ==========
def extract_json(text):
    import re
    try:
        text = re.sub(r'[\x00-\x1F]+', ' ', text)  # Remove illegal control characters
        text = text.strip().strip("```json").strip("```").strip()
        match = re.search(r"\{[\s\S]+?\}", text)
        if not match:
            raise ValueError("No JSON object found")
        json_str = match.group(0)
        json_str = re.sub(r'(?<!")\b([a-zA-Z_]+)\b(?!")(?=\s*:)', r'"\1"', json_str)
        json_str = re.sub(r",\s*}", "}", json_str)
        json_str = re.sub(r",\s*]", "]", json_str)
        json_str = json_str.replace("<integer>", "0")
        parsed = json.loads(json_str)
        parsed.setdefault("score_A", None)
        parsed.setdefault("score_B", None)
        parsed.setdefault("better", "UNKNOWN")
        parsed.setdefault("reasoning", "<think>Missing reasoning</think>")
        parsed.setdefault("final_verdict", "<answer>[[equal]]</answer>")
        if "reasoning" in parsed:
            parsed["reasoning"] = parsed["reasoning"].replace("\n", " ").replace("\r", " ")
            if not parsed["reasoning"].startswith("<think>"):
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

# ========== Sample Processing ==========
def process_sample(key):
    if key not in index_records:
        return {
            "image_path": key[0],
            "question": key[1],
            "error": "not_found_in_input",
            "reasoning": "<think>Key not found in input JSON</think>",
            "final_verdict": "<answer>[[equal]]</answer>"
        }

    record = index_records[key]
    image = record["image"]

    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')

        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        prompt = build_prompt(
            key[0],
            key[1],
            record["qwen2_5_vl_3b_output"],
            record["qwen2_5_vl_7b_output"]
        )
        messages = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            {"type": "text", "text": prompt}
        ]}]

        for attempt in range(MAX_RETRIES):
            try:
                response = model.client.chat.completions.create(
                    model=model.model_name,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=2048,
                )
                parsed = extract_json(response.choices[0].message.content)
                return {
                    "image_path": key[0],
                    "question": key[1],
                    "qwen2_5_vl_3b_output": record["qwen2_5_vl_3b_output"],
                    "qwen2_5_vl_7b_output": record["qwen2_5_vl_7b_output"],
                    **parsed,
                    "identical": record["qwen2_5_vl_3b_output"].strip() == record["qwen2_5_vl_7b_output"].strip()
                }
            except Exception as e:
                print(f"⚠️ Retry {attempt + 1} failed: {e}")
                time.sleep(2)

        return {
            "image_path": key[0],
            "question": key[1],
            "qwen2_5_vl_3b_output": record.get("qwen2_5_vl_3b_output"),
            "qwen2_5_vl_7b_output": record.get("qwen2_5_vl_7b_output"),
            "score_A": None,
            "score_B": None,
            "better": "UNKNOWN",
            "reasoning": "<think>All attempts failed</think>",
            "final_verdict": "<answer>[[equal]]</answer>",
            "identical": False
        }
    
    except Exception as e:
        print(f"[ERROR] Error processing {key}: {e}")
        return {
            "image_path": key[0],
            "question": key[1],
            "error": f"Image processing error: {str(e)}",
            "reasoning": "<think>Image processing error occurred</think>",
            "final_verdict": "<answer>[[equal]]</answer>"
        }

# ========== Main Execution ==========
def main():
    completed = set()
    results_dict = {}
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
                completed = {(x["image_path"], x["question"]) for x in saved}
                for r in saved:
                    results_dict[(r["image_path"], r["question"])] = r
                print(f"⏩ Resumed {len(completed)} existing results")
        except:
            pass

    pending = [k for k in filtered_keys if k not in completed]
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_sample, k): k for k in pending}
        with tqdm(total=len(filtered_keys), initial=len(completed), desc="Processing") as pbar:
            write_buffer = []
            for future in as_completed(futures):
                try:
                    result = future.result()
                    key = (result["image_path"], result["question"])
                    results_dict[key] = result
                    write_buffer.append(key)
                    if len(write_buffer) >= 10:
                        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                            json.dump([results_dict[k] for k in filtered_keys if k in results_dict], f, ensure_ascii=False, indent=2)
                        write_buffer.clear()
                finally:
                    pbar.update(1)

    ordered = [results_dict[k] for k in filtered_keys if k in results_dict]
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(ordered, f, ensure_ascii=False, indent=2)

    valid = [r for r in ordered if isinstance(r.get("score_A"), int) and isinstance(r.get("score_B"), int)]
    if valid:
        analysis = f"Total: {len(ordered)}, Valid: {len(valid)}, A wins: {sum(1 for r in valid if r['better']=='A')}, B wins: {sum(1 for r in valid if r['better']=='B')}, Equal: {sum(1 for r in valid if r['better']=='equal')}"
    else:
        analysis = "No valid results."
    with open(OUTPUT_ANALYSIS, "w", encoding="utf-8") as f:
        f.write(analysis)
    print("✅ Done.")

if __name__ == "__main__":
    main()
