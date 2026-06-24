import os
import json
import time
import base64
import re
from tqdm import tqdm
from multiprocessing import Pool
from mimetypes import guess_type
from argparse import ArgumentParser

# ✅ Environment configuration (use GPU 1, prevent OOM)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['MAX_PIXELS'] = '1003520'
os.environ['VIDEO_MAX_PIXELS'] = '50176'
os.environ['FPS_MAX_FRAMES'] = '20'

# ✅ Swift local model loading
from swift.llm import PtEngine, RequestConfig, get_model_tokenizer, get_template, InferRequest
from swift.tuners import Swift

# ✅ Global variables
engine = None
request_config = None
tokenizer = None
template = None
global_image_folder = None


# --------------------------------------------------
# ★★★ 1. Robust parsing of "better" field ★★★
# --------------------------------------------------
def extract_better(judgment: str):
    """
    Attempt to extract the final decision from model output:
      Returns "A" / "B" / "equal" / None
    """
    if not judgment or judgment in ["None", ""]:
        return None

    # 1) Try strict JSON parsing
    try:
        data = json.loads(judgment)
        # Prefer final_verdict
        if "final_verdict" in data:
            fv = str(data["final_verdict"])
            if "[[A]]" in fv:
                return "A"
            if "[[B]]" in fv:
                return "B"
            if "equal" in fv.lower():
                return "equal"
        # Fallback to "better" field
        if "better" in data:
            b = str(data["better"]).strip().upper()
            if b in ["A", "B"]:
                return b
            if b == "EQUAL":
                return "equal"
    except Exception:
        pass  # Not valid JSON, fallback to regex

    # 2) Regex fallback
    m = re.search(r"\[\[\s*(A|B)\s*\]\]", judgment, flags=re.I)
    if m:
        return m.group(1).upper()

    m = re.search(r'"better"\s*:\s*"?(A|B|equal)"?', judgment, flags=re.I)
    if m:
        val = m.group(1).upper()
        return val if val in ["A", "B"] else "equal"

    return None
# --------------------------------------------------


def init(api_base, api_key, model_name, image_folder):
    """Initialize local model, tokenizer, and template (ignore API params)"""
    global engine, request_config, tokenizer, template, global_image_folder

    base_model_path = "/path/to/base_model"
    lora_checkpoint_path = "/path/to/lora_checkpoint"

    model, tokenizer = get_model_tokenizer(base_model_path)
    model = Swift.from_pretrained(model, lora_checkpoint_path)
    template = get_template(model.model_meta.template, tokenizer)
    engine = PtEngine.from_model_template(model, template, max_batch_size=1)
    request_config = RequestConfig(max_tokens=4096, temperature=0.7, top_p=0.9)

    global_image_folder = image_folder


def encode_image(image_path):
    """Base64 encode image (for GPT API; not needed for local models)"""
    mime_type, _ = guess_type(image_path)
    if not mime_type:
        mime_type = "application/octet-stream"
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{data}"


def prompt_string(data_obj, random_number):
    # Randomize answer order based on random_number
    answers = (
        [data_obj["Output1"], data_obj["Output2"]]
        if random_number == 0
        else [data_obj["Output2"], data_obj["Output1"]]
    )
    order = "A-B" if random_number == 0 else "B-A"

    return f"""You are a helpful and thoughtful AI assistant with expertise in multimodal reasoning.

Please analyze the following image and question, then determine which of the two provided answers is better based on five evaluation dimensions:

1. Fluency and Coherence
2. Relevance to the Question and Image
3. Accuracy and Completeness
4. Reasoning Quality
5. Safety and Ethical Alignment

Before making your judgment:
- First, imagine the most ideal and factually accurate answer (a reference answer) based on the image and the question.
- Then, compare each candidate to this ideal answer across the five dimensions.
- Provide integer scores (0–10) for both answers.
- Write a clear reasoning summary covering all five dimensions.
- Finally, decide which answer is better.

⚠️ IMPORTANT RULES:
- You MUST carefully compare both answers. 
- Do NOT always choose the same letter.  
- The field "better" MUST be either "A", "B", or "equal".  
- The field "final_verdict" MUST match the "better" field.  
- If both answers are equally good, use "equal".  
- If they differ, select the better one.

### Required Output Format (STRICT JSON)
{{
  "score_1": 0–10,
  "score_2": 0–10,
  "better": "A" or "B" or "equal",
  "reasoning": "<think>[detailed analysis across five dimensions]</think>",
  "final_verdict": "<answer>[[A]]</answer>"  // or [[B]] / [[equal]]
}}

⚠️ OUTPUT MUST BE VALID JSON ONLY  
⚠️ No extra text, no markdown, no explanation outside JSON  

### Evaluation Context
Order of answers in this task: {order}  
Question: {data_obj["Text"]}

Answer A: {answers[0]}

Answer B: {answers[1]}
"""


def call_api(example, image_root, max_try=10):
    """Call local Qwen model for image + prompt inference"""
    image_path = os.path.join(image_root, example["Image"])
    content_text = example["Text"]

    # ✅ Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}")
        return "None"

    print(f"✅ Successfully loaded image: {image_path}")

    try:
        infer_request = InferRequest(
            messages=[{"role": "user", "content": content_text}],
            images=[image_path]
        )

        print(f"📨 Running inference, sample ID: {example['ID']}...")

        response = engine.infer([infer_request], request_config)[0]
        output_text = response.choices[0].message.content.strip()

        if output_text in ["", "None", None]:
            print(f"⚠️ Empty inference output! ID: {example['ID']}")
        else:
            print(f"✅ Inference success! ID: {example['ID']}, preview: {output_text[:10]}...")

        return output_text

    except Exception as e:
        print(f"❌ Inference failed: {e}")
        return "None"


def process_example(example):
    """Process a single example (supports multiprocessing)"""
    return call_api(example, image_root=global_image_folder)


def main(args):
    with open(args.question_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    output_path = args.answers_file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    seen_ids = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    seen_ids.add(item["ID"])
                except:
                    continue

    new_examples = []
    for ex in data:
        if ex["ID"] in seen_ids:
            continue
        ex["Text"] = prompt_string(ex, random_number=0)
        new_examples.append(ex)

    with Pool(args.num_workers, initializer=init,
              initargs=(args.api_base, args.api_key, args.model, args.image_folder)) as p:
        with open(output_path, "a", encoding="utf-8") as f:
            for i, output in enumerate(tqdm(p.imap(process_example, new_examples), total=len(new_examples))):
                ex = new_examples[i]

                # >>> Extract "better" field
                pred_better = extract_better(output)

                f.write(json.dumps({
                    "ID": ex["ID"],
                    "Text": ex["Text"],
                    "Image": ex["Image"],
                    "output": output,
                    "PredBetter": pred_better,
                    "Label": "A" if ex.get("Better") == "Output1" else "B" if ex.get("Better") == "Output2" else None,
                    "Meta": {
                        "Output1": ex["Output1"],
                        "Output2": ex["Output2"],
                        "Category": ex.get("Category", "unknown")
                    }
                }, ensure_ascii=False) + "\n")
                if i % 5 == 1:
                    f.flush()


def build_arg_parser():
    p = ArgumentParser()
    p.add_argument("--image-folder", type=str, default="/path/to/images")
    p.add_argument("--question-file", type=str, default="/path/to/questions.json")
    p.add_argument("--answers-file", type=str, default="/path/to/outputs.jsonl")
    p.add_argument("--model", type=str, default="qwen-local")
    p.add_argument("--api-base", type=str, default="http://dummy")
    p.add_argument("--api-key", type=str, default="dummy")
    p.add_argument("--num-workers", type=int, default=1)
    return p


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

    args = build_arg_parser().parse_args()
    main(args)
