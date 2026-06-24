import os
import json
from tqdm import tqdm
import torch
from datasets import load_from_disk
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# ===== Configuration =====
DATA_DIR = "/path/to/dataset/rlaif-v-dataset/train"
MAX_SAMPLES = 10_020
OUTPUT_PATH = "/path/to/save/results/qwen2.5-7B_image/final_matched_7b.json"
MODEL_NAME = "/path/to/model/Qwen2.5-VL-7B-Instruct"
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# ===== Load dataset and deduplicate =====
raw_ds = load_from_disk(DATA_DIR)
print(f"Original number of samples: {len(raw_ds):,}")

seen_keys = set()
selected_indices = []
for idx, (img_path, ques) in enumerate(zip(raw_ds["image_path"], raw_ds["question"])):
    key = (img_path, ques)
    if key in seen_keys:
        continue
    seen_keys.add(key)
    selected_indices.append(idx)
    if len(selected_indices) >= MAX_SAMPLES:
        break

sub_ds = raw_ds.select(selected_indices)
print(f"Final number of samples for inference: {len(sub_ds):,}")

# ===== Load existing results (resume support) =====
if os.path.exists(OUTPUT_PATH):
    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        results = json.load(f)
    done_keys = set((item.get("image_path"), item.get("question")) for item in results)
    print(f"🔁 Detected existing results, skipping {len(done_keys)} samples")
else:
    results = []
    done_keys = set()

# ===== Load model and processor =====
print("Loading Qwen2.5-VL Hugging Face model...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
).eval()

processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

# ===== Build Qwen-formatted input =====
def build_qwen_inputs(image_pil, question_text):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_pil},
                {"type": "text", "text": question_text}
            ]
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[image_pil],
        padding=True,
        return_tensors="pt"
    ).to("cuda")
    return inputs

# ===== Inference loop =====
for i in tqdm(range(len(sub_ds)), desc="Inference"):
    sample = sub_ds[i]
    image = sample["image"]
    question = sample["question"]
    image_path = sample["image_path"]
    key = (image_path, question)

    if key in done_keys:
        continue

    try:
        inputs = build_qwen_inputs(image, question)
        with torch.no_grad():
            gen_ids = model.generate(**inputs, max_new_tokens=128, min_new_tokens=5)
            gen_trimmed = gen_ids[:, inputs.input_ids.shape[1]:]
            answer = processor.batch_decode(gen_trimmed, skip_special_tokens=True)[0].strip()

        if not answer:
            answer = "[NO OUTPUT]"

        results.append({
            "idx": len(results),
            "image_path": image_path,
            "question": question,
            "predicted_answer": answer,
            "chosen": sample.get("chosen", ""),
        })

    except Exception as e:
        results.append({
            "idx": len(results),
            "error": str(e),
            "image_path": image_path
        })
        print(f"[⚠️ Error @ index {len(results)}]: {e}")

    if len(results) % 10 == 0 or i == len(sub_ds) - 1:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved {len(results)} results so far → {OUTPUT_PATH}")

print(f"✅ Inference finished. Final results saved to: {OUTPUT_PATH}")
