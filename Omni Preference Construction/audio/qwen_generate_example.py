#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qwen‑2.5‑Omni‑7B Audio → Text QA Script
(Resume from checkpoints + fuzzy matching + automatic skip of invalid audio)
"""

import os
import json
import torch
import soundfile as sf
import pandas as pd
from tqdm import tqdm
from transformers import (
    Qwen2_5OmniForConditionalGeneration as OmniModel,
    Qwen2_5OmniProcessor as OmniProcessor,
)

# === Path Configuration ===
MODEL_DIR = "/path/to/Qwen2.5-Omni-7B"
AUDIO_DIR = "/path/to/Clotho-AQA/audio_files"
CSV_PATH = "/path/to/Clotho-AQA/clotho_aqa_train_cleaned.csv"
OUT_JSON = "./qwen2.5-omni-7B_audio_text_output.json"
MAX_NEW = 48
DEBUG = True  # If True, only process the first two samples

# === Load Completed Records ===
if os.path.exists(OUT_JSON):
    with open(OUT_JSON, "r", encoding="utf-8") as f:
        results = json.load(f)
    completed_keys = set((r["file_name"], r["question"]) for r in results)
else:
    results = []
    completed_keys = set()

# === Fuzzy Matching Check ===
def is_completed(new_file, new_question):
    """Check if a (file, question) pair is already processed."""
    for file, question in completed_keys:
        if question == new_question and file.startswith(new_file[:6]):
            return True
    return False

# === Load Model ===
print("🚀 Loading Qwen‑2.5‑Omni‑7B …")
model = OmniModel.from_pretrained(
    MODEL_DIR,
    device_map={"": 0},  # change GPU ID as needed
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
model.disable_talker()
model.eval()
processor = OmniProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)

# === Audio Loading Function ===
def load_wav(path: str, target_sr: int = 16000):
    """Load audio file, convert to mono, and resample if needed."""
    wav, sr = sf.read(path, always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(-1)
    if sr != target_sr:
        import torchaudio
        wav = torchaudio.functional.resample(
            torch.from_numpy(wav), sr, target_sr
        ).numpy()
    return wav.astype('float32')

# === Load Dataset and Run Inference ===
df = pd.read_csv(CSV_PATH)
total = 2 if DEBUG else len(df)

for idx, row in tqdm(df.iterrows(), total=total):
    if DEBUG and idx >= 2:
        break

    file_name = row['file_name']
    question = row['QuestionText']

    if is_completed(file_name, question):
        continue

    wav_path = os.path.join(AUDIO_DIR, file_name)
    try:
        wav_in = load_wav(wav_path)
    except Exception as e:
        print(f"[ERROR] Cannot load {file_name}: {e}")
        continue

    conversation = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant for audio QA."}]},
        {"role": "user", "content": [{"type": "audio", "audio": wav_in}, {"type": "text", "text": question}]}
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(
        text=prompt,
        audio=[wav_in], images=None, videos=None,
        return_tensors="pt", padding=True,
    ).to(model.device).to(model.dtype)

    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=MAX_NEW, return_audio=False)

    seqs = out_ids.tolist() if isinstance(out_ids, torch.Tensor) else out_ids
    texts = [processor.tokenizer.decode(ids, skip_special_tokens=True) for ids in seqs]

    if texts:
        split_text = texts[0].rsplit("assistant\n", 1)
        answer = split_text[1].strip() if len(split_text) > 1 else texts[0].strip()
    else:
        answer = ""

    result = {
        "file_name": file_name,
        "question": question,
        "predicted_answer": answer
    }
    results.append(result)
    completed_keys.add((file_name, question))

    # === Incremental Save ===
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n✅ Done! Output written to: {OUT_JSON}")
