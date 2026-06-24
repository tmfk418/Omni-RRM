#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Video Question Answering with Qwen2.5-VL (Hugging Face)
-------------------------------------------------------
* Loads a Qwen2.5-VL model (Vision-Language).
* Extracts frames from videos.
* Runs inference with a custom system + user prompt.
* Saves results incrementally to JSON.
"""

import os
import json
import torch
import cv2
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq

# === Path Configuration ===
model_path = "/path/to/Qwen2.5-VL-3B-Instruct"   # Update with your model checkpoint path
video_dir = "/path/to/video/dataset"             # Root folder for video files
input_json = "/path/to/input.json"               # Input dataset in JSON format
output_json = "/path/to/output.json"             # Output results JSON

# === Load Model and Processor ===
print("🚀 Loading Qwen2.5-VL Hugging Face model with generate interface...")
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    trust_remote_code=True
).cuda().eval()

# === Video Frame Loader ===
def load_video_frames(video_path, max_frames=16):
    """Extract frames from a video file for model input."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // max_frames)
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
        idx += 1
    cap.release()
    return frames

# === Answer Extraction Helper ===
def extract_assistant_answer(text):
    """Extract the assistant's answer from the generated text."""
    if "<|im_start|>assistant" in text:
        return text.split("<|im_start|>assistant\n")[-1].strip()
    elif "assistant\n" in text:
        return text.split("assistant\n")[-1].strip()
    else:
        return text.strip()

# === Inference Function ===
def infer(video_path, question):
    """Run inference on a single video-question pair."""
    try:
        frames = load_video_frames(video_path)
        if not frames:
            return "[ERROR] No frames extracted"

        prompt = (
            "<|im_start|>system\nYou are an intelligent assistant for video question answering.<|im_end|>\n"
            f"<|im_start|>user\nPlease watch the video and answer the following question: <|VIDEO|> {question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        inputs = processor(text=prompt, videos=frames, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256)

        decoded = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        return extract_assistant_answer(decoded)
    except Exception as e:
        return f"[ERROR] {e}"

# === Load Input Data ===
with open(input_json, 'r', encoding='utf-8') as f:
    data = json.load(f)

results = []

print(f"📹 Running inference on {len(data)} items...")
for item in tqdm(data, desc="Inference"):
    video_id = item.get("video_id")
    video_path_rel = item.get("video_path")
    question = item.get("question")

    # Ensure video_path is correctly constructed (avoid duplicate prefixes)
    if "academic_source/" in video_path_rel:
        video_path_rel = video_path_rel.split("academic_source/", 1)[-1]
    video_path = os.path.join(video_dir, video_path_rel)

    print(f"=== Processing video: {os.path.basename(video_path)} ===")
    if not os.path.exists(video_path):
        print(f"[Skip] Not found: {video_path}")
        continue

    pred = infer(video_path, question)
    results.append({
        "video_id": video_id,
        "video_path": video_path,
        "question": question,
        "predicted_answer": pred
    })

    # Save results incrementally
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

print(f"✅ Done! Output saved to: {output_json}")
