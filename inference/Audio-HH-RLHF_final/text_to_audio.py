import os
import json
import time
import base64
import wave
from openai import OpenAI

# Initialize API
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),  # Recommended: use environment variable
    base_url="xxxxx" # 
)

def generate_wav_audio(text_prompt, output_wav_path):
    """Generate WAV audio file from a text prompt using the Qwen2.5-Omni model."""
    try:
        completion = client.chat.completions.create(
            model="qwen2.5-omni-7b",
            messages=[{
                "role": "user",
                "content": f'Read the following sentence aloud exactly as written, and say nothing else:\n"{text_prompt}"'
            }],
            modalities=["text", "audio"],
            audio={"voice": "Chelsie", "format": "wav"},
            stream=True,
            stream_options={"include_usage": True}
        )

        # Collect base64-encoded audio chunks
        audio_base64 = []
        for chunk in completion:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if hasattr(delta, "audio") and delta.audio:
                    b64 = delta.audio.get("data", "")
                    audio_base64.append(b64)

        if not audio_base64:
            return False

        # Decode and save as a standard WAV file
        pcm_data = base64.b64decode("".join(audio_base64))
        with wave.open(output_wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(pcm_data)

        return True
    except Exception as e:
        print(f"❌ Audio generation failed: {e}")
        return False

def process_jsonl(input_path, output_dir, output_jsonl, max_samples=2000):
    """Read a JSONL file, generate audio for text prompts, and save updated JSONL."""
    os.makedirs(output_dir, exist_ok=True)

    # Load already processed data (for resume support)
    processed_set = set()
    if os.path.exists(output_jsonl):
        with open(output_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    audio_path = item.get("audio_prompt")
                    if audio_path:
                        processed_set.add(audio_path)
                except:
                    continue

    processed = len(processed_set)

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_jsonl, "a", encoding="utf-8") as fout:
        for idx, line in enumerate(fin):
            item = json.loads(line)
            text_prompt = item.get("text_prompt", "").strip()
            answer_a = item.get("answer_a", "").strip()
            answer_b = item.get("answer_b", "").strip()
            label = item.get("label", 0)

            if not text_prompt or not answer_a or not answer_b:
                continue

            audio_filename = f"sample_{idx:04d}.wav"
            audio_path = os.path.join(output_dir, audio_filename)

            if audio_path in processed_set or os.path.exists(audio_path):
                print(f"⏩ Skipping already processed: {audio_filename}")
                continue

            success = generate_wav_audio(text_prompt, audio_path)
            if not success:
                print(f"⚠️ Skipping invalid audio: {text_prompt[:30]}...")
                continue

            output_item = {
                "audio_prompt": audio_path,
                "answer_a": answer_a,
                "answer_b": answer_b,
                "label": label
            }
            fout.write(json.dumps(output_item, ensure_ascii=False) + "\n")
            fout.flush()

            processed += 1
            print(f"✅ [{processed}] Saved: {audio_path}")
            if processed >= max_samples:
                break

            time.sleep(0.5)
            
    print(f"\n🎉 Total processed samples (including skipped): {processed}")

# 🚀 Entry point
if __name__ == "__main__":
    process_jsonl(
        input_path="test_filtered.jsonl",       # Replace with your input JSONL file
        output_dir="audio_output",              # Directory to save generated audio
        output_jsonl="final_data_with_audio.jsonl",  # Output JSONL file
        max_samples=2000
    )
