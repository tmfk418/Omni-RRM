#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Replace all file paths in JSON/JSONL with a user-specified prefix.
-----------------------------------------------------------------
- Recursively process all fields in JSON (including "audio", "audios", 
  "messages", "Context", "solution", etc.).
- Extract only the filename from the original path and prepend the 
  user-specified prefix.
- Invalid characters in filenames (e.g., spaces, commas, parentheses) 
  are replaced with underscores.
- User provides the new prefix via --prefix.
"""

import os
import re
import json
import argparse

def replace_with_prefix(text: str, new_prefix: str) -> str:
    """Replace detected file paths in a string with the given prefix."""
    def replacer(match):
        raw_path = match.group(0)
        filename = os.path.basename(raw_path)
        filename = re.sub(r"[ ,()]", "_", filename)  # replace illegal characters
        return os.path.join(new_prefix.rstrip("/"), filename).replace("\\", "/")

    return re.sub(
        r"/[^\"']+?\.(wav|mp3|mp4|avi|mov|flac|jpg|png|json|txt)",
        replacer,
        text
    )

def recursive_replace(obj, new_prefix: str):
    """Recursively replace paths inside dicts, lists, and strings."""
    if isinstance(obj, dict):
        return {k: recursive_replace(v, new_prefix) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [recursive_replace(v, new_prefix) for v in obj]
    elif isinstance(obj, str):
        return replace_with_prefix(obj, new_prefix)
    else:
        return obj

def process_file(input_file: str, output_file: str, new_prefix: str):
    """Process a JSON/JSONL file and save results with updated paths."""
    results = []
    with open(input_file, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                changed = recursive_replace(obj, new_prefix)
                results.append(changed)
            except json.JSONDecodeError:
                print(f"⚠️ Skipping invalid JSON line: {line[:80]}...")

    with open(output_file, "w", encoding="utf-8") as fout:
        for obj in results:
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"✅ Replaced paths saved to {output_file}")
    print(f"Processed {len(results)} valid records.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replace JSON/JSONL file paths with a custom prefix.")
    parser.add_argument("--input", type=str, required=True, help="Input JSON/JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Output JSON/JSONL file")
    parser.add_argument("--prefix", type=str, required=True, help="New prefix for all file paths (e.g., /space/audio/)")
    args = parser.parse_args()

    process_file(args.input, args.output, args.prefix)
