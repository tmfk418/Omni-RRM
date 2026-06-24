#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Change JSON/JSONL file paths by replacing the prefix.
-----------------------------------------------------

This script traverses all string fields in a JSON/JSONL file 
(including audios[], messages.content, solution, etc.) and replaces 
their file path prefixes with a user-specified prefix.

Path replacement rule:
- Original: /data/audio/Sundsvall_Harbour.wav
- Filename: Sundsvall_Harbour.wav
- New prefix: /space/audio_files/
- Result: /space/audio_files/Sundsvall_Harbour.wav
"""

import os
import re
import json
import argparse

def change_path_in_text(text: str, new_prefix: str) -> str:
    """Replace file paths in a string with a new prefix."""
    def replacer(match):
        raw_path = match.group(0)
        filename = os.path.basename(raw_path)
        return os.path.join(new_prefix.rstrip("/"), filename).replace("\\", "/")

    # Match strings starting with / until a common file extension
    return re.sub(
        r"/[^\"'\s]+?\.(wav|mp3|mp4|avi|mov|flac|jpg|png|json|txt)",
        replacer,
        text
    )

def recursive_change(obj, new_prefix: str):
    """Recursively replace paths in dicts, lists, and strings."""
    if isinstance(obj, dict):
        return {k: recursive_change(v, new_prefix) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [recursive_change(v, new_prefix) for v in obj]
    elif isinstance(obj, str):
        return change_path_in_text(obj, new_prefix)
    else:
        return obj

def process_file(input_file: str, output_file: str, new_prefix: str):
    """Process JSON or JSONL file and replace detected file paths."""
    with open(input_file, "r", encoding="utf-8") as fin:
        content = fin.read().strip()

    results = []
    try:
        # Try to parse as a single JSON object/array
        data = json.loads(content)
        if isinstance(data, list):
            results = [recursive_change(obj, new_prefix) for obj in data]
        else:
            results = [recursive_change(data, new_prefix)]
    except json.JSONDecodeError:
        # Fallback: treat as JSONL (one JSON object per line)
        results = []
        with open(input_file, "r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    results.append(recursive_change(obj, new_prefix))
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line: {line[:80]}...")

    with open(output_file, "w", encoding="utf-8") as fout:
        for obj in results:
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Change all file paths in JSON/JSONL files by replacing prefix.")
    parser.add_argument("--input", type=str, required=True, help="Input JSON/JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Output JSON/JSONL file")
    parser.add_argument("--prefix", type=str, required=True, help="New prefix for file paths (e.g., /space/audio_files/)")
    args = parser.parse_args()

    process_file(args.input, args.output, args.prefix)
    print(f"✅ Changed paths saved to {args.output}")
