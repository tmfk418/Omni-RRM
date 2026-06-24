#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch replace JSON/JSONL file paths.

- Detects any path containing "academic_source".
- Keeps the part from "academic_source" onward.
- Prepends a user-provided new prefix before "academic_source".
- Works on both standalone path fields (e.g., "videos") and embedded text (e.g., in "Context").

Example:
    Original: /data/academic_source/youcook2/221/WRtoMalV4Zo/split_3.mp4
    Command : --new-prefix /space
    Result  : /space/academic_source/youcook2/221/WRtoMalV4Zo/split_3.mp4
"""

import os
import re
import json
import argparse

def replace_in_text(text: str, new_prefix: str) -> str:
    """Replace paths so that 'academic_source/...' keeps its structure,
    but the prefix before it is replaced with new_prefix."""
    if not isinstance(text, str):
        return text

    pattern = r"(/[^\"'\s]*academic_source[^\s\"']*)"

    def replacer(match):
        raw_path = match.group(1)
        idx = raw_path.find("academic_source")
        if idx != -1:
            sub_path = raw_path[idx:]  # keep everything from "academic_source" onward
            return os.path.join(new_prefix.rstrip("/"), sub_path).replace("\\", "/")
        return raw_path

    return re.sub(pattern, replacer, text)

def recursive_replace(obj, new_prefix: str):
    """Recursively process dicts, lists, and strings to replace paths."""
    if isinstance(obj, dict):
        return {k: recursive_replace(v, new_prefix) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [recursive_replace(v, new_prefix) for v in obj]
    elif isinstance(obj, str):
        return replace_in_text(obj, new_prefix)
    else:
        return obj

def process_file(input_file: str, output_file: str, new_prefix: str):
    """Process a JSON/JSONL file, replacing paths, and save the result."""
    results = []
    replaced_count = 0
    preview = []

    with open(input_file, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                before = json.dumps(obj, ensure_ascii=False)
                changed = recursive_replace(obj, new_prefix)
                after = json.dumps(changed, ensure_ascii=False)

                if before != after:
                    replaced_count += 1
                    if len(preview) < 3:  # collect up to 3 preview examples
                        preview.append((before, after))

                results.append(changed)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line[:80]}...")

    with open(output_file, "w", encoding="utf-8") as fout:
        for obj in results:
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"✅ Done. Updated {replaced_count} lines.")
    if preview:
        print("\n🔎 Preview of changes:")
        for i, (b, a) in enumerate(preview, start=1):
            print(f"\nExample {i}:")
            print(f"Before: {b[:180]}...")
            print(f"After : {a[:180]}...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replace the prefix before 'academic_source' with a new prefix in JSON/JSONL.")
    parser.add_argument("--input", type=str, default="input.jsonl", help="Input JSON/JSONL file")
    parser.add_argument("--output", type=str, default="output.jsonl", help="Output JSON/JSONL file")
    parser.add_argument("--new-prefix", type=str, default="/space", help="New prefix to prepend before 'academic_source'")
    args = parser.parse_args()

    process_file(args.input, args.output, args.new_prefix)

