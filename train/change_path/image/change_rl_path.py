#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Replace JSON/JSONL file paths so that everything from 'rlaif-v-dataset' onward
is preserved, but the prefix before it is replaced with a user-specified new prefix.
"""

import os
import re
import json
import argparse

def replace_with_custom_prefix(text: str, new_prefix: str) -> str:
    """Replace path prefix before 'rlaif-v-dataset' with new_prefix."""
    pattern = r"(/[^\"']*rlaif-v-dataset[^\s\"']*)"
    
    def replacer(match):
        raw_path = match.group(1)
        idx = raw_path.find("rlaif-v-dataset")
        if idx != -1:
            sub_path = raw_path[idx:]  # preserve everything from 'rlaif-v-dataset'
            return os.path.join(new_prefix.rstrip("/"), sub_path).replace("\\", "/")
        return raw_path

    return re.sub(pattern, replacer, text)

def recursive_replace(obj, new_prefix: str):
    """Recursively replace paths in dicts, lists, and strings."""
    if isinstance(obj, dict):
        return {k: recursive_replace(v, new_prefix) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [recursive_replace(v, new_prefix) for v in obj]
    elif isinstance(obj, str):
        return replace_with_custom_prefix(obj, new_prefix)
    else:
        return obj

def process_file(input_file: str, output_file: str, new_prefix: str):
    """Read JSONL file, replace prefixes, and save the result."""
    with open(input_file, "r", encoding="utf-8") as fin:
        lines = fin.readlines()

    results = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            changed = recursive_replace(obj, new_prefix)
            results.append(changed)
        except json.JSONDecodeError:
            print(f"Skipping invalid JSON line: {line[:80]}...")

    with open(output_file, "w", encoding="utf-8") as fout:
        for obj in results:
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replace path prefixes before 'rlaif-v-dataset' with a custom prefix.")
    parser.add_argument("--input", type=str, default="input.jsonl", help="Path to input JSON/JSONL file")
    parser.add_argument("--output", type=str, default="output.jsonl", help="Path to output JSON/JSONL file")
    parser.add_argument("--prefix", type=str, default="/data", help="New prefix to use before 'rlaif-v-dataset'")
    args = parser.parse_args()

    process_file(args.input, args.output, args.prefix)
    print(f"✅ Updated paths saved to {args.output}")

