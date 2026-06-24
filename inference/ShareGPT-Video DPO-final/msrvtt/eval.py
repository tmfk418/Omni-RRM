#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate the accuracy of Qwen predictions vs. dataset labels
------------------------------------------------------------
• Assumes each JSONL line contains:
    - "label": 0/1
    - "qwen_status": "ok"/"invalid"/"error"
    - "qwen_better": "A"/"B"/"equal"/"error"
• Matching rules:
    label == 0  <=> qwen_better == "A"
    label == 1  <=> qwen_better == "B"
• Outputs:
    - Total number of samples
    - Number of valid samples (status=="ok" and qwen_better ∈ {A,B})
    - Number of matched samples
    - Accuracy
    - Example mismatches
"""

import jsonlines
import argparse
from collections import Counter

def evaluate(jsonl_path, show_mismatch=10):
    total, valid, matched = 0, 0, 0
    mismatches = []

    with jsonlines.open(jsonl_path) as reader:
        for row in reader:
            total += 1
            if row.get("qwen_status") != "ok":
                continue               # Skip invalid / error
            qb = row.get("qwen_better")
            if qb not in ("A", "B"):
                continue               # Skip equal or parsing errors
            expected = "A" if row.get("label") == 0 else "B"
            valid += 1
            if qb == expected:
                matched += 1
            else:
                if len(mismatches) < show_mismatch:
                    mismatches.append({
                        "id": row.get("id"),
                        "label": row.get("label"),
                        "qwen_better": qb
                    })

    accuracy = matched / valid if valid else 0.0
    print("========== Evaluation Report ==========")
    print(f"File               : {jsonl_path}")
    print(f"Total samples      : {total}")
    print(f"Valid samples      : {valid}")
    print(f"Matched samples    : {matched}")
    print(f"Accuracy           : {accuracy:.4f}")
    if mismatches:
        print("\n---- First few mismatches ----")
        for item in mismatches:
            print(item)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Qwen better-vs-label accuracy.")
    parser.add_argument(
        "--file",
        default="/path/to/lora_nested.jsonl",
        help="Path to JSONL result file (e.g., lora_nested.jsonl)"
    )
    parser.add_argument(
        "--show_mismatch",
        type=int,
        default=10,
        help="Show first N mismatches"
    )
    args = parser.parse_args()

    evaluate(args.file, args.show_mismatch)
