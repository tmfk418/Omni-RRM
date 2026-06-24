#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_selected_best_clean.py
────────────────────────────────────────────────────────────
Evaluate pairwise results from a JSONL file.
Expected fields: id, question, qwen_all_runs, selected_best, gt_answer
"""

import jsonlines
import argparse
import re

def clean_answer(ans: str) -> str:
    """Remove <answer> tags if present"""
    if ans is None:
        return ""
    return re.sub(r"</?answer>", "", ans).strip()

def evaluate(jsonl_file: str, show_errors: int = 10):
    total, correct = 0, 0
    errors = []

    with jsonlines.open(jsonl_file, "r") as reader:
        for entry in reader:
            total += 1
            pred = clean_answer(entry.get("selected_best"))
            gold = clean_answer(entry.get("gt_answer"))

            if pred == gold:
                correct += 1
            else:
                if len(errors) < show_errors:
                    errors.append({
                        "id": entry.get("id"),
                        "question": entry.get("question"),
                        "pred": pred,
                        "gold": gold,
                        "runs": entry.get("qwen_all_runs", [])
                    })

    acc = correct / total if total > 0 else 0.0

    print("───────────────────────────────────────")
    print(f"✅ Total samples : {total}")
    print(f"🎯 Correct        : {correct}")
    print(f"📊 Accuracy       : {acc:.4f}")
    print("───────────────────────────────────────")

    if errors:
        print(f"❌ Showing {len(errors)} error cases:")
        for e in errors:
            print(f"- ID: {e['id']}")
            print(f"  Question: {e['question']}")
            print(f"  Predicted: {e['pred']} | Gold: {e['gold']}")
            print(f"  Runs: {e['runs']}")
            print("")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate pairwise selected_best results")
    parser.add_argument("--jsonl_file", default="/path/to/results.jsonl",
                        help="Path to the results JSONL file")
    parser.add_argument("--show_errors", type=int, default=10,
                        help="Number of error cases to show")
    args = parser.parse_args()

    evaluate(args.jsonl_file, args.show_errors)
