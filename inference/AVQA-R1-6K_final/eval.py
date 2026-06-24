#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_audio_image_runs.py
────────────────────────────────────────────────────────────
Evaluate a JSONL file:
    - Eval 1: gt_answer vs qwen_all_runs[0]
    - Eval 2: gt_answer vs qwen_majority
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
    total, correct_first, correct_majority = 0, 0, 0
    errors_first, errors_majority = [], []

    with jsonlines.open(jsonl_file, "r") as reader:
        for entry in reader:
            total += 1
            gold = clean_answer(entry.get("answer"))
            runs = entry.get("qwen_all_runs", [])
            majority = entry.get("qwen_majority")

            # --- Eval 1: First run ---
            pred_first = clean_answer(runs[0]) if runs else ""
            if pred_first == gold:
                correct_first += 1
            elif len(errors_first) < show_errors:
                errors_first.append({
                    "id": entry.get("id"),
                    "question": entry.get("question"),
                    "pred": pred_first,
                    "gold": gold,
                    "runs": runs
                })

            # --- Eval 2: Majority vote ---
            pred_majority = clean_answer(majority) if majority else ""
            if pred_majority == gold:
                correct_majority += 1
            elif len(errors_majority) < show_errors:
                errors_majority.append({
                    "id": entry.get("id"),
                    "question": entry.get("question"),
                    "pred": pred_majority,
                    "gold": gold,
                    "runs": runs
                })

    acc_first = correct_first / total if total > 0 else 0.0
    acc_majority = correct_majority / total if total > 0 else 0.0

    print("───────────────────────────────────────")
    print(f"✅ Total samples       : {total}")
    print(f"🎯 Accuracy (first run): {acc_first:.4f} ({correct_first}/{total})")
    print(f"🎯 Accuracy (majority) : {acc_majority:.4f} ({correct_majority}/{total})")
    print("───────────────────────────────────────")

    if errors_first:
        print(f"❌ First run errors (showing {len(errors_first)}):")
        for e in errors_first:
            print(f"- ID: {e['id']}")
            print(f"  Q: {e['question']}")
            print(f"  Pred: {e['pred']} | Gold: {e['gold']}")
            print(f"  Runs: {e['runs']}")
            print("")

    if errors_majority:
        print(f"❌ Majority errors (showing {len(errors_majority)}):")
        for e in errors_majority:
            print(f"- ID: {e['id']}")
            print(f"  Q: {e['question']}")
            print(f"  Pred: {e['pred']} | Gold: {e['gold']}")
            print(f"  Runs: {e['runs']}")
            print("")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate qwen_all_runs and qwen_majority results")
    parser.add_argument("--jsonl_file", default="/path/to/data_final.jsonl",
                        help="Path to JSONL results file")
    parser.add_argument("--show_errors", type=int, default=10,
                        help="Number of error cases to show per evaluation")
    args = parser.parse_args()

    evaluate(args.jsonl_file, args.show_errors)
