#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_selected_best.py
────────────────────────────────────────────────────────────
Evaluate results from pairwise_best_from_qwen_votes_video.

Input: JSONL, each entry contains:
    id, question, qwen_votes, selected_best, gt_answer
Output:
    - Total number of samples
    - Number of correct predictions
    - Accuracy
    - List of error cases (first N)
"""

import jsonlines
import argparse

def evaluate(jsonl_file: str, show_errors: int = 10):
    total, correct = 0, 0
    errors = []

    with jsonlines.open(jsonl_file, "r") as reader:
        for entry in reader:
            total += 1
            pred = entry.get("selected_best")
            gold = entry.get("gt_answer")

            if pred == gold:
                correct += 1
            else:
                if len(errors) < show_errors:
                    errors.append({
                        "id": entry.get("id"),
                        "question": entry.get("question"),
                        "pred": pred,
                        "gold": gold,
                        "votes": entry.get("qwen_votes", {})
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
            print(f"  Votes: {e['votes']}")
            print("")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate pairwise selected_best results")
    parser.add_argument(
        "--jsonl_file",
        default="/path/to/best_from_votes.jsonl",
        help="Path to results JSONL file"
    )
    parser.add_argument(
        "--show_errors",
        type=int,
        default=10,
        help="Number of error cases to show"
    )
    args = parser.parse_args()

    evaluate(args.jsonl_file, args.show_errors)
