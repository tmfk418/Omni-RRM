#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch Evaluation Script:
1. Evaluate ground truth answer vs qwen_single
2. Evaluate ground truth answer vs qwen_majority
"""

import jsonlines
import argparse

def evaluate_predictions(input_file: str):
    total_single, correct_single = 0, 0
    total_majority, correct_majority = 0, 0
    error_count = 0

    with jsonlines.open(input_file, "r") as reader:
        for record in reader:
            gt_answer = record.get("answer")
            qwen_single = record.get("qwen_single")
            qwen_majority = record.get("qwen_majority")

            # Ground truth answer may be in the form "<answer>A</answer>", extract the letter
            if gt_answer:
                if gt_answer.startswith("<answer>") and gt_answer.endswith("</answer>"):
                    gt_answer = gt_answer.replace("<answer>", "").replace("</answer>", "").strip()
                else:
                    gt_answer = gt_answer.strip()

            # Evaluate single prediction
            if qwen_single and gt_answer and qwen_single != "error":
                total_single += 1
                if gt_answer == qwen_single:
                    correct_single += 1
            elif qwen_single == "error":
                error_count += 1

            # Evaluate majority prediction
            if qwen_majority and gt_answer and qwen_majority != "error":
                total_majority += 1
                if gt_answer == qwen_majority:
                    correct_majority += 1
            elif qwen_majority == "error":
                error_count += 1

    acc_single = correct_single / total_single if total_single > 0 else 0
    acc_majority = correct_majority / total_majority if total_majority > 0 else 0

    results = {
        "total_single": total_single,
        "correct_single": correct_single,
        "accuracy_single": round(acc_single, 4),
        "total_majority": total_majority,
        "correct_majority": correct_majority,
        "accuracy_majority": round(acc_majority, 4),
        "error_count": error_count
    }

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="/path/to/qwen_results.jsonl", 
                        help="Path to input JSONL file")
    args = parser.parse_args()

    results = evaluate_predictions(args.input)

    print("✅ Evaluation Results:")
    for k, v in results.items():
        print(f"{k}: {v}")
