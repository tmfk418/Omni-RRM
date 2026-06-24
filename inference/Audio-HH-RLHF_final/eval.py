#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluation Script (eval.py)
---------------------------
Functions:
1. Compute match rate between label and better (0→A, 1→B), treating better=="equal" as incorrect.
2. Compute match rate between label and the larger of score_A/score_B (ties count as incorrect).
3. Exclude samples where better=="error".
4. Output valid sample count, error count, and invalid line count.
5. Optional: Print first few debug samples with detailed matching info.
"""

import json

def evaluate_custom(output_file: str, debug_samples: int = 5):
    total_1 = matched_1 = 0
    total_2 = matched_2 = 0
    error_count = invalid_count = equal_count = 0
    debug_data = []

    with open(output_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            try:
                data = json.loads(line)
                label = data.get("label")
                better = data.get("better")
                score_a = data.get("score_A")
                score_b = data.get("score_B")

                # Skip error samples
                if better == "error":
                    error_count += 1
                    continue

                # Method 1: label vs better
                if better in ["A", "B", "equal"] and label in [0, 1]:
                    total_1 += 1
                    correct = (
                        (label == 0 and better == "A") or
                        (label == 1 and better == "B")
                    )
                    if correct:
                        matched_1 += 1
                    else:
                        if better == "equal":
                            equal_count += 1
                    if len(debug_data) < debug_samples:
                        debug_data.append({
                            "type": "better_match",
                            "label": label,
                            "better": better,
                            "correct": correct
                        })

                # Method 2: label vs max(score_A, score_B)
                if isinstance(score_a, (int, float)) and isinstance(score_b, (int, float)) and label in [0, 1]:
                    total_2 += 1
                    if score_a == score_b:
                        correct = False   # Tie → count as incorrect
                        predicted = None
                    else:
                        predicted = 0 if score_a > score_b else 1
                        correct = (predicted == label)
                    if correct:
                        matched_2 += 1
                    if len(debug_data) < debug_samples:
                        debug_data.append({
                            "type": "score_match",
                            "label": label,
                            "score_A": score_a,
                            "score_B": score_b,
                            "predicted": predicted,
                            "correct": correct
                        })

            except Exception as e:
                invalid_count += 1
                if len(debug_data) < debug_samples:
                    debug_data.append({"type": "error", "error": str(e), "line": line[:100]})

    acc_1 = round(matched_1 / total_1, 4) if total_1 > 0 else 0.0
    acc_2 = round(matched_2 / total_2, 4) if total_2 > 0 else 0.0

    return {
        "valid_samples_label_vs_better": total_1,
        "matched_label_vs_better": matched_1,
        "accuracy_label_vs_better": acc_1,

        "valid_samples_label_vs_scores": total_2,
        "matched_label_vs_scores": matched_2,
        "accuracy_label_vs_scores": acc_2,

        "error_count (better=error)": error_count,
        "equal_count (better=equal_as_wrong)": equal_count,
        "invalid_lines (json_parse_failed)": invalid_count,

        "debug_samples": debug_data
    }

if __name__ == "__main__":
    output_file = "/path/to/full.jsonl"
    result = evaluate_custom(output_file, debug_samples=5)
    print(json.dumps(result, indent=2, ensure_ascii=False))
