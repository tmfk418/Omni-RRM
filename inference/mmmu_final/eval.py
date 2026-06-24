#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced evaluation for Qwen2.5-Omni predictions:
1. Accuracy vs first run
2. Accuracy vs majority
3. Count diff_cases where first_run != majority
4. Within diff_cases: check how many match the correct answer
"""

import json
from tqdm import tqdm

# Input file path
INPUT_JSONL = "/path/to/final2.jsonl"
SHOW_DIFF_EXAMPLES = 10  # Number of diff cases to display

def evaluate(jsonl_file):
    total_samples = 0
    match_first = 0
    match_majority = 0
    invalid_lines = 0

    diff_cases = 0
    diff_first_correct = 0
    diff_majority_correct = 0
    diff_examples = []

    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Evaluating"):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                gold = data.get("answer")
                all_runs = data.get("qwen_all_runs", [])
                majority = data.get("qwen_majority")

                if gold and all_runs:
                    total_samples += 1
                    first_run = all_runs[0]

                    # ① Compare gold answer vs first_run
                    if first_run == gold:
                        match_first += 1

                    # ② Compare gold answer vs majority
                    if majority == gold:
                        match_majority += 1

                    # ③ Count cases where first_run != majority
                    if first_run != majority:
                        diff_cases += 1
                        if first_run == gold:
                            diff_first_correct += 1
                        if majority == gold:
                            diff_majority_correct += 1
                        if len(diff_examples) < SHOW_DIFF_EXAMPLES:
                            diff_examples.append({
                                "id": data.get("id"),
                                "answer": gold,
                                "first_run": first_run,
                                "majority": majority,
                                "qwen_all_runs": all_runs
                            })

            except json.JSONDecodeError:
                invalid_lines += 1

    # Compute overall accuracies
    acc_first = match_first / total_samples if total_samples > 0 else 0
    acc_majority = match_majority / total_samples if total_samples > 0 else 0

    results = {
        "total_samples": total_samples,
        "invalid_lines": invalid_lines,
        "match_first": match_first,
        "accuracy_first_run": round(acc_first, 4),
        "match_majority": match_majority,
        "accuracy_majority": round(acc_majority, 4),
        "diff_cases (first_run≠majority)": diff_cases,
        "diff_first_correct": diff_first_correct,
        "diff_majority_correct": diff_majority_correct,
        "diff_examples": diff_examples
    }
    return results


if __name__ == "__main__":
    stats = evaluate(INPUT_JSONL)
    print(json.dumps(stats, indent=2, ensure_ascii=False))
