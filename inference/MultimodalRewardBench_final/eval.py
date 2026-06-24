
import json
import argparse
import os
import time
from tqdm import tqdm

import random
from collections import defaultdict


def extract_judgment(judgment):
    """
    Extract the final judgment (A/B/equal) from the model output.
    Supports two formats:
    1. Strict JSON, containing "final_verdict" or "better" field
    2. Legacy plain string, directly searching for [[A]] or [[B]]
    """
    # Attempt to parse JSON
    try:
        data = json.loads(judgment)
        # Prefer final_verdict
        if "final_verdict" in data:
            fv = data["final_verdict"]
            if "[[A]]" in fv:
                return "A"
            elif "[[B]]" in fv:
                return "B"
            elif "equal" in fv.lower():
                return "equal"
        # Fallback to better field
        if "better" in data:
            b = data["better"].upper()
            if b == "A":
                return "A"
            elif b == "B":
                return "B"
            elif b == "EQUAL":
                return "equal"
    except Exception:
        pass

    # If not valid JSON, fallback to legacy logic
    if "[[A]]" in judgment:
        return "A"
    elif "[[B]]" in judgment:
        return "B"
    else:
        return 'A' if random.random() < 0.5 else 'B'


def compute_acc(args):
    accs = defaultdict(list)
    random.seed(123)
    for line in open(args.answers_file, encoding='utf-8'):
        ex = json.loads(line)
        pred = extract_judgment(ex['output'])
        acc = int(pred == ex['Label'])
        accs['all'].append(acc)
        category = ex['Meta']['Category']
        if category == 'safety':
            if ex['ID'].lower().startswith('pairs'):
                category = 'safety/bias'
            else:
                category = 'safety/toxicity'
        elif category == 'reasoning':
            if ex['ID'].lower().startswith('math'):
                category = 'reasoning/math'
            else:
                category = 'reasoning/coding'
        accs[category].append(acc)
    for task in accs:
        print(f"acc {task}: {sum(accs[task])} / {len(accs[task])} = {sum(accs[task])/len(accs[task])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers-file", type=str, default="/path/to/answers.jsonl")
    args = parser.parse_args()

    compute_acc(args)

