#!/usr/bin/env python3
# cal.py —— Statistics of "Model Choice vs. Human Ranking" Agreement (Fixed & Hardened)
#
# Key Fix:
#   • Unified coordinate system: both model choice and ranking are compared in the "display order"
#   • parse_ranking and get_flag_number are made more robust
#
# Supported cases:
#   • ranking can be a list or string (e.g. "[1, 0]", "first=0 second=1")
#   • meta.filter_number / meta.random_number may be strings
#   • meta.filter_choice may contain embedded JSON or plain text like "Answer 1/2"
#
# Reported results:
#   valid   —— number of comparable samples
#   agree   —— model choice matches human ranking
#   reject  —— model choice is opposite to human ranking
#   invalid —— parsing failed or insufficient information

import json
import argparse
import re
from collections import Counter
from pathlib import Path
from tqdm import tqdm

# ---------- Common regex ----------
_BETTER_RE = re.compile(r"Answer\s*([12])", re.I)

# ---------- Ranking parser ----------
def parse_ranking(raw):
    """
    Return a list[int] of length 2 like [0,1] / [1,0], otherwise return [].
    More robust parsing: try JSON, then extract numbers; finally enforce int conversion.
    """
    lst = None

    # Already a list
    if isinstance(raw, list):
        lst = raw

    # If string, try json.loads first
    elif isinstance(raw, str):
        try:
            maybe = json.loads(raw)
            if isinstance(maybe, list):
                lst = maybe
        except json.JSONDecodeError:
            pass

        if lst is None:
            # Fallback: extract first two numbers
            nums = re.findall(r"-?\d+", raw)
            if len(nums) >= 2:
                lst = [int(nums[0]), int(nums[1])]

    if lst is None:
        return []

    # Normalize to two ints
    if len(lst) == 2:
        try:
            a = int(lst[0])
            b = int(lst[1])
            return [a, b]
        except Exception:
            return []

    return []

# ---------- Better flag parser ----------
def get_flag_number(meta):
    """
    Return 1 / 2 / -1.
    Parsing order:
      1) meta['filter_number']
      2) if meta['filter_choice'] is JSON, read 'better'
         - support 1/2 and 0/1 (0/1 will be mapped to 1/2)
      3) regex match "Answer 1/2" in meta['filter_choice']
    """
    # 1) filter_number
    n = meta.get("filter_number", -1)
    try:
        n = int(n)
    except Exception:
        n = -1
    if n in (1, 2):
        return n

    # 2) try parsing filter_choice as JSON
    fc = meta.get("filter_choice", "")
    try:
        fc_obj = json.loads(fc)
        better = fc_obj.get("better")
        if isinstance(better, int):
            if better in (1, 2):
                return better
            if better in (0, 1):
                return better + 1
        if isinstance(better, str):
            if better in ("1", "2"):
                return int(better)
            if better in ("0", "1"):
                return int(better) + 1
    except Exception:
        pass

    # 3) regex fallback
    m = _BETTER_RE.search(fc)
    return int(m.group(1)) if m else -1

# ---------- Main process ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        default="results.jsonl",  # Replace with your own results file
        help="Path to inference results *.jsonl"
    )
    args = parser.parse_args()

    cnt = Counter()

    with Path(args.file).open(encoding="utf-8") as f:
        for ln in tqdm(f, desc="Scanning", unit="samples"):
            try:
                obj = json.loads(ln)
            except json.JSONDecodeError:
                cnt["invalid"] += 1
                continue

            # Parse ranking (human ranking in display order)
            ranking = parse_ranking(obj.get("ranking"))
            if len(ranking) != 2:
                cnt["invalid"] += 1
                continue

            # Smaller value means better (supports [0,1], [1,2], etc.)
            good_idx = 0 if ranking[0] <= ranking[1] else 1
            bad_idx  = 1 - good_idx

            meta = obj.get("meta", {}) if isinstance(obj.get("meta", {}), dict) else {}

            # Parse model choice (Answer 1/2 in display order)
            flag_num = get_flag_number(meta)
            if flag_num not in (1, 2):
                cnt["invalid"] += 1
                continue

            # random_number: 0 = not swapped, 1 = swapped
            # Note: we do NOT adjust cand_idx with rnd; both sides use display order
            try:
                _rnd = int(meta.get("random_number", 0))
            except Exception:
                _rnd = 0

            cand_idx = flag_num - 1  # 1→0, 2→1

            # Compare (both in display order coordinate system)
            if cand_idx == good_idx:
                cnt["agree"] += 1
            elif cand_idx == bad_idx:
                cnt["reject"] += 1
            else:
                cnt["invalid"] += 1

    cnt["valid"] = cnt["agree"] + cnt["reject"]

    # ---------- Output ----------
    print("\n=== Results ===")
    for k in ("valid", "agree", "reject", "invalid"):
        print(f"{k:>7}: {cnt[k]}")
    if cnt["valid"]:
        print(f"\nAgreement Rate = {cnt['agree'] / cnt['valid'] * 100:.2f}%")

if __name__ == "__main__":
    main()



