import asyncio
import re
import textwrap
from copy import deepcopy
from typing import Dict, List, Optional
from dataclasses import dataclass
from typing import Any, Callable
import json
import torch

from swift.llm import PtEngine, RequestConfig, Template, to_device
from swift.llm.infer.protocol import ChatCompletionResponse
from swift.plugin import ORM, orms, rm_plugins
from swift.plugin.rm_plugin import DefaultRMPlugin
from swift.utils import get_logger

logger = get_logger()
"""
Step 1: Define a Reward Class
    Implement your custom reward calculation logic within the __call__ method.
    The method accepts the model's output completions and dataset columns (passed as kwargs) as input parameters.

Step 2: Register the Reward Class in orms
    For example:
    python orms['external_math_acc'] = MathAccuracy

Step 3: Configure the Arguments
    Use the following arguments when running the script:
    bash --plugin /path/to/plugin.py --reward_funcs external_math_acc
"""


# Code borrowed from plugin/orm.py
class MathAccuracy(ORM):

    def __init__(self):
        import importlib.util
        assert importlib.util.find_spec('math_verify') is not None, (
            "The math_verify package is required but not installed. Please install it using 'pip install math_verify'.")

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        from latex2sympy2_extended import NormalizationConfig
        from math_verify import LatexExtractionConfig, parse, verify
        rewards = []
        for content, sol in zip(completions, solution):
            gold_parsed = parse(sol, extraction_mode='first_match', extraction_config=[LatexExtractionConfig()])
            if len(gold_parsed) != 0:
                # We require the answer to be provided in correct latex (no malformed operators)
                answer_parsed = parse(
                    content,
                    extraction_config=[
                        LatexExtractionConfig(
                            normalization_config=NormalizationConfig(
                                nits=False,
                                malformed_operators=False,
                                basic_latex=True,
                                equations=True,
                                boxed=True,
                                units=True,
                            ),
                            # Ensures that boxed is tried first
                            boxed_match_priority=0,
                            try_extract_without_anchor=False,
                        )
                    ],
                    extraction_mode='first_match',
                )
                # Reward 1 if the content is the same as the ground truth, 0 otherwise
                reward = float(verify(answer_parsed, gold_parsed))
            else:
                # If the gold solution is not parseable, we reward 1 to skip this example
                reward = 1.0
            rewards.append(reward)
        return rewards


class MathFormat(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r'^<think>.*?</think>\s*<answer>.*?</answer>(?![\s\S])'
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        return [1.0 if match else 0.0 for match in matches]


class CountdownORM(ORM):

    def __call__(self, completions, target, nums, **kwargs) -> List[float]:
        """
        Evaluates completions based on Mathematical correctness of the answer

        Args:
            completions (list[str]): Generated outputs
            target (list[str]): Expected answers
            nums (list[str]): Available numbers

        Returns:
            list[float]: Reward scores
        """
        rewards = []
        for completion, gt, numbers in zip(completions, target, nums):
            try:
                # Check if the format is correct
                match = re.search(r'<answer>(.*?)<\/answer>', completion)
                if match is None:
                    rewards.append(0.0)
                    continue
                # Extract the "answer" part from the completion
                equation = match.group(1).strip()
                if '=' in equation:
                    equation = equation.split('=')[0]
                # Extract all numbers from the equation
                used_numbers = [int(n) for n in re.findall(r'\d+', equation)]

                # Check if all numbers are used exactly once
                if sorted(used_numbers) != sorted(numbers):
                    rewards.append(0.0)
                    continue
                # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
                allowed_pattern = r'^[\d+\-*/().\s]+$'
                if not re.match(allowed_pattern, equation):
                    rewards.append(0.0)
                    continue

                # Evaluate the equation with restricted globals and locals
                result = eval(equation, {"__builti'ns__": None}, {})
                # Check if the equation is correct and matches the ground truth
                if abs(float(result) - float(gt)) < 1e-5:
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
            except Exception:
                # If evaluation fails, reward is 0
                rewards.append(0.0)
        return rewards


class MultiModalAccuracyORM(ORM):

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        """
        Reward function that checks if the completion is correct.
        Args:
            completions (list[str]): Generated outputs
            solution (list[str]): Ground Truths.

        Returns:
            list[float]: Reward scores
        """
        rewards = []
        from math_verify import parse, verify
        for content, sol in zip(completions, solution):
            reward = 0.0
            # Try symbolic verification first
            try:
                answer = parse(content)
                if float(verify(answer, parse(sol))) > 0:
                    reward = 1.0
            except Exception:
                pass  # Continue to next verification method if this fails

            # If symbolic verification failed, try string matching
            if reward == 0.0:
                try:
                    # Extract answer from solution if it has think/answer tags
                    sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                    ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

                    # Extract answer from content if it has think/answer tags
                    content_match = re.search(r'<answer>(.*?)</answer>', content)
                    student_answer = content_match.group(1).strip() if content_match else content.strip()

                    # Compare the extracted answers
                    if student_answer == ground_truth:
                        reward = 1.0
                except Exception:
                    pass  # Keep reward as 0.0 if both methods fail
            rewards.append(reward)
        return rewards

# ------------------ New MMFormatORM for multimodal tasks ------------------
@dataclass
class Rule:
    name: str
    weight: float
    check: Callable[[Any, dict | None], bool]

class MMFormatORM(ORM):
    """
    Format reward for multimodal preference-alignment tasks.

    Expected model output is a JSON object:
      - "score_A","score_B": int 0-10
      - "better": "A" | "B" | "equal"
      - "reasoning": string, wrapped in <think>...</think>
      - "final_verdict": in the form <answer>[[A|B|equal]]</answer>
    """

    # Required keys
    REQUIRED_KEYS = {"score_A", "score_B", "better", "reasoning", "final_verdict"}

    def __init__(self, weights: Dict[str, float] | None = None):
        self.weights = weights or {
            "json_valid":    0.20,
            "required_keys": 0.20,
            "field_values":  0.15,
            "reasoning_tag": 0.15,
            "verdict_tag":   0.15,
            "consistency":   0.15,
        }
        total = sum(self.weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1 (got {total})")

        # More tolerant but precise regex (supports multiline)
        self.RE_THINK   = re.compile(r"<think>[\s\S]*?</think>", re.DOTALL)
        self.RE_VERDICT = re.compile(r"<answer>\[\[(A|B|equal)\]\]</answer>")

        self.rules: List[Rule] = [
            Rule("json_valid",    self.weights["json_valid"],    self._json_valid),
            Rule("required_keys", self.weights["required_keys"], self._required_keys),
            Rule("field_values",  self.weights["field_values"],  self._field_values),
            Rule("reasoning_tag", self.weights["reasoning_tag"], self._reasoning_tag),
            Rule("verdict_tag",   self.weights["verdict_tag"],   self._verdict_tag),
            Rule("consistency",   self.weights["consistency"],   self._consistency),
        ]

    # ---------- Unified parsing: remove code block wrappers, tolerate single-element arrays, return dict ----------
    @staticmethod
    def _parse_obj(raw: str) -> dict | None:
        # Remove ``` or ```json wrappers
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", str(raw).strip())
        try:
            obj = json.loads(raw)
        except Exception:
            return None
        # If it's a single-element array with an object, take the first
        if isinstance(obj, list) and obj and isinstance(obj[0], dict):
            obj = obj[0]
        return obj if isinstance(obj, dict) else None

    # ---------- Rule implementations (prefer obj, parse raw only if necessary) ----------
    def _json_valid(self, raw: str, obj: dict | None = None) -> bool:
        return obj is not None

    def _required_keys(self, raw: str, obj: dict | None = None) -> bool:
        obj = obj if isinstance(obj, dict) else self._parse_obj(raw)
        return isinstance(obj, dict) and set(obj.keys()) == self.REQUIRED_KEYS

    def _field_values(self, raw: str, obj: dict | None = None) -> bool:
        obj = obj if isinstance(obj, dict) else self._parse_obj(raw)
        if not isinstance(obj, dict):
            return False
        try:
            a, b = obj["score_A"], obj["score_B"]
            if not (isinstance(a, int) and 0 <= a <= 10): return False
            if not (isinstance(b, int) and 0 <= b <= 10): return False
            if obj.get("better") not in {"A", "B", "equal"}: return False
            return True
        except Exception:
            return False

    def _reasoning_tag(self, raw: str, obj: dict | None = None) -> bool:
        obj = obj if isinstance(obj, dict) else self._parse_obj(raw)
        if not isinstance(obj, dict):
            return False
        reasoning = obj.get("reasoning", "")
        return isinstance(reasoning, str) and self.RE_THINK.match(reasoning.strip()) is not None

    def _verdict_tag(self, raw: str, obj: dict | None = None) -> bool:
        obj = obj if isinstance(obj, dict) else self._parse_obj(raw)
        if not isinstance(obj, dict):
            return False
        verdict = obj.get("final_verdict", "")
        return isinstance(verdict, str) and self.RE_VERDICT.match(verdict.strip()) is not None

    def _consistency(self, raw: str, obj: dict | None = None) -> bool:
        obj = obj if isinstance(obj, dict) else self._parse_obj(raw)
        if not isinstance(obj, dict):
            return False
        m = self.RE_VERDICT.match(str(obj.get("final_verdict", "")).strip())
        return m is not None and (m.group(1) == obj.get("better"))

    # ---------- Main entry: parse once per response, pass to all rules ----------
    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        rewards: List[float] = []
        for resp in completions:
            obj = self._parse_obj(resp)
            score = 0.0
            for rule in self.rules:
                try:
                    if rule.check(resp, obj):
                        score += rule.weight
                except Exception:
                    # A single rule failure is not fatal, skip to avoid breaking training
                    pass
            rewards.append(score)
        return rewards


        
import logging
logger = logging.getLogger(__name__)

class MMContentORM(ORM):  # inherits ORM in your project if needed
    # ---------- Initialization ----------
    def __init__(self, weight_dir: float = 0.6, weight_score: float = 0.4):
        if abs(weight_dir + weight_score - 1.0) > 1e-6:
            raise ValueError("weight_dir + weight_score must equal 1")
        self.w_dir = weight_dir
        self.w_score = weight_score

    # ---------- Utilities ----------
    @staticmethod
    def _safe_int(x: Any) -> int:
        """Convert to int & clip to [0,10]"""
        try:
            v = int(x)
        except Exception:
            raise ValueError("Failed to parse score")
        return max(0, min(10, v))

    @staticmethod
    def _parse_pred(raw: str) -> Dict[str, int | str]:
        """Parse model prediction from JSON string"""
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
        try:
            obj = json.loads(raw)
        except Exception:
            raise ValueError("JSON parsing failed")

        if isinstance(obj, list) and obj and isinstance(obj[0], dict):
            obj = obj[0]
        if not isinstance(obj, dict):
            raise ValueError("Model output is not a dict")

        return {
            "score_A": MMContentORM._safe_int(obj["score_A"]),
            "score_B": MMContentORM._safe_int(obj["score_B"]),
            "better": str(obj["better"]),
        }

    @staticmethod
    def _self_consistent(sA: int, sB: int, better: str) -> bool:
        """Check if better is consistent with the scores"""
        if better == "A":
            return sA > sB
        if better == "B":
            return sB > sA
        if better == "equal":
            return sA == sB
        return False

    # ---------- Main entry ----------
    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        """Evaluate content reward for a batch of completions"""
        # --------- ① Get ground-truth ---------
        try:
            gt_better = kwargs["better"]
            gt_sA = kwargs["score_A"]
            gt_sB = kwargs["score_B"]
        except KeyError as e:
            # ---------- ② Fallback: try to get fields from solution ----------
            if "solution" in kwargs:
                try:
                    sol_raw = kwargs["solution"]
                    # Some loaders may read solution as list[str]
                    if isinstance(sol_raw, list):
                        sol_raw = sol_raw[0]
                    sol = json.loads(sol_raw)
                    for k in ("better", "score_A", "score_B"):
                        kwargs[k] = sol[k]
                    gt_better = kwargs["better"]
                    gt_sA = kwargs["score_A"]
                    gt_sB = kwargs["score_B"]
                    logger.debug("[MMContentORM] Successfully filled fields from solution")
                except Exception as ee:
                    logger.error(f"[MMContentORM] Failed to parse solution: {ee}")
                    return [-1.0] * len(completions)
            else:
                logger.error(f"[MMContentORM] Missing required field: {e}")
                return [-1.0] * len(completions)

        # If single values, broadcast them into lists
        if not isinstance(gt_better, list):
            gt_better = [gt_better] * len(completions)
        if not isinstance(gt_sA, list):
            gt_sA = [gt_sA] * len(completions)
        if not isinstance(gt_sB, list):
            gt_sB = [gt_sB] * len(completions)

        rewards: List[float] = []

        # --------- ③ Compute reward for each completion ---------
        for raw, g_pref, g_a, g_b in zip(completions, gt_better, gt_sA, gt_sB):
            try:
                # ---------- Parse model output ----------
                pred = self._parse_pred(raw)
                p_a, p_b, p_pref = pred["score_A"], pred["score_B"], pred["better"]

                # ---------- (1) Self-consistency check ----------
                if not self._self_consistent(p_a, p_b, p_pref):
                    rewards.append(-1.0)
                    continue

                # ---------- (2) Direction correctness ----------
                C_dir = 1.0 if p_pref == str(g_pref) else -1.0

                # ---------- (3) Score closeness ----------
                g_a, g_b = self._safe_int(g_a), self._safe_int(g_b)
                err = (abs(p_a - g_a) + abs(p_b - g_b)) / 20.0  # ∈ [0,1]
                C_score = 1.0 - 2.0 * err                       # map to [-1,1]

                # ---------- (4) Aggregate ----------
                R = self.w_dir * C_dir + self.w_score * C_score
                rewards.append(float(max(-1.0, min(1.0, R))))
            except Exception as e:
                logger.debug(f"[MMContentORM] Parsing/calculation failed: {e}")
                rewards.append(-1.0)

        return rewards


class MMRubricORM(ORM):
    """
    Rubric-level reward for multimodal preference evaluation.
    ---------------------------------------------------------
    ①  Dimension coverage   C_cover = covered_dimensions / 5
    ②  Dimension comparison C_cmp   = compared_dimensions / 5
        ─ Explicit comparison: A and B appear in the same segment
        ─ Implicit comparison: collective terms like 'both/answers/…' + comparative words like better/worse
    ③  Dynamic gain         Δ_cmp   = max(0 , C_cmp − C_cmp_gt)
                     ( if cmp_gt not provided → Δ_cmp = C_cmp )
    Final reward:  R = w_cover·C_cover + w_cmp·Δ_cmp , range [0,1]
        Default w_cover = 0.8 , w_cmp = 0.2
    """

    # ---------- Initialization ----------
    def __init__(self, w_cover: float = 0.8, w_cmp: float = 0.2):
        if abs(w_cover + w_cmp - 1.0) > 1e-6:
            raise ValueError('w_cover + w_cmp must equal 1')
        self.wc, self.wm = w_cover, w_cmp

        # Dimension keywords
        self.dim_patterns: Dict[str, re.Pattern] = {
            'fluency':   re.compile(r'\b(fluency|coherence|flow|coherent)\b', re.I),
            'relevance': re.compile(r'\b(relevance|related|pertinent|alignment)\b', re.I),
            'accuracy':  re.compile(r'\b(accuracy|accurate|correct(ness)?|precision)\b', re.I),
            'reasoning': re.compile(r'\b(reasoning|analysis|logic(al)?|inference)\b', re.I),
            'safety':    re.compile(r'\b(safety|safe|ethical|harmless|toxic(ity)?)\b', re.I),
        }

        # A / B detection
        self.re_A = re.compile(r'(?<![A-Za-z])(?:candidate|answer|response|model|option)?\s*A(?![A-Za-z])', re.I)
        self.re_B = re.compile(r'(?<![A-Za-z])(?:candidate|answer|response|model|option)?\s*B(?![A-Za-z])', re.I)
        self.re_collective  = re.compile(r'\b(both|two|each|either|neither|all|responses?|candidates?|answers?)\b', re.I)
        self.re_comparative = re.compile(
            r'\b(better|worse|superior|inferior|preferable|outperform\w*|more\s+\w+|less\s+\w+)\b', re.I
        )

    # ---------- Utility ----------
    @staticmethod
    def _strip_think(txt: str) -> str:
        """Remove <think> tags"""
        return re.sub(r'^<think>\s*|\s*</think>$', '', txt.strip(), flags=re.I)

    # ---------- Main entry ----------
    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        cmp_gt_val = None
        if 'cmp_gt' in kwargs:
            try:
                cmp_gt_val = max(0.0, min(1.0, float(kwargs['cmp_gt'])))
            except Exception:
                cmp_gt_val = None

        rewards: List[float] = []

        for raw in completions:
            # ---------- Extract reasoning ----------
            try:
                obj = json.loads(raw)
                reasoning = self._strip_think(obj.get('reasoning', ''))
            except Exception as e:
                logger.debug(f'[Rubric] JSON parse error: {e}')
                rewards.append(-1.0)
                continue

            cover_hit, cmp_hit = 0, 0   # number of covered dimensions / number of compared dimensions

            # ---------- Iterate over 5 dimensions ----------
            for dim, rg in self.dim_patterns.items():
                # Find the first occurrence of this dimension
                m = rg.search(reasoning)
                if not m:
                    continue  # this dimension not covered

                cover_hit += 1
                start = m.start()

                # Calculate the end of this dimension segment = min start of next dimension
                next_positions = []
                for other_rg in self.dim_patterns.values():
                    if other_rg is rg:
                        continue
                    m2 = other_rg.search(reasoning, start + 1)
                    if m2:
                        next_positions.append(m2.start())
                end = min(next_positions) if next_positions else len(reasoning)
                segment = reasoning[start:end]

                # ---------- Comparison detection ----------
                explicit  = self.re_A.search(segment) and self.re_B.search(segment)
                implicit  = self.re_collective.search(segment) and self.re_comparative.search(segment)
                if explicit or implicit:
                    cmp_hit += 1

            # ---------- Compute scores ----------
            C_cover = cover_hit / 5.0
            C_cmp   = cmp_hit   / 5.0

            # Dynamic gain: reward only if "better than GT"
            if cmp_gt_val is not None:
                cmp_gain = max(0.0, C_cmp - cmp_gt_val)
            else:
                cmp_gain = C_cmp

            reward = self.wc * C_cover + self.wm * cmp_gain
            rewards.append(float(reward))

        return rewards

    
class MMFormatLiteORM(ORM):
    """
    Lightweight (no-thinking) format reward:
      Expected model output is a strict three-key JSON:
        - "score_A","score_B": int 0-10
        - "better": "A" | "B" | "equal"

      Scoring rules (point-based + weighted):
        - json_valid       : Can be parsed as json (tolerates ```json wrappers, single-element list)
        - required_keys    : Key set must be exactly {"score_A","score_B","better"} (no more, no less)
        - field_values     : Scores must be integers between 0-10, better must be valid
        - no_banned        : Must not contain old-format residual tokens (e.g., <think>, reasoning, final_verdict, etc.)
        - length_ok        : Output must not be too long (avoid verbose/off-topic)
        - direction_consistent (optional, default 0 weight): "better" must be consistent with score direction
    """

    # Strict three keys
    REQUIRED_KEYS = {"score_A", "score_B", "better"}

    # Banned tokens from old-format / reasoning residuals
    BAN_STRS = (
        "<think>", "</think>", "reasoning", "final_verdict"
        # , "<answer>"  # Uncomment for stricter ban; keep disabled for compatibility
    )

    # Max length threshold (enough for three-key JSON; adjust as needed)
    MAX_LEN = 160

    def __init__(self, weights: Dict[str, float] | None = None):
        # Rule weights (must sum to 1.0, same requirement as original class)
        self.weights = weights or {
            "json_valid":          0.30,
            "required_keys":       0.25,
            "field_values":        0.25,
            "no_banned":           0.10,
            "length_ok":           0.10,
            # If you want to delegate "direction consistency" entirely to content reward, keep this at 0.0
            "direction_consistent": 0.00,
        }
        total = sum(self.weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1 (got {total})")

        self.rules: List[Rule] = [
            Rule("json_valid",          self.weights["json_valid"],          self._json_valid),
            Rule("required_keys",       self.weights["required_keys"],       self._required_keys),
            Rule("field_values",        self.weights["field_values"],        self._field_values),
            Rule("no_banned",           self.weights["no_banned"],           self._no_banned),
            Rule("length_ok",           self.weights["length_ok"],           self._length_ok),
            Rule("direction_consistent",self.weights["direction_consistent"],self._direction_consistent),
        ]

    # ---------- Unified parsing: strip code blocks, tolerate single-element arrays, return dict ----------
    @staticmethod
    def _parse_obj(raw: str) -> dict | None:
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", str(raw).strip())
        try:
            obj = json.loads(raw)
        except Exception:
            return None
        if isinstance(obj, list) and obj and isinstance(obj[0], dict):
            obj = obj[0]
        return obj if isinstance(obj, dict) else None

    @staticmethod
    def _is_int_0_10(x) -> bool:
        try:
            v = int(x)
        except Exception:
            return False
        return 0 <= v <= 10

    # ---------- Rule implementations ----------
    def _json_valid(self, raw: str, obj: dict | None = None) -> bool:
        return obj is not None

    def _required_keys(self, raw: str, obj: dict | None = None) -> bool:
        obj = obj if isinstance(obj, dict) else self._parse_obj(raw)
        return isinstance(obj, dict) and set(obj.keys()) == self.REQUIRED_KEYS

    def _field_values(self, raw: str, obj: dict | None = None) -> bool:
        obj = obj if isinstance(obj, dict) else self._parse_obj(raw)
        if not isinstance(obj, dict):
            return False
        try:
            a, b = obj["score_A"], obj["score_B"]
            if not (isinstance(a, int) and 0 <= a <= 10): return False
            if not (isinstance(b, int) and 0 <= b <= 10): return False
            if obj.get("better") not in {"A", "B", "equal"}: return False
            return True
        except Exception:
            return False

    def _no_banned(self, raw: str, obj: dict | None = None) -> bool:
        low = str(raw).lower()
        return not any(k.lower() in low for k in self.BAN_STRS)

    def _length_ok(self, raw: str, obj: dict | None = None) -> bool:
        return len(str(raw)) <= self.MAX_LEN

    def _direction_consistent(self, raw: str, obj: dict | None = None) -> bool:
        """
        Optional rule: "better" consistent with score direction (default weight=0; keep 0 if delegated to content reward)
          - better="A"  => score_A > score_B
          - better="B"  => score_B > score_A
          - better="equal" => |A-B| <= 1
        """
        obj = obj if isinstance(obj, dict) else self._parse_obj(raw)
        if not isinstance(obj, dict):
            return False
        try:
            a, b = int(obj["score_A"]), int(obj["score_B"])
            bet = str(obj["better"])
        except Exception:
            return False
        if bet == "A":    return a > b
        if bet == "B":    return b > a
        if bet == "equal":return abs(a - b) <= 1
        return False

    # ---------- Main entry: single parse, reused across rules ----------
    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        rewards: List[float] = []
        for resp in completions:
            obj = self._parse_obj(resp)
            score = 0.0
            for rule in self.rules:
                try:
                    if rule.check(resp, obj):
                        score += rule.weight
                except Exception:
                    pass
            rewards.append(score)
        return rewards


class CodeReward(ORM):

    def __init__(self):
        import importlib.util
        assert importlib.util.find_spec('e2b') is not None, (
            "The e2b package is required but not installed. Please install it using 'pip install e2b-code-interpreter'."
        )
        from dotenv import load_dotenv
        load_dotenv()

    @staticmethod
    def extract_code(completion: str, language: str) -> str:
        pattern = re.compile(rf'```{language}\n(.*?)```', re.DOTALL)
        matches = pattern.findall(completion)
        extracted_answer = matches[-1] if len(matches) >= 1 else ''
        return extracted_answer

    def run_async_from_sync(self, scripts: List[str], languages: List[str]) -> List[float]:
        """Function wrapping the `run_async` function."""
        # Create a new event loop and set it
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Run the async function and get the result
            rewards = loop.run_until_complete(self.run_async(scripts, languages))
        finally:
            loop.close()

        return rewards

    async def run_async(self, scripts: List[str], languages: List[str]) -> List[float]:
        from e2b_code_interpreter import AsyncSandbox

        # Create the sandbox by hand, currently there's no context manager for this version
        try:
            sbx = await AsyncSandbox.create(timeout=30, request_timeout=3)
        except Exception as e:
            logger.warning(f'Error from E2B executor: {e}')
            return [0.0] * len(scripts)
        # Create a list of tasks for running scripts concurrently
        tasks = [self.run_script(sbx, script, language) for script, language in zip(scripts, languages)]

        # Wait for all tasks to complete and gather their results as they finish
        results = await asyncio.gather(*tasks)
        rewards = list(results)  # collect results

        # Kill the sandbox after all the tasks are complete
        await sbx.kill()

        return rewards

    async def run_script(self, sbx, script: str, language: str) -> float:
        try:
            execution = await sbx.run_code(script, language=language, timeout=30)
        except Exception as e:
            logger.warning(f'Error from E2B executor: {e}')
            return 0.0
        try:
            return float(execution.text)
        except (TypeError, ValueError):
            return 0.0

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that evaluates code snippets using the E2B code interpreter.

        Assumes the dataset contains a `verification_info` column with test cases.
        """
        evaluation_script_template = """
        import subprocess
        import json

        def evaluate_code(code, test_cases):
            passed = 0
            total = len(test_cases)
            exec_timeout = 5

            for case in test_cases:
                process = subprocess.run(
                    ["python3", "-c", code],
                    input=case["input"],
                    text=True,
                    capture_output=True,
                    timeout=exec_timeout
                )

                if process.returncode != 0:  # Error in execution
                    continue

                output = process.stdout.strip()
                if output.strip() == case["output"].strip():
                    passed += 1

            success_rate = (passed / total)
            return success_rate

        code_snippet = {code}
        test_cases = json.loads({test_cases})

        evaluate_code(code_snippet, test_cases)
        """
        verification_info = kwargs['verification_info']
        languages = [info['language'] for info in verification_info]
        code_snippets = [
            self.extract_code(completion, language) for completion, language in zip(completions, languages)
        ]
        scripts = [
            evaluation_script_template.format(
                code=json.dumps(code), test_cases=json.dumps(json.dumps(info['test_cases'])))
            for code, info in zip(code_snippets, verification_info)
        ]
        try:
            rewards = self.run_async_from_sync(scripts, languages)

        except Exception as e:
            logger.warning(f'Error from E2B executor: {e}')
            rewards = [0.0] * len(completions)

        return rewards


class CodeFormat(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        verification_info = kwargs['verification_info']
        rewards = []
        for content, info in zip(completions, verification_info):
            pattern = r'^<think>.*?</think>\s*<answer>.*?```{}.*?```.*?</answer>(?![\s\S])'.format(info['language'])
            match = re.match(pattern, content, re.DOTALL | re.MULTILINE)
            reward = 1.0 if match else 0.0
            rewards.append(reward)
        return rewards


class CodeRewardByJudge0(ORM):
    LANGUAGE_ID_MAP = {
        'assembly': 45,
        'bash': 46,
        'basic': 47,
        'c': 50,
        'c++': 54,
        'clojure': 86,
        'c#': 51,
        'cobol': 77,
        'common lisp': 55,
        'd': 56,
        'elixir': 57,
        'erlang': 58,
        'executable': 44,
        'f#': 87,
        'fortran': 59,
        'go': 60,
        'groovy': 88,
        'haskell': 61,
        'java': 62,
        'javascript': 63,
        'kotlin': 78,
        'lua': 64,
        'multi-file program': 89,
        'objective-c': 79,
        'ocaml': 65,
        'octave': 66,
        'pascal': 67,
        'perl': 85,
        'php': 68,
        'plain text': 43,
        'prolog': 69,
        'python': 71,
        'python2': 70,
        'python3': 71,
        'r': 80,
        'ruby': 72,
        'rust': 73,
        'scala': 81,
        'sql': 82,
        'swift': 83,
        'typescript': 74,
        'visual basic.net': 84
    }
    PYTHON_ID = 71

    def __init__(self):
        import os
        self.endpoint = os.getenv('JUDGE0_ENDPOINT')
        assert self.endpoint is not None, (
            'Judge0 endpoint is not set. Please set the JUDGE0_ENDPOINT environment variable.')
        x_auth_token = os.getenv('JUDGE0_X_AUTH_TOKEN')
        self.headers = {'Content-Type': 'application/json'}
        if x_auth_token is not None:
            self.headers['X-Auth-Token'] = x_auth_token

    @staticmethod
    def extract_code(completion: str, language: str) -> str:
        pattern = re.compile(rf'```{language}\n(.*?)```', re.DOTALL)
        matches = pattern.findall(completion)
        extracted_answer = matches[-1] if len(matches) >= 1 else ''
        return extracted_answer

    @classmethod
    def get_language_id(cls, language):
        if language is None:
            return cls.PYTHON_ID
        return cls.LANGUAGE_ID_MAP.get(language.lower().strip(), cls.PYTHON_ID)

    async def _evaluate_code(self, code, test_cases, language_id):
        import aiohttp
        try:
            passed = 0
            total = len(test_cases)

            for case in test_cases:
                if code is not None and code != '':
                    async with aiohttp.ClientSession() as session:
                        payload = {
                            'source_code': code,
                            'language_id': language_id,
                            'stdin': case['input'],
                            'expected_output': case['output']
                        }
                        logger.debug(f'Payload: {payload}')
                        async with session.post(
                                self.endpoint + '/submissions/?wait=true', json=payload,
                                headers=self.headers) as response:
                            response_json = await response.json()
                            logger.debug(f'Response: {response_json}')
                            if response_json['status']['description'] == 'Accepted':
                                passed += 1

            success_rate = (passed / total)
            return success_rate
        except Exception as e:
            logger.warning(f'Error from Judge0 executor: {e}')
            return 0.0

    def run_async_from_sync(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            rewards = loop.run_until_complete(self.run_async())
        finally:
            loop.close()
        return rewards

    async def run_async(self):
        tasks = [
            self._evaluate_code(code, info['test_cases'], CodeRewardByJudge0.get_language_id(info['language']))
            for code, info in zip(self.code_snippets, self.verification_info)
        ]
        results = await asyncio.gather(*tasks)
        rewards = list(results)
        return rewards

    def __call__(self, completions, **kwargs) -> List[float]:
        self.verification_info = kwargs['verification_info']

        languages = [info['language'] for info in self.verification_info]
        self.code_snippets = [
            self.extract_code(completion, language) for completion, language in zip(completions, languages)
        ]

        try:
            rewards = self.run_async_from_sync()
        except Exception as e:
            logger.warning(f'Error from Judge0 executor: {e}')
            rewards = [0.0] * len(completions)
        return rewards

class UnifiedRLORM(ORM):
    """Unified reward for two tasks:
       - pair: extract 'better' ∈ {A,B,equal} and reward 1/0
       - point: extract score ∈ [0,100] and reward = max(0, 1 - |pred-gt| / point_scale)
    """

    def __init__(self, pair_reward: float = 1.0, point_scale: float = 50.0):
        """
        pair_reward: reward when better matches (default 1.0).
        point_scale: denominator for score distance. smaller => stronger penalty.
                     reward = max(0, 1 - |pred - gt| / point_scale).
        """
        self.pair_reward = float(pair_reward)
        self.point_scale = float(point_scale)

    # ----------------------------- utils -----------------------------
    def _clamp(self, v: int, lo: int, hi: int) -> int:
        return max(lo, min(hi, v))

    def _strip_fences(self, s: str) -> str:
        import re
        if not isinstance(s, str):
            return ""
        # remove ```/```json fences
        return re.sub(r"^```(?:json)?\s*|\s*```$", "", s.strip(), flags=re.IGNORECASE).strip()

    def _try_load_json(self, s: str):
        import json
        try:
            return json.loads(self._strip_fences(s))
        except Exception:
            return None

    def _norm_better(self, tag: str) -> str:
        if not tag:
            return ""
        t = str(tag).strip().lower()
        if t in ("a", "first"):  return "A"
        if t in ("b", "second"): return "B"
        if t == "equal":         return "equal"
        return ""

    # ----------------------- extract for pair ------------------------
    def _extract_better(self, text_or_json) -> str:
        """Try to extract better = A/B/equal (robust to JSON or free text)."""
        import re
        # dict
        if isinstance(text_or_json, dict):
            return self._norm_better(text_or_json.get("better", ""))
        # string
        if isinstance(text_or_json, str):
            j = self._try_load_json(text_or_json)
            if isinstance(j, dict) and "better" in j:
                return self._norm_better(j.get("better", ""))

            txt = self._strip_fences(text_or_json)

            # explicit key: "better": "A"
            m = re.search(r'"better"\s*:\s*"([^"]+)"', txt, re.IGNORECASE)
            if m:
                return self._norm_better(m.group(1))

            # [first]/[second]/[equal]
            m = re.search(r"\[(first|second|equal)\]", txt, re.IGNORECASE)
            if m:
                return self._norm_better(m.group(1))

            # "better: first"
            m = re.search(r"\b(the\s+better\s+response\s*:\s*|\bbetter\s*[:\-]?\s*)(first|second|equal)\b",
                          txt, re.IGNORECASE)
            if m:
                return self._norm_better(m.group(2))
        return ""

    # ---------------------- extract for point ------------------------
    def _extract_score(self, text_or_json) -> int:
        """Try to extract a 0..100 integer score (robust to JSON / Final Score / 9/10)."""
        import re
        # dict → accept score/Score/rating/Rating
        if isinstance(text_or_json, dict):
            for k in ("score", "Score", "rating", "Rating"):
                if k in text_or_json:
                    try:
                        return self._clamp(int(float(text_or_json[k])), 0, 100)
                    except Exception:
                        pass
        # string → try JSON first
        if isinstance(text_or_json, str):
            j = self._try_load_json(text_or_json)
            if isinstance(j, dict):
                return self._extract_score(j)

            txt = self._strip_fences(text_or_json)

            # Final/Overall/Total Score/Rating: 87  or  Score: 9/10
            m = re.search(
                r"\b(final|overall|total)?\s*(score|rating)\s*(?:[:：=\-]\s*|(?:\s+is\s+))(\d{1,3})(?:\s*/\s*(10|100))?\b",
                txt, re.IGNORECASE)
            if m:
                num = int(m.group(3))
                base = m.group(4)
                if base == "10":
                    num = int(round(num * 10))
                return self._clamp(num, 0, 100)

            # English variants only (removed Chinese)
            m = re.search(r"(final(score|grade)|score|grade|points)\s*[:：=\-]\s*(\d{1,3})(?:\s*/\s*(10|100))?",
                          txt, re.IGNORECASE)
            if m:
                num = int(m.group(3))
                return self._clamp(num, 0, 100)

            # loose fallback: last line is a number
            lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
            if lines and re.fullmatch(r"\d{1,3}", lines[-1]):
                return self._clamp(int(lines[-1]), 0, 100)

        return -1  # not found

    # ----------------------------- API ------------------------------
    def __call__(self, completions, **kwargs) -> list[float]:
        """
        Expected kwargs:
            - solution: list[str] or list[dict]  (same length as completions)
            - pair_reward (optional): float, per-call override
            - point_scale (optional): float, per-call override
        Returns:
            list[float]: rewards in [0,1]
        """
        sols = kwargs.get("solution", None)
        if not isinstance(sols, list):
            sols = [sols] * len(completions) if isinstance(sols, (str, dict)) else ["" for _ in completions]

        # allow per-call override
        pair_reward = float(kwargs.get("pair_reward", self.pair_reward))
        point_scale = float(kwargs.get("point_scale", self.point_scale))

        rewards: list[float] = []
        for pred, gt in zip(completions, sols):
            # ---- try pair first ----
            gt_better = self._extract_better(gt)
            if gt_better:
                pred_better = self._extract_better(pred)
                rewards.append(pair_reward if (pred_better == gt_better and pred_better != "") else 0.0)
                continue

            # ---- then point ----
            gt_score = self._extract_score(gt)
            if gt_score >= 0:
                pred_score = self._extract_score(pred)
                if pred_score >= 0:
                    diff = abs(pred_score - gt_score)
                    r = 1.0 - (diff / point_scale)
                    rewards.append(max(0.0, r))
                else:
                    rewards.append(0.0)
                continue

            # ---- unknown task → 0 ----
            rewards.append(0.0)

        return rewards

class IntegratedRouterORM(ORM):
    """
    Route by PROMPT type:
      - 'scored' prompts (ask to rate each candidate / output two scores): use mm_format + mm_content + mm_rubric
      - 'preference-only' prompts (ask only for better A/B/equal): simple correctness on 'better'

    Aggregation for scored prompts: weighted average of [mm_format, mm_content, mm_rubric].
    """

    # ----------------------------- utils -----------------------------
    def _strip_fences(self, s: str) -> str:
        import re
        if not isinstance(s, str): return ""
        return re.sub(r"^```(?:json)?\s*|\s*```$", "", s.strip(), flags=re.IGNORECASE).strip()

    def _try_load_json(self, s: str):
        import json
        try:
            return json.loads(self._strip_fences(s))
        except Exception:
            return None

    def _norm_better(self, tag: str) -> str:
        if not tag: return ""
        t = str(tag).strip().lower()
        if t in ("a", "first"):  return "A"
        if t in ("b", "second"): return "B"
        if t == "equal":         return "equal"
        return ""

    def _extract_better(self, text_or_json) -> str:
        """Try to extract better = A/B/equal (robust to JSON or free text)."""
        import re
        if isinstance(text_or_json, dict):
            return self._norm_better(text_or_json.get("better", ""))
        if isinstance(text_or_json, str):
            j = self._try_load_json(text_or_json)
            if isinstance(j, dict) and "better" in j:
                return self._norm_better(j.get("better", ""))
            txt = self._strip_fences(text_or_json)
            m = re.search(r'"better"\s*:\s*"([^"]+)"', txt, re.IGNORECASE)
            if m: return self._norm_better(m.group(1))
            m = re.search(r"\[(first|second|equal)\]", txt, re.IGNORECASE)
            if m: return self._norm_better(m.group(1))
            m = re.search(r"\b(the\s+better\s+response\s*:\s*|\bbetter\s*[:\-]?\s*)(first|second|equal)\b", txt, re.IGNORECASE)
            if m: return self._norm_better(m.group(2))
        return ""

    # ---------------------- prompt-type detection --------------------
    def _is_scored_prompt(self, prompt: str) -> bool:
        """
        Heuristics for 'scored' tasks :
          - mentions 'two values' line + 'scores'
          - 'each assistant receives a score' / 'rate ... on a scale of' / '1-10' / '0-100'
          - 'Please first output a single line containing only two values'
        """
        import re
        if not isinstance(prompt, str): return False
        txt = prompt.lower()
        patterns = [
            r"please\s+first\s+output\s+a\s+single\s+line\s+containing\s+only\s+two\s+values",
            r"each\s+assistant\s+receives\s+.*\bscore\b",
            r"\brate\b.*\bon\s+a\s+scale\s+of\s+(?:1\s*[-–]\s*10|0\s*[-–]\s*100|0\s*[-–]\s*10|0\s*[-–]\s*5)",
            r"\bscore\s*[:：]",
            r"\bfinal\s*score\b",
            r"\b\d+\s*\/\s*(10|100)\b",
        ]
        return any(re.search(p, txt, re.IGNORECASE) for p in patterns)

    def _get_first_prompt(self, messages, idx: int) -> str:
        """
        Try to fetch the prompt content for the i-th sample.
        Expect messages to be list[list[dict]] or list[dict] with role='user'.
        """
        if isinstance(messages, list) and messages:
            try:
                # layout 1: batched messages[i][0].content
                m = messages[idx]
                if isinstance(m, list) and m and isinstance(m[0], dict):
                    return m[0].get("content", "")
                # layout 2: messages[i] is dict with role
                if isinstance(m, dict) and m.get("role") == "user":
                    return m.get("content", "")
            except Exception:
                pass
        return ""

    # ------------------------------ API ------------------------------
    def __call__(self, completions, **kwargs) -> list[float]:
        """
        Expected kwargs:
          - solution: list[str|dict] (same length as completions)
          - messages: optional, to read prompt content for routing
          - weights: optional dict, e.g., {"format":0.3,"content":0.5,"rubric":0.2}
        Requires ORMs registered in `orms`:
          - "mm_format", "mm_content", "mm_rubric"
        Preference-only branch does simple correctness on 'better'.
        """
        from swift.plugin import orms

        sols = kwargs.get("solution", None)
        if not isinstance(sols, list):
            sols = [sols] * len(completions) if isinstance(sols, (str, dict)) else ["" for _ in completions]

        messages = kwargs.get("messages", None)
        weights = kwargs.get("weights", {"format": 0.2, "content": 0.6, "rubric": 0.2})

        # prepare sub-orms (only instantiate if needed)
        MMFormat = orms.get("mm_format")
        MMContent = orms.get("mm_content")
        MMRubric = orms.get("mm_rubric")

        # minimal fallback: if any missing, degrade gracefully (no crash)
        fmt = MMFormat() if MMFormat else None
        cnt = MMContent() if MMContent else None
        rbc = MMRubric() if MMRubric else None

        rewards: list[float] = []
        for i, (pred, gt) in enumerate(zip(completions, sols)):
            prompt = self._get_first_prompt(messages, i)

            if self._is_scored_prompt(prompt):
                # ---- scored branch: mm_format + mm_content + mm_rubric ----
                sub_scores = []
                if fmt:
                    sub_scores.append(("format", fmt([pred])[0]))
                if cnt:
                    # pass ground-truth via solution for direction check
                    sub_scores.append(("content", cnt([pred], solution=[gt])[0]))
                if rbc:
                    sub_scores.append(("rubric", rbc([pred])[0]))

                if not sub_scores:
                    rewards.append(0.0)
                    continue

                # weighted average over available sub-scores
                num, den = 0.0, 0.0
                for name, val in sub_scores:
                    w = float(weights.get(name, 0.0))
                    num += w * float(val)
                    den += w
                rewards.append(num / den if den > 0 else sum(v for _, v in sub_scores) / len(sub_scores))

            else:
                # ---- preference-only branch: strict better match 1/0 ----
                gt_better = self._extract_better(gt)
                pred_better = self._extract_better(pred)
                if gt_better and pred_better:
                    rewards.append(1.0 if gt_better == pred_better else 0.0)
                else:
                    rewards.append(0.0)

        return rewards



orms['external_math_acc'] = MathAccuracy
orms['external_math_format'] = MathFormat
orms['external_countdown'] = CountdownORM
orms['external_r1v_acc'] = MultiModalAccuracyORM
orms['external_code_reward'] = CodeReward
orms['external_code_format'] = CodeFormat
orms['external_code_reward_by_judge0'] = CodeRewardByJudge0
orms['mm_format'] = MMFormatORM
orms['mm_content'] = MMContentORM
orms['mm_rubric'] = MMRubricORM
orms["mm_format_light"] = MMFormatLiteORM
orms["unified_rl"] = UnifiedRLORM
orms["integrated_router"] = IntegratedRouterORM

# For genrm you can refer to swift/llm/plugin/rm_plugin/GenRMPlugin
class CustomizedRMPlugin:
    """
    Customized Reward Model Plugin, same to DefaultRMPlugin

    It assumes that `self.model` is a classification model with a value head(output dimmension 1).
    The first logits value from the model's output is used as the reward score.
    """

    def __init__(self, model, template):
        self.model = model
        self.template: Template = template

    def __call__(self, inputs):
        batched_inputs = [self.template.encode(deepcopy(infer_request)) for infer_request in inputs]
        reward_inputs = to_device(self.template.data_collator(batched_inputs), self.model.device)
        reward_inputs.pop('labels')

        with torch.inference_mode():
            return self.model(**reward_inputs).logits[:, 0]


class QwenLongPlugin(DefaultRMPlugin):
    # https://arxiv.org/abs/2505.17667
    # NOTE: you should customize the verified reward function, you can refer to
    # https://github.com/Tongyi-Zhiwen/QwenLong-L1/tree/main/verl/verl/utils/reward_score
    # hf_dataset: https://huggingface.co/datasets/Tongyi-Zhiwen/DocQA-RL-1.6K/viewer/default/train
    # ms_dataset: https://modelscope.cn/datasets/iic/DocQA-RL-1.6K
    def __init__(self, model, template, accuracy_orm=None):
        super().__init__(model, template)
        # initilize PTEngine to infer
        self.engine = PtEngine.from_model_template(self.model, self.template, max_batch_size=0)  # 0: no limit
        self.request_config = RequestConfig(temperature=0)  # customise your request config here
        self.system = textwrap.dedent("""
            You are an expert in verifying if two answers are the same.

            Your input consists of a problem and two answers: Answer 1 and Answer 2.
            You need to check if they are equivalent.

            Your task is to determine if the two answers are equivalent, without attempting to solve the original problem.
            Compare the answers to verify they represent identical values or meanings,
            even when expressed in different forms or notations.

            Your output must follow this format:
            1) Provide an explanation for why the answers are equivalent or not.
            2) Then provide your final answer in the form of: [[YES]] or [[NO]]

            Problem: {problem_placeholder}
            Answer 1: {answer1_placeholder}
            Answer 2: {answer2_placeholder}
        """)  # noqa
        self.accuracy_orm = accuracy_orm

    def __call__(self, inputs):
        completions = [example['messages'][-1]['content'] for example in inputs]
        ground_truths = [example['reward_model']['ground_truth'] for example in inputs]
        rm_inputs = self.prepare_rm_inputs(inputs, completions, ground_truths)

        results = self.engine.infer(rm_inputs, self.request_config, use_tqdm=False)
        llm_rewards = self.compute_rewards(results)

        if self.accuracy_orm:
            verified_rewards = self.accuracy_orm(completions, ground_truths)
        else:
            verified_rewards = [0.0] * len(llm_rewards)

        rewards = [max(r1, r2) for r1, r2 in zip(llm_rewards, verified_rewards)]
        return torch.tensor(rewards, dtype=torch.float32)

    def prepare_rm_inputs(self, inputs: List[Dict], completions, ground_truths) -> List[Dict]:
        rm_inputs = []
        for infer_request, completion, ground_truth in zip(inputs, completions, ground_truths):
            # Deep copy to prevent modification of original input
            rm_infer_request = deepcopy(infer_request)
            problem = infer_request['messages'][0]['content']
            start_index = problem.index('</text>')
            end_index = problem.index('Format your response as follows:')
            question = problem[start_index:end_index].replace('</text>', '').strip()
            prompt = self.system.format(
                problem_placeholder=question, answer1_placeholder=completion, answer2_placeholder=ground_truth)

            # Construct new messages tailored for the reward model
            rm_messages = [{'role': 'user', 'content': prompt}]

            # Update the messages in the reward infer request
            rm_infer_request['messages'] = rm_messages
            rm_inputs.append(rm_infer_request)
        return rm_inputs

    @staticmethod
    def extract_reward(model_output: str) -> float:
        match = re.search(r'\[([A-Z]+)\]', model_output)
        if match:
            answer = match.group(1)
            if answer == 'YES':
                return 1.0
            elif answer == 'NO':
                return 0.0
            else:
                logger.warning("Unexpected answer, expected 'YES' or 'NO'.")
                return 0.0
        else:
            logger.warning("Unable to extract reward score from the model's output, setting reward to 0")
            return 0.0  # Or raise ValueError("Format incorrect")

    def compute_rewards(self, results: List[ChatCompletionResponse]) -> List[float]:
        """
        Compute average reward scores from the reward model's outputs.

        Args:
            results (List[ChatCompletionResponse]): A list of results from the reward model.

        Returns:
            List[float]: A list of average reward scores.
        """
        rewards = []
        for idx, output in enumerate(results):
            try:
                cur_rewards = []
                for choice in output.choices:
                    response = choice.message.content
                    reward = self.extract_reward(response)
                    cur_rewards.append(reward)
                cur_rewards = [r for r in cur_rewards if r is not None]
                if cur_rewards:
                    average_reward = sum(cur_rewards) / len(cur_rewards)
                else:
                    average_reward = 0.0
                    logger.warning('No valid rewards extracted. Assigning reward score of 0.0.')

                rewards.append(average_reward)
            except Exception as e:
                logger.error(f'Error computing reward: {e}')
                rewards.append(0.0)  # Assign default reward score on failure
        return rewards


rm_plugins['my_rmplugin'] = CustomizedRMPlugin

rm_plugins['qwenlong'] = QwenLongPlugin
