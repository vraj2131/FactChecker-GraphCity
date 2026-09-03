"""
Detectors for two claim shapes the base NLI/LLM pipeline handles poorly.
Both weaknesses were measured, not guessed — see
backend/scripts/batch_harness/analysis.pdf from the 200-claim evaluation.

1. Absolute quantifiers ("always", "never", "only", "exactly", "entire",
   "globally", "every") — a source can support the general gist of a claim
   while the quantifier alone makes the sentence false. Claims containing one
   scored 79% accuracy vs 98% for everything else, and accounted for 5 of 8
   wrong answers in that run.

2. Numeric threshold claims ("more than 90", "at least 27", "exactly 37") —
   a deliberately bounded v1 of numeric reasoning. This does NOT resolve
   superlatives ("largest", "most") since those require an external
   comparison database this pipeline doesn't have; it only flags a
   thresholded claim against explicit numbers appearing in a source snippet,
   and only when the signal is unambiguous.
"""
import re
from typing import List, Optional, Tuple

ABSOLUTE_QUANTIFIERS: Tuple[str, ...] = (
    "always", "never", "only", "exactly", "entire", "entirely",
    "globally", "every", "all ", "no ", "none", "completely", "totally",
)


def detect_absolute_quantifiers(claim: str) -> List[str]:
    """Return the absolute-quantifier words/phrases present in a claim, lowercased."""
    if not claim:
        return []
    lower = f" {claim.lower()} "
    return [q.strip() for q in ABSOLUTE_QUANTIFIERS if q in lower]


# (operator, regex capturing the threshold number)
_THRESHOLD_PATTERNS: List[Tuple[str, "re.Pattern"]] = [
    (">=", re.compile(r"\bat least\s+([\d,]+(?:\.\d+)?)", re.IGNORECASE)),
    ("<=", re.compile(r"\bat most\s+([\d,]+(?:\.\d+)?)", re.IGNORECASE)),
    ("<=", re.compile(r"\bno more than\s+([\d,]+(?:\.\d+)?)", re.IGNORECASE)),
    (">", re.compile(r"\bmore than\s+([\d,]+(?:\.\d+)?)", re.IGNORECASE)),
    (">", re.compile(r"\bover\s+([\d,]+(?:\.\d+)?)", re.IGNORECASE)),
    ("<", re.compile(r"\bless than\s+([\d,]+(?:\.\d+)?)", re.IGNORECASE)),
    ("<", re.compile(r"\bunder\s+([\d,]+(?:\.\d+)?)", re.IGNORECASE)),
    ("==", re.compile(r"\bexactly\s+([\d,]+(?:\.\d+)?)", re.IGNORECASE)),
    ("==", re.compile(r"\bonly\s+([\d,]+(?:\.\d+)?)", re.IGNORECASE)),
]

_OP_LABEL = {">=": "at least", "<=": "at most", ">": "more than", "<": "less than", "==": "exactly"}

# Bare 4-digit numbers in this range are almost always calendar years, not counts.
_YEAR_RANGE = range(1400, 2100)


def extract_threshold(claim: str) -> Optional[Tuple[str, float]]:
    """Return (operator, value) for the first numeric threshold pattern found, or None."""
    if not claim:
        return None
    for op, pattern in _THRESHOLD_PATTERNS:
        m = pattern.search(claim)
        if m:
            try:
                return op, float(m.group(1).replace(",", ""))
            except ValueError:
                continue
    return None


def _extract_snippet_numbers(text: str) -> List[float]:
    numbers: List[float] = []
    for raw in re.findall(r"\b\d[\d,]*(?:\.\d+)?\b", text or ""):
        try:
            val = float(raw.replace(",", ""))
        except ValueError:
            continue
        if val.is_integer() and int(val) in _YEAR_RANGE:
            continue  # skip likely years, not counts
        numbers.append(val)
    return numbers


def numeric_conflict_hint(claim: str, snippet: str) -> Optional[str]:
    """
    Heuristic only — not full arithmetic reasoning. Flags a snippet as a
    likely numeric conflict when the claim asserts a threshold and EVERY
    number found in the snippet falls on the wrong side of it. Returns None
    whenever the signal is ambiguous (no numbers, or mixed numbers), so it
    only fires on genuinely clear contradictions.
    """
    threshold = extract_threshold(claim)
    if threshold is None:
        return None
    op, value = threshold

    snippet_numbers = _extract_snippet_numbers(snippet)
    if not snippet_numbers:
        return None

    def violates(n: float) -> bool:
        if op == ">=":
            return n < value
        if op == "<=":
            return n > value
        if op == ">":
            return n <= value
        if op == "<":
            return n >= value
        if op == "==":
            return n != value
        return False

    if all(violates(n) for n in snippet_numbers):
        label = _OP_LABEL[op]
        cited = snippet_numbers[0] if len(snippet_numbers) == 1 else min(snippet_numbers)
        return (
            f"claim asserts '{label} {value:g}'; source states {cited:g}, "
            f"which does not satisfy that threshold"
        )
    return None


NUMERIC_KEYWORDS: Tuple[str, ...] = (
    "largest", "smallest", "most", "least", "longest", "shortest",
    "highest", "lowest", "more than", "less than", "at least", "at most",
    "exactly", "only", "over", "under", "percent", "known",
)


def is_count_sensitive_claim(claim: str) -> bool:
    """
    True when a claim's truth depends on a count/quantity that can change
    over time (e.g. "Jupiter has more than 90 known moons") — used to boost
    recency weighting for such claims during retrieval ranking, since a
    stale source can refute a claim that is currently true.
    """
    if not claim:
        return False
    if extract_threshold(claim) is not None:
        return True
    lower = claim.lower()
    return bool(re.search(r"\d", claim)) and any(k in lower for k in NUMERIC_KEYWORDS)
