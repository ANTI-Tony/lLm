"""Loose answer-extraction heuristics for MCQ benchmarks.

MMMU / ScienceQA / MathVista each report accuracy with slightly different
parsers. We keep a tolerant extractor for Phase 0: the goal is not a SOTA
number, it is a CURVE (acc vs num_steps). As long as the parser is applied
consistently across all num_steps values, the *shape* of the curve is
meaningful even if the absolute number is a few points lower than an
optimized LLaVA pipeline.
"""

import re
import string


_LETTER_MAP = {str(i): chr(ord("A") + i - 1) for i in range(1, 10)}


def extract_mcq_letter(pred: str, choices) -> str | None:
    """Try to extract a single letter answer (A/B/C/...) from `pred`.

    choices can be a list of option strings or None.
    """
    if pred is None:
        return None
    text = pred.strip()

    m = re.search(r"(?i)(?:answer|the answer is|option)[^A-Za-z]{0,5}([A-Z])\b",
                  text)
    if m:
        return m.group(1).upper()

    m = re.match(r"\s*([A-Z])\b", text)
    if m:
        return m.group(1).upper()

    m = re.match(r"\s*\(?([A-Z])\)?[\.\):]", text)
    if m:
        return m.group(1).upper()

    m = re.match(r"\s*([1-9])\b", text)
    if m:
        return _LETTER_MAP[m.group(1)]

    if choices:
        lower = text.lower()
        for i, c in enumerate(choices):
            if c and c.strip() and c.strip().lower() in lower:
                return chr(ord("A") + i)
    return None


def numeric_extract(pred: str) -> str | None:
    if pred is None:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", pred.replace(",", ""))
    return m.group(0) if m else None


def norm_text(s: str) -> str:
    s = s.lower().strip()
    s = s.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", s)
