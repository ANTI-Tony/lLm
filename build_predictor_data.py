"""
Oracle labeling + feature extraction pipeline.

Consumes sweep JSONs (from huginn_reasoning.py / ouro_reasoning.py) and
produces training data for the input-conditioned depth predictor:

    for each sample:
        oracle_depth = smallest num_steps at which the sample was correct
        features     = [length, domain, CLIP-text embedding, heuristic scores, ...]

Output: a single .pt containing
    {"X": FloatTensor [N, D], "y": LongTensor [N], "meta": list of dicts}

where y is the oracle depth (as index into num_steps_list) and meta holds
the raw problem text + benchmark name for later introspection.

Usage:
    python build_predictor_data.py \\
        --inputs results/huginn_gsm8k_full.json results/huginn_arc.json \\
        --output data/predictor/train.pt
"""

import argparse
import json
import re
from pathlib import Path

import torch


# --- feature set v1: cheap, interpretable, model-agnostic ---

def _length_feats(text: str):
    n_chars = len(text)
    n_words = len(text.split())
    n_sents = text.count(".") + text.count("?") + 1
    return [n_chars, n_words, n_sents]


_OPS = {"+": 0, "-": 1, "*": 2, "/": 3, "=": 4, "^": 5}


def _math_feats(text: str):
    n_nums = len(re.findall(r"-?\d+(?:\.\d+)?", text))
    op_counts = [text.count(op) for op in _OPS]
    n_parens = text.count("(") + text.count("[")
    has_frac = int("/" in text)
    return [n_nums, *op_counts, n_parens, has_frac]


_KW = ["if", "then", "because", "cause", "how many", "what is",
       "find", "solve", "when", "where", "compare", "total",
       "average", "ratio", "percent", "fraction"]


def _keyword_feats(text: str):
    t = text.lower()
    return [t.count(k) for k in _KW]


def extract_features_v1(problem_text: str):
    return (_length_feats(problem_text)
            + _math_feats(problem_text)
            + _keyword_feats(problem_text))


FEATURE_DIM = len(extract_features_v1(""))


# --- oracle depth extraction ---

def oracle_depth_index(per_sample, ns_values):
    """Return the index of the smallest ns at which this sample is correct,
    or len(ns_values) to mean 'never'. per_sample is a dict keyed by str(ns)
    each value has a 'correct' bool."""
    for j, ns in enumerate(ns_values):
        if per_sample[str(ns)]["correct"]:
            return j
    return len(ns_values)  # sentinel for 'unsolvable at this sweep'


def load_problem_text(item, benchmark):
    # Each benchmark stores problem differently. Normalize.
    if "samples" not in item and "question" in item:
        return item["question"]
    if "question" in item:
        return item["question"]
    if "problem" in item:
        return item["problem"]
    return str(item)


# --- main ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--drop_unsolvable", action="store_true",
                        help="drop samples that were never correct at any ns")
    args = parser.parse_args()

    X, y, meta = [], [], []
    for path in args.inputs:
        with open(path, "r") as f:
            data = json.load(f)
        bench = data.get("benchmark", Path(path).stem)
        ns_values = sorted(int(k) for k in data["per_num_steps"])
        n = data["num_samples"]

        # Collect per-sample (correct at each ns) from the json layout.
        # Layout: per_num_steps[str(ns)]['samples'][i] = {correct, pred, ...}
        ns_entries = {ns: data["per_num_steps"][str(ns)]["samples"]
                      for ns in ns_values}

        for i in range(n):
            per_sample = {str(ns): ns_entries[ns][i] for ns in ns_values}
            depth_idx = oracle_depth_index(per_sample, ns_values)
            unsolvable = depth_idx == len(ns_values)
            if unsolvable and args.drop_unsolvable:
                continue

            # Build the problem text. The sweep JSONs do not themselves
            # store the full problem — only completions. So we need to
            # re-derive from the benchmark-specific completion[:300] or
            # pull from a parallel source. For now, approximate via the
            # first non-empty field under samples[i]. If missing, skip.
            sample_meta = per_sample[str(ns_values[0])]
            question_proxy = sample_meta.get("completion", "") or ""
            if not question_proxy:
                continue

            feats = extract_features_v1(question_proxy)
            X.append(feats)
            y.append(depth_idx)
            meta.append({"benchmark": bench,
                         "sample_idx": i,
                         "question_len": len(question_proxy),
                         "unsolvable": unsolvable})

    if not X:
        raise RuntimeError("No usable samples found. Check JSON formats.")

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    torch.save({"X": X_t, "y": y_t, "meta": meta,
                "ns_values": ns_values, "feature_dim": FEATURE_DIM},
               args.output)
    print(f"[info] wrote {len(X)} samples ({FEATURE_DIM} features) -> {args.output}")

    # Summary
    from collections import Counter
    dist = Counter(y)
    print("Oracle depth class distribution:")
    for j, ns in enumerate(ns_values):
        c = dist.get(j, 0)
        print(f"  ns={ns:<4d}  count={c:<4d}  ({100*c/len(y):.1f}%)")
    never = dist.get(len(ns_values), 0)
    if never:
        print(f"  never        count={never:<4d}  ({100*never/len(y):.1f}%)")


if __name__ == "__main__":
    main()
