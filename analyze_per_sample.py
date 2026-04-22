"""
Per-sample depth variance analysis.

Given a sweep result (from huginn_reasoning.py or huginn_gsm8k.py), find
each sample's oracle optimal depth — the smallest num_steps at which it
was answered correctly. Summarize the distribution.

A wide distribution (many samples need different depths) is the empirical
foundation of Option 4: a per-input predictor is only valuable if
per-sample optimal depth actually varies.

Usage:
    python analyze_per_sample.py --input results/huginn_gsm8k_full.json
"""

import argparse
import json
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    with open(args.input, "r") as f:
        data = json.load(f)

    per_ns = data["per_num_steps"]
    ns_values = sorted(int(k) for k in per_ns.keys())
    n_samples = data["num_samples"]

    # Build a per-sample matrix: correct[i][ns] = bool.
    correct = [[None] * len(ns_values) for _ in range(n_samples)]
    for j, ns in enumerate(ns_values):
        entries = per_ns[str(ns)]["samples"]
        for i, item in enumerate(entries):
            correct[i][j] = item["correct"]

    # Oracle optimal depth: smallest ns that yielded correct; None if never.
    oracle_depth = []
    for i in range(n_samples):
        first_correct_idx = None
        for j in range(len(ns_values)):
            if correct[i][j]:
                first_correct_idx = j
                break
        oracle_depth.append(ns_values[first_correct_idx] if first_correct_idx is not None else None)

    # Distribution summary.
    dist = Counter(oracle_depth)
    print(f"=== Per-sample oracle optimal depth (benchmark: {data.get('benchmark','gsm8k')}) ===")
    print(f"num_samples: {n_samples}")
    print(f"ns_values tested: {ns_values}\n")
    print(f"{'ns':<10}{'count':<10}{'pct':<10}")
    for ns in ns_values:
        c = dist.get(ns, 0)
        print(f"{ns:<10}{c:<10}{100*c/n_samples:.1f}%")
    none_c = dist.get(None, 0)
    print(f"{'never':<10}{none_c:<10}{100*none_c/n_samples:.1f}%")

    # Solvable-but-requires-depth breakdown.
    solved = [d for d in oracle_depth if d is not None]
    n_solved = len(solved)
    print(f"\ntotal solvable (correct at some ns): {n_solved}/{n_samples} "
          f"= {100*n_solved/n_samples:.1f}%")

    # Variance among solved: fraction needing depth > min_ns.
    if n_solved > 0:
        min_ns = min(ns_values)
        needs_more = sum(1 for d in solved if d > min_ns)
        print(f"of solvable: {needs_more}/{n_solved} "
              f"= {100*needs_more/n_solved:.1f}% require ns > {min_ns} to get right")

    # Interesting subsets for paper figures.
    print(f"\n--- Interpretation for Option 4 ---")
    # If a significant fraction is solved ONLY at high depth, per-sample
    # predictor has room to improve.
    tail_depths = [d for d in solved if d >= ns_values[len(ns_values) // 2]]
    frac_tail = len(tail_depths) / n_solved if n_solved else 0
    print(f"fraction of solvable needing depth >= median ({ns_values[len(ns_values)//2]}): "
          f"{100*frac_tail:.1f}%")

    # Hardest samples: solved only at max depth.
    only_deepest = sum(1 for d in solved if d == ns_values[-1])
    print(f"solvable only at max depth ({ns_values[-1]}): "
          f"{only_deepest}/{n_solved} = {100*only_deepest/n_solved:.1f}%" if n_solved else "")

    # Save an augmented JSON with per-sample oracle depth.
    out_path = args.output or args.input.replace(".json", "_oracle.json")
    data["per_sample_oracle_depth"] = oracle_depth
    data["oracle_depth_distribution"] = {
        str(ns): dist.get(ns, 0) for ns in ns_values
    }
    data["oracle_depth_distribution"]["never"] = dist.get(None, 0)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n[info] saved augmented -> {out_path}")


if __name__ == "__main__":
    main()
