"""
Sanity v2 decision: is mixed-K SFT really better than vanilla SFT, or is
the 4pp from v1 just noise?

Reads N seeds × 2 conditions (vanilla, mixed) of K_eval=1 evals and prints:
  * mean ± std per condition
  * delta and Welch's t-statistic (rough — don't over-interpret with N=3)
  * GO / STOP / UNCLEAR verdict

Usage:
    python decide_v2.py --vanilla v_s1.json v_s2.json v_s3.json \\
                        --mixed m_s1.json m_s2.json m_s3.json
"""

import argparse
import json
import math


def acc1(path):
    with open(path) as f:
        return json.load(f)["per_K"]["1"]["accuracy"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vanilla", nargs="+", required=True,
                   help="JSONs from vanilla K=1 SFT eval (different seeds)")
    p.add_argument("--mixed", nargs="+", required=True,
                   help="JSONs from mixed-K SFT eval (different seeds)")
    args = p.parse_args()

    v = [acc1(p) for p in args.vanilla]
    m = [acc1(p) for p in args.mixed]
    n_v, n_m = len(v), len(m)

    mean_v = sum(v) / n_v
    mean_m = sum(m) / n_m
    var_v = sum((x - mean_v) ** 2 for x in v) / max(1, n_v - 1)
    var_m = sum((x - mean_m) ** 2 for x in m) / max(1, n_m - 1)
    std_v = math.sqrt(var_v)
    std_m = math.sqrt(var_m)
    delta = mean_m - mean_v

    # Welch's t (rough — only meaningful with N>=3 each)
    se = math.sqrt(var_v / n_v + var_m / n_m) if (n_v > 1 and n_m > 1) else float("nan")
    t = delta / se if se > 0 else float("nan")

    print("=" * 60)
    print("Sanity v2: mixed-K vs vanilla SFT, K_eval=1, 3 seeds each")
    print("=" * 60)
    print(f"Vanilla K=1 SFT: {v} -> mean={mean_v:.3f} ± {std_v:.3f}")
    print(f"Mixed-K SFT:     {m} -> mean={mean_m:.3f} ± {std_m:.3f}")
    print()
    print(f"delta (mixed - vanilla) = {delta:+.3f}")
    print(f"approx Welch's t = {t:+.2f}  (|t|>1.5 starts to look real)")
    print()

    if delta >= 0.025 and (math.isnan(t) or abs(t) >= 1.5):
        print("VERDICT: GO ✅  Mixed-K SFT is reliably better.")
        print("  -> Path X (random-depth-as-regularization paper).")
    elif delta <= -0.02:
        print("VERDICT: STOP ❌  Mixed-K SFT not better (or worse).")
        print("  -> Path B (negative-result diagnostic paper).")
    else:
        print("VERDICT: UNCLEAR ⚠️  Diff small or t low.")
        print("  -> Need more seeds (>=5) or larger eval set (>=500).")


if __name__ == "__main__":
    main()
