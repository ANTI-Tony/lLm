"""
Sanity v3 cross-task verdict:
Does the +3.7pp from sanity v2 (GSM8K) replicate on MATH-500?

Reads N seeds × 2 conditions of MATH-500 evals (K_eval=1) and prints
mean ± std + delta + paired t. Also reads existing GSM8K v2 results
to give a side-by-side picture.
"""

import argparse
import json
import math
from pathlib import Path


def acc1(path):
    with open(path) as f:
        return json.load(f)["per_K"]["1"]["accuracy"]


def stats(label, vs):
    n = len(vs)
    mean = sum(vs) / n
    var = sum((x - mean) ** 2 for x in vs) / max(1, n - 1)
    std = math.sqrt(var)
    return label, mean, std, vs


def report(name, v_paths, m_paths):
    v = [acc1(p) for p in v_paths]
    m = [acc1(p) for p in m_paths]
    nv, nm = len(v), len(m)
    mv = sum(v) / nv; mm_ = sum(m) / nm
    sv = math.sqrt(sum((x - mv) ** 2 for x in v) / max(1, nv - 1))
    sm = math.sqrt(sum((x - mm_) ** 2 for x in m) / max(1, nm - 1))
    delta = mm_ - mv
    se = math.sqrt(sv * sv / nv + sm * sm / nm) if nv > 1 and nm > 1 else float("nan")
    t = delta / se if se > 0 else float("nan")
    # paired (assumes seeds aligned in order)
    if nv == nm:
        diffs = [m[i] - v[i] for i in range(nv)]
        md = sum(diffs) / len(diffs)
        sd = math.sqrt(sum((d - md) ** 2 for d in diffs) / max(1, len(diffs) - 1))
        tp = md / (sd / math.sqrt(len(diffs))) if sd > 0 else float("nan")
    else:
        diffs, md, sd, tp = [], float("nan"), float("nan"), float("nan")

    print(f"\n=== {name} ===")
    print(f"vanilla: {v}  mean={mv:.3f} ± {sv:.3f}")
    print(f"mixed:   {m}  mean={mm_:.3f} ± {sm:.3f}")
    print(f"delta = {delta:+.3f}    Welch t={t:+.2f}    paired t={tp:+.2f}    paired diffs={diffs}")
    return delta, t, tp, mv, mm_


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gsm_vanilla", nargs="+")
    p.add_argument("--gsm_mixed", nargs="+")
    p.add_argument("--math_vanilla", nargs="+", required=True)
    p.add_argument("--math_mixed", nargs="+", required=True)
    args = p.parse_args()

    print("=" * 60)
    print("Sanity v3: cross-task validation")
    print("=" * 60)

    if args.gsm_vanilla and args.gsm_mixed:
        gd, gt, gtp, gmv, gmm = report("GSM8K", args.gsm_vanilla, args.gsm_mixed)
    md, mt, mtp, mmv, mmm = report("MATH-500", args.math_vanilla, args.math_mixed)

    print()
    print("=" * 60)
    print("VERDICT")
    print("=" * 60)
    if md >= 0.02 and (math.isnan(mtp) or abs(mtp) >= 1.5):
        print(f"GO ✅  MATH-500 also shows +{md*100:.1f}pp (paired t={mtp:.2f}).")
        print("  Cross-task evidence holds. Path X (random-depth-as-regularization paper) is real.")
    elif md <= -0.01:
        print(f"STOP ❌  MATH-500 shows {md*100:.1f}pp — GSM8K signal was task-specific noise.")
        print("  Pivot: write Path B negative-result paper or change framing entirely.")
    else:
        print(f"UNCLEAR ⚠️  MATH-500 delta={md*100:+.1f}pp inside noise.")
        print("  Need: more samples per eval (300+), or 5+ seeds, or a third benchmark.")


if __name__ == "__main__":
    main()
