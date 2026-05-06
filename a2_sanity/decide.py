"""
A2 sanity check decision script.

Reads the three eval JSONs and prints a clear GO/STOP/UNCLEAR verdict.

Usage (called automatically by run_sanity.sh):
    python decide.py --zeroshot ... --k1_trained ... --k4_trained ...
"""

import argparse
import json


def acc(path, K):
    with open(path) as f:
        return json.load(f)["per_K"][str(K)]["accuracy"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zeroshot", required=True)
    p.add_argument("--k1_trained", required=True)
    p.add_argument("--k4_trained", required=True)
    args = p.parse_args()

    print("\n" + "=" * 60)
    print("A2 sanity check decision")
    print("=" * 60)

    z1 = acc(args.zeroshot, 1)
    z2 = acc(args.zeroshot, 2)
    z4 = acc(args.zeroshot, 4)
    print(f"\nZero-shot (no training):")
    print(f"  K=1: {z1:.3f}    K=2: {z2:.3f}    K=4: {z4:.3f}")
    print(f"  (Expected K>1 worse; this just verifies infra.)")

    k1_at1 = acc(args.k1_trained, 1)
    k1_at4 = acc(args.k1_trained, 4)
    print(f"\nTrained at K=1:")
    print(f"  K_eval=1: {k1_at1:.3f}    K_eval=4: {k1_at4:.3f}")

    k4_at1 = acc(args.k4_trained, 1)
    k4_at4 = acc(args.k4_trained, 4)
    print(f"\nTrained at K=4:")
    print(f"  K_eval=1: {k4_at1:.3f}    K_eval=4: {k4_at4:.3f}")

    print("\n" + "=" * 60)
    delta = k4_at4 - k1_at1
    print(f"Key comparison: K=4-trained @ K=4  vs  K=1-trained @ K=1")
    print(f"  {k4_at4:.3f}  -  {k1_at1:.3f}  =  {delta:+.3f}")
    print("=" * 60)

    se = ((0.25 / 200) ** 0.5)  # 200 samples, max SE assuming acc near 0.5
    print(f"(SE on 200 samples ≈ ±{se:.3f})")
    print()

    if delta >= 0.02:
        print("VERDICT: GO ✅")
        print("  Loop SFT helps. Mechanism is viable.")
        print("  Next: scale up to multi-benchmark + cross-model.")
    elif delta <= -0.03:
        print("VERDICT: STOP ❌")
        print("  Loop SFT actively hurts. Dense layers can't be retrofitted")
        print("  to iteration via SFT alone. Pivot to Framing B (distill)")
        print("  or write a 'why this fails' workshop paper.")
    else:
        print("VERDICT: UNCLEAR ⚠️")
        print("  Within noise. Run more training data (5000 samples) +")
        print("  more epochs (5) to see if signal emerges.")
    print()


if __name__ == "__main__":
    main()
