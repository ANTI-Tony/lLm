"""Plot accuracy vs num_steps curves from eval JSON outputs.

Usage:
    python plot_results.py --inputs results/mmmu.json results/mathvista.json \
                                    results/scienceqa.json \
                           --labels MMMU MathVista ScienceQA \
                           --out results/phase0_curve.png

The CURVE is the whole point of Phase 0:
    * If any benchmark shows a peak at num_steps > 1 with task-dependent
      variation -> H1 holds, continue to Phase 1.
    * If all benchmarks are flat or monotonic -> H1 fails, switch to
      Phase 2 fallback paper.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--out", default="results/phase0_curve.png")
    parser.add_argument("--title", default="Phase 0: Accuracy vs Recurrent Depth")
    args = parser.parse_args()

    assert len(args.inputs) == len(args.labels)

    plt.figure(figsize=(7, 4.5))
    markers = ["o", "s", "D", "^", "v", "P", "*"]
    for i, (path, label) in enumerate(zip(args.inputs, args.labels)):
        with open(path, "r") as f:
            data = json.load(f)
        acc = data["summary"]["num_steps_accuracy"]
        xs = sorted(int(k) for k in acc.keys())
        ys = [acc[str(x)] for x in xs]
        plt.plot(xs, ys, marker=markers[i % len(markers)],
                 linewidth=1.8, label=label)

    plt.xscale("log", base=2)
    plt.xlabel("num_steps (recurrent depth)")
    plt.ylabel("Accuracy")
    plt.title(args.title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=160)
    print(f"[info] saved -> {args.out}")

    # Also emit a short decision report.
    decision_path = Path(args.out).with_suffix(".decision.txt")
    lines = ["Phase 0 decision report", "=" * 40, ""]
    any_peak = False
    for path, label in zip(args.inputs, args.labels):
        with open(path, "r") as f:
            data = json.load(f)
        acc = data["summary"]["num_steps_accuracy"]
        xs = sorted(int(k) for k in acc.keys())
        ys = [acc[str(x)] for x in xs]
        peak_idx = max(range(len(ys)), key=lambda i: ys[i])
        peak_x, peak_y = xs[peak_idx], ys[peak_idx]
        delta = peak_y - ys[0]
        verdict = "PEAK" if peak_x != xs[0] and delta >= 0.02 else "flat/monotonic"
        if verdict == "PEAK":
            any_peak = True
        lines.append(
            f"{label:<12s}  peak @ {peak_x:>3d} = {peak_y:.3f}  "
            f"(step-1 baseline {ys[0]:.3f}, Δ={delta:+.3f})  -> {verdict}")

    lines.append("")
    lines.append("H1 (loop helps VLM reasoning): "
                 + ("LIKELY HOLDS -> proceed to Phase 1."
                    if any_peak else
                    "APPEARS TO FAIL -> switch to Phase 2 fallback paper."))
    with open(decision_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
